### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ fc3c2db6-7193-11f0-1b65-7f390e18200d
begin
using Pkg
	
link_path = joinpath(pwd(), "SymlinkReachabilityCascade")
if islink(link_path)
    rm(link_path)
elseif ispath(link_path)
    error("A non-symlink file or directory already exists at $link_path. Aborting.")
end
symlink(pwd(), link_path)

Pkg.activate("SymlinkReachabilityCascade")
end

# ╔═╡ c8d1f042-ad49-4971-aa32-b9c15d241da6
using ReachabilityCascade, LinearAlgebra, Random, LazySets, Flux, Statistics, JuMP, SparseArrays, StaticArrays, HiGHS, Clarabel, JLD2

# ╔═╡ c6105657-7697-462a-84f0-5b6cb0ffb4b1
import ReachabilityCascade.CarDynamics as CD

# ╔═╡ 87dc46a4-8f00-473a-9e37-817fc91e7ffe
function discrete_car(t::Real)
	X = Hyperrectangle(
		vcat([50, 4.0, 0.0, 10.0], zeros(3)),
		[100, 3.0, 1.0, 10.0, 1.0, 1.0, 0.2]
	)

	U = Hyperrectangle(
		zeros(2), [0.4, 10.0]
	)

	V = Hyperrectangle(
		zeros(2), [1.0, 10.0]
	)

	cs = ContinuousSystem(X, U, CD.carfield)
	
	κ = (x, u, t) -> [0.4*(u[1] - x[3]), u[2]]

	return DiscreteRandomSystem(cs, V, κ, t)	
end

# ╔═╡ 48f1de66-e038-42db-8044-b5f83c86eacf
function discrete_vehicles(t::Real)

	X = Hyperrectangle(
		vcat([50, 4.0, 0.0, 10.0], zeros(3), 50.0, 1.75, 5.0, 50.0, 6.0, -5.0),
		[100, 3.0, 1.0, 10.0, 1.0, 1.0, 0.2, 100.0, 0.1, 1.0, 100.0, 0.1, 1.0]
	)	

	V = Hyperrectangle(
		zeros(2), [1.0, 10.0]
	)

	function vehicle_transition(x::AbstractVector, u::AbstractVector)
		
		ds = discrete_car(t)
		ego_next = ds(x[1:7], u)
	
		x8next = x[8] + t*x[10]
		x11next = x[11] + t*x[13]
	
		xnext = vcat(ego_next, x8next, x[9:10], x11next, x[12:13])
	
		return xnext
	end	 

	return DiscreteRandomSystem(X, V, vehicle_transition)

end

# ╔═╡ 68e143e5-fdca-4a68-963f-40f02609732b
function gen_trajectory(ds::DiscreteRandomSystem, x0::AbstractVector, T::Integer, H::AbstractMatrix, d::AbstractVector = zeros(size(H, 1)); start_xmat::Union{Nothing, AbstractMatrix}=nothing, start_umat::Union{Nothing, AbstractMatrix} = nothing, lin_x::AbstractVector = x0)
    # 1) Linearization (pick a nominal, here (x0, u_center))
    A, B, c = linearize(ds, lin_x, ds.U.center)  # <-- was `x`

    # 2) Model
    opt = Model(HiGHS.Optimizer)
	JuMP.set_silent(opt)

    # 3) Dimensions and bounds
    n = length(x0)
    m = length(ds.U.center)
    Xlo, Xhi = low(ds.X),  high(ds.X)   # vectors length n
    Ulo, Uhi = low(ds.U),  high(ds.U)   # vectors length m

    # 4) Decision variables
	if start_xmat != nothing
    	@variable(opt, Xlo[i] .<= x[i=1:n, j=1:(T+1)] .<= Xhi[i], start = start_xmat[i, j])
	else
		@variable(opt, Xlo[i] .<= x[i=1:n, j=1:(T+1)] .<= Xhi[i])
	end

	if start_umat != nothing
    	@variable(opt, Ulo[i] .<= u[i=1:m, j=1:T]   .<= Uhi[i], start = start_umat[i, j])
	else
		@variable(opt, Ulo[i] .<= u[i=1:m, j=1:T]   .<= Uhi[i])	
	end
    @variable(opt, u_abs[1:m, 1:T] .>= 0)

    # encode |u| via two linear inequalities: -u_abs <= u <= u_abs
    @constraint(opt, u .<= u_abs)
    @constraint(opt, -u_abs .<= u)

    # 5) Dynamics x_{t+1} = A x_t + B u_t + c
    # Broadcast c across columns 1..T
    Cmat = repeat(c, 1, T)
    @constraint(opt, A * x[:, 1:T] + B * u + Cmat .== x[:, 2:(T+1)])

    # 6) Initial state
    init_constraint = @constraint(opt, x[:, 1] .== x0)

    # 7) Target at final time
	target_con = @constraint(opt, [i=1:size(H,1)], H[i, :]' * x[:, T+1] >= d[i])	

    # 8) Collision avoidance (axis-aligned rectangular disjunctions)
    #    Front vehicle uses state indices (8,9); oncoming uses (11,12)
    #    At each t, enforce: (x1-x8 <= -5) or (x1-x8 >= 5) or (x2-x9 <= -2) or (x2-x9 >= 2)
    #    Same pattern w.r.t. (11,12).

    # # Compute tight difference bounds from X bounds
    Δx_min_f = Xlo[1] - Xhi[8];  Δx_max_f = Xhi[1] - Xlo[8]
    Δy_min_f = Xlo[2] - Xhi[9];  Δy_max_f = Xhi[2] - Xlo[9]

    Δx_min_o = Xlo[1] - Xhi[11]; Δx_max_o = Xhi[1] - Xlo[11]
    Δy_min_o = Xlo[2] - Xhi[12]; Δy_max_o = Xhi[2] - Xlo[12]

	# safety distances
	a = 5.0
	b = 3.0

    # # Big-M values (front)
    M1f = Δx_max_f + a      # for x1-x8 <= -5 + M*(1-b)
    M2f = a - Δx_min_f      # for x1-x8 >=  5 - M*(1-b)
    M3f = Δy_max_f + b      # for x2-x9 <= -2 + M*(1-b)
    M4f = b - Δy_min_f      # for x2-x9 >=  2 - M*(1-b)

    # # Big-M values (oncoming)
    M1o = Δx_max_o + a
    M2o = a - Δx_min_o
    M3o = Δy_max_o + b
    M4o = b - Δy_min_o

    # # Front-vehicle binaries
    @variable(opt, bf[1:3, 1:T+1], Bin)
    @constraint(opt, [t=1:T+1], sum(bf[:, t]) >= 1)  # at least one active

    @constraint(opt, [t=1:T+1], x[1,t] - x[8,t] <= -a + M1f*(1 - bf[1,t]))
    @constraint(opt, [t=1:T+1], x[1,t] - x[8,t] >=  a - M2f*(1 - bf[2,t]))
    # @constraint(opt, [t=1:T+1], x[2,t] - x[9,t] <= -b + M3f*(1 - bf[3,t]))
    @constraint(opt, [t=1:T+1], x[2,t] - x[9,t] >=  b - M4f*(1 - bf[3,t]))

    # # Oncoming-vehicle binaries
    @variable(opt, bo[1:3, 1:T+1], Bin)
    @constraint(opt, [t=1:T+1], sum(bo[:, t]) >= 1)

    @constraint(opt, [t=1:T+1], x[1,t] - x[11,t] <= -a + M1o*(1 - bo[1,t]))
    @constraint(opt, [t=1:T+1], x[1,t] - x[11,t] >=  a - M2o*(1 - bo[2,t]))
    @constraint(opt, [t=1:T+1], x[2,t] - x[12,t] <= -b + M3o*(1 - bo[3,t]))
    # @constraint(opt, [t=1:T+1], x[2,t] - x[12,t] >=  b - M4o*(1 - bo[4,t]))

    # 9) Objective:
	@variable(opt, ϵ[1:(T+1)] .>= 0)
	@constraint(opt, [t = 1:(T+1)], x[2, t] - 1.5 <= ϵ[t] )
	@constraint(opt, [t = 1:(T+1)], x[2, t] - 1.5 >= -ϵ[t] )
	@objective(opt, Min, sum(ϵ) + sum(u_abs))

    optimize!(opt)

	if !is_solved_and_feasible(opt)
		return nothing, false
	end

	umat = value.(u)

	xmat = value.(x)

	Q = Matrix(1.0I, length(center(ds.X)), length(center(ds.X)))
	R = Matrix(1.0I, length(center(ds.U)), length(center(ds.U)))

	K, S, _ = lqr_lyap(A, B, Q, R)

	x_curr = copy(x0)

	strj = copy(xmat)
	utrj = copy(umat)

	strj, utrj = correct_trajectory(ds, xmat, umat)

	Hobs = [
		1 zeros(1, 6) -1 zeros(1, 5);
		-1 zeros(1, 6) 1 zeros(1, 5);
		0 1 zeros(1, 6) -1 zeros(1, 4);
		0 -1 zeros(1, 6) 1 zeros(1, 4);
		1 zeros(1, 9) -1 zeros(1, 2);
		-1 zeros(1, 9) 1 zeros(1, 2);
		0 1 zeros(1, 9) -1 0;
		0 -1 zeros(1, 9) 1 0;
	]

	# collision avoidance with vehicles LHS
	q = [5.0, 5.0, 2.0, 2.0, 5.0, 5.0, 2.0, 2.0]
	margins = Hobs*strj .- q
	_, for_inds = findmax(margins[1:4, :], dims=1)
	_, on_inds = findmax(margins[5:8, :], dims=1)
	on_inds = [CartesianIndex(ind[1]+4, ind[2]) for ind in on_inds]

	# RHS of obstacle constraint
	dobs = fill(-1e7, 8, (T+1))
	for inds in vcat(for_inds, on_inds)
		dobs[inds] = q[inds[1]]
 	end

	# static bounds
	static_margin = minimum(radius_hyperrectangle(ds.X) .- abs.(strj .- center(ds.X)), dims=1)

	return ( state_trajectory = strj, input_signal=utrj, obs_matrix=Hobs, obs_bounds=dobs, target_matrix=H, target_bounds=d, opt=opt, state_ref=x, input_ref=u, target_constraint=target_con, robustness=min(minimum(static_margin), minimum(Hobs*strj - dobs)) , init_constraint=init_constraint, bf_ref=bf, bo_ref=bo, bf_val=value.(bf), bo_val=value.(bo)), true
end

# ╔═╡ 2ff3f387-c91b-4b77-927d-cc40de737a27
function regen_trajectory(ds::DiscreteRandomSystem, prev_res::NamedTuple, x0::AbstractVector, d::AbstractVector; fix_bools::Bool=true)
	res = deepcopy(prev_res)
	
	# change target constraint
	set_normalized_rhs.(res.target_constraint, d)

	# change initial state constraint
	set_normalized_rhs.(res.init_constraint, x0)

	# fix binary variables
	if fix_bools
		fix.(res.bf_ref, res.bf_val)
		fix.(res.bo_ref, res.bo_val)
	else
		if prod(is_fixed.(res.bf_ref))
			unfix.(res.bf_ref)
		end
		if prod(is_fixed.(res.bo_ref))
			unfix.(res.bo_ref)
		end
	end

	# optimize
	optimize!(res.opt)

	if !is_solved_and_feasible(res.opt)
		return prev_res, false
	end

	# compute trajectory
	strj, utrj = correct_trajectory(ds, value.(res.state_ref), value.(res.input_ref))

	T = size(utrj, 2)

	# collision avoidance with vehicles LHS
	Hobs = res.obs_matrix
	q = [5.0, 5.0, 2.0, 2.0, 5.0, 5.0, 2.0, 2.0]
	margins = Hobs*strj .- q
	_, for_inds = findmax(margins[1:4, :], dims=1)
	_, on_inds = findmax(margins[5:8, :], dims=1)
	on_inds = [CartesianIndex(ind[1]+4, ind[2]) for ind in on_inds]

	# RHS of obstacle constraint
	dobs = fill(-1e7, 8, (T+1))
	for inds in vcat(for_inds, on_inds)
		dobs[inds] = q[inds[1]]
 	end


	# static bounds
	static_margin = minimum(radius_hyperrectangle(ds.X) .- abs.(strj .- center(ds.X)), dims=1)

	res = (
	state_trajectory = strj, 
	input_signal=utrj, 
	obs_matrix=res.obs_matrix, 
	obs_bounds=dobs, 
	target_matrix = res.target_matrix,
	target_bounds = d,
	opt=res.opt, 
	state_ref=res.state_ref, 
	input_ref=res.input_ref, 
	target_constraint=res.target_constraint, 
	robustness=min(minimum(static_margin), minimum(Hobs*strj - dobs)), init_constraint=res.init_constraint,
	bf_ref=res.bf_ref,
	bo_ref=res.bo_ref,
	bf_val = value.(res.bf_ref),
	bo_val = value.(res.bo_ref),
	)

	return res, true
end

# ╔═╡ 85b13f02-e662-43f3-99ec-12db657b472c
function generate_data(ds::DiscreteRandomSystem, samples::AbstractVector, margin::AbstractVector{<:Real}, T::Integer; start_index::Integer = 1, last_index::Integer=length(samples), savefile::String="", data::AbstractVector=[])
	state_dim = length(LazySets.center(ds.X))
	input_dim = length(LazySets.center(ds.U))

	proj = vcat(Matrix(1.0I, state_dim, state_dim), Matrix(-1.0I, state_dim, state_dim))

	start_time = time()
	index = start_index

	for s in @view samples[start_index:last_index]
		if s[1] > s[14]
			continue
		end
		x0 = s[1:state_dim]
		d = proj*s[(state_dim+1):end] - abs.(proj)*margin

		res, status = nothing, false

		try
			res, status = gen_trajectory(ds, x0, T, proj, d)
		catch err
			continue
		end

		if status
			if res.robustness >=0
				push!(data, (state_trajectory=res.state_trajectory, input_signal=res.input_signal, index=index))
			end
		end

		if time() - start_time > 60 && !isempty(savefile)
			JLD2.save(savefile, Dict(
					"samples" => samples,
					"data" => data,
				)
			)

			start_time = time()
		end

		index += 1
	end

	if !isempty(savefile)
		JLD2.save(savefile, Dict(
				"samples" => samples,
				"data" => data,
			)
		)
	end

	return data
end

# ╔═╡ 7b3216d8-787e-4654-9827-7f303e42abff
let 
	Random.seed!(5)
	
	context_dim = 13
	sample_dim = 13*4
	num_blocks = 3


	proj = [1 zeros(1, 6) -1 zeros(1, 5)]

	d_gen = [0.0]
	

	ds = discrete_vehicles(0.25)
	x0 = copy(ds.X.center)
	x0[1] = 10.0
	x0[2] = 1.5
	x0[8] = 30
	x0[11] = 90

	ds_sub = discrete_vehicles(0.01)

	L = linearize(ds, x0, ds.U.center)

	# Q = Matrix(1.0I, 7, 7)
	# R = Matrix(1.0I, 2, 2)

	# K, S, _ = lqr_lyap(A[1:7, 1:7], B[1:7, :], Q, R)

	

	T = 28
	lb = repeat(low(ds.X), 2)
	ub = repeat(high(ds.X), 2)
	lb[[1, 14]] .= 0.0
	ub[[1, 14]] .= 90.0
	lb[[4, 17]] .= 5.0
	ub[[4, 17]] .= 15.0
	lb[8] = 30.0
	ub[8] = 30.0
	lb[11] = 90.0
	ub[11] = 90.0
	rect_serp = Hyperrectangle(low=lb, high=ub)
	grid_dims = [1, 2, 4, 14, 15]
	step = [5.0, 1.0, 2.0, 5.0, 1.0]

	samples = grid_serpentine(rect_serp, step, dims=grid_dims)

	margin = vcat(1.0, 0.5, fill(1e7, 11))

	# generate_data(ds, samples, margin, T, savefile="data/car/test.jld2")

	data_dict = JLD2.load("data/car/test.jld2")

	old_data = data_dict["data"]

	# generate_data(ds, samples, margin, T, data=old_data, start_index=old_data[end].index+1, last_index=200, savefile="data/car/test.jld2")

	# res, status = gen_trajectory(ds, x0, T, proj, d_gen)
	# ref_sample = vcat(x0, res.state_trajectory[:, end])
	# ref_sample[4] = 12

	_, check_ind = findmax([s.state_trajectory[1,1] < s.state_trajectory[8, 1] && s.state_trajectory[1,end] > s.state_trajectory[8, end] && s.state_trajectory[2,end] > 2.5 && abs(s.state_trajectory[6,end]) < 0.1 for s in old_data])

	old_data[check_ind]
	
end

# ╔═╡ Cell order:
# ╠═fc3c2db6-7193-11f0-1b65-7f390e18200d
# ╠═c8d1f042-ad49-4971-aa32-b9c15d241da6
# ╠═c6105657-7697-462a-84f0-5b6cb0ffb4b1
# ╠═87dc46a4-8f00-473a-9e37-817fc91e7ffe
# ╠═48f1de66-e038-42db-8044-b5f83c86eacf
# ╠═68e143e5-fdca-4a68-963f-40f02609732b
# ╠═2ff3f387-c91b-4b77-927d-cc40de737a27
# ╠═85b13f02-e662-43f3-99ec-12db657b472c
# ╠═7b3216d8-787e-4654-9827-7f303e42abff

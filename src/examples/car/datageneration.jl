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
    # @constraint(opt, [t=1:T+1], x[2,t] - x[9,t] <= -b + M3f*(1 - bf[3,t])) Removed to impose overtake from right.
    @constraint(opt, [t=1:T+1], x[2,t] - x[9,t] >=  b - M4f*(1 - bf[3,t]))

    # # Oncoming-vehicle binaries
    @variable(opt, bo[1:3, 1:T+1], Bin)
    @constraint(opt, [t=1:T+1], sum(bo[:, t]) >= 1)

    @constraint(opt, [t=1:T+1], x[1,t] - x[11,t] <= -a + M1o*(1 - bo[1,t]))
    @constraint(opt, [t=1:T+1], x[1,t] - x[11,t] >=  a - M2o*(1 - bo[2,t]))
    @constraint(opt, [t=1:T+1], x[2,t] - x[12,t] <= -b + M3o*(1 - bo[3,t]))
    # @constraint(opt, [t=1:T+1], x[2,t] - x[12,t] >=  b - M4o*(1 - bo[4,t])) Removed to impose staying on right of oncoming vehicle while passing.

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
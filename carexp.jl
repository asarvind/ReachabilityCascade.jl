### A Pluto.jl notebook ###
# v0.20.4

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

# ╔═╡ 23334c32-43a8-43e5-ac84-7d007d6f24d1
begin
	using Random, LinearAlgebra
	import JLD2
	using LazySets: Hyperrectangle, low, high
	using Flux: OptimiserChain, ClipNorm, Adam, sigmoid
	import JuMP
	import JuMP: Model, @variable, @constraint, @objective
	import HiGHS
	import NLopt
	import Distributions
	import Distributions: MvNormal
	import LazySets: center
end

# ╔═╡ 9c205052-f2a5-4616-a99e-4ab9d48f4ddf
using ReachabilityCascade: DiscreteRandomSystem, grid_serpentine, ConditionalFlow, NRLE, load, encode, reach

# ╔═╡ 91675df7-f4ef-48e8-ae23-bdf9e4d1ecf7
import ReachabilityCascade.CarDynamics: discrete_vehicles

# ╔═╡ 9f648ae2-043a-404d-9ed2-e985f5f642ba
import ReachabilityCascade.CarDataGeneration: gen_trajectory, generate_data

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

	data_dict = JLD2.load("data/car/trajectories.jld2")

	old_data = data_dict["data"]

	# generate_data(ds, samples, margin, T, data=old_data, start_index=old_data[end].index+1, last_index=200, savefile="data/car/test.jld2")

	# res, status = gen_trajectory(ds, x0, T, proj, d_gen)
	# ref_sample = vcat(x0, res.state_trajectory[:, end])
	# ref_sample[4] = 12

	_, check_ind = findmax([s.state_trajectory[1,1] < s.state_trajectory[8, 1] && s.state_trajectory[1,end] > s.state_trajectory[8, end] && s.state_trajectory[2,end] > 2.5 && abs(s.state_trajectory[6,end]) < 0.1 for s in old_data])

	old_data[check_ind].state_trajectory

	# ds(old_data[check_ind].state_trajectory[:, 1], old_data[check_ind].input_signal) - old_data[check_ind].state_trajectory

	JLD2.load("data/car/test.jld2")
end

# ╔═╡ 8a457e51-7cca-40f5-8448-fdd43672e8a2
function property_fun(strj::AbstractMatrix{<:Real}, utrj::AbstractMatrix{<:Real})
	T = size(utrj, 2)
	
	_, ov_idx = findmin(norm(strj[1:2, i] - strj[8:9, i], Inf) for i in 1:(T+1))

	_, obs_idx = findmin(norm(strj[1:2, i] - strj[11:12, i], Inf) for i in 1:(T+1))

	return vcat(strj[1:2, ov_idx] - strj[8:9, ov_idx], strj[1:2, obs_idx] - strj[11:12, obs_idx], utrj[:, end], T)
end

# ╔═╡ e8dbdee8-bfd1-42b3-ae73-91547f3f33ef
function safety_fun(p::AbstractVector)
	min(
		max(abs(p[1]) - 5.0, p[2] - 2.0),
		max(abs(p[3]) - 5.0, -p[4] - 2.0)
	)
end

# ╔═╡ dc208608-727d-427d-acc0-9b4d90e9661a
let
	seed = rand(1:100000)
	Random.seed!(seed)
	println(seed)
	data = JLD2.load("data/car/trajectories.jld2", "data")
	time_stamps = [1, 2, 4, 8, 16, 28]

	state_scale = ones(13)
	state_scale[[3, 5, 6]] .*= 5

	prop_scale = ones(7)
	prop_scale[5] *= 5

	adj = 0.2
	state_scale .*= adj
	prop_scale .*= adj

	optimizer = OptimiserChain(ClipNorm(), Adam(1e-3))

	# @time train(NRLE, property_fun, data, time_stamps, state_scaling=state_scale, prop_scaling=prop_scale, epochs=10, optimizer=optimizer, savefile="data/car/nrle.jld2", n_blocks=6, hidden=128, n_glu=2, bias=true)
end

# ╔═╡ 9820b742-e777-411b-ae18-0dbdb3448198
function reach_seq(nrle, s_now::AbstractVector, lat::AbstractMatrix)
	xmat = zeros(length(s_now), (size(lat, 2)+1))
	pmat = zeros(
		size(lat, 1) - length(s_now),
		size(lat, 2)
	)
	
	xmat[:, 1] = s_now

	for i in 1:size(lat, 2)
		r = reach(nrle, xmat[:, i], lat[:, i])
		xmat[:, i+1] = r.next_state
		pmat[:, i] = r.property
	end

	return (states=xmat, properties=pmat)
end

# ╔═╡ aa9c7c28-cb8a-4721-a7ff-07a3dd0d94e0
begin
# ======================

import ReachabilityCascade.ControlUtilities: linearize

function linearize(nrle::NRLE, x_now::AbstractVector{<:Real}, latent::AbstractVector{<:Real}, δ=1f-5)
	# dims
	n = length(x_now)
	m = length(latent)
	
	tup = reach(nrle, x_now, latent)
	x_next, prop, ld = tup.next_state, tup.property, tup.log_det

	# latent batch
	lat_batch = hcat(
		repeat(latent, 1, n),
		reduce(hcat, [latent + v*δ for v in eachcol(Matrix(1.0I, m, m))])
	)
	# state batch
	st_batch = hcat(
			reduce(hcat, [x_now + v*δ for v in eachcol(Matrix(1.0I, n, n))]),
			repeat(x_now, 1, m)
		)

	samples = reach(nrle, st_batch, lat_batch)
	jac_state = (samples.next_state .- x_next) ./δ
	jac_prop = (samples.property .- prop) ./δ
	jac_ld = (samples.log_det .- ld) ./δ 
	

	return (state_jacobian=jac_state, prop_jacobian=jac_prop, ld_jacobian=jac_ld, current_state=x_now, latent=latent, next_state=x_next, prop=prop, log_det=sum(ld)/length(ld))
end

# ==========================
end

# ╔═╡ 7801eff4-c841-4ec4-a041-15b0e47e677a
function cem(nrle::NRLE, s_now::AbstractVector, target_mat::AbstractMatrix, safety_fun::Function, H::Hyperrectangle; pop_num::Integer=100, iter::Integer=10)
	# initialize population of latents
	latent_dim = nrle.state_dim + nrle.prop_dim
	D = MvNormal(zeros(latent_dim), Matrix(1.0I, latent_dim, latent_dim))
	latent_batch = rand(D, pop_num)

	# convert current state into a batch
	cur_batch = repeat(s_now, 1, pop_num)

	# calculate the corresponding reachable states and properties
	tup = reach(nrle, cur_batch, latent_batch)

	for _ in 1:iter
		# evaluate fitness
		fits = min.(
			vec(minimum(target_mat*vcat(tup.next_state, ones(1, pop_num)), dims=1)),
			[safety_fun(vec(p)) for p in eachcol(tup.property)]
		)

		# select threshold of fitness
		sort_fits = sort(fits, rev=true)
		threshold = min(sort_fits[(pop_num÷2) + 1], 0)
		sel_idx = (fits .>= threshold) .& [vec(s) in H for s in eachcol(tup.next_state)]
		fit_pop = latent_batch[:, sel_idx]

		# calculate distribution parameters
		μ = vec(sum(fit_pop, dims=2)/pop_num)
		Σ = (fit_pop .- μ)*(fit_pop .- μ)' / pop_num

		# sample from the distribution
		latent_batch = hcat(fit_pop, rand(MvNormal(μ, Σ), pop_num - size(fit_pop, 2)))
		tup = reach(nrle, cur_batch, latent_batch)
	end

	# select solutions
	fits = min.(
			vec(minimum(target_mat*vcat(tup.next_state, ones(1, pop_num)), dims=1)),
			[safety_fun(vec(p)) for p in eachcol(tup.property)]
		)
	sel_idx = (fits .>= 0) .& [vec(s) in H for s in eachcol(tup.next_state)]
	fit_pop = latent_batch[:, sel_idx]

	if size(fit_pop, 2) > 0
		# evaluate log_likelihood of the solutions upto constant difference
		tup = reach(nrle, cur_batch[:, 1:size(fit_pop, 2)], fit_pop)
		ll = tup.log_det - vec(0.5*sum(fit_pop .^2, dims=1))

		# return state and property with best log-likelihood among solutions
		_, best_id = findmax(ll)
		return (next_state = tup.next_state[:, best_id], property=tup.property[:, best_id])
	else
		return (next_state = [], property=[])
	end
end

# ╔═╡ 04a1c35d-042e-41e8-bfba-0ef1f1a7d3d6
begin
# ==========================

function linopt(nrle::NRLE, s_now::AbstractVector, init_latents::AbstractMatrix, target_mat::AbstractMatrix{<:Real}; log_det_bound::Real=Inf, t1_bound=4.0)

	init_reachable = reach_seq(nrle, s_now, init_latents)

	jacs = [
		linearize(nrle, init_reachable.states[:, i], init_latents[:, i])
		for i in 1:size(init_latents, 2)
	]

	num_points = size(init_latents, 2)
	
	opt = Model(HiGHS.Optimizer)
	JuMP.set_silent(opt)

	# state variable
	@variable(opt, x[1:13, 1:(num_points+1)])
	
	# latent variable
	@variable(opt, lat[1:20, 1:num_points])

	# property variable
	@variable(opt, p[1:7, 1:num_points])

	# log determinant variable
	if log_det_bound != Inf
		@variable(opt, ldvar[i=1:num_points] <= log_det_bound)
	end

	# initial state constraint
	@constraint(
		opt, 
		x[:, 1] .== jacs[1].current_state
	)

	# target constraint
	@constraint(opt, [i=1:size(target_mat, 1)], dot(target_mat[i, 1:13],x[:, end]) + target_mat[i, 14] >= 0)

	# time constraint
	@constraint(opt, p[7, 1] .<= t1_bound)
	@constraint(opt, p[7, :] .>= 1)

	# dynamics constraint
	@constraint(
		opt,
		[i=1:num_points],
		jacs[i].state_jacobian*vcat(
			x[:, i] - jacs[i].current_state,
			lat[:, i] - jacs[i].latent
		) + jacs[i].next_state		
		.== x[:, i+1]		
	)

	@constraint(
		opt,
		[i=1:num_points],
		jacs[i].prop_jacobian*vcat(
			x[:, i] - jacs[i].current_state,
			lat[:, i] - jacs[i].latent
		) + jacs[i].prop	
		.== p[:, i]		
	)

	if log_det_bound != Inf
		@constraint(
			opt,
			[i=1:num_points],
			jacs[i].ld_jacobian'*vcat(
				x[:, i] - jacs[i].current_state,
				lat[:, i] - jacs[i].latent
			) + jacs[i].log_det
			>= ldvar[i]
		)
	end

	# =======property constraint =========

	# big M values
	M_long = 90.0
	M_lat = 10.0

	# binary variables
	@variable(opt, for_bin[1:3], Bin)
	@variable(opt, on_bin[1:3], Bin)

	# disjuction of binaries
	@constraint(opt, sum(for_bin) >= 1)
	@constraint(opt, sum(on_bin) >= 1)

	# collision avoidance with forward vehicle
	@constraint(opt, [i=1:num_points],  p[1, i] >= 5.0 + M_long*(for_bin[1] - 1))
	@constraint(opt, [i=1:num_points],  p[1, i] <= -5.0 - M_long*(for_bin[2] - 1))	
	@constraint(opt, [i=1:num_points],  p[2, i] >= 2.0 + M_lat*(for_bin[3] - 1))

	# collision avoidance with oncoming vehicle
	@constraint(opt, [i=1:num_points], p[3, i] >= 5.0 + M_long*(on_bin[1] - 1))
	@constraint(opt, [i=1:num_points], p[3, i] <= -5.0 - M_long*(on_bin[2] - 1))	
	@constraint(opt, [i=1:num_points], p[4, i] <= -2.0 - M_lat*(on_bin[3] - 1))	

	# objective
	@variable(opt, ϵ[i=1:length(jacs[1].latent), j=1:num_points] >= 0)
	@constraint(opt, [i=1:length(jacs[1].latent), j=1:num_points], lat[i, j] <= ϵ[i, j])
	@constraint(opt, [i=1:length(jacs[1].latent), j=1:num_points], lat[i, j] >= -ϵ[i, j])
	# auxillary bounds on first input
	@variable(opt, uabs[1:2] .>= 0)

	if log_det_bound == Inf
		@objective(
			opt, 
			Min,
			sum(ϵ)/2
		)
	else
		@objective(
			opt, 
			Min,
			sum(ϵ)/2  - sum(ldvar)
		)
	end

	JuMP.optimize!(opt)

	# check satisfiability
	seq = reach_seq(nrle, jacs[1].current_state, JuMP.value.(lat))
	# satisfiability of target
	target_sat = prod(target_mat[:, 1:(end-1)]*seq.states[:, end] + target_mat[:, end] .>= 0)
	# satisfiability of forward collision avoidance
	for_mat = copy(seq.properties[1:2, end])
	for_mat[1, :] = abs.(for_mat[1, :])
	for_sat = prod(maximum(for_mat .- [5.0, 2.0], dims=1) .>= 0)
	
	# satisfiability of oncoming collision avoidance
	on_mat = copy(seq.properties[3:4, end])
	on_mat[1, :] = abs.(on_mat[1, :])
	on_mat[2, :] = -on_mat[2, :]
	on_sat = prod(maximum(on_mat .- [5.0, 2.0], dims=1) .>= 0)

	return JuMP.value.(lat), (target_sat && for_sat && on_sat)
end

function linopt(nrle::NRLE, s_now::AbstractVector, init_latents::AbstractMatrix, target_mat::AbstractMatrix{<:Real}, iter::Integer; kwargs...)
	lat = init_latents
	u = zeros(2)

	for _ in 1:iter
		lat, sat = linopt(nrle, s_now, lat, target_mat; kwargs...)
		if sat 
			break
		end
	end

	return reach_seq(nrle, s_now, lat)
end

# ============================
end

# ╔═╡ 94bff29a-2da3-406f-a5d1-4ecbb9f60785
function mpc(ds::DiscreteRandomSystem, steps::Integer, x_now::AbstractVector, nrle::NRLE, target_mat::AbstractMatrix, init_latents::AbstractMatrix, iter::Integer; kwargs...)
	κ = x::AbstractVector -> linopt(nrle, x, init_latents, target_mat, iter; kwargs...).properties[5:6, 1]

	ds(x_now, κ, steps)
end

# ╔═╡ d8e24f44-4fe2-4128-851c-057e88fce51f
begin
# include("tempcode.jl")
# struct LMTE{E}
#     context_dim::Integer 
#     control_dim::Integer 
#     transfer_dim::Integer
#     experts::Vector{E}
# end
end

# ╔═╡ 3441412f-b56e-4906-863c-1a8b656b7ae9
let
	Random.seed!(0)
	data = JLD2.load("data/car/trajectories.jld2", "data")
	nrle = load(NRLE, "data/car/nrle.jld2", property_fun, data)

	ind = rand(1:length(data))
	# ind = 4802
	println("data index = $ind")
	x_now = data[ind].state_trajectory[:,1]
	x_now[[1, 2, 4]] = [10, 1.7, 10]
	
	x_next, prop = reach(nrle, x_now, randn(20))
	x_next, prop = vec(x_next), vec(prop)

	encode(nrle, x_now, x_next, prop)

	jacs = linearize(nrle, x_now, 0*randn(20))

	target_mat = zeros(2, 14)
	target_mat[1, [1, 8, 14]] = [1, -1, 3.0]
	target_mat[2, [1, 11]] = [1, -1]

	ds = discrete_vehicles(0.25)
	# lincon = linearize(ds, center(ds.X))
	# optlatents = linopt(nrle, x_now, zeros(20, 3), target_mat, 10, t1_bound=10.0)
	# @time for _ in 1:100
	# 	cem(nrle, x_now, target_mat, safety_fun, ds.X; pop_num=100, iter=4)
	# end

	# iter = 4
	# steps = 10
	# x = mpc(ds, steps, x_now, nrle, target_mat, zeros(20, 2), iter, t1_bound=14.0)[:, end]
	# linopt(nrle, xtrj[:,end], zeros(20, 2), target_mat, 10, t1_bound=2.0)
	# x = mpc(ds, 2, x, nrle, target_mat, zeros(20, 2), iter, t1_bound=1.0)[:, end]
	# x = mpc(ds, steps, x, nrle, target_mat, zeros(20, 2), iter)[:, end]
end

# ╔═╡ Cell order:
# ╠═fc3c2db6-7193-11f0-1b65-7f390e18200d
# ╠═23334c32-43a8-43e5-ac84-7d007d6f24d1
# ╠═9c205052-f2a5-4616-a99e-4ab9d48f4ddf
# ╠═91675df7-f4ef-48e8-ae23-bdf9e4d1ecf7
# ╠═9f648ae2-043a-404d-9ed2-e985f5f642ba
# ╠═7b3216d8-787e-4654-9827-7f303e42abff
# ╠═8a457e51-7cca-40f5-8448-fdd43672e8a2
# ╠═e8dbdee8-bfd1-42b3-ae73-91547f3f33ef
# ╠═dc208608-727d-427d-acc0-9b4d90e9661a
# ╠═9820b742-e777-411b-ae18-0dbdb3448198
# ╠═aa9c7c28-cb8a-4721-a7ff-07a3dd0d94e0
# ╠═7801eff4-c841-4ec4-a041-15b0e47e677a
# ╠═04a1c35d-042e-41e8-bfba-0ef1f1a7d3d6
# ╠═94bff29a-2da3-406f-a5d1-4ecbb9f60785
# ╠═d8e24f44-4fe2-4128-851c-057e88fce51f
# ╠═3441412f-b56e-4906-863c-1a8b656b7ae9

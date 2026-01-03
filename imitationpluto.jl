### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ deaa37a2-e572-11f0-1d8d-d3dddde17a2e
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

# ╔═╡ 70fe22ad-d8dc-4776-ba3b-a50399325484
begin
	
	using Random, LinearAlgebra
	import NLopt
	using Flux
	import JLD2
	import LazySets
	import LazySets: center, radius_hyperrectangle
	import ReachabilityCascade.CarDynamics: discrete_vehicles
	import ReachabilityCascade: DiscreteRandomSystem, InvertibleCoupling
	import ReachabilityCascade.InvertibleGame: inclusion_losses, load_game, decode
	import ReachabilityCascade.TrainingAPI: build
	
end

# ╔═╡ 9a1f6973-1363-4207-8a8f-713a47d253ce
function trajectory(ds::DiscreteRandomSystem, model::InvertibleCoupling, x0::AbstractVector, z::AbstractVector, steps::Union{Integer, AbstractVector{<:Integer}})
	zmat = reshape(z, model.dim, length(steps))
	strj = x0	

	for i in 1:length(steps)
		t = steps[i]
		κ = x -> decode(model, zmat[:, i], x)
		x_start = strj[:, end]
		strj = hcat(strj, ds(x_start, κ, t)[:, 2:end])
	end

	return strj
end

# ╔═╡ e4495e22-4860-4a8a-8fd0-7c696b54a310
function optimize_latent(cost_fn::Function, ds::DiscreteRandomSystem, x0::AbstractVector, model::InvertibleCoupling, steps::Union{Integer, AbstractVector{<:Integer}}; algo::Symbol=:LN_BOBYQA, init_z::AbstractVector=zeros(Float32, model.dim*length(steps)), max_time::Real=Inf, seed::Integer=rand(1:10000))
	
	function my_objective_fn(z::AbstractVector, grad::AbstractVector)
		strj = trajectory(ds, model, x0, z, steps)			
		return sum(cost_fn(strj))/size(strj, 2)
	end

	l = model.dim*length(steps)
	
	opt = NLopt.Opt(algo, l)
	NLopt.lower_bounds(opt, -ones(Float64, l))
	NLopt.upper_bounds(opt, ones(Float64, l))
	NLopt.min_objective!(opt, my_objective_fn)
	NLopt.stopval!(opt, 0)
	NLopt.maxtime!(opt, max_time)
	NLopt.srand(seed)

	min_f, min_z, ret = NLopt.optimize(opt, init_z)
end

# ╔═╡ e40e3a60-e812-456d-affc-10bd3e072d21
function mpc(cost_fn::Function, ds::DiscreteRandomSystem, x0::AbstractVector, model::InvertibleCoupling, steps::Integer; algo::Symbol=:LN_PRAXIS, init_z::AbstractVector=zeros(Float32, model.dim*length(steps)), noise_fn::Union{Function, Nothing}=nothing, noise_weight::Real=0.0, noise_rng::Random.AbstractRNG=Random.default_rng(), max_time::Real=Inf, opt_steps::Union{Integer, AbstractVector{<:Integer}}=steps, opt_seed::Integer=rand(1:10000))
	obj_vals = []
	strj = x0
	z = init_z

	if noise_fn == nothing
		nf = () -> LazySets.sample(ds.U; rng=noise_rng)
	else
		nf = noise_fn
	end

	for _ in 1:sum(steps)
		x = strj[:, end]
		obj_val, z, _ = optimize_latent(cost_fn, ds, x, model, opt_steps; algo=algo, init_z=z, max_time=max_time)
		u = decode(model, z[1:model.dim], x)[1:length(center(ds.U))]
		u_noise = nf()
		u = u*(1-noise_weight) + u_noise*noise_weight
		strj = hcat(strj, ds(x, u))
		push!(obj_vals, obj_val)
	end
	
	return strj, sum(cost_fn(strj)), obj_vals
end

# ╔═╡ d0fc8b98-37ca-4a26-8289-2b408846f7b6
	begin
		"""
		    CarFlowIterator(data; idxs=1:length(data), rng=Random.default_rng(), shift_range=0.0)

		Iterator for training InvertibleGame networks from a trajectory dataset.

This iterator assumes `data` is an indexable collection (e.g. vector) whose elements provide:
- `sample.state_trajectory`: a `state_dim × (T+1)` (or `state_dim × T`) array-like object.
- `sample.input_signal`: an `input_dim × T` array-like object.

On each iteration, we return a named tuple:
`(; context, sample)` where:
- `context` is the state vector `state_trajectory[:, t]`
- `sample` is the control vector `input_signal[:, t]`

Traversal rule (single pass)
- We iterate over trajectories in a shuffled order (reshuffled each epoch if enabled).
- For each trajectory, we iterate over all time indices `t = 1:T` in order.
- This means we do *not* miss any `(state, input)` pair within the selected trajectories.

	Notes
	- This iterator is *re-iterable*: `Base.iterate(it)` always starts a fresh epoch from the first trajectory.
	- If `shuffle_each_epoch=true`, trajectory order is reshuffled at the start of each epoch using `rng`.
	- If `shift_range>0`, the longitudinal positions of the 1st, 2nd, and 3rd vehicles
	  (state indices `1`, `8`, `11`) are translated by the same random shift `Δ ∈ [-shift_range, shift_range]`
	  *per returned sample*. This preserves relative geometry while augmenting translation invariance.
		"""
		struct CarFlowIterator
			data
			idxs::Vector{Int}
			rng
			shuffle_each_epoch::Bool
			shift_range::Float64
		end

		function CarFlowIterator(data;
		                         idxs=1:length(data),
		                         rng=Random.default_rng(),
		                         shuffle_each_epoch::Bool=true,
		                         shift_range::Real=0.0)
			shift_range_f = Float64(shift_range)
			shift_range_f >= 0 || throw(ArgumentError("shift_range must be non-negative; got $shift_range"))
			return CarFlowIterator(data, collect(Int, idxs), rng, shuffle_each_epoch, shift_range_f)
		end
	end

# ╔═╡ f16a452c-fc3c-427d-8e43-bae5db10dddd
begin
	function Base.iterate(it::CarFlowIterator, state=nothing)
		# State holds:
		# - traj_pos: position in `it.idxs`
		# - t: time index within the current trajectory
		# - X, U, T: cached arrays for the current trajectory to avoid re-loading every step
		if state === nothing
			# New epoch start: shuffle trajectory order if requested.
			if it.shuffle_each_epoch
				Random.shuffle!(it.rng, it.idxs)
			end

			traj_pos = 1
			traj_pos > length(it.idxs) && return nothing
			d = it.data[it.idxs[traj_pos]]
			X = Array(d.state_trajectory)
			U = Array(d.input_signal)
			T = size(U, 2)
			T >= 1 || throw(ArgumentError("input_signal must have at least one column; got size=$(size(U))"))
			size(X, 2) >= T ||
			    throw(DimensionMismatch("state_trajectory must have at least T=$T columns; got size=$(size(X))"))
			state = (traj_pos=traj_pos, t=1, X=X, U=U, T=T)
		end

		traj_pos = state.traj_pos
		t = state.t
		X = state.X
		U = state.U
		T = state.T

		# If this trajectory is done, advance to the next trajectory and reset t.
		while t > T
			traj_pos += 1
			traj_pos > length(it.idxs) && return nothing
			d = it.data[it.idxs[traj_pos]]
			X = Array(d.state_trajectory)
			U = Array(d.input_signal)
			T = size(U, 2)
			T >= 1 || throw(ArgumentError("input_signal must have at least one column; got size=$(size(U))"))
			size(X, 2) >= T ||
			    throw(DimensionMismatch("state_trajectory must have at least T=$T columns; got size=$(size(X))"))
			t = 1
			end

			context = Vector(X[:, t])
			sample = Vector(U[:, t])

			if it.shift_range > 0
				length(context) >= 11 || throw(DimensionMismatch("context must have at least 11 entries to shift indices (1, 8, 11); got length=$(length(context))"))
				Δ = (2 * rand(it.rng) - 1) * it.shift_range
				context[1] += Δ
				context[8] += Δ
				context[11] += Δ
			end

			next_state = (traj_pos=traj_pos, t=t + 1, X=X, U=U, T=T)
			return ((; context=context, sample=sample), next_state)
		end
	end

# ╔═╡ d7873320-cce3-478b-8fa7-93b1063eb0c4
let
	# Dataset format example: `data/car/trajectories.jld2` (key "data").
	# This follows the same overtake-data filter used in `vehiclepluto.jl`.
	data = JLD2.load("data/car/trajectories.jld2", "data")
	overtake_idx = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 for d in data]
	overtake_data = data[overtake_idx]

	# ----------------------------
	# InvertibleCoupling args (edit these)
	# ----------------------------
	# Spec is a 3×L integer matrix:
	#   row 1: hidden widths
	#   row 2: depths (n_glu)
	#   row 3: coupling type flags (1 affine, 0 additive)
	spec = [128 128 128;
	        1   1   1;
	        1   1   1]
	logscale_clamp = 2.0
	margin_true = 1.0
	margin_adv = 0.0

	# Quick switch for "inclusion-only" training:
	# set `w_reject = 0.0` and `w_fool = 0.0` (keep `w_true = 1.0`).
	w_true = 1.0
	w_reject = 1.0
	w_fool = 1.0

	# ----------------------------
	# Training args (edit these)
	# ----------------------------

	# epochs = 15
	epochs = 0
	batch_size = 100
	opt = Flux.OptimiserChain(Flux.ClipGrad(), Flux.ClipNorm(), Flux.Adam(1f-4))
	use_memory = true
	merge = :sum
	save_path = "data/car/temp/invertiblegame.jld2"
	load_path = save_path
	save_period = 60.0

	# DATA SLICE: edit this to control how much data is used for training.
	# Keep this small while sanity-checking the notebook (the dataset is large).
	thisdata = overtake_data[1:end]
	it = CarFlowIterator(thisdata; rng=Random.MersenneTwister(0))

	# Two-player symmetric training: each network both fools and rejects the other.
	model_a, model_b, losses_a, losses_b = build(InvertibleCoupling, it;
		spec=spec,
		logscale_clamp=logscale_clamp,
		margin_true=margin_true,
		margin_adv=margin_adv,
		w_true=w_true,
		w_reject=w_reject,
		w_fool=w_fool,
		epochs=epochs,
		batch_size=batch_size,
		opt=opt,
		use_memory=use_memory,
		merge = merge,
		save_path=save_path,
		load_path=load_path,
		save_period=save_period,
		rng=Random.MersenneTwister(1),      # latent sampling RNG
		rng_a=Random.MersenneTwister(2),    # model A init RNG (permutations/weights)
		rng_b=Random.MersenneTwister(3),    # model B init RNG (permutations/weights)
	)

	(; losses_a, losses_b)
	
end

# ╔═╡ 5a9f6a0a-9a6a-4d21-ae4b-67d122a2b2b3
begin
	"""
	    CarFlowSequentialIterator(data; idxs=1:length(data))

	Sequential iterator over *all* `(state, input)` pairs in a trajectory dataset.

This iterator assumes `data` is an indexable collection (e.g. vector) whose elements provide:
- `sample.state_trajectory`: a `state_dim × (T+1)` (or `state_dim × T`) array-like object.
- `sample.input_signal`: an `input_dim × T` array-like object.

For each trajectory, we iterate time indices `t = 1:T` in order and return a named tuple:
`(; context, sample)` where:
- `context` is the state vector `state_trajectory[:, t]`
- `sample` is the control vector `input_signal[:, t]`

Notes
- This iterator does *not* skip any time indices (unlike the random sampler used during training).
- It is re-iterable: `Base.iterate(it)` always starts from the first trajectory and `t=1`.
	"""
	struct CarFlowSequentialIterator
		data
		idxs::Vector{Int}
	end

	function CarFlowSequentialIterator(data; idxs=1:length(data))
		return CarFlowSequentialIterator(data, collect(Int, idxs))
	end
end

# ╔═╡ fdbb52af-2e55-4a09-9b9b-7a0bdb9f4c67
begin
	function Base.iterate(it::CarFlowSequentialIterator, state=nothing)
		# State holds:
		# - traj_pos: position in `it.idxs`
		# - t: time index within the current trajectory
		# - X, U, T: cached arrays for the current trajectory to avoid re-loading every step
		if state === nothing
			traj_pos = 1
			traj_pos > length(it.idxs) && return nothing
			d = it.data[it.idxs[traj_pos]]
			X = Array(d.state_trajectory)
			U = Array(d.input_signal)
			T = size(U, 2)
			T >= 1 || throw(ArgumentError("input_signal must have at least one column; got size=$(size(U))"))
			size(X, 2) >= T ||
			    throw(DimensionMismatch("state_trajectory must have at least T=$T columns; got size=$(size(X))"))
			state = (traj_pos=traj_pos, t=1, X=X, U=U, T=T)
		end

		traj_pos = state.traj_pos
		t = state.t
		X = state.X
		U = state.U
		T = state.T

		# If this trajectory is done, advance to the next trajectory and reset t.
		while t > T
			traj_pos += 1
			traj_pos > length(it.idxs) && return nothing
			d = it.data[it.idxs[traj_pos]]
			X = Array(d.state_trajectory)
			U = Array(d.input_signal)
			T = size(U, 2)
			T >= 1 || throw(ArgumentError("input_signal must have at least one column; got size=$(size(U))"))
			size(X, 2) >= T ||
			    throw(DimensionMismatch("state_trajectory must have at least T=$T columns; got size=$(size(X))"))
			t = 1
		end

		context = Vector(X[:, t])
		sample = Vector(U[:, t])

		next_state = (traj_pos=traj_pos, t=t + 1, X=X, U=U, T=T)
		return ((; context=context, sample=sample), next_state)
	end
end

# ╔═╡ 4d7c7db1-9ed8-4c98-9d79-1c0a1a4fa07e
let
	# Ground-truth inclusion testing (single pass).
	#
	# This uses `CarFlowSequentialIterator` so every time index of every selected trajectory is evaluated.

	save_path = "data/car/temp/invertiblegame.jld2"

	model_a, model_b, _ = load_game(save_path)

	# Test-time settings (local to this cell).
	margin_true_test = 1.0
	batch_size_test = 10

	data = JLD2.load("data/car/trajectories.jld2", "data")
	overtake_idx = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 for d in data]
	overtake_data = data[overtake_idx]

	# TEST DATA SLICE: edit this to control how much data is used for testing.
	thisdata_test = overtake_data[1:1]
	it_test = CarFlowSequentialIterator(thisdata_test)

	losses_a_inclusion = inclusion_losses(model_a, it_test; batch_size=batch_size_test, margin_true=margin_true_test)
	losses_b_inclusion = inclusion_losses(model_b, it_test; batch_size=batch_size_test, margin_true=margin_true_test)

	(; losses_a_inclusion, losses_b_inclusion)
end

# ╔═╡ 580a6ae6-5f95-4ec2-ab9d-b5f6d48d330b
function thiscost(x::AbstractMatrix)
	ds = discrete_vehicles(0.25)

	# bounds cost 
	bc = Flux.relu(abs.(x .- center(ds.X)) .- radius_hyperrectangle(ds.X))

	# forward collision cost
	fc = Flux.relu(min.(5.0 .- abs.(x[8:8, :] - x[1:1, :]), 2.0 .+ x[9:9, :] - x[2:2, :]))

	# oncoming collision cost
	oc = Flux.relu(min.(5.0 .- abs.(x[11:11, :] - x[1:1, :]), 2.0 .+ x[2:2, :] - x[12:12, :]))

	# terminal cost
	tc = fill(Flux.relu(x[8, end] - x[1, end] + 3.0), 1, size(x, 2))
	# tc = vcat(tc, fill(Flux.relu(x[2, end] - 3.0), 1, size(x, 2)))

	return vcat(bc, fc, oc, tc)
end

# ╔═╡ 730088b2-08f0-400b-98f2-5298ab5b9eb5
let
	ds = discrete_vehicles(0.25; dt=0.01)
	data = JLD2.load("data/car/trajectories.jld2", "data")
	overtake_idx = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 && d.state_trajectory[1, 1] < d.state_trajectory[8, 1] for d in data]
	overtake_data = data[overtake_idx]
	save_path = "data/car/temp/invertiblegame.jld2"
	model_a, model_b, _ = load_game(save_path)
	z = randn(2)
	z = Float32.(z*rand() / norm(z, 1))
	κ = x -> decode(model_a, z, x)
	κ_z = x -> decode(model_a, z, x)
	idtest = rand(1:length(overtake_data))
	# idtest = 1
	strj, utrj = overtake_data[idtest]
	start_time = 1
	x0 = strj[:, start_time]
	# steps = [15, 28 - start_time - 15 + 1]
	steps = 28

	algo = :LN_COBYLA
	max_time = 0.1

	@time res_a = optimize_latent(thiscost, ds, x0, model_a, steps; algo=algo, max_time=max_time)
	@time res_b = optimize_latent(thiscost, ds, x0, model_b, steps; algo=algo, max_time=max_time)

	res_a, res_b, idtest, x0

	obj_a, z_a, _ = res_a
	obj_b, z_b, _ = res_b

	
	trajectory(ds, model_a, x0, z_a, steps), trajectory(ds, model_b, x0, z_b, steps), obj_a, obj_b, z_a, z_b, x0, idtest

	mpc(thiscost, ds, x0, model_a, 20; algo=algo, max_time=max_time, noise_weight=0.00, noise_rng=MersenneTwister(1), opt_steps=20, opt_seed=1)
end

# ╔═╡ Cell order:
# ╠═deaa37a2-e572-11f0-1d8d-d3dddde17a2e
# ╠═70fe22ad-d8dc-4776-ba3b-a50399325484
# ╠═9a1f6973-1363-4207-8a8f-713a47d253ce
# ╠═e4495e22-4860-4a8a-8fd0-7c696b54a310
# ╠═e40e3a60-e812-456d-affc-10bd3e072d21
# ╠═d0fc8b98-37ca-4a26-8289-2b408846f7b6
# ╠═f16a452c-fc3c-427d-8e43-bae5db10dddd
# ╠═d7873320-cce3-478b-8fa7-93b1063eb0c4
# ╠═5a9f6a0a-9a6a-4d21-ae4b-67d122a2b2b3
# ╠═fdbb52af-2e55-4a09-9b9b-7a0bdb9f4c67
# ╠═4d7c7db1-9ed8-4c98-9d79-1c0a1a4fa07e
# ╠═580a6ae6-5f95-4ec2-ab9d-b5f6d48d330b
# ╠═730088b2-08f0-400b-98f2-5298ab5b9eb5

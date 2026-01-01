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
	
	using Random
	using Flux
	import JLD2
	import ReachabilityCascade.CarDynamics: discrete_vehicles
	import ReachabilityCascade: DiscreteRandomSystem, InvertibleCoupling
	import ReachabilityCascade.InvertibleGame: inclusion_losses, load_game
	import ReachabilityCascade.TrainingAPI: build
	
end

# ╔═╡ d0fc8b98-37ca-4a26-8289-2b408846f7b6
begin
	"""
	    CarFlowIterator(data; idxs=1:length(data), rng=Random.default_rng())

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
	"""
	struct CarFlowIterator
		data
		idxs::Vector{Int}
		rng
		shuffle_each_epoch::Bool
	end

	function CarFlowIterator(data; idxs=1:length(data), rng=Random.default_rng(), shuffle_each_epoch::Bool=true)
		return CarFlowIterator(data, collect(Int, idxs), rng, shuffle_each_epoch)
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
	        0   0   1]
	logscale_clamp = 2.0
	margin_true = 0.9
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

# ╔═╡ 730088b2-08f0-400b-98f2-5298ab5b9eb5
let
	ds = discrete_vehicles(0.25)
end

# ╔═╡ Cell order:
# ╠═deaa37a2-e572-11f0-1d8d-d3dddde17a2e
# ╠═70fe22ad-d8dc-4776-ba3b-a50399325484
# ╠═d0fc8b98-37ca-4a26-8289-2b408846f7b6
# ╠═f16a452c-fc3c-427d-8e43-bae5db10dddd
# ╠═d7873320-cce3-478b-8fa7-93b1063eb0c4
# ╠═5a9f6a0a-9a6a-4d21-ae4b-67d122a2b2b3
# ╠═fdbb52af-2e55-4a09-9b9b-7a0bdb9f4c67
# ╠═4d7c7db1-9ed8-4c98-9d79-1c0a1a4fa07e
# ╠═730088b2-08f0-400b-98f2-5298ab5b9eb5

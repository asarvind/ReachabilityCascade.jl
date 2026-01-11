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
		import ReachabilityCascade: DiscreteRandomSystem, InvertibleCoupling, NormalizingFlow, load
		import ReachabilityCascade.InvertibleGame: inclusion_losses, load_game, decode, load_self
		import ReachabilityCascade: trajectory, optimize_latent, mpc
		import ReachabilityCascade.TrainingAPI: build
		
	end
	

# ╔═╡ 9a1f6973-1363-4207-8a8f-713a47d253ce
	# These helpers live in the package now:
	# - `ReachabilityCascade.trajectory`
	# - `ReachabilityCascade.optimize_latent`
	# - `ReachabilityCascade.mpc`

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

	# epochs = 45
	epochs = 0
	batch_size = 100
	opt = Flux.OptimiserChain(Flux.ClipGrad(), Flux.ClipNorm(), Flux.Adam(1f-4))
	use_memory = true
	# EMA opponent smoothing schedule (used for opponent EMA during gradient computation).
	ema_beta_start = 0.0
	ema_beta_final = 0.999
	ema_tau = 1f4
	# Lower bound on sampled fake latent norm (0 means allow near-zero latents).
	latent_radius_min = 0.0
	# Gradient mode:
	# - :sum            => one gradient on full loss
	# - :orthogonal_adv => split gradients and orthogonalize adversarial component
	grad_mode = :sum
	norm_kind = :l2
	save_path = "data/car/temp/selfinvertible.jld2"
	load_path = save_path
	save_period = 60.0

	# DATA SLICE: edit this to control how much data is used for training.
	# Keep this small while sanity-checking the notebook (the dataset is large).
	thisdata = overtake_data[1:end]
	it_game = CarFlowIterator(thisdata; rng=Random.MersenneTwister(0))

	# Set `mode=:game` (two-player) or `mode=:self` (single-network, EMA provides fakes).
	mode = :self

	if mode == :game
		# Two-player symmetric training: each network both fools and rejects the other.
		model_a, model_b, losses_a, losses_b = build(InvertibleCoupling, it_game;
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
			latent_radius_min=latent_radius_min,
			ema_beta_start=ema_beta_start,
			ema_beta_final=ema_beta_final,
			ema_tau=ema_tau,
			grad_mode=grad_mode,
			norm_kind=norm_kind,
			save_path=save_path,
			load_path=load_path,
			save_period=save_period,
			rng=Random.MersenneTwister(1),      # latent sampling RNG
			rng_a=Random.MersenneTwister(2),    # model A init RNG (permutations/weights)
			rng_b=Random.MersenneTwister(3),    # model B init RNG (permutations/weights)
		)
		(; losses_a, losses_b)
	else
		save_path_self = replace(save_path, "invertiblegame" => "invertiblegame_self")
		load_path_self = save_path_self
		model, ema, losses = build(InvertibleCoupling, it_game, :self;
			spec=spec,
			logscale_clamp=logscale_clamp,
			margin_true=margin_true,
			margin_adv=margin_adv,
			w_true=w_true,
			w_reject=w_reject,
			epochs=epochs,
			batch_size=batch_size,
			opt=opt,
			use_memory=use_memory,
			latent_radius_min=latent_radius_min,
			ema_beta_start=ema_beta_start,
			ema_beta_final=ema_beta_final,
			ema_tau=ema_tau,
			grad_mode=grad_mode,
			norm_kind=norm_kind,
			save_path=save_path_self,
			load_path=load_path_self,
			save_period=save_period,
			rng=Random.MersenneTwister(1),       # latent sampling RNG
			rng_model=Random.MersenneTwister(200), # model init RNG
		)
		(; losses, model, ema)
	end
		
end

# ╔═╡ 3b6f10e2-41c2-4b5a-9e63-5c15b0f8e5a1
let
	# NormalizingFlow baseline (trained on the same dataset/iterator format).
	#
	# NOTE: `epochs=0` means this cell will just load from `load_path_flow` (if present) and save to `save_path_flow`.
	# Set `epochs>0` when you actually want to train.

	# Dataset format example: `data/car/trajectories.jld2` (key "data").
	# This follows the same overtake-data filter used in `vehiclepluto.jl`.
	data = JLD2.load("data/car/trajectories.jld2", "data")
	overtake_idx = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 for d in data]
	overtake_data = data[overtake_idx]

	# ----------------------------
	# NormalizingFlow args (match InvertibleCoupling config)
	# ----------------------------
	spec = [128 128 128;
	        1   1   1;
	        1   1   1]
	logscale_clamp = 2.0

	# ----------------------------
	# Training args (match InvertibleCoupling config)
	# ----------------------------
	epochs = 0
	batch_size = 100
	opt = Flux.OptimiserChain(Flux.ClipGrad(), Flux.ClipNorm(), Flux.Adam(1f-4))
	save_period = 60.0

	# DATA SLICE: edit this to control how much data is used for training.
	# Keep this small while sanity-checking the notebook (the dataset is large).
	thisdata = overtake_data[1:end]

	save_path_flow = "data/car/temp/normalizingflow.jld2"
	load_path_flow = save_path_flow

	# Separate iterator instance so (when `epochs>0`) both trainings can be reproducible independently.
	# Use the same iterator RNG seed as the InvertibleCoupling training cell for reproducibility.
	it_flow = CarFlowIterator(thisdata; rng=Random.MersenneTwister(0))

	flow, losses_flow = build(NormalizingFlow, it_flow;
		spec=spec,
		logscale_clamp=logscale_clamp,
		# Use a fixed init RNG (same seed as `rng_a` in the InvertibleCoupling cell) for reproducibility.
		rng=Random.MersenneTwister(2000),
		epochs=epochs,
		batch_size=batch_size,
		opt=opt,
		use_memory=false,
		save_path=save_path_flow,
		load_path=load_path_flow,
		save_period=save_period,
	)

	(; losses_flow)
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

	save_path = "data/car/selfadversarial.jld2"

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

	losses_a_inclusion = inclusion_losses(model_a, it_test; batch_size=batch_size_test, margin_true=margin_true_test, norm_kind=:l1)
	losses_b_inclusion = inclusion_losses(model_b, it_test; batch_size=batch_size_test, margin_true=margin_true_test, norm_kind=:l1)

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
	save_game = "data/car/unitinvert/selfInitSeed2000Iter0Epoch45EmaL0U999R1f4Latseed1.jld2"
	save_flow = "data/car/flowmodels/flowInitSeed2000Iter0Epoch45.jld2"
	model_a, _ = load_self(save_game)
	model_flow = load(NormalizingFlow, save_flow)
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

	algo = :LN_PRAXIS
	max_time = 0.02

	noise_rng_flow = MersenneTwister(rand(1:10000))
	noise_rng_game_a = deepcopy(noise_rng_flow)
	noise_rng_game_b = deepcopy(noise_rng_flow)

	drift_steps = 2
	res_drift = mpc(thiscost, ds, x0, model_flow, drift_steps; algo=algo, max_time=max_time, noise_weight=0.2, noise_rng=noise_rng_game_b, opt_steps=[14, 14], opt_seed=1)
	x_drift = res_drift.trajectory[:, drift_steps]
	# x_drift = x0

	res_game_a = mpc(thiscost, ds, x_drift, [model_a], 20; algo=algo, max_time=max_time, noise_weight=0.0, noise_rng=noise_rng_game_a, opt_steps=[28], opt_seed=1)

	# res_game_b = mpc(thiscost, ds, x0, model_b, 20; algo=algo, max_time=max_time, noise_weight=0.2, noise_rng=noise_rng_game_b, opt_steps=[10, 10], opt_seed=1)

	res_flow = mpc(thiscost, ds, x_drift, model_flow, 28; algo=algo, max_time=max_time, noise_weight=0.0, noise_rng=noise_rng_flow, opt_steps=[28], opt_seed=1)

	res_game_a, res_flow
end

# ╔═╡ 39a52a27-dd19-446e-9f43-520b170bac8c
let
	ds = discrete_vehicles(0.25; dt=0.01)
	data = JLD2.load("data/car/trajectories.jld2", "data")
	overtake_idx = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 && d.state_trajectory[1, 1] < d.state_trajectory[8, 1] for d in data]
	overtake_data = data[overtake_idx]
	
	invunit_path = "data/car/unitinvert/selfInitSeed200Iter0Epoch45EmaL0U999R1f4Latseed1.jld2"
	flow_path = "data/car/flowmodels/flowInitSeed200Iter0Epoch45.jld2"
	model_invunit, _ = load_self(invunit_path)
	model_flow = load(NormalizingFlow, flow_path)


	algo = :LN_COBYLA
	max_time = 0.02
	drift_steps = 2

	num_sim = 10

	noise_rng = MersenneTwister(200)

	cost_res = Vector{Float64}[]

	data_shuffled = shuffle(noise_rng, overtake_data)

	for i in 1:num_sim 
		strj, utrj = data_shuffled[i]
		x0 = strj[:, 1]
		
		res_drift = mpc(thiscost, ds, x0, model_flow, drift_steps; algo=algo, max_time=max_time, noise_weight=0.2, noise_rng=noise_rng, opt_steps=[28], opt_seed=1)
		x_drift = res_drift.trajectory[:, drift_steps]

		res_invunit = mpc(thiscost, ds, x_drift, model_invunit, 20; algo=algo, max_time=max_time, noise_weight=0.0, opt_steps=[28], opt_seed=1)
		
		res_flow = mpc(thiscost, ds, x_drift, model_flow, 28; algo=algo, max_time=max_time, noise_weight=0.0, opt_steps=[28], opt_seed=1)

		res_id_stair = mpc(thiscost, ds, x_drift, (x, z)->z, 28; algo=algo, max_time=max_time, noise_weight=0.0, opt_steps=[28], opt_seed=1, latent_dim = 2)

		res_id_full = mpc(thiscost, ds, x_drift, (x, z)->z, 28; algo=algo, max_time=max_time, noise_weight=0.0, opt_steps=repeat([1], 28), opt_seed=1,  latent_dim=2)

		push!(cost_res, [res_invunit.total_cost, res_flow.total_cost, res_id_stair.total_cost, res_id_full.total_cost])
	end

	cost_res
end

# ╔═╡ Cell order:
# ╠═deaa37a2-e572-11f0-1d8d-d3dddde17a2e
# ╠═70fe22ad-d8dc-4776-ba3b-a50399325484
# ╠═9a1f6973-1363-4207-8a8f-713a47d253ce
# ╠═d0fc8b98-37ca-4a26-8289-2b408846f7b6
# ╠═f16a452c-fc3c-427d-8e43-bae5db10dddd
# ╠═d7873320-cce3-478b-8fa7-93b1063eb0c4
# ╠═3b6f10e2-41c2-4b5a-9e63-5c15b0f8e5a1
# ╠═5a9f6a0a-9a6a-4d21-ae4b-67d122a2b2b3
# ╠═fdbb52af-2e55-4a09-9b9b-7a0bdb9f4c67
# ╠═4d7c7db1-9ed8-4c98-9d79-1c0a1a4fa07e
# ╠═580a6ae6-5f95-4ec2-ab9d-b5f6d48d330b
# ╠═730088b2-08f0-400b-98f2-5298ab5b9eb5
# ╠═39a52a27-dd19-446e-9f43-520b170bac8c

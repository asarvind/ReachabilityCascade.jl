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
		import NLopt, LazySets, JLD2
		using Flux
		import LazySets: center, radius_hyperrectangle
		import ReachabilityCascade.CarDynamics: discrete_vehicles
		import ReachabilityCascade: DiscreteRandomSystem, InvertibleCoupling
		import ReachabilityCascade.MPC: trajectory, smt_mpc, smt_optimize_latent
		import ReachabilityCascade.TrainingAPI: build, load
		import ReachabilityCascade.InvertibleGame: load_self
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
	# set `w_reject = 0.0` (keep `w_true = 1.0`).
	w_true = 1.0
	w_reject = 1.0

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

	save_path_self = replace(save_path, "invertiblegame" => "invertiblegame_self")
	load_path_self = save_path_self
	model, ema, losses = build(InvertibleCoupling, it_game;
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

# ╔═╡ d4b1f290-b47b-4c9d-8420-845bcb9bf163
"""
Return SMT safety and goal formulas equivalent to `thiscost`.

Each matrix encodes an OR (rows are affine constraints with bound <= 0);
the vector of matrices is an AND across disjuncts.
"""
function thiscost_smt()
	ds = discrete_vehicles(0.25)
	xc = center(ds.X)
	xr = radius_hyperrectangle(ds.X)
	n = length(xc)

	# Bounds: |x - xc| <= xr  =>  (x - xc - xr <= 0) AND (-x + xc - xr <= 0)
	# Each row is its own matrix so the AND semantics are preserved.
	bounds_rows = Matrix{Float32}[]
	for i in 1:n
		row_hi = zeros(Float32, n + 1)
		row_hi[i] = 1.0f0
		row_hi[end] = -Float32(xc[i] + xr[i])
		push!(bounds_rows, reshape(row_hi, 1, :))

		row_lo = zeros(Float32, n + 1)
		row_lo[i] = -1.0f0
		row_lo[end] = Float32(xc[i] - xr[i])
		push!(bounds_rows, reshape(row_lo, 1, :))
	end

	# Forward collision:
	# min(5 - |x8 - x1|, 2 + x9 - x2) <= 0
	# => OR over: (x8 - x1 - 5 >= 0) OR (x1 - x8 - 5 >= 0) OR (x2 - x9 - 2 >= 0)
	fc = zeros(Float32, 3, n + 1)
	# Encode each >= 0 as <= 0 by multiplying by -1.
	fc[1, 8] = -1.0f0; fc[1, 1] = 1.0f0;  fc[1, end] = 5.0f0
	fc[2, 1] = -1.0f0; fc[2, 8] = 1.0f0;  fc[2, end] = 5.0f0
	fc[3, 2] = -1.0f0; fc[3, 9] = 1.0f0;  fc[3, end] = 2.0f0

	# Oncoming collision:
	# min(5 - |x11 - x1|, 2 + x2 - x12) <= 0
	# => OR over: (x11 - x1 - 5 >= 0) OR (x1 - x11 - 5 >= 0) OR (x12 - x2 - 2 >= 0)
	oc = zeros(Float32, 3, n + 1)
	# Encode each >= 0 as <= 0 by multiplying by -1.
	oc[1, 11] = -1.0f0; oc[1, 1] = 1.0f0;  oc[1, end] = 5.0f0
	oc[2, 1]  = -1.0f0; oc[2, 11] = 1.0f0; oc[2, end] = 5.0f0
	oc[3, 12] = -1.0f0; oc[3, 2] = 1.0f0;  oc[3, end] = 2.0f0

	# Goal (eventual): x8 - x1 + 3 <= 0  =>  x8 - x1 + 3 <= 0
	goal = zeros(Float32, 1, n + 1)
	goal[1, 8] = 1.0f0
	goal[1, 1] = -1.0f0
	goal[1, end] = 3.0f0

	smt_safety = vcat(bounds_rows, [fc, oc])
	smt_goal = [goal]
	return smt_safety, smt_goal
end

# ╔═╡ 730088b2-08f0-400b-98f2-5298ab5b9eb5
let
	ds = discrete_vehicles(0.25; dt=0.01)
	data = JLD2.load("data/car/trajectories.jld2", "data")
	overtake_idx = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 && d.state_trajectory[1, 1] < d.state_trajectory[8, 1] for d in data]
	overtake_data = data[overtake_idx]
	save_game = "data/car/unitinvert/selfInitSeed2000Iter0Epoch45EmaL0U999R1f4Latseed1.jld2"
	model_invunit, _ = load_self(save_game)
	model_base = (x,z) -> z
	smt_safety, smt_goal = thiscost_smt()

	res_file = "data/car/results/Seed2000AlgoBOBYQA.jld2"
	rng = MersenneTwister(2000)

	

	algo = :LN_BOBYQA
	# algo = :LD_SLSQP

	result = []
	count = 0
	max_count = 0

	while count <= max_count
		opt_steps_list = [[21], [20, 11], [7, 7, 7]]
		opt_steps_id = rand(rng, 1:3)
		opt_steps = opt_steps_list[opt_steps_id]
		u_len = 2
		trace = true

		id_sample = rand(rng, 1:length(overtake_data))
		strj, _ = overtake_data[id_sample]
		x = strj[:, 1]

		res_invunit = smt_optimize_latent(ds, model_invunit, x, repeat(zeros(2), length(opt_steps)), opt_steps, smt_safety, smt_goal;
			u_len=u_len,
			algo=algo,
			max_time=Inf,
			max_penalty_evals=500,
			seed=0,
			trace=trace,
		)
	
		init_res = trajectory(ds, model_invunit, x, zeros(Float32, 2*length(opt_steps)), opt_steps)
		init_utrj = init_res.input_trajectory
	
		res_base_long = smt_optimize_latent(ds, model_base, x, init_utrj, ones(Int64, sum(opt_steps)), smt_safety, smt_goal;
			algo=algo,
			max_penalty_evals=500,
			seed=0,
			latent_dim=2,
			trace=trace,
		)		
	
		res_base_short = Inf
	try
		start_idxs = cumsum(vcat(1, opt_steps[1:end-1]))
		start_idxs = min.(start_idxs, size(init_utrj, 2))
		init_z_short = vec(init_utrj[:, start_idxs])
		# init_z_short = zeros(3*length(opt_steps))
		res_base_short = smt_optimize_latent(ds, model_base, x, init_z_short, opt_steps, smt_safety, smt_goal;
			algo=algo,
			max_penalty_evals=500,
			seed=0,
			latent_dim=2,
			trace=trace,
		)	
		catch res_base_short
	end

		v = [res_invunit.evals_to_zero_penalty, res_base_long.evals_to_zero_penalty, res_base_short.evals_to_zero_penalty]

		if minimum(v) < Inf && minimum(v) > 1
			push!(result, v)
			count += 1
		end
	end

	if !(isempty(res_file)) && max_count >= 100
		JLD2.save(res_file, "result", result)
	end

	result
end

# ╔═╡ d1a37014-5d08-41a1-a17c-de2a4339d738
let
	v = JLD2.load("data/car/results/Seed2000AlgoBOBYQA.jld2", "result")
	sort([r[3] - r[1] for r in v])
end

# ╔═╡ Cell order:
# ╠═deaa37a2-e572-11f0-1d8d-d3dddde17a2e
# ╠═70fe22ad-d8dc-4776-ba3b-a50399325484
# ╠═d0fc8b98-37ca-4a26-8289-2b408846f7b6
# ╠═f16a452c-fc3c-427d-8e43-bae5db10dddd
# ╠═d7873320-cce3-478b-8fa7-93b1063eb0c4
# ╠═d4b1f290-b47b-4c9d-8420-845bcb9bf163
# ╠═730088b2-08f0-400b-98f2-5298ab5b9eb5
# ╠═d1a37014-5d08-41a1-a17c-de2a4339d738

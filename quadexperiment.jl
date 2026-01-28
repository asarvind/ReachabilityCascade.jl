### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ 25c04dac-f9a7-11f0-342e-7bbd0425248c
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

# ╔═╡ f40fd1c0-49c1-4488-85da-b1ef3c61d58a
	begin
		
		using Random, LinearAlgebra
		import JLD2, LazySets, NLopt
		using Flux
		import LazySets: center, radius_hyperrectangle
		import ReachabilityCascade.QuadDynamics: discrete_quadcopter_ref
		import ReachabilityCascade: DiscreteRandomSystem, InvertibleCoupling, NormalizingFlow
		import ReachabilityCascade.InvertibleGame: inclusion_losses, decode, load_self
		import ReachabilityCascade.MPC: trajectory, optimize_latent, mpc, smt_milp_iterative, smt_milp_receding, smt_cmaes, smt_mpc, smt_optimize_latent
	import ReachabilityCascade.TrainingAPI: build, load
	end

# ╔═╡ 8f2f7a2a-7c4b-4a5f-9f50-0a7a7c1f4d67
begin
	"""
	    quad_smt_formulas(ds; kwargs...) -> (smt_safety, smt_goal)

	Build SMT safety and goal formulas for the quadcopter.

	Safety constraints apply only to angles and angular rates using the bounds
	in `ds.X`. Goal constraints bound the full state around zero with tunable
	position, velocity, angle, and angular-rate limits. Angle and rate rows are
	scaled by `angle_scale` for higher weight.
	"""
	function quad_smt_formulas(ds::DiscreteRandomSystem;
	                           pos_bound::Real=1.0,
	                           vel_bound::Real=0.5,
	                           angle_bound::Real=0.1,
	                           rate_bound::Real=0.1,
	                           angle_scale::Real=5.0)
		hasproperty(ds, :X) || throw(ArgumentError("ds must have field X to infer state bounds"))
		X = getproperty(ds, :X)
		X isa LazySets.Hyperrectangle || throw(ArgumentError("ds.X must be a Hyperrectangle to infer bounds"))
		x_center = center(X)
		x_radius = radius_hyperrectangle(X)
		x_lo = x_center .- x_radius
		x_hi = x_center .+ x_radius

		smt_safety = Matrix{Float32}[]
		for idx in 7:12
			row_hi = zeros(Float32, 13)
			row_hi[idx] = 1.0f0
			row_hi[end] = -Float32(x_hi[idx])
			push!(smt_safety, reshape(row_hi, 1, :))

			row_lo = zeros(Float32, 13)
			row_lo[idx] = -1.0f0
			row_lo[end] = Float32(x_lo[idx])
			push!(smt_safety, reshape(row_lo, 1, :))
		end

		# Goal box is an AND across all inequalities; each row is its own matrix.
		smt_goal = Matrix{Float32}[]
		for (idx, bound) in ((1, pos_bound), (2, pos_bound), (3, pos_bound),
		                     (4, vel_bound), (5, vel_bound), (6, vel_bound),
		                     (7, angle_bound), (8, angle_bound), (9, angle_bound),
		                     (10, rate_bound), (11, rate_bound), (12, rate_bound))
			scale = idx >= 7 ? Float32(angle_scale) : 1.0f0

			row_hi = zeros(Float32, 13)
			row_hi[idx] = scale
			row_hi[end] = -scale * Float32(bound)
			push!(smt_goal, reshape(row_hi, 1, :))

			row_lo = zeros(Float32, 13)
			row_lo[idx] = -scale
			row_lo[end] = -scale * Float32(bound)
			push!(smt_goal, reshape(row_lo, 1, :))
		end

		return smt_safety, smt_goal
	end
end
	

# ╔═╡ a020dd28-d0f8-47a7-b274-5f920d2338a7
let
	ds = discrete_quadcopter_ref(dt=0.01, t=0.05)
	smt_safety, smt_goal = quad_smt_formulas(ds)

	model_base = (x, z) -> z
	opt_steps = ones(Int64, 50)
	z0 = repeat(zeros(3), length(opt_steps))

	# save_path = joinpath(pwd(), "data", "quadcopter", "quadtrajectories.jld2")
	save_path = ""
	max_samples = 0
	seed = 0
	rng = Random.MersenneTwister(seed)
	results = NamedTuple[]

	if max_samples != 0
		for sample_id in 1:max_samples
			x = LazySets.sample(ds.X; rng=rng)
			x[1:6] .= 0.0
			x[3] = rand(rng, -2.0:0.5:2.0)
			x[[9, 12]] .= 0.0

			res = smt_optimize_latent(ds, model_base, x, z0, opt_steps, smt_safety, smt_goal;
				algo=:GN_CMAES,
				max_penalty_evals=200,
				seed=seed + sample_id,
				latent_dim=3
			)

			if isfinite(res.eventual_time_safe)
				traj = trajectory(ds, model_base, x, res.z, opt_steps;
					latent_dim=3,
					output_map=identity,
				)
				t_cut = Int(res.eventual_time_safe)
				state_traj = traj.output_trajectory[:, 1:t_cut]
				input_traj = traj.input_trajectory[:, 1:(t_cut - 1)]
				push!(results, (
					state_trajectory=state_traj,
					input_signal=input_traj,
				))
			end
		end
	end

	if max_samples != 0
		if isempty(save_path)
			# skip saving
		elseif isempty(results)
			@warn "No trajectories found; skipping save."
		else
			mkpath(dirname(save_path))
			JLD2.save(save_path, "trajectories", results)
		end
	end

	results
end

# ╔═╡ 5c5d6d1a-8c7a-4a5f-8e23-2d3f7b7e9a61
begin
	"""
	    QuadFlowIterator(data; idxs=1:length(data), rng=Random.default_rng(),
	                     shuffle_each_epoch=true)

	Iterator for training networks from the quadcopter trajectory dataset.

	This iterator assumes `data` is an indexable collection whose elements provide:
	- `sample.state_trajectory`: a `state_dim × (T+1)` (or `state_dim × T`) array-like object.
	- `sample.input_signal`: an `input_dim × T` array-like object.

	On each iteration, we return a named tuple:
	`(; context, sample)` where:
	- `context` is the state vector `state_trajectory[:, t]`
	- `sample` is the control vector `input_signal[:, t]`

	Notes
	- This iterator is *re-iterable*: `Base.iterate(it)` always starts a fresh epoch.
	- If `shuffle_each_epoch=true`, trajectory order is reshuffled at the start of each epoch using `rng`.
	"""
	struct QuadFlowIterator
		data
		idxs::Vector{Int}
		rng
		shuffle_each_epoch::Bool
	end

	function QuadFlowIterator(data;
	                          idxs=1:length(data),
	                          rng=Random.default_rng(),
	                          shuffle_each_epoch::Bool=true)
		return QuadFlowIterator(data, collect(Int, idxs), rng, shuffle_each_epoch)
	end
end

# ╔═╡ 9d59d5f3-2edb-4f1d-9c83-0a37b76f1b9c
begin
	function Base.iterate(it::QuadFlowIterator, state=nothing)
		# State holds:
		# - traj_pos: position in `it.idxs`
		# - t: time index within the current trajectory
		# - X, U, T: cached arrays for the current trajectory to avoid re-loading every step
		if state === nothing
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

# ╔═╡ 1c2a3c8b-5a48-4df5-8c2f-1c6299c2c9d3
let
	# Dataset format: vector of named tuples with `state_trajectory` and `input_signal`.
	data = JLD2.load("data/quadcopter/quadtrajectories.jld2", "trajectories")
	it_quad = QuadFlowIterator(data; rng=Random.MersenneTwister(0))

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
	deterministic_perms = false
	dim = size(it_quad.data[1].input_signal, 1)
	perms = deterministic_perms ? [collect(1:dim) for _ in 1:size(spec, 2)] : nothing

	# Quick switch for "inclusion-only" training:
	# set `w_reject = 0.0` (keep `w_true = 1.0`).
	w_true = 1.0
	w_reject = 1.0

	# ----------------------------
	# Training args (edit these)
	# ----------------------------
	# epochs = 45
	epochs = 0
	batch_size = 200
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
	save_path = "data/quadcopter/temp/selfinvertible.jld2"
	load_path = save_path
	save_period = 60.0

	model, ema, losses = build(InvertibleCoupling, it_quad;
		spec=spec,
		logscale_clamp=logscale_clamp,
		perms=perms,
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
		save_path=save_path,
		load_path=load_path,
		save_period=save_period,
		rng=Random.MersenneTwister(1),          # latent sampling RNG
		rng_model=Random.MersenneTwister(2), # model init RNG
	)
	(; losses, model, ema)
end

# ╔═╡ 881c409b-ff14-46bd-9ec9-0ef123dda852
let
	ds = discrete_quadcopter_ref(dt=0.01, t=0.05)
	smt_safety, smt_goal = quad_smt_formulas(ds)

	savefile = "data/quadcopter/unitinvert/selfInitSeed2Iter0Epoch45EmaL0U999R1f4Latseed1.jld2"
	# savefile = "data/quadcopter/temp/selfinvertible.jld2"
	model_invunit, _ = load_self(savefile)
	model_base = (x, z) -> z

	x = LazySets.sample(ds.X)
	x[1:6] .= 0.0
	x[3] = rand(-2.0:0.5:2.0)
	x[[9, 12]] .= 0.0

	opt_steps_1 = [50]
	opt_steps_2 = [25, 25]
	opt_steps_3 = [17, 17, 16]
	opt_steps_list = [opt_steps_1, opt_steps_2, opt_steps_3]

	opt_steps_id = rand(1:3)
	# opt_steps_id = 2
	opt_steps = opt_steps_list[opt_steps_id]

	algo = :LN_BOBYQA
	# algo = :LD_SLSQP

	init_res = trajectory(ds, model_invunit, x, zeros(Float32, 3*length(opt_steps)), opt_steps)
	init_utrj = init_res.input_trajectory

	res_invunit = smt_optimize_latent(ds, model_invunit, x, zeros(3*length(opt_steps)), opt_steps, smt_safety, smt_goal;
		algo=algo,
		max_penalty_evals=200,
		seed=0,
		latent_dim=3
	)	

	res_base_long = smt_optimize_latent(ds, model_base, x, init_utrj, ones(Int64, sum(opt_steps)), smt_safety, smt_goal;
		algo=algo,
		max_penalty_evals=200,
		seed=0,
		latent_dim=3
	)		

	res_base_short = Inf
	try
		start_idxs = cumsum(vcat(1, opt_steps[1:end-1]))
		start_idxs = min.(start_idxs, size(init_utrj, 2))
		init_z_short = vec(init_utrj[:, start_idxs])
		# init_z_short = zeros(3*length(opt_steps))
		res_base_short = smt_optimize_latent(ds, model_base, x, init_z_short, opt_steps, smt_safety, smt_goal;
			algo=algo,
			max_penalty_evals=200,
			seed=0,
			latent_dim=3
		)	
		catch res_base_short
	end

	res_invunit, res_base_long, res_base_short, opt_steps
end

# ╔═╡ 586f0b4f-f224-461e-bf5c-833a71771343
let
	ds = discrete_quadcopter_ref(dt=0.01, t=0.05)
	smt_safety, smt_goal = quad_smt_formulas(ds)
	data = JLD2.load("data/quadcopter/quadtrajectories.jld2", "trajectories")
	save_game = "data/quadcopter/unitinvert/selfInitSeed2000Iter0Epoch45EmaL0U999R1f4Latseed1.jld2"
	model_invunit, _ = load_self(save_game)
	model_base = (x,z) -> z

	res_file = "data/quadcopter/results/Seed2000AlgoSLSQP.jld2"
	rng = MersenneTwister(2000)

	algo = :LN_BOBYQA
	# algo = :LD_SLSQP

	result = []
	count = 0
	max_count = 100

	while count <= max_count
		opt_steps_list = [[50], [25, 25], [17, 17, 16]]
		opt_steps_id = rand(rng, 1:3)
		opt_steps = opt_steps_list[opt_steps_id]
		u_len = 3
		trace = true

		x = LazySets.sample(ds.X; rng=rng)
		x[1:6] .= 0.0
		x[3] = rand(-2.0:0.5:2.0)
		x[[9, 12]] .= 0.0	
		

		res_invunit = Inf
		try
			res_invunit = smt_optimize_latent(ds, model_invunit, x, repeat(zeros(3), length(opt_steps)), opt_steps, smt_safety, smt_goal;
			u_len=u_len,
			algo=algo,
			max_time=Inf,
			max_penalty_evals=500,
			seed=0,
			trace=trace,
		)
		catch res_invunit
		end
		res_invunit isa NamedTuple || continue
	
		init_res = trajectory(ds, model_invunit, x, zeros(Float32, 3*length(opt_steps)), opt_steps)
		init_utrj = init_res.input_trajectory
	
		res_base_long = Inf
		try
		res_base_long = smt_optimize_latent(ds, model_base, x, init_utrj, ones(Int64, sum(opt_steps)), smt_safety, smt_goal;
			algo=algo,
			max_penalty_evals=500,
			seed=0,
			latent_dim=3,
			trace=trace,
		)		
		catch res_base_long
		end
	
		res_base_short = Inf
	try
		start_idxs = cumsum(vcat(1, opt_steps[1:end-1]))
		start_idxs = min.(start_idxs, size(init_utrj, 2))
		# init_z_short = vec(init_utrj[:, start_idxs])
		init_z_short = zeros(3*length(opt_steps))
		res_base_short = smt_optimize_latent(ds, model_base, x, init_z_short, opt_steps, smt_safety, smt_goal;
			algo=algo,
			max_penalty_evals=500,
			seed=0,
			latent_dim=3,
			trace=trace,
		)	
		catch res_base_short
	end

		res_base_long_penalty = res_base_long isa NamedTuple ? res_base_long.evals_to_zero_penalty : Inf
		res_base_short_penalty = res_base_short isa NamedTuple ? res_base_short.evals_to_zero_penalty : Inf
		v = [res_invunit.evals_to_zero_penalty, res_base_long_penalty, res_base_short_penalty]

		if minimum(v) < Inf && minimum(v) > 1
			push!(result, v)
			count += 1
		end
	end

	if !isempty(res_file) && max_count >= 100
		JLD2.save(res_file, "result", result)
	end

	result
end

# ╔═╡ d5e32a34-69f7-44f3-8daa-cdbc8c6acf5b
let
	v = JLD2.load("data/quadcopter/results/Seed2AlgoBOBYQA.jld2", "result")
	sort([r[2] - r[1] for r in v])
end

# ╔═╡ Cell order:
# ╠═25c04dac-f9a7-11f0-342e-7bbd0425248c
# ╠═f40fd1c0-49c1-4488-85da-b1ef3c61d58a
# ╠═8f2f7a2a-7c4b-4a5f-9f50-0a7a7c1f4d67
# ╠═a020dd28-d0f8-47a7-b274-5f920d2338a7
# ╠═5c5d6d1a-8c7a-4a5f-8e23-2d3f7b7e9a61
# ╠═9d59d5f3-2edb-4f1d-9c83-0a37b76f1b9c
# ╠═1c2a3c8b-5a48-4df5-8c2f-1c6299c2c9d3
# ╠═881c409b-ff14-46bd-9ec9-0ef123dda852
# ╠═586f0b4f-f224-461e-bf5c-833a71771343
# ╠═d5e32a34-69f7-44f3-8daa-cdbc8c6acf5b

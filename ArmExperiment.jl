### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ 32f4a462-f2ea-11f0-16bd-711456f4b53b
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

# ╔═╡ a472418d-ee68-4db4-a40e-a1cf8ae15ac8
	begin
		
		using Random, LinearAlgebra, Flux
		import JLD2, LazySets, NLopt
		import LazySets: center, radius_hyperrectangle
		import ReachabilityCascade.Robot3DOF: discrete_robot3dof, robot3dof_smt_formulas, joint_positions
		import ReachabilityCascade: DiscreteRandomSystem, InvertibleCoupling, NormalizingFlow
		import ReachabilityCascade.InvertibleGame: inclusion_losses, decode, load_self
		import ReachabilityCascade.MPC: trajectory, optimize_latent, mpc, smt_milp_iterative, smt_milp_receding, smt_cmaes, smt_mpc, smt_optimize_latent
		import ReachabilityCascade.TrainingAPI: build, load
	end

# ╔═╡ c0326d4c-de7e-45ca-9570-15b50e359623
let
	# ------------------------------------------------------------
	# Data generation block
	# - Sweeps box positions/velocities on a grid (shuffled order).
	# - Runs CMA-ES for each configuration.
	# - Stores only successful trajectories and input signals.
	# - Periodically saves to disk.
	# ------------------------------------------------------------
	box1_size = 0.5
	box2_size = 1.0
	ds = discrete_robot3dof(; t=0.1, dt=0.1, box_size=box1_size)
	smt_safety, smt_terminal, output_map = robot3dof_smt_formulas(ds; box1_size=box1_size, box2_size=box2_size)
	model = (x::AbstractVector{<:Real}, z::AbstractVector{<:Real}) -> z 
	q0 = zeros(Float32, 3)
	qd0 = zeros(Float32, 3)
	time_max = 40
	steps = repeat([1], time_max)
	zref = zeros(Float32, time_max * 3)

	θ_vals = range(0.0, stop=pi / 2, step=pi / 20)
	r_vals = range(0.0, stop=1.0, step=0.1)
	vy_vals = 0.2:0.1:0.3

	results = Vector{NamedTuple}()
	rng = Random.MersenneTwister(0)
	max_samples = 0
	sample_count = 0
	save_period = 60.0
	seed = 0
	start_iter = 0
	last_save = time()
	last_saved_iter = start_iter
	samples = Vector{NamedTuple}()
	for r1 in r_vals, θ1 in θ_vals
		for r2 in r_vals, θ2 in θ_vals
			for box1_vy in vy_vals, box2_vy in vy_vals
				push!(samples, (
					r1=r1, θ1=θ1, r2=r2, θ2=θ2,
					box1_vy=box1_vy, box2_vy=box2_vy,
				))
			end
		end
	end
	Random.shuffle!(rng, samples)

	for sample in samples
		if max_samples == 0 || sample_count >= max_samples
			break
		end
		if sample_count < start_iter
			sample_count += 1
			continue
		end
		box1_pos = (sample.r1, sample.θ1, sample.r1 * cos(sample.θ1), sample.r1 * sin(sample.θ1))
		box2_pos = (sample.r2, sample.θ2, sample.r2 * cos(sample.θ2), sample.r2 * sin(sample.θ2))
		box1_state = Float32.([box1_pos[3], box1_pos[4], 0.0, sample.box1_vy])
		box2_state = Float32.([box2_pos[3], box2_pos[4], 0.0, sample.box2_vy])
		x0 = vcat(q0, qd0, box1_state, box2_state)

		res = smt_cmaes(ds, model, x0, zref, steps, smt_safety, smt_terminal;
			latent_dim=3,
			output_map=output_map,
			mu=20,
			sigma0=0.5,
			iterations=100,
			rng=rng,
		)

		if res.objective <= 0.0f0
			traj = trajectory(ds, model, x0, res.z, steps;
				latent_dim=3,
				output_map=identity,
			)
			cmaes_iterations = try
				getproperty(res.result, :iterations)
			catch
				missing
			end
			push!(results, (
				state_trajectory=traj.output_trajectory,
				input_signal=traj.input_trajectory,
				box1=(r=box1_pos[1], theta=box1_pos[2], x=box1_pos[3], y=box1_pos[4], vy=sample.box1_vy),
				box2=(r=box2_pos[1], theta=box2_pos[2], x=box2_pos[3], y=box2_pos[4], vy=sample.box2_vy),
				cmaes_iterations=cmaes_iterations,
				objective=res.objective,
			))
		end

		sample_count += 1

		if max_samples != 0 && time() - last_save >= save_period
			save_path = joinpath(pwd(), "data", "robotarm", "trajectories.jld2")
			if !isempty(results)
				mkpath(dirname(save_path))
				JLD2.save(save_path,
					"trajectories", results,
					"seed", seed,
					"last_saved_iter", sample_count,
				)
			end
			last_save = time()
			last_saved_iter = sample_count
		end
	end

	if max_samples != 0
		if isempty(results)
			@warn "No trajectories found; skipping save."
		else
			save_path = joinpath(pwd(), "data", "robotarm", "trajectories.jld2")
			mkpath(dirname(save_path))
			JLD2.save(save_path,
				"trajectories", results,
				"seed", seed,
				"last_saved_iter", sample_count,
			)
		end
	end

	results
end

# ╔═╡ 4a6f2c43-8ef8-4d6f-9b6a-1b5f1a2b46e8
begin
	"""
	    RobotArmIterator(data; idxs=1:length(data), rng=Random.default_rng(),
	                     shuffle_each_epoch=true)

	Iterator for training networks from the robot-arm trajectory dataset.

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
	struct RobotArmIterator
		data
		idxs::Vector{Int}
		rng
		shuffle_each_epoch::Bool
	end

	function RobotArmIterator(data;
	                          idxs=1:length(data),
	                          rng=Random.default_rng(),
	                          shuffle_each_epoch::Bool=true)
		return RobotArmIterator(data, collect(Int, idxs), rng, shuffle_each_epoch)
	end
end

# ╔═╡ b2c91e2a-8d43-4d1b-9c7a-6c2e2a5b6e24
begin
	function Base.iterate(it::RobotArmIterator, state=nothing)
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

# ╔═╡ 6f5bdf15-1d8c-4a8c-8c7d-05f51a4b96b8
let
	# Dataset format: vector of named tuples with `state_trajectory` and `input_signal`.
	data = JLD2.load("data/robotarm/armtrajectories.jld2", "trajectories")
	it_arm = RobotArmIterator(data; rng=Random.MersenneTwister(0))

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
	dim = size(it_arm.data[1].input_signal, 1)
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
	batch_size = 100
	opt = Flux.OptimiserChain(Flux.ClipGrad(), Flux.ClipNorm(), Flux.Adam(1f-5))
	use_memory = true
	# EMA opponent smoothing schedule (used for opponent EMA during gradient computation).
	ema_beta_start = 0.0
	ema_beta_final = 0.999
	ema_tau = 1f5
	# Lower bound on sampled fake latent norm (0 means allow near-zero latents).
	latent_radius_min = 0.0
	# Gradient mode:
	# - :sum            => one gradient on full loss
	# - :orthogonal_adv => split gradients and orthogonalize adversarial component
	grad_mode = :sum
	norm_kind = :l2
	save_path = "data/robotarm/temp/selfinvertible.jld2"
	load_path = save_path
	save_period = 60.0

	model, ema, losses = build(InvertibleCoupling, it_arm;
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
		rng=Random.MersenneTwister(1),        # latent sampling RNG
		rng_model=Random.MersenneTwister(2000), # model init RNG
	)
	(; losses, model, ema)
end

# ╔═╡ a8967670-cb6a-4d82-940a-f2783bda11d1
let
	box1_size = 0.5
	box2_size = 1.0
	ds = discrete_robot3dof(; t=0.1, dt=0.1, box_size=box1_size)
	smt_safety, smt_terminal, output_map = robot3dof_smt_formulas(ds; box1_size=box1_size, box2_size=box2_size)	
	model_unitinv, _ = load_self("data/robotarm/unitinvert/selfInitSeed2000Iter0Epoch45EmaL0U999R1f5Latseed1.jld2")
	# model_unitinv, _ = load_self("data/robotarm/temp/selfinvertible.jld2")
	model_flow = load(NormalizingFlow, "data/robotarm/temp/normalizingflow.jld2")

	data = JLD2.load("data/robotarm/armtrajectories.jld2", "trajectories")
	idtest = rand(1:length(data))
	strj = data[idtest].state_trajectory
	utrj = data[idtest].input_signal
	start_time = 1
	x0 = strj[:, start_time]

	x0[1] = rand(-pi/6:-pi/2*0.1:-pi/2*0.8)
	x0[2] = rand(-pi/6:-pi/2*0.1:-pi/2*0.8)
	x0[3] = rand(-pi/6:-pi/2*0.1:-pi/2*0.8)

	# x0[10] += rand(-0.1:0.2:0.1)
	# x0[7] = 0.9
	# x0[8] = 0.1
	# x0[11] = 0.5
	# x0[12] = 10.0

	u_len = size(utrj, 1)
	

	opt_steps_1 = [40]
	opt_steps_2 = [20, 20]
	opt_steps_3 = [15, 15, 10]
	opt_steps_id = rand(1:3)
	opt_steps_id = 1
	opt_steps_vec = [opt_steps_1, opt_steps_2, opt_steps_3]
	opt_steps = opt_steps_vec[opt_steps_id]
	steps = sum(opt_steps)

	algo = :LN_BOBYQA
	# algo = :LD_SLSQP
	# algo = :GN_CMAES

	model_base = (x, z) -> z

	@time init_res_base_long = smt_mpc(ds, model_flow, x0, 3, smt_safety, smt_terminal;
		u_len=u_len,
		output_map=output_map,
		opt_steps=sum(opt_steps),
		max_time=0.01,
		max_penalty_evals=0,
		opt_seed=0,
		init_z = randn(3),
		latent_dim=3
	)

	x0 = init_res_base_long.state_trajectory[:, end]

	@time res_unitinv = smt_optimize_latent(ds, model_unitinv, x0, repeat(zeros(3), length(opt_steps)), opt_steps, smt_safety, smt_terminal;
		u_len=u_len,
		output_map=output_map,
		algo=algo,
		max_time=Inf,
		max_penalty_evals=1000,
		seed=0,
	)

	@time res_base_long = smt_optimize_latent(ds, model_base, x0, repeat(zeros(3), sum(opt_steps)), repeat([1], sum(opt_steps)), smt_safety, smt_terminal;
		u_len=u_len,
		output_map=output_map,
		algo=algo,
		max_time=Inf,
		max_penalty_evals=1000,
		seed=0,
		latent_dim=3
	)

	@time res_base = smt_optimize_latent(ds, model_base, x0, repeat(zeros(3), length(opt_steps)), opt_steps, smt_safety, smt_terminal;
		u_len=u_len,
		output_map=output_map,
		algo=algo,
		max_time=Inf,
		max_penalty_evals=1000,
		seed=0,
		latent_dim=3
	)

	
	@time res_unitinv_long = smt_optimize_latent(ds, model_unitinv, x0, repeat(zeros(3), sum(opt_steps)), repeat([1], sum(opt_steps)), smt_safety, smt_terminal;
		u_len=u_len,
		output_map=output_map,
		algo=algo,
		max_time=Inf,
		max_penalty_evals=1000,
		seed=0,
	)

	@time res_cmaes = smt_cmaes(ds, model_base, x0, repeat(zeros(3), sum(opt_steps)), repeat([1], sum(opt_steps)), smt_safety, smt_terminal;
		u_len=u_len,
		output_map=output_map,
		latent_dim=3,
		iterations=200,
		rng=Random.MersenneTwister(0),
	)

	(res_unitinv.evals_to_zero_penalty, res_base.evals_to_zero_penalty, res_base_long.evals_to_zero_penalty, res_unitinv_long.evals_to_zero_penalty, res_cmaes), x0
end

# ╔═╡ 222870a9-4ff5-41f9-a863-5ecd9917d598
let
	box1_size = 0.5
	box2_size = 1.0
	ds = discrete_robot3dof(; t=0.1, dt=0.1, box_size=box1_size)
	smt_safety, smt_goal, output_map = robot3dof_smt_formulas(ds; box1_size=box1_size, box2_size=box2_size)	
	data = JLD2.load("data/robotarm/armtrajectories.jld2", "trajectories")

	save_game = "data/robotarm/unitinvert/selfInitSeed2Iter0Epoch45EmaL0U999R1f5Latseed1.jld2"
	model_invunit, _ = load_self(save_game)
	model_base = (x,z) -> z

	res_file = "data/robotarm/results/Seed2AlgoSLSQP.jld2"
	rng = MersenneTwister(2)

	

	# algo = :LN_BOBYQA
	algo = :LD_SLSQP

	result = []
	count = 0
	max_count = 0

	while count <= max_count
		opt_steps_list = [[40], [20, 20], [14, 14, 12]]
		opt_steps_id = rand(rng, 1:3)
		opt_steps = opt_steps_list[opt_steps_id]
		u_len = 3

		id_sample = rand(rng, 1:length(data))
		strj, _ = data[id_sample]
		x = strj[:, 1]

		# x[1] = rand(-pi/2*0.0:-pi/2*0.1:-pi/2*0.2)
		# x[2] = rand(-pi/2*0.0:-pi/2*0.1:-pi/2*0.2)
		# x[3] = rand(-pi/2*0.0:-pi/2*0.1:-pi/2*0.2)

		x[10] += rand(0.1:0.01:0.12)

		res_invunit = smt_optimize_latent(ds, model_invunit, x, repeat(zeros(3), length(opt_steps)), opt_steps, smt_safety, smt_goal;
			u_len=u_len,
			algo=algo,
			max_time=Inf,
			max_penalty_evals=500,
			seed=0,
			output_map=output_map,
		)
	
		init_res = trajectory(ds, model_invunit, x, zeros(Float32, 3*length(opt_steps)), opt_steps)
		init_utrj = init_res.input_trajectory
	
		res_base_long = smt_optimize_latent(ds, model_base, x, init_utrj, ones(Int64, sum(opt_steps)), smt_safety, smt_goal;
			algo=algo,
			max_penalty_evals=500,
			seed=0,
			latent_dim=3,
			output_map=output_map,
		)		
	
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
			output_map=output_map,
		)	
		catch res_base_short
	end

		v = [res_invunit.evals_to_zero_penalty, res_base_long.evals_to_zero_penalty, res_base_short.evals_to_zero_penalty]

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

# ╔═╡ Cell order:
# ╠═32f4a462-f2ea-11f0-16bd-711456f4b53b
# ╠═a472418d-ee68-4db4-a40e-a1cf8ae15ac8
# ╠═c0326d4c-de7e-45ca-9570-15b50e359623
# ╠═4a6f2c43-8ef8-4d6f-9b6a-1b5f1a2b46e8
# ╠═b2c91e2a-8d43-4d1b-9c7a-6c2e2a5b6e24
# ╠═6f5bdf15-1d8c-4a8c-8c7d-05f51a4b96b8
# ╠═a8967670-cb6a-4d82-940a-f2783bda11d1
# ╠═222870a9-4ff5-41f9-a863-5ecd9917d598

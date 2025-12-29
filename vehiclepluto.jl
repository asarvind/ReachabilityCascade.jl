### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ 329b39a0-cc1f-11f0-372b-33fb7f28c501
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

# ╔═╡ 6b799ba4-d33a-4a98-9623-46ad55a186df
begin
	
	using Random
	using Flux
	import JLD2
	import ReachabilityCascade: RefinementRNN, train_perturbation!, build_perturbation, testrun, load
	import ReachabilityCascade.CarDynamics: discrete_vehicles
	import LazySets: center, radius_hyperrectangle
	
end

# ╔═╡ ed3cdc8b-8fec-4901-8376-be99505a027e
function thiscost(x::AbstractMatrix)
	ds = discrete_vehicles(0.25)

	# bounds cost 
	bc = Flux.leakyrelu(abs.(x .- center(ds.X)) .- radius_hyperrectangle(ds.X))

	# forward collision cost
	fc = Flux.leakyrelu(min.(5.0 .- abs.(x[8:8, :] - x[1:1, :]), 2.0 .+ x[9:9, :] - x[2:2, :]))

	# oncoming collision cost
	oc = Flux.leakyrelu(min.(5.0 .- abs.(x[11:11, :] - x[1:1, :]), 2.0 .+ x[2:2, :] - x[12:12, :]))

	# terminal cost
	tc = fill(Flux.leakyrelu(x[8, end] - x[1, end] + 2.0), 1, size(x, 2))
	# tc = vcat(tc, fill(Flux.relu(x[2, end] - 3.0), 1, size(x, 2)))

	return vcat(bc, fc, oc, tc)
end

# ╔═╡ c1f6d4f2-2b1f-4c42-a3e8-5e24c5b9ad33
let
	sd = rand(-10000:10000)
	data = JLD2.load("data/car/trajectories.jld2", "data")
	overtake_idx = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 for d in data]
	overtake_data = data[overtake_idx]

	# ----------------------------
	# System + cost (edit these)
	# ----------------------------
	τ = 0.25
	dt = 0.01
	sys = discrete_vehicles(τ; dt=dt)
	traj_cost_fn = thiscost

	# ----------------------------
	# Build args (edit these)
	# ----------------------------
	latent_dim = 7
	seq_len = 28
	policy_hidden_dim = 128
	policy_depth = 2
	delta_hidden_dim = 128
	delta_depth = 3

	# Delta network construction kwargs (edit these)
	max_seq_len = nothing
	nheads = 1
	activation = Flux.gelu

	# ----------------------------
	# Perturbation training kwargs (edit these)
	# ----------------------------
	steps = 8
	epochs = 1000
	eval_samples = 10
	δ_max = 1f-2
	δ_min = 1f-5

	rng = Random.default_rng(sd)	
	shuffle = true
	start_idx_range = 1:1
	temperature = 0.5
	step_mode = :terminal
	dual = true

	# NOTE: use a new checkpoint path (old `deltanet.jld2` checkpoints were saved with a different model signature)
	save_path = "data/car/temp/refinementrnn.jld2"
	load_path = save_path

	thisdata = repeat(overtake_data[1:1], length(collect(start_idx_range)))

	@time res_train = build_perturbation(
		RefinementRNN,
		thisdata,
		sys,
		traj_cost_fn,
		latent_dim,
		seq_len,
		policy_hidden_dim,
		policy_depth,
		delta_hidden_dim,
		delta_depth;
		max_seq_len=max_seq_len,
		nheads=nheads,
		activation=activation,
		steps=steps,
		temperature=temperature,
		step_mode=step_mode,
		dual=dual,
		epochs=epochs,
		δ_max=δ_max,
		δ_min=δ_min,
		eval_samples=eval_samples,
		rng=rng,
		shuffle=shuffle,
		start_idx_range=start_idx_range,
		save_path = save_path,
		load_path = load_path
	)

	net = res_train.model
	net_before = res_train.model_before
	accept_flags = res_train.accept_flags
	base_losses = res_train.base_losses
	pert_losses = res_train.pert_losses

	start_idx = collect(start_idx_range)[1]
	res_test = testrun(net, thisdata, sys, traj_cost_fn, steps=steps, start_idx=start_idx, temperature=temperature)
	old_res_test = testrun(net_before, thisdata, sys, traj_cost_fn, steps=steps, start_idx=start_idx, temperature=temperature)

	(; net, accept_flags, base_losses, pert_losses)
	res_test, old_res_test
	# x0 = thisdata[1].state_trajectory[:, start_idx]
	# net(x0, sys, traj_cost_fn, steps)
end

# ╔═╡ 622a5bac-db30-4677-a4a4-f1aadab2218c
let
	data = JLD2.load("data/car/trajectories.jld2", "data")
	overtake_idx = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 for d in data]
	overtake_data = data[overtake_idx]

	# ----------------------------
	# System + cost (edit these)
	# ----------------------------
	τ = 0.25
	dt = 0.01
	sys = discrete_vehicles(τ; dt=dt)
	traj_cost_fn = thiscost

	steps = 8
	start_idx = 2
	temperature = 0.5
	
	net = load(RefinementRNN, "data/car/temp/refinementrnn.jld2")

	thisdata = overtake_data[1:1]
	@time res_test = testrun(net, thisdata, sys, traj_cost_fn, steps=steps, start_idx=start_idx, temperature=temperature)
	# x0 = thisdata[1].state_trajectory[:, start_idx]
	# net(x0, sys, traj_cost_fn, steps)
end

# ╔═╡ Cell order:
# ╠═329b39a0-cc1f-11f0-372b-33fb7f28c501
# ╠═6b799ba4-d33a-4a98-9623-46ad55a186df
# ╠═ed3cdc8b-8fec-4901-8376-be99505a027e
# ╠═c1f6d4f2-2b1f-4c42-a3e8-5e24c5b9ad33
# ╠═622a5bac-db30-4677-a4a4-f1aadab2218c

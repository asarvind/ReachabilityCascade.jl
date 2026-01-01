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
	import ReachabilityCascade: ReactiveDenoisingNet
	import ReachabilityCascade.TrainingAPI: build, load
	import ReachabilityCascade.ReactiveDenoisingNetworks: testrun
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

# ╔═╡ DB52797B-F0ED-4ED5-9C7E-AB1531FF1618
begin
	struct CarImitationIterator{D}
		data::D
		idxs::Vector{Int}
		start_idx::Int
	end

	function CarImitationIterator(data;
	                              idxs=1:length(data),
	                              start_idx::Integer=1)
		start_idx_int = Int(start_idx)
		start_idx_int >= 1 || throw(ArgumentError("start_idx must be ≥ 1"))
		return CarImitationIterator(data, collect(Int, idxs), start_idx_int)
	end
end

# ╔═╡ 3B22A351-C4B6-4935-B77B-4B86F32FCF59
begin
	function Base.iterate(it::CarImitationIterator, state::Int=1)
		state > length(it.idxs) && return nothing
		sample = it.data[it.idxs[state]]
		x_full = Array(sample.state_trajectory)
		it.start_idx <= size(x_full, 2) ||
		    throw(ArgumentError("start_idx=$(it.start_idx) exceeds state_trajectory length $(size(x_full, 2))"))
		x0 = Vector(x_full[:, it.start_idx])
		u_target = Matrix(sample.input_signal)
		return ((; x0=x0, u_target=u_target), state + 1)
	end

	Base.length(it::CarImitationIterator) = length(it.idxs)
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
	# Model + training args (edit these)
	# ----------------------------
	hidden_dim = 128
	depth = 3
	nheads = 1
	activation = Flux.gelu

	# ----------------------------
	# Training kwargs (edit these)
	# ----------------------------
	steps = 8
	epochs = 1000
	opt = Flux.Adam(1f-3)
	rng = Random.default_rng(sd)
	start_idx = 1

	save_path = "data/car/temp/reactivedenoisingnet.jld2"
	load_path = save_path

	# DATA SLICE: edit `idxs` to control which samples are used for training.
	# Keep this small while sanity-checking the notebook (the dataset is large).
	it = CarImitationIterator(overtake_data; idxs=1:1, start_idx=start_idx)

	@time model, losses = build(ReactiveDenoisingNet,
	                            it,
	                            sys,
	                            traj_cost_fn;
	                            hidden_dim=hidden_dim,
	                            depth=depth,
	                            nheads=nheads,
	                            activation=activation,
	                            steps=steps,
	                            epochs=epochs,
	                            opt=opt,
	                            rng=rng,
	                            save_path=save_path,
	                            load_path=load_path)

	(; model, losses)
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
	start_idx = 1
	temperature = 0.5
	
	model = load(ReactiveDenoisingNet, "data/car/temp/reactivedenoisingnet.jld2")

	# DATA SLICE: edit the slice here to control how many initial conditions to test.
	x0s = [Vector(Array(d.state_trajectory)[:, start_idx]) for d in overtake_data[1:1]]
	@time results = testrun(model, x0s, sys, traj_cost_fn; steps=steps, temperature=temperature)
	best_costs = [r.best_score for r in results]
	best_us = [r.best_u for r in results]
	best_xs = [r.best_x for r in results]

	(; best_costs, best_us, best_xs)
end

# ╔═╡ Cell order:
# ╠═329b39a0-cc1f-11f0-372b-33fb7f28c501
# ╠═6b799ba4-d33a-4a98-9623-46ad55a186df
# ╠═ed3cdc8b-8fec-4901-8376-be99505a027e
# ╠═DB52797B-F0ED-4ED5-9C7E-AB1531FF1618
# ╠═3B22A351-C4B6-4935-B77B-4B86F32FCF59
# ╠═c1f6d4f2-2b1f-4c42-a3e8-5e24c5b9ad33
# ╠═622a5bac-db30-4677-a4a4-f1aadab2218c

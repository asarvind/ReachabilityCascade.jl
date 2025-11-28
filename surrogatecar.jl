### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ b7ae46fe-ca97-11f0-0ee1-bb1c8d43727c
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

# ╔═╡ 20043f32-8fbd-4370-8994-ad4aa47a5022
begin
	using Random
	import JLD2
	import Flux
	import ReachabilityCascade: TransitionNetwork, train!, build, load_transition_network
	import ReachabilityCascade.CarDataGeneration: discrete_vehicles, generate_transition_dataset
end

# ╔═╡ db3fb671-9238-47fe-bbe1-8ec9de8660ee
save_transitions = "data/car/transitions.jld2"

# ╔═╡ d9c8daaf-e046-49fd-9316-fe55cbc5aad7
let
	ds = discrete_vehicles(0.25)
	# @time thisdata = generate_transition_dataset(ds, save_transitions; iters=1000000)
end

# ╔═╡ 0badd5c9-c0d5-4588-a595-c8ad05e08d33
save_net = "data/car/vehiclenet.jld2"

# ╔═╡ ca016b95-df0f-4b0b-8236-4a5e445d6cd1
save_sprecher = "data/car/vehiclesprecher.jld2"

# ╔═╡ ebce1d30-117c-4105-bec2-9eeeb290cc05
let
	thisdata = JLD2.load(save_transitions, "data")	
	println(size(thisdata))
	scl = Float32.([1.0, 2.0, 10.0, 1.0, 10.0, 10.0, 10.0])
	# scl = Float32.(ones(7))
	opt = Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam(1e-3))
	# @time build(TransitionNetwork, thisdata, save_net; hidden_dim=128, depth=4, batchsize=25, scale=scl, epochs=60, hard_mining=true, opt=opt)
end

# ╔═╡ Cell order:
# ╠═b7ae46fe-ca97-11f0-0ee1-bb1c8d43727c
# ╠═20043f32-8fbd-4370-8994-ad4aa47a5022
# ╠═db3fb671-9238-47fe-bbe1-8ec9de8660ee
# ╠═d9c8daaf-e046-49fd-9316-fe55cbc5aad7
# ╠═0badd5c9-c0d5-4588-a595-c8ad05e08d33
# ╠═ca016b95-df0f-4b0b-8236-4a5e445d6cd1
# ╠═ebce1d30-117c-4105-bec2-9eeeb290cc05

### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ 0d59538e-be23-11f0-204b-63c120efd6e0
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

# ╔═╡ d787ab8b-d52e-4e18-bb75-3b419339a768
begin
	
using Random, LinearAlgebra
import JLD2
import LazySets: Hyperrectangle, low, high, sample
import Flux
import ReachabilityCascade.CarDynamics: safe_discrete_vehicles
using ReachabilityCascade: 	PerturbationGatingNetwork, refinement_gradient, sample_refinement_batch, train!, load_perturbation_gating_network
	
end

# ╔═╡ ba4f94f4-850d-43f2-a18a-bc38c0716d2f
begin
data = JLD2.load("data/car/trajectories.jld2", "data")
ov_idx = [d.state_trajectory[1, end] - d.state_trajectory[8, end] .> 0 for d in data]
overtake_data = data[ov_idx]
end

# ╔═╡ d56f46d5-dcab-4e66-95cf-f9c1ce934103
begin
	struct VehicleDataIter
		data::Vector
		maxiter::Int
	end

	function Base.iterate(iter::VehicleDataIter, state::Int=1)
		limit = min(iter.maxiter, length(iter.data))
		state > limit && return nothing
		sample = iter.data[state]
		context = Float32.(sample.state_trajectory[:, 1])
		target = Float32.(sample.state_trajectory[:, 2:end])
		initial = zeros(Float32, size(target))
		return (context, target, initial), state + 1
	end
end

# ╔═╡ 6dabe420-bbc0-4a3d-bf4e-8c2cae677cf3
save_path = "data/car/temp/pgnet.jld2"

# ╔═╡ 42888273-1054-4f51-bf61-946c09612fcc
let
	seq_dim = 13
	context_dim = 13

	constructor_args = (seq_dim, context_dim)
	
	constructor_kwargs = (
		num_heads=2,
		num_layers=3,
		pos_dim=6,
		max_period=30
	)
	
	pgnet = PerturbationGatingNetwork(constructor_args...; constructor_kwargs...)	
	
	vdi = VehicleDataIter(overtake_data, 1000)	
	
	if isfile(save_path)
		pgnet = load_perturbation_gating_network(save_path)
		rm(save_path)
	end

	tolerance = [1.0, 0.5, 0.1, 1.0, 0.1, 0.1, 0.02, 1.0, 0.1, 0.1, 1.0, 0.1, 0.1]
	
	@time train!(pgnet,
				 vdi;
				 tolerance=tolerance,
				 λ=0.5,
				 refinement_batch=6,
				 constructor_args=constructor_args,
				 constructor_kwargs=constructor_kwargs,
				 save_path="data/car/temp/pgnet.jld2"
				)
end

# ╔═╡ 281275a6-87dc-4594-8941-ef6487cd5de5
let
	pgnet = load_perturbation_gating_network(save_path)
	idx = rand(1:length(overtake_data))
	idx = 1
	tup = overtake_data[idx]
	strj, utrj = tup.state_trajectory, tup.input_signal
	context = strj[:, 1]
	target = strj[:, 2:end]
	

	pgnet(context, target, 100; λ=0.5), target
end

# ╔═╡ Cell order:
# ╠═0d59538e-be23-11f0-204b-63c120efd6e0
# ╠═d787ab8b-d52e-4e18-bb75-3b419339a768
# ╠═ba4f94f4-850d-43f2-a18a-bc38c0716d2f
# ╠═d56f46d5-dcab-4e66-95cf-f9c1ce934103
# ╠═6dabe420-bbc0-4a3d-bf4e-8c2cae677cf3
# ╠═42888273-1054-4f51-bf61-946c09612fcc
# ╠═281275a6-87dc-4594-8941-ef6487cd5de5

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
		import ReachabilityCascade.Robot3DOF: discrete_robot3dof, joint_positions
		import ReachabilityCascade: DiscreteRandomSystem, InvertibleCoupling, NormalizingFlow
		import ReachabilityCascade.InvertibleGame: inclusion_losses, decode, load_self
		import ReachabilityCascade: trajectory, optimize_latent, mpc
		import ReachabilityCascade.TrainingAPI: build
	end

# ╔═╡ c0326d4c-de7e-45ca-9570-15b50e359623
let
	ds = discrete_robot3dof(; t=0.1, dt=0.1)
	model = (x::AbstractVector{<:Real}, z::AbstractVector{<:Real}) -> z 
	x0 = zeros(Float32, 6)
	steps = repeat([1], 20)
	z = rand(Float32, 20*3)
		trajectory(ds, model, x0, z, steps, output_map=joint_positions, latent_dim=3)
end

# ╔═╡ Cell order:
# ╠═32f4a462-f2ea-11f0-16bd-711456f4b53b
# ╠═a472418d-ee68-4db4-a40e-a1cf8ae15ac8
# ╠═c0326d4c-de7e-45ca-9570-15b50e359623

### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ fc3c2db6-7193-11f0-1b65-7f390e18200d
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

# ╔═╡ c8d1f042-ad49-4971-aa32-b9c15d241da6
using ReachabilityCascade, LinearAlgebra, Random, LazySets, Flux, Statistics

# ╔═╡ c6105657-7697-462a-84f0-5b6cb0ffb4b1
import ReachabilityCascade.CarDynamics as CD

# ╔═╡ 87dc46a4-8f00-473a-9e37-817fc91e7ffe
function discrete_car(t::Real)
	X = Hyperrectangle(
		vcat([50, 3.5, 0.0, 10.0], zeros(3)),
		[50, 3.5, 1.0, 10.0, 1.0, 1.0, 0.2]
	)

	U = Hyperrectangle(
		zeros(2), [0.4, 10.0]
	)

	V = Hyperrectangle(
		zeros(2), [1.0, 10.0]
	)

	cs = ContinuousSystem(X, U, CD.carfield)
	
	κ = (x, u, t) -> [u[1], 0.4*(u[2] - x[3])]

	return DiscreteRandomSystem(cs, V, κ, t)	
end

# ╔═╡ 48f1de66-e038-42db-8044-b5f83c86eacf
function discrete_vehicles(t::Real)

	X = Hyperrectangle(
		vcat([50, 3.5, 0.0, 10.0], zeros(3), 50.0, 1.75, 5.0, 50.0, 5.25, -10.0),
		[50, 3.5, 1.0, 10.0, 1.0, 1.0, 0.2, 50.0, 0.1, 1.0, 50.0, 0.1, 1.0]
	)	

	V = Hyperrectangle(
		zeros(2), [1.0, 10.0]
	)

	function vehicle_transition(x::AbstractVector, u::AbstractVector)
		
		ds = discrete_car(t)
		ego_next = ds(x[1:7], u*0.2)
	
		x8next = x[8] + t*x[10]
		x11next = x[11] + t*x[13]
	
		xnext = vcat(ego_next, x8next, x[9:10], x11next, x[12:13])
	
		return xnext
	end

	return DiscreteRandomSystem(X, V, vehicle_transition)

end

# ╔═╡ 3ec5c7b7-cae2-45a8-83ce-2c77993ae9fe
function safety_fn(x::AbstractVector)
	safety_distance = 2.5

	ds = discrete_car(0.25)

	fol_dist = norm(x[1:2] - x[8:9]) - safety_distance
	on_dist = norm(x[1:2] - x[11:12]) - safety_distance

	constraint_dist = 1.0 .- abs.(x[1:7] .- ds.X.center) ./ (ds.X.radius .+ 1e-8)

	return vcat(fol_dist, on_dist, constraint_dist)
end

# ╔═╡ 73693311-fab8-4ff7-8c11-1814c5873283
function target_fn(x::AbstractVector)
	return [x[1] - x[8]]
end

# ╔═╡ 52218b79-4396-45f6-a06d-c0e8e30e7cb0
function intertarget_fn(x::AbstractVector)
	return x[1:7]
end

# ╔═╡ 127ba1c6-58c5-4609-b354-4b87abe35452
let
	state_dim, batch = 6, 8
	x  = randn(state_dim, batch)
	x0 = randn(state_dim, batch)
	t  = randn(batch)
	model = NRLE(state_dim; depth=4, width=256)
	z  = model(x, x0, t)                      # forward encoding (z)
	ll = loglikelihood(model, x, x0, t)       # log-likelihoods
	xr = inverse(model, z, x0, t)             # reconstruction (x)
	xr - x
end

# ╔═╡ Cell order:
# ╠═fc3c2db6-7193-11f0-1b65-7f390e18200d
# ╠═c8d1f042-ad49-4971-aa32-b9c15d241da6
# ╠═c6105657-7697-462a-84f0-5b6cb0ffb4b1
# ╠═87dc46a4-8f00-473a-9e37-817fc91e7ffe
# ╠═48f1de66-e038-42db-8044-b5f83c86eacf
# ╠═3ec5c7b7-cae2-45a8-83ce-2c77993ae9fe
# ╠═73693311-fab8-4ff7-8c11-1814c5873283
# ╠═52218b79-4396-45f6-a06d-c0e8e30e7cb0
# ╠═127ba1c6-58c5-4609-b354-4b87abe35452

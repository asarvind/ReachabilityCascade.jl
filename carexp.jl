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
using ReachabilityCascade, LinearAlgebra, Random, LazySets, Flux

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

# ╔═╡ 9abc6845-e73b-41c0-ab1e-0d0cfcb04b44
"""
    add_diffusion_noise(x::AbstractVector, num_steps::Int; beta::Float64 = 0.02) -> Matrix

Applies a stable-diffusion-like noise process to the input vector `x`.
Returns a matrix whose columns represent the progressively noised versions of `x` across `num_steps`.

# Arguments
- `x::AbstractVector`: The input vector to be noised.
- `num_steps::Int`: Number of diffusion steps.
- `beta::Float64`: Constant noise variance applied at each step (default = 0.02).

# Returns
- `Matrix`: A matrix of shape `(length(x), num_steps)` where each column is the noised vector at a diffusion step.
"""
function add_diffusion_noise(x::AbstractVector, num_steps::Int; beta::Float64 = 0.02)
    x = Float32.(x)
    alpha = 1 - beta

    D = length(x)
    noise_matrix = Matrix{Float32}(undef, D, num_steps)

    xt = x
    for t in 1:num_steps
        noise = randn(Float32, D)
        xt = sqrt(alpha) * xt .+ sqrt(beta) * noise
        noise_matrix[:, t] = xt
    end

    return noise_matrix
end

# ╔═╡ 0f6cc562-13a2-445f-8bc8-4f3be45d6ecc
function vehicle_transition(x::AbstractVector, u::AbstractVector)
	t = 0.25 
	
	ds = discrete_car(t)
	ego_next = ds(x[1:7], u)

	x8next = x[8] + t*x[10]
	x11next = x[11] + t*x[13]

	xnext = vcat(ego_next, x8next, x[9:10], x11next, x[12:13])

	safety_distance = 2.5

	cur_fol_dist = norm(x[1:2] - x[8:9]) - safety_distance
	cur_on_dist = norm(x[1:2] - x[11:12]) - safety_distance
	next_fol_dist = norm(xnext[1:2] - xnext[8:9]) - safety_distance
	next_on_dist = norm(xnext[1:2] - xnext[11:12]) - safety_distance

	fol_dist = min(cur_fol_dist, next_fol_dist)
	on_dist = min(cur_on_dist, next_on_dist)

	cur_cons_dist = 1.0 .- (x[1:7] .- ds.X.center) ./ (ds.X.radius .+ 1e-8)
	next_cons_dist = 1.0 .- (xnext[1:7] .- ds.X.center) ./ (ds.X.radius .+ 1e-8)
	cons_dist = min(cur_cons_dist, next_cons_dist)

	return xnext, vcat(fol_dist, on_dist, cons_dist)
end

# ╔═╡ Cell order:
# ╠═fc3c2db6-7193-11f0-1b65-7f390e18200d
# ╠═c8d1f042-ad49-4971-aa32-b9c15d241da6
# ╠═c6105657-7697-462a-84f0-5b6cb0ffb4b1
# ╠═87dc46a4-8f00-473a-9e37-817fc91e7ffe
# ╠═9abc6845-e73b-41c0-ab1e-0d0cfcb04b44
# ╠═0f6cc562-13a2-445f-8bc8-4f3be45d6ecc

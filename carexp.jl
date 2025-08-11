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
using ReachabilityCascade, LinearAlgebra, Random, LazySets, Flux, Statistics, NLopt

# ╔═╡ d7cb02f0-06b3-4e8e-85b0-302a0c3f7ca0
# L-BFGS optimizer in pure Julia (no external deps)
# -------------------------------------------------
# Usage
# ------
# Define an objective that can supply value and gradient via `fg!(g, x)`:
#   using .SimpleLBFGS
#   function fg!(g, x)
#       g .= 2 .* x
#       return sum(abs2, x)
#   end
#   x0 = randn(5)
#   res = lbfgs(fg!, x0; m=10, maxiter=200, gtol=1e-8, verbose=true)
#   @show res.x, res.f, res.converged, res.reason
#
# Alternatively, if you already have separate `f(x)` and `grad!(g,x)`,
# use the convenience wrapper:
#   res = lbfgs(x -> sum(abs2, x), (g,x)->(g .= 2x), x0)
#
# This implementation uses a strong-Wolfe line search and the standard
# two-loop recursion to apply the L-BFGS inverse Hessian.

module SimpleLBFGS

using LinearAlgebra
using Printf

export lbfgs, LBFGSResult

# --------------------------- Result type ------------------------------------
Base.@kwdef mutable struct LBFGSResult{T}
    x::Vector{T}
    f::T
    gnorm::T
    iters::Int
    f_evals::Int
    g_evals::Int
    converged::Bool
    reason::String
end

# ----------------------- Strong-Wolfe line search ---------------------------
"""
    strong_wolfe_line_search(fg!, x, f0, g0, p; c1=1e-4, c2=0.9, alpha0=1.0, maxiter=20)

Perform a strong-Wolfe line search along direction `p` from `x`.
`fg!(g, x)` should return `f(x)` and write the gradient into `g`.
Returns `(alpha, f_new, g_new, fevals, gevals)`.
"""
function strong_wolfe_line_search(fg!, x::AbstractVector, f0::Real, g0::AbstractVector,
                                  p::AbstractVector;
                                  c1=1e-4, c2=0.9, alpha0=1.0, maxiter=20)
    @assert c1 > 0 && c1 < c2 < 1 "Require 0 < c1 < c2 < 1"
    @assert dot(p, g0) < 0 "Search direction must be a descent direction"

    gtmp = similar(g0)
    xtrial = similar(x)
    function eval_at!(α)
        @inbounds @. xtrial = x + α * p
        f = fg!(gtmp, xtrial)
        return f, dot(gtmp, p)
    end

    ϕ0 = f0
    dϕ0 = dot(g0, p)

    α_lo, ϕ_lo, dϕ_lo = 0.0, ϕ0, dϕ0
    α_hi = NaN; ϕ_hi = NaN

    α = alpha0
    fevals = 0; gevals = 0

    for _ in 1:maxiter
        ϕ, dϕ = eval_at!(α); fevals += 1; gevals += 1

        if (ϕ > ϕ0 + c1*α*dϕ0) || (!isnan(ϕ_hi) && ϕ ≥ ϕ_lo)
            return zoom!(eval_at!, ϕ0, dϕ0, α_lo, ϕ_lo, dϕ_lo, α, ϕ; c1, c2,
                         gtmp, fevals, gevals)
        end

        if abs(dϕ) ≤ -c2*dϕ0
            return α, ϕ, copy(gtmp), fevals, gevals
        end

        if dϕ ≥ 0
            return zoom!(eval_at!, ϕ0, dϕ0, α, ϕ, dϕ, α_lo, ϕ_lo; c1, c2,
                         gtmp, fevals, gevals)
        end

        α_lo, ϕ_lo, dϕ_lo = α, ϕ, dϕ
        α = 2α
        α_hi = α # keep track of an upper bound (loose)
    end

    # Fallback: last eval
    ϕ, _ = eval_at!(α); fevals += 1; gevals += 1
    return α, ϕ, copy(gtmp), fevals, gevals
end

function zoom!(eval_at!, ϕ0, dϕ0, α_lo, ϕ_lo, dϕ_lo, α_hi, ϕ_hi; c1=1e-4, c2=0.9,
               gtmp=nothing, fevals=0, gevals=0, maxiter=25)
    for _ in 1:maxiter
        α = 0.5*(α_lo + α_hi)
        ϕ, dϕ = eval_at!(α); fevals += 1; gevals += 1
        if (ϕ > ϕ0 + c1*α*dϕ0) || (ϕ ≥ ϕ_lo)
            α_hi, ϕ_hi = α, ϕ
        else
            if abs(dϕ) ≤ -c2*dϕ0
                return α, ϕ, copy(gtmp), fevals, gevals
            end
            if dϕ*(α_hi - α_lo) ≥ 0
                α_hi, ϕ_hi = α_lo, ϕ_lo
            end
            α_lo, ϕ_lo, dϕ_lo = α, ϕ, dϕ
        end
    end
    return α_lo, ϕ_lo, copy(gtmp), fevals, gevals
end

# ------------------------- Two-loop recursion -------------------------------
function apply_lbfgs!(q, gk, S, Y, ρ, αtmp)
    copy!(q, gk)
    k = length(ρ)
    for i in reverse(1:k)
        αtmp[i] = ρ[i] * dot(S[i], q)
        @. q = q - αtmp[i] * Y[i]
    end
    if k > 0
        sy = dot(S[k], Y[k])
        yy = dot(Y[k], Y[k])
        γ = sy / yy
        @. q = γ * q
    end
    for i in 1:k
        β = ρ[i] * dot(Y[i], q)
        @. q = q + (αtmp[i] - β) * S[i]
    end
    return q
end

# ----------------------------- Main solver ---------------------------------
"""
    lbfgs(fg!, x0; m=10, maxiter=1000, gtol=1e-6, ftol=1e-12, xtol=1e-12,
          c1=1e-4, c2=0.9, alpha0=1.0, verbose=false)

Run L-BFGS starting at `x0` for objective provided by `fg!(g, x)` which must
return `f(x)` and write the gradient into `g`.

Keyword arguments:
- `m`: memory (number of correction pairs to keep)
- `maxiter`: maximum iterations
- `gtol`: stop if `norm(g) ≤ gtol * max(1, norm(x))`
- `ftol`: stop if relative decrease in `f` is below this threshold
- `xtol`: stop if step size is extremely small
- `c1`, `c2`: strong-Wolfe parameters
- `alpha0`: initial trial step for the line search
- `verbose`: print iteration log

Returns `LBFGSResult`.
"""
function lbfgs(fg!::Function, x0::AbstractVector;
               m::Int=10, maxiter::Int=1000, gtol::Real=1e-6,
               ftol::Real=1e-12, xtol::Real=1e-12,
               c1::Real=1e-4, c2::Real=0.9, alpha0::Real=1.0,
               verbose::Bool=false)

    T = eltype(x0)
    x = copy(Vector{T}(x0))
    g = similar(x)
    f = fg!(g, x)

    f_evals = 1
    g_evals = 1

    # L-BFGS storage
    S = Vector{Vector{T}}()
    Y = Vector{Vector{T}}()
    ρ = T[]

    # temporaries
    p = similar(x)
    q = similar(x)
    αtmp = T[]

    gnorm = norm(g)
    f_old = f

    if verbose
        println(rpad("iter",6), rpad("f(x)",16), rpad("‖g‖",12), rpad("α",10), "#pairs")
        @printf("%5d  %-14.6e %-10.3e %-8s %d\n", 0, f, gnorm, "-", 0)
    end

    for k in 1:maxiter
        if gnorm ≤ gtol * max(1.0, norm(x))
            return LBFGSResult(x, f, gnorm, k-1, f_evals, g_evals, true, "first-order optimality")
        end

        resize!(αtmp, length(ρ))
        apply_lbfgs!(q, g, S, Y, ρ, αtmp)
        @. p = -q
        if dot(p, g) ≥ 0
            @. p = -g
        end

        α, f_new, g_new, fe, ge = strong_wolfe_line_search(fg!, x, f, g, p; c1, c2, alpha0)
        f_evals += fe
        g_evals += ge

        x_new = x .+ α .* p
        s = x_new .- x
        y = g_new .- g

        sy = dot(s, y)
        if sy > 1e-10
            push!(S, s); push!(Y, y); push!(ρ, 1/sy)
            if length(S) > m
                popfirst!(S); popfirst!(Y); popfirst!(ρ)
            end
        end

        x = x_new
        g = g_new
        f_old, f = f, f_new
        gnorm = norm(g)

        if verbose
            @printf("%5d  %-14.6e %-10.3e %-8.2e %d\n", k, f, gnorm, α, length(S))
        end

        if abs(f_old - f) ≤ ftol * max(1.0, abs(f_old))
            return LBFGSResult(x, f, gnorm, k, f_evals, g_evals, true, "small relative decrease in f")
        end
        if norm(s) ≤ xtol * max(1.0, norm(x))
            return LBFGSResult(x, f, gnorm, k, f_evals, g_evals, true, "step size too small")
        end
    end

    return LBFGSResult(x, f, gnorm, maxiter, f_evals, g_evals, false, "max iterations reached")
end

# Convenience wrapper: separate f(x) and grad!(g,x)
function lbfgs(f::Function, grad!::Function, x0::AbstractVector; kwargs...)
    fg!(g, x) = (grad!(g, x); f(x))
    return lbfgs(fg!, x0; kwargs...)
end

end # module

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

# ╔═╡ 687648fc-2d2f-4cd4-9ff7-d0524615d70c
function batch(xmat::Matrix{<:Real}; gap=1)

	N = size(xmat, 2)÷2
	@assert N > 0 "Number of columns of array should be at least 2"

	batch = [
		(xmat[:, t+k], xmat[:, t], k) for t in 1:gap:N for k in t:gap:N
	]

	return batch
end

# ╔═╡ 77293f6a-0141-4fa0-81da-5a74e35b2537
function linearize(ds::DiscreteRandomSystem, x::AbstractVector, u::AbstractVector=ds.U.center; δ=0.001)
	n = size(x, 1)
	m = size(u, 1)

	A = ones(n, n)
	B = ones(n, m)

	p = ds(x, u)

	for i in 1:n
		x_pert = copy(x)
		x_pert[i] += δ
		A[:, i] = (ds(x_pert, u) - p)/δ		
	end

	for i in 1:m
		u_pert = copy(u)
		u_pert[i] += δ
		B[:, i] = (ds(x, u_pert) - p)/δ
	end

	c = p - A*x - B*u 

	return A, B, c
end

# ╔═╡ c03edf5d-2bf1-42c2-b4e8-8c107f535389
function quadratic_coeffs(f::Flow, context::AbstractVector, transform::Function, x::AbstractVector; δ::Real = 0.001)
	
	n = size(x, 1)

	A = ones(n, n)
	b = ones(n)

	y, ld = f(transform(x), context)

	for _ in 1:n
		x_pert = copy(x)
		x_pert[i] += δ

		y_pert, ld_pert = f(transform(x_pert), context)

		A[:, i] = (y_pert - y)/δ
		b[i] = -1*(ld_pert - ld)/δ  # -1 multiplication because maximize ld
	end

	return A, b, y, ld
end

# ╔═╡ 8c972af1-eb82-44fa-a28b-ad36e6c11644
begin

function search_optimum(f::Flow, context::AbstractVector, transform::Function, x::AbstractVector, r::Real; δ=0.001)
	A, b, y, ld = quadratic_coeffs(f, transform, context, x0)

	# objective: 0.5||Aϵ+y||^2 + bϵ + r||ϵ||^2
	H = Symmetric(A' * A + (2r) * I)      # n×n
	g = A' * y + b                # b is 1×n (row)
	ε = -H\g

	x_star = x + ϵ

	y_star, ld_star = f(transform(x_star), context)

	α = ld - 0.5*norm(y)^2
	β = ld_star - 0.5*norm(y_star)^2

	return x_star, α, β
end

function search_optimum(f::Flow, context::AbstractVector, transform::Function, x::AbstractVector, r_init::Real=0; δ=0.001, maxreg::Real = 10, maxiter = 1000)
	r = r_init

	xup = x

	iter = 0
	while r < maxreg && iter < maxiter
		x_star, α, β = search_optimum(f, context, transform, x, r, δ=δ)
		if β>α
			xup = x_star
			r /= 2
		else
			r *= 2
		end
	end

	return xup
end


end

# ╔═╡ 7b3216d8-787e-4654-9827-7f303e42abff
let
	ds = discrete_vehicles(0.25)

	x = ds.X.center

	linearize(ds, x)
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
# ╠═687648fc-2d2f-4cd4-9ff7-d0524615d70c
# ╠═77293f6a-0141-4fa0-81da-5a74e35b2537
# ╠═c03edf5d-2bf1-42c2-b4e8-8c107f535389
# ╠═8c972af1-eb82-44fa-a28b-ad36e6c11644
# ╠═7b3216d8-787e-4654-9827-7f303e42abff
# ╠═d7cb02f0-06b3-4e8e-85b0-302a0c3f7ca0

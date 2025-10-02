### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 90646a38-9eb9-11f0-3492-0567c353dada
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

# ╔═╡ b3d77e2d-adc1-43e9-a61a-2ce2a71bf622
begin
	import Flux
	using LinearAlgebra, Random
	import LazySets: Hyperrectangle, high, low, dim, radius_hyperrectangle, center
	import JLD2
	import ReachabilityCascade.CarDynamics: discrete_vehicles	
	import ReachabilityCascade: ConditionalFlow, loglikelihoods
	import ReachabilityCascade.ComplexLiftFlow: LiftingFlow
end

# ╔═╡ a624594d-fe9e-42fa-8cdd-9c6ea365b832
let
	d, ℓ = 8, 12
	ctx_dim = 3
	lf = LiftingFlow(d, ℓ; ctx_dim=ctx_dim, num_blocks=4, hidden=128, depth=2,
	                 lift_hidden=64, lift_depth=2, ln=false, clamp=2.5, seed=0)
	N = 16
	X = randn(ComplexF32, d, N)
	C = randn(Float32, ctx_dim, N)
	Y = lf(X, true)    # encode
	Xrec = lf(Y, false) # decode (projects to first d)
	@assert maximum(abs.(X .- Xrec)) < 1e-4
end

# ╔═╡ 9f8ee7b1-38b9-4704-8a9a-ffdc7d8871de
"""
    growth_rates(xmat; lambda = nothing)

Stable growth-rate ratios for real or complex matrices:

    (x[:,3:end] - x[:,2:end-1]) ./ (x[:,2:end-1] - x[:,1:end-2])

Implemented via ridge-regularized complex division:
    r = (num * conj(den)) / (abs2(den) + λ)

- Works for Real and Complex.
- No mutation (Zygote-compatible).
- Default λ = eps(T) where T is the element's underlying real float type.
"""
function growth_rates(xmat::AbstractMatrix{<:Number}; lambda = nothing)
    @assert size(xmat, 2) >= 3 "Matrix should have at least 3 columns"

    # Underlying real float type (e.g., Float64 for ComplexF64)
    T  = float(real(eltype(xmat)))
    λ  = lambda === nothing ? eps(T) : convert(T, lambda)

    x3 = @view xmat[:, 3:end]
    x2 = @view xmat[:, 2:end-1]
    x1 = @view xmat[:, 1:end-2]

    num = x3 .- x2
    den = x2 .- x1

    # Ridge-regularized, complex-safe "division"
    return (num .* conj.(den)) ./ (abs2.(den) .+ λ)
end

# ╔═╡ ef9e1e16-1864-4349-b382-7467698bde62
"""
    clamp_complex(x; δ=1e-12, β=40.0)

Smoothly cap the magnitude of real/complex `x` at 1 (≈ identity when |x| < 1).
- Uses sqrt(abs2(x)+δ) for a smooth magnitude (finite gradient at 0).
- Uses a softplus-based smoothmax to avoid hard branches:
      denom ≈ 1            when |x| <= 1
      denom ≈ |x|          when |x| >> 1
- No mutation; Zygote-friendly.

Tuning:
- Increase β to make the transition sharper (closer to hard clamp).
- Increase δ slightly if you still see gradient noise at exactly 0.
"""
function clamp_complex(x; δ=1e-12, β=40.0)
    mag  = sqrt.(abs2.(x) .+ δ)                       # smooth |x|
    denom = 1 .+ (log1p.(exp.(β .* (mag .- 1))) ./ β) # smooth max(mag, 1)
    x ./ denom
end

# ╔═╡ de605775-6476-4caf-8bcf-e62e6414370a
function growth_loss(lf::LiftingFlow, strj::AbstractMatrix{<:Number}; kwargs...)
	gr = growth_rates(lf(strj, true))
	target_gr = sum(gr, dims=2) / size(gr, 2) # mean growth rate
	target_gr = clamp_complex(target_gr) # bound mean growth rate

	return sum(abs2.(gr .- target_gr)) / size(gr, 2)
end

# ╔═╡ 75dbde3e-095f-4a92-b3af-92ba1bb2a3a9
function train(::Type{LiftingFlow}, l::Integer, dataargs::AbstractVector...; maxiter::Integer = max(length.(dataargs)...), optimiser = Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam()), kwargs...)
	# construct network
	d = size(dataargs[1][1].state_trajectory, 1)
	lf = LiftingFlow(d, l, kwargs...)

	# initialize optimizer state
	opt_state = Flux.setup(optimiser, lf)

	dataargs_shuffled = shuffle.(dataargs)
	for iter in 1:maxiter
		for data in dataargs
			id = iter % length(data) + 1
			strj = data[id].state_trajectory 

			gr = growth_rates(lf(strj, true))
		
			
			grads = Flux.gradient(lf) do model
				growth_loss(model, strj)
			end

			Flux.update!(opt_state, lf, grads[1])
		end
	end
	
	return lf
end

# ╔═╡ 6dd10992-4028-42c4-b66f-b80a0e4ec5a4
let
	Random.seed!()
	ds = discrete_vehicles(0.25)
	data = JLD2.load("data/car/trajectories.jld2", "data")

	lf = train(LiftingFlow, 14, data; maxiter = 1)
end

# ╔═╡ Cell order:
# ╠═90646a38-9eb9-11f0-3492-0567c353dada
# ╠═b3d77e2d-adc1-43e9-a61a-2ce2a71bf622
# ╠═a624594d-fe9e-42fa-8cdd-9c6ea365b832
# ╠═9f8ee7b1-38b9-4704-8a9a-ffdc7d8871de
# ╠═ef9e1e16-1864-4349-b382-7467698bde62
# ╠═de605775-6476-4caf-8bcf-e62e6414370a
# ╠═75dbde3e-095f-4a92-b3af-92ba1bb2a3a9
# ╠═6dd10992-4028-42c4-b66f-b80a0e4ec5a4

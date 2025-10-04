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
function growth_loss(lf::LiftingFlow, strj::AbstractMatrix{<:Number}; lambda = nothing, δ=1f-12, β=Float32.(40.0))
	gr = growth_rates(lf(Float32.(strj), true); lambda=lambda)
	target_gr = sum(gr, dims=2) / size(gr, 2) # mean growth rate
	target_gr = clamp_complex(target_gr; δ=δ, β=β) # bound mean growth rate
	return sum(abs2.(gr .- target_gr)) / size(gr, 2)
end

# ╔═╡ 75dbde3e-095f-4a92-b3af-92ba1bb2a3a9
function train(::Type{LiftingFlow}, l::Integer, dataargs::AbstractVector...; maxiter::Integer = max(length.(dataargs)...), optimiser = Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam()), savefile::String="", loadfile::String=savefile, lambda=eps(Float32), δ=1e-12, β=40, save_period::Real = 60, kwargs...)
	
	# construct network
	if isfile(loadfile) # if a network is previously stored in loadfile
		args = JLD2.load(loadfile, "args") # positional arguments of network constructor
		this_kwargs = JLD2.load(loadfile, "kwargs") # keyword arguments of network constructor
		model_state = JLD2.load(loadfile, "model_state") # stored parameters of network
		lf = LiftingFlow(args...; this_kwargs...) # construct network
		Flux.loadmodel!(lf, model_state) # load stored parameters into the network
	else
		d = size(dataargs[1][1].state_trajectory, 1) # input dimension of network
		args = (d, l) # position arguments for network constructor
		this_kwargs = kwargs # keyword arguments for network constructor
		lf = LiftingFlow(args...; this_kwargs...) # construct network
	end
	
	# initialize optimizer state
	opt_state = Flux.setup(optimiser, lf)

	# shuffle data
	dataargs_shuffled = shuffle.(dataargs)

	start_time = time() # time counter for periodically saving the network

	strj_max = dataargs_shuffled[1][1].state_trajectory

	# iterative gradient descent
	for iter in 1:maxiter
		for data in dataargs
			id = iter % length(data) + 1 # choose data point
			strj = data[id].state_trajectory # trajectory at the data point
					
			grads = Flux.gradient(lf) do model # calculate gradient
				# measure variability of clamped growth rate
				growth_loss(model, strj; lambda=lambda, δ=δ, β=β) 
			end

			Flux.update!(opt_state, lf, grads[1]) # update model

			grads = Flux.gradient(lf) do model # calculate gradient
				# measure variability of clamped growth rate
				growth_loss(model, strj_max; lambda=lambda, δ=δ, β=β) 
			end

			Flux.update!(opt_state, lf, grads[1])

			if growth_loss(lf, strj; lambda=lambda, δ=δ, β=β) < growth_loss(lf, strj; lambda=lambda, δ=δ, β=β)
				strj_max = strj 
			end

			# save network if a file to save if provided
			# and sufficient time has elapsed
			if !isempty(savefile) &&  (time() - start_time > save_period) 
				JLD2.save(
					savefile, 
					Dict(
						"args"=>args, # positional arguments
						"kwargs"=>kwargs, # keyword arguments
						"model_state"=>Flux.state(lf) # parameters of network
					)
				)
				start_time = time()
			end
		end
	end

	# save network
	if !isempty(savefile)
		JLD2.save(
			savefile, 
			Dict(
				"args"=>args, # positional arguments
				"kwargs"=>kwargs, # keyword arguments
				"model_state"=>Flux.state(lf) # parameters of network
			)
		)
	end
	
	return lf
end

# ╔═╡ 6dd10992-4028-42c4-b66f-b80a0e4ec5a4
let
	Random.seed!(0)
	ds = discrete_vehicles(0.25)
	data = JLD2.load("data/car/trajectories.jld2", "data")

	ov_idx = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 for d in data]
	ov_data = data[ov_idx]

	idx = rand(1:length(ov_data))
	println("test id: $idx")

	lambda = eps(Float16)
	δ = 1f-12
	β = Float32(40)
	savefile = "data/car/seqgen/ann.jld2"
	opt = Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam())
	
	lf = train(LiftingFlow, 14, ov_data; maxiter = 0, clamp = 3.0, optimiser=opt, lambda=lambda, δ=δ, β=β, savefile=savefile)
	
	old_loss = growth_loss(lf, ov_data[idx].state_trajectory)

	old_flat, _ = Flux.destructure(lf)
	
	lf = train(LiftingFlow, 14, ov_data; maxiter = 1, clamp = 3.0, optimiser=opt, lambda=lambda, δ=δ, β=β, savefile=savefile)

	new_loss = growth_loss(lf, ov_data[idx].state_trajectory)

	old_loss - new_loss
	
end

# ╔═╡ 87c0899f-1858-4151-a1b5-900bbbbef277
Flux.leakyrelu(rand(ComplexF32))

# ╔═╡ Cell order:
# ╠═90646a38-9eb9-11f0-3492-0567c353dada
# ╠═b3d77e2d-adc1-43e9-a61a-2ce2a71bf622
# ╠═a624594d-fe9e-42fa-8cdd-9c6ea365b832
# ╠═9f8ee7b1-38b9-4704-8a9a-ffdc7d8871de
# ╠═ef9e1e16-1864-4349-b382-7467698bde62
# ╠═de605775-6476-4caf-8bcf-e62e6414370a
# ╠═75dbde3e-095f-4a92-b3af-92ba1bb2a3a9
# ╠═6dd10992-4028-42c4-b66f-b80a0e4ec5a4
# ╠═87c0899f-1858-4151-a1b5-900bbbbef277

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
using JLD2: load
using LazySets
using Flux
import ReachabilityCascade
import ReachabilityCascade: build, RefinementModel
import ReachabilityCascade.TrajectoryRefiner: rollout_guess, refinement_grads, refinement_loss
import ReachabilityCascade.CarDataGeneration: discrete_vehicles
import ReachabilityCascade: TransitionNetwork, load_transition_network
	
end

# ╔═╡ 02466dca-7603-4891-9769-dc779fd65dc8
begin

mutable struct VehicleTrajectoryIterator
    data::Vector{Any}
    rng::Random.AbstractRNG
    start_min::Int
    start_max::Int
    iter::Integer
    epoch::Integer
    max_iter::Integer
    max_epoch::Integer
end

function VehicleTrajectoryIterator(dataset::AbstractVector;
                                   rng::Random.AbstractRNG=Random.default_rng(),
                                   start_min::Int=1,
                                   start_max::Union{Nothing,Int}=nothing,
                                   max_epoch::Int=1,
                                   max_iter::Union{Nothing,Int}=nothing)
    isempty(dataset) && throw(ArgumentError("trajectory dataset is empty"))

    dataset_vec = collect(dataset)
    shuffle!(rng, dataset_vec)
    max_iter_val = max_iter === nothing ? length(dataset_vec) : max_iter
    max_iter_val > 0 || throw(ArgumentError("max_iter must be positive"))
    max_epoch > 0 || throw(ArgumentError("max_epoch must be positive"))
    default_max_start = size(dataset_vec[1].state_trajectory, 2) - 1
    start_max_val = start_max === nothing ? default_max_start : start_max
    start_max_val >= 1 || throw(ArgumentError("start_max must allow at least one step"))
    start_min <= start_max_val || throw(ArgumentError("start_min must be ≤ start_max"))

    return VehicleTrajectoryIterator(dataset_vec, rng, start_min, start_max_val,
                                     1, 1, max_iter_val, max_epoch)
end

function VehicleTrajectoryIterator(; path::AbstractString="data/car/trajectories.jld2", kwargs...)
    dataset_dict = load(path)
    dataset = haskey(dataset_dict, "data") ? dataset_dict["data"] : dataset_dict
    return VehicleTrajectoryIterator(dataset; kwargs...)
end

end

# ╔═╡ 879af5e1-9bb4-4398-bc8b-eda26a775d40
function Base.iterate(iter::VehicleTrajectoryIterator, state::Tuple{Int,Int}=(iter.iter, iter.epoch))
    isempty(iter.data) && return nothing

    idx, epoch = state
    if idx > iter.max_iter
        shuffle!(iter.rng, iter.data)
        epoch += 1
        idx = 1
    end
    epoch > iter.max_epoch && return nothing

    sample = iter.data[idx]
    x_full = Float32.(Array(sample.state_trajectory))
    u_full = Float32.(Array(sample.input_signal))
    T = size(x_full, 2)
    T > 1 || throw(ArgumentError("trajectory must have at least two state samples"))

    upper = min(iter.start_max, T - 1)
    lower = min(iter.start_min, upper)
    start_idx = rand(iter.rng, lower:upper)
    x_segment = x_full[:, start_idx:end]
    body_len = size(x_segment, 2) - 1
    u_segment = u_full[:, start_idx:(start_idx + body_len - 1)]

    # Simple guess: repeat the initial state and zero inputs
    x0 = x_segment[:, 1:1]
    x_guess = repeat(x0, 1, body_len)
    u_guess = zeros(Float32, size(u_segment))
    x_target = Float32.(x_segment[:, 2:end])

    next_state = (idx + 1, epoch)
    iter.iter, iter.epoch = next_state
    bundle = ReachabilityCascade.TrajectoryRefiner.ShootingBundle(reshape(x0, size(x0,1), 1, 1), x_guess, u_guess; x_target=x_target)
    return bundle, next_state
end

# ╔═╡ cc6ab58a-4dfa-4df3-8e68-10e968df6eeb
function cost_fn(x_tensor::AbstractArray)
	x = Float32.(x_tensor)
	scale = Float32.([1.0, 1.0, 10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
	
	# bounds cost 
	X = Hyperrectangle(
		Float32.(vcat([50, 4.0, 0.0, 10.0], zeros(3), 50.0, 1.75, 5.0, 50.0, 6.0, -5.0)),
		Float32.([100, 3.0, 1.0, 10.0, 1.0, 1.0, 0.2, 100.0, 0.1, 1.0, 100.0, 0.1, 1.0])
	)	
	bc = Flux.relu(abs.(x .- center(X)) .- radius_hyperrectangle(X)).*scale 

	# forward collision cost 
	fc = min.(Flux.relu(Float32.(5.0) .- abs.(x[1:1, :] - x[8:8, :])), Flux.relu(Float32.(2.0) .- abs.(x[2:2, :] - x[9:9, :])))
	
	# oncoming collision cost 
	oc = min.(Flux.relu(Float32.(5.0) .- abs.(x[1:1, :] - x[10:10, :])), Flux.relu(Float32.(2.0) .- abs.(x[2:2, :] - x[11:11, :])))

	# terminal cost 
	tc = fill(Flux.relu(x[8, end] - x[1, end]), 1, size(x_tensor, 2))
	
	return vcat(bc, fc, oc, tc)
end

# ╔═╡ c8e15445-b1c1-4f9f-968c-030a00544956
function mismatch_fn(x::AbstractArray, y::AbstractArray)
	scale = Float32.([1.0, 1.0, 10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
	diff = Float32.(x) .- Float32.(y)
	return sum(abs.(diff) .* scale)/size(x, 2)
end

# ╔═╡ eab311cb-4227-4968-8ecf-52e07780b513
let 
	ds = discrete_vehicles(0.25)
	car_data = load("data/car/trajectories.jld2")["data"]
	overidx = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 for d in car_data]
	over_data = car_data[overidx]
	iterator = VehicleTrajectoryIterator(over_data; max_epoch=1, max_iter=10000,
										 start_min = 28)
	transition_model = load_transition_network("data/car/vehiclenet.jld2")
	transition_fn = (x,u) -> Float32.(transition_model(Float32.(x), Float32.(u)))

	# testing error
	test_iterator = VehicleTrajectoryIterator(over_data; max_epoch=1, max_iter=100,
										 start_min = 28)
	sb, _ = iterate(test_iterator)
	x_res = rollout_guess(sb, transition_fn)
	loss_cost = cost_fn(cat(sb.x0, sb.x_guess; dims=2))
	loss_mis = mismatch_fn(x_res, sb.x_guess)


	# Single-sample gradient sanity check before training
	cost_dim = size(cost_fn(cat(sb.x0, sb.x_guess; dims=2)), 1)
	model = RefinementModel(size(sb.x_guess, 1), size(sb.u_guess, 1), cost_dim, 128, 2; activation=Flux.sigmoid)

	scale = Float32.([1.0, 1.0, 10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
	
	refined = model(sb, transition_model, cost_fn, 1)
	x_res = rollout_guess(refined, transition_fn)
	x_body = refined.x_guess
	mismatch_fn(x_body, x_res)
	x_body - x_res

	losses = []
	for sb in test_iterator 
		ls = refinement_loss(model, transition_fn, cost_fn, mismatch_fn, sb, 2; imitation_weight=0.0)
		push!(losses, ls)
	end
	losses, findmax(losses)

	fluxopt = Flux.OptimiserChain(Flux.ClipGrad(), Flux.ClipNorm(), Flux.Adam())
	
	res = build(RefinementModel, iterator, 5, 1, transition_fn, cost_fn, mismatch_fn; depth=3, imitation_weight=0.0, opt=fluxopt, backprop_mode=:min_loss)
	
end

# ╔═╡ fd987af3-e968-4340-b62b-304144a8a691
md"""
- Option to change mismatch function and trajectory loss using softmax to approximate the maximum.
"""

# ╔═╡ Cell order:
# ╠═329b39a0-cc1f-11f0-372b-33fb7f28c501
# ╠═6b799ba4-d33a-4a98-9623-46ad55a186df
# ╠═02466dca-7603-4891-9769-dc779fd65dc8
# ╠═879af5e1-9bb4-4398-bc8b-eda26a775d40
# ╠═cc6ab58a-4dfa-4df3-8e68-10e968df6eeb
# ╠═c8e15445-b1c1-4f9f-968c-030a00544956
# ╠═eab311cb-4227-4968-8ecf-52e07780b513
# ╠═fd987af3-e968-4340-b62b-304144a8a691

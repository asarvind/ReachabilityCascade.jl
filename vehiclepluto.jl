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

"""
    VehicleTrajectoryIterator(epsilon_state::AbstractVector; epsilon_input=nothing,
                              path::AbstractString="data/car/trajectories.jld2",
                              rng::AbstractRNG=Random.default_rng())

Create a mutable iterator over vehicle trajectories stored in `trajectories.jld2`. Each iteration
draws a target trajectory by truncating a random prefix, and produces a noisy guess trajectory by
blending uniform samples from the state/input sets with the data trajectory using elementwise
weights `epsilon_state`/`epsilon_input`: `eps .* random + (1 - eps) .* data`. When the internal state
runs past the dataset length, the data are shuffled and iteration wraps around.
"""
mutable struct VehicleTrajectoryIterator
    data::Vector{Any}
    epsilon_state::Vector{<:Real}
    epsilon_input::Vector{<:Real}
    state_set::LazySet
    input_set::LazySet
    rng::Random.AbstractRNG
    start_min::Int
    start_max::Int
    iter::Integer
    epoch::Integer
    max_iter::Integer
    max_epoch::Integer
end

function VehicleTrajectoryIterator(dataset::AbstractVector,
                                   epsilon_state::AbstractVector,
                                   epsilon_input::AbstractVector,
                                   state_set::LazySet,
                                   input_set::LazySet;
                                   rng::Random.AbstractRNG=Random.default_rng(),
                                   start_min::Int=1,
                                   start_max::Union{Nothing,Int}=nothing,
                                   max_epoch::Int=1,
                                   max_iter::Union{Nothing,Int}=nothing)
    isempty(dataset) && throw(ArgumentError("trajectory dataset is empty"))

    epsilon_state_vec = Float32.(collect(epsilon_state))
    state_dim = size(dataset[1].state_trajectory, 1)
    state_dim == length(epsilon_state_vec) || throw(ArgumentError("epsilon_state length must match state dimension ($state_dim)"))

    input_dim = size(dataset[1].input_signal, 1)
    epsilon_input_vec = Float32.(collect(epsilon_input))
    input_dim == length(epsilon_input_vec) || throw(ArgumentError("epsilon_input length must match input dimension ($input_dim)"))

    dataset_vec = collect(dataset)
    shuffle!(rng, dataset_vec)
    max_iter_val = max_iter === nothing ? length(dataset_vec) : max_iter
    max_iter_val > 0 || throw(ArgumentError("max_iter must be positive"))
    max_epoch > 0 || throw(ArgumentError("max_epoch must be positive"))
    default_max_start = size(dataset_vec[1].state_trajectory, 2) - 1
    start_max_val = start_max === nothing ? default_max_start : start_max
    start_max_val >= 1 || throw(ArgumentError("start_max must allow at least one step"))
    start_min <= start_max_val || throw(ArgumentError("start_min must be ≤ start_max"))

    return VehicleTrajectoryIterator(dataset_vec, epsilon_state_vec, epsilon_input_vec,
                                     state_set, input_set, rng, start_min, start_max_val,
                                     1, 1, max_iter_val, max_epoch)
end

function VehicleTrajectoryIterator(epsilon_state::AbstractVector, epsilon_input::AbstractVector;
                                   path::AbstractString="data/car/trajectories.jld2",
                                   state_set::LazySet,
                                   input_set::LazySet, kwargs...)
    dataset_dict = load(path)
    dataset = haskey(dataset_dict, "data") ? dataset_dict["data"] : dataset_dict
    return VehicleTrajectoryIterator(dataset, epsilon_state, epsilon_input,
                                     state_set, input_set; kwargs...)
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

    # Sample from state/input sets and blend with data using epsilon weights
    x_rand = reduce(hcat, [Float32.(LazySets.sample(iter.state_set)) for _ in 1:(body_len + 1)])
    u_rand = reduce(hcat, [Float32.(LazySets.sample(iter.input_set)) for _ in 1:body_len])
    # Clamp random samples to the set bounds to avoid runaway values
    x_center, x_radius = center(iter.state_set), radius(iter.state_set)
    u_center, u_radius = center(iter.input_set), radius(iter.input_set)
    x_rand = clamp.(x_rand, x_center .- x_radius, x_center .+ x_radius)
    u_rand = clamp.(u_rand, u_center .- u_radius, u_center .+ u_radius)

    eps_state_mat = reshape(iter.epsilon_state, :, 1)
    eps_input_mat = reshape(iter.epsilon_input, :, 1)

    x_body_guess = eps_state_mat .* x_rand[:, 2:end] .+ (1 .- eps_state_mat) .* x_segment[:, 2:end]
    x_guess = hcat(x_segment[:, 1:1], x_body_guess)
    u_guess = eps_input_mat .* u_rand .+ (1 .- eps_input_mat) .* u_segment

    x_guess = Float32.(x_guess)
    u_guess = Float32.(u_guess)
    x_target = Float32.(x_segment[:, 2:end])

    next_state = (idx + 1, epoch)
    iter.iter, iter.epoch = next_state
    bundle = ReachabilityCascade.TrajectoryRefiner.ShootingBundle(x_guess, u_guess; x_target=x_target)
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
	tc = hcat(zeros(1, size(x_tensor, 2)-1), Flux.relu(x[8, end] - x[1, end]))

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
	eps_state = ones(13)*1.0
	eps_input = ones(2)*1.0
	car_data = load("data/car/trajectories.jld2")["data"]
	overidx = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 for d in car_data]
	over_data = car_data[overidx]
	iterator = VehicleTrajectoryIterator(over_data, eps_state, eps_input,
	                                     ds.X, ds.U; max_epoch=8, max_iter=10000,
										 start_min = 28)
	transition_model = load_transition_network("data/car/vehiclenet.jld2")
	transition_fn = (x,u) -> Float32.(transition_model(Float32.(x), Float32.(u)))

	# testing error
	test_iterator = VehicleTrajectoryIterator(over_data, eps_state, eps_input,
	                                     ds.X, ds.U; max_epoch=1, max_iter=100,
										 start_min = 28)
	sb, _ = iterate(test_iterator)
	x_res = rollout_guess(sb, transition_fn)
	loss_cost = cost_fn(sb.x_guess)
	loss_mis = mismatch_fn(x_res, selectdim(sb.x_guess, 2, 2:size(sb.x_guess,2)))


	# Single-sample gradient sanity check before training
	model = RefinementModel(size(sb.x_guess, 1), size(sb.u_guess, 1), 16, 128, 2, activation=Flux.sigmoid)

	scale = Float32.([1.0, 1.0, 10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
	
	refined = model(sb, transition_model, cost_fn, 1)
	x_res = rollout_guess(refined, transition_fn)
	x_body = selectdim(refined.x_guess, 2, 2:size(refined.x_guess, 2))
	mismatch_fn(x_body, x_res)
	x_body - x_res

	losses = []
	for sb in test_iterator 
		ls = refinement_loss(model, transition_fn, cost_fn, mismatch_fn, sb, 2; imitation_weight=0.0)
		push!(losses, ls)
	end
	losses, findmax(losses)

	build(RefinementModel, iterator, 5, 1, transition_fn, cost_fn, mismatch_fn; depth=2, imitation_weight=0.0, opt=Flux.OptimiserChain(Flux.ClipGrad(), Flux.ClipNorm(), Flux.Adam()))
end

# ╔═╡ Cell order:
# ╠═329b39a0-cc1f-11f0-372b-33fb7f28c501
# ╠═6b799ba4-d33a-4a98-9623-46ad55a186df
# ╠═02466dca-7603-4891-9769-dc779fd65dc8
# ╠═879af5e1-9bb4-4398-bc8b-eda26a775d40
# ╠═cc6ab58a-4dfa-4df3-8e68-10e968df6eeb
# ╠═c8e15445-b1c1-4f9f-968c-030a00544956
# ╠═eab311cb-4227-4968-8ecf-52e07780b513

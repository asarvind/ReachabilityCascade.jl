module VehicleExp

using Random
using JLD2: load
import ReachabilityCascade

"""
    VehicleTrajectoryIterator(epsilon_state::AbstractVector; epsilon_input=nothing,
                              path::AbstractString="data/car/trajectories.jld2",
                              rng::AbstractRNG=Random.default_rng())

Create a mutable iterator over vehicle trajectories stored in `trajectories.jld2`. Each iteration
draws a target trajectory by truncating a random prefix, and produces a noisy guess trajectory by
adding elementwise noise `epsilon .* randn` (with `epsilon_state` for states and `epsilon_input` for
inputs). When the internal state runs past the dataset length, the data are shuffled and iteration
wraps around.
"""
mutable struct VehicleTrajectoryIterator
    data::Vector{Any}
    epsilon_state::Vector{Float64}
    epsilon_input::Vector{Float64}
    rng::Random.AbstractRNG
    iter::Int
    epoch::Int
    max_iter::Int
    max_epoch::Int
end

function VehicleTrajectoryIterator(epsilon_state::AbstractVector; epsilon_input::Union{Nothing,AbstractVector}=nothing,
                                   path::AbstractString="data/car/trajectories.jld2",
                                   rng::Random.AbstractRNG=Random.default_rng(),
                                   max_epoch::Int=1,
                                   max_iter::Union{Nothing,Int}=nothing)
    dataset_dict = load(path)
    dataset = haskey(dataset_dict, "data") ? dataset_dict["data"] : dataset_dict
    isempty(dataset) && throw(ArgumentError("trajectory dataset at $path is empty"))

    epsilon_state_vec = collect(Float64, epsilon_state)
    state_dim = size(dataset[1].state_trajectory, 1)
    state_dim == length(epsilon_state_vec) || throw(ArgumentError("epsilon_state length must match state dimension ($state_dim)"))

    input_dim = size(dataset[1].input_signal, 1)
    epsilon_input_vec = epsilon_input === nothing ? zeros(Float64, input_dim) : collect(Float64, epsilon_input)
    input_dim == length(epsilon_input_vec) || throw(ArgumentError("epsilon_input length must match input dimension ($input_dim)"))

    shuffle!(rng, dataset)
    max_iter_val = max_iter === nothing ? length(dataset) : max_iter
    max_iter_val > 0 || throw(ArgumentError("max_iter must be positive"))
    max_epoch > 0 || throw(ArgumentError("max_epoch must be positive"))

    return VehicleTrajectoryIterator(dataset, epsilon_state_vec, epsilon_input_vec, rng,
                                     1, 1, max_iter_val, max_epoch)
end

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
    x_full = Array(sample.state_trajectory)
    u_full = Array(sample.input_signal)
    T = size(x_full, 2)
    T > 1 || throw(ArgumentError("trajectory must have at least two state samples"))

    start_idx = rand(iter.rng, 1:(T - 1))
    x_segment = x_full[:, start_idx:end]
    body_len = size(x_segment, 2) - 1
    u_segment = u_full[:, start_idx:(start_idx + body_len - 1)]

    x_guess = copy(x_segment)
    if body_len > 0
        x_noise = reshape(iter.epsilon_state, :, 1) .* randn(iter.rng, length(iter.epsilon_state), body_len)
        x_guess[:, 2:end] .+= x_noise
    end

    u_guess = copy(u_segment)
    if body_len > 0
        u_noise = reshape(iter.epsilon_input, :, 1) .* randn(iter.rng, length(iter.epsilon_input), body_len)
        u_guess .+= u_noise
    end

    x_target = x_segment[:, 2:end]

    next_state = (idx + 1, epoch)
    iter.iter, iter.epoch = next_state
    bundle = ReachabilityCascade.TrajectoryRefiner.ShootingBundle(x_guess, u_guess; x_target=x_target)
    return bundle, next_state
end

"""
    example(; epsilon_state_fill=0.05, max_epoch=1, max_iter=3, path="data/car/trajectories.jld2")

Run a short demonstration of the iterator and print the shapes of a few sampled bundles.
"""
function example(; epsilon_state_fill=0.05, max_epoch::Int=1, max_iter::Int=3,
                 path::AbstractString="data/car/trajectories.jld2")
    data = load(path)
    dataset = haskey(data, "data") ? data["data"] : data
    eps_state = fill(epsilon_state_fill, size(dataset[1].state_trajectory, 1))
    itr = VehicleTrajectoryIterator(eps_state; max_epoch=max_epoch, max_iter=max_iter, path=path)
    for (i, sample) in enumerate(itr)
        @info "Sample $i" x_guess=size(sample.x_guess) u_guess=size(sample.u_guess)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    example()
end

end # module

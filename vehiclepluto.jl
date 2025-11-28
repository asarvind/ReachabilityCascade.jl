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
import ReachabilityCascade
import ReachabilityCascade.TrajectoryRefiner: rollout_guess
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

function VehicleTrajectoryIterator(epsilon_state::AbstractVector, epsilon_input::AbstractVector;
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
    epsilon_input_vec = collect(Float64, epsilon_input)
    input_dim == length(epsilon_input_vec) || throw(ArgumentError("epsilon_input length must match input dimension ($input_dim)"))

    shuffle!(rng, dataset)
    max_iter_val = max_iter === nothing ? length(dataset) : max_iter
    max_iter_val > 0 || throw(ArgumentError("max_iter must be positive"))
    max_epoch > 0 || throw(ArgumentError("max_epoch must be positive"))

    return VehicleTrajectoryIterator(dataset, epsilon_state_vec, epsilon_input_vec, rng,
                                     1, 1, max_iter_val, max_epoch)
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

# ╔═╡ eab311cb-4227-4968-8ecf-52e07780b513
let 
	ds = discrete_vehicles(0.25)
	eps_state = [1.0, 0.5, 0.1, 1.0, 0.1, 0.1, 0.005, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0]*2
	eps_input = [0.1, 1.0]
	iterator = VehicleTrajectoryIterator(eps_state, eps_input; max_epoch=1, max_iter=3)
	sb, _ = iterate(iterator)
	transition_model = load_transition_network("data/car/vehiclenet.jld2")
	rollout_guess(sb, transition_model), sb.x_target
end

# ╔═╡ Cell order:
# ╠═329b39a0-cc1f-11f0-372b-33fb7f28c501
# ╠═6b799ba4-d33a-4a98-9623-46ad55a186df
# ╠═02466dca-7603-4891-9769-dc779fd65dc8
# ╠═879af5e1-9bb4-4398-bc8b-eda26a775d40
# ╠═eab311cb-4227-4968-8ecf-52e07780b513

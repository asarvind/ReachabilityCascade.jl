import Flux
using Statistics: mean
using ..GatedLinearUnits: glu_mlp
using ..SequenceTransform: SequenceTransformation
using ..ControlSystem: DiscreteRandomSystem

"""
    PrimaryPolicyNetwork(state_dim, input_dim, latent_dim, hidden_dim, depth)

Primary feedback policy network `π(x, z) -> u`.

At each time step, it maps the current state `x` and latent vector `z` to a control input:
`u = π([x; z])`.
"""
struct PrimaryPolicyNetwork{N}
    state_dim::Int
    input_dim::Int
    latent_dim::Int
    hidden_dim::Int
    depth::Int
    net::N
end

Flux.@layer PrimaryPolicyNetwork

function PrimaryPolicyNetwork(state_dim::Integer,
                              input_dim::Integer,
                              latent_dim::Integer,
                              hidden_dim::Integer,
                              depth::Integer)
    state_dim > 0 || throw(ArgumentError("state_dim must be positive"))
    input_dim > 0 || throw(ArgumentError("input_dim must be positive"))
    latent_dim > 0 || throw(ArgumentError("latent_dim must be positive"))
    hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive"))
    depth > 0 || throw(ArgumentError("depth must be positive"))

    in_dim = Int(state_dim + latent_dim)
    net = glu_mlp(in_dim, Int(hidden_dim), Int(input_dim); n_glu=Int(depth))
    return PrimaryPolicyNetwork(Int(state_dim), Int(input_dim), Int(latent_dim),
                                Int(hidden_dim), Int(depth), net)
end

"""
    (m::PrimaryPolicyNetwork)(x, latent) -> u

Compute a single-step feedback input.

Arguments:
- `x`: state vector of length `state_dim`.
- `latent`: latent vector of length `latent_dim`.

Returns:
- `u`: input vector of length `input_dim`.
"""
function (m::PrimaryPolicyNetwork)(x::AbstractVector, latent::AbstractVector)
    length(x) == m.state_dim || throw(DimensionMismatch("x must have length $(m.state_dim)"))
    length(latent) == m.latent_dim || throw(DimensionMismatch("latent must have length $(m.latent_dim)"))
    return vec(m.net(vcat(x, latent)))
end

"""
    DeltaNetwork(state_dim, input_dim, cost_dim, latent_dim, seq_len, hidden_dim, depth;
                 max_seq_len=nothing, nheads=1, activation=Flux.gelu)

Shared-difference latent update network.

Given:
- rollout states excluding the initial state `x_body` (`state_dim × seq_len`),
- the corresponding input sequence `u_guess` (`input_dim × seq_len`),
- per-timestep costs `cost_body` (`cost_dim × seq_len`),
- the current latent vector `latent` (length `latent_dim`),

it predicts a latent update `Δlatent` (length `latent_dim`) via a structured difference:

`Δ_seq = core([x; u; cost], [x0; latent]) - core([x; u; 0], [x0; latent])`

The sequence output `Δ_seq` (a `latent_dim × seq_len` matrix) is reduced to a vector by averaging over time.
"""
struct DeltaNetwork{C}
    state_dim::Int
    input_dim::Int
    cost_dim::Int
    latent_dim::Int
    seq_len::Int
    hidden_dim::Int
    depth::Int
    max_seq_len::Int
    nheads::Int
    activation
    core::C
end

Flux.@layer DeltaNetwork

function DeltaNetwork(state_dim::Integer,
                      input_dim::Integer,
                      cost_dim::Integer,
                      latent_dim::Integer,
                      seq_len::Integer,
                      hidden_dim::Integer,
                      depth::Integer;
                      max_seq_len::Union{Nothing,Integer}=nothing,
                      nheads::Integer=1,
                      activation=Flux.gelu)
    state_dim > 0 || throw(ArgumentError("state_dim must be positive"))
    input_dim > 0 || throw(ArgumentError("input_dim must be positive"))
    cost_dim >= 0 || throw(ArgumentError("cost_dim must be non-negative"))
    latent_dim > 0 || throw(ArgumentError("latent_dim must be positive"))
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive"))
    depth > 0 || throw(ArgumentError("depth must be positive"))

    in_dim = Int(state_dim + input_dim + cost_dim)
    out_dim = Int(latent_dim)
    context_dim = Int(state_dim + latent_dim)
    max_seq_len_int = max_seq_len === nothing ? Int(seq_len) : Int(max_seq_len)
    max_seq_len_int >= Int(seq_len) ||
        throw(ArgumentError("max_seq_len must be ≥ seq_len = $seq_len; got $max_seq_len_int"))

    nheads_int = Int(nheads)
    core = SequenceTransformation(in_dim, Int(hidden_dim), out_dim, Int(depth),
                                  context_dim, activation;
                                  max_seq_len=max_seq_len_int, nheads=nheads_int)

    return DeltaNetwork(Int(state_dim), Int(input_dim), Int(cost_dim), Int(latent_dim), Int(seq_len),
                        Int(hidden_dim), Int(depth), max_seq_len_int, nheads_int, activation, core)
end

"""
    (m::DeltaNetwork)(x0, latent, x_body, u_guess, cost_body) -> Δlatent

Compute the latent update `Δlatent` as a vector of length `latent_dim`.
"""
function (m::DeltaNetwork)(x0::AbstractVector,
                           latent::AbstractVector,
                           x_body::AbstractMatrix,
                           u_guess::AbstractMatrix,
                           cost_body::AbstractMatrix)
    length(x0) == m.state_dim || throw(DimensionMismatch("x0 must have length $(m.state_dim)"))
    length(latent) == m.latent_dim || throw(DimensionMismatch("latent must have length $(m.latent_dim)"))
    size(x_body, 1) == m.state_dim || throw(DimensionMismatch("x_body must have $(m.state_dim) rows"))
    size(u_guess, 1) == m.input_dim || throw(DimensionMismatch("u_guess must have $(m.input_dim) rows"))
    size(cost_body, 1) == m.cost_dim || throw(DimensionMismatch("cost_body must have $(m.cost_dim) rows"))
    size(x_body, 2) == m.seq_len || throw(DimensionMismatch("x_body must have $(m.seq_len) columns"))
    size(u_guess, 2) == m.seq_len || throw(DimensionMismatch("u_guess must have $(m.seq_len) columns"))
    size(cost_body, 2) == m.seq_len || throw(DimensionMismatch("cost_body must have $(m.seq_len) columns"))

    zero_cost = zeros(eltype(cost_body), m.cost_dim, m.seq_len)
    context = vcat(x0, latent)

    in_real = vcat(x_body, u_guess, cost_body)
    in_zero = vcat(x_body, u_guess, zero_cost)

    y_real = m.core(in_real, context)
    y_zero = m.core(in_zero, context)
    y = y_real .- y_zero

    return vec(mean(y, dims=2))
end

"""
    RefinementRNN(state_dim, input_dim, cost_dim, latent_dim, seq_len,
                  policy_hidden_dim, policy_depth,
                  delta_hidden_dim, delta_depth; kwargs...)

Recurrent refinement model with:
- a primary feedback policy `u_t = π(x_t, latent)`, and
- a latent update model `latent ← latent ± Δlatent`.

At each refinement step, the sign `±` is chosen by comparing the scalarized trajectory costs obtained by
rolling out the resulting candidate latent vectors.
"""
struct RefinementRNN{P,D}
    policy::P
    delta::D
end

Flux.@layer RefinementRNN

function RefinementRNN(state_dim::Integer,
                       input_dim::Integer,
                       cost_dim::Integer,
                       latent_dim::Integer,
                       seq_len::Integer,
                       policy_hidden_dim::Integer,
                       policy_depth::Integer,
                       delta_hidden_dim::Integer,
                       delta_depth::Integer;
                       max_seq_len::Union{Nothing,Integer}=nothing,
                       nheads::Integer=1,
                       activation=Flux.gelu)
    policy = PrimaryPolicyNetwork(state_dim, input_dim, latent_dim, policy_hidden_dim, policy_depth)
    delta = DeltaNetwork(state_dim, input_dim, cost_dim, latent_dim, seq_len, delta_hidden_dim, delta_depth;
                         max_seq_len=max_seq_len, nheads=nheads, activation=activation)
    return RefinementRNN(policy, delta)
end

"""
    (m::RefinementRNN)(x0, sys, traj_cost_fn, steps::Integer;
                      temperature=1, dual=false) -> (; latents, u_guesses, x_rollouts, costs, step_costs, best_step)

Run `steps` refinement iterations.

At each step:
1. Roll out a horizon of length `seq_len` by applying the feedback policy `u_t = m.policy(x_t, latent)`.
2. Compute `x_body = x_rollout[:, 2:end]` and `cost_body = traj_cost_fn(x_body)`.
3. Compute `Δlatent = m.delta(x0, latent, x_body, u_guess, cost_body)`.
4. Evaluate both candidates `latent ± Δlatent` by rolling out each candidate and choosing the one with
   the lower scalarized trajectory cost (when `dual=true`). With `dual=false`, only the `+` candidate is evaluated.

Returns:
- `latents`: latent vector log, length `steps+1` (includes `latent₀`).
- `u_guesses`: chosen input sequences, length `steps` (each `input_dim×seq_len`).
- `x_rollouts`: chosen rollouts, length `steps` (each `state_dim×(seq_len+1)`).
- `costs`: chosen cost sequences on `x_body`, length `steps` (each `cost_dim×seq_len`).
- `step_costs`: scalarized cost for each step (the chosen candidate).
- `best_step`: the (1-based) step index with the minimum scalarized cost among `step_costs`.
"""
function (m::RefinementRNN)(x0::AbstractVector,
                            sys::DiscreteRandomSystem,
                            traj_cost_fn,
                            steps::Integer;
                            temperature::Real=1,
                            dual::Bool=false)
    steps >= 1 || throw(ArgumentError("steps must be ≥ 1"))
    temperature > 0 || throw(ArgumentError("temperature must be positive"))

    state_dim = m.delta.state_dim
    input_dim = m.delta.input_dim
    cost_dim = m.delta.cost_dim
    latent_dim = m.delta.latent_dim
    seq_len = m.delta.seq_len

    length(x0) == state_dim || throw(DimensionMismatch("x0 must have length $state_dim"))

    x0_vec = Float64.(Vector(x0))
    latent = zeros(Float32, latent_dim)
    steps_int = Int(steps)

    latents = Vector{Vector{Float32}}(undef, steps_int + 1)
    latents[1] = latent
    u_guesses = Vector{Matrix{Float32}}(undef, steps_int)
    x_rollouts = Vector{Matrix{Float64}}(undef, steps_int)
    costs = Vector{Matrix{Float32}}(undef, steps_int)
    step_costs = Vector{Float32}(undef, steps_int)

    best_step = 1
    best_score = Inf32

    scalar_cost = function (cost_body_mat)
        size(cost_body_mat, 1) == cost_dim || throw(DimensionMismatch("cost must have $cost_dim rows"))
        size(cost_body_mat, 2) == seq_len || throw(DimensionMismatch("cost must have $seq_len columns"))
        return abs(Float32(_softmax_average(cost_body_mat; temperature=temperature)))
    end

    function rollout_with_latent(z::AbstractVector)
        length(z) == latent_dim || throw(DimensionMismatch("latent must have length $latent_dim"))

        x_roll = Matrix{Float64}(undef, state_dim, seq_len + 1)
        u_seq = Matrix{Float32}(undef, input_dim, seq_len)
        x_roll[:, 1] = x0_vec
        for t in 1:seq_len
            x_t = view(x_roll, :, t)
            u_t = m.policy(x_t, z)
            u_seq[:, t] = Float32.(u_t)
            x_roll[:, t + 1] = Float64.(sys(Vector(x_t), Vector(u_t)))
        end

        x_body = x_roll[:, 2:end]
        cost_body = traj_cost_fn(x_body)
        size(cost_body, 1) == cost_dim ||
            throw(DimensionMismatch("traj_cost_fn(x_body) must return $cost_dim rows; got $(size(cost_body, 1))"))
        size(cost_body, 2) == seq_len ||
            throw(DimensionMismatch("traj_cost_fn(x_body) must return $seq_len columns; got $(size(cost_body, 2))"))
        return x_roll, u_seq, Float32.(Matrix(cost_body))
    end

    for k in 1:steps_int
        x_roll_base, u_base, cost_base = rollout_with_latent(latent)
        x_body_base = x_roll_base[:, 2:end]

        Δlatent = m.delta(Float32.(x0_vec),
                          latent,
                          Float32.(Matrix(x_body_base)),
                          u_base,
                          cost_base)

        latent_plus = latent .+ Δlatent
        x_roll_plus, u_plus, cost_plus = rollout_with_latent(latent_plus)
        score_plus = scalar_cost(cost_plus)

        if dual
            latent_minus = latent .- Δlatent
            x_roll_minus, u_minus, cost_minus = rollout_with_latent(latent_minus)
            score_minus = scalar_cost(cost_minus)

            if score_plus <= score_minus
                latent = Float32.(Vector(latent_plus))
                u_guesses[k] = u_plus
                x_rollouts[k] = x_roll_plus
                costs[k] = cost_plus
                step_costs[k] = score_plus
            else
                latent = Float32.(Vector(latent_minus))
                u_guesses[k] = u_minus
                x_rollouts[k] = x_roll_minus
                costs[k] = cost_minus
                step_costs[k] = score_minus
            end
        else
            latent = Float32.(Vector(latent_plus))
            u_guesses[k] = u_plus
            x_rollouts[k] = x_roll_plus
            costs[k] = cost_plus
            step_costs[k] = score_plus
        end

        latents[k + 1] = latent
        if step_costs[k] < best_score
            best_score = step_costs[k]
            best_step = k
        end
    end

    return (; latents=latents,
            u_guesses=u_guesses,
            x_rollouts=x_rollouts,
            costs=costs,
            step_costs=step_costs,
            best_step=best_step)
end

import Flux
using ..SequenceTransform: SequenceTransformation

"""
    ReactiveDenoisingNet(state_dim, input_dim, cost_dim, seq_len, hidden_dim, depth;
                         max_seq_len=nothing, nheads=1, activation=Flux.gelu)

Single-step model-free refiner for an input-sequence guess that *reacts* to trajectory cost.

Inputs (single sample):
- `x0`: current state (length `state_dim`) used as context.
- `x_body`: future state sequence excluding the initial state (`state_dim × seq_len`).
- `u_guess`: input guess sequence (`input_dim × seq_len`).
- `cost_body`: per-timestep cost sequence (`cost_dim × seq_len`).

Outputs:
- `noise`: estimated noise on the guess (`input_dim × seq_len`).
- `U_new`: refined guess `u_guess - noise`.

The noise is computed as a structured difference (cost vs zero-cost):
`noise = core([x; u; cost], x0) - core([x; u; 0], x0)`

Batching:
- `x0`: `state_dim × B`
- `x_body`: `state_dim × seq_len × B`
- `u_guess`: `input_dim × seq_len × B`
- `cost_body`: `cost_dim × seq_len × B`
"""
struct ReactiveDenoisingNet{C}
    state_dim::Int
    input_dim::Int
    cost_dim::Int
    seq_len::Int
    hidden_dim::Int
    depth::Int
    max_seq_len::Int
    nheads::Int
    activation
    core::C
end

Flux.@layer ReactiveDenoisingNet

function ReactiveDenoisingNet(state_dim::Integer,
                              input_dim::Integer,
                              cost_dim::Integer,
                              seq_len::Integer,
                              hidden_dim::Integer,
                              depth::Integer;
                              max_seq_len::Union{Nothing,Integer}=nothing,
                              nheads::Integer=1,
                              activation=Flux.gelu)
    state_dim > 0 || throw(ArgumentError("state_dim must be positive"))
    input_dim > 0 || throw(ArgumentError("input_dim must be positive"))
    cost_dim > 0 || throw(ArgumentError("cost_dim must be positive"))
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive"))
    depth > 0 || throw(ArgumentError("depth must be positive"))

    in_dim = Int(state_dim + input_dim + cost_dim)
    out_dim = Int(input_dim)
    context_dim = Int(state_dim)
    max_seq_len_int = max_seq_len === nothing ? Int(seq_len) : Int(max_seq_len)
    max_seq_len_int >= Int(seq_len) ||
        throw(ArgumentError("max_seq_len must be ≥ seq_len = $seq_len; got $max_seq_len_int"))

    nheads_int = Int(nheads)
    core = SequenceTransformation(in_dim, Int(hidden_dim), out_dim, Int(depth),
                                  context_dim, activation;
                                  max_seq_len=max_seq_len_int, nheads=nheads_int)

    return ReactiveDenoisingNet(Int(state_dim), Int(input_dim), Int(cost_dim), Int(seq_len),
                                Int(hidden_dim), Int(depth), max_seq_len_int, nheads_int, activation, core)
end

function (m::ReactiveDenoisingNet)(x0::AbstractVector,
                                   x_body::AbstractMatrix,
                                   u_guess::AbstractMatrix,
                                   cost_body::AbstractMatrix)
    length(x0) == m.state_dim || throw(DimensionMismatch("x0 must have length $(m.state_dim)"))
    size(x_body, 1) == m.state_dim || throw(DimensionMismatch("x_body must have $(m.state_dim) rows"))
    size(u_guess, 1) == m.input_dim || throw(DimensionMismatch("u_guess must have $(m.input_dim) rows"))
    size(cost_body, 1) == m.cost_dim || throw(DimensionMismatch("cost_body must have $(m.cost_dim) rows"))
    size(x_body, 2) == m.seq_len || throw(DimensionMismatch("x_body must have $(m.seq_len) columns"))
    size(u_guess, 2) == m.seq_len || throw(DimensionMismatch("u_guess must have $(m.seq_len) columns"))
    size(cost_body, 2) == m.seq_len || throw(DimensionMismatch("cost_body must have $(m.seq_len) columns"))

    zero_cost = zeros(eltype(cost_body), m.cost_dim, m.seq_len)
    context = x0
    in_real = vcat(x_body, u_guess, cost_body)
    in_zero = vcat(x_body, u_guess, zero_cost)

    y_real = m.core(in_real, context)
    y_zero = m.core(in_zero, context)
    noise = y_real .- y_zero
    U_new = u_guess .- noise
    return (; U_new=U_new, noise=noise)
end

function (m::ReactiveDenoisingNet)(x0::AbstractMatrix,
                                   x_body::AbstractArray{<:Real,3},
                                   u_guess::AbstractArray{<:Real,3},
                                   cost_body::AbstractArray{<:Real,3})
    size(x0, 1) == m.state_dim || throw(DimensionMismatch("x0 must have $(m.state_dim) rows"))
    size(x_body, 1) == m.state_dim || throw(DimensionMismatch("x_body must have $(m.state_dim) rows"))
    size(u_guess, 1) == m.input_dim || throw(DimensionMismatch("u_guess must have $(m.input_dim) rows"))
    size(cost_body, 1) == m.cost_dim || throw(DimensionMismatch("cost_body must have $(m.cost_dim) rows"))
    size(x_body, 2) == m.seq_len || throw(DimensionMismatch("x_body must have $(m.seq_len) columns"))
    size(u_guess, 2) == m.seq_len || throw(DimensionMismatch("u_guess must have $(m.seq_len) columns"))
    size(cost_body, 2) == m.seq_len || throw(DimensionMismatch("cost_body must have $(m.seq_len) columns"))

    B = size(x0, 2)
    size(x_body, 3) == B || throw(DimensionMismatch("x_body batch size must be $B"))
    size(u_guess, 3) == B || throw(DimensionMismatch("u_guess batch size must be $B"))
    size(cost_body, 3) == B || throw(DimensionMismatch("cost_body batch size must be $B"))

    zero_cost = zeros(eltype(cost_body), m.cost_dim, m.seq_len, B)
    context = x0
    in_real = vcat(x_body, u_guess, cost_body)
    in_zero = vcat(x_body, u_guess, zero_cost)

    y_real = m.core(in_real, context)
    y_zero = m.core(in_zero, context)
    noise = y_real .- y_zero
    U_new = u_guess .- noise
    return (; U_new=U_new, noise=noise)
end

"""
    (m::ReactiveDenoisingNet)(x0, u_guess0, sys, traj_cost_fn; steps=1)

Recursive refinement from an initial guess using a system rollout and a trajectory cost function.

At each refinement iteration `k`:
1. Compute `x_rollout = sys(x0, u_guess)` (must be `state_dim × (seq_len+1)`).
2. Slice `x_body = x_rollout[:, 2:end]`.
3. Compute `cost_body = traj_cost_fn(x_body)` (must be `cost_dim × seq_len`).
4. Apply the single-step denoiser `m(x0, x_body, u_guess, cost_body)` to get a refined guess.

Returns logs:
- `u_guesses`: length `steps+1` (includes initial guess).
- `noises`: length `steps`.
- `x_rollouts`: length `steps`.
- `costs`: length `steps`.
"""
function (m::ReactiveDenoisingNet)(x0::AbstractVector,
                                   u_guess0::AbstractMatrix,
                                   sys,
                                   traj_cost_fn;
                                   steps::Integer=1)
    steps >= 1 || throw(ArgumentError("steps must be ≥ 1"))
    length(x0) == m.state_dim || throw(DimensionMismatch("x0 must have length $(m.state_dim)"))
    size(u_guess0, 1) == m.input_dim || throw(DimensionMismatch("u_guess0 must have $(m.input_dim) rows"))
    size(u_guess0, 2) == m.seq_len || throw(DimensionMismatch("u_guess0 must have $(m.seq_len) columns"))

    steps_int = Int(steps)
    x0_vec = Float32.(Vector(x0))
    u_guess = Float32.(Matrix(u_guess0))

    u_guesses = Vector{Matrix{Float32}}(undef, steps_int + 1)
    noises = Vector{Matrix{Float32}}(undef, steps_int)
    x_rollouts = Vector{Matrix{Float64}}(undef, steps_int)
    costs = Vector{Matrix{Float32}}(undef, steps_int)
    u_guesses[1] = u_guess

    for k in 1:steps_int
        x_roll = sys(Vector(x0), Matrix(u_guess))
        size(x_roll, 1) == m.state_dim ||
            throw(DimensionMismatch("sys(x0, u_guess) must return $(m.state_dim) rows; got $(size(x_roll, 1))"))
        size(x_roll, 2) == m.seq_len + 1 ||
            throw(DimensionMismatch("sys(x0, u_guess) must return $(m.seq_len + 1) columns; got $(size(x_roll, 2))"))

        x_rollouts[k] = Float64.(Matrix(x_roll))
        x_body = x_rollouts[k][:, 2:end]
        cost_body = traj_cost_fn(x_body)
        size(cost_body, 1) == m.cost_dim ||
            throw(DimensionMismatch("traj_cost_fn(x_body) must return $(m.cost_dim) rows; got $(size(cost_body, 1))"))
        size(cost_body, 2) == m.seq_len ||
            throw(DimensionMismatch("traj_cost_fn(x_body) must return $(m.seq_len) columns; got $(size(cost_body, 2))"))

        costs[k] = Float32.(Matrix(cost_body))
        out = m(x0_vec, Float32.(Matrix(x_body)), u_guess, costs[k])
        noises[k] = out.noise
        u_guess = out.U_new
        u_guesses[k + 1] = u_guess
    end

    return (; u_guesses=u_guesses,
            noises=noises,
            x_rollouts=x_rollouts,
            costs=costs)
end

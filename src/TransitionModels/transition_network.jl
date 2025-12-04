using Flux
using ..GatedLinearUnits: glu_mlp

"""
    TransitionNetwork(state_dim::Int, input_dim::Int, hidden_dim::Int;
                      depth::Int=2, act=Flux.σ, bias::Bool=true)

A simple feed-forward transition model built with `glu_mlp` that maps `(state, input)` → `next_state`,
supporting batched inputs.
"""
struct TransitionNetwork{C}
    model::C
    state_dim::Int
    input_dim::Int
    hidden_dim::Int
    depth::Int
end

Flux.@layer TransitionNetwork

function TransitionNetwork(state_dim::Int, input_dim::Int, hidden_dim::Int;
                           depth::Int=2, act=Flux.σ, bias::Bool=true)
    in_dim = state_dim + input_dim
    net = glu_mlp(in_dim, hidden_dim, state_dim; n_glu=depth, act=act, bias=bias)
    return TransitionNetwork(net, state_dim, input_dim, hidden_dim, depth)
end

function _param_eltype(obj, default)
    for t in Flux.trainable(obj)
        if t isa AbstractArray
            return eltype(t)
        else
            inner = _param_eltype(t, nothing)
            inner !== nothing && return inner
        end
    end
    return default
end

function (m::TransitionNetwork)(x::AbstractArray, u::AbstractArray)
    # Align inputs to the eltype of the network parameters (avoids accidental promotion).
    param_T = _param_eltype(m.model, eltype(x))
    x = param_T.(x)
    u = param_T.(u)
    # Supports (features, batch) or (features, time, batch)
    if ndims(x) == 2
        @assert size(x, 1) == m.state_dim "State dimension mismatch"
        @assert size(u, 1) == m.input_dim "Input dimension mismatch"
        @assert ndims(u) == 2 "Expected u to have dimensions (input_dim, batch)"
        xu = cat(x, u; dims=1)                 # (state+input, batch)
        return m.model(xu) .+ x                # residual on state
    elseif ndims(x) == 3
        @assert size(x, 1) == m.state_dim "State dimension mismatch"
        @assert size(u, 1) == m.input_dim "Input dimension mismatch"
        @assert ndims(u) == 3 "Expected u to have dimensions (input_dim, time, batch)"
        f, t, b = size(x)
        xu = cat(x, u; dims=1)                 # (state+input, time, batch)
        xu_flat = reshape(xu, size(xu, 1), t * b)
        y_flat = m.model(xu_flat)              # (state, t*b)
        y = reshape(y_flat, m.state_dim, t, b)
        return y .+ x
    else
        error("Unsupported input dimensions for TransitionNetwork: $(ndims(x))")
    end
end

"""
    (m::TransitionNetwork)(x0::AbstractVector, U::AbstractMatrix)
    (m::TransitionNetwork)(x0::AbstractMatrix, U::AbstractArray{<:Any,3})

Roll out a state trajectory given an initial state (or batched states) and an input signal.

# Arguments
- `x0`: Initial state (`state_dim` vector) or batched states (`state_dim × batch` matrix).
- `U`: Input signal; shape `input_dim × T` for single trajectory, or `input_dim × T × batch` for batched rollout.

# Returns
- State trajectory of shape `state_dim × (T+1)` (single) or `state_dim × (T+1) × batch` (batched),
  including the initial state at index 1.
"""
function (m::TransitionNetwork)(x0::AbstractVector, U::AbstractMatrix)
    @assert length(x0) == m.state_dim "State dimension mismatch"
    @assert size(U, 1) == m.input_dim "Input dimension mismatch"
    T = size(U, 2)
    X = similar(U, m.state_dim, T + 1)
    X[:, 1] = x0
    x_prev = reshape(x0, :, 1)
    for t in 1:T
        u_t = @view U[:, t:t]
        x_prev = m(x_prev, u_t)
        X[:, t + 1] = vec(x_prev)
    end
    return X
end

function (m::TransitionNetwork)(x0::AbstractMatrix, U::AbstractArray{<:Any,3})
    @assert size(x0, 1) == m.state_dim "State dimension mismatch"
    @assert size(U, 1) == m.input_dim "Input dimension mismatch"
    T = size(U, 2)
    B = size(x0, 2)
    @assert size(U, 3) == B "Batch size mismatch between x0 and U"
    X = similar(U, m.state_dim, T + 1, B)
    X[:, 1, :] = x0
    x_prev = x0
    for t in 1:T
        u_t = @view U[:, t, :]
        x_prev = m(x_prev, u_t)
        X[:, t + 1, :] = x_prev
    end
    return X
end

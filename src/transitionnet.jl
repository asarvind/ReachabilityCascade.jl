"""
    TransitionNet(state_dim, control_dim;
                  widths = [128, 128],
                  act = Flux.σ,
                  norm::Bool = true,
                  bias::Bool = true)

Neural network that maps `(x, u) → xnext`, where `x ∈ ℝ^{state_dim}` and
`u ∈ ℝ^{control_dim}`. The body is a stack of GLUs with optional LayerNorm
inserted **between** consecutive GLUs. A final `Dense` projection to
`state_dim` is appended **only if needed** (i.e., when `last(widths) ≠ state_dim`).

Inputs can be vectors or (features, batch) matrices. All inputs are converted to
`Float32` internally for numerical efficiency and GPU friendliness.

Keyword args
- `widths` : vector of hidden widths for the GLU stack (first GLU takes `state_dim + control_dim → widths[1]`)
- `act`    : gate activation for each GLU (default `Flux.σ`)
- `norm`   : if `true` (default), insert `LayerNorm` between GLUs
- `bias`   : include bias in the internal `Dense` layers (default `true`)
"""
struct TransitionNet{L,P}
    layers::L                       # Tuple/Vector of layers (GLUs + optional norms)
    proj::P                         # Dense or Nothing
    state_dim::Int
    control_dim::Int
end

# Register as a Flux layer (trainable fields are detected automatically)
Flux.@layer TransitionNet

# Constructor
function TransitionNet(state_dim::Integer, control_dim::Integer;
                       widths::AbstractVector{<:Integer} = [128, 128],
                       act = Flux.σ,
                       norm::Bool = true,
                       bias::Bool = true)
    @assert !isempty(widths) "`widths` must have at least one element"

    in_dim = state_dim + control_dim

    # Build GLU stack
    lay = Vector{Any}()
    push!(lay, GLU(in_dim => widths[1]; act=act, bias=bias))
    for i in 1:length(widths)-1
        if norm
            push!(lay, LayerNorm(widths[i]))
        end
        push!(lay, GLU(widths[i] => widths[i+1]; act=act, bias=bias))
    end

    # Optional final projection
    last_w = widths[end]
    proj = last_w == state_dim ? nothing : Dense(last_w, state_dim; bias=bias)

    return TransitionNet(tuple(lay...), proj, Int(state_dim), Int(control_dim))
end

# Forward pass: accept (x, u)
function (m::TransitionNet)(x::AbstractVecOrMat, u::AbstractVecOrMat)
    x32 = Float32.(x)
    u32 = Float32.(u)

    @assert size(x32, 1) == m.state_dim "x has incompatible feature size"
    @assert size(u32, 1) == m.control_dim "u has incompatible feature size"

    # Broadcast batch alignment if one of x/u is a vector and the other is a matrix
    if ndims(x32) == 1 && ndims(u32) == 2
        @assert size(u32, 2) > 0
        x32 = reshape(x32, :, 1)
    elseif ndims(x32) == 2 && ndims(u32) == 1
        u32 = reshape(u32, :, 1)
    end

    z = vcat(x32, u32)
    for layer in m.layers
        z = layer(z)
    end
    return m.proj === nothing ? z : m.proj(z)
end

# Convenience: accept a single concatenated input `z` of size (state_dim + control_dim, batch)
function (m::TransitionNet)(z::AbstractVecOrMat)
    z32 = Float32.(z)
    @assert size(z32, 1) == m.state_dim + m.control_dim "concatenated input has wrong feature size"
    for layer in m.layers
        z32 = layer(z32)
    end
    return m.proj === nothing ? z32 : m.proj(z32)
end

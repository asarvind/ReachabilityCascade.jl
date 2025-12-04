"""
    GLU(in => out; act = Flux.σ, bias = true)

Gated Linear Unit layer.

Computes `Y = A .* act(B)` where a single linear map produces `2*out` features
that are split along the feature dimension: `[A; B] = W*X .+ b`.

Arguments
- `in => out` : input and output feature sizes
- `act`       : gate activation (default `Flux.σ` = sigmoid; try `gelu`, `swish`, etc.)
- `bias`      : include bias term (default `true`)

Input shapes
- `x::AbstractVector` (length `in`) → output length `out`
- `x::AbstractMatrix`  of size `(in, N)` → output `(out, N)`

This layer is differentiable end-to-end and compatible with `Flux.params`,
`gpu`, saving/loading, etc.
"""
struct GLU{L,F}
    dense::L   # Dense(in, 2*out)
    act::F     # gate nonlinearity
end

# Register layer
Flux.@layer GLU

# Constructor: GLU(in => out; act = Flux.σ, bias = true)
function GLU(sizes::Pair{<:Integer,<:Integer}; act = Flux.σ, bias::Bool = true)
    in, out = first(sizes), last(sizes)
    d = Dense(in, 2*out; bias=bias)
    return GLU(d, act)
end

# Forward pass for vector, (features, batch) matrices, or (features, seq_len, batch) tensors.
function (m::GLU)(x::AbstractArray)
    # Ensure inputs are Float32 (useful if upstream data is Float64/Int; works on CPU/GPU)
    x32 = Float32.(x)
    if ndims(x32) == 1 || ndims(x32) == 2
        y = m.dense(x32)
        o = size(y, 1) ÷ 2
        A = @view y[1:o, :]
        B = @view y[o+1:end, :]
        return A .* m.act(B)
    elseif ndims(x32) == 3
        # Flatten time and batch, apply GLU, then reshape back.
        f, t, b = size(x32)
        x_flat = reshape(x32, f, t * b)
        y = m.dense(x_flat)
        o = size(y, 1) ÷ 2
        A = @view y[1:o, :]
        B = @view y[o+1:end, :]
        out_flat = A .* m.act(B)                 # (o, t*b)
        return reshape(out_flat, o, t, b)
    else
        error("Unsupported input dimensions for GLU: $(ndims(x))")
    end
end

# Handy size info: returns (in, out)
Base.size(m::GLU) = (size(m.dense.weight, 2), size(m.dense.weight, 1) ÷ 2)

# Pretty printing
function Base.show(io::IO, m::GLU)
    out = size(m.dense.weight, 1) ÷ 2
    in  = size(m.dense.weight, 2)
    print(io, "GLU($in ⇒ $out, act=$(m.act), bias=$(m.dense.bias !== nothing))")
end


"""
    glu_mlp(in_dim::Integer, hidden::Integer, out_dim::Integer;
            n_glu::Integer=2, act=Flux.σ, bias::Bool=true, zero_init::Bool=false)

Build a GLU-based multi-layer perceptron (MLP).
The network has `n_glu` GLU blocks followed by a final linear `Dense` layer
(projecting to `out_dim`). Accepts and returns arrays shaped `(features, batch)`.

# Arguments
- `in_dim`   : input feature size (`Int`)
- `hidden`   : hidden width for each GLU block (`Int`)
- `out_dim`  : output feature size (`Int`)

# Keywords
- `n_glu=1`  : number of GLU blocks before the final Dense
- `act`      : gate activation for each GLU (default `Flux.σ`)
- `bias=true`: include bias in dense layers
- `zero_init=false`: if true, initialize the final Dense to zeros.

# Returns
`Chain` consisting of `[GLU, ..., GLU, Dense]`.
"""
function glu_mlp(in_dim::Integer, hidden::Integer, out_dim::Integer;
                 n_glu::Integer=1, act=Flux.σ, bias::Bool=true, zero_init::Bool=false)
    layers = Any[]
    d = in_dim
    for _ in 1:max(n_glu, 0)
        push!(layers, GLU(d => hidden; act=act, bias=bias))
        push!(layers, Dense(hidden=>hidden, leakyrelu))
        d = hidden
    end

    if zero_init
        push!(layers, Dense(zeros(Float32, out_dim, d)))
    else
        push!(layers, Dense(d, out_dim; bias=bias))
    end

    return Chain(layers...)
end

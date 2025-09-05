# ============================ Utilities =====================================

"""
    glu_mlp(in_dim::Integer, hidden::Integer, out_dim::Integer;
            n_glu::Integer=2, act=Flux.σ, bias::Bool=true)

Build a compact GLU-based MLP to use as a conditioner inside coupling layers.
The network has `n_glu` GLU blocks followed by a final linear `Dense` layer
(projecting to `out_dim`). Accepts and returns arrays shaped `(features, batch)`.

# Arguments
- `in_dim`   : input feature size (`Int`)
- `hidden`   : hidden width for each GLU block (`Int`)
- `out_dim`  : output feature size (`Int`)

# Keywords
- `n_glu=2`  : number of GLU blocks before the final Dense
- `act`      : gate activation for each GLU (default `Flux.σ`)
- `bias=true`: include bias in dense layers

# Returns
`Chain` consisting of `[GLU, ..., GLU, Dense]`.
"""
function glu_mlp(in_dim::Integer, hidden::Integer, out_dim::Integer;
                 n_glu::Integer=2, act=Flux.σ, bias::Bool=true)
    layers = Any[]
    d = in_dim
    for _ in 1:max(n_glu, 0)
        push!(layers, GLU(d => hidden; act=act, bias=bias))
        d = hidden
    end
    push!(layers, Dense(d, out_dim; bias=bias))
    return Chain(layers...)
end

"""
    softclamp(x; limit=3.0f0)

Apply a smooth clamp `limit * tanh(x/limit)` elementwise to keep values within
`[-limit, limit]`. Useful for stabilizing predicted log-scales.

# Keywords
- `limit` : positive scalar clamp magnitude (default `5.0f0`)

# Returns
Array with the same shape as `x`.
"""
softclamp(x; limit=2.0f0) = limit .* tanh.(x ./ limit)

"""
    _bexpand(mask, B)

Broadcast helper that repeats a `(D, 1)` mask along the batch dimension to
match `(D, B)`. If `size(mask, 2) == B`, returns `mask` unchanged.
"""
_bexpand(mask::AbstractVecOrMat, B::Integer) = size(mask, 2) == B ? mask : repeat(mask, 1, B)
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
    softclamp(x; limit=5.0f0)

Apply a smooth clamp `limit * tanh(x/limit)` elementwise to keep values within
`[-limit, limit]`. Useful for stabilizing predicted log-scales.

# Keywords
- `limit` : positive scalar clamp magnitude (default `5.0f0`)

# Returns
Array with the same shape as `x`.
"""
softclamp(x; limit=5.0f0) = limit .* tanh.(x ./ limit)

"""
    _bexpand(mask, B)

Broadcast helper that repeats a `(D, 1)` mask along the batch dimension to
match `(D, B)`. If `size(mask, 2) == B`, returns `mask` unchanged.
"""
_bexpand(mask::AbstractVecOrMat, B::Integer) = size(mask, 2) == B ? mask : repeat(mask, 1, B)

# ===================== Affine Coupling with GLU ==============================

"""
    Permute

A lightweight, differentiable *layer* that reorders feature dimensions using a
fixed permutation `perm`. Supports an inverse call via `inverse=true`.

This is useful to factor non-parametric shuffles out of parametric layers so
that permutation logic is testable and reusable.

# Fields
- `perm::Vector{Int}` : forward permutation indices
- `invp::Vector{Int}` : inverse permutation indices (`invperm(perm)`) 

# Call
```
Y = p(X)                     # forward: X[perm, :]
X = p(Y; inverse=true)       # inverse: Y[invp, :]
```
"""
struct Permute
    perm::Vector{Int}
    invp::Vector{Int}
end
Flux.@layer Permute

"""
    Permute(perm)

Construct a `Permute` layer from a permutation vector. Precomputes the inverse
permutation for efficient inverse calls.
"""
Permute(perm::AbstractVector{<:Integer}) = Permute(collect(Int, perm), invperm(perm))

"""
    (p::Permute)(x; inverse=false)

Apply the permutation (or its inverse) to `x` along the feature dimension.
Expects `x` shaped `(D, B)`.
"""
function (p::Permute)(x::AbstractMatrix; inverse::Bool=false)
    if inverse
        return @views x[p.invp, :]
    else
        return @views x[p.perm, :]
    end
end

"""
    AffineCouplingGLU

Affine coupling transform whose conditioner networks are GLU-based MLPs and
which supports *context conditioning*. It uses a separate `Permute` layer to
reorder features so the pass-through block precedes the transformed block.

# Fields
- `mask::BitVector` : length `D`; `true` = pass-through, `false` = transformed
- `s_net::Chain`    : conditioner network producing `log_s` (size `D_t`)
- `t_net::Chain`    : conditioner network producing `t`     (size `D_t`)
- `ctx_dim::Int`    : dimensionality of the context vector `c`
- `perm_layer::Permute` : feature permutation layer (stores forward + inverse indices)
- `Dp::Int`         : number of pass-through dims
"""
struct AffineCouplingGLU
    mask::BitVector
    s_net::Chain
    t_net::Chain
    ctx_dim::Int
    perm_layer::Permute
    Dp::Int
end

# Register as a Flux layer (no parameters changed by this macro; it enables nice printing and traversal)
Flux.@layer AffineCouplingGLU

"""
    AffineCouplingGLU(mask, x_dim, ctx_dim; hidden=128, n_glu=2, bias=true)

Construct an `AffineCouplingGLU` layer.

# Arguments
- `mask::AbstractVector{Bool}` : length `x_dim`; `true` dims are pass-through
- `x_dim::Int`                 : data dimensionality `D`
- `ctx_dim::Int`               : context dimensionality

# Keywords
- `hidden=128` : hidden width of conditioner GLU stacks
- `n_glu=2`    : number of GLU blocks in each conditioner
- `bias=true`  : include bias terms in dense layers

# Returns
An `AffineCouplingGLU` with GLU conditioners for `log_s` and `t`.
"""
function AffineCouplingGLU(mask::AbstractVector{Bool}, x_dim::Integer, ctx_dim::Integer;
                           hidden::Integer=128, n_glu::Integer=2, bias::Bool=true)
    @assert length(mask) == x_dim "mask length must equal x_dim"
    pass_idx = findall(identity, mask)
    trans_idx = findall(!, mask)
    Dp = length(pass_idx)
    in_cond = Dp + ctx_dim
    s_net = glu_mlp(in_cond, hidden, length(trans_idx); n_glu=n_glu, bias=bias)
    t_net = glu_mlp(in_cond, hidden, length(trans_idx); n_glu=n_glu, bias=bias)
    perm = vcat(pass_idx, trans_idx)
    pl = Permute(perm)
    return AffineCouplingGLU(BitVector(mask), s_net, t_net, ctx_dim, pl, Dp)
end

"""
    (m::AffineCouplingGLU)(x, c; inverse=false)

Apply the affine coupling transform (forward or inverse) with context.

# Arguments
- `x::AbstractMatrix` : input of shape `(D, B)`
- `c::AbstractMatrix` : context of shape `(ctx_dim, B)`

# Keywords
- `inverse=false` : if `false`, compute forward (encode) `y = f(x,c)` and return
  `(y, logdet)`; if `true`, compute inverse (decode) `x = f^{-1}(y,c)` and return
  `(x, -logdet)`

# Returns
A tuple `(y_or_x, logdet_vec)` where `logdet_vec` is a length-`B` vector of
per-sample log-absolute-determinants.
"""
function (m::AffineCouplingGLU)(x::AbstractArray, c::AbstractArray; inverse::Bool=false)
    D, B = size(x)
    @assert size(c, 2) == B "context batch must match x batch"
    @assert size(c, 1) == m.ctx_dim "context feature size mismatch"

    # Reorder x → [x_pass; x_trans]
    xperm = m.perm_layer(x)             # (D, B)
    xp = @views xperm[1:m.Dp, :]
    xt = @views xperm[m.Dp+1:end, :]

    # Conditioner input is [x_pass; c]
    h = vcat(xp, c)
    log_s = softclamp(m.s_net(h); limit=5.0f0)
    t     = m.t_net(h)

    if !inverse
        yt = xt .* exp.(log_s) .+ t               # forward on transformed block
        yperm = vcat(xp, yt)                      # concatenate back
        y = m.perm_layer(yperm; inverse=true)     # undo permutation
        ld = vec(sum(log_s; dims=1))              # transformed dims only
        return y, ld
    else
        yt = xt                                   # treat input as y in permuted order
        xt_rec = (yt .- t) .* exp.(-log_s)        # inverse on transformed block
        xperm_rec = vcat(xp, xt_rec)
        xrec = m.perm_layer(xperm_rec; inverse=true)
        ld = -vec(sum(log_s; dims=1))
        return xrec, ld
    end
end

# =========================== Flow Container ==================================

"""
    ConditionalFlow

A conditional normalizing flow composed of multiple `AffineCouplingGLU` blocks
with alternating masks, plus a **non-differentiable per-feature scale** vector
`scaling` (length `x_dim`).

The scaling multiplies features elementwise (a fixed diagonal transform) and
contributes to the log-determinant. It is intentionally **excluded from training**
via a `Flux.trainable` override.

# Fields
- `blocks::Vector{AffineCouplingGLU}` : sequence of coupling layers
- `x_dim::Int`                        : data dimensionality `D`
- `ctx_dim::Int`                      : context dimensionality
- `scaling::AbstractVector{<:Real}`   : per-feature non-differentiable scaling
"""
struct ConditionalFlow{B}
    blocks::Vector{B}
    x_dim::Int
    ctx_dim::Int
    scaling::Vector{Float32}
end

# Register as a Flux layer container
Flux.@layer ConditionalFlow

"""
    Flux.trainable(cf::ConditionalFlow)

Exclude the `scaling` vector from the trainable parameters so it remains
non-differentiable. Only the parameters inside `blocks` are optimized.
"""
Flux.trainable(cf::ConditionalFlow) = (; blocks = cf.blocks)

"""
    _make_alternating_masks(D::Integer, n_blocks::Integer)

Internal helper to generate `n_blocks` boolean masks of length `D` that
alternate pass-through indices: `T F T F ...` for odd blocks and its complement
for even blocks.

# Returns
`Vector{BitVector}` of length `n_blocks`.
"""
function _make_alternating_masks(D::Integer, n_blocks::Integer)
    masks = Vector{BitVector}(undef, n_blocks)
    base1 = falses(D);  base1[1:2:D] .= true   # T F T F ...
    base2 = .!base1                            # F T F T ...
    for i in 1:n_blocks
        masks[i] = isodd(i) ? copy(base1) : copy(base2)
    end
    return masks
end

"""
    ConditionalFlow(x_dim::Integer, ctx_dim::Integer;
                    n_blocks::Integer=6, hidden::Integer=128,
                    n_glu::Integer=2, bias::Bool=true,
                    scaling::AbstractVector{<:Real}=ones(Float32, x_dim))

Build a conditional normalizing flow with GLU-based affine couplings and a
fixed, non-differentiable per-feature `scaling` vector. The scaling is applied
**before** the coupling stack in `encode` and inverted in `decode`. Its
log-determinant contribution is added/subtracted accordingly.

# Arguments
- `x_dim`  : dimensionality of data `x`
- `ctx_dim`: dimensionality of context `c`

# Keywords
- `n_blocks=6` : number of coupling blocks
- `hidden=128` : hidden width for conditioner GLU stacks
- `n_glu=2`    : number of GLU layers per conditioner
- `bias=true`  : include bias terms in dense layers
- `scaling`    : length-`x_dim` vector; **must be nonzero** per entry

# Returns
A `ConditionalFlow` instance.
"""
function ConditionalFlow(x_dim::Integer, ctx_dim::Integer;
                         n_blocks::Integer=6, hidden::Integer=128,
                         n_glu::Integer=2, bias::Bool=true,
                         scaling::AbstractVector{<:Real}=ones(Float32, x_dim))
    @assert length(scaling) == x_dim "scaling length must equal x_dim"
    @assert all(abs.(scaling) .> 0) "scaling must be nonzero for invertibility"
    masks = _make_alternating_masks(x_dim, n_blocks)
    blocks = [AffineCouplingGLU(m, x_dim, ctx_dim; hidden=hidden, n_glu=n_glu, bias=bias)
              for m in masks]
    return ConditionalFlow{eltype(blocks)}(blocks, x_dim, ctx_dim, Float32.(scaling))
end

# ------------------------------- Encode --------------------------------------

"""
    encode(flow::ConditionalFlow, x::AbstractVecOrMat, c::AbstractVecOrMat)

Forward pass of the flow: map data to latent `z = f(x,c)` and accumulate
per-sample log|det J|. Applies the non-diff `scaling` first.

# Arguments
- `flow::ConditionalFlow` : Conditional flow neural net
- `x::AbstractVecOrMat`   : Sample array to be encoded `(D,)` or `(D,B)`
- `c::AbstractVecOrMat`   : Context array `(C,)` or `(C,B)`

# Returns
`(z, logdet)` where `z` is `(D, B)` and `logdet` is a length-`B` vector.
"""
function encode(flow::ConditionalFlow, x::AbstractVecOrMat, c::AbstractVecOrMat)
    @assert size(x, 1) == flow.x_dim
    @assert size(c, 1) == flow.ctx_dim
    xarr = ndims(x) == 1 ? reshape(x, size(x,1), 1) : x
    carr = ndims(c) == 1 ? reshape(c, size(c,1), 1) : c

    # Apply fixed scaling first (diagonal transform)
    z = flow.scaling .* xarr
    ld = sum(log.(abs.(flow.scaling)))            # scalar
    logdet = fill(Float32(ld), size(z, 2))        # replicate for batch

    for blk in flow.blocks
        z, ld_blk = blk(z, carr; inverse=false)
        logdet = logdet .+ ld_blk
    end
    return z, logdet
end

# ------------------------------- Decode --------------------------------------

"""
    decode(flow::ConditionalFlow, z::AbstractVecOrMat, c::AbstractVecOrMat)

Inverse pass of the flow: map latent to data `x = f^{-1}(z,c)` and accumulate
per-sample log|det J| contributed by the inverse mappings. Applies inverse
scaling at the end.

# Arguments
- `flow::ConditionalFlow` : Conditional flow neural net
- `z::AbstractVecOrMat`   : Sample array to be decoded `(D,)` or `(D,B)`
- `c::AbstractVecOrMat`   : Context array `(C,)` or `(C,B)`

# Returns
`(x, logdet)` where `x` is `(D, B)` and `logdet` is a length-`B` vector.
"""
function decode(flow::ConditionalFlow, z::AbstractVecOrMat, c::AbstractVecOrMat)
    @assert size(z, 1) == flow.x_dim
    @assert size(c, 1) == flow.ctx_dim
    zarr = ndims(z) == 1 ? reshape(z, size(z,1), 1) : z
    carr = ndims(c) == 1 ? reshape(c, size(c,1), 1) : c

    x = zarr
    logdet = zeros(eltype(x), size(x, 2))
    for blk in Iterators.reverse(flow.blocks)
        x, ld_blk = blk(x, carr; inverse=true)
        logdet = logdet .+ ld_blk
    end

    # Invert fixed scaling last
    x = x ./ flow.scaling
    ld = sum(log.(abs.(flow.scaling)))            # scalar
    logdet = logdet .- Float32(ld)
    return x, logdet
end

"""
    (flow::ConditionalFlow)(x::AbstractVecOrMat, c::AbstractVecOrMat; inverse::Bool=false)

Forward pass of the neural net for either encoding (`inverse=false`) or decoding
(`inverse=true`). Handles vector or matrix inputs for convenience.

# Returns
`(y, logdet)` where `y` is the mapped tensor and `logdet` is a batch-length vector.
"""
function (flow::ConditionalFlow)(x::AbstractVecOrMat, c::AbstractVecOrMat; inverse::Bool=false)
    return inverse ? decode(flow, x, c) : encode(flow, x, c)
end


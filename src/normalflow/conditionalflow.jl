# =========================== Flow Container ==================================

"""
    ConditionalFlow

A conditional normalizing flow composed of multiple `AffineCouplingGLU` blocks
with alternating masks, plus two **non-differentiable per-feature scale** vectors:
`x_scaling` for the sample vector and `c_scaling` for the context vector.

Both scalings multiply features elementwise (fixed diagonal transforms) and
contribute to the log-determinant. They are intentionally **excluded from
training** via a `Flux.trainable` override.

# Fields
- `blocks::Vector{AffineCouplingGLU}` : sequence of coupling layers
- `x_dim::Int`                        : data dimensionality `D`
- `ctx_dim::Int`                      : context dimensionality
- `x_scaling::AbstractVector{Float32}`        : per-feature non-diff scaling for samples
- `c_scaling::AbstractVector{Float32}`        : per-feature non-diff scaling for context
- `clamp_lim::Real`                       : log-saturation limit on scaling blocks of affine coupling 
"""
struct ConditionalFlow{B}
    blocks::Vector{B}
    x_dim::Int
    ctx_dim::Int
    x_scaling::AbstractVector{Float32}
    c_scaling::AbstractVector{Float32}
    clamp_lim::Real
end

# Register as a Flux layer container
Flux.@layer ConditionalFlow

"""
    Flux.trainable(cf::ConditionalFlow)

Exclude the scaling vectors from the trainable parameters so they remain
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
                    x_scaling::AbstractVector{<:Real}=ones(Float32, x_dim),
                    c_scaling::AbstractVector{<:Real}=ones(Float32, ctx_dim)),
                    clamp_lim::Real)

Build a conditional normalizing flow with GLU-based affine couplings and fixed,
non-differentiable per-feature scaling vectors for both samples and context.
The scalings are applied **before** the coupling stack in `encode` and inverted
in `decode`. Their log-determinant contributions are added/subtracted
accordingly.

# Arguments
- `x_dim`  : dimensionality of data `x`
- `ctx_dim`: dimensionality of context `c`

# Keywords
- `n_blocks=6` : number of coupling blocks
- `hidden=128` : hidden width for conditioner GLU stacks
- `n_glu=2`    : number of GLU layers per conditioner
- `bias=true`  : include bias terms in dense layers
- `x_scaling`  : length-`x_dim` vector; must be nonzero per entry
- `c_scaling`  : length-`ctx_dim` vector; must be nonzero per entry
- `clamp_lim`      : real scalar limit on log-saturation of scaling in affine coupling blocks

# Returns
A `ConditionalFlow` instance.
"""
function ConditionalFlow(x_dim::Integer, ctx_dim::Integer;
                         n_blocks::Integer=6, hidden::Integer=128,
                         n_glu::Integer=2, bias::Bool=true,
                         x_scaling::AbstractVector{<:Real}=ones(Float32, x_dim),
                         c_scaling::AbstractVector{<:Real}=ones(Float32, ctx_dim), 
                         clamp_lim::Real=3.0)
    @assert length(x_scaling) == x_dim "x_scaling length must equal x_dim"
    @assert length(c_scaling) == ctx_dim "c_scaling length must equal ctx_dim"
    @assert all(abs.(x_scaling) .> 0) "x_scaling must be nonzero for invertibility"
    @assert all(abs.(c_scaling) .> 0) "c_scaling must be nonzero for invertibility"
    masks = _make_alternating_masks(x_dim, n_blocks)
    blocks = [AffineCouplingGLU(m, x_dim, ctx_dim; hidden=hidden, n_glu=n_glu, bias=bias)
              for m in masks]
    return ConditionalFlow{eltype(blocks)}(blocks, x_dim, ctx_dim,
                                           Float32.(x_scaling), Float32.(c_scaling), Float32.(clamp_lim))
end

# ------------------------------- Encode --------------------------------------

"""
    encode(flow::ConditionalFlow, x::AbstractVecOrMat, c::AbstractVecOrMat)

Forward pass of the flow: map data to latent `z = f(x,c)` and accumulate
per-sample log|det J|. Applies the non-diff scalings first.

# Arguments
- `flow::ConditionalFlow` : Conditional flow neural net
- `x::AbstractVecOrMat`   : Sample array to be encoded `(D,)` or `(D,B)`
- `c::AbstractVecOrMat`   : Context array `(C,)` or `(C,B)`

# Returns
Named tuple `(latent=z, logdet=logdet)` where `latent` is `(D, B)` and `logdet`
is a length-`B` vector.
"""
function encode(flow::ConditionalFlow, x::AbstractVecOrMat, c::AbstractVecOrMat)
    @assert size(x, 1) == flow.x_dim
    @assert size(c, 1) == flow.ctx_dim
    xarr = ndims(x) == 1 ? reshape(x, size(x,1), 1) : x
    carr = ndims(c) == 1 ? reshape(c, size(c,1), 1) : c

    # Apply fixed scalings first
    z = flow.x_scaling .* xarr
    cscaled = flow.c_scaling .* carr
    ld = sum(log.(abs.(flow.x_scaling)))         # sample logdet
    logdet = fill(Float32(ld), size(z, 2))

    for blk in flow.blocks
        z, ld_blk = blk(z, cscaled; inverse=false, clamp_lim=flow.clamp_lim)
        logdet = logdet .+ ld_blk
    end
    return (latent = z, logdet = logdet)
end

# ------------------------------- Decode --------------------------------------

"""
    decode(flow::ConditionalFlow, z::AbstractVecOrMat, c::AbstractVecOrMat)

Inverse pass of the flow: map latent to data `x = f^{-1}(z,c)` while applying
inverse scalings at the end. Log-determinant contributions are not returned.

# Arguments
- `flow::ConditionalFlow` : Conditional flow neural net
- `z::AbstractVecOrMat`   : Sample array to be decoded `(D,)` or `(D,B)`
- `c::AbstractVecOrMat`   : Context array `(C,)` or `(C,B)`

# Returns
Decoded tensor `x` with shape `(D, B)`.
"""
function decode(flow::ConditionalFlow, z::AbstractVecOrMat, c::AbstractVecOrMat)
    @assert size(z, 1) == flow.x_dim
    @assert size(c, 1) == flow.ctx_dim
    zarr = ndims(z) == 1 ? reshape(z, size(z,1), 1) : z
    carr = ndims(c) == 1 ? reshape(c, size(c,1), 1) : c

    x = zarr
    cscaled = flow.c_scaling .* carr
    for blk in Iterators.reverse(flow.blocks)
        x, _ = blk(x, cscaled; inverse=true, clamp_lim=flow.clamp_lim)
    end

    # Invert fixed x_scaling last
    x = x ./ flow.x_scaling
    return x
end

"""
    (flow::ConditionalFlow)(x::AbstractVecOrMat, c::AbstractVecOrMat; inverse::Bool=false)

Forward pass of the neural net for either encoding (`inverse=false`) or decoding
(`inverse=true`). Handles vector or matrix inputs for convenience.

# Returns
- When `inverse=false`: named tuple `(latent, logdet)`
- When `inverse=true` : decoded tensor `x`
"""
function (flow::ConditionalFlow)(x::AbstractVecOrMat, c::AbstractVecOrMat; inverse::Bool=false)
    return inverse ? decode(flow, x, c) : encode(flow, x, c)
end

# ============================ Example (optional) =============================
# using Random
# D, C, B = 5, 2, 4
# flow = ConditionalFlow(D, C; n_blocks=4, hidden=64, n_glu=2,
#                        x_scaling=fill(2f0, D), c_scaling=fill(0.5f0, C))
# x = randn(Float32, D, B); c = randn(Float32, C, B)
# out = encode(flow, x, c)
# xr = decode(flow, out.latent, c)
# @show maximum(abs.(xr .- x))

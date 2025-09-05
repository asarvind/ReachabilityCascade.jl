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
- `x_scaling::Vector{Float32}`        : per-feature non-diff scaling for samples
- `c_scaling::Vector{Float32}`        : per-feature non-diff scaling for context
"""
struct ConditionalFlow{B}
    blocks::Vector{B}
    x_dim::Int
    ctx_dim::Int
    x_scaling::Vector{Float32}
    c_scaling::Vector{Float32}
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
                    c_scaling::AbstractVector{<:Real}=ones(Float32, ctx_dim))

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

# Returns
A `ConditionalFlow` instance.
"""
function ConditionalFlow(x_dim::Integer, ctx_dim::Integer;
                         n_blocks::Integer=6, hidden::Integer=128,
                         n_glu::Integer=2, bias::Bool=true,
                         x_scaling::AbstractVector{<:Real}=ones(Float32, x_dim),
                         c_scaling::AbstractVector{<:Real}=ones(Float32, ctx_dim))
    @assert length(x_scaling) == x_dim "x_scaling length must equal x_dim"
    @assert length(c_scaling) == ctx_dim "c_scaling length must equal ctx_dim"
    @assert all(abs.(x_scaling) .> 0) "x_scaling must be nonzero for invertibility"
    @assert all(abs.(c_scaling) .> 0) "c_scaling must be nonzero for invertibility"
    masks = _make_alternating_masks(x_dim, n_blocks)
    blocks = [AffineCouplingGLU(m, x_dim, ctx_dim; hidden=hidden, n_glu=n_glu, bias=bias)
              for m in masks]
    return ConditionalFlow{eltype(blocks)}(blocks, x_dim, ctx_dim,
                                           Float32.(x_scaling), Float32.(c_scaling))
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
`(z, logdet)` where `z` is `(D, B)` and `logdet` is a length-`B` vector.
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
    # ldc = sum(log.(abs.(flow.c_scaling)))      # context logdet (not added to z)
    logdet = fill(Float32(ld), size(z, 2))

    for blk in flow.blocks
        z, ld_blk = blk(z, cscaled; inverse=false)
        logdet = logdet .+ ld_blk
    end
    return z, logdet
end

# ------------------------------- Decode --------------------------------------

"""
    decode(flow::ConditionalFlow, z::AbstractVecOrMat, c::AbstractVecOrMat)

Inverse pass of the flow: map latent to data `x = f^{-1}(z,c)` and accumulate
per-sample log|det J| contributed by the inverse mappings. Applies inverse
scalings at the end.

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
    cscaled = flow.c_scaling .* carr
    for blk in Iterators.reverse(flow.blocks)
        x, ld_blk = blk(x, cscaled; inverse=true)
        logdet = logdet .+ ld_blk
    end

    # Invert fixed x_scaling last
    x = x ./ flow.x_scaling
    ld = sum(log.(abs.(flow.x_scaling)))         # scalar
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

# ------------------------------- Likelihoods ---------------------------------
"""
    loglikelihoods(flow::ConditionalFlow, x::AbstractVecOrMat, c::AbstractVecOrMat;
                   inverse::Bool=false)

Compute **per-sample log-likelihoods** under a standard normal prior in latent
space. Returns a length-`B` vector.

- When `inverse=false` (default): encodes `x` to `z`, and returns
  `log p(z) + log|det J|`.
- When `inverse=true`: treats the input as *already latent* `z`, so it returns
  only `log p(z)` (no Jacobian term), since `z` is in the prior space.

`c` is scaled using `c_scaling` in the same way as the main encode/decode paths.
"""
function loglikelihoods(flow::ConditionalFlow, x::AbstractVecOrMat, c::AbstractVecOrMat;
                        inverse::Bool=false)
    if inverse
        # x is actually z in latent space
        z = ndims(x) == 1 ? reshape(x, size(x,1), 1) : x
        D = size(z, 1)
        ll = -0.5f0 .* sum(z.^2; dims=1) .- (D/2) .* log(2f0*pi)
        return vec(ll)
    else
        z, logdet = flow(x, c)
        D = size(z, 1)
        ll_prior = -0.5f0 .* sum(z.^2; dims=1) .- (D/2) .* log(2f0*pi)
        ll = vec(ll_prior) .+ logdet
        return ll
    end
end

# ============================ Example (optional) =============================
# using Random
# D, C, B = 5, 2, 4
# flow = ConditionalFlow(D, C; n_blocks=4, hidden=64, n_glu=2,
#                        x_scaling=fill(2f0, D), c_scaling=fill(0.5f0, C))
# x = randn(Float32, D, B); c = randn(Float32, C, B)
# z, ld1 = encode(flow, x, c)
# xr, ld2 = decode(flow, z, c)
# @show maximum(abs.(xr .- x))
# @show mean(ld1 .+ ld2)
# ll = loglikelihoods(flow, x, c)
# ll_from_latent = loglikelihoods(flow, z, c; inverse=true)
# @show size(ll), size(ll_from_latent)

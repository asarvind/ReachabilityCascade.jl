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
    log_s = softclamp(m.s_net(h))
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

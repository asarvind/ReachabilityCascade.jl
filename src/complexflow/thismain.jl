# Complex-Conditioned Invertible Flow with GLU Couplings (Julia/Flux)
# ------------------------------------------------------------------
# Invertible flows over *complex-valued* vectors with affine coupling layers and
# GLU-based conditioner networks. Optionally conditioned on a (real) context
# vector. This version uses **native complex weights** via Flux `Dense(W, b, σ)`
# where `W` and `b` are complex arrays.
#
# ✅ Deterministic encode/decode only (no log-determinants)
# ✅ Uses `Flux.@layer`
# ✅ Operates on `Number`-typed arrays to allow both real and complex
# ✅ Accepts **vectors or matrices** (`AbstractVecOrMat`) for inputs/outputs
# ✅ Affine coupling initialized to volume-preserving (scaling output ~ 0)
# ✅ **No in-place mutation** (Zygote-friendly)
#
# Public API
#   flow = ComplexFlow(d; ctx_dim=0, num_blocks=4, hidden=128, depth=2, clamp=3.0, ln=false, seed=nothing)
#   Y = encode(flow, X[, C])    # X: (d,) or (d,N) <:Number ;  C: (ctx_dim,) or (ctx_dim,N) <:Real (optional)
#   X = decode(flow, Y[, C])    # Y: (d,) or (d,N) <:Number

# ------------------------------------------------------------------
# Small shape helpers to support vectors or matrices uniformly
# ------------------------------------------------------------------

_isvec(A) = A isa AbstractVector
_to_mat(A::AbstractVector) = reshape(A, :, 1)
_to_mat(A::AbstractMatrix) = A
_ncols(A::AbstractVector) = 1
_ncols(A::AbstractMatrix) = size(A, 2)
_maybe_vec(A::AbstractMatrix, wasvec::Bool) = wasvec ? vec(A) : A

# ------------------------------------------------------------------
# ComplexDense: wrapper around Flux.Dense with complex weights
# ------------------------------------------------------------------

"""
    ComplexDense(in_dim, out_dim; bias=true, act=identity, initW, initb)

A Flux layer backed by `Dense(W,b,act)` where `W` and `b` are complex arrays.
Inputs and outputs are matrices/vectors with element type <:Number.
"""
struct ComplexDense
    dense::Dense
end
@layer ComplexDense

function ComplexDense(in_dim::Integer, out_dim::Integer;
                      bias::Bool=true, act=identity,
                      initW = () -> (randn(ComplexF32, out_dim, in_dim) ./ sqrt(ComplexF32(in_dim))),
                      initb = () -> (bias ? zeros(ComplexF32, out_dim) : ComplexF32[]))
    W = initW()
    b = initb()
    d = Dense(W, b, act)
    return ComplexDense(d)
end

# Accept vector or matrix
(c::ComplexDense)(X::AbstractVecOrMat{<:Number}) = c.dense(X)

# ------------------------------------------------------------------
# GLU Block (complex features, real gate) — no in-place ops
# ------------------------------------------------------------------

struct CxGLUBlock
    Wy::ComplexDense
    Wg::ComplexDense
    ln::Union{LayerNorm, Nothing}
    φ::Function
end
@layer CxGLUBlock

function CxGLUBlock(d_in::Integer, d_out::Integer; ln::Bool=false, φ = identity)
    Wy = ComplexDense(d_in, d_out)
    Wg = ComplexDense(d_in, d_out)
    ln_layer = ln ? LayerNorm(d_out) : nothing
    return CxGLUBlock(Wy, Wg, ln_layer, φ)
end

function (g::CxGLUBlock)(X::AbstractVecOrMat{<:Number})
    Y = g.Wy(X)
    G = g.Wg(X)
    if g.ln !== nothing
        Yr = g.ln(real.(Y))
        Yi = g.ln(imag.(Y))
        Y  = complex.(Yr, Yi)
    end
    gate = sigmoid.(real.(G))
    return g.φ.(Y) .* gate
end

# ------------------------------------------------------------------
# GLU Network — no in-place ops
# ------------------------------------------------------------------

struct CxGLUNet
    layers::Vector{Any}
    proj::ComplexDense
end
@layer CxGLUNet

function CxGLUNet(in_dim::Integer, out_dim::Integer; hidden::Integer=256, depth::Integer=2, ln::Bool=false, zero_init::Bool=false)
    layers = Any[]
    dcur = in_dim
    for _ in 1:depth
        push!(layers, CxGLUBlock(dcur, hidden; ln=ln))
        dcur = hidden
    end
    if zero_init
        proj = ComplexDense(dcur, out_dim;
                            initW = () -> zeros(ComplexF32, out_dim, dcur),
                            initb = () -> zeros(ComplexF32, out_dim))
    else
        proj = ComplexDense(dcur, out_dim)
    end
    return CxGLUNet(layers, proj)
end

function (net::CxGLUNet)(X::AbstractVecOrMat{<:Number})
    H = X
    for l in net.layers
        H = l(H)
    end
    return net.proj(H)
end

# ------------------------------------------------------------------
# Affine Coupling with GLU conditioner — no in-place ops
# ------------------------------------------------------------------

struct AffineCouplingGLU
    mask::Vector{Float32}
    x_dim::Int
    ctx_dim::Int
    conditioner::CxGLUNet
    clamp::Float32
    scale_from_mean::Bool
end
@layer AffineCouplingGLU

function AffineCouplingGLU(mask::AbstractVector{<:Real}, x_dim::Integer;
                           ctx_dim::Integer,
                           hidden::Integer=256, depth::Integer=2,
                           clamp::Real=3.0, ln::Bool=false,
                           scale_from_mean::Bool=true)
    @assert length(mask) == x_dim
    maskf = Float32.(mask)
    nT = count(==(0f0), maskf)
    in_dim = x_dim + ctx_dim
    # For affine coupling: output 2*nT; scaling head starts at zeros (identity)
    conditioner = CxGLUNet(in_dim, 2*nT; hidden=hidden, depth=depth, ln=ln, zero_init=true)
    return AffineCouplingGLU(maskf, x_dim, ctx_dim, conditioner, Float32(clamp), scale_from_mean)
end

softclamp(x, α) = α * tanh.(x / α)

# helper to build a one-hot selector matrix (no mutation)
_selector_matrix(idxs::Vector{Int}, d::Int) = [i == idxs[j] ? one(Float32) : 0f0 for i in 1:d, j in 1:length(idxs)]

function (ac::AffineCouplingGLU)(X::AbstractVecOrMat{<:Number}, C::AbstractVecOrMat{<:Real})
    Xwasvec = _isvec(X); Xm = _to_mat(X); Cm = _to_mat(C)
    @assert size(Xm,1) == ac.x_dim
    @assert size(Cm,1) == ac.ctx_dim
    @assert _ncols(Xm) == _ncols(Cm)

    N = _ncols(Xm)
    M = repeat(reshape(ac.mask, :, 1), 1, N)
    Xpass = M .* Xm

    Cc = complex.(Cm, zero(eltype(Cm)))
    H  = ac.conditioner(vcat(Xpass, Cc))

    nT = count(==(0f0), ac.mask)
    S_head = @view H[1:nT, :]
    T      = @view H[nT+1:end, :]

    S = real.(S_head)
    S = ac.scale_from_mean ? (S .- mean(S; dims=2)) : S
    S = softclamp(S, ac.clamp)

    # Scatter without mutation: build selector matrix E and multiply
    idxT = findall(x -> x == 0f0, ac.mask)
    E    = _selector_matrix(idxT, ac.x_dim)                # (d, nT)
    Sf   = E * S                                           # (d, N)
    Tf   = E * T                                           # (d, N)

    Y = Xpass .+ (1 .- M) .* ( Xm .* exp.(Sf) .+ Tf )
    return _maybe_vec(Y, Xwasvec)
end

function inverse(ac::AffineCouplingGLU, Y::AbstractVecOrMat{<:Number}, C::AbstractVecOrMat{<:Real})
    Ywasvec = _isvec(Y); Ym = _to_mat(Y); Cm = _to_mat(C)
    @assert size(Ym,1) == ac.x_dim
    @assert size(Cm,1) == ac.ctx_dim
    @assert _ncols(Ym) == _ncols(Cm)

    N = _ncols(Ym)
    M = repeat(reshape(ac.mask, :, 1), 1, N)
    Ypass = M .* Ym

    Cc = complex.(Cm, zero(eltype(Cm)))
    H  = ac.conditioner(vcat(Ypass, Cc))

    nT = count(==(0f0), ac.mask)
    S_head = @view H[1:nT, :]
    T      = @view H[nT+1:end, :]

    S = real.(S_head)
    S = ac.scale_from_mean ? (S .- mean(S; dims=2)) : S
    S = softclamp(S, ac.clamp)

    idxT = findall(x -> x == 0f0, ac.mask)
    E    = _selector_matrix(idxT, ac.x_dim)
    Sf   = E * S
    Tf   = E * T

    X = Ypass .+ (1 .- M) .* ( (Ym .- Tf) .* exp.(-Sf) )
    return _maybe_vec(X, Ywasvec)
end

# ------------------------------------------------------------------
# Permute Channels
# ------------------------------------------------------------------

struct PermuteChannels
    perm::Vector{Int}
    invperm::Vector{Int}
end
@layer PermuteChannels

function PermuteChannels(n::Integer; rng=Random.default_rng())
    p = collect(1:n)
    Random.shuffle!(rng, p)
    ip = invperm(p)
    return PermuteChannels(p, ip)
end

(P::PermuteChannels)(X::AbstractVecOrMat) = _isvec(X) ? X[P.perm] : X[P.perm, :]
function inverse(P::PermuteChannels, Y::AbstractVecOrMat)
    return _isvec(Y) ? Y[P.invperm] : Y[P.invperm, :]
end

# ------------------------------------------------------------------
# Flow container
# ------------------------------------------------------------------

struct ComplexFlow
    layers::Vector{Any}
    ctx_dim::Int
end
@layer ComplexFlow

function ComplexFlow(d::Integer; ctx_dim::Integer=0, num_blocks::Integer=4,
                     hidden::Integer=256, depth::Integer=2, clamp::Real=3.0,
                     ln::Bool=false, seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    m1, m2 = alternating_masks(d)
    layers = Any[]
    for b in 1:num_blocks
        mask = isodd(b) ? m1 : m2
        push!(layers, AffineCouplingGLU(mask, d; ctx_dim=ctx_dim,
                                        hidden=hidden, depth=depth,
                                        clamp=clamp, ln=ln))
        push!(layers, PermuteChannels(d))
    end
    pop!(layers)
    return ComplexFlow(layers, ctx_dim)
end

function encode(flow::ComplexFlow, X::AbstractVecOrMat{<:Number}, C::AbstractVecOrMat{<:Real})
    Xwasvec = _isvec(X); Xm = _to_mat(X); Cm = _to_mat(C)
    @assert _ncols(Xm) == _ncols(Cm)
    Y = Xm
    for L in flow.layers
        Y = L isa AffineCouplingGLU ? L(Y, Cm) : L isa PermuteChannels ? L(Y) : Y
    end
    return _maybe_vec(Y, Xwasvec)
end

function encode(flow::ComplexFlow, X::AbstractVecOrMat{<:Number})
    Xm = _to_mat(X)
    C = zeros(Float32, flow.ctx_dim, _ncols(Xm))
    return encode(flow, Xm, C)
end

function decode(flow::ComplexFlow, Y::AbstractVecOrMat{<:Number}, C::AbstractVecOrMat{<:Real})
    Ywasvec = _isvec(Y); Ym = _to_mat(Y); Cm = _to_mat(C)
    @assert _ncols(Ym) == _ncols(Cm)
    X = Ym
    for L in Iterators.reverse(flow.layers)
        X = L isa AffineCouplingGLU ? inverse(L, X, Cm) : L isa PermuteChannels ? inverse(L, X) : X
    end
    return _maybe_vec(X, Ywasvec)
end

function decode(flow::ComplexFlow, Y::AbstractVecOrMat{<:Number})
    Ym = _to_mat(Y)
    C = zeros(Float32, flow.ctx_dim, _ncols(Ym))
    return decode(flow, Ym, C)
end

# ------------------------------------------------------------------
# Mask helpers
# ------------------------------------------------------------------

function alternating_masks(d::Integer)
    m1 = ones(Float32, d)
    m2 = ones(Float32, d)
    for i in 1:d
        if isodd(i)
            m1[i] = 0f0
        else
            m2[i] = 0f0
        end
    end
    return m1, m2
end

# ------------------------------------------------------------------
# Example usage (commented)
# ------------------------------------------------------------------
# using Random
# d = 8
# ctx_dim = 5
# flow = ComplexFlow(d; ctx_dim=ctx_dim, num_blocks=6, hidden=128, depth=2)
# N = 32
# X = randn(ComplexF32, d, N)
# C = randn(Float32, ctx_dim, N)
# Y = encode(flow, X, C)
# Xrec = decode(flow, Y, C)
# @assert maximum(abs.(X .- Xrec)) < 1e-4
#
# flow0 = ComplexFlow(d; ctx_dim=0, num_blocks=4)
# Y0 = encode(flow0, X)
# X0 = decode(flow0, Y0)
# @assert maximum(abs.(X .- X0)) < 1e-4

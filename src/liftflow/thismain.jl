# Lifting Complex Flow (Julia/Flux)
# ------------------------------------------------------------------
# This module extends `ComplexGLUFlow` by adding a **lifting stage** before the
# usual complex affine-coupling flow. If sample dim is `d` and latent dim is `ℓ>d`:
#   1) A lift network `g: C^d -> C^(ℓ-d)` appends features to form `[x; g(x)] ∈ C^ℓ`.
#   2) A standard `ComplexGLUFlow.ComplexFlow` acts on the **extended** vector.
# Inversion recovers the original sample by decoding with the second net and
# **projecting** back to the first `d` channels.
#
# Shapes are `AbstractVecOrMat` throughout — pass either vectors `(d,)` or
# matrices `(d, N)`. Context is real-valued with shape `(ctx_dim,)` or `(ctx_dim,N)`.

# ------------------------------------------------------------------
# Small shape helpers (vector-or-matrix aware)
# ------------------------------------------------------------------

_isvec(A) = A isa AbstractVector
_to_mat(A::AbstractVector) = reshape(A, :, 1)
_to_mat(A::AbstractMatrix) = A
_ncols(A::AbstractVector) = 1
_ncols(A::AbstractMatrix) = size(A, 2)
_maybe_vec(A::AbstractMatrix, wasvec::Bool) = wasvec ? vec(A) : A

# ------------------------------------------------------------------
# A small complex MLP for the lifting map: g(x) : C^d -> C^(ℓ-d)
# We reuse ComplexGLUFlow.ComplexDense / CxGLUBlock
# ------------------------------------------------------------------

"""
    LiftMLP(d_in, d_out; hidden=256, depth=2, act=identity, ln=false)

A compact complex MLP that maps `d_in -> d_out` using ComplexGLUFlow.ComplexDense
layers (optionally with LayerNorm inside GLU blocks).
"""
struct LiftMLP
    layers::Vector{Any}
    proj::ComplexGLUFlow.ComplexDense
end
@layer LiftMLP

function LiftMLP(d_in::Integer, d_out::Integer; hidden::Integer=256, depth::Integer=2,
                 act=identity, ln::Bool=false)
    layers = Any[]
    dcur = d_in
    for _ in 1:depth
        push!(layers, ComplexGLUFlow.CxGLUBlock(dcur, hidden; ln=ln, φ=act))
        dcur = hidden
    end
    proj = ComplexGLUFlow.ComplexDense(dcur, d_out)
    return LiftMLP(layers, proj)
end

function (net::LiftMLP)(X)
    H = X
    for l in net.layers
        H = l(H)
    end
    return net.proj(H)
end

# ------------------------------------------------------------------
# Lifting Flow container
# ------------------------------------------------------------------

"""
    LiftingFlow(d, ℓ; ctx_dim=0, num_blocks=4, hidden=256, depth=2,
                lift_hidden=256, lift_depth=2, lift_act=identity,
                clamp=3.0, ln=false, seed=nothing)

Build a lifting flow that:
  - computes `z = g(x) ∈ C^(ℓ-d)` via a complex MLP
  - forms `x_ext = vcat(x, z) ∈ C^ℓ`
  - applies `ComplexGLUFlow.ComplexFlow(ℓ, ...)` on `x_ext`

**Functor API**

`lf(X, C, encode::Bool)`
  - If `encode == true`: returns `y ∈ C^ℓ`.
  - If `encode == false`: returns `x ∈ C^d` (inverse → project first `d`).

`lf(X, encode::Bool)` uses zero context of width `ctx_dim`.

All functions accept vectors or matrices (`AbstractVecOrMat`).
"""
struct LiftingFlow
    d::Int
    ℓ::Int
    ctx_dim::Int
    lift_net::LiftMLP
    flow::ComplexGLUFlow.ComplexFlow
end
@layer LiftingFlow

function LiftingFlow(d::Integer, ℓ::Integer; ctx_dim::Integer=0, num_blocks::Integer=4,
                     hidden::Integer=256, depth::Integer=2,
                     lift_hidden::Integer=256, lift_depth::Integer=2,
                     lift_act=identity, clamp::Real=3.0, ln::Bool=false,
                     seed=nothing)
    @assert ℓ > d "Latent dimension ℓ must be greater than sample dimension d."
    if seed !== nothing
        Random.seed!(seed)
    end
    # lifting map g: C^d -> C^(ℓ-d)
    lift_net = LiftMLP(d, ℓ - d; hidden=lift_hidden, depth=lift_depth, act=lift_act, ln=ln)
    # extended flow over C^ℓ (inherits your zero-init scale + ln=false defaults)
    flow = ComplexGLUFlow.ComplexFlow(ℓ; ctx_dim=ctx_dim, num_blocks=num_blocks,
                                      hidden=hidden, depth=depth, clamp=clamp, ln=ln, seed=seed)
    return LiftingFlow(d, ℓ, ctx_dim, lift_net, flow)
end

# ------------------------------------------------------------------
# Functor interface: (lf)(X, C, encode::Bool) and (lf)(X, encode::Bool)
# ------------------------------------------------------------------

function (lf::LiftingFlow)(X::AbstractVecOrMat{<:Number}, C::AbstractVecOrMat{<:Real}, encode::Bool)
    Xwasvec = _isvec(X); Xm = _to_mat(X); Cm = _to_mat(C)
    if encode
        @assert size(Xm, 1) == lf.d "X has $(size(Xm,1)) rows, expected $(lf.d)."
        @assert size(Cm, 1) == lf.ctx_dim "Context has $(size(Cm,1)) rows, expected $(lf.ctx_dim)."
        @assert _ncols(Xm) == _ncols(Cm) "Sample and context batch sizes must match."

        # lifting → extended input
        Zlift = lf.lift_net(Xm)            # (ℓ-d, N)
        Xext  = vcat(Xm, Zlift)            # (ℓ, N)
        Y = ComplexGLUFlow.encode(lf.flow, Xext, Cm)
        return _maybe_vec(Y, Xwasvec)
    else
        @assert size(Xm, 1) == lf.ℓ "Y has $(size(Xm,1)) rows, expected $(lf.ℓ)."
        @assert size(Cm, 1) == lf.ctx_dim "Context has $(size(Cm,1)) rows, expected $(lf.ctx_dim)."
        @assert _ncols(Xm) == _ncols(Cm) "Latent and context batch sizes must match."

        Xext = ComplexGLUFlow.decode(lf.flow, Xm, Cm)  # (ℓ, N)
        Xrec = Xext[1:lf.d, :]
        return _maybe_vec(Xrec, Xwasvec)
    end
end

function (lf::LiftingFlow)(X::AbstractVecOrMat{<:Number}, encode::Bool)
    Xm = _to_mat(X)
    C = zeros(Float32, lf.ctx_dim, _ncols(Xm))
    return lf(Xm, C, encode)
end

# ------------------------------------------------------------------
# Example usage (commented)
# ------------------------------------------------------------------
# using Random
# using ComplexGLUFlow
# d, ℓ = 8, 12
# ctx_dim = 3
# lf = LiftingFlow(d, ℓ; ctx_dim=ctx_dim, num_blocks=4, hidden=128, depth=2,
#                  lift_hidden=64, lift_depth=2, ln=false, clamp=2.5, seed=0)
# N = 16
# X = randn(ComplexF32, d, N)
# C = randn(Float32, ctx_dim, N)
# Y = lf(X, C, true)    # encode
# Xrec = lf(Y, C, false) # decode (projects to first d)
# @assert maximum(abs.(X .- Xrec)) < 1e-4


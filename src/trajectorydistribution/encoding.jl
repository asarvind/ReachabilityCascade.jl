"""
TrajectoryModels — Flux.jl module

A two-part neural architecture for encoding robot state trajectories and
predicting reached states from an embedding, an initial state, and a time span.

- **Transformer encoder** consumes a trajectory matrix `X ∈ ℝ^{statedim×T}`
  (columns are states ordered by time index) and produces a fixed-size
  embedding `z ∈ ℝ^{embeddim}`.
- **State predictor** takes `(z, x0, Δt)` and predicts the state reached
  after `Δt` steps starting from `x0` along the trajectory represented by `z`.

The core user API:

- `TrajectoryEncoder(statedim, embeddim; kwargs...)` → model
- `encode(model, X)` → z
- `predict(model, z, x0, Δt)` → x̂_{t0+Δt}
- `predict_from_traj(model, X, x0, Δt)` → x̂_{t0+Δt}

All computations are non-mutating and respect Flux's AD.
"""

# ----------------------
# Utility: Positional Encoding (sin/cos, transformer-style)
# ----------------------

"""
    positional_encoding(dmodel::Integer, T::Integer)

Return a `(dmodel, T)` sinusoidal positional encoding matrix suitable for
adding to token embeddings of size `(dmodel, T)`.

Zygote-friendly (no explicit setindex!/mutation): we construct sin/cos grids
by broadcasting and interleave along a new axis, then reshape.
"""
function positional_encoding(dmodel::Integer, T::Integer)
    @assert dmodel > 0 "dmodel must be positive"
    k = dmodel ÷ 2
    # frequencies and positions
    invfreq = @. Float32(1) / (Float32(10000) ^ ((Float32(0):Float32(k-1)) / max(Float32(1), Float32(k))))
    pos = collect(Float32.(0:T-1))             # (T,)
    # broadcast to (k, T)
    S = sin.(invfreq .* pos')                  # (k, T)
    C = cos.(invfreq .* pos')                  # (k, T)
    # stack to (k, T, 2) and interleave along first dimension → (2k, T)
    ktz = cat(S, C; dims=3)                    # (k, T, 2)
    pe = reshape(permutedims(ktz, (3, 1, 2)), 2k, T)  # (2k, T) as sin1,cos1,..
    return (2k == dmodel) ? pe : vcat(pe, zeros(Float32, dmodel - 2k, T))
end

# ----------------------
# Transformer Encoder Block
# ----------------------

struct TEBlock{M<:MultiHeadAttention,N1<:LayerNorm,FF<:Chain,N2<:LayerNorm}
    mha::M
    norm1::N1
    ff::FF
    norm2::N2
end
Flux.@layer TEBlock

"""
    TEBlock(dmodel, nheads; ffhidden=4*dmodel, act=leakyrelu)

A standard Transformer encoder block with pre-norm, MHA, and position-wise FFN (no dropout). Inputs are shaped `(dmodel, T, B)`.
"""
function TEBlock(dmodel::Integer, nheads::Integer; ffhidden::Integer=4*dmodel, act=leakyrelu)
    # MultiHeadAttention without dropout (modern practice: rely on LayerNorm, residuals, weight decay)
    mha = MultiHeadAttention(dmodel; nheads=nheads, dropout_prob=0.0, bias=false)
    ff  = Chain(
        Dense(dmodel, ffhidden, act),
        Dense(ffhidden, dmodel)
    )
    return TEBlock(mha, LayerNorm(dmodel), ff, LayerNorm(dmodel))
end

# Forward for TEBlock (non-mutating)
function (blk::TEBlock)(x)
    # x: (dmodel, T, B)
    yattn = blk.mha(x)
    yattn = yattn isa Tuple ? yattn[1] : yattn
    x = blk.norm1(x + yattn)
    E, T, B = size(x)
    x2 = reshape(x, E, T*B)
    y2 = blk.ff(x2)
    y2 = reshape(y2, E, T, B)
    x = blk.norm2(x + y2)
    return x
end

# ----------------------
# Time embedding (Fourier features on integer Δt)
# ----------------------

"""
    time_features(Δt::Integer, d::Integer)

Compute a fixed sinusoidal feature vector of length `d` for integer time spans.
If `d` is odd, the last coordinate is zeros. Zygote-friendly (no mutation).
"""
function time_features(Δt::Integer, d::Integer)
    # Delegate to batched path to keep feature ordering identical
    VF = time_features([Δt], d)
    return vec(VF[:, 1])
end

"""
    time_features(Δt::AbstractVector{<:Integer}, d::Integer)

Batch version: returns a `(d, B)` matrix of time features given `B = length(Δt)`.
Zygote-friendly (no setindex! in loops).
"""
function time_features(Δt::AbstractVector{<:Integer}, d::Integer)
    B = length(Δt)
    d <= 0 && return zeros(Float32, 0, B)
    k = d ÷ 2
    invfreq = @. Float32(1) / (Float32(10000) ^ ((Float32(0):Float32(k-1)) / max(Float32(1), Float32(k))))
    Δ = Float32.(Δt)                             # (B,)
    # broadcast to (k, B)
    S = sin.(invfreq .* reshape(Δ, 1, :))  # (k, B)
    C = cos.(invfreq .* reshape(Δ, 1, :))  # (k, B)
    kbt = cat(S, C; dims=3)                      # (k, B, 2)
    V = reshape(permutedims(kbt, (3, 1, 2)), 2k, B)  # (2k, B)
    return (2k == d) ? V : vcat(V, zeros(Float32, d - 2k, B))
end

# ----------------------
# TrajectoryEncoder Layer
# ----------------------

struct TrajectoryEncoder{D1<:Dense, B<:TEBlock, D2<:Dense, C<:Chain}
    # dims (non-trainable)
    statedim::Int
    dmodel::Int
    embeddim::Int

    # submodules (trainable)
    projin::D1             # statedim -> dmodel
    encblocks::Vector{B}   # NOTE: concrete element type for performance
    projz::D2              # dmodel -> embeddim

    # predictor
    timedim::Int
    predictor::C           # maps [z; x0; tfeat] -> statedim
end
Flux.@layer TrajectoryEncoder trainable=(projin, encblocks, projz, predictor)

"""
    TrajectoryEncoder(statedim::Integer, embeddim::Integer; dmodel=128, nheads=4, numlayers=2,
                      ffhidden=256, act=leakyrelu,
                      timedim=32, predhidden=(256,256))

Build a `TrajectoryEncoder`.

# Arguments
- `statedim`: dimension of state vectors (rows of the trajectory matrix).
- `embeddim` (positional): size of the trajectory embedding `z`.
- `dmodel`: internal token/embed size for the transformer.
- `nheads`: number of attention heads.
- `numlayers`: number of encoder blocks.
- `ffhidden`: hidden size in the transformer feedforward.
- `act`: activation function for feedforward layers.
- `timedim`: dimension of the sinusoidal time-span features.
- `predhidden`: tuple of hidden sizes for the predictor MLP.

The transformer consumes `(statedim, T)` trajectories (columns are states),
adds sinusoidal positional encodings by time index, and mean-pools over time to
produce `z`. The predictor maps `h = [z; x0; time_features(Δt, timedim)]`
through an MLP to predict the reached state after `Δt` steps from `x0`.
"""
function TrajectoryEncoder(statedim::Integer, embeddim::Integer; dmodel::Integer=128, nheads::Integer=4, numlayers::Integer=2,
                           ffhidden::Integer=256, act=leakyrelu,
                           timedim::Integer=32, predhidden::Tuple{Vararg{Int}}=(256,256))
    projin = Dense(statedim, dmodel)
    encblocks = [TEBlock(dmodel, nheads; ffhidden=ffhidden, act=act) for _ in 1:numlayers]
    projz = Dense(dmodel, embeddim)

    # Build predictor MLP: input = embeddim + statedim + timedim
    infeat = embeddim + statedim + timedim
    layers = Vector{Any}()
    last = infeat
    for h in predhidden
        push!(layers, Dense(last, h, act))
        last = h
    end
    push!(layers, Dense(last, statedim))
    predictor = Chain(layers...)

    return TrajectoryEncoder(statedim, dmodel, embeddim, projin, encblocks, projz, timedim, predictor)
end

# ----------------------
# Encoder: trajectory -> embedding
# ----------------------

"""
    encode(model::TrajectoryEncoder, X::AbstractMatrix)

Encode a trajectory matrix `X ∈ ℝ^{statedim×T}` (columns are states) into an
embedding vector `z ∈ ℝ^{embeddim}`.
"""
function encode(model::TrajectoryEncoder, X::AbstractMatrix)
    @assert size(X, 1) == model.statedim "Expected X to have size (statedim, T)."
    T = size(X, 2)
    # Tokenize & add positional encoding (no mutation)
    x = model.projin(X) + positional_encoding(model.dmodel, T)
    # shape to (dmodel, T, B=1)
    x = reshape(x, model.dmodel, T, 1)
    # pass through encoder blocks
    for blk in model.encblocks
        x = blk(x)
    end
    # mean-pool over time, then project to z
    h = dropdims(mean(x; dims=2), dims=2)  # (dmodel, 1)
    h = reshape(h, model.dmodel, :)        # (dmodel, 1)
    z = model.projz(h)                     # (embeddim, 1)
    return vec(z)                          # embeddim
end

# ----------------------
# Predictor: (z, x0, Δt) -> reached state
# ----------------------

"""
    predict(model::TrajectoryEncoder, z::AbstractVector, x0::AbstractVector, Δt::Integer)

Predict the state reached after `Δt` steps starting from `x0`, following the
trajectory represented by `z`.

All vectors are 1-D. Returns a 1-D state vector of length `statedim`.
"""
function predict(model::TrajectoryEncoder, z::AbstractVector, x0::AbstractVector, Δt::Integer)
    # Route through the batched implementation for consistency
    X0 = reshape(Float32.(x0), :, 1)
    Y = predict(model, Float32.(z), X0, [Δt])
    return vec(Y[:, 1])
end

"""
    predict(model::TrajectoryEncoder, z::AbstractVector, x0::AbstractMatrix, Δt::AbstractVector{<:Integer})

**Batched** predict with a fixed embedding `z` and a batch of initial states
`x0 ∈ ℝ^{statedim×B}` and time spans `Δt` of length `B`. Returns a
`statedim×B` matrix.
"""
function predict(model::TrajectoryEncoder, z::AbstractVector, x0::AbstractMatrix, Δt::AbstractVector{<:Integer})
    @assert length(z)  == model.embeddim  "z must have length embeddim"
    @assert size(x0, 1) == model.statedim  "x0 must have size (statedim, B)"
    B = size(x0, 2)
    @assert length(Δt) == B "Δt and x0 batch size must match"
    # build features: (embeddim + statedim + timedim, B)
    zcol = reshape(Float32.(z), :, 1)
    Z = repeat(zcol, 1, B)
    tfeat = time_features(Δt, model.timedim)           # (timedim, B)
    H = vcat(Z, Float32.(x0), tfeat)                   # (infeat, B)
    Y = model.predictor(H)                             # (statedim, B)
    return Y
end

"""
    predict(model::TrajectoryEncoder, z::AbstractVector, x0::AbstractMatrix, Δt::Integer)

Convenience: batched `x0` with scalar `Δt` (broadcast time to the batch).
"""
function predict(model::TrajectoryEncoder, z::AbstractVector, x0::AbstractMatrix, Δt::Integer)
    B = size(x0, 2)
    return predict(model, z, x0, fill(Δt, B))
end

# ----------------------
# predict_from_traj wrappers
# ----------------------

"""
    predict_from_traj(model::TrajectoryEncoder, X::AbstractMatrix, x0::AbstractVector, Δt::Integer)

Encode `X` once and predict a single `(x0, Δt)`.
"""
function predict_from_traj(model::TrajectoryEncoder, X::AbstractMatrix, x0::AbstractVector, Δt::Integer)
    z = encode(model, X)
    # Use batched path to ensure identical behavior
    return predict(model, z, reshape(Float32.(x0), :, 1), [Δt])[:, 1]
end

"""
    predict_from_traj(model::TrajectoryEncoder, X::AbstractMatrix, x0::AbstractMatrix, Δt::AbstractVector{<:Integer})

Batched version: fixed `X`, but `x0` and `Δt` are batched with the same
batch size `B`. Returns `statedim×B`.
"""
function predict_from_traj(model::TrajectoryEncoder, X::AbstractMatrix, x0::AbstractMatrix, Δt::AbstractVector{<:Integer})
    z = encode(model, X)
    return predict(model, z, x0, Δt)
end

"""
    predict_from_traj(model::TrajectoryEncoder, X::AbstractMatrix, x0::AbstractMatrix, Δt::Integer)

Convenience: batched `x0` with scalar `Δt`.
"""
function predict_from_traj(model::TrajectoryEncoder, X::AbstractMatrix, x0::AbstractMatrix, Δt::Integer)
    z = encode(model, X)
    return predict(model, z, x0, Δt)
end

# ----------------------
# Example (commented)
# ----------------------

# using Random
# Random.seed!(42)
# statedim = 6
# model = TrajectoryEncoder(statedim, 48; dmodel=64, nheads=4, numlayers=2,
#                           ffhidden=128, act=leakyrelu,
#                           timedim=16, predhidden=(128,128))
# X = randn(Float32, statedim, 20)  # toy trajectory
# z = encode(model, X)
# x0 = X[:, 1]
# ŷ = predict(model, z, x0, 5)
# # Batched
# x0b = randn(Float32, statedim, 8)
# Δtb = [1,2,3,4,1,2,3,4]
# Ŷ = predict_from_traj(model, X, x0b, Δtb)
# @show size(z), size(ŷ), size(Ŷ)

import Flux
using Random

"""
    NormalizingFlow(dim, context_dim; spec=nothing, logscale_clamp=2.0, rng=Random.default_rng())

Normalizing flow built from GLU-MLP coupling layers (additive or affine), with fixed random permutations.

# Arguments
- `dim::Integer`: data dimension `D` (features).
- `context_dim::Integer`: context dimension `C` (features); use `0` for no context.

# Keyword Arguments
- `spec`: `3×L` integer matrix specifying the coupling stack. Each column corresponds to a *pair* of coupling sublayers
  with complementary masks (so the flow has `2L` sublayers total):
  - row 1: hidden width of the GLU-MLP conditioner (`hidden_dim`)
  - row 2: depth / number of GLU blocks (`n_glu`)
  - row 3: coupling type flag (`1` for affine, `0` for additive)
  Default is 3 layers with widths 128, depths 1, all affine.
- `logscale_clamp::Real=2.0`: scale for `tanh`-bounded log-scales in affine coupling.
- `rng::Random.AbstractRNG=Random.default_rng()`: RNG used to initialize the fixed random permutations.
- `perms=nothing`: optional vector of fixed permutations (length `L`), each a `dim`-length vector of indices `1:dim`.
  If provided, each permutation is used for the two complementary sublayers of the corresponding `spec` column.

# Returns
- `flow::NormalizingFlow`: flow object which supports [`encode`](@ref) and [`decode`](@ref).
"""
struct NormalizingFlow{L}
    dim::Int
    context_dim::Int
    spec::Matrix{Int}
    logscale_clamp::Float32
    layers::L
end

Flux.@layer NormalizingFlow

_default_spec() = begin
    # 3 layers: widths=128, depths=1, all affine
    h = fill(128, 1, 3)
    d = fill(1, 1, 3)
    a = fill(1, 1, 3)
    return vcat(h, d, a)
end

function NormalizingFlow(dim::Integer,
                         context_dim::Integer;
                         spec=nothing,
                         logscale_clamp::Real=2.0,
                         rng::Random.AbstractRNG=Random.default_rng(),
                         perms=nothing)
    dim_int = Int(dim)
    ctx_int = Int(context_dim)
    dim_int > 0 || throw(ArgumentError("dim must be positive"))
    ctx_int >= 0 || throw(ArgumentError("context_dim must be non-negative"))
    logscale_clamp > 0 || throw(ArgumentError("logscale_clamp must be positive"))

    spec_mat = spec === nothing ? _default_spec() : Matrix{Int}(spec)
    size(spec_mat, 1) == 3 || throw(ArgumentError("spec must be a 3×L integer matrix; got size=$(size(spec_mat))"))
    L = size(spec_mat, 2)
    L >= 1 || throw(ArgumentError("spec must have at least one column (layer)"))

    # Validate the coupling type flags.
    flags = spec_mat[3, :]
    all(f -> (f == 0 || f == 1), flags) ||
        throw(ArgumentError("spec[3, :] must contain only 0 (additive) or 1 (affine)"))

    # Important: we want a fixed random permutation per layer.
    # We sample them from a single RNG stream for reproducibility, unless `perms` are provided.
    if perms !== nothing
        length(perms) == L || throw(ArgumentError("perms must have length $L; got length=$(length(perms))"))
        for (i, p) in enumerate(perms)
            length(p) == dim_int || throw(ArgumentError("perms[$i] must have length $dim_int; got length=$(length(p))"))
            all(1 .<= p .<= dim_int) || throw(ArgumentError("perms[$i] must contain indices in 1:$dim_int"))
            length(unique(p)) == dim_int || throw(ArgumentError("perms[$i] must be a permutation of 1:$dim_int"))
        end
    end

    # Each spec column expands into two sublayers with complementary masks that share the same permutation.
    layers = Vector{Any}(undef, 2L)
    for i in 1:L
        hidden_i = spec_mat[1, i]
        depth_i = spec_mat[2, i]
        affine_i = spec_mat[3, i] == 1
        if perms === nothing
            base = CouplingLayer(dim_int, ctx_int, hidden_i, depth_i, affine_i;
                                 logscale_clamp=logscale_clamp, flip=false, rng=rng)
            comp = CouplingLayer(dim_int, ctx_int, hidden_i, depth_i, affine_i, base.perm;
                                 logscale_clamp=logscale_clamp, flip=true)
        else
            base = CouplingLayer(dim_int, ctx_int, hidden_i, depth_i, affine_i, perms[i];
                                 logscale_clamp=logscale_clamp, flip=false)
            comp = CouplingLayer(dim_int, ctx_int, hidden_i, depth_i, affine_i, perms[i];
                                 logscale_clamp=logscale_clamp, flip=true)
        end
        layers[2i - 1] = base
        layers[2i] = comp
    end

    return NormalizingFlow(dim_int, ctx_int, spec_mat, Float32(logscale_clamp), layers)
end

"""
    encode(flow, x, context) -> (z, logdet)

Forward pass of the flow: maps `x` to latent `z` and returns the log-determinant.

# Arguments
- `flow`: [`NormalizingFlow`](@ref).
- `x`: input data (vector or batch matrix).
- `context`: conditioning context (vector or batch matrix).

# Returns
- If `x` is a matrix (`D×B`):
  - `z::Matrix{Float32}`: latent `D×B`.
  - `logdet::Vector{Float32}`: per-sample log determinant (length `B`).
- If `x` is a vector (`D`):
  - `z::Vector{Float32}`: latent `D`.
  - `logdet::Float32`: scalar log determinant.
"""
function encode(flow::NormalizingFlow, x::AbstractMatrix, context::AbstractMatrix)
    ndims(x) == 2 || throw(ArgumentError("x must be a (dim × batch) matrix"))
    size(x, 1) == flow.dim || throw(DimensionMismatch("x must have $(flow.dim) rows"))
    ndims(context) == 2 || throw(ArgumentError("context must be a (context_dim × batch) matrix"))
    size(context, 1) == flow.context_dim || throw(DimensionMismatch("context must have $(flow.context_dim) rows"))
    size(context, 2) == size(x, 2) || throw(DimensionMismatch("context batch must match x batch"))

    # Zygote does not support mutating arrays that participate in AD,
    # so keep this function purely functional (no `.+=` on logdet).
    z = x isa Matrix{Float32} ? x : Float32.(Matrix(x))
    c32 = context isa Matrix{Float32} ? context : Float32.(Matrix(context))
    logdet = zeros(Float32, size(z, 2))

    for layer in flow.layers
        # Each layer returns per-sample logdet contribution.
        z, dlog = forward(layer, z, c32)
        logdet = logdet .+ dlog
    end

    return z, logdet
end

function encode(flow::NormalizingFlow, x::AbstractVector, context::AbstractVector)
    length(x) == flow.dim || throw(DimensionMismatch("x must have length $(flow.dim); got $(length(x))"))
    length(context) == flow.context_dim ||
        throw(DimensionMismatch("context must have length $(flow.context_dim); got $(length(context))"))
    x_mat = reshape(x, :, 1)
    c_mat = reshape(context, :, 1)
    z, logdet = encode(flow, x_mat, c_mat)
    return vec(z), logdet[1]
end

"""
    decode(flow, z, context) -> x

Inverse pass of the flow: maps latent `z` back to data space `x`.

# Arguments
- `flow`: [`NormalizingFlow`](@ref).
- `z`: latent (vector or batch matrix).
- `context`: conditioning context (vector or batch matrix).

# Returns
- If `z` is a matrix (`D×B`): `x::Matrix{Float32}` of size `D×B`.
- If `z` is a vector (`D`): `x::Vector{Float32}` of length `D`.
"""
function decode(flow::NormalizingFlow, z::AbstractMatrix, context::AbstractMatrix)
    ndims(z) == 2 || throw(ArgumentError("z must be a (dim × batch) matrix"))
    size(z, 1) == flow.dim || throw(DimensionMismatch("z must have $(flow.dim) rows"))
    ndims(context) == 2 || throw(ArgumentError("context must be a (context_dim × batch) matrix"))
    size(context, 1) == flow.context_dim || throw(DimensionMismatch("context must have $(flow.context_dim) rows"))
    size(context, 2) == size(z, 2) || throw(DimensionMismatch("context batch must match z batch"))

    x = z isa Matrix{Float32} ? z : Float32.(Matrix(z))
    c32 = context isa Matrix{Float32} ? context : Float32.(Matrix(context))

    # Inverse applies layers in reverse order.
    for layer in Iterators.reverse(flow.layers)
        x = inverse(layer, x, c32)
    end

    return x
end

function decode(flow::NormalizingFlow, z::AbstractVector, context::AbstractVector)
    length(z) == flow.dim || throw(DimensionMismatch("z must have length $(flow.dim); got $(length(z))"))
    length(context) == flow.context_dim ||
        throw(DimensionMismatch("context must have length $(flow.context_dim); got $(length(context))"))
    z_mat = reshape(z, :, 1)
    c_mat = reshape(context, :, 1)
    x = decode(flow, z_mat, c_mat)
    return vec(x)
end

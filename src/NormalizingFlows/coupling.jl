import Flux
using Random
using ..GatedLinearUnits: glu_mlp

"""
    CouplingLayer(dim, context_dim, hidden_dim, depth, affine;
                  logscale_clamp=2.0, rng=Random.default_rng(), flip=false)

One coupling layer with a fixed random permutation and a GLU-MLP conditioner.

This layer implements either:
- additive coupling (`affine=false`): `y_b = x_b + t(x_a, c)`
- affine coupling (`affine=true`): `y_b = x_b .* exp(s(x_a, c)) + t(x_a, c)`,
  with bounded `s = logscale_clamp * tanh(raw_s)`

The input is first permuted with a fixed random permutation (sampled at construction),
then split into `x_a` and `x_b`. By default (`flip=false`) the layer transforms `x_b` conditioned on
`x_a`. With `flip=true` it uses the complementary mask: transforms `x_a` conditioned on `x_b`.
In both cases the inverse permutation is applied at the end.

# Arguments
- `dim::Integer`: data dimension `D`.
- `context_dim::Integer`: context feature dimension `C` (can be 0).
- `hidden_dim::Integer`: hidden width for the GLU-MLP.
- `depth::Integer`: number of GLU blocks (`n_glu`) in the GLU-MLP.
- `affine::Bool`: whether to use affine coupling (`true`) or additive coupling (`false`).

# Keyword Arguments
- `logscale_clamp::Real=2.0`: scale for the `tanh`-bounded log-scale (affine only).
- `rng::Random.AbstractRNG=Random.default_rng()`: RNG used to initialize the fixed random permutation.
- `flip=false`: if `true`, use the complementary mask (transform `x_a` conditioned on `x_b`).
"""
struct CouplingLayer{N}
    dim::Int
    context_dim::Int
    hidden_dim::Int
    depth::Int
    affine::Bool
    flip::Bool
    logscale_clamp::Float32
    perm::Vector{Int}
    invperm::Vector{Int}
    split_a::Int
    split_b::Int
    net::N
end

Flux.@layer CouplingLayer

function CouplingLayer(dim::Integer,
                       context_dim::Integer,
                       hidden_dim::Integer,
                       depth::Integer,
                       affine::Bool;
                       logscale_clamp::Real=2.0,
                       flip::Bool=false,
                       rng::Random.AbstractRNG=Random.default_rng())
    dim_int = Int(dim)
    ctx_int = Int(context_dim)
    hidden_int = Int(hidden_dim)
    depth_int = Int(depth)
    dim_int > 0 || throw(ArgumentError("dim must be positive"))
    ctx_int >= 0 || throw(ArgumentError("context_dim must be non-negative"))
    hidden_int > 0 || throw(ArgumentError("hidden_dim must be positive"))
    depth_int > 0 || throw(ArgumentError("depth must be positive"))
    logscale_clamp > 0 || throw(ArgumentError("logscale_clamp must be positive"))

    # We transform the "b" block and keep the "a" block as conditioning input.
    # For odd `dim`, we keep `a` smaller and `b` larger so every layer can still move most coordinates.
    split_a = dim_int ÷ 2
    split_a >= 1 || throw(ArgumentError("dim must be ≥ 2 for coupling layers; got $dim_int"))
    split_b = dim_int - split_a

    # Fixed random permutation makes the split cover different coordinates across layers.
    perm = randperm(rng, dim_int)
    invperm = similar(perm)
    for (i, p) in enumerate(perm)
        invperm[p] = i
    end

    # Conditioner input and output depend on which half is transformed.
    # flip=false: condition on x_a and output params for x_b
    # flip=true:  condition on x_b and output params for x_a
    in_dim = split_a + ctx_int
    out_dim = affine ? (2 * split_b) : split_b
    if flip
        in_dim = split_b + ctx_int
        out_dim = affine ? (2 * split_a) : split_a
    end

    # Important: `zero_init=true` makes the layer start as (approximately) identity,
    # which stabilizes training and is common in flow initialization.
    net = glu_mlp(in_dim, hidden_int, out_dim; n_glu=depth_int, zero_init=true)

    return CouplingLayer(dim_int, ctx_int, hidden_int, depth_int, affine, flip,
                         Float32(logscale_clamp),
                         perm, invperm, split_a, split_b, net)
end

function CouplingLayer(dim::Integer,
                       context_dim::Integer,
                       hidden_dim::Integer,
                       depth::Integer,
                       affine::Bool,
                       perm_in::AbstractVector{<:Integer};
                       logscale_clamp::Real=2.0,
                       flip::Bool=false)
    dim_int = Int(dim)
    ctx_int = Int(context_dim)
    hidden_int = Int(hidden_dim)
    depth_int = Int(depth)
    dim_int > 0 || throw(ArgumentError("dim must be positive"))
    ctx_int >= 0 || throw(ArgumentError("context_dim must be non-negative"))
    hidden_int > 0 || throw(ArgumentError("hidden_dim must be positive"))
    depth_int > 0 || throw(ArgumentError("depth must be positive"))
    logscale_clamp > 0 || throw(ArgumentError("logscale_clamp must be positive"))

    split_a = dim_int ÷ 2
    split_a >= 1 || throw(ArgumentError("dim must be ≥ 2 for coupling layers; got $dim_int"))
    split_b = dim_int - split_a

    perm = Int.(collect(perm_in))
    length(perm) == dim_int || throw(ArgumentError("perm must have length $dim_int; got length=$(length(perm))"))
    all(1 .<= perm .<= dim_int) || throw(ArgumentError("perm must contain indices in 1:$dim_int"))
    length(unique(perm)) == dim_int || throw(ArgumentError("perm must be a permutation of 1:$dim_int"))

    invperm = similar(perm)
    for (i, p) in enumerate(perm)
        invperm[p] = i
    end

    in_dim = split_a + ctx_int
    out_dim = affine ? (2 * split_b) : split_b
    if flip
        in_dim = split_b + ctx_int
        out_dim = affine ? (2 * split_a) : split_a
    end
    net = glu_mlp(in_dim, hidden_int, out_dim; n_glu=depth_int, zero_init=true)

    return CouplingLayer(dim_int, ctx_int, hidden_int, depth_int, affine, flip,
                         Float32(logscale_clamp),
                         perm, invperm, split_a, split_b, net)
end

_check_batch(x, name::AbstractString, D::Int) = begin
    ndims(x) == 2 || throw(ArgumentError("$name must be a (features × batch) matrix; got ndims=$(ndims(x))"))
    size(x, 1) == D || throw(DimensionMismatch("$name must have $D rows; got $(size(x, 1))"))
    return size(x, 2)
end

_check_context(c, B::Int, C::Int) = begin
    ndims(c) == 2 || throw(ArgumentError("context must be a (context_dim × batch) matrix; got ndims=$(ndims(c))"))
    size(c, 1) == C || throw(DimensionMismatch("context must have $C rows; got $(size(c, 1))"))
    size(c, 2) == B || throw(DimensionMismatch("context batch must match x batch size $B; got $(size(c, 2))"))
    return nothing
end

"""
    forward(layer, x, context) -> (y, logdet)

Forward transform for a single coupling layer.

# Arguments
- `x::AbstractMatrix`: input `D×B`.
- `context::AbstractMatrix`: context `C×B` (use `C=0` to disable conditioning).

# Returns
- `y::Matrix{Float32}`: output `D×B`.
- `logdet::Vector{Float32}`: per-sample log determinant (length `B`).
"""
function forward(layer::CouplingLayer, x::AbstractMatrix, context::AbstractMatrix)
    B = _check_batch(x, "x", layer.dim)
    _check_context(context, B, layer.context_dim)

    x32 = x isa Matrix{Float32} ? x : Float32.(Matrix(x))
    c32 = context isa Matrix{Float32} ? context : Float32.(Matrix(context))

    # 1) Permute features.
    # Use a materialized gather (not a view) because AD through `view(::Matrix, ::Vector, :)`
    # tends to be brittle in Zygote; `getindex` is much more reliable.
    x_p = x32[layer.perm, :]

    # 2) Split into `a` and `b`.
    x_a = @view x_p[1:layer.split_a, :]
    x_b = @view x_p[(layer.split_a + 1):end, :]

    if !layer.flip
        # Base mask: transform `b` conditioned on `a` and context.
        cond = vcat(x_a, c32)
        params = layer.net(cond)
        if layer.affine
            raw_s = @view params[1:layer.split_b, :]
            t = @view params[(layer.split_b + 1):end, :]
            s = layer.logscale_clamp .* tanh.(raw_s)
            y_b = x_b .* exp.(s) .+ t
            logdet = vec(sum(s; dims=1))
            y_p = vcat(x_a, y_b)
            y = y_p[layer.invperm, :]
            return y, Float32.(logdet)
        else
            t = params
            y_b = x_b .+ t
            y_p = vcat(x_a, y_b)
            y = y_p[layer.invperm, :]
            return y, zeros(Float32, B)
        end
    else
        # Complementary mask: transform `a` conditioned on `b` and context.
        cond = vcat(x_b, c32)
        params = layer.net(cond)
        if layer.affine
            raw_s = @view params[1:layer.split_a, :]
            t = @view params[(layer.split_a + 1):end, :]
            s = layer.logscale_clamp .* tanh.(raw_s)
            y_a = x_a .* exp.(s) .+ t
            logdet = vec(sum(s; dims=1))
            y_p = vcat(y_a, x_b)
            y = y_p[layer.invperm, :]
            return y, Float32.(logdet)
        else
            t = params
            y_a = x_a .+ t
            y_p = vcat(y_a, x_b)
            y = y_p[layer.invperm, :]
            return y, zeros(Float32, B)
        end
    end
end

"""
    inverse(layer, z, context) -> x

Inverse transform for a single coupling layer.

# Arguments
- `z::AbstractMatrix`: latent `D×B`.
- `context::AbstractMatrix`: context `C×B`.

# Returns
- `x::Matrix{Float32}`: reconstructed `D×B`.
"""
function inverse(layer::CouplingLayer, z::AbstractMatrix, context::AbstractMatrix)
    B = _check_batch(z, "z", layer.dim)
    _check_context(context, B, layer.context_dim)

    z32 = z isa Matrix{Float32} ? z : Float32.(Matrix(z))
    c32 = context isa Matrix{Float32} ? context : Float32.(Matrix(context))

    # 1) Permute features into the coupling coordinates.
    # See note in `forward`: materialize gather for AD robustness.
    z_p = z32[layer.perm, :]
    z_a = @view z_p[1:layer.split_a, :]
    z_b = @view z_p[(layer.split_a + 1):end, :]

    if !layer.flip
        # Base mask: invert `b` conditioned on `a` and context.
        cond = vcat(z_a, c32)
        params = layer.net(cond)
        if layer.affine
            raw_s = @view params[1:layer.split_b, :]
            t = @view params[(layer.split_b + 1):end, :]
            s = layer.logscale_clamp .* tanh.(raw_s)
            x_b = (z_b .- t) .* exp.(-s)
            x_p = vcat(z_a, x_b)
            x = x_p[layer.invperm, :]
            return x
        else
            t = params
            x_b = z_b .- t
            x_p = vcat(z_a, x_b)
            x = x_p[layer.invperm, :]
            return x
        end
    else
        # Complementary mask: invert `a` conditioned on `b` and context.
        cond = vcat(z_b, c32)
        params = layer.net(cond)
        if layer.affine
            raw_s = @view params[1:layer.split_a, :]
            t = @view params[(layer.split_a + 1):end, :]
            s = layer.logscale_clamp .* tanh.(raw_s)
            x_a = (z_a .- t) .* exp.(-s)
            x_p = vcat(x_a, z_b)
            x = x_p[layer.invperm, :]
            return x
        else
            t = params
            x_a = z_a .- t
            x_p = vcat(x_a, z_b)
            x = x_p[layer.invperm, :]
            return x
        end
    end
end

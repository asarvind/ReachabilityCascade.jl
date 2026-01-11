import Flux
using Random
import NLopt
using ..GatedLinearUnits: glu_mlp

"""
    InvertibleCoupling(dim, context_dim; spec=nothing, logscale_clamp=2.0, rng=Random.default_rng())

Invertible coupling network built from GLU-MLP coupling layers (additive or affine), with fixed random permutations.

This is structurally similar to a normalizing flow, but [`encode`](@ref) returns only the latent `z` (no log-det).

# Arguments
- `dim`: data dimension `D` (features).
- `context_dim`: context dimension `C` (features); use `0` for no context.

# Keyword Arguments
- `spec`: `3×L` integer matrix specifying the coupling stack. Each column corresponds to a *pair* of coupling sublayers
  with complementary masks (so the network has `2L` sublayers total):
  - row 1: hidden width of the GLU-MLP conditioner (`hidden_dim`)
  - row 2: depth / number of GLU blocks (`n_glu`)
  - row 3: coupling type flag (`1` for affine, `0` for additive)
  Default is 3 layers with widths 128, depths 1, all affine.
- `logscale_clamp=2.0`: scale for `tanh`-bounded log-scales in affine coupling.
- `rng=Random.default_rng()`: RNG used to initialize the fixed random permutations (one permutation per `spec` column).
- `perms=nothing`: optional vector of fixed permutations (length `L`), each a `dim`-length vector of indices `1:dim`.
  If provided, each permutation is used for the two complementary sublayers of the corresponding `spec` column.

# Returns
- `net::InvertibleCoupling`: network object which supports [`encode`](@ref) and [`decode`](@ref).
"""
struct InvertibleCoupling{L}
    dim::Int
    context_dim::Int
    spec::Matrix{Int}
    logscale_clamp::Float32
    layers::L
end

Flux.@layer InvertibleCoupling

_default_spec() = begin
    # 3 layers: widths=128, depths=1, all affine
    h = fill(128, 1, 3)
    d = fill(1, 1, 3)
    a = fill(1, 1, 3)
    return vcat(h, d, a)
end

"""
    sample_latent_l1(rng, dim, batch; radius_min=0.0) -> z

Sample a latent batch `z` whose columns lie inside the L1 unit ball, i.e. `‖z[:, j]‖₁ ≤ 1`.

Construction (per column):
1. Draw a random direction `u ~ N(0, I)`.
2. Normalize it to have L1 norm 1.
3. Draw a radius `r ~ Uniform(radius_min, 1)` and scale: `z = r * u / ‖u‖₁`.

# Arguments
- `rng`: random number generator.
- `dim`: latent dimension `D`.
- `batch`: batch size `B` (number of columns).

# Keyword Arguments
- `radius_min=0.0`: minimum radius in `[0, 1]` for each column.

# Returns
- `z`: `Float32` matrix of size `D×B`.
"""
sample_latent_l1(rng::Random.AbstractRNG, dim::Integer, batch::Integer; radius_min::Real=0.0) = begin
    D = Int(dim)
    B = Int(batch)
    D > 0 || throw(ArgumentError("dim must be positive"))
    B > 0 || throw(ArgumentError("batch must be positive"))
    rmin = Float32(radius_min)
    (0f0 <= rmin <= 1f0) || throw(ArgumentError("radius_min must be in [0, 1]; got $radius_min"))

    u = randn(rng, Float32, D, B)
    norms = vec(sum(abs.(u); dims=1)) .+ eps(Float32)
    direction = u ./ reshape(norms, 1, :)
    radius = rmin .+ (1f0 - rmin) .* rand(rng, Float32, 1, B)  # independent per column in [rmin, 1]
    return direction .* radius
end

_batch_norm(z::AbstractMatrix, norm_kind::Symbol) = begin
    norm_kind === :l1 && return vec(sum(abs.(z); dims=1))
    norm_kind === :l2 && return sqrt.(vec(sum(abs2.(z); dims=1)))
    norm_kind === :linf && return vec(maximum(abs.(z); dims=1))
    throw(ArgumentError("unsupported norm_kind=$(repr(norm_kind)); expected :l1, :l2, or :linf"))
end

"""
    sample_latent(rng, dim, batch; norm_kind=:l1, radius_min=0.0) -> z

Sample a latent batch `z` whose columns lie inside the unit ball of the selected norm, i.e.
`‖z[:, j]‖ ≤ 1` where the norm is one of `:l1`, `:l2`, or `:linf`.

Construction (per column):
1. Draw a random direction `u ~ N(0, I)`.
2. Normalize it to have the chosen norm equal to 1.
3. Draw a radius `r ~ Uniform(radius_min, 1)` and scale: `z = r * u / ‖u‖`.

# Arguments
- `rng`: random number generator.
- `dim`: latent dimension `D`.
- `batch`: batch size `B` (number of columns).

# Keyword Arguments
- `norm_kind=:l1`: norm used for normalization (`:l1`, `:l2`, or `:linf`).
- `radius_min=0.0`: minimum radius in `[0, 1]` for each column.

# Returns
- `z`: `Float32` matrix of size `D×B`.
"""
sample_latent(rng::Random.AbstractRNG, dim::Integer, batch::Integer; norm_kind::Symbol=:l1, radius_min::Real=0.0) = begin
    D = Int(dim)
    B = Int(batch)
    D > 0 || throw(ArgumentError("dim must be positive"))
    B > 0 || throw(ArgumentError("batch must be positive"))
    rmin = Float32(radius_min)
    (0f0 <= rmin <= 1f0) || throw(ArgumentError("radius_min must be in [0, 1]; got $radius_min"))

    u = randn(rng, Float32, D, B)
    norms = _batch_norm(u, norm_kind) .+ eps(Float32)
    direction = u ./ reshape(norms, 1, :)
    radius = rmin .+ (1f0 - rmin) .* rand(rng, Float32, 1, B)  # independent per column in [rmin, 1]
    return direction .* radius
end

"""
    decode(model_decode, model_encode, context; kwargs...) -> u
    decode(model_decode, model_encode, context; kwargs...) -> result

Adversarial/consistency decode via latent optimization.

This overload solves for a latent `z` (length `model_decode.dim`) that minimizes the mean of:
1) the norm of `z` itself, and
2) the norm of the re-encoded latent after decoding with `model_decode` and encoding with `model_encode`:
   `z_cross = encode(model_encode, decode(model_decode, z, context), context)`.

The optimized control is then `u = decode(model_decode, z*, context)`.

# Arguments
- `model_decode`: [`InvertibleCoupling`](@ref) used for decoding `z -> u`.
- `model_encode`: [`InvertibleCoupling`](@ref) used for encoding `u -> z_cross`.
- `context`: context vector of length `model_decode.context_dim`.

# Keyword Arguments
- `algo=:LN_BOBYQA`: NLopt algorithm symbol.
- `init_z=nothing`: initial latent guess (defaults to zeros).
- `max_time=Inf`: NLopt time cap (`maxtime`, seconds).
- `seed=rand(1:10000)`: NLopt seed.
- `norm_kind=:l1`: norm used in the objective (`:l1`, `:l2`, or `:linf`).
- `u_len=nothing`: if provided, return only `u[1:u_len]`.
- `return_meta=false`: if `true`, return a named tuple with fields `u`, `z`, `objective`, `result`.

# Returns
- If `return_meta=false`: `u` (vector).
- If `return_meta=true`: named tuple `(; u, z, objective, result)`.
"""
function decode(model_decode::InvertibleCoupling,
                model_encode::InvertibleCoupling,
                context::AbstractVector;
                algo::Symbol=:LN_BOBYQA,
                init_z=nothing,
                max_time::Real=Inf,
                seed::Integer=rand(1:10000),
                norm_kind::Symbol=:l1,
                u_len=nothing,
                return_meta::Bool=false)
    model_decode.dim == model_encode.dim ||
        throw(DimensionMismatch("model_encode.dim must match model_decode.dim"))
    model_decode.context_dim == model_encode.context_dim ||
        throw(DimensionMismatch("model_encode.context_dim must match model_decode.context_dim"))
    length(context) == model_decode.context_dim ||
        throw(DimensionMismatch("context must have length $(model_decode.context_dim)"))

    D = model_decode.dim
    init = init_z === nothing ? zeros(Float32, D) : Float32.(init_z isa AbstractVector ? init_z : vec(init_z))
    length(init) == D || throw(DimensionMismatch("init_z must have length $D; got length=$(length(init))"))

    c32 = Float32.(context)

    function objective_fn(z::AbstractVector, grad::AbstractVector)
        z32 = Float32.(z)
        n_self = _batch_norm(reshape(z32, :, 1), norm_kind)[1]

        u = decode(model_decode, z32, c32)
        z_cross = encode(model_encode, u, c32)
        n_cross = _batch_norm(reshape(z_cross, :, 1), norm_kind)[1]

        return 0.5 * (Float64(n_self) + Float64(n_cross))
    end

    opt = NLopt.Opt(algo, D)
    NLopt.min_objective!(opt, objective_fn)
    NLopt.maxtime!(opt, max_time)
    NLopt.srand(seed)

    min_f, min_z, ret = NLopt.optimize(opt, init)
    z_star = Float32.(min_z)
    u_star = decode(model_decode, z_star, c32)
    if u_len !== nothing
        u_len_i = Int(u_len)
        u_len_i >= 1 || throw(ArgumentError("u_len must be ≥ 1; got $u_len"))
        length(u_star) >= u_len_i ||
            throw(DimensionMismatch("decoded u has length $(length(u_star)), cannot slice to u_len=$u_len_i"))
        u_star = u_star[1:u_len_i]
    end

    return return_meta ? (; u=u_star, z=z_star, objective=Float64(min_f), result=ret) : u_star
end

"""
    CouplingLayer(dim, context_dim, hidden_dim, depth, affine;
                  logscale_clamp=2.0, rng=Random.default_rng())

One coupling layer with a fixed random permutation and a GLU-MLP conditioner.

This layer implements either:
- additive coupling (`affine=false`): `y_b = x_b + t(x_a, c)`
- affine coupling (`affine=true`): `y_b = x_b .* exp(s(x_a, c)) + t(x_a, c)`,
  with bounded `s = logscale_clamp * tanh(raw_s)`

The input is first permuted with a fixed random permutation (sampled at construction),
then split into `x_a` (pass-through) and `x_b` (transformed), then inverted permutation is applied.
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

    split_a = dim_int ÷ 2
    split_a >= 1 || throw(ArgumentError("dim must be ≥ 2 for coupling layers; got $dim_int"))
    split_b = dim_int - split_a

    perm = randperm(rng, dim_int)
    invperm = similar(perm)
    for (i, p) in enumerate(perm)
        invperm[p] = i
    end

    in_dim = split_a + ctx_int
    out_dim = affine ? (2 * split_b) : split_b
    if flip
        # Complementary mask: transform x_a (split_a) conditioned on x_b (split_b) and context.
        in_dim = split_b + ctx_int
        out_dim = affine ? (2 * split_a) : split_a
    end
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
    forward(layer, x, context) -> y

Forward transform for a single coupling layer.
"""
function forward(layer::CouplingLayer, x::AbstractMatrix, context::AbstractMatrix)
    B = _check_batch(x, "x", layer.dim)
    _check_context(context, B, layer.context_dim)

    x32 = x isa Matrix{Float32} ? x : Float32.(Matrix(x))
    c32 = context isa Matrix{Float32} ? context : Float32.(Matrix(context))

    x_p = x32[layer.perm, :]
    x_a = @view x_p[1:layer.split_a, :]
    x_b = @view x_p[(layer.split_a + 1):end, :]

    if !layer.flip
        # Standard mask: transform x_b conditioned on x_a and context.
        cond = vcat(x_a, c32)  # stable even when context_dim==0 (0×B)
        params = layer.net(cond)
        if layer.affine
            raw_s = @view params[1:layer.split_b, :]
            t = @view params[(layer.split_b + 1):end, :]
            s = layer.logscale_clamp .* tanh.(raw_s)
            y_b = x_b .* exp.(s) .+ t
            y_p = vcat(x_a, y_b)
            return y_p[layer.invperm, :]
        else
            y_b = x_b .+ params
            y_p = vcat(x_a, y_b)
            return y_p[layer.invperm, :]
        end
    else
        # Complementary mask: transform x_a conditioned on x_b and context.
        cond = vcat(x_b, c32)
        params = layer.net(cond)
        if layer.affine
            raw_s = @view params[1:layer.split_a, :]
            t = @view params[(layer.split_a + 1):end, :]
            s = layer.logscale_clamp .* tanh.(raw_s)
            y_a = x_a .* exp.(s) .+ t
            y_p = vcat(y_a, x_b)
            return y_p[layer.invperm, :]
        else
            y_a = x_a .+ params
            y_p = vcat(y_a, x_b)
            return y_p[layer.invperm, :]
        end
    end
end

"""
    inverse(layer, z, context) -> x

Inverse transform for a single coupling layer.
"""
function inverse(layer::CouplingLayer, z::AbstractMatrix, context::AbstractMatrix)
    B = _check_batch(z, "z", layer.dim)
    _check_context(context, B, layer.context_dim)

    z32 = z isa Matrix{Float32} ? z : Float32.(Matrix(z))
    c32 = context isa Matrix{Float32} ? context : Float32.(Matrix(context))

    z_p = z32[layer.perm, :]
    z_a = @view z_p[1:layer.split_a, :]
    z_b = @view z_p[(layer.split_a + 1):end, :]

    if !layer.flip
        # Standard mask: invert x_b conditioned on z_a and context.
        cond = vcat(z_a, c32)
        params = layer.net(cond)
        if layer.affine
            raw_s = @view params[1:layer.split_b, :]
            t = @view params[(layer.split_b + 1):end, :]
            s = layer.logscale_clamp .* tanh.(raw_s)
            x_b = (z_b .- t) .* exp.(-s)
            x_p = vcat(z_a, x_b)
            return x_p[layer.invperm, :]
        else
            x_b = z_b .- params
            x_p = vcat(z_a, x_b)
            return x_p[layer.invperm, :]
        end
    else
        # Complementary mask: invert x_a conditioned on z_b and context.
        cond = vcat(z_b, c32)
        params = layer.net(cond)
        if layer.affine
            raw_s = @view params[1:layer.split_a, :]
            t = @view params[(layer.split_a + 1):end, :]
            s = layer.logscale_clamp .* tanh.(raw_s)
            x_a = (z_a .- t) .* exp.(-s)
            x_p = vcat(x_a, z_b)
            return x_p[layer.invperm, :]
        else
            x_a = z_a .- params
            x_p = vcat(x_a, z_b)
            return x_p[layer.invperm, :]
        end
    end
end

function InvertibleCoupling(dim::Integer,
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

    flags = spec_mat[3, :]
    all(f -> (f == 0 || f == 1), flags) ||
        throw(ArgumentError("spec[3, :] must contain only 0 (additive) or 1 (affine)"))

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

    return InvertibleCoupling(dim_int, ctx_int, spec_mat, Float32(logscale_clamp), layers)
end

"""
    encode(net, x, context) -> z

Forward pass of the invertible coupling network: maps `x` to latent `z`.

# Arguments
- `net`: [`InvertibleCoupling`](@ref).
- `x`: input data (vector `D` or matrix `D×B`).
- `context`: conditioning context (vector `C` or matrix `C×B`), must match `x` shape.

# Returns
- `z`: latent with the same shape as `x` (vector or matrix).
"""
function encode(net::InvertibleCoupling, x::AbstractMatrix, context::AbstractMatrix)
    ndims(x) == 2 || throw(ArgumentError("x must be a (dim × batch) matrix"))
    size(x, 1) == net.dim || throw(DimensionMismatch("x must have $(net.dim) rows"))
    ndims(context) == 2 || throw(ArgumentError("context must be a (context_dim × batch) matrix"))
    size(context, 1) == net.context_dim || throw(DimensionMismatch("context must have $(net.context_dim) rows"))
    size(context, 2) == size(x, 2) || throw(DimensionMismatch("context batch must match x batch"))

    z = x isa Matrix{Float32} ? x : Float32.(Matrix(x))
    c32 = context isa Matrix{Float32} ? context : Float32.(Matrix(context))

    for layer in net.layers
        z = forward(layer, z, c32)
    end
    return z
end

function encode(net::InvertibleCoupling, x::AbstractVector, context::AbstractVector)
    length(x) == net.dim || throw(DimensionMismatch("x must have length $(net.dim); got $(length(x))"))
    length(context) == net.context_dim ||
        throw(DimensionMismatch("context must have length $(net.context_dim); got $(length(context))"))
    z = encode(net, reshape(x, :, 1), reshape(context, :, 1))
    return vec(z)
end

"""
    decode(net, z, context) -> x

Inverse pass: maps latent `z` back to data space `x`.
"""
function decode(net::InvertibleCoupling, z::AbstractMatrix, context::AbstractMatrix)
    ndims(z) == 2 || throw(ArgumentError("z must be a (dim × batch) matrix"))
    size(z, 1) == net.dim || throw(DimensionMismatch("z must have $(net.dim) rows"))
    ndims(context) == 2 || throw(ArgumentError("context must be a (context_dim × batch) matrix"))
    size(context, 1) == net.context_dim || throw(DimensionMismatch("context must have $(net.context_dim) rows"))
    size(context, 2) == size(z, 2) || throw(DimensionMismatch("context batch must match z batch"))

    x = z isa Matrix{Float32} ? z : Float32.(Matrix(z))
    c32 = context isa Matrix{Float32} ? context : Float32.(Matrix(context))

    for layer in Iterators.reverse(net.layers)
        x = inverse(layer, x, c32)
    end
    return x
end

function decode(net::InvertibleCoupling, z::AbstractVector, context::AbstractVector)
    length(z) == net.dim || throw(DimensionMismatch("z must have length $(net.dim); got $(length(z))"))
    length(context) == net.context_dim ||
        throw(DimensionMismatch("context must have length $(net.context_dim); got $(length(context))"))
    x = decode(net, reshape(z, :, 1), reshape(context, :, 1))
    return vec(x)
end

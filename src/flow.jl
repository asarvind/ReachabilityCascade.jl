# Conditional Normalizing Flow with Proper Masking (Flux)
# ------------------------------------------------------
# Implements a conditional RealNVP-style flow with:
#   • Permute (fixed per block)
#   • Conditional Affine/Additive Coupling (s,t masked to unmasked dims)
#   • FlowBlock(dim, cdim; ...) and Flow(dim, cdim, nblocks; ...)
#   • loglikelihood / nll under standard Normal prior
# Notes:
#   • Inputs are cast to Float32 before Dense calls for speed.
#   • `clamp` is NOT trainable; only s_net/t_net are trainable.

# -------------------- utils --------------------
make_mask(dim; pattern=:alternating, flip::Bool=false) = begin
    m = falses(dim)
    if pattern === :alternating
        @inbounds for i in 1:dim
            m[i] = isodd(i)
        end
        if flip; m .= .!m; end
    elseif pattern === :half
        h = div(dim, 2)
        if flip
            m[h+1:end] .= true
        else
            m[1:h]      .= true
        end
    else
        error("Unknown mask pattern: $(pattern)")
    end
    m
end

mask_cols(x, m::AbstractVector{Bool}) = x .* reshape(m, :, 1)
_zerold(x) = zeros(eltype(x), size(x, 2))

# -------------------- permutation --------------------
struct Permute
    p::Vector{Int}
    ip::Vector{Int}
end
Permute(p::AbstractVector{<:Integer}) = Permute(collect(Int, p), invperm(collect(Int, p)))
Flux.@layer Permute
(P::Permute)(x) = Float32.(x[P.p, :]), _zerold(x)
inverse(P::Permute, y) = Float32.(y[P.ip, :]), _zerold(y)

# -------------------- conditional coupling --------------------
struct AffineCoupling{M,SN,TN,T}
    mask::M          # Vector{Bool}
    s_net::SN        # maps (dim+cdim, B) -> (dim, B)
    t_net::TN        # maps (dim+cdim, B) -> (dim, B)
    clamp::T         # Float32; stabilizes scales via tanh
    mode::Symbol     # :affine or :additive
    cdim::Int        # context dimension
end
# Only networks are trainable; keep clamp/mask/mode metadata fixed
Flux.@layer AffineCoupling trainable=(s_net, t_net)

feature_mlp(in_dim::Int, hidden::Int, out_dim::Int) = Chain(
    Dense(in_dim, hidden, relu),
    Dense(hidden, hidden, relu),
    Dense(hidden, out_dim)
)

function AffineCoupling(dim::Int, cdim::Int; mask=make_mask(dim), hidden::Int=128,
                        clamp::Real=2.0, mode::Symbol=:affine)
    in_dim = dim + cdim
    s_net  = feature_mlp(in_dim, hidden, dim)
    t_net  = feature_mlp(in_dim, hidden, dim)
    AffineCoupling(mask, s_net, t_net, Float32(clamp), mode, cdim)
end

# Forward: (x, c) -> (y, logdet)
function (L::AffineCoupling)(x, c)
    x, c = Float32.(x), Float32.(c)
    @assert size(c,1) == L.cdim "context has wrong feature dimension"
    @assert size(c,2) == size(x,2) "context and x must share batch size"
    m   = L.mask
    xm  = mask_cols(x, m)
    xum = mask_cols(x, .!m)
    input = vcat(xm, c)
    if L.mode === :affine
        s_full = L.clamp .* tanh.(L.s_net(input))
        t_full = L.t_net(input)
        s = mask_cols(s_full, .!m)   # apply only to unmasked dims
        t = mask_cols(t_full, .!m)
        yum = xum .* exp.(s) .+ t
        y   = xm .+ yum
        logdet = vec(sum(s; dims=1))
        return y, logdet
    else
        t_full = L.t_net(input)
        t = mask_cols(t_full, .!m)
        yum = xum .+ t
        y   = xm .+ yum
        return y, _zerold(x)
    end
end

# Inverse: (y, c) -> (x, logdet)
function inverse(L::AffineCoupling, y, c)
    y, c = Float32.(y), Float32.(c)
    @assert size(c,1) == L.cdim "context has wrong feature dimension"
    @assert size(c,2) == size(y,2) "context and y must share batch size"
    m   = L.mask
    ym  = mask_cols(y, m)
    yum = mask_cols(y, .!m)
    input = vcat(ym, c)
    if L.mode === :affine
        s_full = L.clamp .* tanh.(L.s_net(input))
        t_full = L.t_net(input)
        s = mask_cols(s_full, .!m)
        t = mask_cols(t_full, .!m)
        xum = (yum .- t) .* exp.(-s)
        x   = ym .+ xum
        logdet = -vec(sum(s; dims=1))
        return x, logdet
    else
        t_full = L.t_net(input)
        t = mask_cols(t_full, .!m)
        xum = yum .- t
        x   = ym .+ xum
        return x, _zerold(y)
    end
end

# -------------------- block: permute → coupling --------------------
struct FlowBlock{P,C}
    perm::P
    coup::C
end
Flux.@layer FlowBlock

function (F::FlowBlock)(x, c)
    x, c = Float32.(x), Float32.(c)
    y, ld1 = F.perm(x)
    y, ld2 = F.coup(y, c)
    y, ld1 .+ ld2
end

function inverse(F::FlowBlock, y, c)
    y, c = Float32.(y), Float32.(c)
    x, ld2 = inverse(F.coup, y, c)
    x, ld1 = inverse(F.perm, x)
    x, ld1 .+ ld2
end

function FlowBlock(dim::Int, cdim::Int; hidden::Int=128, clamp::Real=2.0, mode::Symbol=:affine,
                   mask_pattern::Symbol=:alternating, flip_mask::Bool=false, seed=nothing)
    if seed !== nothing; Random.seed!(seed); end
    perm = Permute(randperm(dim))
    mask = make_mask(dim; pattern=mask_pattern, flip=flip_mask)
    coup = AffineCoupling(dim, cdim; mask=mask, hidden=hidden, clamp=clamp, mode=mode)
    FlowBlock(perm, coup)
end

# -------------------- stacked flow --------------------
struct Flow{B}
    blocks::Vector{B}
    cdim::Int
end
Flux.@layer Flow

function (F::Flow)(x, c)
    x, c = Float32.(x), Float32.(c)
    @assert size(c,1) == F.cdim
    ld = _zerold(x)
    for blk in F.blocks
        x, l = blk(x, c)
        ld .+= l
    end
    x, ld
end

function inverse(F::Flow, y, c)
    y, c = Float32.(y), Float32.(c)
    @assert size(c,1) == F.cdim
    ld = _zerold(y)
    for blk in Iterators.reverse(F.blocks)
        y, l = inverse(blk, y, c)
        ld .+= l
    end
    y, ld
end

function Flow(dim::Int, cdim::Int, nblocks::Integer; hidden::Int=128, clamp::Real=2.0,
              mode::Symbol=:affine, mask_pattern::Symbol=:alternating,
              alternate_masks::Bool=true, seed=nothing)
    if seed !== nothing; Random.seed!(seed); end
    blocks = Vector{FlowBlock}(undef, nblocks)
    for i in 1:nblocks
        blocks[i] = FlowBlock(dim, cdim; hidden=hidden, clamp=clamp, mode=mode,
                              mask_pattern=mask_pattern,
                              flip_mask=(alternate_masks && isodd(i)))
    end
    Flow(blocks, cdim)
end

# -------------------- log-likelihood helpers --------------------
function loglikelihood(flow::Flow, x, c)
    x, c = Float32.(x), Float32.(c)
    z, logdet = flow(x, c)
    d = size(z, 1)
    const_term = -0.5f0 * d * log(2f0 * π)
    logpz = const_term .- 0.5f0 .* vec(sum(z .^ 2, dims=1))
    logpz .+ logdet
end

nll(flow::Flow, x, c) = -mean(loglikelihood(flow, x, c))

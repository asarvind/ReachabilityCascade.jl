# ===================== Normalizing Flow Transformer =========================

using Flux

const _SUPPORTED_COUPLINGS = (:affine, :additive)

# -- Helpers -----------------------------------------------------------------

softclamp(x; limit=3.0f0) = limit .* tanh.(x ./ limit)

default_position_fn(pos::Integer, max_seq_len::Integer) = Float32[pos / max_seq_len]

function _build_position_table(position_fn::Function,
                               position_dim::Integer,
                               max_seq_len::Integer)
    table = Array{Float32}(undef, position_dim, max_seq_len)
    for idx in 1:max_seq_len
        raw = position_fn(idx, max_seq_len)
        @assert raw isa AbstractVector "position_fn must return an AbstractVector"
        @assert length(raw) == position_dim "position_fn output dimension mismatch"
        table[:, idx] = Float32.(raw)
    end
    return table
end

function _build_position_tensor(position_table::AbstractMatrix,
                                seq_len::Integer,
                                batch::Integer,
                                ::Type{T}) where {T<:AbstractFloat}
    @assert seq_len > 0 "Sequence length must be positive"
    @assert seq_len <= size(position_table, 2) "Sequence length exceeds stored positional encoding length"
    base = T.(position_table[:, 1:seq_len])
    pos_tensor = reshape(base, size(position_table, 1), seq_len, 1)
    return batch == 1 ? pos_tensor : repeat(pos_tensor, 1, 1, batch)
end

default_mask(d_model::Integer; start_with_pass::Bool=true) = BitVector(((start_with_pass ? 1 : 0) + i) % 2 == 1 for i in 0:d_model-1)

function flipmask(mask::BitVector)
    flipped = .!mask
    @assert any(flipped) && any(.!flipped) "Flipped mask must keep at least one pass-through and one transformed dimension"
    return flipped
end

_apply_layernorm(norm::LayerNorm, x::AbstractArray) = reshape(norm(reshape(x, size(x, 1), :)), size(x))
_apply_layernorm(::Nothing, x::AbstractArray) = x

# -- Flow Transformer Layer ---------------------------------------------------

struct FlowTransformerLayer
    pass_idx::Vector{Int}
    trans_idx::Vector{Int}
    perm::Vector{Int}
    inv_perm::Vector{Int}
    expand::Dense
    attention::MultiHeadAttention
    norm1::Union{LayerNorm, Nothing}
    ff::Chain
    norm2::Union{LayerNorm, Nothing}
    cond_proj::Dense
    cond_norm::Union{LayerNorm, Nothing}
    shift_proj::Dense
    scale_proj::Union{Dense, Nothing}
    coupling::Symbol
    clamp::Float32
    activation_scale::Float32
    embed_dim::Int
    d_model::Int
    context_dim::Int
    position_dim::Int
end

Flux.@layer FlowTransformerLayer

"""
    FlowTransformerLayer(d_model, context_dim, num_heads, ff_hidden,
                         mask, coupling, clamp; activation_scale=1.0, use_layernorm=true)

Construct a single transformer-conditioned coupling block. Pass-through activations,
context, and positional features are concatenated, expanded to `ff_hidden`, processed
by multi-head attention/MLP, and finally reduced to derive shift/scale parameters for
the masked channels.

# Arguments
- `d_model::Int`: feature dimension of the flow (must match input `D`).
- `context_dim::Int`: size of the conditioning context vector.
- `num_heads::Int`: attention heads in `MultiHeadAttention`.
- `ff_hidden::Int`: embedding width used by the attention/MLP stack (must be divisible by `num_heads`).
- `mask::BitVector`: Boolean mask choosing pass-through (`true`) vs transformed (`false`) channels.
- `coupling::Symbol`: either `:affine` (default) or `:additive`.
- `clamp::Real`: soft clamp limit applied to affine log-scales.
- `activation_scale::Real=1.0`: fixed scalar applied to the shift/scale outputs to mitigate exploding activations.

# Keyword Arguments
- `use_layernorm::Bool=true`: toggles the layer-normalization blocks.
- `position_dim::Int`: number of positional features concatenated into the coupling networks.

# Returns
A configured `FlowTransformerLayer`.
"""
function FlowTransformerLayer(d_model::Integer,
                              context_dim::Integer,
                              num_heads::Integer,
                              ff_hidden::Integer,
                              mask::BitVector,
                              coupling::Symbol,
                              clamp::Real;
                              activation_scale::Real=1.0,
                              use_layernorm::Bool=true,
                              position_dim::Integer)
    @assert length(mask) == d_model "Mask length must equal d_model"
    @assert any(mask) "Mask must include at least one pass-through dimension"
    @assert any(.!mask) "Mask must include at least one transformed dimension"
    @assert num_heads > 0 "Number of heads must be positive"
    @assert d_model % num_heads == 0 "d_model must be divisible by num_heads"
    pass_idx = findall(identity, mask)
    trans_idx = findall(!, mask)
    Dp = length(pass_idx)
    Dt = length(trans_idx)
    combined_dim = Dp + context_dim + position_dim
    @assert ff_hidden % num_heads == 0 "ff_hidden must be divisible by num_heads"
    expand = Dense(combined_dim, ff_hidden, gelu)
    attention = MultiHeadAttention(ff_hidden; nheads=num_heads, dropout_prob=0f0)
    norm1 = use_layernorm ? LayerNorm(ff_hidden) : nothing
    ff = Chain(Dense(ff_hidden, ff_hidden, gelu), Dense(ff_hidden, ff_hidden))
    norm2 = use_layernorm ? LayerNorm(ff_hidden) : nothing
    cond_proj = Dense(ff_hidden, Dp)
    cond_norm = use_layernorm ? LayerNorm(Dp) : nothing
    shift_proj = Dense(Dp + context_dim + position_dim, Dt)
    scale_proj = coupling == :affine ? Dense(Dp + context_dim + position_dim, Dt) : nothing
    perm = vcat(pass_idx, trans_idx)
    inv_perm = invperm(perm)
    FlowTransformerLayer(pass_idx,
                         trans_idx,
                         perm,
                         inv_perm,
                         expand,
                         attention,
                         norm1,
                         ff,
                         norm2,
                         cond_proj,
                         cond_norm,
                         shift_proj,
                         scale_proj,
                         coupling,
                         Float32(clamp),
                         Float32(activation_scale),
                         ff_hidden,
                         d_model,
                         context_dim,
                         position_dim)
end

function (layer::FlowTransformerLayer)(x::AbstractArray,
                                       context::AbstractArray,
                                       position::AbstractArray;
                                       inverse::Bool=false)
    D, L, B = size(x)
    ctx_matrix = ndims(context) == 1 ? reshape(context, length(context), 1) : context
    @assert D == layer.d_model "Input feature dimension mismatch"
    @assert size(ctx_matrix, 1) == layer.context_dim "Context feature dimension mismatch"
    @assert size(ctx_matrix, 2) == B "Context batch size mismatch"
    @assert size(position, 1) == layer.position_dim "Position feature dimension mismatch"
    @assert size(position, 2) == L "Position sequence length mismatch"
    @assert size(position, 3) == B "Position batch size mismatch"
    xp = @views x[layer.pass_idx, :, :]
    xt = @views x[layer.trans_idx, :, :]
    ctx_features = reshape(ctx_matrix, layer.context_dim, 1, B)
    ctx_features = repeat(ctx_features, 1, L, 1)
    combined = cat(xp, ctx_features, position; dims=1)
    combined_flat = reshape(combined, size(combined, 1), :)
    embed = layer.expand(combined_flat)
    embed = reshape(embed, layer.embed_dim, L, B)
    attn_out, _ = layer.attention(embed)
    resid1 = embed .+ attn_out
    normed1 = _apply_layernorm(layer.norm1, resid1)
    ff_out = layer.ff(reshape(normed1, layer.embed_dim, :))
    ff_out = reshape(ff_out, layer.embed_dim, L, B)
    resid2 = normed1 .+ ff_out
    normed2 = _apply_layernorm(layer.norm2, resid2)
    cond_base = layer.cond_proj(reshape(normed2, layer.embed_dim, :))
    cond_base = reshape(cond_base, length(layer.pass_idx), L, B)
    cond_base = _apply_layernorm(layer.cond_norm, cond_base)
    features = cat(cond_base, ctx_features, position; dims=1)
    features_flat = reshape(features, size(features, 1), :)
    shift_flat = layer.shift_proj(features_flat) .* layer.activation_scale
    t = reshape(shift_flat, length(layer.trans_idx), L, B)
    if layer.coupling == :affine
        @assert layer.scale_proj !== nothing
        scale_flat = layer.scale_proj(features_flat) .* layer.activation_scale
        log_s = reshape(scale_flat, length(layer.trans_idx), L, B)
        log_s = softclamp(log_s; limit=layer.clamp)
    else
        log_s = zeros(eltype(x), size(t)...)
    end
    if !inverse
        yt = xt .* exp.(log_s) .+ t
        logdet = vec(sum(log_s; dims=(1, 2)))
        stacked = cat(xp, yt; dims=1)
        y = stacked[layer.inv_perm, :, :]
        return y, logdet
    else
        xt_rec = (xt .- t) .* exp.(-log_s)
        logdet = vec(-sum(log_s; dims=(1, 2)))
        stacked = cat(xp, xt_rec; dims=1)
        y = stacked[layer.inv_perm, :, :]
        return y, logdet
    end
end

# -- Encoder / Decoder -------------------------------------------------------

struct FlowTransformer
    layers::Vector{FlowTransformerLayer}
    d_model::Int
    max_seq_len::Int
    position_dim::Int
    position_table::Matrix{Float32}
end

Flux.@layer FlowTransformer

"""
    (flow::FlowTransformer)(x, context; inverse=false)

Apply the normalizing-flow transformer.

# Arguments
- `x::AbstractArray`: tensor of shape `(d_model, seq_len, batch)`.
- `context::AbstractArray`: context matrix `(context_dim, batch)` or vector `(context_dim)`.
- `inverse::Bool=false`: when `false`, runs the forward (density) direction; set to
  `true` to invert the flow.

# Returns
- Forward (`inverse=false`): tuple `(latent, logdet)`.
- Inverse (`inverse=true`): reconstructed tensor with the same shape as `x`.
"""
function (flow::FlowTransformer)(x::AbstractArray, context::AbstractArray; inverse::Bool=false)
    orig_ndims = ndims(x)
    x_norm = orig_ndims == 3 ? x :
             orig_ndims == 2 ? reshape(x, size(x, 1), size(x, 2), 1) :
             reshape(x, size(x, 1), 1, 1)
    ctx = ndims(context) == 1 ? reshape(context, length(context), 1) : context
    D, L, B = size(x_norm)
    @assert D == flow.d_model "Input feature dimension mismatch"
    @assert L <= flow.max_seq_len "Sequence length exceeds configured max_seq_len"
    @assert size(ctx, 2) == B "Context batch size mismatch"
    Tinput = eltype(x_norm)
    pos = _build_position_tensor(flow.position_table, L, B, Tinput)
    restore_shape(arr) = begin
        if orig_ndims == 3
            return arr
        elseif orig_ndims == 2
            return reshape(arr, size(arr, 1), size(arr, 2))
        else
            return reshape(arr, size(arr, 1))
        end
    end
    if !inverse
        state = x_norm
        total_logdet = zeros(eltype(x_norm), B)
        for layer in flow.layers
            state, logdet = layer(state, ctx, pos; inverse=false)
            total_logdet = total_logdet .+ logdet
        end
        return restore_shape(state), total_logdet
    else
        state = x_norm
        for layer in reverse(flow.layers)
            state, _ = layer(state, ctx, pos; inverse=true)
        end
        return restore_shape(state)
    end
end

"""
    FlowTransformer(d_model, context_dim; kwargs...)

Construct a stack of transformer-conditioned coupling layers that can run in
either direction using the `inverse` keyword. Each layer expands the concatenated
pass-through/context/position features to `ff_hidden`, applies attention/MLP in that
space, and then projects back down to derive coupling parameters.

# Arguments
- `d_model::Int`: feature dimension processed per time step.
- `context_dim::Int`: context vector dimension supplied to every layer.

# Keyword Arguments
- `num_layers::Int=4`: number of flow layers.
- `num_heads::Int=1`: attention heads inside each layer.
- `ff_hidden::Int=64`: embedding width for the internal attention/MLP stack (must be divisible by `num_heads`).
- `coupling::Symbol=:affine`: coupling type (`:affine` or `:additive`).
- `mask::AbstractVector{Bool}`: optional custom pass-through mask; defaults to alternating pattern.
- `clamp::Real=3.0f0`: soft clamp limit for affine log-scales.
- `activation_scale::Real=1.0`: multiplier applied to the shift/scale projections in each layer.
- `use_layernorm::Bool=true`: disable to drop layer-normalization sublayers.
- `max_seq_len::Int=512`: maximum sequence length used to parameterise positional encodings.
- `position_fn::Function=default_position_fn`: callback mapping `(position, max_seq_len)`
  to a positional feature vector (must have fixed length). Used to build the stored
  positional table when none is provided.
- `position_dim::Union{Nothing,Int}=nothing`: explicitly set the positional feature
  length; defaults to the length returned by `position_fn` or the height of `position_table`.
- `position_table::Union{Nothing,AbstractMatrix}=nothing`: precomputed positional feature
  table of size `(position_dim, max_seq_len)`; overrides `position_fn` when supplied.

# Notes
- Positional features are concatenated with the pass-through activations and context
  prior to the coupling networks. Ensure `max_seq_len` covers the largest sequence length
  you expect so positional slices are available for every step.

# Returns
A `FlowTransformer` object; call it with `inverse=false` for density evaluation
and with `inverse=true` for sampling.
"""
function FlowTransformer(d_model::Integer,
                         context_dim::Integer;
                         num_layers::Integer=4,
                         num_heads::Integer=1,
                         ff_hidden::Integer=64,
                         coupling::Symbol=:affine,
                         mask::Union{Nothing, AbstractVector{Bool}}=nothing,
                         clamp::Real=3.0f0,
                         activation_scale::Real=1.0,
                         use_layernorm::Bool=true,
                         max_seq_len::Integer=512,
                         position_fn::Function=default_position_fn,
                         position_dim::Union{Nothing, Integer}=nothing,
                         position_table::Union{Nothing, AbstractMatrix}=nothing)
    coupling = Symbol(coupling)
    @assert coupling in _SUPPORTED_COUPLINGS "Unsupported coupling type $(coupling)"
    @assert d_model > 1 "d_model must be greater than 1"
    @assert num_layers > 0 "Number of layers must be positive"
    @assert context_dim > 0 "Context dimension must be positive"
    @assert ff_hidden > 0 "Feed-forward hidden size must be positive"
    @assert ff_hidden % num_heads == 0 "ff_hidden must be divisible by num_heads"
    @assert max_seq_len > 0 "max_seq_len must be positive"
    position_table_matrix, position_dim_val = if position_table === nothing
        sample = position_fn(1, max_seq_len)
        @assert sample isa AbstractVector "position_fn must return an AbstractVector"
        inferred_dim = position_dim === nothing ? length(sample) : position_dim
        @assert length(sample) == inferred_dim "Provided position_dim does not match position_fn output length"
        (_build_position_table(position_fn, inferred_dim, max_seq_len), inferred_dim)
    else
        @assert size(position_table, 2) == max_seq_len "Provided position_table must have max_seq_len columns"
        dim_val = position_dim === nothing ? size(position_table, 1) : position_dim
        @assert size(position_table, 1) == dim_val "Provided position_dim does not match position_table"
        (Float32.(position_table), dim_val)
    end
    base_mask = mask === nothing ? default_mask(d_model) : BitVector(mask)
    layers = FlowTransformerLayer[]
    current_mask = copy(base_mask)
    for i in 1:num_layers
        layer = FlowTransformerLayer(d_model,
                                     context_dim,
                                     num_heads,
                                     ff_hidden,
                                     current_mask,
                                     coupling,
                                     clamp;
                                     activation_scale=activation_scale,
                                     use_layernorm=use_layernorm,
                                     position_dim=position_dim_val)
        push!(layers, layer)
        current_mask = mask === nothing ? flipmask(current_mask) : current_mask
    end
    return FlowTransformer(layers, d_model, max_seq_len, position_dim_val, position_table_matrix)
end

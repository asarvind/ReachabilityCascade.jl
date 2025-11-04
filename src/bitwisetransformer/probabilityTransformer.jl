using Flux
using Flux: Dense, MultiHeadAttention, sigmoid
import Zygote: @nograd

_colmat(x::AbstractVector) = reshape(Float32.(x), :, 1)
_colmat(x::AbstractMatrix) = Float32.(x)

_sequence_matrix(seq::AbstractVector{<:Real}) = reshape(Float32.(seq), :, 1)
_sequence_matrix(seq::AbstractMatrix{<:Real}) = Float32.(seq)

function _positional_encoding(embed_dim::Integer, seq_len::Integer)::Matrix{Float32}
    pe = zeros(Float32, embed_dim, seq_len)
    for pos in 0:seq_len-1
        for i in 0:(div(embed_dim, 2)-1)
            angle = pos / (10000.0f0 ^ (2f0 * i / embed_dim))
            pe[2i + 1, pos + 1] = sin(angle)
            if 2i + 2 <= embed_dim
                pe[2i + 2, pos + 1] = cos(angle)
            end
        end
        if isodd(embed_dim)
            pe[embed_dim, pos + 1] = sin(pos)
        end
    end
    return pe
end

@nograd _positional_encoding

function _normalize_prior_sequence(prior, prior_dim::Integer)
    if prior isa AbstractVector{<:Real}
        prior_dim == 1 || throw(ArgumentError("vector prior requires prior_dim == 1"))
        sequence_length = length(prior)
        tensor = reshape(Float32.(prior), sequence_length, prior_dim, 1)
        return (:vector, sequence_length), tensor
    elseif prior isa AbstractMatrix{<:Real}
        sequence_length, features = size(prior)
        features == prior_dim || throw(ArgumentError("second dimension must equal prior_dim"))
        tensor = reshape(Float32.(prior), sequence_length, prior_dim, 1)
        return (:matrix, (sequence_length, prior_dim)), tensor
    elseif prior isa AbstractArray{<:Real,3}
        sequence_length, features, batch_size = size(prior)
        features == prior_dim || throw(ArgumentError("second dimension must equal prior_dim"))
        tensor = Float32.(prior)
        return (:tensor, (sequence_length, prior_dim, batch_size)), tensor
    else
        throw(ArgumentError("prior_sequence must be vector, matrix, or 3D tensor"))
    end
end

function _restore_sequence_shape(tensor::Array{Float32,3}, info)
    kind, dims = info
    if kind === :vector
        sequence_length = dims
        return reshape(tensor[:, 1, 1], sequence_length)
    elseif kind === :matrix
        sequence_length, prior_dim = dims
        return reshape(tensor[:, :, 1], sequence_length, prior_dim)
    else
        return tensor
    end
end

"""
    ProbabilityTransformer(context_dim; kwargs...)

Map a context vector and a boolean prior sequence to a sequence of probabilities via transformer-style attention.

# Arguments
- `context_dim::Integer`: dimensionality of the context vector.

# Keyword Arguments
- `prior_dim::Integer=1`: dimensionality of the prior token at each sequence position.
- `embed_dim::Integer=64`: embedding size for tokens and context.
- `heads::Integer=4`: number of attention heads.
- `ff_hidden::Integer=128`: hidden width of the feed-forward block.
- `activation`: activation function used in projection and feed-forward layers (default `Flux.relu`).

# Returns
`ProbabilityTransformer` functor that can be applied directly to `(context, prior_sequence)`.
"""
struct ProbabilityTransformer{CP,TP,MH,FF1,FF2,PH}
    context_proj::CP
    token_proj::TP
    attention::MH
    ff1::FF1
    ff2::FF2
    prob_head::PH
    embed_dim::Int
    heads::Int
    bit_position::Int
    prior_dim::Int
end

Flux.@layer ProbabilityTransformer

function ProbabilityTransformer(context_dim::Integer;
                                prior_dim::Integer=1,
                                embed_dim::Integer=64,
                                heads::Integer=4,
                                ff_hidden::Integer=128,
                                activation::Function=Flux.relu,
                                bit_position::Integer=1)::ProbabilityTransformer
    context_proj = Dense(context_dim, embed_dim, activation)
    token_proj = Dense(prior_dim, embed_dim, activation)
    attention = MultiHeadAttention(embed_dim => embed_dim => embed_dim,
                                   nheads=heads, dropout_prob=0f0, bias=false)
    ff1 = Dense(embed_dim, ff_hidden, activation)
    ff2 = Dense(ff_hidden, embed_dim, identity)
    prob_head = Dense(embed_dim, prior_dim, identity)
    return ProbabilityTransformer(context_proj, token_proj, attention,
                                  ff1, ff2, prob_head, embed_dim, heads, bit_position, prior_dim)
end

"""
    (net::ProbabilityTransformer)(context, prior_sequence)

Forward application of the probability transformer. Returns a tuple `(values, probabilities)`
matching the shape of the supplied `prior_sequence`.

# Arguments
- `context`: context vector or matrix `(context_dim, batch_size)`.
- `prior_sequence`: real-valued sequence. Supported shapes:
  * `(sequence_length,)` when `prior_dim == 1` and the batch size is one.
  * `(sequence_length, prior_dim)` for a single sample with multiple dependent variables.
  * `(sequence_length, prior_dim, batch_size)` for batched evaluation.
# Returns
`Tuple{Any,Any}` where the first element is the real-valued output sequence and the second is the probability sequence.
"""
function (net::ProbabilityTransformer)(context::AbstractVecOrMat,
                                       prior_sequence::Union{AbstractVecOrMat,AbstractArray{<:Real,3}})
    prior_info, prior_tensor = _normalize_prior_sequence(prior_sequence, net.prior_dim)
    sequence_length, _, batch_size = size(prior_tensor)

    context_mat = _colmat(context)
    if size(context_mat, 2) == 1 && batch_size > 1
        context_mat = repeat(context_mat, 1, batch_size)
    elseif size(context_mat, 2) != batch_size
        throw(ArgumentError("context columns must match batch size"))
    end

    tokens = permutedims(prior_tensor, (2, 1, 3))
    tokens = reshape(tokens, net.prior_dim, sequence_length * batch_size)
    token_embed = net.token_proj(tokens)
    token_embed = reshape(token_embed, net.embed_dim, sequence_length, batch_size)

    pos = _positional_encoding(net.embed_dim, sequence_length)
    token_embed = token_embed .+ reshape(pos, net.embed_dim, sequence_length, 1)

    context_embed = net.context_proj(context_mat)
    context_token = reshape(context_embed, net.embed_dim, 1, batch_size)

    key_values = cat(token_embed, context_token; dims=2)
    attn_out, _ = net.attention(token_embed, key_values, key_values)
    attn_flat = reshape(attn_out, net.embed_dim, sequence_length * batch_size)

    hidden = net.ff2(net.ff1(attn_flat))
    logits = net.prob_head(hidden)
    logits = reshape(logits, net.prior_dim, sequence_length, batch_size)
    probs_tensor = sigmoid.(logits)
    probs_tensor = permutedims(probs_tensor, (2, 1, 3))
    logits_tensor = permutedims(logits, (2, 1, 3))

    bools = probs_tensor .>= 0.5f0
    scale = Float32(2.0f0^(net.bit_position - 1))
    values_tensor = Float32.(bools) .* scale .+ prior_tensor

    probs = _restore_sequence_shape(probs_tensor, prior_info)
    values = _restore_sequence_shape(values_tensor, prior_info)
    logits_out = _restore_sequence_shape(logits_tensor, prior_info)

    return (values=values, probabilities=probs, logits=logits_out)
end

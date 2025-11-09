using Flux

"""
    sinusoidal_embedding(seq_len, dim; max_period=10_000)

Build a standard sinusoidal positional embedding matrix of size `(dim, seq_len)`.
"""
function sinusoidal_embedding(seq_len::Integer, dim::Integer; max_period::Real=10_000)
    @assert seq_len > 0 "Sequence length must be positive"
    @assert dim > 0 "Embedding dimension must be positive"
    pe = zeros(Float32, dim, seq_len)
    position = collect(0:seq_len-1)
    div_term = exp.((-2 .* (0:div(dim, 2)-1)) .* log(max_period) ./ dim)
    for i in 1:seq_len
        for j in 1:2:dim
            idx = (j + 1) ÷ 2
            value = position[i] / div_term[idx]
            pe[j, i] = sin(value)
            if j + 1 <= dim
                pe[j + 1, i] = cos(value)
            end
        end
    end
    return pe
end

_apply_layernorm(norm::LayerNorm, x::AbstractArray) = reshape(norm(reshape(x, size(x, 1), :)), size(x))
_apply_layernorm(::Nothing, x::AbstractArray) = x

struct SimpleTransformerEncoder
    attention::MultiHeadAttention
    norm1::LayerNorm
    ffn::Chain
    norm2::LayerNorm
end

Flux.@layer SimpleTransformerEncoder

function SimpleTransformerEncoder(d_model::Integer,
                                  num_heads::Integer,
                                  ff_dim::Integer)
    @assert d_model % num_heads == 0 "d_model must be divisible by num_heads"
    attention = MultiHeadAttention(d_model; nheads=num_heads, dropout_prob=0f0)
    norm1 = LayerNorm(d_model)
    ffn = Chain(Dense(d_model, ff_dim, gelu), Dense(ff_dim, d_model))
    norm2 = LayerNorm(d_model)
    return SimpleTransformerEncoder(attention, norm1, ffn, norm2)
end

function (layer::SimpleTransformerEncoder)(x::AbstractArray)
    attn_out, _ = layer.attention(x)
    resid1 = x .+ attn_out
    normed1 = _apply_layernorm(layer.norm1, resid1)
    ff_out = layer.ff(reshape(normed1, size(normed1, 1), :))
    ff_out = reshape(ff_out, size(normed1))
    resid2 = normed1 .+ ff_out
    _apply_layernorm(layer.norm2, resid2)
end

struct ResidualControlTransformer
    encoders::Vector{SimpleTransformerEncoder}
    output_proj::Dense
    pos_dim::Int
    control_dim::Int
    state_dim::Int
    context_dim::Int
    token_dim::Int
end

Flux.@layer ResidualControlTransformer

"""
    ResidualControlTransformer(control_dim, state_dim, context_dim;
                               num_heads=4, ff_dim=256,
                               num_layers=2, pos_dim=32)

Create a transformer that predicts the control correction (delta) given a control
sequence, a matching state sequence, and a context vector. Each timestep's token
is the concatenation of `[u_t; x_t; context; position]`.
"""
function ResidualControlTransformer(control_dim::Integer,
                                     state_dim::Integer,
                                     context_dim::Integer;
                                 num_heads::Integer=4,
                                 ff_dim::Integer=256,
                                 num_layers::Integer=2,
                                 pos_dim::Integer=32)
    @assert num_layers > 0 "Need at least one transformer encoder layer"
    token_dim = control_dim + state_dim + context_dim + pos_dim
    @assert token_dim % num_heads == 0 "Token dimension must be divisible by num_heads"
    encoders = [SimpleTransformerEncoder(token_dim, num_heads, ff_dim)
                for _ in 1:num_layers]
    output_proj = Dense(token_dim, control_dim)
    return ResidualControlTransformer(encoders,
                                      output_proj,
                                      pos_dim,
                                      control_dim,
                                      state_dim,
                                      context_dim,
                                      token_dim)
end

function (model::ResidualControlTransformer)(u_seq::AbstractMatrix,
                                              x_seq::AbstractMatrix,
                                              context::AbstractVector;
                                              pos_embedding::Union{Nothing, AbstractMatrix}=nothing)
    control_dim, seq_len = size(u_seq)
    @assert control_dim == model.control_dim "Control dimension mismatch"
    @assert size(x_seq, 2) == seq_len "State sequence length mismatch"
    @assert size(x_seq, 1) == model.state_dim "State dimension mismatch"
    @assert length(context) == model.context_dim "Context dimension mismatch"
    pos = pos_embedding === nothing ? sinusoidal_embedding(seq_len, model.pos_dim)
                                    : pos_embedding
    @assert size(pos, 1) == model.pos_dim && size(pos, 2) == seq_len "Positional embedding size mismatch"
    ctx_rep = repeat(reshape(context, :, 1), 1, seq_len)
    combined = vcat(u_seq, x_seq, ctx_rep, pos)
    z = reshape(combined, model.token_dim, seq_len, 1)
    for encoder in model.encoders
        z = encoder(z)
    end
    z2 = reshape(z, model.token_dim, :)
    deltas = model.output_proj(z2)
    deltas = reshape(deltas, model.control_dim, seq_len)
    @assert size(deltas) == size(u_seq)
    return deltas
end

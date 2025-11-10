using Flux

"""
    sinusoidal_embedding(seq_len, dim, max_period)

Build a differentiable sinusoidal positional embedding matrix of size `(dim, seq_len)`.
"""
function sinusoidal_embedding(seq_len::Integer, dim::Integer, max_period::Real)
    @assert seq_len > 0 "Sequence length must be positive"
    @assert dim > 0 "Embedding dimension must be positive"
    half_dim = cld(dim, 2)
    positions = reshape(Float32.(0:seq_len-1), seq_len, 1)
    div_term = Float32.(exp.((-2 .* (0:half_dim-1)) .* log(max_period) ./ dim))
    angles = positions ./ reshape(div_term, 1, half_dim)
    sin_part = sin.(angles)
    cos_part = cos.(angles)
    rows = map(1:dim) do row_idx
        col_idx = cld(row_idx, 2)
        values = isodd(row_idx) ? sin_part[:, col_idx] : cos_part[:, col_idx]
        reshape(values, 1, seq_len)
    end
    cat(rows...; dims=1)
end

_apply_layernorm(norm::LayerNorm, x::AbstractArray) = reshape(norm(reshape(x, size(x, 1), :)), size(x))
_apply_layernorm(::Nothing, x::AbstractArray) = x

struct SimpleTransformerEncoder
    attention::MultiHeadAttention
    norm1::LayerNorm
    mlp::Chain
    norm2::LayerNorm
end

Flux.@layer SimpleTransformerEncoder

function SimpleTransformerEncoder(hidden_dim::Integer,
                                  num_heads::Integer)
    @assert hidden_dim % num_heads == 0 "hidden_dim must be divisible by num_heads"
    attention = MultiHeadAttention(hidden_dim; nheads=num_heads, dropout_prob=0f0)
    norm1 = LayerNorm(hidden_dim)
    mlp = Chain(Dense(hidden_dim, hidden_dim, gelu), Dense(hidden_dim, hidden_dim))
    norm2 = LayerNorm(hidden_dim)
    return SimpleTransformerEncoder(attention, norm1, mlp, norm2)
end

function (layer::SimpleTransformerEncoder)(x::AbstractArray)
    attn_out, _ = layer.attention(x)
    resid1 = x .+ attn_out
    normed1 = _apply_layernorm(layer.norm1, resid1)
    mlp_out = layer.mlp(reshape(normed1, size(normed1, 1), :))
    mlp_out = reshape(mlp_out, size(normed1))
    resid2 = normed1 .+ mlp_out
    _apply_layernorm(layer.norm2, resid2)
end

struct ResidualControlTransformer
    input_proj::Dense
    encoders::Vector{SimpleTransformerEncoder}
    output_proj::Dense
    pos_dim::Int
    max_period::Float32
    control_dim::Int
    state_dim::Int
    context_dim::Int
    token_dim::Int
    hidden_dim::Int
end

Flux.@layer ResidualControlTransformer

"""
    ResidualControlTransformer(control_dim, state_dim, context_dim;
                               hidden_dim=128, num_heads=4,
                               num_layers=2, pos_dim=32, max_period=10_000.0)

Create a transformer that predicts the control correction (delta) given a control
sequence, a matching state sequence, and a context vector. Each timestep's token
is the concatenation of `[u_t; x_t; context; position]`, which is linearly projected
to `hidden_dim` before entering the transformer stack.
"""
function ResidualControlTransformer(control_dim::Integer,
                                     state_dim::Integer,
                                     context_dim::Integer;
                                 hidden_dim::Integer=128,
                                 num_heads::Integer=4,
                                 num_layers::Integer=2,
                                 pos_dim::Integer=32,
                                 max_period::Real=10_000.0)
    @assert num_layers > 0 "Need at least one transformer encoder layer"
    token_dim = control_dim + state_dim + context_dim + pos_dim
    input_proj = Dense(token_dim, hidden_dim)
    encoders = [SimpleTransformerEncoder(hidden_dim, num_heads)
                for _ in 1:num_layers]
    output_proj = Dense(hidden_dim, control_dim)
    return ResidualControlTransformer(input_proj,
                                      encoders,
                                      output_proj,
                                      pos_dim,
                                      Float32(max_period),
                                      control_dim,
                                      state_dim,
                                      context_dim,
                                      token_dim,
                                      hidden_dim)
end

function (model::ResidualControlTransformer)(u_seq::AbstractMatrix,
                                              x_seq::AbstractMatrix,
                                              context::AbstractVector;
                                              pos_embedding::Union{Nothing, AbstractMatrix}=nothing)
    u_seq32 = Float32.(u_seq)
    x_seq32 = Float32.(x_seq)
    context32 = Float32.(context)
    control_dim, seq_len = size(u_seq32)
    @assert control_dim == model.control_dim "Control dimension mismatch"
    @assert size(x_seq32, 2) == seq_len "State sequence length mismatch"
    @assert size(x_seq32, 1) == model.state_dim "State dimension mismatch"
    @assert length(context32) == model.context_dim "Context dimension mismatch"
    pos = pos_embedding === nothing ?
          sinusoidal_embedding(seq_len, model.pos_dim, model.max_period) :
          pos_embedding
    pos32 = Float32.(pos)
    @assert size(pos32, 1) == model.pos_dim && size(pos32, 2) == seq_len "Positional embedding size mismatch"
    ctx_rep = repeat(reshape(context32, :, 1), 1, seq_len)
    combined = vcat(u_seq32, x_seq32, ctx_rep, pos32)
    projected = model.input_proj(combined)
    z = reshape(projected, model.hidden_dim, seq_len, 1)
    for encoder in model.encoders
        z = encoder(z)
    end
    z2 = reshape(z, model.hidden_dim, :)
    deltas = model.output_proj(z2)
    deltas = reshape(deltas, model.control_dim, seq_len)
    @assert size(deltas) == size(u_seq)
    return deltas
end

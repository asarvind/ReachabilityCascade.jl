using Flux

"""
    sinusoidal_embedding(seq_len, dim, max_period)

Build a differentiable sinusoidal positional embedding of size `(dim, seq_len)`.
"""
function sinusoidal_embedding(seq_len::Integer, dim::Integer, max_period::Real)
    @assert seq_len > 0 && dim > 0
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

struct SimpleTransformerBlock
    attention::MultiHeadAttention
    norm1::LayerNorm
    mlp::Chain
    norm2::LayerNorm
end

Flux.@layer SimpleTransformerBlock

function SimpleTransformerBlock(hidden_dim::Integer, num_heads::Integer)
    @assert hidden_dim % num_heads == 0
    attention = MultiHeadAttention(hidden_dim; nheads=num_heads, dropout_prob=0f0)
    norm1 = LayerNorm(hidden_dim)
    mlp = Chain(Dense(hidden_dim, hidden_dim, gelu), Dense(hidden_dim, hidden_dim))
    norm2 = LayerNorm(hidden_dim)
    SimpleTransformerBlock(attention, norm1, mlp, norm2)
end

function (block::SimpleTransformerBlock)(x::AbstractArray)
    attn_out, _ = block.attention(x)
    resid1 = x .+ attn_out
    normed1 = _apply_layernorm(block.norm1, resid1)
    mlp_out = block.mlp(reshape(normed1, size(normed1, 1), :))
    mlp_out = reshape(mlp_out, size(normed1))
    resid2 = normed1 .+ mlp_out
    _apply_layernorm(block.norm2, resid2)
end

struct SimpleSequenceTransformer
    input_proj::Dense
    blocks::Vector{SimpleTransformerBlock}
    output_proj::Dense
    seq_dim::Int
    context_dim::Int
    hidden_dim::Int
    pos_dim::Int
    max_period::Float32
end

Flux.@layer SimpleSequenceTransformer

"""
    SimpleSequenceTransformer(seq_dim, context_dim;
                              hidden_dim=128, num_heads=4,
                              num_layers=2, pos_dim=32,
                              max_period=10_000.0)

Create a transformer that maps `(sequence, context)` to another sequence with the
same dimensionality and length.

# Arguments
- `seq_dim::Integer`: number of features per timestep in the input/output sequence.
- `context_dim::Integer`: size of the conditioning context vector.

# Keyword Arguments
- `hidden_dim::Integer=128`: internal embedding width for the transformer.
- `num_heads::Integer=4`: number of attention heads (must divide `hidden_dim`).
- `num_layers::Integer=2`: number of transformer blocks.
- `pos_dim::Integer=32`: dimensionality of positional features concatenated to each token.
- `max_period::Real=10_000.0`: scaling constant used by the default sinusoidal embedding.

# Returns
- `SimpleSequenceTransformer` that can be called as `model(seq, context; pos_embedding=...)`.
"""
function SimpleSequenceTransformer(seq_dim::Integer,
                                   context_dim::Integer;
                                   hidden_dim::Integer=128,
                                   num_heads::Integer=4,
                                   num_layers::Integer=2,
                                   pos_dim::Integer=32,
                                   max_period::Real=10_000.0)
    @assert num_layers > 0
    token_dim = seq_dim + context_dim + pos_dim
    input_proj = Dense(token_dim, hidden_dim)
    blocks = [SimpleTransformerBlock(hidden_dim, num_heads) for _ in 1:num_layers]
    output_proj = Dense(hidden_dim, seq_dim)
    SimpleSequenceTransformer(input_proj,
                              blocks,
                              output_proj,
                              seq_dim,
                              context_dim,
                              hidden_dim,
                              pos_dim,
                              Float32(max_period))
end

function _build_positional_features(seq_len::Integer,
                                    pos_dim::Integer,
                                    max_period::Real,
                                    pos_embedding)
    if pos_embedding === nothing
        return sinusoidal_embedding(seq_len, pos_dim, max_period)
    elseif pos_embedding isa AbstractMatrix
        @assert size(pos_embedding, 1) == pos_dim "Positional embedding dimension mismatch"
        @assert size(pos_embedding, 2) == seq_len "Positional embedding length mismatch"
        return Float32.(pos_embedding)
    elseif pos_embedding isa Function
        cols = map(1:seq_len) do pos
            vec = Float32.(pos_embedding(pos))
            @assert length(vec) == pos_dim "Custom positional embedding must return vector of length $pos_dim"
            reshape(vec, :, 1)
        end
        return reduce(hcat, cols)
    else
        error("pos_embedding must be nothing, a matrix, or a function mapping position -> vector")
    end
end

"""
    (model::SimpleSequenceTransformer)(seq, context; pos_embedding=nothing)

Forward pass that accepts either 2D (`features × length`) or 3D (`features × length × batch`) sequences.
`context` can be a vector (`context_dim`) or a matrix (`context_dim × batch`). The optional `pos_embedding`
can be:
  * `nothing` (default sinusoidal embedding),
  * an explicit matrix of size `pos_dim × seq_len`,
  * a function `pos::Int -> AbstractVector` returning `pos_dim`-length positional features.

Returns a sequence tensor with the same shape as `seq`.
"""
function (model::SimpleSequenceTransformer)(seq::AbstractArray,
                                            context::AbstractArray;
                                            pos_embedding=nothing)
    seq32 = Float32.(seq)
    nd = ndims(seq32)
    @assert nd == 2 || nd == 3 "sequence must be 2D or 3D"
    seq_dim, seq_len = size(seq32, 1), size(seq32, 2)
    @assert seq_dim == model.seq_dim "Sequence feature dimension mismatch"
    batch = nd == 2 ? 1 : size(seq32, 3)
    seq3 = nd == 2 ? reshape(seq32, seq_dim, seq_len, 1) : seq32

    ctx32 = Float32.(context)
    if ndims(ctx32) == 1
        @assert length(ctx32) == model.context_dim "Context dimension mismatch"
        ctx_mat = repeat(reshape(ctx32, :, 1), 1, batch)
    elseif ndims(ctx32) == 2
        @assert size(ctx32, 1) == model.context_dim "Context dimension mismatch"
        @assert size(ctx32, 2) == batch "Context batch mismatch"
        ctx_mat = ctx32
    else
        error("context must be vector or (context_dim, batch) matrix")
    end
    ctx_rep = reshape(ctx_mat, model.context_dim, 1, batch)
    ctx_rep = repeat(ctx_rep, 1, seq_len, 1)

    pos = _build_positional_features(seq_len, model.pos_dim, model.max_period, pos_embedding)
    pos3 = reshape(pos, model.pos_dim, seq_len, 1)
    pos3 = repeat(pos3, 1, 1, batch)

    tokens = cat(seq3, ctx_rep, pos3; dims=1)
    z = model.input_proj(reshape(tokens, size(tokens, 1), seq_len * batch))
    z = reshape(z, model.hidden_dim, seq_len, batch)
    for block in model.blocks
        z = block(z)
    end
    z2 = reshape(z, model.hidden_dim, seq_len * batch)
    out = model.output_proj(z2)
    out = reshape(out, model.seq_dim, seq_len, batch)
    out = nd == 2 ? dropdims(out; dims=3) : out
    @assert size(out) == size(seq32) "Output sequence dimension mismatch"
    out
end

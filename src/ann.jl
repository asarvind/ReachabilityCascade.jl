# === TransformerBlock ===

"""
    struct TransformerBlock

Represents a single transformer block consisting of multi-head self-attention,
a feedforward network, and optional layer normalization.

# Fields
- `mha::MultiHeadAttention`: The multi-head self-attention module.
- `norm1::Union{LayerNorm, Nothing}`: First normalization layer or `nothing`.
- `ff::Chain`: Feedforward subnetwork consisting of two dense layers with GELU activation.
- `norm2::Union{LayerNorm, Nothing}`: Second normalization layer or `nothing`.
"""
struct TransformerBlock
    mha::MultiHeadAttention
    norm1::Union{LayerNorm, Nothing}
    ff::Chain
    norm2::Union{LayerNorm, Nothing}
end

Flux.@layer TransformerBlock

"""
    TransformerBlock(embed_dim::Int, num_heads::Int; use_norm::Bool=true) -> TransformerBlock

Constructs a `TransformerBlock` with the specified embedding dimension and number
of attention heads. Applies optional layer normalization.

# Arguments
- `embed_dim::Int`: Dimension of each embedding vector.
- `num_heads::Int`: Number of attention heads.
- `use_norm::Bool`: If true, applies `LayerNorm` after attention and feedforward stages. Default is `true`.

# Returns
- A `TransformerBlock` instance.
"""
function TransformerBlock(embed_dim::Int, num_heads::Int; use_norm::Bool=true)
    mha = MultiHeadAttention(embed_dim => embed_dim => embed_dim, nheads=num_heads)
    ff = Chain(Dense(embed_dim, 4 * embed_dim, gelu), Dense(4 * embed_dim, embed_dim))
    norm = use_norm ? LayerNorm(embed_dim) : nothing
    return TransformerBlock(mha, norm, ff, norm)
end

"""
    (tb::TransformerBlock)(x::AbstractArray) -> AbstractArray

Applies the transformer block to a sequence of embedded vectors.

# Arguments
- `x::AbstractArray`: Input of shape `(embed_dim, seq_len, batch_size)`

# Returns
- Output of the same shape `(embed_dim, seq_len, batch_size)`
"""
function (tb::TransformerBlock)(x::AbstractArray)
    attn_out, _ = tb.mha(x, x, x)
    x = x + attn_out
    x = tb.norm1 === nothing ? x : tb.norm1(x)

    ff_out = tb.ff(x)
    x = x + ff_out
    x = tb.norm2 === nothing ? x : tb.norm2(x)
    return x
end

# === transformer ===

"""
    transformer(input_dim::Int, embed_dim::Int, seq_len::Int, num_blocks::Int, num_heads::Int,
                out_dim::Int; use_norm::Bool = true) -> Chain

Constructs a transformer-style network for input vectors.

This network first projects the input vector to a sequence of embedded vectors,
then processes the sequence through multiple `TransformerBlock`s, and finally flattens and projects
to the desired output dimension.

# Arguments
- `input_dim::Int`: Dimensionality of each input vector.
- `embed_dim::Int`: Dimensionality of the embedding vectors.
- `seq_len::Int`: Length of the embedded sequence.
- `num_blocks::Int`: Number of transformer blocks.
- `num_heads::Int`: Number of attention heads per block.
- `out_dim::Int`: Final output dimensionality after flattening.
- `use_norm::Bool`: Whether to use layer normalization in each block. Default is `true`.

# Returns
- A `Chain` that maps input tensors of shape `(input_dim,)` or `(input_dim, batch_size)`
  to output tensors of shape `(out_dim,)` or `(out_dim, batch_size)`.
"""
function transformer(input_dim::Int, embed_dim::Int, seq_len::Int, num_blocks::Int, num_heads::Int,
                     out_dim::Int; use_norm::Bool = true)
    input_proj = Dense(input_dim, embed_dim * seq_len)
    blocks = [TransformerBlock(embed_dim, num_heads; use_norm=use_norm) for _ in 1:num_blocks]
    output_proj = Dense(embed_dim * seq_len, out_dim)

    return Chain(
        x -> reshape(input_proj(x), embed_dim, seq_len, :),
        blocks...,
        x -> reshape(x, :, size(x, 3)),
        output_proj
    )
end

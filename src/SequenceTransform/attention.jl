using Flux

_binary_pos_table(max_len::Int) = begin
    max_len > 0 || throw(ArgumentError("max_len must be positive"))
    bits = max(1, ceil(Int, log2(max_len)))
    tbl = falses(bits, max_len)
    for i in 1:max_len
        v = i
        for b in 1:bits
            tbl[b, i] = (v & 0x1) == 0x1
            v >>= 1
        end
    end
    tbl
end

"""
    AttentionFFN(in_dim::Int, hidden_dim::Int, out_dim::Int;
                 activation=Flux.gelu, max_seq_len::Int=512, nheads::Int=1, add_pos::Bool=true)

    Multi-head self-attention with binary positional encoding, followed by
    a two-layer feed-forward projection: Dense -> activation -> Dense (linear output). Input/outputs
    use the convention `(features, seq_len, batch)`.

Arguments:
- `in_dim`: Feature dimension of the input (excluding positional encodings).
    - `hidden_dim`: Placeholder (retained for signature compatibility; not used internally).
    - `out_dim`: Output feature dimension.
    - `activation`: Activation applied before the attention (default `Flux.gelu`).
- `max_seq_len`: Maximum supported sequence length for cached positional encodings.
- `nheads`: Number of attention heads (default 2).
    """
struct AttentionFFN{A,G1,G2,N}
    mha::A
    pos_table::BitMatrix
    proj1::G1
    proj2::G2
    add_pos::Bool
    ln_att::N
end

function AttentionFFN(in_dim::Int, hidden_dim::Int, out_dim::Int;
                      activation=Flux.gelu, max_seq_len::Int=512, nheads::Int=1, add_pos::Bool=true)
    pos_tbl = _binary_pos_table(max_seq_len)
    d_model = add_pos ? in_dim + size(pos_tbl, 1) : in_dim
    d_model % nheads == 0 || throw(ArgumentError("d_model=$d_model must be divisible by nheads=$nheads"))
    mha = Flux.MultiHeadAttention(d_model; nheads=nheads)
    proj1 = Dense(d_model, hidden_dim, activation)
    proj2 = Dense(hidden_dim, out_dim)
    ln_att = Flux.LayerNorm(d_model)
    return AttentionFFN(mha, pos_tbl, proj1, proj2, add_pos, ln_att)
end

Flux.@layer AttentionFFN

function (m::AttentionFFN)(x::AbstractArray)
    # x: (features, seq_len, batch)
    was_2d = ndims(x) == 2
    seq_len = size(x, 2)
    seq_len <= size(m.pos_table, 2) || throw(ArgumentError("sequence length $seq_len exceeds precomputed positional table length $(size(m.pos_table, 2))"))

    x_with_pos = if m.add_pos
        pos_slice = view(m.pos_table, :, 1:seq_len)
        T = eltype(x)
        if ndims(x) == 3
            batch_size = size(x, 3)
            pos_encoding = repeat(reshape(T.(pos_slice), size(pos_slice, 1), seq_len, 1), 1, 1, batch_size)
        else
            pos_encoding = T.(pos_slice)
        end
        vcat(x, pos_encoding)
    else
        x
    end
    was_2d = ndims(x_with_pos) == 2
    x_att = was_2d ? reshape(x_with_pos, size(x_with_pos,1), size(x_with_pos,2), 1) : x_with_pos

    att_out = m.mha(x_att)
    att = att_out isa Tuple ? att_out[1] : att_out

    if ndims(att) == 3
        f, t, b = size(att)
        att_flat = reshape(att, f, t * b)
        proj_flat = m.proj2(m.proj1(m.ln_att(att_flat)))
        out_f = size(proj_flat, 1)
        proj = reshape(proj_flat, out_f, t, b)
        return was_2d ? reshape(proj, out_f, t) : proj
    else
        proj = m.proj2(m.proj1(m.ln_att(att)))
        return proj
    end
end

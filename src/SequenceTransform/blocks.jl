using Flux
using ChainRulesCore


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
ForwardCumsumBlock(in_dim::Int, out_dim::Int, activation=Flux.σ; max_seq_len::Int=512)

A block that applies a feedforward layer, then a cumulative sum along the sequence dimension (forward),
with a precomputed binary positional encoding.
"""
struct ForwardCumsumBlock{F}
    ff::F
    pos_table::BitMatrix
    div_table::Vector{Int}
end

function ForwardCumsumBlock(in_dim::Int, out_dim::Int, activation=Flux.σ; max_seq_len::Int=512)
    pos_tbl = _binary_pos_table(max_seq_len)
    div_tbl = collect(1:max_seq_len)
    ff = glu_mlp(in_dim + size(pos_tbl, 1), out_dim, out_dim; act=activation)
    return ForwardCumsumBlock(ff, pos_tbl, div_tbl)
end

function ForwardCumsumBlock(ff::F; max_seq_len::Int=512) where {F}
    pos_tbl = _binary_pos_table(max_seq_len)
    div_tbl = collect(1:max_seq_len)
    return ForwardCumsumBlock{F}(ff, pos_tbl, div_tbl)
end

Flux.@layer ForwardCumsumBlock

function (m::ForwardCumsumBlock)(x::AbstractArray)
    # x shape: (features + context_dim, seq_len, batch)
    seq_len = size(x, 2)
    seq_len <= size(m.pos_table, 2) || throw(ArgumentError("sequence length $seq_len exceeds precomputed positional table length $(size(m.pos_table, 2))"))
    
    pos_slice = view(m.pos_table, :, 1:seq_len)
    T = eltype(x)
    if ndims(x) == 3
        batch_size = size(x, 3)
        pos_encoding = repeat(reshape(T.(pos_slice), size(pos_slice, 1), seq_len, 1), 1, 1, batch_size)
    else
        pos_encoding = T.(pos_slice)
    end
    
    # Concatenate positional encoding along feature dimension
    x_with_pos = vcat(x, pos_encoding)
    
    h = m.ff(x_with_pos) # (out_dim, seq_len, batch)
    
    # Cumulative sum along sequence dimension (dim 2)
    h_cumsum = cumsum(h, dims=2)
    
    # Compute cumulative average
    k_vec = T.(view(m.div_table, 1:seq_len))
    if ndims(h) == 3
        d = reshape(k_vec, 1, seq_len, 1)
    else
        d = reshape(k_vec, 1, seq_len)
    end
    
    return h_cumsum ./ d
end

"""
ReverseCumsumBlock(in_dim::Int, out_dim::Int, activation=Flux.σ; max_seq_len::Int=512)

A block that applies a feedforward layer, then a cumulative sum along the sequence dimension (reverse order),
with a precomputed binary positional encoding.
"""
struct ReverseCumsumBlock{F}
    ff::F
    pos_table::BitMatrix
    div_table::Vector{Int}
end

function ReverseCumsumBlock(in_dim::Int, out_dim::Int, activation=Flux.σ; max_seq_len::Int=512)
    pos_tbl = _binary_pos_table(max_seq_len)
    div_tbl = collect(1:max_seq_len)
    ff = glu_mlp(in_dim + size(pos_tbl, 1), out_dim, out_dim; act=activation)
    return ReverseCumsumBlock(ff, pos_tbl, div_tbl)
end

function ReverseCumsumBlock(ff::F; max_seq_len::Int=512) where {F}
    pos_tbl = _binary_pos_table(max_seq_len)
    div_tbl = collect(1:max_seq_len)
    return ReverseCumsumBlock{F}(ff, pos_tbl, div_tbl)
end

Flux.@layer ReverseCumsumBlock

function (m::ReverseCumsumBlock)(x::AbstractArray)
    seq_len = size(x, 2)
    seq_len <= size(m.pos_table, 2) || throw(ArgumentError("sequence length $seq_len exceeds precomputed positional table length $(size(m.pos_table, 2))"))
    
    pos_slice = view(m.pos_table, :, 1:seq_len)
    pos_slice = reverse(pos_slice, dims=2)
    T = eltype(x)
    if ndims(x) == 3
        batch_size = size(x, 3)
        pos_encoding = repeat(reshape(T.(pos_slice), size(pos_slice, 1), seq_len, 1), 1, 1, batch_size)
    else
        pos_encoding = T.(pos_slice)
    end
    
    # Concatenate positional encoding along feature dimension
    x_with_pos = vcat(x, pos_encoding)
    
    h = m.ff(x_with_pos)
    
    # Reverse cumulative sum along sequence dimension (dim 2)
    # We can reverse, cumsum, then reverse back.
    h_cumsum = reverse(cumsum(reverse(h, dims=2), dims=2), dims=2)
    
    # Compute reverse cumulative average
    # For reverse cumsum, the divisor corresponds to the number of elements summed from the end.
    # At index t (1-based), we have summed elements t, t+1, ..., T.
    # The count is T - t + 1.
    k_vec = T.(view(m.div_table, 1:seq_len))
    if ndims(h) == 3
        d = reshape(reverse(k_vec), 1, seq_len, 1)
    else
        d = reshape(reverse(k_vec), 1, seq_len)
    end
    
    return h_cumsum ./ d
end

"""
DirectBlock(in_dim::Int, out_dim::Int, activation=Flux.σ)

A simple feedforward block with activation, no cumsum or normalization.
"""
struct DirectBlock{F}
    ff::F
end

function DirectBlock(in_dim::Int, out_dim::Int, activation=Flux.σ)
    return DirectBlock(glu_mlp(in_dim, out_dim, out_dim; act=activation))
end

Flux.@layer DirectBlock

function (m::DirectBlock)(x::AbstractArray)
    return m.ff(x)
end

using Flux


"""
    ForwardCumsumBlock(in_dim::Int, out_dim::Int, activation=relu)

A block that applies a feedforward layer, then a cumulative sum along the sequence dimension (forward),
followed by layer normalization.
"""
struct ForwardCumsumBlock{F}
    dense::F
end

function ForwardCumsumBlock(in_dim::Int, out_dim::Int, activation=relu)
    dense = Dense(in_dim + 1 => out_dim, activation)
    return ForwardCumsumBlock(dense)
end

Flux.@layer ForwardCumsumBlock

function (m::ForwardCumsumBlock)(x::AbstractArray)
    # x shape: (features + context_dim, seq_len, batch)
    seq_len = size(x, 2)
    
    # Create positional encoding k/(k+1) for k=1,2,...,seq_len  
    # Use same device as x (GPU/CPU compatible)
    T = eltype(x)
    # Build positions directly on the same device as x
    k = similar(x, seq_len)
    broadcast!(i -> T(i), k, 1:seq_len)
    pos_encoding = k ./ (k .+ T(1))
    
    # Reshape and broadcast positional encoding to match input dimensions
    if ndims(x) == 3
        batch_size = size(x, 3)
        # keep broadcast on-device; avoid CPU `ones` on GPU by adding zeros of the right shape
        pos_encoding = reshape(pos_encoding, 1, seq_len, 1) .+ fill!(similar(x, 1, 1, batch_size), 0)
    else
        pos_encoding = reshape(pos_encoding, 1, seq_len)
    end
    
    # Concatenate positional encoding along feature dimension
    x_with_pos = vcat(x, pos_encoding)
    
    h = m.dense(x_with_pos) # (out_dim, seq_len, batch)
    
    # Cumulative sum along sequence dimension (dim 2)
    h_cumsum = cumsum(h, dims=2)
    
    # Compute cumulative average
    if ndims(h) == 3
        d = reshape(k, 1, seq_len, 1)
    else
        d = reshape(k, 1, seq_len)
    end
    
    return h_cumsum ./ d
end

"""
    ReverseCumsumBlock(in_dim::Int, out_dim::Int, activation=relu)

A block that applies a feedforward layer, then a cumulative sum along the sequence dimension (reverse order),
followed by layer normalization.
"""
struct ReverseCumsumBlock{F}
    dense::F
end

function ReverseCumsumBlock(in_dim::Int, out_dim::Int, activation=relu)
    dense = Dense(in_dim + 1 => out_dim, activation)
    return ReverseCumsumBlock(dense)
end

Flux.@layer ReverseCumsumBlock

function (m::ReverseCumsumBlock)(x::AbstractArray)
    seq_len = size(x, 2)
    
    # Create positional encoding k/(k+1) for k=1,2,...,seq_len, then reverse it
    # Use same device as x (GPU/CPU compatible)
    T = eltype(x)
    # Build positions directly on the same device as x
    k = similar(x, seq_len)
    broadcast!(i -> T(i), k, 1:seq_len)
    pos_encoding = reverse(k ./ (k .+ T(1)))
    
    # Reshape and broadcast positional encoding to match input dimensions
    if ndims(x) == 3
        batch_size = size(x, 3)
        pos_encoding = reshape(pos_encoding, 1, seq_len, 1) .+ fill!(similar(x, 1, 1, batch_size), 0)
    else
        pos_encoding = reshape(pos_encoding, 1, seq_len)
    end
    
    # Concatenate positional encoding along feature dimension
    x_with_pos = vcat(x, pos_encoding)
    
    h = m.dense(x_with_pos)
    
    # Reverse cumulative sum along sequence dimension (dim 2)
    # We can reverse, cumsum, then reverse back.
    h_cumsum = reverse(cumsum(reverse(h, dims=2), dims=2), dims=2)
    
    # Compute reverse cumulative average
    # For reverse cumsum, the divisor corresponds to the number of elements summed from the end.
    # At index t (1-based), we have summed elements t, t+1, ..., T.
    # The count is T - t + 1.
    if ndims(h) == 3
        d = reshape(reverse(k), 1, seq_len, 1)
    else
        d = reshape(reverse(k), 1, seq_len)
    end
    
    return h_cumsum ./ d
end

"""
    DirectBlock(in_dim::Int, out_dim::Int, activation=relu)

A simple feedforward block with activation, no cumsum or normalization.
"""
struct DirectBlock{F}
    dense::F
end

function DirectBlock(in_dim::Int, out_dim::Int, activation=relu)
    return DirectBlock(Dense(in_dim => out_dim, activation))
end

Flux.@layer DirectBlock

function (m::DirectBlock)(x::AbstractArray)
    return m.dense(x)
end

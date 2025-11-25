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
    dense = Dense(in_dim => out_dim, activation)
    return ForwardCumsumBlock(dense)
end

Flux.@layer ForwardCumsumBlock

function (m::ForwardCumsumBlock)(x::AbstractArray)
    # x shape: (features + context_dim, seq_len, batch)
    
    h = m.dense(x) # (out_dim, seq_len, batch)
    
    # Cumulative sum along sequence dimension (dim 2)
    h_cumsum = cumsum(h, dims=2)
    
    # Compute cumulative average
    seq_len = size(h, 2)
    if ndims(h) == 3
        d = reshape(1:seq_len, 1, seq_len, 1)
    else
        d = reshape(1:seq_len, 1, seq_len)
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
    dense = Dense(in_dim => out_dim, activation)
    return ReverseCumsumBlock(dense)
end

Flux.@layer ReverseCumsumBlock

function (m::ReverseCumsumBlock)(x::AbstractArray)
    h = m.dense(x)
    
    # Reverse cumulative sum along sequence dimension (dim 2)
    # We can reverse, cumsum, then reverse back.
    h_cumsum = reverse(cumsum(reverse(h, dims=2), dims=2), dims=2)
    
    # Compute reverse cumulative average
    seq_len = size(h, 2)
    # For reverse cumsum, the divisor corresponds to the number of elements summed from the end.
    # At index t (1-based), we have summed elements t, t+1, ..., T.
    # The count is T - t + 1.
    if ndims(h) == 3
        d = reshape(reverse(1:seq_len), 1, seq_len, 1)
    else
        d = reshape(reverse(1:seq_len), 1, seq_len)
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

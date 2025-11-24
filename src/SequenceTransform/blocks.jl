using Flux


"""
    ForwardCumsumBlock(in_dim::Int, out_dim::Int, activation=relu)

A block that applies a feedforward layer, then a cumulative sum along the sequence dimension (forward),
followed by layer normalization.
"""
struct ForwardCumsumBlock{F, L}
    dense::F
    norm::L
end

function ForwardCumsumBlock(in_dim::Int, out_dim::Int, activation=relu)
    dense = Dense(in_dim => out_dim, activation)
    norm = LayerNorm(out_dim)
    return ForwardCumsumBlock(dense, norm)
end

Flux.@layer ForwardCumsumBlock

function (m::ForwardCumsumBlock)(x::AbstractArray)
    # x shape: (features + context_dim, seq_len, batch)
    
    h = m.dense(x) # (out_dim, seq_len, batch)
    
    # Cumulative sum along sequence dimension (dim 2)
    h_cumsum = cumsum(h, dims=2)
    
    # LayerNorm expects (features, ...)
    # We want to normalize over the feature dimension for each step? 
    # Usually LayerNorm is applied per sample. 
    # Flux LayerNorm(size) normalizes over the first dimension.
    return m.norm(h_cumsum)
end

"""
    ReverseCumsumBlock(in_dim::Int, out_dim::Int, activation=relu)

A block that applies a feedforward layer, then a cumulative sum along the sequence dimension (reverse order),
followed by layer normalization.
"""
struct ReverseCumsumBlock{F, L}
    dense::F
    norm::L
end

function ReverseCumsumBlock(in_dim::Int, out_dim::Int, activation=relu)
    dense = Dense(in_dim => out_dim, activation)
    norm = LayerNorm(out_dim)
    return ReverseCumsumBlock(dense, norm)
end

Flux.@layer ReverseCumsumBlock

function (m::ReverseCumsumBlock)(x::AbstractArray)
    h = m.dense(x)
    
    # Reverse cumulative sum along sequence dimension (dim 2)
    # We can reverse, cumsum, then reverse back.
    h_cumsum = reverse(cumsum(reverse(h, dims=2), dims=2), dims=2)
    
    return m.norm(h_cumsum)
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

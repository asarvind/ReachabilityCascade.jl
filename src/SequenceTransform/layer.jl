using Flux

"""
    ScanMixer(in_dim::Int, hidden_dim::Int, out_dim::Int, activation=relu)

A layer that processes the input sequence through three parallel paths:
1. ForwardCumsumBlock
2. ReverseCumsumBlock
3. DirectBlock

The outputs of these three blocks are concatenated along the feature dimension and projected
to `out_dim` using a final Dense layer.

Arguments:
- `in_dim`: Dimension of the input features.
- `hidden_dim`: Dimension of the features within each of the three internal blocks.
- `out_dim`: Dimension of the output features.
- `activation`: Activation function used in the internal blocks.
"""
struct ScanMixer{A, B, C, P}
    forward_block::A
    reverse_block::B
    direct_block::C
    projection::P
end

function ScanMixer(in_dim::Int, hidden_dim::Int, out_dim::Int, activation=relu)
    forward_block = ForwardCumsumBlock(in_dim, hidden_dim, activation)
    reverse_block = ReverseCumsumBlock(in_dim, hidden_dim, activation)
    direct_block = DirectBlock(in_dim, hidden_dim, activation)
    
    # Concatenated dimension will be 3 * hidden_dim
    projection = glu_mlp(3 * hidden_dim, hidden_dim, out_dim; act=activation)
    
    return ScanMixer(forward_block, reverse_block, direct_block, projection)
end

Flux.@layer ScanMixer

function (m::ScanMixer)(x::AbstractArray)
    # x shape: (in_dim + context_dim, seq_len, batch)
    
    out_fwd = m.forward_block(x)
    out_rev = m.reverse_block(x)
    out_dir = m.direct_block(x)
    
    # Concatenate along feature dimension (dim 1)
    concatenated = cat(out_fwd, out_rev, out_dir, dims=1)
    
    # Project to output dimensions
    return m.projection(concatenated)
end

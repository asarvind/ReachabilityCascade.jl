"""
    Decoder

A neural network that maps latent vectors `z` (from the unit hypercube) and context `c` to samples `x`.

# Fields
- `model::Chain`: The underlying neural network.
- `z_dim::Int`: Dimension of the latent space.
- `ctx_dim::Int`: Dimension of the context vector.
- `x_dim::Int`: Dimension of the output sample.
"""
struct Decoder{M}
    model::M
    z_dim::Int
    ctx_dim::Int
    x_dim::Int
end

Flux.@layer Decoder

"""
    Decoder(z_dim::Integer, ctx_dim::Integer, x_dim::Integer;
            hidden::Integer=128, n_layers::Integer=2, activation=leakyrelu)

Constructs a Decoder.

# Arguments
- `z_dim`: Dimension of the latent space.
- `ctx_dim`: Dimension of the context vector.
- `x_dim`: Dimension of the output sample.
- `hidden`: Number of hidden units in each layer.
- `n_layers`: Number of hidden layers.
- `activation`: Activation function for hidden layers. Defaults to `leakyrelu`.
"""
function Decoder(z_dim::Integer, ctx_dim::Integer, x_dim::Integer;
                 hidden::Integer=128, n_layers::Integer=2, activation=leakyrelu)
    layers = []
    in_dim = z_dim + ctx_dim
    
    # Input layer
    push!(layers, Dense(in_dim => hidden, activation))
    
    # Hidden layers
    for _ in 1:(n_layers - 1)
        push!(layers, Dense(hidden => hidden, activation))
    end
    
    # Output layer (linear activation)
    push!(layers, Dense(hidden => x_dim))
    
    model = Chain(layers...)
    return Decoder(model, z_dim, ctx_dim, x_dim)
end

"""
    (dec::Decoder)(z::AbstractVecOrMat, c::AbstractVecOrMat)

Forward pass of the Decoder.

# Arguments
- `z`: Latent vectors. Shape `(z_dim, batch_size)` or `(z_dim,)`.
- `c`: Context vectors. Shape `(ctx_dim, batch_size)` or `(ctx_dim,)`.

# Returns
- `x`: Output samples. Shape `(x_dim, batch_size)` or `(x_dim,)`.
"""
function (dec::Decoder)(z::AbstractVecOrMat, c::AbstractVecOrMat)
    # Ensure inputs are at least 2D for concatenation if they are vectors
    z_in = ndims(z) == 1 ? reshape(z, :, 1) : z
    c_in = ndims(c) == 1 ? reshape(c, :, 1) : c
    
    @assert size(z_in, 2) == size(c_in, 2) "Batch sizes must match"
    
    input = vcat(z_in, c_in)
    x = dec.model(input)
    
    # If inputs were 1D, return 1D output
    if ndims(z) == 1 && ndims(c) == 1
        return vec(x)
    end
    return x
end

"""
    Encoder

A neural network that maps samples `x` and context `c` to a latent space `z` on the unit hypercube.
The output is constrained to [-1, 1] using a `tanh` activation.

# Fields
- `model::Chain`: The underlying neural network.
- `x_dim::Int`: Dimension of the input sample.
- `ctx_dim::Int`: Dimension of the context vector.
- `z_dim::Int`: Dimension of the latent space.
"""
struct Encoder{M}
    model::M
    x_dim::Int
    ctx_dim::Int
    z_dim::Int
end

Flux.@layer Encoder

"""
    Encoder(x_dim::Integer, ctx_dim::Integer, z_dim::Integer;
            hidden::Integer=128, n_layers::Integer=2, activation=leakyrelu)

Constructs an Encoder.

# Arguments
- `x_dim`: Dimension of the input sample.
- `ctx_dim`: Dimension of the context vector.
- `z_dim`: Dimension of the latent space.
- `hidden`: Number of hidden units in each layer.
- `n_layers`: Number of hidden layers.
- `activation`: Activation function for hidden layers. Defaults to `leakyrelu`.
"""
function Encoder(x_dim::Integer, ctx_dim::Integer, z_dim::Integer;
                 hidden::Integer=128, n_layers::Integer=2, activation=leakyrelu)
    layers = []
    in_dim = x_dim + ctx_dim
    
    # Input layer
    push!(layers, Dense(in_dim => hidden, activation))
    
    # Hidden layers
    for _ in 1:(n_layers - 1)
        push!(layers, Dense(hidden => hidden, activation))
    end
    
    # Output layer with tanh activation
    push!(layers, Dense(hidden => z_dim, tanh))
    
    model = Chain(layers...)
    return Encoder(model, x_dim, ctx_dim, z_dim)
end

"""
    (enc::Encoder)(x::AbstractVecOrMat, c::AbstractVecOrMat)

Forward pass of the Encoder.

# Arguments
- `x`: Input samples. Shape `(x_dim, batch_size)` or `(x_dim,)`.
- `c`: Context vectors. Shape `(ctx_dim, batch_size)` or `(ctx_dim,)`.

# Returns
- `z`: Latent vectors. Shape `(z_dim, batch_size)` or `(z_dim,)`.
"""
function (enc::Encoder)(x::AbstractVecOrMat, c::AbstractVecOrMat)
    # Ensure inputs are at least 2D for concatenation if they are vectors
    x_in = ndims(x) == 1 ? reshape(x, :, 1) : x
    c_in = ndims(c) == 1 ? reshape(c, :, 1) : c
    
    @assert size(x_in, 2) == size(c_in, 2) "Batch sizes must match"
    
    input = vcat(x_in, c_in)
    z = enc.model(input)
    
    # If inputs were 1D, return 1D output
    if ndims(x) == 1 && ndims(c) == 1
        return vec(z)
    end
    return z
end

using Flux

"""
    SequenceTransformation(in_dim::Int, hidden_dim::Int, out_dim::Int, depth::Int, context_dim::Int=0, activation=relu)

A chain of `ScanMixer` layers.

Arguments:
- `in_dim`: Dimension of the input features.
- `hidden_dim`: Dimension of the features within each internal `ScanMixer` block and the output of internal layers.
- `out_dim`: Dimension of the final output features.
- `depth`: Number of `ScanMixer` layers to stack.
- `context_dim`: Dimension of the context vector (optional).
- `activation`: Activation function used in the internal blocks.

Returns a `Flux.Chain` of `ScanMixer` layers.
"""
struct SequenceTransformation{C}
    chain::C
end

function SequenceTransformation(in_dim::Int, hidden_dim::Int, out_dim::Int, depth::Int, context_dim::Int=0, activation=relu)
    layers = []
    
    # First layer: Projects from input dimension to hidden dimension
    push!(layers, ScanMixer(in_dim + context_dim, hidden_dim, hidden_dim, activation))
    
    # Middle layers: Stay in hidden dimension
    for _ in 2:depth-1
        push!(layers, ScanMixer(hidden_dim + context_dim, hidden_dim, hidden_dim, activation))
    end
    
    # Last layer: Projects from hidden dimension to output dimension
    # If depth is 1, the first layer is also the last layer, so we need to handle that.
    if depth == 1
        # Re-create the single layer to output out_dim
        empty!(layers)
        push!(layers, ScanMixer(in_dim + context_dim, hidden_dim, out_dim, activation))
    else
        push!(layers, ScanMixer(hidden_dim + context_dim, hidden_dim, out_dim, activation))
    end
    
    return SequenceTransformation(Chain(layers...))
end

Flux.@layer SequenceTransformation

function (m::SequenceTransformation)(x::AbstractArray, context::AbstractArray=Float32[])
    x_curr = x
    for layer in m.chain
        # Concatenate context if provided
        if !isempty(context)
            if ndims(x_curr) == 3
                # x: (F, T, B), context: (C, B)
                c_reshaped = reshape(context, size(context, 1), 1, size(context, 2))
                c_repeated = repeat(c_reshaped, 1, size(x_curr, 2), 1)
                x_in = cat(x_curr, c_repeated, dims=1)
            elseif ndims(x_curr) == 2
                # x: (F, T), context: (C,)
                c_reshaped = reshape(context, size(context, 1), 1)
                c_repeated = repeat(c_reshaped, 1, size(x_curr, 2))
                x_in = cat(x_curr, c_repeated, dims=1)
            else
                error("Unsupported input dimension")
            end
        else
            x_in = x_curr
        end
        
        x_curr = layer(x_in)
    end
    return x_curr
end

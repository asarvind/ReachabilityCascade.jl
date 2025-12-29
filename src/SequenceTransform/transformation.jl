using Flux

"""
    SequenceTransformation(in_dim::Int, hidden_dim::Int, out_dim::Int, depth::Int,
                           context_dim::Int=0, activation=Flux.gelu;
                           max_seq_len::Int=512, nheads::Int=1)

    A chain of attention + dense projection layers (`AttentionFFN` by default) with binary positional encodings
    cached up to `max_seq_len`, appended at every layer.

Arguments:
- `in_dim`: Dimension of the input features.
- `hidden_dim`: Dimension of the features within each internal block and the output of internal layers.
- `out_dim`: Dimension of the final output features.
- `depth`: Number of layers to stack.
- `context_dim`: Dimension of the context vector (optional).
    - `activation`: Activation function used in the internal projection blocks (default `Flux.gelu`).
    - `max_seq_len`: Maximum supported sequence length for cached positional encodings.
    - `nheads`: Number of attention heads for the internal attention blocks (default 2).

Returns a `Flux.Chain` of `AttentionFFN` layers.
"""
struct SequenceTransformation{C}
    chain::C
end

function SequenceTransformation(in_dim::Int, hidden_dim::Int, out_dim::Int, depth::Int,
                                context_dim::Int=0, activation=Flux.gelu;
                                max_seq_len::Int=512, nheads::Int=1)
    layers = []

    block_in = in_dim + context_dim
    # Attention path: append positional encodings at every layer.
    push!(layers, AttentionFFN(block_in, hidden_dim, depth == 1 ? out_dim : hidden_dim;
                                 activation=activation, max_seq_len=max_seq_len, nheads=nheads, add_pos=true))
    for _ in 2:depth-1
        push!(layers, AttentionFFN(hidden_dim + context_dim, hidden_dim, hidden_dim;
                                     activation=activation, max_seq_len=max_seq_len, nheads=nheads, add_pos=true))
    end
    if depth > 1
        push!(layers, AttentionFFN(hidden_dim + context_dim, hidden_dim, out_dim;
                                     activation=activation, max_seq_len=max_seq_len, nheads=nheads, add_pos=true))
    end

    return SequenceTransformation(Chain(layers...))
end

Flux.@layer SequenceTransformation

function _param_eltype(obj, default)
    for t in Flux.trainable(obj)
        if t isa AbstractArray
            return eltype(t)
        else
            inner = _param_eltype(t, nothing)
            inner !== nothing && return inner
        end
    end
    return default
end

function (m::SequenceTransformation)(x::AbstractArray, context::AbstractArray=Float32[])
    param_T = _param_eltype(m.chain, eltype(x))
    x_curr = param_T.(x)
    ctx = isempty(context) ? context : param_T.(context)

    for layer in m.chain
        # Concatenate context if provided
        if !isempty(ctx)
            if ndims(x_curr) == 3
                # x: (F, T, B), context: (C, B)
                c_reshaped = reshape(ctx, size(ctx, 1), 1, size(ctx, 2))
                c_repeated = repeat(c_reshaped, 1, size(x_curr, 2), 1)
                x_in = cat(x_curr, c_repeated, dims=1)
            elseif ndims(x_curr) == 2
                # x: (F, T), context: (C,)
                c_reshaped = reshape(ctx, size(ctx, 1), 1)
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

    return  x_curr
end

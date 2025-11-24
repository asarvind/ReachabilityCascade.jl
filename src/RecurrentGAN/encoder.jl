using Flux

"""
    Encoder

Iteratively refines a hidden state from a fixed input `[sample; context]`
to produce a latent code `z` in `[-1, 1]`. This is a fixed-point style
block (no time-varying sequence input).
"""
struct Encoder{B, O}
    block::B
    output_layer::O
    default_iterations::Int
end

Flux.@layer Encoder

# Helper utilities for validation
hidden_size(enc::Encoder) = size(enc.output_layer.weight, 2)
block_input_size(enc::Encoder) = size(enc.block.layer.weight, 2) - hidden_size(enc)

"""
    Encoder(input_dim::Int, context_dim::Int, hidden_dim::Int, latent_dim::Int;
            iterations::Int=5, activation::Function=leakyrelu)

Construct an encoder that iterates `iterations` times over a constant input.

Arguments:
- `input_dim`: dimension of the sample
- `context_dim`: dimension of the context
- `hidden_dim`: size of the recurrent hidden state
- `latent_dim`: size of latent code
- `iterations`: number of refinement steps (default 5)
- `activation`: activation for the iterative block (default `leakyrelu`)

Returns:
- `Encoder` ready for fixed-input refinement
"""
function Encoder(input_dim::Int, context_dim::Int, hidden_dim::Int, latent_dim::Int;
                 iterations::Int=5, activation=leakyrelu)
    
    enc_input_dim = input_dim + context_dim
    # Dense layer input size: hidden + enc_input
    enc_layer_in = hidden_dim + enc_input_dim
    
    block = IterativeBlock(Dense(enc_layer_in => hidden_dim, activation))
    # Map final hidden to z, soft-clamped to [-1, 1]
    output_layer = Dense(hidden_dim => latent_dim, tanh)
    
    return Encoder(block, output_layer, iterations)
end

"""
    encode(enc::Encoder, sample::AbstractVecOrMat, context::AbstractVecOrMat,
           initial_state::AbstractVecOrMat; iterations::Int=enc.default_iterations) -> (z::AbstractMatrix, h_final::AbstractMatrix)

Run fixed-point refinement:

Arguments:
- `enc`: encoder instance
- `sample`: shape `(input_dim,)` or `(input_dim, batch)`
- `context`: shape `(context_dim,)` or `(context_dim, batch)`
- `initial_state`: shape `(hidden_dim,)` or `(hidden_dim, batch)`
- `iterations`: optional refinement steps (default `enc.default_iterations`)

Returns:
- `z`: `(latent_dim, batch)` latent code (`tanh`-clamped)
- `h_final`: `(hidden_dim, batch)` final hidden state

Throws:
- `ArgumentError` if input/hidden shapes or `iterations` do not match the encoder configuration
"""
function encode(enc::Encoder, sample::AbstractVecOrMat, context::AbstractVecOrMat, initial_state::AbstractVecOrMat; iterations::Int=enc.default_iterations)
    s_in = ndims(sample) == 1 ? reshape(sample, :, 1) : sample
    c_in = ndims(context) == 1 ? reshape(context, :, 1) : context
    h = ndims(initial_state) == 1 ? reshape(initial_state, :, 1) : initial_state
    
    input = vcat(s_in, c_in)
    size(h, 1) == hidden_size(enc) || throw(ArgumentError("initial_state has size $(size(h)) but expected $(hidden_size(enc)) rows"))
    size(input, 1) == block_input_size(enc) || throw(ArgumentError("sample/context combined rows $(size(input, 1)) do not match expected $(block_input_size(enc))"))
    iterations > 0 || throw(ArgumentError("iterations must be positive, got $iterations"))
    
    for _ in 1:iterations
        h = enc.block(h, input)
    end
    
    # Pass through output layer
    z = enc.output_layer(h)
    return z, h
end

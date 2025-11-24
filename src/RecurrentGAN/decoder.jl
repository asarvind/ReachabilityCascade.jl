using Flux

"""
    Decoder

Iteratively refines a hidden state from a fixed input `[z; context]`
to reconstruct a sample. Fixed-point style (no time-varying sequence input).
"""
struct Decoder{B, O}
    block::B
    output_layer::O
    default_iterations::Int
end

Flux.@layer Decoder

# Helper utilities for validation
hidden_size(dec::Decoder) = size(dec.output_layer.weight, 2)
block_input_size(dec::Decoder) = size(dec.block.layer.weight, 2) - hidden_size(dec)

"""
    Decoder(latent_dim::Int, context_dim::Int, hidden_dim::Int, output_dim::Int;
            iterations::Int=5, activation::Function=leakyrelu)

Construct a decoder that iterates `iterations` times over a constant input.

Arguments:
- `latent_dim`: size of latent code
- `context_dim`: dimension of the context
- `hidden_dim`: size of the recurrent hidden state
- `output_dim`: dimension of the reconstructed sample
- `iterations`: number of refinement steps (default 5)
- `activation`: activation for the iterative block (default `leakyrelu`)

Returns:
- `Decoder` ready for fixed-input refinement
"""
function Decoder(latent_dim::Int, context_dim::Int, hidden_dim::Int, output_dim::Int;
                 iterations::Int=5, activation=leakyrelu)
    
    dec_input_dim = latent_dim + context_dim
    dec_layer_in = hidden_dim + dec_input_dim
    
    block = IterativeBlock(Dense(dec_layer_in => hidden_dim, activation))
    # Map final hidden to sample
    output_layer = Dense(hidden_dim => output_dim)
    
    return Decoder(block, output_layer, iterations)
end

"""
    decode(dec::Decoder, z::AbstractVecOrMat, context::AbstractVecOrMat,
           initial_state::AbstractVecOrMat; iterations::Int=dec.default_iterations) -> (sample::AbstractMatrix, h_final::AbstractMatrix)

Run fixed-point refinement:

Arguments:
- `dec`: decoder instance
- `z`: shape `(latent_dim,)` or `(latent_dim, batch)`
- `context`: shape `(context_dim,)` or `(context_dim, batch)`
- `initial_state`: shape `(hidden_dim,)` or `(hidden_dim, batch)`
- `iterations`: optional refinement steps (default `dec.default_iterations`)

Returns:
- `sample`: `(output_dim, batch)` reconstruction
- `h_final`: `(hidden_dim, batch)` final hidden state

Throws:
- `ArgumentError` if input/hidden shapes or `iterations` do not match the decoder configuration
"""
function decode(dec::Decoder, z::AbstractVecOrMat, context::AbstractVecOrMat, initial_state::AbstractVecOrMat; iterations::Int=dec.default_iterations)
    z_in = ndims(z) == 1 ? reshape(z, :, 1) : z
    c_in = ndims(context) == 1 ? reshape(context, :, 1) : context
    h = ndims(initial_state) == 1 ? reshape(initial_state, :, 1) : initial_state
    
    input = vcat(z_in, c_in)
    size(h, 1) == hidden_size(dec) || throw(ArgumentError("initial_state has size $(size(h)) but expected $(hidden_size(dec)) rows"))
    size(input, 1) == block_input_size(dec) || throw(ArgumentError("z/context combined rows $(size(input, 1)) do not match expected $(block_input_size(dec))"))
    iterations > 0 || throw(ArgumentError("iterations must be positive, got $iterations"))
    
    for _ in 1:iterations
        h = dec.block(h, input)
    end
    
    # Pass through output layer
    sample = dec.output_layer(h)
    return sample, h
end

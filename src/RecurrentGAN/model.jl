using Flux

"""
    RecurrentGAN

Adversarial autoencoder-style pair: an `Encoder` maps `[sample; context]`
to latent `z`, and a `Decoder` maps `[z; context]` back to the sample.
Both use fixed-point iterative refinement (constant input, recurrent state).
"""
struct RecurrentGAN{E, D}
    encoder::E
    decoder::D
end

Flux.@layer RecurrentGAN

"""
    RecurrentGAN(input_dim::Int, context_dim::Int, latent_dim::Int, hidden_dim::Int;
                 iterations::Int=5, activation::Function=leakyrelu)

Build paired encoder/decoder with matching hidden size and iteration count.
Suitable for latent-space adversarial training where the discriminator
operates on `z` and reconstruction is enforced via the decoder.

Arguments:
- `input_dim`: dimension of the sample
- `context_dim`: dimension of the context
- `latent_dim`: size of latent code
- `hidden_dim`: size of the recurrent hidden state
- `iterations`: number of refinement steps (default 5)
- `activation`: activation for the iterative blocks (default `leakyrelu`)

Returns:
- `RecurrentGAN` struct containing encoder/decoder pair
"""
function RecurrentGAN(input_dim::Int, context_dim::Int, latent_dim::Int, hidden_dim::Int;
                      iterations::Int=5, activation=leakyrelu)
    
    encoder = Encoder(input_dim, context_dim, hidden_dim, latent_dim; 
                      iterations=iterations, activation=activation)
    
    decoder = Decoder(latent_dim, context_dim, hidden_dim, input_dim; 
                      iterations=iterations, activation=activation)
    
    return RecurrentGAN(encoder, decoder)
end

# Convenience methods that delegate to the components; allow overriding iterations per call
encode(rgan::RecurrentGAN, sample, context, h0; iterations::Int=rgan.encoder.default_iterations) =
    encode(rgan.encoder, sample, context, h0; iterations=iterations)
decode(rgan::RecurrentGAN, z, context, h0; iterations::Int=rgan.decoder.default_iterations) =
    decode(rgan.decoder, z, context, h0; iterations=iterations)

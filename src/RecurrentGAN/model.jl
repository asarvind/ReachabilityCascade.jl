using Flux

"""
    IterativeBlock

A custom recurrent block that uses a simple Dense layer for updates.
h_{t+1} = activation(W * [h_t; x] + b)
"""
struct IterativeBlock{L}
    layer::L
end

Flux.@layer IterativeBlock

function (m::IterativeBlock)(h, x)
    # Input to layer is [h; x]
    combined = vcat(h, x)
    h_new = m.layer(combined)
    return h_new
end

"""
    RecurrentGAN

An Adversarial Autoencoder using iterative refinement.
Consists of an Encoder and a Decoder.
"""
struct RecurrentGAN{E, D}
    encoder::E
    decoder::D
    iterations::Int
end

Flux.@layer RecurrentGAN

"""
    RecurrentGAN(input_dim::Int, context_dim::Int, latent_dim::Int, hidden_dim::Int;
                 iterations::Int=5, activation=tanh)

Constructs a RecurrentGAN (Adversarial Autoencoder).
Encoder: [sample; context] -> z (in [-1, 1])
Decoder: [z; context] -> sample
"""
function RecurrentGAN(input_dim::Int, context_dim::Int, latent_dim::Int, hidden_dim::Int;
                      iterations::Int=5, activation=leakyrelu)
    
    # Encoder
    # Input: [sample; context]
    # Iterative update: h = f(h, [sample; context])
    enc_input_dim = input_dim + context_dim
    # Dense layer input size: hidden + enc_input
    enc_layer_in = hidden_dim + enc_input_dim
    
    enc_block = IterativeBlock(Dense(enc_layer_in => hidden_dim, activation))
    # Map final hidden to z, soft-clamped to [-1, 1]
    enc_out = Dense(hidden_dim => latent_dim, tanh)
    encoder = Chain(enc_block, enc_out)
    
    # Decoder
    # Input: [z; context]
    # Iterative update: h = f(h, [z; context])
    dec_input_dim = latent_dim + context_dim
    dec_layer_in = hidden_dim + dec_input_dim
    
    dec_block = IterativeBlock(Dense(dec_layer_in => hidden_dim, activation))
    # Map final hidden to sample
    dec_out = Dense(hidden_dim => input_dim)
    decoder = Chain(dec_block, dec_out)
    
    return RecurrentGAN(encoder, decoder, iterations)
end

"""
    encode(rgan::RecurrentGAN, sample::AbstractVecOrMat, context::AbstractVecOrMat, initial_state::AbstractVecOrMat)

Encodes a sample and context into a latent vector z, starting from an initial hidden state.
"""
function encode(rgan::RecurrentGAN, sample::AbstractVecOrMat, context::AbstractVecOrMat, initial_state::AbstractVecOrMat)
    s_in = ndims(sample) == 1 ? reshape(sample, :, 1) : sample
    c_in = ndims(context) == 1 ? reshape(context, :, 1) : context
    h = ndims(initial_state) == 1 ? reshape(initial_state, :, 1) : initial_state
    
    input = vcat(s_in, c_in)
    
    # Encoder chain: [IterativeBlock, Dense]
    # We need to manually loop the IterativeBlock
    enc_block = rgan.encoder[1]
    enc_out_layer = rgan.encoder[2]
    
    for _ in 1:rgan.iterations
        h = enc_block(h, input)
    end
    
    # Pass through output layer
    z = enc_out_layer(h)
    return z, h
end

"""
    decode(rgan::RecurrentGAN, z::AbstractVecOrMat, context::AbstractVecOrMat, initial_state::AbstractVecOrMat)

Decodes a latent vector z and context into a sample, starting from an initial hidden state.
"""
function decode(rgan::RecurrentGAN, z::AbstractVecOrMat, context::AbstractVecOrMat, initial_state::AbstractVecOrMat)
    z_in = ndims(z) == 1 ? reshape(z, :, 1) : z
    c_in = ndims(context) == 1 ? reshape(context, :, 1) : context
    h = ndims(initial_state) == 1 ? reshape(initial_state, :, 1) : initial_state
    
    input = vcat(z_in, c_in)
    
    # Decoder chain: [IterativeBlock, Dense]
    dec_block = rgan.decoder[1]
    dec_out_layer = rgan.decoder[2]
    
    for _ in 1:rgan.iterations
        h = dec_block(h, input)
    end
    
    # Pass through output layer
    sample = dec_out_layer(h)
    return sample, h
end

using Flux
using Zygote

"""
    latent_consistency_loss(gan::GAN, x::AbstractVecOrMat, c::AbstractVecOrMat)

Computes the latent consistency loss: ||E(D(E(x))) - E(x)||.
"""
function latent_consistency_loss(gan::GAN, x::AbstractVecOrMat, c::AbstractVecOrMat)
    z1 = encode(gan, x, c)
    x_rec = decode(gan, z1, c)
    z2 = encode(gan, x_rec, c)
    return Flux.mse(z2, z1)
end

"""
    compute_gradients(gan::GAN, x_true::AbstractVecOrMat, c_true::AbstractVecOrMat, 
                      z_gen::AbstractVecOrMat, c_gen::AbstractVecOrMat)

Computes gradients for the Encoder and Decoder.

# Arguments
- `gan`: The GAN model.
- `x_true`: Batch of true samples.
- `c_true`: Context for true samples.
- `z_gen`: Batch of latent vectors for generation.
- `c_gen`: Context for generation.

# Returns
A named tuple `(encoder = grads_encoder, decoder = grads_decoder)`.
"""
function compute_gradients(gan::GAN, x_true::AbstractVecOrMat, c_true::AbstractVecOrMat, 
                           z_gen::AbstractVecOrMat, c_gen::AbstractVecOrMat)
    
    # Encoder Gradient
    # Minimize loss on true samples, Maximize loss on generated samples
    grads_encoder = Flux.gradient(gan.encoder) do encoder_model
        # We need to reconstruct the GAN with the traced encoder model to use helper functions
        # or just manually call the parts. To be safe and clean, let's manually call.
        
        # True samples: Minimize L_consist(x_true)
        # L = ||E(D(E(x))) - E(x)||
        # Note: D is fixed here.
        z1_true = encoder_model(x_true, c_true)
        x_rec_true = gan.decoder(z1_true, c_true) # Decoder is fixed
        z2_true = encoder_model(x_rec_true, c_true)
        loss_true = Flux.mse(z2_true, z1_true)
        
        # Generated samples: Maximize L_consist(x_gen)
        # x_gen = D(z_gen). This is fixed input for encoder optimization.
        x_gen = gan.decoder(z_gen, c_gen) # Fixed
        # We treat x_gen as constant input
        x_gen_fixed = Zygote.dropgrad(x_gen)
        
        z1_gen = encoder_model(x_gen_fixed, c_gen)
        x_rec_gen = gan.decoder(z1_gen, c_gen) # Fixed
        z2_gen = encoder_model(x_rec_gen, c_gen)
        loss_gen = Flux.mse(z2_gen, z1_gen)
        
        return loss_true - loss_gen
    end

    # Decoder Gradient
    # Minimize loss on true samples AND generated samples
    grads_decoder = Flux.gradient(gan.decoder) do decoder_model
        # True samples
        z1_true = gan.encoder(x_true, c_true) # Fixed
        x_rec_true = decoder_model(z1_true, c_true)
        z2_true = gan.encoder(x_rec_true, c_true) # Fixed
        loss_true = Flux.mse(z2_true, z1_true)
        
        # Generated samples
        x_gen = decoder_model(z_gen, c_gen)
        z1_gen = gan.encoder(x_gen, c_gen) # Fixed
        x_rec_gen = decoder_model(z1_gen, c_gen)
        z2_gen = gan.encoder(x_rec_gen, c_gen) # Fixed
        loss_gen = Flux.mse(z2_gen, z1_gen)
        
        return loss_true + loss_gen
    end

    return (encoder = grads_encoder, decoder = grads_decoder)
end

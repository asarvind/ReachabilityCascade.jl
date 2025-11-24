using Flux

# Internal helper to create a zero hidden state matching batch size and element type
_zeros_hidden(enc::Encoder, x::AbstractVecOrMat) = begin
    batch = ndims(x) == 1 ? 1 : size(x, 2)
    zeros(eltype(x), hidden_size(enc), batch)
end
_zeros_hidden(dec::Decoder, x::AbstractVecOrMat) = begin
    batch = ndims(x) == 1 ? 1 : size(x, 2)
    zeros(eltype(x), hidden_size(dec), batch)
end

"""
    encoder_reconstruction_loss(rgan::RecurrentGAN, samples::AbstractVecOrMat, context::AbstractVecOrMat;
                                h_enc0::Union{Nothing,AbstractVecOrMat}=nothing,
                                h_dec0::Union{Nothing,AbstractVecOrMat}=nothing,
                                iterations_enc::Union{Nothing,Int}=nothing,
                                iterations_dec::Union{Nothing,Int}=nothing) -> Real

Encode → decode → encode fixed-point reconstruction loss in latent space.

Arguments:
- `rgan`: recurrent GAN (encoder/decoder pair)
- `samples`: input samples `(input_dim,)` or `(input_dim, batch)`
- `context`: context `(context_dim,)` or `(context_dim, batch)`
- `h_enc0`: optional initial encoder hidden state (defaults to zeros)
- `h_dec0`: optional initial decoder hidden state (defaults to zeros)
- `iterations_enc`: optional encoder iterations override
- `iterations_dec`: optional decoder iterations override

Returns:
- `Real` value `mse(z1, z0)` where `z0 = encode(samples, context)`, `x̂ = decode(z0, context)`, `z1 = encode(x̂, context)`
"""
function encoder_reconstruction_loss(rgan::RecurrentGAN, samples::AbstractVecOrMat, context::AbstractVecOrMat;
                                     h_enc0::Union{Nothing,AbstractVecOrMat}=nothing,
                                     h_dec0::Union{Nothing,AbstractVecOrMat}=nothing,
                                     iterations_enc::Union{Nothing,Int}=nothing,
                                     iterations_dec::Union{Nothing,Int}=nothing)
    ienc = isnothing(iterations_enc) ? rgan.encoder.default_iterations : iterations_enc
    idec = isnothing(iterations_dec) ? rgan.decoder.default_iterations : iterations_dec

    h_enc = isnothing(h_enc0) ? _zeros_hidden(rgan.encoder, samples) : h_enc0
    z0, _ = encode(rgan, samples, context, h_enc; iterations=ienc)

    h_dec = isnothing(h_dec0) ? _zeros_hidden(rgan.decoder, z0) : h_dec0
    recon, _ = decode(rgan, z0, context, h_dec; iterations=idec)

    h_enc2 = isnothing(h_enc0) ? _zeros_hidden(rgan.encoder, recon) : h_enc0
    z1, _ = encode(rgan, recon, context, h_enc2; iterations=ienc)

    return Flux.Losses.mse(z1, z0)
end

"""
    decoder_reconstruction_loss(rgan::RecurrentGAN, latents::AbstractVecOrMat, context::AbstractVecOrMat;
                                h_dec0::Union{Nothing,AbstractVecOrMat}=nothing,
                                h_enc0::Union{Nothing,AbstractVecOrMat}=nothing,
                                iterations_dec::Union{Nothing,Int}=nothing,
                                iterations_enc::Union{Nothing,Int}=nothing) -> Real

Decode → encode reconstruction loss in latent space.

Arguments:
- `rgan`: recurrent GAN (encoder/decoder pair)
- `latents`: latent codes `(latent_dim,)` or `(latent_dim, batch)`
- `context`: context `(context_dim,)` or `(context_dim, batch)`
- `h_dec0`: optional initial decoder hidden state (defaults to zeros)
- `h_enc0`: optional initial encoder hidden state (defaults to zeros)
- `iterations_dec`: optional decoder iterations override
- `iterations_enc`: optional encoder iterations override

Returns:
- `Real` value `mse(z1, latents)` where `x̂ = decode(latents, context)`, `z1 = encode(x̂, context)`
"""
function decoder_reconstruction_loss(rgan::RecurrentGAN, latents::AbstractVecOrMat, context::AbstractVecOrMat;
                                     h_dec0::Union{Nothing,AbstractVecOrMat}=nothing,
                                     h_enc0::Union{Nothing,AbstractVecOrMat}=nothing,
                                     iterations_dec::Union{Nothing,Int}=nothing,
                                     iterations_enc::Union{Nothing,Int}=nothing)
    idec = isnothing(iterations_dec) ? rgan.decoder.default_iterations : iterations_dec
    ienc = isnothing(iterations_enc) ? rgan.encoder.default_iterations : iterations_enc

    h_dec = isnothing(h_dec0) ? _zeros_hidden(rgan.decoder, latents) : h_dec0
    recon, _ = decode(rgan, latents, context, h_dec; iterations=idec)

    h_enc = isnothing(h_enc0) ? _zeros_hidden(rgan.encoder, recon) : h_enc0
    z1, _ = encode(rgan, recon, context, h_enc; iterations=ienc)

    return Flux.Losses.mse(z1, latents)
end

"""
    GAN

A Generative Adversarial Network-like structure containing an Encoder and a Decoder.

# Fields
- `encoder::Encoder`: The encoder network.
- `decoder::Decoder`: The decoder network.
- `x_dim::Int`: Dimension of the data.
- `ctx_dim::Int`: Dimension of the context.
- `z_dim::Int`: Dimension of the latent space.
"""
struct GAN{E, D}
    encoder::E
    decoder::D
    x_dim::Int
    ctx_dim::Int
    z_dim::Int
end

Flux.@layer GAN

"""
    GAN(x_dim::Integer, ctx_dim::Integer, z_dim::Integer;
        hidden::Integer=128, n_layers::Integer=2, activation=leakyrelu)

Constructs a GAN.

# Arguments
- `x_dim`: Dimension of the data.
- `ctx_dim`: Dimension of the context.
- `z_dim`: Dimension of the latent space.
- `hidden`: Number of hidden units in encoder/decoder.
- `n_layers`: Number of hidden layers in encoder/decoder.
- `activation`: Activation function for hidden layers. Defaults to `leakyrelu`.
"""
function GAN(x_dim::Integer, ctx_dim::Integer, z_dim::Integer;
             hidden::Integer=128, n_layers::Integer=2, activation=leakyrelu)
    encoder = Encoder(x_dim, ctx_dim, z_dim; hidden=hidden, n_layers=n_layers, activation=activation)
    decoder = Decoder(z_dim, ctx_dim, x_dim; hidden=hidden, n_layers=n_layers, activation=activation)
    return GAN(encoder, decoder, x_dim, ctx_dim, z_dim)
end

"""
    encode(gan::GAN, x::AbstractVecOrMat, c::AbstractVecOrMat)

Encodes samples `x` given context `c` into latent vectors `z`.
"""
function encode(gan::GAN, x::AbstractVecOrMat, c::AbstractVecOrMat)
    return gan.encoder(x, c)
end

"""
    decode(gan::GAN, z::AbstractVecOrMat, c::AbstractVecOrMat)

Decodes latent vectors `z` given context `c` into samples `x`.
"""
function decode(gan::GAN, z::AbstractVecOrMat, c::AbstractVecOrMat)
    return gan.decoder(z, c)
end

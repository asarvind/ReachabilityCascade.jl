using Flux
using Random
using ReachabilityCascade: glu_mlp

"""
    Gan(latent_dim, data_dim; kwargs...)

Convenience container that wires a generator and discriminator built from
`glu_mlp` blocks, ready for GAN-style training loops.
"""
struct Gan{G,D}
    generator::G
    discriminator::D
end

Flux.@layer Gan

function Gan(latent_dim::Integer, data_dim::Integer;
             gen_hidden::Integer=128,
             disc_hidden::Integer=128,
             n_glu_gen::Integer=2,
             n_glu_disc::Integer=2,
             gen_gate=Flux.σ,
             disc_gate=Flux.σ,
             generator_out::Union{Function,Nothing}=Flux.tanh,
             discriminator_out::Union{Function,Nothing}=Flux.σ,
             generator_zero_init::Bool=true,
             discriminator_zero_init::Bool=false)
    gen_core = glu_mlp(latent_dim, gen_hidden, data_dim;
                       n_glu=n_glu_gen, act=gen_gate, zero_init=generator_zero_init)
    generator = generator_out === nothing ? gen_core : Chain(gen_core, generator_out)

    disc_core = glu_mlp(data_dim, disc_hidden, 1;
                        n_glu=n_glu_disc, act=disc_gate, zero_init=discriminator_zero_init)
    discriminator = discriminator_out === nothing ? disc_core : Chain(disc_core, discriminator_out)

    return Gan(generator, discriminator)
end

(gan::Gan)(z) = gan.generator(Flux.f32(z))

"""
    generator_forward(gan::Gan, latent)

Evaluate the generator on a latent sample. Accepts vectors or `(features, batch)` matrices.
"""
generator_forward(gan::Gan, latent) = gan.generator(Flux.f32(latent))

"""
    discriminator_forward(gan::Gan, samples)

Evaluate the discriminator on real or generated samples. Supports batched input.
"""
discriminator_forward(gan::Gan, samples) = gan.discriminator(Flux.f32(samples))

let latent_dim = 4, data_dim = 3, batch = 5
    gan = Gan(latent_dim, data_dim; gen_hidden=16, disc_hidden=16,
              generator_out=nothing, discriminator_out=nothing)

    z = randn(Float32, latent_dim, batch)
    x_fake = generator_forward(gan, z)
    @assert size(x_fake) == (data_dim, batch)

    disc_scores = discriminator_forward(gan, x_fake)
    @assert size(disc_scores, 2) == batch

    println("GAN forward example ran successfully with batch size ", batch)
end

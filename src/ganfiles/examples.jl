module GanExamples

using Random
import ..Gan
import ..generator_forward
import ..discriminator_forward
import ..encoder_forward
import ..gan_gradients
import ..gradient_norm

"""
    basic_forward(; seed=11, latent_dim=4, context_dim=3, data_dim=3, batch=5)

Constructs a `Gan`, runs generator, discriminator, and encoder passes on random
batches, and returns a bundle of intermediate tensors and shape checks.
"""
function basic_forward(; seed::Integer=11,
                       latent_dim::Integer=4,
                       context_dim::Integer=3,
                       data_dim::Integer=3,
                       batch::Integer=5)
    rng = Random.MersenneTwister(seed)
    gan = Gan(latent_dim, context_dim, data_dim; gen_hidden=16, disc_hidden=16,
              generator_out=nothing)

    context = randn(rng, Float32, context_dim, batch)
    z = randn(rng, Float32, latent_dim, batch)
    samples = generator_forward(gan, context, z)
    disc_scores = discriminator_forward(gan, context, samples, z)
    z_rec = encoder_forward(gan, context, samples)

    return (
        gan = gan,
        context = context,
        latent = z,
        generated = samples,
        discriminator_scores = disc_scores,
        reconstructed_latent = z_rec
    )
end

"""
    gradient_check(; seed=21, batch_new=8, batch_old=4)

Runs `gan_gradients` with random fresh and replay batches and returns gradient
norms along with the selected hard examples.
"""
function gradient_check(; seed::Integer=21,
                        latent_dim::Integer=4,
                        context_dim::Integer=3,
                        data_dim::Integer=3,
                        batch_new::Integer=8,
                        batch_old::Integer=4,
                        sorted_limit::Integer=5)
    rng = Random.MersenneTwister(seed)
    gan = Gan(latent_dim, context_dim, data_dim; gen_hidden=16, disc_hidden=16,
              generator_out=nothing)

    fresh_batch = (
        contexts = randn(rng, Float32, context_dim, batch_new),
        samples = randn(rng, Float32, data_dim, batch_new),
    )

    old_batch = (
        contexts = randn(rng, Float32, context_dim, batch_old),
        samples = randn(rng, Float32, data_dim, batch_old),
    )

    result = gan_gradients(gan, fresh_batch; old_batch=old_batch, sorted_limit=sorted_limit)

    return (
        gan = gan,
        fresh_batch = fresh_batch,
        old_batch = old_batch,
        gradients = result,
        encoder_grad_norm = gradient_norm(result.encoder_grad),
        generator_grad_norm = gradient_norm(result.generator_grad),
        discriminator_grad_norm = gradient_norm(result.discriminator_grad)
    )
end

end # module GanExamples

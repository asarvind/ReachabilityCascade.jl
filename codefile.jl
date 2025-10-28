using Flux
using Random
using ReachabilityCascade: Gan, generator_forward, discriminator_forward, encoder_forward,
                           HierarchicalBehaviorCloner, intermediate_level_forward,
                           gan_gradients, gradient_norm

let latent_dim = 4, context_dim = 3, data_dim = 3, batch = 5
    gan = Gan(latent_dim, context_dim, data_dim; gen_hidden=16, disc_hidden=16,
              generator_out=nothing)

    context = randn(Float32, context_dim, batch)
    z = randn(Float32, latent_dim, batch)
    x_fake = generator_forward(gan, context, z)
    @assert size(x_fake) == (data_dim, batch)

    disc_scores = discriminator_forward(gan, context, x_fake, z)
    @assert size(disc_scores, 2) == batch
    z_rec = encoder_forward(gan, context, x_fake)
    @assert size(z_rec) == (latent_dim, batch)
    @assert all(abs.(z_rec) .<= 1f0 + 1f-4)

    println("Conditional GAN example ran successfully with batch size ", batch)
end

let state_dim = 6, goal_dim = 3, latent_dim = 4, control_dim = 2, batch = 2
    model = HierarchicalBehaviorCloner(state_dim, goal_dim, latent_dim, control_dim;
                                       n_intermediate=3,
                                       task_hidden=32,
                                       intermediate_hidden=32,
                                       control_hidden=32)

    current_state = randn(Float32, state_dim, batch)
    goal = randn(Float32, goal_dim, batch)
    latent = randn(Float32, latent_dim, batch)

    result = model(current_state, goal, latent)
    single_level = intermediate_level_forward(model,
                                              length(model.intermediate_gans),
                                              current_state,
                                              result.terminal_state,
                                              latent)

    @assert size(result.control) == (control_dim, batch)
    @assert length(result.intermediate_states) == 3
    @assert all(size(s) == (state_dim, batch) for s in result.intermediate_states)
    @assert size(single_level) == (state_dim, batch)

    println("Hierarchical behavior cloner produced control with shape ", size(result.control))
end

let latent_dim = 4, context_dim = 3, data_dim = 3
    gan = Gan(latent_dim, context_dim, data_dim; gen_hidden=16, disc_hidden=16,
              generator_out=nothing)

    fresh_batch = (
        contexts = randn(Float32, context_dim, 8),
        samples = randn(Float32, data_dim, 8),
    )

    old_batch = (
        contexts = randn(Float32, context_dim, 4),
        samples = randn(Float32, data_dim, 4),
    )

    result = gan_gradients(gan, fresh_batch; old_batch=old_batch, sorted_limit=5)

    encoder_norm = gradient_norm(result.encoder_grad)
    generator_norm = gradient_norm(result.generator_grad)
    discriminator_norm = gradient_norm(result.discriminator_grad)

    println("Encoder grad norm: ", encoder_norm)
    println("Generator grad norm: ", generator_norm)
    println("Discriminator grad norm: ", discriminator_norm)

    @assert generator_norm > 0 && !isnan(generator_norm)
    @assert discriminator_norm > 0 && !isnan(discriminator_norm)
    hard_examples = result.hard_examples
    @assert size(hard_examples.contexts, 2) == size(hard_examples.samples, 2) == length(hard_examples.scores)

    println("Gradient computation succeeded with ", length(hard_examples.scores), " hard examples selected.")
end

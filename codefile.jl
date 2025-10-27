using Flux
using Random
using ReachabilityCascade: Gan, generator_forward, discriminator_forward, encoder_forward
using ReachabilityCascade: HierarchicalBehaviorCloner, intermediate_level_forward

let latent_dim = 4, context_dim = 3, data_dim = 3, batch = 5
    gan = Gan(latent_dim, context_dim, data_dim; gen_hidden=16, disc_hidden=16,
              generator_out=nothing, discriminator_out=nothing)

    context = randn(Float32, context_dim, batch)
    z = randn(Float32, latent_dim, batch)
    x_fake = generator_forward(gan, context, z)
    @assert size(x_fake) == (data_dim, batch)

    disc_scores = discriminator_forward(gan, context, x_fake)
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

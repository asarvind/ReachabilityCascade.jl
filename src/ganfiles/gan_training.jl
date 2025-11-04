function _normalize_batch(batch::NamedTuple)
    @assert haskey(batch, :contexts) && haskey(batch, :samples) "Batch must contain :contexts and :samples"
    contexts = _as_matrix(batch.contexts)
    samples = _as_matrix(batch.samples)
    @assert size(contexts, 2) == size(samples, 2) "Context/sample batch size mismatch"
    return contexts, samples
end

_gradient_norm(x::AbstractArray) = sum(abs, x)
_gradient_norm(x::NamedTuple) = sum(_gradient_norm(v) for v in values(x))
_gradient_norm(x::Tuple) = sum(_gradient_norm(v) for v in x)
_gradient_norm(x::Nothing) = 0.0f0
_gradient_norm(x::Number) = abs(Float32(x))
function _gradient_norm(x)
    total = 0.0f0
    for name in fieldnames(typeof(x))
        isdefined(x, name) || continue
        total += _gradient_norm(getfield(x, name))
    end
    return total
end

"""
    gradient_norm(grad)

Sum of absolute values across all array leaves in a gradient container.
"""
function gradient_norm(grad)
    return Float32(_gradient_norm(grad))
end

"""
    gan_gradients(gan, fresh_batch; old_batch=nothing, sorted_limit=1, loss_fn=binarycrossentropy)

Compute generator, encoder, and discriminator gradients for a conditional GAN.
Returns gradients alongside the lowest-scoring context/sample pairs according to the discriminator.
"""
function gan_gradients(gan::Gan,
                       fresh_batch::NamedTuple;
                       old_batch::Union{NamedTuple,Nothing}=nothing,
                       sorted_limit::Union{Int,Nothing}=1,
                       loss_fn::Function=Flux.Losses.binarycrossentropy,
                       reconstruction_weights::Union{AbstractVector,Nothing}=nothing,
                       latent_sampler::Function = dim -> (2f0 .* rand(Float32, dim) .- 1f0))
    fresh_contexts, fresh_samples = _normalize_batch(fresh_batch)
    @assert sorted_limit === nothing || sorted_limit ≥ 0 "sorted_limit must be non-negative"

    if old_batch === nothing
        full_contexts, full_samples = fresh_contexts, fresh_samples
    else
        old_contexts, old_samples = _normalize_batch(old_batch)
        full_contexts = hcat(fresh_contexts, old_contexts)
        full_samples = hcat(fresh_samples, old_samples)
    end

    total_count = size(full_contexts, 2)
    @assert total_count > 0 "Combined batch must contain at least one sample"

    dims = (gan.latent_dim, gan.context_dim, gan.data_dim)

    fake_latents = reduce(hcat, (Float32.(latent_sampler(gan.latent_dim)) for _ in 1:total_count))
    fake_samples_detached = Flux.f32(generator_forward(gan, full_contexts, fake_latents))
    encoded_latents = encoder_forward(gan, full_contexts, full_samples)
    fake_samples_from_encoded = Flux.f32(generator_forward(gan, full_contexts, encoded_latents))

    disc_grads = Flux.gradient(gan.discriminator) do disc
        temp_gan = Gan(gan.generator, disc, gan.encoder, dims...)
        real_scores = discriminator_forward(temp_gan, full_contexts, full_samples, encoded_latents)
        fake_scores_rand = discriminator_forward(temp_gan, full_contexts, fake_samples_detached, fake_latents)
        fake_scores_enc = discriminator_forward(temp_gan, full_contexts, fake_samples_from_encoded, encoded_latents)
        real_targets = ones(Float32, size(real_scores))
        fake_targets_rand = zeros(Float32, size(fake_scores_rand))
        fake_targets_enc = zeros(Float32, size(fake_scores_enc))
        return loss_fn(real_scores, real_targets) +
               loss_fn(fake_scores_rand, fake_targets_rand) +
               loss_fn(fake_scores_enc, fake_targets_enc)
    end
    discriminator_grad = disc_grads[1]

    rand_count = rand(1:total_count)
    rand_idx = rand(1:total_count, rand_count)
    rand_contexts = full_contexts[:, rand_idx]
    rand_latents = reduce(hcat, (Float32.(latent_sampler(gan.latent_dim)) for _ in 1:rand_count))

    gen_contexts = hcat(rand_contexts, full_contexts)
    gen_latents = hcat(rand_latents, encoded_latents)

    gen_grads = Flux.gradient(gan.generator) do gen
        temp_gan = Gan(gen, gan.discriminator, gan.encoder, dims...)
        fake_samples = generator_forward(temp_gan, gen_contexts, gen_latents)
        fake_scores = discriminator_forward(temp_gan, gen_contexts, fake_samples, gen_latents)
        targets = ones(Float32, size(fake_scores))
        return loss_fn(fake_scores, targets)
    end
    generator_grad = gen_grads[1]

    recon_latents = reduce(hcat, (Float32.(latent_sampler(gan.latent_dim)) for _ in 1:total_count))
    enc_grads = Flux.gradient(gan.encoder) do enc
        temp_gan = Gan(gan.generator, gan.discriminator, enc, dims...)
        samples = generator_forward(temp_gan, full_contexts, recon_latents)
        reconstructed = encoder_forward(temp_gan, full_contexts, samples)
        return Flux.Losses.mse(reconstructed, recon_latents)
    end
    encoder_grad = enc_grads[1]

    recon_samples = generator_forward(gan, full_contexts, encoded_latents)
    diffs = recon_samples .- full_samples
    if reconstruction_weights !== nothing
        weights = Flux.f32(reconstruction_weights)
        @assert size(diffs, 1) == length(weights) "Weight length must match sample dimension"
        diffs = diffs .* weights
    end
    recon_errors = vec(sum(abs2, diffs; dims=1))
    order = sortperm(recon_errors)

    limit = sorted_limit === nothing ? length(order) : min(sorted_limit, length(order))
    if limit == 0
        hard_contexts = full_contexts[:, Int[]]
        hard_samples = full_samples[:, Int[]]
        hard_scores = Float32[]
    else
        selected_idx = order[1:limit]
        hard_contexts = full_contexts[:, selected_idx]
        hard_samples = full_samples[:, selected_idx]
        hard_scores = Flux.f32(recon_errors[selected_idx])
    end

    return (encoder_grad=encoder_grad,
            discriminator_grad=discriminator_grad,
            generator_grad=generator_grad,
            hard_examples=(contexts=hard_contexts, samples=hard_samples, scores=hard_scores))
end

using Test
using ReachabilityCascade: ConditionalFlow, RecurrentConditionalFlow, recurrent_flow_gradients
using ReachabilityCascade.NormalizingFlow: encode_recurrent, decode_recurrent, encode_recurrent_transitions

@testset "Recurrent Conditional Flow" begin
    x_dim = 3
    base_ctx_dim = 2
    time_embed_dim = 2
    steps = 4

    flow = ConditionalFlow(x_dim, base_ctx_dim + time_embed_dim; n_blocks=2, hidden=16, n_glu=2)
    rcf = RecurrentConditionalFlow(flow, base_ctx_dim)

    # single sample vector input
    x_vec = randn(Float32, x_dim)
    ctx_vec = randn(Float32, base_ctx_dim)
    result_vec = encode_recurrent(rcf, x_vec, ctx_vec, steps)
    @test length(result_vec.per_step_latents) == steps
    @test length(result_vec.per_step_logdets) == steps
    @test size(result_vec.latent, 2) == 1
    xr_vec = decode_recurrent(rcf, result_vec.latent, ctx_vec, steps)
    @test size(xr_vec, 2) == 1
    @test maximum(abs.(xr_vec[:, 1] .- x_vec)) < 1f-4

    # batched input matrix
    batch = 3
    x_mat = randn(Float32, x_dim, batch)
    ctx_mat = randn(Float32, base_ctx_dim, batch)
    result = encode_recurrent(rcf, x_mat, ctx_mat, steps)
    @test length(result.per_step_latents) == steps
    @test all(size(lat, 2) == batch for lat in result.per_step_latents)
    transitions = result.transitions
    @test length(transitions) == steps
    @test all(trans.step == i for (i, trans) in enumerate(transitions))
    @test all(size(trans.input, 2) == batch for trans in transitions)
    @test all(size(trans.context, 2) == batch for trans in transitions)

    xr = decode_recurrent(rcf, result.latent, ctx_mat, steps)
    @test maximum(abs.(xr .- x_mat)) < 1f-4

    # transitions helper directly
    transitions_direct = encode_recurrent_transitions(rcf, x_mat, ctx_mat, steps)
    @test transitions_direct == transitions

    old_mat = randn(Float32, x_dim, 2)
    old_ctx = randn(Float32, base_ctx_dim, 2)
    all_samples_combined = hcat(x_mat, old_mat)
    all_context_combined = hcat(ctx_mat, old_ctx)
    transitions_combined = encode_recurrent_transitions(rcf, all_samples_combined,
                                                        all_context_combined, steps)

    grad_result = recurrent_flow_gradients(rcf, x_mat, ctx_mat, steps;
                                           old_samples=old_mat, old_context=old_ctx,
                                           num_lowest=2)
    @test grad_result.loss isa Real
    @test grad_result.transitions == transitions_combined

    @test !isempty(grad_result.grads)
    hard = grad_result.hard_examples
    @test hard !== nothing
    @test size(hard.samples, 2) == 2
    @test length(hard.step) == 2
    @test all(1 <= s <= steps for s in hard.step)
    @test all(diff(hard.loglikelihood) .>= 0)
end

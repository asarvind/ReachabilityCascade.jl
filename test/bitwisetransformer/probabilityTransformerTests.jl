using Test
using ReachabilityCascade

@testset "Probability Transformer" begin
    context_dim = 4
    sequence_length = 7
    prior_dim = 4
    batch_size = 3

    net = ProbabilityTransformer(context_dim;
                                 prior_dim=prior_dim,
                                 embed_dim=16,
                                 heads=2,
                                 ff_hidden=32,
                                 bit_position=3)

    context = rand(Float32, context_dim, batch_size)
    prior = rand(Float32, sequence_length, prior_dim, batch_size)

    result = net(context, prior)
    values, probs, logits = result.values, result.probabilities, result.logits

    @test size(probs) == size(prior)
    @test size(values) == size(prior)
    @test all(0 .<= probs .<= 1)
    @test size(logits) == size(prior)

    scale = Float32(2.0f0^(net.bit_position - 1))
    @test all(values .== Float32.(probs .>= 0.5f0) .* scale .+ prior)

    single_prior = rand(Float32, sequence_length, prior_dim)
    single_context = rand(Float32, context_dim)
    s_result = net(single_context, single_prior)
    s_values, s_probs, s_logits = s_result.values, s_result.probabilities, s_result.logits
    @test size(s_probs) == size(single_prior)
    @test size(s_values) == size(single_prior)
    @test all(0 .<= s_probs .<= 1)
    @test all(s_values .== Float32.(s_probs .>= 0.5f0) .* scale .+ single_prior)
    @test size(s_logits) == size(single_prior)

    scalar_net = ProbabilityTransformer(context_dim;
                                        prior_dim=1,
                                        embed_dim=8,
                                        heads=2,
                                        ff_hidden=16,
                                        bit_position=2)

    vector_prior = rand(Float32, sequence_length)
    v_result = scalar_net(single_context, vector_prior)
    v_values, v_probs, v_logits = v_result.values, v_result.probabilities, v_result.logits
    @test size(v_probs) == (sequence_length,)
    @test size(v_values) == (sequence_length,)
    @test all(0 .<= v_probs .<= 1)
    scalar_scale = Float32(2.0f0^(scalar_net.bit_position - 1))
    @test all(v_values .== Float32.(v_probs .>= 0.5f0) .* scalar_scale .+ vector_prior)
    @test size(v_logits) == (sequence_length,)
end

using Test
using ReachabilityCascade

@testset "Probability Transformer Gradients" begin
    context_dim = 3
    sequence_length = 5
    prior_dim = 2
    batch_size = 4

    net = ProbabilityTransformer(context_dim;
                                 prior_dim=prior_dim,
                                 embed_dim=12,
                                 heads=2,
                                 ff_hidden=20,
                                 bit_position=2)

    context = rand(Float32, context_dim, batch_size)
    prior = rand(Float32, sequence_length, prior_dim, batch_size)
    targets = Float32.(rand(Float32, sequence_length, prior_dim, batch_size) .> 0.5f0)

    grad_result = ReachabilityCascade.gradients(net, context, prior, targets)

    @test isa(grad_result.loss, Number)
    @test grad_result.loss >= 0
    @test grad_result.gradient isa NamedTuple
    @test grad_result.gradient.context_proj.weight isa AbstractMatrix
    @test grad_result.gradient.token_proj.bias isa AbstractVector
    @test size(grad_result.values) == size(prior)
    @test size(grad_result.probabilities) == size(prior)
    @test size(grad_result.logits) == size(prior)
end

@testset "Hierarchical Bitwise Transformer Gradients" begin
    context_dim = 3
    prior_dim = 2
    sequence_length = 4
    batch_size = 2
    bits = [3, 2, 1]
    mean = Float32[0.5, -1.25]
    radius = Float32[1.5, 0.75]

    net = HierarchicalBitwiseTransformer(context_dim, prior_dim;
                                         bit_positions=bits,
                                         mean=mean,
                                         radius=radius,
                                         embed_dim=12,
                                         heads=2,
                                         ff_hidden=20)

    context = rand(Float32, context_dim, batch_size)
    true_bits = [Float32.(rand(0:1, sequence_length, prior_dim, batch_size)) for _ in bits]

    target_norm = zeros(Float32, sequence_length, prior_dim, batch_size)
    for (stage, bit_tensor) in zip(net.stages, true_bits)
        scale = Float32(2.0f0 ^ (stage.bit_position - 1))
        target_norm .+= scale .* bit_tensor
    end

    mean_tensor = reshape(mean, 1, prior_dim, 1)
    radius_tensor = reshape(radius, 1, prior_dim, 1)
    target = mean_tensor .+ radius_tensor .* target_norm

    result = ReachabilityCascade.gradients(net, context, target)

    @test length(result.stages) == length(bits)
    @test length(result.bit_targets) == length(bits)

    for (idx, stage_result) in enumerate(result.stages)
        @test isa(stage_result.loss, Number)
        @test stage_result.loss >= 0
        @test stage_result.gradient isa NamedTuple
        recovered = result.bit_targets[idx]
        @test recovered == true_bits[idx]
    end

    @test result.reconstructed ≈ target atol=1f-4
end

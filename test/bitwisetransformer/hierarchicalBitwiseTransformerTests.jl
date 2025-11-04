using Test
using Flux
using ReachabilityCascade

@testset "Hierarchical Bitwise Transformer" begin
    context_dim = 5
    prior_dim = 3
    sequence_length = 8
    batch_size = 2
    bits = [3, 2, 1]
    mean = Float32[0.5, -1.0, 1.25]
    radius = Float32[2.0, 0.75, 1.5]

    net = HierarchicalBitwiseTransformer(context_dim, prior_dim;
                                         bit_positions=bits,
                                         mean=mean,
                                         radius=radius,
                                         embed_dim=16,
                                         heads=2,
                                         ff_hidden=32)

    context = rand(Float32, context_dim, batch_size)

    result = net(context, sequence_length)
    values = result.values
    probs = result.probabilities

    @test length(probs) == length(bits)
    for p in probs
        @test size(p) == (sequence_length, prior_dim, batch_size)
        @test all(0 .<= p .<= 1)
    end

    @test size(values) == (sequence_length, prior_dim, batch_size)
    trainable = Flux.trainable(net)
    @test haskey(trainable, :stages)
    @test !haskey(trainable, :mean)
    @test !haskey(trainable, :radius)

    batch_size = ndims(context) == 1 ? 1 : size(context, 2)
    prior_tensor = zeros(Float32, sequence_length, prior_dim, batch_size)
    manual_prior = prior_tensor
    mean_tensor = reshape(mean, 1, prior_dim, 1)
    radius_tensor = reshape(radius, 1, prior_dim, 1)
    manual_probs = Vector{Any}(undef, length(bits))
    for idx in reverse(eachindex(net.stages))
        stage = net.stages[idx]
        stage_values, stage_probs = stage(context, manual_prior)
        manual_probs[idx] = stage_probs
        manual_prior = stage_values
    end
    expected_values = manual_prior .* radius_tensor .+ mean_tensor

    @test values ≈ expected_values atol=1f-5
    for (computed, expected) in zip(probs, manual_probs)
        @test computed ≈ expected atol=1f-6
    end
end

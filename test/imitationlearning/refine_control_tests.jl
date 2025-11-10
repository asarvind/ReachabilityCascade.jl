using Test
using Random
using ReachabilityCascade

const SimpleSequenceTransformer = ReachabilityCascade.SimpleSequenceTransformer
const refine_control_sequence = ReachabilityCascade.refine_control_sequence

@testset "Imitation Learning Refinement" begin
    seq_dim = 2
    context_dim = 3
    len = 4
    rng = MersenneTwister(123)

    model = SimpleSequenceTransformer(seq_dim, context_dim;
                                      hidden_dim=8,
                                      num_heads=2,
                                      num_layers=1,
                                      pos_dim=4,
                                      max_period=1_000.0)

    context = randn(rng, context_dim)
    reference = randn(rng, seq_dim, len)
    initial = zeros(seq_dim, len)

    perturb_sampler = dims -> rand(rng, (-1f0, 1f0), dims)

    network = ReachabilityCascade.IterativeRefinementNetwork(model)

    refined = refine_control_sequence(network, context, reference, initial,
                                      perturb_sampler, 3; rng=rng)
    @test size(refined) == size(reference)
end

using Test
using Random
using ReachabilityCascade

const SimpleSequenceTransformer = ReachabilityCascade.SimpleSequenceTransformer
const PerturbationGatingNetwork = ReachabilityCascade.PerturbationGatingNetwork

@testset "Perturbation Gating Network" begin
    seq_dim = 2
    context_dim = 3
    len = 4
    rng = MersenneTwister(123)

    context = randn(rng, context_dim)
    seq = zeros(seq_dim, len)
    perturb = randn(rng, seq_dim, len)
    ctx = randn(rng, context_dim)

    network = PerturbationGatingNetwork(seq_dim, context_dim;
                                        hidden_dim=8,
                                        num_heads=2,
                                        num_layers=1,
                                        pos_dim=4,
                                        max_period=1_000.0)

    outputs = network(ctx, seq, perturb)
    @test size(outputs.decisions) == size(seq)
    @test size(outputs.base) == size(seq)
    @test size(outputs.perturbed) == size(seq)
    @test size(outputs.diff) == size(seq)

    pos = abs.(randn(rng, size(seq)...))
    neg = abs.(randn(rng, size(seq)...))
    refined = network(ctx, seq, 5; pos_perturbations=pos, neg_perturbations=neg, rng=rng)
    @test size(refined) == size(seq)

    default_refined = network(ctx, seq, 3; rng=rng)
    @test size(default_refined) == size(seq)
end

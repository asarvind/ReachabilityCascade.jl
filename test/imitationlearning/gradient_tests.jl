using Test
using Random
using Flux
using ReachabilityCascade

const SimpleSequenceTransformer = ReachabilityCascade.SimpleSequenceTransformer
const IterativeRefinementNetwork = ReachabilityCascade.IterativeRefinementNetwork
const refinement_gradient = ReachabilityCascade.refinement_gradient

function setup_network(rng)
    seq_dim = 2
    context_dim = 3
    model = SimpleSequenceTransformer(seq_dim, context_dim;
                                      hidden_dim=16,
                                      num_heads=2,
                                      num_layers=1,
                                      pos_dim=4,
                                      max_period=1_000.0)
    IterativeRefinementNetwork(model)
end

@testset "Imitation Learning Gradients" begin
    rng = MersenneTwister(7)
    network = setup_network(rng)

    @testset "single sample gradient" begin
        seq = randn(rng, 2, 3)
        ctx = randn(rng, 3)
        pert = randn(rng, 2, 3)
        target = seq .+ randn(rng, 2, 3)
        loss, grads = refinement_gradient(network, ctx, seq, pert, target)
        @test loss isa Real
        grad_vec, _ = Flux.destructure(grads[1])
        @test any(!iszero, grad_vec)
    end

    @testset "batched gradient" begin
        seq_batch = randn(rng, 2, 3, 4)
        ctx_batch = randn(rng, 3, 4)
        pert_batch = randn(rng, 2, 3, 4)
        target_batch = seq_batch .+ randn(rng, 2, 3, 4)
        loss, grads = refinement_gradient(network, ctx_batch, seq_batch, pert_batch, target_batch)
        @test loss isa Real
        grad_vec, _ = Flux.destructure(grads[1])
        @test any(!iszero, grad_vec)
    end
end

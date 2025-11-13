using Test
using Random
using Flux
using ReachabilityCascade

const SimpleSequenceTransformer = ReachabilityCascade.SimpleSequenceTransformer
const PerturbationGatingNetwork = ReachabilityCascade.PerturbationGatingNetwork
const refinement_gradient = ReachabilityCascade.refinement_gradient
const sample_refinement_batch = ReachabilityCascade.sample_refinement_batch

function setup_network(rng)
    seq_dim = 2
    context_dim = 3
    model = SimpleSequenceTransformer(seq_dim, context_dim;
                                      hidden_dim=16,
                                      num_heads=2,
                                      num_layers=1,
                                      pos_dim=4,
                                      max_period=1_000.0)
    PerturbationGatingNetwork(model)
end

@testset "Imitation Learning Gradients" begin
    rng = MersenneTwister(7)
    network = setup_network(rng)

    @testset "single sample gradient" begin
        seq = randn(rng, 2, 3)
        ctx = randn(rng, 3)
        pert = randn(rng, 2, 3)
        target = seq .+ randn(rng, 2, 3)
        result = refinement_gradient(network, ctx, seq, pert, target)
        @test result.loss isa Real
        grad_vec, _ = Flux.destructure(result.gradient)
        @test any(!iszero, grad_vec)
    end

    @testset "batched gradient" begin
        seq_batch = randn(rng, 2, 3, 4)
        ctx_batch = randn(rng, 3, 4)
        pert_batch = randn(rng, 2, 3, 4)
        target_batch = seq_batch .+ randn(rng, 2, 3, 4)
        result = refinement_gradient(network, ctx_batch, seq_batch, pert_batch, target_batch)
        @test result.loss isa Real
        grad_vec, _ = Flux.destructure(result.gradient)
        @test any(!iszero, grad_vec)
    end

    @testset "sampling integration" begin
        seq_dim = 2
        len = 3
        context = randn(rng, 3)
        target = randn(rng, seq_dim, len)
        initial = zeros(seq_dim, len)
        tolerance = fill(0.5f0, seq_dim)
        data = sample_refinement_batch(context,
                                       target,
                                       initial,
                                       2,
                                       tolerance;
                                       max_steps=10,
                                       λ=0.2,
                                       rng=rng)
        result = refinement_gradient(network,
                                     data.context_batch,
                                     data.sequence_batch,
                                     data.perturb_batch,
                                     data.target_batch)
        @test result.loss isa Real
        grad_vec, _ = Flux.destructure(result.gradient)
        @test any(!iszero, grad_vec)
    end
end

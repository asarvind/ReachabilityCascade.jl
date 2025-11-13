using Test
using Random
using Flux
using ReachabilityCascade

const PerturbationGatingNetwork = ReachabilityCascade.PerturbationGatingNetwork
const load_perturbation_gating_network = ReachabilityCascade.load_perturbation_gating_network

"""
    fake_iterator(rng, count, context_dim, seq_dim, len)

Generate a re-iterable vector of `(context, target, initial)` tuples.
"""
function fake_iterator(rng, count, context_dim, seq_dim, len)
    samples = Vector{Tuple{Vector{Float64}, Matrix{Float64}, Matrix{Float64}}}(undef, count)
    for i in 1:count
        context = randn(rng, context_dim)
        target = randn(rng, seq_dim, len)
        initial = zeros(seq_dim, len)
        samples[i] = (context, target, initial)
    end
    samples
end

@testset "Perturbation gating training" begin
    rng = MersenneTwister(42)
    seq_dim = 2
    len = 3
    context_dim = 2

    data = fake_iterator(rng, 5, context_dim, seq_dim, len)
    tolerance = fill(0.5f0, seq_dim)

    network = PerturbationGatingNetwork(seq_dim, context_dim;
                                        hidden_dim=8,
                                        num_layers=1,
                                        num_heads=2,
                                        pos_dim=4,
                                        max_period=128.0)
    weights_before, _ = Flux.destructure(network)

    snapshot_path = tempname() * ".jld2"
    ispath(snapshot_path) && rm(snapshot_path)

    ReachabilityCascade.train!(network,
                               data;
                               tolerance=tolerance,
                               refinement_batch=2,
                               max_batch_size=3,
                               max_steps=5,
                               λ=0.2,
                               epochs=2,
                               rng=rng,
                               save_path=snapshot_path,
                               save_interval=0.01,
                               constructor_args=(seq_dim, context_dim),
                               constructor_kwargs=(; hidden_dim=8,
                                                    num_layers=1,
                                                    num_heads=2,
                                                    pos_dim=4,
                                                    max_period=128.0))
    weights_after, _ = Flux.destructure(network)
    @test weights_before != weights_after

    @test ispath(snapshot_path)
    loaded = load_perturbation_gating_network(snapshot_path)
    weights_loaded, _ = Flux.destructure(loaded)
    @test weights_loaded ≈ weights_after

    rm(snapshot_path; force=true)
end

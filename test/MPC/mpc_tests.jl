using Test
using Random
using LazySets
using ReachabilityCascade

@testset "MPC accepts function model" begin
    # Simple 1D system: x⁺ = x + u, with u ∈ [-1, 1]
    X = Hyperrectangle(zeros(1), ones(1))
    U = Hyperrectangle(zeros(1), ones(1))
    ds = ReachabilityCascade.DiscreteRandomSystem(X, U, (x, u) -> x .+ u)

    # "Model" as a plain function mapping (x, z) -> u.
    predictor = (x, z) -> z

    x0 = [0.0]
    cost_fn = trj -> sum(abs2, trj)

    res = ReachabilityCascade.mpc(cost_fn, ds, x0, predictor, 1;
                                 latent_dim=1,
                                 u_len=1,
                                 opt_steps=1,
                                 opt_seed=1,
                                 max_time=0.01,
                                 noise_fn=rng -> zeros(Float32, 1),
                                 noise_weight=0.0,
                                 noise_rng=Random.MersenneTwister(0))

    @test res.trajectory isa AbstractMatrix
    @test size(res.trajectory) == (1, 2)
    @test length(res.objectives) == 1
    @test size(res.u_noises) == (1, 1)
end


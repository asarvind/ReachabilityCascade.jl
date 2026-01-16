using Test
using Random
using LazySets
using ReachabilityCascade
import ReachabilityCascade.MPC: control_from_latent

struct DummyModel
    dim::Int
end

control_from_latent(model::DummyModel, z, x; u_len) = z[1:Int(u_len)]

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

@testset "optimize_latent supports derivative-free algo" begin
    X = Hyperrectangle(zeros(1), ones(1))
    U = Hyperrectangle(zeros(2), ones(2))
    ds = ReachabilityCascade.DiscreteRandomSystem(X, U, (x, u) -> x)

    model = DummyModel(2)
    x0 = [0.0]
    cost_fn = trj -> sum(abs2, trj)

    res = ReachabilityCascade.optimize_latent(cost_fn, ds, x0, model, 1;
                                             algo=:LN_BOBYQA,
                                             max_time=0.1,
                                             seed=1,
                                             u_len=2)

    @test isfinite(res.objective)
    @test length(res.z) == model.dim
end

@testset "optimize_latent accepts function model" begin
    X = Hyperrectangle(zeros(1), ones(1))
    U = Hyperrectangle(zeros(1), ones(1))
    ds = ReachabilityCascade.DiscreteRandomSystem(X, U, (x, u) -> x .+ u)

    predictor = (x, z) -> z
    x0 = [0.0]
    cost_fn = trj -> sum(abs2, trj)

    res = ReachabilityCascade.optimize_latent(cost_fn, ds, x0, predictor, 1;
                                             algo=:LN_BOBYQA,
                                             max_time=0.1,
                                             seed=1,
                                             u_len=1,
                                             latent_dim=1)

    @test isfinite(res.objective)
    @test length(res.z) == 1
end

@testset "trajectory state_jocobians" begin
    X = Hyperrectangle(zeros(1), ones(1))
    U = Hyperrectangle(zeros(1), ones(1))
    ds = ReachabilityCascade.DiscreteRandomSystem(X, U, (x, u) -> x .+ u)

    model = DummyModel(1)
    x0 = [0.0]
    z = Float32[0.1]

    res = ReachabilityCascade.trajectory(ds, model, x0, z, 2;
                                         u_len=1,
                                         jacobian_times=[1, 2, 3])

    @test size(res.state_trajectory) == (1, 3)
    @test size(res.input_trajectory) == (1, 2)
    @test length(res.state_jocobians) == 3
    @test size(res.state_jocobians[1]) == (1, 1)
    @test isapprox(res.state_jocobians[1][1, 1], 0.0; atol=1e-6)
    @test isapprox(res.state_jocobians[2][1, 1], 1.0; atol=5e-3)
    @test isapprox(res.state_jocobians[3][1, 1], 2.0; atol=5e-3)
end

@testset "trajectory state_jocobians (two-model)" begin
    X = Hyperrectangle(zeros(1), ones(1))
    U = Hyperrectangle(zeros(1), ones(1))
    ds = ReachabilityCascade.DiscreteRandomSystem(X, U, (x, u) -> x .+ u)

    model_decode = ReachabilityCascade.InvertibleGame.InvertibleCoupling(2, 1; rng=Random.MersenneTwister(1))
    model_encode = ReachabilityCascade.InvertibleGame.InvertibleCoupling(2, 1; rng=Random.MersenneTwister(2))

    res = ReachabilityCascade.trajectory(ds, model_decode, model_encode, [0.0], 1;
                                         algo=:LN_PRAXIS,
                                         max_time=0.01,
                                         seed=1,
                                         u_len=1,
                                         jacobian_times=[1, 2])

    @test size(res.state_trajectory, 1) == 1
    @test size(res.input_trajectory) == (1, 1)
    @test length(res.state_jocobians) == 2
    @test size(res.state_jocobians[1]) == (1, 2)
end

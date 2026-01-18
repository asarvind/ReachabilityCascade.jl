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

@testset "trajectory output_jacobians" begin
    X = Hyperrectangle(zeros(1), ones(1))
    U = Hyperrectangle(zeros(1), ones(1))
    ds = ReachabilityCascade.DiscreteRandomSystem(X, U, (x, u) -> x .+ u)

    model = DummyModel(1)
    x0 = [0.0]
    z = Float32[0.1]

    res = ReachabilityCascade.trajectory(ds, model, x0, z, 2;
                                         u_len=1,
                                         jacobian_times=[1, 2, 3])

    @test size(res.output_trajectory) == (1, 3)
    @test size(res.input_trajectory) == (1, 2)
    @test length(res.output_jacobians) == 3
    @test size(res.output_jacobians[1]) == (1, 1)
    @test isapprox(res.output_jacobians[1][1, 1], 0.0; atol=1e-6)
    @test isapprox(res.output_jacobians[2][1, 1], 1.0; atol=5e-3)
    @test isapprox(res.output_jacobians[3][1, 1], 2.0; atol=5e-3)
end

@testset "trajectory output_jacobians (two-model)" begin
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

    @test size(res.output_trajectory, 1) == 1
    @test size(res.input_trajectory) == (1, 1)
    @test length(res.output_jacobians) == 2
    @test size(res.output_jacobians[1]) == (1, 2)
end

@testset "smt_latent feasibility" begin
    X = Hyperrectangle(zeros(1), ones(1))
    U = Hyperrectangle(zeros(1), ones(1))
    ds = ReachabilityCascade.DiscreteRandomSystem(X, U, (x, u) -> x .+ u)

    model = DummyModel(1)
    x0 = [0.0]
    z0 = Float32[0.0]

    # Safety at time 1: x <= -0.5  => [1, 0.5] * [x; 1] <= 0
    safety = [Float64[1.0 0.5]]
    # Terminal at time 2: x <= -0.5
    terminal = [Float64[1.0 0.5]]

    z, info = ReachabilityCascade.MPC.smt_latent(
        ds,
        x0,
        model,
        z0,
        2,
        safety,
        terminal,
        [1],
        nothing;
        big_m=10.0,
    )

    @test info.feasible
    @test z isa AbstractVector
    @test z[1] <= -0.5 + 1e-3
end

@testset "smt_critical_evaluations default input bounds" begin
    X = Hyperrectangle(zeros(1), ones(1))
    U = Hyperrectangle(zeros(2), ones(2))
    ds = ReachabilityCascade.DiscreteRandomSystem(X, U, (x, u) -> x .+ u[1])

    model = DummyModel(2)
    x0 = [0.0]
    z = Float32[0.1, -0.1]

    safety_output = [Float64[1.0 0.0]]
    terminal_output = [Float64[-1.0 0.0]]

    res = ReachabilityCascade.MPC.smt_critical_evaluations(
        ds,
        model,
        x0,
        z,
        2,
        safety_output,
        terminal_output;
        u_len=2,
    )

    @test length(res.safety_output) == 1
    @test length(res.safety_input) == 4
    @test length(res.terminal_output) == 1
    @test length(res.safety_input[1]) == 1
end

@testset "smt_affine_critical matches base at z_ref" begin
    X = Hyperrectangle(zeros(1), ones(1))
    U = Hyperrectangle(zeros(1), ones(1))
    ds = ReachabilityCascade.DiscreteRandomSystem(X, U, (x, u) -> x .+ u[1])

    model = DummyModel(1)
    x0 = [0.0]
    z_ref = Float32[0.2]

    safety_output = [Float64[1.0 0.0]]
    safety_input = [Float64[1.0 0.0]]
    terminal_output = [Float64[-1.0 0.0]]

    base = ReachabilityCascade.MPC.smt_critical_evaluations(
        ds,
        model,
        x0,
        z_ref,
        2,
        safety_output,
        safety_input,
        terminal_output;
        u_len=1,
    )

    affine = ReachabilityCascade.MPC.smt_affine_critical(
        ds,
        model,
        x0,
        z_ref,
        2,
        safety_output,
        safety_input,
        terminal_output;
        u_len=1,
        eps=1f-6,
    )

    @test affine.safety_output[1][1, end] ≈ base.safety_output[1][1]
    @test affine.safety_input[1][1, end] ≈ base.safety_input[1][1]
    @test affine.terminal_output[1][1, end] ≈ base.terminal_output[1][1]
end

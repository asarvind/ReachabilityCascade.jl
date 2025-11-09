using Test
using Random
using LazySets
using ReachabilityCascade

const perturb_input_sequence = ReachabilityCascade.perturb_input_sequence

function simple_discrete_system()
    X = Hyperrectangle(zeros(2), fill(100.0, 2))
    U = Hyperrectangle(zeros(2), fill(100.0, 2))
    f = (x::AbstractVector, u::AbstractVector) -> x .+ u
    DiscreteRandomSystem(X, U, f)
end

function bounded_input_system(radius::Real)
    X = Hyperrectangle(zeros(2), fill(100.0, 2))
    U = Hyperrectangle(zeros(2), fill(radius, 2))
    f = (x::AbstractVector, u::AbstractVector) -> x .+ u
    DiscreteRandomSystem(X, U, f)
end

@testset "Behavioral Cloning Perturbations" begin
    ds = simple_discrete_system()
    x0 = zeros(2)

    @testset "accepts higher cost perturbations" begin
        u_ref = zeros(2, 3)
        perturbation = fill(0.2, 2)
        stage_cost = (_, u) -> sum(abs2, u)
        terminal_cost = x -> sum(abs2, x)
        rng = MersenneTwister(1234)

        result = perturb_input_sequence(ds, x0, u_ref, perturbation, stage_cost, terminal_cost; rng=rng)
        @test size(result.inputs) == size(u_ref)
        @test size(result.states, 2) == size(result.inputs, 2) + 1
        @test any(!iszero, result.inputs)
    end

    @testset "rejects when cost unchanged" begin
        u_ref = zeros(2, 2)
        perturbation = zeros(2)
        stage_cost = (_, u) -> sum(abs2, u)
        terminal_cost = x -> sum(abs2, x)

        result = perturb_input_sequence(ds, x0, u_ref, perturbation, stage_cost, terminal_cost)
        @test size(result.inputs, 2) == 0
        @test size(result.states, 2) == 0
    end

    @testset "applies total cost threshold" begin
        u_ref = ones(2, 1)
        perturbation = fill(0.1, 2)
        stage_cost = (_, u) -> sum(abs2, u)
        terminal_cost = x -> sum(abs2, x)

        result = perturb_input_sequence(ds, x0, u_ref, perturbation, stage_cost, terminal_cost;
                                        total_cost_threshold=0.5)
        @test size(result.inputs, 2) == 0
        @test size(result.states, 2) == 0
    end

    @testset "supports two-argument terminal cost" begin
        u_ref = zeros(2, 1)
        perturbation = fill(0.3, 2)
        stage_cost = (_, u) -> sum(abs2, u)
        terminal_cost = (x, u) -> sum(abs2, x .+ u)

        result = perturb_input_sequence(ds, x0, u_ref, perturbation, stage_cost, terminal_cost; rng=MersenneTwister(1))
        @test size(result.inputs) == size(u_ref)
        @test size(result.states, 2) == size(result.inputs, 2) + 1
    end

    @testset "discount factor affects thresholding" begin
        u_ref = ones(2, 3)
        perturbation = fill(0.5, 2)
        stage_cost = (_, u) -> sum(abs2, u)
        terminal_cost = x -> 0.0

        reject = perturb_input_sequence(ds, x0, u_ref, perturbation, stage_cost, terminal_cost;
                                        total_cost_threshold=4.0, discount_factor=1.0)
        @test size(reject.inputs, 2) == 0

        rng = MersenneTwister(7)
        accept = perturb_input_sequence(ds, x0, u_ref, perturbation, stage_cost, terminal_cost;
                                        total_cost_threshold=4.0, discount_factor=0.0, rng=rng)
        @test size(accept.inputs) == size(u_ref)
    end

    @testset "inputs are clamped to bounds" begin
        ds_bounded = bounded_input_system(0.05)
        u_ref = zeros(2, 3)
        perturbation = fill(1.0, 2)
        stage_cost = (_, u) -> sum(abs2, u)
        terminal_cost = x -> sum(abs2, x)
        rng = MersenneTwister(99)

        results = perturb_input_sequence(ds_bounded, zeros(2), u_ref, perturbation, stage_cost, terminal_cost, 5; rng=rng)
        @test !isempty(results)
        u_lo = low(ds_bounded.U) .- eps(Float64)
        u_hi = high(ds_bounded.U) .+ eps(Float64)
        for res in results
            @test all(res.inputs .>= u_lo)
            @test all(res.inputs .<= u_hi)
        end
    end

    @testset "iterative perturbation accumulates successes" begin
        u_ref = zeros(2, 2)
        perturbation = fill(0.4, 2)
        stage_cost = (_, u) -> sum(abs2, u)
        terminal_cost = x -> sum(abs2, x)
        rng = MersenneTwister(42)

        results = perturb_input_sequence(ds, x0, u_ref, perturbation, stage_cost, terminal_cost, 5; rng=rng)
        @test !isempty(results)
        @test length(results) <= 5
        @test all(size(res.inputs, 2) > 0 for res in results)
        @test all(size(res.states, 2) == size(res.inputs, 2) + 1 for res in results)
        @test all(res -> res isa NamedTuple{(:inputs, :states)}, results)
    end
end

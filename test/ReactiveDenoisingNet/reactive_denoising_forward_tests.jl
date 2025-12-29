using Test
using ReachabilityCascade

@testset "ReactiveDenoisingNet forward pass" begin
    state_dim = 4
    input_dim = 2
    cost_dim = 3
    seq_len = 5

    model = ReactiveDenoisingNet(state_dim, input_dim, cost_dim, seq_len, 16, 2)

    @testset "single" begin
        x0 = rand(Float32, state_dim)
        x_body = rand(Float32, state_dim, seq_len)
        u_guess = rand(Float32, input_dim, seq_len)
        cost_body = rand(Float32, cost_dim, seq_len)
        out = model(x0, x_body, u_guess, cost_body)
        @test size(out.U_new) == size(u_guess)
        @test size(out.noise) == size(u_guess)
        @test out.U_new ≈ (u_guess .- out.noise)
    end

    @testset "batch" begin
        B = 3
        x0 = rand(Float32, state_dim, B)
        x_body = rand(Float32, state_dim, seq_len, B)
        u_guess = rand(Float32, input_dim, seq_len, B)
        cost_body = rand(Float32, cost_dim, seq_len, B)
        out = model(x0, x_body, u_guess, cost_body)
        @test size(out.U_new) == size(u_guess)
        @test size(out.noise) == size(u_guess)
        @test out.U_new ≈ (u_guess .- out.noise)
    end
end

@testset "ReactiveDenoisingNet recursive refinement" begin
    state_dim = 3
    input_dim = 2
    cost_dim = 4
    seq_len = 6
    steps = 3

    model = ReactiveDenoisingNet(state_dim, input_dim, cost_dim, seq_len, 16, 2)

    # Simple deterministic rollout: x_{t+1} = x_t + A*u_t
    A = rand(Float32, state_dim, input_dim)
    sys = function (x0::Vector{<:Real}, U::Matrix{<:Real})
        size(U, 1) == input_dim || throw(DimensionMismatch("U must have $input_dim rows"))
        size(U, 2) == seq_len || throw(DimensionMismatch("U must have $seq_len cols"))
        X = Matrix{Float64}(undef, state_dim, seq_len + 1)
        X[:, 1] = Float64.(x0)
        for t in 1:seq_len
            X[:, t + 1] = X[:, t] .+ Float64.(A * Float32.(U[:, t]))
        end
        return X
    end

    traj_cost_fn = function (x_body::AbstractMatrix)
        size(x_body, 1) == state_dim || throw(DimensionMismatch("x_body must have $state_dim rows"))
        size(x_body, 2) == seq_len || throw(DimensionMismatch("x_body must have $seq_len cols"))
        base = Float32.(sum(abs2, x_body; dims=1))
        return vcat((i * base for i in 1:cost_dim)...)
    end

    x0 = rand(Float32, state_dim)
    u0 = rand(Float32, input_dim, seq_len)
    out = model(x0, u0, sys, traj_cost_fn; steps=steps)

    @test length(out.u_guesses) == steps + 1
    @test length(out.noises) == steps
    @test length(out.x_rollouts) == steps
    @test length(out.costs) == steps
    @test size(out.u_guesses[1]) == (input_dim, seq_len)
    @test size(out.u_guesses[end]) == (input_dim, seq_len)
    @test all(size(n) == (input_dim, seq_len) for n in out.noises)
    @test all(size(X) == (state_dim, seq_len + 1) for X in out.x_rollouts)
    @test all(size(C) == (cost_dim, seq_len) for C in out.costs)
end

@testset "ReactiveDenoisingNet imitation gradient (final-step only)" begin
    state_dim = 3
    input_dim = 2
    cost_dim = 4
    seq_len = 6
    steps = 3

    model = ReactiveDenoisingNet(state_dim, input_dim, cost_dim, seq_len, 16, 2)

    A = rand(Float32, state_dim, input_dim)
    sys = function (x0::Vector{<:Real}, U::Matrix{<:Real})
        X = Matrix{Float64}(undef, state_dim, seq_len + 1)
        X[:, 1] = Float64.(x0)
        for t in 1:seq_len
            X[:, t + 1] = X[:, t] .+ Float64.(A * Float32.(U[:, t]))
        end
        return X
    end

    traj_cost_fn = function (x_body::AbstractMatrix)
        base = Float32.(sum(abs2, x_body; dims=1))
        return vcat((i * base for i in 1:cost_dim)...)
    end

    x0 = rand(Float32, state_dim)
    u0 = rand(Float32, input_dim, seq_len)
    u_target = rand(Float32, input_dim, seq_len)
    scale = rand(Float32, input_dim)

    grads = ReachabilityCascade.TrainingAPI.gradient(model, x0, u0, u_target, sys, traj_cost_fn; steps=steps, scale=scale)
    opt_state = Flux.setup(Flux.Descent(1f-3), model)
    Flux.update!(opt_state, model, grads)  # should be compatible
end

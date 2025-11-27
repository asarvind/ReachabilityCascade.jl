using Test
using Flux
using ReachabilityCascade

@testset "TransitionNetwork" begin
    state_dim = 4
    input_dim = 2
    hidden_dim = 8
    net = TransitionNetwork(state_dim, input_dim, hidden_dim)

    @testset "2D batch" begin
        batch = 3
        x = rand(Float32, state_dim, batch)
        u = rand(Float32, input_dim, batch)
        y = net(x, u)
        @test size(y) == (state_dim, batch)
    end

    @testset "3D time-batch" begin
        batch = 2
        time = 5
        x = rand(Float32, state_dim, time, batch)
        u = rand(Float32, input_dim, time, batch)
        y = net(x, u)
        @test size(y) == (state_dim, time, batch)
    end

    @testset "determinism" begin
        batch = 1
        x = rand(Float32, state_dim, batch)
        u = rand(Float32, input_dim, batch)
        y1 = net(x, u)
        y2 = net(x, u)
        @test y1 ≈ y2
    end

    @testset "rollout single trajectory" begin
        T = 4
        x0 = rand(Float32, state_dim)
        U = rand(Float32, input_dim, T)
        X = net(x0, U)
        @test size(X) == (state_dim, T + 1)
        @test X[:, 1] ≈ x0
        # Manual unroll for comparison
        x_prev = reshape(x0, :, 1)
        for t in 1:T
            u_t = @view U[:, t:t]
            x_prev = net(x_prev, u_t)
            @test X[:, t + 1] ≈ vec(x_prev)
        end
    end

    @testset "rollout batched trajectories" begin
        T = 3
        B = 2
        x0 = rand(Float32, state_dim, B)
        U = rand(Float32, input_dim, T, B)
        X = net(x0, U)
        @test size(X) == (state_dim, T + 1, B)
        @test X[:, 1, :] ≈ x0
        x_prev = x0
        for t in 1:T
            u_t = @view U[:, t, :]
            x_prev = net(x_prev, u_t)
            @test X[:, t + 1, :] ≈ x_prev
        end
    end
end

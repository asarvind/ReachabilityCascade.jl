using Test
using Flux
using ReachabilityCascade

@testset "ReactiveDenoisingNet train! (imitation)" begin
    state_dim = 3
    input_dim = 2
    cost_dim = 2
    seq_len = 4

    model = ReactiveDenoisingNet(state_dim, input_dim, cost_dim, seq_len, 8, 1)

    sys = function (x0::Vector{<:Real}, U::Matrix{<:Real})
        X = zeros(Float64, state_dim, seq_len + 1)
        X[:, 1] = Float64.(x0)
        for t in 1:seq_len
            X[:, t + 1] = X[:, t]
        end
        return X
    end

    traj_cost_fn = function (x_body::AbstractMatrix)
        return rand(Float32, cost_dim, size(x_body, 2))
    end

    data = [(x0=rand(Float32, state_dim), u_target=rand(Float32, input_dim, seq_len)) for _ in 1:3]
    res = ReachabilityCascade.TrainingAPI.train!(model, data, sys, traj_cost_fn; epochs=1, steps=1, save_path="")
    @test res.model === model
    @test !isempty(res.losses)
end

@testset "ReactiveDenoisingNet build" begin
    state_dim = 3
    input_dim = 2
    cost_dim = 2
    seq_len = 4

    sys = function (x0::Vector{<:Real}, U::Matrix{<:Real})
        X = zeros(Float64, state_dim, seq_len + 1)
        X[:, 1] = Float64.(x0)
        for t in 1:seq_len
            X[:, t + 1] = X[:, t]
        end
        return X
    end

    traj_cost_fn = function (x_body::AbstractMatrix)
        return rand(Float32, cost_dim, size(x_body, 2))
    end

    data = [(x0=rand(Float32, state_dim), u_target=rand(Float32, input_dim, seq_len)) for _ in 1:3]

    model, losses = ReachabilityCascade.TrainingAPI.build(ReactiveDenoisingNet, data, sys, traj_cost_fn;
                                                         hidden_dim=8, depth=1,
                                                         epochs=1, steps=1,
                                                         save_path="")
    @test model isa ReactiveDenoisingNet
    @test !isempty(losses)
end

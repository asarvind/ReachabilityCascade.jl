using Test
using Random
using ReachabilityCascade

@testset "ReactiveDenoisingNet testrun" begin
    state_dim = 3
    input_dim = 2
    cost_dim = 2
    seq_len = 4
    steps = 2

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
        return zeros(Float32, cost_dim, size(x_body, 2))
    end

    rng = Random.MersenneTwister(123)
    x0s = [rand(rng, Float32, state_dim) for _ in 1:2]

    res = ReachabilityCascade.ReactiveDenoisingNetworks.testrun(model, x0s, sys, traj_cost_fn; steps=steps, rng=rng, temperature=1.0)
    @test length(res) == length(x0s)

    # Includes initial guess as candidate; with all-zero costs, argmin picks the first (initial guess).
    @test all(r -> r.best_idx == 1, res)

    # Different initial guesses per sample when using a single RNG.
    @test res[1].u_guesses[1] != res[2].u_guesses[1]
end


using Test
using ReachabilityCascade

@testset "Trajectory Transformer" begin
    state_dim, goal_dim, latent_dim = 3, 2, 4
    seq_len = 5

    net = TrajectoryTransformer(state_dim, goal_dim, latent_dim;
                                embed_dim=16, heads=2, ff_hidden=32)

    current_state = rand(Float32, state_dim)
    goal = rand(Float32, goal_dim)
    latents = rand(Float32, latent_dim, seq_len)

    result = transform_sequence(net, current_state, goal, latents; steps=2)

    @test result.states isa Matrix{Float32}
    @test size(result.states) == (state_dim, seq_len)
    @test result.latents isa Matrix{Float32}
    @test size(result.latents) == (latent_dim, seq_len)
    @test length(result.state_history) == 0
    @test length(result.latent_history) == 0

    state_seq = predict_state_sequence(net, current_state, goal, latents; steps=3)
    latent_seq = predict_latent_sequence(net, current_state, goal, latents; steps=3)

    hist_result = transform_sequence(net, current_state, goal, latents;
                                     steps=3, return_history=true)
    @test length(hist_result.state_history) == 3
    @test length(hist_result.latent_history) == 3
    @test hist_result.state_history isa Vector{Matrix{Float32}}
    @test hist_result.latent_history isa Vector{Matrix{Float32}}
    @test hist_result.state_history[1] isa Matrix{Float32}
    @test hist_result.latent_history[1] isa Matrix{Float32}

    @test size(state_seq) == (state_dim, seq_len)
    @test size(latent_seq) == (latent_dim, seq_len)
end

using Test
using Random
using ReachabilityCascade
using Flux

@testset "TransitionNetwork training + save/load" begin
    rng = Random.MersenneTwister(42)

    # Simple linear system: next = state + input
    state_dim, input_dim = 2, 1
    dataset = [(state=s, input=u, next=s .+ u)
               for (s, u) in zip([rand(rng, Float32, state_dim) for _ in 1:200],
                                 [rand(rng, Float32, input_dim) for _ in 1:200])]

    hidden_dim = 8
    depth = 2
    model, losses = fit_transition_network(dataset; hidden_dim=hidden_dim, depth=depth, epochs=20, batchsize=32, opt=Flux.Adam(1e-2), rng=rng)
    @test !isempty(losses)

    sample = dataset[1]
    x_col = reshape(sample.state, :, 1)
    u_col = reshape(sample.input, :, 1)
    pred = model(x_col, u_col)[:, 1]
    @test isapprox(pred, sample.next; atol=0.5)

    temp_dir = mktempdir()
    path = joinpath(temp_dir, "transition_net.jld2")
    save_transition_network(path, model)
    @test isfile(path)

    loaded = load_transition_network(path)
    pred_loaded = loaded(x_col, u_col)[:, 1]
    @test isapprox(pred_loaded, sample.next; atol=0.5)
end

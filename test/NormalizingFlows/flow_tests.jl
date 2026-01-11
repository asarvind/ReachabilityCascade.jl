using Test
using Random
using ReachabilityCascade
using Flux

@testset "NormalizingFlow" begin
    rng = Random.MersenneTwister(123)
    D = 8
    C = 3
    B = 5

    # 2 spec columns -> 4 coupling sublayers (each column expands into two complementary masks)
    spec = [64 32;
            1  2;
            1  0]

    flow = NormalizingFlow(D, C; spec=spec, logscale_clamp=2.0, rng=rng)

    x = randn(rng, Float32, D, B)
    c = randn(rng, Float32, C, B)

    z, logdet = encode(flow, x, c)
    @test size(z) == (D, B)
    @test size(logdet) == (B,)

    x_rec = decode(flow, z, c)
    @test size(x_rec) == (D, B)
    @test x_rec ≈ x atol=1e-4

    # Vector API (single sample) should also invert accurately.
    x1 = x[:, 1]
    c1 = c[:, 1]
    z1, logdet1 = encode(flow, x1, c1)
    @test size(z1) == (D,)
    @test isa(logdet1, Float32)
    x1_rec = decode(flow, z1, c1)
    @test size(x1_rec) == (D,)
    @test x1_rec ≈ x1 atol=1e-4

    # Default flow starts near identity (zero-init last layer), so logdet should be near 0.
    @test maximum(abs.(logdet)) ≤ 1e-3
end

@testset "TrainingAPI.gradient (NormalizingFlow)" begin
    rng = Random.MersenneTwister(7)
    D = 6
    C = 2
    B = 4

    flow = NormalizingFlow(D, C; rng=rng)
    x = randn(rng, Float32, D, B)
    c = randn(rng, Float32, C, B)

    grads = ReachabilityCascade.TrainingAPI.gradient(flow, x, c)
    @test haskey(grads, :layers)
    @test length(grads.layers) == length(flow.layers)

    grads2, loss = ReachabilityCascade.TrainingAPI.gradient(flow, x, c; return_loss=true)
    @test haskey(grads2, :layers)
    @test length(grads2.layers) == length(flow.layers)
    @test isfinite(loss)
    opt_state = Flux.setup(Flux.Descent(1f-3), flow)
    Flux.update!(opt_state, flow, grads)
end

@testset "TrainingAPI.save/load (NormalizingFlow)" begin
    rng = Random.MersenneTwister(1234)
    D = 5
    C = 3
    B = 4

    flow = NormalizingFlow(D, C; rng=rng)
    x = randn(rng, Float32, D, B)
    c = randn(rng, Float32, C, B)

    # Save, then load, then verify encode/decode behavior is preserved.
    mktemp() do path, io
        close(io)
        ReachabilityCascade.TrainingAPI.save(flow, path)
        flow2 = ReachabilityCascade.TrainingAPI.load(NormalizingFlow, path)

        z1, ld1 = encode(flow, x, c)
        z2, ld2 = encode(flow2, x, c)
        @test z2 ≈ z1 atol=1e-5
        @test ld2 ≈ ld1 atol=1e-5

        xrec = decode(flow2, z2, c)
        @test xrec ≈ x atol=1e-4
    end
end

@testset "TrainingAPI.build (NormalizingFlow) return shapes" begin
    data = [(; context=randn(Float32, 2), sample=randn(Float32, 3)) for _ in 1:10]

    flow, losses = ReachabilityCascade.TrainingAPI.build(NormalizingFlow, data; epochs=1, batch_size=4)
    @test flow isa NormalizingFlow
    @test losses isa Vector{Float32}

    flow2, losses2, state_before = ReachabilityCascade.TrainingAPI.build(NormalizingFlow, data;
                                                                        epochs=1,
                                                                        batch_size=4,
                                                                        return_state_before=true)
    @test flow2 isa NormalizingFlow
    @test losses2 isa Vector{Float32}
    @test state_before isa NamedTuple

    flow3, losses3, scores_fresh3, scores_memory3 = ReachabilityCascade.TrainingAPI.build(NormalizingFlow, data;
                                                                                         epochs=1,
                                                                                         batch_size=4,
                                                                                         return_scores=true)
    @test flow3 isa NormalizingFlow
    @test losses3 isa Vector{Float32}
    @test scores_fresh3 isa Vector{Float32}
    @test scores_memory3 isa Vector{Float32}
end

@testset "TrainingAPI.train! state_before snapshot" begin
    rng = Random.MersenneTwister(0)
    data = [(; context=randn(rng, Float32, 2), sample=randn(rng, Float32, 3)) for _ in 1:20]
    flow = NormalizingFlow(3, 2; rng=rng)

    res = ReachabilityCascade.TrainingAPI.train!(flow, data; epochs=1, batch_size=5, opt=Flux.Descent(1f-2))
    @test haskey(res, :state_before)

    # Snapshot should not alias the model parameters (should differ after training).
    θ_before, _ = Flux.destructure(res.state_before)
    θ_after, _ = Flux.destructure(res.model)
    @test θ_after != θ_before
end

@testset "TrainingAPI.train! use_memory" begin
    rng = Random.MersenneTwister(1)
    data = [(; context=randn(rng, Float32, 2), sample=randn(rng, Float32, 3)) for _ in 1:20]
    flow = NormalizingFlow(3, 2; rng=rng)

    # 20 samples, batch_size=5 -> 4 updates per epoch -> 8 losses over 2 epochs.
    res = ReachabilityCascade.TrainingAPI.train!(flow, data;
                                                 epochs=2,
                                                 batch_size=5,
                                                 use_memory=true,
                                                 opt=Flux.Descent(1f-2))
    @test length(res.losses) == 8
    @test length(res.scores_fresh) == 8
    @test length(res.scores_memory) == 8
end

@testset "TrainingAPI.train! use_memory sign_agree" begin
    rng = Random.MersenneTwister(2)
    data = [(; context=randn(rng, Float32, 2), sample=randn(rng, Float32, 3)) for _ in 1:20]
    flow = NormalizingFlow(3, 2; rng=rng)

    res = ReachabilityCascade.TrainingAPI.train!(flow, data;
                                                 epochs=2,
                                                 batch_size=5,
                                                 use_memory=true,
                                                 memory_merge=:sign_agree,
                                                 opt=Flux.Descent(1f-2))
    @test length(res.losses) == 8
    @test length(res.scores_fresh) == 8
    @test length(res.scores_memory) == 8
end

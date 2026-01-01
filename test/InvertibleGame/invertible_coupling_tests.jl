using Test
using Random
using Flux
using ReachabilityCascade

@testset "InvertibleGame.InvertibleCoupling" begin
    rng = Random.MersenneTwister(123)
    D = 8
    C = 3
    B = 5

    spec = [64 32;
            1  2;
            1  0]

    net = ReachabilityCascade.InvertibleGame.InvertibleCoupling(D, C; spec=spec, logscale_clamp=2.0, rng=rng)

    x = randn(rng, Float32, D, B)
    c = randn(rng, Float32, C, B)

    z = ReachabilityCascade.InvertibleGame.encode(net, x, c)
    @test size(z) == (D, B)

    x_rec = ReachabilityCascade.InvertibleGame.decode(net, z, c)
    @test size(x_rec) == (D, B)
    @test x_rec ≈ x atol=1e-4

    # Vector API
    x1 = x[:, 1]
    c1 = c[:, 1]
    z1 = ReachabilityCascade.InvertibleGame.encode(net, x1, c1)
    @test size(z1) == (D,)
    x1_rec = ReachabilityCascade.InvertibleGame.decode(net, z1, c1)
    @test x1_rec ≈ x1 atol=1e-4
end

@testset "TrainingAPI.train!/build (InvertibleCoupling two-player)" begin
    rng_data = Random.MersenneTwister(1)
    rng_a = Random.MersenneTwister(2)
    rng_b = Random.MersenneTwister(3)
    rng_latent = Random.MersenneTwister(4)

    D = 5
    C = 2
    N = 20

    data = [(; context=randn(rng_data, Float32, C), sample=randn(rng_data, Float32, D)) for _ in 1:N]

    model_a, model_b, losses_a, losses_b = ReachabilityCascade.TrainingAPI.build(
        ReachabilityCascade.InvertibleGame.InvertibleCoupling,
        data;
        epochs=1,
        batch_size=8,
        margin_true=0.5,
        margin_adv=0.2,
        rng=rng_latent,
        rng_a=rng_a,
        rng_b=rng_b,
        opt=Flux.Descent(1f-3),
    )

    @test model_a isa ReachabilityCascade.InvertibleGame.InvertibleCoupling
    @test model_b isa ReachabilityCascade.InvertibleGame.InvertibleCoupling
    @test length(losses_a) > 0
    @test length(losses_b) == length(losses_a)
    @test all(isfinite, losses_a)
    @test all(isfinite, losses_b)
end

@testset "TrainingAPI.train!/build (InvertibleCoupling two-player) use_memory" begin
    rng_data = Random.MersenneTwister(21)
    rng_a = Random.MersenneTwister(22)
    rng_b = Random.MersenneTwister(23)
    rng_latent = Random.MersenneTwister(24)

    D = 5
    C = 2
    N = 24
    batch_size = 8

    data = [(; context=randn(rng_data, Float32, C), sample=randn(rng_data, Float32, D)) for _ in 1:N]

    model_a, model_b, losses_a, losses_b = ReachabilityCascade.TrainingAPI.build(
        ReachabilityCascade.InvertibleGame.InvertibleCoupling,
        data;
        epochs=1,
        batch_size=batch_size,
        use_memory=true,
        margin_true=0.5,
        margin_adv=0.2,
        rng=rng_latent,
        rng_a=rng_a,
        rng_b=rng_b,
        opt=Flux.Descent(1f-3),
    )

    @test model_a isa ReachabilityCascade.InvertibleGame.InvertibleCoupling
    @test model_b isa ReachabilityCascade.InvertibleGame.InvertibleCoupling
    @test length(losses_a) > 0
    @test length(losses_b) == length(losses_a)
    @test all(isfinite, losses_a)
    @test all(isfinite, losses_b)
end

@testset "TrainingAPI.gradient (InvertibleCoupling game loss)" begin
    rng = Random.MersenneTwister(9)
    D = 6
    C = 2
    B = 7

    a = ReachabilityCascade.InvertibleGame.InvertibleCoupling(D, C; rng=Random.MersenneTwister(10))
    b = ReachabilityCascade.InvertibleGame.InvertibleCoupling(D, C; rng=Random.MersenneTwister(11))

    x_true = randn(rng, Float32, D, B)
    c = randn(rng, Float32, C, B)

    grads, loss, extras = ReachabilityCascade.TrainingAPI.gradient(a, b, x_true, c;
                                                                   rng=rng,
                                                                   return_loss=true,
                                                                   return_true_hinges=true,
                                                                   return_components=true)
    @test haskey(grads, :layers)
    @test isfinite(loss)
    @test extras isa NamedTuple
    @test haskey(extras, :true_hinges)
    @test length(extras.true_hinges) == B
    @test all(isfinite, extras.true_hinges)
    @test all(isfinite, (extras.accept_true, extras.reject_other, extras.fool_other))
end

@testset "InvertibleGame.inclusion_losses" begin
    rng = Random.MersenneTwister(100)
    D = 5
    C = 2
    N = 17

    model = ReachabilityCascade.InvertibleGame.InvertibleCoupling(D, C; rng=Random.MersenneTwister(101))
    data = [(; context=randn(rng, Float32, C), sample=randn(rng, Float32, D)) for _ in 1:N]

    losses = ReachabilityCascade.InvertibleGame.inclusion_losses(model, data; batch_size=4)
    @test losses isa Vector{Float32}
    @test length(losses) == N
    @test all(isfinite, losses)
end

@testset "TrainingAPI.save/load (InvertibleCoupling)" begin
    rng = Random.MersenneTwister(200)
    D = 6
    C = 3
    B = 4

    model = ReachabilityCascade.InvertibleGame.InvertibleCoupling(D, C; rng=rng)
    x = randn(rng, Float32, D, B)
    c = randn(rng, Float32, C, B)
    z1 = ReachabilityCascade.InvertibleGame.encode(model, x, c)
    xrec1 = ReachabilityCascade.InvertibleGame.decode(model, z1, c)

    mktemp() do path, io
        close(io)
        ReachabilityCascade.TrainingAPI.save(model, path)
        model2 = ReachabilityCascade.TrainingAPI.load(ReachabilityCascade.InvertibleGame.InvertibleCoupling, path)
        z2 = ReachabilityCascade.InvertibleGame.encode(model2, x, c)
        xrec2 = ReachabilityCascade.InvertibleGame.decode(model2, z2, c)
        @test z2 ≈ z1 atol=1e-5
        @test xrec2 ≈ xrec1 atol=1e-5
    end
end

@testset "InvertibleGame.save_game/load_game" begin
    rng = Random.MersenneTwister(300)
    D = 5
    C = 2
    B = 3

    a = ReachabilityCascade.InvertibleGame.InvertibleCoupling(D, C; rng=Random.MersenneTwister(301))
    b = ReachabilityCascade.InvertibleGame.InvertibleCoupling(D, C; rng=Random.MersenneTwister(302))
    x = randn(rng, Float32, D, B)
    c = randn(rng, Float32, C, B)

    z_a1 = ReachabilityCascade.InvertibleGame.encode(a, x, c)
    z_b1 = ReachabilityCascade.InvertibleGame.encode(b, x, c)

    mktemp() do path, io
        close(io)
        ReachabilityCascade.InvertibleGame.save_game(path, a, b)
        a2, b2, _ = ReachabilityCascade.InvertibleGame.load_game(path)
        z_a2 = ReachabilityCascade.InvertibleGame.encode(a2, x, c)
        z_b2 = ReachabilityCascade.InvertibleGame.encode(b2, x, c)
        @test z_a2 ≈ z_a1 atol=1e-5
        @test z_b2 ≈ z_b1 atol=1e-5
    end
end

@testset "TrainingAPI.build (InvertibleCoupling) auto checkpoint" begin
    rng_data = Random.MersenneTwister(400)
    D = 5
    C = 2
    N = 16
    batch_size = 8
    data = [(; context=randn(rng_data, Float32, C), sample=randn(rng_data, Float32, D)) for _ in 1:N]

    mktempdir() do dir
        path = joinpath(dir, "invertiblegame.jld2")
        model_a, model_b, losses_a, losses_b = ReachabilityCascade.TrainingAPI.build(
            ReachabilityCascade.InvertibleGame.InvertibleCoupling,
            data;
            epochs=1,
            batch_size=batch_size,
            margin_true=0.5,
            margin_adv=0.2,
            opt=Flux.Descent(1f-3),
            save_path=path,
            load_path=path,
            save_period=0.0,
        )
        @test isfile(path)

        a2, b2, _ = ReachabilityCascade.InvertibleGame.load_game(path)
        x = randn(rng_data, Float32, D, 3)
        c = randn(rng_data, Float32, C, 3)
        @test ReachabilityCascade.InvertibleGame.encode(a2, x, c) ≈ ReachabilityCascade.InvertibleGame.encode(model_a, x, c) atol=1e-5
        @test ReachabilityCascade.InvertibleGame.encode(b2, x, c) ≈ ReachabilityCascade.InvertibleGame.encode(model_b, x, c) atol=1e-5

        @test length(losses_a) > 0
        @test length(losses_b) == length(losses_a)
    end
end

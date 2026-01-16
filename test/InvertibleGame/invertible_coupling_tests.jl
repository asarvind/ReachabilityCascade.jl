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

@testset "TrainingAPI.train!/build (InvertibleCoupling self-adversarial)" begin
    rng_data = Random.MersenneTwister(500)
    rng_latent = Random.MersenneTwister(501)

    D = 5
    C = 2
    N = 20

    data = [(; context=randn(rng_data, Float32, C), sample=randn(rng_data, Float32, D)) for _ in 1:N]

    model, ema, losses = ReachabilityCascade.TrainingAPI.build(
        ReachabilityCascade.InvertibleGame.InvertibleCoupling,
        data;
        epochs=1,
        batch_size=8,
        margin_true=0.5,
        margin_adv=0.2,
        ema_beta_start=0.9,
        ema_beta_final=0.9,
        ema_tau=1.0,
        use_memory=true,
        rng=rng_latent,
        rng_model=Random.MersenneTwister(502),
        opt=Flux.Descent(1f-3),
    )

    @test model isa ReachabilityCascade.InvertibleGame.InvertibleCoupling
    @test ema isa ReachabilityCascade.InvertibleGame.InvertibleCoupling
    @test length(losses) > 0
    @test all(isfinite, losses)
end

@testset "TrainingAPI.gradient (InvertibleCoupling self loss)" begin
    rng = Random.MersenneTwister(9)
    D = 6
    C = 2
    B = 7

    a = ReachabilityCascade.InvertibleGame.InvertibleCoupling(D, C; rng=Random.MersenneTwister(10))
    b = ReachabilityCascade.InvertibleGame.InvertibleCoupling(D, C; rng=Random.MersenneTwister(11))

    x_true = randn(rng, Float32, D, B)
    c = randn(rng, Float32, C, B)

    for norm_kind in (:l1, :l2, :linf)
        grads, loss, extras = ReachabilityCascade.TrainingAPI.gradient(a, b, x_true, c;
                                                                       rng=rng,
                                                                       norm_kind=norm_kind,
                                                                       return_loss=true,
                                                                       return_true_hinges=true,
                                                                       return_components=true)
        @test haskey(grads, :layers)
        @test isfinite(loss)
        @test extras isa NamedTuple
        @test haskey(extras, :true_hinges)
        @test length(extras.true_hinges) == B
        @test all(isfinite, extras.true_hinges)
        @test all(isfinite, (extras.accept_true, extras.reject_fake))

        grads2, loss2, extras2 = ReachabilityCascade.TrainingAPI.gradient(a, b, x_true, c;
                                                                         rng=rng,
                                                                         norm_kind=norm_kind,
                                                                         mode=:orthogonal_adv,
                                                                         return_loss=true,
                                                                         return_true_hinges=true,
                                                                         return_components=true)
        @test grads2 isa Tuple
        @test length(grads2) == 2
        @test haskey(grads2[1], :layers)
        @test haskey(grads2[2], :layers)
        @test isfinite(loss2)
        @test length(extras2.true_hinges) == B
        @test all(isfinite, extras2.true_hinges)
    end
end

@testset "InvertibleGame.inclusion_losses" begin
    rng = Random.MersenneTwister(100)
    D = 5
    C = 2
    N = 17

    model = ReachabilityCascade.InvertibleGame.InvertibleCoupling(D, C; rng=Random.MersenneTwister(101))
    data = [(; context=randn(rng, Float32, C), sample=randn(rng, Float32, D)) for _ in 1:N]

    for norm_kind in (:l1, :l2, :linf)
        losses = ReachabilityCascade.InvertibleGame.inclusion_losses(model, data; batch_size=4, norm_kind=norm_kind)
        @test losses isa Vector{Float32}
        @test length(losses) == N
        @test all(isfinite, losses)
    end
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

@testset "InvertibleGame.save_self/load_self" begin
    rng = Random.MersenneTwister(600)
    D = 6
    C = 3
    B = 4

    model = ReachabilityCascade.InvertibleGame.InvertibleCoupling(D, C; rng=rng)
    ema = deepcopy(model)
    x = randn(rng, Float32, D, B)
    c = randn(rng, Float32, C, B)
    z1 = ReachabilityCascade.InvertibleGame.encode(model, x, c)

    mktemp() do path, io
        close(io)
        ReachabilityCascade.InvertibleGame.save_self(path, model;
                                                     losses=Float32[1, 2],
                                                     ema=ema,
                                                     ema_beta_start=0.9,
                                                     ema_beta_final=0.9,
                                                     ema_tau=1.0,
                                                     ema_step=7)
        model2, meta = ReachabilityCascade.InvertibleGame.load_self(path)
        z2 = ReachabilityCascade.InvertibleGame.encode(model2, x, c)
        @test z2 ≈ z1 atol=1e-5
        @test meta.losses isa AbstractVector
        @test meta.ema isa ReachabilityCascade.InvertibleGame.InvertibleCoupling
        @test meta.ema_beta_start == 0.9
        @test meta.ema_beta_final == 0.9
        @test meta.ema_tau == 1.0
        @test meta.ema_step == 7
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
        model, ema, losses = ReachabilityCascade.TrainingAPI.build(
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

        model2, meta = ReachabilityCascade.InvertibleGame.load_self(path)
        x = randn(rng_data, Float32, D, 3)
        c = randn(rng_data, Float32, C, 3)
        @test ReachabilityCascade.InvertibleGame.encode(model2, x, c) ≈ ReachabilityCascade.InvertibleGame.encode(model, x, c) atol=1e-5
        @test meta.ema isa ReachabilityCascade.InvertibleGame.InvertibleCoupling
        @test ema isa ReachabilityCascade.InvertibleGame.InvertibleCoupling
        @test length(losses) > 0
    end
end

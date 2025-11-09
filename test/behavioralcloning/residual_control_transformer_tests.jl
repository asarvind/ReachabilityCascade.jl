using Test
using Random
using Flux
using ReachabilityCascade

const ResidualControlTransformer = ReachabilityCascade.ResidualControlTransformer
const sinusoidal_embedding = ReachabilityCascade.sinusoidal_embedding

@testset "Residual Control Transformer" begin
    control_dim = 2
    state_dim = 3
    context_dim = 4
    seq_len = 5
    rng = MersenneTwister(123)

    model = ResidualControlTransformer(control_dim, state_dim, context_dim;
                                       num_heads=4,
                                       ff_dim=64,
                                       num_layers=2,
                                       pos_dim=7,
                                       dropout=0.0)

    u_seq = randn(rng, control_dim, seq_len)
    x_seq = randn(rng, state_dim, seq_len)
    context = randn(rng, context_dim)

    @testset "forward pass shapes" begin
        delta = model(u_seq, x_seq, context)
        @test size(delta) == size(u_seq)
    end

    @testset "custom positional embedding" begin
        custom_pos = randn(rng, model.pos_dim, seq_len)
        delta = model(u_seq, x_seq, context; pos_embedding=custom_pos)
        @test size(delta) == size(u_seq)
    end

    @testset "gradient flow" begin
        loss_fn(u, x, ctx) = sum(abs2, model(u, x, ctx))
        ps = Flux.params(model)
        grads = gradient(() -> loss_fn(u_seq, x_seq, context), ps)
        nonzero = any(param -> any(!iszero, grads[param]), ps)
        @test nonzero
    end
end

using Test
using Random
using ReachabilityCascade

const SimpleSequenceTransformer = ReachabilityCascade.SimpleSequenceTransformer

@testset "Simple Sequence Transformer" begin
    seq_dim = 4
    context_dim = 3
    seq_len = 6
    rng = MersenneTwister(42)

    model = SimpleSequenceTransformer(seq_dim, context_dim;
                                      hidden_dim=16,
                                      num_heads=4,
                                      num_layers=2,
                                      pos_dim=8,
                                      max_period=5_000.0)

    seq = randn(rng, seq_dim, seq_len)
    context = randn(rng, context_dim)

    @testset "default positional encoding" begin
        out = model(seq, context)
        @test size(out) == size(seq)
    end

    @testset "custom positional encoding" begin
        custom_pos = randn(rng, model.pos_dim, seq_len)
        out = model(seq, context; pos_embedding=custom_pos)
        @test size(out) == size(seq)
    end

    @testset "batched inputs" begin
        batch = 3
        seq_batch = randn(rng, seq_dim, seq_len, batch)
        ctx_batch = randn(rng, context_dim, batch)
        out = model(seq_batch, ctx_batch)
        @test size(out) == size(seq_batch)
    end

    @testset "positional embedding function" begin
        pos_fn = pos -> fill(Float32(pos / seq_len), model.pos_dim)
        out = model(seq, context; pos_embedding=pos_fn)
        @test size(out) == size(seq)
    end
end

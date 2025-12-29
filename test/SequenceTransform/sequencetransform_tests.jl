using Test
using Flux
using ReachabilityCascade: AttentionFFN, SequenceTransformation

@testset "SequenceTransform Tests" begin
    
    @testset "AttentionFFN" begin
        in_dim = 3
        hidden_dim = 4
        out_dim = 5
        seq_len = 6
        batch_size = 2

        layer = AttentionFFN(in_dim, hidden_dim, out_dim; max_seq_len=seq_len, nheads=1, add_pos=true)
        x = rand(Float32, in_dim, seq_len, batch_size)
        y = layer(x)
        @test size(y) == (out_dim, seq_len, batch_size)

        # Test 2D input
        x_2d = rand(Float32, in_dim, seq_len)
        y_2d = layer(x_2d)
        @test size(y_2d) == (out_dim, seq_len)
    end

    @testset "SequenceTransformation" begin
        in_dim = 3
        hidden_dim = 4
        out_dim = 5
        depth = 3
        seq_len = 6
        batch_size = 2
        
        model = SequenceTransformation(in_dim, hidden_dim, out_dim, depth; max_seq_len=seq_len)
        x = rand(Float32, in_dim, seq_len, batch_size)
        
        y = model(x)
        
        @test size(y) == (out_dim, seq_len, batch_size)
        
        # Check depth (number of layers in chain)
        # model.chain is a Flux.Chain
        @test length(model.chain) == depth
    end

    @testset "Context Support" begin
        in_dim = 3
        hidden_dim = 4
        out_dim = 5
        context_dim = 2
        seq_len = 6
        batch_size = 2
        
        # Test SequenceTransformation with context
        # SequenceTransformation handles the concatenation
        depth = 2
        model = SequenceTransformation(in_dim, hidden_dim, out_dim, depth, context_dim; max_seq_len=seq_len)
        x = rand(Float32, in_dim, seq_len, batch_size)
        c = rand(Float32, context_dim, batch_size)
        y_chain = model(x, c)
        @test size(y_chain) == (out_dim, seq_len, batch_size)
        
        # Test 2D input with context
        x_2d = rand(Float32, in_dim, seq_len)
        c_1d = rand(Float32, context_dim)
        y_2d = model(x_2d, c_1d)
        @test size(y_2d) == (out_dim, seq_len)
    end
end

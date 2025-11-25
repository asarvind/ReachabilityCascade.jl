using Test
using Flux

# Include the module directly for testing purposes
include("../../src/SequenceTransform/SequenceTransform.jl")
using .SequenceTransform

@testset "SequenceTransform Tests" begin
    
    @testset "ForwardCumsumBlock" begin
        in_dim = 4
        out_dim = 5
        seq_len = 10
        batch_size = 2
        
        block = ForwardCumsumBlock(in_dim, out_dim)
        x = rand(Float32, in_dim, seq_len, batch_size)
        
        y = block(x)
        
        @test size(y) == (out_dim, seq_len, batch_size)
        
        # Test 2D input (unbatched)
        x_2d = rand(Float32, in_dim, seq_len)
        y_2d = block(x_2d)
        @test size(y_2d) == (out_dim, seq_len)

        # Manual verification
        dense_w = ones(Float32, 1, 1)
        dense_b = zeros(Float32, 1)
        simple_block = ForwardCumsumBlock(Dense(dense_w, dense_b, identity))
        
        x_simple = ones(Float32, 1, 5, 1) # 1 feature, 5 steps, 1 batch
        # Dense output: all 1s
        # Cumsum: 1, 2, 3, 4, 5
        # Average: 1/1, 2/2, 3/3, 4/4, 5/5 -> all 1s
        y_simple = simple_block(x_simple)
        @test y_simple[:] ≈ [1.0, 1.0, 1.0, 1.0, 1.0]
        
        # Another example: input [2, 4, 6]
        x_growing = reshape(Float32[2, 4, 6], 1, 3, 1)
        # Cumsum: [2, 6, 12]
        # Divisors: [1, 2, 3]
        # Average: [2, 3, 4]
        y_growing = simple_block(x_growing)
        @test y_growing[:] ≈ [2.0, 3.0, 4.0]
    end

    @testset "ReverseCumsumBlock" begin
        in_dim = 4
        out_dim = 5
        seq_len = 10
        batch_size = 2
        
        block = ReverseCumsumBlock(in_dim, out_dim)
        x = rand(Float32, in_dim, seq_len, batch_size)
        
        y = block(x)
        
        @test size(y) == (out_dim, seq_len, batch_size)

        # Test 2D input
        x_2d = rand(Float32, in_dim, seq_len)
        y_2d = block(x_2d)
        @test size(y_2d) == (out_dim, seq_len)
        
        # Manual verification
        dense_w = ones(Float32, 1, 1)
        dense_b = zeros(Float32, 1)
        simple_block = ReverseCumsumBlock(Dense(dense_w, dense_b, identity))
        
        x_simple = ones(Float32, 1, 5, 1)
        # Dense output: all 1s
        # Reverse Cumsum: 5, 4, 3, 2, 1
        # Divisors: 5, 4, 3, 2, 1
        # Average: 1, 1, 1, 1, 1
        y_simple = simple_block(x_simple)
        @test y_simple[:] ≈ [1.0, 1.0, 1.0, 1.0, 1.0]
        
        # Another example: input [6, 4, 2]
        x_growing = reshape(Float32[6, 4, 2], 1, 3, 1)
        # Reverse Cumsum: [12, 6, 2]
        # Divisors: [3, 2, 1]
        # Average: [4, 3, 2]
        y_growing = simple_block(x_growing)
        @test y_growing[:] ≈ [4.0, 3.0, 2.0]
    end

    @testset "ScanMixer" begin
        in_dim = 3
        hidden_dim = 4
        out_dim = 5
        seq_len = 6
        batch_size = 2
        
        layer = ScanMixer(in_dim, hidden_dim, out_dim)
        x = rand(Float32, in_dim, seq_len, batch_size)
        
        y = layer(x)
        
        @test size(y) == (out_dim, seq_len, batch_size)

        # Test 2D input
        x_2d = rand(Float32, in_dim, seq_len)
        y_2d = layer(x_2d)
        @test size(y_2d) == (out_dim, seq_len)
        
        # Check differentiability
        loss(x) = sum(layer(x))
        grads = gradient(loss, x)
        @test grads[1] !== nothing
    end

    @testset "SequenceTransformation" begin
        in_dim = 3
        hidden_dim = 4
        out_dim = 5
        depth = 3
        seq_len = 6
        batch_size = 2
        
        model = SequenceTransformation(in_dim, hidden_dim, out_dim, depth)
        x = rand(Float32, in_dim, seq_len, batch_size)
        
        y = model(x)
        
        @test size(y) == (out_dim, seq_len, batch_size)
        
        # Check differentiability
        loss(x) = sum(model(x))
        grads = gradient(loss, x)
        @test grads[1] !== nothing
        
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
        
        # Test ScanMixer with context
        # ScanMixer now expects the concatenated input if context_dim > 0
        # But ScanMixer constructor no longer takes context_dim.
        # We initialize it with total input dimension (in_dim + context_dim).
        layer = ScanMixer(in_dim + context_dim, hidden_dim, out_dim)
        
        x = rand(Float32, in_dim, seq_len, batch_size)
        c = rand(Float32, context_dim, batch_size)
        
        # Manually concatenate for ScanMixer test
        c_reshaped = reshape(c, size(c, 1), 1, size(c, 2))
        c_repeated = repeat(c_reshaped, 1, seq_len, 1)
        x_in = cat(x, c_repeated, dims=1)
        
        y = layer(x_in)
        @test size(y) == (out_dim, seq_len, batch_size)
        
        # Test SequenceTransformation with context
        # SequenceTransformation handles the concatenation
        depth = 2
        model = SequenceTransformation(in_dim, hidden_dim, out_dim, depth, context_dim)
        y_chain = model(x, c)
        @test size(y_chain) == (out_dim, seq_len, batch_size)
        
        # Test 2D input with context
        x_2d = rand(Float32, in_dim, seq_len)
        c_1d = rand(Float32, context_dim)
        y_2d = model(x_2d, c_1d)
        @test size(y_2d) == (out_dim, seq_len)
    end
end

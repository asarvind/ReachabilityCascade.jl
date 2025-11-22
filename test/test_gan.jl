using Test
using ReachabilityCascade
using Flux
using Random

@testset "GAN Tests" begin
    x_dim = 5
    ctx_dim = 3
    z_dim = 4
    batch_size = 10

    gan = GAN(x_dim, ctx_dim, z_dim; hidden=32, n_layers=2)

    @testset "Initialization" begin
        @test gan.x_dim == x_dim
        @test gan.ctx_dim == ctx_dim
        @test gan.z_dim == z_dim
        @test gan.encoder isa Encoder
        @test gan.decoder isa Decoder
    end

    @testset "Encoder" begin
        x = randn(Float32, x_dim, batch_size)
        c = randn(Float32, ctx_dim, batch_size)
        
        z = encode(gan, x, c)
        
        @test size(z) == (z_dim, batch_size)
        @test all(z .>= -1.0f0)
        @test all(z .<= 1.0f0)
        
        # Test single vector input
        x_vec = randn(Float32, x_dim)
        c_vec = randn(Float32, ctx_dim)
        z_vec = encode(gan, x_vec, c_vec)
        @test size(z_vec) == (z_dim,)
    end

    @testset "Decoder" begin
        z = rand(Float32, z_dim, batch_size) .* 2 .- 1 # Uniform in [-1, 1]
        c = randn(Float32, ctx_dim, batch_size)
        
        x_rec = decode(gan, z, c)
        
        @test size(x_rec) == (x_dim, batch_size)
        
        # Test single vector input
        z_vec = rand(Float32, z_dim) .* 2 .- 1
        c_vec = randn(Float32, ctx_dim)
        x_rec_vec = decode(gan, z_vec, c_vec)
        @test size(x_rec_vec) == (x_dim,)
    end
    
    @testset "Flux Trainable" begin
        params = Flux.trainable(gan)
        @test length(params) > 0
    end

    @testset "Configurable Activation" begin
        # Test default activation (leakyrelu)
        gan_default = GAN(x_dim, ctx_dim, z_dim; hidden=32, n_layers=2)
        # We can't easily check the function itself in the Chain without inspecting internals, 
        # but we can check if it runs.
        z = encode(gan_default, randn(Float32, x_dim, 1), randn(Float32, ctx_dim, 1))
        @test size(z) == (z_dim, 1)

        # Test custom activation (sigmoid)
        gan_sigmoid = GAN(x_dim, ctx_dim, z_dim; hidden=32, n_layers=2, activation=sigmoid)
        z_sig = encode(gan_sigmoid, randn(Float32, x_dim, 1), randn(Float32, ctx_dim, 1))
        @test size(z_sig) == (z_dim, 1)
        
        # Verify that the activation is actually stored (by checking the first layer of encoder)
        first_layer = gan_sigmoid.encoder.model[1]
        @test first_layer.σ == sigmoid
    end

    @testset "Gradient Computation" begin
        gan = GAN(x_dim, ctx_dim, z_dim; hidden=32, n_layers=2)
        
        x_true = randn(Float32, x_dim, batch_size)
        c_true = randn(Float32, ctx_dim, batch_size)
        z_gen = rand(Float32, z_dim, batch_size) .* 2 .- 1
        c_gen = randn(Float32, ctx_dim, batch_size)
        
        grads = compute_gradients(gan, x_true, c_true, z_gen, c_gen)
        
        @test haskey(grads, :encoder)
        @test haskey(grads, :decoder)
        
        # Check that we have gradients for parameters
        # Flux.gradient(model) do m ... end returns a structure matching the model (NamedTuple/Tuple)
        # We need to check if this structure contains non-zero values.
        
        function has_nonzero_grad(g)
            if g === nothing
                return false
            elseif g isa AbstractArray
                return any(x -> abs(x) > 0, g)
            elseif g isa Tuple || g isa NamedTuple
                return any(has_nonzero_grad, g)
            else
                return false
            end
        end
        
        @test has_nonzero_grad(grads.encoder)
        @test has_nonzero_grad(grads.decoder)
    end
end

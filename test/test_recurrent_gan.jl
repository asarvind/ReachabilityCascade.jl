using Test
using ReachabilityCascade
using Flux

@testset "RecurrentGAN Instantiation" begin
    input_dim = 10
    context_dim = 5
    latent_dim = 4
    hidden_dim = 20
    
    rgan = RecurrentGAN(input_dim, context_dim, latent_dim, hidden_dim)
    
    @test rgan isa RecurrentGAN
    # Check Generator structure
    @test rgan.encoder isa Chain
    @test rgan.decoder isa Chain
    
    println("RecurrentGAN instantiated successfully.")
    
    # Test encode
    # Input: sample and context
    sample_dim = input_dim
    sample = randn(sample_dim)
    context = randn(context_dim)
    # Initial state
    h0 = zeros(Float32, hidden_dim)
    
    z, h_enc = encode(rgan, sample, context, h0)
    # Output should be latent_dim
    @test size(z) == (latent_dim, 1)
    # Check tanh clamping
    @test all(abs.(z) .<= 1.0f0 + 1e-5)
    # Check hidden state
    @test size(h_enc) == (hidden_dim, 1)
    
    # Test batch encode
    batch_size = 3
    sample_batch = randn(sample_dim, batch_size)
    context_batch = randn(context_dim, batch_size)
    h0_batch = zeros(Float32, hidden_dim, batch_size)
    
    z_batch, h_enc_batch = encode(rgan, sample_batch, context_batch, h0_batch)
    @test size(z_batch) == (latent_dim, batch_size)
    @test all(abs.(z_batch) .<= 1.0f0 + 1e-5)
    @test size(h_enc_batch) == (hidden_dim, batch_size)
    
    # Test decode
    # Input: z and context
    decoded, h_dec = decode(rgan, z, context, h0)
    @test size(decoded) == (sample_dim, 1)
    @test size(h_dec) == (hidden_dim, 1)
    
    # Test batch decode
    decoded_batch, h_dec_batch = decode(rgan, z_batch, context_batch, h0_batch)
    @test size(decoded_batch) == (sample_dim, batch_size)
    @test size(h_dec_batch) == (hidden_dim, batch_size)
    
    println("Adversarial Autoencoder with Explicit State and Tuple Return verified.")
end

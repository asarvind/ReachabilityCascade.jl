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
    @test rgan.encoder isa Encoder
    @test rgan.decoder isa Decoder
    
    println("RecurrentGAN instantiated successfully.")
    
    # Test encode
    # Input: sample and context
    sample_dim = input_dim
    sample = randn(Float32, sample_dim)
    context = randn(Float32, context_dim)
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
    sample_batch = randn(Float32, sample_dim, batch_size)
    context_batch = randn(Float32, context_dim, batch_size)
    h0_batch = zeros(Float32, hidden_dim, batch_size)
    
    z_batch, h_enc_batch = encode(rgan, sample_batch, context_batch, h0_batch)
    @test size(z_batch) == (latent_dim, batch_size)
    @test all(abs.(z_batch) .<= 1.0f0 + 1e-5)
    @test size(h_enc_batch) == (hidden_dim, batch_size)

    # Iteration override
    z_once, _ = encode(rgan, sample, context, h0; iterations=1)
    @test size(z_once) == (latent_dim, 1)
    @test_throws ArgumentError encode(rgan, sample, context, h0; iterations=0)

    # Shape validation
    bad_h0 = zeros(Float32, hidden_dim - 1)
    @test_throws ArgumentError encode(rgan, sample, context, bad_h0)
    bad_sample = randn(Float32, sample_dim - 1)
    @test_throws ArgumentError encode(rgan, bad_sample, context, h0)
    
    # Test decode
    # Input: z and context
    decoded, h_dec = decode(rgan, z, context, h0)
    @test size(decoded) == (sample_dim, 1)
    @test size(h_dec) == (hidden_dim, 1)
    
    # Test batch decode
    decoded_batch, h_dec_batch = decode(rgan, z_batch, context_batch, h0_batch)
    @test size(decoded_batch) == (sample_dim, batch_size)
    @test size(h_dec_batch) == (hidden_dim, batch_size)

    decoded_once, _ = decode(rgan, z, context, h0; iterations=1)
    @test size(decoded_once) == (sample_dim, 1)

    bad_z = randn(Float32, latent_dim - 1)
    @test_throws ArgumentError decode(rgan, bad_z, context, h0)
    @test_throws ArgumentError decode(rgan, z, context, h0; iterations=0)
    
    # Parameter registration sanity
    ps = Flux.trainable(rgan)
    @test ps.encoder.output_layer.weight === rgan.encoder.output_layer.weight
    @test ps.decoder.output_layer.weight === rgan.decoder.output_layer.weight
    
    # Determinism with fixed inputs
    z_again, h_enc_again = encode(rgan, sample, context, h0)
    @test z_again == z
    @test h_enc_again == h_enc
    
    println("Adversarial Autoencoder with Explicit State and Tuple Return verified.")
end

@testset "Reconstruction Losses" begin
    input_dim = 6
    context_dim = 3
    latent_dim = 4
    hidden_dim = 8

    rgan = RecurrentGAN(input_dim, context_dim, latent_dim, hidden_dim)

    sample = randn(Float32, input_dim, 2)
    context = randn(Float32, context_dim, 2)
    latents = randn(Float32, latent_dim, 2)
    h0_enc = zeros(Float32, hidden_dim, 2)
    h0_dec = zeros(Float32, hidden_dim, 2)

    l_enc = encoder_reconstruction_loss(rgan, sample, context; h_enc0=h0_enc, h_dec0=h0_dec, iterations_enc=2, iterations_dec=2)
    @test l_enc isa Real

    l_dec = decoder_reconstruction_loss(rgan, latents, context; h_dec0=h0_dec, h_enc0=h0_enc, iterations_dec=2, iterations_enc=2)
    @test l_dec isa Real
end

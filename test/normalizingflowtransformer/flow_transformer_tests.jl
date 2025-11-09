using Test
using ReachabilityCascade

@testset "Normalizing Flow Transformer" begin
    d_model = 6
    seq_len = 5
    context_dim = 3
    batch = 4

    max_seq_len = 7
    flow = FlowTransformer(d_model, context_dim;
                           num_layers=3,
                           num_heads=1,
                           ff_hidden=16,
                           coupling=:affine,
                           max_seq_len=max_seq_len)
    @test flow.max_seq_len == max_seq_len
    @test flow.position_dim == 1
    @test size(flow.position_table) == (flow.position_dim, flow.max_seq_len)

    x = rand(Float32, d_model, seq_len, batch)
    context = rand(Float32, context_dim, batch)

    latent, logdet = flow(x, context)

    @test size(latent) == (d_model, seq_len, batch)
    @test size(logdet) == (batch,)
    @test all(isfinite, logdet)

    recon = flow(latent, context; inverse=true)
    @test maximum(abs.(x .- recon)) < 1f-4

    add_flow = FlowTransformer(d_model, context_dim;
                               num_layers=2,
                               num_heads=1,
                               ff_hidden=12,
                               coupling=:additive,
                               max_seq_len=max_seq_len)
    @test add_flow.max_seq_len == max_seq_len
    @test add_flow.position_dim == 1
    latent_add, logdet_add = add_flow(x, context)
    @test all(abs.(logdet_add) .< 1f-5)

    recon_add = add_flow(latent_add, context; inverse=true)
    @test maximum(abs.(x .- recon_add)) < 1f-4

    no_norm_flow = FlowTransformer(d_model, context_dim;
                                   num_layers=2,
                                   num_heads=1,
                                   ff_hidden=12,
                                   coupling=:affine,
                                   use_layernorm=false,
                                   max_seq_len=max_seq_len)
    @test no_norm_flow.max_seq_len == max_seq_len
    @test no_norm_flow.position_dim == 1
    latent_no_norm, logdet_no_norm = no_norm_flow(x, context)
    @test size(latent_no_norm) == size(x)
    @test all(isfinite, logdet_no_norm)
    recon_no_norm = no_norm_flow(latent_no_norm, context; inverse=true)
    @test maximum(abs.(x .- recon_no_norm)) < 2f-4

    seq_len_long = 7
    x_long = rand(Float32, d_model, seq_len_long, batch)
    latent_long, logdet_long = flow(x_long, context)
    @test size(latent_long) == (d_model, seq_len_long, batch)
    @test size(logdet_long) == (batch,)
    recon_long = flow(latent_long, context; inverse=true)
    @test maximum(abs.(x_long .- recon_long)) < 1f-4

    custom_flow = FlowTransformer(d_model, context_dim;
                                  num_layers=2,
                                  num_heads=1,
                                  ff_hidden=12,
                                  coupling=:affine,
                                  max_seq_len=max_seq_len,
                                  position_fn = (pos, max_len) -> Float32[sin(pos), cos(pos)])
    @test custom_flow.position_dim == 2
    @test size(custom_flow.position_table) == (2, max_seq_len)
    latent_custom, logdet_custom = custom_flow(x, context)
    @test size(latent_custom) == size(x)
    recon_custom = custom_flow(latent_custom, context; inverse=true)
    @test maximum(abs.(x .- recon_custom)) < 1f-4

    @testset "Input shape normalization" begin
        flow_small = FlowTransformer(2, context_dim;
                                     num_layers=2,
                                     num_heads=1,
                                     ff_hidden=8,
                                     coupling=:affine,
                                     max_seq_len=seq_len)
        @test flow_small.max_seq_len == seq_len
        @test flow_small.position_dim == 1

        x_matrix = rand(Float32, 2, seq_len)
        ctx_vec = rand(Float32, context_dim)
        latent_matrix, logdet_matrix = flow_small(x_matrix, ctx_vec)
        @test size(latent_matrix) == size(x_matrix)
        @test length(logdet_matrix) == 1
        recon_matrix = flow_small(latent_matrix, ctx_vec; inverse=true)
        @test maximum(abs.(x_matrix .- recon_matrix)) < 1f-4

        x_vector = rand(Float32, 2)
        latent_vector, logdet_vector = flow_small(x_vector, ctx_vec)
        @test size(latent_vector) == size(x_vector)
        @test length(logdet_vector) == 1
        recon_vector = flow_small(latent_vector, ctx_vec; inverse=true)
        @test maximum(abs.(x_vector .- recon_vector)) < 1f-4
    end

end

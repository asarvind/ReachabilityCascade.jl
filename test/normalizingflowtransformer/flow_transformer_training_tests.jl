using Test
using Flux
using JLD2
using ReachabilityCascade

@testset "Flow Transformer Training" begin
    d_model = 4
    seq_len = 3
    context_dim = 2
    samples = [(rand(Float32, context_dim), rand(Float32, d_model, seq_len)) for _ in 1:4]

    model = train!(FlowTransformer,
                   samples,
                   d_model,
                   context_dim;
                   num_layers=1,
                   num_heads=1,
                   ff_hidden=8,
                   batch_size=2,
                   epochs=2,
                   max_seq_len=seq_len,
                   optimizer=Flux.Optimise.Descent(5e-3))

    @test model isa FlowTransformer

    ctx, seq = samples[1]
    latent, logdet = model(reshape(seq, d_model, seq_len, 1), reshape(ctx, context_dim, 1))
    @test size(latent) == (d_model, seq_len, 1)
    @test length(logdet) == 1

    custom_loss(latent, logdet) = sum(latent .^ 2) / length(latent) + sum(abs, logdet) / length(logdet)
    model_auto = train!(FlowTransformer,
                        samples;
                        num_layers=1,
                        num_heads=1,
                        ff_hidden=8,
                        batch_size=2,
                        epochs=1,
                        max_seq_len=seq_len)
    @test model_auto isa FlowTransformer

    model2 = train!(FlowTransformer,
                    samples,
                    d_model,
                    context_dim;
                    num_layers=1,
                    num_heads=1,
                    ff_hidden=8,
                    epochs=1,
                    optimizer=Flux.Optimise.Descent(1e-3),
                    loss=custom_loss,
                    batch_size=3,
                    max_seq_len=seq_len)
    @test model2 isa FlowTransformer

    varlen_samples = [(rand(Float32, context_dim), rand(Float32, d_model, len))
                      for len in (2, 3, 5, 4, 2)]
    model_varlen = train!(FlowTransformer,
                          varlen_samples,
                          d_model,
                          context_dim;
                          num_layers=1,
                          num_heads=1,
                          ff_hidden=8,
                          epochs=1,
                          batch_size=3,
                          optimizer=Flux.Optimise.Descent(1e-3),
                          max_seq_len=5)
    @test model_varlen isa FlowTransformer

    mktemp() do path, io
        close(io)
        trained = train!(FlowTransformer,
                         samples,
                         d_model,
                         context_dim;
                         num_layers=1,
                         num_heads=1,
                         ff_hidden=8,
                         epochs=1,
                         batch_size=2,
                         max_seq_len=seq_len,
                         save_path=path,
                         save_interval=0.0)
        @test ispath(path)
        saved = JLD2.jldopen(path, "r") do file
            Dict("model_state" => read(file, "model_state"),
                 "constructor_args" => read(file, "constructor_args"),
                 "constructor_kwargs" => read(file, "constructor_kwargs"),
                 "inferred_args" => read(file, "inferred_args"),
                 "inferred_kwargs" => read(file, "inferred_kwargs"),
                 "position_table" => read(file, "position_table"))
        end
        @test saved["constructor_args"] == (d_model, context_dim)
        expected_kwargs = Dict(:max_seq_len => seq_len,
                               :num_layers => 1,
                               :num_heads => 1,
                               :ff_hidden => 8,
                               :position_table => saved["position_table"])
        saved_kwargs_dict = Dict(pairs(saved["constructor_kwargs"]))
        inferred_kwargs_dict = Dict(pairs(saved["inferred_kwargs"]))
        @test saved_kwargs_dict == expected_kwargs
        @test saved["inferred_args"] == (d_model, context_dim)
        @test inferred_kwargs_dict == expected_kwargs
        @test size(saved["position_table"]) == (1, seq_len)
        @test trained.position_table == saved["position_table"]
        loaded = load_flow_transformer(path)
        sample_ctx, sample_seq = samples[1]
        sample_ctx = reshape(sample_ctx, context_dim, 1)
        sample_seq = reshape(sample_seq, d_model, seq_len, 1)
        orig_latent, _ = trained(sample_seq, sample_ctx)
        loaded_latent, _ = loaded(sample_seq, sample_ctx)
        @test orig_latent ≈ loaded_latent atol=1f-6
        @test loaded.position_table == saved["position_table"]

        retrained = train!(FlowTransformer,
                           samples;
                           load_path=path,
                           epochs=1,
                           batch_size=2,
                           max_seq_len=seq_len)
        @test retrained isa FlowTransformer

        rm(path, force=true)
    end
end

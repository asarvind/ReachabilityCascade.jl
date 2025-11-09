using Test
using Flux
using ReachabilityCascade

@testset "Flow Transformer Gradients" begin
    d_model = 4
    seq_len = 3
    context_dim = 2
    batch = 2

    flow = FlowTransformer(d_model, context_dim;
                           num_layers=2,
                           num_heads=1,
                           ff_hidden=12,
                           max_seq_len=seq_len)
    @test flow.position_dim == 1
    x = rand(Float32, d_model, seq_len, batch)
    context = rand(Float32, context_dim, batch)

    grads_default = flow_transformer_gradient(flow, x, context)
    flat_params, re = Flux.destructure(flow)
    flat_grad_default, _ = Flux.destructure(grads_default)
    @test length(flat_grad_default) == length(flat_params)
    step = re(flat_params .- 1f-3 .* flat_grad_default)
    @test typeof(step) === typeof(flow)

    loss_fn = (latent, logdet) -> sum(latent.^2) / length(latent) + sum(logdet) / length(logdet)
    grads = flow_transformer_gradient(flow, x, context; loss_fn=loss_fn)

    opt_state = Flux.setup(Flux.Optimise.Descent(1e-3), flow)
    Flux.update!(opt_state, flow, grads)
end

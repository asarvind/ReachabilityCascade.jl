using Random
using ReachabilityCascade

const NF = ReachabilityCascade.NormalizingFlow

let rng = Random.MersenneTwister(42)
    x_dim, ctx_dim = 4, 2
    flow = NF.ConditionalFlow(x_dim, ctx_dim; n_blocks=2, hidden=32, n_glu=2)

    x = randn(rng, Float32, x_dim, 3)
    c = randn(rng, Float32, ctx_dim, 3)

    enc = flow(x, c)
    decoded = flow(enc.latent, c; inverse=true)

    @assert maximum(abs.(decoded .- x)) < 1f-4
    println("Conditional flow roundtrip logdet sample: ", enc.logdet)

    # Recurrent composition example
    base_ctx_dim = 2
    time_embed_dim = 2
    recur_flow = NF.ConditionalFlow(x_dim, base_ctx_dim + time_embed_dim; n_blocks=2, hidden=32, n_glu=2)
    rcf = NF.RecurrentConditionalFlow(recur_flow, base_ctx_dim)

    base_ctx = randn(rng, Float32, base_ctx_dim, 3)
    recur_result = NF.encode_recurrent(rcf, x, base_ctx, 4)
    xr = NF.decode_recurrent(rcf, recur_result.latent, base_ctx, 4)

    @assert maximum(abs.(xr .- x)) < 1f-4
    println("Recurrent flow total logdet: ", recur_result.logdet)
end

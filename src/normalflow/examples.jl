module ConditionalFlowExamples

using Random
import ..ConditionalFlow
import ..RecurrentConditionalFlow
import ..encode
import ..decode
import ..encode_recurrent
import ..decode_recurrent

"""
    roundtrip(; seed=42)

Construct a small conditional flow, encode a batch, and decode it again to
illustrate the forward/inverse API. Returns a named tuple with intermediate
results.
"""
function roundtrip(; seed::Integer=42)
    rng = Random.MersenneTwister(seed)
    flow = ConditionalFlow(4, 2; n_blocks=2, hidden=32, n_glu=2)
    x = randn(rng, Float32, flow.x_dim, 3)
    c = randn(rng, Float32, flow.ctx_dim, 3)

    encoded = flow(x, c)
    reconstructed = decode(flow, encoded.latent, c)

    return (
        flow = flow,
        input = x,
        context = c,
        latent = encoded.latent,
        logdet = encoded.logdet,
        reconstruction = reconstructed,
        max_reconstruction_error = maximum(abs.(reconstructed .- x))
    )
end

"""
    roundtrip_scaled(; seed=24, scaling=Float32[1.5, 0.75, 1.2, 0.9])

Same as `roundtrip`, but uses a non-uniform `x_scaling` to demonstrate a
non-zero log-determinant without any training.
"""
function roundtrip_scaled(; seed::Integer=24,
                          scaling::AbstractVector{<:Real}=Float32[1.5, 0.75, 1.2, 0.9])
    flow = ConditionalFlow(length(scaling), 2;
                           n_blocks=2,
                           hidden=32,
                           n_glu=2,
                           x_scaling=Float32.(scaling))
    rng = Random.MersenneTwister(seed)
    x = randn(rng, Float32, flow.x_dim, 2)
    c = randn(rng, Float32, flow.ctx_dim, 2)

    encoded = flow(x, c)
    reconstructed = decode(flow, encoded.latent, c)

    return (
        flow = flow,
        input = x,
        context = c,
        latent = encoded.latent,
        logdet = encoded.logdet,
        reconstruction = reconstructed,
        max_reconstruction_error = maximum(abs.(reconstructed .- x))
    )
end

"""
    single_sample(; seed=7)

Encode a single `x`/`c` pair to demonstrate the 1D input handling.
"""
function single_sample(; seed::Integer=7)
    rng = Random.MersenneTwister(seed)
    flow = ConditionalFlow(3, 1; n_blocks=1, hidden=16, n_glu=1)
    x = randn(rng, Float32, flow.x_dim)
    c = randn(rng, Float32, flow.ctx_dim)

    encoded = encode(flow, x, c)
    return (
        flow = flow,
        input = x,
        context = c,
        latent = encoded.latent,
        logdet = encoded.logdet
    )
end

"""
    recurrent_roundtrip(; seed=99, steps=4)

Compose a recurrent conditional flow multiple times and verify that decoding
with the same number of steps reconstructs the input.
"""
function recurrent_roundtrip(; seed::Integer=99, steps::Integer=4)
    rng = Random.MersenneTwister(seed)
    base_ctx_dim = 2
    time_embed_dim = 2
    flow = ConditionalFlow(3, base_ctx_dim + time_embed_dim; n_blocks=2, hidden=32, n_glu=2)
    rcf = RecurrentConditionalFlow(flow, base_ctx_dim)
    x = randn(rng, Float32, flow.x_dim, 2)
    c = randn(rng, Float32, base_ctx_dim, 2)

    enc = encode_recurrent(rcf, x, c, steps)
    xr = decode_recurrent(rcf, enc.latent, c, steps)

    return (
        flow = flow,
        recurrent_flow = rcf,
        input = x,
        base_context = c,
        latent = enc.latent,
        per_step_latents = enc.per_step_latents,
        per_step_logdets = enc.per_step_logdets,
        reconstruction = xr,
        max_reconstruction_error = maximum(abs.(xr .- x)),
        logdet = enc.logdet,
        transitions = enc.transitions
    )
end

end # module ConditionalFlowExamples

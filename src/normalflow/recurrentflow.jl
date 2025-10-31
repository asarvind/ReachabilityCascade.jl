# ===================== Recurrent Conditional Flow ============================

"""
    RecurrentConditionalFlow(flow::ConditionalFlow, base_ctx_dim::Integer)

Wrap a `ConditionalFlow` so it can be composed with itself multiple times. At
each recursion the same flow is reapplied, but the context is augmented with a
fixed sinusoidal step embedding if the wrapped flow exposes additional context
dimensions beyond `base_ctx_dim`. When no extra context is available, the flow
operates without step embeddings.
"""
struct RecurrentConditionalFlow{F}
    flow::F
    base_ctx_dim::Int
    time_embed_dim::Int
end

Flux.@layer RecurrentConditionalFlow

"""
    RecurrentConditionalFlow(flow::ConditionalFlow, base_ctx_dim::Integer)

Construct a wrapper around an existing flow. The time-embedding dimension is
inferred from `flow.ctx_dim - base_ctx_dim`. When the remainder is zero, no
positional embedding is applied; otherwise sinusoidal features are used.
"""
function RecurrentConditionalFlow(flow::ConditionalFlow, base_ctx_dim::Integer)
    @assert 0 < base_ctx_dim && base_ctx_dim <= flow.ctx_dim "base_ctx_dim must be positive and at most flow.ctx_dim"
    embed_dim = flow.ctx_dim - base_ctx_dim
    return RecurrentConditionalFlow(flow, base_ctx_dim, embed_dim)
end

"""
    RecurrentConditionalFlow(x_dim::Integer, base_ctx_dim::Integer, time_embed_dim::Integer;
                             kwargs...)

Convenience constructor that builds the internal `ConditionalFlow` automatically.
The context dimension of the flow will be `base_ctx_dim + time_embed_dim`.
Any additional keyword arguments are forwarded to `ConditionalFlow`. Supply
`time_embed_dim = 0` to disable step embeddings entirely.
"""
function RecurrentConditionalFlow(x_dim::Integer,
                                  base_ctx_dim::Integer,
                                  time_embed_dim::Integer;
                                  kwargs...)
    flow = ConditionalFlow(x_dim, base_ctx_dim + time_embed_dim; kwargs...)
    return RecurrentConditionalFlow(flow, base_ctx_dim)
end

# -------------------------- Positional Embeddings ----------------------------

"""
    sinusoidal_time_embedding(embed_dim::Integer, step::Integer, total_steps::Integer)

Return a length-`embed_dim` Float32 vector using transformer-style sinusoidal
features. The step index is normalized to `[0, 1]`, so using the same
`total_steps` across encode/decode keeps the embedding consistent.
"""
function sinusoidal_time_embedding(embed_dim::Integer, step::Integer, total_steps::Integer)
    @assert embed_dim > 0 "Embedding dimension must be positive"
    @assert step >= 1 && step <= total_steps "step must be within 1..total_steps"
    pos = total_steps > 1 ? Float32(step - 1) / Float32(total_steps - 1) : 0.0f0
    half = fld(embed_dim, 2)
    scale = 1.0f0 ./ (10000.0f0 .^ ((0:half-1) ./ max(1, half)))
    vec = zeros(Float32, embed_dim)
    for (i, ang_base) in enumerate(scale)
        idx = 2*i - 1
        angle = pos * ang_base
        vec[idx] = sin(angle)
        if idx + 1 <= embed_dim
            vec[idx + 1] = cos(angle)
        end
    end
    if isodd(embed_dim)
        vec[end] = sin(pos)
    end
    return vec
end

# ------------------------------ Helper utils --------------------------------

function _ensure_batch(mat::AbstractMatrix, batch::Integer, label::AbstractString)
    if size(mat, 2) == 1 && batch > 1
        return repeat(mat, 1, batch)
    end
    @assert size(mat, 2) == batch "$label batch mismatch (expected $batch, got $(size(mat,2)))"
    return mat
end

function _augment_context(rcf::RecurrentConditionalFlow,
                          base_ctx,
                          step::Integer,
                          total_steps::Integer,
                          batch::Integer)
    base_mat = _ensure_batch(_as_colmat(base_ctx), batch, "Base context")
    @assert size(base_mat, 1) == rcf.base_ctx_dim "Base context dimension mismatch"

    if rcf.time_embed_dim == 0
        return base_mat
    end

    embed_raw = sinusoidal_time_embedding(rcf.time_embed_dim, step, total_steps)
    embed_mat = _as_colmat(embed_raw)
    embed_mat = _ensure_batch(embed_mat, batch, "Time embedding")
    @assert size(embed_mat, 1) == rcf.time_embed_dim "Time embedding dimension mismatch"

    return vcat(base_mat, embed_mat)
end

# ------------------------------ Public API ----------------------------------

"""
    encode_recurrent(rcf::RecurrentConditionalFlow,
                     x::AbstractVecOrMat,
                     context,
                     steps::Integer;
                     total_steps::Integer=steps)

Apply the wrapped flow repeatedly to the same sample batch, composing the flow
`steps` times. The base context is augmented with a step index (if the flow was
configured with a time embedding). Returns the final latent along with per-step
latents and log-determinant contributions.
"""
function encode_recurrent(rcf::RecurrentConditionalFlow,
                          x::AbstractVecOrMat,
                          context,
                          steps::Integer;
                          total_steps::Integer=steps)
    transitions = encode_recurrent_transitions(rcf, x, context, steps; total_steps=total_steps)
    @assert steps > 0 "steps must be positive"
    @assert total_steps >= steps "total_steps must be at least steps"
    latents = Vector{Matrix{Float32}}(undef, steps)
    logdets = Vector{Vector{Float32}}(undef, steps)
    logdet_total = zeros(Float32, length(transitions[1].logdet))
    for (i, trans) in enumerate(transitions)
        latents[i] = trans.output
        logdets[i] = trans.logdet
        logdet_total .+= trans.logdet
    end
    return (latent=transitions[end].output,
            per_step_latents=latents,
            per_step_logdets=logdets,
            logdet=logdet_total,
            transitions=transitions)
end

"""
    decode_recurrent(rcf::RecurrentConditionalFlow,
                     z::AbstractVecOrMat,
                     context,
                     steps::Integer;
                     total_steps::Integer=steps)

Invert the recurrent composition by applying the wrapped flow in reverse order.
Assumes the same `steps` and `total_steps` used during encoding. Returns the
reconstructed sample matrix.
"""
function decode_recurrent(rcf::RecurrentConditionalFlow,
                          z::AbstractVecOrMat,
                          context,
                          steps::Integer;
                          total_steps::Integer=steps)
    @assert steps > 0 "steps must be positive"
    @assert total_steps >= steps "total_steps must be at least steps"
    z_mat = _as_colmat(z)
    @assert size(z_mat, 1) == rcf.flow.x_dim "Latent dimension mismatch"
    batch = size(z_mat, 2)
    base_ctx = _ensure_batch(_as_colmat(context), batch, "Base context")
    @assert size(base_ctx, 1) == rcf.base_ctx_dim "Base context dimension mismatch"

    current = z_mat
    for step in Iterators.reverse(1:steps)
        ctx_mat = _augment_context(rcf, base_ctx, step, total_steps, batch)
        current = decode(rcf.flow, current, ctx_mat)
    end
    return current
end

"""
    encode_recurrent_transitions(rcf::RecurrentConditionalFlow,
                                 x::AbstractVecOrMat,
                                 context,
                                 steps::Integer;
                                 total_steps::Integer=steps)

Return the sequence of intermediate transitions when composing the wrapped flow.
Each element is a named tuple containing the input to the step, the augmented
context used, the output latent, and the step-specific log-determinant. This is
useful for training schemes that optimize each transition separately while
detaching others.
"""
function encode_recurrent_transitions(rcf::RecurrentConditionalFlow,
                                      x::AbstractVecOrMat,
                                      context,
                                      steps::Integer;
                                      total_steps::Integer=steps)
    @assert steps > 0 "steps must be positive"
    @assert total_steps >= steps "total_steps must be at least steps"
    x_mat = _as_colmat(x)
    @assert size(x_mat, 1) == rcf.flow.x_dim "Sample dimension mismatch"
    batch = size(x_mat, 2)
    base_ctx = _ensure_batch(_as_colmat(context), batch, "Base context")
    @assert size(base_ctx, 1) == rcf.base_ctx_dim "Base context dimension mismatch"

    current = x_mat
    transitions = Vector{NamedTuple}(undef, steps)
    for step in 1:steps
        ctx_mat = _augment_context(rcf, base_ctx, step, total_steps, batch)
        out = encode(rcf.flow, current, ctx_mat)
        transitions[step] = (step=step,
                             input=current,
                             context=ctx_mat,
                             output=out.latent,
                             logdet=out.logdet)
        current = out.latent
    end
    return transitions
end

"""
    (rcf::RecurrentConditionalFlow)(x, context, steps; inverse=false, total_steps=steps)

Convenience call overload that dispatches to `encode_recurrent` (default) or
`decode_recurrent` when `inverse=true`, forwarding the `steps` and
`total_steps` keywords.
"""
function (rcf::RecurrentConditionalFlow)(x, context, steps::Integer;
                                         inverse::Bool=false,
                                         total_steps::Integer=steps)
    return inverse ?
        decode_recurrent(rcf, x, context, steps; total_steps=total_steps) :
        encode_recurrent(rcf, x, context, steps; total_steps=total_steps)
end

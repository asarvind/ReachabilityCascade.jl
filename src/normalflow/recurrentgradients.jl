# ================= Recurrent Flow Training Utilities ========================

using Flux

function recurrent_flow_gradients(rcf::RecurrentConditionalFlow,
                                  samples::AbstractVecOrMat,
                                  context::AbstractVecOrMat,
                                  steps::Integer;
                                  total_steps::Integer=steps,
                                  old_samples::Union{Nothing,AbstractVecOrMat}=nothing,
                                  old_context::Union{Nothing,AbstractVecOrMat}=nothing,
                                  num_lowest::Integer=0)
    @assert steps > 0 "steps must be positive"
    @assert total_steps >= steps "total_steps must be at least steps"
    @assert num_lowest >= 0 "num_lowest must be non-negative"
    if (old_samples === nothing) ⊻ (old_context === nothing)
        error("old_samples and old_context must either both be provided or both be nothing")
    end

    fresh_samples = _as_colmat(samples)
    fresh_context = _as_colmat(context)
    @assert size(fresh_samples, 1) == rcf.flow.x_dim
    @assert size(fresh_context, 1) == rcf.base_ctx_dim
    @assert size(fresh_samples, 2) == size(fresh_context, 2)

    all_samples = fresh_samples
    all_context = fresh_context
    if old_samples !== nothing
        old_samples_mat = _as_colmat(old_samples)
        old_context_mat = _as_colmat(old_context)
        @assert size(old_samples_mat, 1) == rcf.flow.x_dim
        @assert size(old_context_mat, 1) == rcf.base_ctx_dim
        @assert size(old_samples_mat, 2) == size(old_context_mat, 2)
        all_samples = hcat(all_samples, old_samples_mat)
        all_context = hcat(all_context, old_context_mat)
    end

    transitions = encode_recurrent_transitions(rcf, all_samples, all_context,
                                               steps; total_steps=total_steps)

    inputs = reduce(hcat, (transitions[i].input for i in 1:steps))
    contexts = reduce(hcat, (transitions[i].context for i in 1:steps))
    enc = encode(rcf.flow, inputs, contexts)
    total_columns = size(inputs, 2)
    ll = _loglikelihood_terms(enc.latent, enc.logdet)
    total_loss = -sum(ll) / total_columns

    grads = Flux.gradient(rcf.flow) do flow
        enc_grad = encode(flow, inputs, contexts)
        ll_grad = _loglikelihood_terms(enc_grad.latent, enc_grad.logdet)
        -sum(ll_grad) / total_columns
    end

    hard_examples = nothing
    if num_lowest > 0
        total_count = size(all_samples, 2)
        total_entries = total_count * steps
        if total_entries > 0
            k = min(num_lowest, total_entries)
            order = sortperm(ll)[1:k]
            sample_idx = ((order .- 1) .% total_count) .+ 1
            step_idx = ((order .- 1) .÷ total_count) .+ 1
            hard_examples = (
                samples = all_samples[:, sample_idx],
                context = all_context[:, sample_idx],
                step = step_idx,
                loglikelihood = ll[order]
            )
        end
    end

    return (loss=total_loss,
            grads=grads,
            transitions=transitions,
            hard_examples=hard_examples)
end

function _loglikelihood_terms(z::AbstractArray, logdet::AbstractVector)
    T = float(eltype(z))
    D = size(z, 1)
    norm_sq = sum(z .^ 2; dims=1)
    const_term = (T(D) / T(2)) * log(T(2pi))
    ll_prior = -T(0.5) .* vec(norm_sq) .- const_term
    return vec(logdet) .+ ll_prior
end

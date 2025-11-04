using Random
using LinearAlgebra: norm
using Flux: destructure, loadmodel!, state
using JLD2: load
using ..NormalizingFlow: _as_colmat, encode_recurrent, RecurrentConditionalFlow,
                               sinusoidal_time_embedding

function _rademacher!(rng::AbstractRNG, arr::AbstractArray{T}) where {T<:Real}
    samples = rand(rng, Bool, size(arr))
    pos = one(T)
    neg = -pos
    @inbounds for idx in eachindex(arr, samples)
        arr[idx] = samples[idx] ? pos : neg
    end
    return arr
end

"""
    train_recurrent_control_perturb!(net,
                                     terminal_data,
                                     intermediate_data,
                                     control_data;
                                     epochs=1,
                                     perturb_scale=1e-3,
                                     rng=Random.default_rng(),
                                     terminal_steps=net.terminal_recurrence,
                                     intermediate_steps=net.intermediate_recurrence,
                                     control_steps=net.control_recurrence,
                                     terminal_total_steps=terminal_steps,
                                     intermediate_total_steps=intermediate_steps,
                                     control_total_steps=control_steps,
                                     carryover_limit=0,
                                     callback=nothing,
                                     save_path::AbstractString="",
                                     load_path::AbstractString=save_path,
                                     save_interval::Real=60.0,
                                     constructor_info=nothing)

Derivative-free training loop for [`RecurrentControlNet`](@ref). The routine
walks through the supplied data in the same fashion as
[`train_recurrent_control!`](@ref), but replaces gradient descent with a
random-search update rule: for each datum we perturb the corresponding flow's
weights by Rademacher noise (each component independently ±`perturb_scale`),
evaluate the *true*
log-likelihood on that datum, and keep the perturbation only when the
log-likelihood improves. Set `carryover_limit > 0` to retain the lowest
log-likelihood samples and append them to the next datum (mirroring the
gradient-based trainer).

The optional `callback` is invoked after each datum is processed with
`callback(component::Symbol, result, epoch::Int)` where `result` is a named
tuple containing:
- `accepted::Bool`
- `previous_loglikelihood`
- `new_loglikelihood`
- `perturb_norm`

Checkpointing mirrors [`train_recurrent_control!`](@ref):
- `save_path` controls where checkpoints are written (disabled when empty).
- `load_path` optionally restores `model_state` before training.
- `save_interval` throttles periodic saves (seconds).
- `constructor_info` can store reconstruction metadata inside the checkpoint.

Returns the trained network `net`.
"""
function train_recurrent_control_perturb!(net::RecurrentControlNet,
                                          terminal_data,
                                          intermediate_data,
                                          control_data;
                                          epochs::Integer=1,
                                          perturb_scale::Real=1e-3,
                                          rng::AbstractRNG=Random.default_rng(),
                                          terminal_steps::Integer=net.terminal_recurrence,
                                          intermediate_steps::Integer=net.intermediate_recurrence,
                                          control_steps::Integer=net.control_recurrence,
                                          terminal_total_steps::Integer=terminal_steps,
                                          intermediate_total_steps::Integer=intermediate_steps,
                                          control_total_steps::Integer=control_steps,
                                          carryover_limit::Integer=0,
                                          callback::Union{Nothing,Function}=nothing,
                                          save_path::AbstractString="",
                                          load_path::AbstractString=save_path,
                                          save_interval::Real=60.0,
                                          constructor_info=nothing)
    epochs > 0 || throw(ArgumentError("epochs must be positive"))
    perturb_scale > 0 || throw(ArgumentError("perturb_scale must be positive"))
    carryover_limit >= 0 || throw(ArgumentError("carryover_limit must be non-negative"))
    save_interval >= 0 || throw(ArgumentError("save_interval must be non-negative"))

    terminal_iter, term_first = _stateful_with_first(terminal_data)
    intermediate_iter, inter_first = _stateful_with_first(intermediate_data)
    control_iter, ctrl_first = _stateful_with_first(control_data)

    term_args, _ = _unpack_training_datum(term_first)
    inter_args, _ = _unpack_training_datum(inter_first)
    ctrl_args, _ = _unpack_training_datum(ctrl_first)

    term_args = _flatten_terminal_args(term_args)
    inter_args = _flatten_intermediate_args(inter_args)
    ctrl_args = _flatten_control_args(ctrl_args)

    state_dim = size(term_args[1], 1)

    size(term_args[2], 1) == state_dim ||
        throw(ArgumentError("current state dimension in terminal data mismatch"))
    size(inter_args[1], 1) == state_dim ||
        throw(ArgumentError("intermediate sample dimension mismatch"))
    size(inter_args[2], 1) == state_dim ||
        throw(ArgumentError("current state dimension in intermediate data mismatch"))
    size(inter_args[3], 1) == state_dim ||
        throw(ArgumentError("terminal state dimension mismatch in intermediate data"))
    size(ctrl_args[2], 1) == state_dim ||
        throw(ArgumentError("current state dimension in control data mismatch"))
    size(ctrl_args[3], 1) == state_dim ||
        throw(ArgumentError("next state dimension in control data mismatch"))

    constructor_meta = constructor_info
    if load_path != "" && isfile(load_path)
        stored = load(load_path)
        if haskey(stored, "model_state")
            state_loaded = stored["model_state"]
            if _tree_finite(state_loaded)
                Flux.loadmodel!(net, state_loaded)
            else
                @warn "Skipping checkpoint load due to non-finite weights" load_path maxlog=1
            end
        end
        if constructor_meta === nothing && haskey(stored, "constructor")
            constructor_meta = stored["constructor"]
        end
    end

    should_save = save_path != ""
    if should_save
        mkpath(dirname(save_path))
    end
    last_save = time()

    for epoch in 1:epochs
        terminal_memory = nothing
        _reset_iterator!(terminal_iter)
        for datum in terminal_iter
            result, terminal_memory = _perturb_terminal!(net.terminal,
                                                         datum,
                                                         rng,
                                                         perturb_scale,
                                                         terminal_steps,
                                                         terminal_total_steps,
                                                         terminal_memory,
                                                         carryover_limit)
            if callback !== nothing
                callback(:terminal, result, epoch)
            end
            if should_save && (time() - last_save) >= save_interval
                last_save = _save_checkpoint(save_path, net, constructor_meta)
            end
        end

        intermediate_memory = nothing
        _reset_iterator!(intermediate_iter)
        for datum in intermediate_iter
            result, intermediate_memory = _perturb_intermediate!(net.intermediate,
                                                                  datum,
                                                                  rng,
                                                                  perturb_scale,
                                                                  intermediate_steps,
                                                                 intermediate_total_steps,
                                                                 intermediate_memory,
                                                                 carryover_limit)
            if callback !== nothing
                callback(:intermediate, result, epoch)
            end
            if should_save && (time() - last_save) >= save_interval
                last_save = _save_checkpoint(save_path, net, constructor_meta)
            end
        end

        control_memory = nothing
        _reset_iterator!(control_iter)
        for datum in control_iter
            result, control_memory = _perturb_control!(net.controller,
                                                       datum,
                                                       rng,
                                                       perturb_scale,
                                                       control_steps,
                                                         control_total_steps,
                                                         control_memory,
                                                         carryover_limit)
            if callback !== nothing
                callback(:control, result, epoch)
            end
            if should_save && (time() - last_save) >= save_interval
                last_save = _save_checkpoint(save_path, net, constructor_meta)
            end
        end
    end

    if should_save
        _save_checkpoint(save_path, net, constructor_meta)
    end

    return net
end

function _perturb_terminal!(rcf::RecurrentConditionalFlow,
                            datum,
                            rng::AbstractRNG,
                            perturb_scale::Real,
                            steps::Integer,
                            total_steps::Integer,
                            memory,
                            carryover_limit::Integer)
    args, _ = _unpack_training_datum(datum)
    samples, current, goals = _flatten_terminal_args(args)

    fresh_samples = _as_colmat(samples)
    fresh_count = size(fresh_samples, 2)
    fresh_context = vcat(_as_colmat(current), _as_colmat(goals))
    all_samples, all_context = _combine_with_memory(fresh_samples, fresh_context, memory)

    ll_vec_base = _recurrent_loglikelihood(rcf, all_samples, all_context, steps, total_steps)
    base_ll = _partitioned_loglikelihood(ll_vec_base, fresh_count)

    candidate_rcf, perturb_norm = _perturb_flow(rcf, rng, perturb_scale)
    ll_vec_candidate = _recurrent_loglikelihood(candidate_rcf, all_samples, all_context, steps, total_steps)
    candidate_ll = _partitioned_loglikelihood(ll_vec_candidate, fresh_count)

    accepted = false
    final_ll_vec = ll_vec_base
    new_ll = base_ll

    if candidate_ll > base_ll
        loadmodel!(rcf.flow, state(candidate_rcf.flow))
        accepted = true
        final_ll_vec = ll_vec_candidate
        new_ll = candidate_ll
    end

    new_memory = _select_hard_samples(all_samples, all_context, final_ll_vec, carryover_limit)

    result = (accepted=accepted,
              previous_loglikelihood=base_ll,
              new_loglikelihood=new_ll,
              perturb_norm=perturb_norm)
    return result, new_memory
end

function _perturb_intermediate!(rcf::RecurrentConditionalFlow,
                                datum,
                                rng::AbstractRNG,
                                perturb_scale::Real,
                                steps::Integer,
                                total_steps::Integer,
                                memory,
                                carryover_limit::Integer)
    args, _ = _unpack_training_datum(datum)
    samples, current, terminal, times, totals = _flatten_intermediate_args(args)

    fresh_samples = _as_colmat(samples)
    current_mat = _as_colmat(current)
    terminal_mat = _as_colmat(terminal)
    batch = size(fresh_samples, 2)
    fresh_count = batch
    steps_vec = times isa Integer ? fill(Int(times), batch) : Int.(collect(times))
    totals_vec = totals isa Integer ? fill(Int(totals), batch) : Int.(collect(totals))
    @assert length(steps_vec) == batch "time_step count mismatch"
    @assert length(totals_vec) == batch "total_time count mismatch"
    time_mat = _intermediate_time_embedding(rcf, steps_vec, totals_vec, batch)
    fresh_context = vcat(current_mat, terminal_mat, time_mat)

    all_samples, all_context = _combine_with_memory(fresh_samples, fresh_context, memory)

    ll_vec_base = _recurrent_loglikelihood(rcf, all_samples, all_context, steps, total_steps)
    base_ll = _partitioned_loglikelihood(ll_vec_base, fresh_count)

    candidate_rcf, perturb_norm = _perturb_flow(rcf, rng, perturb_scale)
    ll_vec_candidate = _recurrent_loglikelihood(candidate_rcf, all_samples, all_context, steps, total_steps)
    candidate_ll = _partitioned_loglikelihood(ll_vec_candidate, fresh_count)

    accepted = false
    final_ll_vec = ll_vec_base
    new_ll = base_ll

    if candidate_ll > base_ll
        loadmodel!(rcf.flow, state(candidate_rcf.flow))
        accepted = true
        final_ll_vec = ll_vec_candidate
        new_ll = candidate_ll
    end

    new_memory = _select_hard_samples(all_samples, all_context, final_ll_vec, carryover_limit)

    result = (accepted=accepted,
              previous_loglikelihood=base_ll,
              new_loglikelihood=new_ll,
              perturb_norm=perturb_norm)
    return result, new_memory
end

function _perturb_control!(rcf::RecurrentConditionalFlow,
                           datum,
                           rng::AbstractRNG,
                           perturb_scale::Real,
                           steps::Integer,
                           total_steps::Integer,
                           memory,
                           carryover_limit::Integer)
    args, _ = _unpack_training_datum(datum)
    samples, current, next_states = _flatten_control_args(args)

    fresh_samples = _as_colmat(samples)
    fresh_count = size(fresh_samples, 2)
    fresh_context = vcat(_as_colmat(current), _as_colmat(next_states))
    all_samples, all_context = _combine_with_memory(fresh_samples, fresh_context, memory)

    ll_vec_base = _recurrent_loglikelihood(rcf, all_samples, all_context, steps, total_steps)
    base_ll = _partitioned_loglikelihood(ll_vec_base, fresh_count)

    candidate_rcf, perturb_norm = _perturb_flow(rcf, rng, perturb_scale)
    ll_vec_candidate = _recurrent_loglikelihood(candidate_rcf, all_samples, all_context, steps, total_steps)
    candidate_ll = _partitioned_loglikelihood(ll_vec_candidate, fresh_count)

    accepted = false
    final_ll_vec = ll_vec_base
    new_ll = base_ll

    if candidate_ll > base_ll
        loadmodel!(rcf.flow, state(candidate_rcf.flow))
        accepted = true
        final_ll_vec = ll_vec_candidate
        new_ll = candidate_ll
    end

    new_memory = _select_hard_samples(all_samples, all_context, final_ll_vec, carryover_limit)

    result = (accepted=accepted,
              previous_loglikelihood=base_ll,
              new_loglikelihood=new_ll,
              perturb_norm=perturb_norm)
    return result, new_memory
end

function _perturb_flow(rcf::RecurrentConditionalFlow,
                       rng::AbstractRNG,
                       perturb_scale::Real)
    flat, reassemble = destructure(rcf.flow)
    noise = similar(flat)
    _rademacher!(rng, noise)
    noise .*= perturb_scale

    candidate_flow = reassemble(flat .+ noise)
    candidate_rcf = RecurrentConditionalFlow(candidate_flow, rcf.base_ctx_dim)
    return candidate_rcf, norm(noise)
end

function _recurrent_loglikelihood(rcf::RecurrentConditionalFlow,
                                  samples::AbstractMatrix,
                                  context::AbstractMatrix,
                                  steps::Integer,
                                  total_steps::Integer)
    enc = encode_recurrent(rcf, samples, context, steps; total_steps=total_steps)
    logdet_total = zeros(eltype(enc.per_step_logdets[1]), size(samples, 2))
    for ld in enc.per_step_logdets
        logdet_total .+= ld
    end
    return _gaussian_loglikelihood(enc.latent, logdet_total)
end

function _gaussian_loglikelihood(z::AbstractArray, logdet::AbstractVector)
    D = size(z, 1)
    T = float(eltype(z))
    norm_sq = sum(abs2, z; dims=1)
    const_term = (T(D) / T(2)) * log(T(2pi))
    ll_prior = -T(0.5) .* vec(norm_sq) .- const_term
    return vec(logdet) .+ ll_prior
end

function _combine_with_memory(fresh_samples::AbstractMatrix,
                              fresh_context::AbstractMatrix,
                              memory)
    if memory === nothing
        return fresh_samples, fresh_context
    end
    @assert size(memory.samples, 1) == size(fresh_samples, 1) "memory sample dimension mismatch"
    @assert size(memory.context, 1) == size(fresh_context, 1) "memory context dimension mismatch"
    all_samples = hcat(fresh_samples, memory.samples)
    all_context = hcat(fresh_context, memory.context)
    return all_samples, all_context
end

function _partitioned_loglikelihood(ll_vec::AbstractVector, fresh_count::Integer)
    total = zero(eltype(ll_vec))
    fresh_count > 0 && (total += sum(@view ll_vec[1:fresh_count]) / fresh_count)
    mem_count = length(ll_vec) - fresh_count
    mem_count > 0 && (total += sum(@view ll_vec[fresh_count + 1:end]) / mem_count)
    return total
end

function _select_hard_samples(samples::AbstractMatrix,
                              context::AbstractMatrix,
                              loglikelihood::AbstractVector,
                              carryover_limit::Integer)
    if carryover_limit <= 0 || isempty(loglikelihood)
        return nothing
    end
    k = min(carryover_limit, length(loglikelihood))
    order = sortperm(loglikelihood)[1:k]
    return (samples = copy(samples[:, order]),
            context = copy(context[:, order]))
end

function _intermediate_time_embedding(rcf::RecurrentConditionalFlow,
                                      steps_vec::AbstractVector{<:Integer},
                                      totals_vec::AbstractVector{<:Integer},
                                      batch::Integer)
    state_dim = rcf.flow.x_dim
    embed_dim = max(rcf.base_ctx_dim - 2 * state_dim, 0)
    T = eltype(rcf.flow.x_scaling)
    if embed_dim == 0
        return zeros(T, 0, batch)
    end
    emb = Matrix{T}(undef, embed_dim, batch)
    for (j, (step, total)) in enumerate(zip(steps_vec, totals_vec))
        @assert total > 0 "total_time must be positive"
        @assert 1 <= step <= total "time_step must be within [1, total_time]"
        vec = sinusoidal_time_embedding(embed_dim, step, total)
        emb[:, j] = convert.(T, vec)
    end
    return emb
end

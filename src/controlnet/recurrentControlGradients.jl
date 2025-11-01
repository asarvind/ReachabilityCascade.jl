using ..NormalizingFlow: _as_colmat, sinusoidal_time_embedding, RecurrentConditionalFlow
using ..NormalizingFlow: recurrent_flow_gradient

"""Construct terminal flow gradients for the provided batches."""
function terminal_flow_gradient(net::RecurrentControlNet,
                                terminal_samples::AbstractVecOrMat,
                                current_states::AbstractVecOrMat,
                                goals::AbstractVecOrMat;
                                steps::Integer=net.terminal_steps,
                                total_steps::Integer=steps,
                                old_samples::Union{Nothing,AbstractVecOrMat}=nothing,
                                old_context::Union{Nothing,AbstractVecOrMat}=nothing,
                                num_lowest::Integer=0)
    samples = _as_colmat(terminal_samples)
    cur = _as_colmat(current_states)
    goal = _as_colmat(goals)
    @assert size(samples, 1) == net.state_dim "terminal sample dimension mismatch"
    @assert size(cur, 1) == net.state_dim
    @assert size(goal, 1) == net.goal_dim
    @assert size(samples, 2) == size(cur, 2) == size(goal, 2) "terminal batch mismatch"

    context = vcat(cur, goal)

    memory_samples = nothing
    memory_context = nothing
    if old_samples !== nothing || old_context !== nothing
        @assert (old_samples === nothing) == (old_context === nothing) "old_samples and old_context must both be provided"
        memory_samples = _as_colmat(old_samples)
        memory_context = _as_colmat(old_context)
        @assert size(memory_samples, 1) == net.state_dim "old sample dimension mismatch"
        @assert size(memory_context, 1) == size(context, 1) "old context dimension mismatch"
        @assert size(memory_samples, 2) == size(memory_context, 2) "old sample/context batch mismatch"
    end

    result = recurrent_flow_gradient(net.terminal, samples, context, steps;
                                     total_steps=total_steps,
                                     old_samples=memory_samples,
                                     old_context=memory_context,
                                     num_lowest=num_lowest)
    return result
end

function intermediate_flow_gradient(net::RecurrentControlNet,
                                    intermediate_samples::AbstractVecOrMat,
                                    current_states::AbstractVecOrMat,
                                    terminal_states::AbstractVecOrMat,
                                    times::Union{Integer,AbstractVector{<:Integer}},
                                    total_times;
                                    steps::Integer=net.intermediate_steps,
                                    total_steps::Integer=steps,
                                    old_samples::Union{Nothing,AbstractVecOrMat}=nothing,
                                    old_context::Union{Nothing,AbstractVecOrMat}=nothing,
                                    num_lowest::Integer=0)
    samples = _as_colmat(intermediate_samples)
    cur = _as_colmat(current_states)
   term = _as_colmat(terminal_states)
    @assert size(samples, 1) == net.state_dim
    @assert size(cur, 1) == net.state_dim
    @assert size(term, 1) == net.state_dim
    @assert size(samples, 2) == size(cur, 2) == size(term, 2)

    steps_vec = times isa Integer ? fill(Int(times), size(samples, 2)) : Int.(collect(times))
    totals_vec = total_times isa Integer ? fill(Int(total_times), size(samples, 2)) : Int.(collect(total_times))
    @assert length(steps_vec) == size(samples, 2)
    @assert length(totals_vec) == size(samples, 2)
    time_mat = _time_embedding_matrix(net, steps_vec, totals_vec, size(samples, 2))
    context = vcat(cur, term, time_mat)

    memory_samples = nothing
    memory_context = nothing
    if old_samples !== nothing || old_context !== nothing
        @assert (old_samples === nothing) == (old_context === nothing) "old_samples and old_context must both be provided"
        memory_samples = _as_colmat(old_samples)
        memory_context = _as_colmat(old_context)
        @assert size(memory_samples, 1) == net.state_dim "old sample dimension mismatch"
        @assert size(memory_context, 1) == size(context, 1) "old context dimension mismatch"
        @assert size(memory_samples, 2) == size(memory_context, 2) "old sample/context batch mismatch"
    end

    result = recurrent_flow_gradient(net.intermediate, samples, context, steps;
                                   total_steps=total_steps,
                                   old_samples=memory_samples,
                                   old_context=memory_context,
                                   num_lowest=num_lowest)
    return result
end

function control_flow_gradient(net::RecurrentControlNet,
                               control_samples::AbstractVecOrMat,
                               current_states::AbstractVecOrMat,
                               next_states::AbstractVecOrMat;
                               steps::Integer=net.control_steps,
                               total_steps::Integer=steps,
                               old_samples::Union{Nothing,AbstractVecOrMat}=nothing,
                               old_context::Union{Nothing,AbstractVecOrMat}=nothing,
                               num_lowest::Integer=0)
    samples = _as_colmat(control_samples)
    cur = _as_colmat(current_states)
    nxt = _as_colmat(next_states)
    @assert size(samples, 1) == net.control_dim
    @assert size(cur, 1) == net.state_dim
    @assert size(nxt, 1) == net.state_dim
    @assert size(samples, 2) == size(cur, 2) == size(nxt, 2)

    context = vcat(cur, nxt)

    memory_samples = nothing
    memory_context = nothing
    if old_samples !== nothing || old_context !== nothing
        @assert (old_samples === nothing) == (old_context === nothing) "old_samples and old_context must both be provided"
        memory_samples = _as_colmat(old_samples)
        memory_context = _as_colmat(old_context)
        @assert size(memory_samples, 1) == net.control_dim "old sample dimension mismatch"
        @assert size(memory_context, 1) == size(context, 1) "old context dimension mismatch"
        @assert size(memory_samples, 2) == size(memory_context, 2) "old sample/context batch mismatch"
    end

    result = recurrent_flow_gradient(net.controller, samples, context, steps;
                                     total_steps=total_steps,
                                     old_samples=memory_samples,
                                     old_context=memory_context,
                                     num_lowest=num_lowest)
    return result
end

function _time_embedding_matrix(net::RecurrentControlNet,
                                steps_input,
                                totals_input,
                                batch::Integer)
    steps_vec = steps_input isa Integer ? fill(Int(steps_input), batch) : Int.(collect(steps_input))
    totals_vec = totals_input isa Integer ? fill(Int(totals_input), batch) : Int.(collect(totals_input))
    @assert length(steps_vec) == batch
    @assert length(totals_vec) == batch
    T = eltype(net.intermediate.flow.x_scaling)
    emb = Matrix{T}(undef, net.time_embed_dim, batch)
    for (j, (step, total)) in enumerate(zip(steps_vec, totals_vec))
        @assert total > 0 "total_time must be positive"
        @assert 1 <= step <= total "time_step must be within [1, total_time]"
        vec = sinusoidal_time_embedding(net.time_embed_dim, step, total)
        emb[:, j] = convert.(T, vec)
    end
    return emb
end

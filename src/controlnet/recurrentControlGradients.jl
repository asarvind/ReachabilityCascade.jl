using ..NormalizingFlow: _as_colmat, sinusoidal_time_embedding, RecurrentConditionalFlow
using ..NormalizingFlow: recurrent_flow_gradient

"""Construct terminal flow gradients for the provided batches."""
function terminal_flow_gradient(net::RecurrentControlNet,
                                terminal_samples::AbstractVecOrMat,
                                current_states::AbstractVecOrMat,
                                goals::AbstractVecOrMat;
                                steps::Integer=net.terminal_steps,
                                total_steps::Integer=steps,
                                old_terminal_samples::Union{Nothing,AbstractVecOrMat}=nothing,
                                old_current_states::Union{Nothing,AbstractVecOrMat}=nothing,
                                old_goals::Union{Nothing,AbstractVecOrMat}=nothing,
                                num_lowest::Integer=0)
    samples = _as_colmat(terminal_samples)
    cur = _as_colmat(current_states)
    goal = _as_colmat(goals)
    @assert size(samples, 1) == net.state_dim "terminal sample dimension mismatch"
    @assert size(cur, 1) == net.state_dim
    @assert size(goal, 1) == net.goal_dim
    @assert size(samples, 2) == size(cur, 2) == size(goal, 2) "terminal batch mismatch"

    context = vcat(cur, goal)

    old_samples = nothing
    old_context = nothing
    if old_terminal_samples !== nothing
        old_cur = _as_colmat(old_current_states)
        old_goal = _as_colmat(old_goals)
        old_samples = _as_colmat(old_terminal_samples)
        @assert size(old_samples, 1) == net.state_dim
        @assert size(old_cur, 1) == net.state_dim
        @assert size(old_goal, 1) == net.goal_dim
        @assert size(old_samples, 2) == size(old_cur, 2) == size(old_goal, 2)
        old_context = vcat(old_cur, old_goal)
    end

    result = recurrent_flow_gradient(net.terminal, samples, context, steps;
                                   total_steps=total_steps,
                                   old_samples=old_samples,
                                   old_context=old_context,
                                   num_lowest=num_lowest)
    return _augment_with_sorted_batches(result, (
        sorted_terminal_samples = samples,
        sorted_current_states = cur,
        sorted_goals = goal,
    ))
end

function intermediate_flow_gradient(net::RecurrentControlNet,
                                    intermediate_samples::AbstractVecOrMat,
                                    current_states::AbstractVecOrMat,
                                    terminal_states::AbstractVecOrMat,
                                    time_steps,
                                    total_times;
                                    steps::Integer=net.intermediate_steps,
                                    total_steps::Integer=steps,
                                    old_intermediate_samples::Union{Nothing,AbstractVecOrMat}=nothing,
                                    old_current_states::Union{Nothing,AbstractVecOrMat}=nothing,
                                    old_terminal_states::Union{Nothing,AbstractVecOrMat}=nothing,
                                    old_time_steps=nothing,
                                    old_total_times=nothing,
                                    num_lowest::Integer=0)
    samples = _as_colmat(intermediate_samples)
    cur = _as_colmat(current_states)
    term = _as_colmat(terminal_states)
    @assert size(samples, 1) == net.state_dim
    @assert size(cur, 1) == net.state_dim
    @assert size(term, 1) == net.state_dim
    @assert size(samples, 2) == size(cur, 2) == size(term, 2)

    steps_vec = time_steps isa Integer ? fill(Int(time_steps), size(samples, 2)) : collect(time_steps)
    totals_vec = total_times isa Integer ? fill(Int(total_times), size(samples, 2)) : collect(total_times)
    @assert length(steps_vec) == size(samples, 2)
    @assert length(totals_vec) == size(samples, 2)
    time_mat = _time_embedding_matrix(net, steps_vec, totals_vec, size(samples, 2))
    context = vcat(cur, term, time_mat)

    old_samples = nothing
    old_context = nothing
    if old_intermediate_samples !== nothing
        @assert old_time_steps !== nothing && old_total_times !== nothing "old time information required"
        old_cur = _as_colmat(old_current_states)
        old_term = _as_colmat(old_terminal_states)
        old_samples = _as_colmat(old_intermediate_samples)
        @assert size(old_samples, 1) == net.state_dim
        @assert size(old_cur, 1) == net.state_dim
        @assert size(old_term, 1) == net.state_dim
        @assert size(old_samples, 2) == size(old_cur, 2) == size(old_term, 2)
        old_steps_vec = old_time_steps isa Integer ? fill(Int(old_time_steps), size(old_samples, 2)) : collect(old_time_steps)
        old_totals_vec = old_total_times isa Integer ? fill(Int(old_total_times), size(old_samples, 2)) : collect(old_total_times)
        @assert length(old_steps_vec) == size(old_samples, 2)
        @assert length(old_totals_vec) == size(old_samples, 2)
        old_time_mat = _time_embedding_matrix(net, old_steps_vec, old_totals_vec, size(old_samples, 2))
        old_context = vcat(old_cur, old_term, old_time_mat)
    end

    result = recurrent_flow_gradient(net.intermediate, samples, context, steps;
                                   total_steps=total_steps,
                                   old_samples=old_samples,
                                   old_context=old_context,
                                   num_lowest=num_lowest)
    return _augment_with_sorted_batches(result, (
        sorted_intermediate_samples = samples,
        sorted_current_states = cur,
        sorted_terminal_states = term,
        sorted_time_steps = steps_vec,
        sorted_total_times = totals_vec,
    ))
end

function control_flow_gradient(net::RecurrentControlNet,
                               control_samples::AbstractVecOrMat,
                               current_states::AbstractVecOrMat,
                               next_states::AbstractVecOrMat;
                               steps::Integer=net.control_steps,
                               total_steps::Integer=steps,
                               old_control_samples::Union{Nothing,AbstractVecOrMat}=nothing,
                               old_current_states::Union{Nothing,AbstractVecOrMat}=nothing,
                               old_next_states::Union{Nothing,AbstractVecOrMat}=nothing,
                               num_lowest::Integer=0)
    samples = _as_colmat(control_samples)
    cur = _as_colmat(current_states)
    nxt = _as_colmat(next_states)
    @assert size(samples, 1) == net.control_dim
    @assert size(cur, 1) == net.state_dim
    @assert size(nxt, 1) == net.state_dim
    @assert size(samples, 2) == size(cur, 2) == size(nxt, 2)

    context = vcat(cur, nxt)

    old_samples = nothing
    old_context = nothing
    if old_control_samples !== nothing
        old_cur = _as_colmat(old_current_states)
        old_nxt = _as_colmat(old_next_states)
        old_samples = _as_colmat(old_control_samples)
        @assert size(old_samples, 1) == net.control_dim
        @assert size(old_cur, 1) == net.state_dim
        @assert size(old_nxt, 1) == net.state_dim
        @assert size(old_samples, 2) == size(old_cur, 2) == size(old_nxt, 2)
        old_context = vcat(old_cur, old_nxt)
    end

    result = recurrent_flow_gradient(net.controller, samples, context, steps;
                                   total_steps=total_steps,
                                   old_samples=old_samples,
                                   old_context=old_context,
                                   num_lowest=num_lowest)
    return _augment_with_sorted_batches(result, (
        sorted_control_samples = samples,
        sorted_current_states = cur,
        sorted_next_states = nxt,
    ))
end

function _augment_with_sorted_batches(result, data::NamedTuple)
    if result.sorted_indices === nothing
        return result
    end
    inds = result.sorted_indices
    ks = keys(data)
    vals = map(k -> _index_data(getproperty(data, k), inds), ks)
    sorted = NamedTuple{ks}(Tuple(vals))
    return merge(result, sorted)
end

_index_data(x::AbstractMatrix, inds) = x[:, inds]
_index_data(x::AbstractVector, inds) = x[inds]
_index_data(x::Number, inds) = fill(x, length(inds))

function _time_embedding_matrix(net::RecurrentControlNet,
                                steps_input,
                                totals_input,
                                batch::Integer)
    steps_vec = steps_input isa Integer ? fill(Int(steps_input), batch) : collect(steps_input)
    totals_vec = totals_input isa Integer ? fill(Int(totals_input), batch) : collect(totals_input)
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

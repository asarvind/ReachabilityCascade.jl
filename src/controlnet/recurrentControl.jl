"""
Collection of building blocks for recurrent-flow control architectures.

Each component reuses `RecurrentConditionalFlow` to model a different step in
reasoning about control: predicting the terminal state, predicting intermediate
states at arbitrary times, and finally predicting the control input required to
reach the next state.
"""

using ..NormalizingFlow: ConditionalFlow, RecurrentConditionalFlow,
                               encode_recurrent, decode_recurrent,
                               _as_colmat, sinusoidal_time_embedding

"""
    RecurrentControlNet(state_dim, goal_dim, control_dim; kwargs...)

Container holding three recurrent conditional flows:

- `terminal`: maps current state + goal → terminal state.
- `intermediate`: maps (current state, terminal state, time) → intermediate state.
- `controller`: maps (current state, next state) → control input.

Keyword arguments allow overriding the recurrent depth (`*_steps`), the time
embedding dimension, and the keyword arguments passed down to each underlying
`ConditionalFlow` constructor.
"""
struct RecurrentControlNet{TF,IF,CF}
    terminal::RecurrentConditionalFlow{TF}
    intermediate::RecurrentConditionalFlow{IF}
    controller::RecurrentConditionalFlow{CF}
    state_dim::Int
    goal_dim::Int
    control_dim::Int
    terminal_recurrence::Int
    intermediate_recurrence::Int
    control_recurrence::Int
    time_embed_dim::Int
end

function RecurrentControlNet(state_dim::Integer, goal_dim::Integer, control_dim::Integer;
                             recurrence_steps::Integer=4,
                             terminal_recurrence::Integer=recurrence_steps,
                             intermediate_recurrence::Integer=recurrence_steps,
                             control_recurrence::Integer=recurrence_steps,
                             time_embed_dim::Integer=4,
                             recurrence_kwargs::NamedTuple=NamedTuple(),
                             terminal_kwargs::NamedTuple=recurrence_kwargs,
                             intermediate_kwargs::NamedTuple=recurrence_kwargs,
                             control_kwargs::NamedTuple=recurrence_kwargs)
    time_embed_dim > 0 || throw(ArgumentError("time_embed_dim must be positive"))

    terminal_flow = RecurrentConditionalFlow(state_dim, state_dim + goal_dim, 0;
                                             terminal_kwargs...)
    intermediate_flow = RecurrentConditionalFlow(state_dim,
                                                 2 * state_dim + time_embed_dim,
                                                 0;
                                                 intermediate_kwargs...)
    controller_flow = RecurrentConditionalFlow(control_dim, 2 * state_dim, 0;
                                               control_kwargs...)

    return RecurrentControlNet(terminal_flow, intermediate_flow, controller_flow,
                               state_dim, goal_dim, control_dim,
                               terminal_recurrence, intermediate_recurrence,
                               control_recurrence, time_embed_dim)
end

"""
    predict_terminal_state(net, current_state, goal; kwargs...)

Decode the terminal state reached after applying `net.terminal` for the
requested number of recurrent steps. Optional `latent` lets you supply a custom
latent code; otherwise a zero latent is used. `total_steps` governs the
positional embedding scale and defaults to `steps`.
"""
function predict_terminal_state(net::RecurrentControlNet,
                                current_state::AbstractVecOrMat,
                                goal::AbstractVecOrMat;
                                latent=nothing,
                                steps::Integer=net.terminal_recurrence,
                                total_steps::Integer=steps)
    state = _as_colmat(current_state)
    goal_mat = _as_colmat(goal)
    @assert size(state, 1) == net.state_dim "current_state dimension mismatch"
    @assert size(goal_mat, 1) == net.goal_dim "goal dimension mismatch"
    @assert size(state, 2) == size(goal_mat, 2) "state/goal batch mismatch"

    context = vcat(state, goal_mat)
    latent_mat = _latent_matrix(net.terminal, latent, size(state, 2))
    return decode_recurrent(net.terminal, latent_mat, context, steps; total_steps=total_steps)
end

"""
    predict_state_at(net, current_state, terminal_state, time_step, total_time; kwargs...)

Decode the state expected at `time_step` (1-indexed) in a total horizon of
`total_time`. Time information is encoded with the same sinusoidal embedding
used by the recurrent flows. `latent`, `steps`, and `total_steps` behave like in
`predict_terminal_state`.
"""
function predict_state_at(net::RecurrentControlNet,
                          current_state::AbstractVecOrMat,
                          terminal_state::AbstractVecOrMat,
                          time_step::Integer,
                          total_time::Integer;
                          latent=nothing,
                          steps::Integer=net.intermediate_recurrence,
                          total_steps::Integer=steps)
    state = _as_colmat(current_state)
    term = _as_colmat(terminal_state)
    @assert size(state, 1) == net.state_dim "current_state dimension mismatch"
    @assert size(term, 1) == net.state_dim "terminal_state dimension mismatch"
    @assert size(state, 2) == size(term, 2) "state/terminal batch mismatch"
    @assert total_time > 0 "total_time must be positive"
    @assert 1 <= time_step <= total_time "time_step must be within [1, total_time]"

    emb = sinusoidal_time_embedding(net.time_embed_dim, time_step, total_time)
    time_mat = repeat(reshape(emb, :, 1), 1, size(state, 2))
    context = vcat(state, term, time_mat)
    latent_mat = _latent_matrix(net.intermediate, latent, size(state, 2))
    return decode_recurrent(net.intermediate, latent_mat, context, steps; total_steps=total_steps)
end

"""
    predict_control_input(net, current_state, next_state; kwargs...)

Decode the immediate control input required to progress from `current_state` to
`next_state` using the controller recurrent flow.
"""
function predict_control_input(net::RecurrentControlNet,
                               current_state::AbstractVecOrMat,
                               next_state::AbstractVecOrMat;
                               latent=nothing,
                               steps::Integer=net.control_recurrence,
                               total_steps::Integer=steps)
    state = _as_colmat(current_state)
    next = _as_colmat(next_state)
    @assert size(state, 1) == net.state_dim "current_state dimension mismatch"
    @assert size(next, 1) == net.state_dim "next_state dimension mismatch"
    @assert size(state, 2) == size(next, 2) "state/next batch mismatch"

    context = vcat(state, next)
    latent_mat = _latent_matrix(net.controller, latent, size(state, 2))
    return decode_recurrent(net.controller, latent_mat, context, steps; total_steps=total_steps)
end

"""
    predict_control(net, current_state, goal, time_step, total_time; kwargs...)

Convenience wrapper that chains the three components:
1. Predict terminal state.
2. Predict intermediate state at the desired time.
3. Predict the control input to reach that intermediate state.

Keyword arguments allow supplying separate latent codes and overriding the
recurrent depth for each submodule.
"""
function predict_control(net::RecurrentControlNet,
                         current_state::AbstractVecOrMat,
                         goal::AbstractVecOrMat,
                         time_step::Integer,
                         total_time::Integer;
                         latent_terminal=nothing,
                         latent_intermediate=nothing,
                         latent_control=nothing,
                         terminal_steps::Integer=net.terminal_recurrence,
                         intermediate_steps::Integer=net.intermediate_recurrence,
                         control_steps::Integer=net.control_recurrence)
    terminal_state = predict_terminal_state(net, current_state, goal;
                                            latent=latent_terminal,
                                            steps=terminal_steps,
                                            total_steps=terminal_steps)
    intermediate_state = predict_state_at(net, current_state, terminal_state,
                                          time_step, total_time;
                                          latent=latent_intermediate,
                                          steps=intermediate_steps,
                                          total_steps=intermediate_steps)
    control = predict_control_input(net, current_state, intermediate_state;
                                    latent=latent_control,
                                    steps=control_steps,
                                    total_steps=control_steps)
    return (terminal_state=terminal_state,
            intermediate_state=intermediate_state,
            control=control)
end

"""Internal helper to normalise latent inputs for the recurrent flows."""
function _latent_matrix(flow::RecurrentConditionalFlow,
                        latent, batch::Integer)
    target_T = eltype(flow.flow.x_scaling)
    if latent === nothing
        return zeros(target_T, flow.flow.x_dim, batch)
    else
        mat = _as_colmat(latent)
        @assert size(mat, 1) == flow.flow.x_dim "latent dimension mismatch"
        @assert size(mat, 2) == batch "latent batch mismatch"
        return convert.(target_T, mat)
    end
end

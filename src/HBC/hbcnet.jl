using Flux
using ..GANModels: Gan, generator_forward, discriminator_forward, encoder_forward

"""
    _as_matrix(x)

Ensure vectors are reshaped into `(features, batch)` form with a singleton batch axis.
"""
_as_matrix(x::AbstractVector) = reshape(x, :, 1)
_as_matrix(x::AbstractMatrix) = x

"""
    _prepare_inputs(args...)

Concatenate inputs along the feature axis after ensuring consistent batch dimensions.
"""
function _prepare_inputs(args...)
    mats = map(_as_matrix, args)
    ncols = size(mats[1], 2)
    @assert all(size(mat, 2) == ncols for mat in mats) "Batch dimension mismatch in GAN inputs"
    return Flux.f32(reduce(vcat, mats))
end

"""
    _select_start_levels(completion_time, n_levels)

Map each completion time to the coarsest dyadic horizon required by the intermediate stack.
"""
function _select_start_levels(completion_time, n_levels::Integer)
    times = vec(Float32.(completion_time))
    clamped = map(t -> max(t, 1f0), times)
    raw_levels = ceil.(Int, log2.(clamped)) .+ 1
    return clamp.(raw_levels, 1, n_levels)
end

"""
    HierarchicalBehaviorCloner(state_dim, goal_dim, latent_dim, control_dim; kwargs...)

Hierarchical controller made of three GAN stages:
1. `task_gan` predicts the terminal state and completion time for a goal.
2. `intermediate_gans` recursively refine intermediate states at dyadic time scales.
3. `control_gan` outputs the control to move toward the one-step target.
"""
struct HierarchicalBehaviorCloner{TG,IG,CG}
    task_gan::TG
    intermediate_gans::Vector{IG}
    control_gan::CG
    state_dim::Int
    goal_dim::Int
    latent_dim::Int
    control_dim::Int
end

Flux.@layer HierarchicalBehaviorCloner

function HierarchicalBehaviorCloner(state_dim::Integer,
                                    goal_dim::Integer,
                                    latent_dim::Integer,
                                    control_dim::Integer;
                                    n_intermediate::Integer=3,
                                    task_hidden::Integer=128,
                                    intermediate_hidden::Integer=128,
                                    control_hidden::Integer=128,
                                    task_glu::Integer=2,
                                    intermediate_glu::Integer=2,
                                    control_glu::Integer=2,
                                    task_gate=Flux.σ,
                                    intermediate_gate=Flux.σ,
                                    control_gate=Flux.σ)
    @assert n_intermediate ≥ 1 "At least one intermediate GAN is required"

    task_context_dim = state_dim + goal_dim
    task_output_dim = state_dim + 1
    task_gan = Gan(latent_dim, task_context_dim, task_output_dim;
                   gen_hidden=task_hidden,
                   disc_hidden=task_hidden,
                   enc_hidden=task_hidden,
                   n_glu_gen=task_glu,
                   n_glu_disc=task_glu,
                   n_glu_enc=task_glu,
                   gen_gate=task_gate,
                   disc_gate=task_gate,
                   enc_gate=task_gate,
                   generator_out=nothing,
                   discriminator_out=nothing,
                   encoder_saturation=tanh,
                   generator_zero_init=false,
                   discriminator_zero_init=false,
                   encoder_zero_init=false)

    intermediate_context_dim = 2 * state_dim
    intermediate_output_dim = state_dim
    intermediate_gans = [Gan(latent_dim, intermediate_context_dim, intermediate_output_dim;
                              gen_hidden=intermediate_hidden,
                              disc_hidden=intermediate_hidden,
                              enc_hidden=intermediate_hidden,
                              n_glu_gen=intermediate_glu,
                              n_glu_disc=intermediate_glu,
                              n_glu_enc=intermediate_glu,
                              gen_gate=intermediate_gate,
                              disc_gate=intermediate_gate,
                              enc_gate=intermediate_gate,
                              generator_out=nothing,
                              discriminator_out=nothing,
                              encoder_saturation=tanh,
                              generator_zero_init=false,
                              discriminator_zero_init=false,
                              encoder_zero_init=false)
                         for _ in 1:n_intermediate]

    control_context_dim = 2 * state_dim
    control_gan = Gan(latent_dim, control_context_dim, control_dim;
                      gen_hidden=control_hidden,
                      disc_hidden=control_hidden,
                      enc_hidden=control_hidden,
                      n_glu_gen=control_glu,
                      n_glu_disc=control_glu,
                      n_glu_enc=control_glu,
                      gen_gate=control_gate,
                      disc_gate=control_gate,
                      enc_gate=control_gate,
                      generator_out=nothing,
                      discriminator_out=nothing,
                      encoder_saturation=tanh,
                      generator_zero_init=false,
                      discriminator_zero_init=false,
                      encoder_zero_init=false)

    return HierarchicalBehaviorCloner(task_gan,
                                      intermediate_gans,
                                      control_gan,
                                      state_dim,
                                      goal_dim,
                                      latent_dim,
                                      control_dim)
end

"""
    task_forward(model, state, goal, latent)

Forward pass through the task-level GAN, returning `(terminal_state, completion_time)`.
The terminal state is produced as the current state plus a predicted delta.
"""
function task_forward(model::HierarchicalBehaviorCloner, state, goal, latent)
    state_f = Flux.f32(state)
    goal_f = Flux.f32(goal)
    latent_f = Flux.f32(latent)
    context = _prepare_inputs(state_f, goal_f)
    output = generator_forward(model.task_gan, context, latent_f)
    delta_state = output[1:model.state_dim, :]
    completion_time = output[model.state_dim + 1, :]
    terminal_state = state_f .+ delta_state
    return terminal_state, completion_time
end

"""
    intermediate_forward(model, state, terminal_state, completion_time, latent)

Propagate intermediate predictions from coarse to fine time scales, starting at the level
whose dyadic horizon covers the predicted completion time. Returns a vector where the
first element corresponds to the one-step target.
"""
function intermediate_forward(model::HierarchicalBehaviorCloner,
                              state,
                              terminal_state,
                              completion_time,
                              latent)
    state_f = Flux.f32(state)
    future_state = Flux.f32(terminal_state)
    latent_f = Flux.f32(latent)
    n_levels = length(model.intermediate_gans)
    start_levels = _select_start_levels(completion_time, n_levels)
    predictions = Vector{Matrix{Float32}}(undef, n_levels)
    for i in reverse(1:n_levels)
        active = findall(start_levels .>= i)
        if !isempty(active)
            current_state = state_f[:, active]
            context = _prepare_inputs(current_state, future_state[:, active])
            delta = generator_forward(model.intermediate_gans[i], context, latent_f[:, active])
            predicted = current_state .+ delta
            future_state[:, active] = predicted
        end
        predictions[i] = copy(future_state)
    end
    return predictions
end

"""
    intermediate_level_forward(model, level, current_state, target_state, latent)

Run a single intermediate GAN level using the provided target state from the next coarser scale.
Returns the predicted state for the previous level.
"""
function intermediate_level_forward(model::HierarchicalBehaviorCloner,
                                    level::Integer,
                                    current_state,
                                    target_state,
                                    latent)
    @assert 1 ≤ level ≤ length(model.intermediate_gans) "Level out of range"
    current_state_m = Flux.f32(_as_matrix(current_state))
    target_state_m = Flux.f32(_as_matrix(target_state))
    latent_m = Flux.f32(_as_matrix(latent))
    context = _prepare_inputs(current_state_m, target_state_m)
    delta = generator_forward(model.intermediate_gans[level], context, latent_m)
    return current_state_m .+ delta
end

"""
    control_forward(model, state, next_state, latent)

Forward pass through the control GAN to generate the control action.
"""
function control_forward(model::HierarchicalBehaviorCloner,
                         state,
                         next_state,
                         latent)
    context = _prepare_inputs(state, next_state)
    return generator_forward(model.control_gan, context, latent)
end

"""
    (model::HierarchicalBehaviorCloner)(state, goal, latent)

Run the complete hierarchy and return a named tuple containing the control and auxiliary predictions.
"""
function (model::HierarchicalBehaviorCloner)(state, goal, latent)
    state_m = Flux.f32(_as_matrix(state))
    goal_m = Flux.f32(_as_matrix(goal))
    latent_m = Flux.f32(_as_matrix(latent))
    terminal_state, completion_time = task_forward(model, state_m, goal_m, latent_m)
    intermediate_states = intermediate_forward(model, state_m, terminal_state, completion_time, latent_m)
    first_step_state = intermediate_states[1]
    control = control_forward(model, state_m, first_step_state, latent_m)
    return (control=control,
            terminal_state=terminal_state,
            completion_time=completion_time,
            intermediate_states=intermediate_states)
end

export HierarchicalBehaviorCloner,
       task_forward,
       intermediate_forward,
       intermediate_level_forward,
       control_forward

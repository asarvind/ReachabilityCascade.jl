using Flux
using Random
using ReachabilityCascade: Gan, generator_forward, discriminator_forward

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

    task_input_dim = state_dim + goal_dim + latent_dim
    task_output_dim = state_dim + 1
    task_gan = Gan(task_input_dim, task_output_dim;
                   gen_hidden=task_hidden,
                   disc_hidden=task_hidden,
                   n_glu_gen=task_glu,
                   n_glu_disc=task_glu,
                   gen_gate=task_gate,
                   disc_gate=task_gate,
                   generator_out=nothing,
                   discriminator_out=nothing,
                   generator_zero_init=false,
                   discriminator_zero_init=false)

    intermediate_input_dim = 2 * state_dim + latent_dim
    intermediate_output_dim = state_dim
    intermediate_gans = [Gan(intermediate_input_dim, intermediate_output_dim;
                              gen_hidden=intermediate_hidden,
                              disc_hidden=intermediate_hidden,
                              n_glu_gen=intermediate_glu,
                              n_glu_disc=intermediate_glu,
                              gen_gate=intermediate_gate,
                              disc_gate=intermediate_gate,
                              generator_out=nothing,
                              discriminator_out=nothing,
                              generator_zero_init=false,
                              discriminator_zero_init=false)
                         for _ in 1:n_intermediate]

    control_input_dim = 2 * state_dim + latent_dim
    control_gan = Gan(control_input_dim, control_dim;
                      gen_hidden=control_hidden,
                      disc_hidden=control_hidden,
                      n_glu_gen=control_glu,
                      n_glu_disc=control_glu,
                      gen_gate=control_gate,
                      disc_gate=control_gate,
                      generator_out=nothing,
                      discriminator_out=nothing,
                      generator_zero_init=false,
                      discriminator_zero_init=false)

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
"""
function task_forward(model::HierarchicalBehaviorCloner, state, goal, latent)
    input = _prepare_inputs(state, goal, latent)
    output = generator_forward(model.task_gan, input)
    terminal_state = output[1:model.state_dim, :]
    completion_time = output[model.state_dim + 1, :]
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
    n_levels = length(model.intermediate_gans)
    start_levels = _select_start_levels(completion_time, n_levels)
    predictions = Vector{Matrix{Float32}}(undef, n_levels)
    future_state = copy(terminal_state)
    for i in reverse(1:n_levels)
        active = findall(start_levels .>= i)
        if !isempty(active)
            input = _prepare_inputs(state[:, active], future_state[:, active], latent[:, active])
            predicted = generator_forward(model.intermediate_gans[i], input)
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
    current_state_m = _as_matrix(current_state)
    target_state_m = _as_matrix(target_state)
    latent_m = _as_matrix(latent)
    input = _prepare_inputs(current_state_m, target_state_m, latent_m)
    return generator_forward(model.intermediate_gans[level], input)
end

"""
    control_forward(model, state, next_state, latent)

Forward pass through the control GAN to generate the control action.
"""
function control_forward(model::HierarchicalBehaviorCloner,
                         state,
                         next_state,
                         latent)
    input = _prepare_inputs(state, next_state, latent)
    return generator_forward(model.control_gan, input)
end

"""
    (model::HierarchicalBehaviorCloner)(state, goal, latent)

Run the complete hierarchy and return a named tuple containing the control and auxiliary predictions.
"""
function (model::HierarchicalBehaviorCloner)(state, goal, latent)
    state_m = _as_matrix(state)
    goal_m = _as_matrix(goal)
    latent_m = _as_matrix(latent)
    terminal_state, completion_time = task_forward(model, state_m, goal_m, latent_m)
    intermediate_states = intermediate_forward(model, state_m, terminal_state, completion_time, latent_m)
    first_step_state = intermediate_states[1]
    control = control_forward(model, state_m, first_step_state, latent_m)
    return (control=control,
            terminal_state=terminal_state,
            completion_time=completion_time,
            intermediate_states=intermediate_states)
end

let latent_dim = 4, data_dim = 3, batch = 5
    gan = Gan(latent_dim, data_dim; gen_hidden=16, disc_hidden=16,
              generator_out=nothing, discriminator_out=nothing)

    z = randn(Float32, latent_dim, batch)
    x_fake = generator_forward(gan, z)
    @assert size(x_fake) == (data_dim, batch)

    disc_scores = discriminator_forward(gan, x_fake)
    @assert size(disc_scores, 2) == batch

    println("GAN forward example ran successfully with batch size ", batch)
end

let state_dim = 6, goal_dim = 3, latent_dim = 4, control_dim = 2, batch = 2
    model = HierarchicalBehaviorCloner(state_dim, goal_dim, latent_dim, control_dim;
                                       n_intermediate=3,
                                       task_hidden=32,
                                       intermediate_hidden=32,
                                       control_hidden=32)

    current_state = randn(Float32, state_dim, batch)
    goal = randn(Float32, goal_dim, batch)
    latent = randn(Float32, latent_dim, batch)

    result = model(current_state, goal, latent)
    single_level = intermediate_level_forward(model,
                                              length(model.intermediate_gans),
                                              current_state,
                                              result.terminal_state,
                                              latent)

    @assert size(result.control) == (control_dim, batch)
    @assert length(result.intermediate_states) == 3
    @assert all(size(s) == (state_dim, batch) for s in result.intermediate_states)
    @assert size(single_level) == (state_dim, batch)

    println("Hierarchical behavior cloner produced control with shape ", size(result.control))
end

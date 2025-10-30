module HBCExamples

using Random
import ..HierarchicalBehaviorCloner
import ..intermediate_level_forward

"""
    rollout(; seed=19, state_dim=6, goal_dim=3, latent_dim=4, control_dim=2,
            batch=2, n_intermediate=3)

Builds a `HierarchicalBehaviorCloner`, runs a forward pass with random inputs,
and returns the model along with its outputs and intermediate computations.
"""
function rollout(; seed::Integer=19,
                  state_dim::Integer=6,
                  goal_dim::Integer=3,
                  latent_dim::Integer=4,
                  control_dim::Integer=2,
                  batch::Integer=2,
                  n_intermediate::Integer=3)
    rng = Random.MersenneTwister(seed)
    model = HierarchicalBehaviorCloner(state_dim, goal_dim, latent_dim, control_dim;
                                       n_intermediate=n_intermediate,
                                       task_hidden=32,
                                       intermediate_hidden=32,
                                       control_hidden=32)

    current_state = randn(rng, Float32, state_dim, batch)
    goal = randn(rng, Float32, goal_dim, batch)
    latent = randn(rng, Float32, latent_dim, batch)

    rollout = model(current_state, goal, latent)
    deepest_level = length(model.intermediate_gans)
    single_level = intermediate_level_forward(model,
                                              deepest_level,
                                              current_state,
                                              rollout.terminal_state,
                                              latent)

    return (
        model = model,
        current_state = current_state,
        goal = goal,
        latent = latent,
        rollout = rollout,
        single_level = single_level
    )
end

end # module HBCExamples

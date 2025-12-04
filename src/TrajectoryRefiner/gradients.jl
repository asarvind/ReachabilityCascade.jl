using Flux
using Statistics

"""
    refinement_loss(network::RefinementModel, transition_fn, traj_cost_fn, mismatch_fn, sample::ShootingBundle, steps::Integer; imitation_weight::Real=1)

Compute a loss as the sum of a trajectory mismatch cost and a terminal cost after applying `steps` refinement iterations.

- `network`: The `RefinementModel`.
- `transition_fn`: Function `(x_prev_seq, u_seq) -> x_next_seq` that supports batched sequences.
- `traj_cost_fn`: Function `x -> cost` mapping guessed trajectories `(state_dim, seq_len+1, batch)` (including initial state) to a scalar or per-batch cost.
- `mismatch_fn`: Function `(x_res, x_guess) -> cost` comparing residual and guess trajectories.
- `sample`: `ShootingBundle` carrying `x_guess` (including the initial state), `u_guess`, and optional `x_target` (imitation trajectory over post-initial states).
- `steps`: Number of refinement iterations.
- `imitation_weight`: Weight applied to the imitation loss factor (default 1).
"""
function refinement_loss(network::RefinementModel, transition_fn, traj_cost_fn, mismatch_fn,
                         sample::ShootingBundle, steps::Integer;
                         imitation_weight::Real=1)
    # Apply refinement, then evaluate the residual and terminal cost on the final guess.
    refined = network(sample, transition_fn, traj_cost_fn, steps)
    x_res = rollout_guess(refined, transition_fn)
    x_body = selectdim(refined.x_guess, 2, 2:size(refined.x_guess, 2))
    traj_cost_raw = traj_cost_fn(refined.x_guess)
    traj_cost = traj_cost_raw isa Number ? traj_cost_raw : mean(traj_cost_raw)
    mismatch = mismatch_fn(x_res, x_body)
    objective_loss = mismatch + traj_cost

    # Optional imitation loss on the refined state trajectory.
    if refined.x_target === nothing
        return (loss=objective_loss, traj_cost=traj_cost, mismatch=mismatch, imitation=zero(objective_loss))
    else
        imitation_loss = mismatch_fn(x_body, refined.x_target)
        total_loss = objective_loss * (1 + imitation_weight * imitation_loss)
        return (loss=total_loss, traj_cost=traj_cost, mismatch=mismatch, imitation=imitation_loss)
    end
end

"""
    refinement_grads(network::RefinementModel, transition_fn, traj_cost_fn, mismatch_fn, sample::ShootingBundle, steps::Integer; imitation_weight::Real=1)

Compute refinement loss and parameter gradients for `network` over a `ShootingBundle`. Refinement is
unrolled for `steps`. Returns `(grads, loss)`, with `grads` targeting `network`.

- `network`: The `RefinementModel`.
- `transition_fn`: Function `(x_prev_seq, u_seq) -> x_next_seq` that supports batched sequences.
- `traj_cost_fn`: Function `x -> cost` mapping guessed trajectories `(state_dim, seq_len+1, batch)` (including initial state) to a scalar or per-batch cost.
- `mismatch_fn`: Function `(x_res, x_guess) -> cost` comparing residual and guess trajectories.
- `sample`: `ShootingBundle` carrying `x_guess` (with initial state), `u_guess`, and optional `x_target`.
- `steps`: Total refinement iterations (negative or zero yields the original `sample`).
- `imitation_weight`: Weight applied to the imitation loss factor (default 1).
"""
function refinement_grads(network::RefinementModel, transition_fn, traj_cost_fn, mismatch_fn,
                          sample::ShootingBundle, steps::Integer;
                          imitation_weight::Real=1)
    total_steps = max(steps, 0)
    metrics_ref = Ref{NamedTuple}()
    grads = Flux.gradient(network) do m
        metrics = refinement_loss(m, transition_fn, traj_cost_fn, mismatch_fn, sample, total_steps;
                                  imitation_weight=imitation_weight)
        metrics_ref[] = metrics
        metrics.loss
    end

    metrics = metrics_ref[]
    return grads[1], metrics
end

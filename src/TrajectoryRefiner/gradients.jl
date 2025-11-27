using Flux

"""
    refinement_loss(network, transition_fn, term_cost_fn, mismatch_fn, sample::ShootingBundle, steps::Integer; imitation_weight::Real=1)

Compute a loss as the sum of a trajectory mismatch cost and a terminal cost after applying `steps` refinement iterations.

- `network`: The `CorrectionNetwork`.
- `transition_fn`: Function `(x_prev_seq, u_seq) -> x_next_seq` that supports batched sequences.
- `term_cost_fn`: Function `x -> val` that supports batched terminal states; negative means satisfied and positive means violation.
- `mismatch_fn`: Function `(x_res, x_guess) -> cost` comparing residual and guess trajectories.
- `sample`: `ShootingBundle` carrying `x_guess` (including the initial state), `u_guess`, and optional `x_target` (imitation trajectory over post-initial states).
- `steps`: Number of refinement iterations.
- `imitation_weight`: Weight applied to the imitation loss factor (default 1).
"""
function refinement_loss(network, transition_fn, term_cost_fn, mismatch_fn,
                         sample::ShootingBundle, steps::Integer;
                         imitation_weight::Real=1)
    # Apply refinement, then evaluate the residual and terminal cost on the final guess.
    refined = network(sample, transition_fn, term_cost_fn, steps)
    x_res = _rollout_guess(refined, transition_fn)
    x_body = selectdim(refined.x_guess, 2, 2:size(refined.x_guess, 2))
    objective_loss = mismatch_fn(x_res, x_body) + term_cost_fn(selectdim(refined.x_guess, 2, size(refined.x_guess, 2)))

    # Optional imitation loss on the refined state trajectory.
    if refined.x_target === nothing
        return objective_loss
    else
        imitation_loss = mismatch_fn(x_body, refined.x_target)
        return objective_loss * (1 + imitation_weight * imitation_loss)
    end
end

"""
    refinement_grads(network, transition_fn, term_cost_fn, mismatch_fn, sample::ShootingBundle, steps::Integer; imitation_weight::Real=1)

Compute refinement loss and parameter gradients for `network` over a `ShootingBundle`. Refinement is
unrolled for `steps`. Returns `(grads, loss)`, with `grads` targeting `network`.

- `network`: The `CorrectionNetwork`.
- `transition_fn`: Function `(x_prev_seq, u_seq) -> x_next_seq` that supports batched sequences.
- `term_cost_fn`: Function `x -> val` that supports batched terminal states; negative means satisfied and positive means violation.
- `mismatch_fn`: Function `(x_res, x_guess) -> cost` comparing residual and guess trajectories.
- `sample`: `ShootingBundle` carrying `x_guess` (with initial state), `u_guess`, and optional `x_target`.
- `steps`: Total refinement iterations (negative or zero yields the original `sample`).
- `imitation_weight`: Weight applied to the imitation loss factor (default 1).
"""
function refinement_grads(network, transition_fn, term_cost_fn, mismatch_fn,
                          sample::ShootingBundle, steps::Integer;
                          imitation_weight::Real=1)
    total_steps = max(steps, 0)
    loss_ref = Ref{Float32}()
    grads = Flux.gradient(network) do m
        l = refinement_loss(m, transition_fn, term_cost_fn, mismatch_fn, sample, total_steps;
                            imitation_weight=imitation_weight)
        loss_ref[] = Float32(l)
        l
    end

    return grads[1], loss_ref[]
end

using Flux
using Statistics

# Softmax-weighted averaging; flattens the array, uses temperature for scaling.
_softmax_average(arr; temperature::Real=1.0) = begin
    vals = vec(arr)
    weights = Flux.softmax(vals ./ temperature)
    sum(vals .* weights)
end

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
                         imitation_weight::Real=1,
                         softmax_temperature::Real=1.0)
    # Apply refinement, then evaluate the residual and terminal cost on the final guess.
    refined = network(sample, transition_fn, traj_cost_fn, steps)
    x_res = rollout_guess(refined, transition_fn)
    x_body = refined.x_guess
    traj_cost_raw = traj_cost_fn(cat(refined.x0, refined.x_guess; dims=2))
    traj_cost = traj_cost_raw isa Number ? traj_cost_raw : _softmax_average(traj_cost_raw; temperature=softmax_temperature)

    mismatch_raw = mismatch_fn(x_res, x_body)
    mismatch = mismatch_raw isa Number ? mismatch_raw : _softmax_average(mismatch_raw; temperature=softmax_temperature)
    objective_loss = mismatch + traj_cost

    # Optional imitation loss on the refined state trajectory.
    if refined.x_target === nothing
        return (loss=objective_loss, traj_cost=traj_cost, mismatch=mismatch, imitation=zero(objective_loss))
    else
        imitation_raw = mismatch_fn(x_body, refined.x_target)
        imitation_loss = imitation_raw isa Number ? imitation_raw : _softmax_average(imitation_raw; temperature=softmax_temperature)
        total_loss = objective_loss * (1 + imitation_weight * imitation_loss)
        return (loss=total_loss, traj_cost=traj_cost, mismatch=mismatch, imitation=imitation_loss)
    end
end

"""
    refinement_grads(network::RefinementModel, transition_fn, traj_cost_fn, mismatch_fn,
                     sample::ShootingBundle, refine_steps::Integer, backprop_steps::Integer;
                     imitation_weight::Real=1, backprop_mode::Symbol=:tail)

Compute refinement loss and parameter gradients for `network` over a `ShootingBundle`. Refinement is
unrolled according to `backprop_mode`:

- `:tail` (default): unroll `refine_steps`, keep only the last `backprop_steps` on the tape (as before).
- `:min_loss`: evaluate loss at every refinement step, start backpropagation at the step with minimal
  loss (including that step) for up to `backprop_steps` remaining refinements (at least one step when
  `refine_steps > 0`).
`softmax_temperature` controls the softmax-weighted averaging used when scalarizing array-valued costs.

Returns `(grads, metrics)` with `grads` targeting `network` and `metrics` from the chosen window.
"""
function refinement_grads(network::RefinementModel, transition_fn, traj_cost_fn, mismatch_fn,
                          sample::ShootingBundle, refine_steps::Integer, backprop_steps::Integer;
                          imitation_weight::Real=1, backprop_mode::Symbol=:tail,
                          softmax_temperature::Real=1.0)
    total_steps = max(refine_steps, 0)
    max_bsteps = clamp(backprop_steps, 0, total_steps)

    start_bundle::ShootingBundle = sample
    steps_for_grad::Int = 0
    best_step::Int = 0  # 0-based index of the step with minimal loss (0 means pre-refinement)

    if backprop_mode == :tail
        fsteps = total_steps - max_bsteps
        start_bundle = fsteps > 0 ? network(sample, transition_fn, traj_cost_fn, fsteps) : sample
        steps_for_grad = max_bsteps
        best_step = total_steps > 0 ? total_steps - max_bsteps : 0
    elseif backprop_mode == :min_loss
        if total_steps == 0
            start_bundle = sample
            steps_for_grad = 0
            best_step = 0
        else
            bundles = Vector{ShootingBundle}(undef, total_steps + 1)
            losses = Vector{Float32}(undef, total_steps + 1)
            bundles[1] = sample
            losses[1] = Float32(refinement_loss(network, transition_fn, traj_cost_fn, mismatch_fn,
                                                sample, 0; imitation_weight=imitation_weight,
                                                softmax_temperature=softmax_temperature).loss)
            for s in 1:total_steps
                bundles[s + 1] = network(bundles[s], transition_fn, traj_cost_fn, 1)
                losses[s + 1] = Float32(refinement_loss(network, transition_fn, traj_cost_fn, mismatch_fn,
                                                        bundles[s + 1], 0; imitation_weight=imitation_weight,
                                                        softmax_temperature=softmax_temperature).loss)
            end
            _, best_idx = findmin(losses)
            best_step = best_idx - 1  # convert to 0-based refinement step count
            # Start backprop from the bundle immediately before the best step (if it exists),
            # so the best refinement is on the tape.
            start_idx = max(1, best_idx - 1)
            steps_done_before_start = start_idx - 1
            steps_available = total_steps - steps_done_before_start
            steps_for_grad = max_bsteps > 0 ? min(max_bsteps, steps_available) : steps_available
            steps_for_grad = max(1, steps_for_grad)
            start_bundle = bundles[start_idx]
        end
    else
        throw(ArgumentError("backprop_mode must be :tail or :min_loss"))
    end

    metrics_ref = Ref{NamedTuple}()
    grads = Flux.gradient(network) do m
        metrics = refinement_loss(m, transition_fn, traj_cost_fn, mismatch_fn, start_bundle, steps_for_grad;
                                  imitation_weight=imitation_weight,
                                  softmax_temperature=softmax_temperature)
        metrics_ref[] = metrics
        metrics.loss
    end

    metrics = metrics_ref[]
    return grads[1], merge(metrics, (best_step=best_step,))
end

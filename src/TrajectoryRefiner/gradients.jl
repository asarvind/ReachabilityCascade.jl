using Flux

"""
    refinement_grads(network, transition_fn, term_cost_fn, mismatch_fn, x_guess_init, u_guess_init, x_0, steps::Integer; backprop_steps::Integer=1, target::Union{Nothing,AbstractArray}=nothing, imitation_weight::Real=1)

Compute gradients of the correction `network` parameters for the refinement loss. Refinement is unrolled for `steps` iterations; only the last `backprop_steps` participate in backpropagation (earlier steps are outside the gradient tape), so `backprop_steps` defaults to 1. Returns the gradient object for `network`.

- `network`: The `CorrectionNetwork`.
- `transition_fn`: Function `(x_prev_seq, u_seq) -> x_next_seq` that supports batched sequences.
- `term_cost_fn`: Function `x -> val` that supports batched terminal states; negative means satisfied and positive means violation.
- `mismatch_fn`: Function `(x_res, x_guess) -> cost` comparing residual and guess trajectories.
- `x_guess_init`, `u_guess_init`: Initial state/input guesses (state_dim, seq_len, batch) and (input_dim, seq_len, batch).
- `x_0`: Initial state context (state_dim, batch).
- `steps`: Total refinement iterations.
- `backprop_steps`: Number of trailing steps kept in the gradient tape (default 1, clamped to `steps`).
- `target`: Optional target state trajectory for imitation learning.
- `imitation_weight`: Weight applied to the imitation loss factor (default 1).
"""
function refinement_grads(network, transition_fn, term_cost_fn, mismatch_fn,
                          x_guess_init::AbstractArray, u_guess_init::AbstractArray, x_0::AbstractArray, steps::Integer;
                          backprop_steps::Integer=1, target::Union{Nothing,AbstractArray}=nothing, imitation_weight::Real=1)
    total_steps = max(steps, 0)
    bsteps = clamp(backprop_steps, 0, total_steps)
    fsteps = total_steps - bsteps

    # Run truncated forward refinement; these steps are not part of the backward tape.
    if fsteps > 0
        x_fwd, u_fwd = refine(network, transition_fn, term_cost_fn, x_guess_init, u_guess_init, x_0, fsteps)
    else
        x_fwd = x_guess_init
        u_fwd = u_guess_init
    end

    grads = Flux.gradient(network) do m
        # Backprop only through the remaining bsteps of refinement.
        refinement_loss(m, transition_fn, term_cost_fn, mismatch_fn, x_fwd, u_fwd, x_0, bsteps;
                        target=target, imitation_weight=imitation_weight)
    end

    return grads[1]
end

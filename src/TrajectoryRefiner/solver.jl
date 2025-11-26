using Flux

"""
    refine(network, transition_fn, term_cost_fn, x_guess::AbstractArray, u_guess::AbstractArray, x_0::AbstractArray, steps::Integer=1)

Trajectory refinement using the correction network. Applies `steps` recursive refinement iterations (`steps <= 0` returns the inputs unchanged).

# Arguments
- `network`: The `CorrectionNetwork`.
- `transition_fn`: Function `(x_prev_seq, u_seq) -> x_next_seq` that supports batched sequences.
- `term_cost_fn`: Function `x -> val` that supports batched terminal states; negative means satisfied and positive means violation.
- `x_guess`: Current state trajectory guess. Shape: (state_dim, seq_len, batch)
- `u_guess`: Current input trajectory guess. Shape: (input_dim, seq_len, batch)
- `x_0`: Initial state context. Shape: (state_dim, batch)
- `steps`: Number of refinement iterations (default 1).

# Returns
- `x_new`: Refined state trajectory.
- `u_new`: Refined input trajectory.
"""
function refine(network, transition_fn, term_cost_fn, x_guess::AbstractArray, u_guess::AbstractArray, x_0::AbstractArray, steps::Integer=1)
    # Nothing to do if no refinement steps are requested.
    if steps <= 0
        return x_guess, u_guess
    end
    x_cur, u_cur = x_guess, u_guess
    for _ in 1:steps
        x_cur, u_cur = _refine_once(network, transition_fn, term_cost_fn, x_cur, u_cur, x_0)
    end
    return x_cur, u_cur
end

# Internal single-step refine used by the public API.
function _refine_once(network, transition_fn, term_cost_fn, x_guess::AbstractArray, u_guess::AbstractArray, x_0::AbstractArray)
    # Dimensions
    state_dim = size(x_guess, 1)
    seq_len = size(x_guess, 2)
    batch_size = size(x_guess, 3)
    
    # Step 1: Compute residuals (multiple shooting from x_0 and shifted x_guess).
    x_res = _compute_residual(x_guess, u_guess, x_0, transition_fn)
    
    # Step 2: Network Forward Pass
    delta_inter, out_term = network(x_res, x_guess, u_guess, x_0)
    
    # Step 3: Terminal constraint
    # Check violation at the last step of x_guess
    x_term = selectdim(x_guess, 2, seq_len) # (state_dim, batch)
    
    # term_cost_fn(x) -> val
    # Negative => satisfied, positive => violation.
    # We assume term_cost_fn handles batching or we map it.
    # Let's assume it returns (1, batch) or (batch,)
    term_val = term_cost_fn(x_term)
    
    # Violation: relu(val). If val > 0, violation is positive; if val < 0, zero.
    term_violation = Flux.relu.(term_val)
    
    # Reshape for broadcasting: (1, 1, batch)
    if ndims(term_violation) == 1
        term_violation = reshape(term_violation, 1, 1, length(term_violation))
    elseif ndims(term_violation) == 2
        # (1, batch) -> (1, 1, batch)
        term_violation = reshape(term_violation, 1, 1, size(term_violation, 2))
    end
    
    delta_term = term_violation .* out_term
    
    # Step 4: Update
    correction = delta_inter + delta_term
    
    # Split correction into state and input parts
    # Assuming correction output dim = state_dim + input_dim
    # and they are concatenated [delta_x; delta_u]
    delta_x = correction[1:state_dim, :, :]
    delta_u = correction[state_dim+1:end, :, :]
    
    x_new = x_guess + delta_x
    u_new = u_guess + delta_u
    
    return x_new, u_new
end

# Helper to compute multiple-shooting residuals given a guess and transition.
function _compute_residual(x_guess::AbstractArray, u_guess::AbstractArray, x_0::AbstractArray, transition_fn)
    state_dim = size(x_guess, 1)
    seq_len = size(x_guess, 2)
    batch_size = size(x_guess, 3)

    x_0_reshaped = reshape(x_0, state_dim, 1, batch_size)
    # Build previous-state sequence: [x_0, x_guess[:, 1:end-1, :]]
    x_prev_seq = seq_len > 1 ? cat(x_0_reshaped, selectdim(x_guess, 2, 1:seq_len-1), dims=2) : x_0_reshaped
    return transition_fn(x_prev_seq, u_guess)
end

"""
    refinement_loss(network, transition_fn, term_cost_fn, mismatch_fn, x_guess_init, u_guess_init, x_0, steps::Integer; target::Union{Nothing,AbstractArray}=nothing, imitation_weight::Real=1)

Compute a loss as the sum of a trajectory mismatch cost and a terminal cost after applying `steps` refinement iterations.

- `network`: The `CorrectionNetwork`.
- `transition_fn`: Function `(x_prev_seq, u_seq) -> x_next_seq` that supports batched sequences.
- `term_cost_fn`: Function `x -> val` that supports batched terminal states; negative means satisfied and positive means violation.
- `mismatch_fn`: Function `(x_res, x_guess) -> cost` comparing residual and guess trajectories.
- `x_guess_init`, `u_guess_init`: Initial state/input guesses (state_dim, seq_len, batch) and (input_dim, seq_len, batch).
- `x_0`: Initial state context (state_dim, batch).
- `steps`: Number of refinement iterations.
- `target`: Optional target state trajectory for imitation learning; when provided, imitation loss is added.
- `imitation_weight`: Weight applied to the imitation loss factor (default 1).
"""
function refinement_loss(network, transition_fn, term_cost_fn, mismatch_fn,
                         x_guess_init::AbstractArray, u_guess_init::AbstractArray, x_0::AbstractArray, steps::Integer;
                         target::Union{Nothing,AbstractArray}=nothing, imitation_weight::Real=1)
    # Apply refinement, then evaluate the residual and terminal cost on the final guess.
    x_guess, u_guess = refine(network, transition_fn, term_cost_fn, x_guess_init, u_guess_init, x_0, steps)
    x_res = _compute_residual(x_guess, u_guess, x_0, transition_fn)
    objective_loss = mismatch_fn(x_res, x_guess) + term_cost_fn(selectdim(x_guess, 2, size(x_guess, 2)))

    # Optional imitation loss on the refined state trajectory.
    if target === nothing
        return objective_loss
    else
        imitation_loss = mismatch_fn(x_guess, target)
        return objective_loss * (1 + imitation_weight * imitation_loss)
    end
end

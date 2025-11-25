using Flux

"""
    RefinerSolver

A solver that refines trajectories using a `CorrectionNetwork`.

# Fields
- `network`: The `CorrectionNetwork`.
- `transition`: Transition network `(x_prev_seq, u_seq) -> x_next_seq`.
- `constraint_fn`: Function `x -> val` (scalar). Positive means satisfied.
"""
struct RefinerSolver{N, T, F}
    network::N
    transition::T
    constraint_fn::F
end

Flux.trainable(m::RefinerSolver) = (m.network,)

"""
    step_refiner(solver::RefinerSolver, x_guess, u_guess, x_0)

Performs one step of trajectory refinement.

# Arguments
- `solver`: The `RefinerSolver`.
- `x_guess`: Current state trajectory guess. Shape: (state_dim, seq_len, batch)
- `u_guess`: Current input trajectory guess. Shape: (input_dim, seq_len, batch)
- `x_0`: Initial state context. Shape: (state_dim, batch)

# Returns
- `x_new`: Refined state trajectory.
- `u_new`: Refined input trajectory.
"""
function step_refiner(solver::RefinerSolver, x_guess::AbstractArray, u_guess::AbstractArray, x_0::AbstractArray)
    # Dimensions
    state_dim = size(x_guess, 1)
    seq_len = size(x_guess, 2)
    batch_size = size(x_guess, 3)
    
    # Step 1: Compute Residuals (Batch Processing)
    # x_res[t] = transition(x_guess[t-1], u_guess[t])
    # We construct x_prev sequence: [x_0, x_guess[:, 1:end-1, :]]
    
    # x_guess: (state_dim, seq_len, batch)
    # x_0: (state_dim, batch)
    
    # We need to unsqueeze x_0 to (state_dim, 1, batch)
    x_0_reshaped = reshape(x_0, state_dim, 1, batch_size)
    
    if seq_len > 1
        x_prev_seq = cat(x_0_reshaped, selectdim(x_guess, 2, 1:seq_len-1), dims=2)
    else
        x_prev_seq = x_0_reshaped
    end
    
    # Compute x_res in one batch call
    # transition(x_prev_seq, u_guess) -> x_res
    x_res = solver.transition(x_prev_seq, u_guess)
    
    # Step 2: Network Forward Pass
    delta_inter, out_term = solver.network(x_res, x_guess, u_guess, x_0)
    
    # Step 3: Terminal Constraint
    # Check violation at the last step of x_guess
    x_term = selectdim(x_guess, 2, seq_len) # (state_dim, batch)
    
    # constraint_fn(x) -> val
    # We assume constraint_fn handles batching or we map it.
    # Let's assume it returns (1, batch) or (batch,)
    term_val = solver.constraint_fn(x_term)
    
    # Violation: max(0, -val). If val > 0, violation is 0.
    # We need to handle shapes carefully.
    term_violation = max.(0f0, .-term_val) # Element-wise max with 0
    
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

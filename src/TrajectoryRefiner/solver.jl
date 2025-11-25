using Flux

"""
    RefinerSolver

A solver that refines trajectories using a `CorrectionNetwork`.

# Fields
- `network`: The `CorrectionNetwork`.
- `transition_fn`: Function `(x, u) -> x_next`.
- `constraint_fn`: Function `x -> val` (scalar). Positive means satisfied.
"""
struct RefinerSolver{N, F1, F2}
    network::N
    transition_fn::F1
    constraint_fn::F2
end

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
    
    # Step 1: Compute Residuals (Multiple Shooting)
    # x_res[i] = transition(x_prev, u_curr)
    # Note: x_guess indices are 1..N. x_0 is index 0.
    # u_guess indices are 1..N.
    
    # We need to iterate through time steps.
    # Since transition_fn might not support broadcasting over time, we loop.
    # But it should support batching.
    
    x_res = similar(x_guess)
    
    # First step: transition from x_0 using u_guess[:, 1, :]
    u_1 = selectdim(u_guess, 2, 1) # (input_dim, batch)
    x_res_1 = solver.transition_fn(x_0, u_1)
    # Assign to x_res[:, 1, :]
    # We use Flux.unsqueeze to match dimensions if needed, or just assignment
    # selectdim returns a view, but we can't assign to a view of a similar array easily if it's not mutable in the right way (e.g. CuArray).
    # For now assuming CPU arrays or Zygote-friendly operations.
    # Actually, to be Zygote-friendly, we should construct x_res as a list and stack.
    
    x_res_list = []
    push!(x_res_list, x_res_1)
    
    for t in 2:seq_len
        x_prev = selectdim(x_guess, 2, t-1)
        u_curr = selectdim(u_guess, 2, t)
        x_next = solver.transition_fn(x_prev, u_curr)
        push!(x_res_list, x_next)
    end
    
    # Stack along time dimension (dim 2)
    # Each element is (state_dim, batch)
    # We want (state_dim, seq_len, batch)
    # stack(x_res_list, dims=2) works in recent Julia versions
    x_res = stack(x_res_list, dims=2)
    
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

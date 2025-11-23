struct CarIterator
    data::AbstractVector 
    batch_size::Integer 
    shuffle::Bool 
end

function Base.iterate(iter::CarIterator, state=1)
    if state > length(iter.data)
        return nothing
    end
    
    # Ensure we don't go out of bounds
    end_idx = min(state + iter.batch_size - 1, length(iter.data))
    
    # We need to collect columns to form a matrix
    contexts_list = Vector{Vector{Float32}}()
    samples_list = Vector{Vector{Float32}}()
    
    # We pick random parameters based on the first element of the batch?
    # Or should we pick for each element?
    # The prompt implied "iteration index" is common for the batch.
    # So we pick one recur_index for the whole batch.
    # But start_time might need to be valid for all.
    
    # Let's assume all trajectories are long enough or we handle it.
    # Using the first element to determine parameters:
    first_traj = iter.data[state].state_trajectory
    T = size(first_traj, 2)
    
    # max_recur depends on the TOTAL horizon of the trajectory
    # This defines the "Top Level" (Global View)
    max_recur = floor(Int, log2(T))
    if max_recur < 1
        max_recur = 1
    end
    
    # Ensure valid start_time
    start_time = rand(1:(T-1)) 
    time_span = T - start_time
    # Ensure time_span is positive for log
    if time_span < 1
        start_time = 1
        time_span = T - 1
    end
    
    # available_recur depends on the REMAINING time
    # We can't predict further than the end of the trajectory
    available_recur = floor(Int, log2(time_span))
    if available_recur < 1
        available_recur = 1
    end
    
    # Pick a recurrence level valid for the current start_time
    recur_index = rand(1:available_recur)

    for i in state:end_idx
        strj = iter.data[i].state_trajectory
        # utrj = iter.data[i].input_signal # Unused?
        
        # Check if this trajectory is compatible with the chosen start_time/recur_index
        # If not, maybe skip or clamp?
        curr_T = size(strj, 2)
        
        # Context: state at start_time
        # If start_time is out of bounds, clamp it?
        st = min(start_time, curr_T)
        context = strj[:, st]
        
        # Sample: state at future time
        # Range: [st + 2^(k-1), st + 2^k]
        min_offset = 2^(recur_index-1)
        max_offset = 2^recur_index
        
        idx_min = st + min_offset
        idx_max = min(st + max_offset, curr_T)
        
        if idx_min > curr_T
            # Fallback: just take the last element
            idx = curr_T
        else
            idx = rand(idx_min:idx_max)
        end
        
        sample = strj[:, idx]
        
        push!(contexts_list, context)
        push!(samples_list, sample)        
    end
    
    # Convert list of vectors to matrix (features x batch_size)
    contexts = reduce(hcat, contexts_list)
    samples = reduce(hcat, samples_list)

    # Return time scale (2^recur_index) and max_recur
    # The gradient computation will derive the iteration index from these.
    time_scale = 2^recur_index
    
    return ((context=contexts, samples=samples, time_scale=time_scale, max_recur=max_recur), state + 1)
end

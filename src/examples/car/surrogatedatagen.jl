using Random
using LazySets
using JLD2

"""
    generate_transition_dataset(sys::DiscreteRandomSystem, outfile::AbstractString;
                                iters::Integer=1000, rng=Random.default_rng(),
                                save_period::Real=60.0)

Generate random transition samples from a `DiscreteRandomSystem` for surrogate data.

- Draw states from `sys.X` and inputs from `sys.U` using `LazySets.sample`.
- Compute `x_next = sys.f(x, u)`; skip samples that throw `DomainError`.
- Keep the sample only if `x_next ∈ sys.X` (state space hyperrectangle).
- Store each sample as a named tuple `(state, input, next)`.
- If `outfile` already exists and contains `data_existing`, append to it.
- Save the vector to `outfile` via JLD2 and return it, periodically persisting every `save_period` seconds.

# Arguments
- `sys`: The discrete random system with sets `X` (states), `U` (inputs), and transition `f`.
- `outfile`: Path to a JLD2 file to write/append the dataset.
- `iters`: Number of sampling attempts (default `1000`).
- `rng`: Random number generator (default `Random.default_rng()`).
- `save_period`: Minimum elapsed seconds between checkpoint saves (default `60.0`).

# Returns
- `Vector{NamedTuple}` where each element has fields `state`, `input`, and `next`.
"""
function generate_transition_dataset(sys::DiscreteRandomSystem, outfile::AbstractString;
                                     iters::Integer=1000, rng=Random.default_rng(),
                                     save_period::Real=60.0)
    data = NamedTuple[]

    if isfile(outfile)
        try
            existing = JLD2.load(outfile)
            if haskey(existing, "data")
                data_existing = existing["data"]
                if isa(data_existing, AbstractVector{<:NamedTuple})
                    append!(data, data_existing)
                end
            end
        catch
            # If loading fails, proceed with empty data.
        end
    end

    last_save = time()
    for _ in 1:iters
        # Sample state and input from the corresponding sets
        # LazySets.sample returns columns of samples when n>1; with n=1, take the first element/column.
        x_sample = LazySets.sample(sys.X, 1; rng=rng)
        if x_sample isa AbstractMatrix
            x = vec(x_sample[:, 1])
        elseif x_sample isa AbstractVector{<:AbstractVector}
            x = x_sample[1]
        else
            x = x_sample
        end

        u_sample = LazySets.sample(sys.U, 1; rng=rng)
        if u_sample isa AbstractMatrix
            u = vec(u_sample[:, 1])
        elseif u_sample isa AbstractVector{<:AbstractVector}
            u = u_sample[1]
        else
            u = u_sample
        end
        try
            x_next = sys.f(x, u)
            # Skip if next state lies outside the state space.
            if LazySets.in(x_next, sys.X)
                push!(data, (state = x, input = u, next = x_next))
            end
        catch e
            if isa(e, DomainError)
                continue
            else
                rethrow()
            end
        end

        # Periodically checkpoint the accumulated samples to disk.
        if (time() - last_save) >= save_period
            @save outfile data
            last_save = time()
        end
    end

    @save outfile data
    return data
end

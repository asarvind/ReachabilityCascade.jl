using LazySets
using JLD2

"""
    generate_transition_dataset(sys::DiscreteRandomSystem, outfile::AbstractString; iters::Integer=1000, rng=Random.default_rng())

Generate random transition samples from a `DiscreteRandomSystem` by sampling states from its state set
and inputs from its input set. Each valid sample is a named tuple `(state, input, next)`.
Samples that raise a `DomainError` during transition are skipped. Results are written to `outfile`
as a vector of named tuples using JLD2, and also returned.
"""
function generate_transition_dataset(sys::DiscreteRandomSystem, outfile::AbstractString; iters::Integer=1000, rng=Random.default_rng())
    data = NamedTuple[]

    if isfile(outfile)
        try
            @load outfile data_existing
            if isa(data_existing, AbstractVector{<:NamedTuple})
                append!(data, data_existing)
            end
        catch
            # If loading fails, proceed with empty data.
        end
    end

    for _ in 1:iters
        # Sample state and input from the corresponding sets
        x = LazySets.sample(sys.X, 1, rng)[1]
        u = LazySets.sample(sys.U, 1, rng)[1]
        try
            x_next = sys.f(x, u)
            push!(data, (state = x, input = u, next = x_next))
        catch e
            if isa(e, DomainError)
                continue
            else
                rethrow()
            end
        end
    end

    @save outfile data
    return data
end

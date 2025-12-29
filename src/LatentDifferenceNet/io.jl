import Flux
using JLD2

import ..TrainingAPI: save, load

_refinement_constructor(model::RefinementRNN)::Tuple{NTuple{9,Int},NamedTuple} = begin
    args = (model.delta.state_dim,
            model.delta.input_dim,
            model.delta.cost_dim,
            model.delta.latent_dim,
            model.delta.seq_len,
            model.policy.hidden_dim,
            model.policy.depth,
            model.delta.hidden_dim,
            model.delta.depth)
    kwargs = (; max_seq_len=model.delta.max_seq_len, nheads=model.delta.nheads, activation=model.delta.activation)
    return args, kwargs
end

"""
    save(model::RefinementRNN, path::AbstractString;
         losses=nothing, best_steps=nothing, min_traj_costs=nothing,
         accept_flags=nothing, accept_choices=nothing,
         base_losses=nothing, pert_losses=nothing, pos_losses=nothing, neg_losses=nothing, deltas=nothing)

Save a `RefinementRNN` checkpoint using `Flux.state`, along with constructor `(args, kwargs)`.

Optional payloads:
- gradient-training logs: `losses`, `best_steps`, `min_traj_costs` (kept for compatibility, may be `nothing`)
- perturbation-training logs: `accept_flags`, `accept_choices`, `base_losses`, `pert_losses`, `pos_losses`, `neg_losses`, `deltas`
"""
function save(model::RefinementRNN, path::AbstractString;
              losses::Union{Nothing,AbstractVector{<:Real}}=nothing,
              best_steps::Union{Nothing,AbstractVector{<:Integer}}=nothing,
              min_traj_costs::Union{Nothing,AbstractVector{<:Real}}=nothing,
              accept_flags::Union{Nothing,AbstractVector{Bool}}=nothing,
              accept_choices::Union{Nothing,AbstractVector{<:Integer}}=nothing,
              base_losses::Union{Nothing,AbstractVector{<:Real}}=nothing,
              pert_losses::Union{Nothing,AbstractVector{<:Real}}=nothing,
              pos_losses::Union{Nothing,AbstractVector{<:Real}}=nothing,
              neg_losses::Union{Nothing,AbstractVector{<:Real}}=nothing,
              deltas::Union{Nothing,AbstractVector{<:Real}}=nothing)
    state = Flux.state(model)
    args, kwargs = _refinement_constructor(model)
    jldsave(path; state, args, kwargs,
            losses, best_steps, min_traj_costs,
            accept_flags, accept_choices, base_losses, pert_losses, pos_losses, neg_losses, deltas)
    return path
end

save(::Type{RefinementRNN}, path::AbstractString, model::RefinementRNN; kwargs...) =
    save(model, path; kwargs...)

"""
    load(::Type{RefinementRNN}, path::AbstractString) -> RefinementRNN

Load a `RefinementRNN` checkpoint saved by [`save`](@ref).
Returns the reconstructed model with parameters loaded.
"""
function load(::Type{RefinementRNN}, path::AbstractString)
    data = JLD2.load(path)
    args = data["args"]
    kwargs = data["kwargs"]
    state = data["state"]
    model = RefinementRNN(args...; kwargs...)
    Flux.loadmodel!(model, state)
    return model
end

_sample_x0(rng::AbstractRNG,
           sample::NamedTuple,
           start_idx_range::Union{Nothing,Integer,AbstractUnitRange{<:Integer},Tuple{<:Integer,<:Integer}}) = begin
    x_full = Array(sample.state_trajectory)
    T_x = size(x_full, 2)
    T_x >= 1 || throw(ArgumentError("trajectory must have at least one state"))

    if start_idx_range === nothing
        start_idx = rand(rng, 1:T_x)
    elseif start_idx_range isa Integer
        s = Int(start_idx_range)
        (1 <= s <= T_x) || throw(ArgumentError("start_idx must be in 1:$T_x; got $s"))
        start_idx = s
    else
        lo = max(1, Int(first(start_idx_range)))
        hi = min(T_x, Int(last(start_idx_range)))
        lo <= hi || throw(ArgumentError("start_idx_range intersects trajectory to an empty range"))
        start_idx = rand(rng, lo:hi)
    end

    return x_full[:, start_idx]
end

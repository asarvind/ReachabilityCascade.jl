import Flux
using JLD2

import ..TrainingAPI: save, load

_reactive_denoising_constructor(model::ReactiveDenoisingNet)::Tuple{NTuple{6,Int},NamedTuple} = begin
    args = (model.state_dim,
            model.input_dim,
            model.cost_dim,
            model.seq_len,
            model.hidden_dim,
            model.depth)
    kwargs = (; max_seq_len=model.max_seq_len, nheads=model.nheads, activation=model.activation)
    return args, kwargs
end

"""
    save(model::ReactiveDenoisingNet, path::AbstractString; losses=nothing)

Save a `ReactiveDenoisingNet` checkpoint using `Flux.state`, along with constructor `(args, kwargs)`.
"""
function save(model::ReactiveDenoisingNet, path::AbstractString;
              losses::Union{Nothing,AbstractVector{<:Real}}=nothing)
    state = Flux.state(model)
    args, kwargs = _reactive_denoising_constructor(model)
    jldsave(path; state, args, kwargs, losses)
    return path
end

save(::Type{ReactiveDenoisingNet}, path::AbstractString, model::ReactiveDenoisingNet; kwargs...) =
    save(model, path; kwargs...)

"""
    load(::Type{ReactiveDenoisingNet}, path::AbstractString) -> ReactiveDenoisingNet

Load a `ReactiveDenoisingNet` checkpoint saved by [`save`](@ref).
Returns the reconstructed model with parameters loaded.
"""
function load(::Type{ReactiveDenoisingNet}, path::AbstractString)
    data = JLD2.load(path)
    args = data["args"]
    kwargs = data["kwargs"]
    state = data["state"]
    model = ReactiveDenoisingNet(args...; kwargs...)
    Flux.loadmodel!(model, state)
    return model
end

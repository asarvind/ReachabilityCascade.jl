import Flux
using JLD2

import ..TrainingAPI: save, load

_flow_constructor(flow::NormalizingFlow)::Tuple{NTuple{2,Int},NamedTuple} = begin
    args = (flow.dim, flow.context_dim)
    perms = [layer.perm for layer in flow.layers]
    kwargs = (; spec=flow.spec, logscale_clamp=flow.logscale_clamp, perms=perms)
    return args, kwargs
end

"""
    save(model, path; losses=nothing)

Save a `NormalizingFlow` checkpoint using `Flux.state`, along with constructor `(args, kwargs)`.

# Arguments
- `model`: [`NormalizingFlow`](@ref) to save.
- `path`: output path (typically `.jld2`).

# Keyword Arguments
- `losses=nothing`: optional training loss trace to store alongside the checkpoint.

# Returns
- `path`: the same path string.
"""
function save(model::NormalizingFlow, path::AbstractString;
              losses::Union{Nothing,AbstractVector{<:Real}}=nothing)
    state = Flux.state(model)
    args, kwargs = _flow_constructor(model)
    jldsave(path; state, args, kwargs, losses)
    return path
end

save(::Type{NormalizingFlow}, path::AbstractString, model::NormalizingFlow; kwargs...) =
    save(model, path; kwargs...)

"""
    load(::Type{NormalizingFlow}, path) -> model

Load a `NormalizingFlow` checkpoint saved by [`save`](@ref).

# Arguments
- `path`: checkpoint path.

# Returns
- `model::NormalizingFlow`: reconstructed model with parameters loaded.
"""
function load(::Type{NormalizingFlow}, path::AbstractString)
    data = JLD2.load(path)
    args = data["args"]
    kwargs = data["kwargs"]
    state = data["state"]

    model = if kwargs isa NamedTuple
        NormalizingFlow(args...; kwargs...)
    elseif kwargs isa AbstractDict
        NormalizingFlow(args...;
                        spec=kwargs["spec"],
                        logscale_clamp=kwargs["logscale_clamp"],
                        perms=kwargs["perms"])
    else
        throw(ArgumentError("checkpoint 'kwargs' must be a NamedTuple or Dict; got $(typeof(kwargs))"))
    end
    Flux.loadmodel!(model, state)
    return model
end

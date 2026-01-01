import Flux
using JLD2

import ..TrainingAPI: save, load

_coupling_constructor(net::InvertibleCoupling)::Tuple{NTuple{2,Int},NamedTuple} = begin
    args = (net.dim, net.context_dim)
    # One permutation per `spec` column (two sublayers share the same permutation).
    L = size(net.spec, 2)
    perms = [net.layers[2i - 1].perm for i in 1:L]
    kwargs = (; spec=net.spec, logscale_clamp=net.logscale_clamp, perms=perms)
    return args, kwargs
end

"""
    save(model, path; losses=nothing)

Save an [`InvertibleCoupling`](@ref) checkpoint using `Flux.state`, along with constructor `(args, kwargs)`.

# Arguments
- `model`: [`InvertibleCoupling`](@ref) to save.
- `path`: output path (typically `.jld2`).

# Keyword Arguments
- `losses=nothing`: optional training loss trace to store alongside the checkpoint.

# Returns
- `path`: the same path string.
"""
function save(model::InvertibleCoupling, path::AbstractString;
              losses::Union{Nothing,AbstractVector{<:Real}}=nothing)
    state = Flux.state(model)
    args, kwargs = _coupling_constructor(model)
    jldsave(path; state, args, kwargs, losses)
    return path
end

save(::Type{InvertibleCoupling}, path::AbstractString, model::InvertibleCoupling; kwargs...) =
    save(model, path; kwargs...)

"""
    load(::Type{InvertibleCoupling}, path) -> model

Load an [`InvertibleCoupling`](@ref) checkpoint saved by [`save`](@ref).

# Arguments
- `path`: checkpoint path.

# Returns
- `model::InvertibleCoupling`: reconstructed model with parameters loaded.
"""
function load(::Type{InvertibleCoupling}, path::AbstractString)
    data = JLD2.load(path)
    args = data["args"]
    kwargs = data["kwargs"]
    state = data["state"]

    model = _construct_coupling(args, kwargs)

    Flux.loadmodel!(model, state)
    return model
end

"""
    save_game(path, model_a, model_b; losses_a=nothing, losses_b=nothing)

Save a two-network InvertibleGame checkpoint (both models in one `.jld2` file).

This is a convenience wrapper around the same `Flux.state` + constructor `(args, kwargs)` pattern used
by `TrainingAPI.save`/`TrainingAPI.load`, but for a *pair* of models.

# Arguments
- `path`: output path (typically `.jld2`).
- `model_a`: first [`InvertibleCoupling`](@ref).
- `model_b`: second [`InvertibleCoupling`](@ref).

# Keyword Arguments
- `losses_a=nothing`: optional loss trace for `model_a`.
- `losses_b=nothing`: optional loss trace for `model_b`.

# Returns
- `path`: the same path string.
"""
function save_game(path::AbstractString,
                   model_a::InvertibleCoupling,
                   model_b::InvertibleCoupling;
                   losses_a::Union{Nothing,AbstractVector{<:Real}}=nothing,
                   losses_b::Union{Nothing,AbstractVector{<:Real}}=nothing)
    state_a = Flux.state(model_a)
    args_a, kwargs_a = _coupling_constructor(model_a)
    state_b = Flux.state(model_b)
    args_b, kwargs_b = _coupling_constructor(model_b)
    jldsave(path; state_a, args_a, kwargs_a, losses_a, state_b, args_b, kwargs_b, losses_b)
    return path
end

_construct_coupling(args, kwargs) = begin
    if kwargs isa NamedTuple
        return InvertibleCoupling(args...; kwargs...)
    elseif kwargs isa AbstractDict
        return InvertibleCoupling(args...;
                                  spec=kwargs["spec"],
                                  logscale_clamp=kwargs["logscale_clamp"],
                                  perms=kwargs["perms"])
    else
        throw(ArgumentError("checkpoint 'kwargs' must be a NamedTuple or Dict; got $(typeof(kwargs))"))
    end
end

"""
    load_game(path) -> (model_a, model_b, meta)

Load a two-network InvertibleGame checkpoint saved by [`save_game`](@ref).

# Arguments
- `path`: checkpoint path.

# Returns
- `model_a::InvertibleCoupling`
- `model_b::InvertibleCoupling`
- `meta::NamedTuple`: `(; losses_a, losses_b)`
"""
function load_game(path::AbstractString)
    data = JLD2.load(path)
    model_a = _construct_coupling(data["args_a"], data["kwargs_a"])
    Flux.loadmodel!(model_a, data["state_a"])

    model_b = _construct_coupling(data["args_b"], data["kwargs_b"])
    Flux.loadmodel!(model_b, data["state_b"])

    losses_a = haskey(data, "losses_a") ? data["losses_a"] : nothing
    losses_b = haskey(data, "losses_b") ? data["losses_b"] : nothing
    return model_a, model_b, (; losses_a, losses_b)
end

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
    save_self(path, model; kwargs...)

Save a single-network checkpoint for self-adversarial training.

This stores both the live model and (optionally) its EMA copy in one `.jld2` file.

# Arguments
- `path`: output path (typically `.jld2`).
- `model`: live [`InvertibleCoupling`](@ref).

# Keyword Arguments
- `losses=nothing`: optional loss trace.
- `ema=nothing`: optional EMA copy of `model` (same architecture).
- `ema_beta_start=nothing`: optional EMA schedule start value.
- `ema_beta_final=nothing`: optional EMA schedule final value.
- `ema_tau=nothing`: optional EMA schedule time constant (in optimizer steps).
- `ema_step=nothing`: optional EMA update step counter.

# Returns
- `path`: the same path string.
"""
function save_self(path::AbstractString,
                   model::InvertibleCoupling;
                   losses::Union{Nothing,AbstractVector{<:Real}}=nothing,
                   ema::Union{Nothing,InvertibleCoupling}=nothing,
                   ema_beta_start::Union{Nothing,Real}=nothing,
                   ema_beta_final::Union{Nothing,Real}=nothing,
                   ema_tau::Union{Nothing,Real}=nothing,
                   ema_step::Union{Nothing,Integer}=nothing)
    state = Flux.state(model)
    args, kwargs = _coupling_constructor(model)
    state_ema = ema === nothing ? nothing : Flux.state(ema)
    jldsave(path; state, args, kwargs, losses, state_ema, ema_beta_start, ema_beta_final, ema_tau, ema_step)
    return path
end

"""
    load_self(path) -> (model, meta)

Load a single-network checkpoint saved by [`save_self`](@ref).

# Arguments
- `path`: checkpoint path.

# Returns
- `model::InvertibleCoupling`: reconstructed live model with parameters loaded.
- `meta::NamedTuple`: `(; losses, ema, ema_beta_start, ema_beta_final, ema_tau, ema_step)`
"""
function load_self(path::AbstractString)
    data = JLD2.load(path)
    model = _construct_coupling(data["args"], data["kwargs"])
    Flux.loadmodel!(model, data["state"])

    losses = haskey(data, "losses") ? data["losses"] : nothing
    ema = if haskey(data, "state_ema") && !(data["state_ema"] === nothing)
        m = _construct_coupling(data["args"], data["kwargs"])
        Flux.loadmodel!(m, data["state_ema"])
        m
    else
        nothing
    end
    ema_beta_start = haskey(data, "ema_beta_start") ? data["ema_beta_start"] : nothing
    ema_beta_final = haskey(data, "ema_beta_final") ? data["ema_beta_final"] : nothing
    ema_tau = haskey(data, "ema_tau") ? data["ema_tau"] : nothing
    ema_step = haskey(data, "ema_step") ? data["ema_step"] : nothing

    # Backward compatibility: older checkpoints saved a constant `ema_beta`.
    if ema_beta_start === nothing && haskey(data, "ema_beta")
        β = data["ema_beta"]
        ema_beta_start = β
        ema_beta_final = β
        ema_tau = 1.0
        ema_step = 0
    end

    return model, (; losses, ema, ema_beta_start, ema_beta_final, ema_tau, ema_step)
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

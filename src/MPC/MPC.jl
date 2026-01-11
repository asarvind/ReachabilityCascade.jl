module MPC

import NLopt
import LazySets
import LazySets: center

using Random

using ..ControlSystem: DiscreteRandomSystem
import ..InvertibleGame
import ..NormalizingFlows
using ..InvertibleGame: InvertibleCoupling
using ..NormalizingFlows: NormalizingFlow

export trajectory, optimize_latent, mpc

struct LatentPolicy{F}
    f::F
    dim::Int
end

"""
    trajectory(ds, model, x0, z, steps; kwargs...) -> state_trajectory

Roll out a state trajectory by decoding one or more latent vectors into a control policy and
simulating a discrete-time system.

The decision variable `z` is a flat vector of length `latent_dim * length(steps)` where:
- `latent_dim` is `model.dim`.
- `steps` can be an integer `T` or a vector `[T₁, T₂, ...]`. When `steps` is a vector, each segment uses its own latent.

Internally, the flat vector is interpreted as a matrix via:
`zmat = reshape(z, latent_dim, length(steps))`.

# Arguments
- `ds`: [`DiscreteRandomSystem`](@ref).
- `model`: either [`InvertibleCoupling`](@ref) or [`NormalizingFlow`](@ref).
- `x0`: initial state vector.
- `z`: flat latent vector.
- `steps`: integer or vector of integers describing horizon splits.

# Keyword Arguments
- `u_len=nothing`: control dimension. If `nothing`, it is inferred as `length(center(ds.U))` when `ds` has a field `U`.
  The decoded output is sliced to `u[1:u_len]`.

# Returns
- `state_trajectory`: state matrix `n×(1+sum(steps))` with the first column equal to `x0`.
"""
function trajectory(ds::DiscreteRandomSystem,
                    model,
                    x0,
                    z,
                    steps;
                    u_len=nothing)
    steps_vec = steps isa Integer ? [Int(steps)] : Int.(collect(steps))
    length(steps_vec) >= 1 || throw(ArgumentError("steps must contain at least one segment"))

    u_len_final = _infer_u_len(ds, u_len)

    x0_vec = x0 isa AbstractVector ? x0 : vec(x0)
    strj = reshape(x0_vec, :, 1)

    z_flat = z isa AbstractVector ? z : vec(z)
    zmat = reshape(z_flat, _latent_dim(model), length(steps_vec))

    for i in 1:length(steps_vec)
        t = steps_vec[i]
        t >= 1 || throw(ArgumentError("each segment length must be ≥ 1; got steps[$i]=$t"))

        κ = x -> control_from_latent(model, zmat[:, i], x; u_len=u_len_final)
        x_start = strj[:, end]
        seg = ds(x_start, κ, t)
        strj = hcat(strj, seg[:, 2:end])
    end

    return strj
end

"""
    trajectory(ds, model_decode, model_encode, x0, steps; kwargs...) -> state_trajectory

Roll out a state trajectory where the control at each step is obtained by solving a latent
optimization problem using two [`InvertibleCoupling`](@ref) networks.

At each step with current state `x` as the context:
1. Solve for `z` that minimizes `0.5*(‖z‖ + ‖encode(model_encode, decode(model_decode, z, x), x)‖)`.
2. Apply `u = decode(model_decode, z, x)` (sliced to `u_len`) for one step.
3. Warm-start the next step using the previous `z`.

# Arguments
- `ds`: [`DiscreteRandomSystem`](@ref).
- `model_decode`: [`InvertibleCoupling`](@ref) used to decode latent to control.
- `model_encode`: [`InvertibleCoupling`](@ref) used to re-encode the decoded control.
- `x0`: initial state vector.
- `steps`: rollout length. If an integer, runs for `steps` iterations. If a vector, runs for `sum(steps)`.

# Keyword Arguments
- `algo=:LN_BOBYQA`: NLopt algorithm symbol for the per-step latent optimization.
- `init_z=nothing`: initial latent guess (defaults to zeros).
- `max_time=Inf`: NLopt time cap (`maxtime`, seconds) per step.
- `seed=rand(1:10000)`: NLopt seed passed to each per-step optimization call.
- `norm_kind=:l1`: norm used in the objective (`:l1`, `:l2`, or `:linf`).
- `u_len=nothing`: control dimension. If `nothing`, inferred as `length(center(ds.U))` when `ds` has a field `U`.

# Returns
- `state_trajectory`: state matrix `n×(1+steps_total)` with the first column equal to `x0`.
"""
function trajectory(ds::DiscreteRandomSystem,
                    model_decode::InvertibleCoupling,
                    model_encode::InvertibleCoupling,
                    x0,
                    steps;
                    algo::Symbol=:LN_BOBYQA,
                    init_z=nothing,
                    max_time::Real=Inf,
                    seed::Integer=rand(1:10000),
                    norm_kind::Symbol=:l1,
                    u_len=nothing)
    steps_total = steps isa Integer ? Int(steps) : sum(Int.(collect(steps)))
    steps_total >= 1 || throw(ArgumentError("steps must be ≥ 1 (or sum to ≥ 1)"))

    u_len_final = _infer_u_len(ds, u_len)

    D = _latent_dim(model_decode)
    z = init_z === nothing ? zeros(Float32, D) : Float32.(init_z isa AbstractVector ? init_z : vec(init_z))
    length(z) == D || throw(DimensionMismatch("init_z must have length $D; got length=$(length(z))"))

    x0_vec = x0 isa AbstractVector ? x0 : vec(x0)
    strj = reshape(x0_vec, :, 1)

    for _ in 1:steps_total
        x = strj[:, end]
        res = InvertibleGame.decode(model_decode, model_encode, x;
                                    algo=algo,
                                    init_z=z,
                                    max_time=max_time,
                                    seed=seed,
                                    norm_kind=norm_kind,
                                    u_len=u_len_final,
                                    return_meta=true)
        z = res.z
        x_next = ds(x, res.u)
        strj = hcat(strj, x_next)
    end

    return strj
end

"""
    optimize_latent(cost_fn, ds, x0, model, steps; kwargs...) -> result

Optimize one or more latents `z` using NLopt to minimize a trajectory cost produced by
`cost_fn(trajectory)`, where `trajectory` is generated by [`trajectory`](@ref).

The NLopt decision variable is a *flat* vector `z` of length `model.dim * length(steps)`.
Inside the objective, this is reinterpreted via `reshape` into multiple latent vectors when
`steps` is a vector.

# Arguments
- `cost_fn`: maps a trajectory to a cost (scalar/vector/matrix). It is scalarized with `sum`.
- `ds`: [`DiscreteRandomSystem`](@ref).
- `x0`: initial state vector.
- `model`: either [`InvertibleCoupling`](@ref) or [`NormalizingFlow`](@ref).
- `steps`: integer or vector of integers; see [`trajectory`](@ref).

# Keyword Arguments
- `algo=:LN_BOBYQA`: NLopt algorithm symbol.
- `init_z=nothing`: initial guess for the flat latent vector. If `nothing`, uses zeros.
- `max_time=Inf`: NLopt `maxtime` (seconds).
- `seed=rand(1:10000)`: NLopt RNG seed.
- `u_len=nothing`: forwarded to [`trajectory`](@ref) to slice the decoded output.

# Returns
Named tuple:
- `objective`: best objective value.
- `z`: best latent vector (flat).
- `result`: NLopt return code.
"""
function optimize_latent(cost_fn,
                         ds::DiscreteRandomSystem,
                         x0,
                         model,
                         steps;
                         algo::Symbol=:LN_BOBYQA,
                         init_z=nothing,
                         max_time::Real=Inf,
                         seed::Integer=rand(1:10000),
                         u_len=nothing)
    steps_vec = steps isa Integer ? [Int(steps)] : Int.(collect(steps))
    length(steps_vec) >= 1 || throw(ArgumentError("steps must contain at least one segment"))

    u_len_final = _infer_u_len(ds, u_len)

    l = _latent_dim(model) * length(steps_vec)
    init_z_vec = init_z === nothing ? zeros(Float32, l) : (init_z isa AbstractVector ? init_z : vec(init_z))
    length(init_z_vec) == l || throw(DimensionMismatch("init_z must have length $l; got length=$(length(init_z_vec))"))

    function my_objective_fn(z::AbstractVector, grad::AbstractVector)
        strj = trajectory(ds, model, x0, z, steps_vec; u_len=u_len_final)
        return sum(cost_fn(strj)) / size(strj, 2)
    end

    opt = NLopt.Opt(algo, l)
    NLopt.min_objective!(opt, my_objective_fn)
    NLopt.stopval!(opt, 0)
    NLopt.maxtime!(opt, max_time)
    NLopt.srand(seed)

    min_f, min_z, ret = NLopt.optimize(opt, init_z_vec)
    return (; objective=min_f, z=min_z, result=ret)
end

"""
    mpc(cost_fn, ds, x0, model, steps; kwargs...) -> result

Run model-predictive control (MPC) by repeatedly optimizing latents and applying the first control.

At each MPC iteration:
1. Optimize latent(s) using [`optimize_latent`](@ref) (warm-started with the previous solution).
2. Decode the *first* latent to a control vector (sliced to `u_len`).
3. Optionally mix with noise: `u = u*(1-w) + u_noise*w`.
4. Apply one step: `x⁺ = ds(x, u)`.

# Arguments
- `cost_fn`: maps a trajectory to a cost (scalar/vector/matrix). Scalarized with `sum`.
- `ds`: [`DiscreteRandomSystem`](@ref).
- `x0`: initial state vector.
- `model`: either [`InvertibleCoupling`](@ref), [`NormalizingFlow`](@ref), or a function `model(x, z) -> u`.
- `steps`: MPC duration. If an integer, MPC runs for `steps` iterations. If a vector, MPC runs for `sum(steps)`.

# Keyword Arguments
- `algo=:LN_PRAXIS`: NLopt algorithm symbol used by [`optimize_latent`](@ref).
- `init_z=nothing`: initial guess for the flat latent vector (warm-start), defaults to zeros.
- `opt_steps=steps`: horizon specification passed to [`optimize_latent`](@ref). May differ from `steps`.
- `opt_seed=rand(1:10000)`: NLopt seed passed to each optimization call.
- `max_time=Inf`: NLopt `maxtime` passed to each optimization call.
- `u_len=nothing`: control dimension. If `nothing`, inferred from `ds.U` as `length(center(ds.U))`.
- `latent_dim=nothing`: required only when `model` is a function; specifies the latent dimension.
- `noise_fn=nothing`: optional noise sampler `noise_fn(rng) -> u_noise`. Defaults to `rng -> LazySets.sample(ds.U; rng=rng)`.
- `noise_weight=0.0`: mixing weight between decoded control and noise.
- `noise_rng=Random.default_rng()`: RNG passed to `noise_fn` for reproducibility.

# Returns
Named tuple:
- `trajectory`: state trajectory matrix (columns over time).
- `total_cost`: scalarized total cost `sum(cost_fn(trajectory))`.
- `objectives`: vector of per-iteration objective values.
- `u_noises`: matrix `u_len×steps_total` of the sampled noise vectors used at each MPC step (one column per step).
- `z`: final latent vector (flat) after the last optimization.
"""
function mpc(cost_fn,
             ds::DiscreteRandomSystem,
             x0,
             model,
             steps;
             algo::Symbol=:LN_PRAXIS,
             init_z=nothing,
             opt_steps=steps,
             opt_seed::Integer=rand(1:10000),
             max_time::Real=Inf,
             u_len=nothing,
             latent_dim::Union{Nothing,Integer}=nothing,
             noise_fn::Union{Function,Nothing}=nothing,
             noise_weight::Real=0.0,
             noise_rng::Random.AbstractRNG=Random.default_rng())
    model_eff = if model isa Function
        latent_dim === nothing && throw(ArgumentError("latent_dim must be provided when model is a function"))
        LatentPolicy(model, Int(latent_dim))
    else
        model
    end

    steps_total = steps isa Integer ? Int(steps) : sum(Int.(collect(steps)))
    steps_total >= 1 || throw(ArgumentError("steps must be ≥ 1 (or sum to ≥ 1)"))

    u_len_final = _infer_u_len(ds, u_len)

    nf = if noise_fn === nothing
        hasproperty(ds, :U) || throw(ArgumentError("noise_fn must be provided when ds has no field U"))
        rng -> LazySets.sample(getproperty(ds, :U); rng=rng)
    else
        noise_fn
    end

    opt_steps_vec = opt_steps isa Integer ? [Int(opt_steps)] : Int.(collect(opt_steps))
    length(opt_steps_vec) >= 1 || throw(ArgumentError("opt_steps must contain at least one segment"))
    l = _latent_dim(model_eff) * length(opt_steps_vec)
    z = init_z === nothing ? zeros(Float32, l) : (init_z isa AbstractVector ? init_z : vec(init_z))
    length(z) == l || throw(DimensionMismatch("init_z must have length $l (model.dim * length(opt_steps)); got length=$(length(z))"))

    x0_vec = x0 isa AbstractVector ? x0 : vec(x0)
    strj = reshape(x0_vec, :, 1)
    objectives = Float64[]
    u_noises = Matrix{Float32}(undef, u_len_final, steps_total)

    for k in 1:steps_total
        x = strj[:, end]
        res = optimize_latent(cost_fn, ds, x, model_eff, opt_steps_vec;
                              algo=algo, init_z=z, max_time=max_time, seed=opt_seed, u_len=u_len_final)
        push!(objectives, Float64(res.objective))
        z = res.z

        u = control_from_latent(model_eff, z[1:_latent_dim(model_eff)], x; u_len=u_len_final)
        u_noise = nf(noise_rng)
        u_noise_vec = u_noise isa AbstractVector ? u_noise : vec(u_noise)
        length(u_noise_vec) >= u_len_final ||
            throw(DimensionMismatch("noise_fn must return at least $u_len_final elements; got length=$(length(u_noise_vec))"))
        u_noise_used = Float32.(u_noise_vec[1:u_len_final])
        u_noises[:, k] = u_noise_used
        u = u * (1 - noise_weight) + u_noise_used * noise_weight

        x_next = ds(x, u)
        strj = hcat(strj, x_next)
    end

    return (; trajectory=strj, total_cost=sum(cost_fn(strj)), objectives, u_noises, z)
end

"""
    mpc(cost_fn, ds, x0, models, steps; kwargs...) -> result

Multi-model MPC variant.

At each MPC iteration, we run one latent optimization per candidate model and pick the
model/latent with the lowest objective (best cost) for applying the control.

This is useful for side-by-side comparisons (e.g., evaluating different learned models on
the same MPC problem) while keeping the rest of the MPC loop identical.

# Arguments
- `cost_fn`: maps a trajectory to a cost (scalar/vector/matrix). Scalarized with `sum`.
- `ds`: [`DiscreteRandomSystem`](@ref).
- `x0`: initial state vector.
- `models`: collection of models (each either [`InvertibleCoupling`](@ref), [`NormalizingFlow`](@ref), or a function `m(x, z) -> u`).
- `steps`: MPC duration. If an integer, MPC runs for `steps` iterations. If a vector, MPC runs for `sum(steps)`.

# Keyword Arguments
- `algo=:LN_PRAXIS`: NLopt algorithm symbol used by [`optimize_latent`](@ref).
- `init_z=nothing`: initial guesses for the *flat* latent vectors (warm-start):
  - `nothing`: uses zeros for each model.
  - a vector of vectors, one per model, each of length `model.dim * length(opt_steps)`.
- `opt_steps=steps`: horizon specification passed to [`optimize_latent`](@ref). May differ from `steps`.
- `opt_seed=rand(1:10000)`: NLopt seed passed to each optimization call (same seed reused across models per MPC step).
- `max_time=Inf`: NLopt `maxtime` passed to each optimization call.
- `u_len=nothing`: control dimension. If `nothing`, inferred from `ds.U` as `length(center(ds.U))`.
- `latent_dim=nothing`: required only when any entry of `models` is a function; specifies the latent dimension.
- `noise_fn=nothing`: optional noise sampler `noise_fn(rng) -> u_noise`. Defaults to `rng -> LazySets.sample(ds.U; rng=rng)`.
- `noise_weight=0.0`: mixing weight between decoded control and noise.
- `noise_rng=Random.default_rng()`: RNG passed to `noise_fn` for reproducibility.

# Returns
Named tuple:
- `trajectory`: state trajectory matrix (columns over time).
- `total_cost`: scalarized total cost `sum(cost_fn(trajectory))`.
- `objectives`: vector of per-iteration *best* objective values (one per MPC step).
- `u_noises`: matrix `u_len×steps_total` of the sampled noise vectors used at each MPC step (one column per step).
- `chosen_models`: vector of selected model indices (one per MPC step).
- `candidate_objectives`: vector (over MPC steps) of objective vectors (one per model).
- `candidate_zs`: vector (over MPC steps) of matrices `l×M` where each column is a candidate flat latent.
- `z`: final selected latent vector (flat) from the last MPC step.
"""
function mpc(cost_fn,
             ds::DiscreteRandomSystem,
             x0,
             models::AbstractVector,
             steps;
             algo::Symbol=:LN_PRAXIS,
             init_z=nothing,
             opt_steps=steps,
             opt_seed::Integer=rand(1:10000),
             max_time::Real=Inf,
             u_len=nothing,
             latent_dim::Union{Nothing,Integer}=nothing,
             noise_fn::Union{Function,Nothing}=nothing,
             noise_weight::Real=0.0,
             noise_rng::Random.AbstractRNG=Random.default_rng())
    isempty(models) && throw(ArgumentError("models must be non-empty"))

    models_eff = map(models) do m
        if m isa Function
            latent_dim === nothing && throw(ArgumentError("latent_dim must be provided when a model is a function"))
            LatentPolicy(m, Int(latent_dim))
        else
            m
        end
    end

    steps_total = steps isa Integer ? Int(steps) : sum(Int.(collect(steps)))
    steps_total >= 1 || throw(ArgumentError("steps must be ≥ 1 (or sum to ≥ 1)"))

    u_len_final = _infer_u_len(ds, u_len)

    nf = if noise_fn === nothing
        hasproperty(ds, :U) || throw(ArgumentError("noise_fn must be provided when ds has no field U"))
        rng -> LazySets.sample(getproperty(ds, :U); rng=rng)
    else
        noise_fn
    end

    opt_steps_vec = opt_steps isa Integer ? [Int(opt_steps)] : Int.(collect(opt_steps))
    length(opt_steps_vec) >= 1 || throw(ArgumentError("opt_steps must contain at least one segment"))

    latent_dim = _latent_dim(models_eff[1])
    for (i, m) in pairs(models_eff)
        _latent_dim(m) == latent_dim || throw(DimensionMismatch("all models must have the same latent dim; models[1].dim=$latent_dim but models[$i].dim=$(_latent_dim(m))"))
    end

    l = latent_dim * length(opt_steps_vec)

    z_list = if init_z === nothing
        [zeros(Float32, l) for _ in 1:length(models_eff)]
    elseif init_z isa AbstractVector{<:AbstractVector}
        length(init_z) == length(models_eff) || throw(DimensionMismatch("init_z must have one vector per model (length(models)=$(length(models_eff)))"))
        zs = [Float32.(z isa AbstractVector ? z : vec(z)) for z in init_z]
        for (i, z) in pairs(zs)
            length(z) == l || throw(DimensionMismatch("init_z[$i] must have length $l; got length=$(length(z))"))
        end
        zs
    else
        throw(ArgumentError("init_z must be nothing or a vector of vectors (one per model)"))
    end

    x0_vec = x0 isa AbstractVector ? x0 : vec(x0)
    strj = reshape(x0_vec, :, 1)

    objectives = Float64[]
    chosen_models = Int[]
    candidate_objectives = Vector{Vector{Float64}}()
    candidate_zs = Vector{Matrix{Float32}}()
    u_noises = Matrix{Float32}(undef, u_len_final, steps_total)

    z_final = zeros(Float32, l)

    for k in 1:steps_total
        x = strj[:, end]

        objs = Vector{Float64}(undef, length(models_eff))
        zs_new = Vector{Vector{Float32}}(undef, length(models_eff))
        for i in eachindex(models_eff)
            res = optimize_latent(cost_fn, ds, x, models_eff[i], opt_steps_vec;
                                  algo=algo, init_z=z_list[i], max_time=max_time, seed=opt_seed, u_len=u_len_final)
            objs[i] = Float64(res.objective)
            zs_new[i] = Float32.(res.z isa AbstractVector ? res.z : vec(res.z))
        end

        # Store candidate latents for inspection/debugging/comparisons.
        zmat = reduce(hcat, zs_new)
        push!(candidate_zs, zmat)
        push!(candidate_objectives, copy(objs))

        i_best = argmin(objs)
        push!(chosen_models, Int(i_best))
        push!(objectives, objs[i_best])

        z_list = zs_new
        z_final = zs_new[i_best]

        u = control_from_latent(models_eff[i_best], z_final[1:latent_dim], x; u_len=u_len_final)
        u_noise = nf(noise_rng)
        u_noise_vec = u_noise isa AbstractVector ? u_noise : vec(u_noise)
        length(u_noise_vec) >= u_len_final ||
            throw(DimensionMismatch("noise_fn must return at least $u_len_final elements; got length=$(length(u_noise_vec))"))
        u_noise_used = Float32.(u_noise_vec[1:u_len_final])
        u_noises[:, k] = u_noise_used
        u = u * (1 - noise_weight) + u_noise_used * noise_weight

        x_next = ds(x, u)
        strj = hcat(strj, x_next)
    end

    return (; trajectory=strj,
            total_cost=sum(cost_fn(strj)),
            objectives,
            u_noises,
            chosen_models,
            candidate_objectives,
            candidate_zs,
            z=z_final)
end

# --- Model-specific latent -> control conversion ---

_latent_dim(model) = getproperty(model, :dim)

control_from_latent(policy::LatentPolicy, z, x; u_len) = begin
    u = policy.f(x, z)
    u_vec = u isa AbstractVector ? u : vec(u)
    return u_vec[1:Int(u_len)]
end

control_from_latent(model::InvertibleCoupling, z, x; u_len) = begin
    u = InvertibleGame.decode(model, z, x)
    return u[1:Int(u_len)]
end

control_from_latent(model::NormalizingFlow, z, x; u_len) = begin
    u = NormalizingFlows.decode(model, z, x)
    return u[1:Int(u_len)]
end

_infer_u_len(ds::DiscreteRandomSystem, u_len) = begin
    if u_len === nothing
        hasproperty(ds, :U) || throw(ArgumentError("u_len must be provided when ds has no field U"))
        return length(center(getproperty(ds, :U)))
    end
    u_len_i = Int(u_len)
    u_len_i >= 1 || throw(ArgumentError("u_len must be ≥ 1; got $u_len"))
    return u_len_i
end

end # module MPC

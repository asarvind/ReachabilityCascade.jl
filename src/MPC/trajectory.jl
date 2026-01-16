struct LatentPolicy{F}
    f::F
    dim::Int
end

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

_rollout_with_zmat(ds::DiscreteRandomSystem, model, x0, zmat, steps_vec, u_len_final) = begin
    x0_vec = x0 isa AbstractVector ? x0 : vec(x0)
    strj = reshape(x0_vec, :, 1)
    utrj = Matrix{Float32}(undef, u_len_final, 0)

    for i in 1:length(steps_vec)
        t = steps_vec[i]
        for _ in 1:t
            x = strj[:, end]
            u = control_from_latent(model, zmat[:, i], x; u_len=u_len_final)
            u_vec = u isa AbstractVector ? u : vec(u)
            u_used = Float32.(u_vec[1:u_len_final])
            utrj = hcat(utrj, u_used)
            x_next = ds(x, u_used)
            strj = hcat(strj, x_next)
        end
    end

    return (; state_trajectory=strj, input_trajectory=utrj)
end

_rollout_with_zsequence(ds::DiscreteRandomSystem, model_decode, x0, zseq, u_len_final) = begin
    x0_vec = x0 isa AbstractVector ? x0 : vec(x0)
    strj = reshape(x0_vec, :, 1)
    utrj = Matrix{Float32}(undef, u_len_final, 0)

    for i in 1:size(zseq, 2)
        x = strj[:, end]
        u = InvertibleGame.decode(model_decode, zseq[:, i], x)
        u_used = Float32.(u[1:u_len_final])
        utrj = hcat(utrj, u_used)
        x_next = ds(x, u_used)
        strj = hcat(strj, x_next)
    end

    return (; state_trajectory=strj, input_trajectory=utrj)
end

_state_jacobians_from_z(ds, model, x0, z_flat, steps_vec, u_len_final, t_idxs::Vector{Int}; eps::Real) = begin
    base = _rollout_with_zmat(ds, model, x0,
                              reshape(z_flat, _latent_dim(model), length(steps_vec)),
                              steps_vec,
                              u_len_final)
    strj = base.state_trajectory
    state_dim = size(strj, 1)
    z_len = length(z_flat)
    jacobians = [zeros(Float64, state_dim, z_len) for _ in 1:length(t_idxs)]

    for k in 1:z_len
        z_pert = copy(z_flat)
        z_pert[k] += eps
        res = _rollout_with_zmat(ds, model, x0,
                                 reshape(z_pert, _latent_dim(model), length(steps_vec)),
                                 steps_vec,
                                 u_len_final)
        diff = (Float64.(res.state_trajectory) .- Float64.(strj)) ./ eps
        for (j, t) in pairs(t_idxs)
            jacobians[j][:, k] = diff[:, t]
        end
    end
    return jacobians
end

_state_jacobians_from_zsequence(ds, model_decode, x0, zseq, u_len_final, t_idxs::Vector{Int}; eps::Real) = begin
    base = _rollout_with_zsequence(ds, model_decode, x0, zseq, u_len_final)
    strj = base.state_trajectory
    state_dim = size(strj, 1)
    z_len = length(zseq)
    jacobians = [zeros(Float64, state_dim, z_len) for _ in 1:length(t_idxs)]

    for k in 1:z_len
        z_pert = copy(vec(zseq))
        z_pert[k] += eps
        res = _rollout_with_zsequence(ds, model_decode, x0,
                                      reshape(z_pert, size(zseq, 1), size(zseq, 2)),
                                      u_len_final)
        diff = (Float64.(res.state_trajectory) .- Float64.(strj)) ./ eps
        for (j, t) in pairs(t_idxs)
            jacobians[j][:, k] = diff[:, t]
        end
    end
    return jacobians
end

"""
    trajectory(ds, model, x0, z, steps; kwargs...) -> result

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
- `jacobian_times=Int[]`: time indices (columns of the state trajectory) for which to compute `∂x_t/∂z`.
  The first column (initial state) is index `1`. If empty, Jacobians are not computed.
- `eps=1f-6`: finite-difference step used when `jacobian_times` is non-empty.

# Returns
Named tuple:
- `state_trajectory`: state matrix `n×(1+sum(steps))` with the first column equal to `x0`.
- `input_trajectory`: control matrix `u_len×sum(steps)` with one column per applied input.
- `state_jocobians` (optional): vector of Jacobian matrices in the same order as `jacobian_times`.
"""
function trajectory(ds::DiscreteRandomSystem,
                    model,
                    x0,
                    z,
                    steps;
                    u_len=nothing,
                    jacobian_times::Vector{<:Integer}=Int[],
                    eps::Real=1f-6)
    steps_vec = steps isa Integer ? [Int(steps)] : Int.(collect(steps))
    length(steps_vec) >= 1 || throw(ArgumentError("steps must contain at least one segment"))

    u_len_final = _infer_u_len(ds, u_len)

    z_flat = z isa AbstractVector ? z : vec(z)
    zmat = reshape(z_flat, _latent_dim(model), length(steps_vec))

    t_idxs = Int.(jacobian_times)
    if isempty(t_idxs)
        res = _rollout_with_zmat(ds, model, x0, zmat, steps_vec, u_len_final)
        return res
    end

    any(t -> t < 1, t_idxs) && throw(ArgumentError("jacobian_times must be ≥ 1"))
    max_t = maximum(t_idxs)
    steps_vec = _truncate_steps(steps_vec, max_t)
    zmat = reshape(z_flat, _latent_dim(model), length(steps_vec))

    res = _rollout_with_zmat(ds, model, x0, zmat, steps_vec, u_len_final)
    max_t <= size(res.state_trajectory, 2) ||
        throw(ArgumentError("jacobian_times contains index $max_t beyond trajectory length $(size(res.state_trajectory, 2))"))

    jacobians = _state_jacobians_from_z(ds, model, x0, z_flat, steps_vec, u_len_final, t_idxs; eps=Float64(eps))
    return (; res..., state_jocobians=jacobians)
end

"""
    trajectory(ds, model_decode, model_encode, x0, steps; kwargs...) -> result

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
- `jacobian_times=Int[]`: time indices (columns of the state trajectory) for which to compute `∂x_t/∂z`.
  The first column (initial state) is index `1`. If empty, Jacobians are not computed.
- `eps=1f-6`: finite-difference step used when `jacobian_times` is non-empty.

# Returns
Named tuple:
- `state_trajectory`: state matrix `n×(1+steps_total)` with the first column equal to `x0`.
- `input_trajectory`: control matrix `u_len×steps_total` with one column per applied input.
- `state_jocobians` (optional): vector of Jacobian matrices in the same order as `jacobian_times`.
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
                    u_len=nothing,
                    jacobian_times::Vector{<:Integer}=Int[],
                    eps::Real=1f-6)
    steps_total = steps isa Integer ? Int(steps) : sum(Int.(collect(steps)))
    steps_total >= 1 || throw(ArgumentError("steps must be ≥ 1 (or sum to ≥ 1)"))

    u_len_final = _infer_u_len(ds, u_len)

    D = _latent_dim(model_decode)
    z = init_z === nothing ? zeros(Float32, D) : Float32.(init_z isa AbstractVector ? init_z : vec(init_z))
    length(z) == D || throw(DimensionMismatch("init_z must have length $D; got length=$(length(z))"))

    t_idxs = Int.(jacobian_times)
    if !isempty(t_idxs)
        any(t -> t < 1, t_idxs) && throw(ArgumentError("jacobian_times must be ≥ 1"))
        max_t = maximum(t_idxs)
        steps_total = min(steps_total, max_t - 1)
    end

    x0_vec = x0 isa AbstractVector ? x0 : vec(x0)
    strj = reshape(x0_vec, :, 1)
    utrj = Matrix{Float32}(undef, u_len_final, 0)
    zseq = Matrix{Float32}(undef, D, steps_total)

    for k in 1:steps_total
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
        zseq[:, k] = z
        u_used = Float32.(res.u[1:u_len_final])
        utrj = hcat(utrj, u_used)
        x_next = ds(x, u_used)
        strj = hcat(strj, x_next)
    end

    res = (; state_trajectory=strj, input_trajectory=utrj)
    if isempty(t_idxs)
        return res
    end

    max_t = maximum(t_idxs)
    max_t <= size(res.state_trajectory, 2) ||
        throw(ArgumentError("jacobian_times contains index $max_t beyond trajectory length $(size(res.state_trajectory, 2))"))

    jacobians = _state_jacobians_from_zsequence(ds, model_decode, x0, zseq, u_len_final, t_idxs; eps=Float64(eps))
    return (; res..., state_jocobians=jacobians)
end

_truncate_steps(steps_vec::Vector{Int}, max_t::Int) = begin
    max_t >= 1 || throw(ArgumentError("jacobian_times must be ≥ 1"))
    total = 0
    new_steps = Int[]
    for s in steps_vec
        if total + s >= max_t - 1
            rem = max_t - 1 - total
            rem >= 0 || throw(ArgumentError("jacobian_times must be ≥ 1"))
            rem > 0 && push!(new_steps, rem)
            break
        else
            push!(new_steps, s)
            total += s
        end
    end
    isempty(new_steps) && throw(ArgumentError("jacobian_times must include a time beyond the initial state"))
    return new_steps
end

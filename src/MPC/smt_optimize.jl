"""
    smt_optimize_latent(ds, model, x0, z0, steps,
                        safety_output, terminal_output; kwargs...) -> result

Optimize a flat latent vector `z` with NLopt to minimize the SMT cost defined over
output and input trajectories (all-time semantics).

# Arguments
- `ds::DiscreteRandomSystem`: system to roll out.
- `model`: [`InvertibleCoupling`](@ref), [`NormalizingFlow`](@ref), or `model(x, z) -> u`.
- `x0::AbstractVector`: initial state vector.
- `z0::AbstractVector`: initial latent vector (flat).
- `steps::Union{Integer,AbstractVector{<:Integer}}`: horizon splits; see [`trajectory`](@ref).
- `safety_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs.
- `terminal_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs (eventual).

# Keyword Arguments
- `algo=:LN_BOBYQA`: NLopt algorithm symbol.
- `max_time=Inf`: NLopt `maxtime` (seconds).
- `max_eval=0`: NLopt `maxeval` (0 means no limit).
- `max_penalty_evals=0`: soft cap on total SMT penalty evaluations (0 means no limit).
- `seed=rand(1:10000)`: NLopt RNG seed.
- `safety_input=nothing`: disjunct matrices acting on inputs; defaults to bounds from `ds.U`.
- `u_len=nothing`: forwarded to [`trajectory`](@ref) to slice decoded output.
- `output_map=identity`: mapping from state to output for SMT evaluation.
- `latent_dim=nothing`: required only when `model` is a function.

# Returns
Named tuple:
- `objective`: best SMT cost.
- `objective_time_bounded`: best SMT cost observed before the `max_time` cutoff.
- `z_time_bounded`: latent vector (flat) at the time-bounded best objective.
- `evals_to_zero`: number of NLopt objective evaluations until penalty <= 0 (Inf if never).
- `evals_to_zero_penalty`: estimated number of SMT penalty evaluations until penalty <= 0.
- `z`: best latent vector (flat).
- `result`: NLopt return code.
- `output_trajectory`: output trajectory for the best `z`.
- `input_trajectory`: input trajectory for the best `z`.
"""
function smt_optimize_latent(ds::DiscreteRandomSystem,
                             model,
                             x0,
                             z0,
                             steps,
                             safety_output::AbstractVector{<:AbstractMatrix},
                             terminal_output::AbstractVector{<:AbstractMatrix};
                             safety_input::Union{Nothing,AbstractVector{<:AbstractMatrix}}=nothing,
                             algo::Symbol=:LN_BOBYQA,
                             max_time::Real=Inf,
                             max_eval::Integer=0,
                             max_penalty_evals::Integer=0,
                             seed::Integer=rand(1:10000),
                             u_len=nothing,
                             output_map::Function=identity,
                             latent_dim::Union{Nothing,Integer}=nothing)
    steps_vec = steps isa Integer ? [Int(steps)] : Int.(collect(steps))
    length(steps_vec) >= 1 || throw(ArgumentError("steps must contain at least one segment"))

    u_len_final = _infer_u_len(ds, u_len)

    model_eff = if model isa Function
        latent_dim === nothing && throw(ArgumentError("latent_dim must be provided when model is a function"))
        LatentPolicy(model, Int(latent_dim))
    else
        model
    end

    l = _latent_dim(model_eff) * length(steps_vec)
    z0_vec = z0 isa AbstractVector ? z0 : vec(z0)
    length(z0_vec) == l || throw(DimensionMismatch("z0 must have length $l; got length=$(length(z0_vec))"))

    use_grad = _nlopt_uses_grad(algo)
    safety_input_eff = safety_input === nothing ? _input_bounds_constraints(ds) : safety_input

    t0 = Ref{Union{Nothing,Float64}}(nothing)
    best_val = Ref(Float64(Inf))
    best_z_time = Ref(copy(z0_vec))
    eval_count = 0
    evals_to_zero = Ref(Float64(Inf))
    total_penalty_evals = 0
    opt_ref = Ref{NLopt.Opt}()
    function my_objective_fn(z::Vector{Float64}, grad::Vector{Float64})
        eval_cost = use_grad ? (length(z0_vec) + 1) : 1
        if max_penalty_evals > 0 && total_penalty_evals + eval_cost > max_penalty_evals
            if !isempty(grad)
                grad .= 0.0
            end
            if opt_ref[] !== nothing
                NLopt.force_stop!(opt_ref[])
            end
            return best_val[]
        end
        eval_count += 1
        total_penalty_evals += eval_cost
        if t0[] === nothing
            t0[] = time()
        end
        z32 = Float32.(z)
        res = smt_penalty(ds, model_eff, x0, z32, steps_vec,
                          safety_output, terminal_output;
                          safety_input=safety_input_eff,
                          u_len=u_len_final,
                          output_map=output_map,
                          return_grad=use_grad)
        if evals_to_zero[] == Inf && res.penalty <= 0
            evals_to_zero[] = eval_count
        end
        val = Float64(res.penalty)
        if best_val[] == Inf || (t0[] !== nothing && time() - t0[] <= max_time)
            if val < best_val[]
                best_val[] = val
                best_z_time[] = copy(z)
            end
        end
        if !isempty(grad)
            if res.grad === nothing
                grad .= 0.0
            else
                grad .= Float64.(res.grad)
            end
        end
        return Float64(res.penalty)
    end

    opt = NLopt.Opt(algo, length(z0_vec))
    opt_ref[] = opt
    NLopt.min_objective!(opt, my_objective_fn)
    NLopt.stopval!(opt, 0)
    NLopt.maxtime!(opt, max_time)
    NLopt.maxeval!(opt, max_eval)
    NLopt.srand(seed)

    min_f, min_z, ret = NLopt.optimize(opt, z0_vec)
    best = smt_penalty(ds, model_eff, x0, Float32.(min_z), steps_vec,
                       safety_output, terminal_output;
                       safety_input=safety_input_eff,
                       u_len=u_len_final,
                       output_map=output_map)

    return (; objective=min_f,
             objective_time_bounded=best_val[],
             z_time_bounded=best_z_time[],
             evals_to_zero=evals_to_zero[],
             evals_to_zero_penalty=use_grad ?
                 (isfinite(evals_to_zero[]) ? (evals_to_zero[] * (length(z0_vec) + 1) - length(z0_vec)) : evals_to_zero[]) :
                 evals_to_zero[],
             z=min_z,
             result=ret,
             output_trajectory=best.output_trajectory,
             input_trajectory=best.input_trajectory)
end

"""
    smt_mpc(ds, model, x0, steps,
            safety_output, terminal_output; kwargs...) -> result

Receding-horizon MPC that minimizes the SMT cost at each step using
[`smt_optimize_latent`](@ref), then applies the first control from the optimized latent.

# Arguments
- `ds::DiscreteRandomSystem`: system to roll out.
- `model`: [`InvertibleCoupling`](@ref), [`NormalizingFlow`](@ref), or `model(x, z) -> u`.
- `x0::AbstractVector`: initial state vector.
- `steps`: MPC duration. If an integer, MPC runs for `steps` iterations. If a vector, MPC runs for `sum(steps)`.
- `safety_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs.
- `terminal_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs (eventual).

# Keyword Arguments
- `safety_input=nothing`: disjunct matrices acting on inputs; defaults to bounds from `ds.U`.
- `algo=:LN_PRAXIS`: NLopt algorithm symbol.
- `init_z=nothing`: warm-start latent vector (flat). Defaults to zeros.
- `opt_steps=steps`: horizon specification passed to [`smt_optimize_latent`](@ref).
- `opt_seed=rand(1:10000)`: NLopt seed for each optimization call.
- `max_time=Inf`: NLopt `maxtime` passed to each optimization call.
- `max_eval=0`: NLopt `maxeval` passed to each optimization call.
- `max_penalty_evals=0`: soft cap on total SMT penalty evaluations per optimization call.
- `u_len=nothing`: control dimension; inferred from `ds.U` if omitted.
- `latent_dim=nothing`: required only when `model` is a function.
- `output_map=identity`: mapping from state to output used for SMT evaluation.

# Returns
Named tuple:
- `output_trajectory`: output trajectory matrix (columns over time).
- `state_trajectory`: state trajectory matrix (columns over time).
- `input_trajectory`: input matrix `u_len×steps_total`.
- `objectives`: vector of per-step SMT objectives.
- `objective_total`: SMT objective over the full MPC rollout.
- `z`: final latent vector (flat) from the last optimization.
"""
function smt_mpc(ds::DiscreteRandomSystem,
                 model,
                 x0,
                 steps,
                 safety_output::AbstractVector{<:AbstractMatrix},
                 terminal_output::AbstractVector{<:AbstractMatrix};
                 safety_input::Union{Nothing,AbstractVector{<:AbstractMatrix}}=nothing,
                 algo::Symbol=:LN_PRAXIS,
                 init_z=nothing,
                 opt_steps=steps,
                 opt_seed::Integer=rand(1:10000),
                 max_time::Real=Inf,
                 max_eval::Integer=0,
                 max_penalty_evals::Integer=0,
                 u_len=nothing,
                 latent_dim::Union{Nothing,Integer}=nothing,
                 output_map::Function=identity)
    steps_total = steps isa Integer ? Int(steps) : sum(Int.(collect(steps)))
    steps_total >= 1 || throw(ArgumentError("steps must be ≥ 1 (or sum to ≥ 1)"))

    u_len_final = _infer_u_len(ds, u_len)
    opt_steps_vec = opt_steps isa Integer ? [Int(opt_steps)] : Int.(collect(opt_steps))
    length(opt_steps_vec) >= 1 || throw(ArgumentError("opt_steps must contain at least one segment"))

    model_eff = if model isa Function
        latent_dim === nothing && throw(ArgumentError("latent_dim must be provided when model is a function"))
        LatentPolicy(model, Int(latent_dim))
    else
        model
    end

    l = _latent_dim(model_eff) * length(opt_steps_vec)
    z = init_z === nothing ? zeros(Float32, l) : (init_z isa AbstractVector ? init_z : vec(init_z))
    length(z) == l || throw(DimensionMismatch("init_z must have length $l (model.dim * length(opt_steps)); got length=$(length(z))"))

    x0_vec = x0 isa AbstractVector ? x0 : vec(x0)
    strj = reshape(x0_vec, :, 1)
    utrj = Matrix{Float32}(undef, u_len_final, 0)
    objectives = Float64[]

    for _ in 1:steps_total
        x = strj[:, end]
        res = smt_optimize_latent(ds, model_eff, x, z, opt_steps_vec,
                                  safety_output, terminal_output;
                                  safety_input=safety_input,
                                  algo=algo,
                                  max_time=max_time,
                                  max_eval=max_eval,
                                  max_penalty_evals=max_penalty_evals,
                                  seed=opt_seed,
                                  u_len=u_len_final,
                                  output_map=output_map)
        push!(objectives, Float64(res.objective_time_bounded))
        z = res.z_time_bounded

        u = control_from_latent(model_eff, z[1:_latent_dim(model_eff)], x; u_len=u_len_final)
        u_used = Float32.(u isa AbstractVector ? u : vec(u))
        utrj = hcat(utrj, u_used[1:u_len_final])

        x_next = ds(x, u_used[1:u_len_final])
        strj = hcat(strj, x_next)
    end

    output_trj = output_map === identity ? strj : _apply_output_map(output_map, strj)
    safety_input_eff = safety_input === nothing ? _input_bounds_constraints(ds) : safety_input

    safety_output_vals = Vector{Vector{Float32}}()
    for mat in safety_output
        for t in 1:size(output_trj, 2)
            push!(safety_output_vals, _matrix_row_values(mat, output_trj[:, t]))
        end
    end

    safety_input_vals = Vector{Vector{Float32}}()
    for mat in safety_input_eff
        for t in 1:size(utrj, 2)
            push!(safety_input_vals, _matrix_row_values(mat, utrj[:, t]))
        end
    end

    objective_total = _smt_penalty(safety_output_vals) +
                      _smt_penalty(safety_input_vals) +
                      _eventual_penalty(terminal_output, output_trj)
    return (; output_trajectory=output_trj,
             state_trajectory=strj,
             input_trajectory=utrj,
             objectives,
             objective_total,
             z)
end

"""
    smt_penalty(ds, model, x0, z, steps,
                safety_output, terminal_output; kwargs...) -> result

Compute the SMT penalty for a given latent vector, with optional finite-difference gradient.

# Arguments
- `ds::DiscreteRandomSystem`: system to roll out.
- `model`: [`InvertibleCoupling`](@ref), [`NormalizingFlow`](@ref), or `model(x, z) -> u`.
- `x0::AbstractVector`: initial state vector.
- `z::AbstractVector`: latent vector (flat).
- `steps::Union{Integer,AbstractVector{<:Integer}}`: horizon splits; see [`trajectory`](@ref).
- `safety_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs.
- `terminal_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs (eventual).

# Keyword Arguments
- `safety_input=nothing`: disjunct matrices acting on inputs; defaults to bounds from `ds.U`.
- `u_len=nothing`: forwarded to [`trajectory`](@ref) to slice decoded output.
- `output_map=identity`: mapping from state to output for SMT evaluation.
- `latent_dim=nothing`: required only when `model` is a function.
- `return_grad=false`: whether to compute a finite-difference gradient.
- `grad_eps=1f-6`: perturbation size for finite differences.

# Returns
Named tuple:
- `penalty`: SMT penalty scalar.
- `grad`: gradient vector (or `nothing` when `return_grad=false`).
- `output_trajectory`: output trajectory matrix (columns over time).
- `input_trajectory`: input trajectory matrix.
"""
function smt_penalty(ds::DiscreteRandomSystem,
                     model,
                     x0,
                     z,
                     steps,
                     safety_output::AbstractVector{<:AbstractMatrix},
                     terminal_output::AbstractVector{<:AbstractMatrix};
                     safety_input::Union{Nothing,AbstractVector{<:AbstractMatrix}}=nothing,
                     u_len=nothing,
                     output_map::Function=identity,
                     latent_dim::Union{Nothing,Integer}=nothing,
                     return_grad::Bool=false,
                     grad_eps::Real=1f-6)
    steps_vec = steps isa Integer ? [Int(steps)] : Int.(collect(steps))
    length(steps_vec) >= 1 || throw(ArgumentError("steps must contain at least one segment"))

    u_len_final = _infer_u_len(ds, u_len)
    model_eff = if model isa Function
        latent_dim === nothing && throw(ArgumentError("latent_dim must be provided when model is a function"))
        LatentPolicy(model, Int(latent_dim))
    else
        model
    end

    l = _latent_dim(model_eff) * length(steps_vec)
    z_vec = z isa AbstractVector ? z : vec(z)
    length(z_vec) == l || throw(DimensionMismatch("z must have length $l; got length=$(length(z_vec))"))

    safety_input_eff = safety_input === nothing ? _input_bounds_constraints(ds) : safety_input

    penalty_for(z_local::AbstractVector) = begin
        z32 = eltype(z_local) === Float32 ? z_local : Float32.(z_local)
        res = trajectory(ds, model_eff, x0, z32, steps_vec;
                         u_len=u_len_final,
                         output_map=output_map)
        ytrj = res.output_trajectory
        utrj = res.input_trajectory

        safety_output_vals = Vector{Vector{Float32}}()
        for mat in safety_output
            for t in 1:size(ytrj, 2)
                push!(safety_output_vals, _matrix_row_values(mat, ytrj[:, t]))
            end
        end

        safety_input_vals = Vector{Vector{Float32}}()
        for mat in safety_input_eff
            for t in 1:size(utrj, 2)
                push!(safety_input_vals, _matrix_row_values(mat, utrj[:, t]))
            end
        end

        penalty = _smt_penalty(safety_output_vals) +
                  _smt_penalty(safety_input_vals) +
                  _eventual_penalty(terminal_output, ytrj)
        return penalty, ytrj, utrj
    end

    penalty0, ytrj0, utrj0 = penalty_for(z_vec)
    if !return_grad
        return (; penalty=penalty0,
                 grad=nothing,
                 output_trajectory=ytrj0,
                 input_trajectory=utrj0)
    end

    eps = Float32(grad_eps)
    grad = zeros(Float32, length(z_vec))
    z_work = Float32.(z_vec)
    for i in eachindex(z_work)
        z_work[i] += eps
        penalty_i, _, _ = penalty_for(z_work)
        grad[i] = (Float32(penalty_i) - Float32(penalty0)) / eps
        z_work[i] -= eps
    end

    return (; penalty=penalty0,
             grad,
             output_trajectory=ytrj0,
             input_trajectory=utrj0)
end

"""
    smt_optimize_latent(ds, model, x0, z0, steps,
                        safety_output, safety_input, terminal_output; kwargs...) -> result

Optimize a flat latent vector `z` with NLopt to minimize the SMT cost defined over
output and input trajectories (all-time semantics).

# Arguments
- `ds::DiscreteRandomSystem`: system to roll out.
- `model`: [`InvertibleCoupling`](@ref), [`NormalizingFlow`](@ref), or `model(x, z) -> u`.
- `x0::AbstractVector`: initial state vector.
- `z0::AbstractVector`: initial latent vector (flat).
- `steps::Union{Integer,AbstractVector{<:Integer}}`: horizon splits; see [`trajectory`](@ref).
- `safety_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs.
- `safety_input::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on inputs.
- `terminal_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs (eventual).

# Keyword Arguments
- `algo=:LN_BOBYQA`: NLopt algorithm symbol.
- `max_time=Inf`: NLopt `maxtime` (seconds).
- `seed=rand(1:10000)`: NLopt RNG seed.
- `u_len=nothing`: forwarded to [`trajectory`](@ref) to slice decoded output.
- `output_map=identity`: mapping from state to output for SMT evaluation.
- `latent_dim=nothing`: required only when `model` is a function.

# Returns
Named tuple:
- `objective`: best SMT cost.
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
                             safety_input::AbstractVector{<:AbstractMatrix},
                             terminal_output::AbstractVector{<:AbstractMatrix};
                             algo::Symbol=:LN_BOBYQA,
                             max_time::Real=Inf,
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
    use_grad && throw(ArgumentError("Derivative-based NLopt algorithms are not supported; use a derivative-free algo."))

    objective_scalar(z::AbstractVector) = begin
        z32 = eltype(z) === Float32 ? z : Float32.(z)
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
        for mat in safety_input
            for t in 1:size(utrj, 2)
                push!(safety_input_vals, _matrix_row_values(mat, utrj[:, t]))
            end
        end

        return _smt_penalty(safety_output_vals) +
               _smt_penalty(safety_input_vals) +
               _eventual_penalty(terminal_output, ytrj)
    end

    function my_objective_fn(z::AbstractVector, grad::AbstractVector)
        return Float64(objective_scalar(z))
    end

    opt = NLopt.Opt(algo, length(z0_vec))
    NLopt.min_objective!(opt, my_objective_fn)
    NLopt.stopval!(opt, 0)
    NLopt.maxtime!(opt, max_time)
    NLopt.srand(seed)

    min_f, min_z, ret = NLopt.optimize(opt, z0_vec)
    best = trajectory(ds, model_eff, x0, Float32.(min_z), steps_vec;
                      u_len=u_len_final,
                      output_map=output_map)

    return (; objective=min_f,
             z=min_z,
             result=ret,
             output_trajectory=best.output_trajectory,
             input_trajectory=best.input_trajectory)
end

"""
    smt_mpc(ds, model, x0, steps,
            safety_output, safety_input, terminal_output; kwargs...) -> result

Receding-horizon MPC that minimizes the SMT cost at each step using
[`smt_optimize_latent`](@ref), then applies the first control from the optimized latent.

# Arguments
- `ds::DiscreteRandomSystem`: system to roll out.
- `model`: [`InvertibleCoupling`](@ref), [`NormalizingFlow`](@ref), or `model(x, z) -> u`.
- `x0::AbstractVector`: initial state vector.
- `steps`: MPC duration. If an integer, MPC runs for `steps` iterations. If a vector, MPC runs for `sum(steps)`.
- `safety_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs.
- `safety_input::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on inputs.
- `terminal_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs (eventual).

# Keyword Arguments
- `algo=:LN_PRAXIS`: NLopt algorithm symbol.
- `init_z=nothing`: warm-start latent vector (flat). Defaults to zeros.
- `opt_steps=steps`: horizon specification passed to [`smt_optimize_latent`](@ref).
- `opt_seed=rand(1:10000)`: NLopt seed for each optimization call.
- `max_time=Inf`: NLopt `maxtime` passed to each optimization call.
- `u_len=nothing`: control dimension; inferred from `ds.U` if omitted.
- `latent_dim=nothing`: required only when `model` is a function.
- `output_map=identity`: mapping from state to output used for SMT evaluation.

# Returns
Named tuple:
- `output_trajectory`: output trajectory matrix (columns over time).
- `input_trajectory`: input matrix `u_len×steps_total`.
- `objectives`: vector of per-step SMT objectives.
- `z`: final latent vector (flat) from the last optimization.
"""
function smt_mpc(ds::DiscreteRandomSystem,
                 model,
                 x0,
                 steps,
                 safety_output::AbstractVector{<:AbstractMatrix},
                 safety_input::AbstractVector{<:AbstractMatrix},
                 terminal_output::AbstractVector{<:AbstractMatrix};
                 algo::Symbol=:LN_PRAXIS,
                 init_z=nothing,
                 opt_steps=steps,
                 opt_seed::Integer=rand(1:10000),
                 max_time::Real=Inf,
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
                                  safety_output, safety_input, terminal_output;
                                  algo=algo,
                                  max_time=max_time,
                                  seed=opt_seed,
                                  u_len=u_len_final,
                                  output_map=output_map)
        push!(objectives, Float64(res.objective))
        z = res.z

        u = control_from_latent(model_eff, z[1:_latent_dim(model_eff)], x; u_len=u_len_final)
        u_used = Float32.(u isa AbstractVector ? u : vec(u))
        utrj = hcat(utrj, u_used[1:u_len_final])

        x_next = ds(x, u_used[1:u_len_final])
        strj = hcat(strj, x_next)
    end

    output_trj = output_map === identity ? strj : _apply_output_map(output_map, strj)
    return (; output_trajectory=output_trj,
             input_trajectory=utrj,
             objectives,
             z)
end

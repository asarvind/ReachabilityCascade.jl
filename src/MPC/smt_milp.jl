import JuMP
import HiGHS

_scale_flat(steps, z_len::Int, scale_steps::Union{Nothing,AbstractVector{<:Real}}) = begin
    segments = steps isa Integer ? 1 : length(steps)
    segments >= 1 || throw(ArgumentError("steps must contain at least one segment"))

    if scale_steps === nothing
        return ones(Float32, z_len)
    end
    length(scale_steps) == segments ||
        throw(DimensionMismatch("scale_steps must have length $segments; got length=$(length(scale_steps))"))
    z_len % segments == 0 ||
        throw(DimensionMismatch("z length $z_len must be divisible by segments $segments"))
    latent_dim = Int(z_len ÷ segments)
    return repeat(Float32.(scale_steps), inner=latent_dim)
end

"""
    smt_milp_iterative(ds, model, x0, z_ref, steps,
                       safety_output, terminal_output; kwargs...) -> result

Iteratively linearize and solve the SMT MILP, updating `z_ref` each iteration.
Stops when the SMT evaluations are satisfied or the iteration limit is reached.

# Arguments
- `ds::DiscreteRandomSystem`: system to roll out.
- `model`: [`InvertibleCoupling`](@ref), [`NormalizingFlow`](@ref), or `model(x, z) -> u`.
- `x0::AbstractVector`: initial state vector.
- `z_ref::AbstractVector`: initial reference latent vector.
- `steps::Union{Integer,AbstractVector{<:Integer}}`: horizon splits; see [`trajectory`](@ref).
- `safety_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs.
- `terminal_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on final output.

# Keyword Arguments
- `u_len=nothing`: control dimension forwarded to [`trajectory`](@ref).
- `latent_dim=nothing`: required only when `model` is a function.
- `output_map=identity`: output map forwarded to [`trajectory`](@ref).
- `eps=1f-6`: finite-difference step used for linearization.
- `big_m=1e4`: big-M constant for disjunction encoding.
- `optimizer=HiGHS.Optimizer`: solver constructor.
- `silent=true`: if `true`, suppress solver output.
- `safety_input=nothing`: disjunct matrices acting on inputs; defaults to bounds from `ds.U`.
- `scale_steps=nothing`: optional per-segment scaling of the L1 objective.
- `max_iters=5`: maximum number of linearization/solve iterations.
- `linearization=:critical`: `:critical` uses critical-time linearization; `:all` uses all-time.

# Returns
Named tuple:
- `z`: final latent vector.
- `info`: solver info from the last MILP.
- `satisfied`: whether the SMT evaluations were satisfied at the final iterate.
- `iters`: number of iterations performed.
"""
function smt_milp_iterative(ds::DiscreteRandomSystem,
                            model,
                            x0,
                            z_ref,
                            steps,
                            safety_output::AbstractVector{<:AbstractMatrix},
                            terminal_output::AbstractVector{<:AbstractMatrix};
                            u_len=nothing,
                            safety_input::Union{Nothing,AbstractVector{<:AbstractMatrix}}=nothing,
                            latent_dim::Union{Nothing,Integer}=nothing,
                            output_map::Function=identity,
                            eps::Real=1f-6,
                            scale_steps::Union{Nothing,AbstractVector{<:Real}}=nothing,
                            big_m::Real=1e4,
                            optimizer=HiGHS.Optimizer,
                            silent::Bool=true,
                            max_iters::Integer=5,
                            linearization::Symbol=:critical)
    z_curr = z_ref isa AbstractVector ? copy(z_ref) : vec(z_ref)
    safety_input_eff = safety_input === nothing ? _input_bounds_constraints(ds) : safety_input

    satisfied = false
    info = (; termination_status=nothing, primal_status=nothing, feasible=false)
    last_evals = nothing

    if !(linearization in (:critical, :all))
        throw(ArgumentError("linearization must be :critical or :all; got $linearization"))
    end

    affine_fn = linearization == :critical ? smt_affine_critical : smt_affine_all
    eval_fn = linearization == :critical ? smt_critical_evaluations : smt_all_evaluations

    for iter in 1:max_iters
        affine_res = affine_fn(ds,
                               model,
                               x0,
                               z_curr,
                               steps,
                               safety_output,
                               terminal_output;
                               u_len=u_len,
                               safety_input=safety_input_eff,
                               latent_dim=latent_dim,
                               output_map=output_map,
                               eps=eps,
                               return_base=true)
        affine = affine_res.affine
        base = affine_res.base

        group_ok(vals_group) = all(minimum(vals) <= 0 for vals in vals_group)
        satisfied = group_ok(base.safety_output) &&
                    group_ok(base.safety_input) &&
                    group_ok(base.terminal_output)
        last_evals = base
        if satisfied
            return (; z=z_curr, info, satisfied=true, iters=iter, evaluations=base)
        end

        z_sol, info = _solve_affine_milp(affine,
                                         z_curr,
                                         steps;
                                         scale_steps=scale_steps,
                                         big_m=big_m,
                                         optimizer=optimizer,
                                         silent=silent,
                                         z_start=z_curr)
        z_sol === nothing && return (; z=nothing, info, satisfied=false, iters=iter, evaluations=base)
        z_curr = z_sol
    end

    final_base = eval_fn(ds,
                         model,
                         x0,
                         z_curr,
                         steps,
                         safety_output,
                         terminal_output;
                         u_len=u_len,
                         safety_input=safety_input_eff,
                         latent_dim=latent_dim,
                         output_map=output_map)
    satisfied = all(minimum(vals) <= 0 for vals in final_base.safety_output) &&
                all(minimum(vals) <= 0 for vals in final_base.safety_input) &&
                all(minimum(vals) <= 0 for vals in final_base.terminal_output)
    return (; z=z_curr, info, satisfied=satisfied, iters=max_iters, evaluations=final_base)
end
_add_affine_disjuncts!(model::JuMP.Model,
                       z::AbstractVector,
                       disjuncts::AbstractVector{<:AbstractMatrix},
                       big_m::Float32;
                       z_start::Union{Nothing,AbstractVector{<:Real}}=nothing) = begin
    z_len = length(z)
    for mat in disjuncts
        R = size(mat, 1)
        y = JuMP.@variable(model, [1:R], Bin)
        JuMP.@constraint(model, sum(y) >= 1)
        if z_start !== nothing
            row_vals = Vector{Float32}(undef, R)
            for i in 1:R
                row = vec(mat[i, :])
                a = Float32.(row[1:z_len])
                b = Float32(row[end])
                row_vals[i] = sum(a .* Float32.(z_start)) + b
            end
            best_idx = argmin(row_vals)
            for i in 1:R
                JuMP.set_start_value(y[i], i == best_idx ? 1.0 : 0.0)
            end
        end
        for i in 1:R
            row = vec(mat[i, :])
            length(row) == z_len + 1 ||
                throw(DimensionMismatch("affine row must have length $(z_len + 1); got length=$(length(row))"))
            a = Float32.(row[1:z_len])
            b = Float32(row[end])
            expr = sum(a[j] * z[j] for j in 1:z_len) + b
            JuMP.@constraint(model, expr <= big_m * (1 - y[i]))
        end
    end
    return nothing
end

_solve_affine_milp(affine::NamedTuple,
                   z_ref_vec::AbstractVector{<:Real},
                   steps;
                   scale_steps::Union{Nothing,AbstractVector{<:Real}}=nothing,
                   big_m::Real=1e4,
                   optimizer=HiGHS.Optimizer,
                   silent::Bool=true,
                   z_start::Union{Nothing,AbstractVector{<:Real}}=nothing) = begin
    z_len = length(z_ref_vec)
    scale_flat = _scale_flat(steps, z_len, scale_steps)

    opt = JuMP.Model(optimizer)
    silent && JuMP.set_silent(opt)

    JuMP.@variable(opt, z[1:z_len])
    JuMP.@variable(opt, s[1:z_len] .>= 0)

    z_ref_f = Float32.(z_ref_vec)
    z_start_f = z_start === nothing ? z_ref_f : Float32.(z_start)
    for i in 1:z_len
        JuMP.set_start_value(z[i], z_start_f[i])
    end

    scale_f = Float32.(scale_flat)
    JuMP.@constraint(opt, scale_f .* (z .- z_ref_f) .<= s)
    JuMP.@constraint(opt, -(scale_f .* (z .- z_ref_f)) .<= s)
    big_m_f = Float32(big_m)
    _add_affine_disjuncts!(opt, z, affine.safety_output, big_m_f; z_start=z_start_f)
    _add_affine_disjuncts!(opt, z, affine.safety_input, big_m_f; z_start=z_start_f)
    _add_affine_disjuncts!(opt, z, affine.terminal_output, big_m_f; z_start=z_start_f)

    JuMP.@objective(opt, Min, sum(s))

    JuMP.optimize!(opt)
    status = JuMP.termination_status(opt)
    primal = JuMP.primal_status(opt)
    feasible = (status == JuMP.OPTIMAL || status == JuMP.FEASIBLE_POINT) &&
               (primal == JuMP.FEASIBLE_POINT || primal == JuMP.NEARLY_FEASIBLE_POINT)

    z_sol = feasible ? Float32.(JuMP.value.(z)) : nothing
    info = (; termination_status=status, primal_status=primal, feasible=feasible)
    return z_sol, info
end

"""
    smt_milp_critical(ds, model, x0, z_ref, steps,
                      safety_output, terminal_output; kwargs...) -> (z, info)

Solve an MILP built from the linearized SMT constraints around `z_ref`.

The SMT linearization is computed by [`smt_affine_critical`](@ref). The objective is
the L1 distance to `z_ref`, implemented with slack variables and linear constraints.

# Arguments
- `ds::DiscreteRandomSystem`: system to roll out.
- `model`: [`InvertibleCoupling`](@ref), [`NormalizingFlow`](@ref), or `model(x, z) -> u`.
- `x0::AbstractVector`: initial state vector.
- `z_ref::AbstractVector`: reference latent vector for linearization and objective.
- `steps::Union{Integer,AbstractVector{<:Integer}}`: horizon splits; see [`trajectory`](@ref).
- `safety_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs.
- `terminal_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on final output.

# Keyword Arguments
- `u_len=nothing`: control dimension forwarded to [`trajectory`](@ref).
- `latent_dim=nothing`: required only when `model` is a function.
- `output_map=identity`: output map forwarded to [`trajectory`](@ref).
- `eps=1f-6`: finite-difference step used for linearization.
- `scale_steps=nothing`: optional per-segment scaling of the L1 objective.
- `penalty_weight=0.9`: weight on SMT violation penalties in the objective (L1 weight is `1 - penalty_weight`).
- `big_m=1e4`: big-M constant for disjunction encoding.
- `optimizer=HiGHS.Optimizer`: solver constructor.
- `silent=true`: if `true`, suppress solver output.
- `safety_input=nothing`: disjunct matrices acting on inputs; defaults to bounds from `ds.U`.

# Returns
- `z`: solution vector if feasible; otherwise `nothing`.
- `info`: named tuple with solver status fields.
"""
function smt_milp_critical(ds::DiscreteRandomSystem,
                           model,
                           x0,
                           z_ref,
                           steps,
                           safety_output::AbstractVector{<:AbstractMatrix},
                           terminal_output::AbstractVector{<:AbstractMatrix};
                           u_len=nothing,
                           safety_input::Union{Nothing,AbstractVector{<:AbstractMatrix}}=nothing,
                           latent_dim::Union{Nothing,Integer}=nothing,
                           output_map::Function=identity,
                           eps::Real=1f-6,
                            scale_steps::Union{Nothing,AbstractVector{<:Real}}=nothing,
                            big_m::Real=1e4,
                            optimizer=HiGHS.Optimizer,
                            silent::Bool=true)
    z_ref_vec = z_ref isa AbstractVector ? z_ref : vec(z_ref)
    z_len = length(z_ref_vec)
    safety_input_eff = safety_input === nothing ? _input_bounds_constraints(ds) : safety_input

    affine = smt_affine_critical(ds,
                                 model,
                                 x0,
                                 z_ref_vec,
                                 steps,
                                 safety_output,
                                 terminal_output;
                                 u_len=u_len,
                                 safety_input=safety_input_eff,
                                 latent_dim=latent_dim,
                                 output_map=output_map,
                                 eps=eps)

    return _solve_affine_milp(affine,
                              z_ref_vec,
                              steps;
                              scale_steps=scale_steps,
                              big_m=big_m,
                              optimizer=optimizer,
                              silent=silent,
                              z_start=z_ref_vec)
end

import JuMP
import HiGHS

_row_affine_parts(row::AbstractVector{<:Real}, n::Int) = begin
    if length(row) == n
        return (Float64.(row), 0.0)
    elseif length(row) == n + 1
        return (Float64.(row[1:n]), Float64(row[end]))
    end
    throw(DimensionMismatch("constraint row must have length $n or $(n + 1); got length=$(length(row))"))
end

_x_lin_at(strj::AbstractMatrix, jacobians::Vector{Matrix{Float64}}, time_to_idx::Dict{Int,Int}, t::Int) = begin
    idx = time_to_idx[t]
    J = jacobians[idx]
    x_ref = Float64.(strj[:, t])
    return x_ref, J
end

_add_disjuncts!(model::JuMP.Model,
                dz::AbstractVector,
                constraints::AbstractVector{<:AbstractMatrix},
                strj::AbstractMatrix,
                jacobians::Vector{Matrix{Float64}},
                time_to_idx::Dict{Int,Int},
                t::Int,
                big_m::Float64) = begin
    x_ref, J = _x_lin_at(strj, jacobians, time_to_idx, t)
    state_dim = length(x_ref)
    z_len = length(dz)

    for mat in constraints
        R = size(mat, 1)
        y = JuMP.@variable(model, [1:R], Bin)
        JuMP.@constraint(model, sum(y) >= 1)
        for i in 1:R
            a, b = _row_affine_parts(vec(mat[i, :]), state_dim)
            expr = b + sum(a[j] * (x_ref[j] + sum(J[j, k] * dz[k] for k in 1:z_len)) for j in 1:state_dim)
            JuMP.@constraint(model, expr <= big_m * (1 - y[i]))
        end
    end
    return nothing
end

"""
    smt_latent(ds, x0, model, z0, steps, safety_constraints, terminal_constraints, safety_times, terminal_time; kwargs...) -> (z, info)

Solve for a flattened latent `z` such that the *linearized* state trajectory satisfies
SMT-style constraints (conjunction of disjunctions of affine inequalities).

Each disjunct is a matrix whose rows encode affine expressions capped by zero:
`a' * x + b <= 0` (row length `n+1`) or `a' * x <= 0` (row length `n`).
A vector of such matrices represents a conjunction (AND) over disjuncts.

The linearization uses a reference `z0` and the Jacobians from [`trajectory`](@ref).

# Arguments
- `ds`: [`DiscreteRandomSystem`](@ref).
- `x0`: initial state vector.
- `model`: model mapping `(x, z) -> u` via [`trajectory`](@ref).
- `z0`: reference flattened latent vector.
- `steps`: horizon definition passed to [`trajectory`](@ref).
- `safety_constraints`: vector of disjunct matrices (OR per matrix, AND across vector).
- `terminal_constraints`: vector of disjunct matrices (OR per matrix, AND across vector).
- `safety_times`: time indices (time 0 is index 0) at which safety constraints are enforced.
- `terminal_time`: terminal time index (time 0 is index 0) for terminal constraints. Defaults to `sum(steps)`.

# Keyword Arguments
- `u_len=nothing`: control dimension passed to [`trajectory`](@ref).
- `output_map=identity`: mapping from state to output used by [`trajectory`](@ref).
- `big_m=1e4`: big-M constant for disjunction encoding.
- `optimizer=HiGHS.Optimizer`: solver constructor.
- `silent=true`: if `true`, suppress solver output.

# Returns
- `z`: feasible `z` if solved; otherwise `nothing`.
- `info`: named tuple with solver status fields.
"""
function smt_latent(ds::DiscreteRandomSystem,
                            x0::AbstractVector,
                            model,
                            z0::AbstractVector,
                            steps::Union{Integer,AbstractVector{<:Integer}},
                            safety_constraints::AbstractVector{<:AbstractMatrix},
                            terminal_constraints::AbstractVector{<:AbstractMatrix},
                            safety_times::AbstractVector{<:Integer},
                            terminal_time::Union{Nothing,Integer}=nothing;
                            u_len=nothing,
                            output_map::Function=identity,
                            big_m::Real=1e4,
                            optimizer=HiGHS.Optimizer,
                            silent::Bool=true)
    big_m_f = Float64(big_m)
    big_m_f > 0 || throw(ArgumentError("big_m must be positive; got $big_m"))

    safety_idxs = Int.(safety_times) .+ 1
    terminal_time_final = terminal_time === nothing ? (steps isa Integer ? Int(steps) : sum(Int.(steps))) : Int(terminal_time)
    terminal_idx = terminal_time_final + 1
    any(t -> t < 1, safety_idxs) && throw(ArgumentError("safety_times must be ≥ 0"))
    terminal_idx >= 1 || throw(ArgumentError("terminal_time must be ≥ 0"))

    jacobian_times = unique(vcat(safety_idxs, terminal_idx))
    res = trajectory(ds, model, x0, z0, steps;
                     u_len=u_len,
                     output_map=output_map,
                     jacobian_times=jacobian_times)
    strj = res.output_trajectory
    jacobians = res.output_jacobians
    state_dim = size(strj, 1)

    time_to_idx = Dict(t => i for (i, t) in pairs(jacobian_times))
    terminal_idx <= size(strj, 2) || throw(ArgumentError("terminal_time is beyond trajectory length"))
    all(t -> t <= size(strj, 2), safety_idxs) ||
        throw(ArgumentError("safety_times contain indices beyond trajectory length"))

    z_len = length(z0)
    model = JuMP.Model(optimizer)
    if silent
        JuMP.set_silent(model)
    end
    JuMP.@variable(model, dz[1:z_len])

    for t in safety_idxs
        _add_disjuncts!(model, dz, safety_constraints, strj, jacobians, time_to_idx, t, big_m_f)
    end
    _add_disjuncts!(model, dz, terminal_constraints, strj, jacobians, time_to_idx, terminal_idx, big_m_f)

    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    primal = JuMP.primal_status(model)
    feasible = (status == JuMP.OPTIMAL || status == JuMP.FEASIBLE_POINT) &&
               (primal == JuMP.FEASIBLE_POINT || primal == JuMP.NEARLY_FEASIBLE_POINT)

    z = if feasible
        Float64.(z0) .+ Float64.(JuMP.value.(dz))
    else
        nothing
    end

    info = (; termination_status=status, primal_status=primal, feasible=feasible)
    return z, info
end

_apply_output_map_vec(output_map::Function, x::AbstractVector) = begin
    y = output_map(x)
    return y isa AbstractVector ? y : vec(y)
end

_update_steps_vec(steps_vec::Vector{Int}) = begin
    steps_next = copy(steps_vec)
    steps_next[1] -= 1
    if steps_next[1] <= 0
        return steps_next[2:end], true
    end
    return steps_next, false
end

_update_z_ref(z_ref_vec::AbstractVector{<:Real},
              z_sol::AbstractVector{<:Real},
              latent_dim::Int,
              dropped::Bool,
              steps_len::Int) = begin
    if dropped
        if steps_len == 0
            return Float32[]
        end
        return Float32.(z_sol[(latent_dim + 1):end])
    end
    return Float32.(z_sol)
end

"""
    smt_milp_receding(ds, model, x0, z_ref, steps,
                      safety_output, terminal_output; kwargs...) -> result

Receding-horizon SMT optimization using MILP linearization at each step.

At each iteration, an SMT MILP is solved around the current `z_ref`. The first
input implied by the solution is applied, the state is advanced, and the horizon
is reduced by one.

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
- `output_map=identity`: output map used for the returned trajectory.
- `eps=1f-6`: finite-difference step used for linearization.
- `scale_steps=nothing`: optional per-segment scaling of the L1 objective.
- `big_m=1e4`: big-M constant for disjunction encoding.
- `optimizer=HiGHS.Optimizer`: solver constructor.
- `silent=true`: if `true`, suppress solver output.
- `safety_input=nothing`: disjunct matrices acting on inputs; defaults to bounds from `ds.U`.
- `linearization=:critical`: `:critical` uses critical-time linearization; `:all` uses all-time.
- `update_z_ref=true`: if `true`, updates the reference latent between steps.

# Returns
Named tuple:
- `output_trajectory`: output matrix `p×(1+steps_total)` with the first column equal to `output_map(x0)`.
- `input_trajectory`: control matrix `u_len×steps_total` with one column per applied input.
- `z_sequence`: latent matrix `latent_dim×steps_total` of applied latents.
- `satisfied`: whether the SMT evaluations were satisfied for the final trajectory.
- `evaluations`: SMT row evaluations from [`smt_critical_evaluations`](@ref) for the final trajectory.
- `infos`: vector of solver info tuples, one per iteration.
"""
function smt_milp_receding(ds::DiscreteRandomSystem,
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
                           linearization::Symbol=:critical,
                           update_z_ref::Bool=true)
    steps_vec = steps isa Integer ? [Int(steps)] : Int.(collect(steps))
    length(steps_vec) >= 1 || throw(ArgumentError("steps must contain at least one segment"))

    u_len_final = _infer_u_len(ds, u_len)
    model_eff = if model isa Function
        latent_dim === nothing && throw(ArgumentError("latent_dim must be provided when model is a function"))
        LatentPolicy(model, Int(latent_dim))
    else
        model
    end
    latent_dim_final = _latent_dim(model_eff)
    z_ref_vec = z_ref isa AbstractVector ? Float32.(z_ref) : Float32.(vec(z_ref))
    z_ref_init = copy(z_ref_vec)

    if !(linearization in (:critical, :all))
        throw(ArgumentError("linearization must be :critical or :all; got $linearization"))
    end
    milp_fn = linearization == :critical ? smt_milp_critical : smt_milp_all

    safety_input_eff = safety_input === nothing ? _input_bounds_constraints(ds) : safety_input

    x_curr = x0 isa AbstractVector ? Float32.(x0) : Float32.(vec(x0))
    y0 = _apply_output_map_vec(output_map, x_curr)
    ytrj = reshape(Float32.(y0), :, 1)
    utrj = Matrix{Float32}(undef, u_len_final, 0)
    ztrj = Matrix{Float32}(undef, latent_dim_final, 0)

    infos = Vector{NamedTuple}(undef, 0)

    while !isempty(steps_vec) && sum(steps_vec) > 0
        z_sol, info = milp_fn(ds,
                              model,
                              x_curr,
                              z_ref_vec,
                              steps_vec,
                              safety_output,
                              terminal_output;
                              u_len=u_len_final,
                              safety_input=safety_input_eff,
                              latent_dim=latent_dim,
                              output_map=output_map,
                              eps=eps,
                              scale_steps=scale_steps,
                              big_m=big_m,
                              optimizer=optimizer,
                              silent=silent)
        push!(infos, info)
        z_sol === nothing && break

        zmat = reshape(Float32.(z_sol), latent_dim_final, length(steps_vec))
        z_first = zmat[:, 1]
        u = control_from_latent(model_eff, z_first, x_curr; u_len=u_len_final)
        u_vec = u isa AbstractVector ? u : vec(u)
        u_used = Float32.(u_vec[1:u_len_final])

        utrj = hcat(utrj, u_used)
        ztrj = hcat(ztrj, z_first)
        x_next = ds(x_curr, u_used)
        x_curr = Float32.(x_next)

        y_next = _apply_output_map_vec(output_map, x_curr)
        ytrj = hcat(ytrj, Float32.(y_next))

        steps_vec, dropped = _update_steps_vec(steps_vec)
        if update_z_ref
            z_ref_vec = _update_z_ref(z_ref_vec, z_sol, latent_dim_final, dropped, length(steps_vec))
        else
            new_len = latent_dim_final * length(steps_vec)
            if new_len == 0
                z_ref_vec = Float32[]
            else
                z_ref_vec = z_ref_init[1:new_len]
            end
        end
    end

    safety_output_vals = [ _critical_row_values(mat, ytrj) for mat in safety_output ]
    safety_input_vals = if size(utrj, 2) == 0
        [Float32[-Inf] for _ in safety_input_eff]
    else
        [ _critical_row_values(mat, utrj) for mat in safety_input_eff ]
    end
    terminal_output_vals = Vector{Vector{Float32}}(undef, length(terminal_output))
    y_final = ytrj[:, end]
    for (i, mat) in pairs(terminal_output)
        terminal_output_vals[i] = _matrix_row_values(mat, y_final)
    end

    evaluations = (; safety_output=safety_output_vals,
                    safety_input=safety_input_vals,
                    terminal_output=terminal_output_vals)

    satisfied = all(minimum(vals) <= 0 for vals in evaluations.safety_output) &&
                all(minimum(vals) <= 0 for vals in evaluations.safety_input) &&
                all(minimum(vals) <= 0 for vals in evaluations.terminal_output)

    return (; output_trajectory=ytrj,
             input_trajectory=utrj,
             z_sequence=ztrj,
             satisfied=satisfied,
             evaluations=evaluations,
             infos=infos)
end

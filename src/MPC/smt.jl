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

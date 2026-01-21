import LazySets
import LazySets: center, radius_hyperrectangle

_row_affine_value(row::AbstractVector{<:Real}, v::AbstractVector{<:Real}) = begin
    n = length(v)
    if length(row) == n
        return sum(Float32.(row) .* Float32.(v))
    elseif length(row) == n + 1
        return sum(Float32.(row[1:n]) .* Float32.(v)) + Float32(row[end])
    end
    throw(DimensionMismatch("constraint row must have length $n or $(n + 1); got length=$(length(row))"))
end


_matrix_row_values(mat::AbstractMatrix{<:Real}, v::AbstractVector{<:Real}) = begin
    vals = Vector{Float32}(undef, size(mat, 1))
    for i in 1:size(mat, 1)
        vals[i] = _row_affine_value(vec(mat[i, :]), v)
    end
    return vals
end


_critical_row_values(mat::AbstractMatrix{<:Real}, traj::AbstractMatrix{<:Real}) = begin
    best_vals = _matrix_row_values(mat, traj[:, 1])
    best_score = minimum(best_vals)

    for t in 2:size(traj, 2)
        vals = _matrix_row_values(mat, traj[:, t])
        score = minimum(vals)
        if score > best_score
            best_score = score
            best_vals = vals
        end
    end
    return best_vals
end


_input_bounds_constraints(ds::DiscreteRandomSystem) = begin
    hasproperty(ds, :U) || throw(ArgumentError("ds must have field U to infer input bounds"))
    U = getproperty(ds, :U)
    U isa LazySets.Hyperrectangle || throw(ArgumentError("ds.U must be a Hyperrectangle to infer bounds"))

    u_center = center(U)
    u_radius = radius_hyperrectangle(U)
    u_lo = u_center .- u_radius
    u_hi = u_center .+ u_radius
    u_dim = length(u_center)

    mats = Matrix{Float32}[]
    for i in 1:u_dim
        row_hi = zeros(Float32, u_dim + 1)
        row_hi[i] = 1.0f0
        row_hi[end] = -Float32(u_hi[i])
        push!(mats, reshape(row_hi, 1, :))

        row_lo = zeros(Float32, u_dim + 1)
        row_lo[i] = -1.0f0
        row_lo[end] = Float32(u_lo[i])
        push!(mats, reshape(row_lo, 1, :))
    end
    return mats
end


_affine_group(vals_group::Vector{Vector{Float32}},
              extract_group::Function,
              ds::DiscreteRandomSystem,
              model,
              x0,
              z_ref_vec::AbstractVector{<:Real},
              steps,
              safety_output::AbstractVector{<:AbstractMatrix},
              safety_input::AbstractVector{<:AbstractMatrix},
              terminal_output::AbstractVector{<:AbstractMatrix};
              eval_fn::Function=smt_critical_evaluations,
              u_len=nothing,
              latent_dim::Union{Nothing,Integer}=nothing,
              output_map::Function=identity,
              eps::Real=1f-6) = begin
    z_len = length(z_ref_vec)
    mats = Vector{Matrix{Float32}}(undef, length(vals_group))
    for (idx, vals_i) in pairs(vals_group)
        row_count = length(vals_i)
        A = zeros(Float32, row_count, z_len)
        for k in 1:z_len
            z_pert = copy(z_ref_vec)
            z_pert[k] += eps
            pert = eval_fn(ds,
                           model,
                           x0,
                           z_pert,
                           steps,
                           safety_output,
                           terminal_output;
                           u_len=u_len,
                           safety_input=safety_input,
                           latent_dim=latent_dim,
                           output_map=output_map)
            pert_vals = extract_group(pert)[idx]
            diff = (Float32.(pert_vals) .- Float32.(vals_i)) ./ Float32(eps)
            A[:, k] = diff
        end
        mats[idx] = hcat(A, reshape(Float32.(vals_i), :, 1))
    end
    return mats
end


"""
    smt_critical_evaluations(ds, model, x0, z, steps,
                             safety_output, terminal_output; kwargs...) -> result

Compute SMT row evaluations at the most critical time for each disjunct.

The trajectory is generated via [`trajectory`](@ref). For each disjunct matrix in
`safety_output` and `safety_input`, the critical time is the point where the minimum
row evaluation is maximized. Returned values are the raw row evaluations at that time.
The terminal output constraints are evaluated only at the final output time.

# Arguments
- `ds::DiscreteRandomSystem`: system to roll out.
- `model`: [`InvertibleCoupling`](@ref), [`NormalizingFlow`](@ref), or `model(x, z) -> u`.
- `x0::AbstractVector`: initial state vector.
- `z::AbstractVector`: flat latent vector.
- `steps::Union{Integer,AbstractVector{<:Integer}}`: horizon splits; see [`trajectory`](@ref).
- `safety_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs.
- `terminal_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on final output.

# Keyword Arguments
- `u_len=nothing`: control dimension forwarded to [`trajectory`](@ref).
- `latent_dim=nothing`: required only when `model` is a function.
- `output_map=identity`: output map forwarded to [`trajectory`](@ref).
- `safety_input=nothing`: disjunct matrices acting on inputs; defaults to bounds from `ds.U`.

# Returns
Named tuple:
- `safety_output`: vector of vectors of row evaluations at each disjunct's critical time.
- `safety_input`: vector of vectors of row evaluations at each disjunct's critical time.
- `terminal_output`: vector of vectors of row evaluations at the final output time.
"""
function smt_critical_evaluations(ds::DiscreteRandomSystem,
                                  model,
                                  x0,
                                  z,
                                  steps,
                                  safety_output::AbstractVector{<:AbstractMatrix},
                                  terminal_output::AbstractVector{<:AbstractMatrix};
                                  u_len=nothing,
                                  safety_input::Union{Nothing,AbstractVector{<:AbstractMatrix}}=nothing,
                                  latent_dim::Union{Nothing,Integer}=nothing,
                                  output_map::Function=identity)
    safety_input_eff = safety_input === nothing ? _input_bounds_constraints(ds) : safety_input
    res = trajectory(ds, model, x0, z, steps;
                     u_len=u_len,
                      latent_dim=latent_dim,
                     output_map=output_map)

    ytrj = res.output_trajectory
    utrj = res.input_trajectory

    safety_output_vals = [ _critical_row_values(mat, ytrj) for mat in safety_output ]
    safety_input_vals = [ _critical_row_values(mat, utrj) for mat in safety_input_eff ]

    terminal_output_vals = Vector{Vector{Float32}}(undef, length(terminal_output))
    y_final = ytrj[:, end]
    for (i, mat) in pairs(terminal_output)
        terminal_output_vals[i] = _matrix_row_values(mat, y_final)
    end

    return (; safety_output=safety_output_vals,
             safety_input=safety_input_vals,
             terminal_output=terminal_output_vals)
end


"""
    smt_affine_critical(ds, model, x0, z_ref, steps,
                        safety_output, terminal_output; kwargs...) -> result

Build affine (linearized) SMT constraints in `z` around `z_ref` using finite differences
on the critical-time SMT evaluations.

The constant terms come from [`smt_critical_evaluations`](@ref). The linear terms are
computed by perturbing each component of `z_ref` with a finite-difference step.

# Arguments
- `ds::DiscreteRandomSystem`: system to roll out.
- `model`: [`InvertibleCoupling`](@ref), [`NormalizingFlow`](@ref), or `model(x, z) -> u`.
- `x0::AbstractVector`: initial state vector.
- `z_ref::AbstractVector`: reference latent vector for linearization.
- `steps::Union{Integer,AbstractVector{<:Integer}}`: horizon splits; see [`trajectory`](@ref).
- `safety_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs.
- `terminal_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on final output.

# Keyword Arguments
- `u_len=nothing`: control dimension forwarded to [`trajectory`](@ref).
- `latent_dim=nothing`: required only when `model` is a function.
- `output_map=identity`: output map forwarded to [`trajectory`](@ref).
- `eps=1f-6`: finite-difference step used for linearization.
- `safety_input=nothing`: disjunct matrices acting on inputs; defaults to bounds from `ds.U`.
- `return_base=false`: when `true`, also return the critical evaluations used to build the affine form.

# Returns
Named tuple of vectors of affine matrices in `z` space:
- `safety_output`: each matrix has rows `[a_z... b]` for `a_z' * z + b <= 0`.
- `safety_input`: same for input SMT.
- `terminal_output`: same for terminal output SMT.
"""
function smt_affine_critical(ds::DiscreteRandomSystem,
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
                             return_base::Bool=false)
    z_ref_vec = z_ref isa AbstractVector ? z_ref : vec(z_ref)
    z_len = length(z_ref_vec)
    safety_input_eff = safety_input === nothing ? _input_bounds_constraints(ds) : safety_input

    base = smt_critical_evaluations(ds,
                                    model,
                                    x0,
                                    z_ref_vec,
                                    steps,
                                    safety_output,
                                    terminal_output;
                                    u_len=u_len,
                                    safety_input=safety_input_eff,
                                    latent_dim=latent_dim,
                                    output_map=output_map)

    safety_output_affine = _affine_group(base.safety_output,
                                         p -> p.safety_output,
                                         ds,
                                         model,
                                         x0,
                                         z_ref_vec,
                                         steps,
                                         safety_output,
                                         safety_input_eff,
                                         terminal_output;
                                         u_len=u_len,
                                         latent_dim=latent_dim,
                                         output_map=output_map,
                                         eps=eps)
    safety_input_affine = _affine_group(base.safety_input,
                                        p -> p.safety_input,
                                        ds,
                                        model,
                                        x0,
                                        z_ref_vec,
                                        steps,
                                        safety_output,
                                        safety_input_eff,
                                        terminal_output;
                                        u_len=u_len,
                                        latent_dim=latent_dim,
                                        output_map=output_map,
                                        eps=eps)
    terminal_output_affine = _affine_group(base.terminal_output,
                                           p -> p.terminal_output,
                                           ds,
                                           model,
                                           x0,
                                           z_ref_vec,
                                           steps,
                                           safety_output,
                                           safety_input_eff,
                                           terminal_output;
                                           u_len=u_len,
                                           latent_dim=latent_dim,
                                           output_map=output_map,
                                           eps=eps)

    affine = (; safety_output=safety_output_affine,
              safety_input=safety_input_affine,
              terminal_output=terminal_output_affine)
    return return_base ? (; affine, base) : affine
end

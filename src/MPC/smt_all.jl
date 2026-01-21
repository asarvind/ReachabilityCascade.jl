"""
    smt_all_evaluations(ds, model, x0, z, steps,
                        safety_output, terminal_output; kwargs...) -> result

Compute SMT row evaluations at every time step for each disjunct.

This mirrors [`smt_critical_evaluations`](@ref) but does not collapse each
disjunct to a single critical time. Instead, every time index is kept as its
own disjunct group.

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
- `safety_output`: vector of vectors of row evaluations, one per disjunct per time.
- `safety_input`: vector of vectors of row evaluations, one per disjunct per time.
- `terminal_output`: vector of vectors of row evaluations at the final output time.
"""
function smt_all_evaluations(ds::DiscreteRandomSystem,
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
    smt_affine_all(ds, model, x0, z_ref, steps,
                   safety_output, terminal_output; kwargs...) -> result

Build affine (linearized) SMT constraints in `z` around `z_ref` using finite
differences on all-time SMT evaluations.

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
- `return_base=false`: when `true`, also return the evaluations used to build the affine form.

# Returns
Named tuple of vectors of affine matrices in `z` space:
- `safety_output`: each matrix has rows `[a_z... b]` for `a_z' * z + b <= 0`.
- `safety_input`: same for input SMT.
- `terminal_output`: same for terminal output SMT.
"""
function smt_affine_all(ds::DiscreteRandomSystem,
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

    base = smt_all_evaluations(ds,
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
                                         eval_fn=smt_all_evaluations,
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
                                        eval_fn=smt_all_evaluations,
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
                                           eval_fn=smt_all_evaluations,
                                           u_len=u_len,
                                           latent_dim=latent_dim,
                                           output_map=output_map,
                                           eps=eps)

    affine = (; safety_output=safety_output_affine,
              safety_input=safety_input_affine,
              terminal_output=terminal_output_affine)
    return return_base ? (; affine, base) : affine
end


"""
    smt_milp_all(ds, model, x0, z_ref, steps,
                 safety_output, terminal_output; kwargs...) -> (z, info)

Solve an MILP built from the all-time linearized SMT constraints around `z_ref`.

The SMT linearization is computed by [`smt_affine_all`](@ref). The objective is
the L1 distance to `z_ref`, implemented with slack variables and linear constraints.
"""
function smt_milp_all(ds::DiscreteRandomSystem,
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
    safety_input_eff = safety_input === nothing ? _input_bounds_constraints(ds) : safety_input

    rollout = trajectory(ds, model, x0, z_ref_vec, steps;
                         u_len=u_len,
                         latent_dim=latent_dim,
                         output_map=output_map)
    ytrj = rollout.output_trajectory
    utrj = rollout.input_trajectory

    _critical_time_index(mat::AbstractMatrix{<:Real}, traj::AbstractMatrix{<:Real}) = begin
        best_t = 1
        best_score = minimum(_matrix_row_values(mat, traj[:, 1]))
        for t in 2:size(traj, 2)
            vals = _matrix_row_values(mat, traj[:, t])
            score = minimum(vals)
            if score > best_score
                best_score = score
                best_t = t
            end
        end
        return best_t
    end

    T_out = size(ytrj, 2)
    T_in = size(utrj, 2)
    num_out = length(safety_output)
    num_in = length(safety_input_eff)

    selected_out = Set{Int}()
    for (i, mat) in pairs(safety_output)
        t = _critical_time_index(mat, ytrj)
        push!(selected_out, (i - 1) * T_out + t)
    end
    selected_in = Set{Int}()
    for (i, mat) in pairs(safety_input_eff)
        t = _critical_time_index(mat, utrj)
        push!(selected_in, (i - 1) * T_in + t)
    end

    affine = smt_affine_all(ds,
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

    z_sol = nothing
    info = (; termination_status=nothing, primal_status=nothing, feasible=false)

    while true
        affine_subset = (; safety_output=affine.safety_output[collect(selected_out)],
                          safety_input=affine.safety_input[collect(selected_in)],
                          terminal_output=affine.terminal_output)

        z_sol, info = _solve_affine_milp(affine_subset,
                                         z_ref_vec,
                                         steps;
                                         scale_steps=scale_steps,
                                         big_m=big_m,
                                         optimizer=optimizer,
                                         silent=silent,
                                         z_start=z_sol === nothing ? z_ref_vec : z_sol)
        z_sol === nothing && return nothing, info

        added = false

        for i in 1:num_out
            best_score = -Inf32
            best_t = 1
            for t in 1:T_out
                idx = (i - 1) * T_out + t
                vals = _matrix_row_values(affine.safety_output[idx], z_sol)
                score = minimum(vals)
                if score > best_score
                    best_score = score
                    best_t = t
                end
            end
            if best_score > 0
                idx = (i - 1) * T_out + best_t
                if !(idx in selected_out)
                    push!(selected_out, idx)
                    added = true
                end
            end
        end

        for i in 1:num_in
            best_score = -Inf32
            best_t = 1
            for t in 1:T_in
                idx = (i - 1) * T_in + t
                vals = _matrix_row_values(affine.safety_input[idx], z_sol)
                score = minimum(vals)
                if score > best_score
                    best_score = score
                    best_t = t
                end
            end
            if best_score > 0
                idx = (i - 1) * T_in + best_t
                if !(idx in selected_in)
                    push!(selected_in, idx)
                    added = true
                end
            end
        end

        if !added ||
           (length(selected_out) == num_out * T_out &&
            length(selected_in) == num_in * T_in)
            break
        end
    end

    return z_sol, info
end

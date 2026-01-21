import Evolutionary
import Evolutionary: CMAES

_smt_penalty(vals_group::AbstractVector{<:AbstractVector}) = begin
    total = 0.0f0
    for vals in vals_group
        total += max(0.0f0, minimum(Float32.(vals)))
    end
    return total
end

_eventual_penalty(terminal_output::AbstractVector{<:AbstractMatrix},
                  ytrj::AbstractMatrix{<:Real}) = begin
    best = Inf32
    for t in 1:size(ytrj, 2)
        worst = -Inf32
        for mat in terminal_output
            vals = _matrix_row_values(mat, ytrj[:, t])
            score = minimum(Float32.(vals))
            if score > worst
                worst = score
            end
        end
        if worst < best
            best = worst
        end
    end
    return max(0.0f0, best)
end

"""
    smt_cmaes(ds, model, x0, z0, steps,
              safety_output, terminal_output; kwargs...) -> result

Optimize the SMT cost using CMA-ES without linearization.

The SMT cost is computed as the sum of `max(0, min(row_vals))` across all safety and
input disjuncts at all time points. Terminal disjuncts use eventual semantics
(minimum over time).

# Arguments
- `ds::DiscreteRandomSystem`: system to roll out.
- `model`: [`InvertibleCoupling`](@ref), [`NormalizingFlow`](@ref), or `model(x, z) -> u`.
- `x0::AbstractVector`: initial state vector.
- `z0::AbstractVector`: initial latent vector (starting individual for CMA-ES).
- `steps::Union{Integer,AbstractVector{<:Integer}}`: horizon splits; see [`trajectory`](@ref).
- `safety_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on outputs.
- `terminal_output::AbstractVector{<:AbstractMatrix}`: disjunct matrices acting on final output.

# Keyword Arguments
- `u_len=nothing`: control dimension forwarded to [`trajectory`](@ref).
- `latent_dim=nothing`: required only when `model` is a function.
- `output_map=identity`: output map forwarded to [`trajectory`](@ref).
- `safety_input=nothing`: disjunct matrices acting on inputs; defaults to bounds from `ds.U`.
- `mu=10`: CMA-ES parent population size.
- `lambda=0`: CMA-ES offspring population size (0 uses the algorithm default, typically `2*mu`).
- `sigma0=0.5`: CMA-ES initial step size (scalar).
- `iterations=200`: CMA-ES iteration limit.
- `rng=Random.default_rng()`: RNG used by CMA-ES.
- `parallelization=:serial`: passed to `Evolutionary.Options`.

# Returns
Named tuple:
- `z`: best latent vector found.
- `objective`: best SMT cost.
- `result`: raw `Evolutionary` optimization result.
"""
function smt_cmaes(ds::DiscreteRandomSystem,
                   model,
                   x0,
                   z0,
                   steps,
                   safety_output::AbstractVector{<:AbstractMatrix},
                   terminal_output::AbstractVector{<:AbstractMatrix};
                   u_len=nothing,
                   safety_input::Union{Nothing,AbstractVector{<:AbstractMatrix}}=nothing,
                   latent_dim::Union{Nothing,Integer}=nothing,
                   output_map::Function=identity,
                   mu::Integer=10,
                   lambda::Integer=0,
                   sigma0::Real=0.5,
                   iterations::Integer=200,
                   rng=Random.default_rng(),
                   parallelization::Symbol=:serial)
    safety_input_eff = safety_input === nothing ? _input_bounds_constraints(ds) : safety_input

    obj = z -> begin
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

        _smt_penalty(safety_output_vals) +
            _smt_penalty(safety_input_vals) +
            _eventual_penalty(terminal_output, ytrj)
    end

    opts = Evolutionary.Options(iterations=Int(iterations),
                                rng=rng,
                                parallelization=parallelization)
    method = CMAES(mu=Int(mu), lambda=Int(lambda), sigma0=sigma0)
    result = Evolutionary.optimize(obj, z0, method, opts)
    z_best = Evolutionary.minimizer(result)
    cost = Evolutionary.minimum(result)
    return (; z=Float32.(z_best), objective=cost, result=result)
end

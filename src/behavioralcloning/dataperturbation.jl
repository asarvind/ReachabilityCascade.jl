using Random
using LazySets: low, high
using ..ControlSystem: DiscreteRandomSystem

"""
    perturb_input_sequence(ds, x0, u_ref, perturbation, stage_cost, terminal_cost;
                           rng=Random.GLOBAL_RNG, total_cost_threshold=nothing,
                           discount_factor=1.0)
    perturb_input_sequence(ds, x0, u_ref, perturbation, stage_cost, terminal_cost,
                           iterations::Integer; kwargs...)

Generate a perturbed control sequence by randomly scaling `perturbation` element-wise
with multipliers from ``\\{-1, 0, 1\\}``, adding it to each column of `u_ref`, and
clamping the result to the admissible input bounds of `ds.U`. The perturbation is
accepted only if the resulting discounted total cost is strictly higher than the original
cost; otherwise, an empty control sequence is returned. The multi-iteration method
repeats this procedure, feeding each accepted perturbation back in as the next
reference sequence.

# Args
- `ds :: DiscreteRandomSystem` — discrete-time dynamics used to roll out trajectories.
- `x0 :: AbstractVector{<:Real}` — initial state.
- `u_ref :: AbstractMatrix{<:Real}` — reference input sequence, one control per column.
- `perturbation :: AbstractVector{<:Real}` — element-wise perturbation magnitudes.
- `stage_cost :: Function` — `(x, u) -> Real` cost accrued at each intermediate step.
- `terminal_cost :: Function` — either `x_final -> Real` or `(x_final, u_last) -> Real` cost applied to the final step.
- `iterations :: Integer` — **only for the iterative method**, sets the maximum number of accepted perturbations to attempt.
- `rng :: AbstractRNG` — optional RNG used to draw random signs.

# Keyword Args
- `total_cost_threshold :: Union{Nothing, Real}` — optional upper bound on the discounted total cost (stage plus terminal) of the reference trajectory.
- `discount_factor :: Real` — exponential discount applied to the stage-cost sum (default `1.0`, i.e., no discount).

# Returns
- *Single-call variant* — `NamedTuple{(:inputs, :states)}` containing the perturbed control matrix and corresponding trajectory; both matrices are empty (0 columns) when the perturbation is rejected.
- *Iterative variant* — `Vector{NamedTuple}` whose entries are the accepted results from each iteration, in chronological order. Failed iterations produce no entry.
"""
function perturb_input_sequence(ds::DiscreteRandomSystem,
                                x0::AbstractVector{<:Real},
                                u_ref::AbstractMatrix{<:Real},
                                perturbation::AbstractVector{<:Real},
                                stage_cost::Function,
                                terminal_cost::Function;
                                rng::AbstractRNG = Random.GLOBAL_RNG,
                                total_cost_threshold::Union{Nothing, Real}=nothing,
                                discount_factor::Real=1.0)
    n_inputs, horizon = size(u_ref)
    length(perturbation) == n_inputs || error("Perturbation vector must have $n_inputs elements.")

    x0_vec = collect(x0)
    u_ref_mat = Matrix(u_ref)

    base_traj = ds(x0_vec, u_ref_mat)
    base_stage_cost, base_terminal_cost = _trajectory_cost(base_traj, u_ref_mat, stage_cost, terminal_cost, discount_factor)
    base_total_cost = base_stage_cost + base_terminal_cost
    if !isnothing(total_cost_threshold) && base_total_cost > total_cost_threshold
        return _empty_result(u_ref_mat, base_traj)
    end

    multipliers = rand(rng, (-1, 0, 1), size(u_ref_mat))
    perturbation_mat = perturbation .* multipliers
    perturbed_inputs = u_ref_mat + perturbation_mat
    u_lo = low(ds.U)
    u_hi = high(ds.U)
    perturbed_inputs = clamp.(perturbed_inputs, u_lo, u_hi)

    pert_traj = ds(x0_vec, perturbed_inputs)
    pert_stage_cost, pert_terminal_cost = _trajectory_cost(pert_traj, perturbed_inputs, stage_cost, terminal_cost, discount_factor)
    pert_total_cost = pert_stage_cost + pert_terminal_cost

    if pert_total_cost > base_total_cost
        return (inputs=perturbed_inputs, states=pert_traj)
    else
        return _empty_result(u_ref_mat, base_traj)
    end
end

function perturb_input_sequence(ds::DiscreteRandomSystem,
                                x0::AbstractVector{<:Real},
                                u_ref::AbstractMatrix{<:Real},
                                perturbation::AbstractVector{<:Real},
                                stage_cost::Function,
                                terminal_cost::Function,
                                iterations::Integer;
                                kwargs...)
    results = NamedTuple[]
    current_ref = u_ref
    for _ in 1:iterations
        result = perturb_input_sequence(ds, x0, current_ref, perturbation, stage_cost, terminal_cost; kwargs...)
        if size(result.inputs, 2) == 0
            continue
        end
        push!(results, result)
        current_ref = result.inputs
    end
    results
end

function _trajectory_cost(x_traj::AbstractMatrix{<:Real},
                          u_seq::AbstractMatrix{<:Real},
                          stage_cost::Function,
                          terminal_cost::Function,
                          discount_factor::Real)
    size(x_traj, 2) == size(u_seq, 2) + 1 || error("Trajectory length must be one more than the number of inputs.")

    stage_sum = 0.0
    weight = 1.0
    for k in axes(u_seq, 2)
        xk = x_traj[:, k]  # materialize to match user cost signatures
        uk = u_seq[:, k]
        stage_sum += weight * stage_cost(xk, uk)
        weight *= discount_factor
    end

    x_final = x_traj[:, size(x_traj, 2)]
    terminal_value = _evaluate_terminal_cost(terminal_cost, x_final, u_seq)
    return stage_sum, terminal_value
end

function _empty_result(u_template::AbstractMatrix, x_template::AbstractMatrix)
    inputs = similar(u_template, size(u_template, 1), 0)
    states = similar(x_template, size(x_template, 1), 0)
    (inputs=inputs, states=states)
end

function _evaluate_terminal_cost(terminal_cost::Function,
                                 x_final::AbstractVector,
                                 u_seq::AbstractMatrix)
    if applicable(terminal_cost, x_final)
        return terminal_cost(x_final)
    elseif size(u_seq, 2) > 0 && applicable(terminal_cost, x_final, u_seq[:, end])
        return terminal_cost(x_final, u_seq[:, end])
    else
        error("`terminal_cost` must accept either (x_final) or (x_final, u_last).")
    end
end

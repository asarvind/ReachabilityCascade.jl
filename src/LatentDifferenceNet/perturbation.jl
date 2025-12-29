import Flux
using Random
using Statistics: mean

"""
    spsa_update!(model::RefinementRNN,
                 samples::AbstractVector,
                 sys::DiscreteRandomSystem,
                 traj_cost_fn;
                 steps::Integer,
                 δ::Real,
                 temperature::Real=1,
                 rng::Random.AbstractRNG=Random.default_rng())

Single SPSA-style accept/reject update for `RefinementRNN`.

This evaluates the average objective (over `samples`) for:
- the current `model` (base)
- a positive perturbation `θ + δ*s`
- a negative perturbation `θ - δ*s`
where `s ∈ {−1,+1}^N` is random

The per-sample objective is the (softmax-averaged) trajectory cost at the **best** recurrent step
`k = argmin(trace.step_costs)` among steps `1:steps`:

`c_k = trace.step_costs[k]`

where `trace = model(x0, sys, traj_cost_fn, steps; temperature=temperature)`.

If either perturbation achieves a lower average objective than the base model, that perturbation's parameters are
loaded into `model`. Otherwise `model` is unchanged.

`samples` must be an iterable of `(x0,)` samples (e.g. a vector of named tuples with an `x0` field).

Returns a named tuple `(accepted, choice, base_loss, pos_loss, neg_loss, pert_loss)`, where:
- `choice` is `+1`, `-1`, or `0` (accepted positive, accepted negative, or rejected).
- `pert_loss` is the accepted perturbation loss when accepted; otherwise equals `base_loss`.
"""
function spsa_update!(model::RefinementRNN,
                      samples::AbstractVector,
                      sys::DiscreteRandomSystem,
                      traj_cost_fn;
                      steps::Integer,
                      δ::Real,
                      step_mode::Symbol=:terminal,
                      dual::Bool=false,
                      temperature::Real=1,
                      rng::Random.AbstractRNG=Random.default_rng())
    steps >= 1 || throw(ArgumentError("steps must be ≥ 1"))
    δ > 0 || throw(ArgumentError("δ must be positive"))
    (step_mode === :terminal || step_mode === :best) ||
        throw(ArgumentError("step_mode must be :terminal or :best; got $step_mode"))
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    isempty(samples) && throw(ArgumentError("samples must be non-empty"))

    θ, re = Flux.destructure(model)
    θT = eltype(θ)
    δT = convert(θT, δ)

    s = rand(rng, Bool, length(θ))
    signed = ifelse.(s, one(θT), -one(θT))

    m_pos = re(θ .+ δT .* signed)
    m_neg = re(θ .- δT .* signed)

    objective_for = function (m::RefinementRNN, x0::AbstractVector)
        trace = m(x0, sys, traj_cost_fn, steps; temperature=temperature, dual=dual)
        k = step_mode === :best ? trace.best_step : Int(steps)
        return Float32(trace.step_costs[k])
    end

    base_sum = 0.0f0
    pos_sum = 0.0f0
    neg_sum = 0.0f0
    for sample in samples
        x0 = sample.x0
        base_sum += objective_for(model, x0)
        pos_sum += objective_for(m_pos, x0)
        neg_sum += objective_for(m_neg, x0)
    end

    base_loss = base_sum / Float32(length(samples))
    pos_loss = pos_sum / Float32(length(samples))
    neg_loss = neg_sum / Float32(length(samples))

    accepted = false
    choice = 0
    pert_loss = base_loss

    if pos_loss < base_loss || neg_loss < base_loss
        accepted = true
        if pos_loss <= neg_loss
            choice = 1
            pert_loss = pos_loss
            Flux.loadmodel!(model, Flux.state(m_pos))
        else
            choice = -1
            pert_loss = neg_loss
            Flux.loadmodel!(model, Flux.state(m_neg))
        end
    end

    return (; accepted=accepted, choice=choice, base_loss=base_loss, pos_loss=pos_loss, neg_loss=neg_loss, pert_loss=pert_loss)
end

# Backward-compatible alias.
const spsa_epoch_update! = spsa_update!

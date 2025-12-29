import Flux
using Random

_softmax_average(x::AbstractArray; temperature::Real=1) = begin
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    v = vec(x)
    w = Flux.softmax(v ./ temperature)
    return sum(w .* v)
end

"""
    testrun(model, x0s, sys, traj_cost_fn; steps=1, temperature=1.0, rng=Random.default_rng(), u0_sampler=nothing)

Run a solver-style refinement pass for each initial state in `x0s`, sampling one initial guess per state.

For each `x0`:
1. Sample an initial guess `u0` using `u0_sampler(rng)` (defaults to i.i.d. Gaussian).
2. Run recursive refinement `model(x0, u0, sys, traj_cost_fn; steps=steps)` to obtain a sequence of guesses.
3. Evaluate each guess (including the initial guess) by rolling out `sys(x0, u)` and computing `cost_body = traj_cost_fn(x_rollout[:, 2:end])`.
4. Scalarize `cost_body` with a softmax-weighted average over all entries (default `temperature=1.0`) and pick the minimum across recursion steps.

Returns a vector of per-sample result named tuples with fields:
- `u_guesses`, `noises`: from the refinement trace.
- `x_rollouts`, `costs`: evaluated for every candidate guess (includes the initial guess and the final refined guess).
- `scores`: scalarized costs per candidate guess.
- `best_idx`: 1-based index into `u_guesses` / `scores` for the best candidate.
- `best_u`, `best_x`, `best_cost_body`, `best_score`: best input, best rollout, best cost array, and best scalar score.

Notes:
- `rng` is consumed (advanced) as samples are drawn.
- `u0_sampler` is state-independent by design (generative-model style).
"""
function testrun(model::ReactiveDenoisingNet,
                 x0s::AbstractVector,
                 sys,
                 traj_cost_fn;
                 steps::Integer=1,
                 temperature::Real=1.0,
                 rng::Random.AbstractRNG=Random.default_rng(),
                 u0_sampler=nothing)
    steps >= 1 || throw(ArgumentError("steps must be ≥ 1"))
    temperature > 0 || throw(ArgumentError("temperature must be positive"))

    sampler = u0_sampler === nothing ?
              (rng_ -> randn(rng_, Float32, model.input_dim, model.seq_len)) :
              u0_sampler

    results = Vector{NamedTuple}(undef, length(x0s))
    for (i, x0) in enumerate(x0s)
        x0_vec = Vector(x0)
        length(x0_vec) == model.state_dim || throw(DimensionMismatch("x0 must have length $(model.state_dim)"))

        u0 = sampler(rng)
        size(u0, 1) == model.input_dim || throw(DimensionMismatch("u0 must have $(model.input_dim) rows"))
        size(u0, 2) == model.seq_len || throw(DimensionMismatch("u0 must have $(model.seq_len) columns"))

        trace = model(x0_vec, Matrix(u0), sys, traj_cost_fn; steps=steps)
        u_guesses = trace.u_guesses

        # Evaluate all candidate guesses, including the final refined one.
        K = length(u_guesses)
        x_rollouts = Vector{Matrix{Float64}}(undef, K)
        costs = Vector{Matrix{Float32}}(undef, K)
        scores = Vector{Float32}(undef, K)

        for k in 1:K
            u_k = u_guesses[k]
            x_roll = sys(Vector(x0_vec), Matrix(u_k))
            size(x_roll, 1) == model.state_dim ||
                throw(DimensionMismatch("sys(x0, u) must return $(model.state_dim) rows; got $(size(x_roll, 1))"))
            size(x_roll, 2) == model.seq_len + 1 ||
                throw(DimensionMismatch("sys(x0, u) must return $(model.seq_len + 1) columns; got $(size(x_roll, 2))"))

            x_rollouts[k] = Float64.(Matrix(x_roll))
            x_body = x_rollouts[k][:, 2:end]
            cost_body = traj_cost_fn(x_body)
            size(cost_body, 1) == model.cost_dim ||
                throw(DimensionMismatch("traj_cost_fn(x_body) must return $(model.cost_dim) rows; got $(size(cost_body, 1))"))
            size(cost_body, 2) == model.seq_len ||
                throw(DimensionMismatch("traj_cost_fn(x_body) must return $(model.seq_len) columns; got $(size(cost_body, 2))"))

            costs[k] = Float32.(Matrix(cost_body))
            scores[k] = Float32(_softmax_average(costs[k]; temperature=temperature))
        end

        best_idx = argmin(scores)
        results[i] = (; u_guesses=u_guesses,
                      noises=trace.noises,
                      x_rollouts=x_rollouts,
                      costs=costs,
                      scores=scores,
                      best_idx=best_idx,
                      best_u=u_guesses[best_idx],
                      best_x=x_rollouts[best_idx],
                      best_cost_body=costs[best_idx],
                      best_score=scores[best_idx])
    end

    return results
end


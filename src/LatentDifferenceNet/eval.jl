import Flux

"""
    testrun(model::RefinementRNN,
            data::AbstractVector,
            sys::DiscreteRandomSystem,
            traj_cost_fn;
            steps::Integer=1,
            start_idx::Integer=1,
            temperature::Real=1)

Run a 1-epoch evaluation pass over `data` and return the best (minimum-over-steps) trajectory cost
for each sample.

The initial state is taken as `sample.state_trajectory[:, start_idx]`. The per-step trajectory cost is
scalarized with `_softmax_average(cost; temperature=temperature)`, and then minimized across
`steps`.

Returns a vector of `Float32` costs with `length == length(data)`.
"""
function testrun(model::RefinementRNN,
                 data::AbstractVector,
                 sys::DiscreteRandomSystem,
                 traj_cost_fn;
                 steps::Integer=1,
                 start_idx::Integer=1,
                 temperature::Real=1)
    steps >= 1 || throw(ArgumentError("steps must be ≥ 1"))
    start_idx >= 1 || throw(ArgumentError("start_idx must be ≥ 1"))
    temperature > 0 || throw(ArgumentError("temperature must be positive"))

    out = Float32[]
    best_steps = Int[]
    for sample in data
        x_full = Array(sample.state_trajectory)
        start_idx <= size(x_full, 2) || throw(ArgumentError("start_idx=$start_idx exceeds state_trajectory length $(size(x_full, 2))"))
        x0 = Vector(x_full[:, start_idx])

        trace = model(x0, sys, traj_cost_fn, steps; temperature=temperature)
        k = trace.best_step
        push!(best_steps, k)
        push!(out, Float32(_softmax_average(trace.costs[k]; temperature=temperature)))
    end
    return (; costs=out, best_steps=best_steps)
end

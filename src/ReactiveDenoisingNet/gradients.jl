import Flux
import ..TrainingAPI: gradient

"""
    gradient(model::ReactiveDenoisingNet,
             x0, u_guess0, u_target,
             sys,
             traj_cost_fn;
             steps=1,
             scale=nothing)

Compute gradients for imitation learning where only the *final* refinement step is differentiated.

This performs `steps-1` refinement iterations outside the gradient scope to obtain a pre-final guess `u_prev`.
It then computes `x_body = sys(x0, u_prev)[:, 2:end]` and `cost_body = traj_cost_fn(x_body)` outside the
gradient scope and differentiates only through the final call:

`u_final = model(x0, x_body, u_prev, cost_body).U_new`

The imitation loss is `Flux.mse(u_final .* s, u_target .* s)` where `s` is an optional per-row scaling vector
of length `input_dim` (broadcast across time). If `scale === nothing`, `s` is all ones.

Returns the gradient object compatible with `Flux.update!(opt_state, model, grads)`.
"""
function gradient(model::ReactiveDenoisingNet,
                  x0::AbstractVector,
                  u_guess0::AbstractMatrix,
                  u_target::AbstractMatrix,
                  sys,
                  traj_cost_fn;
                  steps::Integer=1,
                  scale=nothing)
    steps >= 1 || throw(ArgumentError("steps must be ≥ 1"))
    length(x0) == model.state_dim || throw(DimensionMismatch("x0 must have length $(model.state_dim)"))
    size(u_guess0, 1) == model.input_dim || throw(DimensionMismatch("u_guess0 must have $(model.input_dim) rows"))
    size(u_guess0, 2) == model.seq_len || throw(DimensionMismatch("u_guess0 must have $(model.seq_len) columns"))
    size(u_target) == size(u_guess0) || throw(DimensionMismatch("u_target must match size(u_guess0)"))

    if scale !== nothing
        length(scale) == model.input_dim || throw(DimensionMismatch("scale must have length $(model.input_dim)"))
    end

    x0_vec = Float32.(Vector(x0))
    u_prev = Float32.(Matrix(u_guess0))
    u_target32 = Float32.(Matrix(u_target))

    # Prior refinement iterations (no gradients by construction: outside the closure).
    for _ in 1:(Int(steps) - 1)
        x_roll = sys(Vector(x0), Matrix(u_prev))
        size(x_roll, 1) == model.state_dim ||
            throw(DimensionMismatch("sys(x0, u_guess) must return $(model.state_dim) rows; got $(size(x_roll, 1))"))
        size(x_roll, 2) == model.seq_len + 1 ||
            throw(DimensionMismatch("sys(x0, u_guess) must return $(model.seq_len + 1) columns; got $(size(x_roll, 2))"))

        x_body = Float32.(Matrix(x_roll[:, 2:end]))
        cost_body = traj_cost_fn(x_body)
        size(cost_body, 1) == model.cost_dim ||
            throw(DimensionMismatch("traj_cost_fn(x_body) must return $(model.cost_dim) rows; got $(size(cost_body, 1))"))
        size(cost_body, 2) == model.seq_len ||
            throw(DimensionMismatch("traj_cost_fn(x_body) must return $(model.seq_len) columns; got $(size(cost_body, 2))"))

        out = model(x0_vec, x_body, u_prev, Float32.(Matrix(cost_body)))
        u_prev = out.U_new
    end

    # Final-step rollout + cost (kept outside gradient scope).
    x_roll_last = sys(Vector(x0), Matrix(u_prev))
    size(x_roll_last, 1) == model.state_dim ||
        throw(DimensionMismatch("sys(x0, u_guess) must return $(model.state_dim) rows; got $(size(x_roll_last, 1))"))
    size(x_roll_last, 2) == model.seq_len + 1 ||
        throw(DimensionMismatch("sys(x0, u_guess) must return $(model.seq_len + 1) columns; got $(size(x_roll_last, 2))"))
    x_body_last = Float32.(Matrix(x_roll_last[:, 2:end]))
    cost_last = traj_cost_fn(x_body_last)
    size(cost_last, 1) == model.cost_dim ||
        throw(DimensionMismatch("traj_cost_fn(x_body) must return $(model.cost_dim) rows; got $(size(cost_last, 1))"))
    size(cost_last, 2) == model.seq_len ||
        throw(DimensionMismatch("traj_cost_fn(x_body) must return $(model.seq_len) columns; got $(size(cost_last, 2))"))
    cost_last32 = Float32.(Matrix(cost_last))

    s_mat = scale === nothing ? nothing : reshape(Float32.(collect(scale)), :, 1)

    grads = Flux.gradient(model) do m
        out = m(x0_vec, x_body_last, u_prev, cost_last32)
        u_final = out.U_new
        if s_mat === nothing
            return Flux.mse(u_final, u_target32)
        end
        return Flux.mse(u_final .* s_mat, u_target32 .* s_mat)
    end

    return grads[1]
end

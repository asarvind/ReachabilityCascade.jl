using Flux
using ..SequenceTransform: SequenceTransformation

"""
    RefinementModel

A network that computes corrections for trajectory refinement using a single sequence model.

# Fields
- `core_net`: Network applied to residual and guess branches.
"""
struct RefinementModel{C}
    core_net::C
end

"""
    rollout_guess(sample::ShootingBundle, transition_fn) -> AbstractArray

Roll out the current guess trajectory with `transition_fn` to produce the predicted next-state
sequence used as a residual target.

# Arguments
- `sample::ShootingBundle`: contains `x_guess` (with initial state) and `u_guess`.
- `transition_fn`: function `(x_prev_seq, u_seq) -> x_next_seq` operating on batched sequences of
  shape `(state_dim, seq_len, batch)`.

# Returns
- `x_next_seq`: array of shape `(state_dim, seq_len, batch)` giving the rollout from each previous
  state and input; the initial state from `x_guess` is untouched.
"""
function rollout_guess(sample::ShootingBundle, transition_fn)
    x_guess_full, u_guess = sample.x_guess, sample.u_guess
    seq_len = size(x_guess_full, 2) - 1
    # Previous states for each control step are simply the first `seq_len` slices.
    x_prev_seq = @view x_guess_full[:, 1:seq_len, :]
    return transition_fn(x_prev_seq, u_guess)
end

"""
    RefinementModel(state_dim::Int, input_dim::Int, hidden_dim::Int, depth::Int; activation=Flux.σ)

Construct a `RefinementModel` whose outputs correct both state and input, with output dimension
fixed to `state_dim + input_dim`. Trajectory cost features (assumed `state_dim` rows, post-initial
columns) are appended to the residual branch; zeros are appended to the guess branch.

# Arguments
- `state_dim`: Dimension of the state vector.
- `input_dim`: Dimension of the input vector.
- `hidden_dim`: Hidden dimension for the internal layers.
- `depth`: Depth of the `SequenceTransformation` chains.
- `activation`: Activation function.
"""
function RefinementModel(state_dim::Int, input_dim::Int, hidden_dim::Int, depth::Int; activation=Flux.σ, max_seq_len::Int=512)
    # Inputs to the networks are concatenated: x_res, x_guess, u_guess, cost_body
    net_in_dim = 3 * state_dim + input_dim
    out_dim = state_dim + input_dim

    core_net = SequenceTransformation(net_in_dim, hidden_dim, out_dim, depth, state_dim, activation; max_seq_len=max_seq_len)

    return RefinementModel(core_net)
end

Flux.@layer RefinementModel

function _param_eltype(obj, default)
    for t in Flux.trainable(obj)
        if t isa AbstractArray
            return eltype(t)
        else
            inner = _param_eltype(t, nothing)
            inner !== nothing && return inner
        end
    end
    return default
end

function _normalize_traj_cost(traj_cost, batch_size)
    # Return an array shaped (1, 1, batch_size) for scaling the terminal network.
    if traj_cost isa Number
        return fill(traj_cost, 1, 1, batch_size)
    end

    # Identify a batch dimension if present.
    sz = size(traj_cost)
    nd = ndims(traj_cost)
    batch_dim = findfirst(==(batch_size), sz)

    if nd == 1
        if length(traj_cost) == batch_size
            return reshape(traj_cost, 1, 1, batch_size)
        elseif length(traj_cost) == 1
            return fill(traj_cost[1], 1, 1, batch_size)
        else
            throw(ArgumentError("traj_cost_fn 1D output length $(length(traj_cost)) must be 1 or batch_size=$batch_size"))
        end
    elseif batch_dim !== nothing
        # Sum over all non-batch dimensions to yield one cost per batch element.
        sum_dims = setdiff(1:nd, (batch_dim,))
        reduced = sum(traj_cost; dims=sum_dims)
        reduced = dropdims(reduced; dims=sum_dims)
        vec_cost = vec(reduced)
        length(vec_cost) == batch_size || throw(ArgumentError("traj_cost_fn batch dimension mismatch; expected $batch_size costs"))
        return reshape(vec_cost, 1, 1, batch_size)
    elseif prod(sz) == 1
        return fill(traj_cost[1], 1, 1, batch_size)
    else
        # No batch dimension detected; treat as a single scalar trajectory cost.
        scalar_cost = sum(traj_cost)
        return fill(scalar_cost, 1, 1, batch_size)
    end
end

"""
    (m::RefinementModel)(sample::ShootingBundle, transition_fn, traj_cost_fn[, steps])

Forward pass for multiple-shooting refinement. Given a `ShootingBundle` (carrying `x_guess`
including the initial state, `u_guess`, optional `x_target`), the network computes residuals via
`transition_fn`, predicts intermediate/terminal corrections, and returns a new bundle with updated
trajectories. Passing a `steps::Integer` unrolls the forward refinement that many times (≤ 0 returns
the input bundle).

# Arguments
- `sample`: `ShootingBundle` with shapes `(state_dim, seq_len+1, batch)` for `x_guess` (including the initial state) and `(input_dim, seq_len, batch)` for `u_guess`.
- `transition_fn`: `(x_prev_seq, u_seq) -> x_next_seq` used to form residuals.
- `traj_cost_fn`: `x -> cost` trajectory cost over the guessed trajectory (supports batched inputs).
- `steps` (optional): number of refinement iterations to apply.

# Returns
- `ShootingBundle` with refined `x_guess`/`u_guess` and preserved initial state/x_target.
"""
(m::RefinementModel)(sample::ShootingBundle, transition_fn, traj_cost_fn) = begin
    param_T = _param_eltype(m.core_net, eltype(sample.x_guess))

    x_guess_full = param_T.(sample.x_guess)
    u_guess = param_T.(sample.u_guess)
    x_target = sample.x_target === nothing ? nothing : param_T.(sample.x_target)
    cast_bundle = ShootingBundle(x_guess_full, u_guess; x_target=x_target)

    x_res = rollout_guess(cast_bundle, transition_fn)
    x0 = selectdim(x_guess_full, 2, 1)
    x_guess = selectdim(x_guess_full, 2, 2:size(x_guess_full, 2))
    cost_raw = traj_cost_fn(x_guess_full)
    cost_body = ndims(cost_raw) == 3 && size(cost_raw, 2) == size(x_guess_full, 2) ? selectdim(cost_raw, 2, 2:size(cost_raw, 2)) : cost_raw
    size(cost_body, 2) == size(x_guess, 2) || throw(ArgumentError("traj_cost_fn must return an array with the same number of columns as the trajectory body"))

    # Concatenate inputs along feature dimension
    net_input_res = cat(x_res, x_guess, u_guess, cost_body, dims=1)

    # Compute correction terms using cost for residual branch and zeros for guess branch
    out_res = m.core_net(net_input_res, x0)
    zeros_cost = zeros(eltype(cost_body), size(cost_body))
    net_input_guess = cat(x_guess, x_guess, u_guess, zeros_cost, dims=1)
    out_guess = m.core_net(net_input_guess, x0)
    correction = out_res - out_guess
    state_dim = size(x_guess, 1)
    delta_x = correction[1:state_dim, :, :]
    delta_u = correction[state_dim+1:end, :, :]
    x_new = x_guess + delta_x
    u_new = u_guess + delta_u

    x0_full = reshape(x0, state_dim, 1, size(x_guess_full, 3))
    x_full = cat(x0_full, x_new; dims=2)

    return ShootingBundle(x_full, u_new; x_target=cast_bundle.x_target)
end

function (m::RefinementModel)(sample::ShootingBundle, transition_fn, traj_cost_fn, steps::Integer)
    steps <= 0 && return sample
    bundle = sample
    for _ in 1:steps
        bundle = m(bundle, transition_fn, traj_cost_fn)
    end
    return bundle
end

using Flux
using ..SequenceTransform: SequenceTransformation

"""
    RefinementModel

A network that computes corrections for trajectory refinement using a single sequence model.

# Fields
- `core_net`: Network applied to residual and guess branches.
- `cost_dim`: Number of rows expected from `traj_cost_fn` outputs.
"""
struct RefinementModel{C,I,L}
    core_net::C
    cost_dim::I
    latent_dim::L
end

"""
    rollout_guess(sample::ShootingBundle, transition_fn) -> AbstractArray

Roll out the current guess trajectory with `transition_fn` to produce the predicted next-state
sequence used as a residual target.

# Arguments
- `sample::ShootingBundle`: contains `x0` (initial state), `x_guess` (body), and `u_guess`.
- `transition_fn`: function `(x_prev_seq, u_seq) -> x_next_seq` operating on batched sequences of
  shape `(state_dim, seq_len, batch)`.

# Returns
- `x_next_seq`: array of shape `(state_dim, seq_len, batch)` giving the rollout from each previous
  state and input; the initial state from `x0` is untouched.
"""
function rollout_guess(sample::ShootingBundle, transition_fn)
    x0 = sample.x0
    x_guess, u_guess = sample.x_guess, sample.u_guess
    seq_len = size(x_guess, 2)
    if seq_len > 1
        x_prev_seq = cat(x0, selectdim(x_guess, 2, 1:seq_len-1); dims=2)
    else
        x_prev_seq = x0
    end
    return transition_fn(x_prev_seq, u_guess)
end

"""
    RefinementModel(state_dim::Int, input_dim::Int, cost_dim::Int, hidden_dim::Int, depth::Int; activation=Flux.σ)

Construct a `RefinementModel` whose outputs correct both state and input, with output dimension
fixed to `state_dim + input_dim`. Trajectory cost features (with `cost_dim` rows, post-initial
columns) and an optional latent sequence (`latent_dim` rows) are appended to the residual branch;
zeros are appended to the guess branch.

# Arguments
- `state_dim`: Dimension of the state vector.
- `input_dim`: Dimension of the input vector.
- `hidden_dim`: Hidden dimension for the internal layers.
- `depth`: Depth of the `SequenceTransformation` chains.
- `latent_dim`: Dimension of the latent sequence appended to each timestep (default `state_dim`).
- `activation`: Activation function.
"""
function RefinementModel(state_dim::Int, input_dim::Int, cost_dim::Int, hidden_dim::Int, depth::Int; activation=Flux.σ, max_seq_len::Int=512, latent_dim::Int=state_dim)
    cost_dim > 0 || throw(ArgumentError("cost_dim must be positive"))
    latent_dim > 0 || throw(ArgumentError("latent_dim must be positive"))
    # Inputs to the network are concatenated: x_res, x_guess, u_guess, cost_body
    net_in_dim = 2 * state_dim + input_dim + cost_dim + latent_dim
    out_dim = state_dim + input_dim + latent_dim

    core_net = SequenceTransformation(net_in_dim, hidden_dim, out_dim, depth, state_dim, activation; max_seq_len=max_seq_len)

    return RefinementModel(core_net, cost_dim, latent_dim)
end

RefinementModel(core_net::C, cost_dim::Int, latent_dim::Int) where {C} = RefinementModel{C,Int,Int}(core_net, cost_dim, latent_dim)
RefinementModel(core_net::C, cost_dim::Int; latent_dim::Int=cost_dim) where {C} = RefinementModel{C,Int,Int}(core_net, cost_dim, latent_dim)

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
- `sample`: `ShootingBundle` with shapes `(state_dim, batch)` for `x0`, `(state_dim, seq_len, batch)` for `x_guess`, and `(input_dim, seq_len, batch)` for `u_guess`.
- `transition_fn`: `(x_prev_seq, u_seq) -> x_next_seq` used to form residuals.
- `traj_cost_fn`: `x -> cost` trajectory cost over the guessed trajectory (supports batched inputs).
- `steps` (optional): number of refinement iterations to apply.

# Returns
- `ShootingBundle` with refined `x_guess`/`u_guess` and preserved initial state/x_target.
"""
(m::RefinementModel)(sample::ShootingBundle, transition_fn, traj_cost_fn; latent_state=nothing) = begin
    param_T = _param_eltype(m.core_net, eltype(sample.x_guess))

    x0 = param_T.(sample.x0)
    x_guess = param_T.(sample.x_guess)
    u_guess = param_T.(sample.u_guess)
    x_target = sample.x_target === nothing ? nothing : param_T.(sample.x_target)
    latent_init = latent_state !== nothing ? latent_state : sample.latent
    cast_bundle = ShootingBundle(x0, x_guess, u_guess; x_target=x_target, latent=latent_init)
    x0_ctx = reshape(x0, size(x0, 1), size(x0, 3))

    # latent sequence
    if cast_bundle.latent === nothing
        latent_seq = zeros(param_T, m.latent_dim, size(x_guess, 2), size(x_guess, 3))
    else
        ls = cast_bundle.latent
        nd = ndims(ls)
        if nd == 2
            ls = reshape(ls, size(ls,1), 1, size(ls,2))
        elseif nd != 3
            throw(ArgumentError("latent_state must have 2 or 3 dimensions"))
        end
        size(ls,1) == m.latent_dim || throw(ArgumentError("latent_state first dimension must equal latent_dim=$(m.latent_dim)"))
        size(ls,2) == size(x_guess,2) || throw(ArgumentError("latent_state sequence length must match x_guess"))
        size(ls,3) == size(x_guess,3) || throw(ArgumentError("latent_state batch must match x_guess"))
        latent_seq = param_T.(ls)
    end

    x_res = rollout_guess(cast_bundle, transition_fn)
    x_guess_full = cat(x0, x_guess; dims=2)
    cost_raw = traj_cost_fn(x_guess_full)
    cost_body = if cost_raw isa Number
        fill(param_T(cost_raw), m.cost_dim, size(x_guess, 2), size(x_guess, 3))
    elseif ndims(cost_raw) == 3 && size(cost_raw, 2) == size(x_guess_full, 2)
        selectdim(cost_raw, 2, 2:size(cost_raw, 2))
    elseif ndims(cost_raw) == 3
        cost_raw
    elseif ndims(cost_raw) == 2
        reshape(cost_raw, size(cost_raw, 1), size(cost_raw, 2), 1)
    else
        throw(ArgumentError("traj_cost_fn must return a scalar or array with dimensions (cost_dim, seq_len[, batch])"))
    end
    size(cost_body, 1) == m.cost_dim || throw(ArgumentError("traj_cost_fn must return cost_dim=$(m.cost_dim) rows"))
    size(cost_body, 2) == size(x_guess, 2) || throw(ArgumentError("traj_cost_fn must return an array with the same number of columns as the trajectory body"))

    # Concatenate inputs along feature dimension
    net_input_res = cat(x_res, x_guess, u_guess, cost_body, latent_seq, dims=1)

    # Compute correction terms using cost for residual branch and zeros for guess branch
    out_res = m.core_net(net_input_res, x0_ctx)
    zeros_cost = zeros(eltype(cost_body), size(cost_body))
    zeros_latent = zeros(eltype(latent_seq), size(latent_seq))
    net_input_guess = cat(x_guess, x_guess, u_guess, zeros_cost, zeros_latent, dims=1)
    out_guess = m.core_net(net_input_guess, x0_ctx)
    correction = out_res - out_guess
    state_dim = size(x_guess, 1)
    input_dim = size(u_guess, 1)
    delta_x = correction[1:state_dim, :, :]
    delta_u = correction[state_dim+1:state_dim+input_dim, :, :]
    delta_latent = correction[state_dim+input_dim+1:end, :, :]
    x_new = x_guess + delta_x
    u_new = u_guess + delta_u
    latent_new = delta_latent

    return ShootingBundle(x0, x_new, u_new; x_target=cast_bundle.x_target, latent=latent_new)
end

function (m::RefinementModel)(sample::ShootingBundle, transition_fn, traj_cost_fn, steps::Integer; latent_state=nothing)
    steps <= 0 && return sample
    bundle = sample
    for _ in 1:steps
        bundle = m(bundle, transition_fn, traj_cost_fn; latent_state=latent_state)
    end
    return bundle
end

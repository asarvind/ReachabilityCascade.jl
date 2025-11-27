using Flux
using ..SequenceTransform: SequenceTransformation

"""
    CorrectionNetwork

A network that computes intermediate and terminal corrections for trajectory refinement.

# Fields
- `inter_net`: Network for intermediate correction (dynamics).
- `term_net`: Network for terminal correction (constraints).
"""
struct CorrectionNetwork{I, T}
    inter_net::I
    term_net::T
end

_rollout_guess(sample::ShootingBundle, transition_fn) = begin
    x_guess_full, u_guess = sample.x_guess, sample.u_guess
    seq_len = size(x_guess_full, 2) - 1
    # Previous states for each control step are simply the first `seq_len` slices.
    x_prev_seq = @view x_guess_full[:, 1:seq_len, :]
    return transition_fn(x_prev_seq, u_guess)
end

"""
    CorrectionNetwork(state_dim::Int, input_dim::Int, hidden_dim::Int, out_dim::Int, depth::Int; activation=relu)

Construct a `CorrectionNetwork`.

# Arguments
- `state_dim`: Dimension of the state vector.
- `input_dim`: Dimension of the input vector.
- `hidden_dim`: Hidden dimension for the internal layers.
- `out_dim`: Output dimension (usually state_dim + input_dim).
- `depth`: Depth of the `SequenceTransformation` chains.
- `activation`: Activation function.
"""
function CorrectionNetwork(state_dim::Int, input_dim::Int, hidden_dim::Int, out_dim::Int, depth::Int; activation=relu)
    # Inputs to the networks are concatenated: x_res, x_guess, u_guess
    # Dimensions: state_dim + state_dim + input_dim
    net_in_dim = 2 * state_dim + input_dim

    # Intermediate network
    inter_net = SequenceTransformation(net_in_dim, hidden_dim, out_dim, depth, state_dim, activation)

    # Terminal network
    term_net = SequenceTransformation(net_in_dim, hidden_dim, out_dim, depth, state_dim, activation)

    return CorrectionNetwork(inter_net, term_net)
end

Flux.@layer CorrectionNetwork

"""
    (m::CorrectionNetwork)(sample::ShootingBundle, transition_fn, term_cost_fn[, steps])

Forward pass for multiple-shooting refinement. Given a `ShootingBundle` (carrying `x_guess`
including the initial state, `u_guess`, optional `x_target`), the network computes residuals via
`transition_fn`, predicts intermediate/terminal corrections, and returns a new bundle with updated
trajectories. Passing a `steps::Integer` unrolls the forward refinement that many times (≤ 0 returns
the input bundle).

# Arguments
- `sample`: `ShootingBundle` with shapes `(state_dim, seq_len+1, batch)` for `x_guess` (including the initial state) and `(input_dim, seq_len, batch)` for `u_guess`.
- `transition_fn`: `(x_prev_seq, u_seq) -> x_next_seq` used to form residuals.
- `term_cost_fn`: `x -> val` terminal cost; positive values are treated as violations via `relu`.
- `steps` (optional): number of refinement iterations to apply.

# Returns
- `ShootingBundle` with refined `x_guess`/`u_guess` and preserved initial state/x_target.
"""
(m::CorrectionNetwork)(sample::ShootingBundle, transition_fn, term_cost_fn) = begin
    x_res = _rollout_guess(sample, transition_fn)
    x_guess_full, u_guess = sample.x_guess, sample.u_guess
    x0 = selectdim(x_guess_full, 2, 1)
    x_guess = selectdim(x_guess_full, 2, 2:size(x_guess_full, 2))

    # Concatenate inputs along feature dimension
    net_input = cat(x_res, x_guess, u_guess, dims=1)

    # Compute intermediate correction terms
    out_inter_res = m.inter_net(net_input, x0)
    net_input_guess = cat(x_guess, x_guess, u_guess, dims=1)
    out_inter_guess = m.inter_net(net_input_guess, x0)
    delta_inter = out_inter_res - out_inter_guess

    # Terminal correction
    out_term = m.term_net(net_input, x0)
    x_term = selectdim(x_guess, 2, size(x_guess, 2))
    term_val = term_cost_fn(x_term)
    term_violation = Flux.relu.(term_val)
    if ndims(term_violation) == 1
        term_violation = reshape(term_violation, 1, 1, length(term_violation))
    elseif ndims(term_violation) == 2
        term_violation = reshape(term_violation, 1, 1, size(term_violation, 2))
    end
    delta_term = term_violation .* out_term

    correction = delta_inter + delta_term
    state_dim = size(x_guess, 1)
    delta_x = correction[1:state_dim, :, :]
    delta_u = correction[state_dim+1:end, :, :]
    x_new = x_guess + delta_x
    u_new = u_guess + delta_u

    x0_full = reshape(x0, state_dim, 1, size(x_guess_full, 3))
    x_full = cat(x0_full, x_new; dims=2)

    return ShootingBundle(x_full, u_new; x_target=sample.x_target)
end

function (m::CorrectionNetwork)(sample::ShootingBundle, transition_fn, term_cost_fn, steps::Integer)
    steps <= 0 && return sample
    bundle = sample
    for _ in 1:steps
        bundle = m(bundle, transition_fn, term_cost_fn)
    end
    return bundle
end

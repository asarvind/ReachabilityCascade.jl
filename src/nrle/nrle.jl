"""
    struct NRLE

Neural Reachability Likelihood Estimator built on top of a *conditional flow*.

It models a sample vector that is composed of two parts:
- **state difference**: `(x_next - s_now)` where both have dimension `state_dim`
- **trajectory property** vector (dimension `prop_dim`)

**Context** is just the current state.

NRLE provides:
- `encode(nrle, s_now, x_next, prop)` → `(z, loglik)`, where the encoded sample is
  `y = [x_diff; prop]` with `x_diff = (x_next - s_now)`.
- `reach(nrle, s_now, z)` → `(x_next, prop)` by decoding a latent `z` to
  `y = [x_diff; prop]` and then forming **`x_next = x_diff + s_now`**.

# Fields
- `flow`            : underlying conditional flow
- `state_dim::Int`  : dimension of the state
- `prop_dim::Int`   : dimension of trajectory property part of the sample
"""
struct NRLE{F}
    flow::F
    state_dim::Int
    prop_dim::Int
end

Flux.@layer NRLE

# ------------------------------ Utilities ------------------------------------

_as_colmat(x::AbstractVecOrMat) = ndims(x) == 1 ? reshape(x, size(x,1), 1) : x

# --------------------------- Public API --------------------------------------

function encode(nrle::NRLE, s_now, x_next, prop)
    s = _as_colmat(s_now)
    x = _as_colmat(x_next)
    p = _as_colmat(prop)
    @assert size(s,1) == nrle.state_dim
    @assert size(x,1) == nrle.state_dim
    @assert size(p,1) == nrle.prop_dim
    B = size(x, 2)
    @assert size(p,2) == B
    @assert size(s,2) == B

    x_diff = (x .- s)

    y = vcat(x_diff, p)
    c = s

    z, logdet = nrle.flow(y, c; inverse=false)

    D = size(z,1)
    ll_prior = -0.5f0 .* sum(z.^2; dims=1) .- (D/2) .* log(2f0*pi)

    ll = vec(ll_prior) .+ logdet
    return (latent=z, log_determinant=logdet, log_likelihood = ll)
end

function reach(nrle::NRLE, s_now, z)
    s = _as_colmat(s_now)
    Z = _as_colmat(z)
    B = size(Z, 2)
    @assert size(s,1) == nrle.state_dim

    c = s

    y_norm, ld = nrle.flow(Z, c; inverse=true)

    x_diff = @views y_norm[1:nrle.state_dim, :]
    prop   = @views y_norm[nrle.state_dim+1:end, :]

    x_next = x_diff .+ s
    return (next_state=x_next, property=prop, log_det= -ld) # take negative of -ld to consider encoding likelihood
end

# -------------------------- Automated Constructor ----------------------------

"""
    NRLE(state_dim, prop_dim;
         n_blocks=6, hidden=128, n_glu=2, bias=true,
         state_scaling=ones(Float32, state_dim),
         prop_scaling=ones(Float32, prop_dim))

Construct an NRLE with an internally built `ConditionalFlow`. The flow input
dimension is `state_dim + prop_dim` and the context dimension is `state_dim`
(current state only). Scaling vectors are passed through to the flow for
consistent handling:
- `state_scaling` is used for state difference dimensions
- `prop_scaling` is used for trajectory property dimensions

# Returns
An `NRLE` instance.
"""
function NRLE(state_dim::Integer, prop_dim::Integer;
              n_blocks::Integer=6, hidden::Integer=128, n_glu::Integer=2, bias::Bool=true,
              state_scaling::AbstractVector{<:Real}=ones(Float32, state_dim),
              prop_scaling::AbstractVector{<:Real}=ones(Float32, prop_dim))
    x_dim = state_dim + prop_dim
    ctx_dim = state_dim

    x_scaling = vcat(Float32.(state_scaling), Float32.(prop_scaling))
    c_scaling = Float32.(state_scaling)

    flow = ConditionalFlow(x_dim, ctx_dim; n_blocks=n_blocks, hidden=hidden,
                           n_glu=n_glu, bias=bias,
                           x_scaling=x_scaling, c_scaling=c_scaling)
    return NRLE(flow, state_dim, prop_dim)
end

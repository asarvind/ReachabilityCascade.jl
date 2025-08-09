# Neural Reachability Likelihood Estimator (NRLE)
# ===============================================
# A thin wrapper around a conditional Flow to embed a target state given an
# initial state and time context. Log-likelihoods and inverse mapping are
# exposed as separate methods.

struct NRLE{F}
    net::F
    state_dim::Int
end
Flux.@layer NRLE trainable=(net,)

function NRLE(state_dim::Integer; depth::Integer=2, width::Integer=8*state_dim,
              clamp::Real=2.0, mode::Symbol=:affine, seed::Union{Integer,Nothing}=nothing)
    cdim = state_dim + 1
    net = Flow(Int(state_dim), cdim, Int(depth); hidden=Int(width), clamp=clamp, mode=mode, seed=seed)
    return NRLE(net, Int(state_dim))
end

# Forward pass: only compute the embedded (encoded) vector from the flow.
function (m::NRLE)(x::AbstractVecOrMat, x0::AbstractVecOrMat, t)
    xmat  = ndims(x)  == 1 ? reshape(Float32.(x),  m.state_dim, 1) : Float32.(x)
    x0mat = ndims(x0) == 1 ? reshape(Float32.(x0), m.state_dim, 1) : Float32.(x0)
    @assert size(xmat,1)  == m.state_dim "x has wrong feature dimension"
    @assert size(x0mat,1) == m.state_dim "x0 has wrong feature dimension"
    @assert size(xmat,2)  == size(x0mat,2) "x and x0 must share batch size"
    batch = size(xmat, 2)

    # Normalize time to (1, batch) Float32 without reordering samples.
    if ndims(t) == 0
        @assert batch == 1 "scalar t only allowed when batch == 1"
        tvec = Float32[t]
    else
        tvec = Float32.(vec(t))
    end
    @assert length(tvec) == batch "t must have the same number of elements as the batch size"
    tcol = reshape(tvec, 1, batch)  # (1, batch)

    c = vcat(x0mat, tcol)
    z, _ = m.net(xmat, c)
    return z
end

# Compute per-sample log-likelihoods given x, x0, and t.
function loglikelihood(m::NRLE, x, x0, t)
    xmat  = ndims(x)  == 1 ? reshape(Float32.(x),  m.state_dim, 1) : Float32.(x)
    x0mat = ndims(x0) == 1 ? reshape(Float32.(x0), m.state_dim, 1) : Float32.(x0)
    @assert size(xmat,1)  == m.state_dim "x has wrong feature dimension"
    @assert size(x0mat,1) == m.state_dim "x0 has wrong feature dimension"
    @assert size(xmat,2)  == size(x0mat,2) "x and x0 must share batch size"
    batch = size(xmat, 2)
    tvec = ndims(t) == 0 ? ( @assert batch==1; Float32[t] ) : Float32.(vec(t))
    @assert length(tvec) == batch
    tcol = reshape(tvec, 1, batch)
    c = vcat(x0mat, tcol)
    return loglikelihood(m.net, xmat, c)
end

# Inverse: map from embedded vector back to original space.
function inverse(m::NRLE, z, x0, t)
    zmat  = ndims(z) == 1 ? reshape(Float32.(z),  m.state_dim, 1) : Float32.(z)
    x0mat = ndims(x0) == 1 ? reshape(Float32.(x0), m.state_dim, 1) : Float32.(x0)
    @assert size(zmat,1)  == m.state_dim "z has wrong feature dimension"
    @assert size(x0mat,1) == m.state_dim "x0 has wrong feature dimension"
    @assert size(zmat,2)  == size(x0mat,2) "z and x0 must share batch size"
    batch = size(zmat, 2)
    tvec = ndims(t) == 0 ? ( @assert batch==1; Float32[t] ) : Float32.(vec(t))
    @assert length(tvec) == batch
    tcol = reshape(tvec, 1, batch)
    c = vcat(x0mat, tcol)
    x, _ = inverse(m.net, zmat, c)  # return only x for convenience
    return x
end

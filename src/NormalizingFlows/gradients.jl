import Flux
using Statistics: mean

import ..TrainingAPI: gradient

"""
    gradient(flow, x, context; return_loss=false) -> grads
    gradient(flow, x, context; return_loss=true) -> (grads, loss)

Compute gradients for maximum-likelihood training under a standard normal prior.

The flow defines `z = encode(flow, x, context)` with log-determinant `logdet = log|det(dz/dx)|`.
The (unnormalized) log-likelihood is:

`logp(x) = logp(z) + logdet`, with `logp(z) = -0.5 * sum(z.^2)`

The constant term `-(D/2)*log(2π)` is omitted because it does not affect gradients.

# Arguments
- `flow`: [`NormalizingFlow`](@ref).
- `x`: data sample(s).
  - `AbstractVector` of length `D` (single sample), or
  - `AbstractMatrix` of size `D×B` (batch).
- `context`: context sample(s).
  - `AbstractVector` of length `C` (single sample), or
  - `AbstractMatrix` of size `C×B` (batch).
  The shape (vector vs matrix) must match `x`.

# Returns
- If `return_loss=false` (default): `grads` compatible with `Flux.update!(opt_state, flow, grads)`.
- If `return_loss=true`: `(grads, loss)` where `loss::Float32` is the negative mean log-likelihood.
"""
function gradient(flow::NormalizingFlow, x::AbstractMatrix, context::AbstractMatrix; return_loss::Bool=false)
    ndims(x) == 2 || throw(ArgumentError("x must be a (dim × batch) matrix"))
    size(x, 1) == flow.dim || throw(DimensionMismatch("x must have $(flow.dim) rows"))
    ndims(context) == 2 || throw(ArgumentError("context must be a (context_dim × batch) matrix"))
    size(context, 1) == flow.context_dim || throw(DimensionMismatch("context must have $(flow.context_dim) rows"))
    size(context, 2) == size(x, 2) || throw(DimensionMismatch("context batch must match x batch"))

    # Work in Float32 for consistency with the rest of the library (and to match typical Flux defaults).
    x32 = Float32.(Matrix(x))
    c32 = Float32.(Matrix(context))

    loss_ref = Ref{Float32}(0f0)
    grads = Flux.gradient(flow) do m
        # Flow forward: x -> z, while tracking per-sample log|det(dz/dx)|.
        z, logdet = encode(m, x32, c32)
        # log p(z) under N(0, I) up to an additive constant:
        #   log p(z) = -0.5 * ||z||^2  (+ constant dropped)
        logp_z = -0.5f0 .* vec(sum(z .^ 2; dims=1))
        # Change-of-variables:
        #   log p(x) = log p(z) + log|det(dz/dx)|
        logp_x = logp_z .+ logdet
        # Maximize log-likelihood -> minimize negative mean log-likelihood
        loss = -mean(logp_x)
        loss_ref[] = Float32(loss)
        return loss
    end

    return return_loss ? (grads[1], loss_ref[]) : grads[1]
end

function gradient(flow::NormalizingFlow, x::AbstractVector, context::AbstractVector; return_loss::Bool=false)
    length(x) == flow.dim || throw(DimensionMismatch("x must have length $(flow.dim); got $(length(x))"))
    length(context) == flow.context_dim ||
        throw(DimensionMismatch("context must have length $(flow.context_dim); got $(length(context))"))
    x_mat = reshape(x, :, 1)
    c_mat = reshape(context, :, 1)
    return gradient(flow, x_mat, c_mat; return_loss=return_loss)
end

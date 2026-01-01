import Flux
using Statistics: mean

import ..TrainingAPI: gradient

"""
    gradient(model, other, x_true, context; kwargs...) -> grads
    gradient(model, other, x_true, context; kwargs...) -> (grads, loss)
    gradient(model, other, x_true, context; kwargs...) -> (grads, loss, extras)

Compute gradients for an *adversarial* two-network game update for **one** network `model`,
with the other network `other` treated as fixed.

This defines a single scalar loss for `model` as the sum of three objectives:

1. **Accept true** (pull true samples inside the box in `model`'s latent space):
   `relu(‖encode(model, x_true, c)‖₁ - 1 + margin_true)`
2. **Reject other** (push fakes from `other` outside the box in `model`'s latent space):
   `relu(1 - ‖encode(model, decode(other, z_other, c), c)‖₁ + margin_adv)`
3. **Fool other** (generate fakes that appear inside the box to `other`):
   `relu(‖encode(other, decode(model, z_self, c), c)‖₁ - 1 + margin_adv)`

The total loss is the mean over the batch of each term, summed.

Important: this method returns gradients **only for `model`**. You call it twice per step
to get one update for each network (A vs B, and B vs A).

# Arguments
- `model`: the network being updated.
- `other`: the opponent network, treated as fixed during this gradient computation.
- `x_true`: true sample(s) (`D` vector or `D×B` matrix).
- `context`: context (`C` vector or `C×B` matrix), matching the shape of `x_true`.

# Keyword Arguments
- `margin_true=0.5`: margin used for the true-sample inclusion hinge.
- `margin_adv=0.0`: margin used for both adversarial hinges (rejecting opponent samples and fooling the opponent).
- `w_true=1.0`: weight on the true-sample inclusion loss.
- `w_reject=1.0`: weight on the "reject other" loss.
- `w_fool=1.0`: weight on the "fool other" loss.
- `rng=Random.default_rng()`: RNG used to sample `z_self` and `z_other`.
- `return_loss=false`: if `true`, also returns the scalar loss (and optional extras).
- `return_true_hinges=false`: if `true`, include per-sample true hinge losses in `extras.true_hinges`.
- `return_components=false`: if `true`, include scalar components in `extras`:
  `extras.accept_true`, `extras.reject_other`, `extras.fool_other`.

# Returns
- If `return_loss=false`: `grads` for `model`.
- If `return_loss=true` and no extras requested: `(grads, loss)`.
- If `return_loss=true` and any extras requested: `(grads, loss, extras)`.
"""
function gradient(model::InvertibleCoupling,
                  other::InvertibleCoupling,
                  x_true::AbstractMatrix,
                  context::AbstractMatrix;
                  margin_true::Real=0.5,
                  margin_adv::Real=0.0,
                  w_true::Real=1.0,
                  w_reject::Real=1.0,
                  w_fool::Real=1.0,
                  rng=Random.default_rng(),
                  return_loss::Bool=false,
                  return_true_hinges::Bool=false,
                  return_components::Bool=false)
    size(x_true, 1) == model.dim || throw(DimensionMismatch("x_true must have $(model.dim) rows"))
    size(context, 1) == model.context_dim || throw(DimensionMismatch("context must have $(model.context_dim) rows"))
    size(context, 2) == size(x_true, 2) || throw(DimensionMismatch("context batch must match x batch"))

    # Both models must be compatible with the same sample/context shapes.
    other.dim == model.dim || throw(DimensionMismatch("other.dim must match model.dim"))
    other.context_dim == model.context_dim || throw(DimensionMismatch("other.context_dim must match model.context_dim"))

    x_true32 = Float32.(Matrix(x_true))
    c32 = Float32.(Matrix(context))
    margin_true32 = Float32(margin_true)
    margin_adv32 = Float32(margin_adv)
    w_true32 = Float32(w_true)
    w_reject32 = Float32(w_reject)
    w_fool32 = Float32(w_fool)
    D = model.dim
    B = size(x_true32, 2)

    # Pre-sample latents outside the AD scope (keeps the gradient scope deterministic and avoids RNG mutation in AD).
    z_self = sample_latent_l1(rng, D, B)
    z_other = sample_latent_l1(rng, D, B)

    # Precompute opponent fakes for the "reject other" term. This does not depend on `model`,
    # so we keep it outside the gradient closure.
    x_fake_from_other = decode(other, z_other, c32)

    loss_ref = Ref{Float32}(0f0)
    accept_true_ref = Ref{Float32}(0f0)
    reject_other_ref = Ref{Float32}(0f0)
    fool_other_ref = Ref{Float32}(0f0)
    true_hinges_ref = Ref{Vector{Float32}}(Float32[])

    grads = Flux.gradient(model) do m
        # 1) Accept true: encode true samples with `m`.
        z_true = encode(m, x_true32, c32)
        n_true = vec(sum(abs.(z_true); dims=1))
        true_hinges = Flux.relu.(n_true .- 1f0 .+ margin_true32)
        accept_true = mean(true_hinges)

        # 2) Reject other: encode fakes produced by `other` with `m`.
        z_other_seen = encode(m, x_fake_from_other, c32)
        n_other_seen = vec(sum(abs.(z_other_seen); dims=1))
        reject_other = mean(Flux.relu.(1f0 .- n_other_seen .+ margin_adv32))

        # 3) Fool other: generate fakes with `m`, then encode them with `other`.
        x_fake_from_self = decode(m, z_self, c32)
        z_seen_by_other = encode(other, x_fake_from_self, c32)
        n_seen_by_other = vec(sum(abs.(z_seen_by_other); dims=1))
        fool_other = mean(Flux.relu.(n_seen_by_other .- 1f0 .+ margin_adv32))

        # Weighted sum so callers can enable "inclusion-only" training with `w_reject=0, w_fool=0`.
        loss = w_true32 * accept_true + w_reject32 * reject_other + w_fool32 * fool_other
        loss_ref[] = Float32(loss)
        accept_true_ref[] = Float32(accept_true)
        reject_other_ref[] = Float32(reject_other)
        fool_other_ref[] = Float32(fool_other)
        true_hinges_ref[] = Float32.(true_hinges)
        return loss
    end

    if !return_loss
        return grads[1]
    end

    if !(return_true_hinges || return_components)
        return grads[1], loss_ref[]
    end

    extras = NamedTuple()
    if return_true_hinges
        extras = merge(extras, (; true_hinges=true_hinges_ref[]))
    end
    if return_components
        extras = merge(extras, (; accept_true=accept_true_ref[],
                                 reject_other=reject_other_ref[],
                                 fool_other=fool_other_ref[]))
    end
    return grads[1], loss_ref[], extras
end

function gradient(model::InvertibleCoupling,
                  other::InvertibleCoupling,
                  x_true::AbstractVector,
                  context::AbstractVector;
                  margin_true::Real=0.5,
                  margin_adv::Real=0.0,
                  w_true::Real=1.0,
                  w_reject::Real=1.0,
                  w_fool::Real=1.0,
                  rng=Random.default_rng(),
                  return_loss::Bool=false,
                  return_true_hinges::Bool=false,
                  return_components::Bool=false)
    length(x_true) == model.dim || throw(DimensionMismatch("x_true must have length $(model.dim)"))
    length(context) == model.context_dim || throw(DimensionMismatch("context must have length $(model.context_dim)"))
    x_true_mat = reshape(x_true, :, 1)
    c_mat = reshape(context, :, 1)
    return gradient(model, other, x_true_mat, c_mat;
                    margin_true=margin_true,
                    margin_adv=margin_adv,
                    w_true=w_true,
                    w_reject=w_reject,
                    w_fool=w_fool,
                    rng=rng,
                    return_loss=return_loss,
                    return_true_hinges=return_true_hinges,
                    return_components=return_components)
end

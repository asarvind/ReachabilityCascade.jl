import Flux
import Functors
using Statistics: mean

import ..TrainingAPI: gradient

"""
    gradient(model, other, x_true, context; kwargs...) -> grads
    gradient(model, other, x_true, context; kwargs...) -> (grads, loss)
    gradient(model, other, x_true, context; kwargs...) -> (grads, loss, extras)

Compute gradients for a single-network update where `other` is treated as fixed
(typically an EMA copy used to generate fake samples).

This defines a scalar loss for `model` as the sum of:
1. **Accept true** (pull true samples inside the box in `model`'s latent space):
   `relu(‖encode(model, x_true, c)‖ - 1 + margin_true)`
2. **Reject fakes** (push EMA-generated samples outside the box in `model`'s latent space):
   `relu(1 - ‖encode(model, decode(other, z_other, c), c)‖ + margin_adv)`

# Arguments
- `model`: the network being updated.
- `other`: fixed network used to generate fake samples (e.g., EMA copy).
- `x_true`: true sample(s) (`D` vector or `D×B` matrix).
- `context`: context (`C` vector or `C×B` matrix), matching the shape of `x_true`.

# Keyword Arguments
- `margin_true=0.5`: margin used for the true-sample inclusion hinge.
- `margin_adv=0.0`: margin used for the fake rejection hinge.
- `w_true=1.0`: weight on the true-sample inclusion loss.
- `w_reject=1.0`: weight on the fake rejection loss.
- `mode=:sum`: gradient mode:
  - `:sum` (default): one gradient for the full weighted sum loss.
  - `:orthogonal_adv`: compute two gradients: `g_true = ∇(w_true*accept_true)` and
    `g_adv_orth = ∇(w_reject*reject_fake)` with the component along `g_true`
    removed (global projection), returning `(g_true, g_adv_orth)`.
- `norm_kind=:l1`: norm used in the hinge losses (`:l1`, `:l2`, or `:linf`).
- `rng=Random.default_rng()`: RNG used to sample `z_other`.
- `latent_radius_min=0.0`: minimum radius in `[0, 1]` for sampled fake latents.
- `return_loss=false`: if `true`, also returns the scalar loss (and optional extras).
- `return_true_hinges=false`: if `true`, include per-sample true hinge losses in `extras.true_hinges` and
  the corresponding per-sample true norms in `extras.true_norms`.
- `return_components=false`: if `true`, include scalar components in `extras`:
  `extras.accept_true`, `extras.reject_fake`.

# Returns
- If `return_loss=false`: `grads` for `model`.
- If `return_loss=true` and no extras requested: `(grads, loss)`.
- If `return_loss=true` and any extras requested: `(grads, loss, extras)`.
"""

_grad_dotnorm(g_adv, g_true) = begin
    dot_ref = Ref{Float32}(0f0)
    norm_ref = Ref{Float32}(0f0)

    Functors.fmap((a, b) -> begin
        if a === nothing || b === nothing
            return nothing
        end
        if a isa AbstractArray && b isa AbstractArray
            dot_ref[] += Float32(sum(a .* b))
            norm_ref[] += Float32(sum(b .* b))
            return nothing
        end
        if a isa Number && b isa Number
            dot_ref[] += Float32(a * b)
            norm_ref[] += Float32(b * b)
            return nothing
        end
        return nothing
    end, g_adv, g_true)

    return dot_ref[], norm_ref[]
end

_grad_orthogonalize(g_adv, g_true; eps::Float32=1f-12) = begin
    dot, n2 = _grad_dotnorm(g_adv, g_true)
    α = n2 <= eps ? 0f0 : dot / (n2 + eps)
    g_adv_orth = Functors.fmap((a, b) -> begin
        if a === nothing || b === nothing
            return a
        end
        if a isa AbstractArray && b isa AbstractArray
            return a .- α .* b
        end
        if a isa Number && b isa Number
            return a - α * b
        end
        return a
    end, g_adv, g_true)
    return g_adv_orth
end

function gradient(model::InvertibleCoupling,
                  other::InvertibleCoupling,
                  x_true::AbstractMatrix,
                  context::AbstractMatrix;
                  margin_true::Real=0.5,
                  margin_adv::Real=0.0,
                  w_true::Real=1.0,
                  w_reject::Real=1.0,
                  mode::Symbol=:sum,
                  norm_kind::Symbol=:l1,
                  rng=Random.default_rng(),
                  latent_radius_min::Real=0.0,
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
    D = model.dim
    B = size(x_true32, 2)

    needs_reject = w_reject32 != 0f0

    # Pre-sample latents outside the AD scope (keeps the gradient scope deterministic and avoids RNG mutation in AD).
    z_other = needs_reject ? sample_latent(rng, D, B; norm_kind=norm_kind, radius_min=latent_radius_min) : nothing

    # Precompute opponent fakes for the "reject other" term. This does not depend on `model`,
    # so we keep it outside the gradient closure.
    x_fake_from_other = needs_reject ? decode(other, z_other, c32) : nothing

    loss_ref = Ref{Float32}(0f0)
    accept_true_ref = Ref{Float32}(0f0)
    reject_other_ref = Ref{Float32}(0f0)
    true_hinges_ref = Ref{Vector{Float32}}(Float32[])
    true_norms_ref = Ref{Vector{Float32}}(Float32[])

    function compute_components(m)
        # 1) Accept true: encode true samples with `m`.
        z_true = encode(m, x_true32, c32)
        n_true = _batch_norm(z_true, norm_kind)
        true_hinges = Flux.relu.(n_true .- 1f0 .+ margin_true32)
        accept_true = mean(true_hinges)

        reject_other = 0f0
        if needs_reject
            # 2) Reject other: encode fakes produced by `other` with `m`.
            z_other_seen = encode(m, x_fake_from_other, c32)
            n_other_seen = _batch_norm(z_other_seen, norm_kind)
            reject_other = mean(Flux.relu.(1f0 .- n_other_seen .+ margin_adv32))
        end
        return accept_true, reject_other, true_hinges, n_true
    end

    if !(mode === :sum || mode === :orthogonal_adv)
        throw(ArgumentError("unsupported mode=$(repr(mode)); expected :sum or :orthogonal_adv"))
    end

    # Always compute scalar components once for consistent logging and extras.
    accept_true0, reject_other0, true_hinges0, n_true0 = compute_components(model)
    loss0 = w_true32 * accept_true0 + w_reject32 * reject_other0
    loss_ref[] = Float32(loss0)
    accept_true_ref[] = Float32(accept_true0)
    reject_other_ref[] = Float32(reject_other0)
    true_hinges_ref[] = Float32.(true_hinges0)
    true_norms_ref[] = Float32.(n_true0)

    grads = if mode === :sum
        Flux.gradient(model) do m
            accept_true, reject_other, _, _ = compute_components(m)
            return w_true32 * accept_true + w_reject32 * reject_other
        end
    else
        grads_true = Flux.gradient(model) do m
            accept_true, _, _, _ = compute_components(m)
            return w_true32 * accept_true
        end
        grads_adv = Flux.gradient(model) do m
            _, reject_other, _, _ = compute_components(m)
            return w_reject32 * reject_other
        end
        g_true = grads_true[1]
        g_adv = grads_adv[1]
        g_adv_orth = _grad_orthogonalize(g_adv, g_true)
        ((g_true, g_adv_orth),)
    end

    if !return_loss
        return grads[1]
    end

    if !(return_true_hinges || return_components)
        return grads[1], loss_ref[]
    end

    extras = NamedTuple()
    if return_true_hinges
        extras = Base.merge(extras, (; true_hinges=true_hinges_ref[], true_norms=true_norms_ref[]))
    end
    if return_components
        extras = Base.merge(extras, (; accept_true=accept_true_ref[],
                                      reject_fake=reject_other_ref[]))
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
                  mode::Symbol=:sum,
                  norm_kind::Symbol=:l1,
                  rng=Random.default_rng(),
                  latent_radius_min::Real=0.0,
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
                    mode=mode,
                    norm_kind=norm_kind,
                    rng=rng,
                    latent_radius_min=latent_radius_min,
                    return_loss=return_loss,
                    return_true_hinges=return_true_hinges,
                    return_components=return_components)
end

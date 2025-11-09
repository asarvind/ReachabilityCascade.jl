# ===================== Flow Transformer Gradients ============================

using Flux

"""
    default_flow_loss(latent, logdet)

Negative log-likelihood under a standard normal base distribution, averaged
across the batch and sequence length.
"""
function default_flow_loss(latent::AbstractArray, logdet::AbstractVector)
    D, L, B = size(latent)
    latent_flat = reshape(latent, :, B)
    quad = sum(latent_flat .^ 2; dims=1) .* 0.5
    Tval = promote_type(eltype(latent_flat), eltype(logdet))
    const_term = Tval(0.5) * Tval(D * L) * log(Tval(2π))
    nll = quad .+ const_term .- logdet
    return sum(nll) / (Tval(length(nll)) * Tval(L))
end

"""
    flow_transformer_gradient(flow::FlowTransformer, input, context; loss_fn=default_flow_loss)

Compute parameter gradients for `flow` w.r.t. a scalar loss.

# Arguments
- `flow::FlowTransformer`: the normalizing-flow transformer model.
- `input::AbstractArray`: input tensor `(d_model, seq_len, batch)` passed to the flow.
- `context::AbstractArray`: conditioning context `(context_dim, batch)` or `(context_dim)`.

# Keywords
- `loss_fn`: callable receiving `(latent, logdet)` and returning a scalar loss.
  Defaults to `default_flow_loss`, which computes the mean negative
  log-likelihood under a unit Gaussian base distribution.

# Returns
Gradient object shaped like `flow` (`grads[1]` from a `Flux.gradient` call).
"""
function flow_transformer_gradient(flow::FlowTransformer,
                                   input::AbstractArray,
                                   context::AbstractArray;
                                   loss_fn::Function=default_flow_loss)
    grads = Flux.gradient(flow) do m
        latent, logdet = m(input, context)
        return loss_fn(latent, logdet)
    end
    return grads[1]
end

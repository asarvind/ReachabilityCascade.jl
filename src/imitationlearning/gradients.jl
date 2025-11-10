using Flux

"""
    refinement_gradient(network, context_batch, sequence_batch, perturb_batch,
                        target_batch; loss_fn=Flux.Losses.hinge_loss)

Compute gradients of `network` (an `IterativeRefinementNetwork`) with respect to
its parameters given batched inputs. The `loss_fn` operates on the raw transformer
output differences (pre-sigmoid), using hinge labels derived from `target_batch`.

# Arguments
- `network :: IterativeRefinementNetwork`
- `context_batch :: AbstractArray{<:Real}` — contexts `(context_dim, batch)`.
- `sequence_batch :: AbstractArray{<:Real}` — sequences `(seq_dim, len[, batch])`.
- `perturb_batch :: AbstractArray{<:Real}` — perturbations matching `sequence_batch`.
- `target_batch :: AbstractArray{<:Real}` — desired control sequences for imitation.

# Keyword Arguments
- `loss_fn` — loss applied to raw differences (default `Flux.Losses.hinge_loss`).

# Returns
- `(loss, grads)` where `grads` is the `Flux.gradient` result tree.
"""
function refinement_gradient(network::IterativeRefinementNetwork,
                             context_batch::AbstractArray{<:Real},
                             sequence_batch::AbstractArray{<:Real},
                             perturb_batch::AbstractArray{<:Real},
                             target_batch::AbstractArray{<:Real};
                             loss_fn = Flux.Losses.hinge_loss)
    ctx = Float32.(context_batch)
    seq = Float32.(sequence_batch)
    pert = Float32.(perturb_batch)
    target = Float32.(target_batch)
    @assert size(target) == size(seq) "target_batch must match sequence_batch dimensions"

    direction = sign.(target .- seq)
    labels = replace(direction, 0f0 => -1f0)

    loss_ref = Ref{Float32}(0f0)
    grads = Flux.gradient(network) do net
        _, outputs = net(ctx, seq, pert)
        l = loss_fn(outputs.diff, labels)
        loss_ref[] = l
        l
    end

    loss_ref[], grads
end

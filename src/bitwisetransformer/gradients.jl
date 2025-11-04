using Flux

"""
    gradients(net::ProbabilityTransformer, context, prior_sequence, targets; loss_fn=Flux.Losses.binarycrossentropy)

Compute the loss and gradients of a `ProbabilityTransformer` against ground-truth binary targets.

# Arguments
- `net::ProbabilityTransformer`
- `context`: context vector or matrix `(context_dim, batch_size)`
- `prior_sequence`: prior sequence supplied to the transformer (vector/matrix/tensor)
- `targets`: binary ground-truth sequence with the same shape as `prior_sequence`

# Keyword Arguments
- `loss_fn`: loss applied to the predicted probabilities (defaults to `Flux.Losses.binarycrossentropy`)

# Returns
NamedTuple containing `loss`, `gradient`, and the model outputs (`values`, `probabilities`, `logits`).
"""
function gradients(net::ProbabilityTransformer,
                   context::AbstractVecOrMat,
                   prior_sequence::AbstractArray{<:Real,N},
                   targets::AbstractArray{<:Real,N};
                   loss_fn::Function=Flux.Losses.binarycrossentropy) where {N}
    prior_info, prior_tensor = _normalize_prior_sequence(prior_sequence, net.prior_dim)
    _, target_tensor = _normalize_prior_sequence(targets, net.prior_dim)
    size(prior_tensor) == size(target_tensor) ||
        throw(ArgumentError("targets must match prior_sequence shape"))
    target_tensor = Float32.(target_tensor)
    const_target_tensor = target_tensor

    grad_record = Flux.gradient(net) do model
        res = model(context, prior_tensor)
        _, probs_tensor = _normalize_prior_sequence(res.probabilities, model.prior_dim)
        return loss_fn(probs_tensor, const_target_tensor)
    end
    gradient = grad_record[1]

    result = net(context, prior_tensor)
    _, logits_tensor = _normalize_prior_sequence(result.logits, net.prior_dim)
    _, probs_tensor = _normalize_prior_sequence(result.probabilities, net.prior_dim)
    _, values_tensor = _normalize_prior_sequence(result.values, net.prior_dim)
    loss_value = loss_fn(probs_tensor, const_target_tensor)

    return (loss=loss_value,
            gradient=gradient,
            values=_restore_sequence_shape(values_tensor, prior_info),
            probabilities=_restore_sequence_shape(probs_tensor, prior_info),
            logits=_restore_sequence_shape(logits_tensor, prior_info))
end

"""
    gradients(net::HierarchicalBitwiseTransformer, context, targets; loss_fn)

Compute gradients for every stage inside a [`HierarchicalBitwiseTransformer`](@ref).
Inputs are assumed to live in the original (shifted/scaled) space; the helper
automatically normalizes them using the transformer's `mean` and `radius` vectors
before extracting bit targets.

# Arguments
- `net::HierarchicalBitwiseTransformer`
- `context::AbstractVecOrMat`: context batch shared by all stages.
- `targets::AbstractArray{<:Real}`: final value sequence expected after all
  stages have applied their perturbations.

# Keyword Arguments
- `loss_fn`: loss applied to each stage's predicted probabilities (default
  `Flux.Losses.binarycrossentropy`).

# Returns
Named tuple containing per-stage gradient results, the recovered bit targets for
each stage, and the reconstructed values obtained by accumulating those targets.
"""
function gradients(net::HierarchicalBitwiseTransformer,
                   context::AbstractVecOrMat,
                   targets::AbstractArray{<:Real,N};
                   loss_fn::Function=Flux.Losses.binarycrossentropy) where {N}
    isempty(net.stages) && throw(ArgumentError("hierarchical transformer contains no stages"))

    prior_dim = net.stages[end].prior_dim
    target_info, target_tensor = _normalize_prior_sequence(targets, prior_dim)

    mean_tensor = reshape(net.mean, 1, prior_dim, 1)
    radius_tensor = reshape(net.radius, 1, prior_dim, 1)
    prior_norm = zeros(Float32, size(target_tensor))
    target_norm = (target_tensor .- mean_tensor) ./ radius_tensor

    bit_positions = Int[stage.bit_position for stage in net.stages]
    min_bit = minimum(bit_positions)
    max_bit = maximum(bit_positions)
    base_scale = Float32(2.0f0 ^ (min_bit - 1))

    residual = Float32.(target_norm) .- prior_norm
    normalized = residual ./ base_scale
    normalized_int = round.(Int, normalized)
    if any(abs.(normalized .- normalized_int) .> 1f-2)
        throw(ArgumentError("targets are incompatible with provided bit positions"))
    end

    pad = max_bit - min_bit + 1
    flat_int = vec(normalized_int)
    num_elems = length(flat_int)
    bits_matrix = Array{Int}(undef, pad, num_elems)
    for (idx, value) in enumerate(flat_int)
        bits_matrix[:, idx] = digits(value, base=2, pad=pad)
    end

    stage_results = Vector{NamedTuple}(undef, length(net.stages))
    bit_targets = Vector{Any}(undef, length(net.stages))

    current_prior = prior_norm
    for idx in reverse(eachindex(net.stages))
        stage = net.stages[idx]
        bit_index = stage.bit_position - min_bit + 1
        bit_flat = bits_matrix[bit_index, :]
        bit_tensor = reshape(Float32.(bit_flat), size(current_prior))

        stage_result = gradients(stage,
                                 context,
                                 current_prior,
                                 bit_tensor;
                                 loss_fn=loss_fn)
        stage_results[idx] = stage_result
        bit_targets[idx] = _restore_sequence_shape(bit_tensor, target_info)

        stage_scale = Float32(2.0f0 ^ (stage.bit_position - 1))
        current_prior = current_prior + stage_scale .* bit_tensor
    end

    reconstructed_norm = current_prior
    reconstructed_actual = reconstructed_norm .* radius_tensor .+ mean_tensor
    reconstructed = _restore_sequence_shape(reconstructed_actual, target_info)
    return (stages=stage_results,
            bit_targets=bit_targets,
            reconstructed=reconstructed)
end

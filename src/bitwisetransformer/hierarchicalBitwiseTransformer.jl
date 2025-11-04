using Flux

"""
    HierarchicalBitwiseTransformer(context_dim, prior_dim; kwargs...)

Compose multiple `ProbabilityTransformer`s, each operating at a distinct bit position.
The sequence is processed from the highest bit to the lowest. The output from one stage
serves as the prior sequence for the next (initialised to zero). The transformer also
carries per-dimension `mean` and `radius` vectors that shift/scale the decoded values
without contributing trainable parameters.

# Arguments
- `context_dim::Integer`: dimensionality of the shared context vector.
- `prior_dim::Integer`: dimensionality of the per-step dependent variables.

# Keyword Arguments
- `bit_positions::AbstractVector{<:Integer}`: bit positions to use (highest to lowest).
- `mean::AbstractVector{<:Real}=zeros(Float32, prior_dim)`: per-dimension offset applied
  after decoding.
- `radius::AbstractVector{<:Real}=ones(Float32, prior_dim)`: per-dimension scaling applied
  after decoding (must be positive).
- `embed_dim::Integer=64`, `heads::Integer=4`, `ff_hidden::Integer=128`, `activation=Flux.relu`:
  passed to each underlying `ProbabilityTransformer`.

# Returns
A `HierarchicalBitwiseTransformer` containing a vector of probability transformers ordered from
highest to lowest bit positions, along with fixed `mean` and `radius` vectors.
"""
struct HierarchicalBitwiseTransformer
    stages::Vector{ProbabilityTransformer}
    mean::Vector{Float32}
    radius::Vector{Float32}
end

Flux.@layer HierarchicalBitwiseTransformer

Flux.trainable(net::HierarchicalBitwiseTransformer) = (; stages = net.stages)

function HierarchicalBitwiseTransformer(context_dim::Integer,
                                        prior_dim::Integer;
                                        bit_positions::AbstractVector{<:Integer},
                                        mean::AbstractVector{<:Real}=zeros(Float32, prior_dim),
                                        radius::AbstractVector{<:Real}=ones(Float32, prior_dim),
                                        embed_dim::Integer=64,
                                        heads::Integer=4,
                                        ff_hidden::Integer=128,
                                        activation=Flux.relu)
    isempty(bit_positions) && throw(ArgumentError("bit_positions must be non-empty"))
    length(mean) == prior_dim || throw(ArgumentError("mean length must equal prior_dim"))
    length(radius) == prior_dim || throw(ArgumentError("radius length must equal prior_dim"))
    radius_vals = Float32.(radius)
    any(radius_vals .<= 0f0) &&
        throw(ArgumentError("radius entries must be positive"))
    mean_vals = Float32.(mean)
    stages = Vector{ProbabilityTransformer}(undef, length(bit_positions))
    for (idx, bit) in enumerate(bit_positions)
        stages[idx] = ProbabilityTransformer(context_dim;
                                             prior_dim=prior_dim,
                                             embed_dim=embed_dim,
                                             heads=heads,
                                             ff_hidden=ff_hidden,
                                             activation=activation,
                                             bit_position=bit)
    end
    return HierarchicalBitwiseTransformer(stages, mean_vals, radius_vals)
end

"""
    predict_hierarchical_values(net, context, prior_sequence)

Apply the hierarchical transformer. Processing begins at the last stage (highest bit)
and proceeds backwards. Each stage receives the same context; the output `values`
from stage `i+1` serves as the prior sequence for stage `i`. Internally the network
operates on normalized coordinates, subtracting `mean` and dividing by `radius`, and
maps the final normalized result back into the original range.

# Arguments
- `net::HierarchicalBitwiseTransformer`
- `context`: context vector or matrix `(context_dim, batch_size)`
- `sequence_length::Integer`: number of steps in the generated value sequence.

# Returns
NamedTuple `(values, probabilities)` where:
- `values` is the final real-valued sequence produced after all stages (after applying
  the stored `mean` and `radius`).
- `probabilities` is a vector of per-stage probability tensors (ordered from highest to lowest bit).
"""
function (net::HierarchicalBitwiseTransformer)(context::AbstractVecOrMat,
                                               sequence_length::Integer)
    num_stages = length(net.stages)
    probs_per_stage = Vector{Any}(undef, num_stages)

    prior_dim = net.stages[end].prior_dim
    batch_size = size(_colmat(context), 2)
    info = (:tensor, (sequence_length, prior_dim, batch_size))
    mean_tensor = reshape(net.mean, 1, prior_dim, 1)
    radius_tensor = reshape(net.radius, 1, prior_dim, 1)

    prior_tensor = zeros(Float32, sequence_length, prior_dim, batch_size)
    current_prior = prior_tensor
    for idx in reverse(eachindex(net.stages))
        stage = net.stages[idx]
        values, probs = stage(context, current_prior)
        probs_per_stage[idx] = _restore_sequence_shape(Float32.(probs), info)
        current_prior = values
    end
    values_tensor = current_prior .* radius_tensor .+ mean_tensor
    return (values=_restore_sequence_shape(values_tensor, info),
            probabilities=probs_per_stage)
end

using Flux
using Random
using ..SimpleTransformers: SimpleSequenceTransformer

"""
    IterativeRefinementNetwork(transformer)

Wraps a `SimpleSequenceTransformer` to expose a forward method that accepts
`(context, sequence, perturbation)` and returns per-component acceptance
probabilities along with raw transformer outputs.
"""
struct IterativeRefinementNetwork
    transformer::SimpleSequenceTransformer
end

Flux.@layer IterativeRefinementNetwork

"""
    (network::IterativeRefinementNetwork)(context, sequence, perturbation)

Inputs can be 2D (`features × length`) or 3D (`features × length × batch`).
Returns `(decisions, outputs)` where `decisions` are boolean masks derived from
raw transformer differences (`diff .> 0`), and `outputs` is a named tuple
with fields `base`, `perturbed`, and `diff`, each matching the input shape.
"""
function (network::IterativeRefinementNetwork)(context::AbstractArray{<:Real},
                                               sequence::AbstractArray{<:Real},
                                               perturbation::AbstractArray{<:Real})
    seq32 = Float32.(sequence)
    pert32 = Float32.(perturbation)
    @assert size(pert32) == size(seq32) "Perturbation must match sequence dimensions"
    ctx32 = Float32.(context)

    transformer = network.transformer

    base_out = transformer(seq32, ctx32)
    perturbed_out = transformer(seq32 .+ pert32, ctx32)
    diff = perturbed_out .- base_out
    decisions = diff .> 0

    decisions, (base=base_out, perturbed=perturbed_out, diff=diff)
end

"""
    refine_control_sequence(network, context, reference, initial_guess,
                            perturb_sampler, steps; rng=Random.GLOBAL_RNG)

Iteratively refine an initial control sequence guess using the provided
`IterativeRefinementNetwork`.
"""
function refine_control_sequence(network::IterativeRefinementNetwork,
                                 context::AbstractVector{<:Real},
                                 reference::AbstractMatrix{<:Real},
                                 initial_guess::AbstractMatrix{<:Real},
                                 perturb_sampler::Function,
                                 steps::Integer;
                                 rng::AbstractRNG = Random.GLOBAL_RNG)
    @assert steps > 0 "Number of steps must be positive"
    size(reference) == size(initial_guess) ||
        error("reference and initial_guess must have matching dimensions")

    ctx32 = Float32.(context)
    ref32 = Float32.(reference)
    current = Float32.(initial_guess)
    zero_pert = zeros(Float32, size(current))
    _, outputs = network(ctx32, current, zero_pert)
    base_out = outputs.base

    for _ in 1:steps
        perturb = perturb_sampler(size(current))
        @assert size(perturb) == size(current) "Perturbation sampler returned incorrect size"
        perturb32 = Float32.(perturb)
        candidate = current .+ perturb32

        decisions, outputs = network(ctx32, current, perturb32)

        direction = sign.(ref32 .- current)
        towards_ref = ((candidate .- current) .* direction) .> 0

        accept_mask = decisions .& towards_ref
        current = ifelse.(accept_mask, candidate, current)

        _, base_outputs = network(ctx32, current, zero_pert)
        base_out = base_outputs.base
    end

    current
end

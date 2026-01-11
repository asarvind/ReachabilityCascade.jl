import Flux

"""
    inclusion_losses(model, data_iter; batch_size=32, margin_true=0.5, norm_kind=:l1) -> losses

Compute per-sample inclusion hinge losses for ground-truth samples under an [`InvertibleCoupling`](@ref).

For each ground-truth `(context, sample)` pair, we compute:
1. `z = encode(model, sample, context)`
2. `ℓ = relu(‖z‖ - 1 + margin_true)` where the norm is chosen by `norm_kind`

and return a flat vector of `ℓ` values (one scalar per sample).

This is a **single pass** over `data_iter`. If `data_iter` yields vectors, this function batches them
internally (for speed) and still returns losses per sample. If `data_iter` yields matrices, each yield
is treated as an already-batched mini-batch, and losses are returned per column.

# Arguments
- `model`: [`InvertibleCoupling`](@ref).
- `data_iter`: iterable dataset. Each element must be a named tuple `(; context, sample)`:
  - `context`: `AbstractVector` of length `context_dim` (or `context_dim×B` matrix if already batched).
  - `sample`: `AbstractVector` of length `dim` (or `dim×B` matrix if already batched).

# Keyword Arguments
- `batch_size=32`: batching used when `data_iter` yields vectors.
- `margin_true=0.5`: hinge margin.
- `norm_kind=:l1`: norm used in the hinge (`:l1`, `:l2`, or `:linf`).

# Returns
- `losses::Vector{Float32}`: per-sample hinge losses (flattened).
"""
function inclusion_losses(model::InvertibleCoupling,
                          data_iter;
                          batch_size::Integer=32,
                          margin_true::Real=0.5,
                          norm_kind::Symbol=:l1)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))

    # Helper to enforce the training-style dataset contract: `(; context, sample)`.
    unpack(sample) = begin
        sample isa NamedTuple || throw(ArgumentError("each dataset element must be a NamedTuple"))
        (haskey(sample, :context) && haskey(sample, :sample)) ||
            throw(ArgumentError("each dataset element must have keys `:context` and `:sample`"))
        return sample.context, sample.sample
    end

    # Compute per-column hinge losses for a (dim×B) batch.
    batch_hinges(x_batch::AbstractMatrix, c_batch::AbstractMatrix) = begin
        # We compute in Float32 to match other modules and to keep outputs stable.
        x32 = Float32.(Matrix(x_batch))
        c32 = Float32.(Matrix(c_batch))
        z = encode(model, x32, c32)
        norms = _batch_norm(z, norm_kind)
        return Float32.(Flux.relu.(norms .- 1f0 .+ Float32(margin_true)))
    end

    losses = Float32[]

    # Buffers for vector-yielding iterators (we batch them for speed).
    ctx_buf = Any[]
    x_buf = Any[]

    flush!() = begin
        isempty(x_buf) && return nothing
        x_batch = reduce(hcat, map(x -> x isa AbstractVector ? x : vec(x), x_buf))
        c_batch = reduce(hcat, map(x -> x isa AbstractVector ? x : vec(x), ctx_buf))
        empty!(x_buf)
        empty!(ctx_buf)
        append!(losses, batch_hinges(x_batch, c_batch))
        return nothing
    end

    for item in data_iter
        context, x = unpack(item)

        if (x isa AbstractMatrix) != (context isa AbstractMatrix)
            throw(ArgumentError("context and sample must both be vectors or both be matrices"))
        end

        if x isa AbstractMatrix
            size(x, 1) == model.dim || throw(DimensionMismatch("sample must have $(model.dim) rows"))
            size(context, 1) == model.context_dim || throw(DimensionMismatch("context must have $(model.context_dim) rows"))
            size(context, 2) == size(x, 2) || throw(DimensionMismatch("context batch must match sample batch"))
            append!(losses, batch_hinges(x, context))
            continue
        end

        # Vector case: accumulate into a mini-batch.
        push!(ctx_buf, context)
        push!(x_buf, x)
        if length(x_buf) >= batch_size
            flush!()
        end
    end

    flush!()
    return losses
end

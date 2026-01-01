import Flux
using Random

import ..TrainingAPI: build, train!, gradient

_unpack_flow_sample(sample) = begin
    sample isa NamedTuple || throw(ArgumentError("each dataset element must be a NamedTuple"))
    (haskey(sample, :context) && haskey(sample, :sample)) ||
        throw(ArgumentError("each dataset element must have keys `:context` and `:sample`"))
    return sample.context, sample.sample
end

_as_vec(x) = x isa AbstractVector ? x : vec(x)

"""
    train!(model_a, model_b, data_iter; kwargs...) -> NamedTuple

Train a *pair* of [`InvertibleCoupling`](@ref) networks using the two-network game gradient
[`TrainingAPI.gradient(model, other, x_true, context)`](@ref).

At each update step, the true samples come from the dataset, and each network is updated with a
**single full loss** that combines:
- accept true samples (pull inside box in its own latent space)
- reject opponent samples (push opponent-generated samples outside the box in its own latent space)
- fool opponent (generate samples that appear inside the box to the opponent)

The updates are **symmetric** per step:
1. Form a batch of `(context, x_true)`.
2. Compute gradients for `model_a` vs `model_b`, and `model_b` vs `model_a` (both before any updates).
3. Apply both updates.

The dataset iterator must be **re-iterable** (same meaning as in `NormalizingFlows.train!`): each epoch
starts from the beginning when iterating `data_iter` again.

# Arguments
- `model_a`: first [`InvertibleCoupling`](@ref), updated in-place.
- `model_b`: second [`InvertibleCoupling`](@ref), updated in-place.
- `data_iter`: iterable dataset. Each element must be a named tuple `(; context, sample)`:
  - `context`: `AbstractVector` of length `context_dim` (or a `context_dim×B` matrix if already batched).
  - `sample`: `AbstractVector` of length `dim` (or a `dim×B` matrix if already batched).

# Keyword Arguments
- `epochs=1`: number of dataset passes. `epochs=0` performs no updates (but still loads from `load_path`
  if present and saves to `save_path` if enabled).
- `batch_size=32`: batch size used when `data_iter` yields single samples (vectors).
- `margin_true=0.5`: margin passed to the true inclusion hinge.
- `margin_adv=0.0`: margin passed to adversarial hinges (rejecting and fooling).
- `w_true=1.0`: weight passed to the true inclusion loss.
- `w_reject=1.0`: weight passed to the "reject other" loss.
- `w_fool=1.0`: weight passed to the "fool other" loss.
- `use_memory=false`: if `true`, maintain a memory batch of the highest-hinge-loss *true* samples
  separately for each network. Once filled (first full batch), each update uses the concatenated
  true batch `[fresh  memory]` (so `2*batch_size` columns) and updates the memory by selecting the
  `batch_size` highest per-sample true hinge losses over this combined batch (pre-update).
- `opt=Flux.Adam(1f-3)`: optimiser rule used for *both* models.
- `rng=Random.default_rng()`: RNG passed to [`gradient`](@ref) to sample latents for fake generation.
- `save_path=""`: checkpoint path; empty disables saving.
- `load_path=save_path`: checkpoint path to load from if it exists; empty disables loading.
- `save_period=60.0`: minimum time (seconds) between periodic saves.

# Returns
Named tuple `(; model_a, model_b, losses_a, losses_b)` where:
- `losses_a::Vector{Float32}`: per-update loss trace for `model_a`.
- `losses_b::Vector{Float32}`: per-update loss trace for `model_b`.
"""
function train!(model_a::InvertibleCoupling,
                model_b::InvertibleCoupling,
                data_iter;
                epochs::Integer=1,
                batch_size::Integer=32,
                margin_true::Real=0.5,
                margin_adv::Real=0.0,
                w_true::Real=1.0,
                w_reject::Real=1.0,
                w_fool::Real=1.0,
                use_memory::Bool=false,
                opt=Flux.Adam(1f-3),
                rng::Random.AbstractRNG=Random.default_rng(),
                save_path::AbstractString="",
                load_path::AbstractString=save_path,
                save_period::Real=60.0)
    # Ensure the two networks are compatible with the same dataset batches.
    model_a.dim == model_b.dim || throw(DimensionMismatch("model_a.dim must match model_b.dim"))
    model_a.context_dim == model_b.context_dim || throw(DimensionMismatch("model_a.context_dim must match model_b.context_dim"))
    epochs >= 0 || throw(ArgumentError("epochs must be non-negative"))

    save_path_final = String(save_path)
    load_path_final = String(load_path)
    save_enabled = !isempty(save_path_final)
    load_enabled = !isempty(load_path_final)
    save_period = Float64(save_period)
    save_period >= 0 || throw(ArgumentError("save_period must be non-negative"))

    # If checkpointing is enabled, ensure the parent directory exists.
    if save_enabled
        mkpath(dirname(save_path_final))
    end

    # Optional checkpoint load (happens before optimiser setup, and may replace model objects).
    losses_a_loaded = nothing
    losses_b_loaded = nothing
    if load_enabled && isfile(load_path_final)
        @warn "Loading InvertibleGame checkpoint from $load_path_final"
        loaded_a, loaded_b, meta = load_game(load_path_final)
        model_a = loaded_a
        model_b = loaded_b
        losses_a_loaded = meta.losses_a
        losses_b_loaded = meta.losses_b
    end

    opt_state_a = Flux.setup(opt, model_a)
    opt_state_b = Flux.setup(opt, model_b)

    losses_a = Float32[]
    losses_b = Float32[]
    if losses_a_loaded isa AbstractVector
        append!(losses_a, Float32.(losses_a_loaded))
    end
    if losses_b_loaded isa AbstractVector
        append!(losses_b, Float32.(losses_b_loaded))
    end
    last_save = time()

    # Optional memory of hard (highest hinge-loss) true samples, stored separately per network.
    x_mem_a = zeros(Float32, model_a.dim, batch_size)
    c_mem_a = zeros(Float32, model_a.context_dim, batch_size)
    mem_filled_a = false

    x_mem_b = zeros(Float32, model_b.dim, batch_size)
    c_mem_b = zeros(Float32, model_b.context_dim, batch_size)
    mem_filled_b = false

    update_memory!(x_mem::Matrix{Float32},
                   c_mem::Matrix{Float32},
                   mem_filled::Bool,
                   x_cand::Matrix{Float32},
                   c_cand::Matrix{Float32},
                   true_hinges::AbstractVector{<:Real}) = begin
        size(x_cand, 2) == size(c_cand, 2) || throw(DimensionMismatch("candidate context batch must match candidate sample batch"))
        length(true_hinges) == size(x_cand, 2) || throw(DimensionMismatch("true_hinges length must match candidate batch"))
        size(x_mem, 2) == batch_size || throw(ArgumentError("memory batch size mismatch"))

        # Only update memory when we can keep exactly `batch_size` samples.
        size(x_cand, 2) >= batch_size || return mem_filled

        keep = partialsortperm(Float32.(true_hinges), 1:batch_size; rev=true) # highest hinge losses
        x_mem .= x_cand[:, keep]
        c_mem .= c_cand[:, keep]
        return true
    end

    for _ in 1:Int(epochs)
        ctx_buf = Any[]
        x_buf = Any[]

        flush_batch!() = begin
            isempty(x_buf) && return nothing

            x_true = Float32.(reduce(hcat, map(_as_vec, x_buf)))
            context = Float32.(reduce(hcat, map(_as_vec, ctx_buf)))
            empty!(x_buf)
            empty!(ctx_buf)

            # If memory is enabled and already filled, train on the concatenated true batch.
            x_a = (use_memory && mem_filled_a && size(x_true, 2) == batch_size) ? hcat(x_true, x_mem_a) : x_true
            c_a = (use_memory && mem_filled_a && size(x_true, 2) == batch_size) ? hcat(context, c_mem_a) : context

            x_b = (use_memory && mem_filled_b && size(x_true, 2) == batch_size) ? hcat(x_true, x_mem_b) : x_true
            c_b = (use_memory && mem_filled_b && size(x_true, 2) == batch_size) ? hcat(context, c_mem_b) : context

            # Symmetric gradient computation (both gradients computed before any updates).
            if use_memory
                grads_a, loss_a, extras_a = gradient(model_a, model_b, x_a, c_a;
                                                     margin_true=margin_true, margin_adv=margin_adv, rng=rng,
                                                     w_true=w_true, w_reject=w_reject, w_fool=w_fool,
                                                     return_loss=true, return_true_hinges=true)
                grads_b, loss_b, extras_b = gradient(model_b, model_a, x_b, c_b;
                                                     margin_true=margin_true, margin_adv=margin_adv, rng=rng,
                                                     w_true=w_true, w_reject=w_reject, w_fool=w_fool,
                                                     return_loss=true, return_true_hinges=true)

                # Update memory pre-update, using the true hinge losses over the combined batch.
                mem_filled_a = update_memory!(x_mem_a, c_mem_a, mem_filled_a, x_a, c_a, extras_a.true_hinges)
                mem_filled_b = update_memory!(x_mem_b, c_mem_b, mem_filled_b, x_b, c_b, extras_b.true_hinges)
            else
                grads_a, loss_a = gradient(model_a, model_b, x_a, c_a;
                                           margin_true=margin_true, margin_adv=margin_adv, rng=rng,
                                           w_true=w_true, w_reject=w_reject, w_fool=w_fool,
                                           return_loss=true)
                grads_b, loss_b = gradient(model_b, model_a, x_b, c_b;
                                           margin_true=margin_true, margin_adv=margin_adv, rng=rng,
                                           w_true=w_true, w_reject=w_reject, w_fool=w_fool,
                                           return_loss=true)
            end

            Flux.update!(opt_state_a, model_a, grads_a)
            Flux.update!(opt_state_b, model_b, grads_b)

            push!(losses_a, Float32(loss_a))
            push!(losses_b, Float32(loss_b))

            if save_enabled && (time() - last_save) >= save_period
                save_game(save_path_final, model_a, model_b; losses_a=losses_a, losses_b=losses_b)
                last_save = time()
            end
            return nothing
        end

        for item in data_iter
            context, x = _unpack_flow_sample(item)

            if (x isa AbstractMatrix) != (context isa AbstractMatrix)
                throw(ArgumentError("context and sample must both be vectors or both be matrices"))
            end

            if x isa AbstractMatrix
                size(x, 1) == model_a.dim || throw(DimensionMismatch("sample must have $(model_a.dim) rows"))
                size(context, 1) == model_a.context_dim || throw(DimensionMismatch("context must have $(model_a.context_dim) rows"))
                size(context, 2) == size(x, 2) || throw(DimensionMismatch("context batch must match sample batch"))

                # Treat as already batched: one optimiser update per yielded batch.
                x_true = Float32.(Matrix(x))
                c_batch = Float32.(Matrix(context))
                x_a = (use_memory && mem_filled_a && size(x_true, 2) == batch_size) ? hcat(x_true, x_mem_a) : x_true
                c_a = (use_memory && mem_filled_a && size(x_true, 2) == batch_size) ? hcat(c_batch, c_mem_a) : c_batch
                x_b = (use_memory && mem_filled_b && size(x_true, 2) == batch_size) ? hcat(x_true, x_mem_b) : x_true
                c_b = (use_memory && mem_filled_b && size(x_true, 2) == batch_size) ? hcat(c_batch, c_mem_b) : c_batch

                if use_memory
                    grads_a, loss_a, extras_a = gradient(model_a, model_b, x_a, c_a;
                                                         margin_true=margin_true, margin_adv=margin_adv, rng=rng,
                                                         w_true=w_true, w_reject=w_reject, w_fool=w_fool,
                                                         return_loss=true, return_true_hinges=true)
                    grads_b, loss_b, extras_b = gradient(model_b, model_a, x_b, c_b;
                                                         margin_true=margin_true, margin_adv=margin_adv, rng=rng,
                                                         w_true=w_true, w_reject=w_reject, w_fool=w_fool,
                                                         return_loss=true, return_true_hinges=true)
                    mem_filled_a = update_memory!(x_mem_a, c_mem_a, mem_filled_a, x_a, c_a, extras_a.true_hinges)
                    mem_filled_b = update_memory!(x_mem_b, c_mem_b, mem_filled_b, x_b, c_b, extras_b.true_hinges)
                else
                    grads_a, loss_a = gradient(model_a, model_b, x_a, c_a;
                                               margin_true=margin_true, margin_adv=margin_adv, rng=rng,
                                               w_true=w_true, w_reject=w_reject, w_fool=w_fool,
                                               return_loss=true)
                    grads_b, loss_b = gradient(model_b, model_a, x_b, c_b;
                                               margin_true=margin_true, margin_adv=margin_adv, rng=rng,
                                               w_true=w_true, w_reject=w_reject, w_fool=w_fool,
                                               return_loss=true)
                end
                Flux.update!(opt_state_a, model_a, grads_a)
                Flux.update!(opt_state_b, model_b, grads_b)
                push!(losses_a, Float32(loss_a))
                push!(losses_b, Float32(loss_b))

                if save_enabled && (time() - last_save) >= save_period
                    save_game(save_path_final, model_a, model_b; losses_a=losses_a, losses_b=losses_b)
                    last_save = time()
                end
                continue
            end

            push!(ctx_buf, context)
            push!(x_buf, x)
            if length(x_buf) >= batch_size
                flush_batch!()
            end
        end

        flush_batch!()
    end

    # Save once at the end so short runs still produce a checkpoint even if `save_period` is large.
    if save_enabled
        save_game(save_path_final, model_a, model_b; losses_a=losses_a, losses_b=losses_b)
    end

    return (; model_a, model_b, losses_a, losses_b)
end

"""
    build(InvertibleCoupling, data_iter; kwargs...) -> (model_a, model_b, losses_a, losses_b)

Construct and train *two* [`InvertibleCoupling`](@ref) networks from a dataset iterator.

`dim` and `context_dim` are inferred from the first element of `data_iter` (assumed re-iterable),
and all other construction arguments are provided as keyword arguments.

# Arguments
- `InvertibleCoupling`: network type.
- `data_iter`: iterable dataset of named tuples `(; context, sample)`.

# Keyword Arguments
- `epochs=1`: passed to [`train!`](@ref). `epochs=0` performs no updates (but can still load/save checkpoints).
- `batch_size=32`: passed to [`train!`](@ref).
- `margin_true=0.5`: passed to [`train!`](@ref).
- `margin_adv=0.0`: passed to [`train!`](@ref).
- `w_true=1.0`: passed to [`train!`](@ref).
- `w_reject=1.0`: passed to [`train!`](@ref).
- `w_fool=1.0`: passed to [`train!`](@ref).
- `use_memory=false`: passed to [`train!`](@ref).
- `opt=Flux.Adam(1f-3)`: passed to [`train!`](@ref).
- `rng=Random.default_rng()`: RNG passed to [`train!`](@ref) / [`gradient`](@ref) for latent sampling.
- `save_path=""`: passed to [`train!`](@ref) (empty disables saving).
- `load_path=save_path`: passed to [`train!`](@ref).
- `save_period=60.0`: passed to [`train!`](@ref).
- `rng_a=Random.default_rng()`: RNG used to initialize `model_a` (permutations/weights).
- `rng_b=Random.default_rng()`: RNG used to initialize `model_b` (permutations/weights).
- `spec=nothing`: passed to `InvertibleCoupling(dim, context_dim; spec=...)` for both models.
- `logscale_clamp=2.0`: passed to both models.

# Returns
`(model_a, model_b, losses_a, losses_b)`
"""
function build(::Type{InvertibleCoupling},
               data_iter;
               epochs::Integer=1,
               batch_size::Integer=32,
               margin_true::Real=0.5,
               margin_adv::Real=0.0,
               w_true::Real=1.0,
               w_reject::Real=1.0,
               w_fool::Real=1.0,
               use_memory::Bool=false,
               opt=Flux.Adam(1f-3),
               rng::Random.AbstractRNG=Random.default_rng(),
               save_path::AbstractString="",
               load_path::AbstractString=save_path,
               save_period::Real=60.0,
               rng_a::Random.AbstractRNG=Random.default_rng(),
               rng_b::Random.AbstractRNG=Random.default_rng(),
               spec=nothing,
               logscale_clamp::Real=2.0)
    first_item = iterate(data_iter)
    first_item === nothing && throw(ArgumentError("data_iter is empty"))
    item, _ = first_item
    context0, x0 = _unpack_flow_sample(item)

    if (x0 isa AbstractVector) != (context0 isa AbstractVector)
        throw(ArgumentError("context and sample must both be vectors or both be matrices"))
    end

    dim = x0 isa AbstractVector ? length(x0) : size(x0, 1)
    context_dim = context0 isa AbstractVector ? length(context0) : size(context0, 1)

    model_a = InvertibleCoupling(dim, context_dim; spec=spec, logscale_clamp=logscale_clamp, rng=rng_a)
    model_b = InvertibleCoupling(dim, context_dim; spec=spec, logscale_clamp=logscale_clamp, rng=rng_b)

    out = train!(model_a, model_b, data_iter;
                 epochs=epochs,
                 batch_size=batch_size,
                 margin_true=margin_true,
                 margin_adv=margin_adv,
                 w_true=w_true,
                 w_reject=w_reject,
                 w_fool=w_fool,
                 use_memory=use_memory,
                 opt=opt,
                 rng=rng,
                 save_path=save_path,
                 load_path=load_path,
                 save_period=save_period)

    return out.model_a, out.model_b, out.losses_a, out.losses_b
end

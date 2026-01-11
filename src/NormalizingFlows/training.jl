import Flux
using Random
using Statistics: mean

import ..TrainingAPI: build, train!
import ..TrainingAPI: save, load
import ..TrainingAPI: gradient

_maybe_path(path::Union{Nothing,AbstractString})::Union{Nothing,String} = begin
    path === nothing && return nothing
    s = String(path)
    isempty(s) && return nothing
    return s
end

_unpack_flow_sample(sample) = begin
    sample isa NamedTuple || throw(ArgumentError("each dataset element must be a NamedTuple"))
    (haskey(sample, :context) && haskey(sample, :sample)) ||
        throw(ArgumentError("each dataset element must have keys `:context` and `:sample`"))
    return sample.context, sample.sample
end

_as_vec(x) = x isa AbstractVector ? x : vec(x)

_merge_same_sign_average(a, b) = begin
    if a === nothing && b === nothing
        return nothing
    elseif a isa Number && b isa Number
        return sign(a) == sign(b) ? 0.5 * (a + b) : zero(promote_type(typeof(a), typeof(b)))
    elseif a isa AbstractArray && b isa AbstractArray
        size(a) == size(b) || throw(DimensionMismatch("gradient leaf size mismatch: $(size(a)) vs $(size(b))"))
        if eltype(a) <: Number && eltype(b) <: Number
            mask = sign.(a) .== sign.(b)
            return 0.5 .* (a .+ b) .* mask
        else
            # Container arrays (e.g. Vector of per-layer gradient NamedTuples).
            return map(_merge_same_sign_average, a, b)
        end
    elseif a isa NamedTuple && b isa NamedTuple
        keys(a) == keys(b) || throw(ArgumentError("gradient NamedTuple keys mismatch"))
        return NamedTuple{keys(a)}(map(k -> _merge_same_sign_average(getfield(a, k), getfield(b, k)), keys(a)))
    elseif a isa Tuple && b isa Tuple
        length(a) == length(b) || throw(ArgumentError("gradient tuple length mismatch"))
        return map(_merge_same_sign_average, a, b)
    else
        throw(ArgumentError("unsupported gradient leaf types: $(typeof(a)) and $(typeof(b))"))
    end
end

"""
    train!(model, data_iter; kwargs...) -> NamedTuple

Train a `NormalizingFlow` by maximum-likelihood under a standard normal prior using [`gradient`](@ref).

The training loop runs for `epochs` passes over `data_iter`, which is assumed to be *re-iterable*.

Re-iterable means you can do `for _ in 1:epochs; for sample in data_iter; ...; end; end`
and each epoch starts from the beginning of the dataset. Most collection-like iterables (`Vector`,
`Tuple`, `Base.Generator`, etc.) are re-iterable by default. If you implement your own iterator,
make it re-iterable by ensuring `Base.iterate(iter)` returns the first element every time, and
the iteration progress is stored only in the `state` returned by `iterate` (e.g. an index), not
by mutating `iter` itself.

# Arguments
- `model`: [`NormalizingFlow`](@ref) updated in-place.
- `data_iter`: iterable dataset. Each element must be a named tuple `(; context, sample)`:
  - `context`: `AbstractVector` of length `context_dim` (or a `context_dim×B` matrix if already batched).
  - `sample`: `AbstractVector` of length `dim` (or a `dim×B` matrix if already batched).

# Keyword Arguments
- `epochs=1`: number of dataset passes.
- `epochs=1`: number of dataset passes. `epochs=0` performs no updates (but still loads from `load_path`
  if present and saves to `save_path` if enabled).
- `batch_size=32`: batch size used when `data_iter` yields single samples (vectors).
- `use_memory=false`: if `true`, maintain a memory batch of the lowest log-likelihood samples seen so far.
  Once the memory is filled (first full batch), each gradient step uses the concatenated batch
  `[fresh  memory]` (so `2*batch_size` columns). After every weight update, the memory is updated by
  scoring `fresh` and the previous memory together and keeping the `batch_size` worst (lowest log-likelihood)
  samples.
- `memory_merge=:concat`: how to combine fresh/memory information when `use_memory=true` and the memory is filled:
  - `:concat`: compute one gradient on the concatenated batch `[fresh  memory]`.
  - `:sign_agree`: compute separate gradients on `fresh` and `memory`, then keep only those gradient components
    where both gradients have the same sign (elementwise), averaging them and zeroing the rest.
- `opt=Flux.Adam(1f-3)`: optimiser rule passed to `Flux.setup`.
- `save_path=""`: checkpoint path; empty/`nothing` disables saving.
- `load_path=save_path`: checkpoint path to load from if it exists; empty/`nothing` disables loading.
- `save_period=60.0`: minimum time (seconds) between periodic saves.

# Returns
Named tuple `(; model, losses)` where:
- `model`: updated `NormalizingFlow`.
- `losses::Vector{Float32}`: negative log-likelihood trace (mean over batch).
- `state_before`: `Flux.state(model)` snapshot taken right before the first optimizer update (after optional checkpoint load).
- `scores_fresh::Vector{Float32}`: per-update mean contrastive score on the fresh batch (defined below).
- `scores_memory::Vector{Float32}`: per-update mean contrastive score on the memory batch when available, else `NaN32`.

# Contrastive score (memory selection)
When `use_memory=true`, the memory is updated using a *contrastive* score per sample:
1. For each context `c`, sample latent `z ~ N(0, I)` and generate `x_gen = decode(model, z, c)`.
2. Compute `logp_true = log p(x_true | c)` and `logp_gen = log p(x_gen | c)`.
3. Score: `score = logp_true - logp_gen`.
The memory retains the `batch_size` samples with the *lowest* score among candidates.
"""
function train!(model::NormalizingFlow,
                data_iter;
                epochs::Integer=1,
                batch_size::Integer=32,
                use_memory::Bool=false,
                memory_merge::Symbol=:concat,
                score_rng::Random.AbstractRNG=Random.default_rng(),
                opt=Flux.Adam(1f-3),
                save_path::Union{Nothing,AbstractString}="",
                load_path::Union{Nothing,AbstractString}=save_path,
                save_period::Real=60.0)
    epochs >= 0 || throw(ArgumentError("epochs must be non-negative"))
    batch_size >= 1 || throw(ArgumentError("batch_size must be ≥ 1"))
    save_period >= 0 || throw(ArgumentError("save_period must be non-negative"))
    memory_merge in (:concat, :sign_agree) || throw(ArgumentError("memory_merge must be :concat or :sign_agree"))

    save_path_final = _maybe_path(save_path)
    load_path_final = _maybe_path(load_path)

    if load_path_final !== nothing && isfile(load_path_final)
        @warn "Starting NormalizingFlow training from saved checkpoint" load_path=load_path_final
        loaded = load(NormalizingFlow, load_path_final)
        Flux.loadmodel!(model, Flux.state(loaded))
    end

    # Snapshot parameters/state before any optimizer updates.
    #
    # Important: `Flux.state(model)` returns a nested structure containing arrays that are *owned by the model*.
    # If we return it directly, it will alias the model parameters and will therefore change as training updates
    # the model. `deepcopy` ensures `state_before` is a true snapshot for before/after comparisons.
    state_before = deepcopy(Flux.state(model))
    opt_state = Flux.setup(opt, model)
    losses = Float32[]
    scores_fresh = Float32[]
    scores_memory = Float32[]
    last_save = time()

    # Optional hard-sample memory (stores the lowest log-likelihood samples).
    x_mem = zeros(Float32, model.dim, batch_size)
    c_mem = zeros(Float32, model.context_dim, batch_size)
    mem_filled = false

    logp_per_sample(m::NormalizingFlow, x_batch::AbstractMatrix, c_batch::AbstractMatrix) = begin
        z, logdet = encode(m, x_batch, c_batch)
        logp_z = -0.5f0 .* vec(sum(z .^ 2; dims=1))
        return logp_z .+ logdet
    end

    contrastive_scores(m::NormalizingFlow, x_true::AbstractMatrix, c::AbstractMatrix) = begin
        D, B = size(x_true)
        size(c, 2) == B || throw(DimensionMismatch("context batch must match sample batch"))
        z = randn(score_rng, Float32, D, B)
        x_gen = decode(m, z, c)
        logp_true = logp_per_sample(m, x_true, c)
        logp_gen = logp_per_sample(m, x_gen, c)
        return logp_true .- logp_gen
    end

    update_memory!(x_fresh::Matrix{Float32}, c_fresh::Matrix{Float32}) = begin
        size(x_fresh, 2) == batch_size || return nothing
        if !mem_filled
            x_mem .= x_fresh
            c_mem .= c_fresh
            mem_filled = true
            return nothing
        end

        # Score fresh + previous memory using the updated model, keep the worst (lowest contrastive score).
        x_cand = hcat(x_fresh, x_mem)
        c_cand = hcat(c_fresh, c_mem)
        score = contrastive_scores(model, x_cand, c_cand)
        keep = partialsortperm(score, 1:batch_size; rev=false)
        x_mem .= x_cand[:, keep]
        c_mem .= c_cand[:, keep]
        return nothing
    end

    for _ in 1:Int(epochs)
        # Vector yields -> accumulate into a batch; matrix yields -> treated as already batched.
        ctx_buf = Any[]
        x_buf = Any[]

        flush_batch!() = begin
            isempty(x_buf) && return nothing
            x_batch = Float32.(reduce(hcat, map(_as_vec, x_buf)))
            c_batch = Float32.(reduce(hcat, map(_as_vec, ctx_buf)))
            empty!(x_buf)
            empty!(ctx_buf)

            if use_memory && mem_filled && size(x_batch, 2) == batch_size
                if memory_merge == :concat
                    x_used = hcat(x_batch, x_mem)
                    c_used = hcat(c_batch, c_mem)
                    grads, loss = gradient(model, x_used, c_used; return_loss=true)
                    Flux.update!(opt_state, model, grads)
                    push!(losses, loss)
                else
                    grads_f, loss_f = gradient(model, x_batch, c_batch; return_loss=true)
                    grads_m, loss_m = gradient(model, x_mem, c_mem; return_loss=true)
                    grads = _merge_same_sign_average(grads_f, grads_m)
                    Flux.update!(opt_state, model, grads)
                    push!(losses, 0.5f0 * (loss_f + loss_m))
                end
            else
                grads, loss = gradient(model, x_batch, c_batch; return_loss=true)
                Flux.update!(opt_state, model, grads)
                push!(losses, loss)
            end

            # Log contrastive scores using the updated model (and freshly sampled `z`).
            push!(scores_fresh, Float32(mean(contrastive_scores(model, x_batch, c_batch))))
            if use_memory && mem_filled && size(x_batch, 2) == batch_size
                push!(scores_memory, Float32(mean(contrastive_scores(model, x_mem, c_mem))))
            else
                push!(scores_memory, Float32(NaN))
            end

            if use_memory
                update_memory!(x_batch, c_batch)
            end

            if save_path_final !== nothing && (time() - last_save) >= save_period
                save(model, save_path_final; losses=losses)
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
                size(x, 1) == model.dim || throw(DimensionMismatch("sample must have $(model.dim) rows"))
                size(context, 1) == model.context_dim || throw(DimensionMismatch("context must have $(model.context_dim) rows"))
                size(context, 2) == size(x, 2) || throw(DimensionMismatch("context batch must match sample batch"))

                # If we already buffered vector samples, train them first to keep ordering clean.
                flush_batch!()

                x_batch = Float32.(Matrix(x))
                c_batch = Float32.(Matrix(context))
                if use_memory && mem_filled && size(x_batch, 2) == batch_size
                    if memory_merge == :concat
                        x_used = hcat(x_batch, x_mem)
                        c_used = hcat(c_batch, c_mem)
                        grads, loss = gradient(model, x_used, c_used; return_loss=true)
                        Flux.update!(opt_state, model, grads)
                        push!(losses, loss)
                    else
                        grads_f, loss_f = gradient(model, x_batch, c_batch; return_loss=true)
                        grads_m, loss_m = gradient(model, x_mem, c_mem; return_loss=true)
                        grads = _merge_same_sign_average(grads_f, grads_m)
                        Flux.update!(opt_state, model, grads)
                        push!(losses, 0.5f0 * (loss_f + loss_m))
                    end
                else
                    grads, loss = gradient(model, x_batch, c_batch; return_loss=true)
                    Flux.update!(opt_state, model, grads)
                    push!(losses, loss)
                end

                push!(scores_fresh, Float32(mean(contrastive_scores(model, x_batch, c_batch))))
                if use_memory && mem_filled && size(x_batch, 2) == batch_size
                    push!(scores_memory, Float32(mean(contrastive_scores(model, x_mem, c_mem))))
                else
                    push!(scores_memory, Float32(NaN))
                end
                if use_memory
                    update_memory!(x_batch, c_batch)
                end
            else
                x_vec = Vector(x)
                c_vec = Vector(context)
                length(x_vec) == model.dim || throw(DimensionMismatch("sample must have length $(model.dim)"))
                length(c_vec) == model.context_dim || throw(DimensionMismatch("context must have length $(model.context_dim)"))

                push!(x_buf, x_vec)
                push!(ctx_buf, c_vec)
                if length(x_buf) >= batch_size
                    flush_batch!()
                end
            end
        end

        flush_batch!()
    end

    if save_path_final !== nothing
        save(model, save_path_final; losses=losses)
    end

    return (; model=model,
            losses=losses,
            scores_fresh=scores_fresh,
            scores_memory=scores_memory,
            state_before=state_before)
end

"""
    build(::Type{NormalizingFlow}, data_iter; kwargs...) -> (model, losses)
    build(::Type{NormalizingFlow}, data_iter; return_state_before=true, kwargs...) -> (model, losses, state_before)
    build(::Type{NormalizingFlow}, data_iter; return_scores=true, kwargs...) -> (model, losses, scores_fresh, scores_memory)
    build(::Type{NormalizingFlow}, data_iter; return_scores=true, return_state_before=true, kwargs...) -> (model, losses, scores_fresh, scores_memory, state_before)

Construct and train a `NormalizingFlow` from dataset dimensions.

This is a convenience wrapper around [`train!`](@ref) that:
1. Infers `dim` and `context_dim` from the first element of `data_iter`.
2. Optionally resumes from a checkpoint if `load_path` is provided and exists.

# Keyword Arguments (construction)
- `spec=nothing`: coupling stack spec (see [`NormalizingFlow`](@ref)).
- `logscale_clamp=2.0`: `tanh` clamp for affine log-scales.
- `rng=Random.default_rng()`: RNG used to initialize permutations (only when not resuming from checkpoint).

# Keyword Arguments (training)
All other keyword arguments are forwarded to [`train!`](@ref), including optimiser and checkpoint options.

# Keyword Arguments (return)
- `return_state_before=false`: when `true`, also return `state_before` (the `Flux.state(model)` snapshot
  taken right before training updates).
- `return_scores=false`: when `true`, also return `scores_fresh` and `scores_memory` from [`train!`](@ref).

# Returns
- If `return_state_before=false` (default): `(model, losses)`.
- If `return_state_before=true`: `(model, losses, state_before)`.
- If `return_scores=true`: `(model, losses, scores_fresh, scores_memory)`.
- If `return_scores=true` and `return_state_before=true`: `(model, losses, scores_fresh, scores_memory, state_before)`.
"""
function build(::Type{NormalizingFlow},
               data_iter;
               spec=nothing,
               logscale_clamp::Real=2.0,
               rng::Random.AbstractRNG=Random.default_rng(),
               return_state_before::Bool=false,
               return_scores::Bool=false,
               kwargs...)
    st = iterate(data_iter)
    st === nothing && throw(ArgumentError("data_iter is empty"))
    first_item = st[1]
    context, x = _unpack_flow_sample(first_item)

    if (x isa AbstractMatrix) != (context isa AbstractMatrix)
        throw(ArgumentError("context and sample must both be vectors or both be matrices"))
    end

    dim = x isa AbstractMatrix ? size(x, 1) : length(x)
    context_dim = context isa AbstractMatrix ? size(context, 1) : length(context)

    train_kwargs = (; kwargs...)
    save_path_final = haskey(train_kwargs, :save_path) ? _maybe_path(train_kwargs.save_path) : nothing
    load_path_final = haskey(train_kwargs, :load_path) ? _maybe_path(train_kwargs.load_path) : save_path_final

    model = if load_path_final !== nothing && isfile(load_path_final)
        @warn "Building NormalizingFlow from saved checkpoint (will continue training)" load_path=load_path_final
        load(NormalizingFlow, load_path_final)
    else
        NormalizingFlow(dim, context_dim; spec=spec, logscale_clamp=logscale_clamp, rng=rng)
    end

    res = train!(model, data_iter; train_kwargs...)

    if return_scores && return_state_before
        return (res.model, res.losses, res.scores_fresh, res.scores_memory, res.state_before)
    elseif return_scores
        return (res.model, res.losses, res.scores_fresh, res.scores_memory)
    elseif return_state_before
        return (res.model, res.losses, res.state_before)
    else
        return (res.model, res.losses)
    end
end

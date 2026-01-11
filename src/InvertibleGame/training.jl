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

_ema_beta(step::Integer, beta_start::Real, beta_final::Real, tau::Real)::Float32 = begin
    β0 = Float32(beta_start)
    βf = Float32(beta_final)
    τ = Float32(tau)
    (0f0 <= β0 <= 1f0) || throw(ArgumentError("ema_beta_start must be in [0, 1]; got $beta_start"))
    (0f0 <= βf <= 1f0) || throw(ArgumentError("ema_beta_final must be in [0, 1]; got $beta_final"))
    isfinite(τ) && (τ > 0f0) || throw(ArgumentError("ema_tau must be finite and > 0; got $tau"))
    t = Float32(step)
    return βf - (βf - β0) * exp(-t / τ)
end

_ema_update!(ema_model, live_model, β::Float32) = begin
    ema_ps = Flux.trainables(ema_model)
    live_ps = Flux.trainables(live_model)
    length(ema_ps) == length(live_ps) || throw(ArgumentError("EMA model structure mismatch"))
    α = 1f0 - β
    for i in eachindex(ema_ps)
        ema_ps[i] .= β .* ema_ps[i] .+ α .* live_ps[i]
    end
    return ema_model
end

_update_memory(x_mem::Matrix{Float32},
               c_mem::Matrix{Float32},
               batch_size::Int,
               x_cand::Matrix{Float32},
               c_cand::Matrix{Float32},
               true_norms::AbstractVector{<:Real}) = begin
    size(x_cand, 2) == size(c_cand, 2) ||
        throw(DimensionMismatch("candidate context batch must match candidate sample batch"))
    length(true_norms) == size(x_cand, 2) ||
        throw(DimensionMismatch("true_norms length must match candidate batch"))

    v = Float32.(max.(0, true_norms .- 1))
    idx = findall(>(0f0), v)
    if isempty(idx)
        return (zeros(Float32, size(x_mem, 1), 0), zeros(Float32, size(c_mem, 1), 0), false)
    end

    k = min(batch_size, length(idx))
    keep_local = partialsortperm(view(v, idx), 1:k; rev=true)
    keep = idx[keep_local]
    return (x_cand[:, keep], c_cand[:, keep], true)
end

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
- `norm_kind=:l1`: norm used in hinge losses (`:l1`, `:l2`, or `:linf`).
	- `w_true=1.0`: weight passed to the true inclusion loss.
	- `w_reject=1.0`: weight passed to the "reject other" loss.
	- `w_fool=1.0`: weight passed to the "fool other" loss.
	- `latent_radius_min=0.0`: minimum radius in `[0, 1]` for sampled fake latents used in the adversarial terms.
	- `ema_beta_start=0.0`: initial EMA decay.
	- `ema_beta_final=0.9999`: final EMA decay (asymptotic value).
	- `ema_tau=10000.0`: EMA schedule time constant (in optimizer steps). The decay used at step `t` is
	  `β(t) = β_final - (β_final - β_start) * exp(-t/τ)`.
- `use_memory=false`: if `true`, maintain a *variable-size* memory of hard true samples (cap `batch_size`)
  separately for each network. Each update uses the concatenated true batch `[fresh  memory]` and updates the
  memory by selecting up to `batch_size` samples with the highest *violation hinge*
  `relu(‖z_true‖ - 1)` over this combined batch (pre-update). This ignores samples with `‖z_true‖ ≤ 1`, so
  the memory (and thus the combined batch size) can be smaller than `batch_size`.
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
	                norm_kind::Symbol=:l1,
	                w_true::Real=1.0,
	                w_reject::Real=1.0,
	                w_fool::Real=1.0,
	                grad_mode::Symbol=:sum,
	                ema_beta_start::Real=0.0,
	                ema_beta_final::Real=0.9999,
	                ema_tau::Real=10000.0,
	                latent_radius_min::Real=0.0,
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
	    (grad_mode === :sum || grad_mode === :orthogonal_adv) ||
	        throw(ArgumentError("unsupported grad_mode=$(repr(grad_mode)); expected :sum or :orthogonal_adv"))

    # If checkpointing is enabled, ensure the parent directory exists.
    if save_enabled
        mkpath(dirname(save_path_final))
    end

    # Optional checkpoint load (happens before optimiser setup, and may replace model objects).
    losses_a_loaded = nothing
	    losses_b_loaded = nothing
	    ema_a = nothing
	    ema_b = nothing
	    ema_beta_start_loaded = nothing
	    ema_beta_final_loaded = nothing
	    ema_tau_loaded = nothing
	    ema_step_loaded = nothing
	    if load_enabled && isfile(load_path_final)
	        @warn "Loading InvertibleGame checkpoint from $load_path_final"
	        loaded_a, loaded_b, meta = load_game(load_path_final)
        model_a = loaded_a
        model_b = loaded_b
	        losses_a_loaded = meta.losses_a
		        losses_b_loaded = meta.losses_b
	        ema_a = meta.ema_a
	        ema_b = meta.ema_b
	        ema_beta_start_loaded = get(meta, :ema_beta_start, nothing)
	        ema_beta_final_loaded = get(meta, :ema_beta_final, nothing)
	        ema_tau_loaded = get(meta, :ema_tau, nothing)
	        ema_step_loaded = get(meta, :ema_step, nothing)
	    end

    opt_state_a_true = Flux.setup(opt, model_a)
    opt_state_b_true = Flux.setup(opt, model_b)
    opt_state_a_adv = Flux.setup(opt, model_a)
    opt_state_b_adv = Flux.setup(opt, model_b)

    # EMA opponents (used only as the `other` argument in `gradient`).
    if ema_a === nothing
        ema_a = deepcopy(model_a)
    end
	    if ema_b === nothing
	        ema_b = deepcopy(model_b)
	    end

	    # EMA schedule (resume from checkpoint if present).
	    ema_beta_start_f = Float32(ema_beta_start_loaded === nothing ? ema_beta_start : ema_beta_start_loaded)
	    ema_beta_final_f = Float32(ema_beta_final_loaded === nothing ? ema_beta_final : ema_beta_final_loaded)
	    ema_tau_f = Float32(ema_tau_loaded === nothing ? ema_tau : ema_tau_loaded)
	    ema_step = Int(ema_step_loaded === nothing ? 0 : ema_step_loaded)
	    _ = _ema_beta(ema_step, ema_beta_start_f, ema_beta_final_f, ema_tau_f) # validation

	    losses_a = Float32[]
	    losses_b = Float32[]
    if losses_a_loaded isa AbstractVector
        append!(losses_a, Float32.(losses_a_loaded))
    end
    if losses_b_loaded isa AbstractVector
        append!(losses_b, Float32.(losses_b_loaded))
    end
    last_save = time()

    # Optional memory of hard true samples (variable size up to `batch_size`), stored separately per network.
    # We select by a zero-margin violation hinge `relu(‖z_true‖ - 1)`, so samples with ‖z_true‖ ≤ 1 do not contribute.
    x_mem_a = zeros(Float32, model_a.dim, 0)
    c_mem_a = zeros(Float32, model_a.context_dim, 0)
    mem_filled_a = false

    x_mem_b = zeros(Float32, model_b.dim, 0)
    c_mem_b = zeros(Float32, model_b.context_dim, 0)
    mem_filled_b = false

    for _ in 1:Int(epochs)
        ctx_buf = Any[]
        x_buf = Any[]

        flush_batch!() = begin
            isempty(x_buf) && return nothing

            x_true = Float32.(reduce(hcat, map(_as_vec, x_buf)))
            context = Float32.(reduce(hcat, map(_as_vec, ctx_buf)))
            empty!(x_buf)
            empty!(ctx_buf)

            # If memory is enabled and filled, train on the concatenated true batch.
            x_a = (use_memory && mem_filled_a) ? hcat(x_true, x_mem_a) : x_true
            c_a = (use_memory && mem_filled_a) ? hcat(context, c_mem_a) : context

            x_b = (use_memory && mem_filled_b) ? hcat(x_true, x_mem_b) : x_true
            c_b = (use_memory && mem_filled_b) ? hcat(context, c_mem_b) : context

            # Symmetric gradient computation (both gradients computed before any updates).
	            if use_memory
	                grads_a, loss_a, extras_a = gradient(model_a, ema_b, x_a, c_a;
	                                                     margin_true=margin_true, margin_adv=margin_adv, rng=rng,
	                                                     norm_kind=norm_kind,
	                                                     latent_radius_min=latent_radius_min,
	                                                     w_true=w_true, w_reject=w_reject, w_fool=w_fool,
	                                                     mode=grad_mode,
	                                                     return_loss=true, return_true_hinges=true)
	                grads_b, loss_b, extras_b = gradient(model_b, ema_a, x_b, c_b;
	                                                     margin_true=margin_true, margin_adv=margin_adv, rng=rng,
	                                                     norm_kind=norm_kind,
	                                                     latent_radius_min=latent_radius_min,
	                                                     w_true=w_true, w_reject=w_reject, w_fool=w_fool,
	                                                     mode=grad_mode,
	                                                     return_loss=true, return_true_hinges=true)

                # Update memory pre-update, using the zero-margin violation hinge `relu(‖z_true‖ - 1)`.
                x_mem_a, c_mem_a, mem_filled_a = _update_memory(x_mem_a, c_mem_a, batch_size, x_a, c_a, extras_a.true_norms)
                x_mem_b, c_mem_b, mem_filled_b = _update_memory(x_mem_b, c_mem_b, batch_size, x_b, c_b, extras_b.true_norms)
	            else
	                grads_a, loss_a = gradient(model_a, ema_b, x_a, c_a;
	                                           margin_true=margin_true, margin_adv=margin_adv, rng=rng,
	                                           norm_kind=norm_kind,
	                                           latent_radius_min=latent_radius_min,
	                                           w_true=w_true, w_reject=w_reject, w_fool=w_fool,
	                                           mode=grad_mode,
	                                           return_loss=true)
	                grads_b, loss_b = gradient(model_b, ema_a, x_b, c_b;
	                                           margin_true=margin_true, margin_adv=margin_adv, rng=rng,
	                                           norm_kind=norm_kind,
	                                           latent_radius_min=latent_radius_min,
	                                           w_true=w_true, w_reject=w_reject, w_fool=w_fool,
	                                           mode=grad_mode,
	                                           return_loss=true)
	            end
	
	            if grad_mode === :sum
	                Flux.update!(opt_state_a_true, model_a, grads_a)
	                Flux.update!(opt_state_b_true, model_b, grads_b)
	            else
	                g_a_true, g_a_adv = grads_a
	                g_b_true, g_b_adv = grads_b
	                Flux.update!(opt_state_a_true, model_a, g_a_true)
	                Flux.update!(opt_state_a_adv, model_a, g_a_adv)
	                Flux.update!(opt_state_b_true, model_b, g_b_true)
	                Flux.update!(opt_state_b_adv, model_b, g_b_adv)
	            end
	            β = _ema_beta(ema_step, ema_beta_start_f, ema_beta_final_f, ema_tau_f)
	            _ema_update!(ema_a, model_a, β)
	            _ema_update!(ema_b, model_b, β)
	            ema_step += 1

            push!(losses_a, Float32(loss_a))
            push!(losses_b, Float32(loss_b))

	            if save_enabled && (time() - last_save) >= save_period
	                save_game(save_path_final, model_a, model_b;
	                          losses_a=losses_a, losses_b=losses_b,
	                          ema_a=ema_a, ema_b=ema_b,
	                          ema_beta_start=ema_beta_start_f,
	                          ema_beta_final=ema_beta_final_f,
	                          ema_tau=ema_tau_f,
	                          ema_step=ema_step)
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
                x_a = (use_memory && mem_filled_a) ? hcat(x_true, x_mem_a) : x_true
                c_a = (use_memory && mem_filled_a) ? hcat(c_batch, c_mem_a) : c_batch
                x_b = (use_memory && mem_filled_b) ? hcat(x_true, x_mem_b) : x_true
                c_b = (use_memory && mem_filled_b) ? hcat(c_batch, c_mem_b) : c_batch

	                if use_memory
	                    grads_a, loss_a, extras_a = gradient(model_a, ema_b, x_a, c_a;
	                                                         margin_true=margin_true, margin_adv=margin_adv, rng=rng,
	                                                         norm_kind=norm_kind,
	                                                         latent_radius_min=latent_radius_min,
	                                                         w_true=w_true, w_reject=w_reject, w_fool=w_fool,
	                                                         mode=grad_mode,
	                                                         return_loss=true, return_true_hinges=true)
	                    grads_b, loss_b, extras_b = gradient(model_b, ema_a, x_b, c_b;
	                                                         margin_true=margin_true, margin_adv=margin_adv, rng=rng,
	                                                         norm_kind=norm_kind,
	                                                         latent_radius_min=latent_radius_min,
	                                                         w_true=w_true, w_reject=w_reject, w_fool=w_fool,
	                                                         mode=grad_mode,
	                                                         return_loss=true, return_true_hinges=true)
	                    x_mem_a, c_mem_a, mem_filled_a = _update_memory(x_mem_a, c_mem_a, batch_size, x_a, c_a, extras_a.true_norms)
	                    x_mem_b, c_mem_b, mem_filled_b = _update_memory(x_mem_b, c_mem_b, batch_size, x_b, c_b, extras_b.true_norms)
	                else
	                    grads_a, loss_a = gradient(model_a, ema_b, x_a, c_a;
	                                               margin_true=margin_true, margin_adv=margin_adv, rng=rng,
	                                               norm_kind=norm_kind,
	                                               latent_radius_min=latent_radius_min,
	                                               w_true=w_true, w_reject=w_reject, w_fool=w_fool,
	                                               mode=grad_mode,
	                                               return_loss=true)
	                    grads_b, loss_b = gradient(model_b, ema_a, x_b, c_b;
	                                               margin_true=margin_true, margin_adv=margin_adv, rng=rng,
	                                               norm_kind=norm_kind,
	                                               latent_radius_min=latent_radius_min,
	                                               w_true=w_true, w_reject=w_reject, w_fool=w_fool,
	                                               mode=grad_mode,
	                                               return_loss=true)
	                end
	                if grad_mode === :sum
	                    Flux.update!(opt_state_a_true, model_a, grads_a)
	                    Flux.update!(opt_state_b_true, model_b, grads_b)
	                else
	                    g_a_true, g_a_adv = grads_a
	                    g_b_true, g_b_adv = grads_b
	                    Flux.update!(opt_state_a_true, model_a, g_a_true)
	                    Flux.update!(opt_state_a_adv, model_a, g_a_adv)
	                    Flux.update!(opt_state_b_true, model_b, g_b_true)
	                    Flux.update!(opt_state_b_adv, model_b, g_b_adv)
	                end
		                β = _ema_beta(ema_step, ema_beta_start_f, ema_beta_final_f, ema_tau_f)
		                _ema_update!(ema_a, model_a, β)
		                _ema_update!(ema_b, model_b, β)
		                ema_step += 1
                push!(losses_a, Float32(loss_a))
                push!(losses_b, Float32(loss_b))

	                if save_enabled && (time() - last_save) >= save_period
	                    save_game(save_path_final, model_a, model_b;
	                              losses_a=losses_a, losses_b=losses_b,
	                              ema_a=ema_a, ema_b=ema_b,
	                              ema_beta_start=ema_beta_start_f,
	                              ema_beta_final=ema_beta_final_f,
	                              ema_tau=ema_tau_f,
	                              ema_step=ema_step)
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
	        save_game(save_path_final, model_a, model_b;
	                  losses_a=losses_a, losses_b=losses_b,
	                  ema_a=ema_a, ema_b=ema_b,
	                  ema_beta_start=ema_beta_start_f,
	                  ema_beta_final=ema_beta_final_f,
	                  ema_tau=ema_tau_f,
	                  ema_step=ema_step)
	    end

    return (; model_a, model_b, losses_a, losses_b)
end

"""
    train!(model, data_iter; kwargs...) -> NamedTuple

Self-adversarial training for a single [`InvertibleCoupling`](@ref), using an EMA copy to provide fake samples.

The live model is trained to:
- include true samples (pull inside its own latent ball), and
- reject fake samples produced by the EMA model (push outside its own latent ball).

The fake samples are generated by decoding latents sampled from the chosen norm ball (same sampler used elsewhere).

# Arguments
- `model`: [`InvertibleCoupling`](@ref), updated in-place.
- `data_iter`: iterable dataset of named tuples `(; context, sample)` (same contract as the two-player training).

# Keyword Arguments
- `epochs=1`: number of dataset passes. `epochs=0` performs no updates (but still loads/saves if enabled).
- `batch_size=32`: batch size used when `data_iter` yields single samples (vectors).
- `margin_true=0.5`: margin passed to the true inclusion hinge.
- `margin_adv=0.0`: margin passed to the reject hinge (fake rejection).
- `norm_kind=:l1`: norm used in hinge losses (`:l1`, `:l2`, or `:linf`).
- `w_true=1.0`: weight on the true inclusion loss.
- `w_reject=1.0`: weight on the fake rejection loss.
- `latent_radius_min=0.0`: minimum radius in `[0, 1]` for sampled fake latents.
- `ema_beta_start=0.0`: initial EMA decay.
- `ema_beta_final=0.9999`: final EMA decay.
- `ema_tau=10000.0`: EMA schedule time constant (in optimizer steps).
- `use_memory=false`: reuse the same hard-sample memory mechanism as the two-player training (variable size).
- `opt=Flux.Adam(1f-3)`: optimiser rule used for the model.
- `rng=Random.default_rng()`: RNG passed to [`gradient`](@ref) to sample latents for fake generation.
- `save_path=""`: checkpoint path; empty disables saving.
- `load_path=save_path`: checkpoint path to load from if it exists; empty disables loading.
- `save_period=60.0`: minimum time (seconds) between periodic saves.

# Returns
Named tuple `(; model, ema, losses)` where:
- `ema`: EMA copy used as opponent.
- `losses::Vector{Float32}`: per-update loss trace for `model`.
"""
	function train!(model::InvertibleCoupling,
	                data_iter;
	                epochs::Integer=1,
	                batch_size::Integer=32,
	                margin_true::Real=0.5,
	                margin_adv::Real=0.0,
	                norm_kind::Symbol=:l1,
	                w_true::Real=1.0,
	                w_reject::Real=1.0,
	                grad_mode::Symbol=:sum,
	                ema_beta_start::Real=0.0,
	                ema_beta_final::Real=0.9999,
	                ema_tau::Real=10000.0,
	                latent_radius_min::Real=0.0,
	                use_memory::Bool=false,
	                opt=Flux.Adam(1f-3),
	                rng::Random.AbstractRNG=Random.default_rng(),
	                save_path::AbstractString="",
	                load_path::AbstractString=save_path,
                save_period::Real=60.0)
    epochs >= 0 || throw(ArgumentError("epochs must be non-negative"))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))

    save_path_final = String(save_path)
    load_path_final = String(load_path)
    save_enabled = !isempty(save_path_final)
    load_enabled = !isempty(load_path_final)
	    save_period = Float64(save_period)
	    save_period >= 0 || throw(ArgumentError("save_period must be non-negative"))
	    (grad_mode === :sum || grad_mode === :orthogonal_adv) ||
	        throw(ArgumentError("unsupported grad_mode=$(repr(grad_mode)); expected :sum or :orthogonal_adv"))

    if save_enabled
        mkpath(dirname(save_path_final))
    end

	    losses_loaded = nothing
	    ema = nothing
	    ema_beta_start_loaded = nothing
	    ema_beta_final_loaded = nothing
	    ema_tau_loaded = nothing
	    ema_step_loaded = nothing
	    if load_enabled && isfile(load_path_final)
	        @warn "Loading InvertibleGame self checkpoint from $load_path_final"
	        loaded_model, meta = load_self(load_path_final)
	        model = loaded_model
	        losses_loaded = meta.losses
	        ema = meta.ema
	        ema_beta_start_loaded = get(meta, :ema_beta_start, nothing)
	        ema_beta_final_loaded = get(meta, :ema_beta_final, nothing)
	        ema_tau_loaded = get(meta, :ema_tau, nothing)
	        ema_step_loaded = get(meta, :ema_step, nothing)
	    end

    opt_state_true = Flux.setup(opt, model)
    opt_state_adv = Flux.setup(opt, model)

    losses = Float32[]
    if losses_loaded isa AbstractVector
        append!(losses, Float32.(losses_loaded))
    end
    last_save = time()

	    if ema === nothing
	        ema = deepcopy(model)
	    end

	    ema_beta_start_f = Float32(ema_beta_start_loaded === nothing ? ema_beta_start : ema_beta_start_loaded)
	    ema_beta_final_f = Float32(ema_beta_final_loaded === nothing ? ema_beta_final : ema_beta_final_loaded)
	    ema_tau_f = Float32(ema_tau_loaded === nothing ? ema_tau : ema_tau_loaded)
	    ema_step = Int(ema_step_loaded === nothing ? 0 : ema_step_loaded)
	    _ = _ema_beta(ema_step, ema_beta_start_f, ema_beta_final_f, ema_tau_f) # validation

	    x_mem = zeros(Float32, model.dim, 0)
	    c_mem = zeros(Float32, model.context_dim, 0)
    mem_filled = false

    for _ in 1:Int(epochs)
        ctx_buf = Any[]
        x_buf = Any[]

        flush_batch!() = begin
            isempty(x_buf) && return nothing

            x_true = Float32.(reduce(hcat, map(_as_vec, x_buf)))
            context = Float32.(reduce(hcat, map(_as_vec, ctx_buf)))
            empty!(x_buf)
            empty!(ctx_buf)

            x_batch = (use_memory && mem_filled) ? hcat(x_true, x_mem) : x_true
            c_batch = (use_memory && mem_filled) ? hcat(context, c_mem) : context

            if use_memory
	                grads, loss, extras = gradient(model, ema, x_batch, c_batch;
	                                               margin_true=margin_true, margin_adv=margin_adv, rng=rng,
	                                               norm_kind=norm_kind,
	                                               latent_radius_min=latent_radius_min,
	                                               w_true=w_true, w_reject=w_reject, w_fool=0.0,
	                                               mode=grad_mode,
	                                               return_loss=true, return_true_hinges=true)
                x_mem, c_mem, mem_filled = _update_memory(x_mem, c_mem, batch_size, x_batch, c_batch, extras.true_norms)
            else
	                grads, loss = gradient(model, ema, x_batch, c_batch;
	                                       margin_true=margin_true, margin_adv=margin_adv, rng=rng,
	                                       norm_kind=norm_kind,
	                                       latent_radius_min=latent_radius_min,
	                                       w_true=w_true, w_reject=w_reject, w_fool=0.0,
	                                       mode=grad_mode,
	                                       return_loss=true)
            end

            if grad_mode === :sum
                Flux.update!(opt_state_true, model, grads)
            else
                g_true, g_adv = grads
                Flux.update!(opt_state_true, model, g_true)
                Flux.update!(opt_state_adv, model, g_adv)
            end
            β = _ema_beta(ema_step, ema_beta_start_f, ema_beta_final_f, ema_tau_f)
            _ema_update!(ema, model, β)
            ema_step += 1
            push!(losses, Float32(loss))

            if save_enabled && (time() - last_save) >= save_period
                save_self(save_path_final, model;
                          losses=losses,
                          ema=ema,
                          ema_beta_start=ema_beta_start_f,
                          ema_beta_final=ema_beta_final_f,
                          ema_tau=ema_tau_f,
                          ema_step=ema_step)
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

                x_true = Float32.(Matrix(x))
                c_batch = Float32.(Matrix(context))
                x_batch = (use_memory && mem_filled) ? hcat(x_true, x_mem) : x_true
                c_comb = (use_memory && mem_filled) ? hcat(c_batch, c_mem) : c_batch

	                if use_memory
		                    grads, loss, extras = gradient(model, ema, x_batch, c_comb;
		                                                   margin_true=margin_true, margin_adv=margin_adv, rng=rng,
		                                                   norm_kind=norm_kind,
		                                                   latent_radius_min=latent_radius_min,
		                                                   w_true=w_true, w_reject=w_reject, w_fool=0.0,
		                                                   mode=grad_mode,
		                                                   return_loss=true, return_true_hinges=true)
	                    x_mem, c_mem, mem_filled = _update_memory(x_mem, c_mem, batch_size, x_batch, c_comb, extras.true_norms)
	                else
		                    grads, loss = gradient(model, ema, x_batch, c_comb;
		                                           margin_true=margin_true, margin_adv=margin_adv, rng=rng,
		                                           norm_kind=norm_kind,
		                                           latent_radius_min=latent_radius_min,
		                                           w_true=w_true, w_reject=w_reject, w_fool=0.0,
		                                           mode=grad_mode,
		                                           return_loss=true)
	                end
	
	                if grad_mode === :sum
	                    Flux.update!(opt_state_true, model, grads)
	                else
	                    g_true, g_adv = grads
	                    Flux.update!(opt_state_true, model, g_true)
	                    Flux.update!(opt_state_adv, model, g_adv)
	                end
	                β = _ema_beta(ema_step, ema_beta_start_f, ema_beta_final_f, ema_tau_f)
	                _ema_update!(ema, model, β)
	                ema_step += 1
	                push!(losses, Float32(loss))

	                if save_enabled && (time() - last_save) >= save_period
                    save_self(save_path_final, model;
                              losses=losses,
                              ema=ema,
                              ema_beta_start=ema_beta_start_f,
                              ema_beta_final=ema_beta_final_f,
                              ema_tau=ema_tau_f,
                              ema_step=ema_step)
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

	    if save_enabled
	        save_self(save_path_final, model;
	                  losses=losses,
	                  ema=ema,
	                  ema_beta_start=ema_beta_start_f,
	                  ema_beta_final=ema_beta_final_f,
	                  ema_tau=ema_tau_f,
	                  ema_step=ema_step)
	    end

    return (; model, ema, losses)
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
- `norm_kind=:l1`: passed to [`train!`](@ref).
- `w_true=1.0`: passed to [`train!`](@ref).
- `w_reject=1.0`: passed to [`train!`](@ref).
- `w_fool=1.0`: passed to [`train!`](@ref).
- `ema_beta_start=0.0`: passed to [`train!`](@ref).
- `ema_beta_final=0.9999`: passed to [`train!`](@ref).
- `ema_tau=10000.0`: passed to [`train!`](@ref).
- `latent_radius_min=0.0`: passed to [`train!`](@ref).
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
	               norm_kind::Symbol=:l1,
	               w_true::Real=1.0,
	               w_reject::Real=1.0,
	               w_fool::Real=1.0,
	               grad_mode::Symbol=:sum,
	               ema_beta_start::Real=0.0,
	               ema_beta_final::Real=0.9999,
	               ema_tau::Real=10000.0,
	               latent_radius_min::Real=0.0,
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
	                 norm_kind=norm_kind,
	                 w_true=w_true,
	                 w_reject=w_reject,
	                 w_fool=w_fool,
	                 grad_mode=grad_mode,
	                 ema_beta_start=ema_beta_start,
	                 ema_beta_final=ema_beta_final,
	                 ema_tau=ema_tau,
	                 latent_radius_min=latent_radius_min,
	                 use_memory=use_memory,
	                 opt=opt,
	                 rng=rng,
	                 save_path=save_path,
                 load_path=load_path,
                 save_period=save_period)

    return out.model_a, out.model_b, out.losses_a, out.losses_b
end

"""
    build(InvertibleCoupling, data_iter, mode; kwargs...) -> result

Build variants for InvertibleGame training.

- `mode=:game` trains two networks (same as `build(InvertibleCoupling, data_iter; ...)`).
- `mode=:self` trains a single network with self-adversarial EMA fakes.
"""
	function build(::Type{InvertibleCoupling},
	               data_iter,
	               mode::Symbol;
	               epochs::Integer=1,
	               batch_size::Integer=32,
	               margin_true::Real=0.5,
	               margin_adv::Real=0.0,
	               norm_kind::Symbol=:l1,
	               w_true::Real=1.0,
	               w_reject::Real=1.0,
	               w_fool::Real=1.0,
	               grad_mode::Symbol=:sum,
	               ema_beta_start::Real=0.0,
	               ema_beta_final::Real=0.9999,
	               ema_tau::Real=10000.0,
	               latent_radius_min::Real=0.0,
	               use_memory::Bool=false,
	               opt=Flux.Adam(1f-3),
	               rng::Random.AbstractRNG=Random.default_rng(),
	               save_path::AbstractString="",
               load_path::AbstractString=save_path,
               save_period::Real=60.0,
               rng_model::Random.AbstractRNG=Random.default_rng(),
               spec=nothing,
               logscale_clamp::Real=2.0)
    mode === :self || mode === :game || throw(ArgumentError("unsupported mode=$(repr(mode)); expected :self or :game"))
	    if mode === :game
	        # Delegate to the existing two-player builder; use its own RNG keywords for init.
	        return build(InvertibleCoupling, data_iter;
	                     epochs=epochs,
	                     batch_size=batch_size,
	                     margin_true=margin_true,
	                     margin_adv=margin_adv,
	                     norm_kind=norm_kind,
	                     w_true=w_true,
	                     w_reject=w_reject,
	                     w_fool=w_fool,
	                     grad_mode=grad_mode,
	                     ema_beta_start=ema_beta_start,
	                     ema_beta_final=ema_beta_final,
	                     ema_tau=ema_tau,
	                     latent_radius_min=latent_radius_min,
	                     use_memory=use_memory,
	                     opt=opt,
	                     rng=rng,
                     save_path=save_path,
                     load_path=load_path,
                     save_period=save_period,
                     spec=spec,
                     logscale_clamp=logscale_clamp)
    end

    first_item = iterate(data_iter)
    first_item === nothing && throw(ArgumentError("data_iter is empty"))
    item, _ = first_item
    context0, x0 = _unpack_flow_sample(item)

    if (x0 isa AbstractVector) != (context0 isa AbstractVector)
        throw(ArgumentError("context and sample must both be vectors or both be matrices"))
    end

    dim = x0 isa AbstractVector ? length(x0) : size(x0, 1)
    context_dim = context0 isa AbstractVector ? length(context0) : size(context0, 1)
    model = InvertibleCoupling(dim, context_dim; spec=spec, logscale_clamp=logscale_clamp, rng=rng_model)

	    out = train!(model, data_iter;
	                 epochs=epochs,
	                 batch_size=batch_size,
	                 margin_true=margin_true,
	                 margin_adv=margin_adv,
	                 norm_kind=norm_kind,
	                 w_true=w_true,
	                 w_reject=w_reject,
	                 grad_mode=grad_mode,
	                 ema_beta_start=ema_beta_start,
	                 ema_beta_final=ema_beta_final,
	                 ema_tau=ema_tau,
	                 latent_radius_min=latent_radius_min,
	                 use_memory=use_memory,
	                 opt=opt,
	                 rng=rng,
                 save_path=save_path,
                 load_path=load_path,
                 save_period=save_period)
    return out.model, out.ema, out.losses
end

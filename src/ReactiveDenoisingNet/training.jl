import Flux
using Random

import ..TrainingAPI: build
import ..TrainingAPI: train!
import ..TrainingAPI: save, load
import ..TrainingAPI: gradient

_unpack_imitation_sample(sample) = begin
    if sample isa Tuple && length(sample) == 2
        return sample[1], sample[2]
    end
    if sample isa NamedTuple
        if haskey(sample, :x0)
            x0 = sample.x0
        elseif haskey(sample, :state)
            x0 = sample.state
        elseif haskey(sample, :x)
            x0 = sample.x
        else
            throw(ArgumentError("sample must provide `x0` (or `state`/`x`)"))
        end

        if haskey(sample, :u_target)
            u_target = sample.u_target
        elseif haskey(sample, :u)
            u_target = sample.u
        elseif haskey(sample, :input_signal)
            u_target = sample.input_signal
        else
            throw(ArgumentError("sample must provide `u_target` (or `u`/`input_signal`)"))
        end
        return x0, u_target
    end
    throw(ArgumentError("sample must be a 2-tuple (x0, u_target) or a NamedTuple with x0/state and u_target/u"))
end

_maybe_path(path::Union{Nothing,AbstractString})::Union{Nothing,String} = begin
    path === nothing && return nothing
    s = String(path)
    isempty(s) && return nothing
    return s
end

"""
    train!(model, data_iter, sys, traj_cost_fn; kwargs...) -> NamedTuple

Train `ReactiveDenoisingNet` by imitation learning using the final-step-only [`gradient`](@ref) helper.

The training loop runs for `epochs` passes over `data_iter`, which is assumed to be *re-iterable*.

# Arguments
- `model`: `ReactiveDenoisingNet` model to update in-place.
- `data_iter`: iterable dataset; each element must provide `(x0, u_target)` where `x0` is a state vector
  of length `state_dim` and `u_target` is an `input_dim×seq_len` matrix (columns are time steps).
  Accepted formats:
  - a 2-tuple `(x0, u_target)`, or
  - a `NamedTuple` with keys `(x0|state|x)` and `(u_target|u|input_signal)`.
- `sys`: rollout function `sys(x0::Vector, U::Matrix) -> X::Matrix` returning `state_dim×(seq_len+1)`.
- `traj_cost_fn`: cost function `traj_cost_fn(x_body::Matrix) -> cost_body::Matrix` returning `cost_dim×seq_len`,
  where `x_body = X[:, 2:end]`.

# Keyword Arguments
- `epochs=1`: number of dataset passes.
- `steps=1`: number of refinement iterations used by [`gradient`](@ref).
- `opt=Flux.Adam(1f-3)`: Flux optimiser (or `Optimisers.jl` rule) used with `Flux.setup`.
- `rng=Random.default_rng()`: RNG used for rectified-flow guess sampling.
- `scale=nothing`: optional per-row scaling vector (length `input_dim`) applied to both prediction and target before `Flux.mse`.
- `save_path=""`: checkpoint path; empty/`nothing` disables saving.
- `load_path=save_path`: checkpoint path to load from if it exists; empty/`nothing` disables loading.
- `save_period=60.0`: minimum time (seconds) between periodic saves.

# Notes
Initial guess uses a rectified-flow style mixture:
`u0 = (1-λ) * randn(...) + λ * u_target`, with `λ ~ Uniform(0,1)`.

# Returns
Named tuple `(; model, losses)` where:
- `model`: the updated `ReactiveDenoisingNet`.
- `losses::Vector{Float32}`: MSE trace measured on the refined guess produced by `model(x0, u0, sys, traj_cost_fn; steps=steps)`.
"""
function train!(model::ReactiveDenoisingNet,
                data_iter,
                sys,
                traj_cost_fn;
                epochs::Integer=1,
                steps::Integer=1,
                opt=Flux.Adam(1f-3),
                rng::Random.AbstractRNG=Random.default_rng(),
                scale=nothing,
                save_path::Union{Nothing,AbstractString}="",
                load_path::Union{Nothing,AbstractString}=save_path,
                save_period::Real=60.0)
    epochs >= 1 || throw(ArgumentError("epochs must be ≥ 1"))
    steps >= 1 || throw(ArgumentError("steps must be ≥ 1"))
    save_period > 0 || throw(ArgumentError("save_period must be positive"))

    save_path_final = _maybe_path(save_path)
    load_path_final = _maybe_path(load_path)

    if load_path_final !== nothing && isfile(load_path_final)
        loaded = load(ReactiveDenoisingNet, load_path_final)
        Flux.loadmodel!(model, Flux.state(loaded))
    end

    opt_state = Flux.setup(opt, model)
    losses = Float32[]
    last_save = time()

    for _ in 1:Int(epochs)
        for sample in data_iter
            x0, u_target = _unpack_imitation_sample(sample)
            u_target_mat = Float32.(Matrix(u_target))
            size(u_target_mat, 1) == model.input_dim ||
                throw(DimensionMismatch("u_target must have $(model.input_dim) rows; got $(size(u_target_mat, 1))"))
            size(u_target_mat, 2) == model.seq_len ||
                throw(DimensionMismatch("u_target must have $(model.seq_len) columns; got $(size(u_target_mat, 2))"))

            λ = rand(rng, Float32)
            noise = randn(rng, Float32, size(u_target_mat)...)
            u0 = (1f0 - λ) .* noise .+ λ .* u_target_mat

            grads = gradient(model, Vector(x0), u0, u_target_mat, sys, traj_cost_fn; steps=steps, scale=scale)
            Flux.update!(opt_state, model, grads)

            # Track loss (computed outside the gradient closure on the current parameters).
            out = model(Vector(x0), u0, sys, traj_cost_fn; steps=steps)
            u_final = out.u_guesses[end]
            if scale === nothing
                push!(losses, Float32(Flux.mse(u_final, u_target_mat)))
            else
                s_mat = reshape(Float32.(collect(scale)), :, 1)
                push!(losses, Float32(Flux.mse(u_final .* s_mat, u_target_mat .* s_mat)))
            end

            if save_path_final !== nothing && (time() - last_save) >= save_period
                save(model, save_path_final; losses=losses)
                last_save = time()
            end
        end
    end

    if save_path_final !== nothing
        save(model, save_path_final; losses=losses)
    end

    return (; model=model, losses=losses)
end

"""
    build(::Type{ReactiveDenoisingNet}, data_iter, sys, traj_cost_fn; kwargs...) -> Tuple{ReactiveDenoisingNet, Vector{Float32}}

Construct and train a `ReactiveDenoisingNet` from dataset/system dimensions.

This is a convenience wrapper around [`train!`](@ref) that:
1. Infers `state_dim`, `input_dim`, and `seq_len` from the first element of `data_iter`.
2. Infers `cost_dim` by rolling out `sys(x0, u_target)` and evaluating `traj_cost_fn` on `x_body = X[:, 2:end]`
   (unless `cost_dim` is provided explicitly).
3. Optionally resumes from a checkpoint if `load_path` is provided and exists.

# Keyword Arguments (construction)
- `hidden_dim` (required): hidden width for the `SequenceTransformation`.
- `depth` (required): number of transformer blocks.
- `cost_dim=nothing`: if set, skips cost-dimension inference and uses this value.
- `max_seq_len=nothing`, `nheads=1`, `activation=Flux.gelu`: forwarded to `ReactiveDenoisingNet(...)`.

# Keyword Arguments (training)
All other keyword arguments are forwarded to [`train!`](@ref), including checkpoint options.

# Returns
`(model, losses)` where `model` is the trained `ReactiveDenoisingNet` and `losses` is the training MSE trace.
"""
function build(::Type{ReactiveDenoisingNet},
               data_iter,
               sys,
               traj_cost_fn;
               hidden_dim::Integer,
               depth::Integer,
               cost_dim::Union{Nothing,Integer}=nothing,
               max_seq_len::Union{Nothing,Integer}=nothing,
               nheads::Integer=1,
               activation=Flux.gelu,
               kwargs...)
    first_sample = first(data_iter)
    x0, u_target = _unpack_imitation_sample(first_sample)
    x0_vec = Vector(x0)
    u_target_mat = Matrix(u_target)

    state_dim = length(x0_vec)
    input_dim, seq_len = size(u_target_mat)

    cost_dim_final = if cost_dim === nothing
        x_roll = sys(x0_vec, u_target_mat)
        size(x_roll, 1) == state_dim ||
            throw(DimensionMismatch("sys(x0, u_target) must return $state_dim rows; got $(size(x_roll, 1))"))
        size(x_roll, 2) == seq_len + 1 ||
            throw(DimensionMismatch("sys(x0, u_target) must return $(seq_len + 1) columns; got $(size(x_roll, 2))"))
        x_body = x_roll[:, 2:end]
        cost_body = traj_cost_fn(x_body)
        size(cost_body, 2) == seq_len ||
            throw(DimensionMismatch("traj_cost_fn(x_body) must return $seq_len columns; got $(size(cost_body, 2))"))
        Int(size(cost_body, 1))
    else
        Int(cost_dim)
    end

    train_kwargs = (; kwargs...)
    save_path_final = haskey(train_kwargs, :save_path) ? _maybe_path(train_kwargs.save_path) : nothing
    load_path_final = haskey(train_kwargs, :load_path) ? _maybe_path(train_kwargs.load_path) : save_path_final

    model = if load_path_final !== nothing && isfile(load_path_final)
        load(ReactiveDenoisingNet, load_path_final)
    else
        ReactiveDenoisingNet(state_dim, input_dim, cost_dim_final, seq_len, hidden_dim, depth;
                             max_seq_len=max_seq_len, nheads=nheads, activation=activation)
    end

    res = train!(model, data_iter, sys, traj_cost_fn; train_kwargs...)
    return res.model, res.losses
end

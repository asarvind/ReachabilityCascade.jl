using Flux
using Flux.Optimise: update!
using Flux.Losses: mse
using Random
using JLD2
using Statistics
import ..build

"""
    train!(model::TransitionNetwork, dataset;
           epochs=10, batchsize=32, opt=Flux.ADAM(1e-3),
           rng=Random.default_rng(), scale=nothing,
           save_path=nothing, save_period=60.0, act=Flux.σ, bias=true)

Train an existing `TransitionNetwork` on a dataset of named tuples `(state, input, next)`.

# Arguments
- `model`: A `TransitionNetwork`.
- `dataset`: Vector of named tuples with fields `state`, `input`, `next`; each field should be a `state_dim` or `input_dim` vector (one sample per tuple).
- `epochs`: Number of passes over the dataset.
- `batchsize`: Number of samples per minibatch.
- `rng`: Random number generator used for shuffling.
- `opt`: Optimiser (e.g. `Flux.ADAM(1e-3)` by default).
- `scale`: Optional per-dimension scaling (number, vector, or `nothing`), padded/truncated to `state_dim` and applied to outputs/targets before MSE.
- `save_path`: Optional checkpoint path; saved every `save_period` seconds and at completion.
- `save_period`: Seconds between checkpoints when `save_path` is set.
- `act`, `bias`: Stored alongside checkpoints for consistent reloads (used when saving during training).
- `hard_mining` (default `false`): If `true`, retains the hardest `batchsize` samples seen so far, concatenates them with each new batch, and refreshes the hard pool after every update.

# Returns
- The trained `model` (losses are recorded internally for wrappers/tests).
"""
function train!(model::TransitionNetwork, dataset;
                epochs::Int=10, batchsize::Int=32,
                opt=Flux.ADAM(1e-3), rng=Random.default_rng(), scale=nothing,
                save_path=nothing, save_period::Real=60.0, act=Flux.σ, bias::Bool=true,
                hard_mining::Bool=false)
    _, losses = _train_and_record!(model, dataset;
        epochs=epochs, batchsize=batchsize, opt=opt, rng=rng, scale=scale,
        save_path=save_path, save_period=save_period, act=act, bias=bias,
        hard_mining=hard_mining)
    return model
end

# Internal helper that preserves loss history for wrappers/tests.
function _train_and_record!(model::TransitionNetwork, dataset;
                            epochs::Int=10, batchsize::Int=32,
                            opt=Flux.ADAM(1e-3), rng=Random.default_rng(), scale=nothing,
                            save_path=nothing, save_period::Real=60.0, act=Flux.σ, bias::Bool=true,
                            hard_mining::Bool=false)
    function batch_tensors(batch)
        x = hcat(getfield.(batch, :state)...)
        u = hcat(getfield.(batch, :input)...)
        y = hcat(getfield.(batch, :next)...)
        return x, u, y
    end

    # Build per-dimension scale on the same device/dtype as targets.
    function scale_mask(y::AbstractArray)
        T = eltype(y)
        base = if scale === nothing
            ones(T, model.state_dim)
        elseif isa(scale, Number)
            fill(T(scale), model.state_dim)
        else
            vals = T.(collect(scale))
            if length(vals) >= model.state_dim
                vals[1:model.state_dim]
            else
                vcat(vals, fill(one(T), model.state_dim - length(vals)))
            end
        end
        if ndims(y) == 2
            reshape(base, :, 1)
        else
            reshape(base, :, 1, 1)
        end
    end

    losses = Float32[]
    opt_state = Flux.setup(opt, model)
    n = length(dataset)
    last_save = time()
    SampleT = eltype(dataset)
    hard_buffer = SampleT[]
    for _ in 1:epochs
        perm = randperm(rng, n)
        for i in 1:batchsize:n
            idx = perm[i:min(i + batchsize - 1, n)]
            batch = dataset[idx]
            current_batch = (hard_mining && !isempty(hard_buffer)) ? vcat(batch, hard_buffer) : batch
            x, u, y = batch_tensors(current_batch)
            s = scale_mask(y)
            gs = Flux.gradient(model) do m
                ŷ = m(x, u)
                mse(ŷ .* s, y .* s)
            end
            update!(opt_state, model, gs[1])
            y_out = model(x, u)
            push!(losses, Float32(mse(y_out .* s, y .* s)))

            if hard_mining
                per_sample = vec(mean(abs2, y_out .* s .- y .* s; dims=1))
                if length(per_sample) > batchsize
                    topk = partialsortperm(per_sample, 1:batchsize; rev=true)
                    hard_buffer = current_batch[topk]
                else
                    hard_buffer = current_batch
                end
            end

            if save_path !== nothing && (time() - last_save) >= save_period
                save_transition_network(save_path, model; act=act, bias=bias)
                last_save = time()
            end
        end
    end
    return model, losses
end

"""
    fit_transition_network(dataset; hidden_dim, depth=2, act=Flux.σ, bias=true, kwargs...)

Infer dimensions from the dataset, construct a `TransitionNetwork`, train it via [`train!`](@ref),
and return `(model, losses)`.

# Arguments
- `dataset`: Vector of `(state, input, next)` tuples.
- `hidden_dim`: Hidden width for the network.
- `depth`, `act`, `bias`: Network construction options.
- `kwargs`: Forwarded to [`train!`](@ref) (e.g. `epochs`, `batchsize`, `opt`, `rng`, `scale`, `save_period`, `save_path`).

# Returns
- `(model, losses)` where `losses` is a vector of batch MSE values.
"""
function fit_transition_network(dataset; hidden_dim::Int, depth::Int=2, act=Flux.σ, bias::Bool=true, kwargs...)
    first_item = first(dataset)
    state_dim = length(first_item.state)
    input_dim = length(first_item.input)
    @assert length(first_item.next) == state_dim "next-state dimension must match state dimension"

    # Training kwargs only (constructor is explicit via act/bias/depth)
    train_kwargs = (; kwargs...)

    model = TransitionNetwork(state_dim, input_dim, hidden_dim; depth=depth, act=act, bias=bias)
    model, losses = _train_and_record!(model, dataset; act=act, bias=bias, train_kwargs...)
    return model, losses
end

"""
    save_transition_network(path::AbstractString, model::TransitionNetwork; act=nothing, bias=nothing)

Persist the model architecture metadata (state/input/hidden dimensions and depth) and trained parameters using `Flux.state`.
Stores activation/bias choices for reproducible reloads.

# Arguments
- `path`: Destination path.
- `model`: `TransitionNetwork` to serialize.
- `act`, `bias`: Optional explicit activation/bias to store (defaults to model settings).

# Returns
- `path` (for convenience/chaining).
"""
function save_transition_network(path::AbstractString, model::TransitionNetwork; act=nothing, bias=nothing)
    state = Flux.state(model)
    args = (model.state_dim, model.input_dim, model.hidden_dim, model.depth)
    kwargs = (; act=act, bias=bias)
    jldsave(path; state, args, kwargs)
    return path
end

"""
    load_transition_network(path::AbstractString; act=nothing, bias=nothing)

Reload a saved `TransitionNetwork` (created with [`save_transition_network`](@ref)).

# Arguments
- `path`: Source path.
- `act`, `bias`: Optional overrides for stored activation/bias.

# Returns
- A `TransitionNetwork` with parameters loaded and metadata restored (legacy files without `depth` are handled).
"""
function load_transition_network(path::AbstractString; act=nothing, bias=nothing)
    data = JLD2.load(path)
    stored_args = get(data, "args", nothing)
    stored_kwargs = get(data, "kwargs", (;))
    state = data["state"]

    # Backward compatibility: if args are missing, look for legacy fields
    if stored_args === nothing
        state_dim = get(data, "state_dim", nothing)
        input_dim = get(data, "input_dim", nothing)
        hidden_dim = get(data, "hidden_dim", nothing)
        depth = get(data, "depth", 2)
    else
        if length(stored_args) == 4
            state_dim, input_dim, hidden_dim, depth = stored_args
        else
            state_dim, input_dim, hidden_dim = stored_args
            depth = get(data, "depth", 2)
        end
    end

    act_final = something(act, get(stored_kwargs, :act, nothing), Flux.σ)
    bias_final = something(bias, get(stored_kwargs, :bias, nothing), true)
    model = TransitionNetwork(state_dim, input_dim, hidden_dim; depth=depth, act=act_final, bias=bias_final)
    Flux.loadmodel!(model, state)
    return model
end

"""
    build(::Type{TransitionNetwork}, dataset, path;
          hidden_dim, depth=2, act=Flux.σ, bias=true, save_period=60.0, kwargs...)

Construct a `TransitionNetwork` from dataset dimensions, train it (continuing from a saved model if present),
save it via [`save_transition_network`](@ref), and return the trained model and loss history.

# Arguments
- `dataset`: Vector of `(state, input, next)` tuples.
- `path`: Checkpoint path; if it exists the model is reloaded and **further trained**.
- `hidden_dim`, `depth`, `act`, `bias`: Network construction options.
- `save_period`: Seconds between checkpoints during training.
- `kwargs`: Forwarded to [`train!`](@ref) (e.g., `epochs`, `batchsize`, `opt`, `rng`, `scale`, `save_path`, `save_period`, `hard_mining`); see [`train!`](@ref) for the full list.

# Returns
- `(model, losses)` where training always runs (either fresh or continued) and checkpoints are saved.
"""
function build(::Type{TransitionNetwork}, dataset, path; hidden_dim::Int, depth::Int=2, act=Flux.σ, bias::Bool=true, save_period::Real=60.0, kwargs...)
    train_kwargs = (; kwargs...)

    model = if isfile(path)
        # Reload existing model before additional training
        load_transition_network(path; act=act, bias=bias)
    else
        first_item = first(dataset)
        state_dim = length(first_item.state)
        input_dim = length(first_item.input)
        @assert length(first_item.next) == state_dim "next-state dimension must match state dimension"
        TransitionNetwork(state_dim, input_dim, hidden_dim; depth=depth, act=act, bias=bias)
    end

    model, losses = _train_and_record!(model, dataset; act=act, bias=bias,
                                       save_path=path, save_period=save_period, train_kwargs...)
    save_transition_network(path, model; act=act, bias=bias)
    return model, losses
end

import Flux
import Flux: train!, setup, update!, state, loadmodel!
using Base: time, filesize
import JLD2

const DefaultOptimiser = Flux.Optimise.OptimiserChain(Flux.Optimise.ClipGrad(1.0),
                                                      Flux.Optimise.ClipNorm(1.0),
                                                      Flux.Optimise.Adam())

_ensure_float32(x::AbstractArray) = eltype(x) === Float32 ? x : Flux.f32(x)

function _standardize_sequence(seq::AbstractArray)
    seq32 = _ensure_float32(seq)
    return ndims(seq32) == 2 ? reshape(seq32, size(seq32, 1), size(seq32, 2), 1) : seq32
end

function _standardize_context(ctx::AbstractArray)
    ctx32 = _ensure_float32(ctx)
    return ndims(ctx32) == 1 ? reshape(ctx32, :, 1) : ctx32
end

function _process_batch!(model::FlowTransformer,
                         opt_state,
                         batch::Vector{Tuple{AbstractArray,AbstractArray}},
                         buffer::Vector{Tuple{AbstractArray,AbstractArray}},
                         batch_size::Integer,
                         loss_fn::Function)
    if isempty(batch) && isempty(buffer)
        return
    end

    all_samples = Tuple{AbstractArray,AbstractArray}[]
    append!(all_samples, batch)
    append!(all_samples, buffer)

    isempty(all_samples) && return

    length_groups = Dict{Int, Vector{Tuple{AbstractArray,AbstractArray}}}()
    for sample in all_samples
        seq = sample[2]
        seq_len = ndims(seq) >= 2 ? size(seq, 2) : 1
        push!(get!(length_groups, seq_len, Tuple{AbstractArray,AbstractArray}[]), sample)
    end

    for seq_len in sort!(collect(keys(length_groups)))
        group_samples = length_groups[seq_len]
        ctx_list = [ctx for (ctx, _) in group_samples]
        seq_list = [seq for (_, seq) in group_samples]
        ctx_batch = length(ctx_list) == 1 ? ctx_list[1] : cat(ctx_list...; dims=2)
        seq_batch = length(seq_list) == 1 ? seq_list[1] : cat(seq_list...; dims=3)
        grads = flow_transformer_gradient(model, seq_batch, ctx_batch; loss_fn=loss_fn)
        update!(opt_state, model, grads)
    end

    combined = Vector{Tuple{AbstractArray,AbstractArray,Float64}}()
    for (context, sequence) in all_samples
        latent, logdet = model(sequence, context)
        push!(combined, (context, sequence, Float64(loss_fn(latent, logdet))))
    end

    sort!(combined, by = s -> s[3], rev=true)
    topk = combined[1:min(batch_size, length(combined))]
    empty!(buffer)
    append!(buffer, [(ctx, seq) for (ctx, seq, _) in topk])
end

"""
    train!(::Type{FlowTransformer}, data_iter, constructor_args...;
           optimizer=DefaultOptimiser, epochs::Integer=1,
           loss::Function=default_flow_loss, batch_size::Integer=1,
           save_path::AbstractString="",
           save_interval::Real=Inf,
           constructor_kwargs...)

Build a `FlowTransformer` using `constructor_args`/`constructor_kwargs` (or
by inferring dimensions from the first data sample when positional arguments
are omitted) and train it on samples drawn from `data_iter`. The iterator
must be re-iterable so that each epoch can traverse the data from the start.
Each element from `data_iter` must be a tuple `(context, sequence)` where
`context` is `(context_dim,)` or `(context_dim, batch)` and `sequence` is
`(d_model, seq_len, batch)` (or `(d_model, seq_len)` for single samples).

During training, batches of size `batch_size` are formed from the iterator.
Samples from the previous hardest-buffer are merged with the fresh batch and
all combined samples are used for optimisation (grouped by sequence length so
they can be concatenated safely). After the updates, the hardest `batch_size`
samples under the current loss are stored in the buffer for the next batch.

Training keywords:
- `optimizer`: Flux optimiser instance (default
  `OptimiserChain(ClipGrad(1.0), ClipNorm(1.0), Adam())`).
- `epochs`: number of passes over `data_iter`.
- `loss`: scalar loss applied to `(latent, logdet)` pairs (proportional to negative log-likelihood).
- `batch_size`: number of samples per optimisation step and maximum buffer size.
- `save_path`: output path for periodic/final checkpoints; empty string disables saving.
- `save_interval`: wall-clock seconds between checkpoints (≤ 0 saves every batch, `Inf`
  disables periodic saving). Final state is always written when `save_path` is provided.
- `load_path`: optional checkpoint to initialise from (defaults to `save_path`). If provided,
  constructor arguments must match the checkpoint unless none are supplied.
- Positional `constructor_args`: optional overrides for constructor arguments; when omitted,
  `(d_model, context_dim)` are inferred from the first sample.
- `constructor_kwargs`: forwarded to the `FlowTransformer` constructor alongside
  positional `constructor_args` (e.g. `max_seq_len`, `position_fn`, or a custom positional table).

All unconsumed keyword arguments are forwarded to the `FlowTransformer`
constructor.

# Returns
Trained `FlowTransformer` instance.
"""
function train!(::Type{FlowTransformer},
                data_iter,
                constructor_args...;
                optimizer=DefaultOptimiser,
                epochs::Integer=1,
                loss::Function=default_flow_loss,
                batch_size::Integer=1,
                save_path::AbstractString="",
           save_interval::Real=60,
           load_path::AbstractString=save_path,
                constructor_kwargs...)
    @assert batch_size > 0 "batch_size must be positive"
    @assert epochs > 0 "epochs must be positive"

    first_iter = iterate(data_iter)
    @assert first_iter !== nothing "Training data iterator must yield at least one sample"
    first_pair, _ = first_iter
    first_ctx = _standardize_context(first_pair[1])
    first_seq = _standardize_sequence(first_pair[2])
    default_args = (size(first_seq, 1), size(first_ctx, 1))

    requested_args = Tuple(constructor_args)
    requested_kwargs = (; constructor_kwargs...)
    default_kwargs = (; max_seq_len = max(size(first_seq, 2), 512))
    user_args_supplied = !isempty(requested_args)
    user_kwargs_supplied = length(requested_kwargs) > 0
    effective_args = user_args_supplied ? requested_args : default_args
    effective_kwargs = merge(default_kwargs, requested_kwargs)
    @assert effective_kwargs.max_seq_len >= size(first_seq, 2) "max_seq_len must be at least the first sequence length"
    saved_args = effective_args
    saved_kwargs = effective_kwargs
    recorded_args = user_args_supplied ? saved_args : nothing
    recorded_kwargs = user_kwargs_supplied ? saved_kwargs : nothing
    load_target = load_path
    load_available = !isempty(load_target) && ispath(load_target) && filesize(load_target) > 0

    if load_available
        data = JLD2.jldopen(load_target, "r") do file
            Dict(
                "model_state" => (haskey(file, "model_state") ? read(file, "model_state") : nothing),
                "constructor_args" => (haskey(file, "constructor_args") ? read(file, "constructor_args") : nothing),
                "constructor_kwargs" => (haskey(file, "constructor_kwargs") ? read(file, "constructor_kwargs") : nothing),
                "inferred_args" => (haskey(file, "inferred_args") ? read(file, "inferred_args") : nothing),
                "inferred_kwargs" => (haskey(file, "inferred_kwargs") ? read(file, "inferred_kwargs") : nothing),
                "position_table" => (haskey(file, "position_table") ? read(file, "position_table") : nothing),
            )
        end
        file_args = data["constructor_args"] === nothing ? data["inferred_args"] : data["constructor_args"]
        file_kwargs = data["constructor_kwargs"] === nothing ? data["inferred_kwargs"] : data["constructor_kwargs"]
        @assert file_args !== nothing "Checkpoint missing constructor args information"
        @assert file_kwargs !== nothing "Checkpoint missing constructor kwargs information"
        pos_table = data["position_table"]
        @assert pos_table !== nothing "Checkpoint missing position table information"
        file_args = Tuple(file_args)
        file_kwargs = file_kwargs isa NamedTuple ? file_kwargs : (; file_kwargs...)
        file_kwargs = haskey(file_kwargs, :position_table) ? file_kwargs : merge(file_kwargs, (; position_table = pos_table))
        if user_args_supplied
            @assert file_args == effective_args "Provided constructor args do not match checkpoint"
            saved_args = effective_args
        else
            saved_args = file_args
        end
        if user_kwargs_supplied
            for (name, value) in pairs(requested_kwargs)
                @assert haskey(file_kwargs, name) "Checkpoint missing constructor kwarg $(name)"
                @assert file_kwargs[name] == value "Provided constructor kwarg $(name) does not match checkpoint"
            end
        end
        saved_kwargs = file_kwargs
        recorded_args = user_args_supplied ? saved_args : nothing
        recorded_kwargs = user_kwargs_supplied ? saved_kwargs : nothing
        model = FlowTransformer(saved_args...; saved_kwargs...)
        if data["model_state"] !== nothing
            loadmodel!(model, data["model_state"])
        end
    else
        @assert user_args_supplied || !isempty(default_args) "Unable to infer constructor arguments from data"
        saved_args = effective_args
        saved_kwargs = effective_kwargs
        recorded_args = user_args_supplied ? saved_args : nothing
        recorded_kwargs = user_kwargs_supplied ? saved_kwargs : nothing
        model = FlowTransformer(saved_args...; saved_kwargs...)
    end
    pos_kw = (; position_table = model.position_table)
    saved_kwargs = merge(saved_kwargs, pos_kw)
    recorded_args = saved_args
    recorded_kwargs = saved_kwargs
    opt_state = setup(optimizer, model)
    hard_buffer = Vector{Tuple{AbstractArray,AbstractArray}}()
    save_enabled = !isempty(save_path)
    last_save = Ref(time())

    function maybe_save!(force::Bool=false)
        if save_enabled && (force || save_interval <= 0 || (isfinite(save_interval) && time() - last_save[] >= save_interval))
            JLD2.jldopen(save_path, "w") do file
                file["model_state"] = state(model)
                file["constructor_args"] = recorded_args
                file["constructor_kwargs"] = recorded_kwargs
                file["inferred_args"] = saved_args
                file["inferred_kwargs"] = saved_kwargs
                file["position_table"] = model.position_table
            end
            last_save[] = time()
        end
    end

    for _ in 1:epochs
        batch_samples = Tuple{AbstractArray,AbstractArray}[]
        iter_state = iterate(data_iter)
        @assert iter_state !== nothing "Training data iterator must yield at least one sample (epoch pass)."
        while iter_state !== nothing
            (raw_context, raw_sequence), state = iter_state
            context = _standardize_context(raw_context)
            sequence = _standardize_sequence(raw_sequence)
            push!(batch_samples, (context, sequence))
            if length(batch_samples) == batch_size
                _process_batch!(model, opt_state, batch_samples, hard_buffer, batch_size, loss)
                empty!(batch_samples)
                maybe_save!()
            end
            iter_state = iterate(data_iter, state)
        end
        if !isempty(batch_samples)
            _process_batch!(model, opt_state, batch_samples, hard_buffer, batch_size, loss)
            empty!(batch_samples)
            maybe_save!()
        end
    end

    maybe_save!(true)

    return model
end

"""
    load_flow_transformer(path::AbstractString)

Load a `FlowTransformer` checkpoint produced by `train!`.
Restores the model using stored constructor args/kwargs (including positional encodings)
and applies the saved state.
"""
function load_flow_transformer(path::AbstractString)
    data = JLD2.jldopen(path, "r") do file
        Dict(
            "constructor_args" => (haskey(file, "constructor_args") ? read(file, "constructor_args") : nothing),
            "constructor_kwargs" => (haskey(file, "constructor_kwargs") ? read(file, "constructor_kwargs") : nothing),
            "inferred_args" => (haskey(file, "inferred_args") ? read(file, "inferred_args") : nothing),
            "inferred_kwargs" => (haskey(file, "inferred_kwargs") ? read(file, "inferred_kwargs") : nothing),
            "position_table" => (haskey(file, "position_table") ? read(file, "position_table") : nothing),
            "model_state" => (haskey(file, "model_state") ? read(file, "model_state") : nothing),
        )
    end
    args = data["constructor_args"] === nothing ? data["inferred_args"] : data["constructor_args"]
    kwargs = data["constructor_kwargs"] === nothing ? data["inferred_kwargs"] : data["constructor_kwargs"]
    @assert args !== nothing "Checkpoint missing constructor args information"
    @assert kwargs !== nothing "Checkpoint missing constructor kwargs information"
    pos_table = data["position_table"]
    @assert pos_table !== nothing "Checkpoint missing position table information"
    args = Tuple(args)
    kwargs = kwargs isa NamedTuple ? kwargs : (; kwargs...)
    kwargs = haskey(kwargs, :position_table) ? kwargs : merge(kwargs, (; position_table = pos_table))
    model = FlowTransformer(args...; kwargs...)
    if data["model_state"] !== nothing
        loadmodel!(model, data["model_state"])
    end
    return model
end

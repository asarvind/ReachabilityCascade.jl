using Flux
using JLD2: jldsave, load
import Base.Iterators: Stateful, reset!

export train_recurrent_control!

"""
    train_recurrent_control!(net, terminal_data, intermediate_data, control_data, rule;
                             epochs=1, callback=nothing, carryover_limit=0)

Train a [`RecurrentControlNet`](@ref) using three data iterators that supply
mini-batches for the terminal, intermediate, and control gradient objectives.

Each iterator must yield a pair `(args, kwargs)` where `args` is a tuple of
positional arguments and `kwargs` is a `NamedTuple` with keyword arguments.
For every datum we compute the corresponding flow gradient, then update the
underlying conditional flow with `Flux.update!`. The optional `callback`
is invoked after each parameter update as
`callback(component::Symbol, result, epoch::Int)`, where `component` is one
of `:terminal`, `:intermediate`, or `:control`.

Set `save_path` to a non-empty string to periodically checkpoint the network
weights and (optionally) constructor metadata every `save_interval` seconds
and once more after training completes. When `load_path` points to an existing
checkpoint (defaults to `save_path`), the weights and constructor metadata are
restored before training. Checkpoints store `model_state`, `timestamp`, and
(when supplied) the constructor arguments.

The `carryover_limit` keyword bounds how many of the lowest-likelihood samples
are retained from each gradient call for reuse in the next datum (set to `0`
to keep all available samples).

Returns the trained network `net`.
"""
function train_recurrent_control!(net::RecurrentControlNet,
                                  terminal_data,
                                  intermediate_data,
                                  control_data,
                                  rule;
                                  epochs::Integer=1,
                                  callback=nothing,
                                  save_path::AbstractString="",
                                  load_path::AbstractString=save_path,
                                  save_interval::Real=60.0,
                                  constructor_info=nothing,
                                  carryover_limit::Integer=0)
    epochs > 0 || throw(ArgumentError("epochs must be positive"))
    carryover_limit >= 0 || throw(ArgumentError("carryover_limit must be non-negative"))

    opt_terminal = Flux.setup(rule, net.terminal.flow)
    opt_intermediate = Flux.setup(rule, net.intermediate.flow)
    opt_control = Flux.setup(rule, net.controller.flow)

    terminal_grad = (args, kwargs) -> terminal_flow_gradient(net, args...; kwargs...)
    intermediate_grad = (args, kwargs) -> intermediate_flow_gradient(net, args...; kwargs...)
    control_grad = (args, kwargs) -> control_flow_gradient(net, args...; kwargs...)

    constructor_meta = constructor_info
    if load_path != "" && isfile(load_path)
        stored = load(load_path)
        if haskey(stored, "model_state")
            state_loaded = stored["model_state"]
            if _tree_finite(state_loaded)
                Flux.loadmodel!(net, state_loaded)
            else
                @warn "Skipping checkpoint load due to non-finite weights" load_path maxlog=1
            end
        end
        if constructor_meta === nothing && haskey(stored, "constructor")
            constructor_meta = stored["constructor"]
        end
    end

    should_save = save_path != ""
    last_save = time()

    for epoch in 1:epochs
        terminal_memory = nothing
        _reset_iterator!(terminal_data)
        for datum in terminal_data
            args, kwargs = _unpack_training_datum(datum)
            kwargs = _with_memory(kwargs, terminal_memory)
            kwargs = _ensure_carryover_limit(kwargs, carryover_limit)
            result = terminal_grad(args, kwargs)
            if _grads_finite(result.grads)
                Flux.update!(opt_terminal, net.terminal.flow, result.grads)
            else
                @warn "Skipping terminal update due to non-finite gradients" maxlog=1
                continue
            end
            if callback !== nothing
                callback(:terminal, result, epoch)
            end
            terminal_memory = _next_memory(result)
            if should_save && (time() - last_save) >= save_interval
                last_save = _save_checkpoint(save_path, net, constructor_meta)
            end
        end

        intermediate_memory = nothing
        _reset_iterator!(intermediate_data)
        for datum in intermediate_data
            args, kwargs = _unpack_training_datum(datum)
            kwargs = _with_memory(kwargs, intermediate_memory)
            kwargs = _ensure_carryover_limit(kwargs, carryover_limit)
            result = intermediate_grad(args, kwargs)
            if _grads_finite(result.grads)
                Flux.update!(opt_intermediate, net.intermediate.flow, result.grads)
            else
                @warn "Skipping intermediate update due to non-finite gradients" maxlog=1
                continue
            end
            if callback !== nothing
                callback(:intermediate, result, epoch)
            end
            intermediate_memory = _next_memory(result)
            if should_save && (time() - last_save) >= save_interval
                last_save = _save_checkpoint(save_path, net, constructor_meta)
            end
        end

        control_memory = nothing
        _reset_iterator!(control_data)
        for datum in control_data
            args, kwargs = _unpack_training_datum(datum)
            kwargs = _with_memory(kwargs, control_memory)
            kwargs = _ensure_carryover_limit(kwargs, carryover_limit)
            result = control_grad(args, kwargs)
            if _grads_finite(result.grads)
                Flux.update!(opt_control, net.controller.flow, result.grads)
            else
                @warn "Skipping control update due to non-finite gradients" maxlog=1
                continue
            end
            if callback !== nothing
                callback(:control, result, epoch)
            end
            control_memory = _next_memory(result)
            if should_save && (time() - last_save) >= save_interval
                last_save = _save_checkpoint(save_path, net, constructor_meta)
            end
        end
    end

    if should_save && _tree_finite(Flux.state(net))
        _save_checkpoint(save_path, net, constructor_meta)
    end

    return net
end

"""
    RecurrentControlNet(terminal_data, intermediate_data, control_data, rule;
                        epochs=1, callback=nothing, carryover_limit=0, kwargs...)

Construct a [`RecurrentControlNet`](@ref) whose dimensions are inferred from the
provided datasets, then train it using [`train_recurrent_control!`](@ref).
The iterables must yield `(args, kwargs)` pairs just like those consumed by the
gradient helpers. Optional `save_path`, `save_interval`, and `carryover_limit`
arguments behave like in [`train_recurrent_control!`](@ref).
"""
function RecurrentControlNet(terminal_data_iter,
                             intermediate_data_iter,
                             control_data_iter,
                             rule;
                             epochs::Integer=1,
                             callback=nothing,
                             save_path::AbstractString="",
                             load_path::AbstractString=save_path,
                             save_interval::Real=60.0,
                             carryover_limit::Integer=0,
                             kwargs...)
    terminal_data, term_first = _stateful_with_first(terminal_data_iter)
    intermediate_data, inter_first = _stateful_with_first(intermediate_data_iter)
    control_data, ctrl_first = _stateful_with_first(control_data_iter)

    term_args, _ = _unpack_training_datum(term_first)
    inter_args, _ = _unpack_training_datum(inter_first)
    ctrl_args, _ = _unpack_training_datum(ctrl_first)

    term_args = _flatten_terminal_args(term_args)
    inter_args = _flatten_intermediate_args(inter_args)
    ctrl_args = _flatten_control_args(ctrl_args)

    state_dim = size(term_args[1], 1)
    goal_dim = size(term_args[3], 1)
    control_dim = size(ctrl_args[1], 1)

    size(term_args[2], 1) == state_dim ||
        throw(ArgumentError("current state dimension in terminal data mismatch"))
    size(inter_args[1], 1) == state_dim ||
        throw(ArgumentError("intermediate sample dimension mismatch"))
    size(inter_args[2], 1) == state_dim ||
        throw(ArgumentError("current state dimension in intermediate data mismatch"))
    size(inter_args[3], 1) == state_dim ||
        throw(ArgumentError("terminal state dimension mismatch in intermediate data"))
    size(ctrl_args[2], 1) == state_dim ||
        throw(ArgumentError("current state dimension in control data mismatch"))
    size(ctrl_args[3], 1) == state_dim ||
        throw(ArgumentError("next state dimension in control data mismatch"))

    default_kwargs = (
        terminal_kwargs=(n_blocks=1,),
        intermediate_kwargs=(n_blocks=1,),
        control_kwargs=(n_blocks=1,),
    )
    user_kwargs = NamedTuple(kwargs)
    stored_kwargs_nt = NamedTuple()
    if load_path != "" && isfile(load_path)
        stored = load(load_path)
        stored_constructor = get(stored, "constructor", nothing)
        if stored_constructor !== nothing
            stored_args = get(stored_constructor, "args", nothing)
            if stored_args !== nothing
                @assert stored_args == (state_dim, goal_dim, control_dim) "Constructor args in checkpoint do not match data-provided dimensions"
            end
            stored_kwargs = get(stored_constructor, "kwargs", nothing)
        if stored_kwargs !== nothing
            stored_kwargs_nt = stored_kwargs isa NamedTuple ? stored_kwargs : (; stored_kwargs...)
        end
        end
    end
    merged_kwargs = merge(default_kwargs, stored_kwargs_nt, user_kwargs)

    constructor_info = Dict(
        "args" => (state_dim, goal_dim, control_dim),
        "kwargs" => merged_kwargs
    )

    net = RecurrentControlNet(state_dim, goal_dim, control_dim; merged_kwargs...)
    train_recurrent_control!(net,
                             terminal_data,
                             intermediate_data,
                             control_data,
                             rule;
                             epochs=epochs,
                             callback=callback,
                             save_path=save_path,
                             load_path=load_path,
                             save_interval=save_interval,
                             constructor_info=constructor_info,
                             carryover_limit=carryover_limit)
    return net
end

function _unpack_training_datum(datum)
    if datum isa Tuple && length(datum) == 2 && datum[2] isa NamedTuple
        args_part, kwargs_part = datum
        args_tuple = args_part isa Tuple ? args_part : (args_part,)
        return args_tuple, kwargs_part
    elseif datum isa Tuple
        return datum, NamedTuple()
    elseif datum isa TerminalGradientDatum
        return (datum.samples, datum.current_states, datum.goals), NamedTuple()
    elseif datum isa IntermediateGradientDatum
        return (datum.samples, datum.current_states, datum.terminal_states, datum.times, datum.total_times), NamedTuple()
    elseif datum isa ControlGradientDatum
        return (datum.samples, datum.current_states, datum.next_states), NamedTuple()
    else
        throw(ArgumentError("training datum must be a tuple, (args, kwargs) pair, or gradient datum struct"))
    end
end

function _flatten_terminal_args(args)
    _flatten_args(args, TerminalGradientDatum, d -> (d.samples, d.current_states, d.goals))
end

function _flatten_intermediate_args(args)
    _flatten_args(args, IntermediateGradientDatum, d -> (d.samples, d.current_states, d.terminal_states, d.times, d.total_times))
end

function _flatten_control_args(args)
    _flatten_args(args, ControlGradientDatum, d -> (d.samples, d.current_states, d.next_states))
end

function _flatten_args(args, ::Type{T}, f::Function) where T
    if length(args) == 1 && args[1] isa T
        return f(args[1])
    else
        return args
    end
end

function _stateful_with_first(iterable)
    stateful = iterable isa Stateful ? iterable : Stateful(iterable)
    first_iter = iterate(stateful)
    first_iter === nothing && throw(ArgumentError("training data iterator must produce at least one datum"))
    first_datum, _ = first_iter
    reset!(stateful)
    return stateful, first_datum
end

_reset_iterator!(_) = nothing
_reset_iterator!(data::Stateful) = reset!(data)

function _with_memory(kwargs::NamedTuple, memory)
    memory === nothing && return kwargs
    extra_samples = _materialize(memory.samples)
    extra_context = _materialize(memory.context)
    merged_samples = _merge_memory(get(kwargs, :old_samples, nothing), extra_samples)
    merged_context = _merge_memory(get(kwargs, :old_context, nothing), extra_context)
    return merge(kwargs, (old_samples=merged_samples, old_context=merged_context))
end

function _ensure_carryover_limit(kwargs::NamedTuple, carryover_limit::Integer)
    haskey(kwargs, :num_lowest) ? kwargs : merge(kwargs, (num_lowest=carryover_limit,))
end

function _merge_memory(existing, extra)
    if existing === nothing
        return extra
    elseif extra === nothing
        return existing
    else
        return hcat(existing, extra)
    end
end

_materialize(x) = x === nothing ? nothing : (x isa SubArray ? copy(x) : x)

function _next_memory(result)
    hard = get(result, :hard_examples, nothing)
    if hard === nothing
        return nothing
    else
        return (samples=_materialize(hard.samples),
                context=_materialize(hard.context))
    end
end

_grads_finite(x::Number) = isfinite(x)
_grads_finite(x::AbstractArray) = all(_grads_finite, x)
_grads_finite(x::NamedTuple) = all(_grads_finite, values(x))
_grads_finite(x::Tuple) = all(_grads_finite, x)
_grads_finite(::Nothing) = true
_grads_finite(_) = true

_tree_finite(x::Number) = isfinite(x)
_tree_finite(x::AbstractArray) = all(_tree_finite, x)
_tree_finite(x::NamedTuple) = all(_tree_finite, values(x))
_tree_finite(x::Tuple) = all(_tree_finite, x)
_tree_finite(::Nothing) = true
_tree_finite(_) = true

function _save_checkpoint(path::AbstractString,
                          net::RecurrentControlNet,
                          constructor_info)
    ts = time()
    state = Flux.state(net)
    if constructor_info === nothing
        jldsave(path; model_state=state, timestamp=ts)
    else
        jldsave(path; model_state=state, timestamp=ts, constructor=constructor_info)
    end
    return ts
end

using Flux
using JLD2: jldsave, load
import Base.Iterators: Stateful, reset!

using ..RecurrentControl: TerminalGradientDatum, IntermediateGradientDatum, ControlGradientDatum
using ..GANModels: gan_gradients, gradient_norm

_tree_finite(x::Number) = isfinite(x)
_tree_finite(x::AbstractArray) = all(_tree_finite, x)
_tree_finite(x::NamedTuple) = all(_tree_finite, values(x))
_tree_finite(x::Tuple) = all(_tree_finite, x)
_tree_finite(::Nothing) = true
_tree_finite(_) = true

function _save_checkpoint(path::AbstractString,
                          net::GANControlNet,
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

_stateful_with_first(iterable) = let stateful = iterable isa Stateful ? iterable : Stateful(iterable)
    first_iter = iterate(stateful)
    first_iter === nothing && throw(ArgumentError("training data iterator must produce at least one datum"))
    first_datum, _ = first_iter
    reset!(stateful)
    stateful, first_datum
end

_reset_iterator!(_) = nothing
_reset_iterator!(data::Stateful) = reset!(data)

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
    if length(args) == 1 && args[1] isa TerminalGradientDatum
        d = args[1]
        return d.samples, d.current_states, d.goals
    else
        return args
    end
end

function _flatten_intermediate_args(args)
    if length(args) == 1 && args[1] isa IntermediateGradientDatum
        d = args[1]
        return d.samples, d.current_states, d.terminal_states, d.times, d.total_times
    else
        return args
    end
end

function _flatten_control_args(args)
    if length(args) == 1 && args[1] isa ControlGradientDatum
        d = args[1]
        return d.samples, d.current_states, d.next_states
    else
        return args
    end
end

function _as_colmat(x)
    mat = ndims(x) == 1 ? reshape(x, :, 1) : x
    return Float32.(mat)
end

function _terminal_batch(net::GANControlNet, args, kwargs)
    samples, current, goals = _flatten_terminal_args(args)
    samples = _as_colmat(samples)
    current = _as_colmat(current)
    goals = _as_colmat(goals)
    @assert size(samples, 1) == net.state_dim
    @assert size(current, 1) == net.state_dim
    @assert size(goals, 1) == net.goal_dim
    batch = size(samples, 2)
    @assert size(current, 2) == batch
    @assert size(goals, 2) == batch
    context = vcat(current, goals)
    return (contexts=context, samples=samples)
end

function _intermediate_batch(net::GANControlNet, args, kwargs)
    samples, current, terminal, times, totals = _flatten_intermediate_args(args)
    samples = _as_colmat(samples)
    current = _as_colmat(current)
    terminal = _as_colmat(terminal)
    batch = size(samples, 2)
    @assert size(current, 1) == net.state_dim "current state dim mismatch"
    @assert size(terminal, 1) == net.state_dim
    @assert size(current, 2) == batch
    @assert size(terminal, 2) == batch
    steps_vec = times isa Integer ? fill(Int(times), batch) : Int.(collect(times))
    totals_vec = totals isa Integer ? fill(Int(totals), batch) : Int.(collect(totals))
    @assert length(steps_vec) == batch
    @assert length(totals_vec) == batch
    if net.time_feature_dim == 0
        time_mat = zeros(Float32, 0, batch)
    else
        time_mat = zeros(Float32, net.time_feature_dim, batch)
        for (j, (step, total)) in enumerate(zip(steps_vec, totals_vec))
            time_mat[:, j] = _time_features(net.time_feature_dim, step, total, 1, Float32)[:, 1]
        end
    end
    context = vcat(current, terminal, time_mat)
    rates = similar(samples)
    for j in 1:batch
        step = steps_vec[j]
        if step <= 0
            rates[:, j] .= 0
        else
            rates[:, j] .= (samples[:, j] .- current[:, j]) ./ Float32(step)
        end
    end
    return (contexts=context, samples=rates)
end

function _control_batch(net::GANControlNet, args, kwargs)
    samples, current, next_states = _flatten_control_args(args)
    samples = _as_colmat(samples)
    current = _as_colmat(current)
    next_states = _as_colmat(next_states)
    batch = size(samples, 2)
    @assert size(samples, 1) == net.control_dim
    @assert size(current, 1) == net.state_dim
    @assert size(next_states, 1) == net.state_dim
    @assert size(current, 2) == batch
    @assert size(next_states, 2) == batch
    context = vcat(current, next_states)
    return (contexts=context, samples=samples)
end

function _select_carryover(hard_examples, limit::Integer)
    if hard_examples === nothing || limit <= 0
        return nothing
    end
    count = size(hard_examples.contexts, 2)
    if count == 0
        return nothing
    end
    k = min(limit, count)
    return (contexts=hard_examples.contexts[:, 1:k],
            samples=hard_examples.samples[:, 1:k])
end

function _apply_gan_updates!(gan::Gan, grads, opts)
    Flux.update!(opts.generator, gan.generator, grads.generator_grad)
    Flux.update!(opts.discriminator, gan.discriminator, grads.discriminator_grad)
    Flux.update!(opts.encoder, gan.encoder, grads.encoder_grad)
end

function _gan_step!(gan::Gan, opts, batch, carryover, carryover_limit::Integer;
                    loss_fn=Flux.Losses.binarycrossentropy,
                    latent_sampler=dim -> (2f0 .* rand(Float32, dim) .- 1f0))
    grads = gan_gradients(gan, batch; old_batch=carryover, sorted_limit=carryover_limit,
                          loss_fn=loss_fn, latent_sampler=latent_sampler)
    _apply_gan_updates!(gan, grads, opts)
    new_carry = _select_carryover(grads.hard_examples, carryover_limit)
    stats = (
        generator_grad_norm = gradient_norm(grads.generator_grad),
        discriminator_grad_norm = gradient_norm(grads.discriminator_grad),
        encoder_grad_norm = gradient_norm(grads.encoder_grad),
        hard_examples = grads.hard_examples
    )
    return stats, new_carry
end

function train_gan_control!(net::GANControlNet,
                            terminal_data,
                            intermediate_data,
                            control_data,
                            rule;
                            epochs::Integer=1,
                            callback::Union{Nothing,Function}=nothing,
                            carryover_limit::Integer=0,
                            save_path::AbstractString="",
                            load_path::AbstractString=save_path,
                            save_interval::Real=60.0,
                            constructor_info=nothing,
                            loss_fn=Flux.Losses.binarycrossentropy,
                            latent_sampler=dim -> (2f0 .* rand(Float32, dim) .- 1f0))
    epochs > 0 || throw(ArgumentError("epochs must be positive"))
    carryover_limit >= 0 || throw(ArgumentError("carryover_limit must be non-negative"))
    save_interval >= 0 || throw(ArgumentError("save_interval must be non-negative"))

    term_iter, term_first = _stateful_with_first(terminal_data)
    inter_iter, inter_first = _stateful_with_first(intermediate_data)
    ctrl_iter, ctrl_first = _stateful_with_first(control_data)

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
    if should_save
        mkpath(dirname(save_path))
    end
    last_save = time()

    terminal_opts = (
        generator = Flux.setup(rule, net.terminal.generator),
        discriminator = Flux.setup(rule, net.terminal.discriminator),
        encoder = Flux.setup(rule, net.terminal.encoder)
    )
    intermediate_opts = (
        generator = Flux.setup(rule, net.intermediate.generator),
        discriminator = Flux.setup(rule, net.intermediate.discriminator),
        encoder = Flux.setup(rule, net.intermediate.encoder)
    )
    control_opts = (
        generator = Flux.setup(rule, net.controller.generator),
        discriminator = Flux.setup(rule, net.controller.discriminator),
        encoder = Flux.setup(rule, net.controller.encoder)
    )

    constructor_state = constructor_meta

    terminal_carry = nothing
    intermediate_carry = nothing
    control_carry = nothing

    for epoch in 1:epochs
        _reset_iterator!(term_iter)
        for datum in term_iter
            args, kwargs = _unpack_training_datum(datum)
            batch = _terminal_batch(net, args, kwargs)
            stats, terminal_carry = _gan_step!(net.terminal, terminal_opts, batch, terminal_carry, carryover_limit;
                                              loss_fn=loss_fn, latent_sampler=latent_sampler)
            if callback !== nothing
                callback(:terminal, stats, epoch)
            end
            if should_save && (time() - last_save) >= save_interval
                last_save = _save_checkpoint(save_path, net, constructor_state)
            end
        end

        _reset_iterator!(inter_iter)
        for datum in inter_iter
            args, kwargs = _unpack_training_datum(datum)
            batch = _intermediate_batch(net, args, kwargs)
            stats, intermediate_carry = _gan_step!(net.intermediate, intermediate_opts, batch, intermediate_carry, carryover_limit;
                                                  loss_fn=loss_fn, latent_sampler=latent_sampler)
            if callback !== nothing
                callback(:intermediate, stats, epoch)
            end
            if should_save && (time() - last_save) >= save_interval
                last_save = _save_checkpoint(save_path, net, constructor_state)
            end
        end

        _reset_iterator!(ctrl_iter)
        for datum in ctrl_iter
            args, kwargs = _unpack_training_datum(datum)
            batch = _control_batch(net, args, kwargs)
            stats, control_carry = _gan_step!(net.controller, control_opts, batch, control_carry, carryover_limit;
                                              loss_fn=loss_fn, latent_sampler=latent_sampler)
            if callback !== nothing
                callback(:control, stats, epoch)
            end
            if should_save && (time() - last_save) >= save_interval
                last_save = _save_checkpoint(save_path, net, constructor_state)
            end
        end
    end

    if should_save
        _save_checkpoint(save_path, net, constructor_state)
    end

    return net
end

function _constructor_entry(constructor, key::AbstractString)
    if constructor isa AbstractDict
        return get(constructor, key, nothing)
    elseif constructor isa NamedTuple
        sym = Symbol(key)
        return hasproperty(constructor, sym) ? getproperty(constructor, sym) : nothing
    else
        return nothing
    end
end

function _constructor_kwargs(kwargs_data)
    if kwargs_data === nothing
        return NamedTuple()
    elseif kwargs_data isa NamedTuple
        return kwargs_data
    elseif kwargs_data isa AbstractDict
        keys_vec = collect(keys(kwargs_data))
        vals_vec = collect(values(kwargs_data))
        syms = Symbol.(keys_vec)
        return NamedTuple{Tuple(syms...)}(Tuple(vals_vec))
    else
        return NamedTuple(kwargs_data)
    end
end

function load_gan_control(load_path::AbstractString)
    isfile(load_path) || throw(ArgumentError("checkpoint not found at $load_path"))
    stored = load(load_path)
    constructor = get(stored, "constructor", nothing)
    constructor === nothing &&
        throw(ArgumentError("checkpoint $load_path does not contain constructor metadata"))

    args = _constructor_entry(constructor, "args")
    args === nothing &&
        throw(ArgumentError("checkpoint $load_path is missing constructor args"))
    args isa Tuple ||
        throw(ArgumentError("constructor args in $load_path must be stored as a tuple"))
    length(args) == 3 ||
        throw(ArgumentError("expected (state_dim, goal_dim, control_dim) in constructor args"))

    kwargs_data = _constructor_entry(constructor, "kwargs")
    kwargs_nt = _constructor_kwargs(kwargs_data)

    net = GANControlNet(args...; kwargs_nt...)

    state = get(stored, "model_state", nothing)
    if state === nothing
        @warn "Checkpoint does not contain model_state; returning freshly initialised network" load_path maxlog=1
    elseif _tree_finite(state)
        Flux.loadmodel!(net, state)
    else
        @warn "Skipping weight load due to non-finite values" load_path maxlog=1
    end

    return net
end

module TrajectoryRefinerTraining

using Flux
using JLD2
using ..TrajectoryRefiner: RefinementModel, ShootingBundle, refinement_loss, refinement_grads

"""
    train!(model, data_iter, refine_steps::Int, backprop_steps::Int,
           transition_fn, traj_cost_fn, traj_mismatch_fn;
           opt=Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam()),
           imitation_weight=1.0,
           backprop_mode::Symbol=:tail,
           softmax_temperature::Real=1.0,
           save_path=nothing, save_period::Real=60.0,
           load_path=save_path, construction_args::Union{Nothing,NamedTuple}=nothing)

Train a trajectory refiner (`RefinementModel`) using unrolled refinement steps and truncated
backpropagation. `data_iter` must yield `ShootingBundle` objects carrying `x0` (initial state),
`x_guess` (body), `u_guess`, and optional `x_target` (imitation trajectory over post-initial states).

# Arguments
- `model`: refinement model to train.
- `data_iter`: collection/iterator of training samples.
- `refine_steps::Int`: number of refinement iterations unrolled per update.
- `backprop_steps::Int`: number of trailing refinement steps to keep on the gradient tape (must satisfy `backprop_steps <= refine_steps`).
- `transition_fn`, `traj_cost_fn`, `traj_mismatch_fn`: callbacks forwarded to `refinement_loss`.
- `opt`: Flux optimiser (default `Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam())`).
- `imitation_weight`: weight on the imitation factor when a `target` trajectory is provided.
- `backprop_mode`: `:tail` (default) uses the last `backprop_steps` refinements; `:min_loss`
  picks the refinement step with minimal loss and backpropagates for `backprop_steps` steps from there.
- `softmax_temperature`: temperature used when scalarizing array-valued `traj_cost_fn`/`mismatch_fn`
  outputs via softmax-weighted averaging (default 1.0).
- `save_path`: optional checkpoint path; if provided, `construction_args` must supply model dims for reload.
- `save_period`: seconds between checkpoints when `save_path` is set (default 60).
- `load_path`: optional path to resume from (defaults to `save_path` when provided).
- `construction_args`: named tuple with `state_dim`, `input_dim`, `cost_dim`, `hidden_dim`, `depth`,
  and `activation`/`max_seq_len` used for reconstruction during save/load.

# Returns
`(model, metrics)` with the trained model and a vector of per-step loss components
(`loss`, `traj_cost`, `mismatch`, `imitation`).
"""
function train!(model::RefinementModel, data_iter, refine_steps::Int, backprop_steps::Int,
                transition_fn, traj_cost_fn, traj_mismatch_fn;
                opt=Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam()),
                imitation_weight::Real=1.0,
                save_path=nothing, save_period::Real=60.0,
                load_path=save_path, construction_args::Union{Nothing,NamedTuple}=nothing,
                backprop_mode::Symbol=:tail,
                softmax_temperature::Real=1.0)
    backprop_steps <= refine_steps || throw(ArgumentError("backprop_steps must be ≤ refine_steps"))

    # Optionally resume from checkpoint
    if load_path === nothing && save_path !== nothing
        load_path = save_path
    end
    if load_path !== nothing && isfile(load_path)
        @warn "Resuming training from checkpoint at $load_path"
        model = load_refinement_model(load_path; activation=construction_args === nothing ? nothing : get(construction_args, :activation, nothing))
    end

    opt_state = Flux.setup(opt, model)
    metrics_log = NamedTuple{(:loss, :traj_cost, :mismatch, :imitation, :best_step), NTuple{5, Float32}}[]
    last_save = time()
    worst_bundle = nothing
    worst_loss = -Inf

    for sample in data_iter
        sample isa ShootingBundle || throw(ArgumentError("data_iter must yield ShootingBundle objects"))
        tbatch = sample

        # Gradient/update on current sample; reuse its loss for worst-bundle tracking.
        grads_curr, metrics_curr = refinement_grads(model, transition_fn, traj_cost_fn, traj_mismatch_fn,
                                                   tbatch, refine_steps, backprop_steps;
                                                   imitation_weight=imitation_weight,
                                                   backprop_mode=backprop_mode,
                                                   softmax_temperature=softmax_temperature)
        curr_loss = metrics_curr.loss
        Flux.update!(opt_state, model, grads_curr)

        # Initialize or update the stored worst bundle using cached losses.
        if worst_bundle === nothing
            worst_bundle = deepcopy(tbatch)
            worst_loss = curr_loss
        else
            # Re-evaluate worst bundle loss on the updated model only if needed.
            worst_metrics = refinement_loss(model, transition_fn, traj_cost_fn, traj_mismatch_fn, worst_bundle, refine_steps;
                                            imitation_weight=imitation_weight,
                                            softmax_temperature=softmax_temperature)
            worst_eval = worst_metrics.loss
            if curr_loss > worst_eval
                worst_bundle = deepcopy(tbatch)
                worst_loss = curr_loss
            else
                worst_loss = worst_eval
            end
        end

        # Gradient/update on worst-loss bundle (if tracked).
        if worst_bundle !== nothing
            grads_worst, _ = refinement_grads(model, transition_fn, traj_cost_fn, traj_mismatch_fn,
                                              worst_bundle, refine_steps, backprop_steps;
                                              imitation_weight=imitation_weight,
                                              backprop_mode=backprop_mode,
                                              softmax_temperature=softmax_temperature)
            Flux.update!(opt_state, model, grads_worst)
        end

        push!(metrics_log, (Float32(metrics_curr.loss),
                            Float32(metrics_curr.traj_cost),
                            Float32(metrics_curr.mismatch),
                            Float32(metrics_curr.imitation),
                            Float32(get(metrics_curr, :best_step, 0))))

        if save_path !== nothing && (time() - last_save) >= save_period
            construction_args === nothing && throw(ArgumentError("construction_args must be provided to save checkpoints"))
            save_refinement_model(save_path, model; construction_args...)
            last_save = time()
        end
    end

    if save_path !== nothing
        construction_args === nothing && throw(ArgumentError("construction_args must be provided to save checkpoints"))
        save_refinement_model(save_path, model; construction_args...)
    end

    return model, metrics_log
end

"""
    save_refinement_model(path::AbstractString, model::RefinementModel;
                          state_dim, input_dim, cost_dim, hidden_dim, depth, activation=relu)

Save a refinement model checkpoint along with its construction arguments using `Flux.state`.
"""
function save_refinement_model(path::AbstractString, model::RefinementModel;
                               state_dim::Int, input_dim::Int, cost_dim::Int, hidden_dim::Int, depth::Int,
                               activation=relu, max_seq_len::Int=512, latent_dim::Union{Nothing,Integer}=state_dim,
                               attention_heads::Int=2)
    state = Flux.state(model)
    args = (state_dim, input_dim, cost_dim, hidden_dim, depth, latent_dim, attention_heads)
    kwargs = (; activation=activation, max_seq_len=max_seq_len)
    JLD2.jldsave(path; state, args, kwargs)
    return path
end

"""
    load_refinement_model(path::AbstractString; activation=nothing)

Load a refinement model checkpoint produced by [`save_refinement_model`](@ref). If `activation` is
`nothing`, the stored activation is used; otherwise the provided activation overrides it.
"""
function load_refinement_model(path::AbstractString; activation=nothing)
    data = JLD2.load(path)
    args = data["args"]
    stored_kwargs = get(data, "kwargs", (;))
    state = data["state"]
    state_dim, input_dim, cost_dim, hidden_dim, depth, latent_dim, attention_heads = args
    act = activation === nothing ? get(stored_kwargs, :activation, Flux.σ) : activation
    max_seq_len = get(stored_kwargs, :max_seq_len, 512)
    model = RefinementModel(state_dim, input_dim, cost_dim, hidden_dim, depth;
                            activation=act, max_seq_len=max_seq_len, latent_dim=latent_dim,
                            attention_heads=attention_heads)
    Flux.loadmodel!(model, state)
    return model
end

"""
    build(::Type{RefinementModel}, data_iter, args...;
          hidden_dim::Integer=64, depth::Integer=2, activation=relu, kwargs...)

Construct and train a `RefinementModel`, inferring state/input/cost dimensions from the first element of
`data_iter`. Network construction is controlled by `hidden_dim`, `depth`, and `activation`; all
positional `args...`/`kwargs...` are forwarded to [`train!`](@ref) (e.g., refinement steps,
transition/mismatch/cost functions, optimiser, checkpointing). Returns a named tuple
`(model=trained, metrics=metrics_log, init_model=untrained_copy)`.
"""
function build(::Type{RefinementModel}, data_iter, args...;
               hidden_dim::Integer=64, depth::Integer=2, activation=Flux.σ, max_seq_len::Int=512,
               latent_dim::Union{Nothing,Integer}=nothing, attention_heads::Integer=1, kwargs...)
    first_sample = first(data_iter)
    first_sample isa ShootingBundle || throw(ArgumentError("data_iter must yield ShootingBundle objects"))
    length(args) >= 4 || throw(ArgumentError("build(::Type{RefinementModel}, data_iter, refine_steps, backprop_steps, transition_fn, traj_cost_fn, ...) requires at least four positional training arguments"))
    state_dim = size(first_sample.x_guess, 1)
    input_dim = size(first_sample.u_guess, 1)
    transition_fn = args[3]
    traj_cost_fn = args[4]
    cost_raw = traj_cost_fn(cat(first_sample.x0, first_sample.x_guess; dims=2))
    if cost_raw isa Number
        cost_dim = 1
    elseif ndims(cost_raw) == 3
        cost_dim = size(cost_raw, 1)
    else
        cost_dim = size(cost_raw, 1)
    end
    hidden_dim_val = Int(hidden_dim)
    depth_val = Int(depth)
    latent_dim_val = latent_dim === nothing ? state_dim : Int(latent_dim)
    att_heads_val = Int(attention_heads)

    model = RefinementModel(state_dim, input_dim, cost_dim, hidden_dim_val, depth_val;
                            activation=activation, max_seq_len=max_seq_len,
                            latent_dim=latent_dim_val, attention_heads=att_heads_val)
    init_model = deepcopy(model)
    construction_args = (state_dim=state_dim, input_dim=input_dim, cost_dim=cost_dim, hidden_dim=hidden_dim_val,
                         depth=depth_val, activation=activation, max_seq_len=max_seq_len,
                         latent_dim=latent_dim_val, attention_heads=att_heads_val)

    model, metrics = train!(model, data_iter, args...;
                            construction_args=construction_args, kwargs...)
    return (; model, metrics, init_model)
end

end # module

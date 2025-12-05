module TrajectoryRefinerTraining

using Flux
using JLD2
import ..build
using ..TrajectoryRefiner: RefinementModel, ShootingBundle, refinement_loss, refinement_grads

"""
    train!(model, data_iter, refine_steps::Int, backprop_steps::Int,
           transition_fn, traj_cost_fn, traj_mismatch_fn;
           opt=Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam()),
           imitation_weight=1.0)

Train a trajectory refiner (`RefinementModel`) using unrolled refinement steps and truncated
backpropagation. `data_iter` must yield `ShootingBundle` objects carrying `x_guess` (including the
initial state), `u_guess`, and optional `x_target` (imitation trajectory over post-initial states).

# Arguments
- `model`: refinement model to train.
- `data_iter`: collection/iterator of training samples.
- `refine_steps::Int`: number of refinement iterations unrolled per update.
- `backprop_steps::Int`: number of trailing refinement steps to keep on the gradient tape (must satisfy `backprop_steps <= refine_steps`).
- `transition_fn`, `traj_cost_fn`, `traj_mismatch_fn`: callbacks forwarded to `refinement_loss`.
- `opt`: Flux optimiser (default `Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam())`).
- `imitation_weight`: weight on the imitation factor when a `target` trajectory is provided.
- `save_path`: optional checkpoint path; if provided, `construction_args` must supply model dims for reload.
- `save_period`: seconds between checkpoints when `save_path` is set (default 60).
- `load_path`: optional path to resume from (defaults to `save_path` when provided).
- `construction_args`: named tuple with `state_dim`, `input_dim`, `cost_dim`, `hidden_dim`, `depth`,
  and `activation`/`max_seq_len` used for reconstruction during save/load.

# Returns
`(model, losses)` with the trained model and a vector of per-step losses.
"""
function train!(model::RefinementModel, data_iter, refine_steps::Int, backprop_steps::Int,
                transition_fn, traj_cost_fn, traj_mismatch_fn;
                opt=Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam()),
                imitation_weight::Real=1.0,
                save_path=nothing, save_period::Real=60.0,
                load_path=nothing, construction_args::Union{Nothing,NamedTuple}=nothing)
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
    losses = Float32[]
    last_save = time()

    for sample in data_iter
        sample isa ShootingBundle || throw(ArgumentError("data_iter must yield ShootingBundle objects"))
        tbatch = sample

        total_steps = max(refine_steps, 0)
        bsteps = clamp(backprop_steps, 0, total_steps)
        fsteps = total_steps - bsteps

        fwd_sample = fsteps > 0 ? model(tbatch, transition_fn, traj_cost_fn, fsteps) : tbatch

        grads, metrics = refinement_grads(model, transition_fn, traj_cost_fn, traj_mismatch_fn,
                                          fwd_sample, bsteps;
                                          imitation_weight=imitation_weight)

        Flux.update!(opt_state, model, grads)
        push!(losses, Float32(metrics.loss))

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

    return model, losses
end

"""
    save_refinement_model(path::AbstractString, model::RefinementModel;
                          state_dim, input_dim, cost_dim, hidden_dim, depth, activation=relu)

Save a refinement model checkpoint along with its construction arguments using `Flux.state`.
"""
function save_refinement_model(path::AbstractString, model::RefinementModel;
                               state_dim::Int, input_dim::Int, cost_dim::Int, hidden_dim::Int, depth::Int,
                               activation=relu, max_seq_len::Int=512)
    state = Flux.state(model)
    args = (state_dim, input_dim, cost_dim, hidden_dim, depth)
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
    state_dim, input_dim, cost_dim, hidden_dim, depth = args
    act = activation === nothing ? get(stored_kwargs, :activation, Flux.σ) : activation
    max_seq_len = get(stored_kwargs, :max_seq_len, 512)
    model = RefinementModel(state_dim, input_dim, cost_dim, hidden_dim, depth; activation=act, max_seq_len=max_seq_len)
    Flux.loadmodel!(model, state)
    return model
end

"""
    build(::Type{RefinementModel}, data_iter, args...;
          hidden_dim::Integer=64, depth::Integer=2, activation=relu, kwargs...)

Construct and train a `RefinementModel`, inferring state/input/cost dimensions from the first element of
`data_iter`. Network construction is controlled by `hidden_dim`, `depth`, and `activation`; all
positional `args...`/`kwargs...` are forwarded to [`train!`](@ref) (e.g., refinement steps,
transition/mismatch/cost functions, optimiser, checkpointing). Returns `(model, losses)`.
"""
function build(::Type{RefinementModel}, data_iter, args...;
               hidden_dim::Integer=64, depth::Integer=2, activation=Flux.σ, max_seq_len::Int=512, kwargs...)
    first_sample = first(data_iter)
    first_sample isa ShootingBundle || throw(ArgumentError("data_iter must yield ShootingBundle objects"))
    length(args) >= 4 || throw(ArgumentError("build(::Type{RefinementModel}, data_iter, refine_steps, backprop_steps, transition_fn, traj_cost_fn, ...) requires at least four positional training arguments"))
    state_dim = size(first_sample.x_guess, 1)
    input_dim = size(first_sample.u_guess, 1)
    transition_fn = args[3]
    traj_cost_fn = args[4]
    cost_raw = traj_cost_fn(first_sample.x_guess)
    if cost_raw isa Number
        cost_dim = 1
    elseif ndims(cost_raw) == 3
        cost_dim = size(cost_raw, 1)
    else
        cost_dim = size(cost_raw, 1)
    end
    hidden_dim_val = Int(hidden_dim)
    depth_val = Int(depth)

    model = RefinementModel(state_dim, input_dim, cost_dim, hidden_dim_val, depth_val; activation=activation, max_seq_len=max_seq_len)
    construction_args = (state_dim=state_dim, input_dim=input_dim, cost_dim=cost_dim, hidden_dim=hidden_dim_val,
                         depth=depth_val, activation=activation, max_seq_len=max_seq_len)

    model, losses = train!(model, data_iter, args...;
                           construction_args=construction_args, kwargs...)
    return model, losses
end

end # module

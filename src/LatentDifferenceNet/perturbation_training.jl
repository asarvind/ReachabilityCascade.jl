import Flux
using Random

"""
    train_perturbation!(model::RefinementRNN,
                        data::AbstractVector,
                        sys::DiscreteRandomSystem,
                        traj_cost_fn;
                        epochs::Integer=1,
                        steps::Integer=1,
                        δ_max::Real=1f-3,
                        δ_min::Real=1f-5,
                        temperature::Real=1,
                        eval_samples::Integer=8,
                        rng::Random.AbstractRNG=Random.default_rng(),
                        shuffle::Bool=true,
                        start_idx_range=nothing,
                        save_path=nothing,
                        load_path=save_path,
                        save_period::Real=60.0)

Train a `RefinementRNN` using SPSA-style perturbation accept/reject updates (no gradients).

For each epoch:
1. Iterate over the dataset in batches of size `eval_samples` (after optional shuffling).
2. For each batch, propose SPSA perturbations `θ ± δ*s` (random sign vector `s`) and compare average loss over
   that batch against the unperturbed model.
3. If either perturbation improves the average loss, accept the better one and increase `δ ← clamp(2δ, δ_min, δ_max)`.
   Otherwise reject and decrease `δ ← clamp(δ/2, δ_min, δ_max)`.

Returns a named tuple with fields:
- `model`: trained model
- `model_before`: snapshot before training
- `accept_flags`, `base_losses`, `pert_losses`: per-batch logs
"""
function train_perturbation!(model::RefinementRNN,
                             data::AbstractVector,
                             sys::DiscreteRandomSystem,
                             traj_cost_fn;
                             epochs::Integer=1,
                             steps::Integer=1,
                             δ_max::Real=1f-3,
                             δ_min::Real=1f-5,
                             step_mode::Symbol=:terminal,
                             dual::Bool=false,
                             temperature::Real=1,
                             eval_samples::Integer=8,
                             rng::Random.AbstractRNG=Random.default_rng(),
                             shuffle::Bool=true,
                             start_idx_range=nothing,
                             save_path::Union{Nothing,AbstractString}=nothing,
                             load_path::Union{Nothing,AbstractString}=save_path,
                             save_period::Real=60.0)
    epochs >= 1 || throw(ArgumentError("epochs must be ≥ 1"))
    steps >= 1 || throw(ArgumentError("steps must be ≥ 1"))
    δ_max > 0 || throw(ArgumentError("δ_max must be positive"))
    δ_min > 0 || throw(ArgumentError("δ_min must be positive"))
    δ_min <= δ_max || throw(ArgumentError("δ_min must be ≤ δ_max"))
    (step_mode === :terminal || step_mode === :best) ||
        throw(ArgumentError("step_mode must be :terminal or :best; got $step_mode"))
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    eval_samples >= 1 || throw(ArgumentError("eval_samples must be ≥ 1"))
    isempty(data) && throw(ArgumentError("data must be non-empty"))

    save_path_final = (save_path isa AbstractString && isempty(save_path)) ? nothing : save_path
    load_path_final = (load_path isa AbstractString && isempty(load_path)) ? nothing : load_path

    if load_path_final !== nothing && isfile(load_path_final)
        @warn "Starting perturbation training from saved checkpoint" load_path=load_path_final
        loaded = load(RefinementRNN, load_path_final)
        Flux.loadmodel!(model, Flux.state(loaded))
    end

    θ0, re = Flux.destructure(model)
    model_before = re(copy(θ0))

    n = length(data)
    accept_flags = Bool[]
    accept_choices = Int[]
    base_losses = Float32[]
    pert_losses = Float32[]
    pos_losses = Float32[]
    neg_losses = Float32[]
    deltas = Float32[]
    last_save = time()

    δ_curr = Float32(δ_max)

    for _ in 1:epochs
        order = shuffle ? randperm(rng, n) : collect(1:n)
        batch_size = min(Int(eval_samples), n)
        for batch_start in 1:batch_size:n
            batch_end = min(batch_start + batch_size - 1, n)
            idxs = view(order, batch_start:batch_end)
            batch_samples = Vector{NamedTuple{(:x0,), Tuple{Vector}}}(undef, length(idxs))
            for (j, idx) in enumerate(idxs)
                x0 = _sample_x0(rng, data[idx], start_idx_range)
                batch_samples[j] = (; x0=Vector(x0))
            end

            res = spsa_update!(model, batch_samples, sys, traj_cost_fn;
                               steps=steps,
                               δ=δ_curr,
                               step_mode=step_mode,
                               dual=dual,
                               temperature=temperature,
                               rng=rng)
            push!(accept_flags, res.accepted)
            push!(accept_choices, res.choice)
            push!(base_losses, res.base_loss)
            push!(pert_losses, res.pert_loss)
            push!(pos_losses, res.pos_loss)
            push!(neg_losses, res.neg_loss)
            push!(deltas, δ_curr)

            if res.accepted
                δ_curr = clamp(2f0 * δ_curr, Float32(δ_min), Float32(δ_max))
            else
                δ_curr = clamp(0.5f0 * δ_curr, Float32(δ_min), Float32(δ_max))
            end

            if save_path_final !== nothing && (time() - last_save) >= save_period
                save(model, save_path_final;
                     accept_flags=accept_flags,
                     accept_choices=accept_choices,
                     base_losses=base_losses,
                     pert_losses=pert_losses,
                     pos_losses=pos_losses,
                     neg_losses=neg_losses,
                     deltas=deltas)
                last_save = time()
            end
        end
    end

    if save_path_final !== nothing
        save(model, save_path_final;
             accept_flags=accept_flags,
             accept_choices=accept_choices,
             base_losses=base_losses,
             pert_losses=pert_losses,
             pos_losses=pos_losses,
             neg_losses=neg_losses,
             deltas=deltas)
    end

    return (; model=model,
            model_before=model_before,
            accept_flags=accept_flags,
            accept_choices=accept_choices,
            base_losses=base_losses,
            pert_losses=pert_losses,
            pos_losses=pos_losses,
            neg_losses=neg_losses,
            deltas=deltas)
end

"""
    build_perturbation(::Type{RefinementRNN},
                       data,
                       sys::DiscreteRandomSystem,
                       traj_cost_fn,
                       latent_dim, seq_len,
                       policy_hidden_dim, policy_depth,
                       delta_hidden_dim, delta_depth;
                       kwargs...)

Construct a fresh `RefinementRNN` and train it with
[`train_perturbation!`](@ref). Keyword arguments `kwargs...` are forwarded to `train_perturbation!`.
"""
function build_perturbation(::Type{RefinementRNN},
                            data::AbstractVector,
                            sys::DiscreteRandomSystem,
                            traj_cost_fn,
                            latent_dim::Integer,
                            seq_len::Integer,
                            policy_hidden_dim::Integer,
                            policy_depth::Integer,
                            delta_hidden_dim::Integer,
                            delta_depth::Integer;
                            max_seq_len::Union{Nothing,Integer}=nothing,
                            nheads::Integer=1,
                            activation=Flux.gelu,
                            kwargs...)
    isempty(data) && throw(ArgumentError("data must be non-empty"))

    first_sample = data[1]
    state_dim = size(first_sample.state_trajectory, 1)
    input_dim = size(first_sample.input_signal, 1)

    x0 = Vector(first_sample.state_trajectory[:, 1])
    u_zeros = zeros(eltype(first_sample.input_signal), input_dim, Int(seq_len))
    x_rollout = sys(x0, u_zeros)
    size(x_rollout, 2) == Int(seq_len) + 1 ||
        throw(DimensionMismatch("sys(x0, u_zeros) must return $(Int(seq_len) + 1) columns; got $(size(x_rollout, 2))"))
    x_body = x_rollout[:, 2:end]
    cost = traj_cost_fn(x_body)
    size(cost, 2) == size(x_body, 2) ||
        throw(DimensionMismatch("traj_cost_fn(x_body) must return cost with $(size(x_body, 2)) columns; got $(size(cost, 2))"))
    cost_dim = size(cost, 1)

    model = RefinementRNN(state_dim, input_dim, cost_dim, latent_dim, seq_len,
                          policy_hidden_dim, policy_depth,
                          delta_hidden_dim, delta_depth;
                          max_seq_len=max_seq_len, nheads=nheads, activation=activation)

    return train_perturbation!(model, data, sys, traj_cost_fn; kwargs...)
end

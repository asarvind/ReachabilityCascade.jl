module TrajectoryRefinerTraining

using Flux
using ..TrajectoryRefiner: CorrectionNetwork, ShootingBundle, refinement_loss, refinement_grads

"""
    train_refiner!(model, data_iter, refine_steps::Int, backprop_steps::Int;
                   opt=Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam()),
                   term_cost_fn, transition_fn, traj_mismatch_fn, imitation_weight=1.0)

Train a trajectory refiner (`CorrectionNetwork`) using unrolled refinement steps and truncated
backpropagation. `data_iter` must yield `ShootingBundle` objects carrying `x_guess` (including the
initial state), `u_guess`, and optional `x_target` (imitation trajectory over post-initial states).

# Arguments
- `model`: correction network to train.
- `data_iter`: collection/iterator of training samples.
- `refine_steps::Int`: number of refinement iterations unrolled per update.
- `backprop_steps::Int`: number of trailing refinement steps to keep on the gradient tape (must satisfy `backprop_steps <= refine_steps`).
- `opt`: Flux optimiser (default `Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam())`).
- `term_cost_fn`, `transition_fn`, `traj_mismatch_fn`: callbacks forwarded to `refinement_loss`.
- `imitation_weight`: weight on the imitation factor when a `target` trajectory is provided.

# Returns
`(model, losses)` with the trained model and a vector of per-step losses.
"""
function train_refiner!(model::CorrectionNetwork, data_iter, refine_steps::Int, backprop_steps::Int;
                        opt=Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam()),
                        term_cost_fn, transition_fn, traj_mismatch_fn,
                        imitation_weight::Real=1.0)
    backprop_steps <= refine_steps || throw(ArgumentError("backprop_steps must be ≤ refine_steps"))
    opt_state = Flux.setup(opt, model)
    losses = Float32[]

    for sample in data_iter
        sample isa ShootingBundle || throw(ArgumentError("data_iter must yield ShootingBundle objects"))
        tbatch = sample

        total_steps = max(refine_steps, 0)
        bsteps = clamp(backprop_steps, 0, total_steps)
        fsteps = total_steps - bsteps

        fwd_sample = fsteps > 0 ? model(tbatch, transition_fn, term_cost_fn, fsteps) : tbatch

        grads, loss_val = refinement_grads(model, transition_fn, term_cost_fn, traj_mismatch_fn,
                                           fwd_sample, bsteps;
                                           imitation_weight=imitation_weight)

        Flux.update!(opt_state, model, grads)
        push!(losses, Float32(loss_val))
    end

    return model, losses
end

end # module

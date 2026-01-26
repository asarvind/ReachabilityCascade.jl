using Random
import JLD2
import LazySets
import ReachabilityCascade.QuadDynamics: discrete_quadcopter_ref
import ReachabilityCascade.MPC: trajectory, smt_optimize_latent
import ReachabilityCascade.ControlSystem: DiscreteRandomSystem

"""
    quad_smt_formulas(ds; kwargs...) -> (smt_safety, smt_goal)

Build SMT safety and goal formulas for the quadcopter.

Safety constraints apply only to angles and angular rates using the bounds
in `ds.X`. Goal constraints bound the full state around zero with tunable
position, velocity, angle, and angular-rate limits. Angle and rate rows are
scaled by `angle_scale` for higher weight.
"""
function quad_smt_formulas(ds::DiscreteRandomSystem;
                           pos_bound::Real=1.0,
                           vel_bound::Real=0.5,
                           angle_bound::Real=0.1,
                           rate_bound::Real=0.1,
                           angle_scale::Real=5.0)
    hasproperty(ds, :X) || throw(ArgumentError("ds must have field X to infer state bounds"))
    X = getproperty(ds, :X)
    X isa LazySets.Hyperrectangle || throw(ArgumentError("ds.X must be a Hyperrectangle to infer bounds"))
    x_center = LazySets.center(X)
    x_radius = LazySets.radius_hyperrectangle(X)
    x_lo = x_center .- x_radius
    x_hi = x_center .+ x_radius

    smt_safety = Matrix{Float32}[]
    for idx in 7:12
        row_hi = zeros(Float32, 13)
        row_hi[idx] = 1.0f0
        row_hi[end] = -Float32(x_hi[idx])
        push!(smt_safety, reshape(row_hi, 1, :))

        row_lo = zeros(Float32, 13)
        row_lo[idx] = -1.0f0
        row_lo[end] = Float32(x_lo[idx])
        push!(smt_safety, reshape(row_lo, 1, :))
    end

    # Goal box is an AND across all inequalities; each row is its own matrix.
    smt_goal = Matrix{Float32}[]
    for (idx, bound) in ((1, pos_bound), (2, pos_bound), (3, pos_bound),
                         (4, vel_bound), (5, vel_bound), (6, vel_bound),
                         (7, angle_bound), (8, angle_bound), (9, angle_bound),
                         (10, rate_bound), (11, rate_bound), (12, rate_bound))
        scale = idx >= 7 ? Float32(angle_scale) : 1.0f0

        row_hi = zeros(Float32, 13)
        row_hi[idx] = scale
        row_hi[end] = -scale * Float32(bound)
        push!(smt_goal, reshape(row_hi, 1, :))

        row_lo = zeros(Float32, 13)
        row_lo[idx] = -scale
        row_lo[end] = -scale * Float32(bound)
        push!(smt_goal, reshape(row_lo, 1, :))
    end

    return smt_safety, smt_goal
end

function main()
    # ---------------------------
    # Editable settings
    # ---------------------------
    save_path = joinpath(pwd(), "data", "quadcopter", "quadtrajectories.jld2")
    max_samples = 4000
    start_sample = 1
    seed = 0
    save_period = 60.0

    # Dynamics / optimizer settings
    ds = discrete_quadcopter_ref(dt=0.01, t=0.05)
    model_base = (x, z) -> z
    opt_steps = ones(Int64, 50)
    max_penalty_evals = 200

    # ---------------------------
    # Run data collection
    # ---------------------------
    rng = Random.MersenneTwister(seed)
    smt_safety, smt_goal = quad_smt_formulas(ds)
    results = NamedTuple[]

    last_save = time()
    last_saved_iter = 0

    if max_samples != 0
        for sample_id in 1:max_samples
            # Always advance RNG so resuming preserves the sample sequence.
            x = LazySets.sample(ds.X; rng=rng)
            x[1:6] .= 0.0
            x[3] = rand(rng, -2.0:0.5:2.0)
            x[[9, 12]] .= 0.0

            if sample_id < start_sample
                continue
            end

            z0 = repeat(zeros(3), length(opt_steps))
            res = smt_optimize_latent(ds, model_base, x, z0, opt_steps, smt_safety, smt_goal;
                algo=:GN_CMAES,
                max_penalty_evals=max_penalty_evals,
                seed=seed + sample_id,
                latent_dim=3
            )

            if isfinite(res.eventual_time_safe)
                traj = trajectory(ds, model_base, x, res.z, opt_steps;
                    latent_dim=3,
                    output_map=identity,
                )
                t_cut = Int(res.eventual_time_safe)
                state_traj = traj.output_trajectory[:, 1:t_cut]
                input_traj = traj.input_trajectory[:, 1:(t_cut - 1)]
                push!(results, (
                    state_trajectory=state_traj,
                    input_signal=input_traj,
                ))
            end

            if time() - last_save >= save_period
                if isempty(save_path)
                    # skip saving
                elseif isempty(results)
                    @warn "No trajectories found; skipping save."
                else
                    mkpath(dirname(save_path))
                    JLD2.save(save_path,
                        "trajectories", results,
                        "seed", seed,
                        "last_saved_iter", sample_id,
                    )
                end
                last_save = time()
                last_saved_iter = sample_id
            end

            if sample_id % 10 == 0
                elapsed = round(time() - last_save; digits=2)
                println("samples=", sample_id, " saved=", length(results),
                        " last_saved=", last_saved_iter, " elapsed=", elapsed, "s")
            end
        end
    end

    if max_samples != 0
        if isempty(save_path)
            # skip saving
        elseif isempty(results)
            @warn "No trajectories found; skipping save."
        else
            mkpath(dirname(save_path))
            JLD2.save(save_path,
                "trajectories", results,
                "seed", seed,
                "last_saved_iter", last_saved_iter,
            )
        end
    end

    println("Done. saved=", length(results), " to ", save_path)
end

main()

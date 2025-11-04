using Random
using JLD2
using LazySets
import Flux

using ReachabilityCascade: TerminalGradientDatum, IntermediateGradientDatum, ControlGradientDatum,
    GANControlNet, train_gan_control!, load_gan_control,
    gan_predict_control
import ReachabilityCascade.CarDataGeneration: discrete_vehicles

struct CarControlData{D,S}
    data::D
    sampler::S
end

struct CarInterstateData{D}
    data::D
end

struct CarTerminalData{D}
    data::D
end

function Base.iterate(ccd::CarControlData, idx::Int=1)
    len = length(ccd.data)
    while idx <= len
        datum = ccd.data[idx]
        T = size(datum.input_signal, 2)
        if T == 0
            idx += 1
            continue
        end

        t = rand(1:T)
        current = datum.state_trajectory[:, t]
        next_logged = datum.state_trajectory[:, t + 1]
        logged_control = datum.input_signal[:, t]

        if !(all(isfinite, current) && all(isfinite, logged_control) && all(isfinite, next_logged))
            idx += 1
            continue
        end

        control = logged_control
        next_state = next_logged

        if rand(Bool)
            synthetic_control = sample(ccd.sampler.U)
            synthetic_state = try
                ccd.sampler(current, synthetic_control)
            catch err
                err isa DomainError ? next_logged : rethrow()
            end
            if all(isfinite, synthetic_control) && all(synthetic_state ∈ ccd.sampler.X)
                control = synthetic_control
                next_state = synthetic_state
            end
        end

        return ControlGradientDatum(control, current, next_state), idx + 1
    end
    return nothing
end

function Base.iterate(cid::CarInterstateData, idx::Int=1)
    len = length(cid.data)
    while idx <= len
        datum = cid.data[idx]
        T = size(datum.input_signal, 2)
        if T == 0
            idx += 1
            continue
        end

        start = rand(1:T)
        mid = rand((start + 1):(T + 1))
        finish = rand(mid:(T + 1))

        current = datum.state_trajectory[:, start]
        intermediate = datum.state_trajectory[:, mid]
        terminal = datum.state_trajectory[:, finish]

        if !(all(isfinite, current) && all(isfinite, intermediate) && all(isfinite, terminal))
            idx += 1
            continue
        end

        time_offset = mid - start
        total_offset = finish - start

        return IntermediateGradientDatum(intermediate, current, terminal, time_offset, total_offset), idx + 1
    end
    return nothing
end

function Base.iterate(ctd::CarTerminalData, idx::Int=1)
    len = length(ctd.data)
    while idx <= len
        datum = ctd.data[idx]
        T = size(datum.input_signal, 2)
        if T == 0
            idx += 1
            continue
        end

        start = rand(1:T)
        current = datum.state_trajectory[:, start]
        terminal = datum.state_trajectory[:, end]
        goal = hasproperty(datum, :goal) ? datum.goal : [0]

        if !(all(isfinite, current) && all(isfinite, terminal) && all(isfinite, goal))
            idx += 1
            continue
        end

        return TerminalGradientDatum(terminal, current, goal), idx + 1
    end
    return nothing
end

data = shuffle(JLD2.load("data/car/trajectories.jld2", "data"))
overtake_ids = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 for d in data]
overtake_data = data[overtake_ids]

function train_gan_network(; seed::Integer=rand(1:10^6),
                           epochs::Integer=5,
                           numdata::Integer=30000,
                           save_path::AbstractString="data/car/gancontrol/gan_control.jld2")
    Random.seed!(seed)
    ds = discrete_vehicles(0.25)

    numreach = min(numdata, length(data))
    numovertake = min(numdata, length(overtake_data))
    reach_data = data[1:numreach]
    task_data = overtake_data[1:numovertake]

    isempty(reach_data) && error("reach_data is empty; increase numdata or provide more trajectories.")
    isempty(task_data) && error("overtake_data is empty; cannot sample terminal data.")

    car_control_data = CarControlData(reach_data, ds)
    car_interstate_data = CarInterstateData(reach_data)
    car_terminal_data = CarTerminalData(task_data)

    state_dim = size(reach_data[1].state_trajectory, 1)
    control_dim = size(reach_data[1].input_signal, 1)
    example_goal = hasproperty(task_data[1], :goal) ? task_data[1].goal : [0.0]
    goal_dim = length(example_goal)

    terminal_kwargs = (gen_hidden=128, disc_hidden=128, enc_hidden=128,
                       n_glu_gen=3, n_glu_disc=3, n_glu_enc=3)
    intermediate_kwargs = (gen_hidden=128, disc_hidden=128, enc_hidden=128,
                           n_glu_gen=3, n_glu_disc=3, n_glu_enc=3)
    control_kwargs = (gen_hidden=128, disc_hidden=128, enc_hidden=128,
                      n_glu_gen=3, n_glu_disc=3, n_glu_enc=3)

    gan = GANControlNet(state_dim, goal_dim, control_dim;
                        terminal_kwargs=terminal_kwargs,
                        intermediate_kwargs=intermediate_kwargs,
                        control_kwargs=control_kwargs,
                        time_feature_dim=6)

    constructor_kwargs = (
        terminal_latent=8,
        intermediate_latent=8,
        control_latent=8,
        time_feature_dim=gan.time_feature_dim,
        terminal_kwargs=terminal_kwargs,
        intermediate_kwargs=intermediate_kwargs,
        control_kwargs=control_kwargs
    )

    constructor_info = Dict(
        "args" => (state_dim, goal_dim, control_dim),
        "kwargs" => constructor_kwargs
    )

    rule = Flux.Optimisers.OptimiserChain((Flux.ClipGrad(1.0), Flux.ClipNorm(1.0), Flux.Optimisers.Adam(1e-4)))

    @time train_gan_control!(gan,
                             car_terminal_data,
                             car_interstate_data,
                             car_control_data,
                             rule;
                             epochs=epochs,
                             carryover_limit=50,
                             save_path=save_path,
                             load_path=save_path,
                             constructor_info=constructor_info,
                             save_interval=Inf)

    println("GAN control training complete.")
    println("$(length(data)), $(length(overtake_data))")
    return gan
end

gan_save_path = "data/car/gancontrol/gan_control.jld2"
# train_gan_network(seed=4321, epochs=5, numdata=30000, save_path=gan_save_path)

gan_net = load_gan_control(gan_save_path)

thisdat = overtake_data[[d.state_trajectory[2, 1] <= 2.0 && d.state_trajectory[1, 1] <= 5.0 for d in overtake_data]]
current_state = thisdat[1].state_trajectory[:, 1]
goal = [0.0]
time_step = 1
total_time = size(thisdat[1].state_trajectory, 2) - 1
result = gan_predict_control(gan_net,
                             current_state,
                             goal,
                             time_step,
                             total_time)

current_state, result.intermediate_state, result.control

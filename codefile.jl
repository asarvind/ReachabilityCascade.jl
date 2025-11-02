using Random
using JLD2, LazySets
import Flux
using ReachabilityCascade: TerminalGradientDatum, IntermediateGradientDatum, ControlGradientDatum, RecurrentControlNet
import ReachabilityCascade.CarDataGeneration: discrete_vehicles

let
    Random.seed!(1)
    ds = discrete_vehicles(0.25)

    struct CarControlData
        data::AbstractVector
    end

    struct CarInterstateData
        data::AbstractVector
    end

    struct CarTerminalData
        data::AbstractVector
    end


    data = shuffle(JLD2.load("data/car/trajectories.jld2", "data"))
    overtake_ids = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 for d in data]
    overtake_data = data[overtake_ids]

    numdata = 30000
    numreach = min(numdata, length(data))
    numovertake = min(numdata, length(overtake_data))
    reach_data = data[1:numreach]
    task_data = overtake_data[1:numovertake]

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
                synthetic_control = sample(ds.U)
                synthetic_state = try
                    ds(current, synthetic_control)
                catch err
                    err isa DomainError ? next_logged : rethrow()
                end
                if all(isfinite, synthetic_control) && all(synthetic_state ∈ ds.X)
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

    car_control_data = CarControlData(reach_data);
    car_interstate_data = CarInterstateData(reach_data);
    car_terminal_data = CarTerminalData(task_data);

    rule = Flux.Optimisers.OptimiserChain((Flux.ClipGrad(10.0), Flux.ClipNorm(10.0), Flux.Optimisers.Adam(1e-3)))

    epochs = 5
    save_path = "data/car/recurrentcontrolflow/rcnet.jld2"
    recurrence_kwargs = (num_blocks=2, hidden=64, n_glu=2, bias=true,
                        clamp_lim=2.0)

    @time RecurrentControlNet(car_terminal_data, car_interstate_data, car_control_data, rule;
                        epochs=epochs,
                        save_path=save_path,
                        recurrence_kwargs=recurrence_kwargs,
                        carryover_limit=100);

    println("code execution complete.")
    println("$(length(data)), $(length(overtake_data))")
end

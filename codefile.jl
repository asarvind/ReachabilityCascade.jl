using Random
using JLD2, LazySets
import ReachabilityCascade.CarDataGeneration: discrete_vehicles

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

numdata = 10
data = JLD2.load("data/car/trajectories.jld2", "data")
overtake_ids = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 for d in data]
overtake_data = data[overtake_ids]

reach_data = data[1:numdata]
task_data = overtake_data[1:numdata]

function Base.iterate(ccd::CarControlData, state=1)
    bool = rand(Bool)
    if state < length(ccd.data) && state >= 1
        datum = ccd.data[state]
        t = rand(1:size(datum.input_signal, 2))
        current_states = datum.state_trajectory[:, t]
        if bool
            control_samples = datum.input_signal[:, t]           
            next_states = datum.state_trajectory[:, t+1]
        else
            control_samples = sample(ds.U)   
            next_states = ds(current_states, control_samples) 
        end
        return  ((control_samples, current_states, next_states), state + 1) 
    else
        return nothing
    end
end

function Base.iterate(cid::CarInterstateData, state=1)
    if state < length(cid.data) && state >= 1
        datum = cid.data[state]
        t = rand(1:size(datum.input_signal, 2))
        tnext = rand((t+1):size(datum.state_trajectory, 2))
        tend = rand(tnext:size(datum.state_trajectory, 2))
        current_states = datum.state_trajectory[:, t]
        intermediate_states = datum.state_trajectory[:, tnext] 
        terminal_states = datum.state_trajectory[:, tend]
        times = tnext - t
        total_times = size(datum.input_signal, 2)
        return ((intermediate_states, current_states, terminal_states, times, total_times), state + 1)
    end
end

function Base.iterate(ctd::CarTerminalData, state=1)
    if state < length(ctd.data) && state >= 1
        datum = ctd.data[state]
        t = rand(1:size(datum.input_signal, 2))
        tend = size(datum.state_trajectory, 2)
        current_states = datum.state_trajectory[:, t]
        terminal_states = datum.state_trajectory[:, tend]
        goals = [0]
        total_times = size(datum.input_signal, 2)
        return ((terminal_states, current_states, goals), state + 1)
    end
end
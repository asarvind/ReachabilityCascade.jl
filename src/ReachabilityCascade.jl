module ReachabilityCascade

module ControlSystem
    using Random, LinearAlgebra
    using LazySets
    include("controlsystem.jl")
end
#
using .ControlSystem: ContinuousSystem, DiscreteRandomSystem
export ContinuousSystem, DiscreteRandomSystem

module ControlUtilities
    using ..ControlSystem: ContinuousSystem, DiscreteRandomSystem
    using LazySets, JuMP, LinearAlgebra, Clarabel
    include("controlutils.jl")
end
using .ControlUtilities: linearize, lqr_lyap, correct_trajectory
export linearize, lqr_lyap, correct_trajectory

module Sampling
    using LazySets, StaticArrays
    include("sampling.jl")
end
using .Sampling: grid_serpentine
export grid_serpentine

module GatedLinearUnits
    using Flux
    include("glu.jl")
end
using .GatedLinearUnits: GLU, glu_mlp
export GLU, glu_mlp

module TransitionModels
    using Flux
    using ..GatedLinearUnits: GLU, glu_mlp
    include("TransitionModels/transition_network.jl")
    include("TransitionModels/training.jl")
end
using .TransitionModels: TransitionNetwork, train!, fit_transition_network, save_transition_network, load_transition_network, build
export TransitionNetwork, train!, fit_transition_network, save_transition_network, load_transition_network, build






module CarDynamics
    using LazySets
    using ..ControlSystem: ContinuousSystem, DiscreteRandomSystem
    include("examples/car/dynamics.jl")
    include("examples/car/modelbuilders.jl")
end

module CarDataGeneration
    using LazySets, JuMP, LinearAlgebra, HiGHS, JLD2
    using ..ControlSystem: DiscreteRandomSystem
    using ..ControlUtilities: linearize, lqr_lyap, correct_trajectory
    using ..CarDynamics: discrete_vehicles
    include("examples/car/datageneration.jl")
    include("examples/car/surrogatedatagen.jl")
    export generate_transition_dataset
end

include("SequenceTransform/SequenceTransform.jl")
using .SequenceTransform
export ScanMixer, SequenceTransformation



include("TrajectoryRefiner/TrajectoryRefiner.jl")
using .TrajectoryRefiner
export CorrectionNetwork

end

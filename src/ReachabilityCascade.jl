module ReachabilityCascade

#############
# Core systems
#############
module ControlSystem
    using Random, LinearAlgebra
    using LazySets
    include("controlsystem.jl")
end
using .ControlSystem: ContinuousSystem, DiscreteRandomSystem
export ContinuousSystem, DiscreteRandomSystem

module ControlUtilities
    using ..ControlSystem: ContinuousSystem, DiscreteRandomSystem
    using LazySets, JuMP, LinearAlgebra, Clarabel
    include("controlutils.jl")
end
using .ControlUtilities: linearize, lqr_lyap, correct_trajectory
export linearize, lqr_lyap, correct_trajectory

#############
# Sampling
#############
module Sampling
    using LazySets, StaticArrays
    include("sampling.jl")
end
using .Sampling: grid_serpentine
export grid_serpentine

#############
# Neural blocks
#############
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
using .TransitionModels: TransitionNetwork, fit_transition_network, save_transition_network, load_transition_network
export TransitionNetwork, fit_transition_network, save_transition_network, load_transition_network

#############
# Examples and data
#############
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

#############
# Sequence transforms
#############
module SequenceTransform
    using Flux
    using ..GatedLinearUnits: glu_mlp

    export ScanMixer, ForwardCumsumBlock, ReverseCumsumBlock, DirectBlock, SequenceTransformation

    include("SequenceTransform/blocks.jl")
    include("SequenceTransform/layer.jl")
    include("SequenceTransform/transformation.jl")
end
using .SequenceTransform
export ScanMixer, SequenceTransformation

#############
# Trajectory refiner
#############
module TrajectoryRefiner
    using Flux
    using ..SequenceTransform: SequenceTransformation

    include("TrajectoryRefiner/sample.jl")
    include("TrajectoryRefiner/networks.jl")
    include("TrajectoryRefiner/gradients.jl")
    include("TrajectoryRefiner/training.jl")
    using .TrajectoryRefinerTraining: train!, build

    export ShootingBundle, RefinementModel, refinement_loss, refinement_grads, train!, build,
           save_refinement_model, load_refinement_model
end
using .TrajectoryRefiner
export RefinementModel, train!, save_refinement_model, load_refinement_model

end

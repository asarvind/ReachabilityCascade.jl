module ReachabilityCascade

#############
# Training API (shared generics)
#############
module TrainingAPI
    export train!, build, save, load, gradient
    function train! end
    function build end
    function save end
    function load end
    function gradient end
end

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

module Robot3DOF
    using LazySets
    using ..ControlSystem: ContinuousSystem, DiscreteRandomSystem
    include("examples/robot3dof/dynamics.jl")
    export robot3dof_field, joint_positions, discrete_robot3dof
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
# Normalizing flows
#############
module NormalizingFlows
    import Flux
    using Random

    using ..GatedLinearUnits: glu_mlp

    export CouplingLayer, NormalizingFlow, encode, decode

    include("NormalizingFlows/NormalizingFlows.jl")
end
using .NormalizingFlows: CouplingLayer, NormalizingFlow, encode, decode
export CouplingLayer, NormalizingFlow, encode, decode

#############
# Invertible game (two-player)
#############
module InvertibleGame
    import Flux
    using Random

    using ..GatedLinearUnits: glu_mlp

    export InvertibleCoupling, encode, decode

    include("InvertibleGame/InvertibleGame.jl")
end
using .InvertibleGame: InvertibleCoupling
export InvertibleGame, InvertibleCoupling

#############
# MPC utilities (shared names, dispatch on model)
#############
include("MPC/MPC.jl")
using .MPC: trajectory, optimize_latent, mpc
export trajectory, optimize_latent, mpc

end

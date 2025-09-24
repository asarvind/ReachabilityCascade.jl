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

module NormalizingFlow
    using Flux
    using ..GatedLinearUnits: GLU, glu_mlp
    include("normalflow/thismain.jl")
end
using .NormalizingFlow: ConditionalFlow, loglikelihoods
export ConditionalFlow, loglikelihoods

# module NeuralReachability
#     using Flux, Random, LinearAlgebra
#     import JLD2
#     using ..NormalizingFlow: ConditionalFlow, loglikelihoods 
#     include("nrle/thismain.jl")
# end
# import .NeuralReachability: NRLE, encode, reach, train, load
# export NRLE, encode, reach, train, load 

# examples
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
end

end

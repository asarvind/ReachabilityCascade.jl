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

module GANModels
    using Flux
    using ..GatedLinearUnits: glu_mlp
    include("ganfiles/gan.jl")
    include("ganfiles/gan_training.jl")
end
using .GANModels: Gan, generator_forward, discriminator_forward, encoder_forward,
                  gan_gradients, gradient_norm
export Gan, generator_forward, discriminator_forward, encoder_forward,
       gan_gradients, gradient_norm

module HierarchicalBehaviorCloning
    using Flux
    using ..GANModels: Gan, generator_forward, discriminator_forward, encoder_forward
    include("HBC/hbcnet.jl")
end
using .HierarchicalBehaviorCloning: HierarchicalBehaviorCloner, task_forward,
                                    intermediate_forward, intermediate_level_forward,
                                    control_forward
export HierarchicalBehaviorCloner, task_forward, intermediate_forward,
       intermediate_level_forward, control_forward

module TrajectoryModels
    using Flux, Random
    using Statistics
    import JLD2
    include("trajectorydistribution/thismain.jl")
end

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

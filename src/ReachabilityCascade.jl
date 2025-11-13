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
    include("normalflow/utilities.jl")
    include("normalflow/affinecoupling.jl")
    include("normalflow/conditionalflow.jl")
end
using .NormalizingFlow: ConditionalFlow
export ConditionalFlow

module NormalizingFlowTransformer
    using Flux
    include("normalizingflowtransformer/flowtransformer.jl")
    include("normalizingflowtransformer/gradients.jl")
    include("normalizingflowtransformer/training.jl")
end
using .NormalizingFlowTransformer: FlowTransformer, flow_transformer_gradient, default_flow_loss, train!, load_flow_transformer
export FlowTransformer, flow_transformer_gradient, default_flow_loss, train!, load_flow_transformer


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

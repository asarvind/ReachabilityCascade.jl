module ReachabilityCascade

# using LinearAlgebra, StaticArrays, Random, LazySets, Flux, JuMP, Clarabel, JLD2, Plots, Statistics, Plots.Measures



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
    using JuMP
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
using .GatedLinearUnits: GLU
export GLU

module NormalizingFlow
    using Flux
    using ..GatedLinearUnits: GLU
    include("flow.jl")
end
using .NormalizingFlow: ConditionalFlow
export ConditionalFlow

# examples
module CarDynamics
    using LazySets
    include("examples/car/dynamics.jl")
end


end

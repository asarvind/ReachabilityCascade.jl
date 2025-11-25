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
    include("examples/car/datageneration.jl")
end

include("SequenceTransform/SequenceTransform.jl")
using .SequenceTransform
export ScanMixer, SequenceTransformation

module AdversarialRecurrence
    include("RecurrentGAN/layers.jl")
    include("RecurrentGAN/encoder.jl")
    include("RecurrentGAN/decoder.jl")
    include("RecurrentGAN/model.jl")
    include("RecurrentGAN/losses.jl")
end
using .AdversarialRecurrence: RecurrentGAN, Encoder, Decoder, encode, decode
using .AdversarialRecurrence: encoder_reconstruction_loss, decoder_reconstruction_loss
export RecurrentGAN, Encoder, Decoder, encode, decode
export encoder_reconstruction_loss, decoder_reconstruction_loss

include("TrajectoryRefiner/TrajectoryRefiner.jl")
using .TrajectoryRefiner
export CorrectionNetwork, RefinerSolver, step_refiner

end

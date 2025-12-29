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

    export SequenceTransformation, AttentionFFN

    include("SequenceTransform/attention.jl")
    include("SequenceTransform/transformation.jl")
end
using .SequenceTransform
export SequenceTransformation, AttentionFFN

#############
# Latent difference network
#############
module LatentDifferenceNetworks
    import Flux
    using Random
    using JLD2
    using ..GatedLinearUnits: glu_mlp
    using ..SequenceTransform: SequenceTransformation
    using ..ControlSystem: DiscreteRandomSystem
    import ..TrainingAPI: save, load

    include("LatentDifferenceNet/utils.jl")
    include("LatentDifferenceNet/network.jl")
    include("LatentDifferenceNet/io.jl")
    include("LatentDifferenceNet/perturbation.jl")
    include("LatentDifferenceNet/eval.jl")
    include("LatentDifferenceNet/perturbation_training.jl")

    export RefinementRNN, DeltaNetwork,
           spsa_update!, testrun,
           train_perturbation!, build_perturbation,
           save, load
end
using .LatentDifferenceNetworks: RefinementRNN, DeltaNetwork,
                                 spsa_update!, testrun,
                                 train_perturbation!, build_perturbation,
                                 save, load
export RefinementRNN, DeltaNetwork,
       spsa_update!, testrun,
       train_perturbation!, build_perturbation,
       save, load

#############
# Reactive denoising network (model-free)
#############
module ReactiveDenoisingNetworks
    import Flux
    using ..SequenceTransform: SequenceTransformation

    include("ReactiveDenoisingNet/network.jl")
    include("ReactiveDenoisingNet/io.jl")
    include("ReactiveDenoisingNet/gradients.jl")
    include("ReactiveDenoisingNet/training.jl")
    include("ReactiveDenoisingNet/eval.jl")

    export ReactiveDenoisingNet
end
using .ReactiveDenoisingNetworks: ReactiveDenoisingNet
export ReactiveDenoisingNet

end

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
    include("normalflow/recurrentflow.jl")
    include("normalflow/recurrentgradients.jl")
end
using .NormalizingFlow: ConditionalFlow, RecurrentConditionalFlow, recurrent_flow_gradient
export ConditionalFlow, RecurrentConditionalFlow, recurrent_flow_gradient

# module RecurrentControl
#     include("controlnet/recurrentControl.jl")
#     include("controlnet/recurrentControlGradients.jl")
#     include("controlnet/recurrentTraining.jl")
#     include("controlnet/recurrentControlPerturbation.jl")
# end
# using .RecurrentControl: RecurrentControlNet, predict_terminal_state,
#                          predict_state_at, predict_control_input, predict_control,
#                          terminal_flow_gradient, intermediate_flow_gradient,
#                          control_flow_gradient, train_recurrent_control!,
#                          load_recurrent_control, train_recurrent_control_perturb!,
#                          TerminalGradientDatum, IntermediateGradientDatum,
#                          ControlGradientDatum
# export RecurrentControlNet, predict_terminal_state, predict_state_at,
#        predict_control_input, predict_control, terminal_flow_gradient,
#        intermediate_flow_gradient, control_flow_gradient, train_recurrent_control!,
#        load_recurrent_control, train_recurrent_control_perturb!,
#        TerminalGradientDatum, IntermediateGradientDatum, ControlGradientDatum

module GANModels
    using Flux
    using ..GatedLinearUnits: glu_mlp
    include("ganfiles/gan.jl")
    include("ganfiles/gan_training.jl")
    include("ganfiles/examples.jl")
end
using .GANModels: Gan, generator_forward, discriminator_forward, encoder_forward,
                  gan_gradients, gradient_norm, GanExamples
export Gan, generator_forward, discriminator_forward, encoder_forward,
       gan_gradients, gradient_norm, GanExamples

# module GANControl
#     using Flux
#     include("controlgan/ganControl.jl")
#     include("controlgan/ganControlTraining.jl")
# end
# import .GANControl: GANControlNet, train_gan_control!, load_gan_control
# const gan_predict_terminal_state = GANControl.predict_terminal_state
# const gan_predict_state_at = GANControl.predict_state_at
# const gan_predict_control_input = GANControl.predict_control_input
# const gan_predict_control = GANControl.predict_control
# export GANControlNet, gan_predict_terminal_state, gan_predict_state_at,
#        gan_predict_control_input, gan_predict_control,
#        train_gan_control!, load_gan_control

# module TrajectoryTransformers
#     using Flux
#     include("trajectorytransformer/trajectoryTransformer.jl")
# end
# using .TrajectoryTransformers: TrajectoryTransformer, transform_sequence,
#                               predict_state_sequence, predict_latent_sequence
# export TrajectoryTransformer, transform_sequence,
#        predict_state_sequence, predict_latent_sequence

# module BitwiseTransformers
#     using Flux
#     include("bitwisetransformer/probabilityTransformer.jl")
#     include("bitwisetransformer/hierarchicalBitwiseTransformer.jl")
#     include("bitwisetransformer/gradients.jl")
#     include("bitwisetransformer/hierarchicalBitwiseTraining.jl")
# end
# using .BitwiseTransformers: ProbabilityTransformer,
#                             HierarchicalBitwiseTransformer,
#                             gradients
# import .BitwiseTransformers: train!
# export ProbabilityTransformer,
#        HierarchicalBitwiseTransformer,
#        gradients,
#        train!

# module HierarchicalBehaviorCloning
#     using Flux
#     using ..GANModels: Gan, generator_forward, discriminator_forward, encoder_forward
#     include("HBC/hbcnet.jl")
#     include("HBC/examples.jl")
# end
# using .HierarchicalBehaviorCloning: HierarchicalBehaviorCloner, task_forward,
#                                     intermediate_forward, intermediate_level_forward,
#                                     control_forward, HBCExamples
# export HierarchicalBehaviorCloner, task_forward, intermediate_forward,
#        intermediate_level_forward, control_forward, HBCExamples

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

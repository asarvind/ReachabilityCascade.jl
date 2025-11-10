using Test
using ReachabilityCascade

include("normalizingflowtransformer/flow_transformer_tests.jl")
include("normalizingflowtransformer/flow_transformer_gradient_tests.jl")
include("normalizingflowtransformer/flow_transformer_training_tests.jl")
include("behavioralcloning/data_perturbation_tests.jl")
include("behavioralcloning/residual_control_transformer_tests.jl")
include("simpletransformers/simple_sequence_transformer_tests.jl")
include("imitationlearning/refine_control_tests.jl")
include("imitationlearning/gradient_tests.jl")

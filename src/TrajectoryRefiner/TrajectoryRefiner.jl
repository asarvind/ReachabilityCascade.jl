module TrajectoryRefiner

using Flux
using ..SequenceTransform: SequenceTransformation

include("sample.jl")
include("networks.jl")
include("gradients.jl")
include("training.jl")
using .TrajectoryRefinerTraining: train!

export ShootingBundle, RefinementModel, refinement_loss, refinement_grads, train!, build,
       save_refinement_model, load_refinement_model

end

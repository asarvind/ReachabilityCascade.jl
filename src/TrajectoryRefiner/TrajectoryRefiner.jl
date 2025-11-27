module TrajectoryRefiner

using Flux
using ..SequenceTransform: SequenceTransformation

include("sample.jl")
include("networks.jl")
include("gradients.jl")
include("training.jl")
using .TrajectoryRefinerTraining: train_refiner!

export ShootingBundle, CorrectionNetwork, refinement_loss, refinement_grads, train_refiner!

end

module TrajectoryRefiner

using Flux
using ..SequenceTransform: SequenceTransformation

include("networks.jl")
include("solver.jl")
include("gradients.jl")

export CorrectionNetwork, refine, refinement_loss, refinement_grads

end

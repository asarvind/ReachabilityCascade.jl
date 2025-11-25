module TrajectoryRefiner

using Flux
using ..SequenceTransform: SequenceTransformation

include("networks.jl")
include("solver.jl")

export CorrectionNetwork, RefinerSolver, step_refiner

end

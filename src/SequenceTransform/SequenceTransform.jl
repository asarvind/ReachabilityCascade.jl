module SequenceTransform

using Flux
using ..GatedLinearUnits: glu_mlp


export ScanMixer, ForwardCumsumBlock, ReverseCumsumBlock, DirectBlock, SequenceTransformation

include("blocks.jl")
include("layer.jl")
include("transformation.jl")

end

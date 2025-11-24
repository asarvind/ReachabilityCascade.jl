using Flux

include("layers.jl")
include("encoder.jl")
include("decoder.jl")
include("model.jl")

export RecurrentGAN, Encoder, Decoder
export encode, decode

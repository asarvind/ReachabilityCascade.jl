module KolmogorovRepresentation

using Flux
using ..KolmogorovLayers: SprecherLayer

"""
    SprecherNetwork(input_dim::Int, hidden_dims::AbstractVector{<:Integer}, output_dim::Int; act=relu)

Build a Sprecher-style representation network: a stack of `SprecherLayer`s followed by a final `Dense`
(no activation). `hidden_dims` sets the output dimension of each Sprecher layer in order.
Prepends an initial `Dense` projection (no activation) from `input_dim` to the first hidden width.
"""
struct SprecherNetwork{C}
    model::C
end

Flux.@layer SprecherNetwork

function SprecherNetwork(input_dim::Int, hidden_dims::AbstractVector{<:Integer}, output_dim::Int; act=relu)
    @assert !isempty(hidden_dims) "hidden_dims must contain at least one layer width"
    layers = Any[]
    first_hidden = first(hidden_dims)
    push!(layers, Dense(input_dim, first_hidden; bias=true))  # initial linear projection
    d = first_hidden
    for h in Iterators.drop(hidden_dims, 1)
        push!(layers, SprecherLayer(d, h; act=act))
        d = h
    end
    push!(layers, Dense(d, output_dim; bias=true))
    return SprecherNetwork(Chain(layers...))
end

# Forward pass; supports vector or (features, batch) input as underlying layers do.
function (m::SprecherNetwork)(x)
    # Align input type to first layer parameters to avoid Float64→Float32 slowdowns.
    first_layer = m.model.layers[1]
    T = eltype(first_layer.weight)
    xT = T === eltype(x) ? x : T.(x)
    return m.model(xT)
end

end # module

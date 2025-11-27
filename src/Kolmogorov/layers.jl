module KolmogorovLayers

using Flux

"""
    SprecherLayer(Q::AbstractMatrix, lambda::AbstractVector, bias::AbstractVector)

Compute `y = λ' * act.(x .+ Q)` (columnwise shifts by `Q`) and add `bias`, where `Q` has one column per output.
Accepts vectors or `(features, batch)` matrices; outputs `(out_dim)` or `(out_dim, batch)`.
"""
struct SprecherLayer{T,V,F}
    Q::T        # shift matrix (in_dim × out_dim)
    lambda::V   # weighting vector (length in_dim)
    bias::V     # bias vector (length out_dim)
    act::F      # activation function (defaults to relu)
    out_dim::Int
end

Flux.@layer SprecherLayer

function SprecherLayer(Q::AbstractMatrix, lambda::AbstractVector, bias::AbstractVector; act=relu)
    in_dim, out_dim = size(Q)
    @assert length(lambda) == in_dim "lambda length must match input dimension"
    @assert length(bias) == out_dim "bias length must match output dimension"
    return SprecherLayer(Q, lambda, bias, act, out_dim)
end

"""
    SprecherLayer(input_dim::Int, output_dim::Int)

Convenience constructor with Dense-like initialization for `Q` (Glorot uniform),
`lambda = ones`, `bias = zeros`, and activation `act` (default `relu`).
"""
function SprecherLayer(input_dim::Int, output_dim::Int; act=relu)
    std = sqrt(6f0 / (input_dim + output_dim))
    Q = rand(Float32, input_dim, output_dim) .* (2f0 * std) .- std
    lambda = ones(Float32, input_dim)
    bias = zeros(Float32, output_dim)
    return SprecherLayer(Q, lambda, bias, act, output_dim)
end

function (m::SprecherLayer)(x::AbstractArray)
    function forward_single(xv)
        @assert length(xv) == size(m.Q, 1) "Input dimension mismatch"
        T = eltype(m.Q)
        xvT = T === eltype(xv) ? xv : T.(xv)
        # Shift by each column of Q, apply ReLU elementwise, then weight by lambda
        shifted = m.act.(m.Q .+ reshape(xvT, :, 1))  # in_dim × out_dim
        y = (m.lambda' * shifted)[:] .+ m.bias       # out_dim vector
        return y
    end

    if ndims(x) == 1
        return forward_single(x)
    elseif ndims(x) == 2
        @assert size(x, 1) == size(m.Q, 1) "Input dimension mismatch"
        T = eltype(m.Q)
        xT = T === eltype(x) ? x : T.(x)
        # Broadcast over batch in one go: (in_dim, out_dim, batch)
        shifted = m.act.(m.Q .+ reshape(xT, size(m.Q, 1), 1, size(xT, 2)))
        λ = reshape(m.lambda, :, 1, 1)
        weighted = sum(λ .* shifted; dims=1)                   # 1 × out_dim × batch
        y = reshape(weighted, size(m.Q, 2), size(xT, 2))       # out_dim × batch
        return y .+ reshape(m.bias, :, 1)
    else
        error("Unsupported input dimensions for SprecherLayer: $(ndims(x))")
    end
end

end # module

using Flux

"""
    IterativeBlock

Lightweight recurrent cell for fixed-point refinement.

    h_{t+1} = activation(W * vcat(h_t, x) + b)

Arguments:
- `h`: hidden/state vector or batch
- `x`: constant input vector or batch (already concatenated upstream)

Returns:
- updated hidden state `h_{t+1}` with same shape as `h`

Notes:
- Input `x` is reused each iteration; only `h` evolves.
"""
struct IterativeBlock{L}
    layer::L
end

Flux.@layer IterativeBlock

function (m::IterativeBlock)(h, x)
    # Input to layer is [h; x]
    combined = vcat(h, x)
    h_new = m.layer(combined)
    return h_new
end

# ============================ Utilities =====================================

"""
    softclamp(x; limit=2.0f0)

Apply a smooth clamp `limit * tanh(x/limit)` elementwise to keep values within
`[-limit, limit]`. Useful for stabilizing predicted log-scales.

# Keywords
- `limit` : positive scalar clamp magnitude (default `5.0f0`)

# Returns
Array with the same shape as `x`.
"""
softclamp(x; limit=2.0f0) = limit .* tanh.(x ./ limit)

"""
    _bexpand(mask, B)

Broadcast helper that repeats a `(D, 1)` mask along the batch dimension to
match `(D, B)`. If `size(mask, 2) == B`, returns `mask` unchanged.
"""
_bexpand(mask::AbstractVecOrMat, B::Integer) = size(mask, 2) == B ? mask : repeat(mask, 1, B)
_require_vector(x, name::AbstractString) =
    ndims(x) == 1 ? x : throw(ArgumentError("$name must be a 1D vector; got ndims=$(ndims(x))"))

_require_matrix(x, name::AbstractString) =
    ndims(x) == 2 ? x : throw(ArgumentError("$name must be a 2D array; got ndims=$(ndims(x))"))

_pad_first_col(mat::AbstractMatrix, T::Integer) = begin
    T_int = Int(T)
    size(mat, 2) == T_int && return mat
    size(mat, 2) == T_int - 1 || throw(ArgumentError("expected $(T_int-1) or $T_int columns; got $(size(mat, 2))"))
    return hcat(zeros(eltype(mat), size(mat, 1), 1), mat)
end

import Flux

_softmax_average(x::AbstractArray; temperature::Real=1) = begin
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    v = vec(x)
    w = Flux.softmax(v ./ temperature)
    return sum(w .* v)
end

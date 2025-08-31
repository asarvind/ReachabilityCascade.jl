"""
    grid_serpentine(H, step;
                    dims=nothing) -> Vector{SVector}

Enumerate grid points inside the hyperrectangle `H` in a multidimensional
serpentine (reflected) order so that each consecutive point is a grid neighbor
(differs by one step in exactly one *varied* coordinate).

Arguments
- `H::Hyperrectangle` : a `LazySets.Hyperrectangle`.
- `step::Union{Real,AbstractVector}`: scalar or vector of step sizes **for the varied dimensions only**.
          If `dims` is provided, `step` refers to the axes in the order of `dims`.
          If `dims === nothing` (default, vary all axes), `step` may be a scalar
          (broadcast to all dims) or a length‑`N` vector.
- `dims::Union{Nothing,AbstractVector{<:Integer}}`: indices of dimensions to vary (default: vary **all** dimensions).
          Non‑varied dimensions are fixed at the hyperrectangle *center*.

Returns
- `Vector{SVector{N,Float64}}` of points in neighbor‑preserving order.

Notes
- For each varied axis `i`, the number of samples is
  `n[i] = floor(Int, (hi[i]-lo[i]) / step[i]) + 1` (≥ 2 when the interval is nonzero).
- We build closed ranges `range(lo[i]; length=n[i], stop=hi[i])` so endpoints are included.
- If only one axis varies, the result is a simple 1‑D snake.
"""
function grid_serpentine(H::Hyperrectangle,
                         step::Union{Real,AbstractVector};
                         dims::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    c = LazySets.center(H)
    r = [H.radius[i] for i in 1:dim(H)]
    lo, hi = c .- r, c .+ r
    N = length(lo)

    # normalize dims (which axes vary)
    vary = dims === nothing ? collect(1:N) : sort!(unique!(collect(dims)))
    @assert !isempty(vary) "`dims` must contain at least one dimension."
    @assert all(1 .<= vary .<= N) "`dims` entries must be between 1 and $N."

    # Build a per-axis step vector, using `step` only for the varied axes, in the
    # exact order given by `vary`.
    stepv = Vector{Float64}(undef, N)
    fill!(stepv, NaN)  # unused for fixed axes
    if step isa Real
        for i in vary
            stepv[i] = float(step)
        end
    else
        s = collect(step)
        if dims === nothing
            @assert length(s) == N "When `dims` is not given, `step` must be scalar or length N."
            for (i, si) in enumerate(s)
                stepv[i] = float(si)
            end
        else
            @assert length(s) == length(vary) "`step` length must equal length(dims)."
            for (j, ax) in enumerate(vary)
                stepv[ax] = float(s[j])
            end
        end
    end

    # number of grid points per axis for varied axes; single point for fixed axes
    countv = Vector{Int}(undef, N)
    for i in 1:N
        if i in vary
            w = hi[i] - lo[i]
            @assert isfinite(stepv[i]) && stepv[i] > 0 "Positive `step` required on varied axes."
            n = max(1, floor(Int, w / stepv[i]) + 1)
            if n == 1 && w > 0
                n = 2  # force endpoints when interval is non-degenerate
            end
            countv[i] = n
        else
            countv[i] = 1
        end
    end

    # values per axis: full range for varied axes, center for fixed ones
    axes_vals = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        if countv[i] == 1
            axes_vals[i] = [c[i]]
        else
            axes_vals[i] = collect(range(lo[i]; length=countv[i], stop=hi[i]))
        end
    end

    # recursively build serpentine order
    function serpentine(vals::Vector{Vector{T}}) where {T}
        if length(vals) == 1
            return [[x] for x in vals[1]]
        else
            head = vals[1]
            tail_lists = serpentine(vals[2:end])
            out = Vector{Vector{T}}()
            sizehint!(out, length(head) * length(tail_lists))
            for (i, x) in enumerate(head)
                seq = (i % 2 == 1) ? tail_lists : reverse(tail_lists)
                for tail in seq
                    push!(out, vcat([x], tail))
                end
            end
            return out
        end
    end

    tuples = serpentine(axes_vals)
    return [SVector{N,Float64}(tuple...) for tuple in tuples]
end
function randn_by_id(ids::AbstractVector{<:Integer}, n::Integer; rng=Random.default_rng())
    m = length(ids)
    X = Matrix{Float64}(undef, n, m)
    m == 0 && return X

    p = sortperm(ids)  # group equal ids contiguously

    i = 1
    @views while i <= m
        id = ids[p[i]]
        j = i + 1
        @inbounds while j <= m && ids[p[j]] == id
            j += 1
        end
        j -= 1  # now [i:j] is the run of this id

        col = randn(rng, n)     # one vector for this unique id
        X[:, p[i:j]] .= col     # broadcast into all matching columns

        i = j + 1
    end
    return X
end
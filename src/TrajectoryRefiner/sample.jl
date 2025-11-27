mutable struct ShootingBundle
    x_guess::AbstractArray          # (state_dim, seq_len+1, batch) includes initial state
    u_guess::AbstractArray          # (input_dim, seq_len, batch)
    x_target::Union{Nothing,AbstractArray} # optional imitation target (state_dim, seq_len, batch) over post-initial states
end

# -- Shape helpers -------------------------------------------------------------

_errmsg(name, nd) = ArgumentError("$name must have 1, 2, or 3 dimensions (got $nd)")

function _as_state_seq(x::AbstractArray, name)
    nd = ndims(x)
    if nd == 1
        return reshape(x, :, 1, 1)
    elseif nd == 2
        return reshape(x, size(x, 1), size(x, 2), 1)
    elseif nd == 3
        return x
    else
        throw(_errmsg(name, nd))
    end
end

function _as_input_seq(u::AbstractArray, name)
    nd = ndims(u)
    if nd == 1
        return reshape(u, :, 1, 1)
    elseif nd == 2
        return reshape(u, size(u, 1), size(u, 2), 1)
    elseif nd == 3
        return u
    else
        throw(_errmsg(name, nd))
    end
end

function _match_batch3(arr::AbstractArray, batch::Int, name::AbstractString)
    arr_batch = size(arr, 3)
    if arr_batch == batch
        return arr
    elseif arr_batch == 1 && batch > 1
        return repeat(arr, 1, 1, batch)
    else
        throw(ArgumentError("$name batch size $arr_batch does not match required batch $batch"))
    end
end

function _match_batch2(arr::AbstractArray, batch::Int, name::AbstractString)
    arr_batch = size(arr, 2)
    if arr_batch == batch
        return arr
    elseif arr_batch == 1 && batch > 1
        return repeat(arr, 1, batch)
    else
        throw(ArgumentError("$name batch size $arr_batch does not match required batch $batch"))
    end
end

function _coerce_target(target, state_dim::Int, seq_len::Int, batch::Int)
    target === nothing && return nothing
    (target isa AbstractArray && isempty(target)) && return nothing

    nd = ndims(target)
    if nd == 1
        target = reshape(target, :, 1, 1)
    elseif nd == 2
        target = reshape(target, size(target, 1), size(target, 2), 1)
    elseif nd == 3
        target = target
    else
        throw(_errmsg("x_target", nd))
    end

    size(target, 1) == state_dim || throw(ArgumentError("x_target state dimension $(size(target, 1)) must equal $state_dim"))
    if size(target, 2) != seq_len
        throw(ArgumentError("x_target sequence length $(size(target, 2)) must equal $seq_len"))
    end

    return _match_batch3(target, batch, "x_target")
end

# -- Constructors --------------------------------------------------------------

function ShootingBundle(x_guess::AbstractArray, u_guess::AbstractArray; x_target=nothing)
    x_guess = _as_state_seq(x_guess, "x_guess")
    u_guess = _as_input_seq(u_guess, "u_guess")

    state_dim, seq_len_x, batch = size(x_guess)
    u_guess = _match_batch3(u_guess, batch, "u_guess")
    seq_len_u = size(u_guess, 2)

    seq_len_x >= 2 || throw(ArgumentError("x_guess must include initial state and at least one step"))
    seq_len_x == seq_len_u + 1 || throw(ArgumentError("x_guess length $seq_len_x must equal u_guess length $seq_len_u + 1 (initial state + trajectory)"))

    seq_len_body = seq_len_x - 1
    target = _coerce_target(x_target, state_dim, seq_len_body, batch)

    return ShootingBundle(x_guess, u_guess, target)
end

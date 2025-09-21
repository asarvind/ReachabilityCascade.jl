function trenc(strj::AbstractMatrix{<:Real}, utrj::AbstractMatrix{<:Real})
    # compute vehicle positions at different time steps and append time
    # time steps
    t_max = size(strj, 2)
    divs = [1, 2, 4, 8, 16, 32]
    t_seq = [max(t_max÷i, 2) for i in divs]
    # combined positions
    cpos = vcat(vec(strj[1:2, t_seq]'), t_seq)

    # arguments for minimum safety of the state 
    ds = discrete_vehicles(0.25)
    # rectangular safety 
    rect_margin = vec(minimum(min.(high(ds.X) .- strj, strj .- low(ds.X)), dims=2))[2:7]
    # forward vehicle safety 
    rel_for = max.(abs.(strj[1, :] - strj[8, :]) .- 5.0, strj[2, :] - strj[9, :] .- 2.0)
    rel_on = max.(abs.(strj[1, :] - strj[11, :]) .- 5.0, strj[12, :] - strj[2, :] .- 2.0)
    obspos = vcat(rel_for[findmin(rel_for)[2]], rel_on[findmin(rel_on)[2]])
    
    return vcat(t_seq, cpos, rect_margin, obspos, utrj[:, 1])*0.1
end

function context_car(strj::AbstractMatrix{<:Real}, utrj::AbstractMatrix{<:Real})
    return vcat( strj[:, 1], strj[1, end] - strj[8, end] )
end

# struct LMTE{E}
#     context_dim::Integer 
#     control_dim::Integer 
#     transfer_dim::Integer
#     experts::Vector{E}
# end

# Flux.@layer LMTE 

# # ================== Constructors ======================== 
# function LMTE(context_dim::Integer, control_dim::Integer, transfer_dim::Integer, N::Integer; kwargs...)
#     @assert N >= 2 "number of experts should be at least 2"

#     # construct head expert 
#     head = conditional_flow(transfer_dim, context_dim; kwargs...)

#     # construct intermediate experts
#     body = [conditional_flow(transfer_dim, (transfer_dim + context_dim); kwargs...) for _ in 1:(N-2)] 

#     # construct policy expert 
#     tail = conditional_flow(control_dim, (transfer_dim + context_dim); kwargs...)

#     experts = vcat(head, body, tail)

#     return LMTE(context_dim, control_dim, transfer_dim, experts)
# end
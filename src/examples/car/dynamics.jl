module CarDynamics

using LazySets

# lwb = Float32.(2.5789)
# m = Float32(1093.3)
# μ = 1.0489 |> Float32
# lf = 1.156 |> Float32
# lr = 1.422 |> Float32
# hcg = 0.574 |> Float32
# Iz = 1791.6 |> Float32
# CSf = 20.89 |> Float32
# CSr = 20.89 |> Float32
# g = 9.8 |> Float32

function smoothfield(x, u, v)
    b1 = x[4]*cos(x[5]+x[7])
    b2 = x[4]*sin(x[5]+x[7])
    b3 = u[1]
    b4 = u[2]
    b5 = x[6]   
    b6 = (μ*m/(Iz*(lr+lf)))*( lf*CSf*x[3]*(g*lr-u[2]*hcg) +   (lr*CSr*(g*lf+u[2]*hcg)-lf*CSf*(g*lr-u[2]*hcg))*x[7] - (lf^2*CSf*(g*lr-u[2]*hcg)+ lr^2*CSr*(g*lf+u[2]*hcg))*x[6]/x[4] )
    b7 = (μ/(x[4]*(lr+lf)))*( CSf*(g*lr-u[2]*hcg)*x[3] - (CSr*(g*lf+u[2]*hcg) + CSf*(g*lr-u[2]*hcg))*x[7] - (lf*CSf*(g*lr-u[2]*hcg) - lr*CSr*(g*lf+u[2]*hcg))x[6]/x[4] ) - x[6]
    return [b1, b2, b3, b4, b5, b6, b7]
end


"""
carfield(x::Vector{<:Real}, u::Vector{<:Real}, w::Vector{<:Real}=zeros(7))
"""
function carfield(x::Vector{<:Real}, u::Vector{<:Real}, w::Vector{<:Real}=zeros(7))
    if abs(x[4]) < 0.1
        a = changerate1(x,u)# approximate rate of change of state
    else
        a = changerate2(x,u) # approximate rate of change of state
    end
    return a
end

"""
changerate1(x::Vector{<:Real}, u::Vector{<:Real})

returns approximate rate of change of state vector at a time point when speed is less than ``0.1``
"""
function changerate1(x::Vector{<:Real}, u::Vector{<:Real})
    lwb = Float32.(2.5789)
    m = Float32(1093.3)
    μ = 1.0489 |> Float32
    lf = 1.156 |> Float32
    lr = 1.422 |> Float32
    hcg = 0.574 |> Float32
    Iz = 1791.6 |> Float32
    CSf = 20.89 |> Float32
    CSr = 20.89 |> Float32
    g = 9.8 |> Float32

    a1 = x[4]*cos(x[5] + x[7])
    a2 = x[4]*sin(x[5] + x[7])
    a3 = u[1]
    a4 = u[2]
    a5 = x[4]*cos(x[7])/lwb*tan(x[3])
    a7 = 1/(1 + (tan(x[3])*lr/lwb)^2)*lr/(lwb*(cos(x[3])^2))*u[1]
    a6 = 1/lwb*(u[2]*cos(x[7])*tan(x[3]) - x[4]*sin(x[7])*tan(x[3])*a7 + x[4]*cos(x[7])*u[1]/(cos(x[3])^2) )
    return [a1, a2, a3, a4, a5, a6, a7]
end

"""
changerate2(x::Vector{<:Real}, u::Vector{<:Real})

returns approximate rate of change of state vector at a time point when speed is greater than ``0.1``
"""
function changerate2(x::Vector{<:Real}, u::Vector{<:Real})
    lwb = Float32.(2.5789)
    m = Float32(1093.3)
    μ = 1.0489 |> Float32
    lf = 1.156 |> Float32
    lr = 1.422 |> Float32
    hcg = 0.574 |> Float32
    Iz = 1791.6 |> Float32
    CSf = 20.89 |> Float32
    CSr = 20.89 |> Float32
    g = 9.8 |> Float32

    b1 = x[4]*cos(x[5]+x[7])
    b2 = x[4]*sin(x[5]+x[7])
    b3 = u[1]
    b4 = u[2]
    b5 = x[6]   
    b6 = (μ*m/(Iz*(lr+lf)))*( lf*CSf*x[3]*(g*lr-u[2]*hcg) +   (lr*CSr*(g*lf+u[2]*hcg)-lf*CSf*(g*lr-u[2]*hcg))*x[7] - (lf^2*CSf*(g*lr-u[2]*hcg)+ lr^2*CSr*(g*lf+u[2]*hcg))*x[6]/x[4] )
    b7 = (μ/(x[4]*(lr+lf)))*( CSf*(g*lr-u[2]*hcg)*x[3] - (CSr*(g*lf+u[2]*hcg) + CSf*(g*lr-u[2]*hcg))*x[7] - (lf*CSf*(g*lr-u[2]*hcg) - lr*CSr*(g*lf+u[2]*hcg))x[6]/x[4] ) - x[6]
    return [b1, b2, b3, b4, b5, b6, b7]
end

# end of module
end


module QuadDynamics
######################## begin module

function quadfield(x::Vector{Float64}, u::Vector{Float64}, w::Vector{<:Real}=zeros(12))
    #=== 
        x(1)  = x       (position in X)
        x(2)  = y       (position in Y)
        x(3)  = z       (position in Z)
        x(4)  = dx/dt   (velocity X)
        x(5)  = dy/dt   (velocity Y)
        x(6)  = dz/dt   (velocity Z)
        x(7)  = phi     (roll)
        x(8)  = theta   (pitch)
        x(9)  = psi     (yaw)
        x(10) = p       (angular rate around body-x)
        x(11) = q       (angular rate around body-y)
        x(12) = r       (angular rate around body-z)

        u(1) = T         (total thrust)
        u(2) = tau_phi   (roll torque)
        u(3) = tau_theta (pitch torque)
        u(4) = tau_psi   (yaw torque)
    ===#

    # -- Define parameters --
    m  = 1.5       # mass [kg]
    Ix = 0.02      # inertia around x-axis [kg·m^2]
    Iy = 0.02      # inertia around y-axis [kg·m^2]
    Iz = 0.04      # inertia around z-axis [kg·m^2]
    g  = 9.81      # gravitational acceleration [m/s^2]

    # -- Extract states for readability --
    x1, x2, x3  = x[1],  x[2],  x[3]
    x4, x5, x6  = x[4],  x[5],  x[6]
    x7, x8, x9  = x[7],  x[8],  x[9]
    x10, x11, x12 = x[10], x[11], x[12]

    # -- Extract inputs --
    u1, u2, u3, u4 = u[1], u[2], u[3], u[4]

    # -- Allocate derivative vector --
    dx = zeros(12)

    # 1) Position kinematics
    dx[1] = x4
    dx[2] = x5
    dx[3] = x6

    # 2) Translational dynamics
    dx[4] = -(u1/m) * ( sin(x9)*sin(x7) + cos(x9)*cos(x7)*sin(x8) )
    dx[5] = -(u1/m) * ( cos(x7)*sin(x9)*sin(x8) - cos(x9)*sin(x7) )
    dx[6] = g - (u1/m) * ( cos(x7)*cos(x8) )

    # 3) Euler-angle kinematics
    dx[7] = x10 + sin(x7)*tan(x8)*x11 + cos(x7)*tan(x8)*x12
    dx[8] = cos(x7)*x11 - sin(x7)*x12
    dx[9] = (sin(x7)/cos(x8))*x11 + (cos(x7)/cos(x8))*x12

    # 4) Body-rate dynamics
    dx[10] = (1/Ix) * ( u2 + (Iy - Iz)*x11*x12 )
    dx[11] = (1/Iy) * ( u3 + (Iz - Ix)*x12*x10 )
    dx[12] = (1/Iz) * ( u4 + (Ix - Iy)*x10*x11 )

    return dx
end

















################# end module
end
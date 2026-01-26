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
    dx[4] = (u1/m) * ( cos(x7)*sin(x8)*cos(x9) + sin(x7)*sin(x9) )
    dx[5] = (u1/m) * ( cos(x7)*sin(x8)*sin(x9) - sin(x7)*cos(x9) )
    dx[6] = (u1/m) * ( cos(x7)*cos(x8) ) - g

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

"""
    discrete_quadcopter_ref(; kwargs...) -> DiscreteRandomSystem

Build a discrete-time quadcopter system where the control input is a reference
roll, pitch, and height: `u = [ϕ_ref, θ_ref, z_ref]`. The yaw reference is
fixed to zero and a simple PD controller generates thrust and torques.

# Keyword Arguments
- `t=0.1`: discretization time step.
- `dt=1f-3`: integration time step used inside the continuous rollouts.
- `kp_z=4.0`, `kd_z=2.5`: height PD gains.
- `kp_phi=4.0`, `kd_phi=1.5`: roll PD gains.
- `kp_theta=4.0`, `kd_theta=1.5`: pitch PD gains.
- `kp_psi=2.0`, `kd_psi=1.0`: yaw PD gains (reference is zero).
"""
function discrete_quadcopter_ref(; t::Real=0.1,
                                 dt::Real=1f-3,
                                 kp_z::Real=4.0, kd_z::Real=2.5,
                                 kp_phi::Real=4.0, kd_phi::Real=1.5,
                                 kp_theta::Real=4.0, kd_theta::Real=1.5,
                                 kp_psi::Real=2.0, kd_psi::Real=1.0)
    X_center = zeros(Float32, 12)
    X_radius = Float32.([
        100.0, 100.0, 100.0,  # x, y, z
        100.0, 100.0, 100.0,  # vx, vy, vz
        1.0, 1.0, 1.0,        # phi, theta, psi
        5.0, 5.0, 5.0         # p, q, r
    ])
    X = Hyperrectangle(X_center, X_radius)

    U_center = zeros(Float32, 3)
    U_radius = Float32.([1.0, 1.0, 100.0]) # phi_ref, theta_ref, z_ref
    U = Hyperrectangle(U_center, U_radius)
    V = Hyperrectangle(U_center, U_radius)
    W = Hyperrectangle(zeros(Float32, 12), zeros(Float32, 12))

    field = (x, u, w) -> begin
        x_vec = Float64.(x)
        u_ref = Float64.(u)

        phi_ref = u_ref[1]
        theta_ref = u_ref[2]
        z_ref = u_ref[3]

        x3 = x_vec[3]
        x6 = x_vec[6]
        x7 = x_vec[7]
        x8 = x_vec[8]
        x9 = x_vec[9]
        x10 = x_vec[10]
        x11 = x_vec[11]
        x12 = x_vec[12]

        m = 1.5
        g = 9.81

        T = m * (g + kp_z * (z_ref - x3) + kd_z * (0.0 - x6))
        tau_phi = kp_phi * (phi_ref - x7) + kd_phi * (0.0 - x10)
        tau_theta = kp_theta * (theta_ref - x8) + kd_theta * (0.0 - x11)
        tau_psi = kp_psi * (0.0 - x9) + kd_psi * (0.0 - x12)

        return quadfield(x_vec, [T, tau_phi, tau_theta, tau_psi], w)
    end

    cs = ContinuousSystem(X, U, W, field)
    κ = (x, v, t) -> v
    return DiscreteRandomSystem(cs, V, κ, t; dt=dt)
end

















################# end module

module SurfaceBoat

"""
    boat_dynamics(x::Vector{Float64}, u::Vector{Float64}, w::Vector{Float64}=zeros(7))

Nonlinear continuous-time dynamics for a planar surface water vehicle (boat).

# State vector (x ∈ ℝ⁷)
- x[1] = x-position (m)
- x[2] = y-position (m)
- x[3] = yaw angle ψ (rad)
- x[4] = surge velocity u (m/s)
- x[5] = sway velocity v (m/s)
- x[6] = yaw rate r = dψ/dt (rad/s)
- x[7] = rudder angle δ (rad)

# Control input (u ∈ ℝ²)
- u[1] = thrust force (N)
- u[2] = rudder angular velocity (rad/s)

# Disturbance vector (w ∈ ℝ⁷) — optional additive disturbance on each state.

# Returns
- dx::Vector{Float64} — Time derivative of the state.
"""
function boat_dynamics(x::Vector{Float64}, u::Vector{Float64}, w::Vector{Float64}=zeros(7))
    # Parameters
    m = 50.0         # mass [kg]
    Izz = 10.0       # moment of inertia about z-axis [kg·m^2]
    L = 2.0          # distance from center to rudder [m]

    surge_drag = 5.0    # surge drag coefficient
    sway_drag = 8.0     # sway drag coefficient
    yaw_drag = 2.0      # yaw drag coefficient
    rudder_coeff = 1.5  # lateral force multiplier from rudder deflection

    # Extract state variables
    ψ  = x[3]  # yaw
    u_ = x[4]  # surge
    v  = x[5]  # sway
    r  = x[6]  # yaw rate
    δ  = x[7]  # rudder angle

    # Extract inputs
    T = u[1]    # thrust
    dδ = u[2]   # rudder angular velocity

    dx = zeros(7)

    # Position kinematics (in inertial frame)
    dx[1] = u_ * cos(ψ) - v * sin(ψ)
    dx[2] = u_ * sin(ψ) + v * cos(ψ)

    # Yaw angle rate
    dx[3] = r

    # Surge dynamics (thrust minus drag)
    dx[4] = (T - surge_drag * u_) / m

    # Sway dynamics (rudder-induced lateral force minus drag)
    lateral_force = rudder_coeff * T * sin(δ)
    dx[5] = (lateral_force - sway_drag * v) / m

    # Yaw rate dynamics (rudder moment)
    rudder_moment = L * lateral_force
    dx[6] = (rudder_moment - yaw_drag * r) / Izz

    # Rudder dynamics
    dx[7] = dδ

    return dx + w
end

end # module
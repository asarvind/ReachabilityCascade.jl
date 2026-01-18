using LazySets
using ..ControlSystem: ContinuousSystem, DiscreteRandomSystem

"""
    robot3dof_field(x, u; kwargs...) -> dx

Continuous-time dynamics for a planar 3-link robot arm with torque inputs.

State `x` is joint angles and velocities:
`x = [q1, q2, q3, q1dot, q2dot, q3dot]`.
Input `u` is joint torques:
`u = [τ1, τ2, τ3]`.

The model is horizontal (gravity off by default) and has no damping. Coriolis
terms can be included via finite-difference of the mass matrix.

# Keyword Arguments
- `link_lengths=[0.5, 0.4, 0.3]`: link lengths (m).
- `link_masses=[2.0, 1.5, 1.0]`: link masses (kg).
- `com_offsets=nothing`: COM offsets along each link (m). Defaults to `length/2`.
- `link_inertias=nothing`: planar inertias about each link COM (kg·m^2).
  Defaults to `m*l^2/12` for each link.
- `coriolis=true`: include Coriolis/centrifugal terms.
- `fd_eps=1e-6`: finite-difference step for Coriolis term.
- `w=zeros(6)`: optional additive disturbance on each state derivative.

# Returns
- `dx`: time derivative of the state (length 6).
"""
function robot3dof_field(x, u;
                         link_lengths=[0.5, 0.4, 0.3],
                         link_masses=[2.0, 1.5, 1.0],
                         com_offsets=nothing,
                         link_inertias=nothing,
                         coriolis::Bool=true,
                         fd_eps::Real=1e-6,
                         w=zeros(6))
    length(x) == 6 || throw(DimensionMismatch("state x must have length 6; got length=$(length(x))"))
    length(u) >= 3 || throw(DimensionMismatch("input u must have length at least 3; got length=$(length(u))"))

    q = Float32.(x[1:3])
    qd = Float32.(x[4:6])
    τ = Float32.(u[1:3])

    l = Float32.(link_lengths)
    m = Float32.(link_masses)
    length(l) == 3 || throw(DimensionMismatch("link_lengths must have length 3"))
    length(m) == 3 || throw(DimensionMismatch("link_masses must have length 3"))

    c = com_offsets === nothing ? l ./ 2 : Float32.(com_offsets)
    length(c) == 3 || throw(DimensionMismatch("com_offsets must have length 3"))

    I = if link_inertias === nothing
        (m .* l .* l) ./ 12
    else
        Float32.(link_inertias)
    end
    length(I) == 3 || throw(DimensionMismatch("link_inertias must have length 3"))

    M = _mass_matrix(q, l, m, c, I)
    c_term = coriolis ? _coriolis_term(q, qd, l, m, c, I, fd_eps) : zeros(3)

    qdd = M \ (τ - c_term)
    dx = vcat(qd, qdd) .+ Float32.(w)
    return dx
end

"""
    robot3dof_accel_field(x, u; w=zeros(6)) -> dx

Acceleration-controlled model for the planar 3-link arm.

State `x = [q1, q2, q3, q1dot, q2dot, q3dot]` and input `u = [q1dd, q2dd, q3dd]`.
"""
function robot3dof_accel_field(x, u; w=zeros(Float32, 6))
    length(x) == 6 || throw(DimensionMismatch("state x must have length 6; got length=$(length(x))"))
    length(u) >= 3 || throw(DimensionMismatch("input u must have length at least 3; got length=$(length(u))"))

    qd = Float32.(x[4:6])
    qdd = Float32.(u[1:3])
    dx = vcat(qd, qdd) .+ Float32.(w)
    return dx
end

"""
    discrete_robot3dof(; kwargs...) -> DiscreteRandomSystem

Build a discrete-time system for the planar 3-link arm using the continuous dynamics.

# Keyword Arguments
- `t=0.1`: discretization time step.
- `dt=1f-3`: integration time step used inside the continuous rollouts.
- `link_lengths=[0.5, 0.4, 0.3]`: link lengths (m).
- `link_masses=[2.0, 1.5, 1.0]`: link masses (kg).
- `com_offsets=nothing`: COM offsets along each link (m). Defaults to `length/2`.
- `link_inertias=nothing`: planar inertias about each link COM (kg·m^2).
  Defaults to `m*l^2/12` for each link.
- `coriolis=true`: include Coriolis/centrifugal terms.
- `fd_eps=1e-6`: finite-difference step for Coriolis term.
"""
function discrete_robot3dof(; t::Real=0.1,
                            dt::Real=1f-3)
    X = Hyperrectangle(zeros(Float32, 6), vcat(fill(Float32(pi / 2), 3), fill(2.0f0, 3)))
    U = Hyperrectangle(zeros(Float32, 3), fill(10.0f0, 3))
    V = Hyperrectangle(zeros(Float32, 3), fill(10.0f0, 3))
    W = Hyperrectangle(zeros(Float32, 6), zeros(Float32, 6))

    field = (x, u, w) -> robot3dof_accel_field(x, u; w=w)

    cs = ContinuousSystem(X, U, W, field)
    κ = (x, v, t) -> v
    return DiscreteRandomSystem(cs, V, κ, t; dt=dt)
end

"""
    joint_positions(x; link_lengths=[0.5, 0.4, 0.3]) -> positions

Compute planar joint positions for a 3-link arm given the joint angles in `x`.

The base is fixed at the origin and is not included. The output columns are:
`p1, p2, p3` corresponding to the end of link 1, 2, and 3.

# Arguments
- `x`: state vector with joint angles in `x[1:3]`.

# Keyword Arguments
- `link_lengths=[0.5, 0.4, 0.3]`: link lengths (m).

# Returns
- `positions`: `2×3` matrix, columns are joint positions in the plane.
"""
function joint_positions(x; link_lengths=[0.5, 0.4, 0.3])
    length(x) >= 3 || throw(DimensionMismatch("state x must have at least 3 elements; got length=$(length(x))"))
    l = Float32.(link_lengths)
    length(l) == 3 || throw(DimensionMismatch("link_lengths must have length 3"))

    q1, q2, q3 = Float32.(x[1:3])
    θ1 = q1
    θ2 = q1 + q2
    θ3 = q1 + q2 + q3

    p1 = [l[1] * cos(θ1), l[1] * sin(θ1)]
    p2 = [l[1] * cos(θ1) + l[2] * cos(θ2),
          l[1] * sin(θ1) + l[2] * sin(θ2)]
    p3 = [l[1] * cos(θ1) + l[2] * cos(θ2) + l[3] * cos(θ3),
          l[1] * sin(θ1) + l[2] * sin(θ2) + l[3] * sin(θ3)]

    return hcat(p1, p2, p3)
end

function _mass_matrix(q, l, m, c, I)
    q1, q2, q3 = q
    θ1 = q1
    θ2 = q1 + q2
    θ3 = q1 + q2 + q3

    # COM positions
    p1 = [c[1] * cos(θ1), c[1] * sin(θ1)]
    p2 = [l[1] * cos(θ1) + c[2] * cos(θ2),
          l[1] * sin(θ1) + c[2] * sin(θ2)]
    p3 = [l[1] * cos(θ1) + l[2] * cos(θ2) + c[3] * cos(θ3),
          l[1] * sin(θ1) + l[2] * sin(θ2) + c[3] * sin(θ3)]
    _ = (p1, p2, p3) # keep for clarity; positions used via Jacobians below.

    # Linear velocity Jacobians
    Jv1 = Float32[
        -c[1] * sin(θ1)  0.0f0  0.0f0;
         c[1] * cos(θ1)  0.0f0  0.0f0
    ]
    Jv2 = Float32[
        -l[1] * sin(θ1) - c[2] * sin(θ2)  -c[2] * sin(θ2)  0.0f0;
         l[1] * cos(θ1) + c[2] * cos(θ2)   c[2] * cos(θ2)  0.0f0
    ]
    Jv3 = Float32[
        -l[1] * sin(θ1) - l[2] * sin(θ2) - c[3] * sin(θ3)
        -l[2] * sin(θ2) - c[3] * sin(θ3)
        -c[3] * sin(θ3);
         l[1] * cos(θ1) + l[2] * cos(θ2) + c[3] * cos(θ3)
         l[2] * cos(θ2) + c[3] * cos(θ3)
         c[3] * cos(θ3)
    ]
    Jv3 = reshape(Jv3, 2, 3)

    # Angular velocity Jacobians (planar)
    Jw1 = Float32[1.0, 0.0, 0.0]
    Jw2 = Float32[1.0, 1.0, 0.0]
    Jw3 = Float32[1.0, 1.0, 1.0]

    M = zeros(Float32, 3, 3)
    M .+= m[1] .* (Jv1' * Jv1) .+ I[1] .* (Jw1' * Jw1)
    M .+= m[2] .* (Jv2' * Jv2) .+ I[2] .* (Jw2' * Jw2)
    M .+= m[3] .* (Jv3' * Jv3) .+ I[3] .* (Jw3' * Jw3)
    return M
end

function _coriolis_term(q, qd, l, m, c, I, fd_eps)
    n = 3
    M0 = _mass_matrix(q, l, m, c, I)
    dM = Vector{Matrix{Float32}}(undef, n)
    for k in 1:n
        dq = zeros(Float32, n)
        dq[k] = fd_eps
        M_plus = _mass_matrix(q .+ dq, l, m, c, I)
        M_minus = _mass_matrix(q .- dq, l, m, c, I)
        dM[k] = (M_plus .- M_minus) ./ (2 * fd_eps)
    end

    c_term = zeros(Float32, n)
    for k in 1:n
        acc = 0.0f0
        for i in 1:n
            for j in 1:n
                acc += 0.5f0 * (dM[i][k, j] + dM[j][k, i] - dM[k][i, j]) * qd[i] * qd[j]
            end
        end
        c_term[k] = acc
    end
    return c_term
end

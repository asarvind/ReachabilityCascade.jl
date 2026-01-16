module Robot3DOF

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

    q = Float64.(x[1:3])
    qd = Float64.(x[4:6])
    τ = Float64.(u[1:3])

    l = Float64.(link_lengths)
    m = Float64.(link_masses)
    length(l) == 3 || throw(DimensionMismatch("link_lengths must have length 3"))
    length(m) == 3 || throw(DimensionMismatch("link_masses must have length 3"))

    c = com_offsets === nothing ? l ./ 2 : Float64.(com_offsets)
    length(c) == 3 || throw(DimensionMismatch("com_offsets must have length 3"))

    I = if link_inertias === nothing
        (m .* l .* l) ./ 12
    else
        Float64.(link_inertias)
    end
    length(I) == 3 || throw(DimensionMismatch("link_inertias must have length 3"))

    M = _mass_matrix(q, l, m, c, I)
    c_term = coriolis ? _coriolis_term(q, qd, l, m, c, I, fd_eps) : zeros(3)

    qdd = M \ (τ - c_term)
    dx = vcat(qd, qdd) .+ Float64.(w)
    return dx
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
    l = Float64.(link_lengths)
    length(l) == 3 || throw(DimensionMismatch("link_lengths must have length 3"))

    q1, q2, q3 = Float64.(x[1:3])
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
    Jv1 = [ -c[1] * sin(θ1)  0.0  0.0;
             c[1] * cos(θ1)  0.0  0.0 ]
    Jv2 = [ -l[1] * sin(θ1) - c[2] * sin(θ2)  -c[2] * sin(θ2)  0.0;
             l[1] * cos(θ1) + c[2] * cos(θ2)   c[2] * cos(θ2)  0.0 ]
    Jv3 = [ -l[1] * sin(θ1) - l[2] * sin(θ2) - c[3] * sin(θ3)
            -l[2] * sin(θ2) - c[3] * sin(θ3)
            -c[3] * sin(θ3);
             l[1] * cos(θ1) + l[2] * cos(θ2) + c[3] * cos(θ3)
             l[2] * cos(θ2) + c[3] * cos(θ3)
             c[3] * cos(θ3) ]
    Jv3 = reshape(Jv3, 2, 3)

    # Angular velocity Jacobians (planar)
    Jw1 = [1.0, 0.0, 0.0]
    Jw2 = [1.0, 1.0, 0.0]
    Jw3 = [1.0, 1.0, 1.0]

    M = zeros(3, 3)
    M .+= m[1] .* (Jv1' * Jv1) .+ I[1] .* (Jw1' * Jw1)
    M .+= m[2] .* (Jv2' * Jv2) .+ I[2] .* (Jw2' * Jw2)
    M .+= m[3] .* (Jv3' * Jv3) .+ I[3] .* (Jw3' * Jw3)
    return M
end

function _coriolis_term(q, qd, l, m, c, I, fd_eps)
    n = 3
    M0 = _mass_matrix(q, l, m, c, I)
    dM = Vector{Matrix{Float64}}(undef, n)
    for k in 1:n
        dq = zeros(n)
        dq[k] = fd_eps
        M_plus = _mass_matrix(q .+ dq, l, m, c, I)
        M_minus = _mass_matrix(q .- dq, l, m, c, I)
        dM[k] = (M_plus .- M_minus) ./ (2 * fd_eps)
    end

    c_term = zeros(n)
    for k in 1:n
        acc = 0.0
        for i in 1:n
            for j in 1:n
                acc += 0.5 * (dM[i][k, j] + dM[j][k, i] - dM[k][i, j]) * qd[i] * qd[j]
            end
        end
        c_term[k] = acc
    end
    return c_term
end

end # module Robot3DOF

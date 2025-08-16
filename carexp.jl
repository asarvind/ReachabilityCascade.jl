### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ fc3c2db6-7193-11f0-1b65-7f390e18200d
begin
using Pkg
	
link_path = joinpath(pwd(), "SymlinkReachabilityCascade")
if islink(link_path)
    rm(link_path)
elseif ispath(link_path)
    error("A non-symlink file or directory already exists at $link_path. Aborting.")
end
symlink(pwd(), link_path)

Pkg.activate("SymlinkReachabilityCascade")
end

# ╔═╡ c8d1f042-ad49-4971-aa32-b9c15d241da6
using ReachabilityCascade, LinearAlgebra, Random, LazySets, Flux, Statistics, JuMP, OSQP, SparseArrays, HiGHS, Clarabel

# ╔═╡ b08a9444-65f3-45fa-825a-94ce96bef202
import NLopt

# ╔═╡ c6105657-7697-462a-84f0-5b6cb0ffb4b1
import ReachabilityCascade.CarDynamics as CD

# ╔═╡ 87dc46a4-8f00-473a-9e37-817fc91e7ffe
function discrete_car(t::Real)
	X = Hyperrectangle(
		vcat([50, 3.5, 0.0, 10.0], zeros(3)),
		[100, 3.5, 1.0, 10.0, 1.0, 1.0, 0.2]
	)

	U = Hyperrectangle(
		zeros(2), [0.4, 10.0]
	)

	V = Hyperrectangle(
		zeros(2), [1.0, 10.0]
	)

	cs = ContinuousSystem(X, U, CD.carfield)
	
	κ = (x, u, t) -> [0.4*(u[1] - x[3]), u[2]]

	return DiscreteRandomSystem(cs, V, κ, t)	
end

# ╔═╡ 48f1de66-e038-42db-8044-b5f83c86eacf
function discrete_vehicles(t::Real)

	X = Hyperrectangle(
		vcat([50, 3.5, 0.0, 10.0], zeros(3), 50.0, 1.75, 5.0, 50.0, 5.25, -10.0),
		[100, 3.5, 1.0, 10.0, 1.0, 1.0, 0.2, 100.0, 0.1, 1.0, 100.0, 0.1, 1.0]
	)	

	V = Hyperrectangle(
		zeros(2), [1.0, 10.0]
	)

	function vehicle_transition(x::AbstractVector, u::AbstractVector)
		
		ds = discrete_car(t)
		ego_next = ds(x[1:7], u)
	
		x8next = x[8] + t*x[10]
		x11next = x[11] + t*x[13]
	
		xnext = vcat(ego_next, x8next, x[9:10], x11next, x[12:13])
	
		return xnext
	end	 

	return DiscreteRandomSystem(X, V, vehicle_transition)

end

# ╔═╡ 3ec5c7b7-cae2-45a8-83ce-2c77993ae9fe
function safety_fn(x::AbstractVector)
	safety_distance = 2.5

	ds = discrete_car(0.25)

	fol_dist = norm(x[1:2] - x[8:9]) - safety_distance
	on_dist = norm(x[1:2] - x[11:12]) - safety_distance

	constraint_dist = 1.0 .- abs.(x[1:7] .- ds.X.center) ./ (ds.X.radius .+ 1e-8)

	return vcat(fol_dist, on_dist, constraint_dist)
end

# ╔═╡ 73693311-fab8-4ff7-8c11-1814c5873283
function target_fn(x::AbstractVector)
	return [x[1] - x[8]]
end

# ╔═╡ 52218b79-4396-45f6-a06d-c0e8e30e7cb0
function intertarget_fn(x::AbstractVector)
	return x[1:7]
end

# ╔═╡ 77293f6a-0141-4fa0-81da-5a74e35b2537
function linearize(ds::DiscreteRandomSystem, x::AbstractVector, u::AbstractVector=ds.U.center; δ=0.001)
	n = size(x, 1)
	m = size(u, 1)

	A = ones(n, n)
	B = ones(n, m)

	p = ds(x, u)

	for i in 1:n
		x_pert = copy(x)
		x_pert[i] += δ
		A[:, i] = (ds(x_pert, u) - p)/δ		
	end

	for i in 1:m
		u_pert = copy(u)
		u_pert[i] += δ
		B[:, i] = (ds(x, u_pert) - p)/δ
	end

	c = p - A*x - B*u 

	return A, B, c
end

# ╔═╡ 7bc0465a-d505-4f6f-8dd5-cf614c36ebd4
# Discrete-time LQR + Lyapunov SDP using Clarabel
# Single-call API: lqr_lyap(A,B,Q,R) -> (K, S, info)
# - Computes infinite-horizon discrete LQR gain K for (A,B,Q,R)
# - Synthesizes a *quadratic Lyapunov matrix* S ≻ 0 for Acl = A - B*K via an SDP
#   using Clarabel with true semidefinite constraints.
#
# Lyapunov SDP (discrete-time): find S ≻ 0 s.t.
#   Acl' S Acl - S + Qℓ ≼ 0   ⇔   S - Acl' S Acl - Qℓ ⪰ 0
# We also impose S ⪰ ε I for numerical robustness.
# Objective: minimize trace(S) (well-conditioned smallest certificate).
#

"""
    lqr_lyap(A, B, Q, R; ε=1e-8, Qlyap=nothing, tol=1e-10, maxiter=10_000, min_reg=1e-12, verbose=false)

Compute the discrete-time infinite-horizon LQR gain `K` for `(A,B,Q,R)` and
synthesize a quadratic Lyapunov matrix `S` for the closed loop `Acl = A - B*K`
by solving a semidefinite program (SDP) with Clarabel.

Lyapunov condition (discrete):
    Acl' * S * Acl - S + Qℓ ≼ 0,   S ⪰ ε I,
with objective `min trace(S)`.

If `Qlyap` is not provided, we use the standard choice `Qℓ = Q + K' R K`.

# Arguments
- `A::AbstractMatrix` (n×n), `B::AbstractMatrix` (n×m)
- `Q::AbstractMatrix` (n×n, symmetric, ⪰ 0), `R::AbstractMatrix` (m×m, symmetric, ⪰ 0)

# Keywords
- `ε`     : minimal eigenvalue margin for S (default 1e-8)
- `Qlyap` : custom Qℓ in the Lyapunov inequality; defaults to `Q + K' R K`
- `tol`/`maxiter`/`min_reg` : controls for internal Riccati iteration used to get K
- `verbose`: print solver output

# Returns
- `K::Matrix{Float64}` : LQR gain (m×n)
- `S::Matrix{Float64}` : Lyapunov matrix (n×n), SPD certificate for Acl
- `info::NamedTuple`   : diagnostics (`riccati_iterations`, `riccati_converged`,
                         `spectral_radius`, `sdp_status`, `trace_S`)
"""
function lqr_lyap(A::AbstractMatrix, B::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix;
                  ε::Real=1e-8, Qlyap=nothing, tol::Real=1e-10, maxiter::Integer=10_000,
                  min_reg::Real=1e-12, verbose::Bool=false)
    # --- dimensions & types
    n, nA = size(A); nB, m = size(B)
    @assert n == nA "A must be square (n×n)"
    @assert nB == n "B must be n×m with same n as A"
    @assert size(Q) == (n, n) "Q must be n×n"
    @assert size(R) == (m, m) "R must be m×m"

    As = Array{Float64}(A); Bs = Array{Float64}(B)
    Qs = Symmetric(0.5 .* (Array{Float64}(Q) + Array{Float64}(Q)'))
    Rs = Symmetric(0.5 .* (Array{Float64}(R) + Array{Float64}(R)'))

    # --- Internal LQR gain via fixed-point DARE iteration (robust, no external deps)
    function lqr_gain(As, Bs, Qs, Rs; tol=tol, maxiter=maxiter, min_reg=min_reg)
        Rreg = Matrix(Rs); reg = 0.0
        # ensure Rreg ≻ 0 for numerical stability
        let reg_local = 0.0
            for _ in 1:6
                try
                    cholesky(Hermitian(Rreg + reg_local*I)); reg = reg_local; break
                catch; reg_local = max(10.0 * max(reg_local, float(min_reg)), float(min_reg)); end
            end
            Rreg += reg*I
        end
        P = Matrix(Qs); it = 0
        while it < maxiter
            it += 1
            G = Symmetric(Rreg + Bs' * P * Bs)
            rhs = Bs' * P * As
            K = try cholesky(Hermitian(Matrix(G))) \ rhs catch; G \ rhs end
            Pnext = Symmetric(0.5 .* ((Matrix(Qs) + As'*(P*As - P*Bs*K)) + (Matrix(Qs) + As'*(P*As - P*Bs*K))'))
            if opnorm(Pnext - P, Inf) <= tol * (1 + opnorm(P, Inf))
                P = Matrix(Pnext); break
            end
            P = Matrix(Pnext)
        end
        converged = it < maxiter
        # final K with unregularized R if possible
        Gfin = Symmetric(Matrix(Rs) + Bs' * P * Bs)
        K = try cholesky(Hermitian(Matrix(Gfin))) \ (Bs' * P * As) catch; Gfin \ (Bs' * P * As) end
        return K, it, converged
    end

    K, riters, rconv = lqr_gain(As, Bs, Qs, Rs)
    Acl = As - Bs * K
    Qℓ = Qlyap === nothing ? (Matrix(Qs) + K' * Matrix(Rs) * K) : Array{Float64}(Qlyap)

    # --- SDP for Lyapunov matrix S with Clarabel
    model = Model(Clarabel.Optimizer)
    if !verbose
        set_silent(model)
    end

    @variable(model, S[1:n, 1:n], PSD)    # S ⪰ 0 and symmetric

    # Enforce S ⪰ ε I  ⇔  S - ε I ⪰ 0
    I_n = Matrix{Float64}(I, n, n)
    @constraint(model, S - ε * I_n in PSDCone())

    # Lyapunov LMI:  Acl' S Acl - S + Qℓ ≼ 0  ⇔  S - Acl' S Acl - Qℓ ⪰ 0
    @constraint(model, (S - Acl' * S * Acl - Qℓ) in PSDCone())

    # Objective: minimize trace(S)
    @objective(model, Min, sum(S[i,i] for i in 1:n))

    optimize!(model)

    status = termination_status(model)
    Sval = value.(S)
    ρ = maximum(abs.(eigvals(Acl)))

    info = (riccati_iterations=riters,
            riccati_converged=rconv,
            spectral_radius=ρ,
            sdp_status=status,
            trace_S=sum(diag(Sval)))

    return K, Sval, info
end

# -----------------------------------------------------------------------------
# Example (uncomment to try):
# using Random
# Random.seed!(1)
# A = [1.0 0.1; 0.0 1.0]
# B = [0.0; 0.1]
# Q = I(2)
# R = reshape([0.01], 1, 1)
# K, S, info = lqr_lyap(A,B,Q,R; ε=1e-6, verbose=false)
# @show K, S, info
# Acl = A - B*K; Qℓ = Q + K' * R * K
# @show maximum(eigvals(Acl' * S * Acl - S + Qℓ))   # should be ≤ 0 (numerically ~ 0 or negative)
# -----------------------------------------------------------------------------

# ╔═╡ 68e143e5-fdca-4a68-963f-40f02609732b
function gen_trajectory(ds::DiscreteRandomSystem, x0::AbstractVector, T::Integer, H::AbstractMatrix, d::AbstractVector = zeros(size(H, 1)); start_xmat::Union{Nothing, AbstractMatrix}=nothing, start_umat::Union{Nothing, AbstractMatrix} = nothing, lin_x::AbstractVector = x0)
    # 1) Linearization (pick a nominal, here (x0, u_center))
    A, B, c = linearize(ds, lin_x, ds.U.center)  # <-- was `x`

    # 2) Model
    opt = Model(HiGHS.Optimizer)
	JuMP.set_silent(opt)

    # 3) Dimensions and bounds
    n = length(x0)
    m = length(ds.U.center)
    Xlo, Xhi = low(ds.X),  high(ds.X)   # vectors length n
    Ulo, Uhi = low(ds.U),  high(ds.U)   # vectors length m

    # 4) Decision variables
	if start_xmat != nothing
    	@variable(opt, Xlo[i] .<= x[i=1:n, j=1:(T+1)] .<= Xhi[i], start = start_xmat[i, j])
	else
		@variable(opt, Xlo[i] .<= x[i=1:n, j=1:(T+1)] .<= Xhi[i])
	end

	if start_umat != nothing
    	@variable(opt, Ulo[i] .<= u[i=1:m, j=1:T]   .<= Uhi[i], start = start_umat[i, j])
	else
		@variable(opt, Ulo[i] .<= u[i=1:m, j=1:T]   .<= Uhi[i])	
	end
    @variable(opt, u_abs[1:m, 1:T] .>= 0)

    # encode |u| via two linear inequalities: -u_abs <= u <= u_abs
    @constraint(opt, u .<= u_abs)
    @constraint(opt, -u_abs .<= u)

    # 5) Dynamics x_{t+1} = A x_t + B u_t + c
    # Broadcast c across columns 1..T
    Cmat = repeat(c, 1, T)
    @constraint(opt, A * x[:, 1:T] + B * u + Cmat .== x[:, 2:(T+1)])

    # 6) Initial state
    @constraint(opt, x[:, 1] .== x0)

    # 7) Target at final time
    @constraint(opt, H * x[:, T+1] .>= d)

    # 8) Collision avoidance (axis-aligned rectangular disjunctions)
    #    Front vehicle uses state indices (8,9); oncoming uses (11,12)
    #    At each t, enforce: (x1-x8 <= -5) or (x1-x8 >= 5) or (x2-x9 <= -2) or (x2-x9 >= 2)
    #    Same pattern w.r.t. (11,12).

    # # Compute tight difference bounds from X bounds
    Δx_min_f = Xlo[1] - Xhi[8];  Δx_max_f = Xhi[1] - Xlo[8]
    Δy_min_f = Xlo[2] - Xhi[9];  Δy_max_f = Xhi[2] - Xlo[9]

    Δx_min_o = Xlo[1] - Xhi[11]; Δx_max_o = Xhi[1] - Xlo[11]
    Δy_min_o = Xlo[2] - Xhi[12]; Δy_max_o = Xhi[2] - Xlo[12]

    # # Big-M values (front)
    M1f = Δx_max_f + 5      # for x1-x8 <= -5 + M*(1-b)
    M2f = 5 - Δx_min_f      # for x1-x8 >=  5 - M*(1-b)
    M3f = Δy_max_f + 2      # for x2-x9 <= -2 + M*(1-b)
    M4f = 2 - Δy_min_f      # for x2-x9 >=  2 - M*(1-b)

    # # Big-M values (oncoming)
    M1o = Δx_max_o + 5
    M2o = 5 - Δx_min_o
    M3o = Δy_max_o + 2
    M4o = 2 - Δy_min_o

    # # Front-vehicle binaries
    @variable(opt, bf[1:4, 1:T+1], Bin)
    @constraint(opt, [t=1:T+1], sum(bf[:, t]) >= 1)  # at least one active

    @constraint(opt, [t=1:T+1], x[1,t] - x[8,t] <= -5 + M1f*(1 - bf[1,t]))
    @constraint(opt, [t=1:T+1], x[1,t] - x[8,t] >=  5 - M2f*(1 - bf[2,t]))
    @constraint(opt, [t=1:T+1], x[2,t] - x[9,t] <= -2 + M3f*(1 - bf[3,t]))
    @constraint(opt, [t=1:T+1], x[2,t] - x[9,t] >=  2 - M4f*(1 - bf[4,t]))

    # # Oncoming-vehicle binaries
    @variable(opt, bo[1:4, 1:T+1], Bin)
    @constraint(opt, [t=1:T+1], sum(bo[:, t]) >= 1)

    @constraint(opt, [t=1:T+1], x[1,t] - x[11,t] <= -5 + M1o*(1 - bo[1,t]))
    @constraint(opt, [t=1:T+1], x[1,t] - x[11,t] >=  5 - M2o*(1 - bo[2,t]))
    @constraint(opt, [t=1:T+1], x[2,t] - x[12,t] <= -2 + M3o*(1 - bo[3,t]))
    @constraint(opt, [t=1:T+1], x[2,t] - x[12,t] >=  2 - M4o*(1 - bo[4,t]))

    # 9) Objective: keep inputs small (L1)
    @objective(opt, Min, sum(u_abs))
	# @objective(opt, Max, sum(H * x[:, T+1]))

    @time optimize!(opt)

    return is_solved_and_feasible(opt), value.(x), value.(u)
end

# ╔═╡ 871f4a19-ea87-4f59-b1b9-d8321b7852e8
function track_input(ds::DiscreteRandomSystem, x::AbstractVector, H::AbstractMatrix, target::AbstractVector, S::AbstractMatrix; u0::AbstractVector=ds.U.center, algo=:LN_COBYLA)
	# define objective
	function objfun(u::AbstractVector, grad::Union{Nothing, AbstractVector} = nothing)
		err = H*(ds(x, u) - target)
		err'*S*err
	end

	opt = NLopt.Opt(algo, 2)
	NLopt.lower_bounds!(opt, low(ds.U))
	NLopt.upper_bounds!(opt, high(ds.U))

	NLopt.min_objective!(opt, objfun)

	min_x, min_f, ret = NLopt.optimize(opt, u0)

	return min_x, min_f, objfun(u0), ret
end

# ╔═╡ 7674c735-b41f-4d6d-80f0-0b0bab237767
function mpc(ds::DiscreteRandomSystem, x0::AbstractVector, T::Integer, H::AbstractMatrix, d::AbstractVector = zeros(size(H, 1)); ds_sub::DiscreteRandomSystem = ds, T_sub::Integer=T)
	xmat = x0
	umat = Matrix(undef, length(ds.U.center), 0)

	xnew = copy(x0)

	for _ in 1:T_sub
		feasible, _, umat_pred = gen_trajectory(ds, xnew, T, H, lin_x=x0)
		if !feasible
			return xmat, umat
		end
		xnew = ds_sub(xnew, umat_pred[:, 1])
		xmat = hcat(xmat, copy(xnew))
		umat = hcat(umat, copy(umat_pred[:, 1]))
	end

	return xmat, umat
end

# ╔═╡ 7b3216d8-787e-4654-9827-7f303e42abff
let
	Random.seed!(0)
	
	context_dim = 13
	sample_dim = 13*4
	num_blocks = 3


	proj1 = [1 zeros(1, 6) -1 zeros(1, 5)]
	proj2 = [0 -1 zeros(1, 11)]
	proj3 = [0 1 zeros(1, 11)]
	proj = vcat(proj1, proj2, proj3)

	d = [0.0, -1.5, -1.5]
	

	ds = discrete_vehicles(0.25)
	x0 = copy(ds.X.center)
	x0[1] = 10.0
	x0[2] = 1.7
	x0[8] = 30
	x0[11] = 90

	ds_sub = discrete_vehicles(0.01)

	A, B, c = linearize(ds, x0, ds.U.center)

	Q = Matrix(1.0I, 7, 7)
	R = Matrix(1.0I, 2, 2)

	K, S, _ = lqr_lyap(A[1:7, 1:7], B[1:7, :], Q, R)

	

	T = 40
	# opt, xmat, umat = gen_trajectory(ds, x0, T, proj, d)

	# xmat

	# @time track_input(ds, xmat[:,1], Matrix(1.0I, 7, 13), xmat[:, 2], S; u0=umat[:,1], algo=:LN_NELDERMEAD)
end

# ╔═╡ Cell order:
# ╠═fc3c2db6-7193-11f0-1b65-7f390e18200d
# ╠═c8d1f042-ad49-4971-aa32-b9c15d241da6
# ╠═b08a9444-65f3-45fa-825a-94ce96bef202
# ╠═c6105657-7697-462a-84f0-5b6cb0ffb4b1
# ╠═87dc46a4-8f00-473a-9e37-817fc91e7ffe
# ╠═48f1de66-e038-42db-8044-b5f83c86eacf
# ╠═3ec5c7b7-cae2-45a8-83ce-2c77993ae9fe
# ╠═73693311-fab8-4ff7-8c11-1814c5873283
# ╠═52218b79-4396-45f6-a06d-c0e8e30e7cb0
# ╠═77293f6a-0141-4fa0-81da-5a74e35b2537
# ╠═7bc0465a-d505-4f6f-8dd5-cf614c36ebd4
# ╠═68e143e5-fdca-4a68-963f-40f02609732b
# ╠═871f4a19-ea87-4f59-b1b9-d8321b7852e8
# ╠═7674c735-b41f-4d6d-80f0-0b0bab237767
# ╠═7b3216d8-787e-4654-9827-7f303e42abff

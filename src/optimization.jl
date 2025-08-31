function linearize(ds::DiscreteRandomSystem, x::AbstractVector, u::AbstractVector=LazySets.center(ds.U); δ=0.001)
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

	return (A=A, B=B, c=c)
end


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

function correct_trajectory(ds::DiscreteRandomSystem, xmat::AbstractMatrix, umat::AbstractMatrix; Q::Union{Nothing, AbstractMatrix} = nothing, R::Union{Nothing, AbstractMatrix} = nothing)
	strj = copy(xmat)
	utrj = copy(umat)

	T = size(umat, 2)

	if Q == nothing
		Q = Matrix(1.0I, length(LazySets.center(ds.X)), length(LazySets.center(ds.X)))
	end
	if R == nothing
		R = Matrix(1.0I, length(LazySets.center(ds.U)), length(LazySets.center(ds.U)))
	end

	A, B, c = linearize(ds, xmat[:, 1], umat[:, 1])

	K, S, _ = lqr_lyap(A, B, Q, R)

	for i in 1:T
		utrj[:, i] += K*(xmat[:, i] - strj[:, i])
		strj[:, i+1] = ds(strj[:, i], utrj[:, i])		
	end

	return (state_trajectory=strj, input_trajectory=utrj)
end
function vehicle_field(x::AbstractVector, u::AbstractVector, w::Vector{<:Real}=zeros(7))
	dx_ego = carfield(x[1:7], u)
	dx_for_pos = x[10]
	dx_on_pos = x[13]

	X = Hyperrectangle(
		vcat([50, 4.0, 0.0, 10.0], zeros(3), 50.0, 1.75, 5.0, 50.0, 6.0, -5.0),
		[100, 3.0, 1.0, 10.0, 1.0, 1.0, 0.2, 100.0, 0.1, 1.0, 100.0, 0.1, 1.0]
	)	

	if x ∉ X
		dx_ego *= 0.0
	end

	if abs(x[8] - x[1]) < 5.0 && abs(x[9]- x[2]) < 2.0
		dx_for_pos = 0.0
		dx_ego *= 0.0
	end

	if abs(x[11] - x[1]) < 5.0 && abs(x[12]- x[2]) < 2.0
		dx_on_pos = 0.0
		dx_ego *= 0.0
	end

	return vcat(dx_ego, dx_for_pos, 0.0, 0.0, dx_on_pos, 0.0, 0.0)
end

function safe_discrete_vehicles(t::Real)
	κ = (x, u, t) -> [0.4*(u[1] - x[3]), u[2]]

	V = Hyperrectangle(
		zeros(2), [1.0, 10.0]
	)

	X = Hyperrectangle(
		vcat([50, 4.0, 0.0, 10.0], zeros(3), 50.0, 1.75, 5.0, 50.0, 6.0, -5.0),
		[100, 3.0, 1.0, 10.0, 1.0, 1.0, 0.2, 100.0, 0.1, 1.0, 100.0, 0.1, 1.0]
	)	

	U = Hyperrectangle(
		zeros(2), [0.4, 10.0]
	)

	cs = ContinuousSystem(X, U, vehicle_field)

	return DiscreteRandomSystem(cs, V, κ, t)
end

function discrete_car(t::Real; dt=0.001)
	X = Hyperrectangle(
		vcat([50, 4.0, 0.0, 10.0], zeros(3)),
		[100, 3.0, 1.0, 10.0, 1.0, 1.0, 0.2]
	)

	U = Hyperrectangle(
		zeros(2), [0.4, 10.0]
	)

	V = Hyperrectangle(
		zeros(2), [1.0, 10.0]
	)

	x_lo = X.center .- X.radius
	x_hi = X.center .+ X.radius
	u_lo = U.center .- U.radius
	u_hi = U.center .+ U.radius

	function carfield_clamped(x::Vector{<:Real}, u::Vector{<:Real}, w::Vector{<:Real}=zeros(7))
		x_clamped = clamp.(x, x_lo, x_hi)
		u_clamped = clamp.(u, u_lo, u_hi)
		return carfield(x_clamped, u_clamped, w)
	end

	cs = ContinuousSystem(X, U, carfield_clamped)
	
	κ = (x, u, t) -> [0.4*(u[1] - x[3]), u[2]]

	return DiscreteRandomSystem(cs, V, κ, t; dt=dt)	
end

function discrete_vehicles(t::Real; dt=0.001)

	function vehicle_transition(x::AbstractVector, u::AbstractVector)
		
		ds = discrete_car(t; dt=dt)
		ego_next = ds(x[1:7], u)
	
		x8next = x[8] + t*x[10]
		x11next = x[11] + t*x[13]
	
		xnext = vcat(ego_next, x8next, x[9:10], x11next, x[12:13])
	
		return xnext
	end	 

	X = Hyperrectangle(
		vcat([50, 4.0, 0.0, 10.0], zeros(3), 50.0, 1.75, 5.0, 50.0, 6.0, -5.0),
		[100, 3.0, 1.0, 10.0, 1.0, 1.0, 0.2, 100.0, 0.1, 1.0, 100.0, 0.1, 1.0]
	)	

	V = Hyperrectangle(
		zeros(2), [1.0, 10.0]
	)

	return DiscreteRandomSystem(X, V, vehicle_transition)
end

function discrete_car(t::Real)
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

	cs = ContinuousSystem(X, U, carfield)
	
	κ = (x, u, t) -> [0.4*(u[1] - x[3]), u[2]]

	return DiscreteRandomSystem(cs, V, κ, t)	
end

function discrete_vehicles(t::Real)

	function vehicle_transition(x::AbstractVector, u::AbstractVector)
		
		ds = discrete_car(t)
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
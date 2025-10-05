function train(::Type{TrajectoryEncoder}, data::AbstractVector, embdim::Integer; errorscale::Union{Nothing, AbstractVector{<:Real}}=nothing, optimiser = Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam()), epochs::Integer=1, savefile::String="", loadfile::String=savefile, saveperiod::Real=60, kwargs...)
	
	statedim = size(data[1].state_trajectory, 1)
	
	# construct network 
	if isfile(loadfile)
		ms = JLD2.load(loadfile, "modelstate")
		args = JLD2.load(loadfile, "args")
		this_kwargs = JLD2.load(loadfile, "kwargs")
		net = TrajectoryEncoder(args...; this_kwargs...)
		Flux.loadmodel!(net, ms)
	else
		args = (statedim, embdim)
		this_kwargs = kwargs
		net = TrajectoryEncoder(args...; this_kwargs...)
	end

	optstate = Flux.setup(optimiser, net)

	shufdata = shuffle(data)

	if isnothing(errorscale)
		es = Float32.(ones(statedim))
	else
		es = error_scale
	end

	startclock = time()

	# training epochs
	for _ in 1:epochs
		for d in shufdata 
			X = Float32.(d.state_trajectory)
			T = size(X, 2)
			
			batch_pairs = [
				(X[:, t], Δt, X[:, t+Δt])
				for t in 1:(T-1)
				for Δt in 1:(T - t)
			]
			
			x = reduce(hcat, Float32.(bp[1]) for bp in batch_pairs)
			tseq = reduce(vcat, Int32.(bp[2]) for bp in batch_pairs)
			y = reduce(hcat, Float32.(bp[3]) for bp in batch_pairs)
			
			# gradient
			grads = Flux.gradient(net) do model
				Flux.mse(predict_from_traj(model, X, x, tseq).*es, y.*es)
			end

			Flux.update!(optstate, net, grads[1])

			if !isempty(savefile) && time() - startclock > saveperiod
				JLD2.save(
					savefile,
					Dict(
						"modelstate"=>Flux.state(net),
						"args"=>args,
						"kwargs"=>kwargs
					)			
				)	
				startclock = time()
			end
		end
	end

	if !isempty(savefile)
		JLD2.save(
			savefile,
			Dict(
				"modelstate"=>Flux.state(net),
				"args"=>args,
				"kwargs"=>kwargs
			)			
		)
	end

	return net
end
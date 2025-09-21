function train(::Type{SequenceGenerator}, sampling_fn::Function, maxiter::Integer, dataargs::AbstractVector...; scale::Union{Nothing, Vector{<:Real}}=nothing, batch_size::Integer=200, flow_num::Integer = 5, optimizer = Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam()), save_period::Real=60, savefile::String="", kwargs...)

	# calculate sample and context dimensions
	tup = sampling_fn(dataargs[1][1].state_trajectory, dataargs[1][1].input_signal)
	if !isa(tup, NamedTuple) || !haskey(tup, :sample) || !haskey(tup, :context)
		throw(ArgumentError("The sampling function must return a NamedTuple with fields :sample and :context"))
	end
	
	smp, ctx = tup.sample, tup.context
	sample_dim = size(smp, 1)
	context_dim = size(ctx, 1)

	if isfile(savefile)
		kwargs_loaded = JLD2.load(savefile, "kwargs")
		sg = SequenceGenerator(; sample_dim=sample_dim, context_dim=context_dim, kwargs_loaded...)
		model_state = JLD2.load(savefile, "model_state")
		Flux.loadmodel!(sg, model_state)
	else
		sg = SequenceGenerator(; sample_dim=sample_dim, context_dim=context_dim, kwargs...)
	end
	opt_state = Flux.setup(optimizer, sg)

	numargs = length(dataargs)
	ctx_batches = [Matrix{Float32}(undef, context_dim, 0) for _ in 1:numargs]
	smp_batches = [Matrix{Float32}(undef, sample_dim, 0) for _ in 1:numargs]
	idx_latent = [Vector{Integer}(undef, 0) for _ in 1:numargs]

	iter = 0
	# shuffle data
	dataargs_shuffled = [shuffle(data) for data in dataargs]
	start_time = time()
	while iter < maxiter
		for i in 1:numargs
			data = dataargs_shuffled[i]
			idx = (iter ÷ length(data)) + 1
			strj, utrj = data[idx].state_trajectory, data[idx].input_signal
			tup = sampling_fn(strj, utrj)
			smp, ctx = tup.sample, tup.context
			ctx_batches[i] = hcat(ctx, ctx_batches[i])
			smp_batches[i] = hcat(smp, smp_batches[i])
			idx_latent[i] = vcat(fill(iter, size(ctx, 2)), idx_latent[i])
			latent = randn_by_id(idx_latent[i], sg.latent_dim)

			# compute scale
			if !isnothing(scale)
				sc = vcat(scale, ones(size(latent, 1) - length(scale)))
			else
				sc = ones(size(latent, 1))
			end

			# compute gradient
			flow_times = collect(1/flow_num:1/flow_num:1.0)
			grads = Flux.gradient(sg) do model
				v, v_ref = generator_velocity(model, smp_batches[i], ctx_batches[i], latent, flow_times)
				sum(abs.(v - v_ref).*sc)/size(latent, 2)
			end
			Flux.update!(opt_state, sg, grads[1])

			iter += 1

			bs = min(batch_size, size(ctx_batches[i], 2))
			ctx_batches[i] = ctx_batches[i][:, 1:bs]
			smp_batches[i] = smp_batches[i][:, 1:bs]
			idx_latent[i] = idx_latent[i][1:bs]

			# store model at periodic times 
			if time() - start_time > save_period && !isempty(savefile)
				model_state = Flux.state(sg)
				if isfile(savefile)
					kwargs_loaded = JLD2.load(savefile, "kwargs")
				else
					kwargs_loaded = kwargs
				end
				JLD2.save(
					savefile,
					Dict(
						"model_state"=>model_state,
						"kwargs"=>kwargs_loaded
					)
				)
				start_time = time()
			end
		end
	end
	
	if !isempty(savefile)
		if isfile(savefile)
			kwargs_loaded = JLD2.load(savefile, "kwargs")
		else
			kwargs_loaded = kwargs
		end
		model_state = Flux.state(sg)
		JLD2.save(
			savefile,
			Dict(
				"model_state"=>model_state, 
				"kwargs"=>kwargs_loaded
			)
		)
	end

	return sg
end
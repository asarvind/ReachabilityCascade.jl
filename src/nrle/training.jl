function train(::Type{NRLE}, prop_fun::Function, data::AbstractVector, time_stamps::Vector{<:Integer}; optimizer=Adam(0.001), max_batch_size::Integer = 100, kwargs...)	
	
	# construct the neural network and optimizer
	state_dim = size(data[1].state_trajectory, 1)
	prop_dim = size(prop_fun(data[1].state_trajectory, data[1].input_signal), 1)
	nrle = NRLE(state_dim, prop_dim; kwargs...)
	opt = Flux.setup(optimizer, nrle)
	
	# training iterations
	x0_batch = Matrix{Float32}(undef, state_dim, 0)
	xfin_batch = Matrix{Float32}(undef, state_dim, 0)
	prop_batch = Matrix{Float32}(undef, prop_dim, 0)

	l = length(time_stamps)
	
	for data_tup in shuffle(data)
		strj, utrj = data_tup.state_trajectory, data_tup.input_signal
		
		tseq = [rand(time_stamps[i]:time_stamps[i+1]) for i in 1:(length(time_stamps)-1)]
		
		rand_batch = [
		    ( strj[:, tseq[i]],
		      strj[:, tseq[j] + 1],
		      prop_fun(strj[:, tseq[i]:(tseq[j] + 1)], utrj[:, tseq[i]:tseq[j]]) )
		    for i in 1:(l-2)
		    for j in i:(l-1)
		]
		
		det_batch = [
		    ( strj[:, time_stamps[i]],
		      strj[:, time_stamps[j] + 1],
		      prop_fun(strj[:, time_stamps[i]:(time_stamps[j] + 1)], utrj[:, time_stamps[i]:time_stamps[j]]) )
		    for i in 1:(l-1)
		    for j in i:l
		]
		
		x0_batch = hcat(
			x0_batch, 
			reduce(hcat, [tup[1] for tup in rand_batch]),
			reduce(hcat, [tup[1] for tup in det_batch])
		)
		
		xfin_batch = hcat(
			xfin_batch, 
			reduce(hcat, [tup[2] for tup in rand_batch]),
			reduce(hcat, [tup[2] for tup in det_batch])
		)
		
		prop_batch = hcat(
			prop_batch, 
			reduce(hcat, [tup[3] for tup in rand_batch]),
			reduce(hcat, [tup[3] for tup in det_batch])
		)
		
		grads = Flux.gradient(nrle) do (this_net)
			_, ll = encode(this_net, x0_batch, xfin_batch, prop_batch)
			-sum(ll)/length(ll)  # return negative of log-likelihood for maximization
		end

		Flux.update!(opt, nrle, grads[1])

		_, ll = encode(nrle, x0_batch, xfin_batch, prop_batch)
		bnum = min(size(x0_batch, 2), max_batch_size)
		sorted_inds = sortperm(ll)[1:bnum]
		x0_batch = x0_batch[:, sorted_inds]
		xfin_batch = xfin_batch[:, sorted_inds]
		prop_batch = prop_batch[:, sorted_inds]
	end

	return nrle
end
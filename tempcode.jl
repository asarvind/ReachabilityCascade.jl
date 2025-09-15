function trenc(strj::AbstractMatrix{<:Real}, utrj::AbstractMatrix{<:Real})
    # compute vehicle positions at different time steps and append time
    # time steps
    t_max = size(strj, 2)
    divs = [1, 2, 4, 8, 16, 32]
    t_seq = [max(t_max÷i, 2) for i in divs]
    # combined positions
    cpos = vcat(vec(strj[1:2, t_seq]'), t_seq)

    # arguments for minimum safety of the state 
    ds = discrete_vehicles(0.25)
    # rectangular safety 
    rect_margin = vec(minimum(min.(high(ds.X) .- strj, strj .- low(ds.X)), dims=2))[2:7]
    # forward vehicle safety 
    rel_for = max.(abs.(strj[1, :] - strj[8, :]) .- 5.0, strj[2, :] - strj[9, :] .- 2.0)
    rel_on = max.(abs.(strj[1, :] - strj[11, :]) .- 5.0, strj[12, :] - strj[2, :] .- 2.0)
    obspos = vcat(rel_for[findmin(rel_for)[2]], rel_on[findmin(rel_on)[2]])
    
    return vcat(t_seq, cpos, rect_margin, obspos, utrj[:, 1])*0.1
end

function context_car(strj::AbstractMatrix{<:Real}, utrj::AbstractMatrix{<:Real})
    return vcat( strj[:, 1], strj[1, end] - strj[8, end] )
end

function train(
    ::Type{ConditionalFlow}, 
    data::AbstractVector, 
    context_fn::Function, 
    prop_fn::Function;
    epochs::Integer = 1,
    optimizer=Adam(0.001),
    max_batch_size::Integer = 100,
    savefile::String="", 
    loadfile::String=savefile,
    sampling_time::Integer = 1,
    kwargs...
)

	if isfile(loadfile)
		model_state = JLD2.load(loadfile, "model_state")
		kwargs = JLD2.load(loadfile, "kwargs")
	end

	# construct the neural network and optimizer
	ctx_dim = length(context_fn(data[1].state_trajectory, data[1].input_signal))    
	prop_dim = length(prop_fn(data[1].state_trajectory, data[1].input_signal))
	flow = ConditionalFlow(prop_dim, ctx_dim; kwargs...)
	opt = Flux.setup(optimizer, flow)
    
    # load pretrained weights
	if isfile(loadfile)
		Flux.loadmodel!(flow, model_state)
	end

    # initialize batch 
    ctx_batch = Matrix{Float32}(undef, ctx_dim, 0)
    prop_batch = Matrix{Float32}(undef, prop_dim, 0)

    start_time = time()

    for _ in 1:epochs
        for data_tup in shuffle(data)
            strj, utrj = data_tup.state_trajectory, data_tup.input_signal

            l = size(utrj, 2)
            tseq = [
                rand(τ:1:min(τ+sampling_time-1, l)) for τ in 1:sampling_time:l
            ]

            new_ctx_batch = reduce(hcat, [context_fn(strj[:, t:end], utrj[:, t:end]) for t in tseq])

            new_prop_batch = reduce(hcat, [prop_fn(strj[:, t:end], utrj[:, t:end]) for t in tseq])

            ctx_batch = hcat(ctx_batch, new_ctx_batch)
            prop_batch = hcat(prop_batch, new_prop_batch)

			grads = Flux.gradient(flow) do (this_net)
				z, ld = this_net(prop_batch, ctx_batch, inverse=false)
				-1*( sum(ld) -  sum(0.5*z.^2)/2 )/ length(ld) # return negative of log-likelihood upto constant difference for maximization
			end   
            
            Flux.update!(opt, flow, grads[1])

            z, ld = flow(prop_batch, ctx_batch, inverse=false)
            bnum = min(size(ctx_batch, 2), max_batch_size)
            sorted_idx = sortperm( ld - vec( sum( (0.5*z.^2)/2, dims=1) ) )[1:bnum]
            ctx_batch = ctx_batch[:, sorted_idx]
            prop_batch = prop_batch[:, sorted_idx]

			if time() - start_time > 60 && !isempty(savefile)
				model_state = Flux.state(flow)
				JLD2.save(savefile,
					Dict(
						"model_state" => model_state,
						"kwargs" => kwargs
					)
				)
				start_time = time()
			end
        end
    end

	if !isempty(savefile)
		model_state = Flux.state(flow)
		JLD2.save(savefile,
			Dict(
				"model_state" => model_state,
				"kwargs" => kwargs
			)
		)		
	end

    return flow
end

function load_flow(::Type{ConditionalFlow},loadfile::String, context_fn::Function, prop_fn::Function, data::AbstractVector)
	ctx_dim = length(context_fn(data[1].state_trajectory, data[1].input_signal))
	prop_dim = length(prop_fn(data[1].state_trajectory, data[1].input_signal))
	model_state = JLD2.load(loadfile, "model_state")
	kwargs = JLD2.load(loadfile, "kwargs")
	flow = ConditionalFlow(prop_dim, ctx_dim; kwargs...)
	Flux.loadmodel!(flow, model_state)

	return flow
end
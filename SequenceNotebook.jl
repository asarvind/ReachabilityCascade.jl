### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 6bb100cc-96dd-11f0-3815-357f2700d588
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

# ╔═╡ c681b310-5a90-454e-ac83-0140a2893232
begin
	import Flux
	using LinearAlgebra, Random
	import LazySets: Hyperrectangle, high, low, dim, radius_hyperrectangle, center
	import JLD2
	import ReachabilityCascade.CarDynamics: discrete_vehicles	
	import ReachabilityCascade: ConditionalFlow, loglikelihoods
end

# ╔═╡ 454e0c70-dd06-4859-8ef6-b40c1bbcc1d5
begin
# =================== Sequence Generator Structure ========================
struct SliceNet{M<:ConditionalFlow, N<:ConditionalFlow}
	state_dim::Integer 
	property_dim::Integer
	context_dim::Integer
	term::M
	bisect::N
end

Flux.@layer SliceNet

function SliceNet(state_dim::Integer, property_dim::Integer, context_dim::Integer; state_scale::Vector{<:Real} = ones(state_dim), property_scale::Vector{<:Real} = ones(property_dim), context_scale::Vector{<:Real} = ones(context_dim), kwargs...)
	
	c_scale = vcat(repeat(state_scale, 2), property_scale, context_scale)

	ctx_dim = 2*state_dim + property_dim + context_dim 

	term = ConditionalFlow(state_dim+property_dim+1, state_dim+context_dim; x_scaling=vcat(state_scale, property_scale, 1), c_scaling=vcat(state_scale, context_scale), kwargs...)

	bisect = ConditionalFlow(state_dim + 2*property_dim, ctx_dim; x_scaling=vcat(state_scale, property_scale, property_scale), c_scaling=c_scale, kwargs...)

	return SliceNet(state_dim, property_dim, context_dim, term, bisect)
end

function (sn::SliceNet)(states::AbstractMatrix{<:Real}, properties::AbstractMatrix{<:Real}, times::AbstractVector{<:Real}, context::AbstractVector{<:Real}, latent::AbstractMatrix{<:Real})
	@assert size(latent, 1) == sn.state_dim + 2*sn.property_dim  "number of rows in latent for filling states and properties should have appropriate dimension, i.e.; { size(latent, 1) == sn.state_dim + 2*sn.property_dim }"
	@assert size(states, 2) == length(times) "number of time points should equal the number of state columns"
	@assert size(properties, 2) == length(times) - 1 "number of properties should be equal to number of time intervals, i.e., one less than number of time points"
	@assert issorted(times)  "the time points have to be in an increasing order"
	@assert all(times .>= 1) "all time points have to be positive"
	
	periods = times[2:end] - times[1:(end-1)]
	id_fill = collect(1:(size(states, 2)-1))[periods .> 1] 

	if !isempty(id_fill)
		ctx = vcat(states[:, id_fill], states[:, id_fill .+ 1], properties[:, id_fill], repeat(context, 1, length(id_fill))) 
		lat_idx = rem.(times[id_fill], size(latent, 2)) .+ 1
		fill_cs = sn.bisect(latent[:, lat_idx], ctx; inverse=true)[1]
		fill_states = fill_cs[1:sn.state_dim, :]
		fill_props = fill_cs[(sn.state_dim+1):(sn.state_dim+sn.property_dim), :]
		modify_props = fill_cs[(sn.state_dim+sn.property_dim+1):end, :]	
		fill_times = (times[id_fill] + times[id_fill .+ 1]) .÷ 2
		this_st = hcat(states, fill_states)
		props = copy(properties)
		props[:, id_fill] = modify_props
		this_props = hcat(props, fill_props)
		this_tseq = vcat(times, fill_times)

		sortid_states = sortperm(this_tseq)
		sortid_props = sortperm(this_tseq[2:end])
		return this_st[:, sortid_states], this_props[:, sortid_props], this_tseq[sortid_states]
	else
		return states, properties, times 
	end
end

function (sn::SliceNet)(state::AbstractVector{<:Real}, context::AbstractVector{<:Real}, latent_fill::AbstractMatrix{<:Real}, latent_term::AbstractVector{<:Real})
	@assert size(latent_fill, 2) >= 2 "latent matrix for filling states should have at least 2 columns"
	@assert size(latent_term, 1) == sn.state_dim + sn.property_dim + 1 "latent vector for predicting terminal state and terminal time together should have rows equal to { state dimension plus 1 }"
	
	y = sn.term(latent_term, vcat(state, context); inverse=true)[1]
	states = hcat(state, y[1:sn.state_dim, :])
	properties = y[(sn.state_dim+1):(end-1), :]
	t = max(Int(round(y[end, 1])), 0) + 1
	times = [1, t]

	while size(states, 2) < t
		states, properties, times = sn(states, properties, times, context, latent_fill)
	end

	return states, times
end

# =====================================
end

# ╔═╡ 3a5df80f-0062-4d6b-b187-4efb1b1de950
function train(::Type{SliceNet}, property_fn::Function, dataargs::AbstractVector...; maxiter::Integer=max((length.(dataargs))...), batch_size::Integer=100, optimiser=Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam()), savefile::String="", loadfile::String="savefile", save_period::Real=60, kwargs...)
	# specify arguments for constructing slicenet
	if isfile(loadfile) # if arguments are already stored
		model_state = JLD2.load(loadfile, "model_state")
		this_args = JLD2.load(loadfile, "args")
		this_kwargs = JLD2.load(loadfile, "kwargs")
	else # create new arguments from function specification
		state_dim = size(dataargs[1][1].state_trajectory, 1)
		property_dim = length(property_fn(dataargs[1][1].state_trajectory))
		if hasproperty(dataargs[1][1], :context)
			context_dim = length(dataargs[1][1].context)
		else
			context_dim = 1
		end
		this_args = (state_dim, property_dim, context_dim)
		this_kwargs = kwargs
	end
	# construct the neural network
	sn = SliceNet(this_args...; this_kwargs...)
	if isfile(loadfile)
		Flux.loadmodel!(sn, model_state)
	end
	
	# initialize batch for terminal state and time estimator
	term_batch = (
		x = Matrix{Float32}(undef, sn.state_dim+sn.property_dim+1, 0),
		ctx = Matrix{Float32}(undef, sn.state_dim+sn.context_dim, 0)
	)
	
	# initialize batch for bisection net 
	mid_batch = (
		x = Matrix{Float32}(undef, sn.state_dim+2*sn.property_dim, 0),
		ctx = Matrix{Float32}(undef, 2*sn.state_dim+sn.property_dim+sn.context_dim, 0)
	)

	# optimizer state
	opt_state = Flux.setup(optimiser, sn)

	iter = 0
	start_time = time()
	while iter < maxiter
		for data in dataargs
			id = (iter % length(data)) + 1
			strj = data[id].state_trajectory

			T = size(strj, 2)

			if hasproperty(data[id], :context)
				context = vec(data[id].context)
			else
				context = [0]
			end

			# enlarge terminal estimation batch
			s = rand(1:(T-1))
			x_term = vcat(strj[:, end], property_fn(strj[:, s:end]), T-s)
			ctx_term = vcat(strj[:, s], context)
			term_batch = (
				x = hcat(term_batch.x, x_term),
				ctx = hcat(term_batch.ctx, ctx_term)
			)
			
			# calculate gradient of terminal estimator
			term_grads = Flux.gradient(sn) do model
				z, ld = model.term(term_batch.x, term_batch.ctx; inverse=false)
				sum(0.5*z.^2) - sum(ld)
			end

			# update
			Flux.update!(opt_state, sn, term_grads[1])
			
			# sort terminal batch 
			term_z, term_ld = sn.term(term_batch.x, term_batch.ctx; inverse=false)
			term_ll = term_ld - vec(sum(0.5*term_z.^2, dims=1))
			sortid_term = sortperm(term_ll)
			term_bs = min(size(term_batch.x, 2), batch_size)
			term_batch = (
				x = term_batch.x[:, sortid_term[1:term_bs]],
				ctx = term_batch.ctx[:, sortid_term[1:term_bs]]
			)

			# enlarge midpoint estimation batch 
			s = rand(1:(T-2))
			t = rand((s+2):T)
			x_mid = vcat(
				strj[:, (s+t)÷2], 
				property_fn(strj[:, 1:((s+t)÷2)]),
				property_fn(strj[:, (((s+t)÷2) + 1):end])
			)
			ctx_mid = vcat(strj[:, s], strj[:, t], property_fn(strj[:, s:t]), context)
			mid_batch = (
				x = hcat(mid_batch.x, x_mid),
				ctx = hcat(mid_batch.ctx, ctx_mid)
			)
			
			# calcuate gradient of bisection estimator 
			mid_grads = Flux.gradient(sn) do model
				z, ld = model.bisect(mid_batch.x, mid_batch.ctx; inverse=false)
				sum(0.5*z.^2) - sum(ld)
			end

			# update
			Flux.update!(opt_state, sn, mid_grads[1])

			# sort middle batch 
			mid_z, mid_ld = sn.bisect(mid_batch.x, mid_batch.ctx; inverse=false)
			mid_ll = mid_ld - vec(sum(0.5*mid_z.^2, dims=1))
			sortid_mid = sortperm(mid_ll)
			mid_bs = min(size(mid_batch.x, 2), batch_size)
			mid_batch = (
				x = mid_batch.x[:, sortid_mid[1:mid_bs]],
				ctx = mid_batch.ctx[:, sortid_mid[1:mid_bs]]
			)
		end
		iter += 1

		if !isempty(savefile) && time() - start_time > save_period
			JLD2.save(
				savefile,
				Dict(
					"model_state"=>Flux.state(sn),
					"args"=>this_args,
					"kwargs"=>this_kwargs
				)
			)
		end
	end

	if !isempty(savefile)
		JLD2.save(
			savefile,
			Dict(
				"model_state"=>Flux.state(sn),
				"args"=>this_args,
				"kwargs"=>this_kwargs
			)
		)
	end
	
	return sn
end

# ╔═╡ fc75a0f5-905a-4838-94b5-601d18eaeca6
function context_fn(strj::AbstractMatrix{<:Real})
	return vcat(strj[:, 1], strj[1, end] - strj[8, end])
end

# ╔═╡ 55df17f3-1184-4227-9016-b1782ad4a938
function property_fn(strj::AbstractMatrix{<:Real})
	ds = discrete_vehicles(0.25)

	# forward vehicle safety measure
	fs = maximum(max(abs.(s[1] - s[8]) - 5.0, s[2] - s[9] - 2.0) for s in eachcol(strj))

	# oncoming vehicle safety measure
	os = maximum(max(abs.(s[1] - s[11]) - 5.0, s[12] - s[2] - 2.0) for s in eachcol(strj))

	# rectangular bounds safety measure
	rs = vec(maximum(radius_hyperrectangle(ds.X) .- abs.(strj .- center(ds.X)), dims=2))[2:7]

	return vcat(fs, os, rs)
end

# ╔═╡ 95c75896-63a9-4e2d-a691-f2911eb5952a
function control_fn(strj::AbstractMatrix{<:Real})
	return strj[1:1,end] - strj[8:8, end]
end

# ╔═╡ ac48c4d9-2ec7-4b2d-8142-a2447769ca92
let
	data = JLD2.load("data/car/trajectories.jld2", "data")
	ov_idx = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0.0 for d in data]	
	ov_data = data[ov_idx]
	
	ds = discrete_vehicles(0.25) 

	idx = rand(1:length(ov_data))
	strj, utrj = ov_data[idx].state_trajectory, data[idx].input_signal

	savefile = "data/car/tbmoe/ann.jld2"

	state_scale = 0.1*ones(13)
	state_scale[[3, 5, 6, 7]] .*= 5.0

	property_scale = ones(8)*0.1
	property_scale[[4, 6, 7, 8]] *= 5

	optimizer=Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam(1e-3))

	state_dim = length(state_scale)
	property_dim = length(property_scale)

	times = [1, 2, 5, 11, 13]
	states = strj[:, times]
	context = [0.0]
	properties = reduce(hcat, property_fn(strj[:, times[i]:times[i+1]]) for i in 1:(length(times)-1))

	savefile = "data/car/seqgen/ann.jld2"
	@time sn = train(SliceNet, property_fn, ov_data; maxiter=10, batch_size=200, state_scale=state_scale, property_scale=property_scale, savefile=savefile)
end

# ╔═╡ Cell order:
# ╠═6bb100cc-96dd-11f0-3815-357f2700d588
# ╠═c681b310-5a90-454e-ac83-0140a2893232
# ╠═454e0c70-dd06-4859-8ef6-b40c1bbcc1d5
# ╠═3a5df80f-0062-4d6b-b187-4efb1b1de950
# ╠═fc75a0f5-905a-4838-94b5-601d18eaeca6
# ╠═55df17f3-1184-4227-9016-b1782ad4a938
# ╠═95c75896-63a9-4e2d-a691-f2911eb5952a
# ╠═ac48c4d9-2ec7-4b2d-8142-a2447769ca92

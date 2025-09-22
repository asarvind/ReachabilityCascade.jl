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

# ╔═╡ a3f5a61e-d69a-49ef-8f32-694821ce35c0
let
	d = Dict()
	d[:a] = 10
	d
end

# ╔═╡ 454e0c70-dd06-4859-8ef6-b40c1bbcc1d5
begin
# =================== Sequence Generator Structure ========================

struct TBMoE{CF}
	control_dim::Integer
	context_dim::Integer
	property_dim::Integer  	
	top_expert::CF
	mid_experts::Vector{<:CF}
	bot_expert::CF
end

Flux.@layer TBMoE

function TBMoE(control_dim::Integer, context_dim::Integer, property_dim::Integer; num_experts::Integer=3, context_scale::Vector{<:Real}=ones(context_dim), control_scale::Vector{<:Real}=ones(control_dim), property_scale::Vector{<:Real}=ones(property_dim), kwargs...)
	@assert num_experts >= 3 "at least 3 experts are required"

	# construct top_expert
	top_expert = ConditionalFlow(property_dim+1, context_dim, x_scaling=vcat(property_scale, 1.0), c_scaling=context_scale)

	# construct middle experts
	mid_experts = [
		ConditionalFlow(property_dim, context_dim+property_dim, x_scaling=property_scale, c_scaling=vcat(context_scale, property_scale)) 
		for _ in 1:(num_experts-2)
	]

	# contruct bottom expert 
	bot_expert = ConditionalFlow(control_dim, context_dim+property_dim, x_scaling=control_scale, c_scaling=vcat(context_scale, property_scale))	

	return TBMoE(control_dim, context_dim, property_dim, top_expert, mid_experts, bot_expert)
end


function (moe::TBMoE)(context::Vector{<:Real}, latent::AbstractMatrix{<:Real})
	@assert size(latent, 2) == length(moe.mid_experts) + 2 "number of latent columns should be equal to the number of experts"
	@assert size(latent, 1) >= max(moe.property_dim+1, moe.control_dim) "lenght of latent vectors should not be less than the control dimension and property dimension"

	# results of top expert
	top_res = moe.top_expert(latent[1:(moe.property_dim+1), end], context, inverse=true)[1]
	t = min(top_res[end], 2^(length(moe.mid_experts)))
	property = top_res[1:(end-1)]
	top_res = copy(vec(property))

	# result of middle experts
	idx = Int(floor(log2(max(t, 2))))
	mid_res = Vector{Real}[]
	for i in idx:-1:1
		property = moe.mid_experts[i](latent[1:moe.property_dim, i+1], vcat(context, property), inverse=true)[1]
		push!(mid_res, vec(copy(property)))
	end

	# result of bottom expert
	u = moe.bot_expert(latent[1:moe.control_dim, 1], vcat(context, property), inverse=true)[1]

	return (time=t, top_expert=top_res, mid_experts=mid_res, bot_expert=u)
end
	

# =====================================
end

# ╔═╡ fc75a0f5-905a-4838-94b5-601d18eaeca6
function context_fn(strj::AbstractMatrix{<:Real}, utrj::AbstractMatrix{<:Real})
	return vcat(strj[:, 1], strj[1, end] - strj[8, end])
end

# ╔═╡ 55df17f3-1184-4227-9016-b1782ad4a938
function property_fn(strj::AbstractMatrix{<:Real}, utrj::AbstractMatrix{<:Real})
	ds = discrete_vehicles(0.25)
	
	term_state = strj[:, end]

	# forward vehicle safety measure
	fs = maximum(max(abs.(s[1] - s[8]) - 5.0, s[2] - s[9] - 2.0) for s in eachcol(strj))

	# oncoming vehicle safety measure
	os = maximum(max(abs.(s[1] - s[11]) - 5.0, s[12] - s[2] - 2.0) for s in eachcol(strj))

	# rectangular bounds safety measure
	rs = vec(maximum(radius_hyperrectangle(ds.X) .- abs.(strj .- center(ds.X)), dims=2))[2:7]

	return vcat(term_state, fs, os, rs)
end

# ╔═╡ 95c75896-63a9-4e2d-a691-f2911eb5952a
function control_fn(strj::AbstractMatrix{<:Real}, utrj::AbstractMatrix{<:Real})
	return utrj[:, 1]
end

# ╔═╡ c60da8a4-62ce-4314-a91a-51d453cf134a
function train(::Type{TBMoE}, control_fn::Function, context_fn::Function, property_fn::Function, num_experts::Integer, data::AbstractVector; maxiter::Integer=1, sampling_period::Integer=1, optimizer=Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam()), batch_size=200, save_period=60, savefile::String="", kwargs...)
	
	# control dimension
	strj, utrj = data[1].state_trajectory, data[1].input_signal
	control_dim = length(control_fn(strj, utrj))

	# context dimension
	context_dim = length(context_fn(strj, utrj))

	# property dimension
	property_dim = length(property_fn(strj, utrj))
	
	# construct neural network
	if !isfile(savefile)
		moe = TBMoE(control_dim, context_dim, property_dim; num_experts=num_experts, kwargs...)
	else
		args = JLD2.load(savefile, "args")
		kwargs = JLD2.load(savefile, "kwargs")
		model_state = JLD2.load(savefile, "model_state")
		moe = TBMoE(args...; kwargs...)
		Flux.loadmodel!(moe, model_state)
	end

	# set up optimizer
	opt_state = Flux.setup(optimizer, moe)

	if maxiter < length(data)
		@warn "Number of gradient descent steps is smaller than lenght of the data.  Consider changing the `maxiter` keyword assignment.  The default is set to just 1 for safety reasons."
	end

	# top expert context and sample batches 
	top_ctx_batch = Matrix{Float32}(undef, context_dim, 0)
	top_smp_batch = Matrix{Float32}(undef, property_dim+1, 0)

	# middle expert context and sample batches
	mid_ctx_batches = fill(Matrix{Float32}(undef, context_dim+property_dim, 0), length(moe.mid_experts))
	mid_smp_batches = fill(Matrix{Float32}(undef, property_dim, 0), length(moe.mid_experts))	

	# bottom expert context and sample batches
	bot_ctx_batch = Matrix{Float32}(undef, property_dim+context_dim, 0)
	bot_smp_batch = Matrix{Float32}(undef, control_dim, 0)

	function this_bisect(strj::AbstractMatrix{<:Real}, utrj::AbstractMatrix{<:Real}, st::Integer, t::Integer)
		strj_cut, utrj_cut = strj[:, st:end], utrj[:, st:end]
		
		if t > size(utrj_cut, 2)
			return Matrix{Float32}(undef, context_dim, 0), Matrix{Float32}(undef, property_dim, 0), Matrix{Float32}(undef, property_dim, 0)	
		end		
		
		context_fn(strj_cut[:, 1:(t+1)], utrj_cut[:, 1:t]),
		property_fn(strj_cut[:, 1:(t+1)], utrj_cut[:, 1:t]),
		property_fn(strj_cut[:, 1:((t÷2) + 1)], utrj_cut[:, 1:(t÷2)])
	end

	# shuffle data
	data_shuffled = shuffle(data)

	iter = 0 # iteration count 
	start_time = time() # count time for saving network periodically
	while iter < maxiter
		# extract data pair
		idx = (iter÷length(data)) + 1
		strj, utrj = data_shuffled[idx].state_trajectory, data[idx].input_signal

		# extend top expert batches
		top_data = [(
			context_fn(strj[:, st:end], utrj[:, st:end]),
			property_fn(strj[:, st:end], utrj[:, st:end]), 
			size(utrj[:, st:end], 2)
		)
			for st in 1:sampling_period:size(utrj, 2)
		]
		top_ctx_batch = hcat(
			reduce(hcat, p[1] for p in top_data),
			top_ctx_batch
		)
		top_smp_batch = hcat(
			reduce(hcat, vcat(p[2], p[3]) for p in top_data),
			top_smp_batch
		)

		# compute gradient 
		top_grads = Flux.gradient(moe) do model
			ld, z = model.top_expert(top_smp_batch, top_ctx_batch)
			(0.5*sum(z.^2) - sum(ld)) /length(ld)
		end

		# update top expert 
		Flux.update!(opt_state, moe, top_grads[1])

		# sort batches
		top_ll = loglikelihoods(moe.top_expert, top_smp_batch, top_ctx_batch)
		top_idx = sortperm(top_ll)
		top_smp_batch = top_smp_batch[:, 1:min(batch_size, size(top_smp_batch, 2))]
		top_ctx_batch = top_ctx_batch[:, 1:min(batch_size, size(top_ctx_batch, 2))]

		# iterate over each middle expert
		Threads.@threads for i in 1:length(moe.mid_experts)		
			# extend middle expert batches 
			mid_data = [
				this_bisect(strj, utrj, st, t)
				for t in vcat(rand(2^i:(2^(i+1)-1), sampling_period), 2^(i+1)-1) for st in 1:sampling_period:size(strj, 2)
			]
			mid_ctx_batches[i] = hcat(
				reduce(hcat, vcat(p[1], p[2]) for p in mid_data),
				mid_ctx_batches[i]
			)
			mid_smp_batches[i] = hcat(
				reduce(hcat, p[3] for p in mid_data),
				mid_smp_batches[i]
			)

			# calculate gradient
			mid_grads = Flux.gradient(moe) do model 
				ld, z = model.mid_experts[i](mid_smp_batches[i], mid_ctx_batches[i])
				(0.5*sum(z.^2) - sum(ld)) /length(ld)
			end

			# update middle expert 
			Flux.update!(opt_state, moe, mid_grads[1])

			# sort mid batches
			mid_ll = loglikelihoods(moe.mid_experts[i], mid_smp_batches[i], mid_ctx_batches[i])
			mid_idx = sortperm(mid_ll)
			mid_smp_batches[i] = mid_smp_batches[i][:, 1:min(batch_size, size(mid_smp_batches[i], 2))]
			mid_ctx_batches[i] = mid_ctx_batches[i][:, 1:min(batch_size, size(mid_ctx_batches[i], 2))]			
		end

		# extend bottom expert
		bot_data =  [(
			context_fn(strj[:, st:end], utrj[:, st:end]),
			property_fn(strj[:, st:end], utrj[:, st:end]), 
			control_fn(strj[:, st:end], utrj[:, st:end]), 
		)
			for st in 1:sampling_period:size(utrj, 2)
		]
		bot_ctx_batch = hcat(
			reduce(hcat, vcat(p[1], p[2]) for p in bot_data),
			bot_ctx_batch
		)
		bot_smp_batch = hcat(
			reduce(hcat, p[3] for p in bot_data),
			bot_smp_batch
		)

		# calculate gradient
		bot_grads = Flux.gradient(moe) do model
			ld, z = model.bot_expert(bot_smp_batch, bot_ctx_batch)
			(0.5*sum(z.^2) - sum(ld)) /length(ld)
		end

		# sort bottom batch
		bot_ll = loglikelihoods(moe.bot_expert, bot_smp_batch, bot_ctx_batch)
		bot_idx = sortperm(bot_ll)
		bot_smp_batch = bot_smp_batch[:, 1:min(batch_size, size(bot_smp_batch, 2))]
		bot_ctx_batch = bot_ctx_batch[:, 1:min(batch_size, size(bot_ctx_batch, 2))]

		# update bottom expert
		Flux.update!(opt_state, moe, bot_grads[1])

		iter += 1

		# save model
		if !isempty(savefile) && time() - start_time > save_period
			JLD2.save(
				savefile,
				Dict(
					"args" => (control_dim, context_dim, property_dim),
					"kwargs" => merge((num_experts=num_experts,), kwargs),
					"model_state" => Flux.state(moe)					
				)
			)
		end
	end

	# save model
	if !isempty(savefile)
		JLD2.save(
			savefile,
			Dict(
				"args" => (control_dim, context_dim, property_dim),
				"kwargs" => merge((num_experts=num_experts,), kwargs),
				"model_state" => Flux.state(moe)					
			)
		)		
	end
	
	return moe
end

# ╔═╡ ac48c4d9-2ec7-4b2d-8142-a2447769ca92
let
	data = JLD2.load("data/car/trajectories.jld2", "data")

	ds = discrete_vehicles(0.25) 

	idx = rand(1:length(data))
	strj, utrj = data[idx].state_trajectory, data[idx].input_signal

	moe = TBMoE(2, 14, 21, num_experts=2+5)

	moe(rand(14), rand(22, length(moe.mid_experts)+2))

	savefile = "data/car/tbmoe/ann.jld2"

	@time train(TBMoE, control_fn, context_fn, property_fn, 6, data; sampling_period=4, maxiter=20, savefile=savefile, save_period=60.0, hidden=128)
end

# ╔═╡ Cell order:
# ╠═6bb100cc-96dd-11f0-3815-357f2700d588
# ╠═c681b310-5a90-454e-ac83-0140a2893232
# ╠═a3f5a61e-d69a-49ef-8f32-694821ce35c0
# ╠═454e0c70-dd06-4859-8ef6-b40c1bbcc1d5
# ╠═fc75a0f5-905a-4838-94b5-601d18eaeca6
# ╠═55df17f3-1184-4227-9016-b1782ad4a938
# ╠═95c75896-63a9-4e2d-a691-f2911eb5952a
# ╠═c60da8a4-62ce-4314-a91a-51d453cf134a
# ╠═ac48c4d9-2ec7-4b2d-8142-a2447769ca92

### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ 54132978-a055-11f0-26d2-5f337f47dba2
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

# ╔═╡ 5e39abbe-7d94-4362-80e6-85dd6f216d46
begin
	import Flux
	using LinearAlgebra, Random
	import LazySets: Hyperrectangle, high, low, dim, radius_hyperrectangle, center
	import JLD2
	import ReachabilityCascade.CarDynamics: discrete_vehicles	
	import ReachabilityCascade: ConditionalFlow
	import ReachabilityCascade.TrajectoryModels: TrajectoryEncoder, encode, predict, predict_from_traj
end

# ╔═╡ e783eb29-f9a2-4fe5-b6f8-0106c164fd37
begin

import ReachabilityCascade.TrajectoryModels: train
	
struct TrajectoryDistributor
	encoder::TrajectoryEncoder
	cond_flow::ConditionalFlow
end

Flux.@layer ConditionalFlow 

function (td::TrajectoryDistributor)(state_vec::AbstractVector{<:Real}, time_elapsed::Real, latent_vec::AbstractVector{<:Real})
	z_latent, _ = td.cond_flow(latent_vec, state_vec, inverse=true)
	return predict(td.encoder, vec(z_latent), state_vec, time_elapsed)
end

function train(::Type{TrajectoryDistributor}, encoder::TrajectoryEncoder, dataset::AbstractVector; epochs::Integer=1, batch_size::Integer=1, optimizer=Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam()), save_path::String="", load_path::String=save_path, save_period::Real=60, kwargs...)
	# construct network 
	if isfile(load_path)
		flow_args = JLD2.load(load_path, "args")
		loaded_flow_kwargs = JLD2.load(load_path, "kwargs")
		model_state = JLD2.load(load_path, "modelstate")
		cond_flow = ConditionalFlow(flow_args...; loaded_flow_kwargs...)
		Flux.loadmodel!(cond_flow, model_state)
		if epochs < 1
			return TrajectoryDistributor(encoder, cond_flow)
		end
	else
		flow_args = (encoder.embeddim, encoder.statedim)
		loaded_flow_kwargs = kwargs
		cond_flow = ConditionalFlow(flow_args...; loaded_flow_kwargs...)
	end

	# initialize optimizer
	opt_state = Flux.setup(optimizer, cond_flow)
	
	# shuffle data
	shuffled_data = shuffle(dataset)

	# pre-process data
	encoded_data = [ encode(encoder, Float32.(d.state_trajectory)) for d in shuffled_data]

	sample_batch = Matrix{Float32}(undef, cond_flow.x_dim, 0)
	context_batch = Matrix{Float32}(undef, cond_flow.ctx_dim, 0)

	# clock for saving model periodically
	last_save_at = time()
	
	for _ in 1:epochs
		for idx in 1:length(shuffled_data)
			sample_batch = hcat(sample_batch, encoded_data[idx])
			state_traj = shuffled_data[idx].state_trajectory
			sample_idx = rand(1:size(state_traj, 2))
			context_batch = hcat(context_batch, state_traj[:, sample_idx])

			grads = Flux.gradient(cond_flow) do model
				z, log_det = model(sample_batch, context_batch; inverse=false)
				0.5*sum(abs2.(z)) - sum(log_det)
			end
			Flux.update!(opt_state, cond_flow, grads[1])

			# sort out lowest likelihood samples
			batch_keep = min(batch_size, size(sample_batch, 2))
			z_now, ld_now = cond_flow(sample_batch, context_batch; inverse=false)
			loglike =  ld_now - vec(0.5*sum(abs2.(z_now), dims=1))
			keep_idx = sortperm(loglike)[1:batch_keep]
			sample_batch = sample_batch[:, keep_idx]
			context_batch = context_batch[:, keep_idx]

			if !isempty(save_path) && time() - last_save_at > save_period
				JLD2.save(
					save_path,
					Dict(
						"args"=>flow_args,
						"kwargs"=>kwargs,
						"modelstate"=>Flux.state(cond_flow)
					)
				)
				last_save_at = time()
			end
		end
	end

	if !isempty(save_path)
		JLD2.save(
			save_path,
			Dict(
				"args"=>flow_args,
				"kwargs"=>kwargs,
				"modelstate"=>Flux.state(cond_flow)
			)
		)
	end

	return TrajectoryDistributor(encoder, cond_flow)
end
	
end

# ╔═╡ e450db6d-097d-4871-ba74-4ffdb3492dcf
let 
	Random.seed!()
	data = JLD2.load("data/car/trajectories.jld2", "data")

	overtake_ids = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 for d in data]
	overtake_data = data[overtake_ids]

	embdim = 10
	dmodel = 128
	nheads = 2
	numlayers = 2
	ffhidden = 128
	timedim = 8
	predhidden = (128, 128, 128)

	errorscale = ones(Float32, 13)
	errorscale[[3, 5, 6, 7]] .*= 5.0f0

	epochs = 0
	
	savefile = "data/car/seqgen/encoder.jld2"
	
	@time net = train(TrajectoryEncoder, overtake_data, embdim; dmodel=dmodel, nheads=nheads, numlayers=numlayers, ffhidden=ffhidden, timedim=timedim, predhidden=predhidden, savefile=savefile, epochs=epochs)

	testid = rand(1:length(overtake_data))
	# testid = 3916
	println(testid)
	X = Float32.(overtake_data[testid].state_trajectory)

	T = size(X, 2)

	z = encode(net, X)
	t = rand(1:(T-1))
	Δt = rand(1:(T-t))
	predict_from_traj(net, X, X[:, t], Δt), X[:, t+Δt], t, Δt

	save_flow = "data/car/seqgen/flownet.jld2"
	@time tdnet = train(TrajectoryDistributor, net, overtake_data; save_path=save_flow, epochs=0, batch_size=100)
	tdnet(X[:, t], Δt, randn(tdnet.encoder.embeddim)), X[:, t+Δt], tdnet(X[:, t], Δt, 5*randn(tdnet.encoder.embeddim)), t, Δt
end

# ╔═╡ Cell order:
# ╠═54132978-a055-11f0-26d2-5f337f47dba2
# ╠═5e39abbe-7d94-4362-80e6-85dd6f216d46
# ╠═e783eb29-f9a2-4fe5-b6f8-0106c164fd37
# ╠═e450db6d-097d-4871-ba74-4ffdb3492dcf

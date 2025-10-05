### A Pluto.jl notebook ###
# v0.20.4

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
	encodenet::TrajectoryEncoder
	lantentnet::ConditionalFlow
end

Flux.@layer ConditionalFlow 

function (td::TrajectoryDistributor)(x::AbstractVector{<:Real}, t::Real, latent::AbstractVector{<:Real})
	z = td(latent, x, inverse=true)
	return predict(td.encodenet, z, x, t)
end

function train(::Type{TrajectoryDistributor}, encnet::TrajectoryEncoder, data::AbstractVector; epochs::Integer=1, batchsize::Integer=1, optimiser=Flux.OptimiserChain(Flux.ClipNorm(), Flux.Adam()), savefile::String="", loadfile::String=savefile, saveperiod::Real=60, kwargs...)
	# construct network 
	if isfile(loadfile)
		args = JLD2.load(loadfile, "args")
		thiskwargs = JLD2.load(loadfile, "kwargs")
		ms = JLD2.load(loadfile, "modelstate")
		cf = ConditionalFlow(args...; thiskwargs...)
		Flux.loadmodel!(cf, ms)
	else
		args = (encnet.embeddim, encnet.statedim)
		thiskwargs = kwargs
		cf = ConditionalFlow(args...; thiskwargs...)
	end

	# initialize optimizer
	optstate = Flux.setup(optimiser, cf)
	
	# shuffle data
	shufdata = shuffle(data)

	# pre-process data
	encdata = [ encode(encnet, Float32.(d.state_trajectory)) for d in shufdata]

	smpbatch = Matrix{Float32}(undef, cf.x_dim, 0)
	ctxbatch = Matrix{Float32}(undef, cf.ctx_dim, 0)

	# clock for saving model periodically
	clk = time()
	
	for _ in 1:epochs
		for i in 1:length(shufdata)
			smpbatch = hcat(smpbatch, encdata[i])
			strj = shufdata[i].state_trajectory
			id = rand(1:size(strj, 2))
			ctxbatch = hcat(ctxbatch, strj[:, id])

			grads = Flux.gradient(cf) do model
				z, ld = model(smpbatch, ctxbatch; inverse=false)
				0.5*sum(abs2.(z)) - sum(ld)
			end
			Flux.update!(optstate, cf, grads[1])

			# sort out lowest likelihood samples
			bs = min(batchsize, size(smpbatch, 2))
			thisz, thisld = cf(smpbatch, ctxbatch; inverse=false)
			ll =  thisld - vec(0.5*sum(abs2.(thisz), dims=1))
			sortid = sortperm(ll)[1:bs]
			smpbatch = smpbatch[:, sortid]
			ctxbatch = ctxbatch[:, sortid]

			if !isempty(savefile) && time() - clk > saveperiod
				JLD2.save(
					savefile,
					Dict(
						"args"=>args,
						"kwargs"=>kwargs,
						"modelstate"=>Flux.state(cf)
					)
				)
				clk = time()
			end
		end
	end

	if !isempty(savefile)
		JLD2.save(
			savefile,
			Dict(
				"args"=>args,
				"kwargs"=>kwargs,
				"modelstate"=>Flux.state(cf)
			)
		)
	end

	return TrajectoryDistributor(encnet, cf)
end
	
end

# ╔═╡ e450db6d-097d-4871-ba74-4ffdb3492dcf
let 
	Random.seed!()
	data = JLD2.load("data/car/trajectories.jld2", "data")

	overtakeids = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 0 for d in data]
	overtakedata = data[overtakeids]

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
	
	@time net = train(TrajectoryEncoder, overtakedata, embdim; dmodel=dmodel, nheads=nheads, numlayers=numlayers, ffhidden=ffhidden, timedim=timedim, predhidden=predhidden, savefile=savefile, epochs=epochs)

	testid = rand(1:length(overtakedata))
	# testid = 3916
	println(testid)
	X = Float32.(overtakedata[testid].state_trajectory)

	T = size(X, 2)

	z = encode(net, X)
	t = rand(1:(T-1))
	Δt = rand(1:(T-t))
	predict_from_traj(net, X, X[:, t], Δt), X[:, t+Δt], t, Δt

	saveflow = "data/car/seqgen/flownet.jld2"
	train(TrajectoryDistributor, net, overtakedata[1:3]; savefile=saveflow)
end

# ╔═╡ Cell order:
# ╠═54132978-a055-11f0-26d2-5f337f47dba2
# ╠═5e39abbe-7d94-4362-80e6-85dd6f216d46
# ╠═e783eb29-f9a2-4fe5-b6f8-0106c164fd37
# ╠═e450db6d-097d-4871-ba74-4ffdb3492dcf

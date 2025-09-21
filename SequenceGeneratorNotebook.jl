### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 4b827b3c-9525-11f0-3c26-399b56721e1f
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

# ╔═╡ c748d122-c751-4a57-8ebd-118fa934442a
begin
	import Flux
	using LinearAlgebra, Random
	import LazySets: Hyperrectangle, high, low, dim
	import JLD2
	import ReachabilityCascade.CarDynamics: discrete_vehicles	
	import ReachabilityCascade: SequenceGenerator, train
end

# ╔═╡ 1d27def9-07ea-4810-be60-10f8b3408a41
function sampling_fn(strj::AbstractMatrix{<:Real}, utrj::AbstractMatrix{<:Real}) 
	ctx = []
	smp = []
	tseq = [1, 2, 4, 8, 16, 28]
	for st in 1:4:28
		strj_cut = strj[:, st:end]
		utrj_cut = utrj[:, st:end]
		for i in 1:length(tseq)
			if tseq[i] <= size(utrj_cut, 2)
				push!(ctx, vcat(strj_cut[:, 1], strj_cut[1, end] - strj_cut[8, end], tseq[i]))
				push!(smp, vcat(strj_cut[:, tseq[i]+1], utrj_cut[:, tseq[i]]))
				if i > 1
					trand = rand(tseq[i-1]:tseq[i])
					push!(ctx, vcat(strj_cut[:, 1], strj_cut[1, end] - strj_cut[8, end], trand))
					push!(smp, vcat(strj_cut[:, trand+1], utrj_cut[:, trand]))
				end
			end
		end
	end
	
	return (sample=reduce(hcat, smp), context=reduce(hcat, ctx))
end

# ╔═╡ 3314468b-2f5b-41da-a260-6e543d267d40
let
	data = JLD2.load("data/car/trajectories.jld2", "data")
	ov_idx = [d.state_trajectory[1, end] - d.state_trajectory[8, end] > 5.0 for d in data]
	ov_data = data[ov_idx]
	
	ds = discrete_vehicles(0.25)

	context_dim = dim(ds.X) + 1 + 1 # state_dim + reward_dim + time_dim 
	sample_dim = dim(ds.X) + dim(ds.U) # state_dim + control_dim

	latent_dim = 2*sample_dim 

	sg = SequenceGenerator(; sample_dim=sample_dim, context_dim=context_dim, latent_dim=latent_dim, act=Flux.leakyrelu, gate_act=Flux.σ, widths=[128, 64, 128])

	flow_times = [0.1, 0.3, 0.8]

	# generator_velocity(sg, rand(sample_dim, 2), rand(context_dim, 2), rand(latent_dim, 2), flow_times)

	tseq = [1, 2, 4, 8, 16, 28]
	sampling_fn(data[1].state_trajectory, data[1].input_signal)
	ov_data
	@time train(SequenceGenerator, sampling_fn, 200, data, ov_data, latent_dim=latent_dim, batch_size=100, flow_num=5, widths=[128, 128, 128], savefile="data/car/seqgen/ann.jld2")
end

# ╔═╡ e8d76c7e-9599-4a72-8dd6-af62c6682054
JLD2.load("data/car/seqgen/ann.jld2", "kwargs")

# ╔═╡ Cell order:
# ╠═4b827b3c-9525-11f0-3c26-399b56721e1f
# ╠═c748d122-c751-4a57-8ebd-118fa934442a
# ╠═1d27def9-07ea-4810-be60-10f8b3408a41
# ╠═3314468b-2f5b-41da-a260-6e543d267d40
# ╠═e8d76c7e-9599-4a72-8dd6-af62c6682054

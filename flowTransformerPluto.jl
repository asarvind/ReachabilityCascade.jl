### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ f4a170e4-badf-11f0-2436-c7b7cb9e8770
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

# ╔═╡ 57ca6d27-0179-4bcd-a196-093b648c2873
begin
	
using Random, LinearAlgebra
import JLD2
using LazySets: Hyperrectangle, low, high
import Flux
import ReachabilityCascade.CarDynamics: discrete_vehicles, safe_discrete_vehicles
using ReachabilityCascade: FlowTransformer, train!, load_flow_transformer, perturb_input_sequence
end

# ╔═╡ fd727526-1ac1-4f0f-8529-c31d599bbf39
# begin

# struct CarControlIter
# 	data::AbstractVector
# 	maxiter::Integer
# end

# function Base.iterate(cci::CarControlIter, state=1)
# 	data_tup = cci.data[state]
# 	umat = Float32.(data_tup.input_signal)
# 	T = size(umat, 2)
# 	start_time = rand(1:T)
# 	final_time = T
# 	x = data_tup.state_trajectory[:, start_time]
# 	goal = data_tup.state_trajectory[:, (final_time+1)]
# 	context = Float32.(vcat(x, goal))
# 	if state <= cci.maxiter && state <= length(data)
# 		return (context, umat[:, start_time:final_time]), state + 1 
# 	else
# 		return nothing 
# 	end
# end

# end

# ╔═╡ add5d709-ead7-4340-902d-8f72a2cc9129
begin

struct CarTrajectory
	data::AbstractVector
	maxiter::Integer
end

function Base.iterate(ct::CarTrajectory, state=1)
	data_tup = ct.data[min(state, length(ct.data))]
	strj = Float32.(data_tup.state_trajectory)
	utrj = Float32.(data_tup.input_signal)
	T = size(data_tup.input_signal, 2)
	start_time = rand(1:T)
	final_time = T
	x = data_tup.state_trajectory[:, start_time]
	goal = data_tup.state_trajectory[:, (final_time+1)]
	context = Float32.(vcat(x, goal))
	seq = vcat(strj[:, (start_time+1):(final_time+1)] .- x, utrj[:, start_time:final_time])
	if state <= ct.maxiter && state <= length(ct.data)
		return (context, seq), state + 1 
	else
		return nothing 
	end
end

end

# ╔═╡ b38b5a94-763a-45cf-acb6-ac2e03339e76
data = JLD2.load("data/car/trajectories.jld2", "data")

# ╔═╡ f87720bd-3153-4e23-9eae-48e6a7e31cac
save_path = "data/car/temp/flow.jld2"

# ╔═╡ e65033ba-404e-454a-8248-d167a5d7b13b
let
	maxiter = 30000
	cci = CarTrajectory(shuffle(data), maxiter)

	

	constructor_kwargs = (
		num_layers = 10, 
		num_heads = 1,
		ff_hidden = 64,
		clamp = 2.0,
		use_layernorm = true,
		batch_size = 50,
		max_seq_len = 30,
		activation_scale = 0.01
	)
	
	# @time train!(
	# 	FlowTransformer,
	# 	cci;
	# 	epochs = 10,
	# 	constructor_kwargs...,
	# 	save_path = save_path,
	# )

end

# ╔═╡ 6d8149ac-2090-4f5b-9edf-faa5d776cf97
let
	Random.seed!()
	maxiter = length(data)
	# idx = rand(1:length(data))
	idx = 14309
	println(idx)
	ct = CarTrajectory(data, maxiter)
	flow = load_flow_transformer(save_path)
	
	(context, seq), _ = iterate(ct, min(maxiter, idx))
	latent = randn(Float32, size(seq))
	flow(latent, context, inverse=true)[1:13, :] .+ context[1:13], seq[1:13, :] .+ context[1:13], context
end

# ╔═╡ 243105cf-e8b3-48a9-a9e3-82c3e4bacaa5
let
	ds = safe_discrete_vehicles(0.25)
	ind = rand(1:length(data))
	# ind = 2851
	strj, utrj = data[ind].state_trajectory, data[ind].input_signal
	println(ind)
	norm(strj[1:2, end] - rand(2))
	uper = perturb_input_sequence(ds, strj[:, 1], utrj, [0.05, 0.5], 
								  (x, u) -> 0.0, 
								  (x, u) -> norm(strj[1:2, end] - x[1:2]),
								  3
								 )
	
end

# ╔═╡ Cell order:
# ╠═f4a170e4-badf-11f0-2436-c7b7cb9e8770
# ╠═57ca6d27-0179-4bcd-a196-093b648c2873
# ╠═fd727526-1ac1-4f0f-8529-c31d599bbf39
# ╠═add5d709-ead7-4340-902d-8f72a2cc9129
# ╠═b38b5a94-763a-45cf-acb6-ac2e03339e76
# ╠═f87720bd-3153-4e23-9eae-48e6a7e31cac
# ╠═e65033ba-404e-454a-8248-d167a5d7b13b
# ╠═6d8149ac-2090-4f5b-9edf-faa5d776cf97
# ╠═243105cf-e8b3-48a9-a9e3-82c3e4bacaa5

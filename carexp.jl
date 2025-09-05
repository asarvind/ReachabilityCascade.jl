### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ fc3c2db6-7193-11f0-1b65-7f390e18200d
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

# ╔═╡ 23334c32-43a8-43e5-ac84-7d007d6f24d1
begin
	using Random, LinearAlgebra, JLD2
	using LazySets: Hyperrectangle, low, high
end

# ╔═╡ 9c205052-f2a5-4616-a99e-4ab9d48f4ddf
using ReachabilityCascade: grid_serpentine, ConditionalFlow, loglikelihoods, NRLE, train

# ╔═╡ 91675df7-f4ef-48e8-ae23-bdf9e4d1ecf7
import ReachabilityCascade.CarDynamics: discrete_vehicles

# ╔═╡ 9f648ae2-043a-404d-9ed2-e985f5f642ba
import ReachabilityCascade.CarDataGeneration: gen_trajectory, generate_data

# ╔═╡ 7b3216d8-787e-4654-9827-7f303e42abff
let 
	Random.seed!(5)
	
	context_dim = 13
	sample_dim = 13*4
	num_blocks = 3


	proj = [1 zeros(1, 6) -1 zeros(1, 5)]

	d_gen = [0.0]
	

	ds = discrete_vehicles(0.25)
	x0 = copy(ds.X.center)
	x0[1] = 10.0
	x0[2] = 1.5
	x0[8] = 30
	x0[11] = 90

	ds_sub = discrete_vehicles(0.01)

	# Q = Matrix(1.0I, 7, 7)
	# R = Matrix(1.0I, 2, 2)

	# K, S, _ = lqr_lyap(A[1:7, 1:7], B[1:7, :], Q, R)

	

	T = 28
	lb = repeat(low(ds.X), 2)
	ub = repeat(high(ds.X), 2)
	lb[[1, 14]] .= 0.0
	ub[[1, 14]] .= 90.0
	lb[[4, 17]] .= 5.0
	ub[[4, 17]] .= 15.0
	lb[8] = 30.0
	ub[8] = 30.0
	lb[11] = 90.0
	ub[11] = 90.0
	rect_serp = Hyperrectangle(low=lb, high=ub)
	grid_dims = [1, 2, 4, 14, 15]
	step = [5.0, 1.0, 2.0, 5.0, 1.0]

	samples = grid_serpentine(rect_serp, step, dims=grid_dims)

	margin = vcat(1.0, 0.5, fill(1e7, 11))

	# generate_data(ds, samples, margin, T, savefile="data/car/test.jld2")

	data_dict = JLD2.load("data/car/trajectories.jld2")

	old_data = data_dict["data"]

	# generate_data(ds, samples, margin, T, data=old_data, start_index=old_data[end].index+1, last_index=200, savefile="data/car/test.jld2")

	# res, status = gen_trajectory(ds, x0, T, proj, d_gen)
	# ref_sample = vcat(x0, res.state_trajectory[:, end])
	# ref_sample[4] = 12

	_, check_ind = findmax([s.state_trajectory[1,1] < s.state_trajectory[8, 1] && s.state_trajectory[1,end] > s.state_trajectory[8, end] && s.state_trajectory[2,end] > 2.5 && abs(s.state_trajectory[6,end]) < 0.1 for s in old_data])

	old_data[check_ind].state_trajectory

	# ds(old_data[check_ind].state_trajectory[:, 1], old_data[check_ind].input_signal) - old_data[check_ind].state_trajectory

	JLD2.load("data/car/test.jld2")
end

# ╔═╡ 8a457e51-7cca-40f5-8448-fdd43672e8a2
function property_fun(strj::AbstractMatrix{<:Real}, utrj::AbstractMatrix{<:Real})
	T = size(utrj, 2)
	
	_, ov_idx = findmin(norm(strj[1:2, i] - strj[8:9, i], Inf) for i in 1:(T+1))

	_, obs_idx = findmin(norm(strj[1:2, i] - strj[11:12, i], Inf) for i in 1:(T+1))

	return vcat(strj[1:2, ov_idx] - strj[8:9, ov_idx], strj[1:2, obs_idx] - strj[11:12, obs_idx], utrj[:, end], T)
end

# ╔═╡ dc208608-727d-427d-acc0-9b4d90e9661a
let
	seed = rand(1:100000)
	Random.seed!(seed)
	println(seed)
	data = JLD2.load("data/car/trajectories.jld2", "data")
	time_stamps = [1, 2, 4, 8, 16, 28]

	state_scale = ones(13)
	state_scale[[3, 5, 6]] .*= 5

	prop_scale = ones(7)
	prop_scale[5] *= 5

	adj = 0.2
	state_scale .*= adj
	prop_scale .*= adj

	@time train(NRLE, property_fun, shuffle(data)[1:1000], time_stamps, state_scaling=state_scale, prop_scaling=prop_scale, n_blocks=6, hidden=128, n_glu=2, bias=true)
end

# ╔═╡ Cell order:
# ╠═fc3c2db6-7193-11f0-1b65-7f390e18200d
# ╠═23334c32-43a8-43e5-ac84-7d007d6f24d1
# ╠═9c205052-f2a5-4616-a99e-4ab9d48f4ddf
# ╠═91675df7-f4ef-48e8-ae23-bdf9e4d1ecf7
# ╠═9f648ae2-043a-404d-9ed2-e985f5f642ba
# ╠═7b3216d8-787e-4654-9827-7f303e42abff
# ╠═8a457e51-7cca-40f5-8448-fdd43672e8a2
# ╠═dc208608-727d-427d-acc0-9b4d90e9661a

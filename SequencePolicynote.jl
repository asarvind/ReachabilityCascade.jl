### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ a7463774-9949-11f0-22f7-c75e291fb35b
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

# ╔═╡ 8a75487b-b40f-4133-b9db-10c8e492be43
begin
	import Flux
	using LinearAlgebra, Random
	import LazySets: Hyperrectangle, high, low, dim, radius_hyperrectangle, center
	import JLD2
	import ReachabilityCascade.CarDynamics: discrete_vehicles	
	import ReachabilityCascade: ConditionalFlow, loglikelihoods
end

# ╔═╡ a2455b99-082b-4f05-80ab-6869a56b9fd1
begin
# ======================== TrajectoryMapping ===================== 
struct TrajectoryMapping
	
end

# =============================
end

# ╔═╡ e6264aa4-3231-4815-9b18-ba1c8d249963
let
	x = rand(1000, 1000)
	y = rand(1000)
	@time for _ in 1:1000
		x*y
	end
end

# ╔═╡ e44c21d9-0944-4181-acc1-388676000223
let
	x = rand(1000, 1000)
	y = rand(1000)
	@time for _ in 1:1000
		for i in 1:1000
			dot(view(x, i, :),y)
		end
	end
end

# ╔═╡ Cell order:
# ╠═a7463774-9949-11f0-22f7-c75e291fb35b
# ╠═8a75487b-b40f-4133-b9db-10c8e492be43
# ╠═a2455b99-082b-4f05-80ab-6869a56b9fd1
# ╠═e6264aa4-3231-4815-9b18-ba1c8d249963
# ╠═e44c21d9-0944-4181-acc1-388676000223

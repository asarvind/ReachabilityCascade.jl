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

# ╔═╡ c8d1f042-ad49-4971-aa32-b9c15d241da6
using ReachabilityCascade, LinearAlgebra, Random, LazySets

# ╔═╡ Cell order:
# ╠═fc3c2db6-7193-11f0-1b65-7f390e18200d
# ╠═c8d1f042-ad49-4971-aa32-b9c15d241da6

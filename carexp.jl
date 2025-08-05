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
using ReachabilityCascade, LinearAlgebra, Random, LazySets, Flux

# ╔═╡ 33cd9059-55ad-4352-b2a1-65165f93ec4f
let
# Define parameters
input_dim = 16       # Dimension of each input vector
embed_dim = 32       # Embedding dimension for each token
seq_len = 4          # Number of embedded tokens per input
num_blocks = 2       # Number of Transformer blocks
num_heads = 4        # Number of attention heads
out_dim = 65         # Final output dimension

# Construct the transformer model
model = transformer(input_dim, embed_dim, seq_len, num_blocks, num_heads, out_dim; use_norm=true)

# --- Single input vector (no batch) ---
x_single = rand(Float32, input_dim)        # shape: (input_dim,)
y_single = model(x_single)                 # shape: (out_dim,)
@show size(y_single)                       # expected: (64,)

# --- Batched input vectors ---
batch_size = 10
x_batch = rand(Float32, input_dim, batch_size)  # shape: (input_dim, batch_size)
y_batch = model(x_batch)                        # shape: (out_dim, batch_size)
@show size(y_batch)                             # expected: (64, 10)
Flux.trainable(model)
end

# ╔═╡ Cell order:
# ╠═fc3c2db6-7193-11f0-1b65-7f390e18200d
# ╠═c8d1f042-ad49-4971-aa32-b9c15d241da6
# ╠═33cd9059-55ad-4352-b2a1-65165f93ec4f

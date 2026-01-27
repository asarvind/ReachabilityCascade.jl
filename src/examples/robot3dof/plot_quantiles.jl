using JLD2
using Plots
using ReachabilityCascade: quantile_plot

# ---------------------------
# Editable settings
# ---------------------------
results_dir = joinpath(pwd(), "data", "robotarm", "results")
# Choose algorithm: :BOBYQA or :SLSQP (filters files by name).
algorithm = :BOBYQA
algo_tag = algorithm == :BOBYQA ? "AlgoBOBYQA" : "AlgoSLSQP"
filename_filter = Regex("Seed(2|200|2000)" * algo_tag * "\\.jld2\$")

# Output location (PaperSubs / ICML submission)
output_dir = "/Users/adimoolamarvind/main/PaperSubs/ICML2026ArvindSub/figures/robotarm"
output_file = joinpath(output_dir, "robotarm_quantiles_$(String(algorithm)).pdf")

# Plot appearance
show_legend = false
xlabel = "% quantile"
ylabel = "Evaluations"
linewidth = 2
labelsize = 12
labelticksize = 10
legendfontsize = 10
xlims = (1, 100)
ylims = nothing
inf_replacement = 500.0

# Colors per model (same color across seeds)
model_colors = Dict(
    1 => :blue,   # InvertibleCoupling
    2 => :orange, # Free input (no hold)
    3 => :green,  # Piecewise-constant
)

# ---------------------------
# Helpers
# ---------------------------
function load_result_vectors(path; inf_replacement::Real=500.0, filter_nonfinite::Bool=true)
    data = JLD2.load(path, "result")
    v1 = Float64[]
    v2 = Float64[]
    v3 = Float64[]
    for row in data
        length(row) >= 3 || continue
        vals = (Float64(row[1]), Float64(row[2]), Float64(row[3]))
        fixed = map(v -> isfinite(v) ? v : Float64(inf_replacement), vals)
        push!(v1, fixed[1])
        push!(v2, fixed[2])
        push!(v3, fixed[3])
    end
    if filter_nonfinite
        v1 = filter(isfinite, v1)
        v2 = filter(isfinite, v2)
        v3 = filter(isfinite, v3)
    end
    return (v1, v2, v3)
end

function parse_seed(path)
    m = match(r"Seed(\d+)", basename(path))
    return m === nothing ? "?" : m.captures[1]
end

# ---------------------------
# Build datasets/labels/colors
# ---------------------------
all_data = Vector{Vector{Float64}}()
labels = String[]
colors = Symbol[]

model_labels = Dict(
    1 => "NSIN",
    2 => "Untransformed free input",
    3 => "Untransformed constant hold",
)
files = sort(filter(f -> occursin(filename_filter, f), readdir(results_dir; join=true)))
length(files) > 0 || error("No result files matched in $results_dir for $algorithm")

for file in files
    seed = parse_seed(file)
    v1, v2, v3 = load_result_vectors(file; inf_replacement=inf_replacement)
    data_by_model = Dict(1 => v1, 2 => v2, 3 => v3)
    for model_id in 1:3
        push!(all_data, data_by_model[model_id])
        push!(labels, "$(model_labels[model_id]) Seed $(seed) $(String(algorithm))")
        push!(colors, model_colors[model_id])
    end
end

plt = quantile_plot(all_data...;
    labels=labels,
    legend=show_legend,
    xlabel=xlabel,
    ylabel=ylabel,
    xlims=xlims,
    ylims=ylims === nothing ? :auto : ylims,
    colors=colors,
    linewidth=linewidth,
    labelsize=labelsize,
    ticklabelsize=labelticksize,
    legendfontsize=legendfontsize,
    quartile_levels=[0.25, 0.5, 0.75],
    quartile_lines=false,
    quartile_print=true,
)

mkpath(output_dir)
savefig(plt, output_file)
println("Saved: ", output_file)

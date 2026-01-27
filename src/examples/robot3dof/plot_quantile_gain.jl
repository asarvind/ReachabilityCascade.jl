using JLD2
using Plots
using ReachabilityCascade: quantile_plot

# ---------------------------
# Editable settings
# ---------------------------
results_dir = joinpath(pwd(), "data", "robotarm", "results")
# Choose algorithm: :BOBYQA or :SLSQP
algorithm = :BOBYQA
# Choose comparison: :free (untransformed free input) or :pwc (constant hold)
comparison = :pwc
# Gain definition: :ratio uses (baseline / nsin), :diff uses (baseline - nsin)
# Ratio > 1 means NSIN uses fewer evals than baseline.
gain_mode = :diff

# Output location
output_dir = "/Users/adimoolamarvind/main/PaperSubs/ICML2026ArvindSub/figures/robotarm"
output_file = joinpath(output_dir, "robotarm_gain_$(algorithm)_$(comparison).pdf")

# Plot appearance
show_legend = false
xlabel = "% quantile"
ylabel = "Gain in Evaluations"
linewidth = 2.5
labelsize = 12
labelticksize = 10
legendfontsize = 10
xlims = (1, 100)
ylims = nothing
inf_replacement = 500.0

# Colors per seed
seed_colors = Dict(
    2 => :blue,
    200 => :blue,
    2000 => :blue,
)
seed_linestyles = Dict(
    2 => :solid,
    200 => :dash,
    2000 => :dot,
)

# ---------------------------
# Helpers
# ---------------------------
function load_gain_vector(path; comparison::Symbol, inf_replacement::Real=500.0)
    data = JLD2.load(path, "result")
    gains = Float64[]
    for row in data
        length(row) >= 3 || continue
        nsin_raw = Float64(row[1])
        baseline_raw = comparison == :free ? Float64(row[2]) : Float64(row[3])
        nsin_inf = !isfinite(nsin_raw)
        baseline_inf = !isfinite(baseline_raw)
        (nsin_inf && baseline_inf) && continue
        nsin = nsin_inf ? Float64(inf_replacement) : nsin_raw
        baseline = baseline_inf ? Float64(inf_replacement) : baseline_raw
        if gain_mode == :ratio
            push!(gains, baseline / nsin)
        elseif gain_mode == :diff
            push!(gains, baseline - nsin)
        else
            throw(ArgumentError("gain_mode must be :ratio or :diff"))
        end
    end
    return gains
end

function parse_seed(path)
    m = match(r"Seed(\d+)", basename(path))
    return m === nothing ? missing : parse(Int, m.captures[1])
end

# ---------------------------
# Build datasets/labels/colors
# ---------------------------
algo_tag = algorithm == :BOBYQA ? "AlgoBOBYQA" : "AlgoSLSQP"
filename_filter = Regex("Seed(2|200|2000)" * algo_tag * "\\.jld2\$")

files = sort(filter(f -> occursin(filename_filter, f), readdir(results_dir; join=true)))
length(files) > 0 || error("No result files matched in $results_dir")

all_data = Vector{Vector{Float64}}()
labels = String[]
colors = Symbol[]
linestyles = Symbol[]

for file in files
    seed = parse_seed(file)
    seed === missing && continue
    gain = load_gain_vector(file; comparison=comparison, inf_replacement=inf_replacement)

    push!(all_data, gain)
    push!(labels, "Seed $(seed)")
    push!(colors, get(seed_colors, seed, :blue))
    push!(linestyles, get(seed_linestyles, seed, :solid))
end

plt = quantile_plot(all_data...;
    labels=labels,
    legend=show_legend,
    xlabel=xlabel,
    ylabel=ylabel,
    xlims=xlims,
    ylims=ylims === nothing ? :auto : ylims,
    colors=colors,
    linestyles=linestyles,
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

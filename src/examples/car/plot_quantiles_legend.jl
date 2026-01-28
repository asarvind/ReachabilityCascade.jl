using Plots
using Measures

# ---------------------------
# Editable settings
# ---------------------------
# Output location (PaperSubs / ICML submission)
output_dir = "/Users/adimoolamarvind/main/PaperSubs/ICML2026ArvindSub/figures/car"

# Legend appearance
legendfontsize = 30
linewidth = 6.0
legend_size = (650, 170)

# Legend entries (label, color, filename suffix)
legend_items = [
    (" NAIN", :blue, "nain"),
    (" Untransformed\n constant hold", :green, "pwc"),
    (" Untransformed\n free", :orange, "free"),
]

# ---------------------------
# Legend-only plots (one per file)
# ---------------------------
for (label, color, suffix) in legend_items
    plt = plot(
        legend=:top,
        legend_background_color=:transparent,
        legend_foreground_color=:transparent,
        framestyle=:none,
        axis=nothing,
        grid=false,
        left_margin=0mm,
        right_margin=0mm,
        top_margin=0mm,
        bottom_margin=0mm,
        legendfontsize=legendfontsize,
        size=legend_size,
    )

    plot!(plt, [NaN], [NaN],
        label=label,
        color=color,
        linewidth=linewidth,
    )

    mkpath(output_dir)
    output_file = joinpath(output_dir, "car_legend_$(suffix).pdf")
    savefig(plt, output_file)
    println("Saved: ", output_file)
end

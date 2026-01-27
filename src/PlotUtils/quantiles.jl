using Plots

"""
    quantile_plot(datasets...; kwargs...) -> plot

Plot sorted values against percent-quantile (1..100) for each dataset.

Each dataset must be an `AbstractVector{<:Real}`.

Keyword arguments:
- `labels=nothing`: Vector of labels (same length as datasets) or `nothing`.
- `legend::Bool=true`: Toggle legend display.
- `xlabel::AbstractString="% quantile"`: X-axis label.
- `ylabel::AbstractString="value"`: Y-axis label.
- `xlims=nothing`, `ylims=nothing`: Axis limits.
- `colors=nothing`: Vector of colors (same length as datasets) or `nothing`.
- `linewidth::Real=2`: Line width.
- `labelsize::Integer=10`: Axis label font size.
- `ticklabelsize::Integer=9`: Tick label font size.
- `legendfontsize::Integer=9`: Legend font size.
- `quartile_levels=nothing`: Vector of quantile levels (e.g. `[0.25, 0.5, 0.75]`) or `nothing`.
- `quartile_linestyle=:dash`: Line style for quartile guides.
- `quartile_linewidth::Real=1`: Line width for quartile guides.
- `quartile_alpha::Real=0.6`: Alpha for quartile guides.
- `quartile_ticks::Bool=true`: Add quartile y-values to tick labels.
"""
function quantile_plot(datasets...;
                       labels=nothing,
                       legend::Bool=true,
                       xlabel::AbstractString="% quantile",
                       ylabel::AbstractString="value",
                       xlims=nothing,
                       ylims=nothing,
                       colors=nothing,
                       linestyles=nothing,
                       linewidth::Real=2,
                       labelsize::Integer=10,
                       ticklabelsize::Integer=9,
                       legendfontsize::Integer=9,
                       quartile_levels=nothing,
                       quartile_lines::Bool=true,
                       quartile_linestyle=:dash,
                       quartile_linewidth::Real=1,
                       quartile_alpha::Real=0.6,
                       quartile_ticks::Bool=false,
                       quartile_print::Bool=false)
    nsets = length(datasets)
    nsets > 0 || throw(ArgumentError("at least one dataset is required"))

    if labels !== nothing
        length(labels) == nsets || throw(ArgumentError("labels must match number of datasets"))
    end

    if colors !== nothing
        length(colors) == nsets || throw(ArgumentError("colors must match number of datasets"))
    end
    if linestyles !== nothing
        length(linestyles) == nsets || throw(ArgumentError("linestyles must match number of datasets"))
    end

    plt = nothing
    all_quartile_vals = Float64[]
    quartile_reports = Vector{Any}()
    for (i, data) in enumerate(datasets)
        data isa AbstractVector || throw(ArgumentError("dataset $i must be a vector"))
        isempty(data) && throw(ArgumentError("dataset $i is empty"))
        y = sort(collect(Float64, data))
        n = length(y)
        x = n == 1 ? [100.0] : collect(range(1.0, 100.0, length=n))

        lbl = labels === nothing ? "" : labels[i]
        color = colors === nothing ? nothing : colors[i]
        linestyle = linestyles === nothing ? :solid : linestyles[i]

        if plt === nothing
            plt = plot(x, y;
                label=lbl,
                color=color,
                linestyle=linestyle,
                legend=legend,
                xlabel=xlabel,
                ylabel=ylabel,
                xlims=xlims,
                ylims=ylims,
                linewidth=linewidth,
                xguidefontsize=labelsize,
                yguidefontsize=labelsize,
                xtickfontsize=ticklabelsize,
                ytickfontsize=ticklabelsize,
                legendfontsize=legendfontsize,
            )
        else
            plot!(plt, x, y;
                label=lbl,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        end

        if quartile_levels !== nothing
            qvals = similar(collect(Float64, quartile_levels))
            for (qi, q) in enumerate(quartile_levels)
                if n == 1
                    qvals[qi] = y[1]
                else
                    xq = 100.0 * Float64(q)
                    idx = round(Int, 1 + (xq - 1) * (n - 1) / 99)
                    idx = clamp(idx, 1, n)
                    qvals[qi] = y[idx]
                end
            end
            append!(all_quartile_vals, qvals)
            report_label = lbl == "" ? "dataset $(i)" : lbl
            push!(quartile_reports, (label=report_label, levels=collect(quartile_levels), values=collect(qvals)))
            if quartile_lines
                xq = 100.0 .* collect(quartile_levels)
                vline!(plt, xq;
                    color=color,
                    linestyle=quartile_linestyle,
                    linewidth=quartile_linewidth,
                    alpha=quartile_alpha,
                    label="",
                )
                hline!(plt, qvals;
                    color=color,
                    linestyle=quartile_linestyle,
                    linewidth=quartile_linewidth,
                    alpha=quartile_alpha,
                    label="",
                )
            end
        end
    end

    if quartile_levels !== nothing && quartile_ticks
        try
            ticks = Plots.yticks(plt)
            tick_vals = ticks[1]
            new_vals = sort(unique(vcat(collect(Float64, tick_vals), all_quartile_vals)))
            yticks!(plt, (new_vals, string.(new_vals)))
        catch
            new_vals = sort(unique(all_quartile_vals))
            yticks!(plt, (new_vals, string.(new_vals)))
        end
    end

    if quartile_levels !== nothing && quartile_print
        for report in quartile_reports
            println("quartiles ", report.label, " levels=", report.levels, " values=", report.values)
        end
    end

    return plt
end

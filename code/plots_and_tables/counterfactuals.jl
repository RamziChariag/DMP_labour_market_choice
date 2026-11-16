############################################################
# counterfactuals.jl — the figures of the counterfactual exercises (RoySearch)
#
# LIBRARY FILE — loaded by code/counterfactuals/counterfactuals_main.jl, which owns the
# data; every figure is written to the exercise's own directory (out_counterfactuals).
#
#   plot_welfare_flow(df, f0, f1, months, file, crisis)     net social flow / A on the path
#   plot_incidence_heatmap(x, ratio, pop, file, crisis)     date-0 value ratio by type
#   plot_held_margin(df, months, file, crisis)              unemployment, margin held or not
#   plot_training_cost(df, file, crisis)                    welfare and unemployment by scale
############################################################

function _cf_theme!()
    gr()
    theme(:default)
    default(fontfamily = "Computer Modern", framestyle = :box, titlefontsize = 11,
            guidefontsize = 10, tickfontsize = 8, legendfontsize = 8, linewidth = 1.8,
            grid = true, gridalpha = 0.05, left_margin = 7Plots.mm, bottom_margin = 5Plots.mm)
end

const _CF_EST  = :steelblue
const _CF_ALT  = :firebrick
const _CF_REF  = :gray40

_crisis_label(crisis) = crisis === :fc ? "financial crisis" : crisis === :covid ? "COVID" : String(crisis)

"""
    plot_welfare_flow(df, f0, f1, months, file, crisis)

The net social flow divided by A along the path (`df` from run_welfare), against the
baseline (`f0`) and crisis (`f1`) steady-state flows divided by their own A, with the crisis
window shaded.
"""
function plot_welfare_flow(df, f0::Float64, f1::Float64, months::Real, file::AbstractString, crisis)
    _cf_theme!()
    k = df.t .<= max(4 * months, 120.0)
    p = plot(df.t[k], df.flow_net_over_A[k]; color = _CF_EST, label = "path",
             xlabel = "months since the switch", ylabel = "net social flow / A",
             title = "Social flow, " * _crisis_label(crisis), size = (720, 420))
    vspan!(p, [0.0, months]; color = _CF_REF, alpha = 0.08, label = "crisis window")
    hline!(p, [f0]; color = _CF_REF, linestyle = :dash, label = "baseline steady state")
    hline!(p, [f1]; color = _CF_ALT, linestyle = :dot, label = "crisis steady state")
    savefig(p, file)
    return file
end

"""
    plot_incidence_heatmap(x, ratio, pop, file, crisis)

The type-average ratio (V_0/A_1)/(V_b/A_0) − 1 over the ability nodes, a_U across and a_S
up; types with no population are blank.
"""
function plot_incidence_heatmap(x::AbstractVector, ratio::AbstractMatrix, pop::AbstractMatrix,
                                file::AbstractString, crisis)
    _cf_theme!()
    z = permutedims(ifelse.(pop .> 0.0, ratio, NaN))       # rows a_S, columns a_U
    lim = maximum(abs, filter(isfinite, z))
    p = heatmap(x, x, z; color = :RdBu, clims = (-lim, lim), xlabel = "a_U", ylabel = "a_S",
                colorbar_title = "(V₀/A₁)/(V_b/A₀) − 1",
                title = "Change in value at date 0, " * _crisis_label(crisis), size = (640, 540))
    savefig(p, file)
    return file
end

"""
    plot_held_margin(df, months, file, crisis)

Unemployment rates by market along the estimated path and the path with the education
margin held at the baseline (`df` from run_held_margin).
"""
function plot_held_margin(df, months::Real, file::AbstractString, crisis)
    _cf_theme!()
    k = df.t .<= max(4 * months, 120.0)
    panels = map((("ur_U", "unskilled"), ("ur_S", "skilled"), ("ur_total", "total"))) do (s, lab)
        q = plot(df.t[k], df[k, Symbol(s, "_estimated")]; color = _CF_EST, label = "estimated",
                 xlabel = "months", ylabel = "unemployment rate", title = lab)
        plot!(q, df.t[k], df[k, Symbol(s, "_held")]; color = _CF_ALT, linestyle = :dash,
              label = "τ and d held")
        vspan!(q, [0.0, months]; color = _CF_REF, alpha = 0.08, label = "")
        q
    end
    p = plot(panels...; layout = (1, 3), size = (1100, 360),
             plot_title = "Education margin held at the baseline, " * _crisis_label(crisis))
    savefig(p, file)
    return file
end

"""
    plot_training_cost(df, file, crisis)

Welfare (months of baseline mean unskilled output, relative to the estimated cost) and
unemployment over the crisis window by training-cost scale, on the crisis and the
no-crisis path (`df` from run_training_cost).
"""
function plot_training_cost(df, file::AbstractString, crisis)
    _cf_theme!()
    pw = plot(; xlabel = "training cost scale", ylabel = "ΔW, months of baseline output",
              title = "Welfare")
    pu = plot(; xlabel = "training cost scale", ylabel = "unemployment rate, crisis window",
              title = "Unemployment")
    for (kind, col) in (("crisis", _CF_EST), ("no_crisis", _CF_REF))
        sub = sort(df[df.path .== kind, :], :scale)
        W1 = sub.W_months_base_piU[findfirst(==(1.0), sub.scale)]
        plot!(pw, sub.scale, sub.W_months_base_piU .- W1; color = col, marker = :circle,
              label = replace(kind, "_" => " "))
        plot!(pu, sub.scale, sub.ur_U_window; color = col, marker = :circle,
              label = "unskilled, " * replace(kind, "_" => " "))
        plot!(pu, sub.scale, sub.ur_S_window; color = col, marker = :diamond, linestyle = :dash,
              label = "skilled, " * replace(kind, "_" => " "))
    end
    p = plot(pw, pu; layout = (1, 2), size = (1000, 400),
             plot_title = "Training cost scaled from date 0, " * _crisis_label(crisis))
    savefig(p, file)
    return file
end

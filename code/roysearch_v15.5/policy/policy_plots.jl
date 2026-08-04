############################################################
# policy_plots.jl — Tables and figures for Section 6
#                    (education-subsidy counterfactuals)
#
# Contains:
#   print_policy_table    — console summary
#   write_policy_latex    — LaTeX table for the paper
#   write_policy_csv      — CSV for downstream use
#   fig_policy_bar        — bar chart comparing policies
#   fig_policy_cutoff     — training cutoff shift
#   fig_policy_welfare    — welfare at selected quantiles
#   make_all_policy_plots — master function
############################################################

# NOTE: `using Plots`, `using LaTeXStrings`, `using Printf`,
# `using CSV`, `using DataFrames` are loaded in policy_main.jl.


# ════════════════════════════════════════════════════════════
# Console summary
# ════════════════════════════════════════════════════════════

"""
    print_policy_table(table::PolicyTable)

Print a transposed comparison table to stdout.
Rows = outcome variables, columns = baseline + policy intensities.
"""
# Row definitions shared by all console tables
const _CONSOLE_ROW_SPECS = [
    ("x̄",             :x_bar),
    ("ur",             :ur_total),
    ("Skilled share",  :skilled_share),
    ("Training share", :training_share),
    ("θ_U",            :θU),
    ("θ_S",            :θS),
    ("w̄_U",           :mean_wage_U),
    ("w̄_S",           :mean_wage_S),
    ("f_U",            :f_U),
    ("f_S",            :f_S),
    ("δ_U",            :sep_rate_U),
    ("δ_S",            :sep_rate_S),
]

"""
    _print_one_panel(all_results, col_labels; title)

Print one transposed panel to stdout.
"""
function _print_one_panel(all_results, col_labels; title::String = "")
    nc = length(all_results)
    max_label = maximum(length.(col_labels))
    lw = 16
    cw = max(8, max_label + 1)
    tw = lw + 2 + nc * (cw + 1) + 1

    if !isempty(title)
        @printf("\n%s\n", "═"^tw)
        @printf("  %s\n", title)
    end
    @printf("%s\n", "═"^tw)

    @printf("  %-*s", lw, "")
    for cl in col_labels
        @printf("  %*s", cw, cl)
    end
    @printf("\n")
    @printf("%s\n", "─"^tw)

    for (label, field) in _CONSOLE_ROW_SPECS
        @printf("  %-*s", lw, label)
        for r in all_results
            @printf("  %*.4f", cw, getfield(r, field))
        end
        @printf("\n")
    end

    @printf("%s\n", "─"^tw)
    @printf("  %-*s", lw, "Converged")
    for r in all_results
        @printf("  %*s", cw, r.converged ? "✓" : "✗")
    end
    @printf("\n")
    @printf("%s\n", "═"^tw)
end


function print_policy_table(table::PolicyTable)
    results = table.results
    baseline = results[1]
    non_base = filter(r -> r.spec.policy != :baseline, results)

    # Detect if this is an OO table that needs splitting
    has_bUS = any(r -> r.spec.policy == :bUS, non_base)

    if has_bUS
        lbl = table.baseline_label

        # Panel A: bU only
        pol_bU = filter(r -> r.spec.policy == :bU, non_base)
        la = ["Base"]; ra = [baseline]
        for r in pol_bU;  push!(la, r.spec.label); push!(ra, r)  end
        _print_one_panel(ra, la; title = "Panel A: bU raised  —  $lbl")

        # Panel B: bS only
        pol_bS = filter(r -> r.spec.policy == :bS, non_base)
        lb = ["Base"]; rb = [baseline]
        for r in pol_bS;  push!(lb, r.spec.label); push!(rb, r)  end
        _print_one_panel(rb, lb; title = "Panel B: bS raised  —  $lbl")

        # Panel C: bU and bS together
        pol_bUS = filter(r -> r.spec.policy == :bUS, non_base)
        lc = ["Base"]; rc = [baseline]
        for r in pol_bUS;  push!(lc, r.spec.label); push!(rc, r)  end
        _print_one_panel(rc, lc; title = "Panel C: bU and bS raised jointly  —  $lbl")
    else
        # Single table (education subsidies or anything else)
        col_labels = ["Base"]
        for r in non_base;  push!(col_labels, r.spec.label)  end
        all_results = vcat([baseline], non_base)
        _print_one_panel(all_results, col_labels;
            title = "Policy Counterfactuals  —  $(table.baseline_label)")
    end
end


# ════════════════════════════════════════════════════════════
# LaTeX table
# ════════════════════════════════════════════════════════════

# ── Shared row definitions for all LaTeX tables ──────────
const _LATEX_ROW_SPECS = [
    ("\\(\\bar{x}\\)",       :x_bar),
    ("\\(ur\\)",              :ur_total),
    ("Skilled share",         :skilled_share),
    ("Training share",        :training_share),
    ("\\(\\theta_U\\)",      :θU),
    ("\\(\\theta_S\\)",      :θS),
    ("\\(\\bar{w}_U\\)",     :mean_wage_U),
    ("\\(\\bar{w}_S\\)",     :mean_wage_S),
    ("\\(f_U\\)",             :f_U),
    ("\\(f_S\\)",             :f_S),
    ("\\(\\delta_U\\)",      :sep_rate_U),
    ("\\(\\delta_S\\)",      :sep_rate_S),
]


"""
    _latex_col_header(spec) → String

Build a math-mode LaTeX column header from a PolicySpec.
Format: `\$b_U +10\\%\$`  (parameter first, then intensity).
"""
function _latex_col_header(spec::PolicySpec)
    pct = round(Int, 100 * spec.intensity)
    p = spec.policy
    if     p == :A;   return "\$b_T +$(pct)\\%\$"
    elseif p == :B;   return "\$c(x) \\text{-}$(pct)\\%\$"
    elseif p == :bU;  return "\$b_U +$(pct)\\%\$"
    elseif p == :bS;  return "\$b_S +$(pct)\\%\$"
    elseif p == :bUS; return "\$b_U,b_S +$(pct)\\%\$"
    else              return "Baseline"
    end
end


"""
    _write_one_latex_table(lines, all_results, col_heads)

Append the tabular body (tabular + toprule…bottomrule) to `lines`.
"""
function _write_one_latex_table!(lines, all_results, col_heads)
    nc = length(all_results)
    push!(lines, "\\begin{tabular}{c" * repeat(" c", nc) * "}")
    push!(lines, "\\toprule")

    # Header row
    hdr = " "
    for ch in col_heads
        hdr *= " & " * ch
    end
    hdr *= " \\\\"
    push!(lines, hdr)
    push!(lines, "\\midrule")

    # Data rows
    for (label, field) in _LATEX_ROW_SPECS
        row = label
        for r in all_results
            val = getfield(r, field)
            row *= @sprintf(" & %.4f", val)
        end
        row *= " \\\\"
        push!(lines, row)
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
end


"""
    _write_standalone_latex_file(path, caption, label, all_results, col_heads; notes)

Write a single self-contained LaTeX table environment to `path`.
"""
function _write_standalone_latex_file(
    path        :: String,
    caption     :: String,
    label       :: String,
    all_results,
    col_heads;
    notes :: String = "",
)
    lines = String[]
    push!(lines, "\\begin{table}[htbp]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{$caption}")
    push!(lines, "\\label{$label}")
    push!(lines, "\\small")
    _write_one_latex_table!(lines, all_results, col_heads)
    if !isempty(notes)
        push!(lines, "\\\\[2pt]")
        push!(lines, "\\footnotesize")
        push!(lines, notes)
    end
    push!(lines, "\\end{table}")

    open(path, "w") do io
        for line in lines;  println(io, line)  end
    end
    @printf("  Saved LaTeX table → %s\n", path)
end


"""
    write_policy_latex(table, path)

Write LaTeX tables for the policy exercise.  Each panel becomes its own
file so you can `\\input` them independently and let LaTeX place them
wherever they fit.

The `path` argument is used as the base: for a single-panel exercise the
file is written to `path` directly; for multi-panel exercises the files
are `path_a.tex`, `path_b.tex`, `path_c.tex`.
"""
function write_policy_latex(table::PolicyTable, path::String)
    results = table.results
    baseline = results[1]
    non_base = filter(r -> r.spec.policy != :baseline, results)

    policies_present = unique([r.spec.policy for r in non_base])
    is_educ = any(p -> p in (:A, :B), policies_present)
    is_oo   = any(p -> p in (:bU, :bS, :bUS), policies_present)

    suffix_label = replace(lowercase(table.baseline_label), " " => "_")
    base_path    = replace(path, ".tex" => "")

    educ_notes = "\\textit{Notes:} " *
        "Policy~A raises \\(b_T\\) by the stated percentage. " *
        "Policy~B reduces \\(e^c\\) (and hence \\(c(x) = (1-x)e^{c-x}\\)) by the stated percentage. " *
        "\\(^{\\dagger}\\) indicates non-convergence."

    oo_notes = "\\textit{Notes:} " *
        "Each column raises the indicated flow payoff by the stated percentage, " *
        "holding all other parameters at their estimated baseline values. " *
        "\\(^{\\dagger}\\) indicates non-convergence."

    if is_oo
        pol_bU  = filter(r -> r.spec.policy == :bU,  non_base)
        pol_bS  = filter(r -> r.spec.policy == :bS,  non_base)
        pol_bUS = filter(r -> r.spec.policy == :bUS, non_base)
        lbl = table.baseline_label

        # Panel A: bU
        ha = ["Baseline"]; ra = [baseline]
        for r in pol_bU;  push!(ha, _latex_col_header(r.spec)); push!(ra, r)  end
        _write_standalone_latex_file(
            base_path * "_a.tex",
            "Outside options: \\(b_U\\) raised — $lbl",
            "tab:oo_bU_$(suffix_label)",
            ra, ha; notes = oo_notes)

        # Panel B: bS
        hb = ["Baseline"]; rb = [baseline]
        for r in pol_bS;  push!(hb, _latex_col_header(r.spec)); push!(rb, r)  end
        _write_standalone_latex_file(
            base_path * "_b.tex",
            "Outside options: \\(b_S\\) raised — $lbl",
            "tab:oo_bS_$(suffix_label)",
            rb, hb; notes = oo_notes)

        # Panel C: bU,bS
        hc = ["Baseline"]; rc = [baseline]
        for r in pol_bUS;  push!(hc, _latex_col_header(r.spec)); push!(rc, r)  end
        _write_standalone_latex_file(
            base_path * "_c.tex",
            "Outside options: \\(b_U\\) and \\(b_S\\) raised jointly — $lbl",
            "tab:oo_bUS_$(suffix_label)",
            rc, hc; notes = oo_notes)

    elseif is_educ
        all_res = vcat([baseline], non_base)
        heads   = ["Baseline"]
        for r in non_base;  push!(heads, _latex_col_header(r.spec))  end
        _write_standalone_latex_file(
            path,
            "Education-subsidy counterfactuals — $(table.baseline_label)",
            "tab:educ_$(suffix_label)",
            all_res, heads; notes = educ_notes)

    else
        all_res = vcat([baseline], non_base)
        heads   = ["Baseline"]
        for r in non_base;  push!(heads, _latex_col_header(r.spec))  end
        _write_standalone_latex_file(
            path,
            "Policy counterfactuals — $(table.baseline_label)",
            "tab:policy_$(suffix_label)",
            all_res, heads)
    end
end


# ════════════════════════════════════════════════════════════
# CSV export
# ════════════════════════════════════════════════════════════

"""
    write_policy_csv(table, path)

Write a CSV file with all policy outcomes for downstream analysis.
"""
function write_policy_csv(table::PolicyTable, path::String)
    rows = []
    for pr in table.results
        push!(rows, (
            baseline     = table.baseline_label,
            policy       = string(pr.spec.policy),
            intensity    = pr.spec.intensity,
            label        = pr.spec.label,
            converged    = pr.converged,
            x_bar        = pr.x_bar,
            ur_U         = pr.ur_U,
            ur_S         = pr.ur_S,
            ur_total     = pr.ur_total,
            skilled_share = pr.skilled_share,
            training_share = pr.training_share,
            theta_U      = pr.θU,
            theta_S      = pr.θS,
            f_U          = pr.f_U,
            f_S          = pr.f_S,
            sep_rate_U   = pr.sep_rate_U,
            sep_rate_S   = pr.sep_rate_S,
            ee_rate_S    = pr.ee_rate_S,
            mean_wage_U  = pr.mean_wage_U,
            mean_wage_S  = pr.mean_wage_S,
            wage_premium = pr.wage_premium,
            UU_x25       = pr.UU_x25,
            UU_x50       = pr.UU_x50,
            UU_x75       = pr.UU_x75,
            US_x25       = pr.US_x25,
            US_x50       = pr.US_x50,
            US_x75       = pr.US_x75,
        ))
    end
    df = DataFrame(rows)
    CSV.write(path, df)
    @printf("  Saved CSV → %s\n", path)
end


# ════════════════════════════════════════════════════════════
# Figures
# ════════════════════════════════════════════════════════════

# ── Shared theme (matches existing code) ─────────────────
function _set_policy_theme!()
    gr()
    theme(:default)
    default(
        fontfamily          = "Computer Modern",
        framestyle          = :box,
        titlefontsize       = 11,
        guidefontsize       = 10,
        tickfontsize        = 8,
        legendfontsize      = 8,
        linewidth           = 1.8,
        grid                = true,
        gridalpha           = 0.05,
        left_margin         = 7Plots.mm,
    )
end

const _POL_A_COL = :steelblue
const _POL_B_COL = :firebrick
const _BASE_COL  = :gray60


"""
    fig_policy_bar(table; variable, ylabel, title_str) → Plot

Grouped bar chart comparing Policy A and Policy B across intensities
for one scalar variable.
"""
function fig_policy_bar(
    table     :: PolicyTable;
    variable  :: Symbol,
    ylabel    :: AbstractString = "",
    title_str :: AbstractString = "",
)
    results = table.results
    baseline = results[1]
    base_val = getfield(baseline, variable)

    # Separate policies
    pol_A = filter(r -> r.spec.policy == :A, results)
    pol_B = filter(r -> r.spec.policy == :B, results)

    n = max(length(pol_A), length(pol_B))
    x_labels = [string(round(Int, 100 * pol_A[i].spec.intensity)) * "%" for i in 1:n]

    vals_A = [getfield(r, variable) - base_val for r in pol_A]
    vals_B = [getfield(r, variable) - base_val for r in pol_B]

    p = groupedbar(
        hcat(vals_A, vals_B),
        bar_position = :dodge,
        bar_width    = 0.35,
        xticks       = (1:n, x_labels),
        xlabel       = "Subsidy intensity",
        ylabel       = ylabel,
        title        = title_str,
        label        = ["Policy A (↑ bT)" "Policy B (↓ c(x))"],
        color        = [_POL_A_COL _POL_B_COL],
        legend       = :topright,
        fillalpha    = 0.85,
    )
    hline!(p, [0.0]; color = :black, ls = :dash, lw = 0.8, label = "")
    return p
end


"""
    fig_policy_overview(table) → Plot

2×2 panel: Δ training cutoff, Δ skilled share, Δ unemployment, Δ skilled wages
"""
function fig_policy_overview(table::PolicyTable)
    _set_policy_theme!()

    p1 = fig_policy_bar(table; variable = :x_bar,
         ylabel = L"\Delta \bar{x}", title_str = "Training cutoff shift")
    p2 = fig_policy_bar(table; variable = :skilled_share,
         ylabel = L"\Delta", title_str = "Skilled share change")
    p3 = fig_policy_bar(table; variable = :ur_total,
         ylabel = L"\Delta", title_str = "Unemployment rate change")
    p4 = fig_policy_bar(table; variable = :mean_wage_S,
         ylabel = L"\Delta \bar{w}_S", title_str = "Skilled wage change")

    plot(p1, p2, p3, p4; layout = (2, 2), size = (1000, 700), margin = 5Plots.mm)
end


"""
    fig_policy_tightness(table) → Plot

1×2 panel: Δ θ_U and Δ θ_S
"""
function fig_policy_tightness(table::PolicyTable)
    _set_policy_theme!()

    p1 = fig_policy_bar(table; variable = :θU,
         ylabel = L"\Delta \theta_U", title_str = "Unskilled tightness change")
    p2 = fig_policy_bar(table; variable = :θS,
         ylabel = L"\Delta \theta_S", title_str = "Skilled tightness change")

    plot(p1, p2; layout = (1, 2), size = (900, 380), margin = 5Plots.mm)
end


"""
    fig_policy_welfare(table) → Plot

Welfare (U_U and U_S) at the median ability level, by experiment.
"""
function fig_policy_welfare(table::PolicyTable)
    _set_policy_theme!()

    results = table.results
    baseline = results[1]

    # Extract Δ welfare at median for each non-baseline experiment
    labels = String[]
    dUU    = Float64[]
    dUS    = Float64[]

    for pr in results[2:end]
        push!(labels, pr.spec.label)
        push!(dUU, pr.UU_x50 - baseline.UU_x50)
        push!(dUS, pr.US_x50 - baseline.US_x50)
    end

    n = length(labels)

    p = groupedbar(
        hcat(dUU, dUS),
        bar_position = :dodge,
        bar_width    = 0.35,
        xticks       = (1:n, labels),
        xrotation    = 20,
        xlabel       = "",
        ylabel       = L"\Delta U(x_{50})",
        title        = "Welfare change at median ability",
        label        = ["Unskilled U_U" "Skilled U_S"],
        color        = [:mediumpurple :seagreen],
        legend       = :topright,
        fillalpha    = 0.85,
    )
    hline!(p, [0.0]; color = :black, ls = :dash, lw = 0.8, label = "")
    plot(p; size = (800, 420), margin = 5Plots.mm, bottom_margin = 10Plots.mm)
end


"""
    fig_policy_rates(table) → Plot

2×2 panel: Δ f_U, Δ f_S, Δ sep_U, Δ sep_S
"""
function fig_policy_rates(table::PolicyTable)
    _set_policy_theme!()

    p1 = fig_policy_bar(table; variable = :f_U,
         ylabel = L"\Delta f_U", title_str = "Unskilled JFR change")
    p2 = fig_policy_bar(table; variable = :f_S,
         ylabel = L"\Delta f_S", title_str = "Skilled JFR change")
    p3 = fig_policy_bar(table; variable = :sep_rate_U,
         ylabel = L"\Delta \delta_U", title_str = "Unskilled sep. rate change")
    p4 = fig_policy_bar(table; variable = :sep_rate_S,
         ylabel = L"\Delta \delta_S", title_str = "Skilled sep. rate change")

    plot(p1, p2, p3, p4; layout = (2, 2), size = (1000, 700), margin = 5Plots.mm)
end


# ════════════════════════════════════════════════════════════
# Master function
# ════════════════════════════════════════════════════════════

"""
    make_all_policy_outputs(table; output_dir)

Generate all tables and figures for the policy counterfactual section.
"""
function make_all_policy_outputs(
    table :: PolicyTable;
    output_dir :: String = "output",
)
    plots_dir  = joinpath(output_dir, "plots")
    tables_dir = joinpath(output_dir, "tables")
    mkpath(plots_dir)
    mkpath(tables_dir)

    suffix = replace(lowercase(table.baseline_label), " " => "_")

    _set_policy_theme!()

    # ── Console ───────────────────────────────────────────
    print_policy_table(table)

    # ── LaTeX table ───────────────────────────────────────
    write_policy_latex(table, joinpath(tables_dir, "policy_table_$(suffix).tex"))

    # ── CSV ───────────────────────────────────────────────
    write_policy_csv(table, joinpath(tables_dir, "policy_results_$(suffix).csv"))

    # ── Figures (only for education-subsidy tables with :A/:B) ────
    has_AB = any(r -> r.spec.policy in (:A, :B), table.results)
    if has_AB
        figures = [
            ("policy_overview_$(suffix)",   fig_policy_overview(table)),
            ("policy_tightness_$(suffix)",  fig_policy_tightness(table)),
            ("policy_welfare_$(suffix)",    fig_policy_welfare(table)),
            ("policy_rates_$(suffix)",      fig_policy_rates(table)),
        ]

        for (name, fig) in figures
            path = joinpath(plots_dir, name * ".png")
            savefig(fig, path)
            @printf("  Saved: %s\n", path)
        end
    end

    println("\nAll policy outputs saved.")
    flush(stdout)
    return nothing
end

############################################################
# plots_and_tables.jl — Post-estimation plots and tables
#                         (PS(x) = P_S·x quadratic variant)
#
# LIBRARY FILE — defines functions; does not auto-execute when
# included.  The supported entry point is transition_main.jl, which
# loads modules once, then calls:
#
#   load_all_smm_bundles!(; scenarios, point)
#   make_model_fit_tables(; suffix = tag)
#   make_parameter_tables(; suffix = tag)
#   make_xbar_table(; suffix = tag)
#   make_model_fit_scatter(; window = …, suffix = tag)
#
# with tag = transition_tag(W_SUFFIX, point): every table and plot ends in it, as the
# transition files do.
#
# The file can also be run standalone (`julia plots_and_tables.jl`),
# in which case it sets up its own paths and runs the full pipeline.
#
# Transition dynamics panels live in transition_panel.jl.
############################################################

# ════════════════════════════════════════════════════════════
# Paths — orchestrator defines these; fall back to script-local
# defaults when invoked standalone.
# ════════════════════════════════════════════════════════════

# This file runs two ways: included by transition_main.jl, which has already defined
# these, or standalone. The guards keep both working. paths.jl is included only when
# nothing has loaded it yet — including it twice would redefine its constants.
if !@isdefined(SCRIPT_DIR);   const SCRIPT_DIR   = @__DIR__;                                 end
@isdefined(ROYSEARCH_PATHS_LOADED) || include(joinpath(SCRIPT_DIR, "..", "paths.jl"))
if !@isdefined(SOLVER_DIR);   const SOLVER_DIR   = joinpath(CODE_DIR, "solver");             end
if !@isdefined(SMM_DIR);      const SMM_DIR      = joinpath(CODE_DIR, "smm");                end
if !@isdefined(PLOTS_DIR);    const PLOTS_DIR    = joinpath(OUTPUT_DIR, "plots");            end
if !@isdefined(TABLES_DIR);   const TABLES_DIR   = joinpath(OUTPUT_DIR, "tables");           end
if !@isdefined(SMM_OUT_DIR);  const SMM_OUT_DIR  = out_estimates();                          end
if !@isdefined(TRANS_OUT_DIR); const TRANS_OUT_DIR = out_transition();                        end
if !@isdefined(DERIVED_DIR);  const DERIVED_DIR  = joinpath(DATA_DIR, "derived");            end

mkpath(PLOTS_DIR)
mkpath(TABLES_DIR)

# ════════════════════════════════════════════════════════════
# Packages
# ════════════════════════════════════════════════════════════
print("Loading plots_and_tables packages... "); flush(stdout)

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Distributions
using FastGaussQuadrature
using Interpolations
using Parameters
using Printf
import Base.Threads: @threads, nthreads
using Optim
using CSV
using DataFrames
using Clustering
using Serialization
using Plots
using LaTeXStrings

println("done."); flush(stdout)

# ════════════════════════════════════════════════════════════
# Load solver / SMM modules — only if the orchestrator has not
# already done so (avoids struct-redefinition errors).
# ════════════════════════════════════════════════════════════

if !@isdefined(solve_model)
    print("Loading solver modules... "); flush(stdout)
    include(joinpath(SOLVER_DIR, "grids.jl"))
    include(joinpath(SOLVER_DIR, "params.jl"))
    include(joinpath(SOLVER_DIR, "unskilled.jl"))
    include(joinpath(SOLVER_DIR, "skilled.jl"))
    include(joinpath(SOLVER_DIR, "solver.jl"))
    include(joinpath(SOLVER_DIR, "equilibrium.jl"))
    println("done."); flush(stdout)
end

if !@isdefined(MOMENT_NAMES)
    print("Loading SMM modules... "); flush(stdout)
    include(joinpath(SMM_DIR, "version.jl"))
    include(joinpath(SMM_DIR, "settings.jl"))
    include(joinpath(SMM_DIR, "moments.jl"))
    include(joinpath(SMM_DIR, "smm_params.jl"))
    include(joinpath(SMM_DIR, "bundle.jl"))
    include(joinpath(SMM_DIR, "smm.jl"))
    println("done."); flush(stdout)
end

# The transition file's types, name and reader.
isdefined(Main, :TransitionResult) ||
    include(joinpath(CODE_DIR, "transition", "transition_params.jl"))


# ============================================================
# Configuration
# ============================================================

# The weighting the estimates were produced under; transition_main.jl has already set it.
if !@isdefined(W_SUFFIX)
    const W_SUFFIX = weighting_suffix(run_weighting())
end

# The four estimation windows (ordered for tables)
WINDOW_PAIRS = [
    (:base_fc,    :crisis_fc,    "FC"),
    (:base_covid, :crisis_covid, "COVID"),
]

ALL_WINDOWS = [:base_fc, :crisis_fc, :base_covid, :crisis_covid]

WINDOW_LABELS = Dict(
    :base_fc      => "Baseline FC",
    :crisis_fc    => "Crisis FC",
    :base_covid   => "Baseline COVID",
    :crisis_covid => "Crisis COVID",
)

# ============================================================
# SMM bundle store — populated by load_all_smm_bundles!()
# ============================================================

const smm_bundles = Dict{Symbol, Any}()

"""
    load_all_smm_bundles!(; scenarios, point, suffix = W_SUFFIX)

Populate the package-level `smm_bundles` Dict (window symbol → bundle) with exactly the
bundles the transition of each scenario was solved at: the two files its transition file at
`point` records (`read_transition`, which stops on a file that records another point), so
the tables report the paths' own points. A window of a scenario without a transition file
has no entry, and its columns are blank. Safe to call multiple times — the existing entries
are cleared first.
"""
function load_all_smm_bundles!(; scenarios, point::Symbol, suffix::String = W_SUFFIX)
    empty!(smm_bundles)
    println("\nLoading the estimates the transitions were solved at..."); flush(stdout)
    for sc in scenarios
        tr = read_transition(sc, suffix, point)
        if tr === nothing
            @printf("  %-14s no %s: its columns are blank\n", sc,
                    basename(transition_file(sc, suffix, point)))
            continue
        end
        # By name in this tree's estimates/, so a transition file stays readable when the
        # repository moves (output/transition/ is tracked).
        for (w, file) in zip(tr.windows, joinpath.(SMM_OUT_DIR, basename.(values(tr.bundles))))
            b = read_bundle(file)
            (b === nothing || b.result === nothing || b.spec === nothing) &&
                error("$(basename(file)), which the $sc transition was solved at, is unreadable.")
            require_current_estimate(b, file)
            smm_bundles[w] = b
            @printf("  %-14s %s  %s\n", w, basename(file), bundle_summary(b))
        end
    end
    flush(stdout)
    return smm_bundles
end

# The simulation settings a bundle's point is solved with. A checkpoint bundle carries none,
# so its own spec's are used: the point is solved in memory, and nothing is written back.
_bundle_sim(bundle) = bundle.sim === nothing ? bundle.spec.sim : bundle.sim


# ============================================================
# Helper: reconstruct model moments from an SMM bundle
# ============================================================

"""
    _get_model_moments(bundle) → NamedTuple

Model moments at the optimum, read at its equilibrium (`equilibrium_at`).
"""
function _get_model_moments(bundle)
    sel = equilibrium_at(bundle.result.theta_opt, bundle.spec; sim = _bundle_sim(bundle))
    sel.model === nothing && error("No equilibrium at the stored optimum.")
    return sel.moments
end


# ============================================================
# Helper: load data moments + standard errors for a window
# ============================================================

# The data standard error of a moment is its sampling SD σ̂_k, the one the diagonal weighting
# divides by; NaN for a moment the variance file does not list.
function _load_data_and_se(window::Symbol)
    moments_nt = load_data_moments(; window=window, derived_dir=DERIVED_DIR)
    sv         = load_sampling_variances(; window=window, derived_dir=DERIVED_DIR)
    se_dict    = Dict{Symbol,Float64}(nm => sqrt(max(get(sv, nm, NaN), 0.0)) for nm in MOMENT_NAMES)
    return moments_nt, se_dict
end


# ============================================================
# Shared plot theme (matches existing plots.jl)
# ============================================================

function _set_theme!()
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

# Reuse colour constants from plots.jl if already defined, otherwise set them
if !@isdefined(_C1)
    const _C1 = :steelblue
    const _C2 = :firebrick
    const _C3 = :seagreen
    const _C4 = :darkorange
    const _C5 = :mediumpurple
end


# ============================================================
# Moment metadata for tables and plots
# ============================================================

# Category assignment for each moment
MOMENT_CATEGORY = Dict{Symbol,String}(
    :ur_total      => "Labour-market stocks",
    :ur_U          => "Labour-market stocks",
    :ur_S          => "Labour-market stocks",
    :exp_ur_total  => "Labour-market stocks",
    :exp_ur_U      => "Labour-market stocks",
    :exp_ur_S      => "Labour-market stocks",
    :skilled_share => "Labour-market stocks",
    :training_share=> "Labour-market stocks",
    :emp_var_U     => "Labour-market stocks",
    :emp_cm3_U     => "Labour-market stocks",
    :emp_var_S     => "Labour-market stocks",
    :emp_cm3_S     => "Labour-market stocks",
    :jfr_U         => "Transition rates",
    :sep_rate_U    => "Transition rates",
    :jfr_S         => "Transition rates",
    :sep_rate_S    => "Transition rates",
    :ee_rate_S     => "Transition rates",
    :mean_wage_U   => "Wages",
    :mean_wage_S   => "Wages",
    :p10_wage_U    => "Wages",
    :p10_wage_S    => "Wages",
    :p25_wage_U    => "Wages",
    :p25_wage_S    => "Wages",
    :p50_wage_U    => "Wages",
    :p50_wage_S    => "Wages",
    :p75_wage_U    => "Wages",
    :p75_wage_S    => "Wages",
    :p90_wage_U    => "Wages",
    :p90_wage_S    => "Wages",
    :wage_premium  => "Wages",
    :theta_U       => "Tightness",
    :theta_S       => "Tightness",
    :overlap_UgtS  => "Wages",
    :overlap_SltU  => "Wages",
    :ltu_share_S   => "Labour-market stocks",
)

# Short display names for moments
MOMENT_DISPLAY = Dict{Symbol,String}(
    :ur_total      => "UR total",
    :ur_U          => "UR unskilled",
    :ur_S          => "UR skilled",
    :exp_ur_total  => "exp(UR total)",
    :exp_ur_U      => "exp(UR unskilled)",
    :exp_ur_S      => "exp(UR skilled)",
    :skilled_share => "Skilled share",
    :training_share=> "Training share",
    :emp_var_U     => "Var(w) unskilled",
    :emp_cm3_U     => "CM3(w) unskilled",
    :emp_var_S     => "Var(w) skilled",
    :emp_cm3_S     => "CM3(w) skilled",
    :jfr_U         => "JFR unskilled",
    :sep_rate_U    => "Sep. rate unskilled",
    :jfr_S         => "JFR skilled",
    :sep_rate_S    => "Sep. rate skilled",
    :ee_rate_S     => "EE rate skilled",
    :mean_wage_U   => "Mean wage unskilled",
    :mean_wage_S   => "Mean wage skilled",
    :p10_wage_U    => "p10 wage unskilled",
    :p10_wage_S    => "p10 wage skilled",
    :p25_wage_U    => "p25 wage unskilled",
    :p25_wage_S    => "p25 wage skilled",
    :p50_wage_U    => "p50 wage unskilled",
    :p50_wage_S    => "p50 wage skilled",
    :p75_wage_U    => "p75 wage unskilled",
    :p75_wage_S    => "p75 wage skilled",
    :p90_wage_U    => "p90 wage unskilled",
    :p90_wage_S    => "p90 wage skilled",
    :wage_premium  => "Skill premium",
    :theta_U       => raw"$\theta_U$",
    :theta_S       => raw"$\theta_S$",
    :overlap_UgtS  => "Overlap U>med(S)",
    :overlap_SltU  => "Overlap S<med(U)",
    :ltu_share_S   => "LTU share skilled",
)

# Short tags for scatter plot labels
MOMENT_TAG = Dict{Symbol,String}(
    :ur_total      => "UR",
    :ur_U          => "UR_U",
    :ur_S          => "UR_S",
    :exp_ur_total  => "eUR",
    :exp_ur_U      => "eUR_U",
    :exp_ur_S      => "eUR_S",
    :skilled_share => "sksh",
    :training_share=> "trsh",
    :emp_var_U     => "vU",
    :emp_cm3_U     => "c3U",
    :emp_var_S     => "vS",
    :emp_cm3_S     => "c3S",
    :jfr_U         => "fU",
    :sep_rate_U    => "sU",
    :jfr_S         => "fS",
    :sep_rate_S    => "sS",
    :ee_rate_S     => "ee",
    :mean_wage_U   => "wU",
    :mean_wage_S   => "wS",
    :p10_wage_U    => "w10U",
    :p10_wage_S    => "w10S",
    :p25_wage_U    => "w25U",
    :p25_wage_S    => "w25S",
    :p50_wage_U    => "w50U",
    :p50_wage_S    => "w50S",
    :p75_wage_U    => "w75U",
    :p75_wage_S    => "w75S",
    :p90_wage_U    => "w90U",
    :p90_wage_S    => "w90S",
    :wage_premium  => "prem",
    :theta_U       => "θU",
    :theta_S       => "θS",
    :overlap_UgtS  => "ovUS",
    :overlap_SltU  => "ovSU",
    :ltu_share_S   => "ltuS",
)

# Shape/colour by category for scatter plot
CAT_MARKER = Dict{String,Symbol}(
    "Labour-market stocks" => :square,
    "Transition rates"     => :circle,
    "Wages"                => :utriangle,
    "Tightness"            => :diamond,
)

CAT_COLOR = Dict{String,Symbol}(
    "Labour-market stocks" => _C1,
    "Transition rates"     => _C2,
    "Wages"                => _C3,
    "Tightness"            => _C4,
)


# ============================================================
# 1. MODEL FIT TABLES A & B (LaTeX)
# ============================================================

"""
    _build_model_fit_tex(moments_in_table; windows_for_table, window_labels,
                          data_mom, se_dict, model_mom,
                          caption, label, col_spec) → Vector{String}

Internal helper: build the lines of a LaTeX model-fit table for a given
subset of moments, returning the full table (including \\begin/\\end).
"""
function _build_model_fit_tex(
    moments_in_table :: Vector{Symbol};
    windows_for_table :: Vector{Symbol},
    window_labels     :: Vector{String},
    data_mom          :: Dict{Symbol, NamedTuple},
    se_dict           :: Dict{Symbol, Dict{Symbol,Float64}},
    model_mom         :: Dict{Symbol, NamedTuple},
    caption           :: String,
    label             :: String,
    categories        :: Vector{String},
)
    ncols = 1 + 3 * length(windows_for_table)
    col_spec = "l" * repeat("rrr", length(windows_for_table))

    lines = String[]
    push!(lines, "\\begin{table}[htbp]")
    push!(lines, "  \\centering")
    push!(lines, "  \\caption{$caption}")
    push!(lines, "  \\label{$label}")
    push!(lines, "  \\small")
    push!(lines, "  \\begin{tabular}{$col_spec}")
    push!(lines, "    \\toprule")

    # Multi-column header
    hdr = "    Moment"
    for lbl in window_labels
        hdr *= " & \\multicolumn{3}{c}{$lbl}"
    end
    hdr *= " \\\\"
    push!(lines, hdr)

    # Cmidrule
    cmidr = "   "
    for (i, _) in enumerate(windows_for_table)
        lo = 2 + 3*(i-1)
        hi = lo + 2
        cmidr *= " \\cmidrule(lr){$(lo)-$(hi)}"
    end
    push!(lines, cmidr)

    # Sub-header
    subhdr = "    "
    for _ in windows_for_table
        subhdr *= " & Data & Model & Diff."
    end
    subhdr *= " \\\\"
    push!(lines, subhdr)
    push!(lines, "    \\midrule")

    for cat in categories
        cat_moments = [nm for nm in moments_in_table
                       if get(MOMENT_CATEGORY, nm, "") == cat]
        isempty(cat_moments) && continue

        push!(lines, "    \\multicolumn{$(ncols)}{l}{\\textit{$cat}} \\\\[2pt]")

        for nm in cat_moments
            display = get(MOMENT_DISPLAY, nm, string(nm))
            row = "    \\quad $display"
            se_row = "    "   # SE row indentation

            for w in windows_for_table
                d_val    = haskey(data_mom, w) && haskey(data_mom[w], nm) ? data_mom[w][nm].value : NaN
                m_val    = haskey(model_mom, w) && hasproperty(model_mom[w], nm) ? getproperty(model_mom[w], nm) : NaN
                se_d_val = haskey(se_dict, w) ? get(se_dict[w], nm, NaN) : NaN
                diff     = isfinite(d_val) && isfinite(m_val) ? m_val - d_val : NaN

                fmt_d    = isfinite(d_val)  ? @sprintf("%.4f", d_val)   : "--"
                fmt_m    = isfinite(m_val)  ? @sprintf("%.4f", m_val)   : "--"
                fmt_diff = isfinite(diff)   ? @sprintf("%+.4f", diff)   : "--"

                row *= " & $fmt_d & $fmt_m & $fmt_diff"

                # SE row: the data SE under the data column
                fmt_se_d = isfinite(se_d_val) ? "{\\scriptsize ($(@sprintf("%.4f", se_d_val)))}" : ""
                se_row *= " & $fmt_se_d & &"
            end

            row    *= " \\\\"
            se_row *= " \\\\"
            push!(lines, row)
            push!(lines, se_row)
        end
        push!(lines, "    [4pt]")
    end

    push!(lines, "    \\bottomrule")
    push!(lines, "  \\end{tabular}")
    push!(lines, "\\end{table}")
    return lines
end


"""
    make_model_fit_tables(; suffix)

Generate two LaTeX tables comparing data and model moments.

Table A (`model_fit_A{suffix}.tex`): Labour-market stocks + Transition rates
Table B (`model_fit_B{suffix}.tex`): Wages + Tightness

Each data cell has the data SE, the moment's sampling SD, in parentheses on the row
below (\\scriptsize).
"""
function make_model_fit_tables(; suffix::String=W_SUFFIX)
    println("\n── Generating model fit tables (A & B) ──"); flush(stdout)

    windows_for_table = [:base_fc, :base_covid]
    window_labels     = ["Financial Crisis", "COVID"]

    # Load data moments, SEs, and model moments for each window
    data_mom     = Dict{Symbol, NamedTuple}()
    se_dict      = Dict{Symbol, Dict{Symbol,Float64}}()
    model_mom    = Dict{Symbol, NamedTuple}()

    for w in windows_for_table
        dm, sd = _load_data_and_se(w)
        data_mom[w] = dm
        se_dict[w]  = sd
        if haskey(smm_bundles, w)
            model_mom[w] = _get_model_moments(smm_bundles[w])
            println("  Computed model moments for $w"); flush(stdout)
        else
            @warn "No SMM bundle for $w — model column will be blank"
        end
    end

    # Determine which moments are active (have weight > 0 in at least one spec)
    active_moments = Symbol[]
    for nm in MOMENT_NAMES
        is_active = false
        for w in windows_for_table
            if haskey(smm_bundles, w)
                spec = smm_bundles[w].spec
                if haskey(spec.moments, nm) && spec.moments[nm].weight > 0.0
                    is_active = true
                end
            end
        end
        is_active && push!(active_moments, nm)
    end

    # ── Table A: Labour-market stocks + Transition rates ─────────────────
    cats_A = ["Labour-market stocks", "Transition rates"]
    lines_A = _build_model_fit_tex(
        active_moments;
        windows_for_table = windows_for_table,
        window_labels     = window_labels,
        data_mom          = data_mom,
        se_dict           = se_dict,
        model_mom         = model_mom,
        caption           = "Model Fit (A): Labour-Market Stocks and Transition Rates",
        label             = "tab:model_fit_a",
        categories        = cats_A,
    )
    outpath_A = joinpath(TABLES_DIR, "model_fit_A$(suffix).tex")
    open(outpath_A, "w") do io
        write(io, join(lines_A, "\n"))
    end
    @printf("  Saved: %s\n", outpath_A); flush(stdout)

    # ── Table B: Wages + Tightness ────────────────────────────────────────
    cats_B = ["Wages", "Tightness"]
    lines_B = _build_model_fit_tex(
        active_moments;
        windows_for_table = windows_for_table,
        window_labels     = window_labels,
        data_mom          = data_mom,
        se_dict           = se_dict,
        model_mom         = model_mom,
        caption           = "Model Fit (B): Wages and Tightness",
        label             = "tab:model_fit_b",
        categories        = cats_B,
    )
    outpath_B = joinpath(TABLES_DIR, "model_fit_B$(suffix).tex")
    open(outpath_B, "w") do io
        write(io, join(lines_B, "\n"))
    end
    @printf("  Saved: %s\n", outpath_B); flush(stdout)

    return outpath_A, outpath_B
end


# ============================================================
# 2. PARAMETER ESTIMATES TABLES A & B (LaTeX)
# ============================================================

# Symbol and description of each parameter, keyed as in default_free_params.
# Which table a parameter lands in is read off each bundle's spec, so the tables
# follow FIX_PARAMS without a list of their own.
const PARAM_DISPLAY = Dict{Tuple{Symbol,Symbol},Tuple{String,String}}(
    (:common, :a_ℓ) => (raw"$a_\ell$",       "Ability marginal shape"),
    (:common, :b_ℓ) => (raw"$b_\ell$",       "Ability marginal shape"),
    (:common, :ρ_x) => (raw"$\rho_x$",       "Ability correlation"),
    (:common, :c)   => (raw"$c$",            "Training cost"),
    (:common, :A)   => (raw"$A$",            "Aggregate scale"),
    (:unsk,   :bU)  => (raw"$b_U$",          "Unskilled unemployment income"),
    (:unsk,   :bT)  => (raw"$b_T$",          "Training flow"),
    (:unsk,   :PU)  => (raw"$P_U$",          "Unskilled productivity"),
    (:unsk,   :α_U) => (raw"$\alpha_U$",     "Unskilled redraw shape"),
    (:unsk,   :μ)   => (raw"$\mu_U$",        "Unskilled matching efficiency"),
    (:unsk,   :η)   => (raw"$\eta_U$",       "Unskilled matching elasticity"),
    (:unsk,   :β)   => (raw"$\beta_U$",      "Unskilled bargaining share"),
    (:unsk,   :λ)   => (raw"$\lambda_U$",    "Unskilled shock rate"),
    (:unsk,   :tight) => (raw"$\theta_U$",   "Unskilled tightness"),
    (:unsk,   :k)   => (raw"$k_U$",          "Unskilled vacancy cost"),
    (:unsk,   :ξ)   => (raw"$\xi_U$",        "Unskilled exogenous separation"),
    (:unsk,   :σ_w) => (raw"$\sigma_{w,U}$", "Unskilled wage measurement error"),
    (:skl,    :bS)  => (raw"$b_S$",          "Skilled unemployment income"),
    (:skl,    :PS)  => (raw"$P_S$",          "Skilled productivity"),
    (:skl,    :a_Γ) => (raw"$a_\Gamma$",     "Skilled offer shape"),
    (:skl,    :b_Γ) => (raw"$b_\Gamma$",     "Skilled offer shape"),
    (:skl,    :δ)   => (raw"$\delta_S$",     "Skilled redraw support"),
    (:skl,    :μ)   => (raw"$\mu_S$",        "Skilled matching efficiency"),
    (:skl,    :η)   => (raw"$\eta_S$",       "Skilled matching elasticity"),
    (:skl,    :β)   => (raw"$\beta_S$",      "Skilled bargaining share"),
    (:skl,    :λ)   => (raw"$\lambda_S$",    "Skilled shock rate"),
    (:skl,    :tight) => (raw"$\theta_S$",   "Skilled tightness"),
    (:skl,    :k)   => (raw"$k_S$",          "Skilled vacancy cost"),
    (:skl,    :σ)   => (raw"$\sigma_S$",     "On-the-job search cost"),
    (:skl,    :ξ)   => (raw"$\xi_S$",        "Skilled exogenous separation"),
    (:skl,    :σ_w) => (raw"$\sigma_{w,S}$", "Skilled wage measurement error"),
)

const _PARAM_BLOCK_TITLES = ((:common, "Common"), (:unsk, "Unskilled market"), (:skl, "Skilled market"))


"""
    _extract_param(bundle, field, block) → Float64

Read one parameter, estimated or pinned, off an SMM bundle's optimum.  The vacancy costs
are the ones the bundle records at its optimum (`vacancy_costs`).
"""
function _extract_param(bundle, field::Symbol, block::Symbol)
    field === :k && return block === :unsk ? bundle.vacancy_costs.k.U : bundle.vacancy_costs.k.S
    cp, up, sp, tight = unpack_θ(bundle.result.theta_opt, bundle.spec)
    return _baseline_param_value(block, field, cp, up, sp, tight)
end


"""
    make_parameter_tables(; suffix)

Generate two LaTeX tables with columns
    Parameter | Description | Baseline FC | Crisis FC | Δ FC | Baseline COVID | Crisis COVID | Δ COVID

Table A (`parameter_estimates_A{suffix}.tex`): the external calibration and every
parameter no loaded window estimates, at its pinned value.
Table B (`parameter_estimates_B{suffix}.tex`): every parameter at least one loaded
window estimates, grouped by solver block.

Δ is crisis minus baseline, and prints as --- where the two are equal, as for the
ability distribution, which a crisis window holds at the baseline estimate.
"""
function make_parameter_tables(; suffix::String=W_SUFFIX)
    println("\n── Generating parameter estimates tables (A & B) ──"); flush(stdout)

    calib = load_calibrated_params(; derived_dir=DERIVED_DIR)

    # The vacancy costs are estimated through the tightnesses, and listed with every other
    # estimated parameter of their block.
    params  = vcat(default_free_params(),
                   [ParamSpec(blk, :k, NaN, NaN, label) for (blk, label) in VACANCY_COSTS])
    unlisted = [(ps.block, ps.name) for ps in params if !haskey(PARAM_DISPLAY, (ps.block, ps.name))]
    isempty(unlisted) || error("PARAM_DISPLAY has no entry for $(unlisted).")

    fmt(x)     = isfinite(x) ? @sprintf("%.5f", x) : "--"
    fmtΔ(b, c) = !(isfinite(b) && isfinite(c)) ? "--" : b == c ? "---" : @sprintf("%+.5f", c - b)

    estimated(ps) = ps.name === :k ||
                    any(w -> haskey(smm_bundles, w) &&
                             any(p -> (p.block, p.name) == (ps.block, ps.name), smm_bundles[w].spec.free),
                        ALL_WINDOWS)
    value(w, ps)  = haskey(smm_bundles, w) ? _extract_param(smm_bundles[w], ps.name, ps.block) : NaN

    function row(latex, desc, v)
        return "    \\quad $latex & $desc & $(fmt(v[1])) & $(fmt(v[2])) & $(fmtΔ(v[1], v[2])) & " *
               "$(fmt(v[3])) & $(fmt(v[4])) & $(fmtΔ(v[3], v[4])) \\\\"
    end
    function head(caption, label)
        return ["\\begin{table}[htbp]", "  \\centering", "  \\caption{$caption}", "  \\label{$label}",
                "  \\small", "  \\begin{tabular}{llcccccc}", "    \\toprule",
                "    & & \\multicolumn{3}{c}{Financial Crisis} & \\multicolumn{3}{c}{COVID} \\\\",
                "    \\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
                "    Parameter & Description & Baseline & Crisis & \$\\Delta\$ & Baseline & Crisis & \$\\Delta\$ \\\\",
                "    \\midrule"]
    end
    tail = ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    function blocks!(lines, keep)
        for (blk, title) in _PARAM_BLOCK_TITLES
            sel = [ps for ps in params if ps.block === blk && keep(ps)]
            isempty(sel) && continue
            push!(lines, "    \\multicolumn{8}{l}{\\textit{$title}} \\\\[2pt]")
            for ps in sel
                latex, desc = PARAM_DISPLAY[(ps.block, ps.name)]
                push!(lines, row(latex, desc, [value(w, ps) for w in ALL_WINDOWS]))
            end
            push!(lines, "    [4pt]")
        end
        return lines
    end
    function save(lines, name)
        path = joinpath(TABLES_DIR, "$(name)$(suffix).tex")
        open(io -> write(io, join(lines, "\n")), path, "w")
        @printf("  Saved: %s\n", path); flush(stdout)
        return path
    end

    # ── TABLE A: external calibration and pinned parameters ──────────────
    lines_A = head("Parameter Values (A): Calibrated and Fixed", "tab:param_estimates_a")
    push!(lines_A, "    \\multicolumn{8}{l}{\\textit{Externally calibrated}} \\\\[2pt]")
    for (val, latex, desc) in ((calib.r,   raw"$r$",       "Discount rate"),
                               (calib.nu,  raw"$\nu$",     "Demographic exit rate"),
                               (calib.phi, raw"$\varphi$", "Training completion rate"))
        push!(lines_A, row(latex, desc, fill(val, 4)))
    end
    push!(lines_A, "    [4pt]")
    blocks!(lines_A, ps -> !estimated(ps))
    outpath_A = save(append!(lines_A, tail), "parameter_estimates_A")

    # ── TABLE B: estimated parameters ─────────────────────────────────────
    lines_B = head("Parameter Estimates (B)", "tab:param_estimates_b")
    blocks!(lines_B, estimated)
    outpath_B = save(append!(lines_B, tail), "parameter_estimates_B")

    return outpath_A, outpath_B
end


# ============================================================
# 3. TRAINING CUTOFF TABLE (x̄ across 4 windows)
# ============================================================

"""
    _get_x_bar(bundle) → Float64

Re-solve the model at the SMM optimum and extract the training cutoff x̄.

The cutoff is the type x at which the worker is indifferent between
searching as unskilled and entering training, i.e. U_search(x) = −c(x)+T(x).
To avoid grid-snapping (the quadrature x-grid is coarse and non-uniform), we interpolate the
crossing point of `net_T(x) − U_search(x)` rather than taking the
first grid node where τT > 0.5.

Returns NaN if no worker trains (τT ≡ 0) or everyone trains (τT ≡ 1).
"""
function _get_x_bar(bundle; Nx_fine::Int = 400)
    res  = bundle.result
    spec = bundle.spec
    sim  = _bundle_sim(bundle)

    # Solved on a finer grid so the interpolated cutoff is precise, at the optimum's
    # equilibrium (equilibrium_at).
    sel = equilibrium_at(res.theta_opt, spec; Nx = Nx_fine, sim = sim)
    if sel.model === nothing
        @warn "No equilibrium at the stored optimum on the Nx=$Nx_fine grid — x̄ unavailable"
        return NaN
    end

    obj = compute_equilibrium_objects(sel.model)
    xg  = obj.xg

    # gap(x) = (−c(x) + T(x)) − U_search(x) for the type with a_U = a_S = x;  training
    # chosen where gap ≥ 0
    gap = diag(obj.net_T) .- obj.Usearch

    # Find zero crossing via linear interpolation
    for i in firstindex(xg):lastindex(xg)-1
        if gap[i] < 0.0 && gap[i+1] >= 0.0
            # Linear interpolation for the root
            α = -gap[i] / (gap[i+1] - gap[i])
            return xg[i] + α * (xg[i+1] - xg[i])
        end
    end

    # Fallback: if gap[1] ≥ 0, everyone trains → x̄ = 0
    if gap[1] >= 0.0
        return 0.0
    end

    # No crossing found → nobody trains
    return NaN
end

"""
    make_xbar_table(; suffix)

Generate a LaTeX table with the training cutoff x̄ for all four windows.

Layout:
           Baseline   Crisis   Δ
  FC       x̄_bfc     x̄_cfc    Δ_fc
  COVID    x̄_bcov    x̄_ccov   Δ_covid

Saved to `output/tables/xbar_table{suffix}.tex`.
"""
function make_xbar_table(; suffix::String=W_SUFFIX)
    println("\n── Generating training cutoff (x̄) table ──"); flush(stdout)

    fmt(x)  = isfinite(x) ? @sprintf("%.4f", x)  : "--"
    fmtd(x) = isfinite(x) ? @sprintf("%+.4f", x) : "--"

    # Compute x̄ for each window
    xbars = Dict{Symbol, Float64}()
    for w in ALL_WINDOWS
        if haskey(smm_bundles, w)
            xbars[w] = _get_x_bar(smm_bundles[w])
            @printf("  x̄(%s) = %s\n", w, fmt(xbars[w]))
        else
            xbars[w] = NaN
            @warn "No SMM bundle for $w — x̄ will be blank"
        end
    end
    flush(stdout)

    # Episode rows: (label, base_window, crisis_window)
    episodes = [
        ("Financial Crisis", :base_fc,    :crisis_fc),
        ("COVID",            :base_covid, :crisis_covid),
    ]

    # Build LaTeX table
    lines = String[]
    push!(lines, "\\begin{table}[htbp]")
    push!(lines, "  \\centering")
    push!(lines, "  \\caption{Training Cutoff \$\\bar{x}\$}")
    push!(lines, "  \\label{tab:xbar}")
    push!(lines, "  \\begin{tabular}{lccc}")
    push!(lines, "    \\toprule")
    push!(lines, "    & Baseline & Crisis & \$\\Delta\$ \\\\")
    push!(lines, "    \\midrule")

    for (label, bw, cw) in episodes
        xb = get(xbars, bw, NaN)
        xc = get(xbars, cw, NaN)
        delta = (isfinite(xb) && isfinite(xc)) ? xc - xb : NaN
        push!(lines, "    $label & $(fmt(xb)) & $(fmt(xc)) & $(fmtd(delta)) \\\\")
    end

    push!(lines, "    \\bottomrule")
    push!(lines, "  \\end{tabular}")
    push!(lines, "\\end{table}")

    outpath = joinpath(TABLES_DIR, "xbar_table$(suffix).tex")
    open(outpath, "w") do io
        write(io, join(lines, "\n"))
    end
    @printf("  Saved: %s\n", outpath); flush(stdout)

    return outpath
end


# ============================================================
# 4. MODEL FIT SCATTER PLOT
# ============================================================

function make_model_fit_scatter(; window::Symbol=:base_fc, suffix::String=W_SUFFIX)
    println("\n── Generating model fit scatter plot ──"); flush(stdout)
    _set_theme!()

    if !haskey(smm_bundles, window)
        @warn "No SMM bundle for $window — skipping scatter plot"
        return nothing
    end

    data_mom, se = _load_data_and_se(window)
    m_mom = _get_model_moments(smm_bundles[window])
    println("  Computed model moments for scatter plot ($window)"); flush(stdout)

    # Determine active moments from the spec
    spec = smm_bundles[window].spec
    active = [nm for nm in MOMENT_NAMES
              if haskey(spec.moments, nm) && spec.moments[nm].weight > 0.0]

    # Collect data and model values
    d_vals = Float64[]
    m_vals = Float64[]
    cats   = String[]
    tags   = String[]

    for nm in active
        dv = haskey(data_mom, nm) ? data_mom[nm].value : NaN
        mv = hasproperty(m_mom, nm) ? getproperty(m_mom, nm) : NaN
        if isfinite(dv) && isfinite(mv)
            push!(d_vals, dv)
            push!(m_vals, mv)
            push!(cats, get(MOMENT_CATEGORY, nm, "Other"))
            push!(tags, get(MOMENT_TAG, nm, string(nm)))
        end
    end

    # Plot
    p = plot(; xlabel="Data moment", ylabel="Model moment",
             #title="Model Fit: $(WINDOW_LABELS[window])",
             size=(700, 650), margin=8Plots.mm,
             legend=:topleft)

    # 45-degree line
    all_vals = vcat(d_vals, m_vals)
    rng = maximum(all_vals) - minimum(all_vals)
    buf = max(rng * 0.05, 0.01)
    lo = minimum(all_vals) - buf
    hi = maximum(all_vals) + buf
    plot!(p, [lo, hi], [lo, hi], color=:gray, ls=:dash, lw=1.2, label = "")

    # Plot by category
    unique_cats = ["Labour-market stocks", "Transition rates", "Wages", "Tightness"]
    for cat in unique_cats
        idx = findall(c -> c == cat, cats)
        isempty(idx) && continue
        scatter!(p, d_vals[idx], m_vals[idx],
                 marker=get(CAT_MARKER, cat, :circle),
                 color=get(CAT_COLOR, cat, :black),
                 markersize=7, markerstrokewidth=0.5,
                 label=cat)
    end

    # Add text labels
    for i in eachindex(d_vals)
        annotate!(p, d_vals[i], m_vals[i] + (hi - lo) * 0.015,
                  text(tags[i], 6, :center, :gray40))
    end

    outpath = joinpath(PLOTS_DIR, "model_fit_scatter_$(window)$(suffix).png")
    savefig(p, outpath)
    @printf("  Saved: %s\n", outpath)
    flush(stdout)
    return outpath
end


# ============================================================
# Stand-alone entry point — only fires when this file IS the
# script invoked from the command line.  Otherwise it is a
# library called from transition_main.jl.
# ============================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "="^60)
    println("  Plots & Tables — standalone run")
    println("="^60)
    flush(stdout)

    for point in (:smm, :mcmc)
        load_all_smm_bundles!(; scenarios = [:fc, :covid], point = point)
        isempty(smm_bundles) && continue
        tag = transition_tag(W_SUFFIX, point)
        make_model_fit_tables(; suffix = tag)
        make_parameter_tables(; suffix = tag)
        make_xbar_table(; suffix = tag)
        for w in [:base_fc, :base_covid]
            make_model_fit_scatter(; window = w, suffix = tag)
        end
    end

    println("\n" * "="^60)
    println("  All outputs generated successfully.")
    println("="^60)
    flush(stdout)
end

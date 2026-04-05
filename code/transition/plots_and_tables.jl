############################################################
# plots_and_tables.jl — Post-estimation plots and tables
#
# Generates:
#   1. Model fit tables A & B (LaTeX) — data vs model moments + SEs
#   2. Parameter estimates tables A & B (LaTeX) — 4 windows + deltas
#   3. Model fit scatter plot — data vs model, 45-degree line
#
# Transition dynamics panels are generated separately by
# transition_panel.jl — run that script independently.
#
# Usage (from project root):
#   julia code/transition/plots_and_tables.jl
# (also works from code/smm/ or any code/ subfolder)
#
# Requires completed SMM runs for all four windows.
############################################################

println("="^60)
println("  Segmented Search Model — Plots & Tables")
println("="^60)
flush(stdout)

# ── Paths ─────────────────────────────────────────────────────
# This script can live anywhere under code/ (e.g. code/smm/ or
# code/transition/).  PROJECT_ROOT is always two levels up from
# @__DIR__, and the SMM source files are in code/smm/.
SCRIPT_DIR   = @__DIR__
PROJECT_ROOT = joinpath(SCRIPT_DIR, "..", "..")
SOLVER_DIR   = joinpath(PROJECT_ROOT, "code", "solver")
SMM_SRC_DIR  = joinpath(PROJECT_ROOT, "code", "smm")
OUTPUT_DIR   = joinpath(PROJECT_ROOT, "output")
PLOTS_DIR    = joinpath(OUTPUT_DIR, "plots")
TABLES_DIR   = joinpath(OUTPUT_DIR, "tables")
SMM_OUT_DIR  = joinpath(OUTPUT_DIR, "smm")
TRANS_DIR    = joinpath(OUTPUT_DIR, "transition")
DERIVED_DIR  = joinpath(PROJECT_ROOT, "data", "derived")

mkpath(PLOTS_DIR)
mkpath(TABLES_DIR)

# ── Packages ──────────────────────────────────────────────────
print("Loading packages... "); flush(stdout)

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
using JLD2

println("done."); flush(stdout)

# ── Load solver (needed for structs + model_moments) ──────────
print("Loading solver modules... "); flush(stdout)

include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))

println("done."); flush(stdout)

# ── Load SMM modules ─────────────────────────────────────────
print("Loading SMM modules... "); flush(stdout)

include(joinpath(SMM_SRC_DIR, "moments.jl"))
include(joinpath(SMM_SRC_DIR, "smm_params.jl"))
include(joinpath(SMM_SRC_DIR, "smm.jl"))

println("done."); flush(stdout)


# ============================================================
# Configuration
# ============================================================

# Weight matrix suffix — must match what was used in estimation
W_COND_TARGET = 1e8

# _w_suffix may already be defined via smm.jl include; define only if missing
if !isdefined(Main, :_w_suffix_pt)
    _w_suffix_pt(ct::Float64) = ct == 0.0 ? "_diagonalW" :
                                 ct == 1.0 ? "_compressedW" :
                                 ct == 2.0 ? "_equalW" : "_fullW"
end
W_SUFFIX = _w_suffix_pt(W_COND_TARGET)

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
# 0. Load all SMM bundles
# ============================================================
println("\nLoading SMM results..."); flush(stdout)

smm_bundles = Dict{Symbol, Any}()
for w in ALL_WINDOWS
    jls_path = joinpath(SMM_OUT_DIR, "smm_result_$(w)$(W_SUFFIX).jls")
    if isfile(jls_path)
        data = _load_smm_bundle(jls_path; label=string(w))
        if !isnothing(data)
            smm_bundles[w] = data
            @printf("  %-18s  Q = %.6e  converged = %s\n",
                    w, data.result.loss_opt, data.result.converged)
        else
            @warn "Could not read SMM bundle for $w at $jls_path"
        end
    else
        @warn "SMM result file not found: $jls_path"
    end
end
flush(stdout)


# ============================================================
# Helper: reconstruct model moments from an SMM bundle
# ============================================================

"""
    _get_model_moments(bundle) → NamedTuple

Re-solve the model at the optimum and compute model moments.
"""
function _get_model_moments(bundle)
    res  = bundle.result
    spec = bundle.spec
    sim  = bundle.sim

    cp, rp, up, sp = unpack_θ(res.theta_opt, spec)
    model, sol = solve_model(cp, rp, up, sp, sim;
                             Nx=spec.run.Nx, Np_U=spec.run.Np_U, Np_S=spec.run.Np_S)

    if !sol.ok
        @warn "Model did not fully converge at stored optimum — moments may be unreliable"
    end

    obj = compute_equilibrium_objects(model)
    return model_moments(obj)
end


# ============================================================
# Helper: load data moments + standard errors for a window
# ============================================================

function _load_data_and_se(window::Symbol)
    moments_nt = load_data_moments(; window=window, derived_dir=DERIVED_DIR)
    se_vec     = load_sigma_matrix(; window=window, derived_dir=DERIVED_DIR)
    # se_vec is sqrt(diag(Σ)) in MOMENT_NAMES order
    se_dict = Dict{Symbol,Float64}()
    for (i, nm) in enumerate(MOMENT_NAMES)
        se_dict[nm] = i <= length(se_vec) ? se_vec[i] : NaN
    end
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
    :p25_wage_U    => "Wages",
    :p25_wage_S    => "Wages",
    :p50_wage_U    => "Wages",
    :p50_wage_S    => "Wages",
    :wage_premium  => "Wages",
    :theta_U       => "Tightness",
    :theta_S       => "Tightness",
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
    :p25_wage_U    => "p25 wage unskilled",
    :p25_wage_S    => "p25 wage skilled",
    :p50_wage_U    => "p50 wage unskilled",
    :p50_wage_S    => "p50 wage skilled",
    :wage_premium  => "Skill premium",
    :theta_U       => raw"$\theta_U$",
    :theta_S       => raw"$\theta_S$",
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
    :p25_wage_U    => "w25U",
    :p25_wage_S    => "w25S",
    :p50_wage_U    => "w50U",
    :p50_wage_S    => "w50S",
    :wage_premium  => "prem",
    :theta_U       => "θU",
    :theta_S       => "θS",
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
# SMM Standard Errors (sandwich formula)
# ============================================================

"""
    compute_smm_standard_errors(bundle; derived_dir, cond_target) → (SE_θ, SE_model_mom)

Compute SMM standard errors for parameter estimates and model moments
using the sandwich formula:

    Var(θ̂) = (D'WD)⁻¹ D'W Ŝ W D (D'WD)⁻¹

where:
  - D  = Jacobian ∂g(θ)/∂θ' (K_active × n_free), computed numerically
  - W  = weight matrix from load_weight_matrix
  - Ŝ  = covariance matrix of moments from sigma_{window}.csv

Returns:
  - SE_θ         :: Vector{Float64}  — SEs for each free parameter in spec.free order
  - SE_model_mom :: Dict{Symbol, Float64} — SEs for each active model moment
                    (delta method: SE_g = sqrt(diag(D * Var(θ̂) * D')))
"""
function compute_smm_standard_errors(bundle; derived_dir::String, cond_target::Float64=W_COND_TARGET, window::Symbol=:base_fc)
    res  = bundle.result
    spec = bundle.spec
    sim  = bundle.sim

    θ_opt = res.theta_opt
    n_free = length(θ_opt)

    # Determine active moments (weight > 0) in spec order
    active_moments = [nm for nm in MOMENT_NAMES
                      if haskey(spec.moments, nm) && spec.moments[nm].weight > 0.0]
    K_active = length(active_moments)

    # ── Helper: compute model moment vector at a given θ ────────────────
    function _model_mom_vec(θ)
        cp, rp, up, sp = unpack_θ(θ, spec)
        model_sol, sol = solve_model(cp, rp, up, sp, sim;
                                     Nx=spec.run.Nx, Np_U=spec.run.Np_U, Np_S=spec.run.Np_S)
        obj = compute_equilibrium_objects(model_sol)
        mm  = model_moments(obj)
        return [hasproperty(mm, nm) ? getproperty(mm, nm) : NaN for nm in active_moments]
    end

    # ── Numerical Jacobian D (K_active × n_free) ────────────────────────
    ε = 1e-5
    g0 = _model_mom_vec(θ_opt)
    D  = zeros(K_active, n_free)
    for j in 1:n_free
        θ_fwd = copy(θ_opt); θ_fwd[j] += ε
        θ_bwd = copy(θ_opt); θ_bwd[j] -= ε
        g_fwd = _model_mom_vec(θ_fwd)
        g_bwd = _model_mom_vec(θ_bwd)
        D[:, j] = (g_fwd .- g_bwd) ./ (2ε)
    end

    # ── Load Ŝ (full covariance matrix, subset to active moments) ────────
    sigma_file = joinpath(derived_dir, "sigma_$(window).csv")
    if !isfile(sigma_file)
        @warn "sigma_$(window).csv not found — SMM SEs unavailable"
        return fill(NaN, n_free), Dict{Symbol,Float64}(nm => NaN for nm in active_moments)
    end
    df_sig   = CSV.read(sigma_file, DataFrame)
    csv_cols = Symbol.(names(df_sig))
    Σ_full   = Matrix{Float64}(df_sig)

    # Subset to active moments
    col_idx = [findfirst(==(nm), csv_cols) for nm in active_moments]
    missing_idx = findall(isnothing, col_idx)
    if !isempty(missing_idx)
        @warn "sigma CSV missing some active moments — SE computation may be unreliable"
        # Replace missing with 0
        col_idx = [isnothing(i) ? 0 : i for i in col_idx]
    end
    Ŝ = zeros(K_active, K_active)
    for (r, ci) in enumerate(col_idx), (c, cj) in enumerate(col_idx)
        if ci > 0 && cj > 0
            Ŝ[r, c] = Σ_full[ci, cj]
        end
    end

    # ── Load W matrix ────────────────────────────────────────────────────
    # Determine skip_moments: those with weight == 0
    skip_moments = [nm for nm in MOMENT_NAMES
                    if haskey(spec.moments, nm) && spec.moments[nm].weight <= 0.0]
    W = load_weight_matrix(; window=window, derived_dir=derived_dir,
                             cond_target=cond_target, skip_moments=skip_moments)

    # If W is nothing (equal weights), use identity
    if isnothing(W)
        W = Matrix{Float64}(I, K_active, K_active)
    end

    # ── Sandwich formula ─────────────────────────────────────────────────
    # Var(θ̂) = (D'WD)⁻¹ D'W Ŝ W D (D'WD)⁻¹
    DtW   = D' * W          # n_free × K_active
    DtWD  = DtW * D         # n_free × n_free
    meat  = DtW * Ŝ * W * D # n_free × n_free

    # Regularise DtWD for inversion
    DtWD_reg = DtWD + 1e-12 * I
    DtWD_inv = try
        inv(DtWD_reg)
    catch e
        @warn "DtWD inversion failed ($e) — using pinv"
        pinv(DtWD_reg)
    end

    Var_θ = DtWD_inv * meat * DtWD_inv
    SE_θ  = sqrt.(max.(diag(Var_θ), 0.0))

    # ── Delta-method SEs for model moments ──────────────────────────────
    # Var(g(θ̂)) = D * Var(θ̂) * D'  (note: D is K×p, Var_θ is p×p)
    Var_g = D * Var_θ * D'
    SE_g_vec = sqrt.(max.(diag(Var_g), 0.0))
    SE_model_mom = Dict{Symbol,Float64}(
        nm => SE_g_vec[i] for (i, nm) in enumerate(active_moments)
    )

    return SE_θ, SE_model_mom
end


# ============================================================
# 1. MODEL FIT TABLES A & B (LaTeX)
# ============================================================

"""
    _build_model_fit_tex(moments_in_table; windows_for_table, window_labels,
                          data_mom, se_dict, model_mom, se_model_mom,
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
    se_model_mom      :: Dict{Symbol, Dict{Symbol,Float64}},
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
                se_m_val = haskey(se_model_mom, w) ? get(se_model_mom[w], nm, NaN) : NaN
                diff     = isfinite(d_val) && isfinite(m_val) ? m_val - d_val : NaN

                fmt_d    = isfinite(d_val)  ? @sprintf("%.4f", d_val)   : "--"
                fmt_m    = isfinite(m_val)  ? @sprintf("%.4f", m_val)   : "--"
                fmt_diff = isfinite(diff)   ? @sprintf("%+.4f", diff)   : "--"

                row *= " & $fmt_d & $fmt_m & $fmt_diff"

                # SE row: data SE under data column, model SE under model column
                _sd_str  = isfinite(se_d_val) ? @sprintf("%.4f", se_d_val) : ""
                _sm_str  = isfinite(se_m_val) ? @sprintf("%.4f", se_m_val) : ""
                fmt_se_d = isfinite(se_d_val) ? "{\\scriptsize ($(_sd_str))}" : ""
                fmt_se_m = isfinite(se_m_val) ? "{\\scriptsize ($(_sm_str))}" : ""
                se_row *= " & $fmt_se_d & $fmt_se_m &"
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

Each data cell has the data SE in parentheses on the row below (\\scriptsize).
Each model cell has the delta-method SE for the model moment below it.
"""
function make_model_fit_tables(; suffix::String=W_SUFFIX)
    println("\n── Generating model fit tables (A & B) ──"); flush(stdout)

    windows_for_table = [:base_fc, :base_covid]
    window_labels     = ["Financial Crisis", "COVID"]

    # Load data moments, SEs, and model moments for each window
    data_mom     = Dict{Symbol, NamedTuple}()
    se_dict      = Dict{Symbol, Dict{Symbol,Float64}}()
    model_mom    = Dict{Symbol, NamedTuple}()
    se_model_mom = Dict{Symbol, Dict{Symbol,Float64}}()

    for w in windows_for_table
        dm, sd = _load_data_and_se(w)
        data_mom[w] = dm
        se_dict[w]  = sd
        if haskey(smm_bundles, w)
            model_mom[w] = _get_model_moments(smm_bundles[w])
            println("  Computed model moments for $w"); flush(stdout)

            # Compute SMM standard errors
            println("  Computing SMM standard errors for $w..."); flush(stdout)
            _, se_mm = compute_smm_standard_errors(smm_bundles[w];
                                                    derived_dir=DERIVED_DIR,
                                                    cond_target=W_COND_TARGET,
                                                    window=w)
            se_model_mom[w] = se_mm
            println("  Done SE for $w"); flush(stdout)
        else
            @warn "No SMM bundle for $w — model column will be blank"
            se_model_mom[w] = Dict{Symbol,Float64}()
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
        se_model_mom      = se_model_mom,
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
        se_model_mom      = se_model_mom,
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

# Parameter classification
# Externally calibrated: r, ν, φ
# Deep structural (13): a_ℓ, b_ℓ, c, bU, bT, bS, μ_U, η_U, β_U, μ_S, η_S, β_S, σ_S
# Regime-specific (10): PU, PS, α_U, a_Γ, b_Γ, k_U, λ_U, k_S, ξ_S, λ_S

EXTERNALLY_CALIBRATED = [
    (:r,   "r",            raw"$r$",              "Discount rate"),
    (:ν,   "nu",           raw"$\nu$",            "Demographic exit rate"),
    (:φ,   "phi",          raw"$\varphi$",        "Training completion rate"),
]

DEEP_STRUCTURAL = [
    (:a_ℓ,    :common, raw"$a_\ell$",       "Worker type shape"),
    (:b_ℓ,    :common, raw"$b_\ell$",       "Worker type shape"),
    (:c,      :common, raw"$c$",            "Training cost coeff."),
    (:bU,     :regime, raw"$b_U$",          "Unskilled UI flow"),
    (:bT,     :regime, raw"$b_T$",          "Training flow"),
    (:bS,     :regime, raw"$b_S$",          "Skilled UI flow"),
    (:μ,      :unsk,   raw"$\mu_U$",        "Unskilled matching eff."),
    (:η,      :unsk,   raw"$\eta_U$",       "Unskilled matching elas."),
    (:β,      :unsk,   raw"$\beta_U$",      "Unskilled bargaining"),
    (:μ,      :skl,    raw"$\mu_S$",        "Skilled matching eff."),
    (:η,      :skl,    raw"$\eta_S$",       "Skilled matching elas."),
    (:β,      :skl,    raw"$\beta_S$",      "Skilled bargaining"),
    (:σ,      :skl,    raw"$\sigma_S$",     "OJS flow cost"),
]

REGIME_SPECIFIC = [
    (:PU,   :regime, raw"$P_U$",            "Unskilled productivity"),
    (:PS,   :regime, raw"$P_S$",            "Skilled productivity"),
    (:α_U,  :regime, raw"$\alpha_U$",       "Unskilled damage shape"),
    (:a_Γ,  :regime, raw"$a_\Gamma$",       "Skilled offer shape"),
    (:b_Γ,  :regime, raw"$b_\Gamma$",       "Skilled offer shape"),
    (:k,    :unsk,   raw"$k_U$",            "Unskilled vacancy cost"),
    (:λ,    :unsk,   raw"$\lambda_U$",      "Unskilled damage rate"),
    (:k,    :skl,    raw"$k_S$",            "Skilled vacancy cost"),
    (:ξ,    :skl,    raw"$\xi_S$",          "Skilled exog. sep. rate"),
    (:λ,    :skl,    raw"$\lambda_S$",      "Skilled quality shock"),
]


"""
    _extract_param(bundle, field, block) → Float64

Extract a parameter value from an SMM bundle's optimum.
"""
function _extract_param(bundle, field::Symbol, block::Symbol)
    res  = bundle.result
    spec = bundle.spec
    cp, rp, up, sp = unpack_θ(res.theta_opt, spec)
    if     block == :common; return Float64(getfield(cp, field))
    elseif block == :regime; return Float64(getfield(rp, field))
    elseif block == :unsk;   return Float64(getfield(up, field))
    elseif block == :skl;    return Float64(getfield(sp, field))
    else error("Unknown block: $block")
    end
end


"""
    make_parameter_tables(; suffix)

Generate two LaTeX tables with parameter estimates.

Table A (`parameter_estimates_A{suffix}.tex`):
    Externally calibrated + Deep structural blocks.

Table B (`parameter_estimates_B{suffix}.tex`):
    Regime-specific block.

Each table has columns:
    Parameter | Description | Baseline FC | Crisis FC | Δ FC | Baseline COVID | Crisis COVID | Δ COVID

Deep structural parameters show baseline value in both Baseline and Crisis
columns (crisis = baseline by construction), with --- for Δ.
Regime-specific parameters show actual estimated values with computed Δ.
"""
function make_parameter_tables(; suffix::String=W_SUFFIX)
    println("\n── Generating parameter estimates tables (A & B) ──"); flush(stdout)

    # Load externally calibrated values
    calib = load_calibrated_params(; derived_dir=DERIVED_DIR)

    fmt(x)  = isfinite(x) ? @sprintf("%.5f", x) : "--"
    fmtd(x) = isfinite(x) ? @sprintf("%+.5f", x) : "--"

    # ── TABLE A: Externally calibrated + Deep structural ─────────────────

    lines_A = String[]
    push!(lines_A, "\\begin{table}[htbp]")
    push!(lines_A, "  \\centering")
    push!(lines_A, "  \\caption{Parameter Estimates (A): Calibrated and Deep Structural}")
    push!(lines_A, "  \\label{tab:param_estimates_a}")
    push!(lines_A, "  \\small")
    push!(lines_A, "  \\begin{tabular}{llcccccc}")
    push!(lines_A, "    \\toprule")
    push!(lines_A, "    & & \\multicolumn{3}{c}{Financial Crisis} & \\multicolumn{3}{c}{COVID} \\\\")
    push!(lines_A, "    \\cmidrule(lr){3-5} \\cmidrule(lr){6-8}")
    push!(lines_A, "    Parameter & Description & Baseline & Crisis & \$\\Delta\$ & Baseline & Crisis & \$\\Delta\$ \\\\")
    push!(lines_A, "    \\midrule")

    # Block 1: Externally calibrated
    push!(lines_A, "    \\multicolumn{8}{l}{\\textit{Externally calibrated}} \\\\[2pt]")
    for (field, key, latex, desc) in EXTERNALLY_CALIBRATED
        val = if field == :r;   calib.r
              elseif field == :ν; calib.nu
              else               calib.phi
              end
        fv = @sprintf("%.5f", val)
        push!(lines_A, "    \\quad $latex & $desc & $fv & $fv & --- & $fv & $fv & --- \\\\")
    end
    push!(lines_A, "    [4pt]")
    push!(lines_A, "    \\hline")

    # Block 2: Deep structural
    push!(lines_A, "    \\multicolumn{8}{l}{\\textit{Deep structural}} \\\\[2pt]")
    for (field, block, latex, desc) in DEEP_STRUCTURAL
        vals = Dict{Symbol, Float64}()
        for w in ALL_WINDOWS
            if haskey(smm_bundles, w)
                vals[w] = _extract_param(smm_bundles[w], field, block)
            end
        end

        # Deep structural: estimated in baseline, identical in crisis by construction
        base_fc_val      = get(vals, :base_fc,    NaN)
        base_covid_val   = get(vals, :base_covid,  NaN)
        # Crisis columns show the same value as baseline (deep structural params don't
        # shift with the cycle); delta is --- because there is no cross-regime variation
        crisis_fc_val    = get(vals, :crisis_fc,   base_fc_val)
        crisis_covid_val = get(vals, :crisis_covid, base_covid_val)

        push!(lines_A,
              "    \\quad $latex & $desc & $(fmt(base_fc_val)) & $(fmt(crisis_fc_val)) & --- & $(fmt(base_covid_val)) & $(fmt(crisis_covid_val)) & --- \\\\")
    end
    push!(lines_A, "    [4pt]")

    push!(lines_A, "    \\bottomrule")
    push!(lines_A, "  \\end{tabular}")
    push!(lines_A, "\\end{table}")

    outpath_A = joinpath(TABLES_DIR, "parameter_estimates_A$(suffix).tex")
    open(outpath_A, "w") do io
        write(io, join(lines_A, "\n"))
    end
    @printf("  Saved: %s\n", outpath_A); flush(stdout)

    # ── TABLE B: Regime-specific ──────────────────────────────────────────

    lines_B = String[]
    push!(lines_B, "\\begin{table}[htbp]")
    push!(lines_B, "  \\centering")
    push!(lines_B, "  \\caption{Parameter Estimates (B): Regime-Specific}")
    push!(lines_B, "  \\label{tab:param_estimates_b}")
    push!(lines_B, "  \\small")
    push!(lines_B, "  \\begin{tabular}{llcccccc}")
    push!(lines_B, "    \\toprule")
    push!(lines_B, "    & & \\multicolumn{3}{c}{Financial Crisis} & \\multicolumn{3}{c}{COVID} \\\\")
    push!(lines_B, "    \\cmidrule(lr){3-5} \\cmidrule(lr){6-8}")
    push!(lines_B, "    Parameter & Description & Baseline & Crisis & \$\\Delta\$ & Baseline & Crisis & \$\\Delta\$ \\\\")
    push!(lines_B, "    \\midrule")

    push!(lines_B, "    \\multicolumn{8}{l}{\\textit{Regime-specific}} \\\\[2pt]")
    for (field, block, latex, desc) in REGIME_SPECIFIC
        vals = Dict{Symbol, Float64}()
        for w in ALL_WINDOWS
            if haskey(smm_bundles, w)
                vals[w] = _extract_param(smm_bundles[w], field, block)
            end
        end

        base_fc_val      = get(vals, :base_fc,    NaN)
        crisis_fc_val    = get(vals, :crisis_fc,   NaN)
        base_covid_val   = get(vals, :base_covid,  NaN)
        crisis_covid_val = get(vals, :crisis_covid, NaN)

        delta_fc    = (isfinite(crisis_fc_val) && isfinite(base_fc_val)) ?
                      crisis_fc_val - base_fc_val : NaN
        delta_covid = (isfinite(crisis_covid_val) && isfinite(base_covid_val)) ?
                      crisis_covid_val - base_covid_val : NaN

        push!(lines_B,
              "    \\quad $latex & $desc & $(fmt(base_fc_val)) & $(fmt(crisis_fc_val)) & $(fmtd(delta_fc)) & $(fmt(base_covid_val)) & $(fmt(crisis_covid_val)) & $(fmtd(delta_covid)) \\\\")
    end
    push!(lines_B, "    [4pt]")

    push!(lines_B, "    \\bottomrule")
    push!(lines_B, "  \\end{tabular}")
    push!(lines_B, "\\end{table}")

    outpath_B = joinpath(TABLES_DIR, "parameter_estimates_B$(suffix).tex")
    open(outpath_B, "w") do io
        write(io, join(lines_B, "\n"))
    end
    @printf("  Saved: %s\n", outpath_B); flush(stdout)

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
To avoid grid-snapping (the GL grid is coarse), we interpolate the
crossing point of `net_T(x) − U_search(x)` rather than taking the
first grid node where τT > 0.5.

Returns NaN if no worker trains (τT ≡ 0) or everyone trains (τT ≡ 1).
"""
function _get_x_bar(bundle; Nx_fine::Int = 400)
    res  = bundle.result
    spec = bundle.spec
    sim  = bundle.sim

    cp, rp, up, sp = unpack_θ(res.theta_opt, spec)
    # Solve on a finer grid so the interpolated cutoff is precise
    model, sol = solve_model(cp, rp, up, sp, sim;
                             Nx=Nx_fine, Np_U=spec.run.Np_U, Np_S=spec.run.Np_S)

    if !sol.ok
        @warn "Model did not fully converge at stored optimum — x̄ may be unreliable"
    end

    obj = compute_equilibrium_objects(model)
    xg  = obj.xg

    # gap(x) = (−c(x) + T(x)) − U_search(x);  training chosen where gap ≥ 0
    gap = obj.net_T .- obj.Usearch

    # Find zero crossing via linear interpolation
    for i in 1:length(xg)-1
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
# MAIN — Generate everything
# ============================================================

println("\n" * "="^60)
println("  Generating all outputs")
println("="^60)
flush(stdout)

# 1. Model fit tables (A & B)
make_model_fit_tables()

# 2. Parameter estimates tables (A & B)
make_parameter_tables()

# 3. Training cutoff (x̄) table
make_xbar_table()

# 4. Model fit scatter plots (one per baseline window)
for w in [:base_fc, :base_covid]
    make_model_fit_scatter(; window=w)
end

# Note: Transition dynamics panels are generated by transition_panel.jl.
# Run that script separately to produce the transition dynamics plots.

println("\n" * "="^60)
println("  All outputs generated successfully.")
println("="^60)

############################################################
# code/policies/policy_main.jl — Education-subsidy policy
#                                  counterfactual entry point
#
# Usage (from project root):
#   julia --threads auto code/policies/policy_main.jl
#
# Workflow:
#   1. Load the estimated baseline SMM result (.jls)
#   2. Reconstruct the parameter structs
#   3. Solve the baseline stationary equilibrium (for reference)
#   4. For each policy × intensity, perturb parameters and
#      solve the new stationary equilibrium
#   5. Collect outcomes, produce tables and figures
#
# User-configurable options:
#   BASELINE_WINDOW   :base_fc   or  :base_covid
#   W_COND_TARGET     weight-matrix mode (must match SMM run)
#   INTENSITIES       subsidy levels to evaluate
#   Nx, Np_U, Np_S    grid sizes (match SMM estimation)
#
# Project layout:
#   code/
#     solver/      ← loaded as a library
#     smm/         ← smm_params.jl, smm.jl (for unpack_θ)
#     policies/
#       policy_main.jl         ← this file
#       policy_params.jl
#       policy_solver.jl
#       policy_plots.jl
#   output/
#     smm/                     ← .jls estimation results
#     plots/                   ← figures saved here
#     tables/                  ← tables saved here
############################################################

# ═══════════════════════════════════════════════════════════
# 1. User configuration
# ═══════════════════════════════════════════════════════════

# Which baseline to use for the counterfactuals
const BASELINE_WINDOW = :base_fc

# Weight-matrix mode (must match the SMM estimation)
const W_COND_TARGET = 1e8     # 0.0 = diagonal, 1.0 = compressed, 2.0 = equal, >2.0 = full

# Education-subsidy intensities (Section 6.2)
const INTENSITIES_EDUC = [0.10, 0.25, 0.50]

# Outside-option intensities (Section 6.1)
const INTENSITIES_OO = [0.01, 0.05, 0.10, 0.20]

# Grid sizes (should match SMM estimation for consistency)
const GRID_Nx   = 80
const GRID_Np_U = 80
const GRID_Np_S = 80

# ═══════════════════════════════════════════════════════════
# 2. Packages
# ═══════════════════════════════════════════════════════════

println("="^65)
println("  Segmented Search Model — Education-Subsidy Counterfactuals")
println("="^65)
flush(stdout)

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
using Base.Threads
using Serialization
using CSV
using DataFrames
using Plots
using LaTeXStrings
using StatsPlots          # for groupedbar

println("done."); flush(stdout)

# ═══════════════════════════════════════════════════════════
# 3. Paths
# ═══════════════════════════════════════════════════════════

const POLICY_DIR   = @__DIR__
const SOLVER_DIR   = joinpath(POLICY_DIR, "..", "solver")
const SMM_DIR      = joinpath(POLICY_DIR, "..", "smm")
const PROJECT_ROOT = joinpath(POLICY_DIR, "..", "..")
const OUTPUT_DIR   = joinpath(PROJECT_ROOT, "output")
const SMM_OUT_DIR  = joinpath(OUTPUT_DIR, "smm")
const PLOTS_DIR    = joinpath(OUTPUT_DIR, "plots")
const TABLES_DIR   = joinpath(OUTPUT_DIR, "tables")

# ═══════════════════════════════════════════════════════════
# 4. Load solver modules
# ═══════════════════════════════════════════════════════════

print("Loading solver modules... "); flush(stdout)
include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))
println("done."); flush(stdout)

# ═══════════════════════════════════════════════════════════
# 5. Load SMM modules (for _load_smm_bundle, unpack_θ)
# ═══════════════════════════════════════════════════════════

print("Loading SMM modules... "); flush(stdout)
include(joinpath(SMM_DIR, "moments.jl"))
include(joinpath(SMM_DIR, "smm_params.jl"))
include(joinpath(SMM_DIR, "smm.jl"))
println("done."); flush(stdout)

# ═══════════════════════════════════════════════════════════
# 6. Load policy modules
# ═══════════════════════════════════════════════════════════

print("Loading policy modules... "); flush(stdout)
include(joinpath(POLICY_DIR, "policy_params.jl"))
include(joinpath(POLICY_DIR, "policy_solver.jl"))
include(joinpath(POLICY_DIR, "policy_plots.jl"))
println("done."); flush(stdout)

@printf("Threads available: %d\n\n", Threads.nthreads())
flush(stdout)

# ═══════════════════════════════════════════════════════════
# 7. Determine weight-matrix suffix and file path
# ═══════════════════════════════════════════════════════════

function _w_suffix(cond_target::Float64)
    cond_target == 0.0 && return "_diagonalW"
    cond_target == 1.0 && return "_compressedW"
    cond_target == 2.0 && return "_equalW"
    return "_fullW"
end

const W_SUFFIX = _w_suffix(W_COND_TARGET)

baseline_jls = joinpath(SMM_OUT_DIR, "smm_result_$(BASELINE_WINDOW)$(W_SUFFIX).jls")

@printf("  Baseline window:  %s\n", BASELINE_WINDOW)
@printf("  W suffix:         %s\n", W_SUFFIX)
@printf("  SMM result file:  %s\n\n", baseline_jls)
flush(stdout)

# ═══════════════════════════════════════════════════════════
# 8. Load SMM bundle and reconstruct parameters
# ═══════════════════════════════════════════════════════════

isfile(baseline_jls) || error(
    "Baseline SMM result not found at $baseline_jls.\n" *
    "Run the SMM estimation first (code/smm/main.jl).")

println("Loading SMM result...")
flush(stdout)

baseline_bundle = _load_smm_bundle(baseline_jls; delete_on_fail=false,
                                    label="baseline SMM result")
isnothing(baseline_bundle) && error(
    "Baseline SMM result is unreadable or stale: $baseline_jls")

cp_base, rp_base, up_base, sp_base = unpack_θ(
    baseline_bundle.result.theta_opt, baseline_bundle.spec)

@printf("  Baseline Q = %.6e  (converged = %s)\n",
        baseline_bundle.result.loss_opt, baseline_bundle.result.converged)
flush(stdout)

# SimParams for solving — use standalone-quality settings, not the
# fast SMM settings (which have lower maxit and looser tolerances).
# The SMM bundle's sim is tuned for speed inside the optimizer loop;
# here we solve each equilibrium once and need it to converge reliably.
sim = SimParams(
    tol_inner      = 1e-8,
    tol_outer_U    = 1e-6,
    tol_outer_S    = 1e-7,
    tol_global     = 1e-3,

    maxit_inner    = 500,
    maxit_outer    = 300,
    maxit_global   = 50,

    conv_streak    = 2,

    use_anderson   = true,
    anderson_m     = 1,
    anderson_reg   = 1e-10,

    damp_pstar_U   = 1.30,
    damp_pstar_S   = 1.00,

    verbose        = 0,
    verbose_stride = 10,
)

# ═══════════════════════════════════════════════════════════
# 9. Print baseline parameters
# ═══════════════════════════════════════════════════════════

println("\n  Baseline parameters:")
@printf("    common:  r=%.4f  ν=%.4f  φ=%.4f  c=%.4f  a_ℓ=%.4f  b_ℓ=%.4f\n",
        cp_base.r, cp_base.ν, cp_base.φ, cp_base.c, cp_base.a_ℓ, cp_base.b_ℓ)
@printf("    regime:  PU=%.4f  PS=%.4f  bU=%.4f  bT=%.4f  bS=%.4f\n",
        rp_base.PU, rp_base.PS, rp_base.bU, rp_base.bT, rp_base.bS)
@printf("    regime:  α_U=%.4f  a_Γ=%.4f  b_Γ=%.4f\n",
        rp_base.α_U, rp_base.a_Γ, rp_base.b_Γ)
@printf("    unsk:    μ=%.4f  η=%.4f  k=%.4f  β=%.4f  λ=%.4f\n",
        up_base.μ, up_base.η, up_base.k, up_base.β, up_base.λ)
@printf("    skl:     μ=%.4f  η=%.4f  k=%.4f  β=%.4f  ξ=%.4f  λ=%.4f  σ=%.4f\n",
        sp_base.μ, sp_base.η, sp_base.k, sp_base.β, sp_base.ξ, sp_base.λ, sp_base.σ)
flush(stdout)

# ═══════════════════════════════════════════════════════════
# 10. Education-subsidy exercise (Section 6.2)
# ═══════════════════════════════════════════════════════════

baseline_label = BASELINE_WINDOW == :base_fc ? "Pre-FC" : "Pre-COVID"

println("\n" * "="^65)
println("  Exercise 1: Education Subsidies")
println("="^65)
flush(stdout)

specs_educ = build_policy_experiments(; intensities = INTENSITIES_EDUC)
@printf("  Experiments: %d\n\n", length(specs_educ))
flush(stdout)

table_educ = solve_all_policies(
    specs_educ, cp_base, rp_base, up_base, sp_base, sim;
    baseline_label = baseline_label,
    Nx = GRID_Nx, Np_U = GRID_Np_U, Np_S = GRID_Np_S,
)

make_all_policy_outputs(table_educ; output_dir = OUTPUT_DIR)

# ═══════════════════════════════════════════════════════════
# 11. Outside-option exercise (Section 6.1)
# ═══════════════════════════════════════════════════════════

println("\n" * "="^65)
println("  Exercise 2: Outside Options")
println("="^65)
flush(stdout)

specs_oo = build_outside_option_experiments(; intensities = INTENSITIES_OO)
@printf("  Experiments: %d\n\n", length(specs_oo))
flush(stdout)

table_oo = solve_all_policies(
    specs_oo, cp_base, rp_base, up_base, sp_base, sim;
    baseline_label = baseline_label * " OO",
    Nx = GRID_Nx, Np_U = GRID_Np_U, Np_S = GRID_Np_S,
)

make_all_policy_outputs(table_oo; output_dir = OUTPUT_DIR)

# ═══════════════════════════════════════════════════════════
# 12. Done
# ═══════════════════════════════════════════════════════════

println("\n" * "="^65)
println("  All policy counterfactual exercises complete.")
println("="^65)
flush(stdout)

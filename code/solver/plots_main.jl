############################################################
# code/solver/plots_main.jl — Single-run plots from SMM estimation
#
# Usage (from project root):
#   julia --threads auto code/solver/plots_main.jl
#
# Loads parameters from a saved SMM bundle
#   output/smm/smm_result_<WINDOW><W_SUFFIX>.jls
# solves the model once at those estimates, and writes the
# full set of equilibrium figures defined in single_run_plots.jl
# to output/plots/<WINDOW><W_SUFFIX>/.
############################################################

println("="^60)
println("  Segmented Search Model — Single-Run Plots from Estimation")
println("="^60)
flush(stdout)

# Paths
const SOLVER_DIR   = @__DIR__
const SMM_DIR      = joinpath(SOLVER_DIR, "..", "smm")
const PROJECT_ROOT = joinpath(SOLVER_DIR, "..", "..")
const OUTPUT_DIR   = joinpath(PROJECT_ROOT, "output")
const SMM_OUT_DIR  = joinpath(OUTPUT_DIR, "smm")
const PLOTS_ROOT   = joinpath(OUTPUT_DIR, "plots")

isdir(SMM_DIR) || error(
    "SMM module not found at:\n  $SMM_DIR\n" *
    "Run code/solver/model_main.jl for a standalone solve, or wire up " *
    "the SMM module before using this script.")

# Packages
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
using Optim
using CSV
using DataFrames
using Clustering

using Plots
using LaTeXStrings

println("done."); flush(stdout)

Random.seed!(1234)

# Solver modules
print("Loading solver modules... "); flush(stdout)

include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))

println("done."); flush(stdout)

# SMM modules (needed to deserialize SMMResult / SMMSpec)
print("Loading SMM modules... "); flush(stdout)

include(joinpath(SMM_DIR, "moments.jl"))
include(joinpath(SMM_DIR, "smm_params.jl"))
include(joinpath(SMM_DIR, "smm.jl"))

println("done."); flush(stdout)

# Plotting library
print("Loading plotting modules... "); flush(stdout)

include(joinpath(SOLVER_DIR, "single_run_plots.jl"))

println("done."); flush(stdout)
@printf("Threads available: %d\n\n", Threads.nthreads())
flush(stdout)

# ============================================================
# 1. Estimation to load
# ============================================================
WINDOW        = :base_fc
W_COND_TARGET = 2.0

function _w_suffix(cond_target::Float64)
    cond_target == 0.0 && return "_diagonalW"
    cond_target == 1.0 && return "_compressedW"
    cond_target == 2.0 && return "_equalW"
    return "_fullW"
end
const W_SUFFIX = _w_suffix(W_COND_TARGET)

const SMM_JLS_PATH = joinpath(SMM_OUT_DIR,
    "smm_result_$(WINDOW)$(W_SUFFIX).jls")
const PLOTS_DIR = joinpath(PLOTS_ROOT, string(WINDOW) * W_SUFFIX)

@printf("Loading estimation bundle:\n  %s\n", SMM_JLS_PATH)
flush(stdout)

isfile(SMM_JLS_PATH) || error(
    "Estimation bundle not found at:\n  $SMM_JLS_PATH\n" *
    "Run code/smm/smm_main.jl first with WINDOW = :$WINDOW and " *
    "W_COND_TARGET = $W_COND_TARGET.")

bundle = _load_smm_bundle(SMM_JLS_PATH; delete_on_fail=false,
                          label="estimation bundle")
isnothing(bundle) && error(
    "Could not deserialize estimation bundle at:\n  $SMM_JLS_PATH\n" *
    "Likely a stale on-disk format. Re-run code/smm/smm_main.jl " *
    "with the same WINDOW and W_COND_TARGET.")

const result_smm = bundle.result
const spec_smm   = bundle.spec
const sim_smm    = bundle.sim

common, regime, unsk_par, skl_par = unpack_θ(result_smm.theta_opt, spec_smm)

@printf("Loaded estimation:\n")
@printf("  Window     = %s\n", WINDOW)
@printf("  W mode     = %s  (target κ = %.1e)\n",
        lstrip(W_SUFFIX, '_'), W_COND_TARGET)
@printf("  SMM Q*     = %.6e\n", result_smm.loss_opt)
@printf("  Converged  = %s   (iters = %d)\n",
        result_smm.converged, result_smm.iterations)
@printf("  Free / Fixed parameters = %d / %d\n",
        length(spec_smm.free), length(spec_smm.fixed))
@printf("  Plots will be written to:\n    %s\n", PLOTS_DIR)
flush(stdout)

# ============================================================
# 2. Solver settings for this single run
# ============================================================
USE_SMM_SIM = false

sim = USE_SMM_SIM ? sim_smm : SimParams(
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

    damp_pstar_U   = 1.00,
    damp_pstar_S   = 1.00,

    verbose        = 2,
    verbose_stride = 10,
)

if USE_SMM_SIM
    println("\nUsing SimParams from the SMM bundle (USE_SMM_SIM = true).")
else
    println("\nUsing tightened single-run SimParams (USE_SMM_SIM = false).")
end
flush(stdout)

# ============================================================
# 3. Echo the loaded parameter values
# ============================================================
println("\nParameters loaded from estimation:")
@printf("  CommonParams:    r=%.5f   ν=%.5f   φ=%.5f\n",
        common.r, common.ν, common.φ)
@printf("                   a_ℓ=%.5f  b_ℓ=%.5f  c=%.5f\n",
        common.a_ℓ, common.b_ℓ, common.c)
@printf("  RegimeParams:    PU=%.5f   γ_PS=%.5f\n",
        regime.PU, regime.gamma_PS)
@printf("                   bU=%.5f   bT=%.5f   bS=%.5f\n",
        regime.bU, regime.bT, regime.bS)
@printf("                   α_U=%.5f  a_Γ=%.5f  b_Γ=%.5f\n",
        regime.α_U, regime.a_Γ, regime.b_Γ)
@printf("  UnskilledParams: μ=%.5f   η=%.5f   k=%.5f   β=%.5f   λ=%.5f\n",
        unsk_par.μ, unsk_par.η, unsk_par.k, unsk_par.β, unsk_par.λ)
@printf("  SkilledParams:   μ=%.5f   η=%.5f   k=%.5f   β=%.5f   λ=%.5f   σ=%.5f\n",
        skl_par.μ, skl_par.η, skl_par.k, skl_par.β, skl_par.λ, skl_par.σ)
flush(stdout)

# ============================================================
# 4. Solve
# ============================================================
println("\nSolving model...")
@time model, result = solve_model(common, regime, unsk_par, skl_par, sim;
                                   Nx=200, Np_U=200, Np_S=200)

if result.ok
    @printf("Solver converged  (U=%s  S=%s  global=%s)\n",
            result.converged_U, result.converged_S, result.converged_global)
else
    @printf("WARNING: solver did not fully converge  (U=%s  S=%s  global=%s)\n",
            result.converged_U, result.converged_S, result.converged_global)
    @printf("Plots will still be produced from the (non-converged) model state.\n")
end
flush(stdout)

# ============================================================
# 5. Equilibrium objects and accounting
# ============================================================
obj = compute_equilibrium_objects(model)
print_accounting(obj)

# ============================================================
# 6. Generate all single-run plots
# ============================================================
println("\nGenerating figures...")
flush(stdout)

@time make_all_plots(obj; output_dir = PLOTS_DIR,
                          gamma_PS    = regime.gamma_PS)

println("\nDone.")
flush(stdout)

############################################################
# code/solver/model_main.jl — Standalone single solve + plots
#
# Usage (from project root):
#   julia --threads auto code/solver/model_main.jl
#
# Solves the model at the hard-coded parameters below and writes
# the full set of equilibrium figures from single_run_plots.jl to
# OUTPUT_DIR.  To plot at SMM-estimated parameters, use
# code/solver/plots_main.jl instead.
############################################################

println("="^60)
println("  Segmented Search Model — Standalone Single-Run Plots")
println("="^60)
flush(stdout)

# Paths
const SOLVER_DIR   = @__DIR__
const PROJECT_ROOT = joinpath(SOLVER_DIR, "..", "..")
const OUTPUT_DIR   = joinpath(PROJECT_ROOT, "output")
const PLOTS_ROOT   = joinpath(OUTPUT_DIR, "plots")

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

# Plotting library
print("Loading plotting modules... "); flush(stdout)

include(joinpath(SOLVER_DIR, "single_run_plots.jl"))

println("done."); flush(stdout)
@printf("Threads available: %d\n\n", Threads.nthreads())
flush(stdout)

# ============================================================
# 1. Parameters
# ============================================================
common = CommonParams(
    r   = 0.05,
    ν   = 0.05,
    φ   = 0.20,
    a_ℓ = 2.0,
    b_ℓ = 5.0,
    c   = 1.70,
)

unsk_par = UnskilledParams(
    μ   = 0.74,
    η   = 0.60,
    k   = 0.25,
    β   = 0.40,
    λ   = 0.08,
    PU  = 0.70,
    bU  = 0.00,
    bT  = 0.28,
    α_U = 1.00,
)

skl_par = SkilledParams(
    μ        = 0.90,
    η        = 0.50,
    k        = 0.17,
    β        = 0.32,
    λ        = 0.07,
    σ        = 0.01,
    gamma_PS = 1.85,
    bS       = 0.01,
    a_Γ      = 2.0,
    b_Γ      = 5.0,
)

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

    damp_pstar_U   = 1.00,
    damp_pstar_S   = 1.00,

    verbose        = 2,
    verbose_stride = 10,
)

const PLOTS_DIR = joinpath(PLOTS_ROOT, "standalone_default")

println("Parameters:")
@printf("  CommonParams:    r=%.5f   ν=%.5f   φ=%.5f\n",
        common.r, common.ν, common.φ)
@printf("                   a_ℓ=%.5f  b_ℓ=%.5f  c=%.5f\n",
        common.a_ℓ, common.b_ℓ, common.c)
@printf("  Unsk (regime):   PU=%.5f   bU=%.5f   bT=%.5f   α_U=%.5f\n",
        unsk_par.PU, unsk_par.bU, unsk_par.bT, unsk_par.α_U)
@printf("  Skl  (regime):   γ_PS=%.5f  bS=%.5f  a_Γ=%.5f  b_Γ=%.5f\n",
        skl_par.gamma_PS, skl_par.bS, skl_par.a_Γ, skl_par.b_Γ)
@printf("  UnskilledParams: μ=%.5f   η=%.5f   k=%.5f   β=%.5f   λ=%.5f\n",
        unsk_par.μ, unsk_par.η, unsk_par.k, unsk_par.β, unsk_par.λ)
@printf("  SkilledParams:   μ=%.5f   η=%.5f   k=%.5f   β=%.5f   λ=%.5f   σ=%.5f\n",
        skl_par.μ, skl_par.η, skl_par.k, skl_par.β, skl_par.λ, skl_par.σ)
flush(stdout)

# ============================================================
# 2. Solve
# ============================================================
println("\nSolving model...")
@time model, result = solve_model(common, unsk_par, skl_par, sim;
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
# 3. Equilibrium objects and accounting
# ============================================================
obj = compute_equilibrium_objects(model)
print_accounting(obj)

# ============================================================
# 4. Generate all single-run plots
# ============================================================
println("\nGenerating figures...")
flush(stdout)

@time make_all_plots(obj; output_dir = PLOTS_DIR,
                          gamma_PS    = skl_par.gamma_PS)

println("\nDone.")
flush(stdout)

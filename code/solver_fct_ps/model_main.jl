############################################################
# code/solver_fct_ps/main.jl — Single model run
#
# Usage (from project root):
#   julia --threads auto code/solver_fct_ps/main.jl
#
# Solves the model once at the parameter values entered below,
# prints equilibrium accounting and moment diagnostics.
############################################################

println("="^60)
println("  Segmented Search Model — Single Run (PS(x) = γ·x^{γ−1})")
println("="^60)
flush(stdout)

# ── Paths ─────────────────────────────────────────────────────────────────────
const SOLVER_DIR   = @__DIR__
const SMM_DIR      = joinpath(SOLVER_DIR, "..", "smm_fct_ps")
const PROJECT_ROOT = joinpath(SOLVER_DIR, "..", "..")
const OUTPUT_DIR   = joinpath(PROJECT_ROOT, "output")

# ── Packages ──────────────────────────────────────────────────────────────────
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

println("done."); flush(stdout)

Random.seed!(1234)

# ── Load solver ───────────────────────────────────────────────────────────────
print("Loading solver modules... "); flush(stdout)

include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))

println("done."); flush(stdout)
@printf("Threads available: %d\n\n", Threads.nthreads())
flush(stdout)

# ============================================================
# 1. Solver settings
# ============================================================
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

# ============================================================
# 2. Parameters — EDIT VALUES HERE
# ============================================================

common = CommonParams(
    r   = 0.00417,
    ν   = 0.03841,
    φ   = 0.02222,
    a_ℓ = 1.46357,
    b_ℓ = 4.90168,
    c   = 3.50000,
)

regime = RegimeParams(
    PU       = 1.01122,
    gamma_PS = 2.58787,       # ← PS(x) = γ · x^{γ−1}
    bU       = 0.00000,
    bT       = 0.70000,
    bS       = 1.00000,
    α_U      = 6.32697,
    a_Γ      = 2.81289,
    b_Γ      = 0.97088,
)

unsk_par = UnskilledParams(
    μ = 0.70375,
    η = 0.50000,
    k = 0.06274,
    β = 0.50000,
    λ = 0.02563,
)

skl_par = SkilledParams(
    μ = 1.45977,
    η = 0.50000,
    k = 0.15255,
    β = 0.50000,
    ξ = 0.00000,
    λ = 0.09548,
    σ = 1.10000,
)

# ============================================================
# 3. Solve
# ============================================================
println("Solving model...")
@time model, result = solve_model(common, regime, unsk_par, skl_par, sim;
                                   Nx=200, Np_U=200, Np_S=200)

if result.ok
    @printf("Solver converged  (U=%s  S=%s  global=%s)\n",
            result.converged_U, result.converged_S, result.converged_global)
else
    @printf("WARNING: solver did not fully converge  (U=%s  S=%s  global=%s)\n",
            result.converged_U, result.converged_S, result.converged_global)
end

# ============================================================
# 4. Equilibrium objects and accounting
# ============================================================
obj = compute_equilibrium_objects(model)
print_accounting(obj)

println("\nDone.")
flush(stdout)
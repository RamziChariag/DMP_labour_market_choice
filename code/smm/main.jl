############################################################
# code/smm/main.jl — SMM estimation entry point
#
# Usage (from project root):
#   julia --threads auto code/smm/main.jl
#
# Project layout:
#   code/
#     solver/   ← loaded as a library
#     smm/
#       main.jl        ← this file
#       moments.jl
#       smm_params.jl
#       smm.jl
#   output/
#     plots/
#     tables/
############################################################

println("="^60)
println("  Segmented Search Model — SMM Estimation")
println("="^60)
flush(stdout)

# ── Paths ─────────────────────────────────────────────────────────────────────
const SMM_DIR      = @__DIR__
const SOLVER_DIR   = joinpath(SMM_DIR, "..", "solver")
const PROJECT_ROOT = joinpath(SMM_DIR, "..", "..")
const OUTPUT_DIR   = joinpath(PROJECT_ROOT, "output")
const TABLES_DIR   = joinpath(OUTPUT_DIR, "tables")

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
using Optim

println("done."); flush(stdout)

Random.seed!(1234)

# ── Load solver (as a library — no solving happens yet) ───────────────────────
print("Loading solver modules... "); flush(stdout)

include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))

println("done."); flush(stdout)

# ── Load SMM scripts ──────────────────────────────────────────────────────────
print("Loading SMM modules... "); flush(stdout)

include(joinpath(SMM_DIR, "moments.jl"))
include(joinpath(SMM_DIR, "smm_params.jl"))
include(joinpath(SMM_DIR, "smm.jl"))

println("done."); flush(stdout)
@printf("Threads available: %d\n\n", Threads.nthreads())
flush(stdout)

# ============================================================
# 1. Solver settings (always fixed)
#    Use smaller tolerances than in the single-solve script
#    to keep each SMM iteration fast.
# ============================================================
sim_smm = SimParams(
    tol_inner      = 1e-6,
    tol_outer_U    = 1e-5,
    tol_outer_S    = 1e-4,
    tol_global     = 1e-2,

    maxit_inner    = 300,
    maxit_outer    = 200,
    maxit_global   = 30,

    conv_streak    = 2,

    use_anderson   = true,
    anderson_m     = 1,
    anderson_reg   = 1e-10,

    damp_pstar_U   = 1.30,
    damp_pstar_S   = 0.02,

    verbose        = 0,          # silent during SMM iterations
    verbose_stride = 10,
)

# ============================================================
# 2. Data moments
#    Loaded from moments.jl (placeholder values for now).
# ============================================================
moments = load_data_moments()

# ============================================================
# 3. Fixed parameters
#    Any parameter listed here is excluded from estimation.
#    Leave a parameter out of this NamedTuple to estimate it.
#
#    Example: pin discount rate, demographic rate, and exogenous
#    separation as externally calibrated.
# ============================================================
fixed_params = (
;
)

# ============================================================
# 4. Free parameter list
#    Start from the full default list and trim if desired.
#    To estimate only a subset, build the vector manually:
#
#    free = [p for p in default_free_params()
#              if p.block in (:regime, :unsk)]
# ============================================================
free_params = default_free_params()

# ============================================================
# 5. Run parameters — grids, DE, Nelder-Mead, tracing
# ============================================================
run_params = SMMRunParams(
    # ── Grids (coarser = faster per iteration) ──────────────
    Nx      = 80,
    Np_U    = 80,
    Np_S    = 80,

    # ── DE global search ────────────────────────────────────
    de_max_iter  = 200,       # generations; total evals = max_iter × pop_size
    de_pop_size  = 600,       # 0 = auto (100 × n_free_params)
    de_f         = 0.65,
    de_cr        = 0.85,
    de_patience  = 20,

    # ── Nelder-Mead polish ───────────────────────────────────
    nm_max_iter  = 5_000,
    nm_f_tol     = 1e-6,
    nm_x_tol     = 1e-5,

    # ── Tracing ─────────────────────────────────────────────
    show_trace_members     = false,   # per-member lines within each generation
    show_trace_generations = true,    # end-of-generation summary lines
    trace_stride           = 10,
)

# ============================================================
# 6. Build SMM spec
# ============================================================
spec = build_smm_spec(
    moments, sim_smm;
    fixed      = fixed_params,
    free_specs = free_params,
    run        = run_params,
)

print_spec(spec)

# ============================================================
# 7. Run estimation
# ============================================================
println("Starting SMM optimisation..."); flush(stdout)

# Stage 1: global search
res_de = run_smm(spec; method = :de)

# Stage 2: polish from DE solution
res_pol = run_smm(_spec_with_init(spec, res_de.theta_opt); method = :neldermead)

results = res_pol

# ============================================================
# 8. Save results
# ============================================================
mkpath(TABLES_DIR)
save_results(results, joinpath(TABLES_DIR, "smm_estimates.csv"))

println("\nDone.")
flush(stdout)

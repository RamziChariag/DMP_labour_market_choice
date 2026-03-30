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
const SMM_OUT_DIR  = joinpath(OUTPUT_DIR, "smm")

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
using CSV
using DataFrames
using Serialization

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
    tol_outer_U    = 1e-6,
    tol_outer_S    = 1e-6,
    tol_global     = 1e-3,

    maxit_inner    = 300,
    maxit_outer    = 200,
    maxit_global   = 30,

    conv_streak    = 2,

    use_anderson   = true,
    anderson_m     = 1,
    anderson_reg   = 1e-10,

    damp_pstar_U   = 1.30,
    damp_pstar_S   = 0.80,

    verbose        = 0,          # 0: model is silent; 1: print outer convergence info per iteration; 2: also print inner iteration details
    verbose_stride = 10,
)

# ============================================================
# 2. Select estimation window
#    Valid windows: :base_fc, :crisis_fc, :base_covid, :crisis_covid
# ============================================================
WINDOW = :base_fc    


@printf("Estimation window: %s\n", WINDOW)
flush(stdout)

# ============================================================
# 3. Data moments
#    Load from moments.jl. Attempt to read from CSV if derived
#    files are available, otherwise use placeholders.
# ============================================================
derived_dir = joinpath(PROJECT_ROOT, "data", "derived")
moments = load_data_moments(; window=WINDOW, derived_dir=derived_dir)

# ============================================================
# 4. Optimal weight matrix
#    Try to load the optimal weight matrix from influence functions.
#    If not available or ill-conditioned, construct diagonal matrix
#    from moment standard errors.
# ============================================================
W_opt = load_weight_matrix(; window=WINDOW, derived_dir=derived_dir)

# ============================================================
# 5. Fixed parameters
#    r, nu, phi are externally calibrated and FIXED across all
#    4 estimation windows.  r is set at 5% annual.  nu and phi
#    are computed from data (CPS panels and NSC completion rates)
#    in the data pipeline notebook (Stages 6-7) and loaded here.
# ============================================================
println("\nLoading externally calibrated parameters (r, \u03bd, \u03c6)...")
flush(stdout)
calib = load_calibrated_params(; derived_dir=derived_dir)

fixed_params = (
    r   = calib.r,    # discount rate  (5% annual, externally set)
    ν   = calib.nu,   # demographic turnover  (from CPS matched panels)
    φ   = calib.phi,  # training completion   (from NSC data)
)

@printf("  Fixed:  r = %.6f (= 0.05/12, monthly),  \u03bd = %.5f,  \u03c6 = %.5f\n",
        fixed_params.r, fixed_params.ν, fixed_params.φ)
flush(stdout)

# ============================================================
# 6. Free parameter list
#    Start from the full default list and trim if desired.
#    To estimate only a subset, build the vector manually:
#
#    free = [p for p in default_free_params()
#              if p.block in (:regime, :unsk)]
# ============================================================
free_params = default_free_params()

# ============================================================
# 7. Run parameters — grids, SA, Nelder-Mead, tracing
# ============================================================
run_params = SMMRunParams(
    # ── Grids (coarser = faster per iteration) ──────────────
    Nx      = 80,
    Np_U    = 80,
    Np_S    = 80,

    # ── SA global search ────────────────────────────────────
    sa_max_iter        = 10_000,  # total SA proposals
    sa_T0              = 5.0,     # initial temperature (higher = more uphill acceptance early)
    sa_step            = 0.20,    # initial random-walk step in logit space
    sa_cooling_rate    = 1.0,     # scales t in cooling schedule denominator
    sa_cooling_exp     = 0.75,     # exponent: T0/log(1+rate*t)^exp  (<1 = slower cooling)
    sa_reheat_patience = 300,     # proposals without improvement before reheating
    sa_reheat_factor   = 1.10,     # temperature multiplier on reheat
    sa_max_reheats     = 1,       # cap on total reheats (0 = unlimited)
    sa_adapt_window    = 50,      # rolling window for adaptive step (0 = off)
    sa_target_fin      = 0.90,    # target feasibility rate for adaptive step

    # ── DE global search ────────────────────────────────────
    de_max_iter  = 200,       # generations; total evals = max_iter × pop_size
    de_pop_size  = 600,       # 0 = auto (100 × n_free_params)
    de_f         = 0.70,        #factor for mutation (0.5-0.9 typical)
    de_cr        = 0.85,        #crossover probability (0-1)
    de_patience  = 10,      # how many generations to wait for improvement before early stopping
    de_avg_tol   = 1.00e-4,    # stop when (Q_mean − Q_best) / |Q_best| < this (1 %); set 0.0 to disable


    # ── Nelder-Mead polish ───────────────────────────────────
    nm_max_iter  = 200,       # maximum iterations for Nelder-Mead local search
    nm_f_tol     = 1e-6,        # stop when |Q_new − Q_old| < this; set 0.0 to disable
    nm_x_tol     = 1e-5,        # stop when max|θ_new − θ_old| < this; set 0.0 to disable

    # ── Tracing ─────────────────────────────────────────────
    show_trace_members     = false,   # per-member lines within each generation for DE, prints for stride proposal in SA
    show_trace_generations = true,    # end-of-generation summary lines
    trace_stride           = 10,        # how often to print within DE generations (in members, not generations)
)

# ============================================================
# 8. Build SMM spec
# ============================================================
spec = build_smm_spec(
    moments, sim_smm;
    fixed      = fixed_params,
    free_specs = free_params,
    run        = run_params,
    W          = W_opt,
)

print_spec(spec)

# ============================================================
# 9. Run estimation
# ============================================================
println("Starting SMM optimisation..."); flush(stdout)

# Stage 1: global search
res_sa = run_smm(spec; method = :sa)

# Stage 2: polish from SA solution
res_pol = run_smm(_spec_with_init(spec, res_sa.theta_opt); method = :neldermead)

results = res_pol

# ============================================================
# 10. Save results
# ============================================================
mkpath(TABLES_DIR)
mkpath(SMM_OUT_DIR)

# CSV table of parameter estimates
save_results(results, joinpath(TABLES_DIR, "smm_estimates_$(WINDOW).csv"))

# Serialize full result for the transition solver.
# The .jls file stores: (result, spec, sim_smm) so that
# transition_main.jl can reconstruct the solved model.
smm_jls_path = joinpath(SMM_OUT_DIR, "smm_result_$(WINDOW).jls")
open(smm_jls_path, "w") do io
    serialize(io, (result = results, spec = spec, sim = sim_smm))
end
@printf("Serialized SMM result → %s\n", smm_jls_path)

println("\nDone.")
flush(stdout)

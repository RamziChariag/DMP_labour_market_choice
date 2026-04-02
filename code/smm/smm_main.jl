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
using Clustering

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

    damp_pstar_U   = 1.00,
    damp_pstar_S   = 1.00,

    verbose        = 0,          # 0: model is silent; 1: print outer convergence info per iteration; 2: also print inner iteration details
    verbose_stride = 100,
)

# ============================================================
# 2. Select estimation window
#    Valid windows: :base_fc, :crisis_fc, :base_covid, :crisis_covid
# ============================================================
WINDOW = :base_fc

# ============================================================
# Moments to exclude from the SMM objective.
# Use the SAME list for both load_weight_matrix and build_smm_spec
# so that sigma/W is subsetted consistently with the loss function.
#
# Valid names:
#   :ur_U, :ur_S, :skilled_share, :training_share,
#   :emp_var_U, :emp_cm3_U, :emp_var_S, :emp_cm3_S,
#   :jfr_U, :sep_rate_U, :jfr_S, :sep_rate_S, :ee_rate_S,
#   :mean_wage_U, :mean_wage_S,
#   :p25_wage_U, :p25_wage_S, :p50_wage_U, :p50_wage_S,
#   :wage_premium, :theta_U, :theta_S
# ============================================================
const SKIP_MOMENTS = Symbol[
    #:emp_var_U,
    :emp_var_S,
    :emp_cm3_U,
    #:emp_cm3_S,
    :ee_rate_S,
    :p25_wage_U,
    :p25_wage_S,
    :p50_wage_U,
    :p50_wage_S,
    :mean_wage_U,
    :mean_wage_S,
]

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
#    W_COND_TARGET controls which weighting scheme is used:
#      0.0   →  diagonal from sigma_{window}.csv
#      1.0   →  compressed diagonal: w = log(1 + diag)
#      2.0   →  equal weights (identity, no W matrix)
#      >2.0  →  full optimal W (shrunk if κ > target)
# ============================================================
const W_COND_TARGET = 1e6  # also set in run_params below; keep in sync

"""
    _w_suffix(cond_target) → String

Return the filename suffix that identifies the weight-matrix mode.
  0.0   → "_diagonalW"
  1.0   → "_compressedW"
  2.0   → "_equalW"
  else  → "_fullW"
"""
function _w_suffix(cond_target::Float64)
    cond_target == 0.0 && return "_diagonalW"
    cond_target == 1.0 && return "_compressedW"
    cond_target == 2.0 && return "_equalW"
    return "_fullW"
end

const W_SUFFIX = _w_suffix(W_COND_TARGET)

W_opt = load_weight_matrix(; window=WINDOW, derived_dir=derived_dir,
                             cond_target=W_COND_TARGET,
                             skip_moments=SKIP_MOMENTS)

# ============================================================
# 5. Externally calibrated parameters (r, ν, φ)
#    Always fixed across all 4 estimation windows.
# ============================================================
println("\nLoading externally calibrated parameters (r, ν, φ)...")
flush(stdout)
calib = load_calibrated_params(; derived_dir=derived_dir)

@printf("  Calibrated:  r = %.6f,  ν = %.5f,  φ = %.5f\n",
        calib.r, calib.nu, calib.phi)
flush(stdout)

# ============================================================
# 6. Fixed and free parameters
#    - Baseline windows (:base_fc, :base_covid):
#        fix only (r, ν, φ); estimate all structural + regime params
#    - Crisis windows (:crisis_fc, :crisis_covid):
#        load Stage 1 baseline (base_fc) results;
#        fix deep structural params at baseline values;
#        re-estimate only regime-specific params
#
#    Parameter classification (from Model Notes, §2.3–2.4):
#
#    Deep structural (constant across regimes):
#      common:  a_ℓ, b_ℓ, c
#      regime:  bU, bT, bS          (institutional / policy)
#      unsk:    μ, η, β             (technology / institutions)
#      skl:     μ, η, β, σ          (technology / institutions)
#
#    Regime-specific (re-estimated per crisis window):
#      regime:  PU, PS, α_U, a_Γ, b_Γ
#      unsk:    k, λ
#      skl:     k, λ, ξ
# ============================================================

# Set of (block, name) pairs that are regime-specific
const REGIME_SPECIFIC_PARAMS = Set([
    (:regime, :PU), (:regime, :PS), (:regime, :α_U), (:regime, :a_Γ), (:regime, :b_Γ),
    (:unsk, :k), (:unsk, :λ),
    (:skl, :k), (:skl, :λ), (:skl, :ξ),
])

"""
    _baseline_param_value(ps::ParamSpec, cp, rp, up, sp) → Float64

Extract the value of parameter `ps` from the four solved structs.
"""
function _baseline_param_value(ps::ParamSpec, cp, rp, up, sp) :: Float64
    if     ps.block == :common; return Float64(getfield(cp, ps.name))
    elseif ps.block == :regime; return Float64(getfield(rp, ps.name))
    elseif ps.block == :unsk;   return Float64(getfield(up, ps.name))
    else                        return Float64(getfield(sp, ps.name))
    end
end

if WINDOW in (:crisis_fc, :crisis_covid)
    # ── Stage 2: crisis re-estimation ──────────────────────────────────
    # Load baseline result (always from base_fc with matching W suffix)
    baseline_jls = joinpath(SMM_OUT_DIR, "smm_result_base_fc$(W_SUFFIX).jls")
    @printf("\nCrisis window detected — loading baseline from:\n  %s\n", baseline_jls)
    flush(stdout)
    isfile(baseline_jls) || error(
        "Baseline result not found at $baseline_jls. " *
        "Run the base_fc estimation first (WINDOW = :base_fc, W_COND_TARGET = $W_COND_TARGET).")

    baseline_data = _load_smm_bundle(baseline_jls; delete_on_fail=false, label="baseline file")
    isnothing(baseline_data) && error(
        "Baseline result at $baseline_jls is unreadable or stale. " *
        "Re-run the base_fc estimation first (WINDOW = :base_fc, W_COND_TARGET = $W_COND_TARGET)."
    )

    baseline_result  = baseline_data.result   # SMMResult
    baseline_spec    = baseline_data.spec     # SMMSpec

    # Reconstruct the four parameter structs from the baseline optimum
    cp_base, rp_base, up_base, sp_base =
        unpack_θ(baseline_result.theta_opt, baseline_spec)

    @printf("  Baseline Q = %.6e  (converged = %s)\n",
            baseline_result.loss_opt, baseline_result.converged)
    flush(stdout)

    # ── Build fixed_params: calibrated + deep structural ───────────────
    # For shared names (μ, η, β) use block-qualified keys so that
    # unskilled and skilled blocks receive their own baseline values.
    fixed_params = (
        # Externally calibrated
        r     = calib.r,
        ν     = calib.nu,
        φ     = calib.phi,
        # Deep structural — common block
        a_ℓ   = cp_base.a_ℓ,
        b_ℓ   = cp_base.b_ℓ,
        c     = cp_base.c,
        # Deep structural — regime block (institutional)
        bU    = rp_base.bU,
        bT    = rp_base.bT,
        bS    = rp_base.bS,
        # Deep structural — unskilled (block-qualified for shared names)
        unsk_μ = up_base.μ,
        unsk_η = up_base.η,
        unsk_β = up_base.β,
        # Deep structural — skilled (block-qualified for shared names)
        skl_μ  = sp_base.μ,
        skl_η  = sp_base.η,
        skl_β  = sp_base.β,
        skl_σ  = sp_base.σ,
    )

    # ── Build free_params: regime-specific only, init from baseline ─────
    free_params = ParamSpec[]
    for ps in default_free_params()
        (ps.block, ps.name) in REGIME_SPECIFIC_PARAMS || continue
        init_val = _baseline_param_value(ps, cp_base, rp_base, up_base, sp_base)
        push!(free_params,
              ParamSpec(ps.block, ps.name, ps.lb, ps.ub, init_val, ps.label))
    end

    println("\n  Deep structural parameters FIXED from baseline:")
    @printf("    common:  a_ℓ=%.4f  b_ℓ=%.4f  c=%.4f\n",
            cp_base.a_ℓ, cp_base.b_ℓ, cp_base.c)
    @printf("    regime:  bU=%.4f  bT=%.4f  bS=%.4f\n",
            rp_base.bU, rp_base.bT, rp_base.bS)
    @printf("    unsk:    μ=%.4f  η=%.4f  β=%.4f\n",
            up_base.μ, up_base.η, up_base.β)
    @printf("    skl:     μ=%.4f  η=%.4f  β=%.4f  σ=%.4f\n",
            sp_base.μ, sp_base.η, sp_base.β, sp_base.σ)
    @printf("  Regime-specific parameters FREE (%d params)\n", length(free_params))
    flush(stdout)

else
    # ── Stage 1: baseline estimation (base_fc or base_covid) ───────────
    # Fix only the externally calibrated params; estimate everything else.
    fixed_params = (
        r   = calib.r,
        ν   = calib.nu,
        φ   = calib.phi,
    )

    free_params = default_free_params()

    # ── Warm-start: if a prior base_fc result exists, use its
    #    optimum as the initial values for this run.  This helps
    #    when re-estimating with a different weight matrix. ──────
    _warmstart_jls = joinpath(SMM_OUT_DIR, "smm_result_base_fc$(W_SUFFIX).jls")
    if isfile(_warmstart_jls)
        println("\n  Warm-start: loading prior base_fc result from:")
        @printf("    %s\n", _warmstart_jls)
        flush(stdout)

        _ws_data = _load_smm_bundle(_warmstart_jls; delete_on_fail=true, label="warm-start file")

        if !isnothing(_ws_data)
            _ws_result = _ws_data.result
            _ws_spec   = _ws_data.spec
            _ws_cp, _ws_rp, _ws_up, _ws_sp = unpack_θ(_ws_result.theta_opt, _ws_spec)

            @printf("    Prior Q = %.6e  (converged = %s)\n",
                    _ws_result.loss_opt, _ws_result.converged)
            flush(stdout)

            # Override init values in free_params with the prior optimum
            free_params = [
                let init_val = _baseline_param_value(ps, _ws_cp, _ws_rp, _ws_up, _ws_sp)
                    init_val = clamp(init_val, ps.lb + 1e-10, ps.ub - 1e-10)
                    ParamSpec(ps.block, ps.name, ps.lb, ps.ub, init_val, ps.label)
                end
                for ps in free_params
            ]

            println("    Initial values set from prior optimum.")
        else
            println("    Warm-start skipped — stale/unreadable file was removed.")
        end
    else
        println("\n  No prior base_fc result found — using default initial values.")
    end
    flush(stdout)

    @printf("\n  Baseline mode: all structural + regime params are FREE\n")
    flush(stdout)
end

@printf("  Fixed params:  %d  |  Free params:  %d\n",
        length(fixed_params), length(free_params))
flush(stdout)

# ============================================================
# 7. Run parameters — grids, SA, Nelder-Mead, tracing
# ============================================================
run_params = SMMRunParams(
    # ── Grids (coarser = faster per iteration) ──────────────
    Nx      = 80,
    Np_U    = 80,
    Np_S    = 80,

    # ── Weight matrix conditioning ────────────────────────────
    w_cond_target = W_COND_TARGET,

    # ── SA global search ────────────────────────────────────
    sa_max_iter        = 25_000,  # total SA proposals
    sa_T0              = 5.00,     # initial temperature (higher = more uphill acceptance early). 0.0 auto.
    sa_step            = 0.20,    # initial random-walk step in logit space
    sa_cooling_rate    = 1.0,     # scales t in cooling schedule denominator
    sa_cooling_exp     = 1.0,     # exponent: T0/log(1+rate*t)^exp  (<1 = slower cooling)
    sa_reheat_patience = 800,         # proposals without improvement before reheating
    sa_reheat_factor   = 1.50,     # temperature multiplier on reheat
    sa_max_reheats     = 1,       # cap on total reheats (0 = unlimited)
    sa_adapt_window    = 50,      # rolling window for adaptive step (0 = off)
    sa_target_fin      = 0.90,    # target feasibility rate for adaptive step
    sa_random_init     = false ,   # whether to randomize initial solution for SA (instead of using free_params.init)

    # ── DE global search ────────────────────────────────────
    de_max_iter  = 3_000,       # generations; total evals = max_iter × pop_size
    de_pop_size  = 100,       # 0 = auto (100 × n_free_params)
    de_f         = 0.70,        #factor for mutation (0.5-0.9 typical)
    de_cr        = 0.85,        #crossover probability (0-1)
    de_patience  = 5,           # how many generations to wait for improvement before early stopping
    de_avg_tol   = 0.01,    # stop when (Q_mean − Q_best) / |Q_best| < this (1 %); set 0.0 to disable


    # ── Nelder-Mead polish ───────────────────────────────────
    nm_max_iter  = 500,        # maximum iterations for Nelder-Mead local search
    nm_f_tol     = 1e-6,        # stop when |Q_new − Q_old| < this; set 0.0 to disable
    nm_x_tol     = 1e-4,        # stop when max|θ_new − θ_old| < this; set 0.0 to disable

    # ── Tracing ─────────────────────────────────────────────
    show_trace_members     = false,   # per-member lines within each generation for DE, prints for stride proposal in SA
    show_trace_generations = true,    # end-of-generation summary lines
    trace_stride           = 500,      # how often to print within DE generations (in members, not generations), for SA how often to print
)

# ============================================================
# 8. Build SMM spec
# ============================================================
spec = build_smm_spec(
    moments, sim_smm;
    fixed        = fixed_params,
    free_specs   = free_params,
    run          = run_params,
    W            = W_opt,
    skip_moments = SKIP_MOMENTS,
)

print_spec(spec)

# ============================================================
# 9. Run estimation
# ============================================================
println("Starting SMM optimisation..."); flush(stdout)

# Stage 1: global search :sa or :de 
res = run_smm(spec; method = :sa)

# Stage 2: polish from global optimizer solution
res_pol = run_smm(_spec_with_init(spec, res.theta_opt); method = :neldermead)

results = res_pol

# ============================================================
# 10. Save results
# ============================================================
mkpath(TABLES_DIR)
mkpath(SMM_OUT_DIR)

# CSV table of parameter estimates
save_results(results, joinpath(TABLES_DIR, "smm_estimates_$(WINDOW)$(W_SUFFIX).csv"))

# Serialize full result for the transition solver.
# The .jls file stores: (result, spec, sim_smm) so that
# transition_main.jl can reconstruct the solved model.
smm_jls_path = joinpath(SMM_OUT_DIR, "smm_result_$(WINDOW)$(W_SUFFIX).jls")
open(smm_jls_path, "w") do io
    serialize(io, (result = results, spec = spec, sim = sim_smm))
end
@printf("Serialized SMM result → %s\n", smm_jls_path)

println("\nDone.")
flush(stdout)

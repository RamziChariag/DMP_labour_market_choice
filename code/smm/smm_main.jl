############################################################
# smm/smm_main.jl — SMM estimation entry point
#
# Usage (repo root):
#   ROYSEARCH_WINDOW=<window> julia --project=. --threads auto code/smm/smm_main.jl
#
# Windows (WINDOWS, paths.jl):
#   :baseline      one economy in two periods, base_fc (2003-07) and base_covid
#                  (2015-19), estimated jointly: Q is the sum of the periods' Q, each on
#                  its own moments, weights, ν and σ_w; BASELINE_COMMON is one parameter
#                  for both periods.
#   :crisis_fc,    hold CRISIS_HELD at the baseline estimate (SMM, or the chain's with
#   :crisis_covid  CRISIS_BASE = :mcmc) at the window's period (PAIR_BASE) and estimate
#                  the rest; ν comes from the baseline period.
#
# Inputs: data/derived/{windows.json, moments_<w>.csv, sampling_var_<w>.csv,
# nu_estimation.csv, phi_calibration.csv}. Output: output/estimates/
# estimate_<window>_diagonalW_<point>.jls and its smm_estimates_*.csv, written only here.
# Settings are ROYSEARCH_* environment variables (settings.jl, SETTINGS.md).
############################################################

include(joinpath(@__DIR__, "version.jl"))
println("="^60)
println("  RoySearch v$(ROYSEARCH_VERSION) — SMM Estimation")
println("="^60)
flush(stdout)

const SMM_DIR      = @__DIR__
const SOLVER_DIR   = joinpath(SMM_DIR, "..", "solver")
include(joinpath(SMM_DIR, "..", "paths.jl"))

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
using QuasiMonteCarlo
using JSON3

println("done."); flush(stdout)

Random.seed!(1234)

# BLAS calls run inside threaded regions: one BLAS thread, Base.Threads owns the cores.
LinearAlgebra.BLAS.set_num_threads(1)

print("Loading solver modules... "); flush(stdout)

include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))

println("done."); flush(stdout)

print("Loading SMM modules... "); flush(stdout)

include(joinpath(SMM_DIR, "settings.jl"))
include(joinpath(SMM_DIR, "moments.jl"))
include(joinpath(SMM_DIR, "smm_params.jl"))
include(joinpath(SMM_DIR, "bundle.jl"))
include(joinpath(SMM_DIR, "smm.jl"))
include(joinpath(SMM_DIR, "candidates.jl"))

println("done."); flush(stdout)
@printf("Threads available: %d\n\n", Threads.nthreads())
flush(stdout)

# ============================================================
# 1. Solver settings for the estimation
# ============================================================
sim_smm = SimParams(
    tol_inner          = 1e-7,
    tol_outer_U        = 1e-6,
    tol_outer_S        = 1e-6,

    damp_inner_U       = 0.95,
    damp_inner_S       = 0.95,

    inner_B            = 20,     # inner divergence early-abort burn-in (0 disables)
    inner_K            = 10,     # reject after inner_K divergent outer iterations

    outer_B            = 30,     # outer stall-detect burn-in (0 disables the handback)
    outer_K            = 10,     # outer no-contraction window; hand back to global on stall

    global_B           = 8,      # global stall-detect burn-in (0 disables)
    global_K           = 4,      # global no-improvement window; stops a U↔S limit cycle

    maxit_inner        = 300,
    maxit_outer        = 200,
    maxit_global       = 20,

    conv_streak        = 4,

    use_anderson       = true,
    anderson_reg       = 1e-10,

    damp_pstar_U       = 1.00,
    damp_pstar_S       = 0.50,

    verbose            = 0,      # 0 silent; 1 outer convergence per iteration; 2 inner detail
    verbose_stride     = 100,
)

# ============================================================
# 2. Estimation window and objective
# ============================================================
assert_no_legacy_env()

WINDOW = env_setting(:WINDOW, :baseline)
String(WINDOW) in WINDOWS ||
    error("WINDOW = :$WINDOW is not an estimation window; must be one of :" * join(WINDOWS, ", :") * ".")
# Crisis windows only: hold CRISIS_HELD at the baseline SMM estimate (:smm, writes
# estimate_<crisis>_smm.jls) or at the baseline chain's estimate (:mcmc, writes
# estimate_<crisis>_smm_mcmcbase.jls, the crisis chain's seed).
CRISIS_BASE = env_setting(:CRISIS_BASE, :mcmc)
CRISIS_BASE in (:smm, :mcmc) || error("CRISIS_BASE must be :smm or :mcmc, got :$CRISIS_BASE.")

const CRISIS = is_crisis(WINDOW)

# The estimate this run writes (ESTIMATE_POINTS, paths.jl).
const OUT_POINT = CRISIS && CRISIS_BASE === :mcmc ? :smm_mcmcbase : :smm

# Moments given zero weight in the objective, in every data window; the same list
# subsets W and the spec. Valid names: MOMENT_NAMES (moments.jl). A moment whose target
# is NaN or absent in a data window is held out there automatically.
#   ur_total, wage_premium  exact linear redundancies of the active set
#   ee_step_S               carried through the pipeline with zero weight
#   ltu_share_S             equals usurv27_S, which is active
#   usurv*_U                a CPS unskilled spell ends on any exit, NILF included, while
#                           model unemployed leave only by hiring or at ν; the data force a
#                           withdrawal hazard the model lacks (the skilled data do not).
#                           Reinstate once the model has a withdrawal margin.
SKIP_MOMENTS = Symbol[
    :ur_total,
    :ee_step_S,
    :wage_premium,
    :ltu_share_S,
    :usurv5_U, :usurv14_U, :usurv27_U, :usurv53_U,
]

# Data window => moments given zero weight in that window only (finite targets).
const WINDOW_SKIP_MOMENTS = Dict{Symbol,Vector{Symbol}}()

# Wage-reliability ratio; σ_w is calibrated from it (calibrate_sigma_w), not estimated.
# 0.78 is the middle of the range the wage-validation literature reports.
const LAMBDA_W = env_setting(:LAMBDA_W, 0.78)

@printf("Estimation window: %s\n", WINDOW)
flush(stdout)

# Parameters pinned during estimation, ASCII keys (_ASCII_TO_FIXED_KEY, smm_params.jl).
# r, ν, φ and σ_wU, σ_wS are calibrated and pinned separately. The vacancy costs follow
# from the tightnesses and cannot be pinned; pin :unsk_tight / :skl_tight instead.
FIX_PARAMS = Dict{Symbol,Float64}(
     :unsk_eta => 0.50000,
     :unsk_bet => 0.50000,
     :skl_eta  => 0.50000,
     :skl_bet  => 0.50000,
     :bT       => 0.00000,
     #:bS       => 0.00000,
     :PU       => 1.00000,
     #:alpha_U  => 1.00000,
     :a_Gam    => 1.00000,
     :b_Gam    => 1.00000,
     :unsk_xi  => 0.00000,
     :skl_xi  => 0.00000,
     :skl_delta => 1.00000,
     :a_l       =>1.000000,
     :b_l       =>1.000000,
)

# How the optimiser is seeded:
#   :default         the DEFAULT values; no candidate layer
#   :warmstart       the stored point with the lowest Q under the current code (§8a); the
#                    DEFAULT values if none has a finite Q
#   :warmstart_base  a crisis window starts from the baseline point at its period; the
#                    baseline runs as under :warmstart
#   :clusters        the Sobol → hclust candidate bank (§8b)
# The settings of each strategy (USE_AS_SEED, SEED_PERTURB_FRAC, CLUSTERS_FORCE_REGEN,
# INCLUDE_PREV_OPTIMUM) are SMMRunParams fields, so a bundle records how its run started.
INIT_MODE = env_setting(:INIT_MODE, :warmstart)

# Starting values of the free parameters, ASCII keys (_DEFAULT_PARAM_KEY), reported units:
# A = ln A, c = ln c, b_j and σ_S over A, θ_j vacancies per searcher. Pinned and calibrated
# entries are ignored. DEFAULT_PERIOD_PARAMS holds the values that differ by baseline
# period; a crisis window starts from its baseline period's values. Built from the v60
# chain estimates of the two former base windows, each common parameter at their mean.
const DEFAULT_PARAMS = Dict{Symbol,Float64}(
    :r         => 0.00416667,
    :nu        => 0.00323032,
    :phi       => 0.02222129,

    :a_l       => 1.00000000,   # pinned (FIX_PARAMS)
    :b_l       => 1.00000000,   # pinned (FIX_PARAMS)
    :rho_x     => -0.43246076780329124,

    :PU        => 1.00000000,   # the numeraire

    :bU        => 0.07069092317348318,
    :bT        => 0.00000000,   # pinned: no income while training
    :bS        => 0.177733730234988,

    :a_Gam     => 1.00000000,   # pinned (FIX_PARAMS)
    :b_Gam     => 1.00000000,   # pinned (FIX_PARAMS)
    :skl_delta => 1.00000000,   # pinned (FIX_PARAMS)

    :unsk_eta  => 0.50000000,   # pinned (FIX_PARAMS)
    :unsk_bet  => 0.50000000,   # pinned (FIX_PARAMS)
    :unsk_xi   => 0.00000000,   # pinned (FIX_PARAMS)
    :unsk_sigw => 0.25549225,   # set from λ_w (calibrated per window)

    :skl_eta   => 0.50000000,   # pinned (FIX_PARAMS)
    :skl_bet   => 0.50000000,   # pinned (FIX_PARAMS)
    :skl_sig   => 0.27215781097836383,
    :skl_sigw  => 0.23267843,   # set from λ_w (calibrated per window)
    :skl_xi    => 0.00000000,   # pinned (FIX_PARAMS)
)

const DEFAULT_PERIOD_PARAMS = Dict{Symbol,Dict{Symbol,Float64}}(
    :base_fc => Dict(
        :c          => 11.471265778520698,
        :A          => 7.296791792868002,
        :PS         => 1.7342881760826074,
        :alpha_U    => 3.250407358831487,
        :unsk_mu    => 0.5956864020132866,
        :skl_mu     => 0.28396663842815656,
        :unsk_lam   => 0.17146440372583185,
        :skl_lam    => 0.1542692272823348,
        :unsk_tight => 0.44864608267488587,
        :skl_tight  => 1.1627529893606718,
    ),
    :base_covid => Dict(
        :c          => 10.793347734898436,
        :A          => 7.44296706781371,
        :PS         => 1.6506777497201668,
        :alpha_U    => 1.7898221443599454,
        :unsk_mu    => 0.34552157586435334,
        :skl_mu     => 0.22457734120368542,
        :unsk_lam   => 0.22435828360971466,
        :skl_lam    => 0.2277588335078418,
        :unsk_tight => 0.7779525694683238,
        :skl_tight  => 1.805317325622991,
    ),
)

"""
    default_seed(ps, period) → Float64

The starting value of `ps` in baseline period `period`: `DEFAULT_PERIOD_PARAMS[period]` if
it lists the parameter's ASCII key, `DEFAULT_PARAMS` otherwise. Raises when neither does.
"""
function default_seed(ps::ParamSpec, period::Symbol) :: Float64
    key = get(_DEFAULT_PARAM_KEY, (ps.block, ps.name), nothing)
    isnothing(key) && error(
        "default_seed: $(ps.block):$(ps.name) has no ASCII key in _DEFAULT_PARAM_KEY (smm_params.jl).")
    own = DEFAULT_PERIOD_PARAMS[period]
    haskey(own, key) && return own[key]
    haskey(DEFAULT_PARAMS, key) || error(
        "default_seed: neither DEFAULT_PERIOD_PARAMS[:$period] nor DEFAULT_PARAMS has a :$(key) " *
        "entry, the seed for $(ps.block):$(ps.name).")
    return DEFAULT_PARAMS[key]
end

# ============================================================
# 3. Data windows and targets
#    The baseline is estimated on both of its periods, a crisis window on its own; each
#    data window has its own targets, holdouts, W, calibrated rates and σ_w.
# ============================================================
derived_dir = joinpath(PROJECT_ROOT, "data", "derived")
_win_info   = load_windows(; derived_dir=derived_dir)

# λ_w > 0 pins σ_w at its calibrated value with β free; λ_w = 0 frees σ_w, and β must then
# be pinned, since wage dispersion identifies only one of the two.
if LAMBDA_W == 0.0
    @printf("  λ_w = 0 → σ_w is ESTIMATED (free), not pinned.\n")
    if !(haskey(FIX_PARAMS, :unsk_bet) && haskey(FIX_PARAMS, :skl_bet))
        @warn "λ_w = 0 frees σ_w but β is not fully pinned in FIX_PARAMS " *
              "(:unsk_bet / :skl_bet) — β and σ_w are jointly under-identified."
    end
end

"""
    data_window_setup(dw) → (moments, skip, W, calib, sigw)

Targets and calibration of data window `dw`, each printed once: its moments, the moments
held out of its objective (SKIP_MOMENTS, the NaN or absent targets, WINDOW_SKIP_MOMENTS),
its weight matrix, the calibrated r, ν, φ (`calib`) and the σ_w calibrated from λ_w
(`sigw`, empty when λ_w = 0).
"""
function data_window_setup(dw::Symbol)
    haskey(_win_info.windows, dw) || error(
        "$dw not found in windows.json. Data windows: " * join(string.(_win_info.order), ", "))
    wd = _win_info.windows[dw]
    @printf("\n%s (windows.json):  %s  ym = %d..%d  ASEC %d..%d\n", dw, wd.label,
            wd.ym_start, wd.ym_end, first(wd.asec_years), last(wd.asec_years))
    moments = load_data_moments(; window=dw, derived_dir=derived_dir)

    # A NaN or absent target cannot enter g'Wg: held out of the objective and W alike, and
    # active again once its data appear.
    nan_targets = Symbol[k for k in MOMENT_NAMES
                         if !(k in keys(moments)) || !isfinite(moments[k].value)]
    window_skip = get(WINDOW_SKIP_MOMENTS, dw, Symbol[])
    skip        = sort(unique(vcat(SKIP_MOMENTS, nan_targets, window_skip)))
    isempty(nan_targets) ||
        @printf("  Auto-skipping %d moment(s) with a missing or NaN target: %s\n",
                length(nan_targets), join(string.(nan_targets), ", "))
    isempty(window_skip) ||
        @printf("  Per-window holdout: %d moment(s) excluded from the objective: %s\n",
                length(window_skip), join(string.(window_skip), ", "))

    σ_wU, σ_wS = calibrate_sigma_w(LAMBDA_W, moments)
    @printf("  Calibrated σ_w (λ_w = %.4f):  σ_wU = %.5f,  σ_wS = %.5f\n", LAMBDA_W, σ_wU, σ_wS)
    W = load_weight_matrix(; window=dw, derived_dir=derived_dir, weighting=WEIGHTING,
                             skip_moments=skip)
    calib = load_calibrated_params(; window=dw, derived_dir=derived_dir)
    @printf("  Calibrated:  r = %.6f,  ν = %.5f,  φ = %.5f\n", calib.r, calib.nu, calib.phi)
    flush(stdout)
    return (moments = moments, skip = skip, W = W,
            calib = (r = calib.r, ν = calib.nu, φ = calib.phi),
            sigw  = LAMBDA_W > 0.0 ? (unsk_σ_w = σ_wU, skl_σ_w = σ_wS) : NamedTuple())
end

# ============================================================
# 4. Weighting and output file
# ============================================================
WEIGHTING = run_weighting()
W_SUFFIX  = weighting_suffix(WEIGHTING)
# This run's bundle: its reheat checkpoints, its final estimate, and what a later
# :warmstart or INCLUDE_PREV_OPTIMUM of the same configuration reads.
const OUT_JLS = estimate_path(WINDOW, W_SUFFIX; point = OUT_POINT)

# Display-only divisor of Q; 1 reports the raw objective.
Q_SCALE = 1.0

# ============================================================
# 5. FIX_PARAMS as a fixed NamedTuple
# ============================================================
_extra_fixed = _fix_params_to_nt(FIX_PARAMS)

# ============================================================
# 6. Fixed and free parameters, per data window
#    Merge order (later wins): calibrated r, ν, φ; the baseline values a crisis window
#    holds; FIX_PARAMS; the calibrated σ_w. `fixed_key` writes the keys `unpack_θ` reads.
# ============================================================

if CRISIS
    # The baseline estimate must be current under this code, and a chain's must not have
    # aborted.
    base_period  = PAIR_BASE[WINDOW]
    baseline_jls = estimate_path(:baseline, W_SUFFIX; point = CRISIS_BASE)
    @printf("\nCrisis window — loading the baseline estimate, period %s, from:\n  %s\n",
            base_period, baseline_jls)
    flush(stdout)
    isfile(baseline_jls) || error(
        "Baseline estimate not found at $baseline_jls. Run " *
        (CRISIS_BASE === :smm ? "smm_main.jl" : "a chain of MCMC_main.jl") *
        " on the baseline first (WEIGHTING = $WEIGHTING).")
    _baseline_raw = _load_smm_bundle(baseline_jls; delete_on_fail=false, label="baseline file")
    isnothing(_baseline_raw) && error(
        "Baseline result at $baseline_jls is unreadable. Re-run the baseline estimation first.")
    require_current_estimate(_baseline_raw, baseline_jls)
    get(_baseline_raw, :stage, :smm) === :mcmc_aborted && error(
        "$(basename(baseline_jls)): the baseline chain aborted, so its draws are not a " *
        "stationary sample and the file holds no estimate.")
    @printf("  Baseline Q = %.6e  (converged = %s)\n",
            _baseline_raw.result.loss_opt, _baseline_raw.result.converged)

    baseline_data = period_bundle(_baseline_raw, base_period)
    cp_base, up_base, sp_base, tight_base =
        unpack_θ(baseline_data.result.theta_opt, baseline_data.spec)
    _base_fixed = NamedTuple(
        fixed_key(blk, nm) => _baseline_param_value(blk, nm, cp_base, up_base, sp_base, tight_base)
        for (blk, nm) in CRISIS_HELD)

    _d = data_window_setup(WINDOW)
    # Free: every parameter but the held ones, started from the baseline period's DEFAULT
    # values (§8a may replace the start).
    PERIOD_INPUTS = [(window = WINDOW, data = _d,
                      fixed  = merge(_d.calib, _base_fixed, _extra_fixed, _d.sigw),
                      free   = [with_seed(ps, default_seed(ps, base_period))
                                for ps in default_free_params()
                                if !((ps.block, ps.name) in CRISIS_HELD)])]
else
    # The baseline: every parameter is a candidate for each period's free set;
    # build_smm_spec drops what is pinned and joint_spec merges BASELINE_COMMON.
    baseline_data = nothing
    PERIOD_INPUTS = [(window = dw, data = _d, fixed = merge(_d.calib, _extra_fixed, _d.sigw),
                      free = [with_seed(ps, default_seed(ps, dw)) for ps in default_free_params()])
                     for (dw, _d) in ((dw, data_window_setup(dw)) for dw in BASELINE_PERIODS)]
end

# ============================================================
# 7. Run parameters
#
# The SA pair (SA_T0_REL = 1e-4, SA_STEP = 0.01) is set for warm starts. From a cold start
# (:default, a fresh cluster bank) set ROYSEARCH_SA_T0_REL=0.05 ROYSEARCH_SA_STEP=0.20: at
# 1e-4 a cold start accepted 0 of 13 proposals, and a 0.20 step from a warm start accepted
# 0 of 150 (median ΔQ=1 half-width 3.4e-3).
# ============================================================

run_params = SMMRunParams(
    # Grids: worker types (Gauss-Legendre nodes), unskilled and skilled match quality
    Nx      = 160,
    Np_U    = 160,
    Np_S    = 180,

    # Candidate layer (INIT_MODE = :clusters)
    cand_Nx          = 120,
    cand_Np_U        = 120,
    cand_Np_S        = 160,
    cand_n_sample    = 2048,
    cand_seed        = 42,
    cand_min_cluster = 5,      # min members of a cluster

    weighting = WEIGHTING,

    λ_w = LAMBDA_W,

    # Simulated annealing
    sa_max_iter        = env_setting(:SA_MAX_ITER, 5_000),
    sa_T0              = 0.0,     # ≤ 0 ⇒ T0 solves exp(−sa_t0_rel·Q0 / T0) = sa_t0_accept
    sa_step            = env_setting(:SA_STEP, 0.01),
                                  # fallback step for a coordinate whose width cannot be
                                  # measured; the proposal step is per coordinate
    sa_t0_rel          = env_setting(:SA_T0_REL, 1e-4),
    sa_cooling_rate    = 1.5,     # logarithmic schedule, used only when
    sa_cooling_exp     = 2.0,     # sa_halflife = 0
    sa_reheat_patience = 400,     # steps without improvement before a reheat; 0 disables
    sa_reheat_factor   = 4.00,    # reheat: T ← T · factor
    sa_max_reheats     = env_setting(:SA_MAX_REHEATS, 1),  # 0 = unlimited
    sa_adapt_window    = 50,      # window of the traced acceptance and feasibility
    sa_random_init     = false,   # true ⇒ start from a uniform draw in the box
    sa_parallel_steps  = 100,     # :clusters — warm-up steps per start before pruning
    sa_seed            = 42,

    # Differential evolution. Equal-budget sweep at the base_fc optimum (ΔQ achieved):
    # f 0.5/0.7/0.9 → 25.2/88.3/33.6; cr 0.5/0.9 → 77.2/88.3; pop 48/96 → 88.3/106.0.
    # The population size 10·n_free is extrapolated from that direction, not measured.
    de_max_iter  = env_setting(:DE_MAX_ITER, 3_000),
    de_pop_size  = env_setting(:DE_POP_SIZE, 240),
    de_f         = env_setting(:DE_F, 0.70),
    de_cr        = env_setting(:DE_CR, 0.90),
    de_avg_tol   = env_setting(:DE_AVG_TOL, 0.0),
                                  # off: it ranks a prematurely converged run as the most
                                  # converged (ρ = +0.915 with achieved ΔQ)

    # DE population generator and reheat (generate_population)
    de_gen_per_k   = env_setting(:DE_GEN_PER_K, 60),
    de_adapt_fcr   = env_setting(:DE_ADAPT_FCR, true),
                                  # true: f and cr read off the generator's yield table at
                                  # every reheat; false: de_f and de_cr as set
    de_reheat_flat = env_setting(:DE_REHEAT_FLAT, 6),
    de_reheat_rate = env_setting(:DE_REHEAT_RATE, 0.04),
                                  # both must hold: Q_best alone is flat for up to 12
                                  # generations mid-descent
    de_max_reheats = env_setting(:DE_MAX_REHEATS, 20),

    # Nelder-Mead / gradient polish tolerances. Optim's Nelder-Mead reads neither f_tol nor
    # x_tol, and its g_tol tests sd(Q) over the simplex in Q units (unreachable at Q ~ 1e3;
    # left at Optim's default, and 0 does not disable it): the rate rule stops it.
    nm_max_iter  = env_setting(:NM_MAX_ITER, 10_000),
    nm_f_tol     = env_setting(:NM_F_TOL, 1e-3),   # ΔQ = 1 relative to Q ~ 1e3
    nm_x_tol     = env_setting(:NM_X_TOL, 1e-5),   # absolute in t; backstop only
    nm_g_tol     = env_setting(:NM_G_TOL, 1e-8),
    nm_no_improve = env_setting(:NM_NO_IMPROVE, 4_000),  # evaluations without improvement; 0 disables

    # Start-point construction
    use_as_seed          = env_setting(:USE_AS_SEED, true),
    seed_perturb_frac    = env_setting(:SEED_PERTURB_FRAC, 0.05),
    clusters_force_regen = env_setting(:CLUSTERS_FORCE_REGEN, true),
    include_prev_optimum = env_setting(:INCLUDE_PREV_OPTIMUM, false),

    # Rate-based stops
    sa_rate_tol     = env_setting(:SA_RATE_TOL, 0.05),
    sa_rate_span    = env_setting(:SA_RATE_SPAN, 300),
    # Half the iteration budget: a half-life equal to the budget ends the walk hot, well
    # above its own best; at half it ends inside one plateau cell of its best.
    sa_halflife     = env_setting(:SA_HALFLIFE, 2_500),
    nm_rate_tol     = env_setting(:NM_RATE_TOL, 0.05),
    nm_rate_span    = env_setting(:NM_RATE_SPAN, 300),
    nm_simplex_step = env_setting(:NM_SIMPLEX_STEP, 0.2),

    de_local_sigma = env_setting(:DE_LOCAL_SIGMA, 0.33),

    # Tracing
    show_trace_members     = false,   # per-member trace inside one DE/SA generation
    show_trace_generations = true,    # per-generation / per-iteration summary
    trace_stride           = 100,     # print every N SA iterations
)

# SA proposal measurement (sa_proposal_scale, smm.jl). Passed to run_smm as keyword
# arguments, not SMMRunParams fields (bundles are read by field position).
#   SA_SCALE_P_MOVE  mask density in units of 1/d: one forced coordinate plus a
#                    Binomial(d−1, p/d) tail. Chosen on equal-budget chains over the
#                    one-shot sparsity scan.
#   SA_SCALE_PER_K   draws per sparsity of the diagnostic scan; 0 disables it.
#   SA_SCALE_SIGMA   step as a fraction of each coordinate's measured ΔQ<1 width.
const SA_SCALE_P_MOVE = env_setting(:SA_SCALE_P_MOVE, 1.0)
const SA_SCALE_PER_K  = env_setting(:SA_SCALE_PER_K, 0)
const SA_SCALE_SIGMA  = env_setting(:SA_SCALE_SIGMA, 0.33)

# LBFGS polish from the annealed point (§9). Annealing stops on a criterion rule and leaves
# the gradient large, where J⁻¹ = (G′WG)⁻¹ has no interpretation. FD_STEP is measured (see
# run_smm); MAX_ITER counts LBFGS iterations, each 2d solves plus a line search. G_TOL is
# the ∞-norm of dQ/dt to stop at, three orders below that of an annealed point; missing it
# is safe, since the better of the two points is kept.
const SMM_POLISH          = env_setting(:SMM_POLISH, true)
const SMM_POLISH_FD_STEP  = env_setting(:SMM_POLISH_FD_STEP, 1e-6)
const SMM_POLISH_MAX_ITER = env_setting(:SMM_POLISH_MAX_ITER, 300)
const SMM_POLISH_G_TOL    = env_setting(:SMM_POLISH_G_TOL, 1.0)

# ============================================================
# 8. SMM spec
# ============================================================
_period_specs = [build_smm_spec(pi.data.moments, sim_smm;
                                fixed        = pi.fixed,
                                free_specs   = pi.free,
                                run          = run_params,
                                W            = pi.data.W,
                                q_scale      = Q_SCALE,
                                skip_moments = pi.data.skip)
                 for pi in PERIOD_INPUTS]
spec = CRISIS ? only(_period_specs) : joint_spec(_period_specs, BASELINE_PERIODS)

assert_all_params_accounted(spec; context = "$(WINDOW) spec")

# ============================================================
# 8a. Start point
#     :warmstart scores, under the current code, every stored point that can start this
#     window and starts from the lowest Q (stored losses are never read):
#       baseline       its SMM estimate
#       crisis window  its own output file; its other SMM estimate (_smm or
#                      _smm_mcmcbase); the baseline point at the window's period
#     :warmstart_base keeps only a crisis window's baseline point. Held parameters stay
#     the baseline's whichever candidate wins; a parameter a candidate's run did not
#     estimate keeps its DEFAULT value (the baseline point is read field by field).
# ============================================================
"""
    warm_start(spec, candidates) → SMMSpec

`spec` started at the candidate with the lowest Q under the current code, with one printed
line naming it; `spec` unchanged when no candidate has a finite Q. `candidates` holds
`(path, estimated_only)` pairs (see `point_in_spec`); an absent, unreadable or failing
candidate is passed over. A crisis window reads each at its baseline period.
"""
function warm_start(spec::SMMSpec, candidates)
    best = (Q = Inf, θ = Float64[], name = "")
    for (path, estimated_only) in candidates
        isfile(path) || continue
        b = read_bundle(path)
        (b === nothing || b.result === nothing || b.spec === nothing) && continue
        try
            CRISIS && (b = period_bundle(b, base_period))
            pt = point_in_spec(b, spec; estimated_only = estimated_only)
            Q  = smm_objective(pt.θ, spec)
            Q < best.Q && (best = (Q = Q, θ = pt.θ, name = basename(path)))
        catch
        end
    end
    isfinite(best.Q) ? @printf("\n  warm start: %s (Q = %.6f)\n", best.name, best.Q) :
                       println("\n  warm start: no candidate with a finite Q — the run starts from the DEFAULT values")
    flush(stdout)
    return isfinite(best.Q) ? _spec_with_init(spec, best.θ) : spec
end

if INIT_MODE in (:warmstart, :warmstart_base)
    _other_point = OUT_POINT === :smm ? :smm_mcmcbase : :smm
    _starts = !CRISIS                       ? [(OUT_JLS, true)] :
              INIT_MODE === :warmstart_base ? [(baseline_jls, false)] :
              [(OUT_JLS, true),
               (estimate_path(WINDOW, W_SUFFIX; point = _other_point), true),
               (baseline_jls, false)]
    spec = warm_start(spec, _starts)
else
    @printf("\n  INIT_MODE = :%s — spec init from the DEFAULT values\n", INIT_MODE)
end

if !run_params.use_as_seed
    spec = SMMSpec(perturb_free_init(spec.free, run_params.seed_perturb_frac,
                                     Random.default_rng()),
                   spec.fixed, spec.moments, spec.sim, spec.run, spec.W, spec.q_scale)
    @printf("  Perturbed the start (SD = %.0f%% of each bound range).\n",
            100 * run_params.seed_perturb_frac)
    flush(stdout)
end

print_spec(spec; k = equilibrium_at(pack_θ(spec), spec).k)
print_env_settings()

# ============================================================
# 8b. Candidate bank (INIT_MODE = :clusters)
# ============================================================
seed_bank    = nothing
prev_optimum = nothing

if INIT_MODE == :clusters
    cand_path = candidates_path(WINDOW, W_SUFFIX; point = OUT_POINT)
    seed_bank = load_or_generate_candidates(spec, cand_path;
                                            window      = WINDOW,
                                            force_regen = run_params.clusters_force_regen,
                                            show_trace  = true)
    if isempty(seed_bank.candidates)
        @warn "Candidate bank is empty — SA/DE will fall back to the spec's init point."
        seed_bank = nothing
    end

    # A crisis window adds the baseline optimum as a cluster of its own, so it is certain
    # to be a start (`_sa_starts_from_bank` takes the best member of each cluster). Scored
    # on the coarse grid like the rest of the bank; in memory only, so the cache on disk
    # stays a product of the Sobol layer.
    if !isnothing(baseline_data)
        θ_base = point_in_spec(baseline_data, spec; estimated_only = false).θ
        Q_base = smm_objective(θ_base, spec;
                               Nx   = spec.run.cand_Nx,
                               Np_U = spec.run.cand_Np_U,
                               Np_S = spec.run.cand_Np_S)

        seed_bank = if isnothing(seed_bank)
            SeedBank([θ_base], [Q_base], [1],
                     _candidate_meta(spec, WINDOW, spec.run.cand_n_sample,
                                     spec.run.cand_seed, spec.run.cand_min_cluster))
        else
            SeedBank(vcat(seed_bank.candidates, [θ_base]),
                     vcat(seed_bank.Q,          [Q_base]),
                     vcat(seed_bank.labels,     [maximum(seed_bank.labels) + 1]),
                     seed_bank.meta)
        end

        @printf("  [init] :clusters — baseline optimum injected as cluster %d (coarse-grid Q = %s).\n",
                seed_bank.labels[end],
                isfinite(Q_base) ? @sprintf("%.6e", Q_base) : "Inf")
        flush(stdout)
    end

    # The previous optimum of this run's own file, when its free set and pinned values
    # match the current spec.
    if run_params.include_prev_optimum
        _opt = _load_smm_bundle(OUT_JLS; delete_on_fail=false, label="previous optimum")
        if !isnothing(_opt)
            _o_free = [(ps.block, ps.name) for ps in _opt.spec.free]
            _c_free = [(ps.block, ps.name) for ps in spec.free]
            if _o_free == _c_free && pinned_key(_opt.spec) == pinned_key(spec)
                # The same t decodes to a different θ under a different box: after a bound
                # change, carry θ over and re-pack it in the current box.
                _same_box = all(((o, c),) -> o.lb == c.lb && o.ub == c.ub,
                                zip(_opt.spec.free, spec.free))
                prev_optimum = _same_box ? copy(_opt.result.theta_opt) :
                    pack_θ([with_seed(c, _to_constrained(t, o.lb, o.ub))
                            for (t, o, c) in zip(_opt.result.theta_opt, _opt.spec.free, spec.free)])
                @printf("  [init] INCLUDE_PREV_OPTIMUM: added valid prior optimum (Q=%.6e)%s.\n",
                        _opt.result.loss_opt,
                        _same_box ? "" : ", re-expressed in the current bounds")
            else
                @printf("  [init] INCLUDE_PREV_OPTIMUM: prior optimum invalid (free-set / fixed mismatch) — ignored.\n")
            end
        else
            @printf("  [init] INCLUDE_PREV_OPTIMUM: no readable prior optimum — ignored.\n")
        end
        flush(stdout)
    end
end

# ============================================================
# 9. Estimation
#    Simulated annealing (run_smm, method = :sa) from the chosen start, then an LBFGS
#    polish. The run's own bundle is the checkpoint path: every reheat overwrites it with
#    the incumbent, so a killed run resumes from its best point under :warmstart. The SA
#    keywords are named as their run_params fields (check_forwarding.jl pairs them); the
#    record of what SA received is the [SA config] line printed inside _sa_loop.
# ============================================================
println("Starting SMM optimisation..."); flush(stdout)

res = run_smm(spec; method = :sa, seed_bank = seed_bank, prev_optimum = prev_optimum,
              sa_scale_p_move = SA_SCALE_P_MOVE,
              sa_scale_per_k  = SA_SCALE_PER_K,
              sa_scale_sigma  = SA_SCALE_SIGMA,
              sa_halflife  = run_params.sa_halflife,
              sa_rate_tol  = run_params.sa_rate_tol,
              sa_rate_span = run_params.sa_rate_span,
              checkpoint_path = OUT_JLS)

# The lower-Q of the annealed and polished points is kept: the polish gradient is a finite
# difference on a criterion with residual roughness, so the polish may wander.
if SMM_POLISH
    @printf("\nPolishing with LBFGS from the annealed point (Q = %.6f, fd_step = %.1e)...\n",
            res.loss_opt, SMM_POLISH_FD_STEP)
    flush(stdout)
    res_polish = run_smm(spec; method = :lbfgs,
                         theta_start = copy(res.theta_opt),
                         fd_step     = SMM_POLISH_FD_STEP,
                         max_iter    = SMM_POLISH_MAX_ITER,
                         g_tol       = SMM_POLISH_G_TOL)
    if isfinite(res_polish.loss_opt) && res_polish.loss_opt < res.loss_opt
        @printf("  polish ACCEPTED:  Q %.6f → %.6f   (ΔQ = %.6f)\n",
                res.loss_opt, res_polish.loss_opt, res.loss_opt - res_polish.loss_opt)
        res = res_polish
    else
        @printf("  polish REJECTED:  Q %.6f is not below %.6f — keeping the annealed point\n",
                res_polish.loss_opt, res.loss_opt)
    end
    flush(stdout)
end

results = res

# ============================================================
# 10. Save
# ============================================================
out_estimates()   # ensures the directory exists

# The vacancy costs at the estimate and the attracting check; the check is printed only
# when a verdict fails.
eq_hat     = equilibrium_at(results.theta_opt, spec)
attract    = attracting_check(results.theta_opt, spec)
isempty(attracting_line(attract)) || println(attracting_line(attract))

save_results(results, smm_csv_path(WINDOW, W_SUFFIX; point = OUT_POINT); k = eq_hat.k)

# stage = :smm marks the finished run, replacing the :smm_checkpoint written at reheats.
# Read by the transition, MCMC standard errors, crisis windows and the warm start.
smm_jls_path = OUT_JLS
prov = run_provenance(window = WINDOW, w_suffix = W_SUFFIX, version = ROYSEARCH_VERSION)
write_bundle(smm_jls_path; result = results, spec = spec, stage = :smm,
             provenance = prov, sim = sim_smm,
             vacancy_costs = (k = eq_hat.k, se = nothing, post = nothing), attracting = attract)
@printf("Serialized SMM result → %s\n", smm_jls_path)
@printf("  provenance: schema %d, v%s, %s%s, git %s, host %s\n",
        prov.schema, prov.version, prov.window, prov.w_suffix,
        isempty(prov.git_sha) ? "(none)" : prov.git_sha,
        isempty(prov.host) ? "(none)" : prov.host)

println("\nDone.")
flush(stdout)

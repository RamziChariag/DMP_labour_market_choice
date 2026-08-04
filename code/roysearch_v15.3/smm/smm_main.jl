############################################################
# smm/smm_main.jl — SMM estimation entry point
#
# Usage (from the repo root):
#   julia --project --threads auto smm/smm_main.jl
#
# Project layout:
#   code/
#     solver/   — loaded as a library
#     smm/
#       smm_main.jl    — this file
#       moments.jl
#       smm_params.jl
#       smm.jl
#   data/
#     derived/  — windows.json, moments_{w}.csv, sampling_var_{w}.csv,
#                 nu_estimation.csv, phi_calibration.csv,
#                 training_share_scale.csv (κ_w; optional but
#                                            recommended)
#   output/
#     plots/
#     tables/
#     smm/      — serialised SMMResult bundles
#
# Two-stage workflow (Model Notes §2 / data_and_moments.pdf §18):
#   Baseline window  (:base_fc, :base_covid):
#       Fix (r, ν, φ) from external calibration, η_U/η_S at 0.5 (Hosios),
#       and σ_wU/σ_wS from λ_w.  Estimate the remaining free parameters
#       (all 6 deep + 22 regime specs, less those pinned via FIX_PARAMS).
#       ν is loaded from nu_estimation.csv using the row for the
#       corresponding baseline (one row per crisis pair).
#   Crisis window    (:crisis_fc, :crisis_covid):
#       Load the matching base bundle (base_fc → crisis_fc;
#       base_covid → crisis_covid).
#       Fix the deep structural parameters at the baseline estimates:
#           common  a_ℓ, b_ℓ, ρ_x
#           regime  bU, bT, bS
#       Re-estimate only the regime-specific parameters.
#       ν comes from the baseline row (NOT a crisis-specific value).
#
# WINDOWS are read from data/derived/windows.json — the single source
# of truth shared with the data pipeline. training_share carries the
# NSC κ_w level adjustment, applied upstream when the data pipeline
# writes moments_{w}.csv / sampling_var_{w}.csv; the loaders here read
# the pre-adjusted values (see the moments.jl header for the convention).
############################################################

const ROYSEARCH_VERSION = "15.3"
println("="^60)
println("  RoySearch v$(ROYSEARCH_VERSION) — SMM Estimation")
println("="^60)
flush(stdout)

# Paths
const SMM_DIR      = @__DIR__
const SOLVER_DIR   = joinpath(SMM_DIR, "..", "solver")
const PROJECT_ROOT = joinpath(SMM_DIR, "..", "..")
const OUTPUT_DIR   = joinpath(PROJECT_ROOT, "output")
const TABLES_DIR   = joinpath(OUTPUT_DIR, "tables")
const SMM_OUT_DIR  = joinpath(OUTPUT_DIR, "smm")

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
using Optim
using CSV
using DataFrames
using Serialization
using Clustering          # hclust / cutree for the candidate-clustering layer
using QuasiMonteCarlo
using JSON3

println("done."); flush(stdout)

Random.seed!(1234)

# The dense LU / kernel BLAS calls run inside a @threads region, so multi-threaded
# BLAS would oversubscribe every core with the outer thread count.  Pin BLAS to one
# thread and let Base.Threads own the parallelism.
LinearAlgebra.BLAS.set_num_threads(1)

# Solver as a library (no solving happens yet)
print("Loading solver modules... "); flush(stdout)

include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))

println("done."); flush(stdout)

# SMM modules
print("Loading SMM modules... "); flush(stdout)

include(joinpath(SMM_DIR, "moments.jl"))
include(joinpath(SMM_DIR, "smm_params.jl"))
include(joinpath(SMM_DIR, "smm.jl"))
include(joinpath(SMM_DIR, "candidates.jl"))

println("done."); flush(stdout)
@printf("Threads available: %d\n\n", Threads.nthreads())
flush(stdout)

# ============================================================
# 1. Solver settings
#    Use coarser tolerances than the single-solve script to keep
#    each SMM iteration fast.
# ============================================================
sim_smm = SimParams(
    tol_inner          = 1e-7,
    tol_outer_U        = 1e-6,
    tol_outer_S        = 1e-6,
    tol_global         = 1e-4,

    damp_inner_U       = 0.95,
    damp_inner_S       = 0.95,

    inner_B            = 20,     # inner divergence early-abort burn-in (0 disables)
    inner_K            = 10,      # inner no-contraction window W ≡ K; reject after inner_K divergent outer iters

    outer_B            = 30,     # outer stall-detect burn-in (0 disables the handback)
    outer_K            = 10,     # outer no-contraction window; hand back to global on stall

    global_B           = 8,      # global stall-detect burn-in (0 disables); loss-neutral early stop
    global_K           = 4,      # global no-improvement window; stop a limit-cycling U↔S coupling

    maxit_inner        = 300,
    maxit_outer        = 200,
    maxit_global       = 20,

    conv_streak        = 1,

    use_anderson       = true,
    anderson_m         = 1,
    anderson_reg       = 1e-10,

    damp_pstar_U       = 1.00,
    damp_pstar_S       = 0.50,

    verbose            = 0,      # 0: model silent; 1: outer convergence per iter; 2: also inner detail
    verbose_stride     = 100,
)

# ============================================================
# 2. Estimation window
#    Valid windows are loaded from data/derived/windows.json
#    written by the data pipeline.
# ============================================================
WINDOW = :crisis_covid

# Crisis → baseline pair map. crisis_fc pairs with base_fc; the
# crisis_covid pair with base_covid. Used to pick the right ν row
# and to load the right baseline bundle in a crisis re-estimation.
const _PAIR_BASELINE = Dict(
    :base_fc      => :base_fc,
    :crisis_fc    => :base_fc,
    :base_covid   => :base_covid,
    :crisis_covid => :base_covid,
)

# ============================================================
# Moments given zero weight in the objective. The same list feeds both
# load_weight_matrix and build_smm_spec so sigma / W is subsetted consistently
# with the loss function.
#
# ur_total is an exact linear redundancy of the active set (the
# skilled-share-weighted average of ur_U and ur_S); wage_premium is likewise
# redundant (wage_premium ≡ mean_wage_S − mean_wage_U). Both remain in
# MOMENT_NAMES and the data pipeline.
#
# Valid names (31 moments in MOMENT_NAMES order):
#   :ur_total, :ur_U, :ur_S, :skilled_share, :training_share,
#   :emp_var_U, :emp_cm3_U, :emp_var_S, :emp_cm3_S,
#   :jfr_U, :sep_rate_U, :jfr_S, :sep_rate_S,
#   :ee_rate_S,
#   :mean_wage_U, :mean_wage_S,
#   :p25_wage_U, :p25_wage_S, :p50_wage_U, :p50_wage_S, :p75_wage_U, :p75_wage_S,
#   :wage_premium, :theta_U, :theta_S,
#   :overlap_UgtS, :overlap_SltU, :ltu_share_S,
#   :wchg_rate_U, :wchg_rate_S, :ee_step_S
#
# A moment whose data target is NaN for the active window is auto-held-out of
# the objective (see ACTIVE_SKIP_MOMENTS below), so wchg_rate_j need NOT be
# listed here — it activates automatically once its SIPP data is built.
# ============================================================
SKIP_MOMENTS = Symbol[
    :ur_total,
    # ee_step_S (SIPP skilled EE-move wage step) is computed and carried through
    # the pipeline (target + sampling variance) but given zero objective weight.
    :ee_step_S,
    :wage_premium,  # redundant with mean_wage_S − mean_wage_U
]

# ── Per-window objective holdouts ───────────────────────────────────────────
# Maps a window to the moments given zero weight in that window only, on top of
# SKIP_MOMENTS. Empty ⇒ no per-window holdouts. Add `window => [:moment, …]` to
# exclude a finite-valued moment from the objective in one window.
const WINDOW_SKIP_MOMENTS = Dict{Symbol,Vector{Symbol}}()

# Wage-reliability calibration knob (single source of truth for both the run
# header and calibrate_sigma_w). σ_w is calibrated externally from λ_w rather
# than estimated. Bound–Krueger (1991) avg between men and women:
# λ_w ≈ 0.87 for log annual earnings.
const LAMBDA_W = 0.78

@printf("Estimation window: %s\n", WINDOW)
flush(stdout)

# ============================================================
# Parameters to pin at a fixed value during estimation.
# r / ν / φ are always fixed from external calibration and do not appear here.
# σ_wU / σ_wS are calibrated externally (calibrate_sigma_w from λ_w) and injected
# into the pinned block from the data rather than listed here.
#
# Valid keys (ASCII):
#   common:  :a_l  :b_l  :rho_x  :c  :A
#   regime:  :PU  :PS  :bU  :bT  :bS  :alpha_U  :a_Gam  :b_Gam
#   unsk:    :unsk_mu  :unsk_eta  :unsk_k  :unsk_bet  :unsk_lam  :unsk_xi  :unsk_sigw
#   skl:     :skl_mu   :skl_eta   :skl_k   :skl_bet
#            :skl_lam  :skl_sig   :skl_sigw  :skl_xi  :skl_delta
#   (:unsk_sigw / :skl_sigw = wage measurement-error SD σ_wU / σ_wS)
#   (:skl_delta = offer/shock support ratio δ_S; :skl_xi = exogenous separation ξ_S)
# ============================================================
FIX_PARAMS = Dict{Symbol,Float64}(
     :unsk_eta => 0.50000,
    # :unsk_bet => 0.18800,
     :skl_eta  => 0.50000,
    # :skl_bet  => 0.27200,
)

# ============================================================
# INIT_MODE — how the optimiser is seeded.
#   :default    seed the spec init from DEFAULT_PARAMS; skip the candidate layer.
#   :warmstart  seed from a saved optimum; fall back to DEFAULT_PARAMS if absent.
#   :clusters   seed SA/DE from the Sobol→hclust candidate bank.
# USE_AS_SEED           true: use the entered init verbatim; false: perturb it
#                       before estimation. Under :default the init IS the
#                       optimiser start, so true starts exactly at DEFAULT_PARAMS.
# SEED_PERTURB_FRAC     perturbation SD as a fraction of each bound range
#                       (applies when USE_AS_SEED = false).
# CLUSTERS_FORCE_REGEN  rebuild the candidate cache even if present (:clusters).
# INCLUDE_PREV_OPTIMUM  add a valid saved optimum as a guaranteed seed (:clusters).
#                       This is THIS window's own prior optimum; in a crisis
#                       window the paired baseline optimum is a separate,
#                       always-on start under :clusters (§8b).
# ============================================================
INIT_MODE            = :warmstart
USE_AS_SEED          = true
SEED_PERTURB_FRAC    = 0.05
CLUSTERS_FORCE_REGEN = true
INCLUDE_PREV_OPTIMUM = false

# Starting values for the free parameters, keyed by the ASCII convention in
# _DEFAULT_PARAM_KEY. Fixed r/ν/φ are ignored here (set from calibration files).
const DEFAULT_PARAMS = Dict{Symbol,Float64}(
    :r         => 0.00416667,
    :nu        => 0.00323000,
    :phi       => 0.02222129,

    :a_l       => 0.98709000,
    :b_l       => 2.22587000,
    :rho_x     => -0.92409000,
    :c         => 12.17755000,
    :A         => 5.97385000,

    :PU        => 6.12832000,
    :PS        => 12.10972000,

    :bU        => 0.14372000,
    :bT        => 3.80570000,
    :bS        => 0.04705000,

    :alpha_U   => 2.34302000,

    :a_Gam     => 1.24305000,
    :b_Gam     => 1.68178000,
    :skl_delta => 0.96109000,

    :unsk_mu   => 0.42744000,
    :unsk_eta  => 0.50000000,
    :unsk_k    => 2.28464000,
    :unsk_bet  => 0.52872000,
    :unsk_lam  => 0.15743000,
    :unsk_xi   => 0.00527000,
    :unsk_sigw => 0.23110000,

    :skl_mu    => 0.23529000,
    :skl_eta   => 0.50000000,
    :skl_k     => 1.76695000,
    :skl_bet   => 0.43353000,
    :skl_lam   => 0.13677000,
    :skl_sig   => 0.87475000,
    :skl_sigw  => 0.21047000,
    :skl_xi    => 0.00505000,
)

# (block, unicode name) → DEFAULT_PARAMS key (ASCII).
const _DEFAULT_PARAM_KEY = Dict{Tuple{Symbol,Symbol}, Symbol}(
    (:common, :a_ℓ) => :a_l,     (:common, :b_ℓ)  => :b_l,     (:common, :c)   => :c,
    (:common, :ρ_x) => :rho_x,   (:common, :A)    => :A,
    (:unsk,   :PU)  => :PU,      (:skl,    :PS) => :PS,
    (:unsk,   :bU)  => :bU,      (:unsk,   :bT)   => :bT,      (:skl,    :bS)  => :bS,
    (:unsk,   :α_U) => :alpha_U, (:skl,    :a_Γ)  => :a_Gam,  (:skl,    :b_Γ) => :b_Gam,
    (:unsk,   :μ)   => :unsk_mu, (:unsk,   :η)    => :unsk_eta, (:unsk,  :k)   => :unsk_k,
    (:unsk,   :β)   => :unsk_bet, (:unsk,  :λ)   => :unsk_lam,  (:unsk, :σ_w)  => :unsk_sigw,
    (:skl,    :μ)   => :skl_mu,  (:skl,    :η)    => :skl_eta,  (:skl,   :k)   => :skl_k,
    (:skl,    :β)   => :skl_bet, (:skl,    :λ)   => :skl_lam,
    (:skl,    :σ)   => :skl_sig, (:skl,    :σ_w)  => :skl_sigw,
    (:unsk,   :ξ)   => :unsk_xi, (:skl,    :ξ)   => :skl_xi,  (:skl,    :δ)    => :skl_delta,
)

# ASCII key (FIX_PARAMS convention) → unicode fixed-NamedTuple key.
const _ASCII_TO_FIXED_KEY = Dict{Symbol, Symbol}(
    :r        => :r,
    :nu       => :ν,
    :phi      => :φ,
    :a_l      => :a_ℓ,
    :b_l      => :b_ℓ,
    :rho_x    => :ρ_x,
    :c        => :c,
    :A        => :A,
    :PU       => :PU,
    :PS       => :PS,
    :bU       => :bU,
    :bT       => :bT,
    :bS       => :bS,
    :alpha_U  => :α_U,
    :a_Gam    => :a_Γ,
    :b_Gam    => :b_Γ,
    :unsk_mu  => :unsk_μ,
    :unsk_eta => :unsk_η,
    :unsk_k   => :unsk_k,
    :unsk_bet => :unsk_β,
    :unsk_lam => :unsk_λ,
    :unsk_xi  => :unsk_ξ,
    :unsk_sigw => :unsk_σ_w,
    :skl_mu   => :skl_μ,
    :skl_eta  => :skl_η,
    :skl_k    => :skl_k,
    :skl_bet  => :skl_β,
    :skl_lam  => :skl_λ,
    :skl_sig  => :skl_σ,
    :skl_sigw  => :skl_σ_w,
    :skl_xi   => :skl_ξ,
    :skl_delta => :skl_δ,
)

"""
    _fix_params_to_nt(fix_dict) → NamedTuple
"""
function _fix_params_to_nt(fix_dict::Dict{Symbol,Float64}) :: NamedTuple
    isempty(fix_dict) && return (;)
    keys_vec = Symbol[]
    vals_vec = Float64[]
    for (ascii_key, val) in fix_dict
        if haskey(_ASCII_TO_FIXED_KEY, ascii_key)
            push!(keys_vec, _ASCII_TO_FIXED_KEY[ascii_key])
            push!(vals_vec, val)
        else
            valid_str = join(sort(string.(collect(keys(_ASCII_TO_FIXED_KEY)))), ", ")
            @printf("WARNING: FIX_PARAMS — unrecognised key :%s — ignored.\n", ascii_key)
            @printf("  Valid keys: %s\n", valid_str)
        end
    end
    isempty(keys_vec) && return (;)
    return NamedTuple{Tuple(keys_vec)}(Tuple(vals_vec))
end

# ============================================================
# 3. Windows + data moments
# ============================================================
derived_dir = joinpath(PROJECT_ROOT, "data", "derived")

# Verify WINDOW against windows.json (single source of truth)
_win_info = load_windows(; derived_dir=derived_dir)
if !(WINDOW in keys(_win_info.windows))
    error("WINDOW = :$WINDOW not found in windows.json. " *
          "Valid windows: " * join(string.(_win_info.order), ", "))
end
_wd = _win_info.windows[WINDOW]
@printf("  Window definition (windows.json):  %s  ym = %d..%d  ASEC %d..%d\n",
        _wd.label, _wd.ym_start, _wd.ym_end,
        first(_wd.asec_years), last(_wd.asec_years))
flush(stdout)

moments = load_data_moments(; window=WINDOW, derived_dir=derived_dir)

# ── Auto-hold-out of NaN targets ──────────────────────────────────────────
# A moment whose data target is NaN for this window (no source data yet —
# e.g. wchg_rate_j before base_covid SIPP is built) cannot enter g'Wg. Union
# such moments into the skip set so the objective and weight matrix are all
# subsetted identically. This is what lets a base_covid run proceed
# unchanged today while crisis_covid (with SIPP) uses the moment — no name is
# hardcoded into SKIP_MOMENTS, so each moment activates automatically once its
# data appears.
_nan_targets = Symbol[k for k in keys(moments) if !isfinite(moments[k].value)]

# Per-window holdouts (WINDOW_SKIP_MOMENTS above): a FINITE moment whose data
# construction is unreliable in this window is dropped from the objective here.
# Unlike the NaN path this is deliberate — the target is measured and finite,
# so it would NOT auto-hold-out — hence the explicit per-window spec.
_window_skip = get(WINDOW_SKIP_MOMENTS, WINDOW, Symbol[])
ACTIVE_SKIP_MOMENTS = sort(unique(vcat(SKIP_MOMENTS, _nan_targets, _window_skip)))
if !isempty(_nan_targets)
    @printf("Auto-skipping %d moment(s) with NaN target in %s: %s\n",
            length(_nan_targets), WINDOW, join(string.(_nan_targets), ", "))
    flush(stdout)
end
if !isempty(_window_skip)
    @printf("Per-window holdout: %d moment(s) excluded from objective in %s: %s\n",
            length(_window_skip), WINDOW, join(string.(_window_skip), ", "))
    flush(stdout)
end

# ── σ_w calibration (calibrated model parameters) ─────────────────────────
# σ_{w,j}² = (1−λ_w)·Var̂[log w_j]. Calibrated from the data here, then pinned
# in the fixed block alongside r, ν, φ (§6).
σ_wU_cal, σ_wS_cal = calibrate_sigma_w(LAMBDA_W, moments)
@printf("  Calibrated σ_w (λ_w = %.4f):  σ_wU = %.5f,  σ_wS = %.5f\n",
        LAMBDA_W, σ_wU_cal, σ_wS_cal)

# σ_w identification switch.  λ_w > 0: pin σ_w at the calibrated value, with β
# free.  λ_w == 0: leave σ_w free (estimated) and pin β instead via FIX_PARAMS
# (:unsk_bet, :skl_bet) — only one of {β, σ_w} is identified from wage
# dispersion, so exactly one is fixed.
_sigw_fixed = LAMBDA_W > 0.0 ?
    (unsk_σ_w = σ_wU_cal, skl_σ_w = σ_wS_cal) : NamedTuple()
if LAMBDA_W == 0.0
    @printf("  λ_w = 0 → σ_w is ESTIMATED (free), not pinned.\n")
    if !(haskey(FIX_PARAMS, :unsk_bet) && haskey(FIX_PARAMS, :skl_bet))
        @warn "λ_w = 0 frees σ_w but β is not fully pinned in FIX_PARAMS " *
              "(:unsk_bet / :skl_bet) — β and σ_w are jointly under-identified."
    end
end
flush(stdout)

# ============================================================
# 4. Weight matrix
#    W_COND_TARGET controls which weighting scheme is used:
#      0.0   diagonal-σ: W = Diagonal(1/σ̂²_samp) from sampling_var_{window}.csv
#            (per-moment sampling variances on one footing)
#      2.0   equal weights (identity, no W matrix)
# ============================================================
W_COND_TARGET = 0.0  

function _w_suffix(cond_target::Float64)
    cond_target == 0.0 && return "_diagonalW"
    cond_target == 2.0 && return "_equalW"
    error("W_COND_TARGET must be 0.0 (diagonal-σ) or 2.0 (equal weights); got $cond_target.")
end

W_SUFFIX = _w_suffix(W_COND_TARGET)

W_opt = load_weight_matrix(; window=WINDOW, derived_dir=derived_dir,
                             cond_target=W_COND_TARGET,
                             skip_moments=ACTIVE_SKIP_MOMENTS)

# Reported Q is the raw objective magnitude: the relative-deviation loss
# under equal weights, or g'Wg with the diagonal sampling-variance weights.
# No display-only rescaling is applied (q_scale = 1.0, constant in θ).
Q_SCALE = 1.0
@printf("  q_scale (display-only Q divisor) = %.6e\n", Q_SCALE)


# ============================================================
# 5. Externally calibrated parameters (r, ν, φ)
#    ν is picked per crisis pair: WINDOW ∈ {base_fc, crisis_fc} uses
#    the base_fc row of nu_estimation.csv; the COVID pair uses
#    base_covid.
# ============================================================
println("\nLoading externally calibrated parameters (r, ν, φ)...")
flush(stdout)
calib = load_calibrated_params(; window=WINDOW, derived_dir=derived_dir)

@printf("  Calibrated:  r = %.6f,  ν = %.5f,  φ = %.5f\n",
        calib.r, calib.nu, calib.phi)
flush(stdout)

# ============================================================
# 6. Fixed and free parameters
#
#    r, ν, φ are always fixed from external calibration.  The deep /
#    regime-specific split is defined once, in smm_params.jl, as DEEP_PARAMS and
#    REGIME_SPECIFIC_PARAMS (Model Notes §2); both branches below read those
#    sets rather than restating the membership, so the partition has a single
#    source of truth.  A baseline window estimates everything;  a crisis window
#    estimates REGIME_SPECIFIC_PARAMS and pins DEEP_PARAMS at the paired
#    baseline optimum.
# ============================================================

"""
    _baseline_param_value(block, name, cp, up, sp) → Float64
    _baseline_param_value(ps::ParamSpec, cp, up, sp) → Float64

Read one parameter out of a decoded optimum (`cp`, `up`, `sp`).  The
(block, name) method serves callers that iterate DEEP_PARAMS, which holds bare
pairs rather than ParamSpecs.
"""
function _baseline_param_value(block::Symbol, name::Symbol, cp, up, sp) :: Float64
    if     block == :common; return Float64(getfield(cp, name))
    elseif block == :unsk;   return Float64(getfield(up, name))
    else                     return Float64(getfield(sp, name))
    end
end

_baseline_param_value(ps::ParamSpec, cp, up, sp) :: Float64 =
    _baseline_param_value(ps.block, ps.name, cp, up, sp)

if WINDOW in (:crisis_fc, :crisis_covid)
    # Crisis re-estimation — load the matching baseline bundle:
    #   crisis_fc    ← base_fc
    #   crisis_covid ← base_covid
    baseline_window = _PAIR_BASELINE[WINDOW]
    baseline_jls = joinpath(SMM_OUT_DIR,
                            "smm_result_$(baseline_window)$(W_SUFFIX).jls")
    @printf("\nCrisis window detected — loading baseline (%s) from:\n  %s\n",
            baseline_window, baseline_jls)
    flush(stdout)
    isfile(baseline_jls) || error(
        "Baseline result not found at $baseline_jls. " *
        "Run the $baseline_window estimation first (WINDOW = :$baseline_window, " *
        "W_COND_TARGET = $W_COND_TARGET).")

    baseline_data = _load_smm_bundle(baseline_jls; delete_on_fail=false, label="baseline file")
    isnothing(baseline_data) && error(
        "Baseline result at $baseline_jls is unreadable. " *
        "Re-run the $baseline_window estimation first."
    )

    baseline_result = baseline_data.result
    baseline_spec   = baseline_data.spec

    cp_base, up_base, sp_base =
        unpack_θ(baseline_result.theta_opt, baseline_spec)

    @printf("  Baseline Q = %.6e  (converged = %s)\n",
            baseline_result.loss_opt, baseline_result.converged)
    flush(stdout)

    # Fixed: external calibration (r, ν, φ) + calibrated σ_w + the deep block,
    # every entry of it, read off the baseline optimum.
    #
    # The deep block is assembled by iterating DEEP_PARAMS rather than listing
    # its members here.  A hand-written list is a silent-failure hazard: a deep
    # parameter missing from it is excluded from `free_params` (the loop below
    # keeps only REGIME_SPECIFIC_PARAMS) AND absent from `fixed_params`, so
    # `unpack_θ` falls through to its hard-coded default and the crisis window is
    # estimated at a value unrelated to the baseline. `fixed_key` picks the key
    # convention `unpack_θ` reads back.
    #
    # Later entries win the merge: FIX_PARAMS overrides a baseline-derived deep
    # value, and the calibrated σ_w overrides both.
    _extra_fixed = _fix_params_to_nt(FIX_PARAMS)
    _deep_fixed  = NamedTuple(
        fixed_key(blk, nm) => _baseline_param_value(blk, nm, cp_base, up_base, sp_base)
        for (blk, nm) in DEEP_PARAMS_ORDERED
    )
    fixed_params = merge(
        (
        r = calib.r,
        ν = calib.nu,
        φ = calib.phi,
        ),
        _deep_fixed,
        _extra_fixed,
        _sigw_fixed,
    )

    # Free parameters: regime-specific only.  Init values follow INIT_MODE:
    #   :warmstart → the matched baseline optimum;  otherwise → DEFAULT_PARAMS.
    free_params = ParamSpec[]
    for ps in default_free_params()
        (ps.block, ps.name) in REGIME_SPECIFIC_PARAMS || continue
        init_val = if INIT_MODE == :warmstart
            _baseline_param_value(ps, cp_base, up_base, sp_base)
        else
            ascii_key = get(_DEFAULT_PARAM_KEY, (ps.block, ps.name), nothing)
            isnothing(ascii_key) ? ps.init : get(DEFAULT_PARAMS, ascii_key, ps.init)
        end
        init_val = clamp(init_val, ps.lb + 1e-10, ps.ub - 1e-10)
        push!(free_params,
              ParamSpec(ps.block, ps.name, ps.lb, ps.ub, init_val, ps.label))
    end

    # The baseline optimum in the crisis window's own free-parameter ordering.
    # The crisis free set is a subset of the baseline's, so the map is by
    # (block, name) and never by index.  Under :warmstart these values are
    # already the spec inits; §8b also injects the vector into the seed bank so
    # that :clusters evaluates it as a start.
    baseline_optimum_values = Dict(
        (ps.block, ps.name) => _baseline_param_value(ps, cp_base, up_base, sp_base)
        for ps in free_params
    )

    println("\n  Deep structural parameters FIXED from baseline:")
    @printf("    common:  a_ℓ=%.4f  b_ℓ=%.4f  ρ_x=%.4f\n",
            cp_base.a_ℓ, cp_base.b_ℓ, cp_base.ρ_x)
    @printf("    unsk:    bU=%.4f  bT=%.4f      skl:  bS=%.4f\n",
            up_base.bU, up_base.bT, sp_base.bS)
    @printf("  Regime-specific parameters FREE (%d params); init from %s\n",
            length(free_params),
            INIT_MODE == :warmstart ? "baseline optimum" : "DEFAULT_PARAMS")
    flush(stdout)

else
    # Baseline estimation.  Fix r / ν / φ and the calibrated σ_w, plus anything
    # in FIX_PARAMS.  There is no paired baseline optimum to carry into the seed
    # bank here — the window IS the baseline.
    baseline_optimum_values = nothing

    _extra_fixed = _fix_params_to_nt(FIX_PARAMS)
    fixed_params = merge((
        r = calib.r,
        ν = calib.nu,
        φ = calib.phi,
        ),
        _extra_fixed, 
        _sigw_fixed,
    )

    free_params = default_free_params()

    _warmstart_jls = joinpath(SMM_OUT_DIR, "smm_result_$(WINDOW)$(W_SUFFIX).jls")

    if INIT_MODE == :warmstart && isfile(_warmstart_jls)
        println("\n  INIT_MODE = :warmstart — loading prior parameter values from:")
        @printf("    %s\n", _warmstart_jls)
        flush(stdout)

        _ws_data = _load_smm_bundle(_warmstart_jls; delete_on_fail=false, label="warm-start file")

        if !isnothing(_ws_data)
            _ws_vals = Dict{Tuple{Symbol,Symbol}, Float64}()
            try
                _ws_result = _ws_data.result
                _ws_spec   = _ws_data.spec
                _ws_cp, _ws_up, _ws_sp = unpack_θ(_ws_result.theta_opt, _ws_spec)
                for ps in _ws_spec.free
                    has_field = try
                        _baseline_param_value(ps, _ws_cp, _ws_up, _ws_sp)
                        true
                    catch
                        false
                    end
                    has_field && (_ws_vals[(ps.block, ps.name)] =
                                  _baseline_param_value(ps, _ws_cp, _ws_up, _ws_sp))
                end
                @printf("    Prior Q = %.6e  (converged = %s)\n",
                        _ws_result.loss_opt, _ws_result.converged)
            catch e
                @warn "Warm-start: could not decode prior parameters ($e) — using defaults."
            end

            if !isempty(_ws_vals)
                free_params = [
                    let init_val = get(_ws_vals, (ps.block, ps.name), ps.init)
                        init_val = clamp(init_val, ps.lb + 1e-10, ps.ub - 1e-10)
                        ParamSpec(ps.block, ps.name, ps.lb, ps.ub, init_val, ps.label)
                    end
                    for ps in free_params
                ]
                n_matched = count(ps -> haskey(_ws_vals, (ps.block, ps.name)), free_params)
                @printf("    Warm-start: matched %d / %d parameters from prior run.\n",
                        n_matched, length(free_params))
            end
        else
            println("    Warm-start skipped — could not read file.")
        end
    else
        # :default and :clusters seed the spec's init from DEFAULT_PARAMS.
        # (:warmstart with no prior file also lands here.)
        if INIT_MODE == :warmstart
            println("\n  INIT_MODE = :warmstart but no prior file — seeding from DEFAULT_PARAMS.")
        else
            println("\n  INIT_MODE = :$(INIT_MODE) — seeding free-parameter init from DEFAULT_PARAMS.")
        end
        flush(stdout)

        free_params = [
            let ascii_key = get(_DEFAULT_PARAM_KEY, (ps.block, ps.name), nothing),
                raw_val   = isnothing(ascii_key) ? ps.init :
                                get(DEFAULT_PARAMS, ascii_key, ps.init),
                init_val  = clamp(raw_val, ps.lb + 1e-10, ps.ub - 1e-10)
                ParamSpec(ps.block, ps.name, ps.lb, ps.ub, init_val, ps.label)
            end
            for ps in free_params
        ]

        n_matched = count(ps -> !isnothing(get(_DEFAULT_PARAM_KEY, (ps.block, ps.name), nothing)) &&
                                 haskey(DEFAULT_PARAMS,
                                        get(_DEFAULT_PARAM_KEY, (ps.block, ps.name), :__missing__)),
                          free_params)
        @printf("    Matched %d / %d free parameters from DEFAULT_PARAMS.\n",
                n_matched, length(free_params))
        flush(stdout)
    end

    @printf("\n  Baseline mode: all structural + regime params are FREE (%d total)\n",
            length(free_params))
    flush(stdout)
end

@printf("  Fixed params:  %d  |  Free params:  %d\n",
        length(fixed_params), length(free_params))
if !isempty(FIX_PARAMS)
    _valid_fix = [k for k in keys(FIX_PARAMS) if haskey(_ASCII_TO_FIXED_KEY, k)]
    @printf("  FIX_PARAMS pinned (%d): %s\n",
            length(_valid_fix), join(string.(_valid_fix), ", "))
end
flush(stdout)

# ============================================================
# 7. Run parameters — grids, SA, DE, NM, tracing
# ============================================================
run_params = SMMRunParams(
    # ── Discretisation grids ─────────────────────────────────
    # Nx     : worker-type grid (Gauss-Legendre nodes on [0,1])
    # Np_U   : unskilled match-quality grid
    # Np_S   : skilled match-quality grid
    Nx      = 120,
    Np_U    = 120,
    Np_S    = 120,

    # ── Candidate-generation layer (used when INIT_MODE = :clusters) ──
    cand_Nx          = 40,
    cand_Np_U        = 40,
    cand_Np_S        = 40,
    cand_n_sample    = 2048,
    cand_seed        = 42,
    cand_min_cluster = 5,      # min number of candidates per cluster (hclust leaf)

    # ── Weight-matrix mode (passed to print_spec for the header) ──
    # cond_target semantics (see load_weight_matrix): 0 = diagonal-σ,
    # 2 = equal weights.
    w_cond_target = W_COND_TARGET,

    # ── Wage-reliability calibration knob (header-only; σ_w is calibrated) ──
    λ_w = LAMBDA_W,

    # ── Simulated annealing ──────────────────────────────────
    sa_max_iter        = 20_000,   # max SA iterations
    sa_T0              = 10.0,    # initial temperature (≤0 ⇒ auto-calibrate
                                  # from uphill probes)
    sa_step            = 0.20,    # initial proposal sd in unconstrained space
    sa_cooling_rate    = 1.0,     # rate in T(t) = T_reheat / log(1+rate·t)^exp
    sa_cooling_exp     = 2.0,     # exponent in same schedule (higher = faster cool)
    sa_reheat_patience = 400,   # steps without improvement before a reheat, 0 to disable reheats
    sa_reheat_factor   = 4.00,    # multiplicative reheat: T ← T · factor
    sa_max_reheats     = 8,       # cap on number of reheats per run, 0 is unlimited
    sa_adapt_window    = 50,      # rolling window for adaptive step / acceptance
    sa_target_fin      = 0.90,    # target feasibility (finite-Q) fraction;
                                  # below this, step shrinks
    sa_random_init     = false,   # false ⇒ start from pack_θ(spec); true ⇒
                                  # uniform draw inside [lb, ub]
    sa_parallel_steps  = 100,     # :clusters — parallel SA steps per cluster 
                                  # before pruning to the best basin
    sa_seed            = 42,      # base seed for the parallel SA chains

    # ── Differential evolution ───────────────────────────────
    de_max_iter  = 2_000,         # max generations
    de_pop_size  = 120,           # population size (0 ⇒ 10·n_free_params)
    de_f         = 0.70,          # DE differential weight (mutation strength)
    de_cr        = 0.85,          # DE crossover probability
    de_patience  = 2,             # stop after this many generations with no
                                  # improvement
    de_avg_tol   = 1e-6,          # stop when (Q_mean - Q_best)/|Q_best| < tol;
                                  # 0 disables this early-stop

    # ── Nelder-Mead local polish ─────────────────────────────
    nm_max_iter  = 7_000,         # max NM iterations
    nm_f_tol     = 1e-6,          # function-value tolerance
    nm_x_tol     = 1e-4,          # parameter tolerance (unconstrained space)
    nm_g_tol     = 1e-5,          # gradient tolerance (unconstrained space)
    nm_no_improve = 4_000,        # early-stop: halt NM after this many objective
                                  # evaluations with no improvement in best Q
                                  # (0 disables; same counter as the [iter N] trace)

    # ── Tracing ──────────────────────────────────────────────
    show_trace_members     = false,   # per-member trace inside one DE/SA gen
    show_trace_generations = true,    # per-generation / per-iteration summary
    trace_stride           = 100,     # print every N iterations in SA 
)

# ============================================================
# 8. Build SMM spec
# ============================================================
# USE_AS_SEED = false starts the optimiser from a perturbation of the entered
# init rather than the init itself.
if !USE_AS_SEED
    free_params = perturb_free_init(free_params, SEED_PERTURB_FRAC, Random.default_rng())
    @printf("  Perturbed the seed init (SD = %.0f%% of each bound range).\n",
            100 * SEED_PERTURB_FRAC)
    flush(stdout)
end

spec = build_smm_spec(
    moments, sim_smm;
    fixed        = fixed_params,
    free_specs   = free_params,
    run          = run_params,
    W            = W_opt,
    q_scale      = Q_SCALE,
    skip_moments = ACTIVE_SKIP_MOMENTS,
)

# Every estimable parameter must be either free or pinned.  One that is neither
# takes unpack_θ's hard-coded fallback without any error or header mention — the
# failure mode that left ρ_x at −0.55 in both crisis windows through v14.8.
assert_all_params_accounted(spec; context = "$(WINDOW) spec")

print_spec(spec)

# ============================================================
# 8b. Candidate seed bank (used when INIT_MODE = :clusters)
# ============================================================
seed_bank    = nothing
prev_optimum = nothing

if INIT_MODE == :clusters
    cand_path = joinpath(SMM_OUT_DIR, "candidates_$(WINDOW)$(W_SUFFIX).jls")
    seed_bank = load_or_generate_candidates(spec, cand_path;
                                            window      = WINDOW,
                                            force_regen = CLUSTERS_FORCE_REGEN,
                                            show_trace  = true)
    if isempty(seed_bank.candidates)
        @warn "Candidate bank is empty — SA/DE will fall back to the spec's init point."
        seed_bank = nothing
    end

    # In a crisis window, add the baseline optimum to the bank as a start.
    # The Sobol sample covers the box blind and has no reason to place a point
    # near the baseline estimate, so a cold cluster search would never evaluate
    # the one vector with a strong prior claim on being close to the answer.
    # It gets its own cluster label because `_sa_starts_from_bank` emits the
    # best-Q member of each cluster: in a label of its own it is guaranteed to
    # become a start rather than being outranked by a neighbour's member.
    #
    # Evaluated on the candidate layer's coarse grid so its Q is on the same
    # footing as the rest of the bank.  The injection is in-memory only — the
    # cache on disk stays a pure product of the Sobol layer.
    if !isnothing(baseline_optimum_values)
        θ_base = pack_θ([
            ParamSpec(ps.block, ps.name, ps.lb, ps.ub,
                      clamp(get(baseline_optimum_values, (ps.block, ps.name), ps.init),
                            ps.lb + 1e-10, ps.ub - 1e-10),
                      ps.label)
            for ps in spec.free
        ])
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

    # Optional guaranteed previous optimum.  Valid only if its free-parameter
    # (block, name) sequence AND fixed-parameter values match the current spec.
    if INCLUDE_PREV_OPTIMUM
        _opt_jls = joinpath(SMM_OUT_DIR, "smm_result_$(WINDOW)$(W_SUFFIX).jls")
        _opt = _load_smm_bundle(_opt_jls; delete_on_fail=false, label="previous optimum")
        if !isnothing(_opt)
            _o_free = [(ps.block, ps.name) for ps in _opt.spec.free]
            _c_free = [(ps.block, ps.name) for ps in spec.free]
            if _o_free == _c_free && _opt.spec.fixed == spec.fixed
                prev_optimum = copy(_opt.result.theta_opt)
                @printf("  [init] INCLUDE_PREV_OPTIMUM: added valid prior optimum (Q=%.6e).\n",
                        _opt.result.loss_opt)
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
# 9. Run estimation
# ============================================================
println("Starting SMM optimisation..."); flush(stdout)

# Stage 1: global search (seed_bank/prev_optimum set in §8b).
res = run_smm(spec; method = :sa, seed_bank = seed_bank, prev_optimum = prev_optimum)

# Stage 2: local polish from the global optimum.
res_pol = run_smm(_spec_with_init(spec, res.theta_opt); method = :neldermead)

results = res_pol

# ============================================================
# 10. Save results
# ============================================================
mkpath(TABLES_DIR)
mkpath(SMM_OUT_DIR)

save_results(results, joinpath(TABLES_DIR, "smm_estimates_$(WINDOW)$(W_SUFFIX).csv"))

# The .jls bundle is consumed by solver/plots.jl, the transition entry point
# (transition/transition_main.jl), and MCMC standard errors (smm/MCMC_main.jl).
smm_jls_path = joinpath(SMM_OUT_DIR, "smm_result_$(WINDOW)$(W_SUFFIX).jls")
open(smm_jls_path, "w") do io
    serialize(io, (result = results, spec = spec, sim = sim_smm))
end
@printf("Serialized SMM result → %s\n", smm_jls_path)

println("\nDone.")
flush(stdout)
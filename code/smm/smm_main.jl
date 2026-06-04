############################################################
# code/smm/smm_main.jl — SMM estimation entry point (v7)
#
# Usage (from project root):
#   julia --threads auto code/smm/smm_main.jl
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
#     derived/  — windows.json, moments_{w}.csv, sigma_{w}.csv,
#                 nu_estimation.csv, phi_calibration.csv,
#                 training_share_scale.csv (κ_w; optional but
#                                            recommended — see v7.1)
#   output/
#     plots/
#     tables/
#     smm/      — serialised SMMResult bundles
#
# Two-stage workflow (Model Notes §2 / data_and_moments.pdf §18):
#   Baseline window  (:base_fc, :base_covid):
#       Fix (r, ν, φ) from external calibration.
#       Estimate all 22 structural + regime parameters.
#       ν is loaded from nu_estimation.csv using the row for the
#       corresponding baseline (one row per crisis pair).
#   Crisis window    (:crisis_fc, :crisis_covid):
#       Load the matching base bundle (base_fc → crisis_fc;
#       base_covid → crisis_covid).
#       Fix the deep structural parameters at the baseline estimates:
#           common  a_ℓ, b_ℓ
#           regime  bU, bT, bS
#       Re-estimate only the 17 regime-specific parameters.
#       ν comes from the baseline row (NOT a crisis-specific value).
#
# v7 changes vs. previous SMM:
#   - 24 moments (train_entry_rate_U added; exp_ur_* dropped).
#   - nu_estimate.csv → nu_estimation.csv (two rows; this script
#     picks the correct row per window).
#   - Crisis windows now consume the pair-matched baseline bundle
#     (crisis_covid uses base_covid, not base_fc).
#   - WINDOWS are read from data/derived/windows.json — single source
#     of truth shared with data_pipeline_v7.
#
# v7.1 changes:
#   - training_share is now NSC-level-rescaled by κ_w (read from
#     derived/training_share_scale.csv) at SMM load time. Applied
#     inside load_data_moments (target value), load_weight_matrix
#     (Σ̂ row/col), and load_sigma_matrix (std-err vector). See the
#     moments.jl header for the full convention. If the κ file is
#     missing, κ defaults to 1.0 with a warning (back-compat).
############################################################

println("="^60)
println("  Segmented Search Model — SMM Estimation")
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
using Clustering
using JSON3

println("done."); flush(stdout)

Random.seed!(1234)

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

println("done."); flush(stdout)
@printf("Threads available: %d\n\n", Threads.nthreads())
flush(stdout)

# ============================================================
# 1. Solver settings
#    Use coarser tolerances than the single-solve script to keep
#    each SMM iteration fast.
# ============================================================
sim_smm = SimParams(
    tol_inner      = 1e-7,
    tol_outer_U    = 1e-6,
    tol_outer_S    = 1e-6,
    tol_global     = 1e-4,

    maxit_inner    = 300,
    maxit_outer    = 200,
    maxit_global   = 20,

    conv_streak    = 1,

    use_anderson   = true,
    anderson_m     = 1,
    anderson_reg   = 1e-10,

    damp_pstar_U   = 1.00,
    damp_pstar_S   = 0.50,

    verbose        = 0,          # 0: model is silent; 1: print outer convergence info per iteration; 2: also print inner iteration details
    verbose_stride = 100,
)

# ============================================================
# 2. Estimation window
#    Valid windows are loaded from data/derived/windows.json
#    written by data_pipeline_v7.
# ============================================================
WINDOW = :base_covid

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
# Moments to exclude from the SMM objective.
# Use the SAME list for both load_weight_matrix and build_smm_spec
# so sigma / W is subsetted consistently with the loss function.
#
# Valid names (24 moments in MOMENT_NAMES order):
#   :ur_total, :ur_U, :ur_S, :skilled_share, :training_share,
#   :emp_var_U, :emp_cm3_U, :emp_var_S, :emp_cm3_S,
#   :jfr_U, :sep_rate_U, :jfr_S, :sep_rate_S,
#   :ee_rate_S, :train_entry_rate_U,
#   :mean_wage_U, :mean_wage_S,
#   :p25_wage_U, :p25_wage_S, :p50_wage_U, :p50_wage_S,
#   :wage_premium, :theta_U, :theta_S
# ============================================================
SKIP_MOMENTS = Symbol[
    # :ur_total,
    # :ur_U,
    # :ur_S,
    # :skilled_share,
    # :training_share,
    # :emp_var_U,
    # :emp_var_S,
    # :emp_cm3_U,
    # :emp_cm3_S,
    # :jfr_U,
    # :sep_rate_U,
    # :jfr_S,
    # :sep_rate_S,
    # :ee_rate_S,
    # :train_entry_rate_U,
    # :mean_wage_U,
    # :mean_wage_S,
    # :p25_wage_U,
    # :p25_wage_S,
    # :p50_wage_U,
    # :p50_wage_S,
    # :wage_premium,
    # :theta_U,
    # :theta_S,
]

@printf("Estimation window: %s\n", WINDOW)
flush(stdout)

# ============================================================
# Parameters to pin at a fixed value during estimation.
# r / ν / φ are always fixed from external calibration and do
# not need to appear here.
#
# Valid keys (ASCII):
#   common:  :a_l  :b_l  :c
#   regime:  :PU  :gamma_PS  :bU  :bT  :bS  :alpha_U  :a_Gam  :b_Gam
#   unsk:    :unsk_mu  :unsk_eta  :unsk_k  :unsk_bet  :unsk_lam
#   skl:     :skl_mu   :skl_eta   :skl_k   :skl_bet
#            :skl_lam  :skl_sig
# ============================================================
FIX_PARAMS = Dict{Symbol,Float64}(
    # :a_l      => 1.01131,
    # :b_l      => 2.42423,
    # :c        => 2.94633,
    # :PU       => 1.05948,
    # :gamma_PS => 3.83639,
    # :bU       => 0.00000,
    # :bT       => 0.35082,
    # :bS       => 0.56935,
    # :alpha_U  => 4.80594,
    # :a_Gam    => 4.77377,
    # :b_Gam    => 2.28169,
    # :unsk_mu  => 0.25585,
    # :unsk_eta => 0.50000,
    # :unsk_k   => 0.10061,
    # :unsk_bet => 0.50000,
    # :unsk_lam => 0.20263,
    # :skl_mu   => 0.22462,
    # :skl_eta  => 0.50000,
    # :skl_k    => 0.03317,
    # :skl_bet  => 0.50000,
    # :skl_lam  => 0.17788,
    # :skl_sig  => 1.10000,
)

# ============================================================
# USE_DEFAULT_PARAMS — when true, seed the optimiser from the
#   hard-coded values below and ignore any prior run.
# ============================================================
USE_DEFAULT_PARAMS = true

const DEFAULT_PARAMS = Dict{Symbol,Float64}(
    :r        => 0.00416667,
    :nu       => 0.00323032,
    :phi      => 0.02222129,
    :a_l      => 0.88876000,
    :b_l      => 0.30088000,
    :c        => 7.05302000,
    :PU       => 2.40573000,
    :gamma_PS => 3.81189000,
    :bU       => 0.03062000,
    :bT       => 1.59653000,
    :bS       => 0.01848000,
    :alpha_U  => 1.00174000,
    :a_Gam    => 0.30219000,
    :b_Gam    => 2.36812000,
    :unsk_mu  => 0.06382000,
    :unsk_eta => 0.62036000,
    :unsk_k   => 0.07349000,
    :unsk_bet => 0.86951000,
    :unsk_lam => 0.31930000,
    :skl_mu   => 0.47164000,
    :skl_eta  => 0.89611000,
    :skl_k    => 0.06919000,
    :skl_bet  => 0.85555000,
    :skl_lam  => 0.06319000,
    :skl_sig  => 0.00736000,
)

# (block, unicode name) → DEFAULT_PARAMS key (ASCII).
const _DEFAULT_PARAM_KEY = Dict{Tuple{Symbol,Symbol}, Symbol}(
    (:common, :a_ℓ) => :a_l,     (:common, :b_ℓ)  => :b_l,     (:common, :c)   => :c,
    (:regime, :PU)  => :PU,      (:regime, :gamma_PS) => :gamma_PS,
    (:regime, :bU)  => :bU,      (:regime, :bT)   => :bT,      (:regime, :bS)  => :bS,
    (:regime, :α_U) => :alpha_U, (:regime, :a_Γ)  => :a_Gam,  (:regime, :b_Γ) => :b_Gam,
    (:unsk,   :μ)   => :unsk_mu, (:unsk,   :η)    => :unsk_eta, (:unsk,  :k)   => :unsk_k,
    (:unsk,   :β)   => :unsk_bet, (:unsk,  :λ)   => :unsk_lam,
    (:skl,    :μ)   => :skl_mu,  (:skl,    :η)    => :skl_eta,  (:skl,   :k)   => :skl_k,
    (:skl,    :β)   => :skl_bet, (:skl,    :λ)   => :skl_lam,
    (:skl,    :σ)   => :skl_sig,
)

# ASCII key (FIX_PARAMS convention) → unicode fixed-NamedTuple key.
const _ASCII_TO_FIXED_KEY = Dict{Symbol, Symbol}(
    :r        => :r,
    :nu       => :ν,
    :phi      => :φ,
    :a_l      => :a_ℓ,
    :b_l      => :b_ℓ,
    :c        => :c,
    :PU       => :PU,
    :gamma_PS => :gamma_PS,
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
    :skl_mu   => :skl_μ,
    :skl_eta  => :skl_η,
    :skl_k    => :skl_k,
    :skl_bet  => :skl_β,
    :skl_lam  => :skl_λ,
    :skl_sig  => :skl_σ,
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

# Report κ_w for this window so it is visible in the run log.
_κ_ts = load_training_share_scale(; window=WINDOW, derived_dir=derived_dir)
@printf("  training_share κ_%s = %.4f  (from training_share_scale.csv)\n", WINDOW, _κ_ts)
flush(stdout)


# ============================================================
# 4. Weight matrix
#    W_COND_TARGET controls which weighting scheme is used:
#      0.0   diagonal from sigma_{window}.csv
#      1.0   compressed diagonal:  w = log(1 + 1/σ²)
#      2.0   equal weights (identity, no W matrix)
#      >2.0  full optimal W (shrunk if κ > target)
# ============================================================
W_COND_TARGET = 2.0 

function _w_suffix(cond_target::Float64)
    cond_target == 0.0 && return "_diagonalW"
    cond_target == 1.0 && return "_compressedW"
    cond_target == 2.0 && return "_equalW"
    return "_fullW"
end

W_SUFFIX = _w_suffix(W_COND_TARGET)

W_opt = load_weight_matrix(; window=WINDOW, derived_dir=derived_dir,
                             cond_target=W_COND_TARGET,
                             skip_moments=SKIP_MOMENTS)


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
#    Parameter classification (Model Notes §2):
#
#    Externally calibrated / pre-estimated (always fixed):
#      common:  r, ν, φ
#
#    Deep structural (estimated on the baseline of each pair,
#    then fixed in the crisis re-estimation):
#      common:  a_ℓ, b_ℓ
#      regime:  bU, bT, bS
#
#    Regime-specific (re-estimated within each window):
#      common:  c
#      regime:  PU, gamma_PS, α_U, a_Γ, b_Γ
#      unsk:    μ, η, k, β, λ
#      skl:     μ, η, k, β, λ, σ
# ============================================================

"""
    _baseline_param_value(ps::ParamSpec, cp, rp, up, sp) → Float64
"""
function _baseline_param_value(ps::ParamSpec, cp, rp, up, sp) :: Float64
    if     ps.block == :common; return Float64(getfield(cp, ps.name))
    elseif ps.block == :regime; return Float64(getfield(rp, ps.name))
    elseif ps.block == :unsk;   return Float64(getfield(up, ps.name))
    else                        return Float64(getfield(sp, ps.name))
    end
end

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

    cp_base, rp_base, up_base, sp_base =
        unpack_θ(baseline_result.theta_opt, baseline_spec)

    @printf("  Baseline Q = %.6e  (converged = %s)\n",
            baseline_result.loss_opt, baseline_result.converged)
    flush(stdout)

    # Fixed: external calibration + deep structural from baseline.
    # FIX_PARAMS can additionally pin regime-specific parameters;
    # baseline-derived deep values take priority over FIX_PARAMS.
    _extra_fixed = _fix_params_to_nt(FIX_PARAMS)
    fixed_params = merge(
        _extra_fixed,
        (
        r   = calib.r,
        ν   = calib.nu,
        φ   = calib.phi,
        a_ℓ = cp_base.a_ℓ,
        b_ℓ = cp_base.b_ℓ,
        bU  = rp_base.bU,
        bT  = rp_base.bT,
        bS  = rp_base.bS,
    ))

    # Free parameters: regime-specific only, initialised from baseline.
    free_params = ParamSpec[]
    for ps in default_free_params()
        (ps.block, ps.name) in REGIME_SPECIFIC_PARAMS || continue
        init_val = _baseline_param_value(ps, cp_base, rp_base, up_base, sp_base)
        init_val = clamp(init_val, ps.lb + 1e-10, ps.ub - 1e-10)
        push!(free_params,
              ParamSpec(ps.block, ps.name, ps.lb, ps.ub, init_val, ps.label))
    end

    println("\n  Deep structural parameters FIXED from baseline:")
    @printf("    common:  a_ℓ=%.4f  b_ℓ=%.4f\n",
            cp_base.a_ℓ, cp_base.b_ℓ)
    @printf("    regime:  bU=%.4f  bT=%.4f  bS=%.4f\n",
            rp_base.bU, rp_base.bT, rp_base.bS)
    @printf("  Regime-specific parameters FREE (%d params)\n", length(free_params))
    flush(stdout)

else
    # Baseline estimation.  Fix r / ν / φ plus anything in FIX_PARAMS.
    _extra_fixed = _fix_params_to_nt(FIX_PARAMS)
    fixed_params = merge(_extra_fixed, (
        r = calib.r,
        ν = calib.nu,
        φ = calib.phi,
    ))

    free_params = default_free_params()

    # Warm start: load only parameter values from a prior run.
    _warmstart_jls = joinpath(SMM_OUT_DIR, "smm_result_$(WINDOW)$(W_SUFFIX).jls")
    if USE_DEFAULT_PARAMS
        println("\n  USE_DEFAULT_PARAMS = true — seeding free parameters from DEFAULT_PARAMS.")
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

    elseif isfile(_warmstart_jls)
        println("\n  Warm-start: loading prior parameter values from:")
        @printf("    %s\n", _warmstart_jls)
        flush(stdout)

        _ws_data = _load_smm_bundle(_warmstart_jls; delete_on_fail=false, label="warm-start file")

        if !isnothing(_ws_data)
            _ws_vals = Dict{Tuple{Symbol,Symbol}, Float64}()
            try
                _ws_result = _ws_data.result
                _ws_spec   = _ws_data.spec
                _ws_cp, _ws_rp, _ws_up, _ws_sp = unpack_θ(_ws_result.theta_opt, _ws_spec)
                for ps in _ws_spec.free
                    has_field = try
                        _baseline_param_value(ps, _ws_cp, _ws_rp, _ws_up, _ws_sp)
                        true
                    catch
                        false
                    end
                    has_field && (_ws_vals[(ps.block, ps.name)] =
                                  _baseline_param_value(ps, _ws_cp, _ws_rp, _ws_up, _ws_sp))
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
        println("\n  No prior result found — using default initial values.")
    end
    flush(stdout)

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

    # ── Weight-matrix mode (passed to print_spec for the header) ──
    # See load_weight_matrix for the cond_target semantics
    # (0=diag, 1=compressed diag, 2=identity, >2=full optimal W).
    w_cond_target = W_COND_TARGET,

    # ── Simulated annealing ──────────────────────────────────
    sa_max_iter        = 5_000,   # max SA iterations
    sa_T0              = 20.00,   # initial temperature (≤0 ⇒ auto-calibrate
                                  # from uphill probes; here pinned to 20)
    sa_step            = 0.30,    # initial proposal sd in unconstrained space
    sa_cooling_rate    = 1.0,     # rate in T(t) = T_reheat / log(1+rate·t)^exp
    sa_cooling_exp     = 1.0,     # exponent in same schedule (higher = faster cool)
    sa_reheat_patience = 50,      # steps without improvement before a reheat
    sa_reheat_factor   = 2.00,    # multiplicative reheat: T ← T · factor
    sa_max_reheats     = 30,      # cap on number of reheats per run
    sa_adapt_window    = 50,      # rolling window for adaptive step / acceptance
    sa_target_fin      = 0.90,    # target feasibility (finite-Q) fraction;
                                  # below this, step shrinks
    sa_random_init     = false,   # false ⇒ start from pack_θ(spec); true ⇒
                                  # uniform draw inside [lb, ub]

    # ── Differential evolution ───────────────────────────────
    de_max_iter  = 3_000,         # max generations
    de_pop_size  = 120,           # population size (0 ⇒ 10·n_free_params)
    de_f         = 0.70,          # DE differential weight (mutation strength)
    de_cr        = 0.85,          # DE crossover probability
    de_patience  = 2,             # stop after this many generations with no
                                  # improvement
    de_avg_tol   = 0.0,           # stop when (Q_mean - Q_best)/|Q_best| < tol;
                                  # 0 disables this early-stop

    # ── Nelder-Mead local polish ─────────────────────────────
    nm_max_iter  = 2_000,         # max NM iterations
    nm_f_tol     = 1e-6,          # function-value tolerance
    nm_x_tol     = 1e-4,          # parameter tolerance (unconstrained space)

    # ── Tracing ──────────────────────────────────────────────
    show_trace_members     = false,   # per-member trace inside one DE/SA gen
    show_trace_generations = true,    # per-generation / per-iteration summary
    trace_stride           = 100,     # print every N iterations in SA 
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

# Stage 1: global search
res = run_smm(spec; method = :sa)

# Stage 2: local polish from the global optimum
res_pol = run_smm(_spec_with_init(spec, res.theta_opt); method = :neldermead)

results = res_pol

# ============================================================
# 10. Save results
# ============================================================
mkpath(TABLES_DIR)
mkpath(SMM_OUT_DIR)

save_results(results, joinpath(TABLES_DIR, "smm_estimates_$(WINDOW)$(W_SUFFIX).csv"))

# The .jls bundle is consumed by code/solver/plots_main.jl and any
# transition-dynamics post-processor.
smm_jls_path = joinpath(SMM_OUT_DIR, "smm_result_$(WINDOW)$(W_SUFFIX).jls")
open(smm_jls_path, "w") do io
    serialize(io, (result = results, spec = spec, sim = sim_smm))
end
@printf("Serialized SMM result → %s\n", smm_jls_path)

println("\nDone.")
flush(stdout)
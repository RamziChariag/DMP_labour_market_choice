############################################################
# code/smm/smm_main_temp.jl — TEMPORARY uniform-ℓ check driver
# [SCAFFOLD:TEMP-MAIN] — delete this file once the D1 checks are done.
# smm_main.jl is untouched and remains the pipeline of record.
#
# Self-contained variant of smm_main.jl that consolidates the checks:
#   1. WIDENED BOUNDS applied locally (_TEMP_BOUNDS below); smm_params.jl
#      is NOT modified.  NOTE: if the optimum lands in the widened region,
#      the same boxes must go into default_free_params() before the
#      official smm_main.jl re-run (else its warm start clamps at the old
#      bounds and Nelder–Mead grinds on a box edge).
#   2. PROBE MODE:
#        SMM_PROBE=1 julia --threads auto code/smm/smm_main_temp.jl
#      solves 6 hand-picked interior-cutoff points at the full grid,
#      prints feasibility (with reasons), x̄, masses, key moments vs
#      targets, then exits.  Run this FIRST.
#   3. ESTIMATION MODE (default):
#        julia --threads auto code/smm/smm_main_temp.jl
#      :clusters Sobol bank (separate *_temp cache) + injected interior
#      seeds + previous optimum re-encoded  →  DE first (de_patience=15)
#      →  NM polish  →  D1 report (cutoff, shares, premium, corner tags).
#   4. OUTPUTS:  bundle → the STANDARD path (so smm_main.jl can warm-start
#      from it), after backing up the existing Beta-ℓ bundle ONCE to
#      *_beta_backup.jls (keep it: it is the appendix column).
#      Tables CSV and candidate cache carry a _temp suffix.
#
# Baseline windows only (no crisis branch).
############################################################

println("="^60)
println("  Segmented Search Model — SMM  [TEMP uniform-ℓ check driver]")
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
using QuasiMonteCarlo
using JSON3
println("done."); flush(stdout)

Random.seed!(1234)

print("Loading solver modules... "); flush(stdout)
include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))
println("done."); flush(stdout)

print("Loading SMM modules... "); flush(stdout)
include(joinpath(SMM_DIR, "moments.jl"))
include(joinpath(SMM_DIR, "smm_params.jl"))
include(joinpath(SMM_DIR, "smm.jl"))
include(joinpath(SMM_DIR, "candidates.jl"))
println("done."); flush(stdout)
@printf("Threads available: %d\n\n", Threads.nthreads())
flush(stdout)

# ============================================================
# 1. Solver settings (identical to smm_main.jl)
# ============================================================
sim_smm = SimParams(
    tol_inner          = 1e-7,
    tol_outer_U        = 1e-6,
    tol_outer_S        = 1e-6,
    tol_global         = 1e-4,
    damp_inner_U       = 0.95,
    damp_inner_S       = 0.95,
    inner_B            = 20,
    inner_K            = 10,
    outer_B            = 30,
    outer_K            = 10,
    maxit_inner        = 300,
    maxit_outer        = 200,
    maxit_global       = 20,
    conv_streak        = 1,
    use_anderson       = true,
    anderson_m         = 1,
    anderson_reg       = 1e-10,
    damp_pstar_U       = 1.00,
    damp_pstar_S       = 0.50,
    verbose            = 0,
    verbose_stride     = 100,
)

# ============================================================
# 2. Window / skips / calibration knob / pins  (as smm_main.jl)
# ============================================================
WINDOW = :base_covid
WINDOW in (:base_fc, :base_covid) ||
    error("smm_main_temp.jl supports baseline windows only (got :$WINDOW).")

SKIP_MOMENTS = Symbol[
    :ur_total,
]

const LAMBDA_W = 0.82

FIX_PARAMS = Dict{Symbol,Float64}(
     :a_l      => 1.00000,
     :b_l      => 1.00000,
     :unsk_eta => 0.50000,
     :unsk_bet => 0.50000,
     :skl_eta  => 0.50000,
     :skl_bet  => 0.50000,
)

const DEFAULT_PARAMS = Dict{Symbol,Float64}(
    :r        => 0.00416667,
    :nu       => 0.00336,
    :phi      => 0.02222129,
    :a_l      => 2.59266,
    :b_l      => 1.31532,
    :c        => 10.58040,
    :A        => 4.83051,
    :PU       => 6.53357,
    :gamma_PS => 9.72094,
    :bU       => 1.38551,
    :bT       => 3.81919,
    :bS       => 0.84956,
    :alpha_U  => 1.85715,
    :a_Gam    => 7.50429,
    :b_Gam    => 6.96083,
    :unsk_mu  => 0.32789,
    :unsk_eta => 0.50000,
    :unsk_k   => 1.13394,
    :unsk_bet => 0.50000,
    :unsk_lam => 0.37870,
    :unsk_sigw => 0.23461,
    :skl_mu   => 0.25346,
    :skl_eta  => 0.50000,
    :skl_k    => 2.28365,
    :skl_bet  => 0.50000,
    :skl_lam  => 0.13672,
    :skl_sig  => 0.28564,
    :skl_sigw  => 0.21904,
)

const _DEFAULT_PARAM_KEY = Dict{Tuple{Symbol,Symbol}, Symbol}(
    (:common, :a_ℓ) => :a_l,     (:common, :b_ℓ)  => :b_l,     (:common, :c)   => :c,
    (:common, :A)   => :A,
    (:unsk,   :PU)  => :PU,      (:skl,    :gamma_PS) => :gamma_PS,
    (:unsk,   :bU)  => :bU,      (:unsk,   :bT)   => :bT,      (:skl,    :bS)  => :bS,
    (:unsk,   :α_U) => :alpha_U, (:skl,    :a_Γ)  => :a_Gam,  (:skl,    :b_Γ) => :b_Gam,
    (:unsk,   :μ)   => :unsk_mu, (:unsk,   :η)    => :unsk_eta, (:unsk,  :k)   => :unsk_k,
    (:unsk,   :β)   => :unsk_bet, (:unsk,  :λ)   => :unsk_lam,  (:unsk, :σ_w)  => :unsk_sigw,
    (:skl,    :μ)   => :skl_mu,  (:skl,    :η)    => :skl_eta,  (:skl,   :k)   => :skl_k,
    (:skl,    :β)   => :skl_bet, (:skl,    :λ)   => :skl_lam,
    (:skl,    :σ)   => :skl_sig, (:skl,    :σ_w)  => :skl_sigw,
)

const _ASCII_TO_FIXED_KEY = Dict{Symbol, Symbol}(
    :r        => :r,
    :nu       => :ν,
    :phi      => :φ,
    :a_l      => :a_ℓ,
    :b_l      => :b_ℓ,
    :c        => :c,
    :A        => :A,
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
    :unsk_sigw => :unsk_σ_w,
    :skl_mu   => :skl_μ,
    :skl_eta  => :skl_η,
    :skl_k    => :skl_k,
    :skl_bet  => :skl_β,
    :skl_lam  => :skl_λ,
    :skl_sig  => :skl_σ,
    :skl_sigw  => :skl_σ_w,
)

function _fix_params_to_nt(fix_dict::Dict{Symbol,Float64}) :: NamedTuple
    isempty(fix_dict) && return (;)
    keys_vec = Symbol[]
    vals_vec = Float64[]
    for (ascii_key, val) in fix_dict
        if haskey(_ASCII_TO_FIXED_KEY, ascii_key)
            push!(keys_vec, _ASCII_TO_FIXED_KEY[ascii_key])
            push!(vals_vec, val)
        else
            @printf("WARNING: FIX_PARAMS — unrecognised key :%s — ignored.\n", ascii_key)
        end
    end
    isempty(keys_vec) && return (;)
    return NamedTuple{Tuple(keys_vec)}(Tuple(vals_vec))
end

# ============================================================
# 3. Windows + data moments + σ_w calibration  (as smm_main.jl)
# ============================================================
derived_dir = joinpath(PROJECT_ROOT, "data", "derived")

_win_info = load_windows(; derived_dir=derived_dir)
WINDOW in keys(_win_info.windows) ||
    error("WINDOW = :$WINDOW not found in windows.json.")
_wd = _win_info.windows[WINDOW]
@printf("  Window definition (windows.json):  %s  ym = %d..%d  ASEC %d..%d\n",
        _wd.label, _wd.ym_start, _wd.ym_end,
        first(_wd.asec_years), last(_wd.asec_years))
flush(stdout)

moments = load_data_moments(; window=WINDOW, derived_dir=derived_dir)

σ_wU_cal, σ_wS_cal = calibrate_sigma_w(LAMBDA_W, moments)
@printf("  Calibrated σ_w (λ_w = %.4f):  σ_wU = %.5f,  σ_wS = %.5f\n",
        LAMBDA_W, σ_wU_cal, σ_wS_cal)
_sigw_fixed = (unsk_σ_w = σ_wU_cal, skl_σ_w = σ_wS_cal)
flush(stdout)

# ============================================================
# 4. Weight matrix — equal weights (as the current runs)
# ============================================================
W_COND_TARGET = 2.0
W_SUFFIX      = "_equalW"
W_opt = load_weight_matrix(; window=WINDOW, derived_dir=derived_dir,
                             cond_target=W_COND_TARGET,
                             skip_moments=SKIP_MOMENTS)
Q_SCALE = 1.0

# ============================================================
# 5. Externally calibrated parameters
# ============================================================
calib = load_calibrated_params(; window=WINDOW, derived_dir=derived_dir)
@printf("  Calibrated:  r = %.6f,  ν = %.5f,  φ = %.5f\n",
        calib.r, calib.nu, calib.phi)
flush(stdout)

# ============================================================
# 6. Fixed / free parameters (baseline branch; DEFAULT_PARAMS inits)
# ============================================================
_extra_fixed = _fix_params_to_nt(FIX_PARAMS)
fixed_params = merge((
    r = calib.r,
    ν = calib.nu,
    φ = calib.phi,
    ),
    _extra_fixed,
    _sigw_fixed,
)

free_params = [
    let ascii_key = get(_DEFAULT_PARAM_KEY, (ps.block, ps.name), nothing),
        raw_val   = isnothing(ascii_key) ? ps.init :
                        get(DEFAULT_PARAMS, ascii_key, ps.init),
        init_val  = clamp(raw_val, ps.lb + 1e-10, ps.ub - 1e-10)
        ParamSpec(ps.block, ps.name, ps.lb, ps.ub, init_val, ps.label)
    end
    for ps in default_free_params()
]

# ── [TEMP] widened bounds, applied locally (smm_params.jl untouched) ──
const _TEMP_BOUNDS = Dict{Tuple{Symbol,Symbol},Tuple{Float64,Float64}}(
    (:common, :c)        => (3.0,   14.5),   # SA tagged c(upper)
    (:unsk,   :bU)       => (0.05,  2.5),    # viability b_U < P_U·x once P_U ~ 1-2
    (:unsk,   :bT)       => (0.05,  7.0),    # stipend can sit below unskilled output
    (:skl,    :bS)       => (0.02,  2.0),
    (:unsk,   :PU)       => (0.40,  12.0),
    (:skl,    :gamma_PS) => (1.0,   25.0),
    (:unsk,   :α_U)      => (0.20,  20.5),
    (:unsk,   :μ)        => (0.02,  3.9),    # SA corner
    (:skl,    :μ)        => (0.01,  4.5),
    (:unsk,   :λ)        => (0.02,  1.5),    # SA corner
    (:skl,    :λ)        => (0.005, 0.5),    # SA corner
    (:unsk,   :k)        => (0.02,  6.0),
    (:skl,    :k)        => (0.02,  8.0),
    (:skl,    :σ)        => (0.0,   1.5),
)
free_params = [
    let bb   = get(_TEMP_BOUNDS, (ps.block, ps.name), (ps.lb, ps.ub)),
        init = clamp(ps.init, bb[1] + 1e-10, bb[2] - 1e-10)
        ParamSpec(ps.block, ps.name, bb[1], bb[2], init, ps.label)
    end
    for ps in free_params
]
println("  [TEMP] widened bounds applied to $(length(_TEMP_BOUNDS)) parameters.")

@printf("  Fixed params:  %d  |  Free params:  %d\n",
        length(fixed_params), length(free_params))
flush(stdout)

# ============================================================
# 7. Run parameters — DE-first settings
# ============================================================
run_params = SMMRunParams(
    Nx      = 120,
    Np_U    = 120,
    Np_S    = 120,

    cand_Nx          = 40,
    cand_Np_U        = 40,
    cand_Np_S        = 40,
    cand_n_sample    = 2048,
    cand_seed        = 42,
    cand_min_cluster = 5,

    w_cond_target = W_COND_TARGET,
    λ_w           = LAMBDA_W,

    sa_max_iter        = 5_000,
    sa_T0              = 10.0,
    sa_step            = 0.20,
    sa_cooling_rate    = 0.5,
    sa_cooling_exp     = 1.0,
    sa_reheat_patience = 1_000,
    sa_reheat_factor   = 4.00,
    sa_max_reheats     = 1,
    sa_adapt_window    = 50,
    sa_target_fin      = 0.90,
    sa_random_init     = false,
    sa_parallel_steps  = 100,
    sa_seed            = 42,

    de_max_iter  = 2_000,
    de_pop_size  = 120,
    de_f         = 0.70,
    de_cr        = 0.85,
    de_patience  = 15,            # [TEMP] was 2 — too trigger-happy
    de_avg_tol   = 1e-6,

    nm_max_iter  = 5_000,
    nm_f_tol     = 1e-6,
    nm_x_tol     = 1e-4,
    nm_g_tol     = 1e-5,
    nm_no_improve = 1_300,

    show_trace_members     = false,
    show_trace_generations = true,
    trace_stride           = 100,
)

# ============================================================
# 8. Build spec
# ============================================================
spec = build_smm_spec(
    moments, sim_smm;
    fixed        = fixed_params,
    free_specs   = free_params,
    run          = run_params,
    W            = W_opt,
    q_scale      = Q_SCALE,
    skip_moments = SKIP_MOMENTS,
)

print_spec(spec)

# ============================================================
# Shared helper: solve at θ and print a diagnostic report.
# Returns (feasible::Bool, Q::Float64).
# ============================================================
const _REPORT_MOMENTS = (:skilled_share, :training_share, :wage_premium,
                         :ur_U, :ur_S, :jfr_U, :jfr_S, :sep_rate_U, :sep_rate_S,
                         :theta_U, :theta_S, :mean_wage_U, :mean_wage_S,
                         :emp_var_U, :emp_var_S, :ee_rate_S, :ltu_share_S)

function _solve_and_report(θ::Vector{Float64}, spec; label::String = "")
    isempty(label) || println("\n── $label " * "─"^30)
    cph, uph, sph = unpack_θ(θ, spec)
    local model, sres
    try
        model, sres = solve_model(cph, uph, sph, spec.sim;
                                  Nx = spec.run.Nx, Np_U = spec.run.Np_U,
                                  Np_S = spec.run.Np_S)
    catch err
        println("  SOLVER THREW: ", sprint(showerror, err)); return (false, Inf)
    end
    sres.ok || (println("  solve_result.ok == false (no convergence)"); return (false, Inf))

    local obj, mm
    try
        obj = compute_equilibrium_objects(model)
        mm  = model_moments(obj)
    catch err
        println("  MOMENTS THREW: ", sprint(showerror, err)); return (false, Inf)
    end

    τv = Float64.(vec(obj.tauT))
    xg = obj.xg
    xbar = all(iszero, τv) ? NaN :
           all(isone, τv)  ? 0.0 : xg[findfirst(==(1.0), τv)]
    _dead = obj.agg_eU < 1e-12 || obj.agg_eS < 1e-12
    _degτ = all(iszero, τv) || all(isone, τv) ||
            any(diff(τv) .< 0) || count(!iszero, diff(τv)) > 1

    @printf("  masses:  eU=%.4f  eS=%.4f  uU=%.4f  uS=%.4f  t=%.4f\n",
            obj.agg_eU, obj.agg_eS, obj.agg_uU, obj.agg_uS, obj.agg_t)
    @printf("  x̄ (training cutoff) = %s   τ degenerate: %s   dead market: %s\n",
            isnan(xbar) ? "none train" : @sprintf("%.3f", xbar), _degτ, _dead)

    feas = !_dead && !_degτ
    Qv   = feas ? compute_loss_matrix(mm, spec, spec.W) : Inf
    @printf("  Q = %s\n", isfinite(Qv) ? @sprintf("%.4e", Qv) : "Inf (infeasible)")

    println("  moment            model      target      %dev")
    for k in _REPORT_MOMENTS
        haskey(spec.moments, k) || continue
        hasproperty(mm, k)      || continue
        mv = getproperty(mm, k); tv = spec.moments[k].value
        @printf("  %-16s %9.4f  %9.4f  %+8.1f%%\n", k, mv, tv,
                100 * (mv - tv) / (abs(tv) < 1e-10 ? 1.0 : abs(tv)))
    end

    # Corner tags (within 2% of the constrained range of either bound)
    tags = String[]
    for (k, ps) in enumerate(spec.free)
        x   = _to_constrained(θ[k], ps.lb, ps.ub)
        tol = 0.02 * (ps.ub - ps.lb)
        x - ps.lb < tol && push!(tags, "$(ps.block):$(ps.name)(lower)")
        ps.ub - x < tol && push!(tags, "$(ps.block):$(ps.name)(upper)")
    end
    @printf("  corners: %s\n", isempty(tags) ? "none" : join(tags, ", "))
    return (feas, Qv)
end

# Build a θ vector from (block,name)→value overrides on top of base values.
const _OVERRIDE_KEY = Dict(:gamma_PS => (:skl, :gamma_PS), :PU => (:unsk, :PU),
                           :c => (:common, :c), :bU => (:unsk, :bU),
                           :bT => (:unsk, :bT), :bS => (:skl, :bS))

function _theta_from(base::Dict{Tuple{Symbol,Symbol},Float64}, ov, spec)
    vals = copy(base)
    for (nm, v) in pairs(ov)
        vals[_OVERRIDE_KEY[nm]] = v
    end
    θ = Vector{Float64}(undef, length(spec.free))
    for (k, ps) in enumerate(spec.free)
        x    = clamp(get(vals, (ps.block, ps.name), ps.init),
                     ps.lb + 1e-8, ps.ub - 1e-8)
        θ[k] = _to_unconstrained(x, ps.lb, ps.ub)
    end
    return θ
end

_spec_base_vals = Dict{Tuple{Symbol,Symbol},Float64}(
    (ps.block, ps.name) => ps.init for ps in spec.free)

# ============================================================
# PROBE MODE  (SMM_PROBE=1) — run this first, then exit.
# ============================================================
if get(ENV, "SMM_PROBE", "0") == "1"
    println("\n", "="^60, "\n  INTERIOR PROBE  (full grid: Nx=$(spec.run.Nx))\n", "="^60)
    _probe_points = [
        (gamma_PS=2.5, PU=1.3, c=7.0, bU=0.30, bT=0.60, bS=0.50),
        (gamma_PS=2.0, PU=1.3, c=6.0, bU=0.30, bT=0.60, bS=0.50),
        (gamma_PS=3.0, PU=1.3, c=8.0, bU=0.30, bT=0.60, bS=0.50),
        (gamma_PS=2.5, PU=0.8, c=6.5, bU=0.15, bT=0.35, bS=0.35),
        (gamma_PS=2.5, PU=2.0, c=8.0, bU=0.40, bT=0.80, bS=0.60),
        (gamma_PS=4.0, PU=1.5, c=9.0, bU=0.30, bT=0.60, bS=0.50),
    ]
    for (i, ov) in enumerate(_probe_points)
        _solve_and_report(_theta_from(_spec_base_vals, ov, spec), spec;
                          label = "probe $i: $(ov)")
    end
    println("\nProbe done — exiting (temp driver; no estimation run).")
    exit(0)
end

# ============================================================
# 8b. Candidate bank (temp cache) + interior seed injection
# ============================================================
cand_path = joinpath(SMM_OUT_DIR, "candidates_$(WINDOW)$(W_SUFFIX)_temp.jls")
seed_bank = load_or_generate_candidates(spec, cand_path;
                                        window      = WINDOW,
                                        force_regen = false,
                                        show_trace  = true)
if isempty(seed_bank.candidates)
    @warn "Candidate bank is empty — DE will fall back to the spec's init point."
    seed_bank = nothing
end
prev_optimum = nothing   # the previous optimum enters via the seed block below

if seed_bank !== nothing
    # Companion values: previous optimum bundle if readable (decoded with the
    # bundle's OWN spec, so old-bounds encodings stay valid), else spec inits.
    _companion = copy(_spec_base_vals)
    let _jls = joinpath(SMM_OUT_DIR, "smm_result_$(WINDOW)$(W_SUFFIX).jls")
        _b = isfile(_jls) ? _load_smm_bundle(_jls; delete_on_fail=false,
                                             label="seed companion source") : nothing
        if !isnothing(_b)
            _bcp, _bup, _bsp = unpack_θ(_b.result.theta_opt, _b.spec)
            for ps in spec.free
                _src = ps.block == :common ? _bcp : ps.block == :unsk ? _bup : _bsp
                if hasproperty(_src, ps.name)
                    _companion[(ps.block, ps.name)] =
                        clamp(Float64(getfield(_src, ps.name)),
                              ps.lb + 1e-8, ps.ub - 1e-8)
                end
            end
            @printf("  [interior-seeds] companions from previous optimum (Q=%.4e).\n",
                    _b.result.loss_opt)
        else
            println("  [interior-seeds] no previous optimum — companions from spec inits.")
        end
    end

    _seed_overrides = NamedTuple[]
    for γ in (1.5, 2.0, 2.5, 3.0, 4.0), PU in (0.8, 1.3, 2.0), c in (6.0, 7.5, 9.0)
        push!(_seed_overrides, (gamma_PS=γ, PU=PU, c=c, bU=0.30, bT=0.60, bS=0.50))
    end
    for γ in (2.0, 2.5, 3.0), PU in (0.8, 1.3), c in (6.5, 8.0)
        push!(_seed_overrides, (gamma_PS=γ, PU=PU, c=c, bU=0.15, bT=0.35, bS=0.35))
    end

    _seed_thetas = [_theta_from(_companion, ov, spec) for ov in _seed_overrides]
    push!(_seed_thetas, _theta_from(_companion, NamedTuple(), spec))  # prev optimum as-is

    @printf("  [interior-seeds] evaluating %d seeds on coarse grid...\n",
            length(_seed_thetas)); flush(stdout)
    _seed_Q = fill(Inf, length(_seed_thetas))
    Threads.@threads for j in eachindex(_seed_thetas)
        _seed_Q[j] = smm_objective(_seed_thetas[j], spec;
                                   Nx   = spec.run.cand_Nx,
                                   Np_U = spec.run.cand_Np_U,
                                   Np_S = spec.run.cand_Np_S)
    end
    _ok = findall(isfinite, _seed_Q)
    @printf("  [interior-seeds] feasible: %d / %d   (best interior Q = %s)\n",
            length(_ok), length(_seed_thetas),
            isempty(_ok) ? "—" : @sprintf("%.4e", minimum(_seed_Q[_ok])))
    if isempty(_ok)
        @warn "[interior-seeds] NO interior seed is feasible — run SMM_PROBE=1 and " *
              "inspect why (dead U-market? degenerate τ?) before trusting any verdict."
    else
        _newlab = isempty(seed_bank.labels) ? 1 : maximum(seed_bank.labels) + 1
        seed_bank = SeedBank(
            vcat(seed_bank.candidates, _seed_thetas[_ok]),
            vcat(seed_bank.Q,          _seed_Q[_ok]),
            vcat(seed_bank.labels,     fill(_newlab, length(_ok))),
            seed_bank.meta)
        for j in _ok
            ov = j <= length(_seed_overrides) ? _seed_overrides[j] : "prev-optimum"
            @printf("    seed %-52s  Q = %.4e\n", string(ov), _seed_Q[j])
        end
    end
    flush(stdout)
end

# ============================================================
# 9. Run estimation:  DE (seeded)  →  NM polish
# ============================================================
println("Starting SMM optimisation (TEMP: DE-first)..."); flush(stdout)

res = run_smm(spec; method = :de, seed_bank = seed_bank, prev_optimum = prev_optimum)

res_pol = run_smm(_spec_with_init(spec, res.theta_opt); method = :neldermead)

results = res_pol

# ============================================================
# 10. Save results
#     Bundle → STANDARD path (backing up the Beta-ℓ bundle once);
#     tables CSV → _temp suffix; the temp candidate cache stays _temp.
# ============================================================
mkpath(TABLES_DIR)
mkpath(SMM_OUT_DIR)

save_results(results, joinpath(TABLES_DIR, "smm_estimates_$(WINDOW)$(W_SUFFIX)_temp.csv"))

_bundle_path = joinpath(SMM_OUT_DIR, "smm_result_$(WINDOW)$(W_SUFFIX).jls")
_backup_path = joinpath(SMM_OUT_DIR, "smm_result_$(WINDOW)$(W_SUFFIX)_beta_backup.jls")
if isfile(_bundle_path) && !isfile(_backup_path)
    Base.cp(_bundle_path, _backup_path)
    @printf("Backed up existing bundle (Beta-ℓ optimum — keep for the appendix) →\n  %s\n",
            _backup_path)
end
open(_bundle_path, "w") do io
    serialize(io, (result = results, spec = spec, sim = sim_smm))
end
@printf("Serialized SMM result → %s\n", _bundle_path)

# ============================================================
# 11. D1 report at the polished optimum (full grid)
# ============================================================
println("\n", "="^60, "\n  D1 REPORT — polished optimum\n", "="^60)
_solve_and_report(results.theta_opt, spec; label = "optimum (Q=$(results.loss_opt))")
println("""
D1 acceptance criteria (plan §II.4):
  (i)   x̄ interior in [0.35, 0.90], no corner tag on c / gamma_PS / PU
  (ii)  skilled_share within ±0.05 of target
  (iii) wage_premium within ±0.15 log points
  (iv)  densities non-degenerate (inspect plots)
If all hold → one-parameter P_S stands; else → plan §II.5 (free P_S0).

REMINDER: this driver is scaffolding — delete smm_main_temp.jl (and the
_temp candidate/table files) after the checks. If the optimum uses the
widened region, port _TEMP_BOUNDS into default_free_params() before the
official smm_main.jl re-run.
""")
println("\nDone.")
flush(stdout)
############################################################
# mcmc/MCMC_main.jl — standard errors at a stored SMM estimate (entry point)
#
# Loads the SMM estimate for (WINDOW, WEIGHTING) and either
#   · runs a DE-MC chain (mcmc/demc.jl) on the log quasi-posterior in the unconstrained
#     coordinate t, θ = lb + (ub − lb)/(1 + e^(−t)),
#         logπ(t) = −½ · g(θ(t))′W g(θ(t))      on |t| ≤ MCMC_T_BOX
#     (MCMC_PRIOR = :flat_t, LMR's flat prior on t; :flat_theta adds log|dθ/dt|), or
#   · with MCMC_JAC_ONLY = true, skips the chain and uses Ĵ = Ĝ′WĜ at the SMM estimate.
# Under diagonal W, g′Wg = Σ_k ((data_k − model_k)/σ̂_k)², LMR's 1/SD² weighting, which
# fixes the temperature: the chain's dispersion is in standard-error units.
#
# Standard errors are sandwiches B Ĝ′W Ω W Ĝ B. On the chain path the bread B is the
# covariance of the kept draws and Ĝ = ∂g/∂θ is measured at the posterior mean θ̄;
# without a usable chain B = (Ĝ′WĜ)⁻¹ and Ĝ is measured at θ̂. Two columns: Ω = W⁻¹, and
# the sharp bound over every Ω with the estimated diagonal (`se_bound_diagonal`).
#
# Files:
#   seed      estimate_{window}{W}_smm.jls; for a crisis-window chain,
#             estimate_{window}{W}_smm_mcmcbase.jls (CRISIS_HELD at the baseline chain)
#   JAC_ONLY  mcmc_results_{window}{W}_smm.csv; standard errors added to the seed bundle
#   chain     estimate_{window}{W}_mcmc.jls, mcmc_results_{window}{W}_mcmc.csv,
#             output/chains/chain_{window}{W}.jls (chain, draws, moments, Ĝ; also the
#             periodic snapshot), and with MCMC_CHECKPOINT estimate_{window}{W}_mcmcbest.jls
#
# Usage (from project root):
#   julia --project=. --threads auto code/mcmc/MCMC_main.jl
############################################################

# ── Paths ───────────────────────────────────────────────────────────────
# Defined before the version.jl include: version.jl lives in code/smm/, not here.
const MCMC_DIR   = @__DIR__
const CODE_ROOT  = normpath(joinpath(MCMC_DIR, ".."))
const SMM_DIR    = joinpath(CODE_ROOT, "smm")
const SOLVER_DIR = joinpath(CODE_ROOT, "solver")
include(joinpath(CODE_ROOT, "paths.jl"))       # the only source of output paths
include(joinpath(SMM_DIR, "version.jl"))
println("="^60)
println("  RoySearch v$(ROYSEARCH_VERSION) — DE-MC")
println("="^60)
flush(stdout)

# ── Packages (same set as smm_main.jl so the includes resolve) ──────────
print("Loading packages... "); flush(stdout)
using LinearAlgebra, SparseArrays, Statistics, Random, Distributions
using FastGaussQuadrature, Interpolations, Parameters, Printf
using Base.Threads, Optim, CSV, DataFrames, Serialization
using Clustering, QuasiMonteCarlo, JSON3
println("done."); flush(stdout)

# Solves run inside @threads regions; threaded BLAS there would oversubscribe the cores.
LinearAlgebra.BLAS.set_num_threads(1)

# ── Solver + SMM modules as a library (same include order as smm_main.jl) ─
print("Loading solver + SMM modules... "); flush(stdout)
include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))
include(joinpath(SMM_DIR, "settings.jl"))
include(joinpath(SMM_DIR, "moments.jl"))
include(joinpath(SMM_DIR, "smm_params.jl"))
include(joinpath(SMM_DIR, "bundle.jl"))
include(joinpath(SMM_DIR, "smm.jl"))
include(joinpath(SMM_DIR, "candidates.jl"))   # for include-env parity with smm_main
include(joinpath(MCMC_DIR, "mcmc_diagnostics.jl"))
include(joinpath(MCMC_DIR, "demc.jl"))          # uses min_ess / converged_sequential
println("done.")
@printf("Threads available: %d\n\n", Threads.nthreads()); flush(stdout)

# ========================================================================
# CONFIG
# ========================================================================
# Defaults below; each env_setting is overridable by ROYSEARCH_<NAME> and is listed in
# the [env] block printed below. Magnitudes follow LMR's paper (Appendix C p.86: 100
# chains of 10,000, the last 1,000 pooled); structure follows their code.
assert_no_legacy_env()

WINDOW        = env_setting(:WINDOW, :baseline)                 # window to compute SEs for
WEIGHTING     = run_weighting()                            # :diagonal or :equal (moments.jl).
                                  # Selects both the bundle to load and the weighting the
                                  # chain targets; held-out moments come from the bundle.

# ── DE-MC population and proposal (see mcmc/demc.jl) ─────────────────────
MCMC_N           = env_setting(:MCMC_N, 120)
MCMC_GENS        = env_setting(:MCMC_GENS, 5_000)
# 0.8 of 5,000 retains LMR's absolute 1,000 generations.
MCMC_BURN        = env_setting(:MCMC_BURN, 0.8)
# CR is the probability that a coordinate is perturbed. γ scales with the realised mask
# count, so CR changes the step's direction, not its norm.
MCMC_CR          = env_setting(:MCMC_CR, 1.00)          # LMR main_mpi.f90:127
MCMC_DELTA       = env_setting(:MCMC_DELTA, 2)          # LMR main_mpi.f90:131
# b_mult scales the difference vector elementwise, so its off-ridge size grows with the
# widest coordinate's spread: lower it if acceptance decays as the population spreads.
MCMC_B_MULT      = env_setting(:MCMC_B_MULT, 1e-2)      # LMR shock_mult_std, main_mpi.f90:136
# b_add is an absolute shock in t; under :at_seed it is the only mover in generation 1.
MCMC_B_ADD       = env_setting(:MCMC_B_ADD, 1e-4)       # LMR shock_add_std, main_mpi.f90:137

# ── Stuck-chain replacement (LMR mpi_mcmc_mod.f90:419) ───────────────────
MCMC_OUTLIER_IQR = 2.0             # IQRs below Q1 at which a chain is replaced; 0 disables
# The replacement is not reversible: true confines it to burn-in, false is LMR's timing
# (every generation). A replacement past burn-in contaminates the retained draws.
MCMC_OUTLIER_BURN_ONLY = env_setting(:MCMC_OUTLIER_BURN_ONLY, true)

# ── Target ────────────────────────────────────────────────────────────────
# :flat_t      logπ = −Q/2 (LMR): the seed argmin Q is a stationary point of the target.
# :flat_theta  logπ = −Q/2 + log|dθ/dt| (flat on θ): the seed is off-mode, so an :at_seed
#              start must first walk to the mode. :flat_t instead puts mass near box edges,
#              so a coordinate flat near its bound gets a degenerate interval. Report which
#              prior produced a table.
MCMC_PRIOR       = env_setting(:MCMC_PRIOR, :flat_t)
# Half-width of the prior box on t (LMR ±20, main_mpi.f90:146-147). The box makes the
# target proper: θ(t) saturates, so −Q/2 is asymptotically flat in t. Inf is improper.
MCMC_T_BOX       = env_setting(:MCMC_T_BOX, 20.0)
# Space the chain samples. Only :t is usable. :theta needs (1) a per-coordinate b_add and
# (2) `constrained = true` carried to Q(θ̂), Q(θ̄), Q(theta_best), _Q_from_lp, the Ĝ block
# and the unpack_θ reporting sites; adopting it is a MAJOR bump (the estimand moves).
MCMC_SPACE       = env_setting(:MCMC_SPACE, :t)
MCMC_SPACE in (:theta, :t) ||
    error("MCMC_SPACE = $(MCMC_SPACE) is not recognised; use :theta or :t.")
MCMC_SPACE === :t ||
    error("MCMC_SPACE = :theta is not built yet — see the comment above this check for the " *
          "two things it needs. Short version: (1) MCMC_B_ADD is a scalar added to every " *
          "coordinate and must become per-coordinate before θ can be sampled (the rest of " *
          "the DE-MC proposal is already scale-free); (2) the constrained = true flag " *
          "reaches only logposterior, so Q(θ̂), Q(θ̄), Q(theta_best), _Q_from_lp, the Ĝ " *
          "Jacobian block and unpack_θ at the two reporting sites still read the vector as " *
          "t. Until both are done the run completes and reports nonsense — measured " *
          "Q(θ̂) = 1.66e+07 for a seed that scores 490.55 under :t. Delete this guard in the " *
          "version that adopts the change; it is a MAJOR bump, the estimand moves.")

# ── Initial population ────────────────────────────────────────────────────
#   :at_seed  every chain at θ̂ (LMR mpi_mcmc_mod.f90:268); the population grows to the
#             target's scale. R̂ and mcse have no power from a common start.
#   :widths   disperse at the ΔQ=1 half-widths read from MCMC_WIDTHS_CSV (empty ⇒
#             output/logs/width_audit_<window><W>.csv), times MCMC_INIT_DISPERSE.
#   :screen   draw MCMC_INIT_SCREEN candidates around θ̂ (0 ⇒ 20·N) on the §4b scale, keep
#             the feasible ones, seed from the middle of their ranking.
#   :resume   continue the window's chain file; MCMC_GENS is the TOTAL (§4c).
MCMC_WIDTHS_CSV  = env_setting(:MCMC_WIDTHS_CSV, "")
MCMC_INIT        = env_setting(:MCMC_INIT, :at_seed)
MCMC_INIT_SCREEN = env_setting(:MCMC_INIT_SCREEN, 0)  # candidates for :screen (0 → 20·N)
MCMC_PARALLEL    = true           # thread population over chains (see demc.jl header)
MCMC_SEED        = 20260624
# Generations per progress line; acc, dlp and esjd average over this window.
MCMC_PRINT_EVERY = env_setting(:MCMC_PRINT_EVERY, 250)
# Generations per LMR-format block (their stat_display_rate = 500) and per chain snapshot.
# 0 disables both.
MCMC_LMR_PRINT   = env_setting(:MCMC_LMR_PRINT, 500)

# ── Sequential stop (stop_rule, mcmc_diagnostics.jl) ─────────────────────
# 0 = fixed budget, as LMR run; the thresholds below are then inert. R̂ and ESS are always
# reported per coordinate, never gated: worst R̂ trends upward with the budget on this
# target, so no R̂ threshold is reachable.
MCMC_CHECK_EVERY = env_setting(:MCMC_CHECK_EVERY, 0)
# Accuracy gate: MCSE of the reported mean / reported sd (Flegal–Haran–Jones 2008).
MCMC_MCSE_TARGET = env_setting(:MCMC_MCSE_TARGET, 0.05)
# Adaptive DE scale. MCMC_GAMMA_SCALE multiplies ter Braak's 2.38/√(2δ·n_masked) (1.0 is
# his and LMR's value). The brake (MCMC_GAMMA_EVERY > 0) only shortens the step, toward
# MCMC_GAMMA_TARGET; eta = 2 at cadence 25 is calibrated by a closed-loop simulation.
MCMC_GAMMA_SCALE  = env_setting(:MCMC_GAMMA_SCALE, 1.0)
MCMC_GAMMA_ADAPT  = env_setting(:MCMC_GAMMA_ADAPT, false)
MCMC_GAMMA_TARGET = env_setting(:MCMC_GAMMA_TARGET, 0.234)  # Roberts-Gelman-Gilks optimum
MCMC_GAMMA_ETA    = env_setting(:MCMC_GAMMA_ETA, 2.0)
MCMC_GAMMA_EVERY  = env_setting(:MCMC_GAMMA_EVERY, 0)       # 0 disables the brake
# Multiplier on the :widths half-widths. Must exceed 1: at 1 the start is at the target
# scale, R̂ starts at 1 and the convergence test has no power.
MCMC_INIT_DISPERSE = env_setting(:MCMC_INIT_DISPERSE, 2.0)
# Floor on accepted moves in the retained part (10·d).
MCMC_MOVES_MIN   = env_setting(:MCMC_MOVES_MIN, 230)
# Under :flat_t a climb of L log units is ΔQ = 2L; 1.0 is ΔQ = 2, the smallest difference
# the grids resolve.
MCMC_DRIFT_FLAT  = env_setting(:MCMC_DRIFT_FLAT, 1.0)   # log units; 1.0 ⇒ ΔQ = 2
MCMC_ACC_FLOOR   = env_setting(:MCMC_ACC_FLOOR, 0.02)   # abort-and-diagnose below this, two checks running
# Reported against, not gated. At N ≈ 95, R̂ = 1.11 corresponds to MCSE/se = 0.05.
MCMC_RHAT_MAX    = 1.10
# Abort if the running maximum climbs this far above the seed; 0.0 disables. Disabled
# because on an objective with flat directions any chain wide enough to measure a width
# finds better points, so the abort ends exactly the runs that produce a standard error.
MCMC_DRIFT_MAX   = 0.0
# true writes each new best visited point to estimate_{window}{W}_mcmcbest.jls (stage
# :mcmc_checkpoint). Nothing else depends on it: the chain is snapshotted every
# MCMC_LMR_PRINT generations and theta_best is in the chain file either way.
MCMC_CHECKPOINT  = env_setting(:MCMC_CHECKPOINT, false)
# Improvement in Q needed to write a new best point: the ΔQ the criterion resolves.
PROMOTE_MIN_DQ   = 2.0

# ── Jacobian Ĝ = ∂g/∂θ ────────────────────────────────────────────────────
#   :fd      per-coordinate central differences, step set by the moments' own response,
#            refined by Richardson (jacobian_adaptive_fd)
#   :design  least squares on a cloud of radius rel_step·(ub−lb) (jacobian_from_draws);
#            depends on the search box; kept to reproduce bundles before v35
MCMC_JAC_METHOD  = env_setting(:MCMC_JAC_METHOD, :fd)
# true skips the chain: Ĵ = Ĝ'WĜ at the seed (CH Thm 4 admits it), no quantile columns.
MCMC_JAC_ONLY    = env_setting(:MCMC_JAC_ONLY, false)
# ESS floor reported against (0.0 ⇒ min_ess(d)). 450 is MCSE(q05) ≤ 0.10·sd.
MCMC_ESS_MIN     = env_setting(:MCMC_ESS_MIN, 450.0)
# :screen scale per coordinate = FRAC × its posterior width in t, clamped to [FLOOR, CAP].
# Err narrow: DE-MC grows a narrow population but cannot contract a wide one. The cap is
# required: a railed width is ~124 in t and the logit clamp saturates near ±18.42.
MCMC_SCREEN_FRAC  = env_setting(:MCMC_SCREEN_FRAC, 0.3)   # start at 0.3·sd and grow
MCMC_SCREEN_CAP   = env_setting(:MCMC_SCREEN_CAP, 1.0)
MCMC_SCREEN_FLOOR = env_setting(:MCMC_SCREEN_FLOOR, 1e-3) # guard a zero/absent width
MCMC_JAC_DRAWS   = 600            # thinned retained draws re-solved to store the
                                  # moment vector, from which Ĝ is regressed
# Parameters declared to sit at a corner. A label only: it fills the CSV's
# corner_declared column and the banner, and changes no value. Keys may use any of the
# three spellings `_corner_spellings` accepts, e.g. ROYSEARCH_CORNER_PARAMS="alpha_U,delta_S".
CORNER_PARAMS    = env_setting(:CORNER_PARAMS, Symbol[])

derived_dir = joinpath(PROJECT_ROOT, "data", "derived")

W_SUFFIX = weighting_suffix(WEIGHTING)

@printf("Window: %s   weighting: %s\n", WINDOW, W_SUFFIX)
print_env_settings()

# ========================================================================
# 1. Load the SMM estimate that seeds the run
# ========================================================================
# A crisis-window chain starts from the crisis SMM estimate that holds CRISIS_HELD at the
# baseline chain's estimate (smm_main.jl with CRISIS_BASE = :mcmc); every other route
# starts from the window's SMM estimate.
SEED_POINT = is_crisis(WINDOW) && !MCMC_JAC_ONLY ? :smm_mcmcbase : :smm
seed_jls   = estimate_path(WINDOW, W_SUFFIX; point = SEED_POINT)
isfile(seed_jls) || error(
    "No estimate at $seed_jls — run smm_main.jl on $WINDOW" *
    (SEED_POINT === :smm_mcmcbase ? " with ROYSEARCH_CRISIS_BASE=mcmc" : "") *
    " (WEIGHTING=$WEIGHTING) first.")
bundle = read_bundle(seed_jls)
(bundle === nothing || bundle.result === nothing || bundle.spec === nothing) &&
    error("Could not read $seed_jls.")
require_current_estimate(bundle, seed_jls)
if SEED_POINT === :smm_mcmcbase
    # The baseline chain may have been re-run since; compare the held values, at the
    # crisis window's baseline period.
    base_jls = estimate_path(:baseline, W_SUFFIX; point = :mcmc)
    base_b   = read_bundle(base_jls)
    base_b === nothing || (base_b = period_bundle(base_b, PAIR_BASE[WINDOW]))
    base_held(b) = (blocks = unpack_θ(b.result.theta_opt, b.spec);
                    [_baseline_param_value(blk, nm, blocks...) for (blk, nm) in CRISIS_HELD])
    (base_b !== nothing && base_b.result !== nothing && base_held(base_b) == base_held(bundle)) ||
        error("$(basename(seed_jls)) does not hold the parameters of " *
              "$(basename(base_jls)) a $WINDOW run holds there: run smm_main.jl on $WINDOW " *
              "with ROYSEARCH_CRISIS_BASE=mcmc first.")
end
@printf("Seed: %s\n", basename(seed_jls))

θ0    = collect(float.(bundle.result.theta_opt))   # free params, UNCONSTRAINED space
spec0 = bundle.spec

# ========================================================================
# 2. Rebuild the spec with the estimation's weighting; everything else is taken from the
#    estimation spec, so θ0 is valid and the model is solved identically.
# ========================================================================
"""
    chain_spec(s0, dw) -> SMMSpec

The estimation spec `s0` rebuilt for the chain on data window `dw`: same moments, sim,
fixed and free parameters and run settings, the held-out moments of `s0` (weight ≤ 0),
the window's weight matrix under WEIGHTING, and `q_scale = 1` so the quasi-posterior is
the raw g′Wg.
"""
function chain_spec(s0::SMMSpec, dw::Symbol)
    skip = Symbol[k for k in keys(s0.moments) if s0.moments[k].weight <= 0.0]
    return build_smm_spec(
        s0.moments, s0.sim;
        fixed        = s0.fixed,
        free_specs   = s0.free,
        run          = s0.run,
        W            = load_weight_matrix(; window=dw, derived_dir=derived_dir,
                                            weighting=WEIGHTING, skip_moments=skip),
        q_scale      = 1.0,          # raw g'Wg: the quasi-posterior must not be rescaled
        skip_moments = skip,
    )
end
# A joint baseline is rebuilt period by period, each on its own window's weights.
spec = is_joint(spec0) ?
    _spec_with_init(joint_spec([chain_spec(s, w) for (w, s, _) in period_list(spec0)],
                               BASELINE_PERIODS), pack_θ(spec0.free)) :
    chain_spec(spec0, WINDOW)
@assert length(θ0) == length(spec.free) "seed length ($(length(θ0))) ≠ free params " *
    "($(length(spec.free))) — bundle and spec rebuild disagree."

MOM_KEYS = active_moment_keys(spec)
K, d     = length(MOM_KEYS), length(spec.free)
NK       = length(vacancy_cost_rows(spec))      # the vacancy costs, two per period

# Sampling SDs of the active moments, in MOM_KEYS order; defined only under diagonal W.
σ̂ = WEIGHTING === :diagonal ? [1.0 / sqrt(spec.W[i, i]) for i in 1:K] : fill(NaN, K)

"""
    _corner_spellings(ps) -> Vector{Symbol}

The three names a CORNER_PARAMS key may use for free parameter `ps`: its display symbol
(`param_symbol`, e.g. :b_S), its ParamSpec name (:bS), and its FIX_PARAMS key
(`_DEFAULT_PARAM_KEY`, e.g. :alpha_U).
"""
_corner_spellings(ps::ParamSpec) =
    Symbol[Symbol(param_symbol(ps)), ps.name,
           get(_DEFAULT_PARAM_KEY, (ps.block, ps.name), ps.name)]
# Resolved at startup, so a misspelled key fails before any solve.
CORNER_FLAG = [any(in(CORNER_PARAMS), _corner_spellings(ps)) for ps in spec.free]
let unknown = setdiff(CORNER_PARAMS,
                      reduce(vcat, _corner_spellings.(spec.free); init = Symbol[]))
    isempty(unknown) || error(
        "CORNER_PARAMS names no free parameter of this run: " * join(unknown, ", ") *
        ".\n  Free parameters here: " * join(param_symbol.(spec.free), ", "))
end
if !isempty(CORNER_PARAMS)
    @printf("  CORNER_PARAMS: %d of %d free parameters declared at a corner — %s\n",
            count(CORNER_FLAG), d,
            join([param_symbol(ps) for (k, ps) in enumerate(spec.free) if CORNER_FLAG[k]], ", "))
    println("    a label only, carried to the CSV's corner_declared column; no value changes.")
end

# ========================================================================
# 3. Log quasi-posterior: −½·g'Wg plus the MCMC_PRIOR term. Infeasible → −Inf.
# ========================================================================
# Bundles store t; under :theta the seed is converted once here.
if MCMC_SPACE === :theta
    θ0 = [_to_constrained(θ0[k], spec.free[k].lb, spec.free[k].ub) for k in eachindex(θ0)]
end
# ── init = :widths: per-coordinate, per-direction half-widths in the sampled space ────────
_init_widths    = nothing
_init_widths_dn = nothing
if MCMC_INIT === :widths
    wcsv = isempty(MCMC_WIDTHS_CSV) ?
           joinpath(out_logs(), "width_audit_$(WINDOW)$(W_SUFFIX).csv") : MCMC_WIDTHS_CSV
    isfile(wcsv) ||
        error("MCMC_INIT = :widths needs $(wcsv): a CSV with one row per free parameter " *
              "and columns lb, ub, theta, wQ1_up, wQ1_dn (the ΔQ = 1 half-widths at θ̂). " *
              "Set MCMC_WIDTHS_CSV to read another file.")
    wdf = CSV.read(wcsv, DataFrame)
    nrow(wdf) == length(spec.free) ||
        error("$(basename(wcsv)) has $(nrow(wdf)) rows, expected $(length(spec.free)); " *
              "it was written for a different free-parameter set.")
    # Up and down are kept separate: a coordinate on a floor has a width upward only. A
    # missing direction takes the other one, then 1% of the distance to the near bound.
    # The ENDPOINTS θ ± w are mapped to t (exact); a local derivative is wrong by orders of
    # magnitude near a bound.
    _up = Float64[]; _dn = Float64[]
    for r in eachrow(wdf)
        lbk = Float64(r.lb); ubk = Float64(r.ub); thk = Float64(r.theta)
        span = ubk - lbk
        wu = isfinite(r.wQ1_up) ? Float64(r.wQ1_up) :
             isfinite(r.wQ1_dn) ? Float64(r.wQ1_dn) : 0.01 * min(ubk - thk, thk - lbk)
        wd = isfinite(r.wQ1_dn) ? Float64(r.wQ1_dn) :
             isfinite(r.wQ1_up) ? Float64(r.wQ1_up) : 0.01 * min(ubk - thk, thk - lbk)
        θu = min(thk + wu, lbk + (1.0 - 1e-9) * span)
        θd = max(thk - wd, lbk + 1e-9 * span)
        if MCMC_SPACE === :theta
            push!(_up, θu - thk); push!(_dn, thk - θd)
        else
            tk = _to_unconstrained(thk, lbk, ubk)
            push!(_up, _to_unconstrained(θu, lbk, ubk) - tk)
            push!(_dn, tk - _to_unconstrained(θd, lbk, ubk))
        end
    end
    _init_widths = _up; _init_widths_dn = _dn
    bad = findall(w -> !isfinite(w) || w <= 0.0, _init_widths)
    if !isempty(bad)
        badlbl = join([spec.free[k].label for k in bad], "; ")
        error("$(basename(wcsv)) brackets neither direction for coordinate(s) $(bad) " *
              "[$(badlbl)]; rerun the audit or supply a width by hand before using " *
              "this mode.")
    end
    @printf("  init widths from %s (space = :%s, disperse = %.2f)\n",
            basename(wcsv), MCMC_SPACE, MCMC_INIT_DISPERSE)
    @printf("    up: min=%.4g median=%.4g max=%.4g\n",
            minimum(_init_widths), median(_init_widths), maximum(_init_widths))
    @printf("    dn: min=%.4g median=%.4g max=%.4g\n",
            minimum(_init_widths_dn), median(_init_widths_dn), maximum(_init_widths_dn))
end

# The log-Jacobian term applies only in :t space (in :theta a flat box on θ is already the
# flat-on-θ target). A const Bool, so the hot loop does not branch on a Symbol.
const _USE_LOGJAC = (MCMC_SPACE === :t) && (MCMC_PRIOR === :flat_theta)
MCMC_PRIOR in (:flat_theta, :flat_t) ||
    error("MCMC_PRIOR = $(MCMC_PRIOR) is not recognised; use :flat_theta or :flat_t.")

# Vacancy costs at every point the target evaluates, keyed by the point's hash: a draw is a
# copy of an evaluated point, so its costs are read back instead of re-solved.
const _AT_POINT = Dict{UInt64,Vector{Float64}}()
const _AT_LOCK  = ReentrantLock()
"""
    _at_point(θ) -> Vector{Float64}

The NK vacancy costs stored by `logposterior` at point `θ`, or NaNs if `θ` was never
evaluated feasibly.
"""
_at_point(θ) = lock(() -> get(_AT_POINT, hash(collect(float.(θ))), fill(NaN, NK)), _AT_LOCK)

"""
    logposterior(θ) -> Float64

Log quasi-posterior at the chain vector `θ`: −½·Q, plus `logjac_box` under
`_USE_LOGJAC`; −Inf outside the prior box (tested before the solve) or where no
equilibrium exists. Records the point's vacancy costs in `_AT_POINT`. Thread-safe.
"""
function logposterior(θ)
    # Prior support: |t| ≤ MCMC_T_BOX under :t, the economic box [lb, ub] under :theta.
    if MCMC_SPACE === :theta
        @inbounds for k in eachindex(θ)
            ps = spec.free[k]
            (θ[k] <= ps.lb || θ[k] >= ps.ub) && return -Inf
        end
        ev = equilibrium_at(θ, spec; constrained = true)
    else
        any(abs(t) > MCMC_T_BOX for t in θ) && return -Inf
        ev = equilibrium_at(θ, spec)
    end
    isfinite(ev.Q) || return -Inf
    lock(() -> (_AT_POINT[hash(collect(float.(θ)))] = k_values(ev.k)), _AT_LOCK)
    return _USE_LOGJAC ? -0.5 * ev.Q + logjac_box(collect(float.(θ)), spec.free) :
                         -0.5 * ev.Q
end

"""
    _Q_from_lp(θ, lp) -> Float64

The criterion Q at `θ` recovered from a stored log-target `lp`; the inverse of
`logposterior`, and must stay in step with it.
"""
_Q_from_lp(θ, lp) = _USE_LOGJAC ?
    -2.0 * (lp - logjac_box(collect(float.(θ)), spec.free)) : -2.0 * lp

# ========================================================================
# 4. Run DE-MC (under MCMC_JAC_ONLY `res` is a one-draw stand-in of the same shape)
# ========================================================================
# Q at the seed, computed once: the checkpoint, the drift split and Q_ckpt use it.
seed_eq    = equilibrium_at(θ0, spec)
Q_seed     = seed_eq.Q
Q_ckpt     = Ref(Q_seed)
best_jls   = mcmcbest_path(WINDOW, W_SUFFIX)

"""
    promote!(θ, Q_new, Q_old; label) -> nothing

Write `θ` to estimate_<window><W>_mcmcbest.jls with stage `:mcmc_checkpoint` and
`converged = false`: a record of where the chain went, never an estimate, and never
written into an SMM or chain-estimate bundle. Callers guarantee `Q_new < Q_old`.
"""
function promote!(θ::Vector{Float64}, Q_new::Float64, Q_old::Float64; label::String)
    write_bundle(best_jls;
                 result = SMMResult(θ, _params_to_namedtuple(θ, spec), Q_new, false, 0, spec),
                 spec = spec, stage = :mcmc_checkpoint,
                 provenance = run_provenance(window = WINDOW, w_suffix = W_SUFFIX,
                                             version = ROYSEARCH_VERSION),
                 tag = label)
    # The destination is the same on every call, so it is printed once, on the first.
    Q_old == Q_seed && @printf("  best visited points → %s\n", basename(best_jls))
    @printf("  %s: Q %.6f → %.6f (ΔQ=%.4f)\n", label, Q_old, Q_new, Q_old - Q_new)
    return nothing
end

# The settings that define the chain beyond the spec, recorded in every chain file so that
# MCMC_INIT = :resume can refuse to continue a run sampled differently.
const SAMPLER = (MCMC_SPACE = MCMC_SPACE, MCMC_PRIOR = MCMC_PRIOR, MCMC_T_BOX = MCMC_T_BOX,
                 MCMC_CR = MCMC_CR, MCMC_DELTA = MCMC_DELTA)

"""
    snapshot_chain(ch, lp, acc, out, rep, g, state)

Write the chain so far (history `ch`, log-targets `lp`, accepted/outcome/replaced
records, `g` generations, sampler `state`) to the window's chain path, through a temp
file and a rename so a kill mid-write leaves the previous snapshot intact. The file
carries `spec`, `SAMPLER` and `partial = true`; it is what MCMC_INIT = :resume continues
from, and §7 overwrites it with the full bundle at the end.
"""
function snapshot_chain(ch, lp, acc, out, rep, g, state)
    pth = chain_path(WINDOW, W_SUFFIX)
    tmp = pth * ".tmp"
    mkpath(dirname(pth))
    open(tmp, "w") do io
        serialize(io, (chain = Array(ch), chain_lp = Array(lp),
                       accepted = BitMatrix(acc), outcome = Array(out),
                       replaced = BitMatrix(rep), demc_state = state, sampler = SAMPLER,
                       gens = g, gens_requested = MCMC_GENS,
                       burn = clamp(floor(Int, MCMC_BURN * g), 0, g - 1),
                       spec = spec, window = WINDOW,
                       lb = [ps.lb for ps in spec.free],
                       ub = [ps.ub for ps in spec.free],
                       labels = [ps.label for ps in spec.free],
                       partial = true,
                       provenance = run_provenance(window = WINDOW, w_suffix = W_SUFFIX,
                                                   version = ROYSEARCH_VERSION)))
    end
    mv(tmp, pth; force = true)
    @printf("  [chain snapshot] g=%d → %s (%.0f MB)\n", g, basename(pth),
            filesize(pth) / 1e6); flush(stdout)
end

"""
    checkpoint_best(θ, lp, g) -> nothing

`on_best` callback: when the new incumbent `θ` (log-target `lp`, generation `g`) beats
the last point written by more than PROMOTE_MIN_DQ, write it with `promote!`. Q comes
from `_Q_from_lp`, so no solve is spent.
"""
function checkpoint_best(θ, lp, g)
    Q = _Q_from_lp(θ, lp)
    Q < Q_ckpt[] - PROMOTE_MIN_DQ || return nothing
    promote!(collect(float.(θ)), Q, Q_ckpt[]; label = @sprintf("CHECKPOINT g=%d", g))
    Q_ckpt[] = Q
    return nothing
end

# ------------------------------------------------------------------------
# 4b. Screen preconditioning (init = :screen): a local design and Ĝ around the seed give
#     each coordinate's JOINT width se(J⁻¹) — an order of magnitude for the proposal scale,
#     not a derivative.
# ------------------------------------------------------------------------
need_pre = !MCMC_JAC_ONLY && MCMC_INIT === :screen
Ĝ_pre = nothing; se_curv_pre = nothing; init_scale = nothing; init_width = nothing
if need_pre
    @printf("Preconditioning the screen: local design at %d points... ", MCMC_JAC_DRAWS)
    flush(stdout)
    Xp = local_design(θ0, spec.free; n = MCMC_JAC_DRAWS,
                      rng = MersenneTwister(MCMC_SEED + 1))
    Mp = Matrix{Float64}(undef, K, size(Xp, 2))
    bp = [Vector{Float64}(undef, K) for _ in 1:nthreads()]
    @threads for i in axes(Xp, 2)
        b = bp[threadid()]
        Qi = smm_objective(view(Xp, :, i), spec; moments_out = b)
        @views Mp[:, i] .= isfinite(Qi) ? b : NaN
    end
    kp = vec(all(isfinite, Mp; dims = 1))
    Ĝ_pre, _ = jacobian_from_draws(Xp[:, kp], Mp[:, kp], spec.free)
    _, se_curv_pre = se_bound_diagonal(Ĝ_pre, spec.W, σ̂)
    @printf("%d feasible.\n", count(kp)); flush(stdout)

    # _screen_scale(θ0v, free, se_curv, frac, floor_, cap_) -> (scale, width, n_cap, n_flr, n_fb)
    # Converts each constrained width se_curv[k] to t at the unconstrained seed θ0v,
    # dθ/dt = (ub−lb)·σ(t)(1−σ(t)) evaluated at t (not at θ); no usable width falls back to
    # 1 in t. `width` is unclamped; `scale = frac·width` clamped to [floor_, cap_]. A
    # function so the counters are local under top-level soft scope.
    function _screen_scale(θ0v, free, se_curv, frac, floor_, cap_)
        d_free = length(free)
        scale = Vector{Float64}(undef, d_free)
        width = Vector{Float64}(undef, d_free)   # UNCLAMPED se_t, for the check below
        n_cap = 0; n_flr = 0; n_fb = 0
        for k in 1:d_free
            f  = free[k]
            u  = 1.0 / (1.0 + exp(-θ0v[k]))
            dθ = (f.ub - f.lb) * u * (1 - u)
            se_t = (isfinite(se_curv[k]) && se_curv[k] > 0 && dθ > 0) ?
                       se_curv[k] / dθ : NaN
            if !isfinite(se_t)
                se_t = 1.0; n_fb += 1
            end
            width[k] = se_t                      # the target width, before any clamping
            v = frac * se_t
            v > cap_   && (v = cap_;   n_cap += 1)
            v < floor_ && (v = floor_; n_flr += 1)
            scale[k] = v
        end
        return scale, width, n_cap, n_flr, n_fb
    end
    init_scale, init_width, n_cap, n_flr, n_fb =
        _screen_scale(θ0, spec.free, se_curv_pre,
                      MCMC_SCREEN_FRAC, MCMC_SCREEN_FLOOR, MCMC_SCREEN_CAP)
    d_free = length(spec.free)
    @printf("  screen scale: frac=%.2f  capped %d  floored %d  fallback %d  (of %d)\n",
            MCMC_SCREEN_FRAC, n_cap, n_flr, n_fb, d_free)
    # A floor or cap that raises a scale above the coordinate's own width over-disperses it,
    # and DE-MC cannot contract that direction.
    n_over = count(k -> init_scale[k] > init_width[k], 1:d_free)
    n_over > 0 && @printf("  WARNING: %d/%d coordinates have scale > own posterior width \
(floor/cap raised them); DE-MC cannot contract these and their SD is biased UP.\n",
                          n_over, d_free)
    n_fb > d_free ÷ 4 && @printf("  WARNING: %d/%d coordinates had no usable width — Ĝ may be bad.\n",
                                 n_fb, d_free)
    flush(stdout)
end

# ------------------------------------------------------------------------
# 4c. MCMC_INIT = :resume: continue the window's chain file (completed or snapshot) after
#     its last generation; MCMC_GENS is the total. The target (window, free set, bounds,
#     pinned values, active moments, data, W) and the sampler (N, SAMPLER) must match, and
#     the stored final population must reproduce its log-target bit for bit. Files written
#     before v61.3.0 carry no SAMPLER, so MCMC_CR and MCMC_DELTA go unchecked for them.
# ------------------------------------------------------------------------
RESUME = nothing
if !MCMC_JAC_ONLY && MCMC_INIT === :resume
    stored_jls = chain_path(WINDOW, W_SUFFIX)
    isfile(stored_jls) || error("MCMC_INIT = :resume: no stored run at $stored_jls.")
    stored = open(deserialize, stored_jls)
    g0     = size(stored.chain, 3)
    MCMC_GENS > g0 || error("MCMC_INIT = :resume: MCMC_GENS = $MCMC_GENS is the total, and " *
                            "$(basename(stored_jls)) already holds $g0 generations.")
    let s = stored.spec,
        same = [("window",            stored.window,                         WINDOW),
                ("free parameters",   [(ps.block, ps.name) for ps in s.free],
                                      [(ps.block, ps.name) for ps in spec.free]),
                ("bounds",            [(ps.lb, ps.ub) for ps in s.free],
                                      [(ps.lb, ps.ub) for ps in spec.free]),
                ("pinned values",     pinned_key(s),                         pinned_key(spec)),
                ("active moments",    active_moment_keys(s),                 MOM_KEYS),
                ("moment data",       [s.moments[k].value for k in active_moment_keys(s)],
                                      [spec.moments[k].value for k in MOM_KEYS]),
                ("W",                 s.W,                                   spec.W),
                ("MCMC_N",            size(stored.chain, 2),                 MCMC_N)]
        hasproperty(stored, :sampler) &&
            append!(same, [(String(k), stored.sampler[k], SAMPLER[k]) for k in keys(SAMPLER)])
        bad = [l * (a isa AbstractArray ? "" : " (stored $a, this run $b)")
               for (l, a, b) in same if !isequal(a, b)]
        isempty(bad) || error("MCMC_INIT = :resume: $(basename(stored_jls)) is not this run's " *
                              "chain. It differs in " * join(bad, "; ") * ".")
    end
    hasproperty(stored, :demc_state) || hasproperty(stored, :n_replaced) ||
        error("MCMC_INIT = :resume: $(basename(stored_jls)) is a snapshot written before " *
              "v61.3.0, which records neither the sampler state nor the best visited point.")
    lp_now = Vector{Float64}(undef, MCMC_N)
    _eval_population!(lp_now, logposterior, stored.chain[:, :, g0], MCMC_N, MCMC_PARALLEL)
    lp_now == stored.chain_lp[:, g0] || error(
        "MCMC_INIT = :resume: the stored final population does not reproduce its log-target " *
        "($(count(lp_now .!= stored.chain_lp[:, g0])) of $MCMC_N chains differ, largest " *
        @sprintf("relative difference %.3e", maximum(abs.(lp_now .- stored.chain_lp[:, g0]) ./
                                                     abs.(stored.chain_lp[:, g0]))) *
        "). The code or the data no longer define the target the stored run sampled.")
    # A file written before v61.3.0 supplies the state its final bundle records; the rest
    # takes fresh values in run_demc, the RNG replayed from MCMC_SEED.
    global RESUME = (chain = stored.chain, chain_lp = stored.chain_lp,
                     accepted = stored.accepted, replaced = stored.replaced,
                     outcome = hasproperty(stored, :outcome) ? stored.outcome : nothing,
                     state = hasproperty(stored, :demc_state) ? stored.demc_state :
                             (n_replaced = stored.n_replaced, last_replace = stored.last_replace,
                              lp_best = stored.lp_best, θ_best = stored.theta_best))
end

res = MCMC_JAC_ONLY ?
    (draws = reshape(θ0, :, 1), chain = reshape(θ0, :, 1, 1), accept = NaN,
     lp = [logposterior(θ0)], N = 1, gens = 0, gens_requested = 0, burn = 0,
     lp_seed = logposterior(θ0), lp_best = logposterior(θ0), theta_best = θ0,
     drift = 0.0, aborted = false, n_replaced = 0, last_replace = 0) :
    run_demc(logposterior, θ0;
               N = MCMC_N, gens = MCMC_GENS, burn_frac = MCMC_BURN,
               CR = MCMC_CR, δ = MCMC_DELTA, parallel = MCMC_PARALLEL,
               b_add = MCMC_B_ADD, b_mult = MCMC_B_MULT, t_box = MCMC_T_BOX,
               init_widths = _init_widths, init_widths_dn = _init_widths_dn,
               init_disperse = MCMC_INIT_DISPERSE,
               init = MCMC_INIT, init_screen = MCMC_INIT_SCREEN, resume = RESUME,
               init_scale = init_scale, init_width = init_width,
               outlier_iqr = MCMC_OUTLIER_IQR,
               outlier_burn_only = MCMC_OUTLIER_BURN_ONLY,
               print_every = MCMC_PRINT_EVERY, lmr_print = MCMC_LMR_PRINT,
               on_dump = snapshot_chain,
               gamma_scale0 = MCMC_GAMMA_SCALE,
               labels = [ps.label for ps in spec.free],
               on_best = MCMC_CHECKPOINT ? checkpoint_best : nothing,
               check_every = MCMC_CHECK_EVERY, rhat_max = MCMC_RHAT_MAX,
               ess_min = MCMC_ESS_MIN, drift_max = MCMC_DRIFT_MAX,
               moves_min = MCMC_MOVES_MIN, drift_flat = MCMC_DRIFT_FLAT,
               acc_floor = MCMC_ACC_FLOOR, mcse_target = MCMC_MCSE_TARGET,
               gamma_adapt = MCMC_GAMMA_ADAPT, gamma_target = MCMC_GAMMA_TARGET,
               gamma_eta = MCMC_GAMMA_ETA, gamma_every = MCMC_GAMMA_EVERY,
               lb = [ps.lb for ps in spec.free], ub = [ps.ub for ps in spec.free],
               rng = MersenneTwister(MCMC_SEED))

# The draws reach disk here, before §5 (which re-solves and can throw); §7 overwrites this
# file with the full bundle. Not under JAC_ONLY, whose stub would replace a real chain.
MCMC_JAC_ONLY ||
    snapshot_chain(res.chain, res.chain_lp, res.accepted, res.outcome, res.replaced, res.gens,
                   res.state)

# ========================================================================
# 5. Ĝ and the standard errors
# ========================================================================
#   use_design  Ĵ from a local design around θ̂ (JAC_ONLY, or an aborted chain)
#   draws_ok    the draws can be summarised (mean, SD, R̂) — true for an aborted chain too
#   chain_ok    a usable chain: bread P and Ĝ at θ̄
use_design = MCMC_JAC_ONLY || res.aborted
chain_ok   = !use_design && res.gens > 0
draws_ok   = res.gens > 0 && size(res.draws, 2) > 1
n_kept     = size(res.draws, 2)
# The regression is solved only when read: it is Ĝ under :design, and a chain run keeps its
# draws for curvature_check.
need_reg = MCMC_JAC_METHOD === :design || chain_ok

"""
    _draw_to_θ(x, lb, ub) -> Float64

Map a stored chain coordinate to θ: the logistic box map under :t, the identity under :theta.
"""
_draw_to_θ(x, lb, ub) = MCMC_SPACE === :theta ? x : _to_constrained(x, lb, ub)

# Posterior SD of the pooled kept draws in θ (LMR's reported SE). It never differentiates
# Q, so it remains valid where Q is only piecewise smooth.
se_chain = draws_ok ?
    [std([_draw_to_θ(res.draws[k, t], spec.free[k].lb, spec.free[k].ub)
          for t in axes(res.draws, 2)]) for k in 1:d] :
    fill(NaN, d)
# Posterior median: invariant to the sampled space (a monotone map commutes with
# quantiles). Reported beside the mean, never instead.
θ̃_med = draws_ok ?
    [median([_draw_to_θ(res.draws[k, t], spec.free[k].lb, spec.free[k].ub)
             for t in axes(res.draws, 2)]) for k in 1:d] :
    fill(NaN, d)
# Posterior means. θ̄_con, the REPORTED estimate, maps each draw to θ and then averages, so
# it and se_chain are moments of the same sample. θ̄_bnd = σ(mean t) is only the fallback
# point for the post-mean bundle (§9): the feasible set is not convex, so a mean of
# feasible draws need not solve, and Q(θ̄) is re-solved rather than assumed.
θ̄_con = draws_ok ?
    [mean(_draw_to_θ(res.draws[k, t], spec.free[k].lb, spec.free[k].ub)
          for t in axes(res.draws, 2)) for k in 1:d] :
    fill(NaN, d)
θ̄_bnd = draws_ok && MCMC_SPACE !== :theta ?
    [_to_constrained(mean(res.draws[k, t] for t in axes(res.draws, 2)),
                     spec.free[k].lb, spec.free[k].ub) for k in 1:d] :
    θ̄_con
# Each kept draw's vacancy costs, read back from the evaluation that scored it. A resumed
# run did not evaluate the stored generations, so those draws are solved once here.
draws_ok && let todo = unique([res.draws[:, t] for t in axes(res.draws, 2)
                               if all(isnan, _at_point(view(res.draws, :, t)))])
    @threads for t in todo
        logposterior(t)
    end
end
k_draws = draws_ok ?
    reduce(hcat, [_at_point(view(res.draws, :, t)) for t in axes(res.draws, 2)]) :
    Matrix{Float64}(undef, NK, 0)
k_post = !draws_ok ? nothing :
    let kd = [filter(isfinite, k_draws[i, :]) for i in 1:NK]
        nt(f) = k_struct(spec, f.(kd))
        (mean = nt(mean), sd = nt(std), q025 = nt(x -> quantile(x, 0.025)),
         q500 = nt(median), q975 = nt(x -> quantile(x, 0.975)))
    end

# The bread is the inverse curvature of Q/2. Ĵ = Ĝ'WĜ is only its Gauss–Newton part, and a
# point Hessian does not replace it because Q is not quadratic at the scale of one standard
# error. On the chain path the bread is P, the covariance of the kept draws in θ, and Ĝ is
# measured at θ̄; otherwise the bread is J⁻¹ and Ĝ is measured at θ̂.
θ̄_t = draws_ok ?
    [_to_unconstrained(θ̄_con[k], spec.free[k].lb, spec.free[k].ub) for k in 1:d] : fill(NaN, d)
P   = chain_ok ?
    cov([_draw_to_θ(res.draws[k, t], spec.free[k].lb, spec.free[k].ub)
         for t in axes(res.draws, 2), k in 1:d]) : nothing
θ_G = chain_ok ? θ̄_t : θ0

"""
    moments_and_k(t) -> Vector{Float64}

The K active moments (MOM_KEYS order) followed by the NK vacancy costs at chain vector
`t`, or K + NK NaNs where no equilibrium exists. The costs ride along so their
derivatives cost no extra solve.
"""
function moments_and_k(t)
    ev = equilibrium_at(collect(float.(t)), spec)
    isfinite(ev.Q) || return fill(NaN, K + NK)
    return vcat([getproperty(ev.moments, m) for m in MOM_KEYS], k_values(ev.k))
end

if need_reg
    Xj = if use_design
        local_design(θ0, spec.free; n = MCMC_JAC_DRAWS, rng = MersenneTwister(MCMC_SEED + 1))
    else
        sel = n_kept <= MCMC_JAC_DRAWS ? collect(1:n_kept) :
                 round.(Int, range(1, n_kept; length = MCMC_JAC_DRAWS))
        res.draws[:, sel]
    end
    MK = Matrix{Float64}(undef, K + NK, size(Xj, 2))
    @printf("Storing moments at %d %s for Ĝ... ", size(Xj, 2),
            use_design ? "local-design points" : "thinned draws"); flush(stdout)
    @threads for i in axes(Xj, 2)
        @views MK[:, i] .= moments_and_k(Xj[:, i])
    end
    keep    = vec(all(isfinite, MK; dims = 1))
    Msel    = MK[1:K, keep]
    draws_J = Xj[:, keep]
    Ĝk_reg, _ = jacobian_from_draws(draws_J, MK[K+1:K+NK, keep], spec.free)
    @printf("%d feasible.\n", size(Msel, 2)); flush(stdout)

    # On the design path the §4b pre-pass already built this Ĝ on the same design.
    Ĝ_reg, R2 = (need_pre && use_design && Ĝ_pre !== nothing) ?
                    (Ĝ_pre, fill(NaN, K)) : jacobian_from_draws(draws_J, Msel, spec.free)
else
    Msel, draws_J = Matrix{Float64}(undef, K, 0), Matrix{Float64}(undef, d, 0)
    Ĝ_reg, R2     = nothing, fill(NaN, K)
    Ĝk_reg        = nothing
end
if MCMC_JAC_METHOD === :fd
    print("Adaptive-FD Ĝ: per-coordinate step + Richardson... "); flush(stdout)
    # Failed solves on the ladder, the centre excluded. moments_and_k keys on the returned
    # Q, since smm_objective writes its moment buffer only after its guards.
    n_fail = Threads.Atomic{Int}(0)
    # mom_at(t): moments_and_k(t), counting infeasible non-centre points in n_fail.
    function mom_at(t)
        mk = moments_and_k(t)
        isfinite(mk[1]) || t == θ_G || Threads.atomic_add!(n_fail, 1)
        return mk
    end
    Ĝmk, fd = jacobian_adaptive_fd(θ_G, spec.free, σ̂, mom_at; n_carried = NK)
    global Ĝ, Ĝk = Ĝmk[1:K, :], Ĝmk[K+1:K+NK, :]
    @printf("%d of %d coordinates in %d solves, %d not converged.\n",
            count(fd.ok), d, sum(fd.n_eval), n_fail[])
    flush(stdout)
    # No fallback: a Jacobian assembled from two estimators is not a Jacobian.
    if !all(fd.ok)
        error("adaptive-FD Ĝ could not measure $(count(.!fd.ok)) of $d columns: " *
              join(param_symbol.(spec.free[.!fd.ok]), ", ") *
              ". The model did not converge at enough of the finite-difference steps around " *
              (chain_ok ? "θ̄" : "θ̂") * " " *
              "in these coordinates to form a derivative. No standard error is reported rather " *
              "than one built from mixed sources. " *
              "Set ROYSEARCH_MCMC_JAC_METHOD=design for the pre-v35 regression Jacobian.")
    end
else
    global Ĝ, Ĝk, fd = Ĝ_reg, Ĝk_reg, nothing
end
# se_sw: the sandwich at Ω = W⁻¹ (bread P on the chain path, J⁻¹ otherwise); se_bnd: its
# bound over every Ω with the same diagonal.
se_bnd, se_sw = se_bound_diagonal(Ĝ, spec.W, σ̂; bread = P)

# Vacancy costs are pinned by free entry given the tightnesses: delta-method errors with
# their gradient in the parameter vector, the rows of Ĝk.
k_hat            = k_values(seed_eq.k)
se_bnd_k, se_sw_k = se_bound_diagonal(Ĝ, spec.W, σ̂, Matrix(Ĝk'); bread = P)

# θ̂'s attracting check, one per baseline period. D_ij = ∂ln k_i/∂ln θ_j is read off the
# tightness columns of Ĝk when Ĝ sits at θ̂ and both tightnesses are free; otherwise (chain
# path, or a pinned tightness) the check is solved directly.
"""
    _tight_col(s, idx, blk) -> Union{Int,Nothing}

Index in the full parameter vector (through `idx`) of the tightness coordinate of block
`blk` (`:unsk` or `:skl`) in period spec `s`, or `nothing` when it is not free.
"""
_tight_col(s, idx, blk) =
    (j = findfirst(ps -> (ps.block, ps.name) == (blk, :tight), s.free); j === nothing ? nothing : idx[j])
let cols = [(_tight_col(s, idx, :unsk), _tight_col(s, idx, :skl)) for (_, s, idx) in period_list(spec)]
    global attract = if chain_ok || any(c -> nothing in c, cols)
        attracting_check(θ0, spec)
    else
        checks = map(enumerate(cols)) do (i, (jU, jS))
            θU = _to_constrained(θ0[jU], spec.free[jU].lb, spec.free[jU].ub)
            θS = _to_constrained(θ0[jS], spec.free[jS].lb, spec.free[jS].ub)
            rU, rS = 2i - 1, 2i
            attracting_check([Ĝk[rU, jU] * θU / k_hat[rU]  Ĝk[rU, jS] * θS / k_hat[rU];
                              Ĝk[rS, jU] * θU / k_hat[rS]  Ĝk[rS, jS] * θS / k_hat[rS]])
        end
        is_joint(spec) ? NamedTuple{Tuple(w for (w, _, _) in period_list(spec))}(Tuple(checks)) :
                         only(checks)
    end
end
isempty(attracting_line(attract)) || println(attracting_line(attract))

# cov(Θ) is singular with fewer draws than parameters, so the cross-check needs the chain.
jgap    = chain_ok && size(draws_J, 2) > d ?
          curvature_check(draws_J, Ĝ, spec.W, spec.free) : NaN

# ========================================================================
# 6. Gate diagnostics and the parameter table (each quantity printed once).
# ========================================================================
rhat, ess = draws_ok ? split_rhat_ess(res.chain, res.burn) :
                       (fill(NaN, d), fill(NaN, d))
# The vacancy costs get the same split R̂ and ESS: k_draws follows the column order of
# res.draws (chain fastest), so it reshapes to NK × chain × kept generation.
rhat_k, ess_k = draws_ok && size(k_draws, 2) == size(res.draws, 2) &&
                    size(res.draws, 2) % size(res.chain, 2) == 0 ?
                split_rhat_ess(reshape(k_draws, NK, size(res.chain, 2), :), 0) :
                (fill(NaN, NK), fill(NaN, NK))
rhat_all, ess_all = vcat(rhat, rhat_k), vcat(ess, ess_k)
blo, bhi  = draws_ok ? boundary_mass(res.draws, spec.free) :
                       (fill(NaN, d), fill(NaN, d))
sgrow     = draws_ok ? spread_growth(res.chain, res.burn) : fill(NaN, d)

# Q at the best visited point: for the drift split and for §8.
Q_best = smm_objective(res.theta_best, spec)

# How Ĝ was measured: every standard error in the table rests on it.
R2f = filter(isfinite, R2)
if MCMC_JAC_METHOD === :fd
    hf  = filter(isfinite, fd.h_frac)
    stf = filter(isfinite, vec(fd.stab))
    @printf("  Ĝ: %d×%d adaptive FD + Richardson, %d solves, target ‖Δm/σ̂‖=%.1f\n",
            K, d, sum(fd.n_eval), fd.target_sd)
    @printf("      step/box width: median %.2e, range %.2e–%.2e | Richardson instability median %.1e\n",
            median(hf), minimum(hf), maximum(hf),
            isempty(stf) ? NaN : median(stf))
    fd.centre_ok ||
        @printf("      NOTE: Q(%s) is not finite at %d threads, so Ĝ is centred on a point the \
solver rejects. The differences themselves are unaffected — only %s ± h is evaluated.\n",
                chain_ok ? "θ̄" : "θ̂", Threads.nthreads(), chain_ok ? "θ̄" : "θ̂")
    all(fd.tuned) ||
        @printf("      %d coordinate(s) off the target band (‖Δm/σ̂‖ outside [%.2f, %.1f]): %s\n",
                count(.!fd.tuned), fd.target_sd / 3, fd.target_sd * 3,
                join([@sprintf("%s %.2f", param_symbol(spec.free[j]), fd.disp_sd[j])
                      for j in 1:d if !fd.tuned[j]], ", "))
    # A single extrapolate has nothing to be checked against.
    let bare = [j for j in 1:d if fd.ok[j] && !any(isfinite, view(fd.stab, :, j))]
        isempty(bare) ||
            @printf("      %d column(s) with a single extrapolate, instability unchecked: %s\n",
                    length(bare), join(param_symbol.(spec.free[bare]), ", "))
    end
    # cond(Ĵ) = cond(G̃)²: beyond double precision pinv drops directions, silently giving
    # an unidentified parameter a finite standard error.
    let sv = svdvals(Diagonal(1.0 ./ σ̂) * Ĝ), tol = size(Ĝ, 2) * eps() * sv[1]^2
        @printf("      cond(G̃)=%.2e ⇒ cond(Ĵ)=%.2e; pinv keeps %d of %d directions\n",
                sv[1] / sv[end], (sv[1] / sv[end])^2, count(>(sqrt(tol)), sv), d)
    end
    # The regression Ĝ on the same points, priced in se(bound) under the bread J⁻¹.
    if Ĝ_reg !== nothing
        sb_r, _ = se_bound_diagonal(Ĝ_reg, spec.W, σ̂)
        sb_f, _ = se_bound_diagonal(Ĝ, spec.W, σ̂)
        @printf("      vs design regression (R² median %.3f): se(bound) median ratio FD/reg = %.2f\n",
                isempty(R2f) ? NaN : median(R2f), median(sb_f ./ sb_r))
    end
else
    @printf("  Ĝ: %d×%d design regression from %d draws, moment R² min=%.3f median=%.3f\n",
            K, d, size(Msel, 2), minimum(R2f), median(R2f))
end

# Table layout. The chain path adds se(chain), its |t| and each coordinate's R̂ (✓ where it
# meets MCMC_RHAT_MAX). The sandwich column is named by its bread: sandw. (P) or J⁻¹.
sw  = chain_ok ? "sandw." : "J⁻¹"
hdr = draws_ok ?
    "  block   param        estimate  se(chain)" * lpad("se($sw)", 11) * "   se(bound) |t|(chain)" *
    lpad("|t|($sw)", 12) * "   |t|(bnd)       R̂" * "  " :
    "  block   param        estimate    se(J⁻¹)   se(bound)  |t|(J⁻¹)   |t|(bnd)"
box_w = textwidth(hdr) + 4              # + room for the trailing RAIL / edge flag
"""
    box_rule(l, r)

Print a horizontal rule of the results box with corner characters `l` and `r`.
"""
box_rule(l, r) = println(l, "═"^box_w, r)
box_title = MCMC_JAC_ONLY ? "Standard errors from Ĵ = Ĝ'WĜ" : "DE-MC Quasi-Posterior"

println()
box_rule("╔", "╗")
println("║  ", box_title, " "^(box_w - 2 - textwidth(box_title)), "║")
box_rule("╠", "╣")
# Chain diagnostics only when a chain ran.
if !MCMC_JAC_ONLY
    @printf("  gens=%d/%d %s  burn=%d  chains=%d  accept=%.3f\n",
            res.gens, res.gens_requested,
            res.aborted                          ?
                (hasproperty(res, :abort_why) ?
                     (res.abort_why === :acceptance ? "(ABORTED: acceptance floor)" :
                      res.abort_why === :drift      ? "(ABORTED: seed drift)"       :
                                                      "(ABORTED)") :
                     "(ABORTED)")                                                  :
            res.gens < res.gens_requested        ? "(criteria met, stopped early)" :
            MCMC_CHECK_EVERY > 0                 ? "(BUDGET EXHAUSTED — criteria not met)" :
                                                   "(no stopping test)",
            res.burn, size(res.chain, 2), res.accept)
    # Drift split; under :flat_t the target has no Jacobian term and dlj is exactly 0.
    _lj0, _lj1 = _USE_LOGJAC ?
        (logjac_box(θ0, spec.free), logjac_box(collect(res.theta_best), spec.free)) :
        (0.0, 0.0)
    dQ, dlj = drift_components(Q_seed, Q_best, _lj0, _lj1)
    if (MCMC_CHECK_EVERY > 0 && MCMC_DRIFT_MAX > 0) || res.aborted
        @printf("  seed drift=%+.1f (abort >%.1f) = %+.1f from Q (ΔQ=%+.4g) %+.1f from log|dθ/dt|\n",
                res.drift, MCMC_DRIFT_MAX, dQ, Q_best - Q_seed, dlj)
    end
    # ΔQ = 1 (one moment moving one sampling SD) is the scale worth re-seeding on.
    if res.aborted && Q_best < Q_seed - 1.0
        @printf("  → the chain beat θ̂ by ΔQ=%.2f, so θ̂ is not the optimum; theta_best is in %s\n",
                Q_seed - Q_best, basename(chain_path(WINDOW, W_SUFFIX)))
    elseif res.aborted && dlj > abs(dQ)
        @printf("  → the climb is the Jacobian term, not Q: θ̂ sits near a box edge where\n")
        @printf("    log|dθ/dt| → −∞. Seed the railed coordinates interior; θ̂ stands.\n")
    end
    if res.n_replaced > 0
        @printf("  stuck-chain replacements: %d, last at g=%d%s\n",
                res.n_replaced, res.last_replace,
                res.last_replace > res.burn ? "  ← INSIDE the retained sample" : "")
    end
    # Worst R̂ over the table (vacancy costs included) and smallest ESS, against the gates.
    let r = filter(isfinite, rhat_all), e = filter(isfinite, ess_all)
        draws_ok && @printf("  R̂ max=%.3f (gate ≤ %.2f)  ESS min=%.0f (gate ≥ %.0f%s)\n",
                            isempty(r) ? NaN : maximum(r), MCMC_RHAT_MAX,
                            isempty(e) ? NaN : minimum(e),
                            MCMC_ESS_MIN > 0 ? MCMC_ESS_MIN : min_ess(d),
                            MCMC_ESS_MIN > 0 ? "" : ", minESS")
    end
    @printf("  target: %s   (space = :%s, MCMC_PRIOR = :%s)\n",
            MCMC_SPACE === :theta ? "−½·g'Wg on the θ box" :
                (_USE_LOGJAC ? "−½·g'Wg + log|dθ/dt|" : "−½·g'Wg"),
            MCMC_SPACE, MCMC_PRIOR)
    box_rule("╠", "╣")
end
println(hdr)
println("  ", "─"^(textwidth(hdr) - 2))
"""
    _abs_t(θ, se) -> Float64

|θ|/se, or NaN when `se` is zero or not finite (an unusable se must not read as infinite
significance).
"""
_abs_t(θ, se) = (isfinite(se) && se > 0.0) ? abs(θ) / se : NaN

"""
    _rail_frac(θ, lb, ub) -> Float64

Distance from `θ` to its nearest bound as a fraction of the box width.
"""
_rail_frac(θ, lb, ub) = min(θ - lb, ub - θ) / (ub - lb)
const RAIL_TOL = 1e-3   # below this the coordinate is on the rail: no two-sided interval

out_csv = mcmc_csv_path(WINDOW, W_SUFFIX; point = MCMC_JAC_ONLY ? :smm : :mcmc)
open(out_csv, "w") do io
    # COLUMN ORDER IS APPEND-ONLY: new columns go at the end so earlier readers still parse.
    #   se_/abs_t_{sandwich|curvature}  named by the bread (P on the chain path, J⁻¹ without)
    #   abs_t_*          |estimate|/se, against the reported estimate (θ̄ or θ̂)
    #   corner_declared  0/1 from CORNER_PARAMS
    #   rail_frac        distance of θ̂ to its nearest bound / box width
    # Every free parameter gets a row, including one on a bound (the LaTeX table drops it).
    sw_col = chain_ok ? "sandwich" : "curvature"
    println(io, "block,name,label,point_estimate,post_mean,se_$(sw_col),se_bound,q025,q500,q975,rhat,ess,edge_frac,spread_growth,se_chain,symbol,corner_declared,abs_t_$(sw_col),abs_t_bound,rail_frac")
    for (k, ps) in enumerate(spec.free)
        dk = draws_ok ?
             [_draw_to_θ(res.draws[k, t], ps.lb, ps.ub) for t in 1:n_kept] :
             Float64[]
        q(p) = isempty(dk) ? NaN : quantile(dk, p)
        pmean = isempty(dk) ? NaN : mean(dk)
        pe = _draw_to_θ(θ0[k], ps.lb, ps.ub)
        edge = blo[k] + bhi[k]
        se_rep = draws_ok ? se_chain[k] : se_sw[k]
        # The reported estimate is θ̄ only on a usable chain (chain_ok, not draws_ok): an
        # aborted run reports θ̂, around which it was seeded, since θ̄ may not solve.
        est_rep = chain_ok ? pmean : pe
        tstat   = se_rep > 0 ? abs(est_rep) / se_rep : NaN
        tstat_s = _abs_t(est_rep, se_sw[k])
        tstat_b = _abs_t(est_rep, se_bnd[k])
        # RAIL ahead of edge: an interval needs an interior point.
        rail   = _rail_frac(pe, ps.lb, ps.ub)
        flag   = rail < RAIL_TOL ? " RAIL" :
                 edge > 0.01     ? " edge" : ""
        if draws_ok
            @printf("  %s%s%9.5f %10.5f %10.5f %11.5f %10.2f %11.2f %10.2f %7.3f%s%s\n",
                    padr(ps.block, 8), padr(param_symbol(ps), 12),
                    est_rep, se_chain[k], se_sw[k], se_bnd[k], tstat, tstat_s, tstat_b,
                    rhat[k], rhat[k] <= MCMC_RHAT_MAX ? " ✓" : "  ", flag)
        else
            @printf("  %s%s%9.5f %10.5f %11.5f %9.2f %10.2f%s\n",
                    padr(ps.block, 8), padr(param_symbol(ps), 12),
                    est_rep, se_sw[k], se_bnd[k], tstat_s, tstat_b, flag)
        end
        @printf(io, "%s,%s,%s,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.6f,%.1f,%.6f,%.6f,%.8f,%s,%d,%.6f,%.6f,%.8e\n",
                ps.block, ps.name, ps.label, pe, pmean,
                se_sw[k], se_bnd[k],
                q(0.025), q(0.500), q(0.975),
                rhat[k], ess[k], edge, sgrow[k], se_chain[k], param_symbol(ps),
                CORNER_FLAG[k], tstat_s, tstat_b, rail)
    end
    # The vacancy costs: θ̂'s values with delta-method errors, and the chain's summary of
    # each draw's costs. They have no box, so no rail or edge.
    for (i, (blk, nm, label, sym)) in enumerate(vacancy_cost_rows(spec))
        dk    = draws_ok ? filter(isfinite, k_draws[i, :]) : Float64[]
        q(p)  = isempty(dk) ? NaN : quantile(dk, p)
        pmean = isempty(dk) ? NaN : mean(dk)
        sech  = isempty(dk) ? NaN : std(dk)
        pe    = k_hat[i]
        est_rep = chain_ok ? pmean : pe
        se_rep  = draws_ok ? sech : se_sw_k[i]
        tstat_s = _abs_t(est_rep, se_sw_k[i])
        tstat_b = _abs_t(est_rep, se_bnd_k[i])
        if draws_ok
            @printf("  %s%s%9.5f %10.5f %10.5f %11.5f %10.2f %11.2f %10.2f %7.3f%s\n",
                    padr(blk, 8), padr(sym, 12), est_rep, sech, se_sw_k[i], se_bnd_k[i],
                    _abs_t(est_rep, se_rep), tstat_s, tstat_b,
                    rhat_k[i], rhat_k[i] <= MCMC_RHAT_MAX ? " ✓" : "  ")
        else
            @printf("  %s%s%9.5f %10.5f %11.5f %9.2f %10.2f\n",
                    padr(blk, 8), padr(sym, 12), est_rep, se_sw_k[i], se_bnd_k[i],
                    tstat_s, tstat_b)
        end
        @printf(io, "%s,%s,%s,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.6f,%.1f,%.6f,%.6f,%.8f,%s,%d,%.6f,%.6f,%.8e\n",
                blk, nm, label, pe, pmean, se_sw_k[i], se_bnd_k[i], q(0.025), q(0.500), q(0.975),
                rhat_k[i], ess_k[i], NaN, NaN, sech, sym, 0, tstat_s, tstat_b, NaN)
    end
end

box_rule("╚", "╝")

# Rails are named here: the FD step is capped at the distance to the nearest bound, so a
# railed coordinate's Ĝ column is measured across a sliver of its box and enters every
# interval through Ĵ.
let rails = [(param_symbol(ps), _rail_frac(_draw_to_θ(θ0[k], ps.lb, ps.ub), ps.lb, ps.ub))
             for (k, ps) in enumerate(spec.free)]
    hit = [r for r in rails if r[2] < RAIL_TOL]
    near = [r for r in rails if RAIL_TOL <= r[2] < 0.01]
    if !isempty(hit)
        @printf("\n  ON A RAIL (%d of %d): %s\n", length(hit), length(rails),
                join([@sprintf("%s %.1e of box", r[1], r[2]) for r in hit], ", "))
        println("  These have no two-sided interval, and their Ĝ columns are measured across")
        println("  that same distance. Pin them in FIX_PARAMS and re-estimate: the remaining")
        println("  coordinates' intervals are read off a Ĵ these columns currently contaminate.")
    end
    isempty(near) || @printf("\n  Within 1%% of a bound: %s — same mechanism, weaker.\n",
                             join([@sprintf("%s %.1e", r[1], r[2]) for r in near], ", "))
end

if draws_ok
    # Q at the posterior mean, the seed and the best visited point. The mean averages over
    # the basin, so its Q normally sits above the seed's.
    Q_mean = smm_objective(θ̄_t, spec)
    @printf("\n  Q(θ̄)=%.4f   Q(θ̂)=%.4f   Q(θ_best)=%.4f\n", Q_mean, Q_seed, Q_best)
else
    if MCMC_JAC_ONLY
        println("\n  se(J⁻¹) assumes uncorrelated moment errors. se(bound) holds for any moment")
        println("  covariance with the same diagonal: report se(bound).")
    else
        @printf("\n  The chain ran %d generations and ABORTED, so its %d draws are a record of\n",
                res.gens, n_kept)
        @printf("  where it went, not a stationary sample: no posterior SD can be read off\n")
        @printf("  them. theta_best is in the chain file and is the run's deliverable.\n")
        @printf("  se(J⁻¹) assumes W = Ω⁻¹, i.e. uncorrelated moment errors. se(bound) assumes\n")
        @printf("  nothing about Ω beyond its diagonal and holds for the true Ω whatever it is, so\n")
        @printf("  it is the column to report. Both differentiate Q, which is only piecewise smooth;\n")
        @printf("  %s\n", MCMC_JAC_METHOD === :fd ?
            "the Richardson instability above is the check on whether Ĝ is a derivative." :
            "the moment R² above is the check on whether the local linear fit holds.")
        @printf("  For a posterior SD to set beside them, re-seed from theta_best and run again:\n")
        @printf("  a chain started at a point it cannot improve on has no drift to abort on.\n")
    end
end

# ========================================================================
# 7. Save chain + moments + Ĝ for plots and post-hoc reweighting
# ========================================================================
# Not under JAC_ONLY, which has no chain and would overwrite a real one with a stub.
chain_jls = chain_path(WINDOW, W_SUFFIX)
MCMC_JAC_ONLY || open(chain_jls, "w") do io
    serialize(io, (chain      = res.chain,
                   draws      = res.draws,
                   # Kept so any later estimator needs no re-run: the log-target of every
                   # draw, the draws overwritten by the stuck-chain rule (not from the
                   # target), and which draws are new states.
                   chain_lp   = res.chain_lp,
                   accepted   = res.accepted,
                   replaced   = res.replaced,
                   n_replaced = res.n_replaced,
                   last_replace = res.last_replace,
                   # What MCMC_INIT = :resume continues from and checks.
                   outcome    = res.outcome,
                   demc_state = res.state,
                   sampler    = SAMPLER,
                   # The reported estimate (mean of the θ-mapped draws) and its SD.
                   theta_mean = θ̄_con,
                   # The vacancy costs of every retained draw (2 × kept, rows U and S), in
                   # the order of `draws`, and at θ̂.
                   k_draws    = k_draws,
                   k_hat      = k_struct(spec, k_hat),
                   # The transform-invariant point estimate, q500 of the results CSV.
                   theta_median = θ̃_med,
                   se_chain   = se_chain,
                   # Diagnostic only, never reported: the best single point visited.
                   theta_best = res.theta_best,
                   params_best = _to_constrained.(res.theta_best,
                                                  [ps.lb for ps in spec.free],
                                                  [ps.ub for ps in spec.free]),
                   Q_best     = Q_best,
                   lp_best    = res.lp_best,
                   aborted    = res.aborted,
                   moments    = Msel,
                   draws_jac  = draws_J,
                   moment_keys = MOM_KEYS,
                   G          = Ĝ,
                   G_R2       = R2,
                   # The route that built Ĝ, the FD diagnostics (nothing under :design), and
                   # the regression Ĝ whenever it was solved.
                   G_method   = MCMC_JAC_METHOD,
                   G_fd       = fd,
                   G_reg      = MCMC_JAC_METHOD === :fd ? Ĝ_reg : nothing,
                   sigma_hat  = σ̂,
                   free       = [(ps.block, ps.name) for ps in spec.free],
                   labels     = [ps.label for ps in spec.free],
                   lb         = [ps.lb for ps in spec.free],
                   ub         = [ps.ub for ps in spec.free],
                   accept     = res.accept,
                   burn       = res.burn,
                   window     = WINDOW,
                   weighting  = WEIGHTING,
                   seed_jls   = seed_jls,
                   # The full spec makes theta_best re-evaluable and usable as a warm start.
                   spec       = spec,
                   provenance = run_provenance(window = WINDOW, w_suffix = W_SUFFIX,
                                               version = ROYSEARCH_VERSION)))
end

# ========================================================================
# 8. The best visited point
# ========================================================================
# Metropolis accepts uphill moves, so the chain can leave basins an optimiser is trapped
# in. With MCMC_CHECKPOINT the best point is also written to its own bundle when it beats
# the last one written by more than PROMOTE_MIN_DQ.
if MCMC_CHECKPOINT && Q_best < Q_ckpt[] - PROMOTE_MIN_DQ
    promote!(collect(res.theta_best), Q_best, Q_ckpt[]; label = "END")
    Q_ckpt[] = Q_best
end

# ========================================================================
# 9. Where the standard errors go
# ========================================================================
# JAC_ONLY adds its standard errors, the vacancy costs' errors and the attracting check to
# the seed SMM bundle, leaving point, stage, provenance and sim as written, and only if
# that file still holds the starting point.
#
# A chain run writes estimate_{window}{W}_mcmc.jls:
#   result         the posterior mean re-solved (the t-space mean if the reported one
#                  does not solve)
#   stage          :mcmc, or :mcmc_aborted (not a stationary sample; no consumer takes it
#                  as an estimate)
#   se             sandwich and bound keyed by the bread, plus chain (the posterior SD);
#                  se_source names the bread
#   mcmc           the diagnostics a footnote needs, and the seed file
#   vacancy_costs  at the posterior mean, with the delta-method errors and the chain summary
#   attracting     the free-entry slope check at the posterior mean
se_names  = (chain_ok ? :sandwich : :curvature, :bound, :chain)
se_source = chain_ok ? Symbol("chain_bread_", MCMC_JAC_METHOD) : Symbol("jacobian_", MCMC_JAC_METHOD)
se_record = NamedTuple{se_names}((se_sw, se_bnd, se_chain))
k_record  = (k  = k_struct(spec, k_hat),
             se = NamedTuple{se_names}((k_struct(spec, se_sw_k), k_struct(spec, se_bnd_k),
                                        k_post === nothing ? nothing : k_post.sd)),
             post = k_post)
if MCMC_JAC_ONLY
    _est = read_bundle(seed_jls)
    if _est !== nothing && _est.result !== nothing &&
       _est.result.theta_opt == bundle.result.theta_opt
        write_bundle(seed_jls; result = _est.result, spec = _est.spec, stage = _est.stage,
                     provenance = _est.provenance, sim = _est.sim, se = se_record,
                     se_source = se_source, mcmc = _est.mcmc,
                     vacancy_costs = k_record, attracting = attract, tag = _est.tag)
        @printf("\n  standard errors → %s\n", basename(seed_jls))
    else
        @printf("\n  %s no longer holds the seed: standard errors not written to it\n",
                basename(seed_jls))
    end
elseif !draws_ok
    println("\n  no retained draws: no chain estimate written")
else
    θ̄_t2, Q̄ = θ̄_t, Q_mean
    # Both are posterior means of the same draws (CH Thm 2): the fallback changes the point
    # the bundle carries, never the estimate the table reports.
    if !isfinite(Q̄) && θ̄_bnd !== θ̄_con
        θ̄_alt = [_to_unconstrained(θ̄_bnd[k], spec.free[k].lb, spec.free[k].ub) for k in 1:d]
        Q̄_alt = smm_objective(θ̄_alt, spec)
        if isfinite(Q̄_alt)
            @printf("\n  the reported mean is infeasible; the bundle falls back to the t-space\n")
            @printf("    mean, Q=%.6f. The reported θ̄ ± se(chain) is unchanged.\n", Q̄_alt)
            θ̄_t2, Q̄ = θ̄_alt, Q̄_alt
        end
    end
    mcmc_jls = estimate_path(WINDOW, W_SUFFIX; point = :mcmc)
    if isfinite(Q̄)
        _stage   = res.aborted ? :mcmc_aborted : :mcmc
        attract_mean = attracting_check(θ̄_t2, spec)
        isempty(attracting_line(attract_mean)) || println("  at θ̄: ", attracting_line(attract_mean))
        write_bundle(mcmc_jls;
                     result = SMMResult(θ̄_t2, _params_to_namedtuple(θ̄_t2, spec), Q̄, false, 0, spec),
                     spec   = spec,
                     stage  = _stage,
                     provenance = run_provenance(window = WINDOW, w_suffix = W_SUFFIX,
                                                 version = ROYSEARCH_VERSION),
                     sim    = spec.sim,
                     se     = se_record,
                     se_source = se_source,
                     vacancy_costs = (k = equilibrium_at(θ̄_t2, spec).k, se = k_record.se,
                                      post = k_post),
                     attracting = attract_mean,
                     mcmc   = (seed       = basename(seed_jls),
                               gens       = res.gens,
                               gens_requested = res.gens_requested,
                               burn       = res.burn,
                               kept       = n_kept,
                               accept     = res.accept,
                               aborted    = res.aborted,
                               abort_why  = hasproperty(res, :abort_why) ? res.abort_why : :none,
                               rhat_max   = all(isnan, rhat_all) ? NaN : maximum(filter(!isnan, rhat_all)),
                               ess_min    = all(isnan, ess_all)  ? NaN : minimum(filter(!isnan, ess_all)),
                               chain_ok   = chain_ok,
                               draws_ok   = draws_ok,
                               prior      = MCMC_PRIOR),
                     # What a table's footnote is written from.
                     tag    = _stage === :mcmc_aborted ?
                              "se = posterior SD of a NON-STATIONARY sample; report R̂ beside it" : "")
        @printf("\n  chain estimate → %s\n", basename(mcmc_jls))
        @printf("    %s\n", bundle_summary(read_bundle(mcmc_jls)))
        if _stage === :mcmc_aborted
            @printf("    the chain ABORTED, so any table printing these standard errors must\n")
            @printf("    also print the abort reason, R̂=%.3f and %d accepted moves — both are\n",
                    all(isnan, rhat_all) ? NaN : maximum(filter(!isnan, rhat_all)),
                    round(Int, res.accept * res.gens * MCMC_N))
            @printf("    in the bundle's mcmc field, so the caveat travels with the number.\n")
        end
    else
        println("\n  posterior mean infeasible (Q=Inf): no chain estimate written")
    end
end

@printf("\nWrote %s\n", out_csv)
MCMC_JAC_ONLY || @printf("       %s\n", chain_jls)
flush(stdout)

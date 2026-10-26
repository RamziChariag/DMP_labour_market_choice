# LMR-MATCHED as of v19.8.0. Every value below is theirs, sourced to their code, so a
# table produced here is produced under their procedure. Deviations are marked DEVIATION.
############################################################
# mcmc/MCMC_main.jl — standalone standard-error script (DE-MC)
#
# Runs AFTER an SMM estimation. It does NOT touch smm_main.jl / smm.jl.
# It loads the saved SMM optimum for (WINDOW, WEIGHTING) and runs a
# Differential-Evolution MCMC (mcmc/demc.jl) on the log quasi-posterior, in the
# unconstrained coordinate t of each parameter, θ = lb + (ub − lb)/(1 + e^(−t)):
#       logπ(t) = −½ · g(θ(t))′W g(θ(t))
# that is, a flat prior on t over |t| ≤ MCMC_T_BOX (MCMC_PRIOR = :flat_t, LMR's
# convention). MCMC_PRIOR = :flat_theta adds log|dθ/dt|, a flat prior on θ; see the
# comment at MCMC_PRIOR below for why :flat_t is the default.
# Under the diagonal weighting, g′Wg is Σ_k ((data_k − model_k)/σ̂_k)², LMR's
# 1/SD² scheme (objective_function_mod.f90), which also fixes the temperature:
# the sampling variances put g′Wg on an n·g′Ω̂_d⁻¹g footing, so the chain's
# dispersion has standard-error units without a separate scale factor.
# The box on t is what makes the target proper.
#
# Standard errors are sandwiches, Var(θ̂) ≈ B Ĝ′W Ω W Ĝ B, with Ĝ = ∂g/∂θ and B the inverse
# curvature of Q/2. On the chain path B is the covariance of the kept draws, P, and Ĝ is
# measured at the reported estimate θ̄; without a usable chain (JAC_ONLY, an aborted run) B is
# J⁻¹ = (Ĝ′WĜ)⁻¹ and Ĝ is measured at θ̂. The sandwich column takes Ω = W⁻¹; since Ω is not
# credibly estimable across four sources at three frequencies, the bound column is the sharp
# upper bound over every Ω consistent with the estimated sampling variances — see
# `se_bound_diagonal`.
#
# It reads one SMM estimate and never writes an SMM file, except that the chain-free pass
# adds its standard errors to the SMM bundle it read (point, stage and sim unchanged):
#
#   seed          baseline: estimate_baseline{W}_smm.jls (both routes); crisis window:
#                 estimate_{window}{W}_smm.jls (JAC_ONLY), or for a chain
#                 estimate_{window}{W}_smm_mcmcbase.jls, which must hold CRISIS_HELD at the
#                 baseline chain's estimate_baseline{W}_mcmc.jls
#   JAC_ONLY      output/estimates/mcmc_results_{window}{W}_smm.csv, and the standard
#                 errors into the seed bundle
#   chain         output/estimates/estimate_{window}{W}_mcmc.jls   the chain's estimate
#                 output/estimates/mcmc_results_{window}{W}_mcmc.csv
#                 output/chains/chain_{window}{W}.jls              chain, draws, moments, Ĝ
#
# Usage (from project root — threads strongly recommended):
#   julia --project=. --threads auto code/mcmc/MCMC_main.jl
############################################################

# ── Paths ───────────────────────────────────────────────────────────────
# HOISTED ABOVE THE version.jl INCLUDE. This file used to live in code/smm/ beside
# version.jl, so `include(joinpath(@__DIR__, "version.jl"))` worked and the path constants
# could come after it. Since v19.6.0 it lives in code/mcmc/ and @__DIR__ no longer holds
# version.jl, so SMM_DIR has to exist before the first include rather than after it.
const MCMC_DIR   = @__DIR__
const CODE_ROOT  = normpath(joinpath(MCMC_DIR, ".."))
const SMM_DIR    = joinpath(CODE_ROOT, "smm")
const SOLVER_DIR = joinpath(CODE_ROOT, "solver")
# paths.jl is the sole definition of PROJECT_ROOT, OUTPUT_DIR and every out_*() accessor.
# It replaces the four constants this block used to declare independently of the six other
# files that declared them too.
include(joinpath(CODE_ROOT, "paths.jl"))
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

# The dense LU / kernel BLAS calls run inside a @threads region, so multi-threaded
# BLAS would oversubscribe every core with the outer thread count.  Pin BLAS to one
# thread and let Base.Threads own the parallelism.
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
# Batch overrides: defaults below are the single-run configuration; a batch sets
# ROYSEARCH_* environment variables so it needs no edit to this file. `env_setting`
# (settings.jl) reads them and records each resolution for the [env] block below.
assert_no_legacy_env()

WINDOW        = env_setting(:WINDOW, :baseline)                 # window to compute SEs for
WEIGHTING     = run_weighting()                            # :diagonal or :equal (moments.jl).
                                  # Selects BOTH the estimation bundle to load and
                                  # the weighting the chain targets, as in smm_main.jl.
                                  # The held-out moment set is taken from the
                                  # bundle, so there is no SKIP_MOMENTS here.

# ── the chain IS the estimator ─────────────────────────────────────────────────
# DE-MC controls (see mcmc/demc.jl). N = 0 ⇒ 2·d chains.
#
# THE PAPER IS THE AUTHORITY FOR EVERY MAGNITUDE HERE, not the replication package. LMR's
# main_mpi.f90 ships chain_count = 95 and chain_length = 500, but those are settings, no more
# authoritative for their published run than the values in this file are for one of ours —
# the archive records what was in the file when it was zipped. Structure comes from their
# code (the acceptance rule, the DE-MC proposal, the outlier rule and its scope); magnitudes
# come from Appendix C p.86:
#
#   "we simulate 100 chains in parallel, each of length 10,000, and use the last 1000
#    elements (pooled over the 100 chains) to obtain parameter estimates and the standard
#    errors"
#
# N = 120 gives 6.0 chains per free parameter at d = 20, against their 6.25 at d = 16. More
# chains is the cheap direction for this estimator: the reported mean and sd are pooled, so
# what matters is the total post-burn draw count and how many chains are still MOVING in the
# retained window — a frozen chain contributes one point replicated.
MCMC_N           = env_setting(:MCMC_N, 120)
# 5,000 rather than their 10,000: measured 33.94 s/generation at N = 95 on 10 threads, so
# ~43 s at N = 120 and ~60 h for this budget. Their 10,000 is 94 h. The estimator does not
# change with the budget; only how far the chain gets.
MCMC_GENS        = env_setting(:MCMC_GENS, 5_000)
# Their retained quantity is ABSOLUTE — the last 1,000 generations — so the faithful fraction
# depends on the budget: 0.8 of 5,000 retains 1,000 generations × 120 chains = 120,000
# draws. The 0.9 that matches their 10,000-generation run would retain 500 generations here,
# half their pooled sample.
MCMC_BURN        = env_setting(:MCMC_BURN, 0.8)
# Proposal geometry. All three are env-overridable: they are the knobs a run is tuned
# on, and a source edit to tune them is a reproduction hazard.
#
# CR — the probability each coordinate IS perturbed, so HIGHER means MORE coordinates
# move at once. Two measured effects pull against each other. The CR mask zeroes a
# random subset of the step, which rotates it off the ridge the population has learned:
# at CR=0.75 the off-ridge component is 0.51 of the along-ridge one, at 0.95 it is 0.15,
# at 1.0 exactly zero. But each coordinate perturbed crosses its own grid boundaries and
# adds its own jump, so ΔQ rises with the count — measured 0.77 at one coordinate
# against 3.69 at all 24, same total step length.
#
# The value is 0.95 as of v19.4.0, and the argument above is not the reason. A
# candidate-by-candidate trace (23,424 proposals, base_fc, 2026-08-27) resolved the two
# effects against each other on the criterion's OWN metric. Writing
# κ = −Δlogπ / (½‖step‖²) with the step measured in the target's conditional scales,
# κ = 1 means the step costs exactly what its length predicts:
#
#     coords moved      1       4       8      11
#     median κ      1.307   1.171   0.938   0.781
#     alignment     0.125   0.513   0.601   0.686
#
# Moving MORE coordinates is CHEAPER per unit length, because the step then points along
# the ridge the population has learned rather than into a coordinate subspace. The gap
# survives stratification on step length (ratio 1.19–1.62 within all four quartiles), so
# it is not a length confound, and it is monotone on ranks (Spearman ρ(nmask, −κ) =
# +0.094, ρ(alignment, −κ) = +0.175; Pearson is near zero only because κ is heavy-tailed,
# sd 7.0 against a median of 1.11). The jump-floor effect the paragraph above describes is
# real but smaller than the rotation it trades against.
#
# Not 1.0: that removes the mask entirely, and with it the chain's only route into a
# single coordinate — the one move type whose acceptance the trace measures at 0.260
# against 0.024 at eleven coordinates.
#
# γ compensates for the realised mask count (2.38/√(2δ·n_updated)), so CR redistributes a
# fixed total step rather than resizing it: the step NORM is invariant to CR (n cancels
# against E‖diff‖²). CR is a direction knob, never a step-size knob.
# LMR main_mpi.f90:127. Their value, not the 0.95 the local ESJD sweep preferred — see the
# DEVIATION note at the end of this block.
MCMC_CR          = env_setting(:MCMC_CR, 1.00)
MCMC_DELTA       = env_setting(:MCMC_DELTA, 2)          # LMR main_mpi.f90:131
# b_mult scales the difference vector ELEMENTWISE, so its off-ridge contribution is
# b_mult·γ·‖diff‖ — proportional to the population spread, and therefore driven by the
# WIDEST coordinate. b_S's width is ~124 in t, giving ‖diff‖ ≈ 177 at stationarity and
# an off-ridge excursion of 0.61 against a measured tolerance of 1e-3. LMR use 1e-2
# (mpi_mcmc_mod.f90:580) with no coordinate remotely that wide. Lower this if acceptance
# decays as the population spreads, which is the signature of this term.
#
# DELIBERATELY NOT MATCHED TO LMR, and this is the one shock setting where matching them
# would be wrong. b_add is already at their 1e-4. b_mult stays 1000x below their 1e-2
# because the argument above is about ‖diff‖, which depends on the WIDEST coordinate's
# width in t — and b_S's ~124 has no counterpart in their parametrisation. Raising it to
# 1e-2 is the first thing to try if acceptance is adequate but the chain under-explores;
# it is the last thing to try if acceptance decays with the spread.
MCMC_B_MULT      = env_setting(:MCMC_B_MULT, 1e-2)      # LMR shock_mult_std, main_mpi.f90:136
# b_add is an absolute isotropic shock and does not scale with the spread. At 1e-4 it is
# 0.1x the off-ridge tolerance, and it is the only mover in generation 1 under :at_seed.
MCMC_B_ADD       = env_setting(:MCMC_B_ADD, 1e-4)       # LMR shock_add_std, main_mpi.f90:137

# How the initial population is built. DE-MC's step size IS the population spread, so
# this choice sets the proposal scale for the whole run and cannot be recovered from
# later: a population wider than the target never contracts, because contraction needs
# accepted moves and an over-wide proposal is rejected.
#
#   :at_seed  every chain starts at θ̂ exactly, as Lise-Meghir-Robin do
#             (mpi_mcmc_mod.f90:268). The additive shock b_add is the only mover in
#             generation 1 and the population grows outward to the target's own scale —
#             the direction DE-MC self-corrects in. Measured on a 25-d target of sd
#             1e-3: acceptance 0.30 and spread 9.5e-4 after 600 generations.
#   :screen   draw MCMC_INIT_SCREEN candidates around θ̂, keep only the ones the solver
#             converges on, rank the survivors by log-target and seed from the middle.
#             Use when :at_seed mixes too slowly; it costs the screening solves up front
#             and errors out if too few candidates converge, which is itself the useful
#             signal that θ̂ sits in a basin too narrow to sample.
#   :resume   continue the run stored in the window's chain file, completed or a snapshot,
#             from its final population. MCMC_GENS stays the TOTAL, so a 5,000-generation
#             file and MCMC_GENS = 7,000 run 2,000 more and write and print what one run of
#             7,000 does. See section 4c.
MCMC_OUTLIER_IQR = 2.0             # Replace a chain scoring more than this many IQRs
                                   # below Q1 with the best chain (LMR
                                   # mpi_mcmc_mod.f90:419). A stuck chain's position
                                   # enters every other chain's difference vector, so one
                                   # dead chain degrades all N. Runs during burn-in only,
                                   # since the replacement is not reversible; the log
                                   # reports the last generation it fired, and any value
                                   # at or past the burn boundary invalidates the
                                   # retained draws. 0 disables.
#
# WHEN it runs — the one place matching LMR carries a cost.
#
# LMR run it EVERY generation with no burn-in gate: mpi_mcmc_mod.f90:421-427 sits in the main
# loop unconditionally. Theirs is also more aggressive than ours —
#   all_population_knl(:,C,:) = all_population_knl(:,best_chain,:)
# overwrites the outlier chain's ENTIRE buffer with a copy of the best chain's, not just its
# current position. Their reported SD is therefore computed over a buffer that can hold
# duplicated chains, which mechanically narrows it. We copy only the current position: same
# purpose (unstick the chain, stop it poisoning every other chain's difference vector) without
# duplicating history into the retained sample. That is a DELIBERATE deviation.
#
# false (LMR-matched) lets it fire throughout. The replacement is not a reversible
# transition, so a retained draw from a generation in which it fired is not a draw from the
# target — which is precisely why `last_replace` is reported. A value past the burn boundary is
# the signal that the reported SD is not a posterior SD. At MCMC_BURN = 0.9 the retained window
# is the last 10%, by which point a healthy population should be free of outliers; that is
# LMR's own reason it is harmless at 10,000 generations.
#
# true (default) confines it to burn-in, keeping the retained sample reversible
# at the cost of no longer matching them. At 4,000 generations with MCMC_BURN = 0.75 the
# retained window opens at generation 3,000, not 9,000, so LMR's reason for firing throughout
# does not carry over.
MCMC_OUTLIER_BURN_ONLY = env_setting(:MCMC_OUTLIER_BURN_ONLY, true)
# :screen, not :at_seed. DE-MC's step IS the population spread, and :at_seed starts it
# at zero — every chain at θ̂ — so the population must grow into the target's scale
# before it can sample. On this objective the feasible set around θ̂ has holes (a line
# scan through σ_S found 17 of 41 nearby points with no equilibrium, interleaved with
# feasible ones), so a fixed-radius cloud would be mostly infeasible. :screen keeps
# only points the solver converges on, shrinking the radius until enough survive, and
# reports the radius it settled on — a small one is itself the finding that the basin
# is narrow.
#
# :at_seed, and the argument above is NOT the reason to prefer :screen — it was written
# before any measurement and overstates the case. LMR run this same DE-MC sampler with
# every chain started at the seed (mpi_mcmc_mod.f90:268,
# `one_population_kn = spread(initial_theta, 2, N)`) and publish standard errors from it,
# so a collapsed start is demonstrably workable and is not what breaks a run here.
#
# What DOES break a run from :at_seed is a seed that is not the mode OF THE SAMPLED
# TARGET. logπ = −Q/2 + logjac_box, so the mode of logπ is not argmin Q: the Jacobian
# term rewards coordinates for leaving the saturated tails. A chain seeded at argmin Q
# therefore starts off-mode and must walk, the population acquires the spread of that
# walk rather than the target's local scale, and acceptance collapses. That is a
# property of the SEED, not of :at_seed. LMR do not have it because their target is
# −Q/2 with no Jacobian, so their Nelder-Mead optimum IS their target's mode.
#
# The fix therefore belongs at the seed or in the target, not here. See
# MCMC_LOGJAC_PRIOR below.
#
# THE PRIOR CONVENTION. :flat_t reproduces LMR's target; :flat_theta was the default
# through v19.3.0.
#
#   :flat_theta   logπ = −Q/2 + logjac_box(t)   — a flat prior on θ, expressed in t.
#   :flat_t       logπ = −Q/2                   — a flat prior on t, the unconstrained
#                                                 coordinate. LMR's convention
#                                                 (mpi_mcmc_mod.f90; their prior is a
#                                                 flat box on t at ±20).
#
# WHY THE CHOICE MATTERS HERE, measured 2026-08-27 at the v19.1.0 base_fc optimum
# (Q = 726.71). At an argmin of Q, dQ/dt = 0, so under :flat_theta the target's gradient
# at the seed is exactly dlogjac/dt = 1 − 2σ(t), component by component, with no solves
# required. That gradient is NOT zero: ‖dlogπ/dt‖ = 2.8821 over the 23 free coordinates,
# median |component| 0.537, and the two at-bound parameters are the two largest
# (b_S: +1.0000, the maximum the derivative can take; δ_S: −0.9860). So under
# :flat_theta the seed is not a stationary point of the sampled target, MCMC_INIT =
# :at_seed starts every chain off-mode, and the population acquires the spread of the
# walk to the mode rather than the target's local scale — which is what collapses
# acceptance. The live v19.2.0 run confirmed the prediction: max logπ climbed +5.49 over
# generations 250→500, 23% of the 23.7-log-unit pure-Jacobian headroom.
#
# Under :flat_t the same gradient is −0.5·dQ/dt, which is zero at argmin Q BY
# DEFINITION. The seed becomes exactly stationary and :at_seed is correct by
# construction — which is why LMR can seed all 95 chains at a Nelder-Mead optimum
# (mpi_mcmc_mod.f90:268) and publish standard errors from it.
#
# WHAT IT COSTS, and this is not a free choice. A flat prior on t is proportional to
# 1/|dθ/dt| in θ units, so it puts MORE mass near the box edges. Reweighting the
# measured 1-D b_S profile from :flat_theta to :flat_t moves its marginal from
# mean 9.20e-04, sd 5.58e-04, q95 1.86e-03 to mean 2.21e-06, sd 4.50e-05, q95 at the
# grid edge — i.e. b_S's interval becomes degenerate on its bound. LMR do not hit this
# because their b is genuinely interior (their printed sd of 0.032 forces a mean above
# 0.001025 on (0,1)), so their criterion has curvature there; ours is flat over
# [0, 2e-3]. Same prior, different criterion.
#
# So the two conventions trade acceptance against the bounded coordinates' intervals,
# and neither dominates. Both are bounded and continuous on Θ, which is what
# Chernozhukov-Hong Assumption 4 requires; the theory does not privilege either. Report
# which one produced a given table.
MCMC_PRIOR       = env_setting(:MCMC_PRIOR, :flat_t)
# Half-width of the prior box on t, matching LMR's prior_lower/upper_bound = ∓20
# (main_mpi.f90:146-147). At the base_fc seed max|t| = 12.46 (b_S), so ±20 leaves 7.5 t-units
# — three measured posterior sd — of room at the tightest coordinate. Inf reproduces the
# unbounded target of every version before 20.0.0.
MCMC_T_BOX       = env_setting(:MCMC_T_BOX, 20.0)
# Which space the chain lives in. :t IS THE SHIPPED DEFAULT AND THE ONLY WORKING VALUE: it
# samples the logistic preimage and is the v20.x estimand, so a plain run with no environment
# overrides reproduces v20.0.0.
#
# :theta is PRESENT BUT OFF, and the design is being KEPT — it is not built yet. The guard below
# refuses to run it, because as it stands it does not merely mis-report, it climbs away from the
# seed: max logπ = -5.178e+05 against -2.447e+02 under :t, dlp = +24.37 at gen 10 (positive),
# and Q(θ̂) = 16,561,006 for the very seed vector that scores 490.55 under :t.
#
# TWO THINGS MUST BE DONE FIRST, and both are known. Neither is a reason to abandon the design.
#   (1) b_add MUST BECOME PER-COORDINATE. It is the ONLY non-scale-free element in the whole
#       proposal, and it is what produced the 16.5-million Q — not the change of space. Note
#       what is already fine: DE-MC's step is γ·(X_r1,k − X_r2,k), drawn from the population's
#       own spread in each coordinate, so it is per-coordinate and region-dependent already;
#       MCMC_B_MULT multiplies that difference vector, so it is scale-free too. MCMC_B_ADD is a
#       single SCALAR added to every coordinate. In t the logistic puts every coordinate at
#       O(1), so one scalar serves all of them. In θ the coordinates span 6.3e-06 (b_S) to 11.1
#       (c) and that same scalar is 16x b_S's entire value while being negligible for c. Give
#       b_add a per-coordinate scale and this objection is answered.
#   (2) THE `constrained = true` FLAG REACHES ONLY logposterior. Every other consumer of the
#       chain vector still reads it as t: Q(θ̂), Q(θ̄), Q(res.theta_best), _Q_from_lp, the Ĝ
#       Jacobian block, and unpack_θ at the two reporting call sites below (search this file
#       for `unpack_θ`). Mechanical, but it is six sites and missing one reports a wrong number
#       rather than failing.
# Until both are done, :theta must not produce a number anyone reports. When they are, delete
# the guard in the version that adopts it — and that version is a MAJOR bump, because the
# estimand changes.
#
# WHY THE DESIGN IS WORTH FINISHING, recorded here so the next attempt need not rediscover it:
# a flat prior on θ is flat on the object the paper interprets, while a flat prior on t asserts
# a priori that each parameter is more likely near its own bounds, by an amount set by the box
# width — not a belief anyone holds. For the logistic map dθ/dt = θ-lb, so a coordinate sitting
# a few 1e-06 above its floor needs hundreds of t-units to span its ΔQ=1 width while the
# chain's t-sd is ~0.04. That shortfall is why b_S's reported standard error is too small on
# base_fc, and it is a live defect in the standard error this version ships.
#
# BUT DO NOT QUOTE A MAGNITUDE FOR IT YET. Earlier notes carried "width 0.0037, 586 t-units, a
# factor of 1.5e+04" for b_S; that is not reproduced. code/tools/width_audit.jl at the base_fc
# posterior mean returns ΔQ=1 up = NaN and ΔQ=1 dn = NaN for b_S at θ̂ = 6.2552e-06 — the
# bracket closes on an infeasible point both ways, so b_S has no measured finite width. The
# measured quantity there is se(chain) = 9.843e-09, against ΔQ=1 widths of 1e-05 to 1e-02 for
# every coordinate that does bracket. The undercount is real; its size is not established, and
# establishing it needs a base point at which b_S brackets.
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
# Path to the criterion-width table used by MCMC_INIT = :widths. Produced by
# code/tools/width_audit.jl. Empty means "the default name for this window".
MCMC_WIDTHS_CSV  = env_setting(:MCMC_WIDTHS_CSV, "")
MCMC_INIT        = env_setting(:MCMC_INIT, :at_seed)
MCMC_INIT_SCREEN = env_setting(:MCMC_INIT_SCREEN, 0)  # candidates for :screen (0 → 20·N)
MCMC_PARALLEL    = true           # thread population over chains (see demc.jl header)
MCMC_SEED        = 20260624
                                  # generations between progress lines. acc, dlp and esjd
                                  # each average over exactly this window, so a very short
                                  # stride makes them noisy (with N chains the finest
                                  # acceptance resolution is 1/N). Env-readable so a smoke
                                  # test can see more than one line without a source edit.
MCMC_PRINT_EVERY = env_setting(:MCMC_PRINT_EVERY, 250)
# Generations between LMR-format blocks (their stat_display_rate, main_mpi.f90:139 = 500).
# Additive to the line above, not a replacement: theirs omits the median proposal's Δlogπ and
# ESJD, which are the two numbers that say whether the step is the right size. 0 disables.
MCMC_LMR_PRINT   = env_setting(:MCMC_LMR_PRINT, 500)

# Fixed budget, no sequential stop — LMR have neither. main_mpi.f90 runs max_iteration to
# completion and nothing inspects R̂ or acceptance; their own convergence hook reads
# "CEHCKING CONVERGENCE / TO BE DONE!!!" (mpi_mcmc_mod.f90:473-475).
#
# 0 here also makes MCMC_MOVES_MIN, MCMC_DRIFT_FLAT and MCMC_ACC_FLOOR inert: they are the
# stop's own thresholds, not independent checks. R̂ and ESS are still COMPUTED and reported
# per coordinate — as a statement about which coordinates are identified, which is what they
# can support on a partly unidentified target, rather than as a gate. Measured on the 18.4.1
# base_fc chain over 16 checkpoints, the old gate (worst R̂ ≤ 1.10 and min ESS ≥ 450) fired at
# 0 of 16, and worst R̂ TRENDED UPWARD with budget (3.17 at g=250 to 5.73 at g=4000), so no
# threshold there is reachable at any budget.
#
# Set MCMC_CHECK_EVERY > 0 to restore the stop.
MCMC_CHECK_EVERY = env_setting(:MCMC_CHECK_EVERY, 0)
#
# THE SEQUENTIAL STOP, as of v19.3.0. MCMC_RHAT_MAX and MCMC_ESS_MIN below are now
# REPORTED rather than gated on. Measured on the 18.4.1 base_fc chain (N=64, G=4000,
# 16 checkpoints): the old gate — worst R̂ ≤ 1.10 AND min ESS ≥ 450 over non-exempt
# coordinates — fired at 0 of 16. Its false-stop rate was zero and so was its TRUE-stop
# rate: it could not terminate a run, so every run paid the full MCMC_GENS. Two reasons,
# both measured:
#   worst R̂ TRENDS UPWARD with budget (3.17 at g=250 to 5.73 at g=4000, whole series in
#     [3.17, 5.73], never near 1.10), so no threshold there is satisfiable at any budget;
#   min ESS is an autocorrelation estimator applied to a series that is 99.4% duplicates
#     — ~750 accepted moves over 64 chains is ~12 per chain — so the number it returns
#     (233 to 1400) is not an effective sample size.
# The replacement gates on accepted moves (the deliverable in its natural unit), on the
# running maximum of the log-target having stopped climbing (so a run cannot stop
# mid-descent, where Cov(chain) would measure the trajectory rather than the curvature),
# and on worst R̂ not having increased (a no-worsening test, not a threshold). All three
# are required at TWO CONSECUTIVE checks. See stop_rule in mcmc_diagnostics.jl.
# Accepted moves required in the retained half. Sized on the deliverable: Cov(chain) is
# a d-dimensional covariance and the usual band for a stable one is 10d–100d independent
# draws, so 100·d is the ceiling worth asking for.
#
# It was 5000, justified in this comment as "217·d, near the top of that band". That
# arithmetic was wrong — 100·d is 2300, so 5000 sat 2.2× ABOVE the band — and the two
# base_fc runs of 2026-08-27/28 show what it cost: the retained-half count plateaus near
# 1000 (43·d, comfortably inside the band) at 975, 995, 1014, 975 across four consecutive
# checks. The gate was therefore rejecting an adequate sample, and no run could stop on
# it. That is the same defect as the R̂/ESS gate this rule was written to replace,
# reintroduced by an unreachable threshold of my own.
# THE ACCURACY GATE, and the reason the stop was rewritten in v21.0.0. Monte Carlo error of
# the reported mean as a fraction of the reported sd, maximised over non-exempt coordinates.
# 0.05 is Flegal-Haran-Jones (2008): the third significant figure of the reported number is
# stable. At N = 95 this sits at R-hat ~ 1.11, but it is stated in the units of the
# DELIVERABLE -- a mean and a standard deviation -- rather than as a convention about chain
# agreement. The gates it replaces (accepted moves, flat drift, worst R-hat not increasing)
# are all satisfiable by a chain that never converges. See mcse_ratio in mcmc_diagnostics.jl.
MCMC_MCSE_TARGET = env_setting(:MCMC_MCSE_TARGET, 0.05)
# ADAPTIVE DE SCALE. ter Braak's gamma = 2.38/sqrt(2*delta*n) is optimal only when the
# population already sits at the target's scale. On the v20.0.0 base_fc run it did not:
# the population reached a median 6.1x the criterion's own per-coordinate widths — and
# ANISOTROPICALLY, spanning ~500x, from 25x too WIDE on λ_S down to 0.05x on b_S, which is
# 20x too NARROW. (b_S's denominator is its ΔQ = 1 width of 5.46 t-units, the flat-then-cliff
# distance rather than a posterior scale, so 0.05 overstates how under-dispersed it is; the
# sign of the error is what matters, and it is opposite to every other coordinate's.)
# Acceptance fell 0.193 -> 0.009 between
# g=250 and g=1000, and the run hit the acceptance floor. A scalar cannot correct a shape
# error, but it does not have to: DE-MC reshapes its own population through ACCEPTED MOVES,
# and at 0.9% acceptance there are too few to reshape anything. Restoring acceptance is what
# unblocks the covariance adaptation that fixes the shape.
# Fixed multiplier on ter Braak's DE scale 2.38/sqrt(2*delta*n_masked). 1.0 is his value
# and LMR's, and it is a TUNING constant rather than a derived one. Measured at the v37
# base_fc point: mean acceptance is 0.455 for a per-coordinate step of 1e-3 moving all 20
# coordinates, and 4e-06 at 1e-2, while the population's own spread settles near 1e-2 — so
# the standard value overshoots the usable band by roughly a factor of 4 on this model.
# Lowering this is one constant; MCMC_GAMMA_ADAPT reaches the same place with feedback.
MCMC_GAMMA_SCALE  = env_setting(:MCMC_GAMMA_SCALE, 1.0)
MCMC_GAMMA_ADAPT  = env_setting(:MCMC_GAMMA_ADAPT, false)
MCMC_GAMMA_TARGET = env_setting(:MCMC_GAMMA_TARGET, 0.234)  # Roberts-Gelman-Gilks optimum
# eta and the cadence are CALIBRATED, not conventional. A closed-loop simulation of the
# braked recursion — the population expanding by the measured identity, acceptance from the
# Roberts-Gelman-Gilks relation calibrated to the run's own l = 1.043 at gamma_scale = 1 —
# reproduces the observed collapse under fixed gamma (acc -> 0.000, step 3.34x optimal) and
# gives, by generation 2000:  eta 1 / every 50 -> acc 0.065;  1 / 25 -> 0.138;
# 2 / 25 -> 0.200, step 1.07x optimal;  3 / 25 -> 0.212, 1.04x. eta = 2 is the more
# conservative of the two that recover, so it is the default.
MCMC_GAMMA_ETA    = env_setting(:MCMC_GAMMA_ETA, 2.0)
MCMC_GAMMA_EVERY  = env_setting(:MCMC_GAMMA_EVERY, 0)       # 0 disables the brake
# Multiplier on the per-coordinate criterion widths at initialisation. 1.0 starts the
# population AT the target scale, which is not over-dispersed: if Q is locally quadratic
# then a ΔQ=1 half-width IS one posterior sd, so between- and within-chain variance start
# equal, R-hat starts at 1, and the convergence test has no power. 2.0 starts R-hat near
# 2.2 and lets it FALL, which is the only direction that terminates: every failed run on
# this model had the population growing, and growth has no fixed point. Contraction does,
# because long steps are rejected while accepted moves are biased short.
MCMC_INIT_DISPERSE = env_setting(:MCMC_INIT_DISPERSE, 2.0)
# Lowered from 2300 (= 100·d) to 230 (= 10·d) in v21.0.0. 100·d sizes a stable FULL
# covariance in d dimensions, which is not the deliverable: the reported object is a mean
# and a per-coordinate sd, and MCMC_MCSE_TARGET prices that accuracy directly. At N = 95,
# burn = 0.9, reaching mcse = 0.05 near generation 670 yields ~640 accepted moves, so the
# old threshold would have REFUSED a stop the accuracy gate had already granted. 10·d
# keeps this as a floor against the degenerate case where nothing moved at all.
# Inert while MCMC_CHECK_EVERY = 0. Left at its own justified value rather than zeroed, so
# one switch turns the stop off and each threshold keeps its reasoning for when it is on.
MCMC_MOVES_MIN   = env_setting(:MCMC_MOVES_MIN, 230)
# DRIFT_FLAT in log units. Under MCMC_PRIOR = :flat_t, logπ = −Q/2 exactly, so a climb of
# L log units IS ΔQ = 2L — and paired ΔQ only clears the grids' disagreement above
# |ΔQ| ≈ 2. So 1.0 asks the run to certify the smallest difference the criterion can
# actually resolve. It was 0.5, anchored on PROMOTE_MIN_DQ, which was the wrong anchor:
# that constant governs whether to write a best visited point, not what the objective can
# resolve.
MCMC_DRIFT_FLAT  = env_setting(:MCMC_DRIFT_FLAT, 1.0)   # log units; 1.0 ⇒ ΔQ = 2
# Restored to 0.02 after being zeroed to match LMR, who have no acceptance test. That was
# the wrong trade: this fires only when the chain has stopped moving, and it ends a dead
# 10-hour run at the first check rather than the last -- cheap insurance on the resource
# that is actually binding here.
MCMC_ACC_FLOOR   = env_setting(:MCMC_ACC_FLOOR, 0.02)   # abort-and-diagnose below this, two checks running
# 1.10 is not a borrowed convention here — at N = 95 it IS the Monte-Carlo-error criterion
# for the object this script reports. The estimator is the mean and SD of the pooled draws,
# so the defensible stop is MCSE(mean) small relative to the reported SD (Flegal, Haran &
# Jones 2008). With N independent chains MCSE/se ≈ sqrt(B/W)/sqrt(N) and R̂² ≈ 1 + B/W, so
# MCSE/se = 0.05 lands at R̂ = 1.112. Reading it the other way: R̂ = 8.50, as the failed
# v19.8.0 run reached, is MCSE/se = 0.87 — the Monte Carlo error is 87% of the number being
# reported, which is why that run's standard errors meant nothing.
MCMC_RHAT_MAX    = 1.10            # Gelman et al. (2004) accept ≤1.1. Tighter values are
                                   # out of reach for DE-MC inside this budget: measured on
                                   # an isotropic Gaussian at d = 25 (the easiest target
                                   # this sampler will face), 80k solves gives R̂ ≈ 1.10 and
                                   # 156k gives ≈1.09, while 1.02 needs ~600k. Report the
                                   # attained R̂ rather than claiming a threshold not met.
# Disabled. The abort assumes a climbing running maximum means the seed is not the
# optimum and the chain is walking toward it. That inference fails on an objective
# with unidentified directions: the basin holds many near-equivalent points, so ANY
# chain that spreads far enough to measure a width will find better ones. Aborting
# then terminates exactly the runs that would produce a standard error. The drift is
# still computed and reported in the header, where it belongs — as a diagnostic.
MCMC_DRIFT_MAX   = 0.0             # was 25.0: abort if the running max climbs this far
                                   # above the seed. A stationary chain fluctuates within
                                   # O(d/2); the box Jacobian can shift the mode by at most
                                   # logjac_bound(free) − logjac_box(θ̂) (≈15 units for these
                                   # boxes), so a larger drift means the seed is not the
                                   # optimum of the criterion being sampled and Cov(chain)
                                   # would measure the walk toward it. 0.0 disables.
# Off by default. true writes each new best point the chain visits to
# estimate_{window}{W}_mcmcbest.jls (stage :mcmc_checkpoint) as it finds it, and governs
# that file only: it never writes an SMM or a chain-estimate bundle, the chain is
# snapshotted to output/chains/ every MCMC_LMR_PRINT generations whatever its value, and the
# best visited point is serialised into the chain file as theta_best/params_best either way.
MCMC_CHECKPOINT  = env_setting(:MCMC_CHECKPOINT, false)
                                   # Improvement required to write a new best point. Anchored on what
                                   # the criterion can RESOLVE, not on what it can represent: the
                                   # grid half-range at the reached point is ±3.32 in Q and paired
                                   # differences clear the discretisation above |ΔQ| ≈ 2, which is
                                   # the same quantity MCMC_DRIFT_FLAT = 1.0 log units encodes.
                                   #
                                   # It was 1e-4, reasoned from one moment moving by one sampling
                                   # standard error. That is the resolution of the MOMENTS, not of
                                   # the objective they enter, and at 1e-4 a live chain promotes on
                                   # essentially every generation: the base_fc run at CR = 0.95
                                   # recorded ΔQ = 0.13, 0.05, 0.07 at generations 1, 2 and 5, each
                                   # 500–1300× the threshold and none of them distinguishable from
                                   # zero. That is 4000 full bundle serialisations and 4000 log
                                   # lines bought for nothing.
                                   #
                                   # The checkpoint exists so an interrupted run leaves the best
                                   # REAL point on disk. An unresolvable improvement is not one.
PROMOTE_MIN_DQ   = 2.0
MCMC_JAC_METHOD  = env_setting(:MCMC_JAC_METHOD, :fd)  # How Ĝ = ∂g/∂θ is built.
                                   #   :fd     per-coordinate central differences, step chosen
                                   #           from the moments' own response and refined by
                                   #           Richardson (jacobian_adaptive_fd)
                                   #   :design least squares on a cloud of radius
                                   #           rel_step·(ub−lb) (jacobian_from_draws)
                                   # :design makes the derivative depend on the SEARCH BOX and
                                   # cannot serve coordinates whose sensitivities differ by
                                   # orders of magnitude: on base_covid at the shipped 0.05 its
                                   # per-moment R² had median 0.771 with 25 of 35 below 0.9, so
                                   # it was fitting curvature, not measuring a slope. Kept
                                   # because it is what every bundle before v35 used.
MCMC_JAC_ONLY    = env_setting(:MCMC_JAC_ONLY, false) # true skips the chain (default false): estimate Ĵ = Ĝ'WĜ from a local
                                   # design around the seed (≈10·d solves) instead of from
                                   # Cov(chain) (N·gens solves). CH Theorem 4 admits either.
                                   # The trade is that the reported quantile columns need the
                                   # chain, so they are omitted in this mode.
MCMC_ESS_MIN     = env_setting(:MCMC_ESS_MIN, 450.0)
                                   # What the REPORTED numbers need, not the joint-volume
                                   # floor. minESS(25) at ε = 0.20 is ≈540 and at ε = 0.10
                                   # ≈2159; those size the posterior-mean confidence VOLUME
                                   # and cost 600k–1.6M solves. Set 0.0 to use minESS.
                                   #
                                   # RAISED 250 → 450 in v19.2.0, because what is reported
                                   # changed. 250 was sized for a STANDARD ERROR: at ESS 250
                                   # an se has relative MC error 4.5%, two stable figures,
                                   # and that reasoning is still correct for an se. But with
                                   # diagonal W the chain's se is not a valid CI half-width
                                   # (Chernozhukov–Hong Thm 3 needs W = Ω⁻¹; mcmc_diagnostics
                                   # states outright that diagonal W fails it), so the
                                   # deliverable is the QUANTILE PAIR, and a quantile is
                                   # dearer than an sd. From MCSE(q_p) =
                                   # √(p(1−p))/f(F⁻¹(p)) · sd/√ESS, whose constant is 2.113
                                   # at p = 0.05, ESS 450 is exactly MCSE(q05) ≤ 0.10·sd:
                                   # each interval endpoint precise to a tenth of the width
                                   # it reports.
                                   #
                                   # Report q05/q95, not q025/q975. The constant is 2.671 at
                                   # p = 0.025, and required ESS scales as the SQUARE of the
                                   # constant, so the tighter tail costs
                                   # (2.671/2.113)² − 1 = 59.8% more ESS for the same
                                   # relative precision — 714 against 447 at MCSE ≤ 0.10·sd.
                                   # (An earlier revision of this comment said 37%, which is
                                   # neither the squared ratio nor the unsquared 26.4%; it
                                   # understated the cost of the design choice it was cited
                                   # to justify.) With 35 moments and 21 free parameters the
                                   # 2.5% tail is also the least trustworthy part of the
                                   # estimate, so the cheaper pair is the better report.
                                   #
                                   # This is affordable only because the gate is now
                                   # PER-COORDINATE with automatic exemptions
                                   # (exempt_coordinates, mcmc_diagnostics.jl). Under the old
                                   # minimum(ess) gate, raising the threshold would have made
                                   # an already-unsatisfiable test more unsatisfiable: one
                                   # frozen or railed coordinate held the run hostage, the
                                   # sequential stop never fired, and every run paid its full
                                   # MCMC_GENS budget whatever the reported numbers had done.
# Screen preconditioning (v17.2). The :screen radius is per coordinate, set to a
# fraction of each parameter's own posterior width in UNCONSTRAINED units. Err narrow:
# DE-MC contracts a too-narrow population readily but cannot contract a too-wide one,
# so the failure is asymmetric and the safe side is inside the target.
MCMC_SCREEN_FRAC  = env_setting(:MCMC_SCREEN_FRAC, 0.3)   # start at 0.3·sd and grow
MCMC_SCREEN_CAP   = env_setting(:MCMC_SCREEN_CAP, 1.0)    # required: a railed coordinate's width is
                                  # 124 in t, and the logit clamp saturates near ±18.42,
                                  # so an uncapped scale there draws saturated corners.
MCMC_SCREEN_FLOOR = env_setting(:MCMC_SCREEN_FLOOR, 1e-3) # guard a zero/absent width
MCMC_JAC_DRAWS   = 600            # thinned retained draws re-solved to store the
                                  # moment vector, from which Ĝ is regressed
# Which parameters are AT A CORNER, declared by hand exactly as SKIP_MOMENTS is in
# smm_main.jl and for the same reason: whether an estimate sitting on a bound is an
# acceptable economic zero or a specification problem is not a decision the code should
# take, and a threshold that decided it automatically would be one more number to defend.
# A coordinate on a rail has no two-sided interval whatever the curvature says, so this
# marks the rows a table must footnote or leave blank.
#
# PURELY A LABEL. Nothing is blanked, suppressed, recomputed or reweighted: the flag rides
# in the results CSV's `corner_declared` column and in the banner below, and every other
# value in the file is what it would be with the list empty. `edge_frac` in that same CSV
# is the measurement to choose from.
#
# Keys may be spelled any of the three ways this project spells a parameter (see
# `_corner_declared` below). Override with a comma-separated list:
#   ROYSEARCH_CORNER_PARAMS="alpha_U,delta_S"
CORNER_PARAMS    = env_setting(:CORNER_PARAMS, Symbol[])

derived_dir = joinpath(PROJECT_ROOT, "data", "derived")

W_SUFFIX = weighting_suffix(WEIGHTING)

@printf("Window: %s   weighting: %s\n", WINDOW, W_SUFFIX)
print_env_settings()

# ========================================================================
# 1. Load the SMM estimate (point estimate + spec) that seeds the run
# ========================================================================
# Every route starts from the window's SMM estimate except a chain on a crisis window,
# which starts from the crisis SMM estimate holding CRISIS_HELD at the baseline chain's
# estimate (smm_main.jl with CRISIS_BASE = :mcmc): the crisis chain is seeded from the
# baseline chain, the baseline chain from the baseline SMM estimate.
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
    # The baseline chain's estimate may have been re-run since the crisis SMM estimate was
    # made from it, so the parameters held at it are compared rather than trusted, at the
    # crisis window's period of the baseline.
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
# 2. Rebuild the spec with the SAME weighting the estimation used. Everything
#    else (free set, fixed η/r/ν/φ, moments, grids, sim) is taken verbatim from
#    the estimation spec, so θ0 is valid and the model is solved identically.
# ========================================================================
# Held-out moments come from the bundle, so they cannot disagree with the run that produced
# the seed. The baseline is rebuilt period by period, each on its own data window's weights.
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
spec = is_joint(spec0) ?
    _spec_with_init(joint_spec([chain_spec(s, w) for (w, s, _) in period_list(spec0)],
                               BASELINE_PERIODS), pack_θ(spec0.free)) :
    chain_spec(spec0, WINDOW)
@assert length(θ0) == length(spec.free) "seed length ($(length(θ0))) ≠ free params " *
    "($(length(spec.free))) — bundle and spec rebuild disagree."

MOM_KEYS = active_moment_keys(spec)
K, d     = length(MOM_KEYS), length(spec.free)
NK       = length(vacancy_cost_rows(spec))      # the vacancy costs, two per period

# σ̂ over the active moments, in the SAME order as MOM_KEYS and spec.W. Needed for
# the Ω-free bound; available only under the diagonal-σ weighting.
σ̂ = WEIGHTING === :diagonal ? [1.0 / sqrt(spec.W[i, i]) for i in 1:K] : fill(NaN, K)

# CORNER_PARAMS resolved to one flag per free parameter, HERE — immediately after the free
# set is known — so a misspelled key fails at startup instead of after ~600 solves.
#
# Three spellings are accepted, because all three are in use in this project's own files:
# the display symbol the tables carry (`param_symbol`, :b_S), the ParamSpec name (:bS), and
# the ASCII configuration key of the FIX_PARAMS convention (`_DEFAULT_PARAM_KEY`, :alpha_U,
# :skl_bet). A hand-edited list then does not depend on which file the reader had open.
_corner_spellings(ps::ParamSpec) =
    Symbol[Symbol(param_symbol(ps)), ps.name,
           get(_DEFAULT_PARAM_KEY, (ps.block, ps.name), ps.name)]
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
# 3. Log quasi-posterior, −½·g'Wg plus a prior term set by MCMC_PRIOR (see the
#    block above for the measurement that motivates the switch and what each
#    convention costs). Infeasible θ → Inf → −Inf → the proposal is rejected.
#
#    ONE FUNCTION, not two, and the flag is read once into a Bool rather than
#    per call: a closure that branched on a Symbol 256,000 times would put a
#    dynamic dispatch inside the hot loop for a decision that cannot change
#    mid-run.
# ========================================================================
# Bundles always store t, whatever space the chain runs in, so that a bundle written by one
# space stays loadable by the other. Convert once here; everything downstream — the seed, the
# population, the draws, the reported mean — is then in the sampled space consistently.
if MCMC_SPACE === :theta
    θ0 = [_to_constrained(θ0[k], spec.free[k].lb, spec.free[k].ub) for k in eachindex(θ0)]
end
# ── init = :widths: the criterion's own ΔQ=1 half-widths ─────────────────────────────────
# These are θ-space displacements, so the mode is only meaningful when the chain samples θ.
# Read here rather than computed because the audit costs ~600 solves and belongs in a tool
# that also serves as the pre-table diagnostic: for every coordinate, a (ΔQ=1 width)/se ratio
# far from 1 says the reported standard error is not describing the criterion.
#
# UNREACHABLE AS SHIPPED, and deliberately so. It requires MCMC_SPACE = :theta, which the guard
# near the top of this file refuses to run, so this block cannot execute in v20.0.0 — the two
# checks close a loop rather than contradict each other. It is left in place because the audit
# CSV it reads is a useful diagnostic in its own right (code/tools/width_audit.jl, which is
# standalone and does work).
#
# It also has its OWN unfixed bug, separate from anything in the :theta design. On base_fc it
# errors with "brackets neither direction for coordinate(s) [7] [skilled outside flow b_S]"
# even though that row of the audit CSV carries wQ1_up = 0.003665 and only wQ1_dn is NaN — and
# NaN is the correct entry there, since b_S sits at its floor and the audit cannot bracket
# downward. So the min(up, dn) guard below is misreading one of the two columns; a one-sided
# bracket is the normal case for a floor parameter and must be accepted, not rejected. Fix this
# when the :theta design is settled, not before: it is not reachable until then.
_init_widths    = nothing
_init_widths_dn = nothing
if MCMC_INIT === :widths
    # The :theta requirement this block used to carry has been dropped. It was there because
    # the audit measures widths in θ, and :theta samples θ directly. But the widths can be
    # carried into :t exactly — see below — and :theta is not currently usable (b_add is a
    # single scalar across coordinates spanning 6.3e-06 to 11.1), so requiring it made the
    # only over-dispersed initialisation unreachable from the shipped configuration.
    wcsv = isempty(MCMC_WIDTHS_CSV) ?
           joinpath(out_logs(), "width_audit_$(WINDOW)$(W_SUFFIX).csv") : MCMC_WIDTHS_CSV
    isfile(wcsv) ||
        error("MCMC_INIT = :widths needs $(wcsv); produce it with\n" *
              "  ROYSEARCH_WINDOW=$(WINDOW) julia --project=. code/tools/width_audit.jl")
    wdf = CSV.read(wcsv, DataFrame)
    nrow(wdf) == length(spec.free) ||
        error("$(basename(wcsv)) has $(nrow(wdf)) rows, expected $(length(spec.free)); " *
              "it was written for a different free-parameter set.")
    # PER-COORDINATE HALF-WIDTHS IN THE SAMPLED SPACE, kept SEPARATE by direction.
    #
    # Two changes from the previous single symmetric width, both of which matter.
    #
    # (1) Up and down are not interchangeable. A coordinate resting on a floor — b_S, b_T,
    #     ξ_S — has a bracketable width upward and none downward. Collapsing the pair to
    #     their minimum gave such a coordinate the width of the side that does not exist, and
    #     a symmetric draw then threw half its proposals into infeasibility, exhausted the
    #     attempt budget and dropped those chains back onto the seed: the coordinates most in
    #     need of dispersion were the ones that received none. `hit_infeasible_*` marks a
    #     direction the audit could only bracket by running into an infeasible point; that
    #     side gets a 1%-of-box fallback rather than a width the solver cannot honour.
    #
    # (2) The transform is applied to the ENDPOINTS, not through a local derivative. In :t
    #     space the two are nowhere near proportional: dθ/dt for b_S at the estimate is
    #     6.25e-06 while its ΔQ=1 width spans 6.4 t-units, so a local-derivative conversion is
    #     wrong by three orders of magnitude — an error made and retracted earlier in this
    #     project. Transforming the endpoints is exact, and it is precisely what makes each
    #     coordinate disperse on its own terms: b_S opens a 6.4-unit initial spread because
    #     that is what its criterion width IS in the space being sampled, while a
    #     well-resolved coordinate opens a narrow one. No scale normalisation is imposed.
    _up = Float64[]; _dn = Float64[]
    for r in eachrow(wdf)
        lbk = Float64(r.lb); ubk = Float64(r.ub); thk = Float64(r.theta)
        span = ubk - lbk
        # `hit_infeasible_*` marks a bracket that terminated at the FEASIBILITY boundary
        # rather than at ΔQ = 1. An earlier version read that as "this width is unusable" and
        # substituted 1% of the box, which for a_ℓ replaced a measured 8.7e-04 with 0.079 —
        # ninety times larger. A calibration sweep then showed the joint ΔQ overshooting the
        # quadratic prediction by a CONSTANT factor of ~350 at every multiplier, which is the
        # signature of a handful of coordinates being scaled 100× too wide rather than of any
        # property of the criterion. The flag is not an invalidity marker: a bracket that
        # stopped at infeasibility is a TIGHTER scale bound than ΔQ = 1, because dispersion
        # cannot cross it anyway. So use the measured width whenever it is finite, fall back
        # to the other direction, and only then to a small fraction of the distance to the
        # near bound — never to a fraction of the whole box.
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
        # Built before the message: a join with a quoted separator cannot be nested inside a
        # string interpolation, which is what made this line unparseable on first write.
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

# In :theta space a flat box prior on θ IS the flat-on-θ target, so there is no Jacobian term
# to add — adding one would tilt the target a second time. MCMC_PRIOR therefore only selects
# the estimand in :t space, where :flat_theta reproduces the same density this space samples
# directly. Forced rather than errored so the shipped MCMC_PRIOR = :flat_t default keeps
# working unchanged.
const _USE_LOGJAC = (MCMC_SPACE === :t) && (MCMC_PRIOR === :flat_theta)
MCMC_PRIOR in (:flat_theta, :flat_t) ||
    error("MCMC_PRIOR = $(MCMC_PRIOR) is not recognised; use :flat_theta or :flat_t.")

# The vacancy costs at every point the target is evaluated, keyed by the point's hash: a
# draw is a copy of an evaluated point, so its costs are read back here instead of re-solved.
const _AT_POINT = Dict{UInt64,Vector{Float64}}()
const _AT_LOCK  = ReentrantLock()
_at_point(θ) = lock(() -> get(_AT_POINT, hash(collect(float.(θ))), fill(NaN, NK)), _AT_LOCK)

function logposterior(θ)
    # The prior box on the unconstrained parameter, tested BEFORE the solve: outside it the
    # prior mass is zero, so there is nothing to compute.
    #
    # This box is what makes the target PROPER, and without it there is no posterior at all.
    # θ = lb + (ub−lb)·σ(t) saturates as |t| → ∞, so Q stops changing and −Q/2 is
    # asymptotically FLAT in every coordinate — measured on this model at the base_fc
    # optimum, Q agrees to ten significant figures between t₀+40 and t₀+80, and the whole
    # half-line t_bS ≤ t₀−5 sits within 0.004 of the optimum. Unbounded, the quasi-posterior
    # therefore has infinite mass in all 23 directions: the population diffuses forever
    # because there is no density for it to converge to. MCMC_T_BOX = Inf restores that
    # behaviour for reproducing pre-20.0.0 numbers.
    #
    # WHICH support depends on MCMC_SPACE: the ±MCMC_T_BOX box on t under the shipped :t
    # default, or the economic box [lb, ub] on θ under :theta. The propriety argument above
    # is the same either way; only the coordinates the box is stated in differ.
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

# The inverse map, used wherever a stored logπ has to be turned back into a Q. Keeping
# it beside logposterior is deliberate: these two must agree, and a Q recovered under
# the wrong convention is off by logjac(θ) — 46 units at the current seed, which would
# silently corrupt every checkpoint and every drift split.
_Q_from_lp(θ, lp) = _USE_LOGJAC ?
    -2.0 * (lp - logjac_box(collect(float.(θ)), spec.free)) : -2.0 * lp

# ========================================================================
# 4. Run DE-MC
# ========================================================================
# run_demc prints its own d/N/gens/δ/CR/γ/init header, so the summary block below
# does not repeat those. In MCMC_JAC_ONLY mode the chain is skipped and `res` is a
# one-generation stand-in, so the downstream code has the same shape either way.
# ------------------------------------------------------------------------
# MCMC_CHECKPOINT: the best visited point, in a file of its own
# ------------------------------------------------------------------------
# Q at the seed, computed once: the callback needs it, the drift decomposition
# reuses it, and Q_ckpt tracks the best point written so far.  θ carries the
# tightnesses, so every point, θ̂ and each finite-difference step, is one equilibrium and
# there is no branch to hold.
seed_eq    = equilibrium_at(θ0, spec)
Q_seed     = seed_eq.Q
Q_ckpt     = Ref(Q_seed)
best_jls   = mcmcbest_path(WINDOW, W_SUFFIX)

"""
    promote!(θ, Q_new, Q_old; label) -> nothing

Write `θ` to estimate_<window><W>_mcmcbest.jls, stage `:mcmc_checkpoint`: a record of
where the chain went, never an estimate, and never written into an SMM or chain-estimate
bundle. converged = false because the point came from a sampler, not from an optimiser
meeting a stopping rule. Callers guarantee `Q_new < Q_old`.
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

# Live checkpoint. Raising MCMC_DRIFT_MAX lets the chain keep climbing instead of
# stopping at the first 25 log units, which makes the run long enough that finishing is
# no longer guaranteed — an interrupt, or a kill, would otherwise discard every point it
# found. This writes each new incumbent as it appears, so the best point survives
# regardless of how the run ends. Q is recovered from logπ = −Q/2 + logjac rather than
# re-solved, so the callback costs nothing.

"""
    snapshot_chain(ch, lp, acc, out, rep, g, state)

Write the draws so far to the chain path, atomically.

WHY THIS EXISTS. Before v19.8.0 the draw arrays reached disk exactly once, in section 7,
after run_demc returned AND after ~600 further solves for Ĝ and the diagnostics. A run
killed at generation 9 000 of 10 000 therefore left nothing at all: MCMC_CHECKPOINT saves
the best POINT, never the chain. At the shipped budget that is a 10.5-hour exposure, and it
is the same class of loss that cost a base_covid run at v18.4.0.

Written to a temp file and renamed, so a kill mid-write leaves the previous snapshot intact
rather than a truncated file where the next reader looks. Carries `spec` so a snapshot is
self-contained: every estimator computable from the finished run is computable from a
snapshot of it, over the generations it covers.

`partial = true` marks it as a snapshot rather than a completed run; section 7 overwrites
the same path at the end with the full bundle, including the Ĝ/moment fields a snapshot
cannot have because they are computed afterwards. `outcome`, `demc_state` (the sampler's
state, RNG included) and `SAMPLER` are what MCMC_INIT = :resume continues from and checks.
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

function checkpoint_best(θ, lp, g)
    Q = _Q_from_lp(θ, lp)
    Q < Q_ckpt[] - PROMOTE_MIN_DQ || return nothing
    promote!(collect(float.(θ)), Q, Q_ckpt[]; label = @sprintf("CHECKPOINT g=%d", g))
    Q_ckpt[] = Q
    return nothing
end

# ------------------------------------------------------------------------
# 4b. Screen preconditioning: the per-coordinate scale for init = :screen.
#
# v17.1 computed Ĝ AFTER the chain, so the per-coordinate widths were unavailable
# where the screen needed them. When a screen scale is required, build the local
# design and Ĝ FIRST and reuse it below.
#
# Ĝ is used here as a PRECONDITIONER, not as a derivative. Gate M7 retires it as
# ∂m/∂θ — two independent Jacobians agree to cosine > 0.9 on only 4 of 24 columns —
# but a proposal scale needs the order of magnitude of each coordinate's width, not a
# correct slope, and that Ĝ does supply.
#
# The scale is the JOINT width se(J⁻¹) rather than an own-curvature width: the chain's
# marginal for a coordinate is the joint one. For μ_U they differ by a factor of 66
# (0.0028 against 0.187 in t), so own-curvature would start the population far too
# narrow.
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

    # se_curv is in CONSTRAINED units; the screen draws in unconstrained t. The box map
    # is θ = lb + (ub-lb)·σ(t), so dθ/dt = (ub-lb)·σ(t)·(1-σ(t)) evaluated at the
    # unconstrained seed.
    # Wrapped in a function so the loop counters are function-local: at top level a bare
    # `for` body cannot assign to an outer binding under Julia's soft scope rules.
    function _screen_scale(θ0v, free, se_curv, frac, floor_, cap_)
        d_free = length(free)
        scale = Vector{Float64}(undef, d_free)
        width = Vector{Float64}(undef, d_free)   # UNCLAMPED se_t, for the check below
        n_cap = 0; n_flr = 0; n_fb = 0
        for k in 1:d_free
            f  = free[k]
            # θ0v is the UNCONSTRAINED seed t, which is what logposterior and the screen
            # both work in. The box map is θ = lb + (ub-lb)·σ(t), so evaluate the
            # logistic at t rather than treating t as if it were θ:
            #     u = σ(t) = 1/(1+exp(-t)),   dθ/dt = (ub-lb)·u·(1-u)
            # Reading u as (t-lb)/(ub-lb) puts u outside [0,1] for most coordinates and
            # makes dθ/dt non-positive, which is what produced 17/24 fallbacks.
            u  = 1.0 / (1.0 + exp(-θ0v[k]))
            dθ = (f.ub - f.lb) * u * (1 - u)
            se_t = (isfinite(se_curv[k]) && se_curv[k] > 0 && dθ > 0) ?
                       se_curv[k] / dθ : NaN
            if !isfinite(se_t)
                # No usable width: fall back to this coordinate's own box scale in t
                # rather than a shared constant, so the fallback still respects it.
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
    # The FLOOR (and the CAP, for a coordinate whose width sits below it) is where genuine
    # over-dispersion arises: it raises the scale of a coordinate whose posterior is
    # narrower, and that coordinate then starts wider than its own target. DE-MC cannot
    # contract those directions, so their reported SD is biased upward.
    n_over = count(k -> init_scale[k] > init_width[k], 1:d_free)
    n_over > 0 && @printf("  WARNING: %d/%d coordinates have scale > own posterior width \
(floor/cap raised them); DE-MC cannot contract these and their SD is biased UP.\n",
                          n_over, d_free)
    n_fb > d_free ÷ 4 && @printf("  WARNING: %d/%d coordinates had no usable width — Ĝ may be bad.\n",
                                 n_fb, d_free)
    flush(stdout)
end

# ------------------------------------------------------------------------
# 4c. MCMC_INIT = :resume: the stored run this one continues.
#
# MCMC_GENS stays the total. The run continues the one in the window's chain file, a
# completed run or a snapshot alike, after its last generation, and writes and prints what one
# run of MCMC_GENS generations would. It continues only the same chain, so what defines the
# target (window, free parameters and their boxes, pinned values, active moments, their data
# values, W) and the sampler (N and SAMPLER) is compared first and every difference is named.
# A file written before v61.3.0 records no SAMPLER; for it, MCMC_PRIOR and MCMC_T_BOX are
# checked only through the re-evaluation below, and MCMC_CR and MCMC_DELTA not at all.
#
# Then the stored final population is re-evaluated. Its log-target must equal the stored one
# bit for bit: a solve is the same whether run alone or nested in threaded work, and its
# reductions are maxima, which no grouping changes, so any difference means the code or the
# data no longer define the target the stored run sampled.
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
    # The sampler state of a file written before v61.3.0 is what its final bundle records; the
    # rest takes fresh values in run_demc, the RNG replayed from MCMC_SEED.
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
               # The sequential stop. rhat_max/ess_min above are now REPORTED, not gated
               # on. Forwarded explicitly: the four SA settings that were computed here
               # and never passed cost four versions of silently-defaulted behaviour, and
               # check_forwarding.jl exists because of it.
               moves_min = MCMC_MOVES_MIN, drift_flat = MCMC_DRIFT_FLAT,
               acc_floor = MCMC_ACC_FLOOR, mcse_target = MCMC_MCSE_TARGET,
               gamma_adapt = MCMC_GAMMA_ADAPT, gamma_target = MCMC_GAMMA_TARGET,
               gamma_eta = MCMC_GAMMA_ETA, gamma_every = MCMC_GAMMA_EVERY,
               # For the convergence gate's at-a-bound exemption test, which has to
               # measure pile-up in constrained units.
               lb = [ps.lb for ps in spec.free], ub = [ps.ub for ps in spec.free],
               rng = MersenneTwister(MCMC_SEED))

# ── The draws reach disk HERE, not in section 7 ──────────────────────────
# Everything between this line and section 7 can throw: section 5 re-solves up to
# MCMC_JAC_DRAWS points, and jacobian_from_draws / se_bound_diagonal / curvature_check /
# boundary_mass / split_rhat_ess are all pure computations that can fail on degenerate input
# (a singular covariance, an empty feasible selection). Before v19.8.0 a failure anywhere in
# there discarded the draws of a run that had already completed — at the shipped budget, ten
# hours of sampling.
#
# Section 7 overwrites this same path at the end with the full bundle, adding the Ĝ and moment
# fields that only exist once section 5 has run. So this write is superseded on a clean run
# and is the whole result on a broken one. It carries `spec`, so it stands alone: every
# estimator computable from the finished run is computable from this file.
# Not under JAC_ONLY. That branch has no chain to protect — gens = 0 and `draws` is θ̂
# itself — so the stub carries no chain_lp/accepted/replaced to write, and snapshotting it
# would replace this window's real chain bundle with a single-draw stub.
MCMC_JAC_ONLY ||
    snapshot_chain(res.chain, res.chain_lp, res.accepted, res.outcome, res.replaced, res.gens,
                   res.state)

# ========================================================================
# 5. Points at which to store the model moment vector, from which Ĝ is regressed.
#    Chain mode thins the retained draws; jac-only (or an aborted chain) uses a
#    local design around the seed. A finite difference is not used either way: the
#    step that keeps it local is comparable to the solver's termination precision, so
#    it would measure solver noise rather than curvature.
# ========================================================================
# TWO DIFFERENT QUESTIONS, one flag until v19.7.1.
#
#   use_design — which Ĵ to use. An aborted chain's draws are a poor finite-difference
#                design, so falling back to a local design around θ̂ is right.
#   draws_ok   — whether the draws can be SUMMARISED. That does not depend on how the run
#                ended: 72,064 draws have a mean, a standard deviation, an R̂ and an edge
#                fraction whether or not the sampler reached stationarity.
#
# Conflating them cost the project its entire standard-error column. The chain aborts on the
# ACCEPTANCE FLOOR — a step-scale diagnostic, not anything that invalidates the draws — and
# on the base_fc bundle that meant NaN in all 23 se(chain) slots while the draws on disk gave
# 22 of 23 parameters a relative posterior SD under 35%, 15 of them under 5%. The numbers
# existed; the code declined to compute them.
#
# What a non-stationary sample's spread IS: a conservative measure of the region the
# criterion cannot separate. With R̂ ≈ 1.7 the pooled variance is ≈ 2.9× the within-chain
# variance, so the reported width already absorbs the disagreement between independently
# drifting chains rather than understating it. stage = :mcmc_aborted and the mcmc field carry
# the caveat, which is what that machinery is for.
#
# JAC_ONLY is excluded by construction: its stub sets gens = 0 and a single draw column.
use_design = MCMC_JAC_ONLY || res.aborted
chain_ok   = !use_design && res.gens > 0
draws_ok   = res.gens > 0 && size(res.draws, 2) > 1
n_kept     = size(res.draws, 2)
# The regression is solved only when something reads it: it IS Ĝ under :design, and a chain
# run keeps its draws for curvature_check. Under :fd with no chain (JAC_ONLY) nothing does,
# so its MCMC_JAC_DRAWS solves are skipped.
need_reg = MCMC_JAC_METHOD === :design || chain_ok

# Posterior SD of the pooled post-burn-in draws, in constrained units: the
# quasi-posterior standard error, and what LMR report (read_MCMC_chain.m takes std()
# of the pooled chain). Unlike the two Ĵ-based columns it never differentiates Q, so
# it stays valid where Q is only piecewise smooth — the reservation-cutoff softening
# in grids.jl is continuous but not differentiable, putting a kink wherever p*
# crosses a p-grid node.
# One map from a stored draw to θ, so every reporting site agrees with the sampled space. In
# :theta the draws ARE θ and the map is the identity; in :t it is the logistic.
_draw_to_θ(x, lb, ub) = MCMC_SPACE === :theta ? x : _to_constrained(x, lb, ub)

se_chain = draws_ok ?
    [std([_draw_to_θ(res.draws[k, t], spec.free[k].lb, spec.free[k].ub)
          for t in axes(res.draws, 2)]) for k in 1:d] :
    fill(NaN, d)
# The posterior MEDIAN, reported alongside the mean because it is the one summary that does
# not depend on which space the chain ran in: a monotone map commutes with quantiles, so
# median_θ = to_constrained(median_t) exactly, while the two means differ by Jensen's
# inequality. That difference is not academic here — the long comment below records a bundle
# on which the constrained mean returned Q = Inf while the unconstrained mean was feasible,
# and :theta averages in the space that produced the Inf. CH Thm 2 covers the median as the
# Laplace-type estimator under absolute loss, so it is an estimator in its own right and not
# a fallback. LMR report the mean; this is reported in addition, never instead.
θ̃_med = draws_ok ?
    [median([_draw_to_θ(res.draws[k, t], spec.free[k].lb, spec.free[k].ub)
             for t in axes(res.draws, 2)]) for k in 1:d] :
    fill(NaN, d)
# The posterior mean. AVERAGED IN THE SAMPLED (UNCONSTRAINED) SPACE, THEN TRANSFORMED —
# which is what LMR do, and the reverse of what this block did until v19.7.2.
#
# The old comment here claimed "as LMR do (read_MCMC_chain.m transforms each parameter and
# then averages)". That is right about their STANDARD ERRORS and wrong about their POINT
# ESTIMATE, and the two are computed in different places:
#   mpi_mcmc_mod.f90:485  solution_theta = sum(sum(all_population_knl,3),2)/(N*L)
# averages the RAW CHAIN buffer — chain coordinates, untransformed — and that vector is what
# reaches data/starting_val.raw, whose transform reproduces their published Table 5 to the
# last printed digit. read_MCMC_chain.m transforms draw by draw before std() for the SE, so
# the asymmetry is theirs, not a slip: mean in chain space, SD in model space.
#
# WHY IT MATTERS HERE, measured on the 2250-generation base_fc bundle (64 chains, burn = G/2):
#   mean in constrained space   -> Q = Inf     (no equilibrium: not a point estimate at all)
#   mean in unconstrained space -> Q = 524.81  (feasible)
# The two are the same draws. The box transform is monotone but nonlinear, so
# to_constrained(mean(t)) ≠ mean(to_constrained(t)), and only the first lands inside the
# equilibrium-existence region. This is NOT a multimodality problem: applying LMR's own
# outlier-chain rule (drop chains whose mean log-target is below Q1 − 2·IQR) retains 64 of 64
# and moves neither number, so the ensemble is already unimodal by their test.
#
# The gap is Jensen's inequality, and its SIGN is fully predicted: the box map is a logistic,
# convex below the box midpoint and concave above, so mean(σ(t)) > σ(mean(t)) for a coordinate
# sitting low in its box and < for one sitting high. 23 of 23 coordinates agree. The gaps are
# small — median 0.011 posterior SD, largest 0.223.
#
# The infeasibility localises to ONE coordinate: swapping δ_S alone from the constrained to the
# unconstrained mean restores feasibility at Q = 517.10, and no other single swap does. δ_S sits
# at box position 0.918 with a −0.189 SD gap.
#
# WHAT IS *NOT* ESTABLISHED, and was wrongly asserted when this comment was first written: that
# this is the D2 quadrature perforation. δ_S is indeed the parameter whose shock-density
# singularity sits at p = δ_S, but D2 was FIXED at v19.0.0 — R1 (softened OJS selection), R3
# (exact-CDF cell masses via build_cell_mass_density) and DAMP_THETA_S together took the share of
# nearby points returning a finite Q from 58.0% to 98.3%. And the node-coincidence mechanism was
# tested here and does not hold: the two means sit 0.121 and 0.160 cell widths from their nearest
# node at Np_S = 120, with no node between them. So this instance is one of the residual ~1.7%,
# cause unattributed, not the closed defect.
#
# SO THIS IS NOT A GUARANTEE, and nothing here should be read as one. Neither transform order
# guarantees a feasible mean. That is why Q(θ̄) is RE-SOLVED below and the post-mean bundle is
# gated on isfinite(Q̄) rather than assumed: the check is the safeguard, not the transform order.
#
# CH Thm 2 covers this object either way — it is the Laplace-type estimator under squared
# loss, consistent and first-order equivalent to the extremum estimator, needing no
# derivative of Q. What the transform order decides is whether the estimator is a point the
# model can be SOLVED at, which is a separate requirement and a hard one: every table, figure
# and counterfactual has to be computed somewhere.
# Two DIFFERENT objects, and conflating them made the reported pair inconsistent with itself.
#
#   θ̄_con  the REPORTED estimator: each draw is mapped to θ FIRST, then averaged. This is the
#          order LMR use (read_MCMC_chain.m:65-79 transform, then mean), and — the reason it
#          matters here — it is the order se_chain already uses, so the mean and the SD beside
#          it are now moments of the same sample. Transforming the t-space mean instead gave a
#          point that no reported SD belonged to.
#   θ̄_bnd  the point the POST-MEAN BUNDLE is solved at. Feasibility, not convention, governs
#          this one: the comment above records a bundle on which the θ-space mean returned
#          Q = Inf while the t-space mean was feasible. So θ̄_con is tried first and the
#          t-space mean is the fallback; the isfinite(Q̄) gate below decides between them.
#
# Under MCMC_SPACE = :theta the draws are already θ and the two coincide.
θ̄_con = draws_ok ?
    [mean(_draw_to_θ(res.draws[k, t], spec.free[k].lb, spec.free[k].ub)
          for t in axes(res.draws, 2)) for k in 1:d] :
    fill(NaN, d)
θ̄_bnd = draws_ok && MCMC_SPACE !== :theta ?
    [_to_constrained(mean(res.draws[k, t] for t in axes(res.draws, 2)),
                     spec.free[k].lb, spec.free[k].ub) for k in 1:d] :
    θ̄_con
# Each retained draw's vacancy costs, read back from the evaluation that scored it, and
# their posterior summary. A resumed run evaluated only its own generations, so a retained
# draw from the stored ones, which a retained window opening before the stored run's end
# holds, is solved here first, once per distinct point; otherwise there is none to solve.
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

# The bread of the sandwich. Its true object is the inverse curvature of Q/2, and Ĵ = Ĝ'WĜ is
# only its Gauss–Newton part: it drops the residual × second-derivative term, which is not
# small when Q is far above its degrees of freedom. A point Hessian of Q does not replace it,
# because Q is not quadratic at the scale of one standard error and its second differences
# change with the step. The chain integrates the curvature over the cloud it samples, so on
# the chain path the bread is P, the covariance of the kept draws in θ, and Ĝ is measured at
# θ̄, the estimate the table reports. Without a usable chain the bread is J⁻¹ and Ĝ sits at θ̂.
θ̄_t = draws_ok ?
    [_to_unconstrained(θ̄_con[k], spec.free[k].lb, spec.free[k].ub) for k in 1:d] : fill(NaN, d)
P   = chain_ok ?
    cov([_draw_to_θ(res.draws[k, t], spec.free[k].lb, spec.free[k].ub)
         for t in axes(res.draws, 2), k in 1:d]) : nothing
θ_G = chain_ok ? θ̄_t : θ0

# The active moments at t, then the vacancy costs, NaN where the point is infeasible. The
# costs ride on every evaluation the moments take, so their derivatives — and through the
# delta method their standard errors and the attracting check — cost no solve of their own.
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

    # The screen pre-pass above already produced Ĝ from a local design around the seed. When
    # the chain is the source of the draws we still want the chain-based Ĝ, so recompute;
    # when the design is the source, the pre-pass Ĝ is the same object and is reused.
    Ĝ_reg, R2 = (need_pre && use_design && Ĝ_pre !== nothing) ?
                    (Ĝ_pre, fill(NaN, K)) : jacobian_from_draws(draws_J, Msel, spec.free)
else
    Msel, draws_J = Matrix{Float64}(undef, K, 0), Matrix{Float64}(undef, d, 0)
    Ĝ_reg, R2     = nothing, fill(NaN, K)
    Ĝk_reg        = nothing
end
if MCMC_JAC_METHOD === :fd
    print("Adaptive-FD Ĝ: per-coordinate step + Richardson... "); flush(stdout)
    # Keyed on the objective, not on the buffer: smm_objective writes moments_out only after
    # its guards, so an unwritten buffer would hand back whatever was in that memory. Filled
    # with NaN rather than undef so a partial write could never masquerade as a moment either.
    # fd.n_eval counts every ladder solve, converged or not, but not which failed; the failures
    # are counted here. The centre solve is excluded from both, and reported on its own below.
    n_fail = Threads.Atomic{Int}(0)
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
    # No fallback. A column the ladder cannot measure used to be filled from the regression,
    # which produced a matrix whose header said "adaptive FD" while most of it was the old
    # object at rel_step = 0.05, and then printed a table and wrote a CSV from it with nothing
    # marking which column came from where. A Jacobian assembled from two estimators is not a
    # Jacobian. Either every column is measured the same way or the run stops and says which
    # coordinates defeated it; MCMC_JAC_METHOD = :design asks for the all-regression object
    # explicitly and gets it whole.
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
# se_sw is the sandwich at Ω = W⁻¹ under the bread P on the chain path, which is se(J⁻¹)
# when the bread is J⁻¹; se_bnd is its bound over every Ω with the same diagonal.
se_bnd, se_sw = se_bound_diagonal(Ĝ, spec.W, σ̂; bread = P)

# The vacancy costs are estimated through the tightnesses, so their standard errors are the
# delta method's: their gradient in the parameter vector (natural units, the rows of Ĝk) is the combination
# se_bound_diagonal is evaluated at. The SMM covariance does not depend on the coordinates it
# is expressed in, so these are the errors a k-coordinate estimation would report.
k_hat            = k_values(seed_eq.k)
se_bnd_k, se_sw_k = se_bound_diagonal(Ĝ, spec.W, σ̂, Matrix(Ĝk'); bread = P)

# θ̂'s attracting check off the same columns: D_ij = ∂ln k_i/∂ln θ_j needs the two tightness
# columns of Ĝk, and costs no solve where both tightnesses are estimated and Ĝ is measured at
# θ̂. On the chain path Ĝ is measured at θ̄, so θ̂'s check is solved directly.
# One check per period of the baseline: its tightness columns and its two rows of Ĝk.
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

# cov(Θ) is singular with fewer draws than parameters, so the cross-check is only
# meaningful when the chain supplied the design.
jgap    = chain_ok && size(draws_J, 2) > d ?
          curvature_check(draws_J, Ĝ, spec.W, spec.free) : NaN

# ========================================================================
# 6. Gate diagnostics and the parameter table (each quantity printed once).
# ========================================================================
rhat, ess = draws_ok ? split_rhat_ess(res.chain, res.burn) :
                       (fill(NaN, d), fill(NaN, d))
# The vacancy costs are not coordinates of the chain: free entry gives them at each kept
# draw, and k_draws holds them in the column order of res.draws, which is
# reshape(chain[:, :, burn+1:end], d, :), chain fastest. Laid back out as chain × generation
# they get the same split R̂ and ESS as the coordinates, on the same kept generations.
# rhat_all and ess_all are the table's R̂ column and its ESS, the coordinates then the costs.
rhat_k, ess_k = draws_ok && size(k_draws, 2) == size(res.draws, 2) &&
                    size(res.draws, 2) % size(res.chain, 2) == 0 ?
                split_rhat_ess(reshape(k_draws, NK, size(res.chain, 2), :), 0) :
                (fill(NaN, NK), fill(NaN, NK))
rhat_all, ess_all = vcat(rhat, rhat_k), vcat(ess, ess_k)
blo, bhi  = draws_ok ? boundary_mass(res.draws, spec.free) :
                       (fill(NaN, d), fill(NaN, d))
sgrow     = draws_ok ? spread_growth(res.chain, res.burn) : fill(NaN, d)

# Attribute the climb rather than assert a cause: −½ΔQ indicts the point estimate,
# Δlogjac only says the seed sat near a rail. Q_best is also the best visited point of §8.
Q_best = smm_objective(res.theta_best, spec)

# How Ĝ was measured, printed above the box on both paths: every standard error in the
# table is read off Ĵ = Ĝ'WĜ, so these lines are the provenance of the whole table.
R2f = filter(isfinite, R2)
if MCMC_JAC_METHOD === :fd
    hf  = filter(isfinite, fd.h_frac)
    stf = filter(isfinite, vec(fd.stab))
    @printf("  Ĝ: %d×%d adaptive FD + Richardson, %d solves, target ‖Δm/σ̂‖=%.1f\n",
            K, d, sum(fd.n_eval), fd.target_sd)
    @printf("      step/box width: median %.2e, range %.2e–%.2e | Richardson instability median %.1e\n",
            median(hf), minimum(hf), maximum(hf),
            isempty(stf) ? NaN : median(stf))
    # A coordinate whose step never reached the target displacement band still returns a
    # derivative, but one measured at a radius the geometry did not choose — the defect the
    # method exists to remove, so it is named rather than counted.
    # The centre is not used by a central difference, but a Jacobian expanded about a point the
    # solver rejects is a different object from one expanded about an equilibrium, and the
    # regression route never said so.
    fd.centre_ok ||
        @printf("      NOTE: Q(%s) is not finite at %d threads, so Ĝ is centred on a point the \
solver rejects. The differences themselves are unaffected — only %s ± h is evaluated.\n",
                chain_ok ? "θ̄" : "θ̂", Threads.nthreads(), chain_ok ? "θ̄" : "θ̂")
    all(fd.tuned) ||
        @printf("      %d coordinate(s) off the target band (‖Δm/σ̂‖ outside [%.2f, %.1f]): %s\n",
                count(.!fd.tuned), fd.target_sd / 3, fd.target_sd * 3,
                join([@sprintf("%s %.2f", param_symbol(spec.free[j]), fd.disp_sd[j])
                      for j in 1:d if !fd.tuned[j]], ", "))
    # One extrapolate has nothing to be stable against, so those columns carry a derivative
    # with no validation behind it. Distinct from an off-band step and worth its own line.
    let bare = [j for j in 1:d if fd.ok[j] && !any(isfinite, view(fd.stab, :, j))]
        isempty(bare) ||
            @printf("      %d column(s) with a single extrapolate, instability unchecked: %s\n",
                    length(bare), join(param_symbol.(spec.free[bare]), ", "))
    end
    # se_bound_diagonal inverts Ĵ = Ĝ'WĜ with pinv, whose tolerance is relative to Ĵ's largest
    # singular value. Since cond(Ĵ) = cond(G̃)², a G̃ conditioned past ~1e8 puts Ĵ beyond double
    # precision and pinv starts DISCARDING directions — silently setting those coordinates'
    # variance to zero. Reported because a truncated direction is an unidentified parameter
    # wearing a finite standard error.
    let sv = svdvals(Diagonal(1.0 ./ σ̂) * Ĝ), tol = size(Ĝ, 2) * eps() * sv[1]^2
        @printf("      cond(G̃)=%.2e ⇒ cond(Ĵ)=%.2e; pinv keeps %d of %d directions\n",
                sv[1] / sv[end], (sv[1] / sv[end])^2, count(>(sqrt(tol)), sv), d)
    end
    # The regression Ĝ on the same seed, priced in the units that matter. Printed every run
    # because it is the only place the cost of the method choice is visible.
    if Ĝ_reg !== nothing
        # Both bounds under the bread J⁻¹, so the ratio prices the Jacobian alone.
        sb_r, _ = se_bound_diagonal(Ĝ_reg, spec.W, σ̂)
        sb_f, _ = se_bound_diagonal(Ĝ, spec.W, σ̂)
        @printf("      vs design regression (R² median %.3f): se(bound) median ratio FD/reg = %.2f\n",
                isempty(R2f) ? NaN : median(R2f), median(sb_f ./ sb_r))
    end
else
    @printf("  Ĝ: %d×%d design regression from %d draws, moment R² min=%.3f median=%.3f\n",
            K, d, size(Msel, 2), minimum(R2f), median(R2f))
end

# The table fixes the box width, so it is laid out first. The chain path adds se(chain)
# beside the estimate, its |t| beside se(bound), and each coordinate's R̂ last, ticked where
# it meets MCMC_RHAT_MAX (the two blanks after R̂ hold the ✓); the edge test rides in the flag.
# The sandwich column is named by its bread: sandw. with the chain's covariance, J⁻¹ without.
sw  = chain_ok ? "sandw." : "J⁻¹"
hdr = draws_ok ?
    "  block   param        estimate  se(chain)" * lpad("se($sw)", 11) * "   se(bound) |t|(chain)" *
    lpad("|t|($sw)", 12) * "   |t|(bnd)       R̂" * "  " :
    "  block   param        estimate    se(J⁻¹)   se(bound)  |t|(J⁻¹)   |t|(bnd)"
box_w = textwidth(hdr) + 4              # + room for the trailing RAIL / edge flag
box_rule(l, r) = println(l, "═"^box_w, r)
box_title = MCMC_JAC_ONLY ? "Standard errors from Ĵ = Ĝ'WĜ" : "DE-MC Quasi-Posterior"

println()
box_rule("╔", "╗")
println("║  ", box_title, " "^(box_w - 2 - textwidth(box_title)), "║")
box_rule("╠", "╣")
# Chain diagnostics only when a chain ran; without one they would print zeros and NaN.
if !MCMC_JAC_ONLY
    # A run that exhausts the cap has NOT met the stopping criteria: say so, and let the
    # attained R̂/ESS line below carry what was actually reached.
    @printf("  gens=%d/%d %s  burn=%d  chains=%d  accept=%.3f\n",
            res.gens, res.gens_requested,
            # abort_why distinguishes the two abort paths, which call for opposite fixes:
            # :drift means re-seed from theta_best, :acceptance means rescale the proposal.
            # hasproperty keeps results produced before v19.5.0 readable.
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
    # The Jacobian term is unbounded below, so a climb from it only locates a rail.
    # Under :flat_t there is no Jacobian in the target, so the whole drift is ΔQ by
    # construction and the split has only one live component. Passing equal log-Jacobians
    # rather than skipping the call keeps one code path and makes dlj print as exactly 0.
    _lj0, _lj1 = _USE_LOGJAC ?
        (logjac_box(θ0, spec.free), logjac_box(collect(res.theta_best), spec.free)) :
        (0.0, 0.0)
    dQ, dlj = drift_components(Q_seed, Q_best, _lj0, _lj1)
    # Printed where the drift abort is armed or the run aborted. Otherwise the split acts on
    # nothing, and its ΔQ is the one the best-visited-point line states.
    if (MCMC_CHECK_EVERY > 0 && MCMC_DRIFT_MAX > 0) || res.aborted
        @printf("  seed drift=%+.1f (abort >%.1f) = %+.1f from Q (ΔQ=%+.4g) %+.1f from log|dθ/dt|\n",
                res.drift, MCMC_DRIFT_MAX, dQ, Q_best - Q_seed, dlj)
    end
    # The drift split decides what to do next, so say it rather than leave the reader to
    # infer it from two signed numbers. ΔQ = 1 is one moment moving by one sampling
    # standard error, which is the scale that makes an improvement worth re-seeding on.
    if res.aborted && Q_best < Q_seed - 1.0
        @printf("  → the chain beat θ̂ by ΔQ=%.2f, so θ̂ is not the optimum; theta_best is in %s\n",
                Q_seed - Q_best, basename(chain_path(WINDOW, W_SUFFIX)))
    elseif res.aborted && dlj > abs(dQ)
        @printf("  → the climb is the Jacobian term, not Q: θ̂ sits near a box edge where\n")
        @printf("    log|dθ/dt| → −∞. Seed the railed coordinates interior; θ̂ stands.\n")
    end
    # A replacement inside the retained sample means those draws are not from the target,
    # so report the boundary rather than only the count.
    if res.n_replaced > 0
        @printf("  stuck-chain replacements: %d, last at g=%d%s\n",
                res.n_replaced, res.last_replace,
                res.last_replace > res.burn ? "  ← INSIDE the retained sample" : "")
    end
    # The attained extremes against the gates: R̂ max is the worst of the table's R̂ column,
    # vacancy costs included, and ESS has no column.
    # max and min propagate NaN, so the case with no finite entry is handled outside them.
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
# |t| against a standard error, NaN wherever that error cannot carry a ratio. A zero,
# non-finite or pinv-truncated se divides to Inf, which would read as infinite significance
# — the opposite of what an unusable se means. The row is still written either way.
_abs_t(θ, se) = (isfinite(se) && se > 0.0) ? abs(θ) / se : NaN

# Distance from a point estimate to its nearest box bound, as a fraction of box width.
# Needs no chain, so it is the rail test on the JAC_ONLY path, where edge_frac is blank.
_rail_frac(θ, lb, ub) = min(θ - lb, ub - θ) / (ub - lb)
const RAIL_TOL = 1e-3   # below this the coordinate is on the rail: no two-sided interval

out_csv = mcmc_csv_path(WINDOW, W_SUFFIX; point = MCMC_JAC_ONLY ? :smm : :mcmc)
open(out_csv, "w") do io
    # COLUMN ORDER IS APPEND-ONLY. Every column keeps its name and position and a new one
    # goes on the end, which is what lets a reader written against an earlier version still
    # parse this file — and why se_chain, symbol, corner_declared and the two |t| ratios sit
    # at the tail rather than beside the quantities they belong with.
    #   symbol           the market-suffixed display name (β_S, not the bare β that `name`
    #                    carries for both markets), so a table labels rows without
    #                    re-deriving the suffix.
    #   corner_declared  0/1 from CORNER_PARAMS. With the shipped empty default every other
    #                    field is byte-identical to before it existed.
    #   se_curvature     the sandwich and its bound, named by the bread as the console's
    #   se_bound         columns are: se_sandwich and abs_t_sandwich on the chain path (bread
    #                    P, Ĝ at θ̄), se_curvature and abs_t_curvature without a usable chain
    #                    (bread J⁻¹, Ĝ at θ̂). Same position either way.
    #   abs_t_curvature  |estimate|/se_curvature and |estimate|/se_bound, formed against the
    #   abs_t_bound      reported estimate: θ̄ on the chain path, θ̂ otherwise. They are the
    #                    console's |t| columns on every path.
    #                    A row is emitted for every free parameter, including one sitting on
    #                    a box bound where no two-sided interval exists: the LaTeX table
    #                    drops those, so the CSV has to stay complete rather than pre-filter.
    #   rail_frac        distance from the point estimate to its NEAREST box bound, as a
    #                    fraction of box width. Unlike edge_frac it needs no chain and
    #                    unlike corner_declared it is measured rather than declared, so it
    #                    is the column that answers "is this coordinate on a rail?" on the
    #                    JAC_ONLY path. It also reads on Ĝ: mcmc_diagnostics.jl caps the FD
    #                    step at 0.999·(distance to the nearest bound), so a coordinate with
    #                    rail_frac = 1e-4 has its derivative measured across one ten-thousandth
    #                    of its box and contributes a column to Ĵ = Ĝ'WĜ that is small,
    #                    noisy, and able to dominate the inverse.
    sw_col = chain_ok ? "sandwich" : "curvature"
    println(io, "block,name,label,point_estimate,post_mean,se_$(sw_col),se_bound,q025,q500,q975,rhat,ess,edge_frac,spread_growth,se_chain,symbol,corner_declared,abs_t_$(sw_col),abs_t_bound,rail_frac")
    for (k, ps) in enumerate(spec.free)
        # Quantiles and the posterior mean require a stationary chain; without one
        # the se columns still stand (they come from Ĝ), so those are reported and
        # these are left blank rather than computed from a non-stationary walk.
        dk = draws_ok ?
             [_draw_to_θ(res.draws[k, t], ps.lb, ps.ub) for t in 1:n_kept] :
             Float64[]
        q(p) = isempty(dk) ? NaN : quantile(dk, p)
        pmean = isempty(dk) ? NaN : mean(dk)
        pe = _draw_to_θ(θ0[k], ps.lb, ps.ub)
        edge = blo[k] + bhi[k]
        # The reported SE is the chain's when there is one: it is the only column that
        # survives a non-differentiable objective. Otherwise fall back to the two
        # Ĝ-based columns and say so in the footer.
        se_rep = draws_ok ? se_chain[k] : se_sw[k]
        # All three |t| are formed against the REPORTED estimate, the point the sandwich and
        # its bound are measured at. Deliberately still chain_ok and not draws_ok, so an
        # ABORTED run reports θ̂ rather than the posterior mean — with se_rep now being
        # se(chain), the pairing on that path is θ̂ ± the spread of draws around θ̂, which is
        # what the chain measured (it was seeded there).
        #
        # This is not the "width of one point, location of another" error the old comment
        # here warned about; it is the correction for it. On base_fc the coordinate-wise
        # posterior mean is INFEASIBLE — Q(θ̄) = Inf, and not because of b_S alone, since
        # snapping b_S to its bound leaves it infeasible. The feasible set of a segmented
        # model with an equilibrium-existence condition is not convex, so an average of
        # feasible draws need not be feasible. Reporting θ̄ on a path where it may not solve
        # would put a parameter vector in the table that the model cannot be evaluated at.
        est_rep = chain_ok ? pmean : pe
        tstat   = se_rep > 0 ? abs(est_rep) / se_rep : NaN
        # se(bound) is the supremum of the sandwich over every Ω with the same diagonal, so a
        # coordinate that clears a threshold on |t|(bnd) clears it whatever the correlation
        # structure of the moment errors.
        tstat_s = _abs_t(est_rep, se_sw[k])
        tstat_b = _abs_t(est_rep, se_bnd[k])
        # A railed coordinate is flagged ahead of edge because it invalidates its own
        # interval outright: asymptotic normality needs an interior point.
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
    # The vacancy costs, estimated through the tightnesses: θ̂'s costs with their
    # delta-method standard errors, and the chain's summary of each draw's costs, with their
    # R̂ and ESS (rhat_k, ess_k). They have no box, so no rail or edge.
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

# A rail is not one row's problem. mcmc_diagnostics.jl caps the FD step at the distance to
# the nearest bound, so a railed coordinate's column of Ĝ is a derivative measured across a
# sliver of its box — and every interval in the table above is read off Ĵ = Ĝ'WĜ, which that
# column enters. So the rails are named here, with what to do about them, rather than left
# to be noticed in a CSV column.
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
    # Q at the estimate (the mean of the kept draws), at the seed and at the best point the
    # chain visited. The mean is an average over the basin, not a competitor in a
    # minimisation, so its Q normally sits above the seed's; the best visited point is a
    # record of where the chain went, never the estimate.
    Q_mean = smm_objective(θ̄_t, spec)
    @printf("\n  Q(θ̄)=%.4f   Q(θ̂)=%.4f   Q(θ_best)=%.4f\n", Q_mean, Q_seed, Q_best)
else
    if MCMC_JAC_ONLY
        println("\n  se(J⁻¹) assumes uncorrelated moment errors. se(bound) holds for any moment")
        println("  covariance with the same diagonal: report se(bound).")
    else
        # After an abort a chain ran, sometimes for thousands of generations, and its draws
        # exist; they are simply not a stationary sample, and theta_best is the deliverable.
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
chain_jls = chain_path(WINDOW, W_SUFFIX)
# Not written under JAC_ONLY, for the reason the snapshot above is skipped: this file is
# the CHAIN, and a pass that never sampled would overwrite a real one with a single-draw
# stub that has no chain_lp, no accepted and no replaced to carry. What the chain-free pass
# does produce — θ̂, both Ĵ-based standard errors, and Ĝ's per-moment R² summary — is in the
# results CSV and the console table.
MCMC_JAC_ONLY || open(chain_jls, "w") do io
    serialize(io, (chain      = res.chain,
                   draws      = res.draws,
                   # ── WHAT MAKES A LATER CHANGE OF ESTIMATOR POSSIBLE WITHOUT RE-RUNNING ──
                   # `chain` is the full d × N × gens array and `burn` is recorded, so any
                   # burn-in fraction, any subset of chains and any pooling can be re-derived.
                   # The three fields below are what that is NOT enough for:
                   #
                   #   chain_lp   the log-target of EVERY draw (N × gens). Required by every
                   #              alternative to a plain mean — the modal/MAP draw, an
                   #              lp-trimmed or lp-weighted mean, a basin filter, a recomputed
                   #              stuck-chain score, R̂ on the target rather than on coordinates.
                   #              It was computed and discarded before v19.8.0; recovering it
                   #              afterwards means re-solving every draw, which at the shipped
                   #              budget is 950,000 solves, i.e. the whole run again.
                   #   replaced   which chain-generations the stuck-chain rule overwrote. Those
                   #              draws are NOT from the target, and with the LMR timing
                   #              (MCMC_OUTLIER_BURN_ONLY = false) the rule can fire inside the
                   #              retained window. n_replaced and last_replace say that it
                   #              happened, not where; this says where, so the draws can be
                   #              excised. LMR keep the same object
                   #              (all_outliner_results_nl, mpi_mcmc_mod.f90:423).
                   #   accepted   which draws are new states rather than repeats, so an average
                   #              over unique states is available.
                   #
                   # Together with `spec` below, every estimator computable from this run stays
                   # computable from this file. Cost: ~8 MB of the ~200 MB bundle.
                   chain_lp   = res.chain_lp,
                   accepted   = res.accepted,
                   replaced   = res.replaced,
                   n_replaced = res.n_replaced,
                   last_replace = res.last_replace,
                   # What MCMC_INIT = :resume continues from and checks: the outcome code of
                   # every draw, the sampler's state at the last generation (its RNG included,
                   # so the continuation is exact) and the settings that define the chain.
                   outcome    = res.outcome,
                   demc_state = res.state,
                   sampler    = SAMPLER,
                   # The reported estimate and its standard error. theta_mean averages
                   # over the basin, which is what makes it reportable: on a criterion
                   # with flat directions any single visited point is one draw from the
                   # ruggedness, and se_chain is the SD of the draws around this mean.
                   theta_mean = θ̄_con,
                   # The vacancy costs of every retained draw (2 × kept, rows U and S), in
                   # the order of `draws`, and at θ̂.
                   k_draws    = k_draws,
                   k_hat      = k_struct(spec, k_hat),
                   # The transform-invariant point estimate. Identical to q500 in the results
                   # CSV; carried here because a table script reads the bundle, not the CSV.
                   theta_median = θ̃_med,
                   se_chain   = se_chain,
                   # Diagnostic only, never reported: the best single point visited. Its
                   # Q beats the seed's by a margin no descent path from the seed can
                   # cross, which is the evidence that a local optimiser cannot be
                   # trusted here — not a competing estimate.
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
                   # Which route built Ĝ, and the FD route's own diagnostics: the accepted
                   # step per coordinate, the Richardson rung, and the per-moment instability
                   # that replaces R² when there is no fit. `nothing` under :design. The
                   # regression Ĝ is kept beside it whenever it was solved (see need_reg).
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
                   # The spec is what makes theta_best USABLE rather than merely recorded.
                   # smm_objective(theta, spec) needs the whole object — fixed parameters,
                   # moment targets, sim settings, grids, W — and free/labels/lb/ub above
                   # are not a substitute for any of it. Without this field the chain's
                   # best visited point cannot be re-evaluated, re-solved for moments, or
                   # warm-started from, which is the entire reason the chain is run: it
                   # reaches points the optimiser cannot, and a point that cannot be
                   # re-evaluated is not a result. Written for every chain bundle,
                   # independent of MCMC_CHECKPOINT — that flag governs whether the best
                   # visited point also gets a bundle of its own, a separate decision from
                   # whether this one is self-contained.
                   spec       = spec,
                   provenance = run_provenance(window = WINDOW, w_suffix = W_SUFFIX,
                                               version = ROYSEARCH_VERSION)))
end

# ========================================================================
# 8. The best visited point
# ========================================================================
# The chain visits points the optimiser cannot reach: Metropolis accepts uphill moves, so it
# leaves basins an optimiser is trapped in. The best of them is theta_best in the chain file
# whatever the settings; MCMC_CHECKPOINT = true also writes it to its own bundle, here and
# in the during-run callback, when it beats the last one written by more than
# PROMOTE_MIN_DQ. Neither touches the seed or the chain's estimate.
if MCMC_CHECKPOINT && Q_best < Q_ckpt[] - PROMOTE_MIN_DQ
    promote!(collect(res.theta_best), Q_best, Q_ckpt[]; label = "END")
    Q_ckpt[] = Q_best
end

# ========================================================================
# 9. Where the standard errors go
# ========================================================================
# JAC_ONLY is the SMM path's inference: it adds what it measured to the SMM bundle it read —
# the parameters' standard errors, the vacancy costs' delta-method errors and the attracting
# check — and leaves the point, stage, provenance and sim as the SMM run wrote them. If the
# file no longer holds the point the run started from, nothing is written to it.
#
# A chain run writes its own estimate, estimate_{window}{W}_mcmc.jls, and nothing else in
# estimates/ besides its CSV:
#   result         the posterior mean of the retained draws, re-solved; the t-space mean
#                  where the reported mean does not solve
#   stage          :mcmc, or :mcmc_aborted when the chain hit the acceptance floor or the
#                  drift abort: those draws are not a stationary sample, so no consumer
#                  takes the file as an estimate
#   se             sandwich and bound with the chain's covariance as bread and Ĝ at θ̄, and
#                  chain, the posterior SD; an aborted chain's carries curvature and bound
#                  from Ĵ = Ĝ'WĜ at θ̂ instead. se_source names the bread
#   mcmc           the diagnostics a footnote needs, and `seed`, the file the chain started from
#   vacancy_costs  at the posterior mean, with the delta-method errors and the chain's summary
#   attracting     the free-entry slope check at the posterior mean
# The sandwich column is keyed by its bread, as in the CSV: sandwich (P) or curvature (J⁻¹).
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
    # Both are posterior means of the same draws under CH Thm 2, so the fallback changes
    # which point the bundle carries, never which estimator the table reports.
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

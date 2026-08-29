############################################################
# smm/MCMC_main.jl — standalone standard-error script (DE-MC)
#
# Runs AFTER an SMM estimation. It does NOT touch smm_main.jl / smm.jl.
# It loads the saved SMM optimum for (WINDOW, W_COND_TARGET) and runs a
# Differential-Evolution MCMC (smm/demc.jl) on the log quasi-posterior
#       logπ(θ) = −½ · g(θ)′W g(θ)  +  log|dθ/dt|
# With cond_target = 0.0 the first term is Σ_k ((data_k − model_k)/σ̂_k)², LMR's
# 1/SD² scheme (objective_function_mod.f90), which also fixes the temperature:
# the sampling variances put g′Wg on an n·g′Ω̂_d⁻¹g footing, so the chain's
# dispersion has standard-error units without a separate scale factor.
# The log-Jacobian is what makes the target proper in the sampler's coordinates.
#
# Intervals follow Chernozhukov–Hong (2003) Theorem 4, which needs no
# information equality: Ĵ⁻¹ = n·Cov(chain) combined with an Ω̂. Since Ω is not
# credibly estimable across four sources at three frequencies, the reported
# companion column is the sharp upper bound over every Ω consistent with the
# estimated sampling variances — see `se_bound_diagonal`. It writes a parameter
# table keyed to the point estimates by (block, name):
#
#   output/smm/mcmc_results_{window}{W}.csv  per-parameter estimates, intervals,
#                                            the Ω-free bound, and gate diagnostics
#   output/smm/mcmc_chain_{window}{W}.jls    chain, draws, per-draw moments, Ĝ
#
# Usage (from project root — threads strongly recommended):
#   julia --threads auto roysearch/smm/MCMC_main.jl
#
# Point estimates come from the SMM run (relative/equal weights); the SEs
# here are computed under the diagonal weight, seeded at that optimum.
############################################################

include(joinpath(@__DIR__, "version.jl"))
println("="^60)
println("  RoySearch v$(ROYSEARCH_VERSION) — DE-MC standard errors")
println("="^60)
flush(stdout)

# ── Paths (identical to smm_main.jl) ────────────────────────────────────
const SMM_DIR      = @__DIR__
const SOLVER_DIR   = joinpath(SMM_DIR, "..", "solver")
const PROJECT_ROOT = joinpath(SMM_DIR, "..", "..")
const OUTPUT_DIR   = joinpath(PROJECT_ROOT, "output")
const SMM_OUT_DIR  = joinpath(OUTPUT_DIR, "smm")

# ── Packages (same set as smm_main.jl so the includes resolve) ──────────
print("Loading packages... "); flush(stdout)
using LinearAlgebra, SparseArrays, Statistics, Random, Distributions
using FastGaussQuadrature, Interpolations, Parameters, Printf
using Base.Threads, Optim, CSV, DataFrames, Serialization
using Clustering, QuasiMonteCarlo, JSON3
println("done."); flush(stdout)

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
include(joinpath(SMM_DIR, "smm.jl"))
include(joinpath(SMM_DIR, "candidates.jl"))   # for include-env parity with smm_main
include(joinpath(SMM_DIR, "mcmc_diagnostics.jl"))
include(joinpath(SMM_DIR, "demc.jl"))          # uses min_ess / converged_sequential
println("done.")
@printf("Threads available: %d\n\n", Threads.nthreads()); flush(stdout)

# ========================================================================
# CONFIG
# ========================================================================
# Batch overrides: defaults below are the single-run configuration; a batch sets
# ROYSEARCH_* environment variables so it needs no edit to this file. `env_setting`
# (settings.jl) reads them and records each resolution for the [env] block below.
assert_no_legacy_env()

WINDOW        = env_setting(:WINDOW, :base_fc)            # window to compute SEs for
W_COND_TARGET = env_setting(:W_COND_TARGET, 0.0)          # 0.0 = diagonal-σ, 2.0 = equal weights.
                                  # Selects BOTH the estimation bundle to load and
                                  # the weighting the chain targets — same meaning
                                  # and same admissible values as in smm_main.jl.
                                  # The held-out moment set is taken from the
                                  # bundle, so there is no SKIP_MOMENTS here.

# DE-MC controls (see smm/demc.jl). N = 0 ⇒ 2·d chains.
MCMC_N           = env_setting(:MCMC_N, 64)  # 2·d is ter Braak's minimum; LMR run 95
                                   # chains for 16 parameters (5.9 per dimension) on an
                                   # MPI cluster with one rank per chain. On 10 cores the
                                   # binding constraint is total solves, and measurement
                                   # shows ESS depends on N·gens rather than on the split,
                                   # so fewer chains × more generations is preferred: at a
                                   # fixed budget that is what improves R̂.
MCMC_GENS        = env_setting(:MCMC_GENS, 4000)  # budget cap ≈ 128k solves ≈ 6.5 SMM runs, the point at
                                   # which the gates below are first met at d = 25 (measured:
                                   # R̂ 1.070, ESS 342; at 2 SMM runs it is R̂ 1.25, ESS 133).
                                   # LMR run 10,000
                                   # generations (main_mpi.f90: max_iteration), retaining
                                   # only the last 500 per chain in a ring buffer — their
                                   # "1,000 draws" is what is stored, not what is computed;
                                   # they discard 95% of 950,000 solves.
MCMC_BURN        = 0.5
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
MCMC_CR          = env_setting(:MCMC_CR, 0.95)
MCMC_DELTA       = env_setting(:MCMC_DELTA, 1)
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
MCMC_B_MULT      = env_setting(:MCMC_B_MULT, 1e-5)
# b_add is an absolute isotropic shock and does not scale with the spread. At 1e-4 it is
# 0.1x the off-ridge tolerance, and it is the only mover in generation 1 under :at_seed.
MCMC_B_ADD       = env_setting(:MCMC_B_ADD, 1e-4)

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
MCMC_OUTLIER_IQR = 2.0             # Replace a chain scoring more than this many IQRs
                                   # below Q1 with the best chain (LMR
                                   # mpi_mcmc_mod.f90:419). A stuck chain's position
                                   # enters every other chain's difference vector, so one
                                   # dead chain degrades all N. Runs during burn-in only,
                                   # since the replacement is not reversible; the log
                                   # reports the last generation it fired, and any value
                                   # at or past the burn boundary invalidates the
                                   # retained draws. 0 disables.
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

# Fixed budget: MCMC_GENS generations, no sequential stop. Sequential termination
# assumes a unimodal target the chain can become stationary on; where the objective is
# a rough plateau with several near-equivalent regions the unidentified coordinates
# never clear R̂ ≤ MCMC_RHAT_MAX, so testing only spends the whole cap and then reports
# failure. R̂ and ESS are still computed and printed per parameter, as diagnostics of
# which coordinates are identified rather than as a gate. LMR's package contains no
# convergence test at all. Set MCMC_CHECK_EVERY > 0 to restore the sequential stop.
MCMC_CHECK_EVERY = env_setting(:MCMC_CHECK_EVERY, 250)  # generations between sequential stop checks; 0 disables
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
MCMC_MOVES_MIN   = env_setting(:MCMC_MOVES_MIN, 2300)   # 100·d at d = 23
# DRIFT_FLAT in log units. Under MCMC_PRIOR = :flat_t, logπ = −Q/2 exactly, so a climb of
# L log units IS ΔQ = 2L — and paired ΔQ only clears the grids' disagreement above
# |ΔQ| ≈ 2. So 1.0 asks the run to certify the smallest difference the criterion can
# actually resolve. It was 0.5, anchored on PROMOTE_MIN_DQ, which was the wrong anchor:
# that constant governs whether to overwrite a bundle, not what the objective can resolve.
MCMC_DRIFT_FLAT  = env_setting(:MCMC_DRIFT_FLAT, 1.0)   # log units; 1.0 ⇒ ΔQ = 2
MCMC_ACC_FLOOR   = env_setting(:MCMC_ACC_FLOOR, 0.02)   # abort-and-diagnose below this, two checks running
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
# Default off. With the drift abort disabled the chain is expected to find better
# points, and promotion would overwrite the warm-start bundle mid-run — changing the
# θ̂ the paper reports while the run that measures its uncertainty is still going. The
# best visited point is serialised into the chain bundle as theta_best/params_best
# either way, so nothing is lost; set ROYSEARCH_MCMC_CHECKPOINT=true to promote.
MCMC_CHECKPOINT  = env_setting(:MCMC_CHECKPOINT, true)  # Write each new best point to the
                                   # warm-start bundle as the chain finds it, not only at the end.
                                   # Raising MCMC_DRIFT_MAX makes a run long enough that reaching
                                   # its own end stops being guaranteed; this keeps the best point
                                   # on disk however the run ends.
                                   # Improvement required to overwrite the bundle. Anchored on what
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
MCMC_JAC_ONLY    = env_setting(:MCMC_JAC_ONLY, false)  # Skip the chain: estimate Ĵ = Ĝ'WĜ from a local
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
                                   # to justify.) With 28 moments and 23 free parameters the
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
MCMC_SCREEN_CAP   = env_setting(:MCMC_SCREEN_CAP, 1.0)    # required: b_S's width is
                                  # 124 in t, and the logit clamp saturates near ±18.42,
                                  # so an uncapped scale there draws saturated corners.
MCMC_SCREEN_FLOOR = env_setting(:MCMC_SCREEN_FLOOR, 1e-3) # guard a zero/absent width
MCMC_JAC_DRAWS   = 600            # thinned retained draws re-solved to store the
                                  # moment vector, from which Ĝ is regressed

derived_dir = joinpath(PROJECT_ROOT, "data", "derived")

_w_suffix(ct::Float64) = ct == 0.0 ? "_diagonalW" :
                         ct == 2.0 ? "_equalW" :
                         error("W_COND_TARGET must be 0.0 (diagonal-σ) or 2.0 (equal weights); got $ct.")
W_SUFFIX = _w_suffix(W_COND_TARGET)

@printf("Window: %s   weighting: %s\n", WINDOW, W_SUFFIX)
print_env_settings()

# ========================================================================
# 1. Load the SMM optimum (point estimate + spec) to seed the chain
# ========================================================================
seed_jls = joinpath(SMM_OUT_DIR, "smm_result_$(WINDOW)$(W_SUFFIX).jls")
isfile(seed_jls) || error(
    "No estimation bundle at $seed_jls — run smm_main.jl for WINDOW=$WINDOW " *
    "(W_COND_TARGET=$W_COND_TARGET) first.")
bundle = _load_smm_bundle(seed_jls; delete_on_fail=false, label="estimation bundle")
isnothing(bundle) && error("Could not read $seed_jls.")

θ0    = collect(float.(bundle.result.theta_opt))   # free params, UNCONSTRAINED space
spec0 = bundle.spec

# Held-out moments come from the bundle, so they cannot disagree with the run
# that produced the seed.
BUNDLE_SKIP = Symbol[k for k in keys(spec0.moments) if spec0.moments[k].weight <= 0.0]

# ========================================================================
# 2. Rebuild the spec with the SAME weighting the estimation used. Everything
#    else (free set, fixed η/r/ν/φ, moments, grids, sim) is taken verbatim from
#    the estimation spec, so θ0 is valid and the model is solved identically.
# ========================================================================
W_chain = load_weight_matrix(; window=WINDOW, derived_dir=derived_dir,
                               cond_target=W_COND_TARGET, skip_moments=BUNDLE_SKIP)
spec = build_smm_spec(
    spec0.moments, spec0.sim;
    fixed        = spec0.fixed,
    free_specs   = spec0.free,
    run          = spec0.run,
    W            = W_chain,
    q_scale      = 1.0,          # raw g'Wg: the quasi-posterior must not be rescaled
    skip_moments = BUNDLE_SKIP,
)
@assert length(θ0) == length(spec.free) "seed length ($(length(θ0))) ≠ free params " *
    "($(length(spec.free))) — bundle and spec rebuild disagree."

MOM_KEYS = active_moment_keys(spec)
K, d     = length(MOM_KEYS), length(spec.free)

# σ̂ over the active moments, in the SAME order as MOM_KEYS and spec.W. Needed for
# the Ω-free bound; available only under the diagonal-σ weighting.
σ̂ = W_COND_TARGET == 0.0 ? [1.0 / sqrt(spec.W[i, i]) for i in 1:K] : fill(NaN, K)

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
const _USE_LOGJAC = (MCMC_PRIOR === :flat_theta)
MCMC_PRIOR in (:flat_theta, :flat_t) ||
    error("MCMC_PRIOR = $(MCMC_PRIOR) is not recognised; use :flat_theta or :flat_t.")

function logposterior(θ)
    Q = smm_objective(θ, spec)
    isfinite(Q) || return -Inf
    return _USE_LOGJAC ? -0.5 * Q + logjac_box(collect(float.(θ)), spec.free) :
                         -0.5 * Q
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
# Promotion: write a better point where :warmstart will find it
# ------------------------------------------------------------------------
# smm_main.jl's warm start reads smm_result_{window}{suffix}.jls and needs a bundle
# carrying :result (an SMMResult) and :spec, so the better point is written in exactly
# that shape rather than teaching the loader a second format. The displaced bundle is
# backed up first, suffixed with the Q it held: no two rounds collide, and a directory
# of backups reads as the descent history of the optimise-sample loop.
#
# converged = false because the point came from a sampler, not from an optimiser
# meeting a stopping rule.
# Q at the seed, computed once: the callback needs it, the drift decomposition
# reuses it, and Q_ckpt tracks what the bundle currently holds.
Q_seed = smm_objective(θ0, spec)
Q_ckpt = Ref(Q_seed)

# The backup archives the bundle this run STARTED from, once. Later checkpoints
# overwrite freely: the descent history worth keeping is one entry per run, not one per
# incumbent, and the run's own log records the intermediate climb.
const _backed_up = Ref(false)

"""
    promote!(θ, Q_new, Q_old; label) -> nothing

Back up `seed_jls` on first call, then overwrite it with `θ`.
Callers guarantee `Q_new < Q_old`.
"""
function promote!(θ::Vector{Float64}, Q_new::Float64, Q_old::Float64; label::String)
    backup_jls = replace(seed_jls, r"\.jls$" => @sprintf("_backup_Q%.6f.jls", Q_seed))
    # The backup is taken once, from the original seed, so this also marks the first
    # promotion — which is the only one whose paths are worth printing.
    first_write = !_backed_up[]
    if first_write
        cp(seed_jls, backup_jls; force = true)
        _backed_up[] = true
    end

    cp_b, up_b, sp_b = unpack_θ(θ, spec)
    open(seed_jls, "w") do io
        serialize(io, (result = SMMResult(θ, _params_to_namedtuple(cp_b, up_b, sp_b, spec),
                                          Q_new, false, 0, spec),
                       spec = spec))
    end
    # One line per promotion. The backup filename and the destination path are the same
    # on every call — the backup is taken once, from the seed — so repeating them turns a
    # frequent event into four lines of identical boilerplate. They are printed once, on
    # the first promotion, where they are news.
    if first_write
        @printf("  checkpointing to %s, backup %s\n", basename(seed_jls), basename(backup_jls))
    end
    @printf("  %s: Q %.6f → %.6f (ΔQ=%.4f)\n", label, Q_old, Q_new, Q_old - Q_new)
    return nothing
end

# Live checkpoint. Raising MCMC_DRIFT_MAX lets the chain keep climbing instead of
# stopping at the first 25 log units, which makes the run long enough that finishing is
# no longer guaranteed — an interrupt, or a kill, would otherwise discard every point it
# found. This writes each new incumbent as it appears, so the best point survives
# regardless of how the run ends. Q is recovered from logπ = −Q/2 + logjac rather than
# re-solved, so the callback costs nothing.

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
    @threads for i in 1:size(Xp, 2)
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

res = MCMC_JAC_ONLY ?
    (draws = reshape(θ0, :, 1), chain = reshape(θ0, :, 1, 1), accept = NaN,
     lp = [logposterior(θ0)], N = 1, gens = 0, gens_requested = 0, burn = 0,
     lp_seed = logposterior(θ0), lp_best = logposterior(θ0), theta_best = θ0,
     drift = 0.0, aborted = false, n_replaced = 0, last_replace = 0) :
    run_demc(logposterior, θ0;
               N = MCMC_N, gens = MCMC_GENS, burn_frac = MCMC_BURN,
               CR = MCMC_CR, δ = MCMC_DELTA, parallel = MCMC_PARALLEL,
               b_add = MCMC_B_ADD, b_mult = MCMC_B_MULT,
               init = MCMC_INIT, init_screen = MCMC_INIT_SCREEN,
               init_scale = init_scale, init_width = init_width,
               outlier_iqr = MCMC_OUTLIER_IQR,
               print_every = MCMC_PRINT_EVERY,
               on_best = MCMC_CHECKPOINT ? checkpoint_best : nothing,
               check_every = MCMC_CHECK_EVERY, rhat_max = MCMC_RHAT_MAX,
               ess_min = MCMC_ESS_MIN, drift_max = MCMC_DRIFT_MAX,
               # The sequential stop. rhat_max/ess_min above are now REPORTED, not gated
               # on. Forwarded explicitly: the four SA settings that were computed here
               # and never passed cost four versions of silently-defaulted behaviour, and
               # check_forwarding.jl exists because of it.
               moves_min = MCMC_MOVES_MIN, drift_flat = MCMC_DRIFT_FLAT,
               acc_floor = MCMC_ACC_FLOOR,
               # For the convergence gate's at-a-bound exemption test, which has to
               # measure pile-up in constrained units.
               lb = [ps.lb for ps in spec.free], ub = [ps.ub for ps in spec.free],
               rng = MersenneTwister(MCMC_SEED))

# ========================================================================
# 5. Points at which to store the model moment vector, from which Ĝ is regressed.
#    Chain mode thins the retained draws; jac-only (or an aborted chain) uses a
#    local design around the seed. A finite difference is not used either way: the
#    step that keeps it local is comparable to tol_global, so it would measure
#    solver noise rather than curvature.
# ========================================================================
# With no usable chain (jac-only, or aborted before stationarity) the chain-based
# diagnostics and the quantile columns have no content; they are reported as NaN
# rather than computed from a trajectory.
use_design = MCMC_JAC_ONLY || res.aborted
chain_ok   = !use_design && res.gens > 0
n_kept     = size(res.draws, 2)
Xj = if use_design
    local_design(θ0, spec.free; n = MCMC_JAC_DRAWS, rng = MersenneTwister(MCMC_SEED + 1))
else
    sel = n_kept <= MCMC_JAC_DRAWS ? collect(1:n_kept) :
             round.(Int, range(1, n_kept; length = MCMC_JAC_DRAWS))
    res.draws[:, sel]
end
Msel = Matrix{Float64}(undef, K, size(Xj, 2))
buf  = [Vector{Float64}(undef, K) for _ in 1:nthreads()]
@printf("Storing moments at %d %s for Ĝ... ", size(Xj, 2),
        use_design ? "local-design points" : "thinned draws"); flush(stdout)
@threads for i in 1:size(Xj, 2)
    b = buf[threadid()]
    Q = smm_objective(view(Xj, :, i), spec; moments_out = b)
    @views Msel[:, i] .= isfinite(Q) ? b : NaN
end
keep    = vec(all(isfinite, Msel; dims = 1))
Msel    = Msel[:, keep]
draws_J = Xj[:, keep]
@printf("%d feasible.\n", size(Msel, 2)); flush(stdout)

# The screen pre-pass above already produced Ĝ from a local design around the seed. When
# the chain is the source of the draws we still want the chain-based Ĝ, so recompute;
# when the design is the source, the pre-pass Ĝ is the same object and is reused.
Ĝ, R2   = (need_pre && use_design && Ĝ_pre !== nothing) ?
              (Ĝ_pre, fill(NaN, K)) : jacobian_from_draws(draws_J, Msel, spec.free)
se_bnd, se_curv = se_bound_diagonal(Ĝ, spec.W, σ̂)
# Posterior SD of the pooled post-burn-in draws, in constrained units: the
# quasi-posterior standard error, and what LMR report (read_MCMC_chain.m takes std()
# of the pooled chain). Unlike the two Ĵ-based columns it never differentiates Q, so
# it stays valid where Q is only piecewise smooth — the reservation-cutoff softening
# in grids.jl is continuous but not differentiable, putting a kink wherever p*
# crosses a p-grid node.
se_chain = chain_ok ?
    [std([_to_constrained(res.draws[k, t], spec.free[k].lb, spec.free[k].ub)
          for t in 1:size(res.draws, 2)]) for k in 1:d] :
    fill(NaN, d)
# The posterior mean, in constrained units and coordinate by coordinate, as LMR do
# (read_MCMC_chain.m transforms each parameter and then averages). This is the reported
# estimator: CH Thm 2 gives it consistency and asymptotic equivalence to the extremum
# estimator without needing Q differentiable, which matters both because Q is piecewise
# smooth and because the flat directions hide better points than any descent path from θ̂
# can reach. Averaging in constrained space means θ̄ ≠ to_constrained(mean(t)), so Q(θ̄)
# is re-solved below rather than inferred.
θ̄_con = chain_ok ?
    [mean(_to_constrained(res.draws[k, t], spec.free[k].lb, spec.free[k].ub)
          for t in 1:size(res.draws, 2)) for k in 1:d] :
    fill(NaN, d)
# cov(Θ) is singular with fewer draws than parameters, so the cross-check is only
# meaningful when the chain supplied the design.
jgap    = chain_ok && size(draws_J, 2) > d ?
          curvature_check(draws_J, Ĝ, spec.W, spec.free) : NaN

# ========================================================================
# 6. Gate diagnostics and the parameter table (each quantity printed once).
# ========================================================================
rhat, ess = chain_ok ? split_rhat_ess(res.chain, res.burn) :
                       (fill(NaN, d), fill(NaN, d))
blo, bhi  = chain_ok ? boundary_mass(res.draws, spec.free) :
                       (fill(NaN, d), fill(NaN, d))
sgrow     = chain_ok ? spread_growth(res.chain, res.burn) : fill(NaN, d)

println("\n╔══════════════════════════════════════════════════════╗")
println("║  DE-MC Quasi-Posterior                               ║")
println("╠══════════════════════════════════════════════════════╣")
# A run that exhausts the cap has NOT met the stopping criteria: say so, and let the
# R̂/ESS columns below carry what was actually attained.
@printf("  gens=%d/%d %s  burn=%d  kept=%d  accept=%.3f\n",
        res.gens, res.gens_requested,
        MCMC_JAC_ONLY                        ? "(chain skipped: JAC_ONLY)"    :
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
        res.burn, n_kept, res.accept)
# Attribute the climb rather than assert a cause: −½ΔQ indicts the point estimate,
# Δlogjac only says the seed sat near a rail (the Jacobian term is unbounded below).
Q_best  = smm_objective(res.theta_best, spec)
# Under :flat_t there is no Jacobian in the target, so the whole drift is ΔQ by
# construction and the split has only one live component. Passing equal log-Jacobians
# rather than skipping the call keeps one code path and makes dlj print as exactly 0.
_lj0, _lj1 = _USE_LOGJAC ?
    (logjac_box(θ0, spec.free), logjac_box(collect(res.theta_best), spec.free)) :
    (0.0, 0.0)
dQ, dlj = drift_components(Q_seed, Q_best, _lj0, _lj1)
@printf("  seed drift=%+.1f (abort >%.1f) = %+.1f from Q (ΔQ=%+.4g) %+.1f from log|dθ/dt|\n",
        res.drift, MCMC_DRIFT_MAX, dQ, Q_best - Q_seed, dlj)
# The drift split decides what to do next, so say it rather than leave the reader to
# infer it from two signed numbers. ΔQ = 1 is one moment moving by one sampling
# standard error, which is the scale that makes an improvement worth re-seeding on.
if res.aborted && Q_best < Q_seed - 1.0
    @printf("  → the chain BEAT θ̂ by ΔQ=%.2f (%.1f sampling-SE units). theta_best is in\n",
            Q_seed - Q_best, Q_seed - Q_best)
    @printf("    the bundle: re-seed smm_main.jl from it (INIT_MODE = :warmstart) and\n")
    @printf("    re-estimate before trusting any standard error here.\n")
elseif res.aborted && dlj > abs(dQ)
    @printf("  → the climb is the Jacobian term, not Q: θ̂ sits near a box edge where\n")
    @printf("    log|dθ/dt| → −∞. Seed the railed coordinates interior; θ̂ stands.\n")
end
# A replacement inside the retained sample means those draws are not from the target,
# so report the boundary rather than only the count.
if res.n_replaced > 0
    @printf("  stuck-chain replacements: %d, last at g=%d (burn=%d)%s\n",
            res.n_replaced, res.last_replace, res.burn,
            res.last_replace > res.burn ? "  ← INSIDE the retained sample" : "")
end
@printf("  logπ(θ̂)=%.6e  max logπ=%.6e   gates: R̂≤%.2f, ESS≥%.0f%s\n",
        logposterior(θ0), maximum(res.lp), MCMC_RHAT_MAX,
        MCMC_ESS_MIN > 0 ? MCMC_ESS_MIN : min_ess(d),
        MCMC_ESS_MIN > 0 ? "" : @sprintf(" (minESS, d=%d)", d))
@printf("  target: %s   (MCMC_PRIOR = :%s)\n",
        _USE_LOGJAC ? "−½·g'Wg + log|dθ/dt|" : "−½·g'Wg",
        MCMC_PRIOR)
R2f = filter(isfinite, R2)
@printf("  Ĝ: %d×%d from %d draws, moment R² min=%.3f median=%.3f\n",
        K, d, size(Msel, 2), minimum(R2f), median(R2f))
@printf("  Ĵ from %s%s\n",
        use_design ? "local design (Ĵ = Ĝ'WĜ; CH Thm 4 needs no chain)" :
                     "chain covariance",
        chain_ok ? @sprintf("; cross-check |log10(diag ratio)| median = %.2f", jgap) : "")
println("╠══════════════════════════════════════════════════════╣")
# Two layouts. Without a chain the quantile, R̂, ESS and drift columns have nothing
# behind them, so printing them as NaN says only that the run took the no-chain path,
# which the header already states. |t| is the reportability screen; the flag column
# marks a parameter the theory cannot cover — at a box edge, or with R̂ over the gate.
if chain_ok
    # The rule is built from the header's own display width rather than a hardcoded
    # dash count, so the two cannot drift apart when a column is added or renamed.
    hdr = "  block   param     post.mean  se(chain)     |t|       θ̂      R̂   ESS edge%"
    println(hdr)
    println("  ", "─"^(textwidth(hdr) - 2))
else
    hdr = "  block   param     estimate    se(J⁻¹)   se(bound)     |t|"
    println(hdr)
    println("  ", "─"^(textwidth(hdr) - 2))
end
mkpath(SMM_OUT_DIR)
out_csv = joinpath(SMM_OUT_DIR, "mcmc_results_$(WINDOW)$(W_SUFFIX).csv")
open(out_csv, "w") do io
    # The first 14 columns keep their names and order so existing readers still parse;
    # se_chain is appended rather than inserted.
    # `symbol` is the market-suffixed display name (β_S rather than the bare β that
    # `name` carries for both markets), appended so tables and plots can label rows
    # without re-deriving the suffix. Existing columns keep their names and order.
    println(io, "block,name,label,point_estimate,post_mean,se_curvature,se_bound,q025,q500,q975,rhat,ess,edge_frac,spread_growth,se_chain,symbol")
    for (k, ps) in enumerate(spec.free)
        # Quantiles and the posterior mean require a stationary chain; without one
        # the se columns still stand (they come from Ĵ), so those are reported and
        # these are left blank rather than computed from a non-stationary walk.
        dk = chain_ok ?
             [_to_constrained(res.draws[k, t], ps.lb, ps.ub) for t in 1:n_kept] :
             Float64[]
        q(p) = isempty(dk) ? NaN : quantile(dk, p)
        pmean = isempty(dk) ? NaN : mean(dk)
        pe = _to_constrained(θ0[k], ps.lb, ps.ub)
        edge = blo[k] + bhi[k]
        # The reported SE is the chain's when there is one: it is the only column that
        # survives a non-differentiable objective. Otherwise fall back to the two
        # Ĵ-based columns and say so in the footer.
        se_rep = chain_ok ? se_chain[k] : se_curv[k]
        # |t| is formed against the REPORTED estimate: the posterior mean when a chain
        # ran, the seed otherwise. Pairing se(chain) with θ̂ would divide the width of
        # one point by the location of another.
        est_rep = chain_ok ? pmean : pe
        tstat   = se_rep > 0 ? abs(est_rep) / se_rep : NaN
        flag   = edge > 0.01 ? " edge" :
                 (chain_ok && rhat[k] > MCMC_RHAT_MAX) ? " R̂" : ""
        if chain_ok
            @printf("  %s%s%10.5f %10.5f %7.2f %8.5f %6.3f %6.0f %5.1f%s\n",
                    padr(ps.block, 8), padr(param_symbol(ps), 9),
                    pmean, se_chain[k], tstat, pe,
                    rhat[k], ess[k], 100edge, flag)
        else
            @printf("  %s%s%9.5f %10.5f %11.5f %7.2f%s\n",
                    padr(ps.block, 8), padr(param_symbol(ps), 9),
                    pe, se_curv[k], se_bnd[k], tstat, flag)
        end
        @printf(io, "%s,%s,%s,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.6f,%.1f,%.6f,%.6f,%.8f,%s\n",
                ps.block, ps.name, ps.label, pe, pmean,
                se_curv[k], se_bnd[k],
                q(0.025), q(0.500), q(0.975),
                rhat[k], ess[k], edge, sgrow[k], se_chain[k], param_symbol(ps))
    end
end
println("╚══════════════════════════════════════════════════════╝")
if chain_ok
    # Q at the mean, at the seed, and at the best visited point. The mean's Q is
    # normally worse than the best point's and better than nothing else in particular:
    # it is an average over the basin, not a competitor in a minimisation. Printing all
    # three stops the mean being read as a failed optimisation.
    θ̄_t = [_to_unconstrained(θ̄_con[k], spec.free[k].lb, spec.free[k].ub) for k in 1:d]
    Q_mean = smm_objective(θ̄_t, spec)
    @printf("\n  estimator: POSTERIOR MEAN of the pooled post-burn-in draws (CH 2003 Thm 2),\n")
    @printf("  paired with se(chain), its posterior SD. LMR report this same pair\n")
    @printf("  (read_MCMC_chain.m: mean and std of the pooled chain).\n")
    @printf("    Q(θ̄)=%.4f   Q(θ̂)=%.4f   Q(theta_best)=%.4f\n", Q_mean, Q_seed, Q_best)
    @printf("  θ̄ is an average over the basin, so its Q sits above the best point the\n")
    @printf("  chain visited — a single better point is one draw from a rugged surface,\n")
    @printf("  the mean is not. Report θ̄ ± se(chain); theta_best is a diagnostic.\n")
    @printf("  se(chain) posterior SD of the pooled draws — derivative-free, so it holds where\n")
    @printf("  Q is only piecewise smooth. se(J⁻¹) and se(bound) are in the CSV; both\n")
    @printf("  differentiate Q and are reported there for comparison only.\n")
    @printf("  R̂/ESS are DIAGNOSTIC, not gates: on a partly unidentified target the flat\n")
    @printf("  coordinates do not reach R̂ ≤ %.2f at any budget, and a high R̂ beside a wide\n", MCMC_RHAT_MAX)
    @printf("  se(chain) is the finding rather than a failure. edge%%: draws within 1%% of a\n")
    @printf("  box edge — no interval is valid there, by any route.\n")
else
    # Two ways to land here and they are not the same fact. Under JAC_ONLY no chain was
    # started. After an abort a chain ran — sometimes for thousands of generations — and
    # its draws exist; they are simply not a stationary sample. Reporting the second as
    # "no chain ran" tells the reader their run produced nothing, when in this project
    # such a run produced ΔQ = 200 and the point that every later estimate started from.
    if MCMC_JAC_ONLY
        @printf("\n  No chain ran (MCMC_JAC_ONLY), so there is no posterior SD.\n")
    else
        @printf("\n  The chain ran %d generations and ABORTED, so its %d draws are a record of\n",
                res.gens, n_kept)
        @printf("  where it went, not a stationary sample: no posterior SD can be read off\n")
        @printf("  them. theta_best is in the bundle and is the run's deliverable.\n")
    end
    @printf("  Both columns below differentiate Q, and Q is only piecewise smooth (the cutoff\n")
    @printf("  softening in grids.jl is C⁰, not C¹), so read them as indicative. se(J⁻¹)\n")
    @printf("  assumes W = Ω⁻¹, i.e. uncorrelated moment errors; se(bound) is sharp over all Ω\n")
    @printf("  but attained at a different rank-one adversarial Ω per parameter, so the column\n")
    @printf("  is not jointly attainable.\n")
    # The remedy differs by branch, and pointing an aborted run at JAC_ONLY=false is
    # advice it has already followed.
    if MCMC_JAC_ONLY
        @printf("  For reportable standard errors run with ROYSEARCH_MCMC_JAC_ONLY=false.\n")
    else
        @printf("  For reportable standard errors re-seed from theta_best and run again: a chain\n")
        @printf("  started at a point it cannot improve on has no drift to abort on.\n")
    end
end

# ========================================================================
# 7. Save chain + moments + Ĝ for plots and post-hoc reweighting
# ========================================================================
chain_jls = joinpath(SMM_OUT_DIR, "mcmc_chain_$(WINDOW)$(W_SUFFIX).jls")
open(chain_jls, "w") do io
    serialize(io, (chain      = res.chain,
                   draws      = res.draws,
                   # The reported estimate and its standard error. theta_mean averages
                   # over the basin, which is what makes it reportable: on a criterion
                   # with flat directions any single visited point is one draw from the
                   # ruggedness, and se_chain is the SD of the draws around this mean.
                   theta_mean = θ̄_con,
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
                   sigma_hat  = σ̂,
                   free       = [(ps.block, ps.name) for ps in spec.free],
                   labels     = [ps.label for ps in spec.free],
                   lb         = [ps.lb for ps in spec.free],
                   ub         = [ps.ub for ps in spec.free],
                   accept     = res.accept,
                   burn       = res.burn,
                   window     = WINDOW,
                   w_cond_target = W_COND_TARGET,
                   seed_jls   = seed_jls,
                   # The spec is what makes theta_best USABLE rather than merely recorded.
                   # smm_objective(theta, spec) needs the whole object — fixed parameters,
                   # moment targets, sim settings, grids, W — and free/labels/lb/ub above
                   # are not a substitute for any of it. Without this field the chain's
                   # best visited point cannot be re-evaluated, re-solved for moments, or
                   # warm-started from, which is the entire reason the chain is run: it
                   # reaches points the optimiser cannot, and a point that cannot be
                   # re-evaluated is not a result. Written for every chain bundle,
                   # independent of MCMC_CHECKPOINT — that flag governs whether the
                   # ESTIMATION's bundle is overwritten, a separate decision from whether
                   # this bundle is self-contained.
                   spec       = spec,
                   provenance = run_provenance(window = WINDOW, w_suffix = W_SUFFIX,
                                               version = ROYSEARCH_VERSION)))
end

# ========================================================================
# 8. Promote a better point into the estimation's own bundle
# ========================================================================
# The chain visits points the optimiser cannot reach: Metropolis accepts uphill moves,
# so it leaves basins Nelder-Mead is trapped in. When it finds one, that point is the
# run's most valuable output — but only if the next estimation can start from it.
#
# smm_main.jl's :warmstart reads smm_result_{window}{suffix}.jls and requires a bundle
# carrying :result (an SMMResult) and :spec. The chain bundle above has neither, so this
# writes a SECOND file in exactly that shape rather than teaching the warm-start loader
# a new format: the loop closes with no new reader and no hand-copied parameters.
#
# The threshold is ΔQ = 1, one moment moving by one sampling standard error. Below that
# the improvement sits inside the noise the moments themselves carry, and overwriting a
# published optimum for it would be churn. converged = false because this point came
# from a sampler, not from an optimiser meeting a stopping rule.
# Q_ckpt tracks what the bundle currently holds, so a run whose checkpoints already
# wrote the incumbent reports no further promotion rather than double-counting it.
# The reported estimate needs a bundle of its own, or the next window cannot start from
# it. promote! already writes the shape smm_main.jl's :warmstart requires (:result +
# :spec), so this reuses it at a SEPARATE path: the seed bundle is left untouched, and
# the two-stage base→crisis workflow points at whichever of the two it wants. Q(θ̄) is
# re-solved, not inferred, because averaging in constrained space can land somewhere the
# solver treats differently.
if chain_ok
    θ̄_t2 = [_to_unconstrained(θ̄_con[k], spec.free[k].lb, spec.free[k].ub) for k in 1:d]
    Q̄    = smm_objective(θ̄_t2, spec)
    mean_jls = joinpath(SMM_OUT_DIR, "smm_result_$(WINDOW)$(W_SUFFIX)_postmean.jls")
    if isfinite(Q̄)
        cp_m, up_m, sp_m = unpack_θ(θ̄_t2, spec)
        open(mean_jls, "w") do io
            serialize(io, (result = SMMResult(θ̄_t2,
                                              _params_to_namedtuple(cp_m, up_m, sp_m, spec),
                                              Q̄, false, 0, spec),
                           spec = spec,
                           provenance = run_provenance(window = WINDOW, w_suffix = W_SUFFIX,
                                                       version = ROYSEARCH_VERSION)))
        end
        @printf("\n  posterior mean bundle: Q(θ̄)=%.6f → %s\n", Q̄, basename(mean_jls))
        @printf("    warm-start the next window from it:\n")
        @printf("      cp %s %s\n", basename(mean_jls),
                basename(joinpath(SMM_OUT_DIR, "smm_result_$(WINDOW)$(W_SUFFIX).jls")))
        @printf("    then run smm_main.jl with INIT_MODE = :warmstart.\n")
    else
        @printf("\n  posterior mean is INFEASIBLE (Q=Inf): no bundle written. The mean of a\n")
        @printf("    non-convex feasible region can fall outside it; read this as the chain\n")
        @printf("    straddling a support boundary, and check edge%% above.\n")
    end
end

# MCMC_CHECKPOINT gates promotion at BOTH points it can happen: the during-run on_best
# callback above, and this end-of-run pass. Gating only the callback left the flag
# half-honoured — a "checkpointing off" run still overwrote the seed bundle here, and
# with theta_best, which is a diagnostic and never the estimate. On a reporting run the
# seed bundle must not move: se_chain is the width of the posterior around θ̄, and the
# bundle it is filed beside has to keep holding the point the run actually started from.
if MCMC_CHECKPOINT && Q_best < Q_ckpt[] - PROMOTE_MIN_DQ
    promote!(collect(res.theta_best), Q_best, Q_ckpt[]; label = "PROMOTED")
    Q_ckpt[] = Q_best
end
if Q_ckpt[] < Q_seed - PROMOTE_MIN_DQ
    @printf("\n  Bundle now holds Q %.6e (seed was %.6e, ΔQ=%.2f).\n",
            Q_ckpt[], Q_seed, Q_seed - Q_ckpt[])
    @printf("  Re-run smm_main.jl with INIT_MODE = :warmstart to optimise from it.\n")
else
    @printf("\n  No promotion: chain best Q %.6e vs the seed's %.6e — %s\n",
            Q_best, Q_seed,
            Q_best > Q_seed             ? "the chain found nothing better"          :
            !MCMC_CHECKPOINT            ? "checkpointing off, bundle left untouched" :
                                          @sprintf("gain below PROMOTE_MIN_DQ = %.1f", PROMOTE_MIN_DQ))
end

@printf("\nWrote %s\n       %s\n", out_csv, chain_jls)
flush(stdout)

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

println("="^60)
println("  RoySearch v15.5 — DE-MC standard errors")
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
WINDOW        = :base_fc          # window to compute SEs for
W_COND_TARGET = 0.0               # 0.0 = diagonal-σ, 2.0 = equal weights.
                                  # Selects BOTH the estimation bundle to load and
                                  # the weighting the chain targets — same meaning
                                  # and same admissible values as in smm_main.jl.
                                  # The held-out moment set is taken from the
                                  # bundle, so there is no SKIP_MOMENTS here.

# DE-MC controls (see smm/demc.jl). N = 0 ⇒ 2·d chains.
MCMC_N           = 32              # 2·d is ter Braak's minimum at d = 25; LMR run 95
                                   # chains for 16 parameters (5.9 per dimension) on an
                                   # MPI cluster with one rank per chain. On 10 cores the
                                   # binding constraint is total solves, and measurement
                                   # shows ESS depends on N·gens rather than on the split,
                                   # so fewer chains × more generations is preferred: at a
                                   # fixed budget that is what improves R̂.
MCMC_GENS        = 4000            # budget cap ≈ 128k solves ≈ 6.5 SMM runs, the point at
                                   # which the gates below are first met at d = 25 (measured:
                                   # R̂ 1.070, ESS 342; at 2 SMM runs it is R̂ 1.25, ESS 133).
                                   # LMR run 10,000
                                   # generations (main_mpi.f90: max_iteration), retaining
                                   # only the last 500 per chain in a ring buffer — their
                                   # "1,000 draws" is what is stored, not what is computed;
                                   # they discard 95% of 950,000 solves.
MCMC_BURN        = 0.5
MCMC_CR          = 0.90
MCMC_DELTA       = 1
MCMC_PARALLEL    = true           # thread population over chains (see demc.jl header)
MCMC_SEED        = 20260624
MCMC_PRINT_EVERY = 250            # generations between progress lines. The acc/fin
                                  # figures on each line average over exactly this
                                  # window, so a very short stride makes them noisy
                                  # (with N chains the finest resolution is 1/N).

# Sequential termination. MCMC_GENS becomes a BUDGET CAP: sampling stops as soon as
# R̂ and ESS both clear their thresholds, so a well-mixing window costs far less than
# the cap. MCMC_CHECK_EVERY = 0 disables the test and always runs the full budget.
# MCMC_ESS_MIN = 0.0 uses the Vats–Flegal–Jones minESS floor for the free dimension
# (≈2159 at d = 25, ε = 0.10) rather than a hand-picked number.
MCMC_CHECK_EVERY = 100
MCMC_RHAT_MAX    = 1.10            # Gelman et al. (2004) accept ≤1.1. Tighter values are
                                   # out of reach for DE-MC inside this budget: measured on
                                   # an isotropic Gaussian at d = 25 (the easiest target
                                   # this sampler will face), 80k solves gives R̂ ≈ 1.10 and
                                   # 156k gives ≈1.09, while 1.02 needs ~600k. Report the
                                   # attained R̂ rather than claiming a threshold not met.
MCMC_DRIFT_MAX   = 25.0            # Abort if the running max log-target climbs this far
                                   # above the seed. A stationary chain fluctuates within
                                   # O(d/2); the box Jacobian can shift the mode by at most
                                   # logjac_bound(free) − logjac_box(θ̂) (≈15 units for these
                                   # boxes), so a larger drift means the seed is not the
                                   # optimum of the criterion being sampled and Cov(chain)
                                   # would measure the walk toward it. 0.0 disables.
MCMC_JAC_ONLY    = false           # Skip the chain entirely: estimate Ĵ = Ĝ'WĜ from a local
                                   # design around the seed (≈10·d solves) instead of from
                                   # Cov(chain) (N·gens solves). CH Theorem 4 admits either.
                                   # The trade is that the reported quantile columns need the
                                   # chain, so they are omitted in this mode.
MCMC_ESS_MIN     = 250.0           # What the REPORTED numbers need, not the joint-volume
                                   # floor. The table carries se and a sandwich built from
                                   # Cov(chain): at ESS 250 a standard error has relative
                                   # MC error 4.5%, i.e. two stable significant figures.
                                   # minESS(25) at ε = 0.20 is ≈540 and at ε = 0.10 ≈2159;
                                   # those size the posterior-mean confidence VOLUME and
                                   # cost 600k–1.6M solves. Set 0.0 to use minESS instead.
MCMC_JAC_DRAWS   = 600            # thinned retained draws re-solved to store the
                                  # moment vector, from which Ĝ is regressed

derived_dir = joinpath(PROJECT_ROOT, "data", "derived")

_w_suffix(ct::Float64) = ct == 0.0 ? "_diagonalW" :
                         ct == 2.0 ? "_equalW" :
                         error("W_COND_TARGET must be 0.0 (diagonal-σ) or 2.0 (equal weights); got $ct.")
W_SUFFIX = _w_suffix(W_COND_TARGET)

@printf("Window: %s   weighting: %s\n", WINDOW, W_SUFFIX); flush(stdout)

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
# 3. Log quasi-posterior.  −½·g'Wg plus the box log-Jacobian, without which the
#    density induced on θ has poles at the bounds and no stationary
#    distribution (logjac_box, smm_params.jl). Infeasible θ → Inf → −Inf → the
#    proposal is rejected.
# ========================================================================
function logposterior(θ)
    Q = smm_objective(θ, spec)
    isfinite(Q) || return -Inf
    return -0.5 * Q + logjac_box(collect(float.(θ)), spec.free)
end

# ========================================================================
# 4. Run DE-MC
# ========================================================================
# run_demc prints its own d/N/gens/δ/CR/γ0 header, so the summary block below
# does not repeat those. In MCMC_JAC_ONLY mode the chain is skipped and `res` is a
# one-generation stand-in, so the downstream code has the same shape either way.
res = MCMC_JAC_ONLY ?
    (draws = reshape(θ0, :, 1), chain = reshape(θ0, :, 1, 1), accept = NaN,
     lp = [logposterior(θ0)], N = 1, gens = 0, gens_requested = 0, burn = 0,
     lp_seed = logposterior(θ0), lp_best = logposterior(θ0), theta_best = θ0,
     drift = 0.0, aborted = false) :
    run_demc(logposterior, θ0;
               N = MCMC_N, gens = MCMC_GENS, burn_frac = MCMC_BURN,
               CR = MCMC_CR, δ = MCMC_DELTA, parallel = MCMC_PARALLEL,
               print_every = MCMC_PRINT_EVERY,
               check_every = MCMC_CHECK_EVERY, rhat_max = MCMC_RHAT_MAX,
               ess_min = MCMC_ESS_MIN, drift_max = MCMC_DRIFT_MAX,
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

Ĝ, R2   = jacobian_from_draws(draws_J, Msel, spec.free)
se_bnd, se_curv = se_bound_diagonal(Ĝ, spec.W, σ̂)
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
        res.aborted                          ? "(ABORTED: seed drift)"        :
        res.gens < res.gens_requested        ? "(criteria met, stopped early)" :
        MCMC_CHECK_EVERY > 0                 ? "(BUDGET EXHAUSTED — criteria not met)" :
                                               "(no stopping test)",
        res.burn, n_kept, res.accept)
# Attribute the climb rather than assert a cause: −½ΔQ indicts the point estimate,
# Δlogjac only says the seed sat near a rail (the Jacobian term is unbounded below).
Q_seed  = smm_objective(θ0, spec)
Q_best  = smm_objective(res.theta_best, spec)
dQ, dlj = drift_components(Q_seed, Q_best,
                           logjac_box(θ0, spec.free),
                           logjac_box(collect(res.theta_best), spec.free))
@printf("  seed drift=%+.1f (abort >%.1f) = %+.1f from Q (ΔQ=%+.4g) %+.1f from log|dθ/dt|\n",
        res.drift, MCMC_DRIFT_MAX, dQ, Q_best - Q_seed, dlj)
@printf("  logπ(θ̂)=%.6e  max logπ=%.6e   gates: R̂≤%.2f, ESS≥%.0f%s\n",
        logposterior(θ0), maximum(res.lp), MCMC_RHAT_MAX,
        MCMC_ESS_MIN > 0 ? MCMC_ESS_MIN : min_ess(d),
        MCMC_ESS_MIN > 0 ? "" : @sprintf(" (minESS, d=%d)", d))
@printf("  target: −½·g'Wg + log|dθ/dt|   (log-Jacobian ON)\n")
R2f = filter(isfinite, R2)
@printf("  Ĝ: %d×%d from %d draws, moment R² min=%.3f median=%.3f\n",
        K, d, size(Msel, 2), minimum(R2f), median(R2f))
@printf("  Ĵ from %s%s\n",
        use_design ? "local design (Ĵ = Ĝ'WĜ; CH Thm 4 needs no chain)" :
                     "chain covariance",
        chain_ok ? @sprintf("; cross-check |log10(diag ratio)| median = %.2f", jgap) : "")
println("╠══════════════════════════════════════════════════════╣")
println("  block   parameter                     estimate    se(J⁻¹)   se(bound)     q025     q975    R̂    ESS  edge%  drift")
println("  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────")
mkpath(SMM_OUT_DIR)
out_csv = joinpath(SMM_OUT_DIR, "mcmc_results_$(WINDOW)$(W_SUFFIX).csv")
open(out_csv, "w") do io
    println(io, "block,name,label,point_estimate,post_mean,se_curvature,se_bound,q025,q500,q975,rhat,ess,edge_frac,spread_growth")
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
        @printf("  %-7s %-28s %9.5f %10.5f %11.5f %8.4f %8.4f %5.3f %6.0f %5.1f %6.2f\n",
                ps.block, ps.label, pe, se_curv[k], se_bnd[k],
                q(0.025), q(0.975),
                rhat[k], ess[k], 100edge, sgrow[k])
        @printf(io, "%s,%s,%s,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.6f,%.1f,%.6f,%.6f\n",
                ps.block, ps.name, ps.label, pe, pmean,
                se_curv[k], se_bnd[k],
                q(0.025), q(0.500), q(0.975),
                rhat[k], ess[k], edge, sgrow[k])
    end
end
println("╚══════════════════════════════════════════════════════╝")
@printf("\n  se(J⁻¹) CH Thm 3, exact only under W = Ω⁻¹.  se(bound) CH Thm 4, sharp over all Ω\n")
@printf("  consistent with σ̂ — cannot be too narrow.  edge%% within 1%% of a box edge (Thm needs\n")
@printf("  θ₀ interior).  drift last/first decile cross-chain SD: ≈1 sampled, ≫1 diffusing.\n")

# ========================================================================
# 7. Save chain + moments + Ĝ for plots and post-hoc reweighting
# ========================================================================
chain_jls = joinpath(SMM_OUT_DIR, "mcmc_chain_$(WINDOW)$(W_SUFFIX).jls")
open(chain_jls, "w") do io
    serialize(io, (chain      = res.chain,
                   draws      = res.draws,
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
                   seed_jls   = seed_jls))
end
@printf("\nWrote %s\n       %s\n", out_csv, chain_jls)
flush(stdout)

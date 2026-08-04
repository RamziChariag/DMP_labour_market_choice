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
println("  RoySearch v15.1 — DE-MC standard errors")
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
include(joinpath(SMM_DIR, "demc.jl"))
include(joinpath(SMM_DIR, "mcmc_diagnostics.jl"))
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
MCMC_N           = 0
MCMC_GENS        = 4000
MCMC_BURN        = 0.5
MCMC_CR          = 0.90
MCMC_DELTA       = 1
MCMC_PARALLEL    = true           # thread population over chains (see demc.jl header)
MCMC_SEED        = 20260624
MCMC_PRINT_EVERY = 250            # generations between progress lines. The acc/fin
                                  # figures on each line average over exactly this
                                  # window, so a very short stride makes them noisy
                                  # (with N chains the finest resolution is 1/N).
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
# does not repeat those.
res = run_demc(logposterior, θ0;
               N = MCMC_N, gens = MCMC_GENS, burn_frac = MCMC_BURN,
               CR = MCMC_CR, δ = MCMC_DELTA, parallel = MCMC_PARALLEL,
               print_every = MCMC_PRINT_EVERY,
               rng = MersenneTwister(MCMC_SEED))

# ========================================================================
# 5. Re-solve at a thinned subset of retained draws to store the model moment
#    vector per draw. Ĝ comes from regressing these on the draws, which is what
#    CH Theorem 4 and the Ω-free bound both need; a finite difference on this
#    solver is not stable enough at the step sizes involved.
# ========================================================================
n_kept = size(res.draws, 2)
sel    = n_kept <= MCMC_JAC_DRAWS ? collect(1:n_kept) :
         round.(Int, range(1, n_kept; length = MCMC_JAC_DRAWS))
Msel   = Matrix{Float64}(undef, K, length(sel))
buf    = [Vector{Float64}(undef, K) for _ in 1:nthreads()]
@printf("Storing moments at %d thinned draws for Ĝ... ", length(sel)); flush(stdout)
@threads for i in eachindex(sel)
    b = buf[threadid()]
    Q = smm_objective(view(res.draws, :, sel[i]), spec; moments_out = b)
    @views Msel[:, i] .= isfinite(Q) ? b : NaN
end
keep    = vec(all(isfinite, Msel; dims = 1))
Msel    = Msel[:, keep]
draws_J = res.draws[:, sel[keep]]
@printf("%d feasible.\n", size(Msel, 2)); flush(stdout)

Ĝ, R2   = jacobian_from_draws(draws_J, Msel, spec.free)
se_bnd, se_curv = se_bound_diagonal(Ĝ, spec.W, σ̂)
jgap    = curvature_check(draws_J, Ĝ, spec.W, spec.free)

# ========================================================================
# 6. Gate diagnostics and the parameter table (each quantity printed once).
# ========================================================================
rhat, ess = split_rhat_ess(res.chain, res.burn)
blo, bhi  = boundary_mass(res.draws, spec.free)
sgrow     = spread_growth(res.chain, res.burn)

println("\n╔══════════════════════════════════════════════════════╗")
println("║  DE-MC Quasi-Posterior                               ║")
println("╠══════════════════════════════════════════════════════╣")
@printf("  burn=%d  kept=%d  accept=%.3f   logπ(θ̂)=%.6e  max logπ=%.6e\n",
        res.burn, n_kept, res.accept, logposterior(θ0), maximum(res.lp))
@printf("  target: −½·g'Wg + log|dθ/dt|   (log-Jacobian ON)\n")
R2f = filter(isfinite, R2)
@printf("  Ĝ: %d×%d from %d draws, moment R² min=%.3f median=%.3f\n",
        K, d, size(Msel, 2), minimum(R2f), median(R2f))
@printf("  J cross-check |log10(diag ratio)| median = %.2f   (Cov(draws)⁻¹ vs Ĝ'WĜ)\n", jgap)
println("╠══════════════════════════════════════════════════════╣")
println("  block   parameter                     estimate    se(J⁻¹)   se(bound)     q025     q975    R̂    ESS  edge%  drift")
println("  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────")
mkpath(SMM_OUT_DIR)
out_csv = joinpath(SMM_OUT_DIR, "mcmc_results_$(WINDOW)$(W_SUFFIX).csv")
open(out_csv, "w") do io
    println(io, "block,name,label,point_estimate,post_mean,se_curvature,se_bound,q025,q500,q975,rhat,ess,edge_frac,spread_growth")
    for (k, ps) in enumerate(spec.free)
        dk = [_to_constrained(res.draws[k, t], ps.lb, ps.ub) for t in 1:n_kept]
        pe = _to_constrained(θ0[k], ps.lb, ps.ub)
        edge = blo[k] + bhi[k]
        @printf("  %-7s %-28s %9.5f %10.5f %11.5f %8.4f %8.4f %5.3f %6.0f %5.1f %6.2f\n",
                ps.block, ps.label, pe, se_curv[k], se_bnd[k],
                quantile(dk, 0.025), quantile(dk, 0.975),
                rhat[k], ess[k], 100edge, sgrow[k])
        @printf(io, "%s,%s,%s,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.6f,%.1f,%.6f,%.6f\n",
                ps.block, ps.name, ps.label, pe, mean(dk),
                se_curv[k], se_bnd[k],
                quantile(dk, 0.025), quantile(dk, 0.500), quantile(dk, 0.975),
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

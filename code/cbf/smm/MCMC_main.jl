############################################################
# smm/MCMC_main.jl — standalone standard-error script (DE-MC)
#
# Runs AFTER an SMM estimation. It does NOT touch smm_main.jl / smm.jl.
# It loads a saved SMM optimum, runs a Differential-Evolution MCMC
# (smm/demc.jl) on the DIAGONAL-weighted log quasi-posterior
#       logπ(θ) = −½ · Σ_k ((data_k − model_k) / σ̂_k)²
# which is exactly LMR's 1/SD² scheme (objective_function_mod.f90) and
# makes the chain covariance a valid standard-error estimate (moments
# treated as independent, as in LMR). It writes a parameter table you
# read later and match to the point estimates by (block, name):
#
#   output/smm/mcmc_results_{window}.csv   block,name,label,post_mean,se,
#                                          q025,q500,q975,point_estimate
#   output/smm/mcmc_chain_{window}.jls     raw draws (trace / R̂ / ESS / plots)
#
# Usage (from project root — threads strongly recommended):
#   julia --threads auto code/smm/MCMC_main.jl
#
# Point estimates stay from your SMM run (relative/equal weights); the SEs
# here are computed under the diagonal weight, seeded at that optimum.
############################################################

println("="^60)
println("  Segmented Search Model — DE-MC standard errors")
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
println("done.")
@printf("Threads available: %d\n\n", Threads.nthreads()); flush(stdout)

# ========================================================================
# CONFIG
# ========================================================================
WINDOW            = :base_fc      # the estimation you want SEs for
EST_W_COND_TARGET = 2.0           # W the ESTIMATION used → which seed bundle to load
                                  #   (you estimate with equal/relative weights = 2.0)
SKIP_MOMENTS      = Symbol[]      # MUST match the estimation's SKIP_MOMENTS

# DE-MC controls (see smm/demc.jl). N = 0 ⇒ 2·d chains.
MCMC_N        = 0
MCMC_GENS     = 4000
MCMC_BURN     = 0.5
MCMC_CR       = 0.90
MCMC_DELTA    = 1
MCMC_PARALLEL = true              # thread population over chains (see demc.jl header)
MCMC_SEED     = 20260624

derived_dir = joinpath(PROJECT_ROOT, "data", "derived")

_w_suffix(ct::Float64) = ct == 0.0 ? "_diagonalW" :
                         ct == 1.0 ? "_compressedW" :
                         ct == 2.0 ? "_equalW" : "_fullW"
EST_W_SUFFIX = _w_suffix(EST_W_COND_TARGET)

@printf("Window: %s   seed weighting: %s\n", WINDOW, EST_W_SUFFIX); flush(stdout)

# ========================================================================
# 1. Load the SMM optimum (point estimate + spec) to seed the chain
# ========================================================================
seed_jls = joinpath(SMM_OUT_DIR, "smm_result_$(WINDOW)$(EST_W_SUFFIX).jls")
isfile(seed_jls) || error(
    "No estimation bundle at $seed_jls — run smm_main.jl for WINDOW=$WINDOW " *
    "(W_COND_TARGET=$EST_W_COND_TARGET) first.")
bundle = _load_smm_bundle(seed_jls; delete_on_fail=false, label="estimation bundle")
isnothing(bundle) && error("Could not read $seed_jls.")

θ0    = collect(float.(bundle.result.theta_opt))   # free params, UNCONSTRAINED space
spec0 = bundle.spec
@printf("  Seed loaded: %d free params, SMM Q=%.6e (converged=%s)\n",
        length(θ0), bundle.result.loss_opt, bundle.result.converged); flush(stdout)

# ========================================================================
# 2. Diagonal weight  W = inv(diag Σ̂) = diag(1/σ̂²)  (LMR's 1/SD²)
# ========================================================================
W_diag  = load_weight_matrix(; window=WINDOW, derived_dir=derived_dir,
                               cond_target=0.0, skip_moments=SKIP_MOMENTS)
Q_SCALE = load_sigma_trace(; window=WINDOW, derived_dir=derived_dir,
                             skip_moments=SKIP_MOMENTS)

# ========================================================================
# 3. Rebuild the spec with the diagonal W. Everything else (free set, fixed
#    η/r/ν/φ, moments, grids, sim) is taken verbatim from the estimation
#    spec, so the seed θ0 is valid and the model is solved identically.
# ========================================================================
spec = build_smm_spec(
    spec0.moments, spec0.sim;
    fixed        = spec0.fixed,
    free_specs   = spec0.free,
    run          = spec0.run,
    W            = W_diag,
    q_scale      = Q_SCALE,
    skip_moments = SKIP_MOMENTS,
)
@assert length(θ0) == length(spec.free) "seed length ($(length(θ0))) ≠ free params " *
    "($(length(spec.free))) — SKIP_MOMENTS must match the estimation run."

# ========================================================================
# 4. Log quasi-posterior.  smm_objective returns g'Wg / q_scale, so multiply
#    q_scale back to recover g'Wg = Σ((data−model)/σ̂)². Infeasible θ
#    (non-converged solve etc.) → Inf → −Inf → the proposal is rejected.
# ========================================================================
function logposterior(θ)
    Q = smm_objective(θ, spec)
    return isfinite(Q) ? -0.5 * Q * spec.q_scale : -Inf
end

# quick feasibility check at the seed
let lp0 = logposterior(θ0)
    @printf("  logπ(θ̂) = %.6e\n", lp0); flush(stdout)
    isfinite(lp0) || error("Seed θ̂ is infeasible under the diagonal-W objective — " *
                           "check sigma_$(WINDOW).csv exists and SKIP_MOMENTS matches.")
end

# ========================================================================
# 5. Run DE-MC
# ========================================================================
println("\nRunning DE-MC..."); flush(stdout)
res = run_demc(logposterior, θ0;
               N = MCMC_N, gens = MCMC_GENS, burn_frac = MCMC_BURN,
               CR = MCMC_CR, δ = MCMC_DELTA, parallel = MCMC_PARALLEL,
               rng = MersenneTwister(MCMC_SEED))
@printf("DE-MC done.  acceptance=%.3f  pooled draws=%d\n",
        res.accept, size(res.draws, 2)); flush(stdout)

# ========================================================================
# 6. Posterior summaries in CONSTRAINED space → SE table (keyed block,name)
#    Transform each draw, THEN summarise (mean∘transform ≠ transform∘mean).
# ========================================================================
mkpath(SMM_OUT_DIR)
out_csv = joinpath(SMM_OUT_DIR, "mcmc_results_$(WINDOW).csv")
open(out_csv, "w") do io
    println(io, "block,name,label,post_mean,se,q025,q500,q975,point_estimate")
    for (k, ps) in enumerate(spec.free)
        dk = [_to_constrained(res.draws[k, t], ps.lb, ps.ub) for t in 1:size(res.draws, 2)]
        pe = _to_constrained(θ0[k], ps.lb, ps.ub)
        @printf(io, "%s,%s,%s,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                ps.block, ps.name, ps.label,
                mean(dk), std(dk),
                quantile(dk, 0.025), quantile(dk, 0.5), quantile(dk, 0.975), pe)
    end
end
@printf("Wrote SE table → %s\n", out_csv); flush(stdout)

# ========================================================================
# 7. Save raw chain (unconstrained) for diagnostics / posterior plots
# ========================================================================
chain_jls = joinpath(SMM_OUT_DIR, "mcmc_chain_$(WINDOW).jls")
open(chain_jls, "w") do io
    serialize(io, (chain  = res.chain,
                   draws  = res.draws,
                   free   = [(ps.block, ps.name) for ps in spec.free],
                   labels = [ps.label for ps in spec.free],
                   lb     = [ps.lb for ps in spec.free],
                   ub     = [ps.ub for ps in spec.free],
                   accept = res.accept,
                   window = WINDOW,
                   seed_jls = seed_jls))
end
@printf("Wrote raw chain → %s\n", chain_jls)

println("\nDone.  Read mcmc_results_$(WINDOW).csv and join the `se` column onto your")
println("point-estimate table by (block, name).")
flush(stdout)

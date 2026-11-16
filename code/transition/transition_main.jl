############################################################
# transition/transition_main.jl — transition entry point (RoySearch)
#
# Solves the perfect-foresight transition of each crisis, from the baseline steady state
# to the crisis steady state at the stored point estimates, and saves each to
#   output/transition/transition_<pair><W>_<point>.jls
# holding the whole path, both endpoint models, the path statistics, and every type's
# values and masses at date 0 and at the baseline (transition_types.jl): everything the
# counterfactual exercises and the plotting scripts read.  Nothing is plotted here.
#
# Usage (from the repo root):
#   julia --project=. --threads auto code/transition/transition_main.jl
# Setting: POINT_ESTIMATE below (SETTINGS.md, "Settings that choose a file").
############################################################

include(joinpath(@__DIR__, "..", "smm", "version.jl"))
println("="^60)
println("  RoySearch v$(ROYSEARCH_VERSION) — Transitions")
println("="^60)
flush(stdout)

include(joinpath(@__DIR__, "..", "paths.jl"))

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
using Serialization
using CSV
using DataFrames
using Dates
using Optim
using Clustering
using JSON3
println("done."); flush(stdout)

# The date steps run inside @threads regions; BLAS stays on one thread so it does not
# oversubscribe every core with the outer thread count.
LinearAlgebra.BLAS.set_num_threads(1)

print("Loading solver, SMM and transition modules... "); flush(stdout)
for f in ("grids", "params", "unskilled", "skilled", "solver", "equilibrium")
    include(joinpath(CODE_DIR, "solver", f * ".jl"))
end
for f in ("settings", "moments", "smm_params", "bundle", "smm")
    include(joinpath(CODE_DIR, "smm", f * ".jl"))
end
const TRANS_OUT_DIR = out_transition()
const WEIGHTING     = run_weighting()
const W_SUFFIX      = weighting_suffix(WEIGHTING)
for f in ("transition_params", "transition_values", "transition_solver", "transition_types",
          "transition_simulation")
    include(joinpath(CODE_DIR, "transition", f * ".jl"))
end
println("done."); flush(stdout)

# ============================================================
# Settings
# ============================================================
assert_no_legacy_env()

# The estimate both windows of each pair are read at: :mcmc the chains' posterior means
# (estimate_<window><W>_mcmc.jls, the estimates the paper reports), :smm the SMM estimates.
const POINT_ESTIMATE = :mcmc
POINT_ESTIMATE in (:smm, :mcmc) || error("POINT_ESTIMATE must be :smm or :mcmc, got :$POINT_ESTIMATE.")

# The outer fixed point.  The fine grid spans T_MAX months in N_STEPS half-month steps; the
# horizon beyond it is chosen by the accuracy test of transition_simulation.jl.  DAMP is
# the weight on the free-entry tightness in the damped update (TransitionParams).
const T_MAX   = 120.0
const N_STEPS = 240
const TOL     = 1e-4
const MAXIT   = 200
const DAMP    = 0.3

@printf("  Pairs:           %s\n", join(PAIRS, ", "))
@printf("  POINT_ESTIMATE:  %s\n", POINT_ESTIMATE)
@printf("  WEIGHTING:       %s (%s)\n", WEIGHTING, W_SUFFIX)
@printf("  Julia threads:   %d\n", Threads.nthreads())
println("="^60)
Threads.nthreads() == 1 &&
    @warn "Julia is running on ONE thread: each transition will take several times longer. " *
          "Start Julia with `--threads auto`, or in VS Code set `Julia: Num Threads` to `auto` " *
          "and restart the REPL."
flush(stdout)

# ============================================================
# Run
# ============================================================
for pair in Symbol.(PAIRS)
    t0 = time()
    run_transition_simulation(pair, W_SUFFIX; point = POINT_ESTIMATE, T_max = T_MAX,
                              N_steps = N_STEPS, tol = TOL, maxit = MAXIT, damp = DAMP)
    @printf("[%s] finished in %.1f min\n", pair, (time() - t0) / 60);  flush(stdout)
end

println("\n" * "="^60)
println("  All done.")
println("="^60)
flush(stdout)

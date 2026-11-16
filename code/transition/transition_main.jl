############################################################
# transition/transition_main.jl — entry point: the transition of each crisis (RoySearch)
#
# For each pair of PAIRS, solves the perfect-foresight transition from the baseline steady
# state to the crisis steady state at the stored point estimates and saves it to
#   output/transition/transition_<pair><W>_<point>.jls
# (path, endpoint models, path statistics, per-type values and masses at date 0 and at the
# baseline), the input of the counterfactuals and the transition figures.
#
# Usage (from the repo root):
#   julia --project=. --threads auto code/transition/transition_main.jl
# Settings: below (SETTINGS.md, "Settings that choose a file").
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

# The date steps are threaded; BLAS stays on one thread so it does not oversubscribe.
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

# The estimate both windows of each pair are read at: :mcmc the chains' posterior means (the
# estimates the paper reports), :smm the SMM estimates.
const POINT_ESTIMATE = :mcmc
POINT_ESTIMATE in (:smm, :mcmc) || error("POINT_ESTIMATE must be :smm or :mcmc, got :$POINT_ESTIMATE.")

# The fine grid (T_MAX months in N_STEPS steps; the horizon beyond it is chosen by the test
# of transition_simulation.jl) and the outer fixed point: tolerance on the free-entry gap,
# iteration cap, and DAMP, the weight on the free-entry tightness in the update.
const T_MAX   = 120.0
const N_STEPS = 240
const TOL     = 5e-4
const MAXIT   = 200
const DAMP    = 1.0

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

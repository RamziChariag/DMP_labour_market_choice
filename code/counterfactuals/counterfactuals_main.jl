############################################################
# counterfactuals_main.jl — entry point of the counterfactual exercises (RoySearch)
#
# Run code/transition/transition_main.jl first, at the same POINT_ESTIMATE: every exercise
# reads output/transition/transition_<crisis><W>_<point>.jls.
#
#   :held_margin    held_margin.jl    Exercise 1: each crisis with who trains and where the
#                                     trained search held at the baseline; unemployment per
#                                     head along both paths and at both end steady states
#   :incidence      incidence.jl      Exercise 2: each worker's date-0 value against her
#                                     baseline value, by type, state and training status
#   :training_cost  training_cost.jl  Exercise 3: the steady states of every window with the
#                                     training cost scaled by 0.95 and 1.05; elasticities
# Shared: cf_common.jl, welfare.jl (the steady-state social flow). Each exercise writes data
# only (CSV, JLS) and log.txt to output/counterfactuals/<exercise>/; nothing is plotted or
# tabulated here.
#
# Usage (from the repo root), or run the file in VS Code with `Julia: Num Threads` = auto:
#   julia --project=. --threads auto code/counterfactuals/counterfactuals_main.jl
# Settings: section 1.
############################################################

# ═══════════════════════════════════════════════════════════
# 1. USER CONFIGURATION
# ═══════════════════════════════════════════════════════════

# Which crises, from the transition pairs of paths.jl (PAIRS).
const CRISES = [:fc, :covid]

# Which exercises, in this order; any subset of [:held_margin, :incidence, :training_cost].
const EXERCISES = [:held_margin, :incidence, :training_cost]

# The point estimate the transitions were solved at (transition_main.jl, POINT_ESTIMATE):
# :mcmc the chains' posterior means, the estimates the paper reports; :smm the SMM estimates.
const POINT_ESTIMATE = :mcmc

# The outer fixed point of the held-margin paths: transition_main.jl's tolerance, iteration
# cap and damping.
const CF_TOL   = 5e-4
const CF_MAXIT = 200
const CF_DAMP  = 1.0

# ═══════════════════════════════════════════════════════════
# 2. PACKAGES
# ═══════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════
# 3. PATHS
# ═══════════════════════════════════════════════════════════
const CF_DIR         = @__DIR__
const CODE_ROOT      = normpath(joinpath(CF_DIR, ".."))
const SOLVER_DIR     = joinpath(CODE_ROOT, "solver")
const SMM_DIR        = joinpath(CODE_ROOT, "smm")
const TRANSITION_DIR = joinpath(CODE_ROOT, "transition")
include(joinpath(CODE_ROOT, "paths.jl"))
include(joinpath(SMM_DIR, "version.jl"))
const TRANS_OUT_DIR  = out_transition()

# ═══════════════════════════════════════════════════════════
# 4. LOAD SOLVER / SMM / TRANSITION MODULES
# ═══════════════════════════════════════════════════════════
print("Loading solver, SMM and transition modules... "); flush(stdout)
for f in ("grids", "params", "unskilled", "skilled", "solver", "equilibrium")
    include(joinpath(SOLVER_DIR, f * ".jl"))
end
for f in ("settings", "moments", "smm_params", "bundle", "smm")
    include(joinpath(SMM_DIR, f * ".jl"))
end
const WEIGHTING = run_weighting()
const W_SUFFIX  = weighting_suffix(WEIGHTING)
for f in ("transition_params", "transition_values", "transition_solver", "transition_types",
          "transition_simulation")
    include(joinpath(TRANSITION_DIR, f * ".jl"))
end
println("done."); flush(stdout)

# ═══════════════════════════════════════════════════════════
# 5. EXERCISES
# ═══════════════════════════════════════════════════════════
for f in ("cf_common", "welfare", "incidence", "held_margin", "training_cost")
    include(joinpath(CF_DIR, f * ".jl"))
end

const RUNNERS = Dict(:held_margin => run_held_margin, :incidence => run_incidence,
                     :training_cost => run_training_cost)
all(e -> haskey(RUNNERS, e), EXERCISES) ||
    error("EXERCISES = $(EXERCISES): each must be one of $(collect(keys(RUNNERS))).")
all(c -> String(c) in PAIRS, CRISES) || error("CRISES = $(CRISES): each must be one of $(PAIRS).")

# ═══════════════════════════════════════════════════════════
# 6. BANNER
# ═══════════════════════════════════════════════════════════
println("\n" * "="^65)
println("  RoySearch v$(ROYSEARCH_VERSION) — Counterfactuals")
println("="^65)
@printf("  Crises:               %s\n", join(CRISES, ", "))
@printf("  Exercises:            %s\n", join(EXERCISES, ", "))
@printf("  POINT_ESTIMATE:       %s\n", POINT_ESTIMATE)
@printf("  Transitions read:     %s\n",
        join((basename(transition_file(c, W_SUFFIX, POINT_ESTIMATE)) for c in CRISES), ", "))
@printf("  Threads available:    %d\n", Threads.nthreads())
println("="^65, "\n")
Threads.nthreads() == 1 &&
    @warn "Julia is running on ONE thread: every exercise that solves a path will take several " *
          "times longer. Start Julia with `--threads auto`, or in VS Code set `Julia: Num Threads` " *
          "to `auto` and restart the REPL."
flush(stdout)

# ═══════════════════════════════════════════════════════════
# 7. RUN
# ═══════════════════════════════════════════════════════════
for e in EXERCISES
    t0 = time()
    RUNNERS[e](CRISES)
    @printf("\n[%s] finished in %.1f min → %s\n", e, (time() - t0) / 60, out_counterfactuals(e))
    flush(stdout)
end

println("\n" * "="^65)
println("  All done.")
println("="^65)
flush(stdout)

############################################################
# counterfactuals_main.jl — entry point of the counterfactual exercises (RoySearch)
#
# Run code/transition/transition_main.jl first, at the same POINT_ESTIMATE: every exercise
# reads output/transition/transition_<crisis><W>_<point>.jls.
#
#   :welfare        welfare.jl        social flow and its present value along each
#                                     transition, by market and component, against both
#                                     steady states
#   :incidence      incidence.jl      each type's date-0 value against the baseline, by
#                                     aptitude pair and state
#   :held_margin    held_margin.jl    each transition with τ and d held at the baseline
#   :training_cost  training_cost.jl  the training cost scaled from date 0, with and
#                                     without each crisis
# Shared: cf_common.jl. Outputs (CSV tables, JLS data for the figures, log.txt) go to
# output/counterfactuals/<exercise>/; nothing is plotted here.
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

# Which exercises, in this order; any subset of CF_EXERCISES (paths.jl).
const EXERCISES = [:welfare, :incidence, :held_margin, :training_cost]

# The point estimate the transitions were solved at (transition_main.jl, POINT_ESTIMATE):
# :mcmc the chains' posterior means, the estimates the paper reports; :smm the SMM estimates.
const POINT_ESTIMATE = :mcmc

# Held margin, run in each mode listed: :free_entry re-solves tightness by free entry at the
# estimated vacancy costs, scans the held crisis economy for every tightness pair at which
# both free-entry conditions hold, and follows the estimate's unskilled branch;
# :tightness_held keeps tightness on the estimated transition's path (one backward and one
# forward pass) and reports the free-entry gap this leaves.
const HELD_MODES = [:free_entry, :tightness_held]

# Training cost: the scales of c applied from date 0.  Must contain 1.0, the estimate.
const TRAINING_COST_GRID = [0.8, 0.9, 1.0, 1.1, 1.2]

# The outer fixed point of every path that is solved again (held margin under free entry,
# training cost): transition_main.jl's tolerance, iteration cap and damping.
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

all(e -> e in CF_EXERCISES, EXERCISES) ||
    error("EXERCISES = $(EXERCISES): each must be one of $(CF_EXERCISES).")
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
@printf("  HELD_MODES:           %s\n", join(HELD_MODES, ", "))
@printf("  TRAINING_COST_GRID:   %s\n", join(TRAINING_COST_GRID, ", "))
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
const RUNNERS = Dict(:welfare => run_welfare, :incidence => run_incidence,
                     :held_margin => run_held_margin, :training_cost => run_training_cost)
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

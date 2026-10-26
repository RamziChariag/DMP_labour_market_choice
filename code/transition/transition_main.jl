############################################################
# transition_main.jl — Single-entry-point orchestrator (RoySearch)
#
# One run does everything:
#   1. (Optional) run the transition simulation for each scenario and save
#      output/transition/transition_<sc><W>_<point>.jls
#   2. (Optional) generate the transition panels (data-vs-model and
#      model-decomposition)
#   3. (Optional) generate model-fit & parameter LaTeX tables, the
#      training-cutoff table, and the model-fit scatter
#
# Sub-scripts:
#   transition_simulation.jl   — run_transition_simulation(...)
#   plots_and_tables/transition_panel.jl — make_transition_panel(...),
#                                make_model_decomposition_panel(...)
#   plots_and_tables/transition.jl — load_all_smm_bundles!(; scenarios),
#                                make_model_fit_tables(), make_parameter_tables(),
#                                make_xbar_table(), make_model_fit_scatter(...)
#
# Usage (from the repo root):
#   julia --project=. --threads auto code/transition/transition_main.jl
#
# All user-facing switches live in section 1.
############################################################

# ═══════════════════════════════════════════════════════════
# 1. USER CONFIGURATION
# ═══════════════════════════════════════════════════════════

# Which scenarios to run, from the transition pairs of paths.jl (PAIRS), three per event:
# :fc and :covid run from the base estimate to the crisis estimate, :fc_cf1 / :covid_cf1
# to the cf1 estimate (crisis_<event>_cf1, its choices held at the base equilibrium on the
# whole path) and :fc_cf2 / :covid_cf2 to the cf2 estimate. Every scenario writes the same
# files, named by the scenario.
const SCENARIOS = [:fc, :covid]

# The estimate both windows of a pair are read at; never one of each.
#   :smm  → estimate_<window><W>_smm.jls, the SMM estimates; the crisis one must hold the
#           ability distribution at the base one (smm_main.jl, CRISIS_BASE = :smm)
#   :mcmc → estimate_<window><W>_mcmc.jls, the chains' estimates (the posterior mean of the
#           retained draws); a chain that aborted has none
# Every file the run writes ends in `<W>_<point>`, so the runs at the two points coexist.
# The tables phase reads this point's transition files, stops on one that records another,
# and reports the bundles they record: exactly the points the paths were solved at.
const POINT_ESTIMATE = :smm

# Simulation switch
#   true  → always (re)run the transition simulation for every SCENARIO
#   false → reuse the existing transition_<sc><W>_<point>.jls if on disk; otherwise
#           warn and run a fresh simulation anyway.
const RERUN_SIMULATION = true

# Plot/table switches
const RUN_PANELS           = true   # transition-dynamics panels
const RUN_TABLES_AND_PLOTS = true   # model-fit + parameter tables, scatter

# Transition-algorithm control knobs.  The grids are the estimation's own, read from the
# bundles; the horizon is chosen by the accuracy test in transition_simulation.jl.
const TR_T_MAX   = 120.0   # span of the fine time grid, model months
const TR_N_STEPS = 240     # half-month steps on the fine grid
const TR_TOL     = 1e-4
const TR_MAXIT   = 200
const TR_DAMP    = 0.3   # matches the TransitionParams default; see its docstring

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
using Plots
using LaTeXStrings
using Dates
using Optim
using Clustering
using JSON3

println("done."); flush(stdout)

# The dense LU / kernel BLAS calls run inside a @threads region, so multi-threaded
# BLAS would oversubscribe every core with the outer thread count.  Pin BLAS to one
# thread and let Base.Threads own the parallelism.
LinearAlgebra.BLAS.set_num_threads(1)

# ═══════════════════════════════════════════════════════════
# 3. PATHS  (single source of truth — sub-scripts re-use these)
# ═══════════════════════════════════════════════════════════
const TRANSITION_DIR = @__DIR__
const CODE_ROOT      = normpath(joinpath(TRANSITION_DIR, ".."))
const SOLVER_DIR     = joinpath(CODE_ROOT, "solver")
const SMM_DIR        = joinpath(CODE_ROOT, "smm")
const PT_DIR         = joinpath(CODE_ROOT, "plots_and_tables")
# paths.jl is the sole definition of PROJECT_ROOT, OUTPUT_DIR and the out_*() accessors.
include(joinpath(CODE_ROOT, "paths.jl"))
include(joinpath(SMM_DIR, "version.jl"))
const SMM_OUT_DIR    = out_estimates()
const TRANS_OUT_DIR  = out_transition()
const PLOTS_DIR      = joinpath(OUTPUT_DIR, "plots")
const TABLES_DIR     = joinpath(OUTPUT_DIR, "tables")
const DERIVED_DIR    = joinpath(PROJECT_ROOT, "data", "derived")

mkpath(TRANS_OUT_DIR)
mkpath(PLOTS_DIR)
mkpath(TABLES_DIR)

# ═══════════════════════════════════════════════════════════
# 4. LOAD SOLVER / SMM / TRANSITION MODULES
# ═══════════════════════════════════════════════════════════
print("Loading solver modules... "); flush(stdout)
include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))
println("done."); flush(stdout)

print("Loading SMM modules... "); flush(stdout)
include(joinpath(SMM_DIR, "settings.jl"))
include(joinpath(SMM_DIR, "moments.jl"))
include(joinpath(SMM_DIR, "smm_params.jl"))
include(joinpath(SMM_DIR, "bundle.jl"))
include(joinpath(SMM_DIR, "smm.jl"))
println("done."); flush(stdout)

# The weighting the estimates were produced under (WEIGHTING, run_weighting in moments.jl),
# the same setting smm_main.jl reads. A mismatch is not a silent one: estimate_path() names a
# file that is not on disk and the run stops at the first load.
const WEIGHTING = run_weighting()
const W_SUFFIX  = weighting_suffix(WEIGHTING)

print("Loading transition modules... "); flush(stdout)
include(joinpath(TRANSITION_DIR, "transition_params.jl"))
include(joinpath(TRANSITION_DIR, "transition_values.jl"))
include(joinpath(TRANSITION_DIR, "transition_solver.jl"))
println("done."); flush(stdout)

# ═══════════════════════════════════════════════════════════
# 5. INCLUDE SUB-SCRIPTS (function definitions only)
# ═══════════════════════════════════════════════════════════
print("Loading simulation / panel / table sub-scripts... "); flush(stdout)
include(joinpath(TRANSITION_DIR, "transition_simulation.jl"))
include(joinpath(PT_DIR, "transition_panel.jl"))
include(joinpath(PT_DIR, "transition.jl"))
println("done."); flush(stdout)

# ═══════════════════════════════════════════════════════════
# 6. BANNER
# ═══════════════════════════════════════════════════════════
println("\n" * "="^65)
println("  RoySearch v$(ROYSEARCH_VERSION) — Transition Run")
println("="^65)
@printf("  Scenarios:            %s\n", join(SCENARIOS, ", "))
@printf("  POINT_ESTIMATE:       %s\n", POINT_ESTIMATE)
@printf("  WEIGHTING:            %s  (%s)\n", WEIGHTING, W_SUFFIX)
@printf("  RERUN_SIMULATION:     %s\n", RERUN_SIMULATION)
@printf("  RUN_PANELS:           %s\n", RUN_PANELS)
@printf("  RUN_TABLES_AND_PLOTS: %s\n", RUN_TABLES_AND_PLOTS)
@printf("  Threads available:    %d\n", Threads.nthreads())
println("="^65, "\n")
flush(stdout)

# ═══════════════════════════════════════════════════════════
# 7. SIMULATION PHASE  (per scenario)
# ═══════════════════════════════════════════════════════════
for scenario in SCENARIOS
    out_file = transition_file(scenario, W_SUFFIX, POINT_ESTIMATE)

    if RERUN_SIMULATION
        @printf("[%s] RERUN_SIMULATION=true → running a new simulation.\n", scenario)
        flush(stdout)
        run_transition_simulation(scenario, W_SUFFIX;
            point = POINT_ESTIMATE, T_max = TR_T_MAX, N_steps = TR_N_STEPS,
            tol = TR_TOL, maxit = TR_MAXIT, damp = TR_DAMP)
    elseif isfile(out_file)
        @printf("[%s] Re-using existing transition data: %s\n", scenario, out_file)
        flush(stdout)
    else
        @warn "[$scenario] No existing transition data at $out_file — running a fresh simulation anyway."
        run_transition_simulation(scenario, W_SUFFIX;
            point = POINT_ESTIMATE, T_max = TR_T_MAX, N_steps = TR_N_STEPS,
            tol = TR_TOL, maxit = TR_MAXIT, damp = TR_DAMP)
    end
end

# ═══════════════════════════════════════════════════════════
# 8. PANELS PHASE
# ═══════════════════════════════════════════════════════════
if RUN_PANELS
    println("\n" * "="^65)
    println("  Generating transition panels")
    println("="^65); flush(stdout)
    for scenario in SCENARIOS
        make_transition_panel(; scenario = scenario, point = POINT_ESTIMATE, suffix = W_SUFFIX)
        make_model_decomposition_panel(; scenario = scenario, point = POINT_ESTIMATE,
                                         suffix = W_SUFFIX)
    end
end

# ═══════════════════════════════════════════════════════════
# 9. TABLES & SCATTER PHASE
# ═══════════════════════════════════════════════════════════
if RUN_TABLES_AND_PLOTS
    println("\n" * "="^65)
    println("  Generating model-fit & parameter tables, scatter plots")
    println("="^65); flush(stdout)

    load_all_smm_bundles!(; scenarios = SCENARIOS, point = POINT_ESTIMATE, suffix = W_SUFFIX)
    tag = transition_tag(W_SUFFIX, POINT_ESTIMATE)
    make_model_fit_tables(; suffix = tag)
    make_parameter_tables(; suffix = tag)
    make_xbar_table(; suffix = tag)
    for w in [:base_fc, :base_covid]
        make_model_fit_scatter(; window = w, suffix = tag)
    end
end

println("\n" * "="^65)
println("  All done.")
println("="^65)
flush(stdout)

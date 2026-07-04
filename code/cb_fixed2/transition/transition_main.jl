############################################################
# transition_main.jl — Single-entry-point orchestrator
#                       (PS(x) = γ·x^{γ−1} variant)
#
# One main run does everything:
#   1. (Optional) run the transition simulation for each scenario
#      and save  output/transition/transition_<sc><W>.jld2
#   2. (Optional) generate the transition panels
#      (data-vs-model and model-decomposition)
#   3. (Optional) generate model-fit & parameter LaTeX tables,
#      the training-cutoff table, and the model-fit scatter
#
# Sub-scripts:
#   transition_simulation.jl   — defines run_transition_simulation(...)
#   transition_panel.jl        — defines make_transition_panel(...) and
#                                       make_model_decomposition_panel(...)
#   plots_and_tables.jl        — defines load_all_smm_bundles!(),
#                                       make_model_fit_tables(),
#                                       make_parameter_tables(),
#                                       make_xbar_table(),
#                                       make_model_fit_scatter(...)
#
# Usage (from project root):
#   julia --threads auto code/transition/transition_main.jl
#
# All user-facing switches live in section 1 below.
############################################################

# ═══════════════════════════════════════════════════════════
# 1. USER CONFIGURATION
# ═══════════════════════════════════════════════════════════

# Which scenarios to run.  Use [:fc], [:covid], or [:fc, :covid].
const SCENARIOS = [:fc, :covid]

# Weight-matrix mode used when the SMM results were estimated.
# Must match the W_COND_TARGET that was set in code/smm/smm_main.jl.
#   0.0  →  "_diagonalW"
#   1.0  →  "_compressedW"
#   2.0  →  "_equalW"
#   >2.0 →  "_fullW"
const W_COND_TARGET = 2.0

# Simulation switch
#   true  → always (re)run the transition simulation for every SCENARIO
#   false → use the existing transition_<scenario><W>.jld2 if it is
#           on disk; if it is NOT, print a warning and run a fresh
#           simulation anyway.
const RERUN_SIMULATION = true

# Plot/table switches
const RUN_PANELS           = true   # transition-dynamics 3×2 panels
const RUN_TABLES_AND_PLOTS = true   # model-fit + parameter tables, scatter

# Grid sizes for the stationary-equilibrium solves (transition uses these).
# Intentionally finer than the SMM run for clean convergence on a single
# parameter draw — see the original transition_main for the rationale.
const TR_Nx    = 200
const TR_Np_U  = 200
const TR_Np_S  = 200

# Transition-algorithm control knobs
const TR_T_MAX   = 120.0   # model months (10 years)
const TR_N_STEPS = 240     # half-month steps
const TR_TOL     = 1e-4
const TR_MAXIT   = 200
const TR_DAMP    = 0.3

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
using JLD2
using CSV
using DataFrames
using Plots
using LaTeXStrings
using Arrow
using Dates
using Optim
using Clustering

println("done."); flush(stdout)

# ═══════════════════════════════════════════════════════════
# 3. PATHS  (single source of truth — sub-scripts re-use these)
# ═══════════════════════════════════════════════════════════
const TRANSITION_DIR = @__DIR__
const PROJECT_ROOT   = joinpath(TRANSITION_DIR, "..", "..")
const SOLVER_DIR     = joinpath(PROJECT_ROOT, "code", "solver")
const SMM_DIR        = joinpath(PROJECT_ROOT, "code", "smm")
const OUTPUT_DIR     = joinpath(PROJECT_ROOT, "output")
const SMM_OUT_DIR    = joinpath(OUTPUT_DIR, "smm")
const TRANS_OUT_DIR  = joinpath(OUTPUT_DIR, "transition")
const PLOTS_DIR      = joinpath(OUTPUT_DIR, "plots")
const TABLES_DIR     = joinpath(OUTPUT_DIR, "tables")
const DERIVED_DIR    = joinpath(PROJECT_ROOT, "data", "derived")

mkpath(TRANS_OUT_DIR)
mkpath(PLOTS_DIR)
mkpath(TABLES_DIR)

# Weight-matrix suffix helper (also re-exposed for the sub-scripts).
function _w_suffix(cond_target::Float64)
    cond_target == 0.0 && return "_diagonalW"
    cond_target == 1.0 && return "_compressedW"
    cond_target == 2.0 && return "_equalW"
    return "_fullW"
end
const W_SUFFIX = _w_suffix(W_COND_TARGET)

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
include(joinpath(SMM_DIR, "moments.jl"))
include(joinpath(SMM_DIR, "smm_params.jl"))
include(joinpath(SMM_DIR, "smm.jl"))
println("done."); flush(stdout)

print("Loading transition modules... "); flush(stdout)
include(joinpath(TRANSITION_DIR, "transition_params.jl"))
include(joinpath(TRANSITION_DIR, "transition_solver.jl"))
println("done."); flush(stdout)

# ═══════════════════════════════════════════════════════════
# 5. INCLUDE SUB-SCRIPTS (function definitions only)
# ═══════════════════════════════════════════════════════════
print("Loading simulation / panel / table sub-scripts... "); flush(stdout)
include(joinpath(TRANSITION_DIR, "transition_simulation.jl"))
include(joinpath(TRANSITION_DIR, "transition_panel.jl"))
include(joinpath(TRANSITION_DIR, "plots_and_tables.jl"))
println("done."); flush(stdout)

# ═══════════════════════════════════════════════════════════
# 6. BANNER
# ═══════════════════════════════════════════════════════════
println("\n" * "="^65)
println("  Segmented Search Model — Transition Run (PS(x) variant)")
println("="^65)
@printf("  Scenarios:            %s\n", join(SCENARIOS, ", "))
@printf("  W_COND_TARGET:        %.1f  (%s)\n", W_COND_TARGET, W_SUFFIX)
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
    out_file = joinpath(TRANS_OUT_DIR,
                        "transition_$(scenario)$(W_SUFFIX).jld2")

    if RERUN_SIMULATION
        @printf("[%s] RERUN_SIMULATION=true → running a new simulation.\n",
                scenario)
        flush(stdout)
        run_transition_simulation(
            scenario, W_COND_TARGET;
            Nx      = TR_Nx,
            Np_U    = TR_Np_U,
            Np_S    = TR_Np_S,
            T_max   = TR_T_MAX,
            N_steps = TR_N_STEPS,
            tol     = TR_TOL,
            maxit   = TR_MAXIT,
            damp    = TR_DAMP,
        )
    elseif isfile(out_file)
        @printf("[%s] Re-using existing transition data: %s\n",
                scenario, out_file)
        flush(stdout)
    else
        @warn "[$scenario] No existing transition data at $out_file — " *
              "running a fresh simulation anyway."
        run_transition_simulation(
            scenario, W_COND_TARGET;
            Nx      = TR_Nx,
            Np_U    = TR_Np_U,
            Np_S    = TR_Np_S,
            T_max   = TR_T_MAX,
            N_steps = TR_N_STEPS,
            tol     = TR_TOL,
            maxit   = TR_MAXIT,
            damp    = TR_DAMP,
        )
    end
end

# ═══════════════════════════════════════════════════════════
# 8. PANELS PHASE
# ═══════════════════════════════════════════════════════════
if RUN_PANELS
    println("\n" * "="^65)
    println("  Generating transition panels")
    println("="^65)
    flush(stdout)
    for scenario in SCENARIOS
        make_transition_panel(; scenario = scenario, suffix = W_SUFFIX)
        make_model_decomposition_panel(; scenario = scenario, suffix = W_SUFFIX)
    end
end

# ═══════════════════════════════════════════════════════════
# 9. TABLES & SCATTER PHASE
# ═══════════════════════════════════════════════════════════
if RUN_TABLES_AND_PLOTS
    println("\n" * "="^65)
    println("  Generating model-fit & parameter tables, scatter plots")
    println("="^65)
    flush(stdout)

    load_all_smm_bundles!()       # populates the global `smm_bundles` Dict
    make_model_fit_tables()
    make_parameter_tables()
    make_xbar_table()
    for w in [:base_fc, :base_covid]
        make_model_fit_scatter(; window = w)
    end
end

println("\n" * "="^65)
println("  All done.")
println("="^65)
flush(stdout)

############################################################
# code/transition/main.jl — Transition dynamics entry point
#
# Usage (from project root):
#   julia --threads auto code/transition/main.jl [scenario]
#
# Scenarios:
#   fc       Financial Crisis  (base_fc → crisis_fc)   [default]
#   covid    COVID-19          (base_covid → crisis_covid)
#
# Prerequisites:
#   SMM must have been run for BOTH the baseline and crisis
#   windows of the chosen scenario.  The script loads saved
#   SMM outputs from output/smm/ and checks that both exist.
#
# Examples:
#   julia --threads auto code/transition/main.jl fc
#   julia --threads auto code/transition/main.jl covid
#
# Project layout:
#   code/
#     solver/          ← model solver library
#     smm/             ← SMM estimation
#     transition/
#       main.jl        ← this file
#       transition_params.jl
#       transition_solver.jl
#   output/
#     smm/             ← saved SMM results per window
#     transition/      ← transition dynamics output
############################################################

println("="^60)
println("  Segmented Search Model — Transition Dynamics")
println("="^60)
flush(stdout)

# ── Paths ─────────────────────────────────────────────────────────────────────
const TRANSITION_DIR = @__DIR__
const SOLVER_DIR     = joinpath(TRANSITION_DIR, "..", "solver")
const SMM_DIR        = joinpath(TRANSITION_DIR, "..", "smm")
const PROJECT_ROOT   = joinpath(TRANSITION_DIR, "..", "..")
const OUTPUT_DIR     = joinpath(PROJECT_ROOT, "output")
const SMM_OUTPUT_DIR = joinpath(OUTPUT_DIR, "smm")
const TRANS_OUTPUT_DIR = joinpath(OUTPUT_DIR, "transition")
const DERIVED_DIR    = joinpath(PROJECT_ROOT, "data", "derived")

# ── Packages ──────────────────────────────────────────────────────────────────
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
using Optim
using Serialization
using CSV
using DataFrames

println("done."); flush(stdout)

Random.seed!(1234)

# ── Load solver (as a library) ────────────────────────────────────────────────
print("Loading solver modules... "); flush(stdout)

include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))

println("done."); flush(stdout)

# ── Load SMM scripts ──────────────────────────────────────────────────────────
print("Loading SMM modules... "); flush(stdout)

include(joinpath(SMM_DIR, "moments.jl"))
include(joinpath(SMM_DIR, "smm_params.jl"))
include(joinpath(SMM_DIR, "smm.jl"))

println("done."); flush(stdout)

# ── Load transition scripts ───────────────────────────────────────────────────
print("Loading transition modules... "); flush(stdout)

include(joinpath(TRANSITION_DIR, "transition_params.jl"))
include(joinpath(TRANSITION_DIR, "transition_solver.jl"))

println("done."); flush(stdout)

@printf("Threads available: %d\n\n", Threads.nthreads())
flush(stdout)

# ============================================================
# 1. Parse scenario from command line
# ============================================================
SCENARIO = length(ARGS) >= 1 ? lowercase(ARGS[1]) : "fc"

if SCENARIO == "fc"
    WINDOW_FROM = :base_fc
    WINDOW_TO   = :crisis_fc
    scenario_label = "Financial Crisis"
elseif SCENARIO == "covid"
    WINDOW_FROM = :base_covid
    WINDOW_TO   = :crisis_covid
    scenario_label = "COVID-19"
else
    error("Unknown scenario: '$SCENARIO'. Use 'fc' or 'covid'.")
end

@printf("Scenario:      %s\n", scenario_label)
@printf("Baseline:      %s\n", WINDOW_FROM)
@printf("Crisis:        %s\n", WINDOW_TO)
println()
flush(stdout)

# ============================================================
# 2. Check that both SMM result files exist
# ============================================================
function check_smm_file(window::Symbol)
    path = joinpath(SMM_OUTPUT_DIR, "smm_result_$(window).jls")
    return isfile(path), path
end

exists_from, path_from = check_smm_file(WINDOW_FROM)
exists_to,   path_to   = check_smm_file(WINDOW_TO)

if !exists_from || !exists_to
    println("\n" * "!"^60)
    println("  ERROR: Missing SMM result files")
    println("!"^60)
    !exists_from && @printf("  MISSING: %s\n", path_from)
    !exists_to   && @printf("  MISSING: %s\n", path_to)
    println()
    println("  The transition solver requires BOTH baseline and crisis")
    println("  SMM results.  Run the SMM estimator first:")
    @printf("    julia --threads auto code/smm/main.jl %s\n", WINDOW_FROM)
    @printf("    julia --threads auto code/smm/main.jl %s\n", WINDOW_TO)
    println()
    error("Cannot proceed without both SMM results.")
end

@printf("  ✓ Found %s\n", path_from)
@printf("  ✓ Found %s\n", path_to)
println()
flush(stdout)

# ============================================================
# 3. Load saved SMM results
# ============================================================
function load_smm_result(window::Symbol)
    inpath = joinpath(SMM_OUTPUT_DIR, "smm_result_$(window).jls")
    data = deserialize(inpath)
    @printf("Loaded SMM result: %s  (Q = %.6e)\n", inpath, data.result.loss_opt)
    flush(stdout)
    return data.result, data.spec, data.sim
end

println("Loading SMM results...")
flush(stdout)

res_from, spec_from, sim_from = load_smm_result(WINDOW_FROM)
res_to,   spec_to,   sim_to   = load_smm_result(WINDOW_TO)

# ============================================================
# 4. Reconstruct solved models from estimated parameters
# ============================================================
function solve_model_from_result(res::SMMResult, spec::SMMSpec, sim::SimParams;
                                  label::String="")
    @printf("\nSolving model%s... ", isempty(label) ? "" : " ($label)")
    flush(stdout)

    cp, rp, up, sp = unpack_θ(res.theta_opt, spec)
    model, solve_result = solve_model(cp, rp, up, sp, sim;
                                      Nx   = spec.run.Nx,
                                      Np_U = spec.run.Np_U,
                                      Np_S = spec.run.Np_S)
    if !solve_result.ok
        error("Model did not converge for $label estimated parameters!")
    end
    println("converged ✓")
    flush(stdout)
    return model
end

model_z0 = solve_model_from_result(res_from, spec_from, sim_from;
                                    label=string(WINDOW_FROM))
model_z1 = solve_model_from_result(res_to, spec_to, sim_to;
                                    label=string(WINDOW_TO))

# ── Compute equilibrium objects for both steady states ────────
println("\nComputing equilibrium objects...")
flush(stdout)

obj_z0 = compute_equilibrium_objects(model_z0)
obj_z1 = compute_equilibrium_objects(model_z1)

print_accounting(obj_z0)
print_accounting(obj_z1)

# ============================================================
# 5. Run transition dynamics
# ============================================================
println("\n" * "="^60)
@printf("  Running transition: %s → %s\n", WINDOW_FROM, WINDOW_TO)
println("="^60)
flush(stdout)

tp = TransitionParams(
    20.0,           # T_max: 20 years horizon
    200;            # N_steps: 200 time steps
    tol     = 1e-5,
    maxit   = 100,
    verbose = true,
    damp    = 0.5,
)

path = solve_transition(model_z0, model_z1, tp)

# ============================================================
# 6. Save transition output
# ============================================================
mkpath(TRANS_OUTPUT_DIR)
trans_file = joinpath(TRANS_OUTPUT_DIR,
                      "transition_$(WINDOW_FROM)_to_$(WINDOW_TO).jls")
serialize(trans_file, (path=path, tp=tp,
                       window_from=WINDOW_FROM, window_to=WINDOW_TO,
                       scenario=SCENARIO))
@printf("Transition path saved: %s\n", trans_file)
flush(stdout)

# ── Summary statistics ────────────────────────────────────────
@printf("\n╔══════════════════════════════════════════════════════╗\n")
@printf("║  Transition Summary: %s → %s\n", WINDOW_FROM, WINDOW_TO)
@printf("╠══════════════════════════════════════════════════════╣\n")
@printf("  θ_U:  %.4f → %.4f  (z₀ → z₁ SS)\n", path.theta_U[1], path.theta_U[end])
@printf("  θ_S:  %.4f → %.4f  (z₀ → z₁ SS)\n", path.theta_S[1], path.theta_S[end])
@printf("  Time horizon: %.1f years, %d steps\n", tp.T_max, tp.N_steps)
@printf("╚══════════════════════════════════════════════════════╝\n\n")

println("Done.")
flush(stdout)

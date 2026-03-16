############################################################
# main.jl — Solver entry point
#
# Usage (from project root):
#   julia --threads auto code/solver/main.jl
#
# Project layout assumed:
#   code/
#     solver/
#       main.jl        ← this file
#       grids.jl
#       params.jl
#       unskilled.jl
#       skilled.jl
#       solver.jl
#       equilibrium.jl
#       plots.jl
#   output/
#     plots/
#     tables/
############################################################

# ── Print immediately so we know the script is alive ─────────────────────────
println("="^60)
println("  Segmented Search Model — Stationary Equilibrium Solver")
println("="^60)
flush(stdout)

# ── Paths (set before any using, so they're available everywhere) ─────────────
const SOLVER_DIR   = @__DIR__
const PROJECT_ROOT = joinpath(SOLVER_DIR, "..", "..")
const OUTPUT_DIR   = joinpath(PROJECT_ROOT, "output")
const PLOTS_DIR    = joinpath(OUTPUT_DIR, "plots")

# ── Load packages (may be slow on first run due to precompilation) ────────────
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
using Plots
using LaTeXStrings
using Base.Threads

println("done."); flush(stdout)

Random.seed!(1234)

# ── Load solver modules in dependency order ────────────────────────────────────
print("Loading solver modules... "); flush(stdout)

include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))
include(joinpath(SOLVER_DIR, "plots.jl"))

println("done."); flush(stdout)

@printf("Threads available: %d\n\n", Threads.nthreads())
flush(stdout)

# ── Build and solve ────────────────────────────────────────────────────────────
println("Initialising model..."); flush(stdout)
model = initialise_model()

@printf("\nSolving stationary equilibrium...\n")
@time result = solve_model!(model)

if result.ok
    @printf("Solver converged  (U=%s  S=%s  global=%s)\n",
            result.converged_U, result.converged_S, result.converged_global)
else
    @printf("WARNING: solver did not fully converge  (U=%s  S=%s  global=%s)\n",
            result.converged_U, result.converged_S, result.converged_global)
end

# ── Compute equilibrium objects ────────────────────────────────────────────────
println("\nComputing equilibrium objects..."); flush(stdout)
obj = compute_equilibrium_objects(model)

print_accounting(obj)
flush(stdout)

# ── Generate and save all plots ────────────────────────────────────────────────
println("\nLoading plot packages..."); flush(stdout)

println("Generating and saving plots..."); flush(stdout)
mkpath(PLOTS_DIR)
make_all_plots(obj; output_dir = PLOTS_DIR)

println("\nDone. All output written to: $OUTPUT_DIR")
flush(stdout)

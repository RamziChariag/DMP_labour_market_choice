############################################################
# code/transition/main.jl — Transition dynamics entry point
#
# Usage (from project root):
#   julia --threads auto code/transition/main.jl
#
# User-configurable options (edit below):
#   SCENARIO       :fc   or  :covid
#   W_COND_TARGET  weight-matrix mode used in SMM estimation
#                  (determines which .jls files to load)
#
# Workflow:
#   1. Load the two SMM result bundles (baseline + crisis)
#   2. Reconstruct parameter structs and solve both steady states
#   3. Run the backward-forward transition algorithm
#   4. Save a rich TransitionResult to disk for later analysis
#
# Project layout:
#   code/
#     solver/      ← loaded as a library
#     smm/         ← smm_params.jl, smm.jl (for _load_smm_bundle, unpack_θ)
#     transition/
#       main.jl          ← this file
#       transition_params.jl
#       transition_solver.jl
#   output/
#     smm/               ← .jls estimation results
#     transition/        ← saved here
############################################################

# ═══════════════════════════════════════════════════════════
# 1. User configuration
# ═══════════════════════════════════════════════════════════

# Choose scenario: :fc  (Financial Crisis)  or  :covid  (Covid)
const SCENARIO = :fc

# Weight-matrix mode used when the SMM results were estimated.
# Must match the W_COND_TARGET that was set in code/smm/main.jl.
#   0.0  →  "_diagonalW"
#   1.0  →  "_compressedW"
#   2.0  →  "_equalW"
#   >2.0 →  "_fullW"       (e.g. 1e8 → "_cW1e8" style — see below)
const W_COND_TARGET = 0.0

# ═══════════════════════════════════════════════════════════
# 2. Packages
# ═══════════════════════════════════════════════════════════
print("Loading packages... "); flush(stdout)

using LinearAlgebra
using SparseArrays
using Statistics
using Distributions
using FastGaussQuadrature
using Interpolations
using Parameters
using Printf
using Random
using Base.Threads
using Serialization
using JLD2
using CSV
using DataFrames

println("done."); flush(stdout)

println("="^65)
println("  Segmented Search Model — Transition Dynamics")
println("="^65)
@printf("  Scenario:      %s\n", SCENARIO)
@printf("  W_COND_TARGET: %.1f\n\n", W_COND_TARGET)
flush(stdout)

# ═══════════════════════════════════════════════════════════
# 3. Load solver modules
# ═══════════════════════════════════════════════════════════
const TRANSITION_DIR = @__DIR__
const SOLVER_DIR     = joinpath(TRANSITION_DIR, "..", "solver")
const SMM_DIR        = joinpath(TRANSITION_DIR, "..", "smm")
const PROJECT_ROOT   = joinpath(TRANSITION_DIR, "..", "..")
const OUTPUT_DIR     = joinpath(PROJECT_ROOT, "output")
const SMM_OUT_DIR    = joinpath(OUTPUT_DIR, "smm")
const TRANS_OUT_DIR  = joinpath(OUTPUT_DIR, "transition")

print("Loading solver modules... "); flush(stdout)
include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))
println("done."); flush(stdout)

# ═══════════════════════════════════════════════════════════
# 4. Load SMM modules (for _load_smm_bundle, unpack_θ)
# ═══════════════════════════════════════════════════════════
print("Loading SMM modules... "); flush(stdout)
include(joinpath(SMM_DIR, "moments.jl"))
include(joinpath(SMM_DIR, "smm_params.jl"))
include(joinpath(SMM_DIR, "smm.jl"))
println("done."); flush(stdout)

# ═══════════════════════════════════════════════════════════
# 5. Load transition modules
# ═══════════════════════════════════════════════════════════
print("Loading transition modules... "); flush(stdout)
include(joinpath(TRANSITION_DIR, "transition_params.jl"))
include(joinpath(TRANSITION_DIR, "transition_solver.jl"))
println("done."); flush(stdout)

@printf("Threads available: %d\n\n", Threads.nthreads())
flush(stdout)

# ═══════════════════════════════════════════════════════════
# 6. Determine windows & file paths
# ═══════════════════════════════════════════════════════════

function _w_suffix(cond_target::Float64)
    cond_target == 0.0 && return "_diagonalW"
    cond_target == 1.0 && return "_compressedW"
    cond_target == 2.0 && return "_equalW"
    return "_fullW"
end

const W_SUFFIX = _w_suffix(W_COND_TARGET)

if SCENARIO == :fc
    base_window   = :base_fc
    crisis_window = :crisis_fc
elseif SCENARIO == :covid
    base_window   = :base_covid
    crisis_window = :crisis_covid
else
    error("Unknown SCENARIO: $SCENARIO. Must be :fc or :covid.")
end

base_jls   = joinpath(SMM_OUT_DIR, "smm_result_$(base_window)$(W_SUFFIX).jls")
crisis_jls = joinpath(SMM_OUT_DIR, "smm_result_$(crisis_window)$(W_SUFFIX).jls")

@printf("  Baseline file:  %s\n", base_jls)
@printf("  Crisis file:    %s\n\n", crisis_jls)
flush(stdout)

# ═══════════════════════════════════════════════════════════
# 7. Load SMM bundles and reconstruct parameters
# ═══════════════════════════════════════════════════════════

function _load_bundle_or_error(path, label)
    isfile(path) || error("$label not found: $path\n" *
        "Run the SMM estimation first (code/smm/main.jl).")
    bundle = _load_smm_bundle(path; delete_on_fail=false, label=label)
    isnothing(bundle) && error("$label is unreadable or stale: $path")
    return bundle
end

println("Loading SMM results...")
flush(stdout)

base_bundle   = _load_bundle_or_error(base_jls,   "Baseline SMM result")
crisis_bundle = _load_bundle_or_error(crisis_jls, "Crisis SMM result")

# Reconstruct the four parameter structs from each optimum
cp_base, rp_base, up_base, sp_base = unpack_θ(
    base_bundle.result.theta_opt, base_bundle.spec)
cp_crisis, rp_crisis, up_crisis, sp_crisis = unpack_θ(
    crisis_bundle.result.theta_opt, crisis_bundle.spec)

# SimParams for solving (use the one from the SMM run, but raise verbosity)
sim = base_bundle.sim
sim_solve = SimParams(
    tol_inner    = sim.tol_inner,
    tol_outer_U  = sim.tol_outer_U,
    tol_outer_S  = sim.tol_outer_S,
    tol_global   = sim.tol_global,
    maxit_inner  = sim.maxit_inner,
    maxit_outer  = sim.maxit_outer,
    maxit_global = sim.maxit_global,
    conv_streak  = sim.conv_streak,
    use_anderson = sim.use_anderson,
    anderson_m   = sim.anderson_m,
    anderson_reg = sim.anderson_reg,
    damp_pstar_U = sim.damp_pstar_U,
    damp_pstar_S = sim.damp_pstar_S,
    verbose      = 1,
    verbose_stride = sim.verbose_stride,
)

# Grid sizes (use whatever the SMM estimation used)
const Nx   = base_bundle.spec.run.Nx
const Np_U = base_bundle.spec.run.Np_U
const Np_S = base_bundle.spec.run.Np_S
@printf("  Grid sizes: Nx=%d  Np_U=%d  Np_S=%d\n", Nx, Np_U, Np_S)

@printf("  Baseline Q  = %.6e  (converged = %s)\n",
        base_bundle.result.loss_opt, base_bundle.result.converged)
@printf("  Crisis Q    = %.6e  (converged = %s)\n\n",
        crisis_bundle.result.loss_opt, crisis_bundle.result.converged)
flush(stdout)

# ═══════════════════════════════════════════════════════════
# 8. Solve both stationary equilibria
# ═══════════════════════════════════════════════════════════

println("Solving baseline (z₀) stationary equilibrium...")
flush(stdout)
model_z0, sr_z0 = solve_model(cp_base, rp_base, up_base, sp_base, sim_solve;
                               Nx = Nx, Np_U = Np_U, Np_S = Np_S)
sr_z0.ok || @warn "Baseline model did not converge — transition results may be unreliable."
eq_z0 = compute_equilibrium_objects(model_z0)
@printf("  θ_U = %.4f   θ_S = %.4f   ur_total = %.4f\n\n",
        eq_z0.thetaU, eq_z0.thetaS, eq_z0.ur_total)
flush(stdout)

println("Solving crisis (z₁) stationary equilibrium...")
flush(stdout)
model_z1, sr_z1 = solve_model(cp_crisis, rp_crisis, up_crisis, sp_crisis, sim_solve;
                               Nx = Nx, Np_U = Np_U, Np_S = Np_S)
sr_z1.ok || @warn "Crisis model did not converge — transition results may be unreliable."
eq_z1 = compute_equilibrium_objects(model_z1)
@printf("  θ_U = %.4f   θ_S = %.4f   ur_total = %.4f\n\n",
        eq_z1.thetaU, eq_z1.thetaS, eq_z1.ur_total)
flush(stdout)

# ═══════════════════════════════════════════════════════════
# 9. Transition parameters
# ═══════════════════════════════════════════════════════════

tp = TransitionParams(
    T_max   = 120.0,   # 10 years in months
    N_steps = 240,     # half-month steps
    tol     = 1e-4,
    maxit   = 200,
    damp    = 0.3,
    verbose = true,
)

# ═══════════════════════════════════════════════════════════
# 10. Solve transition
# ═══════════════════════════════════════════════════════════

result = solve_transition(model_z0, model_z1, tp; scenario = SCENARIO)

# ═══════════════════════════════════════════════════════════
# 11. Summary
# ═══════════════════════════════════════════════════════════

println("="^65)
println("  Transition Results Summary")
println("="^65)

Nt = tp.N_steps + 1
@printf("  Converged: %s  (%d iterations, final dist = %.3e)\n",
        result.converged, result.n_iterations, result.final_dist)
@printf("\n  Tightness:\n")
@printf("    θ_U:  %.4f → %.4f\n", result.θU[1], result.θU[end])
@printf("    θ_S:  %.4f → %.4f\n", result.θS[1], result.θS[end])
@printf("\n  Unemployment rates:\n")
@printf("    ur_U:     %.4f → %.4f\n", result.ur_U[1], result.ur_U[end])
@printf("    ur_S:     %.4f → %.4f\n", result.ur_S[1], result.ur_S[end])
@printf("    ur_total: %.4f → %.4f\n", result.ur_total[1], result.ur_total[end])
@printf("\n  Job-finding rates:\n")
@printf("    f_U:  %.4f → %.4f\n", result.fU[1], result.fU[end])
@printf("    f_S:  %.4f → %.4f\n", result.fS[1], result.fS[end])
@printf("\n  Composition:\n")
@printf("    skilled_share:  %.4f → %.4f\n",
        result.skilled_share[1], result.skilled_share[end])
@printf("    training_share: %.4f → %.4f\n",
        result.training_share[1], result.training_share[end])
flush(stdout)

# ═══════════════════════════════════════════════════════════
# 12. Save results
# ═══════════════════════════════════════════════════════════

mkpath(TRANS_OUT_DIR)

out_file = joinpath(TRANS_OUT_DIR, "transition_$(SCENARIO)$(W_SUFFIX).jld2")
@printf("\nSaving transition result → %s\n", out_file)
flush(stdout)

jldsave(out_file;
    result   = result,
    eq_z0    = eq_z0,
    eq_z1    = eq_z1,
    tp       = tp,
    scenario = SCENARIO,
)

@printf("\nDone.\n")
println("="^65)
flush(stdout)

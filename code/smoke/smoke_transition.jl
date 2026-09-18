#!/usr/bin/env julia
############################################################
# smoke_transition.jl — the transition path still runs on the shipped estimate.
#
#   RSROOT=<tree> julia --project=. -t 6 code/smoke/smoke_transition.jl
#
# The transition solver reconstructs the skilled employed density itself rather than
# calling the stationary routine, so it carries its own copy of every margin the
# stationary solve has — and a change to a margin in `solver/` can leave the path
# consuming the old one. This is the gate that catches that: both steady states solve,
# `solve_transition` completes, and every reported series is finite.
#
# The terminal regime is a −3% shift in the aggregate productivity A rather than the
# crisis bundle, so the test needs one estimate on disk and exercises the same code
# path a scenario run would. The outer loop is capped short: this establishes that the
# path runs and stays finite, NOT that it converged.
#
# The grid is the ESTIMATION grid, not a coarser one. base_fc's skilled block does
# not converge at Nx = 60, so the gate at the old default failed on the steady state
# and never reached the path it exists to test. Two solves plus six outer iterations
# at Nx = 120 cost about a minute.
#
# Optional: SMOKE_NX (default 120), SMOKE_MAXIT (6), ROYSEARCH_WINDOW (base_fc).
############################################################

using LinearAlgebra, Statistics, Printf, Serialization
using Distributions, FastGaussQuadrature, Interpolations, Parameters, Base.Threads
using Optim, CSV, DataFrames, Clustering, QuasiMonteCarlo, JSON3
BLAS.set_num_threads(1)

const ROOT = get(ENV, "RSROOT", normpath(joinpath(@__DIR__, "..", "..")))
@isdefined(ROYSEARCH_PATHS_LOADED) || include(joinpath(ROOT, "code", "paths.jl"))
for f in ["version.jl", "settings.jl"];  include(joinpath(ROOT, "code", "smm", f))  end
for f in ["grids.jl", "params.jl", "unskilled.jl", "skilled.jl", "solver.jl",
          "equilibrium.jl"];             include(joinpath(ROOT, "code", "solver", f)) end
for f in ["moments.jl", "smm_params.jl", "bundle.jl", "smm.jl"]
    include(joinpath(ROOT, "code", "smm", f))
end
for f in ["transition_params.jl", "transition_values.jl", "transition_solver.jl"]
    include(joinpath(ROOT, "code", "transition", f))
end

const RULE   = "="^70
const WINDOW = get(ENV, "ROYSEARCH_WINDOW", "base_fc")
const NX     = parse(Int, get(ENV, "SMOKE_NX", "120"))
const MAXIT  = parse(Int, get(ENV, "SMOKE_MAXIT", "6"))

println("\n$RULE\nSMOKE — transition path on the $WINDOW estimate (v$ROYSEARCH_VERSION)\n$RULE")

bundle = deserialize(estimate_path(WINDOW, "_diagonalW"))
cp, up, sp = unpack_θ(bundle.result.theta_opt, bundle.spec)
# @with_kw gives no copy-with-changes constructor, so the fields are splatted by name
# and the override follows — the same idiom model_main.jl uses.
cp1 = CommonParams(; (f => getfield(cp, f) for f in fieldnames(CommonParams))...,
                   A = 0.97 * cp.A)
sim = bundle.spec.sim

@printf("grid Nx = %d   outer cap = %d   A: %.6f → %.6f\n", NX, MAXIT, cp.A, cp1.A)

model_z0, sr0 = solve_model(cp,  up, sp, sim; Nx = NX, Np_U = NX, Np_S = NX)
model_z1, sr1 = solve_model(cp1, up, sp, sim; Nx = NX, Np_U = NX, Np_S = NX)
@printf("steady states: z0 ok = %s   z1 ok = %s\n", sr0.ok, sr1.ok)
(sr0.ok && sr1.ok) || error("SMOKE FAILED: a terminal steady state did not solve.")

tp  = TransitionParams(T_max = 120.0, N_steps = 240, tol = 1e-4, maxit = MAXIT,
                       verbose = false)
res = solve_transition(model_z0, model_z1, tp; scenario = :fc)

series = (:θU, :θS, :fU, :fS, :ur_U, :ur_S, :ur_total, :skilled_share,
          :training_share, :mean_wage_U, :mean_wage_S)
nonfinite = [s for s in series if !all(isfinite, getfield(res, s))]
isempty(nonfinite) ||
    error("SMOKE FAILED: non-finite series — " * join(nonfinite, ", "))

@printf("converged = %s  (%d iterations, final dist = %.3e vs tol = %.1e)\n",
        res.converged, res.n_iter, res.final_dist, tp.tol)
for s in series
    v = getfield(res, s)
    @printf("  %-15s %.6f → %.6f   [%.6f, %.6f]\n",
            s, first(v), last(v), minimum(v), maximum(v))
end

println("\n$RULE\nSMOKE PASSED — path runs, every series finite\n$RULE")

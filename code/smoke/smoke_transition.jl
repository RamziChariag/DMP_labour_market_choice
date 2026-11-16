#!/usr/bin/env julia
############################################################
# smoke/smoke_transition.jl — the transition path runs on a stored estimate.
#
#   RSROOT=<tree> julia --project=. -t 6 code/smoke/smoke_transition.jl
#
# The transition solver rebuilds the skilled employed density itself, so it carries its
# own copy of every stationary margin; this catches a margin changed in solver/ but not
# in the path. It solves the steady state at the window's SMM estimate and one with the
# stored log productivity scaled by 0.97 (cp.A is ln A), runs `solve_transition` between
# them with the outer loop capped, and requires every reported series to be finite. It
# does not test convergence. The grid is the estimation grid: base_fc's skilled block
# does not converge at Nx = 60. About a minute.
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

bundle = period_bundle(read_bundle(estimate_path(estimation_window(WINDOW), "_diagonalW"; point = :smm)),
                       Symbol(WINDOW))
cp, up, sp, tight = unpack_θ(bundle.result.theta_opt, bundle.spec)
# @with_kw has no copy-with-changes constructor: splat the fields, then override A.
cp1 = CommonParams(; (f => getfield(cp, f) for f in fieldnames(CommonParams))...,
                   A = 0.97 * cp.A)
sim = bundle.spec.sim

@printf("grid Nx = %d   outer cap = %d   A: %.6f → %.6f\n", NX, MAXIT, cp.A, cp1.A)

# z0 is the estimate's equilibrium, at its tightnesses; z1 holds z0's vacancy costs and
# moves A, so its tightnesses come from free entry.
model_z0, sr0 = solve_model(cp, up, sp, sim; Nx = NX, Np_U = NX, Np_S = NX, tightness = tight)
model_z1, sr1 = solve_model(cp1, model_z0.unsk_par, model_z0.skl_par, sim; Nx = NX, Np_U = NX, Np_S = NX)
@printf("steady states: z0 ok = %s   z1 ok = %s\n", sr0.ok, sr1.ok)
(sr0.ok && sr1.ok) || error("SMOKE FAILED: a terminal steady state did not solve.")

tp  = TransitionParams(T_max = 120.0, N_steps = 240, tol = 1e-4, maxit = MAXIT,
                       verbose = false)
res = solve_transition(model_z0, model_z1, tp; scenario = :fc)

series = (:θU, :θS, :κU, :κS, :ur_U, :ur_S, :ur_total, :skilled_share,
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

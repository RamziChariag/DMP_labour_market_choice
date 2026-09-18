#!/usr/bin/env julia
############################################################
# transition_audit.jl — does the transition solve a TRANSITORY path?
#
#   RSROOT=<tree> julia --project=. -t 6 code/tools/transition_audit.jl
#
# A backward–forward solver can be wrong in a way no finiteness check sees: it
# can return a sequence of steady states. Three properties separate the two,
# and this script measures all three on the base→crisis pair.
#
#   (i)   The mass-weighted cross-market flow ∬ d f_U u_S is identically zero
#         at BOTH stationary endpoints by self-selection (notes,
#         prop:ss-crossflow). If it is also zero along the whole path, either
#         the path is not transitory or the channel is directional and cannot
#         fire on this pair — the frontier-floor series tells them apart.
#   (ii)  The interior distributions must differ from BOTH endpoints'
#         stationary distributions. A sequence of steady states sits on the
#         segment between them; a transition overshoots or lags.
#   (iii) The terminal date must reproduce the post-switch stationary
#         equilibrium. This is the only check that the forward laws of motion
#         and the stationary KFE agree, so a transcription error in either one
#         shows up here as a terminal gap.
#
# Writes output/logs/transition_audit_<pair>.csv: the per-date series the
# report reads. Reads the estimate bundles; writes nothing under
# output/estimates/ or output/chains/.
#
# TA_NULL=true runs the pair against ITSELF — the same estimate at both ends,
# so the true path is the constant one. Any drift then measures the gap between
# the forward laws of motion and the stationary KFE, with no economics mixed in.
# It is the cheapest way to tell a transition that is wrong from one that is
# merely slow, and it should be the first thing run after touching either.
#
# Environment: TA_PAIR (fc|covid), TA_NX (120 — the estimation grid),
# TA_NSTEPS (240), TA_MAXIT (150), TA_TMAX (120.0), TA_DAMP (the TransitionParams default),
# TA_NULL (false).
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

const RULE    = "="^72
const PAIR    = Symbol(get(ENV, "TA_PAIR", "fc"))
const NX      = parse(Int,     get(ENV, "TA_NX",     "120"))
const NSTEPS  = parse(Int,     get(ENV, "TA_NSTEPS", "240"))
const MAXIT   = parse(Int,     get(ENV, "TA_MAXIT",  "150"))
const TMAX    = parse(Float64, get(ENV, "TA_TMAX",   "120.0"))
const DAMP    = haskey(ENV, "TA_DAMP") ? parse(Float64, ENV["TA_DAMP"]) : TransitionParams().damp
const NULLRUN = lowercase(get(ENV, "TA_NULL", "false")) == "true"
const WSUFFIX = "_diagonalW"

const _PAIR = PAIR === :fc    ? (:base_fc,    :crisis_fc)    :
              PAIR === :covid ? (:base_covid, :crisis_covid) :
              error("TA_PAIR must be fc or covid, got $PAIR")
const PAIR_WINDOWS = NULLRUN ? (_PAIR[1], _PAIR[1]) : _PAIR

println("\n$RULE\nTRANSITION AUDIT — $(PAIR_WINDOWS[1]) → $(PAIR_WINDOWS[2])  (v$ROYSEARCH_VERSION)\n$RULE")
@printf("Nx=%d  Nt=%d  T_max=%.0f  maxit=%d  damp=%.2f  threads=%d\n",
        NX, NSTEPS + 1, TMAX, MAXIT, DAMP, Threads.nthreads())

# ── Stationary endpoints ────────────────────────────────────────────────────
b0 = deserialize(estimate_path(PAIR_WINDOWS[1], WSUFFIX))
b1 = deserialize(estimate_path(PAIR_WINDOWS[2], WSUFFIX))
cp0, up0, sp0 = unpack_θ(b0.result.theta_opt, b0.spec)
cp1, up1, sp1 = unpack_θ(b1.result.theta_opt, b1.spec)
sim = SimParams(; (f => getfield(b0.spec.sim, f) for f in fieldnames(SimParams))...,
                verbose = 0)

t_start = time()
model_z0, sr0 = solve_model(cp0, up0, sp0, sim; Nx = NX, Np_U = NX, Np_S = NX)
model_z1, sr1 = solve_model(cp1, up1, sp1, sim; Nx = NX, Np_U = NX, Np_S = NX)
@printf("z0 ok=%s (U=%s S=%s G=%s)   z1 ok=%s (U=%s S=%s G=%s)   [%.0f s]\n",
        sr0.ok, sr0.converged_U, sr0.converged_S, sr0.converged_global,
        sr1.ok, sr1.converged_U, sr1.converged_S, sr1.converged_global, time() - t_start)

eq0 = compute_equilibrium_objects(model_z0)
eq1 = compute_equilibrium_objects(model_z1)
@printf("z0: θ_U=%.4f θ_S=%.4f ur_U=%.4f ur_S=%.4f skill=%.4f train=%.4f\n",
        eq0.thetaU, eq0.thetaS, eq0.ur_U, eq0.ur_S,
        eq0.agg_mS / (eq0.agg_mS + eq0.agg_mU), eq0.agg_t / eq0.total_pop)
@printf("z1: θ_U=%.4f θ_S=%.4f ur_U=%.4f ur_S=%.4f skill=%.4f train=%.4f\n",
        eq1.thetaU, eq1.thetaS, eq1.ur_U, eq1.ur_S,
        eq1.agg_mS / (eq1.agg_mS + eq1.agg_mU), eq1.agg_t / eq1.total_pop)

# ── Transition ──────────────────────────────────────────────────────────────
tp = TransitionParams(T_max = TMAX, N_steps = NSTEPS, tol = 1e-4,
                      maxit = MAXIT, damp = DAMP, verbose = true)
t_tr = time()
res  = solve_transition(model_z0, model_z1, tp; scenario = PAIR)
@printf("\nconverged=%s  iters=%d  final ‖Δθ‖∞=%.3e (tol=%.1e)  [%.0f s]\n",
        res.converged, res.n_iter, res.final_dist, tp.tol, time() - t_tr)

# ── Endpoint marginals, for the distribution tests ──────────────────────────
# The path reports ability-MARGINAL profiles: unskilled masses over aU (rows),
# skilled masses over aS (columns).  The stationary matrices are collapsed the
# same way so the two are comparable node by node.
row_marg(M) = vec(sum(M, dims = 2))
col_marg(M) = vec(sum(M, dims = 1))

# Stationary counterpart of TransitionResult.frontier_floor: the aS at which the
# least unskilled-able worker is indifferent to training, read off the same net
# training gain the solver builds.  Compared against the path's date-1 value it
# says how far the training frontier jumps at the switch date.
function stationary_floor(eq)
    us = eq.Usearch[1];  Utr = eq.net_T;  x = eq.xg;  Nx = length(x)
    Utr[Nx] < us  && return x[Nx]
    Utr[1]  >= us && return x[1]
    j0 = 2
    while j0 < Nx && Utr[j0] < us
        j0 += 1
    end
    dU = Utr[j0] - Utr[j0 - 1]
    return dU > 0.0 ? x[j0 - 1] + (us - Utr[j0 - 1]) * (x[j0] - x[j0 - 1]) / dU : x[j0]
end

prof0 = (uU = row_marg(eq0.uU), tU = row_marg(eq0.tU),
         uS = col_marg(eq0.uS_mat), mS = col_marg(eq0.mS_mat))
prof1 = (uU = row_marg(eq1.uU), tU = row_marg(eq1.tU),
         uS = col_marg(eq1.uS_mat), mS = col_marg(eq1.mS_mat))

# Sup-norm distance of the date-n profile to a stationary reference, summed
# over the four masses and scaled by the reference's own total mass so the
# four blocks are commensurable.
function prof_dist(res::TransitionResult, ref, n::Int)
    d = 0.0
    for f in (:uU, :tU, :uS, :mS)
        r = getfield(ref, f)
        d += maximum(abs, getfield(res, f)[:, n] .- r) / max(sum(r), 1e-14)
    end
    return d
end

Nt   = length(res.tgrid)
d_z0 = [prof_dist(res, prof0, n) for n in 1:Nt]
d_z1 = [prof_dist(res, prof1, n) for n in 1:Nt]
seg  = d_z0[1] + d_z1[1]                      # endpoint-to-endpoint separation

println("\n$RULE\nTESTS\n$RULE")

# (i) cross-market flow — and, when it is zero, WHY.  Location, population and
#     flow are three different statements and a zero in one implies nothing
#     about the others.
dmax, darg = findmax(res.d_flow)
@printf("(i)   max d-flow  f_U ∬d·u_S = %.6e at t = %.1f months (endpoints %.3e, %.3e)\n",
        dmax, res.tgrid[darg], res.d_flow[1], res.d_flow[end])
@printf("      F_d active-side population ∬d·ℓ:   max = %.6e\n", maximum(res.d_region))
@printf("      F_d active-side trained mass ∬d·m_S: max = %.6e\n", maximum(res.d_mass))
@printf("      F_τ floor along the path: %.6f → %.6f   (min %.6f, max %.6f)\n",
        res.frontier_floor[1], res.frontier_floor[end],
        minimum(res.frontier_floor), maximum(res.frontier_floor))
@printf("      F_τ floor at the stationary endpoints: z0 %.6f   z1 %.6f\n",
        stationary_floor(eq0), stationary_floor(eq1))

# (ii) interior distributions off BOTH endpoints.  Distance to the nearer
#      endpoint says the path is not sitting ON one of them; the triangle
#      excess d(·,z0)+d(·,z1) − d(z0,z1) says it is not on the straight line
#      BETWEEN them either, which is what a sequence of steady states would be.
off_both = [min(d_z0[n], d_z1[n]) for n in 2:(Nt - 1)]
omax, oarg = findmax(off_both)
excess    = [d_z0[n] + d_z1[n] - seg for n in 2:(Nt - 1)]
emax, earg = findmax(excess)
@printf("(ii)  min-distance-to-nearer-endpoint: max over interior = %.6e at t = %.1f months\n",
        omax, res.tgrid[oarg + 1])
@printf("      triangle excess d(·,z0)+d(·,z1) − d(z0,z1): max = %.6e at t = %.1f months\n",
        emax, res.tgrid[earg + 1])
@printf("      endpoint separation d(z0,z1) = %.6e; excess = %.2f%% of it\n",
        seg, 100 * emax / max(seg, 1e-14))

# (iii) terminal date reproduces the reference stationary equilibrium — z1 on a
#       real pair, z0 on a null run, where the reference is also the whole path.
@printf("(iii) terminal distribution gap to %s = %.6e  (date-1 gap to z0 = %.6e)\n",
        NULLRUN ? "z0" : "z1", NULLRUN ? d_z0[end] : d_z1[end], d_z0[1])

# Per-quantity drift against that reference: the terminal deviation is the
# number the changelog table reports, and the running maximum says whether the
# path overshoots on its way there or simply never arrives.
#
# The reference RATES are rebuilt from the stationary masses by the arithmetic
# `_build_result` uses, not read off `compute_equilibrium_objects`.  The two
# layers do not define them identically — the path's labour force excludes
# workers in training, the moment layer's does not, which alone is a 4.4% gap
# in `skilled_share` — and a definitional gap read as drift is a wrong number.
function drift_rows(res::TransitionResult, eq, prof)
    tot(f)  = vec(sum(getfield(res, f), dims = 1))
    uU, t   = sum(prof.uU), sum(prof.tU)
    uS, mS  = sum(prof.uS), sum(prof.mS)
    eU      = max(eq.total_pop - mS, 0.0) - uU - t
    lf_U    = uU + eU
    return (("u_U total",      tot(:uU),           uU),
            ("t total",        tot(:tU),           t),
            ("u_S total",      tot(:uS),           uS),
            ("m_S total",      tot(:mS),           mS),
            ("θ_U",            res.θU,             eq.thetaU),
            ("θ_S",            res.θS,             eq.thetaS),
            ("ur_U",           res.ur_U,           uU / lf_U),
            ("ur_S",           res.ur_S,           uS / mS),
            ("ur_total",       res.ur_total,       (uU + uS) / (lf_U + mS)),
            ("skilled_share",  res.skilled_share,  mS / (lf_U + mS)),
            ("training_share", res.training_share, t / eq.total_pop))
end

@printf("\n      %-15s %12s %12s %10s %10s %8s\n",
        "quantity", "reference", "terminal", "term %", "max %", "at t")
for (lab, series, ref) in drift_rows(res, NULLRUN ? eq0 : eq1, NULLRUN ? prof0 : prof1)
    den  = max(abs(ref), 1e-14)
    rel  = (series .- ref) ./ den
    imax = argmax(abs.(rel))
    @printf("      %-15s %12.6f %12.6f %+10.2f %+10.2f %8.1f\n",
            lab, ref, series[end], 100 * rel[end], 100 * rel[imax], res.tgrid[imax])
end

# ── Per-date series ─────────────────────────────────────────────────────────
out = DataFrame(
    t                 = res.tgrid,
    frontier_floor    = res.frontier_floor,
    d_region          = res.d_region,
    d_mass            = res.d_mass,
    d_flow            = res.d_flow,
    theta_U           = res.θU,
    theta_S           = res.θS,
    f_U               = res.fU,
    f_S               = res.fS,
    skilled_share     = res.skilled_share,
    training_share    = res.training_share,
    ur_U              = res.ur_U,
    ur_S              = res.ur_S,
    dist_to_z0        = d_z0,
    dist_to_z1        = d_z1,
)
csv = joinpath(out_logs(), "transition_audit_$(PAIR)$(NULLRUN ? "_null" : "").csv")
CSV.write(csv, out)
@printf("\nWrote %s  (%d dates)\n", csv, nrow(out))
println("$RULE")

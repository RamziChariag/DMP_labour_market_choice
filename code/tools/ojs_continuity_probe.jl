#!/usr/bin/env julia
############################################################
# ojs_continuity_probe.jl — is the criterion a differentiable function of the OJS
# cutoff p^oj?
#
#   PROBE_MODE=moments  julia --project=. -t 6 code/tools/ojs_continuity_probe.jl
#   PROBE_MODE=jump     julia --project=. -t 6 code/tools/ojs_continuity_probe.jl
#   PROBE_MODE=jacobian PROBE_TOL=1e-8 julia --project=. -t 6 code/tools/ojs_continuity_probe.jl
#
# WHY THIS EXISTS. An SMM estimate, its standard errors and every gradient step
# require Q to be differentiable in θ. A criterion with a step is not: no finite
# difference and no J⁻¹ is defined across the step, whatever the step's size. The
# skilled block reads the on-the-job-search cutoff p^oj at two margins — how a cell's
# employed mass is assigned between the poached and non-poached wage surfaces, and
# which mass contributes to the E-to-E flow — and a hard 1{p < p^oj} at either moves a
# whole grid cell's mass the instant p^oj crosses a node. This probe measures whether
# it still does.
#
# THE THREE MODES, each answering one question:
#
#   moments   Q and the active moment vector at the incumbent, printed to full Float64
#             precision, twice: as solved, and with p^oj snapped to p-grid nodes. The
#             snapped case puts every OJS covered fraction at exactly 0 or 1, which is
#             where a soft weight and a hard indicator must agree EXACTLY — so it is
#             the backward-compatibility test. Run it at -t 1: threaded reductions are
#             non-associative, and a bit-identity claim needs a deterministic solve.
#
#   jump      The size of the discontinuity in Q across a p^oj node crossing, at four
#             solve tolerances. Along +λ_S the crossing sits between h = 3e-7 and 1e-6
#             and along −μ_S between h = −3e-7 and −1e-6 (both at aS node 63). A line
#             is fitted through the three same-side points and the jump is the far-side
#             point's excess over it. Invariance to the tolerance is what distinguishes
#             a genuine discretisation step from a solve-precision artefact: an
#             artefact of scale tol_global shrinks with tol_global, and a step does not.
#
#   jacobian  Per-column relative L2 spread of the whitened moment Jacobian over
#             h ∈ {3e-7, 1e-6, 3e-6}, plus cond(J) at the fixed h = 1e-6. A column
#             whose value depends on the step is not a derivative, and 5% is the
#             tolerance at which a fixed-step Jacobian is usable. Run at PROBE_TOL 1e-4
#             and 1e-8: a spread that improves with the tolerance is a noise floor, one
#             that does not is a step.
#
# Steps are taken in UNCONSTRAINED t, the space the optimiser and the MCMC Jacobian
# both work in, so a step here is the step they take.
#
# READ-ONLY: writes CSVs under output/logs, touches no bundle.
############################################################

using LinearAlgebra, Statistics, Printf, Serialization
using Distributions, FastGaussQuadrature, Interpolations, Parameters, Base.Threads
using Optim, CSV, DataFrames, Clustering, QuasiMonteCarlo, JSON3
BLAS.set_num_threads(1)

# RSROOT lets the probe run against a scratch copy of the tree, which is how a
# before/after comparison is made without two checkouts of the repo.
const ROOT = get(ENV, "RSROOT", normpath(joinpath(@__DIR__, "..", "..")))
@isdefined(ROYSEARCH_PATHS_LOADED) || include(joinpath(ROOT, "code", "paths.jl"))
for f in ["version.jl", "settings.jl"];  include(joinpath(ROOT, "code", "smm", f))  end
for f in ["grids.jl", "params.jl", "unskilled.jl", "skilled.jl", "solver.jl",
          "equilibrium.jl"];             include(joinpath(ROOT, "code", "solver", f)) end
for f in ["moments.jl", "smm_params.jl", "bundle.jl", "smm.jl"]
    include(joinpath(ROOT, "code", "smm", f))
end

const MODE     = Symbol(get(ENV, "PROBE_MODE", "moments"))
const WINDOW   = get(ENV, "ROYSEARCH_WINDOW", "base_fc")
const W_SUFFIX = "_diagonalW"
const TAG      = get(ENV, "PROBE_TAG", "")
const TOL      = parse(Float64, get(ENV, "PROBE_TOL", "1e-4"))

const BUNDLE = deserialize(estimate_path(WINDOW, W_SUFFIX))
const SPEC   = BUNDLE.spec
const Θ0     = collect(float.(BUNDLE.result.theta_opt))
const MK     = active_moment_keys(SPEC)
const FREE   = SPEC.free
const D      = length(FREE)
const PNAME  = [string(param_symbol(f)) for f in FREE]

# Parameters.jl's @with_kw gives no copy-with-changes constructor, so the fields are
# splatted by name and the overrides follow — the idiom model_main.jl uses. Later
# duplicate keywords win.
_sim_with(sim::SimParams; kw...) =
    SimParams(; (f => getfield(sim, f) for f in fieldnames(SimParams))..., kw...)
_spec_with_sim(spec::SMMSpec, sim::SimParams) =
    SMMSpec(spec.free, spec.fixed, spec.moments, sim, spec.run, spec.W, spec.q_scale)

const _FAILED = (ok = false, Q = Inf, m = Float64[], idx = Int[], Gpoj = Float64[],
                 n_straddle = 0, mass_straddle = NaN, w_lo = NaN, w_hi = NaN)

"""
    eval_point(θ, spec; snap_poj = false) -> NamedTuple

Solve at unconstrained `θ` and return, from that one solve: `Q`; the active moment
vector `m`; the p-grid node `idx[j]` each row's `p^oj` sits in and the tabulated
`Γ(p^oj)` read there; how many `(aS, p)` cells have a strictly interior OJS weight
and what employed mass they carry; and the wage grid's endpoints. Everything after
`m` is there to attribute a step in `Q` to a site rather than infer it.

`Q` goes through `compute_loss_matrix` — the objective's own weighted-loss path — so
the probe cannot drift from the criterion it is measuring. On a failed or
non-converged solve every field comes back empty, which is what marks a probe point
unusable rather than merely unlucky.

`snap_poj` replaces each `p^oj(aS)` by the p-grid node at or above it before the
moment layer reads it: the degenerate case where no cell straddles the cutoff and a
covered fraction must equal the hard indicator it replaced.
"""
function eval_point(θ::Vector{Float64}, spec::SMMSpec; snap_poj::Bool = false)
    cp, up, sp = unpack_θ(θ, spec)
    local model, sr
    try
        model, sr = solve_model(cp, up, sp, spec.sim;
                                Nx = spec.run.Nx, Np_U = spec.run.Np_U, Np_S = spec.run.Np_S)
    catch
        return _FAILED
    end
    sr.ok || return _FAILED
    if snap_poj
        pg = model.skl_grids.p
        @inbounds for j in eachindex(model.skl_cache.poj)
            model.skl_cache.poj[j] = pg[pcut_index(pg, clamp01(model.skl_cache.poj[j]))]
        end
    end
    obj = compute_equilibrium_objects(model)
    mm  = model_moments(obj)
    pg  = obj.pg
    idx = [pcut_index(pg, clamp01(obj.poj[j])) for j in eachindex(obj.poj)]

    n_str = 0;  m_str = 0.0
    for j in 1:obj.Nx
        poj_j = clamp01(obj.poj[j])
        for jp in 1:obj.NpS
            s = _soft_oj_weight(pg[jp], poj_j, pg, jp, obj.NpS)
            if s > 1e-14 && s < 1.0 - 1e-14
                n_str += 1;  m_str += obj.eS_pS[j, jp] * obj.wpS[jp]
            end
        end
    end
    return (ok = true,
            Q   = compute_loss_matrix(mm, spec, spec.W),
            m   = [getproperty(mm, k) for k in MK],
            idx = idx,
            Gpoj = model.skl_pre.Γvals[idx],
            n_straddle = n_str, mass_straddle = m_str,
            w_lo = first(obj.wmid), w_hi = last(obj.wmid))
end

_shift(θ, j, h) = (t = copy(θ); t[j] += h; t)
const _IX = Dict(PNAME[i] => i for i in 1:D)

# ── Whitening, as rank_diagnostic.jl defines it ────────────────────────────────
# Rows by the sampling sd the objective weights with (W = diag(1/σ̂²)), columns by
# dθ/dt so a column reads in natural parameter units.
const SIGMA = [1.0 / sqrt(SPEC.W[i, i]) for i in 1:length(MK)]
const DTHDT = [begin
                   u = (_to_constrained(Θ0[j], f.lb, f.ub) - f.lb) / (f.ub - f.lb)
                   (f.ub - f.lb) * u * (1 - u)
               end for (j, f) in enumerate(FREE)]

@printf("probe %s   tree %s   window %s   threads %d\n", MODE, ROOT, WINDOW, nthreads())
@printf("bundle Q = %.9f   free = %d   active moments = %d\n\n",
        BUNDLE.result.loss_opt, D, length(MK))


# ════════════════════════════════════════════════════════════
#  moments — backward compatibility and the moment vector itself
# ════════════════════════════════════════════════════════════
if MODE === :moments
    p_as   = eval_point(Θ0, SPEC)
    p_snap = eval_point(Θ0, SPEC; snap_poj = true)
    p_rep  = eval_point(Θ0, SPEC; snap_poj = true)
    @printf("determinism of the snapped solve in-process: %s\n",
            p_snap.m == p_rep.m ? "bit-identical" : "DIFFERS")
    @printf("Q_assolved = %.17g\nQ_snapped  = %.17g\n", p_as.Q, p_snap.Q)
    for (i, k) in enumerate(MK)
        @printf("M,%s,%.17g,%.17g\n", k, p_as.m[i], p_snap.m[i])
    end
    df = DataFrame(moment = String.(string.(MK)), as_solved = p_as.m, snapped = p_snap.m)
    CSV.write(joinpath(out_logs(), "ojs_moments_$(WINDOW)$(TAG).csv"), df)
end


# ════════════════════════════════════════════════════════════
#  jump — the discontinuity across a p^oj node crossing, by tolerance
# ════════════════════════════════════════════════════════════
if MODE === :jump
    # The three same-side steps and the far-side step that crosses the node, per ray.
    RAYS = [("λ_S", [3e-8, 1e-7, 3e-7], 1e-6),
            ("μ_S", [-3e-8, -1e-7, -3e-7], -1e-6)]
    RUNGS = [("shipped 1e-4", SPEC.sim),
             ("uniform 1e-4", _sim_with(SPEC.sim; tol_global = 1e-4, conv_streak = 2,
                                       global_B = 0, maxit_global = 600)),
             ("uniform 1e-6", _sim_with(SPEC.sim; tol_global = 1e-6, conv_streak = 2,
                                       global_B = 0, maxit_global = 600)),
             ("uniform 1e-8", _sim_with(SPEC.sim; tol_global = 1e-8, conv_streak = 2,
                                       global_B = 0, maxit_global = 600))]

    rows = DataFrame(rung = String[], param = String[], h_far = Float64[],
                     jump = Float64[], slope = Float64[], Q_base = Float64[],
                     Q_far = Float64[], nodes_moved = Int[], all_ok = Bool[])
    for (rname, sim) in RUNGS
        spec = _spec_with_sim(SPEC, sim)
        base = eval_point(Θ0, spec)
        for (pn, hs, h_far) in RAYS
            j  = _IX[pn]
            ps = [eval_point(_shift(Θ0, j, h), spec) for h in hs]
            pf = eval_point(_shift(Θ0, j, h_far), spec)
            # Least-squares line through the three same-side (h, Q) points; the jump is
            # what the far-side point adds on top of that trend.
            X  = hcat(ones(3), hs)
            ab = X \ [p.Q for p in ps]
            jump = pf.Q - (ab[1] + ab[2] * h_far)
            nmv  = (pf.ok && base.ok) ? count(pf.idx .!= base.idx) : -1
            allok = base.ok && pf.ok && all(p.ok for p in ps)
            @printf("JUMP,%s,%s,h_far=%+.0e,jump=%+.9f,slope=%.6f,Q_base=%.12f,Q_far=%.12f,nodes=%d,ok=%s\n",
                    rname, pn, h_far, jump, ab[2], base.Q, pf.Q, nmv, allok)
            push!(rows, (rname, pn, h_far, jump, ab[2], base.Q, pf.Q, nmv, allok))
        end
    end
    CSV.write(joinpath(out_logs(), "ojs_jump_tolerance_$(WINDOW)$(TAG).csv"), rows)
end


# ════════════════════════════════════════════════════════════
#  crossing — which site does a surviving step come from?
# ════════════════════════════════════════════════════════════
if MODE === :crossing
    # The two points that bracket the crossing on the +λ_S ray, at a tolerance tight
    # enough that nothing between them is solve noise.
    spec = _spec_with_sim(SPEC, _sim_with(SPEC.sim; tol_global = 1e-8, conv_streak = 2,
                                         global_B = 0, maxit_global = 600))
    j  = _IX["λ_S"]
    lo = eval_point(_shift(Θ0, j, 3e-7), spec)
    hi = eval_point(_shift(Θ0, j, 1e-6), spec)
    (lo.ok && hi.ok) || error("a bracketing point did not solve — nothing to attribute")

    row = findfirst(lo.idx .!= hi.idx)
    @printf("ΔQ across the crossing = %+.9f   (%.12f → %.12f)\n", hi.Q - lo.Q, lo.Q, hi.Q)
    @printf("aS row %d: p^oj node %d → %d,  Γ(p^oj) %.12g → %.12g  (%+.3e relative)\n",
            row, lo.idx[row], hi.idx[row], lo.Gpoj[row], hi.Gpoj[row],
            (hi.Gpoj[row] - lo.Gpoj[row]) / lo.Gpoj[row])
    @printf("straddling cells %d → %d,  employed mass in them %.6e → %.6e\n",
            lo.n_straddle, hi.n_straddle, lo.mass_straddle, hi.mass_straddle)
    @printf("wage grid endpoints: lo %+.3e   hi %+.3e  (absolute shift)\n",
            hi.w_lo - lo.w_lo, hi.w_hi - lo.w_hi)
    rel = [lo.m[i] == 0 ? NaN : (hi.m[i] - lo.m[i]) / abs(lo.m[i]) for i in eachindex(MK)]
    for i in sortperm(abs.(rel), rev = true)
        abs(rel[i]) > 1e-9 &&
            @printf("MOVED,%s,%.12g,%.12g,%.3e\n", MK[i], lo.m[i], hi.m[i], rel[i])
    end
    CSV.write(joinpath(out_logs(), "ojs_crossing_$(WINDOW)$(TAG).csv"),
              DataFrame(moment = String.(string.(MK)), below = lo.m, above = hi.m,
                        rel_step = rel))
end


# ════════════════════════════════════════════════════════════
#  jacobian — is a fixed-step Jacobian column a derivative?
# ════════════════════════════════════════════════════════════
if MODE === :jacobian
    HS = [3e-7, 1e-6, 3e-6]
    sim = TOL == 1e-4 ? SPEC.sim :
          _sim_with(SPEC.sim; tol_global = TOL, conv_streak = 2, global_B = 0,
                    maxit_global = 600)
    spec = _spec_with_sim(SPEC, sim)
    @printf("tol_global = %.1e   central differences at h = %s\n\n", sim.tol_global, HS)

    # Gw[a][:, j] is the whitened column for parameter j at step HS[a]. Threaded over
    # columns: each solve is already threaded over the ability grid, so the two layers
    # share the same pool and the pool stays saturated either way.
    Gw = [zeros(length(MK), D) for _ in HS]
    feasible = trues(D)
    @threads for j in 1:D
        for (a, h) in enumerate(HS)
            pp = eval_point(_shift(Θ0, j, h), spec)
            pm = eval_point(_shift(Θ0, j, -h), spec)
            if pp.ok && pm.ok
                Gw[a][:, j] = ((pp.m .- pm.m) ./ (2h) ./ SIGMA) .* DTHDT[j]
            else
                feasible[j] = false
            end
        end
    end

    # Spread: how far the three columns sit from their mean, relative to that mean.
    # A derivative is independent of the step, so a usable column has spread ≈ 0.
    Gbar   = (Gw[1] .+ Gw[2] .+ Gw[3]) ./ 3
    nrm    = [norm(Gbar[:, j]) for j in 1:D]
    spread = [nrm[j] > 0 ? maximum(norm(Gw[a][:, j] .- Gbar[:, j]) for a in 1:3) / nrm[j] : NaN
              for j in 1:D]
    # Alternative normalisations, kept because the definition has to be pinned against
    # the v26.0.0 log before before/after numbers can be compared to it.
    spread_pair = [nrm[j] > 0 ? maximum(norm(Gw[a][:, j] .- Gw[b][:, j])
                                        for a in 1:3, b in 1:3) / nrm[j] : NaN for j in 1:D]
    spread_nrm  = [begin
                       ns = [norm(Gw[a][:, j]) for a in 1:3]
                       mean(ns) > 0 ? std(ns) / mean(ns) : NaN
                   end for j in 1:D]

    J    = Gw[2]                              # the fixed h = 1e-6 Jacobian
    sv   = svd(J).S
    tolr = maximum(sv) * max(length(MK), D) * eps(Float64)
    for j in 1:D
        @printf("COL,%s,norm_h1e6=%.12g,norm_mean=%.12g,spread=%.12g,spread_pair=%.12g,spread_nrm=%.12g,fails=%s,feasible=%s\n",
                PNAME[j], norm(J[:, j]), nrm[j], spread[j], spread_pair[j], spread_nrm[j],
                spread[j] > 0.05, feasible[j])
    end
    @printf("\nCOND,tol=%.1e,cond=%.6e,sigma_max=%.6e,sigma_min=%.6e,rank=%d/%d,fails=%d\n",
            sim.tol_global, sv[1] / sv[end], sv[1], sv[end], count(>(tolr), sv), D,
            count(s -> s > 0.05, spread))
    CSV.write(joinpath(out_logs(),
                       @sprintf("ojs_jacobian_tol%.0e_%s%s.csv", TOL, WINDOW, TAG)),
              DataFrame(param = PNAME, colnorm_h1e6 = [norm(J[:, j]) for j in 1:D],
                        colnorm_mean = nrm, spread = spread, spread_pair = spread_pair,
                        spread_nrm = spread_nrm, feasible = feasible))
end

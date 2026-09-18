############################################################
# width_audit.jl — the criterion's own ΔQ=1 half-width for every free parameter,
# beside the chain's reported standard error.
#
# WHY THIS EXISTS. On the χ² ruler a change of 1 in Q is one standard deviation in a single
# coordinate, so the criterion's ΔQ=1 half-width and the chain's posterior sd are estimates of
# the same thing and should agree. Where they disagree by more than a factor of a few, the
# reported standard error is not describing the criterion, and the DIRECTION of the
# disagreement says which defect you have:
#
#   width/se ≫ 1   the chain cannot reach that coordinate's identified range. Under the
#                  logistic map dθ/dt = θ-lb, so a coordinate near a bound needs a huge
#                  t-displacement to move θ at all. Measured on base_fc under v20.0.0:
#                  b_S sat 6.25e-06 above its floor with a ΔQ=1 width of 0.0037, needing 586
#                  t-units against a chain t-sd of 0.039 — and its reported se was 2.46e-07,
#                  low by a factor of 1.5e+04. MCMC_SPACE = :theta is the fix.
#   width/se ≪ 1   that coordinate has not equilibrated. The reported sd is the spread of a
#                  population still growing, not of the target.
#   width/se ≈ 1   the two agree and the standard error means what it says.
#
# Run this on EVERY window before building a table. Which coordinates corner is window-
# specific, so a parameter that is fine on base_fc can be unreportable on crisis_covid.
#
# It doubles as the source of MCMC_INIT = :widths, which starts the DE-MC population at these
# widths instead of spending the run diffusing into them.
#
# COST ~600 solves (23 coordinates × 2 directions × ~13 evaluations), a few minutes.
#
# CAVEATS, all three of which belong in any table built from this.
#   1. These are CONDITIONAL widths: the other d-1 coordinates are held fixed. With
#      near-collinear pairs the marginal (profiled) width is wider, possibly much. Profiling
#      means re-minimising over the other d-1 at each grid point, which is ~d times the cost.
#   2. They are BASE-POINT sensitive. On base_fc the b_S ΔQ=1 width is 0.0037 from the seed
#      and 0.0014 from the posterior mean — a factor of 2.6, both correct for their point.
#   3. The feasible set is perforated, so a bracket can close on an Inf rather than on a
#      criterion rise. Those are reported as `hit_infeasible` rather than silently treated
#      as a bound.
#
# USAGE
#   ROYSEARCH_WINDOW=base_fc julia --project=. code/tools/width_audit.jl
#   ROYSEARCH_WIDTH_BASE=seed        # base point: postmean (default) or seed
############################################################

using LinearAlgebra, Statistics, Random, Printf, Serialization
using Distributions, FastGaussQuadrature, Interpolations, Parameters, Base.Threads
using Optim, CSV, DataFrames, Clustering, QuasiMonteCarlo, JSON3
BLAS.set_num_threads(1)

const RT = get(ENV, "RSROOT", normpath(joinpath(@__DIR__, "..", "..")))
include(joinpath(RT, "code", "paths.jl"))
for f in ["version.jl", "settings.jl"]; include(joinpath(RT, "code", "smm", f)) end
for f in ["grids.jl", "params.jl", "unskilled.jl", "skilled.jl", "solver.jl", "equilibrium.jl"]
    include(joinpath(RT, "code", "solver", f))
end
for f in ["moments.jl", "smm_params.jl", "bundle.jl", "smm.jl", "candidates.jl"]
    include(joinpath(RT, "code", "smm", f))
end

const WINDOW   = Symbol(get(ENV, "ROYSEARCH_WINDOW", "base_fc"))
# W_COND_TARGET is set in the entry points (smm_main.jl:420, MCMC_main.jl:488), not in
# settings.jl, so a tool that does not include an entry point cannot derive the suffix. The
# other four tools in this directory hardcode "_diagonalW"; this one at least lets the caller
# override, so an equalW run can be audited without editing the file.
const W_SUFFIX = get(ENV, "ROYSEARCH_W_SUFFIX", "_diagonalW")
const BASEMODE = Symbol(get(ENV, "ROYSEARCH_WIDTH_BASE", "postmean"))
BASEMODE in (:postmean, :seed) ||
    error("ROYSEARCH_WIDTH_BASE = $(BASEMODE); use postmean or seed.")

@printf("RoySearch v%s — criterion width audit\n", ROYSEARCH_VERSION)
@printf("  window = %s   weighting = %s   base = %s\n", WINDOW, W_SUFFIX, BASEMODE)

# The base point. postmean is the estimator, so it is the point whose widths a table needs;
# seed is offered because the two differ and the difference is itself worth seeing.
pm  = joinpath(out_estimates(), "estimate_$(WINDOW)$(W_SUFFIX)_postmean.jls")
sd_ = joinpath(out_estimates(), "estimate_$(WINDOW)$(W_SUFFIX).jls")
path = BASEMODE === :postmean && isfile(pm) ? pm : sd_
isfile(path) || error("no estimate bundle at $(path); run the SMM or the chain first.")
BASEMODE === :postmean && !isfile(pm) &&
    @printf("  (no postmean bundle; falling back to %s)\n", basename(sd_))
b    = open(deserialize, path)
spec = b.spec
t0   = collect(float.(getfield(b.result, 1)))     # bundles always store t
sp   = spec.free
d    = length(sp)
se   = nothing
chp  = joinpath(out_chains(), "chain_$(WINDOW)$(W_SUFFIX).jls")
if isfile(chp)
    ch = open(deserialize, chp)
    hasproperty(ch, :se_chain) && length(ch.se_chain) == d && (se = collect(float.(ch.se_chain)))
end
@printf("  base bundle: %s   chain se: %s\n\n", basename(path),
        se === nothing ? "not available" : basename(chp))

θof(k)     = sp[k].lb + (sp[k].ub - sp[k].lb) / (1 + exp(-t0[k]))
tof(k, v)  = (z = (v - sp[k].lb) / (sp[k].ub - sp[k].lb);
              (z <= 0 || z >= 1) ? NaN : log(z / (1 - z)))
dθdt(k)    = (s = 1 / (1 + exp(-t0[k])); (sp[k].ub - sp[k].lb) * s * (1 - s))
Q(t)       = smm_objective(t, spec)
const Q0   = Q(t0)
isfinite(Q0) || error("the base point is infeasible (Q = $(Q0)); nothing to profile.")
@printf("  Q at the base point = %.6f\n\n", Q0)

"""
    width(k, dir) -> (halfwidth, hit_infeasible)

Displacement of θ_k in direction `dir` at which Q rises by 1, everything else held. Brackets
on a geometric ladder anchored to the reported se (or to 1% of the box when there is none),
then bisects six times. An `Inf` closes the bracket like a criterion rise, because a point the
model cannot solve is not inside a confidence set either — but it is flagged, because the two
mean different things for a table.
"""
function width(k, dir)
    θ0k  = θof(k)
    room = dir > 0 ? sp[k].ub - θ0k : θ0k - sp[k].lb
    room <= 0 && return (NaN, false)
    # The ladder is anchored to the BOX, not to se(chain). Anchoring it to se made this tool
    # fail in exactly the case it exists to detect: on a short chain with se(b_S) = 9.8e-09,
    # se * 1e5 tops out at 9.8e-04, below the 3.7e-03 where ΔQ = 1 is actually reached, so the
    # search returned NaN for the one coordinate whose se is wrong by orders of magnitude. A
    # box-anchored ladder brackets whatever the criterion does, with or without a chain — which
    # also means this tool can be run before any chain exists.
    lo, hi, inf_hit = 0.0, NaN, false
    for f in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 0.98)
        s  = min(f * room, 0.98 * room)
        tv = tof(k, θ0k + dir * s)
        isnan(tv) && (hi = s; break)
        tt = copy(t0); tt[k] = tv; q = Q(tt)
        if !isfinite(q)
            hi = s; inf_hit = true; break
        elseif q - Q0 >= 1.0
            hi = s; break
        else
            lo = s
        end
        s >= 0.97 * room && return (NaN, false)     # ΔQ=1 not reached inside the box
    end
    isnan(hi) && return (NaN, inf_hit)
    for _ in 1:12
        m  = 0.5 * (lo + hi)
        tv = tof(k, θ0k + dir * m)
        if isnan(tv); hi = m; continue; end
        tt = copy(t0); tt[k] = tv; q = Q(tt)
        if !isfinite(q)
            hi = m; inf_hit = true
        elseif q - Q0 >= 1.0
            hi = m
        else
            lo = m
        end
    end
    return (0.5 * (lo + hi), inf_hit)
end

wup = fill(NaN, d); wdn = fill(NaN, d)
iup = falses(d);    idn = falses(d)
@threads for k in 1:d
    wup[k], iup[k] = width(k, +1)
    wdn[k], idn[k] = width(k, -1)
end

@printf("%-34s %11s %11s %11s %11s %10s %8s\n",
        "parameter", "θ̂", "ΔQ=1 up", "ΔQ=1 dn", "se(chain)", "width/se", "verdict")
rows = NamedTuple[]
for k in 1:d
    wmin = min(isfinite(wup[k]) ? wup[k] : Inf, isfinite(wdn[k]) ? wdn[k] : Inf)
    r = (se !== nothing && isfinite(se[k]) && se[k] > 0 && isfinite(wmin)) ? wmin / se[k] : NaN
    v = !isfinite(r)      ? "-"        :
        r > 10.0          ? "SE LOW"   :
        r < 0.1           ? "SE HIGH"  : "ok"
    @printf("%-34s %11.5g %11.4g %11.4g %11.4g %10.4g %8s\n",
            first(sp[k].label, 34), θof(k), wup[k], wdn[k],
            se === nothing ? NaN : se[k], r, v)
    push!(rows, (param = sp[k].label, name = string(sp[k].name), block = string(sp[k].block),
                 theta = θof(k), lb = sp[k].lb, ub = sp[k].ub,
                 dist_lb = θof(k) - sp[k].lb, dtheta_dt = dθdt(k),
                 wQ1_up = wup[k], wQ1_dn = wdn[k],
                 hit_infeasible_up = iup[k], hit_infeasible_dn = idn[k],
                 se_chain = se === nothing ? NaN : se[k], width_over_se = r, verdict = v))
end

out = joinpath(out_logs(), "width_audit_$(WINDOW)$(W_SUFFIX).csv")
CSV.write(out, DataFrame(rows))
@printf("\n  → %s\n", out)

nlow  = count(r -> r.verdict == "SE LOW",  rows)
nhigh = count(r -> r.verdict == "SE HIGH", rows)
ninf  = count(r -> r.hit_infeasible_up || r.hit_infeasible_dn, rows)
@printf("\n  %d coordinate(s) with se too LOW  (unreachable — use MCMC_SPACE = :theta)\n", nlow)
@printf("  %d coordinate(s) with se too HIGH (not equilibrated — more generations)\n", nhigh)
@printf("  %d coordinate(s) whose bracket closed on an infeasible point\n", ninf)
@printf("\n  These are CONDITIONAL widths at one base point. See the header for the three\n")
@printf("  caveats that belong in any table built from them.\n")

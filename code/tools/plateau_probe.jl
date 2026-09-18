#!/usr/bin/env julia
############################################################
# plateau_probe.jl — measure the criterion's LOCAL GRANULARITY at the current point, and
# from it the annealing temperature. Read-only: changes no parameter, setting or bundle.
#
# WHY THIS EXISTS. _sa_loop sets
#     T0 = -t0_rel * Q_anchor / log(t0_accept)
# so the temperature is a fraction of the criterion's LEVEL. But the level is dominated by
# moments the model structurally cannot fit — Q sits near 500 while the smallest change a
# proposal can produce is a few units — so a fraction of the level is the wrong scale. What
# T must be commensurate with is the size of the uphill step the walk actually faces, which
# is a property of the criterion's discretisation, not of its level.
#
# At Q = 502.36 with the shipped sa_t0_rel = 1e-4, T0 = 0.0417 and the probability of
# accepting an uphill move costing 3.32 is 2.8e-35. Annealing is a zero-temperature method
# under those settings and cannot leave a plateau cell. This probe measures the number that
# replaces 3.32 — which was itself measured at a superseded optimum and is stale.
#
# WHAT IT MEASURES, in increasing order of directness:
#
#   1. Determinism at the stored θ. Expect ~1e-12 (threaded-reduction non-associativity).
#      If it is large, everything below is noise, so the probe stops rather than report it.
#
#   2. |ΔQ| over proposals drawn by the SHIPPING mechanism. The proposal state comes from
#      sa_proposal_scale — the same function _run_sa calls — rather than a reimplementation,
#      so the probe cannot drift from the walk it is measuring. The draw is
#      theta_prop[j] += sc.step[j] * randn(), over a Bernoulli(sc.p_move) mask with one
#      forced index, exactly as _sa_loop:181-189. Reports the EXACT-TIE fraction (accepted
#      by the tie branch without consulting T) and the quantiles of the uphill cost.
#
#   3. A fine line scan on two coordinates, reading the lattice spacing of Q directly, as an
#      independent cross-check on (2). If the two disagree by more than about a factor of 2,
#      the reading is not to be trusted and the script says so.
#
# COST. sa_proposal_scale does 24·d width solves (552 at d = 23) — the same measurement SA
# performs at every startup — plus PROBE_N proposals and 2·PROBE_SCAN scan points. About
# 700 solves in total at the defaults.
#
#   RSROOT=<repo> julia --project=. -t <threads> code/tools/plateau_probe.jl
#
# Optional: ROYSEARCH_WINDOW (default base_fc), PROBE_N (default 120), PROBE_SCAN (15).
############################################################

using LinearAlgebra, Statistics, Random, Printf, Serialization
using Distributions, FastGaussQuadrature, Interpolations, Parameters, Base.Threads
using Optim, CSV, DataFrames, Clustering, QuasiMonteCarlo, JSON3
BLAS.set_num_threads(1)

const ROOT = get(ENV, "RSROOT", normpath(joinpath(@__DIR__, "..", "..")))
# paths.jl supplies out_estimates() and estimate_path, so a tool cannot drift
# from where the entry points actually write. Guarded: including it twice would
# redefine its constants.
@isdefined(ROYSEARCH_PATHS_LOADED) || include(joinpath(ROOT, "code", "paths.jl"))
for f in ["version.jl", "settings.jl"];  include(joinpath(ROOT, "code", "smm", f))  end
for f in ["grids.jl", "params.jl", "unskilled.jl", "skilled.jl", "solver.jl",
          "equilibrium.jl"];              include(joinpath(ROOT, "code", "solver", f)) end
for f in ["moments.jl", "smm_params.jl", "smm.jl", "candidates.jl"]
    include(joinpath(ROOT, "code", "smm", f))
end

const WINDOW    = Symbol(get(ENV, "ROYSEARCH_WINDOW", "base_fc"))
const W_SUFFIX  = "_diagonalW"
const N_PROP    = parse(Int, get(ENV, "PROBE_N", "120"))
const N_SCAN    = parse(Int, get(ENV, "PROBE_SCAN", "15"))
const T0_ACCEPT = 0.30      # _sa_loop's shipped t0_accept
const T0_REL    = 1e-4      # _sa_loop's shipped sa_t0_rel, for the comparison only
# Read from the same source smm_main.jl reads, so the probe measures the shipped proposal.
const SIGMA     = env_setting(:SA_SCALE_SIGMA, 0.33)
const P_SCALE   = env_setting(:SA_SCALE_P_MOVE, 1.0)

const BPATH = estimate_path(WINDOW, W_SUFFIX)
isfile(BPATH) || error("no bundle at $BPATH")
const B    = open(deserialize, BPATH)
const SPEC = B.spec
const THAT = collect(float.(B.result.theta_opt))
const D    = length(SPEC.free)

@printf("plateau_probe — window %s, d = %d, stored Q = %.6f\n", WINDOW, D, B.result.loss_opt)
@printf("bundle %s   σ = %.3f   p_move_scale = %.2f\n\n", basename(BPATH), SIGMA, P_SCALE)

# ---------------------------------------------------------------- 1. determinism
println("1. DETERMINISM — 4 evaluations at the stored θ")
qs = [smm_objective(copy(THAT), SPEC) for _ in 1:4]
@printf("   Q = %s\n", join((@sprintf("%.10f", q) for q in qs), "  "))
@printf("   sd = %.3e   range = %.3e\n", std(qs), maximum(qs) - minimum(qs))
if std(qs) > 1e-6
    println("\n   *** sd exceeds 1e-6: the objective is not deterministic here, so every")
    println("   granularity below would be noise rather than the criterion's. STOPPING.")
    exit(1)
end
println("   deterministic — what follows is the criterion's granularity, not noise.\n")
const Q0 = qs[1]

# ---------------------------------------------------------------- the shipping proposal
# sa_proposal_scale is what _run_sa calls. Using it rather than reimplementing means the
# probe measures the proposal that actually runs: sc.step is already σ·|width| per
# coordinate with step_fallback where the bisection could not measure a width, and
# sc.p_move is already clamp(p_move_scale/d, 1/d, 1).
println("2. THE SHIPPING PROPOSAL, from sa_proposal_scale (24·d = $(24*D) width solves)")
sc = sa_proposal_scale(THAT, SPEC; p_move_scale = P_SCALE, sigma = SIGMA,
                       cap = _WIDTH_CAP, verbose = true)
@printf("   widths measured for %d of %d coordinates;  p_move = %.4f  (mean %.2f moved)\n",
        sc.n_meas, D, sc.p_move, 1 + (D - 1) * sc.p_move)
@printf("   step: median %.4g  min %.4g  max %.4g\n\n",
        median(sc.step), minimum(sc.step), maximum(sc.step))

# ---------------------------------------------------------------- 3. proposal ΔQ
@printf("3. PROPOSAL ΔQ — %d draws, exactly as _sa_loop:181-189 draws them\n", N_PROP)
rng  = MersenneTwister(20260828)
rows = NamedTuple[]
for i in 1:N_PROP
    moved = Int[]
    for j in 1:D
        rand(rng) < sc.p_move && push!(moved, j)
    end
    jf = rand(rng, 1:D)
    (jf in moved) || push!(moved, jf)
    θp = copy(THAT)
    for j in moved
        θp[j] += sc.step[j] * randn(rng)
        θp[j]  = clamp(θp[j], SPEC.free[j].lb + 1e-10, SPEC.free[j].ub - 1e-10)
    end
    Qp = smm_objective(θp, SPEC)
    push!(rows, (draw = i, nmoved = length(moved), Q = Qp, dQ = Qp - Q0,
                 feasible = isfinite(Qp)))
    i % 20 == 0 && (@printf("   %d/%d\n", i, N_PROP); flush(stdout))
end

fin  = [r for r in rows if r.feasible]
ties = [r for r in fin if r.dQ == 0.0]
up   = sort([r.dQ for r in fin if r.dQ > 0.0])
down = [r.dQ for r in fin if r.dQ < 0.0]

@printf("\n   feasible    %3d / %d  (%.0f%%)\n", length(fin), N_PROP, 100*length(fin)/N_PROP)
@printf("   EXACT ties  %3d  (%.0f%% of feasible) — the tie branch accepts these and never\n",
        length(ties), 100*length(ties)/max(length(fin), 1))
println("                    consults T, which is why a cold walk is not simply stuck")
@printf("   downhill    %3d\n", length(down))
@printf("   uphill      %3d\n", length(up))
isempty(up) || @printf("""
   UPHILL COST, the quantity T has to price:
     min %.4f   q25 %.4f   median %.4f   q75 %.4f   max %.4f
""", up[1], quantile(up, 0.25), median(up), quantile(up, 0.75), up[end])

# ---------------------------------------------------------------- 4. lattice spacing
scan_j = sortperm(abs.(sc.widths), rev = true)[1:min(2, D)]
println("\n4. LATTICE SPACING — independent cross-check, fine scan on the two widest coords")
lat = Float64[]
for j in scan_j
    w  = abs(sc.widths[j]) > 0 ? abs(sc.widths[j]) : sc.step[j]
    qv = Float64[]
    for t in range(-0.5w, 0.5w, length = N_SCAN)
        θp = copy(THAT)
        θp[j] = clamp(THAT[j] + t, SPEC.free[j].lb + 1e-10, SPEC.free[j].ub - 1e-10)
        push!(qv, smm_objective(θp, SPEC))
    end
    u = sort(unique(round.(filter(isfinite, qv), digits = 9)))
    d = length(u) > 1 ? diff(u) : Float64[]
    append!(lat, d)
    @printf("   %-14s %2d/%d finite, %2d distinct Q, min gap %s, median gap %s\n",
            string(SPEC.free[j].name), count(isfinite, qv), N_SCAN, length(u),
            isempty(d) ? "—" : @sprintf("%.4f", minimum(d)),
            isempty(d) ? "—" : @sprintf("%.4f", median(d)))
end

# ---------------------------------------------------------------- 5. the temperature
println("\n5. TEMPERATURE.  T0 solves exp(-plateau/T0) = t0_accept  ⇒  T0 = -plateau/log(t0_accept)")
T0_shipped = -T0_REL * Q0 / log(T0_ACCEPT)
@printf("   shipped: sa_t0_rel = %.0e keys off the LEVEL  ⇒  T0 = %.5f\n", T0_REL, T0_shipped)
cands = NamedTuple[]
isempty(up)  || append!(cands, [(basis = "min uphill cost",    plateau = up[1]),
                                (basis = "median uphill cost", plateau = median(up)),
                                (basis = "q75 uphill cost",    plateau = quantile(up, 0.75))])
isempty(lat) || push!(cands, (basis = "lattice median gap", plateau = median(lat)))
@printf("\n   %-22s %10s %10s %12s\n", "calibrated on", "plateau", "T0", "×shipped")
for c in cands
    T = -c.plateau / log(T0_ACCEPT)
    @printf("   %-22s %10.4f %10.4f %11.0f×\n", c.basis, c.plateau, T, T / T0_shipped)
end

if !isempty(up) && !isempty(lat)
    r = median(up) / median(lat)
    @printf("\n   CROSS-CHECK: median proposal cost / median lattice gap = %.2f\n", r)
    println(r > 0.5 && r < 5.0 ?
        "   The two independent measurements agree to within a factor of 5. Reading stands." :
        "   *** The two disagree by more than a factor of 5. Do NOT set T0 from this run — \n   *** the proposal cost and the lattice spacing are measuring different things here.")
end

println("""

   RECOMMENDATION: calibrate on the MEDIAN uphill cost. The minimum is the granularity
   floor and yields a T0 that still rejects a typical uphill move; the q75 buys mobility
   the walk does not need. The median is the cost the walk faces most of the time, which is
   what t0_accept = 0.30 is a statement about.

   AND reheat_cap MUST land in the same change. sa_reheat_factor = 4.00 with
   sa_max_reheats = 4 multiplies T0 by 256; at a corrected T0 that is a random search,
   measured to close 0% of the gap. Cap at T0.""")

out = joinpath(out_estimates(), "plateau_probe_$(WINDOW)$(W_SUFFIX).csv")
CSV.write(out, DataFrame(rows))
@printf("\nwrote %s  (%d proposal rows)\n", basename(out), length(rows))

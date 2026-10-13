#!/usr/bin/env julia
############################################################
# gates/check_hire_margin.jl — the solved hiring margin must be the closed form's
#
# WHAT THIS CATCHES. The unskilled hiring margin is not solved for: by the hireable-set
# lemma a type is hired iff a_U > a^h = b̃_U / P_U, at every tightness, and the solver
# weights each ability cell by its covered fraction above a^h (`hire_fraction`). The
# reservation quality p*_U is solved, and the lemma says p*_U(a_U) < 1 exactly on the
# same set. If the two ever disagree away from a^h, either the lemma's premises have
# been broken by a model change (a non-degenerate entry quality, a new term in the
# surplus) or the p*_U update has, and `hire_fraction` is then weighting the wrong set
# with no error raised anywhere.
#
# A node within PRECISION of a^h may disagree: there 1 − p*_U is of the order of the
# solve tolerance, so the node's classification is decided by rounding. That case is
# reported, not failed — the closed form is exact and is what the solver uses.
#
#   julia --project=. code/gates/check_hire_margin.jl
############################################################

using LinearAlgebra, Statistics, Random, Printf, Serialization
using Distributions, FastGaussQuadrature, Interpolations, Parameters, Base.Threads
using Optim, CSV, DataFrames, Clustering, QuasiMonteCarlo, JSON3
BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "repo_root.jl"))
const ROOT = find_repo_root()
@isdefined(ROYSEARCH_PATHS_LOADED) || include(joinpath(ROOT, "code", "paths.jl"))
for f in ("version", "settings")
    include(joinpath(ROOT, "code", "smm", f * ".jl"))
end
for f in ("grids", "params", "unskilled", "skilled", "solver", "equilibrium")
    include(joinpath(ROOT, "code", "solver", f * ".jl"))
end
for f in ("moments", "smm_params", "bundle", "smm")
    include(joinpath(ROOT, "code", "smm", f * ".jl"))
end

const W_SUFFIX = get(ENV, "ROYSEARCH_W_SUFFIX", "_diagonalW")
# Relative distance to a^h inside which a node's p*_U classification is rounding. 1 − p*_U
# is first order in (a_U − a^h)/a^h, so a node that close sits under the 1e-10 classification
# threshold or the solve error; measured at the base_fc estimate, that band is 1.0e-10 wide.
const PRECISION = 1e-6

"""
    hire_margin_report(model) -> NamedTuple

The closed-form hired set against the solved one at a converged model: the straddling
cell and its χ, and every node where `x_i > a^h` and `p*_U(x_i) < 1` disagree, with its
relative distance to `a^h`.
"""
function hire_margin_report(model::Model)
    x  = model.grids.x
    aʰ = model.unsk_par.bU / model.unsk_par.PU
    χ  = hire_fraction(model)
    p  = clamp01.(model.unsk_cache.pstar)
    disagree = [i for i in eachindex(x) if (x[i] > aʰ) != (p[i] < 1.0 - 1e-10)]
    gap      = [abs(x[i] - aʰ) / max(aʰ, eps()) for i in disagree]
    k        = findfirst(c -> 0.0 < c < 1.0, χ)
    return (aʰ = aʰ, k = k, χk = k === nothing ? NaN : χ[k],
            disagree = disagree, gap = gap, one_minus_p = [1.0 - p[i] for i in disagree])
end

@printf("RoySearch v%s — unskilled hiring-margin gate\n", ROYSEARCH_VERSION)

fails = String[]
n_checked = 0
for window in WINDOWS, point in ESTIMATE_POINTS
    path = estimate_path(window, W_SUFFIX; point = point)
    isfile(path) || continue
    b = read_bundle(path)
    (b === nothing || b.spec === nothing || b.result === nothing) && continue
    sel = equilibrium_at(collect(float.(getfield(b.result, 1))), b.spec)
    if sel.model === nothing
        push!(fails, "$(window) $(point): no certified equilibrium at the stored optimum.")
        continue
    end
    global n_checked += 1
    r = hire_margin_report(sel.model)
    cell = r.k === nothing ? "none (χ ∈ {0,1})" : @sprintf("%d, χ = %.6f", r.k, r.χk)
    @printf("  %-25s a^h = %.10f   straddling cell %s   disagreements %d\n",
            "$(window) $(point)", r.aʰ, cell, length(r.disagree))
    for (i, g, q) in zip(r.disagree, r.gap, r.one_minus_p)
        @printf("      node %d: |x − a^h|/a^h = %.3e, 1 − p* = %.3e  %s\n", i, g, q,
                g <= PRECISION ? "(within precision: closed form wins)" : "FAIL")
        g > PRECISION && push!(fails, @sprintf("%s %s: node %d disagrees at %.3e from a^h.",
                                               window, point, i, g))
    end
end

if n_checked == 0
    println("  SKIPPED: no estimate bundle in $(relpath(out_estimates(), ROOT)).")
    exit(0)
end
if isempty(fails)
    println("  solved hiring margin agrees with the closed form: OK")
    exit(0)
end
println("FINDINGS ($(length(fails))):")
for f in fails; println("    ", f); end
exit(1)

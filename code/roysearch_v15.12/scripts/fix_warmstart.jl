#!/usr/bin/env julia
############################################################
# scripts/fix_warmstart.jl — which spec field rejects the warm-start seed?
#
# check_warmstart_all.jl established that every saved θ̂ reproduces its saved Q under
# the spec its own bundle carries, so the solver is not the cause.  What remains is
# that the spec smm_main.jl assembles today differs from the saved one in a way that
# makes the same θ infeasible.
#
# This script finds which field, by substitution rather than by inspection: it scores
# the bundle θ̂ under today's spec with ONE field replaced by the bundle's version, one
# field at a time.  The field whose substitution turns Inf into a finite Q is the
# cause.  Inspection cannot do this — the λ_S bound differs between the two specs but
# the seed is interior to both, so a diff alone points at an innocent field.
#
# Prerequisite — dump the spec the estimation builds:
#   ROYSEARCH_DUMP_SPEC=true ROYSEARCH_WINDOW=base_fc julia --project=. code/smm/smm_main.jl
#
# Then:
#   julia --project=. code/scripts/fix_warmstart.jl base_fc
#
# Cost: one solve per candidate field (a handful of seconds each).
############################################################

# Mirrors smm_main.jl's preamble exactly, Base.Threads included: the solver files
# call a bare @threads, which is only in scope with that import.
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
using CSV
using DataFrames
using Serialization
using Clustering
using QuasiMonteCarlo
using JSON3

include(joinpath(@__DIR__, "repo_root.jl"))
const ROOT        = find_repo_root()
const CODE        = joinpath(ROOT, "code")
const ALL_WINDOWS = ["base_fc", "crisis_fc", "base_covid", "crisis_covid"]
const W_SUFFIX    = "_diagonalW"

# Same order as smm_main.jl: grids before params, candidates last.
for f in ("solver/grids.jl", "solver/params.jl", "solver/unskilled.jl",
          "solver/skilled.jl", "solver/solver.jl", "solver/equilibrium.jl",
          "smm/moments.jl", "smm/smm_params.jl", "smm/smm.jl", "smm/candidates.jl")
    include(joinpath(CODE, f))
end

"""
    bundle_optimum(path) -> (result, spec)

The reported optimum and the spec it was scored under.  `result` is serialised as
`results`, a single SMMResult for a lone run and a vector for a multistart.
"""
function bundle_optimum(path)
    saved = deserialize(path)
    res   = saved.result isa AbstractVector ?
            argmin(r -> r.loss_opt, saved.result) : saved.result
    return res, saved.spec
end

"""
    with_field(spec, field, value) -> SMMSpec

`spec` with one field replaced.  SMMSpec is immutable and has no validating
constructor, so a positional rebuild is the whole operation.
"""
function with_field(spec :: SMMSpec, field :: Symbol, value)
    return SMMSpec((f === field ? value : getfield(spec, f)
                    for f in fieldnames(SMMSpec))...)
end

"""
    seed_theta(spec_target, res_bundle, spec_bundle) -> Vector{Float64}

The bundle's θ̂ expressed in `spec_target`'s coordinates.  θ is stored unconstrained,
and the logit transform depends on each parameter's box, so a raw θ vector means a
different point under a different box.  Mapping through constrained space is what the
warm-start loader itself does, and skipping it would confound a box change with a
genuine rejection.
"""
function seed_theta(spec_target :: SMMSpec, res_bundle, spec_bundle :: SMMSpec)
    saved = Dict((ps.block, ps.name) =>
                 _to_constrained(res_bundle.theta_opt[i], ps.lb, ps.ub)
                 for (i, ps) in enumerate(spec_bundle.free))
    return [begin
                x = get(saved, (ps.block, ps.name), ps.init)
                _to_unconstrained(clamp(x, ps.lb + 1e-10, ps.ub - 1e-10), ps.lb, ps.ub)
            end
            for ps in spec_target.free]
end

function main(argv = ARGS)
    window = isempty(argv) ? "base_fc" : String(first(argv))
    window in ALL_WINDOWS ||
        error("unknown window $window; valid: " * join(ALL_WINDOWS, ", "))

    bpath = joinpath(ROOT, "output", "smm", "smm_result_$(window)$(W_SUFFIX).jls")
    dpath = joinpath(ROOT, "output", "smm", "spec_dump_$(window)$(W_SUFFIX).jls")
    isfile(bpath) || error("no bundle at $bpath")
    isfile(dpath) || error("""
        no spec dump at $dpath
        Produce it first:
          ROYSEARCH_DUMP_SPEC=true ROYSEARCH_WINDOW=$window julia --project=. code/smm/smm_main.jl""")

    res_b, spec_b = bundle_optimum(bpath)
    spec_now      = deserialize(dpath).spec

    @printf("window: %s\n", window)
    @printf("  bundle spec : %d free, saved Q = %.6e\n", length(spec_b.free), res_b.loss_opt)
    @printf("  today's spec: %d free\n\n", length(spec_now.free))

    # ── the two reference points ─────────────────────────────────────────────
    Q_bundle = smm_objective(res_b.theta_opt, spec_b)
    θ_now    = seed_theta(spec_now, res_b, spec_b)
    Q_now    = smm_objective(θ_now, spec_now)

    fq(q) = isfinite(q) ? @sprintf("%.6e", q) : "Inf"
    @printf("  θ̂ under the BUNDLE's spec : %s\n", fq(Q_bundle))
    @printf("  θ̂ under TODAY's spec      : %s\n\n", fq(Q_now))

    if isfinite(Q_now)
        println("""
        Today's spec already accepts the seed, so there is nothing to bisect. The
        earlier Inf came from a spec that has since changed — re-run the estimation
        and confirm the [SA init] line now reports a finite Q0.""")
        return nothing
    end
    isfinite(Q_bundle) || error("the bundle's own spec also rejects θ̂ — run check_warmstart.jl first")

    # ── substitute one field at a time ───────────────────────────────────────
    # free is excluded: replacing it replaces the boxes AND the seed's coordinates,
    # so a finite Q would not identify which of the two mattered.  Box effects are
    # already visible in check_warmstart.jl's box diff.
    candidates = [:fixed, :moments, :sim, :run, :W, :q_scale]
    println("─── substituting the bundle's version of one field into today's spec ───")
    @printf("  %-10s %-15s %s\n", "field", "Q", "verdict")

    culprits = Symbol[]
    for f in candidates
        trial = with_field(spec_now, f, getfield(spec_b, f))
        # The seed is remapped per trial: substituting `free` is excluded, so the
        # boxes are today's and the mapping is unchanged, but this keeps the call
        # uniform if the candidate list is ever extended.
        q = try
            smm_objective(seed_theta(trial, res_b, spec_b), trial)
        catch e
            @printf("  %-10s %-15s threw %s\n", f, "—", typeof(e))
            continue
        end
        finite = isfinite(q)
        finite && push!(culprits, f)
        @printf("  %-10s %-15s %s\n", f, fq(q),
                finite ? "FINITE — this field is the cause" : "still rejected")
    end

    println()
    if isempty(culprits)
        println("""
        No single substitution restores a finite Q, so the rejection needs more than
        one field. Substitute the whole bundle spec to confirm, then bisect in pairs
        starting from the fields check_warmstart_all.jl showed differing.""")
    else
        @printf("Cause: %s\n", join(string.(culprits), ", "))
        println("""
        Compare that field between the two specs and decide whether today's value is
        correct. If it is, the bundle is simply stale and base_fc should be
        re-estimated from :default; if the bundle's value is correct, the estimation
        is assembling the wrong spec and that is the bug to fix.""")
    end

    return (window = window, Q_bundle = Q_bundle, Q_now = Q_now, culprits = culprits)
end

main()

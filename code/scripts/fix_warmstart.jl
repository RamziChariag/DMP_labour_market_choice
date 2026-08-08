#!/usr/bin/env julia
############################################################
# scripts/fix_warmstart.jl — which spec field rejects the warm-start seed?
#
# check_warmstart_all.jl established that every saved θ̂ reproduces its saved Q under
# the spec its own bundle carries, so the solver is not the cause.  What remains is
# that the spec smm_main.jl assembles today differs from the saved one in a way that
# makes the same θ infeasible.
#
# This script localises the difference by substitution rather than by inspection.
# Inspection cannot do it: the λ_S bound differs between the two specs but the seed is
# interior to both, so a diff alone points at an innocent field.  Three passes, each
# narrower than the last:
#
#   1. the free set        — membership and ORDER, since unpack_θ reads θ by position
#                            and a parameter today's spec adds is seeded from its
#                            generic default, a cold start on that one coordinate
#   2. one field at a time — the field whose substitution turns Inf finite is the
#                            cause, with the complete bundle spec as a control
#   3. one coordinate at a time — when no single field suffices, which PARAMETER the
#                            disagreement bites on, starting from a point today's
#                            spec accepts and moving one saved value in at a time
#
# Prerequisite — dump the spec the estimation builds:
#   ROYSEARCH_DUMP_SPEC=true ROYSEARCH_WINDOW=base_fc julia --project=. code/smm/smm_main.jl
#
# Then:
#   julia --project=. code/scripts/fix_warmstart.jl base_fc
#
# Cost: one solve per candidate field, plus one per free parameter if the coordinate
# scan runs (a handful of seconds each).
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
    bundle_values(res, spec) -> Dict{Tuple{Symbol,Symbol},Float64}

The bundle's θ̂ in constrained space, keyed by (block, name).
"""
bundle_values(res, spec :: SMMSpec) =
    Dict((ps.block, ps.name) => _to_constrained(res.theta_opt[i], ps.lb, ps.ub)
         for (i, ps) in enumerate(spec.free))

"""
    seed_theta(spec_target, res_bundle, spec_bundle) -> Vector{Float64}

The bundle's θ̂ expressed in `spec_target`'s coordinates.  θ is stored unconstrained,
and the logit transform depends on each parameter's box, so a raw θ vector means a
different point under a different box.  Mapping through constrained space is what the
warm-start loader itself does, and skipping it would confound a box change with a
genuine rejection.
"""
function seed_theta(spec_target :: SMMSpec, res_bundle, spec_bundle :: SMMSpec)
    saved = bundle_values(res_bundle, spec_bundle)
    return [begin
                x = get(saved, (ps.block, ps.name), ps.init)
                _to_unconstrained(clamp(x, ps.lb + 1e-10, ps.ub - 1e-10), ps.lb, ps.ub)
            end
            for ps in spec_target.free]
end

"""
    free_diff(spec_bundle, spec_now, saved) -> (fallbacks, dropped, reordered)

How the two free sets disagree.  `unpack_θ` reads θ by POSITION in `spec.free`, so
three things matter and none is visible in a count of free parameters:

  fallbacks  in today's set but not the bundle's — takes the generic ParamSpec
             default, so that one coordinate is a cold start however good the rest
             of the seed is, and a single bad coordinate is enough to return Inf
  dropped    in the bundle's set but not today's — its saved value is discarded
  reordered  same membership, different positions
"""
function free_diff(spec_bundle :: SMMSpec, spec_now :: SMMSpec, saved)
    key(ps)   = (ps.block, ps.name)
    now_keys  = key.(spec_now.free)
    bun_keys  = key.(spec_bundle.free)
    fallbacks = [ps for ps in spec_now.free if !haskey(saved, key(ps))]
    dropped   = [ps for ps in spec_bundle.free if !(key(ps) in now_keys)]
    reordered = Set(now_keys) == Set(bun_keys) && now_keys != bun_keys
    return fallbacks, dropped, reordered
end

"""
    coordinate_scan(spec_now, spec_bundle, saved)

Which single coordinate makes today's spec reject the seed.  Starts from the seed
today's spec accepts by construction — every free parameter at its own ParamSpec
default — and moves ONE coordinate at a time to the bundle's saved value.  A
coordinate whose move flips a finite Q to Inf is one today's spec cannot combine with
the defaults, which names the parameter to look at.

This complements the field bisection: fields say WHICH PART of the spec disagrees,
coordinates say WHICH PARAMETER the disagreement bites on.
"""
function coordinate_scan(spec_now :: SMMSpec, spec_bundle :: SMMSpec, saved)
    θ_default = [_to_unconstrained(clamp(ps.init, ps.lb + 1e-10, ps.ub - 1e-10),
                                   ps.lb, ps.ub) for ps in spec_now.free]
    Q_default = smm_objective(θ_default, spec_now)
    @printf("\n  baseline: every parameter at its ParamSpec default → Q = %s\n",
            isfinite(Q_default) ? @sprintf("%.6e", Q_default) : "Inf")
    if !isfinite(Q_default)
        println("""
          The default point is itself rejected, so there is no accepted baseline to
          move away from and a coordinate scan cannot attribute anything. This is a
          spec that solves nowhere near its own defaults — the case to hand to the
          theorist rather than to debug further here.""")
        return nothing
    end

    println("\n  moving one coordinate from its default to the bundle's saved value:")
    @printf("    %-6s %-6s %-13s %-13s %-15s %s\n",
            "block", "name", "default", "saved", "Q", "")
    offenders = Tuple{Symbol,Symbol}[]
    for (i, ps) in enumerate(spec_now.free)
        x = get(saved, (ps.block, ps.name), nothing)
        x === nothing && continue
        θ    = copy(θ_default)
        θ[i] = _to_unconstrained(clamp(x, ps.lb + 1e-10, ps.ub - 1e-10), ps.lb, ps.ub)
        q    = try smm_objective(θ, spec_now) catch; Inf end
        bad  = !isfinite(q)
        bad && push!(offenders, (ps.block, ps.name))
        @printf("    %-6s %-6s %-13.6f %-13.6f %-15s %s\n",
                ps.block, ps.name, ps.init, x,
                isfinite(q) ? @sprintf("%.6e", q) : "Inf",
                bad ? "← rejects" : "")
    end

    println()
    if isempty(offenders)
        println("""
      No single coordinate rejects on its own, so the infeasibility is genuinely joint
      — the saved point is only feasible under the bundle's own configuration, and no
      one parameter carries it. Treat the bundle as stale and re-estimate this window
      from :default.""")
    else
        @printf("  Coordinates today's spec rejects: %s\n",
                join(["$(b).$(n)" for (b, n) in offenders], ", "))
        println("""
      Each of those is a saved value the current spec cannot accept alongside the
      defaults. Compare its box and the parameters it interacts with — for a shock
      rate that means the (ξ, δ, λ) triple, whose product governs the separation and
      ladder margins, so a bound moved on one of them can make a previously fine value
      infeasible.""")
    end
    return offenders
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

    # ── the free set, before any substitution ────────────────────────────────
    # Checked first because it is the one difference a field count cannot show: two
    # specs can both report 25 free parameters and still disagree on WHICH 25, and
    # any parameter today's spec adds is seeded from its generic ParamSpec default.
    saved = bundle_values(res_b, spec_b)
    fallbacks, dropped, reordered = free_diff(spec_b, spec_now, saved)

    println("─── free set ───")
    if isempty(fallbacks) && isempty(dropped) && !reordered
        println("  identical membership and order")
    end
    if !isempty(fallbacks)
        println("  in today's spec but NOT in the bundle — seeded from ParamSpec default:")
        for ps in fallbacks
            @printf("      %-6s %-6s init = %.8f   box [%.6f, %.6f]\n",
                    ps.block, ps.name, ps.init, ps.lb, ps.ub)
        end
    end
    if !isempty(dropped)
        println("  in the bundle but NOT in today's spec — its saved value is discarded:")
        for ps in dropped
            @printf("      %-6s %-6s saved = %.8f\n",
                    ps.block, ps.name, get(saved, (ps.block, ps.name), NaN))
        end
    end
    reordered && println("  same membership, DIFFERENT order (unpack_θ reads θ by position)")

    # ── substitute one field at a time ───────────────────────────────────────
    # `free` is included, and interpreting it needs care: substituting it restores
    # both the bundle's boxes and its exact free set, so a finite Q there means the
    # cause is in one of those two and the fallback list above says which.
    # `all` is the control: if even the complete bundle spec is rejected under
    # today's code then the difference is not in the spec at all.
    candidates = [:free, :fixed, :moments, :sim, :run, :W, :q_scale]
    println("\n─── substituting the bundle's version of one field into today's spec ───")
    @printf("  %-10s %-15s %s\n", "field", "Q", "verdict")

    culprits = Symbol[]
    for f in candidates
        trial = with_field(spec_now, f, getfield(spec_b, f))
        # Remapped per trial: substituting `free` changes the boxes, and θ is stored
        # unconstrained, so the same raw vector denotes a different point under a
        # different box.  Mapping through constrained space keeps the point fixed.
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

    q_all = smm_objective(seed_theta(spec_b, res_b, spec_b), spec_b)
    @printf("  %-10s %-15s %s\n", "all", fq(q_all),
            isfinite(q_all) ? "control: the complete bundle spec is accepted" :
                              "CONTROL FAILED — see below")

    println()
    if !isempty(culprits)
        @printf("Cause: %s\n", join(string.(culprits), ", "))
        if :free in culprits
            println("""
            `free` is the cause, which means either the boxes or the membership of the
            free set. The free-set section above distinguishes them: any parameter
            listed as seeded from its ParamSpec default is a cold start on that one
            coordinate, and one bad coordinate is enough to return Inf regardless of
            how good the other twenty-four are.""")
        else
            println("""
            Compare that field between the two specs and decide whether today's value is
            correct. If it is, the bundle is stale and this window should be re-estimated
            from :default; if the bundle's value is correct, the estimation is assembling
            the wrong spec and that is the bug to fix.""")
        end
    elseif !isfinite(q_all)
        println("""
        Even the complete bundle spec is rejected here, while check_warmstart.jl scored
        the same bundle finite. The difference between the two is the seed: that script
        uses the stored θ verbatim, this one remaps it through constrained space. A
        remapped seed that fails where the raw one succeeds means a box moved under a
        parameter whose value sits at or beyond the new edge — read the box diff.""")
    else
        # Every single-field substitution failed but the full bundle spec is fine, so
        # no one field is sufficient.  Locating the offending COORDINATE is more useful
        # than continuing to bisect fields: it names the parameter to look at.
        println("""
        No single field restores a finite Q, yet the complete bundle spec is accepted,
        so the rejection needs a combination. Rather than bisect in pairs, the more
        informative question is which COORDINATE today's spec cannot tolerate:""")
        coordinate_scan(spec_now, spec_b, saved)
    end

    return (window = window, Q_bundle = Q_bundle, Q_now = Q_now, culprits = culprits,
            q_all = q_all, fallbacks = [(ps.block, ps.name) for ps in fallbacks],
            dropped = [(ps.block, ps.name) for ps in dropped], reordered = reordered)
end

main()

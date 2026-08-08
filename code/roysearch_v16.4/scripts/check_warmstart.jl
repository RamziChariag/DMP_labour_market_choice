#!/usr/bin/env julia
############################################################
# scripts/check_warmstart.jl — why does a saved optimum not reproduce?
#
# Re-scores ONE window's saved θ̂ under the SPEC THE BUNDLE CARRIES — its own grid,
# tolerances, boxes, fixed block and weight matrix — then reports every field of
# that spec.  Nothing is reconstructed from today's configuration, so the only thing
# differing between the saved run and this one is solver and objective code.
#
#   julia --project=. code/scripts/check_warmstart.jl base_fc
#   julia --project=. code/scripts/check_warmstart.jl                # base_fc
#
# One window per process, deliberately: the solver and SMM files define module-level
# constants, so including them twice in one session prints "redefinition of constant"
# and leaves the second definition in an undefined state.  To cover all four windows
# use scripts/check_warmstart_all.jl, which spawns a fresh process per window.
#
# Alongside its report the script writes output/smm/warmstart_<window>.json, which is
# what the all-windows driver aggregates into a cross-window comparison.
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

# Same order as smm_main.jl: grids before params, candidates last.  The order is
# load-bearing — each file uses names defined by the ones above it.
for f in ("solver/grids.jl", "solver/params.jl", "solver/unskilled.jl",
          "solver/skilled.jl", "solver/solver.jl", "solver/equilibrium.jl",
          "smm/moments.jl", "smm/smm_params.jl", "smm/smm.jl", "smm/candidates.jl")
    include(joinpath(CODE, f))
end

"""
    bundle_optimum(path) -> (result, spec)

The reported optimum and the spec it was scored under.  `result` is serialised as
`results`, a single SMMResult for a lone run and a vector for a multistart; the
reported optimum is the lowest-loss entry either way.
"""
function bundle_optimum(path)
    saved = deserialize(path)
    res   = saved.result isa AbstractVector ?
            argmin(r -> r.loss_opt, saved.result) : saved.result
    return res, saved.spec
end

"""
    spec_fields(spec) -> Dict{String,Any}

Every scalar the objective reads, flattened into one comparable table: the grid, the
solver tolerances and stall knobs, the fixed block, and the weighting summary.  A box
diff alone cannot explain a rejected seed that is interior to both the old and the
new box, so the comparison has to cover the rest of the spec too.
"""
function spec_fields(spec)
    d = Dict{String,Any}()
    for f in (:Nx, :Np_U, :Np_S)
        d["run.$f"] = getproperty(spec.run, f)
    end
    for f in fieldnames(typeof(spec.sim))
        v = getproperty(spec.sim, f)
        v isa Number && (d["sim.$f"] = v)
    end
    for k in keys(spec.fixed)
        d["fixed.$k"] = getproperty(spec.fixed, k)
    end
    wd = diag(spec.W)
    d["W.size"]    = size(spec.W, 1)
    d["W.scored"]  = count(!iszero, wd)
    d["W.cond"]    = cond(spec.W)
    d["q_scale"]   = spec.q_scale
    d["n_free"]    = length(spec.free)
    d["n_moments"] = length(keys(spec.moments))
    return d
end

"""
    box_changes(spec) -> Vector{String}

Boxes that moved between the bundle and the current parameter table, each annotated
with whether the saved value is interior to both — if it is, the box cannot be what
rejects the seed.
"""
function box_changes(spec)
    cur = Dict((ps.block, ps.name) => ps for ps in default_free_params())
    out = String[]
    for ps in spec.free
        c = get(cur, (ps.block, ps.name), nothing)
        if c === nothing
            push!(out, @sprintf("%-6s %-6s missing from current specs", ps.block, ps.name))
        elseif !isapprox(ps.lb, c.lb; atol = 1e-12) || !isapprox(ps.ub, c.ub; atol = 1e-12)
            interior = ps.init > c.lb && ps.init < c.ub
            push!(out, @sprintf("%-6s %-6s bundle [%.6f, %.6f] → current [%.6f, %.6f]%s",
                  ps.block, ps.name, ps.lb, ps.ub, c.lb, c.ub,
                  interior ? "  (seed interior to both)" : "  (SEED CLAMPED)"))
        end
    end
    return out
end

function main(argv = ARGS)
    window = isempty(argv) ? "base_fc" : String(first(argv))
    window in ALL_WINDOWS ||
        error("unknown window $window; valid: " * join(ALL_WINDOWS, ", "))

    path = joinpath(ROOT, "output", "smm", "smm_result_$(window)$(W_SUFFIX).jls")
    isfile(path) || error("no bundle at $path")
    res, spec = bundle_optimum(path)

    @printf("window: %s\n", window)
    @printf("  bundle : %s\n", path)
    @printf("  saved Q = %.10e   converged = %s   free = %d\n",
            res.loss_opt, res.converged, length(spec.free))

    t0   = time()
    Q    = smm_objective(res.theta_opt, spec)
    secs = time() - t0

    # A relative gap is the meaningful comparison: Q is a sum of squared t-statistics
    # of order 1e3, so an absolute difference of 1 is solver noise while a relative
    # difference of 1e-2 is a changed model.
    rel = isfinite(Q) ? abs(Q - res.loss_opt) / abs(res.loss_opt) : Inf
    verdict = !isfinite(Q) ? "REJECTED"   :
              rel < 1e-6   ? "reproduces" :
              rel < 1e-2   ? "drifted"    : "DIFFERENT"

    @printf("\n  θ̂ under the bundle's own spec: %s   rel %s   [%s]  %.1fs\n",
            isfinite(Q) ? @sprintf("%.10e", Q) : "Inf",
            isfinite(rel) ? @sprintf("%.2e", rel) : "—", verdict, secs)

    fields = spec_fields(spec)
    boxes  = box_changes(spec)

    println("\n─── boxes changed vs the current parameter table ───")
    isempty(boxes) ? println("  none") : foreach(s -> println("  ", s), boxes)

    println("\n─── spec the bundle carries ───")
    for k in sort(collect(keys(fields)))
        v = fields[k]
        @printf("  %-22s %s\n", k,
                v isa Integer ? string(v) :
                v isa AbstractFloat ? @sprintf("%.6e", v) : string(v))
    end

    # Written for the all-windows driver: aggregating JSON is robust where parsing
    # this report's text would not be.
    out = joinpath(ROOT, "output", "smm", "warmstart_$(window).json")
    open(out, "w") do io
        JSON3.write(io, (window = window, saved = res.loss_opt,
                         now = isfinite(Q) ? Q : nothing, rel = isfinite(rel) ? rel : nothing,
                         verdict = verdict, secs = secs, boxes = boxes,
                         fields = fields))
    end
    @printf("\nwrote %s\n", out)

    return (window = window, saved = res.loss_opt, now = Q, rel = rel,
            verdict = verdict, boxes = boxes, fields = fields, spec = spec)
end

main()

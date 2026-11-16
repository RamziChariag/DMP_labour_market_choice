#!/usr/bin/env julia
############################################################
# gates/check_threading.jl — a solve nested in a threaded loop must equal the serial one.
#
# The DE population, the width scan and the FD Jacobian evaluate the criterion inside
# `Threads.@threads`. A solver that cannot nest (`@threads :static`, `threadid()`-indexed
# scratch) throws there, and `smm_objective` reports the exception as Inf. This gate
# evaluates Q at a stored estimate serially and inside `Threads.@threads for i in 1:2`, and
# requires the threaded values finite and bitwise equal to the serial one.
#
# Needs more than one thread, or it checks nothing:
#   julia --project=. -t 2 code/gates/check_threading.jl
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

const W_SUFFIX = weighting_suffix(run_weighting())

@printf("RoySearch v%s — threaded-evaluation gate (%d threads)\n",
        ROYSEARCH_VERSION, Threads.nthreads())

if Threads.nthreads() < 2
    println("  SKIPPED: one thread. Re-run with -t 2 or more; this gate cannot fail on one thread.")
    exit(0)
end

# The point is the first stored estimate found: its bundle carries the run's own SimParams
# and weights. `findfirst`, not a `for … break`: at top level an assignment in a loop body
# creates a local, and the selection would stay empty.
found = [estimate_path(w, W_SUFFIX; point = p) for p in ESTIMATE_POINTS for w in WINDOWS]
ip    = findfirst(isfile, found)
if ip === nothing
    println("  SKIPPED: no estimate bundle in $(relpath(out_estimates(), ROOT)) — " *
            "run the SMM for any window first, then re-run this gate.")
    exit(0)
end
path = found[ip]

b = read_bundle(path)
if b === nothing || b.spec === nothing || b.result === nothing
    println("  SKIPPED: $(basename(path)) did not load — " *
            "rebuild the bundles under the current code and re-run.")
    exit(0)
end
spec = b.spec
t̂    = collect(float.(getfield(b.result, 1)))          # bundles always store t
@printf("  point: %s — %s\n", basename(path), bundle_summary(b))

Q_serial = smm_objective(copy(t̂), spec)

# Two tasks make the loop concurrent; a fixed trip count keeps the cost independent of -t.
Q_thr = fill(NaN, 2)
Threads.@threads for i in 1:2
    Q_thr[i] = smm_objective(copy(t̂), spec)
end

@printf("  serial    Q = %.10f\n", Q_serial)
@printf("  threaded  Q = %.10f, %.10f\n", Q_thr[1], Q_thr[2])

fails = String[]
if !isfinite(Q_serial)
    push!(fails, "the serial solve is already infeasible at the stored optimum, so this " *
                 "gate cannot separate a threading defect from a bad point. Check " *
                 "`converged` on the bundle first.")
elseif !all(isfinite, Q_thr)
    push!(fails, "a solve inside `Threads.@threads for i in 1:2` returned a non-finite " *
                 "objective. The usual cause is `@threads :static` or a " *
                 "`threadid()`-indexed buffer reintroduced in the solver: `:static` " *
                 "cannot nest, and smm_objective reports the resulting exception as Inf.")
elseif !all(q -> q === Q_serial, Q_thr)
    # @sprintf needs a literal format string, so the text is concatenated after it.
    push!(fails, @sprintf("threaded and serial disagree (Δ = %.3e, %.3e).", Q_thr[1] - Q_serial,
                          Q_thr[2] - Q_serial) *
                 " A threaded evaluation must be bitwise identical to the serial one; a " *
                 "schedule-dependent reduction is a silent wrong number, which is worse " *
                 "than the Inf it replaces.")
end

if isempty(fails)
    println("  threaded evaluation is finite and bitwise equal to serial: OK")
    exit(0)
end
println("FINDINGS ($(length(fails))):")
for f in fails; println("    ", f); end
exit(1)

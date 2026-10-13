#!/usr/bin/env julia
############################################################
# gates/check_threading.jl — a solve must survive being run inside a threaded loop
#
# WHAT THIS CATCHES, AND WHY IT IS A GATE. Every caller that evaluates the criterion at
# more than one parameter point at a time does it with `Threads.@threads`: the DE
# population, the ΔQ=1 width measurement, the adaptive-FD Jacobian. Inside that loop
# `@threads :static` — which is what a `threadid()`-indexed scratch buffer requires —
# throws `cannot be used concurrently or nested`. `smm_objective` catches every exception
# and returns `Inf`, so the caller reads a solver that cannot nest as a parameter point
# that has no equilibrium. v40.0.0 shipped in exactly that state: the serial path was
# exact and every threaded evaluation returned `Inf`, which disabled the width
# measurement (0 of 23 coordinates), the Jacobian (0 of 23 columns) and the DE.
#
# The regression was invisible because every test ran SERIALLY. This gate is the one that
# does not: it runs a solve inside `Threads.@threads for i in 1:2` and requires the
# result to be finite AND bitwise equal to the serial value.
#
# Run from anywhere, and with more than one thread or it checks nothing:
#   julia --project=. -t 4 code/gates/check_threading.jl
############################################################

using LinearAlgebra, Statistics, Random, Printf, Serialization
using Distributions, FastGaussQuadrature, Interpolations, Parameters, Base.Threads
using Optim, CSV, DataFrames, Clustering, QuasiMonteCarlo, JSON3
BLAS.set_num_threads(1)

# repo_root.jl sits at code/ beside paths.jl since v19.6.0: it is shared by gates/,
# tools/ and ops/, so one path up is the same expression from all three.
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

@printf("RoySearch v%s — threaded-evaluation gate (%d threads)\n",
        ROYSEARCH_VERSION, Threads.nthreads())

if Threads.nthreads() < 2
    println("  SKIPPED: one thread. Re-run with -t 2 or more; this gate cannot fail on one thread.")
    exit(0)
end

# The parameter point comes from a saved estimate bundle rather than from a block written
# here: the bundle carries the authoritative `SimParams` and weight matrix, so the gate
# cannot drift from the configuration the estimation actually runs under. Any window's
# bundle exercises the same two loops, so the first one present is enough.
# `findfirst` and not a `for` loop with a `break`: at top level in a script, an assignment
# inside a loop body creates a local rather than writing the global, so the loop would
# always leave the selection empty and the gate would always skip.
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
            "migrate the bundles (code/tools/migrate_bundles.jl) and re-run.")
    exit(0)
end
spec = b.spec
t̂    = collect(float.(getfield(b.result, 1)))          # bundles always store t
@printf("  point: %s — %s\n", basename(path), bundle_summary(b))

Q_serial = smm_objective(copy(t̂), spec)

# `1:2` and not `1:nthreads()`: two tasks are enough to make the loop concurrent, which is
# the whole condition, and a fixed trip count keeps the gate's cost independent of -t.
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
    # One literal format string: @sprintf is a macro and cannot take a concatenation.
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

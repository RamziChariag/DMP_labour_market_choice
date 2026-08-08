#!/usr/bin/env julia
############################################################
# run_all.jl — batch driver for the four estimation windows
#
# Runs each (task, window) as its OWN Julia process. A same-process loop is not
# possible: smm_main.jl and MCMC_main.jl are scripts whose configuration includes
# `const` bindings (LAMBDA_W and the path constants), and a const cannot be rebound
# between iterations. Separate processes also mean a window that errors, or is
# killed by the OOM reaper, does not take the rest of the batch with it.
#
# Configuration reaches each child through ROYSEARCH_* environment variables read
# by the _env_* helpers in the two drivers. The literal defaults in those files
# remain the single-run configuration, so this script changes nothing about how
# they behave when run directly.
#
# Usage
#   julia code/run_all.jl                          # four windows, estimation only
#   julia code/run_all.jl --task=mcmc              # standard errors for all four
#   julia code/run_all.jl --task=both
#   julia code/run_all.jl --windows=base_fc,crisis_fc
#   julia code/run_all.jl --threads=16 --dry-run
#   julia code/run_all.jl --sa-max-reheats=60 --nm-max-iter=40000
#   julia code/run_all.jl --concurrent          # all windows at once, threads split
#
# Two scheduling modes. The default runs windows SEQUENTIALLY with every thread
# given to one child. `--concurrent` runs all windows at once with the threads
# split between them, which is usually the faster way to finish a batch: the
# solver's @threads loop runs over the Nx grid, so its speedup is subject to the
# serial outer tightness and U↔S coupling loops, and an extra whole window is
# worth more than an extra thread inside one solve. Sequential remains the default
# because concurrent multiplies peak memory by the number of windows.
#
# Logs land in output/logs/{task}_{window}_{timestamp}.log and are tee'd to the
# console, so a detached run (nohup, tmux) still leaves a complete record.
############################################################

using Dates, Printf

const CODE_DIR     = @__DIR__
const PROJECT_ROOT = abspath(joinpath(CODE_DIR, ".."))
const LOG_DIR      = joinpath(PROJECT_ROOT, "output", "logs")

const ALL_WINDOWS = ["base_fc", "base_covid", "crisis_fc", "crisis_covid"]

# --flag → ROYSEARCH_* variable. Only flags actually passed are exported, so an
# unset flag leaves the driver's own default in force.
const ENV_MAP = Dict(
    "sa-max-iter"    => "ROYSEARCH_SA_MAX_ITER",
    "sa-max-reheats" => "ROYSEARCH_SA_MAX_REHEATS",
    "nm-max-iter"    => "ROYSEARCH_NM_MAX_ITER",
    "nm-no-improve"  => "ROYSEARCH_NM_NO_IMPROVE",
    "init-mode"      => "ROYSEARCH_INIT_MODE",
    "w-cond-target"  => "ROYSEARCH_W_COND_TARGET",
    "lambda-w"       => "ROYSEARCH_LAMBDA_W",
    "mcmc-gens"      => "ROYSEARCH_MCMC_GENS",
    "jac-only"       => "ROYSEARCH_JAC_ONLY",
)

"""
    parse_args(argv) -> Dict{String,String}

Accepts `--key=value` and bare `--flag` (stored as "true"). An unknown key is an
error rather than silently ignored, so a typo fails before a long batch starts.
"""
function parse_args(argv)
    known = union(Set(keys(ENV_MAP)),
                  Set(["task", "windows", "threads", "dry-run", "concurrent"]))
    opts  = Dict{String,String}()
    for a in argv
        startswith(a, "--") || error("unexpected argument $a (options start with --)")
        body = a[3:end]
        k, v = occursin('=', body) ? split(body, '=', limit = 2) : (body, "true")
        k in known || error("unknown option --$k; valid: " * join(sort(collect(known)), ", "))
        opts[k] = v
    end
    return opts
end

"""
    run_one(script, window, opts, nthr, dry; tag_lines = false) -> (ok, seconds, logfile)

Run one (script, window) pair in a fresh Julia process, tee'ing output to a
timestamped log. Returns rather than throwing so the batch continues past a
failure; the summary table reports which windows failed.
"""
function run_one(script::String, window::String, opts::Dict{String,String},
                 nthr::Int, dry::Bool; tag_lines::Bool = false)
    mkpath(LOG_DIR)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    tag   = script == "smm_main.jl" ? "estimate" : "mcmc"
    logf  = joinpath(LOG_DIR, "$(tag)_$(window)_$(stamp).log")

    env = copy(ENV)
    env["ROYSEARCH_WINDOW"]  = window
    env["JULIA_NUM_THREADS"] = string(nthr)
    for (flag, var) in ENV_MAP
        haskey(opts, flag) && (env[var] = opts[flag])
    end

    cmd = setenv(`$(Base.julia_cmd()[1]) --project=$PROJECT_ROOT --threads=$nthr
                  $(joinpath(CODE_DIR, "smm", script))`, env)

    @printf("\n%s\n", "="^70)
    @printf("  %-8s  %-13s  threads=%d  start %s\n", tag, window, nthr,
            Dates.format(now(), "HH:MM:SS"))
    @printf("  log: %s\n%s\n", logf, "="^70)
    flush(stdout)

    dry && return (true, 0.0, logf)

    t0 = time()
    ok = false
    try
        open(logf, "w") do io
            proc = open(cmd, "r")
            # Under --concurrent the four children interleave on one console, so
            # every console line carries its window.  The file stays untagged: it
            # holds one window's output already, and a prefix would break the
            # column alignment of the spec and moment tables.
            pre = tag_lines ? rpad("[" * String(window) * "]", 15) : ""
            for line in eachline(proc)          # tee: durable log + live console
                println(io, line)                 # durable log: untagged
                println(pre, line)                # console: tagged when concurrent
                flush(io); flush(stdout)
            end
            wait(proc)
            ok = success(proc)
        end
    catch e
        @printf("  FAILED: %s\n", sprint(showerror, e))
        ok = false
    end
    return (ok, time() - t0, logf)
end

function main(argv)
    opts = parse_args(argv)
    task = get(opts, "task", "estimate")
    task in ("estimate", "mcmc", "both") ||
        error("--task must be estimate, mcmc or both; got $task")

    windows = String.(split(get(opts, "windows", join(ALL_WINDOWS, ",")), ','))
    for w in windows
        w in ALL_WINDOWS || error("unknown window $w; valid: " * join(ALL_WINDOWS, ", "))
    end

    # Default to every thread the machine reports, so one script saturates both a
    # laptop and a large VM with no edit.
    nthr       = parse(Int, get(opts, "threads", string(Sys.CPU_THREADS)))
    dry        = get(opts, "dry-run", "false") == "true"
    concurrent = get(opts, "concurrent", "false") == "true"

    @printf("RoySearch batch\n")
    @printf("  task    : %s\n", task)
    @printf("  windows : %s\n", join(windows, ", "))
    @printf("  threads : %d  (machine reports %d)\n", nthr, Sys.CPU_THREADS)
    @printf("  schedule: %s\n", concurrent ?
            @sprintf("%d windows at once, %d threads each", length(windows),
                     max(1, nthr ÷ length(windows))) : "one window at a time")
    for flag in sort(collect(keys(ENV_MAP)))
        haskey(opts, flag) && @printf("  %-16s = %s\n", ENV_MAP[flag], opts[flag])
    end
    dry && println("  DRY RUN — nothing will be executed")

    # Estimation must finish for a window before its chain can seed from the saved
    # bundle, so with --task=both the script is the OUTER loop.
    scripts = task == "estimate" ? ["smm_main.jl"] :
              task == "mcmc"     ? ["MCMC_main.jl"] :
                                   ["smm_main.jl", "MCMC_main.jl"]

    results = NamedTuple[]
    if concurrent
        # Threads are split between the windows so the total never oversubscribes
        # the machine; at least 1 each. Windows within a script run together, but
        # the scripts stay ordered so a chain still seeds from a finished bundle.
        per = max(1, nthr ÷ length(windows))
        for script in scripts
            tasks = [Threads.@spawn run_one(script, w, opts, per, dry; tag_lines = true)
                     for w in windows]
            for (w, t) in zip(windows, tasks)
                ok, secs, logf = fetch(t)
                push!(results, (script = script, window = w, ok = ok, secs = secs, log = logf))
            end
        end
    else
        for script in scripts, w in windows
            ok, secs, logf = run_one(script, w, opts, nthr, dry)
            push!(results, (script = script, window = w, ok = ok, secs = secs, log = logf))
        end
    end

    println("\n", "="^70); println("  SUMMARY"); println("="^70)
    @printf("  %-13s %-10s %-8s %10s\n", "window", "task", "status", "minutes")
    for r in results
        @printf("  %-13s %-10s %-8s %10.1f\n", r.window,
                r.script == "smm_main.jl" ? "estimate" : "mcmc",
                r.ok ? "ok" : "FAILED", r.secs / 60)
    end
    nfail = count(r -> !r.ok, results)
    @printf("\n  %d/%d succeeded, total %.2f h\n",
            length(results) - nfail, length(results),
            sum(r.secs for r in results) / 3600)
    return nfail == 0 ? 0 : 1
end

exit(main(ARGS))

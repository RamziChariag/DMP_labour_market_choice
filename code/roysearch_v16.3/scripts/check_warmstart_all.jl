#!/usr/bin/env julia
############################################################
# scripts/check_warmstart_all.jl — run check_warmstart.jl on every window
#
# One child process per window.  A fresh process is required rather than merely
# tidy: the solver and SMM files define module-level constants (MOMENT_NAMES,
# DEEP_PARAMS, …), so including them a second time in one session prints
# "redefinition of constant" and leaves those constants in an undefined state — the
# comparison would then be running against a half-overwritten module.
#
# Each child writes output/smm/warmstart_<window>.json; this driver reads those back
# and prints the comparison that a single window cannot give: a spec field differing
# between a reproducing window and a rejected one is a candidate cause, while a field
# constant across all four is not.
#
#   julia --project=. code/scripts/check_warmstart_all.jl
#   julia --project=. code/scripts/check_warmstart_all.jl base_fc crisis_fc
#
# Child stdout is passed through, so each window's own report is still readable.
############################################################

using Printf, JSON3, CSV, DataFrames

include(joinpath(@__DIR__, "repo_root.jl"))
const ROOT        = find_repo_root()
const ALL_WINDOWS = ["base_fc", "crisis_fc", "base_covid", "crisis_covid"]
const CHILD       = joinpath(@__DIR__, "check_warmstart.jl")

"""
    run_window(window) -> Union{Nothing,Dict}

Score one window in a child process and read back the JSON it wrote.  Returns
`nothing` when the child fails, so one bad window does not abort the batch.
"""
function run_window(window :: AbstractString)
    json = joinpath(ROOT, "output", "smm", "warmstart_$(window).json")
    isfile(json) && rm(json)          # never report a stale run as a fresh one

    @printf("\n%s\n  %s\n%s\n", "="^72, window, "="^72)
    flush(stdout)

    # --project inherits this session's environment, so the child resolves the same
    # package versions without needing its own instantiate.
    cmd = `$(Base.julia_cmd()[1]) --project=$(Base.active_project()) $CHILD $window`
    ok  = try
        run(cmd)
        true
    catch e
        @printf("  child failed: %s\n", sprint(showerror, e))
        false
    end

    ok && isfile(json) || return nothing
    return JSON3.read(read(json, String), Dict)
end

function main(argv = ARGS)
    windows = isempty(argv) ? ALL_WINDOWS : collect(String.(argv))
    for w in windows
        w in ALL_WINDOWS || error("unknown window $w; valid: " * join(ALL_WINDOWS, ", "))
    end

    @printf("Re-scoring every saved θ̂ under its own bundle spec\n")
    @printf("  repo : %s\n", ROOT)

    results = Dict{String,Any}()
    for w in windows
        r = run_window(w)
        r === nothing || (results[w] = r)
    end

    println("\n", "="^72)
    println("  CROSS-WINDOW COMPARISON")
    println("="^72)

    live = [(w, results[w]) for w in windows if haskey(results, w)]
    if isempty(live)
        println("\nNo window produced a result.")
        return results
    end

    println("\n─── reproduction ───")
    @printf("  %-13s %-15s %-15s %-10s %s\n", "window", "saved Q", "now", "rel", "verdict")
    for (w, r) in live
        now = get(r, "now", nothing)
        rel = get(r, "rel", nothing)
        @printf("  %-13s %-15s %-15s %-10s %s\n", w,
                @sprintf("%.6e", r["saved"]),
                now === nothing ? "Inf" : @sprintf("%.6e", now),
                rel === nothing ? "—"   : @sprintf("%.2e", rel),
                r["verdict"])
    end

    # The comparison the batch exists for.  Only fields that actually differ are
    # printed: a constant field cannot explain a rejection that hits some windows
    # and not others.
    println("\n─── spec fields that DIFFER across windows ───")
    allkeys = sort(collect(union((Set(keys(r["fields"])) for (_, r) in live)...)))
    varying = String[]
    for k in allkeys
        vals = [get(r["fields"], k, nothing) for (_, r) in live]
        same = all(v -> (v isa Number && vals[1] isa Number) ?
                        isapprox(Float64(v), Float64(vals[1]); rtol = 1e-12) :
                        v == vals[1], vals)
        same || push!(varying, k)
    end

    fmt(v) = v === nothing            ? "—" :
             v isa Integer            ? string(v) :
             v isa AbstractFloat      ? @sprintf("%.4e", v) : string(v)

    if isempty(varying)
        println("  none — every bundle carries the same grid, tolerances, fixed block")
        println("  and weighting, so none of these explains a window-specific rejection.")
    else
        @printf("  %-22s %s\n", "field", join([rpad(w, 15) for (w, _) in live]))
        for k in varying
            @printf("  %-22s %s\n", k,
                    join([rpad(fmt(get(r["fields"], k, nothing)), 15) for (_, r) in live]))
        end
    end

    println("\n─── box changes vs the current parameter table ───")
    for (w, r) in live
        bx = get(r, "boxes", String[])
        isempty(bx) ? @printf("  %-13s none\n", w) :
                      (@printf("  %-13s\n", w); foreach(s -> println("      ", s), bx))
    end

    nrep = count(r -> r["verdict"] == "reproduces", (r for (_, r) in live))
    ninf = count(r -> get(r, "now", nothing) === nothing, (r for (_, r) in live))
    @printf("\n%d/%d reproduce, %d rejected\n", nrep, length(live), ninf)
    if ninf == length(live)
        println("""
        Every bundle is rejected, so the cause is in code shared by all four windows
        — the solver or the objective guards — not in any window's configuration.""")
    elseif ninf > 0
        println("""
        Some bundles reproduce and others do not, so the solver still solves. Read the
        differing-fields table above: a field that separates a reproducing window from
        a rejected one is the candidate cause.""")
    elseif nrep == length(live)
        println("""
        Every saved Q reproduces under its own bundle spec, so the solver is not the
        cause. A warm start that fails does so on the spec smm_main.jl assembles at
        run time; compare the fields above against the estimation header.""")
    end

    summary = [(window = w, saved = r["saved"],
                now = get(r, "now", nothing), rel = get(r, "rel", nothing),
                verdict = r["verdict"]) for (w, r) in live]
    out = joinpath(ROOT, "output", "smm", "check_warmstart.csv")
    CSV.write(out, DataFrame(summary))
    @printf("\nwrote %s\n", out)

    return results
end

main()

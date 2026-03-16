############################################################
# code/data_cleaning/main.jl — Data-pipeline entry point
#
# Usage (from project root):
#   julia code/data_cleaning/main.jl
#
# Project layout expected:
#   project_root/
#     code/
#       solver/           ← model solver (existing)
#       smm/              ← SMM estimation (existing)
#       data_cleaning/    ← THIS FOLDER
#         main.jl             entry point (this file)
#         utils.jl            shared helpers, paths, windows
#         clean_cps_basic.jl  CPS Basic Monthly cleaning
#         clean_cps_asec.jl   CPS ASEC cleaning
#         clean_jolts.jl      JOLTS vacancy cleaning
#         make_transitions.jl transition rates from matched CPS
#         make_moments.jl     24 moments × 4 windows
#     data/
#       raw/
#         cps_basic/      ← IPUMS CPS Basic extract
#         cps_asec/       ← IPUMS CPS ASEC extract
#         jolts/          ← BLS JOLTS flat files
#       derived/          ← cleaned Arrow files + moment CSVs
#     output/
#       plots/
#       tables/
#
# Pipeline stages
# ───────────────
#   1. clean_cps_basic()    raw → cps_basic_clean.arrow
#   2. clean_cps_asec()     raw → cps_asec_clean.arrow
#   3. clean_jolts()        raw → jolts_clean.arrow
#   4. make_transitions()   cps_basic_clean → transitions.arrow
#   5. make_moments()       all derived → moments_{window}.csv
#                                        + sigma_{window}.csv
############################################################

println("="^60)
println("  Segmented Search Model — Data Pipeline")
println("="^60)
flush(stdout)

# ── Packages ──────────────────────────────────────────────────────────────
print("Loading packages... "); flush(stdout)

using DataFrames
using CSV
using Arrow
using Statistics
using Printf

println("done."); flush(stdout)

# ── Load pipeline scripts ─────────────────────────────────────────────────
print("Loading pipeline modules... "); flush(stdout)

include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "clean_cps_basic.jl"))
include(joinpath(@__DIR__, "clean_cps_asec.jl"))
include(joinpath(@__DIR__, "clean_jolts.jl"))
include(joinpath(@__DIR__, "make_transitions.jl"))
include(joinpath(@__DIR__, "make_moments.jl"))

println("done."); flush(stdout)

# ── Verify raw data directories exist ─────────────────────────────────────
println("\nChecking raw data directories...")
for (label, dir) in [("CPS Basic", RAW_CPS_BASIC_DIR),
                      ("CPS ASEC",  RAW_CPS_ASEC_DIR),
                      ("JOLTS",     RAW_JOLTS_DIR)]
    if isdir(dir)
        files = readdir(dir)
        @printf("  ✓ %-12s  %s  (%d files)\n", label, dir, length(files))
    else
        @printf("  ✗ %-12s  %s  (MISSING — create and populate)\n", label, dir)
        mkpath(dir)
    end
end
flush(stdout)


# ============================================================
# Stage 1: Clean CPS Basic Monthly
# ============================================================
println("\n" * "─"^60)
println("Stage 1: Cleaning CPS Basic Monthly")
println("─"^60)
flush(stdout)

cps_basic = clean_cps_basic()


# ============================================================
# Stage 2: Clean CPS ASEC
# ============================================================
println("\n" * "─"^60)
println("Stage 2: Cleaning CPS ASEC")
println("─"^60)
flush(stdout)

cps_asec = clean_cps_asec()


# ============================================================
# Stage 3: Clean JOLTS
# ============================================================
println("\n" * "─"^60)
println("Stage 3: Cleaning JOLTS")
println("─"^60)
flush(stdout)

jolts = clean_jolts()


# ============================================================
# Stage 4: Build transition rates
# ============================================================
println("\n" * "─"^60)
println("Stage 4: Building transition rates from matched CPS")
println("─"^60)
flush(stdout)

transitions = make_transitions()


# ============================================================
# Stage 5: Compute all moments
# ============================================================
println("\n" * "─"^60)
println("Stage 5: Computing 24 moments × 4 windows")
println("─"^60)
flush(stdout)

all_moments = make_moments()


# ── Helper: human-readable file size ──────────────────────────────────────
function _human_size(bytes::Integer)
    bytes < 1024       && return "$bytes B"
    bytes < 1024^2     && return "$(round(bytes/1024; digits=1)) KB"
    return "$(round(bytes/1024^2; digits=1)) MB"
end

# ============================================================
# Summary
# ============================================================
println("\n" * "="^60)
println("  Pipeline complete.")
println("="^60)
println("\nDerived files in: $DERIVED_DIR")
for f in sort(readdir(DERIVED_DIR))
    sz = filesize(joinpath(DERIVED_DIR, f))
    @printf("  %-40s  %s\n", f, _human_size(sz))
end
println()
flush(stdout)

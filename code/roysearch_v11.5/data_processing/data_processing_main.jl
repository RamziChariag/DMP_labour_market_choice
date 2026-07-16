############################################################
# code/data_processing/data_processing_main.jl
#   Data-processing pipeline — entry point
#
# Usage (from the project root):
#   julia --threads auto code/data_processing/data_processing_main.jl
#
# Loads every raw dataset, cleans it, and writes all artefacts the SMM
# step later reads from data/derived/. Run this once before any SMM
# estimation.
#
# Project layout:
#   code/
#     data_processing/
#       data_processing_main.jl   — this driver (paths, using, includes, run)
#       setup.jl                  — windows, MOMENT_NAMES, helpers, _load_arrow
#       cps_basic.jl   cps_asec.jl — CPS Basic / ASEC cleaning
#       jolts.jl       j2j.jl      — JOLTS openings / J2J EE rates
#       nsc.jl                     — NSC enrolment: κ_w level adj. + φ calibration
#       transitions.jl             — worker-flow hazards + ν life-table turnover
#       sipp.jl                    — SIPP within-job wage-change + EE mobility
#       moments.jl     sigma.jl    — 31 moment targets + Σ̂, per window
#       validation.jl              — end-of-run diagnostics
#     smm/                         — separate step; consumes data/derived/
#   data/
#     raw/      cps_basic/  cps_asec/  jolts/  j2j/  nsc/
#     derived/  windows.json, moments_{w}.csv, sigma_{w}.csv,
#               moment_scales_{w}.csv, nu_estimation.csv,
#               phi_calibration.csv, training_share_scale.csv,
#               j2j_ee_rates.csv, sipp_wchg_rates.csv, sipp_ee_rates.csv, *_clean.arrow, ...
#
# Pipeline order (each stage's output feeds the later ones):
#   0  windows.json        write_windows_json()
#   1  CPS Basic           clean_cps_basic()          → cps_basic_clean.arrow (+ skill shares)
#   2  CPS ASEC            clean_cps_asec()           → cps_asec_clean.arrow
#   3  NSC enrolment / κ   enrollment_rate_by_age();  compute_cps_nsc_scale()
#                                                     → training_share_scale.csv
#   4  JOLTS               clean_jolts()              → jolts_clean.arrow
#   5  Transitions         make_transitions()         → transitions_monthly.arrow
#   6  J2J EE rates        import_j2j_ee_rates()      → j2j_ee_rates.csv
#   6b SIPP wchg + EE       make_sipp_wchg()           → sipp_wchg_rates.csv, sipp_ee_rates.csv
#   7  ν turnover          compute_nu()               → nu_estimation.csv
#   8  φ completion        calibrate_phi()            → phi_calibration.csv
#   9  Moments             make_moments()             → moments_{w}.csv      (training_share × κ_w)
#  10  Σ̂ matrix           compute_influence_..._full() → sigma_{w}.csv       (ts row/col × κ_w)
#  11  Validation          run_validation(...)        → diagnostics (prints only)
#
# NSC κ_w convention. The per-window NSC level adjustment for
# training_share is applied HERE, when moments_{w}.csv and sigma_{w}.csv
# are written (Stages 9–10). code/smm/moments.jl reads the pre-adjusted
# values directly.
############################################################

println("="^60)
println("  Segmented Search Model — Data Processing Pipeline")
println("="^60)
flush(stdout)

# ── Paths ──────────────────────────────────────────────────────
const PIPELINE_DIR      = @__DIR__
const PROJECT_ROOT      = joinpath(PIPELINE_DIR, "..", "..")
const DATA_DIR          = joinpath(PROJECT_ROOT, "data")
const RAW_DIR           = joinpath(DATA_DIR, "raw")
const DERIVED_DIR       = joinpath(DATA_DIR, "derived")

const RAW_CPS_BASIC_DIR = joinpath(RAW_DIR, "cps_basic")
const RAW_CPS_ASEC_DIR  = joinpath(RAW_DIR, "cps_asec")
const RAW_JOLTS_DIR     = joinpath(RAW_DIR, "jolts")
const RAW_J2J_DIR       = joinpath(RAW_DIR, "j2j")
const RAW_NSC_DIR       = joinpath(RAW_DIR, "nsc")

mkpath(DERIVED_DIR)
println("PROJECT_ROOT = ", PROJECT_ROOT)
println("DERIVED_DIR  = ", DERIVED_DIR)

# ── Packages ───────────────────────────────────────────────────
print("Loading packages... "); flush(stdout)

using DataFrames
using CSV
using Arrow
using Statistics
using Printf
using Dates                # SIPP day-count-neutral weekly wage (daysinmonth)
using HTTP, JSON3          # download_jolts hits the BLS API
using XLSX                 # NSC Excel workbook
using LinearAlgebra        # influence functions / Σ̂

println("done."); flush(stdout)

# ── Pipeline modules ───────────────────────────────────────────
print("Loading pipeline modules... "); flush(stdout)

include(joinpath(PIPELINE_DIR, "setup.jl"))        # constants, helpers, _load_arrow, write_windows_json
include(joinpath(PIPELINE_DIR, "cps_basic.jl"))    # clean_cps_basic
include(joinpath(PIPELINE_DIR, "cps_asec.jl"))     # clean_cps_asec
include(joinpath(PIPELINE_DIR, "jolts.jl"))        # download_jolts, clean_jolts
include(joinpath(PIPELINE_DIR, "j2j.jl"))          # import_j2j_ee_rates
include(joinpath(PIPELINE_DIR, "sipp.jl"))         # make_sipp_wchg (SIPP wchg_rate_U/S, ee_rate_S, ee_step_S)
include(joinpath(PIPELINE_DIR, "nsc.jl"))          # enrollment_rate_by_age, compute_cps_nsc_scale, _load_training_share_scale, calibrate_phi
include(joinpath(PIPELINE_DIR, "transitions.jl"))  # make_transitions, compute_nu
include(joinpath(PIPELINE_DIR, "moments.jl"))      # make_moments (+ stock/wage/tightness helpers)
include(joinpath(PIPELINE_DIR, "sigma.jl"))        # compute_influence_functions_and_sigma_full
include(joinpath(PIPELINE_DIR, "validation.jl"))   # run_validation

println("done."); flush(stdout)

# ── Stage banner helper ────────────────────────────────────────
function _stage_banner(title::AbstractString)
    println("\n" * "─"^70)
    println("▶ " * title)
    println("─"^70)
    flush(stdout)
end

# ============================================================
# Run the pipeline. Order matters: every stage below consumes
# artefacts written by an earlier one (see the header table).
# ============================================================

_stage_banner("Stage 0 — windows.json (single source of truth for WINDOWS)")
write_windows_json()

_stage_banner("Stage 1 — CPS Basic Monthly")
clean_cps_basic()

_stage_banner("Stage 2 — CPS ASEC")
clean_cps_asec()

_stage_banner("Stage 3 — NSC enrolment: diagnostic + per-window κ_w level adjustment")
enrollment_rate_by_age()
compute_cps_nsc_scale()

_stage_banner("Stage 4 — JOLTS job openings")
clean_jolts()

_stage_banner("Stage 5 — CPS transition hazards (job-finding / separation)")
make_transitions()

_stage_banner("Stage 6 — J2J employer-to-employer (EE) rates")
import_j2j_ee_rates()

_stage_banner("Stage 6b — SIPP within-job wage-change + EE mobility (wchg_rate_U/S, ee_rate_S, ee_step_S)")
make_sipp_wchg()

_stage_banner("Stage 7 — demographic turnover ν (life-table)")
compute_nu()

_stage_banner("Stage 8 — training-completion rate φ (NSC/IPEDS)")
calibrate_phi()

_stage_banner("Stage 9 — moment targets (31 moments × 4 windows; training_share × κ_w)")
all_moments = make_moments()

_stage_banner("Stage 10 — influence functions and Σ̂ (31×31 per window; ts row/col × κ_w)")
all_sigma = compute_influence_functions_and_sigma_full()

_stage_banner("Stage 11 — validation and diagnostics")
run_validation(all_moments, all_sigma)

# ============================================================
# Summary of the derived artefacts the SMM step will load.
# ============================================================
println("\n" * "="^60)
println("Derived files in: $DERIVED_DIR")
for f in sort(readdir(DERIVED_DIR))
    sz = filesize(joinpath(DERIVED_DIR, f))
    @printf("  %-40s  %s\n", f, Base.format_bytes(sz))
end

println("\nKey outputs:")
println("  • windows.json                 — single source of truth for WINDOWS (4 entries)")
println("  • moments_{window}.csv         — 31 moments per window")
println("  • sigma_{window}.csv           — 31×31 variance-covariance matrix (full)")
println("  • moment_scales_{window}.csv   — scale factors used for IF normalisation")
println("  • j2j_ee_rates.csv             — J2J E4-only EE rates by window")
println("  • sipp_wchg_rates.csv          — SIPP within-job wage-change hazards by window (earnings-based + rate-based BBG diagnostic)")
println("  • sipp_ee_rates.csv            — SIPP skilled EE rate + EE-move wage step by window")
println("  • nu_estimation.csv            — ν on base_fc AND base_covid (life-table)")
println("  • phi_calibration.csv          — training completion rate φ (pooled)")
println("  • training_share_scale.csv     — per-window κ_w NSC level adjustment")

println("\nDone."); flush(stdout)

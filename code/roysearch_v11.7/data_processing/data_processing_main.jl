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
using Random               # SIPP BBG break-filter Monte-Carlo calibration (deterministic)
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

# ── Cross-source consistency test ──────────────────────────────
# After every moment, Σ̂, and σ̂²_samp file is written, re-read the FINAL derived
# products and check the cross-source identities the theorist confirmed BIND in
# this model. Each line prints PASS/FAIL with the numeric gap. A FAIL is loud but
# NEVER crashes the pipeline (the user runs this and sends the log); the check is
# wrapped so a genuinely structural break is reported, not swallowed.
#
# The non-binding relations (flow-balance ur = sep/(sep+jfr); wchg-vs-sep) are
# deliberately NOT tested — they do not hold as identities in this specification.
function run_consistency_test(windows_order)
    _stage_banner("Stage 12 — cross-source consistency test (identities that BIND)")
    npass = Ref(0); nfail = Ref(0)
    check(ok, id, lhs, rhs, gap) = begin
        (ok ? npass : nfail)[] += 1
        @printf("  %-4s %-42s %12.6g vs %-12.6g (gap %.3g)\n",
                ok ? "PASS" : "FAIL", id, lhs, rhs, gap)
    end

    # Read a window's final moments_{w}.csv into a moment→value dict.
    read_moments(w) = begin
        f = joinpath(DERIVED_DIR, "moments_$(w).csv")
        isfile(f) || return Dict{String,Float64}()
        df = CSV.read(f, DataFrame)
        Dict(String(r.moment) => Float64(r.value) for r in eachrow(df))
    end
    g(m, k) = get(m, k, NaN)

    # SIPP wchg brackets for the BBG range gate (identity 5) come straight from
    # sipp_wchg_rates.csv: the raw-earnings FC value and the shipped COVID value.
    sipp = isfile(joinpath(DERIVED_DIR, "sipp_wchg_rates.csv")) ?
           CSV.read(joinpath(DERIVED_DIR, "sipp_wchg_rates.csv"), DataFrame) : nothing
    sipp_row(w) = (sipp === nothing) ? nothing :
                  (i = findfirst(==(string(w)), string.(sipp.window)); isnothing(i) ? nothing : sipp[i, :])

    fc_windows    = (:base_fc, :crisis_fc)
    covid_windows = (:base_covid, :crisis_covid)
    redesign_wchg = Float64[]
    for w in covid_windows
        r = sipp_row(w)
        r === nothing && continue
        for c in (:wchg_rate_U, :wchg_rate_S)
            v = Float64(r[c]); isfinite(v) && push!(redesign_wchg, v)
        end
    end
    mean_redesign = isempty(redesign_wchg) ? NaN : sum(redesign_wchg) / length(redesign_wchg)

    for w in windows_order
        m = read_moments(w)
        isempty(m) && (@printf("  (no moments_%s.csv — skipped)\n", w); continue)
        println("  ── $w ──")

        # (1) wage_premium == mean_wage_S − mean_wage_U (same ASEC log universe)
        wp = g(m, "wage_premium"); diff = g(m, "mean_wage_S") - g(m, "mean_wage_U")
        gap1 = abs(wp - diff)
        check(!isnan(gap1) && gap1 <= 1e-4, "(1) wage_premium = mean_S − mean_U", wp, diff, gap1)

        # (2) quantile monotonicity p25 ≤ p50 ≤ p75, each market
        for mk in ("U", "S")
            q25 = g(m, "p25_wage_$mk"); q50 = g(m, "p50_wage_$mk"); q75 = g(m, "p75_wage_$mk")
            ok = !any(isnan, (q25, q50, q75)) && q25 <= q50 <= q75
            worst = ok ? 0.0 : max(q25 - q50, q50 - q75, 0.0)
            check(ok, "(2) quantile monotonicity $mk (p25≤p50≤p75)", q50, q75, worst)
        end

        # (3) ur_total ≈ LF-share-weighted avg of ur_U, ur_S (skilled_share as S weight)
        urt = g(m, "ur_total"); urU = g(m, "ur_U"); urS = g(m, "ur_S"); ss = g(m, "skilled_share")
        if !any(isnan, (urt, urU, urS, ss))
            approx = (1 - ss) * urU + ss * urS
            check(abs(urt - approx) <= 2e-3, "(3) ur_total = Σ share·ur_j", urt, approx, abs(urt - approx))
        end

        # (4) every hazard finite and in (0,1) — all on the monthly-hazard footing
        for hz in ("jfr_U","jfr_S","sep_rate_U","sep_rate_S","ee_rate_S","wchg_rate_U","wchg_rate_S")
            v = g(m, hz)
            check(isfinite(v) && 0.0 < v < 1.0, "(4) hazard $hz ∈ (0,1)", v, NaN, isfinite(v) ? 0.0 : NaN)
        end

        # (5) BBG range gate (FC windows): shipped BBG wchg must sit in the same
        #     order of magnitude as the redesign raw wchg — 0 < bbg < raw-earnings
        #     FC AND within [0.1×, 3×] of the mean redesign wchg. FAILs loud if BBG
        #     left it huge (filter didn't fire) or crushed it to ~0 (over-corrected).
        if w in fc_windows
            r = sipp_row(w)
            if r !== nothing
                for mk in ("U", "S")
                    bbg = Float64(r[Symbol("wchg_rate_$mk")])
                    raw = Float64(r[Symbol("wchg_rate_$(mk)_raw")])
                    lo  = 0.1 * mean_redesign; hi = 3.0 * mean_redesign
                    ok  = isfinite(bbg) && isfinite(raw) && bbg > 0.0 && bbg < raw &&
                          isfinite(mean_redesign) && lo <= bbg <= hi
                    check(ok, "(5) BBG gate $mk: 0<bbg<raw & in[0.1,3]×redesign", bbg, raw,
                          isfinite(mean_redesign) ? bbg / mean_redesign : NaN)
                end
            end
        end

        # (6) ee_rate_S + sep_rate_S < 1; bounded shares ∈ (0,1); θ/jfr co-move
        ee = g(m, "ee_rate_S"); ss_ = g(m, "sep_rate_S")
        if !any(isnan, (ee, ss_))
            check(ee + ss_ < 1.0, "(6a) ee_rate_S + sep_rate_S < 1", ee + ss_, 1.0, ee + ss_ - 1.0)
        end
        for sh in ("skilled_share","training_share","overlap_UgtS","overlap_SltU")
            v = g(m, sh); isnan(v) && continue
            check(0.0 < v < 1.0, "(6b) share $sh ∈ (0,1)", v, NaN, 0.0)
        end
        thU = g(m, "theta_U"); thS = g(m, "theta_S"); jfrU = g(m, "jfr_U"); jfrS = g(m, "jfr_S")
        if !any(isnan, (thU, thS, jfrU, jfrS)) && thS != thU
            comove = (thS > thU) == (jfrS > jfrU)
            check(comove, "(6c) θ/jfr ordering co-moves (θ_S>θ_U ⇒ jfr_S>jfr_U)", thS - thU, jfrS - jfrU, 0.0)
        end

        # σ̂²_samp footing sanity: max/min ratio across moments (5b diagnostic).
        svf = joinpath(DERIVED_DIR, "sampling_var_$(w).csv")
        if isfile(svf)
            sv = CSV.read(svf, DataFrame)
            pos = filter(v -> isfinite(v) && v > 0.0, Float64.(sv.sampling_var))
            if !isempty(pos)
                @printf("  INFO σ̂²_samp footing max/min ratio = %.4g (%d finite moments)\n",
                        maximum(pos) / minimum(pos), length(pos))
            end
        end
    end

    @printf("\n  Consistency test: %d PASS, %d FAIL.\n", npass[], nfail[])
    nfail[] > 0 && @warn "  Consistency test reported $(nfail[]) FAIL line(s) — inspect the log above."
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

# Stage 12 runs last so it reads the FINAL written products. Guarded: a bug in
# the test itself must not lose the derived files the user just built.
try
    run_consistency_test(WINDOWS_ORDER)
catch e
    @error "Consistency test raised (pipeline outputs are intact); inspect:" exception=(e, catch_backtrace())
end

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
println("  • sampling_var_{window}.csv    — per-moment sampling variance σ̂²_samp (diagonal-σ weight)")
println("  • j2j_ee_rates.csv             — J2J E4-only EE rates by window")
println("  • sipp_wchg_rates.csv          — SIPP within-job wage-change hazards by window (shipped: BBG-classic on FC + raw-earnings on COVID; raw-earnings reporting cols)")
println("  • sipp_ee_rates.csv            — SIPP skilled EE rate + EE-move wage step by window")
println("  • nu_estimation.csv            — ν on base_fc AND base_covid (life-table)")
println("  • phi_calibration.csv          — training completion rate φ (pooled)")
println("  • training_share_scale.csv     — per-window κ_w NSC level adjustment")

println("\nDone."); flush(stdout)

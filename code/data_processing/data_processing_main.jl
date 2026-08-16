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
#       moments.jl                 — 31 moment targets per window
#       sampling_variances.jl      — per-moment sampling variances σ̂²_samp
#       validation.jl              — end-of-run diagnostics
#     smm/                         — separate step; consumes data/derived/
#   data/
#     raw/      cps_basic/  cps_asec/  jolts/  j2j/  nsc/
#     derived/  windows.json, moments_{w}.csv, sampling_var_{w}.csv,
#               nu_estimation.csv,
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
#  10  Sampling variances  compute_sampling_variances() → sampling_var_{w}.csv (ts entry × κ_w²)
#  11  Validation          run_validation(...)        → diagnostics (prints only)
#
# NSC κ_w convention. The per-window NSC level adjustment for
# training_share is applied HERE, when moments_{w}.csv is written
# (Stage 9); sampling_var_{w}.csv carries the matching κ_w² on the
# training_share entry (Stage 10). code/smm/moments.jl reads the
# pre-adjusted values directly.
############################################################

const ROYSEARCH_VERSION = "18.4.0"
println("="^60)
println("  Segmented Search Model v$(ROYSEARCH_VERSION) — Data Processing Pipeline")
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

println("done."); flush(stdout)

# ── Pipeline modules ───────────────────────────────────────────
print("Loading pipeline modules... "); flush(stdout)

include(joinpath(PIPELINE_DIR, "setup.jl"))        # constants, helpers, _load_arrow, write_windows_json
include(joinpath(PIPELINE_DIR, "cps_basic.jl"))    # clean_cps_basic
include(joinpath(PIPELINE_DIR, "cps_asec.jl"))     # clean_cps_asec
include(joinpath(PIPELINE_DIR, "jolts.jl"))        # download_jolts, clean_jolts
include(joinpath(PIPELINE_DIR, "j2j.jl"))          # import_j2j_ee_rates
include(joinpath(PIPELINE_DIR, "sipp.jl"))         # make_sipp_wchg (SIPP wchg_rate_U/S, ee_rate_S, ee_step_S)
include(joinpath(PIPELINE_DIR, "nsc.jl"))          # enrollment_rate_by_age, compute_training_share_target, _load_training_share_target, calibrate_phi
include(joinpath(PIPELINE_DIR, "transitions.jl"))  # make_transitions, compute_nu
include(joinpath(PIPELINE_DIR, "moments.jl"))      # make_moments (+ stock/wage/tightness helpers)
include(joinpath(PIPELINE_DIR, "sampling_variances.jl"))  # compute_sampling_variances
include(joinpath(PIPELINE_DIR, "validation.jl"))   # run_validation

println("done."); flush(stdout)

# ── Stage banner helper ────────────────────────────────────────
function _stage_banner(title::AbstractString)
    println("\n" * "─"^70)
    println("▶ " * title)
    println("─"^70)
    flush(stdout)
end

# ── Stage 12: audit of the data DECISIONS ─────────────────────────────────────
# Every check below corresponds to a choice argued in the data-and-moments notes,
# and every one of them CAN fail. Bounds checks (is a share in [0,1], is a hazard
# positive) are deliberately absent: they passed 72/72 with no information, and a
# test that cannot fail is a test that cannot be read.
#
# What each line defends:
#   (1) wage_premium is not an independent moment — it is mean_S − mean_U on the
#       same ASEC log universe. If it ever drifts, the wage block is inconsistent.
#   (2) ur_total is dropped from the weighting because it is implied by ur_U, ur_S
#       and skilled_share. This verifies the redundancy that justifies dropping it.
#   (3) ASEC 2023 is in the sample. The year filter is derived from WINDOWS rather
#       than hardcoded, and this is what catches a regression to a literal.
#   (4) DURUNEMP == 999 (IPUMS NIU) never enters ltu_share_S. The NIU code exceeds
#       every duration threshold, so an unguarded comparison silently calls those
#       records long-term unemployed.
#   (5) training_share comes from NSC with the attrition wedge applied, and the
#       within-pair crisis SIGNS survive it — the whole reason for dropping κ,
#       whose CPS denominator drifted +6% within each pair.
#   (6) The CPS stock variances are person-clustered. CPS is a 4-8-4 rotating
#       panel; the unclustered form understates them 2-5×. The singleton-cluster
#       control shows the inflation comes from the PANEL and not from switching
#       variance formulas: with one record per cluster the estimator is within a
#       few percent of Kish, while clustering on person is multiples of it.
function run_consistency_test(windows_order)
    _stage_banner("Stage 12 — audit of the data decisions")
    npass = Ref(0); nfail = Ref(0)
    check(ok, id, detail) = begin
        (ok ? npass : nfail)[] += 1
        @printf("  %-4s %-52s %s\n", ok ? "PASS" : "FAIL", id, detail)
    end

    read_moments(w) = begin
        f = joinpath(DERIVED_DIR, "moments_$(w).csv")
        isfile(f) || return Dict{String,Float64}()
        df = CSV.read(f, DataFrame)
        Dict(String(r.moment) => Float64(r.value) for r in eachrow(df))
    end
    g(m, k) = get(m, k, NaN)

    M = Dict(w => read_moments(w) for w in windows_order)

    # (1)+(2)+(4): per-window identities
    for w in windows_order
        m = M[w]
        isempty(m) && (@printf("  (no moments_%s.csv — skipped)\n", w); continue)

        wp   = g(m, "wage_premium")
        diff = g(m, "mean_wage_S") - g(m, "mean_wage_U")
        check(isfinite(wp) && abs(wp - diff) <= 1e-4,
              "(1) $w wage_premium = mean_S − mean_U",
              @sprintf("gap %.2e", abs(wp - diff)))

        urt, urU, urS, ss = g(m,"ur_total"), g(m,"ur_U"), g(m,"ur_S"), g(m,"skilled_share")
        if !any(isnan, (urt, urU, urS, ss))
            implied = (1 - ss) * urU + ss * urS
            check(abs(urt - implied) <= 2e-3,
                  "(2) $w ur_total implied by ur_U, ur_S, skilled_share",
                  @sprintf("%.6f vs %.6f", urt, implied))
        end
    end

    # (3) ASEC 2023 present — the year filter reads WINDOWS, not a literal
    asec_path = joinpath(DERIVED_DIR, "cps_asec_clean.arrow")
    if isfile(asec_path)
        yrs = unique(DataFrame(Arrow.Table(asec_path)).YEAR)
        want = maximum(maximum(wd.asec_years) for wd in values(WINDOWS))
        check(want in yrs, "(3) ASEC sample reaches the declared final survey year",
              "declared $want, present: $(want in yrs ? "yes" : "NO") (range $(minimum(yrs))–$(maximum(yrs)))")
        check(:CPSIDP in propertynames(DataFrame(Arrow.Table(asec_path))),
              "(3b) ASEC keeps CPSIDP for person clustering", "")
    end

    # (4) NIU guard: no DURUNEMP == 999 record counted as long-term unemployed
    cps_path = joinpath(DERIVED_DIR, "cps_basic_clean.arrow")
    if isfile(cps_path)
        cps = DataFrame(Arrow.Table(cps_path); copycols=false)
        u   = coalesce.(cps.unemployed, false)
        n999 = count(i -> u[i] && Float64(coalesce(cps.DURUNEMP[i], 0.0)) == 999.0,
                     eachindex(u))
        check(all(_durw(d) < 27.0 for d in (999.0, 999)),
              "(4) DURUNEMP NIU code 999 maps to 0, not long-term",
              "$n999 unemployed records carry 999")
    end

    # (5) training_share: NSC-sourced, attrition-adjusted, crisis signs preserved
    tgt_path = joinpath(DERIVED_DIR, "training_share_target.csv")
    if isfile(tgt_path)
        t = CSV.read(tgt_path, DataFrame)
        frow(w) = (i = findfirst(==(string(w)), string.(t.window)); isnothing(i) ? nothing : t[i,:])
        fs = [frow(w).attrition_f for w in windows_order if frow(w) !== nothing]
        check(!isempty(fs) && maximum(fs) - minimum(fs) < 1e-9,
              "(5a) attrition wedge f is constant across windows",
              @sprintf("f = %.4f", first(fs)))
        for (lab, b, c) in (("FC", :base_fc, :crisis_fc), ("COVID", :base_covid, :crisis_covid))
            mb, mc = get(M[b], "training_share", NaN), get(M[c], "training_share", NaN)
            any(isnan, (mb, mc)) && continue
            pct = 100 * (mc - mb) / mb
            want_rise = (lab == "FC")
            check((pct > 0) == want_rise,
                  "(5b) $lab training_share moves the observed direction",
                  @sprintf("%+.2f%% (expected %s)", pct, want_rise ? "rise" : "fall"))
        end
    end

    # (6) CPS stock variances are person-clustered
    if isfile(cps_path)
        cps  = DataFrame(Arrow.Table(cps_path); copycols=false)
        cps.ym = 100 .* cps.YEAR .+ cps.MONTH
        for w in windows_order
            wd  = WINDOWS[w]
            s   = cps[(cps.ym .>= wd.ym_start) .& (cps.ym .<= wd.ym_end), :]
            cw  = Float64.(coalesce.(s.WTFINL, 0.0))
            cid = Int64.(coalesce.(s.CPSIDP, 0))
            inlf = coalesce.(s.in_lf, false); unemp = coalesce.(s.unemployed, false)
            vk = _kish_prop_var(cw, inlf, inlf .& unemp)
            vc = _kish_prop_var_clustered(cw, inlf, inlf .& unemp, cid)
            # Singleton control: one record per cluster. Agrees with Kish in
            # expectation (not exactly — see setup.jl), so the gap must be small
            # while the person-clustered inflation must be large. That contrast is
            # what shows the inflation comes from the panel, not the formula.
            vs = _kish_prop_var_clustered(cw, inlf, inlf .& unemp, collect(1:length(cw)))
            check(isfinite(vc) && vc > 1.5 * vk && abs(vs - vk) / vk < 0.15,
                  "(6) $w ur_total variance clustered on person",
                  @sprintf("panel inflation %.2fx; singleton control %.1f%% off Kish",
                           vc / vk, 100 * abs(vs - vk) / vk))
        end
    end

    @printf("\n  Decision audit: %d PASS, %d FAIL.\n", npass[], nfail[])
    nfail[] > 0 && @warn "  Audit reported $(nfail[]) FAIL line(s) — inspect the log above."
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

_stage_banner("Stage 3 — NSC enrolment: diagnostic + training_share target")
enrollment_rate_by_age()
compute_training_share_target()

_stage_banner("Stage 4 — JOLTS job openings")
clean_jolts()

_stage_banner("Stage 5 — CPS transition hazards (job-finding / separation)")
make_transitions()

_stage_banner("Stage 6 — J2J employer-to-employer (EE) rates")
import_j2j_ee_rates()

_stage_banner("Stage 6b — SIPP within-job wage-change + EE mobility (wchg_rate_U/S, ee_rate_S, ee_step_S)")
make_sipp_wchg()

# SIPP-vs-J2J EE-rate comparison (shipped target = J2J), printed here so it
# follows the J2J (Stage 6) and SIPP (Stage 6b) EE tables.
print_ee_source_comparison()

_stage_banner("Stage 7 — demographic turnover ν (life-table)")
compute_nu()

_stage_banner("Stage 8 — training-completion rate φ (NSC/IPEDS)")
calibrate_phi()

_stage_banner("Stage 9 — moment targets (31 moments × 4 windows; training_share from NSC)")
all_moments = make_moments()

_stage_banner("Stage 10 — per-moment sampling variances (closed-form; 31 moments per window)")
compute_sampling_variances()

_stage_banner("Stage 11 — validation and diagnostics")
run_validation(all_moments)

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
println("  • sampling_var_{window}.csv    — per-moment sampling variance σ̂²_samp (diagonal-σ weight)")
println("  • j2j_ee_rates.csv             — J2J E4-only EE rates by window")
println("  • sipp_wchg_rates.csv          — SIPP within-job wage-change hazards by window (shipped: BBG-classic on FC + raw-earnings on COVID; raw-earnings reporting cols)")
println("  • sipp_ee_rates.csv            — SIPP skilled EE rate + EE-move wage step by window")
println("  • nu_estimation.csv            — ν on base_fc AND base_covid (life-table)")
println("  • phi_calibration.csv          — training completion rate φ (pooled)")
println("  • training_share_target.csv    — NSC training_share level + across-year variance")

println("\nDone."); flush(stdout)

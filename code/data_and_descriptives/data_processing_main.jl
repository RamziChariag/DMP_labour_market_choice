############################################################
# code/data_and_descriptives/data_processing_main.jl
#
# Data pipeline entry point. Cleans every raw dataset and writes the products
# the estimation reads from data/derived/: windows.json, moments_{w}.csv,
# sampling_var_{w}.csv, nu_estimation.csv, phi_calibration.csv,
# training_share_target.csv, plus intermediate *_clean.arrow and source CSVs.
#
# Usage (from the project root):
#   julia --threads auto code/data_and_descriptives/data_processing_main.jl
#
# Stages, in order (each consumes files written by earlier ones):
#   0  windows.json          write_windows_json()               setup.jl
#   1  CPS Basic             clean_cps_basic()                  cps_basic.jl
#   2  CPS ASEC              clean_cps_asec()                   cps_asec.jl
#   3  enrolment             enrollment_rate_by_age(),
#                            compute_training_share_target()    nsc.jl
#   4  JOLTS                 clean_jolts()                      jolts.jl
#   5  CPS flows             make_transitions()                 transitions.jl
#   6  J2J EE rate           import_j2j_ee_rates()              j2j.jl
#   6b SIPP wchg + EE        make_sipp_wchg()                   sipp.jl
#   7  ν                     compute_nu()                       transitions.jl
#   8  φ                     calibrate_phi()                    nsc.jl
#   9  moments               make_moments()                     moments.jl
#  10  sampling variances    compute_sampling_variances()       sampling_variances.jl
#  11  validation            run_validation()                   validation.jl
#  12  decision audit        run_consistency_test()             this file
# The stage numbers printed inside the stage functions predate this ordering;
# this table is the reference.
############################################################

# The version lives once, in code/smm/version.jl.
include(joinpath(@__DIR__, "..", "smm", "version.jl"))

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
const RAW_NCES_DIR      = joinpath(RAW_DIR, "nces")

mkpath(DERIVED_DIR)
println("PROJECT_ROOT = ", PROJECT_ROOT)
println("DERIVED_DIR  = ", DERIVED_DIR)

# ── Packages ───────────────────────────────────────────────────
print("Loading packages... "); flush(stdout)

using DataFrames
using CSV
using Arrow
using Statistics
using Random               # seeded BBG Monte-Carlo calibration (sipp.jl)
using Printf
using Dates                # daysinmonth (sipp.jl)
using HTTP, JSON3          # BLS API (jolts.jl), windows.json

println("done."); flush(stdout)

# ── Pipeline modules ───────────────────────────────────────────
print("Loading pipeline modules... "); flush(stdout)

include(joinpath(PIPELINE_DIR, "setup.jl"))
include(joinpath(PIPELINE_DIR, "cps_basic.jl"))
include(joinpath(PIPELINE_DIR, "cps_asec.jl"))
include(joinpath(PIPELINE_DIR, "jolts.jl"))
include(joinpath(PIPELINE_DIR, "j2j.jl"))
include(joinpath(PIPELINE_DIR, "sipp.jl"))
include(joinpath(PIPELINE_DIR, "nsc.jl"))
include(joinpath(PIPELINE_DIR, "transitions.jl"))
include(joinpath(PIPELINE_DIR, "moments.jl"))
include(joinpath(PIPELINE_DIR, "sampling_variances.jl"))
include(joinpath(PIPELINE_DIR, "validation.jl"))

println("done."); flush(stdout)

"""
    _stage_banner(title)

Print a stage header line.
"""
function _stage_banner(title::AbstractString)
    println("\n" * "─"^70)
    println("▶ " * title)
    println("─"^70)
    flush(stdout)
end

"""
    run_consistency_test(windows_order)

Audit of the data decisions on the written products; prints PASS/FAIL per check
and warns if any fail. Each check guards one deliberate choice:
(1) wage_premium = mean_S − mean_U on the same ASEC sample;
(2) ur_total is implied by ur_U, ur_S and skilled_share (why it is redundant);
(3) the ASEC sample reaches the last declared survey year (filter from WINDOWS),
    and keeps CPSIDP;
(4) the DURUNEMP NIU code 999 maps to duration 0 (`_durw`);
(5) training_share has complete Fall-year coverage, rises across the FC pair
    (> 5 %) and is flat across the COVID pair (within ±2 %, a magnitude
    statement: the stationary identity would need about +8.6 %);
(6) CPS stock variances are person-clustered: clustered > 1.5× Kish, while the
    singleton-cluster control stays within 15 % of Kish, so the inflation comes
    from the panel and not from the formula.
Bounds checks are deliberately absent.
"""
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

    # (1), (2): per-window identities
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

    # (3) ASEC final survey year present, CPSIDP kept
    asec_path = joinpath(DERIVED_DIR, "cps_asec_clean.arrow")
    if isfile(asec_path)
        yrs = unique(DataFrame(Arrow.Table(asec_path)).YEAR)
        want = maximum(maximum(wd.asec_years) for wd in values(WINDOWS))
        check(want in yrs, "(3) ASEC sample reaches the declared final survey year",
              "declared $want, present: $(want in yrs ? "yes" : "NO") (range $(minimum(yrs))–$(maximum(yrs)))")
        check(:CPSIDP in propertynames(DataFrame(Arrow.Table(asec_path))),
              "(3b) ASEC keeps CPSIDP for person clustering", "")
    end

    # (4) NIU guard
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

    # (5) training_share coverage and direction across each pair
    tgt_path = joinpath(DERIVED_DIR, "training_share_target.csv")
    if isfile(tgt_path)
        t = CSV.read(tgt_path, DataFrame)
        frow(w) = (i = findfirst(==(string(w)), string.(t.window)); isnothing(i) ? nothing : t[i,:])
        spans = [(w, frow(w).n_fall, (WINDOWS[w].ym_end ÷ 100) - (WINDOWS[w].ym_start ÷ 100) + 1)
                 for w in windows_order if frow(w) !== nothing]
        check(!isempty(spans) && all(n == s for (_, n, s) in spans),
              "(5a) every window's Fall-year coverage is complete",
              join(["$w: $n/$s" for (w, n, s) in spans], ", "))
        for (lab, b, c, want) in (("FC", :base_fc, :crisis_fc, :rise),
                                  ("COVID", :base_covid, :crisis_covid, :flat))
            mb, mc = get(M[b], "training_share", NaN), get(M[c], "training_share", NaN)
            any(isnan, (mb, mc)) && continue
            pct = 100 * (mc - mb) / mb
            ok  = want === :rise ? pct > 5.0 : abs(pct) < 2.0
            check(ok, "(5b) $lab training_share moves as the source series does",
                  @sprintf("%+.2f%% (expected %s)", pct, want === :rise ? "rise >5%" : "flat <2%"))
        end
    end

    # (6) CPS stock variances person-clustered, with the singleton control
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
            vs = _kish_prop_var_clustered(cw, inlf, inlf .& unemp, collect(eachindex(cw)))
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
# Run the pipeline (order as in the header table).
# ============================================================

_stage_banner("Stage 0 — windows.json (single source of truth for WINDOWS)")
write_windows_json()

_stage_banner("Stage 1 — CPS Basic Monthly")
clean_cps_basic()

_stage_banner("Stage 2 — CPS ASEC")
clean_cps_asec()

_stage_banner("Stage 3 — enrolment: age diagnostic + training_share target")
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

print_ee_source_comparison()

_stage_banner("Stage 7 — demographic turnover ν (life-table)")
compute_nu()

_stage_banner("Stage 8 — training-completion rate φ")
calibrate_phi()

_stage_banner("Stage 9 — moment targets (35 moments × 4 windows)")
all_moments = make_moments()

_stage_banner("Stage 10 — per-moment sampling variances (closed-form; 35 moments per window)")
compute_sampling_variances()

_stage_banner("Stage 11 — validation and diagnostics")
run_validation(all_moments)

# Last, on the final products; a failure in the audit itself must not cost the
# derived files just built.
try
    run_consistency_test(WINDOWS_ORDER)
catch e
    @error "Consistency test raised (pipeline outputs are intact); inspect:" exception=(e, catch_backtrace())
end

# ============================================================
# Summary of the derived files
# ============================================================
println("\n" * "="^60)
println("Derived files in: $DERIVED_DIR")
for f in sort(readdir(DERIVED_DIR))
    sz = filesize(joinpath(DERIVED_DIR, f))
    @printf("  %-40s  %s\n", f, Base.format_bytes(sz))
end

println("\nKey outputs:")
println("  • windows.json                 — single source of truth for WINDOWS (4 entries)")
println("  • moments_{window}.csv         — 35 moments per window")
println("  • sampling_var_{window}.csv    — per-moment sampling variance σ̂²_samp (diagonal-σ weight)")
println("  • j2j_ee_rates.csv             — J2J E4-only EE rates by window")
println("  • sipp_wchg_rates.csv          — SIPP within-job wage-change hazards by window (shipped: BBG-classic on FC + raw-earnings on COVID; raw-earnings reporting cols)")
println("  • sipp_ee_rates.csv            — SIPP skilled EE rate + EE-move wage step by window")
println("  • nu_estimation.csv            — ν on base_fc AND base_covid (life-table)")
println("  • phi_calibration.csv          — training completion rate φ")
println("  • training_share_target.csv    — training_share level + across-year variance")

println("\nDone."); flush(stdout)

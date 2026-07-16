############################################################
# data_processing/sipp.jl
#
# Stage 6b — SIPP job-mobility and wage moments among the skilled/unskilled.
# One recursive walk over data/raw/sipp/ produces, per estimation window:
#   • wchg_rate_U / wchg_rate_S  within-job wage-change hazard (both markets).
#     A λ_j redraw landing at/above the reservation cutoff survives the match
#     but re-prices it (rate λ_j·[1−G(p*_j)]); ξ_j only ever destroys a match.
#     So wchg_rate loads on λ blind to ξ, and sep_rate then pins ξ as the
#     residual (data-and-moments §sipp, identification section). The wage-change
#     variable is constructed differently by panel era (see below): a BBG break
#     filter on the hourly-rate series for the classic (FC) windows, and the
#     raw earnings-based change for the redesign (COVID) windows.
#   • ee_rate_S  skilled employer-to-employer hazard (an OJS poach). The
#     unskilled market has no on-the-job search, so no EE events exist there —
#     these two moments are skilled-only.
#   • ee_step_S  mean LOG real-weekly-wage step of a skilled EE move, i.e.
#     E[log w_post − log w_pre] (identifies β_S: the proportional step is a
#     fraction of the surplus gain, LMR/CPVR). Log scale matches every other
#     wage moment, which lives on log weekly wages.
#
# WAGE-CHANGE CONSTRUCTION BY PANEL ERA:
#   • Classic FC windows (2001/2004/2008 panels → base_fc, crisis_fc): the
#     reported earnings series carries hours/overtime/rounding noise that
#     crosses any sensible threshold nearly every month. We instead run the
#     Barattieri–Basu–Gottschalk (2014) three-step break filter on the far
#     cleaner HOURLY-RATE series (EPAYHR==1, TPYRATE/100), detect genuine wage
#     steps with a sequential sup-F test, and correct the per-period detection
#     rate for test size and power. The result is a lower bound on the true
#     re-pricing rate (BBG over-filters transitory re-pricings); the raw
#     earnings-based classic number is the upper bound. Both are reported.
#   • Redesign COVID windows (2014-present → base_covid, crisis_covid): the raw
#     earnings-based within-job wage change (|Δ real weekly wage| > ε), the
#     construction the SMM side has always read.
# SIPP_BBG_WINDOWS is the single point of control for which windows take the BBG
# construction; the shipped wchg_rate_j column carries the BBG hazard there and
# the raw earnings hazard elsewhere.
#
# Two file formats feed one per-window measurement:
#   • Redesign (2014-present): comma-delimited pu<Y>.csv annual releases and
#     pu2014w<N>.csv per-wave files, EEDUC / EJB<n>_JOBID / TJB<n>_MSUM (7 job
#     slots), MONTHCODE the reference month within the reference year.
#   • Classic (2001/2004/2008): fixed-width l<YY>puw<N>.dat cores. Census ships
#     the core file for EVERY wave but the core DICTIONARY for wave 1 only
#     (l<YY>puw1d.txt); the other waves' folders carry only Topical-Module
#     dictionaries, which document a different file and lack the core job
#     variables. The core layout is constant across a panel's waves, so the
#     wave-1 core dictionary is resolved ONCE per panel and applied to every
#     wave. Variable NAMES are stable across panels but byte OFFSETS differ per
#     panel, so fields are located by name. EEDUCATE / EENO1-2 / TPMSUM1-2 (2
#     job slots), RHCALYR+RHCALMN the reference month.
#
# WINDOW ASSIGNMENT IS THE SINGLE POINT OF CONTROL. Each person-month pair is
# routed to a window by its OWN calendar month via assign_window (setup.jl),
# read straight from WINDOWS. No file-level reference year and no window years
# are hardcoded here: adding, moving or removing a window in setup.jl WINDOWS
# re-routes every record automatically. A pair is used iff its calendar month
# lands in some window AND its calendar year has a known CPI-U index; a year in
# a window but absent from SIPP_CPIU is a loud error (never a silent skip).
#
# Reads (walked recursively under data/raw/sipp/, so files sitting in per-year
# subfolders are found as well as those directly under sipp/):
#         pu*.csv                        redesign cores (flat annual pu<Y>.csv and
#                                        the 2014 panel's per-wave pu2014w<N>.csv)
#         l<YY>puw<N>.dat                classic cores (in the per-panel subfolders)
#         l<YY>puw1d.txt[.old2/.old.2]   classic wave-1 core dictionary, resolved
#                                        from the same directory as its panel's cores
# Compressed cores (.zip, .dat.gz) with no extracted sibling are skipped with a
# gunzip/unzip hint — the SIPP files ship extracted, so the reader needs no
# zip/gzip dependency.
# Writes: sipp_wchg_rates.csv  (window; shipped wchg_rate_U/S + neff_U/S; raw
#                               earnings wchg_rate_U/S_raw; hourly_share_U/S;
#                               bbg_underflow_U/S; and the BBG variance inputs
#                               bbg_pihat_U/S + bbg_corrfac_U/S consumed by
#                               sigma.jl — see the CSV contract at make_sipp_wchg)
#         sipp_ee_rates.csv    (window, ee_rate_S, ee_step_S, ee_step_sd,
#                               neff_ee, neff_step)
#
# PARALLELISM (bit-deterministic). The walk discovers many independent files;
# they are processed concurrently with Threads.@threads (JULIA_NUM_THREADS),
# but the arithmetic is made independent of thread scheduling so the output is
# identical to a serial run of the same logic:
#   • each file accumulates into its OWN private SippAcc (no shared mutable
#     state during the parallel phase — threads never touch the same object);
#   • after all files finish, the per-file accumulators are summed into one
#     master in a FIXED SORTED file order. Floating-point summation is
#     order-sensitive, so fixing the order makes the merged totals a
#     deterministic function of the inputs alone. Per-file info lines are
#     collected and printed in that same order.
# The BBG rate series are collected per file and concatenated by job id in the
# same fixed order (a classic job spans several wave files), so break detection
# runs on the full within-window series regardless of thread scheduling.
# The per-file readers STREAM row-by-row (CSV.Rows for redesign, eachline for
# classic) and flush per person, so peak memory is one person's records — safe
# to run many threads over multi-GB annual releases on a memory-tight machine.
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages (incl. Dates, CSV, Random, Base.Threads) and path consts come
# from data_processing_main.jl.
############################################################

# ── BBG (Barattieri–Basu–Gottschalk 2014) wage-change break filter ──────────
# A reported monthly wage-rate series is modelled as a step function (the true
# wage, constant within a spell) plus classical measurement error. A "wage
# change" is a genuine step, not every reported movement. We detect steps with a
# sequential sup-F (Bai–Perron) test, giving π̂ = detected changes / (T−1). BBG
# eq. (6) then corrects for test size/power:  π̃ = (π̂ − α)/(γ − α), α the size
# (0.05) and γ the power, both read from the same Monte-Carlo calibration used
# for the critical values.  All logic is base Julia (no external packages).

# Mean-shift sup-F at every interior split of y[lo:hi]; returns (maxF, argsplit).
function _supF(y::AbstractVector{Float64}, lo::Int, hi::Int, htrim::Int)
    n = hi - lo + 1
    n < 2*htrim + 2 && return (0.0, 0)
    ssr0 = sum(abs2, @view(y[lo:hi]) .- mean(@view y[lo:hi]))
    ssr0 <= 0 && return (0.0, 0)
    bestF = 0.0; bestk = 0
    @inbounds for k in (lo+htrim-1):(hi-htrim)
        n1 = k - lo + 1; n2 = hi - k
        m1 = mean(@view y[lo:k]); m2 = mean(@view y[(k+1):hi])
        ssr1 = 0.0
        for i in lo:k;    ssr1 += abs2(y[i]-m1); end
        for i in (k+1):hi; ssr1 += abs2(y[i]-m2); end
        # F for one break (2 vs 1 params): ((ssr0-ssr1)/1)/(ssr1/(n-2))
        F = ssr1 > 0 ? ((ssr0 - ssr1)) / (ssr1 / (n - 2)) : 0.0
        F > bestF && (bestF = F; bestk = k)
    end
    return (bestF, bestk)
end

# Sequential Bai–Perron: count mean-shift breaks in y at critical value `crit`.
function count_breaks(y::AbstractVector{Float64}; crit::Float64, htrim::Int=1)
    T = length(y); T < 2*htrim + 2 && return 0
    segs = [(1, T)]; nbreaks = 0
    while !isempty(segs)
        (lo, hi) = pop!(segs)
        F, k = _supF(y, lo, hi, htrim)
        if k > 0 && F > crit
            nbreaks += 1
            push!(segs, (lo, k)); push!(segs, (k+1, hi))
        end
    end
    return nbreaks
end

# Monte-Carlo calibration under H0 (no true break): constant series + AR(1)
# measurement error (BBG: autocorr ρ, signal/noise s2n). Returns the crit value
# (1−α quantile of the null max-F) AND the power γ against a single mid-series
# break of the given standardized size. One calibration per series length T.
function calibrate(T::Int; alpha::Float64=0.05, rho::Float64=0.482, s2n::Float64=2.64,
                   break_size_sd::Float64=1.0, nsim::Int=4000, htrim::Int=1, seed::Int=1234)
    rng = MersenneTwister(seed + T)
    sig_e = 1.0                          # error sd (scale-free; F is scale-invariant)
    # null distribution of max-F
    nullF = Vector{Float64}(undef, nsim)
    e = Vector{Float64}(undef, T)
    function fill_ar!(v)
        v[1] = randn(rng)
        @inbounds for t in 2:T; v[t] = rho*v[t-1] + sqrt(1-rho^2)*randn(rng); end
        v .*= sig_e
    end
    for s in 1:nsim
        fill_ar!(e); F,_ = _supF(e, 1, T, htrim); nullF[s] = F
    end
    crit = quantile(nullF, 1 - alpha)
    # power: plant one break of size break_size_sd·(signal sd) at T÷2
    sig_signal = sqrt(s2n) * sig_e
    hits = 0
    y = Vector{Float64}(undef, T)
    for s in 1:nsim
        fill_ar!(e)
        step = break_size_sd * sig_signal
        @inbounds for t in 1:T; y[t] = (t > T÷2 ? step : 0.0) + e[t]; end
        F,_ = _supF(y, 1, T, htrim)
        F > crit && (hits += 1)
    end
    gamma = hits / nsim
    return (crit=crit, gamma=gamma)
end

# Self-consistent per-period calibration: at a per-candidate-split critical value
# `crit`, measure the per-period FALSE-break rate α_eff (null: constant + AR error)
# and the per-period detection rate γ_eff (H1: per-period true-step prob π_plant,
# steps ~ signal sd). Both aggregated over worker-periods, so π̂, α_eff, γ_eff all
# share the per-period unit that BBG eq.(6) requires. Returns (α_eff, γ_eff).
function calibrate_perperiod(T::Int; crit::Float64, rho::Float64=0.482, s2n::Float64=2.64,
                             π_plant::Float64=0.05, nsim::Int=4000, htrim::Int=1, seed::Int=4321)
    rng = MersenneTwister(seed + T)
    sig_signal = sqrt(s2n)
    ar!(v) = (v[1]=randn(rng); for t in 2:T; v[t]=rho*v[t-1]+sqrt(1-rho^2)*randn(rng); end)
    # null: false breaks per period
    e = Vector{Float64}(undef,T); false_b = 0
    for s in 1:nsim; ar!(e); false_b += count_breaks(e; crit=crit, htrim=htrim); end
    alpha_eff = false_b / (nsim*(T-1))
    # H1: plant per-period steps, measure detection per truly-changed period
    y = Vector{Float64}(undef,T); det_true = 0; true_periods = 0
    for s in 1:nsim
        ar!(e); lvl=0.0; truebk=Int[]
        for t in 1:T
            if t>1 && rand(rng)<π_plant; lvl += sig_signal*sign(randn(rng))*(0.8+0.4*rand(rng)); push!(truebk,t); end
            y[t]=lvl+e[t]
        end
        isempty(truebk) && continue
        # crude allocation: a detected break within ±1 of a true break counts as a hit
        nb = count_breaks(y; crit=crit, htrim=htrim)
        det_true += min(nb, length(truebk)); true_periods += length(truebk)
    end
    gamma_eff = true_periods>0 ? det_true/true_periods : NaN
    return (alpha_eff=alpha_eff, gamma_eff=gamma_eff)
end

# BBG calibration constants. s2n = 2.64 is the Gottschalk–Huynh signal/noise;
# AK (Altonji–Kang) s2n = 1.80 is the robustness alternative. ρ = 0.482 is the
# BBG measurement-error autocorrelation. nsim is the Monte-Carlo replication
# count; the calibration is seeded (MersenneTwister, deterministic per length T)
# so the correction is reproducible run-to-run.
const SIPP_BBG_S2N  = 2.64   # signal/noise (Gottschalk–Huynh); AK s2n = 1.80 is the robustness alternative
const SIPP_BBG_RHO  = 0.482  # measurement-error AR(1) autocorrelation
const SIPP_BBG_NSIM = 3000   # Monte-Carlo replications per distinct series length

# Windows measured on the pre-2014 classic panels, where the earnings-based
# within-job wage change is measurement-noise-inflated and the BBG break filter
# is applied to the hourly-rate series. Every other window keeps the raw
# earnings construction (post-2014 redesign event-history design). Single point
# of control for the era split — move a window here to switch its construction.
const SIPP_BBG_WINDOWS = (:base_fc, :crisis_fc)

# ── CPI-U deflator ─────────────────────────────────────────────────────────
# BLS CPI-U, US city average, all items, annual average (series CUUR0000SA0,
# 1982–84 = 100). The Census public-use SIPP ships no CPI99 (that is IPUMS-
# only), so weekly earnings are deflated with this external CPI-U series
# rebased to 2013, consistent with the ASEC wage construction (setup.jl
# deflate_wage, 2013 base). Covers 1996–2025 so any window the project is
# likely to use is priced; a window that references a year outside this range
# raises a loud error in sipp_cpi (never deflates with a missing index). Extend
# both ends here if a window ever reaches beyond it.
const SIPP_CPIU = Dict{Int,Float64}(
    1996 => 156.900, 1997 => 160.500, 1998 => 163.000, 1999 => 166.600,
    2000 => 172.200, 2001 => 177.100, 2002 => 179.900, 2003 => 184.000,
    2004 => 188.900, 2005 => 195.300, 2006 => 201.600, 2007 => 207.342,
    2008 => 215.303, 2009 => 214.537, 2010 => 218.056, 2011 => 224.939,
    2012 => 229.594, 2013 => 232.957, 2014 => 236.736, 2015 => 237.017,
    2016 => 240.007, 2017 => 245.120, 2018 => 251.107, 2019 => 255.657,
    2020 => 258.811, 2021 => 270.970, 2022 => 292.655, 2023 => 304.702,
    2024 => 313.689, 2025 => 321.943,
)
const SIPP_CPIU_BASE_YEAR = 2013
const SIPP_CPIU_BASE       = 232.957   # CPI-U annual average, 2013

# CPI-U index for a calendar year, or a loud error naming the year. Called only
# after a pair has been assigned to a window, so it fires exactly when a window
# covers a calendar year with no CPI-U row — a real misconfiguration to fix at
# the SIPP_CPIU dict, never a record to deflate with a missing/zero index.
function sipp_cpi(year::Int)::Float64
    haskey(SIPP_CPIU, year) && return SIPP_CPIU[year]
    lo, hi = extrema(keys(SIPP_CPIU))
    error("SIPP deflation: no CPI-U index for calendar year $year (SIPP_CPIU " *
          "covers $(lo)–$(hi)). A window in setup.jl references $year; add the BLS " *
          "CUUR0000SA0 annual average for $year to SIPP_CPIU in sipp.jl.")
end

# ── Threshold-robustness grid ──────────────────────────────────────────────
# A within-job wage change is counted when |Δ real weekly wage| exceeds a
# threshold ε ($/week). The production moment uses ε = $1 (SIPP_WCHG_EPS); the
# other thresholds are re-measured in parallel for the diagnostic table printed
# at the end of the stage. Edit this vector to change the diagnostic grid; the
# production definition is fixed at $1.
const SIPP_WCHG_EPS_GRID = [1.0, 5.0, 10.0, 20.0]
const SIPP_WCHG_EPS       = 1.0   # locked production threshold ($/week)

# ── Cell coercion ──────────────────────────────────────────────────────────
# Coerce a CSV / fixed-width cell (Int / Float / String / missing) to Float64;
# NaN on failure.
function _sipp_to_float(v)::Float64
    ismissing(v) && return NaN
    v isa Real && return Float64(v)
    s = strip(string(v))
    isempty(s) && return NaN
    p = tryparse(Float64, s)
    return isnothing(p) ? NaN : p
end

# Coerce a cell to Int; `nothing` when missing/blank/unparseable. SIPP core
# files carry `missing` in the categorical columns (MONTHCODE, RMESR, TAGE,
# EEDUC/EEDUCATE), so a bare Int(...) is a latent crash — call sites treat
# `nothing` as "cannot qualify this record" and skip it.
function _sipp_to_int(v)::Union{Int,Nothing}
    f = _sipp_to_float(v)
    (isnan(f) || !isfinite(f)) && return nothing
    return round(Int, f)
end

# ── Real weekly wage ───────────────────────────────────────────────────────
# Day-count-neutral real weekly wage from a job's monthly earnings. Both SIPP
# formats report earnings as a monthly sum that varies with the number of days
# in the month (redesign TJB{n}_MSUM, classic TPMSUM{n}), so a flat ×12/52 (or
# raw monthly differencing) would flag a spurious change every month purely
# from day-count. Normalise to a 7-day week using the calendar month's length
# BEFORE differencing, then deflate to 2013 $.
function sipp_real_weekly_wage(msum::Float64, cal_year::Int, cal_month::Int,
                               cpi_ref::Float64)::Float64
    nominal_week = msum * 7.0 / Dates.daysinmonth(cal_year, cal_month)
    return deflate_wage(nominal_week, cpi_ref, SIPP_CPIU_BASE)
end

# ── Nominal hourly pay rate (BBG step 1) ────────────────────────────────────
# The BBG classic construction detects wage steps on the REPORTED HOURLY PAY
# RATE, not monthly earnings: earnings carry hours/overtime/rounding noise that
# crosses any threshold nearly every month, while the rate is far cleaner
# (Barattieri–Basu–Gottschalk 2014). Break structure is invariant to a monotone
# deflation, so the break filter runs on the NOMINAL rate and no CPI-U is
# applied to the rate series (the earnings-based wage above is still deflated —
# it is differenced, not break-detected).
#
# Classic TPYRATE{n} is stored in dollars-and-cents with two implied decimals
# (core dict value label "Dollars and cents (two implied decimals)"), so the raw
# integer field is divided by 100 to recover dollars. The redesign
# TJB{n}_HOURLY1 already arrives as a decimal dollar amount in the public-use
# CSV and needs no rescale.
#
# BBG step 1 keeps the hourly definition clean: a classic slot flagged hourly is
# admitted to the rate series only if it also carries a usable usual-hours
# reading EJBHRS{n} (a genuine hourly job, not a mislabelled record). EJBHRS = -8
# is the "Hours vary" sentinel and counts as unusable; such a slot is excluded
# (no imputation). The hours value gates inclusion only — the series the filter
# differences is the reported rate itself.
const SIPP_CLASSIC_RATE_SCALE = 100.0
const SIPP_CLASSIC_HOURS_VARY = -8   # EJBHRS{n} = -8 → "Hours vary" (unusable)

# ── Skilled classification ─────────────────────────────────────────────────
# Redesign EEDUC (highest level/degree, scale 31–46) — skilled is Bachelor's or
# higher (43 BA, 44 MA, 45 professional, 46 doctorate), matching the project's
# CPS EDUC>=111 definition. Distinct variable from CPS EDUC, so the shared
# is_skilled (educ>=111) never fires on this scale and is not reused here.
is_skilled_sipp(eeduc::Integer)::Bool = eeduc >= 43

# Classic EEDUCATE (highest degree/grade) uses a DIFFERENT scale from the
# redesign EEDUC: 44 Bachelor's, 45 Master's, 46 professional, 47 doctorate,
# with 43 an Associate degree (2001/2004/2008 core dictionaries, value labels
# for EEDUCATE). Bachelor's-or-higher is therefore ≥ 44, one code above the
# redesign threshold — the two must never share a classifier.
is_skilled_sipp_classic(eeducate::Integer)::Bool = eeducate >= 44

# ── Accumulators ────────────────────────────────────────────────────────────
# Within-job wage-change counts at every ε on the grid, per market (earnings-
# based construction). den/chg are the weighted at-risk and weighted changed
# stayer-pair masses; den2 is the ε-invariant Σw² carrying the Kish effective N
# (den²/den2). Held per window.
mutable struct WchgCounts
    eps    :: Vector{Float64}
    den_U  :: Vector{Float64}
    chg_U  :: Vector{Float64}
    den_S  :: Vector{Float64}
    chg_S  :: Vector{Float64}
    den2_U :: Base.RefValue{Float64}
    den2_S :: Base.RefValue{Float64}
end
WchgCounts(eps::Vector{Float64}) = WchgCounts(copy(eps),
    zeros(length(eps)), zeros(length(eps)), zeros(length(eps)), zeros(length(eps)),
    Ref(0.0), Ref(0.0))

# Bin one within-job stayer pair (same job, consecutive employed months) into
# every ε cell: it is always at risk (den), and changed (chg) where |Δw| > ε.
function _wchg_record_pair!(c::WchgCounts, skilled::Bool, wgt::Float64,
                            w1::Float64, w2::Float64)
    (isnan(w1) || isnan(w2) || !isfinite(wgt) || wgt <= 0.0) && return
    dw  = abs(w2 - w1)
    w2w = wgt * wgt
    if skilled
        c.den2_S[] += w2w
        @inbounds for k in eachindex(c.eps)
            c.den_S[k] += wgt
            dw > c.eps[k] && (c.chg_S[k] += wgt)
        end
    else
        c.den2_U[] += w2w
        @inbounds for k in eachindex(c.eps)
            c.den_U[k] += wgt
            dw > c.eps[k] && (c.chg_U[k] += wgt)
        end
    end
end

# Merge b into a element-wise (used only in the fixed-order serial merge, never
# during the parallel phase).
function _wchg_merge!(a::WchgCounts, b::WchgCounts)
    a.den_U .+= b.den_U;  a.chg_U .+= b.chg_U
    a.den_S .+= b.den_S;  a.chg_S .+= b.chg_S
    a.den2_U[] += b.den2_U[];  a.den2_S[] += b.den2_S[]
end

# Hourly-worker share among employed person-months, per market. den is the
# weighted at-risk mass (every employed in-age person-month assigned to a
# window); hr is the subset whose main job is paid by the hour (the population
# the BBG step-1 rate series is built on). share = hr/den tells how much the
# hourly restriction thins the classic cell, so the BBG effective N can be read.
mutable struct HourlyShare
    den_U :: Base.RefValue{Float64}
    hr_U  :: Base.RefValue{Float64}
    den_S :: Base.RefValue{Float64}
    hr_S  :: Base.RefValue{Float64}
end
HourlyShare() = HourlyShare(Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0))

function _hshare_merge!(a::HourlyShare, b::HourlyShare)
    a.den_U[] += b.den_U[];  a.hr_U[] += b.hr_U[]
    a.den_S[] += b.den_S[];  a.hr_S[] += b.hr_S[]
end

# Add one employed person-month to its market's share tally: always at-risk
# (den), and hourly (hr) when its main job is paid by the hour.
function _hshare_record!(h::HourlyShare, skilled::Bool, wgt::Float64, is_hourly::Bool)
    (isfinite(wgt) && wgt > 0.0) || return
    if skilled
        h.den_S[] += wgt;  is_hourly && (h.hr_S[] += wgt)
    else
        h.den_U[] += wgt;  is_hourly && (h.hr_U[] += wgt)
    end
    return nothing
end

# Skilled EE-mobility counts. ee is the poach hazard (weighted moves over
# weighted at-risk employed pairs); step is the mean LOG real-weekly-wage jump
# on a move (Δlog w = log w_new − log w_old), with its weighted sd and Kish N.
# Skilled only (no unskilled OJS).
mutable struct EeCounts
    den_ee    :: Base.RefValue{Float64}   # Σw at-risk employed-skilled pairs
    chg_ee    :: Base.RefValue{Float64}   # Σw pairs whose main employer changed
    den2_ee   :: Base.RefValue{Float64}   # Σw² at-risk (Kish N for ee_rate)
    den_step  :: Base.RefValue{Float64}   # Σw over movers
    sum_step  :: Base.RefValue{Float64}   # Σ w·Δlog w
    sum_step2 :: Base.RefValue{Float64}   # Σ w·Δlog w² (weighted sd of the log step)
    den2_step :: Base.RefValue{Float64}   # Σw² over movers (Kish N for ee_step)
end
EeCounts() = EeCounts(Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0))

function _ee_merge!(a::EeCounts, b::EeCounts)
    a.den_ee[]    += b.den_ee[];    a.chg_ee[]    += b.chg_ee[]
    a.den2_ee[]   += b.den2_ee[];   a.den_step[]  += b.den_step[]
    a.sum_step[]  += b.sum_step[];  a.sum_step2[] += b.sum_step2[]
    a.den2_step[] += b.den2_step[]
end

# BBG nominal-hourly-rate observations for one window, split by market. Each job
# id maps to its (calendar-month, nominal-rate) readings; the fragments are
# merged across wave files (a classic job spans several waves) and only split
# into contiguous monthly series at output time, so a job's full within-window
# rate series is reconstructed regardless of thread scheduling. Collected for the
# BBG windows only (SIPP_BBG_WINDOWS); the break filter runs on these series.
mutable struct BbgCell
    U :: Dict{String,Vector{Tuple{Int,Float64}}}
    S :: Dict{String,Vector{Tuple{Int,Float64}}}
end
BbgCell() = BbgCell(Dict{String,Vector{Tuple{Int,Float64}}}(),
                    Dict{String,Vector{Tuple{Int,Float64}}}())

function _bbg_merge!(a::BbgCell, b::BbgCell)
    for (jid, obs) in b.U; append!(get!(() -> Tuple{Int,Float64}[], a.U, jid), obs); end
    for (jid, obs) in b.S; append!(get!(() -> Tuple{Int,Float64}[], a.S, jid), obs); end
end

# Record one hourly person-month reading onto its job's rate series.
function _bbg_record!(cell::BbgCell, skilled::Bool, jid::String, cal_m::Int, rate::Float64)
    (isfinite(rate) && rate > 0.0) || return
    d = skilled ? cell.S : cell.U
    push!(get!(() -> Tuple{Int,Float64}[], d, jid), (cal_m, rate))
    return nothing
end

# One file's private accumulator: window-keyed earnings-based wage-change counts,
# hourly-worker share, and skilled EE counts, plus the BBG nominal-hourly-rate
# series store for the classic (FC) windows. No two files share one of these
# during the parallel phase; the fixed-order merge below makes the merged totals
# a deterministic function of the inputs alone.
struct SippAcc
    wchg       :: Dict{Symbol,WchgCounts}    # earnings-based wage-change counts (all windows)
    hshare     :: Dict{Symbol,HourlyShare}   # hourly-worker share (all windows)
    ee         :: Dict{Symbol,EeCounts}      # skilled EE hazard + wage step (all windows)
    bbg        :: Dict{Symbol,BbgCell}       # nominal hourly-rate series (BBG windows only)
    years_used :: Set{Int}                   # calendar years actually deflated (for the CPI-U report)
end
SippAcc() = SippAcc(Dict{Symbol,WchgCounts}(), Dict{Symbol,HourlyShare}(),
                    Dict{Symbol,EeCounts}(), Dict{Symbol,BbgCell}(), Set{Int}())

_wchg_get!(acc::SippAcc, w::Symbol)   = get!(() -> WchgCounts(SIPP_WCHG_EPS_GRID), acc.wchg, w)
_hshare_get!(acc::SippAcc, w::Symbol) = get!(() -> HourlyShare(), acc.hshare, w)
_ee_get!(acc::SippAcc, w::Symbol)     = get!(() -> EeCounts(), acc.ee, w)
_bbg_get!(acc::SippAcc, w::Symbol)    = get!(() -> BbgCell(), acc.bbg, w)

# Sum a file accumulator into the master. Called in fixed sorted file order so
# the merged totals do not depend on thread scheduling.
function _sipp_merge!(master::SippAcc, f::SippAcc)
    for (w, c) in f.wchg
        _wchg_merge!(get!(() -> WchgCounts(SIPP_WCHG_EPS_GRID), master.wchg, w), c)
    end
    for (w, h) in f.hshare
        _hshare_merge!(get!(() -> HourlyShare(), master.hshare, w), h)
    end
    for (w, e) in f.ee
        _ee_merge!(get!(() -> EeCounts(), master.ee, w), e)
    end
    for (w, b) in f.bbg
        _bbg_merge!(_bbg_get!(master, w), b)
    end
    union!(master.years_used, f.years_used)
end

# ── Per-person-month record and the shared qualification routine ────────────
# One non-empty job slot of a person-month. `msum` is the month's earnings, used
# for the earnings-based wage change and the stayer-pair matching on `id`.
# `hourly` flags a job paid by the hour this month (classic EPAYHR=1, redesign
# EJB{n}_PAYHR1=2); `rate` is its NOMINAL reported hourly pay rate (classic
# TPYRATE/100, redesign TJB{n}_HOURLY1), or NaN when the slot is not hourly or
# its rate reading is unusable. The BBG break filter builds its per-job monthly
# series from `rate` on the hourly slots.
struct JobSlot
    id     :: String
    msum   :: Float64
    hourly :: Bool
    rate   :: Float64
end

# A parsed person-month, produced by either format's streaming parser and fed
# to the format-agnostic processor below. `skilled` is already resolved with the
# format's own classifier so the processor stays format-blind.
struct PMRecord
    cal_m   :: Int                      # yr*12 + mn (consecutive-month test + window)
    yr      :: Int
    mn      :: Int
    employed:: Bool                     # RMESR indicates employed this month
    inage   :: Bool                     # 16 ≤ TAGE ≤ 64
    skilled :: Bool
    wgt     :: Float64                  # WPFINWGT
    jobs    :: Vector{JobSlot}
end

# Highest-earning ("main") job of a month, or nothing when no job is held.
function _main_job(rec::PMRecord)
    isempty(rec.jobs) && return nothing
    best = rec.jobs[1]
    @inbounds for k in 2:length(rec.jobs)
        rec.jobs[k].msum > best.msum && (best = rec.jobs[k])
    end
    return best
end

# Process one person's month records into `acc`. Sorts by calendar month and
# walks consecutive employed pairs (Δcal_m == 1, employed and in-age both
# months), each routed to the pair's window by r2's calendar month:
#   • within-job wage change, EARNINGS-based (both markets): every job id held in
#     BOTH months is a stayer pair — its real weekly wages (monthly earnings,
#     day-count-neutral) are differenced and binned;
#   • EE hazard (skilled): the pair is at risk; a move is when the main-job id
#     differs across the two months;
#   • EE wage step (skilled movers): the LOG new main-job weekly wage minus the
#     LOG old main-job weekly wage, on the same day-count-neutral 2013-$ wages as
#     the wage change (a proportional step, matching the log-wage moment scale).
# Weights and skilled status are taken from the later month (r2), matching the
# stock-moment convention.
# A separate person-month pass (below) (i) tallies the hourly-worker share and
# (ii) for the BBG windows collects the nominal hourly-rate reading of each
# hourly main job onto its job-id series, from which the break filter measures
# genuine wage steps at output time.
function _process_person!(acc::SippAcc, recs::Vector{PMRecord})::Int
    length(recs) < 2 && return 0
    sort!(recs; by = r -> r.cal_m)
    n_win_pairs = 0
    @inbounds for i in 1:length(recs)-1
        r1 = recs[i];  r2 = recs[i+1]
        (r2.cal_m - r1.cal_m == 1) || continue
        (r1.employed && r2.employed && r1.inage && r2.inage) || continue

        w = assign_window(r2.yr, r2.mn)
        w == :none && continue
        n_win_pairs += 1

        skilled = r2.skilled
        wgt     = r2.wgt
        (isfinite(wgt) && wgt > 0.0) || continue

        cpi1 = sipp_cpi(r1.yr);  cpi2 = sipp_cpi(r2.yr)
        push!(acc.years_used, r1.yr);  push!(acc.years_used, r2.yr)

        # (a) within-job wage change, earnings-based — one stayer pair per job id
        #     held in both months, its real weekly wages differenced and binned.
        wc = _wchg_get!(acc, w)
        for s1 in r1.jobs
            isempty(s1.id) && continue
            idx2 = findfirst(s -> s.id == s1.id, r2.jobs)
            isnothing(idx2) && continue
            s2 = r2.jobs[idx2]
            w1 = sipp_real_weekly_wage(s1.msum, r1.yr, r1.mn, cpi1)
            w2 = sipp_real_weekly_wage(s2.msum, r2.yr, r2.mn, cpi2)
            _wchg_record_pair!(wc, skilled, wgt, w1, w2)
        end

        # (b),(c) EE hazard + wage step — skilled only, main-job change
        skilled || continue
        mj1 = _main_job(r1);  mj2 = _main_job(r2)
        (isnothing(mj1) || isnothing(mj2)) && continue
        ee = _ee_get!(acc, w)
        ee.den_ee[]  += wgt
        ee.den2_ee[] += wgt * wgt
        if mj1.id != mj2.id                       # employer changed → EE move
            ee.chg_ee[] += wgt
            w_old = sipp_real_weekly_wage(mj1.msum, r1.yr, r1.mn, cpi1)
            w_new = sipp_real_weekly_wage(mj2.msum, r2.yr, r2.mn, cpi2)
            if isfinite(w_old) && isfinite(w_new) && w_old > 0.0 && w_new > 0.0
                step = log(w_new) - log(w_old)    # LOG step: same scale as every other wage moment
                ee.den_step[]  += wgt
                ee.sum_step[]  += wgt * step
                ee.sum_step2[] += wgt * step * step
                ee.den2_step[] += wgt * wgt
            end
        end
    end

    # Person-month pass over employed, in-age records assigned to a window:
    #   • hourly-worker share — fraction whose main job is paid by the hour, per
    #     market (each record counted once with its own weight and skill), the
    #     context for the BBG cell's effective N;
    #   • BBG rate series — for the BBG windows, the main job's nominal hourly
    #     rate is appended to that job's within-window series (keyed by job id and
    #     market). The series are split into contiguous monthly runs and break-
    #     filtered at output time.
    @inbounds for r in recs
        (r.employed && r.inage && isfinite(r.wgt) && r.wgt > 0.0) || continue
        w = assign_window(r.yr, r.mn)
        w == :none && continue
        mj = _main_job(r)
        is_hourly = !isnothing(mj) && mj.hourly
        _hshare_record!(_hshare_get!(acc, w), r.skilled, r.wgt, is_hourly)
        if w in SIPP_BBG_WINDOWS && is_hourly
            _bbg_record!(_bbg_get!(acc, w), r.skilled, mj.id, r.cal_m, mj.rate)
        end
    end
    return n_win_pairs
end

# ── Hazard / effective-N helpers ────────────────────────────────────────────
# Monthly transition probability → continuous-time hazard −log(1−p), matching
# the model's Poisson rates (moments.jl frequency-consistency block leaves these
# untouched because they arrive already converted). NaN denominator ⇒ NaN.
function _sipp_hazard(chg::Float64, den::Float64)::Float64
    den > 0.0 || return NaN
    p = min(chg / den, 1.0 - 1e-12)
    return p > 0.0 ? -log(1.0 - p) : 0.0
end

# Kish effective sample size (Σw)² / Σw² for a weighted proportion / mean.
_sipp_neff(den::Float64, den2::Float64)::Float64 = den2 > 0.0 ? den^2 / den2 : 0.0

# Index of the production ε on the grid.
_sipp_prod_eps_idx()::Int = findfirst(==(SIPP_WCHG_EPS), SIPP_WCHG_EPS_GRID)

# ── BBG cell → corrected wage-change hazard ─────────────────────────────────
# One market's collected nominal-hourly-rate series in a BBG window is turned
# into a size/power-corrected within-job wage-change hazard, BBG (2014):
#   1. split each job's readings into contiguous monthly runs (a gap in the
#      calendar-month index ends a run); each run of length T ≥ 3 is one series;
#   2. count Bai–Perron breaks per series at the length's own critical value;
#      π̂ = Σ breaks / Σ (T−1) over the cell;
#   3. correct π̃ = (π̂ − ᾱ)/(γ̄ − ᾱ), where ᾱ, γ̄ are the (T−1)-weighted means of
#      the per-length α_eff(T), γ_eff(T). Calibration is memoized per distinct T.
# Returns (pihat, pitil, corrfac, wchg, neff, underflow, nseries):
#   pihat    raw per-period break rate;
#   pitil    corrected per-period wage-change probability (0 when π̂ ≤ ᾱ);
#   corrfac  the correction factor γ̄ − ᾱ (sigma.jl scales the π̂ variance by
#            1/corrfac² — the correction inflates the variance);
#   wchg     monthly hazard −log(1 − clamp(π̃, 0, 1−1e-12));
#   neff     Σ(T−1), the number of at-risk worker-periods behind π̂;
#   underflow true when π̃ was forced to 0 — either π̃ ≤ 0 (over-correction on a
#            thin cell) or a non-positive correction factor γ̄ ≤ ᾱ (degenerate
#            calibration, off by BBG design but guarded against);
#   nseries  number of length-≥3 series entering the cell (0 ⇒ no measurement).
function _bbg_cell_hazard(series_by_job::Dict{String,Vector{Tuple{Int,Float64}}},
                          calib::Dict{Int,NamedTuple})::NamedTuple
    tot_breaks = 0
    tot_periods = 0            # Σ(T−1), the effective N for π̂
    sum_alpha_w = 0.0          # Σ (T−1)·α_eff(T)  → ᾱ = /tot_periods
    sum_gamma_w = 0.0          # Σ (T−1)·γ_eff(T)  → γ̄
    nseries = 0
    for obs in values(series_by_job)
        length(obs) < 3 && continue
        sort!(obs; by = first)
        # split at any calendar-month gap into contiguous monthly runs
        run = Float64[obs[1][2]]
        prev = obs[1][1]
        flush_run!() = begin
            T = length(run)
            if T >= 3
                cal = get!(() -> begin
                        c  = calibrate(T; rho=SIPP_BBG_RHO, s2n=SIPP_BBG_S2N, nsim=SIPP_BBG_NSIM)
                        cp = calibrate_perperiod(T; crit=c.crit, rho=SIPP_BBG_RHO,
                                                 s2n=SIPP_BBG_S2N, nsim=SIPP_BBG_NSIM)
                        (crit=c.crit, alpha_eff=cp.alpha_eff, gamma_eff=cp.gamma_eff)
                    end, calib, T)
                nb = count_breaks(run; crit=cal.crit)
                tot_breaks  += nb
                tot_periods += T - 1
                sum_alpha_w += (T - 1) * cal.alpha_eff
                sum_gamma_w += (T - 1) * cal.gamma_eff
                nseries += 1
            end
        end
        for k in 2:length(obs)
            m, rate = obs[k]
            if m == prev + 1
                push!(run, rate)
            else
                flush_run!(); empty!(run); push!(run, rate)
            end
            prev = m
        end
        flush_run!()
    end

    if nseries == 0 || tot_periods == 0
        return (pihat=NaN, pitil=NaN, corrfac=NaN, wchg=NaN,
                neff=0.0, underflow=false, nseries=0)
    end
    pihat   = tot_breaks / tot_periods
    abar    = sum_alpha_w / tot_periods
    gbar    = sum_gamma_w / tot_periods
    corrfac = gbar - abar
    pitil   = corrfac > 0.0 ? (pihat - abar) / corrfac : NaN
    underflow = !(isfinite(pitil)) || pitil <= 0.0
    pitil_c = underflow ? 0.0 : clamp(pitil, 0.0, 1.0 - 1e-12)
    wchg    = -log(1.0 - pitil_c)
    return (pihat=pihat, pitil=(underflow ? 0.0 : pitil), corrfac=corrfac,
            wchg=wchg, neff=Float64(tot_periods), underflow=underflow, nseries=nseries)
end

# ── Redesign format (2014-present, comma-delimited pu<Y>.csv) ───────────────
# Reference calendar year of a redesign file. The 2014 panel ships one file per
# wave, pu2014w<N>.csv, whose reference year is the wave's data year 2012+N
# (wave 1 → 2013, wave 2 → 2014, …). The annual releases pu<YYYY>.csv reference
# the prior calendar year (pu2018.csv → 2017, …). MONTHCODE gives the month
# within that reference year, so window assignment is per record via
# assign_window(ref_year, MONTHCODE) with no window years hardcoded.
function sipp_redesign_ref_year(fname::String)::Int
    m = match(r"^pu(\d{4})w(\d+)"i, fname)
    isnothing(m) || return 2012 + parse(Int, m.captures[2])
    m = match(r"^pu(\d{4})"i, fname)
    isnothing(m) || return parse(Int, m.captures[1]) - 1
    error("sipp_redesign_ref_year: cannot parse reference year from '$fname'")
end

# Non-empty job slots of a redesign row: EJB{n}_JOBID holds the within-file job
# identifier, TJB{n}_MSUM the month's earnings for that job. Slots with a blank
# / "0" id or a non-positive earnings sum are unfilled and dropped. Job ids are
# namespaced with SSUID+PNUM so identical slot numbers across people never
# collide.
#
# For the BBG break filter each slot also reads EJB{n}_PAYHR1 (pay type; value
# 2 = paid by the hour) and TJB{n}_HOURLY1 (primary hourly-rate dollar amount).
# An hourly slot with a non-positive / missing rate carries rate = NaN and is
# simply absent from that job's rate series.
const REDESIGN_HOURLY_CODE = 2   # EJB{n}_PAYHR1 = 2 → "Pay per hour"
const REDESIGN_JOB_SLOTS   = 7   # EJB1..7 job slots in the redesign core

# Per-slot column names the redesign reader touches for one job slot: id,
# earnings, and the two BBG rate fields (pay type, hourly rate). Single source of
# truth for both the parser and the CSV select= list, so adding a field here
# keeps the streamed columns in sync.
_sipp_redesign_slot_fields(n::Int) = (Symbol("EJB$(n)_JOBID"), Symbol("TJB$(n)_MSUM"),
    Symbol("EJB$(n)_PAYHR1"), Symbol("TJB$(n)_HOURLY1"))

function _sipp_redesign_job_records(row, pidkey::String, n_slots::Int)
    jobs = JobSlot[]
    for n in 1:n_slots
        idcol = Symbol("EJB$(n)_JOBID");  wcol = Symbol("TJB$(n)_MSUM")
        (hasproperty(row, idcol) && hasproperty(row, wcol)) || continue
        jidv = getproperty(row, idcol)
        ismissing(jidv) && continue
        jid = strip(string(jidv))
        (isempty(jid) || jid == "0") && continue
        msum = _sipp_to_float(getproperty(row, wcol))
        (isnan(msum) || msum <= 0.0) && continue

        paycol = Symbol("EJB$(n)_PAYHR1")
        pay    = hasproperty(row, paycol) ? _sipp_to_int(getproperty(row, paycol)) : nothing
        hourly = pay === REDESIGN_HOURLY_CODE
        rate = NaN
        if hourly
            hrcol = Symbol("TJB$(n)_HOURLY1")
            r     = hasproperty(row, hrcol) ? _sipp_to_float(getproperty(row, hrcol)) : NaN
            rate  = (isfinite(r) && r > 0.0) ? r : NaN
        end
        push!(jobs, JobSlot("$(pidkey)|$jid", msum, hourly, rate))
    end
    return jobs
end

const REDESIGN_REQUIRED = [:SSUID, :PNUM, :MONTHCODE, :RMESR, :EEDUC, :TAGE, :WPFINWGT]

# Columns the redesign reader actually reads: the required person-month fields
# plus every per-slot job field over all slots. Passed to CSV.Rows as select= so
# CSV.jl tokenises/coerces only these instead of all 5000+ columns of an annual
# pu file (the dominant cost on the multi-GB releases). Derived from the same
# REDESIGN_REQUIRED + slot-field builder the parser uses, so it never drifts out
# of sync. Columns listed but absent from a file are simply not selected —
# CSV.Rows reports only present columns via propertynames, so the missing-column
# guard on REDESIGN_REQUIRED still fires when a required field is genuinely gone.
const REDESIGN_SELECT = vcat(REDESIGN_REQUIRED,
    collect(Iterators.flatten(_sipp_redesign_slot_fields(n) for n in 1:REDESIGN_JOB_SLOTS)))

# Stream one redesign core into `acc`. CSV.Rows reads row-by-row without
# materialising the file (multi-GB annual releases stay within one row of
# memory), and records are flushed per person: rows arrive grouped by
# (SSUID,PNUM) in the Census files, so a person is complete once the id changes.
# Returns (n_pairs, ref_year, in_window) for the caller's ordered info line;
# n_pairs is the skilled-EE at-risk pair count actually accumulated.
function _sipp_accumulate_redesign!(acc::SippAcc, path::String)
    fname    = basename(path)
    ref_year = sipp_redesign_ref_year(fname)
    in_window = any(assign_window(ref_year, mc) != :none for mc in 1:12)

    cur_pid = ""
    recs    = PMRecord[]
    n_pairs = Ref(0)
    checked = false
    flush!() = (n_pairs[] += _process_person!(acc, recs); empty!(recs))

    # SSUID/PNUM forced to String so long numeric ids never overflow or gain a
    # decimal point; every other column is coerced per-cell below. reusebuffer
    # keeps one row of memory live at a time. select= restricts tokenisation to
    # the ~40 columns the reader touches (REDESIGN_SELECT) rather than the annual
    # file's 5000+, the dominant read cost on the multi-GB releases.
    for row in CSV.Rows(path; reusebuffer = true, select = REDESIGN_SELECT,
                        types = Dict(:SSUID => String, :PNUM => String))
        if !checked
            missing_cols = setdiff(REDESIGN_REQUIRED, propertynames(row))
            isempty(missing_cols) ||
                return (0, ref_year, in_window,
                        "missing column(s) " * join(missing_cols, ", "))
            checked = true
        end
        ssuid = strip(string(row.SSUID));  pnum = strip(string(row.PNUM))
        pid   = "$ssuid|$pnum"
        if pid != cur_pid
            isempty(recs) || flush!()
            cur_pid = pid
        end
        mc = _sipp_to_int(row.MONTHCODE)
        (isnothing(mc) || !(1 <= mc <= 12)) && continue
        rmesr = _sipp_to_int(row.RMESR)
        educ  = _sipp_to_int(row.EEDUC)
        age   = _sipp_to_int(row.TAGE)
        wgt   = _sipp_to_float(row.WPFINWGT)
        push!(recs, PMRecord(ref_year * 12 + mc, ref_year, mc,
            rmesr === 1,
            !isnothing(age)  && in_age_range(age),
            !isnothing(educ) && is_skilled_sipp(educ),
            wgt, _sipp_redesign_job_records(row, pid, REDESIGN_JOB_SLOTS)))
    end
    isempty(recs) || flush!()

    return (n_pairs[], ref_year, in_window, "")
end

# ── Classic format (2001/2004/2008, fixed-width l<YY>puw<N>.dat) ────────────
# Parse a classic core DICTIONARY into name → (startpos, width), 1-indexed. The
# `D VARNAME width startpos` lines describe the fixed-width record; the first D
# line for a name is its core definition (later D lines are value labels).
function parse_classic_dict(path::String)::Dict{String,Tuple{Int,Int}}
    pos = Dict{String,Tuple{Int,Int}}()
    for line in eachline(path)
        m = match(r"^D\s+([A-Z0-9_]+)\s+(\d+)\s+(\d+)\s*$", line)
        isnothing(m) && continue
        name = m.captures[1]
        haskey(pos, name) && continue
        pos[name] = (parse(Int, m.captures[3]), parse(Int, m.captures[2]))
    end
    return pos
end

# Extract a fixed-width field by name from a record line; `nothing` when the
# field is out of range or blank. posmap[name] = (startpos, width), 1-indexed.
function _classic_field(line::AbstractString, posmap, name::String)
    haskey(posmap, name) || return nothing
    sp, w = posmap[name]
    stop = sp + w - 1
    stop <= lastindex(line) || return nothing
    s = strip(SubString(line, sp, stop))
    return isempty(s) ? nothing : s
end

# Report a compressed core the walk cannot read and skip it. The SIPP release
# ships extracted files and the project depot carries no zip/gzip codec, so a
# .dat.gz or .zip is never decompressed here. A redundant compressed copy (its
# extracted sibling already present) is ignored silently; otherwise the exact
# gunzip/unzip command is printed so the user can extract and re-run. Never adds
# a dependency, never fails.
function _sipp_warn_compressed(path::String)
    dir, fname = dirname(path), basename(path)
    low = lowercase(fname)
    if endswith(low, ".gz")
        isfile(joinpath(dir, chopsuffix(fname, ".gz"))) && return nothing
        @warn "    $fname is gzipped and not extracted — run `gunzip $path` and re-run the data stage; skipped."
    elseif endswith(low, ".zip")
        isfile(joinpath(dir, chopsuffix(fname, ".zip"))) && return nothing
        @warn "    $fname is a zip archive and not extracted — run `unzip $path -d $dir` and re-run the data stage; skipped."
    end
    return nothing
end

# Non-empty job slots of a classic record: employer entry number EENO<n> paired
# with monthly earnings TPMSUM<n>, two slots (the classic analogue of the
# redesign EJB<n>_JOBID / TJB<n>_MSUM). The id is namespaced with the person key
# so slot ids never collide across people.
#
# For the BBG break filter each slot also reads EPAYHR{n} (paid by the hour,
# value 1 = Yes), TPYRATE{n} (regular hourly pay rate, two implied decimals →
# ÷100), and EJBHRS{n} (usual weekly hours; -8 "Hours vary" is unusable). An
# hourly slot is admitted to the rate series only when BOTH the rate and the
# usual-hours reading are usable; otherwise rate = NaN and the slot is absent
# from that job's series (BBG step 1, no imputation). The hours reading is the
# inclusion gate only — the series value is the rate itself.
const CLASSIC_HOURLY_CODE = 1   # EPAYHR{n} = 1 → "Yes (paid by the hour)"

function _sipp_classic_job_records(line::AbstractString, posmap, pidkey::String)
    jobs = JobSlot[]
    for n in 1:2
        eeno = _classic_field(line, posmap, "EENO$(n)")
        (isnothing(eeno) || eeno == "0") && continue
        msum = _sipp_to_float(_classic_field(line, posmap, "TPMSUM$(n)"))
        (isnan(msum) || msum <= 0.0) && continue

        pay    = _sipp_to_int(_classic_field(line, posmap, "EPAYHR$(n)"))
        hourly = pay === CLASSIC_HOURLY_CODE
        rate = NaN
        if hourly
            rate_raw = _sipp_to_float(_classic_field(line, posmap, "TPYRATE$(n)"))
            hours    = _sipp_to_float(_classic_field(line, posmap, "EJBHRS$(n)"))
            hours    = hours == SIPP_CLASSIC_HOURS_VARY ? NaN : hours
            r = isnan(rate_raw) ? NaN : rate_raw / SIPP_CLASSIC_RATE_SCALE
            rate = (isfinite(r) && r > 0.0 && isfinite(hours) && hours > 0.0) ? r : NaN
        end
        push!(jobs, JobSlot("$(pidkey)|$(eeno)", msum, hourly, rate))
    end
    return jobs
end

const CLASSIC_REQUIRED = ["SSUID", "EPPPNUM", "RHCALYR", "RHCALMN",
                          "WPFINWGT", "EEDUCATE", "RMESR",
                          "EENO1", "EENO2", "TPMSUM1", "TPMSUM2"]

# Classic panel year from a core filename l<YY>puw<N>.dat (YY → 2000+YY).
function sipp_classic_panel_year(fname::AbstractString)
    m = match(r"l(\d{2})puw\d+\.dat"i, fname)
    isnothing(m) && return nothing
    return 2000 + parse(Int, m.captures[1])
end

# Stream one classic core wave into `acc`, locating fields by name from the
# panel's wave-1 core dictionary (resolved once by the caller and passed as
# `posmap`). Records are read line-by-line and grouped into per-person buffers,
# then handed to the shared _process_person! — identical pair qualification to
# the redesign path (consecutive employed in-age months, same-job wage change,
# skilled EE hazard + wage step, 2013 diagnostic). A classic wave is four months
# and modest in size, so buffering one wave's persons is bounded memory.
# Returns (n_pairs, ref_year, in_window, err) for the caller's ordered info
# line; ref_year is the panel year for reporting.
function _sipp_accumulate_classic!(acc::SippAcc, dat_path::String, posmap)
    fname = basename(dat_path)
    panel = sipp_classic_panel_year(fname)

    # The wave-1 dictionary is applied to every wave on the premise that the core
    # layout is constant within a panel. Enforce it: a record shorter than the
    # rightmost byte any CLASSIC_REQUIRED field occupies means a layout mismatch,
    # so the wave is skipped rather than read with misaligned fields.
    reclen_needed = maximum(sp + w - 1 for (sp, w) in (posmap[v] for v in CLASSIC_REQUIRED))

    ref_year = isnothing(panel) ? 0 : panel
    by_pid = Dict{String,Vector{PMRecord}}()
    length_checked = false
    io = open(dat_path)
    try
        for line in eachline(io)
            isempty(strip(line)) && continue
            if !length_checked
                length(line) < reclen_needed &&
                    return (0, ref_year, false,
                            "record length $(length(line)) < $reclen_needed (layout mismatch)")
                length_checked = true
            end
            yr = _sipp_to_int(_classic_field(line, posmap, "RHCALYR"))
            mn = _sipp_to_int(_classic_field(line, posmap, "RHCALMN"))
            (isnothing(yr) || isnothing(mn) || !(1 <= mn <= 12)) && continue
            ssuid = _classic_field(line, posmap, "SSUID")
            pnum  = _classic_field(line, posmap, "EPPPNUM")
            (isnothing(ssuid) || isnothing(pnum)) && continue
            pid   = string(ssuid, "|", pnum)
            rmesr = _sipp_to_int(_classic_field(line, posmap, "RMESR"))
            age   = _sipp_to_int(_classic_field(line, posmap, "TAGE"))
            educ  = _sipp_to_int(_classic_field(line, posmap, "EEDUCATE"))
            wgt   = _sipp_to_float(_classic_field(line, posmap, "WPFINWGT"))
            push!(get!(() -> PMRecord[], by_pid, pid), PMRecord(
                yr * 12 + mn, yr, mn,
                rmesr === 1,
                !isnothing(age)  && in_age_range(age),
                !isnothing(educ) && is_skilled_sipp_classic(educ),
                wgt, _sipp_classic_job_records(line, posmap, pid)))
        end
    finally
        close(io)
    end
    isempty(by_pid) &&
        return (0, ref_year, false, "no parseable records")

    # Persons processed in sorted id order so the file's contribution is a
    # deterministic function of its contents, independent of Dict iteration.
    n_pairs = 0
    for pid in sort(collect(keys(by_pid)))
        n_pairs += _process_person!(acc, by_pid[pid])
    end

    # Window membership and the reported year come from the RECORDS (RHCALYR),
    # not the panel label: a classic panel spans several calendar years, so the
    # out-of-window report must name the data years actually present (a 2008
    # panel that only reaches 2013 should read "no window covers 2013").
    yrs = sort(unique(r.yr for r in Iterators.flatten(values(by_pid))))
    data_year = isempty(yrs) ? ref_year : first(yrs)
    in_window = any(assign_window(r.yr, r.mn) != :none
                    for r in Iterators.flatten(values(by_pid)))
    return (n_pairs, data_year, in_window, "")
end

# Resolve a classic panel's core dictionary from its wave-1 core dictionary and
# return (filename, posmap). `dict_dir` is the directory the panel's own cores
# live in, so the dictionary is read from beside the data it describes. Census
# ships the core dictionary for wave 1 only, so one dictionary documents every
# wave. The plain name l<YY>puw1d.txt is preferred; the 2008 panel's plain file
# survives only as a suffixed revision, so `.old2` then `.old.2` are tried as
# fallbacks. A candidate is accepted only if its posmap carries every
# CLASSIC_REQUIRED variable, which rejects the truncated stubs. Returns nothing
# (with a loud @warn naming the panel) when none is usable.
function _resolve_classic_dict(dict_dir::String, panel_yy::AbstractString)
    candidates = ["l$(panel_yy)puw1d.txt",
                  "l$(panel_yy)puw1d.txt.old2",
                  "l$(panel_yy)puw1d.txt.old.2"]
    for name in candidates
        path = joinpath(dict_dir, name)
        isfile(path) || continue
        posmap = parse_classic_dict(path)
        missing_vars = filter(v -> !haskey(posmap, v), CLASSIC_REQUIRED)
        isempty(missing_vars) && return (name, posmap)
        @warn "    classic panel 20$panel_yy: dictionary $name missing " *
              "$(join(missing_vars, ", ")) — trying next candidate."
    end
    @warn "    classic panel 20$panel_yy: no usable wave-1 core dictionary " *
          "(looked for $(join(candidates, ", "))) — panel skipped."
    return nothing
end

# ── Parallel directory scan and deterministic merge ─────────────────────────
# One unit of work: a discovered file plus everything its reader needs. Classic
# waves carry the resolved wave-1 posmap (classic files with no usable
# dictionary are dropped before the work list is built, so every unit here is
# readable).
struct SippWork
    kind   :: Symbol            # :redesign | :classic
    path   :: String
    sortkey:: String            # fixed merge order: basename, unique across the walk
    posmap :: Union{Nothing,Dict{String,Tuple{Int,Int}}}
end

# Build the ordered work list from a recursive walk. Redesign cores are pu*.csv;
# classic cores are l<YY>puw<N>.dat, grouped by panel YY with the wave-1 core
# dictionary resolved once per panel and shared by every wave (Census ships that
# dictionary for wave 1 only). Compressed cores with no extracted sibling are
# reported and skipped. The returned vector is sorted by basename so the merge
# order below is fixed and independent of walk/thread order.
function _sipp_build_worklist(sipp_dir::String)::Vector{SippWork}
    redesign = String[]
    classic  = String[]
    for (root, _, files) in walkdir(sipp_dir)
        for f in files
            if occursin(r"^pu.*\.csv$"i, f)
                push!(redesign, joinpath(root, f))
            elseif occursin(r"^l\d{2}puw\d+\.dat$"i, f)
                push!(classic, joinpath(root, f))
            elseif occursin(r"^(pu.*\.csv|l\d{2}puw\d+\.dat)\.(gz|zip)$"i, f)
                _sipp_warn_compressed(joinpath(root, f))
            end
        end
    end

    work = SippWork[]
    for path in redesign
        push!(work, SippWork(:redesign, path, basename(path), nothing))
    end

    panels = Dict{String,Vector{String}}()
    for path in classic
        yy = match(r"^l(\d{2})puw"i, basename(path)).captures[1]
        push!(get!(panels, yy, String[]), path)
    end
    for yy in sort(collect(keys(panels)))
        paths = sort(panels[yy])
        dirs  = unique(dirname.(paths))
        if length(dirs) > 1
            @warn "    classic panel 20$yy: waves split across directories " *
                  "($(join(dirs, ", "))) — cannot key one wave-1 dictionary to the panel; skipped."
            continue
        end
        resolved = _resolve_classic_dict(dirs[1], yy)
        isnothing(resolved) && continue
        dict_name, posmap = resolved
        @info "    classic panel 20$yy: dictionary $dict_name applied to $(length(paths)) wave(s)"
        for path in paths
            push!(work, SippWork(:classic, path, basename(path), posmap))
        end
    end

    sort!(work; by = w -> w.sortkey)
    return work
end

# Read every file into its OWN private accumulator concurrently, then merge the
# per-file accumulators into one master in the fixed work-list order. No two
# threads touch the same accumulator during the parallel phase, and the ordered
# merge makes the summed totals a deterministic function of the inputs alone
# (floating-point addition is order-sensitive, so a fixed order is what makes
# the parallel result identical to a serial fold over the same order). Per-file
# report tuples are collected positionally and printed in the same order, so the
# stage log is deterministic too.
function _sipp_accumulate_parallel(sipp_dir::String)
    work = _sipp_build_worklist(sipp_dir)
    n    = length(work)
    accs    = Vector{SippAcc}(undef, n)
    reports = Vector{Tuple{Symbol,String,Int,Int,Bool,String}}(undef, n)  # kind, fname, n_pairs, ref_year, in_window, err

    Threads.@threads for i in 1:n
        w = work[i]
        a = SippAcc()
        if w.kind === :redesign
            np, ry, inw, err = _sipp_accumulate_redesign!(a, w.path)
        else
            np, ry, inw, err = _sipp_accumulate_classic!(a, w.path, w.posmap)
        end
        accs[i]    = a
        reports[i] = (w.kind, basename(w.path), np, ry, inw, err)
    end

    master = SippAcc()
    for i in 1:n
        _sipp_merge!(master, accs[i])
        _sipp_report_file(reports[i]...)
    end
    return master
end

# Per-file stage line (Part-2 print). Three clean cases, no scary warning for a
# file that is merely out of range:
#   • out of window → INFO naming the reference year and that no window covers it;
#   • within window, parsed → INFO with the contributed stayer-pair count;
#   • within window but a real parse/column failure → the only @warn.
# A parse failure that is ALSO out of window is reported as the out-of-window
# INFO line: an unreadable file no window would have used is not a problem.
function _sipp_report_file(kind::Symbol, fname::String, n_pairs::Int,
                           ref_year::Int, in_window::Bool, err::String)
    tag = kind === :redesign ? "redesign" : "classic"
    if !in_window
        yr = ref_year > 0 ? string(ref_year) : "?"
        @info "    $fname ($tag, ref $yr): outside estimation windows — skipped (no window covers $yr)."
    elseif isempty(err)
        @info "    $fname ($tag, ref $ref_year): $n_pairs stayer pair(s) → contributed."
    else
        @warn "    $fname ($tag, ref $ref_year): $err — within an estimation window but not read; check the file."
    end
    return nothing
end

# ── Stage 6b production outputs ─────────────────────────────────────────────
# Within-job wage-change, both markets, one row per window with SIPP data.
# The SHIPPED wchg_rate_j column (what moments.jl and sigma.jl read) is the BBG
# break-filtered hazard on the classic FC windows (SIPP_BBG_WINDOWS) and the raw
# earnings-based hazard elsewhere. Two tables are printed for the reader:
#   TABLE 1 "ALL RAW"  — earnings-based hazard at the production ε
#                        (SIPP_WCHG_EPS), all four windows; this is the shipped
#                        raw value. The wider-ε robustness grid is printed
#                        separately by print_sipp_wchg_robustness.
#   TABLE 2 "SHIPPED"  — the shipped column: BBG hazard for FC windows (with the
#                        raw earnings value printed alongside, so the documented
#                        [BBG lower, raw upper] bracket is visible in one place)
#                        and the raw earnings hazard for the COVID windows.
# Windows with no SIPP contribution are omitted (they stay NaN downstream and
# auto-hold out of the objective).
#
# CSV CONTRACT (sipp_wchg_rates.csv):
#   window
#   wchg_rate_U/S        SHIPPED hazard (BBG on FC, raw earnings on COVID) — the
#                        column the SMM moment side reads unchanged
#   neff_U/S             effective N of the shipped construction (BBG Σ(T−1) on
#                        FC, earnings Kish N on COVID)
#   wchg_rate_U/S_raw    raw earnings hazard at production ε, all windows (Table 1)
#   hourly_share_U/S     BBG step-1 hourly-worker share (all windows where present)
#   bbg_underflow_U/S    1 when the FC correction forced π̃ ≤ 0 to 0 (else 0)
#   bbg_pihat_U/S        raw per-period break rate π̂ on FC (NaN off the BBG
#                        windows) — sigma.jl's delta-method variance input
#   bbg_corrfac_U/S      BBG correction factor γ̄ − ᾱ on FC (NaN elsewhere) —
#                        sigma.jl scales the π̂ variance by 1/corrfac²
function make_sipp_wchg()
    @info "Stage 6b: SIPP within-job wage-change + EE mobility (wchg_rate_U/S, ee_rate_S, ee_step_S)..."

    sipp_dir = joinpath(RAW_DIR, "sipp")
    empty_wchg() = DataFrame(window = String[],
                             wchg_rate_U = Float64[], wchg_rate_S = Float64[],
                             neff_U = Float64[], neff_S = Float64[],
                             wchg_rate_U_raw = Float64[], wchg_rate_S_raw = Float64[],
                             hourly_share_U = Float64[], hourly_share_S = Float64[],
                             bbg_underflow_U = Int[], bbg_underflow_S = Int[],
                             bbg_pihat_U = Float64[], bbg_pihat_S = Float64[],
                             bbg_corrfac_U = Float64[], bbg_corrfac_S = Float64[])
    empty_ee()   = DataFrame(window = String[], ee_rate_S = Float64[], ee_step_S = Float64[],
                             ee_step_sd = Float64[], neff_ee = Float64[], neff_step = Float64[])
    if !isdir(sipp_dir) || isempty(readdir(sipp_dir))
        @warn "  $sipp_dir absent or empty — no SIPP data yet. Writing empty " *
              "sipp_wchg_rates.csv / sipp_ee_rates.csv; every window's SIPP moment stays " *
              "NaN and auto-holds out of the SMM objective."
        w = empty_wchg();  e = empty_ee()
        CSV.write(joinpath(DERIVED_DIR, "sipp_wchg_rates.csv"), w)
        CSV.write(joinpath(DERIVED_DIR, "sipp_ee_rates.csv"),   e)
        return w
    end

    master = _sipp_accumulate_parallel(sipp_dir)
    print_sipp_deflation_basis(master.years_used)

    ip = _sipp_prod_eps_idx()
    calib = Dict{Int,NamedTuple}()   # BBG calibration memoized per distinct series length T
    wchg_rows = NamedTuple[]
    for wname in WINDOWS_ORDER
        haskey(master.wchg, wname) || continue
        c = master.wchg[wname]
        h = get(master.hshare, wname, nothing)
        # raw earnings hazard at the production ε (Table 1 / _raw columns)
        raw_U = _sipp_hazard(c.chg_U[ip], c.den_U[ip])
        raw_S = _sipp_hazard(c.chg_S[ip], c.den_S[ip])
        n_earn_U = _sipp_neff(c.den_U[ip], c.den2_U[])
        n_earn_S = _sipp_neff(c.den_S[ip], c.den2_S[])
        hs_U = isnothing(h) || h.den_U[] <= 0.0 ? NaN : h.hr_U[] / h.den_U[]
        hs_S = isnothing(h) || h.den_S[] <= 0.0 ? NaN : h.hr_S[] / h.den_S[]

        # BBG windows: replace the shipped hazard with the corrected one and
        # carry its π̂ / correction factor for the sigma diagonal.
        if wname in SIPP_BBG_WINDOWS
            cell = get(master.bbg, wname, BbgCell())
            bU = _bbg_cell_hazard(cell.U, calib)
            bS = _bbg_cell_hazard(cell.S, calib)
            ship_U, ship_S = bU.wchg, bS.wchg
            neff_U, neff_S = bU.neff, bS.neff
            uf_U   = bU.underflow ? 1 : 0
            uf_S   = bS.underflow ? 1 : 0
            pih_U, pih_S = bU.pihat, bS.pihat
            cf_U,  cf_S  = bU.corrfac, bS.corrfac
        else
            ship_U, ship_S = raw_U, raw_S
            neff_U, neff_S = n_earn_U, n_earn_S
            uf_U = uf_S = 0
            pih_U = pih_S = NaN
            cf_U  = cf_S  = NaN
        end

        push!(wchg_rows, (window = wname,
                          wchg_rate_U = ship_U, wchg_rate_S = ship_S,
                          neff_U = neff_U, neff_S = neff_S,
                          wchg_rate_U_raw = raw_U, wchg_rate_S_raw = raw_S,
                          hourly_share_U = hs_U, hourly_share_S = hs_S,
                          bbg_underflow_U = uf_U, bbg_underflow_S = uf_S,
                          bbg_pihat_U = pih_U, bbg_pihat_S = pih_S,
                          bbg_corrfac_U = cf_U, bbg_corrfac_S = cf_S))
    end

    covered = [string(r.window) for r in wchg_rows]
    heldout = [string(w) for w in WINDOWS_ORDER if !(string(w) in covered)]
    isempty(heldout) ||
        @warn "  SIPP wchg measured for [$(join(covered, ", "))]; NO SIPP data for " *
              "[$(join(heldout, ", "))] — SIPP moments held out of the SMM objective in " *
              "those windows. Download the SIPP files covering the missing calendar years " *
              "(see data-and-moments §sipp) to close the gap."

    print_sipp_wchg_tables(wchg_rows)

    wchg_out = isempty(wchg_rows) ? empty_wchg() : DataFrame(wchg_rows)
    CSV.write(joinpath(DERIVED_DIR, "sipp_wchg_rates.csv"), wchg_out)
    @info "  Saved: $(joinpath(DERIVED_DIR, "sipp_wchg_rates.csv"))"

    make_sipp_ee(master)
    print_sipp_wchg_robustness(master.wchg)
    return wchg_out
end

# Skilled EE mobility: the poach hazard ee_rate_S and the mean EE-move wage step
# ee_step_S (with its weighted sd), one row per window with skilled EE at-risk
# mass. ee_rate_S is a monthly hazard −log(1−p) consistent with the other rate
# moments; ee_step_S is the mean LOG real-weekly-wage jump E[Δlog w] on a move
# (proportional step, same scale as every other wage moment). Both carry a
# Kish neff for the delta-method variance in sigma.jl. Written to its own CSV so
# the wchg output is byte-unchanged.
function make_sipp_ee(master::SippAcc)
    rows = NamedTuple[]
    for wname in WINDOWS_ORDER
        haskey(master.ee, wname) || continue
        e = master.ee[wname]
        rate    = _sipp_hazard(e.chg_ee[], e.den_ee[])
        neff_ee = _sipp_neff(e.den_ee[], e.den2_ee[])
        dstep   = e.den_step[]
        mean_st = dstep > 0.0 ? e.sum_step[] / dstep : NaN
        var_st  = dstep > 0.0 ? max(e.sum_step2[] / dstep - mean_st^2, 0.0) : NaN
        sd_st   = isnan(var_st) ? NaN : sqrt(var_st)
        neff_st = _sipp_neff(dstep, e.den2_step[])
        push!(rows, (window = wname, ee_rate_S = rate, ee_step_S = mean_st,
                     ee_step_sd = sd_st, neff_ee = neff_ee, neff_step = neff_st))
        @printf("  %-14s ee_rate_S=%.5f  ee_step_S=%.3f (sd=%.3f)  (neff_ee=%.1f, neff_step=%.1f)\n",
                string(wname), rate, mean_st, sd_st, neff_ee, neff_st)
    end
    out = isempty(rows) ?
        DataFrame(window = String[], ee_rate_S = Float64[], ee_step_S = Float64[],
                  ee_step_sd = Float64[], neff_ee = Float64[], neff_step = Float64[]) :
        DataFrame(rows)
    CSV.write(joinpath(DERIVED_DIR, "sipp_ee_rates.csv"), out)
    @info "  Saved: $(joinpath(DERIVED_DIR, "sipp_ee_rates.csv"))"
    return out
end

# ── Deflation basis report ──────────────────────────────────────────────────
# Print the 2013 CPI-U base ONCE, then the CPI-U index for each calendar year
# actually deflated in this run (the years of the loaded records), ascending.
# Replaces the former per-record "CPI-U x → 2013 base y" line: the base is a
# constant, and the per-year indices are the same for every record of a year.
function print_sipp_deflation_basis(years_used::Set{Int})
    @printf("  SIPP deflation base: CPI-U(%d) = %.3f\n", SIPP_CPIU_BASE_YEAR, SIPP_CPIU_BASE)
    isempty(years_used) && return nothing
    parts = ["$(y)=$(@sprintf("%.3f", sipp_cpi(y)))" for y in sort(collect(years_used))]
    println("  CPI-U by year used: ", join(parts, ", "))
    return nothing
end

# ── Shipped-vs-raw tables (wchg) ────────────────────────────────────────────
# TABLE 1 "ALL RAW": earnings-based hazard, all four windows × {U, S}, at the
# production ε. TABLE 2 "SHIPPED": the shipped column (BBG on FC, raw on COVID),
# with the raw earnings value printed alongside on the FC windows so the
# documented [BBG lower, raw upper] bracket is visible, and the BBG effective N /
# hourly share / underflow flag. `rows` are the per-window NamedTuples built in
# make_sipp_wchg (already in WINDOWS_ORDER, windows without SIPP omitted).
function print_sipp_wchg_tables(rows::Vector{NamedTuple})
    isempty(rows) && return nothing
    println("\n  ══ TABLE 1 — ALL RAW (earnings-based within-job wage-change hazard, ε = \$$(SIPP_WCHG_EPS)/week) ══")
    @printf("  %-14s %12s %12s %10s %10s\n", "window", "wchg_U", "wchg_S", "neff_U", "neff_S")
    for r in rows
        # neff shown here is the shipped construction's N; for COVID it is the
        # earnings Kish N, for FC it is the BBG Σ(T−1) (the raw earnings N is the
        # same order and not separately shipped).
        @printf("  %-14s %12.5f %12.5f %10.1f %10.1f\n",
                string(r.window), r.wchg_rate_U_raw, r.wchg_rate_S_raw, r.neff_U, r.neff_S)
    end

    println("\n  ══ TABLE 2 — SHIPPED (BBG-classic + raw-redesign; wchg_rate_U/S in sipp_wchg_rates.csv) ══")
    @printf("  %-14s %12s %12s %10s %10s %8s %8s\n",
            "window", "wchg_U", "wchg_S", "neff_U", "neff_S", "hshr_U", "hshr_S")
    for r in rows
        bbg = r.window in SIPP_BBG_WINDOWS
        tag = bbg ? "BBG " : "raw "
        @printf("  %-10s%4s %12.5f %12.5f %10.1f %10.1f %8.3f %8.3f\n",
                string(r.window), tag, r.wchg_rate_U, r.wchg_rate_S,
                r.neff_U, r.neff_S, r.hourly_share_U, r.hourly_share_S)
        if bbg
            # [BBG lower, raw upper] bracket + correction diagnostics on one line
            @printf("  %-14s   raw earnings: U=%.5f S=%.5f │ π̂: U=%.4f S=%.4f │ γ̄−ᾱ: U=%.4f S=%.4f%s\n",
                    "", r.wchg_rate_U_raw, r.wchg_rate_S_raw,
                    r.bbg_pihat_U, r.bbg_pihat_S, r.bbg_corrfac_U, r.bbg_corrfac_S,
                    (r.bbg_underflow_U == 1 || r.bbg_underflow_S == 1) ?
                        "  [underflow: π̃≤0 forced to 0]" : "")
        end
    end
    println("  Table 2 is the shipped target: BBG break-filtered hazard (lower bound) on the")
    println("  classic FC windows, raw earnings hazard on the redesign COVID windows.")
    println()
    return nothing
end

# ── Threshold-robustness diagnostic (wchg) ──────────────────────────────────
# Earnings-based wage-change hazard for all four windows × {U, S} at every ε in
# the grid. Diagnostic only; the shipped raw value is the ε = SIPP_WCHG_EPS
# column. A window with no SIPP contribution shows NaN throughout.
function print_sipp_wchg_robustness(acc::Dict{Symbol,WchgCounts})
    println("\n  ── wchg threshold robustness — earnings-based (monthly hazard −log(1−p); shipped raw ε = \$$(SIPP_WCHG_EPS)/week) ──")
    @printf("  %-14s", "window")
    for ε in SIPP_WCHG_EPS_GRID
        @printf(" %8s %8s", @sprintf("U ε%g", ε), @sprintf("S ε%g", ε))
    end
    println()
    for wname in WINDOWS_ORDER
        @printf("  %-14s", string(wname))
        c = get(acc, wname, nothing)
        for k in eachindex(SIPP_WCHG_EPS_GRID)
            if isnothing(c)
                @printf(" %8s %8s", "NaN", "NaN")
            else
                @printf(" %8.5f %8.5f",
                        _sipp_hazard(c.chg_U[k], c.den_U[k]),
                        _sipp_hazard(c.chg_S[k], c.den_S[k]))
            end
        end
        println()
    end
    println()
    return nothing
end

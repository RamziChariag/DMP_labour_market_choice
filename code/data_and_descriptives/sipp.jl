############################################################
# code/data_and_descriptives/sipp.jl
#
# SIPP job-mobility moments per estimation window:
#   wchg_rate_U, wchg_rate_S  within-job wage-change hazard (rate at which a
#                             continuing match re-prices its wage)
#   ee_rate_S                 skilled employer-to-employer hazard (comparison only;
#                             the shipped target comes from J2J)
#   ee_step_S                 mean log real weekly wage step on a skilled EE move
# EE moments are skilled only: the unskilled market has no on-the-job search.
#
# The wage-change construction depends on the panel era (SIPP_BBG_WINDOWS):
# on the classic 2001/2004/2008 panels (FC windows) the reported earnings are
# too noisy, so the Barattieri–Basu–Gottschalk (2014) break filter is run on the
# hourly-rate series and corrected for test size and power (a lower bound; the
# raw earnings value is reported as the upper bound). On the 2014+ redesign
# (COVID windows) it is the raw earnings change |Δ real weekly wage| > ε.
#
# Inputs, found by a recursive walk of data/raw/sipp/:
#   pu<Y>.csv, pu2014w<N>.csv       redesign cores (comma-delimited)
#   l<YY>puw<N>.dat                 classic cores (fixed width), laid out by the
#   l<YY>puw1d.txt[.old2|.old.2]    panel's wave-1 core dictionary (Census ships
#                                   the core dictionary for wave 1 only)
# Compressed cores without an extracted sibling are skipped with a hint.
#
# Entry point: make_sipp_wchg()  → sipp_wchg_rates.csv, sipp_ee_rates.csv
#
# Files are read in parallel (Threads.@threads), each into its own accumulator,
# and merged serially in a fixed sorted order, so the result is bit-identical to
# a serial run.
# Definitions only; packages and path constants come from data_processing_main.jl.
############################################################

# ── BBG break filter ────────────────────────────────────────────────────────
# A reported wage-rate series is a step function (the true wage) plus
# measurement error. Steps are detected by a sequential sup-F (Bai–Perron)
# test; the detection rate π̂ is then corrected for size α and power γ,
# π̃ = (π̂ − α)/(γ − α) (BBG eq. 6), both from a Monte-Carlo calibration.

"""
    _supF(y, lo, hi, htrim) -> (maxF, k)

Largest one-break mean-shift F statistic over the interior splits of
`y[lo:hi]` (at least `htrim` points per side) and its split index; `(0.0, 0)`
for a segment too short or constant.
"""
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
        # One break (2 vs 1 parameters): (ssr0 − ssr1) / (ssr1/(n − 2))
        F = ssr1 > 0 ? ((ssr0 - ssr1)) / (ssr1 / (n - 2)) : 0.0
        F > bestF && (bestF = F; bestk = k)
    end
    return (bestF, bestk)
end

"""
    count_breaks(y; crit, htrim = 1) -> Int

Number of mean-shift breaks in `y` found by sequential binary splitting: a
segment is split at its sup-F point whenever the statistic exceeds `crit`.
"""
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

"""
    calibrate(T; alpha = 0.05, rho = 0.482, s2n = 2.64, break_size_sd = 1.0,
              nsim = 4000, htrim = 1, seed = 1234) -> (crit, gamma)

Monte-Carlo calibration for series of length `T` under AR(1) measurement error
(autocorrelation `rho`, signal-to-noise `s2n`): `crit` is the 1 − `alpha`
quantile of the null sup-F, `gamma` the power against one mid-series break of
`break_size_sd` signal sds. Seeded with `seed + T`, so deterministic.
"""
function calibrate(T::Int; alpha::Float64=0.05, rho::Float64=0.482, s2n::Float64=2.64,
                   break_size_sd::Float64=1.0, nsim::Int=4000, htrim::Int=1, seed::Int=1234)
    rng = MersenneTwister(seed + T)
    sig_e = 1.0                          # error sd (F is scale-invariant)
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

"""
    calibrate_perperiod(T; crit, rho = 0.482, s2n = 2.64, π_plant = 0.05,
                        nsim = 4000, htrim = 1, seed = 4321) -> (alpha_eff, gamma_eff)

Per-period size and power of `count_breaks` at critical value `crit` for series
of length `T`: `alpha_eff` is false breaks per period under the null, `gamma_eff`
detected breaks per true step when each period carries a step with probability
`π_plant`. Both are per period, the unit BBG eq. (6) needs. Seeded with `seed + T`.
"""
function calibrate_perperiod(T::Int; crit::Float64, rho::Float64=0.482, s2n::Float64=2.64,
                             π_plant::Float64=0.05, nsim::Int=4000, htrim::Int=1, seed::Int=4321)
    rng = MersenneTwister(seed + T)
    sig_signal = sqrt(s2n)
    ar!(v) = (v[1]=randn(rng); for t in 2:T; v[t]=rho*v[t-1]+sqrt(1-rho^2)*randn(rng); end)
    e = Vector{Float64}(undef,T); false_b = 0
    for s in 1:nsim; ar!(e); false_b += count_breaks(e; crit=crit, htrim=htrim); end
    alpha_eff = false_b / (nsim*(T-1))
    y = Vector{Float64}(undef,T); det_true = 0; true_periods = 0
    for s in 1:nsim
        ar!(e); lvl=0.0; truebk=Int[]
        for t in 1:T
            if t>1 && rand(rng)<π_plant; lvl += sig_signal*sign(randn(rng))*(0.8+0.4*rand(rng)); push!(truebk,t); end
            y[t]=lvl+e[t]
        end
        isempty(truebk) && continue
        # Hits are capped at the number of true steps; break locations are not matched.
        nb = count_breaks(y; crit=crit, htrim=htrim)
        det_true += min(nb, length(truebk)); true_periods += length(truebk)
    end
    gamma_eff = true_periods>0 ? det_true/true_periods : NaN
    return (alpha_eff=alpha_eff, gamma_eff=gamma_eff)
end

"""BBG signal-to-noise ratio (Gottschalk–Huynh); Altonji–Kang's 1.80 is the robustness alternative."""
const SIPP_BBG_S2N  = 2.64
"""BBG measurement-error AR(1) autocorrelation."""
const SIPP_BBG_RHO  = 0.482
"""Monte-Carlo replications per distinct series length."""
const SIPP_BBG_NSIM = 3000

"""
    SIPP_BBG_WINDOWS

Windows measured on the classic panels, where the shipped wage-change hazard is
the BBG one; every other window ships the raw earnings hazard. The single
control for the era split.
"""
const SIPP_BBG_WINDOWS = (:base_fc, :crisis_fc)

# ── CPI-U deflator ─────────────────────────────────────────────────────────

"""
    SIPP_CPIU

BLS CPI-U, US city average, annual average (CUUR0000SA0, 1982–84 = 100). Public
SIPP has no CPI99, so earnings are deflated with this series to 2013 dollars,
the ASEC base.
"""
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
"""Base year of the SIPP real wages."""
const SIPP_CPIU_BASE_YEAR = 2013
"""CPI-U annual average in the base year."""
const SIPP_CPIU_BASE       = 232.957

"""
    sipp_cpi(year) -> Float64

CPI-U index for a calendar year. Errors for a year absent from `SIPP_CPIU`: it
is called only for pairs inside a window, so a missing year is a
misconfiguration, never a record to deflate with a missing index.
"""
function sipp_cpi(year::Int)::Float64
    haskey(SIPP_CPIU, year) && return SIPP_CPIU[year]
    lo, hi = extrema(keys(SIPP_CPIU))
    error("SIPP deflation: no CPI-U index for calendar year $year (SIPP_CPIU " *
          "covers $(lo)–$(hi)). A window in setup.jl references $year; add the BLS " *
          "CUUR0000SA0 annual average for $year to SIPP_CPIU in sipp.jl.")
end

# ── Earnings-based wage-change threshold ───────────────────────────────────

"""Thresholds ε (dollars per week) at which the earnings-based wage change is measured; all but the production one are a diagnostic."""
const SIPP_WCHG_EPS_GRID = [1.0, 5.0, 10.0, 20.0]
"""Production threshold ε (dollars per week); must be on `SIPP_WCHG_EPS_GRID`."""
const SIPP_WCHG_EPS       = 1.0

# ── Cell coercion ──────────────────────────────────────────────────────────

"""
    _sipp_to_float(v) -> Float64

A CSV or fixed-width cell (number, string or missing) as Float64; NaN if blank
or unparseable.
"""
function _sipp_to_float(v)::Float64
    ismissing(v) && return NaN
    v isa Real && return Float64(v)
    s = strip(string(v))
    isempty(s) && return NaN
    p = tryparse(Float64, s)
    return isnothing(p) ? NaN : p
end

"""
    _sipp_to_int(v) -> Union{Int,Nothing}

A cell rounded to Int, or `nothing` if missing, blank or unparseable (callers
then skip the record).
"""
function _sipp_to_int(v)::Union{Int,Nothing}
    f = _sipp_to_float(v)
    (isnan(f) || !isfinite(f)) && return nothing
    return round(Int, f)
end

"""
    sipp_real_weekly_wage(msum, cal_year, cal_month, cpi_ref) -> Float64

Real 2013-dollar weekly wage from a month's earnings: monthly sum × 7 / days in
the month, deflated by `cpi_ref`. The day-count normalisation precedes
differencing, so month length alone never registers as a wage change.
"""
function sipp_real_weekly_wage(msum::Float64, cal_year::Int, cal_month::Int,
                               cpi_ref::Float64)::Float64
    nominal_week = msum * 7.0 / Dates.daysinmonth(cal_year, cal_month)
    return deflate_wage(nominal_week, cpi_ref, SIPP_CPIU_BASE)
end

# ── Hourly pay rate (BBG series) ─────────────────────────────────────────────
# The BBG filter runs on the NOMINAL reported hourly rate: break structure is
# invariant to monotone deflation (earnings, which are differenced, are deflated).

"""Classic `TPYRATE{n}` carries two implied decimals."""
const SIPP_CLASSIC_RATE_SCALE = 100.0
"""Classic `EJBHRS{n}` code for "hours vary"; such a slot is excluded from the rate series."""
const SIPP_CLASSIC_HOURS_VARY = -8

# ── Skilled classification ─────────────────────────────────────────────────

"""Skilled on the redesign `EEDUC` scale: ≥ 43 (bachelor's or higher)."""
is_skilled_sipp(eeduc::Integer)::Bool = eeduc >= 43

"""
Skilled on the classic `EEDUCATE` scale: ≥ 44 (bachelor's or higher; 43 is an
associate degree). Never share a classifier with the redesign scale.
"""
is_skilled_sipp_classic(eeducate::Integer)::Bool = eeducate >= 44

# ── Accumulators ────────────────────────────────────────────────────────────

"""
    WchgCounts(eps)

Earnings-based within-job wage-change tallies for one window at every threshold
in `eps`, per market: `den_j` the weighted stayer pairs at risk, `chg_j` those
with |Δw| > ε, `den2_j` the ε-invariant Σw² (Kish N = den²/den2).
"""
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

"""
    _wchg_record_pair!(c, skilled, wgt, w1, w2)

Add one stayer pair (same job, consecutive months) with weekly wages `w1`, `w2`
and weight `wgt` to every ε cell of its market.
"""
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

"""
    _wchg_merge!(a, b)

Add `b`'s tallies into `a` (serial merge only).
"""
function _wchg_merge!(a::WchgCounts, b::WchgCounts)
    a.den_U .+= b.den_U;  a.chg_U .+= b.chg_U
    a.den_S .+= b.den_S;  a.chg_S .+= b.chg_S
    a.den2_U[] += b.den2_U[];  a.den2_S[] += b.den2_S[]
end

"""
    HourlyShare()

Weighted employed person-months per market (`den_j`) and those whose main job is
paid by the hour (`hr_j`): how much the hourly restriction thins the BBG cell.
"""
mutable struct HourlyShare
    den_U :: Base.RefValue{Float64}
    hr_U  :: Base.RefValue{Float64}
    den_S :: Base.RefValue{Float64}
    hr_S  :: Base.RefValue{Float64}
end
HourlyShare() = HourlyShare(Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0))

"""
    _hshare_merge!(a, b)

Add `b`'s tallies into `a` (serial merge only).
"""
function _hshare_merge!(a::HourlyShare, b::HourlyShare)
    a.den_U[] += b.den_U[];  a.hr_U[] += b.hr_U[]
    a.den_S[] += b.den_S[];  a.hr_S[] += b.hr_S[]
end

"""
    _hshare_record!(h, skilled, wgt, is_hourly)

Add one employed person-month of weight `wgt` to its market's tally.
"""
function _hshare_record!(h::HourlyShare, skilled::Bool, wgt::Float64, is_hourly::Bool)
    (isfinite(wgt) && wgt > 0.0) || return
    if skilled
        h.den_S[] += wgt;  is_hourly && (h.hr_S[] += wgt)
    else
        h.den_U[] += wgt;  is_hourly && (h.hr_U[] += wgt)
    end
    return nothing
end

"""
    EeCounts()

Skilled EE tallies for one window: the at-risk and moving weights with Σw² (the
EE hazard and its Kish N), and over movers Σw, Σw·Δlog w, Σw·(Δlog w)² and Σw²
(the mean log wage step, its weighted sd and Kish N).
"""
mutable struct EeCounts
    den_ee    :: Base.RefValue{Float64}   # Σw at-risk employed-skilled pairs
    chg_ee    :: Base.RefValue{Float64}   # Σw pairs whose main employer changed
    den2_ee   :: Base.RefValue{Float64}   # Σw² at risk
    den_step  :: Base.RefValue{Float64}   # Σw over movers
    sum_step  :: Base.RefValue{Float64}   # Σ w·Δlog w
    sum_step2 :: Base.RefValue{Float64}   # Σ w·(Δlog w)²
    den2_step :: Base.RefValue{Float64}   # Σw² over movers
end
EeCounts() = EeCounts(Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0))

"""
    _ee_merge!(a, b)

Add `b`'s tallies into `a` (serial merge only).
"""
function _ee_merge!(a::EeCounts, b::EeCounts)
    a.den_ee[]    += b.den_ee[];    a.chg_ee[]    += b.chg_ee[]
    a.den2_ee[]   += b.den2_ee[];   a.den_step[]  += b.den_step[]
    a.sum_step[]  += b.sum_step[];  a.sum_step2[] += b.sum_step2[]
    a.den2_step[] += b.den2_step[]
end

"""
    BbgCell()

Nominal hourly-rate readings for one BBG window, per market: job id →
(calendar month, rate). Fragments from different wave files are concatenated
and only split into contiguous runs in `_bbg_cell_hazard`, so a job's full
within-window series is used.
"""
mutable struct BbgCell
    U :: Dict{String,Vector{Tuple{Int,Float64}}}
    S :: Dict{String,Vector{Tuple{Int,Float64}}}
end
BbgCell() = BbgCell(Dict{String,Vector{Tuple{Int,Float64}}}(),
                    Dict{String,Vector{Tuple{Int,Float64}}}())

"""
    _bbg_merge!(a, b)

Append `b`'s readings to `a`, job by job (serial merge only).
"""
function _bbg_merge!(a::BbgCell, b::BbgCell)
    for (jid, obs) in b.U; append!(get!(() -> Tuple{Int,Float64}[], a.U, jid), obs); end
    for (jid, obs) in b.S; append!(get!(() -> Tuple{Int,Float64}[], a.S, jid), obs); end
end

"""
    _bbg_record!(cell, skilled, jid, cal_m, rate)

Append one positive finite hourly-rate reading to job `jid`'s series.
"""
function _bbg_record!(cell::BbgCell, skilled::Bool, jid::String, cal_m::Int, rate::Float64)
    (isfinite(rate) && rate > 0.0) || return
    d = skilled ? cell.S : cell.U
    push!(get!(() -> Tuple{Int,Float64}[], d, jid), (cal_m, rate))
    return nothing
end

"""
    SippAcc()

One file's private, window-keyed accumulator: wage-change counts, hourly share
and EE counts (all windows), BBG rate series (BBG windows only), and the
calendar years deflated. Never shared between threads.
"""
struct SippAcc
    wchg       :: Dict{Symbol,WchgCounts}
    hshare     :: Dict{Symbol,HourlyShare}
    ee         :: Dict{Symbol,EeCounts}
    bbg        :: Dict{Symbol,BbgCell}
    years_used :: Set{Int}
end
SippAcc() = SippAcc(Dict{Symbol,WchgCounts}(), Dict{Symbol,HourlyShare}(),
                    Dict{Symbol,EeCounts}(), Dict{Symbol,BbgCell}(), Set{Int}())

"""Window `w`'s wage-change counts in `acc`, created if absent."""
_wchg_get!(acc::SippAcc, w::Symbol)   = get!(() -> WchgCounts(SIPP_WCHG_EPS_GRID), acc.wchg, w)
"""Window `w`'s hourly-share tally in `acc`, created if absent."""
_hshare_get!(acc::SippAcc, w::Symbol) = get!(() -> HourlyShare(), acc.hshare, w)
"""Window `w`'s EE counts in `acc`, created if absent."""
_ee_get!(acc::SippAcc, w::Symbol)     = get!(() -> EeCounts(), acc.ee, w)
"""Window `w`'s BBG rate series in `acc`, created if absent."""
_bbg_get!(acc::SippAcc, w::Symbol)    = get!(() -> BbgCell(), acc.bbg, w)

"""
    _sipp_merge!(master, f)

Add file accumulator `f` into `master`. Called in the fixed file order, so the
merged totals do not depend on thread scheduling.
"""
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

# ── Person-month records ────────────────────────────────────────────────────

"""
    JobSlot

One non-empty job slot of a person-month: person-namespaced job `id`, monthly
earnings `msum`, whether the job is paid by the hour (`hourly`), and its nominal
hourly `rate` (NaN when not hourly or unusable).
"""
struct JobSlot
    id     :: String
    msum   :: Float64
    hourly :: Bool
    rate   :: Float64
end

"""
    PMRecord

A parsed person-month from either format; `skilled` is resolved with the
format's own classifier so the processing below is format-blind.
"""
struct PMRecord
    cal_m   :: Int                      # yr*12 + mn
    yr      :: Int
    mn      :: Int
    employed:: Bool                     # RMESR == 1
    inage   :: Bool                     # 16 ≤ TAGE ≤ 64
    skilled :: Bool
    wgt     :: Float64                  # WPFINWGT
    jobs    :: Vector{JobSlot}
end

"""
    _main_job(rec) -> Union{JobSlot,Nothing}

The month's highest-earning job, or `nothing` when no job is held.
"""
function _main_job(rec::PMRecord)
    isempty(rec.jobs) && return nothing
    best = rec.jobs[1]
    @inbounds for k in 2:length(rec.jobs)
        rec.jobs[k].msum > best.msum && (best = rec.jobs[k])
    end
    return best
end

"""
    _process_person!(acc, recs) -> Int

Add one person's month records to `acc`. Over consecutive months in which the
person is employed and of working age (routed to the window of the later month,
whose weight and skill are used): every job id held in both months is a stayer
pair for the earnings-based wage change; for the skilled, the pair is at risk of
an EE move, which occurs when the main-job id changes, and a move contributes
its log weekly wage step. Separately, each employed in-age person-month in a
window adds to the hourly share and, in a BBG window, its hourly main job's rate
to that job's series. Returns the number of in-window pairs.
"""
function _process_person!(acc::SippAcc, recs::Vector{PMRecord})::Int
    length(recs) < 2 && return 0
    sort!(recs; by = r -> r.cal_m)
    n_win_pairs = 0
    @inbounds for i in firstindex(recs):lastindex(recs)-1
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

        # (a) within-job wage change, one stayer pair per job id held in both months
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

        # (b), (c) EE hazard and log wage step, skilled only
        skilled || continue
        mj1 = _main_job(r1);  mj2 = _main_job(r2)
        (isnothing(mj1) || isnothing(mj2)) && continue
        ee = _ee_get!(acc, w)
        ee.den_ee[]  += wgt
        ee.den2_ee[] += wgt * wgt
        if mj1.id != mj2.id
            ee.chg_ee[] += wgt
            w_old = sipp_real_weekly_wage(mj1.msum, r1.yr, r1.mn, cpi1)
            w_new = sipp_real_weekly_wage(mj2.msum, r2.yr, r2.mn, cpi2)
            if isfinite(w_old) && isfinite(w_new) && w_old > 0.0 && w_new > 0.0
                step = log(w_new) - log(w_old)
                ee.den_step[]  += wgt
                ee.sum_step[]  += wgt * step
                ee.sum_step2[] += wgt * step * step
                ee.den2_step[] += wgt * wgt
            end
        end
    end

    # Person-month pass: hourly share, and the BBG rate series.
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

# ── Hazards ─────────────────────────────────────────────────────────────────

"""
    _sipp_hazard(chg, den) -> Float64

Monthly hazard −log(1 − p) of the proportion p = chg/den (capped at 1 − 1e-12),
matching the model's Poisson rates; NaN for a non-positive denominator.
"""
function _sipp_hazard(chg::Float64, den::Float64)::Float64
    den > 0.0 || return NaN
    p = min(chg / den, 1.0 - 1e-12)
    return p > 0.0 ? -log(1.0 - p) : 0.0
end

"""Kish effective sample size den²/den2 (0 if den2 ≤ 0)."""
_sipp_neff(den::Float64, den2::Float64)::Float64 = den2 > 0.0 ? den^2 / den2 : 0.0

"""Index of `SIPP_WCHG_EPS` on `SIPP_WCHG_EPS_GRID`."""
_sipp_prod_eps_idx()::Int = findfirst(==(SIPP_WCHG_EPS), SIPP_WCHG_EPS_GRID)

"""
    _bbg_cell_hazard(series_by_job, calib) -> NamedTuple

BBG wage-change hazard of one market in one BBG window. Each job's readings are
split at calendar-month gaps into contiguous runs; every run of length T ≥ 3 is
break-counted at its length's critical value. π̂ = Σ breaks / Σ(T − 1), and
π̃ = (π̂ − ᾱ)/(γ̄ − ᾱ) with ᾱ, γ̄ the (T − 1)-weighted per-period size and power.
Calibrations are memoised per T in `calib`. Returns
- `pihat`: raw per-period break rate;
- `pitil`: corrected probability (0 on underflow);
- `corrfac`: γ̄ − ᾱ (sampling_variances.jl divides the π̂ variance by its square);
- `wchg`: −log(1 − π̃) with π̃ clamped to [0, 1 − 1e-12];
- `neff`: Σ(T − 1);
- `underflow`: π̃ ≤ 0 or γ̄ ≤ ᾱ, forced to 0;
- `nseries`: runs used (0 ⇒ all NaN).
"""
function _bbg_cell_hazard(series_by_job::Dict{String,Vector{Tuple{Int,Float64}}},
                          calib::Dict{Int,NamedTuple})::NamedTuple
    tot_breaks = 0
    tot_periods = 0            # Σ(T−1)
    sum_alpha_w = 0.0          # Σ (T−1)·α_eff(T)
    sum_gamma_w = 0.0          # Σ (T−1)·γ_eff(T)
    nseries = 0
    for obs in values(series_by_job)
        length(obs) < 3 && continue
        sort!(obs; by = first)
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

# ── Redesign format (2014+, comma-delimited) ────────────────────────────────

"""
    sipp_redesign_ref_year(fname) -> Int

Reference calendar year of a redesign file: `pu2014w<N>` → 2012 + N; annual
`pu<YYYY>` → YYYY − 1. MONTHCODE then gives the month within that year.
"""
function sipp_redesign_ref_year(fname::String)::Int
    m = match(r"^pu(\d{4})w(\d+)"i, fname)
    isnothing(m) || return 2012 + parse(Int, m.captures[2])
    m = match(r"^pu(\d{4})"i, fname)
    isnothing(m) || return parse(Int, m.captures[1]) - 1
    error("sipp_redesign_ref_year: cannot parse reference year from '$fname'")
end

"""Redesign `EJB{n}_PAYHR1` code for "pay per hour"."""
const REDESIGN_HOURLY_CODE = 2
"""Number of job slots (EJB1–EJB7) in the redesign core."""
const REDESIGN_JOB_SLOTS   = 7

"""
    _sipp_redesign_slot_fields(n) -> NTuple{4,Symbol}

Redesign column names read for job slot `n`: job id, monthly earnings, pay type
and hourly rate. Feeds `REDESIGN_SELECT`.
"""
_sipp_redesign_slot_fields(n::Int) = (Symbol("EJB$(n)_JOBID"), Symbol("TJB$(n)_MSUM"),
    Symbol("EJB$(n)_PAYHR1"), Symbol("TJB$(n)_HOURLY1"))

"""
    _sipp_redesign_job_records(row, pidkey, n_slots) -> Vector{JobSlot}

Filled job slots of a redesign row (non-blank, non-"0" `EJB{n}_JOBID` with
positive `TJB{n}_MSUM`), ids namespaced by `pidkey`. A slot paid by the hour
carries its `TJB{n}_HOURLY1` rate when positive, else NaN.
"""
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

"""Person-month columns a redesign core must contain."""
const REDESIGN_REQUIRED = [:SSUID, :PNUM, :MONTHCODE, :RMESR, :EEDUC, :TAGE, :WPFINWGT]

"""
    REDESIGN_SELECT

Every column the redesign reader touches, passed to `CSV.Rows(select=)` so only
these are tokenised out of the 5000+ in an annual file. Absent columns are
simply not selected; the `REDESIGN_REQUIRED` check still catches a missing one.
"""
const REDESIGN_SELECT = vcat(REDESIGN_REQUIRED,
    collect(Iterators.flatten(_sipp_redesign_slot_fields(n) for n in 1:REDESIGN_JOB_SLOTS)))

"""
    _sipp_accumulate_redesign!(acc, path) -> (n_pairs, ref_year, in_window, err)

Stream one redesign core row by row into `acc`, flushing each person when
(SSUID, PNUM) changes (rows are grouped by person). `in_window` is whether any
month of the reference year lies in a window; `err` is non-empty if a required
column is missing.
"""
function _sipp_accumulate_redesign!(acc::SippAcc, path::String)
    fname    = basename(path)
    ref_year = sipp_redesign_ref_year(fname)
    in_window = any(assign_window(ref_year, mc) != :none for mc in 1:12)

    cur_pid = ""
    recs    = PMRecord[]
    n_pairs = Ref(0)
    checked = false
    flush!() = (n_pairs[] += _process_person!(acc, recs); empty!(recs))

    # String ids never overflow or gain a decimal point; reusebuffer keeps one
    # row in memory on multi-GB files.
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

# ── Classic format (2001/2004/2008, fixed width) ────────────────────────────

"""
    parse_classic_dict(path) -> Dict{String,Tuple{Int,Int}}

Field layout name → (start, width), 1-indexed, from a classic core dictionary's
`D NAME width start` lines; the first line per name is the core definition.
"""
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

"""
    _classic_field(line, posmap, name) -> Union{SubString,Nothing}

Stripped fixed-width field `name` of a record; `nothing` if unknown, out of
range or blank.
"""
function _classic_field(line::AbstractString, posmap, name::String)
    haskey(posmap, name) || return nothing
    sp, w = posmap[name]
    stop = sp + w - 1
    stop <= lastindex(line) || return nothing
    s = strip(SubString(line, sp, stop))
    return isempty(s) ? nothing : s
end

"""
    _sipp_warn_compressed(path)

Warn, with the gunzip/unzip command, about a compressed core that has no
extracted sibling; silent if the sibling exists. Never decompresses (no codec
dependency).
"""
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

"""Classic `EPAYHR{n}` code for "paid by the hour"."""
const CLASSIC_HOURLY_CODE = 1

"""
    _sipp_classic_job_records(line, posmap, pidkey) -> Vector{JobSlot}

Filled job slots (two per record) of a classic record: employer number
`EENO{n}` ≠ 0 with positive earnings `TPMSUM{n}`, ids namespaced by `pidkey`.
An hourly slot (`EPAYHR{n}` = 1) carries its rate `TPYRATE{n}`/100 only when the
rate is positive and the usual hours `EJBHRS{n}` are usable (not −8, "hours
vary"); the hours gate inclusion only. No imputation.
"""
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

"""Fields a classic core dictionary must define for the panel to be read."""
const CLASSIC_REQUIRED = ["SSUID", "EPPPNUM", "RHCALYR", "RHCALMN",
                          "WPFINWGT", "EEDUCATE", "RMESR",
                          "EENO1", "EENO2", "TPMSUM1", "TPMSUM2"]

"""
    sipp_classic_panel_year(fname) -> Union{Int,Nothing}

Panel year 2000 + YY of a classic core `l<YY>puw<N>.dat`.
"""
function sipp_classic_panel_year(fname::AbstractString)
    m = match(r"l(\d{2})puw\d+\.dat"i, fname)
    isnothing(m) && return nothing
    return 2000 + parse(Int, m.captures[1])
end

"""
    _sipp_accumulate_classic!(acc, dat_path, posmap) -> (n_pairs, data_year, in_window, err)

Read one classic wave line by line with the panel's wave-1 layout `posmap`,
group records by person, and process persons in sorted id order (deterministic
regardless of Dict order). The wave is skipped (`err` set) if its records are
shorter than the rightmost required field, i.e. the layout does not match.
`data_year` is the first calendar year present in the records and `in_window`
is whether any record month lies in a window.
"""
function _sipp_accumulate_classic!(acc::SippAcc, dat_path::String, posmap)
    fname = basename(dat_path)
    panel = sipp_classic_panel_year(fname)

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

    n_pairs = 0
    for pid in sort(collect(keys(by_pid)))
        n_pairs += _process_person!(acc, by_pid[pid])
    end

    yrs = sort(unique(r.yr for r in Iterators.flatten(values(by_pid))))
    data_year = isempty(yrs) ? ref_year : first(yrs)
    in_window = any(assign_window(r.yr, r.mn) != :none
                    for r in Iterators.flatten(values(by_pid)))
    return (n_pairs, data_year, in_window, "")
end

"""
    _resolve_classic_dict(dict_dir, panel_yy) -> Union{(name, posmap), Nothing}

The panel's wave-1 core dictionary from the directory of its cores: tries
`l<YY>puw1d.txt`, then `.old2`, then `.old.2` (the 2008 panel's surviving
revision), accepting the first that defines every `CLASSIC_REQUIRED` field.
Returns `nothing` with a warning if none does.
"""
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

# ── Parallel scan and deterministic merge ───────────────────────────────────

"""
    SippWork

One file to read: `kind` (`:redesign` or `:classic`), `path`, the merge-order
key `sortkey` (basename), and the classic layout `posmap` (`nothing` for redesign).
"""
struct SippWork
    kind   :: Symbol
    path   :: String
    sortkey:: String
    posmap :: Union{Nothing,Dict{String,Tuple{Int,Int}}}
end

"""
    _sipp_build_worklist(sipp_dir) -> Vector{SippWork}

Walk `sipp_dir` recursively for redesign (`pu*.csv`) and classic
(`l<YY>puw<N>.dat`) cores, reporting compressed ones. Classic waves are grouped
by panel, which must sit in one directory, and share the panel's resolved
wave-1 dictionary; panels without one are skipped. Sorted by basename.
"""
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

"""
    _sipp_accumulate_parallel(sipp_dir) -> SippAcc

Read every file of the work list concurrently, each into its own `SippAcc`,
then merge them and print their report lines serially in work-list order.
Floating-point sums depend on order, so the fixed order is what makes the
result identical to a serial run.
"""
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

"""
    _sipp_report_file(kind, fname, n_pairs, ref_year, in_window, err)

One log line per file: info if outside every window (even if unreadable) or
read successfully; a warning only for an in-window file that failed to read.
"""
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

# ── Outputs ─────────────────────────────────────────────────────────────────

"""
    make_sipp_wchg() -> DataFrame

Read all SIPP cores and write sipp_wchg_rates.csv (one row per window with SIPP
data; returned) and, through `make_sipp_ee`, sipp_ee_rates.csv. Columns:
- `wchg_rate_U/S`: shipped hazard (BBG on `SIPP_BBG_WINDOWS`, raw earnings elsewhere);
- `neff_U/S`: its effective N (BBG Σ(T − 1), or earnings Kish N);
- `wchg_rate_U/S_raw`: raw earnings hazard at the production ε, all windows;
- `hourly_share_U/S`: share of employed person-months paid by the hour;
- `bbg_underflow_U/S`: 1 if π̃ was forced to 0;
- `bbg_pihat_U/S`, `bbg_corrfac_U/S`: π̂ and γ̄ − ᾱ on BBG windows (NaN
  elsewhere), the inputs of the variance in sampling_variances.jl.
If data/raw/sipp is absent or empty, both CSVs are written empty and every SIPP
moment is held out. Prints the raw and shipped tables and the ε robustness grid.
"""
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
    calib = Dict{Int,NamedTuple}()   # BBG calibration memoised per series length T
    wchg_rows = NamedTuple[]
    for wname in WINDOWS_ORDER
        haskey(master.wchg, wname) || continue
        c = master.wchg[wname]
        h = get(master.hshare, wname, nothing)
        raw_U = _sipp_hazard(c.chg_U[ip], c.den_U[ip])
        raw_S = _sipp_hazard(c.chg_S[ip], c.den_S[ip])
        n_earn_U = _sipp_neff(c.den_U[ip], c.den2_U[])
        n_earn_S = _sipp_neff(c.den_S[ip], c.den2_S[])
        hs_U = isnothing(h) || h.den_U[] <= 0.0 ? NaN : h.hr_U[] / h.den_U[]
        hs_S = isnothing(h) || h.den_S[] <= 0.0 ? NaN : h.hr_S[] / h.den_S[]

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

"""
    make_sipp_ee(master) -> DataFrame

Write and return sipp_ee_rates.csv, one row per window with skilled EE mass:
`ee_rate_S` (monthly hazard −log(1 − p)), `ee_step_S` (mean log weekly wage step
on a move), `ee_step_sd` (its weighted sd), and the Kish N `neff_ee`, `neff_step`.
"""
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

"""
    print_sipp_deflation_basis(years_used)

Print the CPI-U base once and the CPI-U index of each calendar year deflated.
"""
function print_sipp_deflation_basis(years_used::Set{Int})
    @printf("  SIPP deflation base: CPI-U(%d) = %.3f\n", SIPP_CPIU_BASE_YEAR, SIPP_CPIU_BASE)
    isempty(years_used) && return nothing
    parts = ["$(y)=$(@sprintf("%.3f", sipp_cpi(y)))" for y in sort(collect(years_used))]
    println("  CPI-U by year used: ", join(parts, ", "))
    return nothing
end

"""
    print_sipp_wchg_tables(rows)

Print Table 1 (raw earnings hazard at the production ε, every window), Table 2
(the shipped hazard with method, N and hourly share) and, for the BBG windows,
the raw upper bound, π̂ and the correction factor.
"""
function print_sipp_wchg_tables(rows::Vector{NamedTuple})
    isempty(rows) && return nothing
    println("\n  ══ TABLE 1 — ALL RAW (earnings-based within-job wage-change hazard, ε = \$$(SIPP_WCHG_EPS)/week) ══")
    @printf("  %-14s %12s %12s %10s %10s\n", "window", "wchg_U", "wchg_S", "neff_U", "neff_S")
    for r in rows
        # neff is the shipped construction's N (BBG Σ(T−1) on the FC windows).
        @printf("  %-14s %12.5f %12.5f %10.1f %10.1f\n",
                string(r.window), r.wchg_rate_U_raw, r.wchg_rate_S_raw, r.neff_U, r.neff_S)
    end

    println("\n  ══ TABLE 2 — SHIPPED (BBG-classic + raw-redesign; wchg_rate_U/S in sipp_wchg_rates.csv) ══")
    @printf("  %-14s %-7s %12s %12s %10s %10s %8s %8s\n",
            "window", "method", "wchg_U", "wchg_S", "neff_U", "neff_S", "hshr_U", "hshr_S")
    for r in rows
        tag = r.window in SIPP_BBG_WINDOWS ? "BBG" : "raw"
        @printf("  %-14s %-7s %12.5f %12.5f %10.1f %10.1f %8.3f %8.3f\n",
                string(r.window), tag, r.wchg_rate_U, r.wchg_rate_S,
                r.neff_U, r.neff_S, r.hourly_share_U, r.hourly_share_S)
    end
    println("  Table 2 is the shipped target: BBG break-filtered hazard (lower bound) on the")
    println("  classic FC windows, raw earnings hazard on the redesign COVID windows.")

    bbg_rows = filter(r -> r.window in SIPP_BBG_WINDOWS, rows)
    if !isempty(bbg_rows)
        println("\n  ── BBG correction diagnostics (classic FC windows; corr = γ̄ − ᾱ) ──")
        @printf("  %-14s %10s %10s %8s %8s %8s %8s\n",
                "window", "raw_U", "raw_S", "π̂_U", "π̂_S", "corr_U", "corr_S")
        any_uf = false
        for r in bbg_rows
            @printf("  %-14s %10.5f %10.5f %8.4f %8.4f %8.4f %8.4f\n",
                    string(r.window), r.wchg_rate_U_raw, r.wchg_rate_S_raw,
                    r.bbg_pihat_U, r.bbg_pihat_S, r.bbg_corrfac_U, r.bbg_corrfac_S)
            any_uf |= (r.bbg_underflow_U == 1 || r.bbg_underflow_S == 1)
        end
        any_uf && println("  [underflow: at least one window had π̃ ≤ 0 forced to 0]")
    end
    println()
    return nothing
end

"""
    print_sipp_wchg_robustness(acc)

Print the earnings-based wage-change hazard for every window and market at each
ε on `SIPP_WCHG_EPS_GRID` (diagnostic; NaN for windows without SIPP data).
"""
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

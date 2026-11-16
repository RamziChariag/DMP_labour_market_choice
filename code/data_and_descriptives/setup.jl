############################################################
# code/data_and_descriptives/setup.jl
#
# Shared definitions for the data pipeline: the estimation windows (single
# source of truth, persisted to derived/windows.json), CPS classification
# helpers, wage construction, weighted statistics, closed-form sampling-variance
# helpers, the IND→JOLTS supersector crosswalk and the canonical MOMENT_NAMES.
#
# Entry point: write_windows_json()  → derived/windows.json
# Definitions only; packages and path constants come from data_processing_main.jl.
############################################################

# ============================================================
# Estimation windows
# ============================================================

"""
    WINDOWS

The four estimation windows: CPS months `ym_start`–`ym_end` (as 100·year + month)
and the ASEC survey years `asec_years` (income year = survey year − 1). Every
window-dependent filter in the pipeline derives from this table.
"""
const WINDOWS = Dict{Symbol, NamedTuple}(
    :base_fc      => (label = "Pre-FC baseline",
                      ym_start = 200301, ym_end = 200712,
                      asec_years = 2004:2008),
    :crisis_fc    => (label = "Financial crisis",
                      ym_start = 200801, ym_end = 201012,
                      asec_years = 2009:2011),
    :base_covid   => (label = "Pre-COVID baseline",
                      ym_start = 201501, ym_end = 201912,
                      asec_years = 2016:2020),
    :crisis_covid => (label = "COVID crisis",
                      ym_start = 202001, ym_end = 202212,
                      asec_years = 2021:2023),
)

"""Canonical window order for tables and output files."""
const WINDOWS_ORDER = [:base_fc, :crisis_fc, :base_covid, :crisis_covid]

"""
    ym_int(year, month) -> Int

Integer calendar month 100·year + month.
"""
ym_int(year::Int, month::Int)::Int = 100 * year + month

"""
    assign_window(year, month) -> Symbol

Window containing the calendar month, or `:none`.
"""
function assign_window(year::Int, month::Int)::Symbol
    ym = ym_int(year, month)
    for (wname, wdef) in WINDOWS
        (wdef.ym_start <= ym <= wdef.ym_end) && return wname
    end
    return :none
end

"""
    assign_asec_window(survey_year) -> Symbol

Window whose `asec_years` contain the ASEC survey year, or `:none`.
"""
function assign_asec_window(survey_year::Int)::Symbol
    for (wname, wdef) in WINDOWS
        (survey_year in wdef.asec_years) && return wname
    end
    return :none
end

"""
    _windows_to_dict() -> Dict

`WINDOWS` and `WINDOWS_ORDER` as a JSON-serialisable dictionary.
"""
function _windows_to_dict()
    out = Dict{String, Any}()
    for (k, v) in WINDOWS
        out[String(k)] = Dict(
            "label"      => v.label,
            "ym_start"   => v.ym_start,
            "ym_end"     => v.ym_end,
            "asec_years" => collect(v.asec_years),
        )
    end
    return Dict(
        "windows"       => out,
        "windows_order" => String.(WINDOWS_ORDER),
    )
end

"""
    write_windows_json() -> String

Write `WINDOWS` to derived/windows.json, the copy read by code outside this
pipeline. Rewritten on every run so it cannot go stale. Returns the path.
"""
function write_windows_json()
    mkpath(DERIVED_DIR)
    open(joinpath(DERIVED_DIR, "windows.json"), "w") do io
        JSON3.pretty(io, _windows_to_dict())
    end
    @info "  Wrote derived/windows.json"
    return joinpath(DERIVED_DIR, "windows.json")
end

# ============================================================
# CPS classification (IPUMS codes)
# ============================================================

"""Skilled: CPS `EDUC >= 111` (bachelor's degree or higher)."""
is_skilled(educ::Integer)::Bool = educ >= 111

"""Working age, `lo ≤ age ≤ hi` (default 16–64)."""
in_age_range(age::Integer; lo::Int=16, hi::Int=64)::Bool = lo <= age <= hi

"""Employed: `EMPSTAT ∈ {10, 12}`."""
is_employed(empstat::Integer)::Bool = empstat in (10, 12)

"""Unemployed: `EMPSTAT ∈ {20, 21, 22}`."""
is_unemployed(empstat::Integer)::Bool = empstat in (20, 21, 22)

"""Not in the labour force: `EMPSTAT ∈ {30, …, 36}`."""
is_nilf(empstat::Integer)::Bool = empstat in (30, 31, 32, 33, 34, 36)

"""Wage and salary worker: `CLASSWKR ∈ {22, 25, 27, 28}`."""
is_wage_worker(classwkr::Integer)::Bool = classwkr in (22, 25, 27, 28)

"""
    is_enrolled_no_ba(schlcoll, educ) -> Bool

Enrolled in college (`SCHLCOLL` 3 full-time, 4 part-time) without a bachelor's
degree. Implies no trainee is skilled, which several pool definitions rely on.
"""
is_enrolled_no_ba(schlcoll::Integer, educ::Integer)::Bool = schlcoll in (3, 4) && educ < 111

"""
    valid_match_mish(mish) -> Bool

Month-in-sample whose next calendar month is also in sample (4-8-4 rotation).
"""
valid_match_mish(mish::Integer)::Bool = mish in (1, 2, 3, 5, 6, 7)

"""
    apply_covid_correction(empstat, year, month, whyabsnt) -> Integer

BLS misclassification fix for March–June 2020: employed-but-absent workers with
`WHYABSNT == 15` are recoded to unemployed on temporary layoff (21).
"""
function apply_covid_correction(empstat::Integer, year::Int, month::Int,
                                whyabsnt::Integer)::Integer
    (year == 2020 && 3 <= month <= 6) || return empstat
    if is_employed(empstat) && whyabsnt == 15
        return 21
    end
    return empstat
end

# ============================================================
# Wage construction
# ============================================================

"""
    compute_hourly_wage(incwage, wkswork1, uhrsworkly) -> Float64

Annual wage income over annual hours; NaN if any input is non-positive.
"""
function compute_hourly_wage(incwage, wkswork1, uhrsworkly)::Float64
    (wkswork1 <= 0 || uhrsworkly <= 0 || incwage <= 0) && return NaN
    return Float64(incwage) / (Float64(wkswork1) * Float64(uhrsworkly))
end

"""
    deflate_wage(nominal_wage, cpi_t, cpi_base) -> Float64

Real wage `nominal × cpi_base / cpi_t` for a standard increasing price index
(used by SIPP with CPI-U). Not valid for IPUMS CPI99, which runs the other way.
"""
deflate_wage(nominal_wage::Float64, cpi_t::Float64, cpi_base::Float64)::Float64 =
    nominal_wage * (cpi_base / cpi_t)

"""
    trim_bounds(wages, weights; lo_pct = 0.01, hi_pct = 0.99) -> (lo, hi)

Weighted `lo_pct` and `hi_pct` percentile values of `wages`; `(NaN, NaN)` for an
empty sample or non-positive total weight.
"""
function trim_bounds(wages::AbstractVector{Float64}, weights::AbstractVector{Float64};
                     lo_pct::Float64=0.01, hi_pct::Float64=0.99)
    n = length(wages)
    n == 0 && return (NaN, NaN)
    idx = sortperm(wages)
    cum = 0.0
    total = sum(weights)
    total <= 0.0 && return (NaN, NaN)
    lo_val = wages[idx[1]]
    hi_val = wages[idx[end]]
    for i in idx
        cum += weights[i] / total
        if cum >= lo_pct && lo_val == wages[idx[1]]
            lo_val = wages[i]
        end
        if cum >= hi_pct
            hi_val = wages[i]
            break
        end
    end
    return (lo_val, hi_val)
end

# ============================================================
# Weighted statistics
# ============================================================

"""
    wmean(x, w) -> Float64

Weighted mean; NaN for non-positive total weight.
"""
function wmean(x::AbstractVector, w::AbstractVector)::Float64
    sw = sum(w); sw <= 0.0 && return NaN
    return sum(x .* w) / sw
end

"""
    wmedian(x, w) -> Float64

Weighted median: the first sorted observation whose cumulative normalised
weight reaches 0.5.
"""
function wmedian(x::AbstractVector, w::AbstractVector)::Float64
    n = length(x); n == 0 && return NaN
    idx = sortperm(x)
    cum = 0.0; total = sum(w)
    total <= 0.0 && return NaN
    for i in idx
        cum += w[i] / total
        cum >= 0.5 && return Float64(x[i])
    end
    return Float64(x[idx[end]])
end

"""
    wpercentile(x, w, p) -> Float64

Weighted p-quantile: the first sorted observation whose cumulative weight
reaches p·Σw, the convention `wmedian` applies at p = 0.5. One estimator serves
the whole p10–p90 block so the points cannot drift apart.
"""
function wpercentile(x::AbstractVector, w::AbstractVector, p::Real)::Float64
    n = length(x); n == 0 && return NaN
    ord = sortperm(x)
    xs, ws = x[ord], w[ord]
    cs = cumsum(ws)
    total = cs[end]
    total <= 0.0 && return NaN
    target = p * total
    i = searchsortedfirst(cs, target)
    i = clamp(i, 1, length(xs))
    return Float64(xs[i])
end

"""Weighted 10th percentile (`wpercentile` at 0.10)."""
wpercentile10(x::AbstractVector, w::AbstractVector)::Float64 = wpercentile(x, w, 0.10)
"""Weighted 25th percentile (`wpercentile` at 0.25)."""
wpercentile25(x::AbstractVector, w::AbstractVector)::Float64 = wpercentile(x, w, 0.25)
"""Weighted 75th percentile (`wpercentile` at 0.75)."""
wpercentile75(x::AbstractVector, w::AbstractVector)::Float64 = wpercentile(x, w, 0.75)
"""Weighted 90th percentile (`wpercentile` at 0.90)."""
wpercentile90(x::AbstractVector, w::AbstractVector)::Float64 = wpercentile(x, w, 0.90)

"""
    wvar(x, w) -> Float64

Weighted variance Σw(x − x̄)²/Σw (no small-sample correction).
"""
function wvar(x::AbstractVector, w::AbstractVector)::Float64
    m = wmean(x, w); sw = sum(w)
    sw <= 0.0 && return NaN
    return sum(w .* (x .- m).^2) / sw
end

"""Weighted standard deviation `sqrt(wvar)`."""
wsd(x::AbstractVector, w::AbstractVector)::Float64 = sqrt(max(wvar(x, w), 0.0))

"""
    wcm3(x, w) -> Float64

Weighted third central moment Σw(x − x̄)³/Σw.
"""
function wcm3(x::AbstractVector, w::AbstractVector)::Float64
    m = wmean(x, w); sw = sum(w)
    sw <= 0.0 && return NaN
    return sum(w .* (x .- m).^3) / sw
end

"""
    kde_at_point(x, w, point; bw = NaN) -> Float64

Weighted Gaussian kernel density of `x` at `point`. With `bw = NaN` the
bandwidth is Silverman's 1.06·sd·N_eff^(−1/5) on the Kish N_eff.
"""
function kde_at_point(x::AbstractVector, w::AbstractVector, point::Float64;
                      bw::Float64=NaN)::Float64
    n = length(x)
    n == 0 && return NaN
    if isnan(bw)
        s = wsd(x, w)
        n_eff = sum(w)^2 / sum(w.^2)
        bw = 1.06 * s * n_eff^(-0.2)
    end
    bw <= 0.0 && return NaN
    sw = sum(w)
    density = 0.0
    for i in 1:n
        u = (x[i] - point) / bw
        density += w[i] * exp(-0.5 * u^2) / sqrt(2π)
    end
    return density / (sw * bw)
end

# ============================================================
# Closed-form sampling-variance helpers (used by sampling_variances.jl
# and the decision audit)
# ============================================================

"""
    _kish_prop_var(w, denom, numer) -> Float64

Variance p̂(1−p̂)/N_eff of the weighted share of `numer` within `denom`, treating
records as independent draws (Kish N_eff).
"""
function _kish_prop_var(w::Vector{Float64}, denom::AbstractVector{Bool},
                        numer::AbstractVector{Bool})::Float64
    sw = 0.0; sw2 = 0.0; snum = 0.0
    @inbounds for i in eachindex(w)
        if denom[i]
            sw += w[i]; sw2 += w[i]^2
            numer[i] && (snum += w[i])
        end
    end
    (sw <= 0.0 || sw2 <= 0.0) && return NaN
    neff = sw^2 / sw2
    p    = snum / sw
    return p * (1.0 - p) / neff
end

"""
    _kish_neff(w) -> Float64

Kish effective sample size (Σw)²/Σw²; 0 for an all-zero weight vector.
"""
function _kish_neff(w::AbstractVector{<:Real})::Float64
    sw = 0.0; sw2 = 0.0
    @inbounds for wi in w; sw += wi; sw2 += wi^2; end
    sw2 > 0.0 ? sw^2 / sw2 : 0.0
end

"""
    _wcm(x, w, r) -> Float64

r-th weighted central moment Σw(x − x̄)^r / Σw.
"""
function _wcm(x::AbstractVector{<:Real}, w::AbstractVector{<:Real}, r::Int)::Float64
    m = wmean(x, w); sw = sum(w)
    sw <= 0.0 && return NaN
    s = 0.0
    @inbounds for i in eachindex(x); s += w[i] * (x[i] - m)^r; end
    return s / sw
end

"""
    _wquantile_var(x, w, qhat, p) -> Float64

Sampling variance p(1−p)/(N_eff·f(q̂)²) of a weighted p-quantile, with f the
`kde_at_point` density at `qhat` and N_eff the Kish size of `w`.
"""
function _wquantile_var(x::AbstractVector{<:Real}, w::AbstractVector{<:Real},
                        qhat::Float64, p::Float64)::Float64
    neff = _kish_neff(w)
    (neff <= 0.0 || !isfinite(qhat)) && return NaN
    f = kde_at_point(x, w, qhat)
    (!isfinite(f) || f <= 0.0) && return NaN
    return p * (1.0 - p) / (neff * f^2)
end

"""
    _kish_prop_var_clustered(w, denom, numer, id) -> Float64

Cluster-robust linearised variance of the weighted share of `numer` within
`denom`, Σ_g (t_g − p̂ s_g)² / (Σ_g s_g)², with clusters `id` (s_g, t_g the
cluster's weight in the denominator and numerator). Used with `id = CPSIDP`
because CPS Basic is a 4-8-4 rotating panel: the unclustered Kish form
understates these variances by 2.2–5.6×. With singleton clusters it is the
Huber-White form, equal to Kish in expectation but not sample by sample.
"""
function _kish_prop_var_clustered(w::Vector{Float64}, denom::AbstractVector{Bool},
                                  numer::AbstractVector{Bool},
                                  id::AbstractVector{<:Integer})::Float64
    sw = 0.0
    @inbounds for i in eachindex(w)
        denom[i] && (sw += w[i])
    end
    sw <= 0.0 && return NaN
    p = 0.0
    @inbounds for i in eachindex(w)
        (denom[i] && numer[i]) && (p += w[i])
    end
    p /= sw

    acc = Dict{Int64, Tuple{Float64, Float64}}()
    @inbounds for i in eachindex(w)
        denom[i] || continue
        g    = Int64(id[i])
        s, t = get(acc, g, (0.0, 0.0))
        acc[g] = (s + w[i], t + (numer[i] ? w[i] : 0.0))
    end

    ss = 0.0
    for (_, (s, t)) in acc
        ss += (t - p * s)^2
    end
    return ss / sw^2
end

"""
    _durw(d) -> Float64

Unemployment duration in weeks from `DURUNEMP`, with the IPUMS NIU code 999 (and
missing) mapped to 0. Unguarded, 999 exceeds every duration threshold and counts
NIU records, mostly the March–June 2020 COVID recodes, as long-term unemployed.
"""
@inline _durw(d) = (x = Float64(coalesce(d, 0.0)); x == 999.0 ? 0.0 : x)

# ============================================================
# IND → JOLTS supersector crosswalk
# ============================================================

"""
    ind_to_jolts_supersector(ind) -> String

JOLTS supersector key for a Census IND code (2002/2012/2017 schemes, checked
against the 2024 ACS). Mining and logging uses the internal key "100000"; all
public administration maps to "920000". Agriculture, private households and the
military return "EXCLUDED"; unrecognised codes (e.g. the 1990 scheme) "UNKNOWN".
"""
function ind_to_jolts_supersector(ind::Int)::String
    ind == 0                 && return "EXCLUDED" # N/A / not-applicable code
    ind == 270               && return "100000"  # Logging → Mining & Logging
    170 <= ind <= 290        && return "EXCLUDED" # Agriculture (excl. logging)
    370 <= ind <= 490        && return "100000"  # Mining
    570 <= ind <= 690        && return "480099"  # Utilities → TWU
    ind == 770               && return "230000"  # Construction
    1070 <= ind <= 1990      && return "340000"  # Nondurable mfg (Food–Printing)
    2070 <= ind <= 2390      && return "340000"  # Nondurable mfg (Petrol/Chem/Plastics)
    2470 <= ind <= 3990      && return "320000"  # Durable mfg
    4070 <= ind <= 4590      && return "420000"  # Wholesale
    4670 <= ind <= 5791      && return "440000"  # Retail
    6070 <= ind <= 6390      && return "480099"  # Transport/Warehousing → TWU
    6470 <= ind <= 6781      && return "510000"  # Information (6470 = Newspaper publishers)
    6870 <= ind <= 6992      && return "510099"  # Finance & Insurance (6870 = Banking)
    7070 <= ind <= 7190      && return "510099"  # Real Estate (7070 in 2002/2012; 7071 in 2017)
    7270 <= ind <= 7790      && return "540099"  # Professional & Business Services
    7860 <= ind <= 7890      && return "610000"  # Educational Services
    7970 <= ind <= 8470      && return "620000"  # Health Care & Social Assistance
    8560 <= ind <= 8590      && return "710000"  # Arts, Entertainment & Recreation (8560 in 2002/2012; 8561 in 2017)
    8660 <= ind <= 8690      && return "720000"  # Accommodation & Food Services
    8770 <= ind <= 9190      && return "810000"  # Other Services
    ind == 9290              && return "EXCLUDED" # Private households
    9370 <= ind <= 9590      && return "920000"  # Public Administration → Government
    9670 <= ind <= 9870      && return "EXCLUDED" # Military (2002/2012 scheme)
    ind == 9890              && return "EXCLUDED" # Armed Forces (2017 scheme; civilian-LF sample)
    return "UNKNOWN"
end

# ============================================================
# Moment names
# ============================================================

"""
    DURATION_THRESHOLDS_WK

Thresholds (weeks) of the unemployment-duration survivor moments usurv{t}_j, the
share of market j's unemployed whose ongoing spell has lasted at least t weeks.
Each sits one week above a CPS reporting heap (4, 13, 26, 52) so the moment does
not depend on which side of a heap the comparison falls; t = 27 nests
`ltu_share_S`. Survivor values rather than bin shares, which would sum to one
and make the block collinear. Mirrored in smm/moments.jl and solver/equilibrium.jl.
"""
const DURATION_THRESHOLDS_WK = [5.0, 14.0, 27.0, 53.0]

"""Duration moment names, usurv{t}_U then usurv{t}_S."""
const DURATION_MOMENT_NAMES =
    [Symbol("usurv$(Int(t))_$(m)") for m in ("U", "S") for t in DURATION_THRESHOLDS_WK]

"""
    MOMENT_NAMES

Canonical moment order (35 base moments, then the duration block); the position
is each moment's index in every moments_{w}.csv and sampling_var_{w}.csv.
smm/moments.jl uses the identical order.
"""
const MOMENT_NAMES = [
    :ur_total, :ur_U, :ur_S,
    :skilled_share, :training_share,
    :emp_var_U, :emp_cm3_U, :emp_var_S, :emp_cm3_S,
    :jfr_U, :sep_rate_U, :jfr_S, :sep_rate_S,
    :ee_rate_S,
    :mean_wage_U, :mean_wage_S,
    :p10_wage_U, :p10_wage_S, :p25_wage_U, :p25_wage_S, :p50_wage_U, :p50_wage_S,
    :p75_wage_U, :p75_wage_S, :p90_wage_U, :p90_wage_S,
    :wage_premium, :theta_U, :theta_S,
    :overlap_UgtS, :overlap_SltU, :ltu_share_S,
    :wchg_rate_U, :wchg_rate_S,
    :ee_step_S,
    DURATION_MOMENT_NAMES...,
]

"""Number of moments before the duration block."""
const N_MOMENTS_BASE = 35

@assert length(MOMENT_NAMES) == N_MOMENTS_BASE + length(DURATION_MOMENT_NAMES) """
Expected $(N_MOMENTS_BASE + length(DURATION_MOMENT_NAMES)) moments, \
got $(length(MOMENT_NAMES))"""

"""
    _load_arrow(filename) -> DataFrame

Read derived/`filename`; an empty DataFrame (with a warning) if it is absent.
"""
function _load_arrow(filename::String)::DataFrame
    path = joinpath(DERIVED_DIR, filename)
    !isfile(path) && (@warn "$filename not found"; return DataFrame())
    return DataFrame(Arrow.Table(path))
end

############################################################
# data_processing/setup.jl
#
# Shared foundation for the whole pipeline: estimation windows, the
# canonical MOMENT_NAMES list, classification / wage / weighted-statistics
# helpers, the IND→JOLTS supersector map, and small IO helpers.
# write_windows_json() persists the single source of truth for WINDOWS.
#
# Reads:  —
# Writes: windows.json
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages and path consts come from data_processing_main.jl.
############################################################

# ============================================================
# Estimation windows — vectorised ym-based filter
# ----------------------------------------------------------
# THIS IS THE SINGLE SOURCE OF TRUTH for the four windows.
# Any other notebook (e.g. descriptive_analysis) should read
# them from `derived/windows.json`, written below.
# ============================================================

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

const WINDOWS_ORDER = [:base_fc, :crisis_fc, :base_covid, :crisis_covid]

# Integer ym = 100*year + month for fast comparison
ym_int(year::Int, month::Int)::Int = 100 * year + month

function assign_window(year::Int, month::Int)::Symbol
    ym = ym_int(year, month)
    for (wname, wdef) in WINDOWS
        (wdef.ym_start <= ym <= wdef.ym_end) && return wname
    end
    return :none
end

function assign_asec_window(survey_year::Int)::Symbol
    for (wname, wdef) in WINDOWS
        (survey_year in wdef.asec_years) && return wname
    end
    return :none
end

# ── Persist WINDOWS for sister notebooks ─────────────────────────
# `derived/windows.json` is the canonical record used by
# descriptive_analysis and any downstream consumer. Re-written
# every time this cell runs so the file never goes stale.
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

function write_windows_json()
    mkpath(DERIVED_DIR)
    open(joinpath(DERIVED_DIR, "windows.json"), "w") do io
        JSON3.pretty(io, _windows_to_dict())
    end
    @info "  Wrote derived/windows.json"
    return joinpath(DERIVED_DIR, "windows.json")
end

# ============================================================
# Classification helpers
# ============================================================

is_skilled(educ::Integer)::Bool = educ >= 111
in_age_range(age::Integer; lo::Int=16, hi::Int=64)::Bool = lo <= age <= hi
is_civilian_lf(labforce::Integer)::Bool = labforce == 2
is_employed(empstat::Integer)::Bool = empstat in (10, 12)
is_unemployed(empstat::Integer)::Bool = empstat in (20, 21, 22)
is_nilf(empstat::Integer)::Bool = empstat in (30, 31, 32, 33, 34, 36)
is_wage_worker(classwkr::Integer)::Bool = classwkr in (22, 25, 27, 28)

# Enrolled in college without a BA: SCHLCOLL code 3 = college full-time,
# code 4 = college part-time; EDUC < 111 excludes those who already hold a BA.
is_enrolled_no_ba(schlcoll::Integer, educ::Integer)::Bool = schlcoll in (3, 4) && educ < 111

valid_match_mish(mish::Integer)::Bool = mish in (1, 2, 3, 5, 6, 7)

# ============================================================
# COVID BLS misclassification correction
# ============================================================

function apply_covid_correction(empstat::Integer, year::Int, month::Int,
                                whyabsnt::Integer)::Integer
    (year == 2020 && 3 <= month <= 6) || return empstat
    if is_employed(empstat) && whyabsnt == 15
        return 21  # unemployed, temporary layoff
    end
    return empstat
end

# ============================================================
# Wage construction
# ============================================================

function compute_hourly_wage(incwage, wkswork1, uhrsworkly)::Float64
    (wkswork1 <= 0 || uhrsworkly <= 0 || incwage <= 0) && return NaN
    return Float64(incwage) / (Float64(wkswork1) * Float64(uhrsworkly))
end

deflate_wage(nominal_wage::Float64, cpi_t::Float64, cpi_base::Float64)::Float64 =
    nominal_wage * (cpi_base / cpi_t)

# ============================================================
# Trimming
# ============================================================

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

function wmean(x::AbstractVector, w::AbstractVector)::Float64
    sw = sum(w); sw <= 0.0 && return NaN
    return sum(x .* w) / sw
end

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

function wpercentile25(x::AbstractVector, w::AbstractVector)::Float64
    n = length(x); n == 0 && return NaN
    ord = sortperm(x)
    xs, ws = x[ord], w[ord]
    cs = cumsum(ws)
    total = cs[end]
    total <= 0.0 && return NaN
    target = 0.25 * total
    i = searchsortedfirst(cs, target)
    i = clamp(i, 1, length(xs))
    return Float64(xs[i])
end

function wpercentile75(x::AbstractVector, w::AbstractVector)::Float64
    n = length(x); n == 0 && return NaN
    ord = sortperm(x)
    xs, ws = x[ord], w[ord]
    cs = cumsum(ws)
    total = cs[end]
    total <= 0.0 && return NaN
    target = 0.75 * total
    i = searchsortedfirst(cs, target)
    i = clamp(i, 1, length(xs))
    return Float64(xs[i])
end


function wvar(x::AbstractVector, w::AbstractVector)::Float64
    m = wmean(x, w); sw = sum(w)
    sw <= 0.0 && return NaN
    return sum(w .* (x .- m).^2) / sw
end

wsd(x::AbstractVector, w::AbstractVector)::Float64 = sqrt(max(wvar(x, w), 0.0))

function wcm3(x::AbstractVector, w::AbstractVector)::Float64
    m = wmean(x, w); sw = sum(w)
    sw <= 0.0 && return NaN
    return sum(w .* (x .- m).^3) / sw
end

# Kernel density at a point (for influence function of weighted median)
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
# IND-to-JOLTS supersector mapping (verified against 2024 ACS)
# ============================================================

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
# Moment name list — 28 moments
#
# overlap_UgtS, overlap_SltU and ltu_share_S are the cross-market
# wage-overlap pair and the skilled long-term-unemployment share; they
# are appended at the END so the legacy moment indices are unchanged.
# ============================================================
const MOMENT_NAMES = [
    :ur_total, :ur_U, :ur_S,
    :skilled_share, :training_share,
    :emp_var_U, :emp_cm3_U, :emp_var_S, :emp_cm3_S,
    :jfr_U, :sep_rate_U, :jfr_S, :sep_rate_S,
    :ee_rate_S,
    :mean_wage_U, :mean_wage_S,
    :p25_wage_U, :p25_wage_S, :p50_wage_U, :p50_wage_S, :p75_wage_U, :p75_wage_S,
    :wage_premium, :theta_U, :theta_S,
    :overlap_UgtS, :overlap_SltU, :ltu_share_S,
]

@assert length(MOMENT_NAMES) == 28 "Expected 28 moments, got $(length(MOMENT_NAMES))"

# ============================================================
# Regularization parameter for Σ̂
# ============================================================

const REGULARIZATION_ALPHA = 0.0  # Shrinkage: (1-α)Σ + α·diag(Σ).  0 = off.


# ──────────────────────────────────────────────────────────────────────────
# Helper: load a derived Arrow file → DataFrame
# ──────────────────────────────────────────────────────────────────────────
function _load_arrow(filename::String)::DataFrame
    path = joinpath(DERIVED_DIR, filename)
    !isfile(path) && (@warn "$filename not found"; return DataFrame())
    return DataFrame(Arrow.Table(path))
end
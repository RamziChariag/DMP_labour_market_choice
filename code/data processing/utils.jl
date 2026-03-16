############################################################
# utils.jl — Shared helpers for the data-cleaning pipeline
#
# Contents
# ─────────────────────────────────────────────────────────
#   Path constants
#   Estimation-window definitions
#   Skill classification
#   Sample restrictions
#   Wage construction and trimming
#   Weighted statistics (mean, median, variance, cm3, sd)
#   Influence-function helpers
#   Arrow I/O wrappers
############################################################


# ============================================================
# Paths
# ============================================================

const DATA_CLEANING_DIR = @__DIR__
const CODE_DIR          = joinpath(DATA_CLEANING_DIR, "..")
const PROJECT_ROOT      = joinpath(CODE_DIR, "..")
const DATA_DIR          = joinpath(PROJECT_ROOT, "data")
const RAW_DIR           = joinpath(DATA_DIR, "raw")
const DERIVED_DIR       = joinpath(DATA_DIR, "derived")

const RAW_CPS_BASIC_DIR = joinpath(RAW_DIR, "cps_basic")
const RAW_CPS_ASEC_DIR  = joinpath(RAW_DIR, "cps_asec")
const RAW_JOLTS_DIR     = joinpath(RAW_DIR, "jolts")

# Ensure derived directory exists
mkpath(DERIVED_DIR)


# ============================================================
# Estimation windows
# ============================================================

"""
    WINDOWS :: Dict{Symbol, NamedTuple}

The four estimation windows.  Each entry has:
  label       human-readable name
  ym_start    (year, month) of the first month included
  ym_end      (year, month) of the last  month included
  asec_years  ASEC survey years whose income data falls in this window
"""
const WINDOWS = Dict{Symbol, NamedTuple}(
    :base_fc     => (label = "Pre-FC baseline",
                     ym_start = (2003, 1),  ym_end = (2007, 12),
                     asec_years = 2004:2008),

    :crisis_fc   => (label = "Financial crisis",
                     ym_start = (2008, 1),  ym_end = (2009, 6),
                     asec_years = 2009:2010),

    :base_covid  => (label = "Pre-COVID baseline",
                     ym_start = (2015, 1),  ym_end = (2019, 12),
                     asec_years = 2016:2020),

    :crisis_covid => (label = "COVID crisis",
                      ym_start = (2020, 3),  ym_end = (2021, 12),
                      asec_years = 2021:2022),
)


"""
    in_window(year, month, win) → Bool

Check whether a calendar (year, month) pair falls inside window `win`.
"""
function in_window(year::Int, month::Int, win::NamedTuple) :: Bool
    ym = (year, month)
    return ym >= win.ym_start && ym <= win.ym_end
end

"""
    in_asec_window(survey_year, win) → Bool

Check whether an ASEC survey year belongs to the window's ASEC range.
"""
function in_asec_window(survey_year::Int, win::NamedTuple) :: Bool
    return survey_year in win.asec_years
end


# ============================================================
# Skill classification
# ============================================================

"""
    is_skilled(educ) → Bool

Skilled = bachelor's degree or higher.
IPUMS EDUC code ≥ 111 (Bachelor's degree).
"""
is_skilled(educ::Integer) :: Bool = educ >= 111


# ============================================================
# Sample restrictions
# ============================================================

"""
    in_age_range(age; lo=16, hi=64) → Bool

Age restriction: 16–64 for all samples.
"""
in_age_range(age::Integer; lo::Int = 16, hi::Int = 64) :: Bool =
    lo <= age <= hi

"""
    is_civilian_lf(labforce) → Bool

In the civilian labour force (LABFORCE == 2 in IPUMS).
"""
is_civilian_lf(labforce::Integer) :: Bool = labforce == 2

"""
    is_employed(empstat) → Bool

Employed (EMPSTAT ∈ {10, 12} in IPUMS — at work or has job, not at work).
"""
is_employed(empstat::Integer) :: Bool = empstat in (10, 12)

"""
    is_unemployed(empstat) → Bool

Unemployed (EMPSTAT ∈ {20, 21, 22} in IPUMS).
"""
is_unemployed(empstat::Integer) :: Bool = empstat in (20, 21, 22)

"""
    is_wage_worker(classwkr) → Bool

Wage/salary worker (exclude self-employed and unpaid family workers).
IPUMS CLASSWKR: 22 = wage/salary, private; 25 = federal govt;
26 = armed forces; 27 = state govt; 28 = local govt.
Exclude: 0 (NIU), 10 (self-employed), 13 (self-emp, not incorporated),
14 (self-emp, incorporated), 29 (unpaid family).
"""
is_wage_worker(classwkr::Integer) :: Bool = classwkr in (22, 25, 27, 28)

"""
    is_enrolled_no_ba(schlcoll, educ) → Bool

Currently enrolled in college (SCHLCOLL ∈ {4, 5}) without a BA (EDUC < 111).
Used for the training_share moment.
"""
is_enrolled_no_ba(schlcoll::Integer, educ::Integer) :: Bool =
    schlcoll in (4, 5) && educ < 111

"""
    valid_match_mish(mish) → Bool

Valid month-in-sample for CPS panel matching.
MISH ∈ {1, 2, 3, 5, 6, 7} can be matched to the following month.
"""
valid_match_mish(mish::Integer) :: Bool = mish in (1, 2, 3, 5, 6, 7)


# ============================================================
# Wage construction (ASEC)
# ============================================================

"""
    compute_hourly_wage(incwage, wkswork1, uhrsworkly) → Float64

Real hourly wage from ASEC variables (before deflation).
Returns NaN if inputs are invalid.
"""
function compute_hourly_wage(incwage, wkswork1, uhrsworkly) :: Float64
    (wkswork1 <= 0 || uhrsworkly <= 0 || incwage <= 0) && return NaN
    return Float64(incwage) / (Float64(wkswork1) * Float64(uhrsworkly))
end

"""
    deflate_wage(nominal_wage, cpi_t, cpi_base) → Float64

Deflate nominal wage to constant dollars using CPI ratio.
"""
deflate_wage(nominal_wage::Float64, cpi_t::Float64, cpi_base::Float64) :: Float64 =
    nominal_wage * (cpi_base / cpi_t)


# ============================================================
# Trimming
# ============================================================

"""
    winsorize_bounds(wages, weights; lo_pct=0.01, hi_pct=0.99)
        → (lo_val, hi_val)

Compute the weighted percentile bounds for trimming.
"""
function winsorize_bounds(wages::Vector{Float64}, weights::Vector{Float64};
                          lo_pct::Float64 = 0.01, hi_pct::Float64 = 0.99)
    n = length(wages)
    n == 0 && return (NaN, NaN)

    idx = sortperm(wages)
    cum = 0.0
    total = sum(weights)
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
    wmean(x, w) → Float64

Weighted mean.
"""
function wmean(x::AbstractVector, w::AbstractVector) :: Float64
    sw = sum(w)
    sw <= 0.0 && return NaN
    return sum(x .* w) / sw
end

"""
    wmedian(x, w) → Float64

Weighted median.
"""
function wmedian(x::AbstractVector, w::AbstractVector) :: Float64
    n = length(x)
    n == 0 && return NaN
    idx = sortperm(x)
    cum = 0.0
    total = sum(w)
    total <= 0.0 && return NaN
    for i in idx
        cum += w[i] / total
        cum >= 0.5 && return Float64(x[i])
    end
    return Float64(x[idx[end]])
end

"""
    wvar(x, w) → Float64

Weighted variance.
"""
function wvar(x::AbstractVector, w::AbstractVector) :: Float64
    m = wmean(x, w)
    sw = sum(w)
    sw <= 0.0 && return NaN
    return sum(w .* (x .- m).^2) / sw
end

"""
    wsd(x, w) → Float64

Weighted standard deviation.
"""
wsd(x::AbstractVector, w::AbstractVector) :: Float64 = sqrt(max(wvar(x, w), 0.0))

"""
    wcm3(x, w) → Float64

Weighted third central moment.
"""
function wcm3(x::AbstractVector, w::AbstractVector) :: Float64
    m = wmean(x, w)
    sw = sum(w)
    sw <= 0.0 && return NaN
    return sum(w .* (x .- m).^3) / sw
end


# ============================================================
# COVID-era BLS misclassification correction
# ============================================================

"""
    apply_covid_correction(empstat, year, month, absent_reason, ind)
        → corrected_empstat

BLS correction for March–June 2020: reclassify persons coded
"employed, absent" who report reason "other" and are in industries
with mass layoffs as unemployed on temporary layoff.

Simplified version: reclassify EMPSTAT 12 (has job, absent) to
unemployed if in the COVID window and absent for "other" reasons.
The full correction requires industry-level layoff indicators.
"""
function apply_covid_correction(empstat::Integer, year::Int, month::Int)
    # Only apply during Mar–Jun 2020
    (year == 2020 && 3 <= month <= 6) || return empstat
    # Persons coded "has job, not at work" are candidates
    # Full implementation would check WHYABSNT and industry codes
    return empstat
end


# ============================================================
# Moment name list (for consistent ordering)
# ============================================================

const MOMENT_NAMES = [
    # Stocks
    :ur_total, :ur_U, :ur_S,
    :skilled_share, :training_share,
    :emp_var_U, :emp_cm3_U, :emp_var_S, :emp_cm3_S,
    # Transition rates
    :jfr_U, :sep_rate_U, :jfr_S, :sep_rate_S,
    :ee_rate_S, :training_rate,
    # Wages
    :mean_wage_U, :mean_wage_S,
    :p50_wage_U, :p50_wage_S,
    :wage_premium,
    :wage_sd_U, :wage_sd_S,
    # Tightness
    :theta_U, :theta_S,
]

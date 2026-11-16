############################################################
# code/data_and_descriptives/nsc.jl
#
# Education side of the data: the training_share target (level and variance),
# the completion rate φ, and one CPS enrolment diagnostic.
#
# Entry points:
#   enrollment_rate_by_age()          → enrollment_rate_by_age.csv (diagnostic)
#   compute_training_share_target()   → training_share_target.csv
#   _load_training_share_target(w)    reads that target back (moments, variances)
#   calibrate_phi()                   → phi_calibration.csv
# Reads data/raw/nces/nces_4yr_undergrad_enrollment.csv and cps_basic_clean.arrow.
# Definitions only; packages and path constants come from data_processing_main.jl.
############################################################

"""
    enrollment_rate_by_age() -> DataFrame

Weighted CPS enrolment rate (`in_training` share of population) by age band and
window, plus the overall 16–64 rate; printed and saved to
enrollment_rate_by_age.csv. The SCHLCOLL universe widened from 16–24 to 16–54 in
January 2013, so the FC windows show a structural zero above age 24. This is why
the training_share level is not taken from CPS.
"""
function enrollment_rate_by_age()
    cps_path = joinpath(DERIVED_DIR, "cps_basic_clean.arrow")
    isfile(cps_path) ||
        error("cps_basic_clean.arrow not found in $DERIVED_DIR — run Stage 1 first.")
    cps = DataFrame(Arrow.Table(cps_path))
    cps.enrolled = coalesce.(cps.in_training, false)

    bands = [(16, 19), (20, 24), (25, 29), (30, 34),
             (35, 44), (45, 54), (55, 64)]
    band_label(lo, hi) = lo == hi ? "$(lo)" : "$(lo)–$(hi)"

    # (rate, mean monthly population) for ages lo..hi: monthly weighted
    # enrolment and population, each averaged over months, then divided.
    function _band_rate(sub::DataFrame, lo::Int, hi::Int)
        b = filter(r -> lo <= r.AGE <= hi, sub)
        nrow(b) == 0 && return (NaN, NaN)
        per_month = combine(groupby(b, [:YEAR, :MONTH]),
            :WTFINL => (x -> sum(coalesce.(x, 0.0))) => :pop,
            [:WTFINL, :enrolled] =>
                ((wt, e) -> sum(coalesce.(wt, 0.0) .* e)) => :enr)
        pop = mean(per_month.pop)
        enr = mean(per_month.enr)
        return (pop > 0 ? enr / pop : NaN, pop)
    end

    rows = NamedTuple[]
    for w in WINDOWS_ORDER
        sub = filter(r -> r.window == w, cps)
        nt = (window = w,)
        for (lo, hi) in bands
            rate, _ = _band_rate(sub, lo, hi)
            nt = merge(nt, NamedTuple{(Symbol(band_label(lo, hi)),)}((
                isnan(rate) ? missing : round(rate; digits=4),)))
        end
        rate_all, _ = _band_rate(sub, 16, 64)
        nt = merge(nt, (overall_16_64 = isnan(rate_all) ? missing :
                                         round(rate_all; digits=4),))
        push!(rows, nt)
    end

    df = DataFrame(rows)
    println("\nCPS enrolment rate (in_training share) by age band × window")
    println("(values ≈ 0 for older bands in FC windows reveal the SCHLCOLL universe limit)")
    println("─"^100)
    display(df)
    CSV.write(joinpath(DERIVED_DIR, "enrollment_rate_by_age.csv"), df)
    return df
end

"""
    compute_training_share_target() -> DataFrame

The training_share target per window: the mean over the window's Fall years of
NCES four-year undergraduate enrolment (all attendance) divided by the mean
monthly CPS 16–64 population, with its across-Fall-year variance (n − 1, not
divided by the number of years). Errors if any Fall year in the window is
missing. Saves and returns training_share_target.csv.

The model's trainee state exits only by completion (φ) or demographic exit (ν),
so it is everyone enrolled in the programme that leads to the skilled state.
Hence four-year undergraduate only (the skilled state is BA+), part-time
included (a constant φ already represents slower completion), and no attrition
wedge (a wedge and φ would discount attrition twice; the stationary identity
admits no wedge below 1). The variance is the year-to-year spread because the
source is an administrative census with no sampling error to divide down.
"""
function compute_training_share_target()
    cps_path = joinpath(DERIVED_DIR, "cps_basic_clean.arrow")
    isfile(cps_path) ||
        error("cps_basic_clean.arrow not found in $DERIVED_DIR — run Stage 1 first.")
    cps = DataFrame(Arrow.Table(cps_path))

    enr_path = joinpath(RAW_NCES_DIR, "nces_4yr_undergrad_enrollment.csv")
    isfile(enr_path) || error(
        "nces_4yr_undergrad_enrollment.csv not found in $RAW_NCES_DIR. " *
        "It is NCES Digest 2024 Table 303.70, 4-year institutions, undergraduate total.")
    enr = CSV.read(enr_path, DataFrame)
    enr_by_year = Dict(Int(r.fall_year) => Float64(r.undergrad_4yr_total)
                       for r in eachrow(enr))

    rows = NamedTuple[]
    for w in WINDOWS_ORDER
        wd     = WINDOWS[w]
        y0, y1 = wd.ym_start ÷ 100, wd.ym_end ÷ 100

        sub = filter(r -> r.window == w, cps)
        per_month = combine(groupby(sub, [:YEAR, :MONTH]),
            :WTFINL => (x -> sum(coalesce.(x, 0.0))) => :pop)
        cps_pop = mean(per_month.pop)

        fall_years = sort([yr for yr in keys(enr_by_year) if y0 <= yr <= y1])
        length(fall_years) == y1 - y0 + 1 || error(
            "Window $(w) spans $(y0)-$(y1) but the enrolment file covers only " *
            "$(fall_years); a missing Fall year would silently shrink the average.")
        yearly = [enr_by_year[yr] / cps_pop for yr in fall_years]

        target = mean(yearly)
        svar = length(yearly) > 1 ?
               sum((yearly .- target).^2) / (length(yearly) - 1) : NaN

        push!(rows, (
            window          = w,
            label           = wd.label,
            fall_years      = "$(minimum(fall_years))–$(maximum(fall_years))",
            n_fall          = length(fall_years),
            cps_pop_16_64   = round(Int, cps_pop),
            training_share  = round(target; digits=6),
            sampling_var    = svar,
        ))
    end

    df = DataFrame(rows)
    println("\ntraining_share target — 4-year undergraduate enrolment / CPS 16–64 population")
    println("(all attendance, no attrition wedge: the training state is everyone enrolled)")
    println("─"^100)
    display(df)

    CSV.write(joinpath(DERIVED_DIR, "training_share_target.csv"), df)
    @info "  Saved derived/training_share_target.csv (level + across-year variance)"
    return df
end

"""
    _load_training_share_target(wname) -> (target, sampling_var)

The training_share level and its across-Fall-year variance for `wname`, read
from derived/training_share_target.csv. Errors rather than defaulting: the only
alternative would be the CPS count, which measures a different universe.
"""
function _load_training_share_target(wname::Symbol)
    path = joinpath(DERIVED_DIR, "training_share_target.csv")
    isfile(path) ||
        error("training_share_target.csv not found in $DERIVED_DIR — run Stage 3 first.")
    df = CSV.read(path, DataFrame)
    df.window = Symbol.(df.window)
    rows = filter(:window => ==(wname), df)
    isempty(rows) && error("No row for window=:$wname in $path.")
    return (target = Float64(rows.training_share[1]),
            sampling_var = Float64(rows.sampling_var[1]))
end

"""
    BACHELOR_PROGRAM_MONTHS

Nominal length of a bachelor's programme. 1/φ is this nominal length, not an
observed time to degree: pacing, stop-out and re-enrolment are already time
spent in the training state at the same constant hazard, and a programme
property can be held fixed across a crisis pair.
"""
const BACHELOR_PROGRAM_MONTHS = 48.0

"""
    calibrate_phi() -> Float64

The training-completion hazard φ = 1/`BACHELOR_PROGRAM_MONTHS` per month;
printed, saved to phi_calibration.csv, and returned.
"""
function calibrate_phi()
    phi = 1 / BACHELOR_PROGRAM_MONTHS
    @printf("  φ = %.6f per month  (1/%.0f, nominal four-year programme)\n",
            phi, BACHELOR_PROGRAM_MONTHS)

    CSV.write(joinpath(DERIVED_DIR, "phi_calibration.csv"),
              DataFrame(phi = phi, program_months = BACHELOR_PROGRAM_MONTHS))
    return phi
end

############################################################
# data_and_descriptives/nsc.jl
#
# Education-side data: the training state's level, its completion rate, and one
# enrolment diagnostic.
#   • enrollment_rate_by_age — CPS enrolment-rate diagnostic by age band;
#   • compute_training_share_target — per-window training_share target from
#     four-year undergraduate enrolment (consumed by Stage 9 / Stage 10);
#   • _load_training_share_target — reads that level and its variance back;
#   • calibrate_phi — the training-completion rate φ.
#
# Reads:  data/raw/nces/nces_4yr_undergrad_enrollment.csv, cps_basic_clean.arrow
# Writes: enrollment_rate_by_age.csv, training_share_target.csv, phi_calibration.csv
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages and path consts come from data_processing_main.jl.
############################################################

# ──────────────────────────────────────────────────────────────────────────
# Diagnostic: CPS enrolment rate by age band × window
#
# Shows the share of the population that is enrolled (in_training) for
# each age band within each window. The SCHLCOLL universe expanded from
# 16–24 to 16–54 in Jan 2013, so we expect the FC windows (base_fc,
# crisis_fc) to show ~zero enrolment for ages 25+ — that is the
# structural-zero signature, not a behavioural fact about older workers.
# The COVID windows (base_covid, crisis_covid) should show non-trivial
# enrolment up through age 54 and ~zero only at 55–64.
# ──────────────────────────────────────────────────────────────────────────

function enrollment_rate_by_age()
    cps_path = joinpath(DERIVED_DIR, "cps_basic_clean.arrow")
    isfile(cps_path) ||
        error("cps_basic_clean.arrow not found in $DERIVED_DIR — run Stage 1 first.")
    cps = DataFrame(Arrow.Table(cps_path))
    cps.enrolled = coalesce.(cps.in_training, false)

    # Age bands, both edges inclusive.
    bands = [(16, 19), (20, 24), (25, 29), (30, 34),
             (35, 44), (45, 54), (55, 64)]
    band_label(lo, hi) = lo == hi ? "$(lo)" : "$(lo)–$(hi)"

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
        # Build a row with one column per band (enrolment rate)
        # plus the overall 16–64 rate at the end.
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



# ──────────────────────────────────────────────────────────────────────────
# Attrition wedge constants (see compute_training_share_target below)
# ──────────────────────────────────────────────────────────────────────────
# training_share target — four-year undergraduate enrolment over population
#
# The model's trainee state t is fed by newborns from the unskilled pool and has
# exactly two exits: completion at φ and demographic exit at ν. Nobody drops
# out, so t is simply everyone currently enrolled in the programme that confers
# market access:
#
#     training_share_w  =  mean over Fall years w of ( enr_4yr / CPS_pop_w )
#
# Three choices, each load-bearing:
#
#   (a) FOUR-YEAR UNDERGRADUATE, not total postsecondary. The skilled state is
#       EDUC ≥ 111, a bachelor's or above, so two-year enrolment does not lead
#       to it and graduate enrolment starts from it. Either one in the numerator
#       counts a student the training state cannot represent.
#
#   (b) ALL ATTENDANCE, full- and part-time together. φ is a single hazard on
#       the whole training stock, so a part-time student is one who completes
#       more slowly — which is what a constant hazard already represents — not
#       one to be excluded.
#
#   (c) NO ATTRITION WEDGE. A completion-probability wedge on the stock and a
#       completion hazard φ discount the same attrition twice. The stationary
#       identity also leaves no admissible room for one: inverting
#       skilled_share = (φ/ν)·t/(1−t) at the observed skilled share wants a
#       wedge of 1.008, 0.946, 1.090 and 1.183 across the four windows — three
#       of the four above the ceiling of 1. The single wedge minimising mean
#       |error| in φ/ν is 1.008, and the objective falls monotonically in f
#       below that, so 1 is the boundary optimum on (0, 1] and no admissible
#       wedge does better. It is not the per-window optimum: crisis_fc alone
#       wants 0.946.
#
# The variance is the across-Fall-year dispersion, undivided by the number of
# years: the source is an administrative census, so there is no sampling error
# to divide down and the year-to-year spread IS the uncertainty in the target.
# Same convention as ee_rate_S (sampling_variances.jl).
#
# A CPS-derived enrolment count is deliberately not used for the level: the
# SCHLCOLL universe is age-capped at 24 before 2013 and widens to 54 after, so
# it is not comparable across the two crisis pairs.
# ──────────────────────────────────────────────────────────────────────────

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
    _load_training_share_target(wname) → (target, sampling_var)

The training_share level and its across-Fall-year variance for `wname`,
read from derived/training_share_target.csv. Errors rather than defaulting: a
missing target would silently ship the raw CPS moment, which measures a different
universe (SCHLCOLL, age-capped before 2013) and is not the model's `t`.
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



# ──────────────────────────────────────────────────────────────────────────
# φ — the training-completion hazard
#
# 1/φ is the NOMINAL length of a bachelor's programme, not an observed mean
# time-to-degree. The training state has completion as its only exit, so a
# measured mean would fold in part-time pacing, stop-out and re-enrolment —
# all of which the model already represents as time spent in the state at this
# same constant hazard. Taking the nominal length keeps φ a property of the
# programme rather than of the current composition of enrolment, which is what
# makes it admissible to hold fixed across a crisis pair.
# ──────────────────────────────────────────────────────────────────────────

const BACHELOR_PROGRAM_MONTHS = 48.0

function calibrate_phi()
    phi = 1 / BACHELOR_PROGRAM_MONTHS
    @printf("  φ = %.6f per month  (1/%.0f, nominal four-year programme)\n",
            phi, BACHELOR_PROGRAM_MONTHS)

    CSV.write(joinpath(DERIVED_DIR, "phi_calibration.csv"),
              DataFrame(phi = phi, program_months = BACHELOR_PROGRAM_MONTHS))
    return phi
    return phi
end

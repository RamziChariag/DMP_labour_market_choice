############################################################
# data_processing/nsc.jl
#
# National Student Clearinghouse (NSC) / IPEDS enrolment. Everything that
# reads the raw NSC workbook lives here:
#   • enrollment_rate_by_age — CPS enrolment-rate diagnostic by age band;
#   • compute_training_share_target — per-window training_share target taken
#     DIRECTLY from NSC, attrition-adjusted (consumed by Stage 9 / Stage 10);
#   • _load_training_share_target — reads that level and its variance back;
#   • calibrate_phi — training-completion rate φ from IPEDS Universe counts.
#
# Reads:  data/raw/nsc/*.xlsx, cps_basic_clean.arrow
# Writes: enrollment_rate_by_age.csv, training_share_scale.csv, cps_vs_nsc_enrollment.csv, phi_calibration.csv
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
# q: NSC Research Center six-year completion rate, all starters pooled. Flat
#    across cohorts (63.3% for fall 2007, 62.8% for 2008, 62.2% for 2015 and
#    2017), which is what allows a single constant across all four windows.
#    Pooled rather than the full-time-starter rate (67.1%) because the target's
#    DENOMINATOR is NSC's total enrolment headcount, which contains full-time,
#    mixed and part-time starters alike; a full-time completion rate against an
#    all-starter headcount would mismatch numerator and denominator.
# d_c: months to completion — this project's own 1/φ from calibrate_phi below,
#    not an independent number, so the wedge is internally consistent with the
#    completion rate the model actually uses.
const NSC_COMPLETION_Q   = 0.622
const NSC_PROGRAM_MONTHS = 45.0

# ──────────────────────────────────────────────────────────────────────────
# training_share target — taken directly from NSC, attrition-adjusted
#
# The model's trainee state t is fed by newborns from the unskilled pool and has
# exactly two exits: completion at φ and demographic exit at ν. Nobody drops out.
# NSC's Fall enrolment census counts everyone enrolled, including students who
# will never complete, so the raw NSC share is NOT t. Multiplying by the
# completers' share of the enrolment STOCK, f, removes exactly that group:
#
#     training_share_w  =  (NSC_enr_w / CPS_pop_w) × f
#
# Two things this does NOT do, deliberately:
#
#   (a) It does not remove graduate students. NSC includes them (~13% of
#       enrolment) and the model does not require training to be a bachelor's
#       degree — t is a non-working, non-searching state acquiring market access,
#       which describes an enrolled graduate student. The cost is recorded: a
#       data grad student arrived from the skilled pool, whereas the model's t is
#       fed only from the unskilled one. Bounded, known, and one constant away
#       from being revisited.
#
#   (b) It does not divide the across-year variance by the number of Fall
#       observations. NSC is an administrative near-census, so there is no
#       meaningful sampling error to divide down; the year-to-year dispersion IS
#       the uncertainty about the target.
#
# This replaces the previous κ_w = NSC_enr / CPS_enr bridge. κ was designed to
# cancel so the target carried NSC's level while keeping a CPS-derived variance,
# but its CPS denominator moved with the SCHLCOLL universe (age-capped at 24
# before 2013) and with labour-force participation, so κ drifted +6% WITHIN each
# crisis pair and nearly cancelled the COVID fall in enrolment. Going direct to
# NSC removes the bridge, and f is constant across windows by construction, so
# both crisis signs are preserved exactly.
# ──────────────────────────────────────────────────────────────────────────

function compute_training_share_target()
    cps_path = joinpath(DERIVED_DIR, "cps_basic_clean.arrow")
    isfile(cps_path) ||
        error("cps_basic_clean.arrow not found in $DERIVED_DIR — run Stage 1 first.")
    cps = DataFrame(Arrow.Table(cps_path))
    cps.enrolled = coalesce.(cps.in_training, false)

    # NSC US-Overall IPEDS Universe row → headcount by Fall year
    nsc_files = filter(f -> endswith(f, ".xlsx"), readdir(RAW_NSC_DIR))
    isempty(nsc_files) && error("No NSC .xlsx in $RAW_NSC_DIR")
    nsc_path = joinpath(RAW_NSC_DIR, first(nsc_files))
    nsc = DataFrame(XLSX.readtable(nsc_path, "Enrollments"))
    rename!(nsc, string.(names(nsc)))
    us_overall = filter(r ->
        !ismissing(r["State or Region"]) &&
        !ismissing(r["Institution Sector"]) &&
        r["State or Region"]   == "United States" &&
        r["Institution Sector"] == "Overall", nsc)
    nrow(us_overall) == 1 || error("Expected exactly 1 US-Overall row in NSC.")

    nsc_by_year = Dict{Int, Float64}()
    for col in names(nsc)
        m = match(r"Fall[\s_]+(\d{4})[\s_]+IPEDS[\s_]+Universe", string(col))
        isnothing(m) && continue
        yr = parse(Int, m.captures[1])
        v  = us_overall[1, col]
        (ismissing(v) || isnothing(v)) && continue
        try
            nsc_by_year[yr] = Float64(v)
        catch
        end
    end
    isempty(nsc_by_year) &&
        error("No Fall_YYYY_IPEDS_Universe columns found in NSC Enrollments sheet.")

    # ── The attrition wedge f ────────────────────────────────────────────────
    # Survival to program end under a constant attrition hazard α must equal the
    # observed completion rate q:  exp(−α·d_c) = q  ⇒  α = −ln(q)/d_c.
    # Expected time enrolled for someone who leaves before d_c is the mean of the
    # exponential truncated at d_c:  d_d = 1/α − d_c·q/(1−q).
    # Little's law then gives the completers' share of the enrolment STOCK, which
    # is what NSC's Fall census measures:
    α   = -log(NSC_COMPLETION_Q) / NSC_PROGRAM_MONTHS
    d_d = 1 / α - NSC_PROGRAM_MONTHS * NSC_COMPLETION_Q / (1 - NSC_COMPLETION_Q)
    f   = NSC_COMPLETION_Q * NSC_PROGRAM_MONTHS /
          (NSC_COMPLETION_Q * NSC_PROGRAM_MONTHS + (1 - NSC_COMPLETION_Q) * d_d)
    0.78 < f < 0.79 || error("attrition wedge f = $f outside the expected 0.78–0.79; " *
                             "check NSC_COMPLETION_Q and NSC_PROGRAM_MONTHS.")

    rows = NamedTuple[]
    for w in WINDOWS_ORDER
        wd     = WINDOWS[w]
        y0, y1 = wd.ym_start ÷ 100, wd.ym_end ÷ 100

        sub = filter(r -> r.window == w, cps)
        per_month = combine(groupby(sub, [:YEAR, :MONTH]),
            :WTFINL => (x -> sum(coalesce.(x, 0.0))) => :pop)
        cps_pop = mean(per_month.pop)

        nsc_years = sort([yr for yr in keys(nsc_by_year) if y0 <= yr <= y1])
        isempty(nsc_years) && error("No NSC Fall year inside window $(w) ($(y0)-$(y1)).")
        yearly_raw = [nsc_by_year[yr] / cps_pop for yr in nsc_years]   # per Fall
        yearly     = f .* yearly_raw                                   # completers only

        target = mean(yearly)
        # Across-Fall-year variance, undivided by the number of years: NSC is an
        # administrative near-census, so its sampling error is negligible and the
        # honest uncertainty in the target is the year-to-year dispersion of the
        # series. Same convention as ee_rate_S (sampling_variances.jl).
        svar = length(yearly) > 1 ?
               sum((yearly .- target).^2) / (length(yearly) - 1) : NaN

        push!(rows, (
            window          = w,
            label           = wd.label,
            nsc_fall_years  = "$(minimum(nsc_years))–$(maximum(nsc_years))",
            n_fall          = length(nsc_years),
            cps_pop_16_64   = round(Int, cps_pop),
            nsc_share_raw   = round(mean(yearly_raw); digits=6),
            attrition_f     = round(f; digits=4),
            training_share  = round(target; digits=6),
            sampling_var    = svar,
        ))
    end

    df = DataFrame(rows)
    println("\ntraining_share target — NSC IPEDS Universe, attrition-adjusted")
    println("(target = NSC_enr/CPS_pop × f: f is the completers' share of the enrolment stock)")
    @printf(" f = %.4f  from q = %.3f, d_c = %.0f months  =>  α = %.5f/month, d_d = %.1f months\n",
            f, NSC_COMPLETION_Q, NSC_PROGRAM_MONTHS, α, d_d)
    println("─"^100)
    display(df)

    CSV.write(joinpath(DERIVED_DIR, "training_share_target.csv"), df)
    @info "  Saved derived/training_share_target.csv (level + across-year variance)"
    return df
end

"""
    _load_training_share_target(wname) → (target, sampling_var)

The NSC-based training_share level and its across-Fall-year variance for `wname`,
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



function calibrate_phi()
    @info "Calibrating φ from NSC data..."

    nsc_files = filter(f -> endswith(f, ".xlsx"), readdir(RAW_NSC_DIR))
    isempty(nsc_files) && error("No .xlsx files found in $RAW_NSC_DIR")
    nsc_path = joinpath(RAW_NSC_DIR, first(nsc_files))

    @info "  Reading path"

    # XLSX.eachtablerow requires an Excel Table (ListObject); the NSC sheet is a plain
    # range, so we use XLSX.readtable instead, which works on any rectangular range.
    data = DataFrame(XLSX.readtable(nsc_path, "Enrollments"))

    # XLSX.readtable may return Symbol column names; normalise to String for matching.
    rename!(data, string.(names(data)))

    # Filter to US-level rows for 4-year and 2-year institutions
    us_4yr = filter(r ->
        !ismissing(r["State or Region"]) &&
        !ismissing(r["Institution Sector"]) &&
        r["State or Region"] == "United States" &&
        r["Institution Sector"] == "All 4-year Institutions",
        data)

    us_2yr = filter(r ->
        !ismissing(r["State or Region"]) &&
        !ismissing(r["Institution Sector"]) &&
        r["State or Region"] == "United States" &&
        r["Institution Sector"] == "All 2-year Institutions",
        data)

    @assert nrow(us_4yr) == 1 "Expected 1 US 4-year row, got $(nrow(us_4yr))"
    @assert nrow(us_2yr) == 1 "Expected 1 US 2-year row, got $(nrow(us_2yr))"

    # Collect IPEDS Universe enrollment across all available years.
    # Column names are "Fall_2003_IPEDS_Universe" (early years, underscores)
    # and "Fall 2017 IPEDS Universe" (later years, spaces) — occursin handles both.
    ipeds_cols_4yr = Float64[]
    ipeds_cols_2yr = Float64[]
    for col in names(data)
        sc = string(col)
        (occursin("IPEDS", sc) && occursin("Universe", sc)) || continue
        v4 = us_4yr[1, col]
        v2 = us_2yr[1, col]
        # Skip missing, nothing, or non-numeric (trailing "(*estimated)" column)
        (ismissing(v4) || isnothing(v4) || ismissing(v2) || isnothing(v2)) && continue
        try
            push!(ipeds_cols_4yr, Float64(v4))
            push!(ipeds_cols_2yr, Float64(v2))
        catch
        end
    end

    isempty(ipeds_cols_4yr) &&
        error("No IPEDS Universe columns found — check column names in NSC file")
    length(ipeds_cols_4yr) != length(ipeds_cols_2yr) &&
        error("Mismatched column counts: 4yr=$(length(ipeds_cols_4yr)), 2yr=$(length(ipeds_cols_2yr))")

    E_4yr = mean(ipeds_cols_4yr)
    E_2yr = mean(ipeds_cols_2yr)

    println("  Average IPEDS Universe enrollment ($(length(ipeds_cols_4yr)) years):")
    println("    4-year: $(round(Int, E_4yr))")
    println("    2-year: $(round(Int, E_2yr))")

    # NCES median time-to-degree (months): 49 for bachelor's, 37 for associate's
    d_4yr = 49.0
    d_2yr = 37.0

    # φ = [E_4yr*(1/d_4yr) + E_2yr*(1/d_2yr)] / (E_4yr + E_2yr)
    phi = (E_4yr / d_4yr + E_2yr / d_2yr) / (E_4yr + E_2yr)

    println("  φ = $(round(phi; digits=6)) (monthly completion probability)")
    println("  Implied mean duration = $(round(1/phi; digits=1)) months")

    CSV.write(joinpath(DERIVED_DIR, "phi_calibration.csv"),
              DataFrame(phi=phi, E_4yr=E_4yr, E_2yr=E_2yr, d_4yr=d_4yr, d_2yr=d_2yr))
    return phi
end

############################################################
# data_processing/nsc.jl
#
# National Student Clearinghouse (NSC) / IPEDS enrolment. Everything that
# reads the raw NSC workbook lives here:
#   • enrollment_rate_by_age — CPS enrolment-rate diagnostic by age band;
#   • compute_cps_nsc_scale  — per-window κ_w = NSC_enr / CPS_enr level
#     adjustment for training_share (consumed by Stage 7 / Stage 8);
#   • _load_training_share_scale — reads κ_w back for the moment/Σ̂ stages;
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

    # Age bands. Edit as needed; left edge inclusive, right edge inclusive.
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
# CPS vs NSC enrolment — per-window level adjustment (κ)
#
# Motivation. The CPS training_share numerator is bounded by the SCHLCOLL
# question, whose age universe was 16–24 through 2012 and expanded to 16–54
# in 2013. That break sits between the FC and COVID windows, so raw CPS
# training_share is NOT comparable across windows. NSC IPEDS-Universe counts
# all Title-IV postsecondary enrolments regardless of age, so it gives the
# age-comparable level we need.
#
# Strategy (Option B from the design discussion):
#   • Keep CPS training_share for its rich within-window covariance with all
#     other CPS moments — DO NOT replace the moment with an NSC observation
#     (NSC has only 3–5 Fall obs per window → unusable for Σ̂).
#   • Treat NSC as a deterministic level anchor. Compute per-window
#         κ_w  =  NSC_enr_w / CPS_enr_w
#     and rescale the training_share moment AND its influence function by κ_w.
#     Because κ_w is treated as constant (NSC ≈ census, ~90%+ IPEDS coverage),
#     the entire CPS off-diagonal covariance structure is preserved up to a
#     scalar multiple κ_w on the training_share row/column.
#
# This cell:
#   (1) builds the diagnostic table (raw CPS, NSC, κ, adjusted CPS share);
#   (2) saves derived/training_share_scale.csv for Stage 7 (moments) and
#       Stage 8 (Σ̂) to consume.
#
# Caveat. The CPS in_training flag here excludes EDUC ≥ 111 (grad students),
# while NSC IPEDS Universe includes them. The implied κ therefore absorbs the
# ~13% grad-student offset in addition to the age-universe expansion. This is
# fine for a level adjustment, but document the convention if you report κ_w.
# ──────────────────────────────────────────────────────────────────────────

function compute_cps_nsc_scale()
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

    rows = NamedTuple[]
    for w in WINDOWS_ORDER
        wd     = WINDOWS[w]
        y0, y1 = wd.ym_start ÷ 100, wd.ym_end ÷ 100

        # CPS monthly enrolment & population, then averaged across window
        sub = filter(r -> r.window == w, cps)
        per_month = combine(groupby(sub, [:YEAR, :MONTH]),
            :WTFINL => (x -> sum(coalesce.(x, 0.0))) => :pop,
            [:WTFINL, :enrolled] =>
                ((wt, e) -> sum(coalesce.(wt, 0.0) .* e)) => :enr)
        cps_pop   = mean(per_month.pop)
        cps_enr   = mean(per_month.enr)
        cps_share = cps_enr / cps_pop

        # NSC average IPEDS Universe across Fall years inside the window
        nsc_years = sort([yr for yr in keys(nsc_by_year) if y0 <= yr <= y1])
        nsc_enr   = isempty(nsc_years) ? NaN :
                    mean([nsc_by_year[yr] for yr in nsc_years])
        nsc_share = isnan(nsc_enr) ? NaN : nsc_enr / cps_pop

        kappa     = isnan(nsc_enr) ? NaN : nsc_enr / cps_enr
        adj_share = isnan(kappa)   ? NaN : kappa * cps_share

        push!(rows, (
            window           = w,
            label            = wd.label,
            cps_pop_16_64    = round(Int, cps_pop),
            cps_enrolled     = round(Int, cps_enr),
            cps_share_raw    = round(cps_share; digits=4),
            nsc_fall_years   = isempty(nsc_years) ? "—" :
                               "$(minimum(nsc_years))–$(maximum(nsc_years))",
            nsc_enrolled     = isnan(nsc_enr)   ? missing : round(Int, nsc_enr),
            nsc_share        = isnan(nsc_share) ? missing : round(nsc_share; digits=4),
            kappa            = isnan(kappa)     ? missing : round(kappa; digits=4),
            cps_share_adj    = isnan(adj_share) ? missing : round(adj_share; digits=4),
        ))
    end

    df = DataFrame(rows)
    println("\nCPS training_share vs NSC — per-window level adjustment")
    println("(cps_share_adj = κ · cps_share_raw, where κ = NSC_enr / CPS_enr)")
    println("─"^100)
    display(df)

    # Persist κ for Stage 7 (rescale training_share level) and Stage 8
    # (rescale the training_share row/column of Σ̂ by κ²; off-diagonals by κ).
    scale_df = DataFrame(
        window               = String.(df.window),
        kappa_training_share = [ismissing(k) ? NaN : Float64(k) for k in df.kappa],
    )
    CSV.write(joinpath(DERIVED_DIR, "training_share_scale.csv"), scale_df)
    CSV.write(joinpath(DERIVED_DIR, "cps_vs_nsc_enrollment.csv"), df)
    @info "  Saved derived/training_share_scale.csv  (apply κ to training_share level + IF)"
    @info "  Saved derived/cps_vs_nsc_enrollment.csv (full diagnostic table)"

    return df
end

"""
    _load_training_share_scale(wname) → Float64

κ_w for `wname`, read from derived/training_share_scale.csv (written by
compute_cps_nsc_scale above). Mirrors load_training_share_scale in
moments.jl: returns 1.0 with a warning if the file/row is missing or κ
is non-finite or ≤ 0. Used by Stage 7 (training_share level) and Stage 8
(Σ̂ row/col) to bake the NSC level adjustment into the saved CSVs.
"""
function _load_training_share_scale(wname::Symbol)::Float64
    path = joinpath(DERIVED_DIR, "training_share_scale.csv")
    if !isfile(path)
        @warn "training_share_scale.csv not found in $DERIVED_DIR — using κ = 1.0."
        return 1.0
    end
    df = CSV.read(path, DataFrame)
    df.window = Symbol.(df.window)
    rows = filter(:window => ==(wname), df)
    if isempty(rows)
        @warn "No row for window=:$wname in $path — using κ = 1.0."
        return 1.0
    end
    κ = Float64(rows.kappa_training_share[1])
    if !isfinite(κ) || κ <= 0
        @warn "κ for :$wname is non-finite or non-positive — using κ = 1.0."
        return 1.0
    end
    return κ
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

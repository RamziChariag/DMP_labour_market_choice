############################################################
# code/data_and_descriptives/cps_asec.jl
#
# Cleans the raw IPUMS CPS ASEC extract into the wage sample: wage and salary
# workers aged 16–64, full-time-equivalent weekly earnings in 2013 dollars,
# trimmed at the 1st/99th weighted percentiles within each window. Wages are
# kept in levels (no normalisation); the model's scale A absorbs the dollar level.
#
# Entry point: clean_cps_asec()
#   reads  data/raw/cps_asec/ (one .arrow or .csv[.gz])
#   writes cps_asec_clean.arrow
# Definitions only; packages and path constants come from data_processing_main.jl.
############################################################

"""
    clean_cps_asec() -> DataFrame

Load the raw ASEC extract, restrict to the supplement and to the survey years
spanned by `WINDOWS`, build real weekly earnings `real_wage` (hourly wage × 40,
2013 dollars), assign windows, trim within window, and save and return the
frame with `wage_norm` (equal to `real_wage`), `skilled`, `ASECWT` and `CPSIDP`.
"""
function clean_cps_asec()
    @info "Stage 2: clean_cps_asec — reading raw data..."

    raw_files  = readdir(RAW_CPS_ASEC_DIR)
    csv_file   = filter(f -> endswith(f, ".csv") || endswith(f, ".csv.gz"), raw_files)
    arrow_file = filter(f -> endswith(f, ".arrow"), raw_files)

    if !isempty(arrow_file)
        df = DataFrame(Arrow.Table(joinpath(RAW_CPS_ASEC_DIR, first(arrow_file))))
    elseif !isempty(csv_file)
        df = CSV.read(joinpath(RAW_CPS_ASEC_DIR, first(csv_file)), DataFrame)
    else
        error("No CSV or Arrow file found in $(RAW_CPS_ASEC_DIR)")
    end
    @info "  Raw records: $(nrow(df))"

    rename!(df, [Symbol(uppercase(string(c))) => c for c in names(df)]...)

    if hasproperty(df, :ASECFLAG)
        filter!(row -> coalesce(row.ASECFLAG, 0) == 1, df)
    end

    # Survey-year bounds from WINDOWS, never a literal (a hardcoded ceiling once
    # dropped ASEC 2023 from crisis_covid).
    asec_lo = minimum(minimum(wd.asec_years) for wd in values(WINDOWS))
    asec_hi = maximum(maximum(wd.asec_years) for wd in values(WINDOWS))
    filter!(row -> asec_lo <= row.YEAR <= asec_hi, df)
    @info "  After year filter (ASEC $(asec_lo)–$(asec_hi), from WINDOWS): $(nrow(df))"
    sort!(df, :YEAR)

    # ── Sample restrictions ──────────────────────────────────────
    key_cols = [:AGE, :CLASSWKR, :INCWAGE, :WKSWORK1, :UHRSWORKLY,
                :ASECWT, :EDUC, :YEAR, :SEX]
    present_key_cols = intersect(key_cols, Symbol.(names(df)))
    dropmissing!(df, present_key_cols)
    df.ASECWT = coalesce.(df.ASECWT, 0.0)
    filter!(row -> in_age_range(row.AGE), df)
    filter!(row -> is_wage_worker(row.CLASSWKR), df)
    filter!(row -> row.INCWAGE > 0 && row.WKSWORK1 > 0 && row.UHRSWORKLY > 0, df)
    @info "  After sample restrictions: $(nrow(df))"

    # ── Full-time-equivalent weekly earnings ─────────────────────
    # Hourly rate × a standard 40-hour week, so part-time hours do not enter
    # the wage distribution.
    df.hourly_wage = compute_hourly_wage.(df.INCWAGE, df.WKSWORK1, df.UHRSWORKLY)
    filter!(row -> isfinite(row.hourly_wage) && row.hourly_wage > 0.0, df)
    ft_hours = 40.0
    df.weekly_wage = df.hourly_wage .* ft_hours
    filter!(row -> isfinite(row.weekly_wage) && row.weekly_wage > 0.0, df)

    # ── Deflate to 2013 dollars ──────────────────────────────────
    if hasproperty(df, :CPI99)
        idx_2013 = findfirst(==(2013), df.YEAR)
        cpi_2013 = isnothing(idx_2013) ? nothing : df.CPI99[idx_2013]
        if isnothing(cpi_2013) || ismissing(cpi_2013) || iszero(cpi_2013)
            cpi_2013 = 1.0
            @warn "  Could not identify CPI99 base — wages not deflated"
        end
        # IPUMS CPI99 is a reciprocal-type factor (≈1 in 1999, falling over
        # time), hence × CPI99[t] / CPI99[2013]; deflate_wage would invert it.
        df.real_wage = df.weekly_wage .* Float64.(df.CPI99) ./ Float64(cpi_2013)
    else
        @warn "  CPI99 not in data — wages remain nominal"
        df.real_wage = df.weekly_wage
    end

    df.skilled = is_skilled.(df.EDUC)

    # ── Window assignment ────────────────────────────────────────
    df.window = assign_asec_window.(df.YEAR)
    @info "  In estimation windows: $(count(w -> w != :none, df.window)) / $(nrow(df))"

    # ── Trimming: 1st/99th weighted percentile within each window ─
    df.trimmed = fill(false, nrow(df))
    for wname in keys(WINDOWS)
        mask = df.window .== wname
        !any(mask) && continue
        wages_w   = df.real_wage[mask]
        weights_w = Float64.(df.ASECWT[mask])
        lo, hi    = trim_bounds(wages_w, weights_w)
        for i in findall(mask)
            if df.real_wage[i] < lo || df.real_wage[i] > hi
                df.trimmed[i] = true
            end
        end
        println("  Window $wname: trim bounds [$(round(lo; digits=2)), $(round(hi; digits=2))], trimmed $(count(df.trimmed[mask]))")
    end
    filter!(row -> !row.trimmed, df)

    # ── Levels, not normalised: the window median is printed only ─
    df.wage_norm = copy(df.real_wage)
    for wname in keys(WINDOWS)
        mask = df.window .== wname
        !any(mask) && continue
        med_w = wmedian(df.real_wage[mask], Float64.(df.ASECWT[mask]))
        isfinite(med_w) && println("  Window $wname: median real weekly wage = \$$(round(med_w; digits=2)) (level kept, not normalised)")
    end

    # ── Select and save (CPSIDP feeds the ASEC person-clustering) ─
    cols_to_keep = intersect(
        [:YEAR, :CPSIDP, :ASECWT, :AGE, :SEX, :EDUC,
         :skilled, :real_wage, :wage_norm, :window],
        Symbol.(names(df))
    )
    select!(df, cols_to_keep)

    outpath = joinpath(DERIVED_DIR, "cps_asec_clean.arrow")
    Arrow.write(outpath, df)
    return df
end

############################################################
# data_processing/cps_asec.jl
#
# Stage 2 — load and clean the raw CPS ASEC extract: wage-worker sample,
# hourly-wage construction, CPI deflation, per-window 1/99 trimming, and
# normalisation by the pooled within-window median.
#
# Reads:  data/raw/cps_asec/
# Writes: cps_asec_clean.arrow
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages and path consts come from data_processing_main.jl.
############################################################

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

    # ── ASEC supplement filter ────────────────────────────────────
    if hasproperty(df, :ASECFLAG)
        n_nonsup = count(row -> coalesce(row.ASECFLAG, 0) != 1, eachrow(df))
        filter!(row -> coalesce(row.ASECFLAG, 0) == 1, df)
        #@info "  Non-supplement records dropped: $n_nonsup"
    end

    filter!(row -> 2003 <= row.YEAR <= 2022, df)
    @info "  After year filter (ASEC 2003–2022): $(nrow(df))"
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

    # ── Construct hourly wage, then full-time-equivalent weekly earnings ──
    df.hourly_wage = compute_hourly_wage.(df.INCWAGE, df.WKSWORK1, df.UHRSWORKLY)
    filter!(row -> isfinite(row.hourly_wage) && row.hourly_wage > 0.0, df)
    # Full-time-equivalent weekly earnings: hourly rate times a standard
    # full-time week, applied to every worker so part-time hours variation does
    # not enter the wage distribution.
    ft_hours = 40.0
    df.weekly_wage = df.hourly_wage .* ft_hours
    filter!(row -> isfinite(row.weekly_wage) && row.weekly_wage > 0.0, df)

    # ── Deflate to constant (2012) dollars ───────────────────────
    if hasproperty(df, :CPI99)
        idx_2013 = findfirst(==(2013), df.YEAR)
        cpi_2012 = isnothing(idx_2013) ? nothing : df.CPI99[idx_2013]
        if isnothing(cpi_2012) || ismissing(cpi_2012) || iszero(cpi_2012)
            cpi_2012 = 1.0
            @warn "  Could not identify CPI99 base — wages not deflated"
        end
        # IPUMS CPI99 is a reciprocal-type factor (≈1.0 in 1999, falling over
        # time), so real 2013 dollars are weekly_wage × CPI99[t] / CPI99[2013].
        # This is the direct idiom rather than deflate_wage, which expects a
        # standard increasing CPI (as SIPP uses) and would invert the direction.
        df.real_wage = df.weekly_wage .* Float64.(df.CPI99) ./ Float64(cpi_2012)
    else
        @warn "  CPI99 not in data — wages remain nominal"
        df.real_wage = df.weekly_wage
    end

    df.skilled = is_skilled.(df.EDUC)

    # ── Window assignment ────────────────────────────────────────
    df.window = assign_asec_window.(df.YEAR)
    @info "  In estimation windows: $(count(w -> w != :none, df.window)) / $(nrow(df))"

    # ── Trimming: 1st/99th percentile within each window ─────────
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

    # ── No normalisation: wage_norm holds real weekly-earnings LEVELS ──
    #    (LMR-style; the model's aggregate scale A absorbs the dollar level.
    #     A per-window median is printed as a diagnostic only — NOT divided out.)
    df.wage_norm = copy(df.real_wage)
    for wname in keys(WINDOWS)
        mask = df.window .== wname
        !any(mask) && continue
        med_w = wmedian(df.real_wage[mask], Float64.(df.ASECWT[mask]))
        isfinite(med_w) && println("  Window $wname: median real weekly wage = \$$(round(med_w; digits=2)) (level kept, not normalised)")
    end

    # ── Select and save ──────────────────────────────────────────
    cols_to_keep = intersect(
        [:YEAR, :ASECWT, :AGE, :SEX, :EDUC,
         :skilled, :real_wage, :wage_norm, :window],
        Symbol.(names(df))
    )
    select!(df, cols_to_keep)

    outpath = joinpath(DERIVED_DIR, "cps_asec_clean.arrow")
    Arrow.write(outpath, df)
    #@info "  Saved: $outpath  ($(nrow(df)) rows)"
    return df
end

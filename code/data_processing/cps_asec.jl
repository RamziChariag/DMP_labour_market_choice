############################################################
# cps_asec.jl — Stage 2: Clean CPS ASEC
#
# Reads raw IPUMS CPS ASEC extract.
# Restricts to wage workers, constructs real hourly wages,
# deflates to constant dollars, trims, normalises by
# pooled weighted median within each window.
# Saves cleaned Arrow file to derived/.
#
# Requires: helpers.jl included first.
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
        filter!(row -> coalesce(row.ASECFLAG, 0) == 1, df)
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

    # ── Construct hourly wages ───────────────────────────────────
    df.hourly_wage = compute_hourly_wage.(df.INCWAGE, df.WKSWORK1, df.UHRSWORKLY)
    filter!(row -> isfinite(row.hourly_wage) && row.hourly_wage > 0.0, df)

    # ── Deflate to constant (2012) dollars ───────────────────────
    if hasproperty(df, :CPI99)
        idx_2013 = findfirst(==(2013), df.YEAR)
        cpi_2012 = isnothing(idx_2013) ? nothing : df.CPI99[idx_2013]
        if isnothing(cpi_2012) || ismissing(cpi_2012) || iszero(cpi_2012)
            cpi_2012 = 1.0
            @warn "  Could not identify CPI99 base — wages not deflated"
        end
        df.real_wage = deflate_wage.(df.hourly_wage, Float64.(df.CPI99), Float64(cpi_2012))
    else
        @warn "  CPI99 not in data — wages remain nominal"
        df.real_wage = df.hourly_wage
    end

    df.skilled = is_skilled.(df.EDUC)

    # ── Window assignment (by ASEC survey year) ──────────────────
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

    # ── Normalisation: divide by pooled weighted median ──────────
    df.wage_norm = fill(NaN, nrow(df))
    for wname in keys(WINDOWS)
        mask = df.window .== wname
        !any(mask) && continue
        wages_w   = df.real_wage[mask]
        weights_w = Float64.(df.ASECWT[mask])
        med_w     = wmedian(wages_w, weights_w)
        if isfinite(med_w) && med_w > 0.0
            df.wage_norm[mask] .= df.real_wage[mask] ./ med_w
        else
            df.wage_norm[mask] .= df.real_wage[mask]
        end
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
    @info "  Saved: $outpath  ($(nrow(df)) rows)"
    return df
end

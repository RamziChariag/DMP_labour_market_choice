############################################################
# clean_cps_asec.jl — Clean CPS ASEC extract (wage moments)
#
# Input:  data/raw/cps_asec/   (IPUMS ASEC extract)
# Output: data/derived/cps_asec_clean.arrow
#
# What this script does
# ─────────────────────
# 1. Read the raw IPUMS CPS ASEC extract.
# 2. Restrict to age 16–64, wage/salary workers,
#    positive income/weeks/hours.
# 3. Construct real hourly wages (deflated to 2012 dollars).
# 4. Classify skill: BA+ vs. less-than-BA.
# 5. Assign each observation to an estimation window
#    (using ASEC survey year → income year mapping).
# 6. Within each window, trim wages at 1st/99th percentiles.
# 7. Normalise wages: divide by pooled median within window.
# 8. Save cleaned data as Arrow.
#
# Required columns in raw data
# ────────────────────────────
#   YEAR, CPSID, CPSIDP, ASECWT, AGE, SEX, EDUC,
#   EMPSTAT, INCWAGE, WKSWORK1, UHRSWORKLY, CLASSWKR, CPI99
############################################################


"""
    clean_cps_asec() → DataFrame

Read, clean, and return the CPS ASEC wage panel.
Also saves data/derived/cps_asec_clean.arrow.
"""
function clean_cps_asec()

    @info "clean_cps_asec: reading raw data..."

    # ── 1. Read raw data ─────────────────────────────────────────────────
    raw_files = readdir(RAW_CPS_ASEC_DIR)
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

    # ── 2. Standardise column names ──────────────────────────────────────
    rename!(df, [Symbol(uppercase(string(c))) => c for c in names(df)]...)

    # ── 3. Sample restrictions ───────────────────────────────────────────
    filter!(row -> in_age_range(row.AGE), df)
    filter!(row -> is_wage_worker(row.CLASSWKR), df)
    filter!(row -> row.INCWAGE > 0 && row.WKSWORK1 > 0 && row.UHRSWORKLY > 0, df)

    @info "  After sample restrictions: $(nrow(df))"

    # ── 4. Construct hourly wage ─────────────────────────────────────────
    df.hourly_wage = compute_hourly_wage.(df.INCWAGE, df.WKSWORK1, df.UHRSWORKLY)
    filter!(row -> isfinite(row.hourly_wage) && row.hourly_wage > 0.0, df)

    # ── 5. Deflate to constant (2012) dollars ────────────────────────────
    #    CPI99 in IPUMS is the CPI-U-RS adjustment factor (value in year
    #    relative to 1999 = 1.0).  We pick a base year and deflate.
    #    If CPI99 is not available, wages remain nominal (user must supply
    #    a CPI series in the raw data).
    if hasproperty(df, :CPI99)
        # Normalise so that 2012 = 1.0
        cpi_2012 = df.CPI99[findfirst(==(2013), df.YEAR)]  # ASEC 2013 = income 2012
        if isnothing(cpi_2012) || ismissing(cpi_2012)
            # Fallback: use raw CPI values, user to verify
            cpi_2012 = 1.0
            @warn "Could not identify CPI99 for 2012; wages not deflated."
        end
        df.real_wage = deflate_wage.(df.hourly_wage, Float64.(df.CPI99), Float64(cpi_2012))
    else
        @warn "CPI99 not in data — wages remain nominal."
        df.real_wage = df.hourly_wage
    end

    # ── 6. Skill classification ──────────────────────────────────────────
    df.skilled = is_skilled.(df.EDUC)

    # ── 7. Window assignment (ASEC survey year → income year) ────────────
    #    ASEC reports income for the *previous* calendar year.
    #    So ASEC survey year Y covers income year Y-1.
    df.income_year = df.YEAR .- 1
    df.window = Vector{Symbol}(undef, nrow(df))
    fill!(df.window, :none)
    for i in 1:nrow(df)
        for (wname, wdef) in WINDOWS
            if in_asec_window(df.YEAR[i], wdef)
                df.window[i] = wname
                break
            end
        end
    end

    n_assigned = count(w -> w != :none, df.window)
    @info "  Observations in estimation windows: $n_assigned / $(nrow(df))"

    # ── 8. Trimming: 1st/99th percentile within each window ──────────────
    df.trimmed = fill(false, nrow(df))
    for wname in keys(WINDOWS)
        mask = df.window .== wname
        !any(mask) && continue
        wages_w   = df.real_wage[mask]
        weights_w = Float64.(df.ASECWT[mask])
        lo, hi = winsorize_bounds(wages_w, weights_w)
        for i in findall(mask)
            if df.real_wage[i] < lo || df.real_wage[i] > hi
                df.trimmed[i] = true
            end
        end
    end
    n_trimmed = count(df.trimmed)
    @info "  Trimmed observations: $n_trimmed"
    filter!(row -> !row.trimmed, df)

    # ── 9. Normalisation: divide by pooled median within each window ─────
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
            @warn "Median wage is invalid for window $wname"
            df.wage_norm[mask] .= df.real_wage[mask]
        end
    end

    # ── 10. Select and save ──────────────────────────────────────────────
    cols_to_keep = intersect(
        [:YEAR, :income_year, :ASECWT, :AGE, :SEX, :EDUC,
         :skilled, :real_wage, :wage_norm, :window],
        Symbol.(names(df))
    )
    select!(df, cols_to_keep)

    outpath = joinpath(DERIVED_DIR, "cps_asec_clean.arrow")
    Arrow.write(outpath, df)
    @info "  Saved: $outpath  ($(nrow(df)) rows)"

    return df
end

############################################################
# clean_cps_basic.jl — Clean CPS Basic Monthly extract
#
# Input:  data/raw/cps_basic/   (IPUMS extract, CSV or Arrow)
# Output: data/derived/cps_basic_clean.arrow
#
# What this script does
# ─────────────────────
# 1. Read the raw IPUMS CPS Basic Monthly extract.
# 2. Restrict to age 16–64, civilian labour force.
# 3. Classify skill: skilled = EDUC ≥ 111 (BA+).
# 4. Classify labour-force status: employed / unemployed.
# 5. Flag training: enrolled in college without BA.
# 6. Assign each observation to an estimation window.
# 7. Apply the COVID-era BLS misclassification correction.
# 8. Compute the industry-level skilled share (for JOLTS
#    vacancy allocation).
# 9. Save the cleaned panel as Arrow.
#
# Required columns in raw data
# ────────────────────────────
#   YEAR, MONTH, CPSID, CPSIDP, MISH, WTFINL,
#   EMPSTAT, LABFORCE, EDUC, SCHLCOLL, AGE, SEX,
#   RACE, IND, CLASSWKR
############################################################


"""
    clean_cps_basic() → DataFrame

Read, clean, and return the CPS Basic Monthly panel.
Also saves data/derived/cps_basic_clean.arrow.
"""
function clean_cps_basic()

    @info "clean_cps_basic: reading raw data..."

    # ── 1. Read raw data ─────────────────────────────────────────────────
    raw_files = readdir(RAW_CPS_BASIC_DIR)
    csv_file  = filter(f -> endswith(f, ".csv") || endswith(f, ".csv.gz"), raw_files)
    arrow_file = filter(f -> endswith(f, ".arrow"), raw_files)

    if !isempty(arrow_file)
        df = DataFrame(Arrow.Table(joinpath(RAW_CPS_BASIC_DIR, first(arrow_file))))
    elseif !isempty(csv_file)
        df = CSV.read(joinpath(RAW_CPS_BASIC_DIR, first(csv_file)), DataFrame)
    else
        error("No CSV or Arrow file found in $(RAW_CPS_BASIC_DIR)")
    end

    @info "  Raw records: $(nrow(df))"

    # ── 2. Standardise column names to uppercase ─────────────────────────
    rename!(df, [Symbol(uppercase(string(c))) => c for c in names(df)]...)

    # ── 3. Sample restrictions ───────────────────────────────────────────
    filter!(row -> in_age_range(row.AGE), df)
    filter!(row -> is_civilian_lf(row.LABFORCE), df)

    @info "  After age/LF restriction: $(nrow(df))"

    # ── 4. Skill classification ──────────────────────────────────────────
    df.skilled = is_skilled.(df.EDUC)

    # ── 5. Labour-force status ───────────────────────────────────────────
    df.employed   = is_employed.(df.EMPSTAT)
    df.unemployed = is_unemployed.(df.EMPSTAT)

    # ── 6. COVID BLS correction (Mar–Jun 2020) ───────────────────────────
    #    Reclassify employed-absent with reason "other" in mass-layoff
    #    industries as unemployed on temporary layoff.
    #    Full implementation requires WHYABSNT and industry flags.
    #    Placeholder: apply_covid_correction from utils.jl.
    for i in 1:nrow(df)
        corrected = apply_covid_correction(df.EMPSTAT[i], df.YEAR[i], df.MONTH[i])
        if corrected != df.EMPSTAT[i]
            df.employed[i]   = is_employed(corrected)
            df.unemployed[i] = is_unemployed(corrected)
        end
    end

    # ── 7. Training flag ─────────────────────────────────────────────────
    #    Enrolled in college without BA.
    if hasproperty(df, :SCHLCOLL)
        df.in_training = is_enrolled_no_ba.(df.SCHLCOLL, df.EDUC)
    else
        @warn "SCHLCOLL not in data — setting in_training = false"
        df.in_training = fill(false, nrow(df))
    end

    # ── 8. Valid panel-match flag ────────────────────────────────────────
    df.valid_match = valid_match_mish.(df.MISH)

    # ── 9. Estimation-window assignment ──────────────────────────────────
    df.window = Vector{Symbol}(undef, nrow(df))
    fill!(df.window, :none)
    for i in 1:nrow(df)
        for (wname, wdef) in WINDOWS
            if in_window(df.YEAR[i], df.MONTH[i], wdef)
                df.window[i] = wname
                break
            end
        end
    end

    n_assigned = count(w -> w != :none, df.window)
    @info "  Observations in estimation windows: $n_assigned / $(nrow(df))"

    # ── 10. Industry-level skilled share (for JOLTS allocation) ──────────
    #     For each window × 2-digit NAICS industry, compute the share of
    #     employed workers who are skilled.
    if hasproperty(df, :IND)
        emp_df = filter(row -> row.employed && row.window != :none, df)
        ind_shares = combine(
            groupby(emp_df, [:window, :IND]),
            :skilled => (s -> mean(s)) => :skilled_share_ind,
            :WTFINL  => sum => :weight_ind
        )
        # Save separately for JOLTS allocation
        Arrow.write(joinpath(DERIVED_DIR, "industry_skill_shares.arrow"),
                     ind_shares)
        @info "  Industry skill shares saved ($(nrow(ind_shares)) rows)"
    end

    # ── 11. Select and save ──────────────────────────────────────────────
    cols_to_keep = intersect(
        [:YEAR, :MONTH, :CPSID, :CPSIDP, :MISH, :WTFINL,
         :EMPSTAT, :EDUC, :AGE, :SEX, :IND,
         :skilled, :employed, :unemployed, :in_training,
         :valid_match, :window],
        Symbol.(names(df))
    )
    select!(df, cols_to_keep)

    outpath = joinpath(DERIVED_DIR, "cps_basic_clean.arrow")
    Arrow.write(outpath, df)
    @info "  Saved: $outpath  ($(nrow(df)) rows)"

    return df
end

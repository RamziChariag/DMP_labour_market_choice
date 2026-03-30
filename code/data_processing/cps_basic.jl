############################################################
# cps_basic.jl — Stage 1: Clean CPS Basic Monthly
#
# Reads raw IPUMS CPS Basic Monthly extract.
# Applies sample restrictions, classifies skill/employment,
# applies COVID BLS correction, computes industry skill shares.
# Saves cleaned Arrow file and industry skill shares to derived/.
#
# Requires: helpers.jl included first.
############################################################

function clean_cps_basic()
    @info "Stage 1: clean_cps_basic — reading raw data..."

    # ── 1. Read raw file ──────────────────────────────────────────
    raw_files  = readdir(RAW_CPS_BASIC_DIR)
    csv_file   = filter(f -> endswith(f, ".csv") || endswith(f, ".csv.gz"), raw_files)
    arrow_file = filter(f -> endswith(f, ".arrow"), raw_files)

    if !isempty(arrow_file)
        df = DataFrame(Arrow.Table(joinpath(RAW_CPS_BASIC_DIR, first(arrow_file))))
    elseif !isempty(csv_file)
        df = CSV.read(joinpath(RAW_CPS_BASIC_DIR, first(csv_file)), DataFrame)
    else
        error("No CSV or Arrow file found in $(RAW_CPS_BASIC_DIR)")
    end
    @info "  Raw records: $(nrow(df))"

    # Uppercase column names for consistency
    rename!(df, [Symbol(uppercase(string(c))) => c for c in names(df)]...)

    # ── 2. Sample restrictions ────────────────────────────────────
    filter!(row -> in_age_range(row.AGE), df)

    # Keep both LF (LABFORCE==2) and NILF (LABFORCE==1); drop NIU (LABFORCE==0).
    # We need NILF at t+1 in the lookup to observe LF→NILF transitions (for ν).
    filter!(row -> row.LABFORCE in (1, 2), df)
    df.in_lf = df.LABFORCE .== 2   # true = in labour force; false = NILF
    @info "  After age/civilian restriction: $(nrow(df)) (LF: $(count(df.in_lf)), NILF: $(count(.!df.in_lf)))"

    # Coalesce weights to avoid Missing propagation downstream
    df.WTFINL = coalesce.(df.WTFINL, 0.0)

    # ── 3. Classification ─────────────────────────────────────────
    df.skilled = is_skilled.(df.EDUC)

    # COVID BLS misclassification correction (Mar–Jun 2020)
    if hasproperty(df, :WHYABSNT)
        df.EMPSTAT_CORRECTED = [
            apply_covid_correction(df.EMPSTAT[i], df.YEAR[i], df.MONTH[i], df.WHYABSNT[i])
            for i in 1:nrow(df)
        ]
        n_corrected = count(df.EMPSTAT_CORRECTED .!= df.EMPSTAT)
        @info "  COVID correction: reclassified $n_corrected observations"
    else
        @warn "  WHYABSNT not in data — COVID correction skipped"
        df.EMPSTAT_CORRECTED = df.EMPSTAT
    end

    df.employed   = is_employed.(df.EMPSTAT_CORRECTED)
    df.unemployed = is_unemployed.(df.EMPSTAT_CORRECTED)

    # Training flag: SCHLCOLL ∈ {3, 4} AND EDUC < 111
    if hasproperty(df, :SCHLCOLL)
        df.in_training = is_enrolled_no_ba.(df.SCHLCOLL, df.EDUC)
    else
        @warn "  SCHLCOLL not in data — setting in_training = false"
        df.in_training = fill(false, nrow(df))
    end

    df.valid_match = valid_match_mish.(df.MISH)

    # ── 4. Window assignment (vectorised) ─────────────────────────
    df.window = assign_window.(df.YEAR, df.MONTH)
    lf_in_window = count(i -> df.window[i] != :none && df.in_lf[i], 1:nrow(df))
    @info "  LF observations in estimation windows: $lf_in_window / $(count(df.in_lf))"

    # ── 5. Industry skill shares (WTFINL-weighted, LF only) ───────
    if hasproperty(df, :IND)
        df.IND_JOLTS = ind_to_jolts_supersector.(df.IND)

        n_unknown  = count(df.IND_JOLTS .== "UNKNOWN")
        n_excluded = count(df.IND_JOLTS .== "EXCLUDED")
        @info "  IND mapping: $n_excluded EXCLUDED (agri/private HH/military), $n_unknown UNKNOWN"
        if n_unknown > 0
            unknown_inds = unique(df.IND[df.IND_JOLTS .== "UNKNOWN"])
            @warn "  Unknown IND codes: $unknown_inds"
        end

        # Skill shares: employed LF workers only
        emp_df = filter(row -> row.employed && row.window != :none &&
                               row.IND_JOLTS != "EXCLUDED" && row.IND_JOLTS != "UNKNOWN", df)

        ind_shares = combine(
            groupby(emp_df, [:window, :IND_JOLTS]),
            [:skilled, :WTFINL] => ((s, w) -> let ww = coalesce.(w, 0.0)
                                                   sum(ww .* s) / sum(ww) end) => :skilled_share_ind,
            :WTFINL => (w -> sum(coalesce.(w, 0.0))) => :weight_ind
        )
        Arrow.write(joinpath(DERIVED_DIR, "industry_skill_shares.arrow"), ind_shares)
        @info "  Industry skill shares saved ($(nrow(ind_shares)) rows, WTFINL-weighted)"

        # Economy-wide skill share per window (default fallback for unmatched industries)
        econ_shares = combine(
            groupby(filter(r -> r.window != :none, emp_df), :window),
            [:skilled, :WTFINL] => ((s, w) -> let ww = coalesce.(w, 0.0)
                                                   sum(ww .* s) / sum(ww) end) => :econ_skill_share
        )
        Arrow.write(joinpath(DERIVED_DIR, "economy_skill_shares.arrow"), econ_shares)
        @info "  Economy-wide skill shares:"
        for row in eachrow(econ_shares)
            println("    $(row.window): $(round(coalesce(row.econ_skill_share, NaN); digits=3))")
        end
    end

    # ── 6. Select columns and save ────────────────────────────────
    cols_to_keep = intersect(
        [:YEAR, :MONTH, :CPSID, :CPSIDP, :MISH, :WTFINL,
         :EMPSTAT, :EMPSTAT_CORRECTED, :EDUC, :AGE, :SEX, :IND,
         :skilled, :employed, :unemployed, :in_training,
         :valid_match, :window, :IND_JOLTS, :in_lf],
        Symbol.(names(df))
    )
    select!(df, cols_to_keep)

    outpath = joinpath(DERIVED_DIR, "cps_basic_clean.arrow")
    Arrow.write(outpath, df)
    @info "  Saved: $outpath  ($(nrow(df)) rows)"
    return df
end

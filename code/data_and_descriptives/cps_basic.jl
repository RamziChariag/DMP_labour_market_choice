############################################################
# code/data_and_descriptives/cps_basic.jl
#
# Cleans the raw IPUMS CPS Basic Monthly extract: sample restrictions, skill /
# employment / training flags, the COVID misclassification fix, window
# assignment, and the WTFINL-weighted industry skill shares used to split JOLTS
# openings between the two markets.
#
# Entry point: clean_cps_basic()
#   reads  data/raw/cps_basic/ (one .arrow or .csv[.gz])
#   writes cps_basic_clean.arrow, industry_skill_shares.arrow, economy_skill_shares.arrow
# Definitions only; packages and path constants come from data_processing_main.jl.
############################################################

"""
    clean_cps_basic() -> DataFrame

Load the raw CPS Basic extract and keep ages 16–64 in or out of the labour force
(NIU dropped; NILF kept for LF→NILF transitions and the trainee count). Adds
`in_lf`, `skilled`, `employed`/`unemployed` (on the COVID-corrected `EMPSTAT`),
`in_training`, `valid_match` and `window`; writes the industry and economy-wide
skill shares of employed non-trainees; saves and returns the cleaned frame.
"""
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

    rename!(df, [Symbol(uppercase(string(c))) => c for c in names(df)]...)

    # ── 2. Sample restrictions ────────────────────────────────────
    filter!(row -> in_age_range(row.AGE), df)

    # LABFORCE: 2 = in LF, 1 = NILF (kept), 0 = NIU (dropped).
    filter!(row -> row.LABFORCE in (1, 2), df)
    df.in_lf  = df.LABFORCE .== 2
    df.in_age = trues(nrow(df))
    @info "  After age/civilian restriction: $(nrow(df)) (LF: $(count(df.in_lf)), NILF: $(count(.!df.in_lf)))"

    df.WTFINL = coalesce.(df.WTFINL, 0.0)

    # ── 3. Classification ─────────────────────────────────────────
    df.skilled = is_skilled.(df.EDUC)

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

    if hasproperty(df, :SCHLCOLL)
        df.in_training = is_enrolled_no_ba.(df.SCHLCOLL, df.EDUC)
    else
        @warn "  SCHLCOLL not in data — setting in_training = false"
        df.in_training = fill(false, nrow(df))
    end

    df.valid_match = valid_match_mish.(df.MISH)

    # ── 4. Window assignment ──────────────────────────────────────
    df.window = assign_window.(df.YEAR, df.MONTH)
    lf_in_window = count(i -> df.window[i] != :none && df.in_lf[i], 1:nrow(df))
    @info "  LF observations in estimation windows: $lf_in_window / $(count(df.in_lf))"

    # ── 5. Industry skill shares (employed, not in training) ──────
    # Excluding working students mirrors the model's labour force, which
    # excludes the training mass, so the JOLTS split agrees with skilled_share.
    if hasproperty(df, :IND)
        df.IND_JOLTS = ind_to_jolts_supersector.(df.IND)

        # Diagnostic on in-window rows only: the pre-2003 buffer is coded in the
        # 1990 Census scheme, which the crosswalk does not recognise, but those
        # rows never enter a moment.
        in_window = df.window .!= :none
        n_unknown  = count(df.IND_JOLTS[in_window] .== "UNKNOWN")
        n_excluded = count(df.IND_JOLTS[in_window] .== "EXCLUDED")
        @info "  IND mapping (in-window): $n_excluded EXCLUDED (agri/private HH/military), $n_unknown UNKNOWN"
        if n_unknown > 0
            unknown_inds = unique(df.IND[in_window .& (df.IND_JOLTS .== "UNKNOWN")])
            @warn "  Unknown in-window IND codes: $unknown_inds"
        end

        emp_df = filter(row -> row.employed && !row.in_training &&
                               row.window != :none &&
                               row.IND_JOLTS != "EXCLUDED" && row.IND_JOLTS != "UNKNOWN", df)

        ind_shares = combine(
            groupby(emp_df, [:window, :IND_JOLTS]),
            [:skilled, :WTFINL] => ((s, w) -> let ww = coalesce.(w, 0.0)
                                                   sum(ww .* s) / sum(ww) end) => :skilled_share_ind,
            :WTFINL => (w -> sum(coalesce.(w, 0.0))) => :weight_ind
        )
        Arrow.write(joinpath(DERIVED_DIR, "industry_skill_shares.arrow"), ind_shares)
        @info "  Industry skill shares saved ($(nrow(ind_shares)) rows, LF∩¬train, WTFINL-weighted)"

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
    # CPSIDP: month-pair matching and person clustering; DURUNEMP: duration moments.
    cols_to_keep = intersect(
        [:YEAR, :MONTH, :CPSID, :CPSIDP, :MISH, :WTFINL,
         :EMPSTAT, :EMPSTAT_CORRECTED, :EDUC, :AGE, :SEX, :IND,
         :DURUNEMP,
         :skilled, :employed, :unemployed, :in_training,
         :valid_match, :window, :IND_JOLTS, :in_lf, :in_age],
        Symbol.(names(df))
    )
    select!(df, cols_to_keep)

    outpath = joinpath(DERIVED_DIR, "cps_basic_clean.arrow")
    Arrow.write(outpath, df)
    @info "  Saved: $outpath  ($(nrow(df)) rows)"
    return df
end

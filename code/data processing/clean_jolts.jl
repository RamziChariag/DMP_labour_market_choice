############################################################
# clean_jolts.jl — Clean JOLTS vacancy data
#
# Input:  data/raw/jolts/   (BLS flat files, CSV)
# Output: data/derived/jolts_clean.arrow
#
# What this script does
# ─────────────────────
# 1. Read BLS JOLTS flat files (total and by-industry
#    job openings, seasonally adjusted, level).
# 2. Parse series IDs to extract industry codes.
# 3. Reshape into a panel: (year, month, industry, openings).
# 4. Assign estimation windows.
# 5. Merge with CPS industry-level skilled shares to
#    allocate vacancies to skill segments.
# 6. Compute monthly V_U and V_S by window.
# 7. Save as Arrow.
#
# JOLTS series ID convention
# ──────────────────────────
#   JTS000000000000000JOL  — total nonfarm openings
#   JTS{industry}00000000JOL — by 2-digit NAICS
############################################################


"""
    clean_jolts() → DataFrame

Read, clean, and return the JOLTS vacancy panel with
skill-allocated vacancies.
Also saves data/derived/jolts_clean.arrow.
"""
function clean_jolts()

    @info "clean_jolts: reading raw data..."

    # ── 1. Read raw JOLTS files ──────────────────────────────────────────
    raw_files = readdir(RAW_JOLTS_DIR)
    csv_files = filter(f -> endswith(f, ".csv") || endswith(f, ".csv.gz"), raw_files)

    if isempty(csv_files)
        error("No CSV files found in $(RAW_JOLTS_DIR)")
    end

    # Read and concatenate all JOLTS files
    dfs = DataFrame[]
    for f in csv_files
        push!(dfs, CSV.read(joinpath(RAW_JOLTS_DIR, f), DataFrame))
    end
    df = vcat(dfs...; cols=:union)

    @info "  Raw records: $(nrow(df))"

    # ── 2. Standardise column names ──────────────────────────────────────
    rename!(df, [Symbol(uppercase(string(c))) => c for c in names(df)]...)

    # ── 3. Parse into panel format ───────────────────────────────────────
    #    Expected columns: SERIES_ID (or SERIESID), YEAR, PERIOD, VALUE
    #    PERIOD is M01–M12 for monthly data.
    #    Adjust column names to match BLS flat-file format.
    if hasproperty(df, :SERIES_ID)
        # Already in long format — good
    elseif hasproperty(df, :SERIESID)
        rename!(df, :SERIESID => :SERIES_ID)
    else
        # Try wide format: year columns
        @warn "Unexpected JOLTS format — attempting reshape"
    end

    # Extract month from PERIOD (M01 → 1, M02 → 2, etc.)
    if hasproperty(df, :PERIOD)
        df.MONTH = [parse(Int, replace(string(p), r"^M0?" => ""))
                     for p in df.PERIOD]
        filter!(row -> 1 <= row.MONTH <= 12, df)  # drop annual rows
    end

    # Extract industry from series ID
    # JTS{industry_code}00000000JOL
    df.industry_code = [_parse_jolts_industry(string(s))
                        for s in df.SERIES_ID]

    # ── 4. Keep only job openings (JOL) series ───────────────────────────
    filter!(row -> endswith(string(row.SERIES_ID), "JOL"), df)

    # ── 5. Assign estimation windows ─────────────────────────────────────
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

    # ── 6. Merge with CPS industry skill shares ─────────────────────────
    #    Load the shares computed in clean_cps_basic
    shares_path = joinpath(DERIVED_DIR, "industry_skill_shares.arrow")
    if isfile(shares_path)
        shares = DataFrame(Arrow.Table(shares_path))
        @info "  Loaded industry skill shares ($(nrow(shares)) rows)"

        # Join on window × industry
        df_merged = leftjoin(df, shares;
                             on = [:window => :window, :industry_code => :IND],
                             makeunique = true)

        # Fill missing shares with 0.5 (fallback)
        for i in 1:nrow(df_merged)
            if ismissing(df_merged.skilled_share_ind[i])
                df_merged.skilled_share_ind[i] = 0.5
            end
        end

        # Allocate vacancies
        df_merged.V_S = df_merged.VALUE .* df_merged.skilled_share_ind
        df_merged.V_U = df_merged.VALUE .* (1.0 .- df_merged.skilled_share_ind)
        df = df_merged
    else
        @warn "Industry skill shares not found — using equal split for V_S/V_U"
        df.V_S = df.VALUE .* 0.5
        df.V_U = df.VALUE .* 0.5
    end

    # ── 7. Aggregate: total V_S, V_U by (year, month, window) ───────────
    #    Sum across industries (excluding the total-nonfarm series to
    #    avoid double-counting).
    by_industry = filter(row -> row.industry_code != "000000", df)
    if nrow(by_industry) > 0
        agg = combine(
            groupby(by_industry, [:YEAR, :MONTH, :window]),
            :V_S => sum => :V_S,
            :V_U => sum => :V_U,
        )
    else
        # Fallback: use total nonfarm with 50/50 split
        total_rows = filter(row -> row.industry_code == "000000", df)
        agg = combine(
            groupby(total_rows, [:YEAR, :MONTH, :window]),
            :V_S => sum => :V_S,
            :V_U => sum => :V_U,
        )
    end

    # ── 8. Save ──────────────────────────────────────────────────────────
    outpath = joinpath(DERIVED_DIR, "jolts_clean.arrow")
    Arrow.write(outpath, agg)
    @info "  Saved: $outpath  ($(nrow(agg)) rows)"

    return agg
end


# ── Helper: parse industry code from JOLTS series ID ─────────────────────

"""
    _parse_jolts_industry(series_id) → String

Extract the industry portion from a JOLTS series ID.
Example: "JTS510000000000000JOL" → "510000" (Information sector)
Total nonfarm: "JTS000000000000000JOL" → "000000"
"""
function _parse_jolts_industry(sid::String) :: String
    # Series format: JTS{6-digit industry}{8 zeros}JOL
    # Total length ~21 chars.  Industry is chars 4–9.
    m = match(r"^JTS(\d{6,})\d*JOL$", sid)
    isnothing(m) && return "unknown"
    return m.captures[1]
end

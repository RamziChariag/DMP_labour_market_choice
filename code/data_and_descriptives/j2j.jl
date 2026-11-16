############################################################
# code/data_and_descriptives/j2j.jl
#
# Census Job-to-Job flows: the skilled employer-to-employer rate ee_rate_S from
# the national, seasonally adjusted, bachelor's-or-more (E4) cell.
#
# Entry point: import_j2j_ee_rates()
#   reads  data/raw/j2j/*.csv (first file)
#   writes j2j_ee_rates.csv: window, ee_rate_S (window mean of the monthly
#          rate), ee_rate_S_sd (across-quarter sd; NaN below 2 quarters), n_quarters
# Definitions only; packages and path constants come from data_processing_main.jl.
############################################################

"""
    import_j2j_ee_rates() -> DataFrame

Filter J2J to the production cell (national, seasonally adjusted, all
industries/demographics/firm sizes and ages, education E4; no fallback cell),
form the quarterly rate EEHire/MainB, convert to a monthly rate
1 − (1 − q)^(1/3), assign each quarter to the window of its middle month, and
average within window. Saves and returns j2j_ee_rates.csv.
"""
function import_j2j_ee_rates()
    @info "Stage 5: import_j2j_ee_rates (E4-only production spec)..."

    j2j_files = filter(f -> endswith(f, ".csv"), readdir(RAW_J2J_DIR))
    isempty(j2j_files) && error("No CSV files found in $RAW_J2J_DIR")
    j2j_path = joinpath(RAW_J2J_DIR, first(j2j_files))

    @info "  Reading: $j2j_path"
    j2j = CSV.read(j2j_path, DataFrame; types=Dict(
        :EEHire => Float64, :MainB => Float64,
        :seasonadj => String, :geo_level => String, :industry => String,
        :sex => String, :agegrp => String, :race => String,
        :ethnicity => String, :education => String, :firmage => String,
        :firmsize => String
    ))
    @info "  Total rows: $(nrow(j2j))"

    j2j_nat = filter(row ->
        row.seasonadj == "S" &&
        row.geo_level == "N" &&
        row.industry == "00" &&
        row.sex == "0" &&
        row.agegrp == "A00" &&
        row.race == "A0" &&
        row.ethnicity == "A0" &&
        row.firmage == "0" &&
        row.firmsize == "0" &&
        row.education == "E4",
        j2j
    )
    @info "  After filtering (national, SA, E4 only): $(nrow(j2j_nat))"

    dropmissing!(j2j_nat, [:EEHire, :MainB])
    j2j_nat.ee_quarterly = j2j_nat.EEHire ./ j2j_nat.MainB

    # Quarter → window by its middle month (Feb, May, Aug, Nov).
    quarter_to_months = Dict(1 => [1,2,3], 2 => [4,5,6], 3 => [7,8,9], 4 => [10,11,12])
    j2j_nat.window = Vector{Symbol}(undef, nrow(j2j_nat))
    for (i, row) in enumerate(eachrow(j2j_nat))
        mid_month = quarter_to_months[row.quarter][2]
        j2j_nat.window[i] = assign_window(row.year, mid_month)
    end

    j2j_nat.ee_monthly = 1.0 .- (1.0 .- j2j_nat.ee_quarterly) .^ (1/3)

    results = NamedTuple[]
    for wname in keys(WINDOWS)
        win_data = filter(r -> r.window == wname, j2j_nat)
        isempty(win_data) && continue
        vals = filter(isfinite, win_data.ee_monthly)
        n_dropped = nrow(win_data) - length(vals)
        n_dropped > 0 && @warn "    $wname: $n_dropped quarter(s) dropped (non-finite ee_monthly)"
        ee_S = isempty(vals) ? NaN : mean(vals)
        ee_rate_S_sd = length(vals) >= 2 ? std(vals; corrected=true) : NaN
        push!(results, (window=wname, ee_rate_S=ee_S, ee_rate_S_sd=ee_rate_S_sd, n_quarters=length(vals)))
    end
    ee_df = DataFrame(results)

    outpath = joinpath(DERIVED_DIR, "j2j_ee_rates.csv")
    CSV.write(outpath, ee_df)
    @info "  Saved: $outpath"
    @info "  J2J EE rates (E4-only):"
    for row in eachrow(ee_df)
        @printf("    %s: ee_S=%.4f  sd=%.4f  (n_quarters=%d)\n",
                row.window, row.ee_rate_S, row.ee_rate_S_sd, row.n_quarters)
    end
    return ee_df
end

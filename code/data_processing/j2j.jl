############################################################
# data_processing/j2j.jl
#
# Stage 5 — Job-to-Job (J2J) employer-to-employer hiring. Filters the raw
# Census J2J flows to the national, seasonally-adjusted, Bachelor's+ (E4)
# cell, converts the quarterly EE hazard to a monthly rate, and averages
# within each window to produce ee_rate_S.
#
# Reads:  data/raw/j2j/*.csv
# Writes: j2j_ee_rates.csv
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages and path consts come from data_processing_main.jl.
############################################################

function import_j2j_ee_rates()
    @info "Stage 5: import_j2j_ee_rates (E4-only production spec)..."

    # Find J2J CSV file
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

    # Filter to production spec from data_and_moments.pdf §5:
    #   seasonadj = S, geo_level = N, industry = 00,
    #   sex = 0, agegrp = A00, race = A0, ethnicity = A0,
    #   firmage = 0, firmsize = 0, education = E4.
    # No fallback path is computed; ee_rate_S derives exclusively
    # from the E4 (Bachelor\'s+) cell at the origin firm.
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

    # Quarterly → window via mid-quarter month (Q1→Feb, Q2→May, Q3→Aug, Q4→Nov)
    quarter_to_months = Dict(1 => [1,2,3], 2 => [4,5,6], 3 => [7,8,9], 4 => [10,11,12])
    j2j_nat.window = Vector{Symbol}(undef, nrow(j2j_nat))
    for (i, row) in enumerate(eachrow(j2j_nat))
        mid_month = quarter_to_months[row.quarter][2]
        j2j_nat.window[i] = assign_window(row.year, mid_month)
    end

    # Constant-hazard monthly rate: ee_m = 1 - (1 - ee_q)^(1/3)
    j2j_nat.ee_monthly = 1.0 .- (1.0 .- j2j_nat.ee_quarterly) .^ (1/3)

    # Window-mean of the monthly hazard. Drop quarters with missing/non-
    # finite values; per spec these omissions are logged as diagnostics.
    results = NamedTuple[]
    for wname in keys(WINDOWS)
        win_data = filter(r -> r.window == wname, j2j_nat)
        isempty(win_data) && continue
        vals = filter(isfinite, win_data.ee_monthly)
        n_dropped = nrow(win_data) - length(vals)
        n_dropped > 0 && @warn "    $wname: $n_dropped quarter(s) dropped (non-finite ee_monthly)"
        ee_S = isempty(vals) ? NaN : mean(vals)
        push!(results, (window=wname, ee_rate_S=ee_S, n_quarters=length(vals)))
    end
    ee_df = DataFrame(results)

    outpath = joinpath(DERIVED_DIR, "j2j_ee_rates.csv")
    CSV.write(outpath, ee_df)
    @info "  Saved: $outpath"
    @info "  J2J EE rates (E4-only):"
    for row in eachrow(ee_df)
        @printf("    %s: ee_S=%.4f  (n_quarters=%d)\n", row.window, row.ee_rate_S, row.n_quarters)
    end
    return ee_df
end

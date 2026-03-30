############################################################
# j2j.jl — Stage 5: Import J2J EE Rates
#
# Reads Census J2J quarterly employer-to-employer rates
# by education level. Converts quarterly to monthly,
# aggregates unskilled (E1+E2+E3) and skilled (E4=Bachelor's+).
# Saves to derived/.
#
# Requires: helpers.jl included first.
############################################################

function import_j2j_ee_rates()
    @info "Stage 5: import_j2j_ee_rates..."

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

    # Filter: seasonally adjusted, national, all industries, all demographics
    # except education.
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
        row.education in ("E0", "E1", "E2", "E3", "E4"),
        j2j
    )
    @info "  After filtering (national, SA, by education): $(nrow(j2j_nat))"

    # Compute quarterly EE rate = EEHire / MainB
    dropmissing!(j2j_nat, [:EEHire, :MainB])
    j2j_nat.ee_quarterly = j2j_nat.EEHire ./ j2j_nat.MainB

    # Assign windows (quarterly → use mid-quarter month)
    quarter_to_months = Dict(1 => [1,2,3], 2 => [4,5,6], 3 => [7,8,9], 4 => [10,11,12])
    j2j_nat.window = Vector{Symbol}(undef, nrow(j2j_nat))
    for (i, row) in enumerate(eachrow(j2j_nat))
        months = quarter_to_months[row.quarter]
        mid_month = months[2]
        j2j_nat.window[i] = assign_window(row.year, mid_month)
    end

    # Convert quarterly rate to monthly: ee_monthly ≈ 1 - (1 - ee_quarterly)^(1/3)
    j2j_nat.ee_monthly = 1.0 .- (1.0 .- j2j_nat.ee_quarterly) .^ (1/3)

    # For skilled: E4 = Bachelor's+
    # For unskilled: aggregate E1+E2+E3
    results = NamedTuple[]
    for wname in keys(WINDOWS)
        win_data = filter(r -> r.window == wname, j2j_nat)
        isempty(win_data) && continue

        # Skilled EE rate (E4)
        e4 = filter(r -> r.education == "E4", win_data)
        ee_S = isempty(e4) ? NaN : let vals = filter(isfinite, e4.ee_monthly)
            isempty(vals) ? NaN : mean(vals)
        end

        # Unskilled EE rate: aggregate E1+E2+E3
        e_unsk = filter(r -> r.education in ("E1", "E2", "E3"), win_data)
        if !isempty(e_unsk)
            unsk_q = combine(groupby(e_unsk, [:year, :quarter]),
                             :EEHire => sum => :EEHire, :MainB => sum => :MainB)
            unsk_q.ee_q = unsk_q.EEHire ./ unsk_q.MainB
            unsk_q.ee_m = 1.0 .- (1.0 .- unsk_q.ee_q) .^ (1/3)
            ee_U = let vals = filter(isfinite, unsk_q.ee_m)
                isempty(vals) ? NaN : mean(vals)
            end
        else
            ee_U = NaN
        end

        push!(results, (window=wname, ee_rate_S=ee_S, ee_rate_U=ee_U))
    end
    ee_df = DataFrame(results)

    outpath = joinpath(DERIVED_DIR, "j2j_ee_rates.csv")
    CSV.write(outpath, ee_df)
    @info "  Saved: $outpath"
    @info "  J2J EE rates:"
    for row in eachrow(ee_df)
        @printf("    %s: ee_S=%.4f  ee_U=%.4f\n", row.window, row.ee_rate_S, row.ee_rate_U)
    end
    return ee_df
end

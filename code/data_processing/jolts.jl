############################################################
# jolts.jl — Stage 3: Download and clean JOLTS
#
# Fetches JOLTS vacancy data from BLS API (or reads cached CSV).
# Allocates vacancies to skilled/unskilled using CPS industry
# skill shares from Stage 1.
# Saves cleaned Arrow file to derived/.
#
# Requires: helpers.jl included first.
############################################################

# JOLTS series IDs — supersector-level job openings
const JOLTS_SERIES = [
    "JTS100000000000000JOL", "JTS230000000000000JOL",
    "JTS320000000000000JOL", "JTS340000000000000JOL",
    "JTS420000000000000JOL", "JTS440000000000000JOL",
    "JTS480099000000000JOL", "JTS510000000000000JOL",
    "JTS510099000000000JOL", "JTS540099000000000JOL",
    "JTS610000000000000JOL", "JTS620000000000000JOL",
    "JTS710000000000000JOL", "JTS720000000000000JOL",
    "JTS810000000000000JOL", "JTS910000000000000JOL",
    "JTS920000000000000JOL",
]

# Cross-check aggregate series (not used for allocation — would double-count)
const JOLTS_CROSSCHECK_SERIES = [
    "JTS000000000000000JOL",  # Total nonfarm
    "JTS300000000000000JOL",  # Manufacturing total
    "JTS400000000000000JOL",  # Trade, transportation, utilities
    "JTS600000000000000JOL",  # Education and health
    "JTS700000000000000JOL",  # Leisure and hospitality
    "JTS900000000000000JOL",  # Government total
]

const JOLTS_INDUSTRY_MAP = Dict(
    "100000" => "Mining and logging",
    "230000" => "Construction",
    "320000" => "Durable goods manufacturing",
    "340000" => "Nondurable goods manufacturing",
    "420000" => "Wholesale trade",
    "440000" => "Retail trade",
    "480099" => "Transportation, warehousing and utilities",
    "510000" => "Information",
    "510099" => "Financial activities",
    "540099" => "Professional and business services",
    "610000" => "Educational services",
    "620000" => "Health care and social assistance",
    "710000" => "Arts, entertainment, and recreation",
    "720000" => "Accommodation and food services",
    "810000" => "Other services",
    "910000" => "Federal government",
    "920000" => "State and local government",
)

function _bls_fetch(series_ids, start_year, end_year)
    body = JSON3.write(Dict(
        "seriesid"        => series_ids,
        "startyear"       => string(start_year),
        "endyear"         => string(end_year),
        "registrationkey" => get(ENV, "BLS_API_KEY", ""),
    ))
    resp = HTTP.post(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        ["Content-Type" => "application/json"],
        body,
    )
    return JSON3.read(String(resp.body))
end

function _parse_jolts_industry(sid::AbstractString)::String
    m = match(r"^JTS(\d{6})\d*JOL$", sid)
    isnothing(m) && return "unknown"
    return m.captures[1]
end

function download_jolts()
    outpath = joinpath(RAW_JOLTS_DIR, "jolts_openings.csv")

    # Download guard: skip if file already exists
    if isfile(outpath)
        @info "  JOLTS file already exists: $outpath — skipping download"
        return CSV.read(outpath, DataFrame)
    end

    @info "download_jolts: fetching from BLS API..."
    all_series = vcat(JOLTS_SERIES, JOLTS_CROSSCHECK_SERIES)
    records = @NamedTuple{series_id::String, year::Int, period::String, value::Float64}[]

    for (start_yr, end_yr) in [(2000, 2019), (2020, 2022)]
        @info "  Requesting $(start_yr)–$(end_yr)..."
        result = _bls_fetch(all_series, start_yr, end_yr)
        result["status"] != "REQUEST_SUCCEEDED" &&
            error("BLS API error: $(result["message"])")
        for series in result["Results"]["series"]
            sid = String(series["seriesID"])
            for obs in series["data"]
                String(obs["period"]) == "M13" && continue
                push!(records, (
                    series_id = sid,
                    year      = parse(Int, String(obs["year"])),
                    period    = String(obs["period"]),
                    value     = parse(Float64, replace(String(obs["value"]), "," => "")),
                ))
            end
        end
        sleep(0.5)
    end

    df = DataFrame(records)

    # Missing series check
    missing_series = setdiff(JOLTS_SERIES, unique(df.series_id))
    if !isempty(missing_series)
        error("No data returned for JOLTS series: $missing_series — verify IDs at bls.gov/jlt")
    end

    mkpath(RAW_JOLTS_DIR)
    CSV.write(outpath, df)
    @info "  Saved: $outpath  ($(nrow(df)) rows)"
    return df
end

function clean_jolts()
    @info "Stage 3: clean_jolts..."

    # ── 1. Read raw data ──────────────────────────────────────────
    inpath = joinpath(RAW_JOLTS_DIR, "jolts_openings.csv")
    if !isfile(inpath)
        download_jolts()
    end
    df = CSV.read(inpath, DataFrame)
    rename!(df, [:series_id => :SERIES_ID, :year => :YEAR,
                 :period => :PERIOD, :value => :VALUE])
    @info "  Raw records: $(nrow(df))"

    # ── 2. Parse month ────────────────────────────────────────────
    df.MONTH = [parse(Int, replace(string(p), r"^M0?" => "")) for p in df.PERIOD]
    filter!(row -> 1 <= row.MONTH <= 12, df)

    # ── 3. Parse industry code ────────────────────────────────────
    df.industry_code = [_parse_jolts_industry(string(s)) for s in df.SERIES_ID]
    df.industry_name = [get(JOLTS_INDUSTRY_MAP, ic, "Cross-check ($ic)")
                        for ic in df.industry_code]

    @assert all(df.industry_code .!= "unknown") "Unresolved series IDs found"

    # ── 4. JOLTS units: multiply by 1000 (thousands → persons) ───
    df.VALUE_PERSONS = df.VALUE .* 1000.0
    @info "  JOLTS values converted: thousands → persons (×1000)"

    # ── 5. Window assignment ──────────────────────────────────────
    df.window = assign_window.(df.YEAR, df.MONTH)

    # ── 6. Merge with CPS industry skill shares ──────────────────
    CROSS_CHECK_CODES = Set(["000000","300000","400000","600000","700000","900000"])
    allocation = filter(row -> row.industry_code ∉ CROSS_CHECK_CODES &&
                               row.industry_code != "unknown", df)

    shares_path = joinpath(DERIVED_DIR, "industry_skill_shares.arrow")
    econ_shares_path = joinpath(DERIVED_DIR, "economy_skill_shares.arrow")

    if isfile(shares_path)
        shares = DataFrame(Arrow.Table(shares_path))
        @info "  Loaded industry skill shares ($(nrow(shares)) rows)"

        allocation = leftjoin(allocation, shares;
                              on = [:window => :window, :industry_code => :IND_JOLTS],
                              makeunique = true)

        # Default: economy-wide skill share (not 0.5)
        if isfile(econ_shares_path)
            econ = DataFrame(Arrow.Table(econ_shares_path))
            econ_map = Dict(row.window => row.econ_skill_share for row in eachrow(econ))
        else
            econ_map = Dict{Symbol, Float64}()
        end

        n_defaulted = 0
        for i in 1:nrow(allocation)
            if ismissing(allocation.skilled_share_ind[i])
                default_share = get(econ_map, allocation.window[i], 0.35)
                allocation.skilled_share_ind[i] = default_share
                n_defaulted += 1
            end
        end
        if n_defaulted > 0
            @warn "  $n_defaulted industry-window cells used economy-wide default skill share"
        end
    else
        @warn "  Industry skill shares not found — using 0.35 default"
        allocation.skilled_share_ind = fill(0.35, nrow(allocation))
    end

    # ── 7. Allocate vacancies to skilled/unskilled ────────────────
    allocation.V_S = allocation.VALUE_PERSONS .* allocation.skilled_share_ind
    allocation.V_U = allocation.VALUE_PERSONS .* (1.0 .- allocation.skilled_share_ind)

    # ── 8. Aggregate across industries per (year, month, window) ──
    agg = combine(
        groupby(allocation, [:YEAR, :MONTH, :window]),
        :V_S => sum => :V_S,
        :V_U => sum => :V_U,
    )

    outpath = joinpath(DERIVED_DIR, "jolts_clean.arrow")
    Arrow.write(outpath, agg)
    @info "  Saved: $outpath  ($(nrow(agg)) rows)"

    # ── 9. Cross-check: compare allocation total vs total nonfarm ─
    total_nf = filter(row -> row.industry_code == "000000", df)
    if nrow(total_nf) > 0
        alloc_total = combine(groupby(allocation, [:YEAR, :MONTH]),
                              :VALUE_PERSONS => sum => :alloc_total)
        check = innerjoin(total_nf, alloc_total; on = [:YEAR, :MONTH])
        check.pct_diff = (check.alloc_total .- check.VALUE_PERSONS) ./ check.VALUE_PERSONS .* 100
        max_diff = maximum(abs.(filter(isfinite, check.pct_diff)))
        println("  Cross-check: max |alloc_total − total_nonfarm| / total_nonfarm = $(round(max_diff; digits=2))%")
        if max_diff > 5.0
            @warn "  Allocation total deviates >5% from total nonfarm — investigate"
        end
    end

    return agg
end

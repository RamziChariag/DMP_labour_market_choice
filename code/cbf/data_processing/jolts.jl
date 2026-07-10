############################################################
# data_processing/jolts.jl
#
# Stage 3 — JOLTS job openings. download_jolts() pulls the supersector
# series from the BLS API (cached to data/raw/jolts/); clean_jolts()
# allocates openings to skilled/unskilled using the CPS industry skill
# shares and aggregates V_S / V_U per (year, month, window).
#
# Reads:  data/raw/jolts/jolts_openings.csv (or BLS API), industry_skill_shares.arrow, economy_skill_shares.arrow
# Writes: jolts_openings.csv, jolts_clean.arrow
#
# Set ENV["BLS_API_KEY"] for higher BLS rate limits (optional).
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages and path consts come from data_processing_main.jl.
############################################################

# JOLTS series IDs — supersector-level job openings
# NOTE: Mining and logging is BLS code 110099, NOT 100000.
# 100000 = "Total private" (display_level 1) and would double-count the
# entire private economy into the allocation (≈1.9x total nonfarm).
const JOLTS_SERIES = [
    "JTS110099000000000JOL", "JTS230000000000000JOL",
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
    "920000" => "Government (federal + state & local)",
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
    code = m.captures[1]
    # BLS codes Mining and logging as 110099; the rest of the pipeline
    # (JOLTS_INDUSTRY_MAP and the CPS IND_JOLTS crosswalk) uses the internal
    # key "100000" for this supersector. Normalise so the join matches.
    code == "110099" && return "100000"
    return code
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

    for (start_yr, end_yr) in [(2000, 2019), (2020, 2025)]
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

    # Missing series check — ERROR not warning
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

    # Verify: all JOL series should resolve
    @assert all(df.industry_code .!= "unknown") "Unresolved series IDs found"

    # ── 4. JOLTS units: multiply by 1000 (thousands → persons) ───
    df.VALUE_PERSONS = df.VALUE .* 1000.0
    @info "  JOLTS values converted: thousands → persons (×1000)"

    # ── 5. Window assignment ──────────────────────────────────────
    df.window = assign_window.(df.YEAR, df.MONTH)

    # Merge Federal (910000) into State-and-local (920000) → one Government
    # cell. The CPS Census industry codes (setup.jl) route all public
    # administration to "920000" and cannot separate federal from state/local,
    # so JOLTS Federal openings have no distinct CPS skill share. Collapsing
    # both JOLTS government series onto "920000" lets them share the single
    # CPS government skill share; step 8's groupby then sums their V_S/V_U.
    df.industry_code = [c == "910000" ? "920000" : c for c in df.industry_code]
    df.industry_name = [get(JOLTS_INDUSTRY_MAP, ic, "Cross-check ($ic)")
                        for ic in df.industry_code]

    # ── 6. Merge with CPS industry skill shares ──────────────────
    # Restrict to in-window months. CPS industry skill shares are only
    # produced for window != :none (Stage 1), so out-of-window JOLTS months
    # (2001-02, 2011-14, the Dec-2000 lead-in, etc.) have no share to join
    # to and would otherwise default spuriously. They enter no moment.
    CROSS_CHECK_CODES = Set(["000000","300000","400000","600000","700000","900000"])
    allocation = filter(row -> row.window != :none &&
                               row.industry_code ∉ CROSS_CHECK_CODES &&
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
        defaulted_codes = Dict{String,Int}()
        for i in 1:nrow(allocation)
            if ismissing(allocation.skilled_share_ind[i])
                default_share = get(econ_map, allocation.window[i], 0.35)
                allocation.skilled_share_ind[i] = default_share
                n_defaulted += 1
                code = allocation.industry_code[i]
                defaulted_codes[code] = get(defaulted_codes, code, 0) + 1
            end
        end
        if n_defaulted > 0
            @warn "  $n_defaulted industry-window cells used economy-wide default skill share"
            # Breakdown by supersector: a code defaulting in (near) all 192
            # in-window months means CPS never emits that IND_JOLTS string
            # (structural gap in the crosswalk), not sampling sparsity.
            for (code, n) in sort(collect(defaulted_codes); by = last, rev = true)
                name = get(JOLTS_INDUSTRY_MAP, code, code)
                @warn "    defaulted: $code ($name) — $n cells"
            end
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
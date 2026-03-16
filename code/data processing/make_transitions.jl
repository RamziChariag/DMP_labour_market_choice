############################################################
# make_transitions.jl — Transition rates from matched CPS
#
# Input:  data/derived/cps_basic_clean.arrow
# Output: data/derived/transitions.arrow
#
# What this script does
# ─────────────────────
# 1. Load the cleaned CPS Basic Monthly panel.
# 2. Build matched month-pairs: link person i in month t
#    to the same person in month t+1 using CPSIDP, with
#    MISH ∈ {1,2,3,5,6,7}.
# 3. For each matched pair, classify the transition:
#      UE  (unemployed → employed)      = job finding
#      EU  (employed → unemployed)      = separation
#      EE  (employed → employed, different employer) = EE
# 4. Compute monthly transition rates by skill group.
# 5. Average within each estimation window.
# 6. Save as Arrow.
#
# Transition rate definitions (matching moments.jl)
# ─────────────────────────────────────────────────
#   jfr_j     = Σ_{i: U_t→E_{t+1}, j} w_i / Σ_{i: U_t, j} w_i
#   sep_j     = Σ_{i: E_t→U_{t+1}, j} w_i / Σ_{i: E_t, j} w_i
#   ee_rate_S = Pr(different employer_{t+1} | E_t, S)
############################################################


"""
    make_transitions() → DataFrame

Build CPS matched month-pairs and compute transition rates.
Also saves data/derived/transitions.arrow.
"""
function make_transitions()

    @info "make_transitions: loading cleaned CPS Basic..."

    # ── 1. Load cleaned CPS Basic ────────────────────────────────────────
    inpath = joinpath(DERIVED_DIR, "cps_basic_clean.arrow")
    if !isfile(inpath)
        error("cps_basic_clean.arrow not found — run clean_cps_basic() first")
    end
    df = DataFrame(Arrow.Table(inpath))

    # ── 2. Build matched month-pairs ─────────────────────────────────────
    #    Match person i in (year, month) to the same CPSIDP in
    #    (year, month+1), restricted to valid MISH values.
    @info "  Building matched month-pairs..."

    # Only keep observations that can be matched forward
    matchable = filter(row -> row.valid_match && row.CPSIDP > 0, df)

    # Create a lookup key: (CPSIDP, year, month) → row index
    # Then for each matchable row, look for (CPSIDP, next_year, next_month)
    next_ym(y, m) = m == 12 ? (y + 1, 1) : (y, m + 1)

    # Build dictionary of (CPSIDP, year, month) → row data
    lookup = Dict{Tuple{Int64, Int, Int}, NamedTuple}()
    for row in eachrow(df)
        key = (Int64(row.CPSIDP), row.YEAR, row.MONTH)
        lookup[key] = (skilled    = row.skilled,
                       employed   = row.employed,
                       unemployed = row.unemployed,
                       weight     = row.WTFINL,
                       window     = row.window)
    end

    # Build pairs
    pairs_data = NamedTuple[]
    for row in eachrow(matchable)
        ny, nm = next_ym(row.YEAR, row.MONTH)
        next_key = (Int64(row.CPSIDP), ny, nm)
        haskey(lookup, next_key) || continue
        next = lookup[next_key]

        push!(pairs_data, (
            year_t      = row.YEAR,
            month_t     = row.MONTH,
            skilled     = row.skilled,
            emp_t       = row.employed,
            unemp_t     = row.unemployed,
            emp_t1      = next.employed,
            unemp_t1    = next.unemployed,
            weight      = Float64(row.WTFINL),
            window      = row.window,
        ))
    end
    pairs = DataFrame(pairs_data)
    @info "  Matched pairs: $(nrow(pairs))"

    # ── 3. Compute monthly transition rates by (year, month, skill) ──────
    #    For each month-pair and skill group, compute:
    #      jfr   = Σ w(U_t ∩ E_{t+1}) / Σ w(U_t)
    #      sep   = Σ w(E_t ∩ U_{t+1}) / Σ w(E_t)
    results = NamedTuple[]
    for gk in groupby(pairs, [:year_t, :month_t, :skilled, :window])
        g = DataFrame(gk)
        sk  = g.skilled[1]
        win = g.window[1]

        # Job finding: U_t → E_{t+1}
        u_mask  = g.unemp_t
        ue_mask = g.unemp_t .& g.emp_t1
        denom_jfr = sum(g.weight[u_mask])
        numer_jfr = sum(g.weight[ue_mask])
        jfr = denom_jfr > 0 ? numer_jfr / denom_jfr : NaN

        # Separation: E_t → U_{t+1}
        e_mask  = g.emp_t
        eu_mask = g.emp_t .& g.unemp_t1
        denom_sep = sum(g.weight[e_mask])
        numer_sep = sum(g.weight[eu_mask])
        sep = denom_sep > 0 ? numer_sep / denom_sep : NaN

        push!(results, (
            year    = g.year_t[1],
            month   = g.month_t[1],
            skilled = sk,
            window  = win,
            jfr     = jfr,
            sep     = sep,
            n_pairs = nrow(g),
        ))
    end
    rates = DataFrame(results)

    # ── 4. Average within each estimation window ─────────────────────────
    window_rates = NamedTuple[]
    for gk in groupby(filter(r -> r.window != :none, rates), [:window, :skilled])
        g = DataFrame(gk)
        win = g.window[1]
        sk  = g.skilled[1]

        valid_jfr = filter(isfinite, g.jfr)
        valid_sep = filter(isfinite, g.sep)

        push!(window_rates, (
            window      = win,
            skilled     = sk,
            mean_jfr    = isempty(valid_jfr) ? NaN : mean(valid_jfr),
            mean_sep    = isempty(valid_sep) ? NaN : mean(valid_sep),
            n_months    = nrow(g),
        ))
    end
    agg = DataFrame(window_rates)

    # ── 5. Save ──────────────────────────────────────────────────────────
    # Save both detailed monthly rates and window averages
    Arrow.write(joinpath(DERIVED_DIR, "transitions_monthly.arrow"), rates)
    outpath = joinpath(DERIVED_DIR, "transitions.arrow")
    Arrow.write(outpath, agg)
    @info "  Saved: $outpath  ($(nrow(agg)) rows)"
    @info "  Window-averaged transition rates:"
    for row in eachrow(agg)
        sk_label = row.skilled ? "S" : "U"
        @info "    $(row.window)  $sk_label:  jfr=$(round(row.mean_jfr; digits=4))  sep=$(round(row.mean_sep; digits=4))  ($(row.n_months) months)"
    end

    return agg
end

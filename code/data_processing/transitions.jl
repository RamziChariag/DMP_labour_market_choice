############################################################
# transitions.jl — Stage 4: Build Transition Rates
#
# Matches CPS Basic persons across adjacent months using CPSIDP.
# Computes job-finding, separation, and LF exit rates by skill.
# Saves monthly and window-averaged transition rates to derived/.
#
# Requires: helpers.jl, cps_basic.jl included first.
############################################################

function make_transitions()
    @info "Stage 4: make_transitions — loading cleaned CPS Basic..."

    inpath = joinpath(DERIVED_DIR, "cps_basic_clean.arrow")
    !isfile(inpath) && error("cps_basic_clean.arrow not found — run Stage 1 first")
    df = DataFrame(Arrow.Table(inpath))

    # Check for longitudinal weights
    use_lnk_wt = hasproperty(df, :LNKFW1MWT)
    wt_col = use_lnk_wt ? :LNKFW1MWT : :WTFINL
    @info "  Using weight column: $wt_col"

    @info "  Building matched month-pairs..."
    matchable = filter(row -> row.valid_match && row.CPSIDP > 0, df)
    next_ym(y, m) = m == 12 ? (y + 1, 1) : (y, m + 1)

    # Build lookup of all observations
    lookup = Dict{Tuple{Int64, Int, Int}, NamedTuple}()
    for row in eachrow(df)
        key = (Int64(row.CPSIDP), row.YEAR, row.MONTH)
        empstat = hasproperty(df, :EMPSTAT_CORRECTED) ? row.EMPSTAT_CORRECTED : row.EMPSTAT
        lookup[key] = (skilled=row.skilled,
                       employed=is_employed(empstat),
                       unemployed=is_unemployed(empstat),
                       nilf=is_nilf(empstat),
                       weight=Float64(coalesce(getproperty(row, wt_col), 0.0)),
                       window=row.window)
    end

    # Match pairs
    pairs_data = NamedTuple[]
    for row in eachrow(matchable)
        ny, nm = next_ym(row.YEAR, row.MONTH)
        next_key = (Int64(row.CPSIDP), ny, nm)
        haskey(lookup, next_key) || continue
        next = lookup[next_key]

        empstat_t = hasproperty(df, :EMPSTAT_CORRECTED) ? row.EMPSTAT_CORRECTED : row.EMPSTAT
        push!(pairs_data, (
            year_t=row.YEAR, month_t=row.MONTH, skilled=row.skilled,
            emp_t=is_employed(empstat_t), unemp_t=is_unemployed(empstat_t),
            lf_t=is_employed(empstat_t) || is_unemployed(empstat_t),
            emp_t1=next.employed, unemp_t1=next.unemployed, nilf_t1=next.nilf,
            weight=Float64(coalesce(getproperty(row, wt_col), 0.0)),
            window=row.window,
        ))
    end
    pairs = DataFrame(pairs_data)
    @info "  Matched pairs: $(nrow(pairs))"

    # Monthly transition rates
    results = NamedTuple[]
    for gk in groupby(pairs, [:year_t, :month_t, :skilled, :window])
        g = DataFrame(gk)
        sk = g.skilled[1]; win = g.window[1]

        # Job-finding rate: P(emp_t+1 | unemp_t)
        u_mask  = g.unemp_t
        ue_mask = g.unemp_t .& g.emp_t1
        denom_jfr = sum(g.weight[u_mask])
        numer_jfr = sum(g.weight[ue_mask])
        jfr = denom_jfr > 0 ? numer_jfr / denom_jfr : NaN

        # EU separation rate: P(unemp_t+1 | emp_t)
        e_mask  = g.emp_t
        eu_mask = g.emp_t .& g.unemp_t1
        denom_sep = sum(g.weight[e_mask])
        numer_sep = sum(g.weight[eu_mask])
        sep = denom_sep > 0 ? numer_sep / denom_sep : NaN

        # LF exit rate: P(NILF_t+1 | LF_t) — for computing ν
        lf_mask   = g.lf_t
        nilf_mask = g.lf_t .& g.nilf_t1
        denom_nu  = sum(g.weight[lf_mask])
        numer_nu  = sum(g.weight[nilf_mask])
        nu = denom_nu > 0 ? numer_nu / denom_nu : NaN

        push!(results, (year=g.year_t[1], month=g.month_t[1], skilled=sk,
                         window=win, jfr=jfr, sep=sep, nu=nu, n_pairs=nrow(g)))
    end
    rates = DataFrame(results)

    # Window averages
    window_rates = NamedTuple[]
    for gk in groupby(filter(r -> r.window != :none, rates), [:window, :skilled])
        g = DataFrame(gk)
        valid_jfr = filter(isfinite, g.jfr)
        valid_sep = filter(isfinite, g.sep)
        valid_nu  = filter(isfinite, g.nu)
        push!(window_rates, (
            window=g.window[1], skilled=g.skilled[1],
            mean_jfr=isempty(valid_jfr) ? NaN : mean(valid_jfr),
            mean_sep=isempty(valid_sep) ? NaN : mean(valid_sep),
            mean_nu=isempty(valid_nu)   ? NaN : mean(valid_nu),
            n_months=nrow(g),
        ))
    end
    agg = DataFrame(window_rates)

    Arrow.write(joinpath(DERIVED_DIR, "transitions_monthly.arrow"), rates)
    outpath = joinpath(DERIVED_DIR, "transitions.arrow")
    Arrow.write(outpath, agg)
    @info "  Saved: $outpath  ($(nrow(agg)) rows)"
    return agg
end

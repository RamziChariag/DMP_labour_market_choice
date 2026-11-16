############################################################
# code/data_and_descriptives/transitions.jl
#
# Worker flows from matched CPS Basic month pairs, and the demographic turnover
# rate ν.
#
# Entry points:
#   make_transitions()  → transitions_monthly.arrow (per month × market: hazards,
#                         weighted at-risk pools and flows, Kish N), transitions.arrow
#   compute_nu()        → nu_estimation.csv (life-table ν on the baseline windows,
#                         with the net-flow ν as a cross-check)
# Reads cps_basic_clean.arrow.
# Definitions only; packages and path constants come from data_processing_main.jl.
############################################################

"""
    make_transitions() -> DataFrame

Match each CPS person-month (`CPSIDP > 0`, `valid_match`) to the same person in
the next calendar month and compute, per (month, market, window), the U→E rate
`jfr`, the E→U rate `sep`, the LF→NILF rate `nu`, the weighted at-risk pools and
flows (`w_u`, `w_ue`, `w_e`, `w_eu`) and the Kish N of each at-risk pool
(`neff_jfr`, `neff_sep`). The weight is LNKFW1MWT if present, else WTFINL; the
window is that of month t. Saves the monthly table and the window averages
(ratios of totals); returns the window averages.
"""
function make_transitions()
    @info "Stage 4: make_transitions — loading cleaned CPS Basic..."

    inpath = joinpath(DERIVED_DIR, "cps_basic_clean.arrow")
    !isfile(inpath) && error("cps_basic_clean.arrow not found — run Stage 1 first")
    df = DataFrame(Arrow.Table(inpath))

    use_lnk_wt = hasproperty(df, :LNKFW1MWT)
    wt_col = use_lnk_wt ? :LNKFW1MWT : :WTFINL
    @info "  Using weight column: $wt_col"

    @info "  Building matched month-pairs..."
    matchable = filter(row -> row.valid_match && row.CPSIDP > 0, df)
    next_ym(y, m) = m == 12 ? (y + 1, 1) : (y, m + 1)

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

    results = NamedTuple[]
    for gk in groupby(pairs, [:year_t, :month_t, :skilled, :window])
        g = DataFrame(gk)
        sk = g.skilled[1]; win = g.window[1]

        # Job finding: P(E at t+1 | U at t)
        u_mask  = g.unemp_t
        ue_mask = g.unemp_t .& g.emp_t1
        denom_jfr = sum(g.weight[u_mask])
        numer_jfr = sum(g.weight[ue_mask])
        jfr = denom_jfr > 0 ? numer_jfr / denom_jfr : NaN

        # Separation: P(U at t+1 | E at t)
        e_mask  = g.emp_t
        eu_mask = g.emp_t .& g.unemp_t1
        denom_sep = sum(g.weight[e_mask])
        numer_sep = sum(g.weight[eu_mask])
        sep = denom_sep > 0 ? numer_sep / denom_sep : NaN

        # LF exit: P(NILF at t+1 | LF at t), for the net-flow ν cross-check
        lf_mask   = g.lf_t
        nilf_mask = g.lf_t .& g.nilf_t1
        denom_nu  = sum(g.weight[lf_mask])
        numer_nu  = sum(g.weight[nilf_mask])
        nu = denom_nu > 0 ? numer_nu / denom_nu : NaN

        # Kish N of the at-risk pools: the one source of the flow-hazard
        # sampling variances (summed over months in sampling_variances.jl).
        wu = g.weight[u_mask];  we = g.weight[e_mask]
        neff_jfr = sum(wu) > 0 ? sum(wu)^2 / sum(abs2, wu) : 0.0
        neff_sep = sum(we) > 0 ? sum(we)^2 / sum(abs2, we) : 0.0

        push!(results, (year=g.year_t[1], month=g.month_t[1], skilled=sk,
                         window=win,
                         jfr=jfr, sep=sep, nu=nu,
                         n_pairs=nrow(g), neff_jfr=neff_jfr, neff_sep=neff_sep,
                         w_u=denom_jfr, w_ue=numer_jfr, w_e=denom_sep, w_eu=numer_sep))
    end
    rates = DataFrame(results)

    # Window averages: total flow over total at-risk pool across the window's months.
    window_rates = NamedTuple[]
    for gk in groupby(filter(r -> r.window != :none, rates), [:window, :skilled])
        g = DataFrame(gk)
        valid_nu     = filter(isfinite, g.nu)
        push!(window_rates, (
            window=g.window[1], skilled=g.skilled[1],
            mean_jfr=sum(g.w_u) > 0 ? sum(g.w_ue) / sum(g.w_u) : NaN,
            mean_sep=sum(g.w_e) > 0 ? sum(g.w_eu) / sum(g.w_e) : NaN,
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

"""
    compute_nu() -> DataFrame

The demographic turnover rate ν on the baseline windows (each crisis window uses
its baseline's ν). ν is the life-table rate: the inverse of the weighted mean
remaining working life max(65 − age, 0)·12 months over LF participants. Also
reports, as an independent cross-check and not a bound, the net-flow rate
ν_net = δ − γ(1 − LFPR)/LFPR (δ the LF→NILF rate, γ the NILF→LF rate) and warns
only if the two differ by more than 50 %. Saves and returns nu_estimation.csv;
prints the life-table ν for all four windows as a diagnostic.
"""
function compute_nu()
    @info "Stage 6: compute_nu — life-table on base_fc AND base_covid..."

    cps   = DataFrame(Arrow.Table(joinpath(DERIVED_DIR, "cps_basic_clean.arrow")))
    trans = DataFrame(Arrow.Table(joinpath(DERIVED_DIR, "transitions_monthly.arrow")))

    # Life-table ν on one window.
    function life_table_nu(window::Symbol)
        sub = filter(r -> r.window == window && r.in_lf, cps)
        w   = Float64.(coalesce.(sub.WTFINL, 0.0))
        rem = Float64.(max.(65 .- sub.AGE, 0)) .* 12.0
        mean_rem = wmean(rem, w)
        return (nu = 1.0 / mean_rem,
                mean_rem_months = mean_rem,
                n_obs = nrow(sub))
    end

    # Net-flow ν on one window: δ from the monthly LF→NILF rates (weighted by
    # pair counts), γ from matched NILF→LF pairs, LFPR from the window stock.
    # Elevated retirement with a low return rate (2015–19) pushes ν_net above
    # the life-table rate, so a gap is demography, not a defect.
    function net_flow_estimate(window::Symbol)
        bt = filter(r -> r.window == window && isfinite(r.nu), trans)
        delta = nrow(bt) > 0 ? wmean(bt.nu, Float64.(bt.n_pairs)) : NaN

        nextym(y, m) = m == 12 ? (y + 1, 1) : (y, m + 1)
        sub = filter(r -> r.window == window, cps)
        lookup = Dict{Tuple{Int64,Int,Int}, Bool}()
        sizehint!(lookup, nrow(cps))
        for row in eachrow(cps)
            lookup[(Int64(row.CPSIDP), Int(row.YEAR), Int(row.MONTH))] = row.in_lf
        end
        nilf_w = filter(r -> r.window == window && !r.in_lf &&
                              r.valid_match && r.CPSIDP > 0, cps)
        numer_g = 0.0; denom_g = 0.0
        for row in eachrow(nilf_w)
            ny, nm = nextym(Int(row.YEAR), Int(row.MONTH))
            nk = (Int64(row.CPSIDP), ny, nm)
            haskey(lookup, nk) || continue
            wi = Float64(coalesce(row.WTFINL, 0.0))
            denom_g += wi
            lookup[nk] && (numer_g += wi)
        end
        gamma = denom_g > 0 ? numer_g / denom_g : NaN

        w_all = Float64.(coalesce.(sub.WTFINL, 0.0))
        lfpr  = sum(w_all[sub.in_lf]) / sum(w_all)

        nu_nf = delta - gamma * (1.0 - lfpr) / lfpr
        return (delta=delta, gamma=gamma, lfpr=lfpr, nu_net=nu_nf)
    end

    rows = NamedTuple[]
    for w in (:base_fc, :base_covid)
        lt = life_table_nu(w)
        nf = net_flow_estimate(w)
        @printf("  %s: life-table ν = %.6f  (mean_rem = %.1f months, n = %d)\n",
                w, lt.nu, lt.mean_rem_months, lt.n_obs)
        @printf("            δ = %.6f  γ = %.6f  LFPR = %.4f  ν_net = %.6f\n",
                nf.delta, nf.gamma, nf.lfpr, nf.nu_net)
        rel_gap = (lt.nu - nf.nu_net) / nf.nu_net
        if abs(rel_gap) <= 0.50
            @info @sprintf("    ✓ life-table ν and net-flow ν agree within 50%% (gap = %+.1f%%)", 100 * rel_gap)
        else
            @warn @sprintf("    ⚠ life-table ν and net-flow ν differ by %.1f%% — check window stationarity", 100 * rel_gap)
        end
        push!(rows, (
            window                = String(w),
            nu                    = lt.nu,
            mean_rem_months       = lt.mean_rem_months,
            n_obs                 = lt.n_obs,
            nu_net                = nf.nu_net,
            delta_lf_nilf         = nf.delta,
            gamma_nilf_lf         = nf.gamma,
            lfpr                  = nf.lfpr,
            method                = "life_table",
        ))
    end
    out = DataFrame(rows)

    outpath = joinpath(DERIVED_DIR, "nu_estimation.csv")
    CSV.write(outpath, out)
    @info "Saved $outpath"

    println("\nPer-window life-table ν (diagnostic — only base_fc / base_covid feed the SMM):")
    for wname in WINDOWS_ORDER
        wd = filter(r -> r.window == wname && r.in_lf, cps)
        isempty(wd) && continue
        ww = Float64.(coalesce.(wd.WTFINL, 0.0))
        rr = Float64.(max.(65 .- wd.AGE, 0)) .* 12.0
        @printf("  %-14s  ν = %.6f  (n = %d)\n", wname, 1.0 / wmean(rr, ww), nrow(wd))
    end

    return out
end

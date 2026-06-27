############################################################
# data_processing/transitions.jl
#
# Stage 4 & 6 — labour-market worker flows from the matched CPS Basic
# month-pairs. make_transitions() builds monthly job-finding, separation,
# and LF-exit hazards, and the NET skilled U→NILF hazard ρ_NILF used as the
# competing-risk term in ltu_share_S; compute_nu() turns the demographic
# side into the life-table turnover rate ν for each baseline window.
#
# Reads:  cps_basic_clean.arrow, transitions_monthly.arrow
# Writes: transitions.arrow, transitions_monthly.arrow, rho_nilf.csv,
#         nu_estimation.csv
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages and path consts come from data_processing_main.jl.
############################################################

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

    # Build lookup of all observations.
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

        # LF exit rate: P(NILF_t+1 | LF_t) — for the life-table ν cross-check
        lf_mask   = g.lf_t
        nilf_mask = g.lf_t .& g.nilf_t1
        denom_nu  = sum(g.weight[lf_mask])
        numer_nu  = sum(g.weight[nilf_mask])
        nu = denom_nu > 0 ? numer_nu / denom_nu : NaN

        # Gross U→NILF hazard: P(NILF_t+1 | U_t) — a per-month diagnostic. The
        # value the model consumes is the NET hazard built below, because the
        # gross rate double-counts reversible U↔NILF churn (see the ρ_NILF block).
        u_mask_nilf   = g.unemp_t
        un_mask       = g.unemp_t .& g.nilf_t1
        denom_u_nilf  = sum(g.weight[u_mask_nilf])
        numer_u_nilf  = sum(g.weight[un_mask])
        u_nilf = denom_u_nilf > 0 ? numer_u_nilf / denom_u_nilf : NaN

        push!(results, (year=g.year_t[1], month=g.month_t[1], skilled=sk,
                         window=win,
                         jfr=jfr, sep=sep, nu=nu, u_nilf=u_nilf,
                         n_pairs=nrow(g)))
    end
    rates = DataFrame(results)

    # Window averages
    window_rates = NamedTuple[]
    for gk in groupby(filter(r -> r.window != :none, rates), [:window, :skilled])
        g = DataFrame(gk)
        valid_jfr    = filter(isfinite, g.jfr)
        valid_sep    = filter(isfinite, g.sep)
        valid_nu     = filter(isfinite, g.nu)
        valid_u_nilf = filter(isfinite, g.u_nilf)
        push!(window_rates, (
            window=g.window[1], skilled=g.skilled[1],
            mean_jfr=isempty(valid_jfr) ? NaN : mean(valid_jfr),
            mean_sep=isempty(valid_sep) ? NaN : mean(valid_sep),
            mean_nu=isempty(valid_nu)   ? NaN : mean(valid_nu),
            mean_u_nilf=isempty(valid_u_nilf) ? NaN : mean(valid_u_nilf),
            n_months=nrow(g),
        ))
    end
    agg = DataFrame(window_rates)

    Arrow.write(joinpath(DERIVED_DIR, "transitions_monthly.arrow"), rates)
    outpath = joinpath(DERIVED_DIR, "transitions.arrow")
    Arrow.write(outpath, agg)
    @info "  Saved: $outpath  ($(nrow(agg)) rows)"

    # ── ρ_NILF: NET skilled U→NILF hazard (same netting as the net-flow ν) ──
    # The model's ρ_NILF is a PERMANENT exit from skilled unemployment (the
    # model has no NILF state, so a worker who "leaks" to NILF does not return).
    # The GROSS hazard P(NILF|U) is dominated by reversible U↔NILF churn — the
    # same classification churn that inflates the gross LF→NILF rate (~0.038)
    # relative to the demographic ν (~0.0033). We therefore net it the way ν is
    # netted: subtract the NILF→U return flow, scaled by the NILF/U stock ratio,
    # and floor at ν (a permanent exit cannot be below the demographic rate):
    #
    #     ρ_NILF = max( ρ_UN − ρ_NU · (NILF_S / U_S) ,  ν )
    #
    # ρ_UN = P(NILF|U, skilled), ρ_NU = P(U|NILF, skilled), both pooled over the
    # window; NILF_S/U_S and ν from the skilled cross-section. The gross U→NILF
    # rate is retained in transitions.arrow (mean_u_nilf) as a diagnostic only.
    skl_pairs = filter(r -> Bool(r.skilled), pairs)
    rho_rows  = NamedTuple[]
    for wname in WINDOWS_ORDER
        pw = filter(r -> r.window == wname, skl_pairs)

        # Gross U→NILF (pooled): P(NILF_t+1 | U_t)
        um     = pw.unemp_t
        w_um   = sum(pw.weight[um])
        rho_UN = w_um > 0 ? sum(pw.weight[um .& pw.nilf_t1]) / w_um : NaN

        # Gross NILF→U return (pooled): P(U_t+1 | NILF_t), NILF_t = ¬lf_t
        nm     = .!pw.lf_t
        w_nm   = sum(pw.weight[nm])
        rho_NU = w_nm > 0 ? sum(pw.weight[nm .& pw.unemp_t1]) / w_nm : NaN

        # Skilled NILF/U stock ratio and life-table ν (demographic floor)
        cw   = filter(r -> r.window == wname && r.skilled, df)
        wv   = Float64.(coalesce.(getproperty(cw, wt_col), 0.0))
        nN   = sum(wv[.!cw.in_lf]); nU = sum(wv[cw.unemployed])
        ratio = nU > 0 ? nN / nU : NaN

        lfw  = filter(r -> r.window == wname && r.in_lf, df)
        lw   = Float64.(coalesce.(getproperty(lfw, wt_col), 0.0))
        rem  = Float64.(max.(65 .- lfw.AGE, 0)) .* 12.0
        nu_w = sum(lw) > 0 ? 1.0 / wmean(rem, lw) : NaN

        rho_net = rho_UN - rho_NU * ratio
        rho_use = isfinite(rho_net) ? max(rho_net, nu_w) : nu_w

        push!(rho_rows, (window=String(wname), rho_NILF=rho_use,
                         rho_UN_gross=rho_UN, rho_NU_gross=rho_NU,
                         nilf_u_ratio=ratio, nu_floor=nu_w, rho_net_raw=rho_net))
        @printf("  ρ_NILF[%-13s]  net=%.5f  (gross U→N=%.5f, N→U=%.5f, NILF/U=%.1f, ν floor=%.5f, raw net=%+.5f)\n",
                wname, rho_use, rho_UN, rho_NU, ratio, nu_w, rho_net)
    end
    rho_path = joinpath(DERIVED_DIR, "rho_nilf.csv")
    CSV.write(rho_path, DataFrame(rho_rows))
    @info "  Saved: $rho_path  (NET skilled U→NILF hazard ρ_NILF per window)"

    return agg
end



function compute_nu()
    @info "Stage 6: compute_nu — life-table on base_fc AND base_covid..."

    cps   = DataFrame(Arrow.Table(joinpath(DERIVED_DIR, "cps_basic_clean.arrow")))
    trans = DataFrame(Arrow.Table(joinpath(DERIVED_DIR, "transitions_monthly.arrow")))

    # ── Life-table ν on a single window ───────────────────────────
    # For each LF participant of age a, implied remaining working life
    # is max(65 - a, 0) * 12 months. ν̂ = 1 / weighted-mean remaining life.
    function life_table_nu(window::Symbol)
        sub = filter(r -> r.window == window && r.in_lf, cps)
        w   = Float64.(coalesce.(sub.WTFINL, 0.0))
        rem = Float64.(max.(65 .- sub.AGE, 0)) .* 12.0
        mean_rem = wmean(rem, w)
        return (nu = 1.0 / mean_rem,
                mean_rem_months = mean_rem,
                n_obs = nrow(sub))
    end

    # ── Net-flow ν estimate (independent cross-check, NOT a bound) ─
    # ν_nf = δ − γ·(1−LFPR)/LFPR — the realised net LF→NILF outflow per LF
    # participant. This is a different notion of turnover than the life-table
    # rate (realised net outflow vs a smoothed "everyone exits at 65"), so the
    # two need not coincide. Reversible LF↔NILF churn inflates both δ and γ and
    # largely cancels in the net, so ν_nf is not a one-sided bound; what moves
    # it is the age structure — elevated retirement (e.g. the 2015–19 boomer
    # wave, with a low NILF→LF return γ) pushes the net outflow above the
    # life-table rate.
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

    # ── Run on both baseline windows ──────────────────────────────
    rows = NamedTuple[]
    for w in (:base_fc, :base_covid)
        lt = life_table_nu(w)
        nf = net_flow_estimate(w)
        @printf("  %s: life-table ν = %.6f  (mean_rem = %.1f months, n = %d)\n",
                w, lt.nu, lt.mean_rem_months, lt.n_obs)
        @printf("            δ = %.6f  γ = %.6f  LFPR = %.4f  ν_net = %.6f\n",
                nf.delta, nf.gamma, nf.lfpr, nf.nu_net)
        # Two independent ν estimates: the life-table rate and the net-flow
        # rate ν_net. They coincide when the age structure is stable and
        # diverge when retirement is elevated (the net outflow then runs above
        # the life-table rate). Both are ~0.003–0.005; flag only an
        # order-of-magnitude disagreement, not the routine demographic gap.
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

    # Diagnostic: per-window life-table values for all four windows
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
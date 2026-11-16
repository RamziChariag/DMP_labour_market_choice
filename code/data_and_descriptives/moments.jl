############################################################
# code/data_and_descriptives/moments.jl
#
# Assembles the empirical moment vector (MOMENT_NAMES order) for each window
# from the cleaned datasets.
#
# Entry point: make_moments()
#   reads  cps_basic_clean.arrow, cps_asec_clean.arrow, transitions_monthly.arrow,
#          jolts_clean.arrow, j2j_ee_rates.csv, sipp_wchg_rates.csv,
#          sipp_ee_rates.csv, training_share_target.csv
#   writes moments_{window}.csv
# Definitions only; packages and path constants come from data_processing_main.jl.
############################################################

"""
    _compute_stock_moments(cps_w) -> Dict{Symbol,Float64}

CPS stock moments for one window, computed per month and averaged over months:
`ur_total`, `ur_U` and `skilled_share` on the labour force excluding trainees,
`ur_S` on the whole labour force (no trainee is skilled), and a CPS
`training_share` (NILF trainees over population) that `make_moments` replaces
with the enrolment target.
"""
function _compute_stock_moments(cps_w::DataFrame)::Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()

    lf_excl_train = if hasproperty(cps_w, :in_lf) && hasproperty(cps_w, :in_training)
        filter(r -> r.in_lf && !r.in_training, cps_w)
    elseif hasproperty(cps_w, :in_lf)
        filter(r -> r.in_lf, cps_w)
    else
        cps_w
    end

    lf_all = hasproperty(cps_w, :in_lf) ? filter(r -> r.in_lf, cps_w) : cps_w

    monthly = NamedTuple[]
    for gk in groupby(lf_excl_train, [:YEAR, :MONTH])
        g  = DataFrame(gk)
        w  = Float64.(coalesce.(g.WTFINL, 0.0))
        sw = sum(w)
        sw <= 0 && continue
        yr = Int(g.YEAR[1]); mo = Int(g.MONTH[1])

        n_u_U  = sum(w[g.unemployed .& .!g.skilled])
        n_u_S_excl = sum(w[g.unemployed .& g.skilled])
        n_lf_U = sum(w[.!g.skilled])
        n_lf_S_excl = sum(w[g.skilled])

        ur_total      = sw    > 0 ? (n_u_U + n_u_S_excl) / sw    : NaN
        ur_U          = n_lf_U > 0 ? n_u_U / n_lf_U : NaN
        skilled_share = sw    > 0 ? n_lf_S_excl / sw    : NaN

        push!(monthly, (yr=yr, mo=mo,
                         ur_total=ur_total, ur_U=ur_U,
                         skilled_share=skilled_share))
    end

    monthly_S = NamedTuple[]
    for gk in groupby(lf_all, [:YEAR, :MONTH])
        g  = DataFrame(gk)
        w  = Float64.(coalesce.(g.WTFINL, 0.0))
        n_u_S  = sum(w[g.unemployed .& g.skilled])
        n_lf_S = sum(w[g.skilled])
        ur_S   = n_lf_S > 0 ? n_u_S / n_lf_S : NaN
        push!(monthly_S, (yr=Int(g.YEAR[1]), mo=Int(g.MONTH[1]), ur_S=ur_S))
    end

    monthly_t = NamedTuple[]
    for gk in groupby(cps_w, [:YEAR, :MONTH])
        g    = DataFrame(gk)
        w    = Float64.(coalesce.(g.WTFINL, 0.0))
        pop  = sum(w)
        pop <= 0 && continue
        if hasproperty(g, :in_training) && hasproperty(g, :in_lf)
            trainees = sum(w[coalesce.(g.in_training, false) .& .!g.in_lf])
            push!(monthly_t, (yr=Int(g.YEAR[1]), mo=Int(g.MONTH[1]),
                              training_share = trainees / pop))
        end
    end

    if !isempty(monthly)
        mdf = DataFrame(monthly)
        moments[:ur_total]      = mean(filter(isfinite, mdf.ur_total))
        moments[:ur_U]          = mean(filter(isfinite, mdf.ur_U))
        moments[:skilled_share] = mean(filter(isfinite, mdf.skilled_share))
    end
    if !isempty(monthly_S)
        msdf = DataFrame(monthly_S)
        moments[:ur_S] = mean(filter(isfinite, msdf.ur_S))
    end
    if !isempty(monthly_t)
        mtdf = DataFrame(monthly_t)
        moments[:training_share] = mean(filter(isfinite, mtdf.training_share))
    end
    return moments
end

"""
    _compute_duration_moments(cps_w) -> Dict{Symbol,Float64}

Unemployment-duration survivor moments for one window: for each market and each
threshold in `DURATION_THRESHOLDS_WK`, the WTFINL-weighted share of the
unemployed whose ongoing spell (`_durw`) has lasted at least t weeks, per month
and then averaged. The U pool excludes trainees as `ur_U` does; the S pool needs
no exclusion. `ltu_share_S` is set to `usurv27_S`. All NaN if `DURUNEMP` is absent.
"""
function _compute_duration_moments(cps_w::DataFrame)::Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()
    if !hasproperty(cps_w, :DURUNEMP)
        @warn "  DURUNEMP not in cleaned CPS Basic — the duration profile is left as NaN. " *
              "Add DURUNEMP to the CPS Basic extract and cps_basic.jl::cols_to_keep."
        for k in DURATION_MOMENT_NAMES; moments[k] = NaN; end
        moments[:ltu_share_S] = NaN
        return moments
    end

    intr = hasproperty(cps_w, :in_training) ?
           Vector{Bool}(coalesce.(cps_w.in_training, false)) : falses(nrow(cps_w))
    cps_w = copy(cps_w); cps_w.in_train_flag = intr

    monthly = Dict(k => Float64[] for k in DURATION_MOMENT_NAMES)
    for gk in groupby(cps_w, [:YEAR, :MONTH])
        g = DataFrame(gk)
        for (tag, pool) in (("U", g.unemployed .& .!g.skilled .& .!g.in_train_flag),
                            ("S", g.unemployed .& g.skilled))
            any(pool) || continue
            w   = Float64.(coalesce.(g.WTFINL, 0.0))[pool]
            dur = _durw.(g.DURUNEMP[pool])
            sw  = sum(w)
            sw <= 0 && continue
            for t in DURATION_THRESHOLDS_WK
                push!(monthly[Symbol("usurv$(Int(t))_$(tag)")], sum(w[dur .>= t]) / sw)
            end
        end
    end
    for k in DURATION_MOMENT_NAMES
        v = filter(isfinite, monthly[k])
        moments[k] = isempty(v) ? NaN : mean(v)
    end
    moments[:ltu_share_S] = moments[:usurv27_S]
    return moments
end

"""
    _fill_transition_moments!(moments, trans_w)

Set `jfr_U/S` and `sep_rate_U/S` (monthly probabilities) in `moments` as ratios
of totals over the window's months: total weighted flow over total weighted
at-risk pool. The model's rates are stationary flows over stationary stocks;
averaging monthly ratios would over-weight months with small pools.
"""
function _fill_transition_moments!(moments::Dict{Symbol, Float64},
                                    trans_w::DataFrame)
    for sk_val in (false, true)
        rows = filter(r -> Bool(r.skilled) == sk_val, trans_w)
        isempty(rows) && continue
        jfr_name = sk_val ? :jfr_S      : :jfr_U
        sep_name = sk_val ? :sep_rate_S : :sep_rate_U
        wu = sum(Float64.(rows.w_u)); we = sum(Float64.(rows.w_e))
        moments[jfr_name] = wu > 0 ? sum(Float64.(rows.w_ue)) / wu : NaN
        moments[sep_name] = we > 0 ? sum(Float64.(rows.w_eu)) / we : NaN
    end
end

"""
    _compute_wage_moments_per_year(asec_w) -> Dict{Symbol,Float64}

ASEC log-wage moments for one window, computed per survey year and averaged over
years: mean, variance, third central moment and p10/p25/p50/p75/p90 per market,
the log skill premium, and the overlap pair (`overlap_UgtS`: share of unskilled
above that year's skilled median; `overlap_SltU`: share of skilled below the
unskilled median). Logs because the model's wage moments are in logs.
"""
function _compute_wage_moments_per_year(asec_w::DataFrame)::Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()

    if !hasproperty(asec_w, :YEAR) || nrow(asec_w) == 0
        return moments
    end

    yr_mean_U = Float64[]; yr_mean_S = Float64[]
    yr_var_U  = Float64[]; yr_var_S  = Float64[]
    yr_cm3_U  = Float64[]; yr_cm3_S  = Float64[]
    yr_med_U  = Float64[]; yr_med_S  = Float64[]
    yr_p10_U  = Float64[]; yr_p10_S  = Float64[]
    yr_p25_U  = Float64[]; yr_p25_S  = Float64[]
    yr_p75_U  = Float64[]; yr_p75_S  = Float64[]
    yr_p90_U  = Float64[]; yr_p90_S  = Float64[]
    yr_prem   = Float64[]
    yr_ov_UgtS = Float64[]; yr_ov_SltU = Float64[]

    for gk in groupby(asec_w, :YEAR)
        g = DataFrame(gk)
        unskilled = filter(r -> !r.skilled, g)
        skilled   = filter(r ->  r.skilled, g)

        if nrow(unskilled) > 0
            wu = log.(max.(Float64.(unskilled.wage_norm), 1e-14))
            wt = Float64.(unskilled.ASECWT)
            push!(yr_mean_U, wmean(wu, wt))
            push!(yr_var_U,  wvar(wu, wt))
            push!(yr_cm3_U,  wcm3(wu, wt))
            push!(yr_med_U,  wmedian(wu, wt))
            push!(yr_p10_U,  wpercentile10(wu, wt))
            push!(yr_p25_U,  wpercentile25(wu, wt))
            push!(yr_p75_U,  wpercentile75(wu, wt))
            push!(yr_p90_U,  wpercentile90(wu, wt))
        end

        if nrow(skilled) > 0
            ws = log.(max.(Float64.(skilled.wage_norm), 1e-14))
            wt = Float64.(skilled.ASECWT)
            push!(yr_mean_S, wmean(ws, wt))
            push!(yr_var_S,  wvar(ws, wt))
            push!(yr_cm3_S,  wcm3(ws, wt))
            push!(yr_med_S,  wmedian(ws, wt))
            push!(yr_p10_S,  wpercentile10(ws, wt))
            push!(yr_p25_S,  wpercentile25(ws, wt))
            push!(yr_p75_S,  wpercentile75(ws, wt))
            push!(yr_p90_S,  wpercentile90(ws, wt))
        end

        if nrow(unskilled) > 0 && nrow(skilled) > 0
            log_wu = log.(max.(Float64.(unskilled.wage_norm), 1e-14))
            log_ws = log.(max.(Float64.(skilled.wage_norm),   1e-14))
            wt_u   = Float64.(unskilled.ASECWT)
            wt_s   = Float64.(skilled.ASECWT)
            push!(yr_prem, wmean(log_ws, wt_s) - wmean(log_wu, wt_u))

            med_S = wmedian(log_ws, wt_s)
            med_U = wmedian(log_wu, wt_u)
            sw_u  = sum(wt_u); sw_s = sum(wt_s)
            if sw_u > 0 && sw_s > 0
                push!(yr_ov_UgtS, sum(wt_u[log_wu .> med_S]) / sw_u)
                push!(yr_ov_SltU, sum(wt_s[log_ws .< med_U]) / sw_s)
            end
        end
    end

    finite_mean(v) = isempty(v) ? NaN : mean(filter(isfinite, v))

    moments[:mean_wage_U]  = finite_mean(yr_mean_U)
    moments[:mean_wage_S]  = finite_mean(yr_mean_S)
    moments[:emp_var_U]    = finite_mean(yr_var_U)
    moments[:emp_var_S]    = finite_mean(yr_var_S)
    moments[:emp_cm3_U]    = finite_mean(yr_cm3_U)
    moments[:emp_cm3_S]    = finite_mean(yr_cm3_S)
    moments[:p10_wage_U]   = finite_mean(yr_p10_U)
    moments[:p10_wage_S]   = finite_mean(yr_p10_S)
    moments[:p25_wage_U]   = finite_mean(yr_p25_U)
    moments[:p25_wage_S]   = finite_mean(yr_p25_S)
    moments[:p50_wage_U]   = finite_mean(yr_med_U)
    moments[:p50_wage_S]   = finite_mean(yr_med_S)
    moments[:p75_wage_U]   = finite_mean(yr_p75_U)
    moments[:p75_wage_S]   = finite_mean(yr_p75_S)
    moments[:p90_wage_U]   = finite_mean(yr_p90_U)
    moments[:p90_wage_S]   = finite_mean(yr_p90_S)
    moments[:wage_premium] = finite_mean(yr_prem)
    moments[:overlap_UgtS] = finite_mean(yr_ov_UgtS)
    moments[:overlap_SltU] = finite_mean(yr_ov_SltU)

    return moments
end

"""
    _compute_tightness(jolts_w, cps_w) -> Dict{Symbol,Float64}

Data tightness θ_U and θ_S for one window as a ratio of means: Σ V_j / Σ U_j
over the months where both are observed (the model's θ_j is a stationary V/U;
the mean of monthly ratios would be biased up by the convexity of V/U in U).
U_U counts unskilled unemployed in the labour force excluding trainees and U_S
skilled unemployed in the whole labour force, the same populations as `ur_U`
and `ur_S`, since the model derives each market's rate and tightness from one θ_j.
"""
function _compute_tightness(jolts_w::DataFrame, cps_w::DataFrame)::Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()

    cps_U = if hasproperty(cps_w, :in_lf) && hasproperty(cps_w, :in_training)
        filter(r -> r.in_lf && !r.in_training, cps_w)
    elseif hasproperty(cps_w, :in_lf)
        filter(r -> r.in_lf, cps_w)
    else
        cps_w
    end
    cps_S = hasproperty(cps_w, :in_lf) ? filter(r -> r.in_lf, cps_w) : cps_w

    monthly_UU = Dict{Tuple{Int,Int}, Float64}()
    for gk in groupby(cps_U, [:YEAR, :MONTH])
        g  = DataFrame(gk)
        w  = Float64.(coalesce.(g.WTFINL, 0.0))
        monthly_UU[(Int(g.YEAR[1]), Int(g.MONTH[1]))] = sum(w[g.unemployed .& .!g.skilled])
    end

    monthly_U = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
    for gk in groupby(cps_S, [:YEAR, :MONTH])
        g  = DataFrame(gk)
        w  = Float64.(coalesce.(g.WTFINL, 0.0))
        yr = Int(g.YEAR[1]); mo = Int(g.MONTH[1])
        monthly_U[(yr, mo)] = (
            get(monthly_UU, (yr, mo), NaN),          # U_U on the train-excluded LF
            sum(w[g.unemployed .&  g.skilled])       # U_S on the unrestricted LF
        )
    end

    sV_U = 0.0; sU_U = 0.0; n_U = 0
    sV_S = 0.0; sU_S = 0.0; n_S = 0

    for row in eachrow(jolts_w)
        yr = Int(row.YEAR); mo = Int(row.MONTH)
        (U_U, U_S) = get(monthly_U, (yr, mo), (NaN, NaN))
        if isfinite(U_U) && U_U > 0 && isfinite(row.V_U)
            sV_U += row.V_U; sU_U += U_U; n_U += 1
        end
        if isfinite(U_S) && U_S > 0 && isfinite(row.V_S)
            sV_S += row.V_S; sU_S += U_S; n_S += 1
        end
    end

    moments[:theta_U] = n_U == 0 ? NaN : sV_U / sU_U
    moments[:theta_S] = n_S == 0 ? NaN : sV_S / sU_S
    return moments
end

"""
    make_moments() -> Dict{Symbol,DataFrame}

Build the moment vector for every window in `MOMENT_NAMES` order and write
moments_{window}.csv (columns `moment`, `value`). Sources: CPS stocks, durations
and θ denominators; CPS flows (jfr, sep); ASEC wages; JOLTS vacancies; J2J
(`ee_rate_S`); SIPP (`ee_step_S`, `wchg_rate_U/S`, the shipped columns); the
enrolment target (`training_share`). A moment with no source data is NaN, which
holds it out of the objective. Returns window → moment table.
"""
function make_moments()
    @info "Stage 7: assembling all $(length(MOMENT_NAMES)) moments × 4 windows..."

    cps_basic_m   = _load_arrow("cps_basic_clean.arrow")
    cps_asec_m    = _load_arrow("cps_asec_clean.arrow")
    trans_monthly = _load_arrow("transitions_monthly.arrow")
    jolts_m       = _load_arrow("jolts_clean.arrow")

    j2j_path = joinpath(DERIVED_DIR, "j2j_ee_rates.csv")
    j2j_ee   = isfile(j2j_path) ? CSV.read(j2j_path, DataFrame) : DataFrame()
    if !isempty(j2j_ee) && hasproperty(j2j_ee, :window)
        j2j_ee.window = Symbol.(j2j_ee.window)
    end

    # One row per window that has its own SIPP data (no cross-window borrowing).
    sipp_path = joinpath(DERIVED_DIR, "sipp_wchg_rates.csv")
    sipp_wchg = isfile(sipp_path) ? CSV.read(sipp_path, DataFrame) : DataFrame()
    if !isempty(sipp_wchg) && hasproperty(sipp_wchg, :window)
        sipp_wchg.window = Symbol.(sipp_wchg.window)
    end

    sipp_ee_path = joinpath(DERIVED_DIR, "sipp_ee_rates.csv")
    sipp_ee = isfile(sipp_ee_path) ? CSV.read(sipp_ee_path, DataFrame) : DataFrame()
    if !isempty(sipp_ee) && hasproperty(sipp_ee, :window)
        sipp_ee.window = Symbol.(sipp_ee.window)
    end

    for df in (cps_basic_m, cps_asec_m, trans_monthly, jolts_m)
        hasproperty(df, :window) && (df.window = Symbol.(df.window))
    end

    all_moments = Dict{Symbol, DataFrame}()

    for (wname, wdef) in WINDOWS
        @info "  Window: $(wdef.label) ($wname)"
        moments = Dict{Symbol, Float64}()

        # A. CPS stocks and the duration profile (which carries ltu_share_S)
        cps_w = filter(row -> row.window == wname, cps_basic_m)
        if nrow(cps_w) > 0
            merge!(moments, _compute_stock_moments(cps_w))
            merge!(moments, _compute_duration_moments(cps_w))
        else
            for k in (:ur_total, :ur_U, :ur_S, :skilled_share, :training_share,
                      :ltu_share_S, DURATION_MOMENT_NAMES...)
                moments[k] = NaN
            end
        end

        # B. CPS flows (jfr_j, sep_rate_j)
        trans_w = filter(r -> r.window == wname, trans_monthly)
        if nrow(trans_w) > 0
            _fill_transition_moments!(moments, trans_w)
        else
            for k in (:jfr_U, :sep_rate_U, :jfr_S, :sep_rate_S)
                moments[k] = NaN
            end
        end

        # C. Skilled EE: the rate from J2J (administrative EE flows), the wage
        #    step from SIPP (the only source with the wage change). The SIPP
        #    rate is printed for comparison only.
        j2j_w  = isempty(j2j_ee) ? nothing : filter(r -> r.window == wname, j2j_ee)
        sipp_e = isempty(sipp_ee) ? nothing : filter(r -> r.window == wname, sipp_ee)
        ee_sipp = (!isnothing(sipp_e) && nrow(sipp_e) > 0) ? sipp_e.ee_rate_S[1] : NaN
        ee_j2j  = (!isnothing(j2j_w)  && nrow(j2j_w)  > 0) ? j2j_w.ee_rate_S[1]  : NaN
        moments[:ee_rate_S] = ee_j2j
        moments[:ee_step_S] = (!isnothing(sipp_e) && nrow(sipp_e) > 0) ? sipp_e.ee_step_S[1] : NaN
        @printf("    ee_rate_S source comparison  %-13s SIPP=%s  J2J=%s  (target ← J2J)\n",
                string(wname),
                isfinite(ee_sipp) ? @sprintf("%.5f", ee_sipp) : "  NaN",
                isfinite(ee_j2j)  ? @sprintf("%.5f", ee_j2j)  : "  NaN")

        # C'. SIPP within-job wage-change hazards: the shipped columns (BBG on
        #     the FC windows, raw earnings on COVID), already −log(1−p).
        sipp_w = isempty(sipp_wchg) ? nothing : filter(r -> r.window == wname, sipp_wchg)
        if !isnothing(sipp_w) && nrow(sipp_w) > 0
            moments[:wchg_rate_U] = sipp_w.wchg_rate_U[1]
            moments[:wchg_rate_S] = sipp_w.wchg_rate_S[1]
        else
            moments[:wchg_rate_U] = NaN
            moments[:wchg_rate_S] = NaN
        end

        # D. ASEC wage moments, per survey year then averaged
        asec_w = filter(r -> r.window == wname, cps_asec_m)
        if nrow(asec_w) > 0
            merge!(moments, _compute_wage_moments_per_year(asec_w))
        else
            for k in (:mean_wage_U, :mean_wage_S, :emp_var_U, :emp_cm3_U,
                      :emp_var_S, :emp_cm3_S,
                      :p10_wage_U, :p10_wage_S, :p25_wage_U, :p25_wage_S,
                      :p50_wage_U, :p50_wage_S, :p75_wage_U, :p75_wage_S,
                      :p90_wage_U, :p90_wage_S, :wage_premium,
                      :overlap_UgtS, :overlap_SltU)
                moments[k] = NaN
            end
        end

        # E. Tightness (JOLTS V over CPS U, ratio of window means)
        jolts_w = filter(r -> r.window == wname, jolts_m)
        if nrow(jolts_w) > 0 && nrow(cps_w) > 0
            merge!(moments, _compute_tightness(jolts_w, cps_w))
        else
            moments[:theta_U] = NaN
            moments[:theta_S] = NaN
        end

        # F. Monthly transition probabilities → Poisson hazards −log(1−p), the
        #    model's rate objects. Only the CPS flows: ee_rate_S (J2J) and
        #    wchg_rate_j (SIPP) arrive converted and must not be converted twice;
        #    stocks, ratios and wage moments are not rates.
        for k in (:jfr_U, :jfr_S, :sep_rate_U, :sep_rate_S)
            p = get(moments, k, NaN)
            if isfinite(p) && 0.0 <= p < 1.0
                moments[k] = -log(1.0 - p)
            end
        end

        moment_df = DataFrame(moment=String[], value=Float64[])
        for mname in MOMENT_NAMES
            push!(moment_df, (string(mname), get(moments, mname, NaN)))
        end

        # training_share is the enrolment target (nsc.jl), not the CPS count,
        # whose SCHLCOLL universe is not comparable across the FC and COVID
        # pairs. Replaced before the print and before all_moments, so one
        # number carries the name everywhere downstream, including validation.
        ts_tgt = _load_training_share_target(wname).target
        for r in eachrow(moment_df)
            r.moment == "training_share" && (r.value = ts_tgt)
        end

        CSV.write(joinpath(DERIVED_DIR, "moments_$(wname).csv"), moment_df)
        all_moments[wname] = moment_df

        for row in eachrow(moment_df)
            @printf("    %-22s = %.6g\n", row.moment, row.value)
        end
    end

    @info "  All moment files saved to $(DERIVED_DIR)"
    return all_moments
end

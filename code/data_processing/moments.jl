############################################################
# data_processing/moments.jl
#
# Combine step (Stage 7) — assemble all 23 empirical moments × 4 windows
# from every cleaned dataset and write one moments_{window}.csv per window.
# The training_share level carries the NSC κ_w adjustment (written here so
# the SMM loaders read it pre-adjusted). The in-memory return keeps the raw
# training_share so Stage 9 diagnostics are unaffected.
#
# Reads:  cps_basic_clean.arrow, cps_asec_clean.arrow, transitions_monthly.arrow, jolts_clean.arrow, j2j_ee_rates.csv, training_share_scale.csv
# Writes: moments_{window}.csv  (window ∈ base_fc, crisis_fc, base_covid, crisis_covid)
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages and path consts come from data_processing_main.jl.
############################################################

# ──────────────────────────────────────────────────────────────────────────
# Stage 7 helpers — per-period-then-average approach
#
# Stock denominators apply the LF ∩ ¬train filter for ur_total, ur_U,
# and skilled_share. training_share is the strict variant (NILF trainees
# in numerator, working-age population in denominator).
# ──────────────────────────────────────────────────────────────────────────

function _compute_stock_moments(cps_w::DataFrame)::Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()

    # LF denominator with the working-student exclusion baked in. This
    # propagates to ur_total, ur_U, and skilled_share automatically.
    lf_excl_train = if hasproperty(cps_w, :in_lf) && hasproperty(cps_w, :in_training)
        filter(r -> r.in_lf && !r.in_training, cps_w)
    elseif hasproperty(cps_w, :in_lf)
        filter(r -> r.in_lf, cps_w)
    else
        cps_w
    end

    # For ur_S we use the unrestricted LF (train flag implies EDUC < 111
    # so the skilled denominator never contains a trainee anyway).
    lf_all = hasproperty(cps_w, :in_lf) ? filter(r -> r.in_lf, cps_w) : cps_w

    # Monthly stock values, then averaged across months in the window.
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

    # ur_S on the unrestricted skilled LF (denominator includes no trainees
    # by construction; numerator is the unemployed skilled count).
    monthly_S = NamedTuple[]
    for gk in groupby(lf_all, [:YEAR, :MONTH])
        g  = DataFrame(gk)
        w  = Float64.(coalesce.(g.WTFINL, 0.0))
        n_u_S  = sum(w[g.unemployed .& g.skilled])
        n_lf_S = sum(w[g.skilled])
        ur_S   = n_lf_S > 0 ? n_u_S / n_lf_S : NaN
        push!(monthly_S, (yr=Int(g.YEAR[1]), mo=Int(g.MONTH[1]), ur_S=ur_S))
    end

    # training_share — strict variant: NILF ∩ train numerator,
    # working-age population denominator. cps_w is already restricted
    # to ages 16–64 upstream (in clean_cps_basic).
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

function _fill_transition_moments!(moments::Dict{Symbol, Float64},
                                    trans_w::DataFrame)
    # jfr_j, sep_rate_j: n_pairs-weighted mean across month-pairs in window.
    for sk_val in (false, true)
        rows = filter(r -> Bool(r.skilled) == sk_val, trans_w)
        isempty(rows) && continue
        jfr_name = sk_val ? :jfr_S      : :jfr_U
        sep_name = sk_val ? :sep_rate_S : :sep_rate_U
        valid_jfr = filter(isfinite, Float64.(rows.jfr))
        valid_sep = filter(isfinite, Float64.(rows.sep))
        moments[jfr_name] = isempty(valid_jfr) ? NaN : mean(valid_jfr)
        moments[sep_name] = isempty(valid_sep) ? NaN : mean(valid_sep)
    end
end

function _compute_wage_moments_per_year(asec_w::DataFrame)::Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()

    if !hasproperty(asec_w, :YEAR) || nrow(asec_w) == 0
        return moments
    end

    yr_mean_U = Float64[]; yr_mean_S = Float64[]
    yr_var_U  = Float64[]; yr_var_S  = Float64[]
    yr_cm3_U  = Float64[]; yr_cm3_S  = Float64[]
    yr_med_U  = Float64[]; yr_med_S  = Float64[]
    yr_p25_U  = Float64[]; yr_p25_S  = Float64[]
    yr_prem   = Float64[]

    for gk in groupby(asec_w, :YEAR)
        g = DataFrame(gk)
        unskilled = filter(r -> !r.skilled, g)
        skilled   = filter(r ->  r.skilled, g)

        if nrow(unskilled) > 0
            wu = Float64.(unskilled.wage_norm)
            wt = Float64.(unskilled.ASECWT)
            push!(yr_mean_U, wmean(wu, wt))
            push!(yr_var_U,  wvar(wu, wt))
            push!(yr_cm3_U,  wcm3(wu, wt))
            push!(yr_med_U,  wmedian(wu, wt))
            push!(yr_p25_U,  wpercentile25(wu, wt))
        end

        if nrow(skilled) > 0
            ws = Float64.(skilled.wage_norm)
            wt = Float64.(skilled.ASECWT)
            push!(yr_mean_S, wmean(ws, wt))
            push!(yr_var_S,  wvar(ws, wt))
            push!(yr_cm3_S,  wcm3(ws, wt))
            push!(yr_med_S,  wmedian(ws, wt))
            push!(yr_p25_S,  wpercentile25(ws, wt))
        end

        if nrow(unskilled) > 0 && nrow(skilled) > 0
            log_wu = log.(max.(Float64.(unskilled.wage_norm), 1e-14))
            log_ws = log.(max.(Float64.(skilled.wage_norm),   1e-14))
            prem_yr = wmean(log_ws, Float64.(skilled.ASECWT)) -
                      wmean(log_wu, Float64.(unskilled.ASECWT))
            push!(yr_prem, prem_yr)
        end
    end

    finite_mean(v) = isempty(v) ? NaN : mean(filter(isfinite, v))

    moments[:mean_wage_U]  = finite_mean(yr_mean_U)
    moments[:mean_wage_S]  = finite_mean(yr_mean_S)
    moments[:emp_var_U]    = finite_mean(yr_var_U)
    moments[:emp_var_S]    = finite_mean(yr_var_S)
    moments[:emp_cm3_U]    = finite_mean(yr_cm3_U)
    moments[:emp_cm3_S]    = finite_mean(yr_cm3_S)
    moments[:p50_wage_U]   = finite_mean(yr_med_U)
    moments[:p50_wage_S]   = finite_mean(yr_med_S)
    moments[:p25_wage_U]   = finite_mean(yr_p25_U)
    moments[:p25_wage_S]   = finite_mean(yr_p25_S)
    moments[:wage_premium] = finite_mean(yr_prem)

    return moments
end

function _compute_tightness_per_month(jolts_w::DataFrame, cps_w::DataFrame)::Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()

    monthly_U = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
    for gk in groupby(cps_w, [:YEAR, :MONTH])
        g  = DataFrame(gk)
        w  = Float64.(coalesce.(g.WTFINL, 0.0))
        yr = Int(g.YEAR[1]); mo = Int(g.MONTH[1])
        monthly_U[(yr, mo)] = (
            sum(w[g.unemployed .& .!g.skilled]),
            sum(w[g.unemployed .&  g.skilled])
        )
    end

    theta_U_vals = Float64[]
    theta_S_vals = Float64[]

    for row in eachrow(jolts_w)
        yr = Int(row.YEAR); mo = Int(row.MONTH)
        (U_U, U_S) = get(monthly_U, (yr, mo), (NaN, NaN))
        if isfinite(U_U) && U_U > 0 && isfinite(row.V_U)
            push!(theta_U_vals, row.V_U / U_U)
        end
        if isfinite(U_S) && U_S > 0 && isfinite(row.V_S)
            push!(theta_S_vals, row.V_S / U_S)
        end
    end

    moments[:theta_U] = isempty(theta_U_vals) ? NaN : mean(filter(isfinite, theta_U_vals))
    moments[:theta_S] = isempty(theta_S_vals) ? NaN : mean(filter(isfinite, theta_S_vals))
    return moments
end

# ──────────────────────────────────────────────────────────────────────────
# Stage 7 main: assemble all 23 moments × 4 windows
# ──────────────────────────────────────────────────────────────────────────

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

    for df in (cps_basic_m, cps_asec_m, trans_monthly, jolts_m)
        hasproperty(df, :window) && (df.window = Symbol.(df.window))
    end

    all_moments = Dict{Symbol, DataFrame}()

    for (wname, wdef) in WINDOWS
        @info "  Window: $(wdef.label) ($wname)"
        moments = Dict{Symbol, Float64}()

        # A. Stock moments — ur_total, ur_U, ur_S, skilled_share, training_share
        cps_w = filter(row -> row.window == wname, cps_basic_m)
        if nrow(cps_w) > 0
            merge!(moments, _compute_stock_moments(cps_w))
        else
            for k in (:ur_total, :ur_U, :ur_S, :skilled_share, :training_share)
                moments[k] = NaN
            end
        end

        # B. Transition moments (jfr_j, sep_rate_j)
        trans_w = filter(r -> r.window == wname, trans_monthly)
        if nrow(trans_w) > 0
            _fill_transition_moments!(moments, trans_w)
        else
            for k in (:jfr_U, :sep_rate_U, :jfr_S, :sep_rate_S)
                moments[k] = NaN
            end
        end

        # C. EE rate (J2J, already window-averaged in Stage 5)
        j2j_w = isempty(j2j_ee) ? nothing : filter(r -> r.window == wname, j2j_ee)
        if !isnothing(j2j_w) && nrow(j2j_w) > 0
            moments[:ee_rate_S] = j2j_w.ee_rate_S[1]
        else
            moments[:ee_rate_S] = NaN
        end

        # D. Wage moments (ASEC, per-survey-year then average)
        asec_w = filter(r -> r.window == wname, cps_asec_m)
        if nrow(asec_w) > 0
            merge!(moments, _compute_wage_moments_per_year(asec_w))
        else
            for k in (:mean_wage_U, :mean_wage_S, :emp_var_U, :emp_cm3_U,
                      :emp_var_S, :emp_cm3_S, :p25_wage_U, :p25_wage_S,
                      :p50_wage_U, :p50_wage_S, :wage_premium)
                moments[k] = NaN
            end
        end

        # E. Tightness (JOLTS, per-month theta then average)
        jolts_w = filter(r -> r.window == wname, jolts_m)
        if nrow(jolts_w) > 0 && nrow(cps_w) > 0
            merge!(moments, _compute_tightness_per_month(jolts_w, cps_w))
        else
            moments[:theta_U] = NaN
            moments[:theta_S] = NaN
        end

        # Build moment DataFrame in canonical MOMENT_NAMES order
        moment_df = DataFrame(moment=String[], value=Float64[])
        for mname in MOMENT_NAMES
            push!(moment_df, (string(mname), get(moments, mname, NaN)))
        end

        # NSC level adjustment: write the κ_w-scaled training_share into
        # moments_{window}.csv so the saved target already reflects the NSC
        # IPEDS-Universe level. The in-memory `all_moments` keeps the raw
        # values, so the Stage 9 diagnostics below are unchanged.
        moment_df_out = copy(moment_df)
        κ_ts = _load_training_share_scale(wname)
        if κ_ts != 1.0
            for r in eachrow(moment_df_out)
                r.moment == "training_share" && (r.value *= κ_ts)
            end
        end

        CSV.write(joinpath(DERIVED_DIR, "moments_$(wname).csv"), moment_df_out)
        all_moments[wname] = moment_df

        for row in eachrow(moment_df)
            @printf("    %-22s = %.6g\n", row.moment, row.value)
        end
    end

    @info "  All moment files saved to $(DERIVED_DIR)"
    return all_moments
end

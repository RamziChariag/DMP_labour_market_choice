############################################################
# data_processing/moments.jl
#
# Combine step (Stage 7) — assemble all 31 empirical moments × 4 windows
# from every cleaned dataset and write one moments_{window}.csv per window.
# The training_share level comes from NSC (training_share_target.csv); the
# substitution happens before both the trace and the returned table, so the
# name refers to one quantity everywhere downstream.
#
# Reads:  cps_basic_clean.arrow, cps_asec_clean.arrow, transitions_monthly.arrow, jolts_clean.arrow, j2j_ee_rates.csv, sipp_wchg_rates.csv, training_share_target.csv
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

function _compute_ltu_share(cps_w::DataFrame)::Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()
    # ltu_share_S: WTFINL-weighted share of skilled unemployed whose current
    # spell is long-term (DURUNEMP ≥ 27 weeks), computed per month then
    # averaged across the window. Matches the model survivor ltu_share_S at
    # a* ≈ 6.23 months. Requires DURUNEMP (added to cps_basic.jl cols_to_keep).
    if !hasproperty(cps_w, :DURUNEMP)
        @warn "  DURUNEMP not in cleaned CPS Basic — ltu_share_S left as NaN. " *
              "Add DURUNEMP to the CPS Basic extract and cps_basic.jl::cols_to_keep."
        moments[:ltu_share_S] = NaN
        return moments
    end

    monthly = Float64[]
    for gk in groupby(cps_w, [:YEAR, :MONTH])
        g  = DataFrame(gk)
        skl_u = g.unemployed .& g.skilled
        any(skl_u) || continue
        w   = Float64.(coalesce.(g.WTFINL, 0.0))[skl_u]
        dur = _durw.(g.DURUNEMP[skl_u])
        sw  = sum(w)
        sw <= 0 && continue
        push!(monthly, sum(w[dur .>= 27.0]) / sw)
    end
    moments[:ltu_share_S] = isempty(monthly) ? NaN : mean(filter(isfinite, monthly))
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
    yr_p75_U  = Float64[]; yr_p75_S  = Float64[]
    yr_prem   = Float64[]
    yr_ov_UgtS = Float64[]; yr_ov_SltU = Float64[]

    for gk in groupby(asec_w, :YEAR)
        g = DataFrame(gk)
        unskilled = filter(r -> !r.skilled, g)
        skilled   = filter(r ->  r.skilled, g)

        if nrow(unskilled) > 0
            # LOG wages — model-side wage moments are computed in logs, so the
            # data targets (mean/var/cm3/median/p25/p75) must match.
            wu = log.(max.(Float64.(unskilled.wage_norm), 1e-14))
            wt = Float64.(unskilled.ASECWT)
            push!(yr_mean_U, wmean(wu, wt))
            push!(yr_var_U,  wvar(wu, wt))
            push!(yr_cm3_U,  wcm3(wu, wt))
            push!(yr_med_U,  wmedian(wu, wt))
            push!(yr_p25_U,  wpercentile25(wu, wt))
            push!(yr_p75_U,  wpercentile75(wu, wt))
        end

        if nrow(skilled) > 0
            ws = log.(max.(Float64.(skilled.wage_norm), 1e-14))
            wt = Float64.(skilled.ASECWT)
            push!(yr_mean_S, wmean(ws, wt))
            push!(yr_var_S,  wvar(ws, wt))
            push!(yr_cm3_S,  wcm3(ws, wt))
            push!(yr_med_S,  wmedian(ws, wt))
            push!(yr_p25_S,  wpercentile25(ws, wt))
            push!(yr_p75_S,  wpercentile75(ws, wt))
        end

        if nrow(unskilled) > 0 && nrow(skilled) > 0
            log_wu = log.(max.(Float64.(unskilled.wage_norm), 1e-14))
            log_ws = log.(max.(Float64.(skilled.wage_norm),   1e-14))
            wt_u   = Float64.(unskilled.ASECWT)
            wt_s   = Float64.(skilled.ASECWT)
            push!(yr_prem, wmean(log_ws, wt_s) - wmean(log_wu, wt_u))

            # Cross-market wage overlap, computed within the year against that
            # year's medians (one moment per bargaining weight):
            #   overlap_UgtS = weighted share of unskilled with logw > median(skilled logw)
            #   overlap_SltU = weighted share of skilled   with logw < median(unskilled logw)
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
    moments[:p50_wage_U]   = finite_mean(yr_med_U)
    moments[:p50_wage_S]   = finite_mean(yr_med_S)
    moments[:p25_wage_U]   = finite_mean(yr_p25_U)
    moments[:p25_wage_S]   = finite_mean(yr_p25_S)
    moments[:p75_wage_U]   = finite_mean(yr_p75_U)
    moments[:p75_wage_S]   = finite_mean(yr_p75_S)
    moments[:wage_premium] = finite_mean(yr_prem)
    moments[:overlap_UgtS] = finite_mean(yr_ov_UgtS)
    moments[:overlap_SltU] = finite_mean(yr_ov_SltU)

    return moments
end

function _compute_tightness_per_month(jolts_w::DataFrame, cps_w::DataFrame)::Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()

    # The unemployment counts in the denominators of θ_U and θ_S must be the SAME
    # populations that appear in ur_U and ur_S, or the model — which derives both
    # the rate and the tightness from one θ_j through free entry — faces two
    # mutually inconsistent targets.  ur_U is built on the train-excluded labour
    # force (in_lf & !in_training), so θ_U must be too: `in_training` is an
    # enrolment flag independent of `in_lf`, so an enrolled non-BA worker can be
    # counted unemployed and would otherwise sit in θ_U's denominator while being
    # absent from ur_U's.  ur_S uses the unrestricted labour force (the trainee
    # flag implies EDUC < 111, so no trainee is skilled), so θ_S keeps the raw frame.
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
# Stage 7 main: assemble all 31 moments × 4 windows
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

    # SIPP within-job wage-change hazards (Stage 6b). The CSV carries a row only
    # for windows with their own SIPP data (own-window measurement, no cross-window
    # borrowing). A window with no row ⇒ wchg_rate_j left NaN, which holds the
    # moment out of the SMM objective for that window.
    sipp_path = joinpath(DERIVED_DIR, "sipp_wchg_rates.csv")
    sipp_wchg = isfile(sipp_path) ? CSV.read(sipp_path, DataFrame) : DataFrame()
    if !isempty(sipp_wchg) && hasproperty(sipp_wchg, :window)
        sipp_wchg.window = Symbol.(sipp_wchg.window)
    end

    # SIPP skilled EE mobility (Stage 6b): ee_rate_S (poach hazard) and ee_step_S
    # (mean EE-move wage step). The rate target ee_rate_S is sourced from the J2J
    # value loaded above; the SIPP EE rate is retained only for the printed
    # comparison below. The wage step ee_step_S is sourced from SIPP.
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

        # A. Stock moments — ur_total, ur_U, ur_S, skilled_share, training_share,
        #    and the skilled long-term-unemployment share ltu_share_S.
        cps_w = filter(row -> row.window == wname, cps_basic_m)
        if nrow(cps_w) > 0
            merge!(moments, _compute_stock_moments(cps_w))
            merge!(moments, _compute_ltu_share(cps_w))
        else
            for k in (:ur_total, :ur_U, :ur_S, :skilled_share, :training_share,
                      :ltu_share_S)
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

        # C. Skilled EE moments. The rate target ee_rate_S is the Census J2J E4
        #    monthly employer-to-employer hazard (Stage 5); the wage step
        #    ee_step_S is the SIPP mean log-wage jump on a main-job change
        #    (Stage 6b). The two moments therefore come from different
        #    instruments — a Census administrative flow versus a SIPP household
        #    panel — with different EE definitions: the rate from the
        #    purpose-built EE-flow series, the step from the only source that
        #    carries a wage change. The SIPP EE rate is retained here only for
        #    the printed SIPP-vs-J2J comparison. ee_rate_S is already a monthly
        #    hazard (j2j.jl converts the quarterly rate via 1−(1−ee_q)^(1/3)), so
        #    it is EXCLUDED from the Section-F frequency-consistency transform
        #    below — applying −log(1−p) would re-convert an object that is
        #    already a hazard.
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

        # C'. SIPP within-job wage-change hazards (Stage 6b, already converted
        #     to the model's monthly hazard −log(1−p) in sipp.jl). Stored as
        #     the hazard and therefore NOT touched by the frequency-consistency
        #     block below — see the note there. NaN where the window has no SIPP.
        #     wchg_rate_U/S is the SHIPPED construction: the BBG break-filtered
        #     hazard on the classic FC windows and the raw earnings-based hazard
        #     on the redesign COVID windows (sipp.jl §make_sipp_wchg). The raw
        #     earnings value for every window is also on the CSV (wchg_rate_*_raw)
        #     for reporting; the target reads the shipped column below.
        sipp_w = isempty(sipp_wchg) ? nothing : filter(r -> r.window == wname, sipp_wchg)
        if !isnothing(sipp_w) && nrow(sipp_w) > 0
            moments[:wchg_rate_U] = sipp_w.wchg_rate_U[1]
            moments[:wchg_rate_S] = sipp_w.wchg_rate_S[1]
        else
            moments[:wchg_rate_U] = NaN
            moments[:wchg_rate_S] = NaN
        end

        # D. Wage moments (ASEC, per-survey-year then average), including the
        #    cross-market overlap pair overlap_UgtS / overlap_SltU.
        asec_w = filter(r -> r.window == wname, cps_asec_m)
        if nrow(asec_w) > 0
            merge!(moments, _compute_wage_moments_per_year(asec_w))
        else
            for k in (:mean_wage_U, :mean_wage_S, :emp_var_U, :emp_cm3_U,
                      :emp_var_S, :emp_cm3_S, :p25_wage_U, :p25_wage_S,
                      :p50_wage_U, :p50_wage_S, :p75_wage_U, :p75_wage_S, :wage_premium,
                      :overlap_UgtS, :overlap_SltU)
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

        # F. Frequency consistency — convert transition PROBABILITIES to
        # continuous-time monthly HAZARDS so they match the model objects.
        # The model reports rates as Poisson hazards (f = μθ^{1−η},
        # sep = λ·G(p*), ee = κ·(1−Γ)); the data moments above are monthly
        # transition probabilities. For a monthly probability p the hazard is
        # h = −log(1 − p). Negligible for small rates (separations / EE);
        # ~10–25% for job finding, where p is large. Stocks (ur_*, shares),
        # ratios (theta_*) and wage moments are NOT rates and are left as-is.
        # ee_rate_S, wchg_rate_U and wchg_rate_S are absent here: wchg_rate_U/S
        # already arrive as the −log(1−p) hazard from sipp.jl and ee_rate_S is
        # already a monthly hazard from j2j.jl (quarterly rate converted via
        # 1−(1−ee_q)^(1/3)), so re-converting any of the three would apply a
        # hazard transform twice.
        # ee_step_S is a wage LEVEL, not a rate, and is never transformed.
        for k in (:jfr_U, :jfr_S, :sep_rate_U, :sep_rate_S)
            p = get(moments, k, NaN)
            if isfinite(p) && 0.0 <= p < 1.0
                moments[k] = -log(1.0 - p)
            end
        end

        # Build moment DataFrame in canonical MOMENT_NAMES order
        moment_df = DataFrame(moment=String[], value=Float64[])
        for mname in MOMENT_NAMES
            push!(moment_df, (string(mname), get(moments, mname, NaN)))
        end

        # training_share is taken from NSC directly (attrition-adjusted), not
        # from the CPS SCHLCOLL count: the CPS universe was age-capped at 24
        # before 2013, so its level is not comparable across the FC and COVID
        # window pairs. See nsc.jl::compute_training_share_target.
        #
        # The substitution happens BEFORE the print and before all_moments, so
        # that one number appears under the name `training_share` everywhere —
        # in the trace, in moments_*.csv, and in every Stage 11 diagnostic. An
        # earlier version kept the raw CPS count in memory "for diagnostics";
        # the result was a validation table and a stationary-identity gap
        # computed on a quantity the SMM never sees. What the survey itself says
        # is already reported by Stage 3 (enrollment_rate_by_age,
        # cps_vs_nsc_enrollment.csv), under its own name.
        ts_nsc = _load_training_share_target(wname).target
        for r in eachrow(moment_df)
            r.moment == "training_share" && (r.value = ts_nsc)
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

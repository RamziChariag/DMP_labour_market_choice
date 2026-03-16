############################################################
# make_moments.jl — Compute all 24 moments × 4 windows
#
# Inputs (all from data/derived/):
#   cps_basic_clean.arrow    — stocks, training share
#   cps_asec_clean.arrow     — wage moments
#   transitions.arrow        — transition rates
#   jolts_clean.arrow        — vacancies (for tightness)
#
# Outputs (to data/derived/):
#   moments_{window}.csv     — 24-row moment table per window
#   sigma_{window}.csv       — 24×24 variance-covariance matrix
#
# Moment groups (matching moments.jl in the SMM code)
# ────────────────────────────────────────────────────
#   Stocks (9):      ur_total, ur_U, ur_S, skilled_share,
#                    training_share, emp_var_U, emp_cm3_U,
#                    emp_var_S, emp_cm3_S
#   Transitions (6): jfr_U, sep_rate_U, jfr_S, sep_rate_S,
#                    ee_rate_S, training_rate
#   Wages (7):       mean_wage_U, mean_wage_S, p50_wage_U,
#                    p50_wage_S, wage_premium, wage_sd_U, wage_sd_S
#   Tightness (3):   theta_U, theta_S   (+ theta_ratio computed
#                    downstream if needed)
############################################################


"""
    make_moments() → Dict{Symbol, DataFrame}

Compute the 25 targeted moments for each estimation window.
Returns a Dict mapping window name → DataFrame of moments.
Also saves moments_{window}.csv and sigma_{window}.csv.
"""
function make_moments()

    @info "make_moments: assembling all 24 moments × 4 windows..."

    # ── Load derived data ────────────────────────────────────────────────
    cps_basic = _load_arrow("cps_basic_clean.arrow")
    cps_asec  = _load_arrow("cps_asec_clean.arrow")
    trans     = _load_arrow("transitions.arrow")
    jolts     = _load_arrow("jolts_clean.arrow")

    all_moments = Dict{Symbol, DataFrame}()

    for (wname, wdef) in WINDOWS
        @info "  Window: $(wdef.label) ($wname)"
        moments = Dict{Symbol, Float64}()

        # ────────────────────────────────────────────────────────────────
        # Block 1: Labour-market stocks (from CPS Basic)
        # ────────────────────────────────────────────────────────────────
        cps_w = filter(row -> row.window == wname, cps_basic)

        if nrow(cps_w) > 0
            # Monthly averages of weighted rates
            stock_moments = _compute_stock_moments(cps_w)
            merge!(moments, stock_moments)
        else
            @warn "    No CPS Basic data for window $wname"
            for k in [:ur_total, :ur_U, :ur_S, :skilled_share, :training_share]
                moments[k] = NaN
            end
        end

        # ────────────────────────────────────────────────────────────────
        # Block 2: Transition rates (from matched CPS)
        # ────────────────────────────────────────────────────────────────
        trans_w = filter(row -> row.window == wname, trans)

        if nrow(trans_w) > 0
            _fill_transition_moments!(moments, trans_w)
        else
            @warn "    No transition data for window $wname"
            for k in [:jfr_U, :sep_rate_U, :jfr_S, :sep_rate_S, :ee_rate_S, :training_rate]
                moments[k] = NaN
            end
        end

        # Training rate (cross-sectional proxy)
        if haskey(moments, :training_share) && haskey(moments, :ur_U)
            ts = moments[:training_share]
            ur = moments[:ur_U]
            moments[:training_rate] = (ts + ur) > 0 ? ts / (ts + ur) : NaN
        end

        # ────────────────────────────────────────────────────────────────
        # Block 3: Wages (from CPS ASEC)
        # ────────────────────────────────────────────────────────────────
        asec_w = filter(row -> row.window == wname, cps_asec)

        if nrow(asec_w) > 0
            wage_moments = _compute_wage_moments(asec_w)
            merge!(moments, wage_moments)
        else
            @warn "    No ASEC data for window $wname"
            for k in [:mean_wage_U, :mean_wage_S, :p50_wage_U, :p50_wage_S,
                       :wage_premium, :wage_sd_U, :wage_sd_S,
                       :emp_var_U, :emp_cm3_U, :emp_var_S, :emp_cm3_S]
                moments[k] = NaN
            end
        end

        # ────────────────────────────────────────────────────────────────
        # Block 4: Tightness (from JOLTS + CPS Basic)
        # ────────────────────────────────────────────────────────────────
        jolts_w = filter(row -> row.window == wname, jolts)

        if nrow(jolts_w) > 0 && nrow(cps_w) > 0
            tightness = _compute_tightness(jolts_w, cps_w)
            merge!(moments, tightness)
        else
            @warn "    No JOLTS data for window $wname"
            moments[:theta_U] = NaN
            moments[:theta_S] = NaN
        end

        # ────────────────────────────────────────────────────────────────
        # Assemble into DataFrame and save
        # ────────────────────────────────────────────────────────────────
        moment_df = DataFrame(
            moment = String[],
            value  = Float64[],
            se     = Float64[],   # placeholder — filled by influence functions
            weight = Float64[],   # 1/se^2 (diagonal weight matrix)
        )
        for mname in MOMENT_NAMES
            val = get(moments, mname, NaN)
            # Placeholder SE = 10% of value (to be replaced by IF-based SE)
            se  = isfinite(val) && val != 0.0 ? abs(val) * 0.10 : 1.0
            wt  = isfinite(se) && se > 0.0 ? 1.0 / se^2 : 0.0
            push!(moment_df, (string(mname), val, se, wt))
        end

        # Save moment values
        CSV.write(joinpath(DERIVED_DIR, "moments_$(wname).csv"), moment_df)

        # Placeholder variance-covariance matrix (diagonal)
        K = length(MOMENT_NAMES)
        sigma = zeros(K, K)
        for (i, mname) in enumerate(MOMENT_NAMES)
            se = moment_df.se[i]
            sigma[i, i] = se^2
        end
        sigma_df = DataFrame(sigma, [string(m) for m in MOMENT_NAMES])
        CSV.write(joinpath(DERIVED_DIR, "sigma_$(wname).csv"), sigma_df)

        all_moments[wname] = moment_df

        # Print summary
        @info "    Moments for $wname:"
        for row in eachrow(moment_df)
            @info "      $(rpad(row.moment, 18))  $(round(row.value; digits=5))"
        end
    end

    @info "  All moment files saved to $(DERIVED_DIR)"
    return all_moments
end


# ====================================================================
# Internal helpers
# ====================================================================

function _load_arrow(filename::String) :: DataFrame
    path = joinpath(DERIVED_DIR, filename)
    if !isfile(path)
        @warn "$filename not found in derived — returning empty DataFrame"
        return DataFrame()
    end
    return DataFrame(Arrow.Table(path))
end


"""
Compute stock moments from CPS Basic within one window.
"""
function _compute_stock_moments(cps_w::DataFrame) :: Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()

    # Group by (year, month) and compute weighted monthly rates,
    # then average across months.
    monthly = NamedTuple[]
    for gk in groupby(cps_w, [:YEAR, :MONTH])
        g = DataFrame(gk)
        w = Float64.(g.WTFINL)
        sw = sum(w)
        sw <= 0 && continue

        # Unemployment rates
        n_unemp     = sum(w[g.unemployed])
        n_unemp_U   = sum(w[g.unemployed .& .!g.skilled])
        n_unemp_S   = sum(w[g.unemployed .& g.skilled])
        n_lf        = sw
        n_lf_U      = sum(w[.!g.skilled])
        n_lf_S      = sum(w[g.skilled])

        ur_total = n_lf > 0     ? n_unemp   / n_lf   : NaN
        ur_U     = n_lf_U > 0   ? n_unemp_U / n_lf_U : NaN
        ur_S     = n_lf_S > 0   ? n_unemp_S / n_lf_S : NaN

        # Skilled share of labour force
        skilled_share = n_lf > 0 ? n_lf_S / n_lf : NaN

        # Training share
        if hasproperty(g, :in_training)
            n_training = sum(w[g.in_training])
            training_share = n_lf > 0 ? n_training / n_lf : NaN
        else
            training_share = NaN
        end

        push!(monthly, (ur_total = ur_total, ur_U = ur_U, ur_S = ur_S,
                         skilled_share = skilled_share,
                         training_share = training_share))
    end

    if !isempty(monthly)
        mdf = DataFrame(monthly)
        moments[:ur_total]       = mean(filter(isfinite, mdf.ur_total))
        moments[:ur_U]           = mean(filter(isfinite, mdf.ur_U))
        moments[:ur_S]           = mean(filter(isfinite, mdf.ur_S))
        moments[:skilled_share]  = mean(filter(isfinite, mdf.skilled_share))
        moments[:training_share] = mean(filter(isfinite, mdf.training_share))
    end

    return moments
end


"""
Fill transition-rate moments from the window-averaged transition table.
"""
function _fill_transition_moments!(moments::Dict{Symbol, Float64},
                                   trans_w::DataFrame)
    for row in eachrow(trans_w)
        if row.skilled
            moments[:jfr_S]      = row.mean_jfr
            moments[:sep_rate_S] = row.mean_sep
        else
            moments[:jfr_U]      = row.mean_jfr
            moments[:sep_rate_U] = row.mean_sep
        end
    end

    # ee_rate_S: placeholder — requires employer-change identification
    # in the matched CPS (CPS redesign Jan 2004+).
    # Set to NaN; will be computed from detailed matched data if available.
    moments[:ee_rate_S] = get(moments, :ee_rate_S, NaN)
end


"""
Compute wage moments from CPS ASEC within one window.
"""
function _compute_wage_moments(asec_w::DataFrame) :: Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()

    # Split by skill
    unskilled = filter(row -> !row.skilled, asec_w)
    skilled   = filter(row ->  row.skilled, asec_w)

    # Use normalised wages
    if nrow(unskilled) > 0
        wu = Float64.(unskilled.wage_norm)
        wt = Float64.(unskilled.ASECWT)
        moments[:mean_wage_U] = wmean(wu, wt)
        moments[:p50_wage_U]  = wmedian(wu, wt)
        moments[:wage_sd_U]   = wsd(wu, wt)
        moments[:emp_var_U]   = wvar(wu, wt)
        moments[:emp_cm3_U]   = wcm3(wu, wt)
    end

    if nrow(skilled) > 0
        ws = Float64.(skilled.wage_norm)
        wt = Float64.(skilled.ASECWT)
        moments[:mean_wage_S] = wmean(ws, wt)
        moments[:p50_wage_S]  = wmedian(ws, wt)
        moments[:wage_sd_S]   = wsd(ws, wt)
        moments[:emp_var_S]   = wvar(ws, wt)
        moments[:emp_cm3_S]   = wcm3(ws, wt)
    end

    # Log skill premium: E[log w | S] - E[log w | U]
    # Invariant to normalisation (dividing by median cancels in logs
    # only for ratios, but E[log(w/m)] = E[log w] - log m, so the
    # difference E[log w|S] - E[log w|U] is indeed invariant).
    if nrow(unskilled) > 0 && nrow(skilled) > 0
        log_wu = log.(max.(Float64.(unskilled.wage_norm), 1e-14))
        log_ws = log.(max.(Float64.(skilled.wage_norm), 1e-14))
        wt_u   = Float64.(unskilled.ASECWT)
        wt_s   = Float64.(skilled.ASECWT)
        moments[:wage_premium] = wmean(log_ws, wt_s) - wmean(log_wu, wt_u)
    end

    return moments
end


"""
Compute market tightness from JOLTS vacancies and CPS unemployment.
"""
function _compute_tightness(jolts_w::DataFrame, cps_w::DataFrame) :: Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()

    # Average monthly vacancies in window
    mean_V_U = mean(jolts_w.V_U)
    mean_V_S = mean(jolts_w.V_S)

    # Average monthly unemployment by skill from CPS Basic
    monthly_U = NamedTuple[]
    for gk in groupby(cps_w, [:YEAR, :MONTH])
        g = DataFrame(gk)
        w = Float64.(g.WTFINL)
        u_U = sum(w[g.unemployed .& .!g.skilled])
        u_S = sum(w[g.unemployed .& g.skilled])
        push!(monthly_U, (U_U = u_U, U_S = u_S))
    end
    mdf = DataFrame(monthly_U)
    mean_U_U = mean(mdf.U_U)
    mean_U_S = mean(mdf.U_S)

    # Tightness = V / U
    moments[:theta_U] = mean_U_U > 0 ? mean_V_U / mean_U_U : NaN
    moments[:theta_S] = mean_U_S > 0 ? mean_V_S / mean_U_S : NaN

    return moments
end

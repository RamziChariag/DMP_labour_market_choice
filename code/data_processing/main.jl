############################################################
# main.jl — Data Processing Pipeline Entry Point
#
# Coordinates loading and processing of CPS Basic, CPS ASEC,
# JOLTS, transitions, and J2J data. Produces cleaned datasets
# in derived/ that are consumed by smm/moments.jl.
#
# Usage:
#     julia --threads auto data_processing/main.jl
#
# Assumes raw data directories exist under data/raw/.
############################################################

using DataFrames, CSV, Arrow, Statistics, StatsBase, Printf, Dates
using HTTP, JSON3
using XLSX
using LinearAlgebra

# ============================================================
# Paths — adjust PROJECT_ROOT if running from a different location
# ============================================================

const PROJECT_ROOT      = joinpath(@__DIR__, "..")
const DATA_DIR          = joinpath(PROJECT_ROOT, "data")
const RAW_DIR           = joinpath(DATA_DIR, "raw")
const DERIVED_DIR       = joinpath(DATA_DIR, "derived")

const RAW_CPS_BASIC_DIR = joinpath(RAW_DIR, "cps_basic")
const RAW_CPS_ASEC_DIR  = joinpath(RAW_DIR, "cps_asec")
const RAW_JOLTS_DIR     = joinpath(RAW_DIR, "jolts")
const RAW_J2J_DIR       = joinpath(RAW_DIR, "j2j")
const RAW_NSC_DIR       = joinpath(RAW_DIR, "nsc")

mkpath(DERIVED_DIR)

# ============================================================
# Include processing scripts
# ============================================================

include("helpers.jl")
include("cps_basic.jl")
include("cps_asec.jl")
include("jolts.jl")
include("transitions.jl")
include("j2j.jl")

# ============================================================
# Stage 6: Compute ν (demographic turnover rate)
# ============================================================

function compute_nu()
    @info "Stage 6: compute_nu..."

    trans_monthly = DataFrame(Arrow.Table(joinpath(DERIVED_DIR, "transitions_monthly.arrow")))

    # ν from pre-FC baseline window (pooling skilled + unskilled)
    base_data = filter(r -> r.window == :base_fc && isfinite(r.nu), trans_monthly)

    if isempty(base_data)
        @warn "  No base_fc data for ν — returning NaN"
        return NaN
    end

    # Pool across skill groups (weighted by number of pairs)
    nu_estimate = wmean(base_data.nu, Float64.(base_data.n_pairs))

    println("  ν (demographic turnover, monthly) = $(round(nu_estimate; digits=5))")
    @info "  Based on $(nrow(base_data)) month × skill observations"

    # Also compute per-window for diagnostics
    for wname in keys(WINDOWS)
        w_data = filter(r -> r.window == wname && isfinite(r.nu), trans_monthly)
        isempty(w_data) && continue
        nu_w = wmean(w_data.nu, Float64.(w_data.n_pairs))
        @printf("    %s: ν = %.5f (n=%d)\n", wname, nu_w, nrow(w_data))
    end

    CSV.write(joinpath(DERIVED_DIR, "nu_estimate.csv"),
              DataFrame(nu=nu_estimate, window="base_fc", n_obs=nrow(base_data)))
    return nu_estimate
end

# ============================================================
# Stage 7 helper: calibrate φ from NSC data
# ============================================================

function calibrate_phi()
    @info "Calibrating φ from NSC data..."

    nsc_files = filter(f -> endswith(f, ".xlsx"), readdir(RAW_NSC_DIR))
    isempty(nsc_files) && error("No .xlsx files found in $RAW_NSC_DIR")
    nsc_path = joinpath(RAW_NSC_DIR, first(nsc_files))

    @info "  Reading: $nsc_path"

    data = DataFrame(XLSX.readtable(nsc_path, "Enrollments"))
    rename!(data, string.(names(data)))

    # Filter to US-level rows for 4-year and 2-year institutions
    us_4yr = filter(r ->
        !ismissing(r["State or Region"]) &&
        !ismissing(r["Institution Sector"]) &&
        r["State or Region"] == "United States" &&
        r["Institution Sector"] == "All 4-year Institutions",
        data)

    us_2yr = filter(r ->
        !ismissing(r["State or Region"]) &&
        !ismissing(r["Institution Sector"]) &&
        r["State or Region"] == "United States" &&
        r["Institution Sector"] == "All 2-year Institutions",
        data)

    @assert nrow(us_4yr) == 1 "Expected 1 US 4-year row, got $(nrow(us_4yr))"
    @assert nrow(us_2yr) == 1 "Expected 1 US 2-year row, got $(nrow(us_2yr))"

    ipeds_cols_4yr = Float64[]
    ipeds_cols_2yr = Float64[]
    for col in names(data)
        sc = string(col)
        (occursin("IPEDS", sc) && occursin("Universe", sc)) || continue
        v4 = us_4yr[1, col]
        v2 = us_2yr[1, col]
        (ismissing(v4) || isnothing(v4) || ismissing(v2) || isnothing(v2)) && continue
        try
            push!(ipeds_cols_4yr, Float64(v4))
            push!(ipeds_cols_2yr, Float64(v2))
        catch
        end
    end

    isempty(ipeds_cols_4yr) &&
        error("No IPEDS Universe columns found — check column names in NSC file")
    length(ipeds_cols_4yr) != length(ipeds_cols_2yr) &&
        error("Mismatched column counts: 4yr=$(length(ipeds_cols_4yr)), 2yr=$(length(ipeds_cols_2yr))")

    E_4yr = mean(ipeds_cols_4yr)
    E_2yr = mean(ipeds_cols_2yr)

    println("  Average IPEDS Universe enrollment ($(length(ipeds_cols_4yr)) years):")
    println("    4-year: $(round(Int, E_4yr))")
    println("    2-year: $(round(Int, E_2yr))")

    # NCES median time-to-degree (months): 49 for bachelor's, 37 for associate's
    d_4yr = 49.0
    d_2yr = 37.0

    # φ = [E_4yr*(1/d_4yr) + E_2yr*(1/d_2yr)] / (E_4yr + E_2yr)
    phi = (E_4yr / d_4yr + E_2yr / d_2yr) / (E_4yr + E_2yr)

    println("  φ = $(round(phi; digits=6)) (monthly completion probability)")

    CSV.write(joinpath(DERIVED_DIR, "phi_estimate.csv"),
              DataFrame(phi=phi, E_4yr=E_4yr, E_2yr=E_2yr, d_4yr=d_4yr, d_2yr=d_2yr))
    return phi
end

# ============================================================
# Helpers for moment construction
# ============================================================

function _load_arrow(filename::String)::DataFrame
    path = joinpath(DERIVED_DIR, filename)
    !isfile(path) && (@warn "$filename not found"; return DataFrame())
    return DataFrame(Arrow.Table(path))
end

function _compute_stock_moments(cps_w::DataFrame)::Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()

    # Restrict to LF observations for all stock moments.
    lf_df = hasproperty(cps_w, :in_lf) ? filter(r -> r.in_lf, cps_w) : cps_w

    monthly = NamedTuple[]
    for gk in groupby(lf_df, [:YEAR, :MONTH])
        g = DataFrame(gk)
        w = Float64.(coalesce.(g.WTFINL, 0.0)); sw = sum(w)
        sw <= 0 && continue
        n_unemp   = sum(w[g.unemployed])
        n_unemp_U = sum(w[g.unemployed .& .!g.skilled])
        n_unemp_S = sum(w[g.unemployed .& g.skilled])
        n_lf = sw; n_lf_U = sum(w[.!g.skilled]); n_lf_S = sum(w[g.skilled])
        ur_total = n_lf > 0 ? n_unemp / n_lf : NaN
        ur_U = n_lf_U > 0 ? n_unemp_U / n_lf_U : NaN
        ur_S = n_lf_S > 0 ? n_unemp_S / n_lf_S : NaN
        skilled_share = n_lf > 0 ? n_lf_S / n_lf : NaN
        training_share = hasproperty(g, :in_training) ?
            (n_lf > 0 ? sum(w[coalesce.(g.in_training, false)]) / n_lf : NaN) : NaN
        push!(monthly, (ur_total=ur_total, ur_U=ur_U, ur_S=ur_S,
                         exp_ur_total = isfinite(ur_total) ? exp(ur_total) : NaN,
                         exp_ur_U     = isfinite(ur_U)     ? exp(ur_U)     : NaN,
                         exp_ur_S     = isfinite(ur_S)     ? exp(ur_S)     : NaN,
                         skilled_share=skilled_share, training_share=training_share))
    end
    if !isempty(monthly)
        mdf = DataFrame(monthly)
        moments[:ur_total]       = mean(filter(isfinite, mdf.ur_total))
        moments[:ur_U]           = mean(filter(isfinite, mdf.ur_U))
        moments[:ur_S]           = mean(filter(isfinite, mdf.ur_S))
        moments[:exp_ur_total]   = mean(filter(isfinite, mdf.exp_ur_total))
        moments[:exp_ur_U]       = mean(filter(isfinite, mdf.exp_ur_U))
        moments[:exp_ur_S]       = mean(filter(isfinite, mdf.exp_ur_S))
        moments[:skilled_share]  = mean(filter(isfinite, mdf.skilled_share))
        moments[:training_share] = mean(filter(isfinite, mdf.training_share))
    end
    return moments
end

function _fill_transition_moments!(moments::Dict{Symbol, Float64},
                                    trans_w::DataFrame, j2j_w)
    for row in eachrow(trans_w)
        if row.skilled
            moments[:jfr_S]      = row.mean_jfr
            moments[:sep_rate_S] = row.mean_sep
        else
            moments[:jfr_U]      = row.mean_jfr
            moments[:sep_rate_U] = row.mean_sep
        end
    end
    # EE rate from J2J (not from CPS)
    if !isnothing(j2j_w) && nrow(j2j_w) > 0
        moments[:ee_rate_S] = j2j_w.ee_rate_S[1]
    else
        moments[:ee_rate_S] = NaN
    end
end

function _compute_wage_moments(asec_w::DataFrame)::Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()
    unskilled = filter(row -> !row.skilled, asec_w)
    skilled   = filter(row ->  row.skilled, asec_w)

    if nrow(unskilled) > 0
        wu = Float64.(unskilled.wage_norm); wt = Float64.(unskilled.ASECWT)
        moments[:mean_wage_U] = wmean(wu, wt)
        moments[:p50_wage_U]  = wmedian(wu, wt)
        moments[:wage_sd_U]   = wsd(wu, wt)
        moments[:emp_var_U]   = wvar(wu, wt)
        moments[:emp_cm3_U]   = wcm3(wu, wt)
    end
    if nrow(skilled) > 0
        ws = Float64.(skilled.wage_norm); wt = Float64.(skilled.ASECWT)
        moments[:mean_wage_S] = wmean(ws, wt)
        moments[:p50_wage_S]  = wmedian(ws, wt)
        moments[:wage_sd_S]   = wsd(ws, wt)
        moments[:emp_var_S]   = wvar(ws, wt)
        moments[:emp_cm3_S]   = wcm3(ws, wt)
    end
    if nrow(unskilled) > 0 && nrow(skilled) > 0
        log_wu = log.(max.(Float64.(unskilled.wage_norm), 1e-14))
        log_ws = log.(max.(Float64.(skilled.wage_norm), 1e-14))
        moments[:wage_premium] = wmean(log_ws, Float64.(skilled.ASECWT)) -
                                 wmean(log_wu, Float64.(unskilled.ASECWT))
    end
    return moments
end

function _compute_tightness(jolts_w::DataFrame, cps_w::DataFrame)::Dict{Symbol, Float64}
    moments = Dict{Symbol, Float64}()
    mean_V_U = mean(jolts_w.V_U)
    mean_V_S = mean(jolts_w.V_S)

    monthly_U = NamedTuple[]
    for gk in groupby(cps_w, [:YEAR, :MONTH])
        g = DataFrame(gk); w = Float64.(coalesce.(g.WTFINL, 0.0))
        push!(monthly_U, (U_U=sum(w[g.unemployed .& .!g.skilled]),
                           U_S=sum(w[g.unemployed .& g.skilled])))
    end
    mdf = DataFrame(monthly_U)
    mean_U_U = mean(mdf.U_U); mean_U_S = mean(mdf.U_S)

    moments[:theta_U] = mean_U_U > 0 ? mean_V_U / mean_U_U : NaN
    moments[:theta_S] = mean_U_S > 0 ? mean_V_S / mean_U_S : NaN
    return moments
end

# ============================================================
# Stage 7: Assemble all 24 moments × 4 windows
# ============================================================

function make_moments()
    @info "Stage 7: assembling all 24 moments × 4 windows..."

    cps_basic_m = _load_arrow("cps_basic_clean.arrow")
    cps_asec_m  = _load_arrow("cps_asec_clean.arrow")
    trans       = _load_arrow("transitions.arrow")
    jolts_m     = _load_arrow("jolts_clean.arrow")

    # Load J2J EE rates
    j2j_path = joinpath(DERIVED_DIR, "j2j_ee_rates.csv")
    j2j_ee = isfile(j2j_path) ? CSV.read(j2j_path, DataFrame) : DataFrame()

    if !isempty(j2j_ee) && hasproperty(j2j_ee, :window)
        j2j_ee.window = Symbol.(j2j_ee.window)
    end

    all_moments = Dict{Symbol, DataFrame}()

    for (wname, wdef) in WINDOWS
        @info "  Window: $(wdef.label) ($wname)"
        moments = Dict{Symbol, Float64}()

        # Stock moments from CPS Basic
        cps_w = filter(row -> row.window == wname, cps_basic_m)
        if nrow(cps_w) > 0
            merge!(moments, _compute_stock_moments(cps_w))
        else
            for k in [:ur_total,:ur_U,:ur_S,:exp_ur_total,:exp_ur_U,:exp_ur_S,
                      :skilled_share,:training_share]
                moments[k] = NaN
            end
        end

        # Transition moments from CPS matched panels + J2J
        trans_w = filter(row -> row.window == wname, trans)
        j2j_w = isempty(j2j_ee) ? nothing : filter(row -> row.window == wname, j2j_ee)
        if nrow(trans_w) > 0
            _fill_transition_moments!(moments, trans_w, j2j_w)
        else
            for k in [:jfr_U,:sep_rate_U,:jfr_S,:sep_rate_S,:ee_rate_S]
                moments[k] = NaN
            end
        end

        # Training rate
        if haskey(moments, :training_share) && haskey(moments, :ur_U)
            ts = moments[:training_share]; ur = moments[:ur_U]
            moments[:training_rate] = (ts + ur) > 0 ? ts / (ts + ur) : NaN
        else
            moments[:training_rate] = NaN
        end

        # Wage moments from CPS ASEC
        asec_w = filter(row -> row.window == wname, cps_asec_m)
        if nrow(asec_w) > 0
            merge!(moments, _compute_wage_moments(asec_w))
        else
            for k in [:mean_wage_U,:mean_wage_S,:p50_wage_U,:p50_wage_S,
                      :wage_premium,:wage_sd_U,:wage_sd_S,:emp_var_U,
                      :emp_cm3_U,:emp_var_S,:emp_cm3_S]
                moments[k] = NaN
            end
        end

        # Tightness from JOLTS + CPS
        jolts_w = filter(row -> row.window == wname, jolts_m)
        if nrow(jolts_w) > 0 && nrow(cps_w) > 0
            merge!(moments, _compute_tightness(jolts_w, cps_w))
        else
            moments[:theta_U] = NaN; moments[:theta_S] = NaN
        end

        # Build moment DataFrame
        moment_df = DataFrame(moment=String[], value=Float64[])
        for mname in MOMENT_NAMES
            push!(moment_df, (string(mname), get(moments, mname, NaN)))
        end
        CSV.write(joinpath(DERIVED_DIR, "moments_$(wname).csv"), moment_df)
        all_moments[wname] = moment_df
    end

    @info "  All moment files saved to $(DERIVED_DIR)"
    return all_moments
end

# ============================================================
# Stage 8: Influence functions and variance-covariance matrix
# ============================================================

function compute_influence_functions_and_sigma()
    @info "Stage 8: influence functions and Σ̂..."
    cps_basic_m = _load_arrow("cps_basic_clean.arrow")
    cps_asec_m  = _load_arrow("cps_asec_clean.arrow")
    K = length(MOMENT_NAMES)
    all_sigma = Dict{Symbol, Matrix{Float64}}()
    all_W     = Dict{Symbol, Matrix{Float64}}()

    for (wname, wdef) in WINDOWS
        @info "  Window: $(wdef.label) ($wname)"

        # Load the computed moments for this window
        mpath = joinpath(DERIVED_DIR, "moments_$(wname).csv")
        !isfile(mpath) && (@warn "Moments not found for $wname"; continue)
        mdf = CSV.read(mpath, DataFrame)
        moment_vals = Dict(Symbol(row.moment) => row.value for row in eachrow(mdf))

        # ── Block 1: CPS Basic stock moments ─────────────────────
        cps_w = filter(row -> row.window == wname, cps_basic_m)
        lf_cps_w = hasproperty(cps_w, :in_lf) ? filter(r -> r.in_lf, cps_w) : cps_w

        basic_moments = [:ur_total, :ur_U, :ur_S, :exp_ur_total, :exp_ur_U, :exp_ur_S,
                         :skilled_share, :training_share]
        monthly_stock = NamedTuple[]

        for gk in groupby(lf_cps_w, [:YEAR, :MONTH])
            g = DataFrame(gk)
            w = Float64.(coalesce.(g.WTFINL, 0.0))
            sw = sum(w)
            sw <= 0 && continue
            n_u   = sum(w[g.unemployed])
            n_u_U = sum(w[g.unemployed .& .!g.skilled])
            n_u_S = sum(w[g.unemployed .& g.skilled])
            n_lf  = sw
            n_lf_U = sum(w[.!g.skilled])
            n_lf_S = sum(w[g.skilled])
            ts = hasproperty(g, :in_training) ? sum(w[g.in_training]) / n_lf : NaN
            ur_t  = n_u/n_lf
            ur_U_ = n_lf_U > 0 ? n_u_U/n_lf_U : NaN
            ur_S_ = n_lf_S > 0 ? n_u_S/n_lf_S : NaN
            push!(monthly_stock, (
                ur_total      = ur_t,
                ur_U          = ur_U_,
                ur_S          = ur_S_,
                exp_ur_total  = isfinite(ur_t)  ? exp(ur_t)  : NaN,
                exp_ur_U      = isfinite(ur_U_) ? exp(ur_U_) : NaN,
                exp_ur_S      = isfinite(ur_S_) ? exp(ur_S_) : NaN,
                skilled_share = n_lf_S/n_lf,
                training_share = ts,
            ))
        end

        ms_df = DataFrame(monthly_stock)
        T_basic = nrow(ms_df)

        psi_basic = zeros(T_basic, length(basic_moments))
        for (j, mname) in enumerate(basic_moments)
            m_bar = get(moment_vals, mname, NaN)
            isfinite(m_bar) || continue
            vals = ms_df[!, mname]
            psi_basic[:, j] = vals .- m_bar
        end
        Sigma_basic = (psi_basic' * psi_basic) / T_basic

        # ── Block 2: CPS Basic transition moments ────────────────
        trans_monthly = _load_arrow("transitions_monthly.arrow")
        trans_w = filter(r -> r.window == wname, trans_monthly)
        trans_moments = [:jfr_U, :sep_rate_U, :jfr_S, :sep_rate_S]

        T_trans_months = length(unique(collect(zip(trans_w.year, trans_w.month))))
        psi_trans = zeros(T_trans_months, 4)
        month_keys = unique(collect(zip(trans_w.year, trans_w.month)))

        for (t, (yr, mo)) in enumerate(month_keys)
            for (j, mname) in enumerate(trans_moments)
                sk = mname in (:jfr_S, :sep_rate_S)
                rate_col = mname in (:jfr_U, :jfr_S) ? :jfr : :sep
                row = filter(r -> r.year == yr && r.month == mo && r.skilled == sk, trans_w)
                nrow(row) == 0 && continue
                val = row[1, rate_col]
                m_bar = get(moment_vals, mname, NaN)
                (isfinite(val) && isfinite(m_bar)) || continue
                psi_trans[t, j] = val - m_bar
            end
        end
        Sigma_trans = T_trans_months > 0 ? (psi_trans' * psi_trans) / T_trans_months : zeros(4, 4)

        # ── Block 3: ASEC wage moments ───────────────────────────
        asec_w = filter(row -> row.window == wname, cps_asec_m)
        wage_moments = [:emp_var_U, :emp_cm3_U, :emp_var_S, :emp_cm3_S,
                        :mean_wage_U, :mean_wage_S, :p50_wage_U, :p50_wage_S,
                        :wage_premium, :wage_sd_U, :wage_sd_S]
        n_wage = length(wage_moments)
        N_asec = nrow(asec_w)
        psi_wage = zeros(N_asec, n_wage)

        if N_asec > 0
            wu_all = Float64.(asec_w.wage_norm)
            wt_all = Float64.(asec_w.ASECWT)
            sk_all = asec_w.skilled

            for i in 1:N_asec
                y = wu_all[i]
                w_i = wt_all[i]
                sk = sk_all[i]
                j = 0

                # emp_var_U
                j += 1
                if !sk
                    m = get(moment_vals, :mean_wage_U, NaN)
                    v = get(moment_vals, :emp_var_U, NaN)
                    sw_u = sum(wt_all[.!sk_all])
                    (isfinite(m) && isfinite(v) && sw_u > 0) &&
                        (psi_wage[i, j] = w_i * ((y - m)^2 - v) / sw_u)
                end

                # emp_cm3_U
                j += 1
                if !sk
                    m = get(moment_vals, :mean_wage_U, NaN)
                    cm3 = get(moment_vals, :emp_cm3_U, NaN)
                    v = get(moment_vals, :emp_var_U, NaN)
                    sw_u = sum(wt_all[.!sk_all])
                    (isfinite(m) && isfinite(cm3) && isfinite(v) && sw_u > 0) &&
                        (psi_wage[i, j] = w_i * ((y - m)^3 - cm3 - 3*v*(y - m)) / sw_u)
                end

                # emp_var_S
                j += 1
                if sk
                    m = get(moment_vals, :mean_wage_S, NaN)
                    v = get(moment_vals, :emp_var_S, NaN)
                    sw_s = sum(wt_all[sk_all])
                    (isfinite(m) && isfinite(v) && sw_s > 0) &&
                        (psi_wage[i, j] = w_i * ((y - m)^2 - v) / sw_s)
                end

                # emp_cm3_S
                j += 1
                if sk
                    m = get(moment_vals, :mean_wage_S, NaN)
                    cm3 = get(moment_vals, :emp_cm3_S, NaN)
                    v = get(moment_vals, :emp_var_S, NaN)
                    sw_s = sum(wt_all[sk_all])
                    (isfinite(m) && isfinite(cm3) && isfinite(v) && sw_s > 0) &&
                        (psi_wage[i, j] = w_i * ((y - m)^3 - cm3 - 3*v*(y - m)) / sw_s)
                end

                # mean_wage_U
                j += 1
                if !sk
                    m = get(moment_vals, :mean_wage_U, NaN)
                    sw_u = sum(wt_all[.!sk_all])
                    (isfinite(m) && sw_u > 0) &&
                        (psi_wage[i, j] = w_i * (y - m) / sw_u)
                end

                # mean_wage_S
                j += 1
                if sk
                    m = get(moment_vals, :mean_wage_S, NaN)
                    sw_s = sum(wt_all[sk_all])
                    (isfinite(m) && sw_s > 0) &&
                        (psi_wage[i, j] = w_i * (y - m) / sw_s)
                end

                # p50_wage_U
                j += 1
                if !sk
                    med = get(moment_vals, :p50_wage_U, NaN)
                    if isfinite(med)
                        f_med = kde_at_point(wu_all[.!sk_all], wt_all[.!sk_all], med)
                        f_med > 0 && (psi_wage[i, j] = (Float64(y <= med) - 0.5) / (2 * f_med))
                    end
                end

                # p50_wage_S
                j += 1
                if sk
                    med = get(moment_vals, :p50_wage_S, NaN)
                    if isfinite(med)
                        f_med = kde_at_point(wu_all[sk_all], wt_all[sk_all], med)
                        f_med > 0 && (psi_wage[i, j] = (Float64(y <= med) - 0.5) / (2 * f_med))
                    end
                end

                # wage_premium (log skill premium)
                j += 1
                m_s = get(moment_vals, :mean_wage_S, NaN)
                m_u = get(moment_vals, :mean_wage_U, NaN)
                if isfinite(m_s) && isfinite(m_u) && m_s > 0 && m_u > 0
                    if sk
                        sw_s = sum(wt_all[sk_all])
                        sw_s > 0 && (psi_wage[i, j] = w_i * (log(max(y, 1e-14)) - log(m_s)) / sw_s)
                    else
                        sw_u = sum(wt_all[.!sk_all])
                        sw_u > 0 && (psi_wage[i, j] = -w_i * (log(max(y, 1e-14)) - log(m_u)) / sw_u)
                    end
                end

                # wage_sd_U
                j += 1
                if !sk
                    v = get(moment_vals, :emp_var_U, NaN)
                    sd = get(moment_vals, :wage_sd_U, NaN)
                    m = get(moment_vals, :mean_wage_U, NaN)
                    sw_u = sum(wt_all[.!sk_all])
                    (isfinite(v) && isfinite(sd) && sd > 0 && isfinite(m) && sw_u > 0) &&
                        (psi_wage[i, j] = w_i * ((y - m)^2 - v) / (2 * sd * sw_u))
                end

                # wage_sd_S
                j += 1
                if sk
                    v = get(moment_vals, :emp_var_S, NaN)
                    sd = get(moment_vals, :wage_sd_S, NaN)
                    m = get(moment_vals, :mean_wage_S, NaN)
                    sw_s = sum(wt_all[sk_all])
                    (isfinite(v) && isfinite(sd) && sd > 0 && isfinite(m) && sw_s > 0) &&
                        (psi_wage[i, j] = w_i * ((y - m)^2 - v) / (2 * sd * sw_s))
                end
            end
        end
        Sigma_wage = N_asec > 0 ? (psi_wage' * psi_wage) / N_asec : zeros(n_wage, n_wage)

        # ── Assemble full 24×24 block-diagonal Σ̂ ─────────────────
        Sigma = zeros(K, K)

        idx = Dict(m => i for (i, m) in enumerate(MOMENT_NAMES))

        # Block 1: stock moments (8×8 — ur levels + exp_ur + skilled/training shares)
        for (i, mi) in enumerate(basic_moments), (j, mj) in enumerate(basic_moments)
            Sigma[idx[mi], idx[mj]] = Sigma_basic[i, j]
        end

        # Block 2: transition moments (4×4)
        for (i, mi) in enumerate(trans_moments), (j, mj) in enumerate(trans_moments)
            Sigma[idx[mi], idx[mj]] = Sigma_trans[i, j]
        end

        # Block 3: wage moments (11×11)
        for (i, mi) in enumerate(wage_moments), (j, mj) in enumerate(wage_moments)
            Sigma[idx[mi], idx[mj]] = Sigma_wage[i, j]
        end

        # Remaining moments: plug-in variance estimate (diagonal only)
        plug_in_moments = [:ee_rate_S, :theta_U, :theta_S]
        for mname in plug_in_moments
            i = idx[mname]
            val = get(moment_vals, mname, NaN)
            se = isfinite(val) && abs(val) > 0 ? (0.10 * abs(val))^2 : 1.0
            Sigma[i, i] = se
        end
        all_sigma[wname] = Sigma

        # ── SMM weight matrix ─────────────────────────────────────
        diag_sigma = [Sigma[i,i] for i in 1:K]
        W_diag = diagm([d > 0 ? 1.0/d : 0.0 for d in diag_sigma])

        cond_num = cond(Sigma)
        println("    Σ̂ condition number: $(round(cond_num; sigdigits=2))")
        if cond_num < 1e8 && all(diag_sigma .> 0)
            W_optimal = inv(Sigma)
            @info "    Using optimal weight matrix (Σ̂⁻¹)"
            all_W[wname] = W_optimal
        else
            println("    Σ̂ ill-conditioned (κ=$(round(cond_num; sigdigits=2))) — using diagonal weights")
            all_W[wname] = W_diag
        end

        # Save
        sigma_df = DataFrame(Sigma, [string(m) for m in MOMENT_NAMES])
        CSV.write(joinpath(DERIVED_DIR, "sigma_$(wname).csv"), sigma_df)
        W_df = DataFrame(all_W[wname], [string(m) for m in MOMENT_NAMES])
        CSV.write(joinpath(DERIVED_DIR, "W_$(wname).csv"), W_df)
    end

    @info "  All Σ̂ and W matrices saved"
    return all_sigma, all_W
end

# ============================================================
# Main pipeline
# ============================================================

function main()
    @info "Starting data processing pipeline..."
    @info "  PROJECT_ROOT = $PROJECT_ROOT"
    @info "  DERIVED_DIR  = $DERIVED_DIR"

    # Stage 1: Clean CPS Basic Monthly
    cps_basic = clean_cps_basic()

    # Stage 2: Clean CPS ASEC
    cps_asec = clean_cps_asec()

    # Stage 3: Download and clean JOLTS
    jolts = clean_jolts()

    # Stage 4: Build transition rates
    transitions = make_transitions()

    # Stage 5: Import J2J EE rates
    j2j_ee = import_j2j_ee_rates()

    # Stage 6: Compute ν
    nu_hat = compute_nu()

    # Stage 7: Calibrate φ and compute all moments
    phi_hat = calibrate_phi()
    all_moments = make_moments()

    # Stage 8: Influence functions and variance-covariance matrix
    all_sigma, all_W = compute_influence_functions_and_sigma()

    @info "Data processing pipeline complete!"
    @info "  All derived files saved to: $DERIVED_DIR"
    @info "  Next step: run SMM estimation"

    return (cps_basic=cps_basic, cps_asec=cps_asec, jolts=jolts,
            transitions=transitions, j2j_ee=j2j_ee,
            nu=nu_hat, phi=phi_hat,
            moments=all_moments, sigma=all_sigma, W=all_W)
end

# ============================================================
# Entry point
# ============================================================

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
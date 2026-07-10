############################################################
# data_processing/sigma.jl
#
# Combine step (Stage 8) — the full 26×26 influence-function variance–
# covariance matrix Σ̂ for every window, used to build the SMM weight
# matrix. The training_share row/col of the SAVED Σ̂ carries the NSC κ_w
# adjustment (off-diagonals × κ, diagonal × κ²); the in-memory return
# stays raw for Stage 9 diagnostics. REGULARIZATION_ALPHA comes from setup.jl.
#
# Reads:  moments_{window}.csv, cps_basic_clean.arrow, cps_asec_clean.arrow, jolts_clean.arrow, transitions_monthly.arrow, data/raw/j2j/*.csv, training_share_scale.csv
# Writes: sigma_{window}.csv, moment_scales_{window}.csv
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages and path consts come from data_processing_main.jl.
############################################################

TARGET_KAPPA = 1e8
SHRINK_TOL   = 1e-6
SHRINK_ITERS = 60


function shrink_offdiag(S::AbstractMatrix, ρ::Real)
    M = Matrix(S)
    @inbounds for j in axes(M, 2), i in axes(M, 1)
        if i != j
            M[i, j] *= ρ
        end
    end
    return Symmetric(M)
end

# ─────────────────────────────────────────────────────────────────────────────
# Stage 8: full K×K influence-function matrix Σ̂ for every window.
# ─────────────────────────────────────────────────────────────────────────────
function compute_influence_functions_and_sigma_full()
    @info "Stage 8: influence functions and Σ̂ (full $(length(MOMENT_NAMES))×$(length(MOMENT_NAMES)))..."

    K   = length(MOMENT_NAMES)
    idx = Dict(m => i for (i, m) in enumerate(MOMENT_NAMES))

    cps_basic_m   = _load_arrow("cps_basic_clean.arrow")
    cps_asec_m    = _load_arrow("cps_asec_clean.arrow")
    jolts_m       = _load_arrow("jolts_clean.arrow")
    trans_monthly = _load_arrow("transitions_monthly.arrow")

    for df in (cps_basic_m, cps_asec_m, jolts_m, trans_monthly)
        hasproperty(df, :window) && (df.window = Symbol.(df.window))
    end

    all_sigma = Dict{Symbol, Matrix{Float64}}()

    quarter_to_months    = Dict(1 => [1,2,3], 2 => [4,5,6], 3 => [7,8,9], 4 => [10,11,12])
    month_to_quarter     = Dict(m => q for (q, ms) in quarter_to_months for m in ms)
    quarter_to_mid_month = Dict(q => ms[2] for (q, ms) in quarter_to_months)

    j2j_raw_path = joinpath(RAW_J2J_DIR, first(filter(f -> endswith(f, ".csv"), readdir(RAW_J2J_DIR))))
    j2j_raw = CSV.read(j2j_raw_path, DataFrame; types=Dict(
        :EEHire => Float64, :MainB => Float64,
        :seasonadj => String, :geo_level => String, :industry => String,
        :sex => String, :agegrp => String, :race => String,
        :ethnicity => String, :education => String, :firmage => String,
        :firmsize => String
    ))

    j2j_nat = filter(row ->
        row.seasonadj == "S" && row.geo_level == "N" && row.industry == "00" &&
        row.sex == "0" && row.agegrp == "A00" && row.race == "A0" &&
        row.ethnicity == "A0" && row.firmage == "0" && row.firmsize == "0" &&
        row.education == "E4", j2j_raw)

    dropmissing!(j2j_nat, [:EEHire, :MainB])
    j2j_nat.ee_quarterly = j2j_nat.EEHire ./ j2j_nat.MainB
    j2j_nat.ee_monthly   = 1.0 .- (1.0 .- j2j_nat.ee_quarterly) .^ (1/3)
    j2j_nat.window       = Vector{Symbol}(undef, nrow(j2j_nat))
    for (i, row) in enumerate(eachrow(j2j_nat))
        j2j_nat.window[i] = assign_window(row.year, quarter_to_mid_month[row.quarter])
    end

    # ── Stack helpers ─────────────────────────────────────────────────────────
    function _add_to_stack!(
        stack::Dict{Tuple{Int,Int}, Vector{Float64}},
        key::Tuple{Int,Int},
        psi::AbstractVector{<:Real},
        K::Int
    )
        get!(()->zeros(K), stack, key) .+= psi
    end

    function _stack_to_matrix(stack::Dict{Tuple{Int,Int}, Vector{Float64}}, K::Int)
        keys_sorted = sort(collect(keys(stack)))
        Psi = zeros(length(keys_sorted), K)
        for (t, key) in enumerate(keys_sorted)
            Psi[t, :] .= stack[key]
        end
        return keys_sorted, Psi
    end

    # ── Per-window loop ───────────────────────────────────────────────────────
    for (wname, wdef) in WINDOWS
        @info "  Window: $(wdef.label) ($wname)"

        mpath = joinpath(DERIVED_DIR, "moments_$(wname).csv")
        !isfile(mpath) && (@warn "Moments not found for $wname"; continue)

        mdf = CSV.read(mpath, DataFrame)
        moment_vals = Dict(Symbol(row.moment) => row.value for row in eachrow(mdf))

        # moments_{window}.csv now stores the κ_w-adjusted training_share
        # (applied in Stage 7). Recover the raw level here so the influence
        # functions are centred/scaled on the raw moment exactly as before;
        # κ_w is re-applied to the saved Σ̂ row/col in Section I below.
        κ_ts = _load_training_share_scale(wname)
        if κ_ts != 1.0 && haskey(moment_vals, :training_share)
            moment_vals[:training_share] /= κ_ts
        end

        monthly_stack = Dict{Tuple{Int,Int}, Vector{Float64}}()
        asec_stack    = Dict{Tuple{Int,Int}, Vector{Float64}}()

        # ── A. CPS Basic stock moments ────────────────────────────────────────
        cps_w = filter(row -> row.window == wname, cps_basic_m)
        @assert nrow(cps_w) > 0 "CPS Basic empty for window $wname"

        # Denominators with the working-student exclusion (LF ∩ ¬train)
        lf_excl_train = filter(r -> r.in_lf && !r.in_training, cps_w)
        lf_all        = filter(r -> r.in_lf, cps_w)

        for gk in groupby(lf_excl_train, [:YEAR, :MONTH])
            g  = DataFrame(gk)
            yr = Int(g.YEAR[1]); mo = Int(g.MONTH[1])
            w  = Float64.(coalesce.(g.WTFINL, 0.0))
            sw = sum(w)
            sw <= 0 && continue

            psi    = zeros(K)
            n_u_U  = sum(w[g.unemployed .& .!g.skilled])
            n_u_S  = sum(w[g.unemployed .& g.skilled])
            n_lf_U = sum(w[.!g.skilled])
            n_lf_S = sum(w[g.skilled])

            # ur_total (working-student-excluded LF denominator)
            val_ur_total = (n_u_U + n_u_S) / sw
            mbar = get(moment_vals, :ur_total, NaN)
            isfinite(val_ur_total) && isfinite(mbar) && (psi[idx[:ur_total]] = val_ur_total - mbar)

            # ur_U
            if n_lf_U > 0
                val_ur_U = n_u_U / n_lf_U
                mbar = get(moment_vals, :ur_U, NaN)
                isfinite(val_ur_U) && isfinite(mbar) && (psi[idx[:ur_U]] = val_ur_U - mbar)
            end

            # skilled_share
            val  = n_lf_S / sw
            mbar = get(moment_vals, :skilled_share, NaN)
            isfinite(val) && isfinite(mbar) && (psi[idx[:skilled_share]] = val - mbar)

            _add_to_stack!(monthly_stack, (yr, mo), psi, K)
        end

        # ur_S on the unrestricted skilled LF
        for gk in groupby(lf_all, [:YEAR, :MONTH])
            g  = DataFrame(gk)
            yr = Int(g.YEAR[1]); mo = Int(g.MONTH[1])
            w  = Float64.(coalesce.(g.WTFINL, 0.0))
            n_u_S  = sum(w[g.unemployed .& g.skilled])
            n_lf_S = sum(w[g.skilled])
            psi = zeros(K)
            if n_lf_S > 0
                val_ur_S = n_u_S / n_lf_S
                mbar = get(moment_vals, :ur_S, NaN)
                isfinite(val_ur_S) && isfinite(mbar) && (psi[idx[:ur_S]] = val_ur_S - mbar)
            end
            _add_to_stack!(monthly_stack, (yr, mo), psi, K)
        end

        # training_share — strict (NILF ∩ train numerator, working-age pop denom)
        for gk in groupby(cps_w, [:YEAR, :MONTH])
            g    = DataFrame(gk)
            yr = Int(g.YEAR[1]); mo = Int(g.MONTH[1])
            w    = Float64.(coalesce.(g.WTFINL, 0.0))
            pop  = sum(w)
            pop <= 0 && continue
            psi = zeros(K)
            if hasproperty(g, :in_training) && hasproperty(g, :in_lf)
                trainees = sum(w[coalesce.(g.in_training, false) .& .!g.in_lf])
                val  = trainees / pop
                mbar = get(moment_vals, :training_share, NaN)
                isfinite(val) && isfinite(mbar) && (psi[idx[:training_share]] = val - mbar)
            end
            _add_to_stack!(monthly_stack, (yr, mo), psi, K)
        end

        # ltu_share_S — WTFINL-weighted share of skilled unemployed with a
        # long-term spell (DURUNEMP ≥ 27 weeks). Per-month cell deviation
        # (val_month − m̄), the same across-period ψ convention as the stock
        # moments above; the monthly cells carry its covariance with the rest.
        # TODO(ramzi): the analytic ratio/rate IF (action plan Part C) is in the
        # data appendix; Σ̂ here keeps the file's uniform per-period deviation
        # form so all moments share one variance construction.
        if hasproperty(cps_w, :DURUNEMP)
            for gk in groupby(cps_w, [:YEAR, :MONTH])
                g  = DataFrame(gk)
                yr = Int(g.YEAR[1]); mo = Int(g.MONTH[1])
                skl_u = g.unemployed .& g.skilled
                any(skl_u) || continue
                w   = Float64.(coalesce.(g.WTFINL, 0.0))[skl_u]
                dur = Float64.(coalesce.(g.DURUNEMP[skl_u], 0.0))
                sw  = sum(w)
                sw <= 0 && continue
                psi = zeros(K)
                val  = sum(w[dur .>= 27.0]) / sw
                mbar = get(moment_vals, :ltu_share_S, NaN)
                isfinite(val) && isfinite(mbar) && (psi[idx[:ltu_share_S]] = val - mbar)
                _add_to_stack!(monthly_stack, (yr, mo), psi, K)
            end
        else
            @warn "    DURUNEMP not in cleaned CPS Basic — ltu_share_S column of Σ̂ left at 0 for $wname."
        end

        # ── B. CPS transition moments (jfr, sep) ──────────────────────────────
        trans_w = filter(r -> r.window == wname, trans_monthly)
        @assert nrow(trans_w) > 0 "Transitions empty for window $wname"

        for gk in groupby(trans_w, [:year, :month])
            g  = DataFrame(gk)
            yr = Int(g.year[1]); mo = Int(g.month[1])
            psi = zeros(K)

            for sk_val in (false, true)
                row = filter(r -> Bool(r.skilled) == sk_val, g)
                nrow(row) == 0 && continue
                jfr_name = sk_val ? :jfr_S      : :jfr_U
                sep_name = sk_val ? :sep_rate_S : :sep_rate_U

                val  = Float64(row[1, :jfr])
                mbar = get(moment_vals, jfr_name, NaN)
                isfinite(val) && isfinite(mbar) && (psi[idx[jfr_name]] = val - mbar)

                val  = Float64(row[1, :sep])
                mbar = get(moment_vals, sep_name, NaN)
                isfinite(val) && isfinite(mbar) && (psi[idx[sep_name]] = val - mbar)
            end

            _add_to_stack!(monthly_stack, (yr, mo), psi, K)
        end

        # ── C. JOLTS theta moments ────────────────────────────────────────────
        jolts_w     = filter(row -> row.window == wname, jolts_m)
        theta_U_bar = get(moment_vals, :theta_U, NaN)
        theta_S_bar = get(moment_vals, :theta_S, NaN)

        monthly_U_lookup = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
        for gk in groupby(cps_w, [:YEAR, :MONTH])
            g  = DataFrame(gk)
            yr = Int(g.YEAR[1]); mo = Int(g.MONTH[1])
            w  = Float64.(coalesce.(g.WTFINL, 0.0))
            monthly_U_lookup[(yr, mo)] = (
                sum(w[g.unemployed .& .!g.skilled]),
                sum(w[g.unemployed .&  g.skilled])
            )
        end

        if nrow(jolts_w) > 0
            for row in eachrow(jolts_w)
                yr = Int(row.YEAR); mo = Int(row.MONTH)
                (U_U_t, U_S_t) = get(monthly_U_lookup, (yr, mo), (NaN, NaN))
                psi = zeros(K)
                if isfinite(U_U_t) && U_U_t > 0 && isfinite(row.V_U) && isfinite(theta_U_bar)
                    psi[idx[:theta_U]] = row.V_U / U_U_t - theta_U_bar
                end
                if isfinite(U_S_t) && U_S_t > 0 && isfinite(row.V_S) && isfinite(theta_S_bar)
                    psi[idx[:theta_S]] = row.V_S / U_S_t - theta_S_bar
                end
                _add_to_stack!(monthly_stack, (yr, mo), psi, K)
            end
        end

        # ── D. ASEC wage moments ──────────────────────────────────────────────
        asec_w = filter(row -> row.window == wname, cps_asec_m)
        @assert nrow(asec_w) > 0 "ASEC empty for window $wname"

        for gk in groupby(asec_w, :YEAR)
            g  = DataFrame(gk)
            yr = Int(g.YEAR[1])

            unskilled = filter(r -> !r.skilled, g)
            skilled   = filter(r ->  r.skilled, g)
            psi = zeros(K)

            if nrow(unskilled) > 0
                # LOG wages — match the model-side log-wage moments (and moments.jl).
                wu = log.(max.(Float64.(unskilled.wage_norm), 1e-14))
                wt = Float64.(unskilled.ASECWT)
                for (sym, fn) in ((:mean_wage_U, wmean), (:emp_var_U, wvar),
                                   (:emp_cm3_U, wcm3), (:p50_wage_U, wmedian),
                                   (:p25_wage_U, wpercentile25),
                                   (:p75_wage_U, wpercentile75))
                    val  = fn(wu, wt)
                    mbar = get(moment_vals, sym, NaN)
                    isfinite(val) && isfinite(mbar) && (psi[idx[sym]] = val - mbar)
                end
            end

            if nrow(skilled) > 0
                ws = log.(max.(Float64.(skilled.wage_norm), 1e-14))
                wt = Float64.(skilled.ASECWT)
                for (sym, fn) in ((:mean_wage_S, wmean), (:emp_var_S, wvar),
                                   (:emp_cm3_S, wcm3), (:p50_wage_S, wmedian),
                                   (:p25_wage_S, wpercentile25),
                                   (:p75_wage_S, wpercentile75))
                    val  = fn(ws, wt)
                    mbar = get(moment_vals, sym, NaN)
                    isfinite(val) && isfinite(mbar) && (psi[idx[sym]] = val - mbar)
                end
            end

            if nrow(unskilled) > 0 && nrow(skilled) > 0
                log_wu  = log.(max.(Float64.(unskilled.wage_norm), 1e-14))
                log_ws  = log.(max.(Float64.(skilled.wage_norm),   1e-14))
                wt_u    = Float64.(unskilled.ASECWT)
                wt_s    = Float64.(skilled.ASECWT)
                prem_yr = wmean(log_ws, wt_s) - wmean(log_wu, wt_u)
                mbar = get(moment_vals, :wage_premium, NaN)
                isfinite(prem_yr) && isfinite(mbar) && (psi[idx[:wage_premium]] = prem_yr - mbar)

                # Cross-market overlap — per-year cell deviation, the same
                # across-period ψ convention used for every ASEC wage moment
                # above (val_yr − m̄). Each year contributes one (overlap_UgtS,
                # overlap_SltU) pair to the ASEC stack; U and S are disjoint
                # subsamples of the same ASEC, so their covariance with the
                # other ASEC moments is preserved by the shared cell.
                # TODO(ramzi): the analytic two-sample IF with the estimated-
                # median correction (action plan Part C) is documented in the
                # data appendix; Σ̂ here keeps the file's uniform per-period
                # deviation form so all moments share one variance construction.
                med_S = wmedian(log_ws, wt_s)
                med_U = wmedian(log_wu, wt_u)
                sw_u  = sum(wt_u); sw_s = sum(wt_s)
                if sw_u > 0 && sw_s > 0
                    ov_UgtS = sum(wt_u[log_wu .> med_S]) / sw_u
                    ov_SltU = sum(wt_s[log_ws .< med_U]) / sw_s
                    mbar = get(moment_vals, :overlap_UgtS, NaN)
                    isfinite(ov_UgtS) && isfinite(mbar) && (psi[idx[:overlap_UgtS]] = ov_UgtS - mbar)
                    mbar = get(moment_vals, :overlap_SltU, NaN)
                    isfinite(ov_SltU) && isfinite(mbar) && (psi[idx[:overlap_SltU]] = ov_SltU - mbar)
                end
            end

            _add_to_stack!(asec_stack, (yr + 1, 1), psi, K)
        end

        # ── E. Aggregate to quarterly frequency ───────────────────────────────
        quarterly_stack = Dict{Tuple{Int,Int}, Vector{Float64}}()
        quarter_count   = Dict{Tuple{Int,Int}, Int}()

        for ((yr, mo), psi_m) in monthly_stack
            q    = month_to_quarter[mo]
            qkey = (yr, q)
            _add_to_stack!(quarterly_stack, qkey, psi_m, K)
            quarter_count[qkey] = get(quarter_count, qkey, 0) + 1
        end
        for (qkey, psi_q) in quarterly_stack
            quarterly_stack[qkey] = psi_q ./ get(quarter_count, qkey, 1)
        end
        for ((yr_p1, q), psi_a) in asec_stack
            _add_to_stack!(quarterly_stack, (yr_p1, q), psi_a, K)
        end

        # ── F. Build Ψ (full K columns) and normalise ─────────────────────────
        ee_S_bar = get(moment_vals, :ee_rate_S, NaN)
        j2j_w    = filter(r -> r.window == wname && r.education == "E4", j2j_nat)

        _, Psi = _stack_to_matrix(quarterly_stack, K)
        T = size(Psi, 1)
        @info "    T = $T quarters, K = $K"

        scales = zeros(K)
        for k in 1:K
            v = get(moment_vals, MOMENT_NAMES[k], NaN)
            scales[k] = (isfinite(v) && abs(v) >= 1e-10) ? abs(v) : 1.0
            if !isfinite(v)
                @warn "    Moment $(MOMENT_NAMES[k]) has non-finite value in $wname — scale defaults to 1.0"
            end
        end
        Psi ./= scales'
        @info "    Scale range: $(round(minimum(scales); sigdigits=3)) — $(round(maximum(scales); sigdigits=3))"

        Sigma = (Psi' * Psi) / T

        # ── G. ee_rate_S variance from J2J (E4-only) ──────────────────────────
        ee_k = idx[:ee_rate_S]
        if nrow(j2j_w) > 0 && isfinite(ee_S_bar)
            ee_vals = [Float64(r.ee_monthly) for r in eachrow(j2j_w)]
            T_ee    = length(ee_vals)
            ee_var  = sum((v - ee_S_bar)^2 for v in ee_vals) / T_ee
            Sigma[ee_k, ee_k] = ee_var / scales[ee_k]^2
            @info "    ee_rate_S: variance from $T_ee J2J quarters = $(round(Sigma[ee_k, ee_k]; sigdigits=4))"
        else
            @warn "    No J2J data for $wname; ee_rate_S variance left as 0"
        end

        # ── H. Regularisation and symmetry ────────────────────────────────────
        if REGULARIZATION_ALPHA > 0.0
            Sigma = (1.0 - REGULARIZATION_ALPHA) * Sigma +
                    REGULARIZATION_ALPHA * Diagonal(diag(Sigma))
        end
        Sigma .= (Sigma .+ Sigma') ./ 2

        @printf "    Σ̂ condition number: κ = %.2e\n" cond(Sigma)

        all_sigma[wname] = Sigma

        # ── I. Save ───────────────────────────────────────────────────────────
        # NSC level adjustment: scale the training_share row/col of the SAVED Σ̂
        # by κ_w (off-diagonals × κ, the (ts,ts) diagonal × κ²) so
        # sigma_{window}.csv is consistent with the κ-adjusted training_share
        # moment from Stage 7. The in-memory `all_sigma` (above) keeps the raw
        # matrix, so Stage 9 diagnostics are unchanged.
        Sigma_out = copy(Sigma)
        κ_ts = _load_training_share_scale(wname)
        if κ_ts != 1.0
            ts_idx = idx[:training_share]
            Sigma_out[ts_idx, :] .*= κ_ts
            Sigma_out[:, ts_idx] .*= κ_ts
        end

        col_names = string.(MOMENT_NAMES)
        CSV.write(joinpath(DERIVED_DIR, "sigma_$(wname).csv"),
                  DataFrame(Sigma_out, col_names))
        CSV.write(joinpath(DERIVED_DIR, "moment_scales_$(wname).csv"),
                  DataFrame(moment = col_names, scale = scales))
        @info "    Saved: sigma_$(wname).csv, moment_scales_$(wname).csv  ($K × $K)"
    end

    @info "  Done — full Σ̂ matrices saved for all windows"
    return all_sigma
end

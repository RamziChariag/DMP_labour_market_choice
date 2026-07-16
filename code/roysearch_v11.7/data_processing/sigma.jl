############################################################
# data_processing/sigma.jl
#
# Combine step (Stage 8) — the full 31×31 influence-function variance–
# covariance matrix Σ̂ for every window, used to build the SMM weight
# matrix. The training_share row/col of the SAVED Σ̂ carries the NSC κ_w
# adjustment (off-diagonals × κ, diagonal × κ²); the in-memory return
# stays raw for Stage 9 diagnostics. REGULARIZATION_ALPHA comes from setup.jl.
#
# Also builds a per-moment SAMPLING-variance vector σ̂²_samp (all on one 1/N
# footing) → sampling_var_{window}.csv, read only by the diagonal-σ SMM weight.
# The full Σ̂ IF matrix (sigma_{window}.csv) is unchanged by that addition.
#
# Reads:  moments_{window}.csv, cps_basic_clean.arrow, cps_asec_clean.arrow, jolts_clean.arrow, transitions_monthly.arrow, sipp_wchg_rates.csv, sipp_ee_rates.csv, training_share_scale.csv
# Writes: sigma_{window}.csv, moment_scales_{window}.csv, sampling_var_{window}.csv
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages and path consts come from data_processing_main.jl.
############################################################

TARGET_KAPPA = 1e8
SHRINK_TOL   = 1e-6
SHRINK_ITERS = 60

# Weighted-bootstrap replications for the per-moment sampling-variance vector
# (Stage 8, sampling_var_{window}.csv). Deterministic: the rng is seeded per
# window, so the sampling variances reproduce run-to-run.
const SAMPLING_VAR_BOOT = 500


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
# Per-moment SAMPLING variance σ̂²_samp (Stage 8 companion to Σ̂).
#
# Σ̂ (the influence-function matrix) mixes two footings on its diagonal: an
# across-period time-dispersion variance (÷ number of quarters) for the CPS/ASEC
# moments and a within-sample sampling variance (÷ neff) for the SIPP moments —
# ~100× apart, so 1/diag(Σ̂) is NOT a sound diagonal-σ weight. This vector puts
# EVERY moment on ONE footing: a 1/N-type sampling variance of the pooled window
# cross-section (the window = one steady state observed with sampling noise).
# It feeds ONLY the diagonal-σ SMM weight (W = Diagonal(1/σ̂²_samp)); Σ̂ and the
# full-Σ⁻¹ optimal-weight path are untouched.
#
# Cross-sectional wage/share moments use ONE mechanism: a seeded weighted
# bootstrap of the pooled window cross-section. Persons are resampled with
# probability ∝ survey weight, and each statistic is recomputed on the resample
# as an UNWEIGHTED function of the drawn rows — the survey weighting is already
# embodied in the resampling probabilities, so weighting again would double-count
# (this reproduces moments.jl's weighted point estimate in expectation and gives
# the correct sampling variance for means, variances, third moments and quantiles
# uniformly). Flow hazards and θ use the delta method (a closed form is exact and
# far cheaper than a bootstrap of the matched panel); SIPP moments are already
# sampling variances and are copied from the Σ̂ diagonal construction.

# Draw n row indices with replacement, prob ∝ weight (cw = cumsum of weights).
function _wboot_idx(rng::AbstractRNG, cw::Vector{Float64}, n::Int)::Vector{Int}
    tot = cw[end]
    idx = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        idx[i] = searchsortedfirst(cw, rand(rng) * tot)
    end
    return idx
end

# Sampling variance of a scalar statistic across B weighted-bootstrap reps.
# `stat(idx)` returns one replication's value from a resample index vector; reps
# are drawn from the pooled population `w` (∝ weight). NaN reps are dropped.
function _boot_var(rng::AbstractRNG, w::Vector{Float64}, B::Int, stat)::Float64
    (isempty(w) || sum(w) <= 0.0) && return NaN
    cw = cumsum(w)
    n  = length(w)
    reps = Float64[]
    for _ in 1:B
        v = stat(_wboot_idx(rng, cw, n))
        isfinite(v) && push!(reps, v)
    end
    length(reps) < 2 ? NaN : var(reps)
end

# Weighted-share sampling variance: fraction of the (∝weight) resample lying in
# `pop` that also lies in `num`, i.e. moments.jl's weighted ratio recomputed
# unweighted on the resample. `pop`/`num` are Bool masks over the pooled rows.
function _boot_share_var(rng::AbstractRNG, w::Vector{Float64}, B::Int,
                         pop::AbstractVector{Bool}, num::AbstractVector{Bool})::Float64
    _boot_var(rng, w, B, idx -> begin
        d = 0; k = 0
        @inbounds for i in idx
            if pop[i]; d += 1; num[i] && (k += 1); end
        end
        d == 0 ? NaN : k / d
    end)
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

    quarter_to_months = Dict(1 => [1,2,3], 2 => [4,5,6], 3 => [7,8,9], 4 => [10,11,12])
    month_to_quarter  = Dict(m => q for (q, ms) in quarter_to_months for m in ms)

    # SIPP moments and their Kish effective N (Stage 6b). The SIPP micro-data are
    # not re-read here (only CPS raw is); the point estimates + neff carried in
    # sipp_wchg_rates.csv (wchg_rate_U/S) and sipp_ee_rates.csv (ee_rate_S,
    # ee_step_S) are enough to place a delta-method diagonal variance on each SIPP
    # moment below. These moments have no influence-function column, so their
    # off-diagonals in Σ̂ stay zero (zero covariance, as for wchg).
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

        # ── G. SIPP moment variances (delta-method diagonals, zero covariance) ─
        # The SIPP moments carry no influence-function column above, so their
        # off-diagonals in Σ̂ are already zero; only the diagonal is set here,
        # from each moment's point estimate + Kish neff (the SIPP micro-data are
        # not re-read). The raw-unit variance of each SIPP moment is also kept in
        # `sipp_rawvar` for the sampling-variance vector (Section G2 — those are
        # already sampling variances, so they are copied, not recomputed).
        #
        # • Hazards wchg_rate_U/S and ee_rate_S. The stored moment is the hazard
        #   h = −log(1−p), so the binomial-proportion delta method gives
        #   Var(h) = Var(p)·(dh/dp)² = [p(1−p)/neff]·(1−p)⁻². (ee_rate_S is now
        #   SIPP-sourced like wchg — the former J2J-quarter variance is gone.)
        #   FC windows ship the BBG-corrected wchg hazard, whose variance carries
        #   the correction's uncertainty: the delta-method hazard variance is
        #   evaluated on the RAW break rate π̂ (binomial on the hourly-series neff)
        #   and scaled by 1/(γ̄−ᾱ)², since π̃ = (π̂−ᾱ)/(γ̄−ᾱ) divides by the
        #   correction factor. This inflates the FC wchg variance relative to a
        #   raw hazard — the honest "this is a bound" statement — so under
        #   variance weighting the FC wchg pins γ without dominating.
        # • Mean ee_step_S. A weighted mean of the EE-move wage step, so
        #   Var(mean) = sd²/neff directly from the weighted sd carried in the CSV.
        #
        # Where a window has no SIPP row the diagonal is left at 0 with a warning
        # (the moment is NaN there and auto-holds out of the objective anyway).
        sipp_w  = isempty(sipp_wchg) ? nothing : filter(r -> r.window == wname, sipp_wchg)
        sipp_er = isempty(sipp_ee)   ? nothing : filter(r -> r.window == wname, sipp_ee)
        sipp_rawvar = Dict{Symbol,Float64}()   # raw-unit SIPP variances → Section G2

        # Raw hazard delta-method variance on a proportion p with effective N.
        _haz_var(p, neff) = (neff > 0.0 && 0.0 < p < 1.0) ?
                            (p * (1.0 - p) / neff) / (1.0 - p)^2 : NaN

        # wchg hazards. FC (SIPP_BBG_WINDOWS) uses the BBG π̂/correction-factor
        # path; the redesign COVID windows use the raw earnings hazard.
        is_bbg = wname in SIPP_BBG_WINDOWS
        for (mname, neff_col, pihat_col, corrfac_col) in
                ((:wchg_rate_U, :neff_U, :bbg_pihat_U, :bbg_corrfac_U),
                 (:wchg_rate_S, :neff_S, :bbg_pihat_S, :bbg_corrfac_S))
            wk = idx[mname]
            h  = get(moment_vals, mname, NaN)
            if !isnothing(sipp_w) && nrow(sipp_w) > 0 && isfinite(h) && hasproperty(sipp_w, neff_col)
                neff = Float64(sipp_w[1, neff_col])
                if is_bbg
                    pihat   = Float64(sipp_w[1, pihat_col])
                    corrfac = Float64(sipp_w[1, corrfac_col])
                    var_raw = _haz_var(pihat, neff)
                    var_h   = (isfinite(var_raw) && isfinite(corrfac) && corrfac > 0.0) ?
                              var_raw / corrfac^2 : NaN
                    tag = "BBG delta-method (π̂=$(round(pihat; sigdigits=3)), 1/(γ̄−ᾱ)²=$(round(1/corrfac^2; sigdigits=3)))"
                else
                    var_h = _haz_var(1.0 - exp(-h), neff)
                    tag   = "delta-method"
                end
                if isfinite(var_h)
                    Sigma[wk, wk] = var_h / scales[wk]^2
                    sipp_rawvar[mname] = var_h
                    @info "    $mname: $tag variance (neff=$(round(neff; digits=1))) = $(round(Sigma[wk, wk]; sigdigits=4))"
                else
                    @warn "    $mname: degenerate inputs for $wname; variance left as 0"
                end
            else
                @warn "    No SIPP data for $wname; $mname variance left as 0"
            end
        end

        # ee_rate_S hazard (SIPP), raw delta method.
        let wk = idx[:ee_rate_S], h = get(moment_vals, :ee_rate_S, NaN)
            if !isnothing(sipp_er) && nrow(sipp_er) > 0 && isfinite(h) && hasproperty(sipp_er, :neff_ee)
                neff  = Float64(sipp_er[1, :neff_ee])
                var_h = _haz_var(1.0 - exp(-h), neff)
                if isfinite(var_h)
                    Sigma[wk, wk] = var_h / scales[wk]^2
                    sipp_rawvar[:ee_rate_S] = var_h
                    @info "    ee_rate_S: delta-method variance (neff=$(round(neff; digits=1))) = $(round(Sigma[wk, wk]; sigdigits=4))"
                else
                    @warn "    ee_rate_S: degenerate p/neff for $wname; variance left as 0"
                end
            else
                @warn "    No SIPP data for $wname; ee_rate_S variance left as 0"
            end
        end

        # ee_step_S mean → Var(mean) = sd²/neff from the weighted sd + Kish neff.
        step_k = idx[:ee_step_S]
        if !isnothing(sipp_er) && nrow(sipp_er) > 0 &&
           hasproperty(sipp_er, :ee_step_sd) && hasproperty(sipp_er, :neff_step)
            sd   = Float64(sipp_er[1, :ee_step_sd])
            neff = Float64(sipp_er[1, :neff_step])
            if isfinite(sd) && neff > 0.0
                var_step = sd^2 / neff
                Sigma[step_k, step_k] = var_step / scales[step_k]^2
                sipp_rawvar[:ee_step_S] = var_step
                @info "    ee_step_S: variance sd²/neff (neff=$(round(neff; digits=1))) = $(round(Sigma[step_k, step_k]; sigdigits=4))"
            else
                @warn "    ee_step_S: degenerate sd/neff for $wname; variance left as 0"
            end
        else
            @warn "    No SIPP data for $wname; ee_step_S variance left as 0"
        end

        # ── G2. Per-moment sampling variance σ̂²_samp (one footing) ────────────
        # A 1/N-type sampling variance for every moment, in RAW moment units (the
        # deviation vector g_k = m_k − m̂_k is raw, so W = Diagonal(1/σ̂²_samp)
        # acts on raw deviations). Written to sampling_var_{window}.csv and read
        # ONLY by the diagonal-σ SMM path; Σ̂ above is untouched.
        rng   = MersenneTwister(hash((:sampling_var, wname)))
        svar  = Dict{Symbol,Float64}()
        B     = SAMPLING_VAR_BOOT

        # (a) CPS-Basic shares — weighted bootstrap of the pooled window cross-
        #     section, each on the exact numerator/denominator masks moments.jl
        #     uses. Pooled arrays and masks built once.
        cw   = Float64.(coalesce.(cps_w.WTFINL, 0.0))
        unemp = Vector{Bool}(cps_w.unemployed)
        skl   = Vector{Bool}(cps_w.skilled)
        inlf  = hasproperty(cps_w, :in_lf) ? Vector{Bool}(coalesce.(cps_w.in_lf, false)) : trues(nrow(cps_w))
        intr  = hasproperty(cps_w, :in_training) ? Vector{Bool}(coalesce.(cps_w.in_training, false)) : falses(nrow(cps_w))
        lfx   = inlf .& .!intr                        # LF ∩ ¬train (ur_total, ur_U, skilled_share)
        svar[:ur_total]      = _boot_share_var(rng, cw, B, lfx, lfx .& unemp)
        svar[:ur_U]          = _boot_share_var(rng, cw, B, lfx .& .!skl, lfx .& .!skl .& unemp)
        svar[:ur_S]          = _boot_share_var(rng, cw, B, inlf .& skl, inlf .& skl .& unemp)
        svar[:skilled_share] = _boot_share_var(rng, cw, B, lfx, lfx .& skl)
        # training_share: NILF trainees over working-age pop; the shipped moment
        # carries the κ_w level scale, so its sampling variance scales by κ_ts².
        ts_var_raw = _boot_share_var(rng, cw, B, trues(nrow(cps_w)), intr .& .!inlf)
        svar[:training_share] = isfinite(ts_var_raw) ? ts_var_raw * κ_ts^2 : NaN
        # ltu_share_S: long-term share within the skilled-unemployed pool.
        if hasproperty(cps_w, :DURUNEMP)
            skl_u = skl .& unemp
            ltu   = skl_u .& (Float64.(coalesce.(cps_w.DURUNEMP, 0.0)) .>= 27.0)
            svar[:ltu_share_S] = _boot_share_var(rng, cw, B, skl_u, ltu)
        else
            svar[:ltu_share_S] = NaN
        end

        # (b) ASEC wage moments — weighted bootstrap of the pooled ASEC cross-
        #     section (log wages, ASECWT). One resample per replication drives all
        #     ASEC statistics; the wage-premium and overlap draws use the joint
        #     resample so the estimated-median dependence is captured.
        aw    = Float64.(coalesce.(asec_w.ASECWT, 0.0))
        alogw = log.(max.(Float64.(asec_w.wage_norm), 1e-14))
        askl  = Vector{Bool}(asec_w.skilled)
        ones_of(v) = fill(1.0, length(v))
        # single-subsample statistics (recomputed unweighted on the resample)
        for (sym, sub, fn) in ((:mean_wage_U, .!askl, wmean), (:emp_var_U, .!askl, wvar),
                               (:emp_cm3_U, .!askl, wcm3), (:p25_wage_U, .!askl, wpercentile25),
                               (:p50_wage_U, .!askl, wmedian), (:p75_wage_U, .!askl, wpercentile75),
                               (:mean_wage_S, askl, wmean), (:emp_var_S, askl, wvar),
                               (:emp_cm3_S, askl, wcm3), (:p25_wage_S, askl, wpercentile25),
                               (:p50_wage_S, askl, wmedian), (:p75_wage_S, askl, wpercentile75))
            svar[sym] = _boot_var(rng, aw, B, idx -> begin
                xs = [alogw[i] for i in idx if sub[i]]
                isempty(xs) ? NaN : fn(xs, ones_of(xs))
            end)
        end
        # joint-resample statistics
        svar[:wage_premium] = _boot_var(rng, aw, B, idx -> begin
            xu = [alogw[i] for i in idx if !askl[i]]; xs = [alogw[i] for i in idx if askl[i]]
            (isempty(xu) || isempty(xs)) ? NaN : wmean(xs, ones_of(xs)) - wmean(xu, ones_of(xu))
        end)
        svar[:overlap_UgtS] = _boot_var(rng, aw, B, idx -> begin
            xu = [alogw[i] for i in idx if !askl[i]]; xs = [alogw[i] for i in idx if askl[i]]
            (isempty(xu) || isempty(xs)) ? NaN : count(>(wmedian(xs, ones_of(xs))), xu) / length(xu)
        end)
        svar[:overlap_SltU] = _boot_var(rng, aw, B, idx -> begin
            xu = [alogw[i] for i in idx if !askl[i]]; xs = [alogw[i] for i in idx if askl[i]]
            (isempty(xu) || isempty(xs)) ? NaN : count(<(wmedian(xu, ones_of(xu))), xs) / length(xs)
        end)

        # (c) Flow hazards jfr/sep — delta method on the transition proportion
        #     with the pooled at-risk Kish N (neff_jfr/neff_sep summed across the
        #     window's months, written by Stage 4). Same form as the SIPP hazards.
        for (mname, neff_col) in ((:jfr_U, :neff_jfr), (:jfr_S, :neff_jfr),
                                  (:sep_rate_U, :neff_sep), (:sep_rate_S, :neff_sep))
            skv = mname in (:jfr_S, :sep_rate_S)
            g   = filter(r -> Bool(r.skilled) == skv, trans_w)
            h   = get(moment_vals, mname, NaN)
            if nrow(g) > 0 && isfinite(h) && hasproperty(g, neff_col)
                N = sum(Float64.(coalesce.(g[!, neff_col], 0.0)))
                svar[mname] = _haz_var(1.0 - exp(-h), N)
            else
                svar[mname] = NaN
            end
        end

        # (d) θ = V/U — delta-method ratio dominated by the CPS unemployment
        #     sampling error: Var(θ) ≈ θ²·Var(U_count)/U_count², with the pooled
        #     binomial relative count variance (1−p)/(p·N_lf), p = ur_j, N_lf the
        #     Kish N of the market LF. V (JOLTS aggregate) is treated as measured.
        for (mname, ur_name, sub) in ((:theta_U, :ur_U, .!skl), (:theta_S, :ur_S, skl))
            th = get(moment_vals, mname, NaN); p = get(moment_vals, ur_name, NaN)
            wlf = cw[inlf .& sub]
            N_lf = sum(wlf) > 0 ? sum(wlf)^2 / sum(abs2, wlf) : 0.0
            svar[mname] = (isfinite(th) && isfinite(p) && 0.0 < p < 1.0 && N_lf > 0.0) ?
                          th^2 * (1.0 - p) / (p * N_lf) : NaN
        end

        # (e) SIPP moments — already sampling variances; copy the raw-unit values
        #     built in Section G (FC wchg is the BBG-inflated variance).
        for m in (:wchg_rate_U, :wchg_rate_S, :ee_rate_S, :ee_step_S)
            svar[m] = get(sipp_rawvar, m, NaN)
        end

        sv_col = [get(svar, m, NaN) for m in MOMENT_NAMES]
        CSV.write(joinpath(DERIVED_DIR, "sampling_var_$(wname).csv"),
                  DataFrame(moment = string.(MOMENT_NAMES), sampling_var = sv_col))
        finite_sv = filter(isfinite, sv_col)
        if !isempty(finite_sv)
            pos = filter(>(0.0), finite_sv)
            ratio = isempty(pos) ? NaN : maximum(pos) / minimum(pos)
            @info "    σ̂²_samp: $(length(finite_sv)) finite of $K; max/min ratio = $(round(ratio; sigdigits=4))"
        end
        @info "    Saved: sampling_var_$(wname).csv"

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

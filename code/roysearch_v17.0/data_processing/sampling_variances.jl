############################################################
# data_processing/sampling_variances.jl
#
# Combine step (Stage 8) — the per-moment SAMPLING-variance vector σ̂²_samp,
# one entry per moment on a single 1/N footing (the window = one steady state
# observed with sampling noise). Read only by the diagonal-σ SMM weight
# W = Diagonal(1/σ̂²_samp); the equal-weight path ignores it.
#
# Every entry is a CLOSED FORM — no bootstrap, no resampling, no RNG — so the
# vector is a deterministic function of the data: two runs are bit-identical.
# CPS shares get the Kish weighted-proportion variance; ASEC wage moments get
# analytic delta-method / order-statistic forms on the pooled window cross-
# section, each divided by the Kish N_eff of the relevant sector pool; flow
# hazards and θ use the delta method; SIPP moments are already sampling
# variances (delta-method / sd²·N_eff, FC wchg BBG-corrected); the J2J-sourced
# ee_rate_S uses the across-quarter variance of the quarterly hazards (sd²,
# genuine within-window movement + per-quarter sampling, not divided by n).
#
# The training_share row carries the NSC κ_w level adjustment: moments.jl reads
# the κ-scaled target, so its sampling variance scales by κ_ts².
#
# Reads:  moments_{window}.csv, cps_basic_clean.arrow, cps_asec_clean.arrow, transitions_monthly.arrow, sipp_wchg_rates.csv, sipp_ee_rates.csv, j2j_ee_rates.csv, training_share_scale.csv
# Writes: sampling_var_{window}.csv
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages and path consts come from data_processing_main.jl.
############################################################

# ─────────────────────────────────────────────────────────────────────────────
# The closed-form sampling-variance helpers (_kish_prop_var, _kish_neff, _wcm,
# _wquantile_var) and the cluster-robust _kish_prop_var_clustered live in
# setup.jl, which data_processing_main.jl includes first. They were duplicated
# verbatim here and in sigma.jl; one definition now serves both.
# ─────────────────────────────────────────────────────────────────────────────

# Within-person correlation assumed for the ASEC design effect (see the ASEC
# block below). Log wages are highly persistent within person, so the upper end
# is the relevant one; at ρ = 0.9 and the measured 1.21-1.29 records per person
# the resulting variance inflation is ~19-26%.
const ASEC_CLUSTER_RHO = 0.9

# ─────────────────────────────────────────────────────────────────────────────
# Stage 8: per-moment sampling variance σ̂²_samp for every window.
# ─────────────────────────────────────────────────────────────────────────────
function compute_sampling_variances()
    @info "Stage 8: per-moment sampling variances (closed-form, $(length(MOMENT_NAMES)) moments)..."

    cps_basic_m   = _load_arrow("cps_basic_clean.arrow")
    cps_asec_m    = _load_arrow("cps_asec_clean.arrow")
    trans_monthly = _load_arrow("transitions_monthly.arrow")

    for df in (cps_basic_m, cps_asec_m, trans_monthly)
        hasproperty(df, :window) && (df.window = Symbol.(df.window))
    end

    # SIPP moments carry their own sampling variances (delta-method hazards +
    # sd²/N_eff mean), built below from the point estimates + Kish neff in
    # sipp_wchg_rates.csv (wchg_rate_U/S) and sipp_ee_rates.csv (ee_step_S); the
    # SIPP micro-data are not re-read here. ee_rate_S sources its variance from
    # j2j_ee_rates.csv (across-quarter variance of the quarterly hazards, sd²).
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

    # J2J EE rates (Stage 5): the ee_rate_S target source. Its window-level
    # sampling variance is the across-quarter variance of the quarterly
    # hazards (sd², genuine+sampling), built below from ee_rate_S_sd.
    j2j_ee_path = joinpath(DERIVED_DIR, "j2j_ee_rates.csv")
    j2j_ee = isfile(j2j_ee_path) ? CSV.read(j2j_ee_path, DataFrame) : DataFrame()
    if !isempty(j2j_ee) && hasproperty(j2j_ee, :window)
        j2j_ee.window = Symbol.(j2j_ee.window)
    end

    for (wname, wdef) in WINDOWS
        @info "  Window: $(wdef.label) ($wname)"

        mpath = joinpath(DERIVED_DIR, "moments_$(wname).csv")
        !isfile(mpath) && (@warn "Moments not found for $wname"; continue)

        mdf = CSV.read(mpath, DataFrame)
        moment_vals = Dict(Symbol(row.moment) => row.value for row in eachrow(mdf))

        # training_share comes from NSC, not from this file's CPS closed forms;
        # its level and variance are both read from training_share_target.csv
        # (see nsc.jl). Nothing to recover here.

        cps_w = filter(row -> row.window == wname, cps_basic_m)
        @assert nrow(cps_w) > 0 "CPS Basic empty for window $wname"

        asec_w = filter(row -> row.window == wname, cps_asec_m)
        @assert nrow(asec_w) > 0 "ASEC empty for window $wname"

        trans_w = filter(r -> r.window == wname, trans_monthly)
        @assert nrow(trans_w) > 0 "Transitions empty for window $wname"

        # ── SIPP/J2J moment sampling variances (raw units) ────────────────────
        # Each moment's own sampling variance, from its point estimate + Kish
        # neff (the SIPP micro-data are not re-read). The SIPP wchg hazards
        # wchg_rate_U/S use the binomial-proportion delta method on h = −log(1−p):
        # Var(h) = [p(1−p)/neff]·(1−p)⁻². FC windows ship the BBG-corrected wchg
        # hazard, whose variance is evaluated on the RAW break rate π̂ (binomial on
        # the hourly-series neff) and scaled by 1/(γ̄−ᾱ)², since π̃ = (π̂−ᾱ)/(γ̄−ᾱ)
        # divides by the correction factor, inflating the FC wchg variance.
        # ee_step_S (SIPP) is a weighted mean, so Var(mean) = sd²/neff from the
        # weighted sd. ee_rate_S (J2J) uses the across-quarter variance of the
        # quarterly hazards, sd² (see its block below). Where a window lacks a row
        # the variance is left NaN with a warning (the moment holds out).
        sipp_w  = isempty(sipp_wchg) ? nothing : filter(r -> r.window == wname, sipp_wchg)
        sipp_er = isempty(sipp_ee)   ? nothing : filter(r -> r.window == wname, sipp_ee)
        j2j_er  = isempty(j2j_ee)    ? nothing : filter(r -> r.window == wname, j2j_ee)
        sipp_rawvar = Dict{Symbol,Float64}()

        # Raw hazard delta-method variance on a proportion p with effective N.
        _haz_var(p, neff) = (neff > 0.0 && 0.0 < p < 1.0) ?
                            (p * (1.0 - p) / neff) / (1.0 - p)^2 : NaN

        # wchg hazards. FC (SIPP_BBG_WINDOWS) uses the BBG π̂/correction-factor
        # path; the redesign COVID windows use the raw earnings hazard.
        is_bbg = wname in SIPP_BBG_WINDOWS
        for (mname, neff_col, pihat_col, corrfac_col) in
                ((:wchg_rate_U, :neff_U, :bbg_pihat_U, :bbg_corrfac_U),
                 (:wchg_rate_S, :neff_S, :bbg_pihat_S, :bbg_corrfac_S))
            h = get(moment_vals, mname, NaN)
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
                    sipp_rawvar[mname] = var_h
                    @info "    $mname: $tag variance (neff=$(round(neff; digits=1))) = $(round(var_h; sigdigits=4))"
                else
                    @warn "    $mname: degenerate inputs for $wname; variance left NaN"
                end
            else
                @warn "    No SIPP data for $wname; $mname variance left NaN"
            end
        end

        # ee_rate_S variance = across-quarter variance of the J2J quarterly
        # hazards, Var = sd². The quarter-to-quarter dispersion equals
        # Var(genuine within-window movement) + Var(per-quarter sampling), so
        # this single number carries both sources at full scale. It is NOT
        # divided by n_quarters: that would form the SE of the window mean and
        # average the genuine-movement component away. The micro binomial
        # variance is deliberately not used — its near-zero value would let an
        # administrative moment dominate a variance-weighted objective.
        if !isnothing(j2j_er) && nrow(j2j_er) > 0 &&
           hasproperty(j2j_er, :ee_rate_S_sd) && hasproperty(j2j_er, :n_quarters)
            sd  = Float64(j2j_er[1, :ee_rate_S_sd])
            nq  = Float64(j2j_er[1, :n_quarters])
            if isfinite(sd) && nq >= 2
                var_ee = sd^2
                sipp_rawvar[:ee_rate_S] = var_ee
                @info "    ee_rate_S: across-quarter variance of J2J quarterly hazards sd² (genuine+sampling, n_quarters=$(Int(nq))) = $(round(var_ee; sigdigits=4))"
            else
                @warn "    ee_rate_S: J2J sd NaN or n_quarters < 2 for $wname; variance left NaN"
            end
        else
            @warn "    No J2J data for $wname; ee_rate_S variance left NaN"
        end

        # ee_step_S mean → Var(mean) = sd²/neff from the weighted sd + Kish neff.
        if !isnothing(sipp_er) && nrow(sipp_er) > 0 &&
           hasproperty(sipp_er, :ee_step_sd) && hasproperty(sipp_er, :neff_step)
            sd   = Float64(sipp_er[1, :ee_step_sd])
            neff = Float64(sipp_er[1, :neff_step])
            if isfinite(sd) && neff > 0.0
                var_step = sd^2 / neff
                sipp_rawvar[:ee_step_S] = var_step
                @info "    ee_step_S: variance sd²/neff (neff=$(round(neff; digits=1))) = $(round(var_step; sigdigits=4))"
            else
                @warn "    ee_step_S: degenerate sd/neff for $wname; variance left NaN"
            end
        else
            @warn "    No SIPP data for $wname; ee_step_S variance left NaN"
        end

        # ── Per-moment sampling variance σ̂²_samp (one 1/N footing) ────────────
        # W = Diagonal(1/σ̂²_samp) acts on the RAW deviation vector g_k = m_k − m̂_k
        # (raw moment units), so every entry is a raw-unit 1/N sampling variance.
        svar  = Dict{Symbol,Float64}()

        # (a) CPS-Basic shares — exact analytic Kish-effective-N weighted-proportion
        #     variance on the pooled window cross-section, each on the exact
        #     numerator/denominator masks moments.jl uses (O(N) in the window size).
        #     Pooled arrays and masks built once.
        cw   = Float64.(coalesce.(cps_w.WTFINL, 0.0))
        unemp = Vector{Bool}(cps_w.unemployed)
        skl   = Vector{Bool}(cps_w.skilled)
        inlf  = hasproperty(cps_w, :in_lf) ? Vector{Bool}(coalesce.(cps_w.in_lf, false)) : trues(nrow(cps_w))
        intr  = hasproperty(cps_w, :in_training) ? Vector{Bool}(coalesce.(cps_w.in_training, false)) : falses(nrow(cps_w))
        lfx   = inlf .& .!intr                        # LF ∩ ¬train (ur_total, ur_U, skilled_share)
        # Person-clustered: CPS Basic is a 4-8-4 rotating panel, so the pooled
        # window sample repeats each person 4-8 times. See _kish_prop_var_clustered
        # in setup.jl for why the unclustered form understates these by 2.2-5.6x.
        cid = Int64.(coalesce.(cps_w.CPSIDP, 0))
        svar[:ur_total]      = _kish_prop_var_clustered(cw, lfx, lfx .& unemp, cid)
        svar[:ur_U]          = _kish_prop_var_clustered(cw, lfx .& .!skl, lfx .& .!skl .& unemp, cid)
        svar[:ur_S]          = _kish_prop_var_clustered(cw, inlf .& skl, inlf .& skl .& unemp, cid)
        svar[:skilled_share] = _kish_prop_var_clustered(cw, lfx, lfx .& skl, cid)
        # training_share: read from NSC alongside the level. NSC is an
        # administrative near-census, so the honest uncertainty is the
        # across-Fall-year dispersion of the series, not a survey sampling
        # variance — same convention as ee_rate_S above.
        svar[:training_share] = _load_training_share_target(wname).sampling_var
        # ltu_share_S: long-term share within the skilled-unemployed pool.
        if hasproperty(cps_w, :DURUNEMP)
            skl_u = skl .& unemp
            ltu   = skl_u .& (_durw.(cps_w.DURUNEMP) .>= 27.0)
            svar[:ltu_share_S] = _kish_prop_var_clustered(cw, skl_u, ltu, cid)
        else
            svar[:ltu_share_S] = NaN
        end

        # (b) ASEC wage moments — analytic 1/N sampling variances (NO bootstrap):
        #     closed forms on the pooled ASEC log-wage sample, each divided by the
        #     Kish N_eff of the relevant sector pool.
        aw    = Float64.(coalesce.(asec_w.ASECWT, 0.0))
        alogw = log.(max.(Float64.(asec_w.wage_norm), 1e-14))
        askl  = Vector{Bool}(asec_w.skilled)
        u_mask = .!askl
        # Per-sector pooled log-wage arrays and weights (built once).
        xu = alogw[u_mask]; wu = aw[u_mask]
        xs = alogw[askl];   ws = aw[askl]
        # ASEC person-clustering. The March supplement has a designed two-year
        # overlap: a household in ASEC one March is in it again the next, so the
        # pooled window sample repeats people (measured: 1.21-1.29 records per
        # person, ~15-29% appearing twice, essentially nobody three times). Every
        # ASEC variance below is of the form ·/N_eff, so the whole correction is a
        # single deflator on N_eff: with k records per person and within-person
        # correlation ρ, the design effect is 1 + (k−1)ρ. Log wages are highly
        # persistent within person, so ρ = 0.9 is the relevant end and the honest
        # correction is a ~20-27% variance inflation.
        #
        # This matters mostly RELATIVE to the CPS block, which is clustered above
        # at 2.2-5.6×. Correcting one and not the other would tilt the objective
        # between the wage moments (which identify β off the wage step at a job
        # move) and the unemployment/transition moments by a factor of 2-3.
        _asec_deff(id, w) = begin
            n = count(!iszero, w)
            n == 0 && return 1.0
            k = n / length(unique(id))
            1.0 + (k - 1.0) * ASEC_CLUSTER_RHO
        end
        acid  = Int64.(coalesce.(asec_w.CPSIDP, 0))
        cid_u = acid[u_mask]; cid_s = acid[askl]
        deff_u = _asec_deff(cid_u, wu); deff_s = _asec_deff(cid_s, ws)
        neff_u = _kish_neff(wu) / deff_u; neff_s = _kish_neff(ws) / deff_s
        @info @sprintf("    ASEC person-clustering: deff_U=%.3f deff_S=%.3f (ρ=%.1f)",
                       deff_u, deff_s, ASEC_CLUSTER_RHO)

        # Analytic sector-moment variances. `_mom_vars(x, w, neff)` returns
        # (Var(mean), Var(m2), Var(m3)) with:
        #   Var(mean) = μ2 / N_eff
        #   Var(m2)   = (μ4 − μ2²) / N_eff
        #   Var(m3)   = (μ6 − μ3² − 6·μ4·μ2 + 9·μ2³) / N_eff
        function _mom_vars(x, w, neff)
            (neff <= 0.0 || length(x) == 0) && return (NaN, NaN, NaN)
            μ2 = _wcm(x, w, 2); μ3 = _wcm(x, w, 3)
            μ4 = _wcm(x, w, 4); μ6 = _wcm(x, w, 6)
            v_mean = μ2 / neff
            v_m2   = (μ4 - μ2^2) / neff
            v_m3   = (μ6 - μ3^2 - 6μ4*μ2 + 9μ2^3) / neff
            return (v_mean, v_m2, v_m3)
        end
        vmU, vm2U, vm3U = _mom_vars(xu, wu, neff_u)
        vmS, vm2S, vm3S = _mom_vars(xs, ws, neff_s)
        svar[:mean_wage_U] = vmU; svar[:emp_var_U] = vm2U; svar[:emp_cm3_U] = vm3U
        svar[:mean_wage_S] = vmS; svar[:emp_var_S] = vm2S; svar[:emp_cm3_S] = vm3S

        # Quantile variances p(1−p)/(N_eff·f(q̂)²): use the SHIPPED quantile point
        # estimates (moment_vals) as q̂, densities via kde_at_point on the sector
        # pool (Silverman bandwidth, Kish N_eff).
        for (sym, x, w, p) in ((:p25_wage_U, xu, wu, 0.25), (:p50_wage_U, xu, wu, 0.50),
                               (:p75_wage_U, xu, wu, 0.75), (:p25_wage_S, xs, ws, 0.25),
                               (:p50_wage_S, xs, ws, 0.50), (:p75_wage_S, xs, ws, 0.75))
            qhat = get(moment_vals, sym, NaN)
            svar[sym] = _wquantile_var(x, w, qhat, p)
        end

        # wage_premium = mean_S − mean_U over disjoint subsamples → variances add.
        svar[:wage_premium] = (isfinite(vmU) && isfinite(vmS)) ? vmU + vmS : NaN

        # Overlap moments — analytic two-sample forms with the estimated-median
        # correction. overlap_UgtS = P(logw_U > m_S), m_S = skilled median:
        #   Var = q(1−q)/N_eff_U + [f_U(m_S)²/f_S(m_S)²]·(1/4)/N_eff_S
        # (first term: binomial sampling of the U tail; second: propagation of the
        # sampling error in the estimated m_S, whose own variance is (1/4)/(N_eff_S·f_S²)).
        m_S = get(moment_vals, :p50_wage_S, NaN)
        m_U = get(moment_vals, :p50_wage_U, NaN)
        q_UgtS = get(moment_vals, :overlap_UgtS, NaN)
        q_SltU = get(moment_vals, :overlap_SltU, NaN)
        if isfinite(m_S) && isfinite(q_UgtS) && neff_u > 0.0 && neff_s > 0.0
            fU_mS = kde_at_point(xu, wu, m_S); fS_mS = kde_at_point(xs, ws, m_S)
            svar[:overlap_UgtS] = (isfinite(fU_mS) && isfinite(fS_mS) && fS_mS > 0.0) ?
                q_UgtS*(1.0-q_UgtS)/neff_u + (fU_mS^2/fS_mS^2)*0.25/neff_s : NaN
        else
            svar[:overlap_UgtS] = NaN
        end
        if isfinite(m_U) && isfinite(q_SltU) && neff_u > 0.0 && neff_s > 0.0
            fS_mU = kde_at_point(xs, ws, m_U); fU_mU = kde_at_point(xu, wu, m_U)
            svar[:overlap_SltU] = (isfinite(fS_mU) && isfinite(fU_mU) && fU_mU > 0.0) ?
                q_SltU*(1.0-q_SltU)/neff_s + (fS_mU^2/fU_mU^2)*0.25/neff_u : NaN
        else
            svar[:overlap_SltU] = NaN
        end

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
        #     built above (FC wchg is the BBG-inflated variance).
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
            @info "    σ̂²_samp: $(length(finite_sv)) finite of $(length(MOMENT_NAMES)); max/min ratio = $(round(ratio; sigdigits=4))"
        end
        @info "    Saved: sampling_var_$(wname).csv"
    end

    @info "  Done — per-moment sampling variances saved for all windows"
    return nothing
end

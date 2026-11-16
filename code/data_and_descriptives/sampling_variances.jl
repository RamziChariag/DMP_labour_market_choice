############################################################
# code/data_and_descriptives/sampling_variances.jl
#
# Per-moment sampling variances σ̂²_samp, in raw moment units on a single 1/N
# footing (each window is one steady state observed with sampling noise). They
# form the diagonal SMM weight W = Diagonal(1/σ̂²_samp). Every entry is a closed
# form (no bootstrap, no RNG), so the output is bit-identical across runs.
#
# Entry point: compute_sampling_variances()
#   reads  moments_{window}.csv, cps_basic_clean.arrow, cps_asec_clean.arrow,
#          transitions_monthly.arrow, sipp_wchg_rates.csv, sipp_ee_rates.csv,
#          j2j_ee_rates.csv, training_share_target.csv
#   writes sampling_var_{window}.csv
# The closed-form helpers live in setup.jl.
# Definitions only; packages and path constants come from data_processing_main.jl.
############################################################

"""
    ASEC_CLUSTER_RHO

Within-person correlation of log wages in the ASEC design effect 1 + (k − 1)ρ,
k the records per person (1.21–1.29 from the supplement's two-year overlap).
Log wages are persistent within person, so the upper end applies.
"""
const ASEC_CLUSTER_RHO = 0.9

"""
    compute_sampling_variances()

Write sampling_var_{window}.csv (columns `moment`, `sampling_var`, in
`MOMENT_NAMES` order) for every window:
- CPS shares and duration survivors: person-clustered linearised variance
  (`_kish_prop_var_clustered`) on the same masks as moments.jl;
- `training_share`: the across-Fall-year variance of the enrolment target;
- ASEC wage moments: analytic moment, quantile (p(1−p)/(N f(q̂)²) at the
  shipped q̂) and overlap forms, with N_eff deflated by the ASEC design effect;
- jfr/sep: delta method on h = −log(1−p) with the pooled at-risk Kish N;
- θ_j: θ²(1−p)/(p N_lf), the CPS unemployment-count error (V treated as measured);
- SIPP wchg: delta method, on the raw break rate π̂ scaled by 1/(γ̄−ᾱ)² on the
  BBG windows; `ee_step_S`: sd²/N_eff; `ee_rate_S`: J2J across-quarter sd².
A moment whose source is missing gets NaN (with a warning) and is held out.
"""
function compute_sampling_variances()
    @info "Stage 8: per-moment sampling variances (closed-form, $(length(MOMENT_NAMES)) moments)..."

    cps_basic_m   = _load_arrow("cps_basic_clean.arrow")
    cps_asec_m    = _load_arrow("cps_asec_clean.arrow")
    trans_monthly = _load_arrow("transitions_monthly.arrow")

    for df in (cps_basic_m, cps_asec_m, trans_monthly)
        hasproperty(df, :window) && (df.window = Symbol.(df.window))
    end

    # SIPP and J2J variances are built from the point estimates and N_eff columns
    # of their derived CSVs; the micro data are not re-read.
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

        cps_w = filter(row -> row.window == wname, cps_basic_m)
        @assert nrow(cps_w) > 0 "CPS Basic empty for window $wname"

        asec_w = filter(row -> row.window == wname, cps_asec_m)
        @assert nrow(asec_w) > 0 "ASEC empty for window $wname"

        trans_w = filter(r -> r.window == wname, trans_monthly)
        @assert nrow(trans_w) > 0 "Transitions empty for window $wname"

        # ── SIPP and J2J moments ──────────────────────────────────────────────
        sipp_w  = isempty(sipp_wchg) ? nothing : filter(r -> r.window == wname, sipp_wchg)
        sipp_er = isempty(sipp_ee)   ? nothing : filter(r -> r.window == wname, sipp_ee)
        j2j_er  = isempty(j2j_ee)    ? nothing : filter(r -> r.window == wname, j2j_ee)
        sipp_rawvar = Dict{Symbol,Float64}()

        # Delta-method variance of h = −log(1−p) for a proportion p with N_eff:
        # [p(1−p)/N_eff]·(1−p)⁻².
        _haz_var(p, neff) = (neff > 0.0 && 0.0 < p < 1.0) ?
                            (p * (1.0 - p) / neff) / (1.0 - p)^2 : NaN

        # wchg hazards. BBG windows: π̃ = (π̂−ᾱ)/(γ̄−ᾱ), so the variance is
        # _haz_var evaluated at the raw break rate π̂, divided by corrfac² = (γ̄−ᾱ)².
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

        # ee_rate_S: across-quarter variance sd² of the J2J series, NOT divided by
        # n_quarters. It carries genuine within-window movement plus sampling
        # noise; the near-zero micro binomial variance would let an
        # administrative moment dominate the objective.
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

        # ee_step_S (weighted mean): Var = sd²/N_eff.
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

        svar  = Dict{Symbol,Float64}()

        # (a) CPS shares: person-clustered, on the masks moments.jl uses.
        cw   = Float64.(coalesce.(cps_w.WTFINL, 0.0))
        unemp = Vector{Bool}(cps_w.unemployed)
        skl   = Vector{Bool}(cps_w.skilled)
        inlf  = hasproperty(cps_w, :in_lf) ? Vector{Bool}(coalesce.(cps_w.in_lf, false)) : trues(nrow(cps_w))
        intr  = hasproperty(cps_w, :in_training) ? Vector{Bool}(coalesce.(cps_w.in_training, false)) : falses(nrow(cps_w))
        lfx   = inlf .& .!intr                        # LF ∩ ¬train (ur_total, ur_U, skilled_share)
        cid = Int64.(coalesce.(cps_w.CPSIDP, 0))
        svar[:ur_total]      = _kish_prop_var_clustered(cw, lfx, lfx .& unemp, cid)
        svar[:ur_U]          = _kish_prop_var_clustered(cw, lfx .& .!skl, lfx .& .!skl .& unemp, cid)
        svar[:ur_S]          = _kish_prop_var_clustered(cw, inlf .& skl, inlf .& skl .& unemp, cid)
        svar[:skilled_share] = _kish_prop_var_clustered(cw, lfx, lfx .& skl, cid)
        svar[:training_share] = _load_training_share_target(wname).sampling_var
        # Duration survivors: shares within each market's unemployed pool (U
        # excludes trainees, as in moments.jl); ltu_share_S reuses usurv27_S.
        if hasproperty(cps_w, :DURUNEMP)
            durw  = _durw.(cps_w.DURUNEMP)
            pools = ("U" => unemp .& .!skl .& .!intr, "S" => unemp .& skl)
            for (tag, pool) in pools
                for t in DURATION_THRESHOLDS_WK
                    svar[Symbol("usurv$(Int(t))_$(tag)")] =
                        _kish_prop_var_clustered(cw, pool, pool .& (durw .>= t), cid)
                end
            end
            svar[:ltu_share_S] = svar[:usurv27_S]
        else
            for k in DURATION_MOMENT_NAMES; svar[k] = NaN; end
            svar[:ltu_share_S] = NaN
        end

        # (b) ASEC wage moments on the pooled window sample of log wages.
        aw    = Float64.(coalesce.(asec_w.ASECWT, 0.0))
        alogw = log.(max.(Float64.(asec_w.wage_norm), 1e-14))
        askl  = Vector{Bool}(asec_w.skilled)
        u_mask = .!askl
        xu = alogw[u_mask]; wu = aw[u_mask]
        xs = alogw[askl];   ws = aw[askl]
        # Person clustering through N_eff: the supplement's two-year overlap
        # repeats people, so N_eff is divided by 1 + (k − 1)ρ. Leaving ASEC
        # unclustered while CPS is clustered would tilt the objective between
        # the wage and the labour-flow moments by a factor of 2–3.
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

        # (Var(mean), Var(m2), Var(m3)) = (μ2, μ4 − μ2², μ6 − μ3² − 6μ4μ2 + 9μ2³) / N_eff
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

        # Quantiles at the shipped point estimates q̂.
        for (sym, x, w, p) in ((:p10_wage_U, xu, wu, 0.10), (:p10_wage_S, xs, ws, 0.10),
                               (:p25_wage_U, xu, wu, 0.25), (:p25_wage_S, xs, ws, 0.25),
                               (:p50_wage_U, xu, wu, 0.50), (:p50_wage_S, xs, ws, 0.50),
                               (:p75_wage_U, xu, wu, 0.75), (:p75_wage_S, xs, ws, 0.75),
                               (:p90_wage_U, xu, wu, 0.90), (:p90_wage_S, xs, ws, 0.90))
            qhat = get(moment_vals, sym, NaN)
            svar[sym] = _wquantile_var(x, w, qhat, p)
        end

        # wage_premium = mean_S − mean_U on disjoint subsamples: variances add.
        svar[:wage_premium] = (isfinite(vmU) && isfinite(vmS)) ? vmU + vmS : NaN

        # Overlap: binomial term plus the propagated error of the estimated
        # median of the other market, e.g. for overlap_UgtS = P(logw_U > m_S):
        #   q(1−q)/N_U + [f_U(m_S)/f_S(m_S)]²·(1/4)/N_S
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

        # (c) Flow hazards: delta method with the at-risk Kish N summed over months.
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

        # (d) θ = V/U: Var ≈ θ²(1−p)/(p N_lf), p = ur_j, N_lf the Kish N of the
        #     market's labour force; V is treated as measured.
        for (mname, ur_name, sub) in ((:theta_U, :ur_U, .!skl), (:theta_S, :ur_S, skl))
            th = get(moment_vals, mname, NaN); p = get(moment_vals, ur_name, NaN)
            wlf = cw[inlf .& sub]
            N_lf = sum(wlf) > 0 ? sum(wlf)^2 / sum(abs2, wlf) : 0.0
            svar[mname] = (isfinite(th) && isfinite(p) && 0.0 < p < 1.0 && N_lf > 0.0) ?
                          th^2 * (1.0 - p) / (p * N_lf) : NaN
        end

        # (e) SIPP and J2J moments, built above.
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

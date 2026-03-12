############################################################
# moments.jl — Empirical moment targets
#
# TODO: replace placeholder values with data-based estimates
#       once the data pipeline is ready.
#
# Returns a NamedTuple of (value, weight) pairs.
# Weight = 1 / variance of the moment estimator (or a proxy).
# Higher weight = more tightly targeted.
#
# Wage premium convention
# ───────────────────────
# wage_premium  ≡  E[log w_S] − E[log w_U]
#
# This is the standard Mincer / OLS skill-premium: scale-invariant,
# directly estimable from micro data, and orthogonal to the level
# moments mean_wage_U / mean_wage_S already in the objective.
# Do NOT use E[log w_S]/E[log w_U]: with wages normalised below 1
# the logs are negative and the ratio is non-monotone in the premium.
# Do NOT use E[w_S]/E[w_U]: collinear with the two level means.
############################################################


"""
    load_data_moments() → NamedTuple

Returns the empirical moment targets used in SMM estimation.
Each field is a (value, weight) tuple.

Moment list
───────────────────────────────────────────────────────────
  Labour market stocks
    ur_total          overall unemployment rate
    ur_U              unskilled unemployment rate
    ur_S              skilled unemployment rate
    skilled_share     share of population in skilled segment
    training_share    share of population in training
    emp_var_U         variance of wages among unskilled employed
    emp_cm3_U         third central moment of wages, unskilled employed
    emp_var_S         variance of wages among skilled employed
    emp_cm3_S         third central moment of wages, skilled employed

  Transition rates
    jfr_U             unskilled job-finding rate
    sep_rate_U        unskilled separation rate
    jfr_S             skilled job-finding rate
    sep_rate_S        skilled (endogenous+exogenous) separation rate
    training_rate     flow into training per unskilled unemployed

  Wages
    mean_wage_U       mean wage, unskilled employed
    mean_wage_S       mean wage, skilled employed
    p50_wage_U        median wage, unskilled
    p50_wage_S        median wage, skilled
    wage_premium      E[log w_S] − E[log w_U]  (log skill premium)
    wage_sd_U         std dev of unskilled wages
    wage_sd_S         std dev of skilled wages

  Tightness (if vacancy data available)
    theta_U           unskilled vacancy-unemployment ratio
    theta_S           skilled vacancy-unemployment ratio
───────────────────────────────────────────────────────────
"""
function load_data_moments()

    # ── PLACEHOLDER VALUES ───────────────────────────────────────────────
    # Replace each `value` with the empirically estimated moment.
    # Replace each `weight` with 1/var or a tuned scalar.
    # Set weight = 0.0 to exclude a moment from the objective.
    # ─────────────────────────────────────────────────────────────────────

    return (
        # ── Labour market stocks ─────────────────────────────────────────
        ur_total       = (value = 0.050,  weight = 100.0),
        ur_U           = (value = 0.080,  weight =  80.0),
        ur_S           = (value = 0.025,  weight =  80.0),
        skilled_share  = (value = 0.450,  weight =  90.0),
        training_share = (value = 0.020,  weight =  40.0),
        emp_var_U      = (value = 0.050,  weight =  20.0),
        emp_cm3_U      = (value = 0.005,  weight =  10.0),
        emp_var_S      = (value = 0.120,  weight =  20.0),
        emp_cm3_S      = (value = 0.015,  weight =  10.0),

        # ── Transition rates ─────────────────────────────────────────────
        jfr_U          = (value = 0.220,  weight =  35.0),
        sep_rate_U     = (value = 0.025,  weight =  25.0),
        jfr_S          = (value = 0.140,  weight =  35.0),
        sep_rate_S     = (value = 0.010,  weight =  15.0),
        training_rate  = (value = 0.040,  weight =  15.0),

        # ── Wages ────────────────────────────────────────────────────────
        mean_wage_U    = (value = 0.700,  weight =  40.0),
        mean_wage_S    = (value = 1.250,  weight =  30.0),
        p50_wage_U     = (value = 0.660,  weight =  30.0),
        p50_wage_S     = (value = 1.180,  weight =  20.0),
        wage_premium   = (value = 0.580,  weight =  45.0),   # E[log w_S] − E[log w_U]; ≈ log(1.25/0.70) ≈ 0.58
        wage_sd_U      = (value = 0.220,  weight =  10.0),
        wage_sd_S      = (value = 0.350,  weight =  10.0),

        # ── Tightness ────────────────────────────────────────────────────
        theta_U        = (value = 0.700,  weight =  15.0),
        theta_S        = (value = 1.400,  weight =  15.0),
    )
end


"""
    model_moments(obj) → NamedTuple

Extract the same set of moments from a solved model's equilibrium objects.
`obj` is the NamedTuple returned by `compute_equilibrium_objects`.

Prerequisites — the following fields must be added to the return tuple of
`compute_equilibrium_objects` (see note at bottom of this file):
    f_S         :: Float64             # κ_S = θ_S · q_S(θ_S)
    sep_rate_U  :: Float64             # employment-weighted unskilled separation rate
    sep_rate_S  :: Float64             # employment-weighted skilled separation rate
"""
function model_moments(obj)

    # ── Labour market stocks ──────────────────────────────────────────────
    ur_total       = obj.ur_total
    ur_U           = obj.ur_U
    ur_S           = obj.ur_S
    skilled_share  = obj.agg_mS  / max(obj.total_pop, 1e-14)
    training_share = obj.agg_t   / max(obj.total_pop, 1e-14)

    # emp_var / emp_cm3: variance and third central moment of the employed
    # wage distribution, computed from the same density grids used for
    # mean_wage and wage_sd below.  Evaluated here so they sit logically
    # with the other labour-stock moments; the densities are re-used later.
    wmid_tmp   = obj.wmid
    dens_U_tmp = obj.dens_U
    dens_S_tmp = obj.dens_S
    bw_tmp     = wmid_tmp[2] - wmid_tmp[1]

    _mean_U_tmp = sum(wmid_tmp .* dens_U_tmp) * bw_tmp
    _mean_S_tmp = sum(wmid_tmp .* dens_S_tmp) * bw_tmp

    emp_var_U  = sum((wmid_tmp .- _mean_U_tmp).^2 .* dens_U_tmp) * bw_tmp
    emp_cm3_U  = sum((wmid_tmp .- _mean_U_tmp).^3 .* dens_U_tmp) * bw_tmp
    emp_var_S  = sum((wmid_tmp .- _mean_S_tmp).^2 .* dens_S_tmp) * bw_tmp
    emp_cm3_S  = sum((wmid_tmp .- _mean_S_tmp).^3 .* dens_S_tmp) * bw_tmp

    # ── Transition rates ──────────────────────────────────────────────────
    # jfr_U: θ_U · q_U(θ_U), already stored as f_U in obj
    jfr_U         = obj.f_U

    # jfr_S: θ_S · q_S(θ_S), stored as f_S (added to compute_equilibrium_objects)
    jfr_S         = obj.f_S

    # sep_rate_U: employment-weighted λ_U · G(p*(x)) = λ_U · p*(x)^α_U
    # sep_rate_S: employment-weighted ξ_S + λ_S · Γ(p*_S(x))
    # Both pre-computed in compute_equilibrium_objects (see note below)
    sep_rate_U    = obj.sep_rate_U
    sep_rate_S    = obj.sep_rate_S

    # training_rate: flow into training per unskilled unemployed
    training_rate = obj.agg_t / max(obj.agg_uU, 1e-14)

    # ── Wages ─────────────────────────────────────────────────────────────
    # dens_U and dens_S are employment-mass-weighted wage densities built in
    # compute_equilibrium_objects. Only (x,p) cells with e > 1e-16 contribute,
    # so the means and medians below are automatically over employed workers only.
    # The skilled density pools s=0 and s=1 workers (correct: both are employed).
    wmid   = obj.wmid
    dens_U = obj.dens_U
    dens_S = obj.dens_S
    bw     = wmid[2] - wmid[1]

    mean_wage_U = sum(wmid .* dens_U) * bw
    mean_wage_S = sum(wmid .* dens_S) * bw

    # Median: first bin where cumulative density crosses 0.5
    function _median(wmid, dens, bw)
        cum = 0.0
        for (w, d) in zip(wmid, dens)
            cum += d * bw
            cum >= 0.5 && return w
        end
        return wmid[end]
    end

    p50_wage_U = _median(wmid, dens_U, bw)
    p50_wage_S = _median(wmid, dens_S, bw)

    # Wage premium: E[log w_S] − E[log w_U]
    #
    # This is the log skill premium — the same object estimated by a
    # Mincer regression of log wages on a skill indicator.  It is
    # scale-invariant (unaffected by wage normalisation) and adds
    # identifying information orthogonal to the two level means above.
    #
    # clamp guards against any zero/negative bin midpoints on the wage grid.
    mean_log_wage_U = sum(log.(max.(wmid, 1e-14)) .* dens_U) * bw
    mean_log_wage_S = sum(log.(max.(wmid, 1e-14)) .* dens_S) * bw
    wage_premium    = mean_log_wage_S - mean_log_wage_U

    var_U     = sum((wmid .- mean_wage_U).^2 .* dens_U) * bw
    var_S     = sum((wmid .- mean_wage_S).^2 .* dens_S) * bw
    wage_sd_U = sqrt(max(var_U, 0.0))
    wage_sd_S = sqrt(max(var_S, 0.0))

    # ── Tightness ─────────────────────────────────────────────────────────
    theta_U = obj.thetaU
    theta_S = obj.thetaS

    return (
        ur_total      = ur_total,
        ur_U          = ur_U,
        ur_S          = ur_S,
        skilled_share = skilled_share,
        training_share = training_share,
        emp_var_U     = emp_var_U,
        emp_cm3_U     = emp_cm3_U,
        emp_var_S     = emp_var_S,
        emp_cm3_S     = emp_cm3_S,

        jfr_U         = jfr_U,
        sep_rate_U    = sep_rate_U,
        jfr_S         = jfr_S,
        sep_rate_S    = sep_rate_S,
        training_rate = training_rate,

        mean_wage_U   = mean_wage_U,
        mean_wage_S   = mean_wage_S,
        p50_wage_U    = p50_wage_U,
        p50_wage_S    = p50_wage_S,
        wage_premium  = wage_premium,
        wage_sd_U     = wage_sd_U,
        wage_sd_S     = wage_sd_S,

        theta_U       = theta_U,
        theta_S       = theta_S,
    )
end

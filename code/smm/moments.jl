############################################################
# moments.jl — Empirical moment targets
#
# TODO: replace placeholder values with data-based estimates
#       once the data pipeline is ready.
#
# Returns a NamedTuple of (value, weight) pairs.
# Weight = 1 / variance of the moment estimator (or a proxy).
# Higher weight = more tightly targeted.
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
    wage_premium      log skilled / unskilled mean wage ratio
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
        ur_total       = (value = 0.060,  weight = 100.0),
        ur_U           = (value = 0.090,  weight =  80.0),
        ur_S           = (value = 0.030,  weight =  80.0),
        skilled_share  = (value = 0.400,  weight =  50.0),
        training_share = (value = 0.050,  weight =  40.0),

        # ── Transition rates ─────────────────────────────────────────────
        jfr_U          = (value = 0.300,  weight =  60.0),
        sep_rate_U     = (value = 0.030,  weight =  60.0),
        jfr_S          = (value = 0.200,  weight =  60.0),
        sep_rate_S     = (value = 0.020,  weight =  60.0),
        training_rate  = (value = 0.100,  weight =  30.0),

        # ── Wages ────────────────────────────────────────────────────────
        mean_wage_U    = (value = 0.600,  weight =  0.0),
        mean_wage_S    = (value = 1.100,  weight =  0.0),
        p50_wage_U     = (value = 0.570,  weight =  0.0),
        p50_wage_S     = (value = 1.050,  weight =  0.0),
        wage_premium   = (value = 0.600,  weight =  0.0),   # log ratio
        wage_sd_U      = (value = 0.150,  weight =  0.0),
        wage_sd_S      = (value = 0.250,  weight =  0.0),

        # ── Tightness ────────────────────────────────────────────────────
        theta_U        = (value = 1.000,  weight =   0.0),   # excluded for now
        theta_S        = (value = 1.000,  weight =   0.0),   # excluded for now
    )
end


"""
    model_moments(obj) → NamedTuple

Extract the same set of moments from a solved model's equilibrium objects.
`obj` is the NamedTuple returned by `compute_equilibrium_objects`.
"""
function model_moments(obj)

    # ── Convenience ──────────────────────────────────────────────────────
    wx  = obj.wx
    wpU = obj.wpU
    wpS = obj.wpS

    # ── Labour market stocks ──────────────────────────────────────────────
    ur_total      = obj.ur_total
    ur_U          = obj.ur_U
    ur_S          = obj.ur_S
    skilled_share = obj.agg_mS / max(obj.total_pop, 1e-14)
    training_share = obj.agg_t / max(obj.total_pop, 1e-14)

    # ── Transition rates ──────────────────────────────────────────────────
    jfr_U     = obj.f_U

    θS        = obj.thetaS
    # (f_S not stored directly in obj — recompute from thetaS stored there)
    # We use the fact that obj.thetaS and the skilled matching params are
    # accessible indirectly; for now approximate as the mean hiring rate.
    jfr_S     = obj.thetaS  # placeholder — replace with f_S once stored in obj

    # Separation rate: δ_U = λ_U * E[G(p*(x))]  weighted by employment
    eU_total  = max(obj.agg_eU, 1e-14)
    sep_rate_U = 0.0          # TODO: compute from model internals if needed
    sep_rate_S = 0.0          # TODO

    training_rate = obj.agg_t / max(obj.agg_uU, 1e-14)

    # ── Wages ─────────────────────────────────────────────────────────────
    # Wage densities are already computed in obj
    wmid    = obj.wmid
    dens_U  = obj.dens_U
    dens_S  = obj.dens_S

    bw = wmid[2] - wmid[1]

    mean_wage_U = sum(wmid .* dens_U) * bw
    mean_wage_S = sum(wmid .* dens_S) * bw

    # Median: first bin where cumulative density >= 0.5
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

    wage_premium = log(max(mean_wage_S, 1e-14)) - log(max(mean_wage_U, 1e-14))

    var_U = sum((wmid .- mean_wage_U).^2 .* dens_U) * bw
    var_S = sum((wmid .- mean_wage_S).^2 .* dens_S) * bw

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

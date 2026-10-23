############################################################
# moments.jl — Empirical moment targets
#
# Loads moments from the CSV files produced by the data pipeline and
# computes the matching model moments from a solved equilibrium.
#
# Wage premium convention
#   wage_premium  ≡  E[log w_S] − E[log w_U]
#
# load_calibrated_params reads nu_estimation.csv (two rows) and accepts
# a `window` so each crisis pair gets the ν of its own baseline
# (FC pair → ν_base_fc; COVID pair → ν_base_covid).
#
# training_share target (data pipeline v17.0):
#   The CPS SCHLCOLL universe expanded in Jan 2013 (16–24 → 16–54), so the
#   raw CPS training_share level is not comparable across the FC and COVID
#   window pairs and is NOT used. The target is read directly:
#       target  =  4-year undergraduate enrolment_w / CPS_pop_w
#   where f is the completers' share of the enrolment stock (the model's
#   trainee state has no dropout exit). Its variance is the across-Fall-year
#   dispersion of that series. Both are written by the data pipeline into
#   moments_{window}.csv and sampling_var_{window}.csv; the loaders below
#   read them verbatim and apply no scaling of their own.
#
#   The earlier enr_scale_w bridge (enr_scale_w · CPS_training_share) was
#   removed: the CPS enrolment count was meant to cancel, but stopped cancelling
#   once the moment's numerator was restricted to NILF trainees while the
#   scale's denominator stayed on all enrolled, leaving a participation ratio
#   that drifted ~6% within each crisis pair.
############################################################


# ============================================================
# Moment names (canonical order — 35 moments)
#
# Identical ordering to data_processing/setup.jl. The cross-market
# overlap pair (overlap_UgtS, overlap_SltU) and the skilled long-term-
# unemployment share (ltu_share_S) are appended at the END so the
# moment indices line up with the data-side moment vector.
# ============================================================

# The unemployment-duration profile is derived from the solver's threshold list so
# the model counterparts and these names cannot drift apart; the data-side copy in
# code/data_and_descriptives/setup.jl is the one to edit if the thresholds change.
const DURATION_MOMENT_NAMES =
    [Symbol("usurv$(Int(t))_$(m)") for m in ("U", "S") for t in DURATION_THRESHOLDS_WK]

const MOMENT_NAMES = [
    :ur_total, :ur_U, :ur_S,
    :skilled_share, :training_share,
    :emp_var_U, :emp_cm3_U, :emp_var_S, :emp_cm3_S,
    :jfr_U, :sep_rate_U, :jfr_S, :sep_rate_S,
    :ee_rate_S,
    :mean_wage_U, :mean_wage_S,
    :p10_wage_U, :p10_wage_S, :p25_wage_U, :p25_wage_S, :p50_wage_U, :p50_wage_S,
    :p75_wage_U, :p75_wage_S, :p90_wage_U, :p90_wage_S,
    :wage_premium, :theta_U, :theta_S,
    :overlap_UgtS, :overlap_SltU, :ltu_share_S,
    :wchg_rate_U, :wchg_rate_S,
    :ee_step_S,
    DURATION_MOMENT_NAMES...,
]

const N_MOMENTS_BASE = 35

@assert length(MOMENT_NAMES) == N_MOMENTS_BASE + length(DURATION_MOMENT_NAMES) """
Expected $(N_MOMENTS_BASE + length(DURATION_MOMENT_NAMES)) moments, \
got $(length(MOMENT_NAMES))"""


"""
    load_training_share_scale(; window, derived_dir)

REMOVED in v17.0. The training_share target no longer carries an enr_scale_w
level adjustment: it is read directly (see the header). This stub remains
so that any caller still expecting the old behaviour fails immediately and
visibly, rather than silently defaulting to a scale of 1.0 and producing an estimate
against a target that is not the one the data pipeline shipped.
"""
function load_training_share_scale(; window::Symbol, derived_dir::String) :: Float64
    error("""
    load_training_share_scale was removed in v17.0.

    training_share is no longer rescaled. moments_$(window).csv already holds
    the enrolment target and sampling_var_$(window).csv its across-Fall-year variance;
    read them directly. A stale derived/training_share_scale.csv may still be on
    disk from an earlier pipeline run — it is not an input to anything and can
    be deleted.
    """)
end


"""
    load_data_moments(; window::Symbol = :base_fc, derived_dir::String) → NamedTuple

Return the empirical moment targets used in SMM estimation.  Each
field is a (value, weight) tuple.  Reads from
`moments_{window}.csv` produced by the data pipeline; a counterfactual window reads its
parent's (`data_window`, paths.jl), and so do its sampling variances and calibrated rates.

The training_share row already carries its level from the enrolment target,
written in Stage 9 of the data pipeline, so this returns the moments exactly as
they appear in `moments_{window}.csv` with no further rescaling.

Moment list
  Labour-market stocks (5)
    ur_total, ur_U, ur_S, skilled_share, training_share

  Wage shape (4)
    emp_var_U, emp_cm3_U, emp_var_S, emp_cm3_S

  Transition rates (5)
    jfr_U, sep_rate_U, jfr_S, sep_rate_S, ee_rate_S

  Wages (13)
    mean_wage_U, mean_wage_S,
    p10_wage_U, p10_wage_S, p25_wage_U, p25_wage_S,
    p50_wage_U, p50_wage_S, p75_wage_U, p75_wage_S,
    p90_wage_U, p90_wage_S,
    wage_premium

  Tightness (2)
    theta_U, theta_S

  Cross-market overlap and duration (3)
    overlap_UgtS, overlap_SltU, ltu_share_S
"""
function load_data_moments(; window::Symbol = :base_fc, derived_dir::String)
    moments_file = joinpath(derived_dir, "moments_$(data_window(window)).csv")
    isfile(moments_file) || error("Moments file not found: $moments_file — run the data pipeline first.")
    raw = _read_moments_csv(moments_file)

    # training_share already carries its level from the enrolment target, applied
    # upstream in the data pipeline (Stage 7). Return the moments as read.
    return raw
end


"""
    run_weighting() -> Symbol

The objective's weighting, `:diagonal` or `:equal` (see `load_weight_matrix`), from
`ROYSEARCH_WEIGHTING`. Every entry point reads it here — the estimation, the chains, the
transition, the policy run and the tables — so all of them name and read the same files.
Diagonal is the default: every stored estimate was produced under it.
"""
run_weighting() = env_setting(:WEIGHTING, :diagonal)

"""
    weighting_suffix(weighting) -> String

Filename suffix of a weighting, `_diagonalW` or `_equalW`, so the estimates under the two
never overwrite one another.
"""
function weighting_suffix(weighting::Symbol)
    weighting === :diagonal && return "_diagonalW"
    weighting === :equal    && return "_equalW"
    error("WEIGHTING must be :diagonal or :equal; got :$weighting.")
end

"""
    load_sampling_variances(; window, derived_dir) → Dict{Symbol,Float64}

The per-moment sampling variances σ̂²_k of `sampling_var_{window}.csv` (the parent's for a
counterfactual window), in raw moment units. They are the diagonal weights of the objective
and, as σ̂_k, the data standard errors the tables and panels report.
"""
function load_sampling_variances(; window::Symbol, derived_dir::String)
    sv_name = "sampling_var_$(data_window(window)).csv"
    sv_file = joinpath(derived_dir, sv_name)
    isfile(sv_file) || error("$sv_name not found in $derived_dir — run the data pipeline first.")
    df_sv = CSV.read(sv_file, DataFrame)
    return Dict(Symbol(r.moment) => Float64(r.sampling_var) for r in eachrow(df_sv))
end

"""
    load_weight_matrix(; window, derived_dir, weighting,
                         skip_moments) → Union{Nothing, Matrix{Float64}}

Build the SMM weighting matrix over the active moments (i.e. those in
`MOMENT_NAMES` that are not in `skip_moments`):
  :diagonal  W = Diagonal(1 / σ̂²_k), σ̂²_k the sampling variances of
             `load_sampling_variances` (all moments on one 1/N footing)
  :equal     returns `nothing`; `build_smm_spec` then builds the
             relative-deviation weights Diagonal(weight_k / m̂_k²)

The σ̂²_k are in RAW moment units, matching the raw deviation vector
g_k = m_k − m̂_k in compute_loss_matrix (all scaling lives in W).
"""
function load_weight_matrix(;
    window::Symbol = :base_fc,
    derived_dir::String,
    weighting::Symbol = :diagonal,
    skip_moments::Vector{Symbol} = Symbol[],
)
    K = length(MOMENT_NAMES)

    unknown = setdiff(skip_moments, MOMENT_NAMES)
    if !isempty(unknown)
        @printf("  load_weight_matrix: skip_moments — unrecognised names (ignored): %s\n",
                join(string.(unknown), ", "))
    end
    active_names = [nm for nm in MOMENT_NAMES if !(nm in skip_moments)]
    K_active     = length(active_names)
    if K_active < K
        @printf("  load_weight_matrix: subsetting to %d / %d active moments (skipping: %s)\n",
                K_active, K, join(string.(skip_moments), ", "))
    end

    if weighting === :equal
        @printf("  weighting = equal → no W matrix\n")
        return nothing
    end
    weighting === :diagonal || error("WEIGHTING must be :diagonal or :equal; got :$weighting.")

    # A moment without a positive variance cannot be weighted: error loudly rather than
    # drop it from the objective.
    sv  = load_sampling_variances(; window = window, derived_dir = derived_dir)
    d   = Float64[]
    bad = Symbol[]
    for nm in active_names
        v = get(sv, nm, NaN)
        (isfinite(v) && v > 0.0) ? push!(d, 1.0 / v) : (push!(d, NaN); push!(bad, nm))
    end
    isempty(bad) || error(
        "sampling_var_$(data_window(window)).csv has non-positive / missing variance for active " *
        "moment(s): " * join(string.(bad), ", ") * " — cannot build the diagonal weight.")
    @printf("  weighting = diagonal → W = Diagonal(1/σ̂²_samp) from sampling_var_%s.csv\n",
            data_window(window))
    return Matrix(Diagonal(d))
end


# ============================================================
# Externally calibrated parameters from the data pipeline
# ============================================================

"""
    load_calibrated_params(; window::Symbol = :base_fc,
                            derived_dir::String) → NamedTuple

Load the three externally calibrated parameters held fixed during
SMM estimation.

  r   = 0.05 / 12              monthly discount rate (5% annual)
  ν   from nu_estimation.csv   demographic turnover (life-table)
                                The FC pair (base_fc, crisis_fc) uses
                                the base_fc row; the COVID pair
                                (base_covid, crisis_covid) uses the
                                base_covid row.
  φ   from phi_calibration.csv training completion rate (nominal four-year programme,
                                pooled across Fall semesters)

`window` controls which ν is returned. Every window is mapped to
its baseline (`PAIR_BASE`, paths.jl: crisis_fc → base_fc, crisis_covid → base_covid, and
each counterfactual window to its parent's) per data_and_moments.pdf §21.
"""
function load_calibrated_params(; window::Symbol = :base_fc,
                                  derived_dir::String)
    r_val = 0.05 / 12

    # Which baseline supplies ν for this window
    haskey(PAIR_BASE, window) || error("load_calibrated_params: unrecognised window :$window. " *
                                       "Expected one of $(join(WINDOWS, ", ")).")
    nu_pair = PAIR_BASE[window]

    nu_file = joinpath(derived_dir, "nu_estimation.csv")
    isfile(nu_file) || error(
        "nu_estimation.csv not found in $derived_dir — run the data pipeline first.")
    df_nu = CSV.read(nu_file, DataFrame)
    df_nu.window = Symbol.(df_nu.window)
    rows = filter(:window => ==(nu_pair), df_nu)
    isempty(rows) && error(
        "nu_estimation.csv has no row for window=$nu_pair. " *
        "Expected one row each for :base_fc and :base_covid.")
    nu_val = Float64(rows.nu[1])
    @printf("  Loaded nu = %.5f from %s  (window=%s → uses %s row)\n",
            nu_val, nu_file, window, nu_pair)

    phi_file = joinpath(derived_dir, "phi_calibration.csv")
    isfile(phi_file) || error(
        "phi_calibration.csv not found in $derived_dir — run the data pipeline first.")
    df_phi = CSV.read(phi_file, DataFrame)
    phi_val = Float64(df_phi.phi[1])
    @printf("  Loaded phi = %.5f from %s\n", phi_val, phi_file)

    return (r = r_val, nu = nu_val, phi = phi_val)
end


# ============================================================
# Windows.json loader (single source of truth from the data pipeline)
# ============================================================

"""
    load_windows(; derived_dir::String) → NamedTuple

Read `windows.json` (written by the data pipeline) and return
`(windows = Dict{Symbol,NamedTuple}, order = Vector{Symbol})`.
"""
function load_windows(; derived_dir::String)
    win_path = joinpath(derived_dir, "windows.json")
    isfile(win_path) || error(
        "windows.json not found in $derived_dir — run the data pipeline first.")
    raw = JSON3.read(read(win_path, String))
    wins = Dict{Symbol, NamedTuple}()
    for (k, v) in raw["windows"]
        wins[Symbol(k)] = (
            label      = String(v["label"]),
            ym_start   = Int(v["ym_start"]),
            ym_end     = Int(v["ym_end"]),
            asec_years = Int(first(v["asec_years"])):Int(last(v["asec_years"])),
        )
    end
    order = Symbol.(raw["windows_order"])
    return (windows = wins, order = order)
end


# ============================================================
# CSV helpers
# ============================================================

"""
    _read_moments_csv(filepath) → NamedTuple

Read `moments_{window}.csv` and return a NamedTuple mapping
moment name → (value, weight).  Only rows whose `moment` field
appears in `MOMENT_NAMES` are kept; stale rows are silently
filtered out with a one-line summary.

Weight is set to 1.0 here; effective weights come from the W matrix.
"""
function _read_moments_csv(filepath::String)
    df = CSV.read(filepath, DataFrame)

    pairs = Pair{Symbol, NamedTuple{(:value, :weight), Tuple{Float64, Float64}}}[]
    moment_name_set = Set(MOMENT_NAMES)
    n_skipped = 0
    skipped_names = Symbol[]

    for row in eachrow(df)
        name = Symbol(row.moment)
        if !(name in moment_name_set)
            n_skipped += 1
            push!(skipped_names, name)
            continue
        end
        val = Float64(row.value)
        push!(pairs, name => (value = val, weight = 1.0))
    end

    if n_skipped > 0
        @printf("  _read_moments_csv: skipped %d row(s) not in MOMENT_NAMES: %s\n",
                n_skipped, join(string.(unique(skipped_names)), ", "))
    end

    return NamedTuple(pairs)
end


# ============================================================
# Model-side moments
# ============================================================

# ── Log-wage moments with lognormal measurement error ──────────────────────
# Observed log wage = structural log wage + N(0, σ_w²).  A market's structural
# distribution is its wage points (`wage_points` of compute_equilibrium_objects), one
# (wage, mass) pair per employed cell, every employed worker counted once.  The error
# leaves the mean and the third central moment unchanged (it is symmetric) and adds σ_w²
# to the variance; the percentiles and the overlaps are read off the σ_w-convolved
# distribution F(y) = Σ_k q_k Φ((y − log w_k)/σ_w).  Every moment is evaluated on the
# points themselves, with no histogram in between, so each is a continuous function of
# the wages and masses and carries no discretisation error of its own.

"""
    _logwage_distribution(pts, σ_w) → NamedTuple or nothing

One market's log-wage distribution from its wage points `pts = (w, m)`: the log wages
and normalised masses, mean, variance including σ_w², and third central moment.
`nothing` for a market that employs no one.
"""
function _logwage_distribution(pts, σ_w::Real)
    M = sum(pts.m)
    M < 1e-12 && return nothing
    logw = log.(max.(pts.w, 1e-14))
    q    = pts.m ./ M
    μ    = dot(q, logw)
    dev  = logw .- μ
    v0   = dot(q, dev .^ 2)
    return (logw = logw, q = q, σ_w = float(σ_w),
            mean = μ, var = v0 + σ_w^2, cm3 = dot(q, dev .^ 3), sd = sqrt(v0 + σ_w^2))
end

# The convolved CDF at log wage y and its density; with σ_w → 0 the step CDF of the points.
function _conv_cdf_pdf(d, y::Real)
    F = 0.0;  f = 0.0
    if d.σ_w > 1e-10
        @inbounds for k in eachindex(d.logw)
            z  = (y - d.logw[k]) / d.σ_w
            F += d.q[k] * cdf(Normal(), z);  f += d.q[k] * pdf(Normal(), z)
        end
        f /= d.σ_w
    else
        @inbounds for k in eachindex(d.logw)
            d.logw[k] <= y && (F += d.q[k])
        end
    end
    return F, f
end
_conv_cdf(d, y::Real) = first(_conv_cdf_pdf(d, y))

# Quantile of the convolved distribution: Newton on F from the normal approximation,
# kept inside the bracket that F's monotonicity maintains (bisection whenever a Newton
# step would leave it).  F is smooth with a positive density, so this converges
# quadratically; it stops once the Newton step is below 1e-13 in log w, tested before the
# bracket so that a converged step is never mistaken for one leaving it.  With σ_w → 0,
# the discrete quantile of the points.
function _conv_quantile(d, target::Real)
    if d.σ_w <= 1e-10
        o = sortperm(d.logw);  cum = 0.0
        for k in o
            cum += d.q[k];  cum >= target && return d.logw[k]
        end
        return d.logw[o[end]]
    end
    a = minimum(d.logw) - 8.0 * d.σ_w;  b = maximum(d.logw) + 8.0 * d.σ_w
    y = clamp(d.mean + d.sd * quantile(Normal(), target), a, b)
    for _ in 1:100
        F, f = _conv_cdf_pdf(d, y)
        F < target ? (a = y) : (b = y)
        step = f > 0.0 ? (F - target) / f : Inf
        abs(step) < 1e-13 && return y - step
        y_new = y - step
        y = a < y_new < b ? y_new : 0.5 * (a + b)
    end
    return y
end

# (mean, variance, third moment, p10, p25, p50, p75, p90) of log wages; all 0 for a market
# that employs no one.
function _logwage_moments(d)
    d === nothing && return ntuple(_ -> 0.0, 8)
    return (d.mean, d.var, d.cm3, _conv_quantile(d, 0.10), _conv_quantile(d, 0.25),
            _conv_quantile(d, 0.50), _conv_quantile(d, 0.75), _conv_quantile(d, 0.90))
end

"""
    model_moments(obj) → NamedTuple

Extract the targeted moments from a solved model's equilibrium
objects.  `obj` is the NamedTuple returned by
`compute_equilibrium_objects`.

A market that employs no one has no wage distribution: its wage
moments, the wage premium and both overlaps are 0.
"""
function model_moments(obj)
    # Every within-market object arrives defined, a market that holds no one included:
    # compute_equilibrium_objects reads it at the limit of its composition as it empties.
    # So the moments are read as they are.  A market with no vacancies employs no one even
    # in that limit; its wage moments, and both overlaps, are 0.

    # Labour-market stocks.
    ur_U           = obj.ur_U
    ur_S           = obj.ur_S
    ur_total       = obj.ur_total
    _model_lf      = obj.agg_uU + obj.agg_eU + obj.agg_mS
    skilled_share  = obj.agg_mS  / max(_model_lf, 1e-14)
    training_share = obj.agg_t   / max(obj.total_pop, 1e-14)

    # Log-wage measurement-error SDs, carried from the equilibrium params.
    # Wage moments (mean / variance / third moment / percentiles) are computed
    # in LOG wages below, with lognormal measurement error of SD σ_w.
    σ_wU = hasproperty(obj, :σ_wU) ? obj.σ_wU : 0.0
    σ_wS = hasproperty(obj, :σ_wS) ? obj.σ_wS : 0.0

    # Transition rates and the within-job wage-change rates (SIPP), each built as the
    # data stage measures it: the mean transition probability over the moment's
    # universe, converted as its source converts it (compute_equilibrium_objects,
    # `measured`).
    jfr_U       = obj.measured.jfr_U
    jfr_S       = obj.measured.jfr_S
    sep_rate_U  = obj.measured.sep_rate_U
    sep_rate_S  = obj.measured.sep_rate_S
    ee_rate_S   = obj.measured.ee_rate_S
    wchg_rate_U = obj.measured.wchg_rate_U
    wchg_rate_S = obj.measured.wchg_rate_S

    # Skilled EE-move mean wage step (SIPP moment). Read from the equilibrium;
    # 0.0 when the field is absent.
    ee_step_S = hasproperty(obj, :ee_step_S) ? obj.ee_step_S : 0.0

    # Wages — every moment on LOG wages with lognormal measurement error σ_w: the
    # "mean_wage_*", "emp_var_*", "emp_cm3_*" and "p10…p90_wage_*" moments are the log-wage
    # objects, and the data side computes them on log(wage_norm).
    dist_U = _logwage_distribution(obj.wage_points.U, σ_wU)
    dist_S = _logwage_distribution(obj.wage_points.S, σ_wS)
    (mean_wage_U, emp_var_U, emp_cm3_U,
     p10_wage_U, p25_wage_U, p50_wage_U, p75_wage_U, p90_wage_U) = _logwage_moments(dist_U)
    (mean_wage_S, emp_var_S, emp_cm3_S,
     p10_wage_S, p25_wage_S, p50_wage_S, p75_wage_S, p90_wage_S) = _logwage_moments(dist_S)

    wage_premium = mean_wage_S - mean_wage_U

    # Cross-market wage overlap (one moment per bargaining weight).  Each is
    # a tail probability of one segment's σ_w-convolved log-wage distribution
    # evaluated at the OTHER segment's convolved median:
    #   overlap_UgtS = P(log w_U > med log w_S) = 1 − F_U^conv(p50_wage_S)
    #   overlap_SltU = P(log w_S < med log w_U) =     F_S^conv(p50_wage_U)
    both = !isnothing(dist_U) && !isnothing(dist_S)
    overlap_UgtS = both ? 1.0 - _conv_cdf(dist_U, p50_wage_S) : 0.0
    overlap_SltU = both ? _conv_cdf(dist_S, p50_wage_U)       : 0.0

    # Skilled long-term-unemployment share: survival of a skilled spell past
    # a* ≈ 6.23 months under the type-mixture of exponential exit hazards.
    # Computed in compute_equilibrium_objects (it needs uS, Γ(p*_S), κ_S, wx);
    # read it through here. Absent from objects that predate it ⇒ 0.
    ltu_share_S = hasproperty(obj, :ltu_share_S) ? obj.ltu_share_S : 0.0

    # Duration profile: the solver returns one survivor vector per market, ordered
    # as DURATION_THRESHOLDS_WK. hasproperty keeps this readable against equilibrium
    # objects produced before v21.2.0, which carry ltu_share_S but no survivor vectors.
    _usurv = fld -> hasproperty(obj, fld) ? collect(getproperty(obj, fld)) :
                    fill(0.0, length(DURATION_THRESHOLDS_WK))
    _usurv_U = _usurv(:usurv_U)
    _usurv_S = _usurv(:usurv_S)
    _duration_vals = NamedTuple{Tuple(DURATION_MOMENT_NAMES)}(
        Tuple(vcat(_usurv_U, _usurv_S)))

    # Tightness
    theta_U = obj.thetaU
    theta_S = obj.thetaS

    return (
        ur_total       = ur_total,
        ur_U           = ur_U,
        ur_S           = ur_S,
        skilled_share  = skilled_share,
        training_share = training_share,
        emp_var_U      = emp_var_U,
        emp_cm3_U      = emp_cm3_U,
        emp_var_S      = emp_var_S,
        emp_cm3_S      = emp_cm3_S,

        jfr_U              = jfr_U,
        sep_rate_U         = sep_rate_U,
        jfr_S              = jfr_S,
        sep_rate_S         = sep_rate_S,
        ee_rate_S          = ee_rate_S,
        wchg_rate_U        = wchg_rate_U,
        wchg_rate_S        = wchg_rate_S,
        ee_step_S          = ee_step_S,

        mean_wage_U   = mean_wage_U,
        mean_wage_S   = mean_wage_S,
        p10_wage_U    = p10_wage_U,
        p10_wage_S    = p10_wage_S,
        p25_wage_U    = p25_wage_U,
        p25_wage_S    = p25_wage_S,
        p50_wage_U    = p50_wage_U,
        p50_wage_S    = p50_wage_S,
        p75_wage_U    = p75_wage_U,
        p75_wage_S    = p75_wage_S,
        p90_wage_U    = p90_wage_U,
        p90_wage_S    = p90_wage_S,
        wage_premium  = wage_premium,

        theta_U       = theta_U,
        theta_S       = theta_S,

        overlap_UgtS  = overlap_UgtS,
        overlap_SltU  = overlap_SltU,
        ltu_share_S   = ltu_share_S,

        _duration_vals...,
    )
end
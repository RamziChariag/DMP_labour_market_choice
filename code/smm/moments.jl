############################################################
# smm/moments.jl — data targets, weights, calibrated parameters, model moments
#
# Reads the data pipeline's outputs in data/derived/ (moments_{w}.csv,
# sampling_var_{w}.csv, nu_estimation.csv, phi_calibration.csv, windows.json) and
# computes the model counterpart of every targeted moment from a solved equilibrium.
# Targets are read as written; nothing is rescaled here.
#
# Entry points: load_data_moments, load_sampling_variances, load_weight_matrix,
# load_calibrated_params, load_windows, run_weighting, weighting_suffix, model_moments.
############################################################


# ============================================================
# Moment names, in the canonical order of data_and_descriptives/setup.jl
# ============================================================

# The unemployment-duration profile, generated from the solver's thresholds; edit the
# thresholds data-side (setup.jl).
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
    load_data_moments(; window = :base_fc, derived_dir) → NamedTuple

The empirical targets of data window `window` (`DATA_WINDOWS`, paths.jl), read from
`moments_{window}.csv`: one `(value, weight)` pair per moment of `MOMENT_NAMES`, weight 1.
The training_share target (4-year undergraduate enrolment over the CPS population) is
written by the data pipeline and returned as read.
"""
function load_data_moments(; window::Symbol = :base_fc, derived_dir::String)
    moments_file = joinpath(derived_dir, "moments_$(window).csv")
    isfile(moments_file) || error("Moments file not found: $moments_file — run the data pipeline first.")
    raw = _read_moments_csv(moments_file)
    return raw
end


"""
    run_weighting() → Symbol

The objective's weighting, `:diagonal` (default) or `:equal`, from `ROYSEARCH_WEIGHTING`.
Every entry point reads it here, so all of them name and read the same files.
"""
run_weighting() = env_setting(:WEIGHTING, :diagonal)

"""
    weighting_suffix(weighting) → String

Filename suffix of a weighting: `_diagonalW` or `_equalW`.
"""
function weighting_suffix(weighting::Symbol)
    weighting === :diagonal && return "_diagonalW"
    weighting === :equal    && return "_equalW"
    error("WEIGHTING must be :diagonal or :equal; got :$weighting.")
end

"""
    load_sampling_variances(; window, derived_dir) → Dict{Symbol,Float64}

The per-moment sampling variances σ̂²_k of `sampling_var_{window}.csv`, in raw moment
units: the diagonal weights of the objective and, as σ̂_k, the data standard errors.
"""
function load_sampling_variances(; window::Symbol, derived_dir::String)
    sv_name = "sampling_var_$(window).csv"
    sv_file = joinpath(derived_dir, sv_name)
    isfile(sv_file) || error("$sv_name not found in $derived_dir — run the data pipeline first.")
    df_sv = CSV.read(sv_file, DataFrame)
    return Dict(Symbol(r.moment) => Float64(r.sampling_var) for r in eachrow(df_sv))
end

"""
    load_weight_matrix(; window, derived_dir, weighting, skip_moments) → Matrix{Float64} | nothing

The SMM weight matrix over the active moments (`MOMENT_NAMES` minus `skip_moments`):
  :diagonal  W = Diagonal(1/σ̂²_k), in raw moment units like the deviations g_k = m_k − m̂_k;
             an active moment without a positive variance is an error
  :equal     `nothing`; `build_smm_spec` then builds Diagonal(weight_k / m̂_k²)
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

    sv  = load_sampling_variances(; window = window, derived_dir = derived_dir)
    d   = Float64[]
    bad = Symbol[]
    for nm in active_names
        v = get(sv, nm, NaN)
        (isfinite(v) && v > 0.0) ? push!(d, 1.0 / v) : (push!(d, NaN); push!(bad, nm))
    end
    isempty(bad) || error(
        "sampling_var_$(window).csv has non-positive / missing variance for active " *
        "moment(s): " * join(string.(bad), ", ") * " — cannot build the diagonal weight.")
    @printf("  weighting = diagonal → W = Diagonal(1/σ̂²_samp) from sampling_var_%s.csv\n",
            window)
    return Matrix(Diagonal(d))
end


# ============================================================
# Externally calibrated parameters
# ============================================================

"""
    load_calibrated_params(; window = :base_fc, derived_dir) → (r, nu, phi)

The parameters held fixed at external values: r = 0.05/12 (5% annual, monthly); ν, the
demographic turnover rate, from the row of nu_estimation.csv for the window's baseline
period (`PAIR_BASE`, paths.jl: crisis_fc → base_fc, crisis_covid → base_covid); φ, the
training completion rate, from phi_calibration.csv.
"""
function load_calibrated_params(; window::Symbol = :base_fc,
                                  derived_dir::String)
    r_val = 0.05 / 12

    haskey(PAIR_BASE, window) || error("load_calibrated_params: unrecognised window :$window. " *
                                       "Expected one of $(join(DATA_WINDOWS, ", ")).")
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


"""
    load_windows(; derived_dir) → (windows, order)

The data windows of `windows.json` (written by the data pipeline): `windows` maps each name
to `(label, ym_start, ym_end, asec_years)`, `order` lists the names in file order.
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


"""
    _read_moments_csv(filepath) → NamedTuple

Moment name → `(value, weight = 1.0)` for every row of `filepath` whose moment is in
`MOMENT_NAMES`, in row order; other rows are dropped with a one-line summary. The
effective weights come from W.
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
# Model moments
#
# Observed log wage = model log wage + N(0, σ_w²). A market's model distribution is its
# wage points (`wage_points` of compute_equilibrium_objects), one (wage, mass) pair per
# employed cell. The error leaves the mean and third central moment unchanged and adds
# σ_w² to the variance; percentiles and overlaps are read off the convolved CDF
# F(y) = Σ_k q_k Φ((y − log w_k)/σ_w), evaluated on the points themselves, so every
# moment is continuous in wages and masses.
# ============================================================

"""
    _logwage_distribution(pts, σ_w) → NamedTuple | nothing

One market's log-wage distribution from its wage points `pts = (w, m)`: log wages,
normalised masses, mean, variance including σ_w², third central moment and SD; `nothing`
for a market that employs no one.
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

"""
    _conv_cdf_pdf(d, y) → (F, f)
    _conv_cdf(d, y) → F

The σ_w-convolved CDF of distribution `d` at log wage `y`, and its density; with
σ_w ≤ 1e-10 the step CDF of the points (density 0).
"""
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

"""
    _conv_quantile(d, target) → Float64

The `target` quantile of the convolved distribution `d`: Newton from the normal
approximation, inside a bracket maintained by monotonicity of F, with a bisection step
whenever Newton would leave it. Stops once the Newton step is below 1e-13 (tested before
the bracket update, so a converged step is never mistaken for one leaving the bracket).
With σ_w ≤ 1e-10, the discrete quantile of the points.
"""
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

"""
    _logwage_moments(d) → NTuple{8,Float64}

(mean, variance, third central moment, p10, p25, p50, p75, p90) of log wages under `d`; all
0 for a market that employs no one (`d === nothing`).
"""
function _logwage_moments(d)
    d === nothing && return ntuple(_ -> 0.0, 8)
    return (d.mean, d.var, d.cm3, _conv_quantile(d, 0.10), _conv_quantile(d, 0.25),
            _conv_quantile(d, 0.50), _conv_quantile(d, 0.75), _conv_quantile(d, 0.90))
end

"""
    model_moments(obj) → NamedTuple

The targeted moments of a solved equilibrium, `obj` being the NamedTuple returned by
`compute_equilibrium_objects`. Within-market objects of a market that holds no one arrive
at the limit of its composition and are used as they are; a market that employs no one has
wage moments, wage premium and overlaps of 0. Fields absent from older equilibrium objects
(σ_w, `ee_step_S`, `ltu_share_S`, the survivor vectors) read as 0.
"""
function model_moments(obj)
    ur_U           = obj.ur_U
    ur_S           = obj.ur_S
    ur_total       = obj.ur_total
    _model_lf      = obj.agg_uU + obj.agg_eU + obj.agg_mS
    skilled_share  = obj.agg_mS  / max(_model_lf, 1e-14)
    training_share = obj.agg_n   / max(obj.total_pop, 1e-14)

    σ_wU = hasproperty(obj, :σ_wU) ? obj.σ_wU : 0.0
    σ_wS = hasproperty(obj, :σ_wS) ? obj.σ_wS : 0.0

    # Transition and wage-change rates, each as the data stage measures it over the
    # moment's universe (compute_equilibrium_objects, `measured`).
    jfr_U       = obj.measured.jfr_U
    jfr_S       = obj.measured.jfr_S
    sep_rate_U  = obj.measured.sep_rate_U
    sep_rate_S  = obj.measured.sep_rate_S
    ee_rate_S   = obj.measured.ee_rate_S
    wchg_rate_U = obj.measured.wchg_rate_U
    wchg_rate_S = obj.measured.wchg_rate_S

    ee_step_S = hasproperty(obj, :ee_step_S) ? obj.ee_step_S : 0.0

    # Wage moments on log wages with measurement error, as the data side computes them on
    # log(wage_norm).
    dist_U = _logwage_distribution(obj.wage_points.U, σ_wU)
    dist_S = _logwage_distribution(obj.wage_points.S, σ_wS)
    (mean_wage_U, emp_var_U, emp_cm3_U,
     p10_wage_U, p25_wage_U, p50_wage_U, p75_wage_U, p90_wage_U) = _logwage_moments(dist_U)
    (mean_wage_S, emp_var_S, emp_cm3_S,
     p10_wage_S, p25_wage_S, p50_wage_S, p75_wage_S, p90_wage_S) = _logwage_moments(dist_S)

    wage_premium = mean_wage_S - mean_wage_U

    # Cross-market overlaps: tail of one market's convolved distribution at the other's
    # convolved median, P(log w_U > med log w_S) and P(log w_S < med log w_U).
    both = !isnothing(dist_U) && !isnothing(dist_S)
    overlap_UgtS = both ? 1.0 - _conv_cdf(dist_U, p50_wage_S) : 0.0
    overlap_SltU = both ? _conv_cdf(dist_S, p50_wage_U)       : 0.0

    ltu_share_S = hasproperty(obj, :ltu_share_S) ? obj.ltu_share_S : 0.0

    # Duration profile: one survivor vector per market, ordered as DURATION_THRESHOLDS_WK.
    _usurv = fld -> hasproperty(obj, fld) ? collect(getproperty(obj, fld)) :
                    fill(0.0, length(DURATION_THRESHOLDS_WK))
    _usurv_U = _usurv(:usurv_U)
    _usurv_S = _usurv(:usurv_S)
    _duration_vals = NamedTuple{Tuple(DURATION_MOMENT_NAMES)}(
        Tuple(vcat(_usurv_U, _usurv_S)))

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

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
# NSC-based training_share level adjustment (κ_w):
#   The CPS SCHLCOLL universe expanded in Jan 2013 (16–24 → 16–54), so
#   the raw CPS training_share level is not directly comparable across
#   the FC and COVID windows. derived/training_share_scale.csv holds one
#   κ_w per window, κ_w = NSC_IPEDS_enr_w / CPS_enr_w, applied as:
#       data_target                ←  κ_w · CPS_training_share
#       σ̂²_samp[training_share]    ←  κ_w² · σ̂²_samp[training_share]
#   The model-side training_share is age-uncapped (agg_t / total_pop), so
#   κ-scaling the data target makes the two conceptually comparable. If
#   training_share_scale.csv is missing, κ_w defaults to 1.0 with a
#   warning.
#
#   The κ_w application lives in the data pipeline: Stage 7 writes
#   κ_w·training_share into moments_{window}.csv and Stage 10 writes the
#   κ²-scaled training_share sampling variance into sampling_var_{window}.csv.
#   The loaders below read these pre-adjusted values directly.
#   load_training_share_scale reads κ_w for the run-log report in smm_main.jl.
############################################################


# ============================================================
# Moment names (canonical order — 28 moments)
#
# Identical ordering to data_processing/setup.jl. The cross-market
# overlap pair (overlap_UgtS, overlap_SltU) and the skilled long-term-
# unemployment share (ltu_share_S) are appended at the END so the
# moment indices line up with the data-side moment vector.
# ============================================================

const MOMENT_NAMES = [
    :ur_total, :ur_U, :ur_S,
    :skilled_share, :training_share,
    :emp_var_U, :emp_cm3_U, :emp_var_S, :emp_cm3_S,
    :jfr_U, :sep_rate_U, :jfr_S, :sep_rate_S,
    :ee_rate_S,
    :mean_wage_U, :mean_wage_S,
    :p25_wage_U, :p25_wage_S, :p50_wage_U, :p50_wage_S, :p75_wage_U, :p75_wage_S,
    :wage_premium, :theta_U, :theta_S,
    :overlap_UgtS, :overlap_SltU, :ltu_share_S,
    :wchg_rate_U, :wchg_rate_S,
    :ee_step_S,
]

@assert length(MOMENT_NAMES) == 31 "Expected 31 moments, got $(length(MOMENT_NAMES))"


"""
    load_training_share_scale(; window::Symbol, derived_dir::String) → Float64

Return κ_w for the given window, read from
`derived/training_share_scale.csv` produced by the "CPS vs NSC
enrolment" cell in the data pipeline. Returns 1.0 with a warning if
the file is missing or the window has no row.
"""
function load_training_share_scale(; window::Symbol, derived_dir::String) :: Float64
    path = joinpath(derived_dir, "training_share_scale.csv")
    if !isfile(path)
        @warn "training_share_scale.csv not found in $derived_dir — using κ = 1.0 " *
              "(no NSC-based level adjustment applied). Re-run the CPS-vs-NSC " *
              "cell in the data pipeline to enable κ."
        return 1.0
    end
    df = CSV.read(path, DataFrame)
    df.window = Symbol.(df.window)
    rows = filter(:window => ==(window), df)
    if isempty(rows)
        @warn "No row for window=:$window in $path — using κ = 1.0."
        return 1.0
    end
    κ = Float64(rows.kappa_training_share[1])
    if !isfinite(κ) || κ <= 0
        @warn @sprintf("κ for :%s is non-finite or non-positive (%.4e) — using κ = 1.0.",
                       window, κ)
        return 1.0
    end
    return κ
end


"""
    load_data_moments(; window::Symbol = :base_fc, derived_dir::String) → NamedTuple

Return the empirical moment targets used in SMM estimation.  Each
field is a (value, weight) tuple.  Reads from
`moments_{window}.csv` produced by the data pipeline.

The training_share row already reflects the NSC IPEDS-Universe
level: the κ_w adjustment is applied upstream in the data pipeline
(Stage 7), so this returns the moments exactly as written to
`moments_{window}.csv` with no further rescaling.

Moment list
  Labour-market stocks (5)
    ur_total, ur_U, ur_S, skilled_share, training_share

  Wage shape (4)
    emp_var_U, emp_cm3_U, emp_var_S, emp_cm3_S

  Transition rates (5)
    jfr_U, sep_rate_U, jfr_S, sep_rate_S, ee_rate_S

  Wages (9)
    mean_wage_U, mean_wage_S,
    p25_wage_U, p25_wage_S, p50_wage_U, p50_wage_S,
    p75_wage_U, p75_wage_S,
    wage_premium

  Tightness (2)
    theta_U, theta_S

  Cross-market overlap and duration (3)
    overlap_UgtS, overlap_SltU, ltu_share_S
"""
function load_data_moments(; window::Symbol = :base_fc, derived_dir::String)
    moments_file = joinpath(derived_dir, "moments_$(window).csv")
    isfile(moments_file) || error("Moments file not found: $moments_file — run the data pipeline first.")
    raw = _read_moments_csv(moments_file)

    # training_share already carries the NSC κ_w level adjustment, applied
    # upstream in the data pipeline (Stage 7). Return the moments as read.
    return raw
end


"""
    load_weight_matrix(; window, derived_dir, cond_target,
                         skip_moments) → Union{Nothing, Matrix{Float64}}

Build the SMM weighting matrix over the active moments (i.e. those in
`MOMENT_NAMES` that are not in `skip_moments`).

Conditioning behaviour (controlled by `cond_target`):
  cond_target == 0.0  diagonal-σ: W = Diagonal(1 / σ̂²_samp) from the per-moment
                       SAMPLING-variance vector sampling_var_{window}.csv (all
                       moments on one 1/N footing).
  cond_target == 2.0  equal weights — returns `nothing`

The σ̂²_samp vector is in RAW moment units, matching the raw deviation vector
g_k = m_k − m̂_k in compute_loss_matrix (all scaling lives in W).
"""
function load_weight_matrix(;
    window::Symbol = :base_fc,
    derived_dir::String,
    cond_target::Float64 = 0.0,
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

    if cond_target == 2.0
        @printf("  cond_target == 2.0 → equal weights (no W matrix)\n")
        return nothing
    end

    # Diagonal-σ: W = Diagonal(1/σ̂²_samp) over the active moments, from the
    # per-moment sampling-variance vector (one 1/N footing). A missing file
    # means the data pipeline has not been run — error loudly.
    if cond_target == 0.0
        sv_file = joinpath(derived_dir, "sampling_var_$(window).csv")
        isfile(sv_file) || error(
            "sampling_var_$(window).csv not found in $derived_dir — the diagonal-σ " *
            "weight needs the per-moment sampling variances; run the data pipeline first.")
        df_sv = CSV.read(sv_file, DataFrame)
        sv    = Dict(Symbol(r.moment) => Float64(r.sampling_var) for r in eachrow(df_sv))
        d = Float64[]
        bad = Symbol[]
        for nm in active_names
            v = get(sv, nm, NaN)
            (isfinite(v) && v > 0.0) ? push!(d, 1.0 / v) : (push!(d, NaN); push!(bad, nm))
        end
        isempty(bad) || error(
            "sampling_var_$(window).csv has non-positive / missing variance for active " *
            "moment(s): " * join(string.(bad), ", ") * " — cannot build the diagonal-σ weight.")
        @printf("  cond_target == 0.0 → diagonal-σ W = Diagonal(1/σ̂²_samp) from %s\n", sv_file)
        return Matrix(Diagonal(d))
    end

    error("full-Σ⁻¹ and compressed-diagonal weighting were removed in v14.1; use " *
          "cond_target=0.0 (diagonal sampling-variance) or 2.0 (equal weights). " *
          "Got cond_target=$cond_target.")
end


"""
    load_sigma_matrix(; window::Symbol = :base_fc, derived_dir::String) → Vector{Float64}

Read `sigma_{window}.csv` and return the vector of standard errors
(square root of the diagonal). The training_share entry already
carries the κ_w level adjustment (applied in the data pipeline,
Stage 8), so the returned σ refers to the κ-rescaled data target.
"""
function load_sigma_matrix(; window::Symbol = :base_fc, derived_dir::String)
    sigma_file = joinpath(derived_dir, "sigma_$(window).csv")
    isfile(sigma_file) || error("Sigma file not found: $sigma_file — run the data pipeline first.")
    σ_vec = _read_sigma_csv(sigma_file)
    return σ_vec
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
  φ   from phi_calibration.csv training completion rate (NSC/IPEDS,
                                pooled across Fall semesters)

`window` controls which ν is returned. Crisis windows are mapped to
their baseline (crisis_fc → base_fc, crisis_covid → base_covid)
per data_and_moments.pdf §21.
"""
function load_calibrated_params(; window::Symbol = :base_fc,
                                  derived_dir::String)
    r_val = 0.05 / 12

    # Which baseline supplies ν for this window
    nu_pair = if window in (:base_fc, :crisis_fc)
        :base_fc
    elseif window in (:base_covid, :crisis_covid)
        :base_covid
    else
        error("load_calibrated_params: unrecognised window :$window. " *
              "Expected one of :base_fc, :crisis_fc, :base_covid, :crisis_covid.")
    end

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


"""
    _read_sigma_csv(filepath) → Vector{Float64}

Read a covariance-matrix CSV and return the vector of standard
errors (square root of the diagonal).
"""
function _read_sigma_csv(filepath::String)
    df = CSV.read(filepath, DataFrame)
    Sigma = Matrix{Float64}(df)
    return sqrt.(max.(diag(Sigma), 0.0))
end


# ============================================================
# Model-side moments
# ============================================================

"""
    model_moments(obj) → NamedTuple

Extract the targeted moments from a solved model's equilibrium
objects.  `obj` is the NamedTuple returned by
`compute_equilibrium_objects`.

When aggregate employment is negligible for a segment, all
wage-related moments for that segment collapse to 0 — this avoids
spuriously "good" wage percentile values produced by interpolating
a near-zero density and a misleading wage premium computed over a
dummy grid.
"""
# ── Log-wage moments with lognormal measurement error ──────────────────────
# Observed log wage = structural log wage + N(0, σ_w²).  Effects on moments:
#   mean : unchanged          variance : structural + σ_w²
#   third central moment : unchanged (symmetric error)
#   p25 / p50 / p75 : quantiles of the σ_w-convolved log-wage distribution.
# Input (wmid, dens, bw) is the LEVEL-wage density from the equilibrium;
# returns (mean_log, var_log, cm3_log, p25_log, p50_log, p75_log).
function _logwage_moments(wmid::AbstractVector, dens::AbstractVector,
                          bw::Real, σ_w::Real)
    logw = log.(max.(wmid, 1e-14))
    mass = dens .* bw
    M    = sum(mass)
    M < 1e-12 && return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mass = mass ./ M
    μ    = sum(logw .* mass)
    dev  = logw .- μ
    v0   = sum((dev .^ 2) .* mass)
    cm3  = sum((dev .^ 3) .* mass)
    var  = v0 + σ_w^2
    if σ_w > 1e-10
        nd   = Normal(0.0, float(σ_w))
        Fcdf = y -> begin
            s = 0.0
            @inbounds for j in eachindex(logw)
                s += mass[j] * cdf(nd, y - logw[j])
            end
            s
        end
        lo  = minimum(logw) - 8.0 * σ_w
        hi  = maximum(logw) + 8.0 * σ_w
        p25 = _invert_cdf(Fcdf, 0.25, lo, hi)
        p50 = _invert_cdf(Fcdf, 0.50, lo, hi)
        p75 = _invert_cdf(Fcdf, 0.75, lo, hi)
    else
        p25 = _disc_pctile(logw, mass, 0.25)
        p50 = _disc_pctile(logw, mass, 0.50)
        p75 = _disc_pctile(logw, mass, 0.75)
    end
    return (μ, var, cm3, p25, p50, p75)
end

# Bisection inverse of a monotone-increasing CDF F on [lo, hi].
function _invert_cdf(F, target::Real, lo::Real, hi::Real)
    a = float(lo); b = float(hi)
    F(a) >= target && return a
    F(b) <= target && return b
    @inbounds for _ in 1:60
        m = 0.5 * (a + b)
        (F(m) < target) ? (a = m) : (b = m)
    end
    return 0.5 * (a + b)
end

# Discrete quantile of a normalised log-wage mass vector (σ_w → 0 fallback).
function _disc_pctile(logw::AbstractVector, mass::AbstractVector, target::Real)
    cum = 0.0
    @inbounds for j in eachindex(logw)
        cum += mass[j]
        cum >= target && return logw[j]
    end
    return logw[end]
end

# CDF of the σ_w-convolved log-wage distribution as a callable F(y).  Same
# lognormal-measurement-error layer as _logwage_moments: each structural
# log-wage atom logw[j] (mass[j]) is smeared by N(0, σ_w²), so
# F(y) = Σ_j mass[j] · Φ((y − logw[j]) / σ_w).  With σ_w → 0 this is the
# step CDF of the discrete structural mass.  Returns nothing if the segment
# has negligible employment, signalling the caller to skip the overlap.
function _logwage_conv_cdf(wmid::AbstractVector, dens::AbstractVector,
                           bw::Real, σ_w::Real)
    logw = log.(max.(wmid, 1e-14))
    mass = dens .* bw
    M    = sum(mass)
    M < 1e-12 && return nothing
    mass = mass ./ M
    if σ_w > 1e-10
        nd = Normal(0.0, float(σ_w))
        return y -> begin
            s = 0.0
            @inbounds for j in eachindex(logw)
                s += mass[j] * cdf(nd, y - logw[j])
            end
            s
        end
    else
        return y -> begin
            s = 0.0
            @inbounds for j in eachindex(logw)
                logw[j] <= y && (s += mass[j])
            end
            s
        end
    end
end

function model_moments(obj)
    _emp_tol = 1e-12
    _has_eU  = obj.agg_eU > _emp_tol
    _has_eS  = obj.agg_eS > _emp_tol

    # Labour-market stocks.  When the skilled segment is empty,
    # report ur_S = 1.0 so that the SMM does not interpret a
    # nonexistent skilled labour force as having low unemployment.
    ur_U           = obj.ur_U
    ur_S           = obj.agg_mS > _emp_tol ? obj.ur_S : 1.0
    ur_total       = obj.ur_total
    _model_lf      = obj.agg_uU + obj.agg_eU + obj.agg_mS
    skilled_share  = obj.agg_mS  / max(_model_lf, 1e-14)
    training_share = obj.agg_t   / max(obj.total_pop, 1e-14)

    # Log-wage measurement-error SDs, carried from the equilibrium params.
    # Wage moments (mean / variance / third moment / percentiles) are computed
    # in LOG wages below, with lognormal measurement error of SD σ_w.
    σ_wU = hasproperty(obj, :σ_wU) ? obj.σ_wU : 0.0
    σ_wS = hasproperty(obj, :σ_wS) ? obj.σ_wS : 0.0

    # Transition rates
    jfr_U      = _has_eU ? obj.f_U : 0.0
    jfr_S      = _has_eS ? obj.f_S : 0.0
    sep_rate_U = obj.sep_rate_U
    sep_rate_S = obj.sep_rate_S
    ee_rate_S  = obj.ee_rate_S

    # Within-job wage-change hazards λ_j·(1 − G_j(p*)) (SIPP moments). Read
    # from the equilibrium; 0.0 when the field is absent.
    wchg_rate_U = hasproperty(obj, :wchg_rate_U) ? obj.wchg_rate_U : 0.0
    wchg_rate_S = hasproperty(obj, :wchg_rate_S) ? obj.wchg_rate_S : 0.0

    # Skilled EE-move mean wage step (SIPP moment). Read from the equilibrium;
    # 0.0 when the field is absent.
    ee_step_S = hasproperty(obj, :ee_step_S) ? obj.ee_step_S : 0.0

    # Wages — all moments on LOG wages, with lognormal measurement error σ_w.
    # "mean_wage_*", "emp_var_*", "emp_cm3_*", "p25/p50_wage_*" now hold the
    # corresponding LOG-wage objects (mean/var/3rd-moment/quantiles of log w);
    # the DATA side must compute these on log(wage_norm) to match.
    wmid   = obj.wmid
    dens_U = obj.dens_U
    dens_S = obj.dens_S
    bw     = length(wmid) >= 2 ? wmid[2] - wmid[1] : 1.0

    if _has_eU
        (mean_log_wage_U, emp_var_U, emp_cm3_U, p25_wage_U, p50_wage_U, p75_wage_U) =
            _logwage_moments(wmid, dens_U, bw, σ_wU)
        mean_wage_U = mean_log_wage_U
    else
        mean_wage_U = 0.0; mean_log_wage_U = 0.0
        emp_var_U = 0.0; emp_cm3_U = 0.0; p25_wage_U = 0.0; p50_wage_U = 0.0; p75_wage_U = 0.0
    end

    if _has_eS
        (mean_log_wage_S, emp_var_S, emp_cm3_S, p25_wage_S, p50_wage_S, p75_wage_S) =
            _logwage_moments(wmid, dens_S, bw, σ_wS)
        mean_wage_S = mean_log_wage_S
    else
        mean_wage_S = 0.0; mean_log_wage_S = 0.0
        emp_var_S = 0.0; emp_cm3_S = 0.0; p25_wage_S = 0.0; p50_wage_S = 0.0; p75_wage_S = 0.0
    end

    wage_premium = (_has_eU && _has_eS) ?
                   (mean_log_wage_S - mean_log_wage_U) : 0.0

    # Cross-market wage overlap (one moment per bargaining weight).  Each is
    # a tail probability of one segment's σ_w-convolved log-wage distribution
    # evaluated at the OTHER segment's convolved median:
    #   overlap_UgtS = P(log w_U > med log w_S) = 1 − F_U^conv(p50_wage_S)
    #   overlap_SltU = P(log w_S < med log w_U) =     F_S^conv(p50_wage_U)
    # Both require non-empty employment on both sides; otherwise the overlap
    # is undefined and reported as 0 (matched by the skip on an empty segment).
    if _has_eU && _has_eS
        F_U_conv = _logwage_conv_cdf(wmid, dens_U, bw, σ_wU)
        F_S_conv = _logwage_conv_cdf(wmid, dens_S, bw, σ_wS)
        overlap_UgtS = isnothing(F_U_conv) ? 0.0 : 1.0 - F_U_conv(p50_wage_S)
        overlap_SltU = isnothing(F_S_conv) ? 0.0 : F_S_conv(p50_wage_U)
    else
        overlap_UgtS = 0.0
        overlap_SltU = 0.0
    end

    # Skilled long-term-unemployment share: survival of a skilled spell past
    # a* ≈ 6.23 months under the type-mixture of exponential exit hazards.
    # Computed in compute_equilibrium_objects (it needs uS, Γ(p*_S), κ_S, wx);
    # read it through here. Absent or empty skilled segment ⇒ 0.
    ltu_share_S = (hasproperty(obj, :ltu_share_S) && _has_eS) ? obj.ltu_share_S : 0.0

    # Tightness
    _THETA_CAP = 1e14
    theta_U = _has_eU ? obj.thetaU : _THETA_CAP
    theta_S = _has_eS ? obj.thetaS : _THETA_CAP

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
        p25_wage_U    = p25_wage_U,
        p25_wage_S    = p25_wage_S,
        p50_wage_U    = p50_wage_U,
        p50_wage_S    = p50_wage_S,
        p75_wage_U    = p75_wage_U,
        p75_wage_S    = p75_wage_S,
        wage_premium  = wage_premium,

        theta_U       = theta_U,
        theta_S       = theta_S,

        overlap_UgtS  = overlap_UgtS,
        overlap_SltU  = overlap_SltU,
        ltu_share_S   = ltu_share_S,
    )
end
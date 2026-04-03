############################################################
# moments.jl — Empirical moment targets
#
# Supports loading moments from CSV files produced by the data
# pipeline, with fallback to placeholder values.
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


# ============================================================
# Moment names (in order)
# ============================================================

const MOMENT_NAMES = [
    :ur_total, :ur_U, :ur_S,
    :exp_ur_total, :exp_ur_U, :exp_ur_S,
    :skilled_share, :training_share,
    :emp_var_U, :emp_cm3_U, :emp_var_S, :emp_cm3_S,
    :jfr_U, :sep_rate_U, :jfr_S, :sep_rate_S, :ee_rate_S,
    :mean_wage_U, :mean_wage_S,
    :p25_wage_U, :p25_wage_S, :p50_wage_U, :p50_wage_S,
    :wage_premium, :theta_U, :theta_S,
]


"""
    load_data_moments(; window::Symbol = :base_fc, derived_dir::String) → NamedTuple

Returns the empirical moment targets used in SMM estimation.
Each field is a (value, weight) tuple.

Reads from CSV files produced by the data pipeline:
  - moments_{window}.csv: moment values and standard errors

# Arguments
- `window::Symbol`: window identifier (default `:base_fc`)
- `derived_dir::String`: directory containing derived CSV files

# Moment list (23 moments, matches data_pipeline_v6)
───────────────────────────────────────────────────────────
  Labour market stocks
    ur_U              unskilled unemployment rate
    ur_S              skilled unemployment rate
    ur_total          aggregate (population-weighted) unemployment rate
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
    ee_rate_S         skilled employment-to-employment transition rate

  Wages
    mean_wage_U       mean wage, unskilled employed
    mean_wage_S       mean wage, skilled employed
    p25_wage_U        25th percentile wage, unskilled
    p25_wage_S        25th percentile wage, skilled
    p50_wage_U        median wage, unskilled
    p50_wage_S        median wage, skilled
    wage_premium      E[log w_S] − E[log w_U]  (log skill premium)

  Tightness (if vacancy data available)
    theta_U           unskilled vacancy-unemployment ratio
    theta_S           skilled vacancy-unemployment ratio
───────────────────────────────────────────────────────────
"""
function load_data_moments(; window::Symbol = :base_fc, derived_dir::String)

    moments_file = joinpath(derived_dir, "moments_$(window).csv")
    isfile(moments_file) || error("Moments file not found: $moments_file — run the data pipeline first.")

    return _read_moments_csv(moments_file)
end


"""
    load_weight_matrix(; window, derived_dir, cond_target,
                         skip_moments) → Union{Nothing, Matrix{Float64}}

Load the optimal weight matrix from influence functions.

Returns a matrix already subsetted to the active moments (i.e. all
moments NOT in `skip_moments`), or `nothing` for the equal-weight case.
The returned matrix is always K_active × K_active where
K_active = K − |skip_moments|, so it is dimensionally consistent with
the deviation vector built by `compute_loss_matrix`.

Conditioning behaviour (controlled by `cond_target`):
  - cond_target == 0.0:  Pure diagonal matrix from sigma_{window}.csv.
    Weights are 1/σ² for each moment.
  - cond_target == 1.0:  Compressed diagonal — loads diagonal weights
    (1/σ²) from sigma_{window}.csv, then applies log(1 + d) element-wise
    to compress the dynamic range while preserving ranking.
  - cond_target == 2.0:  Equal weights — returns `nothing`.
    The loss function will use compute_loss with unit diagonal weights,
    i.e. all moments weighted equally (no W matrix).
  - cond_target > 2.0:  Full optimal W = Σ⁻¹ computed fresh from
    sigma_{window}.csv.  The pipeline generates this file already
    subsetted to the active moments, so it is used as-is — no size
    checks, no index subsetting, no dependency on any saved W file.
    If κ(W) > cond_target, shrink off-diagonal elements toward zero via
        W_shrunk = (1 − α) W  +  α diag(W)
    until κ(W_shrunk) ≤ cond_target.  The shrinkage factor α is found
    by bisection.  Shrinkage is applied AFTER subsetting, so the
    condition number is evaluated on the active submatrix only.

- `skip_moments`: vector of moment Symbols to exclude.  Must match
  what is passed to `build_smm_spec`.  Unknown names produce a warning.
"""
function load_weight_matrix(;
    window::Symbol = :base_fc,
    derived_dir::String,
    cond_target::Float64 = 1e8,
    skip_moments::Vector{Symbol} = Symbol[],
)

    K = length(MOMENT_NAMES)

    # Compute active indices (into MOMENT_NAMES) once — used by all branches.
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

    # ── cond_target == 2.0  →  equal weights (identity / nothing) ─────
    if cond_target == 2.0
        @printf("  cond_target == 2.0 → equal weights (no W matrix)\n")
        return nothing   # compute_loss uses unit weights; no subsetting needed
    end

    # ── cond_target == 0.0  →  pure diagonal from active submatrix of Σ ──
    if cond_target == 0.0
        sigma_file = joinpath(derived_dir, "sigma_$(window).csv")
        isfile(sigma_file) || error(
            "sigma_$(window).csv not found in $derived_dir — run the data pipeline first.")
        df_sig   = CSV.read(sigma_file, DataFrame)
        csv_cols = Symbol.(names(df_sig))
        active_col_idx = [findfirst(==(nm), csv_cols) for nm in active_names]
        d = diag(Matrix{Float64}(df_sig)[active_col_idx, active_col_idx])
        W = Diagonal(1.0 ./ max.(d, 1e-14))
        @printf("  cond_target == 0.0 → diagonal W from active submatrix of %s\n", sigma_file)
        return Matrix(W)
    end

    # ── cond_target == 1.0  →  compressed diagonal from active submatrix of Σ
    if cond_target == 1.0
        sigma_file = joinpath(derived_dir, "sigma_$(window).csv")
        isfile(sigma_file) || error(
            "sigma_$(window).csv not found in $derived_dir — run the data pipeline first.")
        df_sig   = CSV.read(sigma_file, DataFrame)
        csv_cols = Symbol.(names(df_sig))
        active_col_idx = [findfirst(==(nm), csv_cols) for nm in active_names]
        d            = diag(Matrix{Float64}(df_sig)[active_col_idx, active_col_idx])
        d_inv        = 1.0 ./ max.(d, 1e-14)
        d_compressed = log.(1.0 .+ d_inv)
        W = Diagonal(d_compressed)
        @printf("  cond_target == 1.0 → compressed diagonal W from active submatrix of %s\n", sigma_file)
        @printf("    raw 1/σ² range:         [%.4e, %.4e]\n", minimum(d_inv), maximum(d_inv))
        @printf("    compressed diag range:   [%.4e, %.4e]\n", minimum(d_compressed), maximum(d_compressed))
        return Matrix(W)
    end

    # ── Full optimal W = inv(Σ_active) ────────────────────────────────────────
    # Read the full K×K sigma CSV, subset to active moments by column name, then
    # invert. Shrinkage is applied to Σ BEFORE inverting: shrinking W = inv(Σ)
    # toward diag(inv(Σ)) is wrong because diag(inv(Σ)) can be negative when Σ
    # is ill-conditioned, producing a negative-definite W. Shrinking Σ first
    # guarantees Σ_shrunk is PD, so W = inv(Σ_shrunk) has positive diagonal.
    sigma_file = joinpath(derived_dir, "sigma_$(window).csv")
    isfile(sigma_file) || error(
        "sigma_$(window).csv not found in $derived_dir — run the data pipeline first.")

    df_sig   = CSV.read(sigma_file, DataFrame)
    csv_cols = Symbol.(names(df_sig))

    col_pos = Dict(nm => findfirst(==(nm), csv_cols) for nm in csv_cols)
    missing_cols = [nm for nm in active_names if !haskey(col_pos, nm)]
    isempty(missing_cols) || error(
        "sigma_$(window).csv is missing columns for active moments: " *
        join(string.(missing_cols), ", ") *
        " — re-run the data pipeline.")
    active_col_idx = [col_pos[nm] for nm in active_names]

    Σ_full = Matrix{Float64}(df_sig)
    Σ      = Σ_full[active_col_idx, active_col_idx]   # K_active × K_active
    Σsym   = Symmetric(Σ)
    cond_Σ = cond(Σsym)

    Σ_to_invert = if cond_Σ <= cond_target
        @printf("  Full W = Σ⁻¹ from active submatrix of %s  (cond(Σ) = %.2e)\n", sigma_file, cond_Σ)
        Σsym
    else
        Σdiag  = Symmetric(Matrix(Diagonal(diag(Σ))))
        κ_diag = cond(Σdiag)

        if !isfinite(κ_diag) || κ_diag >= cond_target
            @warn @sprintf("Even pure diagonal Σ is ill-conditioned (κ = %.2e) — using diagonal W = inv(diag(Σ)).", κ_diag)
            return Matrix(Diagonal(1.0 ./ max.(diag(Σ), 1e-14)))
        end

        # Bisect on ρ: Σ(ρ) = (1−ρ)·diag(Σ) + ρ·Σ.
        # ρ = 0 → pure diagonal; ρ = 1 → original Σ.
        # Find largest ρ such that κ(Σ(ρ)) ≤ cond_target.
        ρ_lo, ρ_hi, best_ρ = 0.0, 1.0, 0.0
        Σ_mid = similar(Matrix(Σsym))
        for _ in 1:60
            ρ_mid  = (ρ_lo + ρ_hi) / 2
            Σ_mid .= (1.0 - ρ_mid) .* Σdiag .+ ρ_mid .* Σsym
            κ_mid  = cond(Symmetric(Σ_mid))
            if isfinite(κ_mid) && κ_mid <= cond_target
                best_ρ = ρ_mid; ρ_lo = ρ_mid
            else
                ρ_hi = ρ_mid
            end
            (ρ_hi - ρ_lo) < 1e-6 && break
        end

        Σ_shrunk = Symmetric((1.0 - best_ρ) .* Σdiag .+ best_ρ .* Σsym)
        κ_shrunk = cond(Σ_shrunk)
        @warn @sprintf("Σ ill-conditioned (κ = %.2e > %.2e) — shrinking off-diagonals (ρ = %.6f → κ(Σ) = %.2e).",
                       cond_Σ, cond_target, best_ρ, κ_shrunk)
        Σ_shrunk
    end

    return Matrix(inv(Σ_to_invert))
end


"""
    _shrink_to_target(W, cond_target; tol, maxiter) → (W_shrunk, α)

Shrink the off-diagonal elements of symmetric matrix W so that
    κ(W_shrunk) ≤ cond_target
using the convex combination
    W(α) = (1 − α) W  +  α diag(W),       α ∈ [0, 1]

At α = 0 we have the original W; at α = 1, a pure diagonal.
The condition number is monotonically non-increasing in α, so
bisection is guaranteed to converge.

Returns the shrunk matrix and the shrinkage factor α used.
"""
function _shrink_to_target(
    W::Matrix{Float64},
    cond_target::Float64;
    tol::Float64  = 1e-3,
    maxiter::Int  = 100,
)
    D = Diagonal(diag(W))

    # Sanity: if pure diagonal already exceeds target, just return it
    if cond(Matrix(D)) > cond_target
        return Matrix(D), 1.0
    end

    lo, hi = 0.0, 1.0
    alpha  = 0.5
    W_shrunk = similar(W)

    for _ in 1:maxiter
        alpha = (lo + hi) / 2.0
        W_shrunk .= (1.0 - alpha) .* W .+ alpha .* D
        κ = cond(W_shrunk)
        if κ <= cond_target
            hi = alpha          # can shrink less
        else
            lo = alpha          # need to shrink more
        end
        (hi - lo) < tol && break
    end

    # Use the conservative (higher-α) endpoint to guarantee κ ≤ target
    alpha = hi
    W_shrunk .= (1.0 - alpha) .* W .+ alpha .* D
    return W_shrunk, alpha
end


"""
    load_sigma_matrix(; window::Symbol = :base_fc, derived_dir::String) → Vector{Float64}

Load the standard error vector from sigma_{window}.csv.

Returns a K-element vector of standard errors (σ), where K = number of moments.
"""
function load_sigma_matrix(; window::Symbol = :base_fc, derived_dir::String)

    sigma_file = joinpath(derived_dir, "sigma_$(window).csv")
    isfile(sigma_file) || error("Sigma file not found: $sigma_file — run the data pipeline first.")

    return _read_sigma_csv(sigma_file)
end


# ============================================================
# Load externally calibrated parameters from data pipeline
# ============================================================

"""
    load_calibrated_params(; derived_dir::String) -> NamedTuple{(:r, :nu, :phi)}

Load the three externally calibrated parameters:
  - r   = 0.05/12 ≈ 0.00417  (monthly discount rate; 5% annual / 12 months)
  - nu  from nu_estimate.csv  (demographic turnover from CPS matched panels)
  - phi from phi_calibration.csv  (training completion rate from NSC data)

These three are FIXED across all 4 estimation windows (base_fc, crisis_fc,
base_covid, crisis_covid).  Only r is truly external; nu and phi are
computed from data but held constant because they represent structural
parameters that do not change with the business cycle.

Requires the data pipeline to have been run first.  Errors if CSV files
are missing — there are no hardcoded fallbacks for data-derived values.
"""
function load_calibrated_params(; derived_dir::String)
    # r: externally set at 5% annual.  The data is monthly, so the
    # per-period discount rate is 0.05/12 ≈ 0.00417.
    r_val = 0.05 / 12

    # ── nu from CPS matched-panel exit hazard ─────────────────────────
    nu_file = joinpath(derived_dir, "nu_estimate.csv")
    isfile(nu_file) || error("nu_estimate.csv not found in $derived_dir — run the data pipeline first.")
    df_nu = CSV.read(nu_file, DataFrame)
    nu_val = Float64(df_nu.nu[1])
    @printf("  Loaded nu = %.5f from %s\n", nu_val, nu_file)

    # ── phi from NSC completion-rate calibration ───────────────────────
    phi_file = joinpath(derived_dir, "phi_calibration.csv")
    isfile(phi_file) || error("phi_calibration.csv not found in $derived_dir — run the data pipeline first.")
    df_phi = CSV.read(phi_file, DataFrame)
    phi_val = Float64(df_phi.phi[1])
    @printf("  Loaded phi = %.5f from %s\n", phi_val, phi_file)

    return (r = r_val, nu = nu_val, phi = phi_val)
end


# ============================================================
# Internal helpers for reading CSV files
# ============================================================

"""
    _read_moments_csv(filepath) → NamedTuple

Read moments_{window}.csv and return a NamedTuple mapping moment name → (value, weight).

Expected format (CSV with header):
  moment, value
  ur_U, 0.080
  ur_S, 0.025
  ...

Weight is set to 1.0 for all moments; actual weights come from the W matrix.
"""
function _read_moments_csv(filepath::String)
    df = CSV.read(filepath, DataFrame)

    # Build NamedTuple with (value, weight) pairs
    # Weight = 1.0 here; actual weights come from the W matrix
    pairs = Pair{Symbol, NamedTuple{(:value, :weight), Tuple{Float64, Float64}}}[]

    for row in eachrow(df)
        name = Symbol(row.moment)
        val = Float64(row.value)
        push!(pairs, name => (value = val, weight = 1.0))
    end

    return NamedTuple(pairs)
end


"""
    _read_sigma_csv(filepath) → Vector{Float64}

Read sigma_{window}.csv and return the standard error vector.

Expected format: CSV with K×K covariance matrix (moment names as column headers).
Returns the square root of the diagonal elements (standard errors).
Used by the cond_target == 0.0 and 1.0 branches; the full K×K matrix is
read directly via CSV.read in the cond_target > 2.0 branch.
"""
function _read_sigma_csv(filepath::String)
    df = CSV.read(filepath, DataFrame)
    Sigma = Matrix{Float64}(df)
    # Return standard errors (sqrt of diagonal)
    return sqrt.(max.(diag(Sigma), 0.0))
end


"""
    model_moments(obj) -> NamedTuple

Extract the 22 targeted moments from a solved model's equilibrium objects.
`obj` is the NamedTuple returned by `compute_equilibrium_objects`.

Prerequisites -- the following fields must be present in obj:
    f_S         :: Float64             # kappa_S = theta_S * q_S(theta_S)
    sep_rate_U  :: Float64             # employment-weighted unskilled separation rate
    sep_rate_S  :: Float64             # employment-weighted skilled separation rate
    ee_rate_S   :: Float64             # skilled EE transition rate
"""
function model_moments(obj)

    # ── Corner-solution detection ────────────────────────────────────────
    #   When aggregate employment is negligible for a segment, all
    #   wage-related moments for that segment are set to 0.  This avoids
    #   two pathologies:
    #     (a) _percentile returning the dummy-grid endpoint (~1.0) when
    #         the density is all-zeros, giving misleadingly "good" wage
    #         percentile values that confuse the SMM optimizer;
    #     (b) mean_log_wage computed over a dummy grid producing an
    #         arbitrary wage premium.
    #   The _emp_tol threshold matches the guard in equilibrium.jl.
    _emp_tol = 1e-12
    _has_eU  = obj.agg_eU > _emp_tol
    _has_eS  = obj.agg_eS > _emp_tol

    # ── Labour market stocks ──────────────────────────────────────────────
    #   ur_U is well-defined when unskilled mass > 0 (which it always is).
    #   ur_S: when the skilled segment is empty (agg_mS ≈ 0), report
    #   ur_S = 1.0 ("if anyone were there, they'd all be unemployed").
    #   Reporting 0 would mislead the SMM into thinking skilled
    #   unemployment is impressively low.
    ur_U           = obj.ur_U
    ur_S           = obj.agg_mS > _emp_tol ? obj.ur_S : 1.0
    ur_total      = obj.ur_total
    exp_ur_total   = exp(ur_total)
    exp_ur_U       = exp(ur_U)
    exp_ur_S       = exp(ur_S)
    skilled_share  = obj.agg_mS  / max(obj.total_pop, 1e-14)
    training_share = obj.agg_t   / max(obj.total_pop, 1e-14)

    # emp_var / emp_cm3: variance and third central moment of the employed
    # wage distribution, computed from the density grids.
    # When employment is zero the density is all-zeros so these are 0.
    wmid_tmp   = obj.wmid
    dens_U_tmp = obj.dens_U
    dens_S_tmp = obj.dens_S
    bw_tmp     = length(wmid_tmp) >= 2 ? wmid_tmp[2] - wmid_tmp[1] : 1.0

    _mean_U_tmp = sum(wmid_tmp .* dens_U_tmp) * bw_tmp
    _mean_S_tmp = sum(wmid_tmp .* dens_S_tmp) * bw_tmp

    emp_var_U  = sum((wmid_tmp .- _mean_U_tmp).^2 .* dens_U_tmp) * bw_tmp
    emp_cm3_U  = sum((wmid_tmp .- _mean_U_tmp).^3 .* dens_U_tmp) * bw_tmp
    emp_var_S  = sum((wmid_tmp .- _mean_S_tmp).^2 .* dens_S_tmp) * bw_tmp
    emp_cm3_S  = sum((wmid_tmp .- _mean_S_tmp).^3 .* dens_S_tmp) * bw_tmp

    # ── Transition rates ──────────────────────────────────────────────────
    jfr_U = _has_eU ? obj.f_U : 0.0
    jfr_S = _has_eS ? obj.f_S : 0.0
    sep_rate_U    = obj.sep_rate_U
    sep_rate_S    = obj.sep_rate_S
    ee_rate_S     = obj.ee_rate_S

    # ── Wages ─────────────────────────────────────────────────────────────
    #   When a segment has no employment, all its wage moments are 0.
    #   This includes mean, percentiles, and the log wage used in the
    #   premium.  The key fix: _percentile must NOT fall through to
    #   wmid[end] when the density integrates to < 1.
    wmid   = obj.wmid
    dens_U = obj.dens_U
    dens_S = obj.dens_S
    bw     = length(wmid) >= 2 ? wmid[2] - wmid[1] : 1.0

    # Percentile helper — returns 0 when the density has no mass
    # (instead of the old fallback to wmid[end] which was ~1.0).
    function _percentile(wmid, dens, bw, target)
        cum = 0.0
        for j in eachindex(wmid)
            mass = dens[j] * bw
            if cum + mass >= target
                frac = mass > 1e-14 ? (target - cum) / mass : 0.5
                return wmid[j] - bw/2 + frac * bw   # interpolate within bin j
            end
            cum += mass
        end
        # Cumulative density never reached the target → density has
        # insufficient mass.  Return 0 (not wmid[end]) so that the
        # SMM objective sees a clean corner value.
        return 0.0
    end

    if _has_eU
        mean_wage_U    = sum(wmid .* dens_U) * bw
        p25_wage_U     = _percentile(wmid, dens_U, bw, 0.25)
        p50_wage_U     = _percentile(wmid, dens_U, bw, 0.50)
        mean_log_wage_U = sum(log.(max.(wmid, 1e-14)) .* dens_U) * bw
    else
        mean_wage_U     = 0.0
        p25_wage_U      = 0.0
        p50_wage_U      = 0.0
        mean_log_wage_U = 0.0
    end

    if _has_eS
        mean_wage_S    = sum(wmid .* dens_S) * bw
        p25_wage_S     = _percentile(wmid, dens_S, bw, 0.25)
        p50_wage_S     = _percentile(wmid, dens_S, bw, 0.50)
        mean_log_wage_S = sum(log.(max.(wmid, 1e-14)) .* dens_S) * bw
    else
        mean_wage_S     = 0.0
        p25_wage_S      = 0.0
        p50_wage_S      = 0.0
        mean_log_wage_S = 0.0
    end

    # Wage premium: E[log w_S] - E[log w_U]
    #   Only meaningful when both segments have employment.
    #   Otherwise 0 — which is far from the positive data target,
    #   giving the SMM a strong signal to move away from this corner.
    wage_premium = (_has_eU && _has_eS) ?
                   (mean_log_wage_S - mean_log_wage_U) : 0.0

    # ── Tightness ─────────────────────────────────────────────────────────
    _THETA_CAP = 100.0
    theta_U = _has_eU ? obj.thetaU : _THETA_CAP
    theta_S = _has_eS ? obj.thetaS : _THETA_CAP

    return (
        ur_total      = ur_total,
        ur_U          = ur_U,
        ur_S          = ur_S,
        exp_ur_total  = exp_ur_total,
        exp_ur_U      = exp_ur_U,
        exp_ur_S      = exp_ur_S,
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
        ee_rate_S     = ee_rate_S,

        mean_wage_U   = mean_wage_U,
        mean_wage_S   = mean_wage_S,
        p25_wage_U    = p25_wage_U,
        p25_wage_S    = p25_wage_S,
        p50_wage_U    = p50_wage_U,
        p50_wage_S    = p50_wage_S,
        wage_premium  = wage_premium,

        theta_U       = theta_U,
        theta_S       = theta_S,
    )
end
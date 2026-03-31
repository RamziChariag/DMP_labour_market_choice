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
    :ur_U, :ur_S, :skilled_share, :training_share,
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

# Moment list (22 moments, matches data_pipeline_v6)
───────────────────────────────────────────────────────────
  Labour market stocks
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
    load_weight_matrix(; window, derived_dir, cond_target) → Union{Nothing, Matrix{Float64}}

Load the optimal weight matrix from influence functions.

Reads `W_{window}.csv` (K × K matrix) from derived_dir.
Returns a K × K symmetric positive definite matrix, or `nothing`
for the equal-weight (identity) case.

Conditioning behaviour (controlled by `cond_target`):
  - cond_target == 0.0:  Pure diagonal matrix from sigma_{window}.csv.
    Weights are 1/σ² for each moment.
  - cond_target == 1.0:  Compressed diagonal — loads diagonal weights
    (1/σ²) from sigma_{window}.csv, then applies log(1 + d) element-wise
    to compress the dynamic range while preserving ranking.
  - cond_target == 2.0:  Equal weights — returns `nothing`.
    The loss function will use compute_loss with unit diagonal weights,
    i.e. all moments weighted equally (no W matrix).
  - cond_target > 2.0:  Full optimal W matrix.  If the loaded W has
    κ(W) > cond_target, shrink off-diagonal elements toward zero via
        W_shrunk = (1 − α) W  +  α diag(W)
    until κ(W_shrunk) ≤ cond_target.  The shrinkage factor α is found
    by bisection.
"""
function load_weight_matrix(;
    window::Symbol = :base_fc,
    derived_dir::String,
    cond_target::Float64 = 1e8,
)

    K = length(MOMENT_NAMES)

    # ── cond_target == 2.0  →  equal weights (identity / nothing) ─────
    if cond_target == 2.0
        @printf("  cond_target == 2.0 → equal weights (no W matrix)\n")
        return nothing
    end

    # ── cond_target == 0.0  →  pure diagonal from Σ ───────────────────
    if cond_target == 0.0
        sigma_file = joinpath(derived_dir, "sigma_$(window).csv")
        isfile(sigma_file) || error(
            "sigma_$(window).csv not found in $derived_dir — run the data pipeline first.")
        sigma_vec = _read_sigma_csv(sigma_file)
        W = Diagonal(1.0 ./ (sigma_vec .^ 2))
        @printf("  cond_target == 0.0 → using pure diagonal W from %s\n", sigma_file)
        return Matrix(W)
    end

    # ── cond_target == 1.0  →  compressed diagonal from Σ ─────────────
    if cond_target == 1.0
        sigma_file = joinpath(derived_dir, "sigma_$(window).csv")
        isfile(sigma_file) || error(
            "sigma_$(window).csv not found in $derived_dir — run the data pipeline first.")
        sigma_vec = _read_sigma_csv(sigma_file)
        d = 1.0 ./ (sigma_vec .^ 2)
        d_compressed = log.(1.0 .+ d)
        W = Diagonal(d_compressed)
        @printf("  cond_target == 1.0 → using compressed diagonal W from %s\n", sigma_file)
        @printf("    raw diag range:        [%.4e, %.4e]\n", minimum(d), maximum(d))
        @printf("    compressed diag range:  [%.4e, %.4e]\n", minimum(d_compressed), maximum(d_compressed))
        return Matrix(W)
    end

    # ── cond_target > 2.0  →  load full optimal W from CSV ────────────
    W_file = joinpath(derived_dir, "W_$(window).csv")
    if isfile(W_file)
        W_loaded = _read_weight_matrix_csv(W_file)
        cond_W = cond(W_loaded)
        if cond_W <= cond_target
            @printf("  Loaded W matrix from %s  (cond = %.2e)\n", W_file, cond_W)
            return W_loaded
        else
            # ── Shrink off-diagonals to reach target κ ─────────────────
            W_shrunk, alpha = _shrink_to_target(W_loaded, cond_target)
            cond_new = cond(W_shrunk)
            msg = @sprintf("W matrix ill-conditioned (κ = %.2e > %.2e). Shrinking off-diagonal elements (α = %.6f → κ = %.2e).",
                           cond_W, cond_target, alpha, cond_new)
            @warn msg
            return W_shrunk
        end
    end

    # ── No W file found — diagonal fallback from sigma_{window}.csv ───
    sigma_file = joinpath(derived_dir, "sigma_$(window).csv")
    isfile(sigma_file) || error(
        "Neither W_$(window).csv nor sigma_$(window).csv found in $derived_dir — run the data pipeline first.")

    sigma_vec = _read_sigma_csv(sigma_file)
    W = Diagonal(1.0 ./ (sigma_vec .^ 2))
    @printf("  W_%s.csv not found — using diagonal W from %s\n",
            window, sigma_file)
    return Matrix(W)
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
    _read_weight_matrix_csv(filepath) → Matrix{Float64}

Read W_{window}.csv and return the K × K weight matrix.

Expected format: CSV with K columns (moment names as headers) and K rows, symmetric positive definite.
"""
function _read_weight_matrix_csv(filepath::String)
    df = CSV.read(filepath, DataFrame)
    return Matrix{Float64}(df)
end


"""
    _read_sigma_csv(filepath) → Vector{Float64}

Read sigma_{window}.csv and return the standard error vector.

Expected format: CSV with 22×22 matrix (moment names as column headers).
Returns the square root of the diagonal elements (standard errors).
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

    # ── Labour market stocks ──────────────────────────────────────────────
    ur_U           = obj.ur_U
    ur_S           = obj.ur_S
    skilled_share  = obj.agg_mS  / max(obj.total_pop, 1e-14)
    training_share = obj.agg_t   / max(obj.total_pop, 1e-14)

    # emp_var / emp_cm3: variance and third central moment of the employed
    # wage distribution, computed from the density grids.
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
    jfr_U         = obj.f_U
    jfr_S         = obj.f_S
    sep_rate_U    = obj.sep_rate_U
    sep_rate_S    = obj.sep_rate_S
    ee_rate_S     = obj.ee_rate_S

    # ── Wages ─────────────────────────────────────────────────────────────
    wmid   = obj.wmid
    dens_U = obj.dens_U
    dens_S = obj.dens_S
    bw     = wmid[2] - wmid[1]

    mean_wage_U = sum(wmid .* dens_U) * bw
    mean_wage_S = sum(wmid .* dens_S) * bw

    # 25th percentile: first bin where cumulative density crosses 0.25
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
        return wmid[end]
    end

    p25_wage_U = _percentile(wmid, dens_U, bw, 0.25)
    p25_wage_S = _percentile(wmid, dens_S, bw, 0.25)

    # Median: first bin where cumulative density crosses 0.5
    p50_wage_U = _percentile(wmid, dens_U, bw, 0.50)
    p50_wage_S = _percentile(wmid, dens_S, bw, 0.50)

    # Wage premium: E[log w_S] - E[log w_U]
    mean_log_wage_U = sum(log.(max.(wmid, 1e-14)) .* dens_U) * bw
    mean_log_wage_S = sum(log.(max.(wmid, 1e-14)) .* dens_S) * bw
    wage_premium    = mean_log_wage_S - mean_log_wage_U

    # ── Tightness ─────────────────────────────────────────────────────────
    theta_U = obj.thetaU
    theta_S = obj.thetaS

    return (
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

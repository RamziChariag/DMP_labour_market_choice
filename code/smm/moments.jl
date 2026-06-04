############################################################
# moments.jl — Empirical moment targets (v7)
#
# Loads moments from CSV files produced by data_pipeline_v7 and
# computes the matching model moments from a solved equilibrium.
#
# Wage premium convention
#   wage_premium  ≡  E[log w_S] − E[log w_U]
#
# Changes vs. previous version (data_and_moments.pdf Part VII):
#   - 24 moments (dropped exp_ur_total / exp_ur_U / exp_ur_S).
#   - Added train_entry_rate_U: monthly hazard of an
#     unskilled-unemployed worker becoming enrolled in training.
#     Model counterpart: ∫_{x̄}^1 u_U(x) dx / agg_uU, equivalently
#     the τ-weighted share of unskilled unemployment.
#   - load_calibrated_params now reads nu_estimation.csv (two rows)
#     and accepts a `window` so each crisis pair gets the ν of its
#     own baseline (FC pair → ν_base_fc; COVID pair → ν_base_covid).
#
# v7.1 — NSC-based training_share level adjustment (κ_w):
#   - The CPS SCHLCOLL universe expanded in Jan 2013 (16–24 → 16–54),
#     so the raw CPS training_share level is not directly comparable
#     across the FC and COVID windows. The new pipeline cell
#     "CPS vs NSC enrolment — per-window level adjustment (κ)" writes
#     derived/training_share_scale.csv with one κ_w per window,
#     κ_w = NSC_IPEDS_enr_w / CPS_enr_w.
#   - Strategy (Option B from the design discussion):
#       data_target          ←  κ_w · CPS_training_share
#       Σ̂[ts, ts]            ←  κ_w² · Σ̂[ts, ts]
#       Σ̂[ts,  ·] (off-diag) ←  κ_w  · Σ̂[ts,  ·]
#     The full CPS off-diagonal covariance structure is preserved up
#     to the linear κ scaling on the training_share row/column.
#   - The model-side training_share is already age-uncapped
#     (agg_t / total_pop), so κ-scaling the data target makes the two
#     conceptually comparable.
#   - If training_share_scale.csv is missing, κ_w defaults to 1.0
#     with a warning (back-compat with older pipeline runs).
############################################################


# ============================================================
# Moment names (canonical order — 24 moments)
# ============================================================

const MOMENT_NAMES = [
    :ur_total, :ur_U, :ur_S,
    :skilled_share, :training_share,
    :emp_var_U, :emp_cm3_U, :emp_var_S, :emp_cm3_S,
    :jfr_U, :sep_rate_U, :jfr_S, :sep_rate_S,
    :ee_rate_S, :train_entry_rate_U,
    :mean_wage_U, :mean_wage_S,
    :p25_wage_U, :p25_wage_S, :p50_wage_U, :p50_wage_S,
    :wage_premium, :theta_U, :theta_S,
]

@assert length(MOMENT_NAMES) == 24 "Expected 24 moments, got $(length(MOMENT_NAMES))"


"""
    load_training_share_scale(; window::Symbol, derived_dir::String) → Float64

Return κ_w for the given window, read from
`derived/training_share_scale.csv` produced by the "CPS vs NSC
enrolment" cell in data_pipeline_v7. Returns 1.0 with a warning if
the file is missing or the window has no row (back-compat with
older pipeline runs that did not write the file).
"""
function load_training_share_scale(; window::Symbol, derived_dir::String) :: Float64
    path = joinpath(derived_dir, "training_share_scale.csv")
    if !isfile(path)
        @warn "training_share_scale.csv not found in $derived_dir — using κ = 1.0 " *
              "(no NSC-based level adjustment applied). Re-run the CPS-vs-NSC " *
              "cell in data_pipeline_v7 to enable κ."
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
`moments_{window}.csv` produced by data_pipeline_v7.

The training_share row is rescaled by κ_w from
`training_share_scale.csv` so the data target reflects the NSC
IPEDS-Universe level (age-comparable across the FC and COVID
windows). If the κ file is missing, κ = 1.0 and a warning is
emitted.

Moment list
  Labour-market stocks (5)
    ur_total, ur_U, ur_S, skilled_share, training_share

  Wage shape (4)
    emp_var_U, emp_cm3_U, emp_var_S, emp_cm3_S

  Transition rates (6)
    jfr_U, sep_rate_U, jfr_S, sep_rate_S, ee_rate_S, train_entry_rate_U

  Wages (7)
    mean_wage_U, mean_wage_S,
    p25_wage_U, p25_wage_S, p50_wage_U, p50_wage_S,
    wage_premium

  Tightness (2)
    theta_U, theta_S
"""
function load_data_moments(; window::Symbol = :base_fc, derived_dir::String)
    moments_file = joinpath(derived_dir, "moments_$(window).csv")
    isfile(moments_file) || error("Moments file not found: $moments_file — run data_pipeline_v7 first.")
    raw = _read_moments_csv(moments_file)

    # κ-rescale training_share to NSC level (see header note).
    κ = load_training_share_scale(; window=window, derived_dir=derived_dir)
    if haskey(raw, :training_share) && κ != 1.0
        old_val = raw.training_share.value
        new_val = κ * old_val
        @printf("  Applied κ_w to training_share level: %.5f → %.5f  (κ_%s = %.4f)\n",
                old_val, new_val, window, κ)
        # Rebuild NamedTuple with the scaled training_share entry.
        scaled_pairs = [k => (k === :training_share ?
                              (value = new_val, weight = v.weight) : v)
                        for (k, v) in Base.pairs(raw)]
        return NamedTuple(scaled_pairs)
    end
    return raw
end


"""
    load_weight_matrix(; window, derived_dir, cond_target,
                         skip_moments) → Union{Nothing, Matrix{Float64}}

Build the SMM weighting matrix from `sigma_{window}.csv`,
subsetted to the active moments (i.e. those in `MOMENT_NAMES`
that are not in `skip_moments`).

Conditioning behaviour (controlled by `cond_target`):
  cond_target == 0.0  pure diagonal W = inv(diag(Σ))
  cond_target == 1.0  compressed diagonal:  W_kk = log(1 + 1/σ_k²)
  cond_target == 2.0  equal weights — returns `nothing`
  cond_target  > 2.0  full optimal W = inv(Σ); if κ(Σ) > cond_target,
                       Σ is shrunk toward its diagonal by bisection
                       on ρ ∈ [0, 1] in  Σ(ρ) = (1−ρ) diag(Σ) + ρ Σ
                       until κ(Σ(ρ)) ≤ cond_target, then W = inv(Σ(ρ)).
"""
function load_weight_matrix(;
    window::Symbol = :base_fc,
    derived_dir::String,
    cond_target::Float64 = 1e8,
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

    sigma_file = joinpath(derived_dir, "sigma_$(window).csv")
    isfile(sigma_file) || error(
        "sigma_$(window).csv not found in $derived_dir — run data_pipeline_v7 first.")
    df_sig   = CSV.read(sigma_file, DataFrame)
    csv_cols = Symbol.(names(df_sig))
    active_col_idx = [findfirst(==(nm), csv_cols) for nm in active_names]
    missing_cols = active_names[isnothing.(active_col_idx)]
    isempty(missing_cols) || error(
        "sigma_$(window).csv is missing columns for active moments: " *
        join(string.(missing_cols), ", ") *
        " — re-run data_pipeline_v7.")

    # Convert to Matrix once; all downstream operations work on Σ_full.
    Σ_full = Matrix{Float64}(df_sig)

    # Apply κ_w to the training_share row/column of Σ̂ so it stays
    # consistent with the κ-scaled data target produced by
    # load_data_moments. Off-diagonals × κ, diagonal × κ². Done on
    # the FULL Σ̂ before subsetting so it works whether or not
    # training_share is in `active_names`.
    κ = load_training_share_scale(; window=window, derived_dir=derived_dir)
    if κ != 1.0
        ts_idx = findfirst(==(:training_share), csv_cols)
        if isnothing(ts_idx)
            @warn "Σ̂ has no training_share column — κ scaling on Σ̂ skipped."
        else
            Σ_full[ts_idx, :] .*= κ
            Σ_full[:, ts_idx] .*= κ
            # The (ts_idx, ts_idx) cell got multiplied twice → κ², which is
            # the correct rescaling for the variance of a κ-scaled moment.
            @printf("  Applied κ_w = %.4f to training_share row/col of Σ̂  (diagonal × κ² = %.4f).\n", κ, κ^2)
        end
    end

    if cond_target == 0.0
        d = diag(Σ_full[active_col_idx, active_col_idx])
        W = Diagonal(1.0 ./ max.(d, 1e-14))
        @printf("  cond_target == 0.0 → diagonal W from active submatrix of %s\n", sigma_file)
        return Matrix(W)
    end

    if cond_target == 1.0
        d            = diag(Σ_full[active_col_idx, active_col_idx])
        d_inv        = 1.0 ./ max.(d, 1e-14)
        d_compressed = log.(1.0 .+ d_inv)
        W = Diagonal(d_compressed)
        @printf("  cond_target == 1.0 → compressed diagonal W from active submatrix of %s\n", sigma_file)
        @printf("    raw 1/σ² range:         [%.4e, %.4e]\n", minimum(d_inv), maximum(d_inv))
        @printf("    compressed diag range:   [%.4e, %.4e]\n", minimum(d_compressed), maximum(d_compressed))
        return Matrix(W)
    end

    # Full W.  Shrinkage is applied to Σ before inverting so that
    # W = inv(Σ_shrunk) is guaranteed PD with positive diagonal.
    Σ      = Σ_full[active_col_idx, active_col_idx]
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

Shrink the off-diagonals of symmetric W so that κ(W_shrunk) ≤ cond_target
via the convex combination W(α) = (1−α) W + α diag(W),  α ∈ [0, 1].
"""
function _shrink_to_target(
    W::Matrix{Float64},
    cond_target::Float64;
    tol::Float64  = 1e-3,
    maxiter::Int  = 100,
)
    D = Diagonal(diag(W))

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
            hi = alpha
        else
            lo = alpha
        end
        (hi - lo) < tol && break
    end

    alpha = hi
    W_shrunk .= (1.0 - alpha) .* W .+ alpha .* D
    return W_shrunk, alpha
end


"""
    load_sigma_matrix(; window::Symbol = :base_fc, derived_dir::String) → Vector{Float64}

Read `sigma_{window}.csv` and return the vector of standard errors
(square root of the diagonal). The training_share entry is scaled
by κ_w so the returned σ refers to the κ-rescaled data target.
"""
function load_sigma_matrix(; window::Symbol = :base_fc, derived_dir::String)
    sigma_file = joinpath(derived_dir, "sigma_$(window).csv")
    isfile(sigma_file) || error("Sigma file not found: $sigma_file — run data_pipeline_v7 first.")
    σ_vec = _read_sigma_csv(sigma_file)

    κ = load_training_share_scale(; window=window, derived_dir=derived_dir)
    if κ != 1.0
        df = CSV.read(sigma_file, DataFrame)
        ts_idx = findfirst(==(:training_share), Symbol.(names(df)))
        if !isnothing(ts_idx)
            σ_vec[ts_idx] *= κ
        end
    end
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
        "nu_estimation.csv not found in $derived_dir — run data_pipeline_v7 first.")
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
        "phi_calibration.csv not found in $derived_dir — run data_pipeline_v7 first.")
    df_phi = CSV.read(phi_file, DataFrame)
    phi_val = Float64(df_phi.phi[1])
    @printf("  Loaded phi = %.5f from %s\n", phi_val, phi_file)

    return (r = r_val, nu = nu_val, phi = phi_val)
end


# ============================================================
# Windows.json loader (single source of truth from data_pipeline_v7)
# ============================================================

"""
    load_windows(; derived_dir::String) → NamedTuple

Read `windows.json` (written by data_pipeline_v7) and return
`(windows = Dict{Symbol,NamedTuple}, order = Vector{Symbol})`.
"""
function load_windows(; derived_dir::String)
    win_path = joinpath(derived_dir, "windows.json")
    isfile(win_path) || error(
        "windows.json not found in $derived_dir — run data_pipeline_v7 first.")
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

train_entry_rate_U (v7, new): the model counterpart of the data
flow hazard is the τ-weighted share of unskilled unemployment
(eqn ∫_{x̄}^1 u_U(x) dx / agg_uU). Equivalently, since τ(x) =
1{x > x̄}, this is dot(τ, uU .* wx) / agg_uU.
"""
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

    # Variance and third central moment of the employed wage
    # distribution.  Zero by construction when the density is zero.
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

    # Transition rates
    jfr_U      = _has_eU ? obj.f_U : 0.0
    jfr_S      = _has_eS ? obj.f_S : 0.0
    sep_rate_U = obj.sep_rate_U
    sep_rate_S = obj.sep_rate_S
    ee_rate_S  = obj.ee_rate_S

    # train_entry_rate_U (v7) — model counterpart of the data flow hazard.
    # Spec: ∫_{x̄}^1 u_U(x) dx / agg_uU.  τT carries the optimal training
    # indicator on the x-grid, so the τ-weighted unemployment integral
    # equals the numerator exactly.
    _agg_uU_w = dot(obj.uU, obj.wx)
    train_entry_rate_U = _agg_uU_w > _emp_tol ?
        dot(obj.tauT .* obj.uU, obj.wx) / _agg_uU_w : 0.0

    # Wages
    wmid   = obj.wmid
    dens_U = obj.dens_U
    dens_S = obj.dens_S
    bw     = length(wmid) >= 2 ? wmid[2] - wmid[1] : 1.0

    function _percentile(wmid, dens, bw, target)
        cum = 0.0
        for j in eachindex(wmid)
            mass = dens[j] * bw
            if cum + mass >= target
                frac = mass > 1e-14 ? (target - cum) / mass : 0.5
                return wmid[j] - bw/2 + frac * bw
            end
            cum += mass
        end
        return 0.0
    end

    if _has_eU
        mean_wage_U     = sum(wmid .* dens_U) * bw
        p25_wage_U      = _percentile(wmid, dens_U, bw, 0.25)
        p50_wage_U      = _percentile(wmid, dens_U, bw, 0.50)
        mean_log_wage_U = sum(log.(max.(wmid, 1e-14)) .* dens_U) * bw
    else
        mean_wage_U     = 0.0
        p25_wage_U      = 0.0
        p50_wage_U      = 0.0
        mean_log_wage_U = 0.0
    end

    if _has_eS
        mean_wage_S     = sum(wmid .* dens_S) * bw
        p25_wage_S      = _percentile(wmid, dens_S, bw, 0.25)
        p50_wage_S      = _percentile(wmid, dens_S, bw, 0.50)
        mean_log_wage_S = sum(log.(max.(wmid, 1e-14)) .* dens_S) * bw
    else
        mean_wage_S     = 0.0
        p25_wage_S      = 0.0
        p50_wage_S      = 0.0
        mean_log_wage_S = 0.0
    end

    wage_premium = (_has_eU && _has_eS) ?
                   (mean_log_wage_S - mean_log_wage_U) : 0.0

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
        train_entry_rate_U = train_entry_rate_U,

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
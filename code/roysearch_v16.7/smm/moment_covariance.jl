############################################################
# moment_covariance.jl — decide WHICH moments to drop
#
# Place in  code/smm/  and run:
#     julia --project=. moment_covariance.jl
#
# The companion moment_diagnostics.jl showed the Σ̂ matrices are not merely
# ill-conditioned but RANK-DEFICIENT (numerically singular): several moments
# are near-exact linear combinations of the others, so no weighting scheme can
# invert Σ̂ honestly.  Greedy-on-cond can't fix that (it plateaus).  This script
# instead does rank-revealing FORWARD SELECTION on the moment correlation
# structure to produce a concrete drop list.
#
# METHOD
#   Work on the correlation matrix R_w of the (raw-units) Σ̂ in every window w.
#   Maintain a "keep" set S.  Repeatedly add the moment whose conditional
#   variance given S — i.e. the share of its variation NOT explained by the
#   moments already kept — is largest in its WORST window.  Stop once even the
#   best remaining moment is explained to within CONDVAR_FLOOR in some window.
#   Conditional variance is the Schur complement  cv_j = 1 − R²_{j|S}; it is the
#   exact pivot a rank-revealing Cholesky would use.  Selecting only moments
#   with cv ≥ floor guarantees the kept set is well-conditioned in every window.
#
#   Everything not selected is redundant → DROP it.  For each dropped moment we
#   report R²_{j|keep} (how explained it is) and the kept moments that explain
#   it, so the choice is transparent rather than mechanical.
#
# NOTE on units (see moment_diagnostics.jl header): selection uses the raw-units
# correlation structure, which is what the efficient-GMM weight Σ̂⁻¹ depends on.
# Redundancy is a property of the correlation pattern, so raw-vs-scaled does not
# change WHICH moments are dependent — only the absolute cond numbers.
############################################################

using LinearAlgebra, Statistics, Printf, CSV, DataFrames

const SMM_DIR      = @__DIR__
const PROJECT_ROOT = joinpath(SMM_DIR, "..", "..")
const DERIVED_DIR  = joinpath(PROJECT_ROOT, "data", "derived")

include(joinpath(SMM_DIR, "moments.jl"))   # MOMENT_NAMES, load_training_share_scale, ...

# ============================================================
# CONFIG — edit these
# ============================================================
const APPLY_KAPPA    = true       # κ on training_share row/col (matches moments.jl)
const CONDVAR_FLOOR  = 1.0e-3     # keep a moment only if ≥0.1% of it is independent
                                  # of the kept set in its WORST window
const DETAIL_WIN     = :base_fc   # window used for the "explained-by" detail
const THRESH_SCAN    = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-6]   # tradeoff table

# Moments to force-keep regardless (core targets you want identified).
# NOTE: ur_total is deliberately NOT here — it is a labour-force-weighted average
# of ur_U and ur_S, so it is one of the redundancies the routine should be free
# to drop. Protecting it would force-keep a dependency. Keep this set to moments
# that are economically essential AND mutually independent.
const PROTECTED = Symbol[:ur_U, :ur_S, :skilled_share, :training_share]

# ============================================================
# Loaders (mirror the κ convention in moments.jl)
# ============================================================
available_windows() = filter([:base_fc,:crisis_fc,:base_covid,:crisis_covid]) do w
    isfile(joinpath(DERIVED_DIR, "sigma_$(w).csv"))
end

const _CACHE = Dict{Symbol,Tuple{Vector{Symbol},Matrix{Float64},Matrix{Float64}}}()

function load_window(window::Symbol)
    get!(_CACHE, window) do
        df   = CSV.read(joinpath(DERIVED_DIR, "sigma_$(window).csv"), DataFrame)
        cols = Symbol.(names(df))
        Σs   = Matrix{Float64}(df)
        if APPLY_KAPPA
            κ = load_training_share_scale(; window=window, derived_dir=DERIVED_DIR)
            if κ != 1.0
                ts = findfirst(==(:training_share), cols)
                ts !== nothing && (Σs[ts, :] .*= κ; Σs[:, ts] .*= κ)
            end
        end
        sf = joinpath(DERIVED_DIR, "moment_scales_$(window).csv")
        D  = if isfile(sf)
            sdf  = CSV.read(sf, DataFrame)
            smap = Dict(Symbol(r.moment) => Float64(r.scale) for r in eachrow(sdf))
            Diagonal([get(smap, c, 1.0) for c in cols])
        else
            @warn "moment_scales_$(window).csv missing — 'raw' = 'scaled'."
            Diagonal(ones(length(cols)))
        end
        Σr = Matrix(D * Σs * D); Σr .= (Σr .+ Σr') ./ 2
        (cols, Σs, Σr)
    end
end

function submatrix(window::Symbol, names::Vector{Symbol}; units::Symbol=:raw)
    cols, Σs, Σr = load_window(window)
    Σ   = units === :raw ? Σr : Σs
    idx = [findfirst(==(m), cols) for m in names]
    @assert all(!isnothing, idx) "sigma_$(window).csv missing: $(names[isnothing.(idx)])"
    return Symmetric(Σ[idx, idx])
end

# ============================================================
# Math helpers
# ============================================================
safe_cond(Σ) = cond(Matrix(Σ))

function corr_from_cov(Σ)
    d = sqrt.(max.(diag(Σ), 1e-300))
    R = Matrix(Σ) ./ (d * d')
    R .= (R .+ R') ./ 2
    return R
end

numerical_rank(Σ; rtol=1e-10) =
    (e = eigen(Symmetric(Matrix(Σ))).values; count(>(rtol * maximum(e)), e))

# conditional variance of moment j given the kept set S, on a correlation matrix R.
function cond_var(R, S::Vector{Int}, j::Int; ridge=1e-12)
    isempty(S) && return R[j, j]
    A = Symmetric(Matrix(R[S, S]) + ridge * I)
    return R[j, j] - dot(R[S, j], A \ R[S, j])
end

# kept moments that best explain j (shares of R²_{j|S}, largest first).
function explain(R, S::Vector{Int}, j::Int; n=3, ridge=1e-12)
    isempty(S) && return Tuple{Symbol,Float64}[]
    b = Symmetric(Matrix(R[S, S]) + ridge * I) \ R[S, j]
    contrib = [(MOMENT_NAMES[S[i]], b[i] * R[S[i], j]) for i in eachindex(S)]
    sort!(contrib; by = x -> abs(x[2]), rev = true)
    return contrib[1:min(n, length(contrib))]
end

# Forward selection across windows (min conditional variance over windows).
function forward_select(Rmats, floor_; protected_idx=Int[])
    K     = size(first(Rmats), 1)
    S     = copy(protected_idx)
    order = Tuple{Symbol,Float64,Bool}[]        # (moment, min-cv-at-add, protected?)
    for pj in protected_idx
        push!(order, (MOMENT_NAMES[pj], NaN, true))
    end
    remaining = setdiff(collect(1:K), S)
    while !isempty(remaining)
        scored = [(j, minimum(cond_var(R, S, j) for R in Rmats)) for j in remaining]
        bi     = argmax([s[2] for s in scored])
        j, cv  = scored[bi]
        cv < floor_ && break
        push!(S, j); push!(order, (MOMENT_NAMES[j], cv, false))
        remaining = setdiff(remaining, [j])
    end
    return S, remaining, order
end

# ============================================================
# Report
# ============================================================
const WINDOWS = available_windows()
isempty(WINDOWS) && error("No sigma_*.csv found in $DERIVED_DIR")

function main()
    win_str = join(string.(WINDOWS), ", ")
    println("="^78)
    println("  MOMENT SELECTION — what to drop")
    println("  windows = $win_str   |   κ applied = $APPLY_KAPPA   |   floor = $CONDVAR_FLOOR")
    println("="^78)

    Rmats  = [corr_from_cov(submatrix(w, MOMENT_NAMES; units=:raw)) for w in WINDOWS]
    protid = [findfirst(==(m), MOMENT_NAMES) for m in PROTECTED]
    @assert all(!isnothing, protid) "PROTECTED has an unknown moment name."
    protid = Vector{Int}(protid)

    # --- [1] How rank-deficient is each window? ----------------------------
    println("\n[1] NUMERICAL RANK per window (raw units)  — deficiency = #redundant moments")
    @printf("    %-14s  %6s  %6s  %10s\n", "window", "K", "rank", "deficiency")
    println("    " * "-"^42)
    for w in WINDOWS
        Σ = submatrix(w, MOMENT_NAMES; units=:raw)
        r = numerical_rank(Σ)
        @printf("    %-14s  %6d  %6d  %10d\n", w, length(MOMENT_NAMES), r, length(MOMENT_NAMES)-r)
    end

    # --- [2] Forward-selection ranking -------------------------------------
    keep, drop, order = forward_select(Rmats, CONDVAR_FLOOR; protected_idx=protid)
    println("\n[2] FORWARD-SELECTION ORDER  (marginal independent info each moment adds)")
    println("    cv = min over windows of conditional variance given everything above it.")
    println("    Selection stops at the line where cv falls below the floor ($CONDVAR_FLOOR).")
    @printf("    %3s  %-22s  %12s\n", "#", "moment", "min cv")
    println("    " * "-"^42)
    for (i, (m, cv, isprot)) in enumerate(order)
        tag = isprot ? "   (protected)" : ""
        cvs = isprot ? "      —" : @sprintf("%.3e", cv)
        @printf("    %3d  %-22s  %12s%s\n", i, m, cvs, tag)
    end
    if length(order) < length(MOMENT_NAMES)
        println("    ── floor reached; everything below is redundant given the set above ──")
    end

    # --- [3] KEEP / DROP ----------------------------------------------------
    keep_names = MOMENT_NAMES[keep]
    drop_names = MOMENT_NAMES[drop]
    @printf("\n[3] RESULT at floor=%.0e :  KEEP %d   DROP %d\n",
            CONDVAR_FLOOR, length(keep_names), length(drop_names))
    println("    KEEP: ", join(string.(keep_names), ", "))
    println("    DROP: ", isempty(drop_names) ? "(none)" : join(string.(drop_names), ", "))

    # --- [4] Why each dropped moment is redundant --------------------------
    if !isempty(drop)
        Rdet = corr_from_cov(submatrix(DETAIL_WIN, MOMENT_NAMES; units=:raw))
        println("\n[4] WHY EACH DROPPED MOMENT IS REDUNDANT")
        println("    R² = share explained by the KEPT set (min cv over windows ⇒ max R²).")
        println("    'explained by' uses window=$DETAIL_WIN.")
        # most-redundant first
        scored = [(j, minimum(cond_var(R, keep, j) for R in Rmats)) for j in drop]
        sort!(scored; by = x -> x[2])
        for (j, cv) in scored
            r2  = 1 - clamp(cv, 0.0, 1.0)
            exps = explain(Rdet, keep, j; n=3)
            estr = join([@sprintf("%.2f·%s", v, m) for (m, v) in exps], " + ")
            @printf("    %-20s  R²=%.5f   ≈ %s\n", MOMENT_NAMES[j], r2, estr)
        end
    end

    # --- [5] Recommendation + verification ---------------------------------
    println("\n[5] RECOMMENDED SKIP_MOMENTS  (paste into smm_main.jl)")
    println("    SKIP_MOMENTS = Symbol[")
    for s in sort(string.(drop_names)); println("        :$s,"); end
    println("    ]")

    println("\n    Conditioning of the KEPT set (cond Σ̂ should now be finite/moderate):")
    @printf("    %-14s  %16s  %16s\n", "window", "cond(raw)", "cond(scaled)")
    println("    " * "-"^50)
    for w in WINDOWS
        cr = safe_cond(submatrix(w, keep_names; units=:raw))
        cs = safe_cond(submatrix(w, keep_names; units=:scaled))
        @printf("    %-14s  %16.3e  %16.3e\n", w, cr, cs)
    end

    # --- [6] Tradeoff: floor vs #kept vs worst conditioning ----------------
    println("\n[6] FLOOR TRADEOFF  (stricter floor ⇒ fewer moments, better conditioning)")
    @printf("    %10s  %6s  %16s  %s\n", "floor", "#keep", "max cond(raw)", "dropped")
    println("    " * "-"^78)
    for θ in THRESH_SCAN
        k, d, _ = forward_select(Rmats, θ; protected_idx=protid)
        kn = MOMENT_NAMES[k]
        mc = maximum(safe_cond(submatrix(w, kn; units=:raw)) for w in WINDOWS)
        dd = isempty(d) ? "(none)" : join(string.(MOMENT_NAMES[d]), ",")
        @printf("    %10.0e  %6d  %16.3e  %s\n", θ, length(k), mc, dd)
    end

    println("\nDone.")
end

main()
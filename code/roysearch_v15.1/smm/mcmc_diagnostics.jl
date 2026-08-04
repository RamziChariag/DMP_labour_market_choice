############################################################
# smm/mcmc_diagnostics.jl — gate diagnostics for the DE-MC chain
#
# Every quantity the inference gates are judged on, computed here and printed
# ONCE by MCMC_main.jl. Nothing in this file runs the model: it consumes the
# chain (d × N × gens), the stored per-draw moment matrix, and the spec.
#
# Reference for the inference theory: Chernozhukov–Hong (2003), "An MCMC
# Approach to Classical Estimation", J. Econometrics 115(2).
#   · Assumptions 1–4 (θ₀ interior; continuous uniformly positive prior;
#     identifiability; local quadratic expansion) → Theorems 1–2:
#     the chain concentrates at J⁻¹/n and its mean is √n-consistent.
#     None of these requires efficient weighting.
#   · Theorem 3 (chain QUANTILES are a CI) additionally needs the generalized
#     information equality J·Ω⁻¹ → I, i.e. W = Ω⁻¹. Diagonal W fails it.
#   · Theorem 4 needs NO information equality: take Ĵ⁻¹ = n·Cov(chain) and
#     combine with any available Ω̂ in the Huber sandwich Ĵ⁻¹Ω̂Ĵ⁻¹.
#     This is the path used here.
#
# Plain include() file: definitions only, no top-level execution.
############################################################

# ─────────────────────────────────────────────────────────────────────────────
# Convergence
# ─────────────────────────────────────────────────────────────────────────────

"""
    split_rhat_ess(chain, burn) -> (rhat, ess)

Rank-normalised split-R̂ and effective sample size per parameter
(Vehtari et al. 2021), on the post-burn-in half of each chain split in two.
Rank-normalisation makes R̂ robust to the heavy tails a saturated coordinate
produces, which is exactly the failure mode being screened for.
"""
function split_rhat_ess(chain::Array{Float64,3}, burn::Int)
    d, N, G = size(chain)
    kept    = G - burn
    kept < 4 && return (fill(NaN, d), fill(NaN, d))
    h       = kept ÷ 2
    M       = 2N                                    # split chains
    rhat = fill(NaN, d); ess = fill(NaN, d)

    for k in 1:d
        # split into M sequences of length h, then rank-normalise pooled draws
        seqs = Vector{Vector{Float64}}(undef, M)
        for c in 1:N
            seqs[2c-1] = chain[k, c, burn+1     : burn+h]
            seqs[2c]   = chain[k, c, burn+h+1   : burn+2h]
        end
        pooled = vcat(seqs...)
        allequal_pooled = all(==(pooled[1]), pooled)
        if allequal_pooled                          # degenerate coordinate
            rhat[k] = 1.0; ess[k] = 0.0; continue
        end
        r  = invperm(sortperm(pooled))               # ranks, 1..length
        z  = [_norminvcdf((r[i] - 0.375) / (length(pooled) + 0.25)) for i in eachindex(r)]
        zs = [z[(m-1)*h+1 : m*h] for m in 1:M]

        means = mean.(zs); vars = var.(zs)
        Wv = mean(vars); Bv = var(means) * h
        Wv <= 0 && (rhat[k] = 1.0; ess[k] = 0.0; continue)
        varplus  = ((h - 1) * Wv + Bv) / h
        rhat[k]  = sqrt(varplus / Wv)

        # ESS from the combined autocorrelation, truncated at the first
        # negative pair sum (Geyer's initial positive sequence).
        ρsum = 0.0
        for lag in 1:(h - 2)
            ρ = mean(_autocorr_at(zs[m], lag) * vars[m] for m in 1:M) / Wv
            lag2 = lag + 1
            lag2 > h - 1 && break
            ρ2 = mean(_autocorr_at(zs[m], lag2) * vars[m] for m in 1:M) / Wv
            (ρ + ρ2) < 0 && break
            ρsum += ρ + ρ2
        end
        ess[k] = clamp(M * h / (1 + 2ρsum), 0.0, Float64(M * h))
    end
    return (rhat, ess)
end

# Acklam-style rational approximation to Φ⁻¹; ample for rank-normalisation.
function _norminvcdf(p::Float64)
    p = clamp(p, 1e-15, 1 - 1e-15)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    dd = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow
        q = sqrt(-2log(p))
        return (((((c[1]q + c[2])q + c[3])q + c[4])q + c[5])q + c[6]) /
               ((((dd[1]q + dd[2])q + dd[3])q + dd[4])q + 1)
    elseif p <= phigh
        q = p - 0.5; r = q * q
        return (((((a[1]r + a[2])r + a[3])r + a[4])r + a[5])r + a[6]) * q /
               (((((b[1]r + b[2])r + b[3])r + b[4])r + b[5])r + 1)
    else
        q = sqrt(-2log(1 - p))
        return -(((((c[1]q + c[2])q + c[3])q + c[4])q + c[5])q + c[6]) /
                ((((dd[1]q + dd[2])q + dd[3])q + dd[4])q + 1)
    end
end

function _autocorr_at(x::Vector{Float64}, lag::Int)
    n = length(x); lag >= n && return 0.0
    m = mean(x); s = 0.0
    @inbounds for i in 1:(n - lag)
        s += (x[i] - m) * (x[i+lag] - m)
    end
    return s / (n - lag)
end

# ─────────────────────────────────────────────────────────────────────────────
# Interiority (Chernozhukov–Hong Assumption 1)
# ─────────────────────────────────────────────────────────────────────────────

"""
    boundary_mass(draws, free; frac=0.01) -> (lo, hi)

Fraction of post-burn-in draws within `frac` of each box edge, in CONSTRAINED
units.  CH Assumption 1 puts θ₀ in the interior of Θ; a parameter piling against
an edge has no theorem covering it and its dispersion measures the box, not the
data, so it must be reported as bound-constrained rather than given a standard
error.
"""
function boundary_mass(draws::Matrix{Float64}, free::Vector{ParamSpec}; frac::Float64 = 0.01)
    d, T = size(draws)
    lo = zeros(d); hi = zeros(d)
    @inbounds for k in 1:d
        ps = free[k]; w = frac * (ps.ub - ps.lb)
        nlo = 0; nhi = 0
        for t in 1:T
            θ = _to_constrained(draws[k, t], ps.lb, ps.ub)
            θ - ps.lb < w && (nlo += 1)
            ps.ub - θ < w && (nhi += 1)
        end
        lo[k] = nlo / T; hi[k] = nhi / T
    end
    return (lo, hi)
end

"""
    spread_growth(chain, burn) -> Vector{Float64}

Ratio of the cross-chain SD of each UNCONSTRAINED coordinate in the last
post-burn-in decile to that in the first.  A sampled direction plateaus (ratio
≈ 1); a direction diffusing because the target is improper keeps widening, and
its "standard error" is then a function of the generation count rather than of
the data.  This is the direct test that the log-Jacobian is doing its job.
"""
function spread_growth(chain::Array{Float64,3}, burn::Int)
    d, N, G = size(chain)
    kept = G - burn
    kept < 20 && return fill(NaN, d)
    w  = max(1, kept ÷ 10)
    r  = fill(NaN, d)
    @inbounds for k in 1:d
        a = std(vec(chain[k, :, burn+1     : burn+w]))
        b = std(vec(chain[k, :, G-w+1      : G]))
        r[k] = a > 0 ? b / a : NaN
    end
    return r
end

# ─────────────────────────────────────────────────────────────────────────────
# Jacobian, sandwich, and the Ω-free bound
# ─────────────────────────────────────────────────────────────────────────────

"""
    jacobian_from_draws(draws, M, free) -> (G, R²)

Rows of Ĝ = ∂g/∂θ′ by OLS of each stored model moment on the draws, in
CONSTRAINED parameter units.  Over thousands of evaluations this is far more
stable than a finite difference on a solver whose tolerance is comparable to a
small perturbation.  `R²` per moment is the validity check: a low value means
the posterior region is too wide for the local linear approximation.
"""
function jacobian_from_draws(draws::Matrix{Float64}, M::Matrix{Float64},
                             free::Vector{ParamSpec})
    d, T = size(draws); K = size(M, 1)
    @assert size(M, 2) == T "moment matrix and draws disagree on draw count"
    Θ = Matrix{Float64}(undef, T, d)
    @inbounds for k in 1:d, t in 1:T
        Θ[t, k] = _to_constrained(draws[k, t], free[k].lb, free[k].ub)
    end
    X  = hcat(ones(T), Θ)
    Gm = Matrix{Float64}(undef, K, d); R2 = fill(NaN, K)
    for i in 1:K
        y = @view M[i, :]
        β = X \ y
        Gm[i, :] = β[2:end]
        ŷ  = X * β
        ss = sum(abs2, y .- mean(y))
        R2[i] = ss > 0 ? 1 - sum(abs2, y .- ŷ) / ss : NaN
    end
    return (Gm, R2)
end

"""
    se_bound_diagonal(G, W, σ̂) -> (se_bound, se_curv)

Standard errors from CH Theorem 4 with the ONLY input about Ω being its
diagonal.  With `a_j = W G J⁻¹ e_j` and `J = G′WG`, the sandwich variance is
`a_j′Ω a_j ≤ (Σ_i |a_ij| σ̂_i)²`, with equality at the rank-one Ω aligned with
sign(a_j) — so the bound is sharp and attainable, not a slack inequality.

No off-diagonal element of Ω is estimated, assumed, or set to zero; no two
moments are ever compared, so moments observed at different frequencies and
from different surveys raise no issue here.  The bound therefore holds for the
true Ω whatever it is, and a reported interval built on it cannot be too
narrow.  Returned alongside `se_curv = sqrt(diag(J⁻¹))`, the curvature-only
standard error (exact under efficient weighting; CH Theorem 3).
"""
function se_bound_diagonal(G::Matrix{Float64}, W::Matrix{Float64}, σ̂::Vector{Float64})
    J  = G' * W * G
    Ji = pinv(J)
    A  = W * G * Ji                                  # column j = a_j
    d  = size(G, 2)
    se_bound = Vector{Float64}(undef, d)
    @inbounds for j in 1:d
        se_bound[j] = sum(abs.(@view A[:, j]) .* σ̂)
    end
    return (se_bound, sqrt.(abs.(diag(Ji))))
end

"""
    curvature_check(draws, G, W, free) -> Float64

Median |log₁₀| discrepancy between two independent routes to J: the inverse
covariance of the draws, and Ĝ′WĜ.  They estimate the same matrix, so a large
discrepancy indicts the chain (short burn-in, or an improper target) rather than
the Jacobian.
"""
function curvature_check(draws::Matrix{Float64}, G::Matrix{Float64},
                         W::Matrix{Float64}, free::Vector{ParamSpec})
    d, T = size(draws)
    Θ = Matrix{Float64}(undef, T, d)
    @inbounds for k in 1:d, t in 1:T
        Θ[t, k] = _to_constrained(draws[k, t], free[k].lb, free[k].ub)
    end
    Jc = pinv(cov(Θ))
    Jg = G' * W * G
    dc = diag(Jc); dg = diag(Jg)
    ok = @. isfinite(dc) & isfinite(dg) & (dc > 0) & (dg > 0)
    any(ok) || return NaN
    return median(abs.(log10.(dc[ok] ./ dg[ok])))
end

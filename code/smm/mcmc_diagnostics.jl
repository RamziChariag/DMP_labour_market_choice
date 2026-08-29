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
#   · Theorem 4 needs NO information equality: combine Ĵ⁻¹ with any available Ω̂
#     in the Huber sandwich Ĵ⁻¹Ω̂Ĵ⁻¹. This is the path used here.
#
# Ĵ has two routes, and the second does not need a chain at all:
#   (a) Ĵ⁻¹ = n·Cov(chain), valid only once the chain is stationary — check
#       `seed_drift` before trusting it, since a chain still climbing toward the
#       mode yields a covariance that measures its trajectory, not the curvature;
#   (b) Ĵ = Ĝ′WĜ from a Ĝ estimated on a `local_design` around the optimum, the
#       standard GMM route, at roughly 10·d solves instead of N·gens.
# `se_bound_diagonal` consumes either.
#
# Needs Statistics, LinearAlgebra and Random; no other dependency.
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
function split_rhat_ess(chain::AbstractArray{Float64,3}, burn::Int)
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
function spread_growth(chain::AbstractArray{Float64,3}, burn::Int)
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


# ─────────────────────────────────────────────────────────────────────────────
# Sequential termination
# ─────────────────────────────────────────────────────────────────────────────

"""
    min_ess(p; alpha=0.05, eps=0.20) -> Float64

Vats–Flegal–Jones (2019, Biometrika 106) minimum effective sample size for a
`p`-dimensional target: the ESS at which a fixed-volume sequential stopping rule
terminates, giving a (1−`alpha`) confidence region whose volume is within `eps`
of the posterior's own generalized variance.

    minESS = 2^{2/p} π / (p Γ(p/2))^{2/p} · χ²_{1−α,p} / ε²

The default ε = 0.20 gives ≈ 540 at p = 25, and the threshold is nearly flat in p
(≈ 534–546 for p between 19 and 32), so one number serves every window. ε = 0.10
would demand ≈ 2159, which measurement puts out of reach here: paired with an
attainable R̂ it needs on the order of 10⁶ solves at d = 25 even on an isotropic
Gaussian, so the rule would never fire and would degrade to always spending the
full budget. ε is the relative precision of the posterior-MEAN confidence volume;
the standard errors reported here rest on Cov(chain), which converges faster.

This is a floor, not a sufficient condition — the 2.5%/97.5% quantiles converge
more slowly than the mean, which is why termination also requires R̂.

`log Γ` and the χ² quantile are written out here (Lanczos, and Wilson–Hilferty
refined by two Newton steps on the regularized incomplete gamma) so this file
needs nothing beyond `Statistics` and `LinearAlgebra`.
"""
function min_ess(p::Int; alpha::Float64 = 0.05, eps::Float64 = 0.20)
    logden = (2.0 / p) * (log(p) + _lgamma(p / 2.0))
    pref   = exp((2.0 / p) * log(2.0)) * pi / exp(logden)
    return pref * _chisq_quantile(1 - alpha, p) / eps^2
end

# Lanczos approximation, g = 7, n = 9; relative error < 1e-13 for real x > 0.
function _lgamma(x::Float64)
    x < 0.5 && return log(pi / sin(pi * x)) - _lgamma(1 - x)
    c = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
    z = x - 1.0
    a = c[1]
    t = z + 7.5
    @inbounds for i in 2:9
        a += c[i] / (z + (i - 1))
    end
    return 0.5 * log(2pi) + (z + 0.5) * log(t) - t + log(a)
end

# Regularized lower incomplete gamma P(a,x): series below the crossover,
# continued fraction above (Numerical Recipes §6.2).
function _gamma_p(a::Float64, x::Float64)
    x <= 0.0 && return 0.0
    lg = _lgamma(a)
    if x < a + 1.0
        term = 1.0 / a; sum_ = term; ap = a
        for _ in 1:500
            ap += 1.0; term *= x / ap; sum_ += term
            abs(term) < abs(sum_) * 1e-15 && break
        end
        return sum_ * exp(-x + a * log(x) - lg)
    else
        b = x + 1.0 - a; c = 1e300; dd = 1.0 / b; h = dd
        for i in 1:500
            an = -i * (i - a)
            b += 2.0
            dd = an * dd + b; abs(dd) < 1e-300 && (dd = 1e-300)
            c  = b + an / c;  abs(c)  < 1e-300 && (c  = 1e-300)
            dd = 1.0 / dd
            del = dd * c; h *= del
            abs(del - 1.0) < 1e-15 && break
        end
        return 1.0 - exp(-x + a * log(x) - lg) * h
    end
end

# χ²_{q,ν}: Wilson–Hilferty start, then Newton on P(ν/2, x/2) − q using the
# closed-form χ² density as the derivative.
function _chisq_quantile(q::Float64, ν::Int)
    a = ν / 2.0
    # Wilson-Hilferty initial guess via a normal quantile
    z = _norminvcdf(q)
    x = ν * (1.0 - 2.0 / (9ν) + z * sqrt(2.0 / (9ν)))^3
    x = max(x, 1e-8)
    for _ in 1:100
        f  = _gamma_p(a, x / 2.0) - q
        lp = (a - 1.0) * log(x / 2.0) - x / 2.0 - _lgamma(a) - log(2.0)
        dfx = exp(lp)
        dfx <= 0.0 && break
        step = f / dfx
        xn = x - step
        xn <= 0.0 && (xn = x / 2.0)
        abs(xn - x) < 1e-12 * max(1.0, x) && (x = xn; break)
        x = xn
    end
    return x
end

"""
    exempt_coordinates(chain, g, burn_frac; n_distinct_min=50, edge_frac_max=0.05, lb, ub)
      -> (exempt::BitVector, n_distinct::Vector{Int}, edge_frac::Vector{Float64})

Which coordinates cannot reach a mixing threshold at ANY budget, decided by
measurement rather than by hand.

Two mechanisms, each with its own test, because they fail for different reasons:

  FROZEN. A coordinate the proposal never moves has no distribution to mix. It is
  visible as a tiny number of distinct visited values — a frozen coordinate shows
  single digits where a live one shows hundreds. `n_distinct` counts them on the
  post-burn-in pooled draws. This is the frozen-coordinate test the diagnosis
  discipline prescribes and it was not previously computed anywhere.

  AT A BOUND. A coordinate pressed against a box edge piles up there. Under the
  logit transform the bound is at infinity in `t`, so "at the bound" is measured
  in CONSTRAINED units: the fraction of draws within 1% of the box width of either
  edge. Its sd is a description of the pile-up, not a confidence half-width, so
  requiring it to mix is requiring a quantity that does not exist.

WHY EXEMPT RATHER THAN LOWER THE THRESHOLD. The old gate took `minimum(ess)` over
all coordinates, so ONE frozen or railed coordinate held the whole run hostage: the
test could not be satisfied at any generation count, the sequential stop never
fired, and every run paid its full budget regardless of whether the reported
numbers had converged. Lowering `ess_min` instead would weaken the test for the
coordinates that DO mix, which is the opposite of what is wanted.

WHAT EXEMPTION DOES NOT MEAN. An exempt coordinate is still reported, still in the
chain, and still in every output. Exemption governs only whether it can BLOCK
termination. It is emphatically not a freeze: nothing is held fixed, and the
exemption is recomputed from the draws at every check, so a coordinate that starts
moving stops being exempt. That matters — an automatic freeze would convert a loud
diagnostic into a clean table with a spurious zero standard error, which is how a
solver defect gets hidden rather than found.
"""
function exempt_coordinates(chain::AbstractArray{Float64,3}, g::Int, burn_frac::Float64;
                            n_distinct_min::Int     = 50,
                            edge_frac_max::Float64  = 0.05,
                            lb::Union{Nothing,AbstractVector} = nothing,
                            ub::Union{Nothing,AbstractVector} = nothing)
    d, N, _ = size(chain)
    b = clamp(floor(Int, burn_frac * g), 0, g - 1)
    kept = g - b
    nd   = zeros(Int, d)
    ef   = zeros(Float64, d)
    kept < 4 && return (falses(d), nd, ef)

    for k in 1:d
        v = vec(@view chain[k, :, b+1:g])
        nd[k] = length(unique(v))
        if lb !== nothing && ub !== nothing
            # Measure pile-up in CONSTRAINED units: the box edge is at infinity in t.
            θ = _to_constrained.(v, lb[k], ub[k])
            w = ub[k] - lb[k]
            ef[k] = w > 0 ? count(x -> (x - lb[k] < 0.01w) || (ub[k] - x < 0.01w), θ) / length(θ) : 0.0
        end
    end
    return (BitVector(nd .< n_distinct_min .|| ef .> edge_frac_max), nd, ef)
end

"""
    converged_sequential(chain, g, burn_frac, p; ...)
      -> (done, worst_rhat, min_ess_seen)

PER-COORDINATE, with automatic exemptions. `min_ess_seen` is the minimum over
NON-EXEMPT coordinates, so the returned number is the one that actually gates.

`ess_min` defaults to 450 rather than `min_ess(p)`. 450 is
`MCSE(q05) <= 0.10 * sd_k` — each reported interval endpoint precise to a tenth of
the width it reports — from `MCSE(q_p) = sqrt(p(1-p))/f(F^-1(p)) * sd/sqrt(ESS)`,
whose constant is 2.113 at p = 0.05. This is the MCSE criterion: it stops when the
ANSWER is precise enough, not when a diagnostic is happy. `min_ess(p)` targets a
joint-volume guarantee over all p coordinates at once, which is a far larger and
different requirement, and is not what the reported per-coordinate intervals need.

Report q05/q95, not q025/q975: the constant is 2.671 at p = 0.025, and required ESS
scales as the SQUARE of the constant, so the tighter tail costs
(2.671/2.113)² − 1 = 59.8% more ESS for the same relative precision — 714 against
447 at MCSE ≤ 0.10·sd. (Equivalently, q05/q95 needs 37.4% LESS; an earlier revision
quoted that 37% figure as the *extra* cost of the tighter tail, which conflates the
two directions and understates it.) With 28 moments the
2.5% tail is the least trustworthy part of the estimate anyway.
"""
function converged_sequential(chain::AbstractArray{Float64,3}, g::Int,
                              burn_frac::Float64, p::Int;
                              rhat_max::Float64 = 1.03,
                              ess_min::Float64  = 450.0,
                              lb::Union{Nothing,AbstractVector} = nothing,
                              ub::Union{Nothing,AbstractVector} = nothing)
    b = clamp(floor(Int, burn_frac * g), 0, g - 1)
    view_g = @view chain[:, :, 1:g]
    rhat, ess = split_rhat_ess(view_g, b)
    exempt, _, _ = exempt_coordinates(chain, g, burn_frac; lb = lb, ub = ub)

    live = .!exempt
    # Every coordinate exempt is not convergence — it means the chain is not sampling
    # anything, which must not read as success.
    any(live) || return (false, NaN, NaN)

    rl = rhat[live]; el = ess[live]
    (all(isfinite, rl) && all(isfinite, el)) || return (false, NaN, NaN)
    wr = maximum(rl)
    me = minimum(el)
    return (wr <= rhat_max && me >= ess_min, wr, me)
end


"""
    accepted_moves(chain, g, burn_frac) -> Int

Number of generations in the RETAINED half in which at least one chain's whole
parameter vector changed, summed over chains. This is the accepted-move count in its
natural unit, and it is the quantity `stop_rule` gates on.

WHY THIS RATHER THAN ESS. Measured on the 18.4.1 base_fc chain (N=64, G=4000): the
per-chain median number of distinct values is 3 for the typical coordinate, so 23 of
23 coordinates sit below `converged_sequential`'s frozen threshold of 50 when counted
per chain. The pooled count clears 50 only because 64 chains × ~3 values ≈ 200. An
autocorrelation estimator applied to a series that is 99.4% duplicates returns a
number (233 to 1400 on that chain) but it is not an effective sample size — there were
~12 accepted moves per chain. Counting transitions directly cannot be fooled that way:
it is the number of times the sampler actually moved.
"""
function accepted_moves(chain::Array{Float64,3}, g::Int, burn_frac::Float64)
    d, N, _ = size(chain)
    b = clamp(floor(Int, burn_frac * g), 0, g - 1)
    g - b < 2 && return 0
    n = 0
    @inbounds for c in 1:N, t in (b + 2):g
        if @views chain[:, c, t] != @views chain[:, c, t - 1]
            n += 1
        end
    end
    return n
end


"""
    stop_rule(chain, g, burn_frac, d; ...) -> (stop, diagnose, moves, drift, wr, me)

The sequential stop. Returns `stop` (terminate, the run has what it came for),
`diagnose` (terminate, the run is not producing a posterior and should be looked at),
and the four quantities the decision is made on.

STOP requires all three, and the caller requires them at TWO CONSECUTIVE checks:
  1. `moves >= moves_min` — accepted moves in the retained half. The deliverable in its
     natural unit.
  2. `drift_since_last < drift_flat` — the running maximum of the log-target has stopped
     climbing. 0.5 log units is ΔQ = 1, one moment moving by one sampling sd, matching
     `PROMOTE_MIN_DQ` and sitting inside the ±3.32 grid resolution on Q's level. Without
     this a rule can stop mid-descent: the criterion is still being minimised, so
     Cov(chain) would measure the trajectory rather than the curvature.
  3. `wr <= rhat_prev` — worst R̂ over non-exempt coordinates has not increased. This
     refuses to stop while the between-chain spread is still deteriorating. It is a
     no-worsening test, NOT a threshold: measured on the 18.4.1 chain, worst R̂ trends
     UPWARD with budget (3.17 at g=250 to 5.73 at g=4000) and never approaches 1.10, so
     a threshold there is unsatisfiable at any budget and the old gate fired at 0 of 16
     checkpoints — decorative in the strict sense that its true-stop rate was also zero.

DIAGNOSE fires when windowed acceptance < `acc_floor` at two consecutive checks. An
acceptance of 0.006 against the Roberts-Gelman-Gilks high-dimensional optimum of 0.234
means the population is mis-scaled relative to the target; more generations cannot fix
it and the run should not silently spend its budget. This is deliberately NOT an abort
on drift — that inference was tried and disabled (see `MCMC_DRIFT_MAX`) because on an
objective with unidentified directions any chain wide enough to measure a width finds
better points, so a drift abort terminates exactly the runs that would produce a
standard error. Low acceptance is a different signal: it says the chain is not moving
at all.

R̂ and ESS are still computed and returned so they can be reported. They are
diagnostics of which coordinates are identified, not gates.
"""
function stop_rule(chain::Array{Float64,3}, g::Int, burn_frac::Float64, d::Int;
                   moves_min::Int          = 2000,
                   drift_flat::Float64     = 0.5,
                   acc_floor::Float64      = 0.02,
                   acc_window::Float64     = NaN,
                   drift_since_last        = NaN,
                   rhat_prev::Float64      = Inf,
                   rhat_max::Float64       = 1.10,
                   ess_min::Float64        = 450.0,
                   lb::Union{Nothing,AbstractVector} = nothing,
                   ub::Union{Nothing,AbstractVector} = nothing)
    _, wr, me = converged_sequential(chain, g, burn_frac, d;
                                     rhat_max = rhat_max, ess_min = ess_min,
                                     lb = lb, ub = ub)
    moves = accepted_moves(chain, g, burn_frac)

    # A NaN on either incoming quantity means "first check, nothing to compare against",
    # so neither branch may fire: a rule that stops on the first check has no evidence.
    drift_ok = isfinite(drift_since_last) && drift_since_last < drift_flat
    rhat_ok  = isfinite(wr) && wr <= rhat_prev
    stop     = moves >= moves_min && drift_ok && rhat_ok

    diagnose = isfinite(acc_window) && acc_window < acc_floor

    return (stop, diagnose, moves, drift_since_last, wr, me)
end


# ─────────────────────────────────────────────────────────────────────────────
# Seed quality
# ─────────────────────────────────────────────────────────────────────────────

"""
    seed_drift(lp_seed, lp_max) -> Float64

How far the chain's running maximum log-target has climbed above the seed, in log
units. A chain started at the mode of its own target fluctuates around the seed and
this stays within the O(d/2) a d-dimensional quadratic gives; a value growing
steadily with the generation count means the chain is still travelling toward the
mode, i.e. optimising rather than sampling.

The distinction matters because while the walk is directional the draws trace a
path rather than a distribution, so their covariance measures the trajectory and
not the curvature. No burn-in fraction repairs this — the run has simply not
arrived — and a longer chain spends the extra budget on the same journey.

A large drift does NOT by itself say the seed is off the optimum of `Q`, because
the target is `−½Q + log|dθ/dt|` and the second term is unbounded below: a
coordinate sitting at `10⁻⁶` of its box width from an edge contributes about −13.8
on its own, and one seeded exactly AT a bound is worse still. Such a chain gains
log-target purely by moving interior, with `Q` unchanged. Use `drift_components`
to attribute the climb before concluding anything about the estimate.
"""
seed_drift(lp_seed::Float64, lp_max::Float64) = lp_max - lp_seed

"""
    drift_components(Q_seed, Q_best, lj_seed, lj_best) -> (dQ, dlj)

Split the rise in log-target into the part from a better `Q` and the part from a
more interior position: `−½(Q_best − Q_seed)` and `lj_best − lj_seed`. Only the
first indicts the point estimate; the second says the seed was near a rail, which
is a bounds question rather than an optimisation failure.
"""
drift_components(Q_seed::Float64, Q_best::Float64,
                 lj_seed::Float64, lj_best::Float64) =
    (-0.5 * (Q_best - Q_seed), lj_best - lj_seed)

"""
    logjac_bound(free) -> Float64

Maximum attainable box log-Jacobian, `Σ_k [log(ub_k−lb_k) − 2 log 2]`, reached at
`t = 0` in every coordinate. Paired with `seed_drift` this separates a legitimate
Jacobian-induced mode shift from a seed that is off the criterion's optimum: only
drift up to `logjac_bound(free) − logjac_box(θ̂, free)` can come from the Jacobian.
"""
logjac_bound(free::Vector{ParamSpec}) =
    sum(log(ps.ub - ps.lb) - 2 * log(2.0) for ps in free)


# ─────────────────────────────────────────────────────────────────────────────
# Chain-free curvature: a local design around the optimum
# ─────────────────────────────────────────────────────────────────────────────

"""
    local_design(θ̂_unc, free; n, rel_step, rng) -> Matrix{Float64}

Design matrix of `n` unconstrained points around `θ̂_unc` for estimating Ĝ without
a Markov chain. Each column perturbs every coordinate by a Gaussian step scaled to
`rel_step` of that coordinate's local `dθ/dt`, so the perturbation is comparable in
CONSTRAINED units across parameters whose boxes differ by orders of magnitude — an
isotropic step in `t` would move `k_U` by 31% of its value and `δ_S` by 0.08%.

The first column is the seed itself, which anchors the regression at the point
whose curvature is wanted.

Randomised rather than a coordinate-wise stencil: with K moments and d parameters
the regression needs only d+1 points, so `n ≈ 10d` is heavily oversampled, and
random directions estimate all d columns of Ĝ jointly instead of spending 2d solves
on one axis at a time. It also avoids the failure mode of a one-at-a-time
difference on this solver, where a step small enough to be local is comparable to
the `tol_global = 1e-4` termination and the difference measures solver noise.
"""
function local_design(θ̂_unc::AbstractVector{Float64}, free::Vector{ParamSpec};
                      n::Int = 10 * length(free), rel_step::Float64 = 0.05,
                      rng::AbstractRNG = MersenneTwister(20260624))
    d = length(free)
    @assert length(θ̂_unc) == d "seed length ≠ number of free parameters"
    X = Matrix{Float64}(undef, d, n)
    @views X[:, 1] .= θ̂_unc
    # Step in t chosen so the induced move in θ is rel_step × the box width:
    # dθ/dt = (ub−lb)·σ(1−σ), so Δt = rel_step / (σ(1−σ)) gives Δθ ≈ rel_step·(ub−lb).
    scale = Vector{Float64}(undef, d)
    @inbounds for k in 1:d
        s = 1.0 / (1.0 + exp(-θ̂_unc[k]))
        scale[k] = rel_step / max(s * (1 - s), 1e-6)
    end
    @inbounds for j in 2:n, k in 1:d
        X[k, j] = θ̂_unc[k] + scale[k] * randn(rng)
    end
    return X
end

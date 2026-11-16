############################################################
# mcmc/mcmc_diagnostics.jl — inference diagnostics for the DE-MC chain
#
# Convergence (split R̂, ESS, accepted moves, MCSE), interiority (boundary mass,
# exemptions), the sequential stop rule, the Jacobian Ĝ (regression on draws or
# adaptive finite differences), the Ω-free standard-error bound, and a local design
# around the optimum. Nothing here solves the model: functions consume the chain
# (d × N × gens), stored moments, or a caller-supplied moment function.
#
# Inference follows Chernozhukov–Hong (2003): with diagonal W the chain quantiles are
# not a confidence interval (Thm 3), so standard errors are Huber sandwiches (Thm 4),
# with Ĵ⁻¹ either n·Cov(chain) or (Ĝ′WĜ)⁻¹.
#
# Plain include() file: definitions only. Needs Statistics, LinearAlgebra, Random.
############################################################

# ─────────────────────────────────────────────────────────────────────────────
# Convergence
# ─────────────────────────────────────────────────────────────────────────────

"""
    split_rhat_ess(chain, burn) -> (rhat, ess)

Rank-normalised split-R̂ and effective sample size per parameter (Vehtari et al. 2021),
on the post-burn-in part of each chain split in two. `chain` is d × N × G; `burn` is the
number of leading generations dropped. Returns two length-d vectors (NaN when fewer than
four generations are kept; R̂ = 1, ESS = 0 for a coordinate that never moves).
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

        # ESS from the combined autocorrelation, truncated at the first negative
        # pair sum (Geyer's initial positive sequence).
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

"""
    _norminvcdf(p) -> Float64

Standard normal quantile Φ⁻¹(p) by Acklam's rational approximation, `p` clamped to
[1e-15, 1 − 1e-15]. Accurate enough for rank-normalisation and a Newton start.
"""
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

"""
    _autocorr_at(x, lag) -> Float64

Autocovariance of `x` at `lag` (divisor n − lag); 0 when `lag ≥ length(x)`.
"""
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

Fraction of draws (d × T, unconstrained) within `frac` of the box width of the lower and
upper bound of each parameter, measured in constrained units. A coordinate piling against
an edge is bound-constrained: its dispersion describes the box, not the data.
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

Ratio of the cross-chain SD of each unconstrained coordinate in the last post-burn-in
decile to that in the first. ≈ 1 for a sampled direction; a ratio that keeps growing
marks a direction diffusing on an improper target. NaN with fewer than 20 kept
generations or a zero initial spread.
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

Ĝ = ∂g/∂θ′ by OLS of each stored model moment (rows of `M`, K × T) on the draws
(`draws`, d × T, unconstrained), in constrained parameter units. Returns the K × d
Jacobian and the per-moment R², which checks the local linear approximation.
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
    jacobian_adaptive_fd(θ̂_unc, free, σ̂, mom; target_sd=1.0, n_rungs=4, seed_frac=0.01,
                         h_lo_frac=1e-7, h_hi_frac=0.25, max_retune=4, n_carried=0)
      -> (G, diag)

∂g/∂θ at `θ̂_unc` (unconstrained) by per-coordinate finite differences with an adaptive
step and Richardson extrapolation, in constrained units (the same units as
`jacobian_from_draws`).

Phase 1 sets each coordinate's step so that the moment displacement measured in sampling
SDs, ‖Δm ./ σ̂‖, is about `target_sd`; the step is predicted from one probe (the
displacement is near-proportional to h), with the box fractions `h_lo_frac`/`h_hi_frac`
only bracketing it. Phase 2 halves the step `n_rungs` times and Richardson-extrapolates
each adjacent pair of the same order, keeping the extrapolate most stable against the
next finer one. Differences are central where both sides solve and one-sided where only
one does; the centre itself need not be feasible.

`mom(θ_unc)` returns the active moments followed by `n_carried` carried outputs (a NaN
vector of the same length when infeasible). It is called concurrently and must be
thread-safe. Carried rows are differentiated at the selected steps without affecting
the selection; `σ̂` covers the moments only.

`diag` holds per coordinate: `h`, `h_frac` (step / box width), `rung`, `ord` (2 central,
1 one-sided), `stab` (per-moment instability, NaN with a single extrapolate), `disp_sd`,
`n_eval`, `ok` (an extrapolate was obtained), `tuned` (phase 1 reached
[target_sd/3, 3·target_sd]); and `centre_ok`, `target_sd`, `n_rungs`.
"""
function jacobian_adaptive_fd(θ̂_unc::Vector{Float64}, free::Vector{ParamSpec},
                              σ̂::Vector{Float64}, mom::Function;
                              target_sd::Float64 = 1.0,
                              n_rungs::Int      = 4,
                              seed_frac::Float64 = 0.01,
                              h_lo_frac::Float64 = 1e-7,
                              h_hi_frac::Float64 = 0.25,
                              max_retune::Int   = 4,
                              n_carried::Int    = 0)
    d = length(free)
    @assert length(θ̂_unc) == d "seed length ≠ number of free parameters"
    @assert n_rungs >= 1 "need at least one halving to extrapolate"
    θ̂ = [_to_constrained(θ̂_unc[k], free[k].lb, free[k].ub) for k in 1:d]
    m0 = mom(θ̂_unc)
    K  = length(m0)
    Km = K - n_carried          # the moments; rows Km+1:K are carried
    Km > 0 || error("mom(θ̂) returned no moments: the active moment count cannot be learned")
    mo = 1:Km
    centre_ok = all(isfinite, m0)

    G    = fill(NaN, K, d)      # ∂g/∂θ, natural units
    STAB = fill(NaN, K, d)      # per-moment instability of the accepted extrapolate
    hsel  = fill(NaN, d)        # accepted step, θ units
    disp  = fill(NaN, d)        # achieved ‖Δm ./ σ̂‖ at the phase-1 step
    rung  = zeros(Int, d)
    nev   = zeros(Int, d)
    ord   = zeros(Int, d)       # leading-error order of the accepted rung: 2 central, 1 one-sided
    ok    = falses(d)           # a Richardson extrapolate was obtained
    tuned = falses(d)           # …and phase 1 reached the target displacement band

    Threads.@threads for j in 1:d
        lb, ub = free[j].lb, free[j].ub
        w      = ub - lb
        # Stay inside the box and clear of the edge, so _to_unconstrained's 1e-8 clamp
        # cannot truncate the perturbation.
        edge = 0.999 * min(θ̂[j] - lb, ub - θ̂[j])
        h_lo = h_lo_frac * w
        h_hi = min(h_hi_frac * w, edge)

        # Difference quotient at step h → (quotient, raw displacement, error order):
        # central (order 2) when both sides solve, one-sided (order 1) when only one does,
        # order 0 when neither. The convergent domain is not an interval around θ̂, so one
        # usable side is kept. Divides by the θ span realised after the θ→t→θ round trip.
        function dquot(h::Float64)
            tp = copy(θ̂_unc); tp[j] = _to_unconstrained(θ̂[j] + h, lb, ub)
            tm = copy(θ̂_unc); tm[j] = _to_unconstrained(θ̂[j] - h, lb, ub)
            θp = _to_constrained(tp[j], lb, ub)
            θm = _to_constrained(tm[j], lb, ub)
            mp, mm = mom(tp), mom(tm)
            nev[j] += 2
            okp = all(isfinite, mp) && θp > θ̂[j]
            okm = all(isfinite, mm) && θm < θ̂[j]
            okp && okm && return ((mp .- mm) ./ (θp - θm), mp .- mm, 2)
            okp && centre_ok && return ((mp .- m0) ./ (θp - θ̂[j]), mp .- m0, 1)
            okm && centre_ok && return ((m0 .- mm) ./ (θ̂[j] - θm), m0 .- mm, 1)
            return (fill(NaN, K), fill(NaN, K), 0)
        end

        h_hi <= h_lo && continue
        h_max   = h_hi                  # largest usable step; only OUTWARD failures lower it
        h       = clamp(seed_frac * w, h_lo, h_hi)
        D, Δ, p = dquot(h)
        while p == 0 && h > h_lo        # a feasible step first, at any size
            h_max = min(h_max, h)       # h failed, so nothing at or above it is usable either
            h = max(0.5h, h_lo)
            D, Δ, p = dquot(h)
        end
        p == 0 && continue              # no feasible step anywhere: column stays NaN

        # Retune toward the target displacement. A failure while growing is an outward
        # barrier (lower h_max, re-predict); a failure while shrinking means a hole lies
        # between θ̂ and the step, so keep the step that works.
        for _ in 1:max_retune
            sd_now = norm(Δ[mo] ./ σ̂)
            (target_sd / 3 <= sd_now <= target_sd * 3) && break
            h_new = clamp(h * target_sd / max(sd_now, 1e-300), h_lo, h_max)
            h_new == h && break                     # the bracket binds; take what it allows
            D2, Δ2, p2 = dquot(h_new)
            if p2 > 0
                h, D, Δ, p = h_new, D2, Δ2, p2
            elseif h_new > h
                h_max = h_new                       # outward barrier: cap and re-predict
            else
                break                               # inward hole: this step is the best there is
            end
        end
        disp[j]  = norm(Δ[mo] ./ σ̂)
        tuned[j] = target_sd / 3 <= disp[j] <= target_sd * 3

        # Ladder h, h/2, …, then (2^p·D(h/2) − D(h))/(2^p − 1) for each adjacent pair of the
        # same order p. A pair straddling a central/one-sided switch is skipped.
        Dl  = Vector{Vector{Float64}}(undef, n_rungs + 1)
        pl  = zeros(Int, n_rungs + 1)
        top = 1
        Dl[1], pl[1] = D, p
        for r in 2:(n_rungs + 1)
            hr = h / 2.0^(r - 1)
            hr < h_lo && break
            Dr, Δr, pr = dquot(hr)
            pr == 0 && break
            Dl[r], pl[r] = Dr, pr
            top = r
        end
        top >= 2 || continue                        # no pair, no extrapolate
        R = Tuple{Int,Vector{Float64}}[]
        for r in 1:(top - 1)
            pl[r] == pl[r + 1] || continue
            f = 2.0^pl[r]
            push!(R, (r, (f .* Dl[r + 1] .- Dl[r]) ./ (f - 1.0)))
        end
        isempty(R) && continue

        # Keep the extrapolate closest to the next finer one. Instability is per moment,
        # relative to the column's typical σ̂-scaled element, so it is scale-free.
        pick = 1
        if length(R) > 1
            gaps = [norm((R[i][2][mo] .- R[i + 1][2][mo]) ./ σ̂) for i in 1:(length(R) - 1)]
            pick = argmin(gaps)
            ref  = max(norm(R[pick][2][mo] ./ σ̂) / sqrt(Km), 1e-300)
            @views STAB[mo, j] .= abs.(R[pick][2][mo] .- R[pick + 1][2][mo]) ./ σ̂ ./ ref
        end
        @views G[:, j] .= R[pick][2]
        hsel[j] = h / 2.0^(R[pick][1] - 1)
        rung[j] = R[pick][1]
        ord[j]  = pl[R[pick][1]]
        ok[j]   = true
    end

    return (G, (h = hsel, h_frac = hsel ./ [free[j].ub - free[j].lb for j in 1:d],
                rung = rung, ord = ord, stab = STAB, disp_sd = disp, n_eval = nev, ok = ok,
                tuned = tuned, centre_ok = centre_ok, target_sd = target_sd,
                n_rungs = n_rungs))
end

"""
    se_bound_diagonal(G, W, σ̂, C = I; bread = nothing) -> (se_bound, se_sandwich)

Standard errors of `C′θ̂` for the extremum estimator using only the diagonal of Ω
(sampling SDs `σ̂`). With bread `B` (inverse curvature of Q/2) and `a_j = W G B c_j`, the
sandwich variance `a_j′Ω a_j` is bounded by `(Σ_i |a_ij| σ̂_i)²` for every Ω with that
diagonal; the bound is sharp (attained by a rank-one Ω). `se_sandwich` is the sandwich at
Ω = W⁻¹, `sqrt(diag(C′BJBC))` with `J = G′WG`.

`bread = nothing` uses `B = pinv(J)` (the sandwich is then `sqrt(diag(C′J⁻¹C))`); on the
chain path the bread is the covariance of the kept draws. `C = I` gives the parameters'
own errors; a gradient matrix gives delta-method errors.
"""
function se_bound_diagonal(G::Matrix{Float64}, W::Matrix{Float64}, σ̂::Vector{Float64},
                           C::AbstractMatrix{Float64} = Matrix{Float64}(I, size(G, 2), size(G, 2));
                           bread::Union{Nothing,AbstractMatrix{Float64}} = nothing)
    J  = G' * W * G
    B  = bread === nothing ? pinv(J) : bread
    A  = W * G * B * C                               # column j = a_j
    m  = size(C, 2)
    se_bound = Vector{Float64}(undef, m)
    @inbounds for j in 1:m
        se_bound[j] = sum(abs.(@view A[:, j]) .* σ̂)
    end
    V  = bread === nothing ? C' * B * C : C' * B * J * B * C
    return (se_bound, sqrt.(abs.(diag(V))))
end

"""
    curvature_check(draws, G, W, free) -> Float64

Median |log₁₀| ratio between the diagonals of two estimates of J: the inverse covariance
of the draws (constrained units) and Ĝ′WĜ. A large value indicts the chain (short
burn-in, improper target) rather than the Jacobian. NaN when no diagonal pair is usable.
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

Vats–Flegal–Jones (2019) minimum effective sample size for a `p`-dimensional target: a
(1−`alpha`) confidence region within relative precision `eps` of the posterior volume,

    minESS = 2^{2/p} π / (p Γ(p/2))^{2/p} · χ²_{1−α,p} / ε².

ε = 0.20 gives ≈ 540 at p = 25, nearly flat in p; ε = 0.10 (≈ 2159) is out of reach for
this sampler's budget.
"""
function min_ess(p::Int; alpha::Float64 = 0.05, eps::Float64 = 0.20)
    logden = (2.0 / p) * (log(p) + _lgamma(p / 2.0))
    pref   = exp((2.0 / p) * log(2.0)) * pi / exp(logden)
    return pref * _chisq_quantile(1 - alpha, p) / eps^2
end

"""
    _lgamma(x) -> Float64

log Γ(x) by the Lanczos approximation (g = 7, n = 9; relative error < 1e-13 for x > 0),
with the reflection formula below 0.5.
"""
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

"""
    _gamma_p(a, x) -> Float64

Regularized lower incomplete gamma P(a, x): series for x < a + 1, continued fraction
otherwise (Numerical Recipes §6.2).
"""
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

"""
    _chisq_quantile(q, ν) -> Float64

χ² quantile of order `q` with `ν` degrees of freedom: Wilson–Hilferty start, then Newton
on P(ν/2, x/2) − q with the χ² density as derivative.
"""
function _chisq_quantile(q::Float64, ν::Int)
    a = ν / 2.0
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

Coordinates that cannot reach a mixing threshold at any budget, measured on the
post-burn-in pooled draws of generations 1:`g`:
  FROZEN      fewer than `n_distinct_min` distinct visited values;
  AT A BOUND  more than `edge_frac_max` of draws within 1% of the box width of an edge,
              in constrained units (needs `lb`, `ub`; skipped without them).
Exemption only stops a coordinate from blocking termination: it is recomputed at every
check, and nothing is frozen or dropped from the outputs.
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
            θ = _to_constrained.(v, lb[k], ub[k])
            w = ub[k] - lb[k]
            ef[k] = w > 0 ? count(x -> (x - lb[k] < 0.01w) || (ub[k] - x < 0.01w), θ) / length(θ) : 0.0
        end
    end
    return (BitVector(nd .< n_distinct_min .|| ef .> edge_frac_max), nd, ef)
end

"""
    converged_sequential(chain, g, burn_frac, p; rhat_max=1.03, ess_min=450.0, lb, ub)
      -> (done, worst_rhat, min_ess_seen)

Worst split-R̂ and smallest ESS over non-exempt coordinates (`exempt_coordinates`) on
generations 1:`g`, and whether they meet `rhat_max` and `ess_min`. ESS 450 is
MCSE(q05) ≤ 0.10·sd. Returns `(false, NaN, NaN)` when every coordinate is exempt or a
statistic is not finite.
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
    # Every coordinate exempt means nothing is being sampled: not convergence.
    any(live) || return (false, NaN, NaN)

    rl = rhat[live]; el = ess[live]
    (all(isfinite, rl) && all(isfinite, el)) || return (false, NaN, NaN)
    wr = maximum(rl)
    me = minimum(el)
    return (wr <= rhat_max && me >= ess_min, wr, me)
end


"""
    accepted_moves(chain, g, burn_frac) -> Int

Number of (chain, generation) pairs in the retained part of generations 1:`g` at which the
chain's parameter vector changed. Counts transitions directly, which an autocorrelation
ESS cannot do on a series that is mostly repeats.
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
    mcse_ratio(chain, g, burn_frac; lb, ub) -> (worst, n_live)

Monte Carlo error of the reported (pooled) mean as a fraction of the reported pooled SD,
maximised over non-exempt coordinates (Flegal–Haran–Jones 2008). The chains are
independent, so MCSE = sd(chain means)/√N. Valid only for an over-dispersed start: from
a common seed the chain means agree and the ratio is ≈ 0 by construction.
"""
function mcse_ratio(chain::AbstractArray{Float64,3}, g::Int, burn_frac::Float64;
                    lb::Union{Nothing,AbstractVector} = nothing,
                    ub::Union{Nothing,AbstractVector} = nothing)
    d, N, _ = size(chain)
    b    = clamp(floor(Int, burn_frac * g), 0, g - 1)
    kept = g - b
    (kept < 4 || N < 2) && return (NaN, 0)
    exempt, _, _ = exempt_coordinates(chain, g, burn_frac; lb = lb, ub = ub)
    worst = -Inf; n_live = 0
    cm = Vector{Float64}(undef, N)
    @inbounds for k in 1:d
        exempt[k] && continue
        for c in 1:N
            s = 0.0
            for tt in (b + 1):g
                s += chain[k, c, tt]
            end
            cm[c] = s / kept
        end
        sp = std(vec(@view chain[k, :, (b + 1):g]))
        sp > 0.0 || continue
        n_live += 1
        worst = max(worst, (std(cm) / sqrt(N)) / sp)
    end
    return (n_live == 0 ? NaN : worst, n_live)
end


"""
    stop_rule(chain, g, burn_frac, d; moves_min=2000, drift_flat=0.5, acc_floor=0.02,
              mcse_target=0.05, acc_window=NaN, drift_since_last=NaN, rhat_max=1.10,
              ess_min=450.0, lb, ub)
      -> (stop, diagnose, moves, drift, wr, me, mcse)

The sequential stop, evaluated at one check (the caller requires two consecutive).
`stop` requires all of:
  1. `moves ≥ moves_min` accepted moves in the retained part (`accepted_moves`);
  2. `drift_since_last < drift_flat`: the running maximum of the log-target has stopped
     climbing (NaN on the first check, which therefore cannot stop);
  3. `mcse < mcse_target` (`mcse_ratio`), the accuracy gate.
`diagnose` is windowed acceptance `acc_window < acc_floor`: the population is mis-scaled
and more generations cannot help. `wr` and `me` (worst R̂, smallest ESS) are reported only.
"""
function stop_rule(chain::Array{Float64,3}, g::Int, burn_frac::Float64, d::Int;
                   moves_min::Int          = 2000,
                   drift_flat::Float64     = 0.5,
                   acc_floor::Float64      = 0.02,
                   mcse_target::Float64    = 0.05,
                   acc_window::Float64     = NaN,
                   drift_since_last        = NaN,
                   rhat_max::Float64       = 1.10,
                   ess_min::Float64        = 450.0,
                   lb::Union{Nothing,AbstractVector} = nothing,
                   ub::Union{Nothing,AbstractVector} = nothing)
    _, wr, me = converged_sequential(chain, g, burn_frac, d;
                                     rhat_max = rhat_max, ess_min = ess_min,
                                     lb = lb, ub = ub)
    moves = accepted_moves(chain, g, burn_frac)
    mcse, _ = mcse_ratio(chain, g, burn_frac; lb = lb, ub = ub)

    drift_ok = isfinite(drift_since_last) && drift_since_last < drift_flat
    mcse_ok  = isfinite(mcse) && mcse < mcse_target
    stop     = moves >= moves_min && drift_ok && mcse_ok

    diagnose = isfinite(acc_window) && acc_window < acc_floor

    return (stop, diagnose, moves, drift_since_last, wr, me, mcse)
end


# ─────────────────────────────────────────────────────────────────────────────
# Seed quality
# ─────────────────────────────────────────────────────────────────────────────

"""
    drift_components(Q_seed, Q_best, lj_seed, lj_best) -> (dQ, dlj)

Split the rise in log-target from the seed to the best visited point into the part from
a better criterion, `−½(Q_best − Q_seed)`, and the part from a more interior position,
`lj_best − lj_seed` (box log-Jacobians). Only the first indicts the point estimate; the
second says the seed sat near a bound.
"""
drift_components(Q_seed::Float64, Q_best::Float64,
                 lj_seed::Float64, lj_best::Float64) =
    (-0.5 * (Q_best - Q_seed), lj_best - lj_seed)


# ─────────────────────────────────────────────────────────────────────────────
# Chain-free curvature: a local design around the optimum
# ─────────────────────────────────────────────────────────────────────────────

"""
    local_design(θ̂_unc, free; n=10d, rel_step=0.05, rng=MersenneTwister(20260624))
      -> Matrix{Float64}

`d × n` design of unconstrained points around `θ̂_unc` for regressing Ĝ without a chain.
Column 1 is the seed. Each other column perturbs every coordinate by a Gaussian step
scaled so the move in θ is about `rel_step` of the box width; a coordinate within
`rel_step` of a bound is stepped one-sided, inward, in θ. One normal draw is taken per
(column, coordinate) in a fixed order, whichever branch uses it.
"""
function local_design(θ̂_unc::AbstractVector{Float64}, free::Vector{ParamSpec};
                      n::Int = 10 * length(free), rel_step::Float64 = 0.05,
                      rng::AbstractRNG = MersenneTwister(20260624))
    d = length(free)
    @assert length(θ̂_unc) == d "seed length ≠ number of free parameters"
    X = Matrix{Float64}(undef, d, n)
    @views X[:, 1] .= θ̂_unc
    # dθ/dt = (ub−lb)·σ(1−σ), so Δt = rel_step / (σ(1−σ)) gives Δθ ≈ rel_step·(ub−lb).
    scale = Vector{Float64}(undef, d)
    # +1 / −1: at the lower / upper bound, stepped inward in θ;  0: interior, stepped in t.
    edge  = zeros(Int, d)
    θ̂c    = [_to_constrained(θ̂_unc[k], free[k].lb, free[k].ub) for k in 1:d]
    @inbounds for k in 1:d
        s = 1.0 / (1.0 + exp(-θ̂_unc[k]))
        scale[k] = rel_step / max(s * (1 - s), 1e-6)
        w = free[k].ub - free[k].lb
        θ̂c[k] - free[k].lb < rel_step * w && (edge[k] = 1)
        free[k].ub - θ̂c[k] < rel_step * w && (edge[k] = -1)
    end
    @inbounds for j in 2:n, k in 1:d
        z = randn(rng)
        X[k, j] = edge[k] == 0 ? θ̂_unc[k] + scale[k] * z :
                  _to_unconstrained(θ̂c[k] + edge[k] * abs(z) * rel_step * (free[k].ub - free[k].lb),
                                    free[k].lb, free[k].ub)
    end
    return X
end

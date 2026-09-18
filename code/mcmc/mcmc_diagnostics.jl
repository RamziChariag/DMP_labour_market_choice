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
    jacobian_adaptive_fd(θ̂_unc, free, σ̂, mom; target_sd, n_rungs, ...) -> (G, diag)

`∂g/∂θ` by per-coordinate central differences with an adaptively chosen step, refined by
Richardson extrapolation. `G` comes back in NATURAL (constrained) units exactly as
`jacobian_from_draws` returns it, so either route feeds `se_bound_diagonal` unchanged.

WHY NOT A COMMON FRACTION OF THE BOX. `local_design` steps by `rel_step·(ub−lb)`, so the
Jacobian moves when a search bound moves — a bound the estimate never approaches still sets
the derivative's resolution. And one fraction cannot serve coordinates whose moment
sensitivities span orders of magnitude. Measured on `base_covid` at the shipped 0.05, the
per-moment R² of the linear fit had median 0.771 with 25 of 35 moments below 0.9; at 0.01 the
median was 0.992 with none below 0.9. The residual was curvature, not solver noise, which is
what makes both halves of this routine worth the evaluations.

PHASE 1, THE STEP FROM THE GEOMETRY. Each coordinate's step is set so the perturbation moves
the moment vector a target distance in the moments' OWN sampling-error units,
`‖Δm ./ σ̂‖ ≈ target_sd`. A coordinate the moments barely respond to therefore gets a LARGE
step and one they respond to sharply a small one — the sensitivity ordering the box cannot
know. The box fraction enters only as a seed; the accepted step is whatever the displacement
condition selects, and `h_lo_frac`/`h_hi_frac` only bracket it.

The target is reached by PREDICTION, not by search: `‖Δm ./ σ̂‖` is very nearly proportional to
`h` near θ̂, so one feasible probe at `h` giving displacement `s` fixes the step directly as
`h·target_sd/s`. Doubling or halving toward the target instead spends the evaluation budget
re-deriving that proportionality for every coordinate, and on this model it exhausted it —
three of 23 coordinates ended at the probe cap with displacements of 0.01, 7.5 and 7.5 against
a target of 1. An infeasible probe is information too: it bounds the step from above.

PHASE 2, CURVATURE REMOVED. A central difference carries `D(h) = g′ + c₂h² + c₄h⁴ + …`, so
`(4·D(h/2) − D(h))/3` cancels the `h²` term and leaves `O(h⁴)`. The ladder `h, h/2, h/4, …`
yields one extrapolate per adjacent pair, and the accepted rung is the one whose estimate is
most stable against the next finer one. That stability is the finite-difference counterpart of
the regression route's R² and replaces it in the diagnostics: there is no fit here, so there is
no R² to report.

The ladder also rescues a phase-1 step that came out too LARGE — it only ever refines downward,
so a coordinate whose displacement overshot the band still reaches a sensible radius a rung or
two in. It cannot rescue one that came out too small, which happens only when the convergent
domain bounds the step from above; that is a fact about the coordinate, not a tuning failure,
and `tuned` marks it. When only one extrapolate is available its instability is left `NaN`
rather than filled: a single estimate has nothing to be stable against.

A CENTRAL DIFFERENCE NEEDS NO CENTRE. Only `θ̂ ± h` is evaluated, so an infeasible `θ̂` does not
stop the measurement and is not treated as an error — `diag.centre_ok` records it and the caller
reports it. This is not a corner case here: `base_fc`'s stored `θ̂` returns `Inf` above one thread
because `converged_S` depends on the reduction order in `skilled.jl`, and the regression route
proceeded silently on that window because its cloud points sit off `θ̂`. A derivative centred on a
point the solver rejects is worth knowing about, not worth crashing over.

`mom(θ_unc)::Vector{Float64}` returns the K active moments at that point, or a NaN vector when
the solve is infeasible — its LENGTH must be K either way, since that is how K is learned. Note
that `smm_objective` writes its `moments_out` buffer only after its guards, so a caller must key
on the returned objective and not on the buffer's contents: an `undef` buffer left unwritten
holds finite garbage. It is called concurrently across coordinates, so it must be thread-safe —
allocate a buffer per call rather than writing into a shared one.
"""
function jacobian_adaptive_fd(θ̂_unc::Vector{Float64}, free::Vector{ParamSpec},
                              σ̂::Vector{Float64}, mom::Function;
                              target_sd::Float64 = 1.0,
                              n_rungs::Int      = 4,
                              seed_frac::Float64 = 0.01,
                              h_lo_frac::Float64 = 1e-7,
                              h_hi_frac::Float64 = 0.25,
                              max_retune::Int   = 4)
    d = length(free)
    @assert length(θ̂_unc) == d "seed length ≠ number of free parameters"
    @assert n_rungs >= 1 "need at least one halving to extrapolate"
    θ̂ = [_to_constrained(θ̂_unc[k], free[k].lb, free[k].ub) for k in 1:d]
    m0 = mom(θ̂_unc)
    K  = length(m0)
    K > 0 || error("mom(θ̂) returned an empty vector: the active moment count cannot be learned")
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
        # The step may not leave the box, and must keep clear of the edge so
        # _to_unconstrained's 1e-8 clamp cannot silently truncate the perturbation.
        edge = 0.999 * min(θ̂[j] - lb, ub - θ̂[j])
        h_lo = h_lo_frac * w
        h_hi = min(h_hi_frac * w, edge)

        # Difference quotient at step h: central when both sides solve, one-sided against θ̂
        # when only one does. The convergent domain is not an interval around θ̂ — on base_fc
        # the solve succeeds AT θ̂ and at radii ≥1e-3 of box width but fails at 1e-6…1e-4 above
        # it, while every radius below it succeeds — so requiring both sides throws away a
        # coordinate one side can measure perfectly well. Returns the quotient, the raw moment
        # displacement, and the ORDER of the leading error term, which sets the Richardson
        # weights: 2 for a central difference, 1 for a one-sided one.
        # Divided by the span the solver ACTUALLY sees after the θ→t→θ round trip rather than
        # by the requested step, so the quotient stays exact at any h.
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

        # Retune toward the target displacement. A failure while GROWING is an outward barrier
        # and lowers h_max; a failure while SHRINKING means a hole lies between θ̂ and the
        # current step, and shrinking further only goes deeper into it — so stop and keep the
        # step that works. Treating the two alike is what pinned every base_fc coordinate at
        # the upper rail with a displacement of 26 against a target of 1.
        for _ in 1:max_retune
            sd_now = norm(Δ ./ σ̂)
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
        disp[j]  = norm(Δ ./ σ̂)
        tuned[j] = target_sd / 3 <= disp[j] <= target_sd * 3

        # Ladder, then one Richardson extrapolate per adjacent pair OF THE SAME ORDER:
        # (2^p·D(h/2) − D(h))/(2^p − 1) cancels the leading h^p term. A pair straddling a
        # switch from central to one-sided is skipped rather than extrapolated with the wrong
        # weights, which would introduce an error larger than the term it removes.
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

        # Accept the extrapolate that agrees best with the next finer one. Instability is
        # reported per moment relative to the column's own typical σ̂-scaled element, so it is
        # scale-free and does not blow up on a moment this coordinate happens not to move.
        pick = 1
        if length(R) > 1
            gaps = [norm((R[i][2] .- R[i + 1][2]) ./ σ̂) for i in 1:(length(R) - 1)]
            pick = argmin(gaps)
            ref  = max(norm(R[pick][2] ./ σ̂) / sqrt(K), 1e-300)
            @views STAB[:, j] .= abs.(R[pick][2] .- R[pick + 1][2]) ./ σ̂ ./ ref
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
    mcse_ratio(chain, g, burn_frac; lb, ub) -> (worst, n_live)

Monte Carlo error of the REPORTED MEAN, as a fraction of the REPORTED STANDARD DEVIATION,
maximised over non-exempt coordinates.

WHY THIS IS THE STOPPING QUANTITY. The estimator is the mean of the pooled retained draws
and the standard error is their standard deviation (LMR Appendix C). So "the numbers are good
enough" is a statement about how much of the reported spread is sampling noise in the mean,
and nothing else. This is the Flegal-Haran-Jones (2008) criterion; the conventional target is
0.05, i.e. the third significant figure of the reported number is stable.

WHY IT REPLACES THE PREVIOUS GATE. The rule this supersedes required accepted moves, flat
drift, and worst R-hat merely NOT INCREASING. All three can hold while the chains sit in
different regions: the pooled SD then measures the between-chain spread rather than the
posterior width, and the run stops with a number that is not a standard error. Those gates
detect a chain that has stopped deteriorating -- a salvage test for a run that never
converges -- not one that has arrived.

WHY NO AUTOCORRELATION ESTIMATOR. The chains are independent, so the N per-chain means are
independent draws and sd(chain means)/sqrt(N) is the standard error of the grand mean
directly. Within-chain autocorrelation is already inside each chain mean's variance. That
matters here because a series which is 99.4% duplicates defeats a spectral ESS (see
`accepted_moves`) while leaving this estimator correct.

RELATION TO R-HAT. With independent chains MCSE/se ~ sqrt(B/W)/sqrt(N) and R-hat^2 ~ 1 + B/W,
so at N = 95 the 0.05 target sits at R-hat ~ 1.11. Read the other way: the v19.8.0 run
reached R-hat = 8.50, i.e. MCSE/se ~ 0.87 -- 87% of the reported number was Monte Carlo noise.
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
    stop_rule(chain, g, burn_frac, d; ...) -> (stop, diagnose, moves, drift, wr, me, mcse)

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
  3. `mcse < mcse_target` — Monte Carlo error of the reported mean as a fraction of the
     reported sd, worst over non-exempt coordinates (`mcse_ratio`). THIS IS THE ACCURACY
     GATE and the only one of the three that is a statement about the deliverable: 0.05
     means the third significant figure of the reported number is stable.

     It replaced a no-worsening test on worst R̂ (`wr <= rhat_prev`) in v21.0.0. That test
     refused to stop while the between-chain spread was still deteriorating, which sounds
     conservative but is not a convergence criterion: a chain can satisfy it, together with
     conditions 1 and 2, while its members sit in different regions — and then the pooled
     sd measures the between-chain spread rather than the posterior width, so the run stops
     with a number that is not a standard error. It detects a chain that has stopped getting
     worse, not one that has arrived; in the working regime it is redundant, and in the
     collapse regime it is the only thing that fires.

     VALIDITY CONDITION, and it binds. Like R̂, this diagnostic requires an OVER-DISPERSED
     start. At `MCMC_INIT = :at_seed` every chain begins at the same point, so the per-chain
     means agree, `mcse ~ 0`, and the gate passes on a chain that has gone nowhere. Conditions
     1 and 2 happen to block that, but by side effect rather than by design. Use
     `MCMC_INIT = :widths`, which starts the population at the criterion's own ΔQ = 1 widths.

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

    # A NaN on drift means "first check, nothing to compare against", so the rule may not
    # fire: a rule that stops on the first check has no evidence.
    drift_ok = isfinite(drift_since_last) && drift_since_last < drift_flat
    # The accuracy gate, and the reason this rule was rewritten. `rhat_prev` is kept in the
    # signature for callers but is NO LONGER GATED ON: "worst R-hat stopped getting worse"
    # is satisfiable by a chain that never converges, so it fired in the collapse regime and
    # was redundant in the working one.
    mcse_ok  = isfinite(mcse) && mcse < mcse_target
    stop     = moves >= moves_min && drift_ok && mcse_ok

    diagnose = isfinite(acc_window) && acc_window < acc_floor

    return (stop, diagnose, moves, drift_since_last, wr, me, mcse)
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

############################################################
# grids.jl — Grid construction, matching technology, the skill
#            copula, and shared numerical utilities (RoySearch)
############################################################

# ============================================================
# Matching technology (Cobb–Douglas)
# ============================================================

@inline q_from_theta(θ::Float64, μ::Float64, η::Float64)    = μ * θ^(-η)
@inline theta_from_q(q::Float64, μ::Float64, η::Float64)    = (μ / q)^(1/η)
@inline jobfinding_rate(θ::Float64, μ::Float64, η::Float64) = θ * q_from_theta(θ, μ, η)


# ============================================================
# Norms and clamp
# ============================================================

@inline supnorm(a, b)       = maximum(abs.(a .- b))
@inline clamp01(x::Float64) = min(max(x, 0.0), 1.0)


# ============================================================
# Gauss–Legendre quadrature on [0, 1]
# ============================================================

"""
    build_gl_grid(N) -> (nodes, weights)

Return `N`-point Gauss–Legendre nodes and weights on [0, 1].
"""
function build_gl_grid(N::Int)
    nodes, weights = gausslegendre(N)
    x = @. 0.5 * (nodes + 1.0)
    w = @. 0.5 * weights
    return x, w
end


# ============================================================
# Ability marginal as a Gauss–Jacobi mass rule
#
# Each ability marginal is the Beta(a_ℓ, b_ℓ) density.  Every marginal
# aggregate in the model is an expectation E_ℓ[g] = ∫ g(a) f_ℓ(a) da with
# g a smooth conditional-occupancy profile.  Gauss–Jacobi integrates
# exactly against the kernel a^{a_ℓ−1}(1−a)^{b_ℓ−1}, so it handles the
# a_ℓ<1 / b_ℓ<1 endpoint singularities that defeat Gauss–Legendre.
#
# `gaussjacobi(N, α, β)` gives nodes/weights for
#   ∫_{−1}^{1} h(t) (1−t)^α (1+t)^β dt.
# Map a = (t+1)/2 and match a^{a_ℓ−1}(1−a)^{b_ℓ−1}: α = b_ℓ−1 (pairs with
# 1−t), β = a_ℓ−1 (pairs with 1+t).  Then for any smooth g,
#   E_ℓ[g] = ∫ g f_ℓ da = (C/B) Σ_i w_i g(a_i),  C = 2^{−(a_ℓ+b_ℓ−1)}.
# Since E_ℓ[1] = 1 forces (C/B) Σ_i w_i = 1, the population mass weights
# are simply the normalised Gauss–Jacobi weights
#   wa_i = w_i / Σ_j w_j,        Σ_i wa_i = 1  (exactly),
# with no Beta normaliser and no division by the kernel (which would
# reintroduce the endpoint singularity).  Every marginal expectation is
# then `dot(g, wa)`, exact to quadrature order for smooth g.
#
# Under the symmetric-marginals baseline both abilities share this grid
# and these weights; the two-dimensional joint is assembled from them and
# the copula density in build_copula (below).
# ============================================================

"""
    build_ability_grid(N, a, b) -> (nodes, weights)

`N`-point Gauss–Jacobi mass rule on [0, 1] for a Beta(a, b) ability
marginal.  Returns nodes and population mass weights `wa` with
`sum(wa) = 1` and `dot(g, wa) ≈ E_ℓ[g]` for any smooth `g`.  Valid for
any `a, b > 0`, including the singular `a<1` / `b<1` regime.
"""
function build_ability_grid(N::Int, a::Float64, b::Float64)
    (a > 0.0 && b > 0.0) ||
        error("build_ability_grid: Beta shape parameters must be positive (got a=$a, b=$b).")
    t, w = gaussjacobi(N, b - 1.0, a - 1.0)        # weight (1−t)^{b−1} (1+t)^{a−1}
    x    = @. 0.5 * (t + 1.0)
    wa   = w ./ sum(w)                              # population mass weights, Σ = 1
    return x, wa
end


# ============================================================
# Skill copula
#
# The joint ability density is (notes eq. copula)
#   ℓ(aU,aS) = f_ℓ(aU) f_ℓ(aS) c_ρ(F_ℓ(aU), F_ℓ(aS)),
# a Gaussian copula with a single correlation ρ_x on the Gaussian ranks
#   ζ_u = Φ⁻¹(F_ℓ(aU)),  ζ_v = Φ⁻¹(F_ℓ(aS)),
#   c_ρ(u,v) = exp(−(ρ²(ζ_u²+ζ_v²) − 2ρ ζ_u ζ_v)/(2(1−ρ²))) / √(1−ρ²).
#
# Because wa^U, wa^S are marginal population masses (Σ = 1 each), the
# two-dimensional aggregation weight is the outer product of the two
# marginal weight vectors, modulated pointwise by the copula density:
#   W2[i,j] = c_ρ(F_i, F_j) · wa^U_i · wa^S_j.
# Then for any surface g(aU,aS),  E_ℓ[g] = Σ_{ij} g_{ij} W2_{ij}.
#
# Copula margins are uniform, so ∫ c_ρ(u,v) dv = 1: the aU-marginal of
# W2 (row sums) returns wa^U exactly, and total mass is 1.  build_copula
# renormalises row/column/total on the finite grid so these identities
# hold to machine precision at every ρ_x (mandatory for the ρ_x → 1
# nesting check against the single-ability model).
#
# ρ_x → 1 concentrates c_ρ on the diagonal aU = aS; W2 becomes diagonal
# with entries wa_i and RoySearch reduces to the one-dimensional model.
# ============================================================

"""
    CopulaGrid

Two-dimensional aggregation structure for the skill copula.  `W2[i,j]`
is the joint population weight at ability pair `(a[i], a[j])` such that
`sum(g .* W2) ≈ ∫∫ g(aU,aS) ℓ(aU,aS) daU daS` for any surface `g`.
`ρ_x` is the copula correlation; `F` holds the marginal CDF values at
the ability nodes.
"""
struct CopulaGrid
    ρ_x :: Float64
    F   :: Vector{Float64}    # marginal CDF F_ℓ at the ability nodes
    ζ   :: Vector{Float64}    # Gaussian ranks Φ⁻¹(F) at the ability nodes
    W2  :: Matrix{Float64}    # joint population weights (rows aU, cols aS)
end

"""
    build_copula(ρ_x, xgrid, wa_U, wa_S, a_ℓ, b_ℓ) -> CopulaGrid

Assemble the joint aggregation weights for correlation `ρ_x` on the
shared ability grid `xgrid` with marginal population weights `wa_U`,
`wa_S`.  Row sums return `wa_U`, column sums return `wa_S`, and the
total is 1, all to machine precision after renormalisation.
"""
function build_copula(ρ_x::Float64, xgrid::Vector{Float64},
                      wa_U::Vector{Float64}, wa_S::Vector{Float64},
                      a_ℓ::Float64, b_ℓ::Float64)
    N    = length(xgrid)
    dist = Beta(a_ℓ, b_ℓ)
    F    = clamp.(cdf.(dist, xgrid), 1e-12, 1.0 - 1e-12)
    ζ    = quantile.(Normal(), F)

    ρ  = clamp(ρ_x, -0.999, 0.999)
    W2 = Matrix{Float64}(undef, N, N)
    if abs(ρ) < 1e-9
        @inbounds for j in 1:N, i in 1:N
            W2[i, j] = wa_U[i] * wa_S[j]        # independence: c_ρ ≡ 1
        end
    else
        den = 2.0 * (1.0 - ρ^2)
        nrm = 1.0 / sqrt(1.0 - ρ^2)
        @inbounds for j in 1:N
            ζv = ζ[j]
            for i in 1:N
                ζu = ζ[i]
                c  = nrm * exp(-(ρ^2 * (ζu^2 + ζv^2) - 2.0 * ρ * ζu * ζv) / den)
                W2[i, j] = c * wa_U[i] * wa_S[j]
            end
        end
    end

    # Sinkhorn-style renormalisation so the discrete margins match wa_U,
    # wa_S exactly (a few sweeps converge to machine precision; the
    # continuous copula already has uniform margins, so this only removes
    # finite-grid quadrature error).
    for _ in 1:64
        rs = vec(sum(W2, dims = 2))
        @inbounds for i in 1:N
            s = rs[i] > 1e-300 ? wa_U[i] / rs[i] : 0.0
            @views W2[i, :] .*= s
        end
        cs = vec(sum(W2, dims = 1))
        @inbounds for j in 1:N
            s = cs[j] > 1e-300 ? wa_S[j] / cs[j] : 0.0
            @views W2[:, j] .*= s
        end
        maximum(abs, vec(sum(W2, dims = 2)) .- wa_U) < 1e-14 && break
    end

    return CopulaGrid(ρ_x, F, ζ, W2)
end


# ============================================================
# Training cost — psychological cost in fixed utility units
#   c(aS) = (1 − aS) · exp(c̃ − aS)      [decreasing in aS]
#
# The cost is NOT scaled by A: it is a utility cost of effort, not a
# dollar outlay.  The training margin compares T(aS) − c(aS) with
# U^search(aU); because T and U^search scale with A while c does not,
# the margin is no longer invariant to A — it shifts with the wage level.
# The argument is the SKILLED aptitude aS: workers more apt at skilled
# work train more cheaply (c'(aS) < 0).
# ============================================================

@inline training_cost(aS::Float64, c::Float64) = (1.0 - aS) * exp(c - aS)


# ============================================================
# Mean sectoral flow output — LMR-style vacancy-cost units
#
#   π̄_U ≡ E[π_U] = exp(A) · P_U · E_ℓ[aU] · E_G[p],  E_G[p] = α_U/(α_U+1)
#   π̄_S ≡ E[π_S] = exp(A) · P_S · E_ℓ[aS] · E_Γ[p],  E_Γ[p] = a_Γ/(a_Γ+b_Γ)
#
# Production is linear in own ability (notes eq. prod), so the ability
# factor is the marginal mean E_ℓ[a] = dot(a, wa).  Vacancy costs are
# k_j · π̄_j with k_j dimensionless (months of average sectoral output),
# exactly LMR (2016).
# ============================================================

function mean_output_U(model)
    gp = model.grids;  up = model.unsk_par;  cp = model.common
    Ea = dot(gp.x, gp.wa_U)                   # E_ℓ[aU]
    Ep = up.α_U / (up.α_U + 1.0)              # E_G[p], G = Beta(α_U, 1)
    return exp(cp.A) * up.PU * Ea * Ep
end

function mean_output_S(model)
    gp = model.grids;  sp = model.skl_par;  cp = model.common
    Ea = dot(gp.x, gp.wa_S)                   # E_ℓ[aS]
    Ep = sp.a_Γ / (sp.a_Γ + sp.b_Γ)           # E_Γ[p], Γ = Beta(a_Γ, b_Γ)
    return exp(cp.A) * sp.PS * Ea * Ep
end


# ============================================================
# Tail-weight vector
#   tail[j] = Σ_{k ≥ j} wp[k]
# Used for Γ-integrals in the skilled block.
# ============================================================

function build_tail_weights(wp::Vector{Float64})
    N    = length(wp)
    tail = similar(wp)
    s    = 0.0
    for i in N:-1:1
        s       += wp[i]
        tail[i]  = s
    end
    return tail
end


# ============================================================
# Soft tail-indicator weight
#
# Linear-in-p smoothing of 1{p ≥ p*} across the grid cell that contains
# p*: returns 1 above the cell, 0 below it, and the fractional coverage
# inside it.  Makes every tail integral ∫_{p*}^1 (·) dG a continuous
# function of p*, which the outer fixed point on the reservation cutoff
# needs to converge smoothly.
# ============================================================

@inline function _soft_weight(pj::Float64, pstar::Float64,
                              pgrid::Vector{Float64}, j::Int, Np::Int)
    pj >= pstar && return 1.0
    j  >= Np     && return 0.0
    p_next = pgrid[min(j + 1, Np)]
    p_next <= pstar && return 0.0
    cell = p_next - pj
    cell < 1e-14 && return 0.0
    return clamp((p_next - pstar) / cell, 0.0, 1.0)
end

# Companion soft weight for the OJS region {p < p^oj}: 1 below the OJS
# cutoff, 0 above it, linear across the straddling cell.
@inline function _soft_oj_weight(pj::Float64, poj::Float64,
                                 pgrid::Vector{Float64}, j::Int, Np::Int)
    pj >= poj && return 0.0
    j  >= Np   && return 1.0
    p_next = pgrid[min(j + 1, Np)]
    p_next <= poj && return 1.0
    cell = p_next - pj
    cell < 1e-14 && return 1.0
    return clamp((poj - pj) / cell, 0.0, 1.0)
end

# Smooth non-negative part: ½(x + √(x²+ε²)) ≈ max(x,0) but C^∞.  Replaces
# max(·,0) on raw surplus expressions so the outer fixed-point map has no
# C⁰-but-not-C¹ kink at grid-aligned sign changes (which admit period-2
# limit cycles in the cutoff iteration).
@inline smooth_pos(x::Float64, ε::Float64 = 1e-8) = 0.5 * (x + sqrt(x * x + ε * ε))


# ============================================================
# Cutoff index helper
# ============================================================

"""
    pcut_index(pgrid, pstar) -> j

First index `j` such that `pgrid[j] >= pstar`. Returns `Np` if none.
"""
@inline function pcut_index(pgrid::Vector{Float64}, pstar::Float64)
    Np = length(pgrid)
    @inbounds for j in 1:Np
        pgrid[j] >= pstar && return j
    end
    return Np
end


# ============================================================
# Anderson acceleration (memory m = 1)
# ============================================================

mutable struct Anderson1
    x_prev      :: Vector{Float64}
    f_prev      :: Vector{Float64}
    initialized :: Bool
end

Anderson1(n::Int) = Anderson1(zeros(n), zeros(n), false)

function anderson1_update!(aa::Anderson1, x::Vector{Float64}, f::Vector{Float64})
    if !aa.initialized
        aa.x_prev      = copy(x)
        aa.f_prev      = copy(f)
        aa.initialized = true
        return copy(f)
    end

    r0    = aa.f_prev .- aa.x_prev
    r1    = f         .- x
    denom = dot(r1 .- r0, r1 .- r0)

    if denom < 1e-12
        aa.x_prev = copy(x)
        aa.f_prev = copy(f)
        return copy(f)
    end

    α     = -dot(r0, r1 .- r0) / denom
    x_new =  α .* f .+ (1.0 - α) .* aa.f_prev

    aa.x_prev = copy(x)
    aa.f_prev = copy(f)
    return x_new
end

############################################################
# grids.jl — Grid construction, matching technology, utilities
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
# Gauss–Jacobi quadrature for the worker-type dimension
#
# The worker-type density is the Beta(a_ℓ, b_ℓ) pdf
#   ℓ(x) = x^{a_ℓ−1} (1−x)^{b_ℓ−1} / B(a_ℓ, b_ℓ).
# When a_ℓ < 1 or b_ℓ < 1 it has an integrable endpoint singularity, and
# plain Gauss–Legendre systematically under-integrates it (e.g. ∫ℓ ≈ 0.967
# at N=200 for b_ℓ≈0.32 — mass is silently lost off the top of the type
# distribution).  Every x-aggregate in the model is ∫ g(x) ℓ(x) dx with
# g(x) = density/ℓ(x) a smooth conditional-occupancy profile (all stocks are
# births ℓ(x) × occupancy), so we integrate against ℓ *exactly* with
# Gauss–Jacobi, whose weight kernel is precisely x^{a−1}(1−x)^{b−1}.
#
# `gaussjacobi(N, α, β)` gives nodes/weights for
#   ∫_{−1}^{1} f(t) (1−t)^α (1+t)^β dt.
# Map x = (t+1)/2 and match x^{a−1}(1−x)^{b−1}: α = b−1 (pairs with 1−t),
# β = a−1 (pairs with 1+t), giving
#   ∫_0^1 g(x) x^{a−1}(1−x)^{b−1} dx = 2^{−(a+b−1)} Σ_i w_i g(x_i).
#
# We return *population weights* wx that fold ℓ in, so that for any state
# density D(x) = g(x) ℓ(x):
#   dot(D, wx) ≈ ∫ D(x) dx      (Gauss–Jacobi accurate for smooth g),
#   dot(ℓ, wx) = 1              (exactly, to machine precision).
# Concretely  wx_i = 2^{−(a+b−1)} · w_i / (x_i^{a−1}(1−x_i)^{b−1});
# the normaliser B(a_ℓ,b_ℓ) cancels and is never formed.
#
# IMPORTANT: because ℓ is folded into wx, these weights integrate DENSITIES
# (objects ∝ ℓ) only.  `dot(density, wx)` is the correct call for every
# aggregate in this model; do NOT use wx to integrate a bare (non-density)
# function of x — that would give ∫ f/ℓ · ℓ = ∫ f only if f/ℓ is smooth,
# which a bare f is not.
# ============================================================

"""
    build_type_grid(N, a, b) -> (nodes, weights)

`N`-point Gauss–Jacobi rule on [0, 1] tuned to the worker-type density
`ℓ = Beta(a, b)`.  Returns nodes `x` and *population weights* `wx` such
that `dot(D, wx) ≈ ∫ D dx` for any density `D(x) ∝ ℓ(x)`, exact to
machine precision for the total mass `dot(ℓ, wx) = 1`.  Valid for any
`a, b > 0`, including the endpoint-singular case `a < 1` or `b < 1`.
"""
function build_type_grid(N::Int, a::Float64, b::Float64)
    (a > 0.0 && b > 0.0) ||
        error("build_type_grid: Beta shape parameters must be positive (got a=$a, b=$b).")
    t, w = gaussjacobi(N, b - 1.0, a - 1.0)        # weight (1−t)^{b−1} (1+t)^{a−1}
    x    = @. 0.5 * (t + 1.0)
    C    = 2.0^(-(a + b - 1.0))
    # Unnormalised Beta kernel at the nodes; Jacobi nodes are interior ⇒
    # strictly positive and finite, so the division below is safe.
    Wker = @. x^(a - 1.0) * (1.0 - x)^(b - 1.0)
    wx   = @. C * w / Wker                          # population weights (ℓ folded in)
    return x, wx
end


# ============================================================
# Training cost — outside-scaling convention
#   c(x)        = (1.0 - x) * exp(c̃ - x)      [scale-free, decreasing in x]
#   dollar cost = exp(A) · c(x)
#
# The stored parameter is the SCALE-FREE coefficient c̃; every caller
# multiplies the FUNCTION by exp(A) outside — training_cost itself never
# sees A.  The training margin therefore compares T(x) − exp(A)·c(x)
# with U^search(x); since T and U^search also scale with exp(A), the
# margin depends on (c̃, primitives) but not on A.
# Interpretation of c̃ on [0,1]:
#   c̃ ≤ −4  → training is effectively free for every x;
#   c̃ ≥ +4  → cost is a near-vertical wall at x = 1 (only the top can train).
# ============================================================

@inline training_cost(x::Float64, c::Float64) = (1.0 - x) * exp(c-x)


# ============================================================
# Ability → productivity map  g(x)   [BASELINE: exp(x)]
#   π_j(x,p) = exp(A) · P_j · g(x) · p ,   j ∈ {U, S},  P_j constant.
#   g(x) = exp(x) ∈ [1, e] on x ∈ [0,1]:  floored at 1 (no dead zone at
#   x = 0), bounded, elegant.  Both sectors share the SAME map; the only
#   sectoral difference is the level P_U vs P_S.
#   *** This is the single place to change the x-map.  Swap to
#       (1.0 + x), (1.0 + γ*x), etc. here and nowhere else. ***
# ============================================================

@inline prod_map(x::Float64) = exp(x)


# ============================================================
# Mean sectoral flow output — LMR-style vacancy-cost units
#
#   π̄_U ≡ E[π_U(x,p)] = exp(A) · P_U · E_ℓ[g(x)]  · E_G[p],  E_G[p] = α_U/(α_U+1)
#   π̄_S ≡ E[π_S(x,p)] = exp(A) · P_S · E_ℓ[g(x)]  · E_Γ[p],  E_Γ[p] = a_Γ/(a_Γ+b_Γ)
#   with g(x) = exp(x)  (EXP-MAP baseline; see prod_map above).
#
# E_ℓ[·] is over the population type distribution ℓ (evaluated on the type
# grid); E_G, E_Γ over the match-quality draw distributions.  Both are
# exogenous functions of the parameters — no endogenous feedback into
# free entry.  Vacancy costs are parameterised as k_j · π̄_j with k_j
# DIMENSIONLESS, measured in months (model periods) of average sectoral
# output — exactly LMR (2016), where the posting cost is c × mean f(x,y)
# and the reported c is "months of average output".
# ============================================================

function mean_output_U(model)
    gp = model.grids;  up = model.unsk_par;  cp = model.common
    Eg = dot(exp.(up.γ_U .* gp.x) .* gp.ℓ, gp.wx) # GAMMA_U: E_ℓ[exp(γ_U·x)]
    Ep = up.α_U / (up.α_U + 1.0)             # E_G[p], G = Beta(α_U, 1)
    return exp(cp.A) * up.PU * Eg * Ep
end

function mean_output_S(model)
    gp = model.grids;  sp = model.skl_par;  cp = model.common
    P_S = sp.PS
    Eg  = dot(exp.(sp.γ_S .* gp.x) .* gp.ℓ, gp.wx) # E_ℓ[exp(γ_S·x)]  (GAMMA_S)
    Ep  = sp.a_Γ / (sp.a_Γ + sp.b_Γ)         # E_Γ[p], Γ = Beta(a_Γ, b_Γ)
    return exp(cp.A) * P_S * Eg * Ep
end


# ============================================================
# Skilled flow-productivity coefficient  (GAMMA_S)
#   PS_of_x(x, P_S, A, γ_S) = A · P_S · exp(γ_S · x)
#   Returns the FULL coefficient: callers compose flow output as
#       π_S(x, p) = PS_of_x(x, P_S, A, γ_S) · p        (NO extra · x).
#   Floored at A·P_S at x = 0; grows with the SKILLED gradient γ_S.
#   γ_S = 1 recovers the old exp(x) map (= unskilled's slope, no comparative
#   advantage). γ_S > 1 makes skilled steeper in x, so unskilled is
#   comparatively more productive at low x (single crossing).
#   NOTE arg order: A is 3rd, γ_S is 4th (both default). Existing 3-arg calls
#   PS_of_x(x, P_S, exp(A)) still compile and give γ_S = 1; production call
#   sites pass the estimated γ_S as the 4th argument.
# ============================================================

@inline function PS_of_x(x::Float64, P_S::Float64, A::Float64=1.0, γ_S::Float64=1.0)
    return A * P_S * exp(γ_S * x)
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
# Cutoff index helpers
# ============================================================

"""
    pcut_index(pgrid, pstar) -> j

First index `j` such that `pgrid[j] >= pstar`. Returns `Np` if none found.
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

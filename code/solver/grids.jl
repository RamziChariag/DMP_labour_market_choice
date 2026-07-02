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
# Training cost
#   c_of_x = x -> (1.0 - x) * exp(cp.c-x)   [decreasing in worker quality x]
#
# NOT scaled by A (it is a utility/shape cost). Note: because the values it is
# compared against scale with the effective scale exp(A), the training-cost
# coefficient c must rise roughly one-for-one with A (≈ log(exp(A)) shift) to
# keep the training margin in place — c and A move along a feasibility ridge.
# ============================================================

@inline training_cost(x::Float64, c::Float64) = (1.0 - x) * exp(c-x)


# ============================================================
# Skilled productivity shifter  PS(x) = γ · x^{γ−1}
#   PDF of Beta(γ, 1) evaluated at x.
#   Zero at x = 0 (for γ > 1), monotonically increasing,
#   equals γ at x = 1.  Single parameter γ controls shape & level.
# ============================================================

@inline function PS_of_x(x::Float64, γ::Float64, A::Float64=1.0)
    x <= 0.0 && return 0.0
    return A * γ * x^(γ - 1.0)
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

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
# Training cost
#   c_of_x = x -> cp.c * (1.0 - x) * exp(-x)   [decreasing in worker quality x]
# ============================================================

@inline training_cost(x::Float64, c::Float64) = c * exp(-6.0 * x)


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

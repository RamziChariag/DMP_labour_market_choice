############################################################
# grids.jl — Grids, matching technology, skill copula and shared numerics
#
# Quadrature on the ability and match-quality grids, the Gaussian skill copula,
# cell-mass densities for the quality distributions, the covered-fraction
# primitives every policy margin is built from, and Anderson(1) acceleration.
############################################################

# ============================================================
# Matching technology (Cobb–Douglas)
# ============================================================

"""
    q_from_theta(θ, μ, η) → q

Vacancy filling rate `q(θ) = μ θ^(−η)`.
"""
@inline q_from_theta(θ::Float64, μ::Float64, η::Float64)    = μ * θ^(-η)

"""
    theta_from_q(q, μ, η) → θ

Tightness at which the vacancy filling rate equals `q`, the inverse of `q_from_theta`.
"""
@inline theta_from_q(q::Float64, μ::Float64, η::Float64)    = (μ / q)^(1/η)

"""
    meeting_rate(θ, μ, η) → κ

Worker meeting rate `κ(θ) = θ q(θ)`.
"""
@inline meeting_rate(θ::Float64, μ::Float64, η::Float64) = θ * q_from_theta(θ, μ, η)


# ============================================================
# Norms and clamp
# ============================================================

"""
    supnorm(a, b) → Float64

Largest absolute elementwise difference between `a` and `b`.
"""
@inline supnorm(a, b)       = maximum(abs.(a .- b))

"""
    clamp01(x) → Float64

`x` clamped to [0, 1].
"""
@inline clamp01(x::Float64) = min(max(x, 0.0), 1.0)


# ============================================================
# Parallel work partition
# ============================================================

"""
    index_chunks(N, [nch]) → Vector{UnitRange{Int}}

Split `1:N` into at most `nch` (default `nthreads()`) contiguous ranges for a
`@sync`/`@spawn` fan-out. `@spawn` nests inside the SMM's own threaded loops, where
`@threads :static` throws; scratch is allocated inside each task, so no `threadid()` is
needed. The partition depends on `N` alone, so a reduction indexed by chunk groups the same
terms on every schedule.
"""
@inline function index_chunks(N::Int, nch::Int = Threads.nthreads())
    N <= 0 && return UnitRange{Int}[]
    n = min(max(nch, 1), N)
    collect(Iterators.partition(1:N, cld(N, n)))
end


# ============================================================
# Quadrature
# ============================================================

"""
    build_gl_grid(N) → (nodes, weights)

`N`-point Gauss–Legendre nodes and weights on [0, 1].
"""
function build_gl_grid(N::Int)
    nodes, weights = gausslegendre(N)
    x = @. 0.5 * (nodes + 1.0)
    w = @. 0.5 * weights
    return x, w
end

"""
    build_ability_grid(N, a, b) → (nodes, wa)

`N`-point Gauss–Jacobi mass rule on [0, 1] for a Beta(a, b) ability marginal: nodes and
population mass weights `wa` with `sum(wa) = 1` and `dot(g, wa) ≈ E[g]` for smooth `g`.
Integrating against the kernel a^(a−1)(1−a)^(b−1) handles the singular `a < 1` / `b < 1`
endpoints. `gaussjacobi(N, α, β)` weights (1−t)^α (1+t)^β on [−1, 1], so α = b − 1 and
β = a − 1 under a = (t+1)/2; the mass weights are the normalised GJ weights (never divided
by the kernel).
"""
function build_ability_grid(N::Int, a::Float64, b::Float64)
    (a > 0.0 && b > 0.0) ||
        error("build_ability_grid: Beta shape parameters must be positive (got a=$a, b=$b).")
    t, w = gaussjacobi(N, b - 1.0, a - 1.0)        # weight (1−t)^{b−1} (1+t)^{a−1}
    x    = @. 0.5 * (t + 1.0)
    wa   = w ./ sum(w)                              # population mass weights, Σ = 1
    return x, wa
end


"""
    node_midpoint_intervals(x) → (xlo, xhi)

The ability cell of each node: midpoints to its neighbours, clipped to the grid's support at
the ends. The one definition of a cell for every covered-fraction margin (τ, d, χ, OJS
participation).
"""
function node_midpoint_intervals(x::AbstractVector{Float64})
    N   = length(x)
    xlo = Vector{Float64}(undef, N);  xhi = Vector{Float64}(undef, N)
    @inbounds for j in 1:N
        xlo[j] = j == 1 ? x[1] : 0.5 * (x[j - 1] + x[j])
        xhi[j] = j == N ? x[N] : 0.5 * (x[j] + x[j + 1])
    end
    return xlo, xhi
end

"""
    fraction_above(xlo, xhi, x, a) → Float64

Fraction of the ability cell `[xlo, xhi]` (node `x`) above the frontier `a`: piecewise linear
in `a`, equal to the indicator when `a` is on a cell edge. A zero-width cell takes the
indicator at its node.
"""
@inline function fraction_above(xlo::Float64, xhi::Float64, x::Float64, a::Float64)
    w = xhi - xlo
    return w > 0.0 ? clamp((xhi - a) / w, 0.0, 1.0) : (x >= a ? 1.0 : 0.0)
end


# ============================================================
# Skill copula
#
# ℓ(aU,aS) = f_ℓ(aU) f_ℓ(aS) c_ρ(F_ℓ(aU), F_ℓ(aS)), a Gaussian copula with correlation ρ_x
# on the ranks ζ = Φ⁻¹(F_ℓ). On the grid W2[i,j] = c_ρ(F_i, F_j) wa_U[i] wa_S[j], so that
# E[g] = Σ g_ij W2_ij. ρ_x → 1 makes W2 diagonal (the single-ability model).
# ============================================================

"""
    CopulaGrid

Joint aggregation weights of the skill copula: `sum(g .* W2) ≈ ∬ g ℓ daU daS`.
Fields: `ρ_x` the copula correlation, `F` the marginal CDF at the nodes, `ζ` the Gaussian
ranks Φ⁻¹(F), `W2` the joint weights (rows aU, columns aS).
"""
struct CopulaGrid
    ρ_x :: Float64
    F   :: Vector{Float64}
    ζ   :: Vector{Float64}
    W2  :: Matrix{Float64}
end

"""
    build_copula(ρ_x, xgrid, wa_U, wa_S, a_ℓ, b_ℓ) → CopulaGrid

Joint weights for correlation `ρ_x` on `xgrid` with marginal mass weights `wa_U`, `wa_S`
of a Beta(`a_ℓ`, `b_ℓ`) marginal. Sinkhorn renormalisation makes row sums `wa_U`, column
sums `wa_S` and the total 1 to machine precision at every ρ_x, as the ρ_x → 1 nesting
requires.
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

    # Sinkhorn sweeps: remove the finite-grid error in the margins.
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
# Training cost, empty markets, vacancy-cost units
# ============================================================

"""
    training_cost(aS, c) → Float64

Psychological training cost `(1 − aS) exp(c − aS)`, `c` the stored ln c. In utility units,
not scaled by A, so the training margin moves with the wage level.
"""
@inline training_cost(aS::Float64, c::Float64) = (1.0 - aS) * exp(c - aS)

# A market whose mass is below EMPTY_MASS holds no one: its within-market objects are read at
# the limit of its composition (marginal_training_cells) and its aggregates are zero.
const EMPTY_MASS = 1.0e-12

"""
    mean_output_U(model) → π̄_U

Mean flow output of a new unskilled match, `exp(A) P_U E[aU]` (a new match starts at p = 1).
Unit of the unskilled vacancy cost: dollar cost = k_U π̄_U.
"""
function mean_output_U(model)
    gp = model.grids;  up = model.unsk_par;  cp = model.common
    Ea = dot(gp.x, gp.wa_U)                   # E_ℓ[aU]; starting quality p = 1
    return exp(cp.A) * up.PU * Ea
end

"""
    mean_output_S(model) → π̄_S

Mean flow output of a new skilled match, `exp(A) P_S E[aS] E_Γ[p]`. Unit of the skilled
vacancy cost: dollar cost = k_S π̄_S.
"""
function mean_output_S(model)
    gp = model.grids;  sp = model.skl_par;  cp = model.common
    Ea = dot(gp.x, gp.wa_S)                   # E_ℓ[aS]
    Ep = sp.a_Γ / (sp.a_Γ + sp.b_Γ)           # E_Γ[p], Γ = Beta(a_Γ, b_Γ)
    return exp(cp.A) * sp.PS * Ea * Ep
end


# ============================================================
# Quality-distribution quadrature
# ============================================================

"""
    build_tail_weights(wp) → tail

Suffix sums `tail[j] = Σ_{k ≥ j} wp[k]`.
"""
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

"""
    build_node_cell_edges(p) → e

Edges of the cells owned by the nodes `p` on [0, 1]: midpoints between nodes, closed by 0
and 1, so the `length(p)` cells tile [0, 1].
"""
function build_node_cell_edges(p::Vector{Float64})
    N = length(p)
    e = Vector{Float64}(undef, N + 1)
    e[1] = 0.0;  e[N + 1] = 1.0
    @inbounds for j in 2:N
        e[j] = 0.5 * (p[j - 1] + p[j])
    end
    return e
end

"""
    build_cell_mass_density(Γ_at, p, wp) → γ̃

Exact-CDF cell mass per unit quadrature weight, `γ̃_j = (Γ(e_{j+1}) − Γ(e_j)) / wp_j`, for
the CDF `Γ_at`. Every `∫ g dΓ` appears as `Σ g_j γ̃_j wp_j`, so it carries the exact cell
mass (total 1, and the mass near a density pole comes from the CDF). Not a pointwise
density: never read it as one.
"""
function build_cell_mass_density(Γ_at, p::Vector{Float64}, wp::Vector{Float64})
    e  = build_node_cell_edges(p)
    Fe = Γ_at.(e)
    return [wp[j] > 0.0 ? max(Fe[j + 1] - Fe[j], 0.0) / wp[j] : 0.0 for j in eachindex(p)]
end


# ============================================================
# Covered fractions
#
# Every margin that moves with the parameters across a fixed grid enters as the covered
# fraction of a cell, never a node indicator, so aggregates are continuous in the parameters.
# ============================================================

"""
    _soft_weight(pj, pstar, pgrid, j, Np) → Float64

Covered fraction of node `j`'s cell [p_j, p_{j+1}] (the top node's is [p_N, 1]) above the
cutoff `pstar`: 1 above the cell, 0 below it, linear inside it.
"""
@inline function _soft_weight(pj::Float64, pstar::Float64,
                              pgrid::Vector{Float64}, j::Int, Np::Int)
    pj >= pstar && return 1.0
    p_next = j < Np ? pgrid[j + 1] : 1.0
    p_next <= pstar && return 0.0
    cell = p_next - pj
    cell < 1e-14 && return 0.0
    return clamp((p_next - pstar) / cell, 0.0, 1.0)
end

"""
    _soft_oj_weight(pj, poj, pgrid, j, Np) → Float64

Covered fraction of node `j`'s cell below the cutoff `poj` (the OJS region): the complement
of `_soft_weight` on the same cells.
"""
@inline function _soft_oj_weight(pj::Float64, poj::Float64,
                                 pgrid::Vector{Float64}, j::Int, Np::Int)
    pj >= poj && return 0.0
    p_next = j < Np ? pgrid[j + 1] : 1.0
    p_next <= poj && return 1.0
    cell = p_next - pj
    cell < 1e-14 && return 1.0
    return clamp((poj - pj) / cell, 0.0, 1.0)
end

"""
    _pos_share(fa, fb, xa, xb, lo, hi) → Float64

Fraction of `[lo, hi] ⊆ [xa, xb]` on which the linear interpolant through `(xa, fa)` and
`(xb, fb)` is positive.
"""
@inline function _pos_share(fa::Float64, fb::Float64, xa::Float64, xb::Float64,
                            lo::Float64, hi::Float64)
    fa >  0.0 && fb >  0.0 && return 1.0
    fa <= 0.0 && fb <= 0.0 && return 0.0
    hi - lo < 1e-14 && return 0.0
    xc = xa + fa / (fa - fb) * (xb - xa)            # zero of the interpolant
    return fa > 0.0 ? clamp((xc - lo) / (hi - lo), 0.0, 1.0) :
                      clamp((hi - xc) / (hi - lo), 0.0, 1.0)
end

"""
    covered_fraction!(share, g, x, xlo, xhi) → share

Fraction of each ability cell `[xlo[k], xhi[k]]` on which `g > 0`, with `g` linear between
nodes. Each half-cell is scored against its own neighbour, so every sign change along the
grid is located; at the grid ends the missing half takes the node's sign. Equal to the
indicator on every cell no sign change cuts.
"""
function covered_fraction!(share::AbstractVector{Float64}, g::AbstractVector{Float64},
                           x::AbstractVector{Float64},
                           xlo::AbstractVector{Float64}, xhi::AbstractVector{Float64})
    Nx = length(x)
    @inbounds for k in 1:Nx
        wl = x[k] - xlo[k];  wr = xhi[k] - x[k]
        sl = k > 1  ? _pos_share(g[k - 1], g[k], x[k - 1], x[k], xlo[k], x[k]) :
                      (g[k] > 0.0 ? 1.0 : 0.0)
        sr = k < Nx ? _pos_share(g[k], g[k + 1], x[k], x[k + 1], x[k], xhi[k]) :
                      (g[k] > 0.0 ? 1.0 : 0.0)
        share[k] = wl + wr > 1e-14 ? (wl * sl + wr * sr) / (wl + wr) :
                                     (g[k] > 0.0 ? 1.0 : 0.0)
    end
    return share
end

"""
    _ojs_share(part, pj, poj, pgrid, j, Np) → Float64

Searching share of skilled cell (aS, p): the type's OJS participation weight `part` times
the p-coverage of the region below `poj`. Every consumer of the OJS margin reads it.
"""
@inline _ojs_share(part::Float64, pj::Float64, poj::Float64,
                   pgrid::Vector{Float64}, j::Int, Np::Int) =
    part * _soft_oj_weight(pj, poj, pgrid, j, Np)

"""
    smooth_pos(x, ε = 1e-8) → Float64

Smooth positive part `(x + √(x² + ε²)) / 2`. Used for the surpluses the inner loops price, so
the cutoff map has no kink (a kink admits period-2 cycles). Never used to locate p*_S: it is
strictly positive and would never cross zero.
"""
@inline smooth_pos(x::Float64, ε::Float64 = 1e-8) = 0.5 * (x + sqrt(x * x + ε * ε))


# ============================================================
# Cutoff helpers
# ============================================================

"""
    pcut_index(pgrid, pstar) → j

First index with `pgrid[j] >= pstar`, or `Np` if none. For loop bounds only: to read a CDF
at a cutoff use `cdf_at_cutoff`.
"""
@inline function pcut_index(pgrid::Vector{Float64}, pstar::Float64)
    Np = length(pgrid)
    @inbounds for j in 1:Np
        pgrid[j] >= pstar && return j
    end
    return Np
end

"""
    cdf_at_cutoff(γcells, pgrid, wp, cutoff) → Γ(cutoff)

CDF of a cell-mass density at an off-node cutoff: the mass of the cells below `cutoff`, the
straddling cell at its covered fraction. Written as the mass below rather than `1 − ∫ω dΓ`
so that a dead margin (`cutoff = 0`) is exactly 0, not a rounding residual.
"""
@inline function cdf_at_cutoff(γcells::AbstractVector{Float64}, pgrid::Vector{Float64},
                               wp::Vector{Float64}, cutoff::Float64)
    Np   = length(pgrid)
    mass = 0.0
    @inbounds for j in 1:Np
        ω = _soft_oj_weight(pgrid[j], cutoff, pgrid, j, Np)
        ω > 0.0 && (mass += ω * γcells[j] * wp[j])
    end
    return mass
end


# ============================================================
# Anderson acceleration (memory 1)
# ============================================================

"""
    Anderson1

State of a memory-1 Anderson accelerator: the previous iterate `x_prev`, its image
`f_prev`, and whether a previous pair exists.
"""
mutable struct Anderson1
    x_prev      :: Vector{Float64}
    f_prev      :: Vector{Float64}
    initialized :: Bool
end

"""
    Anderson1(n) → Anderson1

An empty accelerator for vectors of length `n`.
"""
Anderson1(n::Int) = Anderson1(zeros(n), zeros(n), false)

"""
    anderson1_update!(aa, x, f) → x_new

One Anderson(1) step for the map `x ↦ f`: the mix `α f + (1 − α) f_prev` minimising the
linearised residual. Plain Picard (`f`) on the first call and when the residual difference
is below 1e-12.
"""
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

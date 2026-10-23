############################################################
# unskilled.jl — Unskilled block solver (RoySearch)
#
# Under pure-Roy every unskilled-market VALUE object of a type that searches is a
# function of the unskilled aptitude aU alone (notes §231, §276): U^search, E_U, J_U,
# p* read aU.  The two-dimensional type enters through
#   (i)   the training value T(aU,aS) = [b_T + φ U_S(aU,aS)] / (r+φ+ν), priced on the
#         skilled unemployment value U_S = max{U_S^(0)(aS), U_S^(1)(aU,aS)}, and the
#         frontier τ(aU,aS) = 1{−c(aS) + T(aU,aS) ≥ U^search(aU)};
#   (ii)  the trained workers hired into the unskilled market.  Such a worker is an
#         unskilled worker like any other, so her outside option when the match ends is
#         U_U = max{U^search(aU), −c(aS) + T(aU,aS)}: where the frontier encloses her
#         type it is training, and her match values, reservation quality and wages read
#         the whole type (career_choices!);
#   (iii) the augmented seeker pool ũ_U = ∬ (u_U + d u_S), and
#   (iv)  the stationary densities u_U(aU,aS), t(aU,aS) on the copula grid.
#
# The training decision is NOT gated by d.  b_T discipline comes purely
# from comparative advantage (notes Prop bT): the would-be stipend
# farmers are high-aU / low-aS types who forgo a good unskilled job by
# training, so raising b_T shifts only the marginal genuine trainee at
# the frontier.  (The single-ability code gated τ by d as an explicit
# refinement; RoySearch does not need it and does not use it.)
#
# Outer loop iterates on (θ_U, p*(aU), u_U, t); the inner loop settles
# (U^search, E_U(·,1), J_U(·,1)) at fixed (θ_U, p*).  The cross-market
# pool d·u_S is supplied by the skilled block via uc.duS_carry and held
# fixed within one unskilled solve.
#
# Functions
#   build_unskilled_G_weights        dG(p) quadrature weights, G=Beta(α_U,1)
#   G_cdf_unskilled                  G(p) = p^{α_U}
#   solve_unskilled_surplus_on_grid! surplus S_U(aU,·) on the p-grid
#   entry_surplus_unskilled          surplus S_U(aU,1) of a fresh match
#   training_frontier!, training_policy!, training_value!   the training margin
#   training_choice!, drain_choice!  τ and d at the installed values, or held (Model.held)
#   unskilled_inner_loop!            iterate values at (θ_U, p*)
#   hire_fraction                    hired fraction χ(aU) of each ability cell
#   unemployment_split_unskilled     per-row unemployed share and its hired part
#   hired_unemployed                 hired branch of a cell's unemployed stock
#   career_choices!                  U_S^(1), T, τ, d and the training-slice matches
#   HeldChoices(model)               a solved model's choices, for another model to hold
#   solve_stationary_unskilled!      stationary u_U, t of the searching types
#   stationary_masses!               every stationary mass, crossers included
#   update_pstar_from_surplus!       reservation-quality update
#   update_theta_unskilled           free-entry tightness update
#   solve_unskilled_block!           outer loop
############################################################


# ---------------------------------------------------------------------------
# dG quadrature weights and CDF for the damage shock
#   G = Beta(α_U, 1),  g(p) = α_U p^{α_U−1},  G(p) = p^{α_U}
# ---------------------------------------------------------------------------
function build_unskilled_G_weights(pgrid::Vector{Float64}, wp::Vector{Float64}, α::Float64)
    wG = similar(wp)
    @inbounds for j in eachindex(pgrid)
        p     = pgrid[j]
        g     = (p <= 0.0) ? 0.0 : α * p^(α - 1.0)
        wG[j] = wp[j] * g
    end
    return wG
end

@inline function G_cdf_unskilled(p::Float64, α::Float64)
    (p <= 0.0) && return 0.0
    (p >= 1.0) && return 1.0
    return p^α
end


# ---------------------------------------------------------------------------
# Unskilled surplus on the p-grid, for one aU
#
# Closed-form reduction of the surplus HJB (notes eq:surplusU):
#   (r+ν+λ+ξ) S(p) = A P_U aU p − (r+ν) U^search(aU) + λ ∫_{p*}^1 S dG,
#   S(p) = max( [A P_U aU p − (r+ν) U^search + λ I] / (r+ν+λ+ξ), 0 ),
# with the scalar tail integral I = ∫_{p*}^1 S dG solved analytically.
# The exogenous separation ξ_U enters only the base rate: the worker's
# +ξ_U U^search continuation and the firm's asset extinction combine to
# −ξ_U S, which nets into the base (mirror skilled.jl ϱ_S).  The reservation
# FORMULA in update_pstar_from_surplus! is therefore unchanged by ξ_U — no ξ
# appears in it — but p* itself is not: it reads the tail integral I, which is
# computed off this denominator.  Measured at the Q = 488 base_fc point,
# ξ_U 0 → 0.00634 moves median p*_U from 0.4228 to 0.4582.
# Soft thresholding around p* keeps S(·) continuous in p*.
#
# `PUeff` is the effective productivity A P_U aU already carrying exp(A)
# and the ability level, i.e. the linear-in-aU loading of production.
# ---------------------------------------------------------------------------
function solve_unskilled_surplus_on_grid!(
    Svec::Vector{Float64}, pgrid::Vector{Float64}, wG::Vector{Float64},
    PUeff::Float64, r::Float64, ν::Float64, λ::Float64,
    Ux::Float64, pstar_x::Float64, ξ::Float64 = 0.0,
)
    Np = length(pgrid)

    tail_mass = 0.0
    tail_p1   = 0.0
    @inbounds for j in 1:Np
        ω          = _soft_weight(pgrid[j], pstar_x, pgrid, j, Np)
        tail_mass += ω * wG[j]
        tail_p1   += ω * pgrid[j] * wG[j]
    end

    denom = r + ν + λ + ξ
    num   = (PUeff * tail_p1 - (r + ν) * Ux * tail_mass) / denom
    B     = (λ / denom) * tail_mass
    I     = (abs(1.0 - B) < 1e-14) ? 0.0 : num / (1.0 - B)

    @inbounds for j in 1:Np
        ω       = _soft_weight(pgrid[j], pstar_x, pgrid, j, Np)
        Svec[j] = max(ω * (PUeff * pgrid[j] - (r + ν) * Ux + λ * I) / denom, 0.0)
    end

    return I
end

"""
    entry_surplus_unskilled(PUeff, flow, λ, I, D) → S_U(aU, 1)

Surplus of a fresh unskilled match, which starts at the top of the quality support, p = 1:
`max((A P_U aU · 1 + flow + λ I) / D, 0)`.  `flow` is the rest of the numerator, which does
not involve p (`−(r+ν) U^search` in a steady state; the transition adds its continuation),
`I` the tail integral on the grid and `D` the discount.  p = 1 lies above the top quadrature
node, so reading the grid surface there instead would price entry at `p_N < 1` and leave a
band of hired types, `a^h < aU < a^h/p_N`, at zero surplus.  Evaluated at p = 1, the surplus
is continuous in aU and vanishes at `a^h`, as the hireable-set lemma says.
"""
@inline entry_surplus_unskilled(PUeff::Float64, flow::Float64, λ::Float64, I::Float64, D::Float64) =
    max((PUeff + flow + λ * I) / D, 0.0)


# ---------------------------------------------------------------------------
# The training margin
# ---------------------------------------------------------------------------
"""
    training_frontier!(T, US, bT, φ, r, ν)

Training value `(r + φ + ν) T(a) = b_T + φ U_S(a)`, cell by cell.  Extracted so the
choices and the equilibrium residual that checks them share one implementation: a residual
computed from a second copy of a formula tests the copy, not the solver.
"""
function training_frontier!(T::AbstractMatrix{Float64}, US::AbstractMatrix{Float64},
                            bT::Float64, φ::Float64, r::Float64, ν::Float64)
    d = r + φ + ν
    @inbounds @threads for k in eachindex(T)
        T[k] = (bT + φ * US[k]) / d
    end
    return T
end

"""
    training_policy!(τT, Utr, C, x)

Training choice `τ(a_U, a_S)`: the covered fraction of each cell on which the training
option `Utr(a_U, a_S)` beats the value of searching `C(a_U, a_S)` (covered_fraction!, along
each a_U row).  Utr is increasing in aS where T is the stay-skilled branch's, so the
training set is then an upper interval; where crossing pays T reads aU as well, and each
sign change is located where it lies.  A hard 0/1 at the nodes made every composition
aggregate a step function of the parameters (v22.0.0: one of 14400 cells flipping moved Q
by 0.18).

Shared by the iteration, the equilibrium residual and the transition, for the reason above.
"""
function training_policy!(τT::AbstractMatrix{Float64}, Utr::AbstractMatrix{Float64},
                          C::AbstractMatrix{Float64}, x::AbstractVector{Float64})
    xlo, xhi = node_midpoint_intervals(x)
    @threads for i in axes(τT, 1)
        g = [Utr[i, j] - C[i, j] for j in axes(τT, 2)]
        covered_fraction!(view(τT, i, :), g, x, xlo, xhi)
    end
    return τT
end

"""
    training_value!(Utr, model) → Utr

Value of entering training, net of its cost: `T(a_U, a_S) − c(a_S)`.  The one expression
every reader of the training margin compares against the value of searching.
"""
function training_value!(Utr::AbstractMatrix{Float64}, model::Model)
    cp = model.common;  gp = model.grids;  uc = model.unsk_cache
    @inbounds for j in axes(Utr, 2), i in axes(Utr, 1)
        Utr[i, j] = -training_cost(gp.x[j], cp.c) + uc.T[i, j]
    end
    return Utr
end

# The value of searching on the (aU, aS) grid, constant along each row.
searching_value(model::Model) = repeat(model.unsk_cache.Usearch, 1, length(model.grids.x))

"""
    training_choice!(τT, model, Utr = −c + T) → τT

The training choice at the installed values, `training_policy!` of `Utr` against the value of
searching, or the held τ where the model holds its choices (`Model.held`).  The one call every
reader of the choice makes, so a held choice is held everywhere at once.
"""
function training_choice!(τT::AbstractMatrix{Float64}, model::Model,
                          Utr::AbstractMatrix{Float64} = training_value!(similar(model.unsk_cache.T), model))
    model.held === nothing || return copyto!(τT, model.held.τ)
    return training_policy!(τT, Utr, searching_value(model), model.grids.x)
end

"""
    marginal_training_cells(model; entering, admit = nothing) → Matrix{Float64}

Composition of a market that holds no one, as the limit of its composition while it
empties.  Markets empty through the training margin, and the last cells to cross it are
the first to cross back: for the skilled market (`entering = true`) the cells with the
largest training gain `Utr(a_U, a_S) − U^search(a_U)`, for the unskilled labour force
(`entering = false`) those with the smallest.  Returns a unit mass on those cells, for
the model's own stationary formulas to turn into a composition — unit so that its size
never depends on how thin the copula is at the marginal cells, and only its ratios carry
meaning.  Cells tied to a relative 1e-12 share it in proportion to `W2`.  `admit`
restricts the candidate cells; with none admitted the result is zero.
"""
function marginal_training_cells(model::Model; entering::Bool,
                                 admit::Union{Nothing,AbstractMatrix{Bool}} = nothing)
    uc  = model.unsk_cache;  W2 = model.grids.copula.W2;  Nx = length(uc.Usearch)
    Utr = training_value!(similar(uc.T), model)
    gain(i, j) = entering ? Utr[i, j] - uc.Usearch[i] : uc.Usearch[i] - Utr[i, j]
    admitted(i, j) = admit === nothing || admit[i, j]

    best = -Inf
    @inbounds for j in 1:Nx, i in 1:Nx
        admitted(i, j) && (best = max(best, gain(i, j)))
    end
    w = zeros(Float64, Nx, Nx)
    isfinite(best) || return w
    cut = best - 1e-12 * max(abs(best), 1.0)
    @inbounds for j in 1:Nx, i in 1:Nx
        (admitted(i, j) && gain(i, j) >= cut) && (w[i, j] = W2[i, j])
    end
    tot = sum(w)
    return tot > 0.0 ? w ./ tot : w
end


# ---------------------------------------------------------------------------
# Inner loop
#
# Given θ_U and p*(aU) fixed, iterate on (U^search, E_U(·,1), J_U(·,1))
# until convergence.
#
# At p = 1, Nash bargaining gives J_U(aU,1) = (1−β) S_U(aU,1),
# E_U(aU,1) = U^search(aU) + β S_U(aU,1); the search-value update consumes
# E_U(aU,1) via (r+ν+κ_U) U^search = b_U + κ_U E_U(aU,1).  The training value
# T is formed here on the supplied skilled unemployment value U_S(aU,aS), and the
# frontier τ from it for the distribution step; neither feeds the searchers'
# outside option, which is U^search(aU) alone (notes §276).
# ---------------------------------------------------------------------------
function unskilled_inner_loop!(model::Model; US_in::AbstractMatrix{Float64})
    cp = model.common;  gp = model.grids;  up = model.unsk_par
    ug = model.unsk_grids;  uc = model.unsk_cache;  sim = model.sim

    Nx = length(gp.x);  Np = length(ug.p)
    r  = cp.r;  ν = cp.ν;  φ = cp.φ
    μ  = up.μ;  η = up.η;  β = up.β;  λ = up.λ;  ξ = up.ξ
    PU = exp(cp.A) * up.PU;  bU = up.bU * exp(cp.A);  bT = up.bT * exp(cp.A);  α = up.α_U

    κ  = meeting_rate(uc.θ, μ, η)
    wG = build_unskilled_G_weights(ug.p, ug.wp, α)

    Ivec  = zeros(Float64, Nx)
    E1vec = zeros(Float64, Nx)

    # Fixed partition of the aU grid for the surplus fan-out below.  Computed once per
    # call rather than per iteration: at fixed Nx it is the same partition every time,
    # which is what makes the block bitwise reproducible across schedules.
    chunks = index_chunks(Nx)

    Usearch_old = copy(uc.Usearch)

    # T reads only US_in, which is fixed within the loop, so it is set once; its step
    # enters the first pass's residual, so a pass that moved T never counts as converged.
    T_old = copy(uc.T)
    training_frontier!(uc.T, US_in, bT, φ, r, ν)
    dT = supnorm(uc.T, T_old)

    αU     = sim.damp_inner_U
    B      = sim.inner_B;  K = sim.inner_K
    resid  = Vector{Float64}(undef, sim.maxit_inner)
    status = :maxit;  n_inner = 0;  streak = 0

    for it in 1:sim.maxit_inner
        copyto!(Usearch_old, uc.Usearch)

        # Match surplus at the current U^search outside option → J_U(aU,1), E_U(aU,1).
        # One `Svec` scratch vector per CHUNK, allocated inside the task: see
        # `index_chunks` for why this is a @spawn fan-out and not `@threads :static`.
        @sync for rng in chunks
            Threads.@spawn begin
                Svec = zeros(Float64, Np)
                for ix in rng
                    @inbounds begin
                        PUeff    = PU * gp.x[ix]               # A P_U aU (linear loading)
                        Ivec[ix] = solve_unskilled_surplus_on_grid!(
                            Svec, ug.p, wG, PUeff, r, ν, λ,
                            uc.Usearch[ix], clamp01(uc.pstar[ix]), ξ)
                        S1               = entry_surplus_unskilled(PUeff, -(r + ν) * uc.Usearch[ix],
                                                                   λ, Ivec[ix], r + ν + λ + ξ)
                        uc.Jfrontier[ix] = (1.0 - β) * S1
                        E1vec[ix]        = uc.Usearch[ix] + β * S1
                    end
                end
            end
        end

        # Search value:  (r + ν + κ_U) U^search(aU) = b_U + κ_U E_U(aU,1)
        denom_search = r + ν + κ
        @threads for ix in 1:Nx
            @inbounds begin
                raw            = (bU + κ * E1vec[ix]) / denom_search
                uc.Usearch[ix] = Usearch_old[ix] + αU * (raw - Usearch_old[ix])
            end
        end

        d = it == 1 ? max(supnorm(uc.Usearch, Usearch_old), dT) : supnorm(uc.Usearch, Usearch_old)
        resid[it] = d;  n_inner = it

        if d < sim.tol_inner
            streak += 1
            streak >= sim.conv_streak && (status = :converged; break)
        else
            streak = 0
        end
        if B > 0 && it > B + K && resid[it] >= resid[it - K]
            status = :diverged;  break
        end
    end

    # Training frontier τ(aU,aS): the POPULATION FRACTION of cell (aU,aS) that trains,
    # from Utr(aU,aS) = −c(aS) + T(aU,aS) > U^search(aU), with no d-gate (b_T discipline is
    # by comparative advantage).  c(aS) is a psychological cost in fixed utility units, NOT
    # scaled by A, so the training margin shifts with the wage level.
    training_choice!(uc.τT, model)

    return (κ = κ, Ivec = Ivec, E1vec = E1vec,
            status = status, converged = (status === :converged), iters = n_inner)
end


# ---------------------------------------------------------------------------
# Hiring margin
#
# A fresh unskilled match starts at the top of the quality support, p = 1, so
# a meeting is accepted with probability one or zero: a type is hired iff
# J_U(aU,1) > 0.  By the hireable-set lemma that holds iff A P_U aU > b_U,
# i.e. aU > a^h = b̃_U / P_U (A cancels), at every tightness — the hireable
# set is fixed by primitives.  Never-hired types search, are unemployed and
# stay in the seeker pool; they leave only at ν.
# ---------------------------------------------------------------------------
"""
    hire_fraction(model) → Vector{Float64}

Hired fraction χ(a_U) of each ability cell: the part of the node's midpoint interval
above the hiring threshold `a^h = b̃_U / P_U`, by the quadrature τ_T uses.  A per-node
indicator 1{p*_U < 1} would move a whole row between the hired and never-hired branches
as `a^h` crosses a node, making every aggregate a step function of `b̃_U / P_U`; χ is
continuous and piecewise linear in it, and equals that indicator whenever `a^h` sits on a
cell edge.
"""
function hire_fraction(model::Model)
    x  = model.grids.x
    aʰ = model.unsk_par.bU / model.unsk_par.PU
    xlo, xhi = node_midpoint_intervals(x)
    return [fraction_above(xlo[i], xhi[i], x[i], aʰ) for i in eachindex(x)]
end

"""
    unemployment_split_unskilled(pstar, κ, model) → (χ, usurv, hired)

Per a_U row, the stationary unemployment of its searching population, split into the
hired branch (fraction χ) and the never-hired branch (1 − χ):
- `usurv = χ υʰ + (1 − χ)`, the unemployed share of the searching population, with
  `υʰ = (δ+ν)/(κ+δ+ν)` and `δ = ξ_U + λ_U G(p*)`.  The never-hired branch is wholly
  unemployed.
- `hired = χ υʰ / usurv`, the hired branch's share of the row's unemployed: the share that
  leaves at κ_U + ν rather than ν.

`υʰ` reads the node's own p* and `κ` whatever the node's classification, so the hired
sliver of a node below `a^h` takes the node's values (p* = 1), as a τ cell does.  Both
are row properties, so they apply to any composition over a_S.
"""
function unemployment_split_unskilled(pstar::AbstractVector{Float64}, κ::Float64, model::Model)
    ν = model.common.ν
    λ = model.unsk_par.λ;  α = model.unsk_par.α_U;  ξ = model.unsk_par.ξ
    χ = hire_fraction(model)
    usurv = similar(χ);  hired = similar(χ)
    @inbounds for i in eachindex(χ)
        δ        = ξ + λ * G_cdf_unskilled(clamp01(pstar[i]), α)
        denom    = κ + δ + ν
        υʰ       = denom > 0.0 ? (δ + ν) / denom : 0.0
        usurv[i] = χ[i] * υʰ + (1.0 - χ[i])
        hired[i] = usurv[i] > 0.0 ? χ[i] * υʰ / usurv[i] : 0.0
    end
    return (χ = χ, usurv = usurv, hired = hired)
end

"""
    hired_unemployed(u, χ, τ, ℓ) → Float64

The hired branch of a cell's unemployed mass `u`, read off the stock.  The never-hired
branch is the part of the cell's searching slice below `a^h`, `(1 − χ)(1 − τ)ℓ`, and is
wholly unemployed, so the rest of `u` is the hired branch's.  In a stationary state this
equals `hired · u` (`unemployment_split_unskilled`); off one it is the split the
transition carries, since there the row's unemployed are not at their stationary shares.
"""
@inline hired_unemployed(u::Float64, χ::Float64, τ::Float64, ℓ::Float64) =
    max(u - (1.0 - χ) * (1.0 - τ) * ℓ, 0.0)


# ---------------------------------------------------------------------------
# Career choices of the trained
#
# A trained worker who searches the unskilled market and is hired there becomes an
# unskilled worker: she loses trained status, and when the match ends her outside option
# is that of any unskilled worker of her type.  Outside the training set that is
# U^search(aU), and her match is the row's own (p*_U, E_U(aU,1), hire share χ).  Inside it
# her separations re-enter training, so her match is valued at the outside option
# Utr = −c(aS) + T(aU,aS): it has its own reservation quality p*_t(aU,aS), entry surplus
# S_t(aU,aS,1) and hiring margin S_t ≥ 0 (the training slice).  The value of searching
# the unskilled market as a trained worker is
#   (r+ν+κ_X) U_S^(1) = b_S + κ_X E_U(aU,aS,1),   κ_X = κ_U χ_X,
# with E_U = U_U + β_U S(·,1)⁺, U_U = max{U^search, Utr}, and χ_X the hired share of the
# match she would enter.  Since T = [b_T + φ max{U_S^(0), U_S^(1)}]/(r+φ+ν) prices
# U_S^(1) and U_S^(1) reads T through U_U, a cell inside the training set that crosses
# solves one scalar fixed point.
#
# A cell's draining share d splits as the transition splits it: the share enclosed by
# the training frontier, max(d+τ−1, 0)/d, joins the training slice and the rest the
# searching slice (`enclosed_crossers`).
# ---------------------------------------------------------------------------
"""
    enclosed_crossers(d, τ) → Float64

Share of a cell's draining fraction `d` that lies inside its training fraction `τ`: the
drain set is the lower part of the cell in aS and the training set the upper part, so they
overlap on `max(d + τ − 1, 0)` of the cell.
"""
@inline enclosed_crossers(d::Float64, τ::Float64) = d > 0.0 ? max(d + τ - 1.0, 0.0) / d : 0.0

# Suffix sums of the dG quadrature, Σ_{k≥j} wG_k and Σ_{k≥j} p_k wG_k, so that the tail
# integral at any reservation quality is O(1) after a bisection for the cutoff node.
function unskilled_tails(p::AbstractVector{Float64}, wG::AbstractVector{Float64})
    Np = length(p)
    CM = zeros(Float64, Np + 1);  CP = zeros(Float64, Np + 1)
    @inbounds for j in Np:-1:1
        CM[j] = CM[j + 1] + wG[j]
        CP[j] = CP[j + 1] + p[j] * wG[j]
    end
    return (CM = CM, CP = CP)
end

# Tail integral I = ∫_{p*}^1 S dG of the match with outside option O at reservation
# quality pst: solve_unskilled_surplus_on_grid!'s I, with its soft weight on the node
# straddling pst and full weight above.
function unskilled_tail_integral(pst::Float64, O::Float64, PUeff::Float64,
                                 p::AbstractVector{Float64}, wG::AbstractVector{Float64},
                                 tails, r::Float64, ν::Float64, λ::Float64, D::Float64)
    Np = length(p)
    j1 = searchsortedfirst(p, pst)                       # first node with p ≥ p*
    mass = tails.CM[j1];  p1 = tails.CP[j1]
    if j1 > 1
        s  = j1 - 1                                      # the node straddling p*
        ω  = _soft_weight(p[s], pst, p, s, Np)
        mass += ω * wG[s];  p1 += ω * p[s] * wG[s]
    end
    num = (PUeff * p1 - (r + ν) * O * mass) / D
    B   = (λ / D) * mass
    return abs(1.0 - B) < 1e-14 ? 0.0 : num / (1.0 - B)
end

"""
    unskilled_match_at(O, PUeff, p0, p, wG, tails, r, ν, λ, D) → (pstar, S1)

The unskilled match of a worker with outside option `O`: its reservation quality, the
fixed point of `p* = [(r+ν) O − λ I(p*)] / (A P_U aU)` (`update_pstar_from_surplus!`'s map
at a given outside option), started from `p0`, and its entry surplus at p = 1 before the
positive part.  The map's slope at the fixed point is λ S(p*) g(p*) = 0, so it converges
in a few steps.
"""
function unskilled_match_at(O::Float64, PUeff::Float64, p0::Float64,
                            p::AbstractVector{Float64}, wG::AbstractVector{Float64}, tails,
                            r::Float64, ν::Float64, λ::Float64, D::Float64)
    PUeff <= 1e-14 && return (pstar = 1.0, S1 = (PUeff - (r + ν) * O) / D)
    pst = clamp01(p0)
    for _ in 1:200
        I    = unskilled_tail_integral(pst, O, PUeff, p, wG, tails, r, ν, λ, D)
        pnew = clamp01(((r + ν) * O - λ * I) / PUeff)
        done = abs(pnew - pst) <= 1e-15
        pst  = pnew
        done && break
    end
    I = unskilled_tail_integral(pst, O, PUeff, p, wG, tails, r, ν, λ, D)
    return (pstar = pst, S1 = (PUeff - (r + ν) * O + λ * I) / D)
end

"""
    hire_profile(SU1, model, D) → NamedTuple

The unskilled entry surplus S_U(aU,1) of a searching type along aU before the positive
part, from its positive part `SU1`: below the hiring threshold a^h a type is never hired,
so its continuation and tail are zero and S_U(aU,1) = (A P_U aU − b_U)/D, `D` the
discount of the surplus.  The profile runs linearly through the nodes and through its
exact zero at a^h, which is a knot.  Read by `crosser_hire_share`, in the stationary state
and at every date of the transition.
"""
function hire_profile(SU1::AbstractVector{Float64}, model::Model, D::Float64)
    cp = model.common;  up = model.unsk_par;  x = model.grids.x
    PU = exp(cp.A) * up.PU;  bU = up.bU * exp(cp.A)
    Srow = [SU1[i] > 0.0 ? SU1[i] : (PU * x[i] - bU) / D for i in eachindex(x)]
    aʰ = up.bU / up.PU
    knot_x = collect(x);  knot_S = copy(Srow)
    k = searchsortedfirst(knot_x, aʰ)
    if 1 < k <= length(x) && knot_x[k] != aʰ
        insert!(knot_x, k, aʰ);  insert!(knot_S, k, 0.0)
    end
    xlo, xhi = node_midpoint_intervals(x)
    return (Srow = Srow, χ = hire_fraction(model), knot_x = knot_x, knot_S = knot_S,
            x = x, xlo = xlo, xhi = xhi)
end

"""
    crosser_hire_share(hp, i, S1) → Float64

Hired share of cell `a_U` for a match whose entry surplus at the node is `S1` (before the
positive part).  Its surplus across the cell is taken as the searching type's profile
(`hire_profile`) shifted by `Δ = S_U(a_U,1) − S1`, so the share is the cell's fraction
above the ability at which the profile reaches `Δ`.  At `Δ = 0` that ability is `a^h`
and the share is `χ(a_U)` exactly.
"""
function crosser_hire_share(hp, i::Int, S1::Float64)
    Δ = hp.Srow[i] - S1
    Δ == 0.0 && return hp.χ[i]
    ks = hp.knot_x;  vs = hp.knot_S;  n = length(ks)
    ac = if Δ <= vs[1]
        -Inf
    elseif Δ > vs[n]
        Inf
    else
        k = 2
        while k < n && vs[k] < Δ
            k += 1
        end
        dv = vs[k] - vs[k - 1]
        dv > 0.0 ? ks[k - 1] + (Δ - vs[k - 1]) * (ks[k] - ks[k - 1]) / dv : ks[k]
    end
    return fraction_above(hp.xlo[i], hp.xhi[i], hp.x[i], ac)
end

"""
    crossing_context(model, κU, EU1) → NamedTuple

The objects the career choices of the trained read: the unskilled match of a type that
searches (`E_U(a_U,1)`, `p*_U`, its hire profile), the quadrature and constants the
training-slice matches are solved with, and the held choices (`Model.held`) if any.
"""
function crossing_context(model::Model, κU::Float64, EU1::AbstractVector{Float64})
    cp = model.common;  up = model.unsk_par;  sp = model.skl_par
    ug = model.unsk_grids;  uc = model.unsk_cache
    r = cp.r;  ν = cp.ν;  D = r + ν + up.λ + up.ξ
    wG = build_unskilled_G_weights(ug.p, ug.wp, up.α_U)
    hp = hire_profile(uc.Jfrontier ./ max(1.0 - up.β, 1e-14), model, D)
    return (κU = κU, EU1 = EU1, hp = hp, x = model.grids.x,
            p = ug.p, wG = wG, tails = unskilled_tails(ug.p, wG), PU = exp(cp.A) * up.PU,
            r = r, ν = ν, λ = up.λ, D = D, β = up.β, φ = cp.φ,
            bT = up.bT * exp(cp.A), bS = sp.bS * exp(cp.A), c = cp.c, held = model.held)
end

# The two node choices the values read, U_U = max{U^search, Utr} and
# U_S = max{U_S^(0), U_S^(1)}, at node (i, j): chosen when `held` is nothing, else held.
# Shared with the transition's backward pass.
@inline trains_at(held, i::Int, j::Int, Utr::Float64, Us::Float64) =
    held === nothing ? !(Utr <= Us) : held.trains[i, j]
@inline drains_at(held, i::Int, j::Int, U0::Float64, U1::Float64) =
    held === nothing ? !(U1 <= U0) : held.drains[i, j]
@inline skilled_value_at(held, i::Int, j::Int, U0::Float64, U1::Float64) =
    held === nothing ? max(U0, U1) : (held.drains[i, j] ? U1 : U0)

"""
    training_slice_match(ctx, model, i, j, Utr) → (pstar, S1, χ)

The unskilled match of a trained hire whose separations re-enter training: valued at the
outside option `Utr`, its reservation quality, entry surplus before the positive part and
hired share.
"""
function training_slice_match(ctx, model::Model, i::Int, j::Int, Utr::Float64)
    m = unskilled_match_at(Utr, ctx.PU * ctx.x[i], model.unsk_cache.pstar[i], ctx.p, ctx.wG,
                           ctx.tails, ctx.r, ctx.ν, ctx.λ, ctx.D)
    return (pstar = m.pstar, S1 = m.S1, χ = crosser_hire_share(ctx.hp, i, m.S1))
end

"""
    crossing_value(ctx, model, i, j, T) → Float64

U_S^(1) of type (a_U, a_S) when the value of training is `T`: the match she would enter is
the row's where searching beats training at the node (or is the held choice there), the
training slice's otherwise.
"""
function crossing_value(ctx, model::Model, i::Int, j::Int, T::Float64)
    uc  = model.unsk_cache
    Us  = uc.Usearch[i];  Utr = -training_cost(ctx.x[j], ctx.c) + T
    if !trains_at(ctx.held, i, j, Utr, Us)
        E = ctx.EU1[i];  χX = ctx.hp.χ[i]
    else
        m  = training_slice_match(ctx, model, i, j, Utr)
        E  = Utr + ctx.β * max(m.S1, 0.0);  χX = m.χ
    end
    κX = ctx.κU * χX
    return (ctx.bS + κX * E) / (ctx.r + ctx.ν + κX)
end

# The skilled unemployment value U_S(aU,aS) = max{U_S^(0)(aS), U_S^(1)(aU,aS)} at the
# installed state, or the held branch of each node; U_S^(0) read from the global link sc.U.
function skilled_unemployment_value(model::Model)
    U0 = transpose(model.skl_cache.U);  U1 = model.skl_cache.U1
    return model.held === nothing ? max.(U0, U1) : ifelse.(model.held.drains, U1, U0)
end

# Training value on U_S, the max of the two branches or the held one.
@inline training_value_on(ctx, i::Int, j::Int, U0::Float64, U1::Float64) =
    (ctx.bT + ctx.φ * skilled_value_at(ctx.held, i, j, U0, U1)) / (ctx.r + ctx.φ + ctx.ν)

"""
    crossing_fixed_point(ctx, model, i, j, U0) → (U1, T)

U_S^(1) and T of cell (i, j) solved jointly: at the training value of the stay-skilled
branch when crossing does not pay (or the held choice is to stay), otherwise at the fixed
point U_S^(1) = crossing_value(T(U_S^(1))), a contraction of modulus below φ/(r+φ+ν).
"""
function crossing_fixed_point(ctx, model::Model, i::Int, j::Int, U0::Float64)
    T  = training_value_on(ctx, i, j, U0, U0)
    U1 = crossing_value(ctx, model, i, j, T)
    drains_at(ctx.held, i, j, U0, U1) || return (U1 = U1, T = T)
    for _ in 1:2000
        T   = training_value_on(ctx, i, j, U0, U1)
        U1n = crossing_value(ctx, model, i, j, T)
        done = abs(U1n - U1) <= 1e-14 * max(abs(U1), 1.0)
        U1 = U1n
        done && break
    end
    return (U1 = U1, T = training_value_on(ctx, i, j, U0, U1))
end

"""
    career_choices!(model, κU, EU1)

The choices of the trained and of the untrained at the installed values: U_S^(1), T, the
training choice τ, the drain d, and the training-slice match (p*_t, S_t(·,1), χ_t) of every
cell that trains.  Called once the skilled block has settled U_S^(0); writes `sc.U1`,
`uc.T`, `uc.τT`, `sc.d`, `uc.pstar_t`, `uc.S1_t`, `uc.χ_t`.  A model that holds its choices
(`Model.held`) installs the held τ and d and values every node at its held branches.
"""
function career_choices!(model::Model, κU::Float64, EU1::AbstractVector{Float64})
    uc = model.unsk_cache;  sc = model.skl_cache;  x = model.grids.x
    Nx = length(x)
    ctx = crossing_context(model, κU, EU1)
    @threads for i in 1:Nx
        @inbounds for j in 1:Nx
            fp = crossing_fixed_point(ctx, model, i, j, sc.U0[j])
            sc.U1[i, j] = fp.U1;  uc.T[i, j] = fp.T
        end
    end
    Utr = training_value!(similar(uc.T), model)
    training_choice!(uc.τT, model, Utr)
    drain_choice!(sc.d, model)
    @threads for i in 1:Nx
        @inbounds for j in 1:Nx
            if uc.τT[i, j] > 0.0
                m = training_slice_match(ctx, model, i, j, Utr[i, j])
                uc.pstar_t[i, j] = m.pstar;  uc.S1_t[i, j] = m.S1;  uc.χ_t[i, j] = m.χ
            else
                uc.pstar_t[i, j] = 1.0;  uc.S1_t[i, j] = 0.0;  uc.χ_t[i, j] = 0.0
            end
        end
    end
    return nothing
end

"""
    drain_choice!(d, model) → d

The choice of market at the installed values, `drain_fraction!` of U_S^(1) against U_S^(0),
or the held d (`Model.held`).
"""
function drain_choice!(d::AbstractMatrix{Float64}, model::Model)
    model.held === nothing || return copyto!(d, model.held.d)
    sc = model.skl_cache;  Nx = length(model.grids.x)
    return drain_fraction!(d, sc.U1, repeat(transpose(sc.U0), Nx, 1), model.grids.x)
end

"""
    HeldChoices(model) → HeldChoices

The education choices of the solved `model`, for another model to hold (`Model.held`): its
τ and d, and the node choices behind them at its own values.  A model that holds its choices
returns those.
"""
function HeldChoices(model::Model)
    model.held === nothing || return model.held
    uc = model.unsk_cache;  sc = model.skl_cache;  Nx = length(model.grids.x)
    Utr = training_value!(similar(uc.T), model)
    return HeldChoices(copy(model.grids.x), copy(uc.τT), copy(sc.d),
                       [!(Utr[i, j] <= uc.Usearch[i]) for i in 1:Nx, j in 1:Nx],
                       [!(sc.U1[i, j] <= sc.U0[j]) for i in 1:Nx, j in 1:Nx])
end

"""
    implied_training_value(model, κU, EU1) → Matrix{Float64}

The training value the career choices imply at the installed state: U_S^(1) re-evaluated
at the installed T, and T on max{U_S^(0), U_S^(1)} (on the held branch where the model holds
its choices).  Equal to `uc.T` in equilibrium (R5).
"""
function implied_training_value(model::Model, κU::Float64, EU1::AbstractVector{Float64})
    uc = model.unsk_cache;  sc = model.skl_cache;  Nx = length(model.grids.x)
    ctx = crossing_context(model, κU, EU1)
    T = similar(uc.T)
    @threads for i in 1:Nx
        @inbounds for j in 1:Nx
            T[i, j] = training_value_on(ctx, i, j, sc.U0[j],
                                        crossing_value(ctx, model, i, j, uc.T[i, j]))
        end
    end
    return T
end

"""
    crossing_rates(model, d = sc.d) → (k, ks, kt)

Rate at which a draining trained worker of each cell is hired into the unskilled market,
`k = κ_U d [(1 − ω) χ + ω χ_t]`, split into the searching slice `ks` and the training slice
`kt` by the enclosed share ω (`enclosed_crossers`).  The never-hired share stays trained.
"""
function crossing_rates(model::Model, d::AbstractMatrix{Float64} = model.skl_cache.d)
    uc = model.unsk_cache;  up = model.unsk_par
    κU = meeting_rate(uc.θ, up.μ, up.η)
    χ  = hire_fraction(model)
    ks = zeros(Float64, size(d));  kt = zeros(Float64, size(d))
    @inbounds for j in axes(d, 2), i in axes(d, 1)
        dij = clamp(d[i, j], 0.0, 1.0)
        dij > 0.0 || continue
        ω = enclosed_crossers(dij, clamp(uc.τT[i, j], 0.0, 1.0))
        ks[i, j] = κU * dij * (1.0 - ω) * χ[i]
        kt[i, j] = κU * dij * ω * uc.χ_t[i, j]
    end
    return (k = ks .+ kt, ks = ks, kt = kt)
end

# The total crossing hire rate k alone, the denominator of m_S.
crossing_rate(model::Model, d::AbstractMatrix{Float64} = model.skl_cache.d) = crossing_rates(model, d).k


# ---------------------------------------------------------------------------
# Stationary composition on the copula grid
#
# Match rates depend only on aU, and the frontier τ(aU,aS) determines who
# trains, so the balance is solved per (aU,aS) cell with aU-indexed rates
# and the joint population weight ℓ(aU,aS) = W2[i,j] (notes §332).
#
# τ = 1 (train):   u_U = 0,  t = ν W2 / (φ + ν).
# τ = 0 (search):  u_U = W2 [χ (δ+ν)/(κ_U+δ+ν) + (1−χ)],  t = 0,
#                  δ = ξ_U + λ_U G(p*(aU)),  χ = hire_fraction.
# ---------------------------------------------------------------------------
function solve_stationary_unskilled!(
    u_out::Matrix{Float64}, t_out::Matrix{Float64},
    W2::Matrix{Float64}, τ::Matrix{Float64},
    pstar::AbstractVector{Float64}, κ::Float64, model::Model,
)
    ν = model.common.ν;  φ = model.common.φ
    Nx = length(model.grids.x)
    usurv = unemployment_split_unskilled(pstar, κ, model).usurv

    @inbounds for i in 1:Nx                       # aU index
        for j in 1:Nx                             # aS index
            # τ is the population fraction of the cell that trains, so the cell's weight
            # splits between the two branches rather than being assigned wholly to one.
            # Thresholding at 0.5 here would discard the fractional boundary node and put
            # the step function back, whatever the frontier does upstream.
            τij = clamp(τ[i, j], 0.0, 1.0)
            u_out[i, j] = W2[i, j] * usurv[i] * (1.0 - τij)
            t_out[i, j] = ν * W2[i, j] / (φ + ν) * τij
        end
    end
    return nothing
end


"""
    stationary_masses(model, κ) → (u, uh, t, et, mS)

Every stationary density on the copula grid at the installed policies: u_U and its hired
branch, t, the unskilled employment of the training slice and the trained mass m_S.  A
cell no trained worker leaves by an unskilled hire has the closed forms above and
m_S = φ t / ν.  A cell whose trained workers are hired into the unskilled market at rate
k = ks + kt (`crossing_rates`) balances, with δ = ξ_U + λ_U G(p*) on each slice's own p*,
  (φ + ν) t      = ν τ ℓ + δ_t e_t              separations of the training slice retrain
  (ν + k) m_S    = φ t
  (δ_t + ν) e_t  = kt m_S,
and u_U is (1 − τ) ℓ usurv plus δ_s ks m_S υʰ / (ν (δ_s + ν)), the searching slice's
separations, which join the row's hired unemployed; the searching slice's employment is
the cell's remainder.
"""
function stationary_masses(model::Model, κ::Float64)
    cp = model.common;  up = model.unsk_par;  uc = model.unsk_cache;  d = model.skl_cache.d
    W2 = model.grids.copula.W2;  ν = cp.ν;  φ = cp.φ
    Nx = length(model.grids.x)
    κU = meeting_rate(uc.θ, up.μ, up.η)
    u = similar(W2);  t = similar(W2)
    solve_stationary_unskilled!(u, t, W2, uc.τT, uc.pstar, κ, model)
    split = unemployment_split_unskilled(uc.pstar, κ, model)
    uh = similar(W2);  et = zeros(Float64, size(W2));  mS = similar(W2)
    @inbounds for j in 1:Nx, i in 1:Nx
        uh[i, j] = u[i, j] * split.hired[i]
        dij = clamp(d[i, j], 0.0, 1.0);  τij = clamp(uc.τT[i, j], 0.0, 1.0)
        # Only a cell that trains holds trained workers who can cross.
        k = 0.0
        if dij > 0.0 && τij > 0.0
            ω  = enclosed_crossers(dij, τij)
            ks = κU * dij * (1.0 - ω) * split.χ[i]
            kt = κU * dij * ω * uc.χ_t[i, j]
            k  = ks + kt
        end
        if k > 0.0
            δs = up.ξ + up.λ * G_cdf_unskilled(clamp01(uc.pstar[i]), up.α_U)
            δt = up.ξ + up.λ * G_cdf_unskilled(clamp01(uc.pstar_t[i, j]), up.α_U)
            t[i, j]  = ν * τij * W2[i, j] / (φ + ν - φ * δt * kt / ((δt + ν) * (ν + k)))
            mS[i, j] = φ * t[i, j] / (ν + k)
            et[i, j] = kt * mS[i, j] / (δt + ν)
            extra    = δs * ks * mS[i, j] / (ν * (κ + δs + ν))
            u[i, j]  += extra;  uh[i, j] += extra
        else
            mS[i, j] = φ * t[i, j] / ν
        end
    end
    return (u = u, uh = uh, t = t, et = et, mS = mS)
end

# The unskilled block's own densities at the installed policies; the links m_S and
# d u_S stay as installed.
function install_unskilled_masses!(model::Model, κ::Float64)
    uc = model.unsk_cache
    ms = stationary_masses(model, κ)
    copyto!(uc.u, ms.u);  copyto!(uc.uh, ms.uh);  copyto!(uc.t, ms.t);  copyto!(uc.et, ms.et)
    return ms
end

"""
    install_stationary_masses!(model, κ)

Install `stationary_masses` in the caches, with the cross-market seeker pool
duS_carry = d m_S: the draining trained are all unemployed, and all seek in the unskilled
market.
"""
function install_stationary_masses!(model::Model, κ::Float64)
    uc = model.unsk_cache;  sc = model.skl_cache
    sc.m_S = install_unskilled_masses!(model, κ).mS
    @inbounds for k in eachindex(sc.m_S)
        uc.duS_carry[k] = max(clamp(sc.d[k], 0.0, 1.0) * sc.m_S[k], 0.0)
    end
    return nothing
end


# ---------------------------------------------------------------------------
# Reservation-quality update
#
# Solves S_U(aU, p*) = 0 for p*(aU) (notes eq:pstarUformula):
#   p*(aU) = [(r+ν) U^search(aU) − λ I(aU)] / (A P_U aU),
# recomputing U^search against the proposed θ_new so the new p* is
# consistent with the tightness the outer loop installs.
# ---------------------------------------------------------------------------
function update_pstar_from_surplus!(
    pstar_new::AbstractVector{Float64}, model::Model, Ivec::AbstractVector{Float64};
    θ_new::Float64, E1::AbstractVector{Float64},
)
    cp = model.common;  up = model.unsk_par;  gp = model.grids
    r  = cp.r;  ν = cp.ν
    λ  = up.λ;  μ = up.μ;  η = up.η
    PU = exp(cp.A) * up.PU;  bU = up.bU * exp(cp.A)

    κ_new        = meeting_rate(θ_new, μ, η)
    denom_search = r + ν + κ_new

    @inbounds for i in eachindex(gp.x)
        aU          = gp.x[i]
        Usearch_new = (bU + κ_new * E1[i]) / denom_search
        PUeff       = PU * aU
        pstar_new[i] = (PUeff <= 1e-14) ? 1.0 :
            clamp01(((r + ν) * Usearch_new - λ * Ivec[i]) / PUeff)
    end
    return nothing
end


# ---------------------------------------------------------------------------
# Free entry: the tightness a vacancy cost implies, and its inverse
#
#   q_U(θ_U) = k_U π̄_U ũ_U / J̄_U,   ũ_U = ∬ (u_U + d u_S),
#   J̄_U = ∬ J_U(aU,1) [u_U^h + (1−ω) χ d u_S] + J_t(aU,aS,1) ω χ_t d u_S.
# A vacancy meets every seeker in ũ_U, never-hired types included, but a
# meeting with a never-hired type yields no match: its J_U is zero, exactly
# as at a node below a^h.  So the value of a meeting weights J_U by the
# hired branch u_U^h of u_U, and a draining trained seeker by her own hired
# share and match: the row's for the searching slice, the training slice's
# J_t = (1−β_U) S_t(·,1)⁺ for the share ω her cell's frontier encloses.  The
# dollar posting cost is k_U π̄_U (k_U dimensionless, π̄_U the mean output of a
# new unskilled match).  d·u_S is carried in uc.duS_carry.
# ---------------------------------------------------------------------------
"""
    unskilled_meeting_value(model; u_hired = nothing) → (Jbar, U_total)

The two sides of unskilled free entry at the installed composition: the firm value a
vacancy's meetings yield, J̄_U, and the seeker pool it meets, ũ_U = ∬ (u_U + d u_S).
`u_hired` is the hired branch of `uc.u` on the (aU,aS) grid; by default it is the
installed stationary one, `uc.uh` (`stationary_masses`).  The transition passes the stock
split of its own date.
"""
function unskilled_meeting_value(model::Model; u_hired::Union{Nothing,AbstractMatrix{Float64}} = nothing)
    uc = model.unsk_cache;  up = model.unsk_par
    κ   = meeting_rate(uc.θ, up.μ, up.η)
    u   = uc.u;  duS = uc.duS_carry;  uʰ = u_hired === nothing ? uc.uh : u_hired
    # At a converged corner with everyone training there are no seekers; a vacancy then
    # meets the limit composition of the labour force as it empties (see solve_model!).
    # That composition is stationary, so its hired branch is the stationary split.
    if model.corner_limit && sum(u .+ duS) <= EMPTY_MASS
        w = marginal_training_cells(model; entering = false)
        u = similar(w);  duS = zero(w)
        solve_stationary_unskilled!(u, similar(w), w, zero(w), uc.pstar, κ, model)
        uʰ = u .* unemployment_split_unskilled(uc.pstar, κ, model).hired
    end

    χ  = hire_fraction(model);  d = model.skl_cache.d
    U_total = sum(u .+ duS)
    Jbar    = 0.0
    Nx = size(u, 1)
    @inbounds for j in 1:Nx, i in 1:Nx
        ω = enclosed_crossers(clamp(d[i, j], 0.0, 1.0), clamp(uc.τT[i, j], 0.0, 1.0))
        Jbar += uc.Jfrontier[i] * (uʰ[i, j] + (1.0 - ω) * χ[i] * duS[i, j]) +
                (1.0 - up.β) * max(uc.S1_t[i, j], 0.0) * ω * uc.χ_t[i, j] * duS[i, j]
    end

    return (Jbar = Jbar, U_total = U_total)
end

_no_meeting_value(v) = v.Jbar < 1e-14 || v.U_total < 1e-14 || !isfinite(v.Jbar) || !isfinite(v.U_total)

"""
    update_theta_unskilled(model; u_hired = nothing) → θ_U

Free-entry tightness at the installed composition (`unskilled_meeting_value`).
"""
function update_theta_unskilled(model::Model; u_hired::Union{Nothing,AbstractMatrix{Float64}} = nothing)
    up = model.unsk_par
    v  = unskilled_meeting_value(model; u_hired = u_hired)
    _no_meeting_value(v) && return 1e-14
    q     = up.k * mean_output_U(model) * v.U_total / v.Jbar
    θ_raw = theta_from_q(q, up.μ, up.η)
    return clamp(θ_raw, 1e-14, 100.0)
end

"""
    vacancy_cost_unskilled(model) → k_U

The vacancy cost free entry implies at the installed tightness and composition, the
inverse of `update_theta_unskilled`: k_U = q_U(θ_U) J̄_U / (π̄_U ũ_U), in months of mean
new-match output.  Zero where a meeting is worth nothing.
"""
function vacancy_cost_unskilled(model::Model)
    uc = model.unsk_cache;  up = model.unsk_par
    v  = unskilled_meeting_value(model)
    _no_meeting_value(v) && return 0.0
    return q_from_theta(uc.θ, up.μ, up.η) * v.Jbar / (mean_output_U(model) * v.U_total)
end


# ---------------------------------------------------------------------------
# Outer loop on (θ_U, p*, u_U, t)
#
# Per iteration: inner values → stationary composition → raw (θ, p*) from
# free entry and S_U(aU,p*)=0 → joint Anderson(m=1) install (or damped
# Picard).  Convergence on (Δθ, Δp*, Δu_U).  Sustained inner divergence
# rejects the parameter; an outer stall hands back to the global loop.
#
# `θU_fixed` imposes the unskilled tightness instead of solving free entry for
# it: θ_U is held there on every pass and only (p*, u_U) iterate.
# ---------------------------------------------------------------------------
function solve_unskilled_block!(model::Model; US_in::AbstractMatrix{Float64},
                                θU_fixed::Union{Nothing,Float64} = nothing)
    gp = model.grids;  uc = model.unsk_cache;  sim = model.sim
    Nx = length(gp.x)
    θU_fixed === nothing || (uc.θ = θU_fixed)

    aa_joint = Anderson1(1 + Nx)
    s_θ = 1.0;  s_p = 1.0

    pstar_new = zeros(Float64, Nx)

    streak = 0;  diverge_streak = 0
    oB = sim.outer_B;  oK = sim.outer_K
    outer_resid = Vector{Float64}(undef, sim.maxit_outer)

    for it in 1:sim.maxit_outer
        θ_old     = uc.θ
        pstar_old = copy(uc.pstar)
        u_old     = copy(uc.u)

        inner = unskilled_inner_loop!(model; US_in = US_in)
        inner_conv = inner.converged

        if inner.status === :diverged
            diverge_streak += 1
            if diverge_streak >= sim.inner_K
                sim.verbose >= 1 && @printf("  [outer U]  inner diverged ×%d at it=%d — rejecting parameter\n", diverge_streak, it)
                return (converged = false, rejected = true)
            end
        else
            diverge_streak = 0
        end

        install_unskilled_masses!(model, inner.κ)

        # Under-relaxed free entry, as in the skilled block: the damped map is what both
        # the Anderson and the plain branch iterate on, so acceleration sees a contraction
        # the raw map is not.  The bracket is the one update_theta_unskilled itself
        # enforces, reapplied here because the blend can only land inside it.
        w_θ = DAMP_THETA_U[]
        θ_raw = θU_fixed !== nothing ? θU_fixed :
                clamp(w_θ * update_theta_unskilled(model) + (1.0 - w_θ) * θ_old, 1e-14, 100.0)
        θ_for_pstar = sim.use_anderson ? θ_old : (uc.θ = θ_raw)
        update_pstar_from_surplus!(pstar_new, model, inner.Ivec; θ_new = θ_for_pstar, E1 = inner.E1vec)

        if sim.use_anderson
            if it == 1
                s_θ = max(abs(θ_raw), abs(θ_old), 1.0)
                s_p = max(maximum(abs, pstar_new), 1.0)
            end
            x_old = vcat([θ_old / s_θ], pstar_old ./ s_p)
            f_raw = vcat([θ_raw / s_θ], pstar_new ./ s_p)
            x_new = anderson1_update!(aa_joint, x_old, f_raw)
            # Same bracket as the free-entry solve: the Anderson step extrapolates, so
            # without it the installed θ_U left the range its own solve is clamped to —
            # measured at 107 and 182 against a ceiling of 100.
            uc.θ  = θU_fixed !== nothing ? θU_fixed : clamp(x_new[1] * s_θ, 1e-14, 100.0)
            @inbounds for i in 1:Nx
                uc.pstar[i] = clamp01(x_new[1 + i] * s_p)
            end
        else
            @inbounds for i in 1:Nx
                uc.pstar[i] = clamp01(sim.damp_pstar_U * pstar_new[i] + (1.0 - sim.damp_pstar_U) * pstar_old[i])
            end
        end

        if !isfinite(uc.θ) || any(!isfinite, uc.pstar)
            sim.verbose >= 1 && @printf("  [outer U]  NaN/Inf in state at it=%d — aborting\n", it)
            return (converged = false, rejected = false)
        end

        dθ = abs(uc.θ - θ_old)
        dp = supnorm(uc.pstar, pstar_old)
        du = supnorm(uc.u, u_old)
        d  = max(dθ, dp, du)
        outer_resid[it] = d

        if sim.verbose >= 2 && (it == 1 || it % sim.verbose_stride == 0)
            @printf("  [outer U it=%d]  maxΔ=%.3e  (Δθ=%.3e  Δp*=%.3e  Δu=%.3e)  θ_U=%.4f\n", it, d, dθ, dp, du, uc.θ)
        end

        if d < sim.tol_outer_U
            streak += 1
            if streak >= sim.conv_streak
                sim.verbose >= 2 && @printf("  [outer U]  converged it=%d  d=%.3e  θ_U=%.4f\n", it, d, uc.θ)
                inner_final = unskilled_inner_loop!(model; US_in = US_in)
                install_unskilled_masses!(model, inner_final.κ)
                return (converged = inner_conv, rejected = false)
            end
        else
            streak = 0
        end

        if oB > 0 && it > oB + oK && outer_resid[it] >= outer_resid[it - oK]
            sim.verbose >= 1 && @printf("  [outer U]  stalled at it=%d — handing back to global\n", it)
            inner_final = unskilled_inner_loop!(model; US_in = US_in)
            install_unskilled_masses!(model, inner_final.κ)
            return (converged = false, rejected = false)
        end
    end

    inner_final = unskilled_inner_loop!(model; US_in = US_in)
    install_unskilled_masses!(model, inner_final.κ)
    sim.verbose >= 1 && @printf("  [outer U]  maxit reached without convergence  θ_U=%.4f\n", uc.θ)
    return (converged = false, rejected = false)
end

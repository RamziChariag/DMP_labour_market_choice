############################################################
# unskilled.jl — Unskilled block solver (RoySearch)
#
# Under pure-Roy every unskilled-market VALUE object is a function of the
# unskilled aptitude aU alone (notes §231, §276): U^search, E_U, J_U, p*
# read aU; the training value T reads the skilled aptitude aS.  The
# two-dimensional type enters only through
#   (i)  the training frontier τ(aU,aS) = 1{−c(aS)+T(aS) ≥ U^search(aU)},
#   (ii) the augmented seeker pool ũ_U = ∬ (u_U + d u_S), and
#   (iii)the stationary densities u_U(aU,aS), t(aU,aS) on the copula grid.
#
# The training decision is NOT gated by d.  b_T discipline comes purely
# from comparative advantage (notes Prop bT): the would-be stipend
# farmers are high-aU / low-aS types who forgo a good unskilled job by
# training, so raising b_T shifts only the marginal genuine trainee at
# the frontier.  (The single-ability code gated τ by d as an explicit
# refinement; RoySearch does not need it and does not use it.)
#
# Outer loop iterates on (θ_U, p*(aU), u_U, t); the inner loop settles
# (U^search, T, E_U(·,1), J_U(·,1)) at fixed (θ_U, p*).  The cross-market
# pool d·u_S is supplied by the skilled block via uc.duS_carry and held
# fixed within one unskilled solve.
#
# Functions
#   build_unskilled_G_weights        dG(p) quadrature weights, G=Beta(α_U,1)
#   G_cdf_unskilled                  G(p) = p^{α_U}
#   solve_unskilled_surplus_on_grid! surplus S_U(aU,·) on the p-grid
#   entry_surplus_unskilled          surplus S_U(aU,1) of a fresh match
#   unskilled_inner_loop!            iterate values at (θ_U, p*)
#   hire_fraction                    hired fraction χ(aU) of each ability cell
#   unemployment_split_unskilled     per-row unemployed share and its hired part
#   hired_unemployed                 hired branch of a cell's unemployed stock
#   solve_stationary_unskilled!      stationary u_U, t on the copula grid
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
# Inner loop
#
# Given θ_U and p*(aU) fixed, iterate on (T, U^search, E_U(·,1), J_U(·,1))
# until convergence.  U_S is supplied from the global loop.
#
# At p = 1, Nash bargaining gives J_U(aU,1) = (1−β) S_U(aU,1),
# E_U(aU,1) = U^search(aU) + β S_U(aU,1); the search-value update consumes
# E_U(aU,1) via (r+ν+κ_U) U^search = b_U + κ_U E_U(aU,1).  The frontier τ
# is formed here (2D) for the distribution step but does not feed the
# unskilled outside option, which is U^search(aU) alone (notes §276).
# ---------------------------------------------------------------------------
"""
    training_frontier!(T, US_in, bT, φ, r, ν)

Training value `(r + φ + ν) T(a_S) = b_T + φ U_S(a_S)`.  Extracted so the iteration and
the equilibrium residual that checks it share one implementation: a residual computed from
a second copy of a formula tests the copy, not the solver.
"""
function training_frontier!(T::AbstractVector{Float64}, US_in::AbstractVector{Float64},
                            bT::Float64, φ::Float64, r::Float64, ν::Float64)
    d = r + φ + ν
    @inbounds @threads for ix in eachindex(T)
        T[ix] = (bT + φ * US_in[ix]) / d
    end
    return T
end

"""
    training_policy!(τT, Utr, Usearch, x)

Training choice `τ(a_U, a_S)`: the covered fraction of ability cell `a_S` whose training
value clears the outside option `U^search(a_U)`.  A cell is partly covered because the
frontier generally falls inside it; the hard indicator returns only where `Utr` is
non-monotone in `a_S`, since a crossing located by interpolation means nothing there.

Shared by the iteration and by the equilibrium residual, for the reason above.
"""
function training_policy!(τT::AbstractMatrix{Float64}, Utr::AbstractVector{Float64},
                          Usearch::AbstractVector{Float64}, x::AbstractVector{Float64})
    Nx = length(x)
    xlo, xhi = node_midpoint_intervals(x)

    monotone = true
    @inbounds for j in 2:Nx
        if Utr[j] < Utr[j - 1]
            monotone = false
            break
        end
    end

    @threads for i in 1:Nx
        @inbounds begin
            us = Usearch[i]
            if !monotone
                for j in 1:Nx
                    τT[i, j] = (Utr[j] >= us) ? 1.0 : 0.0
                end
            elseif Utr[Nx] < us                                 # nobody at this aU trains
                for j in 1:Nx
                    τT[i, j] = 0.0
                end
            elseif Utr[1] >= us                                 # everybody at this aU trains
                for j in 1:Nx
                    τT[i, j] = 1.0
                end
            else
                j0 = 2
                while j0 < Nx && Utr[j0] < us
                    j0 += 1
                end
                dU = Utr[j0] - Utr[j0 - 1]
                aS = dU > 0.0 ?
                     x[j0 - 1] + (us - Utr[j0 - 1]) * (x[j0] - x[j0 - 1]) / dU :
                     x[j0]
                for j in 1:Nx
                    τT[i, j] = fraction_above(xlo[j], xhi[j], x[j], aS)
                end
            end
        end
    end
    return τT
end

"""
    training_value!(Utr, model) → Utr

Value of entering training from ability `a_S`, net of its cost: `T(a_S) − c(a_S)`.  The
one expression every reader of the training margin compares against `U^search(a_U)`.
"""
function training_value!(Utr::AbstractVector{Float64}, model::Model)
    cp = model.common;  gp = model.grids;  uc = model.unsk_cache
    @inbounds for j in eachindex(Utr)
        Utr[j] = -training_cost(gp.x[j], cp.c) + uc.T[j]
    end
    return Utr
end

"""
    marginal_training_cells(model; entering, admit = nothing) → Matrix{Float64}

Composition of a market that holds no one, as the limit of its composition while it
empties.  Markets empty through the training margin, and the last cells to cross it are
the first to cross back: for the skilled market (`entering = true`) the cells with the
largest training gain `Utr(a_S) − U^search(a_U)`, for the unskilled labour force
(`entering = false`) those with the smallest.  Returns a unit mass on those cells, for
the model's own stationary formulas to turn into a composition — unit so that its size
never depends on how thin the copula is at the marginal cells, and only its ratios carry
meaning.  Cells tied to a relative 1e-12 share it in proportion to `W2`.  `admit`
restricts the candidate cells; with none admitted the result is zero.
"""
function marginal_training_cells(model::Model; entering::Bool,
                                 admit::Union{Nothing,AbstractMatrix{Bool}} = nothing)
    uc  = model.unsk_cache;  W2 = model.grids.copula.W2;  Nx = length(uc.Usearch)
    Utr = training_value!(similar(uc.Usearch), model)
    gain(i, j) = entering ? Utr[j] - uc.Usearch[i] : uc.Usearch[i] - Utr[j]
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


function unskilled_inner_loop!(model::Model; US_in::AbstractVector{Float64})
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
    T_old       = copy(uc.T)

    αU     = sim.damp_inner_U
    B      = sim.inner_B;  K = sim.inner_K
    resid  = Vector{Float64}(undef, sim.maxit_inner)
    status = :maxit;  n_inner = 0;  streak = 0

    for it in 1:sim.maxit_inner
        copyto!(Usearch_old, uc.Usearch)
        copyto!(T_old,       uc.T)

        training_frontier!(uc.T, US_in, bT, φ, r, ν)

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

        d = max(supnorm(uc.Usearch, Usearch_old), supnorm(uc.T, T_old))
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
    # from Utr(aS) = −c(aS) + T(aS) ≥ U^search(aU).
    # (2D; no d-gate — b_T discipline is by comparative advantage).
    # c(aS) is a psychological cost in fixed utility units — NOT scaled by A —
    # so the training margin shifts with the wage level rather than being scale-invariant.
    #
    # Utr depends only on aS and U^search only on aU, and Utr is increasing in aS, so for
    # each aU the training set is an upper interval in aS with a single crossing aS*(aU)
    # solving Utr(aS*) = U^search(aU).  A hard 0/1 at the nodes made every composition
    # aggregate a STEP FUNCTION of θ: a change in θ too small to move aS* past a node
    # flipped no cell, so Σ τ·W2 was bit-identical and ∂m/∂θ was exactly zero for every
    # moment.  Measured at v22.0.0: c had to move 0.026% of its value before one of 14400
    # cells flipped, and Q then jumped by 0.18 — one cell is 0.18 sampling standard errors.
    # τ is now the fraction of node j's midpoint interval lying above aS*, which is
    # continuous and piecewise-linear in aS* and therefore in θ, and reduces to the old
    # indicator whenever aS* sits on a node.
    Utr = training_value!(similar(uc.Usearch), model)
    training_policy!(uc.τT, Utr, uc.Usearch, gp.x)

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
#   q_U(θ_U) = k_U π̄_U ũ_U / ∬ J_U(aU,1) (u_U^h + d u_S) daU daS,
#   ũ_U = ∬ (u_U + d u_S).
# A vacancy meets every seeker in ũ_U, never-hired types included, but a
# meeting with the never-hired branch of u_U yields no match: its J_U is
# zero, exactly as at a node below a^h.  So the value of a meeting weights
# J_U by the hired branch u_U^h only.  The dollar posting cost is k_U π̄_U
# (k_U dimensionless, π̄_U the mean output of a new unskilled match).  d·u_S is
# carried in uc.duS_carry.  J_U(aU,1) is aU-indexed and broadcast across aS.
# ---------------------------------------------------------------------------
"""
    unskilled_meeting_value(model; u_hired = nothing) → (Jbar, U_total)

The two sides of unskilled free entry at the installed composition: the firm value a
vacancy's meetings yield, ∬ J_U(aU,1) (u_U^h + d u_S), and the seeker pool it meets,
ũ_U = ∬ (u_U + d u_S).  `u_hired` is the hired branch of `uc.u` on the (aU,aS) grid; by
default it is the stationary split `hired · u_U` (`unemployment_split_unskilled`, at the
installed p* and θ_U, the values `uc.u` was built from).  The transition passes the stock
split of its own date.
"""
function unskilled_meeting_value(model::Model; u_hired::Union{Nothing,AbstractMatrix{Float64}} = nothing)
    uc = model.unsk_cache;  up = model.unsk_par
    κ   = meeting_rate(uc.θ, up.μ, up.η)
    u   = uc.u;  duS = uc.duS_carry
    # At a converged corner with everyone training there are no seekers; a vacancy then
    # meets the limit composition of the labour force as it empties (see solve_model!).
    # That composition is stationary, so its hired branch is the stationary split.
    if model.corner_limit && sum(u .+ duS) <= EMPTY_MASS
        w = marginal_training_cells(model; entering = false)
        u = similar(w);  duS = zero(w);  u_hired = nothing
        solve_stationary_unskilled!(u, similar(w), w, zero(w), uc.pstar, κ, model)
    end
    uʰ = u_hired === nothing ? u .* unemployment_split_unskilled(uc.pstar, κ, model).hired : u_hired

    U_total = sum(u .+ duS)
    Jbar    = 0.0
    Nx = size(u, 1)
    @inbounds for j in 1:Nx, i in 1:Nx
        Jbar += uc.Jfrontier[i] * (uʰ[i, j] + duS[i, j])    # J_U reads aU = row i
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
function solve_unskilled_block!(model::Model; US_in::AbstractVector{Float64},
                                θU_fixed::Union{Nothing,Float64} = nothing)
    gp = model.grids;  uc = model.unsk_cache;  sim = model.sim
    Nx = length(gp.x)
    θU_fixed === nothing || (uc.θ = θU_fixed)

    aa_joint = Anderson1(1 + Nx)
    s_θ = 1.0;  s_p = 1.0

    u_new = zeros(Float64, Nx, Nx)
    t_new = zeros(Float64, Nx, Nx)
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

        solve_stationary_unskilled!(u_new, t_new, gp.copula.W2, uc.τT, uc.pstar, inner.κ, model)
        copyto!(uc.u, u_new);  copyto!(uc.t, t_new)

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
                solve_stationary_unskilled!(u_new, t_new, gp.copula.W2, uc.τT, uc.pstar, inner_final.κ, model)
                copyto!(uc.u, u_new);  copyto!(uc.t, t_new)
                return (converged = inner_conv, rejected = false)
            end
        else
            streak = 0
        end

        if oB > 0 && it > oB + oK && outer_resid[it] >= outer_resid[it - oK]
            sim.verbose >= 1 && @printf("  [outer U]  stalled at it=%d — handing back to global\n", it)
            inner_final = unskilled_inner_loop!(model; US_in = US_in)
            solve_stationary_unskilled!(u_new, t_new, gp.copula.W2, uc.τT, uc.pstar, inner_final.κ, model)
            copyto!(uc.u, u_new);  copyto!(uc.t, t_new)
            return (converged = false, rejected = false)
        end
    end

    inner_final = unskilled_inner_loop!(model; US_in = US_in)
    solve_stationary_unskilled!(u_new, t_new, gp.copula.W2, uc.τT, uc.pstar, inner_final.κ, model)
    copyto!(uc.u, u_new);  copyto!(uc.t, t_new)
    sim.verbose >= 1 && @printf("  [outer U]  maxit reached without convergence  θ_U=%.4f\n", uc.θ)
    return (converged = false, rejected = false)
end

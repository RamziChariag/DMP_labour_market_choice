############################################################
# unskilled.jl — Unskilled block, training margin and career choices
#
# The values of a searching type read aU alone (U^search, E_U, J_U, p*_U). The full
# type (aU, aS) enters through the training value T(aU,aS) = [b_T + φ U_S]/(r+φ+ν) and the
# training choice τ, through the trained workers hired into the unskilled market (the
# career choices), through the seeker pool ũ_U = ∬ (u_U + d u_S), and through the
# stationary densities on the copula grid. Training is not gated by d.
#
# The outer loop iterates on (θ_U, p*_U, u_U, t); the inner loop settles
# (U^search, E_U(·,1), J_U(·,1)) at fixed (θ_U, p*_U). d·u_S is carried from the skilled
# block in uc.duS_carry and held fixed within one unskilled solve.
#
# Entry points: solve_unskilled_block!, career_choices!, stationary_masses,
# install_stationary_masses!, update_theta_unskilled, vacancy_cost_unskilled,
# HeldChoices(model).
############################################################


# ---------------------------------------------------------------------------
# Damage-shock distribution G = Beta(α_U, 1), G(p) = p^α_U
# ---------------------------------------------------------------------------

"""
    build_unskilled_G_weights(pgrid, wp, α) → wG

Quadrature weights of dG on the unskilled grid, `wp_j g(p_j)` with `g(p) = α p^(α−1)`.
"""
function build_unskilled_G_weights(pgrid::Vector{Float64}, wp::Vector{Float64}, α::Float64)
    wG = similar(wp)
    @inbounds for j in eachindex(pgrid)
        p     = pgrid[j]
        g     = (p <= 0.0) ? 0.0 : α * p^(α - 1.0)
        wG[j] = wp[j] * g
    end
    return wG
end

"""
    G_cdf_unskilled(p, α) → G(p)

Damage-shock CDF `p^α` on [0, 1].
"""
@inline function G_cdf_unskilled(p::Float64, α::Float64)
    (p <= 0.0) && return 0.0
    (p >= 1.0) && return 1.0
    return p^α
end


# ---------------------------------------------------------------------------
# Surplus
# ---------------------------------------------------------------------------

"""
    solve_unskilled_surplus_on_grid!(Svec, pgrid, wG, PUeff, r, ν, λ, Ux, pstar_x, ξ = 0) → I

Surplus `S_U(aU, ·)` on the grid for one aU, from the closed form of the surplus HJB
`(r+ν+λ+ξ) S(p) = PUeff p − (r+ν) Ux + λ I`, with the tail integral `I = ∫_{p*}^1 S dG`
solved analytically and the soft weight at `pstar_x`. `PUeff = exp(A) P_U aU`, `Ux` the
outside option U^search(aU). ξ_U enters only the discount. Writes `Svec`; returns `I`.
"""
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

Surplus of a fresh unskilled match, which starts at p = 1: `max((PUeff + flow + λ I) / D, 0)`,
`flow` the p-free rest of the numerator (`−(r+ν) U^search` in a steady state), `I` the tail
integral, `D` the discount. Evaluated in closed form at p = 1, above the top grid node, so it
is continuous in aU and vanishes at a^h.
"""
@inline entry_surplus_unskilled(PUeff::Float64, flow::Float64, λ::Float64, I::Float64, D::Float64) =
    max((PUeff + flow + λ * I) / D, 0.0)


# ---------------------------------------------------------------------------
# Training margin
# ---------------------------------------------------------------------------

"""
    training_frontier!(T, US, bT, φ, r, ν) → T

Training value `T = (b_T + φ U_S) / (r + φ + ν)`, cell by cell, on the skilled
unemployment value `US`. Shared by the choices and the residual battery.
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
    training_policy!(τT, Utr, C, x) → τT

Training choice τ(aU, aS): along each aU row, the covered fraction of each cell on which the
training option `Utr` beats the value of searching `C` (`covered_fraction!`). Shared by the
iteration, the residual battery and the transition.
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

Value of entering training net of its cost, `T(aU, aS) − c(aS)`, at the installed T.
"""
function training_value!(Utr::AbstractMatrix{Float64}, model::Model)
    cp = model.common;  gp = model.grids;  uc = model.unsk_cache
    @inbounds for j in axes(Utr, 2), i in axes(Utr, 1)
        Utr[i, j] = -training_cost(gp.x[j], cp.c) + uc.T[i, j]
    end
    return Utr
end

"""
    searching_value(model) → Matrix{Float64}

U^search(aU) on the (aU, aS) grid, constant along each row.
"""
searching_value(model::Model) = repeat(model.unsk_cache.Usearch, 1, length(model.grids.x))

"""
    training_choice!(τT, model, Utr = T − c) → τT

The training choice at the installed values (`training_policy!` of `Utr` against the value
of searching), or the held τ when `model.held` is set. Every reader of the choice calls this.
"""
function training_choice!(τT::AbstractMatrix{Float64}, model::Model,
                          Utr::AbstractMatrix{Float64} = training_value!(similar(model.unsk_cache.T), model))
    model.held === nothing || return copyto!(τT, model.held.τ)
    return training_policy!(τT, Utr, searching_value(model), model.grids.x)
end

"""
    marginal_training_cells(model; entering, admit = nothing) → Matrix{Float64}

Limit composition of a market that holds no one, as a unit mass on the (aU, aS) grid. The
last cells to leave through the training margin are the first to return: for the skilled
market (`entering = true`) the cells with the largest training gain `Utr − U^search`, for
the unskilled labour force (`entering = false`) the smallest. Cells tied to a relative 1e-12
share the mass in proportion to `W2`. `admit` restricts the candidates; with none admitted
the result is zero.
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
# ---------------------------------------------------------------------------

"""
    unskilled_inner_loop!(model; US_in) → (κ, Ivec, E1vec, status, converged, iters)

Iterate (U^search, J_U(·,1), E_U(·,1)) at fixed θ_U and p*_U:
`(r+ν+κ_U) U^search = b_U + κ_U E_U(aU,1)`, with Nash sharing of the entry surplus at p = 1.
T is set once from the skilled unemployment value `US_in` (it does not feed U^search), and its
step enters the first pass's residual. On exit installs the training choice τ. `status` is
`:converged`, `:diverged` (no contraction over `inner_K` sweeps after `inner_B`) or `:maxit`.
"""
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

    # Partition fixed per call, for bitwise reproducibility across schedules (index_chunks).
    chunks = index_chunks(Nx)

    Usearch_old = copy(uc.Usearch)

    T_old = copy(uc.T)
    training_frontier!(uc.T, US_in, bT, φ, r, ν)
    dT = supnorm(uc.T, T_old)

    αU     = sim.damp_inner_U
    B      = sim.inner_B;  K = sim.inner_K
    resid  = Vector{Float64}(undef, sim.maxit_inner)
    status = :maxit;  n_inner = 0;  streak = 0

    for it in 1:sim.maxit_inner
        copyto!(Usearch_old, uc.Usearch)

        # Surplus at the current U^search → J_U(aU,1), E_U(aU,1); scratch per task.
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

    training_choice!(uc.τT, model)

    return (κ = κ, Ivec = Ivec, E1vec = E1vec,
            status = status, converged = (status === :converged), iters = n_inner)
end


# ---------------------------------------------------------------------------
# Hiring margin
#
# A fresh match starts at p = 1, so a type is hired iff J_U(aU,1) > 0, i.e. iff
# aU > a^h = b_U / P_U at every tightness. Never-hired types search, are unemployed and
# stay in the seeker pool; they leave only at ν.
# ---------------------------------------------------------------------------

"""
    hire_fraction(model) → Vector{Float64}

Hired fraction χ(aU) of each ability cell: the part of the cell above the hiring threshold
`a^h = b_U / P_U` (`fraction_above`). Continuous in a^h, equal to the node indicator when
a^h is on a cell edge.
"""
function hire_fraction(model::Model)
    x  = model.grids.x
    aʰ = model.unsk_par.bU / model.unsk_par.PU
    xlo, xhi = node_midpoint_intervals(x)
    return [fraction_above(xlo[i], xhi[i], x[i], aʰ) for i in eachindex(x)]
end

"""
    unemployment_split_unskilled(pstar, κ, model) → (χ, usurv, hired)

Per aU row, the stationary unemployment of its searching population: `usurv = χ υʰ + (1−χ)`
the unemployed share, with `υʰ = (δ+ν)/(κ+δ+ν)`, `δ = ξ_U + λ_U G(p*)`, and the never-hired
branch wholly unemployed; `hired = χ υʰ / usurv` the hired branch's share of the row's
unemployed. υʰ uses the node's own p* and κ whatever its classification.
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

Hired branch of a cell's unemployed stock `u`: `u` less the never-hired part of the
searching slice, `(1 − χ)(1 − τ)ℓ`. Equals `hired · u` in a stationary state; the transition
uses it off one.
"""
@inline hired_unemployed(u::Float64, χ::Float64, τ::Float64, ℓ::Float64) =
    max(u - (1.0 - χ) * (1.0 - τ) * ℓ, 0.0)


# ---------------------------------------------------------------------------
# Career choices of the trained
#
# A trained worker hired into the unskilled market becomes an unskilled worker: when the
# match ends her outside option is U_U = max{U^search(aU), Utr}, Utr = −c(aS) + T(aU,aS).
# Outside the training set her match is the row's. Inside it her separations re-enter
# training, so the match is valued at Utr (the training slice: p*_t, S_t(·,1), χ_t). The
# value of crossing is (r+ν+κ_X) U_S^(1) = b_S + κ_X E_U(aU,aS,1), κ_X = κ_U χ_X. T prices
# U_S^(1) and U_S^(1) reads T, so a cell inside the training set solves a scalar fixed point.
# ---------------------------------------------------------------------------

"""
    enclosed_crossers(d, τ) → Float64

Share of a cell's drain fraction `d` inside its training fraction `τ`, `max(d+τ−1, 0)/d`:
the drain set is the lower part of the cell in aS, the training set the upper part.
"""
@inline enclosed_crossers(d::Float64, τ::Float64) = d > 0.0 ? max(d + τ - 1.0, 0.0) / d : 0.0

"""
    unskilled_tails(p, wG) → (CM, CP)

Suffix sums `Σ_{k≥j} wG_k` and `Σ_{k≥j} p_k wG_k` (length Np+1, zero at the end).
"""
function unskilled_tails(p::AbstractVector{Float64}, wG::AbstractVector{Float64})
    Np = length(p)
    CM = zeros(Float64, Np + 1);  CP = zeros(Float64, Np + 1)
    @inbounds for j in Np:-1:1
        CM[j] = CM[j + 1] + wG[j]
        CP[j] = CP[j + 1] + p[j] * wG[j]
    end
    return (CM = CM, CP = CP)
end

"""
    unskilled_tail_integral(pst, O, PUeff, p, wG, tails, r, ν, λ, D) → I

Tail integral `∫_{p*}^1 S dG` of the match with outside option `O` at reservation quality
`pst`, as in `solve_unskilled_surplus_on_grid!` (soft weight on the straddling node), from
the suffix sums `tails`.
"""
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

Unskilled match of a worker with outside option `O`: the reservation quality, fixed point of
`p* = [(r+ν) O − λ I(p*)] / PUeff` started at `p0`, and the entry surplus at p = 1 before the
positive part. The map has zero slope at the fixed point, so it converges in a few steps.
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
    hire_profile(SU1, model, D) → (Srow, χ, knot_x, knot_S, x, xlo, xhi)

Entry surplus S_U(aU,1) of a searching type along aU before the positive part, from its
positive part `SU1`: below a^h it is `(exp(A) P_U aU − b_U)/D`. The profile is linear between
the nodes and has an exact zero knot at a^h. Read by `crosser_hire_share`, here and in the
transition.
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

Hired share of cell `i` for a match whose entry surplus at the node is `S1` (before the
positive part): the searching profile `hp` shifted by `Δ = S_U(aU,1) − S1`, so the share is
the cell's fraction above the ability where the profile reaches Δ. At Δ = 0 it is χ(aU).
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

Everything the career choices read: κ_U and E_U(aU,1) of the searching match, its hire
profile, the dG quadrature and suffix sums, the scaled flows and rates, and `model.held`.
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

"""
    trains_at(held, i, j, Utr, Us) → Bool

Node choice in U_U = max{U^search, Utr} at (i, j): chosen when `held === nothing`, else held.
Shared with the transition.
"""
@inline trains_at(held, i::Int, j::Int, Utr::Float64, Us::Float64) =
    held === nothing ? !(Utr <= Us) : held.trains[i, j]

"""
    drains_at(held, i, j, U0, U1) → Bool

Node choice in U_S = max{U_S^(0), U_S^(1)} at (i, j): chosen when `held === nothing`, else
held. Shared with the transition.
"""
@inline drains_at(held, i::Int, j::Int, U0::Float64, U1::Float64) =
    held === nothing ? !(U1 <= U0) : held.drains[i, j]

"""
    skilled_value_at(held, i, j, U0, U1) → Float64

U_S at (i, j): `max(U0, U1)`, or the held branch. Shared with the transition.
"""
@inline skilled_value_at(held, i::Int, j::Int, U0::Float64, U1::Float64) =
    held === nothing ? max(U0, U1) : (held.drains[i, j] ? U1 : U0)

"""
    training_slice_match(ctx, model, i, j, Utr) → (pstar, S1, χ)

Unskilled match of a trained hire whose separations re-enter training, valued at the outside
option `Utr`: reservation quality, entry surplus before the positive part, hired share.
"""
function training_slice_match(ctx, model::Model, i::Int, j::Int, Utr::Float64)
    m = unskilled_match_at(Utr, ctx.PU * ctx.x[i], model.unsk_cache.pstar[i], ctx.p, ctx.wG,
                           ctx.tails, ctx.r, ctx.ν, ctx.λ, ctx.D)
    return (pstar = m.pstar, S1 = m.S1, χ = crosser_hire_share(ctx.hp, i, m.S1))
end

"""
    crossing_value(ctx, model, i, j, T) → U_S^(1)

U_S^(1) of cell (i, j) at training value `T`: through the row's match where searching beats
training at the node (or is the held choice), through the training-slice match otherwise.
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

"""
    skilled_unemployment_value(model) → Matrix{Float64}

U_S(aU, aS) = max{U_S^(0)(aS), U_S^(1)(aU,aS)} at the installed state, or the held branch;
U_S^(0) is read from the global link `sc.U`.
"""
function skilled_unemployment_value(model::Model)
    U0 = transpose(model.skl_cache.U);  U1 = model.skl_cache.U1
    return model.held === nothing ? max.(U0, U1) : ifelse.(model.held.drains, U1, U0)
end

"""
    training_value_on(ctx, i, j, U0, U1) → T

Training value on U_S, the max of the two branches or the held one.
"""
@inline training_value_on(ctx, i::Int, j::Int, U0::Float64, U1::Float64) =
    (ctx.bT + ctx.φ * skilled_value_at(ctx.held, i, j, U0, U1)) / (ctx.r + ctx.φ + ctx.ν)

"""
    crossing_fixed_point(ctx, model, i, j, U0) → (U1, T)

U_S^(1) and T of cell (i, j) jointly: at the stay-skilled training value when crossing does
not pay (or the held choice is to stay), else the fixed point U_S^(1) = crossing_value(T(U_S^(1))),
a contraction of modulus below φ/(r+φ+ν).
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

Choices of the trained and untrained at the installed values, once the skilled block has
settled U_S^(0): writes `sc.U1`, `uc.T`, `uc.τT`, `sc.d` and the training-slice match
`uc.pstar_t`, `uc.S1_t`, `uc.χ_t` of every cell that trains. With `model.held` set, installs
the held τ and d and values every node at its held branch.
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

The choice of market at the installed values (`drain_fraction!` of U_S^(1) against
U_S^(0)), or the held d.
"""
function drain_choice!(d::AbstractMatrix{Float64}, model::Model)
    model.held === nothing || return copyto!(d, model.held.d)
    sc = model.skl_cache;  Nx = length(model.grids.x)
    return drain_fraction!(d, sc.U1, repeat(transpose(sc.U0), Nx, 1), model.grids.x)
end

"""
    HeldChoices(model) → HeldChoices

Education choices of the solved `model`, for another model to hold: τ, d and the node
choices behind them. A model that already holds its choices returns those.
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

Training value the career choices imply at the installed state: U_S^(1) re-evaluated at the
installed T, then T on max{U_S^(0), U_S^(1)} (or the held branch). Equals `uc.T` in
equilibrium (residual R5).
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
`kt` by the enclosed share ω. The never-hired share stays trained.
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

"""
    crossing_rate(model, d = sc.d) → Matrix{Float64}

Total crossing hire rate `k` of `crossing_rates`, the denominator of m_S.
"""
crossing_rate(model::Model, d::AbstractMatrix{Float64} = model.skl_cache.d) = crossing_rates(model, d).k


# ---------------------------------------------------------------------------
# Stationary composition on the copula grid
# ---------------------------------------------------------------------------

"""
    solve_stationary_unskilled!(u_out, t_out, W2, τ, pstar, κ, model)

Stationary unemployed and training densities with no crossers: each cell's weight `W2`
splits by its training fraction τ, `u_U = W2 usurv (1 − τ)` and `t = ν W2 τ / (φ + ν)`.
"""
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
branch, t, the training slice's employment and the trained mass m_S. A cell with no crossers
has the closed forms of `solve_stationary_unskilled!` and m_S = φ t / ν. A cell whose trained
cross at rate k = ks + kt balances, with δ = ξ_U + λ_U G(p*) on each slice's own p*,
  (φ + ν) t      = ν τ ℓ + δ_t e_t
  (ν + k) m_S    = φ t
  (δ_t + ν) e_t  = kt m_S,
and the searching slice's separations, δ_s ks m_S υʰ / (ν (δ_s + ν)), join the row's hired
unemployed.
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

"""
    install_unskilled_masses!(model, κ) → NamedTuple

Install the unskilled block's own stationary densities (u, uh, t, et) in the cache and
return `stationary_masses`; m_S and d·u_S stay as installed.
"""
function install_unskilled_masses!(model::Model, κ::Float64)
    uc = model.unsk_cache
    ms = stationary_masses(model, κ)
    copyto!(uc.u, ms.u);  copyto!(uc.uh, ms.uh);  copyto!(uc.t, ms.t);  copyto!(uc.et, ms.et)
    return ms
end

"""
    install_stationary_masses!(model, κ)

Install `stationary_masses` in the caches, m_S included, with the seeker pool
`duS_carry = d m_S` (the draining trained are all unemployed and seek in the unskilled market).
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
# Reservation quality and free entry
# ---------------------------------------------------------------------------

"""
    update_pstar_from_surplus!(pstar_new, model, Ivec; θ_new, E1)

Reservation quality from S_U(aU, p*) = 0, `p* = [(r+ν) U^search − λ I] / (exp(A) P_U aU)`,
with U^search recomputed at the proposed tightness `θ_new` from `E1` = E_U(·,1).
"""
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

"""
    unskilled_meeting_value(model; u_hired = nothing) → (Jbar, U_total)

The two sides of unskilled free entry at the installed composition,
`q_U(θ_U) = k_U π̄_U ũ_U / J̄_U`: the seeker pool ũ_U = ∬ (u_U + d u_S), never-hired types
included, and the firm value of its meetings,
`J̄_U = ∬ J_U(aU,1) [u_U^h + (1−ω) χ d u_S] + J_t(·,1) ω χ_t d u_S`, which weights J_U by the
hired branch only. `u_hired` defaults to the stationary `uc.uh`; the transition passes its
date's stock split. Under `corner_limit` with no seekers, the limit composition is used.
"""
function unskilled_meeting_value(model::Model; u_hired::Union{Nothing,AbstractMatrix{Float64}} = nothing)
    uc = model.unsk_cache;  up = model.unsk_par
    κ   = meeting_rate(uc.θ, up.μ, up.η)
    u   = uc.u;  duS = uc.duS_carry;  uʰ = u_hired === nothing ? uc.uh : u_hired
    # Corner with everyone training: price the limit composition of the labour force,
    # with its stationary hired split.
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

"""
    _no_meeting_value(v) → Bool

Whether a meeting value from `unskilled_meeting_value` is empty or not finite.
"""
_no_meeting_value(v) = v.Jbar < 1e-14 || v.U_total < 1e-14 || !isfinite(v.Jbar) || !isfinite(v.U_total)

"""
    update_theta_unskilled(model; u_hired = nothing) → θ_U

Free-entry tightness at the installed composition (`unskilled_meeting_value`), clamped to
[1e-14, 100]; 1e-14 where a meeting is worth nothing.
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

Vacancy cost free entry implies at the installed tightness and composition,
`k_U = q_U(θ_U) J̄_U / (π̄_U ũ_U)`, the inverse of `update_theta_unskilled`. Zero where a
meeting is worth nothing.
"""
function vacancy_cost_unskilled(model::Model)
    uc = model.unsk_cache;  up = model.unsk_par
    v  = unskilled_meeting_value(model)
    _no_meeting_value(v) && return 0.0
    return q_from_theta(uc.θ, up.μ, up.η) * v.Jbar / (mean_output_U(model) * v.U_total)
end


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------

"""
    solve_unskilled_block!(model; US_in, θU_fixed = nothing) → (converged, rejected)

Outer loop on (θ_U, p*_U, u_U, t) given the skilled unemployment value `US_in`. Each pass:
inner values, stationary masses, free-entry θ_U under-relaxed by `DAMP_THETA_U`, p*_U from
S_U(aU,p*) = 0, then a joint Anderson(1) install on [θ_U; p*_U] (or damped Picard on p*_U).
Converges on max(Δθ_U, Δp*_U, Δu_U) < `tol_outer_U` for `conv_streak` passes. `inner_K`
diverged inner solves in a row reject the parameter; an outer stall hands back to the global
loop. `θU_fixed` holds θ_U and iterates only (p*_U, u_U). Every exit re-runs the inner loop
and reinstalls the masses.
"""
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

        # Under-relaxed free entry: both branches below iterate the damped map.
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
            # The Anderson step extrapolates: keep θ_U inside free entry's own bracket.
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

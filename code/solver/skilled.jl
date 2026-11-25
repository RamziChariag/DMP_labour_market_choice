############################################################
# skilled.jl — Skilled block
#
# Every skilled value reads aS alone: U_S^(0), E_S^0, E_S^1, J_S^0, J_S^1, p*_S, p^oj_S.
# Production is π_S = exp(A) P_S aS p. This block solves the stay-skilled branch
#   U_S^(0)(aS) = (b_S + κ_S β_S I_S(aS)) / (r + ν);
# the crossing branch U_S^(1)(aU,aS) and the drain d are career choices (unskilled.jl).
# The employed surplus uses U_S^(0) as outside option: the draining fraction holds no
# employed mass.
#
# Stationary composition: per-aS unit shapes û(aS), ê(aS,p) of a non-draining type,
# scaled by the trained mass m_S(aU,aS) (stationary_masses). Free entry aggregates over
# aS with the non-draining column masses Σ_aU (1−d) m_S.
#
# Entry points: solve_skilled_block!, cutoff_proposals!, compute_Jbar_skilled,
# update_theta_skilled, vacancy_cost_skilled, drain_fraction!.
############################################################


"""
    build_skilled_precomp(sg, sp) → SkilledPrecomp

Offer CDF Γ_o = Beta(a_Γ, b_Γ) and shock CDF Γ_s(x) = Γ_o(x/δ) on the skilled grid, their
cell masses per unit `wp` and tail weights. The Gauss–Legendre nodes and weights stay:
`wp` is also the Lebesgue measure dp elsewhere.
"""
function build_skilled_precomp(sg::SkilledGrids, sp::SkilledParams)
    dist = Beta(sp.a_Γ, sp.b_Γ)
    δ    = sp.δ

    Γvals = cdf.(dist, sg.p)
    Γs_vals = cdf.(dist, clamp.(sg.p ./ δ, 0.0, 1.0))

    γvals   = build_cell_mass_density(x -> cdf(dist, clamp(x,     0.0, 1.0)), sg.p, sg.wp)
    γs_vals = build_cell_mass_density(x -> cdf(dist, clamp(x / δ, 0.0, 1.0)), sg.p, sg.wp)

    wΓ  = γvals   .* sg.wp
    wΓs = γs_vals .* sg.wp

    return SkilledPrecomp(
        Γvals = Γvals, γvals = γvals, tail_weights = build_tail_weights(wΓ),
        Γs_vals = Γs_vals, γs_vals = γs_vals, tail_weights_s = build_tail_weights(wΓs),
    )
end


# ---------------------------------------------------------------------------
# Cutoff location
# ---------------------------------------------------------------------------

"""
    find_cutoff_from_j0(pgrid, Smax, j0_prev) → p*_S

Zero of `Smax` on the grid by linear interpolation, scanning from the previous cutoff index
`j0_prev`: 1.0 if `Smax < 0` at and above it, 0.0 if `Smax ≥ 0` at and below it.
"""
function find_cutoff_from_j0(pgrid::Vector{Float64}, Smax::AbstractVector{Float64}, j0_prev::Int)
    Np = length(pgrid)
    if Smax[j0_prev] < 0.0
        j_pos = 0
        @inbounds for j in j0_prev:Np
            Smax[j] >= 0.0 && (j_pos = j; break)
        end
        j_pos == 0 && return 1.0
        j_neg = max(j_pos - 1, 1)
    else
        j_neg = 0
        @inbounds for j in j0_prev:-1:1
            Smax[j] < 0.0 && (j_neg = j; break)
        end
        j_neg == 0 && return 0.0
        j_pos = min(j_neg + 1, Np)
    end
    S_lo = Smax[j_neg];  S_hi = Smax[j_pos];  dS = S_hi - S_lo
    abs(dS) < 1e-14 && return pgrid[j_pos]
    return pgrid[j_neg] + clamp(-S_lo / dS, 0.0, 1.0) * (pgrid[j_pos] - pgrid[j_neg])
end

"""
    _quad_crossing(x0, x1, x2, f0, f1, f2) → Float64

Root in [x1, x2] of the quadratic through the three points, or NaN when none is there
(the caller then uses the linear rule).
"""
function _quad_crossing(x0, x1, x2, f0, f1, f2)
    a1 = (f1 - f0) / (x1 - x0)
    a2 = ((f2 - f1) / (x2 - x1) - a1) / (x2 - x0)
    A  = a2
    B  = a1 - a2 * (x0 + x1)
    C  = f0 - a1 * x0 + a2 * x0 * x1
    abs(A) < 1e-14 && return abs(B) < 1e-14 ? NaN : -C / B
    disc = B * B - 4.0 * A * C
    disc < 0.0 && return NaN
    sq = sqrt(disc)
    for r in ((-B + sq) / (2.0 * A), (-B - sq) / (2.0 * A))
        x1 <= r <= x2 && return r
    end
    return NaN
end

"""
    _poj_in_cell(pgrid, diff, j) → Float64

Crossing of `diff` inside cell j, where `diff[j] > 0 >= diff[j+1]`: linear, or quadratic
under `POJ_CROSSING_C1`.
"""
@inline function _poj_in_cell(pgrid::Vector{Float64}, diff::AbstractVector{Float64}, j::Int)
    @inbounds begin
        d_hi = diff[j];  d_lo = diff[j+1];  dD = d_hi - d_lo
        abs(dD) < 1e-14 && return pgrid[j+1]
        if POJ_CROSSING_C1[] && j > 1
            r = _quad_crossing(pgrid[j-1], pgrid[j], pgrid[j+1], diff[j-1], d_hi, d_lo)
            isfinite(r) && return r
        end
        return pgrid[j] + clamp(d_hi / dD, 0.0, 1.0) * (pgrid[j+1] - pgrid[j])
    end
end

"""
    find_poj_from_diff_grid(pgrid, diff, pstar) → p^oj_S

Zero crossing of `diff` = S¹ − S⁰ at or above `pstar`: the highest (`POJ_CROSSING = :last`)
or the lowest (`:first`).
"""
function find_poj_from_diff_grid(pgrid::Vector{Float64}, diff::AbstractVector{Float64}, pstar::Float64)
    Np = length(pgrid)
    j0 = pcut_index(pgrid, pstar)

    if POJ_CROSSING[] === :last
        @inbounds diff[Np] > 0.0 && return 1.0
        @inbounds for j in (Np - 1):-1:j0
            diff[j] > 0.0 && return _poj_in_cell(pgrid, diff, j)
        end
        @inbounds return pgrid[j0]
    end

    @inbounds for j in j0:Np
        if diff[j] <= 0.0
            j > j0 && return _poj_in_cell(pgrid, diff, j - 1)
            return pgrid[j]
        end
    end
    return 1.0
end


# ---------------------------------------------------------------------------
# Covered-fraction margins on the ability grid
# ---------------------------------------------------------------------------

"""
    ojs_participation!(part, M, x) → part

OJS participation weight of each ability cell: the fraction of the cell on which the score
M(aS) = max_{p ≥ p*_S} (S¹ − S⁰) is positive.
"""
ojs_participation!(part::AbstractVector{Float64}, M::AbstractVector{Float64},
                   x::AbstractVector{Float64}) =
    covered_fraction!(part, M, x, node_midpoint_intervals(x)...)

"""
    drain_fraction!(d, U1, U0, x) → d

Drain d(aU, aS): along each aU row, the covered fraction of each cell on which `U1 > U0`.
Shared by the stationary career choices and the transition.
"""
function drain_fraction!(d::AbstractMatrix{Float64}, U1::AbstractMatrix{Float64},
                         U0::AbstractMatrix{Float64}, x::AbstractVector{Float64})
    xlo, xhi = node_midpoint_intervals(x)
    @threads for i in axes(d, 1)
        g = [U1[i, j] - U0[i, j] for j in axes(d, 2)]
        covered_fraction!(view(d, i, :), g, x, xlo, xhi)
    end
    return d
end


# ---------------------------------------------------------------------------
# Inner loop
# ---------------------------------------------------------------------------

"""
    skilled_inner_loop!(model) → (κ, status, converged, iters)

Iterate U_S^(0)(aS) and the surplus surfaces S⁰, S¹ (stored as E0, E1, J0, J1 with the
reservation soft weight) at fixed θ_S, p*_S, p^oj_S and participation. The unemployed value
and the poaching gain read the offer tail, the λ_S redraw the shock tail; both tails weight
the destination by the same search rule as the firm side (`_ojs_share`). Copies U0 into the
link `sc.U` on exit. `status` as in `unskilled_inner_loop!`.
"""
function skilled_inner_loop!(model::Model)
    cp = model.common;  gp = model.grids;  sp = model.skl_par
    sg = model.skl_grids;  pre = model.skl_pre;  sc = model.skl_cache;  sim = model.sim

    Nx = length(gp.x);  Np = length(sg.p)
    r  = cp.r;  ν = cp.ν
    PS = exp(cp.A) * sp.PS;  bS = sp.bS * exp(cp.A)
    β  = sp.β;  λ = sp.λ;  σ = sp.σ * exp(cp.A);  ξ = sp.ξ
    μ  = sp.μ;  η = sp.η

    κ   = meeting_rate(sc.θ, μ, η)                  # κ_S
    wΓo = pre.γvals   .* sg.wp                      # offer  ∫·dΓ_o
    wΓs = pre.γs_vals .* sg.wp                      # shock  ∫·dΓ_s

    # Partition fixed per call with one residual slot per chunk, reduced afterwards in
    # chunk order (index_chunks).
    chunks = index_chunks(Nx)
    dU_ch  = zeros(Float64, length(chunks))
    dS_ch  = zeros(Float64, length(chunks))

    αS = sim.damp_inner_S
    B  = sim.inner_B;  K = sim.inner_K
    resid = Vector{Float64}(undef, sim.maxit_inner)
    status = :maxit;  n_inner = 0;  streak = 0
    denom_nb = max(1.0 - β, 1e-14)

    for it in 1:sim.maxit_inner
        @sync for (c, rng) in enumerate(chunks)
            Threads.@spawn begin
                tailEo = zeros(Float64, Np)                # offer tail (poaching, unemployed search)
                tailEs = zeros(Float64, Np)                # shock tail (λ_S redraw option value)
                dU_loc = 0.0;  dS_loc = 0.0                # chunk-local max residuals
                for k in rng                               # k indexes the skilled aptitude aS
                    @inbounds begin
                        aS      = gp.x[k]
                        PSeff   = PS * aS                  # A P_S aS (linear loading)
                        pstar   = clamp01(sc.pstar[k])
                        poj     = clamp01(sc.poj[k])
                        part    = sc.s_part[k]
                        j0      = pcut_index(sg.p, pstar)
                        j0_soft = max(j0 - 1, 1)

                        # Tails ∫_{p*}^1 S^max dΓ, read from one cell below the cutoff node.
                        # The destination uses the search rule `_ojs_share`, not a pointwise
                        # max: near the pole of γ_o at p = 0 (a_Γ < 1) the pointwise test
                        # finds a spurious crossing.
                        acco = 0.0;  accs = 0.0
                        for j in Np:-1:1
                            s_j        = _ojs_share(part, sg.p[j], poj, sg.p, j, Np)
                            Smax_j     = (s_j * sc.J1[k, j] + (1.0 - s_j) * sc.J0[k, j]) / denom_nb
                            acco      += Smax_j * wΓo[j];  tailEo[j] = acco
                            accs      += Smax_j * wΓs[j];  tailEs[j] = accs
                        end
                        I_o = tailEo[j0_soft]              # offer: fresh-meeting search option
                        I_s = tailEs[j0_soft]              # shock: λ_S redraw option

                        # Stay-skilled unemployment value (offer integral).
                        U0_raw   = (bS + κ * β * I_o) / (r + ν)
                        U0_old   = sc.U0[k]
                        U0_new   = U0_old + αS * (U0_raw - U0_old)
                        sc.U0[k] = U0_new
                        dU_loc = max(dU_loc, abs(U0_new - U0_old))

                        base = r + ν + λ + ξ               # ϱ_S
                        for j in 1:Np
                            pj  = sg.p[j]
                            ω_j = _soft_weight(pj, pstar, sg.p, j, Np)
                            if ω_j <= 0.0
                                sc.E0[k, j] = U0_new;  sc.E1[k, j] = U0_new
                                sc.J0[k, j] = 0.0;     sc.J1[k, j] = 0.0
                                continue
                            end
                            # No-search surplus: λ_S redraw on the shock integral.
                            S0 = smooth_pos((PSeff * pj - (r + ν) * U0_new + λ * I_s) / base)
                            # OJS surplus: redraw on the shock integral; poaching gain and
                            # outflow on the offer tail.
                            tail_mass_j = pre.tail_weights[j]
                            tail_Emax_j = tailEo[max(j, j0_soft)]
                            S1 = smooth_pos((PSeff * pj - (r + ν) * U0_new - σ + λ * I_s +
                                             κ * β * tail_Emax_j) / (base + κ * tail_mass_j))

                            E0_old = sc.E0[k, j];  E1_old = sc.E1[k, j]
                            sc.E0[k, j] = U0_new + β * ω_j * S0
                            sc.E1[k, j] = U0_new + β * ω_j * S1
                            sc.J0[k, j] = (1.0 - β) * ω_j * S0
                            sc.J1[k, j] = (1.0 - β) * ω_j * S1
                            dS_loc = max(dS_loc, abs(sc.E0[k,j]-E0_old), abs(sc.E1[k,j]-E1_old))
                        end
                    end
                end
                dU_ch[c] = dU_loc;  dS_ch[c] = dS_loc
            end
        end

        d = max(maximum(dU_ch), maximum(dS_ch))
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

    @inbounds for j in 1:Nx
        sc.U[j] = sc.U0[j]                         # employed outside option (1D in aS)
    end

    return (κ = κ, status = status, converged = (status === :converged), iters = n_inner)
end


# ---------------------------------------------------------------------------
# Stationary composition
# ---------------------------------------------------------------------------

"""
    solve_stationary_skilled!(model)

Per-aS unit shapes of a non-draining type with trained mass 1: the unemployed share
`sc.u_frac` and the employed density `sc.e_frac` in p, from the flow balance with
ê(p) = α(p) û + β(p) closed by û + ∫ ê dp = 1. Hires and poaching inflows read the offer
density, λ_S redraws (out and in) the shock density.
"""
function solve_stationary_skilled!(model::Model)
    sg = model.skl_grids;  pre = model.skl_pre;  sc = model.skl_cache
    sp = model.skl_par;    cp = model.common
    Nx = length(model.grids.x);  Np = length(sg.p)
    ν = cp.ν;  λ = sp.λ;  ξ = sp.ξ
    κ = meeting_rate(sc.θ, sp.μ, sp.η)

    @threads for k in 1:Nx
        @inbounds begin
            pstar   = clamp01(sc.pstar[k])
            poj     = clamp01(sc.poj[k])
            part    = sc.s_part[k]
            j0      = pcut_index(sg.p, pstar)
            j0_soft = max(j0 - 1, 1)

            α_coef = zeros(Float64, Np)
            β_coef = zeros(Float64, Np)
            ω_arr  = zeros(Float64, Np)
            CumAlpha = 0.0;  CumBeta = 0.0

            for j in j0_soft:Np
                pj  = sg.p[j];  wpj = sg.wp[j]
                γoj = pre.γvals[j]                          # offer: hire + poaching inflow
                γsj = pre.γs_vals[j]                        # shock: λ_S redraw in/outflow
                ω_j = _soft_weight(pj, pstar, sg.p, j, Np)
                ω_arr[j] = ω_j
                ω_j <= 0.0 && continue
                s_j = _ojs_share(part, pj, poj, sg.p, j, Np)

                # A poached worker draws a cell strictly above her own (inflow to cell j comes
                # from the searchers below it, accumulated after j is priced), so the outflow
                # is the offer mass above cell j: tail_weights[j + 1]. Inflow equals outflow.
                acc_mass = j < Np ? pre.tail_weights[j + 1] : 0.0
                a_j = ν + λ + ξ + s_j * κ * acc_mass
                num_α = (κ * γoj - λ * γsj) + κ * γoj * CumAlpha   # unit-mass forcing: κ û γ term
                num_β = (λ * γsj) * 1.0     + κ * γoj * CumBeta    # m_S = 1 in the unit problem
                if a_j < 1e-14
                    α_coef[j] = 0.0;  β_coef[j] = 0.0
                else
                    α_coef[j] = num_α / a_j
                    β_coef[j] = num_β / a_j
                end
                CumAlpha += ω_j * s_j * α_coef[j] * wpj
                CumBeta  += ω_j * s_j * β_coef[j] * wpj
            end

            sum_α_wp = 0.0;  sum_β_wp = 0.0
            for j in j0_soft:Np
                sum_α_wp += ω_arr[j] * α_coef[j] * sg.wp[j]
                sum_β_wp += ω_arr[j] * β_coef[j] * sg.wp[j]
            end

            denom_u = 1.0 + sum_α_wp
            uf = denom_u > 1e-14 ? clamp((1.0 - sum_β_wp) / denom_u, 0.0, 1.0) : 0.0
            sc.u_frac[k] = uf
            for j in 1:Np
                sc.e_frac[k, j] = (j < j0_soft || ω_arr[j] <= 0.0) ? 0.0 :
                                  ω_arr[j] * max(α_coef[j] * uf + β_coef[j], 0.0)
            end
        end
    end
    return nothing
end


"""
    nondrain_column_mass(d, mS) → mcol0

Non-draining trained mass by aS, `mcol0[j] = Σ_i (1 − d[i,j]) m_S[i,j]`.
"""
function nondrain_column_mass(d::AbstractMatrix{Float64}, mS::AbstractMatrix{Float64})
    Nx = size(mS, 2)
    mcol0 = zeros(Float64, Nx)
    @inbounds for j in 1:Nx, i in 1:Nx
        mcol0[j] += (1.0 - d[i, j]) * mS[i, j]
    end
    return mcol0
end

"""
    skilled_stayer_columns(model, d, mS) → mcol0

The skilled seekers' composition by aS: the non-draining column mass, or, when the market
holds none, its limit as it empties (the admissible cells with the largest training gain,
`marginal_training_cells`, carried through the model's own m_S(n)).
"""
function skilled_stayer_columns(model::Model, d::AbstractMatrix{Float64},
                                mS::AbstractMatrix{Float64})
    mcol0 = nondrain_column_mass(d, mS)
    sum(mcol0) > EMPTY_MASS && return mcol0
    cp = model.common
    w  = marginal_training_cells(model; entering = true, admit = d .< 1.0)
    return nondrain_column_mass(d, _mS_from_n(w, crossing_rate(model, d), cp.φ, cp.ν))
end


# ---------------------------------------------------------------------------
# Free entry
# ---------------------------------------------------------------------------

"""
    compute_Jbar_skilled(model) → J̄_S

Expected firm value of a meeting for a skilled vacancy. Seekers are the non-draining
unemployed, û(aS) mcol0(aS), priced over every acceptable destination, and the employed
searchers, s(aS,p) ê(aS,p) mcol0(aS), priced over destinations above their own quality. The
OJS margin enters through `_ojs_share` (continuity, not accuracy) and the reservation margin
through the tail read at one cell below the cutoff node. While m_S has not settled
(num < 1e-12) the tails are weighted by the non-draining marginal ℓ. Under `corner_limit`
the seekers are `skilled_stayer_columns`.
"""
function compute_Jbar_skilled(model::Model)
    gp = model.grids;  sg = model.skl_grids;  pre = model.skl_pre;  sc = model.skl_cache
    Nx = length(gp.x);  Np = length(sg.p)
    wΓ = pre.γvals .* sg.wp
    mcol0 = model.corner_limit ? skilled_stayer_columns(model, sc.d, sc.m_S) :
                                 nondrain_column_mass(sc.d, sc.m_S)

    # Per-node accumulators summed in index order afterwards, so J̄_S does not depend on the
    # schedule (U_S is very sensitive to θ_S, so last-bit differences decide feasibility).
    num_by_k = zeros(Float64, Nx)
    den_by_k = zeros(Float64, Nx)

    @threads for k in 1:Nx
        @inbounds begin
            mk = mcol0[k]
            mk <= 0.0 && continue
            pstar   = clamp01(sc.pstar[k])
            poj     = clamp01(sc.poj[k])
            part    = sc.s_part[k]
            j0_soft = max(pcut_index(sg.p, pstar) - 1, 1)

            tailJ = zeros(Float64, Np)
            acc = 0.0
            for j in Np:-1:1
                s_j = _ojs_share(part, sg.p[j], poj, sg.p, j, Np)
                acc += (s_j * sc.J1[k, j] + (1.0 - s_j) * sc.J0[k, j]) * wΓ[j]
                tailJ[j] = acc
            end

            seeker_e = 0.0
            num_k    = sc.u_frac[k] * tailJ[j0_soft]
            for j in 1:Np
                s_j = _ojs_share(part, sg.p[j], poj, sg.p, j, Np)
                s_j <= 0.0 && continue
                seeker_e += s_j * sc.e_frac[k, j] * sg.wp[j]
                num_k    += s_j * sc.e_frac[k, j] * sg.wp[j] * tailJ[max(j, j0_soft)]
            end
            den_by_k[k] = mk * (sc.u_frac[k] + seeker_e)
            num_by_k[k] = mk * num_k
        end
    end

    num = sum(num_by_k);  den = max(sum(den_by_k), 1e-14)

    if num < 1e-12
        num = 0.0;  den = 0.0
        for k in 1:Nx
            ell = 0.0
            for i in 1:Nx
                ell += (1.0 - sc.d[i, k]) * gp.copula.W2[i, k]
            end
            j0_soft = max(pcut_index(sg.p, clamp01(sc.pstar[k])) - 1, 1)
            poj     = clamp01(sc.poj[k])
            part    = sc.s_part[k]
            acc = 0.0
            for j in Np:-1:j0_soft
                s_j = _ojs_share(part, sg.p[j], poj, sg.p, j, Np)
                acc += (s_j * sc.J1[k, j] + (1.0 - s_j) * sc.J0[k, j]) * wΓ[j]
            end
            num += ell * acc
            den += ell
        end
        den = max(den, 1e-14)
    end

    return num / den
end


"""
    update_theta_skilled(model) → θ_S

Free-entry tightness `q_S(θ_S) = k_S π̄_S / J̄_S`, clamped to [1e-14, 100]; 1e-14 where
J̄_S is below 1e-12 or not finite.
"""
function update_theta_skilled(model::Model)
    sp = model.skl_par
    Jbar = compute_Jbar_skilled(model)
    (Jbar < 1e-12 || !isfinite(Jbar)) && return 1e-14
    q     = sp.k * mean_output_S(model) / Jbar
    θ_raw = theta_from_q(q, sp.μ, sp.η)
    return clamp(θ_raw, 1e-14, 100.0)
end

"""
    vacancy_cost_skilled(model) → k_S

Vacancy cost free entry implies at the installed tightness and composition,
`k_S = q_S(θ_S) J̄_S / π̄_S`, the inverse of `update_theta_skilled`. Zero where a meeting is
worth nothing.
"""
function vacancy_cost_skilled(model::Model)
    sp = model.skl_par
    Jbar = compute_Jbar_skilled(model)
    (Jbar < 1e-12 || !isfinite(Jbar)) && return 0.0
    return q_from_theta(model.skl_cache.θ, sp.μ, sp.η) * Jbar / mean_output_S(model)
end


# ---------------------------------------------------------------------------
# Outer loop on (θ_S, p*_S, p^oj_S, s_part)
# ---------------------------------------------------------------------------

# Most types the convergence test may excuse as boundary outliers before it falls back to the
# plain sup-norm. A constant: a larger value would let a drifting frontier pass.
const CONV_S_MAXOUT = 2

"""
    cutoff_proposals!(pst_prop, poj_prop, Mscore, model, κ)

The cutoffs the stored surpluses imply, a function of the stored state alone (shared by the
block and residual R1): p*_S the interpolated zero of the raw S^max, p^oj_S a crossing of
S¹ − S⁰ above it (`find_poj_from_diff_grid`), and the participation score `Mscore` =
max_{p ≥ p*_S} (S¹ − S⁰). Under `OJS_PARTICIPATION = :fraction` a type with M ≤ 0 takes p^oj
at the peak of S¹ − S⁰, the limit of the crossing. p^oj_S ≥ p*_S.
"""
function cutoff_proposals!(pst_prop::AbstractVector{Float64}, poj_prop::AbstractVector{Float64},
                           Mscore::AbstractVector{Float64}, model::Model, κ::Float64)
    cp = model.common;  gp = model.grids;  sg = model.skl_grids
    sc = model.skl_cache;  sp = model.skl_par;  pre = model.skl_pre
    Nx = length(gp.x);  Np = length(sg.p)
    denom_nb = max(1.0 - sp.β, 1e-14)
    r  = cp.r;  ν = cp.ν
    PS = exp(cp.A) * sp.PS
    β  = sp.β;  λ = sp.λ;  σ = sp.σ * exp(cp.A);  ξ = sp.ξ
    wΓo = pre.γvals   .* sg.wp                      # offer  ∫·dΓ_o
    wΓs = pre.γs_vals .* sg.wp                      # shock  ∫·dΓ_s

    @threads for k in 1:Nx
        @inbounds begin
            aS = gp.x[k];  PSeff = PS * aS;  U0k = sc.U0[k]
            poj_prev = clamp01(sc.poj[k]);  part_prev = sc.s_part[k]
            j0_prev = pcut_index(sg.p, clamp01(sc.pstar[k]))
            j0_soft = max(j0_prev - 1, 1)

            # Tails under the same search rule as the value block and the firm side.
            tailEo = zeros(Float64, Np);  tailEs = zeros(Float64, Np)
            acco = 0.0;  accs = 0.0
            for j in Np:-1:1
                s_j    = _ojs_share(part_prev, sg.p[j], poj_prev, sg.p, j, Np)
                Smax_j = (s_j * sc.J1[k, j] + (1.0 - s_j) * sc.J0[k, j]) / denom_nb
                acco += Smax_j * wΓo[j];  tailEo[j] = acco
                accs += Smax_j * wΓs[j];  tailEs[j] = accs
            end
            I_s = tailEs[j0_soft]              # shock: λ_S redraw option value

            Smax_raw = zeros(Float64, Np);  diff_raw = zeros(Float64, Np)
            base = r + ν + λ + ξ
            for j in 1:Np
                pj = sg.p[j]
                raw_S0 = (PSeff * pj - (r + ν) * U0k + λ * I_s) / base
                tail_mass_j = pre.tail_weights[j]
                tail_Emax_j = tailEo[max(j, j0_soft)]
                raw_S1 = (PSeff * pj - (r + ν) * U0k - σ + λ * I_s +
                          κ * β * tail_Emax_j) / (base + κ * tail_mass_j)
                Smax_raw[j] = max(raw_S0, raw_S1)
                diff_raw[j] = POJ_SCORE[] === :smooth ?
                              smooth_pos(raw_S1) - smooth_pos(raw_S0) :
                              raw_S1 - raw_S0
            end
            pst_prop[k] = clamp01(find_cutoff_from_j0(sg.p, Smax_raw, j0_prev))

            # Participation score and the node attaining it. As M → 0 the searching interval
            # collapses onto that peak, so a straddling cell's searching branch uses it.
            jpk = j0_prev
            for j in (j0_prev + 1):Np
                diff_raw[j] > diff_raw[jpk] && (jpk = j)
            end
            Mscore[k] = diff_raw[jpk]

            poj_prop[k] = max(pst_prop[k],
                              OJS_PARTICIPATION[] === :fraction && Mscore[k] <= 0.0 ?
                              sg.p[jpk] :
                              clamp01(find_poj_from_diff_grid(sg.p, diff_raw, pst_prop[k])))
        end
    end
    return nothing
end


"""
    solve_skilled_block!(model; θS_fixed = nothing) → (converged, rejected)

Outer loop on (θ_S, p*_S, p^oj_S, s_part), a damped fixed-point iteration with no
acceleration. Each pass: inner values; cutoff proposals; participation installed undamped;
cutoffs installed under `damp_pstar_S`; stationary shapes; free-entry θ_S under an adaptive
weight. Converges when the residual (with the boundary-outlier allowance) is below
`tol_outer_S` for `conv_streak` passes. Rejection and stall handback as in the unskilled
block. `θS_fixed` holds θ_S (one evaluation of the skilled free-entry map, skilled_equilibria).
"""
function solve_skilled_block!(model::Model; θS_fixed::Union{Nothing,Float64} = nothing)
    gp = model.grids;  sc = model.skl_cache;  sim = model.sim
    Nx = length(gp.x)
    θS_fixed === nothing || (sc.θ = θS_fixed)

    pst_prop = zeros(Float64, Nx)
    poj_prop = zeros(Float64, Nx)
    Mscore    = zeros(Float64, Nx)     # participation score max_{p >= p*_S} (S^1 − S^0)
    part_prop = ones(Float64, Nx)

    streak = 0;  diverge_streak = 0
    oB = sim.outer_B;  oK = sim.outer_K
    outer_resid = Vector{Float64}(undef, sim.maxit_outer)

    # Adaptive free-entry weight and the previous pass's outlier interpolants. Local to the
    # call: module-level state would make a result depend on evaluation order.
    w_θ_ad   = DAMP_THETA_S[]
    sgn_prev = 0
    itp_prev = Float64[]

    for it in 1:sim.maxit_outer
        θ_old     = sc.θ
        pstar_old = copy(sc.pstar)
        poj_old   = copy(sc.poj)
        part_old  = copy(sc.s_part)

        inner = skilled_inner_loop!(model)
        κ = inner.κ;  inner_conv = inner.converged

        if inner.status === :diverged
            diverge_streak += 1
            if diverge_streak >= sim.inner_K
                sim.verbose >= 1 && @printf("  [outer S]  inner diverged ×%d at it=%d — rejecting parameter\n", diverge_streak, it)
                return (converged = false, rejected = true)
            end
        else
            diverge_streak = 0
        end

        cutoff_proposals!(pst_prop, poj_prop, Mscore, model, κ)

        OJS_PARTICIPATION[] === :fraction && ojs_participation!(part_prop, Mscore, gp.x)
        copyto!(sc.s_part, part_prop)

        damp = sim.damp_pstar_S
        @inbounds for k in 1:Nx
            sc.pstar[k] = clamp01(damp * pst_prop[k] + (1.0 - damp) * pstar_old[k])
            sc.poj[k]   = max(sc.pstar[k], clamp01(damp * poj_prop[k] + (1.0 - damp) * poj_old[k]))
        end

        solve_stationary_skilled!(model)

        # Adaptive under-relaxation of free entry: halve the weight on a sign reversal of the
        # step (floor 0.1), else lengthen it back toward the ceiling DAMP_THETA_S. On the
        # multiplicity band this selects which attracting root the iteration reaches; at the
        # estimates the weight stays at the ceiling.
        sc.θ = if θS_fixed === nothing
            T_θ = update_theta_skilled(model)
            sgn = sign(T_θ - θ_old)
            if sgn_prev != 0 && sgn != 0 && sgn != sgn_prev
                w_θ_ad = max(0.5 * w_θ_ad, 0.1)
            else
                w_θ_ad = min(1.2 * w_θ_ad, DAMP_THETA_S[])
            end
            sgn_prev = sgn
            max(w_θ_ad * T_θ + (1.0 - w_θ_ad) * θ_old, 1e-14)
        else
            θS_fixed
        end

        if !isfinite(sc.θ) || any(!isfinite, sc.pstar) || any(!isfinite, sc.poj)
            sim.verbose >= 1 && @printf("  [outer S]  NaN/Inf in state at it=%d — aborting\n", it)
            return (converged = false, rejected = false)
        end

        # A type with zero participation before and after the pass is inactive: its p^oj
        # enters nothing (and sits on a flat plateau), so its Δp^oj counts as zero, as in R1.
        inactive = [part_old[k] == 0.0 && sc.s_part[k] == 0.0 for k in 1:Nx]
        Δoj = [inactive[k] ? 0.0 : abs(sc.poj[k] - poj_old[k]) for k in 1:Nx]

        dθ = abs(sc.θ - θ_old)
        dp = supnorm(sc.pstar, pstar_old)
        dj = maximum(Δoj)
        ds = supnorm(sc.s_part, part_old)        # identically 0 under :indicator
        d  = max(dθ, dp, dj, ds)

        # Outlier allowance: when 1 to CONV_S_MAXOUT types are above tolerance, replace each
        # outlier's p^oj by a linear interpolation in ability between the nearest
        # in-tolerance active anchors (the one anchor if a side has none), require those
        # interpolants to be stable across passes, and judge convergence on the other types.
        # For the convergence test only: sc.poj is never overwritten. The global residual
        # battery is the backstop.
        tol = sim.tol_outer_S
        out = [k for k in 1:Nx if max(Δoj[k], abs(sc.s_part[k] - part_old[k])) > tol]
        if 1 <= length(out) <= CONV_S_MAXOUT
            itp = Float64[]
            for k in out
                lo = k - 1;  while lo >= 1  && (lo in out || inactive[lo]);  lo -= 1;  end
                hi = k + 1;  while hi <= Nx && (hi in out || inactive[hi]);  hi += 1;  end
                if lo < 1 && hi > Nx
                    itp = Float64[]
                    break
                end
                push!(itp, lo < 1  ? sc.poj[hi] :
                           hi > Nx ? sc.poj[lo] :
                           sc.poj[lo] + (sc.poj[hi] - sc.poj[lo]) *
                                        (gp.x[k] - gp.x[lo]) / (gp.x[hi] - gp.x[lo]))
            end
            stable = length(itp) == length(out) && length(itp_prev) == length(itp) &&
                     all(abs(itp[i] - itp_prev[i]) < tol for i in eachindex(itp))
            itp_prev = itp
            if stable
                dR = 0.0
                @inbounds for k in 1:Nx
                    k in out && continue
                    dR = max(dR, abs(sc.pstar[k]  - pstar_old[k]),
                                 Δoj[k],
                                 abs(sc.s_part[k] - part_old[k]))
                end
                d = max(dθ, dR)
            end
        else
            itp_prev = Float64[]
        end
        outer_resid[it] = d

        if sim.verbose >= 2 && (it == 1 || it % sim.verbose_stride == 0)
            # d is the gated residual; nout types were excused, so d can sit below the components.
            @printf("  [outer S it=%d]  d=%.3e  (Δθ=%.3e  Δp*=%.3e  Δpoj=%.3e  Δpart=%.3e)  nout=%d  θ_S=%.4f\n",
                    it, d, dθ, dp, dj, ds, length(out), sc.θ)
        end

        if d < sim.tol_outer_S
            streak += 1
            if streak >= sim.conv_streak
                sim.verbose >= 2 && @printf("  [outer S]  converged it=%d  d=%.3e  θ_S=%.4f\n", it, d, sc.θ)
                solve_stationary_skilled!(model)
                return (converged = inner_conv, rejected = false)
            end
        else
            streak = 0
        end

        if oB > 0 && it > oB + oK && outer_resid[it] >= outer_resid[it - oK]
            sim.verbose >= 1 && @printf("  [outer S]  stalled at it=%d — handing back to global\n", it)
            solve_stationary_skilled!(model)
            return (converged = false, rejected = false)
        end
    end

    solve_stationary_skilled!(model)
    sim.verbose >= 1 && @printf("  [outer S]  maxit reached without convergence  θ_S=%.4f\n", sc.θ)
    return (converged = false, rejected = false)
end

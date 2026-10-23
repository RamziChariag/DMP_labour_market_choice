############################################################
# skilled.jl — Skilled block solver (RoySearch)
#
# Every skilled VALUE object reads the skilled aptitude aS alone (notes
# §382, §420): U_S^(0), E_S^0, E_S^1, J_S^0, J_S^1, p*_S, p^oj_S, I_S.
# Production is linear, π_S(aS,p) = A P_S aS p (no γ_S).
#
# The directed-search choice compares the two branches (notes eq:dpolicy):
#   U_S^(0)(aS)    = (b_S + κ_S β_S I_S(aS)) / (r + ν)       stay skilled
#   U_S^(1)(aU,aS) = (b_S + κ_U E_U(aU,aS,1)) / (r + ν + κ_U) cross to U
#   d(aU,aS)       = fraction of cell (aU,aS) with U_S^(1) > U_S^(0)
# This block solves the stay-skilled branch; the cross branch and d are the
# career choices of unskilled.jl, since E_U(aU,aS,1) is an unskilled match.
# The employed surplus uses U_S^(0)(aS) as the outside option: the
# non-draining fraction 1−d, which carries all the employed mass, has
# exactly this unemployment value, and the draining fraction d carries
# none, so its 1D-in-aS surplus is never read by the distribution.
#
# Stationary composition factors.  The p-dynamics read only aS, and the
# flow balance is linear-homogeneous in a type's trained mass, so the
# split into unemployed/employed is a per-aS UNIT shape scaled by the 2D
# trained mass m_S(aU,aS):
#   the 1−d part:  u_S = û(aS) m_S,  e_S(p) = ê(aS,p) m_S
#   the d part:    u_S = m_S,        e_S ≡ 0     (these seek in U, not S)
# where m_S(aU,aS) = φ t(aU,aS) / (ν + crossing hire rate) is supplied by
# the stationary masses (stationary_masses).  Free-entry aggregation reduces to a 1D sum over aS weighted by
# the non-draining column masses (1−d)·m_S.
#
# Functions
#   build_skilled_precomp        Γ CDFs + cell masses + tail weights
#   find_cutoff_from_j0          zero-crossing of S^max(aS,·)
#   find_poj_from_diff_grid      zero-crossing of S^1 − S^0
#   drain_fraction!              covered fraction of the drain margin
#   skilled_inner_loop!          iterate (U0, S^0, S^1)
#   solve_stationary_skilled!    per-aS unit shapes (û, ê)
#   compute_Jbar_skilled         expected firm value for free entry
#   update_theta_skilled         free-entry tightness update
#   solve_skilled_block!         outer loop on (θ_S, p*_S, p^oj_S)
############################################################


# ---------------------------------------------------------------------------
# Precompute Γ CDFs, cell masses and tail weights on the skilled p-grid
# ---------------------------------------------------------------------------
function build_skilled_precomp(sg::SkilledGrids, sp::SkilledParams)
    dist = Beta(sp.a_Γ, sp.b_Γ)
    δ    = sp.δ

    # Offer distribution Γ_o = Beta(a_Γ, b_Γ) on [0,1].
    Γvals = cdf.(dist, sg.p)

    # Shock distribution Γ_s(x) = Γ_o(x/δ) on [0,δ]: the shocked quality is
    # δ·(offer draw), so the CDF evaluates the offer at x/δ, with zero mass
    # above δ.  At δ = 1 the map is the identity and the shock arrays
    # reproduce the offer arrays exactly.
    Γs_vals = cdf.(dist, clamp.(sg.p ./ δ, 0.0, 1.0))

    # Both quality densities are EXACT-CDF CELL MASSES per unit wp, not
    # pointwise densities (see build_cell_mass_density in grids.jl).  Every
    # dΓ integral in the model appears as γ_j · wp_j, so each term carries the
    # exact mass of node j's cell: total mass is 1 by construction and the mass
    # around the shock density's pole at p = δ is read off the CDF instead of a
    # pdf sampled at whichever node lands nearest it.  The Gauss–Legendre nodes
    # and weights are deliberately untouched — sg.wp doubles as the Lebesgue dp
    # measure elsewhere, and a Gauss–Jacobi swap breaks it (∫ p dp = 0.4731
    # against the exact 0.5).
    #
    # Consequence: γvals and γs_vals must never be read pointwise.  Their error
    # against γ(p_j) is flat in the grid size (RMS 1.194 at Np_S = 60, 1.202 at
    # 480) because wp_j is a quadrature weight, not the cell's width.
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
# Locate p*_S(aS) by linear interpolation of the zero-crossing of
# S^max(aS,·), scanning from the previous cutoff index.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Locate p^oj_S(aS) at a zero-crossing of S^1 − S^0 above p*_S.
#
# Which crossing is taken is POJ_CROSSING (params.jl): :first is the lowest
# sign change at or above p*_S, :last the highest.  They coincide unless the
# difference changes sign more than once above p*_S, which happens at the
# bottom of the p-grid where the two raw surpluses are nearly equal.
#
# Inside the bracketing cell the crossing is interpolated, linearly by default
# and by the local quadratic under POJ_CROSSING_C1.
# ---------------------------------------------------------------------------

# Root of the quadratic through (x0,f0), (x1,f1), (x2,f2) lying in [x1, x2], in
# Newton divided-difference form.  NaN when no real root sits there, so the
# caller falls back to the linear rule.
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

# Crossing inside cell j, where diff[j] > 0 >= diff[j+1].
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
# OJS participation weight on the ability grid.
#
# M(aS) = max_{p >= p*_S} (S^1 − S^0)(aS, p) is the participation score: on-the-job
# search pays for aS at some acceptable match quality iff M(aS) > 0, so the searching
# set in ability is {M > 0} and its frontier is a zero of M.  The weight of ability
# cell k is the fraction of the cell that set covers (covered_fraction!) — the
# ability-grid analogue of the p-coverage _soft_oj_weight returns.
# ---------------------------------------------------------------------------
ojs_participation!(part::AbstractVector{Float64}, M::AbstractVector{Float64},
                   x::AbstractVector{Float64}) =
    covered_fraction!(part, M, x, node_midpoint_intervals(x)...)


# ---------------------------------------------------------------------------
# Cross-market drain d(aU,aS): the POPULATION FRACTION of cell (aU,aS) that
# abandons skill and searches in the unskilled market, from U_S^(1)(aU,aS) >
# U_S^(0)(aS) (notes eq:dpolicy).
#
# A hard 0/1 at the nodes made every aggregate that reads d a step function of
# the skilled parameters: a change too small to move the margin past a node
# flipped no cell, so the non-draining column masses and the augmented seeker
# pool were bit-identical and ∂m/∂θ was exactly zero.  d is the covered fraction
# of each cell, continuous in the margin and equal to the indicator on every cell
# the margin does not cut.  U_S^(0) is increasing in aS, so where U_S^(1) does not
# vary with aS the drain set is a lower interval; where U_S^(1) reads aS — the
# trained enclosed by the training frontier — it need not be, and each sign change
# is located where it lies.
# ---------------------------------------------------------------------------
"""
    drain_fraction!(d, U1, U0, x)

Drain `d(a_U, a_S)` from the two branch values on the (aU, aS) grid: the covered fraction
of each cell on which `U1 > U0`.  Shared by the stationary career choices and the
transition.
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
#
# Given (θ_S, p*_S, p^oj_S) fixed, iterate on the surplus surfaces
# (S^0_S, S^1_S) and the stay-skilled unemployment value U_S^(0)(aS)
# until convergence.
# ---------------------------------------------------------------------------
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

    # Fixed partition of the aS grid, with one residual slot per CHUNK.  The tail scratch
    # pair is allocated inside each task rather than held here, so no two tasks can alias
    # it; see `index_chunks` for why this is a @spawn fan-out and not `@threads :static`.
    # Computed once per call: at fixed Nx the partition never changes, so the residual
    # reduction groups the same terms on every run and every thread count.
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

                        # Tail integrals I_S(aS) = ∫_{p*}^1 S^max dΓ, one per distribution.
                        # Stored J0/J1 carry the soft weight, so /(1−β) recovers ω·S.
                        # The destination is valued under the SAME search rule the firm prices and
                        # the distributions flow, `_ojs_share(part, ·)`.  A pointwise max here would
                        # be the notes' s* = 1{S¹ ≥ S⁰} read node by node — identical on the exact
                        # solution, but NOT on a finite grid: γ_o has a pole at p = 0 whenever
                        # a_Γ < 1, and the pointwise test picks up a spurious sign change in the
                        # first cell or two.  Measured at the base_fc estimate: the spurious lower
                        # crossing tracks the cell width (15.93× over a 15.9× refinement at one
                        # type, exactly 1.00 cells from the floor at every resolution) while the
                        # upper crossing is stable to 1.1%, as stable as the 76 single-crossing
                        # types.  The cutoff form reads only the stable boundary, so it is the
                        # better approximation to s*, not a departure from it.
                        acco = 0.0;  accs = 0.0
                        for j in Np:-1:1
                            s_j        = _ojs_share(part, sg.p[j], poj, sg.p, j, Np)
                            Smax_j     = (s_j * sc.J1[k, j] + (1.0 - s_j) * sc.J0[k, j]) / denom_nb
                            acco      += Smax_j * wΓo[j];  tailEo[j] = acco
                            accs      += Smax_j * wΓs[j];  tailEs[j] = accs
                        end
                        I_o = tailEo[j0_soft]              # offer: fresh-meeting search option
                        I_s = tailEs[j0_soft]              # shock: λ_S redraw option

                        # Stay-skilled unemployment value (self-referential via the
                        # OFFER integral — an unemployed worker meets fresh offers).
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
                            # No-search surplus (eq SS0): the λ_S redraw reads the SHOCK integral.
                            S0 = smooth_pos((PSeff * pj - (r + ν) * U0_new + λ * I_s) / base)
                            # OJS surplus (eq SS1): λ_S redraw → shock integral; the
                            # poaching gain and outflow hazard → OFFER tail.
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
# Stationary composition — per-aS unit shapes (û, ê)
#
# For a d = 0 type of skilled aptitude aS with trained mass 1, solve the
# flow balance for the employed density ê(aS,p) and unemployed fraction
# û(aS) (notes eq:uSbalance–eq:eSbalance).  Decompose ê(p) = α(p) û +
# β(p) and close with 1 = û + ∫ ê dp.  Actual densities scale linearly
# with the type's true mass m_S(aU,aS), applied in aggregation.
# ---------------------------------------------------------------------------
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

                # κ·γ terms read the OFFER density; both λ·γ terms read the SHOCK
                # density (the −λ·γ outflow from the α-basis and the +λ·γ inflow to
                # the β-basis are the same redraw event, so they must share γ_s and
                # net to zero mass at δ = 1).
                # A poached worker must draw a cell STRICTLY ABOVE her own: the
                # poaching inflow to a cell comes from the searchers below it
                # (CumAlpha/CumBeta are accumulated after cell j is priced), so the
                # offsetting outflow is the offer mass above cell j — tail_weights
                # shifted by one.  With it, ∫inflow = ∫outflow exactly and û is the
                # u_S balance's own ratio; the continuous CDF Γ_o(p_j) left û 0.65%
                # above that ratio, and tail_weights[j] counts the source cell in
                # both directions and is 1.39% off.
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


# ---------------------------------------------------------------------------
# Non-draining column mass of the trained population
#   mcol0[j] = Σ_i (1 − d[i,j]) m_S[i,j]
# The 2D→1D reduction that makes free-entry aggregation O(N²).  Linear in d, so
# a fractional drain splits each cell's mass rather than assigning it wholly.
# ---------------------------------------------------------------------------
function nondrain_column_mass(d::AbstractMatrix{Float64}, mS::AbstractMatrix{Float64})
    Nx = size(mS, 2)
    mcol0 = zeros(Float64, Nx)
    @inbounds for j in 1:Nx, i in 1:Nx
        mcol0[j] += (1.0 - d[i, j]) * mS[i, j]
    end
    return mcol0
end

# ---------------------------------------------------------------------------
# The skilled seekers' composition by aS: the non-draining column mass or, when the
# market holds none, its limit as the market empties — the admissible cells with the
# largest training gain (marginal_training_cells), carried through the model's own
# m_S(t) so the drain split weights them exactly as it would a small positive mass.
# ---------------------------------------------------------------------------
function skilled_stayer_columns(model::Model, d::AbstractMatrix{Float64},
                                mS::AbstractMatrix{Float64})
    mcol0 = nondrain_column_mass(d, mS)
    sum(mcol0) > EMPTY_MASS && return mcol0
    cp = model.common
    w  = marginal_training_cells(model; entering = true, admit = d .< 1.0)
    return nondrain_column_mass(d, _mS_from_t(w, crossing_rate(model, d), cp.φ, cp.ν))
end


# ---------------------------------------------------------------------------
# Expected firm value seen by a randomly arriving skilled vacancy.
#
# Seekers are non-draining unemployed (mass û(aS) mcol0[aS]) and employed
# searchers s*(aS,p) with density ê(aS,p) mcol0[aS].  Firm value tails
# are 1D in aS; the aU-dependence is entirely in mcol0.
#
# Which of J^1/J^0 applies, and whether an employed type searches at all, is
# the same OJS margin solve_stationary_skilled! smooths with _soft_oj_weight,
# so free entry uses that weight too: reading the margin as a hard 1{E¹ ≥ E⁰}
# here made J̄_S jump whenever a single grid cell flipped, leaving the outer
# θ_S map discontinuous in the parameters while the block feeding it was
# continuous.
#
# SHIPPED FOR CONTINUITY, NOT ACCURACY.  Against the hard rule the soft weight
# is 1.7–1.8× worse in RMS and carries a one-signed bias about 60× larger,
# because _soft_oj_weight returns the covered fraction of the INTERVAL
# [p_j, p_{j+1}] while it multiplies the Gauss–Legendre weight wp_j, which is
# not that interval's length.  What it buys is that all 8–16 discontinuities
# per point leave the free-entry map, taking pooled feasibility from 58.0% to
# 94.3% (n = 192 paired directions per radius, 2304 solves, McNemar
# p = 3.9e-45).  An exact version would weight by the cell's MASS fraction
# under dΓ rather than by its length fraction.
#
# The RESERVATION margin enters through the index the firm-value tail is read
# at.  J^0/J^1 already carry the reservation coverage weight ω, so the tail is
# read from one cell BELOW the cutoff node — j0_soft, the index
# skilled_inner_loop! and solve_stationary_skilled! both use — or the cell
# straddling p*_S is dropped from free entry while the block that priced it
# keeps it.
# ---------------------------------------------------------------------------
function compute_Jbar_skilled(model::Model)
    gp = model.grids;  sg = model.skl_grids;  pre = model.skl_pre;  sc = model.skl_cache
    Nx = length(gp.x);  Np = length(sg.p)
    wΓ = pre.γvals .* sg.wp
    mcol0 = model.corner_limit ? skilled_stayer_columns(model, sc.d, sc.m_S) :
                                 nondrain_column_mass(sc.d, sc.m_S)

    # Accumulated PER aS NODE and reduced in index order rather than into thread-local
    # slots: @threads schedules dynamically, so a threadid()-indexed sum groups the terms
    # differently from run to run and J̄_S differs in its last bits.  That is invisible at
    # a well-conditioned point and decisive at a marginal one, because U_S = U_S(θ_S) has
    # a sensitivity of ~2e4 and the global residual is a sup-norm on U_S — which made the
    # feasible/infeasible verdict itself depend on the schedule.
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

            # An unemployed seeker is priced over every acceptable destination; an
            # employed one only over destinations above her own quality, which the
            # reservation floor still binds from below.
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

    # Early-iteration fallback: before m_S has settled, weight the firm-value
    # tails by the non-draining marginal ℓ so free entry gets a non-trivial signal.
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


# ---------------------------------------------------------------------------
# Free entry in the skilled market: the tightness a vacancy cost implies, and
# its inverse.
#   q_S(θ_S) = k_S π̄_S / J̄_S   (k_S in months of average skilled output)
# ---------------------------------------------------------------------------
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

The vacancy cost free entry implies at the installed tightness and composition, the
inverse of `update_theta_skilled`: k_S = q_S(θ_S) J̄_S / π̄_S.  Zero where a meeting is
worth nothing.
"""
function vacancy_cost_skilled(model::Model)
    sp = model.skl_par
    Jbar = compute_Jbar_skilled(model)
    (Jbar < 1e-12 || !isfinite(Jbar)) && return 0.0
    return q_from_theta(model.skl_cache.θ, sp.μ, sp.η) * Jbar / mean_output_S(model)
end


# ---------------------------------------------------------------------------
# Outer loop on (θ_S, p*_S, p^oj_S)
#
# Per iteration: inner values (updates d) → cutoffs from raw surpluses, installed
# under-relaxed by damp_pstar_S → per-aS stationary shapes → free-entry tightness,
# under-relaxed by the adaptive weight.  A plain damped fixed-point iteration with no
# acceleration: an Anderson step on the joint [θ_S; p*_S; gap] takes its mixing weight
# mostly from the cutoff components, which in the base_fc multiplicity band reverses the
# θ_S step on about half the passes and can carry the iteration onto the repelling root
# (SETTINGS.md).  Convergence on (Δθ_S, Δp*_S, Δp^oj_S, Δs_part), Δp^oj_S over the types
# that participate in OJS, with the outlier allowance below.
# ---------------------------------------------------------------------------

# How many ability types the convergence test will treat as outliers before falling back
# to the plain sup-norm.  A CONSTANT, not a setting: at 1 or 2 the boundary has a few
# types whose increment amplifies while their level is stable, which is the case the
# allowance is for; letting a run raise it would let a genuinely drifting frontier pass.
const CONV_S_MAXOUT = 2
# ---------------------------------------------------------------------------
# The cutoff map: the cutoffs the stored surpluses imply.  p*_S is the linearly
# interpolated zero of the raw S^max, p^oj_S a crossing of S^1 − S^0 above it, and Mscore
# the participation score.  A function of the stored state alone, so the block's
# iteration and the equilibrium residual (global_residuals) read one map.
# ---------------------------------------------------------------------------
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

            # Same search rule as the value block and the firm side.  The cutoffs are
            # located FROM these tails, so a pointwise max here would search for a new
            # boundary under a rule the rest of the block does not use.
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
                # λ_S redraw → shock integral; OJS poaching gain/outflow → offer tail.
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

            # The participation score and the node attaining it.  As M falls to 0 the
            # searching interval {p : diff > 0} collapses onto that peak, so the peak
            # is the limit of the crossing — and it is the cutoff the searching branch
            # of a straddling cell must use, where the scan, finding no crossing at
            # all, would otherwise shut the margin outright and jump p^oj by cells.
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


# `θS_fixed` imposes the skilled tightness instead of solving free entry for it: the block
# then solves every other skilled object conditional on that θ_S, which is one evaluation
# of the skilled free-entry map (skilled_equilibria, solver.jl).
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

    # Adaptive free-entry weight, and the previous pass's interpolated boundary values for
    # the convergence test.  Both are LOCAL to this call: carrying either between solves
    # in a module-level Ref would make a result depend on the order in which parameter
    # draws are evaluated, which is the defect class of the thread-nondeterminism bug.
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

        # Cutoffs from raw (unclamped) surpluses.
        cutoff_proposals!(pst_prop, poj_prop, Mscore, model, κ)

        # Type-level OJS participation.  A function of the current iterate, like the cutoff
        # proposals, and installed undamped, so that :indicator and :fraction differ only in
        # the participation encoding.
        OJS_PARTICIPATION[] === :fraction && ojs_participation!(part_prop, Mscore, gp.x)
        copyto!(sc.s_part, part_prop)

        # Install the under-relaxed cutoffs for the stationary solve; p^oj_S is kept at or
        # above p*_S after the blend.
        damp = sim.damp_pstar_S
        @inbounds for k in 1:Nx
            sc.pstar[k] = clamp01(damp * pst_prop[k] + (1.0 - damp) * pstar_old[k])
            sc.poj[k]   = max(sc.pstar[k], clamp01(damp * poj_prop[k] + (1.0 - damp) * poj_old[k]))
        end

        solve_stationary_skilled!(model)

        # Under-relaxed free entry (DAMP_THETA_S in params.jl).
        #
        # The weight ADAPTS, with DAMP_THETA_S as the ceiling it relaxes back toward: on a
        # sign reversal in the undamped target the step is halved, and absent one it
        # lengthens back.
        #
        # What the shorter step does is NOT stabilise the oscillating root.  This map's
        # slope in θ_S is (1/η_S)·dlnJ̄_S/dlnθ_S, so it contracts iff dlnJ̄_S/dlnθ_S < η_S
        # = 0.5; on the ridge that quantity measures 0.52 to 0.90, i.e. slope b = 1.04 to
        # 1.80, and the DAMPED slope is 1 + w(b−1) > 1 for every w in (0,1].  No
        # under-relaxation can stabilise a root with b > 1.  What a shorter step does there
        # is let the iteration be repelled from the unstable root and settle on a different,
        # stable one — so on the multiplicity band the schedule SELECTS an equilibrium.
        #
        # The fixed 0.9 is correct where the estimates live: at θ̂ the skilled free-entry
        # residual contracts geometrically, 2.1e-04 → 3.6e-07 → 3.2e-08, no reversal occurs
        # and the weight simply stays at the ceiling.  θ̂ and the controls land on the SAME
        # root under w = 0.9 and w = 0.5 (Q agreeing to 2.1e-11), so no reported estimate is
        # branch-dependent.  A FIXED 0.5 instead of the ceiling costs coverage (348 of 361
        # against 353 at the fixed 0.9); see SETTINGS.md on DAMP_THETA_S for the band.
        #
        # With θ_S imposed there is no free-entry step: θ_S is held at θS_fixed on every
        # pass and only the cutoffs iterate.
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

        # A type with zero OJS participation before and after the pass is INACTIVE, and its
        # p^oj_S enters no object: every consumer weights it by _ojs_share = s_part × weight
        # (grids.jl), which is exactly 0 there.  Its p^oj_S is then the peak of S^1 − S^0,
        # which for high ability lies on a plateau at the top of the p grid, flat to 1e-12
        # relative, so rounding moves it between nodes by far more than tol.  Its Δp^oj_S
        # therefore counts as zero, as in R1 (global_residuals); its Δp*_S and Δs_part still
        # count, so a type switching participation on or off registers through Δs_part.
        inactive = [part_old[k] == 0.0 && sc.s_part[k] == 0.0 for k in 1:Nx]
        Δoj = [inactive[k] ? 0.0 : abs(sc.poj[k] - poj_old[k]) for k in 1:Nx]

        dθ = abs(sc.θ - θ_old)
        dp = supnorm(sc.pstar, pstar_old)
        dj = maximum(Δoj)
        ds = supnorm(sc.s_part, part_old)        # identically 0 under :indicator
        d  = max(dθ, dp, dj, ds)

        # Outlier allowance on the boundary.  The plain sup-norm lets ONE ability type at
        # the participation frontier veto the other 159: measured there, that type's p^oj
        # LEVEL is stable to five decimals while its per-iteration INCREMENT amplifies, so
        # the sup-norm reports divergence for a boundary that has already settled.  Where
        # at most CONV_S_MAXOUT types are above tolerance and the rest are below it, each
        # outlier's cutoff is replaced by a linear interpolation in ability between its
        # anchors, that interpolated value is required to be stable from one pass to the
        # next, and convergence is judged on the sup-norm over the remaining types.  Any
        # other count falls back to the plain sup-norm.
        #
        # The interpolation decides CONVERGENCE ONLY and never writes sc.poj: installing it
        # would mean the moments are computed from a cutoff no equilibrium condition was
        # ever checked against.  The anchors are the nearest in-tolerance ACTIVE types on
        # each side: an inactive type's p^oj_S is not an equilibrium object, so the walk
        # steps over it.  A side with no anchor is the edge of the region where p^oj_S is
        # defined — the grid edge, or the participation frontier with only inactive types
        # beyond it, which is where the frontier type above sits — and the interpolant is
        # then the anchor on the other side.  With no anchor on either side the test falls
        # back rather than anchoring on the outlier itself.  dθ is a scalar and never the
        # binding component, so it always stands.
        #
        # The stability test matches the previous pass's interpolants by position, so an
        # outlier SET that changes between passes while keeping its size is compared
        # type-against-type; the global residual battery is the backstop that catches a
        # block which exits on that (solver.jl, global_residuals).
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
            # d is the gated residual and nout the number of types the allowance excused,
            # so d sits BELOW the component maxima whenever nout > 0.
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

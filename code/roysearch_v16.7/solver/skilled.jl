############################################################
# skilled.jl — Skilled block solver (RoySearch)
#
# Every skilled VALUE object reads the skilled aptitude aS alone (notes
# §382, §420): U_S^(0), E_S^0, E_S^1, J_S^0, J_S^1, p*_S, p^oj_S, I_S.
# Production is linear, π_S(aS,p) = A P_S aS p (no γ_S).
#
# The directed-search choice reads DIFFERENT abilities on its two
# branches (notes eq:dpolicy):
#   U_S^(0)(aS) = (b_S + κ_S β_S I_S(aS)) / (r + ν)          stay skilled
#   U_S^(1)(aU) = (b_S + f_U E_U(aU,1)) / (r + ν + f_U)      cross to U
#   d(aU,aS)    = 1{ U_S^(1)(aU) > U_S^(0)(aS) }             (2D)
# The employed surplus uses U_S^(0)(aS) as the outside option: on the
# d = 0 region (where all employed mass lives) this IS the skilled
# unemployment value, and d = 1 cells carry no employed mass, so their
# 1D-in-aS surplus is never read by the distribution.
#
# Stationary composition factors.  The p-dynamics read only aS, and the
# flow balance is linear-homogeneous in a type's trained mass, so the
# split into unemployed/employed is a per-aS UNIT shape scaled by the 2D
# trained mass m_S(aU,aS):
#   d = 0:  u_S = û(aS) m_S,  e_S(p) = ê(aS,p) m_S
#   d = 1:  u_S = m_S,        e_S ≡ 0        (these seek in U, not S)
# where m_S(aU,aS) = φ t(aU,aS) / (ν + d f_U) is supplied by the global
# loop.  Free-entry aggregation reduces to a 1D sum over aS weighted by
# the d = 0 column masses of m_S.
#
# Functions
#   build_skilled_precomp        Γ CDF/PDF + tail weights
#   find_cutoff_from_j0          zero-crossing of S^max(aS,·)
#   find_poj_from_diff_grid      zero-crossing of S^1 − S^0
#   skilled_inner_loop!          iterate (U0, U1, S^0, S^1, d)
#   solve_stationary_skilled!    per-aS unit shapes (û, ê)
#   compute_Jbar_skilled         expected firm value for free entry
#   update_theta_skilled         free-entry tightness update
#   solve_skilled_block!         outer loop on (θ_S, p*_S, p^oj_S)
############################################################


# ---------------------------------------------------------------------------
# Precompute Γ CDF/PDF and tail weights on the skilled p-grid
# ---------------------------------------------------------------------------
function build_skilled_precomp(sg::SkilledGrids, sp::SkilledParams)
    dist  = Beta(sp.a_Γ, sp.b_Γ)

    # Offer distribution Γ_o = Beta(a_Γ, b_Γ) on [0,1].
    Γvals = cdf.(dist, sg.p)
    γvals = pdf.(dist, sg.p)
    wΓ    = γvals .* sg.wp

    # Shock distribution Γ_s(x) = Γ_o(x/δ) on [0,δ]: the shocked quality is
    # δ·(offer draw), so the CDF evaluates the offer at x/δ and the density
    # carries the 1/δ Jacobian, with zero mass above δ.  At δ = 1 the map is
    # the identity and these arrays reproduce the offer arrays exactly.
    δ      = sp.δ
    Γs_vals = cdf.(dist, clamp.(sg.p ./ δ, 0.0, 1.0))
    γs_vals = [p <= δ ? pdf(dist, p / δ) / δ : 0.0 for p in sg.p]
    wΓs     = γs_vals .* sg.wp

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
# Locate p^oj_S(aS) by linear interpolation of the zero-crossing of
# S^1 − S^0 above p*_S.
# ---------------------------------------------------------------------------
function find_poj_from_diff_grid(pgrid::Vector{Float64}, diff::AbstractVector{Float64}, pstar::Float64)
    Np = length(pgrid)
    j0 = pcut_index(pgrid, pstar)
    @inbounds for j in j0:Np
        if diff[j] <= 0.0
            if j > j0 && diff[j-1] > 0.0
                d_hi = diff[j-1];  d_lo = diff[j];  dD = d_hi - d_lo
                abs(dD) < 1e-14 && return pgrid[j]
                return pgrid[j-1] + clamp(d_hi / dD, 0.0, 1.0) * (pgrid[j] - pgrid[j-1])
            end
            return pgrid[j]
        end
    end
    return 1.0
end


# ---------------------------------------------------------------------------
# Inner loop
#
# Given (θ_S, p*_S, p^oj_S) fixed, iterate on the surplus surfaces
# (S^0_S, S^1_S) and the stay-skilled unemployment value U_S^(0)(aS)
# until convergence.  Also form the cross branch U_S^(1)(aU) and the 2D
# directed-search policy d(aU,aS) — a by-product of the value iteration,
# no separate fixed point.
# ---------------------------------------------------------------------------
function skilled_inner_loop!(model::Model; fU::Float64, EU1::AbstractVector{Float64})
    cp = model.common;  gp = model.grids;  sp = model.skl_par
    sg = model.skl_grids;  pre = model.skl_pre;  sc = model.skl_cache;  sim = model.sim

    Nx = length(gp.x);  Np = length(sg.p)
    r  = cp.r;  ν = cp.ν
    PS = exp(cp.A) * sp.PS;  bS = sp.bS * exp(cp.A)
    β  = sp.β;  λ = sp.λ;  σ = sp.σ * exp(cp.A);  ξ = sp.ξ
    μ  = sp.μ;  η = sp.η

    f   = jobfinding_rate(sc.θ, μ, η)              # κ_S
    wΓo = pre.γvals   .* sg.wp                      # offer  ∫·dΓ_o
    wΓs = pre.γs_vals .* sg.wp                      # shock  ∫·dΓ_s

    tailEo_tls = [zeros(Float64, Np) for _ in 1:Threads.nthreads()]   # offer tail
    tailEs_tls = [zeros(Float64, Np) for _ in 1:Threads.nthreads()]   # shock tail
    dU_tls     = zeros(Float64, Threads.nthreads())
    dS_tls     = zeros(Float64, Threads.nthreads())

    αS = sim.damp_inner_S
    B  = sim.inner_B;  K = sim.inner_K
    resid = Vector{Float64}(undef, sim.maxit_inner)
    status = :maxit;  n_inner = 0;  streak = 0
    denom_nb = max(1.0 - β, 1e-14)

    # Cross branch U_S^(1)(aU) is fixed within the block (reads f_U, E_U(aU,1)).
    @inbounds for i in 1:Nx
        sc.U1[i] = (bS + fU * EU1[i]) / (r + ν + fU)
    end

    for it in 1:sim.maxit_inner
        fill!(dU_tls, 0.0);  fill!(dS_tls, 0.0)

        @threads for k in 1:Nx                     # k indexes the skilled aptitude aS
            tid    = Threads.threadid()
            tailEo = tailEo_tls[tid]               # offer tail (poaching, unemployed search)
            tailEs = tailEs_tls[tid]               # shock tail (λ_S redraw option value)
            @inbounds begin
                aS      = gp.x[k]
                PSeff   = PS * aS                  # A P_S aS (linear loading)
                pstar   = clamp01(sc.pstar[k])
                j0      = pcut_index(sg.p, pstar)
                j0_soft = max(j0 - 1, 1)

                # Tail integrals I_S(aS) = ∫_{p*}^1 S^max dΓ, one per distribution.
                # Stored J0/J1 carry the soft weight, so /(1−β) recovers ω·S.
                acco = 0.0;  accs = 0.0
                for j in Np:-1:1
                    Smax_j     = max(sc.J0[k, j], sc.J1[k, j]) / denom_nb
                    acco      += Smax_j * wΓo[j];  tailEo[j] = acco
                    accs      += Smax_j * wΓs[j];  tailEs[j] = accs
                end
                I_o = tailEo[j0_soft]              # offer: fresh-meeting search option
                I_s = tailEs[j0_soft]              # shock: λ_S redraw option

                # Stay-skilled unemployment value (self-referential via the
                # OFFER integral — an unemployed worker meets fresh offers).
                U0_raw   = (bS + f * β * I_o) / (r + ν)
                U0_old   = sc.U0[k]
                U0_new   = U0_old + αS * (U0_raw - U0_old)
                sc.U0[k] = U0_new
                dU_tls[tid] = max(dU_tls[tid], abs(U0_new - U0_old))

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
                                     f * β * tail_Emax_j) / (base + f * tail_mass_j))

                    E0_old = sc.E0[k, j];  E1_old = sc.E1[k, j]
                    sc.E0[k, j] = U0_new + β * ω_j * S0
                    sc.E1[k, j] = U0_new + β * ω_j * S1
                    sc.J0[k, j] = (1.0 - β) * ω_j * S0
                    sc.J1[k, j] = (1.0 - β) * ω_j * S1
                    dS_tls[tid] = max(dS_tls[tid], abs(sc.E0[k,j]-E0_old), abs(sc.E1[k,j]-E1_old))
                end
            end
        end

        d = max(maximum(dU_tls), maximum(dS_tls))
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

    # Directed-search policy d(aU,aS) = 1{U1(aU) > U0(aS)} and U_S = max.
    @threads for j in 1:Nx                         # aS column
        @inbounds begin
            U0j = sc.U0[j]
            for i in 1:Nx                          # aU row
                sc.d[i, j] = (sc.U1[i] > U0j) ? 1.0 : 0.0
            end
            sc.U[j] = U0j                          # employed outside option (1D in aS)
        end
    end

    return (f = f, status = status, converged = (status === :converged), iters = n_inner)
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
    f = jobfinding_rate(sc.θ, sp.μ, sp.η)

    @threads for k in 1:Nx
        @inbounds begin
            pstar   = clamp01(sc.pstar[k])
            poj     = clamp01(sc.poj[k])
            j0      = pcut_index(sg.p, pstar)
            j0_soft = max(j0 - 1, 1)

            α_coef = zeros(Float64, Np)
            β_coef = zeros(Float64, Np)
            ω_arr  = zeros(Float64, Np)
            CumAlpha = 0.0;  CumBeta = 0.0

            for j in j0_soft:Np
                pj  = sg.p[j];  wpj = sg.wp[j]
                γoj = pre.γvals[j];   Γoj = pre.Γvals[j]    # offer: hire + poaching inflow, poaching outflow
                γsj = pre.γs_vals[j]                        # shock: λ_S redraw in/outflow
                ω_j = _soft_weight(pj, pstar, sg.p, j, Np)
                ω_arr[j] = ω_j
                ω_j <= 0.0 && continue
                s_j = _soft_oj_weight(pj, poj, sg.p, j, Np)

                # f·γ terms read the OFFER density; both λ·γ terms read the SHOCK
                # density (the −λ·γ outflow from the α-basis and the +λ·γ inflow to
                # the β-basis are the same redraw event, so they must share γ_s and
                # net to zero mass at δ = 1).
                a_j = ν + λ + ξ + s_j * f * (1.0 - Γoj)
                num_α = (f * γoj - λ * γsj) + f * γoj * CumAlpha   # unit-mass forcing: κ û γ term
                num_β = (λ * γsj) * 1.0     + f * γoj * CumBeta    # m_S = 1 in the unit problem
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
# d = 0 column mass of the trained population
#   mcol0[j] = Σ_i (1 − d[i,j]) m_S[i,j]
# The 2D→1D reduction that makes free-entry aggregation O(N²).
# ---------------------------------------------------------------------------
function dzero_column_mass(sc::SkilledCache, Nx::Int)
    mcol0 = zeros(Float64, Nx)
    @inbounds for j in 1:Nx, i in 1:Nx
        mcol0[j] += (1.0 - sc.d[i, j]) * sc.m_S[i, j]
    end
    return mcol0
end


# ---------------------------------------------------------------------------
# Expected firm value seen by a randomly arriving skilled vacancy.
#
# Seekers are d = 0 unemployed (mass û(aS) mcol0[aS]) and employed
# searchers s*(aS,p) with density ê(aS,p) mcol0[aS].  Firm value tails
# are 1D in aS; the aU-dependence is entirely in mcol0.
# ---------------------------------------------------------------------------
function compute_Jbar_skilled(model::Model)
    gp = model.grids;  sg = model.skl_grids;  pre = model.skl_pre;  sc = model.skl_cache
    Nx = length(gp.x);  Np = length(sg.p)
    wΓ = pre.γvals .* sg.wp
    mcol0 = dzero_column_mass(sc, Nx)

    num_tls = zeros(Float64, Threads.nthreads())
    den_tls = zeros(Float64, Threads.nthreads())

    @threads for k in 1:Nx
        tid = Threads.threadid()
        @inbounds begin
            mk = mcol0[k]
            mk <= 0.0 && continue
            pstar = clamp01(sc.pstar[k])
            j0    = pcut_index(sg.p, pstar)

            tailJ = zeros(Float64, Np)
            acc = 0.0
            for j in Np:-1:1
                J   = sc.E1[k, j] >= sc.E0[k, j] ? sc.J1[k, j] : sc.J0[k, j]
                acc += J * wΓ[j]
                tailJ[j] = acc
            end

            seeker_e = 0.0
            num_k    = sc.u_frac[k] * tailJ[j0]
            for j in 1:Np
                if sc.E1[k, j] >= sc.E0[k, j]
                    seeker_e += sc.e_frac[k, j] * sg.wp[j]
                    num_k    += sc.e_frac[k, j] * sg.wp[j] * tailJ[max(j, j0)]
                end
            end
            den_tls[tid] += mk * (sc.u_frac[k] + seeker_e)
            num_tls[tid] += mk * num_k
        end
    end

    num = sum(num_tls);  den = max(sum(den_tls), 1e-14)

    # Early-iteration fallback: before m_S has settled, weight the firm-value
    # tails by the d = 0 marginal ℓ so free entry gets a non-trivial signal.
    if num < 1e-12
        num = 0.0;  den = 0.0
        colmass = vec(sum(gp.copula.W2, dims = 1))         # aS marginal
        for k in 1:Nx
            frac0 = 0.0
            for i in 1:Nx
                frac0 += (1.0 - sc.d[i, k]) * gp.copula.W2[i, k]
            end
            ell = frac0
            j0  = pcut_index(sg.p, clamp01(sc.pstar[k]))
            acc = 0.0
            for j in Np:-1:j0
                J = sc.E1[k,j] >= sc.E0[k,j] ? sc.J1[k,j] : sc.J0[k,j]
                acc += J * wΓ[j]
            end
            num += ell * acc
            den += ell
        end
        den = max(den, 1e-14)
    end

    return num / den
end


# ---------------------------------------------------------------------------
# Free-entry tightness update for the skilled market.
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


# ---------------------------------------------------------------------------
# Outer loop on (θ_S, p*_S, p^oj_S)
#
# Per iteration: inner values (updates d) → cutoffs from raw surpluses →
# per-aS stationary shapes → free-entry tightness → joint Anderson(m=1)
# on [θ_S; p*_S; gap] with gap = p^oj − p*.  Convergence on
# (Δθ_S, Δp*_S, Δp^oj_S).
# ---------------------------------------------------------------------------
function solve_skilled_block!(model::Model; fU::Float64, EU1::AbstractVector{Float64})
    cp = model.common;  gp = model.grids;  sg = model.skl_grids
    sc = model.skl_cache;  sim = model.sim;  sp = model.skl_par;  pre = model.skl_pre

    Nx = length(gp.x);  Np = length(sg.p)
    denom_nb = max(1.0 - sp.β, 1e-14)
    r  = cp.r;  ν = cp.ν
    PS = exp(cp.A) * sp.PS
    β  = sp.β;  λ = sp.λ;  σ = sp.σ * exp(cp.A);  ξ = sp.ξ
    wΓo = pre.γvals   .* sg.wp                      # offer  ∫·dΓ_o
    wΓs = pre.γs_vals .* sg.wp                      # shock  ∫·dΓ_s

    aa_joint = Anderson1(1 + 2 * Nx)
    pst_prop = zeros(Float64, Nx)
    poj_prop = zeros(Float64, Nx)

    streak = 0;  diverge_streak = 0
    oB = sim.outer_B;  oK = sim.outer_K
    outer_resid = Vector{Float64}(undef, sim.maxit_outer)

    for it in 1:sim.maxit_outer
        θ_old     = sc.θ
        pstar_old = copy(sc.pstar)
        poj_old   = copy(sc.poj)

        inner = skilled_inner_loop!(model; fU = fU, EU1 = EU1)
        f = inner.f;  inner_conv = inner.converged

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
        @threads for k in 1:Nx
            @inbounds begin
                aS = gp.x[k];  PSeff = PS * aS;  U0k = sc.U0[k]
                j0_prev = pcut_index(sg.p, clamp01(sc.pstar[k]))
                j0_soft = max(j0_prev - 1, 1)

                tailEo = zeros(Float64, Np);  tailEs = zeros(Float64, Np)
                acco = 0.0;  accs = 0.0
                for j in Np:-1:1
                    Smax_j = max(sc.J0[k, j], sc.J1[k, j]) / denom_nb
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
                              f * β * tail_Emax_j) / (base + f * tail_mass_j)
                    Smax_raw[j] = max(raw_S0, raw_S1)
                    diff_raw[j] = raw_S1 - raw_S0
                end
                pst_prop[k] = clamp01(find_cutoff_from_j0(sg.p, Smax_raw, j0_prev))
                poj_prop[k] = max(pst_prop[k], clamp01(find_poj_from_diff_grid(sg.p, diff_raw, pst_prop[k])))
            end
        end

        # Install cutoffs for the stationary solve.
        if sim.use_anderson
            @inbounds for k in 1:Nx
                sc.pstar[k] = pst_prop[k]
                sc.poj[k]   = max(pst_prop[k], poj_prop[k])
            end
        else
            damp = sim.damp_pstar_S
            @inbounds for k in 1:Nx
                sc.pstar[k] = clamp01(damp * pst_prop[k] + (1.0 - damp) * pstar_old[k])
                sc.poj[k]   = max(sc.pstar[k], clamp01(damp * poj_prop[k] + (1.0 - damp) * poj_old[k]))
            end
        end

        solve_stationary_skilled!(model)
        θ_raw = update_theta_skilled(model)

        if sim.use_anderson
            gap_old  = poj_old  .- pstar_old
            gap_prop = poj_prop .- pst_prop
            x_old = vcat([θ_old], pstar_old, gap_old)
            f_raw = vcat([θ_raw], pst_prop,  gap_prop)
            x_new = anderson1_update!(aa_joint, x_old, f_raw)
            sc.θ = max(x_new[1], 1e-14)
            @inbounds for k in 1:Nx
                pstar_new   = clamp01(x_new[1 + k])
                gap_new     = clamp(x_new[1 + Nx + k], 0.0, 1.0)
                sc.pstar[k] = pstar_new
                sc.poj[k]   = clamp01(pstar_new + gap_new)
            end
        else
            sc.θ = θ_raw
        end

        if !isfinite(sc.θ) || any(!isfinite, sc.pstar) || any(!isfinite, sc.poj)
            sim.verbose >= 1 && @printf("  [outer S]  NaN/Inf in state at it=%d — aborting\n", it)
            return (converged = false, rejected = false)
        end

        dθ = abs(sc.θ - θ_old)
        dp = supnorm(sc.pstar, pstar_old)
        dj = supnorm(sc.poj, poj_old)
        d  = max(dθ, dp, dj)
        outer_resid[it] = d

        if sim.verbose >= 2 && (it == 1 || it % sim.verbose_stride == 0)
            @printf("  [outer S it=%d]  maxΔ=%.3e  (Δθ=%.3e  Δp*=%.3e  Δpoj=%.3e)  θ_S=%.4f\n", it, d, dθ, dp, dj, sc.θ)
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

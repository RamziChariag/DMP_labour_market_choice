############################################################
# skilled.jl — Skilled block solver
#
# Outer loop iterates on (θ_S, p*_S, p^oj_S).  The inner loop
# settles the surplus surfaces (S^0_S, S^1_S), the unemployment
# value U_S, and the cross-market policy d(x).  Inputs from the
# unskilled block: f_U, E_U(·, 1), m_S(·).
#
# Unemployment Bellman:
#   U^(0)_S(x) = (b_S + κ_S β_S I_S(x)) / (r + ν)
#   U^(1)_S(x) = (b_S + f_U E_U(x, 1)) / (r + ν + f_U)
#   d(x)       = 1{ U^(1)_S(x) > U^(0)_S(x) }
#   U_S(x)     = max{ U^(0)_S(x), U^(1)_S(x) }
#
# Stationary distribution branches on d:
#   d(x) = 0  →  standard flow-balance linear system in e_S(x, p)
#                with u_S(x) closed by m_S = u_S + ∫ e_S dp.
#   d(x) = 1  →  closed form  u_S(x) = m_S(x) = ϕ t(x) / (ν + f_U),
#                              e_S(x, p) ≡ 0.
#
# Free entry uses ũ_S = ∫ (1−d) u_S + ∫ s* e_S dp dx (d=1 unemployed
# search the U-market, not the S-market).
#
# Functions
#   build_skilled_precomp           Γ CDF/PDF + tail weights
#   find_cutoff_from_j0             zero-crossing of S^max(x, ·)
#   find_poj_from_diff_grid         zero-crossing of S^1 − S^0
#   skilled_inner_loop!             iterate (U_S, S^0, S^1, d)
#   solve_stationary_skilled_x!     stationary u_S, e_S per type
#   solve_stationary_skilled!       parallel wrapper
#   compute_Jbar_skilled            expected firm value for free entry
#   update_theta_skilled            free-entry tightness update
#   solve_skilled_block!            outer loop
############################################################


# ---------------------------------------------------------------------------
# Precompute Γ CDF/PDF and tail weights on the p-grid
# ---------------------------------------------------------------------------
function build_skilled_precomp(sg::SkilledGrids, sp::SkilledParams)
    Np   = length(sg.p)
    dist = Beta(sp.a_Γ, sp.b_Γ)

    Γvals = Vector{Float64}(undef, Np)
    γvals = Vector{Float64}(undef, Np)

    @inbounds for j in 1:Np
        Γvals[j] = cdf(dist, sg.p[j])
        γvals[j] = pdf(dist, sg.p[j])
    end

    wΓ           = γvals .* sg.wp
    tail_weights = build_tail_weights(wΓ)

    return SkilledPrecomp(Γvals, γvals, tail_weights)
end


# ---------------------------------------------------------------------------
# Locate p*_S(x) by linear interpolation of the zero-crossing of
# S^max(x, ·).  Searching from the previous cutoff index makes the
# scan O(1) per iteration once the cutoff has settled into a region.
# ---------------------------------------------------------------------------
function find_cutoff_from_j0(
    pgrid   :: Vector{Float64},
    Smax    :: AbstractVector{Float64},
    j0_prev :: Int
)
    Np = length(pgrid)

    if Smax[j0_prev] < 0.0
        j_pos = 0
        @inbounds for j in j0_prev:Np
            if Smax[j] >= 0.0
                j_pos = j
                break
            end
        end
        j_pos == 0 && return 1.0
        j_neg = max(j_pos - 1, 1)
    else
        j_neg = 0
        @inbounds for j in j0_prev:-1:1
            if Smax[j] < 0.0
                j_neg = j
                break
            end
        end
        j_neg == 0 && return 0.0
        j_pos = min(j_neg + 1, Np)
    end

    S_lo = Smax[j_neg]
    S_hi = Smax[j_pos]
    dS   = S_hi - S_lo
    if abs(dS) < 1e-14
        return pgrid[j_pos]
    end
    α    = -S_lo / dS
    α    = clamp(α, 0.0, 1.0)
    return pgrid[j_neg] + α * (pgrid[j_pos] - pgrid[j_neg])
end


# ---------------------------------------------------------------------------
# Locate p^oj_S(x) by linear interpolation of the zero-crossing of
# S^1 − S^0 above p*_S.
# ---------------------------------------------------------------------------
function find_poj_from_diff_grid(
    pgrid :: Vector{Float64},
    diff  :: AbstractVector{Float64},
    pstar :: Float64
)
    Np = length(pgrid)
    j0 = pcut_index(pgrid, pstar)

    @inbounds for j in j0:Np
        if diff[j] <= 0.0
            if j > j0 && diff[j-1] > 0.0
                d_hi = diff[j-1]
                d_lo = diff[j]
                dD   = d_hi - d_lo
                if abs(dD) < 1e-14
                    return pgrid[j]
                end
                α = d_hi / dD
                α = clamp(α, 0.0, 1.0)
                return pgrid[j-1] + α * (pgrid[j] - pgrid[j-1])
            end
            return pgrid[j]
        end
    end
    return 1.0
end


# ---------------------------------------------------------------------------
# Soft thresholds around p*_S and p^oj_S.
#
# A hard cutoff at the nearest grid index makes the surplus and
# employment surfaces discontinuous in the cutoff and admits period-2
# limit cycles in the outer fixed point.  These soft weights give a
# linear blend across the cell straddling the cutoff, making both
# surfaces continuous in (p*, p^oj).
# ---------------------------------------------------------------------------
@inline function _soft_weight(
    pj      :: Float64,
    pstar   :: Float64,
    pgrid   :: Vector{Float64},
    j       :: Int,
    Np      :: Int
)
    pj >= pstar && return 1.0
    j >= Np && return 0.0
    p_next = pgrid[min(j + 1, Np)]
    p_next <= pstar && return 0.0
    cell = p_next - pj
    cell < 1e-14 && return 0.0
    return clamp((p_next - pstar) / cell, 0.0, 1.0)
end

@inline function _soft_oj_weight(
    pj    :: Float64,
    poj   :: Float64,
    pgrid :: Vector{Float64},
    j     :: Int,
    Np    :: Int
)
    pj >= poj  && return 0.0
    j  >= Np   && return 1.0
    p_next = pgrid[min(j + 1, Np)]
    p_next <= poj && return 1.0
    cell = p_next - pj
    cell < 1e-14 && return 1.0
    return clamp((poj - pj) / cell, 0.0, 1.0)
end

# Smooth non-negative part:  ≈ max(x, 0) but C^∞.
#   smooth_pos(x, ε) = ½ (x + √(x² + ε²))
# Used in place of max(·, 0) on the raw surplus expressions to avoid a
# C⁰-but-not-C¹ kink in the outer fixed-point map at the grid-aligned
# sign changes.
@inline function smooth_pos(x::Float64, ε::Float64 = 1e-8)
    return 0.5 * (x + sqrt(x * x + ε * ε))
end


# ---------------------------------------------------------------------------
# Inner loop
#
# Given (θ_S, p*_S, p^oj_S), iterate jointly on the surplus surfaces
# (S^0_S, S^1_S) and the unemployment value U_S until convergence.
# The cross-market policy d(x) emerges as a by-product of the U_S
# update (no separate fixed point on d).
# ---------------------------------------------------------------------------
function skilled_inner_loop!(
    model :: Model;
    mS_in :: AbstractVector{Float64},
    fU    :: Float64,
    EU1   :: AbstractVector{Float64},
)
    cp  = model.common
    gp  = model.grids
    sp  = model.skl_par
    sg  = model.skl_grids
    pre = model.skl_pre
    sc  = model.skl_cache
    sim = model.sim

    Nx = length(gp.x)
    Np = length(sg.p)

    r  = cp.r;   ν  = cp.ν
    γPS = sp.gamma_PS;  bS = sp.bS * exp(cp.A)

    β  = sp.β;   λ  = sp.λ;   σ  = sp.σ * exp(cp.A);   ξ  = sp.ξ
    μ  = sp.μ;   η  = sp.η

    θ  = sc.θ
    f  = jobfinding_rate(θ, μ, η)   # = κ_S

    wΓ = pre.γvals .* sg.wp

    tailE_tls = [zeros(Float64, Np) for _ in 1:Threads.nthreads()]
    dU_tls    = zeros(Float64, Threads.nthreads())
    dS_tls    = zeros(Float64, Threads.nthreads())

    αS = sim.damp_inner_S
    B  = sim.inner_B
    K  = sim.inner_K
    resid   = Vector{Float64}(undef, sim.maxit_inner)
    status  = :maxit
    n_inner = 0

    streak = 0
    for it in 1:sim.maxit_inner
        fill!(dU_tls, 0.0)
        fill!(dS_tls, 0.0)

        @threads for ix in 1:Nx
            tid   = Threads.threadid()
            tailE = tailE_tls[tid]

            @inbounds begin
                x     = gp.x[ix]
                PS_x  = PS_of_x(x, γPS, exp(cp.A))
                pstar = clamp01(sc.pstar[ix])
                j0    = pcut_index(sg.p, pstar)
                j0_soft = max(j0 - 1, 1)

                # Tail integral I_S(x) of S^max under dΓ.  The stored
                # J0/J1 already carry the soft weight ω_j, so dividing
                # by (1 − β) recovers ω_j · S without an extra factor.
                denom_nb = max(1.0 - β, 1e-14)
                acc = 0.0
                for j in Np:-1:1
                    Smax_j = max(sc.J0[ix, j], sc.J1[ix, j]) / denom_nb
                    acc      += Smax_j * wΓ[j]
                    tailE[j]  = acc
                end
                I = tailE[j0_soft]

                # Two unemployment branches and the cross-market policy.
                U0 = (bS + f * β * I) / (r + ν)
                U1 = (bS + fU * EU1[ix]) / (r + ν + fU)
                if U1 > U0
                    sc.d[ix] = 1.0
                    U_raw    = U1
                else
                    sc.d[ix] = 0.0
                    U_raw    = U0
                end

                U_old        = sc.U[ix]
                U_new        = U_old + αS * (U_raw - U_old)   # inner under-relaxation
                sc.U[ix]     = U_new
                δU           = U_new - U_old
                dU_tls[tid]  = max(dU_tls[tid], abs(δU))

                # Surplus surfaces on the p-grid.
                base = r + ν + λ + ξ

                for j in 1:Np
                    pj  = sg.p[j]
                    ω_j = _soft_weight(pj, pstar, sg.p, j, Np)

                    if ω_j <= 0.0
                        sc.E0[ix, j] = U_new
                        sc.E1[ix, j] = U_new
                        sc.J0[ix, j] = 0.0
                        sc.J1[ix, j] = 0.0
                        continue
                    end

                    # No-search surplus  (eq 28).
                    S0 = smooth_pos((PS_x * x * pj - (r + ν) * U_new + λ * I) / base)

                    # OJS surplus  (eq 29).
                    tail_mass_j = pre.tail_weights[j]
                    tail_Emax_j = tailE[max(j, j0_soft)]
                    S1 = smooth_pos((PS_x * x * pj - (r + ν) * U_new - σ + λ * I +
                          f * β * tail_Emax_j) / (base + f * tail_mass_j))

                    E0_old = sc.E0[ix, j]
                    E1_old = sc.E1[ix, j]

                    sc.E0[ix, j] = U_new + β * ω_j * S0
                    sc.E1[ix, j] = U_new + β * ω_j * S1
                    sc.J0[ix, j] = (1.0 - β) * ω_j * S0
                    sc.J1[ix, j] = (1.0 - β) * ω_j * S1

                    dS = max(abs(sc.E0[ix, j] - E0_old),
                             abs(sc.E1[ix, j] - E1_old))
                    dS_tls[tid] = max(dS_tls[tid], dS)
                end
            end
        end

        d = max(maximum(dU_tls), maximum(dS_tls))
        resid[it] = d
        n_inner   = it
        if d < sim.tol_inner
            streak += 1
            if streak >= sim.conv_streak
                status = :converged
                break
            end
        else
            streak = 0
        end

        # Divergence early-abort: no contraction over the last K sweeps
        # (disabled when B == 0).  Keyed on |g_eff| ≥ 1, never on slowness.
        if B > 0 && it > B + K && resid[it] >= resid[it - K]
            status = :diverged
            break
        end
    end

    return (f = f, status = status, converged = (status === :converged), iters = n_inner)
end


# ---------------------------------------------------------------------------
# Stationary distribution — per worker type x.
#
# Branches on sc.d[ix]:
#   d = 0:  solve the linear system in {e_S(x, p_j)} from the
#           employment-by-quality balance (eq 36).  Decompose
#           e_S(x, p) = α(p) u_S + β(p), close with the accounting
#           identity m_S(x) = u_S(x) + ∫ e_S dp.
#   d = 1:  closed form  u_S(x) = m_S(x),  e_S(x, p) ≡ 0.
# ---------------------------------------------------------------------------
function solve_stationary_skilled_x!(
    ix    :: Int,
    model :: Model,
    mS_x  :: Float64,
    fU    :: Float64,
)
    sg  = model.skl_grids
    pre = model.skl_pre
    sc  = model.skl_cache
    sp  = model.skl_par
    cp  = model.common

    Np = length(sg.p)
    ν  = cp.ν;   λ  = sp.λ;   ξ  = sp.ξ

    if sc.d[ix] > 0.5
        sc.u[ix] = max(mS_x, 0.0)
        for j in 1:Np
            sc.e[ix, j] = 0.0
        end
        return nothing
    end

    θ  = sc.θ
    f  = jobfinding_rate(θ, sp.μ, sp.η)

    pstar = clamp01(sc.pstar[ix])
    poj   = clamp01(sc.poj[ix])
    j0    = pcut_index(sg.p, pstar)
    j0_soft = max(j0 - 1, 1)

    α_coef = zeros(Float64, Np)
    β_coef = zeros(Float64, Np)
    ω_arr  = zeros(Float64, Np)

    CumAlpha = 0.0
    CumBeta  = 0.0

    @inbounds for j in j0_soft:Np
        pj  = sg.p[j]
        γj  = pre.γvals[j]
        Γj  = pre.Γvals[j]
        wpj = sg.wp[j]

        ω_j = _soft_weight(pj, pstar, sg.p, j, Np)
        ω_arr[j] = ω_j
        ω_j <= 0.0 && continue

        s_j = _soft_oj_weight(pj, poj, sg.p, j, Np)

        a_j = ν + λ + ξ + s_j * f * (1.0 - Γj)
        b_j = f * γj
        c_j = λ * γj

        num_α = (b_j - c_j) + f * γj * CumAlpha
        num_β =  c_j * mS_x  + f * γj * CumBeta

        if a_j < 1e-14
            α_coef[j] = 0.0; β_coef[j] = 0.0
        else
            α_coef[j] = num_α / a_j
            β_coef[j] = num_β / a_j
        end

        CumAlpha += ω_j * s_j * α_coef[j] * wpj
        CumBeta  += ω_j * s_j * β_coef[j] * wpj
    end

    sum_α_wp = 0.0
    sum_β_wp = 0.0
    @inbounds for j in j0_soft:Np
        sum_α_wp += ω_arr[j] * α_coef[j] * sg.wp[j]
        sum_β_wp += ω_arr[j] * β_coef[j] * sg.wp[j]
    end

    denom_u  = 1.0 + sum_α_wp
    sc.u[ix] = denom_u > 1e-14 ?
               clamp((mS_x - sum_β_wp) / denom_u, 0.0, mS_x) : 0.0

    u_ix = sc.u[ix]
    @inbounds for j in 1:Np
        if j < j0_soft || ω_arr[j] <= 0.0
            sc.e[ix, j] = 0.0
        else
            sc.e[ix, j] = ω_arr[j] * max(α_coef[j] * u_ix + β_coef[j], 0.0)
        end
    end
    return nothing
end

function solve_stationary_skilled!(
    model :: Model;
    mS_in :: AbstractVector{Float64},
    fU    :: Float64,
)
    Nx = length(model.grids.x)
    @threads for ix in 1:Nx
        @inbounds solve_stationary_skilled_x!(ix, model, mS_in[ix], fU)
    end
    return nothing
end


# ---------------------------------------------------------------------------
# Expected firm value seen by a randomly arriving skilled vacancy.
#
# Seekers are:
#   - unemployed of types with d(x) = 0   →  weight (1 − d) u_S
#   - employed searchers s*(x, p) = 1     →  weight e_S(x, p)
# d = 1 unemployed are excluded — they seek in the U-market.
# Employed skilled workers never cross-market regardless of d.
# ---------------------------------------------------------------------------
function compute_Jbar_skilled(model::Model)
    gp  = model.grids
    sp  = model.skl_par
    sg  = model.skl_grids
    pre = model.skl_pre
    sc  = model.skl_cache

    Nx = length(gp.x)
    Np = length(sg.p)

    wΓ = pre.γvals .* sg.wp

    num_tls = zeros(Float64, Threads.nthreads())
    den_tls = zeros(Float64, Threads.nthreads())

    @threads for ix in 1:Nx
        tid = Threads.threadid()
        @inbounds begin
            wx    = gp.wx[ix]
            one_minus_d = 1.0 - clamp(sc.d[ix], 0.0, 1.0)
            u_eff = sc.u[ix] * one_minus_d
            pstar = clamp01(sc.pstar[ix])
            j0    = pcut_index(sg.p, pstar)

            tailJ_vec = zeros(Float64, Np)
            acc = 0.0
            for j in Np:-1:1
                J    = sc.E1[ix, j] >= sc.E0[ix, j] ? sc.J1[ix, j] : sc.J0[ix, j]
                acc       += J * wΓ[j]
                tailJ_vec[j] = acc
            end

            seeker_e = 0.0
            for j in 1:Np
                sc.E1[ix, j] >= sc.E0[ix, j] &&
                    (seeker_e += sc.e[ix, j] * sg.wp[j])
            end
            den_tls[tid] += wx * (u_eff + seeker_e)

            num_x = u_eff * tailJ_vec[j0]
            for j in 1:Np
                if sc.E1[ix, j] >= sc.E0[ix, j]
                    num_x += sc.e[ix, j] * sg.wp[j] * tailJ_vec[max(j, j0)]
                end
            end
            num_tls[tid] += wx * num_x
        end
    end

    num = sum(num_tls)
    den = max(sum(den_tls), 1e-14)

    # Early-iteration fallback: if u_S and e_S are still ≈ 0, weight
    # the firm-value tails by ℓ(x) (restricted to d = 0 types) so that
    # free entry gets a non-trivial signal before the distributions
    # have settled.
    if num < 1e-12
        num = 0.0; den = 0.0
        for ix in 1:length(gp.x)
            one_minus_d = 1.0 - clamp(sc.d[ix], 0.0, 1.0)
            wx  = gp.wx[ix]
            ell = gp.ℓ[ix] * one_minus_d
            j0  = pcut_index(sg.p, clamp01(sc.pstar[ix]))
            acc = 0.0
            for j in Np:-1:j0
                J = sc.E1[ix,j] >= sc.E0[ix,j] ? sc.J1[ix,j] : sc.J0[ix,j]
                acc += J * wΓ[j]
            end
            num += wx * ell * acc
            den += wx * ell
        end
        den = max(den, 1e-14)
    end

    return num / den
end


# ---------------------------------------------------------------------------
# Free-entry tightness update for the skilled market.
# ---------------------------------------------------------------------------
function update_theta_skilled(model::Model)
    sp   = model.skl_par
    sc   = model.skl_cache
    Jbar = compute_Jbar_skilled(model)

    (Jbar < 1e-12 || !isfinite(Jbar)) && return 1e-14

    # Dollar posting cost = k_S · π̄_S (k_S in months of average skilled output).
    q     = sp.k * mean_output_S(model) / Jbar
    θ_raw = theta_from_q(q, sp.μ, sp.η)
    return clamp(θ_raw, 1e-14, 100.0)
end


# ---------------------------------------------------------------------------
# Outer loop
#
# Per iteration:
#   1. Inner loop at the current (θ_S, p*_S, p^oj_S), which also
#      updates d(x).
#   2. Update cutoffs from the raw (unclamped) surplus surfaces, so
#      that the zero-crossings used by find_cutoff_from_j0 are not
#      destroyed by the smooth_pos clamp inside the inner loop.
#   3. Stationary distribution branched on d.
#   4. Free-entry tightness.
#   5. Joint Anderson(m = 1) on [θ_S; p*_S; gap] with gap = p^oj − p*.
#      The (p*, gap) reparameterisation folds the constraint p^oj ≥ p*
#      into the state — clamping gap to [0, 1] is a feasibility
#      projection that Anderson can see on the next iteration.
# Convergence is checked on (Δθ_S, Δp*_S, Δp^oj_S).
# ---------------------------------------------------------------------------
function solve_skilled_block!(
    model :: Model;
    mS_in :: AbstractVector{Float64},
    fU    :: Float64,
    EU1   :: AbstractVector{Float64},
)
    cp  = model.common
    gp  = model.grids
    sg  = model.skl_grids
    sc  = model.skl_cache
    sim = model.sim
    sp  = model.skl_par
    pre = model.skl_pre

    Nx       = length(gp.x)
    Np       = length(sg.p)
    denom_nb = max(1.0 - sp.β, 1e-14)

    r  = cp.r;   ν  = cp.ν
    γPS = sp.gamma_PS
    β  = sp.β;   λ  = sp.λ;   σ  = sp.σ * exp(cp.A);   ξ  = sp.ξ

    wΓ = pre.γvals .* sg.wp

    aa_joint = Anderson1(1 + 2 * Nx)

    pst_prop = zeros(Float64, Nx)
    poj_prop = zeros(Float64, Nx)

    streak = 0
    diverge_streak = 0
    oB = sim.outer_B
    oK = sim.outer_K
    outer_resid = Vector{Float64}(undef, sim.maxit_outer)
    for it in 1:sim.maxit_outer
        θ_old     = sc.θ
        pstar_old = copy(sc.pstar)
        poj_old   = copy(sc.poj)

        # 1. Inner loop.
        inner_result = skilled_inner_loop!(model;
                                            mS_in = mS_in,
                                            fU    = fU,
                                            EU1   = EU1)
        f          = inner_result.f
        inner_conv = inner_result.converged

        # Reject the parameter only if the inner map is non-contractive across
        # K consecutive outer iterations (i.e. K distinct (p*, θ) points).
        if inner_result.status === :diverged
            diverge_streak += 1
            if diverge_streak >= sim.inner_K
                sim.verbose >= 1 && @printf(
                    "  [outer S]  inner diverged ×%d at it=%d — rejecting parameter\n",
                    diverge_streak, it)
                return (converged = false, rejected = true)
            end
        else
            diverge_streak = 0
        end

        # 2. Cutoffs from raw (unclamped) surpluses.
        @threads for ix in 1:Nx
            @inbounds begin
                x = gp.x[ix]
                PS_x = PS_of_x(x, γPS, exp(cp.A))
                U_x = sc.U[ix]
                pstar_cur = clamp01(sc.pstar[ix])
                j0_prev = pcut_index(sg.p, pstar_cur)
                j0_soft = max(j0_prev - 1, 1)

                tailE = zeros(Float64, Np)
                acc = 0.0
                for j in Np:-1:1
                    Smax_j = max(sc.J0[ix, j], sc.J1[ix, j]) / denom_nb
                    acc      += Smax_j * wΓ[j]
                    tailE[j]  = acc
                end
                I = tailE[j0_soft]

                Smax_raw = zeros(Float64, Np)
                diff_raw = zeros(Float64, Np)
                base = r + ν + λ + ξ

                for j in 1:Np
                    pj = sg.p[j]
                    raw_S0 = (PS_x * x * pj - (r + ν) * U_x + λ * I) / base

                    tail_mass_j = pre.tail_weights[j]
                    tail_Emax_j = tailE[max(j, j0_soft)]
                    denom_S1 = base + f * tail_mass_j
                    raw_S1 = (PS_x * x * pj - (r + ν) * U_x - σ + λ * I +
                              f * β * tail_Emax_j) / denom_S1

                    Smax_raw[j] = max(raw_S0, raw_S1)
                    diff_raw[j] = raw_S1 - raw_S0
                end

                pst_prop[ix] = clamp01(find_cutoff_from_j0(sg.p, Smax_raw, j0_prev))
                raw_poj      = clamp01(find_poj_from_diff_grid(sg.p, diff_raw, pst_prop[ix]))
                poj_prop[ix] = max(pst_prop[ix], raw_poj)
            end
        end

        # 3. Install cutoffs for the stationary solve.
        if sim.use_anderson
            @inbounds for ix in 1:Nx
                sc.pstar[ix] = pst_prop[ix]
                sc.poj[ix]   = max(pst_prop[ix], poj_prop[ix])
            end
        else
            damp = sim.damp_pstar_S
            @inbounds for ix in 1:Nx
                sc.pstar[ix] = clamp01(
                    damp * pst_prop[ix] + (1.0 - damp) * pstar_old[ix])
                sc.poj[ix]   = max(sc.pstar[ix],
                    clamp01(damp * poj_prop[ix] + (1.0 - damp) * poj_old[ix]))
            end
        end

        solve_stationary_skilled!(model; mS_in = mS_in, fU = fU)

        # 4. Free-entry tightness.
        θ_raw = update_theta_skilled(model)

        # 5. Anderson on [θ; p*; gap].
        if sim.use_anderson
            gap_old  = poj_old  .- pstar_old
            gap_prop = poj_prop .- pst_prop

            x_old = vcat([θ_old], pstar_old, gap_old)
            f_raw = vcat([θ_raw], pst_prop,   gap_prop)
            x_new = anderson1_update!(aa_joint, x_old, f_raw)

            sc.θ = max(x_new[1], 1e-14)
            @inbounds for ix in 1:Nx
                pstar_new     = clamp01(x_new[1 + ix])
                gap_new       = clamp(x_new[1 + Nx + ix], 0.0, 1.0)
                sc.pstar[ix]  = pstar_new
                sc.poj[ix]    = clamp01(pstar_new + gap_new)
            end
        else
            sc.θ = θ_raw
        end

        if !isfinite(sc.θ) || any(!isfinite, sc.pstar) || any(!isfinite, sc.poj)
            sim.verbose >= 1 && @printf("  [outer S]  NaN/Inf in state at it=%d — aborting\n", it)
            return (converged = false, rejected = false)
        end

        # Convergence on (Δθ, Δp*, Δp^oj).
        dθ = abs(sc.θ - θ_old)
        dp = supnorm(sc.pstar, pstar_old)
        dj = supnorm(sc.poj,   poj_old)
        d  = max(dθ, dp, dj)
        outer_resid[it] = d

        if sim.verbose >= 2 && (it == 1 || it % sim.verbose_stride == 0)
            @printf("  [outer S it=%d]  maxΔ=%.3e  (Δθ=%.3e  Δp*=%.3e  Δpoj=%.3e)  θ_S=%.4f\n",
                    it, d, dθ, dp, dj, sc.θ)
        end

        if d < sim.tol_outer_S
            streak += 1
            if streak >= sim.conv_streak
                sim.verbose >= 2 && @printf(
                    "  [outer S]  converged it=%d  d=%.3e  θ_S=%.4f\n", it, d, sc.θ)
                solve_stationary_skilled!(model; mS_in = mS_in, fU = fU)
                return (converged = inner_conv, rejected = false)
            end
        else
            streak = 0
        end

        # Outer stall: no contraction over outer_K iterations (after outer_B
        # burn-in) → hand the block back to the global loop (do NOT reject; the
        # global warm-start refines it).  Disabled when outer_B == 0.
        if oB > 0 && it > oB + oK && outer_resid[it] >= outer_resid[it - oK]
            sim.verbose >= 1 && @printf(
                "  [outer S]  stalled at it=%d (no contraction over %d iters) — handing back to global\n",
                it, oK)
            solve_stationary_skilled!(model; mS_in = mS_in, fU = fU)
            return (converged = false, rejected = false)
        end
    end

    solve_stationary_skilled!(model; mS_in = mS_in, fU = fU)
    sim.verbose >= 1 && @printf("  [outer S]  maxit reached without convergence  θ_S=%.4f\n", sc.θ)
    return (converged = false, rejected = false)
end

############################################################
# skilled.jl — Skilled block solver
#
# Contains:
#   build_skilled_precomp          — CDF/PDF of Γ + tail weights
#   find_cutoff_from_j0            — pstar_S location
#   find_poj_from_diff_grid        — OJS cutoff location
#   skilled_inner_loop!            — iterate values given θ, pstar, poj
#   solve_stationary_skilled_x!    — stationary distribution, per type
#   solve_stationary_skilled!      — wrapper (all types, threaded)
#   compute_Jbar_skilled           — average firm value for free entry
#   update_theta_skilled           — free-entry tightness update
#   solve_skilled_block!           — outer loop
############################################################

# packages loaded by main.jl

# ---------------------------------------------------------------------------
# Precompute Γ CDF/PDF and tail weights on the p-grid
# ---------------------------------------------------------------------------
function build_skilled_precomp(sg::SkilledGrids, rp::RegimeParams)
    Np   = length(sg.p)
    dist = Beta(rp.a_Γ, rp.b_Γ)

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
# Locate pstar_S(x): first grid point where Smax(x, p) >= 0
# ---------------------------------------------------------------------------
function find_cutoff_from_j0(
    pgrid   :: Vector{Float64},
    Smax    :: AbstractVector{Float64},
    j0_prev :: Int
)
    Np = length(pgrid)
    if j0_prev <= 1 || Smax[j0_prev] < 0.0
        @inbounds for j in j0_prev:Np
            Smax[j] >= 0.0 && return pgrid[j]
        end
        return pgrid[Np]
    else
        return pgrid[1]
    end
end


# ---------------------------------------------------------------------------
# Locate OJS cutoff poj(x): first p >= pstar where S1(p) - S0(p) <= 0
# ---------------------------------------------------------------------------
function find_poj_from_diff_grid(
    pgrid :: Vector{Float64},
    diff  :: AbstractVector{Float64},
    pstar :: Float64
)
    Np = length(pgrid)
    j0 = pcut_index(pgrid, pstar)
    @inbounds for j in j0:Np
        diff[j] <= 0.0 && return pgrid[j]
    end
    return pgrid[Np]
end


# ---------------------------------------------------------------------------
# Skilled inner loop
#
# Given θ, pstar, poj held in sc, iterate on U_S(x) and
# surplus surfaces S0(x,p), S1(x,p).
# ---------------------------------------------------------------------------
function skilled_inner_loop!(
    model :: Model;
    mS_in :: AbstractVector{Float64}
)
    cp  = model.common
    rp  = model.regime
    gp  = model.grids
    sp  = model.skl_par
    sg  = model.skl_grids
    pre = model.skl_pre
    sc  = model.skl_cache
    sim = model.sim

    Nx = length(gp.x)
    Np = length(sg.p)

    r  = cp.r;   ν  = cp.ν
    PS = rp.PS;  bS = rp.bS

    β  = sp.β;   ξ  = sp.ξ;   λ  = sp.λ;   σ  = sp.σ
    μ  = sp.μ;   η  = sp.η

    θ  = sc.θ
    f  = jobfinding_rate(θ, μ, η)

    wΓ = pre.γvals .* sg.wp

    tailE_tls  = [zeros(Float64, Np) for _ in 1:Threads.nthreads()]
    dU_tls     = zeros(Float64, Threads.nthreads())
    dS_tls     = zeros(Float64, Threads.nthreads())
    dU_min_tls = fill( Inf, Threads.nthreads())
    dU_max_tls = fill(-Inf, Threads.nthreads())

    # McQueen–Porteus state
    span_prev_S = Inf
    ρ_pm_S      = 0.5

    streak = 0
    for it in 1:sim.maxit_inner
        fill!(dU_tls, 0.0)
        fill!(dS_tls, 0.0)
        fill!(dU_min_tls,  Inf)
        fill!(dU_max_tls, -Inf)

        @threads for ix in 1:Nx
            tid   = Threads.threadid()
            tailE = tailE_tls[tid]

            @inbounds begin
                x     = gp.x[ix]
                pstar = clamp01(sc.pstar[ix])
                j0    = pcut_index(sg.p, pstar)

                # (1) Tail integrals of Smax under dΓ
                denom_nb = max(1.0 - β, 1e-14)
                acc = 0.0
                for j in Np:-1:1
                    Smax_j = max(sc.J0[ix, j], sc.J1[ix, j]) / denom_nb
                    acc      += Smax_j * wΓ[j]
                    tailE[j]  = acc
                end
                I = tailE[j0]

                # (2) Unemployment value: (r+ν) U_S = bS + f·β·I
                U_old        = sc.U[ix]
                U_new        = (bS + f * β * I) / (r + ν)
                sc.U[ix]     = U_new
                δU           = U_new - U_old
                dU_tls[tid]     = max(dU_tls[tid], abs(δU))
                dU_min_tls[tid] = min(dU_min_tls[tid], δU)
                dU_max_tls[tid] = max(dU_max_tls[tid], δU)

                # (3) Surplus surfaces on p-grid
                base = r + ν + ξ + λ

                for j in 1:Np
                    pj = sg.p[j]

                    if j < j0
                        sc.E0[ix, j] = U_new
                        sc.E1[ix, j] = U_new
                        sc.J0[ix, j] = 0.0
                        sc.J1[ix, j] = 0.0
                        continue
                    end

                    # No-search surplus
                    S0 = (PS * x * pj - (r + ν) * U_new + λ * I) / base

                    # OJS surplus
                    tail_mass_j = pre.tail_weights[j]
                    tail_Emax_j = tailE[max(j, j0)]
                    S1 = (PS * x * pj - (r + ν) * U_new - σ + λ * I +
                          f * β * tail_Emax_j) / (base + f * tail_mass_j)

                    E0_old = sc.E0[ix, j]
                    E1_old = sc.E1[ix, j]

                    sc.E0[ix, j] = U_new + β * S0
                    sc.E1[ix, j] = U_new + β * S1
                    sc.J0[ix, j] = (1.0 - β) * S0
                    sc.J1[ix, j] = (1.0 - β) * S1

                    dS = max(abs(sc.E0[ix, j] - E0_old),
                             abs(sc.E1[ix, j] - E1_old))
                    dS_tls[tid] = max(dS_tls[tid], dS)
                end
            end
        end

        # McQueen–Porteus acceleration on U_S
        δ_min_S    = minimum(dU_min_tls)
        δ_max_S    = maximum(dU_max_tls)
        span_new_S = δ_max_S - δ_min_S
        if it >= 2 && span_prev_S > 1e-14
            ρ_pm_S = clamp(span_new_S / span_prev_S, 0.0, 0.9999)
        end
        span_prev_S = span_new_S
        if ρ_pm_S > 1e-8
            shift_S = ρ_pm_S / (1.0 - ρ_pm_S) * (δ_min_S + δ_max_S) / 2.0
            @inbounds for ix in 1:Nx
                sc.U[ix] += shift_S
            end
        end

        d = max(maximum(dU_tls), maximum(dS_tls))
        if d < sim.tol_inner
            streak += 1
            streak >= sim.conv_streak && break
        else
            streak = 0
        end
    end

    return (f = f,)
end


# ---------------------------------------------------------------------------
# Stationary distribution — per worker type x
# ---------------------------------------------------------------------------
function solve_stationary_skilled_x!(
    ix    :: Int,
    model :: Model,
    mS_x  :: Float64
)
    sg  = model.skl_grids
    pre = model.skl_pre
    sc  = model.skl_cache
    sp  = model.skl_par
    cp  = model.common

    Np = length(sg.p)
    ν  = cp.ν;   ξ  = sp.ξ;   λ  = sp.λ

    θ  = sc.θ
    f  = jobfinding_rate(θ, sp.μ, sp.η)

    pstar = clamp01(sc.pstar[ix])
    poj   = clamp01(sc.poj[ix])
    j0    = pcut_index(sg.p, pstar)

    α = zeros(Float64, Np)
    β = zeros(Float64, Np)

    CumAlpha = 0.0
    CumBeta  = 0.0

    @inbounds for j in j0:Np
        pj  = sg.p[j]
        γj  = pre.γvals[j]
        Γj  = pre.Γvals[j]
        wpj = sg.wp[j]

        s_j = (pj < poj) ? 1.0 : 0.0

        a_j = ν + ξ + λ + s_j * f * (1.0 - Γj)
        b_j = f * γj
        c_j = λ * γj

        num_α = (b_j - c_j) + f * γj * CumAlpha
        num_β =  c_j * mS_x  + f * γj * CumBeta

        if a_j < 1e-14
            α[j] = 0.0; β[j] = 0.0
        else
            α[j] = num_α / a_j
            β[j] = num_β / a_j
        end

        CumAlpha += s_j * α[j] * wpj
        CumBeta  += s_j * β[j] * wpj
    end

    sum_α_wp = 0.0
    sum_β_wp = 0.0
    @inbounds for j in j0:Np
        sum_α_wp += α[j] * sg.wp[j]
        sum_β_wp += β[j] * sg.wp[j]
    end

    denom_u  = 1.0 + sum_α_wp
    sc.u[ix] = denom_u > 1e-14 ?
               clamp((mS_x - sum_β_wp) / denom_u, 0.0, mS_x) : 0.0

    u_ix = sc.u[ix]
    @inbounds for j in 1:Np
        sc.e[ix, j] = (j < j0) ? 0.0 : max(α[j] * u_ix + β[j], 0.0)
    end
    return nothing
end

function solve_stationary_skilled!(
    model :: Model;
    mS_in :: AbstractVector{Float64}
)
    Nx = length(model.grids.x)
    @threads for ix in 1:Nx
        @inbounds solve_stationary_skilled_x!(ix, model, mS_in[ix])
    end
    return nothing
end


# ---------------------------------------------------------------------------
# Average firm value seen by a randomly arriving vacancy (free entry)
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
            u     = sc.u[ix]
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
            den_tls[tid] += wx * (u + seeker_e)

            num_x = u * tailJ_vec[j0]
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

    # Fallback: use ℓ(x) as proxy during early iterations
    if num < 1e-12
        wΓ  = pre.γvals .* sg.wp
        num = 0.0; den = 0.0
        for ix in 1:length(gp.x)
            wx  = gp.wx[ix]
            ell = gp.ℓ[ix]
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
# Free-entry tightness update for skilled market
# ---------------------------------------------------------------------------
function update_theta_skilled(model::Model)
    sp   = model.skl_par
    sc   = model.skl_cache
    Jbar = compute_Jbar_skilled(model)

    Jbar < 1e-12 && return sc.θ

    q     = sp.k / Jbar
    θ_raw = theta_from_q(q, sp.μ, sp.η)
    return clamp(θ_raw, 1e-14, 30.0)
end


# ---------------------------------------------------------------------------
# Skilled outer loop
#
# Alternates: inner-loop values → cutoff updates →
# stationary distribution → free-entry θ update.
# ---------------------------------------------------------------------------
function solve_skilled_block!(
    model :: Model;
    mS_in :: AbstractVector{Float64}
)
    gp  = model.grids
    sg  = model.skl_grids
    sc  = model.skl_cache
    sim = model.sim
    sp  = model.skl_par

    Nx       = length(gp.x)
    Np       = length(sg.p)
    denom_nb = max(1.0 - sp.β, 1e-14)

    aaθ = Anderson1(1)

    pst_prop = zeros(Float64, Nx)
    poj_prop = zeros(Float64, Nx)

    streak = 0
    for it in 1:sim.maxit_outer
        θ_old     = sc.θ
        pstar_old = copy(sc.pstar)
        poj_old   = copy(sc.poj)

        # (A) Inner loop: update values given θ, pstar, poj
        skilled_inner_loop!(model; mS_in = mS_in)

        # (B) Update cutoffs from surplus surfaces
        @threads for ix in 1:Nx
            @inbounds begin
                Smax    = zeros(Float64, Np)
                diff    = zeros(Float64, Np)
                j0_prev = pcut_index(sg.p, clamp01(sc.pstar[ix]))

                for j in 1:Np
                    S0      = sc.J0[ix, j] / denom_nb
                    S1      = sc.J1[ix, j] / denom_nb
                    Smax[j] = max(S0, S1)
                    diff[j] = S1 - S0
                end

                pst_prop[ix] = clamp01(find_cutoff_from_j0(sg.p, Smax, j0_prev))
                raw_poj      = clamp01(find_poj_from_diff_grid(sg.p, diff, pst_prop[ix]))
                poj_prop[ix] = max(pst_prop[ix], raw_poj)
            end
        end

        damp = sim.damp_pstar_S
        @inbounds for ix in 1:Nx
            sc.pstar[ix] = clamp01(damp * pst_prop[ix] + (1.0 - damp) * pstar_old[ix])
            sc.poj[ix]   = max(sc.pstar[ix],
                               clamp01(damp * poj_prop[ix] + (1.0 - damp) * poj_old[ix]))
        end

        # (C) Stationary distribution
        solve_stationary_skilled!(model; mS_in = mS_in)

        # (D) Market tightness θ (Anderson on scalar)
        θ_raw = update_theta_skilled(model)
        if sim.use_anderson
            θ_acc = anderson1_update!(aaθ, [θ_old], [θ_raw])[1]
            sc.θ  = max(θ_acc, 1e-14)
        else
            sc.θ  = θ_raw
        end

        # (E) Convergence
        dθ = abs(sc.θ - θ_old)
        dp = supnorm(sc.pstar, pstar_old)
        dj = supnorm(sc.poj,   poj_old)
        d  = dθ

        if sim.verbose >= 2 && (it == 1 || it % sim.verbose_stride == 0)
            @printf("  [outer S it=%d]  maxΔ=%.3e  (Δθ=%.3e  Δp*=%.3e  Δpoj=%.3e)  θ=%.4f\n",
                    it, d, dθ, dp, dj, sc.θ)
        end

        if d < sim.tol_outer_S
            streak += 1
            if streak >= sim.conv_streak
                sim.verbose >= 2 && @printf(
                    "  [outer S]  converged it=%d  d=%.3e  θ=%.4f\n", it, d, sc.θ)
                break
            end
        else
            streak = 0
        end
    end
    return nothing
end

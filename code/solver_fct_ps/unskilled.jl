############################################################
# unskilled.jl — Unskilled block solver
#
# Contains:
#   build_unskilled_G_weights      — dG(p) quadrature weights
#   G_cdf_unskilled                — Beta(α_U,1) CDF
#   solve_unskilled_surplus_on_grid! — surplus fixed-point per type
#   unskilled_inner_loop!          — iterate values given θ, pstar
#   solve_stationary_unskilled_pointwise! — stationary composition
#   update_pstar_from_surplus!     — reservation-quality update
#   update_theta_unskilled         — free-entry tightness update
#   solve_unskilled_block!         — outer loop (θ, pstar, u, t)
############################################################

# packages loaded by main.jl

# ---------------------------------------------------------------------------
# dG quadrature weights
#   G = Beta(α_U, 1),  g(p) = α_U · p^(α_U − 1)
# ---------------------------------------------------------------------------
function build_unskilled_G_weights(
    pgrid :: Vector{Float64},
    wp    :: Vector{Float64},
    α     :: Float64
)
    Np = length(pgrid)
    wG = similar(wp)
    @inbounds for j in 1:Np
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
# Unskilled surplus on the p-grid
#
# Closed-form reduction:
#   S(p) = [PU·x·p − (r+ν)·U + λ·I] / (r+ν+λ)   for p ≥ pstar_x, else 0
# where I satisfies a scalar fixed point solved analytically.
#
# Returns (I, tail_mass, i0)
# ---------------------------------------------------------------------------
function solve_unskilled_surplus_on_grid!(
    Svec    :: Vector{Float64},
    pgrid   :: Vector{Float64},
    wG      :: Vector{Float64},
    PU      :: Float64,
    x       :: Float64,
    r       :: Float64,
    ν       :: Float64,
    λ       :: Float64,
    Ux      :: Float64,
    pstar_x :: Float64
)
    Np = length(pgrid)

    # (1) Soft-weighted tail integrals — continuous in pstar_x
    tail_mass = 0.0
    tail_p1   = 0.0
    for j in 1:Np
        ω          = _soft_weight(pgrid[j], pstar_x, pgrid, j, Np)
        tail_mass += ω * wG[j]
        tail_p1   += ω * pgrid[j] * wG[j]
    end

    # (2) Solve for I analytically (same formula as before)
    denom = r + ν + λ
    A     = (PU * x * tail_p1 - (r + ν) * Ux * tail_mass) / denom
    B     = (λ / denom) * tail_mass
    I     = (abs(1.0 - B) < 1e-14) ? 0.0 : (A / (1.0 - B))

    # (3) Fill surplus surface with soft weights — I is now defined.
    # Clamp to ≥ 0: when pstar is clamped to 1, _soft_weight returns 1 at
    # j=Np (p=pstar=1) but the raw formula is negative there by construction,
    # violating S(p) ≥ 0.  The max enforces the model definition exactly.
    for j in 1:Np
        ω        = _soft_weight(pgrid[j], pstar_x, pgrid, j, Np)
        Svec[j]  = max(ω * (PU * x * pgrid[j] - (r + ν) * Ux + λ * I) / denom, 0.0)
    end

    return I, tail_mass, 0   # i0=0 sentinel, callers only use I
end


# ---------------------------------------------------------------------------
# Unskilled inner loop
#
# Given θ and pstar held fixed, iterate on Usearch, T, U, Jfrontier.
# US_in — current skilled unemployment value U_S(x), length Nx.
# ---------------------------------------------------------------------------
function unskilled_inner_loop!(
    model :: Model;
    US_in :: AbstractVector{Float64}
)
    cp  = model.common
    rp  = model.regime
    gp  = model.grids
    up  = model.unsk_par
    ug  = model.unsk_grids
    uc  = model.unsk_cache
    sim = model.sim

    Nx = length(gp.x)
    Np = length(ug.p)

    r  = cp.r;   ν  = cp.ν;   φ  = cp.φ;   c  = cp.c
    μ  = up.μ;   η  = up.η;   β  = up.β;   λ  = up.λ
    PU = rp.PU;  bU = rp.bU;  bT = rp.bT;  α  = rp.α_U

    θ  = uc.θ
    f  = jobfinding_rate(θ, μ, η)

    wG = build_unskilled_G_weights(ug.p, ug.wp, α)

    Svec_tls = [zeros(Float64, Np) for _ in 1:Threads.nthreads()]

    Ivec  = zeros(Float64, Nx)
    E1vec = zeros(Float64, Nx)

    Usearch_old = copy(uc.Usearch)
    U_old       = copy(uc.U)
    T_old       = copy(uc.T)

    streak = 0
    for it in 1:sim.maxit_inner
        copyto!(Usearch_old, uc.Usearch)
        copyto!(U_old,       uc.U)
        copyto!(T_old,       uc.T)

        # (1) Training value: (r + φ + ν) T(x) = bT + φ · U_S(x)
        @threads for ix in 1:Nx
            @inbounds uc.T[ix] = (bT + φ * US_in[ix]) / (r + φ + ν)
        end

        # (2) Outside value and training indicator
        #     U(x) = max{ Usearch(x),  −c(x) + T(x) }
        @threads for ix in 1:Nx
            @inbounds begin
                Utr = -training_cost(gp.x[ix], c) + uc.T[ix]
                Usr = uc.Usearch[ix]
                if Utr >= Usr
                    uc.U[ix]  = Utr
                    uc.τT[ix] = 1.0
                else
                    uc.U[ix]  = Usr
                    uc.τT[ix] = 0.0
                end
            end
        end

        # (3) Match surplus → Jfrontier and E at p=1
        @threads for ix in 1:Nx
            tid  = Threads.threadid()
            Svec = Svec_tls[tid]
            @inbounds begin
                I, _, _ = solve_unskilled_surplus_on_grid!(
                    Svec, ug.p, wG, PU, gp.x[ix], r, ν, λ, uc.U[ix],
                    clamp01(uc.pstar[ix])
                )
                Ivec[ix] = I

                S1 = Svec[end]
                uc.Jfrontier[ix] = (1.0 - β) * S1
                E1vec[ix]        = uc.U[ix] + β * S1
            end
        end

        # (4) Search value: (r + ν + f) Usearch(x) = bU + f · E(x, p=1)
        denom_search = r + ν + f
        @threads for ix in 1:Nx
            @inbounds uc.Usearch[ix] = (bU + f * E1vec[ix]) / denom_search
        end

        # (5) Convergence
        d1 = supnorm(uc.Usearch, Usearch_old)
        d2 = supnorm(uc.U,       U_old)
        d3 = supnorm(uc.T,       T_old)
        d  = max(d1, d2, d3)

        if d < sim.tol_inner
            streak += 1
            streak >= sim.conv_streak && break
        else
            streak = 0
        end
    end

    return (f = f, Ivec = Ivec)
end


# ---------------------------------------------------------------------------
# Stationary composition — pointwise
# ---------------------------------------------------------------------------
function solve_stationary_unskilled_pointwise!(
    u_out      :: AbstractVector{Float64},
    t_out      :: AbstractVector{Float64},
    ℓvals      :: AbstractVector{Float64},
    τvals      :: AbstractVector{Float64},
    pstar_vals :: AbstractVector{Float64},
    f          :: Float64,
    model      :: Model
)
    ν = model.common.ν
    φ = model.common.φ
    λ = model.unsk_par.λ
    α = model.regime.α_U

    @inbounds for ix in 1:length(model.grids.x)
        ℓx    = ℓvals[ix]
        τ     = τvals[ix]
        pstar = clamp01(pstar_vals[ix])

        if τ > 0.5
            # Training: new entrants go directly to training,
            # bypassing unemployment entirely.
            u_out[ix] = 0.0
            t_out[ix] = ν * ℓx / (φ + ν)
        else
            # Searching: no training; standard search unemployment.
            δ      = λ * G_cdf_unskilled(pstar, α)
            f_hire = (pstar < 1.0 - 1e-10) ? f : 0.0
            denom  = f_hire + δ + ν
            u_out[ix] = denom > 0.0 ? ℓx * (δ + ν) / denom : 0.0
            t_out[ix] = 0.0
        end
    end
    return nothing
end


# ---------------------------------------------------------------------------
# Reservation-quality update from surplus
# ---------------------------------------------------------------------------
function update_pstar_from_surplus!(
    pstar_new :: AbstractVector{Float64},
    model     :: Model,
    Ivec      :: AbstractVector{Float64}
)
    r  = model.common.r;    ν  = model.common.ν
    λ  = model.unsk_par.λ;  PU = model.regime.PU
    uc = model.unsk_cache

    @inbounds for ix in 1:length(model.grids.x)
        x  = model.grids.x[ix]
        Ux = uc.U[ix]
        I  = Ivec[ix]

        if x <= 1e-14 || PU <= 1e-14
            pstar_new[ix] = 1.0
        else
            pstar_new[ix] = clamp01(((r + ν) * Ux - λ * I) / (PU * x))
        end
    end
    return nothing
end


# ---------------------------------------------------------------------------
# Free-entry tightness update for unskilled market
# ---------------------------------------------------------------------------
function update_theta_unskilled(model::Model)
    uc = model.unsk_cache
    gp = model.grids
    up = model.unsk_par

    U_total = dot(uc.u,                 gp.wx)
    Jbar    = dot(uc.Jfrontier .* uc.u, gp.wx)

    if Jbar < 1e-14 || U_total < 1e-14 || !isfinite(Jbar) || !isfinite(U_total)
        # Dead market: no profitable matches → θ should be near-zero,
        # not the stale cache value (which may be the 1.0 initial seed).
        return 1e-14
    end

    q     = up.k * U_total / Jbar
    θ_raw = theta_from_q(q, up.μ, up.η)
    return clamp(θ_raw, 1e-14, 100.0)
end


# ---------------------------------------------------------------------------
# Unskilled outer loop
#
# Alternates: inner-loop values → stationary composition →
# reservation-quality update → free-entry θ update.
# ---------------------------------------------------------------------------
function solve_unskilled_block!(
    model :: Model;
    US_in :: AbstractVector{Float64}
)
    gp  = model.grids
    uc  = model.unsk_cache
    sim = model.sim

    Nx = length(gp.x)

    aa_θ      = Anderson1(1)
    u_new     = zeros(Float64, Nx)
    t_new     = zeros(Float64, Nx)
    pstar_new = zeros(Float64, Nx)

    streak = 0
    for it in 1:sim.maxit_outer
        θ_old     = uc.θ
        pstar_old = copy(uc.pstar)
        u_old     = copy(uc.u)

        # Step 1: inner loop (values + Jfrontier)
        inner = unskilled_inner_loop!(model; US_in = US_in)
        f     = inner.f
        Ivec  = inner.Ivec

        # Step 2: stationary composition
        solve_stationary_unskilled_pointwise!(
            u_new, t_new, gp.ℓ, uc.τT, uc.pstar, f, model
        )
        copyto!(uc.u, u_new)
        copyto!(uc.t, t_new)

        # Step 3: reservation-quality cutoff pstar
        update_pstar_from_surplus!(pstar_new, model, Ivec)
        @inbounds for ix in 1:Nx
            uc.pstar[ix] = clamp01(
                sim.damp_pstar_U * pstar_new[ix] + (1.0 - sim.damp_pstar_U) * pstar_old[ix]
            )
        end

        # Step 4: market tightness θ (Anderson accelerated)
        θ_raw = update_theta_unskilled(model)
        θ_acc = anderson1_update!(aa_θ, [θ_old], [θ_raw])[1]
        uc.θ  = max(θ_acc, 1e-14)

        # NaN/Inf guard — abort immediately
        if !isfinite(uc.θ)
            sim.verbose >= 1 && @printf("  [outer U]  NaN/Inf θ at it=%d — aborting\n", it)
            return false
        end

        # Step 5: convergence
        dθ = abs(uc.θ - θ_old)
        dp = supnorm(uc.pstar, pstar_old)
        du = supnorm(uc.u,     u_old)
        d  = max(dθ, dp, du)

        if sim.verbose >= 2 && (it == 1 || it % sim.verbose_stride == 0)
            @printf("  [outer U it=%d]  maxΔ=%.3e  (Δθ=%.3e  Δp*=%.3e  Δu=%.3e)  θ=%.4f\n",
                    it, d, dθ, dp, du, uc.θ)
        end

        if d < sim.tol_outer_U
            streak += 1
            if streak >= sim.conv_streak
                sim.verbose >= 2 && @printf(
                    "  [outer U]  converged it=%d  d=%.3e  θ=%.4f\n", it, d, uc.θ)
                # Final pass: recompute stationary composition with the
                # converged (θ, p*) so that uc.u and uc.t are consistent
                # with the final state, not the pre-Anderson values.
                inner_final = unskilled_inner_loop!(model; US_in = US_in)
                solve_stationary_unskilled_pointwise!(
                    u_new, t_new, gp.ℓ, uc.τT, uc.pstar, inner_final.f, model
                )
                copyto!(uc.u, u_new)
                copyto!(uc.t, t_new)
                return true   # ← converged
            end
        else
            streak = 0
        end
    end

    # Reached maxit without converging — still re-sync densities
    inner_final = unskilled_inner_loop!(model; US_in = US_in)
    solve_stationary_unskilled_pointwise!(
        u_new, t_new, gp.ℓ, uc.τT, uc.pstar, inner_final.f, model
    )
    copyto!(uc.u, u_new)
    copyto!(uc.t, t_new)
    sim.verbose >= 1 && @printf("  [outer U]  maxit reached without convergence  θ=%.4f\n", uc.θ)
    return false
end
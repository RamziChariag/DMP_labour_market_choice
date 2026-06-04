############################################################
# unskilled.jl — Unskilled block solver
#
# Outer loop iterates on (θ_U, p*_U, u_U, t).  The inner loop
# settles (U^search_U, U_U, T, E_U(·, 1), J_U(·, 1)) at fixed
# (θ_U, p*_U).
#
# Free entry uses the augmented seeker pool
#   ũ_U(x) = u_U(x) + d(x) · u_S(x),
# with the cross-market contribution d·u_S supplied by the
# skilled block via uc.duS_carry.
#
# Functions
#   build_unskilled_G_weights       dG(p) quadrature weights
#   G_cdf_unskilled                 Beta(α_U, 1) CDF
#   solve_unskilled_surplus_on_grid! surplus on the p-grid for one x
#   unskilled_inner_loop!           iterate values at (θ_U, p*_U)
#   solve_stationary_unskilled_pointwise! stationary u_U, t
#   update_pstar_from_surplus!      reservation-quality update
#   update_theta_unskilled          free-entry tightness update
#   solve_unskilled_block!          outer loop on (θ_U, p*_U, u_U, t)
############################################################


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
# Closed-form reduction of equation (11):
#   S(p) = [PU·x·p − (r+ν)·U + λ·I] / (r+ν+λ)   for p ≥ p*(x),  0 otherwise,
# where I = ∫_{p*}^1 S(p′) dG(p′) is the scalar tail integral.
# I satisfies a scalar fixed point, solved analytically.
#
# The surface is built with soft thresholding around p*(x) so that
# S(·, p) is a continuous function of p* — necessary for the outer
# fixed-point iteration on the cutoff.
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

    # Soft-weighted tail integrals.
    tail_mass = 0.0
    tail_p1   = 0.0
    for j in 1:Np
        ω          = _soft_weight(pgrid[j], pstar_x, pgrid, j, Np)
        tail_mass += ω * wG[j]
        tail_p1   += ω * pgrid[j] * wG[j]
    end

    # Scalar fixed point for I.
    denom = r + ν + λ
    A     = (PU * x * tail_p1 - (r + ν) * Ux * tail_mass) / denom
    B     = (λ / denom) * tail_mass
    I     = (abs(1.0 - B) < 1e-14) ? 0.0 : (A / (1.0 - B))

    # Fill the surface; clamp at zero to satisfy S(p) ≥ 0.
    for j in 1:Np
        ω        = _soft_weight(pgrid[j], pstar_x, pgrid, j, Np)
        Svec[j]  = max(ω * (PU * x * pgrid[j] - (r + ν) * Ux + λ * I) / denom, 0.0)
    end

    return I, tail_mass, 0
end


# ---------------------------------------------------------------------------
# Inner loop
#
# Given θ_U and p*_U held fixed, iterate jointly on
#   T, U_U, U^search_U, J_U(·, 1)
# until convergence.  The skilled unemployment value U_S is supplied
# from the global outer loop and treated as exogenous here.
#
# At the entry efficiency p = 1, Nash bargaining gives
#   J_U(x, 1) = (1 − β) S_U(x, 1),
#   E_U(x, 1) = U_U(x) + β   S_U(x, 1),
# which the search-value update then consumes via
#   (r + ν + f_U) U^search_U(x) = b_U + f_U E_U(x, 1).
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

        # Training value:  (r + φ + ν) T(x) = b_T + φ U_S(x)
        @threads for ix in 1:Nx
            @inbounds uc.T[ix] = (bT + φ * US_in[ix]) / (r + φ + ν)
        end

        # Outside value and training indicator:
        #   U_U(x)  = max{ U^search_U(x),  −c(x) + T(x) }
        #   τ(x)    = 1{−c(x) + T(x) ≥ U^search_U(x)}
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

        # Match surplus → J_U(x, 1) and E_U(x, 1).
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

        # Search value:  (r + ν + f_U) U^search_U(x) = b_U + f_U E_U(x, 1)
        denom_search = r + ν + f
        @threads for ix in 1:Nx
            @inbounds begin
                uc.Usearch[ix] = (bU + f * E1vec[ix]) / denom_search
            end
        end

        d1 = supnorm(uc.Usearch, Usearch_old)
        d2 = supnorm(uc.U, U_old)
        d3 = supnorm(uc.T, T_old)
        d  = max(d1, d2, d3)

        if d < sim.tol_inner
            streak += 1
            streak >= sim.conv_streak && break
        else
            streak = 0
        end
    end

    return (f = f, Ivec = Ivec, E1vec = E1vec)
end


# ---------------------------------------------------------------------------
# Stationary composition — pointwise in x
#
# τ(x) = 1 (training type): training is initiated on arrival in
#   unemployment, so u_U(x) = 0 and t(x) = ν ℓ(x) / (φ + ν).
#
# τ(x) = 0 (search type): the standard balance gives
#   u_U(x) = ℓ(x) (δ + ν) / (f_U + δ + ν),    δ = λ_U G(p*(x)),
#   t(x)   = 0.
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
            u_out[ix] = 0.0
            t_out[ix] = ν * ℓx / (φ + ν)
        else
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
# Reservation-quality update
#
# Solves S_U(x, p*) = 0 for p*(x).  Recomputes U_U(x) against the
# proposed θ_new so that the new p* is consistent with the same
# tightness that will be installed by the outer loop.
# ---------------------------------------------------------------------------
function update_pstar_from_surplus!(
    pstar_new :: AbstractVector{Float64},
    model     :: Model,
    Ivec      :: AbstractVector{Float64};
    θ_new     :: Float64,
    E1        :: AbstractVector{Float64},
    T_in      :: AbstractVector{Float64}
)
    cp = model.common
    rp = model.regime
    up = model.unsk_par
    gp = model.grids

    r  = cp.r;   ν  = cp.ν;   c  = cp.c
    λ  = up.λ;   μ  = up.μ;   η  = up.η
    PU = rp.PU;  bU = rp.bU

    f_new        = jobfinding_rate(θ_new, μ, η)
    denom_search = r + ν + f_new

    @inbounds for ix in 1:length(gp.x)
        x = gp.x[ix]
        I = Ivec[ix]

        Usearch_new = (bU + f_new * E1[ix]) / denom_search
        Utr         = -training_cost(x, c) + T_in[ix]
        U_new       = max(Usearch_new, Utr)

        if x <= 1e-14 || PU <= 1e-14
            pstar_new[ix] = 1.0
        else
            pstar_new[ix] = clamp01(((r + ν) * U_new - λ * I) / (PU * x))
        end
    end
    return nothing
end


# ---------------------------------------------------------------------------
# Free-entry tightness update
#
#   q_U(θ_U) = k_U · ũ_U / ∫ J_U(x, 1) · (u_U(x) + d(x) u_S(x)) dx,
# with ũ_U = ∫ (u_U + d u_S) dx.
# The cross-market contribution d·u_S is carried in uc.duS_carry.
# ---------------------------------------------------------------------------
function update_theta_unskilled(model::Model)
    uc = model.unsk_cache
    gp = model.grids
    up = model.unsk_par

    ueff = uc.u .+ uc.duS_carry

    U_total = dot(ueff,                gp.wx)
    Jbar    = dot(uc.Jfrontier .* ueff, gp.wx)

    if Jbar < 1e-14 || U_total < 1e-14 || !isfinite(Jbar) || !isfinite(U_total)
        return 1e-14
    end

    q     = up.k * U_total / Jbar
    θ_raw = theta_from_q(q, up.μ, up.η)
    return clamp(θ_raw, 1e-14, 100.0)
end


# ---------------------------------------------------------------------------
# Outer loop
#
# Per iteration:
#   1. Inner loop at the current (θ_U, p*_U).
#   2. Stationary composition (u_U, t) at the current p*_U.
#   3. Raw proposals (θ_raw, p*_raw) from free entry and S_U(x, p*) = 0.
#   4. Install:
#        Anderson on: joint Anderson(m = 1) on [θ_U; p*_U] with
#        per-block scaling so both components contribute to the
#        residual norm on comparable scales.
#        Anderson off: damped Picard with θ updated first, then p*
#        recomputed against the post-update θ.
# Convergence is checked on (Δθ_U, Δp*_U).
# ---------------------------------------------------------------------------
function solve_unskilled_block!(
    model :: Model;
    US_in :: AbstractVector{Float64}
)
    gp  = model.grids
    uc  = model.unsk_cache
    sim = model.sim

    Nx = length(gp.x)

    aa_joint  = Anderson1(1 + Nx)
    s_θ       = 1.0
    s_p       = 1.0

    u_new     = zeros(Float64, Nx)
    t_new     = zeros(Float64, Nx)
    pstar_new = zeros(Float64, Nx)

    streak = 0
    for it in 1:sim.maxit_outer
        θ_old     = uc.θ
        pstar_old = copy(uc.pstar)
        u_old     = copy(uc.u)

        # 1. Inner loop at the current (θ, p*).
        inner = unskilled_inner_loop!(model; US_in = US_in)
        f     = inner.f
        Ivec  = inner.Ivec
        E1    = inner.E1vec

        # 2. Stationary composition at the current p*.
        solve_stationary_unskilled_pointwise!(
            u_new, t_new, gp.ℓ, uc.τT, uc.pstar, f, model
        )
        copyto!(uc.u, u_new)
        copyto!(uc.t, t_new)

        # 3. Raw proposals for (θ, p*).
        θ_raw = update_theta_unskilled(model)

        if sim.use_anderson
            update_pstar_from_surplus!(
                pstar_new, model, Ivec;
                θ_new = θ_old,
                E1    = E1,
                T_in  = uc.T,
            )
        else
            uc.θ = max(θ_raw, 1e-14)
            update_pstar_from_surplus!(
                pstar_new, model, Ivec;
                θ_new = uc.θ,
                E1    = E1,
                T_in  = uc.T,
            )
        end

        # 4. Install (θ, p*).
        if sim.use_anderson
            if it == 1
                s_θ = max(abs(θ_raw), abs(θ_old), 1.0)
                s_p = max(maximum(abs, pstar_new), 1.0)
            end

            x_old = vcat([θ_old / s_θ], pstar_old ./ s_p)
            f_raw = vcat([θ_raw / s_θ], pstar_new ./ s_p)
            x_new = anderson1_update!(aa_joint, x_old, f_raw)

            uc.θ = max(x_new[1] * s_θ, 1e-14)
            @inbounds for ix in 1:Nx
                uc.pstar[ix] = clamp01(x_new[1 + ix] * s_p)
            end
        else
            @inbounds for ix in 1:Nx
                uc.pstar[ix] = clamp01(
                    sim.damp_pstar_U * pstar_new[ix] +
                    (1.0 - sim.damp_pstar_U) * pstar_old[ix]
                )
            end
        end

        if !isfinite(uc.θ) || any(!isfinite, uc.pstar)
            sim.verbose >= 1 && @printf("  [outer U]  NaN/Inf in state at it=%d — aborting\n", it)
            return false
        end

        # 5. Convergence on (Δθ, Δp*, Δu).
        dθ = abs(uc.θ - θ_old)
        dp = supnorm(uc.pstar, pstar_old)
        du = supnorm(uc.u,     u_old)
        d  = max(dθ, dp, du)

        if sim.verbose >= 2 && (it == 1 || it % sim.verbose_stride == 0)
            @printf("  [outer U it=%d]  maxΔ=%.3e  (Δθ=%.3e  Δp*=%.3e  Δu=%.3e)  θ_U=%.4f\n",
                    it, d, dθ, dp, du, uc.θ)
        end

        if d < sim.tol_outer_U
            streak += 1
            if streak >= sim.conv_streak
                sim.verbose >= 2 && @printf(
                    "  [outer U]  converged it=%d  d=%.3e  θ_U=%.4f\n", it, d, uc.θ)
                inner_final = unskilled_inner_loop!(model; US_in = US_in)
                solve_stationary_unskilled_pointwise!(
                    u_new, t_new, gp.ℓ, uc.τT, uc.pstar, inner_final.f, model
                )
                copyto!(uc.u, u_new)
                copyto!(uc.t, t_new)
                return true
            end
        else
            streak = 0
        end
    end

    inner_final = unskilled_inner_loop!(model; US_in = US_in)
    solve_stationary_unskilled_pointwise!(
        u_new, t_new, gp.ℓ, uc.τT, uc.pstar, inner_final.f, model
    )
    copyto!(uc.u, u_new)
    copyto!(uc.t, t_new)
    sim.verbose >= 1 && @printf("  [outer U]  maxit reached without convergence  θ_U=%.4f\n", uc.θ)
    return false
end

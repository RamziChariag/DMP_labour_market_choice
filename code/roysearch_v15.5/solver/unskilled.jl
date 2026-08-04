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
#   unskilled_inner_loop!            iterate values at (θ_U, p*)
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
# −ξ_U S, which nets into the base, so the −(r+ν)U^search numerator (and
# hence the reservation p*) is invariant to ξ_U (mirror skilled.jl ϱ_S).
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


# ---------------------------------------------------------------------------
# Inner loop
#
# Given θ_U and p*(aU) fixed, iterate on (T, U^search, E_U(·,1), J_U(·,1))
# until convergence.  U_S is supplied from the global loop.
#
# At p = 1, Nash bargaining gives J_U(aU,1) = (1−β) S_U(aU,1),
# E_U(aU,1) = U^search(aU) + β S_U(aU,1); the search-value update consumes
# E_U(aU,1) via (r+ν+f_U) U^search = b_U + f_U E_U(aU,1).  The frontier τ
# is formed here (2D) for the distribution step but does not feed the
# unskilled outside option, which is U^search(aU) alone (notes §276).
# ---------------------------------------------------------------------------
function unskilled_inner_loop!(model::Model; US_in::AbstractVector{Float64})
    cp = model.common;  gp = model.grids;  up = model.unsk_par
    ug = model.unsk_grids;  uc = model.unsk_cache;  sim = model.sim

    Nx = length(gp.x);  Np = length(ug.p)
    r  = cp.r;  ν = cp.ν;  φ = cp.φ
    μ  = up.μ;  η = up.η;  β = up.β;  λ = up.λ;  ξ = up.ξ
    PU = exp(cp.A) * up.PU;  bU = up.bU * exp(cp.A);  bT = up.bT * exp(cp.A);  α = up.α_U

    f  = jobfinding_rate(uc.θ, μ, η)
    wG = build_unskilled_G_weights(ug.p, ug.wp, α)

    Svec_tls = [zeros(Float64, Np) for _ in 1:Threads.nthreads()]
    Ivec  = zeros(Float64, Nx)
    E1vec = zeros(Float64, Nx)

    Usearch_old = copy(uc.Usearch)
    T_old       = copy(uc.T)

    αU     = sim.damp_inner_U
    B      = sim.inner_B;  K = sim.inner_K
    resid  = Vector{Float64}(undef, sim.maxit_inner)
    status = :maxit;  n_inner = 0;  streak = 0

    for it in 1:sim.maxit_inner
        copyto!(Usearch_old, uc.Usearch)
        copyto!(T_old,       uc.T)

        # Training value:  (r + φ + ν) T(aS) = b_T + φ U_S(aS)
        @threads for ix in 1:Nx
            @inbounds uc.T[ix] = (bT + φ * US_in[ix]) / (r + φ + ν)
        end

        # Match surplus at the current U^search outside option → J_U(aU,1), E_U(aU,1).
        @threads for ix in 1:Nx
            Svec = Svec_tls[Threads.threadid()]
            @inbounds begin
                PUeff    = PU * gp.x[ix]                       # A P_U aU (linear loading)
                Ivec[ix] = solve_unskilled_surplus_on_grid!(
                    Svec, ug.p, wG, PUeff, r, ν, λ, uc.Usearch[ix], clamp01(uc.pstar[ix]), ξ)
                S1               = Svec[end]
                uc.Jfrontier[ix] = (1.0 - β) * S1
                E1vec[ix]        = uc.Usearch[ix] + β * S1
            end
        end

        # Search value:  (r + ν + f_U) U^search(aU) = b_U + f_U E_U(aU,1)
        denom_search = r + ν + f
        @threads for ix in 1:Nx
            @inbounds begin
                raw            = (bU + f * E1vec[ix]) / denom_search
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

    # Training frontier τ(aU,aS) = 1{−c(aS) + T(aS) ≥ U^search(aU)}
    # (2D; no d-gate — b_T discipline is by comparative advantage).
    # c(aS) is a psychological cost in fixed utility units — NOT scaled by A —
    # so the training margin shifts with the wage level rather than being scale-invariant.
    @threads for j in 1:Nx
        @inbounds begin
            Utr_j = -training_cost(gp.x[j], cp.c) + uc.T[j]   # depends on aS = x[j]
            for i in 1:Nx
                uc.τT[i, j] = (Utr_j >= uc.Usearch[i]) ? 1.0 : 0.0      # aU = x[i]
            end
        end
    end

    return (f = f, Ivec = Ivec, E1vec = E1vec,
            status = status, converged = (status === :converged), iters = n_inner)
end


# ---------------------------------------------------------------------------
# Stationary composition on the copula grid
#
# Match rates depend only on aU, and the frontier τ(aU,aS) determines who
# trains, so the balance is solved per (aU,aS) cell with aU-indexed rates
# and the joint population weight ℓ(aU,aS) = W2[i,j] (notes §332).
#
# τ = 1 (train):   u_U = 0,  t = ν W2 / (φ + ν).
# τ = 0 (search):  u_U = W2 (δ+ν)/(f_U+δ+ν),  t = 0,   δ = ξ_U + λ_U G(p*(aU)).
# ---------------------------------------------------------------------------
function solve_stationary_unskilled!(
    u_out::Matrix{Float64}, t_out::Matrix{Float64},
    W2::Matrix{Float64}, τ::Matrix{Float64},
    pstar::AbstractVector{Float64}, f::Float64, model::Model,
)
    ν = model.common.ν;  φ = model.common.φ
    λ = model.unsk_par.λ;  α = model.unsk_par.α_U;  ξ = model.unsk_par.ξ
    Nx = length(model.grids.x)

    @inbounds for i in 1:Nx                       # aU index
        pst    = clamp01(pstar[i])
        δ      = ξ + λ * G_cdf_unskilled(pst, α)
        f_hire = (pst < 1.0 - 1e-10) ? f : 0.0
        denom  = f_hire + δ + ν
        usurv  = denom > 0.0 ? (δ + ν) / denom : 0.0
        for j in 1:Nx                             # aS index
            if τ[i, j] > 0.5
                u_out[i, j] = 0.0
                t_out[i, j] = ν * W2[i, j] / (φ + ν)
            else
                u_out[i, j] = W2[i, j] * usurv
                t_out[i, j] = 0.0
            end
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

    f_new        = jobfinding_rate(θ_new, μ, η)
    denom_search = r + ν + f_new

    @inbounds for i in eachindex(gp.x)
        aU          = gp.x[i]
        Usearch_new = (bU + f_new * E1[i]) / denom_search
        PUeff       = PU * aU
        pstar_new[i] = (PUeff <= 1e-14) ? 1.0 :
            clamp01(((r + ν) * Usearch_new - λ * Ivec[i]) / PUeff)
    end
    return nothing
end


# ---------------------------------------------------------------------------
# Free-entry tightness update
#
#   q_U(θ_U) = k_U π̄_U ũ_U / ∬ J_U(aU,1) (u_U + d u_S) daU daS,
#   ũ_U = ∬ (u_U + d u_S).
# The dollar posting cost is k_U π̄_U (k_U dimensionless, π̄_U the mean
# unskilled flow output).  d·u_S is carried in uc.duS_carry.  J_U(aU,1)
# is aU-indexed and broadcast across aS.
# ---------------------------------------------------------------------------
function update_theta_unskilled(model::Model)
    uc = model.unsk_cache;  up = model.unsk_par
    ueff = uc.u .+ uc.duS_carry                    # (aU,aS) augmented seeker density

    U_total = sum(ueff)
    Jbar    = 0.0
    Nx = size(ueff, 1)
    @inbounds for j in 1:Nx, i in 1:Nx
        Jbar += uc.Jfrontier[i] * ueff[i, j]       # J_U reads aU = row i
    end

    (Jbar < 1e-14 || U_total < 1e-14 || !isfinite(Jbar) || !isfinite(U_total)) && return 1e-14

    q     = up.k * mean_output_U(model) * U_total / Jbar
    θ_raw = theta_from_q(q, up.μ, up.η)
    return clamp(θ_raw, 1e-14, 100.0)
end


# ---------------------------------------------------------------------------
# Outer loop on (θ_U, p*, u_U, t)
#
# Per iteration: inner values → stationary composition → raw (θ, p*) from
# free entry and S_U(aU,p*)=0 → joint Anderson(m=1) install (or damped
# Picard).  Convergence on (Δθ, Δp*, Δu_U).  Sustained inner divergence
# rejects the parameter; an outer stall hands back to the global loop.
# ---------------------------------------------------------------------------
function solve_unskilled_block!(model::Model; US_in::AbstractVector{Float64})
    gp = model.grids;  uc = model.unsk_cache;  sim = model.sim
    Nx = length(gp.x)

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

        solve_stationary_unskilled!(u_new, t_new, gp.copula.W2, uc.τT, uc.pstar, inner.f, model)
        copyto!(uc.u, u_new);  copyto!(uc.t, t_new)

        θ_raw = update_theta_unskilled(model)
        θ_for_pstar = sim.use_anderson ? θ_old : (uc.θ = max(θ_raw, 1e-14))
        update_pstar_from_surplus!(pstar_new, model, inner.Ivec; θ_new = θ_for_pstar, E1 = inner.E1vec)

        if sim.use_anderson
            if it == 1
                s_θ = max(abs(θ_raw), abs(θ_old), 1.0)
                s_p = max(maximum(abs, pstar_new), 1.0)
            end
            x_old = vcat([θ_old / s_θ], pstar_old ./ s_p)
            f_raw = vcat([θ_raw / s_θ], pstar_new ./ s_p)
            x_new = anderson1_update!(aa_joint, x_old, f_raw)
            uc.θ  = max(x_new[1] * s_θ, 1e-14)
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
                solve_stationary_unskilled!(u_new, t_new, gp.copula.W2, uc.τT, uc.pstar, inner_final.f, model)
                copyto!(uc.u, u_new);  copyto!(uc.t, t_new)
                return (converged = inner_conv, rejected = false)
            end
        else
            streak = 0
        end

        if oB > 0 && it > oB + oK && outer_resid[it] >= outer_resid[it - oK]
            sim.verbose >= 1 && @printf("  [outer U]  stalled at it=%d — handing back to global\n", it)
            inner_final = unskilled_inner_loop!(model; US_in = US_in)
            solve_stationary_unskilled!(u_new, t_new, gp.copula.W2, uc.τT, uc.pstar, inner_final.f, model)
            copyto!(uc.u, u_new);  copyto!(uc.t, t_new)
            return (converged = false, rejected = false)
        end
    end

    inner_final = unskilled_inner_loop!(model; US_in = US_in)
    solve_stationary_unskilled!(u_new, t_new, gp.copula.W2, uc.τT, uc.pstar, inner_final.f, model)
    copyto!(uc.u, u_new);  copyto!(uc.t, t_new)
    sim.verbose >= 1 && @printf("  [outer U]  maxit reached without convergence  θ_U=%.4f\n", uc.θ)
    return (converged = false, rejected = false)
end

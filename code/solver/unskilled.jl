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
#
# McQueen–Porteus acceleration is applied to Usearch (the primary
# iteration variable — T is pinned to US_in in one step, and U inherits
# from Usearch on searching types via the max).  Mirrors the pattern used
# in skilled_inner_loop! for sc.U.
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

    # McQueen–Porteus state (for the Usearch iteration)
    span_prev_U = Inf
    ρ_pm_U      = 0.5

    # Thread-local accumulators for δ_Usearch across x
    dUs_abs_tls = zeros(Float64, Threads.nthreads())
    dUs_min_tls = fill( Inf, Threads.nthreads())
    dUs_max_tls = fill(-Inf, Threads.nthreads())

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
        #     Track signed increments δ_Usearch for McQueen–Porteus.
        fill!(dUs_abs_tls, 0.0)
        fill!(dUs_min_tls,  Inf)
        fill!(dUs_max_tls, -Inf)
        denom_search = r + ν + f
        @threads for ix in 1:Nx
            tid = Threads.threadid()
            @inbounds begin
                Us_new = (bU + f * E1vec[ix]) / denom_search
                δUs    = Us_new - Usearch_old[ix]
                uc.Usearch[ix]   = Us_new
                dUs_abs_tls[tid] = max(dUs_abs_tls[tid], abs(δUs))
                dUs_min_tls[tid] = min(dUs_min_tls[tid], δUs)
                dUs_max_tls[tid] = max(dUs_max_tls[tid], δUs)
            end
        end

        # McQueen–Porteus acceleration on Usearch — DISABLED for A/B test.
        # Hypothesis: MP misbehaves here because U(x) = max(Usearch, -c+T)
        # decouples Usearch from U on training types, so the spectral-radius
        # estimate from δ_Usearch contaminates the shift.  Re-enable by
        # uncommenting the block below once we've confirmed this is the
        # culprit.  The accumulator fills above are kept so we can still
        # report d1 = max|δUs| as a diagnostic.
        δ_min_U    = minimum(dUs_min_tls)
        δ_max_U    = maximum(dUs_max_tls)
        span_new_U = δ_max_U - δ_min_U
        if it >= 2 && span_prev_U > 1e-14
            ρ_pm_U = clamp(span_new_U / span_prev_U, 0.0, 0.9999)
        end
        span_prev_U = span_new_U
        # --- MP shift disabled ---
        # if ρ_pm_U > 1e-8
        #     shift_U = ρ_pm_U / (1.0 - ρ_pm_U) * (δ_min_U + δ_max_U) / 2.0
        #     @inbounds for ix in 1:Nx
        #         uc.Usearch[ix] += shift_U
        #     end
        # end

        # (5) Convergence — restore the OLD convergence test (supnorm on the
        #     installed Usearch).  This is robust regardless of whether MP
        #     is on or off, since with MP off uc.Usearch == Us_new and the
        #     supnorm equals max|δUs|.
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
#
# IMPORTANT — option-2 fix for the unskilled outer-loop period-2:
#
# The naive Picard outer loop iterates (Usearch, U, pstar, θ) as four loosely
# coupled states.  Within `unskilled_inner_loop!`, Usearch and U are settled
# against the OLD θ (held fixed in uc.θ).  Then the outer loop produces a NEW
# θ from Jfrontier, but `update_pstar_from_surplus!` (in its old form) used
# the U value that was consistent with the OLD θ — even though θ has just
# been updated.  That mismatch creates a sign-flipping map
#       p* high  →  small surplus  →  small θ  →  small Usearch  →  small U
#                →  small p*  ⟶  on next iter, large surplus → large θ → ...
# whose natural attractor is a period-2 orbit.
#
# The fix: pass in the PROPOSED θ_new (computed *before* this routine is
# called) and the cached E1 surface from the inner loop.  Recompute the
# search value Usearch_new analytically against θ_new, then U_new from the
# max with the training branch (T is independent of θ within this outer
# step), and finally form the new cutoff using U_new.  This ensures the new
# p* is consistent with the same θ that is about to be installed.
#
# Inputs added:
#   θ_new — the next-iterate market tightness (post-Anderson, pre-install)
#   E1    — frontier match value E(x, p = 1) from the inner loop, length Nx
#   T_in  — training value T(x) from the inner loop, length Nx (uc.T is fine)
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

        # Recompute Usearch and U at the PROPOSED θ_new.
        # Usearch_new(x) = (bU + f_new · E1(x)) / (r + ν + f_new)
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
# Per outer iteration:
#   (1) Inner loop: solve (Usearch, U, T, Jfrontier) given θ, p*.
#   (2) Stationary composition (u, t) given the inner-loop f, current p*.
#   (3) Compute raw (Picard) proposals (θ_raw, p*_raw) from free-entry
#       and the indifference condition S(p*) = 0.
#   (4) Install:
#         use_anderson = true  →  joint Anderson(m=1) on [θ/s_θ; p*/s_p]
#                                 with per-block scaling.  This is the
#                                 primary mechanism for breaking the
#                                 period-2 orbit in p* that arose from
#                                 the (θ, p*) coupling — see the note
#                                 above update_pstar_from_surplus!.
#         use_anderson = false →  damped Picard fallback in the Option-2
#                                 ordering (θ first, then p* against the
#                                 new θ, with damp_pstar_U applied to p*).
# ---------------------------------------------------------------------------
function solve_unskilled_block!(
    model :: Model;
    US_in :: AbstractVector{Float64}
)
    gp  = model.grids
    uc  = model.unsk_cache
    sim = model.sim

    Nx = length(gp.x)

    # Joint Anderson(m=1) on the (θ, p*) state, length 1 + Nx.  Mirrors the
    # skilled side, except there is no poj on the unskilled market so the
    # state is just [θ; p*] (no gap-reparameterization needed).
    #
    # Per-block scaling: θ lives on O(1)–O(100), p* ∈ [0,1].  Without scaling,
    # the joint residual norm ‖[r_θ; r_p]‖ is dominated by the θ block and
    # Anderson cannot see the p* signal — defeating the point of joint
    # acceleration when the period-2 mode lives in p*.  Scales are locked
    # ONCE on the first iteration's raw outputs (Anderson's first call
    # returns f unchanged regardless of scale, so this is safe).
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

        # Step 1: inner loop (values + Jfrontier) at the CURRENT (θ, p*).
        inner = unskilled_inner_loop!(model; US_in = US_in)
        f     = inner.f
        Ivec  = inner.Ivec
        E1    = inner.E1vec

        # Step 2: stationary composition at the CURRENT p*.
        #         Uses the just-converged f from the inner loop — this is the
        #         f that is consistent with uc.θ (still the old θ at this point)
        #         and uc.pstar (still the old p*).  The resulting uc.u feeds
        #         the free-entry condition.
        solve_stationary_unskilled_pointwise!(
            u_new, t_new, gp.ℓ, uc.τT, uc.pstar, f, model
        )
        copyto!(uc.u, u_new)
        copyto!(uc.t, t_new)

        # Step 3: raw proposals for (θ, p*) — this is the Picard map T(θ, p*)
        #         that Anderson will accelerate.  Both proposals are computed
        #         from the OLD state (θ_old, pstar_old, U from inner loop) so
        #         that Anderson sees a clean vector-valued residual.
        θ_raw = update_theta_unskilled(model)

        if sim.use_anderson
            # Anderson on:  feed RAW p* proposal computed against the old
            # state's U (NOT against θ_raw).  This makes (θ_raw, p*_raw) =
            # T(θ_old, p*_old) a clean vector Picard map; Anderson(m=1) then
            # produces (θ_new, p*_new) jointly.  The Option-2 trick (recompute
            # p* against post-update θ) is unnecessary here — Anderson itself
            # breaks the period-2 by mixing the current and previous proposals.
            update_pstar_from_surplus!(
                pstar_new, model, Ivec;
                θ_new = θ_old,        # use OLD θ to keep the map T well-defined
                E1    = E1,
                T_in  = uc.T,
            )
        else
            # Anderson off:  use the Option-2 ordering for the Picard
            # fallback — update θ first, then p* against the new θ.  Without
            # Anderson there is nothing to break the period-2 orbit, and
            # Option 2 is what makes plain damped Picard convergent.
            uc.θ = max(θ_raw, 1e-14)
            update_pstar_from_surplus!(
                pstar_new, model, Ivec;
                θ_new = uc.θ,
                E1    = E1,
                T_in  = uc.T,
            )
        end

        # Step 4: install (θ, p*).
        if sim.use_anderson
            # Lock per-block scales on the first iteration once both raw
            # proposals are available.
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
            # Picard fallback — uc.θ already installed above; damp p*.
            @inbounds for ix in 1:Nx
                uc.pstar[ix] = clamp01(
                    sim.damp_pstar_U * pstar_new[ix] +
                    (1.0 - sim.damp_pstar_U) * pstar_old[ix]
                )
            end
        end

        # NaN/Inf guard — abort immediately.  Cover both θ and p* because
        # Anderson's joint update can introduce NaN in either component.
        if !isfinite(uc.θ) || any(!isfinite, uc.pstar)
            sim.verbose >= 1 && @printf("  [outer U]  NaN/Inf in state at it=%d — aborting\n", it)
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
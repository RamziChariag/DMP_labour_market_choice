############################################################
# solver.jl — Global equilibrium solver
#
# Public entry points
#   solve_model(common, unsk_par, skl_par, sim; Nx, Np_U, Np_S)
#       → build grids and caches, run the global loop, return (Model, SolveResult).
#   solve_model!(model)
#       → run the global loop in place on an existing Model.
#
# Global fixed point on the link variables (U_S, m_S), both Anderson(m=1)
# accelerated.  The remaining link objects are derived deterministically
# within each global pass:
#   f_U        = θ_U · q_U(θ_U)                   from the unskilled solve
#   E_U(x, 1)  = U_U(x) + β_U · S_U(x, 1)         from the frontier surplus
#   d(x) u_S(x)                                    from the skilled solve
#   m_S(x)     = ϕ t(x) / (ν + d(x) f_U)
#
# Per global iteration:
#   A. Solve the unskilled block with the carried d·u_S.
#   B. Form f_U and E_U(·, 1).
#   C. Build m_S using sc.d (= d from previous pass).
#   D. Solve the skilled block (inner loop updates sc.d).
#   E. Recompute m_S using the updated sc.d.
#   F. Anderson on the joint [U_S; m_S] with per-block scaling.
#   G. Write back: sc.U ← new U_S;  uc.duS_carry ← new d · u_S.
############################################################


# ============================================================
# SolveResult
# ============================================================

"""
    SolveResult

Records whether each layer of the solver converged on the final
global iteration.  All three flags must be `true` for `result.ok`
to be `true`.
"""
struct SolveResult
    converged_U      :: Bool
    converged_S      :: Bool
    converged_global :: Bool
    ok               :: Bool
end

SolveResult(cU::Bool, cS::Bool, cG::Bool) = SolveResult(cU, cS, cG, cU && cS && cG)


# ---------------------------------------------------------------------------
# Cache initialisation
# ---------------------------------------------------------------------------
function _initialise_caches(
    common   :: CommonParams,
    unsk_par :: UnskilledParams,
    skl_par  :: SkilledParams,
    grids    :: CommonGrids,
    u_grids  :: UnskilledGrids,
    s_grids  :: SkilledGrids
)
    Nx   = length(grids.x)
    Np_S = length(s_grids.p)

    r = common.r;   ν = common.ν;   φ = common.φ

    US_guess     = skl_par.bS * exp(common.A) / (r + ν)
    Usearch_init = fill(unsk_par.bU * exp(common.A) / (r + ν), Nx)
    T_init       = [(unsk_par.bT * exp(common.A) + φ * US_guess) / (r + φ + ν) for _ in 1:Nx]
    U_init       = max.(Usearch_init, T_init)
    t_seed       = [(ν / (ν + φ + ν)) * grids.ℓ[ix] for ix in 1:Nx]

    PU_init = max(exp(common.A) * unsk_par.PU, 1e-6)
    pstar_U_init = [clamp(unsk_par.bU * exp(common.A) / (PU_init * max(grids.x[ix], 1e-3)), 0.05, 0.90)
                    for ix in 1:Nx]

    pstar_S_init = [begin
                        xi = max(grids.x[ix], 1e-3)
                        PS_xi = max(PS_of_x(xi, skl_par.gamma_PS, exp(common.A)), 1e-6)
                        clamp(skl_par.bS * exp(common.A) / (PS_xi * xi), 0.05, 0.90)
                    end
                    for ix in 1:Nx]
    poj_init = clamp.(pstar_S_init .+ 0.30, pstar_S_init, 0.95)

    uc = UnskilledCache(
        Usearch   = Usearch_init,
        U         = U_init,
        T         = T_init,
        Jfrontier = zeros(Nx),
        pstar     = pstar_U_init,
        τT        = zeros(Nx),
        u         = 0.4 .* grids.ℓ,
        t         = t_seed,
        duS_carry = zeros(Nx),
        d_carry   = zeros(Nx),   # first pass unrestricted (d⁰ ≡ 0)
        θ         = 0.5,
    )

    US_init = fill(skl_par.bS * exp(common.A) / (r + ν), Nx)

    sc = SkilledCache(
        U     = US_init,
        E0    = zeros(Nx, Np_S),
        E1    = zeros(Nx, Np_S),
        J0    = zeros(Nx, Np_S),
        J1    = zeros(Nx, Np_S),
        pstar = pstar_S_init,
        poj   = poj_init,
        d     = zeros(Nx),
        u     = zeros(Nx),
        e     = zeros(Nx, Np_S),
        θ     = 0.5,
    )

    return uc, sc
end


# ---------------------------------------------------------------------------
# solve_model — allocate grids and caches, then solve
# ---------------------------------------------------------------------------
"""
    solve_model(common, unsk_par, skl_par, sim; Nx, Np_U, Np_S)
        → (Model, SolveResult)
"""
function solve_model(
    common   :: CommonParams,
    unsk_par :: UnskilledParams,
    skl_par  :: SkilledParams,
    sim      :: SimParams;
    Nx   :: Int = 200,
    Np_U :: Int = 200,
    Np_S :: Int = 200
)
    # Worker-type grid: Gauss–Jacobi tuned to ℓ = Beta(a_ℓ, b_ℓ) so that
    # ∫ g·ℓ dx (every model aggregate) is integrated exactly even when
    # a_ℓ < 1 or b_ℓ < 1.  wx are population weights (ℓ folded in): use
    # dot(density, wx) for aggregates.  Match-quality p-grids stay Legendre.
    xgrid,   wx   = build_type_grid(Nx, common.a_ℓ, common.b_ℓ)
    pgrid_U, wp_U = build_gl_grid(Np_U)
    pgrid_S, wp_S = build_gl_grid(Np_S)

    ℓvals = pdf.(Beta(common.a_ℓ, common.b_ℓ), xgrid)

    grids   = CommonGrids(x = xgrid, wx = wx, ℓ = ℓvals)
    u_grids = UnskilledGrids(p = pgrid_U, wp = wp_U)
    s_grids = SkilledGrids(p = pgrid_S, wp = wp_S)

    s_pre = build_skilled_precomp(s_grids, skl_par)

    uc, sc = _initialise_caches(common, unsk_par, skl_par,
                                 grids, u_grids, s_grids)

    model = Model(
        common     = common,
        grids      = grids,
        unsk_par   = unsk_par,
        unsk_grids = u_grids,
        unsk_cache = uc,
        skl_par    = skl_par,
        skl_grids  = s_grids,
        skl_pre    = s_pre,
        skl_cache  = sc,
        sim        = sim,
    )

    result = solve_model!(model)
    return model, result
end


# ---------------------------------------------------------------------------
# m_S(x) = ϕ t(x) / (ν + d(x) f_U)
# ---------------------------------------------------------------------------
function _mS_from_t(t::AbstractVector{Float64}, d::AbstractVector{Float64},
                    φ::Float64, ν::Float64, fU::Float64)
    Nx = length(t)
    mS = zeros(Nx)
    @inbounds for ix in 1:Nx
        denom = ν + clamp(d[ix], 0.0, 1.0) * fU
        mS[ix] = denom > 1e-14 ? max(φ * t[ix] / denom, 0.0) : 0.0
    end
    return mS
end


# ---------------------------------------------------------------------------
# solve_model! — in-place global fixed point
# ---------------------------------------------------------------------------
"""
    solve_model!(model) → SolveResult

Run the global fixed-point loop in place on `model`.  Returns a
`SolveResult` indicating whether each layer converged.
"""
function solve_model!(model::Model) :: SolveResult
    cp  = model.common
    sc  = model.skl_cache
    uc  = model.unsk_cache
    up  = model.unsk_par
    sim = model.sim

    φ  = cp.φ;   ν  = cp.ν
    Nx = length(model.grids.x)
    βU = up.β

    aa = Anderson1(2 * Nx)

    # First-pass m_S uses d ≡ 0  →  (ϕ/ν) · t.
    mS_cur = [max((φ / ν) * uc.t[ix], 0.0) for ix in 1:Nx]

    # Per-block Anderson scales for [U_S; m_S], locked on the first
    # iteration so both blocks contribute to the residual norm on
    # comparable scales.  Anderson's first call is a no-op regardless
    # of scale, so setting them at it = 1 is safe.
    s_U = 1.0
    s_M = 1.0

    last_conv_U = false
    last_conv_S = false
    global_converged = false

    streak = 0
    for it in 1:sim.maxit_global

        US_old = copy(sc.U)
        mS_old = copy(mS_cur)

        # A. Unskilled block with carried d · u_S in uc.duS_carry.
        res_U = solve_unskilled_block!(model; US_in = sc.U)
        res_U.rejected && return SolveResult(false, false, false)
        last_conv_U = res_U.converged

        if !isfinite(uc.θ) || any(!isfinite, uc.t) || any(!isfinite, uc.U) || any(!isfinite, uc.pstar)
            sim.verbose >= 1 && @printf("[global]  NaN/Inf in unskilled block at it=%d — aborting\n", it)
            return SolveResult(false, false, false)
        end

        # B. Form unskilled-side outputs consumed by the skilled block.
        fU  = jobfinding_rate(uc.θ, up.μ, up.η)
        SU1 = uc.Jfrontier ./ max(1.0 - βU, 1e-14)
        EU1 = uc.U .+ βU .* SU1

        # C. m_S using the previous-pass d.
        mS_raw_pre = _mS_from_t(uc.t, sc.d, φ, ν, fU)

        # D. Skilled block; inner loop updates sc.d.
        res_S = solve_skilled_block!(model;
                                     mS_in = mS_raw_pre,
                                     fU    = fU,
                                     EU1   = EU1)
        res_S.rejected && return SolveResult(false, false, false)
        last_conv_S = res_S.converged

        if !isfinite(sc.θ) || any(!isfinite, sc.U) || any(!isfinite, sc.pstar) ||
           any(!isfinite, sc.d) || any(!isfinite, sc.u)
            sim.verbose >= 1 && @printf("[global]  NaN/Inf in skilled block at it=%d — aborting\n", it)
            return SolveResult(false, false, false)
        end

        US_raw = copy(sc.U)

        # E. m_S using the just-updated sc.d.
        mS_raw = _mS_from_t(uc.t, sc.d, φ, ν, fU)

        if it == 1
            s_U = max(maximum(abs, US_raw), 1.0)
            s_M = max(maximum(abs, mS_raw), 1.0)
        end

        # F. Joint Anderson on [U_S; m_S] with per-block scaling.
        x_vec = vcat(US_old ./ s_U, mS_old ./ s_M)
        f_vec = vcat(US_raw ./ s_U, mS_raw ./ s_M)

        x_new_s = sim.use_anderson ?
                  anderson1_update!(aa, x_vec, f_vec) : f_vec

        US_new = x_new_s[1:Nx]           .* s_U
        mS_new = max.(x_new_s[Nx+1:end] .* s_M, 0.0)

        copyto!(sc.U, US_new)
        mS_cur = mS_new

        # G. Refresh the cross-market contribution and the τ-gate policy
        #    for the next pass (both lagged one global iteration).
        @inbounds for ix in 1:Nx
            uc.duS_carry[ix] = max(clamp(sc.d[ix], 0.0, 1.0) * sc.u[ix], 0.0)
            uc.d_carry[ix]   = clamp(sc.d[ix], 0.0, 1.0)
        end

        # Convergence on (ΔU_S, Δm_S).
        dU = supnorm(US_new, US_old)
        dM = supnorm(mS_new, mS_old)
        d  = max(dU, dM)

        if sim.verbose >= 1 && (it == 1 || it % sim.verbose_stride == 0)
            @printf("[global it=%d]  maxΔ=%.3e  (ΔU_S=%.3e  Δm_S=%.3e)  θ_U=%.4f  θ_S=%.4f\n",
                    it, d, dU, dM, uc.θ, sc.θ)
        end

        if d < sim.tol_global
            streak += 1
            if streak >= sim.conv_streak
                global_converged = true
                sim.verbose >= 1 && @printf(
                    "[global]  converged it=%d  d=%.3e\n", it, d)
                break
            end
        else
            streak = 0
        end

        if it == sim.maxit_global && sim.verbose >= 1
            @printf("[global]  maxit reached  it=%d  d=%.3e\n", it, d)
        end
    end

    return SolveResult(last_conv_U, last_conv_S, global_converged)
end

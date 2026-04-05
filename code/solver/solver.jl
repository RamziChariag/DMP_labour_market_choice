############################################################
# solver.jl — Global equilibrium solver
#
# Exports:
#   solve_model(common, regime, unsk_par, skl_par, sim; Nx, Np_U, Np_S)
#       → builds a fresh Model, solves it, returns it.
#
#   solve_model!(model)
#       → solves an already-initialised Model in place.
#
# Algorithm (global fixed-point):
#   Link variables: U_S(x) (skilled unemployment value, length Nx)
#                   m_S(x) (trained worker inflow density, length Nx)
#
#   1. Given U_S in sc.U, solve unskilled block → uc.t
#   2. m_S_raw = (φ/ν) · uc.t
#   3. Given m_S_raw, solve skilled block → updates sc.U
#   4. Anderson(m=1) mix on joint [U_S; m_S]
#   5. Write mixed U_S back to sc.U; repeat.
############################################################

# packages loaded by main.jl

# ============================================================
# SolveResult — convergence report from one model solve
# ============================================================

"""
    SolveResult

Records whether each layer of the solver converged on the final
global iteration.  All three must be `true` for the solve to be
considered valid in SMM.

Fields
──────
  converged_U      :: Bool    unskilled outer loop converged
  converged_S      :: Bool    skilled outer loop converged
  converged_global :: Bool    global fixed-point converged
  ok               :: Bool    all three true (convenience)
"""
struct SolveResult
    converged_U      :: Bool
    converged_S      :: Bool
    converged_global :: Bool
    ok               :: Bool
end

SolveResult(cU::Bool, cS::Bool, cG::Bool) = SolveResult(cU, cS, cG, cU && cS && cG)

# ---------------------------------------------------------------------------
# Cache initialisation (internal helper)
# ---------------------------------------------------------------------------
function _initialise_caches(
    common   :: CommonParams,
    regime   :: RegimeParams,
    unsk_par :: UnskilledParams,
    skl_par  :: SkilledParams,
    grids    :: CommonGrids,
    u_grids  :: UnskilledGrids,
    s_grids  :: SkilledGrids
)
    Nx   = length(grids.x)
    Np_S = length(s_grids.p)

    r = common.r;   ν = common.ν;   φ = common.φ

    US_guess     = regime.bS / (r + ν)
    Usearch_init = fill(regime.bU / (r + ν), Nx)
    T_init       = [(regime.bT + φ * US_guess) / (r + φ + ν) for _ in 1:Nx]
    U_init       = max.(Usearch_init, T_init)
    t_seed       = [(ν / (ν + φ + ν)) * grids.ℓ[ix] for ix in 1:Nx]

    # ── Parameter-informed initial cutoffs ───────────────────────────────
    # pstar_U(x) ≈ (r+ν)·U / (PU·x);  U ≈ bU/(r+ν)  →  pstar_U ≈ bU/(PU·x)
    # Clamped away from corners so the solver has a sensible starting bracket.
    PU_init = max(regime.PU, 1e-6)
    pstar_U_init = [clamp(regime.bU / (PU_init * max(grids.x[ix], 1e-3)), 0.05, 0.90)
                    for ix in 1:Nx]

    # pstar_S(x) ≈ (r+ν)·US / (PS·x);  US ≈ bS/(r+ν)  →  pstar_S ≈ bS/(PS·x)
    PS_init = max(regime.PS, 1e-6)
    US_guess_init = regime.bS / max(r + ν, 1e-6)
    pstar_S_init = [clamp(regime.bS / (PS_init * max(grids.x[ix], 1e-3)), 0.05, 0.90)
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
        θ         = 0.5,     # avoid 1.0 sentinel; will be overwritten on first outer iteration
    )

    US_init = fill(regime.bS / (r + ν), Nx)

    sc = SkilledCache(
        U     = US_init,
        E0    = zeros(Nx, Np_S),
        E1    = zeros(Nx, Np_S),
        J0    = zeros(Nx, Np_S),
        J1    = zeros(Nx, Np_S),
        pstar = pstar_S_init,
        poj   = poj_init,
        u     = zeros(Nx),
        e     = zeros(Nx, Np_S),
        θ     = 0.5,     # avoid 1.0 sentinel; will be overwritten on first outer iteration
    )

    return uc, sc
end


# ---------------------------------------------------------------------------
# solve_model — allocates fresh model and solves
# ---------------------------------------------------------------------------
"""
    solve_model(common, regime, unsk_par, skl_par, sim; Nx, Np_U, Np_S)
        → (Model, SolveResult)

Builds grids and caches from scratch, runs the global fixed-point loop,
and returns the solved Model together with a SolveResult convergence report.
"""
function solve_model(
    common   :: CommonParams,
    regime   :: RegimeParams,
    unsk_par :: UnskilledParams,
    skl_par  :: SkilledParams,
    sim      :: SimParams;
    Nx   :: Int = 200,
    Np_U :: Int = 200,
    Np_S :: Int = 200
)
    xgrid,   wx   = build_gl_grid(Nx)
    pgrid_U, wp_U = build_gl_grid(Np_U)
    pgrid_S, wp_S = build_gl_grid(Np_S)

    ℓvals = pdf.(Beta(common.a_ℓ, common.b_ℓ), xgrid)

    grids   = CommonGrids(x = xgrid, wx = wx, ℓ = ℓvals)
    u_grids = UnskilledGrids(p = pgrid_U, wp = wp_U)
    s_grids = SkilledGrids(p = pgrid_S, wp = wp_S)

    s_pre = build_skilled_precomp(s_grids, regime)

    uc, sc = _initialise_caches(common, regime, unsk_par, skl_par,
                                 grids, u_grids, s_grids)

    model = Model(
        common     = common,
        regime     = regime,
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
# solve_model! — in-place solver (operates on an existing Model)
# ---------------------------------------------------------------------------
"""
    solve_model!(model) → SolveResult

Run the global fixed-point loop on `model`, updating caches in place.
Returns a `SolveResult` indicating whether each layer converged.
All three flags (converged_U, converged_S, converged_global) must be
`true` for `result.ok` to be `true`.
"""
function solve_model!(model::Model) :: SolveResult
    cp  = model.common
    sc  = model.skl_cache
    uc  = model.unsk_cache
    sim = model.sim

    φ  = cp.φ;   ν  = cp.ν
    Nx = length(model.grids.x)

    aa     = Anderson1(2 * Nx)
    mS_cur = [max((φ / ν) * uc.t[ix], 0.0) for ix in 1:Nx]

    # Track convergence flags from the LAST global iteration's block solves
    last_conv_U = false
    last_conv_S = false
    global_converged = false

    streak = 0
    for it in 1:sim.maxit_global

        US_old = copy(sc.U)
        mS_old = copy(mS_cur)

        # Step A: unskilled block — returns Bool
        last_conv_U = solve_unskilled_block!(model; US_in = sc.U)

        # NaN propagation check before continuing — covers θ and all key outputs
        if !isfinite(uc.θ) || any(!isfinite, uc.t) || any(!isfinite, uc.U) || any(!isfinite, uc.pstar)
            sim.verbose >= 1 && @printf("[global]  NaN/Inf in unskilled block at it=%d — aborting\n", it)
            return SolveResult(false, false, false)
        end

        mS_raw = [max((φ / ν) * uc.t[ix], 0.0) for ix in 1:Nx]

        # Step B: skilled block — returns Bool
        last_conv_S = solve_skilled_block!(model; mS_in = mS_raw)

        if !isfinite(sc.θ) || any(!isfinite, sc.U) || any(!isfinite, sc.pstar)
            sim.verbose >= 1 && @printf("[global]  NaN/Inf in skilled block at it=%d — aborting\n", it)
            return SolveResult(false, false, false)
        end

        US_raw = copy(sc.U)

        # Step C: Anderson mixing on joint [U_S; m_S]
        x_vec = vcat(US_old, mS_old)
        f_vec = vcat(US_raw, mS_raw)

        x_new = sim.use_anderson ?
                anderson1_update!(aa, x_vec, f_vec) : f_vec

        US_new = x_new[1:Nx]
        mS_new = max.(x_new[Nx+1:end], 0.0)

        copyto!(sc.U, US_new)
        mS_cur = mS_new

        # Step D: convergence
        dU = supnorm(US_new, US_old)
        dM = supnorm(mS_new, mS_old)
        d  = max(dU, dM)

        if sim.verbose >= 1 && (it == 1 || it % sim.verbose_stride == 0)
            @printf("[global it=%d]  maxΔ=%.3e  (ΔUS=%.3e  ΔmS=%.3e)  θU=%.4f  θS=%.4f\n",
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
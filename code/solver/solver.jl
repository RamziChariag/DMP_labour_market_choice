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

    uc = UnskilledCache(
        Usearch   = Usearch_init,
        U         = U_init,
        T         = T_init,
        Jfrontier = zeros(Nx),
        pstar     = fill(0.10, Nx),
        τT        = zeros(Nx),
        u         = 0.4 .* grids.ℓ,
        t         = t_seed,
        θ         = 1.0,
    )

    US_init = fill(regime.bS / (r + ν), Nx)

    sc = SkilledCache(
        U     = US_init,
        E0    = zeros(Nx, Np_S),
        E1    = zeros(Nx, Np_S),
        J0    = zeros(Nx, Np_S),
        J1    = zeros(Nx, Np_S),
        pstar = fill(0.10, Nx),
        poj   = fill(0.60, Nx),
        u     = zeros(Nx),
        e     = zeros(Nx, Np_S),
        θ     = 1.0,
    )

    return uc, sc
end


# ---------------------------------------------------------------------------
# solve_model — allocates fresh model and solves
# ---------------------------------------------------------------------------
"""
    solve_model(common, regime, unsk_par, skl_par, sim; Nx, Np_U, Np_S) → Model

Builds grids and caches from scratch, runs the global fixed-point loop,
and returns the solved `Model`.
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
) :: Model

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

    solve_model!(model)
    return model
end


# ---------------------------------------------------------------------------
# solve_model! — in-place solver (operates on an existing Model)
# ---------------------------------------------------------------------------
"""
    solve_model!(model)

Run the global fixed-point loop on `model`, updating caches in place.
Use this when the model struct was built via `initialise_model()`.
"""
function solve_model!(model::Model)
    cp  = model.common
    sc  = model.skl_cache
    uc  = model.unsk_cache
    sim = model.sim

    φ  = cp.φ;   ν  = cp.ν
    Nx = length(model.grids.x)

    aa     = Anderson1(2 * Nx)
    mS_cur = [max((φ / ν) * uc.t[ix], 0.0) for ix in 1:Nx]

    streak = 0
    for it in 1:sim.maxit_global

        US_old = copy(sc.U)
        mS_old = copy(mS_cur)

        # Step A: unskilled block
        solve_unskilled_block!(model; US_in = sc.U)

        mS_raw = [max((φ / ν) * uc.t[ix], 0.0) for ix in 1:Nx]

        # Step B: skilled block
        solve_skilled_block!(model; mS_in = mS_raw)

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
    return nothing
end

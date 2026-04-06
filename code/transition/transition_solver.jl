############################################################
# transition_solver.jl — Backward-forward transition dynamics
#
# Core algorithm (Section 11.9 of the model notes):
#   Step 0  Compute stationary equilibria z₀, z₁  (external)
#   Step 1  Initialise tightness paths + distributions
#   Step 2  Backward pass: value functions & policies
#   Step 3  Forward pass:  laws of motion for distributions
#   Step 4  Update tightness via free entry
#   Step 5  Iterate 2-4 until convergence
#
# Public API
#   solve_transition(model_z0, model_z1, tp; scenario)
#       → TransitionResult
############################################################

using Base.Threads: @threads, nthreads, threadid

# ════════════════════════════════════════════════════════════
#  Top-level driver
# ════════════════════════════════════════════════════════════

"""
    solve_transition(model_z0, model_z1, tp; scenario) → TransitionResult

Solve the backward-forward transition from regime z₀ to z₁.

Both models must already be solved (call `solve_model!` first).
`scenario` is a label stored in the result (:fc or :covid).
"""
function solve_transition(
    model_z0 :: Model,
    model_z1 :: Model,
    tp       :: TransitionParams;
    scenario :: Symbol = :unknown,
)
    Nt = tp.N_steps + 1

    if tp.verbose
        @printf("\n%s\n", "="^65)
        @printf("  Transition dynamics  (%s)  T=%.1f  N=%d  dt=%.4f\n",
                scenario, tp.T_max, tp.N_steps, tp.dt)
        @printf("%s\n\n", "="^65)
    end

    # ── Step 1: initialise path ──────────────────────────
    path = _initialise_path(model_z0, model_z1, tp)

    # ── Steps 2-5: iterate ───────────────────────────────
    converged  = false
    final_dist = Inf
    n_iter     = 0

    for iter in 1:tp.maxit
        θU_old = copy(path.θU)
        θS_old = copy(path.θS)

        # Step 2 — backward pass (values & policies)
        _backward_pass!(path, model_z1, tp)

        # Step 3 — forward pass (distributions)
        _forward_pass!(path, model_z1, tp)

        # Step 4 — update tightness from free entry
        _update_tightness!(path, model_z1, tp)

        # Step 5 — convergence check
        dist = max(maximum(abs, path.θU .- θU_old),
                   maximum(abs, path.θS .- θS_old))

        n_iter = iter
        final_dist = dist

        if tp.verbose && (iter <= 3 || iter % 5 == 0 || dist < tp.tol)
            @printf("  [transition it=%3d]  ‖Δθ‖ = %.3e\n", iter, dist)
            flush(stdout)
        end

        if dist < tp.tol
            converged = true
            tp.verbose && @printf("  ✓ converged in %d iterations (dist=%.3e)\n\n",
                                   iter, dist)
            break
        end
    end

    if !converged && tp.verbose
        @printf("  ⚠  maxit reached (%d iterations, dist=%.3e)\n\n",
                tp.maxit, final_dist)
    end

    # ── Build serialisable result ────────────────────────
    return _build_result(path, model_z0, model_z1, tp,
                         scenario, converged, n_iter, final_dist)
end


# ════════════════════════════════════════════════════════════
#  Step 1: Initialisation
# ════════════════════════════════════════════════════════════

function _initialise_path(model_z0::Model, model_z1::Model, tp::TransitionParams)
    path = TransitionPath(model_z0, model_z1, tp)
    Nt  = tp.N_steps + 1
    Nx  = length(model_z0.grids.x)
    NpS = length(model_z0.skl_grids.p)

    # ── Tightness: linear interpolation z₀ → z₁ ─────────
    θU0 = model_z0.unsk_cache.θ;  θU1 = model_z1.unsk_cache.θ
    θS0 = model_z0.skl_cache.θ;   θS1 = model_z1.skl_cache.θ
    for n in 1:Nt
        α = (n - 1) / tp.N_steps
        path.θU[n] = θU0 + α * (θU1 - θU0)
        path.θS[n] = θS0 + α * (θS1 - θS0)
    end

    # ── Distributions at t = 0: copy from z₀ ────────────
    @inbounds for ix in 1:Nx
        path.uU[ix, 1] = model_z0.unsk_cache.u[ix]
        path.tU[ix, 1] = model_z0.unsk_cache.t[ix]
        path.uS[ix, 1] = model_z0.skl_cache.u[ix]
    end

    # mS(x, t=0):  skilled segment mass per type
    # mS(x) = uS(x) + ∫ eS(x,p) dp
    wpS = model_z0.skl_grids.wp
    @inbounds for ix in 1:Nx
        mass = model_z0.skl_cache.u[ix]
        for jp in 1:NpS
            mass += model_z0.skl_cache.e[ix, jp] * wpS[jp]
        end
        path.mS[ix, 1] = mass
    end

    @inbounds for ix in 1:Nx, jp in 1:NpS
        path.eS[ix, jp, 1] = model_z0.skl_cache.e[ix, jp]
    end

    # ── Terminal values at t = T_max: copy from z₁ ──────
    n_end = Nt
    @inbounds for ix in 1:Nx
        path.U[ix, n_end]        = model_z1.unsk_cache.U[ix]
        path.Usearch[ix, n_end]  = model_z1.unsk_cache.Usearch[ix]
        path.T_val[ix, n_end]    = model_z1.unsk_cache.T[ix]
        path.Jfrontier[ix, n_end] = model_z1.unsk_cache.Jfrontier[ix]
        path.τT[ix, n_end]       = model_z1.unsk_cache.τT[ix]
        path.pstar_U[ix, n_end]  = model_z1.unsk_cache.pstar[ix]

        path.US[ix, n_end]       = model_z1.skl_cache.U[ix]
        path.pstar_S[ix, n_end]  = model_z1.skl_cache.pstar[ix]
        path.poj[ix, n_end]      = model_z1.skl_cache.poj[ix]
    end

    @inbounds for ix in 1:Nx, jp in 1:NpS
        path.E0[ix, jp, n_end] = model_z1.skl_cache.E0[ix, jp]
        path.E1[ix, jp, n_end] = model_z1.skl_cache.E1[ix, jp]
        path.J0[ix, jp, n_end] = model_z1.skl_cache.J0[ix, jp]
        path.J1[ix, jp, n_end] = model_z1.skl_cache.J1[ix, jp]
    end

    # ── Also fill terminal distributions from z₁ ────────
    @inbounds for ix in 1:Nx
        path.uU[ix, n_end] = model_z1.unsk_cache.u[ix]
        path.tU[ix, n_end] = model_z1.unsk_cache.t[ix]
        path.uS[ix, n_end] = model_z1.skl_cache.u[ix]
    end
    wpS1 = model_z1.skl_grids.wp
    @inbounds for ix in 1:Nx
        mass = model_z1.skl_cache.u[ix]
        for jp in 1:NpS
            mass += model_z1.skl_cache.e[ix, jp] * wpS1[jp]
        end
        path.mS[ix, n_end] = mass
    end
    @inbounds for ix in 1:Nx, jp in 1:NpS
        path.eS[ix, jp, n_end] = model_z1.skl_cache.e[ix, jp]
    end

    # ── Value functions interior: linear interpolation ───
    for n in 2:(Nt - 1)
        α = (n - 1) / tp.N_steps
        @inbounds for ix in 1:Nx
            path.U[ix, n]        = (1 - α) * path.U[ix, 1]        + α * path.U[ix, n_end]
            path.Usearch[ix, n]  = (1 - α) * path.Usearch[ix, 1]  + α * path.Usearch[ix, n_end]
            path.T_val[ix, n]    = (1 - α) * path.T_val[ix, 1]    + α * path.T_val[ix, n_end]
            path.Jfrontier[ix, n] = (1 - α) * path.Jfrontier[ix, 1] + α * path.Jfrontier[ix, n_end]
            path.τT[ix, n]       = (1 - α) * path.τT[ix, 1]       + α * path.τT[ix, n_end]
            path.pstar_U[ix, n]  = (1 - α) * path.pstar_U[ix, 1]  + α * path.pstar_U[ix, n_end]
            path.US[ix, n]       = (1 - α) * path.US[ix, 1]       + α * path.US[ix, n_end]
            path.pstar_S[ix, n]  = (1 - α) * path.pstar_S[ix, 1]  + α * path.pstar_S[ix, n_end]
            path.poj[ix, n]      = (1 - α) * path.poj[ix, 1]      + α * path.poj[ix, n_end]
        end
    end

    # Initial values at t=0 (from z₀ steady state for warm-start)
    @inbounds for ix in 1:Nx
        path.U[ix, 1]        = model_z0.unsk_cache.U[ix]
        path.Usearch[ix, 1]  = model_z0.unsk_cache.Usearch[ix]
        path.T_val[ix, 1]    = model_z0.unsk_cache.T[ix]
        path.Jfrontier[ix, 1] = model_z0.unsk_cache.Jfrontier[ix]
        path.τT[ix, 1]       = model_z0.unsk_cache.τT[ix]
        path.pstar_U[ix, 1]  = model_z0.unsk_cache.pstar[ix]
        path.US[ix, 1]       = model_z0.skl_cache.U[ix]
        path.pstar_S[ix, 1]  = model_z0.skl_cache.pstar[ix]
        path.poj[ix, 1]      = model_z0.skl_cache.poj[ix]
    end
    @inbounds for ix in 1:Nx, jp in 1:NpS
        path.E0[ix, jp, 1] = model_z0.skl_cache.E0[ix, jp]
        path.E1[ix, jp, 1] = model_z0.skl_cache.E1[ix, jp]
        path.J0[ix, jp, 1] = model_z0.skl_cache.J0[ix, jp]
        path.J1[ix, jp, 1] = model_z0.skl_cache.J1[ix, jp]
    end

    return path
end


# ════════════════════════════════════════════════════════════
#  Step 2: Backward pass  (value functions & policies)
# ════════════════════════════════════════════════════════════

"""
At each date n (backward from Nt-1 to 1), compute equilibrium
value functions and policies under z₁ parameters at the given
tightness θ(n).

Strategy: overwrite the model_z1 caches with warm-start from n+1,
set θ = θ(n), then run the inner loops (which solve the stationary
Bellman given that θ).  This is correct because, with θ given,
the value functions are uniquely pinned down by the parameters;
the backward structure enters only through the θ path.
"""
function _backward_pass!(path::TransitionPath, model::Model, tp::TransitionParams)
    Nt  = tp.N_steps + 1
    Nx  = length(model.grids.x)
    NpS = length(model.skl_grids.p)

    uc = model.unsk_cache
    sc = model.skl_cache

    cp = model.common
    rp = model.regime
    up = model.unsk_par
    sp = model.skl_par
    gp = model.grids
    ug = model.unsk_grids
    sg = model.skl_grids
    pre = model.skl_pre

    αU = rp.α_U
    wG = build_unskilled_G_weights(ug.p, ug.wp, αU)

    denom_nb = max(1.0 - sp.β, 1e-14)
    wΓ = pre.γvals .* sg.wp

    # Scratch arrays for the unskilled surplus computation
    NpU = length(ug.p)
    Svec = zeros(Float64, NpU)

    for n in (Nt - 1):-1:1

        # ── Set tightness ────────────────────────────────
        uc.θ = path.θU[n]
        sc.θ = path.θS[n]

        fU = jobfinding_rate(uc.θ, up.μ, up.η)
        fS = jobfinding_rate(sc.θ, sp.μ, sp.η)

        # ────────────────────────────────────────────────
        # SKILLED BLOCK
        # ────────────────────────────────────────────────

        # Warm-start skilled caches from n+1
        @inbounds for ix in 1:Nx
            sc.U[ix]     = path.US[ix, n + 1]
            sc.pstar[ix] = path.pstar_S[ix, n + 1]
            sc.poj[ix]   = path.poj[ix, n + 1]
        end
        @inbounds for ix in 1:Nx, jp in 1:NpS
            sc.E0[ix, jp] = path.E0[ix, jp, n + 1]
            sc.E1[ix, jp] = path.E1[ix, jp, n + 1]
            sc.J0[ix, jp] = path.J0[ix, jp, n + 1]
            sc.J1[ix, jp] = path.J1[ix, jp, n + 1]
        end

        # Use mS from the current path (from forward pass or init)
        mS_n = @view path.mS[:, max(n, 1)]

        # Run the skilled inner loop to convergence at this θS
        skilled_inner_loop!(model; mS_in = mS_n)

        # ── Update skilled cutoffs from converged surfaces ──
        @inbounds for ix in 1:Nx
            x_val     = gp.x[ix]
            U_x       = sc.U[ix]
            pstar_cur = clamp01(sc.pstar[ix])
            j0_prev   = pcut_index(sg.p, pstar_cur)
            j0_soft   = max(j0_prev - 1, 1)

            # Recompute tail integral I
            tailE = zeros(Float64, NpS)
            acc = 0.0
            @inbounds for j in NpS:-1:1
                Smax_j = max(sc.J0[ix, j], sc.J1[ix, j]) / denom_nb
                acc       += Smax_j * wΓ[j]
                tailE[j]   = acc
            end

            # Raw surplus for cutoff search
            Smax_raw = zeros(Float64, NpS)
            diff_raw = zeros(Float64, NpS)
            base = cp.r + cp.ν + sp.ξ + sp.λ
            I_val = tailE[j0_soft]

            @inbounds for j in 1:NpS
                pj = sg.p[j]
                raw_S0 = (rp.PS * x_val * pj - (cp.r + cp.ν) * U_x + sp.λ * I_val) / base
                tail_mass_j = pre.tail_weights[j]
                tail_Emax_j = tailE[max(j, j0_soft)]
                raw_S1 = (rp.PS * x_val * pj - (cp.r + cp.ν) * U_x - sp.σ + sp.λ * I_val +
                           fS * sp.β * tail_Emax_j) / (base + fS * tail_mass_j)
                Smax_raw[j] = max(raw_S0, raw_S1)
                diff_raw[j] = raw_S1 - raw_S0
            end

            sc.pstar[ix] = clamp01(find_cutoff_from_j0(sg.p, Smax_raw, j0_prev))
            raw_poj      = clamp01(find_poj_from_diff_grid(sg.p, diff_raw, sc.pstar[ix]))
            sc.poj[ix]   = max(sc.pstar[ix], raw_poj)
        end

        # Store skilled results into path
        @inbounds for ix in 1:Nx
            path.US[ix, n]      = sc.U[ix]
            path.pstar_S[ix, n] = sc.pstar[ix]
            path.poj[ix, n]     = sc.poj[ix]
        end
        @inbounds for ix in 1:Nx, jp in 1:NpS
            path.E0[ix, jp, n] = sc.E0[ix, jp]
            path.E1[ix, jp, n] = sc.E1[ix, jp]
            path.J0[ix, jp, n] = sc.J0[ix, jp]
            path.J1[ix, jp, n] = sc.J1[ix, jp]
        end

        # ────────────────────────────────────────────────
        # UNSKILLED BLOCK
        # ────────────────────────────────────────────────

        # Warm-start unskilled caches from n+1
        @inbounds for ix in 1:Nx
            uc.U[ix]        = path.U[ix, n + 1]
            uc.Usearch[ix]  = path.Usearch[ix, n + 1]
            uc.T[ix]        = path.T_val[ix, n + 1]
            uc.Jfrontier[ix] = path.Jfrontier[ix, n + 1]
            uc.pstar[ix]    = path.pstar_U[ix, n + 1]
            uc.τT[ix]       = path.τT[ix, n + 1]
        end

        # Also need uU, tU for the unskilled inner loop's outer calls
        @inbounds for ix in 1:Nx
            uc.u[ix] = path.uU[ix, max(n, 1)]
            uc.t[ix] = path.tU[ix, max(n, 1)]
        end

        # Unskilled inner loop: solves Usearch, U, T, Jfrontier, τT
        inner = unskilled_inner_loop!(model; US_in = sc.U)

        # Update pstar_U from surplus
        pstar_new = zeros(Float64, Nx)
        update_pstar_from_surplus!(pstar_new, model, inner.Ivec)
        copyto!(uc.pstar, pstar_new)

        # Store unskilled results into path
        @inbounds for ix in 1:Nx
            path.U[ix, n]        = uc.U[ix]
            path.Usearch[ix, n]  = uc.Usearch[ix]
            path.T_val[ix, n]    = uc.T[ix]
            path.Jfrontier[ix, n] = uc.Jfrontier[ix]
            path.τT[ix, n]       = uc.τT[ix]
            path.pstar_U[ix, n]  = uc.pstar[ix]
        end
    end
end


# ════════════════════════════════════════════════════════════
#  Step 3: Forward pass  (laws of motion for distributions)
# ════════════════════════════════════════════════════════════

"""
Evolve distributions forward from t = 0 to t = T_max using
explicit Euler, given the policy paths from the backward pass.

Laws of motion (eqs 34-38 of the notes):
  ∂_t t(x)  = τ(x)·uU(x) − (φ+ν)·t(x)
  ∂_t uU(x) = ν·ℓ(x) + δU(x)·eU(x) − (fU + τ(x) + ν)·uU(x)
  ∂_t uS(x) = φ·t(x) + (ξ+δ^end_S(x))·eS_tot(x) − (ν + fS·(1−Γ(p*S)))·uS(x)
  ∂_t mS(x) = φ·t(x) − ν·mS(x)
  ∂_t eS(x,p) = inflow − outflow
"""
function _forward_pass!(path::TransitionPath, model::Model, tp::TransitionParams)
    Nt  = tp.N_steps + 1
    Nx  = length(model.grids.x)
    NpS = length(model.skl_grids.p)
    dt  = tp.dt

    cp  = model.common
    rp  = model.regime
    up  = model.unsk_par
    sp  = model.skl_par
    pre = model.skl_pre
    gp  = model.grids
    sg  = model.skl_grids

    φ  = cp.φ;   ν  = cp.ν
    λU = up.λ;   αU = rp.α_U
    ξS = sp.ξ;   λS = sp.λ

    for n in 1:(Nt - 1)

        # Job-finding rates at this date
        fU = jobfinding_rate(path.θU[n], up.μ, up.η)
        fS = jobfinding_rate(path.θS[n], sp.μ, sp.η)

        # ── Training density ─────────────────────────────
        @inbounds for ix in 1:Nx
            τ_ix   = path.τT[ix, n]
            uU_ix  = path.uU[ix, n]
            tU_ix  = path.tU[ix, n]

            dt_train = τ_ix * uU_ix - (φ + ν) * tU_ix
            path.tU[ix, n + 1] = max(tU_ix + dt * dt_train, 0.0)
        end

        # ── Unskilled unemployed ─────────────────────────
        @inbounds for ix in 1:Nx
            ℓ_ix    = gp.ℓ[ix]
            uU_ix   = path.uU[ix, n]
            tU_ix   = path.tU[ix, n]
            mS_ix   = path.mS[ix, n]
            τ_ix    = path.τT[ix, n]
            pstar_ix = clamp01(path.pstar_U[ix, n])

            # Unskilled segment mass and employment (residual)
            mU_ix = max(ℓ_ix - mS_ix, 0.0)
            eU_ix = max(mU_ix - uU_ix - tU_ix, 0.0)

            δU = λU * G_cdf_unskilled(pstar_ix, αU)

            du_dt = ν * ℓ_ix + δU * eU_ix - (fU + τ_ix + ν) * uU_ix
            path.uU[ix, n + 1] = max(uU_ix + dt * du_dt, 0.0)
        end

        # ── Skilled segment mass ─────────────────────────
        @inbounds for ix in 1:Nx
            tU_ix = path.tU[ix, n]
            mS_ix = path.mS[ix, n]

            dm_dt = φ * tU_ix - ν * mS_ix
            path.mS[ix, n + 1] = max(mS_ix + dt * dm_dt, 0.0)
        end

        # ── Skilled unemployed ───────────────────────────
        @inbounds for ix in 1:Nx
            tU_ix    = path.tU[ix, n]
            uS_ix    = path.uS[ix, n]
            pstar_ix = clamp01(path.pstar_S[ix, n])

            # Total skilled employment
            eS_tot = 0.0
            for jp in 1:NpS
                eS_tot += path.eS[ix, jp, n] * sg.wp[jp]
            end

            j_ps    = pcut_index(sg.p, pstar_ix)
            Γ_pstar = pre.Γvals[j_ps]
            δS_sep  = ξS + λS * Γ_pstar   # ξ + endogenous separation

            du_dt = φ * tU_ix + δS_sep * eS_tot -
                    (ν + fS * (1.0 - Γ_pstar)) * uS_ix
            path.uS[ix, n + 1] = max(uS_ix + dt * du_dt, 0.0)
        end

        # ── Skilled employment density e_S(x, p) ────────
        @inbounds for ix in 1:Nx
            pstar_ix = clamp01(path.pstar_S[ix, n])
            poj_ix   = clamp01(path.poj[ix, n])
            j0       = pcut_index(sg.p, pstar_ix)
            uS_ix    = path.uS[ix, n]

            cum_e = 0.0   # ∫_{p' < p} eS(x, p') dp'

            for jp in 1:NpS
                e_old = path.eS[ix, jp, n]
                pj    = sg.p[jp]

                if jp < j0
                    # Below reservation quality — employment destroyed
                    path.eS[ix, jp, n + 1] = 0.0
                    cum_e += e_old * sg.wp[jp]
                    continue
                end

                γj = pre.γvals[jp]
                Γj = pre.Γvals[jp]

                # Inflow from unemployment (new hires)
                inflow_u = fS * γj * uS_ix

                # Inflow from quality shocks (redistribution from below)
                inflow_λ = λS * γj * cum_e

                # OJS: employed workers searching on the job
                # Workers at (x, p) do OJS if p < poj(x)
                is_ojs = (pj < poj_ix) ? 1.0 : 0.0

                # Outflow: quality shock at rate λ removes worker from (x,p)
                # regardless of new draw (they move to a new p'), plus
                # demographic exit (ν), exogenous separation (ξ), and
                # OJS outflow if searching on the job.
                outflow = (ν + ξS + λS + is_ojs * fS * (1.0 - Γj)) * e_old

                path.eS[ix, jp, n + 1] = max(e_old + dt * (inflow_u + inflow_λ - outflow), 0.0)
                cum_e += e_old * sg.wp[jp]
            end
        end
    end
end


# ════════════════════════════════════════════════════════════
#  Step 4: Update tightness via free entry
# ════════════════════════════════════════════════════════════

"""
At each date n, compute the free-entry tightness from current
distributions and firm values, then apply dampening.
"""
function _update_tightness!(path::TransitionPath, model::Model, tp::TransitionParams)
    Nt  = tp.N_steps + 1
    Nx  = length(model.grids.x)
    NpS = length(model.skl_grids.p)

    gp  = model.grids
    sg  = model.skl_grids
    pre = model.skl_pre
    up  = model.unsk_par
    sp  = model.skl_par

    wΓ  = pre.γvals .* sg.wp
    denom_nb = max(1.0 - sp.β, 1e-14)

    for n in 1:Nt

        # ── Unskilled free entry ─────────────────────────
        J_u_sum = 0.0
        u_U_sum = 0.0
        @inbounds for ix in 1:Nx
            wx  = gp.wx[ix]
            u_ix = path.uU[ix, n]
            J_u_sum += path.Jfrontier[ix, n] * u_ix * wx
            u_U_sum += u_ix * wx
        end

        if J_u_sum > 1e-12 && u_U_sum > 1e-12
            q_U     = up.k * u_U_sum / J_u_sum
            θU_prop = clamp(theta_from_q(q_U, up.μ, up.η), 1e-14, 100.0)
        else
            θU_prop = path.θU[n]
        end
        path.θU[n] = (1.0 - tp.damp) * path.θU[n] + tp.damp * θU_prop

        # ── Skilled free entry ───────────────────────────
        num_S = 0.0
        den_S = 0.0

        @inbounds for ix in 1:Nx
            wx_ix     = gp.wx[ix]
            pstar_ix  = clamp01(path.pstar_S[ix, n])
            j0        = pcut_index(sg.p, pstar_ix)

            # Tail integral of J from p*
            tailJ = 0.0
            for jp in NpS:-1:j0
                J_max = max(path.J0[ix, jp, n], path.J1[ix, jp, n])
                tailJ += J_max * wΓ[jp]
            end

            # Seekers = unemployed + employed doing OJS
            u_ix = path.uS[ix, n]
            seeker_e = 0.0
            for jp in 1:NpS
                if path.E1[ix, jp, n] >= path.E0[ix, jp, n]
                    seeker_e += path.eS[ix, jp, n] * sg.wp[jp]
                end
            end

            seekers = u_ix + seeker_e
            num_S  += wx_ix * seekers * tailJ
            den_S  += wx_ix * seekers
        end

        if num_S > 1e-12 && den_S > 1e-12
            Jbar_S  = num_S / den_S
            q_S     = sp.k / Jbar_S
            θS_prop = clamp(theta_from_q(q_S, sp.μ, sp.η), 1e-14, 100.0)
        else
            θS_prop = path.θS[n]
        end
        path.θS[n] = (1.0 - tp.damp) * path.θS[n] + tp.damp * θS_prop
    end
end


# ════════════════════════════════════════════════════════════
#  Build serialisable result
# ════════════════════════════════════════════════════════════

function _build_result(
    path     :: TransitionPath,
    model_z0 :: Model,
    model_z1 :: Model,
    tp       :: TransitionParams,
    scenario :: Symbol,
    converged :: Bool,
    n_iter    :: Int,
    final_dist :: Float64,
)
    Nt  = tp.N_steps + 1
    Nx  = length(model_z1.grids.x)
    NpS = length(model_z1.skl_grids.p)

    gp  = model_z1.grids
    sg  = model_z1.skl_grids
    up  = model_z1.unsk_par
    sp  = model_z1.skl_par
    rp  = model_z1.regime
    cp  = model_z1.common

    wx  = gp.wx
    wpS = sg.wp

    # ── Aggregate time-series ────────────────────────────
    fU_path             = zeros(Nt)
    fS_path             = zeros(Nt)
    ur_U_path           = zeros(Nt)
    ur_S_path           = zeros(Nt)
    ur_total_path       = zeros(Nt)
    skilled_share_path  = zeros(Nt)
    training_share_path = zeros(Nt)
    mean_wage_U_path    = zeros(Nt)
    mean_wage_S_path    = zeros(Nt)

    for n in 1:Nt
        fU_path[n] = jobfinding_rate(path.θU[n], up.μ, up.η)
        fS_path[n] = jobfinding_rate(path.θS[n], sp.μ, sp.η)

        agg_uU = 0.0;  agg_tU = 0.0;  agg_eU = 0.0;  agg_mU = 0.0
        agg_uS = 0.0;  agg_eS = 0.0;  agg_mS = 0.0
        agg_pop = 0.0

        wage_num_U = 0.0;  wage_den_U = 0.0
        wage_num_S = 0.0;  wage_den_S = 0.0

        for ix in 1:Nx
            w = wx[ix]
            ℓ_ix  = gp.ℓ[ix]
            uU_ix = path.uU[ix, n]
            tU_ix = path.tU[ix, n]
            uS_ix = path.uS[ix, n]
            mS_ix = path.mS[ix, n]

            mU_ix = max(ℓ_ix - mS_ix, 0.0)
            eU_ix = max(mU_ix - uU_ix - tU_ix, 0.0)

            agg_uU  += w * uU_ix
            agg_tU  += w * tU_ix
            agg_eU  += w * eU_ix
            agg_mU  += w * mU_ix
            agg_uS  += w * uS_ix
            agg_mS  += w * mS_ix
            agg_pop += w * ℓ_ix

            # Unskilled wages: w_U(x,p) = PU·x·p·β + (1-β)·(r+ν)·U(x)
            # Average over employed — approximate using frontier (p=1)
            if eU_ix > 1e-14
                w_U_avg = rp.PU * gp.x[ix] * up.β + (1.0 - up.β) * (cp.r + cp.ν) * path.U[ix, n]
                wage_num_U += w * eU_ix * w_U_avg
                wage_den_U += w * eU_ix
            end

            # Skilled wages (average over employed)
            for jp in 1:NpS
                e_ij = path.eS[ix, jp, n]
                if e_ij > 1e-14
                    pj = sg.p[jp]
                    w_S_ij = rp.PS * gp.x[ix] * pj * sp.β +
                             (1.0 - sp.β) * (cp.r + cp.ν) * path.US[ix, n]
                    wage_num_S += w * e_ij * wpS[jp] * w_S_ij
                    wage_den_S += w * e_ij * wpS[jp]
                end
            end

            for jp in 1:NpS
                agg_eS += w * path.eS[ix, jp, n] * wpS[jp]
            end
        end

        lf_U     = agg_uU + agg_eU                    # unskilled LF (excl. training)
        lf_total = lf_U + agg_mS                       # total LF (excl. training)
        ur_U_path[n]  = lf_U > 1e-14 ? agg_uU / lf_U : 0.0
        ur_S_path[n]  = agg_mS > 1e-14 ? agg_uS / agg_mS : 0.0
        ur_total_path[n] = lf_total > 1e-14 ?
            (agg_uU + agg_uS) / lf_total : 0.0
        skilled_share_path[n]  = lf_total > 1e-14 ? agg_mS / lf_total : 0.0
        training_share_path[n] = agg_pop > 1e-14 ? agg_tU / agg_pop : 0.0
        mean_wage_U_path[n] = wage_den_U > 1e-14 ? wage_num_U / wage_den_U : 0.0
        mean_wage_S_path[n] = wage_den_S > 1e-14 ? wage_num_S / wage_den_S : 0.0
    end

    return TransitionResult(
        scenario, converged, n_iter, final_dist,
        copy(path.tgrid),
        copy(path.θU), copy(path.θS),
        fU_path, fS_path,
        ur_U_path, ur_S_path, ur_total_path,
        skilled_share_path, training_share_path,
        mean_wage_U_path, mean_wage_S_path,
        copy(path.uU), copy(path.tU), copy(path.uS), copy(path.mS),
        copy(path.eS),
        copy(path.τT), copy(path.pstar_U), copy(path.pstar_S), copy(path.poj),
        copy(path.U), copy(path.US),
        copy(gp.x), copy(gp.wx), copy(sg.p), copy(sg.wp),
    )
end

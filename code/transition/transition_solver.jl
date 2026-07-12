############################################################
# transition_solver.jl — RoySearch backward–forward transition
#
# Solves the perfect-foresight MIT-shock path from a pre-switch stationary
# equilibrium z₀ to a post-switch stationary equilibrium z₁ (notes §524).
#
# Algorithm (notes §"Numerical algorithm")
#   Step 0  z₀, z₁ solved externally (transition_simulation.jl).
#   Step 1  Initialise tightness paths and inherit z₀ distributions.
#   Step 2  Backward pass  — time-dependent HJBs at the given θ path.
#   Step 3  Forward pass   — 2D laws of motion for the distributions.
#   Step 4  Free-entry tightness update at each date.
#   Step 5  Iterate 2–4 to convergence on the tightness paths.
#
# What the transition does NOT enforce (unlike the stationary solver)
# ───────────────────────────────────────────────────────────────────
# The stationary solver pins distributions to their KFE fixed point (e.g.
# e_S = 0 below p*_S, m_S = (φ/ν)·t at d = 0).  Along the path those hold
# only in the limit.  Here every mass is inherited from z₀ and evolved by
# its own law of motion:
#   · training      ∂_t t   = τ u_U − (φ+ν) t
#   · untrained U   ∂_t u_U = ν ℓ + λ_U G(p*_U) e_U − (f_U + τ + ν) u_U
#   · skilled U     ∂_t u_S = φ t + (ξ_S + λ_S Γ(p*_S)) e_S^tot
#                             − (ν + (1−d) f_S (1−Γ(p*_S)) + d f_U) u_S
#   · trained mass  ∂_t m_S = φ t − ν m_S − d f_U u_S
# The cross-market drain  d f_U u_S  moves mass from the skilled to the
# unskilled segment (absorbed into e_U), preserving the population; it is
# absent in a stationary equilibrium (steady-state d-flow ≈ 0 by
# self-selection) and fires only while the frontier is moving.
############################################################


# ════════════════════════════════════════════════════════════
#  Step 1: Initialise the path
# ════════════════════════════════════════════════════════════

"""
    _init_path!(path, model_z0)

Seed the tightness paths flat at the z₀ values and inherit the z₀
stationary distributions at every date (a warm start that the forward pass
overwrites for t > 0).  Value paths are seeded with the z₁ caches, which
the backward pass overwrites from the terminal date inward.
"""
function _init_path!(path::TransitionPath, model_z0::Model, model_z1::Model)
    Nt  = length(path.tgrid)
    Nx  = length(model_z0.grids.x)
    NpS = length(model_z0.skl_grids.p)

    cp = model_z0.common
    uc0 = model_z0.unsk_cache;  sc0 = model_z0.skl_cache
    W2  = model_z0.grids.copula.W2
    f_U0 = jobfinding_rate(uc0.θ, model_z0.unsk_par.μ, model_z0.unsk_par.η)

    # z₀ stationary distributions, reconstructed exactly as in
    # compute_equilibrium_objects.
    d0    = clamp.(sc0.d, 0.0, 1.0)
    mS0   = _mS_from_t(uc0.t, d0, cp.φ, cp.ν, f_U0)
    uS0   = similar(mS0)
    @inbounds for j in 1:Nx, i in 1:Nx
        uS0[i, j] = d0[i, j] > 0.5 ? mS0[i, j] : sc0.u_frac[j] * mS0[i, j]
    end
    mcol0 = [sum((1.0 .- d0[:, j]) .* mS0[:, j]) for j in 1:Nx]
    eS0   = [sc0.e_frac[j, jp] * mcol0[j] for j in 1:Nx, jp in 1:NpS]

    # Flat tightness paths at z₀.
    path.θU .= uc0.θ
    path.θS .= sc0.θ

    # Inherit distributions at every date.
    @inbounds for n in 1:Nt
        path.uU[:, :, n] .= uc0.u
        path.tU[:, :, n] .= uc0.t
        path.uS[:, :, n] .= uS0
        path.mS[:, :, n] .= mS0
        path.τT[:, :, n] .= uc0.τT
        path.eS[:, :, n] .= eS0
    end

    # Seed value/policy paths with the z₁ terminal caches (backward pass
    # fills the interior; date Nt stays at z₁).
    uc1 = model_z1.unsk_cache;  sc1 = model_z1.skl_cache
    @inbounds for n in 1:Nt, ix in 1:Nx
        path.Usearch[ix, n]   = uc1.Usearch[ix]
        path.T_val[ix, n]     = uc1.T[ix]
        path.Jfrontier[ix, n] = uc1.Jfrontier[ix]
        path.pstar_U[ix, n]   = uc1.pstar[ix]
        path.US[ix, n]        = sc1.U[ix]
        path.pstar_S[ix, n]   = sc1.pstar[ix]
        path.poj[ix, n]       = sc1.poj[ix]
    end
    @inbounds for n in 1:Nt, jp in 1:NpS, ix in 1:Nx
        path.E0[ix, jp, n] = sc1.E0[ix, jp]
        path.E1[ix, jp, n] = sc1.E1[ix, jp]
        path.J0[ix, jp, n] = sc1.J0[ix, jp]
        path.J1[ix, jp, n] = sc1.J1[ix, jp]
    end
    return path
end


# ════════════════════════════════════════════════════════════
#  Step 2: Backward pass  (value functions & policies)
# ════════════════════════════════════════════════════════════

"""
    _backward_pass!(path, model_z1, tp)

Sweep dates `n = Nt-1 … 1` under the z₁ parameters.  At each date the
tightness `(θ_U(n), θ_S(n))` is taken from the path; with θ fixed, the
stationary inner loops uniquely pin the value functions and policies
(cutoffs, the training frontier τ, and the cross-market policy d).  Warm-
starting from date `n+1` gives the backward structure; the θ path is the
only channel through which the future enters.
"""
function _backward_pass!(path::TransitionPath, model::Model, tp::TransitionParams)
    Nt  = length(path.tgrid)
    Nx  = length(model.grids.x)
    NpS = length(model.skl_grids.p)

    cp = model.common;  up = model.unsk_par;  sp = model.skl_par
    uc = model.unsk_cache;  sc = model.skl_cache

    for n in (Nt - 1):-1:1
        uc.θ = path.θU[n]
        sc.θ = path.θS[n]

        fU = jobfinding_rate(uc.θ, up.μ, up.η)

        # ── Skilled block ──────────────────────────────────────────────
        # Warm-start from n+1, install the current trained mass, then run
        # the inner loop (updates U_S, surfaces, cutoffs, and d).
        @inbounds for ix in 1:Nx
            sc.U[ix]     = path.US[ix, n + 1]
            sc.pstar[ix] = path.pstar_S[ix, n + 1]
            sc.poj[ix]   = path.poj[ix, n + 1]
        end
        @inbounds for jp in 1:NpS, ix in 1:Nx
            sc.E0[ix, jp] = path.E0[ix, jp, n + 1]
            sc.E1[ix, jp] = path.E1[ix, jp, n + 1]
            sc.J0[ix, jp] = path.J0[ix, jp, n + 1]
            sc.J1[ix, jp] = path.J1[ix, jp, n + 1]
        end
        sc.m_S .= @view path.mS[:, :, n]

        # Cross branch U_S^(1)(aU) reads next-date unskilled-side values,
        # mirroring solver.jl step B:  E_U(aU,1) = U^search + β_U S_U(aU,1).
        EU1 = [path.Usearch[ix, n + 1] +
               up.β * path.Jfrontier[ix, n + 1] / max(1.0 - up.β, 1e-14)
               for ix in 1:Nx]

        skilled_inner_loop!(model; fU = fU, EU1 = EU1)

        @inbounds for ix in 1:Nx
            path.US[ix, n]      = sc.U[ix]
            path.pstar_S[ix, n] = sc.pstar[ix]
            path.poj[ix, n]     = sc.poj[ix]
        end
        @inbounds for jp in 1:NpS, ix in 1:Nx
            path.E0[ix, jp, n] = sc.E0[ix, jp]
            path.E1[ix, jp, n] = sc.E1[ix, jp]
            path.J0[ix, jp, n] = sc.J0[ix, jp]
            path.J1[ix, jp, n] = sc.J1[ix, jp]
        end

        # ── Unskilled block ────────────────────────────────────────────
        # Warm-start from n+1; the inner loop settles U^search, T, τ, the
        # frontier value J_U(·,1), and p*_U at the fixed θ_U.
        @inbounds for ix in 1:Nx
            uc.Usearch[ix]   = path.Usearch[ix, n + 1]
            uc.T[ix]         = path.T_val[ix, n + 1]
            uc.Jfrontier[ix] = path.Jfrontier[ix, n + 1]
            uc.pstar[ix]     = path.pstar_U[ix, n + 1]
        end
        # d carried from the skilled solve gates the training margin.
        uc.duS_carry .= sc.d .* (@view path.uS[:, :, n])

        unskilled_inner_loop!(model; US_in = uc.Usearch)

        @inbounds for ix in 1:Nx
            path.Usearch[ix, n]   = uc.Usearch[ix]
            path.T_val[ix, n]     = uc.T[ix]
            path.Jfrontier[ix, n] = uc.Jfrontier[ix]
            path.pstar_U[ix, n]   = uc.pstar[ix]
        end
        path.τT[:, :, n] .= uc.τT
    end
    return nothing
end


# ════════════════════════════════════════════════════════════
#  Step 3: Forward pass  (laws of motion for distributions)
# ════════════════════════════════════════════════════════════

"""
    _forward_pass!(path, model_z1, tp)

March the segment masses forward with explicit-Euler steps of the notes
§529 laws of motion.  Masses live on the (aU,aS) copula grid; the skilled
p-composition `e_S(aS,p)` reads only aS (the p-dynamics are aU-independent),
so it is carried per-aS and scaled by the d=0 trained mass in that column.
"""
function _forward_pass!(path::TransitionPath, model::Model, tp::TransitionParams)
    Nt  = length(path.tgrid)
    Nx  = length(model.grids.x)
    NpS = length(model.skl_grids.p)
    dt  = tp.dt

    cp = model.common;  up = model.unsk_par;  sp = model.skl_par
    gp = model.grids;   sg = model.skl_grids;  pre = model.skl_pre
    W2 = gp.copula.W2

    ν = cp.ν;  φ = cp.φ
    λU = up.λ;  αU = up.α_U
    λS = sp.λ;  ξS = sp.ξ

    for n in 1:(Nt - 1)
        fU = jobfinding_rate(path.θU[n], up.μ, up.η)
        fS = jobfinding_rate(path.θS[n], sp.μ, sp.η)

        # ── Training and untrained-unemployment masses (per (aU,aS)) ────
        @inbounds for j in 1:Nx, i in 1:Nx
            τ_ij  = path.τT[i, j, n]
            uU_ij = path.uU[i, j, n]
            tU_ij = path.tU[i, j, n]
            mS_ij = path.mS[i, j, n]
            pstU  = clamp01(path.pstar_U[i, n])   # p*_U reads aU = row i

            # Untrained-segment mass and its (residual) employment.
            mU_ij = max(W2[i, j] - mS_ij, 0.0)
            eU_ij = max(mU_ij - uU_ij - tU_ij, 0.0)

            # Natural unskilled separation: shock landing below p*_U.
            δU = λU * G_cdf_unskilled(pstU, αU)

            dt_t = τ_ij * uU_ij - (φ + ν) * tU_ij
            du   = ν * W2[i, j] + δU * eU_ij - (fU + τ_ij + ν) * uU_ij

            path.tU[i, j, n + 1] = max(tU_ij + dt * dt_t, 0.0)
            path.uU[i, j, n + 1] = max(uU_ij + dt * du, 0.0)
        end

        # ── Skilled unemployment and trained mass (per (aU,aS)) ─────────
        _forward_skilled_masses!(path, model, n, fU, fS)

        # ── Skilled p-composition e_S(aS,p) (per aS; notes §386) ────────
        _forward_skilled_pdist!(path, model, n, fS)
    end
    return nothing
end

"""
    _forward_skilled_masses!(path, model, n, fU, fS)

Advance `u_S` and `m_S` one step under the branched skilled-unemployment
outflow and the cross-market drain `d f_U u_S` (notes §529).  The policy
`d(aU,aS,n) = 1{U_S^(1)(aU) > U_S^(0)(aS)}` is read from the date-`n`
value paths so the step is self-contained.
"""
function _forward_skilled_masses!(path::TransitionPath, model::Model,
                                  n::Int, fU::Float64, fS::Float64)
    Nx  = length(model.grids.x)
    dt  = model_time_step(model, path)
    cp = model.common;  up = model.unsk_par;  sp = model.skl_par
    sg = model.skl_grids;  pre = model.skl_pre
    ν = cp.ν;  φ = cp.φ;  λS = sp.λ;  ξS = sp.ξ

    # d(aU,aS,n): cross-to-unskilled candidate value vs stay-skilled value.
    # U_S^(1)(aU) = (b_S·exp(A) + f_U E_U(aU,1)) / (r+ν+f_U); U_S^(0)(aS) = US.
    bS = sp.bS * exp(cp.A)
    U1 = [(bS + fU * (path.Usearch[i, n] +
                      up.β * path.Jfrontier[i, n] / max(1.0 - up.β, 1e-14))) /
          (cp.r + cp.ν + fU) for i in 1:Nx]

    @inbounds for j in 1:Nx
        pstS    = clamp01(path.pstar_S[j, n])
        j_ps    = pcut_index(sg.p, pstS)
        Γ_pstar = pre.Γvals[j_ps]
        δS_end  = ξS + λS * Γ_pstar
        US0     = path.US[j, n]
        for i in 1:Nx
            d_ij  = U1[i] > US0 ? 1.0 : 0.0
            uS_ij = path.uS[i, j, n]
            mS_ij = path.mS[i, j, n]
            tU_ij = path.tU[i, j, n]
            eS_ij = max(mS_ij - uS_ij, 0.0)            # employed mass in this cell

            du = φ * tU_ij + δS_end * eS_ij -
                 (ν + (1.0 - d_ij) * fS * (1.0 - Γ_pstar) + d_ij * fU) * uS_ij
            dm = φ * tU_ij - ν * mS_ij - d_ij * fU * uS_ij

            path.uS[i, j, n + 1] = max(uS_ij + dt * du, 0.0)
            path.mS[i, j, n + 1] = max(mS_ij + dt * dm, 0.0)
        end
    end
    return nothing
end

"""
    _forward_skilled_pdist!(path, model, n, fS)

Advance the per-aS skilled employment density `e_S(aS,p)` one step.  Mass
below the current reservation quality receives no hires and no within-band
quality-shock inflow; it decays at the natural rate `(ν+ξ_S+λ_S)`.  Mass
above the cutoff gains hires `f_S γ(p) u_S`, quality-shock inflow from below
`λ_S γ(p) ∫_{p'<p} e_S`, and loses the OJS outflow when `p < p^oj`.
"""
function _forward_skilled_pdist!(path::TransitionPath, model::Model,
                                 n::Int, fS::Float64)
    Nx  = length(model.grids.x)
    NpS = length(model.skl_grids.p)
    dt  = model_time_step(model, path)
    cp = model.common;  sp = model.skl_par
    sg = model.skl_grids;  pre = model.skl_pre
    ν = cp.ν;  λS = sp.λ;  ξS = sp.ξ

    # Column-aggregated skilled-unemployment inflow rate per aS (unit shape):
    # in the reconstruction e_S is a per-aS unit density, so the hire inflow
    # uses the per-aS unemployed fraction.  Approximate that fraction by the
    # column d=0 unemployed-to-mass ratio at date n.
    @inbounds for j in 1:Nx
        pstS = clamp01(path.pstar_S[j, n])
        pojS = clamp01(path.poj[j, n])
        j0   = pcut_index(sg.p, pstS)

        # Per-aS unemployed unit fraction (u_S / m_S over the d=0 column).
        mcol = 0.0;  ucol = 0.0
        for i in 1:Nx
            mcol += path.mS[i, j, n]
            ucol += path.uS[i, j, n]
        end
        uS_frac = mcol > 1e-14 ? clamp(ucol / mcol, 0.0, 1.0) : 0.0

        cum_e = 0.0
        for jp in 1:NpS
            e_old = path.eS[j, jp, n]
            pj    = sg.p[jp]
            if jp < j0
                outflow = (ν + ξS + λS) * e_old
                path.eS[j, jp, n + 1] = max(e_old - dt * outflow, 0.0)
                cum_e += e_old * sg.wp[jp]
                continue
            end
            γj = pre.γvals[jp];  Γj = pre.Γvals[jp]
            inflow_u = fS * γj * uS_frac
            inflow_λ = λS * γj * cum_e
            is_ojs   = (pj < pojS) ? 1.0 : 0.0
            outflow  = (ν + ξS + λS + is_ojs * fS * (1.0 - Γj)) * e_old
            path.eS[j, jp, n + 1] = max(e_old + dt * (inflow_u + inflow_λ - outflow), 0.0)
            cum_e += e_old * sg.wp[jp]
        end
    end
    return nothing
end

# Step size helper (paths are uniform in model time).
model_time_step(::Model, path::TransitionPath) =
    length(path.tgrid) > 1 ? (path.tgrid[2] - path.tgrid[1]) : 0.0


# ════════════════════════════════════════════════════════════
#  Step 4: Free-entry tightness update
# ════════════════════════════════════════════════════════════

"""
    _update_tightness!(path, model_z1, tp)

At each date, install the current distributions into the caches and read
the free-entry tightness off the stationary relations
(`update_theta_unskilled`, `update_theta_skilled`), then damp the update.
The unskilled updater consumes the augmented seeker pool `u_U + d·u_S`; the
skilled updater consumes the seeker-corrected skilled pool — both exactly
as in the stationary solver, so the same free-entry condition governs the
path.
"""
function _update_tightness!(path::TransitionPath, model::Model, tp::TransitionParams)
    Nt  = length(path.tgrid)
    Nx  = length(model.grids.x)
    up  = model.unsk_par;  sp = model.skl_par
    uc  = model.unsk_cache;  sc = model.skl_cache

    for n in 1:Nt
        # Install date-n distributions and firm values into the caches.
        uc.u  .= @view path.uU[:, :, n]
        uc.t  .= @view path.tU[:, :, n]
        uc.Jfrontier .= @view path.Jfrontier[:, n]
        sc.d  .= _d_matrix(path, model, n)
        uc.duS_carry .= sc.d .* (@view path.uS[:, :, n])

        θU_prop = update_theta_unskilled(model)
        path.θU[n] = (1.0 - tp.damp) * path.θU[n] + tp.damp * θU_prop

        # Skilled: install the trained mass, per-aS shapes, and surfaces the
        # free-entry aggregation reads.
        sc.m_S   .= @view path.mS[:, :, n]
        sc.pstar .= @view path.pstar_S[:, n]
        sc.poj   .= @view path.poj[:, n]
        @inbounds for jp in 1:size(sc.E0, 2), ix in 1:Nx
            sc.E0[ix, jp] = path.E0[ix, jp, n]
            sc.E1[ix, jp] = path.E1[ix, jp, n]
            sc.J0[ix, jp] = path.J0[ix, jp, n]
            sc.J1[ix, jp] = path.J1[ix, jp, n]
        end
        θS_prop = update_theta_skilled(model)
        path.θS[n] = (1.0 - tp.damp) * path.θS[n] + tp.damp * θS_prop
    end
    return nothing
end

"""
    _d_matrix(path, model, n) -> Matrix

Reconstruct the date-`n` cross-market policy `d(aU,aS) = 1{U_S^(1)(aU) >
U_S^(0)(aS)}` from the value paths.
"""
function _d_matrix(path::TransitionPath, model::Model, n::Int)
    Nx = length(model.grids.x)
    cp = model.common;  up = model.unsk_par;  sp = model.skl_par
    fU = jobfinding_rate(path.θU[n], up.μ, up.η)
    bS = sp.bS * exp(cp.A)
    U1 = [(bS + fU * (path.Usearch[i, n] +
                      up.β * path.Jfrontier[i, n] / max(1.0 - up.β, 1e-14))) /
          (cp.r + cp.ν + fU) for i in 1:Nx]
    d = zeros(Float64, Nx, Nx)
    @inbounds for j in 1:Nx, i in 1:Nx
        d[i, j] = U1[i] > path.US[j, n] ? 1.0 : 0.0
    end
    return d
end


# ════════════════════════════════════════════════════════════
#  Public entry point
# ════════════════════════════════════════════════════════════

"""
    solve_transition(model_z0, model_z1, tp; scenario) -> TransitionResult

Backward–forward perfect-foresight transition from stationary equilibrium
`model_z0` to `model_z1` under the post-switch (`z₁`) parameters.  Both
models must share the same grids (same `Nx`, `Np_S`, and ability nodes).
"""
function solve_transition(model_z0::Model, model_z1::Model, tp::TransitionParams;
                          scenario::Symbol = :unnamed)
    path = allocate_path(model_z1, tp)
    _init_path!(path, model_z0, model_z1)

    θU_prev = copy(path.θU);  θS_prev = copy(path.θS)
    converged = false;  final_dist = Inf;  it = 0

    for outer_it in 1:tp.maxit
        it = outer_it
        copyto!(θU_prev, path.θU);  copyto!(θS_prev, path.θS)

        _backward_pass!(path, model_z1, tp)
        _forward_pass!(path, model_z1, tp)
        _update_tightness!(path, model_z1, tp)

        dθU = supnorm(path.θU, θU_prev)
        dθS = supnorm(path.θS, θS_prev)
        final_dist = max(dθU, dθS)

        if tp.verbose && (outer_it == 1 || outer_it % 10 == 0)
            @printf("[transition it=%d]  maxΔθ=%.3e  (Δθ_U=%.3e  Δθ_S=%.3e)\n",
                    outer_it, final_dist, dθU, dθS);  flush(stdout)
        end

        if final_dist < tp.tol
            converged = true
            tp.verbose && @printf("[transition]  converged it=%d  d=%.3e\n", outer_it, final_dist)
            break
        end
    end

    return _build_result(path, model_z1, tp, scenario, converged, it, final_dist)
end


# ════════════════════════════════════════════════════════════
#  Build serialisable result
# ════════════════════════════════════════════════════════════

"""
    _build_result(path, model_z1, tp, scenario, converged, n_iter, final_dist)

Collapse the 2D path into aggregate time-series and ability-marginal
density profiles.  Aggregates use the joint weights `W2`; the marginal
profiles (over aU for the unskilled masses, over aS for the skilled masses)
are what the panel/table layer integrates against `wx = wa`.
"""
function _build_result(path::TransitionPath, model_z1::Model, tp::TransitionParams,
                       scenario::Symbol, converged::Bool, n_iter::Int, final_dist::Float64)
    Nt  = length(path.tgrid)
    Nx  = length(model_z1.grids.x)
    NpS = length(model_z1.skl_grids.p)

    cp = model_z1.common;  up = model_z1.unsk_par;  sp = model_z1.skl_par
    gp = model_z1.grids;   sg = model_z1.skl_grids
    waU = gp.wa_U;  waS = gp.wa_S;  W2 = gp.copula.W2
    wpS = sg.wp
    PU = exp(cp.A) * up.PU;  PS = exp(cp.A) * sp.PS

    fU_p  = zeros(Nt);  fS_p  = zeros(Nt)
    urU_p = zeros(Nt);  urS_p = zeros(Nt);  urT_p = zeros(Nt)
    skS_p = zeros(Nt);  trS_p = zeros(Nt)
    wU_p  = zeros(Nt);  wS_p  = zeros(Nt)

    # Marginal density profiles (Nx × Nt).
    uU_prof = zeros(Nx, Nt);  tU_prof = zeros(Nx, Nt)
    uS_prof = zeros(Nx, Nt);  mS_prof = zeros(Nx, Nt)

    for n in 1:Nt
        fU_p[n] = jobfinding_rate(path.θU[n], up.μ, up.η)
        fS_p[n] = jobfinding_rate(path.θS[n], sp.μ, sp.η)

        # Aggregate masses (W2 weights already inside the 2D densities).
        agg_uU = sum(@view path.uU[:, :, n])
        agg_tU = sum(@view path.tU[:, :, n])
        agg_uS = sum(@view path.uS[:, :, n])
        agg_mS = sum(@view path.mS[:, :, n])
        agg_mU = max(sum(W2) - agg_mS, 0.0)
        agg_eU = max(agg_mU - agg_uU - agg_tU, 0.0)
        agg_pop = sum(W2)

        lf_U     = agg_uU + agg_eU
        lf_total = lf_U + agg_mS
        urU_p[n] = lf_U > 1e-14 ? agg_uU / lf_U : 0.0
        urS_p[n] = agg_mS > 1e-14 ? agg_uS / agg_mS : 0.0
        urT_p[n] = lf_total > 1e-14 ? (agg_uU + agg_uS) / lf_total : 0.0
        skS_p[n] = lf_total > 1e-14 ? agg_mS / lf_total : 0.0
        trS_p[n] = agg_pop > 1e-14 ? agg_tU / agg_pop : 0.0

        # Marginal profiles: unskilled masses over aU (rows), skilled over aS (cols).
        @inbounds for i in 1:Nx
            uU_prof[i, n] = sum(@view path.uU[i, :, n])
            tU_prof[i, n] = sum(@view path.tU[i, :, n])
        end
        @inbounds for j in 1:Nx
            uS_prof[j, n] = sum(@view path.uS[:, j, n])
            mS_prof[j, n] = sum(@view path.mS[:, j, n])
        end

        # Mean wages: unskilled at the frontier approximation, skilled over
        # the employed p-density (both linear in own ability).
        wnum_U = 0.0;  wden_U = 0.0
        @inbounds for j in 1:Nx, i in 1:Nx
            mU_ij = max(W2[i, j] - path.mS[i, j, n], 0.0)
            eU_ij = max(mU_ij - path.uU[i, j, n] - path.tU[i, j, n], 0.0)
            eU_ij <= 1e-14 && continue
            w_U = PU * gp.x[i] * up.β + (1.0 - up.β) * (cp.r + cp.ν) * path.Usearch[i, n]
            wnum_U += eU_ij * w_U;  wden_U += eU_ij
        end
        wU_p[n] = wden_U > 1e-14 ? wnum_U / wden_U : 0.0

        wnum_S = 0.0;  wden_S = 0.0
        @inbounds for j in 1:Nx
            mcol0 = sum(@view path.mS[:, j, n])   # column mass by aS
            for jp in 1:NpS
                e_jp = path.eS[j, jp, n] * mcol0
                e_jp <= 1e-14 && continue
                w_S = PS * gp.x[j] * sg.p[jp] * sp.β +
                      (1.0 - sp.β) * (cp.r + cp.ν) * path.US[j, n]
                wnum_S += e_jp * wpS[jp] * w_S;  wden_S += e_jp * wpS[jp]
            end
        end
        wS_p[n] = wden_S > 1e-14 ? wnum_S / wden_S : 0.0
    end

    return TransitionResult(
        scenario, converged, n_iter, final_dist,
        copy(path.tgrid), copy(path.θU), copy(path.θS), fU_p, fS_p,
        urU_p, urS_p, urT_p, skS_p, trS_p, wU_p, wS_p,
        uU_prof, tU_prof, uS_prof, mS_prof,
        copy(waU), copy(gp.x), copy(sg.p), copy(wpS),
    )
end

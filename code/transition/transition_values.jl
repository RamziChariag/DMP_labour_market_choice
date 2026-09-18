############################################################
# transition_values.jl — Step 2 of the transition: the backward pass
#
# Solves the TIME-DEPENDENT HJBs along a given tightness path, from the
# terminal date inward.  Every value object V on the path satisfies
#
#     ϱ_V · V(a,t) = flow_V(a, θ(t)) + ∂_t V(a,t),
#
# with ϱ_V the coefficient the object's own stationary HJB carries on the left.
# Discretising ∂_t V ≈ (V(t+dt) − V(t))/dt gives the implicit one-step recursion
#
#     V(t) = [ flow_V(a, θ(t)) + V(t+dt)/dt ] / (ϱ_V + 1/dt),
#
# so the date-t value carries the date-(t+dt) value with weight 1/dt.  Setting
# 1/dt = 0 recovers the stationary equation exactly, which is both the terminal
# condition and the check that the recursion is the right one.
#
# THE DISCOUNT RATES ARE NOT ALL r + ν.  Read off each block's stationary HJB:
#   U^search(aU)     r + ν              unskilled.jl (see the note below)
#   T(aS)            r + φ + ν          training completes at φ
#   S_U(aU,p)        r + ν + λ_U + ξ_U  the damage shock and the exogenous hazard
#   U_S^(0)(aS)      r + ν              stay-skilled unemployment
#   U_S^(1)(aU)      r + ν + f_U        accepting an unskilled job LEAVES the branch
#   S_S^0(aS,p)      r + ν + λ_S + ξ_S  ϱ_S, the skilled base rate
#   S_S^1(aS,p)      ϱ_S + f_S(1−Γ_o(p)) plus the poaching outflow hazard
# E_S and J_S are Nash transforms of the surpluses and need no recursion of
# their own: J = (1−β)·ω·smooth_pos(S), E = U + β·ω·smooth_pos(S).
#
# WHY THE FLOW TERMS ARE NOT THE ONES IN THE STATIONARY FILES.  The stationary
# surplus equations carry −(r+ν)U in the flow, having substituted the
# stationary unemployment HJB to eliminate b + κβI.  Along a path that
# substitution is wrong by exactly ∂_t U: the honest flow is the unreduced
# −b − κβI, which is what the steps below use.  The unemployment values
# themselves are written with the resolved coefficient — `unskilled.jl` writes
# (r+ν+f_U)U^search = b_U + f_U E_U(aU,1), and cancelling f_U U^search from
# both sides leaves (r+ν)U^search = b_U + f_U β_U S_U(aU,1), the form whose
# flow does not contain the value being solved for.
#
# THE MAXIMISATIONS STAY POINTWISE.  The reservation cutoffs p*_U, p*_S, the
# OJS cutoff p^oj_S, the training frontier τ and the directed-search margin d
# are all read off the DATE-t values, never smoothed across dates.
#
# WITHIN-DATE COUPLING IS A LINEAR SOLVE, NOT A STATIONARY ONE.  At a given
# date the tail integrals read the same date's surpluses, so each step iterates
# its own one-step map.  That map is not the stationary map: every pass carries
# V(t+dt)/dt, so its contraction modulus is (λ + κβ)/(ϱ + 1/dt) rather than
# (λ + κβ)/ϱ, and the 1/dt is doing the work.  Measured on base_fc at dt = 0.5
# months: the unskilled step settles in 9 passes to a 8.8e-08 residual and
# reproduces the terminal stationary values to 5.1e-14, while the SAME step at
# 1/dt = 0 — the stationary map, undamped — is still at a residual of 10 after
# 300 passes.  A coarser time grid therefore costs passes, and a residual
# sitting at `maxit_inner` in `solve_transition`'s verbose line means dt is too
# coarse for these rates, not that the step is wrong.
############################################################


# ════════════════════════════════════════════════════════════
#  Terminal condition
# ════════════════════════════════════════════════════════════

"""
    _seed_terminal_values!(path, model_z1)

Write date `Nt` from the converged `z₁` caches: the boundary condition of the
backward recursion.  The 1D values and cutoffs are copied; the raw surplus
surfaces are rebuilt from them, because the caches store only the ω-weighted,
positive-part transforms (`J`), which cannot be inverted where `ω = 0`.

Rebuilding is a single deterministic pass of the stationary equations at `z₁`'s
own converged tails, so it lands on `z₁`'s fixed point rather than iterating
towards it.
"""
function _seed_terminal_values!(path::TransitionPath, model_z1::Model)
    cp = model_z1.common;  up = model_z1.unsk_par;  sp = model_z1.skl_par
    gp = model_z1.grids;   ug = model_z1.unsk_grids;  sg = model_z1.skl_grids
    pre = model_z1.skl_pre
    uc = model_z1.unsk_cache;  sc = model_z1.skl_cache

    Nt = length(path.tgrid);  Nx = length(gp.x)
    NpU = length(ug.p);  NpS = length(sg.p)
    r = cp.r;  ν = cp.ν

    # ── Unskilled ──────────────────────────────────────────────────
    λU = up.λ;  ξU = up.ξ
    PU = exp(cp.A) * up.PU
    wG = build_unskilled_G_weights(ug.p, ug.wp, up.α_U)
    ϱU = r + ν + λU + ξU

    @inbounds for i in 1:Nx
        path.Usearch[i, Nt]   = uc.Usearch[i]
        path.T_val[i, Nt]     = uc.T[i]
        path.Jfrontier[i, Nt] = uc.Jfrontier[i]
        path.pstar_U[i, Nt]   = uc.pstar[i]

        # I_U(aU) = ∫_{p*}^1 S_U dG from the solver's own closed form, then the
        # raw surplus it implies on the grid.
        Svec  = zeros(Float64, NpU)
        PUeff = PU * gp.x[i]
        I_U   = solve_unskilled_surplus_on_grid!(
                    Svec, ug.p, wG, PUeff, r, ν, λU, uc.Usearch[i],
                    clamp01(uc.pstar[i]), ξU)
        for j in 1:NpU
            path.SU[i, j, Nt] =
                (PUeff * ug.p[j] - (r + ν) * uc.Usearch[i] + λU * I_U) / ϱU
        end
    end

    # ── Skilled ────────────────────────────────────────────────────
    βS = sp.β;  λS = sp.λ;  ξS = sp.ξ
    PS = exp(cp.A) * sp.PS;  bS = sp.bS * exp(cp.A);  σS = sp.σ * exp(cp.A)
    fS = jobfinding_rate(sc.θ, sp.μ, sp.η)
    wΓo = pre.γvals   .* sg.wp
    wΓs = pre.γs_vals .* sg.wp
    denom_nb = max(1.0 - βS, 1e-14)
    no_cont  = zeros(Float64, NpS)

    @inbounds for k in 1:Nx
        path.US[k, Nt]      = sc.U0[k]
        path.US1[k, Nt]     = sc.U1[k]
        path.pstar_S[k, Nt] = sc.pstar[k]
        path.poj[k, Nt]     = sc.poj[k]

        # ω·S^max as the stationary solve measures it, read back off the stored
        # firm values: max(J^0, J^1)/(1−β).
        Smax = [max(sc.J0[k, j], sc.J1[k, j]) / denom_nb for j in 1:NpS]
        tailEo = zeros(Float64, NpS);  tailEs = zeros(Float64, NpS)
        _skilled_tails!(tailEo, tailEs, Smax, wΓo, wΓs)

        # inv_dt = 0 makes the continuation arguments inert, so a zero vector
        # stands in for the date that does not exist beyond the horizon.
        s0 = view(path.S0, k, :, Nt);  s1 = view(path.S1, k, :, Nt)
        _skilled_surfaces!(s0, s1, no_cont, no_cont, tailEo, tailEs, sg, pre,
                           PS * gp.x[k], bS, σS, sc.pstar[k], r, ν, λS, ξS, βS, fS,
                           0.0)
    end

    # τ at the terminal date is z₁'s policy; the pre-switch policy is z₀'s and
    # is written by `_init_path!`.
    path.τT[:, :, Nt] .= uc.τT
    return nothing
end


# ════════════════════════════════════════════════════════════
#  Shared pieces of the skilled surplus equations
# ════════════════════════════════════════════════════════════

"""
    _skilled_tails!(tailEo, tailEs, Smax, wΓo, wΓs)

Accumulate `tailE[j] = ∫_{p_j}^1 S^max dΓ` from the top of the grid, once
against the OFFER cell masses (fresh meetings: hiring, poaching) and once
against the SHOCK cell masses (the λ_S redraw of a live match).  `Smax` is the
ω-weighted, positive-part surplus, the form `skilled_inner_loop!` integrates.
"""
function _skilled_tails!(tailEo::AbstractVector{Float64}, tailEs::AbstractVector{Float64},
                         Smax::AbstractVector{Float64},
                         wΓo::Vector{Float64}, wΓs::Vector{Float64})
    acco = 0.0;  accs = 0.0
    @inbounds for j in length(Smax):-1:1
        acco += Smax[j] * wΓo[j];  tailEo[j] = acco
        accs += Smax[j] * wΓs[j];  tailEs[j] = accs
    end
    return nothing
end

"""
    _skilled_surfaces!(s0, s1, s0next, s1next, tailEo, tailEs, sg, pre,
                       PSeff, bS, σS, pstar, r, ν, λ, ξ, β, f, inv_dt) -> Float64

One pass of the two skilled surplus equations at a single ability, writing the
RAW surpluses in place and returning the sup-norm change.  `inv_dt = 0`
reduces the pass to the stationary equations of `skilled_inner_loop!`.

The no-search branch discounts at `ϱ_S = r+ν+λ_S+ξ_S`; the OJS branch adds the
poaching outflow hazard `f_S(1−Γ_o(p))`, which is `p`-dependent.  Both read the
SHOCK tail for the λ_S redraw and the OFFER tail for the poaching gain.
"""
function _skilled_surfaces!(s0::AbstractVector{Float64}, s1::AbstractVector{Float64},
                            s0next::AbstractVector{Float64}, s1next::AbstractVector{Float64},
                            tailEo::AbstractVector{Float64}, tailEs::AbstractVector{Float64},
                            sg::SkilledGrids, pre::SkilledPrecomp,
                            PSeff::Float64, bS::Float64, σS::Float64, pstar::Float64,
                            r::Float64, ν::Float64, λ::Float64, ξ::Float64,
                            β::Float64, f::Float64, inv_dt::Float64)
    Np = length(sg.p)
    ϱ  = r + ν + λ + ξ
    j0_soft = max(pcut_index(sg.p, clamp01(pstar)) - 1, 1)
    I_o = tailEo[j0_soft]          # fresh-meeting search option, out of unemployment
    I_s = tailEs[j0_soft]          # λ_S redraw option, inside a live match
    flow_out = -bS - f * β * I_o   # unreduced outside-option flow (see file header)

    dmax = 0.0
    @inbounds for j in 1:Np
        common = PSeff * sg.p[j] + flow_out + λ * I_s
        new0 = (common + inv_dt * s0next[j]) / (ϱ + inv_dt)
        new1 = (common - σS + f * β * tailEo[max(j, j0_soft)] + inv_dt * s1next[j]) /
               (ϱ + f * pre.tail_weights[j] + inv_dt)
        dmax = max(dmax, abs(new0 - s0[j]), abs(new1 - s1[j]))
        s0[j] = new0;  s1[j] = new1
    end
    return dmax
end


# ════════════════════════════════════════════════════════════
#  Date steps
# ════════════════════════════════════════════════════════════

"""
    _unskilled_value_step!(path, model, n, wG, iters, resid; inv_dt)

Date-`n` unskilled values at the path's own `θ_U(n)`, carrying date `n+1`.

The surplus, the frontier value `S_U(aU,1)` and the tail integral `I_U` are one
linear system per ability — `I_U` reads the surplus the outside option prices,
and the outside option reads the surplus at `p = 1` — so the pass below
iterates until they agree.  `U^search` then follows in closed form, and the
reservation is the zero crossing of the raw surplus, which is what the
stationary formula `p* = ((r+ν)U^search − λ_U I_U)/(A P_U aU)` solves for when
the continuation term is absent.
"""
function _unskilled_value_step!(path::TransitionPath, model::Model, n::Int,
                                wG::Vector{Float64},
                                iters::Vector{Int}, resid::Vector{Float64};
                                inv_dt::Float64)
    cp = model.common;  up = model.unsk_par;  gp = model.grids;  ug = model.unsk_grids
    Nx = length(gp.x);  Np = length(ug.p)
    r = cp.r;  ν = cp.ν;  λ = up.λ;  ξ = up.ξ;  β = up.β
    PU = exp(cp.A) * up.PU;  bU = up.bU * exp(cp.A)
    f  = jobfinding_rate(path.θU[n], up.μ, up.η)
    tol = model.sim.tol_inner;  maxit = model.sim.maxit_inner

    D = r + ν + λ + ξ + inv_dt

    @threads for i in 1:Nx
        @inbounds begin
            PUeff = PU * gp.x[i]
            raw   = view(path.SU, i, :, n)
            nxt   = view(path.SU, i, :, n + 1)

            I   = 0.0
            S1  = 0.0
            pst = clamp01(path.pstar_U[i, n + 1])
            d   = Inf;  it_done = 0

            for it in 1:maxit
                flow_out = -bU - f * β * S1        # unreduced outside option
                I_new = 0.0
                for j in 1:Np
                    raw[j] = (PUeff * ug.p[j] + flow_out + λ * I + inv_dt * nxt[j]) / D
                    I_new += _soft_weight(ug.p[j], pst, ug.p, j, Np) * raw[j] * wG[j]
                end
                S1_new  = max(_soft_weight(ug.p[Np], pst, ug.p, Np, Np) * raw[Np], 0.0)
                pst_new = clamp01(find_cutoff_from_j0(ug.p, raw, pcut_index(ug.p, pst)))

                d = max(abs(I_new - I), abs(S1_new - S1), abs(pst_new - pst))
                I = I_new;  S1 = S1_new;  pst = pst_new
                it_done = it
                d < tol && break
            end

            path.pstar_U[i, n]   = pst
            path.Jfrontier[i, n] = (1.0 - β) * S1
            path.Usearch[i, n]   = (bU + f * β * S1 + inv_dt * path.Usearch[i, n + 1]) /
                                   (r + ν + inv_dt)
            iters[i] = it_done;  resid[i] = d
        end
    end
    return nothing
end

"""
    _skilled_value_step!(path, model, n, iters, resid; inv_dt)

Date-`n` skilled values at the path's own `θ_S(n)`, carrying date `n+1`.  Same
structure as the unskilled step: the two surplus branches, `U_S^(0)` and the
two cutoffs form one within-date system, resolved by iterating the one-step
map; the cutoffs are then the crossings of the date-`n` raw surpluses, exactly
as `solve_skilled_block!` reads them off its own.
"""
function _skilled_value_step!(path::TransitionPath, model::Model, n::Int,
                              iters::Vector{Int}, resid::Vector{Float64};
                              inv_dt::Float64)
    cp = model.common;  sp = model.skl_par;  gp = model.grids
    sg = model.skl_grids;  pre = model.skl_pre
    Nx = length(gp.x);  Np = length(sg.p)
    r = cp.r;  ν = cp.ν;  λ = sp.λ;  ξ = sp.ξ;  β = sp.β
    PS = exp(cp.A) * sp.PS;  bS = sp.bS * exp(cp.A);  σS = sp.σ * exp(cp.A)
    f  = jobfinding_rate(path.θS[n], sp.μ, sp.η)
    tol = model.sim.tol_inner;  maxit = model.sim.maxit_inner

    wΓo = pre.γvals   .* sg.wp
    wΓs = pre.γs_vals .* sg.wp

    @threads for k in 1:Nx
        @inbounds begin
            PSeff = PS * gp.x[k]
            s0 = view(path.S0, k, :, n);      s1 = view(path.S1, k, :, n)
            s0n = view(path.S0, k, :, n + 1); s1n = view(path.S1, k, :, n + 1)
            copyto!(s0, s0n);  copyto!(s1, s1n)      # warm start at the next date

            tailEo = zeros(Float64, Np);  tailEs = zeros(Float64, Np)
            Smax   = zeros(Float64, Np);  Sdiff  = zeros(Float64, Np)

            U0  = path.US[k, n + 1]
            pst = clamp01(path.pstar_S[k, n + 1])
            poj = clamp01(path.poj[k, n + 1])
            d   = Inf;  it_done = 0

            for it in 1:maxit
                for j in 1:Np
                    Smax[j] = _soft_weight(sg.p[j], pst, sg.p, j, Np) *
                              smooth_pos(max(s0[j], s1[j]))
                end
                _skilled_tails!(tailEo, tailEs, Smax, wΓo, wΓs)

                j0_soft = max(pcut_index(sg.p, pst) - 1, 1)
                U0_new  = (bS + f * β * tailEo[j0_soft] + inv_dt * path.US[k, n + 1]) /
                          (r + ν + inv_dt)
                d = _skilled_surfaces!(s0, s1, s0n, s1n, tailEo, tailEs, sg, pre,
                                       PSeff, bS, σS, pst, r, ν, λ, ξ, β, f, inv_dt)

                # Cutoffs from the RAW surpluses, the vectors solve_skilled_block!
                # scans: p*_S where S^max crosses zero, p^oj_S where the OJS gain does.
                for j in 1:Np
                    Smax[j]  = max(s0[j], s1[j])
                    Sdiff[j] = s1[j] - s0[j]
                end
                pst_new = clamp01(find_cutoff_from_j0(sg.p, Smax, pcut_index(sg.p, pst)))
                poj_new = max(pst_new, clamp01(find_poj_from_diff_grid(sg.p, Sdiff, pst_new)))

                d = max(d, abs(U0_new - U0), abs(pst_new - pst), abs(poj_new - poj))
                U0 = U0_new;  pst = pst_new;  poj = poj_new
                it_done = it
                d < tol && break
            end

            path.US[k, n]      = U0
            path.pstar_S[k, n] = pst
            path.poj[k, n]     = poj
            iters[k] = it_done;  resid[k] = d
        end
    end
    return nothing
end

"""
    _margin_step!(path, model, n, dbuf; inv_dt)

The date-`n` objects that couple the two blocks, in the order their inputs
become available: the training value `T` off the same date's `U_S^(0)`, the
cross branch `U_S^(1)` off the same date's `E_U(aU,1)`, and the training
frontier `τ` off the two.

`τ` is the covered fraction of each ability cell ABOVE the crossing of the net
training gain `−c(aS) + T(aS)` against `U^search(aU)`.  That is the same
monotone-crossing geometry as the cross-market margin with the orientation
reversed, so it goes through `drain_fraction!` — which returns the fraction
BELOW its crossing — and takes the complement.  Reusing it keeps one definition
of "covered fraction of a frontier on the ability grid"; checked branch by
branch against `unskilled_inner_loop!`, including both degenerate cases and the
non-monotone fallback.
"""
function _margin_step!(path::TransitionPath, model::Model, n::Int,
                       dbuf::Matrix{Float64}; inv_dt::Float64)
    cp = model.common;  up = model.unsk_par;  sp = model.skl_par;  gp = model.grids
    Nx = length(gp.x)
    r = cp.r;  ν = cp.ν;  φ = cp.φ
    bT = up.bT * exp(cp.A);  bS = sp.bS * exp(cp.A)
    fU = jobfinding_rate(path.θU[n], up.μ, up.η)
    denom_nb = max(1.0 - up.β, 1e-14)

    @inbounds for j in 1:Nx
        path.T_val[j, n] = (bT + φ * path.US[j, n] + inv_dt * path.T_val[j, n + 1]) /
                           (r + φ + ν + inv_dt)
    end

    # E_U(aU,1) = U^search + β_U S_U(aU,1), recovered from the frontier firm value.
    @inbounds for i in 1:Nx
        EU1 = path.Usearch[i, n] + up.β * path.Jfrontier[i, n] / denom_nb
        path.US1[i, n] = (bS + fU * EU1 + inv_dt * path.US1[i, n + 1]) /
                         (r + ν + fU + inv_dt)
    end

    Utr = [-training_cost(gp.x[j], cp.c) + path.T_val[j, n] for j in 1:Nx]
    drain_fraction!(dbuf, view(path.Usearch, :, n), Utr, gp.x)
    @inbounds for j in 1:Nx, i in 1:Nx
        path.τT[i, j, n] = 1.0 - dbuf[i, j]
    end
    return nothing
end


# ════════════════════════════════════════════════════════════
#  Step 2 driver
# ════════════════════════════════════════════════════════════

"""
    _backward_pass!(path, model_z1, tp) -> (iters, resid)

Sweep dates `n = Nt-1 … 1` under the `z₁` parameters, solving the
time-dependent HJBs at the path's own tightness.  Date `Nt` is the boundary
condition and is not touched.

Returns the largest within-date iteration count and residual over the sweep —
the evidence that each date solved its implicit one-step map rather than a
stationary one.
"""
function _backward_pass!(path::TransitionPath, model::Model, tp::TransitionParams)
    Nt = length(path.tgrid);  Nx = length(model.grids.x)
    inv_dt = 1.0 / tp.dt
    wG = build_unskilled_G_weights(model.unsk_grids.p, model.unsk_grids.wp,
                                   model.unsk_par.α_U)

    iters = zeros(Int, Nx);  resid = zeros(Float64, Nx)
    dbuf  = zeros(Float64, Nx, Nx)
    it_max = 0;  res_max = 0.0

    for n in (Nt - 1):-1:1
        _unskilled_value_step!(path, model, n, wG, iters, resid; inv_dt = inv_dt)
        it_max = max(it_max, maximum(iters));  res_max = max(res_max, maximum(resid))

        _skilled_value_step!(path, model, n, iters, resid; inv_dt = inv_dt)
        it_max = max(it_max, maximum(iters));  res_max = max(res_max, maximum(resid))

        _margin_step!(path, model, n, dbuf; inv_dt = inv_dt)
    end
    return (iters = it_max, resid = res_max)
end
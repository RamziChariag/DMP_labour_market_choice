############################################################
# transition_solver.jl — RoySearch backward–forward transition
#
# Solves the unanticipated-permanent-shock path from a pre-switch stationary
# equilibrium z₀ to a post-switch stationary equilibrium z₁ (notes,
# sec:transition).
#
# Algorithm (notes, "Numerical algorithm")
#   Step 0  z₀, z₁ solved externally (transition_simulation.jl).
#   Step 1  Initialise tightness paths and inherit z₀ distributions.
#   Step 2  Backward pass  — time-dependent HJBs at the given θ path
#                            (transition_values.jl).
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
#   · training      ∂_t t   = τ ν ℓ + (ξ_U + λ_U G(p*_U)) e_U^τ − (φ+ν) t
#   · untrained U   ∂_t u_U = (1−τ) ν ℓ + (ξ_U + λ_U G(p*_U)) e_U^s
#                             − (f_U 1{p*_U<1} + ν) u_U
#   · skilled U     ∂_t u_S = φ t + (ξ_S + λ_S Γ_s(p*_S)) e_S^tot
#                             − (ν + (1−d) f_S (1−Γ_o(p*_S)) + d f_U) u_S
#   · trained mass  ∂_t m_S = φ t − ν m_S − d f_U u_S
#   · quality mix   ∂_t ê_S = f_S γ_o(p) (û + ∫_{p*}^p s* ê) + λ_S γ_s(p) ∫_{p*}^1 ê
#                             − (ν + λ_S + ξ_S + s* f_S Ω_o(p)) ê_S,
#                   with Ω_o(p) the offer mass strictly above p
#
# TRAINING IS A DECISION, NOT A HAZARD.  τ(aU,aS) ∈ [0,1] is the population
# FRACTION of a cell whose skilled ability clears the training frontier, and
# that fraction trains whenever unemployed — so it never appears in u_U and its
# unskilled employment is zero.  Entry into training is therefore a SPLIT of
# each inflow in the proportions τ : 1−τ, not an outflow rate τ·u_U out of
# unskilled unemployment; the two have different fixed points, and only the
# split reproduces `solve_stationary_unskilled!` (u_U = 0 and t = νℓ/(φ+ν) at
# τ = 1, u_U = ℓ(δ+ν)/(f_U+δ+ν) and t = 0 at τ = 0, and the convex combination
# in between).  The separation inflow splits on the frontier's own geometry
# rather than on τ: see `_frontier_sweep!` and `_forward_pass!`.
#
# OFFER vs SHOCK.  A fresh meeting draws Γ_o; a λ_S redraw of a live match
# draws the compressed Γ_s on [0,δ].  Hiring, poaching and the acceptance
# margin read Γ_o; endogenous destruction and the redraw inflow read Γ_s.  The
# two coincide only at δ = 1, and the estimates have δ < 1, so reading one for
# the other is a live numerical error, not a limiting case.
#
# The ξ_U term in the untrained-unemployment law follows the stationary block
# (unskilled.jl, equilibrium.jl), which carries an exogenous unskilled hazard;
# the notes' transition section writes that law without ξ_U.  They agree only
# at ξ_U = 0, which the estimates are not.  Flagged rather than resolved here:
# which is right is a specification question, not a numerical one.
#
# The cross-market drain  d f_U u_S  moves mass from the skilled to the
# unskilled segment (absorbed into e_U), preserving the population; it is
# absent in a stationary equilibrium (steady-state d-flow ≈ 0 by
# self-selection) and fires only while the frontier is moving.
############################################################


# ════════════════════════════════════════════════════════════
#  Step 1: Initialise the path
# ════════════════════════════════════════════════════════════

"""
    _init_path!(path, model_z0, model_z1)

Seed the tightness paths flat at the z₀ values, record the pre-switch state,
and inherit the z₀ stationary distributions at every date (a warm start that
the forward pass overwrites for t > 0).  The terminal date's values come from
`_seed_terminal_values!`; the backward pass fills the interior.
"""
function _init_path!(path::TransitionPath, model_z0::Model, model_z1::Model)
    Nt  = length(path.tgrid)
    Nx  = length(model_z0.grids.x)

    cp = model_z0.common
    uc0 = model_z0.unsk_cache;  sc0 = model_z0.skl_cache
    f_U0 = jobfinding_rate(uc0.θ, model_z0.unsk_par.μ, model_z0.unsk_par.η)

    # z₀ stationary distributions, reconstructed exactly as in
    # compute_equilibrium_objects.
    d0    = clamp.(sc0.d, 0.0, 1.0)
    mS0   = _mS_from_t(uc0.t, d0, cp.φ, cp.ν, f_U0)
    uS0   = similar(mS0)
    @inbounds for j in 1:Nx, i in 1:Nx
        uS0[i, j] = d0[i, j] * mS0[i, j] +
                    (1.0 - d0[i, j]) * sc0.u_frac[j] * mS0[i, j]
    end

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
        path.eS[:, :, n] .= sc0.e_frac
    end

    # Pre-switch state: the forward pass restarts date 1 from it each pass.
    path.τT0 .= uc0.τT
    path.uU0 .= uc0.u
    path.tU0 .= uc0.t

    _seed_terminal_values!(path, model_z1)
    return path
end


# ════════════════════════════════════════════════════════════
#  Step 3: Forward pass  (laws of motion for distributions)
# ════════════════════════════════════════════════════════════

"""
    _forward_pass!(path, model_z1, tp)

March the segment masses forward with explicit-Euler steps of the laws of
motion in the notes' transition section.  Masses live on the (aU,aS) copula
grid; the skilled p-composition `e_S(aS,p)` reads only aS (the p-dynamics are
aU-independent), so it is carried as the per-aS UNIT density of a d = 0 type
and scaled by the non-draining column mass wherever it is aggregated.

The drain fraction `d(·,n)` is built once per date and shared by the two
skilled steps, which must read the same margin as the tightness update.

Date 1 is restarted from the pre-switch stocks before the sweep is applied to
it, so re-running the pass inside the outer loop is idempotent.
"""
function _forward_pass!(path::TransitionPath, model::Model, tp::TransitionParams)
    Nt  = length(path.tgrid)
    Nx  = length(model.grids.x)
    dt  = tp.dt

    cp = model.common;  up = model.unsk_par;  sp = model.skl_par
    W2 = model.grids.copula.W2

    ν = cp.ν;  φ = cp.φ
    λU = up.λ;  αU = up.α_U;  ξU = up.ξ

    path.uU[:, :, 1] .= path.uU0
    path.tU[:, :, 1] .= path.tU0

    for n in 1:(Nt - 1)
        fU   = jobfinding_rate(path.θU[n], up.μ, up.η)
        fS   = jobfinding_rate(path.θS[n], sp.μ, sp.η)
        _frontier_sweep!(path, model, n)
        dmat = _d_matrix(path, model, n)

        # ── Training and untrained-unemployment masses (per (aU,aS)) ────
        @inbounds for j in 1:Nx, i in 1:Nx
            τ_ij  = clamp(path.τT[i, j, n], 0.0, 1.0)
            uU_ij = path.uU[i, j, n]
            tU_ij = path.tU[i, j, n]
            mS_ij = path.mS[i, j, n]
            pstU  = clamp01(path.pstar_U[i, n])   # p*_U reads aU = row i

            # Untrained-segment mass and its (residual) employment, split
            # between the two slices of the cell.  The searching slice's
            # population is (1−τ)ℓ and it holds all of u_U, so it can hold at
            # most (1−τ)ℓ − u_U in employment; anything above that belongs to
            # workers the frontier has enclosed while they held a job, and
            # their separations feed training rather than unskilled search.
            # The excess is identically zero in a stationary state, so this
            # split is invisible to a constant frontier and fires only while
            # the frontier moves.
            mU_ij = max(W2[i, j] - mS_ij, 0.0)
            eU_ij = max(mU_ij - uU_ij - tU_ij, 0.0)
            eU_s  = min(eU_ij, max((1.0 - τ_ij) * W2[i, j] - uU_ij, 0.0))
            eU_τ  = eU_ij - eU_s

            # Unskilled separation: exogenous baseline ξ_U plus the endogenous
            # margin, a λ_U shock landing below p*_U.
            δU = ξU + λU * G_cdf_unskilled(pstU, αU)

            # A fresh unskilled match arrives at p = 1, so acceptance is
            # all-or-nothing in p*_U — the gate solve_stationary_unskilled!
            # also applies.
            fU_hire = (pstU < 1.0 - 1e-10) ? fU : 0.0

            dt_t = τ_ij * ν * W2[i, j] + δU * eU_τ - (φ + ν) * tU_ij
            du   = (1.0 - τ_ij) * ν * W2[i, j] + δU * eU_s - (fU_hire + ν) * uU_ij

            path.tU[i, j, n + 1] = max(tU_ij + dt * dt_t, 0.0)
            path.uU[i, j, n + 1] = max(uU_ij + dt * du, 0.0)
        end

        # ── Skilled unemployment, trained mass, and p-composition ───────
        _forward_skilled_masses!(path, model, n, dmat, fU, fS, dt)
        _forward_skilled_pdist!(path, model, n, dmat, fS, dt)
    end
    _frontier_sweep!(path, model, Nt)
    return nothing
end

"""
    _frontier_sweep!(path, model, n)

Move the unemployed the training frontier has just enclosed out of unskilled
search and into training, in place at date `n`.

The training fraction of a cell rose from `τ(n−1)` to `τ(n)`, and all of the
cell's unemployed sit in the searching slice of population `(1−τ(n−1))ℓ`, so
the reclassified slice carries `(τ(n) − τ(n−1))/(1 − τ(n−1))` of `u_U` — all of
it when a cell goes from `τ = 0` to `τ = 1`.  Moving the whole stock at once is
what makes the frontier a decision rather than a hazard: a cell the boundary
has crossed must not keep finding unskilled jobs while it drains at some rate.

The predecessor policy at date 1 is z₀'s, so the sweep also carries the jump in
the frontier at the switch date.  Mass is conserved cell by cell: what leaves
`u_U` is exactly what enters `t`.
"""
function _frontier_sweep!(path::TransitionPath, model::Model, n::Int)
    Nx = length(model.grids.x)
    @inbounds for j in 1:Nx, i in 1:Nx
        τ_now = clamp(path.τT[i, j, n], 0.0, 1.0)
        τ_pre = clamp(n == 1 ? path.τT0[i, j] : path.τT[i, j, n - 1], 0.0, 1.0)
        gap   = 1.0 - τ_pre
        (τ_now <= τ_pre || gap <= 1e-12) && continue
        swept = min((τ_now - τ_pre) / gap, 1.0) * path.uU[i, j, n]
        path.uU[i, j, n] -= swept
        path.tU[i, j, n] += swept
    end
    return nothing
end

"""
    _forward_skilled_masses!(path, model, n, dmat, fU, fS, dt)

Advance `u_S` and `m_S` one step under the branched skilled-unemployment
outflow and the cross-market drain `d f_U u_S`.

The two skilled-quality distributions enter on opposite sides of the balance
and must not be confused: the destruction hazard reads the SHOCK mass below
reservation (a demoting redraw is what can end a match), while the hiring
outflow reads the OFFER mass above it (a fresh meeting).  They coincide only
at δ = 1; at the estimated δ < 1 they do not.  Both margins come from
`_skilled_margin_masses`, which measures them the way the stationary KFE does.
"""
function _forward_skilled_masses!(path::TransitionPath, model::Model, n::Int,
                                  dmat::Matrix{Float64}, fU::Float64, fS::Float64,
                                  dt::Float64)
    Nx = length(model.grids.x)
    cp = model.common;  sp = model.skl_par
    sg = model.skl_grids;  pre = model.skl_pre
    ν = cp.ν;  φ = cp.φ;  λS = sp.λ;  ξS = sp.ξ

    acc_o, below_s = _skilled_margin_masses(model, view(path.pstar_S, :, n))

    @inbounds for j in 1:Nx
        hire_S = fS * acc_o[j]                         # offer mass the worker accepts
        δS_end = ξS + λS * below_s[j]                  # shock mass that kills the match
        for i in 1:Nx
            d_ij  = dmat[i, j]
            uS_ij = path.uS[i, j, n]
            mS_ij = path.mS[i, j, n]
            tU_ij = path.tU[i, j, n]
            eS_ij = max(mS_ij - uS_ij, 0.0)            # employed mass in this cell

            du = φ * tU_ij + δS_end * eS_ij -
                 (ν + (1.0 - d_ij) * hire_S + d_ij * fU) * uS_ij
            dm = φ * tU_ij - ν * mS_ij - d_ij * fU * uS_ij

            path.uS[i, j, n + 1] = max(uS_ij + dt * du, 0.0)
            path.mS[i, j, n + 1] = max(mS_ij + dt * dm, 0.0)
        end
    end
    return nothing
end

"""
    _forward_skilled_pdist!(path, model, n, dmat, fS, dt)

Advance the per-aS skilled employment density `e_S(aS,p)` one step: the
time-dependent form of the stationary balance the notes write as
`eq:eSbalance`.  Above the reservation quality a cell gains

  * hires out of unemployment,     `f_S γ_o(p) û`,
  * a shock redrawing an existing match onto `p`, `λ_S γ_s(p) ∫_{p*}^1 ê`,
  * a poached worker arriving from below, `f_S γ_o(p) ∫_{p*}^p s* ê`,

and loses `(ν + λ_S + ξ_S + s*(p) f_S Ω_o(p)) ê`, with `Ω_o(p)` the offer mass
STRICTLY ABOVE the cell.  Offer and shock
densities are not interchangeable: hiring and poaching read `γ_o`, the redraw
reads `γ_s`.  The redraw inflow is fed by the WHOLE employed mass, the
poaching inflow only by the mass below `p` that is searching.

`ê` and `û` are the per-aS unit shapes of a d = 0 type — the convention
`sc.e_frac` / `sc.u_frac` use — so both the hire inflow and the integrals read
the non-draining part of the column.

Mass sitting below a reservation quality that has risen since it was hired
receives nothing and decays at `(ν+ξ_S+λ_S)`.  The notes define `e_S` only on
`[p*,1]` and give no law there, so this is a regularisation, not a transcription.
"""
function _forward_skilled_pdist!(path::TransitionPath, model::Model, n::Int,
                                 dmat::Matrix{Float64}, fS::Float64, dt::Float64)
    Nx  = length(model.grids.x)
    NpS = length(model.skl_grids.p)
    cp = model.common;  sp = model.skl_par
    sg = model.skl_grids;  pre = model.skl_pre
    ν = cp.ν;  λS = sp.λ;  ξS = sp.ξ

    ufrac = _nondrain_unemp_fraction(path, dmat, n, Nx)

    @inbounds for j in 1:Nx
        pstS    = clamp01(path.pstar_S[j, n])
        pojS    = clamp01(path.poj[j, n])
        j0      = pcut_index(sg.p, pstS)
        uS_frac = ufrac[j]

        # Total employed unit mass feeding the λ_S redraw (the whole band).
        e_tot = 0.0
        for jp in j0:NpS
            e_tot += path.eS[j, jp, n] * sg.wp[jp]
        end

        # pre.γvals / pre.γs_vals are cell masses per unit wp, not pointwise
        # densities (grids.jl), and path.eS is likewise per unit wp — every
        # consumer weights it by sg.wp.  Reading both that way keeps the two
        # sides of the balance in the same units, so ∫ (hire inflow) dp is
        # exactly f_S·û, as the stationary solve has it.
        cum_seek = 0.0                                  # ∫_{p*}^p s* ê dp'
        for jp in 1:NpS
            e_old = path.eS[j, jp, n]
            if jp < j0
                path.eS[j, jp, n + 1] = max(e_old - dt * (ν + ξS + λS) * e_old, 0.0)
                continue
            end
            pj  = sg.p[jp]
            γoj = pre.γvals[jp];  γsj = pre.γs_vals[jp]
            # Two covered fractions, both as in the stationary solve (skilled.jl).
            # ω is the part of the cell that clears reservation and so can
            # receive mass at all; s* is the part below p^oj that searches on
            # the job.  Both keep their hazard continuous in the cutoff.
            ω_res = _soft_weight(pj, pstS, sg.p, jp, NpS)
            s_ojs = _soft_oj_weight(pj, pojS, sg.p, jp, NpS)

            # A poached worker must land STRICTLY ABOVE her own cell: `cum_seek`
            # accumulates after cell jp is priced, so the inflow to a cell comes
            # from searchers below it and the offsetting outflow must run over
            # destinations above it.  `tail_weights` shifted by one, zero at the
            # top node — the convention solve_stationary_skilled! uses, and the
            # only one under which ∫inflow = ∫outflow and the stationary shape is
            # a fixed point of this pass.
            acc_mass = jp < NpS ? pre.tail_weights[jp + 1] : 0.0

            inflow  = ω_res * (fS * γoj * (uS_frac + cum_seek) + λS * γsj * e_tot)
            outflow = (ν + ξS + λS + s_ojs * fS * acc_mass) * e_old
            path.eS[j, jp, n + 1] = max(e_old + dt * (inflow - outflow), 0.0)

            cum_seek += s_ojs * e_old * sg.wp[jp]
        end
    end
    return nothing
end

"""
    _skilled_margin_masses(model, pstar) -> (accept_offer, shock_below)

The two reservation margins of the skilled block, per aS: the OFFER mass a
worker accepts, `∫ ω dΓ_o`, and the SHOCK mass that lands below reservation
and destroys the match, `1 − ∫ ω dΓ_s`.

WHY NOT `Γvals[pcut_index(p, p*)]`.  The cutoff does not sit on a node, and
`Γvals` is the CDF AT a node: reading it there silently relocates the cutoff
to the nearest node below.  At these estimates that is not a rounding
question — `p*_S = 0` at every ability, so the read lands on node 1, where
`Γ_s = 0.017`, and invents an endogenous separation hazard `λ_S · 0.017` that
the model says is exactly zero.  Measured at base_fc (Nx = Np_S = 120): the
spurious term is 55% of ξ_S and inflates stationary `u_S/m_S` from 0.0259 to
0.0341.

`solve_stationary_skilled!` does not make this read — it integrates the cell
masses `γ·wp` from the soft cutoff, with the same `_soft_weight` coverage the
surplus uses.  Mirroring that here is what makes the stationary equilibrium a
fixed point of the forward pass, which is the property the whole transition
rests on.

Note that `equilibrium.jl`'s moment layer DOES take the node read, so the
shipped `sep_rate_S` carries the spurious term while the distribution it is
computed from does not.  That inconsistency is upstream of the transition and
is left alone here: closing it moves an estimated moment.
"""
function _skilled_margin_masses(model::Model, pstar::AbstractVector{Float64})
    sg = model.skl_grids;  pre = model.skl_pre
    Nx = length(model.grids.x);  Np = length(sg.p)
    acc_o = zeros(Float64, Nx);  acc_s = zeros(Float64, Nx)

    @inbounds for k in 1:Nx
        pst = clamp01(pstar[k])
        j0  = max(pcut_index(sg.p, pst) - 1, 1)
        for j in j0:Np
            ω = _soft_weight(sg.p[j], pst, sg.p, j, Np)
            ω <= 0.0 && continue
            acc_o[k] += ω * pre.γvals[j]   * sg.wp[j]
            acc_s[k] += ω * pre.γs_vals[j] * sg.wp[j]
        end
    end
    return acc_o, 1.0 .- acc_s
end


"""
    _nondrain_unemp_fraction(path, dmat, n, Nx) -> Vector{Float64}

Unit unemployed fraction `û(aS)` of a d = 0 type at date `n`.  The skilled
block's per-aS shapes are defined for a non-draining type, so the column
aggregate must exclude the draining mass — which is unemployed in the OTHER
market and is counted there, in the augmented unskilled seeker pool.
"""
function _nondrain_unemp_fraction(path::TransitionPath, dmat::Matrix{Float64},
                                  n::Int, Nx::Int)
    uf = zeros(Float64, Nx)
    @inbounds for j in 1:Nx
        mcol = 0.0;  ucol = 0.0
        for i in 1:Nx
            w     = 1.0 - dmat[i, j]
            mcol += w * path.mS[i, j, n]
            ucol += w * path.uS[i, j, n]
        end
        uf[j] = mcol > 1e-14 ? clamp(ucol / mcol, 0.0, 1.0) : 0.0
    end
    return uf
end


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

`compute_Jbar_skilled` reads the seeker pool off the per-aS unit shapes
`u_frac`/`e_frac`, so those must be installed from the PATH.  Left at the
terminal model's stationary values they would price every date's vacancy
against the post-switch composition, which is what makes a path a sequence of
steady states.
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

        # Skilled: install the trained mass, the date-n composition, and the
        # surfaces the free-entry aggregation reads.
        sc.m_S   .= @view path.mS[:, :, n]
        sc.pstar .= @view path.pstar_S[:, n]
        sc.poj   .= @view path.poj[:, n]
        sc.u_frac .= _nondrain_unemp_fraction(path, sc.d, n, Nx)
        sc.e_frac .= @view path.eS[:, :, n]
        _install_skilled_firm_values!(model, path, n)
        θS_prop = update_theta_skilled(model)
        path.θS[n] = (1.0 - tp.damp) * path.θS[n] + tp.damp * θS_prop
    end
    return nothing
end

"""
    _install_skilled_firm_values!(model, path, n)

Write the date-`n` skilled firm values `J_S^0`, `J_S^1` into the cache that
`compute_Jbar_skilled` reads, by the Nash split of the path's raw surpluses:
`J = (1−β_S)·ω·smooth_pos(S)`, with `ω` the reservation coverage at the date-`n`
cutoff.  This is the transform `skilled_inner_loop!` applies, so free entry
along the path prices the same object it prices at the two steady states.
"""
function _install_skilled_firm_values!(model::Model, path::TransitionPath, n::Int)
    sg = model.skl_grids;  sc = model.skl_cache
    Nx = length(model.grids.x);  Np = length(sg.p)
    share = 1.0 - model.skl_par.β

    @inbounds for k in 1:Nx
        pst = clamp01(path.pstar_S[k, n])
        for j in 1:Np
            ω = _soft_weight(sg.p[j], pst, sg.p, j, Np)
            sc.J0[k, j] = share * ω * smooth_pos(path.S0[k, j, n])
            sc.J1[k, j] = share * ω * smooth_pos(path.S1[k, j, n])
        end
    end
    return nothing
end

"""
    _d_matrix(path, model, n) -> Matrix

The date-`n` cross-market drain fraction, through the same `drain_fraction!` the
stationary solver uses, so the path and its two steady states read one
definition of the margin.
"""
_d_matrix(path::TransitionPath, model::Model, n::Int) =
    drain_fraction!(zeros(Float64, length(model.grids.x), length(model.grids.x)),
                    view(path.US1, :, n), view(path.US, :, n), model.grids.x)


# ════════════════════════════════════════════════════════════
#  Public entry point
# ════════════════════════════════════════════════════════════

"""
    solve_transition(model_z0, model_z1, tp; scenario) -> TransitionResult

Backward–forward transition from stationary equilibrium `model_z0` to
`model_z1` under the post-switch (`z₁`) parameters, following an
unanticipated permanent parameter change.  Both models must share the same
grids (same `Nx`, `Np_S`, and ability nodes).  The value step is the
time-dependent HJB recursion in `transition_values.jl`.

CONSUMES ITS MODELS.  The tightness update installs each date's distributions
into `model_z1`'s caches, so on return they hold the terminal date's path state,
not the stationary solution.  A caller running a second pair must solve fresh
models rather than reuse these — `_init_path!` reads `model_z0`'s caches for the
initial condition and would inherit the previous path.
"""
function solve_transition(model_z0::Model, model_z1::Model, tp::TransitionParams;
                          scenario::Symbol = :unnamed)
    path = allocate_path(model_z1, tp)
    _init_path!(path, model_z0, model_z1)

    θU_prev = copy(path.θU);  θS_prev = copy(path.θS)
    converged = false;  final_dist = Inf;  it = 0;  bwd = (iters = 0, resid = 0.0)

    for outer_it in 1:tp.maxit
        it = outer_it
        copyto!(θU_prev, path.θU);  copyto!(θS_prev, path.θS)

        bwd = _backward_pass!(path, model_z1, tp)
        _forward_pass!(path, model_z1, tp)
        _update_tightness!(path, model_z1, tp)

        dθU = supnorm(path.θU, θU_prev)
        dθS = supnorm(path.θS, θS_prev)
        final_dist = max(dθU, dθS)

        # The backward diagnostics say how hard the per-date implicit step was:
        # a handful of passes is the signature of the 1/dt-dominated map, and a
        # count at maxit_inner would mean it had stopped contracting.
        if tp.verbose && (outer_it == 1 || outer_it % 10 == 0)
            @printf("[transition it=%d]  maxΔθ=%.3e  (Δθ_U=%.3e  Δθ_S=%.3e)  backward: %d passes, resid %.2e\n",
                    outer_it, final_dist, dθU, dθS, bwd.iters, bwd.resid);  flush(stdout)
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
    ffl_p = zeros(Nt);  drg_p = zeros(Nt);  dms_p = zeros(Nt);  dfl_p = zeros(Nt)

    # Marginal density profiles (Nx × Nt).
    uU_prof = zeros(Nx, Nt);  tU_prof = zeros(Nx, Nt)
    uS_prof = zeros(Nx, Nt);  mS_prof = zeros(Nx, Nt)

    for n in 1:Nt
        fU_p[n] = jobfinding_rate(path.θU[n], up.μ, up.η)
        fS_p[n] = jobfinding_rate(path.θS[n], sp.μ, sp.η)

        # Training-frontier location, then the three cross-market objects:
        # where F_d sits, whether that side is populated, and the flow across it.
        dmat     = _d_matrix(path, model_z1, n)
        ffl_p[n] = _frontier_floor(path, model_z1, n)
        drg_p[n] = sum(dmat .* W2)
        dms_p[n] = sum(dmat .* (@view path.mS[:, :, n]))
        dfl_p[n] = fU_p[n] * sum(dmat .* (@view path.uS[:, :, n]))

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

        # ê_S is the unit shape of a d = 0 type, so it scales by the
        # NON-DRAINING column mass — the draining part holds no skilled job.
        wnum_S = 0.0;  wden_S = 0.0
        @inbounds for j in 1:Nx
            mcol0 = 0.0
            for i in 1:Nx
                mcol0 += (1.0 - dmat[i, j]) * path.mS[i, j, n]
            end
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
        ffl_p, drg_p, dms_p, dfl_p,
    )
end


"""
    _frontier_floor(path, model, n) -> Float64

Lowest point of the training frontier `F_τ` at date `n`: the skilled ability
`aS*` at which the LEAST unskilled-able worker is indifferent between training
and unskilled search, `−c(aS*) + T(aS*) = U^search(x[1])`.  The frontier slopes
up (notes, prop:frontier), so its floor sits at the bottom of the aU grid.

Built by the same crossing-interpolation `unskilled_inner_loop!` uses for
`τ(·)`, so the reported floor and the policy that drives the forward pass are
one object.  Returns the grid ends when the frontier leaves the grid: `x[1]`
when everyone trains, `x[end]` when no one does.
"""
function _frontier_floor(path::TransitionPath, model::Model, n::Int)
    gp = model.grids;  cp = model.common
    Nx = length(gp.x)

    us  = path.Usearch[1, n]
    Utr = [-training_cost(gp.x[j], cp.c) + path.T_val[j, n] for j in 1:Nx]

    Utr[Nx] < us  && return gp.x[Nx]
    Utr[1]  >= us && return gp.x[1]

    j0 = 2
    while j0 < Nx && Utr[j0] < us
        j0 += 1
    end
    dU = Utr[j0] - Utr[j0 - 1]
    return dU > 0.0 ?
           gp.x[j0 - 1] + (us - Utr[j0 - 1]) * (gp.x[j0] - gp.x[j0 - 1]) / dU :
           gp.x[j0]
end

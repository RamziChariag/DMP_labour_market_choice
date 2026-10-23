############################################################
# transition_values.jl — Step 2 of the transition: the backward pass
#
# Solves the time-dependent values along a given tightness path, from the terminal
# date inward.  Each value V carries the discount ϱ_V of its stationary HJB, and the
# implicit one-step recursion is
#
#     V(t) = [ flow_V(t) + V̂(t+dt)/dt ] / (ϱ_V + 1/dt),
#
# where V̂(t+dt) is the value of the state the agent is in at t+dt AFTER that date's
# choices: the envelope, never the agent's own branch (transition_spec.md, P0).  With
# 1/dt = 0 the step is the stationary equation, which is the terminal condition.
#
# The choices re-made at every date, and the envelope each continuation is valued on:
#   unskilled unemployed   search or train            U_U = max{C, −c(aS) + T}
#   in training            none: cannot be abandoned  T continues on T
#   skilled unemployed     which market to search     U_S = max{U_S^(0), U_S^(1)}
#   unskilled match        continue or end            [S_U]⁺
#   skilled match          search on the job, end     max{0, S⁰, S¹}
# T continues on U_S, and both crossing branches continue on U_S.
#
# HELD CHOICES.  A z₁ model that holds its education choices (`Model.held`, a cf1 window)
# re-makes neither of the first and third choices: τ and d are the held arrays at every
# date, and every envelope over training or over the market is the held branch of its node
# (`trains_at`, `drains_at`, `skilled_value_at`, unskilled.jl), as in its stationary solve.
# τ then never moves between dates, so no slice is swept or released.
#
# FLOW VALUES OF UNEMPLOYMENT.  A match's outside option enters its surplus as the flow
# z = (r+ν)U − U̇ (spec V4), which the recursion gives without the stationary
# substitution: b_U + κ_U β_U S_U(·,1) for an unskilled searcher, b_S + κ_S β_S ∫S dΓ for
# a skilled searcher, (r+ν)U_tr − (Û − U_tr)/dt for a type in training, and
# (r+ν+1/dt)U_S − Û_S/dt for a skilled type whose envelope is the crossing branch.
#
# TWO DIMENSIONS WHERE A LATER CHOICE DIFFERS.  A value depends on the whole type
# (aU, aS) only through a choice made at some later date.  A type that searches the
# unskilled market at every later date has the one-dimensional U^search(aU) of the
# stationary solver, and a type that stays skilled at every later date the
# one-dimensional U_S^(0)(aS); for them the recursion is exactly the one-dimensional
# one.  The full per-type surfaces are solved on the other types only (`pureU`,
# `pureS`), which is the reduction R1 of the spec, exact by construction.
#
# A CELL HAS TWO SLICES.  τ is the covered fraction of a cell that trains, the
# stationary solver's convention.  The searching slice holds the cell's unskilled
# unemployed and the matches whose outside option is search; the training slice holds
# the matches of workers the frontier has enclosed, whose outside option is training.
# When τ rises between dates the share (τ'−τ)/(1−τ) of the searching slice joins the
# training slice, and when it falls the share (τ−τ')/τ of the training slice's matches
# returns: each slice's continuation mixes the two in those proportions, which is the
# reallocation the forward pass applies.
#
# STRADDLING CELLS.  Cutoffs are represented, as in the stationary solver, by the
# coverage weight ω of the cell containing them, and the node just below a cutoff (the
# boundary node) carries the surplus's linear extension, which locates the zero.  A
# match's continuation is the positive part of the next date's surplus, zero at and below
# the boundary node, since a match whose surplus is negative ends.  The boundary node's raw
# value is added back only where the same node is the boundary at both dates: there it
# keeps a steady state exact, and where the cutoff has moved past the node the match ends
# and its continuation is 0.
############################################################

# A covered fraction that moves by less than this between dates has not moved.  The
# fractions agree at a steady state to a few ulps, and 1e-12 of a cell carries no mass.
const FRONTIER_MOVE_TOL = 1e-12

# Share of a cell's searching slice that the training frontier encloses between two dates,
# and share of its training slice that the frontier releases.
@inline sweep_share(τ_now::Float64, τ_next::Float64) =
    (τ_next - τ_now > FRONTIER_MOVE_TOL && τ_now < 1.0) ? (τ_next - τ_now) / (1.0 - τ_now) : 0.0
@inline return_share(τ_now::Float64, τ_next::Float64) =
    (τ_now - τ_next > FRONTIER_MOVE_TOL && τ_now > 0.0) ? (τ_now - τ_next) / τ_now : 0.0

# Two-slot ring for the per-type surfaces: date n writes its slot, date n+1's is the other.
@inline _slot(n::Int) = isodd(n) ? 1 : 2


# ════════════════════════════════════════════════════════════
#  Shared pieces
# ════════════════════════════════════════════════════════════

# The boundary nodes of a cutoff, which carry the surplus's linear extension: the last node
# below it, or the top two nodes when it lies above the grid, where the zero is extrapolated
# through both (`_unskilled_cutoff`).
@inline function _boundary(p::Vector{Float64}, pstar::Float64)
    Np = length(p);  ps = clamp01(pstar)
    p[Np] < ps && return (Np - 1, Np)
    j0 = max(pcut_index(p, ps) - 1, 1)
    return (j0, j0)
end

# A continuation with no boundary values, and an edge weighted by a slice share.
const NO_EDGE = ((0, 0), 0.0, 0.0)
@inline _scaled(edge, w::Float64) = (edge[1], w * edge[2], w * edge[3])

"""
    _continuation!(E, S, p, pstar) -> edge

The value a match of each quality carries into the date whose raw surface is `S` and
cutoff `pstar`: the positive part of `S` above the boundary nodes (`_boundary`), zero below
them, since a match whose surplus is negative ends and an ended match is worth nothing.  The
boundary nodes keep their raw values, which only locate the zero and keep a steady state
exact; they are returned as `edge = (nodes, S[lo], S[hi])`, and the date's solve takes them
out again where its solution's boundary nodes differ (`_drop_moved_edges!`): there the
cutoff has moved past the node, the match ends, and its continuation is 0.
"""
function _continuation!(E::AbstractVector{Float64}, S::AbstractVector{Float64},
                        p::Vector{Float64}, pstar::Float64)
    b = _boundary(p, pstar)
    @inbounds for j in eachindex(E)
        E[j] = j > b[2] ? max(S[j], 0.0) : (j >= b[1] ? S[j] : 0.0)
    end
    return (b, S[b[1]], S[b[2]])
end

# Take out of the continuation `E` the boundary values of every edge whose nodes are not
# the boundary nodes of the date's cutoff `pst`, which has moved past them; true if any was.
# A date is solved with the boundary values in place, which leaves a steady state and every
# date whose boundary did not move exactly as they were, and solved again from there when
# this returns true.  `converged = false` drops every edge: the solve found no cutoff that
# keeps its boundary nodes (it cycles between cells), so the cutoff has moved.
function _drop_moved_edges!(E::AbstractVector{Float64}, edges, p::Vector{Float64}, pst::Float64,
                            converged::Bool)
    bn = _boundary(p, pst);  moved = false
    for (b, v_lo, v_hi) in edges
        (b[1] == 0 || (converged && b == bn) || (v_lo == 0.0 && v_hi == 0.0)) && continue
        E[b[1]] -= v_lo
        b[2] != b[1] && (E[b[2]] -= v_hi)
        moved = true
    end
    return moved
end

"""
    _unskilled_cutoff(p, raw) -> p*

Zero of a raw unskilled surface over the support [0, 1], on the line through the two
nodes that bracket it, or through the two end nodes when it lies outside the grid.  At a
steady state the surface is affine and this is `update_pstar_from_surplus!`'s formula,
including a zero above the top node.
"""
function _unskilled_cutoff(p::Vector{Float64}, raw::AbstractVector{Float64})
    Np = length(p)
    if raw[1] >= 0.0
        m = (raw[2] - raw[1]) / (p[2] - p[1])
        return m > 0.0 ? clamp(p[1] - raw[1] / m, 0.0, p[1]) : 0.0
    end
    @inbounds for j in 2:Np
        raw[j] >= 0.0 &&
            return p[j - 1] - raw[j - 1] * (p[j] - p[j - 1]) / (raw[j] - raw[j - 1])
    end
    m = (raw[Np] - raw[Np - 1]) / (p[Np] - p[Np - 1])
    return m > 0.0 ? min(p[Np] - raw[Np] / m, 1.0) : 1.0
end

"""
    _unskilled_surface!(raw, E, E1, PUeff, p, wG, λ, D, inv_dt, tol, maxit, pst;
                        zfix, b, κβ, edges) -> (S1, S1_raw, pst, iters, resid)

One date of an unskilled match surface: `raw(p) = [A P_U aU p − z + λ I + E(p)/dt] / D`,
`I = ∫ ω raw dG`, the entry surplus `S(1)` from `entry_surplus_unskilled` with the
continuation `E1` at p = 1 (and its value before the positive part, `S1_raw`), and the cutoff.  `edges` are the boundary values `E` carries
(`_continuation!`); where the solution's cutoff has moved past them they are taken out of
`E`, which is overwritten, and the date is solved again from there.  The flow value of unemployment is `zfix`
when given, otherwise the searcher's `b + κβ S(1)`, which depends on the surface being solved,
so the pass iterates until the three agree.
"""
function _unskilled_surface!(raw::AbstractVector{Float64}, E::AbstractVector{Float64},
                             E1::Float64, PUeff::Float64, p::Vector{Float64},
                             wG::Vector{Float64}, λ::Float64, D::Float64, inv_dt::Float64,
                             tol::Float64, maxit::Int, pst::Float64;
                             zfix::Float64 = NaN, b::Float64 = 0.0, κβ::Float64 = 0.0,
                             edges = ())
    Np = length(p)
    I = 0.0;  S1 = 0.0;  S1_raw = 0.0;  d = Inf;  it_done = 0
    for pass in 1:2
        for it in 1:maxit
            z = isnan(zfix) ? b + κβ * S1 : zfix
            I_new = 0.0
            @inbounds for j in 1:Np
                raw[j] = (PUeff * p[j] - z + λ * I + inv_dt * E[j]) / D
                I_new += _soft_weight(p[j], pst, p, j, Np) * raw[j] * wG[j]
            end
            S1_new  = entry_surplus_unskilled(PUeff, -z + inv_dt * E1, λ, I, D)
            S1_raw  = (PUeff + (-z + inv_dt * E1) + λ * I) / D
            pst_new = PUeff <= 1e-14 ? 1.0 : clamp01(_unskilled_cutoff(p, raw))
            d = max(abs(I_new - I), abs(S1_new - S1), abs(pst_new - pst))
            I = I_new;  S1 = S1_new;  pst = pst_new;  it_done = it
            d < tol && break
        end
        (pass == 2 || !_drop_moved_edges!(E, edges, p, pst, d < tol)) && break
    end
    return (S1 = S1, S1_raw = S1_raw, pst = pst, iters = it_done, resid = d)
end

"""
    _training_slice_hire_share!(path, model, n, D)

The hired share of every training slice's meetings at date n (`crosser_hire_share`), from
the date's row entry surplus and the slice's own; zero where the cell does not train.
"""
function _training_slice_hire_share!(path::TransitionPath, model::Model, n::Int, D::Float64)
    hp = hire_profile(view(path.SU1, :, n), model, D)
    Nx = length(model.grids.x)
    @inbounds for j in 1:Nx, i in 1:Nx
        path.χUt[i, j, n] = path.τT[i, j, n] > 0.0 ? crosser_hire_share(hp, i, path.SUt1[i, j, n]) : 0.0
    end
    return nothing
end

"""
    _skilled_tails!(tailEo, tailEs, Sdest, wΓo, wΓs)

Accumulate `tailE[j] = ∫_{p_j}^1 S^dest dΓ` from the top of the grid, once against the
OFFER cell masses (fresh meetings: hiring, poaching) and once against the SHOCK cell
masses (the λ_S redraw of a live match).
"""
function _skilled_tails!(tailEo::AbstractVector{Float64}, tailEs::AbstractVector{Float64},
                         Sdest::AbstractVector{Float64},
                         wΓo::Vector{Float64}, wΓs::Vector{Float64})
    acco = 0.0;  accs = 0.0
    @inbounds for j in length(Sdest):-1:1
        acco += Sdest[j] * wΓo[j];  tailEo[j] = acco
        accs += Sdest[j] * wΓs[j];  tailEs[j] = accs
    end
    return nothing
end

"""
    _skilled_branches!(b0, b1, Sdest, score, E, tailEo, tailEs, part, poj, pst, sg, pre,
                       PSeff, z, σS, ϱ, λ, β, κ, inv_dt) -> Float64

One pass of the two skilled branch values at a single type, both continuing on the
envelope `E` of the next date (spec V6, Part 4 step 3):

    S^s = [F^s + E/dt] / (ρ^s + 1/dt),   F⁰ = A P_S aS p − z + λ I_s,
    F¹ = F⁰ − σ + κβ ∫_p S dΓ,           ρ⁰ = ϱ,  ρ¹ = ϱ + κ(1 − Γ(p)).

Writes the branches, the destination value of each cell and the OJS score; returns the
sup-norm change of the branches.  `E` is the continuation (`_continuation!`).

The destination value is the envelope `ω·max(S⁰,S¹)⁺` plus the stationary solver's
smoothing of the cell the OJS cutoff straddles, `s R¹ + (1−s) R⁰ − max(R⁰,R¹)` on the
flow-equivalent values `R^s = F^s/ρ^s`; it is zero wherever the OJS share is 0 or 1 on
the branch that is larger.  The score is the OJS gain against the envelope,
`[F¹ − F⁰ − κ(1−Γ) S] / ρ`, divided by the discount of the branch NOT chosen; at a steady
state both equal the solver's objects exactly (`R^s` is its raw branch value, and the
score is R¹ − R⁰), so the cutoffs nest, while off a steady state the score has the sign
of the spec's OJS choice.
"""
function _skilled_branches!(b0::AbstractVector{Float64}, b1::AbstractVector{Float64},
                            Sdest::AbstractVector{Float64}, score::AbstractVector{Float64},
                            E::AbstractVector{Float64},
                            tailEo::AbstractVector{Float64}, tailEs::AbstractVector{Float64},
                            part::Float64, poj::Float64, pst::Float64,
                            sg::SkilledGrids, pre::SkilledPrecomp,
                            PSeff::Float64, z::Float64, σS::Float64, ϱ::Float64,
                            λ::Float64, β::Float64, κ::Float64, inv_dt::Float64)
    Np = length(sg.p)
    j0_soft = max(pcut_index(sg.p, clamp01(pst)) - 1, 1)
    I_s = tailEs[j0_soft]
    dmax = 0.0
    @inbounds for j in 1:Np
        ρ1 = ϱ + κ * pre.tail_weights[j]
        F0 = PSeff * sg.p[j] - z + λ * I_s
        F1 = F0 - σS + κ * β * tailEo[max(j, j0_soft)]
        new0 = (F0 + inv_dt * E[j]) / (ϱ + inv_dt)
        new1 = (F1 + inv_dt * E[j]) / (ρ1 + inv_dt)
        dmax = max(dmax, abs(new0 - b0[j]), abs(new1 - b1[j]))
        b0[j] = new0;  b1[j] = new1

        R0 = F0 / ϱ;  R1 = F1 / ρ1
        s  = _ojs_share(part, sg.p[j], poj, sg.p, j, Np)
        Sdest[j] = _soft_weight(sg.p[j], pst, sg.p, j, Np) *
                   (smooth_pos(max(new0, new1)) +
                    s * smooth_pos(R1) + (1.0 - s) * smooth_pos(R0) - smooth_pos(max(R0, R1)))
        score[j] = (F1 - F0 - κ * pre.tail_weights[j] * max(new0, new1)) / (new1 >= new0 ? ϱ : ρ1)
    end
    return dmax
end

"""
    _skilled_cutoffs(env, b0, b1, score, sg, pst) -> (pst, poj, M)

The cutoffs the branch values imply, by the rules of `cutoff_proposals!`: p*_S where the
envelope crosses zero, the participation score M and p^oj_S where the OJS score does.
`env` is scratch for the envelope.
"""
function _skilled_cutoffs(env::Vector{Float64}, b0::AbstractVector{Float64},
                          b1::AbstractVector{Float64}, score::AbstractVector{Float64},
                          sg::SkilledGrids, pst::Float64)
    Np = length(sg.p)
    @inbounds for j in 1:Np
        env[j] = max(b0[j], b1[j])
    end
    pst_new = clamp01(find_cutoff_from_j0(sg.p, env, pcut_index(sg.p, pst)))
    jp0 = max(pcut_index(sg.p, pst_new), 1)
    jpk = jp0
    @inbounds for j in (jp0 + 1):Np
        score[j] > score[jpk] && (jpk = j)
    end
    M = score[jpk]
    poj = max(pst_new, OJS_PARTICIPATION[] === :fraction && M <= 0.0 ? sg.p[jpk] :
                       clamp01(find_poj_from_diff_grid(sg.p, score, pst_new)))
    return (pst = pst_new, poj = poj, M = M)
end


# ════════════════════════════════════════════════════════════
#  Terminal condition
# ════════════════════════════════════════════════════════════

"""
    _seed_terminal_values!(path, model_z1)

Write date `Nt` from the converged `z₁` caches: the boundary condition of the backward
recursion.  The one-dimensional values and cutoffs are copied; the raw surfaces are
rebuilt from them, because the caches store only the ω-weighted, positive-part
transforms (`J`), which cannot be inverted where `ω = 0`.  Rebuilding is one pass of the
stationary equations at `z₁`'s own converged tails, so it lands on `z₁`'s fixed point.

The per-type values are the solver's own (`career_choices!`): the crossing branch
U_S^(1), the training value on `max{U_S^(0), U_S^(1)}` (or the held branch), and τ.
"""
function _seed_terminal_values!(path::TransitionPath, model_z1::Model)
    cp = model_z1.common;  up = model_z1.unsk_par;  sp = model_z1.skl_par
    gp = model_z1.grids;   ug = model_z1.unsk_grids;  sg = model_z1.skl_grids
    pre = model_z1.skl_pre
    uc = model_z1.unsk_cache;  sc = model_z1.skl_cache

    Nt = length(path.tgrid);  Nx = length(gp.x)
    NpU = length(ug.p);  NpS = length(sg.p)
    r = cp.r;  ν = cp.ν

    # ── Unskilled, one-dimensional ─────────────────────────────────
    λU = up.λ;  ξU = up.ξ;  βU = up.β
    PU = exp(cp.A) * up.PU;  bU = up.bU * exp(cp.A)
    κU = meeting_rate(uc.θ, up.μ, up.η)
    wG = build_unskilled_G_weights(ug.p, ug.wp, up.α_U)
    ϱU = r + ν + λU + ξU

    @inbounds for i in 1:Nx
        path.Usearch[i, Nt] = uc.Usearch[i]
        path.pstar_U[i, Nt] = uc.pstar[i]
        path.SU1[i, Nt]     = uc.Jfrontier[i] / max(1.0 - βU, 1e-14)
        Svec  = zeros(Float64, NpU)
        PUeff = PU * gp.x[i]
        I_U   = solve_unskilled_surplus_on_grid!(Svec, ug.p, wG, PUeff, r, ν, λU, uc.Usearch[i],
                                                 clamp01(uc.pstar[i]), ξU)
        for j in 1:NpU
            path.SU[i, j, Nt] = (PUeff * ug.p[j] - (r + ν) * uc.Usearch[i] + λU * I_U) / ϱU
        end
    end

    # ── Skilled, one-dimensional ───────────────────────────────────
    βS = sp.β;  λS = sp.λ;  ξS = sp.ξ
    PS = exp(cp.A) * sp.PS;  bS = sp.bS * exp(cp.A);  σS = sp.σ * exp(cp.A)
    κS = meeting_rate(sc.θ, sp.μ, sp.η)
    wΓo = pre.γvals   .* sg.wp
    wΓs = pre.γs_vals .* sg.wp
    denom_nb = max(1.0 - βS, 1e-14)
    no_cont  = zeros(Float64, NpS);  score = zeros(Float64, NpS)

    @inbounds for k in 1:Nx
        path.US[k, Nt]      = sc.U0[k]
        path.pstar_S[k, Nt] = sc.pstar[k]
        path.poj[k, Nt]     = sc.poj[k]
        path.s_part[k, Nt]  = sc.s_part[k]

        # The destination as skilled_inner_loop! values it, read back off the stored firm
        # values: (s J¹ + (1−s) J⁰)/(1−β), s the OJS searching share.
        Sdest = zeros(Float64, NpS)
        for j in 1:NpS
            s_j      = _ojs_share(sc.s_part[k], sg.p[j], clamp01(sc.poj[k]), sg.p, j, NpS)
            Sdest[j] = (s_j * sc.J1[k, j] + (1.0 - s_j) * sc.J0[k, j]) / denom_nb
        end
        tailEo = zeros(Float64, NpS);  tailEs = zeros(Float64, NpS)
        _skilled_tails!(tailEo, tailEs, Sdest, wΓo, wΓs)
        j0_soft = max(pcut_index(sg.p, clamp01(sc.pstar[k])) - 1, 1)
        path.zS[k, Nt] = bS + κS * βS * tailEo[j0_soft]

        _skilled_branches!(view(path.S0, k, :, Nt), view(path.S1, k, :, Nt),
                           zeros(Float64, NpS), score, no_cont, tailEo, tailEs,
                           sc.s_part[k], clamp01(sc.poj[k]), sc.pstar[k], sg, pre,
                           PS * gp.x[k], path.zS[k, Nt], σS, r + ν + λS + ξS, λS, βS, κS, 0.0)
        path.Sd[k, :, Nt] .= Sdest
    end

    # ── Per type ───────────────────────────────────────────────────
    @inbounds for j in 1:Nx, i in 1:Nx
        path.U0[i, j, Nt] = sc.U0[j]
        path.U1[i, j, Nt] = sc.U1[i, j]
        path.pureS[i, j, Nt] = !(model_z1.held === nothing ? path.U1[i, j, Nt] > path.U0[i, j, Nt] :
                                                             model_z1.held.drains[i, j])
        path.T[i, j, Nt]   = uc.T[i, j]
        path.Utr[i, j, Nt] = -training_cost(gp.x[j], cp.c) + path.T[i, j, Nt]
        path.UC[i, j, Nt]  = uc.Usearch[i]
        path.τT[i, j, Nt]  = uc.τT[i, j]
        path.pUs[i, j, Nt]  = uc.pstar[i]
        path.SUs1[i, j, Nt] = path.SU1[i, Nt]
        path.zUs[i, j, Nt]  = bU + κU * βU * path.SU1[i, Nt]
        path.pureU[i, j, Nt] = true
    end
    return nothing
end


# ════════════════════════════════════════════════════════════
#  Date steps
# ════════════════════════════════════════════════════════════

"""
    _training_step!(path, model, n; inv_dt)

Training value at date n, `(r+φ+ν) T = b_T + φ U_S + Ṫ`, with the cross-block term
`U_S = max{U_S^(0), U_S^(1)}` of the next date (spec Part 4, step 1), its held branch where
the model holds its choices, and the training option `−c(aS) + T`.
"""
function _training_step!(path::TransitionPath, model::Model, n::Int; inv_dt::Float64)
    cp = model.common;  gp = model.grids;  held = model.held
    Nx = length(gp.x)
    bT = model.unsk_par.bT * exp(cp.A);  den = cp.r + cp.φ + cp.ν + inv_dt
    @inbounds for j in 1:Nx, i in 1:Nx
        US_next = skilled_value_at(held, i, j, path.U0[i, j, n + 1], path.U1[i, j, n + 1])
        path.T[i, j, n]   = (bT + cp.φ * US_next + inv_dt * path.T[i, j, n + 1]) / den
        path.Utr[i, j, n] = -training_cost(gp.x[j], cp.c) + path.T[i, j, n]
    end
    return nothing
end

"""
    _unskilled_search_step!(path, model, n, wG, iters, resid; inv_dt)

Date-n values of a type that searches the unskilled market at every later date: the
one-dimensional U^search(aU), its surface and cutoff, the objects of the stationary
unskilled block.
"""
function _unskilled_search_step!(path::TransitionPath, model::Model, n::Int,
                                 wG::Vector{Float64}, iters::Vector{Int},
                                 resid::Vector{Float64}; inv_dt::Float64)
    cp = model.common;  up = model.unsk_par;  gp = model.grids;  ug = model.unsk_grids
    Nx = length(gp.x);  Np = length(ug.p)
    r = cp.r;  ν = cp.ν
    PU = exp(cp.A) * up.PU;  bU = up.bU * exp(cp.A)
    κ  = meeting_rate(path.θU[n], up.μ, up.η)
    D  = r + ν + up.λ + up.ξ + inv_dt
    tol = model.sim.tol_inner;  maxit = model.sim.maxit_inner

    @threads for i in 1:Nx
        @inbounds begin
            E  = zeros(Float64, Np)
            eg = _continuation!(E, view(path.SU, i, :, n + 1), ug.p, path.pstar_U[i, n + 1])
            o = _unskilled_surface!(view(path.SU, i, :, n), E, path.SU1[i, n + 1],
                                    PU * gp.x[i], ug.p, wG, up.λ, D, inv_dt, tol, maxit,
                                    clamp01(path.pstar_U[i, n + 1]); b = bU, κβ = κ * up.β,
                                    edges = (eg,))
            path.pstar_U[i, n] = o.pst
            path.SU1[i, n]     = o.S1
            path.Usearch[i, n] = (bU + κ * up.β * o.S1 + inv_dt * path.Usearch[i, n + 1]) /
                                 (r + ν + inv_dt)
            iters[i] = o.iters;  resid[i] = o.resid
        end
    end
    return nothing
end

"""
    _slice_continuations!(Es, Et, path, buf, i, j, n, p) -> (edge_s, edge_t)

The two continuation surfaces of cell (i,j) into date n+1, with their boundary values: the
searching slice's (per-type when it was solved per type at n+1, the one-dimensional surface
otherwise) and the training slice's (zero where the cell had no training slice).
"""
function _slice_continuations!(Es::Vector{Float64}, Et::Vector{Float64},
                               path::TransitionPath, buf, i::Int, j::Int, n::Int,
                               p::Vector{Float64})
    s1 = _slot(n + 1)
    es = buf.hasS[i, j, s1] ?
         _continuation!(Es, view(buf.Ss, i, j, :, s1), p, path.pUs[i, j, n + 1]) :
         _continuation!(Es, view(path.SU, i, :, n + 1), p, path.pstar_U[i, n + 1])
    if buf.hasT[i, j, s1]
        et = _continuation!(Et, view(buf.St, i, j, :, s1), p, path.pUt[i, j, n + 1])
    else
        fill!(Et, 0.0);  et = NO_EDGE
    end
    return es, et
end

"""
    _unskilled_slices_step!(path, model, n, wG, buf; inv_dt)

Date-n per-type unskilled values: the training fraction τ, the searching slice's value C
and surface, and the training slice's surface.

The frontier is located on the value of searching at n and at n+1, C⁰: the
one-dimensional U^search where the searching slice searches at every later date, the
per-type value at no sweep otherwise.  That value is smooth in aS, so the covered
fraction it gives is `training_policy!`'s own and a steady state returns its τ.  Where the
next date's frontier encloses more of the cell, the types between the two frontiers train
at n+1 in any case, and they train already at n when that beats searching for one step
first, U_tr ≥ C¹ with C¹ the per-type value at a full sweep.  Locating the frontier on
the swept value instead makes τ a fixed point of C(τ) whose slope is 1/(1−τ) near a full
cell, and the iteration leaves the steady state.  With τ chosen, C is the searching slice's
value at the share (τ'−τ)/(1−τ) that the frontier sweeps.  A model that holds its choices
takes τ as held.
"""
function _unskilled_slices_step!(path::TransitionPath, model::Model, n::Int,
                                 wG::Vector{Float64}, buf; inv_dt::Float64)
    cp = model.common;  up = model.unsk_par;  gp = model.grids;  ug = model.unsk_grids
    Nx = length(gp.x);  Np = length(ug.p)
    r = cp.r;  ν = cp.ν
    PU = exp(cp.A) * up.PU;  bU = up.bU * exp(cp.A)
    κ  = meeting_rate(path.θU[n], up.μ, up.η);  κβ = κ * up.β
    D  = r + ν + up.λ + up.ξ + inv_dt
    tol = model.sim.tol_inner;  maxit = model.sim.maxit_inner
    sn = _slot(n)

    C    = view(path.UC, :, :, n);   Utr = view(path.Utr, :, :, n)
    τn   = view(path.τT, :, :, n);   τ1  = view(path.τT, :, :, n + 1)

    # The searching slice's value at the sweep σ, its surface written to `raw`.
    function searching_value!(raw, Es, Et, Ê, i, j, σ)
        es, et = _slice_continuations!(Es, Et, path, buf, i, j, n, ug.p)
        @. Ê = (1.0 - σ) * Es + σ * Et
        Ê1 = (1.0 - σ) * path.SUs1[i, j, n + 1] + σ * max(path.SUt1[i, j, n + 1], 0.0)
        o  = _unskilled_surface!(raw, Ê, Ê1, PU * gp.x[i], ug.p, wG, up.λ, D, inv_dt, tol,
                                 maxit, clamp01(path.pUs[i, j, n + 1]); b = bU, κβ = κβ,
                                 edges = (_scaled(es, 1.0 - σ), _scaled(et, σ)))
        Ĉ = (1.0 - σ) * path.UC[i, j, n + 1] + σ * path.Utr[i, j, n + 1]
        return (C = (bU + κβ * o.S1 + inv_dt * Ĉ) / (r + ν + inv_dt), o = o)
    end

    # The frontier on C⁰, then the types between it and the next date's frontier; or the
    # held τ.
    if model.held === nothing
        @threads for i in 1:Nx
            Es = zeros(Float64, Np);  Et = zeros(Float64, Np);  Ê = zeros(Float64, Np)
            raw = zeros(Float64, Np)
            @inbounds for j in 1:Nx
                C[i, j] = path.pureU[i, j, n + 1] ? path.Usearch[i, n] :
                          searching_value!(raw, Es, Et, Ê, i, j, 0.0).C
            end
        end
        training_policy!(τn, Utr, C, gp.x)
        @threads for i in 1:Nx
            Es = zeros(Float64, Np);  Et = zeros(Float64, Np);  Ê = zeros(Float64, Np)
            raw = zeros(Float64, Np)
            @inbounds for j in 1:Nx
                sweep_share(τn[i, j], τ1[i, j]) > 0.0 || continue
                Utr[i, j] >= searching_value!(raw, Es, Et, Ê, i, j, 1.0).C && (τn[i, j] = τ1[i, j])
            end
        end
    else
        copyto!(τn, model.held.τ)
    end

    # The searching slice at the chosen τ.
    @threads for i in 1:Nx
        Es = zeros(Float64, Np);  Et = zeros(Float64, Np);  Ê = zeros(Float64, Np)
        @inbounds for j in 1:Nx
            σ = sweep_share(τn[i, j], τ1[i, j])
            pure = path.pureU[i, j, n + 1] && σ == 0.0
            path.pureU[i, j, n] = pure
            buf.hasS[i, j, sn]  = !pure
            if pure
                C[i, j]            = path.Usearch[i, n]
                path.pUs[i, j, n]  = path.pstar_U[i, n]
                path.SUs1[i, j, n] = path.SU1[i, n]
                path.zUs[i, j, n]  = bU + κβ * path.SU1[i, n]
                continue
            end
            v = searching_value!(view(buf.Ss, i, j, :, sn), Es, Et, Ê, i, j, σ)
            C[i, j]            = v.C
            path.pUs[i, j, n]  = v.o.pst
            path.SUs1[i, j, n] = v.o.S1
            path.zUs[i, j, n]  = bU + κβ * v.o.S1
        end
    end

    # The training slice: its unemployed are in training, its matches end into training,
    # and the share (τ−τ')/τ the frontier releases returns to the searching slice.
    @threads for i in 1:Nx
        Es = zeros(Float64, Np);  Et = zeros(Float64, Np);  Ê = zeros(Float64, Np)
        @inbounds for j in 1:Nx
            buf.hasT[i, j, sn] = τn[i, j] > 0.0
            if !buf.hasT[i, j, sn]
                path.pUt[i, j, n] = 1.0;  path.SUt1[i, j, n] = 0.0;  path.zUt[i, j, n] = 0.0
                continue
            end
            ς  = return_share(τn[i, j], τ1[i, j])
            es, et = _slice_continuations!(Es, Et, path, buf, i, j, n, ug.p)
            @. Ê = (1.0 - ς) * Et + ς * Es
            Ê1 = (1.0 - ς) * max(path.SUt1[i, j, n + 1], 0.0) + ς * path.SUs1[i, j, n + 1]
            Û  = (1.0 - ς) * path.Utr[i, j, n + 1] + ς * path.UC[i, j, n + 1]
            z  = (r + ν) * Utr[i, j] - inv_dt * (Û - Utr[i, j])
            o  = _unskilled_surface!(view(buf.St, i, j, :, sn), Ê, Ê1, PU * gp.x[i], ug.p,
                                     wG, up.λ, D, inv_dt, tol, maxit,
                                     clamp01(path.pUt[i, j, n + 1]); zfix = z,
                                     edges = (_scaled(et, 1.0 - ς), _scaled(es, ς)))
            path.pUt[i, j, n]  = o.pst
            path.SUt1[i, j, n] = o.S1_raw
            path.zUt[i, j, n]  = z
        end
    end
    _training_slice_hire_share!(path, model, n, D)
    return nothing
end

"""
    _skilled_value_step!(path, model, n, iters, resid; inv_dt)

Date-n values of a type that stays skilled at every later date: U_S^(0)(aS), the two
branch values, their cutoffs and the destination values, the objects of the stationary
skilled block.  Both branches continue on the envelope max{S⁰, S¹} of the next date.

The participation weight `part` of a cell is a covered fraction across NEIGHBOURING
types (`ojs_participation!`), so it cannot be formed inside the per-type loop that needs
it: the date is swept with `part` held fixed, the weight is rebuilt from the scores the
sweep produced, and the sweep repeats until the weight settles to `tol_outer_S`, the
tolerance the stationary block applies to it.
"""
function _skilled_value_step!(path::TransitionPath, model::Model, n::Int,
                              iters::Vector{Int}, resid::Vector{Float64};
                              inv_dt::Float64)
    cp = model.common;  sp = model.skl_par;  gp = model.grids
    sg = model.skl_grids;  pre = model.skl_pre
    Nx = length(gp.x);  Np = length(sg.p)
    r = cp.r;  ν = cp.ν;  λ = sp.λ;  β = sp.β
    PS = exp(cp.A) * sp.PS;  bS = sp.bS * exp(cp.A);  σS = sp.σ * exp(cp.A)
    ϱ  = r + ν + λ + sp.ξ
    κ  = meeting_rate(path.θS[n], sp.μ, sp.η)
    tol = model.sim.tol_inner;  maxit = model.sim.maxit_inner

    wΓo = pre.γvals   .* sg.wp
    wΓs = pre.γs_vals .* sg.wp

    part     = view(path.s_part, :, n)
    part_new = ones(Float64, Nx)
    Mscore   = zeros(Float64, Nx)
    copyto!(part, view(path.s_part, :, n + 1))
    fill!(iters, 0)

    for sweep in 1:model.sim.maxit_outer
        src = sweep == 1 ? n + 1 : n          # warm start: the next date, then this one
        @threads for k in 1:Nx
            @inbounds begin
                b0 = view(path.S0, k, :, n);  b1 = view(path.S1, k, :, n)
                sweep == 1 && (copyto!(b0, view(path.S0, k, :, n + 1));
                               copyto!(b1, view(path.S1, k, :, n + 1)))
                env = [max(path.S0[k, j, n + 1], path.S1[k, j, n + 1]) for j in 1:Np]
                E   = zeros(Float64, Np)
                eg  = _continuation!(E, env, sg.p, path.pstar_S[k, n + 1])
                tailEo = zeros(Float64, Np);  tailEs = zeros(Float64, Np)
                Sdest  = view(path.Sd, k, :, n);  score = zeros(Float64, Np);  cbuf = zeros(Float64, Np)
                sweep == 1 && copyto!(Sdest, view(path.Sd, k, :, n + 1))

                U0  = path.US[k, src]
                pst = clamp01(path.pstar_S[k, src])
                poj = clamp01(path.poj[k, src])
                z   = 0.0;  d = Inf;  it_done = 0

                for pass in 1:2
                    for it in 1:maxit
                        _skilled_tails!(tailEo, tailEs, Sdest, wΓo, wΓs)
                        j0_soft = max(pcut_index(sg.p, pst) - 1, 1)
                        z      = bS + κ * β * tailEo[j0_soft]
                        U0_new = (z + inv_dt * path.US[k, n + 1]) / (r + ν + inv_dt)
                        d = _skilled_branches!(b0, b1, Sdest, score, E, tailEo, tailEs,
                                               part[k], poj, pst, sg, pre, PS * gp.x[k], z,
                                               σS, ϱ, λ, β, κ, inv_dt)
                        c = _skilled_cutoffs(cbuf, b0, b1, score, sg, pst)
                        Mscore[k] = c.M
                        d = max(d, abs(U0_new - U0), abs(c.pst - pst), abs(c.poj - poj))
                        U0 = U0_new;  pst = c.pst;  poj = c.poj
                        it_done = it
                        d < tol && break
                    end
                    (pass == 2 || !_drop_moved_edges!(E, (eg,), sg.p, pst, d < tol)) && break
                end

                path.US[k, n]      = U0
                path.zS[k, n]      = z
                path.pstar_S[k, n] = pst
                path.poj[k, n]     = poj
                iters[k] += it_done;  resid[k] = d
            end
        end

        OJS_PARTICIPATION[] === :fraction && ojs_participation!(part_new, Mscore, gp.x)
        Δpart = supnorm(part_new, part)
        copyto!(part, part_new)
        Δpart < model.sim.tol_outer_S && break
    end
    return nothing
end

"""
    _skilled_types_step!(path, model, n, buf; inv_dt)

Date-n skilled branch values per type (spec V3, reading (A), the author's choice): the
skilled unemployed choose the market at every date, and both branches continue on
`U_S = max{U_S^(0), U_S^(1)}` of the next date,

    U_S^(1) = [b_S + κ_X E_U(a,1) + Û_S/dt] / (r+ν+κ_X+1/dt),   κ_X = κ_U χ_X,
    U_S^(0) = [b_S + κ_S β_S ∫S dΓ + Û_S/dt] / (r+ν+1/dt).

A type that stays skilled at every later date and does not cross at n has the
one-dimensional U_S^(0)(aS).  Every other type has a skilled outside option that prices
the crossing, so its own skilled surface is solved, at the flow `z = (r+ν+1/dt)U_S −
Û_S/dt`, with the column's OJS participation and cutoff.  The match a crosser enters and
its hired share χ_X are `crossing_value`'s: the training slice's where training beats
searching at the node, the searching slice's otherwise.  A model that holds its choices
takes the market choice, the training choice at the node and every U_S as held.
"""
function _skilled_types_step!(path::TransitionPath, model::Model, n::Int, buf;
                              inv_dt::Float64)
    cp = model.common;  sp = model.skl_par;  up = model.unsk_par;  gp = model.grids
    sg = model.skl_grids;  pre = model.skl_pre
    Nx = length(gp.x);  Np = length(sg.p)
    r = cp.r;  ν = cp.ν;  λ = sp.λ;  β = sp.β
    PS = exp(cp.A) * sp.PS;  bS = sp.bS * exp(cp.A);  σS = sp.σ * exp(cp.A)
    ϱ  = r + ν + λ + sp.ξ
    κ  = meeting_rate(path.θS[n], sp.μ, sp.η)
    κU = meeting_rate(path.θU[n], up.μ, up.η);  χU = hire_fraction(model)
    tol = model.sim.tol_inner;  maxit = model.sim.maxit_inner
    wΓo = pre.γvals   .* sg.wp
    wΓs = pre.γs_vals .* sg.wp
    sn = _slot(n);  s1 = _slot(n + 1);  held = model.held

    @threads for i in 1:Nx
        env = zeros(Float64, Np);  E = zeros(Float64, Np);  cbuf = zeros(Float64, Np)
        tailEo = zeros(Float64, Np);  tailEs = zeros(Float64, Np)
        Sdest = zeros(Float64, Np);  score = zeros(Float64, Np)
        @inbounds for j in 1:Nx
            Û  = skilled_value_at(held, i, j, path.U0[i, j, n + 1], path.U1[i, j, n + 1])
            # A crosser hired into the unskilled market is an unskilled worker like any
            # other: her match's outside option is the slice she joins, training where the
            # frontier encloses her type.
            train = held === nothing ? path.Utr[i, j, n] > path.UC[i, j, n] : held.trains[i, j]
            κX = κU * (train ? path.χUt[i, j, n] : χU[i])
            EU = train ? path.Utr[i, j, n] + up.β * max(path.SUt1[i, j, n], 0.0) :
                         path.UC[i, j, n] + up.β * path.SUs1[i, j, n]
            U1 = (bS + κX * EU + inv_dt * Û) / (r + ν + κX + inv_dt)
            path.U1[i, j, n] = U1

            pure = path.pureS[i, j, n + 1] &&
                   !(held === nothing ? U1 > path.US[j, n] : held.drains[i, j])
            path.pureS[i, j, n] = pure
            buf.hasK[i, j, sn]  = !pure
            if pure
                path.U0[i, j, n] = path.US[j, n]
                continue
            end

            b0 = view(buf.K0, i, j, :, sn);  b1 = view(buf.K1, i, j, :, sn)
            if buf.hasK[i, j, s1]
                for q in 1:Np
                    env[q] = max(buf.K0[i, j, q, s1], buf.K1[i, j, q, s1])
                end
                eg = _continuation!(E, env, sg.p, buf.pK[i, j, s1])
            else
                for q in 1:Np
                    env[q] = max(path.S0[j, q, n + 1], path.S1[j, q, n + 1])
                end
                eg = _continuation!(E, env, sg.p, path.pstar_S[j, n + 1])
            end
            copyto!(b0, view(path.S0, j, :, n));  copyto!(b1, view(path.S1, j, :, n))
            copyto!(Sdest, view(path.Sd, j, :, n))
            part = path.s_part[j, n];  poj = clamp01(path.poj[j, n])
            pst  = clamp01(path.pstar_S[j, n]);  U0 = path.US[j, n];  d = Inf
            for pass in 1:2
                for it in 1:maxit
                    _skilled_tails!(tailEo, tailEs, Sdest, wΓo, wΓs)
                    US_now = skilled_value_at(held, i, j, U0, U1)
                    z      = (r + ν + inv_dt) * US_now - inv_dt * Û
                    j0_soft = max(pcut_index(sg.p, pst) - 1, 1)
                    U0_new = (bS + κ * β * tailEo[j0_soft] + inv_dt * Û) / (r + ν + inv_dt)
                    d = _skilled_branches!(b0, b1, Sdest, score, E, tailEo, tailEs, part, poj,
                                           pst, sg, pre, PS * gp.x[j], z, σS, ϱ, λ, β, κ,
                                           inv_dt)
                    pst_new = _skilled_cutoffs(cbuf, b0, b1, score, sg, pst).pst
                    d = max(d, abs(U0_new - U0), abs(pst_new - pst))
                    U0 = U0_new;  pst = pst_new
                    d < tol && break
                end
                (pass == 2 || !_drop_moved_edges!(E, (eg,), sg.p, pst, d < tol)) && break
            end
            path.U0[i, j, n] = U0
            buf.pK[i, j, sn] = pst
        end
    end
    return nothing
end

"""
    _terminal_slices!(path, model, buf)

The terminal date's per-type surfaces the recursion carries: the training slice of every
cell with τ > 0, a stationary surface at the outside option (r+ν)U_tr, and the skilled
surface of every type that crosses at the terminal date.
"""
function _terminal_slices!(path::TransitionPath, model::Model, buf)
    Nt = length(path.tgrid)
    cp = model.common;  up = model.unsk_par;  gp = model.grids;  ug = model.unsk_grids
    Nx = length(gp.x);  NpU = length(ug.p)
    D  = cp.r + cp.ν + up.λ + up.ξ
    PU = exp(cp.A) * up.PU
    wG = build_unskilled_G_weights(ug.p, ug.wp, up.α_U)
    tol = model.sim.tol_inner;  maxit = model.sim.maxit_inner
    sN = _slot(Nt)
    fill!(view(buf.hasS, :, :, sN), false)
    @threads for i in 1:Nx
        E = zeros(Float64, NpU)
        @inbounds for j in 1:Nx
            buf.hasT[i, j, sN] = path.τT[i, j, Nt] > 0.0
            if !buf.hasT[i, j, sN]
                path.pUt[i, j, Nt] = 1.0;  path.SUt1[i, j, Nt] = 0.0;  path.zUt[i, j, Nt] = 0.0
                continue
            end
            z = (cp.r + cp.ν) * path.Utr[i, j, Nt]
            o = _unskilled_surface!(view(buf.St, i, j, :, sN), E, 0.0, PU * gp.x[i], ug.p, wG,
                                    up.λ, D, 0.0, tol, maxit, 0.5; zfix = z)
            path.pUt[i, j, Nt] = o.pst;  path.SUt1[i, j, Nt] = o.S1_raw;  path.zUt[i, j, Nt] = z
        end
    end
    _training_slice_hire_share!(path, model, Nt, D)

    # A type crossing at the terminal date: the skilled surface at its own outside option.
    sp = model.skl_par;  sg = model.skl_grids;  pre = model.skl_pre;  NpS = length(sg.p)
    PS = exp(cp.A) * sp.PS;  bS = sp.bS * exp(cp.A);  σS = sp.σ * exp(cp.A)
    ϱ  = cp.r + cp.ν + sp.λ + sp.ξ
    κ  = meeting_rate(model.skl_cache.θ, sp.μ, sp.η)     # z₁'s own tightness
    wΓo = pre.γvals .* sg.wp;  wΓs = pre.γs_vals .* sg.wp
    @threads for i in 1:Nx
        tailEo = zeros(Float64, NpS);  tailEs = zeros(Float64, NpS)
        Sdest = zeros(Float64, NpS);  score = zeros(Float64, NpS);  E = zeros(Float64, NpS)
        cbuf = zeros(Float64, NpS)
        @inbounds for j in 1:Nx
            buf.hasK[i, j, sN] = !path.pureS[i, j, Nt]
            buf.hasK[i, j, sN] || continue
            b0 = view(buf.K0, i, j, :, sN);  b1 = view(buf.K1, i, j, :, sN)
            copyto!(b0, view(path.S0, j, :, Nt));  copyto!(b1, view(path.S1, j, :, Nt))
            copyto!(Sdest, view(path.Sd, j, :, Nt))
            pst = clamp01(path.pstar_S[j, Nt])
            z   = (cp.r + cp.ν) * path.U1[i, j, Nt]
            for it in 1:maxit
                _skilled_tails!(tailEo, tailEs, Sdest, wΓo, wΓs)
                d = _skilled_branches!(b0, b1, Sdest, score, E, tailEo, tailEs,
                                       path.s_part[j, Nt], clamp01(path.poj[j, Nt]), pst, sg,
                                       pre, PS * gp.x[j], z, σS, ϱ, sp.λ, sp.β, κ, 0.0)
                pst_new = _skilled_cutoffs(cbuf, b0, b1, score, sg, pst).pst
                d = max(d, abs(pst_new - pst));  pst = pst_new
                d < tol && break
            end
            buf.pK[i, j, sN] = pst
        end
    end
    return nothing
end


# ════════════════════════════════════════════════════════════
#  Step 2 driver
# ════════════════════════════════════════════════════════════

"""
    _backward_pass!(path, model_z1, tp) -> (iters, resid)

Sweep dates `n = Nt-1 … 1` under the `z₁` parameters, solving the time-dependent values
at the path's own tightness: the training value off the next date, the unskilled block,
then the skilled block, which uses the same date's unskilled values (spec Part 4).
Date `Nt` is the boundary condition.  Returns the largest within-date iteration count and
residual of the one-dimensional blocks.
"""
function _backward_pass!(path::TransitionPath, model::Model, tp::TransitionParams)
    Nt = length(path.tgrid);  Nx = length(model.grids.x)
    NpU = length(model.unsk_grids.p);  NpS = length(model.skl_grids.p)
    wG = build_unskilled_G_weights(model.unsk_grids.p, model.unsk_grids.wp,
                                   model.unsk_par.α_U)
    # The flags are Array{Bool}, not BitArray: threads write neighbouring cells, and a
    # BitArray packs 64 of them into one word.
    buf = (Ss = zeros(Nx, Nx, NpU, 2), St = zeros(Nx, Nx, NpU, 2),
           hasS = fill(false, Nx, Nx, 2), hasT = fill(false, Nx, Nx, 2),
           K0 = zeros(Nx, Nx, NpS, 2), K1 = zeros(Nx, Nx, NpS, 2),
           pK = zeros(Nx, Nx, 2), hasK = fill(false, Nx, Nx, 2))
    _terminal_slices!(path, model, buf)

    iters = zeros(Int, Nx);  resid = zeros(Float64, Nx)
    it_max = 0;  res_max = 0.0
    for n in (Nt - 1):-1:1
        inv_dt = 1.0 / (path.tgrid[n + 1] - path.tgrid[n])
        _training_step!(path, model, n; inv_dt = inv_dt)
        _unskilled_search_step!(path, model, n, wG, iters, resid; inv_dt = inv_dt)
        it_max = max(it_max, maximum(iters));  res_max = max(res_max, maximum(resid))
        _unskilled_slices_step!(path, model, n, wG, buf; inv_dt = inv_dt)
        _skilled_value_step!(path, model, n, iters, resid; inv_dt = inv_dt)
        it_max = max(it_max, maximum(iters));  res_max = max(res_max, maximum(resid))
        _skilled_types_step!(path, model, n, buf; inv_dt = inv_dt)
    end
    return (iters = it_max, resid = res_max)
end

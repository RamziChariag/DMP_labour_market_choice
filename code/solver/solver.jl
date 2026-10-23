############################################################
# solver.jl — Global equilibrium solver (RoySearch)
#
# Public entry points
#   solve_model(common, unsk_par, skl_par, sim; Nx, Np_U, Np_S, tightness)
#       → build grids and caches and solve, return (Model, SolveResult): at the vacancy
#         costs in the parameter blocks (the global loop), or at `tightness = (U, S)`.
#   solve_model!(model)
#       → run the global loop in place on an existing Model (vacancy costs given).
#   solve_at_tightness!(model, θU, θS)
#       → the equilibrium at given tightnesses, in one pass, with the vacancy costs
#         free entry implies installed in the parameter blocks.  The estimation's solve.
#   skilled_equilibria(model)
#       → every attracting skilled equilibrium at the parameters of a converged Model.
#
# Global fixed point on the link variables (U_S^(0), m_S), both Anderson(m=1)
# accelerated.  U_S^(0)(aS) is one-dimensional in the skilled aptitude; the
# trained mass m_S(aU,aS) is two-dimensional on the copula grid.  The
# remaining link objects are derived deterministically each pass:
#   κ_U         = θ_U q_U(θ_U)                     from the unskilled solve
#   E_U(aU,1)   = U^search(aU) + β_U S_U(aU,1)     from the frontier surplus
#   U_S^(1), T, τ, d and the training-slice match   from the career choices
#   m_S(aU,aS)  = φ t(aU,aS) / (ν + k(aU,aS))       k the crossing hire rate
#   d(aU,aS) u_S(aU,aS) = d m_S                     the cross-market seekers
#
# Per global iteration:
#   A. Solve the unskilled block with the carried d·u_S.
#   B. Form κ_U and E_U(·,1).
#   C. Career choices at the installed U_S^(0); masses; install m_S.
#   D. Solve the skilled block.
#   E. Career choices at the new U_S^(0); masses.
#   F. Anderson on the joint [U_S^(0); vec(m_S)] with per-block scaling.
#   G. Write back: sc.U ← new U_S^(0); sc.m_S ← new m_S; uc.duS_carry ← d·m_S.
#   H. Evaluate the equilibrium residual battery; the pass converges when it is
#      inside tolerance on every component (global_residuals below).
############################################################


# ============================================================
# SolveResult
# ============================================================

"""
    SolveResult

Records whether each layer of the solver converged on the final global
iteration.  All three flags true ⇒ `result.ok`.
"""
struct SolveResult
    converged_U      :: Bool
    converged_S      :: Bool
    converged_global :: Bool
    ok               :: Bool
end

SolveResult(cU::Bool, cS::Bool, cG::Bool) = SolveResult(cU, cS, cG, cU && cS && cG)


# ---------------------------------------------------------------------------
# solve_model — allocate grids and caches from parameter blocks, then solve
# ---------------------------------------------------------------------------
"""
    solve_model(common, unsk_par, skl_par, sim; Nx, Np_U, Np_S, tightness = nothing,
                held = nothing) → (Model, SolveResult)

With `tightness = nothing` the vacancy costs `unsk_par.k`, `skl_par.k` are given and the
global loop solves for the tightnesses (`solve_model!`).  With
`tightness = (U = θ_U, S = θ_S)` the tightnesses are given and the vacancy costs are
computed from free entry (`solve_at_tightness!`); the blocks' `k` are then not read.
`held` holds the education choices at those of another equilibrium (`HeldChoices`).
"""
function solve_model(common::CommonParams, unsk_par::UnskilledParams,
                     skl_par::SkilledParams, sim::SimParams;
                     Nx::Int = 200, Np_U::Int = 200, Np_S::Int = 200,
                     tightness::Union{Nothing,NamedTuple} = nothing,
                     held::Union{Nothing,HeldChoices} = nothing)
    grids   = build_common_grids(common, Nx)
    pU, wpU = build_gl_grid(Np_U);  u_grids = UnskilledGrids(p = pU, wp = wpU)
    pS, wpS = build_gl_grid(Np_S);  s_grids = SkilledGrids(p = pS, wp = wpS)
    model   = build_model(common, grids, unsk_par, u_grids, skl_par, s_grids, sim; Nx, Np_S,
                          held = held)
    result  = tightness === nothing ? solve_model!(model) :
                                      solve_at_tightness!(model, tightness.U, tightness.S)
    return model, result
end


# ---------------------------------------------------------------------------
# m_S(aU,aS) = φ t(aU,aS) / (ν + k(aU,aS)), k the rate at which the cell's trained
# are hired into the unskilled market (crossing_rates)
# ---------------------------------------------------------------------------
function _mS_from_t(t::Matrix{Float64}, k::Matrix{Float64}, φ::Float64, ν::Float64)
    mS = similar(t)
    @inbounds for c in eachindex(t)
        denom = ν + k[c]
        mS[c] = denom > 1e-14 ? max(φ * t[c] / denom, 0.0) : 0.0
    end
    return mS
end


# ---------------------------------------------------------------------------
# Equilibrium residuals
#
# One tolerance per component, because the components do not live on a common
# scale: R1 is a displacement in match quality, R2 and R5 are relative gaps, R3 a
# fraction.  These are properties of the conditions rather than choices a run should
# vary, so they are constants here and not SimParams fields.
# ---------------------------------------------------------------------------
const TOL_CUTOFF   = 1.0e-6      # a cutoff, in units of p ∈ [0,1]; the skilled block's own step tolerance
const TOL_RELATIVE = 1.0e-6      # free entry and the education margin, relative
const TOL_POLICY   = 1.0e-6      # the training choice, a fraction in [0,1]

"""
    global_residuals(model) → (R1, R2, R3, R5, Rmax)

The equilibrium conditions THEMSELVES, evaluated at the state the solver currently holds.
Each component is returned as a RATIO TO ITS OWN TOLERANCE, so all of them read on one
scale and `Rmax < 1` is the convergence gate.

  R1  the skilled cutoffs.  The skilled values are re-solved at the installed cutoffs and
      tightness, and the cutoffs those values imply (`cutoff_proposals!`, the map the
      block iterates) are compared with the installed ones, in p.  `p*_S` is read at every
      type — at the grid floor it is implied at the floor exactly when `S ≥ 0` holds there
      — and `p^oj` wherever some of the type searches; at `s_part = 0` it enters nothing.
      A displacement, so it is zero at a fixed point whatever the grid: the surplus read at
      a node, which it replaces, reported the slope times the distance from the cutoff to
      that node and halved with every doubling of the p-grid.
  R2  free entry, both markets, relative in θ.
  R3  the training choice τ, already a fraction in [0,1].
  R5  the education margin.  R1–R3 are all satisfied by the degenerate state where nobody
      trains and the skilled market is empty — free entry closes on the ℓ-weighted
      fallback, there is no mass to misclassify, and the identities hold trivially.  R5 is
      the training value the current U_S = max{U_S^(0), U_S^(1)} implies against the one
      the state holds, U_S^(1) re-evaluated at the installed T (implied_training_value),
      and it is the only component coupling the two markets.

There is no R4.  The population adding-up residual would go there, and it is deliberately
absent: (E5) DEFINES `m_U := ℓ − m_S` and `e_U := m_U − u_U − t`, so every distributional
identity holds by construction, reads exactly zero everywhere, and can detect no
non-convergence at all.  A vacuous component is not free — it is somewhere for an
arithmetic slip to hide.

R3 and R5 call `training_choice!` and the career choices' own `crossing_value`, the
implementations the iteration uses: a residual built from a second copy of a formula tests
the copy.  Where the model holds its choices (`Model.held`) the training choice is not a
condition of the state, and R3 reads zero.

With `θS_fixed` the skilled tightness is imposed rather than solved for (solve_model!), so
skilled free entry is not a condition of that state and R2 reads the unskilled market only.
"""
function global_residuals(model::Model; θS_fixed::Union{Nothing,Float64} = nothing)
    uc = model.unsk_cache;  sc = model.skl_cache;  up = model.unsk_par

    R1 = cutoff_residual(model)

    # `update_theta_*` RETURN the tightness free entry implies at this state, so the gap
    # against the installed tightness is the free-entry residual expressed in θ — the
    # equation, not a step between iterates.
    R2 = abs(update_theta_unskilled(model) - uc.θ) / max(uc.θ, 1e-14)
    θS_fixed === nothing &&
        (R2 = max(R2, abs(update_theta_skilled(model) - sc.θ) / max(sc.θ, 1e-14)))

    τ_imp = training_choice!(similar(uc.τT), model)
    R3 = maximum(abs, τ_imp .- uc.τT)

    κU    = meeting_rate(uc.θ, up.μ, up.η)
    EU1   = uc.Usearch .+ up.β .* (uc.Jfrontier ./ max(1.0 - up.β, 1e-14))
    T_imp = implied_training_value(model, κU, EU1)
    R5 = maximum(abs, T_imp .- uc.T) / max(maximum(abs, uc.T), 1e-14)

    r1 = R1 / TOL_CUTOFF;  r2 = R2 / TOL_RELATIVE
    r3 = R3 / TOL_POLICY;  r5 = R5 / TOL_RELATIVE
    return (R1 = r1, R2 = r2, R3 = r3, R5 = r5, Rmax = max(r1, r2, r3, r5))
end

"""
    implied_cutoffs(model) → (pst, poj, M)

The skilled cutoffs the model implies at the installed state, and the on-the-job search
participation score M(aS) behind p^oj (`cutoff_proposals!`).  The values are re-solved on
a copy of the skilled cache, so the state is read, never moved.
"""
function implied_cutoffs(model::Model)
    uc = model.unsk_cache;  up = model.unsk_par;  sc = model.skl_cache
    m = Model(common = model.common, grids = model.grids,
              unsk_par = up, unsk_grids = model.unsk_grids, unsk_cache = uc,
              skl_par = model.skl_par, skl_grids = model.skl_grids, skl_pre = model.skl_pre,
              skl_cache = deepcopy(sc), sim = model.sim, corner_limit = model.corner_limit,
              held = model.held)
    inner = skilled_inner_loop!(m)

    Nx  = length(sc.pstar)
    pst = zeros(Float64, Nx);  poj = zeros(Float64, Nx);  M = zeros(Float64, Nx)
    cutoff_proposals!(pst, poj, M, m, inner.κ)
    return (pst = pst, poj = poj, M = M)
end

"""
    cutoff_residual(model) → Float64

Largest displacement between the installed skilled cutoffs and the ones the model implies
at the installed state (R1 in `global_residuals`).
"""
function cutoff_residual(model::Model)
    sc = model.skl_cache
    pst, poj, _ = implied_cutoffs(model)
    Nx = length(sc.pstar)
    R  = 0.0
    @inbounds for k in 1:Nx
        R = max(R, abs(pst[k] - sc.pstar[k]))
        sc.s_part[k] > 0.0 && (R = max(R, abs(max(pst[k], poj[k]) - sc.poj[k])))
    end
    return R
end


# ---------------------------------------------------------------------------
# solve_model! — in-place global fixed point
# ---------------------------------------------------------------------------
"""
    solve_model!(model) → SolveResult

Run the global fixed-point loop in place on `model`.

Free entry at a market that holds no one is priced two ways.  While iterating it is priced
over the population: the first passes of an interior solve often sit at the nobody-trains
corner, and that is the rule the iteration is robust under.  An equilibrium AT a corner is
priced at the limit of the market's composition as it empties, the model's own rule and the
one that keeps every moment continuous where the market closes.  So a solve that converges
at a corner continues from that state under the model's rule; an interior solve never
reaches that step and is unaffected by it.

`θS_fixed` holds the skilled tightness at that value and solves every other object of the
model given it — θ_U, the training frontier, values, cutoffs, participation and masses —
with skilled free entry dropped from the convergence battery.  That is one evaluation of the
skilled free-entry map; `skilled_equilibria` is its only caller.
"""
function solve_model!(model::Model; θS_fixed::Union{Nothing,Float64} = nothing)::SolveResult
    model.corner_limit = false
    result = _global_loop!(model; θS_fixed = θS_fixed)
    if result.ok && market_empty(model)
        model.corner_limit = true
        result = _global_loop!(model; θS_fixed = θS_fixed)
    end
    return result
end

"""
    solve_at_tightness!(model, θU, θS) → SolveResult

The equilibrium at the tightnesses (θU, θS), and the vacancy costs free entry implies
there.  Given both tightnesses the stationary state is unique and needs no iteration
between the markets, because the dependence between the blocks runs one way:
  1. the unskilled block at θ_U: values, the hiring margin and the reservation qualities.
     U^search(a_U) does not read U_S (training enters only the choice τ), so these are final;
  2. the skilled block at θ_S: the stay-skilled values, cutoffs and OJS participation;
  3. the career choices at the solved U_S^(0), with κ_U and E_U(a_U,1) from step 1:
     U_S^(1), T, τ, the drain d and the training-slice match (career_choices!), then the
     masses, which are linear given every policy (stationary_masses);
  4. k_U = q_U(θ_U) J̄_U / (π̄_U ũ_U) and k_S = q_S(θ_S) J̄_S / π̄_S from the same
     free-entry objects `update_theta_*` read (`vacancy_cost_unskilled/skilled`), installed
     in `model.unsk_par` / `model.skl_par`.  A market that holds no one is priced at the
     limit of its composition (`solve_model!`).
With the costs installed the state is an equilibrium of the model at those costs in every
condition, so the residual battery applies unchanged and is the convergence gate: free
entry holds to rounding by construction, and R1, R3, R5 check the rest.
"""
function solve_at_tightness!(model::Model, θU::Float64, θS::Float64)::SolveResult
    up = model.unsk_par;  uc = model.unsk_cache;  sc = model.skl_cache
    model.corner_limit = false

    res_U = solve_unskilled_block!(model; US_in = skilled_unemployment_value(model), θU_fixed = θU)
    res_U.rejected && return SolveResult(false, false, false)
    (any(!isfinite, uc.Usearch) || any(!isfinite, uc.pstar)) && return SolveResult(false, false, false)

    κU  = meeting_rate(θU, up.μ, up.η)
    EU1 = uc.Usearch .+ up.β .* (uc.Jfrontier ./ max(1.0 - up.β, 1e-14))
    res_S = solve_skilled_block!(model; θS_fixed = θS)
    res_S.rejected && return SolveResult(false, false, false)
    (any(!isfinite, sc.U) || any(!isfinite, sc.pstar) || any(!isfinite, sc.u_frac)) &&
        return SolveResult(false, false, false)

    career_choices!(model, κU, EU1)
    any(!isfinite, sc.U1) && return SolveResult(false, false, false)
    install_stationary_masses!(model, κU)

    model.corner_limit = market_empty(model)
    model.unsk_par = _with_k(up, vacancy_cost_unskilled(model))
    model.skl_par  = _with_k(model.skl_par, vacancy_cost_skilled(model))

    R = global_residuals(model)
    return SolveResult(res_U.converged, res_S.converged, R.Rmax < 1.0)
end

# A parameter block with its vacancy cost replaced.
_with_k(p::T, k::Float64) where {T <: Union{UnskilledParams,SkilledParams}} =
    T(; (f => (f === :k ? k : getfield(p, f)) for f in fieldnames(T))...)

"""
    market_empty(model) → Bool

Whether either market holds no one at the current state: no non-draining skilled mass, or
no one seeking in the unskilled market.  The same two tests select the limit composition
in compute_Jbar_skilled and update_theta_unskilled.
"""
function market_empty(model::Model)
    sc = model.skl_cache;  uc = model.unsk_cache
    return sum(nondrain_column_mass(sc.d, sc.m_S)) <= EMPTY_MASS ||
           sum(uc.u) + sum(uc.duS_carry) <= EMPTY_MASS
end

function _global_loop!(model::Model; θS_fixed::Union{Nothing,Float64} = nothing)::SolveResult
    cp = model.common;  sc = model.skl_cache;  uc = model.unsk_cache
    up = model.unsk_par;  sim = model.sim
    φ  = cp.φ;  ν = cp.ν
    Nx = length(model.grids.x);  βU = up.β

    aa = Anderson1(Nx + Nx * Nx)

    # First-pass m_S: nobody crosses, so (φ/ν) t.
    sc.m_S .= (φ / ν) .* uc.t
    mS_cur = copy(sc.m_S)

    s_U = 1.0;  s_M = 1.0
    last_conv_U = false;  last_conv_S = false;  global_converged = false
    streak = 0
    gB = sim.global_B;  gK = sim.global_K
    global_resid = Vector{Float64}(undef, sim.maxit_global)

    for it in 1:sim.maxit_global
        US_old = copy(sc.U)
        mS_old = copy(mS_cur)

        # A. Unskilled block with carried d·u_S.
        res_U = solve_unskilled_block!(model; US_in = skilled_unemployment_value(model))
        res_U.rejected && return SolveResult(false, false, false)
        last_conv_U = res_U.converged
        if !isfinite(uc.θ) || any(!isfinite, uc.t) || any(!isfinite, uc.Usearch) || any(!isfinite, uc.pstar)
            sim.verbose >= 1 && @printf("[global]  NaN/Inf in unskilled block at it=%d — aborting\n", it)
            return SolveResult(false, false, false)
        end

        # B. Unskilled-side outputs consumed by the skilled block.
        κU  = meeting_rate(uc.θ, up.μ, up.η)
        SU1 = uc.Jfrontier ./ max(1.0 - βU, 1e-14)
        EU1 = uc.Usearch .+ βU .* SU1

        # C. Career choices at the installed U_S^(0); install m_S for the skilled solve.
        career_choices!(model, κU, EU1)
        sc.m_S = stationary_masses(model, κU).mS

        # D. Skilled block.
        res_S = solve_skilled_block!(model; θS_fixed = θS_fixed)
        res_S.rejected && return SolveResult(false, false, false)
        last_conv_S = res_S.converged
        if !isfinite(sc.θ) || any(!isfinite, sc.U) || any(!isfinite, sc.pstar) ||
           any(!isfinite, sc.u_frac)
            sim.verbose >= 1 && @printf("[global]  NaN/Inf in skilled block at it=%d — aborting\n", it)
            return SolveResult(false, false, false)
        end

        US_raw = copy(sc.U)

        # E. Career choices and masses at the new U_S^(0).
        career_choices!(model, κU, EU1)
        ms     = install_unskilled_masses!(model, κU)
        mS_raw = ms.mS

        if it == 1
            s_U = max(maximum(abs, US_raw), 1.0)
            s_M = max(maximum(abs, mS_raw), 1.0)
        end

        # F. Joint Anderson on [U_S; vec(m_S)] with per-block scaling.
        x_vec = vcat(US_old ./ s_U, vec(mS_old) ./ s_M)
        f_vec = vcat(US_raw ./ s_U, vec(mS_raw) ./ s_M)
        x_new = sim.use_anderson ? anderson1_update!(aa, x_vec, f_vec) : f_vec

        US_new = x_new[1:Nx] .* s_U
        mS_new = reshape(max.(x_new[Nx+1:end] .* s_M, 0.0), Nx, Nx)

        copyto!(sc.U, US_new)
        sc.m_S = mS_new
        mS_cur = mS_new

        # G. Refresh the cross-market contribution for the next pass.  Only the
        #    draining fraction seeks in the U-market, and all of its mass is
        #    unemployed there, so the carry is d·m_S.  Thresholding d at 0.5 to
        #    pick a branch would discard the fractional boundary node and put the
        #    step function back, whatever the producer does upstream.
        @inbounds for j in 1:Nx, i in 1:Nx
            dij = clamp(sc.d[i, j], 0.0, 1.0)
            uc.duS_carry[i, j] = max(dij * mS_new[i, j], 0.0)
        end

        # Convergence on the equilibrium RESIDUALS rather than on the step between
        # iterates.  A step test is equivalent to a residual test only for a contraction,
        # and this map is not one; it also has no natural scale.  The superseded test
        # compared an ABSOLUTE sup-norm on a U_S of order 1e+05 against the same number as
        # a sup-norm on m_S, so the m_S half never bound and the U_S half demanded nine
        # significant digits of pass-to-pass agreement.  Measured at the base_fc estimate:
        # θ_U, θ_S and m_S are pinned from pass 4 (Δm_S ≈ 1e-13) while ΔU_S oscillates
        # between 1e-05 and 2e-04 against tol_global = 1e-4, so the streak never completes
        # and the stall rule stopped the solve as non-converged — while the battery reads
        # 0.18 of tolerance at that same state.
        R = global_residuals(model; θS_fixed = θS_fixed)
        d = R.Rmax
        global_resid[it] = d

        if sim.verbose >= 1 && (it == 1 || it % sim.verbose_stride == 0)
            @printf("[global it=%d]  Rmax=%.3e  (R1=%.2e  R2=%.2e  R3=%.2e  R5=%.2e)  θ_U=%.4f  θ_S=%.4f\n",
                    it, d, R.R1, R.R2, R.R3, R.R5, uc.θ, sc.θ)
        end

        # Each component is already a ratio to its own tolerance, so the gate is 1.
        if d < 1.0
            streak += 1
            if streak >= sim.conv_streak
                global_converged = true
                sim.verbose >= 1 && @printf("[global]  converged it=%d  d=%.3e\n", it, d)
                break
            end
        else
            streak = 0
        end
        # Stall-out: after the burn-in, if the residual has not beaten its best-so-far
        # over the last gK iterations, the coupling is limit-cycling — UNLESS the residual
        # is already inside tolerance, in which case "no longer improving" is exactly what
        # a converged sequence looks like and the rule was reading convergence as failure.
        # conv_streak demands four CONSECUTIVE sub-tolerance passes while this rule can
        # fire from pass gB+gK+1, so a point whose residual dips in and out of tolerance
        # was killed before it could complete the streak: measured under the superseded
        # step test at a_ℓ-0.20 c+0.02, streak 3 of 4 with d = 2.7e-05 against a tolerance
        # of 1e-04, stopped at pass 13.
        if gB > 0 && it > gB + gK && d >= 1.0 &&
           minimum(@view global_resid[it-gK:it]) >= minimum(@view global_resid[1:it-gK])
            sim.verbose >= 1 && @printf("[global]  stalled it=%d  d=%.3e — no improvement over %d iters, stopping\n", it, d, gK)
            break
        end
        if it == sim.maxit_global && sim.verbose >= 1
            @printf("[global]  maxit reached  it=%d  d=%.3e\n", it, d)
        end
    end

    return SolveResult(last_conv_U, last_conv_S, global_converged)
end


# ============================================================
# Multiple skilled equilibria
#
# Given θ_S every other object of the model is unique — θ_U = Θ_U(θ_S), the training
# frontier, values, cutoffs, OJS participation and masses — so an equilibrium is a root of
# one scalar function of θ_S.  solve_model!(m; θS_fixed = θ) solves the model given θ, and
#     D(θ) = ln(θ̃/θ),   θ̃ = update_theta_skilled at that state,
# carries the sign of the value of a skilled vacancy V_S(θ) = [q_S J̄_S − k_S π̄_S]/(r + q_S):
# positive at small θ, negative at large θ.  An evaluation is always a global solve: the
# skilled block alone iterates a map conditional on the lagged θ_U and m_S, whose shape
# differs.
#
# The detector rests on two properties of D, held on every curve measured (203):
#   A  D falls strictly left of the OJS onset θ_on, the lowest θ at which any type searches;
#   B  right of θ_on D has at most one hump — it may fall to a dip, rise to a top, then
#      falls — and every measured dip lies at 1.016–1.058 θ_on.
# So there are one or three equilibria, three iff D(dip) < 0 < D(top).  The attracting ones
# cross from above: L, left of the dip, and H, right of the top.  The middle root M crosses
# from below and is repelling, so it is never located.  The skilled block's damped free-entry
# step is repelled from it (restarted at M or 0.3% from it, crisis_fc at 0.995 k̂_S, the solve
# leaves for L or H), but a solve that does end there is recognised and discarded.
#
# Whether a point is left of the onset is exact and free: every s_part is 0.  The onset
# itself is bracketed (_onset) and only aims the detector: a jump to ONSET_JUMP × the
# bracket's upper end lands past every dip, and a root beyond that point cannot be L.
#
# A change in D counts only when it exceeds TOL_RELATIVE, the free-entry tolerance D is
# resolved to, so a turning point finer than the solve can see is not a turning point.
# Walks step EQ_STEP in ln θ, and a hump narrower than a step is ignored by design.
#
# Types enter on-the-job search one ability node at a time, and each entry puts a sawtooth
# of about 3e-4 on D (crisis_fc, one tooth per ≈0.003 in ln θ).  Where D is flat — near the
# top or the dip — a tooth can cross zero, making a pair of crossings about 3e-4 apart that
# no step of the walk resolves, and the normal solve can converge to either.  The attracting
# equilibrium is the OUTERMOST crossing on its side, so every root is probed EQ_PROBE
# outward and the search continues past a pair.  At the crisis_fc estimate the normal solve
# lands on such a pair at 0.655916; the outermost crossing is 0.65769.
# ============================================================

const EQ_STEP     = 0.005    # ln θ_S; the grid on which A, B and the curvature rule were measured
const EQ_PROBE    = 0.001    # ln θ_S; clears the pair of crossings one type switching makes
const EQ_MAXEVAL  = 200      # a search this long means the map does not have the shape above
const ONSET_SLOPE = 4515.0   # dM/d ln θ_S at the onset (crisis_fc); aims the first onset step only
const ONSET_TOL   = 0.02     # ln θ_S; onset bracket width, below ln(1.08/1.058)
const ONSET_JUMP  = 1.08     # jumps land at this multiple of the onset's upper bound, past every dip
# ln(θ/θ*) searched for the onset.  With no onset above, nobody searches and D falls throughout
# (A); with none below, the hump would sit where D is of order +10, far from any root.
const ONSET_RANGE = (-12.0, 5.0)

struct FreeEntryMapFailure <: Exception end

# The free-entry map and its warm-start state.  `base` is the normal solve, silenced, and
# `work` is moved by every evaluation; the caller's model is never touched, so a lone root
# keeps its model, and its Q, bit for bit.  A located root is handed back under the caller's
# SimParams.
mutable struct FreeEntryMap
    base   :: Model
    work   :: Model
    sim    :: SimParams
    n_eval :: Int
end

function FreeEntryMap(model::Model)
    base = deepcopy(model)
    base.sim = SimParams((f === :verbose ? 0 : getfield(model.sim, f) for f in fieldnames(SimParams))...)
    return FreeEntryMap(base, deepcopy(base), model.sim, 0)
end

# D at a solved state, whether anyone searches there, and the participation score M = max
# over types (implied_cutoffs), which crosses 0 at the onset.
function _map_point(m::Model)
    sc = m.skl_cache
    return (θ = sc.θ, D = log(update_theta_skilled(m) / sc.θ),
            searching = any(>(0.0), sc.s_part), M = maximum(implied_cutoffs(m).M))
end

# One evaluation, warm-started from the last.  A warm start can fail where the normal solve's
# own state converges (measured at crisis_covid, next to a root), so a failure is retried
# once from that state before it is final.
function (fm::FreeEntryMap)(θ::Float64)
    for attempt in 1:2
        fm.n_eval += 1
        fm.n_eval > EQ_MAXEVAL && throw(FreeEntryMapFailure())
        attempt == 2 && (fm.work = deepcopy(fm.base))
        solve_model!(fm.work; θS_fixed = θ).ok && return _map_point(fm.work)
    end
    throw(FreeEntryMapFailure())
end

_rises(a, b) = b.D - a.D >  TOL_RELATIVE
_falls(a, b) = b.D - a.D < -TOL_RELATIVE
_nudge(fm::FreeEntryMap, p, dir::Int) = fm(p.θ * exp(dir * EQ_STEP))

# The onset, bracketed: `lo`, where nobody searches, and `hi`, where someone does, at most
# ONSET_TOL apart in ln θ.  Starts from points already evaluated and steps outward until the
# participation flips.  Each step is aimed at M = 0 by the secant on M, and capped at a length
# that doubles from 0.25: M is near-linear in ln θ only close to the onset, and far from it
# is flat enough that an uncapped secant leaves the map.  The bracket is then narrowed by
# Illinois regula falsi on M.  `nothing` when the onset lies outside ONSET_RANGE.
function _onset(fm::FreeEntryMap, c, pts)
    above = filter(p -> p.searching, pts)
    hi = isempty(above) ? nothing : above[argmin([p.θ for p in above])]
    below = filter(p -> !p.searching && (hi === nothing || p.θ < hi.θ), pts)
    lo = isempty(below) ? nothing : below[argmax([p.θ for p in below])]

    cap = 0.25;  prev = nothing
    while lo === nothing || hi === nothing
        p, dir = lo === nothing ? (hi, -1) : (lo, +1)
        slope = prev === nothing || prev.M == p.M ? ONSET_SLOPE : (p.M - prev.M) / log(p.θ / prev.θ)
        Δ = slope > 0.0 ? clamp(-dir * p.M / slope, EQ_STEP, cap) : cap
        u = log(p.θ) + dir * Δ
        ONSET_RANGE[1] <= u - log(c.θ) <= ONSET_RANGE[2] || return nothing
        q = fm(exp(u))
        q.searching ? (hi = q) : (lo = q)
        prev = p;  cap *= 2.0
    end

    fl, fh = lo.M, hi.M;  side = 0
    while log(hi.θ / lo.θ) > ONSET_TOL
        ul, uh = log(lo.θ), log(hi.θ)
        u = fl < 0.0 < fh ? (ul * fh - uh * fl) / (fh - fl) : 0.5 * (ul + uh)
        q = fm(exp(clamp(u, ul + 0.5 * ONSET_TOL, uh - 0.5 * ONSET_TOL)))
        if q.searching
            hi = q;  fh = q.M;  side == -1 && (fl *= 0.5);  side = -1
        else
            lo = q;  fl = q.M;  side == +1 && (fh *= 0.5);  side = +1
        end
    end
    return (lo = lo, hi = hi)
end

# Which attracting root the normal solve returned, c neither M nor left of the onset.  Beyond
# ONSET_JUMP × the onset it is H: L lies left of the dip.  Nearer the onset, D is convex at L
# and concave at H (770 of 770 measured roots); where the curvature is flat, walking left to
# the onset settles it, since only from H does the walk cross the top.
function _classify(fm::FreeEntryMap, c, lo, hi, on)
    c.θ >= ONSET_JUMP * on.hi.θ && return (:H, nothing)
    curv = lo.D + hi.D - 2.0 * c.D
    curv >  TOL_RELATIVE && return (:L, nothing)
    curv < -TOL_RELATIVE && return (:H, nothing)
    s = _seek_dip(fm, c, on; land = false)
    return (s.hit !== nothing || s.hump ? :H : :L, s)
end

# From L, the top: land past the dip; D > 0 there or on the climb to the top proves three
# equilibria.  A top bracketed with D ≤ 0 is a hump below zero; falling back to L, or to the
# onset, without a turn means no hump right of L.
function _seek_top(fm::FreeEntryMap, L, on)
    b = fm(max(ONSET_JUMP * on.hi.θ, L.θ * exp(EQ_STEP)))
    b.D > TOL_RELATIVE && return (hit = b, ext = b.D, hump = true)
    c = _nudge(fm, b, +1)
    if _rises(b, c)
        while true
            c.D > TOL_RELATIVE && return (hit = c, ext = c.D, hump = true)
            n = _nudge(fm, c, +1)
            _falls(c, n) && return (hit = nothing, ext = c.D, hump = true)
            c = n
        end
    end
    while true
        c = _nudge(fm, b, -1)
        (c.θ <= L.θ * exp(0.5 * EQ_STEP) || !c.searching) && return (hit = nothing, ext = NaN, hump = false)
        c.D > TOL_RELATIVE && return (hit = c, ext = c.D, hump = true)
        _falls(b, c) && return (hit = nothing, ext = b.D, hump = true)
        b = c
    end
end

# From H, the dip: land past it and walk left, over the top if the landing is beyond it.
# D < 0 anywhere proves three equilibria.  A dip bracketed with D ≥ 0 is a hump above zero;
# reaching the onset without D having fallen means no hump.  `land = false` walks from H
# itself.
function _seek_dip(fm::FreeEntryMap, H, on; land::Bool = true)
    b = land ? fm(min(ONSET_JUMP * on.hi.θ, H.θ * exp(-EQ_STEP))) : H
    b.D < -TOL_RELATIVE && return (hit = b, ext = b.D, hump = true)
    ext = b.D;  fallen = false
    while true
        c = _nudge(fm, b, -1)
        c.D < -TOL_RELATIVE && return (hit = c, ext = c.D, hump = true)
        _rises(c, b) && (fallen = true)
        fallen && _falls(c, b) && return (hit = nothing, ext = b.D, hump = true)
        ext = min(ext, c.D)
        c.searching || return (hit = nothing, ext = fallen ? ext : NaN, hump = fallen)
        b = c
    end
end

# The attracting root beyond p in direction dir, where D has the sign of dir between the
# middle root and it: step outward, doubling, until D changes sign, then Illinois regula
# falsi in ln θ to half the free-entry tolerance, and past any tooth pair (_beyond).  The
# state there is certified by the full battery, skilled free entry included.
function _locate(fm::FreeEntryMap, p, dir::Int)
    a = p
    while dir * a.D <= 0.5 * TOL_RELATIVE
        a = _nudge(fm, a, dir)
    end
    s = EQ_STEP
    b = fm(a.θ * exp(dir * s))
    while dir * b.D > 0.5 * TOL_RELATIVE
        a = b;  s *= 2.0
        b = fm(a.θ * exp(dir * s))
    end
    ua, fa = log(a.θ), a.D;  ub, fb = log(b.θ), b.D
    side = 0;  q = b
    while abs(q.D) > 0.5 * TOL_RELATIVE
        u = (ua * fb - ub * fa) / (fb - fa)
        q = fm(exp(u))
        if sign(q.D) == sign(fb)
            ub, fb = u, q.D;  side == -1 && (fa *= 0.5);  side = -1
        else
            ua, fa = u, q.D;  side == +1 && (fb *= 0.5);  side = +1
        end
    end
    m = deepcopy(fm.work)
    m.sim = fm.sim
    further = _beyond(fm, q, dir)
    further === nothing || return further
    return (θ = q.θ, model = m, certified = global_residuals(m).Rmax < 1.0)
end

# Whether the root at r is the outermost crossing in direction dir: D must already have the
# outer sign EQ_PROBE beyond it.  If not, r is one of a tooth pair, and the root is located
# from the probe.
function _beyond(fm::FreeEntryMap, r, dir::Int)
    o = fm(r.θ * exp(dir * EQ_PROBE))
    return dir * o.D > 0.5 * TOL_RELATIVE ? _locate(fm, o, dir) : nothing
end

"""
    skilled_equilibria(model) → NamedTuple

Every attracting skilled equilibrium at the parameters of `model`, a converged solve.

  roots     the attracting equilibria in increasing θ_S, each `(label, θ, model, certified)`:
            one, or the pair `:L` and `:H`.  A lone root is labelled by its side of the hump
            — `:L` left of it, `:H` right of it, `:U` where D has no hump — so that one branch
            can be followed across parameter points (`branch` in select_equilibrium).
  start     which root the normal solve returned: `:L`, `:H`, `:U`, or `:M`, discarded.
  moved     whether the normal solve's root was replaced: it was one of a tooth pair, or a
            sub-cell root with D of one sign a step either side.  When false, the root is
            the normal solve itself, model untouched, and so is its Q.
  D_ext     D at the extremum that decided the count — the top seen from L, the dip seen
            from H — at walk resolution, so its distance from 0 is the distance to the fold;
            NaN without a hump, and from M, where no extremum is needed.
  n_eval    evaluations of the free-entry map spent.
  complete  false when an evaluation failed: the roots found by then are returned, and the
            start root stands in for a lone root unless it was M.

A solve that converged at a corner (`corner_limit`, a market holding no one) has no
interior skilled market to be multiple in, and is returned as the lone root.
"""
function skilled_equilibria(model::Model)
    c  = _map_point(model)
    r0 = (θ = c.θ, model = model, certified = true)
    result(roots, start, D_ext, n_eval, complete) =
        (roots = sort!(identity.(roots); by = r -> r.θ), start = start,
         moved = !any(r -> r.model === model, roots), D_ext = D_ext, n_eval = n_eval, complete = complete)
    model.corner_limit && return result([(label = :U, r0...)], :U, NaN, 0, true)

    fm    = FreeEntryMap(model)
    start = :U;  D_ext = NaN
    roots = []
    try
        # The root at step resolution.  Left of the onset nobody switches and D falls, so the
        # normal solve's root is resolved.  Otherwise a step either side: D of one sign on
        # both means a sub-cell root (a tooth, see above), and the resolved root lies the way
        # that sign points; it replaces the normal solve's.
        nudged(c) = c.searching ? (_nudge(fm, c, -1), _nudge(fm, c, +1)) : (c, c)
        lo, hi = nudged(c);  resolved = !c.searching
        while c.searching && sign(lo.D) == sign(hi.D)
            dir = lo.D < 0.0 ? -1 : +1
            r0  = _locate(fm, dir < 0 ? lo : hi, dir);  resolved = true
            c   = _map_point(r0.model)
            lo, hi = nudged(c)
        end

        if _rises(lo, hi)
            # D < 0 a step left and > 0 a step right: three equilibria, by the intermediate
            # value theorem alone.
            start = :M
            push!(roots, (label = :L, _locate(fm, lo, -1)...), (label = :H, _locate(fm, hi, +1)...))
        else
            on = _onset(fm, c, unique([c, lo, hi]))
            if on !== nothing
                start, s = c.searching ? _classify(fm, c, lo, hi, on) : (:L, nothing)
                s === nothing || s.hit !== nothing || s.hump || (s = nothing)
                s === nothing && (s = start === :L ? _seek_top(fm, c, on) : _seek_dip(fm, c, on))
            end
            if on === nothing || (s.hit === nothing && !s.hump)
                start = :U
            else
                D_ext = s.ext
            end
            # The start root, unless it is one of a tooth pair.  Without a hump D falls at a
            # slope near −0.9, ten times what a tooth can reverse, so no pair forms there.
            if !resolved && start !== :U
                further = _beyond(fm, c, start === :L ? -1 : +1)
                further === nothing || (r0 = further)
            end
            push!(roots, (label = start, r0...))
            if on !== nothing && s.hit !== nothing
                start === :L ? push!(roots, (label = :H, _locate(fm, s.hit, +1)...)) :
                               push!(roots, (label = :L, _locate(fm, s.hit, -1)...))
            end
        end
    catch e
        e isa FreeEntryMapFailure || rethrow()
        start === :M || any(r -> r.model === r0.model, roots) || push!(roots, (label = start, r0...))
        return result(roots, start, D_ext, fm.n_eval, false)
    end
    return result(roots, start, D_ext, fm.n_eval, true)
end

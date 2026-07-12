############################################################
# solver.jl — Global equilibrium solver (RoySearch)
#
# Public entry points
#   solve_model(common, unsk_par, skl_par, sim; Nx, Np_U, Np_S)
#       → build grids and caches, run the global loop, return (Model, SolveResult).
#   solve_model!(model)
#       → run the global loop in place on an existing Model.
#
# Global fixed point on the link variables (U_S, m_S), both Anderson(m=1)
# accelerated.  U_S(aS) is one-dimensional in the skilled aptitude; the
# trained mass m_S(aU,aS) is two-dimensional on the copula grid.  The
# remaining link objects are derived deterministically each pass:
#   f_U         = θ_U q_U(θ_U)                     from the unskilled solve
#   E_U(aU,1)   = U^search(aU) + β_U S_U(aU,1)     from the frontier surplus
#   d(aU,aS) u_S(aU,aS)                            from the skilled solve
#   m_S(aU,aS)  = φ t(aU,aS) / (ν + d(aU,aS) f_U)
#
# Per global iteration:
#   A. Solve the unskilled block with the carried d·u_S.
#   B. Form f_U and E_U(·,1).
#   C. Build m_S using sc.d (from the previous pass) and set sc.m_S.
#   D. Solve the skilled block (inner loop updates sc.d).
#   E. Recompute m_S using the updated sc.d.
#   F. Anderson on the joint [U_S; vec(m_S)] with per-block scaling.
#   G. Write back: sc.U ← new U_S; sc.m_S ← new m_S; uc.duS_carry ← d·u_S.
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
    solve_model(common, unsk_par, skl_par, sim; Nx, Np_U, Np_S) → (Model, SolveResult)
"""
function solve_model(common::CommonParams, unsk_par::UnskilledParams,
                     skl_par::SkilledParams, sim::SimParams;
                     Nx::Int = 200, Np_U::Int = 200, Np_S::Int = 200)
    grids   = build_common_grids(common, Nx)
    pU, wpU = build_gl_grid(Np_U);  u_grids = UnskilledGrids(p = pU, wp = wpU)
    pS, wpS = build_gl_grid(Np_S);  s_grids = SkilledGrids(p = pS, wp = wpS)
    model   = build_model(common, grids, unsk_par, u_grids, skl_par, s_grids, sim; Nx, Np_S)
    result  = solve_model!(model)
    return model, result
end


# ---------------------------------------------------------------------------
# m_S(aU,aS) = φ t(aU,aS) / (ν + d(aU,aS) f_U)
# ---------------------------------------------------------------------------
function _mS_from_t(t::Matrix{Float64}, d::Matrix{Float64},
                    φ::Float64, ν::Float64, fU::Float64)
    mS = similar(t)
    @inbounds for k in eachindex(t)
        denom = ν + clamp(d[k], 0.0, 1.0) * fU
        mS[k] = denom > 1e-14 ? max(φ * t[k] / denom, 0.0) : 0.0
    end
    return mS
end


# ---------------------------------------------------------------------------
# solve_model! — in-place global fixed point
# ---------------------------------------------------------------------------
"""
    solve_model!(model) → SolveResult

Run the global fixed-point loop in place on `model`.
"""
function solve_model!(model::Model)::SolveResult
    cp = model.common;  sc = model.skl_cache;  uc = model.unsk_cache
    up = model.unsk_par;  sim = model.sim
    φ  = cp.φ;  ν = cp.ν
    Nx = length(model.grids.x);  βU = up.β

    aa = Anderson1(Nx + Nx * Nx)

    # First-pass m_S uses d ≡ 0 ⇒ (φ/ν) t.
    sc.m_S .= (φ / ν) .* uc.t
    mS_cur = copy(sc.m_S)

    s_U = 1.0;  s_M = 1.0
    last_conv_U = false;  last_conv_S = false;  global_converged = false
    streak = 0

    for it in 1:sim.maxit_global
        US_old = copy(sc.U)
        mS_old = copy(mS_cur)

        # A. Unskilled block with carried d·u_S.
        res_U = solve_unskilled_block!(model; US_in = sc.U)
        res_U.rejected && return SolveResult(false, false, false)
        last_conv_U = res_U.converged
        if !isfinite(uc.θ) || any(!isfinite, uc.t) || any(!isfinite, uc.Usearch) || any(!isfinite, uc.pstar)
            sim.verbose >= 1 && @printf("[global]  NaN/Inf in unskilled block at it=%d — aborting\n", it)
            return SolveResult(false, false, false)
        end

        # B. Unskilled-side outputs consumed by the skilled block.
        fU  = jobfinding_rate(uc.θ, up.μ, up.η)
        SU1 = uc.Jfrontier ./ max(1.0 - βU, 1e-14)
        EU1 = uc.Usearch .+ βU .* SU1

        # C. m_S using the previous-pass d; install for the skilled solve.
        sc.m_S = _mS_from_t(uc.t, sc.d, φ, ν, fU)

        # D. Skilled block; inner loop updates sc.d.
        res_S = solve_skilled_block!(model; fU = fU, EU1 = EU1)
        res_S.rejected && return SolveResult(false, false, false)
        last_conv_S = res_S.converged
        if !isfinite(sc.θ) || any(!isfinite, sc.U) || any(!isfinite, sc.pstar) ||
           any(!isfinite, sc.d) || any(!isfinite, sc.u_frac)
            sim.verbose >= 1 && @printf("[global]  NaN/Inf in skilled block at it=%d — aborting\n", it)
            return SolveResult(false, false, false)
        end

        US_raw = copy(sc.U)

        # E. m_S using the just-updated d.
        mS_raw = _mS_from_t(uc.t, sc.d, φ, ν, fU)

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

        # G. Refresh the cross-market contribution for the next pass.
        #    u_S(aU,aS) = û(aS) m_S(aU,aS) on d=0 cells, = m_S on d=1 cells.
        @inbounds for j in 1:Nx, i in 1:Nx
            dij = clamp(sc.d[i, j], 0.0, 1.0)
            uS_ij = dij > 0.5 ? mS_new[i, j] : sc.u_frac[j] * mS_new[i, j]
            uc.duS_carry[i, j] = max(dij * uS_ij, 0.0)
        end

        dU = supnorm(US_new, US_old)
        dM = maximum(abs, mS_new .- mS_old)
        d  = max(dU, dM)

        if sim.verbose >= 1 && (it == 1 || it % sim.verbose_stride == 0)
            @printf("[global it=%d]  maxΔ=%.3e  (ΔU_S=%.3e  Δm_S=%.3e)  θ_U=%.4f  θ_S=%.4f\n", it, d, dU, dM, uc.θ, sc.θ)
        end

        if d < sim.tol_global
            streak += 1
            if streak >= sim.conv_streak
                global_converged = true
                sim.verbose >= 1 && @printf("[global]  converged it=%d  d=%.3e\n", it, d)
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

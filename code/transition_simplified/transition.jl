############################################################
# transition.jl — Forward simulation of transition dynamics
#
# Implements the simplified transition-dynamics computation:
#   - Policies jump immediately to the crisis steady state
#   - Distributions evolve forward from baseline initial conditions
#   - Tightness is recomputed at each step via free entry
#
# This is a valid approximation for permanent unexpected shocks.
# For the full forward-backward algorithm (Section 11.9 of the
# notes), one would additionally need a backward pass over
# time-dependent HJB equations.
#
# Key function:
#   simulate_transition(model_base, model_crisis,
#                       obj_base, obj_crisis; T_max, dt)
#       → TransitionResult (NamedTuple of time paths)
############################################################


# ============================================================
# TransitionResult — stores all time paths
# ============================================================

"""
    TransitionResult

NamedTuple containing time paths of aggregate quantities
along the transition from baseline to crisis steady state.
"""
const TransitionResult = NamedTuple{(
    :times,
    :ur_total, :ur_U, :ur_S,
    :skilled_share, :training_share,
    :theta_U, :theta_S,
    :f_U, :f_S,
    :agg_uU, :agg_uS, :agg_t, :agg_eU, :agg_eS,
    :agg_mU, :agg_mS,
), NTuple{17, Vector{Float64}}}


# ============================================================
# Main transition simulation
# ============================================================

"""
    simulate_transition(model_base, model_crisis,
                        obj_base, obj_crisis;
                        T_max=120.0, dt=0.05)
        → TransitionResult

Forward-simulate the distribution dynamics after a permanent
unexpected regime switch from baseline to crisis.

Arguments
─────────
  model_base, model_crisis : solved Model objects
  obj_base, obj_crisis     : equilibrium NamedTuples from
                             compute_equilibrium_objects()
  T_max : simulation horizon in months
  dt    : Euler step size in months

Returns
───────
  TransitionResult : NamedTuple of time paths (vectors of length N+1)

Algorithm
─────────
  1. Initial distributions = baseline steady state.
  2. Policies (τ, p*_U, p*_S, p^oj) = crisis steady state.
  3. At each time step:
     a. Compute derived densities (eU, eS_tot, mU).
     b. Recompute tightness via free entry (using crisis
        value functions and current distributions).
     c. Compute transition hazards.
     d. Euler update on (uU, t, uS, mS).
     e. Record aggregate quantities.
"""
function simulate_transition(
    model_base    :: Model,
    model_crisis  :: Model,
    obj_base,
    obj_crisis;
    T_max :: Float64 = 120.0,
    dt    :: Float64 = 0.05,
)
    # ── Extract parameters ───────────────────────────────────
    cp   = model_crisis.common
    rp   = model_crisis.regime
    up   = model_crisis.unsk_par
    sp   = model_crisis.skl_par
    pre  = model_crisis.skl_pre

    r  = cp.r;   ν  = cp.ν;   φ  = cp.φ
    PU = rp.PU;  PS = rp.PS;  αU = rp.α_U

    xg  = model_crisis.grids.x
    wx  = model_crisis.grids.wx
    ℓ   = model_crisis.grids.ℓ
    Nx  = length(xg)

    pg  = model_crisis.skl_grids.p
    wpS = model_crisis.skl_grids.wp
    NpS = length(pg)

    # ── Crisis steady-state policies ─────────────────────────
    τ_crisis     = copy(obj_crisis.tauT)        # training policy
    pstar_U_cr   = copy(obj_crisis.pstar_U)     # unskilled separation cutoff
    pstar_S_cr   = copy(obj_crisis.pstar_S)     # skilled separation cutoff
    poj_cr       = copy(obj_crisis.poj)         # OJS cutoff

    # Crisis steady-state firm values at frontier (for free entry)
    JU_frontier_cr = copy(obj_crisis.JU_frontier)

    # Crisis steady-state skilled surplus (for free entry)
    # J_S values depend on distribution composition in general,
    # but for the simplified approach we use crisis SS values.
    Jskl_surface_cr = copy(obj_crisis.Jskl_surface)

    # ── Precomputed Γ values for skilled hazards ─────────────
    # Γ(p*_S(x)) for each x — needed for skilled separation hazard
    Γ_pstar_S = zeros(Nx)
    for ix in 1:Nx
        j0 = pcut_index(pg, clamp(pstar_S_cr[ix], 0.0, 1.0))
        Γ_pstar_S[ix] = pre.Γvals[j0]
    end

    # ── Initial conditions from baseline steady state ────────
    uU  = copy(obj_base.uU)        # untrained unemployed density
    t   = copy(obj_base.tU)        # training density
    uS  = copy(obj_base.uS)        # skilled unemployed density
    mS  = copy(obj_base.mS_vec)    # skilled segment mass by type

    # ── Allocate storage ─────────────────────────────────────
    N_steps = Int(ceil(T_max / dt))
    N_out   = N_steps + 1

    times          = zeros(N_out)
    ur_total_path  = zeros(N_out)
    ur_U_path      = zeros(N_out)
    ur_S_path      = zeros(N_out)
    skilled_share_path  = zeros(N_out)
    training_share_path = zeros(N_out)
    theta_U_path   = zeros(N_out)
    theta_S_path   = zeros(N_out)
    f_U_path       = zeros(N_out)
    f_S_path       = zeros(N_out)
    agg_uU_path    = zeros(N_out)
    agg_uS_path    = zeros(N_out)
    agg_t_path     = zeros(N_out)
    agg_eU_path    = zeros(N_out)
    agg_eS_path    = zeros(N_out)
    agg_mU_path    = zeros(N_out)
    agg_mS_path    = zeros(N_out)

    # ── Helper: compute tightness from free entry ────────────
    function compute_theta_U(uU_cur)
        # q_U(θ) = k_U / ∫ J_U(x,1) u_U(x) dx
        # Using crisis J_U frontier values
        denom = dot(JU_frontier_cr .* uU_cur, wx)
        if denom <= 1e-14
            return obj_crisis.thetaU  # fallback to crisis SS
        end
        qU = up.k / denom
        # Invert: θ = (μ / q)^(1/η)  since q(θ) = μ θ^(-η)
        if qU >= up.μ
            return 1e-4  # very low tightness
        end
        return (up.μ / qU)^(1.0 / up.η)
    end

    function compute_theta_S(uS_cur, mS_cur)
        # Simplified: use the ratio of current seekers to
        # crisis SS seekers to scale crisis θ_S.
        # Full version would recompute J_S from current distributions.
        agg_uS_cur = dot(uS_cur, wx)
        agg_uS_ss  = dot(obj_crisis.uS, wx)
        if agg_uS_cur <= 1e-14
            return obj_crisis.thetaS
        end
        # Scale: more seekers → lower θ (more congestion)
        ratio = agg_uS_ss / max(agg_uS_cur, 1e-14)
        return obj_crisis.thetaS * clamp(ratio^(1.0 / sp.η), 0.1, 10.0)
    end

    # ── Record initial state ─────────────────────────────────
    function record_state!(idx, time, uU_c, t_c, uS_c, mS_c, θU, θS)
        times[idx] = time

        # Derived quantities
        mU = max.(ℓ .- (φ/ν) .* t_c, 0.0)     # approximate unskilled-segment mass
        eU = max.(mU .- uU_c .- t_c, 0.0)      # unskilled employed (residual)
        eS_tot = max.(mS_c .- uS_c, 0.0)       # skilled employed total

        a_uU  = dot(uU_c, wx)
        a_t   = dot(t_c, wx)
        a_eU  = dot(eU, wx)
        a_mU  = dot(mU, wx)
        a_uS  = dot(uS_c, wx)
        a_eS  = dot(eS_tot, wx)
        a_mS  = dot(mS_c, wx)
        total = a_mU + a_mS

        fU = θU * q_from_theta(θU, up.μ, up.η)
        fS = θS * q_from_theta(θS, sp.μ, sp.η)

        agg_uU_path[idx]         = a_uU
        agg_t_path[idx]          = a_t
        agg_eU_path[idx]         = a_eU
        agg_mU_path[idx]         = a_mU
        agg_uS_path[idx]         = a_uS
        agg_eS_path[idx]         = a_eS
        agg_mS_path[idx]         = a_mS
        ur_U_path[idx]           = a_uU / max(a_mU, 1e-14)
        ur_S_path[idx]           = a_uS / max(a_mS, 1e-14)
        ur_total_path[idx]       = (a_uU + a_uS) / max(total, 1e-14)
        skilled_share_path[idx]  = a_mS / max(total, 1e-14)
        training_share_path[idx] = a_t  / max(total, 1e-14)
        theta_U_path[idx]        = θU
        theta_S_path[idx]        = θS
        f_U_path[idx]            = fU
        f_S_path[idx]            = fS
    end

    # ── Initial tightness (jumps toward crisis values) ───────
    # At t=0+, policies are crisis SS but distributions are baseline.
    # Recompute tightness from free entry with baseline distributions.
    θU = compute_theta_U(uU)
    θS = compute_theta_S(uS, mS)

    record_state!(1, 0.0, uU, t, uS, mS, θU, θS)

    # ── Forward simulation (Euler method) ────────────────────
    for n in 1:N_steps
        time = n * dt

        # Current derived quantities
        mU  = max.(ℓ .- (φ/ν) .* t, 0.0)
        eU  = max.(mU .- uU .- t, 0.0)
        eS_tot = max.(mS .- uS, 0.0)

        # Current rates
        fU = θU * q_from_theta(θU, up.μ, up.η)
        fS = θS * q_from_theta(θS, sp.μ, sp.η)

        # Destruction hazards (using crisis policies)
        # δ_U(x) = λ_U · p*(x)^α_U
        δU = up.λ .* clamp.(pstar_U_cr, 0.0, 1.0) .^ αU

        # δ_S^end(x) = ξ + λ_S · Γ(p*_S(x))
        δS_end = sp.ξ .+ sp.λ .* Γ_pstar_S

        # ── ODEs (Section 11.3–11.4 of notes) ───────────────

        # Training stock:
        #   ∂t t(x) = τ(x) u_U(x) − (φ + ν) t(x)
        dt_dt = τ_crisis .* uU .- (φ + ν) .* t

        # Untrained unemployment:
        #   ∂t u_U(x) = ν ℓ(x) + δ_U(x) e_U(x)
        #              − (f_U + τ(x) + ν) u_U(x)
        du_U = ν .* ℓ .+ δU .* eU .- (fU .+ τ_crisis .+ ν) .* uU

        # Skilled unemployment:
        #   ∂t u_S(x) = φ t(x) + δ_S^end(x) e_S^tot(x)
        #              − (ν + f_S(1 − Γ(p*_S(x)))) u_S(x)
        acceptance = 1.0 .- Γ_pstar_S    # prob of acceptable offer
        du_S = φ .* t .+ δS_end .* eS_tot .- (ν .+ fS .* acceptance) .* uS

        # Skilled segment mass:
        #   ∂t m_S(x) = φ t(x) − ν m_S(x)
        dm_S = φ .* t .- ν .* mS

        # ── Euler update ─────────────────────────────────────
        uU .+= dt .* du_U
        t  .+= dt .* dt_dt
        uS .+= dt .* du_S
        mS .+= dt .* dm_S

        # Enforce non-negativity
        uU .= max.(uU, 0.0)
        t  .= max.(t,  0.0)
        uS .= max.(uS, 0.0)
        mS .= max.(mS, 0.0)

        # Ensure consistency: u_S ≤ m_S
        uS .= min.(uS, mS)

        # ── Recompute tightness ──────────────────────────────
        θU = compute_theta_U(uU)
        θS = compute_theta_S(uS, mS)

        # ── Record state ─────────────────────────────────────
        record_state!(n + 1, time, uU, t, uS, mS, θU, θS)
    end

    # ── Return TransitionResult ──────────────────────────────
    return (
        times          = times,
        ur_total       = ur_total_path,
        ur_U           = ur_U_path,
        ur_S           = ur_S_path,
        skilled_share  = skilled_share_path,
        training_share = training_share_path,
        theta_U        = theta_U_path,
        theta_S        = theta_S_path,
        f_U            = f_U_path,
        f_S            = f_S_path,
        agg_uU         = agg_uU_path,
        agg_uS         = agg_uS_path,
        agg_t          = agg_t_path,
        agg_eU         = agg_eU_path,
        agg_eS         = agg_eS_path,
        agg_mU         = agg_mU_path,
        agg_mS         = agg_mS_path,
    )
end


# ============================================================
# Helper: pcut_index (find first grid point ≥ cutoff)
# (may already be defined in skilled.jl — guard against redef)
# ============================================================
if !@isdefined(_pcut_index_transition)
    const _pcut_index_transition = true
    # pcut_index is assumed to be available from skilled.jl
    # If not, uncomment:
    # function pcut_index(pgrid, cutoff)
    #     j = searchsortedfirst(pgrid, cutoff)
    #     return clamp(j, 1, length(pgrid))
    # end
end

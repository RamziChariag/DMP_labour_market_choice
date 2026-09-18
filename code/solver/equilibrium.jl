############################################################
# equilibrium.jl — Post-solution equilibrium objects (RoySearch)
#
# compute_equilibrium_objects(model)
#   Returns a NamedTuple with the stationary densities, value / surplus /
#   firm surfaces, wages, population accounting, and transition rates —
#   everything model_moments consumes.
#
# Dimensionality.  Value/surplus/wage surfaces are 1D in own ability
# (unskilled in aU, skilled in aS) × the p-grid, exactly as solved.
# Stationary densities are 2D on the (aU,aS) copula grid.  The skilled
# employed density factors as e_S(aU,aS,p) = ê(aS,p) · m_S(aU,aS) over the
# non-draining fraction 1−d of each cell (notes §472–483); marginal
# aggregates use the copula weights W2 and the marginals wa_U, wa_S.
############################################################

"""
    compute_equilibrium_objects(model) → NamedTuple
"""
# Unemployment-duration thresholds, in WEEKS, at which the survivor is reported.
# These mirror DURATION_THRESHOLDS_WK in code/data_and_descriptives/setup.jl, which
# is canonical because the data pipeline runs standalone; code/smm/moments.jl derives
# its moment names from THIS constant, so a change here propagates to the SMM layer
# automatically and only the data-side copy needs editing by hand.
# Each threshold sits one week above a CPS duration heap (4/13/26/52) so a threshold
# never splits a heap — the convention ltu_share_S already followed with 27.
const DURATION_THRESHOLDS_WK = [5.0, 14.0, 27.0, 53.0]

const _WEEKS_TO_MONTHS = 12.0 / 52.0

function compute_equilibrium_objects(model::Model)
    cp = model.common;  uc = model.unsk_cache;  sc = model.skl_cache
    up = model.unsk_par;  sp = model.skl_par;  pre = model.skl_pre

    xg  = model.grids.x
    waU = model.grids.wa_U;  waS = model.grids.wa_S;  W2 = model.grids.copula.W2
    Fell = model.grids.copula.F                        # marginal CDF (ability rank) at nodes
    pgU = model.unsk_grids.p;  wpU = model.unsk_grids.wp
    pg  = model.skl_grids.p;   wpS = model.skl_grids.wp
    Nx  = length(xg);  NpU = length(pgU);  NpS = length(pg)

    r  = cp.r;  ν = cp.ν;  φ = cp.φ
    βU = up.β;  λU = up.λ;  αU = up.α_U;  ξU = up.ξ
    βS = sp.β;  λS = sp.λ;  σS = sp.σ * exp(cp.A);  ξS = sp.ξ
    PU = exp(cp.A) * up.PU;  PS = exp(cp.A) * sp.PS;  bS = sp.bS * exp(cp.A)

    wGU = build_unskilled_G_weights(pgU, wpU, αU)
    θU  = uc.θ;  f_U = jobfinding_rate(θU, up.μ, up.η)

    # ── Stationary densities (2D on the copula grid) ─────────────────────
    uU     = copy(uc.u)                                # untrained unskilled unemployed
    tU     = copy(uc.t)                                # in training
    d_mat  = clamp.(copy(sc.d), 0.0, 1.0)
    mS_mat = _mS_from_t(tU, d_mat, φ, ν, f_U)          # trained mass
    mU_mat = max.(W2 .- mS_mat, 0.0)                   # untrained-segment mass
    eU_mat = max.(mU_mat .- uU .- tU, 0.0)             # unskilled employed mass

    # Skilled unemployed / employed reconstructed from per-aS unit shapes, with the
    # cell SPLIT by the drain fraction rather than assigned wholly to one branch:
    #   the d part:    u_S = m_S,        e_S ≡ 0 (these seek in the U-market)
    #   the 1−d part:  u_S = û(aS) m_S,  e_S(p) = ê(aS,p) m_S
    # Thresholding at 0.5 here would discard the fractional boundary node and put
    # the step function back, whatever the producer does upstream.  The two pieces
    # are kept apart because the flow rates below apply a different exit hazard to
    # each — a crosser leaves at f_U, a stayer at κ_S(1−Γ_o(p*_S)).
    uS_drain = similar(mS_mat)                         # crossers: all of the sub-mass
    uS_stay  = similar(mS_mat)                         # stayers: unemployed share only
    @inbounds for j in 1:Nx, i in 1:Nx
        uS_drain[i, j] = d_mat[i, j] * mS_mat[i, j]
        uS_stay[i, j]  = (1.0 - d_mat[i, j]) * sc.u_frac[j] * mS_mat[i, j]
    end
    uS_mat = uS_drain .+ uS_stay
    # eS on (aS,p) aggregated over aU (only non-draining mass carries employment):
    mcol0  = [sum((1.0 .- d_mat[:, j]) .* mS_mat[:, j]) for j in 1:Nx]   # by aS
    eS_pS  = [sc.e_frac[j, jp] * mcol0[j] for j in 1:Nx, jp in 1:NpS]    # (aS, p) employed density
    eS_totS = [dot(eS_pS[j, :], wpS) for j in 1:Nx]                      # employed mass per aS

    # ── Unskilled employment surface e_U(aU, p) ──────────────────────────
    # Row mass in aU: integrate the 2D employed density over aS.
    eU_rowU = [dot(eU_mat[i, :], ones(Nx)) for i in 1:Nx]               # Σ_j eU_mat[i,j] (mass, W2 already in it)
    uU_rowU = [sum(uU[i, :]) for i in 1:Nx]
    eU_surface = zeros(Nx, NpU)
    for i in 1:Nx
        pstar_x = clamp01(uc.pstar[i]);  denom_eU = ν + λU
        for jp in 1:NpU
            p   = pgU[jp]
            ω_j = _soft_weight(p, pstar_x, pgU, jp, NpU)
            if ω_j > 0.0
                g_p = (p <= 0.0) ? 0.0 : αU * p^(αU - 1.0)
                eU_surface[i, jp] = ω_j * λU * g_p / denom_eU * eU_rowU[i]
            end
        end
        eU_rowU[i] > 0.0 && (eU_surface[i, end] += f_U * uU_rowU[i] / (ν + λU))
    end

    # Force tiny aggregate employment to a clean zero (corner solutions).
    _emp_tol = 1e-12
    agg_eU_raw = sum(eU_mat)
    agg_eS_raw = sum(eS_totS[j] * sum(W2[:, j]) for j in 1:Nx)   # placeholder; recomputed below
    if sum(eU_mat) < _emp_tol
        fill!(eU_mat, 0.0);  fill!(eU_surface, 0.0)
    end
    if sum(eS_pS) < _emp_tol
        fill!(eS_pS, 0.0);  fill!(eS_totS, 0.0)
    end

    # ── Unskilled values and policy (1D in aU; T in aS) ──────────────────
    Usearch = copy(uc.Usearch);  Tv = copy(uc.T)
    c_of_a  = a -> training_cost(a, cp.c)   # psychological cost: NOT scaled by A
    net_T   = [-c_of_a(xg[j]) + Tv[j] for j in 1:Nx]               # net training value in aS
    τ_mat   = copy(uc.τT)
    JU_frontier = copy(uc.Jfrontier)
    SU1     = JU_frontier ./ max(1.0 - βU, 1e-14)
    EU1     = Usearch .+ βU .* SU1
    pstar_U = copy(uc.pstar)

    # ── Unskilled surplus / firm / worker surfaces (1D in aU) ────────────
    SU_surface = zeros(Nx, NpU)
    for i in 1:Nx
        Svec = zeros(NpU)
        solve_unskilled_surplus_on_grid!(Svec, pgU, wGU, PU * xg[i], r, ν, λU, Usearch[i], clamp01(uc.pstar[i]), ξU)
        SU_surface[i, :] = Svec
    end
    JU_surface = (1.0 - βU) .* SU_surface
    EU_surface = Usearch .+ βU .* SU_surface

    # ── Skilled value / surplus surfaces (1D in aS) ──────────────────────
    denom_nb     = max(1.0 - βS, 1e-14)
    S0_surface   = sc.J0 ./ denom_nb
    S1_surface   = sc.J1 ./ denom_nb
    Smax_surface = max.(S0_surface, S1_surface)
    pstar_S = copy(sc.pstar);  poj = copy(sc.poj);  US = copy(sc.U)

    # ── Wages ─────────────────────────────────────────────────────────────
    θS = sc.θ;  κS = jobfinding_rate(θS, sp.μ, sp.η)
    wΓ = pre.γvals .* wpS
    tailS = zeros(Nx, NpS)
    for j in 1:Nx
        acc = 0.0
        for jp in NpS:-1:1
            acc += Smax_surface[j, jp] * wΓ[jp];  tailS[j, jp] = acc
        end
    end
    # Read from one cell BELOW the reservation node: S^max carries the soft
    # coverage weight, so the cell straddling p*_S contributes its covered
    # fraction.  Same index skilled_inner_loop! forms I_S at.
    I_full = [tailS[j, max(pcut_index(pg, clamp01(pstar_S[j])) - 1, 1)] for j in 1:Nx]

    # w_U(aU,p) = β_U A P_U aU p + (1−β_U)(r+ν) U^search(aU)
    wU_surface = fill(NaN, Nx, NpU)
    for i in 1:Nx
        pst = clamp01(pstar_U[i]);  outside = (1.0 - βU) * (r + ν) * Usearch[i]
        for jp in 1:NpU
            _soft_weight(pgU[jp], pst, pgU, jp, NpU) > 0.0 &&
                (wU_surface[i, jp] = βU * PU * xg[i] * pgU[jp] + outside)
        end
    end

    # w^0_S(aS,p), w^1_S(aS,p): linear in aS, no γ_S.
    wS0_surface = fill(NaN, Nx, NpS)
    wS1_surface = fill(NaN, Nx, NpS)
    for j in 1:Nx
        pst = clamp01(pstar_S[j]);  PSeff = PS * xg[j]
        flow0 = (1.0 - βS) * bS + βS * (1.0 - βS) * κS * I_full[j]
        flow1 = (1.0 - βS) * (bS + σS)
        for jp in 1:NpS
            if _soft_weight(pg[jp], pst, pg, jp, NpS) > 0.0
                wS0_surface[j, jp] = βS * PSeff * pg[jp] + flow0
                I_low = max(I_full[j] - tailS[j, jp], 0.0)
                wS1_surface[j, jp] = βS * PSeff * pg[jp] + flow1 + βS * (1.0 - βS) * κS * I_low
            end
        end
    end

    # ── Wage densities (mass-weighted histogram) ─────────────────────────
    function _wage_density(wages, weights, wgrid; mass_tol = 1e-12)
        Nb = length(wgrid) - 1;  bw = step(wgrid);  dens = zeros(Nb)
        for (w, m) in zip(wages, weights)
            raw  = (w - first(wgrid)) / bw
            j_lo = clamp(floor(Int, raw) + 1, 1, Nb)
            j_hi = clamp(j_lo + 1, 1, Nb)
            α    = clamp(raw - (j_lo - 1), 0.0, 1.0)
            dens[j_lo] += (1.0 - α) * m;  dens[j_hi] += α * m
        end
        total = sum(dens) * bw
        return total > mass_tol ? dens ./ total : dens
    end

    wages_U  = Float64[];  mass_U  = Float64[]
    wages_S0 = Float64[];  mass_S0 = Float64[]
    wages_S1 = Float64[];  mass_S1 = Float64[]

    # Unskilled: the wage w_U(aU,p) = β_U P_U aU p + outside is linear in p, and
    # eU_surface[i,·] is a genuine density over p (frontier spike at p=1 plus the
    # damaged tail), so each (aU,p) cell contributes its own wage and mass.
    for i in 1:Nx, jp in 1:NpU
        w = wU_surface[i, jp];  e = eU_surface[i, jp]
        (!isnan(w) && e > 1e-16) && (push!(wages_U, w);  push!(mass_U, e))
    end
    # Skilled: employed density eS_pS[j,·] is per-aS mass, split between the poached-worker
    # surface wS1 and the non-searching surface wS0 by the OJS covered fraction s_j — the
    # same weight the stationary solve applies to the poaching margin (skilled.jl).  A cell
    # straddling p^oj contributes to BOTH surfaces in proportion, so every wage moment is a
    # continuous function of the cutoff; the hard 1{p < p^oj} this replaces moved a whole
    # node's mass across at once.  Both surfaces are written under the same p*_S test above,
    # so their NaN patterns coincide and the split conserves the cell's mass exactly.
    for j in 1:Nx
        poj_j = clamp01(poj[j])
        for jp in 1:NpS
            e = eS_pS[j, jp];  e <= 1e-16 && continue
            m   = e * wpS[jp]
            s_j = _soft_oj_weight(pg[jp], poj_j, pg, jp, NpS)
            if s_j > 0.0
                w = wS1_surface[j, jp]
                !isnan(w) && (push!(wages_S1, w);  push!(mass_S1, s_j * m))
            end
            if s_j < 1.0
                w = wS0_surface[j, jp]
                !isnan(w) && (push!(wages_S0, w);  push!(mass_S0, (1.0 - s_j) * m))
            end
        end
    end

    all_wages = [wages_U; wages_S0; wages_S1]
    Nbins = 120
    if isempty(all_wages)
        wgrid = range(0.0, 1.0; length = Nbins + 1)
        wmid  = collect(wgrid[1:end-1] .+ step(wgrid) / 2)
        dens_U = zeros(Nbins);  dens_S0 = zeros(Nbins);  dens_S1 = zeros(Nbins);  dens_S = zeros(Nbins)
    else
        w_lo = quantile(all_wages, 0.002);  w_hi = quantile(all_wages, 0.998)
        wgrid = range(w_lo, w_hi; length = Nbins + 1)
        wmid  = collect(wgrid[1:end-1] .+ step(wgrid) / 2)
        dens_U  = _wage_density(wages_U,  mass_U,  wgrid)
        dens_S0 = _wage_density(wages_S0, mass_S0, wgrid)
        dens_S1 = _wage_density(wages_S1, mass_S1, wgrid)
        dens_S  = _wage_density([wages_S0; wages_S1], [mass_S0; mass_S1], wgrid)
    end

    # ── Population accounting (all masses already carry W2) ──────────────
    agg_uU = sum(uU);  agg_t = sum(tU);  agg_eU = sum(eU_mat);  agg_mU = sum(mU_mat)
    agg_uS = sum(uS_mat)
    agg_eS = sum(eS_totS[j] for j in 1:Nx)      # eS_pS carries mcol0 (2D mass) already
    agg_mS = agg_uS + agg_eS
    agg_mS_flow = sum(mS_mat)

    total_pop = agg_mU + agg_mS
    lf_U      = agg_uU + agg_eU
    lf_total  = lf_U + agg_mS
    ur_U      = lf_U     > 1e-12 ? agg_uU / lf_U   : 1.0
    ur_S      = agg_mS   > 1e-12 ? agg_uS / agg_mS : 1.0
    ur_total  = lf_total > 1e-12 ? (agg_uU + agg_uS) / lf_total : 1.0

    # ── Transition rates ──────────────────────────────────────────────────
    # Reservation-cutoff CDFs: the OFFER CDF governs job-finding/acceptance
    # (fresh meetings), the SHOCK CDF governs endogenous separation (a λ_S
    # redraw landing below p*_S).  They coincide at δ = 1.
    #
    # Both integrate the cell masses at the cutoff — the convention
    # solve_stationary_skilled! uses — so a rate moment is read off the same
    # margin as the stationary distribution it is weighted by.  Where p*_S = 0
    # the endogenous skilled margin is dead and both CDFs are exactly 0.
    Γo_pstarS = [cdf_at_cutoff(pre.γvals,   pg, wpS, clamp01(pstar_S[j])) for j in 1:Nx]
    Γs_pstarS = [cdf_at_cutoff(pre.γs_vals, pg, wpS, clamp01(pstar_S[j])) for j in 1:Nx]

    # Skilled job-finding, averaged over the skilled-unemployed composition:
    # stayers accept S-offers at κ_S(1−Γ_o(p*_S)); crossers accept U-offers at f_U.
    # Both masses come from the split above, so the composition stays a partition of
    # uS_mat at a fractional drain (re-deriving one of them as d·uS_mat would double
    # the drain and break the identity).
    accept_S = 1.0 .- Γo_pstarS
    uS_d1mass  = [sum(uS_drain[:, j]) for j in 1:Nx]               # by aS
    uS_d0mass  = [sum(uS_stay[:, j])  for j in 1:Nx]
    uS_colmass = uS_d0mass .+ uS_d1mass
    uS_mass    = sum(uS_colmass)
    if uS_mass > 1e-14
        hire_S = κS * dot(accept_S, uS_d0mass)
        hire_X = f_U * sum(uS_d1mass)
        f_S = (hire_S + hire_X) / uS_mass
    else
        f_S = κS
    end

    # Unskilled separation hazard δ_U(aU) = ξ_U + λ_U p*(aU)^{α_U}, employed-weighted.
    # ξ_U is the exogenous baseline; the endogenous part reads the SHOCK CDF
    # G(p*) = p*^{α_U} (a match separates when a λ_U redraw lands below p*).
    δU_by_a    = ξU .+ λU .* clamp01.(pstar_U) .^ αU
    sep_rate_U = dot(δU_by_a, eU_rowU) / max(agg_eU, 1e-14)

    # Skilled separation hazard δ_S(aS) = ξ_S + λ_S Γ_s(p*_S), employed-weighted.
    # The endogenous part reads the SHOCK CDF: a match separates when a λ_S
    # redraw lands below the reservation cutoff, and redraws are shock draws.
    δS_by_a    = ξS .+ λS .* Γs_pstarS
    sep_rate_S = dot(δS_by_a, eS_totS) / max(agg_eS, 1e-14)

    # Within-job wage-change hazard: a λ_j redraw landing AT OR ABOVE the
    # reservation cutoff survives the match but re-prices it, at rate
    # λ_j·(1 − G_j(p*)).  ξ_j never re-prices a survivor, so wchg_rate loads on
    # λ_j blind to ξ_j — the moment that separates the two (SIPP; data-and-
    # moments §sipp).  Same employed weights and cutoff CDFs as sep_rate above.
    wchg_by_aU  = λU .* (1.0 .- clamp01.(pstar_U) .^ αU)
    wchg_rate_U = dot(wchg_by_aU, eU_rowU) / max(agg_eU, 1e-14)
    wchg_by_aS  = λS .* (1.0 .- Γs_pstarS)
    wchg_rate_S = dot(wchg_by_aS, eS_totS) / max(agg_eS, 1e-14)

    # Skilled E-to-E rate and EE-move wage step. Each cell's OJS-searching fraction
    # s(p) poaches at κ_S(1−Γ_o(p)); ee_rate_S is that flow over the employed-skilled
    # mass.
    # ee_step_S is the mass-weighted mean LOG-wage jump of the very same poaching
    # flow (identifies β_S — the proportional step is a fraction of the surplus
    # gain, LMR/CPVR). A mover at (aS, p) sits on the poached-worker surface wS1
    # at its current p; the poach delivers a destination match quality p′ from
    # the offer distribution, accepted only when it improves the match (p′ ≥ p),
    # so the expected post-move LEVEL wage is the offer-weighted (wΓ) mean of wS1
    # over the destinations p′ ≥ p on that same surface. The step is log of that
    # mean level minus log of the pre-move wage — the log taken after the level
    # mean, matching the SIPP construction (which logs the realized new wage).
    # Both accumulated in one pass over the moving mass; empty moving mass ⇒ 0.0
    # (held out like the other SIPP moments).
    ee_mass = 0.0;  ee_flow = 0.0
    ee_step_flow = 0.0;  ee_step_massw = 0.0
    for j in 1:Nx
        poj_j = clamp01(poj[j])

        # Reverse-cumulative offer-weighted wS1 wage from each rung upward, so the
        # expected post-move wage at floor jp is post_sw[jp] / post_w[jp] (offer-
        # distribution mean over destinations p′ ≥ pg[jp]). NaN surface entries
        # (below p*, no valid wage) carry zero weight.
        post_sw = zeros(NpS);  post_w = zeros(NpS)
        acc_sw = 0.0;  acc_w = 0.0
        for jp in NpS:-1:1
            w_d = wS1_surface[j, jp]
            if !isnan(w_d)
                acc_sw += wΓ[jp] * w_d;  acc_w += wΓ[jp]
            end
            post_sw[jp] = acc_sw;  post_w[jp] = acc_w
        end

        for jp in 1:NpS
            e_ij = eS_pS[j, jp] * wpS[jp]
            ee_mass += e_ij
            # p^oj is the PARTICIPATION margin — whether this quality searches on the
            # job at all — and s_j is its covered fraction on the straddling cell.  The
            # MOVE requires an offer above her own quality, so the flow reads
            # κ_S(1−Γ_o(p_j)) at her own node, where the offer CDF is tabulated exactly
            # and where solve_stationary_skilled! applies the same hazard to the same
            # mass.  Reading it at the cutoff instead charged a worker deep below p^oj
            # the marginal searcher's moving rate.
            s_j = _soft_oj_weight(pg[jp], poj_j, pg, jp, NpS)
            s_j <= 0.0 && continue
            flow_ij = s_j * e_ij * κS * (1.0 - pre.Γvals[jp])
            ee_flow += flow_ij
            w_pre = wS1_surface[j, jp]
            if !isnan(w_pre) && w_pre > 0.0 && post_w[jp] > 0.0
                w_post = post_sw[jp] / post_w[jp]
                if w_post > 0.0
                    ee_step_flow  += flow_ij * (log(w_post) - log(w_pre))
                    ee_step_massw += flow_ij
                end
            end
        end
    end
    ee_rate_S = ee_mass > 1e-14 ? ee_flow / ee_mass : 0.0
    ee_step_S = ee_step_massw > 1e-14 ? ee_step_flow / ee_step_massw : 0.0

    # ── Unemployment-duration survivors ───────────────────────────────────
    # Survival of an ONGOING unemployment spell past t months, per market, as a
    # mixture of exponential exit hazards over that market's unemployed stock.
    #
    # Skilled.  The exit hazard must match the one used for f_S above: a stayer
    # leaves skilled unemployment by accepting a fresh skilled OFFER at
    # κ_S(1−Γ_o(p*_S)), while a crosser leaves by taking an UNSKILLED job at f_U.
    # Both also exit demographically at ν.  Applying the stayer hazard to the whole
    # stock understated the crossers' exit rate and overstated their survival.
    δS_unemp0 = κS .* (1.0 .- Γo_pstarS) .+ ν      # stayer: fresh OFFER acceptance
    δS_unemp1 = f_U + ν                            # crosser: an UNSKILLED job
    _surv_S = t -> uS_mass > 1e-14 ?
        (dot(uS_d0mass, exp.(-δS_unemp0 .* t)) +
         sum(uS_d1mass) * exp(-δS_unemp1 * t)) / uS_mass : 0.0

    # Unskilled.  A searcher leaves at f_U — the same f_hire that
    # solve_stationary_unskilled! applies — while a worker at the participation
    # corner (p*_U = 1) is never hired and leaves only at ν.  That corner is the
    # model's own discouragement margin, so the unskilled survivor carries a
    # floor at the non-searching mass share, which the skilled one does not.
    uU_by_a  = vec(sum(uU, dims = 2))
    uU_mass  = sum(uU_by_a)
    δU_unemp = [clamp01(pstar_U[i]) < 1.0 - 1e-10 ? f_U + ν : ν
                for i in eachindex(uU_by_a)]
    _surv_U = t -> uU_mass > 1e-14 ?
        dot(uU_by_a, exp.(-δU_unemp .* t)) / uU_mass : 0.0

    usurv_S = [_surv_S(wk * _WEEKS_TO_MONTHS) for wk in DURATION_THRESHOLDS_WK]
    usurv_U = [_surv_U(wk * _WEEKS_TO_MONTHS) for wk in DURATION_THRESHOLDS_WK]
    # Kept as its own field and evaluated through the same function, so it stays
    # correct even if 27 weeks is later dropped from the threshold list.
    ltu_share_S = _surv_S(27.0 * _WEEKS_TO_MONTHS)

    return (
        xg = xg, Fell = Fell, pgU = pgU, pg = pg, waU = waU, waS = waS, W2 = W2,
        wpU = wpU, wpS = wpS, Nx = Nx, NpU = NpU, NpS = NpS,

        uU = uU, tU = tU, eU_mat = eU_mat, mU_mat = mU_mat, mS_mat = mS_mat,
        uS_mat = uS_mat, eS_pS = eS_pS, eS_totS = eS_totS,
        eU_surface = eU_surface, eU_rowU = eU_rowU,

        Usearch = Usearch, Tv = Tv, net_T = net_T, τ_mat = τ_mat,
        JU_frontier = JU_frontier, SU1 = SU1, EU1 = EU1, pstar_U = pstar_U,
        SU_surface = SU_surface, JU_surface = JU_surface, EU_surface = EU_surface,

        US = US, pstar_S = pstar_S, poj = poj, d = d_mat,
        S0_surface = S0_surface, S1_surface = S1_surface, Smax_surface = Smax_surface,
        E0_surface = copy(sc.E0), E1_surface = copy(sc.E1),
        J0_surface = copy(sc.J0), J1_surface = copy(sc.J1),

        wU_surface = wU_surface, wS0_surface = wS0_surface, wS1_surface = wS1_surface,
        wmid = wmid, dens_U = dens_U, dens_S = dens_S, dens_S0 = dens_S0, dens_S1 = dens_S1,

        agg_uU = agg_uU, agg_t = agg_t, agg_eU = agg_eU, agg_mU = agg_mU,
        agg_uS = agg_uS, agg_eS = agg_eS, agg_mS = agg_mS, agg_mS_flow = agg_mS_flow,
        total_pop = total_pop, ur_U = ur_U, ur_S = ur_S, ur_total = ur_total,
        thetaU = θU, thetaS = θS,

        f_U = f_U, f_S = f_S, sep_rate_U = sep_rate_U, sep_rate_S = sep_rate_S,
        ee_rate_S = ee_rate_S, ee_step_S = ee_step_S,
        wchg_rate_U = wchg_rate_U, wchg_rate_S = wchg_rate_S,
        ltu_share_S = ltu_share_S,
        # Survivors at DURATION_THRESHOLDS_WK, indexed in the same order.
        # Additive: ltu_share_S above is unchanged, so existing consumers are unaffected.
        usurv_S = usurv_S, usurv_U = usurv_U,
        σ_wU = up.σ_w, σ_wS = sp.σ_w,
    )
end


"""
    print_accounting(obj)

Formatted population-accounting table from `compute_equilibrium_objects`.
"""
function print_accounting(obj)
    @printf("\n┌─────────────────────────────────────────────────────┐\n")
    @printf("│  Equilibrium accounting                               │\n")
    @printf("├─────────────────────────────────────────────────────┤\n")
    @printf("│  UNSKILLED  (pop share %6.4f)                        │\n", obj.agg_mU)
    @printf("│    u_U %6.4f   e_U %6.4f   t %6.4f   ur_U %5.1f%%  │\n", obj.agg_uU, obj.agg_eU, obj.agg_t, 100 * obj.ur_U)
    @printf("│  SKILLED    (pop share %6.4f)                        │\n", obj.agg_mS)
    @printf("│    u_S %6.4f   e_S %6.4f            ur_S %5.1f%%  │\n", obj.agg_uS, obj.agg_eS, 100 * obj.ur_S)
    @printf("│  TOTAL POP %6.4f   global u-rate %6.4f             │\n", obj.total_pop, obj.ur_total)
    @printf("│  θ_U %8.4f   θ_S %8.4f                          │\n", obj.thetaU, obj.thetaS)
    @printf("└─────────────────────────────────────────────────────┘\n")
end

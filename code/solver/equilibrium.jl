############################################################
# equilibrium.jl — Post-solution equilibrium objects
#
# compute_equilibrium_objects(model)
#   Returns a NamedTuple with all stationary densities,
#   value/surplus/firm surfaces, wages, and population accounting.
#
# print_accounting(obj)
#   Prints a formatted table of population accounting.
############################################################

"""
    compute_equilibrium_objects(model) → NamedTuple

Derive all equilibrium objects from a solved `Model`:
  - stationary densities (u_U, t, e_U, u_S, e_S, …)
  - value and surplus surfaces (S_U, J_U, E_U, S0/S1_S, …)
  - wage surfaces and densities
  - population accounting aggregates
"""
function compute_equilibrium_objects(model::Model)

    cp  = model.common
    rp  = model.regime
    uc  = model.unsk_cache
    sc  = model.skl_cache
    up  = model.unsk_par
    sp  = model.skl_par
    pre = model.skl_pre

    xg  = model.grids.x
    wx  = model.grids.wx
    ell = model.grids.ℓ
    pgU = model.unsk_grids.p
    wpU = model.unsk_grids.wp
    pg  = model.skl_grids.p
    wpS = model.skl_grids.wp

    Nx  = length(xg)
    NpU = length(pgU)
    NpS = length(pg)

    r  = cp.r;  ν  = cp.ν;  φ  = cp.φ
    βU = up.β;  λU = up.λ
    βS = sp.β;  λS = sp.λ;  σS = sp.σ

    PU = rp.PU;  γPS = rp.gamma_PS;  bS = rp.bS;  αU = rp.α_U

    c_of_x = x -> training_cost(x, cp.c)

    # ── dG quadrature weights ──────────────────────────────────────────────
    wGU = build_unskilled_G_weights(pgU, wpU, αU)

    # ── Stationary densities ───────────────────────────────────────────────
    uU     = copy(uc.u)
    tU     = copy(uc.t)
    mU_x   = max.(ell .- (φ / ν) .* tU, 0.0)
    eU_vec = max.(mU_x .- uU .- tU, 0.0)
    mS_vec = [(φ / ν) * tU[ix] for ix in 1:Nx]

    uS     = copy(sc.u)
    eS_mat = copy(sc.e)
    eS_tot = [dot(eS_mat[ix, :], wpS) for ix in 1:Nx]

    # ── Unskilled employment surface (Nx × NpU) ───────────────────────────
    #   Uses the same _soft_weight approach as the skilled block so that
    #   the surface is a continuous function of pstar — eliminating the
    #   grid-snapping oscillation that caused the wage density to wiggle.
    θU  = uc.θ
    f_U = θU * q_from_theta(θU, up.μ, up.η)

    eU_surface = zeros(Nx, NpU)
    for ix in 1:Nx
        pstar_x    = clamp01(uc.pstar[ix])
        eU_total_x = eU_vec[ix]
        denom_eU   = ν + λU

        for jp in 1:NpU
            p   = pgU[jp]
            ω_j = _soft_weight(p, pstar_x, pgU, jp, NpU)
            if ω_j > 0.0
                g_p = (p <= 0.0) ? 0.0 : αU * p^(αU - 1.0)
                eU_surface[ix, jp] = ω_j * λU * g_p / denom_eU * eU_total_x
            end
        end
        if eU_vec[ix] > 0.0
            eU_surface[ix, end] += f_U * uU[ix] / (ν + λU)
        end
    end

    # ── Zero-employment guard (consistent for both segments) ──────────────
    #   When aggregate employment is negligible, force the surfaces to
    #   exact zeros so that downstream objects (wages, densities, moments)
    #   see a clean corner solution rather than normalised numerical noise.
    _emp_tol = 1e-12
    agg_eU_raw = dot(eU_vec, wx)
    agg_eS_raw = sum(eS_mat .* wpS' .* reshape(wx, :, 1))

    if agg_eU_raw < _emp_tol
        fill!(eU_vec,    0.0)
        fill!(eU_surface, 0.0)
    end
    if agg_eS_raw < _emp_tol
        fill!(eS_mat, 0.0)
        fill!(eS_tot, 0.0)
    end

    # ── Unskilled values and policy ────────────────────────────────────────
    Usearch     = copy(uc.Usearch)
    Tv          = copy(uc.T)
    UU          = copy(uc.U)
    tauT        = copy(uc.τT)
    net_T       = [-c_of_x(xg[ix]) + Tv[ix] for ix in 1:Nx]
    JU_frontier = copy(uc.Jfrontier)
    SU1         = JU_frontier ./ max(1.0 - βU, 1e-14)
    EU1         = UU .+ βU .* SU1
    pstar_U     = copy(uc.pstar)

    # ── Unskilled surplus / firm / worker surfaces ─────────────────────────
    SU_surface = zeros(Nx, NpU)
    for ix in 1:Nx
        Svec = zeros(NpU)
        solve_unskilled_surplus_on_grid!(
            Svec, pgU, wGU, PU, xg[ix], r, ν, λU, UU[ix], uc.pstar[ix]
        )
        SU_surface[ix, :] = Svec
    end
    JU_surface = (1.0 - βU) .* SU_surface
    EU_surface = UU .+ βU .* SU_surface

    # ── Skilled value / surplus surfaces ──────────────────────────────────
    denom_nb     = max(1.0 - βS, 1e-14)
    S0_surface   = sc.J0 ./ denom_nb
    S1_surface   = sc.J1 ./ denom_nb
    E0_surface   = copy(sc.E0)
    E1_surface   = copy(sc.E1)
    J0_surface   = copy(sc.J0)
    J1_surface   = copy(sc.J1)
    Smax_surface = max.(S0_surface, S1_surface)
    Jskl_surface = ifelse.(E1_surface .>= E0_surface, J1_surface, J0_surface)

    pstar_S = copy(sc.pstar)
    poj     = copy(sc.poj)
    US      = copy(sc.U)

    # ── Total employment surface on skilled p-grid ─────────────────────────
    eU_on_pg = zeros(Nx, NpS)
    for ix in 1:Nx
        itp = linear_interpolation(pgU, eU_surface[ix, :], extrapolation_bc = 0.0)
        eU_on_pg[ix, :] = max.(itp.(pg), 0.0)
    end

    eS_surface = eS_mat
    e_total_surface = eU_on_pg .+ eS_mat

    # ── Wages ──────────────────────────────────────────────────────────────
    θS = sc.θ
    κS = θS * q_from_theta(θS, sp.μ, sp.η)
    wΓ = pre.γvals .* wpS

    tailS = zeros(Nx, NpS)
    for ix in 1:Nx
        acc = 0.0
        for jp in NpS:-1:1
            acc += Smax_surface[ix, jp] * wΓ[jp]
            tailS[ix, jp] = acc
        end
    end

    I_full = zeros(Nx)
    for ix in 1:Nx
        j0         = pcut_index(pg, clamp01(pstar_S[ix]))
        I_full[ix] = tailS[ix, j0]
    end

    # w_U(x,p)  — soft-weighted to match the employment surface
    wU_surface = fill(NaN, Nx, NpU)
    for ix in 1:Nx
        pst     = clamp01(pstar_U[ix])
        outside = (1.0 - βU) * (r + ν) * UU[ix]
        for jp in 1:NpU
            ω_j = _soft_weight(pgU[jp], pst, pgU, jp, NpU)
            if ω_j > 0.0
                wU_surface[ix, jp] = βU * PU * xg[ix] * pgU[jp] + outside
            end
        end
    end

    # w_S^0(x,p) — soft-weighted
    wS0_surface = fill(NaN, Nx, NpS)
    for ix in 1:Nx
        pst         = clamp01(pstar_S[ix])
        PS_x        = PS_of_x(xg[ix], γPS)
        flow_out    = (1.0 - βS) * bS
        ladder_term = βS * (1.0 - βS) * κS * I_full[ix]
        for jp in 1:NpS
            ω_j = _soft_weight(pg[jp], pst, pg, jp, NpS)
            if ω_j > 0.0
                wS0_surface[ix, jp] = βS * PS_x * xg[ix] * pg[jp] + flow_out + ladder_term
            end
        end
    end

    # w_S^1(x,p) — soft-weighted
    wS1_surface = fill(NaN, Nx, NpS)
    for ix in 1:Nx
        pst      = clamp01(pstar_S[ix])
        PS_x     = PS_of_x(xg[ix], γPS)
        flow_out = (1.0 - βS) * (bS + σS)
        for jp in 1:NpS
            ω_j = _soft_weight(pg[jp], pst, pg, jp, NpS)
            if ω_j > 0.0
                I_low = max(I_full[ix] - tailS[ix, jp], 0.0)
                wS1_surface[ix, jp] =
                    βS * PS_x * xg[ix] * pg[jp] +
                    flow_out +
                    βS * (1.0 - βS) * κS * I_low
            end
        end
    end

    Δw_surface = wS1_surface .- wS0_surface

    # ── Wage densities ─────────────────────────────────────────────────────
    function _wage_density(wages, weights, wgrid; mass_tol = 1e-12)
        Nb   = length(wgrid) - 1
        bw   = step(wgrid)
        dens = zeros(Nb)
        for (w, m) in zip(wages, weights)
            # fractional bin position (0-indexed from left edge of bin 1)
            raw  = (w - first(wgrid)) / bw
            j_lo = clamp(floor(Int, raw) + 1, 1, Nb)
            j_hi = clamp(j_lo + 1,            1, Nb)
            α    = clamp(raw - (j_lo - 1), 0.0, 1.0)   # weight on j_hi
            dens[j_lo] += (1.0 - α) * m
            dens[j_hi] +=        α  * m
        end
        total = sum(dens) * bw
        # Guard: if total employment mass is negligible, return zeros
        # instead of normalising numerical noise into a fake density.
        return total > mass_tol ? dens ./ total : dens
    end

    wages_U  = Float64[];  mass_U  = Float64[]
    wages_S0 = Float64[];  mass_S0 = Float64[]
    wages_S1 = Float64[];  mass_S1 = Float64[]

    for ix in 1:Nx
        wx_ix  = wx[ix]
        poj_ix = clamp01(poj[ix])

        for jp in 1:NpU
            w = wU_surface[ix, jp]
            e = eU_surface[ix, jp]
            if !isnan(w) && e > 1e-16
                push!(wages_U, w)
                push!(mass_U, e * wx_ix * wpU[jp])
            end
        end

        for jp in 1:NpS
            e = eS_mat[ix, jp]
            e <= 1e-16 && continue
            m = e * wx_ix * wpS[jp]
            if pg[jp] < poj_ix
                w = wS1_surface[ix, jp]
                if !isnan(w);  push!(wages_S1, w);  push!(mass_S1, m)  end
            else
                w = wS0_surface[ix, jp]
                if !isnan(w);  push!(wages_S0, w);  push!(mass_S0, m)  end
            end
        end
    end

    all_wages = [wages_U; wages_S0; wages_S1]
    Nbins     = 120
    if isempty(all_wages)
        # No employed workers — return empty densities on a dummy grid
        wgrid = range(0.0, 1.0; length = Nbins + 1)
        wmid  = collect(wgrid[1:end-1] .+ step(wgrid) / 2)
        dens_U  = zeros(Nbins)
        dens_S0 = zeros(Nbins)
        dens_S1 = zeros(Nbins)
        dens_S  = zeros(Nbins)
    else
        w_lo  = quantile(all_wages, 0.002)
        w_hi  = quantile(all_wages, 0.998)
        wgrid = range(w_lo, w_hi; length = Nbins + 1)
        wmid  = collect(wgrid[1:end-1] .+ step(wgrid) / 2)

        dens_U  = _wage_density(wages_U,  mass_U,  wgrid)
        dens_S0 = _wage_density(wages_S0, mass_S0, wgrid)
        dens_S1 = _wage_density(wages_S1, mass_S1, wgrid)
        dens_S  = _wage_density([wages_S0; wages_S1], [mass_S0; mass_S1], wgrid)
    end

    # ── Population accounting ──────────────────────────────────────────────
    agg_uU      = dot(uU, wx)
    agg_t       = dot(tU, wx)
    agg_eU      = dot(eU_vec, wx)
    agg_mU      = dot(mU_x, wx)

    agg_uS      = dot(uS, wx)
    agg_eS      = sum(eS_mat .* wpS' .* reshape(wx, :, 1))
    agg_mS      = agg_uS + agg_eS
    agg_mS_flow = (φ / ν) * agg_t

    total_pop   = agg_mU + agg_mS
    lf_U        = agg_uU + agg_eU                           # unskilled LF (excl. training)
    lf_total    = lf_U + agg_mS                              # total LF (excl. training)
    ur_U        = lf_U     > 1e-12 ? agg_uU / lf_U     : 1.0
    ur_S        = agg_mS   > 1e-12 ? agg_uS / agg_mS   : 1.0
    ur_total    = lf_total > 1e-12 ? (agg_uU + agg_uS) / lf_total : 1.0

    # ── Transition rates for model_moments ────────────────────────────────
    # ΓpstarS used by both f_S and sep_rate_S — compute once here.
    ΓpstarS    = [pre.Γvals[pcut_index(pg, clamp01(pstar_S[ix]))] for ix in 1:Nx]

    # f_S: skilled job-finding rate = κ_S · E[1 − Γ(p*_S(x)) | u_S]
    #
    # κ_S = θ_S · q_S(θ_S) is the contact rate for any job seeker.
    # In the skilled market a contact draws p̃ ~ Γ_z; the unemployed worker
    # accepts only if p̃ ≥ p*_S(x), so the per-type acceptance probability is
    # (1 − Γ_z(p*_S(x))).  The data moment Pr(E_{t+1}|U_t,S) is this
    # acceptance-weighted rate averaged over the unemployed composition,
    # NOT the raw contact rate κ_S.
    # (Contrast with the unskilled market where all contacts match at p=1,
    # so f_U = θ_U · q_U(θ_U) is already the correct job-finding rate.)
    accept_S = 1.0 .- ΓpstarS          # (1 − Γ(p*_S(x))) for each x
    uS_mass  = dot(uS, wx)             # ∫ u_S(x) dx
    f_S = uS_mass > 1e-14 ?
          κS * dot(accept_S, uS .* wx) / uS_mass :
          κS

    # sep_rate_U: employment-weighted unskilled destruction hazard
    #   δ_U(x) = λ_U · G(p*(x)) = λ_U · p*(x)^α_U   [G(p) = p^α_U]
    δU_by_x    = λU .* clamp01.(pstar_U) .^ αU
    sep_rate_U = dot(δU_by_x, eU_vec .* wx) / max(agg_eU, 1e-14)

    # sep_rate_S: employment-weighted skilled separation hazard into unemployment
    #   δ_S(x) = ξ_S + λ_S · Γ(p*_S(x))
    #   job-to-job quits are excluded — they do not create unemployment
    δS_by_x    = sp.ξ .+ λS .* ΓpstarS
    sep_rate_S = dot(δS_by_x, eS_tot .* wx) / max(agg_eS, 1e-14)

    # ee_rate_S: skilled employment-to-employment transition rate
    # For each employed skilled worker at (x,p):
    #   - s=0 workers (not doing OJS): do not search on the job, so EE rate = 0
    #   - s=1 workers (doing OJS, p < poj): receive offers at rate kappa_S,
    #     accept those above poj, so EE rate = kappa_S * (1 - Gamma(poj(x)))
    # We approximate: the fraction of skilled employed doing OJS is captured
    # by the mass in e_S with p < poj, and their EE rate is kappa_S * (1 - Gamma(poj(x))).
    ee_mass = 0.0
    ee_flow = 0.0
    for ix in 1:Nx
        poj_ix   = clamp01(poj[ix])
        j_poj    = pcut_index(pg, poj_ix)
        # Gamma(poj) = CDF of offer distribution at poj
        Gamma_poj = pre.Γvals[j_poj]
        for jp in 1:NpS
            e_ij = eS_mat[ix, jp] * wx[ix] * wpS[jp]
            if pg[jp] < poj_ix
                # This worker does OJS: EE flow rate = kappa_S * (1 - Gamma(poj))
                ee_flow += e_ij * κS * (1.0 - Gamma_poj)
            end
            ee_mass += e_ij
        end
    end
    ee_rate_S = ee_mass > 1e-14 ? ee_flow / ee_mass : 0.0

    return (
        # grids
        xg = xg, pgU = pgU, pg = pg, wx = wx, wpU = wpU, wpS = wpS,
        Nx = Nx, NpU = NpU, NpS = NpS,

        # stationary densities
        uU = uU, tU = tU, eU_vec = eU_vec, mU_x = mU_x, mS_vec = mS_vec,
        uS = uS, eS_mat = eS_mat, eS_tot = eS_tot,

        # employment surfaces
        eU_surface      = eU_surface,
        eS_surface      = eS_surface,
        e_total_surface = e_total_surface,

        # unskilled values & policy
        Usearch     = Usearch, Tv = Tv, UU = UU, net_T = net_T, tauT = tauT,
        JU_frontier = JU_frontier, SU1 = SU1, EU1 = EU1,
        pstar_U     = pstar_U,

        # unskilled value surfaces
        SU_surface = SU_surface, JU_surface = JU_surface, EU_surface = EU_surface,

        # skilled values & policy
        US = US, pstar_S = pstar_S, poj = poj,

        # skilled value surfaces
        S0_surface   = S0_surface,   S1_surface  = S1_surface,
        Smax_surface = Smax_surface,
        E0_surface   = E0_surface,   E1_surface  = E1_surface,
        J0_surface   = J0_surface,   J1_surface  = J1_surface,
        Jskl_surface = Jskl_surface,

        # wages
        wU_surface  = wU_surface,
        wS0_surface = wS0_surface,   wS1_surface = wS1_surface,
        Δw_surface  = Δw_surface,

        # wage densities
        wmid    = wmid,   dens_U  = dens_U,
        dens_S  = dens_S, dens_S0 = dens_S0, dens_S1 = dens_S1,

        # population accounting
        agg_uU  = agg_uU, agg_t   = agg_t,   agg_eU  = agg_eU, agg_mU = agg_mU,
        agg_uS  = agg_uS, agg_eS  = agg_eS,  agg_mS  = agg_mS,
        agg_mS_flow = agg_mS_flow, total_pop = total_pop,
        ur_U    = ur_U,   ur_S    = ur_S,     ur_total = ur_total,
        thetaU  = begin
            _uU = dot(uc.u, wx)
            _JU = dot(uc.Jfrontier .* uc.u, wx)
            (_JU > 1e-14 && _uU > 1e-14) ?
                max(theta_from_q(up.k * _uU / _JU, up.μ, up.η), 1e-14) : 1e-14
        end,
        thetaS  = begin
            _JS = compute_Jbar_skilled(model)
            _JS > 1e-12 ?
                max(theta_from_q(sp.k / _JS, sp.μ, sp.η), 1e-14) : 1e-14
        end,

        # job-finding and separation rates
        f_U        = f_U,
        f_S        = f_S,
        sep_rate_U = sep_rate_U,
        sep_rate_S = sep_rate_S,
        ee_rate_S  = ee_rate_S,
    )
end


"""
    print_accounting(obj)

Print a formatted population accounting table from the output of
`compute_equilibrium_objects`.
"""
function print_accounting(obj)
    @printf("\n╔═════════════════════════════════════════════════════╗\n")
    @printf("║  Equilibrium accounting                             ║\n")
    @printf("╠═════════════════════════════════════════════════════╣\n")
    @printf("║  UNSKILLED SEGMENT  (share of pop = %6.4f)         ║\n", obj.agg_mU)
    @printf("║    unemployed  u_U   %6.4f  (%5.1f%% of LF_U)       ║\n", obj.agg_uU, 100 * obj.ur_U)
    @printf("║    employed    e_U   %6.4f                         ║\n", obj.agg_eU)
    @printf("║    training    t     %6.4f                         ║\n", obj.agg_t)
    @printf("║    m_U (flow)        %6.4f                         ║\n", obj.agg_mU)
    @printf("╠═════════════════════════════════════════════════════╣\n")
    @printf("║  SKILLED SEGMENT    (share of pop = %6.4f)         ║\n", obj.agg_mS)
    @printf("║    unemployed  u_S   %6.4f  (%5.1f%% of seg)        ║\n", obj.agg_uS, 100 * obj.ur_S)
    @printf("║    employed    e_S   %6.4f                         ║\n", obj.agg_eS)
    @printf("║    m_S (flow)        %6.4f                         ║\n", obj.agg_mS_flow)
    @printf("╠═════════════════════════════════════════════════════╣\n")
    @printf("║  TOTAL POPULATION    %6.4f  (should be 1.000)      ║\n", obj.total_pop)
    @printf("║  LF (excl training)  %6.4f                         ║\n", obj.agg_uU + obj.agg_eU + obj.agg_mS)
    @printf("║  Global u-rate (LF)  %6.4f                         ║\n", obj.ur_total)
    @printf("║  θ_U = %6.4f        θ_S = %6.4f                   ║\n", obj.thetaU, obj.thetaS)
    @printf("╚═════════════════════════════════════════════════════╝\n")
end
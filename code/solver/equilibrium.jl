############################################################
# equilibrium.jl — Equilibrium objects of a solved model
#
# compute_equilibrium_objects(model) returns every object the moments, plots and
# transitions read: stationary densities on the (aU, aS) copula grid, value, surplus and
# wage surfaces (1D in own ability × p, except the training slice, which reads the whole
# type), wage points, population accounting, transition hazards, the rate moments as the
# data measure them, and unemployment-duration survivors.
#
# Entry points: compute_equilibrium_objects, print_accounting.
############################################################

# Unemployment-duration thresholds in weeks, one week above each CPS duration heap
# (4/13/26/52). Mirrors DURATION_THRESHOLDS_WK in data_and_descriptives/setup.jl, the
# canonical copy; moments.jl derives its moment names from this constant.
const DURATION_THRESHOLDS_WK = [5.0, 14.0, 27.0, 53.0]

const _WEEKS_TO_MONTHS = 12.0 / 52.0
const _MONTHS_PER_QUARTER = 3.0

"""
    _hazard_of_mean_probability(prob, mass, empty) → Float64

A CPS or SIPP rate moment as the data measure it: the mean monthly transition probability
over the universe (cell probabilities `prob` weighted by `mass`), converted to a hazard by
`−log(1 − p̄)`. Returns `empty` when the universe holds no one.
"""
function _hazard_of_mean_probability(prob::AbstractVector{Float64}, mass::AbstractVector{Float64},
                                     empty::Float64)
    M = sum(mass)
    M > 1e-14 || return empty
    return -log1p(-dot(prob, mass) / M)
end

"""
    compute_equilibrium_objects(model) → NamedTuple

Equilibrium objects of the solved `model`. Mass arrays are the true masses (zero for a
market that holds no one); within-market objects (rates, wage points, survivors) are ratios
over the market's composition, which for an empty market is its limit composition
(`marginal_training_cells`). Rates are returned as hazards (`κ_U`, `f_S`, `sep_rate_*`,
`ee_rate_S`, `wchg_rate_*`) and, under `measured`, as the data measure each rate moment.
`usurv_S`, `usurv_U` are survivors at `DURATION_THRESHOLDS_WK`.
"""
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
    θU  = uc.θ;  κ_U = meeting_rate(θU, up.μ, up.η)

    # ── Stationary densities (2D on the copula grid) ─────────────────────
    uU     = copy(uc.u)                                # untrained unskilled unemployed
    nU     = copy(uc.n)                                # in training
    d_mat  = clamp.(copy(sc.d), 0.0, 1.0)
    cross  = crossing_rates(model, d_mat)              # unskilled hires of the trained
    mS_mat = _mS_from_n(nU, cross.k, φ, ν)             # trained mass
    mU_mat = max.(W2 .- mS_mat, 0.0)                   # untrained-segment mass
    eU_mat = max.(mU_mat .- uU .- nU, 0.0)             # unskilled employed mass, both slices
    eUt    = copy(uc.et)                               # of which the training slice

    # ── Market compositions ──────────────────────────────────────────────
    # In an open market the composition is the mass array. A market holding no one takes
    # the limit of its composition as it empties, the one its free entry prices.
    skl_open = sum(mS_mat) > EMPTY_MASS
    unk_open = sum(uU) + sum(eU_mat) > EMPTY_MASS
    mS_c = skl_open ? mS_mat :
           _mS_from_n(marginal_training_cells(model; entering = true), cross.k, φ, ν)
    if unk_open
        uU_c = uU;  eU_c = eU_mat;  uh_c = copy(uc.uh);  eUt_c = eUt
    else
        wU   = marginal_training_cells(model; entering = false)
        uU_c = similar(wU)
        solve_stationary_unskilled!(uU_c, similar(wU), wU, zeros(size(wU)), uc.pstar, κ_U, model)
        eU_c  = wU .- uU_c
        uh_c  = uU_c .* unemployment_split_unskilled(uc.pstar, κ_U, model).hired
        eUt_c = zero(wU)
    end
    # Employment below EMPTY_MASS is rounding residue of the population identity.
    sum(eU_c) < EMPTY_MASS && (fill!(eU_c, 0.0);  fill!(eUt_c, 0.0))

    # Skilled unemployed split by the drain fraction: crossers (all of d m_S, seeking in the
    # unskilled market) and stayers (û share of (1−d) m_S). Kept apart because they exit at
    # different hazards.
    uS_drain = similar(mS_mat)
    uS_stay  = similar(mS_mat)
    @inbounds for j in 1:Nx, i in 1:Nx
        uS_drain[i, j] = d_mat[i, j] * mS_c[i, j]
        uS_stay[i, j]  = (1.0 - d_mat[i, j]) * sc.u_frac[j] * mS_c[i, j]
    end
    uS_mat = uS_drain .+ uS_stay
    # Skilled employment on (aS, p), from the non-draining mass by aS.
    mcol0  = [sum((1.0 .- d_mat[:, j]) .* mS_c[:, j]) for j in 1:Nx]     # by aS
    # With no stayers the employed composition is their limit, so the employed-weighted
    # objects stay defined; the employed mass is then zero.
    stayers = sum(mcol0) > EMPTY_MASS
    mcolE   = stayers ? mcol0 : skilled_stayer_columns(model, d_mat, mS_c)
    eS_pS  = [sc.e_frac[j, jp] * mcolE[j] for j in 1:Nx, jp in 1:NpS]    # (aS, p) employed density
    eS_totS = [dot(eS_pS[j, :], wpS) for j in 1:Nx]                      # employed mass per aS
    sum(eS_pS) < EMPTY_MASS && (fill!(eS_pS, 0.0);  fill!(eS_totS, 0.0))

    # ── Unskilled employment surface e_U(aU, p) ──────────────────────────
    # The searching slice reads aU alone. Its hires (the row's hired unemployed and the
    # searching-slice crossers, rate ks) form the atom at p = 1, `eU_entry`; the damaged
    # tail is `eU_density`. The training slice `eUt` enters the wage points and rates
    # directly.
    eUs_c   = max.(eU_c .- eUt_c, 0.0)
    eU_rowU = [dot(eUs_c[i, :], ones(Nx)) for i in 1:Nx]                # Σ_j (mass, W2 already in it)
    uU_rowU = [sum(uU_c[i, :]) for i in 1:Nx]
    hires_s = [κ_U * sum(uh_c[i, :]) + dot(cross.ks[i, :], mS_c[i, :]) for i in 1:Nx]
    # Hired share of each row's unemployed, the searching slice's separated crossers included.
    hired_U = [uU_rowU[i] > 0.0 ? sum(uh_c[i, :]) / uU_rowU[i] : 0.0 for i in 1:Nx]
    eU_density = zeros(Nx, NpU)
    eU_entry   = zeros(Nx)
    for i in 1:Nx
        pstar_x = clamp01(uc.pstar[i]);  denom_eU = ν + λU
        for jp in 1:NpU
            p   = pgU[jp]
            ω_j = _soft_weight(p, pstar_x, pgU, jp, NpU)
            if ω_j > 0.0
                g_p = (p <= 0.0) ? 0.0 : αU * p^(αU - 1.0)
                eU_density[i, jp] = ω_j * λU * g_p / denom_eU * eU_rowU[i]
            end
        end
        eU_rowU[i] > 0.0 && (eU_entry[i] = hires_s[i] / (ν + λU))
    end
    # Mixed units: density in p, plus the entry atom's mass in the top column.
    eU_surface = copy(eU_density);  eU_surface[:, end] .+= eU_entry

    # ── Unskilled values and policy (1D in aU; T on the whole type) ──────
    Usearch = copy(uc.Usearch);  Tv = copy(uc.T)
    net_T   = training_value!(similar(Tv), model)                  # −c(aS) + T(aU,aS)
    pstar_t = copy(uc.pstar_t)
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
    s_part  = copy(sc.s_part)

    # ── Wages ─────────────────────────────────────────────────────────────
    θS = sc.θ;  κS = meeting_rate(θS, sp.μ, sp.η)
    wΓ = pre.γvals .* wpS
    tailS = zeros(Nx, NpS)
    for j in 1:Nx
        acc = 0.0
        for jp in NpS:-1:1
            acc += Smax_surface[j, jp] * wΓ[jp];  tailS[j, jp] = acc
        end
    end
    # I_S read one cell below the reservation node, as in skilled_inner_loop!.
    I_full = [tailS[j, max(pcut_index(pg, clamp01(pstar_S[j])) - 1, 1)] for j in 1:Nx]

    # Wages at every node (`*_node`), whatever its mass, so the histogram edges built on them
    # move continuously; the reported surfaces keep a wage only where the node has mass.
    # w_U(aU,p) = β_U A P_U aU p + (1−β_U)(r+ν) U^search(aU); a new hire is paid w_U(aU,1).
    wU_node    = zeros(Nx, NpU)
    wU_surface = fill(NaN, Nx, NpU)
    wU_entry   = zeros(Nx)
    for i in 1:Nx
        pst = clamp01(pstar_U[i]);  outside = (1.0 - βU) * (r + ν) * Usearch[i]
        for jp in 1:NpU
            wU_node[i, jp] = βU * PU * xg[i] * pgU[jp] + outside
            _soft_weight(pgU[jp], pst, pgU, jp, NpU) > 0.0 && (wU_surface[i, jp] = wU_node[i, jp])
        end
        wU_entry[i] = βU * PU * xg[i] + outside
    end

    # w^0_S(aS,p) (no OJS) and w^1_S(aS,p) (OJS).
    wS0_node    = zeros(Nx, NpS);       wS1_node    = zeros(Nx, NpS)
    wS0_surface = fill(NaN, Nx, NpS);   wS1_surface = fill(NaN, Nx, NpS)
    for j in 1:Nx
        pst = clamp01(pstar_S[j]);  PSeff = PS * xg[j]
        flow0 = (1.0 - βS) * bS + βS * (1.0 - βS) * κS * I_full[j]
        flow1 = (1.0 - βS) * (bS + σS)
        for jp in 1:NpS
            I_low = max(I_full[j] - tailS[j, jp], 0.0)
            wS0_node[j, jp] = βS * PSeff * pg[jp] + flow0
            wS1_node[j, jp] = βS * PSeff * pg[jp] + flow1 + βS * (1.0 - βS) * κS * I_low
            if _soft_weight(pg[jp], pst, pg, jp, NpS) > 0.0
                wS0_surface[j, jp] = wS0_node[j, jp]
                wS1_surface[j, jp] = wS1_node[j, jp]
            end
        end
    end
    node_wages = [vec(wU_node); wU_entry; vec(wS0_node); vec(wS1_node)]

    # Mass-weighted wage histogram on `wgrid`, linear in the bin position, normalised to a
    # density (left unnormalised below `mass_tol`).
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

    # Wage points: one (wage, mass) pair per employed cell. Unskilled searching slice: the
    # damaged tail at density × weight, the entry atom at the entry wage.
    for i in 1:Nx
        for jp in 1:NpU
            w = wU_surface[i, jp];  e = eU_density[i, jp] * wpU[jp]
            (!isnan(w) && e > 1e-16) && (push!(wages_U, w);  push!(mass_U, e))
        end
        eU_entry[i] > 1e-16 && (push!(wages_U, wU_entry[i]);  push!(mass_U, eU_entry[i]))
    end
    # Training slice, cell by cell at its own reservation quality and outside option
    # Utr = −c(aS) + T(aU,aS): w_t = β_U A P_U aU p + (1−β_U)(r+ν) Utr.
    for j in 1:Nx, i in 1:Nx
        e = eUt_c[i, j];  e > 1e-16 || continue
        pst = clamp01(pstar_t[i, j])
        outside = (1.0 - βU) * (r + ν) * net_T[i, j]
        for jp in 1:NpU
            p   = pgU[jp]
            ω_j = _soft_weight(p, pst, pgU, jp, NpU)
            ω_j > 0.0 || continue
            g_p = (p <= 0.0) ? 0.0 : αU * p^(αU - 1.0)
            m   = ω_j * λU * g_p / (ν + λU) * e * wpU[jp]
            m > 1e-16 && (push!(wages_U, βU * PU * xg[i] * p + outside);  push!(mass_U, m))
        end
        m1 = cross.kt[i, j] * mS_c[i, j] / (ν + λU)
        m1 > 1e-16 && (push!(wages_U, βU * PU * xg[i] + outside);  push!(mass_U, m1))
    end
    # Skilled: each cell's mass split between wS1 and wS0 by the OJS covered share, so wage
    # moments are continuous in p^oj. Both surfaces share the p*_S test, so the split
    # conserves the cell's mass.
    for j in 1:Nx
        poj_j = clamp01(poj[j])
        for jp in 1:NpS
            e = eS_pS[j, jp];  e <= 1e-16 && continue
            m   = e * wpS[jp]
            s_j = _ojs_share(s_part[j], pg[jp], poj_j, pg, jp, NpS)
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

    # The wage moments are read off these points (model_moments).
    wage_points = (U = (w = wages_U, m = mass_U),
                   S = (w = [wages_S0; wages_S1], m = [mass_S0; mass_S1]))

    # Level-wage histograms for figures; edges at quantiles of the node wages.
    Nbins = 120
    wgrid = range(quantile(node_wages, 0.002), quantile(node_wages, 0.998); length = Nbins + 1)
    wmid  = collect(wgrid[1:end-1] .+ step(wgrid) / 2)
    dens_U  = _wage_density(wages_U,  mass_U,  wgrid)
    dens_S0 = _wage_density(wages_S0, mass_S0, wgrid)
    dens_S1 = _wage_density(wages_S1, mass_S1, wgrid)
    dens_S  = _wage_density(wage_points.S.w, wage_points.S.m, wgrid)

    # ── Population accounting (all masses already carry W2) ──────────────
    # c* are within-market (composition) masses; agg_* the true masses.
    cuU = sum(uU_c);  ceU = sum(eU_c)
    cuS = sum(uS_mat)
    ceS = sum(eS_totS[j] for j in 1:Nx)         # eS_pS carries mcol0 (2D mass) already
    cmS = cuS + (stayers ? ceS : 0.0)

    agg_uU = unk_open ? cuU : 0.0;  agg_eU = unk_open ? ceU : 0.0
    agg_n  = sum(nU);  agg_mU = sum(mU_mat)
    agg_uS = skl_open ? cuS : 0.0;  agg_eS = (skl_open && stayers) ? ceS : 0.0
    agg_mS = agg_uS + agg_eS

    total_pop = agg_mU + agg_mS
    lf_total  = agg_uU + agg_eU + agg_mS
    ur_U      = cuU + ceU > 1e-12 ? cuU / (cuU + ceU) : 1.0
    ur_S      = cmS       > 1e-12 ? cuS / cmS : 1.0
    ur_total  = lf_total  > 1e-12 ? (agg_uU + agg_uS) / lf_total : 1.0

    # ── Transition rates ──────────────────────────────────────────────────
    # The offer CDF governs acceptance of fresh meetings, the shock CDF endogenous
    # separation. Both read at the cutoff by cell mass (cdf_at_cutoff), exactly 0 where
    # p*_S = 0.
    Γo_pstarS = [cdf_at_cutoff(pre.γvals,   pg, wpS, clamp01(pstar_S[j])) for j in 1:Nx]
    Γs_pstarS = [cdf_at_cutoff(pre.γs_vals, pg, wpS, clamp01(pstar_S[j])) for j in 1:Nx]

    # Skilled job finding over the skilled unemployed: stayers accept offers at
    # κ_S(1−Γ_o(p*_S)); crossers meet unskilled vacancies at κ_U and are hired with their
    # cell's hired share (1−ω)χ + ω χ_t.
    accept_S = 1.0 .- Γo_pstarS
    χ_row    = hire_fraction(model)
    hire_X_share = [d_mat[i, j] > 0.0 ? cross.k[i, j] / (κ_U * d_mat[i, j]) : χ_row[i]
                    for i in 1:Nx, j in 1:Nx]
    uS_d1mass  = [sum(uS_drain[:, j]) for j in 1:Nx]               # by aS
    uS_d0mass  = [sum(uS_stay[:, j])  for j in 1:Nx]
    uS_colmass = uS_d0mass .+ uS_d1mass
    uS_mass    = sum(uS_colmass)
    if uS_mass > 1e-14
        hire_S = κS * dot(accept_S, uS_d0mass)
        hire_X = κ_U * dot(hire_X_share, uS_drain)
        f_S = (hire_S + hire_X) / uS_mass
    else
        f_S = κS
    end

    # Unskilled separation δ_U = ξ_U + λ_U G(p*), employed-weighted; the training slice at
    # its own p*_t.
    δU_by_a    = ξU .+ λU .* clamp01.(pstar_U) .^ αU
    δU_t       = ξU .+ λU .* clamp01.(pstar_t) .^ αU
    sep_rate_U = (dot(δU_by_a, eU_rowU) + dot(δU_t, eUt_c)) / max(ceU, 1e-14)

    # Skilled separation δ_S = ξ_S + λ_S Γ_s(p*_S), employed-weighted.
    δS_by_a    = ξS .+ λS .* Γs_pstarS
    sep_rate_S = dot(δS_by_a, eS_totS) / max(ceS, 1e-14)

    # Within-job wage-change hazard λ_j (1 − G_j(p*)): a redraw the match survives
    # re-prices it; ξ_j never does.
    wchg_by_aU  = λU .* (1.0 .- clamp01.(pstar_U) .^ αU)
    wchg_U_t    = λU .* (1.0 .- clamp01.(pstar_t) .^ αU)
    wchg_rate_U = (dot(wchg_by_aU, eU_rowU) + dot(wchg_U_t, eUt_c)) / max(ceU, 1e-14)
    wchg_by_aS  = λS .* (1.0 .- Γs_pstarS)
    wchg_rate_S = dot(wchg_by_aS, eS_totS) / max(ceS, 1e-14)

    # Skilled E-to-E rate and its mean log-wage step. The searching share s of cell (aS, p)
    # moves at κ_S(1−Γ_o(p)), the offer CDF at her own node. The expected post-move wage is
    # the offer-weighted mean of wS1 over destinations p′ ≥ p; the step is the log of that
    # mean level minus the log of the pre-move wage (as the SIPP logs the realised wage).
    ee_mass = 0.0;  ee_flow = 0.0;  ee_qprob = 0.0
    ee_step_flow = 0.0;  ee_step_massw = 0.0
    for j in 1:Nx
        poj_j = clamp01(poj[j])

        # Reverse-cumulative offer-weighted wS1 (NaN entries carry no weight).
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
            s_j = _ojs_share(s_part[j], pg[jp], poj_j, pg, jp, NpS)
            s_j <= 0.0 && continue
            flow_ij = s_j * e_ij * κS * (1.0 - pre.Γvals[jp])
            ee_flow += flow_ij
            # Quarterly move probability of the searching share, 1 − e^{−3h}.
            ee_qprob += s_j * e_ij * -expm1(-_MONTHS_PER_QUARTER * κS * (1.0 - pre.Γvals[jp]))
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

    # ── Rate moments as the data measure them ────────────────────────────
    # Each cell's monthly probability 1 − e^{−h}, averaged over the moment's universe and
    # converted as its source converts it. A cell split by a covered fraction into a moving
    # and a non-moving branch contributes share × the moving branch's probability.
    #   jfr_U        CPS U→E, all unskilled unemployed (only the hired branch is hired);
    #   jfr_S        CPS U→E, all skilled unemployed (stayers and crossers);
    #   sep_rate_j   CPS E→U; wchg_rate_j SIPP within-job re-pricing; both −log(1−p̄);
    #   ee_rate_S    J2J quarterly move probability q̄ over the employed skilled, reported
    #                as the monthly probability 1 − (1 − q̄)^{1/3} (j2j.jl).
    prob_κ_U  = -expm1(-κ_U)
    meas_ee_S = ee_mass > 1e-14 ? 1.0 - (1.0 - ee_qprob / ee_mass)^(1.0 / _MONTHS_PER_QUARTER) : 0.0
    measured = (
        jfr_U       = _hazard_of_mean_probability(hired_U .* prob_κ_U, uU_rowU, κ_U),
        jfr_S       = _hazard_of_mean_probability([-expm1.(-κS .* accept_S); vec(hire_X_share) .* prob_κ_U],
                                                  [uS_d0mass; vec(uS_drain)], κS),
        sep_rate_U  = _hazard_of_mean_probability([-expm1.(-δU_by_a); vec(-expm1.(-δU_t))],
                                                  [eU_rowU; vec(eUt_c)], 0.0),
        sep_rate_S  = _hazard_of_mean_probability(-expm1.(-δS_by_a), eS_totS, 0.0),
        ee_rate_S   = meas_ee_S,
        wchg_rate_U = _hazard_of_mean_probability([-expm1.(-wchg_by_aU); vec(-expm1.(-wchg_U_t))],
                                                  [eU_rowU; vec(eUt_c)], 0.0),
        wchg_rate_S = _hazard_of_mean_probability(-expm1.(-wchg_by_aS), eS_totS, 0.0),
    )

    # ── Unemployment-duration survivors ───────────────────────────────────
    # Survival of an ongoing spell past t months, a mixture of exponentials over the stock.
    # Skilled: stayers exit at κ_S(1−Γ_o(p*_S)) + ν; crossers at κ_U + ν on their hired
    # share and at ν otherwise.
    δS_unemp0 = κS .* (1.0 .- Γo_pstarS) .+ ν      # stayer: fresh OFFER acceptance
    δS_unemp1 = κ_U + ν                            # crosser: an UNSKILLED job
    _surv_S = t -> uS_mass > 1e-14 ?
        (dot(uS_d0mass, exp.(-δS_unemp0 .* t)) +
         dot(uS_drain, hire_X_share .* exp(-δS_unemp1 * t) .+ (1.0 .- hire_X_share) .* exp(-ν * t))) /
        uS_mass : 0.0

    # Unskilled: the hired branch exits at κ_U + ν, the never-hired branch only at ν.
    uU_by_a = vec(sum(uU_c, dims = 2))
    uU_mass = sum(uU_by_a)
    _surv_U = t -> uU_mass > 1e-14 ?
        dot(uU_by_a, hired_U .* exp(-(κ_U + ν) * t) .+ (1.0 .- hired_U) .* exp(-ν * t)) /
        uU_mass : 0.0

    usurv_S = [_surv_S(wk * _WEEKS_TO_MONTHS) for wk in DURATION_THRESHOLDS_WK]
    usurv_U = [_surv_U(wk * _WEEKS_TO_MONTHS) for wk in DURATION_THRESHOLDS_WK]
    # At 27 weeks on its own, independent of the threshold list.
    ltu_share_S = _surv_S(27.0 * _WEEKS_TO_MONTHS)

    return (
        xg = xg, Fell = Fell, pgU = pgU, pg = pg, waU = waU, waS = waS, W2 = W2,
        wpU = wpU, wpS = wpS, Nx = Nx, NpU = NpU, NpS = NpS,

        uU = uU, nU = nU, eUt = unk_open ? eUt : zero(eUt),
        mU_mat = mU_mat, mS_mat = mS_mat,
        uS_mat = skl_open ? uS_mat : zero(uS_mat),
        eS_pS   = (skl_open && stayers) ? eS_pS   : zero(eS_pS),
        eS_totS = (skl_open && stayers) ? eS_totS : zero(eS_totS),
        eU_surface = unk_open ? eU_surface : zero(eU_surface),
        eU_density = unk_open ? eU_density : zero(eU_density),
        eU_entry   = unk_open ? eU_entry   : zero(eU_entry),
        eU_rowU    = unk_open ? eU_rowU    : zero(eU_rowU),

        Usearch = Usearch, net_T = net_T, τ_mat = τ_mat, pstar_t = pstar_t,
        JU_frontier = JU_frontier, SU1 = SU1, EU1 = EU1, pstar_U = pstar_U,
        SU_surface = SU_surface, JU_surface = JU_surface, EU_surface = EU_surface,

        US = US, pstar_S = pstar_S, poj = poj, s_part = s_part, d = d_mat,
        Smax_surface = Smax_surface,
        E0_surface = copy(sc.E0), E1_surface = copy(sc.E1),
        J0_surface = copy(sc.J0), J1_surface = copy(sc.J1),

        wU_surface = wU_surface, wU_entry = wU_entry, wS0_surface = wS0_surface, wS1_surface = wS1_surface,
        wage_points = wage_points,
        wmid = wmid, dens_U = dens_U, dens_S = dens_S, dens_S0 = dens_S0, dens_S1 = dens_S1,

        agg_uU = agg_uU, agg_n = agg_n, agg_eU = agg_eU, agg_mU = agg_mU,
        agg_uS = agg_uS, agg_eS = agg_eS, agg_mS = agg_mS,
        total_pop = total_pop, ur_U = ur_U, ur_S = ur_S, ur_total = ur_total,
        thetaU = θU, thetaS = θS,

        κ_U = κ_U, f_S = f_S, sep_rate_U = sep_rate_U, sep_rate_S = sep_rate_S,
        ee_rate_S = ee_rate_S, ee_step_S = ee_step_S,
        wchg_rate_U = wchg_rate_U, wchg_rate_S = wchg_rate_S,
        measured = measured,
        ltu_share_S = ltu_share_S,
        usurv_S = usurv_S, usurv_U = usurv_U,
        σ_wU = up.σ_w, σ_wS = sp.σ_w,
    )
end


"""
    print_accounting(obj)

Print the population-accounting table of `compute_equilibrium_objects`.
"""
function print_accounting(obj)
    @printf("\n┌─────────────────────────────────────────────────────┐\n")
    @printf("│  Equilibrium accounting                               │\n")
    @printf("├─────────────────────────────────────────────────────┤\n")
    @printf("│  UNSKILLED  (pop share %6.4f)                        │\n", obj.agg_mU)
    @printf("│    u_U %6.4f   e_U %6.4f   n %6.4f   ur_U %5.1f%%  │\n", obj.agg_uU, obj.agg_eU, obj.agg_n, 100 * obj.ur_U)
    @printf("│  SKILLED    (pop share %6.4f)                        │\n", obj.agg_mS)
    @printf("│    u_S %6.4f   e_S %6.4f            ur_S %5.1f%%  │\n", obj.agg_uS, obj.agg_eS, 100 * obj.ur_S)
    @printf("│  TOTAL POP %6.4f   global u-rate %6.4f             │\n", obj.total_pop, obj.ur_total)
    @printf("│  θ_U %8.4f   θ_S %8.4f                          │\n", obj.thetaU, obj.thetaS)
    @printf("└─────────────────────────────────────────────────────┘\n")
end

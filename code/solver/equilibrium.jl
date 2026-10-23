############################################################
# equilibrium.jl — Post-solution equilibrium objects (RoySearch)
#
# compute_equilibrium_objects(model)
#   Returns a NamedTuple with the stationary densities, value / surplus /
#   firm surfaces, wages, population accounting, and transition rates —
#   everything model_moments consumes.  The rates come twice: as hazards
#   (κ_U, f_S, sep_rate_j, ...), and under `measured` as the data stage
#   measures each rate moment, which is what model_moments targets.
#
# Dimensionality.  Value/surplus/wage surfaces are 1D in own ability
# (unskilled in aU, skilled in aS) × the p-grid, exactly as solved, except
# the unskilled matches of the trained whose outside option is training
# (the training slice, career_choices!), which read the whole type.
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
const _MONTHS_PER_QUARTER = 3.0

# A CPS or SIPP rate moment as the data stage measures it (Data and Moments, Frequency
# consistency): the mean monthly transition probability over the moment's universe,
# converted to a hazard by −log(1 − p̄). `prob` holds each cell's monthly probability and
# `mass` its weight in that universe; a universe holding no one returns `empty`.
function _hazard_of_mean_probability(prob::AbstractVector{Float64}, mass::AbstractVector{Float64},
                                     empty::Float64)
    M = sum(mass)
    M > 1e-14 || return empty
    return -log1p(-dot(prob, mass) / M)
end

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
    tU     = copy(uc.t)                                # in training
    d_mat  = clamp.(copy(sc.d), 0.0, 1.0)
    cross  = crossing_rates(model, d_mat)              # unskilled hires of the trained
    mS_mat = _mS_from_t(tU, cross.k, φ, ν)             # trained mass
    mU_mat = max.(W2 .- mS_mat, 0.0)                   # untrained-segment mass
    eU_mat = max.(mU_mat .- uU .- tU, 0.0)             # unskilled employed mass, both slices
    eUt    = copy(uc.et)                               # of which the training slice

    # ── Market compositions ──────────────────────────────────────────────
    # Every within-market object below — rates, wage densities, survivors — is a ratio
    # over that market's own mass, so it depends on the market's composition and not on
    # its size.  A market holding no one has no composition of its own; it takes the
    # limit of its composition as it empties (marginal_training_cells), the same one its
    # free entry prices, so each such object is continuous where the market closes.  The
    # population aggregates keep the true masses.  In an open market the composition IS
    # the mass array, so nothing below changes there.
    skl_open = sum(mS_mat) > EMPTY_MASS
    unk_open = sum(uU) + sum(eU_mat) > EMPTY_MASS
    mS_c = skl_open ? mS_mat :
           _mS_from_t(marginal_training_cells(model; entering = true), cross.k, φ, ν)
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
    # Employment below EMPTY_MASS is none: at that size it is the rounding residue of the
    # population identity above, not a composition.
    sum(eU_c) < EMPTY_MASS && (fill!(eU_c, 0.0);  fill!(eUt_c, 0.0))

    # Skilled unemployed / employed reconstructed from per-aS unit shapes, with the
    # cell SPLIT by the drain fraction rather than assigned wholly to one branch:
    #   the d part:    u_S = m_S,        e_S ≡ 0 (these seek in the U-market)
    #   the 1−d part:  u_S = û(aS) m_S,  e_S(p) = ê(aS,p) m_S
    # Thresholding at 0.5 here would discard the fractional boundary node and put
    # the step function back, whatever the producer does upstream.  The two pieces
    # are kept apart because the flow rates below apply a different exit hazard to
    # each — a crosser leaves at κ_U, a stayer at κ_S(1−Γ_o(p*_S)).
    uS_drain = similar(mS_mat)                         # crossers: all of the sub-mass
    uS_stay  = similar(mS_mat)                         # stayers: unemployed share only
    @inbounds for j in 1:Nx, i in 1:Nx
        uS_drain[i, j] = d_mat[i, j] * mS_c[i, j]
        uS_stay[i, j]  = (1.0 - d_mat[i, j]) * sc.u_frac[j] * mS_c[i, j]
    end
    uS_mat = uS_drain .+ uS_stay
    # eS on (aS,p) aggregated over aU (only non-draining mass carries employment):
    mcol0  = [sum((1.0 .- d_mat[:, j]) .* mS_c[:, j]) for j in 1:Nx]     # by aS
    # With no stayers the employed composition is their limit instead, so the employed-
    # weighted objects below stay defined; the market's employed MASS is then zero.
    stayers = sum(mcol0) > EMPTY_MASS
    mcolE   = stayers ? mcol0 : skilled_stayer_columns(model, d_mat, mS_c)
    eS_pS  = [sc.e_frac[j, jp] * mcolE[j] for j in 1:Nx, jp in 1:NpS]    # (aS, p) employed density
    eS_totS = [dot(eS_pS[j, :], wpS) for j in 1:Nx]                      # employed mass per aS
    sum(eS_pS) < EMPTY_MASS && (fill!(eS_pS, 0.0);  fill!(eS_totS, 0.0))

    # ── Unskilled employment surface e_U(aU, p) ──────────────────────────
    # The searching slice — every unskilled match whose outside option is U^search(aU),
    # the trained hired from outside the training set among them — reads aU alone.  Row
    # mass in aU: integrate its 2D employed density over aS.  Its hires, from the hired
    # branch of the row's unemployed (uc.uh) and from the trained of the searching slice
    # (rate ks), feed the hiring atom at p = 1, `eU_entry`; the damaged tail is
    # `eU_density`.  p = 1 is not a grid node, so the reported surface carries the atom in
    # its top column, and the wage block pays the atom the entry wage w_U(aU,1).  The
    # training slice eUt has its own reservation quality per cell and enters the wage
    # points and the rates directly.
    eUs_c   = max.(eU_c .- eUt_c, 0.0)
    eU_rowU = [dot(eUs_c[i, :], ones(Nx)) for i in 1:Nx]                # Σ_j (mass, W2 already in it)
    uU_rowU = [sum(uU_c[i, :]) for i in 1:Nx]
    hires_s = [κ_U * sum(uh_c[i, :]) + dot(cross.ks[i, :], mS_c[i, :]) for i in 1:Nx]
    # Hired share of each row's unemployed: its stationary split, with the separated
    # crossers of the searching slice in the hired branch.
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
    # Mixed units: density in p plus the new matches' MASS in the top column (mass: eU_density·wpU, eU_entry).
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
    # Read from one cell BELOW the reservation node: S^max carries the soft
    # coverage weight, so the cell straddling p*_S contributes its covered
    # fraction.  Same index skilled_inner_loop! forms I_S at.
    I_full = [tailS[j, max(pcut_index(pg, clamp01(pstar_S[j])) - 1, 1)] for j in 1:Nx]

    # Each wage is written at EVERY node (`w*_node`), whatever the node's mass: that set
    # never changes size, so the histogram edges placed on it below move continuously with
    # the parameters.  The reported surfaces keep a wage only where the node carries mass.
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

    # w^0_S(aS,p), w^1_S(aS,p): linear in aS, no γ_S.
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

    # Wage points: one (wage, mass) pair per employed cell, so each market's points count
    # every employed worker once, as the data count employed workers.  Unskilled: the
    # damaged tail below p = 1 is a density in p, so each (aU,p) node carries density ×
    # quadrature weight at its wage; the hiring atom at p = 1 is a mass already and is paid
    # the entry wage w_U(aU,1).
    for i in 1:Nx
        for jp in 1:NpU
            w = wU_surface[i, jp];  e = eU_density[i, jp] * wpU[jp]
            (!isnan(w) && e > 1e-16) && (push!(wages_U, w);  push!(mass_U, e))
        end
        eU_entry[i] > 1e-16 && (push!(wages_U, wU_entry[i]);  push!(mass_U, eU_entry[i]))
    end
    # The training slice, cell by cell at the cell's own reservation quality, with the same
    # tail and atom as the searching slice: w_t(aU,aS,p) = β_U A P_U aU p + (1−β_U)(r+ν) Utr,
    # Utr = −c(aS) + T(aU,aS) the outside option of a hire whose separations re-enter training.
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

    # Level-wage histograms, for figures; no SMM moment reads them.  Their edges are
    # quantiles of the node wages, so they too move continuously with the parameters.
    Nbins = 120
    wgrid = range(quantile(node_wages, 0.002), quantile(node_wages, 0.998); length = Nbins + 1)
    wmid  = collect(wgrid[1:end-1] .+ step(wgrid) / 2)
    dens_U  = _wage_density(wages_U,  mass_U,  wgrid)
    dens_S0 = _wage_density(wages_S0, mass_S0, wgrid)
    dens_S1 = _wage_density(wages_S1, mass_S1, wgrid)
    dens_S  = _wage_density(wage_points.S.w, wage_points.S.m, wgrid)

    # ── Population accounting (all masses already carry W2) ──────────────
    # Within-market masses (c_*) are the compositions; the aggregates are the true
    # masses, zero for a market that holds no one.
    cuU = sum(uU_c);  ceU = sum(eU_c)
    cuS = sum(uS_mat)
    ceS = sum(eS_totS[j] for j in 1:Nx)         # eS_pS carries mcol0 (2D mass) already
    cmS = cuS + (stayers ? ceS : 0.0)

    agg_uU = unk_open ? cuU : 0.0;  agg_eU = unk_open ? ceU : 0.0
    agg_t  = sum(tU);  agg_mU = sum(mU_mat)
    agg_uS = skl_open ? cuS : 0.0;  agg_eS = (skl_open && stayers) ? ceS : 0.0
    agg_mS = agg_uS + agg_eS
    agg_mS_flow = sum(mS_mat)

    total_pop = agg_mU + agg_mS
    lf_total  = agg_uU + agg_eU + agg_mS
    ur_U      = cuU + ceU > 1e-12 ? cuU / (cuU + ceU) : 1.0
    ur_S      = cmS       > 1e-12 ? cuS / cmS : 1.0
    ur_total  = lf_total  > 1e-12 ? (agg_uU + agg_uS) / lf_total : 1.0

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
    # stayers accept S-offers at κ_S(1−Γ_o(p*_S)); crossers meet U-vacancies at κ_U and
    # are hired with their cell's hired share h = (1−ω)χ + ω χ_t (crossing_rates).
    # Both masses come from the split above, so the composition stays a partition of
    # uS_mat at a fractional drain (re-deriving one of them as d·uS_mat would double
    # the drain and break the identity).
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

    # Unskilled separation hazard δ_U(aU) = ξ_U + λ_U p*(aU)^{α_U}, employed-weighted.
    # ξ_U is the exogenous baseline; the endogenous part reads the SHOCK CDF
    # G(p*) = p*^{α_U} (a match separates when a λ_U redraw lands below p*).
    # The training slice separates at its own cutoff, δ_t = ξ_U + λ_U p*_t^{α_U}.
    δU_by_a    = ξU .+ λU .* clamp01.(pstar_U) .^ αU
    δU_t       = ξU .+ λU .* clamp01.(pstar_t) .^ αU
    sep_rate_U = (dot(δU_by_a, eU_rowU) + dot(δU_t, eUt_c)) / max(ceU, 1e-14)

    # Skilled separation hazard δ_S(aS) = ξ_S + λ_S Γ_s(p*_S), employed-weighted.
    # The endogenous part reads the SHOCK CDF: a match separates when a λ_S
    # redraw lands below the reservation cutoff, and redraws are shock draws.
    δS_by_a    = ξS .+ λS .* Γs_pstarS
    sep_rate_S = dot(δS_by_a, eS_totS) / max(ceS, 1e-14)

    # Within-job wage-change hazard: a λ_j redraw landing AT OR ABOVE the
    # reservation cutoff survives the match but re-prices it, at rate
    # λ_j·(1 − G_j(p*)).  ξ_j never re-prices a survivor, so wchg_rate loads on
    # λ_j blind to ξ_j — the moment that separates the two (SIPP; data-and-
    # moments §sipp).  Same employed weights and cutoff CDFs as sep_rate above.
    wchg_by_aU  = λU .* (1.0 .- clamp01.(pstar_U) .^ αU)
    wchg_U_t    = λU .* (1.0 .- clamp01.(pstar_t) .^ αU)
    wchg_rate_U = (dot(wchg_by_aU, eU_rowU) + dot(wchg_U_t, eUt_c)) / max(ceU, 1e-14)
    wchg_by_aS  = λS .* (1.0 .- Γs_pstarS)
    wchg_rate_S = dot(wchg_by_aS, eS_totS) / max(ceS, 1e-14)

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
    ee_mass = 0.0;  ee_flow = 0.0;  ee_qprob = 0.0
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
            s_j = _ojs_share(s_part[j], pg[jp], poj_j, pg, jp, NpS)
            s_j <= 0.0 && continue
            flow_ij = s_j * e_ij * κS * (1.0 - pre.Γvals[jp])
            ee_flow += flow_ij
            # The searching share moves within a quarter with probability 1 − e^{−3h};
            # the rest of the cell does not search (rate moments block below).
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

    # ── Rate moments as the data stage measures them ─────────────────────
    # The hazards above are flows over mass, a mean of type hazards. The data build each
    # rate from worker-level transitions: the mean transition probability over the
    # moment's universe, then converted as that moment's source converts it. Each cell's
    # probability is 1 − e^{−h} at its constant hazard h, averaged over the same mass the
    # hazard above is weighted by. A cell split by a covered fraction into a branch that
    # moves and one that does not (the hired share of a row's unemployed, the searching
    # share of an employed cell) is a mixture: its probability is the share times the
    # moving branch's, not the probability of the share-weighted hazard.
    #   jfr_U   CPS U→E over all unskilled unemployed: only the hired branch is hired;
    #   jfr_S   CPS U→E over all skilled unemployed: stayers at κ_S(1−Γ_o(p*_S)),
    #           crossers at κ_U on their hired share;
    #   sep_rate_U, wchg_rate_U   over both slices of unskilled employment;
    #   sep_rate_j, wchg_rate_j   CPS E→U and SIPP within-job re-pricing, both −log(1−p̄);
    #   ee_rate_S   J2J: the quarterly move probability q̄ over the employed skilled,
    #           reported as the monthly probability 1 − (1 − q̄)^{1/3} (j2j.jl).
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
    # Survival of an ONGOING unemployment spell past t months, per market, as a
    # mixture of exponential exit hazards over that market's unemployed stock.
    #
    # Skilled.  The exit hazard must match the one used for f_S above: a stayer
    # leaves skilled unemployment by accepting a fresh skilled OFFER at
    # κ_S(1−Γ_o(p*_S)), while a crosser leaves by taking an UNSKILLED job at κ_U on her
    # cell's hired share, its never-hired share only at ν.  All exit demographically at ν.
    # Applying the stayer hazard to the whole stock understated the crossers' exit rate
    # and overstated their survival.
    δS_unemp0 = κS .* (1.0 .- Γo_pstarS) .+ ν      # stayer: fresh OFFER acceptance
    δS_unemp1 = κ_U + ν                            # crosser: an UNSKILLED job
    _surv_S = t -> uS_mass > 1e-14 ?
        (dot(uS_d0mass, exp.(-δS_unemp0 .* t)) +
         dot(uS_drain, hire_X_share .* exp(-δS_unemp1 * t) .+ (1.0 .- hire_X_share) .* exp(-ν * t))) /
        uS_mass : 0.0

    # Unskilled.  Each row's unemployed are a mixture of the two branches
    # solve_stationary_unskilled! balances: the hired branch (share hired_U)
    # leaves at κ_U + ν, and the never-hired branch, below the hiring threshold
    # a^h, is unemployed and searching but never hired, so it leaves only at ν.
    # The never-hired mass puts a floor under the unskilled survivor that the
    # skilled one does not have.
    uU_by_a = vec(sum(uU_c, dims = 2))
    uU_mass = sum(uU_by_a)
    _surv_U = t -> uU_mass > 1e-14 ?
        dot(uU_by_a, hired_U .* exp(-(κ_U + ν) * t) .+ (1.0 .- hired_U) .* exp(-ν * t)) /
        uU_mass : 0.0

    usurv_S = [_surv_S(wk * _WEEKS_TO_MONTHS) for wk in DURATION_THRESHOLDS_WK]
    usurv_U = [_surv_U(wk * _WEEKS_TO_MONTHS) for wk in DURATION_THRESHOLDS_WK]
    # Kept as its own field and evaluated through the same function, so it stays
    # correct even if 27 weeks is later dropped from the threshold list.
    ltu_share_S = _surv_S(27.0 * _WEEKS_TO_MONTHS)

    return (
        xg = xg, Fell = Fell, pgU = pgU, pg = pg, waU = waU, waS = waS, W2 = W2,
        wpU = wpU, wpS = wpS, Nx = Nx, NpU = NpU, NpS = NpS,

        # Mass arrays are the true masses; for a market that holds no one they are zero.
        uU = uU, tU = tU, eU_mat = eU_mat, eUt = unk_open ? eUt : zero(eUt),
        mU_mat = mU_mat, mS_mat = mS_mat,
        uS_mat = skl_open ? uS_mat : zero(uS_mat),
        eS_pS   = (skl_open && stayers) ? eS_pS   : zero(eS_pS),
        eS_totS = (skl_open && stayers) ? eS_totS : zero(eS_totS),
        eU_surface = unk_open ? eU_surface : zero(eU_surface),
        eU_density = unk_open ? eU_density : zero(eU_density),
        eU_entry   = unk_open ? eU_entry   : zero(eU_entry),
        eU_rowU    = unk_open ? eU_rowU    : zero(eU_rowU),

        Usearch = Usearch, Tv = Tv, net_T = net_T, τ_mat = τ_mat, pstar_t = pstar_t,
        JU_frontier = JU_frontier, SU1 = SU1, EU1 = EU1, pstar_U = pstar_U,
        SU_surface = SU_surface, JU_surface = JU_surface, EU_surface = EU_surface,

        US = US, pstar_S = pstar_S, poj = poj, s_part = s_part, d = d_mat,
        S0_surface = S0_surface, S1_surface = S1_surface, Smax_surface = Smax_surface,
        E0_surface = copy(sc.E0), E1_surface = copy(sc.E1),
        J0_surface = copy(sc.J0), J1_surface = copy(sc.J1),

        wU_surface = wU_surface, wU_entry = wU_entry, wS0_surface = wS0_surface, wS1_surface = wS1_surface,
        wage_points = wage_points,
        wmid = wmid, dens_U = dens_U, dens_S = dens_S, dens_S0 = dens_S0, dens_S1 = dens_S1,

        agg_uU = agg_uU, agg_t = agg_t, agg_eU = agg_eU, agg_mU = agg_mU,
        agg_uS = agg_uS, agg_eS = agg_eS, agg_mS = agg_mS, agg_mS_flow = agg_mS_flow,
        total_pop = total_pop, ur_U = ur_U, ur_S = ur_S, ur_total = ur_total,
        thetaU = θU, thetaS = θS,

        κ_U = κ_U, f_S = f_S, sep_rate_U = sep_rate_U, sep_rate_S = sep_rate_S,
        ee_rate_S = ee_rate_S, ee_step_S = ee_step_S,
        wchg_rate_U = wchg_rate_U, wchg_rate_S = wchg_rate_S,
        # The seven rate moments as the data measure them, keyed by moment name; the
        # hazards above keep their meaning for the notes, plots and transitions.
        measured = measured,
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

############################################################
# transition_solver.jl — RoySearch backward–forward transition
#
# Solves the perfect-foresight path from a pre-switch stationary equilibrium z₀ to a
# post-switch one z₁ after an unanticipated change in the parameters at date 0
# (transition_spec.md; notes, sec:transition).  Free entry holds at every date, so the
# path is off steady state, not off equilibrium: vacancies are a jump variable and
# V_j(t) = 0 throughout.
#
# Algorithm (spec Part 4)
#   Step 0  z₀, z₁ solved externally (transition_simulation.jl).
#   Step 1  Initialise: tightness at z₁, every mass at z₀ (date 0⁻), values at z₁ (date N).
#   Step 2  Backward pass  — time-dependent values at the given θ path (transition_values.jl).
#   Step 3  Forward pass   — at each date, reallocate with that date's policies, then
#                            advance the laws of motion with that date's rates.
#   Step 4  Free entry     — θ_j(n) = (μ_j 𝒥_j(n)/k_j)², from date n's values and its
#                            masses after the reallocation; damped update.
#   Step 5  Iterate 2–4 until the free-entry gap max_n |ln θ^new − ln θ| < tol.
#
# Masses, per type (aU,aS) and per slice of the cell (see transition_values.jl):
#   u_U  unskilled unemployed (searching slice)      t    in training
#   e_U^s, e_U^τ  unskilled matches of the two slices, each a measure over p with an
#        atom at p = 1 (a new match starts there; the redraws spread it over G)
#   u_S  skilled unemployed                           m_S  skilled segment
#   ê_S  per-aS unit shape of skilled employment over p (exact as a column aggregate
#        for every type that stays skilled; see Known gaps in the changelog)
#
# Reallocations at a date (spec L6), in this order:
#   (a) the frontier encloses the share (τ−τ⁻)/(1−τ⁻) of the searching slice: its
#       unemployed enter training, its matches join the training slice; the share
#       (τ⁻−τ)/τ⁻ of the training slice's matches returns (training is never abandoned);
#   (b) every unskilled match below its slice's cutoff ends — into u_U from the searching
#       slice, into training from the training slice — and so does every match that
#       changed slice and lies below the new slice's cutoff;
#   (c) every skilled match below the cutoff ends into u_S.
# Laws of motion between reallocations (spec L1–L5), explicit Euler:
#   ∂_t u_U = (1−τ)νℓ + D^s − κ_U u_U^h − ν u_U,       D^s = (ξ_U + λ_U G(p*^s)) e_U^s
#   ∂_t t   = τνℓ + D^τ − (φ+ν) t
#   ∂_t e_U^s(dp) = κ_U (u_U^h + χ d u_S) δ₁ + λ_U e_U^s dG 1{p ≥ p*^s} − (λ_U+ξ_U+ν) e_U^s(dp)
#   ∂_t u_S = φ t + (ξ_S + λ_S Γ_s(p*_S)) e_S − (ν + (1−d) κ_S (1−Γ_o(p*_S)) + d κ_U χ) u_S
#   ∂_t m_S = φ t − ν m_S − d κ_U χ u_S
# u_U^h is the hired branch of u_U (`hired_unemployed`) and χ the hired fraction of the
# cell (`hire_fraction`): a crosser is hired where an unskilled match forms, like anyone.
# Each law is linear, so the population identity u_U + t + e_U^s + e_U^τ + m_S = ℓ holds
# at every date to rounding, with no residual stock.
############################################################


# ════════════════════════════════════════════════════════════
#  Step 1: Initialise the path
# ════════════════════════════════════════════════════════════

"""
    _unskilled_p_shape!(e, atom_mass, rest, pstar, p, wG)

Spread a cell's unskilled employment over quality: `atom_mass` at p = 1 (slot Np+1) and
`rest` over the nodes the cutoff covers, in proportion to the redraw density, ω·dG; with no
node covered, `rest` joins the atom.
"""
function _unskilled_p_shape!(e::AbstractVector{Float64}, atom_mass::Float64, rest::Float64,
                             pstar::Float64, p::Vector{Float64}, wG::Vector{Float64})
    Np = length(p)
    tot = 0.0
    @inbounds for q in 1:Np
        e[q] = _soft_weight(p[q], pstar, p, q, Np) * wG[q];  tot += e[q]
    end
    @inbounds for q in 1:Np
        e[q] = tot > 0.0 ? rest * e[q] / tot : 0.0
    end
    # Where the cutoff covers no node (p* = 1) every match is at p = 1.
    e[Np + 1] = tot > 0.0 ? atom_mass : atom_mass + rest
    return e
end

"""
    _init_path!(path, model_z0, model_z1)

Masses at every date start at z₀ (date 0⁻, spec B1), tightness at z₁ (the spec's first
guess), values at date N at z₁.  z₀'s unskilled employment is spread over quality at z₀'s
own stationary shape: the atom κ u^h/(λ_U+ξ_U+ν) at p = 1 and the redraws over the
accepted part of G, which is a fixed point of the law of motion at z₀'s rates.
"""
function _init_path!(path::TransitionPath, model_z0::Model, model_z1::Model)
    Nt  = length(path.tgrid)
    Nx  = length(model_z0.grids.x);  NpU = length(model_z0.unsk_grids.p)

    cp = model_z0.common;  up0 = model_z0.unsk_par
    uc0 = model_z0.unsk_cache;  sc0 = model_z0.skl_cache
    W2  = model_z0.grids.copula.W2
    κ_U0 = meeting_rate(uc0.θ, up0.μ, up0.η)
    χ0   = hire_fraction(model_z0)
    wG0  = build_unskilled_G_weights(model_z0.unsk_grids.p, model_z0.unsk_grids.wp, up0.α_U)

    d0  = clamp.(sc0.d, 0.0, 1.0)
    mS0 = _mS_from_t(uc0.t, d0, cp.φ, cp.ν, κ_U0)
    uS0 = similar(mS0)
    @inbounds for j in 1:Nx, i in 1:Nx
        uS0[i, j] = d0[i, j] * mS0[i, j] + (1.0 - d0[i, j]) * sc0.u_frac[j] * mS0[i, j]
        e   = max(W2[i, j] - mS0[i, j] - uc0.u[i, j] - uc0.t[i, j], 0.0)
        uh  = hired_unemployed(uc0.u[i, j], χ0[i], clamp(uc0.τT[i, j], 0.0, 1.0), W2[i, j])
        atom = min(κ_U0 * uh / (up0.λ + up0.ξ + cp.ν), e)
        _unskilled_p_shape!(view(path.eU0p, i, j, :), atom, e - atom, clamp01(uc0.pstar[i]),
                            model_z0.unsk_grids.p, wG0)
    end

    path.θU .= model_z1.unsk_cache.θ
    path.θS .= model_z1.skl_cache.θ
    @inbounds for n in 1:Nt
        path.uU[:, :, n]  .= uc0.u
        path.tU[:, :, n]  .= uc0.t
        path.uS[:, :, n]  .= uS0
        path.mS[:, :, n]  .= mS0
        path.eUs[:, :, n] .= dropdims(sum(path.eU0p, dims = 3), dims = 3)
        path.eS[:, :, n]  .= sc0.e_frac
    end
    path.τT0 .= uc0.τT
    path.uU0 .= uc0.u
    path.tU0 .= uc0.t
    path.pU0 .= uc0.pstar
    path.pS0 .= sc0.pstar
    path.χU0 .= χ0

    _seed_terminal_values!(path, model_z1)
    return path
end


# ════════════════════════════════════════════════════════════
#  Step 3: Forward pass
# ════════════════════════════════════════════════════════════

# Fraction of a node's mass that a cutoff moving from `pold` to `pnew` leaves in place:
# the ratio of the two coverage weights where the cutoff rose into or past the node.
@inline function _kept_share(pj::Float64, pold::Float64, pnew::Float64,
                             p::Vector{Float64}, j::Int, Np::Int)
    ωo = _soft_weight(pj, pold, p, j, Np);  ωn = _soft_weight(pj, pnew, p, j, Np)
    return ωo > 0.0 ? min(ωn / ωo, 1.0) : (ωn > 0.0 ? 1.0 : 0.0)
end

"""
    _end_below!(e, pold, pnew, p, atom_kept) -> ended mass

End, in place, the part of an unskilled quality measure that a cutoff moving from `pold`
to `pnew` places below reservation (spec L6a), and the share `1 − atom_kept` of the atom at
p = 1.  The atom is not judged by the cutoff: p* = 1 says only that no quality below 1 is
accepted, and on a row the hiring margin crosses the matches at p = 1 of the hired share
have positive surplus.
"""
function _end_below!(e::AbstractVector{Float64}, pold::Float64, pnew::Float64,
                     p::Vector{Float64}, atom_kept::Float64)
    Np = length(p);  ended = 0.0
    @inbounds for q in 1:pcut_index(p, max(pold, pnew))     # every node above keeps all its mass
        k = _kept_share(p[q], pold, pnew, p, q, Np)
        ended += (1.0 - k) * e[q];  e[q] *= k
    end
    if atom_kept < 1.0
        ended += (1.0 - atom_kept) * e[Np + 1];  e[Np + 1] *= atom_kept
    end
    return ended
end

# Share of a cell's crossers that the training frontier encloses.  The crossing set is the
# lower part of the cell in aS and the training set the upper part, so it is their overlap.
@inline _enclosed_crossers(d::Float64, τ::Float64) = d > 0.0 ? max(d + τ - 1.0, 0.0) / d : 0.0

"""
    _redraw_weights!(ω, pstar, p, wG, α) -> (wsum, G)

The accepted part of the redraw distribution for a cutoff: the covered share ω of each
node's cell, the accepted mass Σ ω·dG, and the exact mass G(p*) below the cutoff.  One
evaluation serves every cell that shares the cutoff.
"""
function _redraw_weights!(ω::Vector{Float64}, pstar::Float64, p::Vector{Float64},
                          wG::Vector{Float64}, α::Float64)
    Np = length(p);  wsum = 0.0
    @inbounds for q in 1:Np
        ω[q] = _soft_weight(p[q], pstar, p, q, Np);  wsum += ω[q] * wG[q]
    end
    return (wsum = wsum, G = G_cdf_unskilled(pstar, α))
end

"""
    _advance_unskilled!(e, hires, ω, rw, wG, λ, ξ, ν, dt) -> separations

One explicit step of an unskilled quality measure (spec L1): hires into the atom at p = 1,
the λ_U redraw of every match spread over the accepted part of G (`ω`, `rw` from
`_redraw_weights!`), exits at λ_U + ξ_U + ν.  The accepted redraws are scaled to the exact
mass 1 − G(p*), as the stationary balance counts them, and the returned separations are
(ξ_U + λ_U G(p*))·e.
"""
function _advance_unskilled!(e::AbstractVector{Float64}, hires::Float64, ω::Vector{Float64},
                             rw, wG::Vector{Float64}, λ::Float64, ξ::Float64, ν::Float64,
                             dt::Float64)
    Np = length(ω)
    etot = sum(e)
    etot == 0.0 && hires == 0.0 && return 0.0          # an empty measure stays empty
    scale = rw.wsum > 0.0 ? λ * etot * (1.0 - rw.G) / rw.wsum : 0.0
    out = λ + ξ + ν
    @inbounds for q in 1:Np
        e[q] += dt * (scale * ω[q] * wG[q] - out * e[q])
    end
    e[Np + 1] += dt * (hires - out * e[Np + 1])
    return (ξ + λ * (rw.wsum > 0.0 ? rw.G : 1.0)) * etot
end

"""
    _forward_pass!(path, model_z1, tp)

March the masses from z₀ through the path: at each date the reallocations of the date's
policies, the stock record (the masses free entry and the statistics read), then one
explicit step of the laws of motion at the date's rates.
"""
function _forward_pass!(path::TransitionPath, model::Model, tp::TransitionParams)
    Nt  = length(path.tgrid)
    Nx  = length(model.grids.x);  NpU = length(model.unsk_grids.p)
    cp = model.common;  up = model.unsk_par
    W2 = model.grids.copula.W2;  x = model.grids.x;  ug = model.unsk_grids
    ν = cp.ν;  φ = cp.φ;  βU = up.β
    PU = exp(cp.A) * up.PU
    χU = hire_fraction(model)                      # post-switch primitives, constant on the path
    wG = build_unskilled_G_weights(ug.p, ug.wp, up.α_U)

    uU = copy(path.uU0);  tU = copy(path.tU0)
    Es = copy(path.eU0p);  Et = zeros(Float64, Nx, Nx, NpU + 1)
    uS = copy(path.uS[:, :, 1]);  mS = copy(path.mS[:, :, 1]);  eS = copy(path.eS[:, :, 1])
    τprev = copy(path.τT0);  pUs_prev = repeat(path.pU0, 1, Nx);  pUt_prev = ones(Nx, Nx)
    pS_prev = copy(path.pS0)
    uS_n = similar(uS);  mS_n = similar(mS);  tU_n = similar(tU)
    mv_s = zeros(Float64, NpU + 1);  mv_t = zeros(Float64, NpU + 1)

    for n in 1:Nt
        d = _d_matrix(path, model, n)

        # ── Reallocation at date n (spec L6) ─────────────────────────────
        # Each moved or remaining mass is judged against the cutoff it was accepted under
        # and the cutoff of the slice it is in now.
        ended = 0.0;  quit = 0.0;  swept = 0.0
        @inbounds for j in 1:Nx, i in 1:Nx
            τ  = clamp(path.τT[i, j, n], 0.0, 1.0)
            τ0 = clamp(τprev[i, j], 0.0, 1.0)
            σ  = sweep_share(τ0, τ);  ς = return_share(τ0, τ)
            ps = path.pUs[i, j, n];  pt = path.pUt[i, j, n]
            es = view(Es, i, j, :);  et = view(Et, i, j, :)
            # The atom of the searching slice is the hired share χ of the row (S_U(a,1) ≥ 0
            # exactly above a^h), which moves only when the primitives switch; in the
            # training slice it is kept while that slice's raw surplus at p = 1 is
            # non-negative, and ends the first date it is not.
            ks = n == 1 && path.χU0[i] > 0.0 ? min(χU[i] / path.χU0[i], 1.0) : 1.0
            kt = path.SUt1[i, j, n] >= 0.0 ? 1.0 : 0.0

            if σ == 0.0 && ς == 0.0            # no slice moves: only the cutoffs can end matches
                e_s = _end_below!(es, pUs_prev[i, j], ps, ug.p, ks)
                e_t = _end_below!(et, pUt_prev[i, j], pt, ug.p, kt)
            else
                mv_u = σ * uU[i, j]
                uU[i, j] -= mv_u;  tU[i, j] += mv_u;  swept += mv_u
                @. mv_s = σ * es;  @. mv_t = ς * et
                es .-= mv_s;  et .-= mv_t
                e_s = _end_below!(es, pUs_prev[i, j], ps, ug.p, ks) +
                      _end_below!(mv_t, pUt_prev[i, j], ps, ug.p, ks)
                e_t = _end_below!(et, pUt_prev[i, j], pt, ug.p, kt) +
                      _end_below!(mv_s, pUs_prev[i, j], pt, ug.p, kt)
                es .+= mv_t;  et .+= mv_s
            end
            uU[i, j] += e_s;  tU[i, j] += e_t
            ended += e_s;  quit += e_t
        end
        path.ended_U[n] = ended;  path.quit_T[n] = quit;  path.swept_T[n] = swept
        path.ended_S[n] = _end_skilled_below!(eS, uS, mS, d, pS_prev, view(path.pstar_S, :, n),
                                              model.skl_grids)

        # ── Stock record at date n ───────────────────────────────────────
        wnum = 0.0;  wden = 0.0
        fill!(view(path.eUsp, :, :, n), 0.0);  fill!(view(path.eUtp, :, :, n), 0.0)
        @inbounds for j in 1:Nx, i in 1:Nx
            path.uU[i, j, n] = uU[i, j];  path.tU[i, j, n] = tU[i, j]
            path.uS[i, j, n] = uS[i, j];  path.mS[i, j, n] = mS[i, j]
            es = view(Es, i, j, :);  et = view(Et, i, j, :)
            path.eUs[i, j, n] = sum(es);  path.eUt[i, j, n] = sum(et)
            for q in 1:(NpU + 1)
                path.eUsp[i, q, n] += es[q];  path.eUtp[i, q, n] += et[q]
                y = βU * PU * x[i] * (q <= NpU ? ug.p[q] : 1.0)
                ms = es[q];  mt = et[q]
                ms > 0.0 && (wnum += ms * log(max(y + (1.0 - βU) * path.zUs[i, j, n], 1e-14));  wden += ms)
                mt > 0.0 && (wnum += mt * log(max(y + (1.0 - βU) * path.zUt[i, j, n], 1e-14));  wden += mt)
            end
        end
        path.eS[:, :, n] .= eS
        path.wU_mean[n] = wden > 1e-14 ? wnum / wden : 0.0
        n == Nt && break

        # ── Laws of motion, date n → n+1 (spec L1–L5) ───────────────────
        # At the date's rates and policies, in equal sub-steps of at most FORWARD_SUBSTEP.
        Δ    = path.tgrid[n + 1] - path.tgrid[n]
        nsub = max(ceil(Int, Δ / FORWARD_SUBSTEP - 1e-9), 1)
        dt   = Δ / nsub
        κU = meeting_rate(path.θU[n], up.μ, up.η)
        κS = meeting_rate(path.θS[n], model.skl_par.μ, model.skl_par.η)
        for _ in 1:nsub
            copyto!(uS_n, uS);  copyto!(mS_n, mS);  copyto!(tU_n, tU)
            _advance_unskilled_cells!(uU, tU, Es, Et, uS_n, d, κU, χU, path, model, wG, n, dt)
            _advance_skilled_masses!(uS, mS, tU_n, uS_n, mS_n, d, κU, κS, χU, model,
                                     view(path.pstar_S, :, n), dt)
            _advance_skilled_shape!(eS, uS_n, mS_n, mS, d, κS, model, view(path.pstar_S, :, n),
                                    view(path.poj, :, n), view(path.s_part, :, n), dt)
        end

        copyto!(τprev, view(path.τT, :, :, n))
        copyto!(pUs_prev, view(path.pUs, :, :, n));  copyto!(pUt_prev, view(path.pUt, :, :, n))
        copyto!(pS_prev, view(path.pstar_S, :, n))
    end
    return nothing
end

"""
    _advance_unskilled_cells!(uU, tU, Es, Et, uS, d, κU, χU, path, model, wG, n, dt)

One explicit step of every cell's unskilled stocks at date n's rates: births split by τ,
hires out of the hired branch of u_U and out of the crossers, each slice's quality measure,
and each slice's separations, into u_U from the searching slice and into training from the
training slice.  Rows are independent, so they are threaded.
"""
function _advance_unskilled_cells!(uU::Matrix{Float64}, tU::Matrix{Float64},
                                   Es::Array{Float64,3}, Et::Array{Float64,3},
                                   uS::Matrix{Float64}, d::Matrix{Float64}, κU::Float64,
                                   χU::Vector{Float64}, path::TransitionPath, model::Model,
                                   wG::Vector{Float64}, n::Int, dt::Float64)
    Nx = size(uU, 1);  ug = model.unsk_grids;  NpU = length(ug.p)
    up = model.unsk_par;  ν = model.common.ν;  φ = model.common.φ
    W2 = model.grids.copula.W2
    @threads for i in 1:Nx
        # Cells of a row share the searching slice's cutoff wherever they search at every
        # later date, so its redraw weights are formed once per row and per cutoff.
        ws = zeros(Float64, NpU);  wt = zeros(Float64, NpU)
        ps_row = path.pstar_U[i, n]
        rs = _redraw_weights!(ws, ps_row, ug.p, wG, up.α_U)
        @inbounds for j in 1:Nx
            τ  = clamp(path.τT[i, j, n], 0.0, 1.0)
            hU = κU * hired_unemployed(uU[i, j], χU[i], τ, W2[i, j])
            # Crossers hired into unskilled jobs join the slice of their type.
            hX = d[i, j] * κU * χU[i] * uS[i, j]
            ωX = _enclosed_crossers(d[i, j], τ)
            ps = path.pUs[i, j, n]
            ps == ps_row || (rs = _redraw_weights!(ws, ps, ug.p, wG, up.α_U);  ps_row = ps)
            Ds = _advance_unskilled!(view(Es, i, j, :), hU + (1.0 - ωX) * hX, ws, rs, wG,
                                     up.λ, up.ξ, ν, dt)
            et = view(Et, i, j, :)
            Dt = (ωX == 0.0 && sum(et) == 0.0) ? 0.0 :
                 _advance_unskilled!(et, ωX * hX, wt,
                                     _redraw_weights!(wt, path.pUt[i, j, n], ug.p, wG, up.α_U),
                                     wG, up.λ, up.ξ, ν, dt)
            uU[i, j] += dt * ((1.0 - τ) * ν * W2[i, j] + Ds - hU - ν * uU[i, j])
            tU[i, j] += dt * (τ * ν * W2[i, j] + Dt - (φ + ν) * tU[i, j])
        end
    end
    return nothing
end

"""
    _end_skilled_below!(eS, uS, mS, d, pold, pnew, sg) -> ended mass

End every skilled match a cutoff moving from `pold` to `pnew` places below reservation
(spec L6b).  ê is the per-aS unit shape of the non-draining types, so the ended unit mass
of a column moves each cell's (1−d)·m_S share of it into that cell's u_S.
"""
function _end_skilled_below!(eS::Matrix{Float64}, uS::Matrix{Float64}, mS::Matrix{Float64},
                             d::Matrix{Float64}, pold::AbstractVector{Float64},
                             pnew::AbstractVector{Float64}, sg::SkilledGrids)
    Nx = size(eS, 1);  Np = length(sg.p);  ended = 0.0
    @inbounds for j in 1:Nx
        δ = 0.0
        for q in 1:Np
            k = _kept_share(sg.p[q], clamp01(pold[j]), clamp01(pnew[j]), sg.p, q, Np)
            δ += (1.0 - k) * eS[j, q] * sg.wp[q];  eS[j, q] *= k
        end
        δ > 0.0 || continue
        for i in 1:Nx
            m = (1.0 - d[i, j]) * mS[i, j] * δ
            uS[i, j] += m;  ended += m
        end
    end
    return ended
end

"""
    _advance_skilled_masses!(uS, mS, tU, uS_n, mS_n, d, κU, κS, χU, model, pstar, dt)

One step of `u_S` and `m_S` from the date-n stocks: training completions in, the
destruction hazard on the employed, and the outflow branched on the crossing choice.  The
destruction hazard uses the SHOCK mass below reservation and the hiring outflow the
OFFER mass above it (`_skilled_margin_masses`); a crosser leaves at the unskilled hiring
rate of her cell, κ_U χ.
"""
function _advance_skilled_masses!(uS::Matrix{Float64}, mS::Matrix{Float64}, tU::Matrix{Float64},
                                  uS_n::Matrix{Float64}, mS_n::Matrix{Float64},
                                  d::Matrix{Float64}, κU::Float64, κS::Float64,
                                  χU::Vector{Float64}, model::Model,
                                  pstar::AbstractVector{Float64}, dt::Float64)
    Nx = size(uS, 1);  cp = model.common;  sp = model.skl_par
    acc_o, below_s = _skilled_margin_masses(model, pstar)
    @inbounds for j in 1:Nx
        hire_S = κS * acc_o[j]
        δS_end = sp.ξ + sp.λ * below_s[j]
        for i in 1:Nx
            cross = d[i, j] * κU * χU[i] * uS_n[i, j]
            uS[i, j] = uS_n[i, j] + dt * (cp.φ * tU[i, j] + δS_end * (mS_n[i, j] - uS_n[i, j]) -
                                          (cp.ν + (1.0 - d[i, j]) * hire_S) * uS_n[i, j] - cross)
            mS[i, j] = mS_n[i, j] + dt * (cp.φ * tU[i, j] - cp.ν * mS_n[i, j] - cross)
        end
    end
    return nothing
end

"""
    _advance_skilled_shape!(eS, uS_n, mS_n, mS_next, d, κS, model, pstar, poj, part, dt)

One step of the per-aS unit density ê_S(aS,p), the time-dependent form of the stationary
balance (notes eq:eSbalance).  Above the reservation quality a cell gains hires out of
unemployment `κ_S γ_o(p) û`, redraws `λ_S γ_s(p) ∫ê` and poached workers from below
`κ_S γ_o(p) ∫_{p*}^p s* ê`, and loses `(ν + λ_S + ξ_S + s* κ_S Ω_o(p)) ê`, Ω_o(p) the offer
mass strictly above the cell.  The step is taken in levels, `e = ê m₀` with m₀ the
non-draining column mass, and divided by the column mass it reaches, so `û + ∫ê = 1`
holds at every date: the `u_S` and `m_S` steps are the same balance in levels.  Mass
below the cutoff was ended by the reallocation and stays zero.
"""
function _advance_skilled_shape!(eS::Matrix{Float64}, uS_n::Matrix{Float64},
                                 mS_n::Matrix{Float64}, mS_next::Matrix{Float64},
                                 d::Matrix{Float64}, κS::Float64, model::Model,
                                 pstar::AbstractVector{Float64}, poj::AbstractVector{Float64},
                                 part::AbstractVector{Float64}, dt::Float64)
    Nx = size(eS, 1);  NpS = length(model.skl_grids.p)
    sp = model.skl_par;  sg = model.skl_grids;  pre = model.skl_pre
    ν = model.common.ν;  λS = sp.λ;  ξS = sp.ξ
    m0_now  = nondrain_column_mass(d, mS_n)
    m0_next = nondrain_column_mass(d, mS_next)
    u0_now  = nondrain_column_mass(d, uS_n)

    @inbounds for j in 1:Nx
        ufrac   = m0_now[j] > 1e-14 ? clamp(u0_now[j] / m0_now[j], 0.0, 1.0) : 0.0
        rescale = m0_next[j] > 1e-14 ? m0_now[j] / m0_next[j] : 1.0
        pstS = clamp01(pstar[j]);  pojS = clamp01(poj[j])
        j0   = max(pcut_index(sg.p, pstS) - 1, 1)       # the straddling cell is covered by ω
        e_tot = 0.0
        for jp in j0:NpS
            e_tot += eS[j, jp] * sg.wp[jp]
        end
        cum_seek = 0.0                                  # ∫_{p*}^p s* ê dp'
        for jp in 1:NpS
            e_old = eS[j, jp]
            if jp < j0
                eS[j, jp] = 0.0
                continue
            end
            pj = sg.p[jp]
            ω_res = _soft_weight(pj, pstS, sg.p, jp, NpS)
            s_ojs = _ojs_share(part[j], pj, pojS, sg.p, jp, NpS)
            acc_mass = jp < NpS ? pre.tail_weights[jp + 1] : 0.0
            inflow  = ω_res * (κS * pre.γvals[jp] * (ufrac + cum_seek) + λS * pre.γs_vals[jp] * e_tot)
            outflow = (ν + ξS + λS + s_ojs * κS * acc_mass) * e_old
            eS[j, jp] = max(e_old + dt * (inflow - outflow), 0.0) * rescale
            cum_seek += s_ojs * e_old * sg.wp[jp]
        end
    end
    return nothing
end

"""
    _skilled_margin_masses(model, pstar) -> (accept_offer, shock_below)

The two reservation margins of the skilled block, per aS: the OFFER mass a worker
accepts, `∫ ω dΓ_o`, and the SHOCK mass that lands below reservation and destroys the
match, `1 − ∫ ω dΓ_s`, integrated from the soft cutoff with the coverage the surplus uses,
as `solve_stationary_skilled!` integrates them.  Reading `Γ` at the node instead would
relocate the cutoff to the nearest node below.
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

Unit unemployed fraction `û(aS)` of the non-draining types at date `n`; the draining mass
is unemployed in the OTHER market and is counted there, in the unskilled seeker pool.
"""
function _nondrain_unemp_fraction(path::TransitionPath, dmat::Matrix{Float64},
                                  n::Int, Nx::Int)
    m0 = nondrain_column_mass(dmat, view(path.mS, :, :, n))
    u0 = nondrain_column_mass(dmat, view(path.uS, :, :, n))
    return [m0[j] > 1e-14 ? clamp(u0[j] / m0[j], 0.0, 1.0) : 0.0 for j in 1:Nx]
end

"""
    _d_matrix(path, model, n) -> Matrix

The date-n crossing fraction from the date-n branch values (`_crossing_fraction!`).
"""
_d_matrix(path::TransitionPath, model::Model, n::Int) =
    _crossing_fraction!(zeros(Float64, length(model.grids.x), length(model.grids.x)),
                        view(path.U1, :, :, n), view(path.U0, :, :, n), model.grids.x)


# ════════════════════════════════════════════════════════════
#  Step 4: Free entry at every date
# ════════════════════════════════════════════════════════════

"""
    _free_entry_tightness(path, model_z1) -> (θU, θS)

The tightness free entry implies at every date (spec F1–F3): θ_j = (μ_j 𝒥_j / k_j)² at
η_j = 1/2, from the date's values and its masses after the reallocation.

Unskilled: a vacancy meets every seeker in `u_U + d u_S` and is filled by those of them an
unskilled match forms with, the hired branch `u_U^h` and the crossers' share χ, at the
firm value `(1−β_U) S(a,1)` of the slice the hire joins: the searching slice, and for the
crossers the frontier encloses (`_advance_unskilled_cells!`) the training slice.  Skilled: `compute_Jbar_skilled`
on the date's composition, with the destination values of the path installed as both
firm values, `J = (1−β_S)·ω[s S¹ + (1−s) S⁰]⁺`, the object `skilled_inner_loop!` hands it.
"""
function _free_entry_tightness(path::TransitionPath, model::Model)
    Nt = length(path.tgrid);  Nx = length(model.grids.x)
    up = model.unsk_par;  sc = model.skl_cache;  W2 = model.grids.copula.W2
    χU = hire_fraction(model)
    k_π = up.k * mean_output_U(model)
    share_S = 1.0 - model.skl_par.β
    θU = similar(path.θU);  θS = similar(path.θS)

    for n in 1:Nt
        d = _d_matrix(path, model, n)
        Useek = 0.0;  Jnum = 0.0
        @inbounds for j in 1:Nx, i in 1:Nx
            u  = path.uU[i, j, n];  uS = path.uS[i, j, n]
            τ  = clamp(path.τT[i, j, n], 0.0, 1.0)
            uh = hired_unemployed(u, χU[i], τ, W2[i, j])
            ωX = _enclosed_crossers(d[i, j], τ)
            cross = χU[i] * d[i, j] * uS
            Useek += u + d[i, j] * uS
            Jnum  += (1.0 - up.β) * path.SUs1[i, j, n] * (uh + (1.0 - ωX) * cross)
            ωX > 0.0 && (Jnum += (1.0 - up.β) * max(path.SUt1[i, j, n], 0.0) * ωX * cross)
        end
        θU[n] = (Jnum < 1e-14 || Useek < 1e-14) ? 1e-14 :
                clamp(theta_from_q(k_π * Useek / Jnum, up.μ, up.η), 1e-14, 100.0)

        sc.d      .= d
        sc.m_S    .= @view path.mS[:, :, n]
        sc.pstar  .= @view path.pstar_S[:, n]
        sc.poj    .= @view path.poj[:, n]
        sc.s_part .= @view path.s_part[:, n]
        sc.u_frac .= _nondrain_unemp_fraction(path, d, n, Nx)
        sc.e_frac .= @view path.eS[:, :, n]
        @inbounds for k in 1:Nx, q in axes(sc.J0, 2)
            sc.J0[k, q] = share_S * path.Sd[k, q, n];  sc.J1[k, q] = sc.J0[k, q]
        end
        θS[n] = update_theta_skilled(model)
    end
    return θU, θS
end


# ════════════════════════════════════════════════════════════
#  Public entry point
# ════════════════════════════════════════════════════════════

"""
    solve_transition(model_z0, model_z1, tp; scenario) -> TransitionResult

Backward–forward transition from stationary equilibrium `model_z0` to `model_z1` under the
post-switch (`z₁`) parameters, after an unanticipated change in the parameters at date 0.
Both models must share the ability grid, its weights and both quality grids.  Iterates
until the free-entry gap `max_n |ln θ^new − ln θ|` over both markets is below `tp.tol`.

CONSUMES ITS MODELS.  Free entry installs each date's composition into `model_z1`'s
skilled cache, so on return it holds the terminal date's path state, not the stationary
solution; `_init_path!` takes the base state from `model_z0`'s caches.  A second pair
needs fresh models.
"""
solve_transition(model_z0::Model, model_z1::Model, tp::TransitionParams;
                 scenario::Symbol = :unnamed) =
    solve_transition_path(model_z0, model_z1, tp; scenario = scenario).result

"""
    solve_transition_path(model_z0, model_z1, tp; scenario, guess) -> (result, path)

`solve_transition`, also returning the path.  `guess = (tgrid, θU, θS)` starts the
tightness paths from an earlier solution, linearly interpolated onto `tp`'s dates and held
at its last value beyond them; without it the first guess is z₁'s tightness.
"""
function solve_transition_path(model_z0::Model, model_z1::Model, tp::TransitionParams;
                               scenario::Symbol = :unnamed, guess = nothing)
    POJ_SCORE[] === :raw ||
        error("solve_transition: the path scores on-the-job search by the gain against the " *
              "envelope, whose steady-state value is the :raw score; POJ_SCORE = $(POJ_SCORE[]) " *
              "has no path counterpart.")
    g0 = model_z0.grids;  g1 = model_z1.grids
    (g0.x == g1.x && g0.copula.W2 == g1.copula.W2 &&
     model_z0.unsk_grids.p == model_z1.unsk_grids.p && model_z0.skl_grids.p == model_z1.skl_grids.p) ||
        error("solve_transition: z₀ and z₁ must share the ability grid, its population weights " *
              "and both match-quality grids; the ability distribution is held between the windows.")
    path = allocate_path(model_z1, tp)
    _init_path!(path, model_z0, model_z1)
    if guess !== nothing
        for (n, t) in enumerate(path.tgrid)
            k = clamp(searchsortedlast(guess.tgrid, t), 1, length(guess.tgrid) - 1)
            w = clamp((t - guess.tgrid[k]) / (guess.tgrid[k + 1] - guess.tgrid[k]), 0.0, 1.0)
            path.θU[n] = (1.0 - w) * guess.θU[k] + w * guess.θU[k + 1]
            path.θS[n] = (1.0 - w) * guess.θS[k] + w * guess.θS[k + 1]
        end
    end
    return (result = _iterate_transition!(path, model_z1, tp, scenario), path = path)
end

"""
    _iterate_transition!(path, model_z1, tp, scenario) -> TransitionResult

The outer fixed point on the tightness paths from the path's current guess (Steps 2–5).
"""
function _iterate_transition!(path::TransitionPath, model_z1::Model, tp::TransitionParams,
                              scenario::Symbol)
    converged = false;  gap = Inf;  it = 0
    for outer_it in 1:tp.maxit
        it = outer_it
        bwd = _backward_pass!(path, model_z1, tp)
        _forward_pass!(path, model_z1, tp)
        θU_new, θS_new = _free_entry_tightness(path, model_z1)
        gU  = maximum(abs, log.(θU_new) .- log.(path.θU))
        gS  = maximum(abs, log.(θS_new) .- log.(path.θS))
        gap = max(gU, gS)

        if tp.verbose && (outer_it == 1 || outer_it % 10 == 0)
            @printf("[transition it=%d]  free-entry gap %.3e  (U %.3e  S %.3e)  backward: %d passes, resid %.2e\n",
                    outer_it, gap, gU, gS, bwd.iters, bwd.resid);  flush(stdout)
        end
        if gap < tp.tol
            converged = true
            tp.verbose && @printf("[transition]  converged it=%d  gap=%.3e\n", outer_it, gap)
            break
        end
        @. path.θU = (1.0 - tp.damp) * path.θU + tp.damp * θU_new
        @. path.θS = (1.0 - tp.damp) * path.θS + tp.damp * θS_new
    end
    return _build_result(path, model_z1, tp, scenario, converged, it, gap)
end


# ════════════════════════════════════════════════════════════
#  Build serialisable result
# ════════════════════════════════════════════════════════════

"""
    _build_result(path, model_z1, tp, scenario, converged, n_iter, final_dist)

Collapse the path into aggregate series and ability-marginal profiles, every one read off
the date's masses (no stationary identity).  Mean wages are mean LOG wages, the units of the
data moments `mean_wage_U/S`, of the rebargained wages of the spec (V8),
`β·output + (1−β) z`, with z the date's flow value of unemployment: unskilled over each
slice's quality measure (recorded by the forward pass), skilled over ê_S with the OJS wage
on the searching share.
"""
function _build_result(path::TransitionPath, model_z1::Model, tp::TransitionParams,
                       scenario::Symbol, converged::Bool, n_iter::Int, final_dist::Float64)
    Nt  = length(path.tgrid);  Nx = length(model_z1.grids.x);  NpS = length(model_z1.skl_grids.p)
    cp = model_z1.common;  up = model_z1.unsk_par;  sp = model_z1.skl_par
    gp = model_z1.grids;   sg = model_z1.skl_grids;  pre = model_z1.skl_pre
    W2 = gp.copula.W2;  wpS = sg.wp
    PS = exp(cp.A) * sp.PS;  σS = sp.σ * exp(cp.A)
    wΓo = pre.γvals .* sg.wp

    κU_p  = zeros(Nt);  κS_p  = zeros(Nt)
    urU_p = zeros(Nt);  urS_p = zeros(Nt);  urT_p = zeros(Nt)
    skS_p = zeros(Nt);  trS_p = zeros(Nt);  wS_p = zeros(Nt)
    ffl_p = zeros(Nt);  drg_p = zeros(Nt);  dms_p = zeros(Nt);  dfl_p = zeros(Nt)
    uU_prof = zeros(Nx, Nt);  tU_prof = zeros(Nx, Nt)
    uS_prof = zeros(Nx, Nt);  mS_prof = zeros(Nx, Nt)
    χU = hire_fraction(model_z1)

    for n in 1:Nt
        κU_p[n] = meeting_rate(path.θU[n], up.μ, up.η)
        κS_p[n] = meeting_rate(path.θS[n], sp.μ, sp.η)
        dmat     = _d_matrix(path, model_z1, n)
        ffl_p[n] = _frontier_floor(path, model_z1, n)
        drg_p[n] = sum(dmat .* W2)
        dms_p[n] = sum(dmat .* (@view path.mS[:, :, n]))
        dfl_p[n] = κU_p[n] * sum(dmat .* χU .* (@view path.uS[:, :, n]))

        agg_uU = sum(@view path.uU[:, :, n]);  agg_tU = sum(@view path.tU[:, :, n])
        agg_uS = sum(@view path.uS[:, :, n]);  agg_mS = sum(@view path.mS[:, :, n])
        agg_eU = sum(@view path.eUs[:, :, n]) + sum(@view path.eUt[:, :, n])
        lf_U = agg_uU + agg_eU;  lf_total = lf_U + agg_mS;  pop = sum(W2)
        urU_p[n] = lf_U > 1e-14 ? agg_uU / lf_U : 0.0
        urS_p[n] = agg_mS > 1e-14 ? agg_uS / agg_mS : 0.0
        urT_p[n] = lf_total > 1e-14 ? (agg_uU + agg_uS) / lf_total : 0.0
        skS_p[n] = lf_total > 1e-14 ? agg_mS / lf_total : 0.0
        trS_p[n] = pop > 1e-14 ? agg_tU / pop : 0.0

        @inbounds for i in 1:Nx
            uU_prof[i, n] = sum(@view path.uU[i, :, n]);  tU_prof[i, n] = sum(@view path.tU[i, :, n])
        end
        @inbounds for j in 1:Nx
            uS_prof[j, n] = sum(@view path.uS[:, j, n]);  mS_prof[j, n] = sum(@view path.mS[:, j, n])
        end

        # Skilled wages: w⁰ = β·output + (1−β) z_S, and on the searching share
        # w¹ = w⁰ + (1−β)σ_S − β(1−β) κ_S ∫_p S dΓ.
        m0 = nondrain_column_mass(dmat, view(path.mS, :, :, n))
        wnum = 0.0;  wden = 0.0
        @inbounds for j in 1:Nx
            tail = 0.0;  tails = zeros(NpS)
            for q in NpS:-1:1
                tail += path.Sd[j, q, n] * wΓo[q];  tails[q] = tail
            end
            for q in 1:NpS
                e = path.eS[j, q, n] * m0[j] * wpS[q]
                e <= 1e-14 && continue
                s  = _ojs_share(path.s_part[j, n], sg.p[q], clamp01(path.poj[j, n]), sg.p, q, NpS)
                w0 = sp.β * PS * gp.x[j] * sg.p[q] + (1.0 - sp.β) * path.zS[j, n]
                w1 = w0 + (1.0 - sp.β) * σS - sp.β * (1.0 - sp.β) * κS_p[n] * tails[q]
                wnum += e * (s * log(max(w1, 1e-14)) + (1.0 - s) * log(max(w0, 1e-14)));  wden += e
            end
        end
        wS_p[n] = wden > 1e-14 ? wnum / wden : 0.0
    end

    return TransitionResult(
        scenario, converged, n_iter, final_dist,
        copy(path.tgrid), copy(path.θU), copy(path.θS), κU_p, κS_p,
        urU_p, urS_p, urT_p, skS_p, trS_p, copy(path.wU_mean), wS_p,
        uU_prof, tU_prof, uS_prof, mS_prof,
        copy(gp.wa_U), copy(gp.x), copy(sg.p), copy(wpS),
        ffl_p, drg_p, dms_p, dfl_p,
    )
end

"""
    _frontier_floor(path, model, n) -> Float64

Lowest point of the training frontier at date n: the skilled ability at which the least
unskilled-able worker is indifferent between training and search, the crossing of
`−c(aS) + T` against the searching slice's value in the bottom aU row, by the
interpolation `_training_fraction!` uses.  Returns the grid ends when the frontier leaves
the grid.
"""
function _frontier_floor(path::TransitionPath, model::Model, n::Int)
    x = model.grids.x;  Nx = length(x)
    Utr = view(path.Utr, 1, :, n);  C = view(path.UC, 1, :, n)
    Utr[Nx] < C[Nx] && return x[Nx]
    Utr[1] >= C[1]  && return x[1]
    j0 = 2
    while j0 < Nx && Utr[j0] < C[j0]
        j0 += 1
    end
    den = (Utr[j0] - Utr[j0 - 1]) - (C[j0] - C[j0 - 1])
    return den > 0.0 ? x[j0 - 1] + (C[j0 - 1] - Utr[j0 - 1]) * (x[j0] - x[j0 - 1]) / den : x[j0]
end


# ════════════════════════════════════════════════════════════
#  Path statistics
# ════════════════════════════════════════════════════════════

"""
    path_statistics(path, model_z1) -> NamedTuple of length-Nt vectors

The flows between states at every date and the rate moments measured on them.  Each date
is read from its masses after its reallocation and its own rates (per month); no
steady-state identity is used (spec, Path statistics).

Flows are the laws of motion's own terms: hires out of unskilled unemployment, crossers'
hires, separations by slice, entries into and completions of training, skilled hires,
separations and job-to-job moves; the date's reallocations (ended matches, quits and
sweeps into training) are the masses `_forward_pass!` recorded.  The rate moments are
measured as `compute_equilibrium_objects` measures them, cell by cell from the date's
hazards: the mean monthly transition probability over the moment's universe as a hazard,
and the job-to-job rate as the monthly equivalent of the quarterly move probability.
Duration, survival and wage-dispersion moments have no counterpart off the steady state in
this form and are not reported.
"""
function path_statistics(path::TransitionPath, model::Model)
    Nt = length(path.tgrid);  Nx = length(model.grids.x)
    cp = model.common;  up = model.unsk_par;  sp = model.skl_par
    sg = model.skl_grids;  pre = model.skl_pre;  W2 = model.grids.copula.W2
    NpS = length(sg.p);  χU = hire_fraction(model)
    z() = zeros(Nt)
    hires_U = z();  hires_X = z();  sep_U = z();  sep_T = z();  train_in = z();  train_done = z()
    hires_S = z();  sep_S = z();  ee_S = z()
    jfr_U = z();  jfr_S = z();  sep_rate_U = z();  sep_rate_S = z();  ee_rate_S = z()
    wchg_rate_U = z();  wchg_rate_S = z()
    hazard(pm, m) = m > 1e-14 ? -log1p(-pm / m) : 0.0

    for n in 1:Nt
        κU = meeting_rate(path.θU[n], up.μ, up.η);  κS = meeting_rate(path.θS[n], sp.μ, sp.η)
        prob_κU = -expm1(-κU)
        d  = _d_matrix(path, model, n)
        acc_o, below_s = _skilled_margin_masses(model, view(path.pstar_S, :, n))
        m0 = nondrain_column_mass(d, view(path.mS, :, :, n))
        # Probability-weighted masses of each moment's universe: (Σ prob × mass, Σ mass).
        pU = mU = pS = mSu = pσU = mE = pωU = pσS = mES = pωS = pee = mee = 0.0
        @inbounds for j in 1:Nx, i in 1:Nx
            τ  = clamp(path.τT[i, j, n], 0.0, 1.0)
            uU = path.uU[i, j, n];  uS = path.uS[i, j, n]
            hired = hired_unemployed(uU, χU[i], τ, W2[i, j])
            hires_U[n] += κU * hired;  pU += prob_κU * hired;  mU += uU
            cross = d[i, j] * uS
            hires_X[n] += κU * χU[i] * cross;  pS += χU[i] * prob_κU * cross;  mSu += cross
            stay = uS - cross
            hires_S[n] += κS * acc_o[j] * stay;  pS += -expm1(-κS * acc_o[j]) * stay;  mSu += stay
            for (pc, e, rec) in ((path.pUs[i, j, n], path.eUs[i, j, n], sep_U),
                                 (path.pUt[i, j, n], path.eUt[i, j, n], sep_T))
                G = G_cdf_unskilled(pc, up.α_U)
                δ = up.ξ + up.λ * G;  ω = up.λ * (1.0 - G)
                rec[n] += δ * e;  pσU += -expm1(-δ) * e;  pωU += -expm1(-ω) * e;  mE += e
            end
            train_in[n]   += τ * cp.ν * W2[i, j]
            train_done[n] += cp.φ * path.tU[i, j, n]
            eS = path.mS[i, j, n] - uS
            δS = sp.ξ + sp.λ * below_s[j];  ωS = sp.λ * (1.0 - below_s[j])
            sep_S[n] += δS * eS;  pσS += -expm1(-δS) * eS;  pωS += -expm1(-ωS) * eS;  mES += eS
        end
        @inbounds for j in 1:Nx, q in 1:NpS
            e = path.eS[j, q, n] * sg.wp[q] * m0[j]
            mee += e
            s = _ojs_share(path.s_part[j, n], sg.p[q], clamp01(path.poj[j, n]), sg.p, q, NpS)
            s <= 0.0 && continue
            ee_S[n] += s * κS * (q < NpS ? pre.tail_weights[q + 1] : 0.0) * e
            pee += s * e * -expm1(-3.0 * κS * (1.0 - pre.Γvals[q]))
        end
        jfr_U[n]       = mU > 1e-14 ? hazard(pU, mU) : κU
        jfr_S[n]       = mSu > 1e-14 ? hazard(pS, mSu) : κS
        sep_rate_U[n]  = hazard(pσU, mE);   wchg_rate_U[n] = hazard(pωU, mE)
        sep_rate_S[n]  = hazard(pσS, mES);  wchg_rate_S[n] = hazard(pωS, mES)
        ee_rate_S[n]   = mee > 1e-14 ? 1.0 - (1.0 - pee / mee)^(1.0 / 3.0) : 0.0
    end
    return (hires_U = hires_U, hires_cross = hires_X, sep_U = sep_U, sep_U_into_training = sep_T,
            train_entry_births = train_in, train_completions = train_done,
            hires_S = hires_S, sep_S = sep_S, ee_S = ee_S,
            ended_U = copy(path.ended_U), quit_into_training = copy(path.quit_T),
            swept_into_training = copy(path.swept_T), ended_S = copy(path.ended_S),
            jfr_U = jfr_U, jfr_S = jfr_S, sep_rate_U = sep_rate_U, sep_rate_S = sep_rate_S,
            ee_rate_S = ee_rate_S, wchg_rate_U = wchg_rate_U, wchg_rate_S = wchg_rate_S)
end

# A covered fraction that moves by less than this between dates has not changed direction:
# a steady state holds its fractions to about 1e-8 of a cell (the stationary solve's own
# tolerance), and a millionth of a cell is above that and below any mass that matters.
const REVERSAL_TOL = 1e-6

# Directions a series takes, each opened by a move of more than `tol` away from the extreme
# reached in the current direction; a reversal is every direction after the first.
function _directions(seq::AbstractVector{Float64}, tol::Float64)
    n_dirs = 0;  dir = 0;  ext = seq[1]
    @inbounds for v in seq
        if dir != 1 && v - ext > tol
            n_dirs += 1;  dir = 1;  ext = v
        elseif dir != -1 && ext - v > tol
            n_dirs += 1;  dir = -1;  ext = v
        elseif (dir == 1 && v > ext) || (dir == -1 && v < ext)
            ext = v
        end
    end
    return n_dirs
end

"""
    policy_reversals(path, model_z1) -> NamedTuple

How often each re-made choice changes direction along the path (spec A14).  For the crossing
choice: the cells whose covered fraction d is ever positive, the number of reversals (a change
of direction after the first change, counted on the covered fraction at `REVERSAL_TOL`), the
cells reversing, the node-level count (U_S^(1) > U_S^(0) switching back after its first
switch), and the skilled mass on the cells that reverse.  For training: node-level reversals
of `−c + T ≥ C`.  For on-the-job search: reversals of the participation weight.
"""
function policy_reversals(path::TransitionPath, model::Model)
    Nt = length(path.tgrid);  Nx = length(model.grids.x)
    D = [_d_matrix(path, model, n) for n in 1:Nt]
    reversals(seq) = max(_directions(seq, REVERSAL_TOL) - 1, 0)
    flips(b) = max(count(n -> b[n] != b[n - 1], 2:length(b)) - 1, 0)
    cells_d = 0;  cells_d_mass = 0;  d_rev = 0;  cells_rev = 0;  d_node_rev = 0
    rev_mS = 0.0;  rev_uS = 0.0;  τ_rev = 0
    for j in 1:Nx, i in 1:Nx
        seq = [D[n][i, j] for n in 1:Nt]
        any(>(0.0), seq) && (cells_d += 1)
        any(n -> D[n][i, j] > 0.0 && path.mS[i, j, n] > 1e-14, 1:Nt) && (cells_d_mass += 1)
        r = reversals(seq)
        if r > 0
            d_rev += r;  cells_rev += 1
            rev_mS = max(rev_mS, maximum(view(path.mS, i, j, :)))
            rev_uS = max(rev_uS, maximum(view(path.uS, i, j, :)))
        end
        d_node_rev += flips([path.U1[i, j, n] > path.U0[i, j, n] for n in 1:Nt])
        τ_rev      += flips([path.Utr[i, j, n] >= path.UC[i, j, n] for n in 1:Nt])
    end
    s_rev = sum(k -> reversals(view(path.s_part, k, :)), 1:Nx)
    return (cells_d = cells_d, cells_d_with_skilled_mass = cells_d_mass, d_reversals = d_rev,
            cells_d_reversing = cells_rev, d_node_reversals = d_node_rev,
            max_mS_on_reversing_cells = rev_mS, max_uS_on_reversing_cells = rev_uS,
            tau_node_reversals = τ_rev, s_part_reversals = s_rev)
end

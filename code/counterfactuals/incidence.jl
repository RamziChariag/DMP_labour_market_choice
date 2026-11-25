############################################################
# incidence.jl — who bears each crisis, by type, baseline state and training status
#
# Library file, loaded by counterfactuals_main.jl; `run_incidence` is Exercise 2.
#
# Everyone alive at the switch, weighted by the baseline steady state's masses m_x⁰(a)
# (before the switch). Each worker is valued at date 0 of the estimated transition in the
# state the date-0 reallocation puts her in, against her value in her baseline state:
#   ΔV(a) = (1/ℓ(a)) Σ_x m_x⁰(a) [V_x¹(a)/A_crisis − V_x⁰(a)/A_base],
# each window at its own A, and the same in levels. The unskilled destinations are the
# solver's own reallocation (`_reallocate!`, linear in the masses) applied to each baseline
# unskilled state alone: matches below the new cutoff end into u_U (searching slice) or into
# training (training slice), the frontier sweeps unemployed into training and moves matches
# between slices. Trained workers stay trained: the unemployed choose their market at date 0
# (U_S = max{U_S⁰, U_S¹}), and a skilled match is valued at E = U_S + βS⁺, which is U_S
# where the crisis destroys it. Values are those transition_main.jl saved (`types`).
#
# Training status compares the two steady states' frontiers. In a cell with training
# fractions τ⁰ (baseline) and τ¹ (crisis) the covered regions nest, so the baseline
# training side (trainees, the training slice, the trained) splits into "trains in both"
# min(τ¹/τ⁰, 1) and "stops", and the other side (searching slice) into "starts"
# max(τ¹ − τ⁰, 0)/(1 − τ⁰) and "trains in neither". The starts and stops are the starters
# and stoppers of `switchers` (cf_common.jl), the sets Exercises 1 and 3 report.
############################################################

# The baseline state classes, and those on the training side of the baseline frontier.
const INCIDENCE_STATES = (:unskilled_unemployed, :unskilled_employed, :unskilled_employed_will_train,
                          :training, :trained_unemployed, :skilled_employed)
const TRAINING_SIDE = (:unskilled_employed_will_train, :training, :trained_unemployed, :skilled_employed)
const TRAINING_STATUS = (:trains_both, :trains_neither, :starts, :stops)

# Masses at or below this are not valued (the transition's mass guard).
const INCIDENCE_MASS_FLOOR = 1e-14

"""
    _valued(m, v) -> Float64

m·v for a mass `m` in a state worth `v`; 0 for a mass at or below the floor. A mass above
it in a state without a value (NaN) is an error.
"""
function _valued(m::Real, v::Real)
    m > INCIDENCE_MASS_FLOOR || return 0.0
    isfinite(v) || error("incidence: mass $m in a state without a value.")
    return Float64(m) * Float64(v)
end

"""
    _date0_destinations(L, model, origin) -> forward state

The date-0 reallocation of the estimated transition `L` applied to the baseline mass of
one unskilled state alone (`origin` ∈ :uU, :Es, :Et): the mass left in u_U, the entries into
training (`nU`), and each slice's matches by p (`Es`, `Et`).
"""
function _date0_destinations(L, model::Model, origin::Symbol)
    st = _forward_state(L.path)
    origin === :uU || fill!(st.uU, 0.0)
    origin === :Es || fill!(st.Es, 0.0)
    origin === :Et || fill!(st.Et, 0.0)
    fill!(st.nU, 0.0)
    _reallocate!(st, L.path, model, hire_fraction(model), _d_matrix(L.path, model, 1), 1)
    return st
end

"""
    incidence_by_type(L) -> NamedTuple

Per type and baseline state class: the baseline mass and the sums Σ m V at the baseline
(`Vb`) and at date 0 (`V0`), in levels; the two steady states' training fractions; and
A_base, A_crisis.
"""
function incidence_by_type(L)
    ty = L.tr.types;  v0 = ty.values_date0;  vb = ty.values_base;  db = ty.density_base
    model = deepcopy(L.m1);  Nx = length(model.grids.x)
    base = _forward_state(L.path)
    dest = Dict(o => _date0_destinations(L, model, o) for o in (:uU, :Es, :Et))
    mass = Dict(s => zeros(Nx, Nx) for s in INCIDENCE_STATES)
    Vb = Dict(s => zeros(Nx, Nx) for s in INCIDENCE_STATES)
    V0 = Dict(s => zeros(Nx, Nx) for s in INCIDENCE_STATES)
    @inbounds for j in 1:Nx, i in 1:Nx
        # Date-0 value of an unskilled origin's destinations: u_U, training entries (−c + T),
        # the searching and the training slice's matches.
        reached(st) = _valued(st.uU[i, j], v0.UC[i, j]) + _valued(st.nU[i, j], v0.Utr[i, j]) +
                      sum(_valued(st.Es[i, j, q], v0.EUs_p[i, j, q]) +
                          _valued(st.Et[i, j, q], v0.EUt_p[i, j, q]) for q in axes(st.Es, 3))
        ij = CartesianIndex(i, j)
        mass[:unskilled_unemployed][ij] = base.uU[i, j]
        Vb[:unskilled_unemployed][ij]   = _valued(base.uU[i, j], vb.UC[i, j])
        V0[:unskilled_unemployed][ij]   = reached(dest[:uU])
        for (s, E, Ev, o) in ((:unskilled_employed, base.Es, vb.EUs_p, :Es),
                              (:unskilled_employed_will_train, base.Et, vb.EUt_p, :Et))
            mass[s][ij] = sum(view(E, i, j, :))
            Vb[s][ij]   = sum(_valued(E[i, j, q], Ev[i, j, q]) for q in axes(E, 3))
            V0[s][ij]   = reached(dest[o])
        end
        mass[:training][ij] = base.nU[i, j]
        Vb[:training][ij]   = _valued(base.nU[i, j], vb.T[i, j])
        V0[:training][ij]   = _valued(base.nU[i, j], v0.T[i, j])
        mu = Float64(db.uS_skilled[i, j] + db.uS_cross[i, j])
        mass[:trained_unemployed][ij] = mu
        Vb[:trained_unemployed][ij]   = _valued(mu, vb.US[i, j])
        V0[:trained_unemployed][ij]   = _valued(mu, v0.US[i, j])
        mass[:skilled_employed][ij] = sum(Float64, view(db.eS_p, i, j, :))
        Vb[:skilled_employed][ij]   = sum(_valued(db.eS_p[i, j, q], vb.ES_p[i, j, q]) for q in axes(db.eS_p, 3))
        V0[:skilled_employed][ij]   = sum(_valued(db.eS_p[i, j, q], v0.ES_p[i, j, q]) for q in axes(db.eS_p, 3))
    end
    return (mass = mass, Vb = Vb, V0 = V0, tau_base = copy(L.m0.unsk_cache.τT),
            tau_crisis = copy(L.m1.unsk_cache.τT), A_base = ty.A_base, A_crisis = ty.A_crisis)
end

"""
    status_shares(τ0, τ1) -> NamedTuple

The shares of a cell's baseline training side that train in both steady states and that
stop, and of its other side that start and that train in neither (see the file header):
the cell fractions of `switch_fractions` over the side each comes from, so the starts and
stops are the starters and stoppers of `switchers`. A side with no mass keeps the
convention that it all stops (τ0 = 0) or all trains in neither (τ0 = 1).
"""
function status_shares(τ0::Float64, τ1::Float64)
    f = switch_fractions(τ0, τ1);  τ0 = clamp(τ0, 0.0, 1.0)
    stops  = τ0 > 0.0 ? f.stops / τ0 : 1.0
    starts = τ0 < 1.0 ? f.starts / (1.0 - τ0) : 0.0
    return (trains_both = 1.0 - stops, stops = stops, starts = starts, trains_neither = 1.0 - starts)
end

"""
    run_incidence(crises)

For each crisis, writes
  incidence_<crisis>.csv            one row per type: population, the steady states' τ, ΔV over
                                    all states and by baseline state (per head of the state,
                                    over A and in levels), baseline masses, status shares
  incidence_by_status_<crisis>.csv  ΔV by training status in the two steady states, and overall
  incidence_status_aptitudes_<crisis>.csv   by training status, and overall: the mass (share
                                    of the population) and its mean a_U and mean a_S
  incidence_<crisis>.jls            the same as matrices on the ability grid
"""
function run_incidence(crises)
    dir = cf_log_start(out_counterfactuals(:incidence), "Exercise 2: who bears each crisis")
    for crisis in crises
        t0 = time()
        L  = load_transition(crisis)
        ib = incidence_by_type(L);  A0 = ib.A_base;  A1 = ib.A_crisis
        gp = L.m1.grids;  Nx = length(gp.x);  pop = gp.copula.W2
        dA = Dict(s => ib.V0[s] ./ A1 .- ib.Vb[s] ./ A0 for s in INCIDENCE_STATES)   # Σ m ΔV, over A
        dL = Dict(s => ib.V0[s] .- ib.Vb[s] for s in INCIDENCE_STATES)               # Σ m ΔV, levels
        per(x, m) = ifelse.(m .> INCIDENCE_MASS_FLOOR, x ./ m, NaN)
        mtot = sum(ib.mass[s] for s in INCIDENCE_STATES)
        dA_all = sum(values(dA));  dL_all = sum(values(dL))

        df = DataFrame(i = repeat(1:Nx, outer = Nx), j = repeat(1:Nx, inner = Nx),
                       a_U = repeat(gp.x, outer = Nx), a_S = repeat(gp.x, inner = Nx),
                       population = vec(pop), tau_base = vec(ib.tau_base), tau_crisis = vec(ib.tau_crisis),
                       diff_over_A_all = vec(per(dA_all, mtot)), diff_level_all = vec(per(dL_all, mtot)))
        for s in INCIDENCE_STATES
            df[!, Symbol("mass_", s)]        = vec(ib.mass[s])
            df[!, Symbol("diff_over_A_", s)] = vec(per(dA[s], ib.mass[s]))
            df[!, Symbol("diff_level_", s)]  = vec(per(dL[s], ib.mass[s]))
        end

        # Training status: each side of the baseline frontier split by the cell's τ⁰, τ¹.
        sm = Dict(g => zeros(Nx, Nx) for g in TRAINING_STATUS)
        sA = Dict(g => zeros(Nx, Nx) for g in TRAINING_STATUS)
        sL = Dict(g => zeros(Nx, Nx) for g in TRAINING_STATUS)
        for c in CartesianIndices(mtot)
            sh = status_shares(ib.tau_base[c], ib.tau_crisis[c])
            for s in INCIDENCE_STATES
                gs = s in TRAINING_SIDE ? (:trains_both, :stops) : (:starts, :trains_neither)
                for g in gs
                    w = getfield(sh, g)
                    sm[g][c] += w * ib.mass[s][c];  sA[g][c] += w * dA[s][c];  sL[g][c] += w * dL[s][c]
                end
            end
        end
        for g in TRAINING_STATUS
            df[!, Symbol("share_", g)] = vec(per(sm[g], mtot))
        end
        CSV.write(joinpath(dir, "incidence_$(crisis).csv"), df)

        M = sum(mtot)
        srows = [(crisis = String(crisis), status = String(g), mass = sum(sm[g]), population_share = sum(sm[g]) / M,
                  diff_over_A = sum(sA[g]) / sum(sm[g]), diff_level = sum(sL[g]) / sum(sm[g]))
                 for g in TRAINING_STATUS]
        push!(srows, (crisis = String(crisis), status = "all", mass = M, population_share = 1.0,
                      diff_over_A = sum(dA_all) / M, diff_level = sum(dL_all) / M))
        CSV.write(joinpath(dir, "incidence_by_status_$(crisis).csv"), DataFrame(srows))

        # The aptitudes of each status group, weighted by its baseline masses.
        arows = [merge((crisis = String(crisis), status = String(g)), set_summary(sm[g], gp.x, sum(pop)))
                 for g in TRAINING_STATUS]
        push!(arows, merge((crisis = String(crisis), status = "all"), set_summary(mtot, gp.x, sum(pop))))
        CSV.write(joinpath(dir, "incidence_status_aptitudes_$(crisis).csv"), DataFrame(arows))

        open(joinpath(dir, "incidence_$(crisis).jls"), "w") do io
            serialize(io, (crisis = crisis, version = ROYSEARCH_VERSION, point = POINT_ESTIMATE,
                           x = gp.x, population = pop, A_base = A0, A_crisis = A1,
                           tau_base = ib.tau_base, tau_crisis = ib.tau_crisis,
                           mass_by_state = ib.mass, value_sum_base = ib.Vb, value_sum_date0 = ib.V0,
                           diff_over_A_all = per(dA_all, mtot), diff_level_all = per(dL_all, mtot),
                           status_mass = sm, by_status = DataFrame(srows),
                           status_aptitudes = DataFrame(arows)))
        end
        cf_log(dir, @sprintf("[%s] ΔV/A per head %+.5f (types %+.5f to %+.5f); baseline mass vs population, largest cell gap %.1e; %.1f s",
                             crisis, sum(dA_all) / M, minimum(filter(isfinite, per(dA_all, mtot))),
                             maximum(filter(isfinite, per(dA_all, mtot))), maximum(abs, mtot .- pop), time() - t0))
        for r in srows
            cf_log(dir, @sprintf("[%s] %-14s share %.4f, ΔV/A %+.5f, ΔV %+.4f", crisis, r.status,
                                 r.population_share, r.diff_over_A, r.diff_level))
        end
        cf_log(dir, "[$crisis] mean (a_U, a_S): " *
                    join((@sprintf("%s (%.4f, %.4f)", r.status, r.mean_aU, r.mean_aS) for r in arows), ", "))
        sw = switchers(L.m0, L.m1)
        cf_log(dir, @sprintf("[%s] starts and stops are Exercise 1's starters and stoppers: mass %.12f and %.12f against %.12f and %.12f (gaps %.1e, %.1e)",
                             crisis, sum(sm[:starts]), sum(sm[:stops]), sum(sw.starters), sum(sw.stoppers),
                             abs(sum(sm[:starts]) - sum(sw.starters)), abs(sum(sm[:stops]) - sum(sw.stoppers))))
    end
    return nothing
end

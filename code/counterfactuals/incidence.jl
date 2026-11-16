############################################################
# incidence.jl — who gains and who loses from each crisis, by type and state (RoySearch)
#
# LIBRARY FILE — loaded by counterfactuals_main.jl; `run_incidence` is the exercise.
#
# For every type (a_U, a_S) and every state the type can be in, the value at date 0 of the
# estimated transition against the value in the same state at the baseline steady state,
# relative to general productivity:  (V_0 / A_crisis) / (V_base / A_base) − 1.
# States: unskilled unemployed (searching), unskilled employed at each p (searching slice,
# and the training slice of workers who train when the match ends), in training, trained
# and unemployed searching the skilled market, trained and searching the unskilled market,
# skilled employed at each p.  Worker values follow the Nash split, E = outside option +
# β S⁺.  A type's average weighs its states by its date-0 masses (after the date-0
# reallocation).  The date-0 values are recomputed by one backward pass over the stored
# path, which reproduces it; the baseline values by the baseline steady state's own.
############################################################

# The state classes of the table, each with its value and its mass fields.
const INCIDENCE_STATES = (
    (:unskilled_unemployed,         :UC,    :uU),
    (:unskilled_employed,           :EUs_p, :eUs_p),
    (:unskilled_employed_will_train, :EUt_p, :eUt_p),
    (:training,                     :T,     :tU),
    (:skilled_unemployed,           :U0,    :uS_skilled),
    (:trained_searching_unskilled,  :U1,    :uS_cross),
    (:skilled_employed,             :ES_p,  :eS_p),
)

"""
    type_values(path, buf, model) -> NamedTuple

Per-type values at date 1 of `path` in every state, from its date-1 values and the
per-type surfaces in `buf` slot 1 (`backward_pass_keep!`), the one-dimensional surfaces
where the type is pure.  p-indexed arrays carry the unskilled atom p = 1 in the last slot;
the training slice is NaN where the cell has none.
"""
function type_values(path::TransitionPath, buf, model::Model)
    Nx = length(model.grids.x);  NpU = length(model.unsk_grids.p);  NpS = length(model.skl_grids.p)
    βU = model.unsk_par.β;  βS = model.skl_par.β
    UC = path.UC[:, :, 1];  T = path.T[:, :, 1];  Utr = path.Utr[:, :, 1]
    U0 = path.U0[:, :, 1];  U1 = path.U1[:, :, 1];  US = max.(U0, U1)
    EUs = fill(NaN, Nx, Nx, NpU + 1);  EUt = fill(NaN, Nx, Nx, NpU + 1);  ES = fill(NaN, Nx, Nx, NpS)
    @inbounds for j in 1:Nx, i in 1:Nx
        Ss = buf.hasS[i, j, 1] ? view(buf.Ss, i, j, :, 1) : view(path.SU, i, :, 1)
        for q in 1:NpU
            EUs[i, j, q] = UC[i, j] + βU * max(Ss[q], 0.0)
        end
        EUs[i, j, NpU + 1] = UC[i, j] + βU * max(path.SUs1[i, j, 1], 0.0)
        if buf.hasT[i, j, 1]
            for q in 1:NpU
                EUt[i, j, q] = Utr[i, j] + βU * max(buf.St[i, j, q, 1], 0.0)
            end
            EUt[i, j, NpU + 1] = Utr[i, j] + βU * max(path.SUt1[i, j, 1], 0.0)
        end
        S0 = buf.hasK[i, j, 1] ? view(buf.K0, i, j, :, 1) : view(path.S0, j, :, 1)
        S1 = buf.hasK[i, j, 1] ? view(buf.K1, i, j, :, 1) : view(path.S1, j, :, 1)
        for q in 1:NpS
            ES[i, j, q] = US[i, j] + βS * max(0.0, S0[q], S1[q])
        end
    end
    return (UC = UC, T = T, Utr = Utr, U0 = U0, U1 = U1, US = US, EUs_p = EUs, EUt_p = EUt,
            ES_p = ES, tau = path.τT[:, :, 1])
end

"""
    type_density(st, model) -> NamedTuple

The masses per type and state of the stocks `st` (`welfare_pv(...).date1`): the crossers
are the share d of the skilled unemployed, and skilled employment of a cell is its
non-draining mass (1 − d) m_S spread over p by the column's unit shape ê_S.
"""
function type_density(st, model::Model)
    sg = model.skl_grids;  Nx = length(model.grids.x);  NpS = length(sg.p)
    eSp = zeros(Nx, Nx, NpS)
    @inbounds for j in 1:Nx, i in 1:Nx, q in 1:NpS
        eSp[i, j, q] = (1.0 - st.d[i, j]) * st.mS[i, j] * st.eS[j, q] * sg.wp[q]
    end
    return (uU = copy(st.uU), tU = copy(st.tU), uS_cross = st.d .* st.uS,
            uS_skilled = (1.0 .- st.d) .* st.uS, eUs_p = copy(st.Es), eUt_p = copy(st.Et),
            eS_p = eSp, d = copy(st.d))
end

"""
    incidence_ratios(v0, vb, ρ0, A0, A1) -> NamedTuple

Per type: the ratio (V_0/A_1)/(V_b/A_0) − 1 of each state class averaged over its states
(its p nodes) with the date-0 masses, its date-0 mass, and the average over every state.
A state with date-0 mass and no baseline value (a training slice the baseline cell does not
have) is left out and its mass reported.
"""
function incidence_ratios(v0, vb, ρ0, A0::Float64, A1::Float64)
    Nx = size(v0.UC, 1)
    num_all = zeros(Nx, Nx);  den_all = zeros(Nx, Nx);  unmatched = zeros(Nx, Nx)
    by_state = Dict{Symbol,Matrix{Float64}}();  mass = Dict{Symbol,Matrix{Float64}}()
    for (name, vf, mf) in INCIDENCE_STATES
        V0 = getfield(v0, vf);  Vb = getfield(vb, vf);  ρ = getfield(ρ0, mf)
        num = zeros(Nx, Nx);  den = zeros(Nx, Nx)
        @inbounds for c in CartesianIndices(ρ)
            m = ρ[c];  m > 0.0 || continue
            ij = CartesianIndex(c[1], c[2])
            if isfinite(V0[c]) && isfinite(Vb[c]) && Vb[c] != 0.0
                rr = (V0[c] / A1) / (Vb[c] / A0) - 1.0
                num[ij] += m * rr;  den[ij] += m
            else
                unmatched[ij] += m
            end
        end
        by_state[name] = ifelse.(den .> 0.0, num ./ den, NaN)
        mass[name] = den
        num_all .+= num;  den_all .+= den
    end
    all_states = ifelse.(den_all .> 0.0, num_all ./ den_all, NaN)
    return (all = all_states, by_state = by_state, mass = mass, unmatched = unmatched)
end

"""
    run_incidence(crises)

For each crisis, the incidence table `incidence_<crisis>.csv` (one row per type: the
population, the average ratio and the ratio and date-0 mass of each state class) and the
heatmap `incidence_<crisis>.png` of the average ratio over (a_U, a_S).  With
`INCIDENCE_SAVE_VALUES` the per-type values and masses themselves are also serialised,
`type_values_<crisis>.jls` (several hundred MB per crisis).
"""
function run_incidence(crises)
    dir = cf_log_start(out_counterfactuals(:incidence), "Incidence: date-0 values against the baseline")
    for crisis in crises
        t0 = time()
        L = load_transition(crisis)
        stored = (UC = copy(L.path.UC[:, :, 1]), U0 = copy(L.path.U0[:, :, 1]), T = copy(L.path.T[:, :, 1]))
        buf  = backward_pass_keep!(L.path, deepcopy(L.m1), L.tp)
        bgap = maximum(maximum(abs, getfield(stored, f) .- getfield(L.path, f)[:, :, 1]) /
                       maximum(abs, getfield(stored, f)) for f in (:UC, :U0, :T))
        cf_log(dir, @sprintf("[%s] backward pass over the stored path: date-1 values reproduced to %.2e (relative)",
                             crisis, bgap))
        v0 = type_values(L.path, buf, L.m1)
        ρ0 = type_density(welfare_pv(L.path, deepcopy(L.m1), L.tp).date1, L.m1)
        p0, tp0 = self_path(L.m0)
        vb = type_values(p0, backward_pass_keep!(p0, deepcopy(L.m0), tp0), L.m0)
        A0 = exp(L.m0.common.A);  A1 = exp(L.m1.common.A)
        ir = incidence_ratios(v0, vb, ρ0, A0, A1)

        gp = L.m1.grids;  Nx = length(gp.x)
        df = DataFrame(i = repeat(1:Nx, outer = Nx), j = repeat(1:Nx, inner = Nx),
                       a_U = repeat(gp.x, outer = Nx), a_S = repeat(gp.x, inner = Nx),
                       population = vec(gp.copula.W2), ratio_all_states = vec(ir.all))
        for (name, _, _) in INCIDENCE_STATES
            df[!, Symbol("ratio_", name)] = vec(ir.by_state[name])
            df[!, Symbol("mass_", name)]  = vec(ir.mass[name])
        end
        df.mass_without_baseline_value = vec(ir.unmatched)
        CSV.write(joinpath(dir, "incidence_$(crisis).csv"), df)
        plot_incidence_heatmap(gp.x, ir.all, gp.copula.W2, joinpath(dir, "incidence_$(crisis).png"), crisis)

        if INCIDENCE_SAVE_VALUES
            open(joinpath(dir, "type_values_$(crisis).jls"), "w") do io
                serialize(io, (crisis = crisis, version = ROYSEARCH_VERSION, x = gp.x, W2 = gp.copula.W2,
                               pU = L.m1.unsk_grids.p, pS = L.m1.skl_grids.p, A_base = A0, A_crisis = A1,
                               values_date0 = v0, values_base = vb, density_date0 = ρ0,
                               density_base = type_density(steady_state_flow(L.m0).replay.date1, L.m0)))
            end
        end
        pop = gp.copula.W2
        k = findall(isfinite, ir.all)
        cf_log(dir, @sprintf("[%s] population-weighted mean ratio %.5f; range over types [%.5f, %.5f]; mass without a baseline value %.2e",
                             crisis, sum(ir.all[k] .* pop[k]) / sum(pop[k]), minimum(ir.all[k]),
                             maximum(ir.all[k]), sum(ir.unmatched)))
        cf_log(dir, @sprintf("[%s] done in %.1f s", crisis, time() - t0))
    end
    return nothing
end

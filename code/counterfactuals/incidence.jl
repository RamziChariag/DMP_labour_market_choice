############################################################
# incidence.jl — who gains and who loses from each crisis, by type and state (RoySearch)
#
# Library file, loaded by counterfactuals_main.jl; `run_incidence` is the exercise.
#
# For every type (a_U, a_S) and state, the value at date 0 of the estimated transition
# against the value in the same state at the baseline steady state, relative to general
# productivity: (V_0 / A_crisis) / (V_base / A_base) − 1, averaged with the date-0 masses.
# Values and masses are those transition_main.jl saved (`types`, transition_types.jl).
############################################################

# The state classes: name, value field and mass field of `types`.
const INCIDENCE_STATES = (
    (:unskilled_unemployed,          :UC,    :uU),
    (:unskilled_employed,            :EUs_p, :eUs_p),
    (:unskilled_employed_will_train, :EUt_p, :eUt_p),
    (:training,                      :T,     :tU),
    (:skilled_unemployed,            :U0,    :uS_skilled),
    (:trained_searching_unskilled,   :U1,    :uS_cross),
    (:skilled_employed,              :ES_p,  :eS_p),
)

"""
    incidence_ratios(types) -> NamedTuple

Per type: the ratio (V_0/A_1)/(V_b/A_0) − 1 of each state class averaged over its p nodes
with the date-0 masses, the class's date-0 mass, and the average over every state. A state
with date-0 mass and no baseline value (a training slice the baseline cell lacks) is left
out and its mass returned as `unmatched`.
"""
function incidence_ratios(types)
    v0 = types.values_date0;  vb = types.values_base;  ρ0 = types.density_date0
    A0 = types.A_base;  A1 = types.A_crisis
    Nx = size(v0.UC, 1)
    num_all = zeros(Nx, Nx);  den_all = zeros(Nx, Nx);  unmatched = zeros(Nx, Nx)
    by_state = Dict{Symbol,Matrix{Float64}}();  mass = Dict{Symbol,Matrix{Float64}}()
    for (name, vf, mf) in INCIDENCE_STATES
        V0 = getfield(v0, vf);  Vb = getfield(vb, vf);  ρ = getfield(ρ0, mf)
        num = zeros(Nx, Nx);  den = zeros(Nx, Nx)
        @inbounds for c in CartesianIndices(ρ)
            m = Float64(ρ[c]);  m > 0.0 || continue
            ij = CartesianIndex(c[1], c[2])
            if isfinite(V0[c]) && isfinite(Vb[c]) && Vb[c] != 0
                num[ij] += m * ((V0[c] / A1) / (Vb[c] / A0) - 1.0);  den[ij] += m
            else
                unmatched[ij] += m
            end
        end
        by_state[name] = ifelse.(den .> 0.0, num ./ den, NaN)
        mass[name] = den
        num_all .+= num;  den_all .+= den
    end
    return (all = ifelse.(den_all .> 0.0, num_all ./ den_all, NaN), by_state = by_state,
            mass = mass, unmatched = unmatched)
end

"""
    run_incidence(crises)

For each crisis, writes `incidence_<crisis>.csv` (one row per type: population, average
ratio, and the ratio and date-0 mass of each state class) and `incidence_<crisis>.jls` (the
same as matrices on the ability grid, for the figures).
"""
function run_incidence(crises)
    dir = cf_log_start(out_counterfactuals(:incidence), "Incidence: date-0 values against the baseline")
    for crisis in crises
        t0 = time()
        L  = load_transition(crisis)
        ir = incidence_ratios(L.tr.types)
        gp = L.m1.grids;  Nx = length(gp.x);  pop = gp.copula.W2
        df = DataFrame(i = repeat(1:Nx, outer = Nx), j = repeat(1:Nx, inner = Nx),
                       a_U = repeat(gp.x, outer = Nx), a_S = repeat(gp.x, inner = Nx),
                       population = vec(pop), ratio_all_states = vec(ir.all))
        for (name, _, _) in INCIDENCE_STATES
            df[!, Symbol("ratio_", name)] = vec(ir.by_state[name])
            df[!, Symbol("mass_", name)]  = vec(ir.mass[name])
        end
        df.mass_without_baseline_value = vec(ir.unmatched)
        CSV.write(joinpath(dir, "incidence_$(crisis).csv"), df)
        open(joinpath(dir, "incidence_$(crisis).jls"), "w") do io
            serialize(io, (crisis = crisis, version = ROYSEARCH_VERSION, point = POINT_ESTIMATE,
                           x = gp.x, population = pop, ratio_all_states = ir.all,
                           ratio_by_state = ir.by_state, mass_by_state = ir.mass,
                           mass_without_baseline_value = ir.unmatched))
        end
        k = findall(isfinite, ir.all)
        cf_log(dir, @sprintf("[%s] population-weighted mean ratio %.5f; range over types [%.5f, %.5f]; mass without a baseline value %.2e; %.1f s",
                             crisis, sum(ir.all[k] .* pop[k]) / sum(pop[k]), minimum(ir.all[k]),
                             maximum(ir.all[k]), sum(ir.unmatched), time() - t0))
    end
    return nothing
end

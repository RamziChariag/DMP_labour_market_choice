############################################################
# welfare.jl — social flow and present value along a transition (RoySearch)
#
# LIBRARY FILE — loaded by counterfactuals_main.jl; `run_welfare` is the exercise, and
# `welfare_pv` / `steady_state_flow` are reused by held_margin.jl and training_cost.jl.
#
# The social flow at a date, per month, in levels (the code stores ln A, b_j/A, σ_S/A,
# ln c and k_j in months of mean new-match output; `level_params` converts):
#   + output of every match: A P_U a_U p (unskilled), A P_S a_S p (skilled)
#   + unemployment income: b_U for the unskilled unemployed (the never-hired included),
#     b_S for every trained worker without a job — those searching the skilled market and
#     those searching the unskilled market, who keep trained status and collect b_S until
#     they accept an unskilled job (U_S^(1) in transition_values.jl) — and b_T in training
#   − vacancy costs k_U v_U + k_S v_S, v_j = θ_j × the seekers free entry counts
#   − σ_S for every employed skilled worker who searches on the job
#   − c(a_S) = (1 − a_S) e^{c̃ − a_S} paid once by every worker who enters training
# Components are booked by market: U (unskilled output, b_U, crossers' b_S, k_U v_U), S
# (skilled output, b_S of the skilled-market seekers, k_S v_S, σ_S) and T (b_T, training
# costs).  The present value is ∫ e^{−rt} flow dt over the path plus the end steady state's
# flow / r discounted from the horizon, reported in levels, in months of the baseline mean
# unskilled new-match output A₀ P_U E[a_U], and divided by the date's A.
############################################################

const COMPONENTS = (:output_U, :output_S, :uinc_U, :uinc_X, :uinc_S, :uinc_T,
                    :vac_U, :vac_S, :search_S, :train_cost)
const NCOMP = length(COMPONENTS)
const COST_COMPONENTS = (:vac_U, :vac_S, :search_S, :train_cost)
const COMPONENT_MARKET = Dict(:output_U => :U, :uinc_U => :U, :uinc_X => :U, :vac_U => :U,
                              :output_S => :S, :uinc_S => :S, :vac_S => :S, :search_S => :S,
                              :uinc_T => :T, :train_cost => :T)
comp_sign(c::Symbol) = c in COST_COMPONENTS ? -1.0 : 1.0
net_flow(f) = sum(comp_sign(COMPONENTS[c]) * f[c] for c in 1:NCOMP)

"""
    level_params(model; resource_c = model.common.c) -> NamedTuple

The model's flow parameters in levels per month.  `resource_c` is the ln c at which
training is costed: the model's own by default; the estimated one when a policy changes c
and the change is a transfer, not a resource.
"""
function level_params(model::Model; resource_c::Float64 = model.common.c)
    cp = model.common;  up = model.unsk_par;  sp = model.skl_par;  A = exp(cp.A)
    return (A = A, r = cp.r, PU = A * up.PU, PS = A * sp.PS, bU = up.bU * A, bS = sp.bS * A,
            bT = up.bT * A, σS = sp.σ * A, kU = up.k * mean_output_U(model),
            kS = sp.k * mean_output_S(model), πU = mean_output_U(model),
            cost = [training_cost(x, resource_c) for x in model.grids.x])
end

"""
    flow_components(st, d, θU, θS, part, poj, model, lp) -> Vector

The social flow at the forward state `st` (`_forward_state`) with crossing fractions `d`,
by `COMPONENTS`, the training cost left at 0 (it is booked from the entries).  Every stock
is read as the code reads it: unskilled matches from each slice's quality measure, the atom
at p = 1 in the last slot; skilled matches from the per-a_S unit shape ê_S on the
non-draining column mass (`path_statistics`); the unskilled seekers u_U + d u_S
(`_free_entry_tightness`); the skilled seekers the denominator of `compute_Jbar_skilled`,
û m₀ plus the employed who search on the job, each of whom pays σ_S.
"""
function flow_components(st, d::Matrix{Float64}, θU::Float64, θS::Float64,
                         part::AbstractVector{Float64}, poj::AbstractVector{Float64},
                         model::Model, lp)
    gp = model.grids;  ug = model.unsk_grids;  sg = model.skl_grids
    Nx = length(gp.x);  NpU = length(ug.p);  NpS = length(sg.p)
    yU = 0.0;  seekU = 0.0;  bu = 0.0;  bx = 0.0;  bs = 0.0;  bt = 0.0
    @inbounds for j in 1:Nx, i in 1:Nx
        for q in 1:(NpU + 1)
            yU += (st.Es[i, j, q] + st.Et[i, j, q]) * lp.PU * gp.x[i] * (q <= NpU ? ug.p[q] : 1.0)
        end
        cross = d[i, j] * st.uS[i, j]
        seekU += st.uU[i, j] + cross
        bu += st.uU[i, j];  bx += cross;  bs += st.uS[i, j] - cross;  bt += st.tU[i, j]
    end
    m0 = model.corner_limit ? skilled_stayer_columns(model, d, st.mS) : nondrain_column_mass(d, st.mS)
    u0 = nondrain_column_mass(d, st.uS)
    yS = 0.0;  seekS = 0.0;  ojs = 0.0
    @inbounds for k in 1:Nx
        ufrac = m0[k] > 1e-14 ? clamp(u0[k] / m0[k], 0.0, 1.0) : 0.0
        se = 0.0
        for q in 1:NpS
            e = st.eS[k, q] * sg.wp[q]
            yS += lp.PS * gp.x[k] * sg.p[q] * e * m0[k]
            se += _ojs_share(part[k], sg.p[q], clamp01(poj[k]), sg.p, q, NpS) * e
        end
        seekS += m0[k] * (ufrac + se);  ojs += m0[k] * se
    end
    return [yU, yS, lp.bU * bu, lp.bS * bx, lp.bS * bs, lp.bT * bt,
            lp.kU * θU * seekU, lp.kS * θS * seekS, lp.σS * ojs, 0.0]
end

disc_int(r, a, h) = (exp(-r * a) - exp(-r * (a + h))) / r      # ∫_a^{a+h} e^{−rt} dt

# The largest gap between the forward state and the stocks the path stores at date n.
function _stock_gap(st, path::TransitionPath, n::Int)
    g = 0.0
    @inbounds for j in axes(st.uU, 2), i in axes(st.uU, 1)
        g = max(g, abs(st.uU[i, j] - path.uU[i, j, n]), abs(st.tU[i, j] - path.tU[i, j, n]),
                abs(st.uS[i, j] - path.uS[i, j, n]), abs(st.mS[i, j] - path.mS[i, j, n]),
                abs(sum(view(st.Es, i, j, :)) - path.eUs[i, j, n]),
                abs(sum(view(st.Et, i, j, :)) - path.eUt[i, j, n]))
    end
    return g
end

"""
    welfare_pv(path, model, tp; flow_end = nothing, resource_c = model.common.c) -> NamedTuple

The present value at date 0 of the social flow along a solved `path` under `model`, its
post-switch (z₁) model.  The forward pass is replayed from the path's date-0⁻ masses with
the path's own policies and tightness, through the same steps `_forward_pass!` takes
(`_reallocate!`, `_advance!`), and the accounts are booked on the way: within each
sub-step the flow at its starting stocks, held for the sub-step as the laws of motion hold
their rates; training costs on every entry, the births and training-slice separations of
each sub-step read off the training stock's own Euler step, and the mass each reallocation
moves into training as a lump at its date.  `flow_end` is the end steady state's flow,
which values the tail (the last date's flow when not given).  Returns the PV by component
(path, tail, total), the per-date flows at the post-reallocation stocks, the lumps, and the
largest gap between the replayed and the stored stocks.
"""
function welfare_pv(path::TransitionPath, model::Model, tp::TransitionParams;
                    flow_end = nothing, resource_c::Float64 = model.common.c)
    Nt = length(path.tgrid);  Nx = length(model.grids.x);  ug = model.unsk_grids
    χU = hire_fraction(model)
    wG = build_unskilled_G_weights(ug.p, ug.wp, model.unsk_par.α_U)
    lp = level_params(model; resource_c = resource_c);  r = lp.r
    φν = model.common.φ + model.common.ν
    st = _forward_state(path)
    flows = zeros(Nt, NCOMP);  lump = zeros(Nt);  pv = zeros(NCOMP);  gap = 0.0
    entry_cost(before, after, dt) =
        sum(lp.cost[j] * (after[i, j] - before[i, j] + dt * φν * before[i, j]) for i in 1:Nx, j in 1:Nx)

    for n in 1:Nt
        d  = _d_matrix(path, model, n)
        hX = _crosser_hire_shares(path, χU, d, n)
        _reallocate!(st, path, model, χU, d, n)
        lump[n] = sum(lp.cost[j] * st.into_T[i, j] for i in 1:Nx, j in 1:Nx)
        pv[NCOMP] += exp(-r * path.tgrid[n]) * lump[n]
        gap = max(gap, _stock_gap(st, path, n))
        flow() = flow_components(st, d, path.θU[n], path.θS[n], view(path.s_part, :, n),
                                 view(path.poj, :, n), model, lp)
        if n == Nt                     # the last date's flow, its training entries by one dry step
            f = flow();  h = 1e-3;  dry = deepcopy(st)
            _advance!(dry, path, model, χU, hX, d, wG, n, h)
            f[NCOMP] = entry_cost(dry.tU_n, dry.tU, h) / h
            flows[n, :] .= f
            break
        end
        sub = _substeps(path, n)
        for k in 1:sub.nsub
            f = flow()
            _advance!(st, path, model, χU, hX, d, wG, n, sub.dt)
            f[NCOMP] = entry_cost(st.tU_n, st.tU, sub.dt) / sub.dt
            k == 1 && (flows[n, :] .= f)
            pv .+= disc_int(r, path.tgrid[n] + (k - 1) * sub.dt, sub.dt) .* f
        end
        _carry_policies!(st, path, n)
    end
    tail = exp(-r * path.tgrid[end]) / r .* (flow_end === nothing ? flows[Nt, :] : flow_end)
    return (pv_path = pv, pv_tail = tail, pv = pv .+ tail, flows = flows, lump = lump,
            lp = lp, replay_gap = gap)
end

"""
    steady_state_flow(model; resource_c) -> (flow, replay)

The social flow of the steady state `model`, from its self-transition (`self_path`): the
flow at date 1, whose stocks are the steady state's.
"""
function steady_state_flow(model::Model; resource_c::Float64 = model.common.c)
    p, tp = self_path(model)
    w = welfare_pv(p, deepcopy(model), tp; resource_c = resource_c)
    return (flow = w.flows[1, :], replay = w)
end

# One row per component, net per market, and the total, in the four units.
function welfare_rows(crisis::Symbol, object::AbstractString, pv::AbstractVector, πU0::Float64,
                      A::Float64, πU::Float64)
    rows = NamedTuple[]
    add(market, comp, s, v) = push!(rows, (crisis = String(crisis), object = object,
        market = String(market), component = String(comp), sign = s, W_level = v,
        W_months_base_piU = v / πU0, W_over_A = v / A, W_months_own_piU = v / πU))
    for (c, comp) in enumerate(COMPONENTS)
        add(COMPONENT_MARKET[comp], comp, comp_sign(comp), pv[c])
    end
    for m in (:U, :S, :T)
        add(m, :net, 1.0, sum(comp_sign(comp) * pv[c] for (c, comp) in enumerate(COMPONENTS)
                              if COMPONENT_MARKET[comp] == m))
    end
    add(:all, :total, 1.0, net_flow(pv))
    return rows
end

"""
    run_welfare(crises)

For each crisis: the present value of the social flow along the stored transition, and of
the baseline and the crisis steady states held forever, by market and component, with the
path's per-date stocks and flows.  Writes `welfare_<crisis>.csv`, `path_<crisis>.csv` and
`flows_ss_<crisis>.csv`.
"""
function run_welfare(crises)
    dir = cf_log_start(out_counterfactuals(:welfare), "Welfare along the estimated transitions")
    for crisis in crises
        t0 = time()
        L  = load_transition(crisis)
        Nt = length(L.path.tgrid)
        cf_log(dir, @sprintf("[%s] %s: %d dates, horizon %.0f months", crisis,
                             basename(transition_file(crisis, W_SUFFIX, POINT_ESTIMATE)), Nt, L.tp.horizon))
        ss0 = steady_state_flow(L.m0);  ss1 = steady_state_flow(L.m1)
        F0 = ss0.flow;  F1 = ss1.flow
        w  = welfare_pv(L.path, deepcopy(L.m1), L.tp; flow_end = F1)
        lp0 = level_params(L.m0);  lp1 = level_params(L.m1);  r = lp1.r
        cf_log(dir, @sprintf("[%s] replay of the stored masses: largest cell gap %.2e (steady states %.2e, %.2e)",
                             crisis, w.replay_gap, ss0.replay.replay_gap, ss1.replay.replay_gap))

        rows = vcat(welfare_rows(crisis, "path", w.pv, lp0.πU, lp1.A, lp1.πU),
                    welfare_rows(crisis, "path_0_to_horizon", w.pv_path, lp0.πU, lp1.A, lp1.πU),
                    welfare_rows(crisis, "path_tail_after_horizon", w.pv_tail, lp0.πU, lp1.A, lp1.πU),
                    welfare_rows(crisis, "baseline_ss_forever", F0 ./ r, lp0.πU, lp0.A, lp0.πU),
                    welfare_rows(crisis, "crisis_ss_forever", F1 ./ r, lp0.πU, lp1.A, lp1.πU))
        CSV.write(joinpath(dir, "welfare_$(crisis).csv"), DataFrame(rows))

        res = L.tr.result;  stt = L.tr.statistics;  p = L.path
        θU_fe, θS_fe = _free_entry_tightness(p, deepcopy(L.m1))
        crossers = [sum(_d_matrix(p, L.m1, n) .* view(p.uS, :, :, n)) for n in 1:Nt]
        df = DataFrame(t = p.tgrid, theta_U = p.θU, theta_S = p.θS,
                       fe_gap_U = abs.(log.(θU_fe) .- log.(p.θU)), fe_gap_S = abs.(log.(θS_fe) .- log.(p.θS)),
                       uU = [sum(view(p.uU, :, :, n)) for n in 1:Nt],
                       eU = [sum(view(p.eUs, :, :, n)) + sum(view(p.eUt, :, :, n)) for n in 1:Nt],
                       in_training = [sum(view(p.tU, :, :, n)) for n in 1:Nt],
                       uS = [sum(view(p.uS, :, :, n)) for n in 1:Nt],
                       mS = [sum(view(p.mS, :, :, n)) for n in 1:Nt],
                       ur_U = res.ur_U, ur_S = res.ur_S, ur_total = res.ur_total,
                       skilled_share = res.skilled_share, training_share = res.training_share,
                       crossers_d_uS = crossers, crossers_hired = res.d_flow,
                       train_entry_births = stt.train_entry_births,
                       swept_into_training = stt.swept_into_training,
                       quit_into_training = stt.quit_into_training, training_cost_lump = w.lump)
        for (c, comp) in enumerate(COMPONENTS)
            df[!, Symbol("flow_", comp)] = w.flows[:, c]
        end
        df.flow_net = [net_flow(w.flows[n, :]) for n in 1:Nt]
        df.flow_net_over_A = df.flow_net ./ lp1.A
        CSV.write(joinpath(dir, "path_$(crisis).csv"), df)

        CSV.write(joinpath(dir, "flows_ss_$(crisis).csv"),
                  DataFrame(component = [String.(COMPONENTS)..., "net"],
                            baseline_ss = [F0..., net_flow(F0)], crisis_ss = [F1..., net_flow(F1)],
                            path_date0 = [w.flows[1, :]..., net_flow(w.flows[1, :])],
                            path_last_date = [w.flows[end, :]..., net_flow(w.flows[end, :])]))

        Wp = net_flow(w.pv);  W0 = net_flow(F0) / r;  W1 = net_flow(F1) / r
        cf_log(dir, @sprintf("[%s] W, months of baseline mean unskilled output: path %.4f, baseline SS %.4f, crisis SS %.4f",
                             crisis, Wp / lp0.πU, W0 / lp0.πU, W1 / lp0.πU))
        cf_log(dir, @sprintf("[%s] W / A: path %.4f, baseline SS %.4f, crisis SS %.4f; last-date flow vs crisis SS %+.2e (relative)",
                             crisis, Wp / lp1.A, W0 / lp0.A, W1 / lp1.A,
                             net_flow(w.flows[end, :]) / net_flow(F1) - 1.0))
        cf_log(dir, @sprintf("[%s] done in %.1f s", crisis, time() - t0))
    end
    return nothing
end

############################################################
# welfare.jl — the social flow of a steady state (RoySearch)
#
# Library file, loaded by counterfactuals_main.jl; `steady_state_flow` is what
# training_cost.jl reads. The flow is booked by replaying the forward pass of a path
# (`welfare_pv`), here the one-step self-transition of the steady state.
#
# The social flow per month, in levels (`level_params` converts the code's units):
#   + output of every match, A P_U a_U p and A P_S a_S p
#   + b_U for the unskilled unemployed (the never-hired included), b_S for every trained
#     worker without a job (also those searching the unskilled market), b_T in training
#   − vacancy costs k_U v_U + k_S v_S, v_j = θ_j × the seekers free entry counts
#   − σ_S for every employed skilled worker searching on the job
#   − c(a_S) = (1 − a_S) e^{c̃ − a_S} once per entry into training
# in the components `COMPONENTS`: output by market, b_U (uinc_U), crossers' b_S (uinc_X),
# skilled-market seekers' b_S (uinc_S), b_T (uinc_T), k_U v_U, k_S v_S, σ_S, training costs.
############################################################

const COMPONENTS = (:output_U, :output_S, :uinc_U, :uinc_X, :uinc_S, :uinc_T,
                    :vac_U, :vac_S, :search_S, :train_cost)
const NCOMP = length(COMPONENTS)
const COST_COMPONENTS = (:vac_U, :vac_S, :search_S, :train_cost)

"""
    comp_sign(c) -> Float64

−1 for a cost component, +1 otherwise.
"""
comp_sign(c::Symbol) = c in COST_COMPONENTS ? -1.0 : 1.0

"""
    net_flow(f) -> Float64

The net social value of a vector `f` ordered by `COMPONENTS`.
"""
net_flow(f) = sum(comp_sign(COMPONENTS[c]) * f[c] for c in 1:NCOMP)

"""
    level_params(model; resource_c = model.common.c) -> NamedTuple

The model's flow parameters in levels per month, and the training cost at each a_S node.
`resource_c` is the ln c at which training is costed: the model's own by default, the
estimated one when a policy's change in c is a transfer.
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

The social flow by `COMPONENTS` at the forward state `st` (`_forward_state`) with crossing
fractions `d` and tightnesses (θU, θS), the training cost left at 0 (booked from the
entries). Stocks are read as the solver reads them: unskilled matches from each slice's
quality measure; skilled matches from ê_S on the non-draining column mass; the unskilled
seekers u_U + d u_S; the skilled seekers û m₀ plus the on-the-job searchers, each paying σ_S.
"""
function flow_components(st, d::Matrix{Float64}, θU::Float64, θS::Float64,
                         part::AbstractVector{Float64}, poj::AbstractVector{Float64},
                         model::Model, lp)
    gp = model.grids;  ug = model.unsk_grids;  sg = model.skl_grids
    Nx = length(gp.x);  NpU = length(ug.p);  NpS = length(sg.p)
    yU = 0.0;  seekU = 0.0;  bu = 0.0;  bx = 0.0;  bs = 0.0;  bn = 0.0
    @inbounds for j in 1:Nx, i in 1:Nx
        for q in 1:(NpU + 1)
            yU += (st.Es[i, j, q] + st.Et[i, j, q]) * lp.PU * gp.x[i] * (q <= NpU ? ug.p[q] : 1.0)
        end
        cross = d[i, j] * st.uS[i, j]
        seekU += st.uU[i, j] + cross
        bu += st.uU[i, j];  bx += cross;  bs += st.uS[i, j] - cross;  bn += st.nU[i, j]
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
    return [yU, yS, lp.bU * bu, lp.bS * bx, lp.bS * bs, lp.bT * bn,
            lp.kU * θU * seekU, lp.kS * θS * seekS, lp.σS * ojs, 0.0]
end

"""
    disc_int(r, a, h) -> Float64

∫_a^{a+h} e^{−rt} dt.
"""
disc_int(r, a, h) = (exp(-r * a) - exp(-r * (a + h))) / r

"""
    _stock_gap(st, path, n) -> Float64

The largest gap between the forward state `st` and the stocks `path` stores at date n.
"""
function _stock_gap(st, path::TransitionPath, n::Int)
    g = 0.0
    @inbounds for j in axes(st.uU, 2), i in axes(st.uU, 1)
        g = max(g, abs(st.uU[i, j] - path.uU[i, j, n]), abs(st.nU[i, j] - path.nU[i, j, n]),
                abs(st.uS[i, j] - path.uS[i, j, n]), abs(st.mS[i, j] - path.mS[i, j, n]),
                abs(sum(view(st.Es, i, j, :)) - path.eUs[i, j, n]),
                abs(sum(view(st.Et, i, j, :)) - path.eUt[i, j, n]))
    end
    return g
end

"""
    welfare_pv(path, model, tp; flow_end = nothing, resource_c = model.common.c) -> NamedTuple

The present value at date 0 of the social flow along the solved `path` under its z₁
`model`. The forward pass is replayed from date 0⁻ with the path's own policies and
tightness, through the steps `_forward_pass!` takes, booking within each sub-step the flow
at its starting stocks and the training cost of its entries (read off the training stock's
Euler step), and each reallocation's entries into training as a lump at its date. The tail
is valued at `flow_end` (default: the last date's flow). Returns the PV by component (path,
tail, total), the per-date flows at the post-reallocation stocks, the lumps, the level
parameters and the largest gap between the replayed and the stored stocks.
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
            f[NCOMP] = entry_cost(dry.nU_n, dry.nU, h) / h
            flows[n, :] .= f
            break
        end
        sub = _substeps(path, n)
        for k in 1:sub.nsub
            f = flow()
            _advance!(st, path, model, χU, hX, d, wG, n, sub.dt)
            f[NCOMP] = entry_cost(st.nU_n, st.nU, sub.dt) / sub.dt
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

The social flow of the steady state `model`, at the first date of its self-transition
(`self_path`), and the replay that produced it.
"""
function steady_state_flow(model::Model; resource_c::Float64 = model.common.c)
    p, tp = self_path(model)
    w = welfare_pv(p, deepcopy(model), tp; resource_c = resource_c)
    return (flow = w.flows[1, :], replay = w)
end

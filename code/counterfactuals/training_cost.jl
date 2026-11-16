############################################################
# training_cost.jl — the cost of training scaled from date 0, with and without each crisis
#
# Library file, loaded by counterfactuals_main.jl; `run_training_cost` is the exercise.
#
# For each scale s of `TRAINING_COST_GRID`, the training cost becomes s·c(a_S) at date 0,
# unannounced and permanent, on two paths from the baseline steady state:
#   crisis      the crisis parameters from date 0 (at s = 1, the stored transition)
#   no_crisis   the baseline parameters throughout (at s = 1, the baseline steady state)
# Tightness is re-solved by free entry at the estimated vacancy costs from the matching
# estimated path, on the stored horizon; the end steady state follows the estimate's branch
# (`stationary_at_costs!`, a switch flagged by `unskilled_branch`). A scale with no steady
# state on that branch (a fold) is reported and its path not solved. Welfare counts the
# subsidy (s < 1) or tax (s > 1) as a transfer: training is costed at the estimated c.
############################################################

"""
    training_cost_endpoint(model, s, kU, kS) -> (model, solve, branch)

The steady state of `model` with its training cost scaled by `s`, at the vacancy costs
(kU, kS), from `model`'s tightnesses, with its unskilled branch checked against `model`'s.
"""
function training_cost_endpoint(model::Model, s::Float64, kU::Float64, kS::Float64)
    θ = (U = model.unsk_cache.θ, S = model.skl_cache.θ)
    mc = with_training_cost(model, s)
    st = stationary_at_costs!(mc, kU, kS, θ)
    st.ok || @warn "Training cost ×$s: steady state at the estimated costs not found (residual $(st.resid))."
    b_ref = unskilled_branch(model, θ.U, θ.S, kU)
    b_new = unskilled_branch(mc, st.θU, st.θS, kU)
    switched = length(b_ref.roots) != length(b_new.roots) || b_ref.rank != b_new.rank
    return (model = mc, solve = st, branch = (reference = b_ref, new = b_new, switched = switched))
end

"""
    run_training_cost(crises)

For each crisis, writes `training_cost_<crisis>.csv` (one row per path and scale: welfare
in the four units, unemployment over the crisis window and at the horizon, the end steady
state's tightnesses and branch, convergence) and `training_cost_<crisis>.jls` (the
TransitionResult of every path). Logs the number of transition solves first.
"""
function run_training_cost(crises)
    dir = cf_log_start(out_counterfactuals(:training_cost), "Training cost scaled from date 0")
    grid = Float64.(TRAINING_COST_GRID)
    1.0 in grid || error("TRAINING_COST_GRID must contain 1.0, the estimated cost every scale is compared with.")
    n_solves = length(crises) * 2 * count(!=(1.0), grid)
    cf_log(dir, "Grid $(grid); $(n_solves) transition solves (s = 1 reuses the stored paths), " *
                "up to $(CF_MAXIT) outer iterations each.")
    for crisis in crises
        L  = load_transition(crisis)
        m0 = L.m0;  m1 = L.m1;  t = L.path.tgrid
        c0 = m0.common.c;  c1 = m1.common.c
        lp0 = level_params(m0);  πU0 = lp0.πU
        F0 = steady_state_flow(m0).flow;  F1 = steady_state_flow(m1).flow
        θ0 = (U = m0.unsk_cache.θ, S = m0.skl_cache.θ)
        rows = NamedTuple[];  results = Dict{Tuple{Symbol,Float64},Any}()
        for kind in (:crisis, :no_crisis), s in grid
            t0 = time()
            mref = kind === :crisis ? m1 : m0
            cres = kind === :crisis ? c1 : c0
            A = exp(mref.common.A);  πU = level_params(mref).πU
            if s == 1.0
                if kind === :crisis
                    w = welfare_pv(L.path, deepcopy(m1), L.tp; flow_end = F1)
                    W = net_flow(w.pv);  res = L.tr.result
                else
                    W = net_flow(F0) / lp0.r;  res = L.tr.baseline
                end
                ep = (solve = (ok = true, θU = mref.unsk_cache.θ, θS = mref.skl_cache.θ, resid = 0.0,
                               iters = 0), branch = (switched = false,))
                conv = true;  gap = 0.0;  its = 0
            else
                ep = training_cost_endpoint(mref, s, mref.unsk_par.k, mref.skl_par.k)
                if !ep.solve.ok
                    cf_log(dir, @sprintf("[%s] %-9s c×%.2f: no steady state at the estimated costs on the estimate's branch (Newton residual %.2e at θ_U %.4f): path not solved",
                                         crisis, kind, s, ep.solve.resid, ep.solve.θU))
                    push!(rows, (crisis = String(crisis), path = String(kind), scale = s,
                                 W_level = NaN, W_months_base_piU = NaN, W_over_A = NaN,
                                 W_months_own_piU = NaN, ur_U_window = NaN, ur_S_window = NaN,
                                 ur_total_window = NaN, ur_U_end = NaN, ur_S_end = NaN,
                                 training_share_end = NaN, training_share_window = NaN,
                                 skilled_share_window = NaN, skilled_share_end = NaN, theta_U_end = ep.solve.θU,
                                 theta_S_end = ep.solve.θS, steady_state_found = false,
                                 branch_switch = ep.branch.switched, converged = false,
                                 iterations = 0, fe_gap = NaN, seconds = time() - t0))
                    continue
                end
                guess = kind === :crisis ? (tgrid = t, θU = L.path.θU, θS = L.path.θS) :
                                           (tgrid = t, θU = fill(θ0.U, length(t)), θS = fill(θ0.S, length(t)))
                sol = solve_cf_transition(m0, ep.model, L.tp; guess = guess,
                                          label = Symbol(crisis, "_", kind, "_c", s))
                Fend = steady_state_flow(ep.model; resource_c = cres).flow
                w = welfare_pv(sol.path, deepcopy(ep.model), L.tp; flow_end = Fend, resource_c = cres)
                W = net_flow(w.pv);  res = sol.result
                conv = res.converged;  gap = res.final_dist;  its = res.n_iter
            end
            results[(kind, s)] = res
            win(s_) = kind === :no_crisis && s == 1.0 ? getfield(res, s_)[1] :
                                                        window_mean(getfield(res, s_), res.tgrid, L.months)
            push!(rows, (crisis = String(crisis), path = String(kind), scale = s,
                         W_level = W, W_months_base_piU = W / πU0, W_over_A = W / A,
                         W_months_own_piU = W / πU,
                         ur_U_window = win(:ur_U), ur_S_window = win(:ur_S), ur_total_window = win(:ur_total),
                         ur_U_end = getfield(res, :ur_U)[end], ur_S_end = getfield(res, :ur_S)[end],
                         training_share_end = getfield(res, :training_share)[end],
                         training_share_window = win(:training_share),
                         skilled_share_window = win(:skilled_share),
                         skilled_share_end = getfield(res, :skilled_share)[end],
                         theta_U_end = ep.solve.θU, theta_S_end = ep.solve.θS,
                         steady_state_found = ep.solve.ok, branch_switch = ep.branch.switched,
                         converged = conv, iterations = its, fe_gap = gap, seconds = time() - t0))
            cf_log(dir, @sprintf("[%s] %-9s c×%.2f: W %.4f months of baseline output, ur_U %.5f, ur_S %.5f over the window; %s, %.0f s",
                                 crisis, kind, s, W / πU0, rows[end].ur_U_window, rows[end].ur_S_window,
                                 conv ? "converged" : "NOT converged", rows[end].seconds))
            ep.branch.switched && cf_log(dir, "[$crisis] SWITCH: c×$s moves the end steady state off the estimate's unskilled branch.")
        end
        df = DataFrame(rows)
        CSV.write(joinpath(dir, "training_cost_$(crisis).csv"), df)
        open(joinpath(dir, "training_cost_$(crisis).jls"), "w") do io
            serialize(io, (crisis = crisis, version = ROYSEARCH_VERSION, point = POINT_ESTIMATE,
                           grid = grid, table = df, results = results, months = L.months))
        end
        for kind in ("crisis", "no_crisis")
            sub = df[(df.path .== kind) .& isfinite.(df.W_level), :]
            b = argmax(sub.W_level)
            cf_log(dir, @sprintf("[%s] best scale on the %s path: c×%.2f (W %.4f vs %.4f at c×1, months of baseline output)",
                                 crisis, kind, sub.scale[b], sub.W_months_base_piU[b],
                                 sub.W_months_base_piU[findfirst(==(1.0), sub.scale)]))
        end
    end
    return nothing
end

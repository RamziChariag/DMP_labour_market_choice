############################################################
# training_cost.jl — the cost of training scaled from date 0, with and without each crisis
#
# LIBRARY FILE — loaded by counterfactuals_main.jl; `run_training_cost` is the exercise.
#
# For each scale s of `TRAINING_COST_GRID`, the training cost c(a_S) becomes s·c(a_S) at
# date 0, unannounced, and stays there, on two paths from the baseline steady state:
#   crisis      the crisis parameters from date 0 (the estimated transition at s = 1)
#   no_crisis   the baseline parameters throughout (the baseline steady state at s = 1)
# Tightness is re-solved by free entry at the estimated vacancy costs, warm-started from
# the corresponding estimated path (the stored transition; the baseline steady state's
# tightness); the end steady state follows the estimate's branch from its tightnesses
# (`stationary_at_costs!`) and `unskilled_branch` flags a switch.  The horizon is the
# stored transition's.  s = 1 needs no solve: it is the stored transition and the baseline
# steady state.  A scale at which Newton finds no steady state on the estimate's branch (a
# fold: the branch has ended) is reported and its path not solved.
#
# Welfare is the present value of the social flow with the subsidy (s < 1) or tax (s > 1)
# counted as a transfer: the training cost that enters it is the resource cost at the
# estimated c, (1 − a_S) e^{c̃ − a_S}, paid on the entries the scaled cost produces.
############################################################

"""
    training_cost_endpoint(model, s, kU, kS) -> (model, solve, branch)

The steady state of `model`'s parameters with the training cost scaled by `s`, at the
vacancy costs (kU, kS), from `model`'s tightnesses, with its unskilled branch checked
against `model`'s.
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

Writes `training_cost_<crisis>.csv` (one row per path and scale: welfare in the four units,
unemployment by market over the crisis window and at the horizon, the end steady state's
tightnesses and branch, the solve's convergence) and `training_cost_<crisis>.png`.  Prints
the number of transition solves before starting.
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
        rows = NamedTuple[]
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
                                 training_share_end = NaN, theta_U_end = ep.solve.θU,
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
            win(s_) = kind === :no_crisis && s == 1.0 ? getfield(res, s_)[1] :
                                                        window_mean(getfield(res, s_), res.tgrid, L.months)
            push!(rows, (crisis = String(crisis), path = String(kind), scale = s,
                         W_level = W, W_months_base_piU = W / πU0, W_over_A = W / A,
                         W_months_own_piU = W / πU,
                         ur_U_window = win(:ur_U), ur_S_window = win(:ur_S), ur_total_window = win(:ur_total),
                         ur_U_end = getfield(res, :ur_U)[end], ur_S_end = getfield(res, :ur_S)[end],
                         training_share_end = getfield(res, :training_share)[end],
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
        for kind in ("crisis", "no_crisis")
            sub = df[(df.path .== kind) .& isfinite.(df.W_level), :]
            b = argmax(sub.W_level)
            cf_log(dir, @sprintf("[%s] best scale on the %s path: c×%.2f (W %.4f vs %.4f at c×1, months of baseline output)",
                                 crisis, kind, sub.scale[b], sub.W_months_base_piU[b],
                                 sub.W_months_base_piU[findfirst(==(1.0), sub.scale)]))
        end
        plot_training_cost(df, joinpath(dir, "training_cost_$(crisis).png"), crisis)
    end
    return nothing
end

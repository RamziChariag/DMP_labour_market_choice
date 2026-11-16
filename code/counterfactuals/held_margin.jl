############################################################
# held_margin.jl — each crisis with the education margin held at the baseline (RoySearch)
#
# Library file, loaded by counterfactuals_main.jl; `run_held_margin` is the exercise.
#
# The transition at the estimated parameters with who trains (τ) and which market trained
# workers search (d) held at the baseline steady state's choices at every date, known to
# every agent (`HeldChoices`, `Model.held`). Tightness, by `HELD_FREE_ENTRY`:
#   false  the estimated path's: one backward and one forward pass; the end steady state is
#          the crisis model with the held choices at the crisis tightnesses, and the
#          free-entry gap this leaves is reported
#   true   re-solved by free entry at the estimated vacancy costs from the estimated path;
#          the end steady state follows the crisis equilibrium's branch
#          (`stationary_at_costs!`) and `unskilled_branch` reports a switch
# Reported per crisis and market: unemployment along both paths, its rise over the baseline
# (window mean and peak), the margin's share (rise_est − rise_held)/rise_est (negative: the
# margin absorbed part of the rise), and the welfare difference.
############################################################

const HELD_SERIES = (ur_U = :ur_U, ur_S = :ur_S, ur_total = :ur_total, skilled_share = :skilled_share,
                     training_share = :training_share, theta_U = :θU, theta_S = :θS)

"""
    held_endpoint(m0, m1) -> (model, solve, branch)

The crisis steady state with the baseline's education choices (`m1`'s parameters, `m0`'s
choices) at `m1`'s tightnesses, its vacancy costs set to the estimated ones; under
`HELD_FREE_ENTRY` the tightnesses are re-solved at those costs and the unskilled branch
checked against the estimate's.
"""
function held_endpoint(m0::Model, m1::Model)
    held = HeldChoices(m0)
    θ1 = (U = m1.unsk_cache.θ, S = m1.skl_cache.θ)
    mh, sr = solve_model(m1.common, m1.unsk_par, m1.skl_par, m1.sim;
                         Nx = length(m1.grids.x), Np_U = length(m1.unsk_grids.p),
                         Np_S = length(m1.skl_grids.p), tightness = θ1, held = held)
    sr.ok || @warn "The held-choice crisis steady state at the crisis tightnesses fails the residual battery."
    mh.grids.x == m1.grids.x || error("held_endpoint: the rebuilt grid differs from the crisis model's.")
    kU = m1.unsk_par.k;  kS = m1.skl_par.k
    if !HELD_FREE_ENTRY
        mh.unsk_par = _with_k(mh.unsk_par, kU);  mh.skl_par = _with_k(mh.skl_par, kS)
        return (model = mh, solve = (ok = sr.ok, θU = θ1.U, θS = θ1.S, resid = NaN, iters = 0),
                branch = nothing)
    end
    st = stationary_at_costs!(mh, kU, kS, θ1)
    st.ok || @warn "The held-choice crisis steady state at the estimated costs was not found (residual $(st.resid))."
    b_est  = unskilled_branch(m1, θ1.U, θ1.S, kU)
    b_held = unskilled_branch(mh, st.θU, st.θS, kU)
    switched = length(b_est.roots) != length(b_held.roots) || b_est.rank != b_held.rank
    return (model = mh, solve = st, branch = (estimate = b_est, held = b_held, switched = switched))
end

"""
    held_path(L, mh) -> (result, path, seconds, fe_gap)

The held-margin transition from the baseline steady state to `mh` on the stored dates:
re-solved by free entry under `HELD_FREE_ENTRY`, otherwise one backward and one forward
pass at the estimated path's tightness, with the free-entry gap it leaves.
"""
function held_path(L, mh::Model)
    t0 = time()
    if HELD_FREE_ENTRY
        sol = solve_cf_transition(L.m0, mh, L.tp; label = :held_margin,
                                  guess = (tgrid = L.path.tgrid, θU = L.path.θU, θS = L.path.θS))
        return (result = sol.result, path = sol.path, seconds = sol.seconds, fe_gap = sol.result.final_dist)
    end
    m = deepcopy(mh)
    path = allocate_path(m, L.tp)
    _init_path!(path, deepcopy(L.m0), m)
    path.θU .= L.path.θU;  path.θS .= L.path.θS
    _backward_pass!(path, m, L.tp)
    _forward_pass!(path, m, L.tp)
    θU_fe, θS_fe = _free_entry_tightness(path, deepcopy(mh))
    gap = max(maximum(abs, log.(θU_fe) .- log.(path.θU)), maximum(abs, log.(θS_fe) .- log.(path.θS)))
    res = _build_result(path, m, L.tp, :held_margin, false, 1, gap)
    return (result = res, path = path, seconds = time() - t0, fe_gap = gap)
end

"""
    run_held_margin(crises)

For each crisis, writes `held_margin_path_<crisis>.csv` (both paths by date),
`held_margin_<crisis>.csv` (unemployment per market), `held_margin_welfare_<crisis>.csv` and
`held_margin_<crisis>.jls` (both TransitionResults, the end steady state and its branch).
"""
function run_held_margin(crises)
    mode = HELD_FREE_ENTRY ? "tightness re-solved by free entry" : "tightness held on the estimated path"
    dir = cf_log_start(out_counterfactuals(:held_margin), "Education margin held at the baseline ($mode)")
    for crisis in crises
        t0 = time()
        L  = load_transition(crisis)
        ep = held_endpoint(L.m0, L.m1)
        cf_log(dir, @sprintf("[%s] held end steady state: θ_U %.6f, θ_S %.6f (crisis estimate %.6f, %.6f)",
                             crisis, ep.solve.θU, ep.solve.θS, L.m1.unsk_cache.θ, L.m1.skl_cache.θ))
        if ep.branch !== nothing
            b = ep.branch
            cf_log(dir, "[$crisis] unskilled free-entry zeros, estimate: $(round.(b.estimate.roots, digits = 6)) " *
                        "(rank $(b.estimate.rank)); held: $(round.(b.held.roots, digits = 6)) (rank $(b.held.rank))")
            b.switched && cf_log(dir, "[$crisis] SWITCH: the held steady state is not on the estimate's unskilled branch.")
        end
        if !ep.solve.ok
            cf_log(dir, @sprintf("[%s] no held steady state at the estimated costs on the estimate's branch (Newton residual %.2e at θ_U %.4f): path not solved",
                                 crisis, ep.solve.resid, ep.solve.θU))
            continue
        end
        hp  = held_path(L, ep.model)
        est = L.tr.result;  hld = hp.result;  t = L.path.tgrid
        cf_log(dir, @sprintf("[%s] held path in %.1f s; free-entry gap %.3e%s", crisis, hp.seconds, hp.fe_gap,
                             HELD_FREE_ENTRY ? (hld.converged ? " (converged)" : " (NOT converged)") :
                                               " (tightness held: no free-entry iteration)"))
        if HELD_FREE_ENTRY
            jump = maximum(abs, log.(hld.θU) .- log.(est.θU))
            cf_log(dir, @sprintf("[%s] largest |ln θ_U^held − ln θ_U^est| along the path: %.4f", crisis, jump))
        end

        df = DataFrame(t = t)
        for (name, s) in pairs(HELD_SERIES)
            df[!, Symbol(name, "_estimated")] = getfield(est, s)
            df[!, Symbol(name, "_held")]      = getfield(hld, s)
        end
        CSV.write(joinpath(dir, "held_margin_path_$(crisis).csv"), df)

        base = L.tr.baseline
        rows = NamedTuple[]
        for (market, s) in (("U", :ur_U), ("S", :ur_S), ("total", :ur_total))
            u0 = getfield(base, s)[1]
            ue = getfield(est, s);  uh = getfield(hld, s)
            rise_e = window_mean(ue, t, L.months) - u0;  rise_h = window_mean(uh, t, L.months) - u0
            peak_e = maximum(ue) - u0;  peak_h = maximum(uh) - u0
            push!(rows, (crisis = String(crisis), market = market, u_baseline = u0,
                         mean_rise_estimated = rise_e, mean_rise_held = rise_h,
                         margin_share_of_mean_rise = (rise_e - rise_h) / rise_e,
                         peak_rise_estimated = peak_e, peak_rise_held = peak_h,
                         margin_share_of_peak_rise = (peak_e - peak_h) / peak_e,
                         u_end_estimated = ue[end], u_end_held = uh[end]))
        end
        CSV.write(joinpath(dir, "held_margin_$(crisis).csv"), DataFrame(rows))

        F1 = steady_state_flow(L.m1).flow;  Fh = steady_state_flow(ep.model).flow
        we = welfare_pv(L.path, deepcopy(L.m1), L.tp; flow_end = F1)
        wh = welfare_pv(hp.path, deepcopy(ep.model), L.tp; flow_end = Fh)
        lp0 = level_params(L.m0);  lp1 = level_params(L.m1);  A1 = lp1.A
        wrows = vcat(welfare_rows(crisis, "estimated", we.pv, lp0.πU, A1, lp1.πU),
                     welfare_rows(crisis, "held", wh.pv, lp0.πU, A1, lp1.πU),
                     welfare_rows(crisis, "held_minus_estimated", wh.pv .- we.pv, lp0.πU, A1, lp1.πU))
        CSV.write(joinpath(dir, "held_margin_welfare_$(crisis).csv"), DataFrame(wrows))
        open(joinpath(dir, "held_margin_$(crisis).jls"), "w") do io
            serialize(io, (crisis = crisis, version = ROYSEARCH_VERSION, point = POINT_ESTIMATE,
                           free_entry = HELD_FREE_ENTRY, estimated = est, held = hld,
                           end_state = ep.solve, branch = ep.branch, fe_gap = hp.fe_gap,
                           months = L.months))
        end

        for r in rows
            cf_log(dir, @sprintf("[%s] %-5s mean rise over the window: estimated %+.5f, held %+.5f; margin share %+.3f",
                                 crisis, r.market, r.mean_rise_estimated, r.mean_rise_held,
                                 r.margin_share_of_mean_rise))
        end
        cf_log(dir, @sprintf("[%s] W held − W estimated: %+.4f months of baseline mean unskilled output (%+.4f / A); replay gaps %.1e, %.1e",
                             crisis, net_flow(wh.pv .- we.pv) / lp0.πU, net_flow(wh.pv .- we.pv) / A1,
                             we.replay_gap, wh.replay_gap))
        cf_log(dir, @sprintf("[%s] done in %.1f s", crisis, time() - t0))
    end
    return nothing
end

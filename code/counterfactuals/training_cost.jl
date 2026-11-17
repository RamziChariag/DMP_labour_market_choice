############################################################
# training_cost.jl — the steady state's response to the cost of training (RoySearch)
#
# Library file, loaded by counterfactuals_main.jl; `run_training_cost` is Exercise 3.
#
# Local and stationary, no transitions. For each window of each crisis pair (the baseline
# and the crisis), the stationary equilibrium at the estimated vacancy costs with the
# training cost c(a_S) scaled by 1 − h, 1 and 1 + h (h = TRAINING_COST_STEP), by Newton
# continuation from the estimate; the solve must stay on the estimate's equilibrium (same
# number and rank of unskilled free-entry zeros, `unskilled_branch`). Welfare is the
# steady-state social flow per month (welfare.jl) with training costed at the estimated c,
# so the change in c is a transfer. Every outcome's elasticity to c is the central
# difference in logs between 1 − h and 1 + h. Headline per crisis: the welfare gain of
# 1 − h over 1 in the baseline and in the crisis window, and their ratio.
############################################################

# The training cost is scaled by 1 − h, 1 and 1 + h.
const TRAINING_COST_STEP   = 0.05
const TRAINING_COST_SCALES = (1.0 - TRAINING_COST_STEP, 1.0, 1.0 + TRAINING_COST_STEP)

# The θ_U scan of `unskilled_branch`: its range in ln θ_U around the point, and its nodes.
const BRANCH_SCAN_LOG = (-3.0, 1.5)
const BRANCH_SCAN_N   = 31

"""
    unskilled_branch(model, θU, θS, kU) -> NamedTuple

Every zero of unskilled free entry of `model` at θ_S = θS and cost kU on a scan of θ_U
(the sign changes of ln k_U(θ_U) − ln kU), with `rank` the position from the top of the
zero nearest θU. Two equilibria are on the same branch when their scans find as many zeros
and give them the same rank. Returns `(roots, rank, n_failed)`.
"""
function unskilled_branch(model::Model, θU::Float64, θS::Float64, kU::Float64)
    m = deepcopy(model)
    grid = exp.(range(log(θU) + BRANCH_SCAN_LOG[1], log(θU) + BRANCH_SCAN_LOG[2];
                      length = BRANCH_SCAN_N))
    g = map(grid) do θ
        r = solve_at_tightness!(m, θ, θS)
        r.ok ? log(m.unsk_par.k / kU) : NaN
    end
    roots = Float64[]
    for n in 1:(length(grid) - 1)
        (isfinite(g[n]) && isfinite(g[n + 1]) && sign(g[n]) != sign(g[n + 1])) || continue
        w = g[n] / (g[n] - g[n + 1])
        push!(roots, exp((1.0 - w) * log(grid[n]) + w * log(grid[n + 1])))
    end
    sort!(roots, rev = true)
    rank = isempty(roots) ? 0 : argmin(abs.(log.(roots) .- log(θU)))
    return (roots = roots, rank = rank, n_failed = count(!isfinite, g))
end

"""
    training_cost_point(model, s, ref) -> NamedTuple

The steady state of the estimated window `model` with its training cost scaled by `s`, at
its estimated vacancy costs, from its tightnesses; `ref` is the estimate's
`unskilled_branch`, which the solve at s ≠ 1 must keep. Returns the tightnesses, unemployment per
head and rates, the training and skilled shares and the wage premium (as the moments
measure them), and the social flow with its components, training costed at `model`'s c.
"""
function training_cost_point(model::Model, s::Float64, ref)
    kU = model.unsk_par.k;  kS = model.skl_par.k
    mc = with_training_cost(model, s)
    st = stationary_at_costs!(mc, kU, kS, (U = model.unsk_cache.θ, S = model.skl_cache.θ))
    st.ok || error(@sprintf("Training cost ×%.2f: no steady state at the estimated costs from the estimate (Newton residual %.2e).",
                            s, st.resid))
    if s != 1.0
        b = unskilled_branch(mc, st.θU, st.θS, kU)
        (length(b.roots) == length(ref.roots) && b.rank == ref.rank) || error(
            "Training cost ×$s leaves the estimate's equilibrium: unskilled free-entry zeros " *
            "$(round.(b.roots, digits = 5)) (rank $(b.rank)) against $(round.(ref.roots, digits = 5)) (rank $(ref.rank)).")
    end
    obj = compute_equilibrium_objects(mc);  mm = model_moments(obj)
    fl  = steady_state_flow(mc; resource_c = model.common.c).flow
    W   = net_flow(fl)
    out = (scale = s, theta_U = st.θU, theta_S = st.θS,
           uU = obj.agg_uU / obj.total_pop, uS = obj.agg_uS / obj.total_pop,
           u = (obj.agg_uU + obj.agg_uS) / obj.total_pop,
           ur_U = mm.ur_U, ur_S = mm.ur_S, ur_total = mm.ur_total,
           training_share = mm.training_share, skilled_share = mm.skilled_share,
           wage_premium = mm.wage_premium, W = W, W_over_A = W / exp(mc.common.A))
    return merge(out, NamedTuple{Tuple(Symbol("flow_", c) for c in COMPONENTS)}(Tuple(fl)),
                 (newton_iterations = st.iters, newton_residual = st.resid))
end

# The outcomes of `training_cost_point` that get an index and an elasticity.
const TC_OUTCOMES = (:theta_U, :theta_S, :uU, :uS, :u, :ur_U, :ur_S, :ur_total, :training_share,
                     :skilled_share, :wage_premium, :W, :W_over_A,
                     (Symbol("flow_", c) for c in COMPONENTS)...)

"""
    tc_elasticity(lo, hi) -> Float64

The elasticity to c by central difference in logs between the scales 1 − h and 1 + h; NaN
unless both values are positive.
"""
tc_elasticity(lo::Real, hi::Real) =
    lo > 0 && hi > 0 ? log(hi / lo) / log(TRAINING_COST_SCALES[3] / TRAINING_COST_SCALES[1]) : NaN

"""
    run_training_cost(crises)

For each crisis pair, writes
  training_cost_<crisis>.csv               one row per window and scale: every outcome, the
                                           social flow by component, and each outcome
                                           indexed to 100 at the estimated c (`index_*`)
  training_cost_elasticities_<crisis>.csv  one row per window and outcome: its value at the
                                           three scales and its elasticity to c
and `training_cost_headline.csv`, one row per crisis: the welfare gain (%) of the scale
1 − h over 1 in the baseline and in the crisis window, and their ratio.
"""
function run_training_cost(crises)
    dir = cf_log_start(out_counterfactuals(:training_cost), "Exercise 3: the steady state's response to the cost of training")
    lo, one, hi = TRAINING_COST_SCALES
    head = NamedTuple[]
    for crisis in crises
        L = load_transition(crisis);  wb, wc = _scenario_windows(crisis)
        rows = NamedTuple[];  erows = NamedTuple[];  gain = Dict{String,Float64}()
        for (role, window, m) in (("baseline", wb, L.m0), ("crisis", wc, L.m1))
            t0  = time()
            ref = unskilled_branch(m, m.unsk_cache.θ, m.skl_cache.θ, m.unsk_par.k)
            pts = Dict(s => training_cost_point(m, s, ref) for s in TRAINING_COST_SCALES)
            for s in TRAINING_COST_SCALES
                idx = NamedTuple{Tuple(Symbol("index_", o) for o in TC_OUTCOMES)}(
                    Tuple(100 * getfield(pts[s], o) / getfield(pts[one], o) for o in TC_OUTCOMES))
                push!(rows, merge((crisis = String(crisis), window = String(window), role = role), pts[s], idx))
            end
            for o in TC_OUTCOMES
                y = [getfield(pts[s], o) for s in TRAINING_COST_SCALES]
                push!(erows, (crisis = String(crisis), window = String(window), role = role,
                              outcome = String(o), value_low = y[1], value_estimate = y[2],
                              value_high = y[3], elasticity = tc_elasticity(y[1], y[3])))
            end
            gain[role] = 100 * (pts[lo].W / pts[one].W - 1)
            cf_log(dir, @sprintf("[%s] %-12s W ×%.2f %+.4f%%, ×%.2f %+.4f%% against c×1; elasticity of W %+.5f; %.0f s",
                                 crisis, window, lo, gain[role], hi, 100 * (pts[hi].W / pts[one].W - 1),
                                 tc_elasticity(pts[lo].W, pts[hi].W), time() - t0))
        end
        CSV.write(joinpath(dir, "training_cost_$(crisis).csv"), DataFrame(rows))
        CSV.write(joinpath(dir, "training_cost_elasticities_$(crisis).csv"), DataFrame(erows))
        push!(head, (crisis = String(crisis), window_baseline = String(wb), window_crisis = String(wc),
                     scale = lo, gain_baseline_pct = gain["baseline"], gain_crisis_pct = gain["crisis"],
                     ratio_crisis_to_baseline = gain["crisis"] / gain["baseline"]))
        cf_log(dir, @sprintf("[%s] welfare gain of c×%.2f: baseline %+.4f%%, crisis %+.4f%%, ratio %.3f",
                             crisis, lo, head[end].gain_baseline_pct, head[end].gain_crisis_pct,
                             head[end].ratio_crisis_to_baseline))
    end
    CSV.write(joinpath(dir, "training_cost_headline.csv"), DataFrame(head))
    return nothing
end

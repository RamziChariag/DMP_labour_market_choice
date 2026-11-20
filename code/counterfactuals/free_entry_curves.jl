############################################################
# free_entry_curves.jl — the shape of the free-entry conditions (RoySearch)
#
# Library file, loaded by counterfactuals_main.jl; `run_free_entry_curves` writes the data of
# the model notes' free-entry figure. Needs held_margin.jl (`free_entry_curve`, `curve_grid`,
# `skilled_residual`).
#
# At the estimate of the crisis window `FE_CURVE_CRISIS` (POINT_ESTIMATE), with the training
# cost c(a_S) scaled by each of `FE_CURVE_SCALES` and the vacancy costs at their estimates
# k̂_U, k̂_S:
#   unskilled  on the θ_U grid of Exercise 1's curve, θ_S solving skilled free entry at k̂_S,
#              and the vacancy cost unskilled free entry supports there, ln(k_U(θ_U)/k̂_U).
#              Given both tightnesses the steady state does not depend on k_U, so neither
#              does the curve: at a cost k_U the equilibria are where it equals ln(k_U/k̂_U).
#              With it, the unskilled unemployed pool's mean a_U and the share of the pool
#              at the corner p* = 0.
#   skilled    at the estimated θ_U, on the same span around the estimated θ_S, the cost
#              skilled free entry supports, ln(k_S(θ_S)/k̂_S).
# And, at the estimated training cost, the local minima and maxima of the unskilled curve:
# between a minimum's and a maximum's values lie the vacancy costs k_U with three equilibria.
############################################################

# The crisis window whose estimate the curves are drawn at, and the training-cost scales.
const FE_CURVE_CRISIS = :covid
const FE_CURVE_SCALES = (0.85, 1.0, 1.075)

"""
    curve_extrema(θ, g) -> Vector{NamedTuple}

The interior local minima and maxima of a curve g on the nodes θ (uniform in ln θ), at
nodes whose two neighbours are finite: the node, and the vertex of the parabola in ln θ
through it and its neighbours. Returns `(kind, node, θ_node, g_node, θ_vertex, g_vertex)`
per extremum.
"""
function curve_extrema(θ::AbstractVector, g::AbstractVector)
    rows = NamedTuple[]
    for n in 2:(length(g) - 1)
        g0, g1, g2 = g[n - 1], g[n], g[n + 1]
        all(isfinite, (g0, g1, g2)) || continue
        kind = (g1 < g0 && g1 < g2) ? "local_min" : (g1 > g0 && g1 > g2) ? "local_max" : ""
        isempty(kind) && continue
        h = (log(θ[n + 1]) - log(θ[n - 1])) / 2;  c = g0 - 2g1 + g2
        push!(rows, (kind = kind, node = n, θ_node = θ[n], g_node = g1,
                     θ_vertex = exp(log(θ[n]) + h * (g0 - g2) / (2c)), g_vertex = g1 - (g2 - g0)^2 / (8c)))
    end
    return rows
end

"""
    run_free_entry_curves(crises)

Writes, for the crisis window `FE_CURVE_CRISIS` (`crises` is not read: the figure is drawn at
that one estimate),
  fe_curve_unskilled_<crisis>.csv          per scale and θ_U node, every skilled root: θ_S,
                                           ln(k_U/k̂_U), the unskilled unemployed pool's mean
                                           a_U and its share at the corner p* = 0
  fe_curve_skilled_<crisis>.csv            per scale and θ_S node, at the estimated θ_U:
                                           ln(k_S/k̂_S)
  fe_curve_unskilled_extrema_<crisis>.csv  the local extrema of the unskilled curve at the
                                           estimated training cost (`curve_extrema` on the
                                           continued branch)
"""
function run_free_entry_curves(crises)
    crisis = FE_CURVE_CRISIS
    dir = cf_log_start(out_counterfactuals(:free_entry_curves), "Free-entry curves at the $(crisis) crisis estimate")
    L  = load_transition(crisis);  m1 = L.m1
    kU = m1.unsk_par.k;  kS = m1.skl_par.k;  θ1 = (U = m1.unsk_cache.θ, S = m1.skl_cache.θ)
    cf_log(dir, @sprintf("[%s] estimate: θ_U %.6f, θ_S %.6f, k_U %.6f, k_S %.6f (months of mean new-match output)",
                         crisis, θ1.U, θ1.S, kU, kS))
    gU = curve_grid(θ1.U);  gS = curve_grid(θ1.S)
    urows = NamedTuple[];  srows = NamedTuple[];  xrows = NamedTuple[]
    for s in FE_CURVE_SCALES
        t0 = time();  m = with_training_cost(m1, s)
        cu = free_entry_curve(m, kU, kS, θ1, gU; objects = true)
        for r in cu.roots
            push!(urows, (scale = s, theta_U = r.θU, theta_S = r.θS, root_rank = r.root_rank,
                          log_kU_ratio = r.gU, mean_aU_unskilled_unemployed = r.mean_aU_pool,
                          corner_share = r.corner_share))
        end
        ms = deepcopy(m)                       # the skilled curve's own warm-started copy
        for θS in gS
            push!(srows, (scale = s, theta_U = θ1.U, theta_S = θS, log_kS_ratio = skilled_residual(ms, θ1.U, log(θS), kS)))
        end
        s == 1.0 && append!(xrows, [(scale = s, kind = r.kind, theta_U_node = r.θ_node,
                                     theta_S_node = cu.θS[r.node], log_kU_ratio_node = r.g_node,
                                     theta_U_vertex = r.θ_vertex, log_kU_ratio_vertex = r.g_vertex)
                                    for r in curve_extrema(cu.θU, cu.gU)])
        cf_log(dir, @sprintf("[%s] c×%.3f: unskilled curve %d of %d nodes with a skilled root, skilled curve %d of %d solved; %.0f s",
                             crisis, s, count(isfinite, cu.gU), length(gU),
                             count(r -> r.scale == s && isfinite(r.log_kS_ratio), srows), length(gS), time() - t0))
    end
    CSV.write(joinpath(dir, "fe_curve_unskilled_$(crisis).csv"), DataFrame(urows))
    CSV.write(joinpath(dir, "fe_curve_skilled_$(crisis).csv"), DataFrame(srows))
    CSV.write(joinpath(dir, "fe_curve_unskilled_extrema_$(crisis).csv"),
              DataFrame((f => [getfield(r, f) for r in xrows]
                         for f in (:scale, :kind, :theta_U_node, :theta_S_node, :log_kU_ratio_node,
                                   :theta_U_vertex, :log_kU_ratio_vertex))...))
    for r in xrows
        cf_log(dir, @sprintf("[%s] unskilled curve at c×1, %s: θ_U %.5f, ln(k_U/k̂_U) %+.5f (k_U/k̂_U %.5f); node θ_U %.5f, %+.5f",
                             crisis, r.kind, r.theta_U_vertex, r.log_kU_ratio_vertex, exp(r.log_kU_ratio_vertex),
                             r.theta_U_node, r.log_kU_ratio_node))
    end
    isempty(xrows) && cf_log(dir, "[$crisis] the unskilled curve at c×1 has no interior local extremum on the grid")
    return nothing
end

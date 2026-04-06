############################################################
# policy_solver.jl — Solve stationary equilibria under
#                     education-subsidy counterfactuals
#
# Public API:
#   solve_policy_experiment(spec, common, regime, unsk_par,
#                           skl_par, sim; Nx, Np_U, Np_S)
#       → PolicyResult
#
#   solve_all_policies(specs, common, regime, unsk_par,
#                      skl_par, sim; kwargs...)
#       → PolicyTable
############################################################


# ──────────────────────────────────────────────────────────
# Helper: extract scalar outcomes from a solved model
# ──────────────────────────────────────────────────────────

"""
    _extract_policy_result(model, obj, spec, converged) → PolicyResult

Build a PolicyResult from the solved model and its equilibrium objects.
`obj` is the NamedTuple from `compute_equilibrium_objects`.
"""
function _extract_policy_result(
    model     :: Model,
    obj,
    spec      :: PolicySpec,
    converged :: Bool,
)
    xg  = obj.xg
    wx  = obj.wx
    Nx  = obj.Nx
    NpS = obj.NpS

    # ── Training cutoff x_bar ─────────────────────────────
    # First x where τ(x) switches to 1
    x_bar_idx = findfirst(obj.tauT .> 0.5)
    x_bar = isnothing(x_bar_idx) ? 1.0 : xg[x_bar_idx]

    # ── Mean wages (employment-weighted) ──────────────────
    # Unskilled
    wU_num = 0.0;  wU_den = 0.0
    pgU = obj.pgU;  wpU = obj.wpU
    for ix in 1:Nx, jp in 1:length(pgU)
        e_ij = obj.eU_surface[ix, jp]
        w_ij = obj.wU_surface[ix, jp]
        if !isnan(w_ij) && e_ij > 1e-16
            m = e_ij * wx[ix] * wpU[jp]
            wU_num += m * w_ij
            wU_den += m
        end
    end
    mean_wage_U = wU_den > 1e-14 ? wU_num / wU_den : 0.0

    # Skilled
    wS_num = 0.0;  wS_den = 0.0
    pg  = obj.pg;   wpS = obj.wpS
    for ix in 1:Nx
        poj_ix = clamp01(obj.poj[ix])
        for jp in 1:NpS
            e_ij = obj.eS_mat[ix, jp]
            e_ij <= 1e-16 && continue
            m = e_ij * wx[ix] * wpS[jp]
            w_ij = pg[jp] < poj_ix ?
                   obj.wS1_surface[ix, jp] : obj.wS0_surface[ix, jp]
            if !isnan(w_ij)
                wS_num += m * w_ij
                wS_den += m
            end
        end
    end
    mean_wage_S = wS_den > 1e-14 ? wS_num / wS_den : 0.0

    # Wage premium
    wage_premium = (mean_wage_U > 1e-14 && mean_wage_S > 1e-14) ?
                   log(mean_wage_S) - log(mean_wage_U) : NaN

    # ── Welfare at quantiles ──────────────────────────────
    # Find grid indices closest to x = 0.25, 0.50, 0.75
    function _nearest_ix(target)
        _, idx = findmin(abs.(xg .- target))
        return idx
    end
    ix25 = _nearest_ix(0.25)
    ix50 = _nearest_ix(0.50)
    ix75 = _nearest_ix(0.75)

    return PolicyResult(
        spec, converged,
        x_bar,
        obj.agg_uU, obj.agg_t, obj.agg_eU, obj.agg_mU,
        obj.agg_uS, obj.agg_eS, obj.agg_mS,
        obj.ur_U, obj.ur_S, obj.ur_total,
        obj.agg_mS / max(obj.agg_uU + obj.agg_eU + obj.agg_mS, 1e-14),   # skilled_share (LF-based)
        obj.agg_t  / max(obj.total_pop, 1e-14),   # training_share
        obj.thetaU, obj.thetaS,
        obj.f_U, obj.f_S,
        obj.sep_rate_U, obj.sep_rate_S, obj.ee_rate_S,
        mean_wage_U, mean_wage_S, wage_premium,
        obj.UU[ix25], obj.UU[ix50], obj.UU[ix75],
        obj.US[ix25], obj.US[ix50], obj.US[ix75],
    )
end


# ──────────────────────────────────────────────────────────
# Solve one experiment
# ──────────────────────────────────────────────────────────

"""
    solve_policy_experiment(spec, common, regime, unsk_par, skl_par, sim;
                            Nx, Np_U, Np_S) → PolicyResult

Perturb the parameters according to `spec`, solve the new stationary
equilibrium, and return a `PolicyResult`.
"""
function solve_policy_experiment(
    spec     :: PolicySpec,
    common   :: CommonParams,
    regime   :: RegimeParams,
    unsk_par :: UnskilledParams,
    skl_par  :: SkilledParams,
    sim      :: SimParams;
    Nx   :: Int = 200,
    Np_U :: Int = 200,
    Np_S :: Int = 200,
)
    # Apply policy perturbation
    regime_p, common_p = perturb_params(regime, common, spec)

    # Solve
    model, sr = solve_model(common_p, regime_p, unsk_par, skl_par, sim;
                             Nx = Nx, Np_U = Np_U, Np_S = Np_S)

    if !sr.ok
        @printf("  ⚠  %s did not converge\n", spec.label)
    end

    # Equilibrium objects
    obj = compute_equilibrium_objects(model)

    return _extract_policy_result(model, obj, spec, sr.ok)
end


# ──────────────────────────────────────────────────────────
# Solve all experiments
# ──────────────────────────────────────────────────────────

"""
    solve_all_policies(specs, common, regime, unsk_par, skl_par, sim;
                       baseline_label, Nx, Np_U, Np_S) → PolicyTable

Solve each experiment in `specs` and collect the results into a `PolicyTable`.
"""
function solve_all_policies(
    specs    :: Vector{PolicySpec},
    common   :: CommonParams,
    regime   :: RegimeParams,
    unsk_par :: UnskilledParams,
    skl_par  :: SkilledParams,
    sim      :: SimParams;
    baseline_label :: String = "Baseline",
    Nx   :: Int = 200,
    Np_U :: Int = 200,
    Np_S :: Int = 200,
)
    results = PolicyResult[]

    for (i, spec) in enumerate(specs)
        @printf("\n── Experiment %d/%d: %s ──\n", i, length(specs), spec.label)
        flush(stdout)

        pr = solve_policy_experiment(
            spec, common, regime, unsk_par, skl_par, sim;
            Nx = Nx, Np_U = Np_U, Np_S = Np_S,
        )

        push!(results, pr)

        @printf("  converged=%s  x̄=%.4f  ur=%.4f  θU=%.6f  θS=%.6f  sksh=%.4f  w̄_U=%.4f  w̄_S=%.4f\n",
                pr.converged, pr.x_bar, pr.ur_total, pr.θU, pr.θS,
                pr.skilled_share, pr.mean_wage_U, pr.mean_wage_S)
        flush(stdout)
    end

    return PolicyTable(baseline_label, results)
end

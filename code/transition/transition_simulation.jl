############################################################
# transition_simulation.jl — Run a single-scenario RoySearch
#                             transition and persist the result.
#
# LIBRARY FILE — loaded by transition_main.jl (the only supported entry
# point); it does not auto-execute when included.
#
# Public API
#   run_transition_simulation(scenario, w_suffix; point, T_max, N_steps, tol, maxit, damp)
#     1. Loads the two windows' estimates at the same `point` (:smm or :mcmc) and checks
#        that the pair is one: the same grids and the same ability distribution and
#        demographic rates.
#     2. Solves the selected stationary equilibrium of each window at its own grid and
#        its own estimation settings.
#     3. Chooses the horizon by an accuracy test and solves the transition there.
#     4. Serialises the bundle to <TRANS_OUT_DIR>/transition_<scenario><W>_<point>.jls.
#
# Expects in scope (provided by transition_main.jl): the solver, SMM and transition
# modules, paths.jl and JSON3.
############################################################

# Bundle shape of the transition output.  Bumped only when its field set changes.
const TRANSITION_SCHEMA = 2

# The horizon test: the horizon doubles, from the fine grid's span, until no reported
# series moves on the first HORIZON_TEST_MONTHS by more than HORIZON_TOL relative to its
# own scale; HORIZON_MAX bounds the search.  A series in logs is held to HORIZON_TOL in log
# units instead: its change is relative already, and dividing it by the size of the log
# (about 6.5 for the wages at the estimates) would loosen the test that many times.  The
# tolerance is ten times the outer loop's default free-entry tolerance, so the test is not
# reading the iteration's own noise.
const HORIZON_TEST_MONTHS = 36.0
const HORIZON_TOL         = 1e-3
const HORIZON_MAX         = 7680.0

# The series the horizon test compares, every one a reported path, and those in logs.
const REPORTED_SERIES = (:θU, :θS, :ur_U, :ur_S, :ur_total, :skilled_share,
                         :training_share, :mean_wage_U, :mean_wage_S)
const LOG_SERIES      = (:mean_wage_U, :mean_wage_S)

# A scenario is a transition pair of paths.jl (PAIRS): from the base window to `crisis_<pair>`.
function _scenario_windows(scenario::Symbol)
    String(scenario) in PAIRS ||
        error("Unknown scenario: $scenario. Must be one of :" * join(PAIRS, ", :") * ".")
    wc = Symbol("crisis_", scenario)
    return (PAIR_BASE[wc], wc)
end

"""
    crisis_window_months(window) -> (months, ym_start, ym_end, label)

The crisis window's length in months, from `data/derived/windows.json`, the list every
other layer uses (a counterfactual window's is its parent's, `data_window`).
"""
function crisis_window_months(window::Symbol)
    w  = JSON3.read(read(joinpath(PROJECT_ROOT, "data", "derived", "windows.json"), String))
    wd = w.windows[data_window(window)]
    ys, ms = divrem(wd.ym_start, 100);  ye, me = divrem(wd.ym_end, 100)
    return (months = 12 * (ye - ys) + (me - ms) + 1, ym_start = wd.ym_start,
            ym_end = wd.ym_end, label = String(wd.label))
end

"""
    point_estimate(window, w_suffix, point) -> (bundle, θ, source)

The estimate a window is read at: `:smm` the SMM estimate, `estimate_<window><W>_smm.jls`;
`:mcmc` the chain's, `estimate_<window><W>_mcmc.jls` (the posterior mean of the retained
draws). Both windows of a pair are read at the same `point`. The bundle must be an estimate
under this code (`require_current_estimate`), and a chain's from a chain that did not abort.
"""
function point_estimate(window::Symbol, w_suffix::String, point::Symbol)
    point in (:smm, :mcmc) || error("POINT_ESTIMATE must be :smm or :mcmc, got :$point.")
    file = estimate_path(window, w_suffix; point = point)
    isfile(file) || error("$window: $(basename(file)) not found.")
    b = read_bundle(file)
    (b === nothing || b.result === nothing || b.spec === nothing) &&
        error("$window: $(basename(file)) is unreadable or carries no result/spec.")
    require_current_estimate(b, file)
    b.stage === :mcmc_aborted && error(
        "$window: the chain behind $(basename(file)) aborted; its draws are not a " *
        "stationary sample, so the file holds no estimate.")
    return (bundle = b, θ = b.result.theta_opt, source = basename(file))
end

"""
    check_pair(base, crisis, wb, wc)

A transition runs on one grid and one population, so the two windows must share the grid
sizes of their runs, the ability distribution and the demographic and training rates, and a
counterfactual crisis window the training cost it holds at the base estimate.
"""
function check_pair(base, crisis, wb::Symbol, wc::Symbol)
    rb = base.bundle.spec.run;  rc = crisis.bundle.spec.run
    (rb.Nx, rb.Np_U, rb.Np_S) == (rc.Nx, rc.Np_U, rc.Np_S) ||
        error("$wb and $wc were estimated on different grids: (Nx, Np_U, Np_S) = " *
              "$((rb.Nx, rb.Np_U, rb.Np_S)) against $((rc.Nx, rc.Np_U, rc.Np_S)).")
    cb, _, _ = unpack_θ(base.θ, base.bundle.spec)
    cc, _, _ = unpack_θ(crisis.θ, crisis.bundle.spec)
    for f in (last.(base_held_params(wc))..., :r, :ν, :φ)
        getfield(cb, f) == getfield(cc, f) ||
            error("$wb and $wc are not a pair: $f = $(getfield(cb, f)) in the base point and " *
                  "$(getfield(cc, f)) in the crisis point. The crisis window must hold " *
                  "$(join(last.(base_held_params(wc)), ", ")) at the base estimate it was paired with.")
    end
    return (Nx = rb.Nx, Np_U = rb.Np_U, Np_S = rb.Np_S)
end

"""
    solve_endpoint(pt, grid, label) -> equilibrium

The equilibrium at the window's point (`equilibrium_at`): the solve at its tightnesses, with
its vacancy costs computed there, on its own grid and under its own estimation settings,
verbosity aside.  The path holds each window's costs fixed from those.
"""
function solve_endpoint(pt, grid, label::String)
    spec = pt.bundle.spec
    sim  = SimParams(; (f => getfield(spec.sim, f) for f in fieldnames(SimParams))..., verbose = 1)
    @printf("Solving %s stationary equilibrium (%s)...\n", label, pt.source);  flush(stdout)
    sel = equilibrium_at(pt.θ, spec; Nx = grid.Nx, Np_U = grid.Np_U, Np_S = grid.Np_S, sim = sim)
    sel.model === nothing && error("No certified equilibrium at the $label point.")
    @printf("  θ_U = %.6f   θ_S = %.6f   k_U = %.6f   k_S = %.6f   Q = %.6f\n\n",
            sel.model.unsk_cache.θ, sel.model.skl_cache.θ, sel.k.U, sel.k.S, sel.Q)
    flush(stdout)
    return sel
end

"""
    stationary_record(model) -> TransitionResult

The series the transition reports, at the steady state `model`: a one-step transition of
the model to itself, whose dates both hold that steady state to the stationary solve's
tolerance (the self-transition property).  It is the pre-switch value of every reported
series, read with the path's own definitions; date 0 of a path is not, because it comes
after the date-0 reallocation.
"""
stationary_record(model) =
    solve_transition(deepcopy(model), deepcopy(model),
                     TransitionParams(T_max = 0.5, N_steps = 1, maxit = 1, verbose = false);
                     scenario = :steady_state)

"""
    solve_with_horizon(sel0, sel1, scenario; T_max, N_steps, tol, maxit, damp)

Solve the transition at a horizon chosen by its accuracy test.  The first trial ends at the
fine grid's span; each next one doubles the horizon (the fine grid is shared, so the dates
of the test window coincide) and starts from the previous tightness path, interpolated onto
its dates.  The test passes when no reported series moves by more than HORIZON_TOL on the
first HORIZON_TEST_MONTHS: relative to its scale, or in log units for the series in logs.
Returns the accepted solve and the record of every trial.
"""
function solve_with_horizon(sel0, sel1, scenario::Symbol; T_max, N_steps, tol, maxit, damp)
    trials = NamedTuple[]
    prev = nothing;  H = Float64(T_max)
    while true
        tp = TransitionParams(T_max = T_max, N_steps = N_steps, horizon = H, tol = tol,
                              maxit = maxit, damp = damp, verbose = true)
        m0 = deepcopy(sel0.model);  m1 = deepcopy(sel1.model)
        @printf("── horizon %.0f months (%d dates) ──\n", H, length(time_grid(tp)));  flush(stdout)
        guess = prev === nothing ? nothing :
                (tgrid = prev.path.tgrid, θU = prev.path.θU, θS = prev.path.θS)
        res, path = solve_transition_path(m0, m1, tp; scenario = scenario, guess = guess)
        res.converged || @warn "transition at horizon $H did not converge (gap $(res.final_dist))"
        cur = (tp = tp, path = path, res = res, model_z1 = m1)
        change = NaN
        if prev !== nothing
            k = count(<=(HORIZON_TEST_MONTHS + 1e-9), prev.res.tgrid)
            change = maximum(REPORTED_SERIES) do s
                a = getfield(cur.res, s)[1:k];  b = getfield(prev.res, s)[1:k]
                scale = s in LOG_SERIES ? 1.0 : max(maximum(abs, b), 1e-300)
                maximum(abs, a .- b) / scale
            end
        end
        push!(trials, (horizon = H, dates = length(path.tgrid), iterations = res.n_iter,
                       converged = res.converged, gap = res.final_dist, change_first_months = change))
        @printf("   horizon %.0f: %d iterations, gap %.2e, change on months 0-%.0f vs previous %.3e\n",
                H, res.n_iter, res.final_dist, HORIZON_TEST_MONTHS, change);  flush(stdout)
        (isfinite(change) && change < HORIZON_TOL) && return (accepted = cur, trials = trials)
        2H > HORIZON_MAX && error("No horizon up to $(HORIZON_MAX) months passes the accuracy test " *
                                  "(last change $change against $(HORIZON_TOL)).")
        prev = cur;  H *= 2
    end
end

"""
    run_transition_simulation(scenario, w_suffix; point = :smm,
                              T_max = 120.0, N_steps = 240, tol = 1e-4, maxit = 200,
                              damp = 0.3) -> String

Solve the backward–forward transition for one scenario and serialise the bundle to
`transition_file(scenario, w_suffix, point)`.  Both windows of the scenario are read at
`point` (`point_estimate`).  `T_max`, `N_steps` are the fine grid; the horizon is
chosen by the test in `solve_with_horizon`.  Returns the path of the saved file.
"""
function run_transition_simulation(scenario::Symbol, w_suffix::String;
                                   point::Symbol = :smm, T_max::Real = 120.0,
                                   N_steps::Int = 240, tol::Real = 1e-4, maxit::Int = 200,
                                   damp::Real = TransitionParams().damp)
    wb, wc = _scenario_windows(scenario)
    fb = estimate_path(wb, w_suffix; point = point)
    fc = estimate_path(wc, w_suffix; point = point)
    println("\n" * "="^65)
    println("  Transition simulation — $scenario   (W = $w_suffix, point = $point)")
    println("="^65)
    @printf("  Baseline bundle: %s\n  Crisis bundle:   %s\n\n", fb, fc);  flush(stdout)

    pb = point_estimate(wb, w_suffix, point);  pc = point_estimate(wc, w_suffix, point)
    grid = check_pair(pb, pc, wb, wc)
    @printf("  Grid (from both bundles' runs): Nx=%d  Np_U=%d  Np_S=%d\n\n", grid.Nx, grid.Np_U, grid.Np_S)

    sel0 = solve_endpoint(pb, grid, "baseline (z₀)")
    sel1 = solve_endpoint(pc, grid, "crisis (z₁)")
    eq_z0 = compute_equilibrium_objects(sel0.model)
    eq_z1 = compute_equilibrium_objects(sel1.model)

    sol = solve_with_horizon(sel0, sel1, scenario; T_max = T_max, N_steps = N_steps,
                             tol = tol, maxit = maxit, damp = damp)
    path = sol.accepted.path;  result = sol.accepted.res;  tp = sol.accepted.tp
    cw   = crisis_window_months(wc)
    stats = path_statistics(path, sol.accepted.model_z1)
    revs  = policy_reversals(path, sol.accepted.model_z1)

    println("="^65)
    @printf("  %s: converged %s (%d iterations, free-entry gap %.3e), horizon %.0f months\n",
            scenario, result.converged, result.n_iter, result.final_dist, tp.horizon)
    @printf("  θ_U: %.4f → %.4f    θ_S: %.4f → %.4f\n", result.θU[1], result.θU[end],
            result.θS[1], result.θS[end])
    @printf("  crossing choice: %d reversals in %d cells; largest skilled mass on them %.3e\n",
            revs.d_reversals, revs.cells_d_reversing, revs.max_mS_on_reversing_cells)
    flush(stdout)

    mkpath(TRANS_OUT_DIR)
    out_file = transition_file(scenario, w_suffix, point)
    record(m) = (common = m.common, unsk_par = m.unsk_par, skl_par = m.skl_par)
    open(out_file, "w") do io
        serialize(io, (
            schema        = TRANSITION_SCHEMA,
            version       = ROYSEARCH_VERSION,
            provenance    = run_provenance(window = Symbol("transition_", scenario),
                                           w_suffix = w_suffix, version = ROYSEARCH_VERSION),
            scenario      = scenario,
            windows       = (base = wb, crisis = wc),
            point         = point,
            bundles       = (base = fb, crisis = fc),
            sources       = (base = pb.source, crisis = pc.source),
            theta         = (base = pb.θ, crisis = pc.θ),
            parameters    = (base = record(sel0.model), crisis = record(sel1.model)),
            settings      = (grid = grid, T_max = tp.T_max, N_steps = tp.N_steps, dt = tp.dt,
                             horizon = tp.horizon, tail_growth = TAIL_GROWTH,
                             forward_substep = FORWARD_SUBSTEP, tol = tp.tol, maxit = tp.maxit,
                             damp = tp.damp, sim_base = pb.bundle.spec.sim,
                             sim_crisis = pc.bundle.spec.sim,
                             horizon_test = (months = HORIZON_TEST_MONTHS, tol = HORIZON_TOL,
                                             trials = sol.trials)),
            crisis_window = (months = cw.months, ym_start = cw.ym_start, ym_end = cw.ym_end,
                             label = cw.label, in_window = path.tgrid .< cw.months),
            result        = result,
            baseline      = stationary_record(sel0.model),
            path          = (; (f => getfield(path, f) for f in fieldnames(TransitionPath))...),
            statistics    = stats,
            reversals     = revs,
            model_z0      = sel0.model,
            model_z1      = sel1.model,
            eq_z0         = eq_z0,
            eq_z1         = eq_z1,
        ))
    end
    @printf("\nSaved → %s (%.1f MB)\n", out_file, filesize(out_file) / 2^20);  flush(stdout)
    return out_file
end

# ──────────────────────────────────────────────────────────
# Stand-alone safety net — this file is a library.
# ──────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    error("transition_simulation.jl is a library and cannot be run standalone. " *
          "Use transition_main.jl as the entry point.")
end

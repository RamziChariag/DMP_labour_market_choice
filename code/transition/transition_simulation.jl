############################################################
# transition_simulation.jl — one transition, from the stored estimates to the saved bundle
#
# Library file, loaded by transition_main.jl. `run_transition_simulation` reads both
# windows of a pair at the same point estimate, checks they are one pair, solves each
# window's selected steady state, solves the path to a horizon at the last month of the
# crisis data window, adds the per-type date-0 values (`date0_types`) and serialises the bundle
# (`transition_file`). Expects the solver, SMM and transition files, paths.jl and JSON3.
############################################################

# Shape of the transition bundle; bumped only when its field set changes.
const TRANSITION_SCHEMA = 3

"""
    _scenario_windows(scenario) -> (base window, crisis window)

The two data windows of the transition pair `scenario` (one of `PAIRS`).
"""
function _scenario_windows(scenario::Symbol)
    String(scenario) in PAIRS ||
        error("Unknown scenario: $scenario. Must be one of :" * join(PAIRS, ", :") * ".")
    wc = Symbol("crisis_", scenario)
    return (PAIR_BASE[wc], wc)
end

"""
    crisis_window_months(window) -> (months, ym_start, ym_end, label)

The length in months, the first and last months and the label of `window`, from
`data/derived/windows.json`.
"""
function crisis_window_months(window::Symbol)
    w  = JSON3.read(read(joinpath(PROJECT_ROOT, "data", "derived", "windows.json"), String))
    wd = w.windows[window]
    ys, ms = divrem(wd.ym_start, 100);  ye, me = divrem(wd.ym_end, 100)
    return (months = 12 * (ye - ys) + (me - ms) + 1, ym_start = wd.ym_start,
            ym_end = wd.ym_end, label = String(wd.label))
end

"""
    point_estimate(window, w_suffix, point) -> (bundle, θ, source)

The estimate of data window `window` at `point` (`:smm` the SMM estimate, `:mcmc` the
chain's posterior mean), read from the bundle of the estimation window that holds it, as
that period's own estimate for a baseline period (`period_bundle`). Errors unless the
bundle is current (`require_current_estimate`) and, for a chain, did not abort. Returns the
bundle, the parameter vector and the file name.
"""
function point_estimate(window::Symbol, w_suffix::String, point::Symbol)
    point in (:smm, :mcmc) || error("POINT_ESTIMATE must be :smm or :mcmc, got :$point.")
    file = estimate_path(estimation_window(window), w_suffix; point = point)
    isfile(file) || error("$window: $(basename(file)) not found.")
    b = read_bundle(file)
    (b === nothing || b.result === nothing || b.spec === nothing) &&
        error("$window: $(basename(file)) is unreadable or carries no result/spec.")
    require_current_estimate(b, file)
    b.stage === :mcmc_aborted && error(
        "$window: the chain behind $(basename(file)) aborted; its draws are not a " *
        "stationary sample, so the file holds no estimate.")
    b = period_bundle(b, window)
    return (bundle = b, θ = b.result.theta_opt, source = basename(file))
end

"""
    check_pair(base, crisis, wb, wc)

Error unless the two windows can share a transition: the same grid sizes, the same r, ν, φ,
and every `CRISIS_HELD` parameter equal. Returns the grid sizes `(Nx, Np_U, Np_S)`.
"""
function check_pair(base, crisis, wb::Symbol, wc::Symbol)
    rb = base.bundle.spec.run;  rc = crisis.bundle.spec.run
    (rb.Nx, rb.Np_U, rb.Np_S) == (rc.Nx, rc.Np_U, rc.Np_S) ||
        error("$wb and $wc were estimated on different grids: (Nx, Np_U, Np_S) = " *
              "$((rb.Nx, rb.Np_U, rb.Np_S)) against $((rc.Nx, rc.Np_U, rc.Np_S)).")
    pb = unpack_θ(base.θ, base.bundle.spec)
    pc = unpack_θ(crisis.θ, crisis.bundle.spec)
    for (blk, f) in (CRISIS_HELD..., (:common, :r), (:common, :ν), (:common, :φ))
        vb = _baseline_param_value(blk, f, pb...);  vc = _baseline_param_value(blk, f, pc...)
        vb == vc ||
            error("$wb and $wc are not a pair: $blk:$f = $vb in the base point and $vc in the " *
                  "crisis point. The crisis window must hold " *
                  "$(join(last.(CRISIS_HELD), ", ")) at the baseline estimate it was paired with.")
    end
    return (Nx = rb.Nx, Np_U = rb.Np_U, Np_S = rb.Np_S)
end

"""
    solve_endpoint(pt, grid, label) -> equilibrium

The selected equilibrium at the window's point (`equilibrium_at`), on `grid` and under the
window's own estimation settings (verbosity aside), with its vacancy costs computed there;
the path holds those costs fixed.
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

The reported series at the steady state `model`, from a one-step transition of the model
to itself: the pre-switch value of every series under the path's own definitions (date 0
of a path comes after the date-0 reallocation and is not).
"""
stationary_record(model) =
    solve_transition(deepcopy(model), deepcopy(model),
                     TransitionParams(T_max = 0.5, N_steps = 1, maxit = 1, verbose = false);
                     scenario = :steady_state)

"""
    run_transition_simulation(scenario, w_suffix; point = :smm, dt = 0.5, tol = 5e-4,
                              maxit = 200, damp = 1.0) -> String

Solve the transition of the pair `scenario` with both windows read at `point` and
serialise it to `transition_file(scenario, w_suffix, point)`. The horizon is the length of
the crisis data window (`crisis_window_months`): the path's last date is the last month of
the data the crisis estimate was measured on. `dt` is the step in months; `tol`, `maxit`,
`damp` the outer fixed point. Returns the saved file's path.
"""
function run_transition_simulation(scenario::Symbol, w_suffix::String;
                                   point::Symbol = :smm, dt::Real = 0.5, tol::Real = 5e-4,
                                   maxit::Int = 200, damp::Real = 1.0)
    wb, wc = _scenario_windows(scenario)
    fb = estimate_path(estimation_window(wb), w_suffix; point = point)
    fc = estimate_path(estimation_window(wc), w_suffix; point = point)
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

    cw = crisis_window_months(wc)
    H  = Float64(cw.months)
    tp = TransitionParams(T_max = H, N_steps = round(Int, H / dt), horizon = H, tol = tol,
                          maxit = maxit, damp = damp, verbose = true)
    m0 = deepcopy(sel0.model);  m1 = deepcopy(sel1.model)
    @printf("── horizon %.0f months, %s to %s (%d dates) ──\n", H, cw.ym_start, cw.ym_end,
            length(time_grid(tp)));  flush(stdout)
    result, path = solve_transition_path(m0, m1, tp; scenario = scenario)
    result.converged || @warn "transition did not converge (gap $(result.final_dist))"
    stats = path_statistics(path, m1)
    revs  = policy_reversals(path, m1)

    println("="^65)
    @printf("  %s: converged %s (%d iterations, free-entry gap %.3e), horizon %.0f months\n",
            scenario, result.converged, result.n_iter, result.final_dist, tp.horizon)
    @printf("  θ_U: %.4f → %.4f    θ_S: %.4f → %.4f\n", result.θU[1], result.θU[end],
            result.θS[1], result.θS[end])
    @printf("  crossing choice: %d reversals in %d cells; largest skilled mass on them %.3e\n",
            revs.d_reversals, revs.cells_d_reversing, revs.max_mS_on_reversing_cells)
    flush(stdout)

    print("Per-type values and masses at date 0 and at the baseline... ");  flush(stdout)
    types = date0_types(path, sel0.model, sel1.model, tp)
    println("done.");  flush(stdout)

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
                             sim_crisis = pc.bundle.spec.sim),
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
            types         = types,
        ))
    end
    @printf("\nSaved → %s (%.1f MB)\n", out_file, filesize(out_file) / 2^20);  flush(stdout)
    return out_file
end

# This file is a library.
if abspath(PROGRAM_FILE) == @__FILE__
    error("transition_simulation.jl is a library and cannot be run standalone. " *
          "Use transition_main.jl as the entry point.")
end

############################################################
# transition_simulation.jl — Run a single-scenario transition
#                              and persist the result.
#                              (PS(x) = γ·x^{γ−1} variant)
#
# This file is a LIBRARY — it does not auto-execute when included.
# It is loaded by transition_main.jl, which is the only supported
# entry point.
#
# Public API
# ──────────
#   run_transition_simulation(scenario::Symbol, w_cond_target::Float64;
#                             Nx, Np_U, Np_S,
#                             T_max, N_steps, tol, maxit, damp)
#       1. Loads the baseline + crisis SMM bundles for that scenario
#       2. Re-builds the parameter structs
#       3. Solves the two stationary equilibria z₀, z₁
#       4. Runs the backward-forward transition algorithm
#       5. Saves a rich TransitionResult to
#            <TRANS_OUT_DIR>/transition_<scenario><W>.jld2
#       6. Returns the absolute path of the saved file.
#
# Expects the following to be in scope (provided by transition_main.jl):
#   - all solver modules           (solve_model, SimParams, …)
#   - all smm modules              (_load_smm_bundle, unpack_θ, …)
#   - transition_params.jl         (TransitionParams)
#   - transition_solver.jl         (solve_transition)
#   - SMM_OUT_DIR, TRANS_OUT_DIR          (path constants)
############################################################

# ──────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────

function _ts_w_suffix(cond_target::Float64)
    cond_target == 0.0 && return "_diagonalW"
    cond_target == 1.0 && return "_compressedW"
    cond_target == 2.0 && return "_equalW"
    return "_fullW"
end

function _ts_load_bundle_or_error(path, label)
    isfile(path) || error("$label not found: $path\n" *
        "Run the SMM estimation first (code/smm/smm_main.jl).")
    bundle = _load_smm_bundle(path; delete_on_fail = false, label = label)
    isnothing(bundle) && error("$label is unreadable or stale: $path")
    return bundle
end

# ──────────────────────────────────────────────────────────
# Public function
# ──────────────────────────────────────────────────────────

"""
    run_transition_simulation(scenario, w_cond_target;
                              Nx = 200, Np_U = 200, Np_S = 200,
                              T_max = 120.0, N_steps = 240,
                              tol = 1e-4, maxit = 200, damp = 0.3)

Solve the backward-forward transition for a single scenario and save
the TransitionResult bundle to
    <TRANS_OUT_DIR>/transition_<scenario><W_SUFFIX>.jld2

Arguments
  scenario        :fc or :covid
  w_cond_target   weight-matrix mode used by the SMM estimation
                  (0.0, 1.0, 2.0, or > 2.0 → diagonal / compressed / equal / full)

Keyword arguments
  Nx, Np_U, Np_S  grid sizes for the stationary-equilibrium solves
                  (intentionally finer than SMM for single-run consistency)
  T_max           horizon length in model months
  N_steps         number of time steps (Nt = N_steps + 1)
  tol             convergence tolerance on ‖Δθ‖∞
  maxit           maximum backward-forward iterations
  damp            dampening on the tightness update

Returns
  absolute path of the saved .jld2 file
"""
function run_transition_simulation(
    scenario      :: Symbol,
    w_cond_target :: Float64;
    Nx      :: Int  = 200,
    Np_U    :: Int  = 200,
    Np_S    :: Int  = 200,
    T_max   :: Real = 120.0,
    N_steps :: Int  = 240,
    tol     :: Real = 1e-4,
    maxit   :: Int  = 200,
    damp    :: Real = 0.3,
)
    w_suffix = _ts_w_suffix(Float64(w_cond_target))

    println("\n" * "="^65)
    println("  Transition simulation — $scenario   (W = $w_suffix)")
    println("="^65)
    flush(stdout)

    # ── Determine windows ────────────────────────────────
    if scenario == :fc
        base_window   = :base_fc
        crisis_window = :crisis_fc
    elseif scenario == :covid
        base_window   = :base_covid
        crisis_window = :crisis_covid
    else
        error("Unknown scenario: $scenario. Must be :fc or :covid.")
    end

    # ── SMM result file paths ────────────────────────────
    base_jls   = joinpath(SMM_OUT_DIR,
                          "smm_result_$(base_window)$(w_suffix).jls")
    crisis_jls = joinpath(SMM_OUT_DIR,
                          "smm_result_$(crisis_window)$(w_suffix).jls")

    @printf("  Baseline file:  %s\n", base_jls)
    @printf("  Crisis file:    %s\n\n", crisis_jls)
    flush(stdout)

    # ── Load SMM bundles and reconstruct parameters ───────
    println("Loading SMM results..."); flush(stdout)
    base_bundle   = _ts_load_bundle_or_error(base_jls,   "Baseline SMM result")
    crisis_bundle = _ts_load_bundle_or_error(crisis_jls, "Crisis SMM result")

    cp_base,  up_base,  sp_base  = unpack_θ(
        base_bundle.result.theta_opt,   base_bundle.spec)
    cp_crisis, up_crisis, sp_crisis = unpack_θ(
        crisis_bundle.result.theta_opt, crisis_bundle.spec)

    # ── SimParams: inherit tolerances, raise verbosity ────
    sim = base_bundle.sim
    sim_solve = SimParams(
        tol_inner      = sim.tol_inner,
        tol_outer_U    = sim.tol_outer_U,
        tol_outer_S    = sim.tol_outer_S,
        tol_global     = sim.tol_global,
        maxit_inner    = sim.maxit_inner,
        maxit_outer    = sim.maxit_outer,
        maxit_global   = sim.maxit_global,
        conv_streak    = sim.conv_streak,
        use_anderson   = sim.use_anderson,
        anderson_m     = sim.anderson_m,
        anderson_reg   = sim.anderson_reg,
        damp_pstar_U   = sim.damp_pstar_U,
        damp_pstar_S   = sim.damp_pstar_S,
        verbose        = 1,
        verbose_stride = sim.verbose_stride,
    )

    # Grid sizes — intentionally NOT inherited from the SMM bundle.
    # See the original transition_main for the rationale.
    @printf("  Grid sizes: Nx=%d  Np_U=%d  Np_S=%d  " *
            "(overriding spec.run.Nx=%d for single-run consistency)\n",
            Nx, Np_U, Np_S, base_bundle.spec.run.Nx)
    @printf("  Baseline Q  = %.6e  (converged = %s)\n",
            base_bundle.result.loss_opt, base_bundle.result.converged)
    @printf("  Crisis Q    = %.6e  (converged = %s)\n\n",
            crisis_bundle.result.loss_opt, crisis_bundle.result.converged)
    flush(stdout)

    # ── Solve both stationary equilibria ─────────────────
    println("Solving baseline (z₀) stationary equilibrium..."); flush(stdout)
    model_z0, sr_z0 = solve_model(cp_base, up_base, sp_base, sim_solve;
                                   Nx = Nx, Np_U = Np_U, Np_S = Np_S)
    sr_z0.ok || @warn "Baseline model did not converge — " *
                       "transition results may be unreliable."
    eq_z0 = compute_equilibrium_objects(model_z0)
    @printf("  θ_U = %.4f   θ_S = %.4f   ur_total = %.4f\n\n",
            eq_z0.thetaU, eq_z0.thetaS, eq_z0.ur_total)
    flush(stdout)

    println("Solving crisis (z₁) stationary equilibrium..."); flush(stdout)
    model_z1, sr_z1 = solve_model(cp_crisis, up_crisis, sp_crisis,
                                   sim_solve;
                                   Nx = Nx, Np_U = Np_U, Np_S = Np_S)
    sr_z1.ok || @warn "Crisis model did not converge — " *
                       "transition results may be unreliable."
    eq_z1 = compute_equilibrium_objects(model_z1)
    @printf("  θ_U = %.4f   θ_S = %.4f   ur_total = %.4f\n\n",
            eq_z1.thetaU, eq_z1.thetaS, eq_z1.ur_total)
    flush(stdout)

    # ── Transition parameters & solve ────────────────────
    tp = TransitionParams(
        T_max   = T_max,
        N_steps = N_steps,
        tol     = tol,
        maxit   = maxit,
        damp    = damp,
        verbose = true,
    )

    result = solve_transition(model_z0, model_z1, tp; scenario = scenario)

    # ── Summary ──────────────────────────────────────────
    println("="^65)
    println("  Transition Results Summary — $scenario")
    println("="^65)
    @printf("  Converged: %s  (%d iterations, final dist = %.3e)\n",
            result.converged, result.n_iterations, result.final_dist)
    @printf("\n  Tightness:\n")
    @printf("    θ_U:  %.4f → %.4f\n", result.θU[1], result.θU[end])
    @printf("    θ_S:  %.4f → %.4f\n", result.θS[1], result.θS[end])
    @printf("\n  Unemployment rates:\n")
    @printf("    ur_U:     %.4f → %.4f\n", result.ur_U[1], result.ur_U[end])
    @printf("    ur_S:     %.4f → %.4f\n", result.ur_S[1], result.ur_S[end])
    @printf("    ur_total: %.4f → %.4f\n", result.ur_total[1], result.ur_total[end])
    @printf("\n  Job-finding rates:\n")
    @printf("    f_U:  %.4f → %.4f\n", result.fU[1], result.fU[end])
    @printf("    f_S:  %.4f → %.4f\n", result.fS[1], result.fS[end])
    @printf("\n  Composition:\n")
    @printf("    skilled_share:  %.4f → %.4f\n",
            result.skilled_share[1], result.skilled_share[end])
    @printf("    training_share: %.4f → %.4f\n",
            result.training_share[1], result.training_share[end])
    flush(stdout)

    # ── Save ─────────────────────────────────────────────
    mkpath(TRANS_OUT_DIR)
    out_file = joinpath(TRANS_OUT_DIR,
                        "transition_$(scenario)$(w_suffix).jld2")
    @printf("\nSaving transition result → %s\n", out_file); flush(stdout)

    jldsave(out_file;
        result   = result,
        eq_z0    = eq_z0,
        eq_z1    = eq_z1,
        tp       = tp,
        scenario = scenario,
    )

    @printf("Done.\n"); flush(stdout)
    return out_file
end

# ──────────────────────────────────────────────────────────
# Stand-alone safety net.  This file is a library and must be
# invoked from transition_main.jl.  If someone runs it directly,
# fail loudly with an actionable message.
# ──────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    error("transition_simulation.jl is a library and cannot be run " *
          "standalone.  Use transition_main.jl as the entry point.")
end

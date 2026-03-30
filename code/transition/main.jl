"""
    main.jl

Entry point for solving transition dynamics.

This script:
1. Loads required packages and modules
2. Solves the initial regime z₀ (e.g., baseline)
3. Solves the terminal regime z₁ (e.g., crisis)
4. Computes the backward-forward transition path
5. Optionally saves and visualizes results

Usage:
    julia transition/main.jl
"""

# ============================================================================
# Packages
# ============================================================================

using Printf
using Plots
using JLD2

# ============================================================================
# Load solver modules
# ============================================================================

# Adjust paths as needed; assumes structure:
#   code/
#     solver/
#       params.jl, grids.jl, solver_stationary.jl, ...
#     transition/
#       main.jl, transition_params.jl, transition_solver.jl

solver_dir = joinpath(@__DIR__, "..", "solver")
include(joinpath(solver_dir, "params.jl"))
include(joinpath(solver_dir, "grids.jl"))
include(joinpath(solver_dir, "solver_stationary.jl"))

# Load transition modules
include(joinpath(@__DIR__, "transition_params.jl"))
include(joinpath(@__DIR__, "transition_solver.jl"))

# ============================================================================
# Main execution
# ============================================================================

function main()
    @printf("\n%s\n", "="^70)
    @printf("Transition Dynamics: Backward-Forward Algorithm\n")
    @printf("%s\n\n", "="^70)

    # ========================================================================
    # Step 1: Setup parameter regimes
    # ========================================================================

    @printf("Setting up parameter regimes...\n")

    # Regime z₀: Baseline (example parameters)
    regime_z0 = RegimeParams(
        z=1.0,          # Productivity / regime indicator
        sep_U=0.05,     # Unskilled separation rate
        sep_S=0.03,     # Skilled separation rate
        A=1.0           # Matching efficiency
    )

    # Regime z₁: Crisis / different regime (example: lower productivity, higher sep)
    regime_z1 = RegimeParams(
        z=0.9,          # Lower productivity
        sep_U=0.08,     # Higher unskilled separation
        sep_S=0.05,     # Higher skilled separation
        A=0.95          # Lower matching efficiency
    )

    # Common parameters (shared across regimes)
    common_params = CommonParams(
        r=0.01,         # Interest rate
        k_U=0.5,        # Unskilled recruitment cost
        k_S=1.0,        # Skilled recruitment cost
        mu_U=0.5,       # Unskilled matching elasticity parameter
        mu_S=0.5,       # Skilled matching elasticity parameter
        eta_U=0.5,      # Unskilled matching curvature
        eta_S=0.5,      # Skilled matching curvature
        f_U=0.03,       # Unskilled job finding rate
        f_S=0.04,       # Skilled job finding rate
        nu=0.02,        # Flow into outside sector
        phi=0.1,        # Training-to-skilled flow rate
        w_U=1.0,        # Unskilled wage floor
        w_S=2.0         # Skilled wage floor
    )

    # Unskilled and Skilled parameters (fixed across regimes)
    unskilled_params = UnskilledParams(
        rho_U=0.8,      # Preference parameter
        gamma_U=0.5,    # Elasticity parameter
        beta_U=0.6      # Bargaining power
    )

    skilled_params = SkilledParams(
        rho_S=0.8,
        gamma_S=0.5,
        beta_S=0.6,
        xi_S=0.1,       # OJS flow rate
        delta_S_end=0.02 # Separation from skilled employment
    )

    # Grids (fixed across regimes)
    grids = CommonGrids(
        x=build_gl_grid(21, 0.1, 1.0),  # Productivity grid
        p_S=build_gl_grid(15, 0.0, 1.0)  # Skilled match quality grid
    )

    @printf("  Regime z₀: z=%.2f, sep_U=%.3f, sep_S=%.3f\n",
            regime_z0.z, regime_z0.sep_U, regime_z0.sep_S)
    @printf("  Regime z₁: z=%.2f, sep_U=%.3f, sep_S=%.3f\n",
            regime_z1.z, regime_z1.sep_U, regime_z1.sep_S)
    @printf("  Grid: Nx=%d, Np_S=%d\n\n", length(grids.x), length(grids.p_S))

    # ========================================================================
    # Step 2: Solve stationary models
    # ========================================================================

    @printf("Solving stationary model under regime z₀...\n")
    model_z0 = Model(
        regime_z0, common_params, unskilled_params, skilled_params, grids
    )
    solve_model!(model_z0, verbose=true)

    @printf("\nSolving stationary model under regime z₁...\n")
    model_z1 = Model(
        regime_z1, common_params, unskilled_params, skilled_params, grids
    )
    solve_model!(model_z1, verbose=true)

    # ========================================================================
    # Step 3: Setup transition parameters
    # ========================================================================

    @printf("\n%s\n", "="^70)
    @printf("Transition Parameters\n")
    @printf("%s\n", "="^70)

    tp = TransitionParams(
        T_max=10.0,      # 10 calendar time periods
        N_steps=100,     # 101 time points (including t=0 and t=T_max)
        tol=1e-4,        # Convergence tolerance
        maxit=50,        # Maximum iterations
        verbose=true,
        damp=0.5         # Dampening factor
    )

    # ========================================================================
    # Step 4: Solve transition dynamics
    # ========================================================================

    @printf("\n")
    path = solve_transition(model_z0, model_z1, tp)

    # ========================================================================
    # Step 5: Post-processing and output
    # ========================================================================

    @printf("\nTransition Path Results:\n")
    @printf("  Initial θ_U: %.4f → Final θ_U: %.4f\n",
            path.theta_U[1], path.theta_U[end])
    @printf("  Initial θ_S: %.4f → Final θ_S: %.4f\n",
            path.theta_S[1], path.theta_S[end])

    # Summary statistics
    @printf("\nDistributions (t=0 to t=T_max):\n")
    @printf("  ∫ u_U:  %.4f → %.4f\n",
            sum(path.u_U[:, 1] .* grids.weights_x),
            sum(path.u_U[:, end] .* grids.weights_x))
    @printf("  ∫ t:    %.4f → %.4f\n",
            sum(path.t_dens[:, 1] .* grids.weights_x),
            sum(path.t_dens[:, end] .* grids.weights_x))
    @printf("  ∫ u_S:  %.4f → %.4f\n",
            sum(path.u_S[:, 1] .* grids.weights_x),
            sum(path.u_S[:, end] .* grids.weights_x))
    @printf("  m_S:    %.4f → %.4f\n", path.m_S[1], path.m_S[end])

    # ========================================================================
    # Step 6: Save results (optional)
    # ========================================================================

    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)

    # Save to JLD2 (binary format)
    jld_file = joinpath(output_dir, "transition_path.jld2")
    @printf("\nSaving transition path to: %s\n", jld_file)
    save(jld_file,
         "path", path,
         "model_z0", model_z0,
         "model_z1", model_z1,
         "tp", tp)

    # ========================================================================
    # Step 7: Visualization (optional)
    # ========================================================================

    @printf("\nGenerating plots...\n")

    # Plot tightness paths
    p1 = plot(path.tgrid, path.theta_U, label="θ_U", xlabel="Time", ylabel="Tightness", legend=:topleft)
    plot!(p1, path.tgrid, path.theta_S, label="θ_S")
    savefig(p1, joinpath(output_dir, "tightness_paths.png"))

    # Plot distribution paths (selected x-grid points)
    x_indices = [1, div(length(grids.x), 2), length(grids.x)]
    p2 = plot(path.tgrid, path.u_U[x_indices[1], :], label="u_U (low x)", xlabel="Time", ylabel="Density")
    plot!(p2, path.tgrid, path.u_U[x_indices[2], :], label="u_U (mid x)")
    plot!(p2, path.tgrid, path.u_U[x_indices[3], :], label="u_U (high x)")
    savefig(p2, joinpath(output_dir, "unskilled_unemployment.png"))

    p3 = plot(path.tgrid, path.u_S[x_indices[1], :], label="u_S (low x)", xlabel="Time", ylabel="Density")
    plot!(p3, path.tgrid, path.u_S[x_indices[2], :], label="u_S (mid x)")
    plot!(p3, path.tgrid, path.u_S[x_indices[3], :], label="u_S (high x)")
    savefig(p3, joinpath(output_dir, "skilled_unemployment.png"))

    # Plot training density
    p4 = plot(path.tgrid, path.t_dens[x_indices[1], :], label="t (low x)", xlabel="Time", ylabel="Density")
    plot!(p4, path.tgrid, path.t_dens[x_indices[2], :], label="t (mid x)")
    plot!(p4, path.tgrid, path.t_dens[x_indices[3], :], label="t (high x)")
    savefig(p4, joinpath(output_dir, "training_density.png"))

    # Plot skilled segment mass
    p5 = plot(path.tgrid, path.m_S, label="m_S", xlabel="Time", ylabel="Mass", legend=:topleft)
    savefig(p5, joinpath(output_dir, "skilled_mass.png"))

    @printf("Plots saved to: %s\n", output_dir)

    @printf("\n%s\n", "="^70)
    @printf("Transition dynamics solved successfully!\n")
    @printf("%s\n", "="^70)

    return path, model_z0, model_z1
end

# ============================================================================
# Run main if executed as script
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    path, model_z0, model_z1 = main()
end

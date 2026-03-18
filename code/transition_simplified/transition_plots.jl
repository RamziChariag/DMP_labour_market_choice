############################################################
# transition_plots.jl — Plotting functions for transition dynamics
#
# Uses Plots.jl (GR backend by default).
#
# Key function:
#   make_transition_plots(trans, obj_base, obj_crisis;
#                         crisis_name, output_dir)
#
# Generates the following plots:
#   1. Unemployment rates (total, unskilled, skilled)
#   2. Skilled share and training share
#   3. Market tightness (θ_U, θ_S)
#   4. Job-finding rates (f_U, f_S)
#   5. Segment masses (M_U, M_S)
#   6. Summary dashboard (all panels combined)
############################################################


"""
    make_transition_plots(trans, obj_base, obj_crisis;
                          crisis_name, output_dir)

Generate and save all transition-dynamics plots.

Arguments
─────────
  trans       : TransitionResult from simulate_transition()
  obj_base    : baseline equilibrium objects
  obj_crisis  : crisis equilibrium objects
  crisis_name : display name for plot titles
  output_dir  : directory to save PNG files
"""
function make_transition_plots(
    trans,
    obj_base,
    obj_crisis;
    crisis_name :: String = "Crisis",
    output_dir  :: String = ".",
)
    mkpath(output_dir)

    # Convert time to quarters for readability
    t_quarters = trans.times ./ 3.0

    # Common plot settings
    default(;
        linewidth  = 2,
        legend     = :topright,
        grid       = true,
        gridalpha  = 0.3,
        fontfamily = "Computer Modern",
        size       = (700, 450),
    )

    # ── Plot 1: Unemployment rates ───────────────────────────
    p1 = plot(
        t_quarters, 100 .* trans.ur_total;
        label  = "Total UR",
        color  = :black,
        xlabel = "Quarters after shock",
        ylabel = "Unemployment rate (%)",
        title  = "Unemployment rates — $crisis_name",
    )
    plot!(p1, t_quarters, 100 .* trans.ur_U;
          label = "Unskilled UR", color = :steelblue, linestyle = :dash)
    plot!(p1, t_quarters, 100 .* trans.ur_S;
          label = "Skilled UR", color = :firebrick, linestyle = :dot)

    # Horizontal lines for steady states
    hline!(p1, [100 * obj_base.ur_total];
           label = "Baseline SS (total)", color = :gray, linestyle = :dashdot, alpha = 0.5)
    hline!(p1, [100 * obj_crisis.ur_total];
           label = "Crisis SS (total)", color = :gray, linestyle = :dashdotdot, alpha = 0.5)

    savefig(p1, joinpath(output_dir, "unemployment_rates.png"))


    # ── Plot 2: Skilled share and training share ─────────────
    p2 = plot(
        t_quarters, 100 .* trans.skilled_share;
        label  = "Skilled share",
        color  = :darkgreen,
        xlabel = "Quarters after shock",
        ylabel = "Share of population (%)",
        title  = "Population composition — $crisis_name",
    )
    plot!(p2, t_quarters, 100 .* trans.training_share;
          label = "Training share", color = :darkorange, linestyle = :dash)

    hline!(p2, [100 * obj_base.agg_mS / max(obj_base.total_pop, 1e-14)];
           label = "Baseline skilled", color = :darkgreen, alpha = 0.3, linestyle = :dashdot)
    hline!(p2, [100 * obj_crisis.agg_mS / max(obj_crisis.total_pop, 1e-14)];
           label = "Crisis skilled", color = :darkgreen, alpha = 0.3, linestyle = :dashdotdot)

    savefig(p2, joinpath(output_dir, "population_composition.png"))


    # ── Plot 3: Market tightness ─────────────────────────────
    p3 = plot(
        t_quarters, trans.theta_U;
        label  = "θ_U (unskilled)",
        color  = :steelblue,
        xlabel = "Quarters after shock",
        ylabel = "Market tightness (θ)",
        title  = "Market tightness — $crisis_name",
    )
    plot!(p3, t_quarters, trans.theta_S;
          label = "θ_S (skilled)", color = :firebrick, linestyle = :dash)

    hline!(p3, [obj_base.thetaU];
           label = "Baseline θ_U", color = :steelblue, alpha = 0.3, linestyle = :dashdot)
    hline!(p3, [obj_crisis.thetaU];
           label = "Crisis θ_U", color = :steelblue, alpha = 0.3, linestyle = :dashdotdot)
    hline!(p3, [obj_base.thetaS];
           label = "Baseline θ_S", color = :firebrick, alpha = 0.3, linestyle = :dashdot)
    hline!(p3, [obj_crisis.thetaS];
           label = "Crisis θ_S", color = :firebrick, alpha = 0.3, linestyle = :dashdotdot)

    savefig(p3, joinpath(output_dir, "market_tightness.png"))


    # ── Plot 4: Job-finding rates ────────────────────────────
    p4 = plot(
        t_quarters, trans.f_U;
        label  = "f_U (unskilled JFR)",
        color  = :steelblue,
        xlabel = "Quarters after shock",
        ylabel = "Job-finding rate",
        title  = "Job-finding rates — $crisis_name",
    )
    plot!(p4, t_quarters, trans.f_S;
          label = "f_S (skilled JFR)", color = :firebrick, linestyle = :dash)

    hline!(p4, [obj_base.f_U];
           label = "Baseline f_U", color = :steelblue, alpha = 0.3, linestyle = :dashdot)
    hline!(p4, [obj_crisis.f_U];
           label = "Crisis f_U", color = :steelblue, alpha = 0.3, linestyle = :dashdotdot)

    savefig(p4, joinpath(output_dir, "job_finding_rates.png"))


    # ── Plot 5: Segment masses ───────────────────────────────
    p5 = plot(
        t_quarters, trans.agg_mU;
        label  = "M_U (unskilled segment)",
        color  = :steelblue,
        xlabel = "Quarters after shock",
        ylabel = "Segment mass",
        title  = "Segment masses — $crisis_name",
    )
    plot!(p5, t_quarters, trans.agg_mS;
          label = "M_S (skilled segment)", color = :firebrick, linestyle = :dash)
    plot!(p5, t_quarters, trans.agg_t;
          label = "Training stock", color = :darkorange, linestyle = :dot)

    savefig(p5, joinpath(output_dir, "segment_masses.png"))


    # ── Plot 6: Summary dashboard (2×3) ──────────────────────
    p_dash = plot(
        p1, p2, p3, p4, p5;
        layout = @layout([a b; c d; e _]),
        size   = (1400, 1200),
        title  = "",
    )
    savefig(p_dash, joinpath(output_dir, "transition_dashboard.png"))


    # ── Save time series to CSV ──────────────────────────────
    csv_path = joinpath(output_dir, "transition_paths.csv")
    open(csv_path, "w") do io
        println(io, "time_months,time_quarters,ur_total,ur_U,ur_S," *
                    "skilled_share,training_share,theta_U,theta_S," *
                    "f_U,f_S,agg_mU,agg_mS,agg_uU,agg_uS,agg_t,agg_eU,agg_eS")
        for i in 1:length(trans.times)
            @printf(io, "%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    trans.times[i], trans.times[i]/3.0,
                    trans.ur_total[i], trans.ur_U[i], trans.ur_S[i],
                    trans.skilled_share[i], trans.training_share[i],
                    trans.theta_U[i], trans.theta_S[i],
                    trans.f_U[i], trans.f_S[i],
                    trans.agg_mU[i], trans.agg_mS[i],
                    trans.agg_uU[i], trans.agg_uS[i],
                    trans.agg_t[i], trans.agg_eU[i], trans.agg_eS[i])
        end
    end
    @printf("  Time series saved to: %s\n", csv_path)

    # ── Print summary table ──────────────────────────────────
    println()
    @printf("╔═══════════════════════════════════════════════════════════╗\n")
    @printf("║  Transition Summary: %-36s  ║\n", crisis_name)
    @printf("╠═══════════════════════════════════════════════════════════╣\n")
    @printf("║  %-25s  %10s  %10s  %10s   ║\n", "", "Baseline", "t=0+", "Crisis SS")
    @printf("╠═══════════════════════════════════════════════════════════╣\n")
    @printf("║  %-25s  %9.2f%%  %9.2f%%  %9.2f%%   ║\n",
            "Total UR",
            100*obj_base.ur_total, 100*trans.ur_total[1], 100*obj_crisis.ur_total)
    @printf("║  %-25s  %9.2f%%  %9.2f%%  %9.2f%%   ║\n",
            "Unskilled UR",
            100*obj_base.ur_U, 100*trans.ur_U[1], 100*obj_crisis.ur_U)
    @printf("║  %-25s  %9.2f%%  %9.2f%%  %9.2f%%   ║\n",
            "Skilled UR",
            100*obj_base.ur_S, 100*trans.ur_S[1], 100*obj_crisis.ur_S)
    @printf("║  %-25s  %9.2f%%  %9.2f%%  %9.2f%%   ║\n",
            "Skilled share",
            100*obj_base.agg_mS/max(obj_base.total_pop,1e-14),
            100*trans.skilled_share[1],
            100*obj_crisis.agg_mS/max(obj_crisis.total_pop,1e-14))
    @printf("║  %-25s  %10.4f  %10.4f  %10.4f   ║\n",
            "θ_U",
            obj_base.thetaU, trans.theta_U[1], obj_crisis.thetaU)
    @printf("║  %-25s  %10.4f  %10.4f  %10.4f   ║\n",
            "θ_S",
            obj_base.thetaS, trans.theta_S[1], obj_crisis.thetaS)
    @printf("╚═══════════════════════════════════════════════════════════╝\n\n")

    # Print half-life estimates
    _print_half_lives(trans, obj_base, obj_crisis)

    flush(stdout)
end


"""
    _print_half_lives(trans, obj_base, obj_crisis)

Estimate and print the half-life of adjustment for key quantities.
Half-life = first time the quantity has closed 50% of the gap
between baseline and crisis steady states.
"""
function _print_half_lives(trans, obj_base, obj_crisis)
    function _half_life(path, ss_base, ss_crisis)
        gap = ss_crisis - ss_base
        abs(gap) < 1e-12 && return NaN
        target = ss_base + 0.5 * gap
        for i in 2:length(path)
            if gap > 0
                path[i] >= target && return trans.times[i]
            else
                path[i] <= target && return trans.times[i]
            end
        end
        return NaN  # didn't reach half-life within horizon
    end

    hl_ur = _half_life(trans.ur_total, obj_base.ur_total, obj_crisis.ur_total)
    hl_sk = _half_life(trans.skilled_share,
                       obj_base.agg_mS / max(obj_base.total_pop, 1e-14),
                       obj_crisis.agg_mS / max(obj_crisis.total_pop, 1e-14))
    hl_tU = _half_life(trans.theta_U, obj_base.thetaU, obj_crisis.thetaU)

    @printf("  Half-lives (months):  UR_total=%.1f  Skilled_share=%.1f  θ_U=%.1f\n",
            hl_ur, hl_sk, hl_tU)
    flush(stdout)
end

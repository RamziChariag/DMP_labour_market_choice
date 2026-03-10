############################################################
# plots.jl — All equilibrium plots
#
# make_all_plots(obj; output_dir)
#   Generates and saves every figure to output_dir.
#   obj — NamedTuple from compute_equilibrium_objects()
#
# NOTE: `using Plots`, `using LaTeXStrings`, and
# `using Interpolations` are loaded in main.jl before
# this file's functions are called.
############################################################

# ── Shared theme ────────────────────────────────────────────────────────────
function _set_theme!()
    gr()
    theme(:default)
    default(
        fontfamily    = "Computer Modern",
        framestyle    = :box,
        titlefontsize = 11,
        guidefontsize = 10,
        tickfontsize  = 8,
        legendfontsize = 8,
        linewidth     = 1.8,
        grid          = true,
        gridalpha     = 0.10,
    )
end

const _C1 = :steelblue
const _C2 = :firebrick
const _C3 = :seagreen
const _C4 = :darkorange
const _C5 = :mediumpurple


# ============================================================
# Individual figure constructors
# ============================================================

function fig_densities(obj)
    xg = obj.xg

    p1a = plot(xg, obj.uU,    label=L"u_U(x)",           color=_C1)
    plot!(p1a, xg, obj.tU,    label=L"t(x)",             color=_C2)
    plot!(p1a, xg, obj.eU_vec,label=L"e_U(x)",           color=_C3)
    plot!(p1a, xg, obj.mU_x,  label=L"m_U(x)",           color=:black, ls=:dash)
    title!(p1a, "Unskilled segment densities")
    xlabel!(p1a, L"x");  ylabel!(p1a, "Density")

    p1b = plot(xg, obj.uS,    label=L"u_S(x)",           color=_C1)
    plot!(p1b, xg, obj.eS_tot,label=L"e_S^{\rm tot}(x)", color=_C3)
    plot!(p1b, xg, obj.mS_vec,label=L"m_S(x)",           color=:black, ls=:dash)
    title!(p1b, "Skilled segment densities")
    xlabel!(p1b, L"x");  ylabel!(p1b, "Density")

    plot(p1a, p1b, layout=(1,2), size=(900,380), margin=5Plots.mm)
end


function fig_unskilled_values(obj)
    xg = obj.xg
    x_bar_idx = findfirst(obj.tauT .> 0.5)
    x_bar = isnothing(x_bar_idx) ? NaN : xg[x_bar_idx]

    p = plot(xg, obj.Usearch, label=L"U_U^{\rm search}(x)", color=_C1)
    plot!(p, xg, obj.net_T,   label=L"-c(x) + T(x)",        color=_C2)
    plot!(p, xg, obj.UU,      label=L"U_U(x)=\max",         color=:black, ls=:dash, lw=2.2)
    if !isnan(x_bar)
        annotate!(p, xg[end]*0.95, minimum(obj.Usearch)*1.02,
                  text(L"\bar{x} \approx %$(round(x_bar, digits=3))", :right, 8, :darkgray))
    end
    title!(p, "Unskilled unemployment values")
    xlabel!(p, L"x");  ylabel!(p, "Value")
    p
end


function fig_employment_heatmaps(obj)
    p3a = heatmap(obj.pgU, obj.xg, obj.eU_surface,
                  xlabel=L"p", ylabel=L"x",
                  title=L"Unskilled employment $e_U(x,p)$", color=:plasma)
    contour!(p3a, obj.pgU, obj.xg, obj.eU_surface,
             color=:white, alpha=0.4, lw=0.8, levels=10)
    plot!(p3a, obj.pstar_U, obj.xg,
          label=L"p^*(x)", color=:red, lw=2, ls=:dash)

    p3b = heatmap(obj.pg, obj.xg, obj.eS_mat,
                  xlabel=L"p", ylabel=L"x",
                  title=L"Skilled employment $e_S(x,p)$", color=:viridis)
    contour!(p3b, obj.pg, obj.xg, obj.eS_mat,
             color=:white, alpha=0.4, lw=0.8, levels=10)
    plot!(p3b, obj.pstar_S, obj.xg,
          label=L"p_S^*(x)", color=:red, lw=2, ls=:dash)
    plot!(p3b, obj.poj, obj.xg,
          label=L"p^{\rm oj}(x)", color=:white, lw=2, ls=:dot)

    plot(p3a, p3b, layout=(1,2), size=(1100,440), margin=5Plots.mm)
end


function fig_total_employment(obj)
    p = heatmap(obj.pg, obj.xg, obj.e_total_surface,
                xlabel=L"p", ylabel=L"x",
                title=L"Total employment $e_U(x,p)+e_S(x,p)$",
                color=:viridis, key=false)
    contour!(p, obj.pg, obj.xg, obj.e_total_surface,
             color=:white, alpha=0.4, lw=0.8, levels=10)
    p
end


function fig_training_policy(obj)
    p = plot(obj.xg, obj.tauT, label=L"\tau_T(x)", color=_C2, lw=2)
    title!(p, L"Training policy $\tau_T(x)$")
    xlabel!(p, L"x");  ylabel!(p, "Train indicator")
    ylims!(p, -0.05, 1.15)
    p
end


function fig_unskilled_frontier(obj)
    p5a = plot(obj.xg, obj.JU_frontier,
               label=L"J_U(x,1)", color=_C1)
    hline!(p5a, [0.0], ls=:dot, color=:gray, label="")
    title!(p5a, L"Frontier firm value $J_U(x,p{=}1)$")
    xlabel!(p5a, L"x");  ylabel!(p5a, "Value")

    p5b = plot(obj.xg, obj.pstar_U,
               label=L"p^*(x)", color=_C4)
    title!(p5b, L"Unskilled reservation rule $p^*(x)$")
    xlabel!(p5b, L"x");  ylabel!(p5b, L"p^*")
    ylims!(p5b, 0, 1)

    plot(p5a, p5b, layout=(1,2), size=(900,380), margin=5Plots.mm)
end


function fig_unskilled_value_surfaces(obj)
    p6a = heatmap(obj.pgU, obj.xg, obj.JU_surface,
                  xlabel=L"p", ylabel=L"x",
                  title=L"Firm value $J_U(x,p)$", color=:plasma)
    contour!(p6a, obj.pgU, obj.xg, obj.JU_surface,
             color=:white, alpha=0.4, lw=0.8, levels=10)
    plot!(p6a, obj.pstar_U, obj.xg,
          label=L"p^*(x)\ [J_U=0]", color=:red, lw=2, ls=:dash)

    p6b = heatmap(obj.pgU, obj.xg, obj.EU_surface,
                  xlabel=L"p", ylabel=L"x",
                  title=L"Worker value $E_U(x,p)$", color=:plasma)
    contour!(p6b, obj.pgU, obj.xg, obj.EU_surface,
             color=:white, alpha=0.4, lw=0.8, levels=10)
    plot!(p6b, obj.pstar_U, obj.xg,
          label=L"p_U^*(x)", color=:red, lw=2, ls=:dash)

    plot(p6a, p6b, layout=(1,2), size=(1000,420), margin=5Plots.mm)
end


function fig_skilled_cutoffs(obj)
    p = plot(obj.xg, obj.pstar_S,
             label=L"p_S^*(x)", color=_C1)
    plot!(p, obj.xg, obj.poj,
          label=L"p^{\rm oj}(x)", color=_C2)
    title!(p, "Skilled cutoffs by type")
    xlabel!(p, L"x");  ylabel!(p, L"p")
    ylims!(p, 0, 1)
    p
end


function fig_skilled_worker_values(obj)
    p8a = heatmap(obj.pg, obj.xg, obj.E0_surface,
                  xlabel=L"p", ylabel=L"x",
                  title=L"E_S^0(x,p)", color=:matter)
    contour!(p8a, obj.pg, obj.xg, obj.E0_surface,
             color=:white, alpha=0.4, lw=0.8, levels=10)

    p8b = heatmap(obj.pg, obj.xg, obj.E1_surface,
                  xlabel=L"p", ylabel=L"x",
                  title=L"E_S^1(x,p)", color=:viridis)
    contour!(p8b, obj.pg, obj.xg, obj.E1_surface,
             color=:white, alpha=0.4, lw=0.8, levels=10)

    plot(p8a, p8b, layout=(1,2), size=(1000,400), margin=5Plots.mm)
end


function fig_skilled_firm_values(obj)
    p9a = heatmap(obj.pg, obj.xg, obj.J0_surface,
                  xlabel=L"p", ylabel=L"x",
                  title=L"J_S^0(x,p)", color=:matter)
    contour!(p9a, obj.pg, obj.xg, obj.J0_surface,
             color=:white, alpha=0.4, lw=0.8, levels=10)

    p9b = heatmap(obj.pg, obj.xg, obj.J1_surface,
                  xlabel=L"p", ylabel=L"x",
                  title=L"J_S^1(x,p)", color=:viridis)
    contour!(p9b, obj.pg, obj.xg, obj.J1_surface,
             color=:white, alpha=0.4, lw=0.8, levels=10)

    plot(p9a, p9b, layout=(1,2), size=(1000,400), margin=5Plots.mm)
end


function fig_surplus_heatmaps(obj)
    p10a = heatmap(obj.pgU, obj.xg, obj.SU_surface,
                   xlabel=L"p", ylabel=L"x",
                   title=L"Unskilled surplus $S_U(x,p)$", color=:plasma)
    contour!(p10a, obj.pgU, obj.xg, obj.SU_surface,
             color=:white, alpha=0.4, lw=0.8, levels=10)
    plot!(p10a, obj.pstar_U, obj.xg,
          label=L"p_U^*(x)", color=:red, lw=2, ls=:dash)

    p10b = heatmap(obj.pg, obj.xg, obj.Smax_surface,
                   xlabel=L"p", ylabel=L"x",
                   title=L"Skilled surplus $\max(S^0,S^1)$", color=:viridis)
    contour!(p10b, obj.pg, obj.xg, obj.Smax_surface,
             color=:white, alpha=0.4, lw=0.8, levels=10)
    plot!(p10b, obj.pstar_S, obj.xg,
          label=L"p_S^*(x)", color=:red, lw=2, ls=:dash)
    plot!(p10b, obj.poj, obj.xg,
          label=L"p^{\rm oj}(x)", color=:white, lw=2, ls=:dot)

    plot(p10a, p10b, layout=(1,2), size=(1100,440), margin=5Plots.mm)
end


function fig_unemployment_values(obj)
    p = plot(obj.xg, obj.US, label=L"U_S(x)", color=_C3, lw=2)
    plot!(p, obj.xg, obj.UU, label=L"U_U(x)", color=_C1, lw=2)
    title!(p, "Unemployment values")
    xlabel!(p, L"x");  ylabel!(p, "Value")
    p
end


function fig_unskilled_wage(obj)
    p = heatmap(obj.pgU, obj.xg, obj.wU_surface,
                xlabel=L"p", ylabel=L"x",
                title=L"Unskilled wage $w_U(x,p)$", color=:plasma)
    contour!(p, obj.pgU, obj.xg, obj.wU_surface,
             color=:white, alpha=0.45, lw=0.8, levels=12)
    plot!(p, obj.pstar_U, obj.xg,
          label=L"p_U^*(x)", color=:red, lw=2, ls=:dash)
    p
end


function fig_skilled_wages(obj)
    pW2a = heatmap(obj.pg, obj.xg, obj.wS0_surface,
                   xlabel=L"p", ylabel=L"x",
                   title=L"Skilled wage, no OJS: $w_S^0(x,p)$", color=:matter)
    contour!(pW2a, obj.pg, obj.xg, obj.wS0_surface,
             color=:white, alpha=0.45, lw=0.8, levels=12)
    plot!(pW2a, obj.pstar_S, obj.xg,
          label=L"p_S^*(x)", color=:red, lw=2, ls=:dash)
    plot!(pW2a, obj.poj, obj.xg,
          label=L"p^{\rm oj}(x)", color=:white, lw=2, ls=:dot)

    pW2b = heatmap(obj.pg, obj.xg, obj.wS1_surface,
                   xlabel=L"p", ylabel=L"x",
                   title=L"Skilled wage, OJS: $w_S^1(x,p)$", color=:viridis)
    contour!(pW2b, obj.pg, obj.xg, obj.wS1_surface,
             color=:white, alpha=0.45, lw=0.8, levels=12)
    plot!(pW2b, obj.pstar_S, obj.xg,
          label=L"p_S^*(x)", color=:red, lw=2, ls=:dash)
    plot!(pW2b, obj.poj, obj.xg,
          label=L"p^{\rm oj}(x)", color=:white, lw=2, ls=:dot)

    plot(pW2a, pW2b, layout=(1,2), size=(1100,440), margin=5Plots.mm)
end


function fig_skill_premium(obj)
    # Realised skilled wage: w_S^1 where p < poj(x), w_S^0 elsewhere
    wS_actual = fill(NaN, obj.Nx, obj.NpS)
    for ix in 1:obj.Nx
        poj_ix = clamp01(obj.poj[ix])
        for jp in 1:obj.NpS
            if obj.pg[jp] < poj_ix
                wS_actual[ix, jp] = obj.wS1_surface[ix, jp]
            else
                wS_actual[ix, jp] = obj.wS0_surface[ix, jp]
            end
        end
    end

    # Interpolate w_U onto the skilled p-grid for each x
    wU_on_pg = fill(NaN, obj.Nx, obj.NpS)
    for ix in 1:obj.Nx
        itp = linear_interpolation(obj.pgU, obj.wU_surface[ix, :],
                                   extrapolation_bc=NaN)
        wU_on_pg[ix, :] = itp.(obj.pg)
    end

    ΔwSU_surface = wS_actual .- wU_on_pg

    p = heatmap(obj.pg, obj.xg, ΔwSU_surface,
                xlabel=L"p", ylabel=L"x",
                title=L"Skill premium $w_S(x,p) - w_U(x,p)$", color=:cividis)
    contour!(p, obj.pg, obj.xg, ΔwSU_surface,
             color=:black, alpha=0.35, lw=0.7, levels=10, colorbar=true)
    plot!(p, obj.pstar_S, obj.xg,
          label=L"p_S^*(x)", color=:red, lw=2, ls=:dash)
    plot!(p, obj.poj, obj.xg,
          label=L"p^{\rm oj}(x)", color=:black, lw=2, ls=:dot)
    p
end


function fig_wage_densities(obj)
    pW4a = plot(obj.wmid, obj.dens_U, label="Unskilled",
                color=:steelblue, lw=2, fill=(0, 0.15, :steelblue))
    plot!(pW4a, obj.wmid, obj.dens_S, label="Skilled (all)",
          color=:firebrick, lw=2, fill=(0, 0.15, :firebrick))
    title!(pW4a, "Wage density: unskilled vs skilled")
    xlabel!(pW4a, "Wage");  ylabel!(pW4a, "Density")

    pW4b = plot(obj.wmid, obj.dens_S0, label=L"Skilled,\ s{=}0\ (no\ OJS)",
                color=:seagreen, lw=2, fill=(0, 0.15, :seagreen))
    plot!(pW4b, obj.wmid, obj.dens_S1, label=L"Skilled,\ s{=}1\ (OJS)",
          color=:darkorange, lw=2, fill=(0, 0.15, :darkorange))
    title!(pW4b, "Wage density: skilled by OJS status")
    xlabel!(pW4b, "Wage");  ylabel!(pW4b, "Density")

    plot(pW4a, pW4b, layout=(1,2), size=(1100,400), margin=5Plots.mm)
end


function fig_wage_densities_pooled(obj)
    p = plot(obj.wmid, obj.dens_U,  label="Unskilled",       color=:steelblue,  lw=2)
    plot!(p, obj.wmid, obj.dens_S0, label=L"Skilled\ s{=}0", color=:seagreen,   lw=2)
    plot!(p, obj.wmid, obj.dens_S1, label=L"Skilled\ s{=}1", color=:darkorange, lw=2, ls=:dash)
    plot!(p, obj.wmid, obj.dens_S,  label="Skilled (pooled)", color=:firebrick,  lw=2.2, ls=:dot)
    title!(p, "Wage densities: all employment types")
    xlabel!(p, "Wage");  ylabel!(p, "Density")
    plot(p, size=(820,380), margin=5Plots.mm)
end


# ============================================================
# Master function — make and save all figures
# ============================================================

"""
    make_all_plots(obj; output_dir = "output/plots")

Generate all equilibrium figures from `obj` (output of
`compute_equilibrium_objects`) and save them as PNG files
into `output_dir`.
"""
function make_all_plots(obj; output_dir::String = "output/plots")
    mkpath(output_dir)
    _set_theme!()

    figures = [
        ("fig01_densities",           fig_densities(obj)),
        ("fig02_unskilled_values",     fig_unskilled_values(obj)),
        ("fig03_employment_heatmaps",  fig_employment_heatmaps(obj)),
        ("fig03b_total_employment",    fig_total_employment(obj)),
        ("fig04_training_policy",      fig_training_policy(obj)),
        ("fig05_unskilled_frontier",   fig_unskilled_frontier(obj)),
        ("fig06_unskilled_surfaces",   fig_unskilled_value_surfaces(obj)),
        ("fig07_skilled_cutoffs",      fig_skilled_cutoffs(obj)),
        ("fig08_skilled_worker_vals",  fig_skilled_worker_values(obj)),
        ("fig09_skilled_firm_vals",    fig_skilled_firm_values(obj)),
        ("fig10_surplus_heatmaps",     fig_surplus_heatmaps(obj)),
        ("fig11_unemployment_values",  fig_unemployment_values(obj)),
        ("figW1_unskilled_wage",       fig_unskilled_wage(obj)),
        ("figW2_skilled_wages",        fig_skilled_wages(obj)),
        ("figW3_skill_premium",        fig_skill_premium(obj)),
        ("figW4_wage_densities",       fig_wage_densities(obj)),
        ("figW4b_wage_densities_pool", fig_wage_densities_pooled(obj)),
    ]

    for (name, fig) in figures
        path = joinpath(output_dir, name * ".png")
        savefig(fig, path)
        println("  saved: $path")
    end

    println("\nAll figures saved to: $output_dir")
    return nothing
end

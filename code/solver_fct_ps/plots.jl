############################################################
# plots.jl — All equilibrium plots (revised)
#
# make_all_plots(obj; output_dir)
#   Generates and saves every figure to output_dir.
#   obj — NamedTuple from compute_equilibrium_objects()
#
# NOTE: `using Plots`, `using LaTeXStrings`, and
# `using Interpolations` are loaded in main.jl before
# this file's functions are called.
############################################################

# ── Shared theme ────────────────────────────────────────────
function _set_theme!()
    gr()
    theme(:default)
    default(
        fontfamily          = "Computer Modern",
        framestyle          = :box,
        titlefontsize       = 11,
        guidefontsize       = 10,
        tickfontsize        = 8,
        legendfontsize      = 8,
        linewidth           = 1.8,
        grid                = true,
        gridalpha           = 0.05,
        left_margin         = 7Plots.mm,
    )
end

const _C1 = :steelblue
const _C2 = :firebrick
const _C3 = :seagreen
const _C4 = :darkorange
const _C5 = :mediumpurple

# ============================================================
# Individual figure constructors (new style)
# ============================================================

function fig_densities(obj)
    xg = obj.xg

    p1a = plot(xg, obj.uU,    label=L"u_U(x)",           color=_C1)
    plot!(p1a, xg, obj.tU,    label=L"t(x)",             color=_C2)
    plot!(p1a, xg, obj.eU_vec,label=L"e_U(x)",           color=_C3)
    plot!(p1a, xg, obj.mU_x,  label=L"m_U(x)",           color=:black, ls=:dash)
    title!(p1a, "Unskilled segment densities")
    xlabel!(p1a, L"x")
    ylabel!(p1a, "Density")

    p1b = plot(xg, obj.uS,    label=L"u_S(x)",           color=_C1)
    plot!(p1b, xg, obj.eS_tot,label=L"e_S^{\rm tot}(x)", color=_C3)
    plot!(p1b, xg, obj.mS_vec,label=L"m_S(x)",           color=:black, ls=:dash)
    title!(p1b, "Skilled segment densities")
    xlabel!(p1b, L"x")
    ylabel!(p1b, "Density")

    plot(p1a, p1b, layout=(1,2), size=(900,380), margin=5Plots.mm)
end

function fig_unskilled_values(obj)
    xg = obj.xg
    x_bar_idx = findfirst(obj.tauT .> 0.5)
    x_bar = isnothing(x_bar_idx) ? NaN : xg[x_bar_idx]

    p2 = plot(xg, obj.Usearch, label=L"U_U^{\rm search}(x)", color=_C1)
    plot!(p2, xg, obj.net_T,   label=L"-c(x) + T(x)",        color=_C2)
    plot!(p2, xg, obj.UU,      label=L"U_U(x)=\max",         color=:black, ls=:dash, lw=2.2)

    if !isnan(x_bar)
        annotate!(p2, xg[end]*0.95, minimum(obj.Usearch)*1.02,
                  text(L"\bar{x} \approx %$(round(x_bar, digits=3))", :right, 8, :darkgray))
    end
    #title!(p2, "Unskilled unemployment values")
    xlabel!(p2, L"x")
    ylabel!(p2, "Value")
    p2
end

function fig_employment_heatmaps(obj; percentile_print = 1.00)
    xg  = obj.xg
    pgU = obj.pgU
    pg  = obj.pg

    # zoom window
    x_lo,  x_hi  = 0.0, percentile_print
    pU_lo, pU_hi = 0.0, percentile_print
    pS_lo, pS_hi = 0.0, percentile_print

    ix = x_lo  .≤ xg  .≤ x_hi
    iU = pU_lo .≤ pgU .≤ pU_hi
    iS = pS_lo .≤ pg  .≤ pS_hi

    xg_w  = xg[ix]
    pgU_w = pgU[iU];  eU_w = obj.eU_surface[ix, iU]
    pg_w  = pg[iS];   eS_w = obj.eS_mat[ix, iS]

    pstar_U_w = obj.pstar_U[ix]
    pstar_S_w = obj.pstar_S[ix]
    poj_w     = obj.poj[ix]

    p3a = heatmap(xg_w, pgU_w, eU_w',
                  xlabel=L"x", ylabel=L"p",
                  title=L"Unskilled employment $e_U(x,p)$",
                  color=:plasma, legend=false, grid=false,
                  yguidefontrotation=-90)
    contour!(p3a, xg_w, pgU_w, eU_w',
             color=:white, alpha=0.4, lw=0.8, levels=15)
    plot!(p3a, xg_w, pstar_U_w, color=:red, lw=2, ls=:dash)
    ylims!(p3a, 0.0, 1.0)
    p3b = heatmap(xg_w, pg_w, eS_w',
                  xlabel=L"x", ylabel=L"p",
                  title=L"Skilled employment $e_S(x,p)$",
                  color=:viridis, legend=false, grid=false,
                  yguidefontrotation=-90)
    contour!(p3b, xg_w, pg_w, eS_w',
             color=:white, alpha=0.4, lw=0.8, levels=15)
    plot!(p3b, xg_w, pstar_S_w, color=:red, lw=2, ls=:dash)
    plot!(p3b, xg_w, poj_w,     color=:white, lw=2, ls=:dot)
    ylims!(p3b, 0.0, 1.0)
    plot(p3a, p3b, layout=(1,2), size=(1100,440), margin=5Plots.mm)
end

function fig_total_employment(obj; percentile_print = 1.00)
    xg  = obj.xg
    pg  = obj.pg
    pgU = obj.pgU

    x_lo,  x_hi  = 0.0, percentile_print
    pU_lo, pU_hi = 0.0, percentile_print
    pS_lo, pS_hi = 0.0, percentile_print

    ix = x_lo  .≤ xg  .≤ x_hi
    iU = pU_lo .≤ pgU .≤ pU_hi
    iS = pS_lo .≤ pg  .≤ pS_hi

    xg_w  = xg[ix]
    pg_w  = pg[iS]
    eU_w  = obj.eU_surface[ix, iU]
    eS_w  = obj.eS_mat[ix, iS]

    e_total_w = eU_w .+ eS_w

    p3c = heatmap(xg_w, pg_w, e_total_w',
                  xlabel=L"x", ylabel=L"p",
                  title=L"Total employment $e_U(x,p)+e_S(x,p)$",
                  color=:viridis, legend=false, grid=false,
                  yguidefontrotation=-90)
    contour!(p3c, xg_w, pg_w, e_total_w',
             color=:white, alpha=0.4, lw=0.8, levels=15)
    ylims!(p3c, 0.0, 1.0)
    p3c
end

function fig_training_policy(obj)
    p4 = plot(obj.xg, obj.tauT,
              label=L"\tau(x)", color=_C2, lw=2, legend=false)
    #title!(p4, L"Training policy $\tau(x)$")
    xlabel!(p4, L"x")
    ylabel!(p4, L"\tau(x)")
    plot!(p4, yguidefontrotation=-90)
    ylims!(p4, -0.05, 1.15)
    p4
end

function fig_unskilled_frontier(obj)
    xg = obj.xg

    p5a = plot(xg, obj.JU_frontier, label=L"J_U(x,1)", color=_C1, legend=false)
    hline!(p5a, [0.0], ls=:dot, color=:gray, label="")
    title!(p5a, L"Frontier firm value $J_U(x,p{=}1)$")
    xlabel!(p5a, L"x")
    ylabel!(p5a, L"$J_U$", yguidefontrotation=-90)

    p5b = plot(xg, obj.pstar_U, label=L"p^*(x)", color=_C4, legend=false)
    title!(p5b, L"Unskilled reservation rule $p^*(x)$")
    xlabel!(p5b, L"x")
    ylabel!(p5b, L"p^*")
    plot!(p5b, yguidefontrotation=-90)
    ylims!(p5b, 0, 1)

    plot(p5a, p5b, layout=(1,2), size=(900,380), margin=5Plots.mm)
end

function fig_unskilled_value_surfaces(obj)
    xg  = obj.xg
    pgU = obj.pgU

    p6a = heatmap(xg, pgU, obj.JU_surface',
                  xlabel=L"x", ylabel=L"p",
                  title=L"Firm value $J_U(x,p)$", color=:plasma,
                  legend=false, grid=false, yguidefontrotation=-90,
                  left_margin=12Plots.mm)
    contour!(p6a, xg, pgU, obj.JU_surface',
             color=:white, alpha=0.4, lw=0.8, levels=15)
    plot!(p6a, xg, obj.pstar_U, color=:red, lw=2, ls=:dash)
    ylims!(p6a, 0.0, 1.0)

    p6b = heatmap(xg, pgU, obj.EU_surface',
                  xlabel=L"x", ylabel=L"p",
                  title=L"Worker value $E_U(x,p)$", color=:plasma,
                  legend=false, grid=false, yguidefontrotation=-90)
    contour!(p6b, xg, pgU, obj.EU_surface',
             color=:white, alpha=0.4, lw=0.8, levels=15)
    plot!(p6b, xg, obj.pstar_U, color=:red, lw=2, ls=:dash)
    ylims!(p6b, 0.0, 1.0)

    plot(p6a, p6b, layout=(1,2), size=(1000,420), margin=5Plots.mm)
end

function fig_skilled_cutoffs(obj)
    xg = obj.xg
    p7 = plot(xg, obj.pstar_S, label=L"p_S^*(x)", color=_C1, legend=false)
    plot!(p7, xg, obj.poj, label=L"p^{\rm oj}(x)", color=_C2, legend=false)
    #title!(p7, "Skilled cutoffs by type")
    xlabel!(p7, L"x")
    ylabel!(p7, L"p")
    plot!(p7, yguidefontrotation=-90, left_margin=5Plots.mm)
    ylims!(p7, 0, 1)
    p7
end

function fig_skilled_worker_values(obj)
    xg = obj.xg
    pg = obj.pg

    p8a = heatmap(xg, pg, obj.E0_surface',
                  xlabel=L"x", ylabel=L"p",
                  title=L"E_S^0(x,p)", color=:matter,
                  legend=false, grid=false, yguidefontrotation=-90)
    contour!(p8a, xg, pg, obj.E0_surface',
             color=:white, alpha=0.4, lw=0.8, levels=15)
    ylims!(p8a, 0.0, 1.0)

    p8b = heatmap(xg, pg, obj.E1_surface',
                  xlabel=L"x", ylabel=L"p",
                  title=L"E_S^1(x,p)", color=:viridis,
                  legend=false, grid=false, yguidefontrotation=-90)
    contour!(p8b, xg, pg, obj.E1_surface',
             color=:white, alpha=0.4, lw=0.8, levels=15)
    ylims!(p8b, 0.0, 1.0)

    plot(p8a, p8b, layout=(1,2), size=(1000,400), margin=5Plots.mm)
end

function fig_skilled_firm_values(obj)
    xg = obj.xg
    pg = obj.pg

    p9a = heatmap(xg, pg, obj.J0_surface',
                  xlabel=L"x", ylabel=L"p",
                  title=L"J_S^0(x,p)", color=:matter,
                  legend=false, grid=false, yguidefontrotation=-90)
    contour!(p9a, xg, pg, obj.J0_surface',
             color=:white, alpha=0.4, lw=0.8, levels=15)
    ylims!(p9a, 0.0, 1.0)

    p9b = heatmap(xg, pg, obj.J1_surface',
                  xlabel=L"x", ylabel=L"p",
                  title=L"J_S^1(x,p)", color=:viridis,
                  legend=false, grid=false, yguidefontrotation=-90)
    contour!(p9b, xg, pg, obj.J1_surface',
             color=:white, alpha=0.4, lw=0.8, levels=15)
    ylims!(p9b, 0.0, 1.0)

    plot(p9a, p9b, layout=(1,2), size=(1000,400), margin=5Plots.mm)
end

function fig_surplus_heatmaps(obj)
    xg  = obj.xg
    pgU = obj.pgU
    pg  = obj.pg

    p10a = heatmap(xg, pgU, obj.SU_surface',
                   xlabel=L"x", ylabel=L"p",
                   title=L"Unskilled surplus $S_U(x,p)$", color=:plasma,
                   legend=false, grid=false, yguidefontrotation=-90,
                   left_margin=12Plots.mm)
    contour!(p10a, xg, pgU, obj.SU_surface',
             color=:white, alpha=0.4, lw=0.8, levels=15)
    plot!(p10a, xg, obj.pstar_U, color=:red, lw=2, ls=:dash)
    ylims!(p10a, 0.0, 1.0)

    p10b = heatmap(xg, pg, obj.Smax_surface',
                   xlabel=L"x", ylabel=L"p",
                   title=L"Skilled surplus $\max(S_U^0,S_S^1)$", color=:viridis,
                   legend=false, grid=false, yguidefontrotation=-90)
    contour!(p10b, xg, pg, obj.Smax_surface',
             color=:white, alpha=0.4, lw=0.8, levels=15)
    plot!(p10b, xg, obj.pstar_S, color=:red, lw=2, ls=:dash)
    plot!(p10b, xg, obj.poj,     color=:white, lw=2, ls=:dot)
    ylims!(p10b, 0.0, 1.0)

    plot(p10a, p10b, layout=(1,2), size=(1100,440), margin=5Plots.mm)
end

function fig_unemployment_values(obj)
    xg = obj.xg
    p11 = plot(xg, obj.US, label=L"U_S(x)", color=_C3, lw=2)
    plot!(p11, xg, obj.UU, label=L"U_U(x)", color=_C1, lw=2)
    #title!(p11, "Unemployment values")
    xlabel!(p11, L"x")
    ylabel!(p11, "Value")
    p11
end

function fig_unskilled_wage(obj)
    xg  = obj.xg
    pgU = obj.pgU

    pW1 = heatmap(xg, pgU, obj.wU_surface',
                  xlabel=L"x", ylabel=L"p",
                  #title=L"Unskilled wage $w_U(x,p)$", color=:plasma,
                  legend=false, grid=false, yguidefontrotation=-90,
                  left_margin=12Plots.mm)
    contour!(pW1, xg, pgU, obj.wU_surface',
             color=:white, alpha=0.45, lw=0.8, levels=15)
    plot!(pW1, xg, obj.pstar_U, color=:red, lw=2, ls=:dash)
    ylims!(pW1, 0.0, 1.0)
    pW1
end

function fig_skilled_wages(obj)
    xg = obj.xg
    pg = obj.pg

    pW2a = heatmap(xg, pg, obj.wS0_surface',
                   xlabel=L"x", ylabel=L"p",
                   title=L"Skilled wage, no OJS: $w_S^0(x,p)$", color=:matter,
                   legend=false, grid=false, yguidefontrotation=-90,
                   left_margin=12Plots.mm)
    contour!(pW2a, xg, pg, obj.wS0_surface',
             color=:white, alpha=0.45, lw=0.8, levels=15)
    plot!(pW2a, xg, obj.pstar_S, color=:red, lw=2, ls=:dash)
    plot!(pW2a, xg, obj.poj,     color=:white, lw=2, ls=:dot)
    ylims!(pW2a, 0.0, 1.0)

    pW2b = heatmap(xg, pg, obj.wS1_surface',
                   xlabel=L"x", ylabel=L"p",
                   title=L"Skilled wage, OJS: $w_S^1(x,p)$", color=:viridis,
                   legend=false, grid=false, yguidefontrotation=-90,
                   left_margin=12Plots.mm)
    contour!(pW2b, xg, pg, obj.wS1_surface',
             color=:white, alpha=0.45, lw=0.8, levels=15)
    plot!(pW2b, xg, obj.pstar_S, color=:red, lw=2, ls=:dash)
    plot!(pW2b, xg, obj.poj,     color=:white, lw=2, ls=:dot)
    ylims!(pW2b, 0.0, 1.0)

    plot(pW2a, pW2b, layout=(1,2), size=(1100,440), margin=5Plots.mm)
end

# New: wage premium by x (integrated over p)
function fig_skill_premium(obj)
    xg = obj.xg
    pg = obj.pg

    # realised skilled wage
    wS_actual = fill(NaN, obj.Nx, obj.NpS)
    for ix in 1:obj.Nx
        poj_ix = clamp01(obj.poj[ix])
        for jp in 1:obj.NpS
            wS_actual[ix, jp] = pg[jp] < poj_ix ? obj.wS1_surface[ix, jp] : obj.wS0_surface[ix, jp]
        end
    end

    # interpolate unskilled wage onto skilled grid
    wU_on_pg = fill(NaN, obj.Nx, obj.NpS)
    for ix in 1:obj.Nx
        itp = linear_interpolation(obj.pgU, obj.wU_surface[ix, :],
                                   extrapolation_bc=NaN)
        wU_on_pg[ix, :] = itp.(pg)
    end

    # interpolate unskilled employment onto skilled grid
    eU_on_pg = zeros(obj.Nx, obj.NpS)
    for ix in 1:obj.Nx
        itp = linear_interpolation(obj.pgU, obj.eU_surface[ix, :],
                                   extrapolation_bc=0.0)
        eU_on_pg[ix, :] = max.(itp.(pg), 0.0)
    end

    eS_mat = obj.eS_mat
    premium_log_by_x = fill(NaN, obj.Nx)

    for ix in 1:obj.Nx
        numS = 0.0
        denS = 0.0
        numU = 0.0
        denU = 0.0
        for jp in 1:obj.NpS-1
            dp = pg[jp+1] - pg[jp]

            wS1 = wS_actual[ix, jp]
            wS2 = wS_actual[ix, jp+1]
            eS1 = eS_mat[ix, jp]
            eS2 = eS_mat[ix, jp+1]

            wU1 = wU_on_pg[ix, jp]
            wU2 = wU_on_pg[ix, jp+1]
            eU1 = eU_on_pg[ix, jp]
            eU2 = eU_on_pg[ix, jp+1]

            if isfinite(wS1) && isfinite(wS2)
                numS += 0.5 * (wS1*eS1 + wS2*eS2) * dp
                denS += 0.5 * (eS1 + eS2) * dp
            end

            if isfinite(wU1) && isfinite(wU2)
                numU += 0.5 * (wU1*eU1 + wU2*eU2) * dp
                denU += 0.5 * (eU1 + eU2) * dp
            end
        end

        if denS > 0 && denU > 0
            wSbar = numS / denS
            wUbar = numU / denU
            premium_log_by_x[ix] = log(wSbar) - log(wUbar)
        end
    end

    pW3 = plot(xg, premium_log_by_x,
               xlabel=L"x",
               ylabel=L"ln(\bar w_S(x)) - ln(\bar w_U(x))",
               title=L"Skill premium by type $x$",
               lw=2,
               label="")
    hline!(pW3, [0.0], color=:black, ls=:dash, lw=1, label="")
    pW3
end

function fig_wage_densities(obj)
    pW4a = plot(obj.wmid, obj.dens_U, label="Unskilled",
                color=:steelblue, lw=2, fill=(0, 0.15, :steelblue))
    plot!(pW4a, obj.wmid, obj.dens_S, label="Skilled (all)",
          color=:firebrick, lw=2, fill=(0, 0.15, :firebrick))
    title!(pW4a, "Wage density: unskilled vs skilled")
    xlabel!(pW4a, "Wage")
    ylabel!(pW4a, "Density")

    pW4b = plot(obj.wmid, obj.dens_S0, label=L"Skilled,\ s{=}0\ (no\ OJS)",
                color=:seagreen, lw=2, fill=(0, 0.15, :seagreen))
    plot!(pW4b, obj.wmid, obj.dens_S1, label=L"Skilled,\ s{=}1\ (OJS)",
          color=:darkorange, lw=2, fill=(0, 0.15, :darkorange))
    title!(pW4b, "Wage density: skilled by OJS status")
    xlabel!(pW4b, "Wage")
    ylabel!(pW4b, "")

    plot(pW4a, pW4b, layout=(1,2), size=(1100,400), margin=5Plots.mm)
end

function fig_wage_densities_pooled(obj)
    pW4c = plot(obj.wmid, obj.dens_U, label="Unskilled", color=:steelblue, lw=2)
    plot!(pW4c, obj.wmid, obj.dens_S0, label=L"Skilled\ s{=}0", color=:seagreen, lw=2)
    plot!(pW4c, obj.wmid, obj.dens_S1, label=L"Skilled\ s{=}1", color=:darkorange, lw=2, ls=:dash)
    plot!(pW4c, obj.wmid, obj.dens_S,  label="Skilled (pooled)", color=:firebrick, lw=2.2, ls=:dot)
    title!(pW4c, "Wage densities: all employment types")
    xlabel!(pW4c, "Wage")
    ylabel!(pW4c, "Density")
    plot(pW4c, size=(720,480), margin=5Plots.mm)
end

# Optional: extra pooled wage-density figure from notebook (W5)
function fig_wage_pooled_density(obj)
    dens_pooled = obj.dens_U .+ obj.dens_S
    dw          = obj.wmid[2] - obj.wmid[1]
    Z_w         = sum(0.5 .* (dens_pooled[1:end-1] .+ dens_pooled[2:end])) * dw
    dens_pooled_norm = dens_pooled ./ Z_w

    pW5a = plot(obj.wmid, dens_pooled_norm,
                color=:darkviolet, lw=2.2, fill=(0, 0.15, :darkviolet),
                label="Pooled")
    plot!(pW5a, obj.wmid, obj.dens_U ./ Z_w, label="Unskilled",
          color=_C1, lw=1.4, ls=:dash)
    plot!(pW5a, obj.wmid, obj.dens_S ./ Z_w, label="Skilled",
          color=_C2, lw=1.4, ls=:dash)
    #title!(pW5a, "Pooled wage density")
    xlabel!(pW5a, "Wage")
    ylabel!(pW5a, "Density")
    plot(pW5a, size=(720,480), margin=5Plots.mm)
end

"""
    fig_skilled_employment_by_PS(obj, gamma_PS; percentile_print=1.00)

Skilled employment density heatmap with P_S(x) = γ·x^{γ−1} on the
horizontal axis and match quality p on the vertical axis.

The GL quadrature x-grid is non-uniformly spaced, and the mapping
x → P_S is a power law that further warps the spacing.  Passing a
non-uniform vector directly to `heatmap` (GR backend) renders each
node as an equal-width pixel column, producing an all-zero appearance
because the mass is spread across hundreds of imperceptible columns.

Fix: compute PSg = P_S(xg) on the GL grid (monotone for γ > 1),
build a UNIFORM PS grid that spans [PSg[1], γ] with the same number
of points, then interpolate each p-slice of eS_mat and the cutoff
curves onto that uniform grid before calling heatmap.
"""
function fig_skilled_employment_by_PS(obj, gamma_PS::Float64;
                                       percentile_print::Float64 = 1.00)
    xg  = obj.xg
    pg  = obj.pg
    Nx  = obj.Nx

    # ── P_S values at each GL node (monotone ↑ for γ > 1) ────────────────
    PSg = [PS_of_x(x, gamma_PS) for x in xg]   # matches solver convention

    # ── Uniform PS grid: same length as xg, spans [PSg[1], γ] ───────────
    # PSg[1] ≈ γ·xg[1]^{γ-1} ≈ 0;  PSg[end] ≈ γ (PS at x≈1)
    PS_lo      = PSg[1]
    PS_hi      = gamma_PS                        # theoretical max = PS(1) = γ
    PS_uniform = collect(range(PS_lo, PS_hi; length = Nx))

    # ── Window on p (same convention as other heatmaps) ──────────────────
    iS    = 0.0 .≤ pg .≤ percentile_print
    pg_w  = pg[iS]
    NpW   = sum(iS)
    eS_w  = obj.eS_mat[:, iS]                   # (Nx × NpW) on original PSg

    # ── Interpolate each p-slice from PSg → PS_uniform ───────────────────
    # PSg is sorted ↑, so linear_interpolation is valid.
    # Extrapolation at the left edge (PS < PSg[1] ≈ 0) returns 0 (no mass).
    eS_on_PS = zeros(Nx, NpW)
    for jp in 1:NpW
        itp = linear_interpolation(PSg, eS_w[:, jp]; extrapolation_bc = 0.0)
        eS_on_PS[:, jp] = max.(itp.(PS_uniform), 0.0)
    end

    # ── Interpolate cutoff curves from PSg → PS_uniform ──────────────────
    itp_pstar = linear_interpolation(PSg, obj.pstar_S; extrapolation_bc = NaN)
    itp_poj   = linear_interpolation(PSg, obj.poj;     extrapolation_bc = NaN)
    pstar_PS  = itp_pstar.(PS_uniform)
    poj_PS    = itp_poj.(PS_uniform)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig = heatmap(PS_uniform, pg_w, eS_on_PS',
                  xlabel = L"P_S(x) = \gamma\, x^{\gamma-1}",
                  ylabel = L"p",
                  title  = L"Skilled employment $e_S$ by $(P_S(x),\, p)$",
                  color  = :viridis, legend = false, grid = false,
                  yguidefontrotation = -90)
    contour!(fig, PS_uniform, pg_w, eS_on_PS',
             color = :white, alpha = 0.4, lw = 0.8, levels = 15)
    plot!(fig, PS_uniform, pstar_PS,
          color = :red,   lw = 2, ls = :dash, label = L"p_S^*(x)")
    plot!(fig, PS_uniform, poj_PS,
          color = :white, lw = 2, ls = :dot,  label = L"p^{\rm oj}(x)")
    xlims!(fig, 0.0, PS_hi)
    ylims!(fig, 0.0, 1.0)
    fig
end


# ============================================================
# Master function — make and save all figures
# ============================================================

function make_all_plots(obj; output_dir::String = "output/plots", gamma_PS::Union{Float64,Nothing} = nothing)
    mkpath(output_dir)
    _set_theme!()

    figures = [
        ("fig01_densities",           fig_densities(obj)),
        ("fig02_unskilled_values",    fig_unskilled_values(obj)),
        ("fig03_employment_heatmaps", fig_employment_heatmaps(obj)),
        ("fig03b_total_employment",   fig_total_employment(obj)),
        ("fig03c_skilled_emp_by_PS",   isnothing(gamma_PS) ? nothing : fig_skilled_employment_by_PS(obj, gamma_PS)),
        ("fig04_training_policy",     fig_training_policy(obj)),
        ("fig05_unskilled_frontier",  fig_unskilled_frontier(obj)),
        ("fig06_unskilled_surfaces",  fig_unskilled_value_surfaces(obj)),
        ("fig07_skilled_cutoffs",     fig_skilled_cutoffs(obj)),
        ("fig08_skilled_worker_vals", fig_skilled_worker_values(obj)),
        ("fig09_skilled_firm_vals",   fig_skilled_firm_values(obj)),
        ("fig10_surplus_heatmaps",    fig_surplus_heatmaps(obj)),
        ("fig11_unemployment_values", fig_unemployment_values(obj)),
        ("figW1_unskilled_wage",      fig_unskilled_wage(obj)),
        ("figW2_skilled_wages",       fig_skilled_wages(obj)),
        ("figW3_skill_premium",       fig_skill_premium(obj)),
        ("figW4_wage_densities",      fig_wage_densities(obj)),
        ("figW4b_wage_densities_pool", fig_wage_densities_pooled(obj)),
        ("figW5_pooled_wage_density",  fig_wage_pooled_density(obj)),
    ]

    for (name, fig) in figures
        isnothing(fig) && continue
        path = joinpath(output_dir, name * ".png")
        savefig(fig, path)
        println("  saved: $path")
    end

    println("\nAll figures saved to: $output_dir")
    return nothing
end
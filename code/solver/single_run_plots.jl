############################################################
# single_run_plots.jl — RoySearch equilibrium figures
#
#   make_all_plots(obj; output_dir)
#     Generates and saves every figure to output_dir.
#     obj — NamedTuple from compute_equilibrium_objects()
#
# All ability axes are in RANK space F_ℓ(a) = cumulative marginal
# weight, because the ability mass concentrates at the support edges
# (Lise–Meghir–Robin convention).  The model is 2D in (a_U, a_S):
# unskilled objects are indexed by a_U, skilled objects by a_S, and
# aggregate planes live on the (a_U, a_S) copula grid.
#
# `using Plots`, `using LaTeXStrings`, and `using Interpolations`
# are loaded in the driver before these functions are called.
############################################################

# ── Shared style ────────────────────────────────────────────
# One place. Every figure draws from these so the set stays
# visually consistent (axis labels, palette, in-plot text).
function _set_theme!()
    gr()
    theme(:default)
    default(
        fontfamily     = "Computer Modern",
        framestyle     = :box,
        titlefontsize  = 11,
        guidefontsize  = 10,
        tickfontsize   = 8,
        legendfontsize = 8,
        linewidth      = 1.8,
        grid           = true,
        gridalpha      = 0.05,
        left_margin    = 7Plots.mm,
    )
end

# Segment palette — unskilled, skilled, training/τ, directed search d,
# unemployment, reservation-cutoff overlay.
const _C_U     = :steelblue
const _C_S     = :firebrick
const _C_TRAIN = :seagreen
const _C_D     = :darkorange
const _C_UNEMP = :dimgray
const _C_STAR  = :red

# Canonical rank-axis labels (used verbatim by every figure).
const _LAB_RANK_U = L"unskilled-ability rank $F_\ell(a_U)$"
const _LAB_RANK_S = L"skilled-ability rank $F_\ell(a_S)$"

# Ability rank = midpoint cumulative marginal weight on the ability grid.
_rank(wa) = (cumsum(wa) .- 0.5 .* wa) ./ sum(wa)

# Aggregate a 2D (a_U, a_S) mass to a per-a_U vector (sum over a_S) …
_row_agg(M) = vec(sum(M; dims = 2))
# … and to a per-a_S vector (sum over a_U).
_col_agg(M) = vec(sum(M; dims = 1))

# In-plot region text: one treatment for all figures (matches the
# sorting-plane "stay unskilled" style — white, non-bold, centred).
_region!(p, x, y, s; color = :white) =
    annotate!(p, x, y, text(s, color, :center, 11))

# ============================================================
# 1. Sorting plane — the mechanism figure
#    Copula density on ability ranks, with the training frontier τ
#    and the directed-search band d overlaid.  τ, d line labels are
#    drawn horizontally at a fixed offset above each line.
# ============================================================
function fig_sorting_plane(obj)
    rkU = _rank(obj.waU)
    rkS = _rank(obj.waS)

    # Copula density on the rank grid: W2 already carries the marginal
    # weights, so divide them out to expose the copula c_ρ(u,v) implied
    # by (a_ℓ, b_ℓ, ρ_x).
    dens = obj.W2 ./ (obj.waU * obj.waS')

    # Cap the colour range at a robust upper percentile.  At strong |ρ_x|
    # the copula density spikes in two corners, which otherwise saturates
    # the scale and renders the whole interior at magma's dark end (the
    # "no surface" symptom); clims exposes the interior gradient.
    hi = quantile(vec(dens), 0.98)
    p = heatmap(rkU, rkS, dens',
                xlabel = _LAB_RANK_U, ylabel = _LAB_RANK_S,
                color = :magma, legend = false, grid = false,
                clims = (0.0, hi),
                yguidefontrotation = -90, left_margin = 10Plots.mm)

    # Training frontier τ(a_U,a_S): boundary of the train region on the
    # rank grid.  Column-wise first ranks where τ switches to 1.
    τ_edge = fill(NaN, length(rkU))
    for i in 1:length(rkU)
        j = findfirst(obj.τ_mat[i, :] .> 0.5)
        τ_edge[i] = isnothing(j) ? NaN : rkS[j]
    end
    plot!(p, rkU, τ_edge, color = _C_TRAIN, lw = 2.4)

    # Directed-search band d (transition-only): boundary of the d>0 set.
    d_edge = fill(NaN, length(rkU))
    for i in 1:length(rkU)
        j = findfirst(obj.d[i, :] .> 0.5)
        d_edge[i] = isnothing(j) ? NaN : rkS[j]
    end
    plot!(p, rkU, d_edge, color = _C_D, lw = 1.5, ls = :dot)

    # Region labels — identical font, pulled toward region centres.
    _region!(p, 0.30, 0.80, "train")
    _region!(p, 0.62, 0.20, "stay unskilled")

    # Line labels: horizontal, same x, identical vertical offset above
    # each curve.  Interpolate each edge at x_lab to sit on its line.
    x_lab, gap = 0.70, 0.055
    yτ = _interp_edge(rkU, τ_edge, x_lab)
    yd = _interp_edge(rkU, d_edge, x_lab)
    isnan(yτ) || annotate!(p, x_lab, yτ + gap, text(L"\tau(a_U,a_S)", _C_TRAIN, :left, 11))
    isnan(yd) || annotate!(p, x_lab, yd + gap, text(L"d(a_U,a_S)",   _C_D,     :left, 11))

    xlims!(p, 0, 1); ylims!(p, 0, 1)
    p
end

# Linear interpolation of an edge curve (with NaNs) at a query rank.
function _interp_edge(x, y, xq)
    ok = .!isnan.(y)
    (count(ok) < 2) && return NaN
    linear_interpolation(x[ok], y[ok]; extrapolation_bc = NaN)(xq)
end

# ============================================================
# 2. Segment densities over ability rank
#    m_U / m_S segment-mass lines removed (per request); the
#    unemployed / training / employed shares remain.
# ============================================================
function fig_densities(obj)
    rkU = _rank(obj.waU)
    rkS = _rank(obj.waS)

    uU = _row_agg(obj.uU);   tU = _row_agg(obj.tU);   eU = _row_agg(obj.eU_mat)
    uS = _col_agg(obj.uS_mat)

    p1a = plot(rkU, uU, label = L"u_U(a_U)", color = _C_UNEMP)
    plot!(p1a, rkU, tU, label = L"t(a_U)",   color = _C_TRAIN)
    plot!(p1a, rkU, eU, label = L"e_U(a_U)", color = _C_U)
    xlabel!(p1a, _LAB_RANK_U); ylabel!(p1a, "Density")

    p1b = plot(rkS, uS,          label = L"u_S(a_S)",           color = _C_UNEMP)
    plot!(p1b, rkS, obj.eS_totS, label = L"e_S^{\rm tot}(a_S)", color = _C_S)
    xlabel!(p1b, _LAB_RANK_S); ylabel!(p1b, "Density")

    plot(p1a, p1b, layout = (1, 2), size = (900, 380), margin = 5Plots.mm)
end

# ============================================================
# 3. Segment values — two panels on their own governing rank
#    (unskilled search value in a_U; net training value in a_S).
# ============================================================
function fig_unskilled_values(obj)
    rkU = _rank(obj.waU)
    rkS = _rank(obj.waS)

    p2a = plot(rkU, obj.Usearch, label = L"U_U^{\rm search}(a_U)",
               color = _C_U, legend = :bottomright)
    xlabel!(p2a, _LAB_RANK_U); ylabel!(p2a, "Value")

    p2b = plot(rkS, obj.net_T, label = L"-c(a_S) + T(a_S)",
               color = _C_TRAIN, legend = :bottomright)
    hline!(p2b, [0.0], ls = :dot, color = :gray, label = "")
    xlabel!(p2b, _LAB_RANK_S); ylabel!(p2b, "Value")

    plot(p2a, p2b, layout = (1, 2), size = (900, 380), margin = 5Plots.mm)
end

# ============================================================
# 4. Employment heatmaps — ability rank × match quality p
# ============================================================
function fig_employment_heatmaps(obj)
    rkU = _rank(obj.waU); rkS = _rank(obj.waS)

    p3a = heatmap(rkU, obj.pgU, obj.eU_surface',
                  xlabel = _LAB_RANK_U, ylabel = L"p",
                  title = L"Unskilled employment $e_U(a_U,p)$",
                  color = :plasma, legend = false, grid = false,
                  yguidefontrotation = -90)
    contour!(p3a, rkU, obj.pgU, obj.eU_surface',
             color = :white, alpha = 0.4, lw = 0.8, levels = 15)
    plot!(p3a, rkU, obj.pstar_U, color = _C_STAR, lw = 2, ls = :dash)
    ylims!(p3a, 0.0, 1.0)
    xlims!(p3a, 0.0, 1.0)

    p3b = heatmap(rkS, obj.pg, obj.eS_pS',
                  xlabel = _LAB_RANK_S, ylabel = L"p",
                  title = L"Skilled employment $e_S(a_S,p)$",
                  color = :viridis, legend = false, grid = false,
                  yguidefontrotation = -90)
    contour!(p3b, rkS, obj.pg, obj.eS_pS',
             color = :white, alpha = 0.4, lw = 0.8, levels = 15)
    plot!(p3b, rkS, obj.pstar_S, color = _C_STAR, lw = 2, ls = :dash)
    plot!(p3b, rkS, obj.poj,     color = :white,  lw = 2, ls = :dot)
    ylims!(p3b, 0.0, 1.0)
    xlims!(p3b, 0.0, 1.0)

    plot(p3a, p3b, layout = (1, 2), size = (1100, 440), margin = 5Plots.mm)
end

# ============================================================
# 5. Total employment — a plane over the two ability ranks
#    (like the sorting plane).  No title.
# ============================================================
function fig_total_employment(obj)
    rkU = _rank(obj.waU); rkS = _rank(obj.waS)

    # e_U is 2D mass on (a_U,a_S); e_S is per-a_S.  Broadcast e_S across
    # a_U by the copula column shares to place both on the same plane.
    eU_plane = obj.eU_mat
    colshare = obj.W2 ./ (sum(obj.W2; dims = 1) .+ eps())
    eS_plane = colshare .* obj.eS_totS'
    e_total  = eU_plane .+ eS_plane

    p = heatmap(rkU, rkS, e_total',
                xlabel = _LAB_RANK_U, ylabel = _LAB_RANK_S,
                color = :viridis, legend = false, grid = false,
                yguidefontrotation = -90, left_margin = 10Plots.mm)
    contour!(p, rkU, rkS, e_total',
             color = :white, alpha = 0.4, lw = 0.8, levels = 15)
    p
end

# ============================================================
# 6. Unskilled value surfaces — J_U(a_U,p), E_U(a_U,p)
# ============================================================
function fig_unskilled_value_surfaces(obj)
    rkU = _rank(obj.waU)

    p6a = heatmap(rkU, obj.pgU, obj.JU_surface',
                  xlabel = _LAB_RANK_U, ylabel = L"p",
                  title = L"Firm value $J_U(a_U,p)$", color = :plasma,
                  legend = false, grid = false, yguidefontrotation = -90,
                  left_margin = 12Plots.mm)
    contour!(p6a, rkU, obj.pgU, obj.JU_surface',
             color = :white, alpha = 0.4, lw = 0.8, levels = 15)
    plot!(p6a, rkU, obj.pstar_U, color = _C_STAR, lw = 2, ls = :dash)
    ylims!(p6a, 0.0, 1.0)
    xlims!(p6a, 0.0, 1.0)

    p6b = heatmap(rkU, obj.pgU, obj.EU_surface',
                  xlabel = _LAB_RANK_U, ylabel = L"p",
                  title = L"Worker value $E_U(a_U,p)$", color = :plasma,
                  legend = false, grid = false, yguidefontrotation = -90)
    contour!(p6b, rkU, obj.pgU, obj.EU_surface',
             color = :white, alpha = 0.4, lw = 0.8, levels = 15)
    plot!(p6b, rkU, obj.pstar_U, color = _C_STAR, lw = 2, ls = :dash)
    ylims!(p6b, 0.0, 1.0)
    xlims!(p6b, 0.0, 1.0)

    plot(p6a, p6b, layout = (1, 2), size = (1000, 420), margin = 5Plots.mm)
end

# ============================================================
# 7. Skilled worker / firm value surfaces (appendix)
# ============================================================
function fig_skilled_worker_values(obj)
    rkS = _rank(obj.waS)

    p8a = heatmap(rkS, obj.pg, obj.E0_surface',
                  xlabel = _LAB_RANK_S, ylabel = L"p",
                  title = L"E_S^0(a_S,p)", color = :matter,
                  legend = false, grid = false, yguidefontrotation = -90)
    contour!(p8a, rkS, obj.pg, obj.E0_surface',
             color = :white, alpha = 0.4, lw = 0.8, levels = 15)
    ylims!(p8a, 0.0, 1.0)

    p8b = heatmap(rkS, obj.pg, obj.E1_surface',
                  xlabel = _LAB_RANK_S, ylabel = L"p",
                  title = L"E_S^1(a_S,p)", color = :viridis,
                  legend = false, grid = false, yguidefontrotation = -90)
    contour!(p8b, rkS, obj.pg, obj.E1_surface',
             color = :white, alpha = 0.4, lw = 0.8, levels = 15)
    ylims!(p8b, 0.0, 1.0)

    plot(p8a, p8b, layout = (1, 2), size = (1000, 400), margin = 5Plots.mm)
end

function fig_skilled_firm_values(obj)
    rkS = _rank(obj.waS)

    p9a = heatmap(rkS, obj.pg, obj.J0_surface',
                  xlabel = _LAB_RANK_S, ylabel = L"p",
                  title = L"J_S^0(a_S,p)", color = :matter,
                  legend = false, grid = false, yguidefontrotation = -90)
    contour!(p9a, rkS, obj.pg, obj.J0_surface',
             color = :white, alpha = 0.4, lw = 0.8, levels = 15)
    ylims!(p9a, 0.0, 1.0)

    p9b = heatmap(rkS, obj.pg, obj.J1_surface',
                  xlabel = _LAB_RANK_S, ylabel = L"p",
                  title = L"J_S^1(a_S,p)", color = :viridis,
                  legend = false, grid = false, yguidefontrotation = -90)
    contour!(p9b, rkS, obj.pg, obj.J1_surface',
             color = :white, alpha = 0.4, lw = 0.8, levels = 15)
    ylims!(p9b, 0.0, 1.0)

    plot(p9a, p9b, layout = (1, 2), size = (1000, 400), margin = 5Plots.mm)
end

# ============================================================
# 8. Surplus heatmaps (appendix) — ability rank × p
# ============================================================
function fig_surplus_heatmaps(obj)
    rkU = _rank(obj.waU); rkS = _rank(obj.waS)

    p10a = heatmap(rkU, obj.pgU, obj.SU_surface',
                   xlabel = _LAB_RANK_U, ylabel = L"p",
                   title = L"Unskilled surplus $S_U(a_U,p)$", color = :plasma,
                   legend = false, grid = false, yguidefontrotation = -90,
                   left_margin = 12Plots.mm)
    contour!(p10a, rkU, obj.pgU, obj.SU_surface',
             color = :white, alpha = 0.4, lw = 0.8, levels = 15)
    plot!(p10a, rkU, obj.pstar_U, color = _C_STAR, lw = 2, ls = :dash)
    ylims!(p10a, 0.0, 1.0)
    xlims!(p10a, 0.0, 1.0)

    p10b = heatmap(rkS, obj.pg, obj.Smax_surface',
                   xlabel = _LAB_RANK_S, ylabel = L"p",
                   title = L"Skilled surplus $\max(S_S^0,S_S^1)$", color = :viridis,
                   legend = false, grid = false, yguidefontrotation = -90)
    contour!(p10b, rkS, obj.pg, obj.Smax_surface',
             color = :white, alpha = 0.4, lw = 0.8, levels = 15)
    plot!(p10b, rkS, obj.pstar_S, color = _C_STAR, lw = 2, ls = :dash)
    plot!(p10b, rkS, obj.poj,     color = :white,  lw = 2, ls = :dot)
    ylims!(p10b, 0.0, 1.0)
    xlims!(p10b, 0.0, 1.0)

    plot(p10a, p10b, layout = (1, 2), size = (1100, 440), margin = 5Plots.mm)
end

# ============================================================
# 9. Unemployment values — U_U(a_U) and U_S(a_S)
#    RoySearch has no scalar UU field: the unskilled unemployment
#    value is U_U^search(a_U).  Each series is plotted against its
#    own governing rank.
# ============================================================
function fig_unemployment_values(obj)
    rkU = _rank(obj.waU); rkS = _rank(obj.waS)

    p11a = plot(rkU, obj.Usearch, label = L"U_U(a_U)", color = _C_U, lw = 2)
    xlabel!(p11a, _LAB_RANK_U); ylabel!(p11a, "Value")

    p11b = plot(rkS, obj.US, label = L"U_S(a_S)", color = _C_S, lw = 2)
    xlabel!(p11b, _LAB_RANK_S); ylabel!(p11b, "Value")

    plot(p11a, p11b, layout = (1, 2), size = (900, 380), margin = 5Plots.mm)
end

# ============================================================
# W1. Unskilled wage surface — ability rank × p
# ============================================================
function fig_unskilled_wage(obj)
    rkU = _rank(obj.waU)

    pW1 = heatmap(rkU, obj.pgU, obj.wU_surface',
                  xlabel = _LAB_RANK_U, ylabel = L"p",
                  color = :plasma, legend = false, grid = false,
                  yguidefontrotation = -90, left_margin = 12Plots.mm)
    contour!(pW1, rkU, obj.pgU, obj.wU_surface',
             color = :white, alpha = 0.45, lw = 0.8, levels = 15)
    plot!(pW1, rkU, obj.pstar_U, color = _C_STAR, lw = 2, ls = :dash)
    ylims!(pW1, 0.0, 1.0)
    xlims!(pW1, 0.0, 1.0)
    pW1
end

# ============================================================
# W2. Skilled wage surfaces — no-OJS and OJS
# ============================================================
function fig_skilled_wages(obj)
    rkS = _rank(obj.waS)

    pW2a = heatmap(rkS, obj.pg, obj.wS0_surface',
                   xlabel = _LAB_RANK_S, ylabel = L"p",
                   title = L"Skilled wage, no OJS: $w_S^0(a_S,p)$", color = :matter,
                   legend = false, grid = false, yguidefontrotation = -90,
                   left_margin = 12Plots.mm)
    contour!(pW2a, rkS, obj.pg, obj.wS0_surface',
             color = :white, alpha = 0.45, lw = 0.8, levels = 15)
    plot!(pW2a, rkS, obj.pstar_S, color = _C_STAR, lw = 2, ls = :dash)
    plot!(pW2a, rkS, obj.poj,     color = :white,  lw = 2, ls = :dot)
    ylims!(pW2a, 0.0, 1.0)
    xlims!(pW2a, 0.0, 1.0)

    pW2b = heatmap(rkS, obj.pg, obj.wS1_surface',
                   xlabel = _LAB_RANK_S, ylabel = L"p",
                   title = L"Skilled wage, OJS: $w_S^1(a_S,p)$", color = :viridis,
                   legend = false, grid = false, yguidefontrotation = -90)
    contour!(pW2b, rkS, obj.pg, obj.wS1_surface',
             color = :white, alpha = 0.45, lw = 0.8, levels = 15)
    plot!(pW2b, rkS, obj.pstar_S, color = _C_STAR, lw = 2, ls = :dash)
    ylims!(pW2b, 0.0, 1.0)
    xlims!(pW2b, 0.0, 1.0)

    plot(pW2a, pW2b, layout = (1, 2), size = (1000, 400), margin = 5Plots.mm)
end

# ============================================================
# W4. Wage densities
#    Left  — unskilled vs skilled (pooled).
#    Right — skilled by OJS status, s=0 and s=1 on ONE joint
#            normalization so they collectively integrate to 1;
#            distinct fills retained.
# ============================================================
function fig_wage_densities(obj)
    pW4a = plot(obj.wmid, obj.dens_U, label = "Unskilled",
                color = _C_U, lw = 2, fill = (0, 0.15, _C_U))
    plot!(pW4a, obj.wmid, obj.dens_S, label = "Skilled (all)",
          color = _C_S, lw = 2, fill = (0, 0.15, _C_S))
    xlabel!(pW4a, "Wage"); ylabel!(pW4a, "Density")

    # Joint normalization of the two OJS-status densities: rescale by
    # their common integral so ∫(f_{S0}+f_{S1}) dw = 1.  This preserves
    # their relative scale (mass share) instead of normalizing each to 1.
    dw    = obj.wmid[2] - obj.wmid[1]
    Zjoint = sum(obj.dens_S0 .+ obj.dens_S1) * dw
    dS0 = obj.dens_S0 ./ Zjoint
    dS1 = obj.dens_S1 ./ Zjoint

    pW4b = plot(obj.wmid, dS0, label = L"Skilled $s{=}0$ (no OJS)",
                color = _C_TRAIN, lw = 2, fill = (0, 0.30, _C_TRAIN))
    plot!(pW4b, obj.wmid, dS1, label = L"Skilled $s{=}1$ (OJS)",
          color = _C_D, lw = 2, fill = (0, 0.30, _C_D))
    xlabel!(pW4b, "Wage"); ylabel!(pW4b, "Density (joint)")

    plot(pW4a, pW4b, layout = (1, 2), size = (1100, 400), margin = 5Plots.mm)
end

# ============================================================
# W5. Pooled wage density
# ============================================================
function fig_wage_pooled_density(obj)
    dens_pooled = obj.dens_U .+ obj.dens_S
    dw          = obj.wmid[2] - obj.wmid[1]
    Z_w         = sum(0.5 .* (dens_pooled[1:end-1] .+ dens_pooled[2:end])) * dw

    pW5 = plot(obj.wmid, dens_pooled ./ Z_w,
               color = :mediumpurple, lw = 2.2, fill = (0, 0.15, :mediumpurple),
               label = "Pooled")
    plot!(pW5, obj.wmid, obj.dens_U ./ Z_w, label = "Unskilled",
          color = _C_U, lw = 1.4, ls = :dash)
    plot!(pW5, obj.wmid, obj.dens_S ./ Z_w, label = "Skilled",
          color = _C_S, lw = 1.4, ls = :dash)
    xlabel!(pW5, "Wage"); ylabel!(pW5, "Density")
    plot(pW5, size = (720, 480), margin = 5Plots.mm)
end

# ============================================================
# Master function — make and save all figures
# ============================================================
function make_all_plots(obj; output_dir::String = "output/plots")
    mkpath(output_dir)
    _set_theme!()

    figures = [
        ("fig01_sorting_plane",       fig_sorting_plane(obj)),
        ("fig02_densities",           fig_densities(obj)),
        ("fig03_segment_values",      fig_unskilled_values(obj)),
        ("fig04_employment_heatmaps", fig_employment_heatmaps(obj)),
        ("fig05_total_employment",    fig_total_employment(obj)),
        ("fig06_unskilled_surfaces",  fig_unskilled_value_surfaces(obj)),
        ("fig07_unemployment_values", fig_unemployment_values(obj)),
        ("fig08_wage_densities",      fig_wage_densities(obj)),
        ("fig09_pooled_wage_density", fig_wage_pooled_density(obj)),
        ("figA1_skilled_worker_vals", fig_skilled_worker_values(obj)),
        ("figA2_skilled_firm_vals",   fig_skilled_firm_values(obj)),
        ("figA3_surplus_heatmaps",    fig_surplus_heatmaps(obj)),
        ("figA4_unskilled_wage",      fig_unskilled_wage(obj)),
        ("figA5_skilled_wages",       fig_skilled_wages(obj)),
    ]

    for (name, fig) in figures
        path = joinpath(output_dir, name * ".png")
        savefig(fig, path)
        println("  saved: $path")
    end

    println("\nAll figures saved to: $output_dir")
    return nothing
end

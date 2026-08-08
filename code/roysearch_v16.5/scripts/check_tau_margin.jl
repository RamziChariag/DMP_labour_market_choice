#!/usr/bin/env julia
############################################################
# scripts/check_tau_margin.jl — how close is the training frontier to degenerate?
#
# The objective rejects a parameter vector when the training frontier is degenerate:
#
#   solver/unskilled.jl   uc.τT[i, j] = (Utr_j >= uc.Usearch[i]) ? 1.0 : 0.0
#   smm/smm.jl            all(iszero, τv) || all(isone, τv)  →  return Inf
#
# τ is a 0/1 indicator over an Nx×Nx grid, so that guard is DISCRETE in a quantity
# that is CONTINUOUS in θ.  Where the frontier admits only a handful of cells, a
# perturbation far below any economically meaningful size — 1e-15, the rounding of a
# transform round trip — flips the last cell to zero and the point becomes infeasible.
# That is consistent with the observed pattern: a warm start reproduces for
# base_covid and is rejected for base_fc, the window with the smallest training share.
#
# Nothing in the run reports the cell count, so this script measures it: how many
# cells are on the training side, and how large the smallest surviving margin
# Utr_j − Usearch[i] is.  A margin at 1e-12 is a knife edge; a margin at 1e-2 is not.
#
#   julia --project=. code/scripts/check_tau_margin.jl                 # all bundles
#   julia --project=. code/scripts/check_tau_margin.jl base_fc
#
# One solve per window.
############################################################

# Mirrors smm_main.jl's preamble exactly, Base.Threads included: the solver files
# call a bare @threads, which is only in scope with that import.
using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Distributions
using FastGaussQuadrature
using Interpolations
using Parameters
using Printf
using Base.Threads
using Optim
using CSV
using DataFrames
using Serialization
using Clustering
using QuasiMonteCarlo
using JSON3

include(joinpath(@__DIR__, "repo_root.jl"))
const ROOT        = find_repo_root()
const CODE        = joinpath(ROOT, "code")
const ALL_WINDOWS = ["base_fc", "crisis_fc", "base_covid", "crisis_covid"]
const W_SUFFIX    = "_diagonalW"

for f in ("solver/grids.jl", "solver/params.jl", "solver/unskilled.jl",
          "solver/skilled.jl", "solver/solver.jl", "solver/equilibrium.jl",
          "smm/moments.jl", "smm/smm_params.jl", "smm/smm.jl", "smm/candidates.jl")
    include(joinpath(CODE, f))
end

bundle_optimum(path) = let saved = deserialize(path)
    (saved.result isa AbstractVector ? argmin(r -> r.loss_opt, saved.result) : saved.result,
     saved.spec)
end

"""
    tau_report(θ_unc, spec) -> NamedTuple

Solve at `θ_unc` and measure the training frontier: how many cells sit on each side,
what mass they carry, and the tightest margin on either side of the boundary.

The margin is the quantity that matters.  τ is set by `Utr_j >= Usearch[i]`, so the
smallest positive `Utr_j − Usearch[i]` is the distance to losing that cell.  If it is
of order machine epsilon relative to the values involved, the cell — and with it the
feasibility of the whole point — is decided by rounding.
"""
function tau_report(θ_unc :: AbstractVector{Float64}, spec :: SMMSpec)
    cp, up, sp = unpack_θ(θ_unc, spec)
    model, sr  = solve_model(cp, up, sp, spec.sim;
                             Nx = spec.run.Nx, Np_U = spec.run.Np_U, Np_S = spec.run.Np_S)
    # Which of the three flags failed matters: ok = converged_U && converged_S &&
    # converged_global (solver.jl:47), and the global flag is itself a discrete test
    # `d < tol_global` evaluated under an iteration cap.  A marginal global
    # convergence is a second knife edge, independent of the frontier one, and only
    # the flag breakdown tells them apart.
    sr.ok || return (ok = false, cU = sr.converged_U, cS = sr.converged_S,
                     cG = sr.converged_global)

    obj = compute_equilibrium_objects(model)
    τ   = obj.τ_mat
    n1  = count(x -> x > 0.5, τ)
    n   = length(τ)

    # Rebuild the two sides of the comparison the solver used, so the margin is the
    # same number that decides each cell rather than a proxy for it.
    xg      = model.grids.x
    Nx      = length(xg)
    net_T   = obj.net_T                     # −c(aS) + T(aS), indexed by aS
    Usearch = obj.Usearch                   # indexed by aU

    gap_pos =  Inf     # smallest margin among cells currently ON  (τ = 1)
    gap_neg = -Inf     # largest  margin among cells currently OFF (τ = 0), i.e. closest to switching on
    for j in 1:Nx, i in 1:Nx
        g = net_T[j] - Usearch[i]
        if g >= 0.0
            gap_pos = min(gap_pos, g)
        else
            gap_neg = max(gap_neg, g)
        end
    end

    scale = max(maximum(abs, net_T), maximum(abs, Usearch), 1.0)
    # training_share is a moment, not an equilibrium field: the object carries the
    # training mass agg_t and total_pop, and their ratio is the model's share.
    return (ok = true, n_train = n1, n_cells = n,
            frac_cells = n1 / n, gap_pos = gap_pos, gap_neg = gap_neg,
            scale = scale, rel_gap = gap_pos / scale,
            train_mass = obj.total_pop > 0 ? obj.agg_t / obj.total_pop : NaN)
end

function main(argv = ARGS)
    windows = isempty(argv) ? ALL_WINDOWS : collect(String.(argv))
    for w in windows
        w in ALL_WINDOWS || error("unknown window $w; valid: " * join(ALL_WINDOWS, ", "))
    end

    @printf("Training-frontier margin at each window's saved optimum\n  repo : %s\n\n", ROOT)
    @printf("  %-13s %-9s %-11s %-13s %-13s %s\n",
            "window", "cells on", "share", "min margin", "rel to scale", "verdict")

    rows = NamedTuple[]
    for w in windows
        path = joinpath(ROOT, "output", "smm", "smm_result_$(w)$(W_SUFFIX).jls")
        if !isfile(path)
            @printf("  %-13s no bundle\n", w)
            continue
        end
        res, spec = bundle_optimum(path)
        r = tau_report(res.theta_opt, spec)
        if !r.ok
            # Naming the failed flag separates the two knife edges: a false global
            # flag means the coupling loop ran out of iterations, which is a
            # tolerance-and-cap question, not a frontier one.
            @printf("  %-13s REJECTED before the frontier — converged_U=%s converged_S=%s converged_global=%s\n",
                    w, r.cU, r.cS, r.cG)
            push!(rows, (window = w, cells_on = -1, cells = -1, frac_cells = NaN,
                         min_margin = NaN, rel_margin = NaN, train_mass = NaN,
                         verdict = r.cG ? "solver block failed" : "GLOBAL LOOP NOT CONVERGED"))
            continue
        end

        # A relative margin near machine epsilon means the cell is decided by
        # rounding; 1e-6 and above is a real economic distance from the boundary.
        verdict = r.n_train == 0            ? "ALREADY DEGENERATE" :
                  r.rel_gap < 1e-13         ? "KNIFE EDGE"         :
                  r.rel_gap < 1e-8          ? "fragile"            : "safe"
        @printf("  %-13s %-9d %-11s %-13s %-13s %s\n", w,
                r.n_train, @sprintf("%.4f", r.frac_cells),
                @sprintf("%.3e", r.gap_pos), @sprintf("%.3e", r.rel_gap), verdict)
        push!(rows, (window = w, cells_on = r.n_train, cells = r.n_cells,
                     frac_cells = r.frac_cells, min_margin = r.gap_pos,
                     rel_margin = r.rel_gap, train_mass = r.train_mass,
                     verdict = verdict))
    end

    isempty(rows) && (println("\nNothing measured."); return rows)

    println()
    unconv = filter(r -> r.verdict == "GLOBAL LOOP NOT CONVERGED", rows)
    if !isempty(unconv)
        @printf("Global loop not converged: %s\n", join([r.window for r in unconv], ", "))
        println("""
        For these windows the solve never reaches ok, so the frontier guard is never
        evaluated and the frontier is not the cause.  `ok` requires
        `d < tol_global` to hold within maxit_global iterations (solver.jl:179, :47).
        The bundle carries tol_global = 1e-4 with maxit_global = 20 and conv_streak = 1,
        so a point whose residual only just clears the tolerance at the cap flips to
        unconverged under a perturbation of any size — including the 1e-15 of a
        transform round trip.  Raising maxit_global is the direct test: if the same
        seed is accepted at maxit_global = 60, the cap was the binding constraint.""")
    end

    knife = filter(r -> r.verdict in ("KNIFE EDGE", "ALREADY DEGENERATE", "fragile"), rows)
    if isempty(knife) && isempty(unconv)
        println("""
        Every window solves and sits a real distance from the frontier's boundary, so
        neither knife edge explains the rejection and the cause is elsewhere.""")
    elseif isempty(knife)
        println("""
        Among the windows that did solve, none sits near the frontier's boundary, so the
        frontier guard is not the cause for those.""")
    else
        @printf("Fragile: %s\n", join([r.window for r in knife], ", "))
        println("""
        For these windows the training side of the frontier survives by a margin at or
        near rounding error, so whether the point is feasible is decided by the last
        bits of the arithmetic — which is why an identical seed can be accepted when
        read verbatim and rejected when passed through a transform round trip.

        The guard itself is the thing to reconsider: rejecting on an EXACTLY empty
        frontier makes feasibility discontinuous in θ, and a tolerance-based test
        (a minimum share of cells, or a minimum training mass) would put the
        boundary somewhere economically meaningful instead. That is a specification
        choice, not a coding one.""")
    end

    out = joinpath(ROOT, "output", "smm", "tau_margin.csv")
    CSV.write(out, DataFrame(rows))
    @printf("\nwrote %s\n", out)
    return rows
end

main()

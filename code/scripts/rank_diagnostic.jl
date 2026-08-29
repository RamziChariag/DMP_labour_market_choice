#!/usr/bin/env julia
# rank_diagnostic.jl — is the moment set able to separate the free parameters?
#
#   ROYSEARCH_WINDOW=base_fc julia --project=. -t 8 code/scripts/rank_diagnostic.jl
#
# Builds the whitened moment Jacobian Ĝ = ∂m/∂θ at the incumbent by central finite
# differences, then reads its singular values. The question a rank test answers is
# prior to every other diagnostic: a direction in parameter space along which no
# moment moves is a direction the data cannot speak to, and no optimiser setting,
# no sampler tuning and no additional runtime can recover it. If the effective rank
# is below the number of free parameters, the standard errors on the null directions
# are not standard errors, and adding moments is the only repair.
#
# Whitening by the sampling standard deviations puts every moment in units of its own
# noise, which is the same weighting the objective uses (W = diag(1/σ̂²)). Singular
# values are then comparable across moments of different scale.
#
# The step is taken in UNCONSTRAINED t, where the box map keeps every trial point
# inside the bounds, and the reported Jacobian is converted to constrained θ so the
# singular vectors read in natural parameter units.
#
# READ-ONLY: writes one CSV, touches no bundle.

using LinearAlgebra, SparseArrays, Statistics, Random, Printf, Serialization
using Distributions, FastGaussQuadrature, Interpolations, Parameters, Base.Threads
using Optim, CSV, DataFrames, Clustering, QuasiMonteCarlo, JSON3
BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "repo_root.jl"))
const R = find_repo_root()
for f in ("grids", "params", "unskilled", "skilled", "solver", "equilibrium")
    include(joinpath(R, "code", "solver", f * ".jl"))
end
for f in ("moments", "smm_params", "smm")
    include(joinpath(R, "code", "smm", f * ".jl"))
end

const WINDOW   = get(ENV, "ROYSEARCH_WINDOW", "base_fc")
const W_SUFFIX = "_diagonalW"
const OUT      = get(ENV, "RANK_OUT",
                     joinpath(R, "output", "tables",
                              "rank_diagnostic_$(WINDOW)$(W_SUFFIX).csv"))
# Step in unconstrained t. Large enough to clear the criterion's ΔQ grain (measured
# at ≈0.59 as h→0, a kink signature rather than numerical noise), small enough to stay
# inside one smooth region. 1e-3 sat in the stable band on the earlier step-size sweep.
const H = parse(Float64, get(ENV, "RANK_H", "1e-3"))

bundle = deserialize(joinpath(R, "output", "smm",
                              "smm_result_$(WINDOW)$(W_SUFFIX).jls"))
spec   = bundle.spec
θ0     = collect(float.(bundle.result.theta_opt))
free   = spec.free
d      = length(free)

mk = active_moment_keys(spec)
k  = length(mk)
# W is diag(1/σ̂²) over the active moments in this same order, so the whitening the
# objective applies is already in the matrix. Reading it from there rather than from
# the sigma file keeps the diagnostic and the criterion in step if either is rebuilt.
σ̂  = [1.0 / sqrt(spec.W[i, i]) for i in 1:k]

@printf("window %s   moments k=%d   free params d=%d   r = k-d = %d\n", WINDOW, k, d, k - d)
@printf("bundle Q = %.6f   step h = %.1e (unconstrained)   threads = %d\n\n",
        bundle.result.loss_opt, H, nthreads())

"""
Moment vector at an unconstrained point, or `nothing` if the solve fails.

Goes through `smm_objective`'s own `moments_out` buffer rather than rebuilding the
moment vector here, so the ordering can never drift from the one `spec.W` is built
against — the objective and this diagnostic read the same source of truth.
"""
function moments_at(θ)
    buf = Vector{Float64}(undef, k)
    Q   = smm_objective(θ, spec; moments_out = buf)
    (isfinite(Q) && all(isfinite, buf)) ? buf : nothing
end

m0 = moments_at(θ0)
m0 === nothing && error("the incumbent does not solve — nothing to differentiate")

# Each coordinate gets the largest step at which BOTH sides solve, found by shrinking
# from H. A single step for all of them is the wrong instrument here: the feasible set
# has holes rather than a boundary — which coordinates fail changes with the step — so
# a fixed H reports an empty column for a parameter that differentiates perfectly well
# at half the distance. Shrinking is safe in one direction only: too large risks a hole
# or leaving the linear region, too small risks the noise floor, so the search stops at
# the first step that works and records it.
#
# Central differences where both sides are feasible, one-sided where only one is at the
# smallest step tried. A coordinate with neither side feasible anywhere is reported
# rather than silently zeroed: a zero column is indistinguishable from a flat direction.
const H_SHRINK = [1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3]

G      = zeros(k, d)
scheme = fill("central", d)
h_used = fill(NaN, d)
@threads for j in 1:d
    for frac in H_SHRINK
        h  = H * frac
        mp = moments_at((θ = copy(θ0); θ[j] += h; θ))
        mm = moments_at((θ = copy(θ0); θ[j] -= h; θ))
        if mp !== nothing && mm !== nothing
            G[:, j] = (mp .- mm) ./ (2h);  h_used[j] = h
            break
        elseif frac == last(H_SHRINK)
            # Last resort at the smallest step: one side, or nothing.
            if mp !== nothing
                G[:, j] = (mp .- m0) ./ h;  scheme[j] = "forward";  h_used[j] = h
            elseif mm !== nothing
                G[:, j] = (m0 .- mm) ./ h;  scheme[j] = "backward"; h_used[j] = h
            else
                scheme[j] = "INFEASIBLE"
            end
        end
    end
end

# Whiten rows by the sampling sd, matching the objective's own weighting, and rescale
# columns to constrained units so a singular vector reads in natural parameters.
dθdt = [begin
            u = (_to_constrained(θ0[j], f.lb, f.ub) - f.lb) / (f.ub - f.lb)
            (f.ub - f.lb) * u * (1 - u)
        end for (j, f) in enumerate(free)]
Gw = (G ./ σ̂) .* dθdt'

F   = svd(Gw)
sv  = F.S
tol = maximum(sv) * max(k, d) * eps(Float64)      # LAPACK's default rank tolerance
rk  = count(>(tol), sv)

@printf("singular values (largest first):\n")
for (i, s) in enumerate(sv)
    @printf("  %2d  %12.4e   ratio to largest %10.3e%s\n",
            i, s, s / sv[1], s > tol ? "" : "   <- BELOW TOLERANCE")
end
@printf("\neffective rank %d of %d   condition number %.3e\n", rk, d, sv[1] / sv[end])
@printf("infeasible columns: %d   %s\n", count(==("INFEASIBLE"), scheme),
        join([free[j].name for j in 1:d if scheme[j] == "INFEASIBLE"], ", "))

# The weakest directions are the ones the data cannot resolve. Printing their loadings
# names the parameter combinations a new moment would have to move.
@printf("\nthe %d weakest directions, by loading:\n", min(4, d))
for i in d:-1:max(1, d - 3)
    v = F.V[:, i]
    ord = sortperm(abs.(v), rev = true)[1:min(6, d)]
    @printf("  σ=%.4e (%.2e of largest): ", sv[i], sv[i] / sv[1])
    println(join([@sprintf("%s %+.3f", free[j].name, v[j]) for j in ord], "  "))
end

CSV.write(OUT, DataFrame(
    index        = 1:d,
    singular_val = sv,
    ratio        = sv ./ sv[1],
    above_tol    = sv .> tol,
))
@printf("\nwrote %s\n", OUT)

# The whitened Jacobian itself, one row per moment and one column per parameter, so the
# sensitivity pattern can be read directly rather than only through its spectrum. A
# column of near-zeros is a parameter no moment responds to; two proportional columns
# are a pair the moments cannot separate.
jac_out = replace(OUT, ".csv" => "_jacobian.csv")
jdf = DataFrame(moment = String.(string.(mk)))
for (j, f) in enumerate(free)
    jdf[!, param_symbol(f)] = Gw[:, j]
end
CSV.write(jac_out, jdf)

sch_out = replace(OUT, ".csv" => "_scheme.csv")
CSV.write(sch_out, DataFrame(param     = [param_symbol(f) for f in free],
                             scheme    = scheme,
                             h_used    = h_used,
                             dtheta_dt = dθdt))
@printf("wrote %s\n     %s\n", jac_out, sch_out)

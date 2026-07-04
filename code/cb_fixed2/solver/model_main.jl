############################################################
# code/solver/model_main.jl — Standalone single solve + plots
#
# Usage (from project root):
#   julia --threads auto code/solver/model_main.jl
#
# Solves the model at the hard-coded parameters below and writes
# the full set of equilibrium figures from single_run_plots.jl to
# OUTPUT_DIR.  To plot at SMM-estimated parameters, use
# code/solver/plots_main.jl instead.
############################################################

println("="^60)
println("  Segmented Search Model — Standalone Single-Run Plots")
println("="^60)
flush(stdout)

# Paths
const SOLVER_DIR   = @__DIR__
const PROJECT_ROOT = joinpath(SOLVER_DIR, "..", "..")
const OUTPUT_DIR   = joinpath(PROJECT_ROOT, "output")
const PLOTS_ROOT   = joinpath(OUTPUT_DIR, "plots")

# Packages
print("Loading packages... "); flush(stdout)

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

using Plots
using LaTeXStrings

println("done."); flush(stdout)

Random.seed!(1234)

# Solver modules
print("Loading solver modules... "); flush(stdout)

include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))

println("done."); flush(stdout)

# Plotting library
print("Loading plotting modules... "); flush(stdout)

include(joinpath(SOLVER_DIR, "single_run_plots.jl"))

println("done."); flush(stdout)
@printf("Threads available: %d\n\n", Threads.nthreads())
flush(stdout)

# ============================================================
# 1. Parameters
# ============================================================
# SMM run: Q = 2.0886753874, converged, 3982 iterations
common = CommonParams(
    r   = 0.00416667,   # fixed (= 0.05/12, monthly)
    ν   = 0.00323032,   # fixed
    φ   = 0.02222129,   # fixed
    a_ℓ = 1.00000000,
    b_ℓ = 1.00000000,
    c   = 2.50765113,
    A   = 7.75582139,   # LOG scale ⇒ exp(A) ≈ 2335.9
)

unsk_par = UnskilledParams(
    μ   = 0.50989021,
    η   = 0.50000000,   # fixed
    k   = 2.24709,      # months of average U output (= 0.74439621 in old exp(A) units)
    β   = 0.18800000,   # fixed
    λ   = 0.50000000,
    PU  = 1.22383516,
    bU  = 0.00000000,
    bT  = 0.69730101,
    α_U = 1.18038658,
)

skl_par = SkilledParams(
    μ        = 0.16021916,
    η        = 0.50000000,   # fixed
    k        = 4.58674,      # months of average S output (= 2.77085776 in old exp(A) units)
    β        = 0.27200000,   # fixed
    λ        = 0.08614766,
    σ        = 0.00012508,   # OJS flow cost σ_S (not σ_w)
    gamma_PS = 8.99778194,
    bS       = 0.00000000,
    a_Γ      = 5.69684672,
    b_Γ      = 2.79019185,
    ξ        = 0.11000000,
)

sim = SimParams(
    tol_inner          = 1e-7,
    tol_outer_U        = 1e-6,
    tol_outer_S        = 1e-6,
    tol_global         = 1e-4,

    damp_inner_U       = 0.95,
    damp_inner_S       = 0.95,

    inner_B            = 20,     # inner divergence early-abort burn-in (0 disables)
    inner_K            = 10,      # inner no-contraction window W ≡ K; reject after inner_K divergent outer iters

    outer_B            = 30,     # outer stall-detect burn-in (0 disables the handback)
    outer_K            = 10,     # outer no-contraction window; hand back to global on stall

    maxit_inner        = 300,
    maxit_outer        = 200,
    maxit_global       = 20,

    conv_streak        = 1,

    use_anderson       = true,
    anderson_m         = 1,
    anderson_reg       = 1e-10,

    damp_pstar_U       = 1.00,
    damp_pstar_S       = 0.50,

    verbose            = 2,      # 0: model silent; 1: outer convergence per iter; 2: also inner detail
    verbose_stride     = 10,
)

const PLOTS_DIR = joinpath(PLOTS_ROOT, "standalone_default")

println("Parameters:")
@printf("  CommonParams:    r=%.5f   ν=%.5f   φ=%.5f\n",
        common.r, common.ν, common.φ)
@printf("                   a_ℓ=%.5f  b_ℓ=%.5f  c=%.5f\n",
        common.a_ℓ, common.b_ℓ, common.c)
@printf("  Unsk (regime):   PU=%.5f   bU=%.5f   bT=%.5f   α_U=%.5f\n",
        unsk_par.PU, unsk_par.bU, unsk_par.bT, unsk_par.α_U)
@printf("  Skl  (regime):   γ_PS=%.5f  bS=%.5f  a_Γ=%.5f  b_Γ=%.5f\n",
        skl_par.gamma_PS, skl_par.bS, skl_par.a_Γ, skl_par.b_Γ)
@printf("  UnskilledParams: μ=%.5f   η=%.5f   k=%.5f   β=%.5f   λ=%.5f\n",
        unsk_par.μ, unsk_par.η, unsk_par.k, unsk_par.β, unsk_par.λ)
@printf("  SkilledParams:   μ=%.5f   η=%.5f   k=%.5f   β=%.5f   λ=%.5f   σ=%.5f\n",
        skl_par.μ, skl_par.η, skl_par.k, skl_par.β, skl_par.λ, skl_par.σ)
flush(stdout)

# ============================================================
# 2. Solve
# ============================================================
println("\nSolving model...")
@time model, result = solve_model(common, unsk_par, skl_par, sim;
                                   Nx=200, Np_U=200, Np_S=200)

if result.ok
    @printf("Solver converged  (U=%s  S=%s  global=%s)\n",
            result.converged_U, result.converged_S, result.converged_global)
else
    @printf("WARNING: solver did not fully converge  (U=%s  S=%s  global=%s)\n",
            result.converged_U, result.converged_S, result.converged_global)
    @printf("Plots will still be produced from the (non-converged) model state.\n")
end
flush(stdout)

# ============================================================
# 3. Equilibrium objects and accounting
# ============================================================
obj = compute_equilibrium_objects(model)
print_accounting(obj)

# ============================================================
# 4. Generate all single-run plots
# ============================================================
println("\nGenerating figures...")
flush(stdout)

@time make_all_plots(obj; output_dir = PLOTS_DIR,
                          gamma_PS    = skl_par.gamma_PS)

println("\nDone.")
flush(stdout)

# ──────────────────────────────────────────────────────────────────────────
#  Moment-fit diagnostic table  (paste at the bottom of model_main.jl)
#
#    abs_dev      = model − target
#    rel_dev_pct  = 100 · abs_dev / target
#    contribution = w · abs_dev²          (w = 1/target² by default)
#    pct_of_Q     = 100 · contribution / Q,      Q = Σ contribution
#
#  Default w = 1/target² ⇒ contribution == (rel_dev_pct/100)² (the % objective),
#  which reproduces your reported Q. Pass your own diagonal `weights`
#  (e.g. 1 ./ diag(Σ̂)) to decompose a differently-weighted Q instead.
# ──────────────────────────────────────────────────────────────────────────
using Printf

function moment_fit_table(names::AbstractVector, targets::AbstractVector,
                          model::AbstractVector;
                          weights=nothing, sorted::Bool=true, io::IO=stdout)
    n = length(names)
    @assert length(targets) == n == length(model) "names/targets/model length mismatch"

    tgt = collect(float.(targets))
    mdl = collect(float.(model))
    w   = weights === nothing ? (1 ./ tgt.^2) : collect(float.(weights))

    abs_dev     = mdl .- tgt
    rel_dev_pct = 100 .* abs_dev ./ tgt
    contrib     = w .* abs_dev.^2
    Q           = sum(contrib)
    pct_of_Q    = 100 .* contrib ./ Q

    p = sorted ? sortperm(contrib; rev=true) : collect(1:n)
    nm  = collect(string.(names))[p]
    tgt = tgt[p]; mdl = mdl[p]; abs_dev = abs_dev[p]
    rel_dev_pct = rel_dev_pct[p]; contrib = contrib[p]; pct_of_Q = pct_of_Q[p]

    f6(x) = @sprintf("%.6g", x)      # value columns
    f4(x) = @sprintf("%.4f", x)      # percent columns
    wname = max(length("moment"), maximum(length, nm))

    cols(c1,c2,c3,c4,c5,c6,c7,c8) = string(
        lpad(c1,3), "  ", rpad(c2,wname), "  ",
        lpad(c3,12), "  ", lpad(c4,12), "  ", lpad(c5,12), "  ",
        lpad(c6,10), "  ", lpad(c7,13), "  ", lpad(c8,8))

    header = cols("#","moment","target","model","abs_dev","rel_dev%","contribution","%_of_Q")
    println(io, header)
    println(io, "─"^length(header))
    for i in 1:n
        println(io, cols(string(i), nm[i], f6(tgt[i]), f6(mdl[i]), f6(abs_dev[i]),
                         f4(rel_dev_pct[i]), f6(contrib[i]), f4(pct_of_Q[i])))
    end
    println(io, "─"^length(header))
    @printf(io, "Q = Σ contribution = %.7g   (%d moments)\n", Q, n)

    return (; moment=nm, target=tgt, model=mdl, abs_dev,
              rel_dev_pct, contribution=contrib, pct_of_Q)
end

# convenience: pass two Dicts keyed by moment symbol (aligns on target's keys)
function moment_fit_table(targets::AbstractDict, model::AbstractDict;
                          order = collect(keys(targets)), kwargs...)
    ks = collect(order)
    @assert all(haskey(model, k) for k in ks) "model is missing some target keys"
    moment_fit_table(ks, [targets[k] for k in ks], [model[k] for k in ks]; kwargs...)
end

# ── usage ──────────────────────────────────────────────────────────────────
# Wire in whatever your pipeline already holds at this point. Two common shapes:
#
#   # (A) parallel vectors (names + aligned target/model):
#   res = moment_fit_table(moment_names, target_moments, model_moments)
#
#   # (B) dictionaries keyed by moment symbol:
#   res = moment_fit_table(target_moments, model_moments)
#
#   # to decompose YOUR Σ̂-weighted Q instead of the 1/target² default:
#   res = moment_fit_table(moment_names, target_moments, model_moments;
#                          weights = 1 ./ diag(Σ̂))
#
# keep poking at it:
#   sum(res.pct_of_Q[1:5])          # share of Q from the 5 worst-fit moments
#   using DataFrames; DataFrame(res)  # if you'd rather work with a frame

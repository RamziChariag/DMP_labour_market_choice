############################################################
# solver/model_main.jl — Standalone single solve + plots
#
# Usage (from the repo root):
#   julia --project --threads auto solver/model_main.jl
#
# Solves the model at the hard-coded parameters below, prints the
# SMM moment-fit table (targets from the MOMENT_WINDOW bundle), and
# writes the equilibrium figure set from single_run_plots.jl to
# OUTPUT_DIR.
############################################################

println("="^60)
println("  RoySearch — Standalone Single-Run Plots")
println("="^60)
flush(stdout)

# Paths
const SOLVER_DIR   = @__DIR__
const SMM_DIR      = joinpath(SOLVER_DIR, "..", "smm")
const PROJECT_ROOT = joinpath(SOLVER_DIR, "..", "..")
const OUTPUT_DIR   = joinpath(PROJECT_ROOT, "output")
const PLOTS_ROOT   = joinpath(OUTPUT_DIR, "plots")

# ── Moment-table / run-settings switch ──────────────────────────────────────
# Which estimation window supplies the targets, weights, SimParams, and grid.
# NOTE: the hard-coded parameter block in §1 should be the estimate for this
# window, otherwise the table compares apples to oranges.
const MOMENT_WINDOW = :base_covid     # :base_fc | :crisis_fc | :base_covid | :crisis_covid
const W_SUFFIX      = "_equalW"
const SMM_BUNDLE    = joinpath(OUTPUT_DIR, "smm",
                               "smm_result_$(MOMENT_WINDOW)$(W_SUFFIX).jls")

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
using Serialization
using DataFrames
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

# SMM modules — model_moments for the fit table, plus the SMMSpec/SMMResult
# type definitions WITHOUT which deserialize() cannot read the .jls bundle
# (same include set and order as the sensitivity notebook; candidates.jl is
# deliberately not included).
print("Loading SMM modules... "); flush(stdout)
include(joinpath(SMM_DIR, "moments.jl"))
include(joinpath(SMM_DIR, "smm_params.jl"))   # ParamSpec, SMMSpec, run params
include(joinpath(SMM_DIR, "smm.jl"))          # SMMResult, _load_smm_bundle
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
# SMM run: Q = 0.885395387, converged = false, 5000 iterations

common = CommonParams(
    r   = 0.00417,       # fixed (= 0.05/12, monthly)
    ν   = 0.00336,       # fixed (base_covid life-table)
    φ   = 0.02222,       # fixed
    a_ℓ = 0.98056,       # worker type shape a_ℓ
    b_ℓ = 1.17407,       # worker type shape b_ℓ
    ρ_x = -0.66879,      # ability correlation ρ_x
    c   = 0.04676,       # training cost coeff c
    A   = 6.77762,       # aggregate production scale A
)

unsk_par = UnskilledParams(
    μ   = 0.47152,       # unskilled matching eff μ_U
    η   = 0.50000,       # fixed (unsk_η)
    k   = 2.41482,       # unskilled vacancy cost k_U (months of avg U output)
    β   = 0.18800,       # fixed (unsk_β)
    λ   = 0.25486,       # unskilled damage rate λ_U
    PU  = 4.00274,       # unskilled productivity P_U
    α_U = 1.03234,       # unskilled damage shape α_U
    bU  = 0.50045,       # unskilled outside flow b_U
    bT  = 0.00000,       # training flow b_T
    σ_w = 0.23461,       # fixed (unsk_σ_w)
)

skl_par = SkilledParams(
    μ   = 0.18357,       # skilled matching eff μ_S
    η   = 0.50000,       # fixed (skl_η)
    k   = 1.70131,       # skilled vacancy cost k_S (months of avg S output)
    β   = 0.27200,       # fixed (skl_β)
    λ   = 0.92000,       # skilled quality shock λ_S
    σ   = 0.19055,       # OJS flow cost σ_S
    PS  = 5.45172,       # skilled productivity P_S
    a_Γ = 20.00000,      # skilled offer shape a_Γ
    b_Γ = 17.99885,      # skilled offer shape b_Γ
    bS  = 0.15184,       # skilled outside flow b_S
    ξ   = 0.00438,       # skilled exogenous separation ξ_S
    σ_w = 0.21904,       # fixed (skl_σ_w)
)

# Fallback SimParams — used only if the SMM bundle cannot be read.
sim_fallback = SimParams(
    tol_inner          = 1e-7,
    tol_outer_U        = 1e-6,
    tol_outer_S        = 1e-6,
    tol_global         = 1e-4,

    damp_inner_U       = 0.95,
    damp_inner_S       = 0.95,

    inner_B            = 20,
    inner_K            = 10,

    outer_B            = 30,
    outer_K            = 10,

    maxit_inner        = 300,
    maxit_outer        = 200,
    maxit_global       = 20,

    conv_streak        = 1,

    use_anderson       = true,
    anderson_m         = 1,
    anderson_reg       = 1e-10,

    damp_pstar_U       = 1.00,
    damp_pstar_S       = 0.50,

    verbose            = 2,
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
@printf("  Skl  (regime):   P_S=%.5f  bS=%.5f  a_Γ=%.5f  b_Γ=%.5f\n",
        skl_par.PS, skl_par.bS, skl_par.a_Γ, skl_par.b_Γ)
@printf("  UnskilledParams: μ=%.5f   η=%.5f   k=%.5f   β=%.5f   λ=%.5f\n",
        unsk_par.μ, unsk_par.η, unsk_par.k, unsk_par.β, unsk_par.λ)
@printf("  SkilledParams:   μ=%.5f   η=%.5f   k=%.5f   β=%.5f   λ=%.5f   σ=%.5f\n",
        skl_par.μ, skl_par.η, skl_par.k, skl_par.β, skl_par.λ, skl_par.σ)
flush(stdout)

# ============================================================
# 1b. SMM bundle — estimation SimParams, grid, targets, weights
# ============================================================
# Reproducing an SMM fit requires the SAME grid and SimParams as the
# estimation run: this fit sits near a training/no-training knife edge,
# and other (grid, sim) combinations can land on the degenerate τ≡0 twin.
# Everything is wrapped in functions ⇒ no top-level soft-scope issues.

function load_smm_bundle(path::AbstractString)
    isfile(path) || (@warn "SMM bundle not found: $path"; return nothing)
    # Use the SMM module's own loader (handles stale formats gracefully).
    b = _load_smm_bundle(path; delete_on_fail = false,
                         label = "bundle($MOMENT_WINDOW)")
    b === nothing && @warn "Could not read SMM bundle: $path"
    return b
end

function run_settings(bundle, sim_fb)
    bundle === nothing && return (sim_fb, 120, 120, 120)   # estimation-grid fallback
    spec = bundle.spec
    @printf("Loaded run settings from bundle: Nx=%d  Np_U=%d  Np_S=%d\n",
            spec.run.Nx, spec.run.Np_U, spec.run.Np_S)
    # VERBOSE FIX: keep the bundle's grid + solver settings, but let the LOCAL
    # verbose win.  The bundle was estimated with verbose=0 (silent inside SMM),
    # so returning spec.sim as-is muted the inner/outer loop prints on every
    # standalone run.  Rebuild sim with the fallback's verbose/verbose_stride.
    kept = filter(f -> f ∉ (:verbose, :verbose_stride), collect(fieldnames(SimParams)))
    sim  = SimParams(; (f => getfield(spec.sim, f) for f in kept)...,
                       verbose        = sim_fb.verbose,
                       verbose_stride = sim_fb.verbose_stride)
    return (sim, spec.run.Nx, spec.run.Np_U, spec.run.Np_S)
end

const BUNDLE = load_smm_bundle(SMM_BUNDLE)
sim_run, Nx_run, NpU_run, NpS_run = run_settings(BUNDLE, sim_fallback)

# ============================================================
# 1c. Moment-fit diagnostic table
#
#    abs_dev      = model − target
#    rel_dev_pct  = 100 · abs_dev / target
#    contribution = w · abs_dev²
#    pct_of_Q     = 100 · contribution / Q,      Q = Σ contribution
#
#  With weights taken from the bundle (diag(spec.W)/q_scale) the printed
#  Q reproduces the SMM objective exactly (diagonal weight schemes).
# ============================================================

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

# Same accounting as the sensitivity notebook's decompose_Q: iterate the
# bundle's moment dict in order, keep weight>0 entries, index the (already
# subset) W diagonal sequentially, divide by q_scale.
function print_smm_moment_table(obj, bundle; io::IO=stdout)
    if bundle === nothing
        @warn "No SMM bundle — skipping the moment-fit table."
        return nothing
    end
    spec = bundle.spec
    mm   = model_moments(obj)

    ks = Symbol[]; tgt = Float64[]; mdl = Float64[]
    for k in keys(spec.moments)
        spec.moments[k].weight > 0 || continue
        hasproperty(mm, k)         || continue
        push!(ks, k); push!(tgt, spec.moments[k].value); push!(mdl, getproperty(mm, k))
    end
    Wd = Float64[spec.W[i,i] / spec.q_scale for i in eachindex(ks)]

    @printf(io, "\nMoment fit vs %s targets (%s weights, %d active moments):\n",
            MOMENT_WINDOW, W_SUFFIX, length(ks))
    res = moment_fit_table(ks, tgt, mdl; weights = Wd, io = io)
    if bundle.result !== nothing
        @printf(io, "Saved SMM Q for %s: %.7g   (match ⇒ same equilibrium & grid as the estimation)\n",
                MOMENT_WINDOW, bundle.result.loss_opt)
    end
    return res
end

# ============================================================
# 2. Solve
# ============================================================
println("\nSolving model...")
@time model, result = solve_model(common, unsk_par, skl_par, sim_run;
                                   Nx=Nx_run, Np_U=NpU_run, Np_S=NpS_run)

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
# 3b. Moment-fit table (printed before the graphs)
# ============================================================
fit_res = print_smm_moment_table(obj, BUNDLE)
flush(stdout)

# ============================================================
# 4. Generate all single-run plots
# ============================================================
println("\nGenerating figures...")
flush(stdout)

@time make_all_plots(obj; output_dir = PLOTS_DIR)

println("\nDone.")
flush(stdout)
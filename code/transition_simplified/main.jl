############################################################
# code/simulate_transition/main.jl
#
# Transition-dynamics entry point.
#
# Usage (from project root):
#   julia --threads auto code/simulate_transition/main.jl
#
# ── Two crises, each with independent switches ────────────
#   ESTIMATE_FINANCIAL_CRISIS  : run crisis re-estimation
#   ESTIMATE_COVID             : run crisis re-estimation
#   PLOT_FINANCIAL_CRISIS      : compute transition & plot
#   PLOT_COVID                 : compute transition & plot
#
# ── Workflow per crisis ───────────────────────────────────
#   1. (Optional) Estimate baseline on pre-crisis data
#      → done separately via code/smm/main.jl
#   2. (Optional) Re-estimate regime-specific parameters
#      on crisis-window data, fixing deep structural params
#      → controlled by ESTIMATE_* booleans
#   3. Load saved parameters → solve both steady states
#      → forward-simulate transition → make plots
#      → controlled by PLOT_* booleans
#
# Project layout:
#   code/
#     solver/               ← model solver (loaded as library)
#     smm/                  ← SMM infrastructure (loaded as library)
#     simulate_transition/  ← this folder
#       main.jl             ← this file
#       transition.jl       ← forward simulation of distributions
#       crisis_estimation.jl← re-estimation wrapper
#       transition_plots.jl ← plotting functions
#   output/
#     financial_crisis/     ← plots + saved params
#     covid/                ← plots + saved params
############################################################

println("="^60)
println("  Segmented Search Model — Transition Dynamics")
println("="^60)
flush(stdout)

# ══════════════════════════════════════════════════════════════
# USER SWITCHES — set these before running
# ══════════════════════════════════════════════════════════════
const ESTIMATE_FINANCIAL_CRISIS = false    # re-estimate regime params for FC?
const ESTIMATE_COVID            = false    # re-estimate regime params for COVID?
const PLOT_FINANCIAL_CRISIS     = true     # compute transition + plot for FC?
const PLOT_COVID                = true     # compute transition + plot for COVID?

# ══════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════
const TRANS_DIR      = @__DIR__
const SOLVER_DIR     = joinpath(TRANS_DIR, "..", "solver")
const SMM_DIR        = joinpath(TRANS_DIR, "..", "smm")
const PROJECT_ROOT   = joinpath(TRANS_DIR, "..", "..")
const OUTPUT_DIR     = joinpath(PROJECT_ROOT, "output")

# ══════════════════════════════════════════════════════════════
# Packages
# ══════════════════════════════════════════════════════════════
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
using Optim
using Plots

println("done."); flush(stdout)

Random.seed!(2024)

# ══════════════════════════════════════════════════════════════
# Load solver modules
# ══════════════════════════════════════════════════════════════
print("Loading solver modules... "); flush(stdout)

include(joinpath(SOLVER_DIR, "grids.jl"))
include(joinpath(SOLVER_DIR, "params.jl"))
include(joinpath(SOLVER_DIR, "unskilled.jl"))
include(joinpath(SOLVER_DIR, "skilled.jl"))
include(joinpath(SOLVER_DIR, "solver.jl"))
include(joinpath(SOLVER_DIR, "equilibrium.jl"))

println("done."); flush(stdout)

# ══════════════════════════════════════════════════════════════
# Load SMM modules
# ══════════════════════════════════════════════════════════════
print("Loading SMM modules... "); flush(stdout)

include(joinpath(SMM_DIR, "moments.jl"))
include(joinpath(SMM_DIR, "smm_params.jl"))
include(joinpath(SMM_DIR, "smm.jl"))

println("done."); flush(stdout)

# ══════════════════════════════════════════════════════════════
# Load transition modules
# ══════════════════════════════════════════════════════════════
print("Loading transition modules... "); flush(stdout)

include(joinpath(TRANS_DIR, "crisis_estimation.jl"))
include(joinpath(TRANS_DIR, "transition.jl"))
include(joinpath(TRANS_DIR, "transition_plots.jl"))

println("done."); flush(stdout)
@printf("Threads available: %d\n\n", Threads.nthreads())
flush(stdout)


# ══════════════════════════════════════════════════════════════
# Solver settings (shared across all solves in this script)
# ══════════════════════════════════════════════════════════════
const SIM_TRANSITION = SimParams(
    tol_inner      = 1e-8,
    tol_outer_U    = 1e-6,
    tol_outer_S    = 1e-7,
    tol_global     = 1e-3,

    maxit_inner    = 500,
    maxit_outer    = 300,
    maxit_global   = 50,

    conv_streak    = 2,

    use_anderson   = true,
    anderson_m     = 1,
    anderson_reg   = 1e-10,

    damp_pstar_U   = 1.30,
    damp_pstar_S   = 1.00,

    verbose        = 1,
    verbose_stride = 10,
)

# Grid sizes for solving (can be finer than SMM since we only solve twice)
const NX   = 150
const NP_U = 150
const NP_S = 150


# ══════════════════════════════════════════════════════════════
# CRISIS 1: FINANCIAL CRISIS (2008–2009)
# ══════════════════════════════════════════════════════════════

const FC_NAME        = "financial_crisis"
const FC_OUTPUT_DIR  = joinpath(OUTPUT_DIR, FC_NAME)
const FC_PARAMS_FILE = joinpath(FC_OUTPUT_DIR, "estimated_params.csv")

# ── Deep structural parameters (fixed across regimes) ────────
# These come from the baseline (pre-FC 2003–2007) estimation.
# Replace with your actual baseline estimates.
const FC_DEEP_PARAMS = (
    r   = 0.05,     # discount rate (calibrated)
    ν   = 0.02,     # demographic turnover (pre-estimated from CPS)
    φ   = 0.20,     # training completion rate (calibrated)
    a_ℓ = 2.00,     # worker-type Beta shape 1
    b_ℓ = 5.00,     # worker-type Beta shape 2
    c   = 1.70,     # training cost coefficient
    η_U = 0.60,     # unskilled matching elasticity
    η_S = 0.50,     # skilled matching elasticity
    μ_U = 0.74,     # unskilled matching efficiency
    μ_S = 0.90,     # skilled matching efficiency
    β_U = 0.40,     # unskilled Nash bargaining weight
    β_S = 0.32,     # skilled Nash bargaining weight
    σ   = 0.01,     # OJS flow cost
    bU  = 0.00,     # unskilled UI flow value
    bT  = 0.28,     # training flow value
    bS  = 0.01,     # skilled UI flow value
)

# ── Baseline regime parameters (pre-FC 2003–2007) ────────────
# Replace with your actual baseline estimates.
const FC_BASELINE_REGIME = RegimeParams(
    PU  = 0.70,
    PS  = 1.85,
    bU  = 0.00,
    bT  = 0.28,
    bS  = 0.01,
    α_U = 1.00,
    a_Γ = 2.00,
    b_Γ = 5.00,
)

# ── Crisis moments (FC 2008–2009) ────────────────────────────
# Placeholder values — replace with actual data moments.
function load_fc_crisis_moments()
    return (
        ur_total       = (value = 0.085,  weight = 100.0),
        ur_U           = (value = 0.130,  weight =  80.0),
        ur_S           = (value = 0.045,  weight =  80.0),
        skilled_share  = (value = 0.440,  weight =  90.0),
        training_share = (value = 0.025,  weight =  40.0),
        emp_var_U      = (value = 0.055,  weight =  20.0),
        emp_cm3_U      = (value = 0.006,  weight =  10.0),
        emp_var_S      = (value = 0.130,  weight =  20.0),
        emp_cm3_S      = (value = 0.016,  weight =  10.0),

        jfr_U          = (value = 0.160,  weight =  35.0),
        sep_rate_U     = (value = 0.035,  weight =  25.0),
        jfr_S          = (value = 0.100,  weight =  35.0),
        sep_rate_S     = (value = 0.015,  weight =  15.0),
        training_rate  = (value = 0.045,  weight =  15.0),

        mean_wage_U    = (value = 0.650,  weight =  40.0),
        mean_wage_S    = (value = 1.200,  weight =  30.0),
        p50_wage_U     = (value = 0.610,  weight =  30.0),
        p50_wage_S     = (value = 1.130,  weight =  20.0),
        wage_premium   = (value = 0.610,  weight =  45.0),
        wage_sd_U      = (value = 0.230,  weight =  10.0),
        wage_sd_S      = (value = 0.360,  weight =  10.0),

        theta_U        = (value = 0.450,  weight =  15.0),
        theta_S        = (value = 0.950,  weight =  15.0),
    )
end


# ══════════════════════════════════════════════════════════════
# CRISIS 2: COVID-19 (2020–2021)
# ══════════════════════════════════════════════════════════════

const COVID_NAME        = "covid"
const COVID_OUTPUT_DIR  = joinpath(OUTPUT_DIR, COVID_NAME)
const COVID_PARAMS_FILE = joinpath(COVID_OUTPUT_DIR, "estimated_params.csv")

# ── Deep structural parameters (same as FC, from baseline) ───
# For COVID, the baseline is pre-COVID (2015–2019).
# Deep params are held fixed from the first baseline estimation.
# If you want different baseline values, change here.
const COVID_DEEP_PARAMS = FC_DEEP_PARAMS   # same deep params

# ── Baseline regime parameters (pre-COVID 2015–2019) ─────────
# Replace with your actual pre-COVID baseline estimates.
const COVID_BASELINE_REGIME = RegimeParams(
    PU  = 0.72,
    PS  = 1.90,
    bU  = 0.00,
    bT  = 0.28,
    bS  = 0.01,
    α_U = 1.00,
    a_Γ = 2.00,
    b_Γ = 5.00,
)

# ── Crisis moments (COVID 2020–2021) ─────────────────────────
# Placeholder values — replace with actual data moments.
function load_covid_crisis_moments()
    return (
        ur_total       = (value = 0.095,  weight = 100.0),
        ur_U           = (value = 0.150,  weight =  80.0),
        ur_S           = (value = 0.050,  weight =  80.0),
        skilled_share  = (value = 0.460,  weight =  90.0),
        training_share = (value = 0.015,  weight =  40.0),
        emp_var_U      = (value = 0.060,  weight =  20.0),
        emp_cm3_U      = (value = 0.007,  weight =  10.0),
        emp_var_S      = (value = 0.140,  weight =  20.0),
        emp_cm3_S      = (value = 0.018,  weight =  10.0),

        jfr_U          = (value = 0.180,  weight =  35.0),
        sep_rate_U     = (value = 0.040,  weight =  25.0),
        jfr_S          = (value = 0.120,  weight =  35.0),
        sep_rate_S     = (value = 0.018,  weight =  15.0),
        training_rate  = (value = 0.030,  weight =  15.0),

        mean_wage_U    = (value = 0.680,  weight =  40.0),
        mean_wage_S    = (value = 1.280,  weight =  30.0),
        p50_wage_U     = (value = 0.640,  weight =  30.0),
        p50_wage_S     = (value = 1.210,  weight =  20.0),
        wage_premium   = (value = 0.630,  weight =  45.0),
        wage_sd_U      = (value = 0.240,  weight =  10.0),
        wage_sd_S      = (value = 0.370,  weight =  10.0),

        theta_U        = (value = 0.500,  weight =  15.0),
        theta_S        = (value = 1.100,  weight =  15.0),
    )
end


# ══════════════════════════════════════════════════════════════
# SMM settings for crisis re-estimation
# (only regime-specific parameters are free)
# ══════════════════════════════════════════════════════════════
const SIM_SMM = SimParams(
    tol_inner      = 1e-6,
    tol_outer_U    = 1e-6,
    tol_outer_S    = 1e-6,
    tol_global     = 1e-3,

    maxit_inner    = 300,
    maxit_outer    = 200,
    maxit_global   = 30,

    conv_streak    = 2,

    use_anderson   = true,
    anderson_m     = 1,
    anderson_reg   = 1e-10,

    damp_pstar_U   = 1.30,
    damp_pstar_S   = 0.80,

    verbose        = 0,
    verbose_stride = 10,
)

const RUN_PARAMS_CRISIS = SMMRunParams(
    Nx      = 80,
    Np_U    = 80,
    Np_S    = 80,

    sa_max_iter        = 8_000,
    sa_T0              = 4.0,
    sa_step            = 0.20,
    sa_cooling_rate    = 1.0,
    sa_cooling_exp     = 0.5,
    sa_reheat_patience = 300,
    sa_reheat_factor   = 1.10,
    sa_max_reheats     = 2,
    sa_adapt_window    = 50,
    sa_target_fin      = 0.90,

    nm_max_iter  = 300,
    nm_f_tol     = 1e-6,
    nm_x_tol     = 1e-5,

    show_trace_members     = false,
    show_trace_generations = true,
    trace_stride           = 50,
)


# ══════════════════════════════════════════════════════════════
# Transition dynamics settings
# ══════════════════════════════════════════════════════════════
const T_MAX       = 120.0    # simulation horizon (months)
const DT          = 0.05     # Euler step size (months)


# ══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════

# ── Financial Crisis ─────────────────────────────────────────
if ESTIMATE_FINANCIAL_CRISIS
    println("\n" * "="^60)
    println("  Estimating: Financial Crisis (2008–2009)")
    println("="^60)
    flush(stdout)

    mkpath(FC_OUTPUT_DIR)

    run_crisis_estimation(
        crisis_name     = FC_NAME,
        crisis_moments  = load_fc_crisis_moments(),
        deep_params     = FC_DEEP_PARAMS,
        baseline_regime = FC_BASELINE_REGIME,
        sim_smm         = SIM_SMM,
        run_params      = RUN_PARAMS_CRISIS,
        output_file     = FC_PARAMS_FILE,
    )
end

if PLOT_FINANCIAL_CRISIS
    println("\n" * "="^60)
    println("  Transition Dynamics: Financial Crisis (2008–2009)")
    println("="^60)
    flush(stdout)

    mkpath(FC_OUTPUT_DIR)

    # Load crisis parameters
    if !isfile(FC_PARAMS_FILE)
        @error """
        Parameter file not found: $(FC_PARAMS_FILE)
        You must first run the estimation by setting ESTIMATE_FINANCIAL_CRISIS = true.
        """
    else
        fc_crisis_regime = load_crisis_regime(FC_PARAMS_FILE)

        # Build struct containers from deep params
        fc_common = CommonParams(
            r = FC_DEEP_PARAMS.r, ν = FC_DEEP_PARAMS.ν, φ = FC_DEEP_PARAMS.φ,
            a_ℓ = FC_DEEP_PARAMS.a_ℓ, b_ℓ = FC_DEEP_PARAMS.b_ℓ, c = FC_DEEP_PARAMS.c,
        )
        fc_unsk = UnskilledParams(
            μ = FC_DEEP_PARAMS.μ_U, η = FC_DEEP_PARAMS.η_U,
            k = 0.25, β = FC_DEEP_PARAMS.β_U, λ = 0.08,
        )
        fc_skl = SkilledParams(
            μ = FC_DEEP_PARAMS.μ_S, η = FC_DEEP_PARAMS.η_S,
            k = 0.17, β = FC_DEEP_PARAMS.β_S,
            ξ = 0.03, λ = 0.07, σ = FC_DEEP_PARAMS.σ,
        )

        # Solve baseline equilibrium
        println("\nSolving baseline (pre-FC) equilibrium...")
        flush(stdout)
        model_base, res_base = solve_model(
            fc_common, FC_BASELINE_REGIME, fc_unsk, fc_skl, SIM_TRANSITION;
            Nx = NX, Np_U = NP_U, Np_S = NP_S,
        )
        @assert res_base.ok "Baseline model did not converge!"
        obj_base = compute_equilibrium_objects(model_base)
        println("  Baseline solved.  θ_U=$(round(obj_base.thetaU; digits=4))  θ_S=$(round(obj_base.thetaS; digits=4))")

        # Solve crisis equilibrium
        println("\nSolving crisis (FC) equilibrium...")
        flush(stdout)
        model_crisis, res_crisis = solve_model(
            fc_common, fc_crisis_regime, fc_unsk, fc_skl, SIM_TRANSITION;
            Nx = NX, Np_U = NP_U, Np_S = NP_S,
        )
        @assert res_crisis.ok "Crisis model did not converge!"
        obj_crisis = compute_equilibrium_objects(model_crisis)
        println("  Crisis solved.  θ_U=$(round(obj_crisis.thetaU; digits=4))  θ_S=$(round(obj_crisis.thetaS; digits=4))")

        # Compute transition dynamics
        println("\nSimulating transition dynamics...")
        flush(stdout)
        trans = simulate_transition(
            model_base, model_crisis,
            obj_base, obj_crisis;
            T_max = T_MAX, dt = DT,
        )
        println("  Transition simulation complete: $(length(trans.times)) time steps.")

        # Make plots
        println("\nGenerating plots...")
        flush(stdout)
        make_transition_plots(
            trans, obj_base, obj_crisis;
            crisis_name = "Financial Crisis (2008–2009)",
            output_dir  = FC_OUTPUT_DIR,
        )
        println("  Plots saved to: $(FC_OUTPUT_DIR)")
    end
end


# ── COVID-19 ─────────────────────────────────────────────────
if ESTIMATE_COVID
    println("\n" * "="^60)
    println("  Estimating: COVID-19 (2020–2021)")
    println("="^60)
    flush(stdout)

    mkpath(COVID_OUTPUT_DIR)

    run_crisis_estimation(
        crisis_name     = COVID_NAME,
        crisis_moments  = load_covid_crisis_moments(),
        deep_params     = COVID_DEEP_PARAMS,
        baseline_regime = COVID_BASELINE_REGIME,
        sim_smm         = SIM_SMM,
        run_params      = RUN_PARAMS_CRISIS,
        output_file     = COVID_PARAMS_FILE,
    )
end

if PLOT_COVID
    println("\n" * "="^60)
    println("  Transition Dynamics: COVID-19 (2020–2021)")
    println("="^60)
    flush(stdout)

    mkpath(COVID_OUTPUT_DIR)

    # Load crisis parameters
    if !isfile(COVID_PARAMS_FILE)
        @error """
        Parameter file not found: $(COVID_PARAMS_FILE)
        You must first run the estimation by setting ESTIMATE_COVID = true.
        """
    else
        covid_crisis_regime = load_crisis_regime(COVID_PARAMS_FILE)

        covid_common = CommonParams(
            r = COVID_DEEP_PARAMS.r, ν = COVID_DEEP_PARAMS.ν, φ = COVID_DEEP_PARAMS.φ,
            a_ℓ = COVID_DEEP_PARAMS.a_ℓ, b_ℓ = COVID_DEEP_PARAMS.b_ℓ, c = COVID_DEEP_PARAMS.c,
        )
        covid_unsk = UnskilledParams(
            μ = COVID_DEEP_PARAMS.μ_U, η = COVID_DEEP_PARAMS.η_U,
            k = 0.25, β = COVID_DEEP_PARAMS.β_U, λ = 0.08,
        )
        covid_skl = SkilledParams(
            μ = COVID_DEEP_PARAMS.μ_S, η = COVID_DEEP_PARAMS.η_S,
            k = 0.17, β = COVID_DEEP_PARAMS.β_S,
            ξ = 0.03, λ = 0.07, σ = COVID_DEEP_PARAMS.σ,
        )

        # Solve baseline equilibrium
        println("\nSolving baseline (pre-COVID) equilibrium...")
        flush(stdout)
        model_base_c, res_base_c = solve_model(
            covid_common, COVID_BASELINE_REGIME, covid_unsk, covid_skl, SIM_TRANSITION;
            Nx = NX, Np_U = NP_U, Np_S = NP_S,
        )
        @assert res_base_c.ok "Baseline model did not converge!"
        obj_base_c = compute_equilibrium_objects(model_base_c)
        println("  Baseline solved.  θ_U=$(round(obj_base_c.thetaU; digits=4))  θ_S=$(round(obj_base_c.thetaS; digits=4))")

        # Solve crisis equilibrium
        println("\nSolving crisis (COVID) equilibrium...")
        flush(stdout)
        model_crisis_c, res_crisis_c = solve_model(
            covid_common, covid_crisis_regime, covid_unsk, covid_skl, SIM_TRANSITION;
            Nx = NX, Np_U = NP_U, Np_S = NP_S,
        )
        @assert res_crisis_c.ok "Crisis model did not converge!"
        obj_crisis_c = compute_equilibrium_objects(model_crisis_c)
        println("  Crisis solved.  θ_U=$(round(obj_crisis_c.thetaU; digits=4))  θ_S=$(round(obj_crisis_c.thetaS; digits=4))")

        # Compute transition dynamics
        println("\nSimulating transition dynamics...")
        flush(stdout)
        trans_c = simulate_transition(
            model_base_c, model_crisis_c,
            obj_base_c, obj_crisis_c;
            T_max = T_MAX, dt = DT,
        )
        println("  Transition simulation complete: $(length(trans_c.times)) time steps.")

        # Make plots
        println("\nGenerating plots...")
        flush(stdout)
        make_transition_plots(
            trans_c, obj_base_c, obj_crisis_c;
            crisis_name = "COVID-19 (2020–2021)",
            output_dir  = COVID_OUTPUT_DIR,
        )
        println("  Plots saved to: $(COVID_OUTPUT_DIR)")
    end
end


println("\n" * "="^60)
println("  Done.")
println("="^60)
flush(stdout)

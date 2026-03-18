############################################################
# crisis_estimation.jl — Crisis re-estimation wrapper
#
# Re-estimates only regime-specific parameters on crisis-window
# moments, holding deep structural parameters fixed.
#
# Key functions:
#   run_crisis_estimation(...)  → runs SA + NM polish, saves CSV
#   load_crisis_regime(path)   → reads saved CSV → RegimeParams
#   regime_free_params(baseline_regime) → regime-only ParamSpec vector
############################################################


# ============================================================
# Free parameter list for crisis re-estimation
# Only regime-specific parameters are free.
# Deep structural parameters are fixed (from baseline).
# ============================================================

"""
    regime_free_params(baseline_regime) → Vector{ParamSpec}

Build the list of free parameters for crisis re-estimation.
Only regime-specific parameters are included.
Starting values are taken from the baseline regime.
"""
function regime_free_params(baseline_regime::RegimeParams) :: Vector{ParamSpec}
    return [
        ParamSpec(:regime, :PU,  0.001,  5.00, baseline_regime.PU,  "unskilled productivity PU"),
        ParamSpec(:regime, :PS,  0.001, 10.00, baseline_regime.PS,  "skilled productivity PS"),
        ParamSpec(:regime, :α_U, 0.001, 10.00, baseline_regime.α_U, "unskilled damage shape α_U"),
        ParamSpec(:regime, :a_Γ, 0.001,  8.00, baseline_regime.a_Γ, "skilled offer shape a_Γ"),
        ParamSpec(:regime, :b_Γ, 0.001,  8.00, baseline_regime.b_Γ, "skilled offer shape b_Γ"),

        # Vacancy costs and shock rates (regime-specific)
        ParamSpec(:unsk, :k,  0.00001, 2.00, 0.25, "unskilled vacancy cost k_U"),
        ParamSpec(:unsk, :λ,  0.001,   0.99, 0.08, "unskilled damage rate λ_U"),
        ParamSpec(:skl,  :k,  0.00001, 2.00, 0.17, "skilled vacancy cost k_S"),
        ParamSpec(:skl,  :ξ,  0.001,   0.99, 0.03, "skilled exog. sep rate ξ"),
        ParamSpec(:skl,  :λ,  0.001,   0.99, 0.07, "skilled quality shock λ_S"),
    ]
end


# ============================================================
# Main estimation function
# ============================================================

"""
    run_crisis_estimation(;
        crisis_name, crisis_moments, deep_params,
        baseline_regime, sim_smm, run_params, output_file
    )

Run a crisis re-estimation:
  1. Fix deep structural parameters from `deep_params`.
  2. Free only regime-specific parameters (starting from baseline_regime).
  3. Run SA → Nelder-Mead polish.
  4. Save estimated parameters to `output_file`.
"""
function run_crisis_estimation(;
    crisis_name     :: String,
    crisis_moments  :: NamedTuple,
    deep_params     :: NamedTuple,
    baseline_regime :: RegimeParams,
    sim_smm         :: SimParams,
    run_params      :: SMMRunParams,
    output_file     :: String,
)
    @printf("\n── Crisis re-estimation: %s ──\n", crisis_name)
    flush(stdout)

    # Build the fixed-parameters NamedTuple from deep_params
    # This pins all deep structural parameters so SMM only
    # searches over regime-specific ones.
    fixed_params = (
        r   = deep_params.r,
        ν   = deep_params.ν,
        φ   = deep_params.φ,
        a_ℓ = deep_params.a_ℓ,
        b_ℓ = deep_params.b_ℓ,
        c   = deep_params.c,
        # Matching elasticities and efficiencies (deep)
        # Note: these are shared names across blocks, so we
        # handle them via the free_specs below. The SMM will
        # only estimate what's in free_specs and fix the rest.
        β   = deep_params.β_U,    # placeholder — see note
        σ   = deep_params.σ,
        bU  = deep_params.bU,
        bT  = deep_params.bT,
        bS  = deep_params.bS,
    )

    # Free parameters: only regime-specific
    free = regime_free_params(baseline_regime)

    # Build SMM specification
    spec = build_smm_spec(
        crisis_moments, sim_smm;
        fixed      = fixed_params,
        free_specs = free,
        run        = run_params,
    )

    print_spec(spec)

    # Stage 1: Simulated Annealing global search
    @printf("Stage 1: Simulated Annealing...\n"); flush(stdout)
    res_sa = run_smm(spec; method = :sa)

    # Stage 2: Nelder-Mead polish
    @printf("Stage 2: Nelder-Mead polish...\n"); flush(stdout)
    spec_polished = _spec_with_init(spec, res_sa.theta_opt)
    res_nm = run_smm(spec_polished; method = :neldermead)

    results = res_nm

    # Save results
    mkpath(dirname(output_file))
    save_crisis_params(results, output_file)

    @printf("\n── Crisis estimation complete: %s ──\n", crisis_name)
    @printf("  Q = %.6e  |  saved to: %s\n", results.loss_opt, output_file)
    flush(stdout)

    return results
end


# ============================================================
# Save / load crisis parameters
# ============================================================

"""
    save_crisis_params(result, path)

Save crisis-estimated parameters to CSV.
Format: one row per parameter with block, name, value.
The RegimeParams fields are extracted for easy reconstruction.
"""
function save_crisis_params(result::SMMResult, path::String)
    cp, rp, up, sp = unpack_θ(result.theta_opt, result.spec)

    open(path, "w") do io
        println(io, "block,name,value")

        # Regime parameters
        @printf(io, "regime,PU,%.10f\n",  rp.PU)
        @printf(io, "regime,PS,%.10f\n",  rp.PS)
        @printf(io, "regime,bU,%.10f\n",  rp.bU)
        @printf(io, "regime,bT,%.10f\n",  rp.bT)
        @printf(io, "regime,bS,%.10f\n",  rp.bS)
        @printf(io, "regime,alpha_U,%.10f\n", rp.α_U)
        @printf(io, "regime,a_Gamma,%.10f\n", rp.a_Γ)
        @printf(io, "regime,b_Gamma,%.10f\n", rp.b_Γ)

        # Unskilled regime-specific
        @printf(io, "unsk,k,%.10f\n",  up.k)
        @printf(io, "unsk,lambda,%.10f\n", up.λ)

        # Skilled regime-specific
        @printf(io, "skl,k,%.10f\n",  sp.k)
        @printf(io, "skl,xi,%.10f\n", sp.ξ)
        @printf(io, "skl,lambda,%.10f\n", sp.λ)

        # Metadata
        @printf(io, "\n# Q = %.10e\n", result.loss_opt)
        @printf(io, "# converged = %s\n", result.converged)
    end

    @printf("Crisis parameters saved to: %s\n", path)
    flush(stdout)
end


"""
    load_crisis_regime(path) → RegimeParams

Read a saved crisis parameter file and reconstruct the
RegimeParams.  Also returns any unskilled/skilled parameters
that were re-estimated.
"""
function load_crisis_regime(path::String) :: RegimeParams
    @printf("Loading crisis parameters from: %s\n", path)
    flush(stdout)

    # Parse CSV
    params = Dict{String, Float64}()
    for line in readlines(path)
        startswith(line, "#") && continue
        startswith(line, "block") && continue  # header
        isempty(strip(line)) && continue
        parts = split(line, ",")
        length(parts) >= 3 || continue
        key = strip(parts[1]) * "." * strip(parts[2])
        val = tryparse(Float64, strip(parts[3]))
        val !== nothing && (params[key] = val)
    end

    # Reconstruct RegimeParams with defaults for any missing fields
    rp = RegimeParams(
        PU  = get(params, "regime.PU",      0.70),
        PS  = get(params, "regime.PS",      1.85),
        bU  = get(params, "regime.bU",      0.00),
        bT  = get(params, "regime.bT",      0.28),
        bS  = get(params, "regime.bS",      0.01),
        α_U = get(params, "regime.alpha_U", 1.00),
        a_Γ = get(params, "regime.a_Gamma", 2.00),
        b_Γ = get(params, "regime.b_Gamma", 5.00),
    )

    @printf("  Loaded: PU=%.4f  PS=%.4f  α_U=%.4f  a_Γ=%.4f  b_Γ=%.4f\n",
            rp.PU, rp.PS, rp.α_U, rp.a_Γ, rp.b_Γ)
    flush(stdout)

    return rp
end

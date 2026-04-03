############################################################
# smm_params.jl — SMM parameter specification
#
# Key types
# ─────────────────────────────────────────────────────────
#   ParamSpec      metadata for a single free parameter
#   SMMSpec        full SMM problem specification
#
# Key functions
# ─────────────────────────────────────────────────────────
#   build_smm_spec(...)    construct an SMMSpec
#   unpack_θ(θ, spec)      free vector → (Common, Regime, Unsk, Skl) structs
#   pack_θ(spec)           initial values → free vector
#   print_spec(spec)       display the estimation problem
############################################################


# ============================================================
# ParamSpec — metadata for one free parameter
# ============================================================

"""
    ParamSpec

Describes a single free parameter in the SMM problem.

Fields
──────
  block   :: Symbol   which struct (:common, :regime, :unsk, :skl)
  name    :: Symbol   field name within that struct
  lb      :: Float64  lower bound (used for log/logit transform)
  ub      :: Float64  upper bound
  init    :: Float64  starting value
  label   :: String   human-readable label for printing
"""
struct ParamSpec
    block :: Symbol
    name  :: Symbol
    lb    :: Float64
    ub    :: Float64
    init  :: Float64
    label :: String
end


# ============================================================
# SMMRunParams — optimisation and grid settings
# ============================================================

"""
    SMMRunParams

All runtime settings for the SMM estimation: grid sizes,
global search (DE), local polish (Nelder-Mead), and tracing.

Construct with keyword arguments; all have defaults.

Fields — grids
──────────────
  Nx, Np_U, Np_S      grid sizes passed to the solver
                      (coarser = faster iterations)

Fields — global search (DE)
───────────────────────────
  de_max_iter         maximum generations
  de_pop_size         population size (0 = auto: 100 × n_free_params)
  de_f                mutation scale ∈ (0, 2), typically 0.5–0.9
  de_cr               crossover probability ∈ (0, 1), typically 0.7–0.9
  de_patience         early-stop after this many stagnant generations
  de_avg_tol          early-stop when (Q_mean − Q_best) / |Q_best| < this
                      value (population has converged around the best).
                      Set to 0.0 to disable. Default: 0.01 (1 %).

Fields — weight matrix conditioning
────────────────────────────────────
  w_cond_target       weight-matrix mode selector.
                      0.0   →  diagonal from Σ (weights = 1/σ²)
                      1.0   →  compressed diagonal: log(1 + 1/σ²)
                      2.0   →  equal weights (identity, no W matrix)
                      >2.0  →  full optimal W (shrunk if κ > target)
                      Should match TARGET_KAPPA in the data pipeline.

Fields — local polish (Nelder-Mead)
────────────────────────────────────
  nm_max_iter         maximum iterations
  nm_f_tol            function value tolerance
  nm_x_tol            parameter tolerance

Fields — tracing
────────────────
  show_trace_members     print within-generation member progress
  show_trace_generations print end-of-generation summary line
  trace_stride           print member line every N members
"""
Base.@kwdef struct SMMRunParams
    # ── Grids ─────────────────────────────────────────────────────────
    Nx      :: Int     = 80
    Np_U    :: Int     = 80
    Np_S    :: Int     = 80

    # ── DE global search ──────────────────────────────────────────────
    de_max_iter  :: Int     = 200
    de_pop_size  :: Int     = 0        # 0 = 100 × n_free_params
    de_f         :: Float64 = 0.65
    de_cr        :: Float64 = 0.85
    de_patience  :: Int     = 20
    de_avg_tol   :: Float64 = 0.01   # stop when (Q_mean−Q_best)/|Q_best| < tol; 0 = off

    # ── Simulated annealing global search ────────────────────────────
    sa_max_iter      :: Int     = 5_000   # total SA proposals
    sa_T0            :: Float64 = 2.0     # initial temperature
    sa_step          :: Float64 = 0.15    # initial random-walk step (logit space)
    # Cooling schedule: T(t) = T0 / (1 + cooling_rate * t)^cooling_exp
    # Default is log-cooling (cooling_rate=1, exp≈0 absorbed into log);
    # increase cooling_exp toward 1.0 to cool faster, decrease toward 0 to cool slower.
    sa_cooling_rate  :: Float64 = 1.0     # scales t in denominator
    sa_cooling_exp   :: Float64 = 0.5     # exponent on log: T0 / log(1 + rate*t)^exp
    # Reheating: if best hasn't improved for sa_reheat_patience steps, reset
    # current→best and multiply T by sa_reheat_factor (>1 to warm back up).
    # Set sa_reheat_patience=0 to disable.
    sa_reheat_patience :: Int     = 200   # proposals without improvement before reheat
    sa_reheat_factor   :: Float64 = 2.0   # T multiplier on reheat
    sa_max_reheats     :: Int     = 5     # cap on total reheats (0 = unlimited)
    # Adaptive step: every sa_adapt_window proposals, rescale step so that
    # the feasibility rate (fin/total) stays near sa_target_fin.
    # Set sa_adapt_window=0 to disable.
    sa_adapt_window  :: Int     = 50      # rolling window for step adaptation
    sa_target_fin    :: Float64 = 0.90    # target feasibility rate
    sa_random_init     :: Bool    = false     # whether to randomize initial solution for SA (instead of using free_params.init)

     # ── DE global search ──────────────────────────────────────────────

    # ── Weight matrix conditioning ──────────────────────────────────────
    w_cond_target :: Float64 = 1e8   # 0=diagonal, 1=compressed, 2=equal, >2=full W (shrink if κ>target)

    # ── Nelder-Mead polish ────────────────────────────────────────────
    nm_max_iter  :: Int     = 5_000
    nm_f_tol     :: Float64 = 1e-6
    nm_x_tol     :: Float64 = 1e-5

    # ── Tracing ───────────────────────────────────────────────────────
    show_trace_members     :: Bool = false   # within-generation member lines
    show_trace_generations :: Bool = true    # end-of-generation summary lines
    trace_stride           :: Int  = 10
end


# ============================================================
# SMMSpec — full problem specification
# ============================================================

"""
    SMMSpec

Holds everything the SMM estimation needs.

Fields
──────
  free        vector of ParamSpec (parameters to estimate)
  fixed       NamedTuple of pinned parameter values
  moments     NamedTuple from load_data_moments() — (value, weight) pairs
  sim         SimParams — solver settings (tolerances, maxit, verbose…)
  run         SMMRunParams — grid sizes and optimiser settings
  W           Union{Nothing, Matrix{Float64}} — weight matrix
              (Nothing = equal weights; Matrix = diagonal/compressed/full W)
"""
struct SMMSpec
    free    :: Vector{ParamSpec}
    fixed   :: NamedTuple
    moments :: NamedTuple
    sim     :: SimParams
    run     :: SMMRunParams
    W       :: Union{Nothing, Matrix{Float64}}
end


# ============================================================
# Constructor
# ============================================================

"""
    build_smm_spec(moments, sim;
                   fixed        = (;),
                   free_specs   = default_free_params(),
                   run          = SMMRunParams(),
                   W            = nothing,
                   skip_moments = Symbol[])

Build an `SMMSpec`.

- `fixed`: a NamedTuple of parameter values to hold fixed during
  estimation.  Any field name from CommonParams / RegimeParams /
  UnskilledParams / SkilledParams is valid.
  Example: `fixed = (r = 0.05, ν = 0.05, ξ = 0.03)`

- `free_specs`: vector of ParamSpec.  Any entry whose name also
  appears in `fixed` is silently dropped.

- `run`: an `SMMRunParams` controlling grid sizes, DE settings,
  Nelder-Mead settings, and tracing.

- `W`: optional weight matrix from influence functions, already
  subsetted to the active moments by `load_weight_matrix`.
  If nothing (default), uses diagonal weights from moment variances.

- `skip_moments`: vector of moment name Symbols to exclude from the
  SMM objective.  Their weights are set to zero in the stored moments
  NamedTuple, so both `compute_loss` and `compute_loss_matrix` skip
  them automatically.  The moment values are still stored and visible
  in diagnostics (they just show as inactive in `print_spec`).
  Example: `skip_moments = [:emp_cm3_U, :emp_cm3_S]`
  Unknown names produce a warning at spec-build time.
"""
function build_smm_spec(
    moments      :: NamedTuple,
    sim          :: SimParams;
    fixed        :: Any                           = (;),
    free_specs   :: Vector{ParamSpec}             = default_free_params(),
    run          :: SMMRunParams                  = SMMRunParams(),
    W            :: Union{Nothing, Matrix{Float64}} = nothing,
    skip_moments :: Vector{Symbol}                = Symbol[],
)
    # Coerce () or any non-NamedTuple to an empty NamedTuple
    fixed_nt    = (fixed isa NamedTuple) ? fixed : NamedTuple()
    fixed_names = keys(fixed_nt)

    # Drop any free spec whose field appears in `fixed`
    # Supports both plain names (e.g. :r) and block-qualified names
    # (e.g. :unsk_μ, :skl_μ) for disambiguating shared names across blocks.
    active_free = filter(ps -> begin
        qualified = Symbol(string(ps.block) * "_" * string(ps.name))
        !(ps.name in fixed_names || qualified in fixed_names)
    end, free_specs)

    if length(active_free) < length(free_specs)
        dropped = [ps.name for ps in free_specs
                   if ps.name in fixed_names ||
                      Symbol(string(ps.block) * "_" * string(ps.name)) in fixed_names]
        @printf("SMMSpec: fixed override dropped free params: %s\n",
                join(string.(dropped), ", "))
    end

    # Zero out weights for any moments in skip_moments.
    # Unknown names get a warning so typos are caught immediately.
    if !isempty(skip_moments)
        unknown = setdiff(skip_moments, keys(moments))
        if !isempty(unknown)
            @printf("SMMSpec: skip_moments — unrecognised moment names (ignored): %s\n",
                    join(string.(unknown), ", "))
        end
        moments = NamedTuple(
            k => (k in skip_moments ? (value = v.value, weight = 0.0) : v)
            for (k, v) in pairs(moments)
        )
        active_skipped = intersect(skip_moments, keys(moments))
        if !isempty(active_skipped)
            @printf("SMMSpec: skip_moments zeroed weights for (%d): %s\n",
                    length(active_skipped), join(string.(active_skipped), ", "))
        end
    end

    # Validate that W, if provided, matches the number of active moments.
    # Catches stale W matrices (e.g. loaded from a previous .jls bundle or CSV
    # with a different SKIP_MOMENTS list) before the optimisation starts.
    if !isnothing(W)
        K_active = count(v -> v.weight > 0.0, values(moments))
        if size(W, 1) != K_active || size(W, 2) != K_active
            error(
                "W matrix size $(size(W,1))×$(size(W,2)) does not match the number of " *
                "active moments ($K_active). This usually means a stale W was loaded from " *
                "a previous run with a different SKIP_MOMENTS list. " *
                "Call load_weight_matrix(..., skip_moments=SKIP_MOMENTS) with the current " *
                "SKIP_MOMENTS so the matrix is subsetted correctly before building the spec."
            )
        end
    end

    return SMMSpec(active_free, fixed_nt, moments, sim, run, W)
end


# ============================================================
# Default free parameter list
# All structural parameters across the four structs.
# Modify bounds/inits here to tune the estimation.
# ============================================================

"""
    default_free_params() → Vector{ParamSpec}

Returns the full list of estimable parameters with default
bounds and starting values.  Pass a subset to `build_smm_spec`
to estimate only selected parameters.
"""
function default_free_params() :: Vector{ParamSpec}
    return [
        # ── CommonParams ───────────────────────────────────────────────
        ParamSpec(:common, :r,   0.001, 0.10,  0.05,  "discount rate r"),
        ParamSpec(:common, :ν,   0.001, 0.10,  0.05,  "demographic exit ν"),
        ParamSpec(:common, :φ,   0.005, 0.40,  0.20,  "training completion φ"),
        ParamSpec(:common, :a_ℓ, 0.30,  6.00,  2.00,  "worker type shape a_ℓ"),
        ParamSpec(:common, :b_ℓ, 0.30,  6.00,  5.00,  "worker type shape b_ℓ"),
        ParamSpec(:common, :c,   0.01,  3.00,  1.70,  "training cost coeff c"),

        # ── RegimeParams ───────────────────────────────────────────────
        ParamSpec(:regime, :PU,  0.05,  3.00,  0.70,  "unskilled productivity PU"),
        ParamSpec(:regime, :PS,  0.10,  5.00,  1.85,  "skilled productivity PS"),
        ParamSpec(:regime, :bU,  0.000, 2.00,  0.00,  "unskilled UI flow bU"),
        ParamSpec(:regime, :bT,  0.000, 2.00,  0.28,  "training flow bT"),
        ParamSpec(:regime, :bS,  0.000, 2.00,  0.01,  "skilled UI flow bS"),
        ParamSpec(:regime, :α_U, 0.10,  6.00,  1.00,  "unskilled damage shape α_U"),
        ParamSpec(:regime, :a_Γ, 0.30,  6.00,  2.00,  "skilled offer shape a_Γ"),
        ParamSpec(:regime, :b_Γ, 0.30,  6.00,  5.00,  "skilled offer shape b_Γ"),

        # ── UnskilledParams ────────────────────────────────────────────
        ParamSpec(:unsk, :μ,    0.01,  1.50,  0.74,  "unskilled matching eff μ_U"),
        ParamSpec(:unsk, :η,    0.01,  0.90,  0.60,  "unskilled matching elas η_U"),
        ParamSpec(:unsk, :k,    0.001, 1.50,  0.25,  "unskilled vacancy cost k_U"),
        ParamSpec(:unsk, :β,    0.05,  0.95,  0.40,  "unskilled bargaining β_U"),
        ParamSpec(:unsk, :λ,    0.001, 0.30,  0.08,  "unskilled damage rate λ_U"),

        # ── SkilledParams ──────────────────────────────────────────────
        ParamSpec(:skl, :μ,    0.10,  1.50,  0.90,  "skilled matching eff μ_S"),
        ParamSpec(:skl, :η,    0.10,  0.90,  0.50,  "skilled matching elas η_S"),
        ParamSpec(:skl, :k,    0.001, 1.50,  0.17,  "skilled vacancy cost k_S"),
        ParamSpec(:skl, :β,    0.05,  0.95,  0.32,  "skilled bargaining β_S"),
        ParamSpec(:skl, :ξ,    0.0000001, 0.15,  0.03,  "skilled exog. sep rate ξ"),
        ParamSpec(:skl, :λ,    0.001, 0.30,  0.07,  "skilled quality shock λ_S"),
        ParamSpec(:skl, :σ,    0.0000001, 0.20,  0.01,  "OJS flow cost σ"),
    ]
end


# ============================================================
# Parameter transforms: map (lb, ub) → ℝ for unconstrained optimisation
# ============================================================

# logit transform: x ∈ (lb, ub) ↦ t ∈ ℝ
@inline function _to_unconstrained(x::Float64, lb::Float64, ub::Float64)
    p = (x - lb) / (ub - lb)
    p = clamp(p, 1e-8, 1.0 - 1e-8)
    return log(p / (1.0 - p))
end

# inverse logit: t ∈ ℝ ↦ x ∈ (lb, ub)
@inline function _to_constrained(t::Float64, lb::Float64, ub::Float64)
    sig = 1.0 / (1.0 + exp(-t))
    return lb + (ub - lb) * sig
end


# ============================================================
# pack_θ / unpack_θ
# ============================================================

"""
    pack_θ(spec) → Vector{Float64}

Return the initial free-parameter vector in *unconstrained* space.
"""
function pack_θ(spec::SMMSpec) :: Vector{Float64}
    return [_to_unconstrained(ps.init, ps.lb, ps.ub) for ps in spec.free]
end


"""
    unpack_θ(θ_unc, spec) → (CommonParams, RegimeParams, UnskilledParams, SkilledParams)

Convert an unconstrained free-parameter vector back to the four
model structs, merging with any fixed values.
"""
function unpack_θ(
    θ_unc :: AbstractVector{Float64},
    spec  :: SMMSpec
)
    # Step 1: constrained free values
    free_vals = Dict{Symbol, Float64}()
    for (i, ps) in enumerate(spec.free)
        free_vals[ps.name] = _to_constrained(θ_unc[i], ps.lb, ps.ub)
    end

    # Step 2: merge helper — fixed takes priority, then free, then default
    # Supports block-qualified fixed keys (e.g. :unsk_μ, :skl_μ) to
    # disambiguate shared field names across unskilled/skilled blocks.
    function _get(name::Symbol, block::Symbol, default::Float64) :: Float64
        qualified = Symbol(string(block) * "_" * string(name))
        haskey(spec.fixed, qualified)   && return Float64(spec.fixed[qualified])
        haskey(spec.fixed, name)        && return Float64(spec.fixed[name])
        haskey(free_vals, name)         && return free_vals[name]
        return default
    end

    # Step 3: build each struct using defaults from initialise_model()
    #  (defaults are the notebook's calibrated values)
    cp = CommonParams(
        r   = _get(:r,   :common, 0.05),
        ν   = _get(:ν,   :common, 0.05),
        φ   = _get(:φ,   :common, 0.20),
        a_ℓ = _get(:a_ℓ, :common, 2.00),
        b_ℓ = _get(:b_ℓ, :common, 5.00),
        c   = _get(:c,   :common, 1.70),
    )

    rp = RegimeParams(
        PU  = _get(:PU,  :regime, 0.70),
        PS  = _get(:PS,  :regime, 1.85),
        bU  = _get(:bU,  :regime, 0.00),
        bT  = _get(:bT,  :regime, 0.28),
        bS  = _get(:bS,  :regime, 0.01),
        α_U = _get(:α_U, :regime, 1.00),
        a_Γ = _get(:a_Γ, :regime, 2.00),
        b_Γ = _get(:b_Γ, :regime, 5.00),
    )

    up = UnskilledParams(
        μ = _get(:μ,  :unsk, 0.74),   # note: μ and other names shared across blocks
        η = _get(:η,  :unsk, 0.60),   # disambiguation via block-qualified fixed keys
        k = _get(:k,  :unsk, 0.25),   # (e.g. :unsk_μ vs :skl_μ in spec.fixed)
        β = _get(:β,  :unsk, 0.40),
        λ = _get(:λ,  :unsk, 0.08),
    )

    sp = SkilledParams(
        μ = _get(:μ,  :skl, 0.90),
        η = _get(:η,  :skl, 0.50),
        k = _get(:k,  :skl, 0.17),
        β = _get(:β,  :skl, 0.32),
        ξ = _get(:ξ,  :skl, 0.03),
        λ = _get(:λ,  :skl, 0.07),
        σ = _get(:σ,  :skl, 0.01),
    )

    # ── Handle shared field names (μ, η, k, β, λ) ─────────────────────
    # The structs UnskilledParams and SkilledParams share field names.
    # We disambiguate by walking the free list in order and applying each
    # value to the correct block.
    up_fields = Dict{Symbol,Float64}(
        :μ => up.μ, :η => up.η, :k => up.k, :β => up.β, :λ => up.λ
    )
    sp_fields = Dict{Symbol,Float64}(
        :μ => sp.μ, :η => sp.η, :k => sp.k, :β => sp.β,
        :ξ => sp.ξ, :λ => sp.λ, :σ => sp.σ
    )

    for (i, ps) in enumerate(spec.free)
        v = _to_constrained(θ_unc[i], ps.lb, ps.ub)
        if ps.block == :unsk && haskey(up_fields, ps.name)
            up_fields[ps.name] = v
        elseif ps.block == :skl && haskey(sp_fields, ps.name)
            sp_fields[ps.name] = v
        end
    end
    # Apply block-qualified fixed overrides (e.g. :unsk_μ, :skl_η)
    for (nm, val) in pairs(spec.fixed)
        s = string(nm)
        if startswith(s, "unsk_")
            field = Symbol(s[6:end])
            haskey(up_fields, field) && (up_fields[field] = Float64(val))
        elseif startswith(s, "skl_")
            field = Symbol(s[5:end])
            haskey(sp_fields, field) && (sp_fields[field] = Float64(val))
        end
    end

    up = UnskilledParams(
        μ = up_fields[:μ], η = up_fields[:η],
        k = up_fields[:k], β = up_fields[:β],
        λ = up_fields[:λ],
    )
    sp = SkilledParams(
        μ = sp_fields[:μ], η = sp_fields[:η],
        k = sp_fields[:k], β = sp_fields[:β],
        ξ = sp_fields[:ξ], λ = sp_fields[:λ],
        σ = sp_fields[:σ],
    )

    return cp, rp, up, sp
end


# ============================================================
# Printing
# ============================================================

"""
    print_spec(spec)

Display the estimation problem: free parameters with bounds,
fixed overrides, and active moments.
"""
function print_spec(spec::SMMSpec)
    @printf("\n╔══════════════════════════════════════════════════════╗\n")
    @printf("║  SMM Estimation Specification                        ║\n")
    @printf("╠══════════════════════════════════════════════════════╣\n")
    @printf("║  Free parameters (%d)                                 ║\n",
            length(spec.free))
    @printf("╠══════════════════════════════════════════════════════╣\n")
    @printf("  %-6s  %-22s  %8s  %8s  %8s\n",
            "block", "name", "lb", "ub", "init")
    @printf("  %s\n", "─"^56)
    for ps in spec.free
        @printf("  %-6s  %-22s  %8.4f  %8.4f  %8.4f\n",
                ps.block, ps.label, ps.lb, ps.ub, ps.init)
    end

    if length(spec.fixed) > 0
        @printf("\n╠══════════════════════════════════════════════════════╣\n")
        @printf("║  Fixed parameters (%d)                                ║\n",
                length(spec.fixed))
        @printf("╠══════════════════════════════════════════════════════╣\n")
        for (k, v) in pairs(spec.fixed)
            @printf("  %-20s = %.6f\n", k, v)
        end
    end

    # Partition moments into active (weight > 0) and skipped (weight == 0)
    active_moments  = [(k, v) for (k, v) in pairs(spec.moments) if v.weight > 0.0]
    skipped_moments = [(k, v) for (k, v) in pairs(spec.moments) if v.weight <= 0.0]

    @printf("\n╠══════════════════════════════════════════════════════╣\n")
    @printf("║  Active moments (%2d)                                 ║\n", length(active_moments))
    @printf("╠══════════════════════════════════════════════════════╣\n")
    if !isnothing(spec.W)
        @printf("  (Weighting: full optimal W matrix)\n\n")
        @printf("  %-22s  %10s  %12s\n", "moment", "target", "W diag wt")
        @printf("  %s\n", "─"^48)
        for (idx, (k, v)) in enumerate(active_moments)
            w_diag = (idx <= size(spec.W, 1)) ? spec.W[idx, idx] : NaN
            @printf("  %-22s  %10.4f  %12.4e\n", k, v.value, w_diag)
        end
    else
        @printf("  %-22s  %10s  %10s\n", "moment", "target", "weight")
        @printf("  %s\n", "─"^46)
        for (k, v) in active_moments
            @printf("  %-22s  %10.4f  %10.2f\n", k, v.value, v.weight)
        end
    end

    if !isempty(skipped_moments)
        @printf("\n╠══════════════════════════════════════════════════════╣\n")
        @printf("║  Skipped moments (%2d, weight = 0)                   ║\n", length(skipped_moments))
        @printf("╠══════════════════════════════════════════════════════╣\n")
        @printf("  %-22s  %10s\n", "moment", "target")
        @printf("  %s\n", "─"^34)
        for (k, v) in skipped_moments
            @printf("  %-22s  %10.4f\n", k, v.value)
        end
    end
    @printf("\n  Grid: Nx=%d  Np_U=%d  Np_S=%d\n",
            spec.run.Nx, spec.run.Np_U, spec.run.Np_S)
    @printf("  SA:  max_iter=%d  T0=%.2f  step=%.2f  cooling_rate=%.2f  cooling_exp=%.2f  reheat_patience=%d  reheat_factor=%.2f  adapt_window=%d  target_fin=%.2f\n",
            spec.run.sa_max_iter, spec.run.sa_T0, spec.run.sa_step,
            spec.run.sa_cooling_rate, spec.run.sa_cooling_exp,
            spec.run.sa_reheat_patience, spec.run.sa_reheat_factor,
            spec.run.sa_adapt_window, spec.run.sa_target_fin)
    @printf("  DE:   max_iter=%d  pop_size=%s  f=%.2f  cr=%.2f  patience=%d  avg_tol=%s\n",
            spec.run.de_max_iter,
            spec.run.de_pop_size == 0 ? "auto" : string(spec.run.de_pop_size),
            spec.run.de_f, spec.run.de_cr, spec.run.de_patience,
            spec.run.de_avg_tol > 0.0 ? @sprintf("%.1e", spec.run.de_avg_tol) : "off")
    @printf("  NM:   max_iter=%d  f_tol=%.0e  x_tol=%.0e\n",
            spec.run.nm_max_iter, spec.run.nm_f_tol, spec.run.nm_x_tol)
    if spec.run.w_cond_target == 2.0
        @printf("  W:    equal weights (identity, no W matrix)\n")
    elseif !isnothing(spec.W)
        cond_W = cond(spec.W)
        if spec.run.w_cond_target == 0.0
            @printf("  W:    diagonal from moment Σ (cond(W)=%.2e)\n", cond_W)
        elseif spec.run.w_cond_target == 1.0
            @printf("  W:    compressed diagonal log(1 + 1/σ²) (cond(W)=%.2e)\n", cond_W)
        else
            @printf("  W:    optimal weight matrix (cond(W)=%.2e, target κ=%.1e)\n",
                    cond_W, spec.run.w_cond_target)
        end
    else
        @printf("  W:    using diagonal weights from moment variances (target κ=%.1e)\n",
                spec.run.w_cond_target)
    end
    @printf("╚══════════════════════════════════════════════════════╝\n\n")
end
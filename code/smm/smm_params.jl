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
# SMMSpec — full problem specification
# ============================================================

"""
    SMMSpec

Holds everything the SMM loop needs.

Fields
──────
  free        vector of ParamSpec (parameters to estimate)
  fixed       NamedTuple: any field from the four param structs whose
              value is pinned.  Keys are Symbol field names; the SMM
              loop will override model defaults with these values.
  moments     NamedTuple from load_data_moments() — (value, weight) pairs
  sim         SimParams — solver settings, always fixed
  Nx / Np_U / Np_S   grid sizes passed through to the solver
"""
struct SMMSpec
    free    :: Vector{ParamSpec}
    fixed   :: NamedTuple
    moments :: NamedTuple
    sim     :: SimParams
    Nx      :: Int
    Np_U    :: Int
    Np_S    :: Int
end


# ============================================================
# Constructor
# ============================================================

"""
    build_smm_spec(moments, sim;
                   fixed        = NamedTuple(),
                   free_specs   = default_free_params(),
                   Nx = 80, Np_U = 80, Np_S = 80)

Build an `SMMSpec`.

- `fixed`: a NamedTuple of parameter values to hold fixed during
  estimation.  Any field name from CommonParams / RegimeParams /
  UnskilledParams / SkilledParams is valid.
  Example: `fixed = (r = 0.05, ν = 0.05, ξ = 0.03)`

- `free_specs`: vector of ParamSpec.  If a field appears in both
  `fixed` and `free_specs`, the fixed value takes precedence (the
  free entry is silently dropped).

- Grid sizes are set small by default for fast SMM iterations;
  increase for final estimation.
"""
function build_smm_spec(
    moments  :: NamedTuple,
    sim      :: SimParams;
    fixed    :: NamedTuple    = NamedTuple(),
    free_specs :: Vector{ParamSpec} = default_free_params(),
    Nx   :: Int = 80,
    Np_U :: Int = 80,
    Np_S :: Int = 80,
)
    fixed_names = keys(fixed)

    # Drop any free spec whose field appears in `fixed`
    active_free = filter(ps -> !(ps.name in fixed_names), free_specs)

    if length(active_free) < length(free_specs)
        dropped = [ps.name for ps in free_specs if ps.name in fixed_names]
        @printf("SMMSpec: fixed override dropped free params: %s\n",
                join(string.(dropped), ", "))
    end

    return SMMSpec(active_free, fixed, moments, sim, Nx, Np_U, Np_S)
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
        ParamSpec(:common, :r,   0.01,  0.15,  0.05,  "discount rate r"),
        ParamSpec(:common, :ν,   0.01,  0.20,  0.05,  "demographic exit ν"),
        ParamSpec(:common, :φ,   0.05,  0.60,  0.20,  "training completion φ"),
        ParamSpec(:common, :a_ℓ, 0.01,  8.00,  2.00,  "worker type shape a_ℓ"),
        ParamSpec(:common, :b_ℓ, 0.01,  8.00,  5.00,  "worker type shape b_ℓ"),
        ParamSpec(:common, :c,   0.01, 10.00,  1.70,  "training cost coeff c"),

        # ── RegimeParams ───────────────────────────────────────────────
        ParamSpec(:regime, :PU,  0.10,  2.00,  0.70,  "unskilled productivity PU"),
        ParamSpec(:regime, :PS,  0.20,  5.00,  1.85,  "skilled productivity PS"),
        ParamSpec(:regime, :bU,  0.00,  0.80,  0.00,  "unskilled UI flow bU"),
        ParamSpec(:regime, :bT,  0.00,  0.80,  0.28,  "training flow bT"),
        ParamSpec(:regime, :bS,  0.00,  0.80,  0.01,  "skilled UI flow bS"),
        ParamSpec(:regime, :α_U, 0.05,  5.00,  1.00,  "unskilled damage shape α_U"),
        ParamSpec(:regime, :a_Γ, 0.01,  8.00,  2.00,  "skilled offer shape a_Γ"),
        ParamSpec(:regime, :b_Γ, 0.01,  8.00,  5.00,  "skilled offer shape b_Γ"),

        # ── UnskilledParams ────────────────────────────────────────────
        ParamSpec(:unsk, :μ,  0.10,  2.00,  0.74,  "unskilled matching eff μ_U"),
        ParamSpec(:unsk, :η,  0.10,  0.90,  0.60,  "unskilled matching elas η_U"),
        ParamSpec(:unsk, :k,  0.01,  2.00,  0.25,  "unskilled vacancy cost k_U"),
        ParamSpec(:unsk, :β,  0.01,  0.90,  0.40,  "unskilled bargaining β_U"),
        ParamSpec(:unsk, :λ,  0.001,  0.50,  0.08,  "unskilled damage rate λ_U"),

        # ── SkilledParams ──────────────────────────────────────────────
        ParamSpec(:skl, :μ,  0.10,  2.00,  0.90,  "skilled matching eff μ_S"),
        ParamSpec(:skl, :η,  0.10,  0.90,  0.50,  "skilled matching elas η_S"),
        ParamSpec(:skl, :k,  0.01,  2.00,  0.17,  "skilled vacancy cost k_S"),
        ParamSpec(:skl, :β,  0.01,  0.90,  0.32,  "skilled bargaining β_S"),
        ParamSpec(:skl, :ξ,  0.01, 0.20,  0.03,  "skilled exog. sep rate ξ"),
        ParamSpec(:skl, :λ,  0.01,  0.50,  0.07,  "skilled quality shock λ_S"),
        ParamSpec(:skl, :σ,  0.001, 0.15,  0.01,  "OJS flow cost σ"),
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
    function _get(name::Symbol, default::Float64) :: Float64
        haskey(spec.fixed, name)    && return Float64(spec.fixed[name])
        haskey(free_vals, name)     && return free_vals[name]
        return default
    end

    # Step 3: build each struct using defaults from initialise_model()
    #  (defaults are the notebook's calibrated values)
    cp = CommonParams(
        r   = _get(:r,   0.05),
        ν   = _get(:ν,   0.05),
        φ   = _get(:φ,   0.20),
        a_ℓ = _get(:a_ℓ, 2.00),
        b_ℓ = _get(:b_ℓ, 5.00),
        c   = _get(:c,   1.70),
    )

    rp = RegimeParams(
        PU  = _get(:PU,  0.70),
        PS  = _get(:PS,  1.85),
        bU  = _get(:bU,  0.00),
        bT  = _get(:bT,  0.28),
        bS  = _get(:bS,  0.01),
        α_U = _get(:α_U, 1.00),
        a_Γ = _get(:a_Γ, 2.00),
        b_Γ = _get(:b_Γ, 5.00),
    )

    up = UnskilledParams(
        μ = _get(:μ,  0.74),   # note: μ and other names shared across blocks
        η = _get(:η,  0.60),   # disambiguation handled by block in ParamSpec,
        k = _get(:k,  0.25),   # but _get looks up by name only.
        β = _get(:β,  0.40),   # For shared names (μ,η,k,β,λ), the free_vals
        λ = _get(:λ,  0.08),   # dict holds the *last* block's value — see note.
    )

    sp = SkilledParams(
        μ = _get(:μ,  0.90),
        η = _get(:η,  0.50),
        k = _get(:k,  0.17),
        β = _get(:β,  0.32),
        ξ = _get(:ξ,  0.03),
        λ = _get(:λ,  0.07),
        σ = _get(:σ,  0.01),
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
    # Also apply fixed overrides per block
    for (nm, val) in pairs(spec.fixed)
        # We cannot know the block from the fixed NamedTuple alone,
        # so fixed params with shared names apply to BOTH blocks unless
        # those names also appear in free (already handled above).
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

    @printf("\n╠══════════════════════════════════════════════════════╣\n")
    @printf("║  Active moments                                      ║\n")
    @printf("╠══════════════════════════════════════════════════════╣\n")
    @printf("  %-22s  %10s  %10s\n", "moment", "target", "weight")
    @printf("  %s\n", "─"^46)
    for (k, v) in pairs(spec.moments)
        v.weight > 0.0 || continue
        @printf("  %-22s  %10.4f  %10.2f\n", k, v.value, v.weight)
    end
    @printf("\n  Grid: Nx=%d  Np_U=%d  Np_S=%d\n", spec.Nx, spec.Np_U, spec.Np_S)
    @printf("╚══════════════════════════════════════════════════════╝\n\n")
end

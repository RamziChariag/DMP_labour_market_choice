############################################################
# smm_params.jl — SMM parameter specification
#
# Types
#   ParamSpec      metadata for a single free parameter
#   SMMRunParams   runtime settings (grids, DE / SA / NM, tracing)
#   SMMSpec        full estimation problem
#
# Functions
#   build_smm_spec(...)    construct an SMMSpec
#   default_free_params()  full list of estimable parameters
#   pack_θ(spec)           initial values → unconstrained vector
#   unpack_θ(θ, spec)      unconstrained vector → 3 model structs
#   print_spec(spec)       display the estimation problem
#
# Parameter classification (Model Notes §2 / data_and_moments.pdf §18)
#
#   Externally set, always fixed at SMM runtime:
#     r       calibrated to 4% annual real risk-free rate (monthly r = 0.05/12)
#     ν       pre-estimated from CPS life-table (one row per crisis pair)
#     φ       pre-estimated from NSC/IPEDS (pooled across Fall semesters)
#
#   Deep structural (estimated on base_fc, then HELD FIXED at the baseline
#   estimates during crisis re-estimation):
#     common:  a_ℓ, b_ℓ, ρ_x     ability-distribution primitives (aU marginal
#                                  shapes + concordance correlation)
#     unsk:    bU, bT             unskilled / training institutional flow values
#     skl:     bS                 skilled outside flow value
#
#   Regime-specific (re-estimated within EACH window):
#     common:  c                              training cost coefficient
#     unsk:    PU, α_U                         unskilled productivity / damage shape
#     skl:     PS, a_Γ, b_Γ              skilled productivity / offer shape
#     unsk:    μ, η, k, β, λ, ξ               matching, vacancy cost,
#                                              bargaining, damage hazard,
#                                              exogenous separation ξ_U
#     skl:     μ, η, k, β, λ, σ, ξ, δ         same + OJS flow cost
#                                              + exogenous separation ξ_S
#                                              + offer/shock support ratio δ_S
#
#   Production is linear in own ability under pure-Roy (π_j = exp(A)·P_j·a_j·p),
#   so there is no ability gradient to estimate: the single-index γ_U / γ_S of
#   the old model are gone, replaced by the ρ_x separation device.
#
# Note on ξ_U / ξ_S: each of UnskilledParams / SkilledParams carries an
# exogenous-separation field ξ, defaulting to 0.0. Total separation into
# unemployment is ξ_U + λ_U · G(p*_U) (unskilled) and ξ_S + λ_S · Γ(p*_S)
# (skilled). Pinning a ξ to 0 (FIX_PARAMS :unsk_xi / :skl_xi => 0.0, or
# dropping it from the free list) recovers the purely-endogenous variant
# for that sector exactly.
############################################################


# ============================================================
# ParamSpec
# ============================================================

"""
    ParamSpec

Metadata for one free parameter.

Fields
  block   :: Symbol   which struct (:common, :unsk, :skl)
  name    :: Symbol   field name within that struct
  lb      :: Float64  lower bound (used for the logit transform)
  ub      :: Float64  upper bound
  init    :: Float64  starting value (constrained space)
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
# SMMRunParams
# ============================================================

"""
    SMMRunParams

Runtime settings for the SMM estimation — grid sizes, DE, SA, NM,
weighting and tracing options.  Construct with keyword arguments;
all have defaults.
"""
Base.@kwdef struct SMMRunParams
    # Grids (real SA / DE / NM runs)
    Nx      :: Int     = 80
    Np_U    :: Int     = 80
    Np_S    :: Int     = 80

    # Candidate-generation layer (coarse grid, Sobol sample, hierarchical clustering)
    cand_Nx          :: Int = 40
    cand_Np_U        :: Int = 40
    cand_Np_S        :: Int = 40
    cand_n_sample    :: Int = 2000
    cand_seed        :: Int = 20240601
    cand_min_cluster :: Int = 5

    # Differential evolution
    de_max_iter  :: Int     = 200
    de_pop_size  :: Int     = 0        # 0 ⇒ 10 × n_free_params
    de_f         :: Float64 = 0.65
    de_cr        :: Float64 = 0.85
    de_patience  :: Int     = 20
    de_avg_tol   :: Float64 = 0.01     # stop when (Q_mean−Q_best)/|Q_best| < tol; 0 disables

    # Simulated annealing
    sa_max_iter      :: Int     = 5_000
    sa_T0            :: Float64 = 2.0
    sa_step          :: Float64 = 0.15
    sa_cooling_rate  :: Float64 = 1.0
    sa_cooling_exp   :: Float64 = 0.5
    sa_reheat_patience :: Int     = 200
    sa_reheat_factor   :: Float64 = 2.0
    sa_max_reheats     :: Int     = 5
    sa_adapt_window  :: Int     = 50
    sa_target_fin    :: Float64 = 0.90
    sa_random_init   :: Bool    = false
    sa_parallel_steps :: Int    = 100         # parallel SA steps before pruning to the best chain
    sa_seed           :: Int    = 20240601    # base seed for parallel SA chains (replicability)

    # Weight-matrix mode (see load_weight_matrix)
    w_cond_target :: Float64 = 1e8

    # Wage-reliability calibration knob. σ_w is NOT a free parameter: it is
    # calibrated externally from λ_w via σ_{w,j}² = (1−λ_w)·Var̂[log w_j]
    # (calibrate_sigma_w), so λ_w configures the run and is reported in the
    # header only — it never appears in the parameter tables. Bound–Krueger
    # (1991) give λ_w ≈ 0.82 for log annual earnings.
    λ_w :: Float64 = 0.82

    # Nelder-Mead polish
    nm_max_iter  :: Int     = 5_000
    nm_f_tol     :: Float64 = 1e-6
    nm_x_tol     :: Float64 = 1e-5
    nm_g_tol     :: Float64 = 1e-5   # gradient tolerance (unconstrained space)
    nm_no_improve :: Int    = 0      # early-stop: halt NM after this many objective
                                     # evaluations with no improvement in best Q
                                     # (0 disables; same units as the [iter N] trace)

    # Tracing
    show_trace_members     :: Bool = false
    show_trace_generations :: Bool = true
    trace_stride           :: Int  = 10
end


# ============================================================
# SeedBank — output of the candidate-generation layer
# ============================================================

"""
    SeedBank

The product of the candidate-generation layer (candidates.jl): a bank of
feasible candidate parameter vectors (unconstrained space), their coarse-grid
Q values, and hierarchical-clustering cluster labels (noise already dropped).  Passed as a
single object into `run_smm` and forwarded to the SA / DE seeders.

Fields
  candidates :: Vector{Vector{Float64}}   feasible θ (unconstrained)
  Q          :: Vector{Float64}           coarse-grid Q for each candidate
  labels     :: Vector{Int}               cluster id (>= 1; noise removed)
  meta       :: NamedTuple                staleness key (see candidates.jl)
"""
struct SeedBank
    candidates :: Vector{Vector{Float64}}
    Q          :: Vector{Float64}
    labels     :: Vector{Int}
    meta       :: NamedTuple
end


# ============================================================
# SMMSpec
# ============================================================

"""
    SMMSpec

Holds everything the SMM estimation needs.

Fields
  free     vector of ParamSpec (parameters to estimate)
  fixed    NamedTuple of pinned parameter values
  moments  NamedTuple from load_data_moments() — (value, weight) pairs
  sim      SimParams — solver settings
  run      SMMRunParams — grid sizes, optimiser settings, weighting
  W        Matrix — weight matrix.  Never `nothing` after construction:
           the equal-weight scheme is stored as the diagonal matrix
           Diagonal(weight_k / m̂_k²), so every scheme shares one
           weighted-loss path (see compute_loss_matrix).
  q_scale  Float64 — DISPLAY-ONLY positive constant dividing the
           reported scalar Q (e.g. tr(Σ̂) for matrix schemes, 1.0 for
           equal weight).  Constant in θ, so it does not move the argmin.
"""
struct SMMSpec
    free    :: Vector{ParamSpec}
    fixed   :: NamedTuple
    moments :: NamedTuple
    sim     :: SimParams
    run     :: SMMRunParams
    W       :: Matrix{Float64}
    q_scale :: Float64
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

- `fixed` is a NamedTuple of parameter values to pin.  Keys may be
  plain names (e.g. `:r`) or block-qualified (e.g. `:unsk_μ`, `:skl_μ`)
  to disambiguate shared field names across the unskilled and
  skilled blocks.  Any free spec whose name matches an entry in
  `fixed` is silently dropped from the free list.

- `skip_moments` is a vector of moment Symbols to deactivate.  Their
  weights are zeroed in the stored moments NamedTuple so that
  `compute_loss_matrix` skips them automatically.  Unknown names
  produce a warning.

- `W` selects the weighting scheme.  If `W === nothing` (equal
  weights), the constructor builds the diagonal matrix
  Diagonal(weight_k / m̂_k²) over the active moments, so that
  g' W g = Σ_k weight_k · (g_k / m̂_k)² is the scale-normalised
  relative-deviation loss.  If a matrix is passed (diagonal-σ or full
  Σ̂⁻¹ from `load_weight_matrix`) it must be square with size equal to
  the number of active moments; a mismatch raises an error — typically
  a sign that `W` was built with a different `skip_moments` list.

- `q_scale` is a DISPLAY-ONLY positive constant that divides the
  reported scalar Q (pass `tr(Σ̂)` for the matrix schemes to bring the
  number into a readable range; leave at 1.0 for equal weight).  It is
  constant in θ and therefore does not move the argmin.
"""
function build_smm_spec(
    moments      :: NamedTuple,
    sim          :: SimParams;
    fixed        :: Any                           = (;),
    free_specs   :: Vector{ParamSpec}             = default_free_params(),
    run          :: SMMRunParams                  = SMMRunParams(),
    W            :: Union{Nothing, Matrix{Float64}} = nothing,
    q_scale      :: Float64                       = 1.0,
    skip_moments :: Vector{Symbol}                = Symbol[],
)
    fixed_nt    = (fixed isa NamedTuple) ? fixed : NamedTuple()
    fixed_names = keys(fixed_nt)

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

    # Resolve the weight matrix.  Active moments are those with
    # weight > 0, taken in `keys(moments)` order — the SAME order
    # compute_loss_matrix uses to assemble the deviation vector, so the
    # diagonal lines up element-for-element.
    active_keys = [k for k in keys(moments) if moments[k].weight > 0.0]
    K_active    = length(active_keys)

    W_final::Matrix{Float64} = if isnothing(W)
        # Equal-weight scheme as a diagonal matrix:
        #   W_kk = weight_k / m̂_k²   ⟹   g' W g = Σ_k weight_k (g_k/m̂_k)²
        d = Vector{Float64}(undef, K_active)
        for (i, k) in enumerate(active_keys)
            t      = moments[k]
            scale2 = max(abs(t.value), 1e-10)^2
            d[i]   = t.weight / scale2
        end
        Matrix(Diagonal(d))
    else
        if size(W, 1) != K_active || size(W, 2) != K_active
            error(
                "W matrix size $(size(W,1))×$(size(W,2)) does not match the number of " *
                "active moments ($K_active). Rebuild spec via build_smm_spec with the " *
                "correct W from load_weight_matrix(..., skip_moments=SKIP_MOMENTS)."
            )
        end
        W
    end

    q_scale > 0.0 || error("build_smm_spec: q_scale must be positive (got $q_scale).")

    return SMMSpec(active_free, fixed_nt, moments, sim, run, W_final, q_scale)
end


# ============================================================
# σ_w calibration from the wage-reliability knob λ_w
# ============================================================

"""
    calibrate_sigma_w(λ_w, moments) → (σ_wU, σ_wS)

Calibrate the log-wage measurement-error SDs from the reliability ratio
λ_w via the classical-error decomposition σ_{w,j}² = (1−λ_w)·Var̂[log w_j],
mirroring the external calibration of ν and φ. `moments` is the data
NamedTuple from `load_data_moments`; the empirical log-wage variances are
its `emp_var_U` / `emp_var_S` targets. The returned pair is injected into
the pinned (calibrated) block exactly like r, ν, φ.
"""
function calibrate_sigma_w(λ_w::Real, moments::NamedTuple)
    σ_wU = sqrt(max((1.0 - λ_w) * moments[:emp_var_U].value, 0.0))
    σ_wS = sqrt(max((1.0 - λ_w) * moments[:emp_var_S].value, 0.0))
    return (σ_wU, σ_wS)
end


# ============================================================
# Default free parameter list
# ============================================================

"""
    default_free_params() → Vector{ParamSpec}

Full list of estimable parameters with default bounds and starting
values.  Pass a subset to `build_smm_spec` to estimate only some.

Excluded from this list because they are always fixed:
  r       calibrated externally at 4% annual → monthly r = 0.05/12
  ν       pre-estimated from CPS data (one value per crisis pair)
  φ       pre-estimated from NSC/IPEDS data (pooled)
"""
function default_free_params() :: Vector{ParamSpec}
    return [
        # Deep structural — common block.  a_ℓ, b_ℓ shape the aU marginal;
        # ρ_x is the ability correlation, the RoySearch separation device
        # (Gola concordance).  All three are population primitives: estimated
        # on base_fc, then held fixed across the business cycle.
        ParamSpec(:common, :a_ℓ,        0.1000,   8.0000,   2.8000, "worker type shape a_ℓ"),
        ParamSpec(:common, :b_ℓ,        0.0500,   4.0000,   1.2000, "worker type shape b_ℓ"),
        # ρ_x ∈ (−1,1): bounds shy of ±1 so the Gaussian copula ζ-map stays
        # finite.  init −0.55 is the identification-validated point (negative
        # concordance rescues the unskilled market at fixed productivity).
        ParamSpec(:common, :ρ_x,       -0.9900,   0.9900,  -0.5500, "ability correlation ρ_x"),

        # Common block — training cost coeff and aggregate scale
        ParamSpec(:common, :c,          0.0000,  15.0000,  11.0000, "training cost coeff c"),
        ParamSpec(:common, :A,          0.0000,  13.0000,   5.6000, "aggregate production scale A"),
 
        # Institutional flow values (stored by consuming block).
        ParamSpec(:unsk,   :bU,         0.0000,   2.5000,   1.5000, "unskilled outside flow b_U"),
        ParamSpec(:unsk,   :bT,         0.0000,   7.0000,   2.6500, "training flow b_T"),
        ParamSpec(:skl,    :bS,         0.0000,   2.0000,   0.6600, "skilled outside flow b_S"),
 
        # Productivity levels.  Under pure-Roy production is linear in own
        # ability, π_U = exp(A) P_U aU p and π_S = exp(A) P_S aS p (no
        # ability gradient — the redesign replaced the exp(γx) map with the
        # ρ_x separation device).  Only exp(A)·P_j is pinned by wage levels,
        # so (A, P_U, P_S) share one flat direction; fix A (e.g. A=0) or one
        # P to identify them individually.  The premium loads on log(P_S/P_U).
        ParamSpec(:unsk,   :PU,         0.0001,   8.0000,   1.8000, "unskilled productivity P_U"),
        ParamSpec(:skl,    :PS,         0.0001,  20.0000,   4.5000, "skilled productivity P_S"),
        ParamSpec(:unsk,   :α_U,        0.2000,  6.0000,   1.0000, "unskilled damage shape α_U"),
        ParamSpec(:skl,    :a_Γ,        0.1000,  12.0000,   5.2000, "skilled offer shape a_Γ"),
        ParamSpec(:skl,    :b_Γ,        0.1000,  10.0000,   9.2000, "skilled offer shape b_Γ"),
        ParamSpec(:skl,    :δ,          0.0500,   1.0000,   0.6000, "shock/offer support ratio δ_S"),
 
        # Matching efficiency, matching elasticity, bargaining (U/S paired)
        ParamSpec(:unsk,   :μ,          0.0001,   2.0000,   0.3500, "unskilled matching eff μ_U"),
        ParamSpec(:skl,    :μ,          0.0001,   2.0000,   0.1900, "skilled matching eff μ_S"),
        ParamSpec(:unsk,   :η,          0.0001,   1.0000,   0.5000, "unskilled matching elas η_U"),
        ParamSpec(:skl,    :η,          0.0001,   1.0000,   0.5000, "skilled matching elas η_S"),
        ParamSpec(:unsk,   :β,          0.0010,   0.9000,   0.5000, "unskilled bargaining β_U"),
        ParamSpec(:skl,    :β,          0.0001,   0.9000,   0.5000, "skilled bargaining β_S"),
 
        # Shock arrival rates (U/S paired)
        # λ_U init lowered 0.80 → 0.10: sep_rate_U ≈ λ_U·G(p*) ≈ 0.013/mo wants a
        # SMALL λ_U; starting high pushes the chain into the tightness-breaking
        # region that drove λ_U → 0 in prior runs.  Bounds unchanged.
        ParamSpec(:unsk,   :λ,          0.0001,   0.95000,   0.1000, "unskilled damage rate λ_U"),
        ParamSpec(:skl,    :λ,          0.0001,   0.20000,   0.0400, "skilled quality shock λ_S"),

        # Vacancy costs (U/S paired) — DIMENSIONLESS, in months of average
        # sectoral output: dollar posting cost = k_j · π̄_j (see grids.jl,
        # mean_output_U/S).  LMR (2016) estimate 2.34 (HS) / 1.58 (college)
        # in the same units.
        ParamSpec(:unsk,   :k,          0.0005,  12.0000,   2.2500, "unskilled vacancy cost k_U (months of avg U output)"),
        ParamSpec(:skl,    :k,          0.0005,  12.0000,   4.6000, "skilled vacancy cost k_S (months of avg S output)"),
 
        # Skilled block — OJS cost, exogenous separation, offer/shock support ratio.
        # δ_S compresses the λ_S-shock redraw onto [0,δ_S]; it carries endogenous
        # separation and the EE ladder (both dead at δ_S = 1, the single-distribution
        # limit).  ξ_S is the exogenous baseline hazard; (ξ_S, δ_S, λ_S) jointly
        # split sep_rate_S / ee_rate_S into baseline and endogenous margins.
        # Exogenous separation baseline ξ_U — a quality-independent hazard on
        # top of the endogenous margin λ_U·G(p*).  Mirrors the skilled ξ_S
        # below (same bounds); gives sep_rate_U a live lever when p*_U → 0
        # collapses the endogenous part.
        ParamSpec(:skl,    :σ,          0.0000,   1.5000,   0.0900, "OJS flow cost σ_S"),
        ParamSpec(:unsk,   :ξ,          0.0000,   0.0500,   0.0000, "unskilled exogenous separation ξ_U"),
        ParamSpec(:skl,    :ξ,          0.0000,   0.0500,   0.0000, "skilled exogenous separation ξ_S"),
 
        # Wage measurement error (per sector; pinned when λ_w > 0)
        ParamSpec(:unsk,   :σ_w,        0.0000,   1.0000,   0.6000, "unskilled wage meas. error σ_wU"),
        ParamSpec(:skl,    :σ_w,        0.0000,   1.0000,   0.6000, "skilled wage meas. error σ_wS"),
    ]
end


# Regime-specific (block, name) pairs.  Used by smm_main to decide
# which parameters to free in a crisis re-estimation (deep parameters
# get fixed at the baseline estimate; regime-specific ones stay free).
const REGIME_SPECIFIC_PARAMS = Set([
    (:common, :c),   (:common, :A),
    (:unsk,   :PU),  (:skl,  :PS),
    (:unsk,   :α_U), (:skl,  :a_Γ),   (:skl,  :b_Γ),
    (:unsk,   :μ),   (:unsk, :η),   (:unsk, :k),  (:unsk, :β),  (:unsk, :λ),  (:unsk, :ξ),  (:unsk, :σ_w),
    (:skl,    :μ),   (:skl, :η),    (:skl, :k),   (:skl, :β),   (:skl, :λ),
    (:skl,    :σ),   (:skl, :ξ),    (:skl, :δ),   (:skl, :σ_w),
])

# Convenience: the complement is the deep set (everything in
# default_free_params that is NOT regime-specific).  Useful for
# pinning baseline-derived deep parameters in a crisis run.  The
# ability-distribution primitives (a_ℓ, b_ℓ, ρ_x) are deep: population
# structure does not switch at the business-cycle frequency.
const DEEP_PARAMS = Set([
    (:common, :a_ℓ), (:common, :b_ℓ), (:common, :ρ_x),
    (:unsk,   :bU),  (:unsk,  :bT),  (:skl,  :bS),
])


# ============================================================
# Bounds-aware transforms:  (lb, ub) ↔ ℝ
# ============================================================

@inline function _to_unconstrained(x::Float64, lb::Float64, ub::Float64)
    p = (x - lb) / (ub - lb)
    p = clamp(p, 1e-8, 1.0 - 1e-8)
    return log(p / (1.0 - p))
end

@inline function _to_constrained(t::Float64, lb::Float64, ub::Float64)
    sig = 1.0 / (1.0 + exp(-t))
    return lb + (ub - lb) * sig
end


# ============================================================
# pack_θ / unpack_θ
# ============================================================

"""
    pack_θ(spec) → Vector{Float64}

Return the initial free-parameter vector in unconstrained space.
"""
function pack_θ(spec::SMMSpec) :: Vector{Float64}
    return [_to_unconstrained(ps.init, ps.lb, ps.ub) for ps in spec.free]
end


"""
    unpack_θ(θ_unc, spec) → (CommonParams, UnskilledParams, SkilledParams)

Convert an unconstrained free-parameter vector back to the three
model structs, merging with any fixed values.  PU, bU, bT and α_U live
in `UnskilledParams`; PS, bS, a_Γ and b_Γ live in `SkilledParams`.
"""
function unpack_θ(
    θ_unc :: AbstractVector{Float64},
    spec  :: SMMSpec
)
    # 1. Constrained free values (keyed by bare name)
    free_vals = Dict{Symbol, Float64}()
    for (i, ps) in enumerate(spec.free)
        free_vals[ps.name] = _to_constrained(θ_unc[i], ps.lb, ps.ub)
    end

    # 2. Merge helper.  Fixed takes priority, then free, then default.
    #    Supports block-qualified fixed keys (e.g. :unsk_μ, :skl_μ).
    function _get(name::Symbol, block::Symbol, default::Float64) :: Float64
        qualified = Symbol(string(block) * "_" * string(name))
        haskey(spec.fixed, qualified)   && return Float64(spec.fixed[qualified])
        haskey(spec.fixed, name)        && return Float64(spec.fixed[name])
        haskey(free_vals, name)         && return free_vals[name]
        return default
    end

    cp = CommonParams(
        r   = _get(:r,   :common, 0.05),
        ν   = _get(:ν,   :common, 0.05),
        φ   = _get(:φ,   :common, 0.20),
        a_ℓ = _get(:a_ℓ, :common, 2.00),
        b_ℓ = _get(:b_ℓ, :common, 5.00),
        ρ_x = _get(:ρ_x, :common, -0.55),
        c   = _get(:c,   :common, 1.70),
        A   = _get(:A,   :common, 0.00),   # log scale; model uses exp(A)
    )

    # First-pass build.  The names unique to one block (PU, bU, bT, α_U /
    # PS, bS, a_Γ, b_Γ) are resolved correctly here because free_vals is
    # keyed by bare name and these names are unique across blocks.
    up = UnskilledParams(
        μ   = _get(:μ,   :unsk, 0.74),
        η   = _get(:η,   :unsk, 0.60),
        k   = _get(:k,   :unsk, 0.25),
        β   = _get(:β,   :unsk, 0.40),
        λ   = _get(:λ,   :unsk, 0.08),
        PU  = _get(:PU,  :unsk, 0.70),
        bU  = _get(:bU,  :unsk, 0.00),
        bT  = _get(:bT,  :unsk, 0.28),
        α_U = _get(:α_U, :unsk, 1.00),
        ξ   = _get(:ξ,   :unsk, 0.0),
        σ_w = _get(:σ_w, :unsk, 0.0),
    )

    sp = SkilledParams(
        μ        = _get(:μ,        :skl, 0.90),
        η        = _get(:η,        :skl, 0.50),
        k        = _get(:k,        :skl, 0.17),
        β        = _get(:β,        :skl, 0.32),
        λ        = _get(:λ,        :skl, 0.07),
        σ        = _get(:σ,        :skl, 0.01),
        PS       = _get(:PS,       :skl, 1.85),
        bS       = _get(:bS,       :skl, 0.01),
        a_Γ      = _get(:a_Γ,      :skl, 2.00),
        b_Γ      = _get(:b_Γ,      :skl, 5.00),
        δ        = _get(:δ,        :skl, 1.0),
        ξ        = _get(:ξ,        :skl, 0.0),
        σ_w      = _get(:σ_w,      :skl, 0.0),
    )

    # 3. Disambiguate the SHARED field names (μ, η, k, β, λ, and σ) across the
    #    unskilled and skilled blocks: free_vals is keyed by bare name, so the
    #    first pass cannot tell :unsk_μ from :skl_μ.  The eight block-unique
    #    names are already correct above and are carried through unchanged by
    #    seeding the dicts with the full first-pass structs.
    up_fields = Dict{Symbol,Float64}(
        :μ => up.μ, :η => up.η, :k => up.k, :β => up.β, :λ => up.λ,
        :PU => up.PU, :bU => up.bU, :bT => up.bT, :α_U => up.α_U,
        :ξ => up.ξ, :σ_w => up.σ_w,
    )
    sp_fields = Dict{Symbol,Float64}(
        :μ => sp.μ, :η => sp.η, :k => sp.k, :β => sp.β, :λ => sp.λ, :σ => sp.σ,
        :PS => sp.PS, :bS => sp.bS, :a_Γ => sp.a_Γ, :b_Γ => sp.b_Γ,
        :δ => sp.δ, :ξ => sp.ξ, :σ_w => sp.σ_w,
    )

    for (i, ps) in enumerate(spec.free)
        v = _to_constrained(θ_unc[i], ps.lb, ps.ub)
        if ps.block == :unsk && haskey(up_fields, ps.name)
            up_fields[ps.name] = v
        elseif ps.block == :skl && haskey(sp_fields, ps.name)
            sp_fields[ps.name] = v
        end
    end

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
        PU = up_fields[:PU], bU = up_fields[:bU],
        bT = up_fields[:bT], α_U = up_fields[:α_U],
        ξ = up_fields[:ξ], σ_w = up_fields[:σ_w],
    )
    sp = SkilledParams(
        μ = sp_fields[:μ], η = sp_fields[:η],
        k = sp_fields[:k], β = sp_fields[:β],
        λ = sp_fields[:λ], σ = sp_fields[:σ],
        PS = sp_fields[:PS], bS = sp_fields[:bS],
        a_Γ = sp_fields[:a_Γ], b_Γ = sp_fields[:b_Γ],
        δ = sp_fields[:δ], ξ = sp_fields[:ξ], σ_w = sp_fields[:σ_w],
    )

    return cp, up, sp
end


# ============================================================
# Printing
# ============================================================

"""
    print_spec(spec)

Display the estimation problem: free parameters with bounds, fixed
overrides, and active / skipped moments.
"""
function print_spec(spec::SMMSpec)
    @printf("\n╔══════════════════════════════════════════════════════╗\n")
    @printf("║  SMM Estimation Specification                        ║\n")
    @printf("╠══════════════════════════════════════════════════════╣\n")
    @printf("║  Free parameters (%d)                                ║\n",
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

    active_moments  = [(k, v) for (k, v) in pairs(spec.moments) if v.weight > 0.0]
    skipped_moments = [(k, v) for (k, v) in pairs(spec.moments) if v.weight <= 0.0]

    @printf("\n╠══════════════════════════════════════════════════════╣\n")
    @printf("║  Active moments (%2d)                                 ║\n", length(active_moments))
    @printf("╠══════════════════════════════════════════════════════╣\n")
    is_equal_weight = (spec.run.w_cond_target == 2.0)
    if is_equal_weight
        @printf("  (Weighting: equal weights — W = Diagonal(weight/m̂²), relative-deviation loss)\n")
        @printf("  (Q divided by q_scale = %.6e for display)\n\n", spec.q_scale)
        @printf("  %-22s  %10s  %10s\n", "moment", "target", "weight")
        @printf("  %s\n", "─"^46)
        for (k, v) in active_moments
            @printf("  %-22s  %10.4f  %10.2f\n", k, v.value, v.weight)
        end
    else
        @printf("  (Weighting: matrix W — raw deviations g'Wg)\n")
        @printf("  (Q divided by q_scale = %.6e for display)\n\n", spec.q_scale)
        @printf("  %-22s  %10s  %12s\n", "moment", "target", "W diag wt")
        @printf("  %s\n", "─"^48)
        for (idx, (k, v)) in enumerate(active_moments)
            w_diag = (idx <= size(spec.W, 1)) ? spec.W[idx, idx] : NaN
            @printf("  %-22s  %10.4f  %12.4e\n", k, v.value, w_diag)
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
    @printf("  λ_w (wage-reliability calibration knob, σ_w = √((1−λ_w)·Var̂[log w])): %.4f\n",
            spec.run.λ_w)
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
    @printf("  NM:   max_iter=%d  f_tol=%.0e  x_tol=%.0e  g_tol=%.0e  no_improve=%s\n",
            spec.run.nm_max_iter, spec.run.nm_f_tol, spec.run.nm_x_tol, spec.run.nm_g_tol,
            spec.run.nm_no_improve > 0 ? string(spec.run.nm_no_improve) : "off")
    if spec.run.w_cond_target == 2.0
        @printf("  W:    equal weights — Diagonal(weight/m̂²) (cond(W)=%.2e)\n", cond(spec.W))
    else
        cond_W = cond(spec.W)
        if spec.run.w_cond_target == 0.0
            @printf("  W:    diagonal from moment Σ (cond(W)=%.2e)\n", cond_W)
        elseif spec.run.w_cond_target == 1.0
            @printf("  W:    compressed diagonal log(1 + 1/σ²) (cond(W)=%.2e)\n", cond_W)
        else
            @printf("  W:    optimal weight matrix (cond(W)=%.2e, target κ=%.1e)\n",
                    cond_W, spec.run.w_cond_target)
        end
    end
    @printf("  Q scale (display-only divisor): %.6e\n", spec.q_scale)
    @printf("╚══════════════════════════════════════════════════════╝\n\n")
end
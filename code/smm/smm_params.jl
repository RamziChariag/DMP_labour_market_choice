############################################################
# smm/smm_params.jl — the estimation problem: parameters, specs, provenance
#
# Types: ParamSpec (one free parameter), SMMRunParams (run settings), RunProvenance,
# SeedBank, SMMSpec (the full problem). Serialised bundles store these structs, which
# the deserialiser fills by field POSITION: a field may not be added, removed or
# reordered without migrating every stored bundle.
#
# Parameters. r, ν, φ are calibrated externally and σ_wU, σ_wS are calibrated from λ_w;
# every other parameter is estimated unless pinned via FIX_PARAMS. The baseline
# estimates its two periods jointly, BASELINE_COMMON shared and every other parameter
# taking one value per period; a crisis window holds CRISIS_HELD at its baseline
# period's estimate. Stored units: A = ln A, b_j and σ_S over A, c = ln c, θ_j in
# vacancies per searcher, k_j in months of mean new-match output.
#
# Entry points: default_free_params, build_smm_spec, joint_spec, period_list,
# pack_θ, unpack_θ, assert_all_params_accounted, print_spec.
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
  lb      :: Float64  lower bound of the logistic box
  ub      :: Float64  upper bound
  init    :: Float64  starting value (constrained); NaN until seeded
  label   :: String   human-readable label; its last word is the display symbol

The parameter table is built unseeded; the estimation driver writes the seed in, from
DEFAULT_PARAMS (smm_main.jl) or a stored optimum, and `pack_θ` refuses an unseeded entry.
"""
struct ParamSpec
    block :: Symbol
    name  :: Symbol
    lb    :: Float64
    ub    :: Float64
    init  :: Float64
    label :: String
end

"""
    ParamSpec(block, name, lb, ub, label) → ParamSpec

An unseeded entry (`init = NaN`).
"""
ParamSpec(block::Symbol, name::Symbol, lb::Real, ub::Real, label::AbstractString) =
    ParamSpec(block, name, Float64(lb), Float64(ub), NaN, String(label))

"""
    with_seed(ps, value) → ParamSpec

`ps` with starting value `value`, pulled 1e-10 inside the box so its logit is finite.
"""
with_seed(ps::ParamSpec, value::Real) =
    ParamSpec(ps.block, ps.name, ps.lb, ps.ub,
              clamp(value, ps.lb + 1e-10, ps.ub - 1e-10), ps.label)

"""
    param_symbol(ps) → String

The parameter's market-suffixed display symbol (`μ_U`, not the bare `μ` of `ps.name`): the
last word of `ps.label`, a trailing parenthetical dropped.
"""
function param_symbol(ps::ParamSpec)
    lab = replace(ps.label, r"\s*\([^)]*\)\s*$" => "")
    sym = last(split(lab))
    isempty(sym) ? string(ps.name) : String(sym)
end

"""
    padr(s, n) → String

`s` left-justified in a field `n` DISPLAY columns wide (`%-ns` pads bytes, which misaligns
names such as `a_ℓ`).
"""
padr(s, n::Int) = (t = string(s); t * " "^max(0, n - textwidth(t)))

# Display symbols of fixed parameters, keyed as in the fixed NamedTuple (no ParamSpec, so
# no label to read the symbol from).
const FIXED_SYMBOLS = Dict(
    :unsk_eta  => "η_U", :skl_eta  => "η_S",
    :unsk_η    => "η_U", :skl_η    => "η_S",
    :unsk_σ_w  => "σ_wU", :skl_σ_w => "σ_wS",
    :unsk_tight => "θ_U", :skl_tight => "θ_S",
    :PU        => "P_U",  :PS      => "P_S",
    :bT        => "b_T",  :bU      => "b_U", :bS => "b_S",
)

"""
    fixed_symbol(key) → String

Display symbol for a fixed parameter; the key itself for unsuffixed ones (`r`, `ν`, `φ`).
"""
fixed_symbol(key) = get(FIXED_SYMBOLS, Symbol(key), string(key))

# ============================================================
# SMMRunParams
# ============================================================

"""
    SMMRunParams

Run settings of an SMM estimation: grids, candidate layer, DE, SA, Nelder-Mead, weighting,
start-point construction and tracing. Keyword constructor; every field has a default.
smm_main.jl sets them on every run; the defaults matter only to a bundle read outside it.
`de_patience` and `sa_target_fin` are read by nothing and are kept for bundle layout only.
"""
Base.@kwdef struct SMMRunParams
    # Grids
    Nx      :: Int     = 80
    Np_U    :: Int     = 80
    Np_S    :: Int     = 80

    # Candidate layer (coarse grid, Sobol sample, hierarchical clustering)
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
    de_patience  :: Int     = 20       # unread
    de_avg_tol   :: Float64 = 0.0      # stop when (Q_mean−Q_best)/|Q_best| < tol; 0 disables

    # DE population generator and reheat
    de_gen_per_k   :: Int     = 30     # candidates drawn at each sparsity k = 1:n_free
    de_reheat_flat :: Int     = 6      # reheat once Q_best has been flat this many generations
    de_reheat_rate :: Float64 = 0.04   # and the improved-member rate is below this
    de_max_reheats :: Int     = 50
    de_adapt_fcr   :: Bool    = false  # take de_f, de_cr from the generator's draws

    # Simulated annealing
    sa_max_iter      :: Int     = 5_000
    sa_T0            :: Float64 = 2.0      # ≤ 0 ⇒ calibrated from sa_t0_rel, sa_t0_accept
    sa_step          :: Float64 = 0.01     # fallback step for an unmeasurable coordinate
    sa_cooling_rate  :: Float64 = 1.0      # logarithmic schedule (sa_halflife = 0)
    sa_cooling_exp   :: Float64 = 0.5
    sa_reheat_patience :: Int     = 200
    sa_reheat_factor   :: Float64 = 2.0
    sa_max_reheats     :: Int     = 5
    sa_adapt_window  :: Int     = 50
    sa_target_fin    :: Float64 = 0.90     # unread
    # T0 solves exp(−sa_t0_rel·Q0 / T0) = sa_t0_accept: a move worsening Q by sa_t0_rel of
    # its level is accepted with probability sa_t0_accept. 1e-4 is for warm starts.
    sa_t0_rel        :: Float64 = 1e-4
    sa_t0_accept     :: Float64 = 0.30
    # At a reheat, snap back to the incumbent when the walk is more than this fraction
    # above Q_best; otherwise reheat in place.
    sa_reheat_reset_tol :: Float64 = 0.25
    sa_random_init   :: Bool    = false
    sa_parallel_steps :: Int    = 100         # warm-up steps per start before pruning
    sa_seed           :: Int    = 20240601    # base seed of the SA chains

    # The objective's weighting, :diagonal or :equal
    weighting :: Symbol = :diagonal

    # Wage-reliability ratio; σ_{w,j}² = (1−λ_w)·Var̂[log w_j] (calibrate_sigma_w)
    λ_w :: Float64 = 0.78

    # Nelder-Mead (f_tol and x_tol are read by :lbfgs/:bfgs only; Optim's Nelder-Mead
    # ignores them, and its g_tol tests sd(Q) over the simplex in Q units)
    nm_max_iter  :: Int     = 10_000
    nm_f_tol     :: Float64 = 1e-3
    nm_x_tol     :: Float64 = 1e-5
    nm_g_tol     :: Float64 = 1e-8
    nm_no_improve :: Int    = 4_000  # evaluations without improvement before a stop; 0 disables

    # Tracing
    show_trace_members     :: Bool = false
    show_trace_generations :: Bool = true
    trace_stride           :: Int  = 10

    # Start-point construction
    use_as_seed          :: Bool    = true   # start from the chosen point itself
    seed_perturb_frac    :: Float64 = 0.05   # else jitter it by this fraction of each box width
    clusters_force_regen :: Bool    = true   # rebuild the candidate bank rather than reuse it
    include_prev_optimum :: Bool    = false  # add this window's previous optimum as a start

    # Rate-based stops (RateStop, smm.jl); tol = 0 disables
    sa_rate_tol   :: Float64 = 0.05
    sa_rate_span  :: Int     = 300
    sa_halflife   :: Int     = 5_000   # geometric cooling half-life, iterations
    nm_rate_tol   :: Float64 = 0.05
    nm_rate_span  :: Int     = 300

    # Nelder-Mead initial simplex cap, transformed units; 0 keeps Optim's default
    nm_simplex_step :: Float64 = 0.2

    # DE perturbation, as a fraction of each coordinate's measured ΔQ<1 width
    de_local_sigma :: Float64 = 0.33

end

# ============================================================
# RunProvenance
# ============================================================

"""
Bundle-format version (`RunProvenance.schema`, bundle.jl). Bump only when the top-level
field set of a bundle changes: 3 added `vacancy_costs` and `attracting`; 2 introduced the
constant field set; 1 is the earlier per-writer shapes.
"""
const BUNDLE_SCHEMA = 3

"""
    RunProvenance

Where a serialised bundle came from.

Fields
  schema         bundle-format version (`BUNDLE_SCHEMA`)
  version        ROYSEARCH_VERSION of the code that last WROTE the file
  window         estimation window
  w_suffix       weighting suffix
  git_sha        short commit hash; empty when git is unavailable
  host           host name; empty when unavailable
  julia          Julia version
  written_at     ISO 8601 UTC timestamp
  unrecorded     fields reconstructed rather than read from the original run
  migrated_from  versions the bundle passed through in migrations, oldest first; empty
                 for an estimated bundle
"""
Base.@kwdef struct RunProvenance
    schema        :: Int             = 1
    version       :: String          = "unknown"
    window        :: Symbol          = :unknown
    w_suffix      :: String          = ""
    git_sha       :: String          = ""
    host          :: String          = ""
    julia         :: String          = string(VERSION)
    written_at    :: String          = ""
    unrecorded    :: Vector{Symbol}  = Symbol[]
    migrated_from :: Vector{String}  = String[]
end

"""
    run_provenance(; window, w_suffix, version, unrecorded = Symbol[], migrated_from = String[])
        → RunProvenance

Provenance for the current process. The git SHA and host are probed and left empty on
failure. `migrated_from` lists the files a bundle was rebuilt from, when it was.
"""
function run_provenance(; window::Symbol, w_suffix::AbstractString,
                          version::AbstractString, unrecorded::Vector{Symbol} = Symbol[],
                          migrated_from::Vector{String} = String[])
    sha = try
        strip(read(`git rev-parse --short HEAD`, String))
    catch
        ""
    end
    host = try
        gethostname()
    catch
        ""
    end
    return RunProvenance(; schema = BUNDLE_SCHEMA, version = String(version),
                         window = window, w_suffix = String(w_suffix),
                         git_sha = String(sha), host = String(host),
                         written_at = _utc_now_iso(), unrecorded = unrecorded,
                         migrated_from = migrated_from)
end

"""
    _utc_now_iso() → String

The current UTC time as `YYYY-MM-DDThh:mm:ssZ`, without Dates (the solver path includes
this file).
"""
function _utc_now_iso()
    t = round(Int, time())
    days, rem = divrem(t, 86_400)
    h, rem2   = divrem(rem, 3_600)
    mi, s     = divrem(rem2, 60)
    y, mo, d  = _civil_from_days(days)
    return @sprintf("%04d-%02d-%02dT%02d:%02d:%02dZ", y, mo, d, h, mi, s)
end

"""
    _civil_from_days(z) → (year, month, day)

The civil date `z` days after 1970-01-01 (Hinnant's algorithm).
"""
function _civil_from_days(z::Int)
    z += 719_468
    era = fld(z, 146_097)
    doe = z - era * 146_097
    yoe = fld(doe - fld(doe, 1_460) + fld(doe, 36_524) - fld(doe, 146_096), 365)
    y   = yoe + era * 400
    doy = doe - (365 * yoe + fld(yoe, 4) - fld(yoe, 100))
    mp  = fld(5 * doy + 2, 153)
    d   = doy - fld(153 * mp + 2, 5) + 1
    m   = mp < 10 ? mp + 3 : mp - 9
    return (m <= 2 ? y + 1 : y), m, d
end

# ============================================================
# SeedBank
# ============================================================

"""
    SeedBank

A bank of candidate start points (candidates.jl), passed to `run_smm`.

Fields
  candidates :: Vector{Vector{Float64}}   feasible θ (unconstrained)
  Q          :: Vector{Float64}           coarse-grid Q of each candidate
  labels     :: Vector{Int}               cluster id (≥ 1; noise removed)
  meta       :: NamedTuple                cache key (`_candidate_meta`)
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

The estimation problem.

Fields
  free     Vector{ParamSpec}, the estimated parameters
  fixed    NamedTuple of pinned values; for the baseline's joint spec, only its periods
           under `PERIODS`
  moments  NamedTuple of `(value, weight)` targets; weight 0 = held out
  sim      SimParams, the solver settings
  run      SMMRunParams
  W        weight matrix over the active moments (never `nothing`: equal weights are
           stored as Diagonal(weight_k / m̂_k²))
  q_scale  display-only positive divisor of Q; constant in θ
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

"""
    build_smm_spec(moments, sim; fixed, free_specs, run, W, q_scale, skip_moments) → SMMSpec

Build an `SMMSpec`.

- `fixed`: pinned values, keyed bare (`:r`) or block-qualified (`:unsk_μ`) as `fixed_key`
  writes them; a free spec that is pinned is dropped from the free list.
- `skip_moments`: moments given weight 0 (unknown names are reported and ignored).
- `W`: `nothing` builds the equal-weight Diagonal(weight_k / m̂_k²), so g′Wg is the
  relative-deviation loss; a matrix (`load_weight_matrix`) must be square in the number of
  active moments, which are `keys(moments)` with weight > 0 in that order — the order of
  the deviation vector in `compute_loss_matrix`.
- `q_scale`: display-only positive divisor of Q.
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
    fixed_nt = (fixed isa NamedTuple) ? fixed : NamedTuple()

    active_free = filter(ps -> !is_pinned(fixed_nt, ps.block, ps.name), free_specs)

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
    end

    active_keys = [k for k in keys(moments) if moments[k].weight > 0.0]
    K_active    = length(active_keys)

    W_final::Matrix{Float64} = if isnothing(W)
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
# The baseline's joint spec
# ============================================================

# The tag a period's parameters carry in tables: the event its window precedes.
const PERIOD_TAG = Dict(:base_fc => "fc", :base_covid => "covid")

"""
    joint_spec(specs, windows) → SMMSpec

The baseline's spec: one parameter vector for its periods (data windows `windows`, with
single-period specs `specs` sharing grids, run settings and free set). A parameter of
`BASELINE_COMMON` is one coordinate for every period and must have the same box and start
in each; every other free parameter is one coordinate per period, named
`<name>@<window>`. The objective is the sum of the periods' objectives: moments are keyed
`<moment>@<window>` and `W` is block diagonal. The periods are recorded under
`fixed[PERIODS]`.
"""
function joint_spec(specs::AbstractVector{SMMSpec}, windows)
    s1 = first(specs)
    names = [(ps.block, ps.name) for ps in s1.free]
    all(s -> [(ps.block, ps.name) for ps in s.free] == names, specs) ||
        error("joint_spec: the periods do not estimate the same parameters.")
    all(s -> s.q_scale == s1.q_scale, specs) ||
        error("joint_spec: the periods' q_scale differ, so their objectives do not add up.")
    free  = ParamSpec[]
    index = [Vector{Int}(undef, length(names)) for _ in specs]
    for (k, (blk, nm)) in enumerate(names)
        pss = [s.free[k] for s in specs]
        if (blk, nm) in BASELINE_COMMON
            all(ps -> (ps.lb, ps.ub, ps.init) == (pss[1].lb, pss[1].ub, pss[1].init), pss) ||
                error("joint_spec: $blk:$nm is common to the periods but its box or start differs.")
            push!(free, pss[1])
            foreach(ix -> ix[k] = length(free), index)
        else
            for (i, (ps, w)) in enumerate(zip(pss, windows))
                push!(free, ParamSpec(blk, Symbol(nm, "@", w), ps.lb, ps.ub, ps.init,
                                      "$(ps.label)[$(PERIOD_TAG[w])]"))
                index[i][k] = length(free)
            end
        end
    end
    moments = NamedTuple(Symbol(k, "@", w) => v
                         for (s, w) in zip(specs, windows) for (k, v) in pairs(s.moments))
    W = zeros(sum(size(s.W, 1) for s in specs), sum(size(s.W, 1) for s in specs))
    o = 0
    for s in specs
        n = size(s.W, 1);  W[o+1:o+n, o+1:o+n] .= s.W;  o += n
    end
    periods = (windows = Tuple(Symbol.(windows)), specs = Tuple(specs), index = Tuple(index))
    return SMMSpec(free, NamedTuple{(PERIODS,)}((periods,)), moments, s1.sim, s1.run, W,
                   s1.q_scale)
end

"""
    is_joint(spec) → Bool

Whether `spec` is the baseline's joint spec (`joint_spec`).
"""
is_joint(spec::SMMSpec) = haskey(spec.fixed, PERIODS)

"""
    period_list(spec) → Vector of (window, spec, index)

A spec's periods: for the joint spec, each data window with its single-period spec and the
positions of that spec's coordinates in the joint vector; for any other spec, the spec
itself with `window = nothing` at every position.
"""
function period_list(spec::SMMSpec)
    is_joint(spec) || return [(nothing, spec, collect(eachindex(spec.free)))]
    P = spec.fixed[PERIODS]
    return [(P.windows[i], P.specs[i], P.index[i]) for i in eachindex(P.specs)]
end

"""
    pinned_key(spec) → Vector

The pinned values of `spec`, period by period: what two specs must share for an estimate of
one to start the other.
"""
pinned_key(spec::SMMSpec) = [pinned_values(s.fixed) for (_, s, _) in period_list(spec)]

"""
    pinned_table(spec) → Vector of (key, symbol, value)

The pinned parameters for display and the CSV: a value common to every period once; one
that differs across the baseline's periods once per period, keyed `<key>@<window>`.
"""
function pinned_table(spec::SMMSpec)
    pl  = period_list(spec)
    out = Tuple{Symbol,String,Float64}[]
    for key in keys(pinned_values(pl[1][2].fixed))
        vals = [Float64(pinned_values(s.fixed)[key]) for (_, s, _) in pl]
        if all(==(vals[1]), vals)
            push!(out, (key, fixed_symbol(key), vals[1]))
        else
            for ((w, _, _), v) in zip(pl, vals)
                push!(out, (Symbol(key, "@", w), "$(fixed_symbol(key))[$(PERIOD_TAG[w])]", v))
            end
        end
    end
    return out
end

"""
    is_tightness(ps) → Bool

Whether `ps` is a tightness coordinate. The tightnesses are what the optimiser moves; the
vacancy costs follow from free entry and are reported in their place.
"""
is_tightness(ps::ParamSpec) = ps.name === :tight

"""
    vacancy_cost_rows(spec) → Vector of (block, name, label, symbol)

The vacancy costs a spec's estimate carries, in the order of `k_values`: k_U and k_S, once
per period of the baseline.
"""
vacancy_cost_rows(spec::SMMSpec) =
    [(blk, w === nothing ? :k : Symbol("k@", w),
      w === nothing ? label : "$label[$(PERIOD_TAG[w])]",
      (blk === :unsk ? "k_U" : "k_S") * (w === nothing ? "" : "[$(PERIOD_TAG[w])]"))
     for (w, _, _) in period_list(spec) for (blk, label) in VACANCY_COSTS]

"""
    k_values(k) → Vector{Float64}
    k_struct(spec, v) → NamedTuple

The vacancy costs of an equilibrium as a vector in the order of `vacancy_cost_rows`, and
back. A single-period spec's costs are `(U, S)`; the baseline's are one `(U, S)` per
period, keyed by data window.
"""
k_values(k::NamedTuple) = haskey(k, :U) ? [k.U, k.S] :
                          reduce(vcat, [[x.U, x.S] for x in values(k)])

function k_struct(spec::SMMSpec, v::AbstractVector)
    pl = period_list(spec)
    pl[1][1] === nothing && return (U = v[1], S = v[2])
    return NamedTuple{Tuple(w for (w, _, _) in pl)}(
        Tuple((U = v[2i-1], S = v[2i]) for i in eachindex(pl)))
end

# ============================================================
# σ_w calibration
# ============================================================

"""
    calibrate_sigma_w(λ_w, moments) → (σ_wU, σ_wS)

Log-wage measurement-error SDs from the reliability ratio: σ_{w,j}² = (1−λ_w)·Var̂[log w_j],
with Var̂[log w_j] the `emp_var_U` / `emp_var_S` targets of the data `moments`.
"""
function calibrate_sigma_w(λ_w::Real, moments::NamedTuple)
    σ_wU = sqrt(max((1.0 - λ_w) * moments[:emp_var_U].value, 0.0))
    σ_wS = sqrt(max((1.0 - λ_w) * moments[:emp_var_S].value, 0.0))
    return (σ_wU, σ_wS)
end

# ============================================================
# Parameter table
# ============================================================

"""
    default_free_params() → Vector{ParamSpec}

Every estimable parameter with its box, unseeded. r, ν and φ are not in it: they are always
calibrated externally. Production is π_j = e^A P_j a_j p, so only e^A·P_j is pinned by wage
levels and P_U = 1 is pinned as the numeraire.
"""
function default_free_params() :: Vector{ParamSpec}
    return [
        # Ability distribution: a_ℓ, b_ℓ shape the a_U marginal; ρ_x is the ability
        # correlation, the segment-separation device, boxed strongly negative because the
        # estimate is not stable otherwise.
        ParamSpec(:common, :a_ℓ,    0.1000,    8.0000,  "worker type shape a_ℓ"),
        ParamSpec(:common, :b_ℓ,    0.0500,    4.0000,  "worker type shape b_ℓ"),
        ParamSpec(:common, :ρ_x,   -0.90000,    -0.10000,  "ability correlation ρ_x"),

        ParamSpec(:common, :c,      6.00000,   15.00000,  "training cost coeff c"),
        ParamSpec(:common, :A,      4.00000,   10.00000,  "aggregate production scale A"),

        # Outside flows (stored in the block that consumes them)
        ParamSpec(:unsk,   :bU,     0.000000,  0.200000, "unskilled outside flow b_U"),
        ParamSpec(:unsk,   :bT,     0.000000,  1.500000, "training flow b_T"),
        ParamSpec(:skl,    :bS,     0.000000,  0.500000, "skilled outside flow b_S"),

        # Productivity levels and match-quality distributions
        ParamSpec(:unsk,   :PU,     0.000016,  1.303171, "unskilled productivity P_U"),
        ParamSpec(:skl,    :PS,     1.000016,  5.000000, "skilled productivity P_S"),
        ParamSpec(:unsk,   :α_U,    0.1000,    6.0000,  "unskilled damage shape α_U"),
        ParamSpec(:skl,    :a_Γ,    0.1000,   12.0000,  "skilled offer shape a_Γ"),
        ParamSpec(:skl,    :b_Γ,    0.1000,   10.0000,  "skilled offer shape b_Γ"),
        ParamSpec(:skl,    :δ,      0.0500,    1.0000,  "shock/offer support ratio δ_S"),

        # Matching efficiency, matching elasticity, bargaining
        ParamSpec(:unsk,   :μ,      0.0001,    2.0000,  "unskilled matching eff μ_U"),
        ParamSpec(:skl,    :μ,      0.0001,    2.0000,  "skilled matching eff μ_S"),
        ParamSpec(:unsk,   :η,      0.0001,    1.0000,  "unskilled matching elas η_U"),
        ParamSpec(:skl,    :η,      0.0001,    1.0000,  "skilled matching elas η_S"),
        ParamSpec(:unsk,   :β,      0.0010,    0.9000,  "unskilled bargaining β_U"),
        ParamSpec(:skl,    :β,      0.0001,    0.9000,  "skilled bargaining β_S"),

        # Shock arrival rates
        ParamSpec(:unsk,   :λ,      0.0001,    0.5500,  "unskilled damage rate λ_U"),
        ParamSpec(:skl,    :λ,      0.0001,    0.5000,  "skilled quality shock λ_S"),

        # Tightnesses, vacancies per searcher: estimated in place of the vacancy costs.
        # Given (θ_U, θ_S) the stationary state is unique and free entry gives each cost
        # in closed form; costs → tightnesses is a correspondence.
        ParamSpec(:unsk,   :tight,  0.0500,    3.0000,  "unskilled tightness θ_U"),
        ParamSpec(:skl,    :tight,  0.1000,    5.0000,  "skilled tightness θ_S"),

        # OJS cost and exogenous separation. ξ_j is a quality-independent hazard on top of
        # the endogenous margin λ_j·G_j(p*_j); δ_S = 1 is the single-distribution limit.
        ParamSpec(:skl,    :σ,      0.0000,    0.5000,  "OJS flow cost σ_S"),
        ParamSpec(:unsk,   :ξ,      0.0000,    0.0200,  "unskilled exogenous separation ξ_U"),
        ParamSpec(:skl,    :ξ,      0.0000,    0.0200,  "skilled exogenous separation ξ_S"),

        # Wage measurement error (pinned from λ_w when λ_w > 0)
        ParamSpec(:unsk,   :σ_w,    0.0000,    1.0000,  "unskilled wage meas. error σ_wU"),
        ParamSpec(:skl,    :σ_w,    0.0000,    1.0000,  "skilled wage meas. error σ_wS"),
    ]
end

# The vacancy costs, (block, label): computed from free entry at every evaluation and
# listed beside the free parameters, in months of the mean output of a new match.
const VACANCY_COSTS = ((:unsk, "unskilled vacancy cost k_U"),
                       (:skl,  "skilled vacancy cost k_S"))

# The joint ability distribution. A tuple, so its order (which enters the candidate-cache
# key through the pinned block) is fixed.
const ABILITY_PARAMS = ((:common, :a_ℓ), (:common, :b_ℓ), (:common, :ρ_x))

# Parameters the baseline's two periods share (the estimation design, Table 1).
const BASELINE_COMMON = (ABILITY_PARAMS..., (:unsk, :bU), (:skl, :bS), (:skl, :σ))

# Parameters a crisis window holds at its baseline period's estimate (Table 3).
const CRISIS_HELD = (ABILITY_PARAMS..., (:common, :c), (:skl, :PS), (:unsk, :μ), (:skl, :μ),
                     (:skl, :σ))

# Key of `SMMSpec.fixed` under which the joint spec records its periods (`joint_spec`).
# Not a parameter value: read the pinned values through `pinned_values`.
const PERIODS = :periods

"""
    pinned_values(fixed) → NamedTuple

The pinned parameter values of a fixed NamedTuple, the `PERIODS` record removed.
"""
pinned_values(fixed::NamedTuple) = Base.structdiff(fixed, NamedTuple{(PERIODS,)})

# ============================================================
# Fixed-key convention
# ============================================================

# Field names carried by more than one block (μ, η, tight, β, λ, ξ, σ_w), derived from the
# parameter table.
const _SHARED_PARAM_NAMES = let counts = Dict{Symbol,Int}()
    for ps in default_free_params()
        counts[ps.name] = get(counts, ps.name, 0) + 1
    end
    Set(nm for (nm, n) in counts if n > 1)
end

"""
    fixed_key(block, name) → Symbol

The key under which to write a pinned value into `SMMSpec.fixed`: `<block>_<name>` for a name
both blocks carry, the bare `name` otherwise.
"""
fixed_key(block::Symbol, name::Symbol) :: Symbol =
    name in _SHARED_PARAM_NAMES ? Symbol(string(block) * "_" * string(name)) : name

"""
    is_pinned(fixed, block, name) → Bool

Whether (block, name) is pinned in `fixed`, under its qualified or its bare key (as
`unpack_θ` reads it).
"""
is_pinned(fixed, block::Symbol, name::Symbol) :: Bool =
    haskey(fixed, Symbol(string(block) * "_" * string(name))) || haskey(fixed, name)

"""
    assert_all_params_accounted(spec; context = "spec")

Raise, listing them all, if any parameter of `default_free_params()` is neither estimated nor
pinned in `spec` (per period for the joint spec). Run before any solve.
"""
function assert_all_params_accounted(spec::SMMSpec; context::String = "spec")
    if is_joint(spec)
        for (w, s, _) in period_list(spec)
            assert_all_params_accounted(s; context = "$context, $w")
        end
        return nothing
    end
    estimated   = Set((ps.block, ps.name) for ps in spec.free)
    unaccounted = Tuple{Symbol,Symbol}[]
    for ps in default_free_params()
        (ps.block, ps.name) in estimated          && continue
        is_pinned(spec.fixed, ps.block, ps.name)  && continue
        push!(unaccounted, (ps.block, ps.name))
    end
    isempty(unaccounted) && return nothing
    error(
        "$context: parameter(s) neither estimated nor pinned — unpack_θ has no value for " *
        join(("$(b):$(n)" for (b, n) in unaccounted), ", ") *
        ". Add them to the fixed block or to the free set."
    )
end

# ============================================================
# ASCII ↔ Unicode key maps
#
# smm_main.jl configures runs with ASCII keys (:a_l, :rho_x, :unsk_bet); the solver
# structs and the parameter table use Unicode names (:a_ℓ, :ρ_x, :β).
# ============================================================

# (block, name) → ASCII key of DEFAULT_PARAMS / DEFAULT_PERIOD_PARAMS.
const _DEFAULT_PARAM_KEY = Dict{Tuple{Symbol,Symbol}, Symbol}(
    (:common, :a_ℓ) => :a_l,     (:common, :b_ℓ)  => :b_l,     (:common, :c)   => :c,
    (:common, :ρ_x) => :rho_x,   (:common, :A)    => :A,
    (:unsk,   :PU)  => :PU,      (:skl,    :PS) => :PS,
    (:unsk,   :bU)  => :bU,      (:unsk,   :bT)   => :bT,      (:skl,    :bS)  => :bS,
    (:unsk,   :α_U) => :alpha_U, (:skl,    :a_Γ)  => :a_Gam,  (:skl,    :b_Γ) => :b_Gam,
    (:unsk,   :μ)   => :unsk_mu, (:unsk,   :η)    => :unsk_eta, (:unsk,  :tight) => :unsk_tight,
    (:unsk,   :β)   => :unsk_bet, (:unsk,  :λ)   => :unsk_lam,  (:unsk, :σ_w)  => :unsk_sigw,
    (:unsk,   :ξ)   => :unsk_xi,
    (:skl,    :μ)   => :skl_mu,  (:skl,    :η)    => :skl_eta,  (:skl,   :tight) => :skl_tight,
    (:skl,    :β)   => :skl_bet, (:skl,    :λ)   => :skl_lam,
    (:skl,    :σ)   => :skl_sig, (:skl,    :σ_w)  => :skl_sigw,
    (:skl,    :ξ)   => :skl_xi,  (:skl,    :δ)    => :skl_delta,
)

# ASCII key (FIX_PARAMS) → key of the fixed NamedTuple.
const _ASCII_TO_FIXED_KEY = Dict{Symbol, Symbol}(
    :r        => :r,
    :nu       => :ν,
    :phi      => :φ,
    :a_l      => :a_ℓ,
    :b_l      => :b_ℓ,
    :rho_x    => :ρ_x,
    :c        => :c,
    :A        => :A,
    :PU       => :PU,
    :PS       => :PS,
    :bU       => :bU,
    :bT       => :bT,
    :bS       => :bS,
    :alpha_U  => :α_U,
    :a_Gam    => :a_Γ,
    :b_Gam    => :b_Γ,
    :unsk_mu  => :unsk_μ,
    :unsk_eta => :unsk_η,
    :unsk_tight => :unsk_tight,
    :unsk_bet => :unsk_β,
    :unsk_lam => :unsk_λ,
    :unsk_xi  => :unsk_ξ,
    :unsk_sigw => :unsk_σ_w,
    :skl_mu   => :skl_μ,
    :skl_eta  => :skl_η,
    :skl_tight => :skl_tight,
    :skl_bet  => :skl_β,
    :skl_lam  => :skl_λ,
    :skl_sig  => :skl_σ,
    :skl_sigw  => :skl_σ_w,
    :skl_xi   => :skl_ξ,
    :skl_delta => :skl_δ,
)

# The vacancy costs follow from the tightnesses through free entry, so they cannot be pinned.
const _COST_KEYS = (:unsk_k, :skl_k)

"""
    _fix_params_to_nt(fix_dict) → NamedTuple

FIX_PARAMS (ASCII keys) as a fixed NamedTuple. Pinning a vacancy cost is an error; an
unrecognised key prints a WARNING and is ignored, so that parameter is estimated.
"""
function _fix_params_to_nt(fix_dict::Dict{Symbol,Float64}) :: NamedTuple
    isempty(fix_dict) && return (;)
    pinned_k = [k for k in keys(fix_dict) if k in _COST_KEYS]
    isempty(pinned_k) || error(
        "FIX_PARAMS: " * join((":$k" for k in pinned_k), ", ") * " cannot be pinned. The " *
        "vacancy costs are computed from the tightnesses in every evaluation; pin " *
        ":unsk_tight / :skl_tight instead.")
    keys_vec = Symbol[]
    vals_vec = Float64[]
    for (ascii_key, val) in fix_dict
        if haskey(_ASCII_TO_FIXED_KEY, ascii_key)
            push!(keys_vec, _ASCII_TO_FIXED_KEY[ascii_key])
            push!(vals_vec, val)
        else
            valid_str = join(sort(string.(collect(keys(_ASCII_TO_FIXED_KEY)))), ", ")
            @printf("WARNING: FIX_PARAMS — unrecognised key :%s — ignored.\n", ascii_key)
            @printf("  Valid keys: %s\n", valid_str)
        end
    end
    isempty(keys_vec) && return (;)
    return NamedTuple{Tuple(keys_vec)}(Tuple(vals_vec))
end

"""
    _baseline_param_value(block, name, cp, up, sp, tight) → Float64
    _baseline_param_value(ps::ParamSpec, cp, up, sp, tight) → Float64

One parameter of a decoded point (`unpack_θ`'s `cp`, `up`, `sp` and tightnesses `tight`).
"""
function _baseline_param_value(block::Symbol, name::Symbol, cp, up, sp, tight) :: Float64
    name === :tight && return Float64(block == :unsk ? tight.U : tight.S)
    if     block == :common; return Float64(getfield(cp, name))
    elseif block == :unsk;   return Float64(getfield(up, name))
    else                     return Float64(getfield(sp, name))
    end
end

_baseline_param_value(ps::ParamSpec, cp, up, sp, tight) :: Float64 =
    _baseline_param_value(ps.block, ps.name, cp, up, sp, tight)

# ============================================================
# Box transforms (lb, ub) ↔ ℝ
# ============================================================

"""
    _to_unconstrained(x, lb, ub) → Float64
    _to_constrained(t, lb, ub) → Float64

The logit of x's position in [lb, ub] (clamped to [1e-8, 1 − 1e-8]), and its inverse.
"""
@inline function _to_unconstrained(x::Float64, lb::Float64, ub::Float64)
    p = (x - lb) / (ub - lb)
    p = clamp(p, 1e-8, 1.0 - 1e-8)
    return log(p / (1.0 - p))
end

@inline function _to_constrained(t::Float64, lb::Float64, ub::Float64)
    sig = 1.0 / (1.0 + exp(-t))
    return lb + (ub - lb) * sig
end

"""
    logjac_box(θ_unc, free) → Float64

log |dθ/dt| of the box transform summed over the free parameters,
Σ log(ub−lb) + log σ(t) + log(1−σ(t)). A sampler moving in t must add it to the log target
for the induced prior on θ to be flat on the box; without it the target has poles at the
bounds and, Q being asymptotically flat, is improper.
"""
function logjac_box(θ_unc::AbstractVector{Float64}, free::Vector{ParamSpec})
    lj = 0.0
    @inbounds for (k, ps) in enumerate(free)
        t   = θ_unc[k]
        at  = abs(t)
        # log σ(t) + log(1−σ(t)) = −|t| − 2·log1p(exp(−|t|)), stable for large |t|
        lj += log(ps.ub - ps.lb) - at - 2.0 * log1p(exp(-at))
    end
    return lj
end

# ============================================================
# pack_θ / unpack_θ
# ============================================================

"""
    pack_θ(free_specs) → Vector{Float64}
    pack_θ(spec)       → Vector{Float64}

The starting values of a free-parameter list (or of `spec.free`) as an unconstrained
vector. Raises, naming them, on unseeded (non-finite) entries.
"""
function pack_θ(free_specs::Vector{ParamSpec}) :: Vector{Float64}
    unseeded = ["$(ps.block):$(ps.name)" for ps in free_specs if !isfinite(ps.init)]
    isempty(unseeded) || error(
        "pack_θ: no starting value for " * join(unseeded, ", ") * ". Seed the spec from " *
        "DEFAULT_PARAMS (smm_main.jl) or a stored optimum before packing it.")
    return [_to_unconstrained(ps.init, ps.lb, ps.ub) for ps in free_specs]
end

pack_θ(spec::SMMSpec) :: Vector{Float64} = pack_θ(spec.free)

"""
    perturb_free_init(free_specs, frac, rng) → Vector{ParamSpec}

`free_specs` with each start jittered by N(0, (frac·(ub−lb))²), kept inside the box.
"""
function perturb_free_init(free_specs::Vector{ParamSpec}, frac::Float64, rng) :: Vector{ParamSpec}
    return [with_seed(ps, ps.init + frac * (ps.ub - ps.lb) * randn(rng)) for ps in free_specs]
end

"""
    unpack_θ(θ_unc, spec; constrained = false) → (CommonParams, UnskilledParams, SkilledParams, tight)

The model's parameter structs and tightnesses `tight = (U = θ_U, S = θ_S)` at the
unconstrained vector `θ_unc` of a single-period spec. Each field is pinned (qualified key,
then bare; pinned beats free) or free; a field that is neither raises. The vacancy costs
are NaN until a solve at these tightnesses derives them. `constrained = true` takes
`θ_unc` as the constrained values themselves and skips the box map (θ → t → θ is not exact
next to a bound).
"""
function unpack_θ(
    θ_unc :: AbstractVector{Float64},
    spec  :: SMMSpec;
    constrained :: Bool = false
)
    is_joint(spec) && error("unpack_θ: the baseline's spec has one parameter set per period; " *
                            "decode a period through period_list(spec).")
    free_vals = Dict{Symbol, Float64}()
    for (i, ps) in enumerate(spec.free)
        free_vals[fixed_key(ps.block, ps.name)] =
            constrained ? θ_unc[i] : _to_constrained(θ_unc[i], ps.lb, ps.ub)
    end

    function _get(name::Symbol, block::Symbol) :: Float64
        qualified = Symbol(string(block) * "_" * string(name))
        haskey(spec.fixed, qualified) && return Float64(spec.fixed[qualified])
        haskey(spec.fixed, name)      && return Float64(spec.fixed[name])
        v = get(free_vals, fixed_key(block, name), nothing)
        isnothing(v) || return v
        error("unpack_θ: $(block):$(name) is neither free nor pinned in this spec, so it " *
              "has no value. Free it or pin it (spec.fixed) — see assert_all_params_accounted.")
    end

    cp = CommonParams(
        r   = _get(:r,   :common),
        ν   = _get(:ν,   :common),
        φ   = _get(:φ,   :common),
        a_ℓ = _get(:a_ℓ, :common),
        b_ℓ = _get(:b_ℓ, :common),
        ρ_x = _get(:ρ_x, :common),
        c   = _get(:c,   :common),
        A   = _get(:A,   :common),   # ln A
    )

    up = UnskilledParams(
        μ   = _get(:μ,   :unsk),
        η   = _get(:η,   :unsk),
        k   = NaN,
        β   = _get(:β,   :unsk),
        λ   = _get(:λ,   :unsk),
        PU  = _get(:PU,  :unsk),
        bU  = _get(:bU,  :unsk),
        bT  = _get(:bT,  :unsk),
        α_U = _get(:α_U, :unsk),
        ξ   = _get(:ξ,   :unsk),
        σ_w = _get(:σ_w, :unsk),
    )

    sp = SkilledParams(
        μ   = _get(:μ,   :skl),
        η   = _get(:η,   :skl),
        k   = NaN,
        β   = _get(:β,   :skl),
        λ   = _get(:λ,   :skl),
        σ   = _get(:σ,   :skl),
        PS  = _get(:PS,  :skl),
        bS  = _get(:bS,  :skl),
        a_Γ = _get(:a_Γ, :skl),
        b_Γ = _get(:b_Γ, :skl),
        δ   = _get(:δ,   :skl),
        ξ   = _get(:ξ,   :skl),
        σ_w = _get(:σ_w, :skl),
    )

    return cp, up, sp, (U = _get(:tight, :unsk), S = _get(:tight, :skl))
end

# ============================================================
# Printing
# ============================================================

"""
    print_spec(spec; k = nothing)

Print the estimation problem: free parameters with bounds and seeds (the vacancy costs in
place of the tightnesses, with their seed values when `k` is given), fixed parameters,
active and skipped moments, and the requested run settings. The settings are those of
`spec.run`; what SA actually uses is the `[SA config]` line printed by `_sa_loop`.
"""
function print_spec(spec::SMMSpec; k = nothing)
    costs = vacancy_cost_rows(spec)
    @printf("\n╔══════════════════════════════════════════════════════╗\n")
    @printf("║  SMM Estimation Specification                        ║\n")
    @printf("╠══════════════════════════════════════════════════════╣\n")
    @printf("║  Free parameters (%d)                                ║\n",
            count(!is_tightness, spec.free) + length(costs))
    @printf("╠══════════════════════════════════════════════════════╣\n")
    @printf("  %s%s%10s  %10s  %10s\n",
            padr("block", 8), padr("param", 11), "lb", "ub", "seed")
    @printf("  %s\n", "─"^53)
    for ps in spec.free
        is_tightness(ps) && continue
        @printf("  %s%s%10.4f  %10.4f  %10.4f\n",
                padr(ps.block, 8), padr(param_symbol(ps), 11), ps.lb, ps.ub, ps.init)
    end
    for ((blk, _, _, sym), v) in zip(costs, k === nothing ? fill(NaN, length(costs)) : k_values(k))
        @printf("  %s%s%10s  %10s  %10.4f\n", padr(blk, 8), padr(sym, 11), "", "", v)
    end

    pinned = pinned_table(spec)
    if length(pinned) > 0
        @printf("\n╠══════════════════════════════════════════════════════╣\n")
        @printf("║  Fixed parameters (%d)                                ║\n",
                length(pinned))
        @printf("╠══════════════════════════════════════════════════════╣\n")
        for (_, sym, v) in pinned
            @printf("  %s = %.6f\n", padr(sym, 20), v)
        end
    end

    active_moments  = [(k, v) for (k, v) in pairs(spec.moments) if v.weight > 0.0]
    skipped_moments = [(k, v) for (k, v) in pairs(spec.moments) if v.weight <= 0.0]

    @printf("\n╠══════════════════════════════════════════════════════╣\n")
    @printf("║  Active moments (%2d)                                 ║\n", length(active_moments))
    @printf("╠══════════════════════════════════════════════════════╣\n")
    if spec.run.weighting === :equal
        @printf("  (Weighting: equal weights — W = Diagonal(weight/m̂²), relative-deviation loss)\n")
        @printf("  (Q divided by q_scale = %.6e for display)\n\n", spec.q_scale)
        @printf("  %-26s  %10s  %10s\n", "moment", "target", "weight")
        @printf("  %s\n", "─"^50)
        for (k, v) in active_moments
            @printf("  %-26s  %10.4f  %10.2f\n", k, v.value, v.weight)
        end
    else
        @printf("  (Weighting: diagonal — W = Diagonal(1/σ̂²_samp), raw deviations g'Wg)\n")
        @printf("  (Q divided by q_scale = %.6e for display)\n\n", spec.q_scale)
        @printf("  %-26s  %10s  %12s\n", "moment", "target", "W diag wt")
        @printf("  %s\n", "─"^52)
        for (idx, (k, v)) in enumerate(active_moments)
            w_diag = (idx <= size(spec.W, 1)) ? spec.W[idx, idx] : NaN
            @printf("  %-26s  %10.4f  %12.4e\n", k, v.value, w_diag)
        end
    end

    if !isempty(skipped_moments)
        @printf("\n╠══════════════════════════════════════════════════════╣\n")
        @printf("║  Skipped moments (%2d, weight = 0)                    ║\n", length(skipped_moments))
        @printf("╠══════════════════════════════════════════════════════╣\n")
        @printf("  %-26s  %10s\n", "moment", "target")
        @printf("  %s\n", "─"^38)
        for (k, v) in skipped_moments
            @printf("  %-26s  %10.4f\n", k, v.value)
        end
    end
    @printf("\n  Grid: Nx=%d  Np_U=%d  Np_S=%d\n",
            spec.run.Nx, spec.run.Np_U, spec.run.Np_S)
    @printf("  λ_w (wage-reliability calibration knob, σ_w = √((1−λ_w)·Var̂[log w])): %.4f\n",
            spec.run.λ_w)
    @printf("  SA:  max_iter=%d  T0=%.2f  step=%.2f  cooling_rate=%.2f  cooling_exp=%.2f  reheat_patience=%d  reheat_factor=%.2f  adapt_window=%d\n",
            spec.run.sa_max_iter, spec.run.sa_T0, spec.run.sa_step,
            spec.run.sa_cooling_rate, spec.run.sa_cooling_exp,
            spec.run.sa_reheat_patience, spec.run.sa_reheat_factor,
            spec.run.sa_adapt_window)
    @printf("  DE:   max_iter=%d  pop_size=%s  f=%.2f  cr=%.2f  avg_tol=%s\n",
            spec.run.de_max_iter,
            spec.run.de_pop_size == 0 ? "auto" : string(spec.run.de_pop_size),
            spec.run.de_f, spec.run.de_cr,
            spec.run.de_avg_tol > 0.0 ? @sprintf("%.1e", spec.run.de_avg_tol) : "off")
    @printf("  NM:   max_iter=%d  f_tol=%.0e  x_tol=%.0e  g_tol=%.0e  no_improve=%s\n",
            spec.run.nm_max_iter, spec.run.nm_f_tol, spec.run.nm_x_tol, spec.run.nm_g_tol,
            spec.run.nm_no_improve > 0 ? string(spec.run.nm_no_improve) : "off")
    @printf("  SA (requested):   cooling %s;  stop ΔQ < %s per 100 iters\n",
            spec.run.sa_halflife > 0 ?
                @sprintf("geometric, half-life %d", spec.run.sa_halflife) : "logarithmic",
            spec.run.sa_rate_tol > 0 ? @sprintf("%.3g", spec.run.sa_rate_tol) : "off")
    @printf("  NM (requested):   stop ΔQ < %s per 100 evals, sustained over %d evals\n",
            spec.run.nm_rate_tol > 0 ? @sprintf("%.3g", spec.run.nm_rate_tol) : "off",
            spec.run.nm_rate_span)
    @printf("  DE:   population — %d draws at each k=1:%d at %.2f×measured width; slots and f,cr from yield\n",
            spec.run.de_gen_per_k, length(spec.free), spec.run.de_local_sigma)
    if spec.run.weighting === :equal
        @printf("  W:    equal weights — Diagonal(weight/m̂²) (cond(W)=%.2e)\n", cond(spec.W))
    else
        @printf("  W:    diagonal-σ — Diagonal(1/σ̂²_samp) (cond(W)=%.2e)\n", cond(spec.W))
    end
    @printf("  Q scale (display-only divisor): %.6e\n", spec.q_scale)
    @printf("╚══════════════════════════════════════════════════════╝\n\n")
end

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
#   unpack_θ(θ, spec)      unconstrained vector → 3 model structs and the tightnesses
#   print_spec(spec)       display the estimation problem
#
# Parameters
#
#   Externally set, always fixed at SMM runtime:
#     r       calibrated to 4% annual real risk-free rate (monthly r = 0.05/12)
#     ν       pre-estimated from CPS life-table (one row per crisis pair)
#     φ       pre-estimated from NSC/IPEDS (pooled across Fall semesters)
#
#   Every other parameter is estimated in each window unless pinned via
#   FIX_PARAMS, except that a crisis window holds the ability distribution
#   (ABILITY_PARAMS: a_ℓ, b_ℓ, ρ_x) at its paired baseline estimate.
#
#   Production is linear in own ability under pure-Roy (π_j = exp(A)·P_j·a_j·p),
#   so there is no ability gradient to estimate; segment separation runs through
#   the ρ_x separation device (Gola concordance).
#
# Note on ξ_U / ξ_S: each of UnskilledParams / SkilledParams carries an
# exogenous-separation field ξ, defaulting to 0.0. Total separation into
# unemployment is ξ_U + λ_U · G(p*_U) (unskilled) and ξ_S + λ_S · Γ(p*_S)
# (skilled) — see Model Notes §sep_asymmetry. Pinning a ξ to 0 (FIX_PARAMS
# :unsk_xi / :skl_xi => 0.0, or dropping it from the free list) recovers the
# purely-endogenous variant for that sector exactly.
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
  init    :: Float64  starting value (constrained space); NaN until seeded
  label   :: String   human-readable label for printing

`init` is a carrier, not a default: the parameter table builds every entry unseeded,
and the estimation driver writes the seed in — from DEFAULT_PARAMS in smm_main.jl or
from a stored optimum. `pack_θ` refuses an unseeded entry, so a path that forgets to
seed fails at its first use instead of starting from a value nobody chose.
"""
struct ParamSpec
    block :: Symbol
    name  :: Symbol
    lb    :: Float64
    ub    :: Float64
    init  :: Float64
    label :: String
end

ParamSpec(block::Symbol, name::Symbol, lb::Real, ub::Real, label::AbstractString) =
    ParamSpec(block, name, Float64(lb), Float64(ub), NaN, String(label))

"""
    with_seed(ps, value) → ParamSpec

`ps` carrying `value` as its starting value, pulled 1e-10 inside the box so the logit
map in `pack_θ` stays finite at a seed that sits on a bound.
"""
with_seed(ps::ParamSpec, value::Real) =
    ParamSpec(ps.block, ps.name, ps.lb, ps.ub,
              clamp(value, ps.lb + 1e-10, ps.ub - 1e-10), ps.label)

"""
    param_symbol(ps) → String

The parameter's canonical symbol, market suffix included: `μ_U` rather than the bare
`μ` that `ps.name` carries for both markets. Taken from the tail of `ps.label`, which
already spells each symbol the way the model notes and the paper do, so the suffix
convention lives in one place — the label — rather than in a second table that can
drift from it. A trailing parenthetical (units, as on the vacancy costs) is dropped.
"""
function param_symbol(ps::ParamSpec)
    lab = replace(ps.label, r"\s*\([^)]*\)\s*$" => "")
    sym = last(split(lab))
    isempty(sym) ? string(ps.name) : String(sym)
end

"""
    padr(s, n) → String

Left-justify `s` in a field `n` columns wide, counting DISPLAY columns rather than
bytes. Julia's `%-ns` pads to a byte count, so a name containing a multi-byte glyph
(`a_ℓ` is three characters in five bytes) comes out short and drags every following
column left. Every parameter table here carries such names.
"""
padr(s, n::Int) = (t = string(s); t * " "^max(0, n - textwidth(t)))

# Fixed parameters are keyed `block_name` in the fixed NamedTuple and have no
# ParamSpec to carry a label, so their display symbols are spelled here. The market
# suffix follows the same convention as `param_symbol`, which reads it from a label.
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

Display symbol for a fixed parameter. Falls back to the key itself for the genuinely
unsuffixed ones (`r`, `ν`, `φ`), which need no market label.
"""
fixed_symbol(key) = get(FIXED_SYMBOLS, Symbol(key), string(key))

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
    # RETIRED v19.5.0, kept only so stored bundles stay loadable. The DE stall test is
    # de_reheat_flat + de_reheat_rate; this field's value never reached that test. Delete
    # it at the next struct change that migrates bundles anyway. See CHANGELOG.md.
    de_patience  :: Int     = 20
    de_avg_tol   :: Float64 = 0.0      # stop when (Q_mean−Q_best)/|Q_best| < tol; 0 disables.
                                       # Off: across a six-configuration sweep this rule
                                       # correlated +0.915 with achieved ΔQ, i.e. it ranked
                                       # the run that converged prematurely as the most
                                       # converged one.

    # Population generator.  Appended, never inserted: Serialization rebuilds a struct by
    # field position, so a field added anywhere but the end shifts every later value.
    de_gen_per_k   :: Int     = 30     # candidates drawn at each sparsity k = 1:n_free
    de_reheat_flat :: Int     = 6      # reheat once Q_best has been flat this many
                                       # generations and
    de_reheat_rate :: Float64 = 0.04   # the improved-member rate is below this
    de_max_reheats :: Int     = 50     # a reheat that finds no improvement ends the run,
                                       # so this is a guard rather than the usual stop
    de_adapt_fcr   :: Bool    = false  # take de_f and de_cr from the generator's own
                                       # draws instead of the values set above. Off by
                                       # default: the derived cr is the winning sparsity
                                       # as a fraction of the free count, which on a
                                       # near-flat allocation lands near 1/n_free and
                                       # moves one coordinate per proposal.

    # Simulated annealing
    sa_max_iter      :: Int     = 5_000
    sa_T0            :: Float64 = 2.0
    # Proposal sd in unconstrained space.  Sized to the measured ΔQ = 1 feasible
    # half-widths (median 3.4e-03) rather than to the ±4 box: at 0.20 the walk
    # accepted 0 of 150 proposals at 97% feasibility, at 0.01 it accepted 7 and
    # descended.  smm_main.jl sets this on every run (ROYSEARCH_SA_STEP); the
    # default matters only to a bundle read outside that path.
    sa_step          :: Float64 = 0.01
    sa_cooling_rate  :: Float64 = 1.0
    sa_cooling_exp   :: Float64 = 0.5
    sa_reheat_patience :: Int     = 200
    sa_reheat_factor   :: Float64 = 2.0
    sa_max_reheats     :: Int     = 5
    sa_adapt_window  :: Int     = 50
    # RETIRED v19.5.0, kept only so stored bundles stay loadable. It was meant to shrink
    # the step when feasibility fell below it, and _sa_loop never read it. The mechanism it
    # would have counteracted was measured and is not there: at realised feasibility 0.88
    # the walk closes 94.4% of the gap against 97.2% at full feasibility, and the adapted
    # step is unchanged. Delete at the next bundle-migrating struct change. See CHANGELOG.md.
    sa_target_fin    :: Float64 = 0.90
    # T0 auto-calibration (used when sa_T0 <= 0).  T0 solves
    # exp(-sa_t0_rel*Q0 / T0) = sa_t0_accept: a proposal that worsens the
    # objective by sa_t0_rel of its current level is accepted with probability
    # sa_t0_accept.  Anchored to Q rather than to probed move sizes, so it is
    # invariant to the weighting matrix and to the objective's scale.
    # 0.05 was set for a cold start and gives T0 = 48 at Q ≈ 1164 — a move of 58 in Q,
    # which is a different basin. Warm starts, which is how every window now runs, need
    # the run's own per-100-iteration gain of ~0.1 in Q as the reference scale.
    sa_t0_rel        :: Float64 = 1e-4
    sa_t0_accept     :: Float64 = 0.30
    # On a reheat, snap back to the incumbent when the walk has drifted more than
    # this fraction above Q_best; within the tolerance it reheats in place.
    sa_reheat_reset_tol :: Float64 = 0.25
    sa_random_init   :: Bool    = false
    sa_parallel_steps :: Int    = 100         # parallel SA steps before pruning to the best chain
    sa_seed           :: Int    = 20240601    # base seed for parallel SA chains (replicability)

    # Weight-matrix mode (see load_weight_matrix): 0.0 diagonal-σ, 2.0 equal weights
    w_cond_target :: Float64 = 2.0

    # Wage-reliability calibration knob. σ_w is NOT a free parameter: it is
    # calibrated externally from λ_w via σ_{w,j}² = (1−λ_w)·Var̂[log w_j]
    # (calibrate_sigma_w), so λ_w configures the run and is reported in the
    # header only — it never appears in the parameter tables.
    # smm_main.jl sets this from LAMBDA_W on every run; the default matters only to a
    # bundle read outside that path, so it tracks the value the project uses.
    λ_w :: Float64 = 0.78

    # Nelder-Mead polish.
    #
    # Optim's Nelder-Mead cannot use f_reltol or x_abstol. Its assess_convergence
    # method (nelder_mead.jl:317) returns the four flags as
    # `(false, false, g_converged, false)` — x_converged and f_converged are literals,
    # so no code path reads those two options for this solver. Sweeping f_reltol over
    # 1e-12 … 5e-1 leaves the iteration count unchanged at 111 on a smooth problem,
    # confirming it. Both fields are still passed because the same branch serves
    # :lbfgs and :bfgs, where they are live.
    #
    # nm_f_tol is set from the scale of Q rather than convention. Q is a chi-square,
    # so ΔQ = 1 is one moment moving by one sampling standard error; at Q of order
    # 1e3 that is a relative tolerance of about 1e-3. Measured solver noise is
    # ΔQ ≈ 1e-6, six orders below, so this threshold is statistical, not numerical.
    nm_max_iter  :: Int     = 10_000
    nm_f_tol     :: Float64 = 1e-3
    nm_x_tol     :: Float64 = 1e-5
    # g_abstol is Nelder-Mead's ONLY built-in stopping test, and it is not a gradient
    # test. Optim compares it against nmobjective = sqrt(var(f_simplex)·m/n), i.e. the
    # standard deviation of Q across the simplex vertices — an ABSOLUTE quantity in Q
    # units. At d = 25 that factor is 1.02, so the test is effectively sd(Q) ≤ tol.
    #
    # Left at Optim's own default. On a Q of order 1e3 it is unreachable (it demands
    # sd(Q) ≤ 1e-8, some 1e-9 % of Q), so in practice the rate rule below is what
    # stops the run; the default is kept rather than lowered so this field never
    # silently becomes a second, scale-dependent f-tolerance.
    nm_g_tol     :: Float64 = 1e-8
    nm_no_improve :: Int    = 4_000  # halt after this many evaluations with no
                                     # improvement in best Q (0 disables; units of
                                     # the [iter N] trace)

    # Tracing
    show_trace_members     :: Bool = false
    show_trace_generations :: Bool = true
    trace_stride           :: Int  = 10

    # ── Start-point construction ─────────────────────────────
    # How the optimiser's starting points are built. These were top-level constants in
    # smm_main.jl, which meant a serialised bundle did not record them and a run could
    # not be reproduced from its own output.
    use_as_seed          :: Bool    = true   # feed the warm-start optimum in as a seed
    seed_perturb_frac    :: Float64 = 0.05   # fraction of each box width to jitter it by
    clusters_force_regen :: Bool    = true   # rebuild the seed bank rather than reuse it
    include_prev_optimum :: Bool    = false  # add this window's previous optimum as a start

    # ── Rate-based stopping (SA and Nelder-Mead) ─────────────
    # Both optimisers stop on the RATE of improvement over a trailing window rather than
    # on an absolute tolerance: on a Q of order 1e3 an absolute threshold is either
    # unreachable or fires immediately, while a rate is scale-free. tol = 0 disables the
    # test and falls back to the iteration caps above.
    sa_rate_tol   :: Float64 = 0.05
    sa_rate_span  :: Int     = 300
    sa_halflife   :: Int     = 5_000   # geometric cooling half-life, in iterations
    nm_rate_tol   :: Float64 = 0.05
    nm_rate_span  :: Int     = 300

    # Nelder-Mead initial simplex edge, as a fraction of each box width. Optim's default
    # builds a simplex whose vertices are mostly infeasible here; 0.2 keeps them inside.
    nm_simplex_step :: Float64 = 0.2

    # ── Local search around incumbents ───────────────────────
    # The generator perturbs each chosen coordinate by de_local_sigma of its own ΔQ<1
    # width — a units-free fraction of a measured scale, not a fraction of the box.
    #
    # sa_subset_k was REMOVED here in v19.1.0. BUNDLE STATUS AT THE TIME OF THIS EDIT:
    # the neutral intermediates were extracted BEFORE the removal and the rebuild has run,
    # but NOTHING IS INSTALLED YET — the .new files sit beside the originals in output/smm.
    # base_fc and base_covid round-trip theta EXACTLY (max|dtheta| = 0.000e+00) and are the
    # two windows to install; crisis_fc and crisis_covid do NOT (max|dtheta| = 3.34 and
    # 4.55) because they encode a pre-P_U-normalisation 19-parameter spec and must be
    # re-estimated rather than migrated. Do not read this comment as "the bundles on disk
    # are current" until that install has happened.
    # It set a fixed number of coordinates for SA to perturb per iteration; the
    # proposal now draws its own mask from a measured p_move and takes its step scale
    # from the measured ΔQ=1 half-widths (sa_proposal_scale, SA_SCALE_* in smm_main.jl).
    # Removal required a migration because Serialization rebuilds a struct by field
    # POSITION: it sat between nm_simplex_step (Float64) and de_local_sigma (Float64),
    # so deleting an Int between two Float64s is the loud TypeError case rather than the
    # silent shift — but only by luck of the types, which is not a reason to rely on it.
    # See the graveyard entry in SETTINGS.md for the measurement that retired it.
    de_local_sigma :: Float64 = 0.33

end

# ============================================================
# RunProvenance — where a serialised bundle came from
# ============================================================

"""
Current bundle-format version. Bump only on a shape change; see `RunProvenance` and
`bundle.jl`.

3 (v56.0.0): the optimiser's coordinates are the tightnesses, and the vacancy costs are
computed from free entry rather than carried in θ, so the field set gains `vacancy_costs`
(the costs at the estimate, with their standard errors or posterior when a run produced
them) and `attracting` (the free-entry slope check at the estimate).

2 (v19.6.0): one shape for every writer, with a CONSTANT field set —
`schema, stage, result, spec, provenance, sim, se, se_source, mcmc, tag` — where absent
information is `nothing` rather than an absent field. Written by `write_bundle` only;
normalised on read by `read_bundle`, which infers `stage` for a schema-1 file.

1: four writers, four different field sets, and whichever touched a window last decided
what its file contained.
"""
const BUNDLE_SCHEMA = 3

"""
    RunProvenance

Identifies the run that produced a bundle. Serialised alongside the estimate so a
result carries its own origin: with several machines estimating different windows in
parallel, "which code produced this number" cannot be answered from the filename.

`schema` is the bundle-format version, and it is the field a loader checks. It changes
only when the top-level FIELD SET of a serialised bundle changes, not when a parameter
value does, and not when a struct carried inside it gains or loses a field: such a file
fails to deserialise before `schema` can be read, so the number cannot help there.
`@kwdef` buys nothing on read — `Serialization` fills fields positionally and never calls
the constructor — so adding a field makes existing bundles unreadable (`EOFError`) exactly
as removing one does (`TypeError`). Either way the bundles must be migrated; see
`code/tools/migrate_bundles.jl`.

`unrecorded` names fields that were reconstructed rather than read from the original
run. A bundle migrated from a version that did not store a setting carries that
setting's name here rather than silently presenting a default as fact.

`version` means "the code that last WROTE this file", so a migration necessarily
overwrites it and the version that produced the ESTIMATE would be lost. `migrated_from`
carries that: each migration appends the version the bundle arrived with, oldest first,
so a twice-migrated bundle records both hops and an un-migrated one is empty.
"""
Base.@kwdef struct RunProvenance
    schema        :: Int             = 1
    version       :: String          = "unknown"   # ROYSEARCH_VERSION at write time
    window        :: Symbol          = :unknown
    w_suffix      :: String          = ""
    git_sha       :: String          = ""          # empty when git is unavailable
    host          :: String          = ""
    julia         :: String          = string(VERSION)
    written_at    :: String          = ""          # ISO 8601, UTC
    unrecorded    :: Vector{Symbol}  = Symbol[]
    migrated_from :: Vector{String}  = String[]    # versions that produced the estimate
end

"""
    run_provenance(; window, w_suffix, version, unrecorded = Symbol[], migrated_from = String[])

Build a `RunProvenance` for the current process. `git_sha` and `host` are probed and
left empty on failure rather than raising: provenance is metadata, and a run must not
die because a machine has no git.

`migrated_from` is empty for an estimated bundle and is supplied only by
`migrate_bundles.jl`, which is the only writer that rewrites a file it did not produce.
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

# Formatted without Dates: smm_params.jl is included by the solver path too, and the
# dependency would be carried for one timestamp.
function _utc_now_iso()
    t = round(Int, time())
    days, rem = divrem(t, 86_400)
    h, rem2   = divrem(rem, 3_600)
    mi, s     = divrem(rem2, 60)
    y, mo, d  = _civil_from_days(days)
    return @sprintf("%04d-%02d-%02dT%02d:%02d:%02dZ", y, mo, d, h, mi, s)
end

# Howard Hinnant's days-from-civil inverse, for a UTC date without a date library.
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
           reported scalar Q (default 1.0).  Constant in θ, so it does
           not move the argmin.
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
  relative-deviation loss.  If a matrix is passed (the diagonal-σ
  W = Diagonal(1/σ̂²_samp) from `load_weight_matrix`) it must be square
  with size equal to the number of active moments; a mismatch raises an
  error — typically a sign that `W` was built with a different
  `skip_moments` list.

- `q_scale` is a DISPLAY-ONLY positive constant (default 1.0) that
  divides the reported scalar Q.  It is constant in θ and therefore
  does not move the argmin.
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

    # Silent: a parameter pinned here appears in the spec's own Fixed table with its value,
    # so announcing the drop separately says the same thing twice — and the ASCII `name`
    # field it printed duplicated across markets (two η, two σ_w), which read as a defect.
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

Full list of estimable parameters with their bounds.  Pass a subset to
`build_smm_spec` to estimate only some.

The entries are UNSEEDED (`init = NaN`).  Starting values have exactly two sources,
both in the estimation driver: DEFAULT_PARAMS in smm_main.jl, or a stored optimum
(warm start, crisis-window baseline).  A start written here would be a third source
that could reach a run without anyone having chosen it.

Excluded from this list because they are always fixed:
  r       calibrated externally at 4% annual → monthly r = 0.05/12
  ν       pre-estimated from CPS data (one value per crisis pair)
  φ       pre-estimated from NSC/IPEDS data (pooled)
"""
function default_free_params() :: Vector{ParamSpec}
    return [
        # Ability distribution — common block.  a_ℓ, b_ℓ shape the aU marginal;
        # ρ_x is the ability correlation, the RoySearch separation device
        # (Gola concordance).
        ParamSpec(:common, :a_ℓ,    0.1000,    8.0000,  "worker type shape a_ℓ"),
        ParamSpec(:common, :b_ℓ,    0.0500,    4.0000,  "worker type shape b_ℓ"),

        # ρ_x: must be highly negative for the estimate to be stable
        ParamSpec(:common, :ρ_x,   -0.98000,    -0.20000,  "ability correlation ρ_x"),

        # Common block — training cost coeff and aggregate scale
        ParamSpec(:common, :c,      6.00000,   15.00000,  "training cost coeff c"),
        ParamSpec(:common, :A,      4.00000,   10.00000,  "aggregate production scale A"),

        # Unemployment income and the training flow (stored by consuming block).
        ParamSpec(:unsk,   :bU,     0.000000,  0.200000, "unskilled outside flow b_U"),
        ParamSpec(:unsk,   :bT,     0.000000,  1.500000, "training flow b_T"),
        ParamSpec(:skl,    :bS,     0.000000,  0.500000, "skilled outside flow b_S"),

        # Productivity levels.  Under pure-Roy production is linear in own
        # ability, π_U = exp(A) P_U aU p and π_S = exp(A) P_S aS p (no ability
        # gradient).  Only exp(A)·P_j is pinned by wage levels,
        # so (A, P_U, P_S) share one flat direction; fix A (e.g. A=0) or one
        # P to identify them individually.  The premium loads on log(P_S/P_U).
        ParamSpec(:unsk,   :PU,     0.000016,  1.303171, "unskilled productivity P_U"),
        ParamSpec(:skl,    :PS,     1.000016,  5.000000, "skilled productivity P_S"),
        ParamSpec(:unsk,   :α_U,    0.1000,    6.0000,  "unskilled damage shape α_U"),
        ParamSpec(:skl,    :a_Γ,    0.1000,   12.0000,  "skilled offer shape a_Γ"),
        ParamSpec(:skl,    :b_Γ,    0.1000,   10.0000,  "skilled offer shape b_Γ"),
        ParamSpec(:skl,    :δ,      0.0500,    1.0000,  "shock/offer support ratio δ_S"),

        # Matching efficiency, matching elasticity, bargaining (U/S paired)
        ParamSpec(:unsk,   :μ,      0.0001,    2.0000,  "unskilled matching eff μ_U"),
        ParamSpec(:skl,    :μ,      0.0001,    2.0000,  "skilled matching eff μ_S"),
        ParamSpec(:unsk,   :η,      0.0001,    1.0000,  "unskilled matching elas η_U"),
        ParamSpec(:skl,    :η,      0.0001,    1.0000,  "skilled matching elas η_S"),
        ParamSpec(:unsk,   :β,      0.0010,    0.9000,  "unskilled bargaining β_U"),
        ParamSpec(:skl,    :β,      0.0001,    0.9000,  "skilled bargaining β_S"),

        # Shock arrival rates (U/S paired)
        ParamSpec(:unsk,   :λ,      0.0001,    0.5500,  "unskilled damage rate λ_U"),
        ParamSpec(:skl,    :λ,      0.0001,    0.5000,  "skilled quality shock λ_S"),

        # Market tightnesses (U/S paired), vacancies per searcher.  Estimated in place of
        # the vacancy costs: given (θ_U, θ_S) the stationary state is unique, and free entry
        # gives each cost in closed form (solve_at_tightness!, solver.jl), while the map
        # from costs to tightnesses is a correspondence.  The costs remain estimated
        # parameters: every evaluation computes them from free entry, in months of the
        # mean output of a new match (mean_output_U/S).
        ParamSpec(:unsk,   :tight,  0.0500,    3.0000,  "unskilled tightness θ_U"),
        ParamSpec(:skl,    :tight,  0.1000,    5.0000,  "skilled tightness θ_S"),

        # OJS cost and the two exogenous separation baselines.  δ_S compresses the
        # λ_S-shock redraw onto [0,δ_S]; it carries endogenous separation and the EE
        # ladder (both dead at δ_S = 1, the single-distribution limit).  Each ξ_j is a
        # quality-independent hazard on top of the endogenous margin λ_j·G_j(p*_j):
        # (ξ_S, δ_S, λ_S) jointly split sep_rate_S / ee_rate_S into baseline and
        # endogenous margins, and ξ_U gives sep_rate_U a lever the endogenous part
        # alone cannot reach.  Same box for both sectors.
        ParamSpec(:skl,    :σ,      0.0000,    0.5000,  "OJS flow cost σ_S"),
        ParamSpec(:unsk,   :ξ,      0.0000,    0.0200,  "unskilled exogenous separation ξ_U"),
        ParamSpec(:skl,    :ξ,      0.0000,    0.0200,  "skilled exogenous separation ξ_S"),

        # Wage measurement error (per sector; pinned when λ_w > 0)
        ParamSpec(:unsk,   :σ_w,    0.0000,    1.0000,  "unskilled wage meas. error σ_wU"),
        ParamSpec(:skl,    :σ_w,    0.0000,    1.0000,  "skilled wage meas. error σ_wS"),
    ]
end

# The vacancy costs, estimated through the tightnesses: every evaluation computes them
# (solve_at_tightness!) and every table lists them beside the free parameters, keyed
# (block, :k) and labelled here, in months of the mean output of a new match.
const VACANCY_COSTS = ((:unsk, "unskilled vacancy cost k_U (months of new-match U output)"),
                       (:skl,  "skilled vacancy cost k_S (months of new-match S output)"))

# The joint ability distribution.  A crisis window holds it at the estimate of its
# paired baseline window, since the ability composition of prime-age workers does
# not move within a pair; every other parameter is a candidate for the free set in
# every window.  A tuple, so iteration order (the field order of the pinned block,
# which enters the candidate-cache key) is fixed.
const ABILITY_PARAMS = ((:common, :a_ℓ), (:common, :b_ℓ), (:common, :ρ_x))

# ============================================================
# Fixed-key convention
# ============================================================

# Field names carried by more than one block (μ, η, tight, β, λ, ξ, σ_w).  Derived
# from default_free_params() rather than written out, so adding a parameter
# cannot leave a stale list behind.
const _SHARED_PARAM_NAMES = let counts = Dict{Symbol,Int}()
    for ps in default_free_params()
        counts[ps.name] = get(counts, ps.name, 0) + 1
    end
    Set(nm for (nm, n) in counts if n > 1)
end

"""
    fixed_key(block, name) → Symbol

The key under which to WRITE a pinned value into `SMMSpec.fixed`.  `unpack_θ`
resolves a fixed entry by trying the block-qualified key `<block>_<name>` first
and the bare `name` second, so a name carried by only one block is written bare
— the convention already used for r, ν, φ, bU, bT, bS, α_U, … — while a name
shared by the unskilled and skilled blocks must be qualified, or pinning one
block's copy would pin the other's too.
"""
fixed_key(block::Symbol, name::Symbol) :: Symbol =
    name in _SHARED_PARAM_NAMES ? Symbol(string(block) * "_" * string(name)) : name

"""
    is_pinned(fixed, block, name) → Bool

Whether (block, name) is pinned in a fixed NamedTuple.  Both key forms count,
qualified and bare, matching how `unpack_θ` READS a fixed entry: a caller is
free to qualify a block-unique name (FIX_PARAMS does so for σ_S and δ_S) and
`unpack_θ` still finds it.
"""
is_pinned(fixed, block::Symbol, name::Symbol) :: Bool =
    haskey(fixed, Symbol(string(block) * "_" * string(name))) || haskey(fixed, name)

"""
    assert_all_params_accounted(spec; context = "spec")

Check that every entry of `default_free_params()` is either estimated (in
`spec.free`) or pinned (in `spec.fixed`), and raise if any is neither.

`unpack_θ` raises on such a parameter too, but only when it first decodes θ and
only for the first field it cannot resolve.  This check runs at spec construction,
before any solve, and lists every unaccounted parameter at once.
"""
function assert_all_params_accounted(spec::SMMSpec; context::String = "spec")
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
# The estimation entry point configures runs with ASCII keys (:a_l, :rho_x,
# :unsk_bet) so the file is editable without Unicode input; the solver structs and
# ParamSpec table use the Unicode names (:a_ℓ, :ρ_x, :β). These maps are the single
# translation layer between the two conventions, and they live beside the ParamSpec
# table so a parameter added there is renamed in the same file.
# ============================================================

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

# ASCII key (FIX_PARAMS convention) → unicode fixed-NamedTuple key.

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

# The vacancy costs follow from the tightnesses through free entry, so neither can be pinned.
const _COST_KEYS = (:unsk_k, :skl_k)

"""
    _fix_params_to_nt(fix_dict) → NamedTuple
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

Read one parameter out of a decoded optimum (`unpack_θ`'s `cp`, `up`, `sp` and the
tightnesses `tight`).  The (block, name) method serves callers that iterate ABILITY_PARAMS,
which holds bare pairs rather than ParamSpecs.
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

"""
    logjac_box(θ_unc, free) -> Float64

Log |dθ/dt| of the box transform, summed over free parameters:
`log(ub−lb) + log σ(t) + log(1−σ(t))`.

A sampler moving in `t` must add this to the log-target, or the density it
induces on θ is `∝ e^{−Q/2} / [(θ−lb)(ub−θ)]` — U-shaped with poles at both
bounds instead of flat on the box.  That is not a cosmetic prior: since Q tends
to a finite limit as a parameter saturates, the un-Jacobianed target is not
integrable in `t`, so it has no stationary distribution and a flat direction
diffuses without bound.  Adding this term makes it proper (∫σ(1−σ)dt = 1) and
satisfies the Chernozhukov–Hong (2003) Assumption 2(iv) requirement of a
continuous, uniformly positive prior.  Written with `logsigmoid` so the tails
stay finite in floating point.
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

Return the init values of a free-parameter list — or of `spec.free` — as an
unconstrained vector.  The list method is what maps an arbitrary set of
constrained values onto a spec's own parameter ordering (build the list with
those values as inits, then pack), keeping one implementation of the transform.

Raises on a non-finite init, naming the parameter: the table builds every entry
unseeded, so a NaN here means the caller never wrote a starting value in.
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

Return `free_specs` with each init jittered by a Gaussian of SD `frac·(ub−lb)`,
clamped to the open interval. Used to start the optimiser from a perturbation
of the entered seed rather than the seed itself.
"""
function perturb_free_init(free_specs::Vector{ParamSpec}, frac::Float64, rng) :: Vector{ParamSpec}
    return [with_seed(ps, ps.init + frac * (ps.ub - ps.lb) * randn(rng)) for ps in free_specs]
end

"""
    unpack_θ(θ_unc, spec) → (CommonParams, UnskilledParams, SkilledParams, tight)

Convert an unconstrained free-parameter vector back to the three
model structs and the tightnesses `tight = (U = θ_U, S = θ_S)`, merging with any fixed
values.  PU, bU, bT and α_U live in `UnskilledParams`; PS, bS, a_Γ and b_Γ live in
`SkilledParams`.

Every field comes from the spec: pinned (block-qualified key, then bare) or free.
A field that is neither raises, naming it — there is no fallback value.  The vacancy
costs are not parameters: the blocks carry `k = NaN` until the solve at `θ` derives
them (`solve_model(...; tightness = tight)`), so a cost read before that is NaN, not a number.
"""
function unpack_θ(
    θ_unc :: AbstractVector{Float64},
    spec  :: SMMSpec;
    constrained :: Bool = false
)
    # 1. Constrained free values, keyed the way `fixed_key` keys a pinned one: bare for a
    #    name unique to one block, block-qualified for a name both blocks carry (μ, η, tight,
    #    β, λ, ξ, σ_w), so the unskilled and skilled copies cannot stand in for each other.
    #
    # `constrained = true` says the caller passed θ itself rather than the unconstrained t,
    # so the box map is skipped. This exists for MCMC_SPACE = :theta (v21.0.0), which samples
    # the natural parameter directly. It is NOT equivalent to mapping θ→t→θ around the
    # existing path: logit((θ-lb)/(ub-lb)) underflows for a θ within ~1e-300 of a bound and
    # the round trip returns the bound exactly, destroying precision in precisely the
    # coordinates that motivated the change. Skipping the map keeps θ bit-exact.
    free_vals = Dict{Symbol, Float64}()
    for (i, ps) in enumerate(spec.free)
        free_vals[fixed_key(ps.block, ps.name)] =
            constrained ? θ_unc[i] : _to_constrained(θ_unc[i], ps.lb, ps.ub)
    end

    # 2. Resolve one field.  A pinned value takes priority over a free one, matching
    #    `is_pinned`, which drops a pinned parameter from the free list.  A field in
    #    neither set is what assert_all_params_accounted rejects at spec construction; it
    #    reaches here only through a spec built another way — typically a stored bundle
    #    written under a different free set — and is an error, not a substituted value.
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
        A   = _get(:A,   :common),   # log scale; model uses exp(A)
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
    # The market-suffixed symbol, not the label: these tables are read against each
    # other, and a description column wide enough for the longest label pushes the
    # numbers out of alignment. The labels survive in the CSV writers, where width
    # does not matter.
    @printf("  %s%s%10s  %10s  %10s\n",
            padr("block", 8), padr("param", 8), "lb", "ub", "seed")
    @printf("  %s\n", "─"^50)
    for ps in spec.free
        @printf("  %s%s%10.4f  %10.4f  %10.4f\n",
                padr(ps.block, 8), padr(param_symbol(ps), 8), ps.lb, ps.ub, ps.init)
    end

    if length(spec.fixed) > 0
        @printf("\n╠══════════════════════════════════════════════════════╣\n")
        @printf("║  Fixed parameters (%d)                                ║\n",
                length(spec.fixed))
        @printf("╠══════════════════════════════════════════════════════╣\n")
        for (k, v) in pairs(spec.fixed)
            @printf("  %s = %.6f\n", padr(fixed_symbol(k), 20), v)
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
        @printf("║  Skipped moments (%2d, weight = 0)                    ║\n", length(skipped_moments))
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
    # sa_target_fin and de_patience were printed here until v19.5.0 and neither reached a
    # consumer that read it — a banner asserting settings nothing acted on. They survive
    # as struct fields only so stored bundles stay loadable; see CHANGELOG.md.
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
    # REQUESTED, not effective — every line here reads spec.run, which is what the
    # bundle records, not what the optimiser received. The two can differ: these
    # settings reach run_smm as keyword arguments, and an omitted one takes run_smm's
    # own default in silence. The authoritative line is [SA config], printed from
    # inside _sa_loop off the values it actually holds; if the two disagree, believe
    # [SA config] and fix the forwarding.
    # The proposal itself is not reported here: it is measured at the start point, so
    # the only honest source for it is [SA config], printed from inside the walk.
    @printf("  SA (requested):   cooling %s;  stop ΔQ < %s per 100 iters\n",
            spec.run.sa_halflife > 0 ?
                @sprintf("geometric, half-life %d", spec.run.sa_halflife) : "logarithmic",
            spec.run.sa_rate_tol > 0 ? @sprintf("%.3g", spec.run.sa_rate_tol) : "off")
    @printf("  NM (requested):   stop ΔQ < %s per 100 evals, sustained over %d evals\n",
            spec.run.nm_rate_tol > 0 ? @sprintf("%.3g", spec.run.nm_rate_tol) : "off",
            spec.run.nm_rate_span)
    @printf("  DE:   population — %d draws at each k=1:%d at %.2f×measured width; slots and f,cr from yield\n",
            spec.run.de_gen_per_k, length(spec.free), spec.run.de_local_sigma)
    if spec.run.w_cond_target == 2.0
        @printf("  W:    equal weights — Diagonal(weight/m̂²) (cond(W)=%.2e)\n", cond(spec.W))
    else
        @printf("  W:    diagonal-σ — Diagonal(1/σ̂²_samp) (cond(W)=%.2e)\n", cond(spec.W))
    end
    @printf("  Q scale (display-only divisor): %.6e\n", spec.q_scale)
    @printf("╚══════════════════════════════════════════════════════╝\n\n")
end
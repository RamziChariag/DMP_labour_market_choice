############################################################
# smm/bundle.jl — the estimate bundle: one shape, one writer, one reader
#
# A bundle is a NamedTuple serialised to estimate_<window><suffix>_<point>.jls
# (`estimate_path`, paths.jl). Every field is always present; information a writer
# does not produce is `nothing`. Fields:
#
#   schema         BUNDLE_SCHEMA (smm_params.jl)
#   stage          what the file is (BUNDLE_STAGES):
#                    :smm             finished smm_main run; SEs only once MCMC_main's
#                                     chain-free pass has added them
#                    :smm_checkpoint  incumbent written at an SA or DE reheat; resumable,
#                                     not an estimate
#                    :mcmc_checkpoint best point a chain visited (MCMC_CHECKPOINT); a record
#                    :mcmc            the chain's estimate, the posterior mean of the draws
#                    :mcmc_aborted    chain hit the acceptance floor or drift abort; never an
#                                     estimate, but its Jacobian SEs are kept
#                    :superseded      the point no longer solves under the current spec
#   result, spec   SMMResult and SMMSpec (positional structs: see smm_params.jl)
#   provenance     RunProvenance
#   sim            SimParams of the run (final :smm bundles only)
#   se, se_source  standard-error columns and how they were produced: (sandwich, bound,
#                  chain) with the chain covariance as bread (:chain_bread_fd/_design), or
#                  (curvature, bound, chain) with (Ĝ′WĜ)⁻¹ (:jacobian_fd/_design)
#   mcmc           chain record, including the file it was seeded from
#   vacancy_costs  (k = (U, S), se, post): the costs at the estimate, their delta-method
#                  SEs and the chain's posterior summary
#   attracting     (D, market, joint) of `attracting_check` (smm.jl)
#   tag            free-text label (checkpoints)
#
# Entry points: write_bundle, read_bundle, require_current_estimate, bundle_summary.
############################################################

const BUNDLE_STAGES = (:smm, :smm_checkpoint, :mcmc_checkpoint, :mcmc, :mcmc_aborted,
                       :superseded)

"""
    provenance_from_path(path) → RunProvenance

Provenance for a writer that does not know its window (`write_checkpoint`), parsed from the
bundle filename. Every filename generation is recognised (`smm_result_<w><suffix>.jls`,
`estimate_<w><suffix>.jls`, `estimate_<w><suffix>_<point>.jls`); an unrecognised name
records `window` and `w_suffix` as `unrecorded` rather than guessing them.
"""
function provenance_from_path(path::AbstractString)
    m = match(r"^(?:estimate|smm_result)_(.+?)(_[a-zA-Z]+W)(?:_[a-z_]+)?\.jls$", basename(path))
    m === nothing && return run_provenance(window = :unknown, w_suffix = "",
                                           version = ROYSEARCH_VERSION,
                                           unrecorded = [:window, :w_suffix])
    return run_provenance(window = Symbol(m.captures[1]), w_suffix = String(m.captures[2]),
                          version = ROYSEARCH_VERSION)
end

"""
    write_bundle(path; result, spec, stage, provenance, sim, se, se_source, mcmc,
                 vacancy_costs, attracting, tag) → path

Serialise an estimate bundle to `path` in the documented shape and return `path`; an empty
`path` writes nothing. `stage` must be one of `BUNDLE_STAGES`. The write goes to a temporary
file and is then moved into place, so a kill mid-write leaves the previous bundle intact.
"""
function write_bundle(path::AbstractString;
                      result,
                      spec,
                      stage::Symbol,
                      provenance,
                      sim       = nothing,
                      se        = nothing,
                      se_source = nothing,
                      mcmc      = nothing,
                      vacancy_costs = nothing,
                      attracting = nothing,
                      tag::AbstractString = "")
    stage in BUNDLE_STAGES ||
        error("write_bundle: unknown stage :$stage — must be one of $(BUNDLE_STAGES)")
    isempty(path) && return path

    bundle = (schema     = BUNDLE_SCHEMA,
              stage      = stage,
              result     = result,
              spec       = spec,
              provenance = provenance,
              sim        = sim,
              se         = se,
              se_source  = se_source,
              mcmc       = mcmc,
              vacancy_costs = vacancy_costs,
              attracting = attracting,
              tag        = String(tag))

    mkpath(dirname(path))
    tmp = path * ".tmp"
    open(tmp, "w") do io
        serialize(io, bundle)
    end
    mv(tmp, path; force = true)
    return path
end

"""
    read_bundle(path) → NamedTuple | nothing

Load the bundle at `path`, or `nothing` when the file is absent or cannot be deserialised.
A schema-1 file is normalised to the current shape, with `stage` inferred conservatively:
`:smm_checkpoint` if it carries the checkpoint flag, `:smm` if it carries `sim`,
`:smm_checkpoint` otherwise.
"""
function read_bundle(path::AbstractString)
    isfile(path) || return nothing
    raw = try
        open(deserialize, path)
    catch
        return nothing
    end
    hasproperty(raw, :schema) && raw.schema >= 2 && return raw

    stage = hasproperty(raw, :checkpoint) && raw.checkpoint ? :smm_checkpoint :
            hasproperty(raw, :sim)                          ? :smm            :
                                                              :smm_checkpoint
    return (schema     = 1,
            stage      = stage,
            result     = hasproperty(raw, :result)     ? raw.result     : nothing,
            spec       = hasproperty(raw, :spec)       ? raw.spec       : nothing,
            provenance = hasproperty(raw, :provenance) ? raw.provenance : nothing,
            sim        = hasproperty(raw, :sim)        ? raw.sim        : nothing,
            se         = nothing,
            se_source  = nothing,
            mcmc       = nothing,
            vacancy_costs = nothing,
            attracting = nothing,
            tag        = hasproperty(raw, :tag)        ? String(raw.tag) : "")
end

"""
    bundle_has_se(b) → Bool

Whether bundle `b` carries standard errors a table may print: an `:smm` bundle with the
chain-free pass's standard errors, or a chain's `:mcmc` / `:mcmc_aborted` bundle. Under
`:mcmc_aborted` a table must also print the abort reason, R̂ and the accepted-move count
(see the footer in MCMC_main.jl).
"""
bundle_has_se(b) = b.se !== nothing && b.stage in (:smm, :mcmc, :mcmc_aborted)

"""
    _major(v) → Int | nothing

The MAJOR component of a version string `"M.m.p"`.
"""
_major(v::AbstractString) = tryparse(Int, first(split(v, '.')))

# Relative agreement with its stored Q at which a bundle of another MAJOR version is read as
# an estimate under this code: rounding, not a change of equilibrium.
const REPRODUCE_RTOL = 1.0e-10

"""
    require_current_estimate(b, path) → evaluation | nothing

Raise unless bundle `b` (read from `path`) is an estimate under this code. A bundle written
by this MAJOR version is; one written by another is if its stored parameter vector,
re-evaluated with its own spec (`equilibrium_at`), reproduces its stored Q to within
`REPRODUCE_RTOL`. Nothing is written. Returns the evaluation when one was made, `nothing`
otherwise.
"""
function require_current_estimate(b, path::AbstractString)
    p = hasproperty(b, :provenance) ? b.provenance : nothing
    v = p === nothing ? "?" : p.version
    _major(v) == _major(ROYSEARCH_VERSION) && return nothing
    Q_stored = b.result === nothing ? NaN : b.result.loss_opt
    ev  = (b.result === nothing || b.spec === nothing) ? nothing :
          equilibrium_at(b.result.theta_opt, b.spec)
    Q   = ev === nothing ? NaN : ev.Q
    rel = abs(Q / Q_stored - 1.0)
    if isfinite(Q) && rel <= REPRODUCE_RTOL
        @printf("%s: written by v%s, Q reproduces under v%s (rel %.1e); accepted\n",
                basename(path), v, ROYSEARCH_VERSION, rel)
        flush(stdout)
        return ev
    end
    error("$(basename(path)) was written by v$v; v$(ROYSEARCH_VERSION) reads estimates of " *
          "its own MAJOR version only — re-estimate this window first. " *
          @sprintf("Q stored %.10f, Q under v%s %.10f.", Q_stored, ROYSEARCH_VERSION, Q))
end

"""
    bundle_summary(b) → String

One log line for bundle `b`: stage, stored Q, free-parameter count, the version that wrote
the file (and, for a migrated bundle, the versions it came through), the standard-error
source and the tag.
"""
function bundle_summary(b)
    q = b.result === nothing ? NaN : b.result.loss_opt
    d = b.spec   === nothing ? -1  : length(b.spec.free)
    v = b.provenance === nothing ? "?" : b.provenance.version
    from = b.provenance === nothing || isempty(b.provenance.migrated_from) ? "" :
           " from=" * join(b.provenance.migrated_from, "→")
    return @sprintf("stage=%s Q=%s d=%d v=%s%s se=%s%s",
                    b.stage,
                    isfinite(q) ? @sprintf("%.6f", q) : string(q),
                    d, v, from,
                    bundle_has_se(b) ? string(b.se_source) : "none",
                    isempty(b.tag) ? "" : " tag=\"" * b.tag * "\"")
end

############################################################
# bundle.jl — ONE shape for the estimate bundle, one writer, one reader.
#
# THE PROBLEM THIS SOLVES. Before v19.6.0 four different sites serialised an estimate
# bundle and each wrote a different field set:
#
#   smm_main.jl   (SMM final)        result, spec, sim, provenance
#   smm.jl        (SA/DE checkpoint) result, spec, checkpoint, tag
#   MCMC_main.jl  (MCMC checkpoint)  result, spec
#   MCMC_main.jl  (posterior mean)   result, spec, provenance
#
# While MCMC overwrote the SMM bundle in place, whichever writer touched a window last
# decided what the file contained. The audit that prompted this found
# smm_result_base_fc_diagonalW.jls carrying TWO fields, written by the MCMC checkpoint: no
# provenance, so a table could not record which version produced the number, and no
# marker, so a table could not tell an optimiser point from a sampled one. Since v57.0.0
# each estimate has a file of its own (`estimate_path`, paths.jl), and no run writes into
# another's: smm_main writes the SMM estimates, a chain run the chain's estimate, and the
# chain-free pass only adds its standard errors to the SMM bundle it read.
#
# WHY A CONSTANT FIELD SET RATHER THAN OPTIONAL FIELDS. Every field below is always
# present; absent information is `nothing`. A reader then asks `b.se === nothing` instead
# of `hasproperty(b, :se)`, and the difference matters: `hasproperty` returning false is
# ambiguous between "this writer does not produce standard errors" and "this bundle predates
# the field", and those call for different handling. A constant shape removes the ambiguity.
#
# WHY THIS IS SAFE FOR EXISTING FILES. The bundle is a NamedTuple, read by name, so adding
# fields cannot break an old file — an old bundle simply lacks the new names, and
# `read_bundle` below normalises it to this shape in memory. Adding a field to SMMResult
# would NOT be safe: that is a struct, read positionally by the deserialiser, and a field
# added anywhere in it makes every stored bundle unreadable. Nothing here touches SMMResult.
#
# THE `stage` FIELD is the one that makes a plotting layer possible. It answers, without
# probing, the only two questions a consumer has: is this a finished estimate or a mid-run
# resume artifact, and are there standard errors in it.
#
#   :smm             smm_main finished. sim present. Standard errors only once the
#                    chain-free pass of MCMC_main has added its own (se_source
#                    :jacobian_fd or :jacobian_design, after MCMC_JAC_METHOD).
#   :smm_checkpoint  written at an SA or DE reheat. Resumable, not final.
#   :mcmc_checkpoint the best point a chain visited, in estimate_<w>_mcmcbest.jls, written
#                    only with MCMC_CHECKPOINT = true. A record, never an estimate.
#   :mcmc            the chain's estimate: the posterior mean of the retained draws, with
#                    the chain's standard errors and diagnostics.
#   :mcmc_aborted    the same file when the chain hit the acceptance floor or the drift
#                    abort. The draws are not a stationary sample, so no consumer takes
#                    this file as an estimate.
#   :superseded      the point no longer solves under the current specification. Present so
#                    a stale file can say so instead of being silently read as an estimate.
#
# A chain run records the file it was seeded from as `mcmc.seed`. RunProvenance is a
# struct, read positionally, so a field added to it would make every stored bundle
# unreadable; the mcmc NamedTuple is read by name.
#
# `vacancy_costs` holds the vacancy costs at the estimate. They are estimated parameters,
# but the optimiser's coordinates are the tightnesses and the solve computes the costs from
# free entry, so they are not in θ and are stored here: `(k = (U, S), se, post)`, with `se`
# their delta-method standard errors (`(curvature, bound, chain)`, each `(U, S)`) and `post`
# the chain's posterior summary, each `nothing` when the writer did not produce it. `attracting` is
# the free-entry slope check at the estimate, `(D, market, joint)` (smm.jl,
# `attracting_check`), or `nothing`.
#
# `se` is a NamedTuple of columns (e.g. `(jinv = [...], bound = [...])`) or `nothing`, and
# `se_source` names how they were produced. An ABORTED chain still yields Jacobian-based
# standard errors, so the refusal to report belongs at the table, not here: a bundle that
# discards a computed number destroys information, while a table that prints one without its
# caveat misleads. Both failures are avoidable; only one is avoidable here.
############################################################

# BUNDLE_SCHEMA lives in smm_params.jl, where it already existed with exactly this meaning
# ("Current bundle-format version. Bump only on a shape change") and feeds
# RunProvenance.schema. Defining a second one here shadowed it: Julia warned on every run,
# and provenance.schema would have reported 1 or 2 depending on include order — a value
# declared in one place and effective in another, which is the defect class check_forwarding
# exists for. It is bumped to 2 at its own definition instead.

const BUNDLE_STAGES = (:smm, :smm_checkpoint, :mcmc_checkpoint, :mcmc, :mcmc_aborted,
                       :superseded)

"""
    provenance_from_path(path) -> RunProvenance

Build provenance for a writer that does not know which window it is writing.

`write_checkpoint` is called from inside `_run_sa` and `_run_de`, neither of which receives
the window: `SMMSpec` carries no window field, and threading one down two call layers to
label a resume artifact is not worth the signature churn. The filename is constructed at
exactly one place — `estimate_path` in paths.jl — so parsing it back is reliable.

Every filename generation is matched: `smm_result_<window><suffix>.jls` (before v19.6.0),
`estimate_<window><suffix>.jls` (before v57.0.0) and the tagged
`estimate_<window><suffix>_<point>.jls`, so an old bundle still yields its window rather
than losing it.

When neither pattern matches, `window` and `w_suffix` are listed in `unrecorded` rather than
guessed. `run_provenance` supplies the git SHA, host and timestamp itself, so those are never
fabricated here either. Before v19.6.0 the checkpoint carried no provenance at all while its
own docstring claimed it did — which is how a resume artifact became indistinguishable from
a finished estimate on disk.
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
                 vacancy_costs, attracting, tag)

Serialise an estimate bundle in the one documented shape, atomically.

`stage` must be one of `BUNDLE_STAGES`; an unknown value is an error rather than a warning,
because a bundle whose stage a consumer cannot interpret is worse than no bundle.

The write is temp-then-`mv`. That is not cosmetic: MCMC overwrites the SMM bundle in place,
so the file being written is the only copy of the incumbent, and a kill mid-write must leave
the previous bundle intact rather than a truncated file where the warm start looks.
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
    read_bundle(path) -> NamedTuple | nothing

Load a bundle and normalise a schema-1 file to the current shape. Returns
`nothing` when the file is absent or cannot be deserialised — a caller that needs the
distinction should check `isfile` first.

`stage` is INFERRED for a pre-v2 file, from the fields that writer left behind:

  has `checkpoint`  -> :smm_checkpoint   (only write_checkpoint set that flag)
  has `sim`         -> :smm             (only smm_main's final write carried sim)
  otherwise         -> :smm_checkpoint

The fallback is deliberately the conservative one. A two-field bundle could have come from
either the MCMC checkpoint or an early writer, and calling it a checkpoint understates it at
worst; calling it a finished estimate would let a table report it as one.
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
    bundle_has_se(b) -> Bool

Whether `b` carries standard errors a table may print: an `:smm` bundle the chain-free
pass has added its standard errors to, or a chain's `:mcmc` / `:mcmc_aborted` bundle. A
table printing them under `:mcmc_aborted` must also print the abort reason, the realised
R-hat and the accepted-move count; see the footer in MCMC_main.jl for the wording that has
to travel with the number.
"""
bundle_has_se(b) = b.se !== nothing && b.stage in (:smm, :mcmc, :mcmc_aborted)

"""
    bundle_is_final(b) -> Bool

Whether `b` is a finished estimate rather than a mid-run resume artifact. A plotting or
table script should refuse a non-final bundle: a checkpoint is a point the run had reached,
not a point it stopped at, and a figure captioned as an estimate should not be built from one.
"""
bundle_is_final(b) = b.stage in (:smm, :mcmc, :mcmc_aborted)

_major(v::AbstractString) = tryparse(Int, first(split(v, '.')))

"""
    require_current_major(b, path)

Stop unless the bundle at `path` was written by this MAJOR version. A MAJOR bump moves Q at
every point, so an estimate written under another one is not an estimate under this code;
such a file is read only as a warm start, which re-evaluates it.
"""
function require_current_major(b, path::AbstractString)
    p = hasproperty(b, :provenance) ? b.provenance : nothing
    v = p === nothing ? "?" : p.version
    _major(v) == _major(ROYSEARCH_VERSION) || error(
        "$(basename(path)) was written by v$v; v$(ROYSEARCH_VERSION) reads estimates of " *
        "its own MAJOR version only — re-estimate this window first.")
    return nothing
end

"""
    bundle_summary(b) -> String

One line for a log or a banner: stage, Q, free-parameter count, version, and whether
standard errors are present. Printed wherever a bundle is loaded, so a run's log records
which kind of file it actually opened.

`v=` is the version that WROTE the file. A migrated bundle was written by a version that
did not produce its estimate, so the chain it came through is printed beside it — without
that, the one line a reader sees would name the wrong code.
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

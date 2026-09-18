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
# Since MCMC overwrites the SMM bundle in place — that is the design, not an accident —
# whichever writer touched a window last decided what the file contained. The audit that
# prompted this found smm_result_base_fc_diagonalW.jls carrying TWO fields, written by the
# MCMC checkpoint: no provenance, so a table could not record which version produced the
# number, and no marker, so a table could not tell an optimiser point from a sampled one.
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
#   :smm             smm_main finished. sim present. No standard errors.
#   :smm_checkpoint  written at an SA or DE reheat. Resumable, not final.
#   :mcmc_checkpoint written when a chain found a better point mid-run. Resumable.
#   :mcmc            a chain terminated on the stop rule. Standard errors reportable.
#   :mcmc_aborted    a chain hit the acceptance floor or the drift abort. theta_best is
#                    real and better than the seed, so the POINT is the deliverable; the
#                    draws are not a stationary sample. Standard errors from a local-design
#                    Jacobian are still carried — they do not depend on the chain (CH Thm 4)
#                    — but `se_source` says so and a table must print the caveat with them.
#   :mcmc_postmean   the posterior mean over retained draws. Kept for the record; averaging
#                    over a basin is not the reported estimate.
#   :superseded      the point no longer solves under the current specification. Present so
#                    a stale file can say so instead of being silently read as an estimate.
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
                       :mcmc_postmean, :superseded)

"""
    provenance_from_path(path) -> RunProvenance

Build provenance for a writer that does not know which window it is writing.

`write_checkpoint` is called from inside `_run_sa` and `_run_de`, neither of which receives
the window: `SMMSpec` carries no window field, and threading one down two call layers to
label a resume artifact is not worth the signature churn. The filename is constructed at
exactly one place — `estimate_path` in paths.jl — so parsing it back is reliable.

Both filename generations are matched. v19.6.0 renamed the bundle from
`smm_result_<window><suffix>.jls` to `estimate_<window><suffix>.jls`, because `smm_result`
became a wrong name once MCMC started overwriting the same file; the old pattern stays here
so a bundle written before the rename still yields its window rather than losing it.

When neither pattern matches, `window` and `w_suffix` are listed in `unrecorded` rather than
guessed. `run_provenance` supplies the git SHA, host and timestamp itself, so those are never
fabricated here either. Before v19.6.0 the checkpoint carried no provenance at all while its
own docstring claimed it did — which is how a resume artifact became indistinguishable from
a finished estimate on disk.
"""
function provenance_from_path(path::AbstractString)
    m = match(r"^(?:estimate|smm_result)_(.+?)(_[a-zA-Z]+W)\.jls$", basename(path))
    m === nothing && return run_provenance(window = :unknown, w_suffix = "",
                                           version = ROYSEARCH_VERSION,
                                           unrecorded = [:window, :w_suffix])
    return run_provenance(window = Symbol(m.captures[1]), w_suffix = String(m.captures[2]),
                          version = ROYSEARCH_VERSION)
end

"""
    write_bundle(path; result, spec, stage, provenance, sim, se, se_source, mcmc, tag)

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

Load a bundle and normalise it to the v2 shape, whatever version wrote it. Returns
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
            tag        = hasproperty(raw, :tag)        ? String(raw.tag) : "")
end

"""
    bundle_has_se(b) -> Bool

Whether `b` carries standard errors a table may print. True for `:mcmc` and
`:mcmc_aborted` when `se` is populated — the second because a local-design Jacobian does
not depend on the chain having converged. A table printing them under `:mcmc_aborted` must
also print the abort reason, the realised R-hat and the accepted-move count; see the footer
in MCMC_main.jl for the wording that has to travel with the number.
"""
bundle_has_se(b) = b.se !== nothing && b.stage in (:mcmc, :mcmc_aborted)

"""
    bundle_is_final(b) -> Bool

Whether `b` is a finished estimate rather than a mid-run resume artifact. A plotting or
table script should refuse a non-final bundle: a checkpoint is a point the run had reached,
not a point it stopped at, and a figure captioned as an estimate should not be built from one.
"""
bundle_is_final(b) = b.stage in (:smm, :mcmc, :mcmc_aborted, :mcmc_postmean)

"""
    bundle_summary(b) -> String

One line for a log or a banner: stage, Q, free-parameter count, version, and whether
standard errors are present. Printed wherever a bundle is loaded, so a run's log records
which of the seven kinds of file it actually opened.
"""
function bundle_summary(b)
    q = b.result === nothing ? NaN : b.result.loss_opt
    d = b.spec   === nothing ? -1  : length(b.spec.free)
    v = b.provenance === nothing ? "?" : b.provenance.version
    return @sprintf("stage=%s Q=%s d=%d v=%s se=%s%s",
                    b.stage,
                    isfinite(q) ? @sprintf("%.6f", q) : string(q),
                    d, v,
                    bundle_has_se(b) ? string(b.se_source) : "none",
                    isempty(b.tag) ? "" : " tag=\"" * b.tag * "\"")
end

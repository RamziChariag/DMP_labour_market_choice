#!/usr/bin/env julia
#
# migrate_bundles.jl — carry serialised SMM bundles across a struct change.
#
#   julia code/tools/migrate_bundles.jl extract    # BEFORE editing the structs
#   julia code/tools/migrate_bundles.jl rebuild    # AFTER editing the structs
#   julia code/tools/migrate_bundles.jl verify     # check the rebuilt bundles
#
# WHY TWO STAGES
#
# Julia's Serialization reconstructs a struct BY FIELD POSITION: the stream records the
# field count and types, and the reader fills fields in order without ever calling the
# constructor. `Base.@kwdef` defaults are therefore never consulted on read, and ADDING
# a field to SMMRunParams makes every existing bundle unreadable — the reader expects
# more data than the file holds and raises EOFError.
#
# So a bundle cannot be read by the code that will hold it next. `extract` runs against
# the OLD struct definitions and writes a version-neutral intermediate of plain
# NamedTuples and arrays, which carry no struct identity and survive any struct change.
# `rebuild` then runs against the NEW definitions and reconstructs from that.
#
# THE PROCEDURE, in order. Do not skip step 1.
#
#   1. On the CURRENT code, before touching any struct:  migrate_bundles.jl extract
#   2. Edit the structs.
#   3. On the new code:                                  migrate_bundles.jl rebuild
#   4. Confirm:                                          migrate_bundles.jl verify
#   5. Only then delete the .pre_migration backups.
#
# THE ACCEPTANCE TEST is: same θ, same code, same Q, before and after the struct edit. A
# struct change moves no economics, so the objective must be unmoved by it. `extract`
# therefore evaluates the objective at each bundle's own θ under the code as it stands at
# extract time and stores that reference beside the intermediate; `rebuild` gates on the
# reference, not on the bundle's stored `loss_opt`. The stored value is the right
# reference only when no respecification has happened since the bundle was written, and
# several have: base_fc stores loss_opt 1071.079481 while v39.2.0 gives 1071.280270 at
# that same θ — a gap of 0.20 against a 1.1e-03 tolerance, which would reject a correct
# migration. Intermediates written before this reference existed fall back to `loss_opt`;
# the rebuild table prints both numbers, so specification drift and the struct-change
# invariant read separately.
#
# Nothing is overwritten in place: `rebuild` stages every rebuilt bundle under /.migration/
# and installs only once EVERY bundle of the named windows has reproduced its reference Q,
# moving each original to `*.jls.pre_migration` first.

using Serialization, Printf

# Deserialization needs the struct definitions in scope — the stream names its types and
# Serialization looks them up rather than carrying them. So even `extract`, which
# immediately discards struct identity, must load the module chain first. `extract` loads
# the OLD definitions (run it before editing); `rebuild` loads the NEW ones.
using LinearAlgebra, SparseArrays, Statistics, Random
using Distributions, FastGaussQuadrature, Interpolations, Parameters, Base.Threads
using Optim, CSV, DataFrames, Clustering, QuasiMonteCarlo, JSON3

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
# paths.jl supplies out_estimates() and estimate_path, so a tool cannot drift
# from where the entry points actually write. Guarded: including it twice would
# redefine its constants.
@isdefined(ROYSEARCH_PATHS_LOADED) || include(joinpath(ROOT, "code", "paths.jl"))

for f in ("grids", "params", "unskilled", "skilled", "solver", "equilibrium")
    include(joinpath(ROOT, "code", "solver", f * ".jl"))
end
include(joinpath(ROOT, "code", "smm", "settings.jl"))
include(joinpath(ROOT, "code", "smm", "moments.jl"))
include(joinpath(ROOT, "code", "smm", "smm_params.jl"))
# bundle.jl owns write_bundle, the one writer: a migrated bundle must come out the same
# shape as an estimated one rather than being assembled here.
include(joinpath(ROOT, "code", "smm", "bundle.jl"))
include(joinpath(ROOT, "code", "smm", "smm.jl"))
include(joinpath(ROOT, "code", "smm", "version.jl"))
const SMM  = out_estimates()
# Staging lives OUTSIDE output/ and is removed on success: a migration that succeeds
# leaves nothing, one that fails leaves everything.
const STAGE = joinpath(ROOT, ".migration")
const MID  = STAGE       # the version-neutral intermediate
const MODE = isempty(ARGS) ? "" : ARGS[1]

# Which windows to act on. Defaults to every window paths.jl knows about; pass names after
# the mode to restrict, e.g. `migrate_bundles.jl rebuild base_fc base_covid`. All-or-nothing
# install applies to whatever set is named, so a partial run installs only if every bundle in
# it passes.  Named MIG_WINDOWS because paths.jl already owns the constant WINDOWS, and the
# window list has one home.
const MIG_WINDOWS = length(ARGS) > 1 ? ARGS[2:end] : collect(WINDOWS)
const W_SUFFIX = "_diagonalW"

# The bundles to carry across: every estimate file a window can hold at the settled
# weighting, one per point (paths.jl ESTIMATE_POINTS). A file left in the old shape would
# fail to load, or load with shifted fields. Chains and seed banks are regenerable and
# deliberately excluded.
const MIG_FILES = [estimate_path(w, W_SUFFIX; point = p) for w in MIG_WINDOWS
                   for p in ESTIMATE_POINTS]
mid_path(f) = joinpath(MID, replace(basename(f), r"\.jls$" => ".neutral.jls"))

"""
    to_neutral(x)

Strip struct identity recursively, leaving only types whose serialised form does not
depend on a struct definition: NamedTuples, Vectors, Dicts and primitives. A struct
becomes a NamedTuple of its fields, tagged with `__struct__` so `rebuild` knows what to
reconstruct without guessing from the field names.
"""
function to_neutral(x)
    T = typeof(x)
    if x isa Union{Number, AbstractString, Symbol, Nothing, Bool}
        return x
    elseif x isa AbstractArray
        return map(to_neutral, x)
    elseif x isa Tuple
        return map(to_neutral, x)
    elseif x isa NamedTuple
        return NamedTuple{keys(x)}(map(to_neutral, values(x)))
    elseif x isa AbstractDict
        return Dict(k => to_neutral(v) for (k, v) in x)
    elseif isstructtype(T)
        fields = fieldnames(T)
        vals   = NamedTuple{fields}(Tuple(to_neutral(getfield(x, f)) for f in fields))
        return (__struct__ = string(nameof(T)), fields = vals)
    else
        return x
    end
end

# ── extract ──────────────────────────────────────────────────────────────────────
if MODE == "extract"
    mkpath(MID)
    n_ok = Ref(0)
    for p in filter(isfile, MIG_FILES)
        b = deserialize(p)
        # The reference for the acceptance test: the objective at this bundle's own θ
        # under the code as it stands NOW, which is the only quantity a struct change is
        # required to leave alone. The stored loss_opt carries every respecification
        # since the estimate was written and cannot serve.
        θ     = collect(float.(b.result.theta_opt))
        Q_ref = smm_objective(θ, b.spec)
        serialize(mid_path(p), (neutral = to_neutral(b), Q_extract = Q_ref))
        @printf("  %-50s extracted  Q_ref=%.6f  (stored=%.6f)  d=%d\n",
                basename(p), Q_ref, b.result.loss_opt, length(b.spec.free))
        n_ok[] += 1
    end
    @printf("\nextracted %d bundles to %s\n", n_ok[], MID)
    println("Now edit the structs, then run: migrate_bundles.jl rebuild")

# ── rebuild ──────────────────────────────────────────────────────────────────────
elseif MODE == "rebuild"

    """
        nt_of(x)

    The field NamedTuple of an extracted value. `to_neutral` tags a struct as
    `(__struct__, fields)` but leaves a NamedTuple as itself, so the bundle's own top
    level — which was already a NamedTuple — has no `fields`. This unwraps either form.
    """
    nt_of(x) = (x isa NamedTuple && haskey(x, :__struct__)) ? x.fields : x

    """
        from_neutral(x, T) → (value, added, dropped)

    Rebuild a `T` from an extracted value BY FIELD NAME, and report how the shape moved:
    `added` names fields `T` has that the intermediate does not (reconstructed from `T`'s
    own default), `dropped` names fields the intermediate carries that `T` no longer has.

    Matching by name is the whole point of the neutral form. Positional reconstruction
    would put field *k* of the old struct into slot *k* of the new one, so a removal
    anywhere but the end shifts every field after it into the wrong slot — silently,
    whenever the types happen to line up.

    `@kwdef` structs (`SMMRunParams`, `SimParams`) take the keyword constructor, which
    supplies the default for an added field for free. Plain structs (`ParamSpec`) have
    only the positional constructor; there every field must be present, and a missing one
    is an error naming it rather than a guess.
    """
    function from_neutral(x, ::Type{T}) where {T}
        nt      = nt_of(x)
        fields  = fieldnames(T)
        added   = Symbol[f for f in fields   if !haskey(nt, f)]
        dropped = Symbol[k for k in keys(nt) if !(k in fields)]

        if hasmethod(T, Tuple{}, fields)
            return T(; (f => nt[f] for f in fields if haskey(nt, f))...), added, dropped
        end
        isempty(added) || error("from_neutral: $T takes positional arguments only and the " *
                                "intermediate is missing " * join(added, ", ") *
                                " — give it a keyword constructor or migrate it by hand")
        return T((nt[f] for f in fields)...), added, dropped
    end

    # Where a shape difference occurred. Several struct types feed one `unrecorded` list,
    # and a bare :seed would not say which of them reconstructed it.
    qualify(prefix, fs) = Symbol[Symbol(prefix, ".", f) for f in fs]

    """
        has_tagged(x)

    Whether `x` still holds a `__struct__` tag anywhere inside it, i.e. a struct
    `to_neutral` stripped and nothing has rebuilt. The optional bundle slots are carried
    through verbatim, and writing a tagged NamedTuple into one would hand the next
    reader a NamedTuple where a struct belongs — a corrupt bundle that loads.
    """
    function has_tagged(x)
        x isa NamedTuple   && return haskey(x, :__struct__) || any(has_tagged, values(x))
        x isa AbstractDict && return any(has_tagged, values(x))
        (x isa AbstractArray || x isa Tuple) && return any(has_tagged, x)
        return false
    end

    # Every file that exists or was extracted: one in the old shape without an intermediate
    # would be left behind unreadable, so it fails the install.
    todo = filter(f -> isfile(f) || isfile(mid_path(f)), MIG_FILES)
    n_ok = Ref(0)
    for p in todo
        mp = mid_path(p)
        if !isfile(mp)
            @printf("  %-50s no intermediate — run `extract` on the OLD code first\n", basename(p))
            continue
        end
        mid = deserialize(mp)
        # Intermediates written before the reference Q existed are the bare neutral
        # bundle; those fall back to the stored loss_opt, which is what the tool used to
        # gate on, so nothing already on disk becomes unusable.
        has_ref = mid isa NamedTuple && haskey(mid, :neutral) && haskey(mid, :Q_extract)
        neutral = has_ref ? mid.neutral : mid

        old_b    = nt_of(neutral)
        old_spec = nt_of(old_b.spec)
        old_res  = nt_of(old_b.result)

        run,  run_add, run_drop = from_neutral(old_spec.run, SMMRunParams)
        sim,  sim_add, sim_drop = from_neutral(old_spec.sim, SimParams)
        # Every ParamSpec has the same shape, so the difference is reported once rather
        # than once per free coordinate.
        free_specs = ParamSpec[]
        ps_add, ps_drop = Symbol[], Symbol[]
        for ps in old_spec.free
            p, a, d = from_neutral(ps, ParamSpec)
            push!(free_specs, p)
            union!(ps_add, a); union!(ps_drop, d)
        end
        # The optional payload travels unchanged. `stage` in particular must not be
        # rewritten: base_fc is an :mcmc_checkpoint, and restamping it :smm would make
        # bundle_is_final (bundle.jl) accept a mid-chain checkpoint as a finished
        # estimate. The top-level `sim` is a struct and so needs rebuilding; se,
        # se_source and mcmc are carried as they came, guarded against holding a struct
        # this tool has not put back together.
        bsim, bsim_add, bsim_drop = old_b.sim === nothing ? (nothing, Symbol[], Symbol[]) :
                                                            from_neutral(old_b.sim, SimParams)
        for (slot, v) in ((:se, old_b.se), (:se_source, old_b.se_source), (:mcmc, old_b.mcmc))
            has_tagged(v) && error("rebuild: bundle field `$slot` holds a struct this tool " *
                                   "does not rebuild — extend from_neutral before migrating it")
        end

        added   = vcat(qualify(:run, run_add),  qualify(:sim, sim_add),
                       qualify(:free, ps_add),  qualify(:bundle_sim, bsim_add))
        dropped = vcat(qualify(:run, run_drop), qualify(:sim, sim_drop),
                       qualify(:free, ps_drop), qualify(:bundle_sim, bsim_drop))

        # Rebuild the spec around the new run params. build_smm_spec re-derives the
        # active-moment set and validates the free set against `fixed`, so the result is
        # internally consistent rather than assembled field by field. The stored W is
        # passed through: re-deriving it would re-read the sampling-variance file, and a
        # migration must not silently change the weighting a result was estimated under.
        #
        # The moment NamedTuple carries its own per-moment weights, and `free` holds the
        # ACTIVE (post-pinning) specs — build_smm_spec re-applies pinning, so passing
        # them back is idempotent.
        spec = build_smm_spec(old_spec.moments, sim;
                              fixed        = old_spec.fixed,
                              free_specs   = free_specs,
                              run          = run,
                              W            = old_spec.W,
                              q_scale      = old_spec.q_scale)

        θ      = collect(float.(old_res.theta_opt))
        Q_ref  = has_ref ? mid.Q_extract : old_res.loss_opt

        # The migration is correct only if the rebuilt spec reproduces the objective the
        # same θ gave before the struct edit. θ itself is untouched: a struct change is a
        # change of container, and anything that moves the point belongs in a
        # reparametrisation, not here.
        Q1 = smm_objective(θ, spec)
        ok = isfinite(Q1) && abs(Q1 - Q_ref) < 1e-6 * max(1.0, abs(Q_ref))

        result = SMMResult(θ, _params_to_namedtuple(θ, spec), Q1,
                           old_res.converged, old_res.iterations, spec)

        # `version` is overwritten with the running version here, so the code that
        # produced the estimate survives only in `migrated_from`. Appending rather than
        # replacing keeps every hop, in order, for a bundle migrated more than once.
        old_prov = old_b.provenance === nothing ? nothing : nt_of(old_b.provenance)
        chain    = old_prov === nothing ? String[] :
                   push!(haskey(old_prov, :migrated_from) ?
                             collect(String.(old_prov.migrated_from)) : String[],
                         String(old_prov.version))

        prov = run_provenance(window = provenance_from_path(p).window, w_suffix = W_SUFFIX,
                              version = ROYSEARCH_VERSION, unrecorded = added,
                              migrated_from = chain)

        # Through write_bundle since v19.6.0, so a migrated bundle has the same shape as
        # a written one. A .new sibling is exactly the debris a migration should not
        # leave, so it stages under /.migration/ at the repo root and the caller installs
        # it after reading the verification table.
        mkpath(STAGE)
        out = joinpath(STAGE, basename(p))
        old_tag = String(old_b.tag)
        write_bundle(out; result = result, spec = spec, stage = Symbol(old_b.stage),
                     provenance = prov, sim = bsim, se = old_b.se,
                     se_source = old_b.se_source, mcmc = old_b.mcmc,
                     vacancy_costs = old_b.vacancy_costs, attracting = old_b.attracting,
                     tag = isempty(old_tag) ? "migrated" : old_tag * " | migrated")
        # Three numbers, because they answer different questions: `stored` is what the
        # estimate was written with, `ref` is what the pre-edit code gives at that θ, and
        # `Δ` against `ref` is the struct-change invariant. stored ≠ ref is specification
        # drift and does not condemn the migration.
        @printf("  %-50s %s  stored=%.6f  ref=%.6f (%s)  rebuilt=%.6f  Δ=%.3e  unrecorded=%d dropped=%d\n",
                basename(p), ok ? "OK  " : "MISMATCH", old_res.loss_opt, Q_ref,
                has_ref ? "extract" : "loss_opt", Q1, abs(Q1 - Q_ref),
                length(added), length(dropped))
        !isempty(added)   && @printf("      defaulted: %s\n", join(added, ", "))
        !isempty(dropped) && @printf("      dropped:   %s\n", join(dropped, ", "))
        @printf("      estimate from: %s\n",
                isempty(chain) ? "(no provenance in the original)" : join(chain, " → "))
        ok && (n_ok[] += 1)
    end

    if !isempty(todo) && n_ok[] == length(todo)
        for p in todo
            isfile(p) && mv(p, p * ".pre_migration"; force = true)
            mv(joinpath(STAGE, basename(p)), p; force = true)
        end
        # Clear only what this run consumed. A run restricted to some windows must not
        # strand the intermediates of the others: once the structs have been edited
        # those can no longer be regenerated. The directory goes when nothing is left in
        # it, so "staging present" means "a migration is unfinished".
        foreach(p -> rm(mid_path(p); force = true), todo)
        isempty(readdir(STAGE)) && rm(STAGE; recursive = true, force = true)
        @printf("\nall %d bundles rebuilt and installed; originals at *.jls.pre_migration\n", n_ok[])
        println("Confirm with: migrate_bundles.jl verify")
    else
        @printf("\n%d of %d reproduced their reference Q. NOTHING INSTALLED — the rebuilt\n",
                n_ok[], length(todo))
        @printf("bundles are staged in %s for inspection.\n", STAGE)
        exit(1)
    end

# ── verify ───────────────────────────────────────────────────────────────────────
elseif MODE == "verify"

    for p in filter(isfile, MIG_FILES)
        b = deserialize(p)
        has_prov = hasproperty(b, :provenance) && b.provenance !== nothing
        Q = smm_objective(collect(float.(b.result.theta_opt)), b.spec)
        @printf("  %-50s loads  Q=%.6f  stored=%.6f  match=%s  schema=%s  version=%s  from=%s  unrecorded=%d\n",
                basename(p), Q, b.result.loss_opt,
                abs(Q - b.result.loss_opt) < 1e-6 ? "yes" : "NO",
                has_prov ? string(b.provenance.schema) : "—",
                has_prov ? b.provenance.version : "—",
                has_prov && !isempty(b.provenance.migrated_from) ?
                    join(b.provenance.migrated_from, "→") : "—",
                has_prov ? length(b.provenance.unrecorded) : -1)
    end

else
    println("usage: migrate_bundles.jl extract|rebuild|verify")
    println()
    println("  extract   run on the OLD code, BEFORE editing structs")
    println("  rebuild   run on the NEW code, after editing structs")
    println("  verify    confirm the installed bundles load and reproduce their Q")
    exit(1)
end

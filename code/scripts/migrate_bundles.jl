#!/usr/bin/env julia
#
# migrate_bundles.jl — carry serialised SMM bundles across a struct change.
#
#   julia code/scripts/migrate_bundles.jl extract    # BEFORE editing the structs
#   julia code/scripts/migrate_bundles.jl rebuild    # AFTER editing the structs
#   julia code/scripts/migrate_bundles.jl verify     # check the rebuilt bundles
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
# Nothing is overwritten in place: `rebuild` writes `*.jls.new` and moves the original
# to `*.jls.pre_migration`, so a failed rebuild leaves the originals recoverable.

using Serialization, Printf

# Deserialization needs the struct definitions in scope — the stream names its types and
# Serialization looks them up rather than carrying them. So even `extract`, which
# immediately discards struct identity, must load the module chain first. `extract` loads
# the OLD definitions (run it before editing); `rebuild` loads the NEW ones.
using LinearAlgebra, SparseArrays, Statistics, Random
using Distributions, FastGaussQuadrature, Interpolations, Parameters, Base.Threads
using Optim, CSV, DataFrames, Clustering, QuasiMonteCarlo, JSON3

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

for f in ("grids", "params", "unskilled", "skilled", "solver", "equilibrium")
    include(joinpath(ROOT, "code", "solver", f * ".jl"))
end
include(joinpath(ROOT, "code", "smm", "moments.jl"))
include(joinpath(ROOT, "code", "smm", "smm_params.jl"))
include(joinpath(ROOT, "code", "smm", "smm.jl"))
include(joinpath(ROOT, "code", "smm", "version.jl"))
const SMM  = joinpath(ROOT, "output", "smm")
const MID  = joinpath(SMM, "_migration")       # the version-neutral intermediate
const MODE = isempty(ARGS) ? "" : ARGS[1]

# The bundles to carry across: one canonical estimate per window at the settled
# weighting. Chains and seed banks are regenerable and deliberately excluded.
# Which bundles to act on. Defaults to all four; pass names after the mode to restrict,
# e.g. `migrate_bundles.jl rebuild base_fc base_covid`. All-or-nothing install applies to
# whatever set is named, so a partial run installs only if every bundle in it passes.
const ALL_WINDOWS = ["base_fc", "crisis_fc", "base_covid", "crisis_covid"]
const WINDOWS     = length(ARGS) > 1 ? ARGS[2:end] : ALL_WINDOWS
const W_SUFFIX = "_diagonalW"

bundle_path(w) = joinpath(SMM, "smm_result_$(w)$(W_SUFFIX).jls")
mid_path(w)    = joinpath(MID, "$(w)$(W_SUFFIX).neutral.jls")

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
    for w in WINDOWS
        p = bundle_path(w)
        if !isfile(p)
            @printf("  %-14s no bundle at %s\n", w, p)
            continue
        end
        b = deserialize(p)
        neutral = to_neutral(b)
        serialize(mid_path(w), neutral)
        # Report the two numbers that must survive the round trip unchanged.
        @printf("  %-14s extracted  Q=%.6f  d=%d  → %s\n",
                w, b.result.loss_opt, length(b.spec.free), basename(mid_path(w)))
        n_ok[] += 1
    end
    @printf("\nextracted %d of %d bundles to %s\n", n_ok[], length(WINDOWS), MID)
    println("Now edit the structs, then run: migrate_bundles.jl rebuild")

# ── rebuild ──────────────────────────────────────────────────────────────────────
elseif MODE == "rebuild"
    # The new struct definitions, and the objective needed to confirm each rebuilt
    # bundle still evaluates to the Q it was extracted with.

    """
        rebuild_runparams(old_nt)

    Reconstruct `SMMRunParams` from the extracted NamedTuple. Fields the old struct
    carried are taken from it; fields added since are left at their new defaults and
    named in `unrecorded`, so the bundle states which of its settings are reconstructed
    rather than observed. A migrated bundle must not present a default as a fact.
    """
    function rebuild_runparams(old_nt::NamedTuple)
        new_fields = fieldnames(SMMRunParams)
        carried    = Dict{Symbol,Any}()
        for f in new_fields
            haskey(old_nt, f) && (carried[f] = old_nt[f])
        end
        added = Symbol[f for f in new_fields if !haskey(old_nt, f)]
        dropped = Symbol[k for k in keys(old_nt) if !(k in new_fields)]
        return SMMRunParams(; carried...), added, dropped
    end

    """
        nt_of(x)

    The field NamedTuple of an extracted value. `to_neutral` tags a struct as
    `(__struct__, fields)` but leaves a NamedTuple as itself, so the bundle's own top
    level — which was already a NamedTuple — has no `fields`. This unwraps either form.
    """
    nt_of(x) = (x isa NamedTuple && haskey(x, :__struct__)) ? x.fields : x

    """
        from_neutral(x, T)

    Rebuild a struct of type `T` from an extracted value. Plain structs (`ParamSpec`)
    only take positional arguments, so every field must be present and in order;
    `@kwdef` structs (`SimParams`) accept keywords and tolerate a missing field by
    falling back to its default. Trying positional first and keyword second covers both
    without asking which kind `T` is.
    """
    function from_neutral(x, ::Type{T}) where {T}
        nt = nt_of(x)
        fields = fieldnames(T)
        if all(f -> haskey(nt, f), fields)
            return T((nt[f] for f in fields)...)
        end
        return T(; (f => nt[f] for f in fields if haskey(nt, f))...)
    end

    """
        to_PU_numeraire(θ, spec) → (θ′, P_U_old)

    Move a point onto the P_U = 1 ray. `A` enters the solver only as `exp(A)` multiplying
    every output-denominated level, so

        A → A + s,   (P_U, P_S, b_U, b_S, σ_S) → × exp(−s)

    leaves every product, value function, policy margin and moment unchanged. Taking
    `s = log(P_U)` sends P_U to exactly 1 and fixes the numéraire; P_S then reads as the
    skilled/unskilled productivity ratio. Verified on this solver at s = 0.01, 0.05, 0.20:
    |ΔQ| ≤ 9e-12 against an objective noise floor of 0.59.

    Without this, a warm start under the P_U = 1 pin would load the old `A` against the
    new P_U and change every `exp(A)·level` product by a factor of the old P_U — a
    different model, not the same one relabelled. Returns θ unchanged when P_U is already
    1 or is not a free coordinate.
    """
    function to_PU_numeraire(θ::Vector{Float64}, spec::SMMSpec)
        names = [string(ps.block, ":", ps.name) for ps in spec.free]
        i_PU  = findfirst(==("unsk:PU"), names)
        i_A   = findfirst(==("common:A"), names)
        (i_PU === nothing || i_A === nothing) && return θ, NaN

        con(i) = _to_constrained(θ[i], spec.free[i].lb, spec.free[i].ub)
        PU = con(i_PU)
        (isfinite(PU) && abs(PU - 1.0) > 1e-12) || return θ, PU

        inbox(v, ps) = clamp(v, ps.lb + 1e-12 * (ps.ub - ps.lb),
                                ps.ub - 1e-12 * (ps.ub - ps.lb))
        # b_T is in the list because unskilled.jl:121 scales it by exp(A) like any other
        # flow value. It is pinned at 0 in current runs, where the division is a no-op,
        # but an older bundle can carry it free — and leaving it behind while A rises
        # multiplies the training flow relative to everything else and inverts the
        # training margin.
        θ2 = copy(θ)
        for nm in ("unsk:PU", "skl:PS", "unsk:bU", "skl:bS", "skl:σ", "unsk:bT")
            i = findfirst(==(nm), names)
            i === nothing && continue
            θ2[i] = _to_unconstrained(inbox(con(i) / PU, spec.free[i]),
                                      spec.free[i].lb, spec.free[i].ub)
        end
        θ2[i_A] = _to_unconstrained(inbox(con(i_A) + log(PU), spec.free[i_A]),
                                    spec.free[i_A].lb, spec.free[i_A].ub)
        return θ2, PU
    end

    n_ok = Ref(0)
    for w in WINDOWS
        mp = mid_path(w)
        if !isfile(mp)
            @printf("  %-14s no intermediate — run `extract` on the OLD code first\n", w)
            continue
        end
        neutral = deserialize(mp)

        old_spec = nt_of(nt_of(neutral).spec)
        old_res  = nt_of(nt_of(neutral).result)
        run, added, dropped = rebuild_runparams(nt_of(old_spec.run))

        # Rebuild the spec around the new run params. build_smm_spec re-derives the
        # active-moment set and validates the free set against `fixed`, so the result is
        # internally consistent rather than assembled field by field. The stored W is
        # passed through: re-deriving it would re-read the sampling-variance file, and a
        # migration must not silently change the weighting a result was estimated under.
        #
        # The moment NamedTuple carries its own per-moment weights, and `free` holds the
        # ACTIVE (post-pinning) specs — build_smm_spec re-applies pinning, so passing
        # them back is idempotent.
        spec = build_smm_spec(old_spec.moments, from_neutral(old_spec.sim, SimParams);
                              fixed        = old_spec.fixed,
                              free_specs   = [from_neutral(ps, ParamSpec)
                                              for ps in old_spec.free],
                              run          = run,
                              W            = old_spec.W,
                              q_scale      = old_spec.q_scale)

        θ  = collect(float.(old_res.theta_opt))
        Q0 = old_res.loss_opt

        # Move the point onto the P_U = 1 ray before checking Q. The conversion is exactly
        # flat, so it does not weaken the acceptance test: Q must still reproduce, and if
        # the arithmetic were wrong it would fail here rather than reach a warm start.
        θ, PU_old = to_PU_numeraire(θ, spec)

        # The migration is only correct if the rebuilt spec reproduces the extracted Q.
        Q1 = smm_objective(θ, spec)
        ok = isfinite(Q1) && abs(Q1 - Q0) < 1e-6 * max(1.0, abs(Q0))

        cp, up, sp = unpack_θ(θ, spec)
        result = SMMResult(θ, _params_to_namedtuple(cp, up, sp, spec), Q1,
                           old_res.converged, old_res.iterations, spec)

        prov = run_provenance(window = Symbol(w), w_suffix = W_SUFFIX,
                              version = ROYSEARCH_VERSION, unrecorded = added)

        out = bundle_path(w) * ".new"
        open(out, "w") do io
            serialize(io, (result = result, spec = spec, provenance = prov))
        end
        @printf("  %-14s %s  Q: %.6f → %.6f  (Δ=%.2e)  P_U %s→1  unrecorded=%d dropped=%d\n",
                w, ok ? "OK  " : "MISMATCH", Q0, Q1, abs(Q1 - Q0),
                isnan(PU_old) ? "n/a" : @sprintf("%.4f", PU_old),
                length(added), length(dropped))
        !isempty(dropped) && @printf("      dropped fields: %s\n", join(dropped, ", "))
        ok && (n_ok[] += 1)
    end

    if n_ok[] == length(WINDOWS)
        for w in WINDOWS
            p = bundle_path(w)
            isfile(p) && mv(p, p * ".pre_migration"; force = true)
            mv(p * ".new", p; force = true)
        end
        @printf("\nall %d bundles rebuilt and installed; originals at *.jls.pre_migration\n", n_ok[])
        println("Confirm with: migrate_bundles.jl verify")
    else
        @printf("\n%d of %d reproduced their Q. NOTHING INSTALLED — the .new files are\n",
                n_ok[], length(WINDOWS))
        println("left beside the originals for inspection.")
        exit(1)
    end

# ── verify ───────────────────────────────────────────────────────────────────────
elseif MODE == "verify"

    for w in WINDOWS
        p = bundle_path(w)
        if !isfile(p); @printf("  %-14s MISSING\n", w); continue; end
        b = deserialize(p)
        has_prov = hasproperty(b, :provenance)
        Q = smm_objective(collect(float.(b.result.theta_opt)), b.spec)
        @printf("  %-14s loads  Q=%.6f  stored=%.6f  match=%s  schema=%s  version=%s  unrecorded=%d\n",
                w, Q, b.result.loss_opt,
                abs(Q - b.result.loss_opt) < 1e-6 ? "yes" : "NO",
                has_prov ? string(b.provenance.schema) : "—",
                has_prov ? b.provenance.version : "—",
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

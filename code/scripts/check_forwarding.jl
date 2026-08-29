#!/usr/bin/env julia
# check_forwarding.jl — catch settings that are computed and then never delivered.
#
# THE DEFECT CLASS. A setting the caller computes and does not pass is legal Julia: the
# callee's keyword default takes over, nothing crashes, the run produces numbers, and a
# settings block sourced from the caller reports the value the caller intended rather
# than the one the consumer received. Between v18.4.0 and v19.0.0 four SA settings ran
# that way — sa_subset_k, sa_halflife, sa_rate_tol, sa_rate_span, all computed into
# run_params and none forwarded to run_smm — and the visible symptom was an annealing
# stage that accepted 4% of proposals and then 0%, because sa_subset_k = 0 disabled
# per-coordinate adaptation and the scalar fallback clamped its step at roughly 3x the
# measured ΔQ=1 half-width, so every proposal overshot. (sa_subset_k has since been
# retired; SETTINGS.md records what replaced it.)
#
# No compiler catches it. The defect is semantic: a value exists in the caller, a
# parameter of the same name exists in the callee, and nothing connects them. Only a
# check holding both name sets can see it.
#
# FOUR CHECKS, each failing the gate on its own.
#   1  forwarding    every run_smm / _run_sa / _run_de / run_demc keyword argument that
#                    shadows a value its caller holds is actually passed.
#   2  registry      every ROYSEARCH_* key read by the code has a SETTINGS_REGISTRY row,
#                    and every row corresponds to a key some file reads. This is what
#                    closes the NAME blind spot the previous version documented: the
#                    registry names each setting's consumer, so a key that reaches no
#                    consumer is a finding regardless of how it is spelled downstream.
#   3  name parity   SMMRunParams fields passed on to a keyword argument must be spelled
#                    identically at both ends. Check 1 matches on name and therefore
#                    cannot see a rename — this check is what makes that assumption safe
#                    rather than a documented hole.
#   4  legacy keys   a renamed key must be listed in LEGACY_ENV_KEYS so a stale launcher
#                    fails loudly instead of running under silently-substituted defaults.
#
# An INERT setting — forwarded correctly but on a code path the shipped method does not
# take — is not a defect and must not fail the gate. Those are declared :inert in the
# registry with the reason, and the allowlist below is derived from that declaration
# rather than being a second hand-maintained list that can disagree with it.
#
# Usage:  julia --project=. code/scripts/check_forwarding.jl

include(joinpath(@__DIR__, "repo_root.jl"))
const ROOT = find_repo_root()

read_src(parts...) = read(joinpath(ROOT, parts...), String)

const SMM      = read_src("code", "smm", "smm.jl")
const MAIN     = read_src("code", "smm", "smm_main.jl")
const PARAMS   = read_src("code", "smm", "smm_params.jl")
const MCMC     = read_src("code", "smm", "MCMC_main.jl")
const DEMC     = read_src("code", "smm", "demc.jl")
const SETTINGS = read_src("code", "smm", "settings.jl")

# ============================================================
# Source parsing
# ============================================================

"""
    kwargs_of(src, fname, terminator) → Set{String}

Keyword-argument names of `fname`'s definition, reading up to `terminator`. Matches
names at the signature's own indentation, so a default value spanning lines cannot
contribute a spurious entry.
"""
function kwargs_of(src::AbstractString, fname::AbstractString, terminator::AbstractString)
    i = findfirst("function $fname(", src)
    i === nothing && error("could not find `function $fname(`")
    tail = src[first(i):end]
    j = findfirst(terminator, tail)
    j === nothing && error("could not find terminator `$terminator` after $fname")
    sig = tail[1:first(j)]
    Set(String(m.captures[1])
        for m in eachmatch(r"(?m)^\s{2,}([a-z_][A-Za-z0-9_]*)\s*(?:::[^=\n]+)?=", sig))
end

"""
    passed_at_call(src, anchor) → Set{String}

Names appearing as `name =` in the call beginning at `anchor`, tracking parenthesis
depth so the call's true end is found rather than guessed from a blank line.

Depth counting starts at the LAST `(` of the anchor, not at the anchor's first
character: an anchor like `_sa_stage(sp, starts) = _run_sa(` contains a balanced pair
of its own, which would otherwise close the scan before the argument list begins and
report an empty call — a false PASS on exactly the forwarding this checks.
"""
function passed_at_call(src::AbstractString, anchor::AbstractString)
    i = findfirst(anchor, src)
    i === nothing && error("could not find call anchor `$anchor`")
    open_rel = findlast('(', anchor)
    open_rel === nothing && error("call anchor `$anchor` contains no `(`")
    tail  = src[(first(i) + open_rel - 1):end]
    depth = 0
    stop  = lastindex(tail)
    for (k, ch) in enumerate(tail)
        ch == '(' && (depth += 1)
        if ch == ')'
            depth -= 1
            depth == 0 && (stop = k; break)
        end
    end
    call = tail[1:stop]
    Set(String(m.captures[1]) for m in eachmatch(r"([a-z_][A-Za-z0-9_]*)\s*=", call))
end

"""Field names of the `SMMRunParams` definition — the settings a bundle records."""
function runparams_fields(src::AbstractString)
    i = findfirst("Base.@kwdef struct SMMRunParams", src)
    i === nothing && error("could not find SMMRunParams")
    tail = src[first(i):end]
    body = tail[1:first(findfirst(r"(?m)^end", tail))]
    Set(String(m.captures[1])
        for m in eachmatch(r"(?m)^\s{4}([a-zA-Zλ_][A-Za-z0-9_]*)\s*::", body))
end

"""Keyword defaults of `fname`, for reporting what an omission would silently select."""
function defaults_of(src::AbstractString, fname::AbstractString, terminator::AbstractString)
    i = findfirst("function $fname(", src)
    tail = src[first(i):end]
    sig  = tail[1:first(findfirst(terminator, tail))]
    d = Dict{String,String}()
    for m in eachmatch(r"(?m)^\s{2,}([a-z_][A-Za-z0-9_]*)\s*(?:::[^=\n]+)?=\s*([^,\n]+)", sig)
        d[String(m.captures[1])] = strip(String(m.captures[2]))
    end
    d
end

# ============================================================
# The registry, read out of settings.jl
# ============================================================

# Parsed from source rather than by including the file: the check must run without
# loading the solver stack, and a gate that imports the code it audits can be defeated
# by the same error it is looking for.
const REG_ROWS = [(key = String(m.captures[1]), status = String(m.captures[2]))
                  for m in eachmatch(
                      r"SettingDecl\(:([A-Z0-9_]+),\s*\"[^\"]*\",\s*:(\w+)", SETTINGS)]
const REG_KEYS   = Set(r.key for r in REG_ROWS)
const REG_INERT  = Set(r.key for r in REG_ROWS if r.status == "inert")
const LEGACY_OLD = Set(String(m.captures[1])
                       for m in eachmatch(r"\"ROYSEARCH_([A-Z0-9_]+)\"\s*=>", SETTINGS))

# Keys the code actually reads, across every file that reads one.
const READ_KEYS = Set{String}()
for src in (MAIN, MCMC, SMM, PARAMS, DEMC)
    for m in eachmatch(r"env_setting\(:([A-Z0-9_]+)", src)
        push!(READ_KEYS, String(m.captures[1]))
    end
end

# A bare ENV read bypasses the registry entirely, so it cannot be classified and
# print_env_settings cannot report it. Permitted only for the settings layer's own
# lookups, which are how the mechanism is implemented.
const RAW_ENV_READS = String[]
for (name, src) in (("smm_main.jl", MAIN), ("MCMC_main.jl", MCMC),
                    ("smm.jl", SMM), ("smm_params.jl", PARAMS), ("demc.jl", DEMC))
    for m in eachmatch(r"ENV\[\"(ROYSEARCH_[A-Z0-9_]+)\"\]", src)
        push!(RAW_ENV_READS, "$name reads $(m.captures[1]) directly")
    end
end

# ============================================================
# Check 1 — forwarding
# ============================================================

const RUNPARAMS = runparams_fields(PARAMS)

"""
    forwarding_findings(kw, held, passed) → Vector{String}

Names present in both the callee's keyword arguments and the caller's own values, and
absent from the call. `held` is what the caller has to give; `passed` is what it gives.
"""
forwarding_findings(kw, held, passed) =
    sort(collect(setdiff(intersect(kw, held), passed)))

const SMM_KW    = kwargs_of(SMM, "run_smm", ") :: SMMResult")
const SMM_DEF   = defaults_of(SMM, "run_smm", ") :: SMMResult")
const SA_KW     = kwargs_of(SMM, "_run_sa", "\n)\n")
const DE_KW     = kwargs_of(SMM, "_run_de", "\n)\n")
const DEMC_KW   = kwargs_of(DEMC, "run_demc", "on_best = nothing)")

# run_smm's caller holds every SMMRunParams field (it built the struct). The two SA
# stages inside run_smm hold r.* fields plus run_smm's own keyword arguments.
const SMM_PASSED = passed_at_call(MAIN, "res = run_smm(")
const SA_PASSED  = union(passed_at_call(SMM, "_sa_stage(sp, starts) = _run_sa("),
                         passed_at_call(SMM, "theta_opt, loss_opt, niters = _run_sa("))
const DE_PASSED  = union(passed_at_call(SMM, "_de_stage(sp, bank, prev) = _run_de("),
                         passed_at_call(SMM, "theta_opt, loss_opt, niters = _run_de("))

# MCMC_main holds its settings as MCMC_-prefixed globals; run_demc's parameters are
# unprefixed, so the caller's holdings are mapped to the callee's spelling before
# intersecting. The mapping is the reason check 3 exists.
const MCMC_HELD = Set(lowercase(String(m.captures[1]))
                      for m in eachmatch(r"(?m)^MCMC_([A-Z0-9_]+)\s*=", MCMC))
const DEMC_PASSED = passed_at_call(MCMC, "run_demc(logposterior, θ0;")

# The stage helpers drop the subsystem prefix: SMMRunParams' `sa_max_iter` becomes
# _run_sa's `max_iter`. Comparing the two name spaces therefore requires stripping the
# prefix from the fields that carry it, or the intersection is empty and the check
# passes without having examined anything.
strip_prefix(fields, pre) =
    Set(f[(length(pre) + 1):end] for f in fields if startswith(f, pre))

# `sa_halflife` is the exception to the prefix convention: _run_sa spells it
# `cooling_halflife`, so the pair is asserted explicitly here rather than left to a
# rule that does not cover it.
const SA_HELD = union(strip_prefix(RUNPARAMS, "sa_"), Set(["cooling_halflife"]))
const DE_HELD = strip_prefix(RUNPARAMS, "de_")

findings = Tuple{String,String,String}[]   # (check, subject, detail)

for (label, kw, held, passed, defs, pre) in (
        ("run_smm",   SMM_KW,  RUNPARAMS, SMM_PASSED,  SMM_DEF,               ""),
        ("_run_sa",   SA_KW,   SA_HELD,   SA_PASSED,   Dict{String,String}(), "sa_"),
        ("_run_de",   DE_KW,   DE_HELD,   DE_PASSED,   Dict{String,String}(), "de_"),
        ("run_demc",  DEMC_KW, MCMC_HELD, DEMC_PASSED, Dict{String,String}(), ""))
    for k in forwarding_findings(kw, held, passed)
        # An inert setting is declared as such in the registry; the allowlist is that
        # declaration, keyed by the setting's uppercase env spelling. The prefix is
        # restored first, since the registry keys the full setting name.
        uppercase(pre * k) in REG_INERT && continue
        d = get(defs, k, "?")
        push!(findings, ("forwarding", "$label($k)",
                         "computed by the caller, never passed; silently defaults to $d"))
    end
end

# ============================================================
# Check 2 — registry completeness
# ============================================================

for k in sort(collect(setdiff(READ_KEYS, REG_KEYS)))
    push!(findings, ("registry", "ROYSEARCH_$k",
                     "read by the code with no SETTINGS_REGISTRY row — unclassified"))
end
for k in sort(collect(setdiff(REG_KEYS, READ_KEYS)))
    push!(findings, ("registry", "ROYSEARCH_$k",
                     "declared in SETTINGS_REGISTRY but no code reads it — obsolete row"))
end
for msg in sort(RAW_ENV_READS)
    push!(findings, ("registry", msg,
                     "bypasses env_setting, so it is neither classified nor reported"))
end

# ============================================================
# Check 3 — caller/callee name parity
# ============================================================

# Every `kwarg = run_params.field` pair at a call site: the two names must agree, or
# check 1's name matching silently stops covering that setting.
#
# One transformation is permitted, because it is the codebase's own convention rather
# than an ad-hoc exception: a field carrying a SUBSYSTEM PREFIX may drop it at the
# boundary of a consumer that serves only that subsystem, as SMMRunParams' `sa_max_iter`
# becomes _run_sa's `max_iter`. Check 1 restores the prefix before comparing, so these
# pairs stay covered. Any other difference is a genuine blind spot.
const NAME_PREFIXES = ("sa_", "de_", "nm_", "cand_", "clusters_", "seed_")

names_agree(kw, field) =
    kw == field || any(p -> startswith(field, p) && field[(length(p) + 1):end] == kw,
                       NAME_PREFIXES)

for m in eachmatch(r"([a-z_][A-Za-z0-9_]*)\s*=\s*run_params\.([a-z_][A-Za-z0-9_]*)", MAIN)
    kw, field = String(m.captures[1]), String(m.captures[2])
    names_agree(kw, field) && continue
    push!(findings, ("name parity", "$kw = run_params.$field",
                     "keyword and field names differ by more than a subsystem prefix, "
                     * "so a forwarding gap here is invisible to check 1"))
end

# ============================================================
# Check 4 — renamed keys are declared
# ============================================================

# A key present in the registry under a new spelling while the old spelling still
# appears in a launcher is the failure mode LEGACY_ENV_KEYS exists to make loud.
for f in ("vm_run.sh", "setup_vm.sh", "vm_go.sh")
    path = joinpath(ROOT, "code", "scripts", f)
    isfile(path) || continue
    for m in eachmatch(r"ROYSEARCH_([A-Z0-9_]+)", read(path, String))
        k = String(m.captures[1])
        (k in REG_KEYS || k == "VERSION" || k in LEGACY_OLD) && continue
        push!(findings, ("legacy key", "$f: ROYSEARCH_$k",
                         "set by a launcher but matches no registry row"))
    end
end

# ============================================================
# Report
# ============================================================

println("check_forwarding — settings delivery, registry parity and name parity")
println("  registry rows:                 ", length(REG_ROWS),
        "  (", length(REG_INERT), " declared inert)")
println("  keys read via env_setting:     ", length(READ_KEYS))
println("  SMMRunParams fields:           ", length(RUNPARAMS))
println("  kwargs  run_smm/_run_sa/_run_de/run_demc:  ",
        join(length.((SMM_KW, SA_KW, DE_KW, DEMC_KW)), "/"))
println()

if isempty(findings)
    println("PASS — every setting reaches its consumer, every key is classified,")
    println("and no caller/callee pair disagrees on a name.")
    exit(0)
end

println("FINDINGS ($(length(findings))):")
for check in unique(f[1] for f in findings)
    println("\n  [$check]")
    for (_, subject, detail) in filter(f -> f[1] == check, findings)
        println("    $subject")
        println("        $detail")
    end
end
println()
println("Each is a settings-lifecycle violation. See SETTINGS.md for the rule that")
println("applies and the sequence for adding, renaming or retiring a setting.")
exit(1)

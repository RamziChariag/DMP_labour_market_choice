#!/usr/bin/env julia
# gates/check_forwarding.jl — settings that are computed and never delivered.
#
# A setting the caller holds and does not pass is legal Julia: the callee's keyword
# default takes over silently. Four checks, each failing the gate on its own:
#   1  forwarding    every run_smm / _run_sa / _run_de / run_demc keyword that shadows a
#                    value its caller holds is passed;
#   2  registry      every ROYSEARCH_* key read by the code has a SETTINGS_REGISTRY row,
#                    every row is read somewhere, and no file reads ENV directly;
#   3  name parity   a `kwarg = run_params.field` pair differs at most by a subsystem
#                    prefix, which is what check 1's name matching assumes;
#   4  legacy keys   every ROYSEARCH_* key a launcher in ops/ sets is registered or
#                    declared in LEGACY_ENV_KEYS.
# Settings declared :inert in the registry (forwarded, on a path the shipped method does
# not take) are exempt from check 1.
#
# Reads source text only: the gate must run without loading the solver stack.
#
#   julia --project=. code/gates/check_forwarding.jl

include(joinpath(@__DIR__, "..", "repo_root.jl"))
const ROOT = find_repo_root()

"""Contents of the repository file at `joinpath(ROOT, parts...)`."""
read_src(parts...) = read(joinpath(ROOT, parts...), String)

const SMM      = read_src("code", "smm", "smm.jl")
const MAIN     = read_src("code", "smm", "smm_main.jl")
const PARAMS   = read_src("code", "smm", "smm_params.jl")
const MCMC     = read_src("code", "mcmc", "MCMC_main.jl")
const DEMC     = read_src("code", "mcmc", "demc.jl")
const SETTINGS = read_src("code", "smm", "settings.jl")
const MOMENTS  = read_src("code", "smm", "moments.jl")

# ============================================================
# Source parsing
# ============================================================

"""
    kwargs_of(src, fname, terminator) → Set{String}

Keyword-argument names in the signature of `function fname(` in `src`, read up to the
first `terminator`. Only names indented by at least two spaces count, so a default value
spanning lines adds no spurious entry.
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

Names appearing as `name =` in the call that begins at the literal `anchor` in `src`,
up to its matching `)`. Depth counting starts at the anchor's last `(`, because an anchor
such as `_sa_stage(sp, starts) = _run_sa(` holds a balanced pair of its own.
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

"""Field names of `Base.@kwdef struct SMMRunParams` in `src` (4-space indented `name ::`)."""
function runparams_fields(src::AbstractString)
    i = findfirst("Base.@kwdef struct SMMRunParams", src)
    i === nothing && error("could not find SMMRunParams")
    tail = src[first(i):end]
    body = tail[1:first(findfirst(r"(?m)^end", tail))]
    Set(String(m.captures[1])
        for m in eachmatch(r"(?m)^\s{4}([a-zA-Zλ_][A-Za-z0-9_]*)\s*::", body))
end

"""Keyword name => default-expression text for `fname`'s signature in `src`, read up to
`terminator`; reported as the value an omission would select."""
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

# Registry rows (key, status), registry keys, :inert keys and legacy keys, parsed from
# settings.jl's source.
const REG_ROWS = [(key = String(m.captures[1]), status = String(m.captures[2]))
                  for m in eachmatch(
                      r"SettingDecl\(:([A-Z0-9_]+),\s*\"[^\"]*\",\s*:(\w+)", SETTINGS)]
const REG_KEYS   = Set(r.key for r in REG_ROWS)
const REG_INERT  = Set(r.key for r in REG_ROWS if r.status == "inert")
const LEGACY_OLD = Set(String(m.captures[1])
                       for m in eachmatch(r"\"ROYSEARCH_([A-Z0-9_]+)\"\s*=>", SETTINGS))

# Keys the code actually reads, across every file that reads one.
const READ_KEYS = Set{String}()
for src in (MAIN, MCMC, SMM, PARAMS, DEMC, MOMENTS)
    for m in eachmatch(r"env_setting\(:([A-Z0-9_]+)", src)
        push!(READ_KEYS, String(m.captures[1]))
    end
end

# A bare ENV read bypasses the registry, so it is neither classified nor reported.
const RAW_ENV_READS = String[]
for (name, src) in (("smm_main.jl", MAIN), ("MCMC_main.jl", MCMC),
                    ("smm.jl", SMM), ("smm_params.jl", PARAMS), ("demc.jl", DEMC),
                    ("moments.jl", MOMENTS))
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

Sorted names in both the callee's keywords `kw` and the caller's values `held` that are
absent from the call's `passed` names.
"""
forwarding_findings(kw, held, passed) =
    sort(collect(setdiff(intersect(kw, held), passed)))

const SMM_KW    = kwargs_of(SMM, "run_smm", ") :: SMMResult")
const SMM_DEF   = defaults_of(SMM, "run_smm", ") :: SMMResult")
const SA_KW     = kwargs_of(SMM, "_run_sa", "\n)\n")
const DE_KW     = kwargs_of(SMM, "_run_de", "\n)\n")
const DEMC_KW   = kwargs_of(DEMC, "run_demc", "on_best = nothing)")

# run_smm's caller holds every SMMRunParams field.
const SMM_PASSED = passed_at_call(MAIN, "res = run_smm(")
# SA has one call site (_sa_stage). DE has two (_de_stage and the `:de` branch), so a
# keyword counts as passed if either call passes it.
const SA_PASSED  = passed_at_call(SMM, "_sa_stage(sp, starts) = _run_sa(")
const DE_PASSED  = union(passed_at_call(SMM, "_de_stage(sp, bank, prev) = _run_de("),
                         passed_at_call(SMM, "theta_opt, loss_opt, niters = _run_de("))

# MCMC_main holds its settings as MCMC_-prefixed globals; run_demc's keywords are the
# lowercased names without the prefix.
const MCMC_HELD = Set(lowercase(String(m.captures[1]))
                      for m in eachmatch(r"(?m)^MCMC_([A-Z0-9_]+)\s*=", MCMC))
const DEMC_PASSED = passed_at_call(MCMC, "run_demc(logposterior, θ0;")

# The stage helpers drop the subsystem prefix (`sa_max_iter` → `max_iter`); without
# stripping it the intersection is empty and the check examines nothing.
"""Names in `fields` that start with `pre`, with the prefix removed."""
strip_prefix(fields, pre) =
    Set(f[(length(pre) + 1):end] for f in fields if startswith(f, pre))

# `sa_halflife` is spelled `cooling_halflife` by _run_sa: the one exception to the prefix rule.
const SA_HELD = union(strip_prefix(RUNPARAMS, "sa_"), Set(["cooling_halflife"]))
const DE_HELD = strip_prefix(RUNPARAMS, "de_")

findings = Tuple{String,String,String}[]   # (check, subject, detail)

for (label, kw, held, passed, defs, pre) in (
        ("run_smm",   SMM_KW,  RUNPARAMS, SMM_PASSED,  SMM_DEF,               ""),
        ("_run_sa",   SA_KW,   SA_HELD,   SA_PASSED,   Dict{String,String}(), "sa_"),
        ("_run_de",   DE_KW,   DE_HELD,   DE_PASSED,   Dict{String,String}(), "de_"),
        ("run_demc",  DEMC_KW, MCMC_HELD, DEMC_PASSED, Dict{String,String}(), ""))
    for k in forwarding_findings(kw, held, passed)
        # Inert settings are exempt; the registry keys the full, prefixed name.
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

# Every `kwarg = run_params.field` pair in smm_main.jl must agree up to a subsystem prefix,
# which check 1 restores; any other difference hides a forwarding gap from check 1.
const NAME_PREFIXES = ("sa_", "de_", "nm_", "cand_", "clusters_", "seed_")

"""True when `kw == field`, or `field` is `kw` with one of `NAME_PREFIXES` in front."""
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

# A launcher setting an unregistered, undeclared key would run on silent defaults.
for f in ("vm_run.sh", "setup_vm.sh", "vm_go.sh")
    path = joinpath(ROOT, "code", "ops", f)
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

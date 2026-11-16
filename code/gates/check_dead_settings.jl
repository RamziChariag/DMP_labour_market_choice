#!/usr/bin/env julia
# gates/check_dead_settings.jl — keyword arguments that arrive and are never read.
#
# check_forwarding.jl confirms a keyword is passed; this confirms the callee reads it.
# For every function in FILES, each keyword in its signature must appear in its own body
# (nested closures included). Works on the AST, so comments, docstrings and multi-line
# annotations cannot produce false matches. A finding is printed for a human: a keyword
# kept for signature compatibility is legitimately unread and says so beside it.
#
#   julia --project=. code/gates/check_dead_settings.jl      # exit 0 clean, 1 findings

using Printf

# Files scanned, relative to the repository root. A missing file is skipped, so this
# list must follow the code layout.
const FILES = ["code/smm/smm.jl", "code/smm/bundle.jl",
               "code/mcmc/demc.jl", "code/mcmc/mcmc_diagnostics.jl"]

"""
    kwargs_of(sig) -> Vector{Symbol}

Keyword names of the function-signature expression `sig`, read from its
`Expr(:parameters, ...)` block (bare, annotated or with a default).
"""
function kwargs_of(sig::Expr)
    out = Symbol[]
    for a in sig.args
        a isa Expr && a.head === :parameters || continue
        for p in a.args
            nm = p
            p isa Expr && p.head === :kw   && (nm = p.args[1])
            nm isa Expr && nm.head === :(::) && (nm = nm.args[1])
            nm isa Symbol && push!(out, nm)
        end
    end
    return out
end

"""
    symbols_in(x, acc) -> Set{Symbol}

Every symbol appearing anywhere in the expression tree `x`, accumulated into `acc`.
"""
function symbols_in(x, acc = Set{Symbol}())
    x isa Symbol && (push!(acc, x); return acc)
    x isa Expr   && (for a in x.args; symbols_in(a, acc); end)
    return acc
end

"""
    walk(x, found)

Push `(name, kwargs, body)` onto `found` for every function definition in `x` that has
keywords, nested definitions included; `body` is the set of symbols in its body.
"""
function walk(x, found)
    if x isa Expr && x.head in (:function, :(=)) && !isempty(x.args) &&
       x.args[1] isa Expr && x.args[1].head === :call
        sig  = x.args[1]
        name = sig.args[1] isa Symbol ? sig.args[1] : :anonymous
        kws  = kwargs_of(sig)
        if !isempty(kws)
            push!(found, (name = name, kwargs = kws,
                          body = symbols_in(length(x.args) > 1 ? x.args[2] : nothing)))
        end
    end
    x isa Expr && for a in x.args; walk(a, found); end
    return found
end

"""
    audit(root) -> (findings, n_checked)

Scan every file of `FILES` under `root`. Returns `(file, function, keyword)` for each
keyword absent from its own body, and the number of keywords checked.
"""
function audit(root::String)
    findings = Tuple{String,Symbol,Symbol}[]
    n_checked = 0
    for rel in FILES
        path = joinpath(root, rel)
        isfile(path) || continue
        for f in walk(Meta.parseall(read(path, String)), [])
            for k in f.kwargs
                n_checked += 1
                k in f.body || push!(findings, (rel, f.name, k))
            end
        end
    end
    return findings, n_checked
end

findings, n_checked = audit(abspath(joinpath(@__DIR__, "..", "..")))

println("check_dead_settings — keyword arguments declared and never read\n")
@printf("  scanned %d keyword arguments across %d files\n\n", n_checked, length(FILES))

if isempty(findings)
    println("PASS — every declared keyword argument is read in its own body.")
    exit(0)
else
    println("FINDINGS ($(length(findings))). Each is declared, plumbed, and unread:")
    for (rel, fn, k) in findings
        println("  $rel   $fn($k = ...)")
    end
    println()
    println("Purge each one that the current version does not need, and record what it")
    println("was for in CHANGELOG.md's rejected section. A keyword kept deliberately for")
    println("signature compatibility should say so in a comment beside it.")
    exit(1)
end

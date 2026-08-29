#!/usr/bin/env julia
# check_dead_settings.jl — find keyword arguments that ARRIVE and are never READ.
#
# THE GAP THIS CLOSES. check_forwarding.jl answers "does the caller's value reach the
# callee?" and passes as soon as the keyword is present at the call site. It cannot see
# whether the callee then does anything with it. sa_target_fin was forwarded from
# run_params through _run_sa into _sa_loop, printed in the settings banner as though it
# governed the walk, and never read in the loop body — so the banner asserted a target
# feasibility fraction that nothing acted on. That is the same failure mode as the
# unforwarded SA settings, one layer deeper.
#
# WHY THE PARSER RATHER THAN grep. A keyword name also appears in comments, in docstrings,
# in unrelated locals, and in nested closures; and a signature's type annotation may span
# commas and braces. Julia's own parser resolves all of that exactly: the function AST
# gives the kwarg list and the body as separate objects, so "declared" and "read" are
# structural facts rather than pattern matches.
#
# USAGE
#   julia --project=. code/scripts/check_dead_settings.jl
#
# EXIT CODE
#   0 clean, 1 findings. A finding is not automatically a bug — a keyword kept for
#   signature compatibility with an older bundle loader is legitimately unread — so each
#   is printed for a human, and the ones that are genuinely obsolete get purged with a
#   note in CHANGELOG.md's rejected section rather than deleted silently.

using Printf

const FILES = ["code/smm/smm.jl", "code/smm/demc.jl", "code/smm/mcmc_diagnostics.jl"]

"""
    kwargs_of(sig) -> Vector{Symbol}

The keyword names of a function-definition signature expression. Keywords live in the
`Expr(:parameters, ...)` block that follows the semicolon; each entry is either a bare
symbol, a `::` annotation, or a `kw` default assignment.
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

Every symbol appearing anywhere in an expression tree. Comments and docstrings are
already gone by parse time, which is the point of doing this on the AST.
"""
function symbols_in(x, acc = Set{Symbol}())
    x isa Symbol && (push!(acc, x); return acc)
    x isa Expr   && (for a in x.args; symbols_in(a, acc); end)
    return acc
end

"""
    walk(x, found)

Collect `(name, kwargs, body_symbols)` for every function definition in the tree,
including nested ones — a keyword read only inside a closure is read.
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

Scan every file in `FILES`, returning one entry per keyword argument that is declared in
a signature and absent from its own body.
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

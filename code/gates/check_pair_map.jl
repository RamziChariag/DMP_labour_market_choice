#!/usr/bin/env julia
############################################################
# gates/check_pair_map.jl — smm_main.jl's pair map against paths.jl's
#
# smm_main.jl keeps its own crisis → baseline map, `_PAIR_BASELINE`, beside the
# CRISIS_BASE switch; every other script pairs windows through `PAIR_BASE` in paths.jl.
# A crisis window held at one base window's estimate and paired with another downstream
# would mix two estimates, so the two maps must name the same pairs. The driver's map is
# read from its parsed source and never evaluated: nothing in smm_main.jl runs.
#
# Run from anywhere:  julia code/gates/check_pair_map.jl
############################################################

include(joinpath(@__DIR__, "..", "repo_root.jl"))
const ROOT = find_repo_root()
include(joinpath(ROOT, "code", "paths.jl"))

"""
    driver_pair_map(path) -> Dict{Symbol,Symbol}

The pairs of `const _PAIR_BASELINE = Dict(:a => :b, ...)` in the source at `path`. Anything
but a literal Dict of symbol pairs is an error, since only a literal can be read without
running the file.
"""
function driver_pair_map(path::AbstractString)
    rhs = Ref{Any}(nothing)
    function walk(ex)
        ex isa Expr || return
        if ex.head === :const && ex.args[1] isa Expr && ex.args[1].head === :(=) &&
           ex.args[1].args[1] === :_PAIR_BASELINE
            rhs[] = ex.args[1].args[2]
        else
            foreach(walk, ex.args)
        end
    end
    walk(Meta.parseall(read(path, String); filename = path))
    d = rhs[]
    literal = d isa Expr && d.head === :call && d.args[1] === :Dict &&
              all(a -> a isa Expr && a.head === :call && a.args[1] === :(=>) &&
                       all(x -> x isa QuoteNode && x.value isa Symbol, a.args[2:3]),
                  d.args[2:end])
    literal || error("$(relpath(path, ROOT)): no `const _PAIR_BASELINE = Dict(:a => :b, ...)` literal")
    return Dict{Symbol,Symbol}(a.args[2].value => a.args[3].value for a in d.args[2:end])
end

println("check_pair_map — smm_main.jl's _PAIR_BASELINE against paths.jl's PAIR_BASE")
driver = driver_pair_map(joinpath(ROOT, "code", "smm", "smm_main.jl"))
if driver == PAIR_BASE
    println("  the two maps name the same $(length(PAIR_BASE)) pairs: OK")
else
    for w in sort(collect(union(keys(driver), keys(PAIR_BASE))))
        a, b = get(driver, w, missing), get(PAIR_BASE, w, missing)
        isequal(a, b) || println("  $w: smm_main.jl → $a, paths.jl → $b")
    end
    exit(1)
end

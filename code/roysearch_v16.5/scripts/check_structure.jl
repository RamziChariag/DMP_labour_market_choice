#!/usr/bin/env julia
############################################################
# scripts/check_structure.jl — structural checks a parser cannot make
#
# Julia parses an assignment inside an open call as a keyword argument, so a
# top-level constant accidentally pasted into a constructor is SYNTACTICALLY VALID
# until something downstream breaks — which is how a misplaced block once closed
# SMMRunParams early and turned the tracing fields into assignments to `false`.
# Meta.parseall and Meta.lower both pass on that state; only bracket depth catches it.
#
# Run from anywhere:  julia scripts/check_structure.jl
############################################################

include(joinpath(@__DIR__, "repo_root.jl"))
const ROOT = find_repo_root()
const CODE = joinpath(ROOT, "code")

"""
    depth_profile(src) -> Vector{Int}

Bracket depth at the START of each line, ignoring brackets inside string literals
and comments. Depth 0 means top level.
"""
function depth_profile(src::AbstractString)
    depths = Int[]
    d = 0
    for line in split(src, '\n')
        push!(depths, d)
        instr = false; esc = false
        for (k, ch) in enumerate(line)
            if esc;                       esc = false; continue; end
            if ch == '\\' && instr;       esc = true;  continue; end
            if ch == '"';                 instr = !instr; continue; end
            instr && continue
            ch == '#' && break                     # rest of the line is a comment
            ch in ('(', '[', '{') && (d += 1)
            ch in (')', ']', '}') && (d -= 1)
        end
    end
    depths
end

"""
    orphan_assignments(path) -> Vector{Tuple{Int,String}}

Lines that LOOK like top-level constant assignments (a bare NAME = ... starting in
column 1) but sit at nonzero bracket depth, i.e. inside an open call.
"""
function orphan_assignments(path::AbstractString)
    src = read(path, String)
    depths = depth_profile(src)
    out = Tuple{Int,String}[]
    for (i, line) in enumerate(split(src, '\n'))
        occursin(r"^[A-Z_][A-Z0-9_]*\s*=", line) || continue
        depths[i] == 0 || push!(out, (i, strip(line)))
    end
    out
end

"""
    orphan_kwargs(path) -> Vector{Tuple{Int,String}}

Lines that look like a keyword argument — indented `name = value,` with a trailing
comma — but sit at bracket depth 0. A constructor closed early leaves its remaining
fields in exactly this state, and Julia then parses `show_trace_members = false,`
as a tuple assignment rather than a keyword, which is the failure that shipped once.
"""
function orphan_kwargs(path::AbstractString)
    src = read(path, String)
    depths = depth_profile(src)
    out = Tuple{Int,String}[]
    for (i, line) in enumerate(split(src, '\n'))
        occursin(r"^\s+[A-Za-z_][A-Za-z0-9_]*\s*=.*,\s*(#.*)?$", line) || continue
        depths[i] == 0 || continue
        # A multi-line call's OWN first line also ends in a comma at depth 0
        # (`p = plot(x, y,`), so require the line to close every bracket it opens.
        depths[i + 1] == 0 || continue
        push!(out, (i, strip(line)))
    end
    out
end

"""
    unbalanced(path) -> Int

Net bracket depth at end of file; anything but 0 means an unclosed construct.
"""
unbalanced(path::AbstractString) = (p = depth_profile(read(path, String)); isempty(p) ? 0 : last(p))

function main()
    files = String[]
    for (root, _, fs) in walkdir(CODE), f in fs
        endswith(f, ".jl") && push!(files, joinpath(root, f))
    end
    sort!(files)
    nbad = 0
    for f in files
        rel = relpath(f, ROOT)
        u = unbalanced(f)
        if u != 0
            println("  UNBALANCED  $rel  (net depth $u at EOF)")
            nbad += 1
        end
        for (ln, txt) in orphan_assignments(f)
            println("  MISPLACED   $rel:$ln  inside an open call:  $(first(txt, 60))")
            nbad += 1
        end
        for (ln, txt) in orphan_kwargs(f)
            println("  ORPHANED    $rel:$ln  keyword field outside its call:  $(first(txt, 55))")
            nbad += 1
        end
    end
    println(nbad == 0 ? "  $(length(files)) files: structure OK" :
                        "  $(length(files)) files: $nbad problem(s)")
    nbad
end

main()

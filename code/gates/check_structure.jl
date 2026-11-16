#!/usr/bin/env julia
############################################################
# gates/check_structure.jl — structural checks a parser cannot make
#
# Julia parses `NAME = value` inside an open call as a keyword argument, so a constant
# pasted into a constructor, or a constructor closed early, still parses and lowers.
# Bracket depth per line catches both. Prints every finding; scans every .jl under code/.
#
#   julia code/gates/check_structure.jl
############################################################

include(joinpath(@__DIR__, "..", "repo_root.jl"))
const ROOT = find_repo_root()
const CODE = joinpath(ROOT, "code")

"""
    depth_profile(src) -> Vector{Int}

Bracket depth at the start of each line of `src`, ignoring brackets inside
double-quoted strings and after `#`. Depth 0 is top level.
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

Lines of the file at `path` that look like a top-level constant assignment (an
uppercase `NAME = ...` in column 1) but sit inside an open call. Returns
`(line number, text)` pairs.
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

Lines of the file at `path` that look like a keyword argument (indented
`name = value,`) but sit at depth 0, as the remaining fields of a constructor closed
early do. Returns `(line number, text)` pairs.
"""
function orphan_kwargs(path::AbstractString)
    src = read(path, String)
    depths = depth_profile(src)
    out = Tuple{Int,String}[]
    for (i, line) in enumerate(split(src, '\n'))
        occursin(r"^\s+[A-Za-z_][A-Za-z0-9_]*\s*=.*,\s*(#.*)?$", line) || continue
        depths[i] == 0 || continue
        # The first line of a multi-line call (`p = plot(x, y,`) also matches; require
        # the next line to be back at depth 0.
        depths[i + 1] == 0 || continue
        push!(out, (i, strip(line)))
    end
    out
end

"""
    unbalanced(path) -> Int

Net bracket depth at the end of the file at `path`; nonzero means an unclosed construct.
"""
unbalanced(path::AbstractString) = (p = depth_profile(read(path, String)); isempty(p) ? 0 : last(p))

"""
    main() -> Int

Run the three checks on every .jl file under code/, print each finding, and return the
number of findings.
"""
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

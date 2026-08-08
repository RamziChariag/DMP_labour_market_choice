############################################################
# scripts/repo_root.jl — locate the repository root from any script
#
# Scripts are not pinned to one depth: `scripts/` may sit at the repo root or
# under `code/`, and a fixed `joinpath(@__DIR__, "..")` silently resolves to the
# wrong directory in the second case — which both writes generated files to the
# wrong place and builds include paths like `code/code/solver`.  Walking up for a
# marker makes the answer independent of where the script is placed.
############################################################

"""
    find_repo_root(start = @__DIR__) -> String

Nearest ancestor of `start` that holds both `code/smm/smm_main.jl` and
`data/derived/windows.json`.  Both are required: `code/` alone also matches a
stale snapshot of the tree, and `data/derived/` alone matches nothing useful.
"""
function find_repo_root(start :: AbstractString = @__DIR__) :: String
    dir = abspath(start)
    while true
        isfile(joinpath(dir, "code", "smm", "smm_main.jl")) &&
            isfile(joinpath(dir, "data", "derived", "windows.json")) && return dir
        parent = dirname(dir)
        parent == dir && error("""
            Could not find the repository root above $(abspath(start)).
            Expected an ancestor containing both
              code/smm/smm_main.jl
              data/derived/windows.json""")
        dir = parent
    end
end

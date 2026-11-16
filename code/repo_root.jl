############################################################
# code/repo_root.jl — locate the repository root from any script
#
# Walks up from the caller to the first directory holding the repository's markers, so a
# script finds the root whatever its depth in the tree.
############################################################

"""
    find_repo_root(start = @__DIR__) -> String

Nearest ancestor of `start` holding both `code/smm/smm_main.jl` and
`data/derived/windows.json` (`code/` alone also matches a stale snapshot of the tree).
Errors if there is none.
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

#!/usr/bin/env julia
############################################################
# scripts/export_env.jl — pin the estimation stack into the repo
#
# The estimation runs from a STACK of environments, not one.  Julia's LOAD_PATH is
# ["@", "@v#.#", "@stdlib"]: `using` searches the active project first, then the
# shared default environment, then the standard library.  On this project that
# split is real — QuasiMonteCarlo resolves from the repo's own Project.toml, and
# CSV, Optim, Distributions and the rest resolve from the default environment.
# Neither holds the whole stack, which is why the code runs while no single
# environment looks complete.
#
# A stack is not reproducible: the default half lives outside the repo and drifts
# as other projects add packages.  This script reads the version and UUID of each
# package from whichever environment actually supplies it, and writes them into one
# project-local Project.toml.  Nothing about how the code runs here changes; the
# repo simply stops depending on a directory that is not in it.
#
# Usage (a plain terminal, from anywhere):
#   julia scripts/export_env.jl
#
# Writes: Project.toml  (deps + exact "=" version pins)
#         Manifest.toml (full resolved tree)
############################################################

using Pkg, Printf

# Packages the two estimation entry points load.  Deliberately excludes Plots and
# LaTeXStrings (figures) and HTTP/XLSX/Arrow (data construction): a compute VM runs
# the estimation only, and those pull large binary artifacts that add install time
# and failure modes without ever being used.
const ESTIMATION_DEPS = [
    "CSV", "Clustering", "DataFrames", "Distributions", "FastGaussQuadrature",
    "Interpolations", "JSON3", "Optim", "Parameters", "QuasiMonteCarlo",
]

# Standard-library modules the code loads.  A project environment must declare them
# to load them, but they carry no version.
const STDLIBS = [
    "Dates", "DelimitedFiles", "LinearAlgebra", "Printf", "Random",
    "Serialization", "SparseArrays", "Statistics",
]

include(joinpath(@__DIR__, "repo_root.jl"))
const REPO_ROOT = find_repo_root()

"""
    stack_in(env) -> Dict{String,Tuple{String,String}}

Packages installed in `env`, as `name => (uuid, version)`.  UUIDs come from the
resolved dependency graph, so they are the ones this machine uses — never
transcribed by hand and never needing a registry lookup.

`Pkg.dependencies()` raises on a project whose Manifest is absent or stale, which
is the ordinary state of a partially-populated environment, so an unreadable one
contributes nothing rather than aborting the search.
"""
function stack_in(env)
    Pkg.activate(env; io = devnull)
    out = Dict{String,Tuple{String,String}}()
    deps = try
        Pkg.dependencies()
    catch
        return out
    end
    for (uuid, info) in deps
        info.version === nothing && continue     # stdlibs carry no version
        out[info.name] = (string(uuid), string(info.version))
    end
    return out
end

function main()
    original = Base.active_project()

    # Walk the load path in Julia's own order, so a package present in two
    # environments is taken from the one `using` would actually reach first.
    envs = filter(p -> endswith(p, ".toml"), Base.load_path())

    @printf("Julia %s\n\nEnvironment stack (LOAD_PATH order):\n", VERSION)

    found  = Dict{String,Tuple{String,String}}()   # name => (uuid, version)
    source = Dict{String,String}()                 # name => environment that supplies it
    for env in envs
        stack = stack_in(env)
        mine  = [p for p in ESTIMATION_DEPS if haskey(stack, p) && !haskey(found, p)]
        @printf("  %s\n", env)
        if isempty(mine)
            println("      (supplies none of the estimation packages)")
        else
            for p in sort(mine)
                found[p]  = stack[p]
                source[p] = env
                @printf("      %-22s %s\n", p, stack[p][2])
            end
        end
    end
    original === nothing || Pkg.activate(original; io = devnull)

    absent = filter(p -> !haskey(found, p), ESTIMATION_DEPS)
    if !isempty(absent)
        @printf("\nNot installed anywhere on the load path: %s\n", join(absent, ", "))
        println("Add them to your default environment, then re-run. Nothing was written.")
        return 1
    end

    n_env = length(unique(values(source)))
    @printf("\nAll %d packages found across %d environment%s.\n",
            length(ESTIMATION_DEPS), n_env, n_env == 1 ? "" : "s")

    deps   = Dict{String,String}(p => found[p][1] for p in ESTIMATION_DEPS)
    compat = Dict{String,String}("julia" => string(VERSION.major, ".", VERSION.minor))
    for p in ESTIMATION_DEPS
        compat[p] = "=" * found[p][2]            # exact pin: the VM gets this version
    end
    for s in STDLIBS
        id = Base.identify_package(s)
        id === nothing ? @printf("  (skipping unknown stdlib %s)\n", s) :
                         (deps[s] = string(id.uuid))
    end

    open(joinpath(REPO_ROOT, "Project.toml"), "w") do io
        println(io, "# Generated by scripts/export_env.jl — do not edit by hand.")
        println(io, "# Collapses the environment stack this project used to run from")
        println(io, "# (repo project + shared default) into one self-contained project.")
        println(io, "# Restore with:  julia --project=. -e 'using Pkg; Pkg.instantiate()'")
        println(io)
        println(io, "name = \"RoySearch\"")
        println(io)
        println(io, "[deps]")
        for k in sort(collect(keys(deps)))
            println(io, "$k = \"$(deps[k])\"")
        end
        println(io)
        println(io, "[compat]")
        for k in sort(collect(keys(compat)))
            println(io, "$k = \"$(compat[k])\"")
        end
    end

    Pkg.activate(REPO_ROOT; io = devnull)
    try
        Pkg.resolve()
    catch e
        println("\nProject.toml was written, but resolve failed:")
        println("  ", sprint(showerror, e))
        println("The exact \"=\" pins may not co-resolve from a clean state. Relax the")
        println("offending [compat] entry to a caret bound and re-run Pkg.resolve().")
        return 1
    end
    original === nothing || Pkg.activate(original; io = devnull)

    @printf("\nWrote %s\n      %s\n",
            joinpath(REPO_ROOT, "Project.toml"), joinpath(REPO_ROOT, "Manifest.toml"))
    println("Commit both. On the VM: julia --project=. -e 'using Pkg; Pkg.instantiate()'")
    return 0
end

exit(main())

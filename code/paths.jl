############################################################
# paths.jl — the ONLY place an output path is constructed.
#
# THE PROBLEM THIS SOLVES. Before v19.6.0, OUTPUT_DIR / PLOTS_DIR / TABLES_DIR /
# SMM_OUT_DIR were declared independently in seven files — smm_main.jl, MCMC_main.jl,
# transition_main.jl, transition_panel.jl, model_main.jl, policy_main.jl and
# plots_and_tables.jl — and PROJECT_ROOT in ten. The last two used `if !@isdefined(...)`
# guards, so their effective values depended on which file was included first. That is the
# same defect class as the settings-forwarding bug: a value that is declared in one place
# and effective in another, with nothing checking that the two agree.
#
# With five exhibit categories times four windows the duplication would have got much
# worse, so it is centralised before the restructure rather than after.
#
# WHY VALIDATION IS THE POINT. `out_plots(:model, "base_cf")` must be a startup error, not
# a directory called base_cf sitting quietly beside base_fc with one figure in it. Every
# accessor checks its category against CATEGORIES and its unit against the list that
# category admits, so a typo fails loudly at the call rather than silently on disk.
#
# THE TREE THIS DEFINES
#
#   output/
#   ├── estimates/    TRACKED   estimate_<window>_diagonalW.{jls,csv} — one per window,
#   │                           progressively refined: smm_main writes it, MCMC overwrites
#   │                           it with the better point plus the standard errors
#   ├── chains/       ignored   transient MCMC draws, deletable once the SEs are sealed
#   ├── transition/   TRACKED   what plots_and_tables/transition.jl reads
#   ├── logs/         TRACKED   the record of what each run did
#   ├── plots/        ignored   descriptives/ fit/<w>/ model/<w>/ transition/<pair>/
#   │                           policy/<w>/ manual/
#   └── tables/       ignored   .tex ONLY, same subdivision
#
# `manual/` exists only under plots/: model_main.jl solves at arbitrary parameters, which
# produces figures to look at and never a paper table. It replaces standalone_default/ and
# single_run/, which were one thing under two names.
#
# The transition unit is the regime PAIR — "fc" is base_fc → crisis_fc — so two
# subdirectories there, not four.
############################################################

const ROYSEARCH_PATHS_LOADED = true

# This file lives at <root>/code/paths.jl, so the root is one level above its directory.
# Deriving it here rather than taking it as an argument means a scratch copy of the repo
# resolves to the scratch root automatically, which is how the verification harnesses run.
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

const WINDOWS    = ("base_fc", "crisis_fc", "base_covid", "crisis_covid")
const PAIRS      = ("fc", "covid")
const CATEGORIES = (:descriptives, :fit, :model, :transition, :policy, :manual)

# Which units each category admits. A category with an empty tuple takes no unit at all,
# and passing one is an error rather than a silently-created subdirectory.
const _CATEGORY_UNITS = Dict(:descriptives => (),
                            :fit          => WINDOWS,
                            :model        => WINDOWS,
                            :transition   => PAIRS,
                            :policy       => WINDOWS,
                            :manual       => ())

const OUTPUT_DIR = joinpath(PROJECT_ROOT, "output")
const CODE_DIR   = joinpath(PROJECT_ROOT, "code")
const DATA_DIR   = joinpath(PROJECT_ROOT, "data")

function _check(category::Symbol, unit::AbstractString)
    category in CATEGORIES ||
        error("paths: unknown category :$category — must be one of $(CATEGORIES)")
    admitted = _CATEGORY_UNITS[category]
    if isempty(unit)
        isempty(admitted) && return
        error("paths: category :$category requires a unit, one of $(admitted)")
    end
    isempty(admitted) &&
        error("paths: category :$category takes no unit, got \"$unit\"")
    unit in admitted ||
        error("paths: \"$unit\" is not a valid unit for :$category — must be one of $(admitted)")
    return
end

"""
    out_plots(category, unit = "") -> String

Directory for figures of `category`, creating it if absent. `unit` is the window for
`:fit`, `:model` and `:policy`, the regime pair for `:transition`, and must be omitted for
`:descriptives` and `:manual`.
"""
function out_plots(category::Symbol, unit::AbstractString = "")
    _check(category, unit)
    d = isempty(unit) ? joinpath(OUTPUT_DIR, "plots", String(category)) :
                        joinpath(OUTPUT_DIR, "plots", String(category), unit)
    mkpath(d); return d
end

"""
    out_tables(category, unit = "") -> String

Directory for `.tex` tables of `category`. Same unit rules as `out_plots`. Nothing but
`.tex` belongs here: a `.csv` is a machine artifact and lives beside the bundle it came
from, so that a table directory can be deleted and regenerated without thinking about it.
"""
function out_tables(category::Symbol, unit::AbstractString = "")
    _check(category, unit)
    category === :manual &&
        error("paths: :manual has no tables — a solve at arbitrary parameters produces " *
              "figures to look at, never a paper table")
    d = isempty(unit) ? joinpath(OUTPUT_DIR, "tables", String(category)) :
                        joinpath(OUTPUT_DIR, "tables", String(category), unit)
    mkpath(d); return d
end

"""
    out_estimates() -> String

Where the estimate bundles and their sibling CSVs live. Formerly `output/smm`, renamed
because MCMC writes here too and `smm` became a wrong name the moment it did.
"""
out_estimates() = (d = joinpath(OUTPUT_DIR, "estimates"); mkpath(d); d)

"""
    out_chains() -> String

Transient MCMC draws. Not tracked: once the standard errors are sealed into the estimate
bundle nothing needs the draw arrays, and they are large.
"""
out_chains() = (d = joinpath(OUTPUT_DIR, "chains"); mkpath(d); d)

out_transition() = (d = joinpath(OUTPUT_DIR, "transition"); mkpath(d); d)

"""
    out_logs() -> String

Run logs AND measurement records. A diagnostic CSV — a width scan, an arm comparison, a
feasibility probe — is a record of what a run measured, not an estimate, so it belongs here
rather than in `estimates/`, which holds only the bundles and their siblings. Tracked, for
the same reason logs are: nothing needs them to run, but they are the only evidence behind a
design choice, and a choice whose evidence has been deleted is a choice nobody can revisit.
"""
out_logs()       = (d = joinpath(OUTPUT_DIR, "logs");       mkpath(d); d)

"""
    estimate_path(window, w_suffix; ext = "jls") -> String

The one filename convention for an estimate bundle and its CSV sibling. Going through this
rather than interpolating at each site is what lets a gate assert that `output/estimates/`
contains nothing else.
"""
function estimate_path(window, w_suffix::AbstractString; ext::AbstractString = "jls")
    w = String(window)
    w in WINDOWS || error("paths: unknown window \"$w\" — must be one of $(WINDOWS)")
    return joinpath(out_estimates(), "estimate_$(w)$(w_suffix).$(ext)")
end

"""
    chain_path(window, w_suffix) -> String

The chain-draw bundle for a window.
"""
function chain_path(window, w_suffix::AbstractString)
    w = String(window)
    w in WINDOWS || error("paths: unknown window \"$w\" — must be one of $(WINDOWS)")
    return joinpath(out_chains(), "chain_$(w)$(w_suffix).jls")
end

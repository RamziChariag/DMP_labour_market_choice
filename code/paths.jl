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
#   ├── estimates/    TRACKED   estimate_<window>_diagonalW_<point>.jls and their CSV
#   │                           siblings, one file per estimate (ESTIMATE_POINTS below):
#   │                           smm_main writes the SMM ones, a chain run the _mcmc one,
#   │                           and neither writes the other's
#   ├── chains/       ignored   transient MCMC draws
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
# The transition unit is the regime PAIR — "fc" is the base_fc period of the baseline →
# crisis_fc — so one subdirectory per pair there, not one per window.
############################################################

const ROYSEARCH_PATHS_LOADED = true

# This file lives at <root>/code/paths.jl, so the root is one level above its directory.
# Deriving it here rather than taking it as an argument means a scratch copy of the repo
# resolves to the scratch root automatically, which is how the verification harnesses run.
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

# The estimation windows (WINDOW): the baseline, one economy estimated jointly on its two
# periods, and the two crisis windows.
const WINDOWS      = ("baseline", "crisis_fc", "crisis_covid")
# The data windows of windows.json, each with its own moments, sampling variances and ν.
const DATA_WINDOWS = ("base_fc", "crisis_fc", "base_covid", "crisis_covid")
# The baseline's periods, in the order its parameter vector and moments run.
const BASELINE_PERIODS = (:base_fc, :base_covid)
# A transition runs from the baseline period PAIR_BASE[crisis_<pair>] to `crisis_<pair>`.
const PAIRS      = ("fc", "covid")
const CATEGORIES = (:descriptives, :fit, :model, :transition, :policy, :manual)

# Which units each category admits. A category with an empty tuple takes no unit at all,
# and passing one is an error rather than a silently-created subdirectory.
const _CATEGORY_UNITS = Dict(:descriptives => (),
                            :fit          => DATA_WINDOWS,
                            :model        => DATA_WINDOWS,
                            :transition   => PAIRS,
                            :policy       => DATA_WINDOWS,
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

# The baseline period each data window is paired with. A crisis window holds parameters at
# that period of the baseline estimate, a transition runs between the two, and ν is read off
# that period's row.
const PAIR_BASE = Dict(:base_fc    => :base_fc,    :crisis_fc    => :base_fc,
                       :base_covid => :base_covid, :crisis_covid => :base_covid)
is_crisis(window) = Symbol(window) in (:crisis_fc, :crisis_covid)

# The data windows an estimation window is estimated on, and the estimation window whose
# estimate holds a data window.
data_windows(window) = Symbol(window) === :baseline ? BASELINE_PERIODS : (Symbol(window),)
estimation_window(dw) = Symbol(dw) in BASELINE_PERIODS ? :baseline : Symbol(dw)

# The estimates a window can hold, each in a file of its own so that no run overwrites
# another's:
#   :smm           the SMM estimate (smm_main.jl; a crisis window with CRISIS_BASE = :smm)
#   :smm_mcmcbase  a crisis window's SMM estimate with its held parameters at the baseline
#                  chain's estimate (smm_main.jl, CRISIS_BASE = :mcmc); the crisis chain's seed
#   :mcmc          the chain's estimate (MCMC_main.jl, a chain run)
const ESTIMATE_POINTS = (:smm, :smm_mcmcbase, :mcmc)

function _estimates_file(stem::AbstractString, window, w_suffix::AbstractString,
                         tag::AbstractString, ext::AbstractString)
    w = String(window)
    w in WINDOWS || error("paths: unknown window \"$w\" — must be one of $(WINDOWS)")
    return joinpath(out_estimates(), "$(stem)_$(w)$(w_suffix)$(tag).$(ext)")
end

function _point_tag(point::Symbol, admitted)
    point in admitted || error("paths: point :$point — must be one of $(admitted)")
    return "_" * String(point)
end

"""
    estimate_path(window, w_suffix; point) -> String

The estimate bundle of `window` for `point` (one of `ESTIMATE_POINTS`). `point` has no
default, so every reader and writer names the estimate it means. Going through this rather
than interpolating at each site is what lets a gate assert that `output/estimates/` contains
nothing else.
"""
estimate_path(window, w_suffix::AbstractString; point::Symbol) =
    _estimates_file("estimate", window, w_suffix, _point_tag(point, ESTIMATE_POINTS), "jls")

"""
    mcmcbest_path(window, w_suffix) -> String

The best point a chain visited, written only with MCMC_CHECKPOINT = true.
"""
mcmcbest_path(window, w_suffix::AbstractString) =
    _estimates_file("estimate", window, w_suffix, "_mcmcbest", "jls")

"""
    candidates_path(window, w_suffix; point) -> String

The `:clusters` candidate bank, tagged like the bundle its run writes (`:smm` or
`:smm_mcmcbase`): the bank's Q values depend on the held parameters.
"""
candidates_path(window, w_suffix::AbstractString; point::Symbol) =
    _estimates_file("candidates", window, w_suffix, _point_tag(point, (:smm, :smm_mcmcbase)), "jls")

# The CSV siblings: smm_main.jl's estimates (`:smm` or `:smm_mcmcbase`, the tag of its
# bundle) and MCMC_main.jl's standard errors (`:smm` for the chain-free pass on the SMM
# estimate, `:mcmc` for a chain run).
smm_csv_path(window, w_suffix::AbstractString; point::Symbol) =
    _estimates_file("smm_estimates", window, w_suffix, _point_tag(point, (:smm, :smm_mcmcbase)), "csv")
mcmc_csv_path(window, w_suffix::AbstractString; point::Symbol) =
    _estimates_file("mcmc_results", window, w_suffix, _point_tag(point, (:smm, :mcmc)), "csv")

"""
    chain_path(window, w_suffix) -> String

The chain-draw bundle for a window.
"""
function chain_path(window, w_suffix::AbstractString)
    w = String(window)
    w in WINDOWS || error("paths: unknown window \"$w\" — must be one of $(WINDOWS)")
    return joinpath(out_chains(), "chain_$(w)$(w_suffix).jls")
end

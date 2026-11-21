############################################################
# paths.jl — every output path, and the names of windows, pairs and estimates
#
# The only place an output path is built. Each accessor checks its category against
# CATEGORIES and its unit against the units that category admits, so a typo fails at the
# call instead of creating a stray directory.
#
#   output/
#   ├── estimates/        TRACKED  estimate bundles (one file per ESTIMATE_POINTS entry),
#   │                              candidate banks and their CSV siblings
#   ├── chains/           ignored  transient MCMC draws
#   ├── transition/       TRACKED  transition bundles (transition_main.jl)
#   ├── counterfactuals/           <exercise>/: CSV tables, figure data and log.txt
#   ├── logs/             TRACKED  run logs and measurement records
#   ├── plots/            ignored  descriptives/ fit/<w>/ model/<w>/ transition/<pair>/
#   │                              manual/
#   └── tables/           ignored  .tex only, same subdivision without manual/
#
# The transition unit is the regime pair: "fc" runs from the base_fc period of the
# baseline to crisis_fc.
############################################################

const ROYSEARCH_PATHS_LOADED = true

# This file is <root>/code/paths.jl, so a scratch copy of the repo resolves to its own root.
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

# The estimation windows (WINDOW): the baseline, estimated jointly on its two periods, and
# the two crisis windows.
const WINDOWS      = ("baseline", "crisis_fc", "crisis_covid")
# The data windows of windows.json, each with its own moments, sampling variances and ν.
const DATA_WINDOWS = ("base_fc", "crisis_fc", "base_covid", "crisis_covid")
# The baseline's periods, in the order its parameter vector and moments run.
const BASELINE_PERIODS = (:base_fc, :base_covid)
# A transition runs from the baseline period PAIR_BASE[crisis_<pair>] to `crisis_<pair>`.
const PAIRS      = ("fc", "covid")
const CATEGORIES = (:descriptives, :fit, :model, :transition, :manual)

# The units each category admits; an empty tuple means the category takes no unit.
const _CATEGORY_UNITS = Dict(:descriptives => (),
                            :fit          => DATA_WINDOWS,
                            :model        => DATA_WINDOWS,
                            :transition   => PAIRS,
                            :manual       => ())

const OUTPUT_DIR = joinpath(PROJECT_ROOT, "output")
const CODE_DIR   = joinpath(PROJECT_ROOT, "code")
const DATA_DIR   = joinpath(PROJECT_ROOT, "data")

"""
    _check(category, unit)

Error unless `category` is one of `CATEGORIES` and `unit` is one it admits (empty when it
admits none).
"""
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
`:fit` and `:model`, the regime pair for `:transition`, and must be omitted for
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

Directory for the `.tex` tables of `category`, creating it if absent; same unit rules as
`out_plots`. Only `.tex` belongs here (a `.csv` lives beside the bundle it came from), and
`:manual` has no tables.
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

Directory of the estimate bundles, candidate banks and their sibling CSVs.
"""
out_estimates() = (d = joinpath(OUTPUT_DIR, "estimates"); mkpath(d); d)

"""
    out_chains() -> String

Directory of the transient MCMC draws (not tracked).
"""
out_chains() = (d = joinpath(OUTPUT_DIR, "chains"); mkpath(d); d)

"""
    out_transition() -> String

Directory of the transition bundles.
"""
out_transition() = (d = joinpath(OUTPUT_DIR, "transition"); mkpath(d); d)

# The exercises of code/counterfactuals/, one output directory each.
const CF_EXERCISES = (:held_margin, :incidence, :training_cost, :free_entry_curves)

"""
    out_counterfactuals(exercise) -> String

Directory of one counterfactual exercise (`CF_EXERCISES`): its CSV tables, figure data and
log.
"""
function out_counterfactuals(exercise)
    e = Symbol(exercise)
    e in CF_EXERCISES ||
        error("paths: unknown counterfactual exercise :$e — must be one of $(CF_EXERCISES)")
    d = joinpath(OUTPUT_DIR, "counterfactuals", String(e));  mkpath(d);  return d
end

"""
    out_logs() -> String

Directory of run logs and measurement records (a diagnostic CSV is a record of what a run
measured, not an estimate, so it lives here and not in `estimates/`).
"""
out_logs()       = (d = joinpath(OUTPUT_DIR, "logs");       mkpath(d); d)

# The baseline period each data window is paired with: a crisis window holds parameters at
# that period of the baseline estimate, a transition runs between the two, and ν is read off
# that period's row.
const PAIR_BASE = Dict(:base_fc    => :base_fc,    :crisis_fc    => :base_fc,
                       :base_covid => :base_covid, :crisis_covid => :base_covid)

"""
    is_crisis(window) -> Bool

Whether `window` is a crisis window.
"""
is_crisis(window) = Symbol(window) in (:crisis_fc, :crisis_covid)

"""
    data_windows(window) -> Tuple

The data windows the estimation window `window` is estimated on.
"""
data_windows(window) = Symbol(window) === :baseline ? BASELINE_PERIODS : (Symbol(window),)

"""
    estimation_window(dw) -> Symbol

The estimation window whose estimate holds the data window `dw`.
"""
estimation_window(dw) = Symbol(dw) in BASELINE_PERIODS ? :baseline : Symbol(dw)

# The estimates a window can hold, each in a file of its own so that no run overwrites
# another's:
#   :smm           the SMM estimate (smm_main.jl; a crisis window with CRISIS_BASE = :smm)
#   :smm_mcmcbase  a crisis window's SMM estimate with its held parameters at the baseline
#                  chain's estimate (smm_main.jl, CRISIS_BASE = :mcmc); the crisis chain's seed
#   :mcmc          the chain's estimate (MCMC_main.jl, a chain run)
const ESTIMATE_POINTS = (:smm, :smm_mcmcbase, :mcmc)

"""
    _estimates_file(stem, window, w_suffix, tag, ext) -> String

`output/estimates/<stem>_<window><w_suffix><tag>.<ext>`, erroring unless `window` is one of
`WINDOWS`.
"""
function _estimates_file(stem::AbstractString, window, w_suffix::AbstractString,
                         tag::AbstractString, ext::AbstractString)
    w = String(window)
    w in WINDOWS || error("paths: unknown window \"$w\" — must be one of $(WINDOWS)")
    return joinpath(out_estimates(), "$(stem)_$(w)$(w_suffix)$(tag).$(ext)")
end

"""
    _point_tag(point, admitted) -> String

The file tag `"_<point>"`, erroring unless `point` is in `admitted`.
"""
function _point_tag(point::Symbol, admitted)
    point in admitted || error("paths: point :$point — must be one of $(admitted)")
    return "_" * String(point)
end

"""
    estimate_path(window, w_suffix; point) -> String

The estimate bundle of `window` for `point` (one of `ESTIMATE_POINTS`). `point` has no
default, so every reader and writer names the estimate it means.
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

"""
    smm_csv_path(window, w_suffix; point) -> String

The CSV of smm_main.jl's estimates, tagged like its bundle (`:smm` or `:smm_mcmcbase`).
"""
smm_csv_path(window, w_suffix::AbstractString; point::Symbol) =
    _estimates_file("smm_estimates", window, w_suffix, _point_tag(point, (:smm, :smm_mcmcbase)), "csv")

"""
    mcmc_csv_path(window, w_suffix; point) -> String

The CSV of MCMC_main.jl's standard errors: `:smm` for the chain-free pass at the SMM
estimate, `:mcmc` for a chain run.
"""
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

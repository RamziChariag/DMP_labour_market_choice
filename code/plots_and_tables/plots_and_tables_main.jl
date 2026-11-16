############################################################
# plots_and_tables/plots_and_tables_main.jl — figures and tables of the paper
#
# Usage (from the repo root):
#   julia --project=. code/plots_and_tables/plots_and_tables_main.jl
#
# Reads the stored results (output/estimates, output/transition, output/counterfactuals)
# and writes figures to output/plots/<category>/ and .tex tables to
# output/tables/<category>/. Every directory comes from out_plots(category, unit) and
# out_tables(category, unit) in paths.jl; no path is built here.
############################################################

include(joinpath(@__DIR__, "..", "smm", "version.jl"))
println("="^60)
println("  RoySearch v$(ROYSEARCH_VERSION) — Plots and tables")
println("="^60)
flush(stdout)

include(joinpath(@__DIR__, "..", "paths.jl"))

# One file per category of plots and tables (paths.jl CATEGORIES) is included here as it
# is written; each defines and calls the functions that draw and save its exhibits.

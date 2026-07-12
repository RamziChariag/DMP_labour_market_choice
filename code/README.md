# RoySearch — replication package

A continuous-time, two-market search-and-matching model of education choice,
estimated by the simulated method of moments (SMM). Workers carry two abilities,
`a_U` (unskilled aptitude) and `a_S` (skilled aptitude), correlated through a
Gaussian copula parameter `ρ_x`. Production is linear in own ability on each
side (`π_U = exp(A)·P_U·a_U·p`, `π_S = exp(A)·P_S·a_S·p`); the negative
concordance between the two abilities (Gola mechanism) is what keeps both
markets populated at fixed productivity. Training is an education-choice
frontier `τ(a_U, a_S)`; the cross-market search option `d` is always priced but
carries ≈0 steady-state flow by self-selection.

This README is the reproduction contract: a reader who has read the model notes
can reproduce every reported number from the scripts named here.

## Layout

```
roysearch/
  Project.toml, Manifest.toml   pinned environment (Julia 1.10.10)
  solver/                       model solver, loaded as a library
    grids.jl                    matching tech, Gauss–Jacobi ability grid,
                                  Gaussian copula, training cost, tail weights
    params.jl                   Parameters.jl structs; Unicode fields; caches
    unskilled.jl                unskilled block (values 1D in a_U; τ frontier)
    skilled.jl                  skilled block (values 1D in a_S; directed d)
    solver.jl                   global nested fixed point (U–S coupling)
    equilibrium.jl              stationary densities, wages, all moment inputs
    plots.jl                    equilibrium plot set (rank space; shared style)
  smm/
    smm_main.jl                 estimation ENTRY POINT
    moments.jl                  data-moment / weight loaders + model_moments
    smm_params.jl               parameter names, bounds, transforms, SMMSpec
    smm.jl                      objective, weighting, SA/DE/NM, serialization
    candidates.jl               Sobol → hclust → multistart seed bank
    MCMC_main.jl                standard-error ENTRY POINT (DE-MC quasi-posterior)
    demc.jl                     differential-evolution MCMC sampler (model-agnostic)
  transition/
    transition_main.jl          transition/out-of-sample-validation ENTRY POINT
    transition_params.jl        TransitionParams / TransitionPath / TransitionResult
    transition_solver.jl        backward–forward transition (nested-FP per date)
    transition_simulation.jl    per-scenario driver: z₀/z₁ solves → solve_transition
    transition_panel.jl         data-vs-model dynamics panels
    plots_and_tables.jl         model-fit + parameter LaTeX tables, fit scatter
  data/
    derived/                    windows.json (single source of truth),
                                  moments_{w}.csv, sigma_{w}.csv,
                                  nu_estimation.csv, phi_calibration.csv,
                                  training_share_scale.csv
  output/
    tables/                     smm_estimates_{w}_equalW.csv
    smm/                        serialised SMMResult bundles (.jls)
```

## Object → output crosswalk

The estimation entry point is a single command; which window it targets is set
by the `WINDOW` constant at the top of `smm/smm_main.jl`. Each run writes one
CSV of estimates and one serialised bundle.

| Output | Produced by | Command | Written to |
|---|---|---|---|
| Baseline estimates (FC) | `smm/smm_main.jl`, `WINDOW=:base_fc` | `julia --project --threads auto smm/smm_main.jl` | `output/tables/smm_estimates_base_fc_equalW.csv`, `output/smm/smm_result_base_fc_equalW.jls` |
| Baseline estimates (COVID) | `smm/smm_main.jl`, `WINDOW=:base_covid` | same | `output/tables/smm_estimates_base_covid_equalW.csv`, `output/smm/smm_result_base_covid_equalW.jls` |
| Crisis estimates (FC) | `smm/smm_main.jl`, `WINDOW=:crisis_fc` | same (loads `base_fc` bundle, fixes deep params) | `output/tables/smm_estimates_crisis_fc_equalW.csv`, `output/smm/smm_result_crisis_fc_equalW.jls` |
| Crisis estimates (COVID) | `smm/smm_main.jl`, `WINDOW=:crisis_covid` | same (loads `base_covid` bundle) | `output/tables/smm_estimates_crisis_covid_equalW.csv`, `output/smm/smm_result_crisis_covid_equalW.jls` |
| Candidate seed bank | `smm/candidates.jl` via `INIT_MODE=:clusters` | same | `output/smm/candidates_{w}_equalW.jls` |
| Standard errors (per window) | `smm/MCMC_main.jl` (DE-MC over the SMM objective) | `julia --project --threads auto smm/MCMC_main.jl` | `output/smm/mcmc_{w}_equalW.jls`, SE table in `output/tables/` |
| Transition (per scenario) | `transition/transition_main.jl` | `julia --project --threads auto transition/transition_main.jl` | `output/transition/transition_{fc,covid}_equalW.jls` |
| Transition panels + fit tables | `transition/transition_main.jl` (`RUN_PANELS`, `RUN_TABLES_AND_PLOTS`) | same | `output/plots/`, `output/tables/` |

The two-stage workflow: estimate the two baseline windows first (they produce
the `.jls` bundles the crisis runs read), then the two crisis windows. Crisis
runs fix the deep parameters (`a_ℓ, b_ℓ, ρ_x, bU, bT, bS`) at the baseline
estimates and re-estimate only the regime-specific block.

`MCMC_main.jl` reads a window's `smm_result_{w}_equalW.jls`, runs a
differential-evolution MCMC chain over the SMM objective around the stored
optimum, and reports the quasi-posterior standard errors. `transition_main.jl`
reads the baseline and crisis bundles for a scenario (`:fc` or `:covid`), solves
the two stationary equilibria, and runs the backward–forward transition between
them.

The transition between base and crisis windows is deliberately **not** an SMM
target — it is held back as out-of-sample validation. All transition and SMM
bundles use one serialization convention (`Serialization` / `.jls`); the model
carries no JLD2 dependency.

## Data provenance

The estimation reads only `data/derived/`. Those files are written by the
user's data pipeline (`data_processing/`, not part of this solver+smm package)
from CPS-ASEC, CPS-basic, JOLTS, J2J, and NSC sources:

```
raw CPS/JOLTS/J2J/NSC  →  data_processing/*.jl  →  data/derived/
    moments_{w}.csv     long format (moment, value); 28 moments per window
    sigma_{w}.csv       28×28 moment covariance (diagonal used for weights)
    windows.json        the four estimation windows (single source of truth)
    nu_estimation.csv    demographic turnover ν (life-table), one row per baseline
    phi_calibration.csv  training completion rate φ (NSC/IPEDS)
    training_share_scale.csv  κ_w level adjustment for training_share
```

`windows.json` is the single source of truth for window definitions, shared with
the data pipeline; `smm_main.jl` validates `WINDOW` against it. `r = 0.05/12`,
`ν`, and `φ` are calibrated externally and held fixed; the wage measurement-error
SDs `σ_wU, σ_wS` are calibrated from the reliability ratio `λ_w = 0.82`.

## Run recipe

```bash
# 1. Restore the pinned environment (the depot + Manifest are the anchor).
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 2. Estimate a window. Set WINDOW at the top of smm/smm_main.jl, then:
julia --project=. --threads auto smm/smm_main.jl
```

Julia 1.10.10. `--threads auto` uses `Base.Threads` inside the solver;
`Random.seed!(1234)` in the entry point fixes reproducibility. `INIT_MODE`
(`:default` / `:warmstart` / `:clusters`) selects how the optimiser is seeded;
`:warmstart` reuses a saved optimum from `output/smm/` when present.

## Parameters

Production is linear in own ability under pure Roy, so there is no ability
gradient to estimate — the single-index `γ_U, γ_S` of the earlier model are
gone, replaced by the correlation `ρ_x`.

**Calibrated (fixed from external data):** `r = 0.05/12`; `ν` from
`nu_estimation.csv`; `φ` from `phi_calibration.csv`; `η_U = η_S = 0.5` (Hosios);
`σ_wU, σ_wS` from `λ_w`.

**Deep structural (estimated on the baseline, held fixed across the cycle):**
`a_ℓ, b_ℓ` (aU-marginal Beta shapes), `ρ_x` (ability correlation), `bU, bT`
(unskilled / training flow values), `bS` (skilled flow value). 6 parameters.

**Regime-specific (re-estimated within each window):** `c` (training cost),
`A` (aggregate scale, log), `PU, PS` (productivity levels), `α_U` (damage
shape), `a_Γ, b_Γ` (skilled offer Beta shapes), and the block matching /
bargaining / hazard parameters `μ, k, λ` (both blocks), `β` (both, freed once
`σ_w` is calibrated), `σ` (OJS flow cost) and `ξ` (exogenous skilled
separation) — 17 estimated. The `REGIME_SPECIFIC_PARAMS` set has 21 entries;
the extra four are `η_U, η_S` (pinned at 0.5 via FIX_PARAMS) and `σ_wU, σ_wS`
(calibrated from `λ_w`), which the classification keeps in the regime block but
FIX_PARAMS / external calibration hold fixed, so they are not searched over.

Bounds, transforms, and starting values live in `smm/smm_params.jl`
(`default_free_params`). Final estimates are written to
`output/tables/smm_estimates_{w}_equalW.csv` and the full `SMMResult` bundle to
`output/smm/smm_result_{w}_equalW.jls`.

## Environment anchor

`Project.toml` / `Manifest.toml` pin `QuasiMonteCarlo` and its dependency
closure (Julia 1.10.10, `project_hash = 6f3295dc54b0bf7c0710d35f253429c1c48921a3`).
The remaining packages the code loads — Distributions, FastGaussQuadrature,
Interpolations, Parameters, CSV, DataFrames, JSON3, Optim, Clustering, and (for
the transition/plot entry points only) Plots, LaTeXStrings, Arrow — resolve from
the ambient Julia environment through the stacked load path. This matches the
development setup: a lean project layer over a shared base depot.

`julia --project=. -e 'using Pkg; Pkg.instantiate()'` restores the pinned layer.
The **solver, SMM, and transition-solving** paths require no plotting backend and
run against this environment directly. Only the plotting/table entry points
(`transition/transition_main.jl`, `solver/plots.jl`) additionally need Plots (GR
backend), LaTeXStrings, and Arrow; install these into the ambient environment
(`Pkg.add(["Plots", "LaTeXStrings", "Arrow"])`) before running them.

# DMP_labour_market_choice

RoySearch — a continuous-time, segmented search-and-matching model of education choice,
estimated by SMM. `CHANGELOG.md` is the live record of what each version changed and
whether its numbers are still comparable; `SETTINGS.md` holds the run-settings lifecycle.
(`VERSION_NOTES.md` is the frozen release note for v19.1.0, kept for the solver
respecifications it documents.)

## Entry points

One named entry point per task; each prints a version banner before it loads anything.

| task | entry point |
|---|---|
| SMM estimation for one window | `code/smm/smm_main.jl` |
| standard errors (DE-MC) | `code/mcmc/MCMC_main.jl` |
| single equilibrium solve + plots | `code/solver/model_main.jl` |
| transition path | `code/transition/transition_main.jl` |
| policy experiments | `code/policy/policy_main.jl` |

Run from the repository root with the pinned environment:

```
julia --project=. -t auto code/smm/smm_main.jl
```

Settings come from `ROYSEARCH_*` environment variables, resolved and logged by
`code/smm/settings.jl`; every key is declared in `SETTINGS_REGISTRY`. The data windows live
only in `data/derived/windows.json`; the four counterfactual windows, which are estimated on
their parent crisis window's data, are declared in `COUNTERFACTUALS` (`code/paths.jl`).

## Starting values and the parameter box

| object | single source | read by |
|---|---|---|
| bounds of every estimable parameter | `default_free_params()`, `code/smm/smm_params.jl` | `build_smm_spec`; the logistic map in `pack_θ` / `unpack_θ` |
| starting values, `INIT_MODE = :default` / `:clusters` | `DEFAULT_PARAMS`, `code/smm/smm_main.jl` | `default_seed` |
| starting values, `INIT_MODE = :warmstart` | the stored point with the lowest Q under the current code among the window's candidates (next section) | §8a of `smm_main.jl`, one line per candidate; a parameter a candidate's run did not estimate is seeded from `DEFAULT_PARAMS` and listed |
| final values | `output/estimates/estimate_<w>_diagonalW_<point>.jls` and its CSV (next section) | every downstream entry point, which names the point it reads |

The parameter table carries no starting values: its entries are built with `init = NaN`,
and `pack_θ` raises, naming the parameter, on any spec that reaches it unseeded. A
parameter that is neither estimated nor pinned has no value either — `unpack_θ` raises
rather than substituting one. A stored θ lives in its own bundle's box, so changing a bound
leaves every stored Q unchanged but invalidates the candidate caches (their key includes
the bounds), and a prior optimum reused as a start is carried over as θ, not as t.

## Estimate files, and the two estimation paths

From v57.0.0 each estimate has a file of its own in `output/estimates/`, named by
`estimate_path(window, W; point)` in `code/paths.jl`, and no run writes into another's
file (`<w>` a window, `W` the weighting suffix: `_diagonalW` under `WEIGHTING = :diagonal`, the
default, `_equalW` under `:equal`; see `SETTINGS.md`):

| file | written by | read by |
|---|---|---|
| `estimate_<w>W_smm.jls`, `smm_estimates_<w>W_smm.csv` | `smm_main.jl` on a base window, or on a crisis window with `CRISIS_BASE = :smm`; the run's checkpoints go to the same file | a crisis window's run with `CRISIS_BASE = :smm`; `MCMC_main.jl` on every route but a crisis chain; `POINT_ESTIMATE = :smm`; the warm start |
| `estimate_<w>W_smm_mcmcbase.jls`, `smm_estimates_<w>W_smm_mcmcbase.csv` | `smm_main.jl` on a crisis window with `CRISIS_BASE = :mcmc` | the crisis chain, as its seed; the warm start |
| `estimate_<w>W_mcmc.jls`, `mcmc_results_<w>W_mcmc.csv` | a chain run of `MCMC_main.jl`: the posterior mean of the retained draws, stage `:mcmc`, or `:mcmc_aborted` when the chain aborted | a crisis window's run with `CRISIS_BASE = :mcmc`; the crisis chain's ability check; `POINT_ESTIMATE = :mcmc`. Every reader refuses `:mcmc_aborted` |
| `mcmc_results_<w>W_smm.csv` | the chain-free pass of `MCMC_main.jl` (`MCMC_JAC_ONLY = true`), which also adds its standard errors to `estimate_<w>W_smm.jls` and leaves its point, stage and `sim` unchanged | tables |
| `estimate_<w>W_mcmcbest.jls` | a chain run with `MCMC_CHECKPOINT = true` only: the best point the chain visited, stage `:mcmc_checkpoint` | nothing; a record of where the chain went |
| `candidates_<w>W_smm.jls`, `candidates_<w>W_smm_mcmcbase.jls` | `INIT_MODE = :clusters`, tagged like the run's bundle | the next `:clusters` run of the same configuration |
| `estimate_<w>W.jls` (untagged) | versions before v57.0.0 | the warm start only |

A reader of an estimate accepts a file written by another MAJOR version only if its stored
point still scores its stored Q under this code (relative 1e-10, `require_current_estimate`):
it re-evaluates the point, prints one line saying so, and rewrites nothing. Otherwise it stops
in one line with both values of Q. The warm start reads any such file, and re-scores it. The other files versions before v57.0.0 left in `output/estimates/`
(`estimate_<w>W_postmean.jls`, the `_backup_Q` copies and the untagged candidate banks) are
read by nothing.

**The SMM path.**

1. `smm_main.jl` on each base window (`ROYSEARCH_WINDOW=base_fc`, then `base_covid`): writes
   `estimate_<base>W_smm.jls` and its CSV.
2. `smm_main.jl` on each crisis window, `CRISIS_BASE = :smm` (the default): reads
   `estimate_<base>W_smm.jls` and holds the ability distribution (a_ℓ, b_ℓ, ρ_x) at it;
   writes `estimate_<crisis>W_smm.jls` and its CSV.
3. `MCMC_main.jl` with `ROYSEARCH_MCMC_JAC_ONLY=true` on any window: reads
   `estimate_<w>W_smm.jls`; writes `mcmc_results_<w>W_smm.csv` and adds the standard errors
   to the bundle it read.
4. `transition_main.jl` with `POINT_ESTIMATE = :smm`: reads both windows' `_smm` files; writes
   `output/transition/transition_<pair>W_smm.jls` and tables and plots ending in `W_smm`.

**The MCMC path**, after step 1 of the SMM path.

1. `MCMC_main.jl` (chain) on each base window: seeds from `estimate_<base>W_smm.jls`; writes
   `estimate_<base>W_mcmc.jls`, `mcmc_results_<base>W_mcmc.csv` and
   `output/chains/chain_<base>W.jls`.
2. `smm_main.jl` on each crisis window with `ROYSEARCH_CRISIS_BASE=mcmc`: reads
   `estimate_<base>W_mcmc.jls` and holds the ability distribution at it; writes
   `estimate_<crisis>W_smm_mcmcbase.jls` and its CSV.
3. `MCMC_main.jl` (chain) on each crisis window: seeds from
   `estimate_<crisis>W_smm_mcmcbase.jls`, and stops unless its ability distribution equals
   that of `estimate_<base>W_mcmc.jls` bit for bit; writes `estimate_<crisis>W_mcmc.jls`,
   its CSV and the chain file.
4. `transition_main.jl` with `POINT_ESTIMATE = :mcmc`: reads both windows' `_mcmc` files;
   writes `output/transition/transition_<pair>W_mcmc.jls` and tables and plots ending in
   `W_mcmc`, beside the SMM path's.

`CRISIS_BASE` (`smm_main.jl`, directly below `WINDOW`; `ROYSEARCH_CRISIS_BASE`) is read only
on a crisis window, so it can stay set while the window changes: a base window writes the
same file with the same Q under either value. `MCMC_main.jl` does not read it.

`POINT_ESTIMATE` is a constant at the top of each script that reads an estimate
(`transition_main.jl`, `model_main.jl`, `policy_main.jl` and the tools in `code/tools/`):
`:smm` reads `estimate_<w>W_smm.jls`, `:mcmc` reads `estimate_<w>W_mcmc.jls`. A transition
reads both windows of a pair at the same point, never one of each.

The warm start (`INIT_MODE = :warmstart`, §8a of `smm_main.jl`) scores every candidate
under the current code, prints one line per candidate with its Q, and starts from the
lowest; with none readable it starts from `DEFAULT_PARAMS`. A base window's candidates are
its `_smm` file and the untagged file; a crisis window's are its own output file, its other
SMM file (`_smm` or `_smm_mcmcbase`), the untagged crisis file and the base point. A crisis
window's ability distribution is the base point's whichever candidate wins.

## Counterfactual windows

Each crisis window is estimated twice more under a restriction. The four windows are
declared once, in `COUNTERFACTUALS` (`code/paths.jl`), each with its parent crisis window:

| window | parent | restriction |
|---|---|---|
| `crisis_fc_cf1`, `crisis_covid_cf1` | `crisis_fc`, `crisis_covid` | the training choice τ and the choice of market d held, at every type and every solve, at their values in the base stationary equilibrium; the training cost c held at the base estimate |
| `crisis_fc_cf2`, `crisis_covid_cf2` | `crisis_fc`, `crisis_covid` | the training cost c held at the base estimate; τ and d chosen |

Common to all four: the data are the parent's (`data_window`: `moments_<parent>.csv`,
`sampling_var_<parent>.csv`, the parent's `windows.json` entry, ν, φ, r and σ_w); the base
estimate is the parent's, at the point `CRISIS_BASE` selects, and ρ_x and c are held there
(`base_held_params`, `code/smm/smm_params.jl`), so 12 parameters are free; `training_share`
and `skilled_share` are held out of the objective (`WINDOW_SKIP_MOMENTS`, `smm_main.jl`), 33
moments are targeted, and both stocks are still computed and listed with the skipped
moments. The spec records the base estimate under `spec.fixed.paired_base` (`PAIRED_BASE`):
its window, point, file (from the project root) and SHA-256, and the decoded point. A cf1
evaluation reads its held choices from that record (`held_choices`, `code/smm/smm.jl`): the
base point is solved once per grid under the current code and its τ, d and the node choices
behind them (`HeldChoices`, `code/solver/params.jl`) are carried by the model (`Model.held`),
so every reader of a cf1 bundle, in every entry point, solves the restricted model. At the
base estimate on the base data the cf1 model reproduces the base equilibrium, every moment
and Q bit for bit.

Every file is named by the window (`estimate_crisis_fc_cf1_diagonalW_smm_mcmcbase.jls`) and
is written exactly as a crisis window's: `smm_main.jl` writes `_smm` (`CRISIS_BASE = :smm`) or
`_smm_mcmcbase` (`:mcmc`); `MCMC_main.jl` seeds a chain from `_smm_mcmcbase` and checks ρ_x
and c against the base chain's estimate. The warm start reads the window's own output file
if there is one, else the parent's SMM estimate at the same point, on the 12 free parameters.

## Verification

| check | script | what fails the gate |
|---|---|---|
| settings reach their consumers | `code/gates/check_forwarding.jl` | a computed setting not forwarded, an unclassified key, a caller/callee name mismatch, an undeclared rename |
| clone completeness | `code/gates/check_repo.sh` | a missing estimation input |
| dead settings | `code/gates/check_dead_settings.jl` | a keyword argument declared and never read |
| threaded evaluation | `code/gates/check_threading.jl` | a solve inside `Threads.@threads` returning a non-finite objective, or one that is not bitwise equal to the serial value |
| unskilled hiring margin | `code/gates/check_hire_margin.jl` | a node, farther than 1e-6 relative from a^h = b̃_U/P_U, where the solved p*_U < 1 and the closed-form hireable set a_U > a^h disagree |
| crisis → baseline pairs | `code/gates/check_pair_map.jl` | `smm_main.jl`'s own `_PAIR_BASELINE` and `PAIR_BASE` in `code/paths.jl`, which every other script reads, naming different pairs |
| output-tree layout | `code/gates/check_output_tree.jl` | a `.csv` in `tables/`, an artifact under `code/`, a `savefig` outside `plots_and_tables/`, a hardcoded output path, an `smm → mcmc` include |
| bundle compatibility | `code/tools/migrate_bundles.jl` | a rebuilt bundle not reproducing the reference Q recorded at extract time |
| transition path runs | `code/smoke/smoke_transition.jl` | a terminal steady state that does not solve, or a non-finite series on the path |
| transition solves a *transitory* path | `code/tools/transition_audit.jl` | the path drifts off the steady state it started from under a null shock (`TA_NULL=true`), sits on the segment between its endpoints, or misses the post-switch equilibrium at the terminal date |

A bundle written before v40.0.0 will not load: `SimParams` lost a field in each of v39.3.0
(`anderson_m`) and v40.0.0 (`tol_global`), and Julia's serialiser reads structs
positionally. Carry it across with `migrate_bundles.jl extract` on
the code that wrote it, then `rebuild` on this code — `provenance.migrated_from` then names
the version that produced the estimate, while `provenance.version` names the code that
last wrote the file.

## What "converged" means

From v40.0.0 the global loop converges on the equilibrium conditions themselves, not on the
step between iterates. `global_residuals` in `code/solver/solver.jl` returns four
components, each as a ratio to its own internal tolerance, and the gate is `Rmax < 1` held
for `conv_streak` consecutive passes:

| component | condition | tolerance |
|---|---|---|
| `R1` | the skilled cutoffs: the `p*_S` and `p^oj` the values re-solved at the installed state imply, against the installed ones, in units of p | `TOL_CUTOFF` = 1e-6 |
| `R2` | free entry in both markets, relative in θ (unskilled only while θ_S is imposed) | `TOL_RELATIVE` = 1e-6 |
| `R3` | the training choice τ, recomputed by `training_policy!` | `TOL_POLICY` = 1e-6 |
| `R5` | the education margin, recomputed by `training_frontier!`; the only component coupling the two markets | `TOL_RELATIVE` = 1e-6 |

The tolerances are internal constants, not settings: they are properties of the
conditions.

The unskilled block and the global loop are Anderson-accelerated (memory 1, `use_anderson`).
The skilled block is not: from v45.0.0 it is a plain damped fixed-point iteration, with cutoffs
under-relaxed by `damp_pstar_S` and θ_S by the adaptive free-entry weight under
`DAMP_THETA_S`. From v45.1.0 its step test counts Δp^oj_S only for the types that search on
the job before or after the pass, the set `R1` checks: every other type's p^oj_S carries zero
weight in every object. `SETTINGS.md` records why each loop is iterated the way it is. There is no `R4` — the population adding-up residual would go
there and is vacuous, since (E5) defines `m_U := ℓ − m_S`. The `v40.0.0` changelog entry
carries the measured separation between a converged state and a state perturbed off each
condition in turn; the `v42.0.0` entry the same for the cutoff form of `R1`.

Three margins on the ability grid are covered fractions of a node's midpoint interval rather
than node indicators. They are the training frontier τ_T, the cross-market drain d and, from
v46.0.0, the unskilled hiring margin χ (`hire_fraction`, `code/solver/unskilled.jl`). A new
unskilled match starts at p = 1, so a type is hired iff a_U > a^h = b̃_U/P_U, whatever the
tightness. χ is the share of each cell above a^h, and it is not solved for: `check_hire_margin`
confirms that the solved p*_U selects the same set. Never-hired types are unemployed and stay
in the seeker pool. From v47.0.0 a vacancy's meeting value counts only the hired branch of
each row's unemployed (`update_theta_unskilled`), because a meeting with a never-hired type
yields no match, while the seeker pool it divides by still counts everyone. From v48.0.0 every
entry object is read at p = 1 itself, which is not a quadrature node: `entry_surplus_unskilled`
gives the closed-form S_U(a_U,1), so J_U(a_U,1) rises continuously from zero at a^h, and the
hiring atom `eU_entry` is paid the entry wage w_U(a_U,1).

`transition_audit.jl` is the gate the smoke test cannot be: finiteness does not distinguish
a transition from a sequence of steady states. `TA_NULL=true` runs a window against itself,
where the true path is the constant one, so any drift measures the gap between the forward
laws of motion and the stationary KFE with no economics mixed in. Run it after any change
to either. Per-date series land in `output/logs/transition_audit_{pair}[_null].csv`.

## Transition path

`code/transition/transition_main.jl` solves, for each scenario in `SCENARIOS`, the perfect-
foresight path from the baseline window's equilibrium to the crisis window's after an
unanticipated, lasting change of every estimated parameter, the vacancy costs included, at
date 0, then draws the panels
and tables. Its settings are the constants in its section 1:

| setting | meaning |
|---|---|
| `SCENARIOS` | any of the transition pairs `PAIRS` (`code/paths.jl`), three per event: `:fc`, `:covid` run to the crisis estimate, `:fc_cf1`, `:covid_cf1` to the cf1 estimate, `:fc_cf2`, `:covid_cf2` to the cf2 estimate; default `[:fc, :covid]` |
| `POINT_ESTIMATE` | `:smm` reads both windows of each pair at their SMM estimates, `estimate_<w>W_smm.jls`; `:mcmc` at their chains' estimates, `estimate_<w>W_mcmc.jls` (the posterior mean of the retained draws), and stops for a window whose chain aborted or never ran |
| `TR_T_MAX`, `TR_N_STEPS` | the fine time grid (120 months in half-month steps) |
| `TR_TOL`, `TR_MAXIT`, `TR_DAMP` | the outer loop on the tightness paths: tolerance on max over dates of \|ln θ^new − ln θ\|, iteration cap, damping |

The grids are each bundle's own estimation grid (`spec.run`); the run stops if the two windows
differ in the grid or in the ability distribution and demographic rates, which a crisis window
holds at its paired baseline (and a counterfactual window the training cost too), or if either file is not an estimate under this code (written by
another MAJOR version and its Q does not reproduce). The tables
phase reads the transition files of the `POINT_ESTIMATE` it runs under, stops on one that records
another point, and reads the two estimate files each records, so it reports exactly the points
the paths were solved at. Each endpoint is the solve at its window's tightnesses
(`equilibrium_at`), under that window's estimation settings (`spec.sim`), and its vacancy costs
are computed there; along the path the costs switch at date 0 and the tightness paths θ_U(t),
θ_S(t) are solved from free entry at every date. The horizon is not a setting: it doubles from
120 months, with a geometric tail beyond the fine grid, until no reported series moves by more
than 1e-3 of its scale over the first 36 months, and neither mean log wage by more than 1e-3 in
log units (`solve_with_horizon`, `transition_simulation.jl`); every trial is recorded. A cf1
scenario runs under the cf1 model on the whole path: its endpoint model carries the held
choices, so at every date τ and d are the held arrays and every envelope over training or over
the market is the held branch (`transition_values.jl`, `_d_matrix` in `transition_solver.jl`).

The output is `output/transition/transition_{scenario}{W}_{point}.jls`, one NamedTuple (`schema =
2`), the same field set for every scenario: version and provenance; scenario, point, bundle files and the point of each window; both
windows' parameters; settings, horizon and the horizon trials; the crisis window (its length in
months from `windows.json` and a per-date marker); `result` (the `TransitionResult` the panels
plot; its mean wages are mean log wages, the units of the data moments they are plotted against);
`baseline`, the same series at the pre-switch steady state, which the panels draw before the
switch; `path`, every value function, policy and mass by date (`TransitionPath`, by field name);
`statistics`, the flows and the rate moments at every date; `reversals`, how often the crossing,
training and on-the-job-search choices change direction; and both endpoint models with their
equilibrium objects. Every table and plot of the run ends in the same `{W}_{point}`
(`model_fit_A_diagonalW_smm.tex`, `transition_panel_covid_diagonalW_mcmc.png`), so the runs at the
two points sit side by side; `transition_file` and `read_transition` (`transition_params.jl`) name
and read the file. The data standard error printed under each data moment of the model-fit tables
(`model_fit_A`, `model_fit_B`, `make_model_fit_tables` in `plots_and_tables/transition.jl`) and
drawn as the panels' bands is the moment's sampling SD σ̂_k, the square root of its entry in
`data/derived/sampling_var_{window}.csv` (`load_sampling_variances`, `code/smm/moments.jl`), the
same σ̂_k the diagonal weighting divides by.

## The tightnesses as parameters

From v56.0.0 the optimiser's coordinates for the two markets are the tightnesses θ_U and
θ_S (`(:unsk, :tight)`, `(:skl, :tight)` in `default_free_params()`), and the vacancy costs
k_U and k_S are computed from free entry in every evaluation. The model and the estimator
are unchanged: k_U and k_S are estimated parameters and every table lists them as such.
Given the tightnesses the stationary state is unique and needs no iteration between the
markets (`solve_at_tightness!`, `code/solver/solver.jl`): the unskilled block at θ_U, the
skilled block at θ_S, then the training choice and the masses, then
k_j = q_j(θ_j) J̄_j / π̄_j from the free-entry objects the fixed-cost solve uses. Going the
other way, from the costs to the tightnesses, is a correspondence: unskilled free entry has
three zeros at both COVID estimates. So minimising Q over the parameter vector with the
tightnesses in it is the best-fitting-equilibrium estimator exactly, and Q is one
equilibrium's at every point. `equilibrium_at` (`code/smm/smm.jl`) is that evaluation;
`smm_objective` returns its Q, and every consumer that needs the equilibrium at an estimate
takes it from there.

The costs at an estimate are stored in the bundle's `vacancy_costs` field and written as
rows of the estimates CSV and the MCMC table. `attracting_check` measures
D_ij = ∂ln k_i/∂ln θ_j at the estimate, by central differences of 0.01 in ln θ_j (four solves
at the end of SMM; read off the Jacobian's tightness columns in `MCMC_main.jl`); the
equilibrium attracts market by market when D_UU < 0 and D_SS < 0, and jointly when
tr D < 0 and det D > 0. Both verdicts go in the bundle's `attracting` field, and a run
prints one line only when one fails.

At given vacancy costs, which the counterfactuals hold, the global loop solves for the
tightnesses (`solve_model!`), and `skilled_equilibria` with `select_equilibrium(model, spec)`
finds and scores the attracting skilled roots there: given θ_S every other object is unique,
so an equilibrium is a root of one function of θ_S, with one or three roots, L and H
attracting and the middle one repelling. The section header in `solver.jl` states the
properties of that map the detector relies on.

## Standard errors

`code/mcmc/MCMC_main.jl` runs after an estimation and takes its point from the window's SMM
estimate, `output/estimates/estimate_{window}{W}_smm.jls`, except a chain on a crisis window,
which starts from `estimate_{window}{W}_smm_mcmcbase.jls` (the MCMC path above). It never
moves an SMM estimate. Every standard error is a sandwich, Var ≈ B Ĝ′WΩWĜ B, with Ĝ = ∂g/∂θ
by adaptive finite differences (`MCMC_JAC_METHOD = :fd`; `:design` regresses it on a cloud of
points) and B the inverse curvature of Q/2, the bread. Two routes, one flag:

| route | how to run it | bread B, and where Ĝ is measured | cost |
|---|---|---|---|
| chain-free | `ROYSEARCH_MCMC_JAC_ONLY=true` | J⁻¹ = (Ĝ′WĜ)⁻¹, Ĝ at θ̂ | under `:fd`, 168 solves at base_covid's 14 free parameters (12 each), more where a step is re-tuned |
| DE-MC chain (**the default**) | `ROYSEARCH_MCMC_JAC_ONLY=false` | P, the covariance of the kept draws in θ, Ĝ at the chain estimate θ̄; plus the posterior SD | N·gens solves |

J⁻¹ is the Gauss–Newton curvature: it drops the residual × second-derivative term of Q's
curvature, which is not small when Q is far above its degrees of freedom. The chain samples
exp(−Q/2) itself, so P integrates the true curvature over the cloud; an aborted chain falls
back to the chain-free columns. Two columns come from each bread: the sandwich at Ω = W⁻¹
(uncorrelated moment errors), **`se_sandwich`** = `sqrt(diag(P Ĝ′WĜ P))` on the chain route
and `se_curvature` = `sqrt(diag(J⁻¹))` on the chain-free route; and **`se_bound`** =
Σᵢ|(B Ĝ′W)ₖᵢ| σ̂ᵢ, the sharp bound over every Ω consistent with the estimated sampling
variances, so it holds for the true Ω whatever it is (`se_bound_diagonal`,
`code/mcmc/mcmc_diagnostics.jl`). Every |t| is |estimate|/se at the reported estimate.

The vacancy costs are not coordinates of the parameter vector, so their standard errors
are the delta method's, from the same evaluations: every Jacobian step also returns k, and
the gradient of k in the parameter vector is the combination both errors are evaluated at.
The SMM covariance does not depend on the coordinates it is expressed in, so these are the
errors a k-coordinate estimation would report. A chain records k for every draw. Its
prior is `MCMC_PRIOR`'s over the parameter vector, so flat in θ_U and θ_S.

| object | file | notes |
|---|---|---|
| per-parameter estimates and standard errors | `output/estimates/mcmc_results_{window}{W}_smm.csv` (chain-free), `mcmc_results_{window}{W}_mcmc.csv` (chain) | 20 columns; `post_mean`, `q025/q500/q975`, `rhat`, `ess`, `edge_frac`, `spread_growth` and `se_chain` are NaN on the chain-free route, which has no draws; the rows `k` of each block follow the free parameters, with the box-dependent columns NaN; columns 6 and 18 are `se_sandwich` and `abs_t_sandwich` on the chain route and `se_curvature` and `abs_t_curvature` without a usable chain; `abs_t_*` are the console's \|t\|(sandw.) or \|t\|(J⁻¹) and \|t\|(bnd) |
| chain, per-draw moments, Ĝ and its per-moment R² (`G_R2`) | `output/chains/chain_{window}{W}.jls` | chain route only; the chain-free route writes nothing here, so it can never overwrite a real chain with a stub |
| the chain's estimate: the posterior mean θ̄, with every standard error, the diagnostics and the seed's file name (`mcmc.seed`) | `output/estimates/estimate_{window}{W}_mcmc.jls` | chain route only, written when Q(θ̄) is finite; stage `:mcmc`, or `:mcmc_aborted`, which every reader refuses; read under `POINT_ESTIMATE = :mcmc` and by a crisis window under `CRISIS_BASE = :mcmc` |
| standard errors of the SMM estimate | `output/estimates/estimate_{window}{W}_smm.jls` | chain-free route only: `se`, `se_source` and the vacancy costs' errors are added; point, stage and `sim` are left as `smm_main.jl` wrote them |
| the best point the chain visited | `output/chains/chain_{window}{W}.jls` (`theta_best`), and `estimate_{window}{W}_mcmcbest.jls` with `MCMC_CHECKPOINT = true` | a record, never an estimate. The chain file is snapshotted every `MCMC_LMR_PRINT` generations (500) whatever `MCMC_CHECKPOINT` says, so a run killed before it returns leaves its draws |

A coordinate sitting on a box bound has no two-sided interval however tight its
curvature, and whether such a corner is an acceptable economic zero is not a decision
the code takes. Declare those parameters by hand in `CORNER_PARAMS`
(`code/mcmc/MCMC_main.jl`, empty by default, `ROYSEARCH_CORNER_PARAMS="b_S,alpha_U"`);
they are flagged in the CSV's `corner_declared` column and nothing else about the file
changes.

## Continuity of the criterion

Every margin that moves with `θ` across a fixed quadrature grid is a covered fraction of a
grid cell rather than a 0/1 indicator, so the aggregates it feeds — and therefore `Q` — are
piecewise-linear in the crossing and differentiable in the parameters.

Two are on the ability grid: the training frontier `τ_T` (`solver/unskilled.jl:196-254`) and
the cross-market drain `d` (`solver/drain_fraction!`, `solver/skilled.jl:132-204`). Both
express the fraction against the node intervals from `node_midpoint_intervals`
(`solver/grids.jl:84-101`), and each requires its value function to be monotone in the
ability it is inverted against, checked at runtime.

One is on the match-quality grid: the on-the-job-search cutoff `p^oj`, whose covered
fraction `_soft_oj_weight` (`solver/grids.jl:322-331`) is read by the stationary solve
(`solver/skilled.jl:373`), by the wage split and the E-to-E flow in the moment layer
(`solver/equilibrium.jl:197-219`, `:330-347`), by the transition's poaching outflow
(`transition/transition_solver.jl:448`) and by the two wage consumers in `policy/` and
`plots_and_tables/`. One hard read of `p^oj` survives — `Γ` evaluated at the cutoff's grid
node (`solver/equilibrium.jl:314`) — and it is what remains of the jump below.

From v57.0.0 the top node's cell is [p_N, 1] in both covered-fraction rules (`_soft_weight`,
`_soft_oj_weight`, `solver/grids.jl`), so a cutoff crossing p_N fades that node out like any
other. The wage moments are read off the wage points of `compute_equilibrium_objects`
(`wage_points`: one (wage, mass) pair per employed cell, so each employed worker counts once;
the unskilled tail by density × quadrature weight, the new matches at p = 1 by their mass at
the entry wage), with the percentiles and overlaps evaluated exactly on the σ_w-convolved
distribution (`smm/moments.jl`). No histogram lies in between; the 120-bin level histograms
serve figures only.

| object | file | what it establishes |
|---|---|---|
| drain step-ladder, before and after the `d` fix | `output/logs/drain_continuity_step_ladder_base_fc.csv` | `dQ/h` on `μ_S`, `λ_S`, `β_S`, `k_S` at six steps from 3e-8 to 1e-5; the surviving ΔQ = 0.359 jump on `λ_S` at h = 1e-6 and `k_S` at h = 3e-6 is unchanged by the fix |
| fixed-step Jacobian viability | `output/logs/drain_continuity_jacobian_stability_base_fc.csv` | per-column step spread at a fixed h = 1e-6; 5 of 23 columns fail a 5% stability test before and after, the same five, so the Jacobian still needs a per-column step |
| is the ΔQ jump solve precision or a threshold? | `output/logs/drain_jump_tolerance_invariance_base_fc.csv` | the jump measured at `tol_global` = 1e-4, 1e-6, 1e-8 with all other settings held fixed; it moves by < 1e-5 relative across four orders of magnitude, against the factor 1e-4 a precision artefact would give, so it is a property of the criterion. Produced under the pre-v40.0.0 global tolerance, which no longer exists; a fresh run sweeps the block tolerances instead (`PROBE_TOL`) and is not comparable rung-for-rung |

Softening the moment layer's two `p^oj` reads in v27.0.0 cut that jump by 15.9× and the
per-moment steps behind it by 31–2070×. Everything below is produced by
`code/tools/ojs_continuity_probe.jl` (`PROBE_MODE` ∈ `moments`, `jump`, `crossing`,
`jacobian`), run once against a pre-v27 copy of the tree and once against this one.

| object | file | what it establishes |
|---|---|---|
| backward compatibility where no cell straddles | `output/logs/ojs_moments_base_fc_{before,after}.csv` | with `p^oj` snapped to p-grid nodes, `Q` = 963.90590480318895 and all 31 moments are bit-identical across the change — the covered fraction equals the indicator it replaced at `s ∈ {0,1}` |
| ΔQ jump by solve tolerance, before and after | `output/logs/ojs_jump_tolerance_base_fc_{before,after}.csv` | the jump across a `p^oj` node crossing at four tolerances: +0.3600 → −0.0226, invariant in both, so what remains is a threshold and not solve precision |
| what carries the residual step | `output/logs/ojs_crossing_base_fc_{before,after}.csv` | per-moment steps across the crossing; `ee_rate_S` 1.21e-2 → 3.91e-4 and the wage moments ~2000× smaller, with the residual traced to `Γ(p^oj)` stepping −10.15% as row 63's node index moves 8 → 7 |
| fixed-step Jacobian viability, two tolerances | `output/logs/ojs_jacobian_tol1e-0{4,8}_base_fc_{before,after}.csv` | on the 16 stable columns the step spread falls from a median 1.06e-3 at `tol_global` = 1e-4 to 7.13e-8 at 1e-8 — a ratio of 1.0e-4, exactly the tolerance ratio — so a solve-precision noise floor sits underneath the threshold as a separate effect; `σ_max` 4.27e+04 → 4.47e+03 and `cond(J)` 3.53e+17 → 7.22e+16. The `tol1e-04`/`tol1e-08` filenames name the pre-v40.0.0 global tolerance; a fresh run tags the block tolerance it swept (`PROBE_TOL`, default `bundle`) |

`Q` at the shipped `base_fc` estimate is 963.905904803 before and 969.804678028 after, so
the stored optima are points on the old surface: see the comparability note under v27.0.0
in `CHANGELOG.md`.

## The annealing proposal, and the measurements behind it

Each iteration draws an independent Bernoulli(`p_move`) mask over coordinates plus one
forced index — the same construction DE uses for crossover, so the number of coordinates
moving is random and unbounded above with a controlled mean — and moves each selected
coordinate by its own step, seeded from that coordinate's measured ΔQ = 1 half-width and
adapted by Corana from there. No fixed coordinate count and no shared scalar step exist
in the code.

`sa_proposal_scale` (`code/smm/smm.jl`) measures the step scale at the start point;
`[SA config]`, printed from inside the walk, is the authoritative record of what the
optimiser received, and the trace reports the **realised** moved-count distribution
rather than the nominal `p_move`.

Measured at the stored `base_fc` optimum under the v19.0.0 solver, 23 free parameters:

| object | file | what it establishes |
|---|---|---|
| per-coordinate ΔQ = 1 half-widths | `output/logs/sa_proposal_widths_base_fc.csv` | widths span 0.0022–0.20 (91x, median 0.021); 10 of 23 coordinates fall below the 0.01 the retired scalar step clamped at |
| proposal comparison, 3 seeds at equal budget | `output/logs/sa_proposal_arm_comparison_base_fc.csv` | mean ΔQ 81.2 at the retired fixed k=3, 87.8 for a dense mask, 90.3 for the shipped mask, 91.5 for a single-coordinate move — reported with acceptance **and** distance moved |
| joint vs component feasibility | `output/logs/sa_proposal_joint_feasibility_base_fc.csv` | 0 of 90 draws had a feasible joint move with all components infeasible — the retired justification for multi-coordinate moves does not replicate here |
| matched-displacement sparsity scan | `output/logs/sa_proposal_sparsity_scan_base_fc.csv` | feasibility saturated (0.958–1.000 across k), so it cannot identify the count; this scan favoured k ≈ 11, which the chains above overruled |

Two methodological points, both load-bearing and both recorded in `SETTINGS.md`'s
graveyard entry for `sa_subset_k`. Acceptance and expected squared jump distance are
both functions of `k·σ²` to second order, so neither identifies the coordinate count
separately from the step scale — which is why `p_move` is fixed rather than adapted
against acceptance, and why every comparison here reports descent alongside acceptance.
And a sparsity ranking taken from one-shot draws disagreed with the same ranking taken
from chains: a large joint move is the better single bet and the worse thing to repeat.
Sparsity questions get measured on chains.
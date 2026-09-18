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
`code/smm/settings.jl`; every key is declared in `SETTINGS_REGISTRY`. The window list
lives only in `data/derived/windows.json`.

## Verification

| check | script | what fails the gate |
|---|---|---|
| settings reach their consumers | `code/gates/check_forwarding.jl` | a computed setting not forwarded, an unclassified key, a caller/callee name mismatch, an undeclared rename |
| clone completeness | `code/gates/check_repo.sh` | a missing estimation input |
| dead settings | `code/gates/check_dead_settings.jl` | a keyword argument declared and never read |
| output-tree layout | `code/gates/check_output_tree.jl` | a `.csv` in `tables/`, an artifact under `code/`, a `savefig` outside `plots_and_tables/`, a hardcoded output path, an `smm → mcmc` include |
| bundle compatibility | `code/tools/migrate_bundles.jl` | a rebuilt bundle not reproducing its stored Q |
| transition path runs | `code/smoke/smoke_transition.jl` | a terminal steady state that does not solve, or a non-finite series on the path |
| transition solves a *transitory* path | `code/tools/transition_audit.jl` | the path drifts off the steady state it started from under a null shock (`TA_NULL=true`), sits on the segment between its endpoints, or misses the post-switch equilibrium at the terminal date |

`transition_audit.jl` is the gate the smoke test cannot be: finiteness does not distinguish
a transition from a sequence of steady states. `TA_NULL=true` runs a window against itself,
where the true path is the constant one, so any drift measures the gap between the forward
laws of motion and the stationary KFE with no economics mixed in. Run it after any change
to either. Per-date series land in `output/logs/transition_audit_{pair}[_null].csv`.

## Standard errors

`code/mcmc/MCMC_main.jl` runs after an estimation and takes its point from
`output/estimates/estimate_{window}{W}.jls`. Two routes to Ĵ, one flag:

| route | how to run it | Ĵ from | cost |
|---|---|---|---|
| chain-free (**the reported one**) | `ROYSEARCH_MCMC_JAC_ONLY=true` | Ĝ′WĜ, Ĝ regressed on a `local_design` around θ̂ | ≈ 600 solves |
| DE-MC chain | `ROYSEARCH_MCMC_JAC_ONLY=false` | Cov(chain), plus the posterior SD | N·gens solves |

Both follow Chernozhukov–Hong (2003) Thm 4, which needs no information equality and no
chain. The reported column is **`se_bound`**: the sharp bound over every Ω consistent
with the estimated sampling variances, so it holds for the true Ω whatever it is.
`se_curvature` is `sqrt(diag(pinv(Ĵ)))`, exact only under W = Ω⁻¹.

| object | file | notes |
|---|---|---|
| per-parameter estimates and standard errors | `output/estimates/mcmc_results_{window}{W}.csv` | 17 columns; `post_mean`, `q025/q500/q975`, `rhat`, `ess`, `edge_frac`, `spread_growth` and `se_chain` are NaN on the chain-free route, which has no draws |
| chain, per-draw moments, Ĝ and its per-moment R² (`G_R2`) | `output/chains/chain_{window}{W}.jls` | chain route only; the chain-free route writes nothing here, so it can never overwrite a real chain with a stub |

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
(`transition/transition_solver.jl:351`) and by the two wage consumers in `policy/` and
`plots_and_tables/`. One hard read of `p^oj` survives — `Γ` evaluated at the cutoff's grid
node (`solver/equilibrium.jl:314`) — and it is what remains of the jump below.

| object | file | what it establishes |
|---|---|---|
| drain step-ladder, before and after the `d` fix | `output/logs/drain_continuity_step_ladder_base_fc.csv` | `dQ/h` on `μ_S`, `λ_S`, `β_S`, `k_S` at six steps from 3e-8 to 1e-5; the surviving ΔQ = 0.359 jump on `λ_S` at h = 1e-6 and `k_S` at h = 3e-6 is unchanged by the fix |
| fixed-step Jacobian viability | `output/logs/drain_continuity_jacobian_stability_base_fc.csv` | per-column step spread at a fixed h = 1e-6; 5 of 23 columns fail a 5% stability test before and after, the same five, so the Jacobian still needs a per-column step |
| is the ΔQ jump solve precision or a threshold? | `output/logs/drain_jump_tolerance_invariance_base_fc.csv` | the jump measured at `tol_global` = 1e-4, 1e-6, 1e-8 with all other settings held fixed; it moves by < 1e-5 relative across four orders of magnitude, against the factor 1e-4 a precision artefact would give, so it is a property of the criterion |

Softening the moment layer's two `p^oj` reads in v27.0.0 cut that jump by 15.9× and the
per-moment steps behind it by 31–2070×. Everything below is produced by
`code/tools/ojs_continuity_probe.jl` (`PROBE_MODE` ∈ `moments`, `jump`, `crossing`,
`jacobian`), run once against a pre-v27 copy of the tree and once against this one.

| object | file | what it establishes |
|---|---|---|
| backward compatibility where no cell straddles | `output/logs/ojs_moments_base_fc_{before,after}.csv` | with `p^oj` snapped to p-grid nodes, `Q` = 963.90590480318895 and all 31 moments are bit-identical across the change — the covered fraction equals the indicator it replaced at `s ∈ {0,1}` |
| ΔQ jump by solve tolerance, before and after | `output/logs/ojs_jump_tolerance_base_fc_{before,after}.csv` | the jump across a `p^oj` node crossing at four tolerances: +0.3600 → −0.0226, invariant in both, so what remains is a threshold and not solve precision |
| what carries the residual step | `output/logs/ojs_crossing_base_fc_{before,after}.csv` | per-moment steps across the crossing; `ee_rate_S` 1.21e-2 → 3.91e-4 and the wage moments ~2000× smaller, with the residual traced to `Γ(p^oj)` stepping −10.15% as row 63's node index moves 8 → 7 |
| fixed-step Jacobian viability, two tolerances | `output/logs/ojs_jacobian_tol1e-0{4,8}_base_fc_{before,after}.csv` | on the 16 stable columns the step spread falls from a median 1.06e-3 at `tol_global` = 1e-4 to 7.13e-8 at 1e-8 — a ratio of 1.0e-4, exactly the tolerance ratio — so a solve-precision noise floor sits underneath the threshold as a separate effect; `σ_max` 4.27e+04 → 4.47e+03 and `cond(J)` 3.53e+17 → 7.22e+16 |

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
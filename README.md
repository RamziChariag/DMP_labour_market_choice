# DMP_labour_market_choice

RoySearch — a continuous-time, segmented search-and-matching model of education choice,
estimated by SMM. See `SETTINGS.md` for the run-settings lifecycle and `VERSION_NOTES.md`
for what changed in the current version.

## Entry points

One named entry point per task; each prints a version banner before it loads anything.

| task | entry point |
|---|---|
| SMM estimation for one window | `code/smm/smm_main.jl` |
| standard errors (DE-MC) | `code/smm/MCMC_main.jl` |
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
| settings reach their consumers | `code/scripts/check_forwarding.jl` | a computed setting not forwarded, an unclassified key, a caller/callee name mismatch, an undeclared rename |
| clone completeness | `code/scripts/check_repo.sh` | a missing estimation input |
| bundle compatibility | `code/scripts/migrate_bundles.jl` | a rebuilt bundle not reproducing its stored Q |

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
| per-coordinate ΔQ = 1 half-widths | `output/smm/sa_proposal_widths_base_fc.csv` | widths span 0.0022–0.20 (91x, median 0.021); 10 of 23 coordinates fall below the 0.01 the retired scalar step clamped at |
| proposal comparison, 3 seeds at equal budget | `output/smm/sa_proposal_arm_comparison_base_fc.csv` | mean ΔQ 81.2 at the retired fixed k=3, 87.8 for a dense mask, 90.3 for the shipped mask, 91.5 for a single-coordinate move — reported with acceptance **and** distance moved |
| joint vs component feasibility | `output/smm/sa_proposal_joint_feasibility_base_fc.csv` | 0 of 90 draws had a feasible joint move with all components infeasible — the retired justification for multi-coordinate moves does not replicate here |
| matched-displacement sparsity scan | `output/smm/sa_proposal_sparsity_scan_base_fc.csv` | feasibility saturated (0.958–1.000 across k), so it cannot identify the count; this scan favoured k ≈ 11, which the chains above overruled |

Two methodological points, both load-bearing and both recorded in `SETTINGS.md`'s
graveyard entry for `sa_subset_k`. Acceptance and expected squared jump distance are
both functions of `k·σ²` to second order, so neither identifies the coordinate count
separately from the step scale — which is why `p_move` is fixed rather than adapted
against acceptance, and why every comparison here reports descent alongside acceptance.
And a sparsity ranking taken from one-shot draws disagreed with the same ranking taken
from chains: a large joint move is the better single bet and the worse thing to repeat.
Sparsity questions get measured on chains.
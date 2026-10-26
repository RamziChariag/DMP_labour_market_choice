# RoySearch changelog

One file, newest first, one entry per version. **This is the answer to "what changed, and
can I still compare my old numbers to my new ones?"** — the question a version number
exists to answer and a git log does not, because a commit message records what was edited
while an entry here records what it did to the results.

## How to use this file

**Every version bump writes an entry here, in the same pass as the bump.** Not afterwards
and not from memory: the release gate is what establishes the numbers, so the entry is
written while they are in hand. A version whose entry says "TODO" has not shipped.

Each entry states, in this order:

1. **Layer and why** — MAJOR / MINOR / PATCH, and the one sentence that justifies it
   against the layer test (MAJOR: would the theorist reading the paper need to be told?
   MINOR: does a run reproduce the old numbers? PATCH: is every number bit-identical?).
2. **Comparability** — whether results from the previous version can be compared to this
   one, and if not, exactly which stored outputs are superseded. This is the field readers
   come for.
3. **What changed** — grouped by subsystem, with `file:line` where a reader would look.
4. **Measured, not asserted** — the numbers the release gate produced. A claim without a
   number in this section is a claim the next reader should not trust.
5. **Settings** — any setting added, retired, or whose default moved. Cross-reference
   `SETTINGS.md`; a setting change that is not in both files is a settings bug waiting to
   happen.
6. **Known gaps** — what was not verified, what is still broken, what was tried and
   rejected. Writing "nothing" here is a claim too.

**On honesty in the "what was tried and rejected" lines.** They are the highest-value part
of this file and the easiest to omit. A rejected fix that is not recorded gets re-proposed,
re-implemented, and re-measured — three times the cost of one sentence. Record what the
measurement was, not just the verdict.

---

## v61.3.0 — 2026-10-24

**MINOR: `MCMC_INIT = :resume` continues a stored DE-MC run.** With `MCMC_GENS` the new
total, the run takes the final population of the window's chain file, completed or a
snapshot, runs the remaining generations, and writes and prints what one run of that total
writes. A run that does not resume reproduces v61.2.1 bit for bit.

**Comparability.** Full; nothing is superseded. A fresh chain is bit-identical to v61.2.1
(measured below). Chain files now also carry `outcome`, `demc_state` and `sampler`, and the
snapshot's `prior` moved into `sampler`; every older chain file still loads, and the stored
`chain_baseline_diagonalW.jls` (5,000 generations) resumes.

### What changed
- `code/mcmc/demc.jl`
  - `:55` `_proposal_draws`: the random part of a proposal (difference pairs, crossover
    mask, the two shocks), taken out of the generation loop (`:694`) in the same order, so
    the sampler and the replay draw one stream.
  - `:73` `_replay_rng!`: advances an RNG through every draw of `gens` generations without a
    solve; how many numbers a generation draws does not depend on the target.
  - `:95` `_outcome_from_history`: LMR's outcome codes for a file that did not record them,
    from `accepted` and `chain_lp`.
  - `:110` `_DEMC_STATE`, `_demc_state`: the loop state (RNG, stuck-chain count and last
    generation, best point, DE-scale adaptation, the print, adaptation and check windows),
    read off the sampler's locals by name, so one list serves snapshot, return and resume.
  - `run_demc`: keyword `resume` (`:201`); `g0`, the generations already run (`:304`);
    `init = :resume` takes the stored final population and log-target (`:490`); the stored
    history and state are restored, the RNG replayed from the base seed where a file has
    none, and a file without adaptation or check state refuses a run that uses either
    (`:629`); the loop runs from `g0 + 1` (`:650`); the chain dump moved to the end of the
    generation and passes `outcome` and the state (`:1027`); `state` is returned (`:1054`).
    The header's `init=` reads `resume from gen <g0>`.
- `code/mcmc/MCMC_main.jl`
  - `:212` `:resume` documented beside the other starts.
  - `:955` `SAMPLER` (`MCMC_SPACE`, `MCMC_PRIOR`, `MCMC_T_BOX`, `MCMC_CR`, `MCMC_DELTA`),
    recorded in every chain file.
  - `:986` `snapshot_chain` writes `outcome`, `demc_state` and `sampler`.
  - `:1108` §4c: loads the chain file; refuses `MCMC_GENS` at or below its generations and
    any difference in window, free parameters, bounds, pinned values, active moments, their
    data values, W, `MCMC_N` or `SAMPLER`, naming each; re-solves the stored final
    population and requires the stored log-target bit for bit.
  - `:1182` `resume = RESUME` forwarded; `:1222` the post-run snapshot carries the new fields.
  - `:1361` a retained draw whose vacancy costs are not in the solve cache is solved once per
    distinct point before `k_draws` is built: on a resume, the draws of stored generations
    when `MCMC_BURN` opens the retained window before the stored run's end; otherwise none.
  - `:1893` the final chain file carries `outcome`, `demc_state` and `sampler`.
- `code/smm/settings.jl:247` the `MCMC_INIT` entry lists `:resume`. `SETTINGS.md` (MCMC
  defaults) and `README.md` (the MCMC path): one entry each.

### Measured, not asserted
Scratch copy, Julia 1.10.10, 2 threads.
- **The stored baseline chain resumes.** `chain_baseline_diagonalW.jls` (24 × 120 × 5,000,
  v61.2.1, n_replaced 33, last replacement at generation 3,935, no outcome codes, no RNG
  state) passes §4c at `MCMC_GENS = 7000`, and the 120 re-solved final log-targets equal the
  stored ones bit for bit (max |Δ| = 0) at 2 threads, against the thread count it ran at.
- **Its RNG is rebuilt exactly.** Replaying its draws from `MCMC_SEED` through generation
  3,935 and recomputing every proposal of generations 3,936–5,000 from the stored
  populations reproduces all 26,947 accepted positions bit for bit; every accepted move
  passes its replayed Metropolis draw; no candidate left the box. This confirms the run's
  δ = 2, CR = 1.0, shocks and DE scale, which the file does not record, and that the outcome
  codes rebuilt for generations 3,936–5,000 (the 7,000-generation run's LMR blocks read from
  4,101) are exact.
- **Exact continuation, through `run_demc`** (an analytic 4-d target with an infeasible
  region; N = 8, box ±6 binding 120 times, 17 replacements, DE-scale braking, check every
  25): a straight 100-generation run against 60 generations resumed to 100 from the returned
  state: chain, chain_lp, accepted, outcome, replaced, draws, population, best point,
  replacement counts, DE scale and the whole state bitwise identical, and all 17 printed
  lines after generation 60 identical. Resumed from the straight run's snapshot at 40:
  identical. With braking firing (n_adapt 6): identical.
- **No RNG state** (a v61.2.1-style file: n_replaced, last_replace and the best point only,
  outcome rebuilt): two resumes identical to each other; against the straight run every
  array and every printed line identical, and the state differs only in four counters
  (`wacc_ad`, `wfin_ad`, `wacc_chk`, `wprop_chk`) that a run with adaptation and the check
  off never reads. With box rejections and replacements present the arrays are still
  identical; 116 of 480 rebuilt outcome codes then differ from the recorded ones, 37 box
  rejections read as weak and the rest in cells the stuck-chain rule overwrote.
- **`MCMC_main.jl` end to end** (crisis_fc, N = 6, `MCMC_BURN = 0`, LMR blocks and
  snapshots every 2 generations, `MCMC_JAC_METHOD = design`): a straight 8-generation run
  against 4 generations resumed to 8: chain file identical byte for byte in every field but
  provenance (chain, draws, chain_lp, accepted, outcome, replaced, k_draws, θ̄, se, Ĝ, spec,
  demc_state with its RNG), the estimate bundle identical (θ̄, Q 2878.235201990958, se, mcmc, vacancy costs),
  the results CSV byte-identical, and the 52 console lines from generation 6 on identical.
  The retained window reached into the stored generations, so the vacancy costs of their
  draws were solved after the run: 96 of 96 finite and equal. The resumed run killed after
  its generation-6 snapshot (`partial = true`, holding generations 1–6) and resumed from that
  snapshot: chain file, estimate, CSV and the console from generation 8 identical to the
  straight run's again.
- **Refusals**: `MCMC_GENS = 5000` on the 5,000-generation file; `MCMC_N = 60` ("It differs in
  MCMC_N (stored 120, this run 60)"); ρ_x's upper bound changed in the stored spec ("It
  differs in bounds"), all before any solve; in `run_demc`, gens at or below the stored
  count, a different N, a missing `resume`, and `check_every > 0` on a file without state.
- **A fresh run is unchanged**: v61.2.1's `demc.jl` against this one on the analytic target
  (100 generations, burn 0.8, braking and check on): all 22 returned fields, the 44 printed
  lines and the 4 snapshots identical.
- **Gates**, against v61.2.1: `check_forwarding` reports the standing
  `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` row and nothing new (run_demc's keywords 29 → 30, `resume`
  forwarded); `check_dead_settings` PASS (196 keywords, 195 before); `check_structure` the
  same 7 files; `check_output_tree` PASS.

### Settings
`MCMC_INIT` gains `:resume`; its default stays `:at_seed`. No setting added or re-defaulted.
Bundle audit skipped: the change reaches the sampler and `MCMC_main.jl`'s chain I/O only,
not the solver, the objective or the moments, and a fresh chain is bit-identical (above).

### Known gaps
- The stored generations are the stored run's. A 5,000-generation run ended burn-in at
  4,000, so its stuck-chain rule stopped there and overwrote 1,000 generations back, where
  one 7,000-generation run would have kept it on to 5,600 and overwritten 1,400 back. From
  generation 5,001 the resumed run applies the 7,000-generation rules; the stored run's last
  replacement was at 3,935.
- A run never checks its own last generation, so with `MCMC_CHECK_EVERY > 0` a resume from a
  completed file skips the check one run would make there; from a snapshot it does not. And
  when a completed file's last generation is not a multiple of `MCMC_PRINT_EVERY`, the first
  progress line after the resume covers a shorter window than one run's.
- A file written before v61.3.0: `MCMC_CR` and `MCMC_DELTA` go unchecked (it records neither);
  the RNG replay is exact for a run started `:at_seed` only; rebuilt outcome codes read a box
  rejection as weak and, in spans the stuck-chain rule overwrote, read the copied chain.
- Resumes were run end to end on crisis_fc at N = 6 only; on the baseline window only the
  start (load, guard, re-evaluation) was run, since its chain takes days.
- Prints: with `MCMC_CHECK_EVERY > 0` a snapshot line now follows its generation's check line.

### Tried and rejected
- Seeding the continuation from a fresh stream (`MCMC_SEED` and the stored generation count),
  as first specified: reproducible, but not the stored run's stream. The replay costs no
  solve and makes the continuation of the stored baseline chain exact (measured above).

---

## v61.2.1 — 2026-10-24

**PATCH: the parameter tables print k_U and k_S, not the tightness pair.** The tightness
pair is the coordinate the optimiser searches over; the parameters are the vacancy costs,
which follow from free entry. Prints only; every number in every output file is unchanged.

**Comparability.** Full.

### What changed
- `code/smm/smm_params.jl:695` `is_tightness`: names the search coordinate.
- `print_spec` (`smm_params.jl`) lists every free parameter except the tightness pair, then
  k_U and k_S (per period for the baseline) with blank bounds and their values at the seed;
  the count in the header counts k, not the tightness pair. `smm_main.jl` passes the costs
  at the seed (one solve per period).
- `print_results` (`smm.jl:2825`) skips the tightness rows; the k rows below it are unchanged.

### Measured, not asserted
`crisis_covid` estimate printed in a scratch copy: 8 free parameters, A … λ_S then k_U, k_S;
no tightness row in either table.

### Settings
None.

### Known gaps
The CSV writer and the `MCMC_main.jl` table still carry the tightness rows.

---

## v61.2.0 — 2026-10-24

**MINOR: `MCMC_main.jl` and `transition_main.jl` pin BLAS to one thread, as `smm_main.jl`
does.** Before, both ran with OpenBLAS's default of one thread per core unless
`OPENBLAS_NUM_THREADS=1` was exported, and that oversubscribed the cores inside their
threaded solves.

**Comparability.** A run that exported `OPENBLAS_NUM_THREADS=1` (`ops/vm_run.sh`,
`ops/setup_vm.sh`, every measurement in this file) is unchanged. A run that did not export
it may differ in the last bits, where a multi-threaded BLAS call summed in another order.

### What changed
- `code/mcmc/MCMC_main.jl:75`, `code/transition/transition_main.jl:95`:
  `LinearAlgebra.BLAS.set_num_threads(1)` after the package load, with the comment of
  `smm_main.jl:87`.

### Measured, not asserted
Both files parse. No run: the call is the one `smm_main.jl` has made since before v40.

### Settings
None. `OPENBLAS_NUM_THREADS` is no longer needed for any of the three entry points.

### Known gaps
`model_main.jl`, `policy_main.jl` and the tools do not pin BLAS.

---

## v61.1.0 — 2026-10-24

**MINOR: `INIT_MODE = :warmstart_base`.** A crisis window starts from the baseline point at
its period and reads none of its own stored files; the baseline runs as under `:warmstart`.
The existing modes are unchanged.

**Comparability.** Full; no stored output is affected.

### What changed
- `code/smm/smm_main.jl:942` §8a: the candidate list is the baseline point alone for a crisis
  window under `:warmstart_base`; the INIT_MODE comment (:287) and the §8a header (:907) say
  so. No print added: the warm start's one line names the file it started from.
- `code/smm/settings.jl:68`, `README.md` (start-point table) list the mode.

### Measured, not asserted
Scratch copy, `crisis_fc`, `CRISIS_BASE = :smm`, with the v60 crisis files present and a
baseline bundle at the seeds: `:warmstart` starts from
`estimate_crisis_fc_diagonalW_smm_mcmcbase.jls` (Q 6954.935581); `:warmstart_base` from
`estimate_baseline_diagonalW_smm.jls` (Q 258796.199088). The baseline under `:warmstart_base`
prints the `:warmstart` line (no candidate, the DEFAULT values). `check_forwarding`: the known
`ROYSEARCH_MCMC_JAC_ALLOW_GAPS` only.

### Settings
`INIT_MODE` gains `:warmstart_base`; the default is unchanged.

### Known gaps
None found.

---

## v61.0.1 — 2026-10-24

**PATCH: the baseline's period tags read `[fc]` and `[covid]` instead of `[03]` and `[15]`.**
Labels only; every number in every output is unchanged.

**Comparability.** Full. A bundle written under v61.0.0 carries the old tags in its stored
parameter labels; nothing matches on a label (starts and caches match on block and name), and
`MCMC_main.jl` rebuilds the joint spec from the period specs, so it prints the new tags.

### What changed
- `code/smm/smm_params.jl:600` `PERIOD_TAG`: `base_fc => "fc"`, `base_covid => "covid"`.
- Param columns widened for the longer tags: `print_spec` (8 → 11), the `MCMC_main.jl`
  parameter and vacancy-cost rows (9 → 12) and both headers.
- `README.md`, "The estimation windows".

### Measured, not asserted
Baseline spec built from `smm_main.jl` in a scratch copy: joint Q at the seeds
14756.3110199568, as under v61.0.0; the spec, results and pinned tables print `[fc]`/`[covid]`.

### Settings
None.

### Known gaps
No bundle audit: the only v61 bundle is the author's running baseline.

---

## v61.0.0 — 2026-10-24

**MAJOR: three estimation windows replace the four base and crisis windows and the four
counterfactual windows.** `baseline` estimates the two former base windows jointly, with
ρ_x, b_U, b_S and σ_S common to both periods; `crisis_fc` and `crisis_covid` hold the
training cost, skilled productivity, matching efficiency and the cost of search on the job at
the baseline's period, beside the ability distribution, and estimate the rest. The estimation
problem changed, so Q moves and the theorist must be told.

**Comparability.** None for estimates. Every stored estimate, chain, CSV and candidate bank
in `output/estimates/` and `output/chains/` belongs to a window that no longer exists and is
superseded; all three windows need estimating. No migration: the change is a
respecification, not a reparameterisation. The solver and the moments are unchanged: a
baseline period evaluated at the v60 chain estimate of its old window reproduces that
window's Q and vacancy costs bit for bit (below).

### What changed
- **Windows.** `code/paths.jl:51` `WINDOWS` (`baseline`, `crisis_fc`, `crisis_covid`), :53
  `DATA_WINDOWS` (the four data windows), :55 `BASELINE_PERIODS`; `PAIR_BASE` (:151) maps data
  windows to their baseline period; `data_windows` (:157) and `estimation_window` (:158) map
  between the two lists. `COUNTERFACTUALS`, `is_counterfactual`, `data_window` and
  `legacy_estimate_path` are removed. `code/smm/moments.jl`: the loaders read a data window's
  own files.
- **The designs.** `code/smm/smm_params.jl:844` `BASELINE_COMMON`, :848 `CRISIS_HELD`; :855
  `PERIODS`, the `spec.fixed` record of the baseline's periods (it replaces `PAIRED_BASE`).
- **The joint spec.** `joint_spec` (`smm_params.jl:614`): one parameter vector in the
  canonical order, a common parameter once and every other free parameter once per period,
  named `<name>@<period>` and labelled `[03]`/`[15]`; moments keyed `<moment>@<period>`;
  block-diagonal W; common starts checked equal. `period_list` (:659), `pinned_key` (:667),
  `pinned_table` (:676), `vacancy_cost_rows` (:698), `k_values`/`k_struct` (:712, :715);
  `assert_all_params_accounted` (:906) checks period by period; `unpack_θ` refuses a joint
  spec (:1136); `print_spec` prints the pinned table.
- **Evaluation.** `code/smm/smm.jl:220` `equilibrium_at` sends a joint spec to
  `_joint_equilibrium_at` (:244): the periods are solved in turn, never in parallel, and Q is
  their sum; `moments`, `model` and `k` are per period. `attracting_check` (:367) and
  `attracting_line` (:401) run per period. `point_in_spec` (:89) maps period by period.
  `print_results`/`save_results` print every period's costs and pinned values.
  `period_bundle` (:2948) is one period of a baseline bundle as a single-period estimate: its
  coordinates, spec, standard errors, vacancy costs and attracting check. The held-choice
  lookup (`held_choices`) is removed; the solver's `HeldChoices` machinery stays, inert when
  unset, for the counterfactual.
- **`smm_main.jl`.** `WINDOW` defaults to `:baseline` and is validated (:165);
  `DEFAULT_PERIOD_PARAMS` (:345) holds each period's seeds, and `default_seed` (:380) takes the
  period; `data_window_setup` (:420) loads one data window's moments, holdouts, W, r/ν/φ and σ_w;
  §4–§6 build the per-period inputs; §8 joins them (:890); the warm start (:941) reads a crisis
  window's baseline candidate at its period. `_PAIR_BASELINE` and the counterfactual holdouts
  are removed; `WINDOW_SKIP_MOMENTS` is keyed by data window and empty.
- **Seeds.** Each period's `DEFAULT_PERIOD_PARAMS` are the v60 chain estimate of its old
  window; the four common parameters start at the mean of the two.
- **`MCMC_main.jl`.** The spec is rebuilt period by period on each data window's weights
  (`chain_spec`, :679); the vacancy costs are `NK` (:701) entries, two per period, through the
  Jacobian (:1354), the posterior summaries, R̂/ESS, the table and the CSV; the attracting
  check reads each period's tightness columns (:1393); a crisis chain checks `CRISIS_HELD`
  against the baseline chain's period.
- **Transitions and tables.** `point_estimate` (`transition_simulation.jl`) reads a baseline
  period through `period_bundle`; `check_pair` compares every `CRISIS_HELD` parameter and
  r, ν, φ (the old loop read held fields off `CommonParams` only);
  `load_all_smm_bundles!` (`plots_and_tables/transition.jl:159`) and `model_main.jl` read the
  period view; `transition_panel.jl` drops `data_window`.
- **Gates, smoke, tools.** `check_pair_map.jl` is removed with the duplicate map it checked;
  `check_repo.sh` lists data and estimation windows separately; `check_output_tree.jl`,
  `check_hire_margin.jl` (per data window, period view), `smoke_transition.jl` and
  `migrate_bundles.jl` follow the windows. `settings.jl`, `README.md` (new section "The
  estimation windows"), `SETTINGS.md` updated.

### Measured, not asserted
All in a scratch copy at 2 threads, production grid (Nx 160, Np_U 160, Np_S 180).
- Every file parses (`Meta.parseall`, 0 failures). `smm_main.jl` builds all three specs;
  `assert_all_params_accounted` passes for each.
- Baseline spec: 24 free coordinates, 70 moments, W 70×70, 21 pinned rows (ν and σ_w per
  period). Joint Q at the seeds 14756.3110199568 = 6201.4565569033 + 8554.8544630536, bit for
  bit, and `equilibrium_at(...).Q == smm_objective(...)`.
- Each period's spec at its old window's v60 chain estimate, every coordinate at the old value:

  | period | Q, v61 | Q, v60 | rel | k_U, k_S |
  |---|---|---|---|---|
  | base_fc | 4164.9378811539 | 4164.937881153949 | 0 | 1.020923, 1.019927 (identical) |
  | base_covid | 3645.7681633087 | 3645.768163308681 | 0 | 0.660014, 0.505749 (identical) |
- Crisis specs: 8 free (A, b_U, b_S, α_U, λ_U, λ_S, θ_U, θ_S); `CRISIS_HELD` and ν at the
  baseline's matching period. `period_bundle` round-trips a written baseline bundle;
  `point_in_spec` of a baseline bundle into its own spec returns its vector bit for bit.
- `MCMC_main.jl` with `MCMC_JAC_ONLY = true` on a scratch baseline bundle at the seeds: 24 of
  24 coordinates in 288 solves, 0 not converged, 93 s; table and CSV with k_U, k_S per period.
- Transition loader: `point_estimate(:base_fc)` reads the baseline's period (14 coordinates),
  `point_estimate(:crisis_fc)` the crisis bundle (8), `check_pair` passes.
- Gates: `check_threading -t 2` on the joint baseline, Q 14756.3110199568 serial and twice
  threaded, bitwise equal; `check_forwarding` reports the known
  `ROYSEARCH_MCMC_JAC_ALLOW_GAPS`; `check_dead_settings` PASS; `check_structure` the same 7
  files; `check_output_tree` PASS.

### Settings
`WINDOW` takes `:baseline`, `:crisis_fc` or `:crisis_covid` (default `:baseline` in both
entry points); any other value stops the run. No setting added or retired.

### Known gaps
- Not run: a DE-MC chain (the joint k summaries, R̂/ESS and `k_post` paths), a transition
  path, the smoke tests.
- The diagnostic tools that read `WINDOW` directly (`rank_diagnostic`, `width_audit`,
  `plateau_probe`, `check_tau_margin`, `transition_audit`, `ojs_continuity_probe`,
  `test_generator`) were not ported: on the baseline's bundle they stop at `unpack_θ`'s guard.
  `policy/` does not load (pre-existing).

---

## v60.2.0 — 2026-10-22

**MINOR: the four counterfactual windows `crisis_fc_cf1`, `crisis_covid_cf1`, `crisis_fc_cf2`,
`crisis_covid_cf2`, and transitions to them, through the existing entry points.** A run under
this version reproduces every number of v60.1.1 bit for bit on the four existing windows (Q,
every moment, the vacancy costs, every replayed chain bundle and CSV, the transition path); the
new windows are new estimation problems, named by their window, and supersede nothing.

**Comparability.** Full. No stored output is superseded and no struct that a stored file holds
changed: the solver's `Model` gains a field (`held`, below), but no stored file contains a
`Model` (`output/transition/` is empty).

### What changed
- **The windows.** `code/paths.jl:158` `COUNTERFACTUALS` declares them once, each with its parent
  crisis window and `holds_choices` (cf1); `is_counterfactual` (:163), `data_window` (:167, the
  window whose data files a window reads), `WINDOWS` (:49) and `PAIR_BASE` (:147) list them;
  `PAIRS` (:53) gains `fc_cf1`, `covid_cf1`, `fc_cf2`, `covid_cf2`, the transition scenarios.
- **Data side.** `code/smm/moments.jl:129,169`: `load_data_moments` and `load_sampling_variances`
  read `data_window(window)`'s files, so every consumer (the estimation, the chain's W, the
  tables and panels) reads a counterfactual window's parent's data; `load_calibrated_params`
  (:265) takes the ν row from `PAIR_BASE`. No data file and nothing in `code/data_and_descriptives/`
  changed.
- **Held parameters and moments.** `code/smm/smm_params.jl:717` `base_held_params(window)`: the
  ability distribution, and in a counterfactual window the training cost c too, held at the
  paired base estimate (12 free parameters). `code/smm/smm_main.jl:254`: `WINDOW_SKIP_MOMENTS`
  holds `training_share` and `skilled_share` out in the four windows (33 targeted), the existing
  per-window holdout and its existing print.
- **The base record.** `code/smm/smm_params.jl:726` `PAIRED_BASE`: a counterfactual spec records
  under `spec.fixed.paired_base` the base estimate it holds parameters at (`window`, `point`,
  `file` from the project root, `sha256`, `holds_choices`, the decoded point and its `sim`);
  built by `paired_base` (`smm_main.jl:401`, `using SHA` at :86). `pinned_values` (:727) is the
  pinned-parameter view the spec print, the results print and the results CSV iterate
  (`smm_params.jl:1105`, `smm.jl:2818,2887`), so the record prints nowhere and enters no CSV.
- **cf1 in the solver.** `code/solver/params.jl:371` `HeldChoices` (τ, d and the two node choices
  behind U_U = max{U^search, −c + T} and U_S = max{U_S^(0), U_S^(1)}); `Model.held` (:404);
  `build_model(...; held)` (:497) and `solve_model(...; held)` (`solver.jl:76`).
  `code/solver/unskilled.jl`: `training_choice!` (:200) and `drain_choice!` (:689) are the only
  places τ and d are set, from the values or as held (inner loop :338, `career_choices!` :668,
  residual R3 `solver.jl:164`); `trains_at`, `drains_at`, `skilled_value_at` (:576) are the node
  choices, used by `crossing_value` (:606), `training_value_on` (:624), `crossing_fixed_point`
  (:637), `implied_training_value` (:724) and `skilled_unemployment_value` (:618);
  `HeldChoices(model)` (:702) reads them off a solved model. `implied_cutoffs` carries `held`
  (`solver.jl:190`).
- **cf1 in the SMM layer.** `code/smm/smm.jl:193` `held_choices(spec; grid)`: the base point of a
  cf1 spec solved once per process and grid (cache and lock, :182) and its `HeldChoices`;
  `equilibrium_at` (:234) and `attracting_check` (:360) pass them to the solve. The restriction
  therefore travels in the spec: every reader of a cf1 bundle (warm start, chain, audit,
  transition, tables) solves the restricted model with no further code.
- **smm_main.jl.** `_PAIR_BASELINE` (:183) gains the four windows; `CRISIS` (:194) replaces the
  two-window tests at the output point (:197) and §6 (:532); the windows.json check reads
  `data_window` (:417); §6 holds `base_held_params` (:575, :597) and appends the record (:588);
  the warm start of a counterfactual window (:1037) is its own output file if there is one, else
  its parent's SMM estimate at the same point (`_smm` or `_smm_mcmcbase`), on its free parameters.
- **MCMC_main.jl:660.** The chain seed check compares `base_held_params` (c included in a
  counterfactual window) against the base chain's estimate.
- **Transition.** One entry point, three scenarios per event (`transition_main.jl:35` comment;
  `_scenario_windows`, `transition_simulation.jl:42`, derives the pair from `PAIRS`). Every
  scenario writes the same file, `transition_<scenario>_diagonalW_<point>.jls`, with the same field set.
  A cf1 scenario runs under the cf1 model on the whole path: `transition_values.jl` (header :22)
  holds τ (:552), every U_S envelope (:422, :733, :772), the crosser's node choice (:737) and
  the stay-skilled test (:745, terminal :391) at the held values; `_d_matrix`
  (`transition_solver.jl:541`) returns the held d. `check_pair` (`transition_simulation.jl:99`)
  also checks c for a counterfactual pair; `crisis_window_months` (:57) reads the parent's
  window. `plots_and_tables/transition_panel.jl:157` `episode_calendar` maps every pair to its
  event's calendar; the two panel file names use the scenario name (:843, :1076).
- **Gates and docs.** `gates/check_output_tree.jl:78,80` admit the new windows and pairs;
  `smm/settings.jl:61` the `WINDOW` row; `README.md` (Counterfactual windows; Transition path);
  `SETTINGS.md` (`WINDOW`, `CRISIS_BASE`).

### Measured, not asserted
All in a scratch copy (data symlinked, at most 2 threads).
- The 17 changed `.jl` files parse (`Meta.parseall`, no `:error` or `:incomplete` node); each
  entry point loaded and ran in the smokes below.
- Bitwise at the 8 readable stored estimates (Q, k_U, k_S, all 43 moments), old code vs new:
  identical in every value, e.g. base_fc `_mcmc` Q 4164.937881153949, crisis_covid `_mcmc`
  1945.6895063033105.
- Baseline transition base_covid → crisis_covid at the `_mcmc` points, 13 dates, 3 outer
  iterations: all 82 result and path fields bit-identical old vs new.
- **cf1 invariance**, the base estimate on the base data with its choices held at its own base
  equilibrium, both base windows at both points: Q, θ_U, θ_S, k_U, k_S and every one of the 43
  moments bit-identical to the unrestricted solve (base_fc `_mcmc` Q 4164.9378811539, base_covid
  `_mcmc` Q 3645.7681633087; held τ has 118/135 and d 113/101 fractional cells at
  base_fc/base_covid).
- Held self-transition base_covid → (cf1 at the base estimate), 13 dates: stays at the steady
  state as the unrestricted one does (max deviation θ 0, ur_S 5.3e-10, training share 0).
- Seed evaluation, smm_main.jl's own path to §9, CRISIS_BASE = :mcmc: 12 free, 33 targeted, both
  stocks listed as skipped; seeds are the parents' `_smm_mcmcbase`. Q at the seed:
  crisis_fc_cf1 2739.8658865062, crisis_covid_cf1 4576.4115977525, crisis_fc_cf2
  2625.3633535691, crisis_covid_cf2 3751.7433129520.
- End-to-end smokes (SA 3, NM 3, polish 1; chain N = 8, 4 generations) for crisis_fc_cf1,
  crisis_covid_cf1, crisis_covid_cf2: every file named by the window; the chain bundles have
  stage `:mcmc`; every cf bundle re-scores its stored Q from its own spec (rel 0).
- Transition smokes (one 6-month trial, 3 iterations, the horizon search replaced in the
  harness only), covid at `:mcmc`: the three files have the same 715 field paths and types;
  only data-dependent sizes differ (`theta.crisis`, 13 vs 12; the wage-point vectors of
  `eq_z1`) and `model_z1.held` (`nothing` vs `HeldChoices`). In cf1 τ never moves.
- Threaded cf1 evaluation (2 threads, the held choices first solved inside the threaded
  region): Q bitwise equal to serial.
- Gates: check_forwarding (only the known `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` finding),
  check_dead_settings, check_pair_map (8 pairs), check_output_tree, check_structure,
  check_hire_margin (also on the six smoke cf bundles), check_threading: same results as v60.1.1.
- Bundle audit (`rebuild_bundles.jl audit`) on this code tree: 12 OK (the four `_mcmc` replays
  identical, their CSVs byte-identical); 2 BROKEN, `estimate_crisis_{fc,covid}_diagonalW_smm.jls`
  (written 2026-10-13/14), which do not load under v60.1.1 either. The bump was taken with the
  bundle gate skipped for that reason; nothing was installed.

### Settings
None added. `WINDOW` admits the four new windows; `SCENARIOS` (a constant of
`transition_main.jl`) admits the four new pairs.

### Known gaps
- Plots cannot load in the verification sandbox (GR's Qt build needs a CPU feature it lacks), so
  the panels and tables were not run; the calendar each new pair uses was checked by evaluating
  `episode_calendar`. The model-fit and parameter tables iterate the four estimation windows
  only, so a counterfactual scenario's window has no column there.
- In cf1, `frontier_floor` in the transition result is still read off the values (the point
  where training would beat search), not off the held τ.
- The two BROKEN crisis `_smm` files mean a counterfactual run with `CRISIS_BASE = :smm` finds no
  parent seed and starts from `DEFAULT_PARAMS` until the crisis windows are re-estimated at :smm.

### Tried and rejected
- The held choices as a module `Ref` set by the entry point: a cf1 bundle would then score
  differently depending on who loaded it; the record in `spec.fixed` makes it self-contained.
- The base file and checksum as new `RunProvenance` fields: serialisation is by field position,
  so every stored bundle would fail to load or replay STALE.
- Storing the held τ and d arrays in the spec: they are tied to the grid the bundle was solved
  on; re-solving the base point once per grid under the current code keeps them consistent.

---

## v60.1.1 — 2026-10-20

**PATCH: two always-zero shares are dropped from the chain's periodic acceptance block.** Only a
print changed; every computed value and every output file is bit-identical.

**Comparability.** Full. Nothing stored is superseded.

### What changed
- `code/mcmc/demc.jl:802–806`: the block printed every `MCMC_LMR_PRINT` generations now reads
  `accept  strong … weak …` and `reject  weak …`. Removed: `equal` (Δ log-posterior exactly zero,
  which a continuous criterion does not produce) and `outside prior` (a candidate outside the
  |t| ≤ `MCMC_T_BOX` box, already reported on the progress line as `box=`, shown only when
  nonzero). The five outcome codes are still recorded in `outcome`; only the print changed.

### Measured, not asserted
- `demc.jl` parses (`Meta.parseall`, no `:error` or `:incomplete` nodes).
- `run_demc` on a three-dimensional standard normal (N = 12, 40 generations, `lmr_print = 20`) in
  a scratch copy prints the two new lines at generations 20 and 40 and returns.
- Bundle audit skipped: print-only change in `demc.jl`, no computed value touched.

### Settings
None.

### Known gaps
None found by the gate above. The base_covid chain running at the time of this edit loaded the
v60.1.0 code and prints the old five-way block until it finishes.

---

## v60.1.0 — 2026-10-20

**MINOR: the objective's weighting is one two-valued setting, `WEIGHTING` (`:diagonal` or
`:equal`), and every trace of a moment covariance is gone from the code.** No objective value
moves: at every stored estimate Q is bit-identical, and the diagonal W and the equal-weight Q
are bitwise those of v60.0.0 (below). The model-fit tables change what they print under each
moment (see Comparability), which is reporting, not estimation.

**Comparability.** Every estimate is comparable with v60.0.0. The serialised struct changed:
`SMMRunParams.w_cond_target :: Float64` is now `weighting :: Symbol`, in the same position, so
a bundle written before this version does not load under it (`TypeError: in new, expected
Symbol, got a value of type Float64`; a loud failure, never a silent one). The 14 stored
estimates and 4 chains have been rebuilt agent-side into that form; they are not installed
(Known gaps). The model-fit tables `model_fit_{A,B}` now print the data SE as the sampling SD
σ̂_k of `sampling_var_<w>.csv`, the SE the diagonal weighting uses, where they printed √diag of
the 16-July `sigma_<w>.csv`, which listed 31 of the 43 moments and differed from σ̂_k by a
median factor of 4–12 per window (measured before today's data rerun, so indicative only). The
delta-method model-SE row under each model value is dropped with the sandwich that produced it.

### What changed
- `code/smm/moments.jl`: `run_weighting()` (:146) is the one reader of `ROYSEARCH_WEIGHTING`,
  default `:diagonal`; `weighting_suffix` (:154) maps it to `_diagonalW` / `_equalW` and stops
  on anything else; `load_sampling_variances` (:167) is the one reader of
  `sampling_var_<w>.csv`, used by the weight matrix and by the tables and panels;
  `load_weight_matrix` (:189) takes `weighting::Symbol`. `load_sigma_matrix`, `_read_sigma_csv`,
  `_w_suffix` and the removed-scheme error are deleted.
- `code/smm/smm_params.jl:222`: `weighting :: Symbol = :diagonal` replaces the Float64 code
  (whose struct default was 2.0, equal); the spec print tests it (:1106, :1172).
- `code/smm/smm_main.jl:459`, `code/mcmc/MCMC_main.jl:100`, `code/transition/transition_main.jl:132`,
  `code/policy/policy_main.jl:144`, `code/plots_and_tables/transition.jl:108`,
  `code/plots_and_tables/transition_panel.jl:96`, `code/gates/check_hire_margin.jl:39`,
  `code/gates/check_threading.jl:43`, `code/tools/width_audit.jl:64` read the weighting through
  `run_weighting` and name files through `weighting_suffix`. The four private suffix functions
  (smm_main via moments.jl, MCMC_main, transition_main, policy_main, plus `_w_suffix_pt` in
  the tables) are gone. The chain file's top-level `w_cond_target` is `weighting`
  (`MCMC_main.jl:1832`); `MCMC_main.jl:698` gates σ̂ on `WEIGHTING === :diagonal`.
- `code/plots_and_tables/transition.jl`: `compute_smm_standard_errors` (a sandwich on the full
  `sigma_<w>.csv`, used only for the model-moment SE row) is deleted; `_load_data_and_se`
  (:194) reads `load_sampling_variances`; `_build_model_fit_tex` (:382) and
  `make_model_fit_tables` (:486) lose the `se_model_mom` argument.
- `code/plots_and_tables/transition_panel.jl`: `load_sigma_ses` becomes `load_data_ses`
  (:203, σ̂_k); standalone, the file now loads the solver and the SMM layer (:85) instead of
  stubbing `Model`, since the SMM layer needs the solver's duration thresholds.
- `code/smm/settings.jl:69`: registry row `WEIGHTING`; `code/gates/check_forwarding.jl:57,150`
  scans `moments.jl` for `env_setting` reads; `code/gates/check_repo.sh:48` requires exactly
  `moments_<w>.csv` and `sampling_var_<w>.csv` per window.
- Comments and docs: `smm/smm.jl` (compute_loss_matrix docstring), `tools/rank_diagnostic.jl:65`,
  `smoke/README.md`, `README.md` (file table; the data-SE source of the model-fit tables and
  panels), `SETTINGS.md` ("Settings that choose a file": `WEIGHTING`).
- ρ_x box, `smm_params.jl:640`: (−0.90, −0.10), from (−0.98, −0.20), set by the author (final
  edit 13:52). The widening to (−0.98, +0.50) proposed in this pass was withdrawn: at the
  base_covid estimate of 13:5x (Q 3645.52, ρ_x = −0.4242) the bound is not near binding. Stored
  ρ_x lie in [−0.494, −0.250], inside the new box, so no stored point is clamped.

### Measured, not asserted
- The 15 changed `.jl` files parse (`Meta.parseall`).
- Q at each stored estimate, v60.0.0 on the original file against this code on the rebuilt
  file (`equilibrium_at(θ, spec).Q`), all 14 bit-identical to 17 digits:

  | estimate | Q |
  |---|---|
  | base_fc smm / mcmc / untagged | 4192.601310556026 / 4192.784440507764 / 5609.8086833222142 |
  | crisis_fc smm / smm_mcmcbase / mcmc / untagged | 2630.7827858309083 / 2632.166542032021 / 2630.5028745527684 / 3419.6247679486833 |
  | base_covid smm / mcmc / untagged | 3657.0470277512504 / 3657.2874084619361 / 4927.6919718308072 |
  | crisis_covid smm / smm_mcmcbase / mcmc / untagged | 1876.5391298933014 / 1876.5855393527359 / 1874.7799006591492 / 2245.9994946963843 |

- Leaf-by-leaf comparison of the 18 rebuilt files against the originals (9,849 leaves, the
  sha256 of each serialised leaf): the only differences are the retyped field (32 instances,
  0.0 → `:diagonal` in all), the chain key (4) and the provenance stamps (version → 60.1.0,
  the arriving version appended to `migrated_from`, `written_at`, and `git_sha` in the chains).
- The author's base_covid SMM checkpoint written at 13:06 with this struct
  (`estimate_base_covid_diagonalW_smm.jls`, v60.0.0 stamp) loads and re-evaluates to
  3826.3796097593822 = its stored Q.
- `load_weight_matrix` under `:diagonal` returns a W bitwise equal to v60.0.0's
  `cond_target = 0.0` at the four `_smm` specs (sha256 of the bytes); under `:equal` it returns
  `nothing` and Q at the stored point is bitwise v60.0.0's `cond_target = 2.0` value: base_fc
  3.8270795404850753, crisis_fc 8.5260262624785739, base_covid 88.234038447226311,
  crisis_covid 879.33037222611404.
- With `sigma_*` and `moment_scales_*` absent (scratch `data/derived` view): `transition_main.jl`
  loads, prints `WEIGHTING: diagonal (_diagonalW)`, and writes the model-fit, parameter and
  x̄ tables; with the four `_smm` bundles loaded, `model_fit_{A,B}` carry 43/43 finite data SEs
  per window; the panels' `load_data_ses` returns 43 per window, standalone and under the
  orchestrator. Under v60.0.0 the same view makes `check_repo.sh` report 8 files MISSING.
- Gates on this tree, against v60.0.0: `check_forwarding` 79 rows / 78 keys read, the one known
  finding (`MCMC_JAC_ALLOW_GAPS`); `check_dead_settings` PASS (190 kwargs); `check_structure`
  the same 7 files; `check_output_tree` PASS; `check_pair_map` OK; `check_threading -t 2` at
  base_fc smm: serial and threaded Q = 4192.6013105560, bitwise equal; `check_hire_margin` OK
  at all 10 estimates, 0 disagreements; `check_version_sync` in sync.
- ρ_x box (−0.98, +0.50), measured on the stored points through `point_in_spec`, the warm
  start's own mapping (the old box's t decoded to θ, re-packed in the new box): the author's
  13:06 base_covid checkpoint (ρ_x = −0.2502987278, box (−0.98, −0.20), t 2.674655 → −0.027836)
  keeps every natural-unit value to 8.9e-16 and re-evaluates to 3826.3796097594 under its own
  moments and W, rel 8.9e-16. The six base-window bundles keep their natural values to
  ≤ 2.2e-16; the four tagged ones (`_smm`, `_mcmc`) re-evaluate to their stored Q to ≤ 1.1e-14
  rel, and the two untagged pre-v57.0.0 files to their own-box re-evaluation, 4927.6919718308
  and 5609.8086833222, exactly (their stored `loss_opt`, 3877.568 and 4377.328, predates the
  current code). None is clamped (stored ρ_x lie in [−0.494, −0.250]). The eight
  crisis bundles hold ρ_x in `spec.fixed` (`ABILITY_PARAMS`), so the box does not enter them.
- Grep over `code/`, `README.md`, `SETTINGS.md`: zero hits for `sigma_$`, `moment_scales`,
  `cond_target`, `COND_TARGET`, `full-Σ`, `Σ_full`, `load_sigma`, `_w_suffix`, `fullW`,
  `compressedW`, `ROYSEARCH_W_SUFFIX`. `compressed` has 5 hits of other meanings (SIPP file
  compression in `sipp.jl`; a shock distribution in `solver/params.jl:211`).

### Settings
- `W_COND_TARGET` (Float64, 0.0 diagonal / 2.0 equal) retired; replaced by `WEIGHTING`
  (`:diagonal` default, `:equal`), `SETTINGS.md`. No `LEGACY_ENV_KEYS` entry, by the order that
  the old construction leave no trace: an exported `ROYSEARCH_W_COND_TARGET` is now ignored.
- `ROYSEARCH_W_SUFFIX` (a raw ENV read in two gates and `width_audit.jl`) retired; they read
  `WEIGHTING`.
- Behaviour of defaults: `plots_and_tables/transition.jl` and `transition_panel.jl` run
  standalone defaulted to `_equalW`, and `policy_main.jl` to `_fullW` (`W_COND_TARGET = 1e8`);
  all now follow `WEIGHTING`, default diagonal. `solver/model_main.jl`'s constant
  `W_SUFFIX = "_equalW"` is a setting of that script and is unchanged.

### Known gaps
- **Stored bundles: rebuilt, not installed.** Under this code the files in `output/` written
  before it do not load; the warm start of the 13:0x base_covid run found no candidate for that
  reason and started from `DEFAULT_PARAMS`. The rebuilt files and a manifest (sha256 of each
  staged and repo file) are in a scratch stage outside the repo. The staged
  `estimate_base_covid_diagonalW_smm.jls` must NOT be installed: the repo copy is the author's
  new file. The estimate it replaced (Q 3657.0470277512504 under the v60 data) survives in the
  stage. The bump was taken with the bundle gate skipped for this reason.
- The bundle audit (`rebuild_bundles.jl audit`) was run on a scratch repo holding this code and
  the rebuilt set and stopped after the first replay, which showed the supersession below; the
  tool's replay now exports `ROYSEARCH_WEIGHTING` instead of the retired key (agent-side).
- The data pipeline was rerun at 12:31–12:42 (v60.0.0 targets). That supersedes every stored
  `_mcmc` replay: `MCMC_main.jl` rebuilds W from the new `sampling_var_<w>.csv`, and the
  replay of base_fc's chain gives Q(θ̄) = 4194.8592 against the stored 4192.7844. The diagonal W
  rebuilt from today's files differs from the stored specs' W by up to 8–83% per window. Not
  caused by this change, which reproduces both under either data.
- `policy_main.jl` still does not load: it stops at `smm/smm.jl:2185` (`Optim` not in scope),
  identically under v60.0.0, before it reaches its weighting line or any data read.
- Plots/GR does not load in the agent sandbox, so the scatter and the panel figures were not
  rendered; their data reads were.
- `data/` was not written: `data/derived/README.md` still lists `sigma_{w}.csv` and
  `training_share_scale.csv`. Files in `data/derived/` that nothing in `code/` writes or reads:
  `sigma_<w>.csv`, `moment_scales_<w>.csv`, `sigma_reduced_<w>.csv`, `W_<w>.csv`,
  `W_reduced_<w>.csv`, `scale_<w>.csv` (4 each), `training_share_scale.csv`; also unreferenced:
  `nu_estimate.csv`, `rho_nilf.csv`.

### Tried and rejected
- Renaming the Float64 field in place (loads with no migration): rejected, the stored value
  would still be a numeric code for a two-valued choice.
- A `LEGACY_ENV_KEYS` entry for the old key: rejected under the order that nothing name the
  old construction; the cost is the silent-ignore above.

---

## v60.0.0 — 2026-10-20

**MAJOR: the tightness targets `theta_U` and `theta_S` are now the ratio of window means,
mean V_j over mean U_j, instead of the mean of the monthly V_j/U_j. The transition rates
`jfr_j` and `sep_rate_j` are built the same way: total weighted flow over total weighted
at-risk pool across the window's month-pairs, instead of the mean of the monthly rates.**
These are different data targets, so the theorist has to be told. (The rates were added to
this entry before any run under v60.0.0.)

**Comparability.** None for any window once `data_processing_main.jl` is rerun. Every
`moments_<window>.csv` and `sampling_var_<window>.csv` in `data/derived/` changes, and with them
every stored `_smm`, `_smm_mcmcbase` and `_mcmc` estimate, chain CSV and transition built on them.
Until the data are rerun, the stored estimates still reproduce their Q (audit below), because
`data/derived/` is untouched.

**Why.** The model's θ_j is a stationary V/U, one stock over another. Its data counterpart is
the ratio of the window's average stocks. V/U is convex in U, so averaging the monthly ratios
over a window in which U moves gives a number above that ratio, by more the more U moves.
In 2020–22 the skilled target was 2.321 against 1.713 as a ratio of means, and the model's
equilibrium sat well above the centre of the data cloud in the Beveridge figure.

### What changed
- `code/data_and_descriptives/moments.jl`: `_compute_tightness_per_month` is renamed
  `_compute_tightness` (:271). It sums V_j and U_j over the months where both are observed
  and returns their ratio (:310–330); the population of each U_j is unchanged (U_U on the
  train-excluded labour force, U_S on the unrestricted one). The section comment at :463 says so.
- `code/data_and_descriptives/sigma.jl` is deleted (moved to the Trash). It was not included by
  `data_processing_main.jl` and is not to be used. The comments in `setup.jl` and
  `sampling_variances.jl` that named it are trimmed. Its outputs `sigma_<window>.csv` and
  `moment_scales_<window>.csv` in `data/derived/` are dated 16 July and stay as they are; see
  Known gaps.
- `code/data_and_descriptives/transitions.jl`: each monthly row of `transitions_monthly.arrow`
  now also carries the weighted pools and flows `w_u`, `w_ue`, `w_e`, `w_eu`; the window
  averages in `transitions.arrow` (read by nothing) use the same ratio of totals.
- `code/data_and_descriptives/moments.jl`, `_fill_transition_moments!`: `jfr_j = Σw_ue/Σw_u`,
  `sep_rate_j = Σw_eu/Σw_e` over the window's month-pairs. The old comment called the mean
  n_pairs-weighted; it was unweighted.
- `sampling_variances.jl` (c), the flow-hazard variances, is unchanged: it already uses the
  pooled at-risk Kish N of the window, which is the variance of a pooled proportion.
- `sampling_variances.jl` (d) is unchanged: Var(θ_j) ≈ θ_j²(1 − p)/(p·N_lf) is the delta-method
  variance of a ratio with the CPS unemployment count as the noisy term. It now evaluates at
  the new θ_j.

### Measured, not asserted
- Both edited files parse (`Meta.parseall`, no `:error` or `:incomplete` nodes).
- `_compute_tightness` run alone on the live `cps_basic_clean.arrow` and `jolts_clean.arrow`
  (nothing written):

  | window | θ_U old | θ_U new | θ_S old | θ_S new |
  |---|---|---|---|---|
  | base_fc | 0.458682 | 0.448638 | 1.210646 | 1.162976 |
  | crisis_fc | 0.225768 | 0.204633 | 0.646090 | 0.551108 |
  | base_covid | 0.815122 | 0.778023 | 1.836916 | 1.805411 |
  | crisis_covid | 1.051214 | 0.833493 | 2.320822 | 1.713459 |

- Transition rates: a scratch rebuild of the month-pairs from `cps_basic_clean.arrow` (same
  codes, weights and window assignment as `make_transitions`) reproduces the stored monthly
  means exactly (8 cells, 5 decimals), and `_fill_transition_moments!` run on it gives
  (probabilities, before the hazard conversion in section F):

  | window | jfr_U | jfr_S | sep_U | sep_S |
  |---|---|---|---|---|
  | base_fc | 0.26105 → 0.25985 | 0.27412 → 0.27334 | 0.01514 → 0.01513 | 0.00622 → 0.00621 |
  | crisis_fc | 0.18995 → 0.18281 | 0.21138 → 0.20339 | 0.02035 → 0.02031 | 0.00814 → 0.00813 |
  | base_covid | 0.25650 → 0.25389 | 0.27166 → 0.27497 | 0.01319 → 0.01318 | 0.00590 → 0.00589 |
  | crisis_covid | 0.26974 → 0.27715 | 0.27360 → 0.28402 | 0.02093 → 0.02064 | 0.01033 → 0.01021 |

  The new θ_S equals, to three decimals, mean V_S over mean U_S computed independently from
  the monthly plotting series. θ_U does not, because that series counts U_U without excluding
  trainees.
- Bundle audit on this code tree (14 stored estimates and chain CSVs): all OK. Every stored
  estimate reproduces its Q, as it must while `data/derived/` is unchanged.
- An independent rebuild of the monthly series from the same arrow files with the populations
  of `moments.jl` gives the same ratio of means in all eight cells, to four decimals.

### Settings
None.

### Known gaps
- Nothing writes `sigma_<window>.csv` any more, and the files on disk (16 July) predate the
  current moments. Three readers remain: `plots_and_tables/transition.jl` (`load_sigma_matrix`
  at :206 and the SMM-SE block at :446) and `plots_and_tables/transition_panel.jl`
  (`load_sigma_ses`, SE bands at :785). `code/gates/check_repo.sh:48` requires the sigma and
  moment_scales files to exist. Not changed.
- `transitions_monthly.arrow` gains four columns; `moments.jl` requires them, so
  `make_transitions()` must run before `make_moments()`; `data_processing_main.jl` runs both.

---

## v59.1.0 — 2026-10-19

**MINOR: `MCMC_FROM_CHAIN` is purged from `code/mcmc/MCMC_main.jl`, `code/smm/settings.jl`,
`README.md` and `SETTINGS.md`.** It was always off by default, so a run reproduces v59.0.0's numbers.
The chain path, which is now the only path, is unchanged.

**Comparability.** Full. Nothing stored is superseded. v59.0.0's advice to bring a crisis_covid
chain written before v59.0.0 up to date with `MCMC_FROM_CHAIN=true` no longer applies (see
Tried and rejected).

### What changed
- `MCMC_main.jl`, by v59.0.0 line:
  - the sentence about the setting in the header (:29–31);
  - the setting and its exclusion guard against `MCMC_JAC_ONLY` (:564–571);
  - the `!MCMC_FROM_CHAIN` term in `need_pre` (:1025);
  - `chain_from_file` and its docstring (:1098–1129);
  - the `elseif MCMC_FROM_CHAIN` branch of `res` (:1131–1138, and its closing `end`), with `res` back to v58.2.1's
    `MCMC_JAC_ONLY ? stub : run_demc(…)`;
  - the `MCMC_FROM_CHAIN` conditions on the chain snapshot (:1184–1186), the chain-file write
    (:1807–1808) and the closing print (:2021).
- `settings.jl` drops the `SettingDecl(:MCMC_FROM_CHAIN, …)` row. The file is byte-identical to
  v58.2.1's. `SETTINGS.md` is byte-identical to v58.2.1's. `README.md` drops the paragraph on the
  setting.

### Gate
- Every changed file parses.
- **Scratch chain**, run in fresh trees under v59.0.0 (exact copy) and v59.1.0 with the repo's
  seed. Settings: base_covid, `MCMC_N = 8`, `MCMC_GENS = 40`, 2 threads, same seed.
  - `mcmc_results_base_covid_diagonalW_mcmc.csv`: byte-identical.
  - Bundle: every field byte-identical apart from `provenance.version` (59.0.0 → 59.1.0) and
    `written_at`.
  - Console from the table onward: identical apart from timings.
  - Run times: 216 s and 224 s, slowed by a 10-thread chain running on the same machine.
- `check_forwarding`: 79 registry rows, 78 keys read. One finding, the known pre-existing
  `ROYSEARCH_MCMC_JAC_ALLOW_GAPS`.
- `check_dead_settings`: PASS. `check_structure`: the same 7 files. `check_output_tree`: PASS.
- `check_version_sync`: in sync at 59.1.0.
- No stored output was written or read for writing.

### Settings
- Retired: `MCMC_FROM_CHAIN` (added in v59.0.0, default `false`, never set by a run of the author's).

### Known gaps
None found.

### Tried and rejected
- **`MCMC_FROM_CHAIN`, a setting that rebuilds a window's `_mcmc` bundle and CSV from its saved
  chain** (v59.0.0). Rejected by the author: "I want you to remove MCMC_from_chain, I did not ask
  you to build that. The code is assumed to be working for one model, for one version. no
  rebuilding. Rebuilding is a job for you."
  - Rebuilding stored outputs after a code change is now agent-side work. It runs in a
    standalone tool kept outside the repository (`rebuild_bundles.jl`, with modes audit,
    rebuild and install).
  - The tool runs a scratch copy of the current code. The only change it makes is to substitute
    the sampler call in `MCMC_main.jl`, and it checks that every other line matches the repo.
  - Before the setting was removed it was measured exactly: a fresh run and a rebuild from that
    run's chain gave a byte-identical CSV and bundle (v59.0.0 gate).

## v59.0.0 — 2026-10-19

**MAJOR: the chain path's standard errors use the chain covariance as the bread of the sandwich,
with Ĝ measured at the reported estimate θ̄** (`code/mcmc/MCMC_main.jl`,
`code/mcmc/mcmc_diagnostics.jl`). No point estimate moves, but every reported `se_curvature`
(now `se_sandwich`), `se_bound` and their |t| on the chain path does, so the inference a reader
takes from a chain table is not comparable across this boundary (the v35.0.0 precedent).

**Comparability.** Point estimates, posterior means, quantiles, R̂, ESS and se(chain) are
unchanged. Superseded: every chain-path `se_curvature`/`se_bound`/`abs_t_*` written before
v59.0.0 and the `se`/`vacancy_costs.se` of every `estimate_<w>W_mcmc.jls`. The three finished
windows were rebuilt in this version (below). `crisis_covid` was being re-estimated at release
and was not touched: a chain of it written before v59.0.0 carries the old columns and is
brought to v59.0.0 by `MCMC_FROM_CHAIN=true`. The chain-free route (`MCMC_JAC_ONLY`) is
bit-identical to v58.2.1.

### What changed
- **The estimator** (decided by the author). θ̂ = argmin (m(θ)−m̂)′W(m(θ)−m̂), W = diag(1/σ̂²);
  Var(θ̂) ≈ A Ω A′ with A = H⁻¹Ĝ′W and H the curvature of Q/2. Ĵ = Ĝ′WĜ drops H's residual ×
  second-derivative term, which is large here (Q is 100–200× its degrees of freedom); the
  covariance of the kept draws in θ, P, estimates H⁻¹ over the sampled cloud. On the chain path:
  se(sandw.)ₖ = √[P Ĝ′WĜ P]ₖₖ, se(bound)ₖ = Σᵢ|(P Ĝ′W)ₖᵢ| σ̂ᵢ; for k_U, k_S the same with
  gₖ′P in place of P (gₖ the carried rows of the same Jacobian). Ĝ is `jacobian_adaptive_fd` at
  θ̄ (transform-then-mean, the point written to the `_mcmc` bundle), no longer at the SMM seed.
  All three |t| are |θ̄|/se. JAC_ONLY and an aborted chain keep J⁻¹ as bread and Ĝ at θ̂.
- `se_bound_diagonal` (`mcmc_diagnostics.jl:443`) takes `bread = nothing` (J⁻¹, the old
  computation, operation for operation) or a matrix; returns `(se_bound, se_sandwich)`.
- `MCMC_main.jl`: the draw summaries (se_chain, θ̃_med, θ̄_con, θ̄_bnd, k_draws, k_post) move
  ahead of §5 because Ĝ's centre needs θ̄; `θ̄_t`, `P` and the centre `θ_G` (:1340–1345) feed
  `jacobian_adaptive_fd(θ_G, …)` (:1399) and both `se_bound_diagonal` calls (:1425, :1432).
  θ̂'s attracting check is solved directly on the chain path (:1439; 8 solves), since Ĝk is no
  longer at θ̂. The FD/regression ratio line compares both Jacobians under J⁻¹ (unchanged meaning).
- Console: header `block param estimate se(chain) se(sandw.) se(bound) |t|(chain) |t|(sandw.)
  |t|(bnd) R̂` on the chain path (:1538); `se(J⁻¹)`/`|t|(J⁻¹)` where there is no usable chain.
  No line added or removed.
- CSV `mcmc_results_<w>W_mcmc.csv`: columns 6 and 18 are named by the bread, `se_sandwich` and
  `abs_t_sandwich` (chain) or `se_curvature` and `abs_t_curvature` (no usable chain), same
  position; `abs_t_*` are formed against the reported estimate. No code reads these columns
  or `bundle.se`'s keys (checked: `bundle.jl`, `plots_and_tables/`, `transition/`, `tools/`).
- Bundle: `se = (sandwich, bound, chain)` with `se_source = :chain_bread_fd` on the chain path;
  `(curvature, bound, chain)` with `:jacobian_fd` without a usable chain (an aborted chain's was
  `:chain`). `vacancy_costs.se` is keyed the same way. NamedTuples read by name, so every
  existing bundle still loads; `bundle.jl` header documents the keys.
- **`MCMC_FROM_CHAIN`** (new, default `false`; `MCMC_main.jl:568`, `chain_from_file` :1107):
  reads the finished chain file instead of sampling and runs every later step as a fresh run,
  writing only the `_mcmc` bundle and CSV; never the chain file. Stops for a snapshot, an
  aborted chain, a different seed (basename and k at θ̂ bit for bit) or a generation count
  `MCMC_GENS` cannot produce. The draws' k values are put into the run's own cache, so
  `k_draws` comes out of the same code as after sampling.

### Measured
- **Reproduction of the theorist's reference** (`mcmc_se_corrected.csv`; P from the chain files,
  Ĝ at θ̄ by `jacobian_adaptive_fd`, defaults), max relative difference over all rows incl. k:
  full precision (bundle): base_fc 4.6e-14 / 1.0e-14 (sandwich / bound), crisis_fc 1.5e-14 /
  3.6e-14, base_covid 1.1e-13 / 8.3e-14. From the CSV: 4.0e-6 / 2.0e-6, 2.8e-6 / 5.9e-6,
  5.2e-6 / 3.9e-6 — the CSV's `%.8f` rounding, whose bound 5e-9/se is 4.3e-6, 8.6e-6, 6.9e-6.
- **Rebuild path = fresh path.** Scratch tree, base_covid, `MCMC_N = 8`, `MCMC_GENS = 40`,
  2 threads: a fresh chain (103 s), then `MCMC_FROM_CHAIN=true` on its chain file (53 s). CSV
  byte-identical; bundle: every field byte-identical, provenance equal but `written_at`, and the
  whole serialised bundle byte-identical once `written_at`/`host` are aligned (16,886 bytes);
  console table identical; chain file's sha256 and mtime unchanged.
- **JAC_ONLY unchanged.** base_covid under v58.2.1 and v59.0.0 (2 threads): CSV byte-identical,
  console table identical, the seed bundle it writes byte-identical in every field.
- **Estimates unchanged.** Same short chain under v58.2.1 and v59.0.0: 16 of 20 CSV columns
  identical; only `se_*`/`abs_t_*` (columns 6, 7, 18, 19) differ.
- **Rebuilt windows** (`MCMC_FROM_CHAIN`, 2 threads, 110–123 s each; 600 regression + 156/168
  FD + 8 attracting solves). Against the v58.2.1 bundles: `result`, `spec`, `sim`, `stage`,
  `attracting`, `tag`, `se.chain`, `vacancy_costs.k/post` byte-identical; `mcmc` identical except
  base_covid `ess_min` 1277.685 → 1255.477 (the k rows' ESS, v58.2.0, now in the bundle).
  se(sandw.)/se(chain): base_fc 0.70–1.43, crisis_fc 0.89–1.20, base_covid 0.72–1.08. New/old
  se(bound): 0.43–1.66, 0.82–1.65, 0.49–1.36. Min |t|(bnd): base_fc 4.34 (b_S), crisis_fc 3.70
  (σ_S), base_covid 6.84 (b_U). base_fc's FD at θ̄: 168 solves, 1 not converged, 14/14 columns.
- Gates: every changed file parses; `check_forwarding` 80 registry rows / 79 keys read, one
  finding, the known `ROYSEARCH_MCMC_JAC_ALLOW_GAPS`; `check_dead_settings` PASS;
  `check_structure` the same 7 files as v58.2.1; `check_output_tree` PASS; `check_threading -t 2`
  bitwise (base_fc Q 4192.6013105560); `check_hire_margin` OK; `check_version_sync` 59.0.0.

### Stored output rebuilt
`output/estimates/estimate_{base_fc, crisis_fc, base_covid}_diagonalW_mcmc.jls` and
`mcmc_results_{…}_diagonalW_mcmc.csv`, by `MCMC_FROM_CHAIN=true` in the repository on the saved
chain files (untouched). Pre-rebuild copies are kept outside the repository (session artifact
`backup_v58.2.1`), not in `output/backups/`.

### Settings
- Added `MCMC_FROM_CHAIN` (default `false`), `SETTINGS_REGISTRY` and `SETTINGS.md`.

### Known gaps
- An aborted chain cannot be rebuilt: the chain file does not record `abort_why`.
- `crisis_covid` was not touched (its run was live).
- The aborted-chain path (J⁻¹ columns with se(chain)) was not exercised: no chain aborts under
  the shipped `MCMC_CHECK_EVERY = 0`.

### Tried and rejected
- A point finite-difference Hessian of Q at θ̄ as the bread: indefinite in base_fc and crisis_fc
  along σ_S/b_S/λ_S at steps of 0.5 and 0.25 chain-sd, its inverse changing up to 10× between
  the two steps; positive definite in base_covid but step-dependent up to 2× on the unskilled
  ridge. Q is not quadratic within one standard error.

## v58.2.1 — 2026-10-17

**PATCH: the chain summary in `code/mcmc/MCMC_main.jl`.** Console only; every output file is
bit-identical to v58.2.0's.

**Comparability.** Full. No stored output is superseded.

### What changed
- The first summary line shows `chains=N` in place of `kept=…`.
- The stuck-chain line no longer repeats `(burn=…)`; the `← INSIDE the retained sample` flag stays.
- The `Ĵ from …; cross-check …` line is removed.
- After the table, `estimate = mean of the pooled post-burn-in draws; se(chain) = their SD` and
  `best visited point: Q … (seed …) → chain_….jls` are replaced by one line,
  `Q(θ̄)=…   Q(θ̂)=…   Q(θ_best)=…`: Q at the estimate, at the seed and at the best point the
  chain visited. The estimate written to `estimate_{window}{W}_mcmc.jls` is θ̄, as before;
  the best visited point is still saved in the chain file as `theta_best`, and written to its own
  file only under `MCMC_CHECKPOINT = true` (default false).

### Gate
- A scratch chain on base_covid (`MCMC_N = 8`, `MCMC_GENS = 40`, 2 threads, same seed) under
  v58.2.0 and v58.2.1: `mcmc_results_base_covid_diagonalW_mcmc.csv` byte-identical; the console
  differs only in the lines above.
- `check_version_sync`: in sync at 58.2.1.

### Settings
None added, retired or re-defaulted.

### Known gaps
- `jgap` (the curvature cross-check) is still computed, and no longer printed.

## v58.2.0 — 2026-10-17

**MINOR: R̂ and ESS for the vacancy costs in `code/mcmc/MCMC_main.jl`.** No estimate, standard
error or quantile changes; the k rows of the chain CSV gain their `rhat` and `ess`, which were NaN.

**Comparability.** Full. Every estimate and standard error under v58.1.2 is reproduced.

### What changed (`code/mcmc/MCMC_main.jl`)
- k_U and k_S are not coordinates of the chain: free entry gives them at each kept draw
  (`k_draws`, in the column order of `res.draws = reshape(chain[:, :, burn+1:end], d, :)`).
  After the coordinates' `split_rhat_ess`, `k_draws` is laid back out as chain × generation and
  passed to the same function, giving `rhat_k`, `ess_k` on the same kept generations.
- The table's k rows print R̂ with the ✓ at `MCMC_RHAT_MAX`, like every other row.
- `mcmc_results_{window}{W}_mcmc.csv`: the k rows' `rhat` and `ess` carry `rhat_k`, `ess_k`.
- The `R̂ max … ESS min …` line, the chain bundle's `mcmc.rhat_max` / `mcmc.ess_min` and the
  abort note take the extremes over the 14 coordinates and the two costs (`rhat_all`, `ess_all`).
- The stopping test during the run (`converged_sequential`, off by default with
  `MCMC_CHECK_EVERY = 0`) still checks the coordinates only: the costs are computed after the run.

### Gate
- A scratch chain on base_covid (`MCMC_N = 8`, `MCMC_GENS = 40`, 2 threads) under v58.1.2 and
  v58.2.0, same seed: the CSVs differ only in the k rows' `rhat` and `ess` (NaN → 2.156198, 64.0
  and 2.016435, 64.0); the console only in the two k rows' new R̂ cell.
- The in-run `rhat_k`, `ess_k` equal an independent `split_rhat_ess` on the chain file's
  `k_draws`, reshaped as above, to every printed digit.
- `check_version_sync`: in sync at 58.2.0.

### Stored output updated
- The three finished chains' CSVs, `mcmc_results_{base_fc, crisis_fc, base_covid}_diagonalW_mcmc.csv`,
  had their k rows' `rhat` and `ess` filled from the saved chain files (modification times kept):
  base_fc k_U 1.070684 / 1142.8, k_S 1.059069 / 1269.0; crisis_fc k_U 1.046683 / 1612.0,
  k_S 1.039740 / 1493.4; base_covid k_U 1.049817 / 1255.5, k_S 1.053233 / 1483.6.
- Their `_mcmc` bundles were not rewritten. `mcmc.rhat_max` is unchanged in all three (the
  costs' R̂ is below the coordinates' maximum). `mcmc.ess_min` would be 1255.5 for base_covid
  under v58.2.0 (stored 1277.7, from the coordinates); base_fc and crisis_fc are unchanged.

### Settings
None added, retired or re-defaulted.

### Known gaps
None found.

## v58.1.2 — 2026-10-15

**PATCH: the vacancy-cost label in the estimate CSVs.** Only a text field changed; every
number in every output file is bit-identical to v58.1.1's.

**Comparability.** Full. No stored output is superseded.

### What changed
- `code/smm/smm_params.jl:704-705`, `VACANCY_COSTS`: the labels "unskilled vacancy cost k_U
  (months of new-match U output)" and "skilled vacancy cost k_S (months of new-match S output)"
  are now "unskilled vacancy cost k_U" and "skilled vacancy cost k_S". The value is unchanged:
  k_j/π̄_j, whose units are stated in Data and Moments.
- The label is read only by the two CSV writers (`smm/smm.jl:2852`, `mcmc/MCMC_main.jl:1639`)
  and by the parameter tables (`plots_and_tables/transition.jl:805`), which print
  `PARAM_DISPLAY`'s names, not the label. No code reads a CSV's label column.
- The 12 CSVs in `output/estimates/` that carried the old text were edited in place: the two k
  rows of each lost the parenthesis and nothing else, with modification times kept. They are
  `smm_estimates_{window}_diagonalW.csv` and `smm_estimates_{window}_diagonalW_smm.csv` for the
  four windows, and `mcmc_results_{window}_diagonalW.csv` for the four windows. No bundle (.jls)
  in `output/` stored the label, so none was touched.

### Gate
- `save_results` under v58.1.2, run in scratch on each of the four `_smm` bundles, writes a CSV
  byte-identical to the edited `smm_estimates_{window}_diagonalW_smm.csv`. Q reproduces at all
  four (base_fc 4192.6013105560, crisis_fc 2630.7827858309, base_covid 3657.0470277513,
  crisis_covid 1876.5391298933).
- `check_version_sync`: in sync at 58.1.2.

### Settings
None added, retired or re-defaulted.

### Known gaps
- The base_fc DE-MC chain running at the time of this change loaded v58.1.0, so its
  `mcmc_results_base_fc_diagonalW_mcmc.csv` will carry the old label when it finishes. Its
  bundles store no label. That one CSV is to be edited the same way after the run.

## v58.1.1 — 2026-10-15

**PATCH: one comment block in `code/mcmc/MCMC_main.jl`.** No code line changed, so every
output is bit-identical to v58.1.0's.

**Comparability.** Full. No stored output is superseded.

### What changed (`code/mcmc/MCMC_main.jl`, header, lines 4–18)
- The target was written as `logπ(θ) = −½ g′Wg + log|dθ/dt|`, the `:flat_theta` form that was
  the default only through v19.3.0. It now reads `logπ(t) = −½ g(θ(t))′W g(θ(t))`, a flat prior
  on t over |t| ≤ `MCMC_T_BOX`, with the logistic map θ = lb + (ub − lb)/(1 + e^(−t)), which is
  what the code samples under the default `MCMC_PRIOR = :flat_t`. The comment names
  `:flat_theta` as the alternative and points to the comment at `MCMC_PRIOR`.
- "The log-Jacobian is what makes the target proper" became "The box on t is what makes the
  target proper": under `:flat_t` the target has no Jacobian term.
- The paths `smm/MCMC_main.jl` and `smm/demc.jl` became `mcmc/MCMC_main.jl` and `mcmc/demc.jl`.

### Gate
- The file with every comment line removed is byte-identical to v58.1.0's.
- `Meta.parseall`: 0 error or incomplete nodes.
- `check_version_sync`: in sync at 58.1.1; all three readers include `version.jl`.
- No struct, setting or output path changed, so no bundle, CSV or run is affected. A run
  started under v58.1.0 keeps its 58.1.0 stamp.

### Settings
None added, retired or re-defaulted.

### Known gaps
None found.

## v58.1.0 — 2026-10-13

**MINOR: a reader accepts an estimate written by another MAJOR version when its stored Q still
reproduces under this code, and four console lines of `smm_main.jl` go.** Every number is
unchanged: an `smm_main.jl` run reproduces v58.0.0's CSV byte for byte and its bundle in every
field but `provenance.written_at`.

**Comparability.** Interchangeable with v58.0.0. The four `_smm` estimates written by v57.1.0
are now read as estimates by every reader: each one's Q reproduces under v58.1.0, to 2.2e-16
relative at most. No file is rewritten, and each keeps the provenance it was written with.

### What changed
- **The reader's gate** (`smm/bundle.jl`:226–261). `require_current_major` is replaced by
  `require_current_estimate(b, path)` (:241). A bundle written by this MAJOR version passes as
  before. One written by another MAJOR version has its stored parameter vector re-evaluated with
  its own spec (`equilibrium_at`). It passes when that Q is finite and within
  `REPRODUCE_RTOL = 1e-10` (:228) of the stored `loss_opt`, and the reader prints one line:
  `<file>: written by v<old>, Q reproduces under v<new> (rel <x>); accepted`. Otherwise it stops
  with the old message plus both values of Q. The function returns the evaluation it made
  (`nothing` when none was needed). No reader reuses it yet: the MCMC seed re-solves under its
  own rebuilt spec and the transition under its own grid and SimParams. The premise it replaces,
  "a MAJOR bump moves Q at every point", is false for v58.0.0.
- **Callers** renamed: `smm/smm_main.jl`:521 (crisis baseline), `mcmc/MCMC_main.jl`:650
  (seed), `transition/transition_simulation.jl`:75 (`point_estimate`),
  `plots_and_tables/transition.jl`:170 (`load_all_smm_bundles!`). Their comments say "an
  estimate under this code" (`smm_main.jl`:504–505, `transition_simulation.jl`:65–66). README
  §estimates and §transition say the same.
- **Console lines removed** (v58.0.0 line numbers):
  - `smm/smm_main.jl`:562–563: "Ability distribution FIXED at the baseline estimate: a_ℓ=… b_ℓ=…
    ρ_x=…";
  - `smm/smm_main.jl`:564: "Every other parameter is a candidate for the free set";
  - `smm/smm_main.jl`:982: "INIT_MODE = :warmstart — candidates, Q under v…";
  - `smm/smm_main.jl`:1002: one line per candidate, carrying the status built at :985–1000
    ("absent" :987, "unreadable" :989, "Q = …" with the DEFAULT_PARAMS/clamped notes, "unusable:
    …" :999, and the base point's "(base point)" note :1017);
  - `smm/smm_main.jl`:1004: "start: …";
  - `smm/smm_params.jl`:558–559: "SMMSpec: skip_moments zeroed weights for (n): …".

  `warm_start` (`smm_main.jl`:978) now prints one line, "  warm start: <file> (Q = <value>)"
  (:991), or, when no candidate has a finite Q, "  warm start: no candidate with a finite Q —
  the run starts from DEFAULT_PARAMS" (:992). Candidates are `(path, estimated_only)` pairs.
  The choice of start is unchanged.

### Measured, not asserted (scratch trees on copies of the four `_smm` bundles, 2 threads)
- **The four bundles in every reader**: the gate itself, `point_estimate`,
  `load_all_smm_bundles!` (on scratch transition records naming the bundles) and `MCMC_main.jl`
  up to its seed line (base_covid; stopped once the chain began). Each prints the accepted line:
  base_fc rel 2.2e-16 (Q 4192.6013105560), crisis_fc 0 (2630.7827858309), base_covid 0
  (3657.0470277513), crisis_covid 0 (1876.5391298933).
- **A point that does not reproduce stops.** The first unconstrained coordinate of a scratch
  copy of the base_covid bundle was raised by 1e-6. The gate and `point_estimate` both stop with
  "… Q stored 3657.0470277513, Q under v58.1.0 3657.0470281352." (rel 1.05e-10). The scratch file
  was restored byte for byte.
- **crisis_covid `smm_main.jl`**, ROYSEARCH_SA_MAX_ITER=2, SA_MAX_REHEATS=0, SMM_POLISH=false.
  It passes the gate (base_covid accepted, rel 0), finishes SA (Q_best 1876.539, 2 iterations)
  and writes its CSV and bundle. The same run with v58.0.0's prints over the v58.1.0 gate gives
  a log that differs only in the removed lines, the one warm-start line, paths and timings. The
  CSV is byte-identical, and the bundle matches in every field (result, spec, sim, se, mcmc,
  vacancy_costs, attracting, tag) but `provenance.written_at`.
- **Gates.** The 7 changed files parse; `check_forwarding` only the known
  `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` row; `check_dead_settings` PASS; `check_structure` the same 7
  findings; `check_threading` bitwise equal to serial (Q 4192.6013105560); `check_output_tree`
  PASS; `check_version_sync` 58.1.0.

### Settings
None added, retired or re-defaulted.

### Known gaps
- The held-out moments are still printed: by `print_spec` in `smm_main.jl` ("Skipped moments
  (n, weight = 0)") and by `load_weight_matrix` ("skipping: …"). `MCMC_main.jl` does not call
  `print_spec`, so there they appear only in the `load_weight_matrix` line.
- A warm-start candidate that is unreadable or unusable is now passed over without a line. The
  start is still the lowest finite Q.
- The re-evaluation costs one solve per reader per file (about 0.15 s each).

## v58.0.0 — 2026-10-13

**MAJOR: the stationary solver no longer imposes that no trained worker searches the unskilled
market; the equilibrium decides it.** U_S^(1), the training value T and the drain d are now
functions of the whole type (a_U, a_S), a trained hire into the unskilled market is an unskilled
worker whose outside option is U_U = max{U^search, −c(a_S) + T}, her hires obey the hiring
margin, and m_S, the unskilled masses, free entry and every unskilled moment carry her. The
theorist must be told: the model's equilibrium set changes (old Proposition 1 is no longer
imposed). At every stored estimate no trained worker gains by crossing, so the estimates and Q
reproduce to rounding.

**Comparability.** Every estimate, bank, chain and report CSV written under v57.1.0 stays
valid: Q at the four untagged estimates and at the live `base_fc` `_smm` checkpoint equals
v57.1.0's to 3.2e-16 relative, every moment to 4.3e-16. Nothing in `output/` was rebuilt; no
serialised struct changed. A transition solved under v57.1.0 is superseded where the training
frontier moves: the forward pass's rule for keeping the p = 1 atom of the training slice
changed (below), and at Nx = 60 on the covid pair it moves the path in the fifth significant
digit (θ_S at the horizon 2.343477 → 2.343493). Self-transitions are unchanged.

### What changed
- **Training value and choice on the whole type** (`solver/unskilled.jl`). `uc.T` is a
  matrix (`solver/params.jl`:166–171, built at :470); `training_frontier!` (:142) prices
  T(a_U,a_S) on U_S = max{U_S^(0), U_S^(1)}; `training_value!` (:180) and
  `marginal_training_cells` (:204) read it per cell. `training_policy!` (:164) takes the value
  of searching per cell (`searching_value` :189) and scores each cell by `covered_fraction!`
  (`solver/grids.jl`:416), the half-cell rule `ojs_participation!` already used, which locates
  every sign change of Utr − U^search along a row; where Utr is increasing it reproduces the
  old single-crossing interpolation to rounding. The inner loop sets T once per call from its
  fixed input and enters its step in the first pass's residual only (:265), which is what the
  old per-pass recomputation did.
- **Career choices of the trained** (`solver/unskilled.jl`:409–697). `career_choices!` (:629),
  called once the skilled block has settled U_S^(0), solves per cell U_S^(1) and T jointly
  (`crossing_fixed_point` :607, a contraction of modulus below φ/(r+φ+ν)), then τ, the drain d
  and the training-slice match. `crossing_value` (:578): (r+ν+κ_X) U_S^(1) = b_S + κ_X E_U with
  κ_X = κ_U χ_X; where training beats searching at the node the match is the training slice's,
  valued at Utr (`training_slice_match` :566, `unskilled_match_at` :471, an O(1) tail integral
  on suffix sums :433/:446), otherwise the row's. The hired share χ_X of a crosser's match is
  `crosser_hire_share` (:521) on the row's entry-surplus profile through its exact zero at a^h
  (`hire_profile` :497): equal to χ(a_U) where the match is the row's, continuous otherwise.
  `implied_training_value` (:661) is R5's re-evaluation (`solver/solver.jl`:167).
- **Drain on the whole type** (`solver/skilled.jl`:186–227). `drain_fraction!` takes U_S^(1)
  and U_S^(0) as matrices and scores cells by `covered_fraction!`; `skilled_inner_loop!` (:231)
  and `solve_skilled_block!` (:708) no longer take κ_U or E_U(·,1) and no longer form U1 or d.
- **Densities** (`solver/unskilled.jl`:680–806). The crossing hire rate of a cell is
  k = κ_U d [(1−ω) χ + ω χ_t], ω = max(d+τ−1,0)/d the share the training frontier encloses
  (`crossing_rates` :680, `enclosed_crossers` :429). `stationary_masses` (:749): m_S = φ t/(ν+k)
  (was φ t/(ν + d κ_U), which ignored the hiring margin); the training slice's employment
  e_t = kt m_S/(δ_t+ν) and its separations re-enter training, (φ+ν) t = ν τ ℓ + δ_t e_t; the
  searching slice's separations join the row's hired unemployed. Cells no trained worker
  leaves keep the old closed forms bit for bit. `_mS_from_t` (`solver/solver.jl`:89) takes the
  rate matrix. The unemployed hired branch is a cache field `uc.uh`.
- **Free entry** (`solver/unskilled.jl`:864–889). J̄_U weights a draining seeker by her own
  hired share and match: the row's J_U for the searching slice, (1−β_U) S_t(·,1)⁺ for the
  training slice.
- **Solve order** (`solver/solver.jl`). `solve_at_tightness!` (:266): unskilled block →
  skilled block → `career_choices!` (:281) → `install_stationary_masses!` (:283) → vacancy
  costs → battery. The global loop calls the career choices before and after the skilled block
  (:347, :363); the skilled block now reads d as a lagged link, like m_S.
- **Moments** (`solver/equilibrium.jl`). The training slice `eUt` enters the unskilled wage
  points at its own reservation quality and outside option Utr (:269), separations and
  within-job re-pricing (:384, :399, measured :487–493); the searching slice's hiring atom adds
  the searching-slice crossers (:140); jfr_U and the unskilled survivor read the row's hired
  share from `uc.uh` (:143). Skilled job finding, f_S, jfr_S and the skilled survivor give a
  crosser her cell's hired share (:365, :487, :510). `net_T` and `Tv` are matrices (:163); the
  readers take the diagonal (`plots_and_tables/model.jl`:82, `plots_and_tables/transition.jl`:907)
  or the cell (`tools/check_tau_margin.jl`:113, `tools/transition_audit.jl`:124).
- **Transition** (A2: the same rules at every date and at both endpoints). The terminal date is
  seeded with the stationary U_S^(1), T and τ (`transition/transition_values.jl`:384–392); τ
  and d are `training_policy!` and `drain_fraction!` at every date (:532,
  `transition/transition_solver.jl`:540), so `_training_fraction!` and `_crossing_fraction!` are
  deleted; the date step's κ_X is κ_U times the hired share of the slice the crosser joins
  (`transition_values.jl`:723), the training slice's share χ_Ut at every date
  (`_training_slice_hire_share!` :194, a new path field `transition_params.jl`:135). Forward:
  crossers' hires split by slice at each slice's share (`transition_solver.jl`:370) and leave
  m_S at κ_U h (:433, `_crosser_hire_shares` :179); z₀'s training slice is a pre-switch state
  (`transition_params.jl`:171–174, `transition_solver.jl`:96–109, :245); the p = 1 atom of the
  training slice is kept in the share its hired share still covers (:274–275), replacing
  "kept while S_t(1) ≥ 0 at the node, ended whole otherwise". Free entry (:580) and the path
  statistics (:724, :845) use the same shares.
- **Removed** every comment citing prop:ss-crossflow (`transition_params.jl`:275,
  `tools/transition_audit.jl`:11) and the seed's "where Proposition 1 holds". There was no C3
  field to remove. No console line added.

### Measured, not asserted (scratch tree, 2 threads, against an unmodified v57.1.0 copy)
- **T1, the stored estimates.** `equilibrium_at` at each point, v57.1.0 → v58.0.0:

  | point | Q | rel ΔQ | max rel Δ moment | max \|Δτ\| | max \|Δd\| | max [U_S^(1)−U_S^(0)]⁺ on τ=1 | Σ d m_S |
  |---|---|---|---|---|---|---|---|
  | base_fc | 5609.8086833222 | 3.2e-16 | 4.3e-16 | 5.6e-17 | 2.2e-15 | 0 | 0 |
  | crisis_fc | 3419.6247679487 | 1.3e-16 | 1.7e-16 | 5.6e-17 | 5.6e-17 | 0 | 0 |
  | base_covid | 4927.6919718308 | 0 | 0 | 5.6e-17 | 5.6e-17 | 0 | 0 |
  | crisis_covid | 2245.9994946964 | 2.0e-16 | 3.5e-16 | 2.6e-14 | 5.6e-17 | 0 | 0 |
  | base_fc `_smm` | 4192.6013105560 | 2.2e-16 | 2.2e-16 | 0 | 5.6e-17 | 0 | 0 |

  Battery Rmax 0.107 at every point under both versions.
- **T2, a crossing equilibrium**: base_covid at θ_U × 0.002 (κ_U 0.01277 < φ 0.0208) and
  ln c − 6, θ_S at the estimate. Battery R1 6.7e-3, R2 3.6e-10, R3 0, R5 6.7e-10. 504 cells with
  τ = 1, d = 1 and t > 0 (1.65% of the population); Σ d m_S 3.087e-3, Σ e_t 1.171e-2;
  (U_S^(1) − U_S^(0))/U_S^(0) ∈ [4.6e-3, 0.127] over them. Per-cell flow balances (t, m_S,
  e_t, u_U, searching-slice employment) ≤ 8.7e-17 of the cell's mass; over the crossers T,
  U_S^(1) and E_U satisfy their equations to 9.5e-17, and S_t(·,1) re-solved on the p-grid at
  its own outside option agrees to 1.2e-14, p*_t exactly. Refining to Nx 200 / Np_U 200 /
  Np_S 220: 803 crossing cells, population share +1.99%, Σ d m_S +0.66%, Σ e_t +0.87%, largest
  moment change 3.0e-3 (the same refinement moves the estimate's moments by 3.8e-3).
- **T3, cost.** One evaluation at base_fc, median of 5, three alternating runs each:
  v57.1.0 0.1412 / 0.1400 / 0.1413 s, v58.0.0 0.1393 / 0.1416 / 0.1414 s.
- **T4, self-transitions.** `transition_audit.jl` with TA_NULL = true and each window at both
  ends (base_fc, crisis_fc, base_covid, crisis_covid): every printed line identical to
  v57.1.0's; converged at iteration 1, terminal distribution gaps 2.3e-9 to 4.6e-9.
- **A real transition**, covid pair at Nx = 60, Nt = 121: converges in 21 iterations as before;
  restoring the old atom rule alone reproduces v57.1.0's log line for line, so that rule is the
  whole difference.
- **Gates.** All 13 changed files parse; `check_threading` (2 threads) bitwise equal to serial,
  Q 4192.6013105560; `check_hire_margin` OK; `check_forwarding` only the known
  `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` row; `check_dead_settings` PASS; `check_structure` the same 7
  findings; `check_output_tree` PASS; `check_pair_map` OK; `smoke_transition.jl` PASSED (109 s).

### Settings
None added, retired or re-defaulted.

### Known gaps
- **The transition does not hold a crossing steady state.** At the T2 point (Nx = 120) the
  terminal seed reproduces the stationary U_S^(1), T, τ and d exactly and the training slice to
  the transition's tolerance (p*_t 8.3e-6, χ_t 1.1e-10), but a self-transition does not converge
  (free-entry gap 2.0e-3 after 150 iterations, θ_S drifting 5.2% in logs, e_t 1.3e-3 of its
  total). Cause, measured: the date step's covered fractions read value differences on the
  one-date scale. A negative S_t(·,1) has continuation 0, so at a steady state it is the
  stationary value times D/(D+1/dt) (0.104151 measured, = D/(D+1/dt) to six digits), and χ_t
  differs by up to 0.45. On the drain margin a crossing node's U_S^(1) − U_S^(0) is scaled by
  (r+ν)/(r+ν+1/dt) and a non-crossing node's by (r+ν+κ_X)/(r+ν+κ_X+1/dt) (1.00 vs 238.53 and
  −1.22 vs −121.14 at the worst cell), which moves the interpolated zero (d 0 vs 0.161). Neither
  binds at any stored estimate, where no trained worker crosses. Not fixed: the remedy (express
  both gaps on the stationary scale before interpolating, exact at a steady state) is the
  author's call.
- The transition still counts crossers as d u_S where the stationary solve counts d m_S; they
  agree wherever d ∈ {0, 1} (the T2 path's crossing flow 3.80e-5–3.87e-5 against 3.89e-5).
- Crossers' unskilled matches enter the unskilled moments, as instructed; the data assign a
  degree holder to the skilled side, so this is a choice the moment definitions must confirm.
- The fc pair's coarse transition could not be run in the scratch tree: its `crisis_fc` copy
  holds ability parameters from an older base_fc than the live `_smm` checkpoint, so the grids
  differ (same error under v57.1.0).

### Tried and rejected
- **The row-wide monotone test for τ and d.** Kept from v57.1.0 with U_S^(1) now varying in
  a_S inside the training set, it tripped on rows whose gap is non-monotone far from the
  crossing and switched the whole row to the 0/1 indicator: d moved by 0.49 in cells holding no
  mass at base_fc, crisis_fc and base_fc `_smm`. Replaced by the half-cell rule.
- **U_S^(1) as a value mixture** χ (b_S+κ_U E)/(r+ν+κ_U) + (1−χ) b_S/(r+ν). It does not
  survive the transition's date step (a mixture of values is not a value under mixed
  discounting), so a steady state would not be a fixed point; the rate mixture κ_X = κ_U χ_X is,
  and matches m_S's outflow.
- **T inside every inner-loop pass**, as v57.1.0 did. With T a matrix (frontier and supnorm on
  25 600 cells per pass) an evaluation at base_fc took 0.50–0.55 s against 0.36 s with T set
  once, on a machine loaded by another run; the result is the same.

## v57.1.0 — 2026-10-13

**MINOR: a transition's files are named by the point they were solved at, and its horizon test
holds the mean log wages to 1e-3 in log units.** The SMM and MCMC code is unchanged: no file
that `smm_main.jl` or `MCMC_main.jl` loads changes except `version.jl`, so every estimate and
every Q reproduces bit for bit. The transition reproduces v57.0.0 at both points on the covid
pair: the same accepted horizon, every printed line of the solve, and all ten tables byte for
byte; only the files' names and the horizon test's recorded criterion differ.

**Comparability.** Estimates written under v57.0.0 and v57.1.0 are interchangeable. A transition
file written under v57.0.0 (`transition_<pair>_diagonalW.jls`, untagged) is read by nothing:
re-run the transition. `output/transition/transition_fc_fullW.jld2` is untouched.

### What changed
- **The transition file** (`transition/transition_params.jl`:281–305). A run is tagged
  `<W>_<point>` (`transition_tag` :288), and every file it writes ends in the tag:
  `transition_file(scenario, w_suffix, point)` (:290) names
  `output/transition/transition_<pair><W>_<point>.jls`, and the tables and plots made from it
  carry the same ending. `read_transition` (:299) returns the record, `nothing` when there is
  none, and stops on a file that records another point. Every writer and reader of a
  transition file loads `transition_params.jl`; the five places that built the name before
  (`transition_main.jl`, `transition_simulation.jl`, two in `transition_panel.jl`,
  `plots_and_tables/transition.jl`) call these instead.
- **Writer** (`transition/transition_simulation.jl`:228): `run_transition_simulation` saves to
  `transition_file`.
- **Orchestrator** (`transition/transition_main.jl`). The re-use check names the tagged file
  (:180); the panels take `point = POINT_ESTIMATE` (:207); the tables phase loads the bundles of
  this point's transition files (:221) and passes `suffix = transition_tag(W_SUFFIX,
  POINT_ESTIMATE)` to every table and to the scatter (:222).
- **Panels** (`plots_and_tables/transition_panel.jl`). `make_transition_panel` and
  `make_model_decomposition_panel` take `point` (:758, :880), read through `read_transition`
  (:772, :887) and save `transition_panel_<pair><W>_<point>.png` (:847) and
  `model_transition_<pair><W>_<point>.png` (:1080). Run standalone, the file draws every
  pair at both points it finds (:1100).
- **Tables** (`plots_and_tables/transition.jl`). `load_all_smm_bundles!(; scenarios, point)`
  (:154) reads through `read_transition` (:158), so the tables phase stops on a transition file
  that records another point. The table writers and the scatter are unchanged: their `suffix`
  names the file, and the orchestrator now passes the tag. Run standalone, the file loads
  `transition_params.jl` (:98), which it needs to read a transition file, and makes the tables
  of each point that has transition files (:1086).
- **Horizon test** (`transition/transition_simulation.jl`:25–39, :165). `LOG_SERIES =
  (:mean_wage_U, :mean_wage_S)` are compared by their absolute change on months 0–36, every
  other reported series by its change relative to its own scale, both against `HORIZON_TOL =
  1e-3`. Since v57.0.0 the path's mean wages are mean log wages, whose change is relative
  already; dividing it by the log's own size (about 6.5) had made the test that many times
  looser on them. 1e-3 in log units is the strictness of the test on level wages to first
  order. The record's `change_first_months` is the new criterion.
- **Docs.** README: the two paths' transition steps name the files they write; the transition
  section gives the tagged names, the tables phase's refusal and the horizon rule. SETTINGS.md:
  `POINT_ESTIMATE` tags every file of a transition run.
- **Not changed:** every file `smm_main.jl` and `MCMC_main.jl` load (`paths.jl`, `solver/`,
  `smm/`, `mcmc/`) except `version.jl`; the x̄ table (its function is untouched, and its file
  name follows the tag through its `suffix` argument like every table's); `.gitignore` (the
  proposed `/output/transition/transition_*.jls` matches the new names too).

### Measured, not asserted (scratch tree, v57 covid estimates, 2 threads, one run at a time)
- **The transition at both points**, on copies of the v57.0.0 scratch estimates of the covid
  pair (`using Plots` removed and a stand-in that records each `savefig` path, since GR does
  not build in the sandbox):

  | | `:smm` | `:mcmc` |
  |---|---|---|
  | accepted horizon, new rule | 240 months | 240 months |
  | iterations at horizon 120 / 240 | 26 / 14 | 25 / 14 |
  | largest change on months 0–36 at 240 | 3.0493e-4 | 3.0549e-4 |
  | transition file, bytes | 1,586,038,860 | 1,586,038,637 |
  | wall time | 2206 s | 2161 s |

  The largest change equals v57.0.0's at both points, so a series not in logs sets it, and the
  two mean log wages moved by at most that, 3.05e-4 in log units against 1e-3. Every printed
  line of the two solves (14 each: iterations, gaps, the horizon results, θ_U and θ_S at the
  endpoints and on the path, the crossing reversals) equals v57.0.0's.
- **The two runs coexist.** At the end the scratch `output/` held
  `transition_covid_diagonalW_smm.jls`, `transition_covid_diagonalW_mcmc.jls` and both sets of
  five tables (`model_fit_A`, `model_fit_B`, `parameter_estimates_A`, `parameter_estimates_B`,
  `xbar_table`, each `_diagonalW_smm.tex` and `_diagonalW_mcmc.tex`), every table byte-identical to
  the one v57.0.0 wrote at the same point. The plots went to
  `transition_panel_covid_diagonalW_<point>.png`, `model_transition_covid_diagonalW_<point>.png`
  and `model_fit_scatter_base_covid_diagonalW_<point>.png`; the panels ran on a second pass that
  re-used each transition file (`RERUN_SIMULATION = false`).
- **Refusal.** A file named `transition_covid_diagonalW_mcmc.jls` that records `point = :smm`
  stops the tables phase under `POINT_ESTIMATE = :mcmc` in one line
  ("transition_covid_diagonalW_mcmc.jls records point = :smm, not :mcmc."), and nothing is
  written.
- **Gates.** The six changed files parse; `check_forwarding` reports only the known
  `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` row; `check_dead_settings` PASS; `check_structure` the same 7
  findings; `check_output_tree` PASS on the scratch output holding both runs; `check_pair_map`
  OK; `smoke_transition.jl` on base_covid PASSED (100 s, nothing written); `check_version_sync`
  in sync at 57.1.0. The runs preceded the bump, so the two files record version 57.0.0.

### Settings
None added, retired or re-defaulted. `POINT_ESTIMATE` (`transition_main.jl`) now also names a
transition run's files.

### Known gaps
- **`transition_tag` and `transition_file` belong in `paths.jl`**, beside `out_transition()`,
  where every other output path is built. They are in `transition_params.jl` because this
  version leaves every file the SMM and MCMC drivers load untouched; moving them changes no
  output.
- **Plots does not build in the sandbox** (GR_jll), so no panel or scatter was drawn: the
  panels and the scatter ran with a stand-in whose `savefig` records the path it is given.
- **The fc pair's horizon under the new rule is not measured**; only the covid pair ran.
- **The x̄ table is still blank** in every cell; it is with the author.

### Tried and rejected
- **A subdirectory per point** (`output/tables/smm/`, `output/plots/smm/`). The suffix is the
  convention the estimate files and the transition file already use (`…W_<point>.jls`), and it
  keeps the two runs' files side by side in one listing; a subdirectory would be a second
  convention for the same tag.

---

## v57.0.0 — 2026-10-13

**MAJOR: every wage moment is a different number.** The unskilled wage distribution now
counts each employed worker once: the shipped code weighted matches below p = 1 by their
density in p rather than by their mass, so the hiring atom carried 0.065% of the weight while
holding 11.2% of unskilled employment (base_fc). Mean, variance and third moment, the ten
percentiles and both overlaps are now computed from the wage points themselves rather than
from a 120-bin level histogram, and the top quality node gets a covered-fraction cell. Q moves
at every estimate (base_fc 4377.33 → 5609.81).

**And every estimate has a file of its own.** `smm_main.jl` writes `estimate_<w>W_smm.jls`
(or `_smm_mcmcbase.jls`, a crisis window held at the base chain's estimate, chosen by the new
`CRISIS_BASE`), a chain run of `MCMC_main.jl` writes `estimate_<w>W_mcmc.jls`, and neither
writes the other's; every reader names the file it reads, and refuses one written by another
MAJOR version.

**Comparability.** No wage moment, and no Q, from v56.0.4 or earlier can be compared with
this version. Every stored estimate (`estimate_<w>_diagonalW.jls` for the four windows, and
the ξ_S backup) remains loadable, because no struct changed and the moment layer reads
only freshly computed objects. They are no longer optima of the objective, so every window
needs re-estimation (the Gauss–Newton steps below). The candidate banks and the base_fc chain
are superseded for the same reason. Non-wage moments change only through the top-cell rule,
at points where a cutoff sits in [p_N, 1]. The untagged `estimate_<w>_diagonalW.jls` files
are read from now on only as warm-start candidates, which re-score them under this code;
`estimate_base_fc_diagonalW_postmean.jls`, the `_backup_Q` copies and the untagged candidate
banks are read by nothing. Transition files written before this version name untagged
bundles, which the tables phase refuses: re-run the transition.

### What changed — (A) the wage moments
- **Top quality cell** (`solver/grids.jl`:369, :381). `_soft_weight` and `_soft_oj_weight`
  give node N the cell [p_N, 1] (`p_next = 1` at j = Np, :372, :384), so the covered-fraction
  rule now applies at every node. A cutoff crossing p_N fades the top node out instead of
  switching it off. Every caller inherits the change: the stationary solve, the skilled block,
  the transition, and the wage surfaces.
- **Wage points** (`solver/equilibrium.jl`). The unskilled tail point carries
  `eU_density[i, jp] * wpU[jp]`, density × quadrature weight (:250); the hiring atom keeps its
  mass `eU_entry`. A new field `wage_points = (U = (w, m), S = (w, m))` (:280, :513) holds
  one (wage, mass) pair per employed cell.
- **Node wages** (`solver/equilibrium.jl`:190–223). Every wage formula is evaluated at every
  node whatever the node's mass (`wU_node`, `wU_entry`, `wS0_node`, `wS1_node`); the reported
  surfaces keep a wage only where the node has mass, as before. The 120-bin level histograms
  (`wmid`, `dens_U`, `dens_S`, `dens_S0`, `dens_S1`) take their edges from the 0.2% and 99.8%
  quantiles of this fixed-size set (:286), so they move continuously with the parameters.
  They serve figures only; no SMM moment reads them.
- **Moment layer** (`smm/moments.jl`:387–468, :518–534).
  - `_logwage_distribution(pts, σ_w)` computes the mean, the variance + σ_w² and the third
    central moment of log wages from the points.
  - `_conv_cdf_pdf` evaluates the σ_w-convolved CDF F(y) = Σ_k q_k Φ((y − log w_k)/σ_w)
    and its density.
  - `_conv_quantile` finds p10–p90 by Newton on F, starting from the normal approximation and
    falling back to bisection inside the bracket; it stops when the step falls below 1e-13 in
    log w.
  - The overlaps are 1 − F_U(p50_S) and F_S(p50_U). The premium is mean_S − mean_U.
  - Removed as dead: `_logwage_moments(wmid, dens, bw, σ_w)`, `_logwage_conv_cdf`,
    `_invert_cdf`, `_disc_pctile`. The σ_w → 0 branches are kept.
  - The `model_moments` docstring now sits on `model_moments`; it was attached to
    `_logwage_moments`.
- **The consumers outside the moment layer** (audit of `wage_fix_report.md`). An unskilled
  match below p = 1 counts with density × quadrature weight, a new match at p = 1 with its
  mass, and the top cell [p_N, 1] is part of the tail:
  - `solver/equilibrium.jl`:144: `eU_surface` is the density in p plus, in the top column,
    the new matches' mass, and says so; the returned object now carries `eU_density` (:499) so
    that consumers weight it explicitly.
  - `plots_and_tables/model.jl`:394 (`fig_skill_premium`): each market's mean over its own
    quality grid, the unskilled tail by density × wpU at every node and the new matches by
    their mass at the entry wage, the skilled cells by density × wpS. The trapezoid on the
    skilled grid, which dropped [p_N, 1] and treated the atom as a density at p_N, is gone.
  - `plots_and_tables/model.jl`:139 (`fig_total_employment`): employment by ability, each
    market's mass summed over its own grid. It added the unskilled surface (Np_U columns) to
    the skilled one (Np_S columns), a DimensionMismatch at the shipped grids.
  - `policy/policy_solver.jl`:43–61: `mean_wage_U` with the same weighting. `policy/` still
    does not load (known: it reads `obj.wx` and `obj.eS_mat`, which do not exist).
  - `transition/transition_solver.jl`:284, :730: the path's mean wages are mean LOG wages.
    They were level means plotted in `plots_and_tables/transition_panel.jl` against the data
    moments `mean_wage_U/S`, which are mean log wages; the panels now compare like with like.
    The points and masses are the path's own, as before (`TransitionResult` fields documented,
    `transition_params.jl`:242).
  - No change at A2, A5, A7, A8, A10 and A11.

### What changed — (B) one file per estimate
- **Files** (`code/paths.jl`). `estimate_path(window, W; point)` (:177) takes a required
  `point` ∈ `ESTIMATE_POINTS = (:smm, :smm_mcmcbase, :mcmc)` (:155), with no default, so no
  caller reads a file it did not name: `estimate_<w>W_smm.jls` (smm_main.jl; a crisis window
  with `CRISIS_BASE = :smm`), `estimate_<crisis>W_smm_mcmcbase.jls` (smm_main.jl, crisis window,
  `CRISIS_BASE = :mcmc`, the crisis chain's seed), `estimate_<w>W_mcmc.jls` (a chain run). Also
  `legacy_estimate_path` (:186, the untagged pre-v57 file, read only as a warm start),
  `mcmcbest_path` (:194), `candidates_path` (:203, the `:clusters` bank, tagged like its run's
  bundle), `smm_csv_path` / `mcmc_csv_path` (:209, :211), and `PAIR_BASE` / `is_crisis`
  (:144, :146) for every script but smm_main.jl, which keeps its own `_PAIR_BASELINE`.
- **Bundles** (`code/smm/bundle.jl`). `require_current_major(b, path)` (:233): every reader of an
  estimate stops in one line on a file written by another MAJOR version. `:mcmc_postmean`
  leaves `BUNDLE_STAGES` (:77); `:mcmc` and `:mcmc_aborted` now mark the chain's own file;
  `bundle_has_se` (:213) admits an `:smm` bundle the chain-free pass added errors to;
  `provenance_from_path` (:101) parses the tagged names.
- **SMM driver** (`code/smm/smm_main.jl`). `CRISIS_BASE = env_setting(:CRISIS_BASE, :smm)` (:169),
  directly below `WINDOW` and validated, then the unchanged `_PAIR_BASELINE` block (:175), then
  `OUT_POINT` (:183); `OUT_JLS` (:465) names the one file the run writes (checkpoints, CSV and
  final bundle). The crisis branch reads `estimate_path(base; point = CRISIS_BASE)` (:507) and,
  after the unchanged loader, refuses another MAJOR version and stage `:mcmc_aborted` (:521);
  nothing else in the crisis wiring changed. §8a (:963, `warm_start` :981, :1009) replaces the
  two warm-start blocks: every candidate is scored under the current code, one line each, and
  the lowest Q starts; a parameter its run did not estimate keeps `DEFAULT_PARAMS` and is
  listed. Candidates: base window — its `_smm` file, the untagged file; crisis window — its own
  output file, its other SMM file, the untagged file, the base point (read field by field, as
  before). A crisis window's ability distribution is the base point's whichever wins.
  `USE_AS_SEED = false` perturbs the chosen start (:1024). `:clusters` reads the tagged bank
  (:1043) and injects the base point through `point_in_spec` (:1065); `INCLUDE_PREV_OPTIMUM`
  reads `OUT_JLS` (:1091). `point_in_spec(b, spec; estimated_only)` (`smm.jl`:84) maps a stored
  point into a spec, bit for bit where free set and box agree.
- **MCMC** (`code/mcmc/`). Seed (`MCMC_main.jl`:641): `estimate_<w>W_smm.jls` on every route but a
  crisis chain, which reads `estimate_<crisis>W_smm_mcmcbase.jls` and stops unless its ability
  distribution equals that of `estimate_<base>W_mcmc.jls` bit for bit (:651); another MAJOR
  version stops (:650). It does not read `CRISIS_BASE`. A chain run writes
  `estimate_<w>W_mcmc.jls` (:1883): the posterior mean (the t-space mean where the reported one
  does not solve), stage `:mcmc` or `:mcmc_aborted`, `sim = spec.sim`, se (curvature, bound,
  chain), `se_source = :chain`, the vacancy costs at θ̄ with the delta-method errors, the
  attracting check at θ̄ (:1886, four solves, printed only when it fails), the diagnostics and
  `mcmc.seed`, the seed file's name. It never writes an SMM bundle. The chain-free pass adds
  `se` (with `se_source = :jacobian_<MCMC_JAC_METHOD>`), the costs' errors and the attracting
  check to the SMM bundle it read, only if that file still holds the point it started from
  (:1858). `MCMC_CHECKPOINT` governs only `estimate_<w>W_mcmcbest.jls` (`promote!` :922, stage
  `:mcmc_checkpoint`); the `_backup_Q` copy and the "Re-run smm_main.jl" advice are gone, and
  the end-of-run line says where the best visited point is (:1826). The chain snapshot to
  `output/chains/chain_<w>W.jls` every `MCMC_LMR_PRINT` generations no longer depends on it
  (`on_dump = snapshot_chain` :1098), so a run killed before the sampler returns keeps its
  draws. CSVs: `mcmc_results_<w>W_smm.csv` (chain-free) or `_mcmc.csv` (chain) (:1545).
  `demc.jl`:423 takes the seed's logπ at θ0 itself instead of chain 1's, which is wrong unless
  `MCMC_INIT` is `:at_seed` or `:widths`.
- **Transition** (`code/transition/`, `code/plots_and_tables/transition.jl`). `transition_main.jl`
  includes `bundle.jl` (:140) and `version.jl` (:105), whose absence stopped every run at the
  first `read_bundle` (UndefVarError) and at the save. `POINT_ESTIMATE` (:40) picks the `_smm`
  or the `_mcmc` file for both windows of a pair; `SCENARIO_BUNDLES` and its ξ_S backup path
  are gone (the backup stays on disk). `point_estimate(window, W, point)`
  (`transition_simulation.jl`:64) refuses another MAJOR version and `:mcmc_aborted`;
  `check_pair` is unchanged. The tables phase reads the two files each transition file
  records (`load_all_smm_bundles!(; scenarios)` :146, by name in the tree's `estimates/`); a
  bundle without `sim` is solved at its own `spec.sim` in memory (`_bundle_sim` :173), never
  written back. That removes the TypeError the tables phase hit (v56.0.4 `transition.jl`:178) on a
  bundle with `sim = nothing`.
- **Other readers** pass `point` explicitly: `POINT_ESTIMATE` (default `:smm`) in
  `solver/model_main.jl`, `policy/policy_main.jl` and `tools/` (`check_tau_margin`,
  `ojs_continuity_probe`, `plateau_probe`, `rank_diagnostic`, `transition_audit`, `width_audit`,
  where it replaces `ROYSEARCH_WIDTH_BASE`, now in `LEGACY_ENV_KEYS` with a pointer to the
  constant and checked at the tool's start, :59); `:smm` in `tools/test_generator.jl` and
  `smoke/smoke_transition.jl`. `tools/migrate_bundles.jl` carries every estimate file of a
  window, the untagged one included (:93); the smoke tests read the base point through
  `point_in_spec`.
- **Gates.** `gates/check_pair_map.jl` (new, run by `check_repo.sh`) reads smm_main.jl's
  `_PAIR_BASELINE` from its parsed source, without running the driver, and fails unless it names
  the same pairs as `PAIR_BASE`. `check_hire_margin.jl` checks every tagged estimate on disk
  (:68), `check_threading.jl` the first (:60); `check_output_tree.jl` lists `_postmean` as debris
  (:109); `check_repo.sh` lists the warm-start candidates. `check_forwarding.jl` check 4 reads
  the launchers in `code/ops/` (:288): it read `code/scripts/`, empty since the launchers moved,
  so it had checked nothing.
- **VM script** (`code/ops/vm_run.sh`). `ROYSEARCH_INIT_MODE=warmstart` with a one-line comment
  in place of `default` and a comment no longer true; the force-add of
  `output/tables/smm_estimates_*.csv`, a path that has not existed since v19.6.0, is gone
  (`git add -A output/estimates` covers every v57 name: `git check-ignore` matches none).
- **Docs.** README: "Estimate files, and the two estimation paths" (file table; each path step by
  step with the file each step reads and writes; `CRISIS_BASE`; `POINT_ESTIMATE`; the warm
  start); the wage points and the top cell under "Continuity of the criterion"; the transition
  and standard-error sections; a verification row for `check_pair_map.jl`. The copy-over
  warm-start advice is gone. SETTINGS.md: "Settings that choose a file", and graveyard entries
  for `SCENARIO_BUNDLES` and `ROYSEARCH_WIDTH_BASE`.

### Measured, not asserted (scratch trees, estimation grids, 2 threads)
- **Q at the stored points, and the gap to the exact-quantile reference:**

  | window | v56.0.4 | v57 | v57 − reference |
  |---|---|---|---|
  | base_fc | 4377.327871 | 5609.8087 | −1.7e-5 |
  | crisis_fc | 2730.395707 | 3419.6248 | −3.2e-5 |
  | base_covid | 3877.568377 | 4927.6920 | −2.8e-5 |
  | base_covid_ξS | 3935.549800 | 5249.7881 | −4.9e-5 |
  | crisis_covid | 2115.636010 | 2245.9995 | −5.3e-6 |

  Every gap is the reference's 4-decimal rounding. Q against an independent 60-step
  bisection on the same points agrees within 1.6e-11. On the integrated tree the warm start
  prints the same four values for the untagged files (5609.808683, 3419.624768, 4927.691972,
  2245.999495).
- **Percentiles and overlaps against exact evaluation:** within 3.8e-13 sampling SDs at all
  five points.
- **Wage-point masses** (scratch exact). Skilled: equal to employed mass to 3e-11. Unskilled
  at base_fc: aggregate +0.28% (points 0.625287 against rows 0.623526), tail +0.32%
  (quadrature 0.555136 against the identity 0.553375), largest row 0.79% (row 35). Unskilled
  at base_covid: +0.19%, +0.20%, 0.56% (row 28).
- **Continuity.** Bisected crossings, with max |ΔQ| at a crossing:
  - base_fc: u 43 crossings, 7.3e-9; w 91, 7.8e-10; b_U 231, 9.0e-10; Gauss–Newton step 92,
    2.8e-7; σ_S 306, 3.6e-6 (one non-wage crossing, see Known gaps).
  - A + P_S at the other four windows: 56 / 70 / 62 / 72 crossings, max 3.7e-8.
  - Before the change the same scans gave jumps up to 1.75 (base_fc) and 4.9e-2 (base_covid_ξS).
- **Gauss–Newton step at the stored points** (Ĝ adaptive-FD, se the J⁻¹ errors). ‖δ/se‖ is
  8.77 (base_fc), 18.96 (crisis_fc), 24.66 (base_covid), 11.72 (crisis_covid). At base_covid
  the linearisation holds to t ≈ 0.1 and the line minimum is Q 4258.29 at t = 0.447, 11.0 se
  from the stored point; the other three reach theirs at t ≈ 1.01–1.02.
- **Grid refinement at base_fc**, ΔQ against (160,160,180): Np_U 320 −18.41; Nx 240 −0.52;
  Np_S 360 +0.19. The largest wage-moment changes are p10_wage_U +0.154 SD (Np_U) and
  p25_wage_S +0.135 SD (Np_S).
- **Time per evaluation at base_fc:** 0.1325 → 0.1361 s at 2 threads and 0.2033 → 0.2069 s
  at 1 thread; the moment layer takes 5.3 → 14.9 ms.
- **`transition_audit` with `TA_NULL=true`:** fc and covid are bit-identical to v56.0.4 at
  `TA_NX` = 120 (terminal gap 2.23e-9 and 4.58e-9).
- **The two paths, end to end** (a scratch tree whose `output/` held copies of the four untagged
  estimates and the base_fc chain; SMM at `SA_MAX_ITER=2 SA_MAX_REHEATS=0 SMM_POLISH=false`,
  chains at `MCMC_N=8 MCMC_GENS=12`):
  - base_covid by SMM: the warm start lists `estimate_base_covid_diagonalW_smm.jls` absent and
    the untagged file at Q 4927.691972, and starts there; only the `_smm` bundle and its CSV
    are written. With `CRISIS_BASE=mcmc` from the same state: the same file, the CSV
    byte-identical, the parameter vector bitwise equal.
  - crisis_covid by SMM on the SMM base: four candidate lines (own and `_smm_mcmcbase` absent,
    the untagged file 2245.999495, the base point 14405.713516); writes the crisis `_smm`.
  - base_covid chain: writes `_mcmc` (Q(θ̄) 4898.152649; best visited 4875.519, seed 4927.692),
    its CSV and the chain file; the `_smm` file is byte-identical before and after.
  - base_covid chain-free pass: the `_smm` parameter vector, loss, stage, `sim`, fixed values,
    provenance and tag unchanged; `se` (curvature, bound, chain) and `se_source = :jacobian_fd`
    added; `mcmc_results_base_covid_diagonalW_smm.csv` written; 168 solves.
  - crisis_covid by SMM on the chain base: writes `_smm_mcmcbase`, its fixed ρ_x
    −0.4941375408702589, the base `_mcmc` value bit for bit (−0.49412406238821294 at the SMM
    base). The crisis chain seeds from it, passes the ability check and writes the crisis
    `_mcmc` (Q(θ̄) 2234.090955); given a seed made on the SMM base it stops in one line and
    writes nothing.
  - Refusals, each one line, nothing written: a v56 file under a v57 name (smm_main.jl on a
    crisis window, MCMC_main.jl, the transition), an aborted base chain, `CRISIS_BASE=foo`, a
    crisis chain without `_smm_mcmcbase`, an exported `ROYSEARCH_WIDTH_BASE`. A warm start with
    no candidate starts from `DEFAULT_PARAMS`; an unreadable one is listed as such.
  - The transition at both `POINT_ESTIMATE` values on the covid pair, in a copy without Plots:
    the endpoints solve at Q 4927.691972 and 2245.999495 (`:smm`) and 4898.152649 and
    2234.090955 (`:mcmc`). Both accept the 240-month horizon (`:smm` 26 + 14 iterations, change
    on months 0–36 3.049e-4, as before the fix; `:mcmc` 25 + 14, 3.055e-4) and write the
    transition file and the five tables from exactly the two recorded bundles; nothing in
    `estimates/` changes. At the base point the stationary mean log wages agree with the moment
    layer to 3.0e-4 (U) and 2e-6 (S).
- **The re-weighted figures** at the stored base_covid point (Plots replaced by stubs that
  record each series): employment by ability sums to the wage-point masses (unskilled
  0.553173422246, skilled 0.347208772837, both to 2.2e-16 relative), with the new matches
  7.25% of unskilled employment; the skill premium is finite for the 103 of 160 types employed
  in both markets, 0.2847 to 0.4491, and equals a direct evaluation from the wage points' own
  definitions to every printed digit at the median type.
- **Gates:** every changed file parses; `check_forwarding` reports only the known
  `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` row, and check 4, now reading `code/ops/`, finds nothing in
  the three launchers; `check_pair_map` OK; `check_threading -t 2` at base_fc Q =
  5609.8086833222, threaded bitwise equal to serial; `check_hire_margin` 0 disagreements on
  every tagged estimate; `check_dead_settings` PASS; `check_structure` the same 7 files as
  before; `check_output_tree` PASS on the scratch output and lists a `_postmean` file as
  debris; `migrate_bundles.jl verify` loads all ten estimate files; smoke `guard` and `clusters`
  PASS. `vm_run.sh` ran in scratch as far as its julia call, with git and shutdown stubbed and
  the call pointed at the spec-only driver: every export accepted, the warm start chose the
  untagged base_covid file.

### Settings
- **Added:** `CRISIS_BASE` (`smm_main.jl`:169, `:smm` | `:mcmc`, default `:smm`; registry
  `:inert`, since the shipped `WINDOW` is a base window).
- **Re-defaulted:** `MCMC_CHECKPOINT` `true` → `false` (`:inert`), and it now governs only the
  `_mcmcbest` file; `MCMC_BURN` 0.75 → 0.8, the last 1,000 of the 5,000-generation budget.
- **Retired:** `SCENARIO_BUNDLES` (transition_main.jl constant) and `ROYSEARCH_WIDTH_BASE`
  (`width_audit.jl`, now in `LEGACY_ENV_KEYS`); `POINT_ESTIMATE` constants replace both. See
  SETTINGS.md.
- **Launcher:** `code/ops/vm_run.sh` exports `ROYSEARCH_INIT_MODE=warmstart`.

### Known gaps
- **Tail quadrature, not part of the fix.** The unskilled tail by quadrature exceeds its
  identity eU_rowU − eU_entry by 0.32% at base_fc (0.20% at base_covid). The excess is first
  order in the p-grid: +0.161% at Np_U 320 and +0.081% at 640. It is the source of the
  Np_U refinement gap above. The weights are not normalised.
- **A non-wage step at the floor of the skilled grid, present in v56.0.4 too.** Along σ_S at
  base_fc, s = 4.1220759, p*_S of type 108 crosses p_1. Q steps by −3.6e-6 (ur_S +5.9e-7 SD,
  sep_rate_S +5.2e-7 SD); v56.0.4 gives −3.56e-6 at the same point.
- **A p^oj crossing (type 88, s = 10.447) steps Q by −9.4e-7**, with p10_wage_S 1.3e-8 SD
  and ee_rate_S 1.2e-8 SD. It is below the 1e-6 criterion.
- **Rejected solves met by the scans**, identical in v56.0.4:
  - base_fc along σ_S at s = −22.655 (R1 = 1.62) and s = 8.718 (R1 = 2.99);
  - crisis_fc along A + P_S at s = 0.0535 (R1 = 8014);
  - crisis_covid's window [0.020511177, 0.020565618] (R1 = 1.282).

  In each case both blocks report converged and the battery rejects on R1. The scans skipped
  the rest of each affected interval.
- **A transition file is about 1.59 GB** (1,586,038,862 bytes for the covid pair) and lands in
  `output/transition/`, which git tracks; GitHub refuses files over 100 MB. Proposed, not
  applied: `/output/transition/transition_*.jls` in `.gitignore`.
- **The transition file and the tables do not name the point.** `transition_<sc>W.jls` and
  `output/tables/*W.tex` are the same at `POINT_ESTIMATE = :smm` and `:mcmc`, so the second run
  replaces the first; the transition file records the point and its two bundles.
- **The x̄ table is blank, as before v57.** `_get_x_bar` (`plots_and_tables/transition.jl`:881)
  looks for a sign change of net_T − Usearch node by node, but net_T is indexed by a_S and
  Usearch by a_U; at both covid points the difference is negative at every node, so every
  cell prints "--".
- **The horizon test reads the mean wages in logs now.** Its criterion is each series' change
  relative to its own scale, so on the wage series it is about as loose as the log wage is
  large (≈ 6.5). On the covid pair it does not bind: the largest change is the one measured
  before the fix (3.049e-4 at `:smm`, against 1e-3), so a non-wage series sets it, and the
  accepted horizon is 240 months either way. Testing the log-wage series by their absolute
  change would restore the level test's strictness to first order.
- **Plots does not build in the sandbox** (GR_jll: "Incompatible processor"), so
  `transition_main.jl` and `model_main.jl` stop at `using Plots`; the panels, the scatter and
  the figures themselves were not drawn. The transition and its tables ran in a copy without
  Plots; the two re-weighted figures' series were checked through stubs.
- **`run_smoke.sh warmstart` fails SMOKE 4**, as before v57 (max |Δ| 6.284e-02, the same): the
  test's zero-step iteration takes a real move since v40.1.0. Its fix awaits the author.
- **`policy/` does not load** (known); only its unskilled mean-wage weighting changed.

### Tried and rejected
- **A uniform log-wage grid with cloud-in-cell assignment, its range from the min and max of
  every node's log wage, at 2,000 cells** (the specified reference implementation). At base_covid_ξS
  one skilled node wage is 0.038 (log −3.27) while the points lie in [4.43, 7.67], so the
  range spans 11 log units. The grid then misses the exact quantiles by up to 1.35e-3 SD,
  halving the spacing moves p90_wage_S by 1.0e-3 SD, and Q differs from exact by −0.118
  (−0.007 to −0.025 at the other four points). At 4,000 cells the gaps are 3.4e-4 SD and Q
  −0.029. The moment layer took 17.3 ms at 2,000 cells, against 14.9 ms for exact evaluation.
  Rejected for exact evaluation, which is both faster and free of discretisation error.
- **Newton with the convergence test after the bracket test.** A converged step at 4e-16 rounded
  to the bracket end and was read as leaving the bracket. S p90 then took 28 passes over the
  points instead of 5. The test now comes first.
- **One pair map for every script, smm_main.jl included.** It changed the block the author
  uses as the landmark for the crisis switch; smm_main.jl keeps `_PAIR_BASELINE` byte for byte
  and `check_pair_map.jl` holds the two equal, statically rather than by a runtime line.
- **Tagging the files by a field inside RunProvenance** (the seed file's name): the struct is
  read by field position, so a new field would make every stored bundle unreadable. The chain
  estimate records it as `mcmc.seed`, a NamedTuple read by name.

---

## v56.0.4 — 2026-10-12

**PATCH: one console column of `MCMC_main.jl`.** The chain route's table gains an R̂ column
after |t|(bnd), ticked ✓ where R̂ ≤ `MCMC_RHAT_MAX`; every CSV is byte-identical to v56.0.3's
(measured below).

**Comparability.** Full. No stored output is superseded.

### What changed (`code/mcmc/MCMC_main.jl`)
- Header (:1460) and parameter rows (:1625): `%7.3f` of `rhat[k]`, the split R̂ on the retained
  draws that the `R̂ max=… (gate ≤ …)` line takes its maximum over, then ` ✓` where
  `rhat[k] <= MCMC_RHAT_MAX` and two blanks otherwise, so the RAIL / edge flag stays in one
  column. The header reserves the two blanks, so the box still spans the widest row (✓ and
  RAIL: 109 of the box's 110 columns). The k_U, k_S rows leave the cell blank; the JAC_ONLY
  table has no R̂ column. The merged R̂ / ESS line is unchanged; comments :1456 and :1522 say so.

### Measured, not asserted
Scratch tree, the v56.0.3 test (base_fc at a copy of the stored estimate, 2 threads; chain
`ROYSEARCH_MCMC_N=8 ROYSEARCH_MCMC_GENS=24`; `ROYSEARCH_MCMC_JAC_ONLY=true`; every other
setting at its current default): both CSVs byte-identical to v56.0.3's (sha256 694b32fb…,
b9f4c5cd…); the JAC_ONLY console identical apart from the version. On the chain route the 14
R̂ cells equal the CSV's `rhat` to three decimals, their maximum is the header line's 2.703, and
none is ticked (1.624 to 2.703 at this budget; the author's 2026-10-12 base_fc run, read from
its CSV, spans 1.298 to 1.656, so it would print no tick either). The tick and flag layout was
checked by rendering the row format with R̂ = 1.042, 1.099, 1.100, 2.703 and NaN beside RAIL and
edge. `MCMC_main.jl` parses; `check_version_sync` in sync at 56.0.4; `check_forwarding`,
`check_dead_settings`, `check_structure` and `check_output_tree` report v56.0.3's findings.

### Settings
None.

### Known gaps
- No run of the chain route has produced a tick: its branch was checked by rendering only.

### Tried and rejected
Nothing was measured and rejected in this version.

---

## v56.0.3 — 2026-10-12

**PATCH: one console flag of `MCMC_main.jl`.** The parameter rows of the chain route's table
no longer end in an `R̂` flag; R̂ is stated once, as the attained maximum against its gate, on
the box-header line. Every CSV is byte-identical to the unedited tree's (measured below).

**Comparability.** Full. No stored output is superseded.

### What changed
- `code/mcmc/MCMC_main.jl`:1620: the row flag is `RAIL` or `edge` only; the `R̂` branch
  (`draws_ok && rhat[k] > MCMC_RHAT_MAX`) is gone. The comments at :1457 and :1462 no longer
  list R̂ among the flags. The JAC_ONLY table never carried it (no draws).

### Measured, not asserted
Scratch trees of the repository before and after the edit, base_fc at a copy of the stored
estimate, 2 threads, chain `ROYSEARCH_MCMC_N=8 ROYSEARCH_MCMC_GENS=24` and
`ROYSEARCH_MCMC_JAC_ONLY=true`, every other setting at its current default: both CSVs
byte-identical before and after (sha256 694b32fb…, b9f4c5cd…; the chain hash is not the one in
v56.0.2's entry because the test runs at the current defaults). Apart from the scratch trees'
paths, the console differs only in the 14 parameter rows, which lose their trailing `R̂`, and in
the version. `MCMC_main.jl` parses; `check_version_sync` in sync at 56.0.3; `check_forwarding`,
`check_dead_settings`, `check_structure` and `check_output_tree` report v56.0.2's findings.

### Settings
None.

### Known gaps
Nothing new; v56.0.1's other gaps stand.

### Tried and rejected
Nothing was measured and rejected in this version.

---

## v56.0.2 — 2026-10-12

**PATCH: one console line of `MCMC_main.jl`.** The posterior-mean bundle line no longer
prints Q(θ̄), which the estimate line above it already gives; every CSV is byte-identical
to v56.0.1's (measured below).

**Comparability.** Full. No stored output is superseded.

### What changed
- `code/mcmc/MCMC_main.jl`:1872: `posterior mean bundle → <file>` in place of
  `posterior mean bundle: Q(θ̄)=… → <file>`, the duplicate listed under v56.0.1's known
  gaps. On the t-space fallback the bundle's own Q still prints, on the fallback line.

### Measured, not asserted
Scratch tree, the v56.0.1 test (base_fc at a copy of the stored estimate, 2 threads; chain
`ROYSEARCH_MCMC_N=8 ROYSEARCH_MCMC_GENS=24`; `ROYSEARCH_MCMC_JAC_ONLY=true`): both CSVs
byte-identical to v56.0.0's and v56.0.1's (sha256 694b32fb…, fc278af8…). Apart from the scratch
trees' paths, the console differs from v56.0.1's in that line and the version only. `MCMC_main.jl`
parses; `check_version_sync` in sync at 56.0.2; `check_forwarding`, `check_dead_settings`,
`check_structure` and `check_output_tree` report v56.0.1's findings.

### Settings
None.

### Known gaps
Nothing new; v56.0.1's other gaps stand.

### Tried and rejected
Nothing was measured and rejected in this version.

---

## v56.0.1 — 2026-10-12

**PATCH: console output of `MCMC_main.jl` only.** Both routes write a byte-identical CSV, and
every field of the chain file and of both bundles is equal to v56.0.0's apart from the
version and time stamped in their provenance (measured below).

**Comparability.** Full. No stored output is superseded.

### What changed (`code/mcmc/MCMC_main.jl`)
- Table, last column (:1615, :1654, :1658): |t|(bnd) is formed against θ̂ on the chain route
  too, `_abs_t(pe, se_bnd[k])` and `_abs_t(pe, se_bnd_k[i])`, so it is the JAC_ONLY column
  and the CSV's `abs_t_bound` in every row. |t|(chain) and |t|(J⁻¹) are still formed against
  the reported estimate; the CSV header comment (:1563) and the row comments say which is
  which. Under JAC_ONLY the reported estimate is θ̂, so that table does not change.
- Box header (:1524): one line, `R̂ max=… (gate ≤ …)  ESS min=… (gate ≥ …)`. The attained
  values printed NaN on every run, because the reductions were seeded with `init = NaN` and
  max and min propagate NaN. The line `logπ(θ̂)=…  max logπ=…  gates: …` is gone, and with it
  the solve `logposterior(θ0)` it alone used: logπ(θ̂) = −Q(θ̂)/2 is printed as Q(θ̂) below the
  table, and max logπ is on the last progress line.
- `seed drift=…` (:1498) prints only where the drift abort is armed (`MCMC_CHECK_EVERY > 0`
  and `MCMC_DRIFT_MAX > 0`, the condition at `demc.jl`:823-825) or the run aborted.
- Below the table (:1693): `estimate = mean of the pooled post-burn-in draws; se(chain) =
  their SD` and `Q(θ̄)=…   Q(θ̂)=…`, two lines in place of fourteen. Q(theta_best) is on the
  promotion line.
- Posterior-mean bundle (:1872): one line, `posterior mean bundle: Q(θ̄)=… → <file>`, without
  the warm-start instructions or the `(reported)` / `(t_space_fallback)` tag (a fallback
  prints its own line); the variable `used` that held the tag is gone. The infeasible
  branch is one line.
- Bundle sealing (:1959): the NOTE on a bundle without a simulated panel is deleted, with
  its comment block; one comment at `sim = _est.sim` records that
  `plots_and_tables/transition.jl` reads `bundle.sim` unguarded (:178, :399, :876; the
  deleted comment named `transition/plots_and_tables.jl`).
- `README.md`, Standard errors: the CSV row says 20 columns (it said 17) and that the
  console's |t|(bnd) is `abs_t_bound`; a new row lists `estimate_{window}{W}_postmean.jls`,
  with the warm-start use the console no longer prints.

### Measured, not asserted
Scratch trees (code copied, `data/` symlinked), base_fc seeded at a copy of the stored
estimate (sha256 d05cc98d…), restored before every run, 2 threads. Chain route
`ROYSEARCH_MCMC_N=8 ROYSEARCH_MCMC_GENS=24` (6 retained generations, 48 draws; split R̂ needs
4, so 12 generations give NaN by construction); chain-free route `ROYSEARCH_MCMC_JAC_ONLY=true`.

| | v56.0.0 | v56.0.1 |
|---|---|---|
| CSV, JAC_ONLY (sha256) | 694b32fb… | 694b32fb… (byte-identical) |
| CSV, chain (sha256) | fc278af8… | fc278af8… (byte-identical) |
| JAC_ONLY bundle | — | byte-identical |
| chain file, chain-route bundles | — | equal in every field but provenance (and the chain file's `seed_jls`, the scratch path) |
| console \|t\|(bnd) = CSV `abs_t_bound`, JAC_ONLY | 16 of 16 rows | 16 of 16 |
| console \|t\|(bnd) = CSV `abs_t_bound`, chain | 10 of 16 | 16 of 16, and equal to the JAC_ONLY column |
| R̂ max / ESS min printed (CSV: 3.430 / 48) | NaN / NaN | 3.430 / 48 |

The six rows that moved on the chain route: A 195.22 → 195.21, α_U 5.25 → 5.26, μ_S 40.26 →
40.28, λ_U 101.57 → 101.61, θ_U 373.33 → 373.37, θ_S 153.38 → 153.37. No other console
column moved, and the JAC_ONLY console is identical apart from the banner.

Gate: `MCMC_main.jl` parses (`Meta.parseall`); both routes run end to end;
`check_version_sync` in sync at 56.0.1; `check_forwarding` only the known
`ROYSEARCH_MCMC_JAC_ALLOW_GAPS`; `check_dead_settings` PASS (188 keyword arguments);
`check_structure` the same 7 files; `check_output_tree` PASS on the scratch tree. Every
finding is identical to v56.0.0's on the same tree.

### Settings
None.

### Known gaps
- The printing branch of the seed-drift line did not run: the test had no stopping test and
  no abort. Its condition is the one `demc.jl` aborts on.
- With a stopping test armed, a run that stops between progress lines no longer prints its
  final population's max logπ: demc prints the progress line at multiples of
  `MCMC_PRINT_EVERY` and at the last requested generation only.
- On a converged chain the estimate column holds θ̄ while |t|(bnd) is |θ̂|/se(bound), so in
  those rows |t|(bnd) is not the row's estimate over its se(bound).
- Q(θ̄) is printed twice, on the estimator line and on the posterior-mean bundle line, from two
  solves of the same point unless the t-space fallback fires.
- `check_threading` and `check_hire_margin` were not run: no solver or SMM file changed.
- README, chain-free route row: still "Ĝ regressed on a `local_design` around θ̂, ≈ 600
  solves"; under the shipped `MCMC_JAC_METHOD = :fd` it is the adaptive FD, 168 solves at
  base_fc. Not changed here.

### Tried and rejected
Nothing was measured and rejected in this version.

---

## v56.0.0 — 2026-10-11

**MAJOR: the optimiser's coordinates for the two markets are the tightnesses θ_U and θ_S,
and the vacancy costs k_U and k_S are computed from free entry in every evaluation**
(`tightness_reparam_spec.md`, approved by the author 2026-10-11). The model and the
estimator are unchanged: given (θ_U, θ_S) the stationary state is unique and free entry
gives each cost in closed form, k_j = q_j(θ_j) J̄_j / π̄_j, while the map from costs to
tightnesses is a correspondence (unskilled free entry has three zeros at both COVID
estimates). Minimising Q over the parameter vector with the tightnesses in it is the
best-fitting-equilibrium estimator exactly, every evaluation is one equilibrium solved in
one pass, and Q has no branch switches. k_U and k_S stay estimated parameters and every
table lists them. The parameter vector, the bundle schema and the free-parameter keys change,
so the theorist must be told; Q at every stored estimate is reproduced to solver tolerance.

**Comparability.** Q and every moment at the stored estimates are unchanged to the solver's
tolerance (table below), so v55 estimates and v56 estimates are comparable. Every estimate
bundle in use was rebuilt into the new parameter vector, once, by hand, outside the codebase
(below). Superseded and not rebuilt: the candidate banks (their cache key holds the free-
parameter names and the box, so the next `:clusters` run regenerates them; the base_covid
bank was already invalid after today's b_S box edit), the chain `output/chains/
chain_base_fc_diagonalW.jls` (a chain's draws under a prior over k are not draws under the
prior over the new parameter vector, so no conversion is exact), and the ten
`estimate_base_fc_diagonalW_*` backup/postmean files, which no current code can read (struct
shapes from before v49).

### What changed
- **Solver.** `solve_at_tightness!(model, θU, θS)` (`solver/solver.jl`:265): the unskilled
  block at θ_U (`solve_unskilled_block!(...; θU_fixed)`, `unskilled.jl`:582), then the skilled
  block at θ_S with κ_U and E_U(a_U,1) from it, then τ and the masses at the solved U_S, then
  the costs, installed in the parameter blocks (`_with_k` :298). U^search(a_U) does not read
  U_S (training enters only τ), so the dependence between the blocks runs one way and no
  outer loop is needed. A market that holds no one is priced at the limit of its composition
  (`corner_limit`, as in `solve_model!`). With the costs installed the state is an
  equilibrium of the model at those costs in every condition, so the residual battery is the
  certification unchanged (R2 = 0 to rounding by construction). `solve_model(...; tightness)`
  (:70) selects it. Free entry is split into its objects and the two directions:
  `unskilled_meeting_value` (`unskilled.jl`:516), `update_theta_unskilled` (:547, the same
  expression as before, so every fixed-cost solve is bitwise unchanged),
  `vacancy_cost_unskilled` (:563), `vacancy_cost_skilled` (`skilled.jl`:676).
  `solve_model!`, `skilled_equilibria` and the FreeEntryMap stay for exercises that hold the
  costs fixed.
- **Parameters** (`smm/smm_params.jl`). `(:unsk, :tight)` in [0.05, 3.0] and `(:skl, :tight)`
  in [0.10, 5.0] replace the two costs (:687); ASCII keys `:unsk_tight` / `:skl_tight` in both
  key maps; `FIX_PARAMS` with `:unsk_k` or `:skl_k` raises (`_COST_KEYS` :847).
  `unpack_θ` (:987) returns `(cp, up, sp, tight)`, the blocks carrying `k = NaN` until the
  solve computes it; `_baseline_param_value` (:884) reads the tightnesses. `VACANCY_COSTS`
  (:710) labels the cost rows of every table. `BUNDLE_SCHEMA = 3` (:337).
- **Evaluation** (`smm/smm.jl`). `equilibrium_at(θ, spec)` (:163) → `(Q, moments, model, k)`
  replaces `select_equilibrium(θ, spec)`; `smm_objective` (:256) keeps its signature and
  return and loses `branch`. `select_equilibrium(model, spec; branch)` (:208) and
  `equilibrium_tag` stay for fixed-cost exercises. `attracting_check` (:297): D_ij =
  ∂ln k_i/∂ln θ_j by central differences of `ATTRACT_STEP` = 0.01 in ln θ_j (:284, 4 solves);
  market verdict D_UU < 0 and D_SS < 0, joint tr D < 0 and det D > 0; `attracting_line` (:326)
  is the one line printed when a verdict fails. `print_results` and `save_results` list
  k_U, k_S beside the free parameters; `_params_to_namedtuple(θ, spec)` (:2841).
- **Bundle** (`smm/bundle.jl`): fields `vacancy_costs = (k, se, post)` and
  `attracting = (D, market, joint)`.
- **SMM driver** (`smm/smm_main.jl`): `DEFAULT_PARAMS` has `:unsk_tight => 0.8224582555625224`,
  `:skl_tight => 1.8287428145020777` (:342, the hand point M0.1's solved tightnesses under
  v54.0.0) in place of the two costs; every other seed and the b_S box are as the author set
  them today; the seed comment says so. §10 (:1223) computes the costs and the attracting
  check at the estimate and stores both.
- **Standard errors and the chain** (`mcmc/`). `jacobian_adaptive_fd(...; n_carried)`
  (`mcmc_diagnostics.jl`:282) differentiates trailing outputs at the steps the moments select;
  MCMC_main carries k_U, k_S through every Jacobian evaluation (`moments_and_k` :1181), so
  their standard errors are the delta method's at no extra solve:
  `se_bound_diagonal(G, W, σ̂, C)` (:437) evaluates both errors at linear combinations, here
  the gradient of k. The SMM covariance does not depend on the coordinates, so these are the
  errors a k-coordinate estimation reports, and the other parameters' errors are unchanged by
  the change of coordinates. D is read off the Jacobian's tightness columns. The table and
  `mcmc_results_*.csv` gain rows `k` per block; JAC_ONLY writes `vacancy_costs` and
  `attracting` into the estimate bundle and nothing else. The chain records each draw's costs
  from the evaluation that scored it (`_AT_POINT` :830, keyed by the point's hash), stored as
  `k_draws` in the chain file and summarised in `vacancy_costs.post`. The prior is
  `MCMC_PRIOR`'s over the new parameter vector, so flat in θ_U, θ_S where it was flat in the
  costs. No gate changed.
- **Transition** (`transition_simulation.jl`): each endpoint is the solve at its window's
  tightnesses (`solve_endpoint` :132) and its costs are computed there; along the path every
  parameter, the costs included, switches at date 0 and the tightness paths are solved from
  free entry at every date, as before. No equilibrium selection at the endpoints.
- **Other readers**: `gates/check_hire_margin.jl`, `tools/transition_audit.jl`,
  `tools/check_tau_margin.jl`, `tools/ojs_continuity_probe.jl` and `tools/migrate_bundles.jl`
  solve at the tightnesses; `smoke_transition.jl` solves z₀ at them and z₁ at z₀'s costs;
  `plots_and_tables/transition.jl` reads moments from `equilibrium_at` and lists θ_U, θ_S, k_U,
  k_S in the parameter tables (costs from `vacancy_costs`). `policy/` is untouched (it did not
  load before either). README: "The tightnesses as parameters" replaces "Multiple skilled
  equilibria"; the transition and standard-error sections say how the costs enter.

### Bundle rebuild (scratch scripts outside the repository)
The five bundles in use were copied to `output/backups/tightness_reparam_2026-10-11/` and
byte-verified first. Each stored point was solved once under v55.0.0 (`select_equilibrium`, every
one a unique root), and its seed in `spec.free` too; the rebuilt bundle keeps every other
coordinate of θ bit for bit, puts the solved tightnesses in place of the costs in the new box
(all inside it, so no bound moved), records the stored costs in `vacancy_costs.k`, and the
attracting check in `attracting`; `migrated_from` gains the version the estimate came from.

| bundle | sha256 before → after | Q, v55.0.0 | Q, v56.0.0 | ΔQ (relative) | k_U computed vs stored | k_S computed vs stored | max moment gap |
|---|---|---|---|---|---|---|---|
| base_fc | 92e3da9b → ff4de6c1 | 3560.0920827910 | 3560.0920747456 | −2.3e-09 | −3.4e-12 | −6.2e-09 | 1.9e-07 |
| crisis_fc | 4c56efbb → bf001ecd | 1997.8722326122 | 1997.8722532032 | +1.0e-08 | −7.8e-09 | −9.6e-09 | 3.3e-07 |
| base_covid (live, the author's SA checkpoint) | 85685140 → 5a9468a1 | 9516.9530618521 | 9516.9530683760 | +6.9e-10 | −7.3e-10 | +1.3e-12 | 1.6e-08 |
| base_covid (ξ_S-route estimate, the backup) | 23c2ae2b → 2dfe1def | 3935.5499739559 | 3935.5497996394 | −4.4e-08 | −9.3e-10 | +1.3e-08 | 1.6e-06 |
| crisis_covid | 32c8f63b → 4a174ef4 | 2207.7930668364 | 2207.7932128767 | +6.6e-08 | −3.6e-08 | +2.3e-08 | 1.1e-05 |

Q is not bit-identical because the old Q came from the global loop and the new from one
pass: both are the same equilibrium, each to its own tolerance (the old battery R2 at these
points was 1.1e-05 to 3.9e-01 of tolerance). The stored `loss_opt` of base_fc (2957.41) and
crisis_fc (1923.73) predate later code; the rebuilt bundles carry the v56 Q.

Attracting check at the rebuilt estimates (D = ∂ln k/∂ln θ, rows k_U, k_S; columns θ_U, θ_S):
base_fc [−0.5037 0.0110; 0.0029 −0.9904], crisis_fc [−0.6870 0.0130; 0.0051 −0.9560],
base_covid live [−0.2208 0.0077; 0.0051 −0.9721], base_covid ξ_S route [−0.2080 0.0087;
0.0043 −0.8525], crisis_covid [−0.2628 0.0105; 0.0078 −0.7671]: every verdict holds.

### Measured, not asserted (scratch tree, estimation grids)
Speed of one objective evaluation (median of 5 after one warm-up), and block solves per
evaluation:

| estimate | v55, 1 thread | v56, 1 thread | v55, 2 threads | v56, 2 threads | blocks v55 (U + S, global passes) | blocks v56 |
|---|---|---|---|---|---|---|
| base_fc | 6.467 s | 0.242 s | 4.399 s | 0.154 s | 122 + 122 (122) | 1 + 1 (0) |
| crisis_fc | 6.365 s | 0.226 s | 4.267 s | 0.151 s | 122 + 122 (122) | 1 + 1 (0) |
| base_covid | 8.050 s | 0.254 s | 5.402 s | 0.169 s | 117 + 117 (117) | 1 + 1 (0) |
| crisis_covid | 7.279 s | 0.300 s | 4.904 s | 0.205 s | 97 + 97 (97) | 1 + 1 (0) |

The v55 counts include the solves `skilled_equilibria` makes along the free-entry map.

COVID check (base_covid ξ_S-route estimate, θ_S held at 1.8368271058): at the three unskilled
zeros the multiplicity experiments report, θ_U = 0.5822, 0.6871, 0.8154, Q = 27326.303949,
11908.993750, 3935.524598, and k_U is the estimate's 0.6174421292 to 2.3e-05, 1.1e-05, 1.0e-05
(the roots' four digits). Refined to the zero of k_U(θ_U) − k̂_U: θ_U = 0.58218045, 0.68713786,
0.81536030, k_U equal to 1e-14, Q = 27329.219151, 11904.409769, 3935.549800, battery Rmax
8.0e-02 at each; D_UU = −0.674, +0.197, −0.208, so the attracting check marks the middle root
as repelling.

Delta method against a direct k-coordinate Jacobian (base_fc, unique equilibrium; simple
central differences of 1e-3 of each value in both, the k-coordinate columns re-solving the
tightnesses by Newton steps from the estimated root, 122 solves): se(J⁻¹) of k_U 4.198e-02
against 4.178e-02 (gap −0.47%), se(bound) 1.871e-01 against 1.862e-01 (−0.51%); k_S 8.300e-03
against 8.301e-03 (+0.007%), 1.7006e-02 against 1.7009e-02 (+0.018%); every other parameter's
two errors agree to at most 1.0% (b_U −0.97%, λ_U se(bound) +1.0%), the finite-difference noise
at these steps.

`MCMC_JAC_ONLY` end to end at base_fc, 2 threads: 14 of 14 coordinates in 166 solves, cond(Ĵ)
4.09e+06; θ_U 0.45875 (se(J⁻¹) 0.00119, se(bound) 0.00121), θ_S 1.21036 (0.00778, 0.00782),
k_U 1.12064 (0.03867, 0.17011), k_S 1.25608 (0.00831, 0.01684); D from the Jacobian
[−0.5036 0.0110; 0.0029 −0.9996], both verdicts hold, nothing printed.

Threaded (2) and serial (1) agree bitwise: `check_threading -t 2` Q = 3560.0920747456 three times.

Gate: changed files parse; the entry points load; `assert_all_params_accounted` passes for
every rebuilt spec; `check_hire_margin` 0 disagreements; `check_dead_settings` PASS (188
keyword arguments); `check_forwarding` only the known `ROYSEARCH_MCMC_JAC_ALLOW_GAPS`;
`check_structure` the same 7 files; `smoke_transition.jl`, `run_smoke.sh guard` and
`run_smoke.sh clusters` pass (an 18-candidate bank, the paired baseline injected as start 6);
`transition_audit.jl TA_NULL=true` runs.

### Settings
New: none. `DEFAULT_PARAMS` keys `:unsk_k`, `:skl_k` became `:unsk_tight`, `:skl_tight`.
No other setting added, retired or moved.

### Known gaps
- `run_smoke.sh warmstart` fails at its SMOKE 4 test ("SA's first solve is not the baseline
  vector", max |Δ| = 6.284e-02); v55.0.0 fails it identically on the repository as it stands,
  so it comes from the current settings and stored bundles, not from this change. Not
  investigated.
- `check_output_tree` rule 6 flags the `.pre_migration` files and Q-named backups in
  `output/estimates/`, in v55.0.0 as in v56.0.0.
- The stationary-solver standard errors of θ_U, θ_S in a bundle written before this version do
  not exist; no bundle in use carried standard errors.
- The single pass assumes the blocks' dependence runs one way, which holds in the code as
  written for every κ_U (the training option does not enter U^search). Should the model
  later give a crossing type a training option in its unskilled value, that changes.

### Tried and rejected
Nothing was measured and rejected in this version.

---

## v55.0.0 — 2026-10-11

**MAJOR: the transition's match continuation ends a match where the cutoff has moved past
it, the training slice's p = 1 atom is judged on its raw surplus, a crosser hired joins the
slice of its type, and the output bundle carries the pre-switch steady state (`schema = 2`).**
Every transition number changes (the fc and covid paths below); solver/, smm/ (except
`version.jl`), mcmc/, gates/ and smoke/ are byte-identical to v54.0.0, and so is every Q.

**Comparability.** Every Q, moment, selected root and estimate bundle is unchanged (table
below). Every transition path from v54.0.0 or earlier is superseded. No v54 transition bundle
exists in `output/transition/` (only the April `transition_fc_fullW.jld2`, which no code reads),
so nothing is rebuilt. A schema-1 bundle has no `baseline`; the panels require schema 2.

### What changed
- **Continuation at a moving cutoff** (`transition/transition_values.jl`). `_continuation!`
  (:99) is the positive part of the next date's surface above the boundary nodes (`_boundary`
  :77: the last node below the cutoff, or the top two nodes when the cutoff lies above the
  grid), the raw value at the boundary nodes, and 0 below them; it returns the boundary values
  as an edge. Each date is solved with the edges in place; where the solution's boundary nodes
  differ from an edge's, or the solve did not converge, `_drop_moved_edges!` (:114) takes that
  edge's values out (the cutoff has moved past the node, the match ends, its continuation is 0)
  and the date is solved again (`_unskilled_surface!` :160, and the skilled values and types
  steps). Edges of zero weight are skipped. A steady state and every date whose boundary does
  not move are solved exactly as before. The training slices mix their two continuations with
  their edges weighted by the slice shares (`_scaled` :86, :612, :678).
- **Training slice's p = 1 atom** (A1). `SUt1` stores the training slice's raw surplus at
  p = 1 (`transition_values.jl`:680, :897; `transition_params.jl`:133); its positive part
  continues wherever it is read as a value (:815, `transition_solver.jl`:552), and the forward
  pass keeps the atom only where the raw surplus is ≥ 0 (`transition_solver.jl`:249). Before,
  the stored value was the positive part, so the test never ended the atom.
- **Crossers' hires** (A2). A skilled unemployed hired in the unskilled market joins the slice
  of her type: the hire's value is U_tr + β S_t(a,1)⁺ where U_tr ≥ U_C, else U_C + β S_s(a,1)
  (`transition_values.jl`:815). Forward, the share ω_X = max(d + τ − 1, 0)/d of a cell's
  crossers that the frontier encloses enters the training slice's atom (`_enclosed_crossers`
  `transition_solver.jl`:164, :342), and free entry values those hires at the training slice's
  surplus (:548).
- **Pre-switch steady state** (`transition/transition_simulation.jl`). `stationary_record`
  (:152) is a one-step self-transition at z₀, measured by `path_statistics` like the path;
  the bundle field `baseline` (:271) holds it; `TRANSITION_SCHEMA = 2` (:24). The panels
  draw it before the switch (`plots_and_tables/transition_panel.jl`, `build_model_series`
  :422, `_splice` :919, `_splice_agg` :931). Before, they took date 0, which is after the
  date-0 reallocation (fc: ur 0.051392 and θ_U 0.179858 against the steady state's 0.053935
  and 0.458752).
- README: "Transition path" lists `baseline` and `schema = 2`.

### Measured, not asserted (scratch tree, estimation grid Nx = Np_U = 160, Np_S = 180, 1 thread unless stated)
Self-transitions (241 dates, 3 outer iterations, tol 1e-12), free-entry gap: base_fc 5.917e-12,
crisis_fc 8.116e-08, base_covid 1.656e-09, crisis_covid 5.115e-08, the v54 values to every
printed digit. Against v54 on the base_fc self-transition (same runs, 3 iterations): every θ,
mass and rate moment agrees to ≤ 4.5e-13 (relative) or 1e-17 (shares); the value arrays agree
to ≤ 6e-11 except `SUt1` (it now holds the raw surplus, up to −5160 where v54 stored 0) and
`U_S^(1)` (A2, below); `d_region` (∬ d ℓ) is 0.441906 against 0.441004.

base_fc → crisis_fc: 56 iterations at horizon 120 (gap 9.36e-05), 10 more at 240 (gap
7.12e-05); the reported series moved by at most 1.199e-04 of their scale over months 0–36, so
240 is the horizon. 5980 s. Free entry ≤ 2.896e-05 (U, t = 4), 7.122e-05 (S); population
identity 5.692e-19; no negative mass. **A15: no (row, node) pair on any date in either block
(v54: 284 pairs on dates 1–11).** Date-1 reallocation: 1.8823e-03 of the population quit into
training (v54 1.5894e-03; the difference is A1), 2.6631e-03 swept, no match ended. Crossing
choice: 336 reversals in 106 cells (1 at node level), skilled mass on them 0; τ 0 node-level
reversals; OJS participation 3. At month 120 the path is away from its month-240 values by:
training share +3.469e-02, ur_S +1.498e-02, skilled share −4.580e-03, ur +3.214e-03, ur_U
−4.307e-04, θ_S −7.631e-04, θ_U −4.417e-06; within 1% from month θ_S 3.5, θ_U 10, ur_U 21.5,
ur 57.5, skilled 85.5, ur_S 138.8, training 173.4. Terminal θ against z₁: U 1.809e-05, S −1.933e-04.

base_covid → crisis_covid: 100 iterations at 120 (gap 9.46e-05), 10 at 240 (gap 9.90e-05,
change 1.711e-04). 10337 s. Free entry ≤ 4.545e-05 (U), 9.902e-05 (S); identity 2.345e-18; no
negative mass. A15: 0 dates (v54: 4). Date-1 quits into training 3.5073e-03 (v54 3.0919e-03),
swept 3.7713e-03. Crossing: 780 reversals in 169 cells (10 at node level), skilled mass 0; τ 3;
OJS 15. θ_U 0.9551 → 1.0513, θ_S 2.4236 → 2.3205. The path jumps at t = 4.5 (below).

A1: on fc, 1,759,465 (cell, date) pairs in 6005 cells have a training slice with raw
S_t(a,1) < 0; on covid 2,064,208 in 7047 cells. The atom mass carried into them is 0 to the
reported precision: the atom there is the mass swept in at the date, which now quits into
training at once (date-1 quits +2.929e-04 fc, +4.154e-04 covid). A2: no (cell, date) pair on
either path has crossers enclosed by the frontier, so the forward split moves no mass.

Step halving (fc, Nx = Np = 60, T_max = 120, tol 1e-5): Δ = 0.5 converges in 45 iterations
(gap 8.473e-06); Δ = 0.25 does not converge in 300 (gap 3.245e-04, oscillating between 1.2e-05
and 3.2e-04, the worst dates t = 0.5–2). Its θ_U is 3.888e-03 from Δ = 0.5 (t = 2), θ_S
1.176e-04 (t = 5.5), masses ≤ 3.0e-06 (v54, converged in 46: 3.1e-03, 1.1e-04, 3.2e-06).
Guess independence (z₀ against z₁ guess, Δ = 0.5): θ_U 9.664e-07, θ_S 3.340e-08, masses ≤ 2.1e-11.

Threaded (2) and serial (1) agree bitwise in every path array and result field (fc and covid,
Nx = 60, 4 iterations).

Q at the stored estimates (bundle grids, `select_equilibrium`), unchanged:

| estimate | bundle sha256 | Q | battery R2 |
|---|---|---|---|
| base_fc | 92e3da9b | 3560.0920827909513 | 1.10e-05 |
| crisis_fc | 4c56efbb | 1997.8722326121679 | 2.69e-02 |
| base_covid | 23c2ae2b | 3935.5499739559359 | 1.71e-03 |
| crisis_covid | 32c8f63b | 2207.7930668364093 | 3.88e-01 |

Gate: changed files parse; the panel's `build_model_series` reads the fc bundle with its
`baseline` (78 dates, ur 0.05394 first, the steady state); `smoke_transition.jl` passes;
`transition_audit.jl` with `TA_NULL=true` runs; `check_threading -t 2` OK (3560.0920827910);
`check_hire_margin` 0 disagreements; `check_dead_settings`, `check_output_tree` PASS;
`check_forwarding` only the known `ROYSEARCH_MCMC_JAC_ALLOW_GAPS`; `check_structure` the same 7 files.

### Settings
None added, retired or moved.

### Known gaps
- **The covid path is not unique.** The unskilled free-entry map has three roots at both covid
  endpoints (base_covid ≈ 0.59 and 0.8154 attracting, ≈ 0.69 repelling; crisis_covid ≈ 0.70 and
  1.0512 attracting, ≈ 0.97 repelling). The path's θ_U starts at 0.955, falls to 0.937 by
  t = 4 and jumps to 1.034 at t = 4.5 (Δ ln θ_U 0.0926), crossing the repelling root; 5.99e-03
  of the population's matches end at that date, and ur_U goes 0.0603 → 0.0713 (in v54, at
  t = 4.0: rows a_U ≥ 0.067, qualities p 0.062–0.124 that accumulated in the first months). The values are continuous
  through it (per-step change of U^search ≤ 1.7e-04, relative); only their slope kinks. It is a
  perfect-foresight path, and not the only one: at Nx = 60 the z₀ guess converges to another
  (θ_U 0.78 by month 6, a jump at month 119.5). v54 jumped at t = 4.0 (0.0862, 5.48e-03); the
  continuation moved it, A1 did not (A1 on and off: same date, size and ended mass).
- **Step halving does not converge at Δ = 0.25** (above). The number of edges dropped per pass
  is constant (14,623) from iteration 35, so the continuation rule is not toggling; the cause is
  not identified.
- **A2's value side has no stationary counterpart.** The stationary solver values a crosser's
  hire at U^search(a_U) + β S_U(a_U,1) (`solver.jl`:174), with no training option. On the base_fc
  self-transition U_S^(1) differs from v54 in 5,644 of 25,600 types, all in the node-level
  training region, by up to 10% (row 27, a_S top); no node-level crossing decision changes and
  no mass moves (≤ 1e-17), but `d_region` moves 0.441004 → 0.441906.
- The p = 1 atom of a training slice is tested at the node: a cell whose slice surplus is
  negative ends its whole atom.
- The training-slice continuation mixes in the other slice's edge with weight ς; where ς is tiny
  this gives ~1e-08 differences from v54 in `SUt1`'s positive part on a self-transition.
- Output size: each bundle is 1.5 GB.
- The node-level approximations of the v54 entry remain.

### Tried and rejected
- Keeping the next date's boundary node at its raw value in every case (v54): where the
  cutoff rises past that node it inherits a negative continuation (A15, 284 pairs on fc).
- The positive part at every node: the boundary values locate the zero, and clipping them
  moves a steady state's cutoff.
- Solving each date once with the edges in place, without the fallback for a non-converging
  pass: on fc at horizon 240, row 57 at t = 4 cycled between cutoffs 0.41115 and 0.39795 for
  300 passes (residual 3.75e-02); with the edges dropped it converges at 0.3979.

---

## v54.0.0 — 2026-10-10

**MAJOR: the transition layer implements the theorist's transition specification
(`transition_spec.md`) with the author's reading (A) of the crossing choice, and its entry
point is rebuilt around the stored estimates.** The path now solves every choice at every
date per type where a later choice can differ, conserves mass in levels, and reports what the
author asked for in one output bundle. Every transition number changes; the theorist must be
told. solver/, smm/ (except `version.jl`), mcmc/, plots_and_tables/, gates/ and smoke/ are
byte-identical to v53.0.0, and so is every Q.

**Comparability.** Every Q, moment, selected root and estimate bundle is unchanged (table
below). Every transition path from v53.0.0 or earlier is superseded. No stored transition
bundle exists in the v53 format (the only file in `output/transition/` is an April
`transition_fc_fullW.jld2`, which no code reads), so nothing is rebuilt.

### What changed
- **Backward pass** (`transition/transition_values.jl`, rewritten). Values continue on the next
  date's envelope (spec P0). Per-type values are stored as (aU, aS, date) and solved only on
  the types whose later choices can differ (`pureU`, `pureS`); the others are the solver's
  one-dimensional values exactly (spec R1). Reading (A): the skilled unemployed choose the
  market at every date, and both branches and T continue on max{U_S^(0), U_S^(1)} of the next
  date (`_skilled_types_step!` :736, `_training_step!` :458). A cell has a searching and a
  training slice, and the share the training frontier sweeps or releases between dates mixes
  their continuations (`_slice_continuations!` :512).
- **Training frontier within a date** (`_unskilled_slices_step!` :543). τ is located on the
  value of searching at n and n+1, C⁰, and the types between that frontier and the next date's
  train at n when U_tr ≥ C¹ (the value at a full sweep). Locating it on the swept value made τ
  a fixed point of C(τ) with slope 1/(1−τ): on crisis_fc and base_covid a self-transition left
  the steady state (τ moved 0.48 at the first backward step, free-entry gap 1.1e-4).
- **Forward pass** (`transition/transition_solver.jl`). The masses are advanced in levels at the
  date's rates, in sub-steps of at most `FORWARD_SUBSTEP` = 1 month (`_forward_pass!` :209,
  `_advance_unskilled_cells!` :318). Date-n reallocations follow spec L6: matches a cutoff
  passes end, the frontier's sweep moves the searching slice's unemployed and matches into
  training. The p = 1 atom is judged by the hiring margin, not by the cutoff (`_end_below!`
  :149, kept share χ₁/χ₀ at date 1 :243): p* = 1 on the row the hiring margin crosses holds
  matches of positive surplus, and ending them moved 1.5e-4 of the population at date 1 of
  every base_fc self-transition at Nx = 160. Where the base cutoff covers no node, the row's
  employment is all at p = 1 (`_unskilled_p_shape!` :60); dropping it lost 7.0e-10 of the
  population on crisis_fc.
- **Free entry** (`_free_entry_tightness` :524) with the per-type firm values; convergence on
  max over dates of |ln θ^new − ln θ| (spec); first guess z₁.
- **Time grid** (`transition_params.jl`, `time_grid` :87): the fine grid to `T_max`, then a tail
  growing by `TAIL_GROWTH` = 1.05 to `horizon`. `horizon = T_max` is the old uniform grid.
- **Threads.** The backward pass's per-type flags were `BitArray`s written from threads, so two
  threads wrote bits of one word: threaded and serial disagreed from the second outer iteration
  (U_S^(0) at 3 dates, then everything). They are `Array{Bool}` (`_backward_pass!` :893).
- **Speed** (bit-identical to the unoptimised reference; see below): the cutoff scan stops at the
  last node a cutoff can reach; slice transfers and empty measures are skipped; redraw weights
  are formed once per row and cutoff (`_redraw_weights!` :169); the unskilled cells are advanced
  in threads by row.
- **Outputs.** `path_statistics` (:776): flows by date (hires, crossers' hires d·κ_U·χ·u_S,
  separations by slice, entries into and completions of training, skilled hires, separations
  and job-to-job moves, the date's reallocations) and the rate moments measured as
  `compute_equilibrium_objects` measures them. `policy_reversals` (:871, `REVERSAL_TOL` = 1e-6
  of a cell): reversals of d (with the skilled mass on the cells), τ and OJS participation.
  `TransitionPath` gains the unskilled employment by (aU, p) of each slice (`eUsp`, `eUtp`).
  `solve_transition_path` (:588) returns the path and takes a warm start; `solve_transition`
  is unchanged for its callers.
- **Entry point** (`transition_main.jl`, `transition_simulation.jl`, rewritten). Grids from each
  bundle's `spec.run`, an error if the windows differ there or in a_ℓ, b_ℓ, ρ_x, r, ν, φ
  (`check_pair` :109). `POINT_ESTIMATE` (`transition_main.jl`:37) = `:smm` (θ̂) or `:mcmc`, the
  estimate MCMC_main reports: θ̂ when the chain failed its gates, else the posterior mean of the
  `_postmean` bundle, checked against the chain's mean; an error for a window no chain sealed
  (`point_estimate` :72). `SCENARIO_BUNDLES` (:42): the covid baseline is the backup
  `output/backups/base_covid_xiS_route_2026-10-10/estimate_base_covid_diagonalW.jls`. Endpoints by
  `select_equilibrium` under each window's `spec.sim`. Horizon by test (`solve_with_horizon`
  :152): doubling from 120 months until no reported series moves by more than `HORIZON_TOL` =
  1e-3 of its scale over the first 36 months; every trial recorded. Crisis-window length from
  `windows.json` (`crisis_window_months` :54). Output `output/transition/transition_<sc><W>.jls`,
  a NamedTuple with `schema = 1` (README, "Transition path").
- Settings: `TR_Nx`, `TR_Np_U`, `TR_Np_S` removed; `POINT_ESTIMATE`, `SCENARIO_BUNDLES` added.
- `tools/transition_audit.jl`: two comment words. README: a "Transition path" section; one line
  reference moved (`transition_solver.jl:448`).

### Measured, not asserted (scratch tree, 2 threads unless stated, estimation grid Nx = Np_U = 160, Np_S = 180)
Self-transitions (a window against itself, 241 dates, 3 outer iterations, tol 1e-12). Sup over
dates of the distance to the steady state, and the free-entry residual the stationary battery
leaves at that endpoint (R2, normalised by its tolerance):

| | base_fc | crisis_fc | base_covid | crisis_covid |
|---|---|---|---|---|
| free-entry gap | 5.9e-12 | 8.1e-08 | 1.7e-09 | 5.1e-08 |
| θ_U / θ_S, relative | 8.6e-12 / 1.4e-13 | 1.2e-07 / 2.5e-10 | 2.0e-09 / 1.1e-11 | 5.5e-08 / 1.7e-10 |
| U^search, U_S^(0), relative | 3.0e-12, 1.5e-14 | 4.7e-09, 6.9e-12 | 3.1e-11, 3.6e-13 | 1.4e-09, 1.4e-12 |
| τ, absolute | 2.5e-10 | 2.8e-06 | 5.2e-08 | 3.3e-07 |
| masses, share of own total | ≤ 2.1e-13 | ≤ 4.0e-09 | ≤ 1.4e-12 | ≤ 1.1e-10 |
| rate moments, max \|path − z1\| | ≤ 1.3e-12 | ≤ 9.8e-09 | ≤ 3.9e-10 | ≤ 1.4e-08 |
| population identity | 6.2e-19 | 4.2e-19 | 6.4e-19 | 1.5e-18 |
| battery R2 at the endpoint | 1.1e-05 | 2.7e-02 | 1.7e-03 | 3.9e-01 |

The ordering follows the endpoint's own free-entry residual: the path settles on its exact
fixed point, which the stationary solve reaches to its tolerance. On a grid that is not an
estimation grid, base_fc at Nx = Np_U = Np_S = 160 (1 thread): gap 1.9e-10, θ_U 2.6e-10, τ 3.7e-09.

base_fc → crisis_fc, the entry path: 53 iterations at horizon 120 (gap 9.4e-05), 9 more at 240
(gap 7.3e-05); the reported series moved by at most 1.3e-04 of their scale over months 0–36
between the two, so 240 is the horizon. 4262 s. Free entry ≤ 7.3e-05 (U), 6.8e-05 (S) at every
date; population identity 5.8e-19; no negative mass. Date-1 reallocation: 1.59e-03 of the
population quit into training, 2.66e-03 swept into training, no match ended. Crossing choice:
d > 0 on 10,498 cells at some date, none holding skilled mass at any date; 575 reversals in 150
cells (1 at node level), skilled mass on them 0. τ: 0 node-level reversals; OJS participation: 5.

At month 120 the path is still away from its month-240 values by: training share +3.5%, ur_S
+1.5%, skilled share −0.46%, ur +0.32%, ur_U −0.04%, θ_S −0.09%, θ_U −1e-06. It stays within
1% of them from month: θ_S 3.5, θ_U 10, ur_U 21.5, ur 57.5, skilled share 85.5, ur_S 139,
training share 173.

Other checks (Nx = Np = 60, fc pair, tol 1e-5, 1 thread): base-θ guess against crisis-θ guess,
θ paths within 1.8e-07 (A11); halving the step to 0.25 moves θ_U by 3.1e-03 and θ_S by 1.1e-04
at most, in the first 6 months (A10). A backward step with 1/dt = 0 returns the stationary
envelope max{U_S^(0), U_S^(1)} to 1.6e-11 with no change of d (A2); T, and through it τ, keeps
the next date's U_S by construction.

Threaded (2) and serial (1) agree bitwise in every path array and result field (fc and covid,
Nx = 60, 4 iterations); the optimised tree agrees bitwise with the unoptimised reference (same
runs, and fc at Nx = Np = 160, 2 iterations: 58.4 s reference, 33.4 s optimised).

Q at the stored estimates (bundle grids, `select_equilibrium`), unchanged:

| estimate | bundle sha256 | Q |
|---|---|---|
| base_fc | 92e3da9b | 3560.0920827909513 |
| crisis_fc | 4c56efbb | 1997.8722326121679 |
| base_covid | 23c2ae2b | 3935.5499739559359 |
| crisis_covid | 32c8f63b | 2207.7930668364093 |

Gate: changed files parse; the transition chain loads; `run_transition_simulation(:fc)` runs
end to end and writes to the scratch tree; the panel's `build_model_series` reads its `result`
(Plots does not precompile in the sandbox, so the panels themselves were not drawn);
`smoke_transition.jl` passes; `transition_audit.jl` with `TA_NULL=true` runs; `check_threading
-t 2` OK (3560.0920827910); `check_hire_margin` 0 disagreements; `check_dead_settings`,
`check_output_tree` PASS; `check_forwarding` only the known `ROYSEARCH_MCMC_JAC_ALLOW_GAPS`;
`check_structure` the same 7 files.

### Settings
`TR_Nx`, `TR_Np_U`, `TR_Np_S` removed (grids come from the bundles). New: `POINT_ESTIMATE = :smm`,
`SCENARIO_BUNDLES` (covid baseline at the 2026-10-10 backup). Unchanged: `SCENARIOS`, `TR_T_MAX`
120, `TR_N_STEPS` 240, `TR_TOL` 1e-4, `TR_MAXIT` 200, `TR_DAMP` 0.3.

### Known gaps
- **A15 fails near a rising unskilled cutoff.** On dates 1–11 of the fc path, 284 (row, node)
  pairs just above p*_U have a raw unskilled surface below the node beneath them, by up to 0.7%
  of the row's top-node surplus (row 32, date 1: −0.52, 0.086, −1.552, 7.527 at p = 0.731 …
  0.756, with p* rising 0.738 → 0.750). The continuation keeps the next date's straddling node at
  its raw (negative) linear extension; when the cutoff rises past a node, that node is above
  today's cutoff and inherits a negative continuation where the match's value at n+1 is 0. The
  masses are unaffected (acceptance is by the covered fraction above p*); the surface, and the
  redraw integral through it, are biased at those nodes. Not seen in the self-transitions; at
  Nx = 60 the fc path shows it on 5 dates. Not fixed.
- **Output size**: the fc bundle is 1.5 GB (293 dates; the per-type arrays are 60 MB each).
- Approximations, all at node level: non-pure skilled types use their column's OJS
  participation and cutoff; skilled employment of crossing types takes the column's p-shape
  (inert here: no crossing cell holds skilled mass); the training slice's continuation is
  formed per node; the p = 1 atom of a training slice ends when that slice's entry surplus is
  negative; the frontier's two-step test is per node.
- T continues on the next date's U_S (explicit in that term), so A2 holds for every value but
  T and τ; A10 measures the time error at 3e-3 of θ_U in the first months.
- Path statistics carry the flow and rate moments; duration, survival and wage-dispersion
  moments have no off-steady-state counterpart here and are not reported.
- The panels take the date-0 value of each series as the pre-switch steady state; date 1 is
  after the date-0 reallocation (`plots_and_tables/transition_panel.jl`, untouched).
- The crisis_covid second stationary equilibrium (v53.0.0 entry) is not revisited.
- `:mcmc` was tested on synthetic sealed bundles in scratch; no stored estimate is MCMC-sealed.

### Tried and rejected
- Locating the training frontier on the swept searching value: unstable (above).
- Ending the p = 1 atom when the cutoff reaches 1: ends matches of positive surplus.

## v53.0.0 — 2026-10-10

**MAJOR: the transition layer is brought back in line with the stationary solver.** Every change
is numerics or forwarding, not economics: the transition now integrates the same objects the
solver integrates, solves its endpoints under the settings the estimation used, and conserves
the skilled p-composition. Every transition path changes, so a transition result is not
comparable across this version and the theorist must be told. solver/, smm/ and mcmc/ are
byte-identical to v52.1.0, and so is every Q.

**Comparability.** Every Q, moment, selected root and bundle is unchanged (table below). Every
transition path computed under v52.1.0 or earlier is superseded, including the base_fc → base_fc
null path (it moved the training frontier). Nothing in `output/` was touched; no stored
`TransitionResult` exists (byte scan in v52.1.0), so nothing is rebuilt.

### What changed (`transition/`, `tools/transition_audit.jl`)
- **Value integrals use the solver's search rule.** The skilled date step and the terminal seed
  integrated the destination over max(S⁰, S¹). `skilled_inner_loop!` integrates
  s S¹ + (1−s) S⁰ with s = `_ojs_share(part, ·)`, and so do both now (`transition_values.jl`,
  `_seed_terminal_values!`, `_skilled_value_step!`). The participation weight of a cell reads
  its neighbours' scores, so a date is swept with the weight fixed, the weight is rebuilt and
  the sweep repeats until it settles to `tol_outer_S`. `POJ_SCORE` is honoured.
- **Training policy through `training_policy!`** (`_margin_step!`) instead of the complement of
  `drain_fraction!`, which differs in the last bits, so the null path's τ never equalled the
  stationary τ.
- **Unskilled reservation** (`_unskilled_value_step!`): the exact zero of the affine raw surplus
  on [0, 1], as `update_pstar_from_surplus!` gives without the continuation. The node scan
  returned p* = 1 for hired types whose zero lies above the top node.
- **Skilled p-composition** (`transition_solver.jl`, `_forward_skilled_pdist!`): the step is taken
  in levels and divided by the non-draining column mass it reaches. Advancing the unit shape by
  the level law omitted −ê ṁ₀/m₀. The loop starts at the straddling cell, as the stationary
  block's does.
- **Endpoint settings** (`transition_simulation.jl`): each window is solved under its own
  `spec.sim`. The hand-rebuilt SimParams dropped `damp_inner_U/S` (0.95 → 1.0), `inner_B/K`
  (20/10 → 0/5) and `outer_B/K` (30/10 → 0/5), and the crisis endpoint used the base
  window's settings.
- **`solve_transition`** asserts the shared ability grid, population weights and p-grids it
  already assumed.
- **`tools/transition_audit.jl`** takes its endpoints from `select_equilibrium` under each
  window's `spec.sim` (it used `solve_model`, whatever root the normal solve reached).
- Comments: the header's claims that the estimates carry ξ_U ≠ 0 and δ_S < 1, and the
  `_skilled_margin_masses` claim about the moment layer, are removed; unskilled/skilled replace
  untrained/trained in `transition_params.jl` and the solver header; `transition_main.jl`'s
  usage line names the real path.

### Measured, not asserted (scratch copies of v52.1.0 and v53.0.0, 2 threads, Nx = Np = 120, 241 dates, T_max 120)
Self-transition (base → base, outer tol 1e-12), sup over dates of the distance to the steady state:

| | v52.1.0 base_fc | v53.0.0 base_fc | v52.1.0 base_covid | v53.0.0 base_covid |
|---|---|---|---|---|
| outer iterations | 40 | 8 | 60 (not converged, 1.2e-11) | 50 |
| θ_U / θ_S, relative | 2.9e-07 / 2.3e-06 | 4.6e-11 / 1.0e-12 | 4.2e-07 / 1.4e-06 | 1.6e-08 / 9.4e-12 |
| U_S⁽⁰⁾, T, relative | 1.1e-05, 8.4e-06 | 2.9e-14, 7.4e-12 | 4.2e-06, 3.2e-06 | 3.2e-13, 2.9e-12 |
| τ, absolute | 4.1e-03 | 9.7e-09 | 1.9e-03 | 2.3e-07 |
| masses, share of own total | ≤ 2.1e-07 | ≤ 7.7e-13 | ≤ 2.9e-08 | ≤ 1.5e-12 |
| max\|1 − û − ∫ê\| | 1.9e-06 | 5.9e-15 | 1.4e-06 | 9.3e-15 |

The stationary battery's free-entry residual at these endpoints is 4.3e-11 (base_fc) and
2.5e-08 (base_covid), relative.

Real pairs (tol 1e-4): fc 22 iterations in both versions, covid 65. Closure max|1 − û − ∫ê|
over columns holding more than 1e-6 of the skilled mass: 1.96 → 1.8e-15 (fc), 2.11 → 1.8e-15
(covid); mass-weighted 3.0e-02 → 4.1e-16 and 2.8e-02 → 4.4e-16. θ_S at the last date against
z1: 2.9e-03 → 1.5e-03 (fc). Free-entry residual at every date ≤ 6.1e-04 (U) and 3.5e-04 (S) on fc,
the outer tolerance at work. Population identity ≤ 3.3e-16 at every date. Threaded (2) and
serial (1) agree bitwise in every TransitionResult field and path array (fc, both versions).

Q at the stored estimates (bundle grids, `select_equilibrium`), identical in both trees:

| estimate | bundle sha256 | Q |
|---|---|---|
| base_fc | 92e3da9b | 3560.0920827909513 |
| crisis_fc | 4c56efbb | 1997.8722326121679 |
| base_covid | 23c2ae2b | 3935.5499739559359 |
| crisis_covid | 32c8f63b | 2207.7930668364093 |

The first three equal the v52.1.0 entry's numbers. The crisis_covid bundle was rewritten on
2026-10-10 05:10 UTC by the running estimation, so it is not the file v52.1.0 measured.

Gate: changed files parse; `run_transition_simulation(:fc)` runs through the changed entry path
at Nx = 120 (22 iterations, 9.741e-05) and writes to the scratch tree; `check_threading -t 2` OK
(3560.0920827910); `check_hire_margin` 0 disagreements; `check_dead_settings`,
`check_output_tree` PASS; `check_forwarding` the known `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` finding;
`check_structure` the same 7 files; versions in sync at 53.0.0. Backup of the pre-change files,
byte-verified: `output/backups/pre_transition_sync_2026-10-10/`.

### Settings
None added, retired or re-defaulted.

### Known gaps
- **The transition economics is still the v35 one** (to be replaced by the spec in
  `transition_spec.md`): every continuation is on its own branch (U^search, U_S⁽⁰⁾, U_S⁽¹⁾, the raw
  surfaces, each OJS branch), T continues on U_S⁽⁰⁾, wages use (r+ν)U, unskilled employment has
  no p-distribution, so a rising reservation destroys nothing, and incumbents never quit across
  margins.
- **A second stationary equilibrium at the crisis_covid estimate** (bundle 32c8f63b). It lies in
  the unskilled block: θ_U = 0.698670, θ_S = 2.313589, battery Rmax 1.0e-04, median p*_U 0.0007,
  Q 26764.17, against the selected θ_U = 1.051248, θ_S = 2.321161, Q 2207.79. These are at the
  estimation grid; it was found by warm-starting the stationary solver at the covid transition's
  month-60 state. `skilled_equilibria` assumes uniqueness given θ_S and cannot find it. The
  lowest-Q rule is not affected. The covid path sits within 3e-4 of θ_U = 0.6975 from month 20 to
  month 100. Not checked: whether the other three estimates have such a root.
- At T_max = 120 the fc terminal masses miss z1 by t +4.6%, m_S −1.0% (rate ≈ ν).

### Tried and rejected
Nothing.

## v52.1.0 — 2026-10-09

**MINOR: the meeting rates are named κ throughout the code.** The notes' convention is now
κ_j = θ_j q_j(θ_j), the rate at which a job seeker in market j meets a vacancy, for unemployed
and on-the-job searchers alike; f_j is no longer a meeting rate anywhere. Every identifier
holding θ_j q_j(θ_j) moves to κ: the matching-function helper, locals, keyword arguments,
NamedTuple and struct fields, and every reader of them. No equilibrium object is redefined
and every number reproduces bitwise, so the theorist does not need to be told about results.
The layer is MINOR, not PATCH, because identifiers that scripts call or read changed: a
function name, a keyword argument, two struct fields and one field of
`compute_equilibrium_objects`.

**Comparability.** Every number is bit-identical to v52.0.0: Q, the selected root, θ_S and all 43
moments at the four stored estimates, all 83 fields of `compute_equilibrium_objects`, the
transition smoke series and the `transition_audit` (TA_NULL) CSV. No serialised object stores a
renamed field or type (measured below), so no bundle, bank or chain is rebuilt and nothing in
`output/` was touched. Two writers now emit renamed CSV columns: `transition_audit.jl`
(`f_U`, `f_S` → `kappa_U`, `kappa_S`) and `write_policy_csv` (`f_U` → `kappa_U`). No code reads
either file. The stored `output/logs/transition_audit_fc*.csv` and
`output/tables/policy_results_pre-fc*.csv` keep their old headers; they are outputs and are not
rewritten.

### What changed
- **Matching technology** (`solver/grids.jl:12`): `jobfinding_rate(θ, μ, η)` → `meeting_rate`.
  θ q(θ) is the meeting rate; job finding is meeting × acceptance. All 19 call sites follow.
- **Solver** (rename only):
  - `solver/unskilled.jl`: the unskilled meeting rate `f` → `κ` in `unskilled_inner_loop!`
    (l.265–353; its returned field `f` → `κ`), `unemployment_split_unskilled` (l.400) and
    `solve_stationary_unskilled!` (l.442) arguments, `update_theta_unskilled` (`f_new` → `κ_new`,
    l.480), `solve_unskilled_block!` readers `inner.κ` / `inner_final.κ` (l.584–655).
  - `solver/skilled.jl`: keyword `fU` → `κU` in `skilled_inner_loop!` (l.298) and
    `solve_skilled_block!` (l.766). The skilled meeting rate `f` → `κ` (l.308, 446, 694–734), and
    the inner loop's returned field `f` → `κ` (l.428, read at l.796).
  - `solver/solver.jl`: `fU` → `κU` (`_mS_from_t` l.76, the global loop l.173–298), `inner.κ`
    (l.179). In `_classify` (l.517–519) the curvature local `κ` → `curv`, so the letter means
    one thing in the solver.
  - `solver/equilibrium.jl`: the unskilled meeting rate `f_U` → `κ_U`, and the returned field
    `f_U` → `κ_U` (l.511). `prob_f_U` → `prob_κ_U`. **`f_S` is kept** (l.330): it is
    (κ_S·Σ(1−Γ_o(p*_S))u_stay + κ_U·u_drain)/u_S, the skilled unemployed's job-finding hazard
    (meeting × acceptance), which is what f_j now means.
- **Transition** (`transition/`): struct `TransitionResult` fields `fU`, `fS` → `κU`, `κS`
  (`transition_params.jl:185`). `fU`/`fS`/`f_U0`/`fU_p`/`fS_p` and the local `f` → κ forms in
  `transition_solver.jl` and `transition_values.jl`. **`fU_hire` is kept**
  (`transition_solver.jl:191`): it is κ_U × the hired share of the cell's unemployed, a
  job-finding hazard.
- **Readers**: `smoke/smoke_transition.jl:69`, `tools/transition_audit.jl:235`,
  `plots_and_tables/transition_panel.jl:961`, `policy/policy_params.jl:217` (`PolicyResult.f_U`
  → `κ_U`; `f_S` kept), `policy/policy_solver.jl:106`, `plots_and_tables/policy.jl` (table rows,
  CSV column, the Δκ_U panel label and title, l.39–534).
- **Comments** that named a meeting rate f follow the rename. The κ_w scale of the retired
  enrolment bridge is named `enr_scale_w` (`smm/moments.jl:25–84`, `smm/smm_main.jl:39`).
  Three comments that still described that bridge as live now state what the code does:
  `smm/smm_main.jl:39–42`, `data_and_descriptives/sigma.jl:6–8` and `validation.jl:7`. No code
  variable carried κ_w (removed in v17.0).
- **Version** `smm/version.jl:11`.

### Measured, not asserted (scratch copies of v52.0.0 and v52.1.0, 4 threads per process, bundle grids)
| estimate | root | Q v52.0.0 | Q v52.1.0 | bitwise | moments equal | objects equal |
|---|---|---|---|---|---|---|
| base_fc | U | 3560.0920827909513 | 3560.0920827909513 | yes | 43 / 43 | 83 / 83 |
| crisis_fc | U | 1997.8722326121679 | 1997.8722326121679 | yes | 43 / 43 | 83 / 83 |
| base_covid | U | 3935.5499739559359 | 3935.5499739559359 | yes | 43 / 43 | 83 / 83 |
| crisis_covid | U | 2032.0733666850374 | 2032.0733666850374 | yes | 43 / 43 | 83 / 83 |

The estimate bundles used were byte-identical to the repo copies at the time of the
comparison (sha256 prefixes 92e3da9b…, 4c56efbb…, 23c2ae2b…, 3e5e2fa1…). Objects were compared
with `isequal`, under the key map f_U → κ_U.
- Bundle check. Every `.jls` under `output/estimates`, `output/chains`, `output/smm` was
  deserialised and walked, collecting every struct field name, NamedTuple key, Dict key,
  Symbol and Function. The search covered f_U, fU, f_S, fS, f, κ*, kappa, jobfinding_rate and
  meeting_rate. There was no hit in the four estimates, the four candidate banks or the
  chain, whose types are SMMResult, SMMSpec, ParamSpec, RunProvenance and SeedBank. 20 legacy
  files do not deserialise under v52.0.0 either (TypeError, pre-existing). Neither
  `TransitionResult` nor `PolicyResult` occurs in any `.jls` (byte scan).
  `output/transition/transition_fc_fullW.jld2` (2026-04-05) holds an older TransitionResult
  layout (`n_iterations`, no θU/θS), and JLD2 is not a project dependency.
- `smoke/smoke_transition.jl`: identical output except the two series names.
  `tools/transition_audit.jl` with `TA_NULL=true`: the CSV body is byte-identical and the header
  carries `kappa_U,kappa_S`.
- Every changed file parses. `smm_main`, `MCMC_main` and `data_processing_main` load their
  include chains as shipped. `transition_main` and `model_main` load with a Plots stub, because
  Plots does not precompile in the sandbox. `model_main` also needs `using Optim` supplied, since
  `smm.jl:2092` needs it and `model_main.jl` does not import it. `policy_main` stops at
  `policy_params.jl:109` (`RegimeParams`, retired). Every outcome is identical under v52.0.0.
- Gates, identical to v52.0.0: `check_threading` (2 threads) serial = threaded Q
  3560.0920827910 at base_fc; `check_hire_margin` 0 disagreements at all four estimates;
  `check_dead_settings`, `check_output_tree` pass; `check_forwarding` reports only the known
  `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` finding (so `check_repo.sh --verify-only` exits 1 as it did);
  `check_structure` flags the same 7 files; `check_version_sync` in sync at 52.1.0.

### Settings
None.

### Known gaps
- `plots_and_tables/transition_panel.jl:1040` plots κ_U, κ_S under the panel title
  "Job-finding rate", and `_compute_sep` (l.975) uses them as the job-finding rate in its
  approximate flow identity. Both were left as they were: the label and the approximation are
  for the author to decide.
- `plots_and_tables/transition.jl:333–335` tags the moments `jfr_U`/`jfr_S` "fU"/"fS". These
  are moment labels and f is the job-finding hazard, so they stay.
- `model_main.jl` does not load as shipped (no `using Optim` for `smm.jl`). This predates the
  change and was not touched.
- Legacy notebooks in `code/notebooks/` define their own `jobfinding_rate`, or include files
  that no longer exist. They are not part of any entry point and were not edited.
- The theorist updates `latex/` in parallel; nothing under `latex/` was edited.

### Tried and rejected
Nothing.

## v52.0.0 — 2026-10-07

**MAJOR: the seven rate moments are built as the data stage builds them.** Every model-side
rate moment is now the mean transition probability over that moment's universe, converted as its
data source converts it, instead of a mean of type hazards. For `jfr_U` this also changes what is
counted: the CPS U→E rate counts hires over all unskilled unemployed, and the moment was the
meeting rate f_U, which counts a meeting with a type no firm hires. The solver, the model and
every object the solve computes are untouched (author's condition: only the model moment
computation changes). Moments, hence Q, change at every estimate, so the theorist must be told.

**Comparability.** No parameter changed, so every stored bundle loads unchanged and needs no
rebuild, and every stored estimate is still a valid parameter point. Every stored Q is
superseded as a measure of fit: the stored `loss_opt` in each bundle, and the
`smm_estimates_*.csv` / `mcmc_results_*.csv` reports, are v51 moment values. Every window needs
re-estimation under v52 before its fit is compared with anything. At each of the four estimates
there is a single certified root (U), so the selected equilibrium is the same under both moment
sets.

### What changed
- **Post-solve layer only** (`solver/equilibrium.jl`, `compute_equilibrium_objects`).
  - `_hazard_of_mean_probability(prob, mass, empty)` (l.38): −log(1 − p̄), p̄ the mass-weighted
    mean of the cells' monthly probabilities; `_MONTHS_PER_QUARTER = 3.0` (l.32).
  - The E-to-E loop accumulates the searching share's quarterly move probability
    s·e·(1 − e^{−3κ_S(1−Γ_o(p))}) beside the existing flow (l.406).
  - New block "Rate moments as the data stage measures them" (l.420–446) and one new returned
    field, `measured`, a NamedTuple keyed by moment name (l.516). Per moment:
    - `jfr_U`: hired_U(a_U)·(1 − e^{−f_U}) over the unskilled-unemployed rows (`uU_rowU`). The
      never-hired branch of a row is unemployed and searching but is never hired.
    - `jfr_S`: stayers 1 − e^{−κ_S(1−Γ_o(p*_S))} over `uS_d0mass`, crossers 1 − e^{−f_U} over
      `uS_d1mass`.
    - `sep_rate_U/S`: 1 − e^{−δ_j(a)} over `eU_rowU` / `eS_totS`. `wchg_rate_U/S`:
      1 − e^{−λ_j(1−G_j(p*))} over the same masses. All four are converted by −log(1 − p̄), as
      `data_and_descriptives/moments.jl` (CPS) and `sipp.jl` (`_sipp_hazard`,
      `_bbg_cell_hazard`) convert theirs.
    - `ee_rate_S`: the quarterly probability q̄ over the employed skilled, reported as
      1 − (1 − q̄)^{1/3}, the conversion `j2j.jl:70` applies to EEHire/MainB.
  - A cell split by a covered fraction (the hired share of a row's unemployed, the searching
    share of an employed cell) is a mixture: share × the moving branch's probability.
  - Every existing field (`f_U`, `f_S`, `sep_rate_U/S`, `ee_rate_S`, `wchg_rate_U/S`, ...) keeps its
    meaning and its value, because the notes, plots, transitions and policy code read them.
- **`smm/moments.jl`, `model_moments`** (l.521–531): the seven rate moments read
  `obj.measured`. The `hasproperty` fallbacks on `wchg_rate_U/S` are gone, since every object
  carries the field.
- **Untouched, byte-verified against v51.0.1**: `solver/grids.jl`, `params.jl`, `unskilled.jl`,
  `skilled.jl`, `solver.jl`, and every file outside the two above.
- Backup of the pre-change files, byte-verified: `output/backups/pre_rate_moments_2026-10-07/`.

### Measured (scratch copies of v51.0.1 and v52.0.0, 4 threads per process, bundle grids, `select_equilibrium` at each stored estimate)
The solve is bit-identical. "Solve fields" are θ_U, θ_S, p*_U, p*_S, p^oj, s_part, d, every mass
array (u_U, t, e_U, m_U, m_S, u_S, e_S, row and column masses, the e_U surface), the aggregates,
f_U, f_S, the existing hazard fields, the survivors and the wage densities, compared with
`isequal`:

| estimate | root | θ_U | θ_S | solve fields bitwise equal | non-rate moments bitwise equal |
|---|---|---|---|---|---|
| base_fc | U | 0.458752030675076 | 1.21036198933046 | 35 / 35 | 36 / 36 |
| crisis_fc | U | 0.225767401330156 | 0.646112536469146 | 35 / 35 | 36 / 36 |
| base_covid | U | 0.830743278223685 | 1.84410585649858 | 35 / 35 | 36 / 36 |
| crisis_covid | U | 1.03733094318729 | 2.32079704846696 | 35 / 35 | 36 / 36 |

Rate moments (every other moment is bitwise unchanged):

| estimate | moment | v51.0.1 | v52.0.0 | change | data |
|---|---|---|---|---|---|
| base_fc | jfr_U | 0.3042998787 | 0.2549790924 | -1.621e-01 rel | 0.302524 |
| base_fc | jfr_S | 0.2764928892 | 0.2764928892 | -2.220e-16 rel | 0.320369 |
| base_fc | sep_rate_U | 0.0148374406 | 0.0147476682 | -6.050e-03 rel | 0.015251 |
| base_fc | sep_rate_S | 0.0050249718 | 0.0050249718 | -2.220e-16 rel | 0.006241 |
| base_fc | ee_rate_S | 0.0093876771 | 0.0065374186 | -3.036e-01 rel | 0.007969 |
| base_fc | wchg_rate_U | 0.1567947176 | 0.1567003790 | -6.017e-04 rel | 0.150512 |
| base_fc | wchg_rate_S | 0.1584881279 | 0.1584881279 | -3.331e-16 rel | 0.129177 |
| crisis_fc | jfr_U | 0.2122322121 | 0.1966506601 | -7.342e-02 rel | 0.210655 |
| crisis_fc | jfr_S | 0.2010450075 | 0.2010450075 | +4.441e-16 rel | 0.237466 |
| crisis_fc | sep_rate_U | 0.0198976174 | 0.0198447410 | -2.657e-03 rel | 0.020563 |
| crisis_fc | sep_rate_S | 0.0064577500 | 0.0064577500 | +2.220e-16 rel | 0.008171 |
| crisis_fc | ee_rate_S | 0.0070309798 | 0.0053832277 | -2.344e-01 rel | 0.006150 |
| crisis_fc | wchg_rate_U | 0.1124371956 | 0.1123825601 | -4.859e-04 rel | 0.103098 |
| crisis_fc | wchg_rate_S | 0.1123775397 | 0.1123775397 | +4.441e-16 rel | 0.091078 |
| base_covid | jfr_U | 0.3713716405 | 0.1572955872 | -5.764e-01 rel | 0.296387 |
| base_covid | jfr_S | 0.2462842390 | 0.2462842390 | -2.220e-16 rel | 0.316990 |
| base_covid | sep_rate_U | 0.0082032135 | 0.0080120434 | -2.330e-02 rel | 0.013278 |
| base_covid | sep_rate_S | 0.0039170296 | 0.0039170296 | +8.882e-16 rel | 0.005919 |
| base_covid | ee_rate_S | 0.0006668503 | 0.0004726203 | -2.913e-01 rel | 0.008508 |
| base_covid | wchg_rate_U | 0.2059929260 | 0.2057865086 | -1.002e-03 rel | 0.208465 |
| base_covid | wchg_rate_S | 0.2282015549 | 0.2282015549 | -2.220e-16 rel | 0.222416 |
| crisis_covid | jfr_U | 0.3212887626 | 0.2718833275 | -1.538e-01 rel | 0.314352 |
| crisis_covid | jfr_S | 0.2485890160 | 0.2485890160 | +2.220e-16 rel | 0.319650 |
| crisis_covid | sep_rate_U | 0.0198911693 | 0.0196219841 | -1.353e-02 rel | 0.021155 |
| crisis_covid | sep_rate_S | 0.0074488333 | 0.0074488333 | -2.220e-16 rel | 0.010383 |
| crisis_covid | ee_rate_S | 0.0004545586 | 0.0003208037 | -2.943e-01 rel | 0.009151 |
| crisis_covid | wchg_rate_U | 0.2379992337 | 0.2377099449 | -1.216e-03 rel | 0.236831 |
| crisis_covid | wchg_rate_S | 0.2396000500 | 0.2396000500 | +6.661e-16 rel | 0.239340 |

Q at each stored estimate:

| estimate | stored loss_opt | Q v51.0.1 | Q v52.0.0 | Q with the /tmp/nm scratch's jfr_U and ee_rate_S |
|---|---|---|---|---|
| base_fc | 2957.413051 | 2957.413051 | 3560.092083 | 3557.734669 |
| crisis_fc | 1923.728847 | 1923.728847 | 1997.872233 | 1997.805302 |
| base_covid | 8228.151227 | 8228.151227 | 11006.443827 | 10974.612779 |
| crisis_covid | 1816.503405 | 1844.894080 | 2032.073367 | 2029.576186 |

Q contributions of the active rate moments:

| estimate | moment | Q contribution v51.0.1 | v52.0.0 |
|---|---|---|---|
| base_fc | jfr_U | 0.832 | 596.271 |
| base_fc | jfr_S | 82.298 | 82.298 |
| base_fc | sep_rate_U | 16.158 | 23.930 |
| base_fc | sep_rate_S | 150.094 | 150.094 |
| base_fc | ee_rate_S | 7.733 | 7.880 |
| base_fc | wchg_rate_U | 22.773 | 22.094 |
| base_fc | wchg_rate_S | 70.256 | 70.256 |
| crisis_fc | jfr_U | 0.925 | 72.905 |
| crisis_fc | jfr_S | 81.001 | 81.001 |
| crisis_fc | sep_rate_U | 17.197 | 20.038 |
| crisis_fc | sep_rate_S | 141.503 | 141.503 |
| crisis_fc | ee_rate_S | 0.915 | 0.692 |
| crisis_fc | wchg_rate_U | 39.001 | 38.546 |
| crisis_fc | wchg_rate_S | 34.623 | 34.623 |
| base_covid | jfr_U | 1056.269 | 3634.339 |
| base_covid | jfr_S | 217.054 | 217.054 |
| base_covid | sep_rate_U | 2187.854 | 2355.780 |
| base_covid | sep_rate_S | 455.819 | 455.819 |
| base_covid | ee_rate_S | 601.431 | 631.595 |
| base_covid | wchg_rate_U | 12.267 | 14.401 |
| base_covid | wchg_rate_S | 38.504 | 38.504 |
| crisis_covid | jfr_U | 4.651 | 174.318 |
| crisis_covid | jfr_S | 161.167 | 161.167 |
| crisis_covid | sep_rate_U | 36.408 | 53.570 |
| crisis_covid | sep_rate_S | 282.112 | 282.112 |
| crisis_covid | ee_rate_S | 25.406 | 26.194 |
| crisis_covid | wchg_rate_U | 1.008 | 0.571 |
| crisis_covid | wchg_rate_S | 0.041 | 0.041 |

Two of the scratch's constructions were changed (see Tried and rejected). The scratch's shared
moments (`jfr_S`, `sep_rate_U/S`, `wchg_rate_U/S`) are reproduced to ≤ 1e-14 relative. With its
`jfr_U` and `ee_rate_S` substituted, Q reproduces the scratch run (3557.734669, 1997.805302,
10974.612779) to ≤ 2.8e-15 relative. Moments unchanged to rounding (`jfr_S`, `sep_rate_S`,
`wchg_rate_S`) have a skilled hazard that is constant over the universe at these estimates
(Γ(p*_S) = 0 wherever the skilled hold mass, and the drained mass is 0), and the mean probability of a constant hazard converts back to it.

Release gate: both changed files parse. The include chains of `smm_main.jl`, `MCMC_main.jl`,
`transition_main.jl` (through `transition_simulation.jl`) and `model_main.jl` (through §solver
and §SMM) load, and every name the edits call resolves. `using Plots` cannot precompile GR_jll
in the sandbox depot, so the plotting includes were not load-tested. Gate results:
- `check_threading.jl -t 2`: serial and threaded Q = 3560.0920827910 at base_fc, bitwise equal.
- `check_hire_margin.jl`: 0 disagreements at all four estimates.
- `check_dead_settings`, `check_output_tree`: pass.
- `check_forwarding`: the known v49 baseline finding only (`ROYSEARCH_MCMC_JAC_ALLOW_GAPS`).
- `check_structure`: the same 7 files as v51.0.1.
- `check_version_sync`: in sync at 52.0.0.

### Settings
None.

### Known gaps
- Convention shared by every rate moment: each cell keeps its hazard over the measurement
  interval (a month, or a quarter for J2J), with no state change inside it. This is the reading
  under which the data stage's −log(1−p) is a hazard. A hire followed by a separation inside the
  month, a λ redraw, an exit or a second move inside a J2J quarter are ignored.
- The SIPP universe is same-job stayer pairs. The model weights the employed at the start of
  the month, which differs only by the type-specific probability of staying in the match.
- The J2J denominator caveat in Data and Moments (section 'Data–model definitional gaps', item 2) is unchanged.
- crisis_covid: the stored `loss_opt` 1816.503405 is not the v51.0.1 re-evaluation
  1844.894080. That gap predates this change (recorded at v50.0.0).
- Smoke tests not run: `smoke warmstart` calls `_run_sa` and `smoke clusters` builds a Sobol
  bank. Neither path is touched; no parameter or spec changed.
- `plots_and_tables/transition.jl:333` labels `jfr_U` "fU" in the fit table, which now shows
  the measured moment. That is a label only, and was not changed.
- `transition/transition_solver.jl` (`_skilled_margin_masses` docstring) says the moment layer
  reads `sep_rate_S` at the node; `equilibrium.jl` integrates the cell masses at the cutoff
  (`cdf_at_cutoff`). The docstring is stale, predates this change, and was not touched.
- The notes (`latex/notes_delivery_2026-10-05/data_and_moments.tex`) still describe the
  hazards as the model moments. Each such sentence is listed in `rate_moments_change.md` for
  the theorist.

### Tried and rejected
- **Per-row hazard f_U·hired_U for `jfr_U`** (the scratch at `/tmp/nm`). hired_U is the share
  of a row's unemployed in the hired branch, and the rest has hazard 0. The row's probability is
  therefore hired·(1 − e^{−f_U}), not 1 − e^{−f_U·hired}, which overstates it by Jensen.
  - Only the row straddling a^h has 0 < hired < 1 (rows 26, 25, 29, 20 in the four windows).
  - Measured gap at base_fc / crisis_fc / base_covid: `jfr_U` 3.5e-4 / 1.4e-4 / 2.0e-4
    relative, ΔQ 2.211 / 0.290 / 1.667.
- **`ee_rate_S` left as the hazard** (the scratch). The target is the monthly probability
  1 − (1 − ee^Q)^{1/3} of a quarterly rate. Mirroring it lowers `ee_rate_S` by 23–30%, not by
  the half percent the notes cite for the data-side conversion.
  - The flow is concentrated: 3.6% of base_fc skilled employed search on the job, at a mean
    hazard of 0.263 per month, so their quarterly move probability is 1 − e^{−0.79} = 0.55
    against 0.79 for the linear reading (base_covid: 0.27% at 0.245).
  - ΔQ against the hazard: +0.147 base_fc, −0.223 crisis_fc, +30.164 base_covid.

## v51.0.1 — 2026-10-05

**PATCH: comments only.** Four comments used wording the paper does not use or attributed a
setting to a source it does not come from; no code line changed.

**Comparability.** Every number is bit-identical to v51.0.0: only comment lines changed.

### What changed
- `solver/params.jl` header: "Market separation is by comparative advantage (Roy 1951; Gola
  2016, 2021)" becomes "Both markets are populated through comparative advantage (Roy 1951;
  concordance as in Gola 2021)". Gola 2016 is an unpublished earlier version of Gola 2021.
- `smm/smm_main.jl`, `LAMBDA_W` comment, and `smm/smm_params.jl`, `λ_w` field comment: the
  attribution of λ_w to Bound–Krueger (1991) is removed. 0.78 is set by the author as the middle
  of the range the wage-validation literature reports (Data and Moments, Observed wages and
  measurement error).
- `transition/transition_solver.jl` header and `solve_transition` docstring: "unanticipated
  permanent shock" becomes an unanticipated change in the parameters at date T, the notes'
  wording.

### Measured, not asserted
The four files parse (`Meta.parseall`). No solve was run: a comment cannot change a number.

### Settings
None.

### Known gaps
Nothing.

## v51.0.0 — 2026-10-04

**MAJOR: a crisis window estimates unemployment income.** Up to v50.0.1 a crisis window held
b_U, b_T and b_S at the estimate of its paired baseline window, together with the ability
distribution (a_ℓ, b_ℓ, ρ_x). It now holds only the ability distribution; b_U is estimated in
every window, and b_T and b_S stay pinned at zero through FIX_PARAMS (author's decision). A crisis
window has 13 free parameters, a baseline window 14.

**Comparability.** Baseline windows are unchanged. No crisis window had been estimated under
v50, so no stored estimate is superseded; the stored crisis bundles predate v50 and their
candidate banks are stale (the free set changed), so a crisis run starts from `:clusters`.

### What changed
- `smm/smm_params.jl`: `REGIME_SPECIFIC_PARAMS`, `DEEP_PARAMS` and `DEEP_PARAMS_ORDERED` are
  replaced by one tuple, `ABILITY_PARAMS = ((:common, :a_ℓ), (:common, :b_ℓ), (:common, :ρ_x))`.
  The file header no longer classifies parameters.
- `smm/smm_main.jl`, §6: the crisis branch pins `ABILITY_PARAMS` at the baseline optimum and offers
  every other parameter to `build_smm_spec`, which drops what FIX_PARAMS pins. Header, FIX_PARAMS
  key list (now grouped by solver block) and run-log lines follow.
- `plots_and_tables/transition.jl`: the parameter tables read which parameters are estimated off
  each bundle's spec, through one `PARAM_DISPLAY` map, instead of two hand-kept lists.
- `solver/model_main.jl`, `solver/params.jl`, `smoke/`: labels and comments.

Checked on a scratch copy: the crisis_fc spec builds with the 13 free parameters above, the
ability distribution pinned at the base_fc estimate, and `assert_all_params_accounted` passing.

### Settings
None.

## v50.0.1 — 2026-10-04

**PATCH: console output of `MCMC_main.jl` only.** Every number, every CSV column and every
bundle field is unchanged; the JAC_ONLY table on base_fc is digit-for-digit the v50.0.0 one.

**Comparability.** Full. No stored output is superseded.

### What changed (`code/mcmc/MCMC_main.jl`, §6 and §8)
- The Ĝ diagnostics (FD scheme, step/box width, Richardson instability, cond(G̃), pinv rank)
  print above the summary box on both paths.
- JAC_ONLY: the box is titled "Standard errors from Ĵ = Ĝ'WĜ" and holds only the table. The
  chain lines (gens, seed drift, logπ, gates, target, Ĵ source) print only when a chain ran.
  The footer is two lines; the "No promotion" line is not printed, since there is no chain.
- Chain path: the table is the JAC_ONLY table plus se(chain) beside the estimate and |t|(chain)
  after se(bound). R̂ and the edge test stay in the trailing flag, and the attained R̂ max and
  ESS min print against the gates in the box header. The estimate column is the reported
  estimate (posterior mean once the chain converged, θ̂ otherwise), as |t| already used.
- The box rules are drawn to the table's width.

### Measured, not asserted
JAC_ONLY on a scratch copy of the repo at the base_fc estimate: 166 solves, 14 of 14
coordinates, the table identical to the v50.0.0 run's. Chain-path layout checked by rendering
the row format with representative values (no chain was run).

### Settings
None.

## v50.0.0 — 2026-10-03

**MAJOR: the unit of the unskilled vacancy cost k_U changes.** The posting cost is k_U·π̄_U, and
π̄_U is now the mean output of a new unskilled match, exp(A)·P_U·E_ℓ[a_U]. The factor
E_G[p] = α_U/(α_U+1), the mean of the shock redraw, is removed: unskilled matches start at
p = 1. This is the LMR (2016) convention, where the posting cost is c times the mean of
f(x, y) over independent draws of worker and job type (their `solver.f90:89`). π̄_S already
followed it, because a skilled match starts from a draw of the offer distribution Γ. k_U's unit
no longer moves with α_U, which is free and regime-specific (author's decision).

**Comparability.** The change is an exact reparameterisation: k_U(v50) = k_U(v49)·α_U/(α_U+1)
gives the identical equilibrium. Every stored bundle was rebuilt once, by hand, into the new
unit, so stored Q values are unchanged and v50 bundles are read with no conversion code.
- k_U values in v49 bundles, reports or notes are **not** comparable to v50 values without the
  factor α_U/(α_U+1) at that point's α_U.
- The `smm_estimates_*.csv` and `mcmc_results_*.csv` reports in `output/estimates` still carry
  v49-unit k_U. They are reports, not inputs, and were not rewritten.
- A bundle written by a v49 process after this change carries v49-unit k_U and a 49.0.0
  provenance stamp.

### What changed
- **Normaliser** (`solver/grids.jl`, `mean_output_U`): E_G[p] removed. The block comment states
  both conventions. The only call site is unskilled free entry (`solver/unskilled.jl:537`);
  `mean_output_S` is unchanged.
- **Bound** (`smm/smm_params.jl`, `default_free_params()`): k_U lb 0.05 → 0.025, so the box holds
  every rebuilt candidate. The v49 floor of 0.05 maps to 0.025 at α_U = 1. ub is unchanged at 12.
- **Comments and labels**: `solver/params.jl` (k field), `smm/smm_params.jl` (vacancy-cost
  block and k_U/k_S labels), `solver/unskilled.jl` (free-entry header).
- **Stored bundles rebuilt once by a scratch script outside the repo** (`/tmp/kU/convert.jl`):
  k_U → k_U·α_U/(α_U+1) with each bundle's own α_U, free or pinned. Each value was converted in
  natural units and re-expressed in that bundle's box. Bundles on the live v49 box [0.05, 12]
  take [0.025, 12]; older boxes keep their numbers. Only the k_U coordinate moves. The script
  also converted `result.params_opt.unsk_k`, the k_U seed in `spec.free`, candidate banks and
  chain draws, and restamped provenance to 50.0.0 (`migrated_from` gains 49.0.0).

| file | k_U v49 → v50 | α_U |
|---|---|---|
| estimate_base_fc | 3.67840 → 1.83920 | 1 (pinned) |
| estimate_crisis_fc | 1.15197 → 0.88758 | 3.357 |
| estimate_base_covid | 0.86276 → 0.71155 | 4.706 |
| estimate_crisis_covid | 0.50610 → 0.34521 | 2.146 |
| candidates_base_fc (1725) | → [0.02976, 5.99844] | 1 (pinned) |
| candidates_crisis_fc / base_covid / crisis_covid | → [0.00807, 10.192] / [0.00327, 10.152] / [0.00807, 10.192] | per candidate |
| chains/chain_base_fc (180000 draws) | → [0.54115, 0.69927] | per draw |

  24 older side bundles (`*_backup_Q*`, `_postmean`, `_pre_xiU_restore*`, `_rejected*` and every
  `*.jls.pre_migration`) already failed to deserialise under v49. They were not touched.
- Backup of the nine rebuilt files, byte-verified: `output/backups/pre_kU_unit_2026-10-03/`.

### Measured, not asserted
Q at each stored estimate, v50 code on the rebuilt bundle against v49 code on the original:

| estimate | Q v50 | Q v49 | relative difference |
|---|---|---|---|
| base_fc | 16385.802626441855 | 16385.802626441819 | 2.2e-15 |
| crisis_fc | 710.081449797087 | 710.081449797083 | 5.1e-15 |
| base_covid | 2884.581528455633 | 2884.581528455622 | 3.8e-15 |
| crisis_covid | 1844.894079700002 | 1844.894079700002 | 0 |

The stored `loss_opt` already differed from the v49 re-evaluation for crisis_fc and both covid
windows (2.8e-4, 1.8e-2, 1.6e-2). That gap predates this change and is unchanged by it.

### Settings
None.

## v49.0.0 — 2026-10-02

**MAJOR: five parameter boxes move, and the parameter table no longer carries starting values.**
The box is part of the estimation problem: ρ_x was restricted to [−1, −0.1], which ruled out
zero and positive concordance a priori, and it is now [−0.98, 0.90]. A re-estimation under
v49.0.0 therefore solves a different problem from v48.0.0 (author's decision). The removal of
the `smm_params.jl` starting values is MINOR on its own. It changes no seed that reaches any
current run, and the measurements below show that bit for bit.

**Comparability.** Every stored v48.0.0 number stays valid. Q at the four estimates is
bit-identical, because each bundle carries its own spec and box. All four estimates lie
strictly inside the new boxes. What is superseded is the estimation set-up:
- All four `candidates_<w>_diagonalW.jls` banks are stale. Their cache key includes lb/ub, so
  the next `:clusters` run regenerates them, at 2048 Sobol points per window.
- A v49.0.0 re-estimate is not comparable to a v48.0.0 one where the old box bound.
- MCMC run from a stored bundle is unaffected, since it reads the bundle's own box. A chain on
  a bundle written under v49.0.0 lives in the new box.

### What changed
- **Bounds** (`smm/smm_params.jl`, `default_free_params()`). Exactly these five; every other
  lb/ub is unchanged.
  - ρ_x: [−1, −0.1] → [−0.98, 0.90]. The comment now states the reason for each end of the box.
  - P_S: ub 3.257928 → 5.0.
  - σ_S: ub 0.244345 → 0.5.
  - k_U and k_S: lb 0.0005 → 0.05.
- **No starting values in the table.** `ParamSpec(block, name, lb, ub, label)` is a new
  5-argument constructor that sets `init = NaN`, and the table is written with it.
  - The `ParamSpec` struct is unchanged, so no bundle migrates.
  - `with_seed(ps, value)` (`smm_params.jl`) writes a seed, clamped 1e-10 inside the box. It is
    now the only place that clamp is written. `perturb_free_init` and every seeding site in
    `smm_main.jl` use it.
- **`pack_θ` raises on a non-finite init, naming `block:name`.** A path that forgets to seed
  fails at its first use instead of starting from a value nobody chose.
- **`unpack_θ` has no fallback values.** It resolves each field in this order: the
  block-qualified pinned key, then the bare pinned key, then the free value. A field in none of
  these raises, naming `block:name`.
  - Free values are keyed like `fixed_key`: bare for a name unique to one block, qualified for
    μ, η, k, β, λ, ξ, σ_w. The function is now one pass. The second pass, which repaired the
    shared names, is gone.
  - That second pass applied `_to_constrained` even under `constrained = true`. So under
    `MCMC_SPACE = :theta`, the shared-name coordinates were box-mapped twice. The path is
    unreachable as shipped (MCMC_main guards `:theta` off), but it is one candidate for part of
    the 1.66e+07 Q recorded at that guard. That has not been measured.
- **`smm_main.jl` seeds every path from `DEFAULT_PARAMS` or a stored optimum.**
  `default_seed(ps)` looks the value up through `_DEFAULT_PARAM_KEY` and raises, naming the
  parameter, when either map lacks it.
  - Crisis window, non-warmstart: `default_seed`.
  - Baseline window: every candidate is seeded from `DEFAULT_PARAMS` first. A readable
    warm-start bundle then overwrites the parameters it estimated. This also closes two silent
    paths that fell back to the table literals: an unreadable warm-start file, and one that
    could not be decoded.
  - The message reads `Unmatched (seeded from DEFAULT_PARAMS)`.
  - §8b baseline-optimum injection: a strict lookup in `baseline_optimum_values`. The crisis
    free set is a subset of it by construction, so a miss is a bug, not a case for a default.
  - §8b `INCLUDE_PREV_OPTIMUM` keeps the stored t when the box is unchanged. After a bound
    change it carries the prior optimum over as θ and re-packs it in the current box. Without
    this, the stored t would decode to a different point. At the base_fc optimum it would give
    ρ_x −0.5210 for −0.7803, P_S 3.8166 for 2.4868, σ_S 0.3126 for 0.1527, k_U 0.7992 for
    0.7528 and k_S 3.0406 for 3.0034. The remap reproduces θ to ≤ 2.2e-16 relative.
- **Text only.**
  - `print_spec`'s column header is now `seed`.
  - Comments and error text that described a hard-coded fallback in `assert_all_params_accounted`,
    §6 and §8 are updated.
  - The ξ_U provenance line reads "pinned in the baseline spec".
  - `smoke_warmstart.jl` no longer prints the −0.55 fallback.
- **Smoke runner** (`code/smoke/run_smoke.sh`, `README.md`). It now sets `ROYSEARCH_WINDOW` and
  `ROYSEARCH_INIT_MODE`. Its old text substitutions targeted literals that no longer exist,
  so as left it would have run `base_fc` under `:clusters` with the full 2048-point bank. The
  grid regex now matches any integer. The README states that `clusters` must run in a scratch
  copy (Known gaps).

### Measured (scratch copies of v48.0.0 and v49.0.0; v49 also with the old boxes restored)
- **A. Q at the four stored estimates, through each bundle's own spec**, bit-identical: v48
  run twice, then v49.

  | window | Q (v48.0.0 = v49.0.0) |
  |---|---|
  | base_fc | 1059.5806221932073 |
  | crisis_fc | 710.0814497970832 |
  | base_covid | 2884.581528455622 |
  | crisis_covid | 1844.8940797000023 |

  - These equal the v48.0.0 entry's values.
  - Every field `unpack_θ` returns for the four bundles is identical between trees.
- **B. Seeds, through `smm_main.jl`'s own construction (truncated before §8b).**
  - Run for 4 windows × {`:clusters`, `:warmstart`}, with 18 / 13 / 18 / 13 free parameters.
  - With the boxes held fixed, `pack_θ(spec)` and every init are bit-identical in all 8 cases.
  - No warm-start parameter fell back to a table literal. The six unmatched candidates (unsk:bT,
    unsk:PU, unsk:η, skl:η, unsk:σ_w, skl:σ_w) are all pinned by FIX_PARAMS or the λ_w
    calibration, so none reaches `spec.free`.
  - Under the new boxes the constrained seeds are again bit-identical in all 8 cases. t moves
    only on the re-boxed free coordinates: 5 in the baseline windows, 4 in the crisis windows,
    where ρ_x is pinned. `Clamped into current bounds` fires in no window.
- **Estimates against the new boxes.** All are strictly inside. The only value on any bound is
  b_T = 0, which is pinned and whose box is unchanged.
  - ρ_x: −0.7803 (FC), −0.2465 (COVID).
  - P_S: 2.487, 2.689, 2.832, 3.097.
  - σ_S: 0.1527, 0.1458, 0.0174, 0.0705.
  - k_U: 0.753, 1.152, 0.863, 0.506.
  - k_S: 3.003, 5.408, 0.528, 0.485.
- **C. Guards.**
  - `pack_θ(default_free_params())` raises and lists all 29 parameters.
  - One NaN seed raises: `pack_θ: no starting value for common:ρ_x`.
  - A spec with ρ_x neither free nor pinned raises in `unpack_θ`: `common:ρ_x is neither free
    nor pinned`.
  - Freeing skl:μ while unsk:μ is neither free nor pinned raises, naming unsk:μ. The other
    block's copy is not borrowed.
- **D. Bundles.** Deserialisation status is identical in both trees.
  - The four estimate bundles and the four candidate banks load.
  - Ten `estimate_base_fc_diagonalW_*` side bundles fail, as before, with `TypeError: in new,
    expected Int64, got a value of type Float64`. These are the seven `backup_Q*`, plus
    `postmean`, `pre_xiU_restore_Q4960.615185` and `rejected_Q27002.469793`.
- **E. Smoke tests** (crisis_covid, scratch copy): `warmstart`, `guard` and `clusters` all
  PASS.
  - The guard message now reads "unpack_θ has no value for common:ρ_x".
  - `clusters` regenerated its bank.
- **Gates.**
  - `check_forwarding`: the pre-existing `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` finding only.
  - `check_threading` and `check_hire_margin`: OK.
  - `check_version_sync`: in sync at 49.0.0.
  - `MCMC_main.jl` loads up to the DE-MC launch for base_fc (23 free) and crisis_fc (18 free).
    Its spec builds, `unpack_θ` decodes and logπ is finite at the seed. No chain was run.

### Settings
None added, retired or re-defaulted.

### Known gaps
- `FIX_PARAMS` carries `:a_gam` / `:b_gam`, but the ASCII keys are `:a_Gam` / `:b_Gam`. Every
  run warns `unrecognised key … ignored`, so a_Γ and b_Γ are estimated, not pinned at 1. This
  predates this version and is left to the author, who owns FIX_PARAMS.
- `policy/policy_main.jl:178` destructures `unpack_θ` into four values. It returns three, so
  that line raises. This predates this version and is not touched.
- `smoke_clusters` writes a 32-point bank to `output/estimates/`. `_meta_matches` does not
  compare `n_sample`, so a later estimation would accept that bank. Run it in a scratch copy.
- `run_smoke.sh` is tracked as mode 100644. Invoke it as `bash run_smoke.sh <test>`.
- The stored `loss_opt` of the four bundles predates v48.0.0 and differs from the Q they
  re-evaluate to (for example 1046.76 against 1059.58 at base_fc). This is the v48.0.0
  respecification, not this change.

### Tried and rejected
- **Removing `ParamSpec.init`.** The field is the seed carrier for `pack_θ`, `perturb_free_init`
  and `_spec_with_init`, and it is serialised in every bundle. Removing it is the silent
  "remove a field, types match" case for some layouts, and it would migrate every bundle for no
  change in behaviour.
- **A `DEFAULT_PARAMS` fallback at the §8b baseline injection.** It would put a non-baseline
  vector into the bank under the label "baseline optimum". A strict lookup is correct.
- **Rejecting a prior optimum whose box differs.** That discards a valid start. Re-packing it as
  θ keeps the start, bit-faithful to 2.2e-16.

---

## v48.0.0 — 2026-10-01

**MAJOR: every unskilled entry object is evaluated at p = 1, not at the top quadrature node.**
A new unskilled match starts at the top of the quality support, p = 1. The unskilled p grid is
Gauss–Legendre (Np_U = 160), so its top node is p_N = 1 − 5.612e-5, and v47.0.0 read the entry
surplus off the grid, `Svec[end] = S_U(a_U, p_N)`. That understated S_U(a_U,1) by
A P_U a_U (1 − p_N)/(r+ν+λ_U+ξ_U) for every hired type. It also gave zero surplus to the band of
hired types a^h < a_U < a^h/p_N, since `_soft_weight` returns 0 on the top cell. The entry
surplus is now the model's closed form at p = 1,
`S_U(a_U,1) = max((A P_U a_U − (r+ν) U^search + λ_U I)/(r+ν+λ_U+ξ_U), 0)`, with the tail integral
I kept on its grid quadrature. J_U(a_U,1), E_U(a_U,1), the U^search update, the p* update,
free entry, E_U(a_U,1) as consumed by the skilled block, and the wage paid to a new hire all
read it. S_U(a_U,1) now rises from 0 at a^h, continuously in a_U, as the hireable-set lemma
says. The hired set (χ, a^h = b̃_U/P_U), the never-hired convention and every skilled-side
formula are unchanged. The layer is MAJOR: an equilibrium object is redefined and Q moves at
every point, all four estimates included (theorist's decision).

**Comparability.** None for Q, and the windows need re-estimation. Every v47.0.0 Q, moment,
θ_U and transition path is superseded; relative moves are of order 1e-4 (table below). No
struct changed. All four bundles load and re-evaluate unchanged. The equilibrium-object
NamedTuple gains two fields, `eU_entry` and `wU_entry`. No serialised object carries it.

### What changed
- `solver/unskilled.jl:119` `entry_surplus_unskilled(PUeff, flow, λ, I, D)` returns
  `max((A P_U a_U·1 + flow + λ I)/D, 0)`, where `flow` is the part of the numerator without p.
  - `unskilled_inner_loop!:302` calls it with `flow = −(r+ν) U^search`.
  - `Jfrontier = (1−β) S1` and `E1vec = U^search + β S1` are formed from that value, as before.
  - Every downstream reader of `uc.Jfrontier` / `E1vec` follows: the U^search update,
    `update_pstar_from_surplus!`, `update_theta_unskilled`, the skilled block's `EU1`
    (`solver.jl:174, :279`), and `equilibrium.jl`'s `JU_frontier`, `SU1`, `EU1`.
- `transition/transition_values.jl:260` `_unskilled_value_step!` calls the same helper with
  the transition's own numerator, `flow_out + inv_dt · S^raw_{n+1}(1)`.
  - The continuation `S^raw_{n+1}(1)` (`:251`) is extrapolated from the top two nodes.
  - This is exact, because the raw surplus is affine in p at every date: the terminal
    surface is affine, and each step adds a flow affine in p.
  - The terminal date copies the stationary `Jfrontier`, which is already at p = 1.
- `solver/equilibrium.jl`:
  - The hiring atom is its own vector, `eU_entry` (`:128`), and the damaged tail is
    `eU_density` (`:125`).
  - The reported `eU_surface` is `eU_density` with the atom in its top column, as before (`:130`).
  - The wage block pays the atom `wU_entry = β_U A P_U a_U·1 + (1−β_U)(r+ν) U^search`
    (`:184`). Each row's atom enters the unskilled wage list as its own support point (`:229`).
  - Under v47 the atom shared the top node's entry at w_U(a_U, p_N), and it was dropped
    altogether when p* > p_N, because that node's wage is NaN.
- `transition_solver.jl` and `policy/` contain no entry-surplus read of their own. The
  transition's forward pass and free entry read `path.Jfrontier`. Its mean-wage series already
  pays unskilled employment at p = 1 (`transition_solver.jl:677`).

### Measured (scratch copies of v47.0.0 and v48.0.0, 1 thread per process, bundle grids, `select_equilibrium`)
- **Q at the estimates**. "values" is the solver change alone: the v48 `unskilled.jl` and
  `transition_values.jl` on the v47 `equilibrium.jl`. "wage" is the atom change alone.
  "bins" is the wage change with the atom split off but still paid w_U(a_U, p_N).

  | window | v47.0.0 | v48.0.0 | ΔQ | rel | values | wage | of which bins |
  |---|---|---|---|---|---|---|---|
  | base_fc | 1059.3616874672787 | 1059.5806221932073 | +0.21894 | +2.07e-4 | +0.24253 | −0.02355 | −0.02356 |
  | crisis_fc | 709.9846318828362 | 710.08144979708322 | +0.09682 | +1.36e-4 | +0.06239 | +0.03446 | +0.03446 |
  | base_covid | 2884.5949766504946 | 2884.581528455622 | −0.01345 | −4.7e-6 | −0.11367 | +0.10026 | +0.10041 |
  | crisis_covid | 1845.7773747608371 | 1844.8940797000023 | −0.88330 | −4.79e-4 | −0.90488 | +0.02165 | +0.02171 |

  - `smm_objective` returns the same four values.
  - **Value channel.** Raising S_U(a_U,1) by A P_U a_U(1−p_N)/D raises U^search, by a median
    of 7.6e-6 to 1.8e-5 relative across windows. It also moves p*_U up at every hired node,
    by a median of 5.8e-5 to 2.0e-4. The moments that move most are `emp_cm3_U` (up to 9.9e-4
    relative), `sep_rate_U` (6e-5 to 8.5e-4) and `ur_U` (up to 4.8e-4).
  - **Wage channel.** Almost all of it is the histogram range, not the wage itself. Paying
    the atom w_U(a_U,1) instead of w_U(a_U,p_N) moves Q by at most 1.5e-4. Giving the atom
    its own support point moves Q by up to 0.10, because `_wage_density`'s range is
    `quantile(all_wages, 0.002 / 0.998)` over unweighted support points (Known gaps).
  - At all four estimates the straddling node lies below a^h (node 26 at −3.2e-7 relative in
    the FC windows), so no band type exists there, and v47's zero-surplus band does not bite.
- **(1) b_U sweep at base_fc** (41 points over ±1e-4 relative, the 11-point x_26 ladder, and
  39 points over s ∈ [−6.0e-5, −5.3e-5] around the old onset s* = −5.6444e-5):
  - **The onset step is gone.** At s = −5.645e-5 → −5.644e-5, Q is 1059.462176 → 1059.490359
    under v47 (+0.0282) and 1059.707564 → 1059.707541 under v48 (−2.3e-5, the local slope
    × 1e-8).
  - A quadratic fits Q over the 39-point window with max residual 1.5e-2 under v47 and
    2.2e-10 under v48.
  - On the 41-point sweep, the largest interval excess over the neighbouring slopes is
    0.02826 (v47) and 4e-5 (v48). Max |second difference| is 2.84e-2 (v47) and 4.9e-5 (v48).
  - **The x_26 crossing stays continuous.** The ladder's largest excess is 2.2e-9.
  - All points are feasible and labelled H in both trees. Relative ΔQ along the sweep lies in
    [2.05e-4, 2.33e-4].
- **(6) J_U(a_U,1) across a^h at fixed parameters.**
  - At every hired straddling node, J_U(x_26,1) equals
    (1−β_U) A P_U (x − a^h)/(r+ν+λ_U+ξ_U+f_U β_U) to 7 digits (λI ≈ 0 there):
    1.097513e-6 at x−a^h = 1e-8 relative, 1.097513e-4 at 1e-6, and 6.159119e-3 at the old
    onset. It is exactly 0 at and below a^h.
  - v47 gave 0, 0 and 0 at those three points.
  - At the base_fc estimate the three nodes are x_25 (J = 0), x_26 (3.2e-7 below a^h, J = 0)
    and x_27 (J = 8.712663; v47 8.706580; the closed form without λI gives 8.470699).
- **(3) base_fc 19×19 (a_ℓ, c) plane.**
  - 361/361 feasible in both trees: 0 gained, 0 lost, labels identical (304 H, 57 L).
  - Relative ΔQ lies in [−4.60e-4, +5.58e-4], median +9.6e-5. Max |Δθ_U|/θ_U is 6.9e-5.
- **(4) Transition** (`transition_audit.jl`, `TA_PAIR=fc`, Nx = 120):
  - It converges in 23 iterations in both trees (final ‖Δθ‖∞ 7.277e-5; v47 7.278e-5).
  - The terminal gap to z1 is 3.534021e-3 (v47 3.535302e-3).
  - The per-date series move by at most 4.0e-4 relative (ur_U), 8.8e-5 (θ_U) and 5e-6 (θ_S).
  - With `TA_NULL=true` the terminal gap is 3.358951e-7 (v47 3.358890e-7) after 1 iteration.
- **(5) Gates** (v48 scratch):
  - All changed files parse.
  - `check_threading` -t 2: serial and threaded are bitwise equal at 1059.5806221932.
  - `check_hire_margin`: OK, 0 disagreements at all four estimates.
  - `check_dead_settings`: PASS. `check_output_tree`: PASS on a tree holding the four bundles.
    The live `output/estimates/` fails rule 6 on its `.pre_migration` files, which predate
    this change.
  - `check_forwarding`: the pre-existing `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` finding.
    `check_structure`: the same 7 files.
  - `check_repo.sh --verify-only` exits 1 from that forwarding finding, as under v47.0.0.
  - `check_version_sync` is in sync at 48.0.0.
  - No optimiser or chain was run.

### Settings
None added, retired or re-defaulted.

### Known gaps
- **The unskilled wage histogram's range depends on the number of support points.** The range
  is `quantile(all_wages, 0.002 / 0.998)` over unweighted points. Adding one support point per
  row at the same mass therefore moves Q by up to 0.10 at the estimates. This is the "bins"
  column above. The new support point is the model's own (a new hire is paid w_U(a_U,1)), but
  the range rule should be a property of the distribution, not of how it is listed.
- The `eU_surface` density-versus-mass weighting (reported against v22.0.0) is unchanged.
  The tail enters the wage list as a density value and the atom as a mass.
- **The `policy/` layer does not load at v47.0.0 or v48.0.0, for reasons that predate this
  change.** `policy_params.jl:125` names the retired `RegimeParams`, and `policy_solver.jl:33`
  reads `obj.wx`, which `compute_equilibrium_objects` does not return. Its unskilled
  mean-wage loop still prices the top column at w_U(a_U, p_N). When the layer is revived it
  should read `eU_entry` and `wU_entry`.
- The top p-cell [p_N, 1] is still never split in the tail integral I or in any p-density.
  Entry no longer reads the grid there, so the cell matters only through the λ_U redraw tail,
  where it is ordinary quadrature.
- No window has been re-estimated under v48.0.0.

### Tried and rejected
- **Splitting the top p-cell by (1 − p*)/(1 − p_N) instead.** Not implemented, by the theorist's
  decision. It would still price entry at p_N, a point the model never uses. Evaluating at
  p = 1 is the model's own object and makes S_U(a_U,1) → 0 continuously at a^h.

---

## v47.0.0 — 2026-10-01

**MAJOR: unskilled free entry values a meeting only through the hired branch of the
unemployed.** The expected value of a meeting is `Jbar / ũ_U`. Under v46.0.0 the numerator
was `Σ_ij J_U(x_i,1) (u_ij + d u_S,ij)`, so in a cell that straddles a^h the never-hired part
of the row's unemployed entered at the node's J_U. Those types produce no match when met, and
their J_U is 0, as it already is for a node below a^h. The numerator now weights u_U by the
hired branch's share of the row's unemployed, `hired_i = χ_i υʰ_i / usurv_i`
(`unemployment_split_unskilled`). The seeker pool ũ_U is unchanged: never-hired types still
search, are still met, and are still unemployed (the author's convention, untouched). This
closes the v46.0.0 known gap "free entry prices the never-hired sliver of an open straddling
node at that node's J". The layer is MAJOR because neither lower test holds. MINOR requires a
run to reproduce the old numbers, and it does not: Q moves at 171 of 361 plane points and along
b_U within 6e-5 relative of b̂_U at base_fc. The discretised free-entry condition is redefined,
and re-estimation can move θ̂. The continuum free-entry equation in the notes is unchanged.

**Comparability.** Every v46.0.0 number at the four stored estimates is reproduced
**bitwise**: Q, and the moments behind it. In all four windows the straddling node lies below
the point where J_U at the top p-node turns on, so J_U(x_k,·) = 0 there and the weight does not
bite. Superseded:
- any v46.0.0 Q or θ_U at a point whose straddling node has J_U > 0 (on the base_fc plane,
  every point with χ_26 ≥ 0.518);
- transition paths on a grid where that is the case (`transition_audit.jl` at its default
  Nx = 120).

No struct changed, and all bundles load.

### What changed
- `solver/unskilled.jl:498` `update_theta_unskilled(model; u_hired = nothing)`. The default
  hired branch is `u_U .* hired`, from `unemployment_split_unskilled` at the installed p* and
  θ_U, which are the values `uc.u` was built from (`solve_unskilled_block!:541`). The
  numerator is `Σ J_i (u^h_ij + d u_S,ij)` (`:516`), and `ũ_U = Σ (u + d u_S)` is unchanged.
  The `duS_carry` term is not weighted (zero in any stationary state). The corner-limit branch
  reads the stationary limit composition, so its hired branch is the stationary split
  whatever the caller passes. `global_residuals` R2 calls the same function with the default,
  so the residual and the iteration test one condition.
- `solver/unskilled.jl:408` `hired_unemployed(u, χ, τ, ℓ) = (u − (1−χ)(1−τ)ℓ)⁺`, the stock
  form of the same split. This is the expression the transition's forward pass had inline,
  hoisted so that both of its uses read one definition.
- `transition/transition_solver.jl`:
  - `_forward_pass!:190` calls `hired_unemployed`. The expression is identical, so the
    arithmetic is unchanged.
  - `_update_tightness!:481` passes the date's stock split as `u_hired`. Off a steady state
    the row's unemployed are not at their stationary shares, and the stock split is the share
    the forward pass actually hires at f_U, so free entry prices the matches that path makes.
    At a converged stationary state the two forms agree to max|row − stock| / max u ≤ 1.2e-16
    (all four estimates).
- No other J-weighted meeting value over unskilled seekers exists. `equilibrium.jl` uses
  `JU_frontier` only for the reported surfaces `SU1`, `EU1` and `JU_surface`. No moment reads
  J_U. `policy_solver.jl` reports θ only.

### Measured (scratch copies of v46.0.0 and v47.0.0, 1 thread each, bundle grids, `select_equilibrium`)
- **Q at the estimates: bitwise unchanged.** base_fc 1059.3616874672787, crisis_fc
  709.9846318828362, base_covid 2884.5949766504946, crisis_covid 1845.7773747608371.
  `smm_objective` gives the same four values. At the straddling node J_U(x_k,·) = 0 exactly:
  cell 26 (χ = 0.508852, hired = 0.2705 base_fc / 0.2780 crisis_fc) and cell 20 (χ = 0.511365,
  hired = 0.3055 / 0.3194). So the meeting value Jbar is equal to 12 digits (24.919867743 at
  base_fc).
- **No straddling cell, 32 random points** (the v46.0.0 invariance set: seed 20261001,
  t̂ + 0.03·N(0,1), a^h < x_1 or b_U/P_U exactly on a cell edge). The **full solve** is bitwise
  at **32/32** in Q and θ_U, with identical labels. This is stronger than v46's formula-level
  criterion. Here χ is not solved for in either tree, and `hired` is exactly 1.0 where χ = 1,
  and J_U is 0 where χ = 0.
- **Continuity along b_U at base_fc** (41 points over ±1e-4 relative, plus the 11-point
  ladder around b_U* = P_U x_26):
  - **The crossing a^h = x_26 is bitwise unchanged.** All ladder points are equal, and so is
    every sweep point with s ≥ −5.5e-5. There is no step at the node in either tree.
  - **The weight bites only for s ≤ −5.75e-5**, where Q falls by 2.0e-8 to 2.4e-7 relative.
  - Away from the step below, max |second difference| is 1.23e-4 in both trees.
- **The residual b_U step is diagnosed (it is not this margin, and it is unchanged).** The
  v46.0.0 step between s = −5.75e-5 and −5.5e-5 (excess +0.02826 v46, +0.02826 v47) sits
  exactly where J_U at node 26 turns on. The unskilled p grid is Gauss–Legendre (Np_U = 160),
  so its top node is p_N = 1 − 5.612e-5, not 1. In `_soft_weight` the top cell [p_N, 1] is a
  hard indicator: `j ≥ Np` returns 0, so it is never split. For a^h < x < a^h/p_N the type is
  therefore hired (p* < 1, χ counts it) but has zero surplus at p_N, and J_U = 0.
  - Predicted onset: s* = x_26 p_N P_U / b_U − 1 = −5.6444e-5.
  - Measured: J_26 = 6.3e-7 at s = −5.645e-5 and 0 at −5.644e-5. Q is 1059.462176 and
    1059.490359 at those two points (Δ −0.0282), with `emp_cm3_U` −0.03977580 → −0.03973408
    and `emp_var_U` 0.29144084 → 0.29141210.
  - J_U itself is continuous through the onset; the unskilled wage moments are not. See known
    gaps.
- **base_fc 19×19 (a_ℓ, c) plane:**
  - 361/361 feasible in both trees: 0 gained, 0 lost, labels identical.
  - 190 points are bitwise, and they are exactly those with χ_26 ≤ 0.508852.
  - The other 171 (χ_26 ∈ [0.518169, 0.592524]) move by relative ΔQ in [−2.6e-5, +9.0e-6]
    (median −1.4e-5), with max |Δθ_U|/θ_U 3.7e-6.
- **Transition** (`transition_audit.jl`, Nx = 120):
  - With `TA_NULL=true` (base_fc against itself) the terminal distribution gap is 3.358890e-7
    (v46: 3.358897e-7), and u_U total is 0.042311 → 0.042311. The forward law and the
    stationary KFE still agree.
  - On the fc pair the per-date series move by at most 2.5e-5 relative (θ_U) and the
    terminal gap to z1 is 3.535302e-3 (v46: 3.535293e-3). At Nx = 120 the straddling node is
    open, so the endpoints themselves move (θ_U(z1) 0.226543 → 0.226541).
- **Gates** (v47 scratch):
  - Both changed files parse.
  - `check_threading` -t 2: serial and threaded bitwise at 1059.3616874673.
  - `check_hire_margin`: OK, 0 disagreements at all four estimates.
  - `check_dead_settings` and `check_output_tree`: PASS.
  - `check_forwarding`: the pre-existing `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` finding.
    `check_structure`: the same 7 files.
  - `check_repo.sh --verify-only` exits 1 from that forwarding finding, as under v46.0.0.
  - `check_version_sync` is in sync at 47.0.0.
  - No optimiser or chain was run.

### Settings
None added, retired or re-defaulted. `u_hired` is a keyword of `update_theta_unskilled`, not a
setting. Its default is the stationary split, and only the transition passes it.

### Known gaps
- **The top p-cell [p_N, 1] is a hard indicator** (`grids.jl` `_soft_weight`, `j ≥ Np →
  0`). This is the mechanism of the residual b_U step above.
  - It also means the hireable-set lemma's entry point p = 1 is not on the grid. A type in
    the band (a^h, a^h/p_N) is hired at f_U, yet its firm value is 0, so free entry gains
    nothing from meeting it.
  - Two fixes are candidates: split the top cell by (1 − p*)/(1 − p_N), or evaluate the
    entry surplus at p = 1. Both would change Q at all four estimates. Neither is applied: it
    is a decision for the theorist.
- **No window has been re-estimated under v46 or v47.** At the stored points the two versions
  agree, but within 6e-5 relative of b̂_U they do not, so a re-estimation can differ.
- **The cross-market drain ignores χ** (unchanged from v46.0.0, inert in any stationary state).

### Tried and rejected
- **Applying the stationary row share `hired_i` inside the transition**, which would have
  meant calling `unemployment_split_unskilled` at the date's p* and θ_U. Not used. Off a
  steady state it weights a different mass from the one the forward pass hires, so free entry
  would price matches the path does not make. The stock split equals the row share to 1e-16 at
  every stationary state measured.

---

## v46.0.0 — 2026-10-01

**MAJOR: the unskilled hiring margin is a covered fraction of the ability cell, not a node
indicator.** A fresh unskilled match starts at p = 1, so a type is hired iff p*_U(a_U) < 1,
and by the hireable-set lemma that holds iff a_U > a^h = b̃_U / P_U at every tightness. The
code read the lemma's indicator per node, so a row moved wholesale between the hired and
never-hired branches when a^h crossed a node. Each cell is now weighted by its hired
fraction χ_i = clamp((xhi_i − a^h)/(xhi_i − xlo_i), 0, 1) on the midpoint intervals that τ_T
and d use. This is the v39.0.0 pattern (OJS participation) applied to the third margin on the
ability grid. The layer is MAJOR rather than MINOR: an equilibrium object is redefined, Q
moves at every point where a cell straddles a^h (all four estimates and every plane point),
and b̂_U at base_fc was pinned by a node location rather than by the data. The theorist has to
be told, because the lemma enters the paper and b_U gains an identifying channel through the
mass at the threshold.

**Comparability.** None for Q, and the estimates must be re-run. Every stored Q and every
downstream number at a window is superseded. The base_fc θ̂ in particular was parked on the
node-26 lip and should not be reported. No struct changed: all four bundles load unchanged and
re-evaluate under the new code. Never-hired types are still unemployed and still in the seeker
pool and in measured unemployment (the author's convention). The free-entry formulas are
unchanged.

### What changed
- `solver/grids.jl` — `fraction_above(xlo, xhi, x, a)`, the cell's quadrature of 1{ability ≥ a}.
  `training_policy!` and `hire_fraction` share it, and the arithmetic is the expression
  `training_policy!` had inline.
- `solver/unskilled.jl`, new section "Hiring margin" before the stationary composition:
  - `hire_fraction(model)` → χ over a_U cells, closed form `a^h = up.bU / up.PU`. It does not
    interpolate p*_U.
  - `unemployment_split_unskilled(pstar, f, model)` → per row `(χ, usurv, hired)`. Here
    `usurv = χ υʰ + (1 − χ)`, with `υʰ = (δ+ν)/(f+δ+ν)` computed with f whatever the node's
    classification, and `hired = χ υʰ / usurv` is the hired branch's share of the row's
    unemployed.
- `solver/unskilled.jl` `solve_stationary_unskilled!`: `u = W2 · usurv_i · (1 − τ)` replaces
  the `f_hire = (p* < 1 − 1e-10) ? f : 0` gate. Employment is the residual
  `W2(1−τ)χ(1−υʰ)`, so mass is conserved by construction.
- `solver/equilibrium.jl`:
  - The hiring atom of `eU_surface` at p = 1 is fed by `hired_U · u_row`, not the whole row.
  - The unskilled duration survivor is the per-row mixture
    `hired·e^{−(f_U+ν)t} + (1−hired)·e^{−νt}`, which replaces the per-node switch at the old
    `:413`.
  - The comment calling never-hired types the "non-searching mass" now says they are
    unemployed and searching but never hired.
  - Every other consumer of the hired set reads masses or the node's own p* (`sep_rate_U`,
    `wchg_rate_U`, the wage surface), so the hired sliver of a node below a^h takes that node's
    values (p* = 1), as a τ cell does.
- `transition/transition_solver.jl` `_forward_pass!`: the never-hired slice
  `(1−χ)(1−τ)ℓ` is wholly unemployed. Only the rest, `u_U^h = (u_U − (1−χ)(1−τ)ℓ)⁺`, leaves at
  f_U, written as `fU_hire = f_U · u_U^h/u_U`. χ is computed once per pass from the post-switch
  primitives. The law of motion in the file header is updated.
- `gates/check_hire_margin.jl` (new), wired into `check_repo.sh` after `check_threading`. At
  each stored estimate it checks that the solved classification `p*_U(x_i) < 1 − 1e-10` and
  the closed form `x_i > a^h` agree. A disagreement within 1e-6 relative of a^h is reported
  rather than failed, because there the closed form is exact and the node's classification is
  rounding.

### Measured (scratch copies of v45.1.0 and v46.0.0, 1 thread each, bundle grids 160/160/180, `select_equilibrium`)
- **Q at the estimates, v45.1.0 → v46.0.0.** v45.1.0 reproduces each bundle's `loss_opt`.
  Each window has one straddling cell:
  - base_fc: 1046.760228531 → **1059.361687467**. Cell 26, χ = 0.508852, row mass 0.009502.
    θ_U 0.458721 → 0.462099, and θ_U 0.465394 on the other side of the old lip. Unique root H.
  - crisis_fc: 710.281317497 → 709.984631883. Same b_U and P_U as base_fc (both fixed), so
    the same cell and χ.
  - base_covid: 2834.326345368 → 2884.594976650. Cell 20, χ = 0.511365, row mass 0.008250.
  - crisis_covid: 1816.503468941 → 1845.777374761. Same cell and χ.
  - 33 of the 35 active moments move by more than 1e-9 relative at each estimate. The largest
    are unskilled: at base_fc
    `sep_rate_U` +1.25%, `theta_U` +0.74%, `jfr_U` +0.37% and `ur_U` −0.28%.
- **Continuity along b_U at base_fc** (41 points over ±1e-4 relative):
  - Under v45.1.0 Q steps by **−48.362** between s = −5e-6 and 0. Bisected to adjacent floats
    b_U = 0.061827492366036539 | …546, it is 1095.107140 | 1046.760220, with θ_U 0.465394 |
    0.458721.
  - Under v46.0.0 the same two floats give 1059.3624075762166 | 1059.3624075762937 (Δ 7.7e-11),
    with θ_U equal to 11 digits.
  - The node crossing itself, a^h = x_26 = 0.0618274923722, is a kink: the slope is
    −2.2717e3 to the left and −2.2467e3 to the right, per unit relative b_U. Q is piecewise
    linear with max |second difference| 7.5e-6 away from the step below. θ_U moves at most
    2.1e-6 between sweep points (v45.1.0: 6.7e-3).
- **A second, smaller step survives and is not this margin.** Between s = −5.75e-5 and
  −5.5e-5, Q rises +0.0225 against the local trend (excess +0.028 over one sweep step). It is
  present in v45.1.0 at the same place (excess +0.030). It is carried by the unskilled
  wage-distribution moments: `emp_cm3_U` +1.05e-3 relative across the step, `emp_var_U`
  −9.9e-5. θ_S and the root label do not move. See known gaps.
- **Invariance where no cell straddles a^h.** 32 random points, 8 per window, seed 20261001,
  t̂ + 0.03·N(0,1). b_U is then set so that a^h < x_1 (χ ≡ 1, 16 points) or so that
  b_U/P_U equals a cell edge exactly (χ ∈ {0,1} with both values present, 16 points).
  - Formula level, bitwise: **32/32**. From each v45.1.0 converged state, both versions
    re-derive u_U and t and then the objects, the moments and Q, and every array and Q is `==`.
  - Full solve: 15/32 bitwise. The other 17 differ by at most 1.0e-12 relative (median
    1.5e-14). θ_U differs by at most 2.4e-13 relative, and labels are identical.
  - Why some full solves differ: instrumenting v45.1.0's gate shows the 17 are **exactly** the
    points where the per-node gate disagreed with the closed form at a transient iterate. That
    happened at 2 to 36 node-calls, all in the solve's first four stationary calls, from the
    initial p*. On the 15 points with no transient disagreement Q is bitwise equal. The
    difference is an iteration path to the same fixed point, not a different quadrature.
- **base_fc 19×19 (a_ℓ, c) plane, offsets ±0.2 natural units:** 361/361 feasible under both
  versions, so 0 gained and 0 lost, with identical labels. The straddling cell is 26
  everywhere, with χ ∈ [0.4248, 0.5925]. The relative ΔQ has median −4.1% and lies in
  [−9.9%, +7.0%].
- **Mass at the estimate (base_fc).**
  - max |e_U − W2(1−τ)χ(1−υʰ)| = 5.4e-20, and max |m_U − u − t − e| = 9.5e-20.
  - Cell 26 holds e = 2.433e-4 and u = 5.015e-4, and 27.05% of its unemployed are in the hired
    branch.
  - The never-hired unemployed are 3.814e-3, which is 9.0% of u_U. The marginal mass below a^h
    is 0.1170.
- **Hiring-margin gate:** 0 disagreements at all four estimates. Along the b_U ladder, the node
  within 1e-12 relative of a^h disagrees, which is reported as within precision. Under
  v45.1.0 the misclassified band is 1.0e-10 wide.
- **Transition.**
  - `run_transition_simulation(:fc)` with `transition_main.jl`'s settings (grid 200, 240
    steps, tol 1e-4) converges in 23 iterations (d = 7.18e-5); v45.1.0 does the same in 23
    (7.19e-5).
  - Inherited stocks at t = 0 match z₀ (ur_S exact). The terminal gaps to z₁ are θ_U 3.8e-4,
    θ_S 4.5e-3, ur_U 6.9e-4 and ur_S 2.0e-2 relative, the same as v45.1.0's to two digits.
  - `transition_audit.jl` with `TA_NULL=true` (base_fc against itself, cell 26 straddling)
    gives a terminal distribution gap of 3.36e-7 (v45.1.0: 3.36e-7) and u_U total
    0.042311 → 0.042311. The forward law with the χ split reproduces the stationary KFE.
- **Gates:**
  - Every changed file parses.
  - `check_threading` -t 2: serial and threaded bitwise at 1059.3616874673.
  - `check_hire_margin`: OK.
  - `check_dead_settings` and `check_output_tree`: PASS.
  - `check_forwarding`: 1 finding (`ROYSEARCH_MCMC_JAC_ALLOW_GAPS`). `check_structure`: the
    same 7 files. Both are pre-existing.
  - `check_repo.sh --verify-only` exits 1 under both versions, from that forwarding finding.
  - `check_version_sync` is in sync at 46.0.0.
  - No optimiser or chain was run.

### Settings
None added, retired or re-defaulted. There is no switch back to the node indicator. A
formula-level `:indicator` path would reproduce v45.1.0 only where no cell straddles a^h, which
the invariance test above already establishes.

### Known gaps
- **No window has been re-estimated under v46.** The four Q values above are evaluations at
  v45-era points. The README equilibrium table (base_fc 1050.153253, crisis_fc 970.511766) was
  already stale and is now superseded too.
- **Free entry prices the never-hired sliver of an open straddling node at that node's J.**
  The numerator `Σ J_i u_i` is unchanged, as instructed. For a node above a^h whose cell
  straddles it, the never-hired part of the row enters at J_i > 0, although those types have
  J = 0. The error is O(cell width²) and continuous across the crossing. The exact form
  weights the numerator by `hired`, and it is a decision for the theorist.
- **The cross-market drain ignores χ.** `hire_X = f_U·Σu_S^drain` (equilibrium) and
  `d·f_U·u_S` (transition) let crossers at a_U ≤ a^h find unskilled jobs. This is inert in
  any stationary state (d u_S ≡ 0, Prop. 1) and live only while the frontier moves.
- **The residual b_U step (ΔQ ≈ +0.028) sits in the unskilled wage-moment layer**
  (`emp_cm3_U`, `emp_var_U`) and is present in v45.1.0. Its mechanism (the wage-grid quantile
  endpoints or the 1e-16 mass filter in `_wage_density` are the candidates) has not been
  diagnosed.
- **Pre-existing, found while testing, not fixed: `unpack_θ(…; constrained = true)` maps
  the unskilled and skilled block parameters through `_to_constrained` a second time**
  (`smm/smm_params.jl`, the disambiguation loop of step 3 ignores the flag). At the base_fc
  θ, `select_equilibrium(θ; constrained = true)` returns Q = 1.13e6 with b_U read as 0.2099,
  where the unconstrained path gives Q = 1046.76. This affects `MCMC_SPACE = :theta`
  (`MCMC_main.jl:849`). The default `:t` is unaffected.

### Tried and rejected
- **Interpolating the raw p*_U to locate the threshold.** Not implemented. The closed form is
  exact and needs no extrapolation below a^h. The lemma makes the threshold independent of
  tightness, so p*_U adds no information.
- **A bitwise full-solve invariance test.** Rejected as the criterion after measurement, not
  on grounds. The 17/32 full-solve differences coincide one-for-one with transient gate
  disagreements from the initial p*, so the criterion that can be bitwise is the formula-level
  one, and it is: 32/32.

---

## v45.1.0 — 2026-09-27

**MINOR: the skilled block's step test no longer counts p^oj_S for types that do not
search on the job.** A type is inactive in a pass when its OJS participation is exactly 0
before and after it. Its p^oj_S enters no object, because every consumer weights it by
`_ojs_share = s_part × weight`. For high-ability types that p^oj_S sits on a plateau at the
top of the p grid, where S^1 − S^0 is flat to about 1e-12 relative, so rounding decides the
node and p^oj_S hops by 2e-4 to 8e-4 against a tolerance of 1e-6. Every iterate is computed
and installed as before. Only the stopping decision changes. The layer is MINOR, not MAJOR:
no estimate, label, root or feasible point changes, and Q moves only because a block may stop
a pass earlier at the same tolerance (at most 4.9e-14 relative at the estimates, 3.2e-7 on
the plane). It is not PATCH, because no stored Q is reproduced bit for bit.

**Comparability.** Every v45.0.0 result is comparable, and no stored bundle is superseded.
No struct changed. All four bundles load and round-trip under the new types.

### What changed
- `solver/skilled.jl:869-881` (`solve_skilled_block!`): `inactive[k]` is
  `part_old[k] == 0.0 && s_part[k] == 0.0`, and `Δoj[k]` is 0 for such a type and
  `|p^oj_k − p^oj_old,k|` otherwise. `dj`, the outlier list `out` (`:911`) and the allowance
  residual `dR` (`:934`) read `Δoj`. The Δp*_S and Δs_part terms still cover every type, so a
  type switching participation registers through Δs_part. `R1` in `global_residuals` already
  skipped these types (`solver.jl:196`), so the block and the battery now read the same set.
- `solver/skilled.jl:911-924`, the outlier allowance's anchors: the walk to the nearest
  in-tolerance neighbour now also steps over inactive types, whose p^oj_S is not an
  equilibrium object. A side with no anchor is the edge of the region where p^oj_S is defined,
  which is either the grid edge or the participation frontier with only inactive types beyond
  it. The interpolant is then the anchor on the other side. With no anchor on either side the
  test falls back, as before.
- `solver/skilled.jl:672-679`: the loop header now names Δs_part among the tested components
  and states that Δp^oj_S is taken over participating types.
- `README.md`, "What converged means": one sentence on the step test.

### Measured (scratch copies, 2 threads, v45.0.0 and v45.1.0 loaded side by side in one process and interleaved per point)
- **Q at the estimates, v45.0.0 → v45.1.0** (normal solve and selection; no value is
  bitwise equal; label, tag and θ_S are unchanged, with |Δθ_S|/θ_S ≤ 3.7e-14):
  - base_fc: 1048.744645932, relative change −3.9e-15, unique H, θ_S 1.210839. This is the live
    bundle, an SMM checkpoint the author's run wrote at 07:51.
  - crisis_fc: 970.511765754, −3.4e-14, "H of 2", θ_S 0.657687.
  - base_covid: 2834.326345368, +4.8e-16, unique, θ_S 1.812483.
  - crisis_covid: 1816.503468941, −4.9e-14, unique, θ_S 2.321072.
  - base_fc v45-era θ̂ (candidate id 1, the plane centre): 1050.153252946, −6.5e-16, unique H.
- **Coverage, base_fc 19×19 (a_ℓ, c) plane, offsets ±0.2 natural units around the v45-era
  θ̂:** 360/361 under both versions, the same set, so 0 gained and 0 lost. The failing point is
  still (a_ℓ −0.156, c −0.022), where the skilled block converges and the global loop does not.
  On the 360 common points no Q is bitwise equal. The relative |ΔQ| has median 2.7e-14, 90th
  percentile 2.8e-11, 99th 4.7e-9 and maximum 3.2e-7, with none above 1e-6. The maximum
  relative |Δθ_S| is 3.1e-10, and label and tag are identical at all 360 points.
- **Why no Q is bitwise equal.** A block that stops a pass earlier hands a different state to
  the global loop, and a selection runs 60 to 120 skilled blocks, so one earlier stop anywhere
  changes the last bits. The tolerance is unchanged, so the change is bounded by the
  tolerance-level spread above.
- **Skilled-block passes.**
  - Normal solve, total over its block calls, v45.0.0 → v45.1.0: base_fc 95 → 90, crisis_fc
    178 → 171, base_covid 77 → 71, crisis_covid 80 → 76, base_fc v45-era 92 → 87.
  - Selection: 607 → 593, 356 → 344, 736 → 724, 692 → 682, 608 → 594.
  - Plane: 92.1 → 84.3 per normal solve. Every one of the 361 points takes fewer passes, with a
    per-point ratio of median 0.941 (10th percentile 0.824, 90th 0.948). Per selection, the
    mean falls from 538.3 to 521.0.
- **id168** (design candidate 168, nobody searching on the job):
  - v45.0.0: 7 block calls of 42, 41, 36, 31, 25, 26 and 35 passes (236 in all). Calls 1 and 2
    end on stall hand-back (d 1.8e-4 and 9.8e-5). Calls 3, 4, 6 and 7 end through the
    allowance, anchored on inactive types. 6 of the 7 calls still have inactive movers above
    tolerance at their last pass (6, 3, 1, 2, 0, 2, 2).
  - v45.1.0: 7 calls of 20, 12, 9, 7, 4, 4 and 4 passes (60), all ending on the plain test.
    The test counts 0 movers in every call. The inactive types still hopping at the last pass
    (156, 68, 31, 31, 29, 29, 29) are the ones it now ignores.
  - Q is 1850130.495337 → 1850130.495336 (−5.9e-13), unique U.
- **Negative control:** 193 passes, over the normal solves and selections at the six points
  above, on which at least one type switched participation. The count is identical in both
  versions. On every one of them Δs_part ≥ 2.0e-2 and d ≥ 2.0e-2, and none was excused by the
  allowance.
  - Example: crisis_fc, normal solve, call 4. 13 switching passes (passes 3 to 18, up to 16
    types in one pass) take the searching set from 56 types to 11, with Δs_part 0.21 to 1.0 on
    those passes. The block converges at pass 32 (v45.0.0: 33).
- **The allowance depended on inactive anchors.** Under v45.0.0 all 1180 allowance exits on
  the plane had an inactive anchor: 637 in normal solves and 543 in selections. The frontier
  type the allowance exists for borders the non-searching region, so its only in-tolerance
  neighbour on that side was an inactive type. Under v45.1.0 the allowance still takes 1122
  exits, none of them one-sided at the grid edge. The grid-edge extension exists in the code
  but never fired.
- **Runtime, the five estimate points, four ABBA repetitions (repetition spread median
  4.1%).** v45.1.0/v45.0.0 per repetition:
  - normal solve: median 0.998, range 0.94–1.12; means 3.705 s and 3.713 s.
  - solve plus selection: median 0.996, range 0.97–1.05; means 11.958 s and 11.955 s.
  - On the plane, solve plus selection: median ratio 0.993.
  - The machine was shared with the author's base_fc SMM run.
- **Gates:** `skilled.jl` parses clean. `check_threading` passes, with serial and two threaded
  evaluations bitwise equal at 1048.7446459324. `check_dead_settings` and `check_output_tree`
  pass. `check_forwarding` reports 1 finding (`ROYSEARCH_MCMC_JAC_ALLOW_GAPS`) and
  `check_structure` 7, both pre-existing. `check_repo.sh --verify-only` passes. All gate logs
  are identical to v45.0.0's. `check_version_sync` is in sync at 45.1.0.
- **Entry path:** `smm_objective` → `select_equilibrium` runs at every point above. No
  optimiser or chain was run.

### Settings
None added, retired or re-defaulted. `CONV_S_MAXOUT` and `tol_outer_S` are unchanged.

### Known gaps
- `transition/transition_values.jl:352` has its own backward-induction step test, and it also
  counts Δp^oj_S for every type. It is outside this change, and whether it shows the same false
  alarm has not been measured.
- The allowance's stability test is nearly automatic once its anchors are fixed. An
  interpolant between in-tolerance anchors, or a single in-tolerance anchor, moves by less than
  tol by construction. So it binds only when the outlier set or the anchors change between
  passes. This is true of v45.0.0 as well, and it is not changed here.
- The README estimate table still shows base_fc at 1050.153253. The live bundle is now the
  author's 07:51 SMM checkpoint (1048.744646), and the table was left for the author to update
  when his run finishes.

### Tried and rejected
- **Anchors that step over inactive types, with no one-sided fallback** (the literal reading of
  the brief). This removes the allowance: 0 exits against v45.0.0's 1180.
  - Skilled passes rise from 92.1 to 98.4 per solve on the plane. At the estimates, crisis_fc
    goes 178 → 218, base_fc 95 → 101 and crisis_covid 80 → 90.
  - Q moves more: median 1.8e-10, maximum 1.1e-6 at (a_ℓ +0.044, c −0.133).
  - It gains the one failing point, (a_ℓ −0.156, c −0.022), at Q 22756.961976, θ_S 1.279888,
    unique H. That root is the genuine zero diagnosed on v43. The gain is a path effect: the
    stricter test keeps the block iterating, so the global loop is handed a different state.
    It is not a consequence of ignoring irrelevant cutoffs.
- **Pinning p^oj_S = p*_S for inactive types** (the author's option 2). This was not
  implemented: it changes every iterate after a type switches on, and it discards the peak
  rule just below the switch.

---


## v45.0.0 — 2026-09-26

**MAJOR: the skilled block is no longer Anderson-accelerated.** Its outer loop is a plain
damped fixed-point iteration: cutoffs under-relaxed by `damp_pstar_S`, θ_S by the shipped
adaptive free-entry weight. The equilibrium conditions, the detector and the selection rule
are unchanged, so every estimate keeps its equilibrium. The layer is still MAJOR, because the
feasible set of the objective changed. On the base_fc coverage plane, 18 of 361 parameter
points go from Q = Inf to finite. One point changes root inside its H tooth, with Q moving
1.1e-3 relative. And `damp_pstar_S`, a stored setting that no shipped path read, now drives
the skilled map. A run under v45 does not reproduce v44's numbers beyond the four estimates,
and an optimiser can now reach regions v44 scored as infeasible. The theorist has to be told.

**Comparability.** The four estimates are comparable. Each keeps its root and label, with
|ΔQ|/Q ≤ 1.3e-7 and |Δθ_S|/θ_S ≤ 1.7e-7, which is the difference between two iterations that
stop at the same tolerance. No stored bundle is superseded by this version. The crisis_fc
bundle is still owed its v44 re-estimate. No struct changed, and all four bundles load and
round-trip. Standard errors and chains computed under v44 remain valid at their θ̂. A local
design that reaches the band points recovered here would now score those points.

### What changed
- `solver/skilled.jl:669-677, 765-955`, `solve_skilled_block!`: the `Anderson1(1 + 2Nx)`
  state, the raw-install branch and the joint `[θ_S; p*_S; gap]` update are deleted. What
  remains is the former `use_anderson = false` branch: cutoffs blended by `damp_pstar_S`,
  then θ_S by the adaptive weight (`w` halves on a sign reversal of the undamped target,
  floor 0.1, otherwise ×1.2 back toward `DAMP_THETA_S[]`), unchanged. Under `θS_fixed`, θ_S is
  set on entry and then again on every damped pass (`skilled.jl:849`). The unskilled block
  (`unskilled.jl:496,499`) and the global loop (`solver.jl:308`) are untouched and keep
  Anderson under `use_anderson`.
- `solver/params.jl:51-56` (v44 numbering): `DAMP_POJ_S` is retired, together with its block
  at the end of the skilled pass. It damped p^oj_S only for the Anderson branch, which
  installed p^oj_S raw. Nothing assigned it, and at `1.0` the block was skipped, so this
  changes no number.
- `solver/params.jl:305-310`: the `use_anderson`, `damp_pstar_U` and `damp_pstar_S` field
  comments now say which loops read them.
- `solver/solver.jl:397-400`: the detector header no longer says the normal solve converges to
  M "because skilled Anderson is a secant step". It records the measurement instead: a
  restart at or next to M leaves for L or H. The M check itself is unchanged.

### Measured (scratch copy, 2 threads; v44 and v45 loaded side by side in one process and interleaved per point)
The author's base_fc SMM run was using the machine throughout, so absolute times are about
2.5× the v44 entry's serial times. Compare v44 with v45 through the per-point ratios, not
against earlier entries.

- **Coverage, base_fc 19×19 (a_ℓ, c) plane, offsets ±0.2 natural units (Q finite after
  selection):** v44 342/361, v45 360/361. Gained 18, lost 0. All 18 gained points are unique
  L roots at θ_S 1.031–1.042, with nobody searching on the job. Under v44 each failed with
  `converged_S = false`, and v45 solves them with 6–8 detector evaluations. The remaining
  point, (a_ℓ −0.156, c −0.022), fails under both. In v44 the skilled block does not converge
  there. In v45 it converges, but the global loop does not. This is the isolated free-entry
  point diagnosed on v43. No start or tag changes on any of the 342 common points (v44 and v45
  both: 263 unique, 79 "H of 2"). No normal solve on the plane ended at M in either version.
- **Q at the estimates (v44 → v45):** base_fc 1050.153256439 → 1050.153252946
  (−3.3e-9), unique H, θ_S 1.211645. crisis_fc 970.511641237 → 970.511765754 (+1.3e-7),
  "H of 2", θ_S 0.657687. base_covid 2834.326345368 → 2834.326345368 (−1.3e-13), unique.
  crisis_covid 1816.503405026 → 1816.503468941 (+3.5e-8), unique. **crisis_fc selects H.** Its
  normal solve now lands on L (0.606479), as on v43 with skilled Anderson off. The detector
  finds H with 12 evaluations where v44 used 23, because v44's normal solve sat on a tooth
  crossing at 0.655916.
- **|ΔQ|/Q on the 342 points feasible in both:** median 1.9e-8, 90th percentile 2.1e-7,
  99th 8.6e-7, max 1.1e-3. Two points exceed 1e-6:
  - (a_ℓ −0.044, c 0): 1.2e-6. Same unique H root, with θ_S equal to 1.9e-10, so this is the
    same equilibrium and the difference is the tolerance-level spread of the non-θ states.
  - (a_ℓ +0.156, c +0.133): 1.1e-3, Q 8251.691 → 8261.097. Label H in both, but a different
    crossing. A fixed-θ sweep at step 1e-4 shows three crossings of D inside one OJS tooth,
    the same map under both versions: 1.1425–1.1426 falling (attracting, 60 types
    searching), 1.1430–1.1431 rising (repelling, 61 types), and 1.1433–1.1434 falling
    (attracting, 61 types). v44 scored the **repelling** crossing (θ_S 1.143029). v45 scores
    the inner attracting one (1.142571). Neither scores the outermost, because the tooth is
    narrower than `EQ_PROBE`. See Known gaps.
- **Runtime, 20 points (each window's estimate plus four joint ±5% draws), best of two
  interleaved repetitions (repetition noise, median 5.7%):**
  - normal solve: mean 2.99 s (v44), 3.05 s (v45); ratio v45/v44 median 1.02, range 0.83–1.36.
  - solve plus detector: 9.73 s against 10.16 s; ratio median 1.05, range 0.75–1.22.
  - detector alone: median ratio 1.09. The mean number of evaluations is unchanged (17.7
    against 17.2; crisis_fc θ̂ drops from 23 to 12). Each fixed-θ evaluation now takes more
    skilled passes (next item).
  - On the plane, with both threads busy, the medians are 1.005 for the normal solve and 1.09
    for solve plus detector.
- **Skilled-block iterations at the estimates:**
  - normal solve, total over the global passes (mean / max per call), v44 → v45: base_fc 80
    (10.0/18) → 92 (11.5/25); crisis_fc 117 (13.0/43) → 178 (17.8/41), with one more global
    pass (10 against 9); base_covid 69 (9.9/22) → 77 (11.0/27); crisis_covid 69 (9.9/19) → 80
    (11.4/26). Global passes are otherwise equal, and unskilled iterations are equal except at
    crisis_fc (81 → 91).
  - fixed-θ evaluation (θ* × 0.98, 0.99, 1.01, 1.02): 22.4 against 32.5 skilled passes, max 23
    against 35, with 5 global passes in both. The cutoffs converge at the geometric rate of
    the 0.5 blend, where Anderson had extrapolated them.
- **Fixed-θ path:** at all four estimates, `solve_model!(m; θS_fixed)` at θ* × {1, 0.99,
  1.01} returns ok with θ_S held exactly (`===`). At θ* the residual D is 9e-12 to 2e-9, and
  its sign off θ* is correct in all twelve cases.
- **Middle root, crisis_fc at 0.995 k̂_S** (L 0.610612, M 0.636095, H 0.671231), restarting
  the normal solve from the state solved at θ_S = θ0:
  - θ0 = 0.636095 (M): v44 stays on M; v45 → L.
  - θ0 = 0.638: v44 → M; v45 → H.
  - θ0 = 0.634: both → L.
- **Policy settings** (`policy_main.jl`, `damp_pstar_S = 1.00`, now live): converged at all
  four estimates under v45, with the same θ_S as v44, and again at 0.50. The solve takes
  0.85–1.14 s.
- **Gates:**
  - Changed files parse clean.
  - `check_threading` passes: base_fc θ̂, serial and 2 threaded evaluations bitwise equal.
  - `check_dead_settings` and `check_output_tree` pass.
  - `check_forwarding` reports one finding (`ROYSEARCH_MCMC_JAC_ALLOW_GAPS`, an obsolete
    registry row) and `check_structure` seven. Both logs are identical under v44, in files
    this version does not touch.
  - Bundle round-trip under the v45 types preserves `loss_opt` and θ̂ bitwise.
- **Entry path:** `smm_objective` → `select_equilibrium` → `skilled_equilibria` runs at all four
  estimates (the Q values above) and threaded (`check_threading`). No optimiser was run.

### Settings
None added and none re-defaulted. `DAMP_POJ_S` is retired (a module `Ref`, not a serialised
field). `damp_pstar_S` moves from dead to live at its stored values: 0.50 in the bundles,
`smm_main.jl` and `model_main.jl`, and 1.00 in `policy_main.jl` and `initialise_model`.
`use_anderson` now covers the unskilled block and the global loop only. Both are recorded in
`SETTINGS.md`.

### Known gaps
- **Tooth pairs narrower than `EQ_PROBE`.** At (a_ℓ +0.156, c +0.133) on the base_fc plane
  one OJS tooth carries three crossings within 9e-4 in θ_S, so the probe 0.001 outward jumps
  over the outermost one. v44 accepted the repelling crossing there, and v45 accepts the inner
  attracting one. The author's rule that sub-cell structure is ignored covers this. It is
  recorded because the Q difference between the two crossings is 1.1e-3 relative, and because
  v44's choice was a repelling root.
- `anderson_reg` is a `SimParams` field that no code reads, since `anderson1_update!` uses
  its own 1e-12 guard. Retiring it needs a bundle migration and is not done here.
- `damp_pstar_U` is read only when `use_anderson = false`, which no entry point selects.
- `smm_main.jl` was not run end to end: the author runs estimations himself, and no optimiser
  is started for a release gate. The SMM layer is unchanged in this version.
- The timings were taken on a shared machine. The interleaved ratios are the comparison. The
  absolute seconds are not a cost model.

### Tried and rejected
- `use_anderson = false` in every loop (v43): the plane collapses to 45/361 because the
  unskilled block stops converging. Only the skilled block's accelerator is removed.
- Keeping skilled Anderson behind a switch: rejected by the author, who wants fewer settings.
  The removal leaves no toggle.

---


## v44.0.0 — 2026-09-26

**MAJOR: the objective is scored at the best-fitting attracting skilled equilibrium.** Every
converged solve is followed by a detector that finds every attracting equilibrium of the
skilled market, one or the pair L and H, and Q is the lowest Q among those that pass the full
residual battery. Where v43 scored whichever root the solver converged to, including the
repelling middle root M and sub-cell roots made by one type entering on-the-job search, the
estimate can move. The theorist has to be told.

**Comparability.** A candidate with a single equilibrium keeps its v43 Q bit for bit: 387 of
388 test points (below), including the base_fc, base_covid and crisis_covid estimates. The
one change is the **crisis_fc estimate**: Q(θ̂) goes from 970.350485 to 970.511641. v43 had
converged to a tooth crossing at θ_S = 0.655916, and v44 uses the outermost H crossing at
0.657687; L at 0.606479 scores 1286.056069. `estimate_crisis_fc_diagonalW.jls` still loads,
but its `loss_opt` is the v43 value and θ̂ was optimised under the v43 objective, so it should
be re-estimated under v44. The other three bundles are not superseded. No struct changed, and
every bundle loads. Reported standard errors change wherever the local design around θ̂
crosses a fold: a design point is now scored on θ̂'s own branch or not at all. At base_fc,
32 of the 240 design points (unique L, where θ̂ is H) leave the Ĝ regression.

### What changed
- `solver/skilled.jl:762`, `solver/solver.jl:133,223,246`: a keyword `θS_fixed` on
  `solve_skilled_block!`, `solve_model!`, `_global_loop!` and `global_residuals` holds θ_S
  fixed and solves every other object given it. Skilled free entry leaves `R2` in that case.
  It is a keyword, not a `SimParams` field, because a field would break every serialised
  bundle. Without it, the code path is operation-for-operation unchanged.
- `solver/solver.jl:167`: `implied_cutoffs`, factored out of `cutoff_residual`, which now
  calls it. It also returns the participation score M that locates the onset.
- `solver/solver.jl:381-755`: `skilled_equilibria(model)`, the detector. The section header
  states the two properties of D = ln(θ̃/θ) it relies on and the curves they were measured on.
  It works in five steps. First, a step either side of the normal solve's root: a rise is M,
  and one sign on both sides is a sub-cell root, which is moved to the step-resolved root the
  signs point to. Second, the OJS onset, bracketed on the exact participation indicator to
  0.02 in ln θ. Third, classification: nobody searching means L, beyond 1.08 × onset means H,
  and otherwise curvature decides. Fourth, one jump to 1.08 × onset, past every measured dip,
  then a walk to the top (from L) or the dip (from H). Fifth, only a missing attracting root
  is located, by regula falsi, then certified with the full battery. Every root is probed
  0.001 outward, so the outermost crossing of a tooth pair is used.
- `smm/smm.jl:169,191,241`: `select_equilibrium(θ, spec; branch)` and
  `select_equilibrium(model, spec; branch)` are the single path to "the equilibrium at θ".
  `smm_objective` wraps them. `branch = :L/:H` holds one side of the fold, and a lone root
  with no hump (`:U`) qualifies for either. The run report prints the selected root
  (`smm.jl:2668`, `print_results`).
- `mcmc/MCMC_main.jl:886`: the seed's label is held by every FD step, the preconditioning
  design and the local design (`branch_hat`). Chain draws are scored as the chain scores
  them. The `mom` contract in `mcmc_diagnostics.jl` states the requirement.
- Consumers: `transition/transition_simulation.jl:146` (base and crisis steady states),
  `plots_and_tables/transition.jl:177,412,944` (moment table, the legacy FD SE with the
  optimum's branch held, x̄), and `solver/model_main.jl:339` (against the loaded bundle's
  targets).

### Measured (scratch copy, 4–10 threads; serial timings on 1)
| case | detector | brute-force sweep, 0.25% grid | evaluations |
|---|---|---|---|
| crisis_fc k×1.012 | L 0.598699 | 1 root 0.598699 | 4 |
| crisis_fc k̂ | L 0.606479, H 0.657687 | L 0.606479, M 0.651316, H 0.657721 | 23 |
| crisis_fc k×0.995 | L 0.610612, H 0.671231 | L 0.610666, M 0.636193, H 0.671203 | 22 |
| crisis_fc k×0.985 | H 0.683523 | 1 root 0.683528 | 17 |
| crisis_fc k×0.995 from M (0.63609) | start = M; L 0.61061, H 0.67123 | — | 24 |
| base_fc θ̂ | H 1.211645 (dip D = +0.0173) | 1 root 1.211755 | 19 |
| base_fc k×1.02 | L 1.058814, H 1.189748 | L 1.058814, M 1.103517, H 1.189815 | 21 |
| base_covid, crisis_covid θ̂ | U 1.812483, U 2.321072 | 1 root each | 24, 23 |

- Detector vs sweep on 35 curves: the estimates, ten k_S scalings, 20 joint ±25% draws around
  base_fc and crisis_fc, and the one multiple-equilibrium design point. Same count on 35 of
  35; every root within 1.0e-4 in ln θ of the sweep's outermost crossing.
- Bitwise Q against v43 on 388 points: the four estimates, 36 joint ±25% draws per window,
  and the 240-point base_fc local design. 387 identical, including the 2 points infeasible in
  both versions. The exception is crisis_fc θ̂. The one design point with two equilibria
  (base_fc design 120: L 0.754963, H 0.837369) keeps its Q, because the normal solve's root
  already scored lower. Detector evaluations: mean 22–25 per window, max 30; no incomplete
  detection among converged solves.
- Q at both roots: crisis_fc θ̂ L 1286.056069 / H 970.511641; crisis_fc k×0.995
  L 1257.570885 / H 1034.115757; base_fc k×1.02 L 2783.663043 / H 1113.345265.
- Branch holding: crisis_fc k×1.012 gives Q(:H) = Inf, Q(:L) = Q(:best); k×0.985 gives
  Q(:L) = Inf; at base_fc θ̂, Q(:H) equals Q(:best) bitwise and Q(:L) = Inf.
- Cost per candidate, serial: normal solve 1.16 s, selection 4.09 s (+2.93 s, ×3.6) with one
  root (n = 35); 1.41 s → 5.24 s (+3.83 s, ×3.7) with two (n = 5). One evaluation takes
  0.13–0.17 s, and the walk to the onset is most of the count.
- `check_threading` passes, and so does an 8-thread repeat at crisis_fc k̂, k×0.995 and
  base_fc k×1.02 (bitwise equal to serial). `check_dead_settings` and `check_output_tree`
  pass. `check_forwarding` reports one finding and `check_structure` seven; both counts are
  identical on v43, in files this version does not touch.
- The transition stationary stage selects "unique" at base_fc and "H of 2" (θ_S 0.6577) at
  crisis_fc.

### Settings
None added. The detector's constants (`EQ_STEP`, `EQ_PROBE`, `EQ_MAXEVAL`, `ONSET_*`,
`solver.jl`) are properties of the measured curves, not run settings.

### Known gaps
- Where the normal solve fails, the detector does not run: the point stays infeasible, as in
  v43. At base_fc k×1.04 the solve stalls on a sub-cell root at 1.1419, while the detector,
  started from that state, finds the certified L at 1.037295 (sweep: 1.037295). Rescuing such
  points would enlarge the feasible set, which is a decision for the theorist.
- Assumptions A and B, the 1.058 dip bound and the curvature rule are measured, not proved.
  A detection that exhausts `EQ_MAXEVAL` falls back to the normal solve's root with
  `complete = false`; none occurred on the 388 points.
- `model_main.jl` could not be run end to end in the test sandbox (Plots/GR failed to
  precompile there). Its new block was exercised through `select_equilibrium(model, spec)`.
- `policy/policy_solver.jl` calls a `solve_model` method with a `RegimeParams` argument that
  no longer exists, so that path is stale independently of this change and was not routed.

### Tried and rejected
- Aiming onset jumps from the linear rule θ_on ≈ θ exp(−M/4515) alone: M is near-linear in
  ln θ only close to the onset. Far from it, the uncapped secant jumped to θ_S = 37.7, where
  the frozen solve fails, and landings short of the onset left walks of 200 evaluations. On
  the 388 points this left 19 detections incomplete. Replaced by the bracket on participation.
- Classifying a rise across θ* as M without checking signs: at base_fc k×1.04 a sub-cell root
  rises while negative on both sides, and the search for a non-existent H ran to the cap.

---

## v43.0.0 — 2026-09-24

**MAJOR: a market with no vacancies is scored.** When free entry closes a market (tightness at
its floor, so no one is employed there even in the limit of its composition), its wage moments
and both cross-market overlaps are now 0 rather than NaN, so the point returns a finite `Q`
instead of `Inf`. The feasible set is larger; the theorist has to be told.

**Comparability.** Every point that was finite under v42.0.0 is bitwise unchanged: `Q(θ̂)` in
base_fc, base_covid and crisis_fc, and 240 of 240 local-design points. Stored bundles load and
no stored output is superseded. The MCMC_main error for an unmeasurable FD column is reworded
(it no longer points at the bundle's optimiser `converged` flag); no number changes.

### What changed
- `smm/moments.jl`: `_logwage_moments` returns eight zeros for an empty density; the overlaps
  are 0 when either side has no wage distribution. `model_moments` header comment updated.
- `smm/smm.jl`: `smm_objective` docstring states the convention; the finite-moment check stays,
  for moments the equilibrium could not compute.
- `mcmc/MCMC_main.jl`: FD error text.

### Measured
At base_covid:

| point | v42 | v43 | skilled / unskilled mean log wage |
|---|---|---|---|
| `P_S × 0.07`, `× 0.005` (skilled closed: every cell drains) | Inf | 2.790e+07 | 0 / 6.268 |
| `P_U × 0.05` (unskilled closed: no viable match) | Inf | 1.890e+07 | 6.690 / 0 |
| `P_S × 0.1`, `P_U × 0.1` (open) | 1.202e+06, 1.070e+07 | unchanged | |

`check_threading` and `check_dead_settings` pass.

---

## v42.0.0 — 2026-09-24

**MAJOR: R1 tests where the cutoffs are, not the surplus at a node.** The equilibrium test
now accepts every point it accepted before, at the same `Q`, plus points it used to reject
for a reason that came from the grid. So the feasible set of the estimator is larger, and the
theorist has to be told.

**Comparability.** Every point that passed under v41.0.0 passes here. Its `Q` is bitwise equal
except at 2 of 574 points tested, which the old test had stopped early (see below). Stored
bundles load unchanged. `Q(θ̂)` is unchanged in every window tested. No stored output is
superseded. A `:design` Jacobian is not comparable, because it now regresses on more
feasible draws.

### What changed

**R1** (`solver/solver.jl:161` `cutoff_residual`, called from `global_residuals` at :128).
- **New definition.** The skilled values are re-solved at the installed cutoffs and tightness,
  on a copy of the skilled cache, so the state is only read. The cutoffs those values imply
  are then compared with the installed cutoffs, in units of p:
  - `p*_S` at every type;
  - `p^oj` wherever `s_part > 0`.
- **Tolerance.** `TOL_CUTOFF = 1e-6` (:92), the skilled block's own step tolerance.
  `TOL_INDIFF` is retired.
- **Old definition.** It read the stored surplus at the first p-node above each cutoff. At an
  exact equilibrium that reports roughly the surplus slope times the distance from the cutoff
  to the node. The node test was also blind to a misplaced cutoff (measured below).

**The cutoff map** (`solver/skilled.jl:689` `cutoff_proposals!`). The per-type cutoff scan
moved out of `solve_skilled_block!` unchanged. The block and R1 now read one routine. The
block's unused locals went with it.

**README** battery table updated.

### Measured, not asserted

R1 at θ̂ as a ratio to its gate, p grid ×1 / ×2 / ×4:

| | node (v41) | cutoff (v42) |
|---|---|---|
| base_fc | 0.123 / 0.085 / 0.060 | 5.9e-6 / 1.6e-7 / 1.2e-7 |
| base_covid | 0.033 / 0.020 / 0.010 | 1.9e-9 / 2.3e-9 / 2.9e-9 |

The converged base_fc state, perturbed one object at a time:

| perturbation | node R1 | cutoff R1 | R2 |
|---|---|---|---|
| none | 0.123 | 5.9e-6 | 2.6e-5 |
| p\* +1 cell | 0.123 (**blind**) | 234 | 2.6e-5 |
| p^oj +1 cell | 0.150 (**blind**) | 945 | 152 |
| θ_S +5% | 0.123 | 1.9e4 | 4.8e4 |
| θ_U +5% | 0.123 | 5.9e-6 | 4.8e4 |

θ_U is correctly left to R2. The residual leaves the state bitwise untouched.

Coverage against v41.0.0:

| set | finite v41 | finite v42 | gained | lost | `Q` bitwise equal |
|---|---|---|---|---|---|
| local design, base_fc (120) | 112 | 120 | 8 | 0 | 112/112 |
| local design, base_covid (120) | 120 | 120 | 0 | 0 | 120/120 |
| 19×19 (a_ℓ, c) plane, base_fc | 342 | 342 | 0 | 0 | 340/342 |

- **The two plane points that moved** (by 3.5e-07 and 3.7e-08 relative). The old R1 let the
  global loop stop while the cutoff map was still moving: the cutoff R1 read 1.13 and 2.31 on
  passes the node R1 accepted. They now take 9 and 11 passes instead of 8.
- **Everybody trains, base_covid** (`c − 16`, `b_S = 5…30 b_U`, shipped grid). This corner was
  rejected everywhere by R1 (1.6 to 35) and now converges at every step, with Rmax ≤ 2.5e-07.
- **`P_S × 0.1`**: gained.
- **Nobody trains** (`c` path): unchanged.
- `JAC_ONLY :fd`, base_fc: 23 of 23 coordinates in 278 solves, 2 not converged. The table is
  identical to v41.0.0.
- `JAC_ONLY :design`: 598 of 600 feasible, against 549.
- `check_threading`: threaded `Q` is bitwise equal to serial. `check_dead_settings` passes.

### Known gaps
- **Markets with no vacancies** (`P_S × 0.05`, `P_S × 0.005`). They converge and now pass R1,
  but still return `Inf`: no one is employed even in the limit, so the wage moments are
  undefined. This is unchanged from v41.0.0, and it is a question for the model.
- **The plane's 19 failures** are the same 19 under both forms of R1. They lie on a diagonal
  band at high a_ℓ (rows 14 to 18) and are not R1. Not diagnosed.
- `check_forwarding`: the one finding is the pre-existing obsolete row
  `ROYSEARCH_MCMC_JAC_ALLOW_GAPS`.

### Tried and rejected
- **The surplus read at the installed cutoff**, re-solved and linearly interpolated between
  nodes. This keeps the surplus-based reading adopted in v40.0.0 while removing the node. It
  failed the base_covid estimate itself (R1 = 1.23).
  - It lost 90 of 112 base_fc design points and 113 of 120 base_covid points.
  - It lost every corner step on all three walks.
  - It was as blind as the node form to a moved `p*` or `p^oj`.
  - Linear interpolation of a curved surplus still carries a grid error at its own zero.
- **Keeping the node reading and scaling `TOL_INDIFF` by cell width.** Not run. That would
  make the gate depend on the grid, and it would stay blind to a misplaced cutoff.

---

## v41.0.0 — 2026-09-24

**MAJOR: corner equilibria are equilibria.** A converged equilibrium in which nobody trains,
everybody trains, or a market holds no one used to be rejected with `Q = Inf` by two guards in
`smm_objective`. It is now scored. Its within-market moments are read at the **limit of the
market's composition as it empties**. The theorist has to be told, because the feasible set of
the estimator is larger.

**Comparability.** Every interior point is bit-identical to v40.1.0: the estimates, the
standard errors, and every stored `Q`. Stored bundles load unchanged. Only points that v40.1.0
rejected as corners can now return a finite `Q`. No stored output is superseded.

### What changed

**The limit composition** (`solver/unskilled.jl:218`, `marginal_training_cells`). Markets empty
through the training margin. The last cells to leave a market are therefore the first to come
back: for the skilled market, the cells with the largest training gain `Utr(a_S) − U^search(a_U)`;
for the unskilled labour force, the cells with the smallest. The limit is a unit mass on those
cells, with ties shared in proportion to `W2`. The model's own stationary formulas turn it into
a composition, so it is weighted exactly as a small positive mass would be:
- the skilled side goes through `_mS_from_t` and the drain split;
- the unskilled side goes through `solve_stationary_unskilled!`.

The training value `T − c` now has one home, `training_value!` (`unskilled.jl:197`), which
replaces three copies of the loop.

**Free entry at an empty market** (`skilled.jl:536` `skilled_stayer_columns`,
`unskilled.jl:425`). A vacancy in a market that holds no one is priced at the limit
composition. This rule applies only once the solve has converged at a corner:
- `solve_model!` (`solver/solver.jl:199`) runs the unchanged loop (`_global_loop!`);
- if the result is a corner (`market_empty`), it continues from that state under the limit
  rule (`Model.corner_limit`, `params.jl:374`).

The reason for this order is below: applying the limit rule during iteration loses interior
points.

**Equilibrium objects** (`solver/equilibrium.jl:58`). Rates, wage densities, survivors,
unemployment rates and job-finding rates are ratios over a market's own mass. They are now
computed from the market's composition: its masses when it is open, the limit when it is not.
Aggregates and the returned mass arrays keep the true masses, which are zero for an empty
market. The corner-zeroing block and its dead `agg_eU_raw`/`agg_eS_raw` are gone.
`EMPTY_MASS = 1e-12` (`grids.jl:254`) is the single threshold.

**Moments** (`smm/moments.jl:501`). All empty-market placeholders are removed:
- `ur_S = 1`, `jfr = 0`, wages `= 0`, `θ = 1e14`, overlaps, LTU share and survivors `= 0`.

Moments are read as the equilibrium objects define them. A wage distribution with no one
employed even in the limit has no moments: `_logwage_moments` returns NaN (`moments.jl:413`).

**Objective** (`smm/smm.jl:194`). Both guards are removed: the empty-employment guard and the
degenerate-τ guard. They are replaced by one rule: a point whose scored moments are not all
finite returns `Inf`.

**Also in this version** (made earlier on 2026-09-24, unversioned until now):
- `local_design` (`mcmc/mcmc_diagnostics.jl:916`): a coordinate within one step of a bound
  steps inward only. Interior coordinates keep bit-identical draws.
- `MCMC_main.jl:1166`: the design regression runs only when something reads it (`:design`, or
  a chain's draws). The FD line reports the number of candidates that did not converge.

### Measured, not asserted
- Serial `Q(θ̂)` matches v40.1.0 to the last digit: base_fc 1050.1532564386 and base_covid
  2834.3263453681.
- Over 240 local-design points (120 per window, same draws):
  - finite: 112/120 in base_fc and 120/120 in base_covid, identical to v40.1.0;
  - every finite `Q` is bitwise equal: 232/232;
  - none lost and none gained.
- `check_threading`: threaded `Q` is bitwise equal to serial. `check_dead_settings` passes.
- `MCMC_main` JAC_ONLY `:fd`: 23 of 23 coordinates in 278 solves, 2 not converged. The table
  is identical to the v40.1.0 run.
- `MCMC_main` JAC_ONLY `:design`: 549 of 600 feasible, up from 321 before the `local_design`
  change.
- **Nobody trains, base_covid, walking `c` upward.** Every point converges, and `Q` and every
  moment are continuous through the corner. The training share falls 3.3e-05 → 1.6e-07 →
  9.5e-11 → exactly 0 (from `c + 11`). `θ_S`, `jfr_S`, `ur_S` and `mean_wage_S` are 3.427,
  0.3756, 0.02057 and 7.60 on both sides. `Q` is 5.629e+05 on both sides. v40.1.0 returned
  `Inf` from `c + 11`.
- **Everybody trains, unskilled limit** (base_covid, `c − 16`, `b_S/b_U` from 12 to 20,
  Np_S ×4). The unskilled moments are continuous through the corner:
  - `θ_U` 2.195 → 2.733 → 2.808;
  - mean log wage 7.18 → 7.38 → 7.41;
  - `sep_rate_U` 0.0123 → 0.0139 → 0.0142.

### Known gaps
- **The everybody-trains corner is still unreachable at shipped settings, because R1 fails
  it.** This is a solver test, not a guard. At the default Np_S, R1 is 2.55 at `b_S = 8 b_U`
  and 35.3 at `30 b_U`. At Np_S ×4 it is 0.69 and 0.96 at 8 and 12, which pass, and 2.97 to
  9.24 from 16 upward. R1 halves with every refinement at every corner point probed, which is
  the same grid effect as at the interior. It also grows as the market empties. Not yet
  diagnosed.
- **A market with no vacancies has no wages.** At `P_S × 0.05`, and at `P_U × 0.01` in the
  unskilled market, free entry drives tightness to its floor and no one is employed, even in
  the limit. The wage moments are undefined and the point returns `Inf`. This is the only
  corner still rejected, and it is a question for the model, not the code.
- `check_forwarding`: the one finding is the pre-existing obsolete row
  `ROYSEARCH_MCMC_JAC_ALLOW_GAPS`, which is untouched.

### Tried and rejected
- **Pricing free entry at the limit composition from the first iteration.** The skilled
  market is empty on the first passes of an interior solve, because `m_S` has not yet received
  the training flow. In 18 of 95 calls at the base_fc estimate, the limit rule changed the path
  the iteration took:
  - `Q(θ̂)` moved 6.2e-09 relative;
  - no interior `Q` of 232 stayed bitwise equal;
  - 5 of 240 interior points were lost, 4 of them to the skilled block failing to converge.

  Gating the rule on the training flow `t` instead of `m_S` changed nothing, because `t` is
  already positive when `m_S` is still zero. The iteration keeps its population-weighted rule,
  and the limit is reached by continuation from a converged corner.

---

## v40.1.0 — 2026-09-23

**MINOR: the solver's two `@threads :static` loops are rewritten as a chunked
`@sync`/`@spawn` fan-out with task-local scratch. Every converged result reproduces
bitwise; what changed is that a threaded evaluation now returns a number at all.**

**v40.0.0 could not run a threaded evaluation.** That is the plain statement this entry
exists to make. Its serial path was exact — `Q(θ̂)` = 1071.2814042604 on `base_fc`, the
number every v40.0.0 report quotes — but a solve called from inside any
`Threads.@threads` loop raised `@threads :static cannot be used concurrently or nested`,
`smm_objective` caught it and returned `Inf`, and the caller read a solver that cannot
nest as a parameter point that has no equilibrium. Every caller that evaluates the
criterion at more than one point at a time does so with `@threads`, so the whole
estimation path above a single solve was disabled: the ΔQ=1 width measurement
(`smm.jl:_feasible_widths`) returned 0 of 23 measurable coordinates, the adaptive-FD
Jacobian (`mcmc_diagnostics.jl:295`) 0 of 23 columns in 814 solves, and the DE at
`pop_size` = 240 would have handed over an all-`Inf` population. SA was unaffected only
because it evaluates serially — and therefore ran **detuned**, on the uniform 0.01 step
fallback rather than on measured per-coordinate widths.

The regression was invisible because every test that certified v40.0.0 ran serially.
`code/gates/check_threading.jl` is the gate that does not; see below.

**Where `:static` came from, and why this supersedes rather than reverts.** `:static` is not
committed anywhere: `git` HEAD has 0 occurrences in both `unskilled.jl` and `skilled.jl`
against 1 each in the working tree, while the `threadid()`-indexed scratch it was protecting
dates from 2026-03-08 (`6267519`, `Svec_tls` and `dU_tls`) and the threaded DE population
loop from 2026-04-01 and 2026-08-16 (`a7997c5`, `412f80e`). So `:static` was added by the
recent uncommitted determinism work, and `skilled.jl:312–313` said so in its own words:
":static so the id is stable for a whole chunk and no two tasks share a slot." That fix
closed a **real race** — dynamic scheduling letting two tasks share a `threadid()` slot and
corrupt a reduction — by forbidding concurrency, and since nothing in its acceptance test
ran two solves at once, the cost was never measured. The history the author now has is
therefore: from March until the determinism fix, parallel evaluation worked *with* a latent
race, so the earlier DE and Sobol runs were not corrupted by this defect because it did not
yet exist; from the determinism fix onward, all parallel evaluation returned `Inf`.

This change is not a workaround for `:static`. It **supersedes the determinism fix's
remedy** with a better one and does not revert it: task-local scratch removes the shared
slot entirely, so there is no race to protect against *and* no concurrency prohibition. The
invariant that fix was protecting — no two tasks writing one buffer — is now guaranteed **by
construction** rather than by scheduling discipline. Each of the two prior states was
defective in one direction (a race, then an inability to run in parallel); this one is
neither.

The acceptance criterion is the author's: *the model solve should be identical when run by
itself and while running others.* Held to **bitwise** identity, not finiteness and not
agreement to a tolerance — battery items 1, 2 and 6 below, all of which compare with `===`.

### Comparability

**Every stored v40.0.0 result stands and is directly comparable.** The criterion is
unchanged: `Q(θ̂)` on `base_fc` is bitwise identical, and the 19×19 `(a_ℓ, c)` coverage
plane reproduces 356 of 361 with the *same five* infeasible points and every finite `Q`
equal to the v40.0.0 reference at the precision that reference stores. No parameter,
state, moment, or equilibrium object changed; no struct gained, lost, or reordered a
field, so **no bundle migration is owed** and every existing `.jls` still loads.

One consequence is not a comparability break but should be stated: an SA run launched
under v40.0.0 was walking on the uniform 0.01 step fallback, and the same run under
v40.1.0 walks on measured widths (median 0.005322, range 0.000781‥0.2). The criterion is
the same function; the *search path* through it is not, so a v40.0.0 SA trace is not
reproducible under v40.1.0. That is optimiser behaviour, not an estimator change.

### What changed

- `code/solver/grids.jl:26` — **new** `index_chunks(N, [nch])`, the single definition of the
  work partition, with the reason the fan-out is `@spawn` and not `@threads :static` stated
  once where the partition lives rather than twice at the call sites.
- `code/solver/unskilled.jl:204,228` — the `Svec_tls` thread-local vector is gone; the
  surplus loop over `aU` is a `@sync`/`@spawn` fan-out over `chunks`, allocating its `Svec`
  scratch **inside** the task.
- `code/solver/skilled.jl:312,333,406,410` — `tailEo_tls`/`tailEs_tls` become task-local
  allocations; `dU_tls`/`dS_tls` become `dU_ch`/`dS_ch`, one slot per **chunk**, written once
  per chunk from task-local accumulators and reduced with `maximum` in chunk-index order.
- `Threads.threadid()` no longer appears anywhere in `code/solver/`. `@threads :static`
  no longer appears as code anywhere in `code/`.
- The other eight plain `@threads` loops in those two files nest correctly and are untouched.
- `code/gates/check_threading.jl` — **new gate**. Runs a solve inside
  `Threads.@threads for i in 1:2` and fails unless the result is finite **and** bitwise
  equal to the serial value. Wired into `check_repo.sh` with `-t 2`; listed in `README.md`
  and as step 5 of the release gate in `SETTINGS.md`.
- `code/mcmc/mcmc_diagnostics.jl:255` — the `centre_ok` docstring attributed `base_fc`'s
  infeasible centre to "`converged_S` depends on the reduction order in `skilled.jl`". That
  diagnosis was wrong and the behaviour it described is gone; it now records the actual cause.

### Measured, not asserted

Acceptance battery, `base_fc` at its stored optimum, 6 threads unless stated, run from a
scratch copy of the tree.

| # | check | v40.0.0 | v40.1.0 |
|---|---|---|---|
| 1 | threaded vs serial, **bitwise**, at θ̂ and `a_ℓ` +0.1%/+1%/+10% of box | `Inf` threaded | **4/4 bitwise equal** |
| 2 | the DE pattern: 9 points in ONE `@threads` loop | 0 of 9 finite | **9/9 finite, max rel err 0.000e+00** |
| 3 | `@threads for i in 1:1` containing a solve (minimal repro) | `Inf` | **finite, equals serial** |
| 4 | `_feasible_widths` — the SA/NM proposal scale | **0/23** measurable | **23/23** in 552 solves |
| 5 | `jacobian_adaptive_fd` | **0/23** columns in 814 solves | **23/23 in 274 solves**, `centre_ok=true` |
| 6 | same solve 3× | identical | identical |
| 6 | `-t 1` vs `-t 6` | — | identical, 1071.2814042604 |
| 7 | 19×19 `(a_ℓ, c)` coverage plane | 356/361 | **356/361, same 5 points, every finite `Q` equal** |
| 8 | `check_threading.jl` on the fixed tree / on the v40.0.0 loops | — | **exit 0 / exit 1** with the nesting finding |

Item 1, the four points: θ̂ 1071.2814042604; `a_ℓ` +0.1% 1110.8649498833; +1%
4672.9884962874; +10% 75395.6474190237 — serial and threaded agree to the last bit at all
four. Item 2's nine points likewise, `===` on every one.

Item 5's **274 solves against v40.0.0's 814** is itself a check: with a solve that works,
phase 1 predicts its step immediately instead of halving down to `h_lo`. `rank(Ĵ)` = 23 of
23 at pinv tolerance 2.11e-06 (`s_max` 4.12e+08, `s_min` 6.28e-05), so the retained-variance
fraction is 1.0000 for every coordinate and no standard error rests on a truncated
direction. Both CH identities hold exactly, 0 of 23 violations each: `se_bound ≥ se_curv`,
and `|t|_bound ≤ |t|_curv`. The four coordinates clearing `|t|` = 2 on the bound:

| | θ̂ | se(bound) | se(curv) | `|t|`(bound) |
|---|---|---|---|---|
| `μ_U` | 0.444905 | 0.0035242 | 0.00293386 | 126.2 |
| `μ_S` | 0.256265 | 0.00510723 | 0.00185311 | 50.18 |
| `λ_S` | 0.130263 | 0.00350488 | 0.00349699 | 37.17 |
| `ξ_S` | 0.00509751 | 0.000154584 | 7.10502e-05 | 32.98 |

The full 23-row table ships beside this entry as `se_table_base_fc.csv`; it was written to a
scratch directory, not into `output/estimates/`, because a live estimation held that tree
while the battery ran. Every other coordinate
has `|t|`(bound) below 2 — `c` is closest at 1.876, `k_U` the furthest at 0.003705 on an
`se`(bound) of 201.5 — which is a statement about what 35 moments can separate, not about
this fix.

**Why the chunked reduction is at least as reproducible as the one it replaces.** The
partition is a function of `Nx` alone, so the grouping of terms does not depend on how the
scheduler assigned work — where a `threadid()`-indexed reduction grouped them differently
at a different thread count. Measured: `-t 1` and `-t 6` give the same `Q(θ̂)` to the last
bit, and three repeats at 6 threads are identical. This is the property that mattered:
dropping `:static` without the chunk rewrite would have traded a loud `Inf` for a silent
schedule-dependent number.

**Gates.** `check_dead_settings` PASS (177 keyword arguments, 4 files).
`check_forwarding` reports the single pre-existing finding v40.0.0 already records — the
obsolete `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` registry row — and nothing new.
`check_structure` reports the same seven pre-existing net-depth entries as v40.0.0, none of
them in `code/solver/` and none of them a file touched here. All changed files parse with
zero `:error`/`:incomplete` nodes; the solver + SMM + MCMC chain loads in driver order and
`index_chunks` resolves. `check_version_sync` in sync at 40.1.0 across one declaration and
three readers.

### Settings

None added, retired, or re-defaulted. `_WIDTH_CAP` = 0.20 and `WIDTH_DQ` = 1.0 are
unchanged — the widths they govern are simply measurable again.

### Known gaps

- **Not run: an optimiser, a chain, the transition path, or the policy entry point.** The
  brief excluded them and a live SMM run was using the tree. `smm_main.jl` and
  `MCMC_main.jl` are covered by parse plus a driver-order load, not by a run. In particular
  the DE has not been run at `pop_size` = 240; what is measured is that its *pattern* — N
  points in one `@threads` loop — now returns N finite values equal to their serial ones.
- The battery ran on `base_fc` only. The two loops are window-independent, but the
  coverage plane, the widths and the Jacobian are `base_fc` numbers.
- Item 6 was measured at `-t 1` against `-t 6`, not `-t 9`: the run was capped at 6 threads
  to leave the machine to a live estimation.
- The 19×19 plane's definition still lives only in this file (v40.0.0 recorded the same
  gap) and in the verification harness, not in a shipped script. Reconstructed here from the
  v40.0.0 entry — offsets `LinRange(-0.2, 0.2, 19)` in natural units on both coordinates,
  mapped back through the box transform — and it reproduces that entry point-for-point,
  which is evidence the reconstruction is the plane that was measured. Shipping it as a
  script is still open.
- `unpack_θ`'s `constrained = true` path remains broken and untouched (see v40.0.0).

### Tried and rejected

- **Simply dropping `:static`, keeping the `threadid()` scratch.** Rejected without
  measuring, because it is precisely the state the determinism fix was written to escape:
  `threadid()`-indexed scratch is safe only while the id is pinned for a chunk, so under the
  dynamic scheduler two tasks can land on one buffer and the loops return a *silently wrong*
  number instead of a loud `Inf` — strictly worse. `compute_Jbar_skilled`
  (`skilled.jl:568`) carries the note from the last time this was diagnosed: `J̄_S` differed
  in its last bits run to run, which at a sensitivity of ~2e4 on `U_S` made the
  feasible/infeasible verdict itself depend on the schedule. Removing the shared slot is
  what makes dropping `:static` safe; dropping it alone is not.
- **Serialising the nine `@threads` call sites instead of fixing the two loops** —
  `candidates.jl:182`, `smm.jl:591`, `1404`, `1760`, `1806`, `1926`, `2100`,
  `MCMC_main.jl:1014`, `1174`. Measured to work: a serial `jacobian_adaptive_fd` gives the
  same 23 of 23 in 274 solves and the same `se`(bound) values to six significant figures, so
  it is a correct fallback and was held in reserve. Rejected because the chunked rewrite
  passed the whole battery: serialising the call sites leaves the defect in the solver and
  costs the DE its outer parallelism — 240 sequential solves per generation at
  `pop_size` = 240, where a single solve's internal threading does not recover the loss.
- **Hoisting the per-chunk scratch out of the iteration loop** (one buffer array indexed by
  chunk, allocated once per call). Equally safe by construction, since a chunk index is
  unique to one task where a thread id is not. Not adopted: the measured cost of allocating
  inside the task is not visible against the solve — items 4 and 5 run at 552 solves in 222s
  and 274 solves in 122s on 6 threads — and allocating where it is used is the form that
  cannot be made unsafe by a later edit.

---

## v40.0.0 — 2026-09-23

**MAJOR: the global loop now converges on the equilibrium conditions instead of on the step
between iterates.** Three solver changes ship together, and each was measured on the same
19×19 (`a_ℓ`, `c`) plane at `base_fc`. The layer is MAJOR on two independent rows of the
test: a numerical control was **removed from a serialised struct** (`SimParams.tol_global`),
and the solver's acceptance region changed, so `Q` moves at points where the two criteria
disagree — for reasons that are not optimiser noise. The theorist has to be told, because
the headline is that **v39.3.0 could not reproduce its own `base_fc` estimate**: `Q` at the
stored θ̂ returned `Inf`, and under this version it returns 1071.281404.

**Comparability: every stored estimate is preserved, and `Q` at each θ̂ is now finite and
within 1.7e-05 relative of the value the bundle recorded.** No θ moves. The four canonical
bundles were migrated in this pass and reproduce their extract-time reference exactly.
What is superseded is any *coverage* or *convergence-flag* claim made under v39.3.0: a
parameter point that v39.3.0 reported as non-converged may converge here, which is the
intended effect. The two `tol_global` rungs in `README.md`'s probe crosswalk are marked as
produced under the retired criterion.

### What changed

**1. The equilibrium residual battery replaces the global step test.** `solver.jl:124`
adds `global_residuals`, four components each returned as a **ratio to its own internal
tolerance**, gated at `Rmax < 1` (`solver.jl:285`, `:295`; stall rule `:314`).

- `R1` indifference at the installed cutoffs relative to the surplus sup, as two-sided
  complementary slackness: the equality only where the cutoff is interior, and `max(0, −S)`
  at the grid floor. At the `base_fc` estimate `p*_S` sits at the floor for **159 of 160**
  types, so testing the equality there would read the bottom-node surplus as an error when
  nothing is indifferent.
- `R2` free entry in **both** markets, as `|update_theta_j(model) − θ_j|/θ_j`. Those
  functions return the tightness the condition implies, so this is the equation and not a
  step between iterates.
- `R3` the training choice, τ recomputed by `training_policy!`.
- `R5` the education margin, `T` recomputed by `training_frontier!`. R1–R3 are all
  satisfied by the degenerate state where nobody trains; `R5` is the only component
  coupling the two markets and is what rules that state out.

Tolerances are internal constants, not settings — `TOL_INDIFF` = 1e-3, `TOL_RELATIVE` =
1e-6, `TOL_POLICY` = 1e-6 (`solver.jl:93`). **There is no `R4`**: the population adding-up
residual would go there and is vacuous, since (E5) defines `m_U := ℓ − m_S`; it reads
exactly zero everywhere and can detect no non-convergence.

**2. `conv_S` uses interpolate-and-persist instead of the sup-norm** (`skilled.jl:882`).
The sup-norm let one ability type at the participation frontier veto the other 159: that
type's `p^oj` level is stable to five decimals while its per-iteration increment amplifies.
Where between 1 and `CONV_S_MAXOUT` = 2 types exceed `tol_outer_S` and the rest are inside
it, each outlier's cutoff is replaced by a linear interpolation in ability from its nearest
in-tolerance neighbours, that value is required to be stable across consecutive passes
(`skilled.jl:898`), and convergence is judged on the sup-norm over the remaining types. Any
other count falls back to the plain sup-norm. **The interpolation decides convergence only
and never writes `sc.poj`** — installing it would mean the moments were computed from a
cutoff no equilibrium condition was ever checked against.

**3. Adaptive under-relaxation on skilled free entry** (`skilled.jl:812`). The weight halves
on a sign reversal in the undamped target (floor 0.1) and relaxes back by ×1.2 toward
`DAMP_THETA_S[]`, which becomes a **ceiling** rather than the weight. State is local to
`solve_skilled_block!` (`skilled.jl:688`), deliberately: a module-level `Ref` carrying it
between solves would make results depend on the order in which draws are evaluated.

**4. `tol_global` retired from `SimParams`** (`params.jl:298`). With `Rmax` a ratio gated at
1 there is no threshold for it to carry, and a stored `1e-4` must not silently come to mean
something new. Construction sites updated in `params.jl`, `solver/model_main.jl`,
`smm/smm_main.jl`, `policy/policy_main.jl`, `transition/transition_simulation.jl`. The
global verbose trace now prints the four components rather than `ΔU_S`/`Δm_S`, which
nothing gated on.

**5. `ojs_continuity_probe.jl` precision rungs repointed to the block tolerances**
(`:98`). Its `jump`, `crossing` and `jacobian` modes varied `tol_global`; the global loop no
longer has a tolerance to vary, so `PROBE_TOL` and the rungs now move `tol_inner` and
`tol_outer_{U,S}` together, which is what sets how precisely a solve is resolved. Default
is `bundle` (the bundle's own values) and the CSV tag names the rung.

### Measured, not asserted

**`Q` at θ̂, all four windows** — `Inf` at `base_fc` before, finite everywhere after:

| window | stored `loss_opt` | v39.3.0 | v40.0.0 | rel. gap to stored |
|---|---|---|---|---|
| `base_fc` | 1071.280270 | **Inf** | 1071.281404 | 1.1e-06 |
| `crisis_fc` | 713.209187 | not measured | 713.208772 | 5.8e-07 |
| `base_covid` | 2834.373205 | not measured | 2834.326345 | 1.7e-05 |
| `crisis_covid` | 1816.503256 | not measured | 1816.503405 | 8.2e-08 |

**Why v39.3.0 returned `Inf` at its own estimate.** Traced pass by pass: `θ_U` and `θ_S` are
pinned to four decimals from pass 4 and `Δm_S` reaches 1e-13, while `ΔU_S` oscillates
between 1.0e-05 and 2.1e-04 against an **absolute** `tol_global` = 1e-4 — a relative 8e-10
on a `U_S` of order 2.5e+05. `conv_streak` = 4 never completes and the stall rule stops the
solve at pass 13 with `d` = 1.912e-04. The battery reads `Rmax` = 0.1834 at that same state
and converges at pass 8.

**The battery's separation, at the `base_fc` estimate.** Each break perturbs the converged
state off one condition and fires on that component and no other:

| state | `Rmax` | `R1` | `R2` | `R3` | `R5` |
|---|---|---|---|---|---|
| equilibrium | **0.1834** | 0.1834 | 3.31e-05 | 0 | 6.58e-06 |
| `θ_S` × 1.05 | 4.762e+04 | 0.1834 | **4.762e+04** | 0 | 6.58e-06 |
| `θ_U` × 1.05 | 4.762e+04 | 0.1834 | **4.762e+04** | 0 | 6.58e-06 |
| τ shifted one ability cell | 1.000e+06 | 0.1834 | 3.31e-05 | **1.000e+06** | 6.58e-06 |

5.4 orders between the equilibrium and the free-entry breaks, 6.7 to the τ break.

**`R1`'s three grid floors, relative to the surplus sup**, which is why the OJS margin is
read only where the participation fraction is strictly interior: `S = 0` crossings floor at
**2.3e-07**, interior OJS crossings at **1.8e-04**, fully-participating OJS crossings at
**1.8e-03**. One tolerance cannot straddle 2.3e-07 and 1.8e-03. At this estimate 81 of 160
types have `s_part` = 0, 78 have `s_part` = 1, and the single interior type is what `R1`
reads — so `Rmax` = 0.1834 is that one type's crossing.

**Coverage, 19×19 = 361 points, `base_fc`.** Offsets of ±0.2 in natural units on both
`a_ℓ` and `c`, 19 per axis (`LinRange(-0.2, 0.2, 19)`, step 0.02222), mapped back through
the box transform so every other coordinate stays bit-exact: `a_ℓ ∈ [0.868, 1.268]`,
`c ∈ [10.861, 11.261]`.

| version | finite | note |
|---|---|---|
| v40.0.0 | **356 of 361** | 5 failures, all at the high-`a_ℓ` edge (rows 18–19) |
| v39.3.0, same plane | 270 of 361 | failures concentrated at low `a_ℓ`: 11, 6, 9, 6, 8, 8 in rows 1–6 |

**On this plane v40.0.0 is a strict improvement: 86 points gained, 0 lost.** Its finite set
is a superset of v39.3.0's — all five points it fails are also failures under v39.3.0 —
so nothing that solved before stops solving. The two versions fail in different regions:
v39.3.0's failures mass at low `a_ℓ` (11 of 19 in row 1), which v40.0.0 clears entirely,
and the five that remain sit at the high-`a_ℓ` edge where v39.3.0 also loses 6.

**Bundle migration.** All four rebuilt and installed, each reproducing the reference `Q`
recorded at extract time at **Δ = 0.000e+00**, `dropped` = `sim.tol_global` (plus
`bundle_sim.tol_global` for three), `unrecorded` = 0 — no field was reconstructed from a
default. `verify` passes on all four. Originals at `*.jls.pre_migration`.

**Gates.** `check_forwarding`, `check_dead_settings` and `check_structure` are
**byte-identical before and after**: one pre-existing `check_forwarding` finding (the
obsolete `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` registry row) and the same seven pre-existing
`check_structure` net-depth reports, none in `code/solver/`. All 12 touched files parse; the
solver and SMM chain loads; `smm_objective` at θ̂ reproduces post-migration.

### Settings

`tol_global` **retired** from `SimParams` — see `SETTINGS.md`. It was never env-settable and
has no `SETTINGS_REGISTRY` row, so no `LEGACY_ENV_KEYS` entry is owed.

`DAMP_THETA_S` **re-defined from a weight to a ceiling**, value unchanged at 0.9;
`SETTINGS.md`'s entry is extended for the adaptive schedule and for the contraction
condition below. `PROBE_TOL` **re-defined** from the global tolerance to the block
tolerance, default `1e-4` → `0` meaning the bundle's own. `CONV_S_MAXOUT` = 2 and
`TOL_INDIFF`/`TOL_RELATIVE`/`TOL_POLICY` are internal constants, deliberately **not**
settings: they are properties of the conditions and of the grid, and a run that could vary
them could pass a state no equilibrium condition certifies.

### Known gaps

- **Under-relaxation cannot stabilise the oscillating roots, and the adaptive schedule
  therefore SELECTS an equilibrium on the multiplicity band.** The skilled outer map's slope
  in θ_S is `(1/η_S)·dlnJ̄_S/dlnθ_S`, contracting iff `dlnJ̄_S/dlnθ_S < η_S = 0.5`; on the
  ridge that measures 0.52–0.90, i.e. slope `b` = 1.04–1.80, and the damped slope
  `1 + w(b−1) > 1` for every `w ∈ (0,1]`. A shorter step lets the iteration be repelled from
  the unstable root onto a different, stable one. Of the 3 points adaptive damping recovers
  over fixed `w` = 0.9, two are genuine numerics and one is a re-rooting: `a_ℓ+0.2000
  c−0.0333` gives `θ_S` 1.099861 adaptive against 1.012626 at `w` = 0.5, relative Δθ_S
  7.9e-02 and Δ`Q` 4.3e-02. The other two agree to 2.2e-09 and 1.3e-07. **θ̂ and the controls
  are on the same root under both weights** (`Q` agreeing to 2.1e-11), so no reported
  estimate is branch-dependent and no published number moves. The band is ~50 of 361 points
  (13.9%) and carries `Q` ≥ 6210 against 1071 at θ̂, so no optimum lies in it — but a chain
  walks through it, so the selection is not purely academic.
- **`R1` does not check the OJS margin for fully-participating types.** Their crossings
  floor at 1.8e-03, a decade above `TOL_INDIFF`; reading them would fail every solve and
  loosening the tolerance to admit them would cost the `S = 0` half four decades of
  sensitivity. Whether the OJS crossing needs a tolerance of its own is a specification
  question, not a porting one, and is left open.
- **The `conv_S` stability test matches interpolants by position.** An outlier set that
  changes between passes while keeping its size is compared type-against-type. The residual
  battery is the backstop: a block exiting on that shows up as `R1` or `R2` above 1 in the
  global loop, so the state is not certified as converged.
- **The v39.3.0 coverage baseline does not reproduce at 348 of 361, and the ladder's lower
  rungs are not all the shipped configuration.** Measured here on the plane above, raw
  v39.3.0 gives **270 of 361** (74.8%), not 348. The tree used for that
  arm is a faithful v39.3.0 — `params.jl` differs only by `tol_global`, `grids.jl`,
  `unskilled.jl` and `equilibrium.jl` are byte-identical, every damping and participation
  constant is equal, and `Q` at θ̂ reproduces `Inf` — so the arm is not the discrepancy.
  Two things are: **the plane's definition is recorded nowhere in the tree**, so the
  reconstruction above may not be the plane the ladder was measured on; and the working
  record of the ladder is internally inconsistent about the **worker-firm rule**, which
  shipped v39.3.0 applies unconditionally at both `Smax_j` sites. One account has the 348
  arm run without that rule and adding it to the battery arm costing 33 points (351 → 318);
  another has every rung run with it. On the first reading the shipped v39.3.0
  configuration was never measured at all, and 348 is an envelope-rule arm. The top rung is
  unaffected — v40.0.0 reproduces 356 of 361 exactly, and that arm is unambiguously the
  shipped configuration. The improvement itself is not in doubt — on a common plane it is
  270 → 356 with nothing lost — so what is open is the ladder's *lower rungs*, not the
  verdict. Defining the plane in a shipped script, and re-measuring those rungs in the
  shipped configuration, are both left open.
- The four migrated bundles carry `provenance.version` = "39.3.0", the value of the
  constant at write time: the release gate requires the migration to pass before a version
  number is assigned, so the rebuild necessarily ran before the bump. The `migrated_from`
  trail plus this entry identify it as the v40.0.0 migration. Nothing reads that field for
  control flow.
- Not run: an optimiser, a chain, the transition path, or the policy entry point.
  `transition_simulation.jl`, `policy_main.jl`, `MCMC_main.jl` and `mcmc_diagnostics.jl`
  are covered by parse plus the shared `SimParams` constructor, not by a run.
- `unpack_θ`'s `constrained = true` path is **broken, pre-existing, and untouched**: step 3
  re-applies `_to_constrained` unconditionally (`smm_params.jl:1077`), so every
  block-shared parameter is transformed twice and `Q` at θ̂ returns `Inf` on that path
  against 1071.281404 on the standard one. It is the path `MCMC_SPACE = :theta` uses. Not
  fixed here because it is outside these three changes and fixing it would move a sampling
  path this pass does not test.

### Tried and rejected

- **A population adding-up component as a fifth residual.** It is construction under (E5),
  reads exactly 0 everywhere, and cannot detect non-convergence. A vacuous component is not
  free: it is somewhere for an arithmetic slip to hide.
- **A population-weighted MEAN over the boundary for `conv_S`**, in place of
  interpolate-and-persist. It scores higher on coverage (355 of 361 against 353 for the
  interpolation alone), and it is wrong: `wa_S` is the Beta ability marginal with a minimum
  of 2.96e-09, so 20 of 160 types can move their cutoff by more than 1e-2 undetected. It was
  caught passing two points with `Rmax` = 3811 and 3658.
- **A fixed `DAMP_THETA_S` = 0.5.** Measured at 348 of 361, a regression against 353 at the
  fixed 0.9 — it fixes the oscillating minority and costs the contracting majority.
- **Reading the OJS indifference for every searching type** (participation fraction > 0
  rather than strictly interior). Measured: `R1` = 1.8188 at the `base_fc` equilibrium
  against 0.1834, i.e. above the gate, so no solve would converge. The exclusion is the
  grid, not slackness.
- **Deferring the relative-norm fix to its own version**, which the superseded comment at
  `solver.jl` proposed. The battery makes it moot: normalising the step by the sup of its
  own iterate would still be a step test on a map that is not a contraction.

---

## v39.3.0 — 2026-09-23

**MINOR: a bundle now records the version that produced its estimate, and the dead
`anderson_m` is gone.** Neither change touches the model. `Q` at every window's θ̂ is
reproduced **exactly** — `Δ = 0.000e+00` on all four bundles, not within tolerance — θ is
bit-identical, and the 66-point `base_fc` identification grid is unchanged to the last
printed digit. The four stored estimates were migrated in the same pass and no result
moves.

**Why MINOR and not MAJOR, since a stored bundle no longer loads.** The layer test is
whether a run under the new version reproduces the old numbers, and it does, bitwise. The
MAJOR row that looks close — "a state, a parameter, or an equilibrium object added,
removed or redefined" — does not apply: `anderson_m` is neither a state nor a model
parameter nor an equilibrium object, but a numerical control that no code path read, and
`migrated_from` is metadata about the file. Nothing in the paper changes and the theorist
has nothing to be told. The bundle break is a **compatibility** event, which is what the
comparability field below is for, not a layer escalation — the precedent is v19.6.0, which
changed the bundle shape outright (schema 1 → 2) plus the filename, and is MINOR on
exactly this reasoning.

**Comparability: every estimate is preserved and comparable; every bundle file written
before this version must be migrated to be read.** The four canonical estimates were
carried across and are unchanged. A pre-v39.3.0 bundle raises
`TypeError: in new, expected Float64, got a value of type Int64` — the loud failure, because
`anderson_m::Int` sat immediately before `anderson_reg::Float64`, so the positional read
lands an `Int` in a `Float64` slot. To carry one across: run `migrate_bundles.jl extract`
under the code that wrote it, then `rebuild` under this code. The `candidates_*.jls` banks
are **unaffected** and still load — they carry no `SimParams` — so `INIT_MODE = :clusters`
needs nothing.

### Provenance: `migrated_from`

`rebuild` overwrites `provenance.version` with the running version, so migrating erased
which code produced each estimate — the one thing provenance exists to record, and needed
for the replication package. `RunProvenance` gains `migrated_from :: Vector{String} =
String[]`, appended last (`smm_params.jl:362`): each migration **appends** the version the
bundle arrived with, so a twice-migrated bundle carries both hops in order and an
estimated bundle carries an empty vector. `version` keeps its meaning — the code that last
WROTE the file — and the two are now separately recoverable.

**`BUNDLE_SCHEMA` stays at 2**, and the docstring that said otherwise is corrected
(`smm_params.jl:338`). The bundle's top-level field set is unchanged — `schema, stage,
result, spec, provenance, sim, se, se_source, mcmc, tag` — and a file whose *inner* structs
moved fails to deserialise before `schema` can be read, so bumping the number would help no
reader. The old text claimed a `@kwdef` default keeps old bundles loadable when a field is
added; it does not. `Serialization` fills fields positionally and never calls the
constructor, so an addition raises `EOFError` exactly as a removal raises `TypeError`.

`Vector{String}` was checked against how provenance is consumed before it was chosen.
There are three readers and no table, caption or figure among them: `bundle_summary`
(`bundle.jl:213`, the one line printed wherever a bundle is opened, called at
`MCMC_main.jl:1903`), the write-time banner at `smm_main.jl:1257` which prints a
provenance it just built, and `migrate_bundles.jl verify`. A vector therefore costs no
reader a reshape, and the two that read a stored file were updated so the field prints:
`bundle_summary` appends ` from=38.0.0` (hops joined by `→`) after `v=`, and `verify`
gained a `from=` column. `rebuild` prints `estimate from:` per window.

### The `anderson_m` purge — six sites

The field (`solver/params.jl:310`) and five `SimParams` constructions:
`solver/params.jl:441` in `initialise_model`, `solver/model_main.jl:164`,
`smm/smm_main.jl:149`, `policy/policy_main.jl:202`,
`transition/transition_simulation.jl:132`. `grep -rn anderson_m --include=*.jl code/`
returns nothing. It was dead: `Anderson1` / `anderson1_update!` (`grids.jl:416-447`) is
hardcoded memory-1, so a run at `anderson_m = 3` is bitwise identical to shipped. The
historical notebooks under `code/notebooks/` still contain the name; they are frozen
records of earlier model versions and were left alone.

### Measured, not asserted

Proven end to end in a scratch tree before the live tree was touched, then re-measured
live. Live rebuild:

| window | stored `loss_opt` | reference `Q` at extract | rebuilt `Q` | Δ | `migrated_from` | d |
|---|---:|---:|---:|---:|:--|---:|
| `base_fc` | 1071.079481 | 1071.280270 | 1071.280270 | **0.000e+00** | 38.0.0 | 23 |
| `crisis_fc` | 713.207566 | 713.209187 | 713.209187 | **0.000e+00** | 34.0.0 | 18 |
| `base_covid` | 2834.373205 | 2834.373205 | 2834.373205 | **0.000e+00** | 34.0.0 | 23 |
| `crisis_covid` | 1816.503262 | 1816.503256 | 1816.503256 | **0.000e+00** | 34.0.0 | 18 |

The free sets are **not** uniform — 23, 18, 23, 18 — and the provenance versions differ,
which is exactly the information `migrated_from` preserves and the reason nothing in the
migration path may hardcode `d`. Nothing does: `from_neutral` matches by field NAME, and
both widths rebuilt from the same code path.

- `_load_smm_bundle` on each rebuilt bundle: **`max|Δθ| = 0`** on all four.
- `stage` survives: `base_fc` is still `:mcmc_checkpoint`, not an estimate. `tag` is
  preserved with ` | migrated` appended, so `"CHECKPOINT g=289"` reads
  `"CHECKPOINT g=289 | migrated"` — the generation is still legible and the file still
  fails `bundle_is_final`.
- `sim` payload survives where it existed (absent on `base_fc`, present on the other three).
- `unrecorded` is **empty on all four**: no struct gained a field, so nothing was
  reconstructed from a default. `dropped` names `sim.anderson_m`, and
  `bundle_sim.anderson_m` on the three that carry a top-level `sim`.
- The four unmigrated originals are refused under the new structs with the `TypeError`
  above, and `read_bundle` returns `nothing` for each rather than a half-built bundle.
- `smm_objective` at θ̂ for `base_fc`: **1071.280270**, unchanged.
- The coarse 6-parameter × 11-offset `base_fc` grid (`a_ℓ`, `A`, `μ_S`, `k_S`, `c`, `ξ_U`;
  offsets ±0.02 … ±0.30 applied to the unconstrained coordinate): **66/66 finite, 0
  failures, 0 isolated**, before and after, and all 66 `Q` values agree to every printed
  digit.
- Gates: all nine changed files parse; `check_dead_settings` PASS (177 keyword arguments,
  4 files); `check_forwarding` reports one finding, the pre-existing obsolete
  `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` registry row already carried in the v37.1.0–v37.3.0
  entries; `check_structure`'s 7 `UNBALANCED` files are all pre-existing and none is a file
  this version touched; `check_output_tree` reports the four new `.pre_migration` files
  under rule 6, which is the gate noticing debris that is being kept deliberately.

### The ten historical `base_fc` snapshots — carried in a second pass

The first pass migrated the four canonical estimates and left the ten `base_fc` snapshots
(`_backup_Q*`, `_postmean`, `_rejected_Q*`, `_pre_xiU_restore_Q*`) unreadable. That was a
replication-package regression, not housekeeping debris: `_backup_Q1074.224541` is the only
readable copy of the v35 `base_fc` estimate and `_postmean` the only copy of the chain's
posterior mean. **All ten are now carried and load under v39.3.0.** `candidates_*.jls` were
never affected — they carry no `SimParams`.

The route is the neutral intermediate's own property: it holds no struct identity, so it
crosses code versions. `extract` ran under the pre-v39.3.0 structs in an out-of-tree copy
and `rebuild` under the live ones; the live `code/` was never reverted, not even briefly.

A snapshot is not an estimate, so the acceptance test is **not** a re-solve. Several of
these are deliberately bad points — `Q = 27002.469793` (rejected), `Q = 23552.793052`, a
posterior mean — and re-solving would measure specification drift since they were written,
not the struct carry. The invariant applied instead is *same values in, same values out,
new struct shape*: `loss_opt` and `theta_opt` are carried, never recomputed, and compared
bit-for-bit. **`max|Δθ| = 0` and `loss_opt` bit-identical on all ten**, with `stage` and
`tag` preserved exactly — and, unlike the four, **no ` | migrated` appended to the tag**,
because these files are identified by their tags.

| file | d | stage | `migrated_from` | `loss_opt`, unchanged |
|---|---:|---|:--|---:|
| `_backup_Q1074.224541` | 23 | `smm` | 35.1.0 | 1074.224541 |
| `_backup_Q23552.793052` | 20 | `smm` | 28.0.0 | 23552.793052 |
| `_backup_Q488.548412` | 23 | `mcmc_aborted` | 20.0.0 | 488.548412 |
| `_backup_Q490.550869` | 23 | `mcmc` | 20.0.0 | 490.550869 |
| `_backup_Q497.786594` | 23 | `smm_checkpoint` | 19.5.0 | 497.786594 |
| `_backup_Q970.339482` | 23 | `smm` | 22.0.0 | **968.097038** |
| `_backup_QInf` | 23 | `smm` | 34.0.0 | **1073.220950** |
| `_postmean` | 23 | `mcmc_postmean` | 20.0.0 | 486.500699 |
| `_pre_xiU_restore_Q4960.615185` | 20 | `smm` | 28.0.0 | 4960.615185 |
| `_rejected_Q27002.469793` | 23 | `smm` | 35.1.0 | 27002.469793 |

Each carries **its own** recorded version in `migrated_from`, not `base_fc`'s — six
distinct versions (19.5.0, 20.0.0, 22.0.0, 28.0.0, 34.0.0, 35.1.0) across the ten, which is
the whole point of the field. `unrecorded` is empty on all ten: only `anderson_m` was dropped, and no struct
gained a field any of them lacked, so nothing was reconstructed from a default. The two
`mcmc` stages carried their `se` payload through (`se=chain`). Free-set widths are 23 on
eight and **20** on two, handled by the same name-matched path.

**Two filenames disagree with their contents, and did before this pass:**
`_backup_Q970.339482` stores `loss_opt = 968.097038` and `_backup_QInf` stores
`1073.220950`, not `Inf`. The values are carried as found; the filenames are simply not a
reliable index of what is inside.

`.pre_migration` backups written for all ten and kept, so `output/estimates/` now holds 14
of them.

**Correction to the v36.0.0 entry.** It points a reader at `.migration/` for the readable
record of the v35 `base_fc` point at `Q = 1074.224541`. That has not been true since the
`extract` run of 2026-09-22 overwrote those intermediates (recorded in v39.2.0), and
`.migration/` is now empty because this version's migration consumed it. The v35 point is
readable at `output/estimates/estimate_base_fc_diagonalW_backup_Q1074.224541.jls` —
provenance `migrated_from = 35.1.0`, stage `smm`, 23 free parameters,
`loss_opt = 1074.224541` — which the v37.2.0 acceptance test used and which is now carried
to v39.3.0 shape. That same entry's "the four stored bundles are NOT migrated and must not
be" stands as written: it forbids laundering a *respecification* as a migration, which is a
different operation from this version's struct carry at `Δ = 0`.

### Settings

`anderson_m` retired — graveyard entry in `SETTINGS.md`. It was never env-settable and had
no `SETTINGS_REGISTRY` row, so no `LEGACY_ENV_KEYS` entry is owed. No setting added and no
default moved.

`SETTINGS.md` also corrects a false sentence in the `DAMP_THETA_S` entry, which claimed the
damper "changes the solver's fixed point" and so contradicted the `DAMP_THETA_U` entry
twenty lines above. Under-relaxation leaves the fixed point algebraically invariant —
measured at `2.7e-10` relative on `Q` at θ̂ between `w = 0.9` and `w = 0.5`. What it does do
is **select among** fixed points where the skilled free-entry locus has more than one root:
on a 19×19 (`a_ℓ`, `c`) plane at `base_fc`, `w = 0.9` against `w = 0.5` recovers 13, loses
17 and re-roots 20, with `Q` moving up to `8.3e-02` relative and `θ_S` up to `0.134`, and
three re-rooted points solve under both values with both states satisfying all three
adding-up identities to machine zero. The value stays at `0.9`; this is documentation only.

### Known gaps

- **The ten historical `base_fc` snapshots in `output/estimates/` were not migrated and no
  longer load** — the `_backup_Q*`, `_postmean`, `_rejected_Q*` and `_pre_xiU_restore_Q*`
  files. They are superseded debris already listed for cleanup in the v19.6.0 entry, they
  are not in `MIG_WINDOWS`, and nothing reads them. Recovery, if one is ever wanted: check
  the pre-v39.3.0 code out of git, run `extract` naming that file, then `rebuild` here.
  All 18 tracked `.jls` are committed at `599d219` and the four migrated ones were verified
  byte-identical to their HEAD blobs before the run, so `git checkout` restores any of them.
- `output/smm/warmstart_*.json` still record `sim.anderson_m` in their field dumps. They are
  reports from a past verification session, no code reads them, and they are left as the
  historical record they are.
- The `.pre_migration` backups are kept, at
  `output/estimates/estimate_{base_fc,crisis_fc,base_covid,crisis_covid}_diagonalW.jls.pre_migration`.
  They are the only readable copy of the pre-migration files outside git.
- Not run: an optimiser, a chain, or the transition path. `transition_simulation.jl:132` was
  edited and is covered by parse plus the shared `SimParams` constructor, not by a
  transition run.

### Tried and rejected

- **Migrating the two changes as two passes.** Both touch structs inside the serialised
  bundle, so each pass would rewrite all four bundles and stamp its own hop into
  `migrated_from`, leaving the four estimates carrying a spurious intermediate version.
  One extract → edit → rebuild cycle rewrites them once.
- **Making `anderson_m` a module-level `Ref` instead of deleting it**, on the
  `DAMP_THETA_S` pattern. Rejected: that pattern exists to keep a *live* knob out of the
  serialised struct, and this one is read by nothing. A `Ref` would preserve a switch that
  cannot change a result.

---

## v39.2.0 — 2026-09-22

**MINOR: the global loop's stall-out detector no longer fires on a converged residual.**
One condition added at `solver.jl`. No stored result changes — `Q` is reproduced **bitwise**
on all 24 control points and at θ̂ — and three parameter points that returned `Q = Inf` now
return a finite value. The rule can only let the loop run longer; it cannot cause a
different state to be accepted, which is why the reproduction is exact rather than
approximate.

**The defect.** `conv_streak = 4` requires four **consecutive** passes with `d < tol_global`,
while the stall-out rule fires from pass `global_B + global_K + 1 = 13` whenever the last
`global_K` passes have not beaten the best-so-far. A residual that has already reached
tolerance stops improving — that is what convergence is — so the two rules were in direct
conflict, and the stall-out won because it fires first. Traced at `a_ℓ-0.200 c+0.022`:

```
pass     6        7        8        9        10       11       12       13
dU     8.72e-05 8.11e-05 4.38e-06 1.35e-04 1.42e-04 2.02e-05 1.90e-05 2.69e-05
<tol   YES      YES      YES      no       no       YES      YES      YES
streak 0        1        2        3        0        0        1        2
```

`streak` is the count **entering** each test, before that pass's increment. It reaches 3 of
the 4 required at pass 8 — one pass short — and pass 9's excursion above tolerance resets it
to zero; it rebuilds to 2 entering pass 13, and the stall-out fires there. So the loop
stopped from a state **inside** its own tolerance, having twice come within one pass of the
required streak. `min(dU)` over the run was 4.38e-06, twenty-three times below tolerance.

**What it recovers, and what it does not.** On the 16 residual failures of the 19×19
`a_ℓ × c` plane: 3 recovered, 0 of 24 controls lost, `Q` unchanged on every control to the
last bit. The remaining 13 need more than extra passes — the residual oscillates in and out
of tolerance with a period shorter than the four-pass streak, so extra budget alone never
completes it.

**The scale defect behind that oscillation is measured but NOT fixed here.** `d = max(dU, dM)`
compares two absolute sup-norms against one threshold, and the two objects differ by nine
orders of magnitude: `|U_S| ≈ 2.5e+05` makes `tol_global = 1e-4` a **relative 3.9e-10** on
`U_S` — nine significant digits of pass-to-pass agreement — while `|m_S| ≈ 2.6e-04` makes the
same threshold a **relative 0.4**, no constraint at all. Across eight traced points and
100+ passes, **`dM` bound on zero of them**: the global test is `dU < 1e-4` alone. Under a
relative norm the binding component flips — `dU_rel` spans 1.2e-11 to 1.3e-08 while `dM_rel`
reaches 2.8e-05 — so the absolute form tests the wrong one of the two.

Four arms were measured on the same 16 failures and 24 controls:

| arm | recovered of 16 | controls with `Q` moved | worst relative move |
|---|---|---|---|
| shipped | 0 | — | — |
| **stall guard (this version)** | **3** | **0 of 24** | **0** |
| guard + relative 1e-6 | 6 | 3 of 24 | 2.4e-07 |
| guard + relative 1e-5 | 6 | 12 of 24 | 3.9e-07 |
| guard + relative 1e-4 | 7 | 19 of 24 | 4.6e-07 |

`Q` at θ̂ is 1071.280270 under every arm. The arms are **not nested**: the relative norm at
1e-6 and 1e-5 loses `a_ℓ+0.233 c+0.133`, which the guard alone recovers, because `m_S`
becomes binding there.

**The recovery classes partition exactly, which is the evidence the diagnosis is right.**
Attributing all 16 failures: `conv_U` false at 0, `conv_global` false at 16, `conv_S` false at
9. The criterion fix recovers **7 of the 7** points where both blocks converged and **0 of the
9** where the skilled block failed. So the global criterion owns precisely the pure-global
failures, and the remaining 9 are a skilled-block defect with its own cause.

**`migrate_bundles.jl rebuild` was broken and is now repaired.** It had never run, and there
were two independent fatals on the one path — which is itself the evidence. `write_bundle`
was not in the include chain (it lives at `bundle.jl:107`), and the install step did
`mv(p * ".new", p)` against a filename the tool never writes, since `rebuild` stages under
`.migration/`. Both fixed, and `bundle.jl` joins the include chain.

A first reading of this reported that `from_neutral` and `to_PU_numeraire` were undefined.
**That was wrong.** Both are defined inside the `elseif MODE == "rebuild"` branch of the tool
itself, at `HEAD:169` and `HEAD:196`; the grep behind the claim excluded `migrate_bundles.jl`
from its own search path, so it looked everywhere except the file that holds them. `rebuild`
reaches the `write_bundle` call only after both have executed.

**`to_PU_numeraire` is deleted rather than repaired.** `P_U` is pinned at 1 wherever it is set
(`FIX_PARAMS` `smm_main.jl:270`, `DEFAULT_PARAMS` `smm_main.jl:326`, `ParamSpec` init
`smm_params.jl:638`) and `PU_free = false` in all four bundles, so its guard returned early
every time. The decisive reason is not that it was inert: a struct migration's invariant is
*same θ, same code, same `Q`*, and that function deliberately **moves θ**. It is a
reparametrisation, and running one under cover of a struct change would relabel an estimate
while the gate printed OK.

**The acceptance test is fixed in the same pass, and it would have rejected half the
bundles.** It gated on `abs(Q1 - Q0) < 1e-6 * max(1, abs(Q0))` with `Q0` the **stored**
`loss_opt`, which is the right reference only when no respecification has happened since the
bundle was written. Measured against the four on disk:

| window | stored `loss_opt` | `Q` at that θ now | gap | old tolerance | old gate |
|---|---:|---:|---:|---:|:--|
| `base_fc` | 1071.079481 | 1071.280270 | 2.01e-01 | 1.07e-03 | **reject** |
| `crisis_fc` | 713.207566 | 713.209187 | 1.62e-03 | 7.13e-04 | **reject** |
| `base_covid` | 2834.373205 | 2834.373205 | 0 | 2.83e-03 | pass |
| `crisis_covid` | 1816.503262 | 1816.503256 | 6.00e-06 | 1.82e-03 | pass |

`extract` now evaluates `Q` at the bundle's own θ under pre-edit code and records it beside
the intermediate; `rebuild` gates on that, with a documented `loss_opt` fallback so
intermediates written by the older tool remain usable. The invariant is *same θ, same code,
same `Q`, before and after the edit*.

**Proven on the `anderson_m` purge, both directions, in a scratch tree.** Removal: all four
windows rebuild with **Δ = 0 exactly**, `_load_smm_bundle` returns `max|Δθ| = 0`, and the
unmigrated originals are correctly refused under the new struct with a `TypeError` — the loud
failure, because `anderson_m::Int` sits before `anderson_reg::Float64`. Addition: proven
separately with a field inserted **mid-struct**, all 22 carried fields exact, the new one at
its declared default and named in `unrecorded`.

Two things the purge inventory had missed. `anderson_m` has **six** sites, not four —
`params.jl:441` inside `initialise_model` was not on the list. And the four bundles are not
uniform: `base_fc` is `stage = :mcmc_checkpoint`, `tag = "CHECKPOINT g=289"`, while the other
three carry a top-level `SimParams`. `rebuild` hardcoded `stage = :smm` and passed no
payload, so migrating as written would have flipped `bundle_is_final` (`bundle.jl:200`) and
let a table caption a mid-chain checkpoint at generation 289 as a finished estimate, and
would have dropped a `sim` that `transition/plots_and_tables.jl` reads unguarded. Both now
travel through.

**`anderson_m` is confirmed dead and its removal is staged, not applied.** `Anderson1` /
`anderson1_update!` (`grids.jl:393-424`) is hardcoded memory-1, so a run at `anderson_m = 3`
is bitwise identical to shipped — max absolute difference 0.0 across every field and all 200
iterations at both test points. It is `SimParams` field 10 of 23 and `SMMSpec` carries `sim`,
so removal rewrites every stored bundle; with the migration path now working and proven at
Δ = 0 that is cheap, and the patch is ready as a separate change.

It is held for one reason, which is not the migration: `rebuild` overwrites
`provenance.version`, stamping `39.2.0` over the `34.0.0` and `38.0.0` the four bundles
carry. That erases which code produced each estimate, which is exactly what provenance
exists to record, and it matters more for the replication package than one dead `Int`
matters. The fix is a `migrated_from` field on `RunProvenance` — a struct change of its own,
and one that should go in before any bundle is migrated rather than after.

**Correction to the v36.0.0 entry.** That entry states the four version-neutral intermediates
in `.migration/` carry `base_fc` 1074.224541. **They no longer do** — an `extract` run on
2026-09-22 overwrote them with the current bundles, and `base_fc` now reads 1071.079481, the
author's own improved point. The other three are unchanged in content: `crisis_fc`
713.207566, `base_covid` 2834.373205, `crisis_covid` 1816.503262, identical to the values
that entry records. The v35 `base_fc` point is recoverable from git at `645c3e0`.

**Global-loop acceleration measured, and kept.** `use_anderson` is one flag read in five
places, so a global-scoped switch was added in a scratch tree to isolate the global loop. Over
17 points: **153 global passes with acceleration against 160 without**, a 4.4% saving, feasible
17/17 either way, winning on 7 points, losing on 1 (`a_ℓ-0.100 c-0.061`, 12 against 9) and
tying on 9. Kept on that evidence.

**A claim from the v39.0.0 work is withdrawn.** That the global test measures the accelerator's
step rather than the fixed-point residual is true, and immaterial: over 77 passes with both
quantities read in the Anderson vector's own units, the ratio is min 0.971, median 1.000, max
1.04, and at `tol_global = 1e-4` the two criteria **agree on all 77** — 35 converged each, zero
disagreements. At memory-1 the correction is about 3% of the step. The first divergence figure
reported for this compared an unscaled step against a scaled residual and so measured `s_U`,
the Anderson scaling constant, not a divergence. No fix was made: it changes no decision, and
the quantity is being replaced.

**Why the relative norm is not in this version.** It would need its tolerance retuned in the
same change, and reading the stored `sim.tol_global` relatively makes a bundle's own recorded
tolerance mean something 2.6e+06 times looser than when it was written. The clean form is a
separate module-level tolerance on the `DAMP_THETA_S` pattern with `sim.tol_global` declared
`:inert`, and it moves `Q` in the seventh significant digit on most points — below the
solver's own accuracy, but it is a change to reported numbers and belongs in its own version,
taken deliberately before a re-estimation rather than folded into a bug fix.

---

## v39.1.0 — 2026-09-22

**MINOR: the unskilled free-entry map is under-relaxed, as the skilled one already was.**
`const DAMP_THETA_U = Ref(0.9)`. No stored result changes — `Q` at every point that already
solved is reproduced to between 1.1e-10 and 4.1e-07 relative — but a contiguous band of
parameter points that previously returned `Q = Inf` now returns a finite value, so a
re-estimation can reach points the optimiser could not price before.

**The defect.** `unskilled.jl` iterated `θ_U = update_theta_unskilled(model)` **raw**, while
`skilled.jl:757-761` has under-relaxed its own free entry since the knob was introduced —
and that code's own comment states the reason: *"the damped map is what both the Anderson and
the plain branch iterate on, so acceleration sees a contraction the raw map is not."* The
argument was never carried across to the unskilled block.

**Measured, on the `a_ℓ` line at `base_fc` under `OJS_PARTICIPATION = :fraction`**, which is
where v39.0.0 left a contiguous band of failures:

| w | points solving, of 9 |
|---|---|
| 1.00 (shipped) | 4 |
| **0.90** | **9** |
| 0.50 | 8 |

**θ_U ran an exact period-8 orbit**, and the mechanism is visible in one column of the trace:
at the floor `θ_U = 1e-14` the job-finding rate is 4.4e-08, so essentially nobody is hired, an
unfilled vacancy is worth everything, and free entry answers with the ceiling. The descent
from the ceiling then takes seven iterations — 82.1, 52.0, 2.3e-04, 1.64, 0.541, 0.217,
0.0822 — and the eighth lands back on the floor. At `a_ℓ-0.35` the shipped solver terminates
with `θ_U` **exactly 100.00000**, the clamp ceiling, against a sane tightness of 0.379 at the
estimate. The outer-U residual is of order 1e+02, not 1e-03: this was never a grid-scale
limit cycle like the skilled one, and no amount of iteration budget would have closed it —
the block stalled out and handed back on all 13 global passes, 41 iterations each.

**The recovered band is smooth, which is the evidence it was never a real feature of the
objective.** `Q` along `a_ℓ` from −0.40 to 0: 139098.7, 103668.1, 75274.4, 47835.7, 30453.8,
17071.0, 11014.6, 7815.0, 1071.3 — monotone, with `θ_U` monotone beside it (0.180, 0.201,
0.225, 0.263, 0.298, 0.337, 0.362, 0.379, 0.458). Five of those nine were `Inf` before.

**Why this is MINOR and the participation change was MAJOR.** Under-relaxation leaves the
fixed point of the map exactly invariant — `w·g(x) + (1−w)·x = x` iff `g(x) = x` — so it
cannot move an equilibrium. The measurement confirms it: across w = 1.0, 0.9 and 0.5, `Q` at
the four points all three solve agrees to 1.1e-10 at the widest offset and 4.2e-10 at θ̂.
This also corrects a loose sentence in `SETTINGS.md`, which said of `DAMP_THETA_S` that "it
changes the solver's fixed point" — it does not; it changes which fixed point a path reaches
when there is more than one, and how fast.

**Second change, same two lines.** The Anderson branch installed `max(x_new[1]·s_θ, 1e-14)`,
with no upper bound, while `update_theta_unskilled` clamps its own answer to `[1e-14, 100]`.
Anderson extrapolates, so the installed `θ_U` left the range its own solve is restricted to —
measured at 107.6 and 182.3. The installed value now carries the same bracket.

**Not changed, deliberately.** `s_θ = max(|θ_raw|, |θ_old|, 1.0)` is still frozen at the
first outer iteration (`unskilled.jl:406-409`). At a point where iteration 1 is anomalous —
θ_raw = 82.1 in the failing band — the scale is ~200× a sane tightness, which compresses the
θ component of the Anderson vector and makes the acceleration nearly blind to the one
coordinate that was diverging. With the map damped this no longer bites, and re-scaling
mid-run would change the accelerator's history in a way that needs its own measurement.
Recorded here so it is not rediscovered as a new defect.

**Gate.** Both changed files parse. `DAMP_THETA_U` is never assigned at runtime, as with
`DAMP_THETA_S`. `SETTINGS.md` gains entries for `DAMP_THETA_U` and for v39.0.0's
`OJS_PARTICIPATION`, which had been documented in `params.jl` and the changelog but not
there.

---

## v39.0.0 — 2026-09-22

**MAJOR: on-the-job-search participation is a covered fraction of the ability cell, not a
node indicator.** An equilibrium object is redefined — the searching set in ability is now
represented by a per-cell weight rather than a per-node 0/1 — so the allocation changes and
no estimate under v38 or earlier is comparable.

**Comparability.** None for `Q`, and the estimates must be re-run. At the stored `base_fc`
point `Q` moves 1071.219979 → **1071.280269** (+0.060291, +5.63e-05 relative), and every
mover is a skilled-market moment: `ur_S` +0.281, `usurv27_S` −0.270, `usurv53_S` −0.193,
`usurv5_S` +0.140, `jfr_S` +0.133, with Σ|ΔQ| = 1.561 against a net of 0.060 — the moments
largely offset. **Bundles still deserialise**: the new field is on `SkilledCache`, which
`SMMSpec` does not hold (it carries `free, fixed, moments, sim, run, W, q_scale`), verified by
loading `estimate_base_fc_diagonalW.jls` under the new structs. `OJS_PARTICIPATION[] =
:indicator` reproduces every v38.3.0 number exactly and is kept permanently for that purpose.

**Why.** The participation score is `M(aS) = max_{p ≥ p*_S} (S¹ − S⁰)(aS, p)`: on-the-job
search pays for `aS` at some acceptable quality iff `M(aS) > 0`. Under the node rule a type
sitting where `M` crosses zero switches its whole OJS margin between consecutive outer
iterations, `p^oj` collapses onto `p*_S`, and the outer map is discontinuous there whatever
crossing rule is used inside the searching region. On a finite ability grid a "type" is a
CELL of positive width carrying Gauss-weighted mass, so the correct weight is the fraction of
the cell on which `M > 0` and the node indicator is the approximation to it — the v22.0.0
covered-fraction pattern the training frontier, the drain and the tail integrals already
carry, applied one level up.

**Measured, at `base_fc`, shipped grid (Nx = 160, Np_U = 160, Np_S = 180):**

- From the identical frozen global-pass-2 state the indicator runs a closed **period-4** orbit
  for 200 outer iterations (lag-4 recurrence 3.331e-15) while the fraction **converges in 13**
  to `d` = 3.346e-07 against `tol_outer_S` = 1e-6.
- The 24-point audit (6 parameters × 4 offsets in the unconstrained vector) goes **19/24 →
  24/24**. Nothing is lost.
- At the fixed point exactly **one ability cell of 160 is fractional** — type 82,
  `aS` = 0.511246, weight 0.086222 — with 78 cells at 1 and 81 at 0. At the cycling state
  that same cell's weight is 0.458271.
- **One type switches on, zero switch off.** Mass-weighted OJS participation moves
  0.0290818 → 0.0291256, **+0.150% relative**.
- Soundness at all 25 points: `U_S⁰` monotone in ability (min increment 59.47), adding-up
  `|mU + mS − pop|` **exactly 0.0**, `min(p^oj − p*_S)` = +4.44e-05 > 0, `sep_rate_S` ≥ 0.0051,
  `ee_rate_S` ≥ 0 and zero only where the margin is shut under both encodings.

**Changed, by subsystem.**

- `solver/grids.jl` — `_pos_share`, the one-dimensional covered-fraction primitive behind
  `_soft_oj_weight`, written separately so it applies on the ABILITY grid too; `_ojs_share`,
  the product of the participation weight and the p-coverage, which is now what every
  consumer of the OJS margin calls.
- `solver/params.jl` — `OJS_PARTICIPATION` (`:fraction` | `:indicator`);
  `SkilledCache.s_part` and its default of ones.
- `solver/skilled.jl` — `ojs_participation!`, the weight as a covered fraction of the node's
  midpoint interval, each half-cell scored against its own neighbour. The outer loop records
  the score `M` and the node attaining it; **where `M ≤ 0` the cutoff is that peak node**
  rather than the scan's fall-through, because as `M` falls to zero the searching interval
  collapses onto the peak — the fall-through instead shut the margin outright and jumped
  `p^oj` by four to six cells. The weight joins the convergence test as `Δpart`, identically
  zero under `:indicator`, and the outer-S trace names it.
- `solver/equilibrium.jl`, `policy/policy_solver.jl`, `plots_and_tables/model.jl` — the wage
  split, the EE flow and the wage surface read `_ojs_share`; `s_part` is exported on the
  equilibrium objects.
- `transition/` — `TransitionPath.s_part` (Nx, Nt), written at the terminal date from the
  stationary weight, computed in the backward induction from the same score, and installed on
  the forward pass. **The backward induction computes the weight in a second pass after the
  per-type loop**, because a cell's weight depends on its NEIGHBOURS' scores and the per-type
  loop is `@threads`.
- `tools/ojs_continuity_probe.jl` — reports `n_part_straddle` beside `n_straddle`. They are
  counted apart deliberately: a p-straddle is a cell the CUTOFF cuts, a participation straddle
  is an ability cell the searching FRONTIER cuts, and only the second can switch a whole type's
  margin, so only the second admits a limit cycle.

**Determinism, shipped unconditionally and independent of the above.** `compute_Jbar_skilled`
accumulated into `threadid()`-indexed slots while `@threads` schedules dynamically, so `J̄_S`
was grouped differently from run to run. That is invisible at a well-conditioned point and
decisive at a marginal one — `dU_S/dθ_S ≈ 2e+04` and the global residual is a sup-norm on
`U_S` — which made the **feasible/infeasible verdict itself depend on the schedule**: two
identical solves at 8 threads diverged from pass 2 with a maximum relative residual gap of
2.8e-05, while 1 thread was bit-identical. Now accumulated per `aS` node and reduced in index
order. Separately, the two loops that index scratch buffers by `threadid()`
(`skilled.jl` inner loop, `unskilled.jl` surplus loop) are annotated `:static`, because under
dynamic scheduling a migrated task can **alias another task's buffer** — a race, not a
reordering. Cost: three repeats of `smm_objective` at `θ̂` now return
1071.2802693076353 bit-identically at 8 threads; measured overhead is inside the run-to-run
spread.

**Gate.** Every changed file parses (`Meta.parseall`, no `:error`/`:incomplete`). The solver
path loads and runs against real data in a scratch depot. `check_version_sync` in sync on one
declaration with three readers. `estimate_base_fc_diagonalW.jls` loads unchanged and
`:indicator` reproduces its `Q` to the printed digit. `:fraction` reproduces the test-5 value
to 3.1e-07, i.e. the precision that value was reported at — two independent implementations,
one in a scratch tree and one here, agreeing.

**Settings.** `OJS_PARTICIPATION` added, default `:fraction`. Nothing retired. The bundle's
stored `loss_opt` = 1071.079481 predates `POJ_CROSSING = :last` and is not the `Q` at its own
`theta_opt` under either encoding — that mismatch is from v38.3.0 and is resolved by
re-estimating, not by this entry.

**Known gaps.**

- **No window has been re-estimated under v39.** Every stored estimate is an estimate of the
  previous allocation. The `base_fc` and `base_covid` numbers above are evaluations at a v38
  point, not optima.
- **Participation is now the slowest outer coordinate**, tail contraction ratio 0.9434 per
  iteration. It is not binding at `tol_outer_S` = 1e-6 — the block's worst iteration count
  over the audit actually FELL, 42 → 38 — but it would dominate at a tighter tolerance.
- **The global convergence test is still absolute and still has the defect.**
  `tol_global` = 1e-4 on a `U_S` of order 1e+05 is a relative 4e-10, against a residual floor
  of `tol_outer_S · dU_S/dθ_S` ≈ 2e-02 — two decades ABOVE the tolerance, so a block converged
  to eight digits cannot pass it. LMR's relative form (`solver.f90:264-270`) is the fix and is
  documented in place at the test, but it REINTERPRETS the stored `sim.tol_global`: read
  relatively, 1e-4 is 2.6e+06 times looser than it is now. Deferred to its own version where
  the tolerance is retuned against the measured floor rather than inherited. Three of the five
  v38.3.0 audit failures were this defect and are recovered here for a different reason — the
  discontinuity the global loop inherited through the training flow is gone.
- **Two audit points need `Np_S` ≥ 360 under the previous encoding.** At the shipped grid the
  skilled free-entry residual crossed zero by a JUMP of 8.7e-04 and 1.1e-03 rather than by a
  zero, because `J̄_S` reads the firm-value tail at a hard cell index while weighting the
  margin smoothly; at `Np_S` = 360 the jump collapses to 2e-16 and 8e-16 and both solve.
  Interpolating the tail at the cutoff — the same pattern a third time — is what would make
  the refinement unnecessary, and it is NOT in this version: which of the two cutoffs carries
  the jump has not been measured, and the fix should not be written against a reading of two
  lines. See "tried and rejected" for why this is not the participation margin.
- `anderson_m` remains dead: acceleration is hardcoded memory-1 (`Anderson1`,
  `grids.jl:393-424`), so a run at `anderson_m = 3` is bitwise identical to shipped across 200
  iterations. Removing it breaks bundle deserialisation, so it is a purge with a migration
  attached.

**Tried and rejected — nine routes, each with the measurement that killed it.** Recorded so
none is re-proposed.

1. **The cycle is a U↔S feedback, a block handing over an unconverged iterate.** Refuted.
   Freezing all three coupling channels (`f_U`, `E_U(·,1)`, the training flow `m_S`) does not
   stop the cycling, and the mechanism is never exercised: at `μ_S+0.03` BOTH blocks report
   convergence at every one of 60 global passes in 4 outer iterations with residuals ≤ 2.6e-08
   and the global loop still never converges. Where a cycle survives, one channel carries it —
   the training flow — with a period-doubling ladder 4 → 9 → 24 → converged, and the basin is
   irrelevant (90/100 cells of a frozen-state × start-state grid converge; the 10 failures are
   the frozen state at every start state).
2. **Under-relaxing `p^oj`.** Refuted at ω = 0.5, 0.25 and 0.05 — all worse. The increment
   autocorrelation is −0.148, nowhere near the −1 an over-relaxation cycle requires.
3. **A quadratic in-cell crossing** (`POJ_CROSSING_C1`). Measured no-op, |ΔQ| ≤ 4.2e-09 over
   25 points. `find_poj_from_diff_grid` already interpolates inside the bracketing cell.
4. **Consistent `smooth_pos` scoring in the scan** (`POJ_SCORE = :smooth`). Measured no-op:
   `smooth_pos` is strictly increasing, so the crossing stays in the same cell.
5. **An a-priori conditioning predictor** `κ = |diff| / (input residual × |S|)`. Refuted. It
   separates cycling from converging points ex post but has no discriminating power in advance
   and the ordering is inverted at early passes, because its denominator is dominated by the
   per-iteration move of `θ_S` — an OUTPUT of the same loop whose cutoff is flipping.
6. **Tightening input precision.** Refuted. Sweeping `tol_inner` 1e-7 → 1e-13 and
   `tol_outer_U` 1e-6 → 1e-12 leaves every recorded input residual unchanged to 9 significant
   digits, and the period, the iteration count and the exit reason identical.
7. **"An exact orbit means the inputs have converged; a decaying envelope means they have
   not."** Refuted. There is no independent input state: `f` is a deterministic function of
   `θ_S` (|Δlog f|/|Δlog θ_S| = 0.500) and the inner loop converges at all 200 outer
   iterations. What separated two recorded figures was the PERIOD against the recording
   window — a period-22 orbit recorded over 1.9 periods falls two decades within a period and
   resets, which reads as a decaying trend. It never flattens. The installer sets the
   attractor (period 4 at amplitude 2.58e-03 accelerated, period 2 at 1.80e-03 damped Picard,
   period 2 at 1.35e-01 undamped — a 52× spread) but no arm decays.
8. **A cycle detector.** Rejected on grounds, not measurement. A cycle in the iterates is not
   the defect, and a detector on orbit geometry has no economic content; every fix here is a
   statement about the model or the discretisation.
9. **A degenerate corner with an inactive skilled market at the two hardest points.** Refuted.
   The free-entry residual `F = [q_S J̄_S − k_S π̄_S]/(k_S π̄_S)` changes sign exactly twice over
   eight decades of `θ_S`, and only the second is a root: the first is the market-activation
   discontinuity where `F` leaps from −1 to +1.1e+03 between adjacent points. Below
   `θ_S` = 1.1095e-04 the drain fraction is identically 1, `J̄_S` = 0, and `F` = −1 exactly —
   free entry fails at the corner by the MAXIMUM amount, not by zero. The root exists at all
   four points tested.

---

## v38.3.0 — 2026-09-22

**MINOR: the SMM header stops reporting parameter counts, which were duplicated and wrong.**
House rule applied: a print says a thing once. The spec tables a few lines below list every
free and fixed parameter with its bounds and value, so the counts above them were the same
information in a worse form — and, because they were taken at the wrong point in the
pipeline, a *different* number.

Removed:

- `smm_main.jl` — `Fixed params: N | Free params: N`. `free_params` there is the CANDIDATE
  list, before `build_smm_spec` drops whatever `FIX_PARAMS` and the external calibration pin,
  so it read **29** against the spec table's **23**.
- `smm_main.jl` — `FIX_PARAMS pinned (N): ...`. The same parameters appear in the spec's Fixed
  table with their values.
- `smm_params.jl` — `SMMSpec: fixed override dropped free params: ...`. Same information
  again, and it printed the ASCII `name` field, which duplicates across markets — the line
  read `bT, PU, η, η, σ_w, σ_w`, two of each, which looks like a defect and is not one.
- `smm_main.jl` — `Baseline mode: all structural + regime params are FREE (N total)` no longer
  carries a count. It was the same pre-pin 29, and the sentence asserted that all of them are
  free, which is false whenever `FIX_PARAMS` is non-empty. It now states what the list is:
  candidates for the free set.

**Still there, and not changed here:** the warm-start line reads
`matched 23 / 29 parameters from prior run` with
`Unmatched (kept at spec default): unsk:bT, unsk:PU, unsk:η, skl:η, unsk:σ_w, skl:σ_w`. The
ratio is genuinely informative — it says how much of the free set the prior bundle supplied —
but the denominator is the same pre-pin 29, and **"kept at spec default" is wrong for those
six**: they are pinned by `FIX_PARAMS` or set by the calibration, not left at a spec default.
Left alone because the line carries real information and rewording it is a judgement about
what the author wants it to say.

**A worked acceptance example, at the restored `base_fc` point, 23 free parameters.** One
chain position and one DE-MC proposal built as `demc.jl` builds it, at two population
spreads. `logπ = −Q/2 + Σ log|dθ_k/dt_k|`, and acceptance is `min(1, exp(Δlogπ))`.

| | largest step in t | Q | −Q/2 | log&#124;jac&#124; | logπ | Δlogπ | α |
|---|---|---|---|---|---|---|---|
| current position | — | 1071.079481 | −535.539741 | −42.872103 | −578.411843 | — | — |
| proposal, spread 2e-3 | 1.92e-03 | 1071.943589 | −535.971794 | −42.871717 | −578.843511 | −0.431668 | **0.649** |
| proposal, spread 2e-2 | 1.90e-02 | 1356.853909 | −678.426954 | −42.866498 | −721.293452 | −142.881609 | **0.000** |

Both moved 18 of 23 coordinates at `γ = 0.2805`. **The Jacobian term is negligible** — 0.09%
and 0.004% of `|Δlogπ|` — so in practice the only thing that moves the log-posterior is the
weighted moment misfit. A 9.9x larger step raises `ΔQ` by **331x** (0.864 to 285.8), an
implied `ΔQ ∝ step^2.53` on these two points, and `α = exp(−ΔQ/2)` turns that into 0.649
against zero.

`Q` is additive over moments, so `ΔQ` attributes exactly. At the well-scaled step the five
largest contributions are `ur_S` −3.87, `usurv27_S` +2.89, `usurv53_S` +2.05, `p50_wage_S`
+1.74, `usurv5_S` −1.46 — **each several times the +0.864 total**, so the moments largely
cancel. At the oversized step `theta_U` alone contributes +183.55 of +285.77.

**What this implies for the aborted run.** The restored point is well behaved at the right
step: `α = 0.649` at a spread of 2e-3 is if anything slightly too high. Taking the
Roberts-Gelman-Gilks 0.234 optimum, which needs `ΔQ = 2.905`, the two-point exponent puts the
target spread at **3.2e-03**. The aborted run was at 2.06e-02 by generation 250 — **6.4x too
wide** — implying `MCMC_GAMMA_SCALE ≈ 0.16`.

**Known gaps.**

- The exponent is fitted on two points and the target spread inherits that. It is a starting
  value for a short run, not a calibration.
- Still no run with `MCMC_GAMMA_SCALE < 1.0`.
- The two proposals share a seed and a mask, so they differ only in spread; they are one
  comparison, not a distribution over proposals.

---

## v38.2.0 — 2026-09-22

**MINOR: the periodic block prints only what a decision is taken on, and `sprd` names the
widest coordinate.** Prompted by a real run: the 4,000-generation `base_fc` attempt was
aborted at generation 750 and the console could not answer the one question being asked of
it.

**Dropped from the periodic block:** the running mean and best-point parameter VECTORS. At
d = 23 those were two unlabelled 23-number rows per block, in two different spaces, that no
decision is taken on — and the estimator writes the same quantities to
`mcmc_results_<window><suffix>.csv` with labels, quantiles and standard errors. The
`mean (θ)` / `mean (t)` pair went with them; the distinction it recorded (their print
averages raw stored values, the paper's estimator averages constrained draws) is settled in
the estimator itself and documented at `MCMC_main.jl`. **Kept:** the five-way accept/reject
breakdown, which cannot be recovered afterwards and is the reason to watch a live run.

**The header now reads `gen N, last K generations`** rather than LMR's literal `LOOP N`. The
label was copied from `mpi_mcmc_mod.f90` for fidelity, but every other line in this log
counts generations, and two names for one quantity is a defect whatever its provenance.

**`sprd` now reports the widest coordinate by name**, e.g.
`sprd=3.33e-02 (widest b_S 9.62e-02)`. The average alone was actively misleading, and the
run that prompted this shows why: at generation 500 the per-coordinate population sd ranged
over a factor of **30**, from `A` at 3.21e-03 to `b_S` at 9.62e-02. `run_demc` takes a new
`labels` keyword for this, defaulting to `nothing` and falling back to the coordinate index.
The v38.1.0 entry named this limitation as a known gap; it took one run to bite.

**Fixed: the outlier count in the block was two different quantities under one label.**
`n_replaced` counts chains; `count(replaced[:, lo:g])` counts chain-GENERATION cells, and a
replacement marks the replaced chain's whole trailing window — so the window figure exceeded
the run figure (5 against 1 in a 14-chain test). The window now counts distinct chains.

**What the aborted run established, from its own snapshot.** The population was **not frozen**
— `moved` read 120/120 at generations 250 and 500 and 112/120 at 750. It was **diffusing**:

| gen | acc | moved | sprd | dlp | esjd | max logπ |
|---|---|---|---|---|---|---|
| 250 | 0.181 | 120/120 | 2.06e-02 | −4.37 | 1.347 | −537.66 |
| 500 | 0.026 | 120/120 | 3.33e-02 | −22.53 | 0.123 | −535.36 |
| 750 | 0.016 | 112/120 | 4.53e-02 | −41.80 | 0.077 | −534.83 |

`sprd` rising monotonically with no sign of levelling, the median proposal's `Δlogπ` at −41.8
(acceptance probability e^−41.8, so the realised 0.016 is carried entirely by a tail), `esjd`
down 17-fold, and `max logπ` still IMPROVING — the chain was still descending, not sampling.

**The per-coordinate spread identifies which directions are diffusing, and they are the
directions the identification work already flagged.** Growth from generation 1 to 500, sd in
t-space: `b_S` 1960x, `δ_S` 1650x, `ξ_U` 1130x, `σ_S` 1003x, `α_U` 898x, `b_Γ` 886x, `ρ_x`
885x, `β_U` 817x — against `A` 61x and `c` 71x at the other end. The population expands where
the criterion is flat and stays tight where it is not.

**The consequence is a limit on the chain route itself, and it belongs on the record.** The
widest coordinate sat at 0.48% of the ±20 prior box after 500 generations and was still
growing. `logposterior` already documents why there is nothing to stop it — Q agrees to ten
significant figures between t₀+40 and t₀+80, and the half-line `t_bS ≤ t₀−5` is within 0.004
of the optimum — so in those directions the quasi-posterior is the PRIOR BOX, and a posterior
sd computed there measures `MCMC_T_BOX`, not the data. That is the mirror image of the
`pinv` failure the chain was adopted to escape: fake certainty there, fake uncertainty here.
**The chain reports non-identification honestly as a wide interval, but the width is a
modelling choice, not an estimate.**

**Known gaps.**

- Still no run with `MCMC_GAMMA_SCALE < 1.0`. The aborted run's numbers imply a step of
  0.248·√2·4.53e-02 ≈ 1.6e-02 against a usable 1e-03, but the equilibrium spread is an
  outcome of the recursion rather than a fixed multiple of the step, so the required value
  cannot be read off directly.
- Lowering the scale addresses MIXING, not identification. A well-mixed chain would give
  genuine posterior sds for the coordinates the criterion actually constrains and box-width
  intervals for the rest; nothing here makes the second group reportable.
- The abort at 750 means no window has a v38 chain estimate.

---

## v38.1.0 — 2026-09-21

**MINOR: `moved` and `sprd` on the live `[demc]` progress line.** Two diagnostics, no
behaviour change, no new setting, nothing gated on either. They exist so a run that is going
nowhere can be recognised from the console rather than from the bundle afterwards.

- **`moved=n/N`** — how many chains accepted at least one move in the print window. `acc` is
  a rate over proposals and cannot separate "every chain crawling" from "a few alive and the
  rest frozen", and those call for opposite responses: the second is what the outlier rule
  exists to repair. On the 2026-08-31 run, 47 of 95 chains made no move at all in the
  retained window while `acc` read 0.008 — only this number shows that.
- **`sprd`** — mean per-coordinate population sd, the quantity the DE step is built from.
  Rising means the population is still finding its scale; flat means it has settled, and
  whether it settled somewhere usable is the whole question. `spread_growth` already existed
  but is computed at `MCMC_main.jl:1331`, i.e. only after the run, so the diagnostic was
  invisible while it mattered.

**The calibration that makes `sprd` readable.** The proposal step is
`γ·(X_r1 − X_r2)`, and with `γ = 2.38/sqrt(2·δ·n_masked) ≈ 0.29` at `δ = 2` and ~17 of 23
coordinates masked in, the per-coordinate step sd is about **0.41·sprd**. Against the
v37.3.0 perturbation scan — a step of 1e-3 accepted 45% of the time, 1e-2 accepted 4e-06 of
the time — that puts the usable band at **`sprd` ≈ 2-3e-3** and the dead zone at
**`sprd` ≳ 1e-2**.

A 14-chain, 24-generation smoke run at the restored `base_fc` point confirms the whole
picture is coherent, and shows the inflation phase doing exactly what it should:

```
gen  6/24  acc=0.726  moved=14/14  sprd=2.11e-04  dlp=-0.01  esjd=10.683
gen 12/24  acc=0.786  moved=14/14  sprd=6.41e-04  dlp=-0.08  esjd=12.486
gen 18/24  acc=0.631  moved=14/14  sprd=8.76e-04  dlp=-0.16  esjd=3.897
gen 24/24  acc=0.595  moved=14/14  sprd=1.25e-03  dlp=-0.36  esjd=4.453
```

`sprd` climbing, `acc` falling from 1, `dlp` growing more negative, every chain moving. At
`sprd = 1.25e-03` the implied step is 5.1e-04 and the observed `acc = 0.595`, which is where
the scan puts it — so the scan, the calibration and the sampler agree.

**Known gaps.**

- 24 generations is the inflation phase and says nothing about where the spread settles. It
  establishes that the diagnostics are correct and the run starts healthy, not that it stays
  healthy.
- `sprd` averages over coordinates, so it cannot show a population that is wide in the flat
  directions and collapsed in the sharp ones. The per-coordinate version of that is
  `spread_growth` in the results CSV, still end-of-run only.

---

## v38.0.0 — 2026-09-21

**MAJOR: the v36 specification is reverted. Two distinct per-market flow values and both
bargaining weights free, exactly as before v36.** `common.b` and the `b(x) = b*x*exp(A)`
loading are gone; `up.bU`, `up.bT` and `sp.bS` are back as flat scalars added to the value
functions, `beta_U` is free again, and the free set returns to 23.

**This is a file-level revert, not a rewrite.** The solver and SMM spec files were restored
verbatim from commit `645c3e0` ("all 23 params free, huge standard errors"), which is the
pre-v36 state: `code/solver/{params,unskilled,skilled,equilibrium,model_main}.jl`,
`code/smm/{smm_params,smm_main,settings}.jl`, `code/transition/transition_values.jl`,
`code/plots_and_tables/transition.jl`, `code/tools/{migrate_bundles,width_audit}.jl`.
**`code/mcmc/` was deliberately NOT reverted**, so v37.1-37.3 (the LMR-faithful defaults, the
LMR print block, `outcome`, `on_dump`, `MCMC_GAMMA_SCALE`, the theta-space posterior mean)
survive intact. Two spec-independent fixes were re-applied by hand after the checkout,
because the revert swept them away with the files they lived in:

- `migrate_bundles.jl` defines `const WINDOWS`, which `paths.jl:48` owns, so the tool will
  not load at all. Its own selection is `MIG_WINDOWS`, defaulting to the `paths.jl` list.
- the `MCMC_LMR_PRINT` and `MCMC_GAMMA_SCALE` rows in `SETTINGS_REGISTRY`, which describe
  settings consumed in `code/mcmc/` and are unrelated to the specification.

**ACCEPTANCE TEST: the restored bundle reproduces its stored objective to 2.96e-12.**
`output/estimates/estimate_base_fc_diagonalW.jls` was restored from the same commit —
provenance version 35.1.0, stage `smm`, 23 free parameters, `loss_opt = 1074.224541` — and
re-evaluating `smm_objective` at its `theta_opt` under the reverted code returns
**1074.224541**. Same model, same criterion, same value: the revert is exact, and no
translation or migration was needed anywhere.

The free set reads `A P_S a_Gamma a_ell b_S b_U b_Gamma b_ell c k_S k_U alpha_U beta_S beta_U
delta_S lambda_S lambda_U mu_S mu_U xi_S xi_U rho_x sigma_S`, with `bT = 0.0` among the fixed
entries.

**`b_T` returns as a fixed-at-zero entry, which is what it always was.** The v36 purge removed
a `ParamSpec` that `FIX_PARAMS` already pinned at zero, so the training value function at
`645c3e0` is `(0 + phi*U_S)/(r+phi+nu)` — numerically identical to the v36 form that dropped
the term. Reverting therefore resurrects a dead entry, not a live parameter, and the model is
unchanged by its presence.

**Why the reversion is justified, since v36.0.0 argued the other way.** The v36 restrictions
were imposed to rescue the DERIVATIVE-based standard errors: freeing `beta_U` put it entirely
in the discarded subspace of the truncated `pinv` (retained-variance fraction 0.0000 while
reporting `|t| = 13.3`) and inflated the measured derivative noise floor 16x. Both harms are
properties of `se_bound = f(Ghat, Jhat^-1)`. With the chain as the estimator there is no
`Ghat`, no `pinv` and no noise floor, so the justification does not survive the change of
estimator. Under chain-based standard errors an unidentified coordinate reports a WIDE
posterior SD, which is the honest result, rather than a spuriously tight `se_bound`.

A second v36-era claim also falls: that the free-parameter count and the acceptance collapse
were the same problem. The v37.3.0 scan settles it — at a correct step, moving 20 coordinates
instead of 1 costs a factor of 1.9 in acceptance, so 23 coordinates is nearly free.

**Notes.** No revert was required. The v36 notes were delivered as artifacts and never
installed into `latex/`, so `latex/roysearch_model_notes.tex` and `latex/data_and_moments.tex`
still carry `b_T`, `b_U` and `b_S` and contain no v36 language. They are, however, OLDER than
the copies the author pasted on 2026-09-20 — `data_and_moments.tex` by about 17,000
characters — so the repository copies are not the working versions.

**Comparability.** Every `Q` under v35.1.0 and earlier is directly comparable again, which is
the point of a verbatim revert. The v36 and v37 `Q` values are not comparable to these, and
the v37 `base_fc` bundle that the restored file replaced is the only record of that
specification's estimate; it is preserved in the git object store at the prior working-tree
state and in `.migration/` as the version-neutral intermediates.

**Known gaps.**

- No window has been re-estimated under v38. The restored `base_fc` point is the v35.1.0
  estimate, which is where the `se_bound` work started and is a seed, not a new result.
- `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` remains an obsolete `SETTINGS_REGISTRY` row, pre-existing
  and untouched.

---

## v37.3.0 — 2026-09-21

**MINOR: the DE scale is a settable constant, and the acceptance collapse now has a measured
cause.** `MCMC_GAMMA_SCALE` (new, default 1.0 = ter Braak's and LMR's value) multiplies
2.38/sqrt(2*delta*n_masked). One constant, no feedback loop; `MCMC_GAMMA_ADAPT` reaches the
same place adaptively and stays off by default.

**The measurement that motivates it.** A perturbation scan at the v37 `base_fc` point, 150
solves, asking directly what step the Metropolis rule will accept (alpha = min(1, exp(-dQ/2))):

| coordinates moved | step 1e-4 | 1e-3 | 1e-2 |
|---|---|---|---|
| 1 | 0.998 | 0.886 | 0.799 |
| 5 | 0.997 | 0.739 | 0.266 |
| 20 | 0.961 | **0.455** | **4.1e-06** |

Three things follow, and two of them retract earlier claims in this file.

- **The posterior is not a spike, and seeding at the optimiser's point is not the problem.**
  Moving all 20 coordinates by 1e-3 is accepted 45.5% of the time — above the
  Roberts-Gelman-Gilks optimum of 0.234. A well-scaled chain would mix here. At 1e-4 the
  median dQ is NEGATIVE at k = 5 and k = 10, so theta_hat is not a local minimum at the
  solver's own resolution, which is the arithmetic behind "the optimiser keeps wiggling".
- **The usable step is ~1e-3 and the population settles near 1e-2** (measured: spread grew
  4.5e-04 to 1.03e-02 over 60 generations, still rising). The standard gamma therefore
  overshoots the band by roughly a factor of 4 on this model. That, not the start and not the
  dimension, is the collapse.
- **`MCMC_CR` is NOT the fix, and the claim that it was is withdrawn.** At the right step,
  going from 1 to 20 coordinates costs a factor of 1.9 in acceptance; at the wrong step it
  costs a factor of 194,285. The coordinate count only matters once the step is already too
  large. **The corollary is that the free-parameter count is nearly free**: 23 coordinates
  instead of 20 is worth a few percent of acceptance at a correct step, so "the spec choice
  and the acceptance problem are the same choice" was wrong and is withdrawn.

**Two claims about LMR corrected, from their own files.**

- **They seed at their published estimate.** `estimation_ed1_no_growth/data/starting_val.raw`
  holds 16 unconstrained values; pushed through their own transforms
  (`read_MCMC_chain.m:65-79`) all **14 comparable entries reproduce Table 5 to every digit
  the paper prints** (max |diff| 4.3e-04): eta 0.650, s 0.332, xi 0.011, delta 0.017, b 0.122,
  c 2.344, A 6.964, alpha 0.607, rho 0.938, beta 0.188, sigma^2 0.006, and both Beta shapes.
  So the shipped configuration starts the chain AT the answer. Seeding from an optimiser
  output is their practice too, and "LMR hand-calibrated a cold start" is withdrawn.
- **Their objective is not smoother than ours.** Earlier entries lean on a contrast between
  their smooth simulated criterion and our C0-kinked one. Simulated moments under common
  random numbers are piecewise CONSTANT in theta wherever a simulated agent's discrete choice
  flips, which is rougher than a kink, and the scan above shows our own criterion is locally
  very well behaved (alpha = 0.998 at a 1e-4 step). The contrast does not support the weight
  put on it.

**Known gaps.**

- No value below 1.0 has been RUN. The scan says where the step should land; it does not
  establish that a fixed lower gamma holds the population there, because the equilibrium
  spread is an outcome of the whole recursion. A short two-arm comparison at 1.0 against
  ~0.3 is the test.
- The scan omits the log-Jacobian term of logpi, which is second order in the step at an
  interior point but not zero.

---

## v37.2.0 — 2026-09-21

**MINOR: `MCMC_main.jl` now runs the LMR-faithful chain by default.** No script, no
environment block — `julia --project=. -t 10 code/mcmc/MCMC_main.jl` is the run. The
`run_lmr_chain.sh` launcher added in v37.1.0 is **removed**; a launcher that duplicates the
defaults is a second place to keep them correct.

**Defaults changed.** Each comment was rewritten where the old justification argued for the
old value; the accreted budget block above `MCMC_N` was rewritten whole rather than patched.

| setting | was | now | why |
|---|---|---|---|
| `MCMC_N` | 95 | **120** | 6.0 chains per free parameter at d = 20, against LMR's 6.25 at d = 16 |
| `MCMC_GENS` | 10 000 | **4 000** | ~48 h at the measured 43 s/generation; theirs is 94 h |
| `MCMC_BURN` | 0.9 | **0.75** | their retained quantity is the last **1 000 generations**, absolute; 0.9 of 4 000 keeps 400 |
| `MCMC_JAC_ONLY` | true | **false** | the chain is the estimator now, not the local design |
| `MCMC_GAMMA_ADAPT` | true | **false** | no step-size brake in their sampler |
| `MCMC_GAMMA_EVERY` | 25 | **0** | same |
| `MCMC_CHECK_EVERY` | 250 | **0** | they compute no R̂, no ESS, no MCSE anywhere |
| `MCMC_LMR_PRINT` | 0 | **500** | their `stat_display_rate` |

`MCMC_ACC_FLOOR`, `MCMC_MOVES_MIN`, `MCMC_DRIFT_FLAT` and `MCMC_MCSE_TARGET` keep their own
justified values and are **inert** while `MCMC_CHECK_EVERY = 0`, because they are that stop's
thresholds rather than independent checks. One switch turns the stop off and each threshold
keeps its reasoning for when it is back on. Reverting `MCMC_ACC_FLOOR` to 0 would discard the
argument recorded at its declaration — that it ends a dead ten-hour run at the first check —
and that argument is still right; it is simply not this estimator's to apply, since a chain
whose acceptance is falling is not yet a chain that has failed.

**`on_dump`, a new `run_demc` callback, closes a real exposure this change opens.** With no
abort, a run ends when the operator ends it, and `MCMC_CHECKPOINT` saves the best POINT and
never the draws — so a hand-stopped run would have left nothing the estimator can read. The
existing `snapshot_chain` is now called periodically at the `lmr_print` cadence, which is
LMR's own structure: their `if (op%dump_chain)` sits inside the `stat_display_rate` block
(`mpi_mcmc_mod.f90:459-468`), so a printed block and the draws behind it always correspond.
Suppressed at `g == gens`, where the caller writes its own snapshot.

**What the August chain says about this estimator, measured from the bundle.** These are the
numbers the settings above have to survive, and they are not reassuring:

| | |
|---|---|
| accepted moves, whole run | **5 768** of 95 000 proposals (6.07%) |
| accepted moves, **retained window** | **77** of 9 500 (0.81%) |
| per chain in the retained window | **0.81** (min 0, median 1, max 4) |
| chains that never moved in the retained window | **47 of 95** |
| distinct values per coordinate, pooled retained sample | **157** of 9 500 draws (1.65%) |

and the moves are front-loaded exactly as the author observed: 3 621 in generations 1–100,
decaying monotonically to 77 in 901–1 000. **So the retained window is the part of that run
with the fewest moves, not the most.** Burn-in discards the early generations because they
depend on the start; that logic presupposes stationarity, and a chain still descending has
not reached it. The consequence is visible in the reported table: `se_chain` of 3.2e-04 on
`a_ℓ` with `|t| = 4230`, and 1e-05 on `b_ℓ` with `|t| = 4.8e+05`. **Those are the spread of a
frozen population, not a posterior standard deviation.** Near-zero acceptance in the
retained window manufactures precisely the spurious precision the `se_bound` work was trying
to avoid, by a different route.

**Known gaps.**

- **Nothing here makes the chain stay alive.** It records LMR's acceptance breakdown so the
  failure is visible — a high strong-accept share means the chain is still climbing — and
  dumps the draws so a hand-stop is recoverable. Neither keeps acceptance up. On the
  evidence above, the LMR-exact configuration applied to this model produces a frozen
  population, and the knob that addresses it within their own parameterisation is `MCMC_CR`:
  at 0.75 each proposal perturbs ~15 of 20 coordinates and all must be jointly acceptable,
  so acceptance falls geometrically in the number moved. Lowering it trades per-coordinate
  mixing for far more accepted moves. **Untested here.**
- The `check_structure.jl` net-depth finding on `demc.jl` grows again for the same reason as
  in v37.1.0: the gate counts every `for` as a block opener and comprehensions have no `end`.
  Julia's parser accepts the file.
- `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` remains an obsolete registry row, pre-existing.

---

## v37.1.0 — 2026-09-21

**MINOR: an LMR-faithful chain can now be launched from one script, and it reports what LMR
report.** Nothing about the model, the moments or the objective changed; stored bundles load
unchanged and every `Q` stays comparable.

**Why this exists.** The reported standard errors come from `Ĵ = Ĝ'WĜ`, a numerical
derivative of a criterion with C⁰ kinks from the grid cutoff softening, and the derivative is
the weakest link: `cond(Ĵ) = 6.35e+11`, `σ_min` below the measured derivative noise floor,
and one coordinate's finite-difference step capped at 1.6e-04 of its box by the distance to a
bound. Lise, Meghir and Robin (2016, RED) footnote 21 hit the same wall and resolved it by
abandoning the derivative: *"Because of the lack of precision in the numerical approximation
of the gradient G(θ) … we report the variance of the MC chain."* Their estimator needs no
derivative at all, and Appendix C states it exactly:

> "In practice, we simulate 100 chains in parallel, each of length 10,000, and use the last
> 1000 elements (pooled over the 100 chains) to obtain parameter estimates and the standard
> errors."

**`run_lmr_chain.sh`** (new, repo root) is the one named entry point for that run. It hard-sets
every setting rather than reading the ambient environment, so the configuration is
reproducible from the file alone; `--smoke` is the single documented exception, running 12
chains × 10 generations to exercise the whole path.

*What it switches OFF.* Every mechanism this codebase added after a run of ours misbehaved,
because none has a counterpart in LMR's sampler and with them on the estimator is not theirs:
`MCMC_GAMMA_ADAPT`, `MCMC_GAMMA_EVERY`, `MCMC_ACC_FLOOR`, `MCMC_CHECK_EVERY`,
`MCMC_MCSE_TARGET`, `MCMC_MOVES_MIN`, and `MCMC_JAC_ONLY`.

*What was already faithful and is left alone.* `MCMC_CR = 0.75`, `MCMC_DELTA = 2`,
`MCMC_B_MULT = 1e-2`, `MCMC_B_ADD = 1e-4`, `MCMC_T_BOX = 20.0`, the mode jump every 10th
generation, `MCMC_OUTLIER_IQR = 2.0` running throughout, and `MCMC_INIT = :at_seed` — which
already places every chain at exactly the seed with zero spread, citing
`mpi_mcmc_mod.f90:268`. `MCMC_INIT_DISPERSE` is not in that code path and never was.

*Burn.* 0.75 at 4,000 generations, which retains the **last 1,000 generations** — LMR's
absolute quantity, not the 0.9 fraction. 0.9 would keep 400, a quarter of their pooled
sample.

**`MCMC_LMR_PRINT`** (new setting, default 0 = off; registered, forwarded, consumed in
`demc.jl:run_demc`) emits their periodic block, `mpi_mcmc_mod.f90:429-455`: the loop header,
the mean estimate, the max estimate, the posterior spread, and the five-way accept/reject
breakdown. It is **additive** — the existing `[demc]` line stays, because it carries the
median proposal's `Δlogπ` and ESJD, the two step-size numbers their block omits.

*One difference is printed rather than silently absorbed.* Their mean-estimate line averages
the raw stored values, which under `MCMC_SPACE = :t` is a `t`-space mean; the paper's
estimator is the mean of the constrained draws. Both are printed, labelled `mean (θ)` and
`mean (t)`.

**`run_demc` now returns `outcome`**, an `Int8` N×G matrix carrying LMR's five-way code
(3 strong accept / 2 equal / 1 weak accept / 0 weak reject / −1 no prior mass). `accepted`
says whether a move happened; this says which kind. The distinction is load-bearing: a high
strong-accept share means the chain is climbing, so it is still descending rather than
sampling. Measured at 12 chains × 10 generations from the v37 `base_fc` seed, generation 5
reported **0.417 strong / 0.567 weak accept / 0.017 weak reject**.

**One inconsistency fixed in the reported estimator.** `post_mean` was the transform of the
`t`-space mean while `se_chain` was the SD of the θ-draws, so the pair were moments of two
different samples and the code claimed an equivalence with LMR it did not deliver. The mean
now maps each draw to θ first, matching both `se_chain` and `read_MCMC_chain.m:65-79`. A
`θ̄_bnd` fallback keeps the old `t`-space mean for the post-mean *bundle* only, because the
θ-space mean has been observed infeasible on one base point; the `isfinite(Q̄)` gate chooses,
and the console names which was used. **Measured Jensen gap on the 2026-08-31 chain: median
0.009 reported SD, max 0.13 on `b_S`, zero coordinates above 1 SD** — so the pair is now
internally consistent and no reported number moves.

**Cost, measured rather than extrapolated.** 33.94 s per generation at `N = 95` on 10
threads, timed through `run_demc` itself so it includes the real threading — candidates are
spread *across* threads, which amortises to 0.357 s per evaluation, not the 1.84 s a single
10-threaded solve takes. At `N = 120`: ~43 s/generation, so 4,000 generations is ~48 h on 10
threads. One full LMR run of 10,000 is 94 h.

**Known gaps.**

- **Nothing here says the chain will stabilise.** The longest test run is 60 generations, at
  which acceptance was 0.35 — still above the Roberts-Gelman-Gilks optimum of 0.234 — with
  the population spread still growing (4.5e-04 → 1.03e-02, a 157× inflation from the
  zero-spread start). Whether it settles near 0.234 or falls through to 0.02 as the
  2026-08-31 run did is **not established**. The diagnostic that separates the two is
  `spread_growth`, already computed at `MCMC_main.jl:1324` and emitted in the results CSV —
  not `accept`, which falls in a healthy run too.
- One explanation was tested and **refuted**: that the population expands along the flat
  directions and so poisons the single DE-MC step vector for the sharp ones. Holding `α_U`,
  and then `α_U + a_Γ + A`, at the seed left the acceptance decay identical (final block
  0.350, 0.350, 0.367) and spread growth non-monotone in `d` (157×, 113×, 158×). Do not
  revive it.
- `check_structure.jl` reports `demc.jl` at net depth 11 against 5 before this change. The
  gate's counter treats every `for` as a block opener, and comprehensions and generator
  expressions use `for` with no `end`; the block added here contains six. Julia's own
  `Meta.parseall` accepts both versions, and the gate already flagged seven files that all
  parse. The counter is the defect, not the files, and it is left for its own pass.
- `ROYSEARCH_MCMC_JAC_ALLOW_GAPS` remains an obsolete `SETTINGS_REGISTRY` row with no
  reader, pre-existing and untouched here.

---

## v37.0.0 — 2026-09-21

**MAJOR: `b` is pinned at zero, and a railed coordinate is now detected rather than left to
be noticed.** `b` free lands on its own floor, and a coordinate on a rail does not merely
lack an interval of its own — it caps its own finite-difference step and so corrupts the `Ĵ`
that every other coordinate's interval is read off.

**Comparability.** The v36.0.0 `base_fc` estimate (`Q = 1183.127505`, 21 free) is superseded:
the next run estimates 20 parameters with `b ≡ 0`. Nothing else about the model changed, so
`Q` values remain on the same scale and are comparable as fits of nested specifications — v37
is v36 with one coordinate restricted, and the restriction is at the boundary the free
estimate chose anyway.

**What changed.**

*`FIX_PARAMS` gains `:b => 0.00000`* (`smm_main.jl:270`). Free set on a baseline window
21 → 20.

*A measured rail test* (`MCMC_main.jl`). `_rail_frac(θ, lb, ub)` and `RAIL_TOL = 1e-3` next to
`_abs_t`; a new `rail_frac` column on `mcmc_results_<window><suffix>.csv`; a ` RAIL` flag
ahead of ` edge` and ` R̂` in the console table; and a block after the table that names every
railed coordinate and says what to do about it. The two columns that were supposed to cover
this do not: `corner_declared` is a manual label read off the `CORNER_PARAMS` setting
(`MCMC_main.jl:678`, and its own print says "a label only"), and `edge_frac` is a chain
statistic, so it is `NaN` on the `JAC_ONLY` path — which is the path this project reports
from. The footer no longer tells the reader to consult `edge_frac` there.

**Measured, not asserted.** All at the v36.0.0 `base_fc` point, `Ĝ` from the adaptive FD,
244 solves, 21 of 21 columns central.

- `b = 0.00022052` on a `[0, 1.4]` box: **1.57e-04 of box width from the floor**, the only
  coordinate under `1e-3`. Nothing in the v36 output said so — `corner_declared = 0` because
  `b` is not in `CORNER_PARAMS`, and `edge_frac` was blank.
- That rail is why the run reported `b 0.25` as the one coordinate outside the FD target band.
  `mcmc_diagnostics.jl:33` sets `edge = 0.999·min(θ̂−lb, ub−θ̂)` and line 302 caps
  `h_hi = min(0.25·w, edge)`, so `b`'s derivative is measured across **1/6355 of its box**.
  It is not that `b` fails to move the moments: at 2% of box width `‖Δm/σ̂‖ = 17.4`, at 10%
  it is 88.3, and at 50% it is 1534. The response is there; the room to measure it is not.
- Dropping `b`'s column: `cond(Ĵ)` 6.353e11 → 2.266e11, `σ_min` 0.0207 → 0.0347, margin over
  the noise floor 0.36 → 0.60, and coordinates with retained variance above 0.99 **4 → 8**.
  The gainers are the skilled block — `σ_S` 0.318→0.9999, `b_Γ` 0.337→0.997, `k_S`
  0.416→0.9998, `β_S` 0.529→0.994, `a_Γ` 0.246→0.881, `δ_S` 0.279→0.865, `a_ℓ` 0.272→0.783.
  Two lose: `k_U` 0.369→0.006 and `α_U` 0.024→0.000, which now carry the flat direction
  alone. Their `se_bound` falls to 0.104 and 0.697 respectively and **must not be reported** —
  that is the spurious-precision pattern, a near-zero interval on a coordinate with no
  retained variance.
- v36.0.0 was not the failure it looked like. Reportable coordinates at `|t|(bound) ≥ 2` went
  **5 of 23 under v35.1.0 to 9 of 21 under v36.0.0**, with none lost: `A` 0.22→4.90
  (`se_bound` 34.97→1.62), `λ_U` 0.13→39.28 (1.273→0.0038), `ξ_U` 0.00→5.96 (1.272→0.0026),
  `ρ_x` 1.68→2.43.
- The `:b => 0.0` spec builds, pins, and evaluates: free set 20, `b` absent from the free set,
  `0.00000000` reaching the solver, `assert_all_params_accounted` clean, finite `Q`.

**Settings.** `FIX_PARAMS` gains `:b`. No registry setting added or retired, so
`check_forwarding` is unaffected. `RAIL_TOL` is a module constant, not an env setting — it is
a reporting threshold, not a knob a run should vary.

**Known gaps.**

- **Three more coordinates are pressing bounds and only one of them looks economic.** `δ_S`
  0.997604 on `[0.05, 1.0]` is 2.5e-03 of box width from the **ceiling**, and `δ_S = 1` is the
  single-distribution limit where a shocked match redraws a fresh offer — so that one is the
  data speaking. But `b_ℓ` 3.976461 on `[0.05, 4.0]` is 5.95e-03 from its ceiling and `P_S`
  3.214928 on `[1.6e-05, 3.257928]` is 1.32e-02 from its, and those read as boxes that are too
  tight rather than as corners. Whether to widen them is not settled here. In all, **13 of 21
  coordinates** have their FD window capped by a bound rather than by the 0.25 ceiling.
- No re-estimation has been run with `b` pinned, so the 4 → 8 figure is the geometry at the
  v36 point with a column removed, not the geometry at the v37 argmin.
- The rail summary block reads `θ0` through the same `_draw_to_θ` idiom the CSV writer uses
  two lines above it, and was verified against the writer's own output rather than by running
  `MCMC_main.jl` end to end.
- Two b's was considered and rejected on measurement, not taste: under v35.1.0 `b_S = 0.00003`
  on `[0, 0.325793]` was **also** on its floor (9.2e-05 of box width) with `|t| = 0.00`, and
  `b_U` was interior with `|t| = 0.03`. Going back buys two contaminated columns where there
  is now one, and the earlier scenario sweep already found the single shared `b` gave the
  tightest `se_bound` of the configurations tried (0.2951 against 0.3783 for `b_S` free).

---

## v36.0.0 — 2026-09-20

**MAJOR: one non-employment flow value, loading on ability, replaces `b_U`, `b_S` and `b_T`;
`β_U` is fixed at the symmetric Nash split.** A parameter was removed, two were merged and
redefined, and the training value function lost a term — every stored estimate is an estimate
of a different model.

**Comparability.** None. Every `Q` under v35 and earlier is an objective value of the old
specification and cannot be compared with a v36 `Q`. **The four stored bundles in
`output/estimates/` are NOT migrated and must not be.** `migrate_bundles.jl` exists to carry a
bundle across a *struct* change while holding the model fixed, and its acceptance test is that
the rebuilt bundle reproduces its stored `Q`; a respecification cannot satisfy that test by
construction. `extract` was run before the struct edit and the four version-neutral
intermediates are in `.migration/` with the `Q` values they carried — `base_fc` 1074.224541,
`crisis_fc` 713.207566, `base_covid` 2834.373205, `crisis_covid` 1816.503262. They are the
readable record of the v35 point and the source of the v36 seed; `rebuild` was deliberately not
run.

> **Corrected in v39.3.0.** Those intermediates no longer hold the v35 point — an `extract`
> run on 2026-09-22 overwrote them and v39.3.0's migration consumed `.migration/` entirely.
> The readable record of `base_fc` at `Q = 1074.224541` is
> `output/estimates/estimate_base_fc_diagonalW_backup_Q1074.224541.jls`, carried to v39.3.0
> struct shape with `migrated_from = 35.1.0` and its `loss_opt` bit-identical.

**What changed.**

*Model.* The flow payoff of a worker searching in market `j` is now `b(x) = b·x·exp(A)`, where
`x` is her aptitude in the market she is searching: `a_U` when searching unskilled — including
a skilled-unemployed worker who has crossed over — and `a_S` when searching skilled. One
parameter `b` in `CommonParams` (`params.jl:66`) replaces `up.bU` and `sp.bS`. The asymmetry the
data show is now produced rather than estimated: skilled searchers have higher `a_S` and draw a
larger flow without a second parameter. The replacement rate is `b(x)/π_j(x,p) = b/(P_j·p)`,
independent of ability and falling in match quality, so benefits track earnings *capacity* and
not realised earnings — what a base-period formula does. This is **not** a capped schedule; a
cap would make the rate fall in ability, and it does not.

*`b_T` purged.* Training carries no flow payoff: `(r+φ+ν)T(a_S) = φ U_S(a_S)`
(`unskilled.jl:145`, `transition_values.jl`). The parameter, its `ParamSpec`, its `FIX_PARAMS`
pin and its key-map entries are gone.

*`β_U` fixed* at 0.50000 in `FIX_PARAMS` (`smm_main.jl:266`) — the symmetric Nash split, which
is also the Hosios value at `η_U = 0.5`. `β_S` remains free. Free set on a baseline window:
23 → 21.

*Policy levers.* `UnskilledParams.b_mult` and `SkilledParams.b_mult`, default 1.0, never
estimated. With a single common `b` a market-targeted benefit reform is otherwise
inexpressible; these keep the three one-sided policy experiments available as perturbations of
a symmetric baseline rather than as separate parameters.

*Tooling.* `migrate_bundles.jl` defined `const WINDOWS`, which `paths.jl:48` has since claimed;
the tool would not load. Its own selection is now `MIG_WINDOWS` and defaults to `paths.jl`'s
list, so the window set has one home.

**Measured, not asserted.**

- Structs after the edit: `CommonParams` 9 fields (`b` appended last), `UnskilledParams` 10
  (`bU`, `bT` removed, `b_mult` appended), `SkilledParams` 13 (`bS` removed, `b_mult`
  appended). The removals were the hazard the migration existed for: `bU`/`bT`/`bS` are
  `Float64` and so are the fields that follow them, so an unmigrated bundle would have loaded
  **silently** with `α_U`, `ξ`, `σ_w`, `a_Γ`, `b_Γ` all shifted one slot.
- Solve at the translated `base_fc` point, grids 160/160/180 from the bundle's own
  `SMMRunParams`: converged on all three loops, `θ_U` 1.338301, `θ_S` 0.909368, `u_U` 0.031569,
  `u_S` 0.009902. `U^search` and `U_S^(0)` are both monotone increasing in ability, which they
  could not be before — `b` entered as a constant.
- Seed translation, `base_fc`: `b = b_U/E_ℓ[a] = 0.06608778/0.304428 = 0.21708812`, so the
  *average* flow is unchanged and the unskilled market starts near where it was. At that seed
  `Q = 25348.13` against 1074.22 under v35 — the point is not an optimum of the new model and
  re-estimation is required.
- **`β_U` earns its pin.** Adaptive-FD Jacobian at the same seed, 262 and 250 solves. Free:
  retained-variance fraction above 0.99 for 4 of 22 parameters, derivative noise floor 1.237,
  `σ_min/floor` 0.01. Fixed: 6 of 21 above 0.99, floor 0.07869, margin 0.18. Seven parameters
  move from below 0.10 retained to above 0.85 — `a_Γ` 0.0015→0.9963, `P_S` 0.0005→0.9949,
  `b_Γ` 0.0020→0.9863, `k_S` 0.0004→0.9851, `β_S` 0.0001→0.9774, `σ_S` 0.0096→0.9589,
  `δ_S` 0.0010→0.8814. Free `β_U` has retained fraction **0.0000** with `se_bound` 0.0509 and
  `|t|` 13.3 — the spurious-precision failure mode exactly: a parameter lying entirely in the
  discarded subspace reported as sharply estimated.
- Gates: `check_dead_settings` PASS. `check_forwarding` and `check_structure` report the same
  findings as the committed tree and none of them names a file changed here.

**Settings.** No setting added or retired. `FIX_PARAMS` gains `:unsk_bet => 0.50000` and loses
`:bT => 0.00000`; `DEFAULT_PARAMS` loses `:bU`, `:bT`, `:bS` and gains `:b => 0.20144823`
(`0.06247633 / E_ℓ[a]` at that block's own shapes, `0.31013591`).

**Known gaps.**

- **`code/policy/` is dead and was left dead.** It is not a casualty of this change: it
  references a type `RegimeParams` that is defined nowhere in the repository and calls
  `solve_model` with five positional arguments against the one method that takes four. It
  could not load before this version either. Reviving it means rebuilding it against the
  current `solve_model` signature; the `b_mult` fields are there for when that happens.
- No window has been re-estimated under v36, so there is no v36 estimate of `b` and no
  statement here about whether the standard errors improve at a genuine optimum. The
  `β_U` measurement above is at the translated seed, which is far from one, and the geometry
  at the argmin may differ.
- The `|t|` column of that measurement is uninformative at a non-optimum and was not used to
  draw the conclusion; the retained-variance fraction was.

---

## v35.1.0 — 2026-09-17

**MINOR: the adaptive-FD Jacobian handles a convergent domain that is not an interval around
`θ̂`, and an unmeasurable column now stops the run instead of being filled from the regression.**
Windows that already worked reproduce exactly; one more window became measurable; runs that
previously wrote invalid numbers now refuse to.

**Comparability.** `base_covid` is bit-identical to v35.0.0 — 280 solves, instability median
1.57e-05, step range 2.86e-07–1.00e-02, 20 of 23 on target, `cond(G̃)` 1.07e9. The rewritten
phase 1 only behaves differently at a *failed* probe, which that window never hits. What changed:
`crisis_fc` now measures 17 of 18 columns where it previously could not, and `base_fc` and
`crisis_fc` raise rather than emit a `mcmc_results_*.csv` built from two estimators. **Any
`mcmc_results_base_fc_diagonalW.csv` written under v35.0.0 must be discarded**: 18 of its 23
columns came from the design regression at `rel_step = 0.05` while the header read "adaptive FD",
and nothing in the file marked which column came from where.

### Per-window coverage, measured

| window | threads | free | measured | on target | one-sided | `Q(θ̂)` finite | instability | `cond(G̃)` | valid |
|---|---|---|---|---|---|---|---|---|---|
| `base_fc` | 10 | 23 | 5 | 0 | 0 | **no** | 2.69e-01 | — | no |
| **`base_fc`** | **1** | 23 | **23** | 22 | **23** | yes | 6.86e-03 | 2.79e6 | **yes** |
| `crisis_fc` | 10 or 1 | 18 | 17 | 17 | **17** | yes | 1.26e-04 | 6.45e4 | conditional |
| `base_covid` | 10 | 23 | 23 | 20 | 0 | yes | 1.57e-05 | 1.07e9 | **yes** |
| `crisis_covid` | 10 | 18 | 18 | 18 | 0 | yes | 2.16e-05 | 4.10e5 | **yes** |

**`base_fc` requires `JULIA_NUM_THREADS=1`** and is valid there: its `θ̂` has a finite `Q` at one
thread, which makes the one-sided branch available, and all 23 columns then measure — every one of
them one-sided, since the hole above `θ̂` rules out a central difference at any step. The cost is
accuracy: a one-sided difference is first-order, so Richardson leaves `O(h²)` where a central
difference leaves `O(h⁴)`, and the instability is 6.86e-03 against 1.57e-05 at `base_covid`.
Those are the least accurate standard errors in the set and the gap is a property of the point.
`crisis_fc` is identical at 1 and 10 threads, so its one missing column is not a threading
artefact.

`pinv` keeps 18 of 23 directions at `base_covid` and all 18 at `crisis_covid`, so the truncation
recorded as a known gap in v35.0.0 is real on one window rather than everywhere.

### What changed

* `mcmc_diagnostics.jl` — `cdiff` becomes `dquot`, returning the difference quotient, the moment
  displacement, and the **order of the leading error term**: 2 when both sides solve, 1 when only
  one does and the centre is used instead. The convergent domain is not an interval — on
  `base_fc` the solve succeeds at `θ̂` and at radii ≥1e-3 of box width but fails at 1e-6…1e-4
  *above* it while every radius below succeeds — so requiring a central difference discarded
  coordinates one side measures cleanly. This alone took `crisis_fc` from unusable to 17 of 18.
* `mcmc_diagnostics.jl` — Richardson is now order-aware: `(2^p·D(h/2) − D(h))/(2^p − 1)`, applied
  only to adjacent rungs of the **same** order. A pair straddling a switch from central to
  one-sided is skipped rather than extrapolated with weights for the wrong order, which would add
  more error than it removes.
* `mcmc_diagnostics.jl` — the step search distinguishes the two directions a probe can fail in. A
  failure while GROWING is an outward barrier and lowers the cap; a failure while SHRINKING means
  a hole lies between `θ̂` and the current step, so shrinking further goes deeper into it and the
  search stops with the step that works. Treating them alike is what pinned every `base_fc`
  coordinate at the upper rail with a displacement of 26 against a target of 1.
* `mcmc_diagnostics.jl` — an infeasible `θ̂` is recorded as `diag.centre_ok` instead of raising.
  A central difference evaluates only `θ̂ ± h`, so it never needed the centre; the assertion was
  the sole reason a `base_fc` run could not start. Note the coupling this exposes: the one-sided
  branch *does* need the centre, which is precisely why `base_fc` gets neither difference.
* `MCMC_main.jl` — no fallback. An unmeasurable column raises, naming the coordinates, pointing at
  `converged` on the bundle as the usual cause and at `MCMC_JAC_METHOD=:design` for the pre-v35
  object whole. Either every column is measured the same way or there is no table.
* `MCMC_main.jl` — the summary reports the off-target coordinates with their displacements,
  columns holding a single extrapolate (no stability check behind them), the `centre_ok` note, and
  the rank `pinv` retains.
* `mom_at` fills its buffer with `NaN` rather than `undef` and keys on the returned objective.
  `smm_objective` writes `moments_out` only after its five guards, so an unwritten `undef` buffer
  holds finite garbage — a trap that made a diagnostic of mine report "35 of 35 moments finite" at
  a point where `Q` was `Inf` and no moment had been written.

### Measured, not asserted

Parse clean on both changed files. `base_covid` reproduces v35.0.0 exactly on all eight recorded
quantities. All four windows run end to end through the JAC_ONLY construction: 462 solves/550 s
(`base_fc`, 10 threads), 210/116 s (`crisis_fc`), 280/62 s (`base_covid`), 214/56 s
(`crisis_covid`).

Solver determinism, measured on `base_covid` because the question came up while diagnosing
`base_fc`: **12 repeats of the identical call give one bit-identical value at 1 thread and ten
distinct values at 10 threads, relative spread 9.47e-15 = 42.6 × machine epsilon.** The
1-thread answer lies inside the 10-thread range, so the jitter is symmetric about the serial
answer rather than biased. That is a reduction-order effect from `@threads :dynamic`, not a data
race — a race on the `threadid()`-indexed buffers in `skilled.jl`/`unskilled.jl` would corrupt
values at order one, not in the fourteenth digit. It is numerically harmless and is recorded only
because the replication target wants a bit-reproducible run.

### Known gaps

* **`base_fc` cannot be run multithreaded on this route.** At 10 threads it can use neither a
  central difference (a hole immediately above `θ̂`) nor a one-sided one (`Q(θ̂)` is not finite
  there, and the one-sided branch needs the centre), so it measures 5 of 23. At one thread it
  measures all 23. Independently of the differencing, its bundle records `converged = false` at 42
  iterations and it does not converge at the transition's finer grids — so the point itself is
  suspect. **Re-estimating the FC pair from `:clusters` rather than a warm start is the indicated
  move, on the `converged = false` alone**, and a valid Jacobian around a non-converged point is
  not an endorsement of the estimate.
* **`a_Γ` has no local derivative at `crisis_fc`'s `θ̂`, and this is a finding rather than a
  tuning failure.** Measured along that coordinate from the stored point (box `[0.1, 12]`, `θ̂` =
  0.449, i.e. 2.9% of box width off its floor): the `+` side solves at 1e-2 and 5e-3 of box width,
  both sides solve at 5e-3, then **four consecutive rungs fail on both sides** (2.5e-3 through
  3e-4), the `−` side alone solves at 1e-4, the `+` side alone at 1e-5, and both fail again at
  1e-6 and 1e-7. Feasibility alternates sides with a dead band in the middle, so a halving ladder
  from any feasible start dies at the next rung. The one same-side factor-2 pair that does exist
  sits at displacements of 57 and 70 σ̂ — with displacement *rising* as the step shrinks, which is
  itself evidence the moments are not locally smooth there. No step rule recovers a derivative
  from that.
* **`pinv` still truncates 5 of 23 directions at `base_covid`** and `se_bound_diagonal` is
  unchanged. Carried forward from v35.0.0; now known to be window-specific.
* The `converged_S` test at `skilled.jl:679` is a bare threshold on a residual that `base_fc`'s
  `θ̂` sits exactly on, which is why 42 eps of scheduling jitter flips `Q` between finite and
  `Inf` there. Not touched: changing it changes what the optimiser accepts, not only what the
  diagnostics say.
* No estimation was re-run. `mcmc_results_*.csv` still has no readers anywhere in the repo.

---

## v35.0.0 — 2026-09-17

**MAJOR: `Ĝ = ∂g/∂θ` is now measured by per-coordinate central differences with the step set
from the moments' own response and refined by Richardson extrapolation, replacing a least-squares
plane fitted on a cloud of radius `rel_step·(ub−lb)`.** Point estimates do not move — no
estimation is re-run — but every reported standard error does, by a median factor of 3.4 on
`base_covid`, so the inference a reader takes from the parameter table is not comparable across
this boundary.

**Every stored `se_curvature`, `se_bound`, `abs_t_curvature` and `abs_t_bound` is superseded**,
in `output/estimates/mcmc_results_*.csv` and in any chain bundle's `se`, `G` and `G_R2`. The
point estimates, `theta_opt` and `loss_opt` are untouched. `SMMResult` and `SMMSpec` are
unchanged and the chain bundle is a NamedTuple, so **all four stored estimation bundles still
deserialise** (verified: `base_fc` d=23 Q=1073.220950, `crisis_fc` d=18 Q=713.207566,
`base_covid` d=23 Q=2834.373205, `crisis_covid` d=18 Q=1816.503262, 10 fields each).

### Why the old route could not be repaired by retuning it

`local_design` stepped by `rel_step·(ub−lb)`, so the derivative depended on the **search box**:
a bound the estimate never approaches still set the resolution at which the slope was measured,
and widening a bound changed `Ĵ`, both standard errors and every `|t|`. The regression reported
the failure itself. On `base_covid`, per-moment `R²` of the linear fit by design radius:

| `rel_step` | feasible | `R²` min | `R²` median | `R² < 0.9` | `cond(G̃)` |
|---|---|---|---|---|---|
| 0.05 (shipped) | 210/230 | 0.221 | 0.771 | 25 of 35 | 3.96e4 |
| 0.02 | 230/230 | 0.725 | 0.978 | 8 | 6.55e4 |
| 0.01 | 230/230 | 0.921 | 0.992 | 0 | 8.45e4 |
| 0.005 | 230/230 | 0.895 | 0.998 | 1 | 2.46e5 |
| 0.002 | 230/230 | 0.650 | 0.999 | 1 | 3.95e5 |
| 0.001 | 230/230 | 0.591 | 0.999 | 2 | 4.32e5 |

A median `R²` of 0.771 means roughly a quarter of each moment's movement over the ball was not
linear in θ, so the fitted plane was averaging curvature rather than measuring a slope. The
cause is curvature and not solver noise: the median rises monotonically as the ball shrinks and
only the *minimum* turns over below 0.005, where one or two moments reach their noise floor.

Retuning `rel_step` to 0.01 fixes the `R²` and does **not** fix the method: one radius cannot
serve coordinates whose moment sensitivities differ by four orders of magnitude, and the steps
the new route selects span 2.86e-07 to 1.00e-02 of box width — a factor of 3.5e4 — where the old
code used 0.05 for all 23.

### What changed

* `code/mcmc/mcmc_diagnostics.jl:211` — new `jacobian_adaptive_fd`. Phase 1 sets each
  coordinate's step so `‖Δm ./ σ̂‖ ≈ target_sd`, i.e. the perturbation moves the moment vector a
  target distance in the moments' **own sampling-error units**, so an insensitive coordinate gets
  a large step and a sensitive one a small step. The target is reached by prediction rather than
  search — displacement is very nearly proportional to `h`, so one feasible probe fixes the step
  as `h·target_sd/s` — and an infeasible probe bounds the step from above. Phase 2 walks
  `h, h/2, h/4, …`, forms `(4·D(h/2) − D(h))/3` to cancel the `h²` term, and accepts the
  extrapolate that agrees best with the next finer one. `G` comes back in natural units exactly
  as `jacobian_from_draws` returns it, so `se_bound_diagonal` is unchanged.
* `code/mcmc/MCMC_main.jl:1180` — the route is selected by `MCMC_JAC_METHOD`. The design
  regression is computed under **both** methods: its solves are already budgeted, it keeps
  `moments` and `draws_jac` populated for `curvature_check`, and its `R²` is the standing
  evidence for the choice. A coordinate the ladder cannot measure falls back to the regression
  column and is named on screen rather than putting a `NaN` into `Ĵ`.
* `code/mcmc/MCMC_main.jl:1349` — the summary block is method-aware and now reports the step
  range, the Richardson instability, the count of coordinates off the target band with their
  displacements, columns with a single extrapolate and therefore no stability check, and the
  numerical rank `pinv` retains.
* `code/mcmc/MCMC_main.jl:1580` — chain bundles gain `G_method`, `G_fd` (step, rung, per-moment
  instability, `tuned`, `n_eval` per coordinate) and `G_reg`. Additive on a NamedTuple; older
  bundles simply lack the keys.
* `code/mcmc/MCMC_main.jl:1419` — the console table gains `|t|(J⁻¹)` and `|t|(bnd)`, both
  `|θ̂|/se`. Plain ratios, not test statistics: zero is a meaningful null only for `ρ_x`.
* `code/smm/settings.jl:217` — `MCMC_JAC_METHOD` registered.

### Measured, not asserted

Release gate on `base_covid`, 8 threads. `Meta.parseall` on all four changed files: 0
error/incomplete nodes. Full include order including `demc.jl` loads and every name the edits
call resolves. `check_forwarding.jl` PASS (75 registry rows, 75 keys read). `check_dead_settings`
PASS. `check_version_sync` in sync, one declaration, three readers include it.

The FD route: **23 of 23 coordinates measured in 280 solves and 63 s**, against 600 for the
design. Richardson instability median **1.57e-05**, p90 9.03e-04. Achieved displacement median
**1.000** against a target of 1. Accepted rungs spread over 1–3, so the ladder is doing work
rather than defaulting. Deepening the ladder from 3 rungs to 4 cut the median instability 2.7×
(4.19e-05 → 1.57e-05) and is the shipped default.

`se(bound)` FD against the shipped 0.05 regression, per parameter: median ratio **0.29**, range
0.0003 (`b_U`) to 3.53 (`δ_S`). The direction is not uniform — the correct local geometry
tightens 18 of 23 coordinates and widens 5 — so this is not a rescaling of the old table.
Parameters with independent-signal fraction below 0.02: **3 of 23** under FD, against 5 at the
0.05 regression and 13 at the 0.01 regression.

### Settings

* **Added `MCMC_JAC_ALLOW_GAPS`**, default `false`. When `:fd` cannot measure every column the
  run raises by default. Set true to drop the unmeasured columns and report the rest as standard
  errors on the measured subspace — **conditional** on the dropped coordinates being known, hence
  SMALLER than unconditional ones, because the dropped coordinate's collinearity with the rest no
  longer inflates them. Printed as such, and not comparable to a fully measured window's column.
  This is what makes `crisis_fc` reportable at 17 of 18.
* **Added `MCMC_JAC_METHOD`**, default `:fd`. `:design` selects the previous least-squares route
  and is what every bundle before v35 used; it is kept for exactly that reason and because its
  `R²` is the diagnostic that condemns it. Registered in `SETTINGS_REGISTRY`
  (`code/smm/settings.jl:217`), which is the catalogue — `SETTINGS.md` states the policy for
  settings and does not enumerate them, so there is nothing to add there. `SETTINGS.md`'s
  "Adding a setting" steps 3 and 4 do not apply: like `MCMC_JAC_ONLY` this is read once at
  `MCMC_main.jl` top level and consumed in the same file rather than forwarded to a callee, which
  is why `check_forwarding` passes without a matching keyword argument.
* `local_design`'s `rel_step` is unchanged at 0.05 and is still never passed by the call site.
  It is now reached only under `:design`.

### Known gaps

* **`pinv` truncates 5 of 23 directions at the FD Jacobian, and this is new.** `cond(G̃)` is
  1.07e9 against 3.96e4 at the shipped design radius, so `cond(Ĵ) = cond(G̃)²` is past double
  precision and `pinv` keeps 18 of 23 directions — silently assigning zero variance to five
  parameter combinations. The spectrum is `3.5e7, 1.8e4, 1.4e4, …, 3.3e-02`, one direction
  dominating the next by 1944×. The run now prints the retained rank, but **`se_bound_diagonal`
  is unchanged and the truncation is not handled**; a truncated direction is an unidentified
  parameter wearing a finite standard error. This is the open decision this version hands over,
  and it reverses an earlier session's finding that `pinv` never bites — that finding was correct
  for the design Jacobian and is false for this one.
* **Three coordinates cannot reach the target band and are reported, not fixed.** `a_ℓ` 0.002,
  `b_ℓ` 8.113, `b_U` 7.500 against a target of 1. `b_U` and `b_ℓ` are at the `h_lo_frac = 1e-7`
  floor — the moments respond so sharply that one sampling sd of movement needs a smaller
  relative step than the floor allows. `a_ℓ` is bounded from above by the convergent domain: at
  the largest feasible step the moments move 0.002 sd, which is a fact about `a_ℓ` rather than a
  tuning failure. `a_ℓ` also has a single extrapolate and so no stability check.
* **Measured on `base_covid` only.** It was chosen because `base_fc`'s stored `θ̂` evaluates to
  `Q = Inf` above one thread — `converged_S` depends on the reduction order in
  `skilled.jl:236-237, 467-468` — so `base_fc` cannot be swept multithreaded. The `R²` pathology
  was consistent across all four windows at the shipped radius, so the diagnosis should transfer,
  but the per-parameter magnitudes above are window-specific and the other three are unmeasured.
* `check_structure.jl` reports 7 unbalanced files including `code/smm/settings.jl`. **Pre-existing
  and not caused by this version**: `settings.jl` is flagged identically with and without the new
  registry row, all seven parse with zero error nodes, and the gate flags its own
  `check_forwarding.jl`, so its depth counter is confused by strings and comments.
* No estimation was re-run and no transition was run. The FD route has not been exercised through
  `MCMC_main.jl` end to end — it was verified through a harness that reproduces the `JAC_ONLY`
  construction (bundle → `spec0` verbatim → `load_weight_matrix(cond_target=0.0)` →
  `jacobian_adaptive_fd` → `se_bound_diagonal`).

### Tried and rejected

* **Retuning `rel_step` to 0.01 instead of replacing the method.** Measured: it does fix the
  `R²` (median 0.992, none below 0.9). Rejected because it leaves the derivative a function of
  the search box and still applies one radius to coordinates whose sensitivities span four orders
  of magnitude. Kept as the `:design` route for reproducing pre-v35 bundles.
* **Restricting Ω to be block-diagonal across data sources** to tighten `se_bound`. Rejected on
  two grounds, one measured and one economic. Measured: the source partition is two blocks of 19
  — CPS Basic carries `ur_*`, `skilled_share`, the eight survivals, `ltu_share_S`, `theta_U/S`
  plus the four transition moments off the same longitudinal extract; ASEC carries the two mean
  wages, `emp_var`/`emp_cm3`, the ten percentiles, `wage_premium` and the two overlaps — so
  almost nothing would have been restricted and the available factor of `√blocks` would not have
  been approached. Economic, and the author's own objection: the correlation is between the
  *quantities*, not the collectors. The ten percentiles are order statistics of one distribution,
  `wage_premium` is a function of the two means, the survivals are nested tails, and `ur_U` sits
  in a flow identity with `jfr_U` and `sep_rate_U`. A source partition cuts across none of that,
  and there are no structural zeros in the true Ω to exploit.
* **Estimating Ω in any form — bootstrap, full sandwich, or efficient second step.** Ruled out by
  the author and recorded here so it is not re-proposed: the three sources differ in frequency
  and granularity so no joint resampling scheme exists, the moments are close to collinear so a
  real Ω would be near-singular, and some moments (`ee_rate_S` from J2J) are published aggregates
  with no microdata to resample. `se_bound` exists precisely because it needs only the diagonal.
* **Complex-step differentiation and automatic differentiation.** Complex-step is unavailable:
  the solver uses comparisons, `clamp` and an interpolation layer, so it is not complex-analytic.
  AD is the only genuinely step-free route and is blocked by two things beyond the engineering —
  the OJS softening in `grids.jl` is `C⁰` and not `C¹`, so at some θ the derivative does not
  exist, and the outer loops terminate on tolerance tests that are not differentiable. Recorded
  as a real option with a real obstacle, not as a dead end.

---

## v34.0.0 — 2026-09-13

**MAJOR: the training state is redefined as four-year undergraduate enrolment, and φ is
the nominal length of that programme.** Both the `training_share` target and the
completion rate change, so every estimated moment moves.

**Every stored `Q` is stale, including v33.0.0's** — re-estimate all four windows.
`SMMResult` and `SMMSpec` are unchanged and φ is read from `phi_calibration.csv` rather
than serialised, so **every stored bundle still deserialises** and the estimates remain
usable as warm starts. `output/estimates/` and `output/chains/` were hashed before and
after and are byte-identical (13 files, aggregate SHA-256
`ca86bdd9645b00ea2910aca4da8a2f1868eea668b34f2d09de19db4501acd6ad`).

### What the target is now

`training_share` = four-year undergraduate fall enrolment, all attendance, over the CPS
16–64 population, averaged over the window's Fall years. Was: total postsecondary
enrolment from the NSC workbook's `Overall` row — all institution levels, all attendance,
graduate students included — multiplied by an attrition wedge of 0.7813.

Three reasons, in the order they bind:

* **The universe was wrong for the model's skilled state.** Skilled is `EDUC ≥ 111`, a
  bachelor's or above. Two-year enrolment does not lead to it and graduate enrolment
  starts from it, so both counted students the training state cannot represent.
* **The attrition wedge double-counted.** A completion probability on the stock and a
  completion hazard φ discount the same attrition twice. The stationary identity also
  leaves no admissible room for one: inverting `skilled_share = (φ/ν)·t/(1−t)` at the
  observed skilled share wants a wedge of 1.008, 0.946, 1.090 and 1.183 across the four
  windows, three of the four above the ceiling of 1. The single wedge minimising mean
  |error| in φ/ν is 1.008 and the objective falls monotonically in f below it, so 1 is the
  boundary optimum on (0, 1] — 8.0% mean error against 7.6% at the unconstrained optimum
  and 26.6% at the old 0.7813. It is not the per-window optimum: crisis_fc alone wants
  0.946. The wedge is gone, with its two constants.
* **The level was unreachable.** The identity's coefficient φ/ν is fixed within a pair, and
  under the old target the model's value sat 24–86% away from what the target demanded in
  every window. It now sits within 0.9–16.3%, mean 8.0%. The training-share moment was
  asking for something the identity forbids and carrying that misfit into every other
  parameter through the criterion.

| window | old target | new target |
|---|---|---|
| base_fc | 0.071679 | 0.044317 |
| crisis_fc | 0.079689 | 0.050055 |
| base_covid | 0.075957 | 0.052498 |
| crisis_covid | 0.071743 | 0.052488 |

The FC pair still rises, +12.95%. The COVID pair is now flat, −0.02%, where the old target
fell 5.55%. The identity needs roughly +8.6% there, so the data no longer contradict the
model's direction — they are silent on it. That is a weaker limitation, not a solved one.

### φ

`φ = 1/48`, the nominal length of a bachelor's programme, from the single constant
`BACHELOR_PROGRAM_MONTHS`. Was an enrolment-weighted blend of NCES median time-to-degree
for bachelor's (49 months) and associate's (37 months), giving 45.0 months. The blend is
gone with the two-year universe. It also had a latent defect: its enrolment weights were
averaged across every IPEDS column in the NSC workbook, including the post-2016 rows where
that workbook's institution-sector series are transposed, which put 2-year enrolment at
4.76m against a real pre-break level near 6.5m.

1/φ is nominal rather than an observed mean time-to-degree on purpose: completion is the
state's only exit, so a measured mean would fold in part-time pacing, stop-out and
re-enrolment, all of which the model already represents as time spent in the state at this
same hazard.

### New raw input

`data/raw/nces/nces_4yr_undergrad_enrollment.csv` — NCES Digest of Education Statistics
2024, Table 303.70, four-year institutions, undergraduate total and full-time, fall
1970–2023. Source and URL are on every row. The NSC workbook is no longer a pipeline
input; its institution-sector rows are unusable after 2016, which is why the four-year
series cannot come from it. `RAW_NSC_DIR` and the `XLSX` dependency are removed from the
driver.

### Validation

* Gate (5a) now checks that each window's Fall-year coverage is complete, so a missing
  year in the external file cannot silently shrink a window's average. It replaces the
  check that the attrition wedge was constant across windows.
* Gate (5b) expects the FC pair to rise above 5% and the COVID pair to be flat within 2%.
  Flat is a magnitude statement, not a sign one: a ±0.02% move has no meaningful sign, and
  the identity asks for +8.6%.

### Purged

`data/derived/training_share_scale.csv` is no longer a required input in
`code/gates/check_repo.sh`, and no longer appears in the Reads headers of
`sampling_variances.jl` and `sigma.jl`. Nothing has produced it since κ was removed in
v17.0, so a fresh clone plus pipeline run failed that gate on a file it could never create.

## v33.0.0 — 2026-09-11

**MAJOR, two independent halves in one release: the stationary skilled distribution is now
the fixed point of its own flow balance, and the transition prices perfect foresight with
training as a decision rather than a hazard.** The solver half moves every estimated
moment; the transition half moves no moment but supersedes every transition path.

**Every stored `Q` is stale, including v32.0.0's** — re-estimate all four windows.
`SMMResult`, `SMMSpec` and `TransitionResult` are unchanged, so **every stored bundle
still deserialises** and the estimates remain usable as warm starts. `output/estimates/`
and `output/chains/` were hashed before and after both halves and are byte-identical
(18 files, aggregate SHA-256
`97c2a196f315dd2cfbd096920cfbf531293c4659811ae606b5a08662d3afacc2`).

This closes the defect class v31.0.0 opened — a distribution layer and another layer
disagreeing about one margin — at its fourth and last localised site. The sweep of every
remaining site in `code/solver` and `code/smm` is in *Known gaps* below, with each entry
marked fixed, identity, or deliberately left.

### Comparability

**Solver half: every stored `Q` and every fit table is superseded.** 33 to 35 of the 35
active moments move in every window, because the fix moves `θ_S` and free entry propagates
it. Measured travel at the stored points, v32.0.0 → v33.0.0: base_fc 1523.6554 →
1523.5000, crisis_fc 949.1226 → 948.3957, base_covid 2753.5195 → **2745.7797**,
crisis_covid 1553.7651 → **1550.3380**. Against the last comparable fit — the v29.x
numbers the bundles carry — base_fc travels 1025.4734 → 1523.5000.

**Transition half: every transition path computed before this version is superseded**,
including v30.0.0's and v31.0.0's. `TransitionPath` is internal and changed shape.

### What changed — solver (`code/solver/skilled.jl`)

- `skilled.jl:382` — **the poaching outflow hazard integrates cell masses.** The OJS
  outflow read the offer CDF at the grid node, `a_j = ν+λ+ξ+s_j f (1−Γ_o(p_j))`, while the
  poaching inflow that must offset it is built from the cell masses `γ_o,k wp_k`. It now
  reads `pre.tail_weights[j+1]`, the offer mass STRICTLY ABOVE cell `j`, and 0 at the top
  node. Strictly above is the object the inflow construction implies: `CumAlpha`/`CumBeta`
  are accumulated after cell `j` is priced, so a cell receives poaching inflow from the
  searchers below it, and the offsetting outflow must therefore be over destinations above
  it. With that spelling `∫inflow = ∫outflow` exactly and `û` sits on the `u_S` balance's
  own ratio. `Γ_o` is no longer read anywhere in `skilled.jl`; `pre.Γvals` survives for
  `equilibrium.jl:351`, where the argument is a worker's own node and the CDF is exact.
- **This is not the surplus's spelling, and the surplus's spelling would be worse.**
  `skilled.jl:296` and `:613` use `pre.tail_weights[j]`, the own-cell-INCLUSIVE tail,
  which counts the source cell in both directions. Measured at the stored base_fc
  estimate, mass-weighted `û` with free entry held fixed: own-cell tail 0.026305835611977735
  (+1.3888% off the balance's ratio), node read 0.026113747302782923 (+0.6484%), strictly
  above 0.025945516953090986 (−0.0000%). No new helper and no fourth spelling — the
  existing `tail_weights` array, offset by one, which is why `cdf_at_cutoff` is not used
  here: that reads a mass at an off-node CUTOFF, and this margin's argument is a cell
  boundary, not a cutoff.

### What changed — transition (`code/transition/`)

Landed by a separate pass; text below is that pass's, carried into this entry.

- `transition/transition_values.jl` (new) — **the backward pass solves the time-dependent
  HJBs.** The previous `_backward_pass!` ran the STATIONARY inner loops to their fixed
  point at each date's tightness, which solves `(r+ν)U = flow(a, θ(t))` with `∂_t U`
  absent: agents priced today's tightness as permanent. Every value on the path now solves
  `V(t) = [flow(θ(t)) + V(t+Δ)/Δ]/(ρ_V + 1/Δ)` with `ρ_V` read off that object's own
  stationary HJB — `r+ν` for `U^search` and `U_S^(0)`, `r+φ+ν` for `T`, `r+ν+f_U` for
  `U_S^(1)`, `r+ν+λ_j+ξ_j` for the surpluses, plus `f_S(1−Γ_o(p))` on the OJS branch. The
  flow terms are the UNREDUCED ones: the stationary surplus equations carry `−(r+ν)U` from
  a substitution of the stationary unemployment HJB that is wrong by `∂_t U` off steady
  state. The maximisations stay pointwise at each date.
- `transition_params.jl` — the path stores the RAW surpluses `SU`, `S0`, `S1` in place of
  `E0/E1/J0/J1`. The raw surplus is what the recursion prices and its zero crossing is the
  reservation; the `ω`-weighted firm values cannot be inverted where `ω = 0`.
  `_install_skilled_firm_values!` rebuilds `J_S^0`, `J_S^1` for free entry by the same Nash
  split `skilled_inner_loop!` applies. Also adds `US1` and the pre-switch state
  `τT0`/`uU0`/`tU0`.
- `transition_solver.jl:_frontier_sweep!` (new) — **training is an indicator, not a
  hazard.** `∂_t t = τ u_U − (φ+ν)t` read `τ` as a rate of one per month, so a training
  cell carried `u_U > 0` while `solve_stationary_unskilled!` holds it at zero. Cells the
  frontier has just enclosed now transfer their ENTIRE unskilled-unemployment stock into
  training, by the covered fraction `(τ(n) − τ(n−1))/(1 − τ(n−1))`, at the date the
  boundary crosses them. Entry into training is a SPLIT of each inflow in the proportions
  `τ : 1−τ`, not an outflow rate.
- `transition_params.jl` — `damp` default `0.30` → `0.15`. With the genuine backward pass
  the skilled free-entry map is stiffer; at 0.30 `base_fc → crisis_fc` limit-cycles in
  `θ_S` at `‖Δθ‖∞ ≈ 5.7e-04` after 150 iterations.
- `tools/transition_audit.jl` — per-quantity drift table against `z0` on a null run and
  `z1` otherwise, with reference rates rebuilt from the stationary masses by
  `_build_result`'s own arithmetic.

### Measured, not asserted — solver

Stored estimates, `Nx = Np_U = Np_S = 120`, 6 threads, one objective evaluation per window
per version.

| window | θ_U before | θ_U after | θ_S before | θ_S after | ur_S before | ur_S after | ur_S target | Q before | Q after |
|---|---|---|---|---|---|---|---|---|---|
| base_fc | 0.45897048401802326 | 0.45896825394335578 | 1.2128121705475032 | 1.2126789659683888 | 0.026113747 | 0.025946905 | 0.026013317 | 1523.6554 | 1523.5000 |
| crisis_fc | 0.22588433906871835 | 0.22588281253582121 | 0.64642722560212684 | 0.64632443509839832 | 0.042320797 | 0.042199729 | 0.041720286 | 949.1226 | 948.3957 |
| base_covid | 0.81514533242961706 | 0.81514550139395736 | 1.8369753265551172 | 1.8370068984604659 | 0.027905561 | 0.027824376 | 0.024671709 | 2753.5195 | 2745.7797 |
| crisis_covid | 1.0513813794931595 | 1.0513820024629648 | 2.3217879349056889 | 2.3218906158018688 | 0.041769975 | 0.041703461 | 0.037051123 | 1553.7651 | 1550.3380 |

`θ_S` moves in the fourth decimal (base_fc `−1.332e-04`, crisis_fc `−1.028e-04`,
base_covid `+3.157e-05`, crisis_covid `+1.027e-04`) — four to eleven orders of magnitude
above the thread-nondeterminism floor, unlike the v32.0.0 free-entry fix. 33 of 35 moments
move at base_fc and crisis_fc, 35 of 35 in the COVID pair. `ur_S` falls in every window and
improves in three: base_fc from +0.39% to −0.26% of target, crisis_fc +1.44% → +1.15%,
base_covid +13.11% → +12.78%, crisis_covid +12.74% → +12.56%. `Q` falls in all four.

**Acceptance against the flow balance.** At the stored base_fc estimate with free entry
held fixed, mass-weighted `û` is 0.025945516953090986 against the `u_S` balance's own
ratio `(ξ+λΓ_s(p*)+ν)/(ξ+λΓ_s(p*)+ν+f(1−Γ_o(p*)))` of 0.025945516953091035 — agreement to
4.9e-17, i.e. `−0.0000%`, reproducing the 0.02594552 the transition pass measured for the
forward law's fixed point. The shipped node read gave 0.026113747302782923, `+0.6484%`
above the ratio, reproducing its 0.02611375. In FULL equilibrium, where free entry
re-converges to the new `θ_S`, base_fc `ur_S` is 0.025946905: the residual 1.4e-06 against
the fixed-`θ` figure is the tightness response, not a remaining gap.

The two tails differ materially, so this was never going to be empty: against
`1 − Γ_o(p_j)`, the own-cell tail is up to `+1.613e-02` (RMS `4.562e-03`) and the
strictly-above tail up to `−3.651e-02` (RMS `5.290e-03`).

### Measured, not asserted — transition

Null shock, `base_fc` against itself, `Nx = 120`, `Nt = 241`, terminal deviation from the
steady state the path starts at:

| quantity | v30 value step | this version |
|---|---:|---:|
| `u_U` total | −7.16% | −0.00% |
| `θ_U` | +10.58% | +0.00% |
| `ur_U` | −7.48% | −0.00% |
| `t` total | −0.98% | +0.00% |
| `m_S` total | −0.66% | −0.00% |
| `u_S` total | −1.32% | −0.61% |
| `ur_S` | −0.66% | −0.61% |
| `θ_S` | −0.50% | −0.08% |
| `training_share` | −0.98% | +0.00% |

Outer iterations 15 → 7; terminal distribution gap to `z0` `1.900e-02` → `2.922e-04`;
triangle-excess peak `4.109e-02` → `5.944e-04`. Terminal condition: the seeded date `Nt`
reproduces `z1` to `0.000e+00` on every 1D value and cutoff and to `1.7e-11` on the
rebuilt `J_S`; one value step at `z1`'s own tightness and the path's step size reproduces
it at date `Nt−1` to `5.1e-14` (`U^search`) and `3.8e-08` (`p*_U`), in 9 within-date
passes to a `8.8e-08` residual. Mass conservation: population `1.000000000000` at every
date; over 3,456,000 cell-steps the worst untrained-employment overshoot is `2.3e-15` of
its own cell mass. Real pair `base_fc → crisis_fc`: 52 outer iterations
(`‖Δθ‖∞ = 8.627e-05`); terminal gap to `z1` `3.757e-03` (was `1.265e-02`).

The `−0.61%` residual `u_S` drift in that table IS the defect the solver half fixes, so
that row is superseded by this release rather than describing it.

### Gates

12/12 `.jls` in `output/estimates/` deserialise, schema 1 and 2, versions 19.5.0–29.1.0. A
fresh spec written and read back has `spec`, `result` and `theta_opt` identical
field-for-field and reproduces `Q` to 2.501e-12 against a 1.5e-03 tolerance.
`check_forwarding.jl` PASS. Every changed solver file parses with zero
`:error`/`:incomplete` nodes. Not run: `check_dead_settings.jl`, `check_output_tree.jl`,
`code/smoke/run_smoke.sh` (it writes generated drivers into `code/smm/` and launches with
`--threads auto`), the MCMC path, and — on the solver half — the transition smoke gate and
audit.

### Settings

`TransitionParams.damp` default `0.30` → `0.15`; `TR_DAMP` in `transition_main.jl` follows,
and `TA_DAMP`'s default is now the `TransitionParams` default rather than a repeated
literal. **`SETTINGS.md` still needs the corresponding row** — not added here, that file is
outside this pass's ownership. No solver-side setting added, retired or re-defaulted.

### Known gaps

**The cutoff/node sweep, closed.** Every site in `code/solver` and `code/smm` where a
quantity is evaluated AT a node or cutoff while the corresponding distribution integrates
ACROSS one:

1. `equilibrium.jl:263-264` reservation CDFs `Γ_o(p*_S)`, `Γ_s(p*_S)` — **fixed** v31.0.0,
   `cdf_at_cutoff`.
2. `equilibrium.jl:147` `I_full`, unemployed search option — **fixed** v31.0.0, soft index.
3. `equilibrium.jl:351` poaching hazard in `ee_rate_S` — **fixed** v32.0.0; reads
   `Γ_o(p_j)` at the worker's own node, which is where `Γvals` is exact and where
   `solve_stationary_skilled!` applies the same hazard.
4. `skilled.jl:491,496,518` firm-value tails in `compute_Jbar_skilled` — **fixed**
   v32.0.0, `j0_soft`. Numerically empty at all four stored estimates (the straddling cell
   carries an integrand that vanishes at the reservation).
5. `skilled.jl:382` poaching outflow hazard in `solve_stationary_skilled!` — **fixed
   here**.
6. `skilled.jl:296,613` the SURPLUS's poaching hazard, `pre.tail_weights[j]` — **deliberately
   left, and now the largest remaining instance.** The value layer prices the OJS outflow
   at the own-cell-inclusive tail while the distribution applies the strictly-above tail;
   the two differ by up to one cell's offer mass. It is not a one-term change: the hazard
   in the denominator is paired with `tail_Emax_j = tailEo[max(j, j0_soft)]`, the
   destination-value tail over the SAME index set, so moving one requires moving the
   other, and that pair moves `p*_S`, `θ_S` and every moment. Whether the own-cell
   destination — a within-cell move worth zero surplus gain — belongs in the value
   equation at all is a specification question, not a numerics one.
7. `skilled.jl:49,55` `Γvals`/`Γs_vals = cdf.(dist, sg.p)` — **identity**, tabulation at
   nodes, with exactly one remaining reader (site 3) whose argument is a node.
8. `skilled.jl:116,353,477,514,595` `pcut_index` — **identity**, loop bounds and
   crossing-search starts only, as its docstring now requires.
9. `unskilled.jl:280-282` and `equilibrium.jl:287,301,388` — **identity.** The unskilled
   offer CDF is analytic, `G(p) = p^{α_U}`, and both layers evaluate it at `p*_U` exactly;
   the hard hiring gate `1{p*_U < 1−1e-10}` is character-for-character the same in both.
   The unskilled block never had this defect class.
10. `unskilled.jl:83-97` the unskilled SURPLUS's tail integrals — **deliberately left.**
    They integrate `∫_{p*}^1 dG` as `Σ_j ω_j wG_j` with Gauss–Legendre weights times the
    POINTWISE density, while the distribution uses the analytic `1 − (p*_U)^{α_U}`. At the
    stored base_fc `α_U = 4.13299601` the total mass is exact to `7.994e-15`, so this is
    not a quadrature failure; the gap is the straddling cell's linear-coverage
    approximation, rising with the cutoff: `8.461e-07` at `p* = 0.05`, `2.968e-03` at
    `0.5`, `1.161e-02` at `0.9` (`+0.0001%`, `+0.3147%`, `+3.2892%` relative). Whether it
    binds depends on where `p*_U` sits at each estimate, **which was not extracted** — the
    skilled block's exact-CDF cell masses (`build_cell_mass_density`) are the fix if it
    does.
11. `_soft_oj_weight` and `_soft_weight` return the covered fraction of the interval
    `[p_j, p_{j+1}]` while multiplying the quadrature weight `wp_j`, which is not that
    interval's length — **deliberately left**, shared by every layer, documented in
    `skilled.jl` as shipped for continuity rather than accuracy. An exact version weights
    by the cell's mass fraction under `dΓ`.
12. `code/smm/` — **no site.** Nothing in the SMM layer evaluates a CDF or a cutoff; its
    only `Γ` occurrences are the parameter names `a_Γ`, `b_Γ` and their ASCII key maps.

**`_forward_skilled_pdist!` must now take the same one-term change, and has not.**
`transition_solver.jl:344` deliberately mirrors the node read so that `e_frac` is a fixed
point of the forward pass. After the solver fix that mirroring is what breaks the
property: the new stationary `ê` is the strictly-above object and the forward pass still
carries `1 − Γ_o(p_j)`, so the null run should now drift the mirror of the `−0.6442%` it
was drifting before. **Not verified — no transition run was made in this pass, and
`code/transition/` is frozen to this pass.** It is a one-line change on line 344 to
`tail_weights[jp+1]`, and it must land before any transition path is regenerated.

Carried forward from the transition pass, unchanged by this release: fractional `τ` and
the split of a partially-covered cell's separations have no rule in the notes; the notes
write no law for the skilled quality mix `ê_S`; arrival at `z1` is diagnosed, not
enforced; `transition_main.jl` is not reproducible from the pinned environment (`using
Plots`, `using LaTeXStrings`, in neither `Project.toml` nor `Manifest.toml`);
`solve_transition` consumes its models.

### Tried and rejected

- **Spelling the distribution's hazard as the surplus's `pre.tail_weights[j]`**, on the
  brief's reading that the surplus equations already use "the cell-mass tail". Measured:
  `û` = 0.026305835611977735, `+1.3888%` off the flow balance's ratio — worse than the
  node read it would replace, because the own-cell-inclusive tail counts the source cell as
  a destination while the inflow does not. The two spellings are different objects and the
  distribution needs the strictly-above one.
- **Adding a `tail_weights_up` field to `SkilledPrecomp`** for the shifted tail. One site
  reads it; `pre.tail_weights[j+1]` with the top-node guard says the same thing without a
  new name to keep in sync, and `SkilledPrecomp` is reachable from every solver file.

## v32.0.0 — 2026-09-11

**MAJOR: `ee_rate_S` was pricing every job-to-job mover at the marginal searcher's moving
rate, and free entry was dropping the cell straddling the reservation.** Two sites, the
same defect class v31.0.0 opened — a moment layer and a distribution layer disagreeing
about one margin — and the `ee_rate_S` one is the more serious of the two because it is
about WHICH object the moment is, not how precisely it is read. **Every stored `Q` is now
stale**, including the ones v31.0.0 produced: re-estimate all four windows from these
bundles as warm starts.

With `sep_rate_S` (v31.0.0) and `ee_rate_S` (here), both skilled hazard moments have now
been rebuilt to match the law of motion the stationary distribution obeys.

### Comparability

**Every stored `Q` and every fit table is superseded; the parameter vectors are
unchanged and still load.** Measured travel at the stored points, v31.0.0 → v32.0.0:
base_fc 1524.3036 → 1523.6554, crisis_fc 950.5542 → 949.1226, base_covid 2753.4906 →
2753.5195, crisis_covid 1553.9470 → 1553.7651. Against the LAST COMPARABLE fit — the
v29.x/v30.0.0 numbers the bundles carry — base_fc travels 1025.4734 → 1523.6554.

No struct changed and no bundle format changed. All 12 `.jls` in `output/estimates/`
deserialise, a fresh spec round-trips, and both protected directories are byte-identical
to their pre-work hash.

### What changed

- `equilibrium.jl:349` — **the poaching flow reads the offer CDF at the worker's own
  quality, not at the OJS cutoff.** `p^oj` is the PARTICIPATION margin: whether a worker
  of quality `p` searches on the job at all, which the soft weight `s_j` already carries.
  Given that she searches, she moves only if the offer beats what she holds, so the flow
  is `s_j · κ_S · (1 − Γ_o(p_j))` at her own node — where the CDF is tabulated exactly and
  where `solve_stationary_skilled!:363` applies the same hazard to the same mass. The old
  `κ_S(1 − Γ_o(p^oj))` charged a worker deep below the cutoff the marginal searcher's
  moving rate, so the moment and the distribution it was computed from obeyed different
  poaching hazards. `cdf_at_cutoff` is no longer called at `p^oj`: v31.0.0 made that read
  exact, and this makes it the wrong object to read at all. The wage-step half of the loop
  needed no change — it already averaged destinations over `p′ ≥ p_j`.
- `skilled.jl:445` `compute_Jbar_skilled` — the firm-value tail is read at `j0_soft =
  max(pcut_index − 1, 1)`, the index `skilled_inner_loop!` and `solve_stationary_skilled!`
  both use, in the main loop (`tailJ[j0_soft]`, `tailJ[max(j, j0_soft)]`) and in the
  early-iteration fallback (`for j in Np:-1:j0_soft`). `J^0`/`J^1` carry the reservation
  coverage weight, so the hard node dropped the straddling cell from free entry while the
  block that priced it kept it. Deferred at v31.0.0 on sequencing grounds; folded in here
  so the author re-estimates once rather than twice.
- `skilled.jl:490` — dead local `colmass` removed from the fallback branch (computed, never
  read), and `frac0` folded into `ell`, which was a rename of it.

### Measured, not asserted

Stored estimates, `Nx = Np_U = Np_S = 120`, 6 threads, one objective evaluation per window
per version.

**Tightness and objective, v31.0.0 → v32.0.0:**

| window | θ_U before | θ_U after | θ_S before | θ_S after | Q before | Q after |
|---|---|---|---|---|---|---|
| base_fc | 0.45897048401802443 | 0.45897048401802326 | 1.2128121705475032 | 1.2128121705475032 | 1524.3036 | 1523.6554 |
| crisis_fc | 0.22588433906873873 | 0.22588433906871835 | 0.64642722560212706 | 0.64642722560212684 | 950.5542 | 949.1226 |
| base_covid | 0.81514533242961518 | 0.81514533242961706 | 1.8369753265551141 | 1.8369753265551172 | 2753.4906 | 2753.5195 |
| crisis_covid | 1.0513813794931612 | 1.0513813794931595 | 2.3217879349056934 | 2.3217879349056889 | 1553.9470 | 1553.7651 |

**The free-entry fix is a measured no-op at all four stored points, including the COVID
windows where `p*_S > 0`.** `θ_S` moves by at most 4.441e-15 and `θ_U` by at most
2.037e-14, inside the same-code thread-nondeterminism floor (up to 1.271e-14 on `θ_U`
between two runs of identical code, v31.0.0 entry). `ee_rate_S` is the ONLY moment that
moved by more than 1e-10 in any window; the largest change among the other 34 is 2.040e-14,
and the whole `Q` travel equals the `ee_rate_S` contribution to the digit in all four
windows. So the entire travel above is fix 2.

The reason is structural, not luck. At base_covid the 19 abilities with interior `p*_S` are
`k = 14…32`, the bottom of the `aS` grid: the trained column mass `mcol0` is exactly 0 for
`k ≤ 26` and at most 1.527e-04 for `k = 27…32`, and the relative change in the firm-value
tail from moving the read index is **exactly 0.000e+00 at every one of the 19** — because
`p*_S` is defined as the zero of `S^max`, so the straddling cell the hard index dropped
carries an integrand that vanishes there. The site was a genuine inconsistency with no
numerical content at any point tested. crisis_covid was not probed ability-by-ability.

**`ee_rate_S`, v31.0.0 → v32.0.0, against target:**

| window | before | after | target | dev before | dev after | ΔQ |
|---|---|---|---|---|---|---|
| base_fc | 0.0075260255 | 0.0081359713 | 0.0079692324 | −5.56% | +2.09% | −0.6482 |
| crisis_fc | 0.0041060048 | 0.0044287116 | 0.0061497279 | −33.23% | −27.99% | −1.4316 |
| base_covid | 0.0083715571 | 0.0086553652 | 0.0085082638 | −1.61% | +1.73% | +0.0289 |
| crisis_covid | 0.0077580433 | 0.0079682595 | 0.0091506665 | −15.22% | −12.92% | −0.1819 |

The correction is upward everywhere, as the economics requires: a worker below the cutoff
was being charged the cutoff's acceptance probability, which is a lower moving rate than
her own. Three of four windows fit better; base_covid crosses the target from −1.61% to
+1.73% and its `Q` contribution rises slightly.

**Reservation structure at the stored estimates**, which is what the dead-margin statement
rests on. The FC pair has `p*_S = 0` at all 120 abilities. The COVID pair does not:

| window | `p*_S = 0` | interior | `p*_S = 1` | emp. share at 0 | interior | at 1 |
|---|---|---|---|---|---|---|
| base_covid | 88 | 19 | 13 | 0.99941614 | 0.00058386 | 0.0 |
| crisis_covid | 89 | 17 | 14 | 0.99832443 | 0.00167557 | 0.0 |

So **the dead-margin statement does not survive as "zero at every ability that carries
skilled employment"** — 19 and 17 abilities have a strictly interior reservation and they
do carry employment, 0.058% and 0.168% of the skilled total (largest single cell 4.2e-04
and 1.0e-03 of it). Every ability at the opposite corner `p*_S = 1` carries exactly zero
employment, which is mechanical: a worker who accepts nothing is never employed. The
defensible statement is: the endogenous skilled margin is exactly dead in the FC pair, and
alive but carrying under 0.2% of skilled employment in the COVID pair, where
`sep_rate_S − ξ_S` is 2.861e-06 and 3.633e-06.

**Round-trip gate:** 12/12 bundles load, schema 1 and 2, versions 19.5.0–29.1.0. Fresh
spec written and read back with `spec`, `result` and `theta_opt` identical field-for-field
and `Q` reproduced to 1.864e-11 against a 1.5e-03 tolerance. `check_forwarding.jl` PASS.
Every changed file parses with zero `:error`/`:incomplete` nodes.
`output/estimates` + `output/chains`, 18 files, aggregate SHA-256
`97c2a196f315dd2cfbd096920cfbf531293c4659811ae606b5a08662d3afacc2` before and after —
unchanged since v30.0.0.

### Settings

None added, retired or re-defaulted.

### Known gaps

- **The cutoff audit is now clean, with two shared approximations left standing.** Every
  cutoff-sensitive quantity in the moment layer uses the same object and the same coverage
  convention as the layer that builds the distribution: the reservation CDFs
  (`equilibrium.jl:263-264`), the OJS participation weight (`:212`, `:349`), the firm-value
  tails (`:147`, `skilled.jl:472`), and the poaching hazard (`:351`). What remains is
  shared by BOTH layers, so it is a model-level convention rather than a layer
  disagreement: (a) `_soft_oj_weight` returns the covered fraction of the interval
  `[p_j, p_{j+1}]` while multiplying the Gauss–Legendre weight `wp_j`, which is not that
  interval's length — documented in `skilled.jl` as shipped for continuity, not accuracy,
  and an exact version would weight by the cell's mass fraction under `dΓ`; (b) the
  unskilled participation corner is a hard switch, `f_hire = (p* < 1 − 1e-10) ? f_U : 0`,
  in `solve_stationary_unskilled!` and in `δU_unemp` identically. The unskilled block has
  no analogue of the v31.0.0 defect: its offer CDF is analytic, `G(p) = p^{α_U}`, and both
  layers evaluate it at `p*_U` exactly.
- The two fixes were not separated by re-running each alone; the decomposition above is
  from the moment vectors (only `ee_rate_S` moved, and `ΔQ` equals its contribution
  change), not from four extra solves.
- `code/smoke/run_smoke.sh` was not run: it writes generated drivers into `code/smm/` and
  launches with `--threads auto`. The spec-construction and objective path it covers was
  exercised by the round-trip gate instead.

### Tried and rejected

- **Spelling the free-entry reservation index through `cdf_at_cutoff`.** It reads a MASS,
  and free entry needs a Γ-weighted tail of firm VALUES, `∫_{p*}^1 [s J^1 + (1−s) J^0]
  dΓ_o`. Since `J^0`/`J^1` already carry the coverage weight `ω`, that integral is exactly
  `tailJ[j0_soft]`, so `j0_soft` is the existing spelling of the same convention rather
  than a third one.
- **Leaving `compute_Jbar_skilled` for a later pass** (the v31.0.0 position). The
  sequencing argument protects attribution, which nobody needs here: all four windows are
  re-estimated in one pass. Measured after folding it in, the deferral would have cost
  nothing either — the site is numerically empty at every point tested — but that was not
  knowable before the measurement, and the consistency is worth having.

## v31.0.0 — 2026-09-11

**MAJOR: 35% of the model's skilled separation rate was a grid artifact.** The moment
layer read the skilled reservation CDFs at a grid NODE — `Γvals[pcut_index(p, p*)]` —
instead of integrating the cell masses at the cutoff. At the shipped base_fc estimate
`p*_S = 0` at every ability, so the read landed on node 1 where `Γ_s = 0.017`, and the
endogenous separation term `λ_S·Γ_s(p*_S)` came out at 0.00221031 — 54.8% of `ξ_S` and
35.4% of the model's `sep_rate_S` — where the model says it is exactly zero. `sep_rate_S`
is a targeted moment, so every stored estimate was fitted against a moment the code was
computing wrong. The same defect was repaired in the transition solver at v30.0.0 and
deliberately left standing here; this closes the last site. A separate transition change
is landing in its own pass.

### Comparability

**All four stored estimates are superseded as FITS and remain valid WARM STARTS.** The
parameter vectors are unchanged and every bundle still loads; what changes is the moment
vector they are scored against, so `Q` at the stored points and the fit tables built from
them are stale. Re-estimate all four windows. Measured travel at base_fc: `Q` 1025.4734 →
1524.3036.

No struct changed and no bundle format changed, so nothing needs migrating: all 12 `.jls`
files in `output/estimates/` deserialise under this version, and a freshly built spec
round-trips. `output/estimates/` and `output/chains/` were hashed before and after the
work and are byte-identical — nothing was re-written.

### What changed

- `grids.jl:360` — `cdf_at_cutoff(γcells, pgrid, wp, cutoff)` (new). The CDF of a
  cell-mass density at an off-node cutoff: the mass of the cells below it, the straddling
  cell entering at its covered fraction. This is the integration convention
  `solve_stationary_skilled!` already uses, named once so the moment layer cannot drift
  from it again. Written as the mass BELOW rather than `1 − ∫ω dΓ` so a dead margin is
  exactly 0.0 whatever the cell masses total.
- `grids.jl:344` — `pcut_index`'s docstring now says it is for a LOOP BOUND only, and
  points at `cdf_at_cutoff` for a CDF read. The defect class was reachable because the
  helper looked like a way to evaluate `Γ` at a cutoff.
- `equilibrium.jl:263-264` — `Γo_pstarS` / `Γs_pstarS` integrate the cell masses. These
  feed `sep_rate_S`, `wchg_rate_S`, `jfr_S` and the skilled duration survivors.
- `equilibrium.jl:327` — the offer mass below the OJS cutoff, `Γ_o(p^oj)`, integrates the
  cell masses. It feeds `ee_rate_S`. The node read never fell below `Γ_o(p_1) = 0.0161`
  however low `p^oj` went, and stepped by a whole cell's offer mass as the cutoff crossed
  a node.
- `equilibrium.jl:147` — `I_full` reads the firm-value tail from one cell BELOW the
  reservation node, the index `skilled_inner_loop!` forms `I_S` at, so the cell straddling
  `p*_S` contributes its covered fraction. Exactly identity wherever `p*_S = 0`.

### Measured, not asserted

Stored base_fc estimate, `Nx = Np_U = Np_S = 120`, 6 threads. The pre-change run reproduced
the bundle's own `Q = 1025.473440` to ten digits, so the before/after comparison is against
the shipped number and not a re-solve artifact.

- Dead margin, acceptance (a): `Γ_s(p*_S)` integrated `= 0` and the employed-weighted
  endogenous term `λ_S·Σ Γ_s(p*_S)·e_S / e_S = 0` — exactly, all 120 abilities. The node
  read gave 0.017075822873104243 and 0.0022103063982921627.
- `sep_rate_S`, acceptance (b): 0.0040369596353730473 against `ξ_S =`
  0.0040369596353730473, difference 0.000e+00. Before: 0.00624726603366521 against a data
  target of 0.0062410537266473593, i.e. the fit was 0.1% off on 65% real and 35% artifact.
- `Q`: 1025.4734395410 → 1524.3035663077 over 35 active moments (8 of the 43 in
  `MOMENT_NAMES` carry zero weight). `sep_rate_S` alone goes from 0.004 to 493.06 of it —
  the model now says skilled separation is `ξ_S`, and `ξ_S` is 35% below the target the
  artifact was covering.
- Acceptance (d): 8 moments moved, all skilled and all consumers of the three fixed reads
  — `sep_rate_S` (−0.00221031), `wchg_rate_S` (+0.00221031, the same redraw event's
  complement, now exactly `λ_S`), `jfr_S` (+0.00440104), `usurv5/14/27/53_S` (−0.0037,
  −0.0059, −0.0050, −0.0019), `ee_rate_S` (+6.78e-05). The other 27 moved by at most
  1.271e-14; two runs of the IDENTICAL post-change code differ by up to 3.569e-14 on 21 of
  35 moments, so that residual is the solve's thread-scheduling nondeterminism and not the
  change. No unskilled moment, wage moment or stock moment moved: the unskilled CDF is
  analytic (`G(p) = p^{α_U}`, evaluated at `p*_U` exactly) and the stationary distribution
  is computed upstream of the moment layer.
- Bundle serialisation: 12/12 `.jls` in `output/estimates/` load (`read_bundle`), spanning
  schema 1 and 2 and versions 19.5.0 through 29.1.0. A spec rebuilt through
  `build_smm_spec` and written with `write_bundle` reads back with `spec`, `result` and
  `theta_opt` identical field-for-field, and reproduces its objective to 6.937e-10 against
  a 1.5e-03 tolerance (the residual is the same thread nondeterminism, amplified by
  `W_kk ≈ 1e8` on `sep_rate_S`).
- `p*_S` at the stored estimates, which is what makes the margin dead: base_fc and
  crisis_fc have `p*_S = 0` at all 120 abilities; **base_covid and crisis_covid do not** —
  88/120 and 89/120 zeros, with a corner at `p*_S = 1` elsewhere. So the endogenous
  skilled margin is alive in the COVID windows, and `sep_rate_S − ξ_S` there is 2.861e-06
  and 3.633e-06 after the change rather than 0.
- Post-change `Q` at the other three stored points: crisis_fc 950.5542, base_covid
  2753.4906, crisis_covid 1553.9470 (their bundles carry 599.3787, 2753.3745, 1553.8087).
  Only base_fc was run before the change, so only its travel is attributable; the pattern
  is consistent with the two `p*_S = 0` windows taking the whole correction and the two
  COVID windows barely moving, but that is inference, not measurement.

### Settings

None added, retired or re-defaulted.

### Known gaps

- **`compute_Jbar_skilled` reads the firm-value tail at the hard cutoff node**
  (`skilled.jl:472`, `skilled.jl:500`), dropping the straddling cell's covered fraction —
  the same class as `equilibrium.jl:147` and not fixed here. It is EXACTLY identity
  wherever `p*_S = 0`, which covers base_fc and crisis_fc but not the COVID windows. It
  sits inside free entry, so changing it moves `θ_S` and therefore every moment: folding
  it into this bump would confound the re-estimation with a second change. Sequence it
  deliberately.
- **`ee_rate_S` may be reading the wrong object, not just reading it wrongly.**
  `equilibrium.jl:327` prices poaching at `κ_S(1 − Γ_o(p^oj))` — the acceptance
  probability at the OJS SEARCH cutoff — while `solve_stationary_skilled!` prices the same
  outflow at `κ_S(1 − Γ_o(p_j))`, the CDF at the worker's OWN current quality
  (`skilled.jl:363`). The stationary distribution the moment is weighted by therefore
  obeys a different poaching hazard than the moment reports. This bump only made the
  cutoff read exact; which object `ee_rate_S` should use is a specification question.
- `sep_rate_S − ξ_S` is exactly 0.0 at base_fc but −2.602e-18 at crisis_fc. That is the
  employed-weighted mean's own rounding (`dot(δ_S, e_S)/agg_e_S` with `agg_e_S` summed in
  a different order), not a surviving endogenous term: the integrated `Γ_s(p*_S)` is
  exactly 0 at every ability in both windows.
- Not verified: the pre-change moment vector at crisis_fc, base_covid and crisis_covid,
  and hence the per-window decomposition of their `Q` travel. Only base_fc was run under
  both versions.

### Tried and rejected

- **`1 − ∫ω dΓ_s` as the dead-margin form**, mirroring the transition solver's
  `_skilled_margin_masses`. Measured at the base_fc grid: the offer and shock cell masses
  total exactly 1.0, so this form also returns exactly 0.0 there and the two are
  indistinguishable at this grid. `cdf_at_cutoff` integrates the mass BELOW the cutoff
  anyway, because that is 0.0 whatever the totals round to, and a dead margin must not
  depend on a telescoping sum landing on 1.0 exactly.
- **Defining the helper as a Model-level function beside `solve_stationary_skilled!`**, or
  reusing the transition solver's private `_skilled_margin_masses`. Rejected: the same
  name and signature in two included files silently overwrites one method with the other,
  and the transition layer is not a dependency of the solver library. The primitive is a
  grid/measure operation, so it belongs in `grids.jl` next to `build_cell_mass_density` and
  `pcut_index`, where both layers can reach it.

## v30.0.0 — 2026-09-11

**MAJOR: the transition path was not solving the model it claims to solve.** Six defects
in `transition/transition_solver.jl`, each independently sufficient to invalidate a
transition figure. The worst placed the training frontier off the wrong outside option,
so the education margin — the mechanism the paper is about — was wrong at every interior
date. No estimate, moment or `Q` moves: nothing outside `code/transition/` computes with
these routines. But every transition number does, so the theorist has to be told.

### Comparability

**Every transition path computed before this version is superseded.** No stored path is
affected, because none exists in a readable format: `output/transition/` holds one
`transition_fc_fullW.jld2` from April, and the codebase dropped JLD2 before the current
`.jls` convention. Estimates, chains, moments and `Q` are untouched and remain
comparable across the bump — `output/estimates/` and `output/chains/` were hashed before
and after the work and are byte-identical.

`TransitionResult` gained four fields (appended). A bundle written by v29.1.0 would
raise `EOFError`; there is none to migrate.

### What changed

- `transition_solver.jl:_backward_pass!` — **the training value read the wrong outside
  option.** `unskilled_inner_loop!` was called with `US_in = uc.Usearch`, so
  `T(aS) = (b_T + φ·U_S(aS))/(r+φ+ν)` was built from the UNSKILLED search value instead of
  the skilled unemployment value, putting `U^search` on both sides of the training
  comparison. `solver.jl:112` passes `sc.U` for exactly this reason. Measured: the
  frontier floor was pinned near 0.3015 at every interior date and then jumped
  discontinuously to 0.2711 at the terminal node, which is where the defect was visible.
  Now `US_in = @view path.US[:, n]`, the skilled value the same date's skilled block just
  produced.
- `transition_solver.jl:_skilled_margin_masses` (new) — **a corner reservation was
  generating separations.** The skilled acceptance and destruction margins were read as
  `Γvals[pcut_index(p, p*)]`, the CDF at a grid NODE. At these estimates `p*_S = 0` at
  every ability, so the read lands on node 1 where `Γ_s = 0.0171` and `Γ_o = 0.0161`,
  inventing an endogenous separation hazard `λ_S·0.0171` — 55% of `ξ_S` — and shaving 1.6%
  off the hire rate. Both margins are now cell-mass integrals `∫ω dΓ` from the soft
  cutoff, which is what `solve_stationary_skilled!` integrates.
- `transition_solver.jl:_forward_skilled_masses!` — the destruction hazard read the OFFER
  CDF where the model uses the SHOCK CDF. They coincide only at `δ = 1`; the estimates
  have `δ = 0.859` (base_fc) and `0.828` (crisis_fc).
- `transition_solver.jl:_forward_skilled_pdist!` — did not implement `eq:eSbalance`. The
  `λ_S` redraw inflow used the offer density and was fed only by mass BELOW `p`, and the
  poaching inflow `κ_S γ_o(p) ∫_{p*}^p s* e_S` was absent entirely. All three inflows are
  now present with their own densities, and the reservation coverage `ω` is applied as in
  the stationary solve.
- `transition_solver.jl:_init_path!`, `_build_result` — `path.eS` was seeded as a
  mass-scaled density, evolved as a unit density, and consumed scaled by column mass
  again. It is now the per-aS unit shape throughout, the `sc.e_frac` convention, and the
  scaling uses the NON-draining column mass. Visible in the smoke gate's `mean_wage_S`,
  which ranged over 1075–1302 and now over 877–887.
- `transition_solver.jl:_update_tightness!` — skilled free entry reads its seeker pool off
  `sc.u_frac`/`sc.e_frac`, which the path never installed, so `compute_Jbar_skilled` priced
  every date's vacancy against the POST-switch stationary composition. Both are now
  installed from the path.
- `transition_main.jl:36` — `W_COND_TARGET` was 2.0 (`_equalW`) while every shipped
  estimate is `_diagonalW` and `smm_main.jl` defaults to 0.0, so the entry point stopped at
  the first bundle load. Now 0.0.
- `smoke/smoke_transition.jl` — `SMOKE_NX` default 60 → 120. base_fc's skilled block does
  not converge at Nx = 60, so the gate failed on its own steady state and never reached
  the path. It passes at 120 in about a minute.
- `code/tools/transition_audit.jl` (new) — the gate that would have caught all of this.
  Runs a pair, or with `TA_NULL=true` runs a window against ITSELF, where the true path is
  the constant one and any drift measures the gap between the forward laws of motion and
  the stationary KFE with no economics mixed in.

### Measured, not asserted

base_fc → crisis_fc, Nx = Np = 120, Nt = 241, T_max = 120 months, damp = 0.3. Converges in
22 outer iterations to ‖Δθ‖∞ = 9.44e-05 against a 1e-04 tolerance, in 35 s on 6 threads.
Both steady states solve.

Null shock (base_fc against itself), terminal deviation from the stationary equilibrium the
path started at, before → after:

| quantity | v29.1.0 | v30.0.0 |
|---|---|---|
| `u_S` total | +30.05% | −1.32% |
| `ur_S` | +30.94% | −0.66% |
| `u_U` total | −7.15% | −7.16% |
| `θ_U` | +10.6% | +10.6% |

The skilled block now reproduces its own steady state. The unskilled block does not, and
that residual is a specification question, not a numerical one — see known gaps.

On the real pair: the cross-market flow `f_U ∬d·u_S` is exactly 0 at every one of the 241
dates, while the active side of `F_d` holds 46.5% of the POPULATION and 0.000e+00 trained
mass at every date. The channel is unpopulated, not switched off. `F_τ`'s floor jumps from
0.29222 (z0) to 0.27039 at the switch date and is flat thereafter. The triangle excess
`d(·,z0) + d(·,z1) − d(z0,z1)` peaks at 134% of the endpoint separation, so the interior
is not on the line between the endpoints. The terminal distribution is 1.265e-02 from z1,
38.7% of the endpoint separation; lengthening the horizon to 1200 months moves it only to
1.115e-02, so it is a fixed-point gap and not slow convergence.

### Settings

`SMOKE_NX` default 60 → 120. `transition_main.jl:W_COND_TARGET` 2.0 → 0.0. New, all
optional with the defaults above: `TA_PAIR`, `TA_NX`, `TA_NSTEPS`, `TA_MAXIT`, `TA_TMAX`,
`TA_DAMP`, `TA_NULL`. None is read outside `code/tools/transition_audit.jl`.

### Known gaps

- **The backward pass is not a backward pass.** It runs the STATIONARY inner loops at each
  date's tightness, so the date-`t` value functions are the post-switch stationary values
  at `θ(t)` and the `∂_t V` term is absent — agents price today's tightness but not the
  anticipated path of tomorrow's. Warm-starting from `n+1` does not supply the derivative;
  the fixed point is the same whatever the warm start. The notes' algorithm specifies a
  backward pass for the time-dependent HJBs. This is a respecification, not a patch, and
  was left for the theorist. The visible consequence is that `θ` and `F_τ` jump to
  essentially their terminal values within the first months and are flat thereafter.
- **The unskilled block does not reproduce its own steady state** (`u_U` −7.2%, `θ_U`
  +10.6% under a null shock). The two descriptions of entry into training differ: the
  notes' transition law makes `τ` a HAZARD out of unskilled unemployment
  (`∂_t t = τ u_U − (φ+ν)t`, so a training cell carries `u_U > 0`), while the notes'
  stationary distribution and `solve_stationary_unskilled!` take the decision AT BIRTH
  (`τ = 1 ⇒ u_U = 0, t = νℓ/(φ+ν)`). These have different fixed points. Not resolved here;
  picking one changes the model.
- **`ξ_U` disagreement.** The forward pass carries `ξ_U + λ_U G(p*_U)` as the unskilled
  separation hazard, following `unskilled.jl` and `equilibrium.jl`; the notes' transition
  law writes that outflow without `ξ_U`, and the notes' §sep_asymmetry argues unskilled
  separation is endogenous in its entirety. `ξ_U` is 0.00365 (base_fc) and 0.00360
  (crisis_fc), so the two are not numerically the same. Flagged in the file header, not
  changed.
- **`equilibrium.jl`'s moment layer keeps the node read** that `_skilled_margin_masses`
  replaces, so the shipped `sep_rate_S` carries the spurious `λ_S Γ_s(node 1)` term while
  the distribution it is computed from does not. Closing that moves an estimated moment
  and every estimate with it; deliberately out of scope here and recorded so it is not
  rediscovered.
- **`transition_main.jl` is not reproducible from the pinned environment.** It does
  `using Plots` and `using LaTeXStrings`, neither of which is in `Project.toml` or
  `Manifest.toml`; they resolve from the user's shared `v1.10` environment via the default
  `LOAD_PATH`. Not fixed here — adding them re-resolves the manifest, which is the
  reproducibility anchor.
- Gate items not run: `check_forwarding.jl`, `check_dead_settings.jl`,
  `check_output_tree.jl`. Only the transition path and the transition smoke gate were
  exercised. `check_version_sync` is clean.

### Tried and rejected

- **Blaming the terminal gap on the demographic horizon.** `ln2/ν ≈ 215` months against a
  120-month horizon made "the path has not had time to arrive" the obvious explanation.
  Measured: extending to 1200 months (5.6 half-lives) moved the terminal gap from 1.265e-02
  to 1.115e-02, so at most a tenth of it is horizon. Do not re-propose lengthening
  `TR_T_MAX` as the fix.
- **Blaming the training mass for the null-shock drift.** The τ-as-hazard vs τ-at-birth
  difference predicts a training stock ~17% BELOW stationary; measured, `t` is 1.0% below
  and `u_S` was 30% above. The τ semantics are a real disagreement (above) but they are not
  what the drift was.
- **Changing `uc.duS_carry` to `d·m_S`** to match `solver.jl:168`. The path tracks `u_S`
  separately from `m_S`, so `d·u_S` is the correct seeker pool along a path and the two
  coincide wherever `d ∈ {0,1}`. Measured irrelevant here regardless: `∬d·m_S` is exactly
  zero at every date on this pair.

---

## v29.1.0 — 2026-09-04

**MINOR: the JAC_ONLY standard-error path runs.** It could not: `MCMC_main.jl:1065` called
`snapshot_chain(res.chain, res.chain_lp, …)` unconditionally and the JAC_ONLY `res` stub has
no `chain_lp`, so every run died with `type NamedTuple has no field chain_lp` **before the
first solve**. Not PATCH, because the results CSV gains a column and so is not
byte-identical as a file; not MAJOR, because no economic content changed, the parameter
vector and the objective were not touched, and every number the chain route produces is
reproduced.

### Comparability

**Nothing is superseded.** No stored estimate, moment target or Q moves. On the chain route
every field of `mcmc_results_{window}{W}.csv` is unchanged and the file gains one appended
column. The JAC_ONLY route has no prior output to compare against — it had never produced a
file. Readers that index the CSV by position are safe: the 16 existing column names hold
their positions and `corner_declared` is 17th.

### What changed

- `mcmc/MCMC_main.jl:1065` — the chain snapshot is skipped under JAC_ONLY. There is nothing
  to protect (`gens = 0`, `draws` is θ̂ itself) and the write targets `chain_path()`, so a
  JAC_ONLY run would have replaced this window's 20 MB, 9500-generation chain bundle with a
  single-draw stub. That data loss was latent, not observed: the field error fired first.
- `mcmc/MCMC_main.jl:1424` — section 7's chain bundle likewise skipped under JAC_ONLY (it
  breaks on the same three absent fields: `chain_lp`, `accepted`, `replaced`). The closing
  `Wrote …` line now names only the files that were written. **Consequence:** on the
  JAC_ONLY route `G`, `G_R2` and the 600 stored moment vectors are not persisted; the
  console's `moment R² min/median` line is the record. Deliberate — the alternative, a
  `design_*.jls` beside the chain bundle, was offered and declined.
- `mcmc/MCMC_main.jl` — new `CORNER_PARAMS`, a hand-declared list of parameters sitting at a
  corner, in the style of `smm_main.jl`'s `SKIP_MOMENTS`. Whether a corner is an acceptable
  economic zero is not a decision the code takes. Resolved to `CORNER_FLAG` immediately
  after the free set is known, so a misspelled key fails at startup rather than after ~600
  solves, and accepted in any of the three spellings this project uses for a parameter
  (display symbol `:b_S`, ParamSpec name `:bS`, ASCII configuration key `:alpha_U`).
- `mcmc/MCMC_main.jl:1315` — `corner_declared` (0/1) appended to the results CSV as its
  17th column. Purely a label: no existing value, name, order or format changes, and with
  the shipped empty default it is 0 on every row.
- `mcmc/MCMC_main.jl:1393,1400-1412` — the printed advice was inverted. It said "For
  reportable standard errors run with `ROYSEARCH_MCMC_JAC_ONLY=false`", pointing the reader
  at the route this project dropped. JAC_ONLY's `se_curvature` and `se_bound` **are** the
  reported standard errors (CH Thm 4 needs no chain), so the block now says which column to
  report, why, and where it is. The shared paragraph no longer calls both columns merely
  "indicative".
- `smm/settings.jl:266` — `_env_parse(::Type{Vector{Symbol}}, …)`, comma-separated with
  entries trimmed and empties dropped, so a set-valued setting can be driven from one
  variable. `SETTINGS_REGISTRY` gains the `:CORNER_PARAMS` row; its note is built with `*`
  rather than a backslash continuation because `check_structure.jl`'s string tracker resets
  at each newline and a multi-line literal inside a call leaves the file reading unbalanced.
- `README.md` — new "Standard errors" section: the two routes to Ĵ with the flag that
  selects them, which column is reported and why, the object-to-file crosswalk for the CSV
  and the chain bundle, and how to declare a corner.

### Measured, not asserted

Run gate, all four steps on `base_fc` at `--threads 4`, seeded from
`output/estimates/estimate_base_fc_diagonalW.jls` (Q(θ̂) = 1025.4726):

- **Pre-fix**: `ROYSEARCH_MCMC_JAC_ONLY=true` dies at `MCMC_main.jl:1065`,
  `type NamedTuple has no field chain_lp`, before any Ĝ solve. Reproduced in a scratch tree.
- **Post-fix**: exit 0 in 5m15s (scratch) and 4m37s (repo). 542 of 600 local-design points
  feasible; Ĝ is 35×23; moment R² min = 0.190, median = 0.704; `gens=0/0 (chain skipped:
  JAC_ONLY)`. 23 free parameters, ξ_U and both β free.
- **Does not read the chain bundle.** The scratch tree was built with an EMPTY
  `output/chains/`, and the run completed and wrote the full SE table there — so no chain
  object entered it. Statically, `MCMC_main.jl` contains no `deserialize`/`read_bundle` of
  `chain_path()`; the only bundle read is the seed estimate. `output/chains/` stayed empty
  after the run.
- **CORNER_PARAMS.** `ROYSEARCH_CORNER_PARAMS="b_Q,skl_bet"` fails at startup listing all 23
  free parameters (`skl_bet` accepted, `b_Q` rejected). `"b_S, alpha_U"` — mixed spellings —
  flags exactly 2 of 23 rows, `corner_declared = 1` on `b_S` and `α_U`, exit 0.
- **Not bit-identical, and not because of this change.** Three runs at identical settings
  agree exactly on `point_estimate` and `se_curvature` for all 23 rows. `se_bound` differs
  in its LAST printed digit on 5 rows (a_ℓ, b_ℓ, a_Γ, b_Γ, k_U; e.g. a_Γ 5.34027257 /
  ...258 / ...259 across three runs) — 1e-8 absolute, 2e-9 relative. The same spread appears
  between two runs with an EMPTY `CORNER_PARAMS`, so it is the solver's run-to-run
  floating-point floor, not the setting: `CORNER_FLAG` is read only by the banner and the
  CSV's `%d` field. Stripping that field, the corner-flagged and empty-default runs are
  otherwise identical.
- **Gates.** `check_forwarding` PASS (74 registry rows / 74 keys read, `:CORNER_PARAMS`
  counted on both sides). `check_dead_settings` PASS. `check_output_tree`: 12 findings, all
  pre-existing (`.csv` in `tables/`, `_backup_Q*.jls` in `estimates/`), none naming
  `mcmc_results`. `check_structure`: 7 findings, unchanged in count and content from before
  this version, `settings.jl` still at its pre-existing net depth 1.
- **No `.jls` moved.** All 29 bundles under `output/` hold their pre-run mtimes and sizes.

### Settings

`CORNER_PARAMS` added: `Vector{Symbol}`, default `Symbol[]`, consumer
`MCMC_main.jl:top level`, `:live`. It needs no `SMMRunParams` field — it is a reporting
label, and adding one would make every bundle on disk unreadable — and no keyword
forwarding, since its consumer is the entry point itself, as with `MCMC_JAC_ONLY`.
No setting was retired or re-defaulted.

### Known gaps

- **`edge_frac` is NaN on the JAC_ONLY route, for every row.** It is a statistic of DRAWS
  (the share within 1% of a box edge) and this route has none. So the column intended as the
  input for choosing corners by hand is unavailable exactly where corners will be chosen;
  `point_estimate` against the box bounds is what is left, and `CORNER_PARAMS` is the
  declaration. Not fixed here: the CSV's existing values were out of scope.
- The run is not bit-reproducible at full `%.8f` width (see above). The mechanism was not
  traced; a threaded reduction inside the solve or in BLAS is the obvious candidate.
- `b_S = 6.23e-06` came back with `se_curvature = 0.0440` and `se_bound = 0.1713` — no
  spuriously tiny standard error in this run, unlike the `se(b_S) = 2.46e-07` this project
  saw before. Whether b_S is at its bound was not decided here, by design.
- The section-7 bundle is not written under JAC_ONLY, so `G_R2` — which already exists in
  that bundle, alongside `G = Ĝ`, and needed no code change — is reachable only from a chain
  run.

### Tried and rejected

- **Automatic corner detection and NaN-ing the SE of a cornered coordinate.** Designed and
  written, then withdrawn on the user's instruction: the output file is not to be altered
  automatically, and which parameters are corners is a hand-made choice made when the tables
  are built. What it would have measured, kept here so it is not re-proposed blind: realised
  θ-span per coordinate across the feasible design as a share of box width; `‖√W·Ĝ[:,k]‖`
  against its median; and the share of `e_k` lying in the numerically unresolvable subspace
  of the column-normalised `Ĵ` (the scaling matters — raw `cond(Ĵ)` is large from parameter
  scale alone). It also revoked values, which is a reported-number change; `CORNER_PARAMS`
  achieves the labelling without touching one.
- **`%.8f` → `%.10g` for the CSV's numeric columns.** Correct in principle — at `%.8f` a
  parameter of order 1e-6 keeps two significant figures, and `b_S` is that scale — but it
  rewrites every value in the output file, which is off limits. Reverted before shipping.
- **A `design_{window}{W}.jls` for the JAC_ONLY bundle**, so Ĝ and `G_R2` would persist
  without ever touching `chain_*.jls`. Offered and declined in favour of writing nothing
  into `output/chains/` on that route.

## v29.0.1 — 2026-09-03

**PATCH: prints only.** No numeric path was touched, so every number in every output file
is bit-identical to v29.0.0. The substantive event in this version is a **bundle
promotion**, which is a change to `output/`, not to the code.

### The promotion

v29.0.0 withheld the `Q = 488` promotion because it was gated on reproducing
`488.548412`, and that value cannot be reproduced: it was written under v20.0.0, before
v27.0.0 replaced the two hard `1{p < p^oj}` indicators with the covered fraction.
Preserving the v27 fix and reproducing a pre-v27 number are mutually exclusive. The gate
was re-set to v29's own value at that θ, `494.354151833344`, and the promotion performed.

- **Gate: PASSES.** Evaluated at the promoted bundle's own stored spec (23 free, 28 scored
  moments), `Q = 494.354151833431` against the target `494.354151833344` —
  Δ = **8.69e-11**, solver precision.
- `output/estimates/estimate_base_fc_diagonalW.jls` was **preserved first** to
  `estimate_base_fc_diagonalW_pre_xiU_restore_Q4960.615185.jls`, verified byte-identical
  (`cmp` clean, sha256 `dce15b1b…` on both), then overwritten from
  `estimate_base_fc_diagonalW_backup_Q488.548412.jls`. The source backup is untouched
  (sha256 `911e24fc…`, mtime `Aug 31 17:01` before and after). Nothing was deleted.
- **A byte copy is the migrated object here.** v29.0.0 established by shape inventory that
  no serialised layout changed, so there was nothing to rebuild; the promoted file is
  byte-identical to the backup (sha256 `911e24fc…`).
- The promoted bundle's stored `loss_opt` is still the v20 value `488.548411666322`, so the
  warm-start banner prints `Prior Q = 4.885484e+02` while v29's actual value at that θ is
  `494.354151833344`. `loss_opt` is print-only on this path — the seed is built from
  `theta_opt` — and it was left alone rather than rewritten, because editing it without
  also editing `provenance.version = "20.0.0"` would produce a bundle that misreports its
  own vintage.

### ξ_U now reaches both `fc` windows

The promotion fixes the silent default that v29.0.0 flagged. `crisis_fc` reads its regime
parameters off the decoded `base_fc` baseline, so a 23-free baseline carrying `ξ_U` feeds
it through:

| window | ξ_U seeded before | ξ_U seeded after | matched | `Q`(init) before | `Q`(init) after |
|---|---|---|---|---|---|
| `base_fc` | 0.0063398088 (spec default) | **0.00362809182024** (estimated) | 20/29 → **23/29** | 30773.498574767 | 230030.740259860 |
| `crisis_fc` | 2e-10 (silent default) | **0.00362809182024** (estimated) | field read | 195202.137814127 | 1654354.021787388 |

`base_covid` and `crisis_covid` are unchanged at `ξ_U = 0.00985778400948`, `Q`(init)
`80413.217215659` and `68158.739525956`.

### A warm start from this bundle does NOT begin at Q = 494.4

`Q`(init) rises rather than falls, and the cause is not the promotion. The launch spec is
not the bundle's spec: `smm_main` builds the current default free set and then applies the
current `FIX_PARAMS` on top, which **overrides two parameters the bundle estimated**.

| | free | scored | β_U | β_S | `Q` |
|---|---|---|---|---|---|
| bundle's own stored spec | 23 | 28 | 0.6818957492 | 0.3216124987 | **494.354151833431** |
| launch spec, shipped `FIX_PARAMS` | 21 | 35 | 0.5 (pinned) | 0.5 (pinned) | 240075.320313209 |
| launch spec, β pins lifted | 23 | 35 | 0.6818957492 | 0.3216124987 | 1129.948844969 |

Lifting the two β pins moves `Q` by a factor of 212, so the β override is the dominant
term; the residual `1129.9` against `494.4` is the seven extra moments the current
catalogue scores (35 vs the bundle's 28) and does not fit at this θ. Both effects are
properties of the current `FIX_PARAMS` and moment set against a v20 bundle, and both
applied to the previous warm-start file too. **To start a run in the 488 basin, `unsk_bet`
and `skl_bet` have to come out of `FIX_PARAMS`** — a specification decision, not a code fix,
so nothing was changed.

The two launch numbers for `base_fc` differ slightly by harness (`230030.740259860` from
the four-window launch, `240075.320313209` here) because the former uses `smm_main`'s
`SimParams` and the latter the bundle's stored one. At a point this badly fit the
equilibrium is sensitive to solver tolerance; both are ~2.3–2.4e5 and neither is near
`494.4`.

### Prints

- **The warm-start unmatched and clamped lists print block-qualified names**
  (`unsk:ξ`, `skl:ξ`) instead of bare ones, `smm_main.jl:645`, `:658`. `μ, η, k, β, λ, ξ`
  and `σ_w` each exist in both blocks, so the old line read
  `Unmatched: bT, PU, η, η, β, β, ξ, σ_w, σ_w` — six of nine entries ambiguous duplicates,
  and a bare `ξ` that did not say which block. That ambiguity is what hid `unsk:ξ` falling
  back to its default in v29.0.0. Both vectors are display-only: they are built,
  `isempty`-tested and `join`ed into a `@printf`, and read nowhere else.
- **The crisis branch now reports its regime seeding**, `smm_main.jl:581`:
  `Regime init: read N fields from the baseline; ξ_U = <value> (<provenance>)`, where the
  provenance says whether the baseline's spec actually freed `unsk:ξ` or whether the value
  is `unpack_θ`'s default. This branch seeds by reading fields off decoded structs, so
  every parameter "matches" whether or not the baseline estimated it and there is no
  matched N/M line to inspect — that is exactly how `crisis_fc` came to start at `2e-10`
  in v29.0.0 with no warning. A one-line `@printf`; no logic touched.

### Verification

`smm_main.jl` parses with zero `:error` / `:incomplete` nodes. `check_version_sync` in sync
at `29.0.1`. All ten pre-existing bundles still load. `output/estimates` gained exactly the
two files named above and `output/chains` was not written to.

### Known gaps

- **No re-estimation was run**, so the promoted point is a v20 argmin re-scored under v29,
  not a v29 optimum.
- **`Q`(init) for both `fc` windows is now much worse than before the promotion** — see the
  β section above. This is expected and diagnosed, but it means the shipped warm start is
  not a good start until `FIX_PARAMS` and the bundle agree on β.
- The promoted bundle reports `provenance.version = "20.0.0"` and a stale `loss_opt`;
  deliberate, see above.
- **Not verified.** The two print changes were confirmed by parse and by inspection of
  every read of the affected names; `smm_main.jl` was not run end to end, because doing so
  would enter the optimiser.

---

## v29.0.0 — 2026-09-03

**MAJOR, because the parameter vector and `Q` both change.** `ξ_U`, the exogenous
unskilled separation hazard, is **added back** to the model. This is not a revert of
v28.0.0: it is an addition of `ξ_U` onto the v28 tree, so every fix made since the purge
survives — the v27.0.0 OJS moment-layer covered-fraction split, the LBFGS polish stage and
its settings, `code/tools/width_audit.jl`, the DE-MC replacement reporting, the
moment-catalogue growth to 43 names (35 scored) and the settings gates are all untouched.
`ξ_U` was re-inserted at its pre-purge position and with its pre-purge box; nothing was
restored by checking out a file.

The reason is estimability, not symmetry. v28 removed `ξ_U` on the identification argument
that `sep_rate_U + wchg_rate_U = λ_U` pins `λ_U` and `G(p*_U)` from two moments; the cost
was the fit. `sep_rate_U` cannot reach its target from the endogenous margin alone, and
the shortfall dominates the criterion. With `ξ_U` free the Jacobian at the old basin
inverts and standard errors become reportable, which is what the restoration is for.

The free set goes from 28 to **29** entries in `default_free_params()`, and from 20 to
**21** under the shipped `FIX_PARAMS` (`β_U`, `β_S`, `η_U`, `η_S`, `b_T`, `P_U` pinned).

### Comparability

- **`Q` is a different function; no v28 number is comparable.** At the shipped `base_fc`
  warm-start point `Q` moves from **4960.615185021 to 30773.498574767**, but that
  comparison is not a like-for-like: the v28 bundle carries no `ξ_U`, so the v29 spec seeds
  it from the ParamSpec default `0.0063398088` rather than from an estimate. See the launch
  table below — this is a seeding artefact, not a measurement of the restoration.
- **The restoration is an exact inverse of the purge, measured.** Every pre-purge bundle
  re-evaluated under v29 reproduces the `v27 Q` column recorded in the v28.0.0 entry, to
  ≤ 1.1e-9 across six bundles. That is the acceptance test that matters: v29 = v27 at every
  stored point, with v27's own moment-layer fix intact.
- **No bundle needed migrating, and none was rewritten.** `UnskilledParams` is reachable
  from no serialised bundle — measured, not assumed: the struct-shape inventory over all
  ten `.jls` files is **byte-identical before and after** the field addition, and all ten
  still deserialise. `output/` was not written to at all; every bundle retains its original
  mtime and sha256.
- **The stored `Q = 488.548412` does not reproduce and cannot.** It recomputes to
  `494.354151833`, Δ = **+5.805740167**. The value was written under v20.0.0, before
  v27.0.0 replaced the two hard `1{p < p^oj}` indicators with the covered fraction; the
  v28.0.0 entry already records `494.354151833` as this bundle's v27 value. Preserving the
  v27 fix and reproducing a pre-v27 number are mutually exclusive. **The requested
  promotion of this point onto the warm-start path was therefore withheld** and
  `output/estimates/estimate_base_fc_diagonalW.jls` is unchanged.

### What changed

- **Solver.** `UnskilledParams.ξ` restored at field position 10 of 11, between `α_U` and
  `σ_w` — its pre-purge slot, `params.jl:93`; the asymmetry note on `SkilledParams.ξ`
  rewritten, `params.jl:150-154`. Surplus base rate is `r+ν+λ+ξ`, `unskilled.jl:88`, and
  `solve_unskilled_surplus_on_grid!` regained its trailing optional `ξ` argument
  (default `0.0`, so every existing caller still compiles), `unskilled.jl:76`. Stationary
  unskilled separation `δ = ξ_U + λ_U G(p*)`, `unskilled.jl:277`. `ξ` bound in
  `unskilled_inner_loop!`, `unskilled.jl:120`.
- **Moment layer.** `δU_by_a = ξ_U .+ λ_U · G(p*)`, `equilibrium.jl:279`; `ξU` bound at
  `equilibrium.jl:44`; the surplus call carries it, `equilibrium.jl:121`; the `wchg_rate`
  comment rescoped from skilled-only back to both blocks, `equilibrium.jl:290-292`.
- **SMM layer.** `ParamSpec(:unsk, :ξ, 0.0000, 0.0200, 0.0063398088, …)` restored to
  `default_free_params()` at index 26, immediately before `skl:ξ` — the pre-purge ordering,
  `smm_params.jl:673`. `(:unsk,:ξ)` back in `REGIME_SPECIFIC_PARAMS`, `smm_params.jl:687`;
  `(:unsk,:ξ) => :unsk_xi` in `_DEFAULT_PARAM_KEY`, `smm_params.jl:798`; `:unsk_xi =>
  :unsk_ξ` in `_ASCII_TO_FIXED_KEY`, `smm_params.jl:819`; the three `unpack_θ` sites
  restored, `smm_params.jl:1013`, `:1041`, `:1075`.
- **`unpack_θ` cross-block leak closed** — see the separate section below. This is the one
  change beyond the enumerated purge sites.
- **Entry points and downstream.** `:unsk_xi => 0.00633981` back in `DEFAULT_PARAMS` and in
  the `FIX_PARAMS` valid-key list, `smm_main.jl`; `ξ = 0.0` in the unskilled block of
  `model_main.jl:120`; transition law of motion `δU = ξ_U + λ_U G(p*_U)`,
  `transition_solver.jl:237`, with its header comment at `:22`; the `ξ_U` row restored to
  `REGIME_SPECIFIC`, `plots_and_tables/transition.jl:751`.
- **Counts.** `MCMC_main.jl:563` now reads "35 moments and 21 free parameters", both
  measured from the launch table rather than decremented by hand; it previously said 28 and
  22, and the moment figure had been stale since the catalogue grew.
- **Not edited, because they carry no `ξ_U` site** (verified, contra the brief's list):
  `code/policy/policy_solver.jl`, `code/smm/settings.jl`, `code/gates/*`,
  `code/plots_and_tables/model.jl`. `ξ_U` was never a `ROYSEARCH_*` setting, so
  `SETTINGS_REGISTRY` is untouched.

### A latent defect the restoration re-armed, and closed

`unpack_θ`'s first pass reads each block's fields through `_get(name, block, default)`,
which fell through to a **bare-name** lookup in `free_vals`. `free_vals` is keyed by bare
name, so for a field carried by both blocks it holds whichever block's spec came last. The
disambiguation loop that follows is the authority for shared names — but it only corrects a
block whose spec is present, so a spec that frees one block's `ξ` while leaving the other
neither free nor pinned silently read **the wrong block's value**.

Before v28 this could not fire: `ξ` was in both blocks and always freed together. v28 made
`ξ` skilled-only, hence block-unique, hence safe. Restoring `ξ_U` re-armed it, and two
stored bundles trigger it — the 20-free v28 bundles free `skl:ξ` and know nothing of
`unsk:ξ`. Measured: they decoded to `ξ_U = ξ_S = 0.00507325728787` and
`0.00519548684032`, and `Q` at their own stored points came out **24030.263613314** and
**75941.735439643** against stored `4960.615185021` and `23552.793052229`.

`smm_params.jl:986-999` now returns the default for a name in `_SHARED_PARAM_NAMES` instead
of crossing blocks. `_SHARED_PARAM_NAMES` is derived from `default_free_params()`, so it
picked `ξ` up automatically when the row was restored.

- **Acceptance test.** Both v28 bundles now reproduce their own stored `Q`:
  `4960.615185021483` (Δ = 1.3e-11) and `23552.793052230489` (Δ = 1.5e-9), both with
  `ξ_U = 0` — the correct reading of a spec that neither frees nor pins it.
- **No-op where both `ξ` are free.** The `Q = 488` bundle gives `494.354151833344` after
  the fix against `494.354151833491` before, Δ = 1.5e-10 — the threaded-reduction floor.
- Blast radius is confined to stored specs written under a version whose free set differed;
  `assert_all_params_accounted` rejects that state for any spec built through the normal
  path.

### Measured, not asserted

Single solves at `Nx = Np_U = Np_S = 120` as stored in each bundle's own `SMMRunParams`,
`-t 4`. No optimisation of any kind was run: no annealing, no Nelder-Mead, no DE, no LBFGS
polish, no `smm_main` end to end.

- **v29 = v27 at every stored point.** Recomputed `Q` against the `v27 Q` column of the
  v28.0.0 entry:

  | bundle | scored moments | stored `Q` | v27 `Q` (recorded) | v29 `Q` (measured) | Δ to v27 |
  |---|---|---|---|---|---|
  | `…_backup_Q488.548412.jls` | 28 | 488.548411666 | 494.354151833 | 494.354151833344 | 3.4e-10 |
  | `…_backup_Q490.550869.jls` | 28 | 490.550868777 | 493.413940847 | 493.413940847025 | 2.5e-11 |
  | `…_backup_Q497.786594.jls` | 28 | 497.786594464 | 497.439349012 | 497.439349012299 | 3.0e-10 |
  | `…_backup_Q970.339482.jls` | 31 | 968.097038448 | 974.385004593 | 974.385004592550 | 4.5e-10 |
  | `…_postmean.jls` | 28 | 486.500699496 | 493.751132085 | 493.751132085133 | 1.3e-10 |
  | `estimate_base_covid_diagonalW.jls` | 28 | `Inf` | 2151.030293595 | 2151.030293595177 | 1.8e-10 |

- **`Q = 488.548412` reproduction: FAILS at `494.354151833344`, Δ = +5.805740167022.**
  The 28-row decomposition sums to `494.3541518334912` and is dominated by
  `training_share` (203.996, 41.3% of `Q`), `ltu_share_S` (118.208, 23.9%) and `jfr_S`
  (116.453, 23.6%) — 88.7% from three moments, none of them unskilled-separation.
  `sep_rate_U` is **+0.58% off target contributing 0.748 (0.15% of `Q`)** and `ur_U` is
  −0.28% off contributing 0.411, so the two moments the purge broke are fit again at this
  point. Note `ltu_share_S` carries weight in this v20 spec but is held out by the current
  `SKIP_MOMENTS` as redundant with `usurv27_S`.
- **`p*_U` is NOT invariant to `ξ_U`.** The pre-purge comment claimed it was; the claim was
  measured and is false, so it was corrected rather than restored (`unskilled.jl:64-70`).
  At the `Q = 488` point, `ξ_U` 0 → 0.0063398088 moves `p*_U` min 0.3676806562 → 0.4072393724,
  median 0.4228440109 → 0.4581835142, max |Δ| = 3.955872e-2 (9.4% of the median). The
  reservation *formula* is genuinely unchanged — no `ξ` appears in it — but `p*` reads the
  tail integral, which carries the base rate.
- **The splitting identity generalises exactly.** At the same point,
  `sep_rate_U + wchg_rate_U` = `0.162073171604` at `ξ_U = 0` against
  `λ_U = 0.162073171604`, and `0.168412980404` at `ξ_U = 0.0063398088` against
  `λ_U + ξ_U = 0.168412980404`. Both to 12 digits. So the v28 identification argument does
  not survive: the ratio no longer returns `1 − G(p*_U)`.
- **All four windows launch.** Setup path only, objective evaluated once at the resolved
  initial point, then stopped. Window names read from `data/derived/windows.json`
  (`base_fc, crisis_fc, base_covid, crisis_covid`). 35 of 43 moments scored in every
  window; **no** moment auto-skipped for a missing or NaN target anywhere.

  | window | free | init source | matched | `ξ_U` | `ξ_U` seeded | `Q`(init) |
  |---|---|---|---|---|---|---|
  | `base_fc` | 21 | warm start, `estimate_base_fc_diagonalW.jls` (20 free) | 20 / 29 | **DEFAULTED**, reported unmatched | 0.0063398088 | 30773.498574767127 |
  | `crisis_fc` | 16 | baseline `base_fc` (20 free) | field read, no report | **DEFAULTED SILENTLY** | 2e-10 | 195202.137814126705 |
  | `base_covid` | 21 | warm start, `estimate_base_covid_diagonalW.jls` (25 free) | 25 / 29 | MATCHED | 0.00985778400948 | 80413.217215659257 |
  | `crisis_covid` | 16 | baseline `base_covid` (25 free) | field read, no report | MATCHED | 0.00985778400948 | 68158.739525955898 |

- **Release gate.** All nine changed files parse with zero `:error` / `:incomplete` nodes,
  by walking the `Meta.parseall` tree. Module chain loads in entry-point order.
  `check_forwarding.jl` PASS (73 registry rows, 73 keys read, 57 `SMMRunParams` fields).
  `check_dead_settings.jl` PASS (167 keyword arguments). `check_version_sync` in sync at
  `29.0.0`, one declaration, three readers, no duplicates.
  `assert_all_params_accounted` passes on all four window specs and on both a `ξ_U`-free
  and a `ξ_U`-pinned spec.

### Settings

None added, retired or re-defaulted. `ξ_U` is a model parameter, never a `ROYSEARCH_*`
setting, so `SETTINGS_REGISTRY` is untouched. The `FIX_PARAMS` key `:unsk_xi` is **valid
again** — pinning it to `0.0` recovers the v28 model exactly.

### Known gaps

- **The `Q = 488.548412` promotion was not performed.** It was gated on that value
  reproducing exactly; it reproduces to `494.354151833344`, so nothing was promoted, no
  preservation copy was written, and `estimate_base_fc_diagonalW.jls` is byte-unchanged
  (sha256 `dce15b1b…`, mtime `Sep 3 14:48`). The warm-start path therefore still holds the
  `Q = 4960.6` point. **Superseded by v29.0.1**, which re-gated on `494.354151833344` —
  v29's own value at that θ — and performed the promotion.
- **No re-estimation was run**, so no v29 optimum exists and nothing is claimed about where
  the argmin now sits. The four `Q`(init) values above are start points, not fits.
- **Two of four windows seed `ξ_U` from a default, not an estimate**, and `crisis_fc` does
  so with **no report at all**: the crisis branch reads fields off the decoded baseline
  structs (`smm_main.jl:547`), so every regime parameter "matches" by construction and
  there is no matched N/M line to inspect. `crisis_fc` starts at `ξ_U = 2e-10`, the clamped
  zero. A warm-start-style report for the crisis branch is not implemented.
- **The warm-start unmatched list prints bare names.** `base_fc` reports
  "Unmatched: bT, PU, η, η, β, β, ξ, σ_w, σ_w" — six of the nine are ambiguous duplicates
  and `ξ` does not say which block. It happens to be `unsk:ξ` here, but the line cannot be
  read without knowing that.
- **The chain bundle still cannot be re-interpreted.** `chain_base_fc_diagonalW.jls` loads
  and its arrays are `draws 23×9500`, `chain 23×95×1000`, `G 28×23`, `se_chain` length 23 —
  a marginal of the *old* posterior over a free set that has now changed twice. It needs a
  fresh chain; `se_chain` and `G`-based standard errors are not this model's.
- **`check_structure.jl` reports 7 unbalanced files** (`check_forwarding.jl`, `demc.jl`,
  `repo_root.jl`, `moments.jl`, `settings.jl`, `check_tau_margin.jl`, `plateau_probe.jl`).
  Pre-existing, exit code 0, and **none** is a file this version edited; its depth counter
  is confused by `end` inside strings and comments.
- **The Model Notes still assert the one-channel architecture.** Ten `ξ_U`-bearing objects
  were compared line by line against the code; 2 agree, 8 disagree, 6 of those being pure
  omissions and 2 substantive (the splitting identity at L930-936, and the
  deliberate-asymmetry prose at L69/L903/L928). Nothing in the `.tex` was edited. The
  comparison is written up separately; note that `latex/roysearch_model_notes.tex` is **not
  tracked by git**, so no pre-purge equation text exists to diff against.
- **`migrate_bundles.jl` was not run**, because the shape inventory showed nothing to
  migrate. Its install bug recorded in the v28.0.0 entry (`rebuild` writes
  `.migration/<name>.jls` but installs `mv(p * ".new", p)`, `:293`) is **still present and
  still unexercised** — found again, not fixed, out of scope.
- **Not verified.** `code/policy/` remains broken at `policy_params.jl` on
  `UndefVarError: RegimeParams`, pre-existing and unrelated. The transition and plotting
  edits are consistency-only and were not executed. `TARGETING_NOTE.md`, `README.md`,
  `SETTINGS.md` and `VERSION_NOTES.md` still cite pre-v28 parameter counts and were left as
  written.

### Tried and rejected

- **Reverting files to their pre-purge state with `git checkout`.** Rejected on
  instruction, and it would have been wrong regardless: `HEAD` is v19.5.0, nine majors
  behind the working tree, so a file-level revert would have taken out the v27 OJS split
  and everything after it. `HEAD` was used only as a read-only reference for the pre-purge
  `ξ_U` text and the exact `ParamSpec` row. Note the purge itself was **never committed** —
  the working tree was at v28.0.0 against a v19.5.0 `HEAD` — so `git log`/`git diff` could
  not enumerate the purge sites; the v28.0.0 changelog entry's `file:line` list served
  instead, and it proved complete except for the four files it correctly says carry no site.
- **Promoting the `Q = 488` point despite the 5.8058 shortfall.** Rejected on instruction
  and on the merits: the promoted bundle would not reproduce its own stored `Q`, which is
  the property a warm-start file most needs.
- **Rewriting the two v28 bundles to carry an explicit `ξ_U = 0` pin.** Rejected as
  unnecessary once `_get` was fixed: they reproduce their stored `Q` to 1e-11 read as-is,
  and rewriting them would have changed mtimes for no gain.

---

## v28.0.0 — 2026-09-02

**MAJOR, because the parameter vector and `Q` both change.** `ξ_U`, the exogenous
unskilled separation hazard, is removed from the model. It was not identified: with it in
the model `sep_rate_U = ξ_U + λ_U G(p*_U)` and `wchg_rate_U = λ_U (1 − G(p*_U))` are two
moments in three unknowns. With it gone the two rates sum to `λ_U` exactly and their ratio
gives `G(p*_U)`, so the pair identifies what it could not before. The justification for
keeping `ξ_S` while dropping `ξ_U` is the interiority asymmetry — `p*_U` is interior at
every ability, so unskilled separation is entirely endogenous, whereas `p*_S = 0` at every
ability, so skilled separation must be entirely exogenous. That argument lives in the Model
Notes under `sep_asymmetry` and is not restated in the code.

The free set goes from 23 to 22 (28 entries in `default_free_params()`, from 29), and 20
once `β_U`/`β_S` are pinned as `FIX_PARAMS` now does.

### Comparability

- **No stored `base_fc` number survives as an optimum.** At the migrated estimate `Q` goes
  from **969.804678028 to 8953.740100966**, a factor of 9.2. The stored point leaned on
  `ξ_U = 0.0038387913` to carry `sep_rate_U`; with the hazard gone the endogenous margin
  alone cannot reach the target. Every window needs re-estimating, and the shipped
  `output/estimates/estimate_base_fc_diagonalW.jls` is a warm start, not an estimate.
- **The eight stored `.jls` bundles are unchanged on disk and still load.** No struct that
  appears inside a bundle gained, lost or reordered a field: `SMMResult.params_opt` is a
  `NamedTuple` and `SMMSpec` holds `Vector{ParamSpec}`, `SimParams`, `SMMRunParams` and
  `RunProvenance` — `UnskilledParams` is in none of them. **No bundle was rewritten and no
  file in `output/` was touched**; all eight retain their original mtime and sha256.
- **A stored 23-free bundle is read correctly by the 22-free codebase.** The warm start
  maps by `(block, name)` behind a `has_field` guard (`smm_main.jl:604-620`), so the
  retired coordinate is skipped rather than shifted; `INCLUDE_PREV_OPTIMUM` compares the
  full `(block, name)` sequence (`smm_main.jl:1136`) and rejects a 23-free vector outright
  instead of misreading it positionally.

### What changed

- **Solver.** `UnskilledParams.ξ` removed, `params.jl:93`; the asymmetry documented on
  `SkilledParams.ξ`, `params.jl:150-154`. Surplus base rate is now `r+ν+λ`,
  `unskilled.jl:87`, and `solve_unskilled_surplus_on_grid!` lost its trailing `ξ`
  argument. Stationary unskilled separation `δ = λ_U G(p*)`, `unskilled.jl:276`.
- **Moment layer.** `δU_by_a = λ_U · G(p*)`, `equilibrium.jl:282`; the `wchg_rate`
  comment now scopes the `ξ`-vs-`λ` separation argument to the skilled block only,
  `equilibrium.jl:293-296`.
- **SMM layer.** `ParamSpec(:unsk, :ξ, …)` deleted from `default_free_params()`;
  `(:unsk,:ξ)` out of `REGIME_SPECIFIC_PARAMS`; both key maps lose their entry
  (`smm_params.jl:798`, and `:unsk_xi` from `_ASCII_TO_FIXED_KEY`); the three `unpack_θ`
  sites that read or wrote the field are gone.
- **Entry points and downstream.** `:unsk_xi` removed from `DEFAULT_PARAMS` and from the
  `FIX_PARAMS` valid-key list, `smm_main.jl`; `ξ` dropped from the unskilled block in
  `model_main.jl`; transition law of motion `δU = λ_U G(p*_U)`,
  `transition_solver.jl:237`; the `ξ_U` row removed from `REGIME_SPECIFIC` in
  `plots_and_tables/transition.jl`.
- **Counts.** `MCMC_main.jl:563` decremented 23 → 22.

### Measured, not asserted

All from single solves at `Nx = Np_U = Np_S` as stored in each bundle's own `SMMRunParams`,
`-t 6`, 31 active moments.

- **Purge is a faithful restriction.** `Q` under v28 at the migrated `base_fc` point =
  `8953.740100965`; `Q` under v27 with `ξ_U` pinned to `0` at the same point =
  `8953.740100966`. **Δ = 5.6e-10.** Removing the field is arithmetically identical to
  zeroing it, which is the whole claim. Two v28 solves at the same point returned
  `…100966` and `…100965`, so ~1e-9 is the threaded-reduction reproducibility floor and
  the Δ above sits at it.
- **The identity the removal is for.** At the migrated estimate `sep_rate_U =
  0.009661835148`, `wchg_rate_U = 0.152414372093`, sum `= 0.162076207241`, `λ_U =
  0.162076207241`. **`sep_rate_U + wchg_rate_U − λ_U = −2.78e-17`** — machine precision.
- **`sep_rate_U` moves by −0.00561738**, from `0.0152792` (+0.18% of the `0.0152512`
  target) to `0.00966184` (−36.65%). Its `Q` contribution goes `0.074 → 2948.74`.
- **Three moments carry 98.65% of the `Q` increase**: `sep_rate_U` (+2948.67), `ur_U`
  (+2647.26, model `0.0635675 → 0.0495346`, −22.14% of target), `theta_U` (+2280.06,
  `0.458695 → 0.401627`, −12.44%). `jfr_U` adds +104.35; the remaining 27 moments move the
  objective by less than 3 in total. 29 of 31 moments move by more than 1e-9.
- **Bundle table** — every file loads under v28, each dropping exactly `unsk:ξ`:

  | bundle | d | stored `Q` | v27 `Q` | v28 `Q` |
  |---|---|---|---|---|
  | `estimate_base_fc_diagonalW.jls` | 23→22 | 963.905904803 | 969.804678028 | 8953.740100966 |
  | `estimate_base_covid_diagonalW.jls` | 25→24 | `Inf` | 2151.030293595 | 29657.582718534 |
  | `…_backup_Q488.548412.jls` | 23→22 | 488.548411666 | 494.354151833 | 7517.182304135 |
  | `…_backup_Q490.550869.jls` | 23→22 | 490.550868777 | 493.413940847 | 7601.763544055 |
  | `…_backup_Q497.786594.jls` | 23→22 | 497.786594464 | 497.439349012 | 9588.077262126 |
  | `…_backup_Q970.339482.jls` | 23→22 | 968.097038448 | 974.385004593 | 8959.577855082 |
  | `…_postmean.jls` | 23→22 | 486.500699496 | 493.751132085 | 7567.180932035 |
  | `chains/chain_base_fc_diagonalW.jls` (`theta_best`) | 23→22 | 488.038115184 | 492.849805735 | 7545.913600610 |

- **Free set.** 23 → 22 under the stored `base_fc` pin set; the removed entry is `unsk:ξ`
  and the order of the retained 22 is unchanged. `assert_all_params_accounted` passes.
- **Betas pinned.** The 20-free spec builds and `assert_all_params_accounted` passes, but
  `Q = Inf` at the migrated `base_fc` point, from `converged_U = false` — not a degeneracy
  guard. `β_U` at the stored estimate is `0.68699` against the pinned `0.18800`, a move of
  −0.499 across a box of width 0.899, so the migrated point is simply not feasible under
  the pin. The mechanism works; the start point does not.
- **`p*_U` at the migrated estimate**: min `0.368`, median `0.423`, max `1.0`, zero
  abilities at or below 0.
- **Release gate.** `:error` = 0 and `:incomplete` = 0 on all nine changed files, by
  walking the `Meta.parseall` tree rather than trusting its exit code. Module load in
  entry-point order succeeds. `check_forwarding.jl` PASS (73 registry rows, 73 keys read).
  `check_dead_settings.jl` PASS (167 keyword arguments). `check_version_sync` in sync at
  `28.0.0`, one declaration, three readers, no duplicates.

### Settings

None added, retired or re-defaulted. `SETTINGS_REGISTRY` is untouched: `ξ_U` was a model
parameter, never a `ROYSEARCH_*` setting. The `FIX_PARAMS` key `:unsk_xi` is retired and no
longer valid — passing it now warns and is ignored, which is the documented behaviour for an
unrecognised key.

### Known gaps

- **No bundle was migrated, because none needed migrating**, and rewriting them was
  declined on purpose: the brief required all eight to keep their original mtimes, and a
  rewrite would have changed them for no gain. `code/tools/migrate_bundles.jl` was
  therefore not run.
- **`migrate_bundles.jl` has an unexercised install bug** (found, not fixed, out of scope):
  `rebuild` writes the new bundle to `.migration/<name>.jls` but installs with
  `mv(p * ".new", p)` (`:293`), a file it never creates, so the install step would fail
  after a successful rebuild. It also enumerates the four canonical windows only and cannot
  see the four backups, the postmean or the chain.
- **The stored `Q` in every bundle already failed to reproduce under v27.0.0**, before this
  change: ΔQ was 0.347 to 7.250 across the seven estimates, and `base_covid` stores `Inf`
  but recomputes to 2151.03. The stored values were written by versions 19.0.0–24.0.0 and
  the objective has moved repeatedly since. "Reproduces its stored `Q`" is therefore not a
  test this repo can currently pass for any bundle, and the v27-vs-v28 columns above are
  the comparison that isolates this change.
- **`estimate_base_fc_diagonalW_backup_Q970.339482.jls` stores `loss_opt =
  968.097038448`**, not the `970.339482` in its filename. Unexplained; the filename is
  wrong or was written from a different quantity.
- **The chain bundle cannot be re-interpreted under this version.** Its `draws` are
  `23 × 9500`, `chain` is `23 × 95 × 1000`, `G` is `28 × 23` and `se_chain` is length 23.
  Deleting the `ξ_U` column is mechanically trivial but the remaining 22 columns are a
  marginal of the *old* posterior, not a sample from the new one, and `se_chain`/`G`-based
  standard errors would not be the new model's. Left untouched; it needs a fresh chain.
- **Not verified.** No estimation was run — no annealing, no LBFGS polish, no `smm_main`
  end to end — so no re-estimated optimum exists under v28 and nothing is claimed about
  where the argmin now sits. `code/policy/` was not edited (it carries no `ξ_U` site) and
  remains broken at `policy_params.jl` on `UndefVarError: RegimeParams`, pre-existing.
  The transition and plotting edits are consistency-only and were not executed.
- **Research notes still assert `ξ_U`** and are left for the theorist: `TARGETING_NOTE.md:38`
  ("23 parameters, 28 moments — 1.2 per parameter", now 22 and 1.27) and `:111` (names the
  tenure-separation moment as what separates `α_U` from `ξ_U`). `README.md:98`,
  `SETTINGS.md:200` and `VERSION_NOTES.md:19` cite "23 free parameters" as historical
  measurements under the v19.0.0 solver and were deliberately left as written.
- **Historical counts left in place.** `MCMC_main.jl:276` and `:1085` say "23 free
  coordinates" and "22 of 23 parameters", but both report measurements taken on the stored
  23-free bundle; decrementing them would falsify the record. Only `:563`, which describes
  the current estimation problem, was changed.

### Tried and rejected

- **Rewriting the bundles into 22-free form and installing them.** Rejected: it is
  unnecessary (nothing in a bundle changes layout — measured, all eight deserialise
  untouched) and it conflicts with the requirement that their mtimes be unchanged. The
  load-time coordinate drop already in `smm_main.jl` does the same work without writing.
- **Adding a runtime guard that errors when `p*_U = 0` at any ability.** Rejected on the
  user's reasoning: `ur_U` and `sep_rate_U` both carry high weight, so an ability at
  `p*_U = 0` collapses unemployment and is punished heavily by the criterion itself. The
  corner is not reachable under estimation, so the guard is defensive code for a state the
  objective already excludes. Measured at the migrated estimate: min `p*_U = 0.368`, no
  ability within 0.36 of the corner.
- **Changing the poaching-outflow rate from `1−Γ(p^oj)` to `1−Γ(p_j)`** (the third OJS
  site, `equilibrium.jl:322`/`:345`). Withdrawn before any edit: the case for it rested on
  two adjacent code comments rather than on the model's own equations, which is not a
  sufficient basis for changing a working solver. The question is open and belongs with the
  Model Notes. The site is byte-identical to v27.0.0 — verified by inverting this version's
  four `equilibrium.jl` edits and confirming that no added or removed line mentions
  `Gamma_poj`, `flow_ij`, `poj_j` or `pcut_index`.

---

## v27.0.0 — 2026-09-02

**MAJOR, because `Q` moves.** The moment layer read the on-the-job-search cutoff `p^oj`
through two hard `1{p < p^oj}` indicators — one assigning a cell's employed mass wholly to
the poached or the non-poached wage surface, one gating which mass enters the E-to-E flow —
while the stationary solve has smoothed the same margin with `_soft_oj_weight` since
v19.x. Those were two different models of one margin. Both indicators are replaced by the
covered fraction `s_j`, so a cell straddling `p^oj` now contributes to both surfaces in
proportion. At the shipped `base_fc` estimate `Q` moves from **963.905904803 to
969.804678028**, so the theorist has to be told: the criterion is a different function.

The reason is continuity, not node-level accuracy. Within a straddling cell the correct
wages are `wS1` below `p^oj` and `wS0` above, and using the node's tabulated surfaces for
both sub-intervals is first-order accurate in the cell width — the same order as every
other quadrature in the solver. The hard assignment was equally `O(dp)` in aggregate *and*
discontinuous, and a criterion with a step admits no finite difference, no gradient and no
`J⁻¹` across it. Continuity is a property the estimator requires; node accuracy at a
single node is not.

### Comparability

- **No pre-v27 `base_fc` number is comparable.** `Q` at the shipped estimate moves by
  +5.899, and 15 of 31 moments move — `ee_rate_S` by 7.7%, `emp_cm3_S` by 4.9%,
  `emp_var_S` by 1.4%. `output/estimates/estimate_base_fc_diagonalW.jls` records
  `Q = 963.905905` against the criterion this version replaced, so **the stored optimum is
  superseded as an optimum**: it is a point on the old surface, not the argmin of the new
  one. Every window needs re-estimating; only `base_fc` was measured here.
- **The equilibrium itself is untouched.** All 16 rate and stock moments — `jfr_U/S`,
  `ur_U/S`, `sep_rate_U/S`, `wchg_rate_U/S`, the four `usurv_S`, `theta_U/S`,
  `skilled_share`, `training_share` — are bit-identical before and after. The change is
  confined to the moment layer, and the measured moves are the wage-distribution and EE
  moments only.
- **Stored bundles load unchanged.** No struct gained, lost or reordered a field; all
  seven `.jls` in `output/estimates/` deserialise (`Q` = 963.905905, 488.548412,
  490.550869, 497.786594, 968.097038, 486.500699, and `Inf` for `base_covid`).
- **The four unskilled wage quantiles move at ≤ 7.1e-8 relative even though nothing
  unskilled changed.** `_wage_density` builds one histogram grid from
  `quantile(all_wages, 0.002/0.998)` over the pooled U and S wage *list*, so changing the
  skilled entries moves the shared grid. Pre-existing coupling, newly visible.

### What changed

**Moment layer.** `compute_equilibrium_objects` (`solver/equilibrium.jl:197-219`) splits
each cell's employed mass as `s_j·m` onto `wages_S1` and `(1−s_j)·m` onto `wages_S0`,
skipping a side whose weight is zero. Both surfaces are written under the same `p*_S` test
(`equilibrium.jl:160-168`), so their NaN patterns coincide and the split conserves the
cell's mass exactly. The E-to-E block (`equilibrium.jl:330-347`) weights the poaching flow
by the same `s_j` instead of gating on the comparison; `ee_step_S` inherits the weight
through `flow_ij`, so the step and the rate are computed off one moving mass.

**The same assignment, wherever it was duplicated.** The hard comparison appeared at three
more sites, all now reading `_soft_oj_weight`:
`transition_solver.jl:351` (the `e_S` poaching outflow along the path — the transition
reconstructs the density itself and carried its own copy of the margin),
`policy_solver.jl:63-72` (mean skilled wage) and
`plots_and_tables/model.jl:405-417` (the realised-wage surface behind the skill-premium
figure, now the `s`-weighted mix of the two surfaces, so the figure shows the wage the
moments are computed from).

**New tools.** `tools/ojs_continuity_probe.jl` has four modes — `moments`, `jump`,
`crossing`, `jacobian` — and produced every number below; `smoke/smoke_transition.jl` is
the named gate for "the transition path still runs on the shipped estimate".

### Measured, not asserted

**(a) Exact backward compatibility where no cell straddles.** With `p^oj` snapped to p-grid
nodes after the solve, every covered fraction is exactly 0 or 1 and the new code must
reproduce the old bit for bit. It does: `Q` = **963.90590480318895 under both trees**, and
**0 of 31 moments differ in a single bit** (`ojs_moments_base_fc_{before,after}.csv`). Run
at `-t 1`, with the snapped solve repeated in-process and confirmed bit-identical, so this
is a claim about the code and not about thread scheduling.

**(b) The ΔQ = 0.36 jump is 15.9× smaller and no longer the dominant step.** Same protocol
as v26.0.0's tolerance-invariance table — steps in unconstrained `t`, a line fitted through
the three same-side points at `h = 3e-8, 1e-7, 3e-7`, the jump being the far-side point's
excess over it. The crossing sits at `+1e-6` on the `λ_S` ray and `−1e-6` on the `μ_S` ray,
both at `aS` node 63 (`ojs_jump_tolerance_base_fc_{before,after}.csv`):

| rung | `λ_S` before | `μ_S` before | `λ_S` after | `μ_S` after |
|---|---|---|---|---|
| shipped 1e-4 (`conv_streak` 4, `global_B` 8) | +0.360018261 | +0.359952019 | **−0.022636373** | **−0.022676011** |
| uniform 1e-4 | +0.360031221 | +0.359967662 | −0.022717504 | −0.022764761 |
| uniform 1e-6 | +0.360016315 | +0.359947984 | −0.022615963 | −0.022649194 |
| uniform 1e-8 | +0.360014977 | +0.359947766 | −0.022615142 | −0.022647874 |

The before column reproduces v26.0.0's recorded numbers rung by rung (+0.360015464 /
+0.359947420 at the shipped rung) to **≤ 7.8e-7 at the two tight rungs** — 2.1e-7 and 5.7e-7
at 1e-8 — and to 2.8e-6 / 4.6e-6 at the shipped rung. The two `uniform 1e-4` cells are the
loosest agreement, 1.37e-5 (`λ_S`) and 2.29e-5 (`μ_S`), which is where it should be: at a
1e-4 residual the three same-side points carry solve noise of that scale, and the fitted
slope moves with them (−468.5 here against the recorded −451.7). So the harness reproduces
the earlier one wherever the earlier one is itself precise.

**The residual is still tolerance-invariant, so there is a third site, and it is
identified.** `PROBE_MODE=crossing` (`ojs_crossing_base_fc_{before,after}.csv`) evaluates
the two bracketing points at `tol_global = 1e-8` and attributes the step:

| | before | after |
|---|---|---|
| ΔQ across the crossing | +0.359701220 | −0.022010391 |
| `ee_rate_S` step | −1.207e−2 | +3.905e−4 |
| `emp_cm3_S` step | +1.061e−2 | +5.123e−6 |
| `emp_var_S` step | −2.487e−3 | −1.264e−6 |
| `overlap_SltU` step | −8.215e−4 | below 1e−9 |

The wage moments' steps fell by 1970–2070×, `ee_rate_S`'s by 31×, and the residual `ΔQ` is
carried almost entirely by `ee_rate_S`. Its source is the one remaining hard read of
`p^oj`: `Gamma_poj = pre.Γvals[pcut_index(pg, poj_j)]` (`equilibrium.jl:314`) evaluates the
offer CDF at the cutoff's *grid node*, so when row 63's node index moves 8 → 7,
**`Γ(p^oj)` steps 0.0950795 → 0.0854336, −10.15% relative.** `Γ` is not small at these
cutoffs — `a_Γ = 0.931 < 1` puts unbounded density at `p → 0`, so a single cell near
`p^oj ≈ 0.0077` carries 10% of `Γ` itself. Not fixed here, and not a code question: the
stationary solve prices the poaching outflow at `1−Γ(p_j)`, the worker's *own* match
quality (`skilled.jl:373`), while the moment layer prices it at `1−Γ(p^oj)`. Those are
different flows, and which one `ee_rate_S` should target is the theorist's call. Whichever
it is, the read should be continuous — `cdf(Beta(a_Γ, b_Γ), ·)` is available in closed form
and `grids.jl:292` already prefers the exact CDF to a tabulated node value.

**Two things that are NOT the residual, both measured rather than argued.** The histogram
grid is continuous across the crossing: its endpoints shift by −3.7e-7 and −8.8e-5
absolute on 166.6 and 2744.0, *identically before and after*, so the pooled-quantile grid
is not a step site. And the number of straddling cells does not change across a crossing —
**58 of 14400 either side**, carrying `1.03e-3` of employed skilled mass against
`agg_eS = 0.2817` (0.37%) — because as `p^oj` crosses a node one cell stops straddling and
its neighbour starts. So the split cannot introduce a step by changing how many entries the
wage list holds.

**(c) A solve-precision noise floor is real, and it is a second effect.** Per-column
relative L2 spread of the whitened Jacobian over `h ∈ {3e-7, 1e-6, 3e-6}`, central
differences, at both tolerances and on both trees
(`ojs_jacobian_tol1e-0{4,8}_base_fc_{before,after}.csv`). On the 16 columns that pass a 5%
stability test, the spread collapses when the tolerance is tightened:

| | median spread | max spread | columns failing 5% |
|---|---|---|---|
| before, `tol_global` 1e-4 | 1.063e−3 | 3.531e−2 | 7 of 23 |
| before, `tol_global` 1e-8 | 6.511e−8 | 1.953e−6 | 5 of 23 |
| after, `tol_global` 1e-4 | 1.056e−3 | 3.337e−2 | 7 of 23 |
| after, `tol_global` 1e-8 | 7.134e−8 | 9.874e−7 | 5 of 23 |

Column by column, the median of `spread(1e-8)/spread(1e-4)` is **5.08e−5 after and 5.72e−5
before** (IQR 3.8–8.1e-5 and 3.6–7.8e-5; the ratio of the medians in the table is 6.75e-5
and 6.12e-5). That is the tolerance ratio 1e-4 to within a factor of two — the scaling a
noise floor proportional to the solve residual predicts, not a coincidence of magnitude, and
the same on both trees. So the answer to the question this test was
built to settle is: **the spread at the shipped tolerance is a noise floor that scales with
`tol_global`, it lives underneath the threshold rather than being caused by it, and it
survives the fix unchanged.** The threshold was not the whole story; there are two separate
effects. Two columns, `b_S` (a null column, norm 1e−11) and `σ_S`, fail at 1e-4 and pass at
1e-8, which is what a pure noise-floor failure looks like. Five fail at either tolerance:
`a_ℓ`, `k_S`, `β_S`, `λ_S`, `μ_S`.

**(d) `cond(J)` and the columns the jump was contaminating.** At the fixed `h = 1e-6`:

| | before | after |
|---|---|---|
| `Q` at the estimate | 963.905904803 | 969.804678028 |
| `cond(J)`, `tol_global` 1e-4 | 3.525256e+17 | 7.215254e+16 |
| `σ_max`, `tol_global` 1e-4 | 4.272382e+04 | 4.467134e+03 |
| `cond(J)`, `tol_global` 1e-8 | 1.138215e+18 | 8.862910e+16 |
| `σ_max`, `tol_global` 1e-8 | 4.272387e+04 | 4.462080e+03 |
| numerical rank | 22 of 23 | 22 of 23 |

`σ_max` falls by 9.6× because the jumped columns stop dominating it: `μ_S`'s column norm
goes **39022.3 → 547.8** (71×), `λ_S`'s **17376.7 → 246.8** (70×), and `k_S`'s mean-over-`h`
norm 42361 → 619 (68×). `cond(J)` is still ~1e17 and the rank is still deficient — the null
`b_S` column and `a_ℓ` set that, and neither is touched here.

The moments that moved at the estimate, largest first: `ee_rate_S` 7.705e-2, `emp_cm3_S`
4.892e-2, `emp_var_S` 1.434e-2, `overlap_SltU` 8.334e-3, `overlap_UgtS` 1.790e-3,
`p25_wage_S` 3.913e-4, `mean_wage_S` 3.674e-4, `p50_wage_S` 9.906e-5, `emp_cm3_U`
3.795e-5, `p75_wage_S` 1.966e-5, `emp_var_U` 2.133e-6, and the four unskilled wage
quantiles at ≤ 7.1e-8. The six named as the likely movers in v26.0.0's known-gaps section
are the six largest.

**(e) The transition path runs; the policy layer still does not load.**
`smoke/smoke_transition.jl` at `Nx = 60` on the `base_fc` estimate with a −3% shift in `A`
as the terminal regime: both steady states solve, `solve_transition` completes with every
series finite, before and after. Capped at 6 outer iterations and stopped at
`final_dist = 1.955e-3` against `tol = 1e-4`, so this establishes that the path runs and
stays finite, not that it converged. Every reported series is identical before and after to
six digits except the terminal `mean_wage_S`, **998.341779 → 998.338129** (3.7e-6
relative) — the whole effect of softening the path's poaching outflow, which is small
because straddling cells carry 0.37% of the employed skilled mass. `code/policy/` fails to
load at `policy/policy_params.jl` on `UndefVarError: RegimeParams`, identically before and
after; that break predates this change, so **the `policy_solver.jl` edit is unverified by
execution.**

**Gate.** `Meta.parseall` on the seven changed or added files: 0 `:error`/`:incomplete`
nodes, followed by a real module load of solver + smm + candidates + transition in the
entry point's own order, with every name the edits call confirmed to resolve.
`check_forwarding.jl`: PASS (73 registry rows, 73 keys read, no name disagreement).
`check_version_sync`: in sync at 27.0.0, three readers, no duplicate declaration.
`smm_main.jl` end to end against real `data/derived/` in a scratch tree —
`INIT_MODE = :default`, `SA_MAX_ITER = 6`, `DE_MAX_ITER = 1`, `DE_POP_SIZE = 8`, 21 free
parameters, 35 active moments — built the spec, measured the proposal over 504 width
solves, completed SA on the `sa-stop` branch at `Q = 2.260867e+06` and entered the LBFGS
polish, descending to 386192.35 by its second gradient. **The polish was stopped there
deliberately**: the gate is that the path runs and the optimiser descends under the new
criterion, not that a from-defaults run converges. No live `output/` file was written.

### Settings

None added, retired or re-defaulted. The tight-tolerance rungs used `tol_global` 1e-6/1e-8
with `conv_streak = 2`, `global_B = 0`, `maxit_global = 600` as a **diagnostic override
inside the probe**; the shipped defaults are unchanged. The probe's and smoke test's own
env knobs (`PROBE_MODE`, `PROBE_TOL`, `PROBE_TAG`, `RSROOT`, `SMOKE_NX`, `SMOKE_MAXIT`) are
tool arguments, not run settings, and carry no `SETTINGS.md` row — the same convention
`plateau_probe.jl` and `rank_diagnostic.jl` follow.

### Known gaps

- **The residual `ΔQ` = −0.0226 step is not fixed**, only localised to `Γ(p^oj)` read at a
  grid node (`equilibrium.jl:314`) and measured at −10.15% relative on `Γ` itself. The
  criterion is still not differentiable across a `p^oj` node crossing. See above for why
  this is a specification question and not a code move.
- **Only `base_fc` was measured, and only at the stored optimum.** The other three windows
  were not solved under v27.0.0. The straddling-cell mass share (0.37%) is a `base_fc`
  number; at a point where more mass sits in straddling cells the change moves `Q` more.
- **The stored optima are superseded as optima and were not re-estimated.** Every number in
  `output/estimates/` and every table built from them belongs to the pre-v27 criterion.
- **`policy_solver.jl` was edited but not executed** — the layer does not load, for reasons
  that predate this change (it also calls `solve_model` with a five-argument signature the
  solver does not have, and reads equilibrium fields
  `compute_equilibrium_objects` does not return).
- **`plots_and_tables/model.jl` was edited but the figure was not rendered.** The
  `s`-weighted realised-wage surface is verified only by the parse gate and by agreeing
  with the old expression at `s ∈ {0,1}` by construction.
- **The MCMC layer was not exercised.** It reads its moment keys from the spec and does not
  touch the OJS margin, but no chain was run under v27.0.0 — and since `Q` moved, the
  stored chain in `output/smm/mcmc_chain_base_fc_diagonalW.jls` is a sample from the old
  criterion.
- **`tools/rank_diagnostic.jl` does not load.** It includes the solver and SMM files without
  `smm/settings.jl` or `smm/bundle.jl`, so it dies on `UndefVarError: env_setting` at
  `smm/smm.jl:2092`. Pre-existing and untouched here; the probe added in this version
  carries its own copy of the whitening convention (`σ̂` from `spec.W`, columns by `dθ/dt`)
  rather than depending on it.

### Tried and rejected

**Keeping the hard assignment because it is more accurate at the node.** The argument was
that within a straddling cell the true wages are `wS1` below `p^oj` and `wS0` above, so
splitting the node's mass across two surfaces evaluated at the *same* tabulated wage is an
approximation the hard rule avoids — and that the soft weight is measurably worse on the
solver side, 1.7–1.8× in RMS with a one-signed bias ~60× larger (recorded under v19.x, see
`VERSION_NOTES.md` §R1). **Rejected, and the reason is not a tie-break on accuracy.** The
split is first-order accurate in the cell width, which is the same order as every other
quadrature in the solver, so nothing is lost relative to the surrounding discretisation.
The hard assignment is `O(dp)` in aggregate *too* — and additionally discontinuous, which
is not an accuracy defect but a well-posedness one: SMM, its standard errors and every
gradient step require `Q` to be differentiable, and no finite difference is defined across
a step of any size. Node-level accuracy is a quantity the estimator can absorb; continuity
is a property it requires. The measurement above is the closing argument: the moment-level
steps fell by 31–2070× and `σ_max` by 9.6×, at the cost of a first-order error inside 58 of
14400 cells holding 0.37% of the skilled employed mass. **Do not re-propose the hard rule
on accuracy grounds.**

**Suspecting the pooled wage-histogram grid as the residual site.** Because the split can
add an entry to `wages_S0` with near-zero mass while `_wage_density`'s grid comes from an
*unweighted* `quantile(all_wages, ·)`, a zero-mass entry appearing at `s_j → 1⁻` would move
the grid discontinuously. **Measured and rejected**: across the crossing the grid endpoints
shift by −3.7e-7 and −8.8e-5 absolute (on 166.6 and 2744.0), the same to three digits before
and after, and the straddling-cell count is 58 on both sides — a crossing swaps which cell
straddles rather than changing how many do. The grid is not a step site, before or after.

---

## v26.0.0 — 2026-09-02

**MAJOR, because an equilibrium object is redefined.** The cross-market drain `d(aU,aS)` was
a hard indicator `1{U_S^(1)(aU) > U_S^(0)(aS)}` on the ability grid; it is now the population
fraction of cell `(aU,aS)` that abandons skill, computed from the exact crossing `aS*(aU)` of
the two value functions. It is the same class of object as the training frontier `τ_T`, fixed
the same way, and it was the last hard 0/1 indicator on the ability grid in the solver. The
theorist has to be told because `d` is exported in the equilibrium NamedTuple and enters the
trained mass, the non-draining column masses, the augmented unskilled seeker pool and the
skilled unemployment composition.

At the shipped `base_fc` estimate the change moves nothing: `Q` is **963.905904803 before and
after**, and all 31 active moments agree to every printed digit. That is not a null result —
it is a measurement of where the drain margin sits relative to the training frontier, and it
is reported below.

### Comparability

- **Every stored number is still comparable at the shipped `base_fc` estimate**, where `Q` and
  all 31 moments are unchanged to 12 significant digits. No stored output is superseded.
- **But the object's definition changed, so this is not a guarantee at other points.** The
  drain crossing carries zero trained mass at this estimate; at a parameter point where it
  does not, the fractional cells contribute and results will differ from any pre-v26 run.
  Only `base_fc` was measured.
- **Stored bundles load unchanged.** No struct gained, lost or reordered a field. Verified:
  `estimate_base_fc_diagonalW.jls` (stored `Q` = 963.905905, 23 free) and
  `estimate_base_covid_diagonalW.jls` (stored `Q` = Inf, 25 free) both deserialise.
- `dzero_column_mass` is renamed `nondrain_column_mass`. It is internal to the solver with one
  call site; no bundle or output file names it.

### What changed

**Producer.** `drain_fraction!` (`solver/skilled.jl:132-204`) is the single definition of the
margin: it verifies `U_S^(0)` is increasing in `aS` at runtime, locates `aS*(aU)` by linear
interpolation of the crossing, and sets `d[i,j] = clamp((aS* − xlo[j])/w, 0, 1)` — the fraction
of node `j`'s midpoint interval lying *below* `aS*`, since abandonment is optimal on a lower
interval in `aS`. The fully-included and fully-excluded rows keep their exact 0 and 1 branches,
and non-monotone `U_S^(0)` keeps the hard indicator rather than interpolating a crossing that
would not be unique (`skilled.jl:176`).

**The interval definition is now shared.** `node_midpoint_intervals` (`solver/grids.jl:84-101`)
is the one place the ability nodes' midpoint intervals are constructed; the training frontier
reads it (`unskilled.jl:204`) instead of rebuilding it inline. Both continuous frontiers now
express a covered fraction against the same intervals.

**Consumers re-expressed as splits, not thresholds.** Two sites picked a branch with `d > 0.5`,
which would have discarded the fractional boundary node and put the step function back:

- `solver/solver.jl:161-169` — the U-market carry is now `d·m_S`: only the draining fraction
  seeks in the unskilled market and all of its mass is unemployed there. Identical to the old
  expression at `d ∈ {0,1}`.
- `solver/equilibrium.jl:58-74` — the skilled unemployed stock is now built as two pieces,
  `uS_drain = d·m_S` and `uS_stay = (1−d)·û(aS)·m_S`, summing to `uS_mat`. They are kept apart
  because the flow block applies a different exit hazard to each, so `uS_d1mass` /
  `uS_d0mass` (`equilibrium.jl:246-252`) now read the two pieces directly. Deriving one of them
  as `d·uS_mat`, as before, double-counts the drain once `d` is fractional and breaks the
  partition identity.

**The transition path reads the same producer.** `_d_matrix`
(`transition/transition_solver.jl:411-427`) now calls `drain_fraction!`, and the forward mass
step calls `_d_matrix` rather than rebuilding the indicator inline
(`transition_solver.jl:262-268`); the `z₀` initialisation splits `u_S` the same way as
`compute_equilibrium_objects` (`transition_solver.jl:57-63`). The path and its two steady states
therefore share one definition of the margin.

**Consumers verified linear in `d` and left alone:** `_mS_from_t` (`solver.jl:71-79`, smooth
rational in `d`), `nondrain_column_mass` (`skilled.jl:343-349`), the early-iteration free-entry
fallback (`skilled.jl:425`), `mcol0` in `compute_equilibrium_objects` (`equilibrium.jl:73`), the
augmented seeker pool `ueff = u_U + d·u_S` (`unskilled.jl:342`), the transition carry
(`transition_solver.jl:176, 397`) and the reported drain profile
(`plots_and_tables/model.jl:212`). `code/policy/` does not read `d` at all.

### Measured, not asserted

All at the shipped `base_fc` estimate, `Nx = Np_U = Np_S = 120`, 23 free parameters, 31 active
moments; "before" is the same tree with the five solver files reverted (byte-identical to the
pre-edit sources).

**Monotonicity, the property the exact crossing rests on.** `U_S^(0)` is increasing in `aS` at
**119/119** increments, smallest increment **1.0683e+02** on a range of 26.51 to 249870.58. The
runtime guard therefore never fires here. `U_S^(1)` is *not* strictly monotone in `aU` — 100/119
increments positive, minimum exactly 0.0 — which is why the crossing is inverted in `aS` and
not in `aU`.

**The margin is now fractional, and it carries no mass.**

| | before | after |
|---|---|---|
| `d` cells at 1 / at 0 / fractional | 6624 / 7776 / 0 | 6554 / 7726 / **120** |
| `τ_T` cells at 1 / at 0 / fractional | 4960 / 9349 / 91 | 4960 / 9349 / 91 |
| cells with both margins fractional | 0 | 0 |

120 fractional cells is exactly one per `aU` row, as the construction implies. But the trained
mass in those 120 cells is **0.000000e+00** against a total `m_S` of 2.892089e-01, the training
stock in them is 0.000000e+00 against 4.204239e-02, and `τ_T` is 0 in every one of them. In the
91 rows where both margins cut, the training frontier sits **16 to 36 aS-nodes above** the drain
crossing: at this estimate nobody who trains ever prefers to abandon skill, so `d` is 0 wherever
there is anything to drain. This is why `Q` does not move, and it is the reason the fix is
correct but inert here.

**(a) The step ladder is unchanged. The ΔQ = 0.359 jump is not the drain.** `dQ/h` from a
forward difference at the shipped estimate (`output/logs/drain_continuity_step_ladder_base_fc.csv`):

| param | 3e-8 | 1e-7 | 3e-7 | 1e-6 | 3e-6 | 1e-5 |
|---|---|---|---|---|---|---|
| `μ_S` | 4.510e+02 | 4.586e+02 | 4.608e+02 | 4.618e+02 | 4.629e+02 | 4.657e+02 |
| `λ_S` | −4.506e+02 | −4.488e+02 | −4.519e+02 | **3.596e+05** | 1.194e+05 | 3.532e+04 |
| `β_S` | 2.006e+02 | 1.964e+02 | 1.985e+02 | 1.993e+02 | 1.997e+02 | 2.004e+02 |
| `k_S` | −2.780e+02 | −2.779e+02 | −2.774e+02 | −2.780e+02 | **1.197e+05** | 3.561e+04 |

Before and after agree to 4 significant digits in every cell. `μ_S` and `β_S` are stable across
the whole ladder at this point; `λ_S` breaks at `h = 1e-6` and `k_S` at `h = 3e-6`, and
`3.596e+05 × 1e-6 = 0.3596` is the jump.

**(b) A fixed step is still not viable, and by the same margin.** Per-column relative L2 spread
of the whitened Jacobian column over `h ∈ {3e-7, 1e-6, 3e-6}`, failing above 5%
(`output/logs/drain_continuity_jacobian_stability_base_fc.csv`): **5 of 23 columns fail before
and 5 after, the same five.**

| param | spread before | spread after | column norm |
|---|---|---|---|
| `a_ℓ` | 17102.34% | 17102.38% | 5.99e+02 |
| `b_S` | 160.24% | 55.85% | 1.0e−10 (null column) |
| `β_S` | 8.04% | 8.03% | 6.34e+01 |
| `λ_S` | 100.13% | 100.13% | 3.48e+04 |
| `k_S` | 37847.84% | 37847.80% | 6.80e+02 |

The other 18 columns' norms agree before/after to ≤ **2.49e−5** relative. `b_S` is a zero column
(norm 1e−10), so its spread is noise on nothing, not a derivative.

**(c) `cond(J)` at the fixed `h = 1e-6`: 3.0046e+17 → 3.2423e+17.** Both are dominated by the
jumped columns and by the null `b_S` column, so the change is not informative; `σ_max` is
3.4808e+04 in both and numerical rank is 22 of 23 in both.

**(d) `Q` and the five moments that carry it.** `Q` = **963.905904803** before and after. At the
neighbouring probe points the two trees differ only in the 12th significant digit
(`λ_S + 3e-7`: 963.905769246 vs 963.905769245). The five largest contributors are unchanged:

| moment | target | model | contribution | share of Q |
|---|---|---|---|---|
| `usurv53_S` | 0.099477 | 0.036598 | 368.8871 | 38.27% |
| `training_share` | 0.071679 | 0.042042 | 203.5603 | 21.12% |
| `jfr_S` | 0.320369 | 0.267215 | 120.7843 | 12.53% |
| `usurv5_S` | 0.695207 | 0.731944 | 114.9332 | 11.92% |
| `usurv27_S` | 0.232638 | 0.185430 | 112.8491 | 11.71% |

**(e) The transition solver runs through the shared producer.** At `Nx = 60` on the `base_fc`
estimate with a −3% shift in `A` as the terminal regime: both steady states solve (`ok = true`,
60 fractional `d` cells each, one per `aU` row), and `solve_transition` completes with every
series finite — `θ_U` 0.4579 → 0.4937 along the path, `ur_S ∈ [0.02620, 0.03810]`,
`training_share ∈ [0.03802, 0.04202]`. The run was capped at 6 outer iterations for the smoke
test and stopped at `final_dist = 1.992e-3` against `tol = 1e-4`, so this establishes that the
path runs and stays finite, not that it converged. **`code/policy/` does not load**, at
`policy/policy_params.jl:109`, on `UndefVarError: RegimeParams` — a type that exists nowhere in
`code/`. That break predates this change and is untouched by it: the policy layer also calls
`solve_model` with a five-argument signature the solver does not have, and reads equilibrium
fields (`obj.wx`, `obj.tauT`, `obj.eS_mat`, `obj.UU`) that `compute_equilibrium_objects` does
not return.

**Gate.** `Meta.parseall` on the six edited files: 0 `:error`/`:incomplete` nodes.
`check_forwarding.jl`: PASS (73 registry rows, 73 keys read, no name disagreement).
`check_version_sync`: in sync at 26.0.0, three readers, no duplicate declaration.

### Settings

None added, retired or re-defaulted. The tight-tolerance runs below used
`tol_global = 1e-9`, `conv_streak = 2`, `global_B = 0`, `maxit_global = 400` as a **diagnostic
override in a probe script**; the shipped defaults (`tol_global = 1e-4`, `conv_streak = 4`,
`global_B = 8`, `global_K = 4`, `maxit_global = 20`) are unchanged.

### Known gaps

- **The ΔQ = 0.359 discontinuity is still there, and it is now localised but not fixed.** See
  the rejected section for what it is not. What it is: at a tight global tolerance the
  *equilibrium* moves continuously across the crossing while the *moment map* does not. Across
  `λ_S + 3e-7 → +1e-6`, both solves converged to residual ≈ 7e-10 and 1.7e-10, and the
  equilibrium objects moved by `θ_U` 3.9e-9, `θ_S` 5.6e-8, `p*_S` exactly 0, `m_S` 1.3e-7,
  `û(aS)` 1.8e-7, `e_S(aS,p)` 4.9e-6, `d` 6.8e-6, `p^oj` 7.1e-6 (all relative) — while
  `emp_var_S` moved 2.5e-3, `emp_cm3_S` 1.1e-2 and `ee_rate_S` 1.2e-2 relative, three to four
  orders of magnitude more than any state. The discrete event is one row: **`aS` node 63's
  `p^oj` crosses p-grid node 8 → 7** (0.007726780356 → 0.007726367893). The suspect sites are
  the two hard reads of `p^oj` in the moment layer — the wage-density split `pg[jp] < poj_j`
  (`equilibrium.jl:199-207`) which assigns a node's employed mass wholly to the poached or the
  non-poached wage surface, and `Γvals[pcut_index(pg, poj_j)]` with the same comparison in the
  E-to-E block (`equilibrium.jl:304, 323`). The solver's own free-entry path already smooths
  this margin with `_soft_oj_weight` (`skilled.jl:462, 470, 497`); the moment layer never got
  the same treatment. **Not fixed here**: whether the moment layer should carry the soft weight
  is an accuracy-versus-continuity trade — `skilled.jl:425-435` records that the soft weight is
  1.7–1.8× worse in RMS on the solver side — and that is a specification decision, not a code
  move to guess at.
- **Only `base_fc` was measured.** The other three windows were not solved under v26.0.0, so
  the claim that the drain crossing carries no trained mass is a `base_fc` claim. Nothing rules
  out a window or a candidate point where it carries mass — that is precisely the case in which
  this fix changes results.
- **`d` and `τ_T` are both fractions of the same `aS` interval, and the code multiplies them.**
  Where a cell were cut by both margins, `τ_T·(1−d)` treats the two sub-intervals as independent
  when both are upper/lower intervals in the same coordinate, for which the exact
  trained-and-not-draining fraction is `min(τ_T, 1−d)`. Zero cells are cut by both at the
  `base_fc` estimate, so nothing is wrong today; if a future point puts the two frontiers in the
  same cell, that product is an approximation of ~one cell's mass.
- **The transition carries one `u_S` per cell, so a fractional `d` mixes two populations there.**
  `uc.duS_carry = d·path.uS` (`transition_solver.jl:176, 397`) equals `d·m_S` only at `d ∈ {0,1}`;
  at a fractional `d` it understates the drain carry by the factor `(1−d)(1−û)`, and the branched
  outflow `(ν + (1−d)f_S(1−Γ) + d f_U)·u_S` applies a blended hazard to a mixed cell. This is
  bounded to the ≤ `Nx` cells the margin cuts (120 of 14400 at `Nx` = 120), and making it exact
  means splitting the `u_S` state in `TransitionPath` — a respecification, not a patch.
- **The MCMC layer was not exercised.** It reads its moment keys from the spec and does not touch
  `d`, but no chain was run under v26.0.0.

### Tried and rejected

**The stall branch, as the source of the jump.** The hypothesis was that `solve_model!` returns
a stalled iterate (`solver.jl:191-195`) rather than the fixed point, and that two nearby `θ`
stall at different iterations. **Measured and rejected**: at the shipped estimate, 25 solves —
base plus `μ_S`, `λ_S`, `β_S`, `k_S` at `h = ±1e-7, ±1e-6, ±3e-6` — *every one* terminated on
the converged branch with all three layer flags true. The stall branch never fired, and neither
did `maxit`. The trace is in the diagnostic, not in the repo.

**The termination tolerance, as the source of the jump.** The residual the global loop stops at
does jump across the crossing — 2.4e-5 on the lower branch versus 1.6e-7 on the upper one, with
`n_global` 9-12 either side — which made an early-exit artefact the obvious next suspect. The
arithmetic case for it is strong: `0.359 / (2·1e-6) = 1.795e+05` reproduces the anomalous
"derivative" exactly, which is the signature of differencing an inexact function
(`err = ε_Q / h`), and an ε_Q of 0.359 corresponds to a uniform relative moment error of only
~8.4e-6 — well inside what a solve at `tol_global = 1e-4` can leave behind.

**Measured and rejected.** The jump is invariant to solve precision over four orders of
magnitude. Each rung below holds every setting fixed and varies only `tol_global`
(`conv_streak = 2`, `global_B = 0`, `maxit_global = 600`), with the shipped configuration as a
reference row; "jump" is the excess of the far-side `Q` over the linear trend fitted through the
three same-side points (`output/logs/drain_jump_tolerance_invariance_base_fc.csv`):

| rung | reached | `λ_S` jump | `μ_S` jump | local slope `μ_S` |
|---|---|---|---|---|
| shipped 1e-4 (`conv_streak` 4, `global_B` 8) | yes | +0.360015464 | +0.359947420 | 461.895 |
| uniform 1e-4 | yes | +0.360017511 | +0.359944793 | 462.175 |
| uniform 1e-6 | yes | +0.360015538 | +0.359948308 | 462.184 |
| uniform 1e-8 | yes | +0.360015187 | +0.359948337 | 462.182 |

Tightening the tolerance by 1e4 changes the jump by **6.5e-6 relative** (`λ_S`) and **9.8e-6
relative** (`μ_S`). A solve-precision artefact of scale `tol_global` predicts the 1e-8 rung at
1e-4 of the 1e-4 rung — 0.0000360 — and it measures 0.3600152, a factor of **10000** off. Every
one of the 32 solves reached its target residual and terminated on the converged branch; none
stalled or hit `maxit`. At the 1e-8 rung the three same-side `μ_S` points converge at
`n_global` = 64/66/66 with residuals 9.26e-9/9.55e-9/9.37e-9 — identical to within 3% — and the
far-side point at residual 1.16e-9, so both sides are at their fixed points and the jump is
still 0.3599.

Two further readings point the same way. The local slope is stable to 4-6 significant figures
across every rung (`μ_S` 461.9-462.2), so the criterion is smooth *except* across the surface —
noise at scale ε_Q would degrade the slope too. And the displacement across the crossing is
concentrated, not uniform: at `tol_global = 1e-9`, `emp_var_S` moves 2.5e-3, `emp_cm3_S` 1.1e-2
and `ee_rate_S` 1.2e-2 relative, 300-1400× the ~8.4e-6 a uniform tolerance-scale error implies,
while every state variable moves ≤ 7e-6.

**Inverting the drain crossing in `aU` instead of `aS`.** Splitting the `aU` interval for each
`aS` column would make `d` and `τ_T` fractions of *different* coordinates, which would make
their product the exact area fraction of the intersection. Rejected on measurement: `U_S^(1)` is
only weakly monotone in `aU` (100/119 increments positive, 19 exactly flat), so the crossing in
`aU` is not unique on the flat stretches and the interpolated root would be arbitrary there.
`U_S^(0)` is strictly increasing in `aS` at all 119 increments, so that is the direction in
which the crossing is exact.

## v25.0.0 — 2026-09-02

**MAJOR, because the objective changed: four moments joined the battery, so no `Q` from an
earlier version is comparable to one from this version.** The wage-percentile block is now
p10, p25, p50, p75, p90 in both markets — ten moments where there were six. `MOMENT_NAMES`
goes 31 → 35 base names (43 with the duration profile) on the data side
(`data_and_descriptives/setup.jl:416`) and on the SMM side (`smm/moments.jl:55`), which are
independent lists with their own `N_MOMENTS_BASE` and their own assertion.

The tails were added to sharpen the wage-shape parameters: `ρ_x`, `λ_U`, `a_ℓ` and `b_ℓ`.
They cost fit rather than creating it — see the numbers below, which are worse than the
pre-implementation estimate and say something about the skilled upper tail.

### Comparability

- **`Q` is not comparable across the bump.** At the shipped `base_fc` estimate, `Q` moves
  **963.905905 → 1005.587007** once the four targets are on disk (+41.681), and the point
  should be re-optimised. Every earlier `Q` in this file is a 31-moment number.
- **Stored bundles load unchanged.** No struct gained, lost or reordered a field, so every
  `.jls` in `output/estimates/` still deserialises and still reports its own stored
  `loss_opt` — the 31-moment value it was optimised against. Verified on
  `estimate_base_fc_diagonalW.jls`.
- **With the moment CSVs currently on disk, this version reproduces v24.0.2 exactly.** The
  four names are absent from `moments_{w}.csv` until the data pipeline is re-run, and
  `smm_main.jl:390` iterates `MOMENT_NAMES` rather than `keys(moments)`, so absence
  auto-holds-out exactly as a NaN target does. The activation is therefore a data-pipeline
  run, not a code switch: **Stages 9–10 must be re-run to put the four targets and their
  sampling variances on disk.** They have NOT been re-run in the live tree — the release
  gate wrote to a scratch `data/derived/` — so a run today still gives 963.905905.

### What changed

**Data side.** `wpercentile25` / `wpercentile75` were the same function twice with a
different literal; they are now one estimator, `wpercentile(x, w, p)`, with
`wpercentile10/25/75/90` as its named points (`setup.jl:184-200`). The convention is
unchanged — the first observation whose cumulative normalised weight reaches `p`, which is
what `wmedian` does at 0.5 — so `p25` and `p75` are bit-identical to v24.0.2. `wmedian`
keeps its own name because the cross-market overlap moments read the median directly.

- per-year accumulators and pushes: `moments.jl:185-188, 206-217`
- assignment into the moment dict, canonical order: `moments.jl:254-263`
- empty-ASEC NaN fallback: `moments.jl:448-452`
- order-statistic sampling variances at p = 0.10 and 0.90: `sampling_variances.jl:303-307`
- plausibility ranges: `validation.jl:90-99`
- `sigma.jl` influence functions and quantile variances: `323-329, 338-344, 600-604`
  (that file is on no running path — `data_processing_main.jl` does not include it — but it
  is the only other place the moment set is enumerated, so it is not left inconsistent)

**Model side.** `_logwage_moments` returns an 8-tuple — `(mean, var, cm3, p10, p25, p50,
p75, p90)` — with the new points taken through the SAME two paths as the existing three:
`_invert_cdf` on the σ_w-convolved CDF when `σ_w > 1e-10`, `_disc_pctile` otherwise
(`smm/moments.jl:425-444`). Both call sites, both zero-employment fallbacks and the returned
NamedTuple follow (`549-573, 636-645`). The convolution is why no percentile may be computed
off `wU_surface`/`eU_surface` directly — see the rejected section.

**Reporting.** Group, label and short-code maps in `plots_and_tables/transition.jl:273-282,
312-321, 351-360` (`w10U`, `w10S`, `w90U`, `w90S`). The SE-to-name mapping there already
matched by sigma-CSV header rather than by position, which is what makes this insertion
safe against sigma files written before it (`transition.jl:203-206`). Moment counts in the
`smm_main.jl` valid-names block, the pipeline stage banners and the two moment-list headers
now read 35.

### Measured, not asserted

Data targets, from a re-run of Stages 9–10 against the real ASEC micro data (scratch
`data/derived`; every pre-existing row bit-identical to the shipped file in all four
windows, the four new rows the only difference):

| window | `p10_wage_U` | `p90_wage_U` | `p10_wage_S` | `p90_wage_S` |
|---|---|---|---|---|
| base_fc | 5.799651 | 7.195263 | 6.289566 | 7.535608 |
| crisis_fc | 5.808459 | 7.205888 | 6.290485 | 7.543915 |
| base_covid | 5.821442 | 7.231741 | 6.277167 | 7.570823 |
| crisis_covid | 5.894292 | 7.233388 | 6.328504 | 7.589112 |

`p25/p50/p75_wage_U` for `base_fc` stayed at 6.176478 / 6.555541 / 6.914229. Sampling SDs
from `_wquantile_var` at `base_fc`: 0.006819 (p10_U), 0.005115 (p90_U), 0.006911 (p10_S),
0.004959 (p90_S) — against 0.005072 for the existing p25_U, so the same order.

Model side at the shipped `base_fc` estimate, through the extended `_logwage_moments`:

| | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| U | 5.801942 | 6.162544 | 6.556583 | 6.921634 | 7.211901 |
| S | 6.292381 | 6.615154 | 6.948171 | 7.254500 | 7.508079 |

`p25/p50/p75_wage_U` are unchanged to all printed digits, and `Q` with the four held out is
bit-identical at 963.905905 — so the extension perturbs nothing.

`z = (model − target)/σ̂` at that point, on the pipeline's own σ̂:

| | p10_U | p10_S | p90_U | p90_S |
|---|---|---|---|---|
| z | +0.336 | +0.407 | +3.252 | **−5.552** |
| z² | 0.113 | 0.166 | 10.578 | 30.824 |

Sum 41.681, which is the whole of the `Q` increment. **The p90 pair, and the skilled one in
particular, is where the cost is**: the model's skilled p90 sits 0.0275 log points below the
data. The lower tail is essentially free (0.28 of 41.68).

### Settings

None added, retired or re-defaulted.

### Known gaps

- **The `Q` increment is 41.68, not the ~6.6 estimated before implementation.** The
  estimate used the across-year SEs of the measurement table (0.0073 at p10_U, 0.0065 at
  p90_U) and counted the unskilled pair only. The objective's `W` uses the pipeline's own
  order-statistic variances, which are smaller (0.0051 at p90_U), and the skilled p90 is the
  large residual. On the across-year SEs the same four cost 30.9, so the σ̂ convention
  explains ~11 units of the gap and the omitted skilled pair the rest.
- **The identification figures were not re-measured here.** `|t|` on `ρ_x` moving 1.45 →
  3.98 and the 2–3× sharpening of `λ_U`, `a_ℓ`, `b_ℓ` come from the pre-implementation
  measurement pass at the current estimate, as does the finding that `β_U` (6.714 → 6.142)
  and `k_U` (21.4 → 20.29) barely move because the fixed measurement-error convolution
  contributes 22.2% of the reported log-wage variance and blunts these moments as
  instruments. Those numbers were taken before the four were in the objective and have not
  been recomputed under v25.0.0.
- **The live `data/derived/` was not regenerated.** The gate ran against a scratch tree.
- **The plausibility brackets are a convention, not a measurement.** They extend the
  existing 0.2-step ladder outward (p10_U 5.6–6.8, p90_U 6.4–7.6, p10_S 6.0–7.1, p90_S
  6.8–8.0); all eight measured values sit inside with ≥0.28 margin and validation flags 0.
- **`plots_and_tables/transition.jl` was not loaded.** `Plots` and `LaTeXStrings` are not in
  `Project.toml`, so the file cannot be included in the pinned environment. Its three maps
  were parsed and evaluated in isolation and cover all ten percentiles; the rest of the file
  is unexercised by this gate.
- **The moment-count prose in the MCMC layer and in `TARGETING_NOTE.md` still says 28.** It
  was already stale at 31; the MCMC was out of scope for this change and reads its moment
  keys from the spec, so nothing is wrong in the code — only in the sentences.

### Tried and rejected

**p05 and p95.** Measured before implementation: they add 22.9 units of misfit at the
current estimate for a further ~5% on the same parameters the p10/p90 pair sharpens, and
p05 alone sits at z = +3.87. The tails are already the binding part of the wage block; the
extra pair buys almost no identification and asserts a fit the model cannot deliver.

**A separate quantile routine for the new points.** The obvious implementation — a weighted
percentile of `wU_surface` against `eU_surface` — omits the lognormal measurement-error
convolution the reported percentiles carry, and was wrong by **+3.54, −1.93 and −9.38
sampling SEs** at p25/p50/p75 when checked against the model's own values. The new points go
through `_logwage_moments` for that reason, and any future percentile must too.

## v24.0.2 — 2026-09-02

**PATCH: reporting only.** The estimate table in `print_results` was 30 characters wide inside
a 54-character frame. Column widths are now `2 + 12 + 20 + 20 = 54`, so the estimate column
right-aligns flush with the frame's right edge and the rule spans the table exactly
(`smm.jl:2639-2644`). The fixed-parameter block uses `4 + 30 + 20 = 54`, so those values land
on the same right edge despite the deeper indent (`smm.jl:2651`). No number changes; the
values printed are the same values to the same 5 decimal places.

`@printf` requires literal widths, so the three cannot be driven from one constant. The rule
is written as `"─"^(12 + 20 + 20)` rather than `"─"^52` so the column sum appears once in a
form that shows its parts, and a comment states the arithmetic against the frame.

### Known gaps

- **The `Q = … | converged = … | iters = …` footer overflows the frame** at 68 characters
  against the frame's 56. Pre-existing, unchanged here, and not in scope for this request —
  fixing it means either narrowing the footer or widening the frame, and the frame width is a
  literal repeated across the three border lines.

### Tried and rejected

**Driving the widths from a single width constant.** Julia's `@printf` needs a compile-time
format string and does not support `%*s` dynamic width, so the literals cannot be
parameterised without moving to a runtime `Printf.format`. Not worth that machinery for a
table layout; the comment plus the summed rule carries the intent instead.

## v24.0.1 — 2026-09-01

**PATCH: reporting only.** One line per LBFGS iteration during the polish. No number in any
output file changes — the line is printed from inside the gradient function, which already
computes everything it reports, so the stage costs exactly what it cost at v24.0.0.

```
  [lbfgs grad    1]  ‖g‖∞=3.0030e+02 (λ_S)  ‖g‖₂=6.5571e+02  Q=966.612555
```

`‖g‖∞` is the quantity `SMM_POLISH_G_TOL` is compared against and the one that decides
whether `J⁻¹` is a curvature at a stationary point; `‖g‖₂` sits beside it because one large
component and a broadly large gradient call for different reads. The coordinate carrying the
max is named, and the counts `1sided=` and `dead=` appear when a component fell back to a
one-sided difference or had no feasible side at all — printed only when non-zero, so a clean
iteration stays one line. No stride knob: the stage runs ≤ `SMM_POLISH_MAX_ITER` iterations,
so the trace is a page rather than the 30 000 lines SA would emit.

The first gradient evaluation is at `theta_start`, so line 1 is the baseline for free.

### What the trace immediately showed

Measured on `base_fc`, warm start, `SA_MAX_ITER=20`, `SMM_POLISH_MAX_ITER=12`:

| gradient | `‖g‖∞` | `Q` |
|---|---|---|
| 1 | 3.0030e+02 | 966.612555 |
| 6 | 2.9264e+02 | 966.586339 |
| **7** | **1.3584e+05** | 966.583896 |
| 8 | 3.2917e+02 | 966.555919 |
| 11 | 5.4764e+02 | 966.122791 |
| 13 | 5.3998e+02 | 966.034666 |

Three things, none of which were visible before this line existed:

- **`Q` descends monotonically and the descent is essentially complete by iteration 12**
  (ΔQ = 0.578 here, 0.590 with a cap of 25 stopping at 17). The iteration cap is not the
  binding constraint; the line search giving up is.
- **`‖g‖∞` does not fall — it rises, 300 → 540.** So `SMM_POLISH_G_TOL = 1.0` will not fire,
  and the stage will always terminate on the cap or a line-search stall. The default is left
  at 1.0 because it is harmless (the better point is kept regardless) and because changing it
  would imply a stopping rule that has not been justified — see the gap below.
- **Gradient 7 spiked 450× above its neighbours, and `‖g‖∞` rises and stays up afterwards**,
  which is what a corrupt gradient entering LBFGS's curvature memory would look like.

### Known gaps

- **The spike's cause is not established.** The `1sided=` / `dead=` counters were added to
  attribute it, but the confirming run slowed inside a line search and was stopped at
  gradient 4 without reaching iteration 7. What that run does show is the format rendering
  correctly and `λ_S` carrying `‖g‖∞` at every one of the first four iterations. So the
  instrument exists and the observation does not.
- **A raw gradient-norm stopping rule is probably the wrong rule here** and this entry should
  not be read as endorsing it. With `cond(J) ≈ 1e15`, a large `‖g‖∞` in a near-flat direction
  says the minimum is far away along a direction the data does not identify — not that the
  point is unconverged in any direction that matters. The interpretable criterion is the
  predicted remaining `ΔQ` (a Newton decrement, `gᵀJ⁻¹g`), which is in criterion units and
  cannot be computed inside the polish without the Jacobian the polish is a prerequisite for.
  Unresolved, and recorded here rather than left in nobody's memory.

### Tried and rejected

**Computing the gradient norm in `smm_main.jl` around the polish call.** Would have cost
`2d = 46` extra solves per report for a number the gradient function already holds, and would
have given two points (before, after) where the trace gives one per iteration.

**A print-stride setting mirroring SA's `trace_stride`.** SA emits up to 30 000 lines and
needs one; the polish emits at most `SMM_POLISH_MAX_ITER`. A knob whose only correct value is
1 is a knob that will be set wrong.

## v24.0.0 — 2026-09-01

**MAJOR: a change that moves the estimate for reasons other than optimiser noise.** An LBFGS
polish now runs from the annealed point, and the theorist has to be told for a second reason
beyond the moved estimate: the standard error this project reports is `J⁻¹ = (G'WG)⁻¹`, a
curvature **at a stationary point**, and annealing does not deliver one. Measured at v23.0.0
on `base_fc`, `‖dQ/dt‖ = 1206` at an SA optimum with `A` at +806 and `λ_S` at −389. Any
`J⁻¹` computed at such a point carried no interpretation, so this stage is a precondition for
the inference, not a refinement of the point.

**Comparability.** Estimates from v23.x are superseded: a v24 run reports a lower `Q` at a
different `θ̂`. The criterion is unchanged, so `Q` values remain on the same scale and are
directly comparable — unlike the v22→v23 break. All bundles stay loadable: nothing gained a
struct field, which is why the three new controls are keyword arguments (`SMMRunParams`
cannot gain a field without invalidating every `.jls` on disk — Julia's serialiser reads
structs positionally by field count).

**Only viable from v23.0.0.** While the training frontier was a 0/1 indicator the criterion
was a step function in the margin parameters — `c` had to move 0.026% of its value before one
of 14,400 cells flipped, so every derivative was exactly zero over a neighbourhood. No
gradient method could move, and Nelder-Mead was measured closing 0% of the gap. Both facts
changed when the frontier became continuous.

### What changed

- **`smm.jl:run_smm`** gains four keyword arguments, all defaulting to the previous
  behaviour: `theta_start`, `fd_step`, `max_iter`, `g_tol`.
- **`smm.jl`** Optim branch: starts at `theta_start` when supplied; supplies an explicit
  central-difference gradient for `:lbfgs`/`:bfgs`; switches to a gradient-norm stop when
  `g_tol > 0`; uses a BackTracking line search.
- **`smm_main.jl` §9**: the polish stage, after annealing, keeping whichever point has the
  lower `Q`.
- **`settings.jl`**: four entries registered.

**`fd_step` is measured, not conventional.** At v23.0.0 all 23 columns are stable to 4–5
significant figures for `h ∈ [3e-7, 3e-6]`, and several break by `h = 1e-4` where `λ_S`'s
derivative changes sign. Optim's own default is `cbrt(eps) ≈ 6e-6`, just outside that band.

**The gradient calls `smm_objective` directly, not `obj_traced`.** At `d = 23` a single
iteration's `2d` gradient evaluations would consume half of a 100-evaluation window and the
rate stop would fire on arithmetic rather than on stalling. The line search still runs
through `obj_traced`, so the incumbent and corner reporting stay correct.

### Gate

| item | result |
|---|---|
| 1. changed files parse | `smm.jl`, `smm_main.jl`, `settings.jl` — all clean |
| 2. entry point loads | `smm_main.jl` ran to `Done.`, exit 0 |
| 3. runs on real data, scratch dir | `/tmp/pol/tree`, live `data/` symlinked, `output/` local |
| 4. `check_version_sync` | in sync, one declaration, 3 readers ok |
| 5. old bundles load | `INIT_MODE=:warmstart` read `estimate_base_fc_diagonalW.jls` and the run proceeded; no struct changed |
| 6. `check_forwarding.jl` | PASS; `check_dead_settings.jl` PASS |
| 7. changelog | this entry |

Measured on `base_fc`, warm start, `SA_MAX_ITER=30`, `SMM_POLISH_MAX_ITER=25`:

| | `Q` | iterations |
|---|---|---|
| annealed point | 966.612555 | 30 |
| after polish | **966.022189** | 17 |
| ΔQ recovered | **0.590366** | |

### Settings added

| setting | default | note |
|---|---|---|
| `SMM_POLISH` | `true` | run the stage |
| `SMM_POLISH_FD_STEP` | `1e-6` | central-difference step, in `t` units |
| `SMM_POLISH_MAX_ITER` | `200` | LBFGS iterations, each ≈ `2d` + line-search solves |
| `SMM_POLISH_G_TOL` | `1.0` | gradient ∞-norm stop; from max ≈ 806 at the annealed point |

### Known gaps

- **The gradient norm after the polish was not measured.** That is the stage's whole purpose,
  and the gate establishes only that `Q` fell by 0.59. Whether `g_tol = 1.0` is reachable is
  unknown; the 25-iteration run stopped at 17 with `converged = false`, meaning the line
  search gave up rather than the gradient criterion being met.
- Gated on `base_fc` only, at `max_iter = 25`. The shipped default is 200 and has not been
  run to completion on any window.
- The gradient is serial: `2d = 46` solves per iteration, roughly 17 s, so 200 iterations is
  about an hour per window.
- `eU_surface`'s density-versus-mass inconsistency (reported against v22.0.0) is still
  unfixed and still affects `emp_var_U`, `emp_cm3_U`, the three U percentiles, both overlap
  moments and `wage_premium`.

### Tried and rejected

**Rebuilding the spec with the annealed values as `init`, instead of `theta_start`.** The
Optim branch starts at `pack_theta(spec)`, so re-anchoring that way is the obvious route —
but it forces a `θ→t→θ` round trip, and `check_tau_margin.jl` documents a 1e-15 round-trip
rounding flipping a frontier cell. `theta_start` is one line and no transform.

**Optim's own finite-difference gradient.** Step outside the measured stable band, above.

**HagerZhang, Optim's default line search.** Two independent reasons. Cost: it re-evaluates
the *gradient* at each trial step — 46 solves — where BackTracking needs one objective call.
Correctness: with the pinned Optim/LineSearches pair its `ϕdϕ` destructures
`value_gradient!(df, x)` as a 2-tuple while Optim's `ManifoldObjective` returns the value
alone, raising `BoundsError: attempt to access Float64 at index [2]` before the first
iteration completes. Measured, not inferred — it is what the first live gate returned. The
Nelder-Mead path never calls `value_gradient!`, which is why this was latent.

**Inheriting the Nelder-Mead tolerances for the polish.** Measured: LBFGS halted after ONE
iteration on `f_reltol`, recovering ΔQ = 0.005334 while `max|dQ/dt|` was still ≈ 806. Each of
`f_reltol`, `x_abstol` and the rate callback stops on a small *move*, and near a stationary
point the moves are small while the gradient need not be. Hence `g_tol` disables all three.

**Adding fields to `SMMRunParams` for the new controls.** Would make every `.jls` on disk
unreadable.

**Using `obj_traced` inside the gradient.** Would inflate the evaluation counter by `2d` per
iteration and fire the rate stop on arithmetic.

## v23.0.0 — 2026-09-01

**MAJOR, because the criterion changed and no earlier `Q` is comparable.** The training
frontier is no longer a 0/1 indicator on the ability grid; it is the population fraction of
each cell that trains. At the v22.0.0 `base_fc` estimate `Q` moves **968.097038 → 970.339482**
— the old frontier was flattering the fit by 2.24 units — and the point estimate should be
re-optimised. Bundles stay loadable: no struct field changed.

### The defect

`unskilled.jl` set `τT[i,j] = (Utr_j >= Usearch[i]) ? 1.0 : 0.0`, and
`solve_stationary_unskilled!` then read it through `if τ[i,j] > 0.5`. Both halves had to
change: fractional values produced upstream were re-hardened downstream, so fixing either
alone is a no-op (verified — an earlier covered-fraction probe returned bit-identical
output for exactly this reason).

Because every composition aggregate is `Σ τ·W2`, a hard indicator made them **step
functions of θ**. A change in θ too small to move the frontier past a grid node flipped no
cell, so the trained mass was bit-identical and `∂m/∂θ` was **exactly zero** for every
moment. Measured at v22.0.0 on `base_fc`:

| step in `c` (`t`-units) | `Δθ` | cells flipped | `ΔQ` |
|---|---|---|---|
| 1e-08 … 1e-04 | ≤ 2.9e-04 | **0** | machine noise, ~1e-12 |
| 1e-03 | 0.0029 | 1 | **+0.181** |
| 3e-03 | 0.0086 | 2 | +0.738 |
| 3e-02 | 0.085 | 48 | +85.4 |

So `c` — the training cost, the parameter most directly on the margin — had to move
**0.026% of its own value** before one of 14,400 cells flipped, and `Q` then jumped in
quanta of about **0.18 sampling standard errors per cell**. The same mechanism made all
seven unskilled parameters read exactly zero on `training_share` and on every skilled
moment in the v22.0.0 Jacobian: 33.6% of the unskilled block's cells were exactly zero.

### The replacement

`Utr(aS) = −c(aS) + T(aS)` depends only on `aS` and `U^search(aU)` only on `aU`, and `Utr`
is increasing in `aS` (verified: 119/119 increments), so for each `aU` the training set is
an upper interval with a single crossing `aS*(aU)` solving `Utr(aS*) = U^search(aU)`. `τ` is
now the fraction of node `j`'s midpoint interval lying above `aS*`, found by linear
interpolation of the crossing. That is continuous and piecewise-linear in `aS*`, hence in
`θ`, and reduces to the old indicator whenever `aS*` falls on a node.

Where `Utr` is *not* monotone the crossing is not unique, so that case keeps the hard
indicator rather than interpolating a crossing that means nothing — which confines the
change to the boundary node exactly when the maintained property holds.

`solve_stationary_unskilled!` now splits the cell's weight, `u ∝ (1−τ)` and `t ∝ τ`,
instead of thresholding. Identical at `τ ∈ {0,1}`, linear in between, and it is the
mass-consistent reading of a fractional frontier.

### Gate

| check | result |
|---|---|
| `c`'s derivative, `ΔQ/h` at `h` = 1e-08 / 1e-07 / 1e-06 / 1e-04 | **−1315.32 / −1315.32 / −1315.32 / −1315.30** |
| same column at v22.0.0 | **exactly 0 for every `h ≤ 1e-04`** |
| fractional cells | 91, one per each of the 91 rows with an interior crossing |
| cells at 1 / at 0 | 4,954 / 9,355 (sums to 14,400) |
| trained mass `Σ τ·W2` | 0.33110099, from 0.33135 — −0.075% |
| `estimate_base_fc_diagonalW.jls` loads | yes, 10 fields |

A constant `ΔQ/h` across five decades is the definition of a derivative existing. The
0.7% wobble at `h` = 1e-05 is the kink at a node boundary, which is expected of a
piecewise-linear frontier and is orders of magnitude smaller than the discontinuity it
replaces.

### Not changed, and why

- **`transition_solver.jl`** uses `τ_ij` as a *flow rate* (`τ_ij * uU_ij`, and inside
  `(fU + τ_ij + ν)`), so it is already linear in `τ` and a fractional value is coherent
  there with no edit. It receives the new `τ` through `path.τT .= uc.τT` unchanged.
- **`plots_and_tables/model.jl`** already reduces `τ` by `sum(τ, dims=2)/Nx`, which a
  fractional field makes more accurate, not wrong. `_get_x_bar` in
  `plots_and_tables/transition.jl` already interpolates the crossing of
  `net_T − U^search` rather than reading `τ > 0.5`, and its own docstring explains why —
  it was doing the right thing before this change.
- **`check_tau_margin.jl`** keeps its purpose; its header is updated. That script was
  written to measure a cliff whose existence it attributed to exactly this indicator
  ("that guard is DISCRETE in a quantity that is CONTINUOUS in θ"), so the codebase had
  already identified the defect in its own tooling.

### Consequences to expect

- The point estimate needs re-optimising; `Q` at the old `θ̂` is 2.24 higher.
- The unskilled → skilled block of the Jacobian should stop being exactly zero, and
  `training_share` should acquire a gradient in the unskilled parameters. That is the
  identification repair, not a side effect: without it those directions are unidentified
  by construction rather than by economics.
- The `ΔQ=1` half-widths change meaning for margin parameters. Under the old frontier a
  "width" for `c` measured the distance to the nearest tread edge — a property of where
  `θ` happened to sit, not of curvature — which is why `c`'s width came out as the
  smallest of all 23 while its derivative was zero. Widths are now curvature.

### Known gaps

- The 2.24-unit `Q` change is measured only at the `base_fc` estimate; the other three
  windows were not re-solved.
- The boundary correction is exact in the `aS` direction only for a monotone `Utr`, and
  monotonicity was verified at one parameter vector, not proved.
- The `eU_surface` density-versus-mass inconsistency reported against v22.0.0 —
  the U wage histogram weighting cells with no `wp` factor while the S branch uses
  `m = e*wpS[jp]` — is **not** addressed here and still affects `emp_var_U`, `emp_cm3_U`,
  the three U percentiles, both overlap moments and `wage_premium`.

## v22.0.0 — 2026-09-01

**MAJOR, because the objective changed.** Eight unemployment-duration moments join the
moment set and four of them carry objective weight, so `Q` under this version is not
comparable to any earlier number: at the shipped `base_fc` estimate `Q` goes from 480.9
to 971.638. Point estimates and standard errors from v21.x remain valid *for the
criterion that produced them*, and every bundle stays loadable, but the two criteria are
different objects and must not be compared.

The four **skilled** points are active. The four **unskilled** points are computed and
reported but held out (see Known gaps): they measure an exit concept the model does not
implement, and at the shipped estimate they alone would contribute Σz² = 5312.4 against
589.7 for the skilled four. Holding them out was decided before any v22.0.0 run existed,
so this entry describes the shipped default rather than correcting a released one.

### The new category: unemployment-duration profile

`usurv{t}_j` is the survivor of the ONGOING unemployment spell in market `j` at `t`
weeks — the weighted share of that market's unemployed whose current spell has lasted at
least `t` weeks — for `t ∈ {5, 14, 27, 53}` and `j ∈ {U, S}`. `MOMENT_NAMES` goes 31 → 39
in both copies (`data_and_descriptives/setup.jl` and `smm/moments.jl`).

Three construction choices, each for a reason that would otherwise be re-litigated:

- **Thresholds one week above the CPS heaps** (4/13/26/52). Respondents round, so a
  threshold placed *on* a heap makes the moment depend on which side of the comparison
  the heap falls. This is the convention `ltu_share_S` already followed with 27.
- **Survivor values, not bin shares.** Bin shares within a market sum to one, so a
  complete set is exactly collinear and the block would be singular by construction.
- **Thresholds defined once**, in `solver/equilibrium.jl::DURATION_THRESHOLDS_WK`.
  `smm/moments.jl` derives the moment names from it, so the model counterparts and the
  SMM names cannot drift apart. The data pipeline keeps its own copy because it runs
  standalone; that copy is canonical and is the one to edit.

### `ltu_share_S` is now held out, not deleted

`usurv27_S` reproduces `ltu_share_S` exactly — verified identical in all four windows,
sampling variance included — so weighting both would count one restriction twice.
`ltu_share_S` moves to `SKIP_MOMENTS`: still computed, still returned by the solver, still
read by every table and bundle, but carrying no objective weight.

Backward compatibility was checked rather than assumed: `obj.ltu_share_S` = 0.188431 at
the shipped estimate, unchanged, and `obj.ltu_share_S == obj.usurv_S[3]` is `true`. It is
evaluated through the same survivor function as the vector, so it stays correct even if 27
weeks is later dropped from the threshold list. All 31 pre-existing data moments reproduce
the previous pipeline exactly (maximum relative deviation 0.00e+00 across 31 × 4 windows).

### Model side

`compute_equilibrium_objects` now returns `usurv_S` and `usurv_U`, additively. The skilled
survivor generalises the existing `ltu_share_S` mixture (d=0 searchers exit at
`κ_S(1−Γ_o(p*_S))+ν`, d=1 crossers at `f_U+ν`). The unskilled survivor is new: a searcher
exits at `f_U`, matching the `f_hire` in `solve_stationary_unskilled!`, while a worker at
the participation corner `p*_U = 1` is never hired and exits only at `ν`. That corner holds
9.32% of the unskilled unemployed stock at the estimate and gives the unskilled survivor a
floor the skilled one does not have.

### What the profile immediately shows

At the shipped `base_fc` estimate, model against target in sampling-SE units:

| threshold | 5 wk | 14 wk | 27 wk | 53 wk |
|---|---|---|---|---|
| `usurv*_U` z | +50.7 | +37.3 | +18.4 | +31.8 |
| `usurv*_S` z | +11.4 | +2.6 | **−9.9** | **−18.8** |

The unskilled model survivor is above the data at every threshold. **The skilled one
crosses**: too flat before 14 weeks, too steep after, 0.0378 against 0.0995 at 53 weeks.
A crossing cannot be removed by rescaling either hazard, so no value of `f_S`, `f_U` or
the crosser share fixes it — it is a shape failure, and that is the finding a single point
at 27 weeks could not produce.

The same arithmetic explains why the old `ltu_share_S` residual never moved. With both
hazards pinned by `jfr_S`, `jfr_U` and the tightness moments, the survivor at 27 weeks is
confined to `[0.1496, 0.1884]` for any crosser share in [0,1]; the target 0.2326 sits
9.95 SEs **above** that ceiling, and the model was parked at the ceiling with zero
crossers. A moment at a boundary identifies nothing. Fixing `jfr_S` makes it worse: raising
`f_S` to its target lowers the ceiling to 0.1331.

### Also in this release

**The version is now declared once.** `data_and_descriptives/data_processing_main.jl`
included its own `const ROYSEARCH_VERSION` and had to be remembered on every release;
it now reads `code/smm/version.jl` instead (`data_processing_main.jl`:61, using
`@__DIR__` because the banner prints before the Paths block defines `PIPELINE_DIR`).
All three banners — `smm_main.jl`, `MCMC_main.jl`, `data_processing_main.jl` — now
include the one declaration, and a bump touches one line. This was a live drift risk,
not a hypothetical: the pipeline once read 15.12 against 16.7.1 in the estimation code.

The `roysearch-versioning` skill was updated to match, since the old check treated a
banner without the constant as a fault and would have blocked every future bump. It now
verifies the invariant directly — one declaring file, and every banner either includes
it (`readers_ok`) or is reported as a `duplicate` declaration or `missing`. The reader
test skips comment lines, because `MCMC_main.jl` describes its include path in prose
above the real call and a whole-file regex passes on that.

`_nan_targets` in `smm_main.jl` now iterates `MOMENT_NAMES` instead of `keys(moments)`, so
a moment **absent** from the moments CSV auto-holds-out exactly as a present-but-NaN one
does. Previously only NaN was covered, which meant adding a name to `MOMENT_NAMES` before
re-running the data pipeline failed downstream instead of holding out — defeating the
auto-activation the surrounding comment promises. The log line now says "missing or NaN".

### Gate

Both data states verified against the shipped `base_fc` estimate:

| | held out | active | `W` | `Q` |
|---|---|---|---|---|
| old CSVs (optimiser run before the data script) | 12 | 27 | 27×27 | 381.922 |
| new CSVs, all eight weighted (not shipped) | 4 | 35 | 35×35 | 6283.977 |
| **new CSVs, shipped default** | **8** | **31** | **31×31** | **971.638** |

Each row was cross-checked against a Σz² computed independently in Python from the
per-moment model/target/σ̂ table: `381.922 + 589.7 = 971.6` against the measured
971.638306, and `381.922 + 5902.1 = 6284.0` against the measured 6283.977 — two separate
calculations agreeing to six and four significant figures respectively.
`check_forwarding.jl` and `check_dead_settings.jl` both PASS. Old bundles load
(`estimate_base_fc_diagonalW.jls`, read repeatedly during the gate).
`check_output_tree.jl` reports 10 findings and `check_structure.jl` 7, all pre-existing
and none naming a file this release touched: non-`.tex` CSVs in `tables/` and three
`estimate_*_backup_Q*.jls` files in `estimates/`.

### Known gaps

- **The four survivor points per market are nested events**, hence strongly positively
  correlated, and a diagonal `W` treats them as independent — so the block is overweighted
  relative to its true information content. Conditional continuation ratios
  `q_k = S(t_{k+1})/S(t_k)` carry the same information, are asymptotically independent
  across `k`, and have no adding-up constraint. Not adopted here; the survivor form is what
  reproduces `ltu_share_S` exactly and was verified against it.
- **`usurv*_U` is held out because it depends on an exit concept the model does not
  implement.** A CPS duration
  spell ends on *any* exit, including to NILF; the model's unemployed leave only by hiring
  or at `ν`, which is floored at the demographic life-table rate (0.00323/month) because
  the measured *net* labour-force outflow is negative in all four windows. The gross
  unemployed→NILF flow is 44–55× larger. The unskilled data force a withdrawal hazard of
  at least 0.015–0.065/month; the skilled data force none, which is why `ltu_share_S` was
  defensible and a `ltu_share_U` was never added.
- **`sigma.jl`** (full Σ̂, not included by any entry point) has no influence functions for
  the new moments, so its Σ̂ would be rank-deficient by eight. Untouched by request — the
  diagonal path in `sampling_variances.jl` does cover them and is unaffected.

## v21.1.0 — 2026-09-01

**MINOR.** The `γ` brake introduced in v21.0.0 measured acceptance against the wrong
denominator, so on the `base_fc` run it could never reach its target and shortened the step at
every window for 2000 generations. It now measures acceptance among *feasible* candidates.

### Comparability

**Point estimates carry forward; the sampled sequence does not.** The target, the prior, the
estimator and the moment set are untouched, so `argmin Q` is unchanged and every existing
bundle stays valid and loadable — the `base_fc` seed remains `Q = 480.880318`. The chain's
path changes, so a run under this version will not reproduce v21.0.1's draws. Nothing is
superseded by that, because **v21.0.1 produced no usable standard errors**: `mcse` sat at its
`1/√N` ceiling for eight consecutive checks. There is no inference result to invalidate.

### The defect

`fin ≈ 0.77` on the `base_fc` run — roughly a quarter of proposals were infeasible — while
acceptance among feasible candidates saturated near 0.26. Unconditional acceptance was
therefore capped at about `0.77 × 0.26 ≈ 0.20`, **below the 0.234 target the brake was
chasing.** Being one-sided, it braked at every one of ~80 adaptation windows and never stopped:

| `g` | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 |
|---|---|---|---|---|---|---|---|---|
| `γ×` | 0.545 | 0.335 | 0.250 | 0.208 | 0.184 | 0.168 | 0.154 | 0.142 |
| `esjd` | 1.387 | 0.136 | 0.070 | 0.048 | 0.038 | 0.032 | 0.027 | 0.022 |
| `acc` | 0.232 | 0.153 | 0.166 | 0.185 | 0.194 | 0.200 | 0.199 | 0.202 |
| `acc \| feasible` | 0.301 | 0.204 | 0.218 | 0.243 | 0.249 | 0.260 | 0.258 | 0.259 |
| `R̂` | 6.814 | 7.216 | 7.544 | 7.690 | 7.479 | 7.688 | 7.886 | 7.853 |

`esjd` fell **63×**. Decomposing `esjd = acc·(γ·spread)²`, the population contracted 1.73×
between g = 250 and g = 750 and then stopped (1.12× over the next 1250 generations) — so after
g = 750 the falling step was entirely `γ`, not the population.

### The fix

An infeasible candidate is a prior-zero rejection. The 0.234 optimum describes the Metropolis
accept/reject on the target, and the infeasible rate does not respond to the step — `fin` stayed
at 0.75–0.78 while the step fell 7×. Conditioning on feasibility is therefore the correct
correspondence, not a tolerance widened to fit. On the `base_fc` numbers above the brake would
have fired at g = 500 and 750 and **stopped at g = 1000**, where conditional acceptance reached
0.243, holding `γ× ≈ 0.208` and `esjd ≈ 0.048`.

No dead band was added: the update is proportional to `(acc − target)`, which shrinks as the
gap closes, so it self-limits without a threshold knob.

`wprop_ad` — the old denominator — is removed rather than left unread, and `acc_ad_last` is now
printed on the check line as `accf`. The absence of that field is why this took a full 2000-
generation run to identify: the brake was steering by a quantity no log recorded.

### Measured, not asserted

Live gate, `base_fc`, N = 24, `GAMMA_EVERY` = 25:

```
gen   100/150  acc=0.214  fin=0.77  ...  rep=1 RETAINED
check g=100    R̂=2.74  γ×1.000 accf=0.332  moves=248/230
```

`acc = 0.214` is **below** the 0.234 target while `accf = 0.332` is **above** it, and `γ×`
stayed at 1.000. Under the old denominator the brake fires at this point; under the new one it
idles. That is the change, observed directly. The `rep=N RETAINED` branch also fired live here
for the first time, having previously been verified only by enumerating its conditional.

Both settings audits pass; `code/mcmc/demc.jl` parses with no `:error` or `:incomplete` nodes.

### Known gaps

**The brake is not what blocks the standard errors, and this release does not unblock them.**
Between g = 1000 and g = 2000 the step fell 1.543× and acceptance rose only 0.185 → 0.202,
where a smooth criterion of the same curvature gives 0.390 — the step bought **8%** of its due.
Acceptance among feasible candidates is flat near 0.26 across a 1.5× step reduction; on a smooth
target a symmetric proposal climbs toward 0.5 as the step shrinks. The criterion is rough at the
scale being sampled, which caps conditional acceptance and holds `R̂` near 7.7 whatever the
proposal does. **The two hard boundaries in `GRID_INVARIANCE.md` — the training frontier's `0/1`
indicator and the participation margin snapped to an ability node — are now the blocking item
for inference, not a parallel task.**

**Not tested:** a forced-braking arm (`GAMMA_TARGET = 0.90`) was queued and stopped before it
started, to free cores. It would have confirmed only that braking still fires when the
conditional rate is below target, which the v21.0.0 calibration and the 2000-generation run
already establish.

### Tried and rejected

**A dead band around the target.** Considered and dropped: the update is already proportional to
the shortfall, so it decays to nothing as the gap closes. A band would have been a second knob
masking the wrong denominator rather than fixing it.

**Targeting `esjd` instead of acceptance.** Mixing, not acceptance, is the objective, and a
controller on `esjd` would be the principled instrument. Not built: `esjd` has no
scale-free optimum to target the way acceptance has 0.234, so it would need a reference value
this model cannot yet supply — the criterion is not smooth enough for one to mean anything.

## v21.0.1 — 2026-08-31

**PATCH.** Every number in every output file is unchanged; only the source form of fourteen
index ranges changed, to clear the IDE lint *"Indexing with indices obtained from `length`,
`size` etc is discouraged. Use `eachindex` or `axes` instead."*

### Comparability

Fully comparable with v21.0.0. `eachindex(v)` on a one-based `Vector` is `OneTo(length(v))`
and `axes(A, d)` is `OneTo(size(A, d))` — each replacement is exactly the range it replaced.
Verified by re-evaluating the criterion at the shipped `base_fc` seed after the change:
`Q = 486.359094` against the bundle's own recorded `486.359094`, identical to six decimals.
`candidates.jl`'s pairwise-distance kernel was checked separately and returns an identical
matrix under both forms.

### What changed

Eleven direct swaps: `1:size(A, d)` → `axes(A, d)`, `1:length(v)` → `eachindex(v)`.
Eleven direct + two pairwise + one non-indexing = the fourteen occurrences below.

- `code/mcmc/MCMC_main.jl` — the two `@threads` loops over proposal columns
  (`axes(Xp, 2)`, `axes(Xj, 2)`), the four `res.draws` reductions behind the posterior mean
  and median (`axes(res.draws, 2)`), and the two parameter-vector loops in the box check and
  the constrained transform (`eachindex(θ)`, `eachindex(θ0)`).
- `code/smm/candidates.jl` — the distance kernel's row loop (`axes(X, 1)`).
- `code/transition/transition_solver.jl` — the path unpack (`axes(sc.E0, 2)`).
- `code/policy/policy_solver.jl` — the surface loop (`eachindex(pgU)`).

Two pairwise loops where `eachindex` does **not** apply, because the body reads `i` and
`i+1` and the range must stop one short:

- `code/data_and_descriptives/sipp.jl:562` and `code/plots_and_tables/transition.jl:962` —
  now `firstindex(x):lastindex(x)-1`. `eachindex(x)[1:end-1]` would also silence the lint but
  allocates a vector per call, which inside `sipp.jl`'s `@inbounds` record loop is a real cost.

One that was not indexing at all:

- `code/data_and_descriptives/data_processing_main.jl:240` — singleton cluster ids built as
  `collect(1:length(cw))` are now `collect(eachindex(cw))`, the same vector and a more direct
  statement of "one cluster per observation".

### Measured, not asserted

Fourteen occurrences replaced, **zero remaining** (`grep '1:length(\|1:size('` over
`code/**/*.jl` is empty, and no `.jl` outside `code/` carried the pattern). All seven touched
files parse with no `:error` or `:incomplete` nodes.

### Known gaps

**Load-and-run verification covers the estimation chain only.** `MCMC_main.jl` and
`candidates.jl` were loaded and exercised. The four files in other subsystems — `sipp.jl`,
`transition.jl`, `transition_solver.jl`, `policy_solver.jl` — are parse-verified only, because
each belongs to a driver with its own data dependencies. The argument for accepting that: every
replacement changed only the *wrapper* around an identifier already present in the line
(`recs`, `xg`, `sc.E0`, `pgU`), introducing and altering no names, so there is no failure mode
a load would surface that a parse does not.

## v21.0.0 — 2026-08-31

**MAJOR, because a run under this version does not reproduce v20.0.0's numbers and the
theorist has to be told why.** v20.0.0 made the target proper, which was necessary and is
unchanged here; it did not make the chain mix. The v20.0.0 `base_fc` run aborted at
generation 1000 on the acceptance floor with acceptance at 0.009, so the standard errors it
would have produced are not standard errors of anything. `γ` is now braked toward the
acceptance optimum instead of held fixed.

### Comparability

**Posterior standard errors from v20.0.0 are superseded** — there are none to supersede in
practice, since no v20.0.0 chain reached a stationary state. **Point estimates are not
superseded:** the brake changes only the proposal scale, never the target, so `argmin Q` is
unchanged and every `estimate_*` and `chain_*` bundle remains valid and loadable — verified,
both `base_fc` bundles deserialize under this version (10 and 32 fields).

The v20.0.0 run improved the point estimate and the two numbers involved are distinct.
`estimate_base_fc_diagonalW.jls` holds **Q = 488.548412**, the g=31 checkpoint
(`ΔQ = 2.0025` from the 490.550869 seed, just clearing `PROMOTE_MIN_DQ = 2.0`), and that is
the current `base_fc` seed. The chain went on to reach **Q = 488.038115**, which lives in
`chain_base_fc_diagonalW.jls` as `theta_best` and was deliberately *not* promoted: the
further gain is `ΔQ = 0.51`, below the threshold and far inside this criterion's own
numerical resolution, since `Q(θ̂)` moves 19.4 units between `Np_S` 120 and 160. Promoting it
would be chasing quadrature noise.

Setting `MCMC_GAMMA_ADAPT = false` reproduces v20.0.0 exactly: `gamma_scale` initialises to
1.0 and is only ever multiplied, so with the brake disabled every proposal is bit-identical.

### What was wrong

ter Braak's `γ = 2.38/√(2δn)` is optimal only when the population already sits at the
target's scale, and the difference vector is drawn from the population itself — so nothing
in DE-MC references an absolute scale, and a population that has drifted wider proposes
proportionally longer steps forever. Measured on the v20.0.0 `base_fc` chain
(`chain_base_fc_diagonalW.jls`, 23 × 95 × 1000):

| generation | 250 | 500 | 750 | 1000 |
|---|---|---|---|---|
| `acc` | 0.193 | 0.027 | 0.014 | 0.009 |
| `R̂` | 8.658 | 10.384 | 10.026 | 9.664 |
| `esjd` | 1.397 | 0.119 | 0.062 | 0.043 |
| `mcse` | 0.1020 | 0.1023 | 0.1022 | 0.1023 |

The scale was right at the first check and the step drifted to roughly twice too long over
the next 750 generations. Feasibility is not the cause: `fin ≈ 0.74`, so acceptance
conditional on feasibility is 0.0125 — **98.75% of feasible proposals fail the Metropolis
test.** Inverting `acc ≈ 2Φ(−ℓ√d/2)` at `d = 23` gives `ℓ = 1.043` against the
Roberts–Gelman–Gilks optimum `ℓ* = 2.38/√23 = 0.496`.

The population's error is a **shape** error, not a scale error. Pooled sd against the
criterion's own `ΔQ = 1` widths, both in `t`:

| coordinate | pooled sd | width | ratio |
|---|---|---|---|
| `λ_S` | 0.02624 | 0.00105 | 24.98 |
| `k_U` | 0.06634 | 0.00298 | 22.27 |
| `b_Γ` | 0.09430 | 0.00535 | 17.62 |
| `β_U` | 0.08387 | 0.00482 | 17.39 |
| `λ_U` | 0.01022 | 0.00376 | 2.72 |
| `c` | 0.00442 | 0.00209 | 2.11 |
| `ξ_S` | 0.02345 | 0.01731 | 1.36 |
| `b_S` | 0.2711 | 5.464 | 0.05 |

Median 6.06, spanning ~500×. `b_S` is the only coordinate too *narrow*, and its denominator
is the flat-then-cliff distance rather than a posterior scale, so 0.05 overstates the
magnitude — but the sign is opposite to every other coordinate's and that is what matters.

**A scalar cannot correct a shape error, and does not have to.** DE-MC reshapes its own
population through accepted moves; at 0.9% acceptance there were ~60 per chain in 1000
generations, far too few to reshape 23 dimensions. Restoring acceptance restarts the
covariance adaptation that fixes the shape. That step is mechanistic, not measured — see
*Known gaps*.

### What changed

**`code/mcmc/demc.jl`** — `gamma_scale`, a one-sided Robbins–Monro brake on the DE step,
updated every `gamma_every` generations by `exp((η/√k)·(acc − target))` and applied only
when `acc < target`. The mode jump keeps `γ = 1`: it is a deliberate full-population hop and
rescaling it would defeat its purpose. `n_adapt` counts braking events only, so the
diminishing step is not spent during the transient. Returned as `gamma_scale`, `n_adapt`.

**`code/mcmc/demc.jl`** — the check line now carries `mcse`'s **ceiling**, `R̂`, and the live
`γ×`. `mcse` is bounded above by `1/√N` (0.103 at N = 95), so a chain at `R̂ = 9.7` printed
`mcse = 0.102` against a 0.05 target — reading as a factor of two when it is a factor of nine.

**`code/mcmc/MCMC_main.jl`** — `MCMC_GAMMA_ADAPT`, `MCMC_GAMMA_TARGET`, `MCMC_GAMMA_ETA`,
`MCMC_GAMMA_EVERY` declared and forwarded.

**`code/smm/settings.jl`** — the four registry entries.

**`code/mcmc/MCMC_main.jl`** — `MCMC_MOVES_MIN` 2300 → 230 (`100·d` → `10·d`). `100·d` sizes
a stable *full* covariance, which is not the deliverable; at N = 95 reaching `mcse = 0.05`
near generation 670 yields ~640 accepted moves, so the old value would have refused a stop
the accuracy gate had already granted.

### Measured, not asserted

`η` and the cadence are calibrated rather than conventional. A closed-loop simulation —
population expanding by the measured identity, acceptance from the RGG relation calibrated
to this run's own `ℓ = 1.043` at `gamma_scale = 1` — **reproduces the observed collapse under
fixed `γ`**, which is what licenses its other rows:

| variant | acc @2000 | final step / optimal |
|---|---|---|
| fixed `γ` (v20.0.0) | 0.0001 | 3.34 |
| `η=1`, every 50 | 0.065 | 1.51 |
| `η=1`, every 25 | 0.138 | 1.23 |
| **`η=2`, every 25 (shipped)** | **0.200** | **1.07** |
| `η=3`, every 25 | 0.212 | 1.04 |

**Live A/B at the shipped defaults**, `base_fc`, N = 24, 120 generations, identical seed,
differing only in `MCMC_GAMMA_ADAPT`:

| generation | brake off | brake on |
|---|---|---|
| 30 | 0.724 | 0.724 |
| 60 | 0.342 | 0.342 |
| 90 | 0.199 | 0.199 |
| **120** | **0.140** | **0.182** |
| final `accept` | 0.351 | 0.361 |
| `Q(θ̄)` | 487.5617 | 487.4926 |

The arms are identical to the last printed digit through g = 90 — `acc`, `dlp`, `esjd`, `R̂`
and `max logπ` all match — which is the backward-compatibility guarantee verified on live
data rather than argued from the code: the adaptation windows at g 1–25, 26–50 and 51–75 were
all above target, so the brake stayed idle and an idle brake must reproduce the unbraked run
exactly. The g 76–100 window was the first below target, the first braking event fired at
g = 101, and acceptance at g = 120 is **0.140 → 0.182, +30%** twenty generations later.

Release gate: all seven items pass. `check_forwarding.jl` and `check_dead_settings.jl` both
clean; both version constants at 21.0.0.

Gate 5 (old bundles still load) was established on a **second, standalone run**, and the
reason is worth recording so the first attempt's output is not misread. Inside the combined
gate script the check aborted with a `Base.EnvDict` lookup stacktrace — `RSROOT` was not
exported for that invocation, an error in the harness rather than in the tree. Rerun with the
variable set, it reports `estimate_base_fc_diagonalW.jls` at 10 fields, `Q = 488.548412`, and
`chain_base_fc_diagonalW.jls` at 32 fields, `Q_best = 488.038115`. Those are the field counts
and values quoted above; no struct gained a field in this version, so deserialization was
never at risk.

### Settings

Added: `MCMC_GAMMA_ADAPT` (`true`), `MCMC_GAMMA_TARGET` (0.234),
`MCMC_GAMMA_ETA` (2.0), `MCMC_GAMMA_EVERY` (25).
Re-defaulted: `MCMC_MOVES_MIN` 2300 → 230.

### Known gaps

**The brake is verified on the aggregate step scale, not on the anisotropy.** The closed-loop
simulation is a scalar surrogate of a 23-dimensional shape problem. The claim that the shape
self-corrects once accepting resumes follows from DE-MC's covariance adaptation but has not
been measured on this model; the first `base_fc` run under this version is the test, and the
per-coordinate spread-to-width table above is the diagnostic to recompute against it.

**The live A/B at the shipped defaults ran at N = 24 / 120 generations**, enough to confirm
the brake engages and the backward-compatible path reproduces, not enough to observe
convergence.

**Solve cost rises as the population disperses.** A 300-generation N = 40 control covered
g = 0→100 in ~600 s and then exceeded that on the next 50 generations before being stopped.
A collapsing chain is progressively more expensive, which strengthens the case for the
acceptance-floor abort.

**The grid dependence in `GRID_INVARIANCE.md` is untouched by this version** and still
contaminates any standard error: `Q(θ̂)` moves 19.4 units between `Np_S` 120 and 160, and
four ability-grid sizes cannot evaluate `θ̂` at all.

### Tried and rejected

**A symmetric (two-sided) `γ` controller — measured worse than the one-sided brake.** From
the shipped collapsed start acceptance is 0.739 at g=25, high because the population has not
spread rather than because the step is short. The symmetric rule read "step too short" and
raised `γ` by **65%** (`exp(1.0·(0.739 − 0.234)) = 1.657`), accelerating the very expansion
it exists to brake. Matched arms at N = 40, `gens` = 150, same seed, differing only in the
multiplier:

| arm | `γ×` at g=50 | `acc` at g=50 |
|---|---|---|
| one-sided brake (idle, `acc` was above target) | 1.000 | **0.370** |
| symmetric controller | 1.657 | **0.196** |

A fixed-`γ` arm was also run but at `gens` = 300, which moves `burn = burn_frac·gens` and so
the outlier-replacement schedule; its 0.555 at g=50 is therefore **not** a matched comparator
and is not used here. Rejected in favour of the one-sided brake. The
asymmetry is licensed by the measured expansion identity — an accepted symmetric proposal
adds its squared step to the population variance and that step is proportional to the
variance already present, with no inward pull — so widening is intrinsic and needs no help,
and overshoot is self-correcting through the same mechanism.

**`η = 1` with a 50-generation cadence — the first defaults shipped, and the worst braking
option measured.** Acceptance only 0.065 by generation 2000 against 0.200 at `η = 2`/25.
Superseded before release.

**The box as the fix for the expansion — falsified.** The v20.0.0 run's gen lines carry no
`box=` field, and that field prints only when nonzero: **zero proposals were rejected by the
prior box in 1000 generations.** The box makes the target proper, which the estimator's
definition requires, but it never engages within this horizon and cannot brake the step.

**"The population is frozen" — falsified.** The `mcse` flatline at 0.1023 across four checks
looked like a frozen population, since a frozen one gives `mcse = 1/√N = 0.1026` identically.
The chain bundle says chains moved on **6.01%** of generations (median 60 whole-vector moves
per chain, 47 distinct values per coordinate). The correct account is
`mcse = (1/√N)·√(1 − R̂⁻²)`, which at `R̂ = 9.664` gives 0.1020 — pinned near the ceiling
because `R̂` stayed near 10, not because nothing moved.

**Diagonal preconditioning of the proposal — a no-op by construction, not retested.** The
DE step is `γ(X_r1,k − X_r2,k)`, so rescaling coordinate `k` rescales the population and the
step identically and cancels.

## v20.0.0 — 2026-08-30

**MAJOR, and it is a respecification of the sampled density, not a numerical fix.** Before
this version the quasi-posterior had no finite normalising constant, so no chain could
converge to it and no standard error computed from one meant anything. `MCMC_T_BOX = 20.0`
now bounds the unconstrained parameter, matching LMR (`main_mpi.f90:146-147`). Posterior
standard errors produced under v19.x are superseded. **Point estimates are not:** inside the
box the target is bit-identical to v19.x, so `argmin Q` is unchanged and every
`smm_result_*` / `estimate_*` bundle remains valid and loadable.

### Why the old target had no posterior

`θ = lb + (ub − lb)·σ(t)` saturates as `|t| → ∞`, so `Q` stops changing and `−Q/2` — the
whole target under `MCMC_PRIOR = :flat_t` — is asymptotically **flat in every coordinate**.
Measured on this model at the `base_fc` optimum (`Q = 490.550869`), pushing one coordinate out
while holding the rest:

| coordinate | Δ=0 | Δ=+15 | Δ=+40 | Δ=+80 |
|---|---|---|---|---|
| `δ_S` | 490.5509 | 8691.6038 | **8691.6063** | **8691.6063** |
| `A` | 490.5509 | 13306216.88 | **13306234.3957** | **13306234.3957** |
| `a_ℓ` | 490.5509 | 241147.24 | **241147.5735** | **241147.5735** |
| `b_S` (downward) | 490.5509 | 490.5550 | **490.5550** | **490.5550** |

Identical to ten significant figures between Δ=40 and Δ=80, in every coordinate, in both
directions — and the entire half-line `t_bS ≤ t₀ − 5` sits within **0.004** of the optimum.
With no bound on `t` the posterior therefore has infinite mass in all 23 directions.

That is what the diagnosis campaign had been chasing. From the 32,000-candidate instrumented
trace: **0 of 23 coordinates equilibrate**, median `d log(sd)/d log(g) = 0.56` against 0.5 for
free diffusion, and total population variance grew **40,258×** over 500 generations. Every
tuning fix failed because none of them addressed this: `CR`, `δ` and the shock magnitudes
cannot make a divergent integral converge, and LMR's outlier-chain replacement is exactly
scale-invariant, so a uniformly expanding ensemble triggers it 0 times in 100.

### Rejection, not LMR's reject-and-redraw

LMR redraw a candidate up to 100 times until it lands in the box (`mpi_mcmc_mod.f90:294-304`).
This version rejects instead, deliberately. Redrawing makes the effective proposal
`q(x→y)·1{y∈B} / Z(x)` with `Z(x)` varying over `x`, so it is no longer symmetric and detailed
balance fails — biasing the retained sample, which is the sample the standard errors come from.
Rejecting is exact, and because the box is tested before the solve it costs nothing: measured
on a synthetic target, 195 solves with a binding box against 492 without, **297 skipped**.

### Reporting

`fin` counts feasibility among candidates the solver actually **saw**; a `box=` field reports
the out-of-box fraction and is silent when zero. The two are kept apart because they mean
different things — a binding box on an everywhere-finite target would otherwise print
`fin = 0.35`, reporting the prior as model fragility. The header records the box beside `δ`
and `CR`, since a log that omits it cannot say which density was sampled.

### Gate

| item | result |
|---|---|
| 1. parse | OK — `demc.jl`, `MCMC_main.jl`, `settings.jl` |
| 2+3. entry point on real data | exit 0, `box=±20` in the header, `Q(θ̄) = 490.0780` finite |
| 4. version constants agree | `20.0.0` in both |
| 5. old bundles load | no struct or field changed; `t_box = Inf` reproduces v19.x bit-for-bit (`max|Δchain| = 0.000e+00`) |
| 6. `check_forwarding`, `check_dead_settings` | PASS |
| 7. changelog | this |

Three box branches verified individually on a synthetic target, because a short run on the
real model never trips them — the first gate attempt at `±20` and `±13` came back
**bit-identical**, since early steps are ~1e-4 and no candidate leaves the box in 30
generations:

- out-of-box candidate rejected without a solve (297 skipped, `box=0.58` printed);
- seed outside the box aborts naming the coordinate and value, rather than running all-reject;
- a non-binding box is bit-identical to no box.

**A trap for anyone comparing two box widths.** `MCMC_CHECKPOINT` overwrites
`estimate_base_fc_diagonalW.jls`, so a second run in the same tree seeds from the first run's
`theta_best` and the arms are no longer comparable. It caught the follow-up gate here: at
±12.5 then ±12.0, `Q(θ̂)` went `490.5509 → 488.5299` — the first run's best point — so the
±12.0 arm never tested the seed-outside-box path it was written for, since its inherited seed
was inside. Set `MCMC_CHECKPOINT = false` for any width comparison. (The ±20/±13 pair above is
unaffected: `Q(θ̂) = 490.5509` in both, so nothing was promoted and the identical result is the
box genuinely never binding, which contamination could not have produced — it makes runs
differ, not agree.)

### Known gaps

- **The fix is unverified on the real chain.** That it makes the target proper is established
  from the table above. That it holds the population over hundreds of generations is shown
  only on a surrogate — a quadratic in `θ` composed with the real box transform, calibrated to
  the trace's measured conditional scales, on which the log-variance drift per 1000 generations
  was **+17.87** unbounded against **−0.033** boxed. The diagnostic pair that would settle it on
  this criterion (`T_BOX = Inf` versus `20.0`, N=40, 400 generations, ~11 min each) has not
  been run.
- **The box width is a prior choice.** At the `base_fc` seed `max|t| = 12.46` (`b_S`), so ±20
  leaves 7.5 `t`-units — three measured posterior sd — at the tightest coordinate, and 1 of 23
  would have `seed ± 3s_k` cross it. A persistent nonzero `box=` field means it is binding on
  an identified part rather than only on a flat tail, and wants widening.
- **`MCMC_PRIOR = :flat_theta` is the other way to make the target proper** and was rejected
  for this release, not overlooked. It moves the estimate: at `b_S`'s seed the Jacobian
  gradient is `+0.999992` against a criterion gradient of `−2.73e-05`, a ratio of **36,585**, so
  it would push `b_S` off zero — and `b_S = 0` is the entry that has to sit beside LMR's
  `0.000 (0.032)`. The box leaves the target untouched inside it. This one is the theorist's
  call, not a numerical preference.
- `check_output_tree` rule 6 flags backup debris in `output/estimates/` — the
  `_backup_Q497.786594.jls` left by the killed v19.8.0 run. Pre-existing, not introduced here.
- **Sampling `θ` directly instead of its logistic preimage was started, not finished, and is
  DEFERRED rather than rejected.** The scaffolding ships behind a guard (`MCMC_SPACE`, default
  `:t`; `:theta` errors out with the reason. Also `MCMC_WIDTHS_CSV` and `MCMC_INIT = :widths`,
  unreachable because they require `:theta`). Nothing in v20.0.0's numbers comes from it.

  **Why it is worth finishing.** `dθ/dt = θ−lb` for the box map, so under `:t` a coordinate
  sitting a few 1e-06 above its floor needs hundreds of `t`-units to span its ΔQ=1 width while
  the chain's `t`-sd is ~0.04 — orders of magnitude short, and that shortfall is the reason
  `b_S`'s reported standard error is too small on `base_fc`. The specific figures quoted for
  this coordinate in earlier notes (width 0.0037, 586 `t`-units, a factor of 1.5e+04) are
  **NOT reproduced and should not be cited**: re-running `code/tools/width_audit.jl` on the
  `base_fc` postmean base point returns `ΔQ=1 up = NaN` and `ΔQ=1 dn = NaN` for
  `skilled outside flow b_S` at `θ̂ = 6.2552e-06`, i.e. the bracket closes on an infeasible
  point in both directions, so no finite width for `b_S` has actually been measured. What IS
  measured at that base point is `se(chain) = 9.843e-09` for `b_S` against ΔQ=1 widths of
  1e-05 to 1e-02 for every bracketable coordinate — the qualitative undercount stands, its
  magnitude does not, and pinning it down needs a base point where `b_S` brackets. A flat prior on
  `θ` is also flat on the object the paper interprets, whereas a flat prior on `t` asserts a
  priori that each parameter is more likely near its own bounds, by an amount set by the box
  width — not a belief anyone holds.

  **Why it is off.** Measured on `base_fc`: `max logπ = −5.178e+05` against `−2.447e+02` under
  `:t`, `dlp = +24.37` at gen 10 (positive — climbing away from the seed), and
  `Q(θ̂) = 16,561,006` for the very seed vector that scores 490.55 under `:t`. **Two known
  causes, both fixable; neither is a reason to abandon the design.**
  1. **`MCMC_B_ADD` must become per-coordinate.** It is the only non-scale-free element in the
     proposal and it is what produced the 16.5-million `Q` — not the change of space. What is
     already fine: DE-MC's step is `γ·(X_r1,k − X_r2,k)`, drawn from the population's own
     spread in each coordinate, so per-coordinate and region-dependent already; `MCMC_B_MULT`
     multiplies that difference vector, so it is scale-free too. `MCMC_B_ADD` is a single
     SCALAR added to every coordinate: in `t` the logistic puts every coordinate at O(1) so one
     scalar serves all, but in `θ` they span 6.3e-06 (`b_S`) to 11.1 (`c`) and that scalar is
     16x `b_S`'s whole value while being negligible for `c`.
  2. **`constrained = true` reaches only `logposterior`.** `Q(θ̂)`, `Q(θ̄)`,
     `Q(res.theta_best)`, `_Q_from_lp`, the `Ĝ` Jacobian block and `unpack_θ` at both
     reporting call sites still read the chain vector as `t`. Mechanical, but six sites, and
     missing one reports a wrong number rather than failing.

  **Do not switch `:theta` on before both are done** — it completes and reports nonsense.
  Adopting it later is a MAJOR bump: the estimand moves.
- **`code/tools/width_audit.jl` is new here and does work** (exit 0): per-coordinate ΔQ=1
  half-widths for a window, from which a (width)/se ratio far from 1 says the reported
  standard error is not describing the criterion. It is standalone — it reads no `:theta`
  code — and it is what produced the `b_S` finding above. Its consumer inside
  `MCMC_main.jl` (`MCMC_INIT = :widths`) is the part that is disabled, not the tool.
  That loader also has its own unfixed bug: it rejects a one-sided bracket, erroring on
  `b_S` although `wQ1_up = 0.003665` and only `wQ1_dn` is NaN — correctly NaN, since `b_S`
  sits at its floor. Unreachable as shipped, so not fixed here.

---

## v19.8.1 — 2026-08-29

**PATCH.** Prints only. Every number in every output file is bit-identical to v19.8.0 — the
criterion, the sampler and the serialised fields are untouched, so nothing stored is superseded
and the `base_fc` run in flight under v19.8.0 is unaffected either way (Julia read the files at
launch).

### The gap this closes

`n_replaced` and `last_replace` were computed and **returned**, never printed. With
`MCMC_CHECK_EVERY = 0` and no mid-run chain write, an 11-hour run therefore gave no live signal
on the stuck-chain replacement — the one mechanism that concentrates a dispersed ensemble, and
the thing the `base_fc` run's `R̂ = 7.567` at generation 250 turns on. The information existed
only in the bundle written at the very end.

### What the print now says

`demc.jl:517-519` adds a conditional field to the generation line, matching the `fin` pattern:

- **silent** when no replacement fired since the last print. On a healthy population the rule
  flags roughly 0.08 chains per generation, so its *presence* is the signal.
- **`rep=N`** — N events in this window, below the burn boundary. Harmless: those draws are
  discarded.
- **`rep=N RETAINED`** — past the burn boundary. This is the reading that changes what the
  reported SD *means*: a replacement there puts duplicated draws in the pooled sample, and those
  draws are not from the target, which is the requirement CH Theorem 2 integrates against. Spelled
  out rather than flagged, because it is the one case a reader must not miss.

The window counter `wrep` (`demc.jl:366`) resets with the other per-print counters.

Also: the MCMC banner is now `RoySearch vX — DE-MC` rather than `— DE-MC standard errors`, since
the run produces the point estimate as well; the matching section header in
`settings.jl:157` follows it.

### Gate

| item | result |
|---|---|
| 1. every changed file parses | OK — `demc.jl`, `MCMC_main.jl`, `settings.jl` |
| 2+3. entry point loads and runs on real data, scratch dir | exit 0, `Q(θ̄) = 492.8447` finite |
| 4. both version constants agree | `19.8.1` in both (see the helper fix below) |
| 5. old serialised bundles load | no struct or field changed in this version |
| 6. `check_forwarding.jl` | PASS, as do `check_dead_settings` and `check_structure` |
| 7. changelog entry | this |

**Verified live:** `gen 40/60 … fin=0.79 rep=1`, with the field absent from the other five
generation lines of the same run.

**Verified by enumeration, not by a run: the `RETAINED` branch.** `burn` is defined once at
`demc.jl:157` from the requested `gens` and is the same object the replacement gate reads at
`:448`, so the two agree by construction, and the printed expression is a pure ternary on
`g > burn` whose three cases are exhaustive. A live test was launched and abandoned after 25
minutes having written **zero bytes** — so it never reached its first print, and the cause is
undetermined: contention with the production run holding all 10 threads on a 10-core machine is
the likely explanation but was not established. Either way, competing with an 11-hour job to
exercise a print line was the wrong trade. Recorded here rather than left implied.

### Fixed alongside: the `roysearch-versioning` helper had been broken since v19.6.0

`check_version_sync` listed `code/data_processing/data_processing_main.jl`, which became
`code/data_and_descriptives/` in the v19.6.0 restructure. It counted the absent path as
`missing`, `in_sync` required `not missing`, and `bump_version` refuses when not in sync — so the
helper had refused **every** bump from v19.6.0 onward, and v19.6.0 through v19.8.0 were all
bumped by hand instead.

The fix distinguishes two faults the original conflated: a listed path this tree does not have is
`absent` and skipped, while a path that exists *without* the constant is still `missing` and
still blocks, because that one really is a banner that cannot read the version. `VERSION_FILES`
now lists both names, and `in_sync` additionally requires at least one file found.

### Known gaps

- The `RETAINED` branch has not been exercised by a live run (above).
- `rep` reports replacement **events**, not cells touched. Under the LMR scope one event rewrites
  `retain_len` cells, so the cell count is `events × retain_len` and is recoverable exactly from
  the `replaced` mask in the bundle. The print reports events because that is the quantity a
  tuning decision reads.

---

## v19.8.0 — 2026-08-29

**MINOR.** The MCMC is configured to match Lise, Meghir and Robin (2016) — sampler settings,
budget, burn-in, stopping behaviour and outlier handling — so a table produced here is produced
under their published procedure. The criterion is untouched; only the sampler's configuration
changes, so stored estimates remain comparable.

### The diff, every row sourced

| | LMR | was | now |
|---|---|---|---|
| `CR` | 0.75 (`main_mpi.f90:127`) | 0.95 | **0.75** |
| `δ` difference pairs | 2 (`:131`) | 1 | **2** |
| `b_mult` (`shock_mult_std`) | 1e-2 (`:136`) | 1e-5 | **1e-2** |
| `b_add` (`shock_add_std`) | 1e-4 (`:137`) | 1e-4 | unchanged |
| mode jump `γ = 1` | every 10th gen (`mpi_mcmc_mod.f90:295`) | every 10th | unchanged |
| chains | 100 (paper p.86) | 64 | **100** |
| generations | 10,000 (`max_iteration`) | 4,000 | **10,000** |
| burn-in | last 1,000 of 10,000 = 0.9 (p.86) | 0.5 | **0.9** |
| initialisation | all chains at one vector | `:at_seed` | unchanged |
| prior | uniform on the **transformed** parameter | `:flat_t` | unchanged |
| sequential stop | none | `check_every = 250` | **0** |
| acceptance-floor abort | none | 0.02 | inert (stop off) |
| outlier replacement, timing | every generation, ungated | burn-in only | **ungated** |
| outlier replacement, scope | the chain's whole retained buffer | its current position only | **retained window** |
| point estimate | mean of chain-space draws, then transformed | ✓ since v19.7.2 | unchanged |
| standard error | SD of transformed draws | ✓ | unchanged |

### The prior, verified rather than assumed

`starting_val.raw` holds **unconstrained** values (−4.31, −1.10, 1.94, …) and
`policy_UI_ed1_RED/src/model/params.f90:156-172` transforms each one — `exp(θ)` for the positive
parameters, `exp(θ)/(exp(θ)+exp(−θ))` for the unit-interval ones, `0.5·logistic(θ)` for `delta`
and `chi`. So LMR sample in transformed space exactly as this codebase does, their prior box is a
box **on the transformed parameter**, and the matching convention is `MCMC_PRIOR = :flat_t` —
already the default. `mpi_mcmc_mod.f90:485` averaging that buffer and `InitExogenousParameters`
transforming it is independently the same operation as the v19.7.2 mean fix.

This corrects an earlier reading in which `:flat_theta` looked like the LMR match on the grounds
that a uniform prior on the *economic* parameter is what a box rejection implies. Their box is
not on the economic parameter.

### Where each number comes from, and one correction to how I first sourced them

**Magnitudes from the paper; mechanism from their code.** The replication package's
`main_mpi.f90` ships `chain_count = 95` and `chain_length = 500`, and an earlier draft of this
entry treated those as authoritative — concluding 95 chains and a 0.95 burn-in, and calling the
paper's "100 chains … last 1000 elements" a contradiction. That was wrong: those are *settings*,
no more authoritative about the published run than this file's own `MCMC_*` defaults are about
one of ours. They record what was in the archive when it was zipped. Appendix C p.86 is the
description of what produced Table 5, so **100 chains and a 0.9 burn-in** ship.

What legitimately comes from the code is *structure*, which is not a setting: the acceptance rule,
the DE-MC proposal form, `solution_theta` being the buffer mean, and the outlier rule together
with its scope. `CR`, `δ` and the two shock magnitudes are stated nowhere in the paper — footnote
22 defers tuning to Robert & Casella and Vrugt et al. — so the package is the only evidence
available for them and ships as such, which is weaker ground than the rest of the table and is
recorded here as weaker.

### The outlier replacement now matches their scope, not just their timing

LMR do `all_population_knl(:,C,:) = all_population_knl(:,best_chain,:)`: the outlier chain's
entire buffer is overwritten with a copy of the best chain's. Ours holds the full generation
history rather than a cyclical buffer, so the faithful analogue is the span that will actually be
pooled — `retain_len = gens - burn`, derived from `burn` so it tracks any `burn_frac`.

**The consequence, stated because it bears on the reported SD:** a replacement firing inside the
retained window puts duplicated draws in the pooled sample, which narrows the SD, and those draws
are not from the target — the requirement Theorem 2 integrates against. LMR's defence is that at
10,000 generations replacement is extinct long before the retained window. That is now checkable
rather than assumed: `replaced` records every cell touched, so `last_replace` past the burn
boundary is the signal, and an uncontaminated mean is a one-line alternative.

Verified live: 2 replacement events marked **12 of 1200** cells at `retain_len = 6`, i.e. 2 × 6 —
a single-cell implementation would have marked 2.

### One deliberate deviation

1. **`CR = 0.75` is theirs, not the local optimum.** A measured sweep on this criterion preferred
   0.95: the cost multiplier κ falls from 1.330 at `nmask ≤ 4` to 0.868 at `nmask ≥ 8`, monotone
   under stratification on step length and significant on all three rank correlations. Their value
   ships because the instruction was to match them; `ROYSEARCH_MCMC_CR=0.95` restores the local
   optimum if the acceptance comes back too low to be usable.

### Persisted so the estimator can be changed without re-running

`run_demc` computed a per-chain log-target history for the outlier score and **discarded it**. It
is now returned and serialised, with two masks:

- **`chain_lp`** — the log-target of every draw. Under `:flat_t`, `logπ = −Q/2`, so this *is* the
  Q of every candidate: `Q = −2·chain_lp`, exactly, no solves. Without it the 1,000,000 candidates
  in the bundle have no fit attached, and recovering it means re-solving all of them — the whole
  run again. With it, any estimator that selects or weights draws by fit is post-processing.
- **`replaced`** — which chain-generations the outlier rule overwrote, so non-target draws can be
  excised. LMR keep the same object (`all_outliner_results_nl`).
- **`accepted`** — which draws are new states rather than repeats.

The draws also now reach disk **the moment `run_demc` returns**, not in section 7 after ~600
further solves for Ĝ and the diagnostics. Anything throwing in between previously discarded the
draws of a run that had already completed; at the shipped budget that is eleven hours.

Verified by rebuilding five estimators from the saved file alone — mean, median, best draw,
best-decile-by-Q mean, uncontaminated mean — all feasible, no re-solving beyond scoring the five.

**The boundary, since it is the one thing this does not cover:** per-candidate *moment vectors*
are not stored (only the 600 thinned draws in `moments`/`draws_jac` carry them), so an estimator
that reweights the moments under a different `W` would still need re-solving.

### What turning the stop off means

`MCMC_CHECK_EVERY = 0` disables the sequential stop, which also makes `MCMC_MOVES_MIN`,
`MCMC_DRIFT_FLAT` and `MCMC_ACC_FLOOR` inert — they are that stop's own thresholds. R̂ and ESS are
still **computed and reported**; they simply cannot terminate or abort a run, which is LMR's
behaviour (`main_mpi.f90` runs `max_iteration` to completion and nothing inspects either). A
consequence worth knowing: no run can now end in `stage = :mcmc_aborted`, so the v19.7.1
draws-survive-an-abort path is belt-and-braces rather than the main road.

### Cost

95 × 10,000 = **950,000 solves**. At the measured 0.395 s per solve that is ≈ 104 h
single-threaded, ≈ **10.5 h on 10 threads**. `ROYSEARCH_MCMC_N` and `ROYSEARCH_MCMC_GENS` shorten
it without changing the procedure.

### Verified

Parse clean; `check_forwarding`, `check_dead_settings`, `check_structure` all pass. A short live
run at the new settings with the budget cut only: `δ=2 CR=0.75 init=at_seed`,
`gens=40/40 (no stopping test)`, `burn=36`, `Q(θ̄) = 493.66` — **feasible** — `stage=mcmc`,
`se=chain`, post-mean bundle written.

---

## v19.7.2 — 2026-08-29

**MINOR.** The posterior mean is now averaged in the **sampled (unconstrained)** space and then
transformed, instead of transformed draw by draw and then averaged. That is what LMR do, and it
is the difference between a reported estimate the model can be solved at and one it cannot.

### The measurement

Same draws, same burn-in, 2250-generation × 64-chain `base_fc` bundle:

| | Q(θ̄) |
|---|---|
| mean in **constrained** space (through v19.7.1) | **Inf** — no equilibrium |
| mean in **unconstrained** space (v19.7.2) | **524.81** — feasible |

The box transform is monotone but nonlinear, so `to_constrained(mean(t)) ≠
mean(to_constrained(t))`, and only the first lands inside the equilibrium-existence region.

**The gap is Jensen's inequality and its sign is fully predicted.** The box map is a logistic,
convex below the box midpoint and concave above, so `mean(σ(t)) > σ(mean(t))` for a coordinate
whose mass sits low in its box and `<` for one sitting high. Measured across all 23 coordinates:
**23/23 signs agree** with that prediction, and `corr(|gap|, |box position − ½|) = +0.55`. The
gaps are small — median **0.011** posterior SD, largest 0.223.

**And the infeasibility localises to ONE coordinate.** Swapping each coordinate of the
constrained mean to its unconstrained-mean value, one at a time: `δ_S` alone restores feasibility
at **Q = 517.10**; no other single swap does. `δ_S` sits at box position 0.918 with a −0.189 SD
gap. So this supersedes the "not through `b_S` alone, therefore joint" reasoning in the v19.7.1
entry below — it is not joint, it is `δ_S`.

`δ_S` is the shock-support upper bound, i.e. the parameter whose density singularity sits at
`p = δ_S` on a fixed Gauss-Legendre grid. **It is tempting to call this the D2 perforation and
that would be wrong on two counts, both checked:**

- **D2 was fixed at v19.0.0.** R1 (softened OJS selection in `compute_Jbar_skilled`), R3
  (exact-CDF cell masses via `build_cell_mass_density`) and `DAMP_THETA_S = 0.9` together took
  the share of nearby points returning a finite `Q` from **58.0% to 98.3%**, and the grid CV of
  `Q` over `Np_S ∈ {100…200}` from 20.33% to 0.061%.
- **The node-coincidence mechanism does not explain this instance.** The two means sit 0.121 and
  0.160 cell widths from their nearest node at `Np_S = 120`, with **no node between them**. At
  the *old* optimum `δ_S = 0.9726` sat 2.25e-07 cell widths from node 108 — and `Np_S = 120` was
  the global argmin of that distance over `N ∈ [60, 1200]`, which is what an optimiser does with
  a free quadrature bonus. The current `δ_S ≈ 0.9226` is nowhere near that coincidence.

So this is one of the residual ~1.7%, cause **unattributed**. That residual was measured at the
v19.0.0-era optimum and has not been re-measured at the current point.

**This is not a multimodality problem.** Applying LMR's own outlier-chain rule — drop any chain
whose mean log-target falls below Q1 − 2·IQR, `mpi_mcmc_mod.f90` — retains **64 of 64** and moves
neither number. The ensemble is already unimodal by their test. The transform order was the whole
of it.

### Why this is what LMR do

The old comment claimed *"as LMR do (read_MCMC_chain.m transforms each parameter and then
averages)"*. That is correct about their **standard errors** and wrong about their **point
estimate**, and the two live in different files:

- `mpi_mcmc_mod.f90:485` — `solution_theta = sum(sum(all_population_knl,3),2)/(N*L)` averages the
  **raw chain buffer**, in chain coordinates, untransformed. That vector is what reaches
  `data/starting_val.raw`, whose transform reproduces their published Table 5 to the last printed
  digit.
- `read_MCMC_chain.m` transforms draw by draw **before** `std()` for the standard error.

So the asymmetry is theirs and deliberate: **mean in chain space, SD in model space.** This
release matches it. `se_chain` was already computed on transformed draws and is unchanged.

### What it does not change

CH Theorem 2 covers this object under either transform order — it is the Laplace-type estimator
under squared loss, consistent and first-order equivalent to the extremum estimator, requiring no
derivative of Q. What the order decides is whether the estimator is a point the model can be
**solved** at, which is a separate and harder requirement: every table, figure and counterfactual
has to be computed somewhere.

Verified in a live forced-abort run: `Q(θ̄) = 494.8076`, finite, and the post-mean bundle writes
again (its `isfinite(Q̄)` guard had been silently suppressing it on every run).

---

## v19.7.1 — 2026-08-29

**MINOR.** An aborted chain no longer discards its own draws. `se(chain)`, the posterior
mean, R̂, ESS, the edge fractions, the quantiles and the spread growth are now computed
whenever draws exist, instead of only when the run terminated cleanly. The criterion is
untouched and no previously-reported value changes — the affected slots previously held
`NaN`.

### The defect

`MCMC_main.jl` decided two different questions with one flag:

```julia
use_design = MCMC_JAC_ONLY || res.aborted
chain_ok   = !use_design && res.gens > 0
```

and `chain_ok` then gated **eleven** downstream quantities. *Which Ĵ to use* and *whether the
draws have a standard deviation* are not the same question, and they do not have the same
answer. A non-stationary sample still has a spread.

The consequence was not hypothetical. The chain aborts on the **acceptance floor** — a
step-scale diagnostic that says the proposal is too long for the basin, not anything that
invalidates the draws. On the shipped `base_fc` chain bundle (2250 generations × 64 chains,
72,064 retained draws) that meant `NaN` in all 23 `se_chain` slots, while those same draws on
disk give **22 of 23 parameters a relative posterior SD under 35%, 15 of them under 5%** —
`ρ_x`, the parameter the paper's contribution rests on, at −0.8166 (0.0112), a 1.4% width. The
numbers were there. The code declined to compute them.

### The split

`use_design` keeps its meaning: an aborted chain's draws are a poor finite-difference design,
so the Ĵ columns fall back to a local design around θ̂. A new `draws_ok = res.gens > 0 &&
size(res.draws, 2) > 1` governs the draw-derived quantities. `JAC_ONLY` is excluded by
construction — its stub sets `gens = 0` and a single draw column.

Thirteen sites moved to `draws_ok`. Three deliberately did **not**:

- `jgap` and its banner line — the cross-check compares the *chain covariance* against Ĵ, and
  under abort the design is local, so there is genuinely nothing to cross-check.
- `est_rep` — so an aborted run reports **θ̂, not the posterior mean**. With `se_rep` now being
  `se(chain)`, that path reports θ̂ ± the spread of draws around θ̂, which is what the chain
  measured: it was seeded there. The alternative would put a possibly-infeasible vector in the
  table, because on `base_fc` the coordinate-wise posterior mean gives **Q(θ̄) = Inf** — and not
  through `b_S` alone, since snapping `b_S` to its bound leaves it infeasible. The feasible set
  of a segmented model with an equilibrium-existence condition is not convex, so an average of
  feasible draws need not be feasible.

The post-mean bundle was moved to `draws_ok` safely: it re-solves `Q(θ̄)` and its existing
`isfinite(Q̄)` guard means an infeasible mean writes nothing, so a warm start cannot be
poisoned by it.

### Reporting honesty

The sealed bundle's `tag` read `"standard errors from Ĵ; draws are not stationary"` for every
aborted run. True while an abort discarded the draws; false now. It names the route actually
taken, because that string is what a table footnote gets written from. `se_source` is `:chain`
on this path, and the `mcmc` field now records **both** flags — `draws_ok` true with `chain_ok`
false is precisely "these standard errors are the spread of a non-stationary sample."

What that spread is, stated for the paper rather than hidden: with R̂ ≈ 1.7 the pooled variance
is ≈ 2.9× the within-chain variance, so the reported width already absorbs the disagreement
between independently drifting chains rather than understating it. It is not a posterior SD in
the Chernozhukov–Hong Theorem 2 sense, which needs stationarity; it is a conservative measure
of the region the criterion cannot separate. Report it, print R̂ beside it, and say so.

### Verified by observation

A forced acceptance-floor abort (40 generations, `ACC_FLOOR = 0.99`, aborting at g=24) in a
scratch tree produced `stage=mcmc_aborted se=chain`, the correct tag, and a reportable CSV
with **23/23 finite** in `se_chain`, `post_mean`, `rhat`, `ess`, `edge_frac`, `q025` and
`q975`. An earlier 12-generation run showed `NaN` for R̂/ESS — three post-burn generations
cannot be split — which is the toy size, not the code.

---

## v19.7.0 — 2026-08-29

**MINOR.** The shipped optimiser is `method = :sa`. Differential evolution is no longer run.
A run does not reproduce v19.6.0's optimiser path, but the **criterion is unchanged** — the
same θ scores the same Q — so every stored estimate stays comparable and no reported number
moves on account of this release.

### Why DE was dropped

Priced from the live `base_fc` trace at the measured 0.395 s per solve:

| | solves | wall time | descent |
|---|---|---|---|
| SA, observed | 500 | 3.3 min | **6.375 Q units** |
| DE population generator, *per reheat* | 1,932 | 12.7 min | **zero** |
| one DE generation (`pop_size` 240) | 240 | 1.6 min | — |
| DE at `max_iter` 3000 | 720,000 | **79 h** | — |
| SA at `max_iter` 30000 | 30,000 | 3.3 h | — |

DE's generator setup alone costs 3.9× what SA spent to find 6.4 Q units, and buys nothing
before its first generation. Its full budget is 79 hours against SA's 3.3. SA measures its own
proposal widths (`[SA scale]`, 552 solves), so it is self-contained and loses nothing when DE
is removed.

The operational evidence agreed: across six weeks the pattern was "runs, stalls, then the MCMC
finds better points." DE's selection rule is `Q_u < Q_old` — strict, with no temperature — so a
plateau is absorbing for it by construction, while SA accepts uphill moves and the DE-MC
sampler accepts them too. That is why the sampler kept beating the optimiser.

DE is **not deleted**: `method = :sa_de` and `method = :de` still work, and every `de_*`
setting is still read on those paths. It is no longer the shipped default.

### The bug this surfaced

Switching the method exposed a forwarding gap the previous default had hidden. The `:sa`
branch of `run_smm` held its **own copy** of the 28-keyword `_run_sa` argument list, and the
copy had drifted: it omitted `checkpoint_path`. `run_smm` accepted the argument and the branch
dropped it, so `_sa_loop` took the `""` default and wrote nothing. Because the checkpoint
fires at **every reheat** (`smm.jl:960`), an `:sa` run would have serialised only at the very
end, and a kill would have lost the point — the same loss that cost a `base_covid` run at
v18.4.0.

The duplicate list was **deleted rather than repaired**. Both methods now reach the annealer
through the single `_sa_stage` closure, so `:sa` and `:sa_de`'s first stage cannot diverge
again and a new SA setting reaches both or neither. The comment above `_sa_stage` had already
predicted this: *"a second copy of either argument list would be a second place to update when
a setting is added."*

Verified by observation, not by inspection: with `sa_reheat_patience` lowered in a scratch tree,
a run wrote four reheat checkpoints (`SA reheat 1..4`, Q = 497.7866 → 492.6360) and the bundle's
mtime changed mid-run.

### `check_forwarding.jl` was complicit, and is now stricter

The gate unioned two SA anchors, so a keyword present at **either** call site counted as
forwarded. That is exactly how the `checkpoint_path` gap passed it for a version. With one call
site the union is gone and the check is strict. DE keeps its union, because its `:de` branch does
still call `_run_de` directly.

### Unchanged, deliberately

No budget, temperature, reheat, or population setting was touched. `sa_max_iter` stays at
30,000, `sa_reheat_patience` at 400, `sa_max_reheats` at its env-settable default of 4,
`sa_t0_rel` at 1e-4. The reheat behaviour was diagnosed, not altered.

Two things observed and left alone, recorded here so they are not rediscovered:

- **`sa_reheat_patience` and `sa_reheat_factor` are not env-settable** (hardcoded 400 and 4.00),
  while `sa_max_reheats` is (`ROYSEARCH_SA_MAX_REHEATS`, default 4). Tuning reheat *triggering*
  currently needs a code edit.
- **A steadily-improving run writes no checkpoint.** The checkpoint fires only at reheats, and a
  reheat needs 400 consecutive non-improving iterations, so a run that is descending well has
  nothing on disk until it finishes.

---

## v19.6.0 — 2026-08-29

**MINOR.** The repository was restructured: one bundle shape, one path module, `smm` and
`mcmc` separated, exhibits given their own directory, dead code purged. **The criterion is
untouched** — `base_fc` re-solves at `Q = 497.786594` from the relocated bundle against the
same stored value, so every existing estimate stays comparable and no result moves.

What changed for a *reader* of the tree, which is the part worth knowing: results now live
in `output/estimates/` as `estimate_<window><suffix>.jls`, not in `output/smm/` as
`smm_result_...`. An external script pointing at the old path will not error — it will find
nothing.

### The bundle contract

Four writers produced four different field sets, and since MCMC overwrites the SMM bundle
in place — that is the design, not an accident — whichever writer touched a window last
decided what its file contained:

| writer | fields it wrote |
|---|---|
| `smm_main.jl` (SMM final) | `result, spec, sim, provenance` |
| `smm.jl` (SA/DE checkpoint) | `result, spec, checkpoint, tag` |
| `MCMC_main.jl` (MCMC checkpoint) | `result, spec` |
| `MCMC_main.jl` (posterior mean) | `result, spec, provenance` |

The audit that prompted this found `base_fc` — the one converged window — carrying **two
fields**, written by the poorest of the four. No provenance, so a table could not record
which version produced the number; no marker, so a table could not tell an optimiser point
from a sampled one.

`code/smm/bundle.jl` is now the only writer. The field set is **constant** —
`schema, stage, result, spec, provenance, sim, se, se_source, mcmc, tag` — with absent
information held as `nothing` rather than an absent field, because an absent field is
ambiguous between "this writer does not produce this" and "this file predates the field",
and those call for different handling. `read_bundle` normalises a schema-1 file to the same
shape in memory, inferring `stage` conservatively.

`stage` is the load-bearing addition. `:smm` and `:mcmc` are finished estimates;
`:smm_checkpoint` and `:mcmc_checkpoint` are mid-run resume artifacts that a figure must
refuse; `:mcmc_aborted` says the draws are not stationary, so a table printing its standard
errors owes the reader the abort reason; `:mcmc_postmean` marks a file that has always had
an estimate's name shape and a different meaning; `:superseded` marks a point that no longer
solves.

**`SMMResult` is untouched.** It is a struct, read positionally by the deserialiser, so
adding a field would break every stored bundle. The bundle is a NamedTuple read by name,
which is what makes the shape change safe.

**New at the end of an MCMC run:** the estimate bundle is re-sealed with the point the run
leaves *plus its standard errors* — all three columns, named, with `se_source` saying which
to lead with, and an `mcmc` field carrying the diagnostics a footnote needs. Until now the
standard errors went only to a CSV and the chain bundle, so a plotting script had to open
two files and trust they described the same point.

### One path module

`OUTPUT_DIR`, `PLOTS_DIR`, `TABLES_DIR` and `SMM_OUT_DIR` were declared independently in
**seven** files and `PROJECT_ROOT` in **ten**; two of them used `if !@isdefined(...)` guards,
so their effective value depended on which file was included first. That is the same defect
class as a setting computed in one place and read in another.

`code/paths.jl` is now the sole definition, and validation is the point:
`out_plots(:model, "base_cf")` is a startup error, not a directory called `base_cf` sitting
quietly beside `base_fc` with one figure in it. Thirteen sites constructed the bundle
filename; all thirteen now route through `estimate_path`, so there is one place to change.

    output/
    ├── estimates/    TRACKED   estimate_<w><suffix>.{jls,csv} — one per window
    ├── chains/       ignored   draw arrays, deletable once the SEs are sealed
    ├── transition/   TRACKED   read by plots_and_tables/transition.jl
    ├── logs/         TRACKED   the only record of what a run did
    ├── plots/        ignored   descriptives/ fit/<w>/ model/<w>/ transition/<pair>/ policy/<w>/ manual/
    └── tables/       ignored   .tex ONLY

A `.csv` is a machine artifact and lives beside the bundle it came from, so a table
directory can be deleted and regenerated without thinking about it. `manual/` replaces
`standalone_default/` and `single_run/`, which were one thing under two names, and
`out_tables(:manual)` is an error — a solve at arbitrary parameters produces figures to look
at, never a paper table.

### Layout

- **`code/mcmc/`** — `MCMC_main.jl`, `demc.jl`, `mcmc_diagnostics.jl`. The dependency
  direction is now stated and checked: `mcmc` may include from `smm`, never the reverse,
  because the MCMC samples the SMM criterion and is downstream of it.
- **`code/plots_and_tables/`** — `model.jl`, `transition.jl`, `transition_panel.jl`,
  `policy.jl`, one per category, at the level of `smm/`, `solver/` and `transition/`. The
  files were **moved, not rewritten**: their three mains still include them from the new
  location, so behaviour is unchanged. Their content — and the entry point that will replace
  those includes — waits on the exhibit list.
- **`code/gates/`**, **`code/tools/`**, **`code/ops/`** replace `code/scripts/`, which held
  release gates, one-off diagnostics and VM launchers in one directory.
- **`code/data_and_descriptives/`** replaces `code/data_processing/`.
- **`code/repo_root.jl`** sits beside `paths.jl`: it is shared by `gates/`, `tools/` and
  `ops/`, so one path up is the same expression from all three.

### Purged

- **`moment_covariance.jl`** (252 lines) — referenced by nothing, standalone by its own
  header. It computed a full moment covariance for a non-diagonal `W`; that work is now done
  by `load_weight_matrix` with `W_COND_TARGET`. Deleted rather than kept, because a file
  nothing calls is a file nobody notices has gone stale.
- **`code/single_run_plots_all/`** (20 PNGs) and **`code/notebooks/`** (37 files) untracked.
  Both carried `.gitignore` rules that had never bitten — a rule cannot untrack a path
  already in the index — along with `output/tables/` (9) and `output/plots/` (191). The
  `git rm --cached` in this release is what made those four rules effective, removing 257
  files from version control with no code change.

### New gate

`code/gates/check_output_tree.jl` — nine rules, each a filesystem check or a grep: `.tex`
only in `tables/`, images only in `plots/`, no artifacts under `code/`, `savefig` only in
`plots_and_tables/`, exact window and pair subdirectories per category, permitted filename
patterns only in `estimates/`, no path literal outside `paths.jl`, no `smm → mcmc` include,
and every bundle `serialize` through `write_bundle`. Same discipline as the settings
registry: enforced rather than remembered.

Building it caught two of its own findings as over-broad rules and three genuine offenders
(`rank_diagnostic.jl` writing a CSV into `tables/`, and `migrate_bundles.jl` writing a `.new`
sibling and staging inside `output/`). It currently reports **7 findings, all legacy content
in `output/tables/`** awaiting the cleanup pass — the gate is doing its job, not failing.

### Consequences elsewhere

- **`BUNDLE_SCHEMA` was nearly duplicated.** `bundle.jl` defined its own alongside the one in
  `smm_params.jl` — which already existed with exactly this meaning ("Current bundle-format
  version. Bump only on a shape change") and feeds `RunProvenance.schema`. Julia warned on
  every run, and `provenance.schema` would have reported 1 or 2 **depending on include
  order**: a value declared in one place and effective in another. The duplicate is gone and
  the original is bumped to 2.
- **`check_dead_settings.jl`'s `FILES` list named two files that moved**, which would have
  made the gate pass by scanning nothing — the same silent failure it was written to catch.
  It now scans 154 keyword arguments across 4 files, up from 145 across 3.
- **`repo_root.jl` was included by six files via `@__DIR__`.** Moving it would have crashed
  every gate and tool on startup. All six repointed.
- **`transition/plots_and_tables.jl` reads `bundle.sim` unguarded** (lines 178, 395, 942), and
  a bundle that has been through an MCMC checkpoint has never carried `sim` — the checkpoint
  writers do not simulate a panel. So the transition step cannot use `base_fc` until
  `smm_main` runs on it again. That was true before this release too; it was just invisible,
  because an absent field and a field holding `nothing` fail in different places and neither
  said why. The seal now prints a NOTE when it happens.
- **The two `smm_estimates_*.csv` and `mcmc_results_*.csv` writers were not harmonised.**
  Both now sit in `output/estimates/` beside the bundle, but they have different columns, so
  a reader still has to know which it opened. Merging them into one CSV per window is the
  next step and belongs with the exhibit list, not with a path move.

### Migration

`base_fc` relocated and **verified**: re-read from the new path, re-solved, and
`Q = 497.786594` reproduces the stored value exactly. `base_covid` relocated as
`:superseded` (stored `Q = Inf`, 25 free parameters — a superseded specification, and
nothing reads it, because a crisis window warm-starts from its *paired baseline*, never from
its own file). `crisis_fc` and `crisis_covid` **cannot be deserialized** — pre-struct-change
files — and were left untouched with nothing written.

Sources were left in place: a relocation that cannot prove the point survived is not a
relocation, and removing the source is a separate decision taken after reading the
verification table. `output/smm/` therefore still holds the originals plus the accumulated
`_backup_Q*`, `.pre_migration`, `.pre_v1910` and `.new` debris, pending the cleanup pass.

Migration staging moved to `/.migration/` at the repo root, outside `output/`, and is
removed on success — a migration that succeeds leaves nothing, one that fails leaves
everything, which is the right asymmetry. A `.new` sibling beside the original is exactly
the trace a migration should not leave.

---

## v19.5.0 — 2026-08-28

**MINOR.** One live bug fixed, two settings retired, one new release gate. The criterion is
untouched, so stored estimates stay comparable and every bundle still loads.

### Fixed

- **SA reheat checkpoints were never written.** `_sa_loop` reads `checkpoint_path` and calls
  `write_checkpoint` at each reheat, but `_run_sa` declared the keyword and dropped it — so
  the SA half of "every reheat saves automatically" has never worked, while the DE half has.
  Now forwarded on the single-chain and final-chain paths. The parallel warm-up chains
  deliberately pass `""`: they are short runs from dispersed starts, so their best point is
  routinely worse than the incumbent, and writing it would demote the warm start.
- **The MCMC footer's three false statements.** The abort reason was one flag covering two
  paths that call for opposite fixes (`:drift` → re-seed, `:acceptance` → rescale the
  proposal), so `run_demc` now returns `abort_why` and the footer reads it through
  `hasproperty` so pre-19.5.0 results still print. The target line was hardcoded to one
  prior convention while the same footer's drift line correctly reported the other. And
  "No chain ran" printed after a chain had run 1,250 generations and kept 40,000 draws —
  now distinguished from `MCMC_JAC_ONLY`, with the remedy differing by branch.
- **`MCMC_MOVES_MIN` was unreachable.** It was 5,000, justified in its own comment as
  "217·d, near the top" of the 10d–100d band — but 100·d is 2,300, so it sat 2.2× above
  the band, and both base_fc runs plateau near 1,000 (43·d, inside it). Now 2,300.

### Retired

`check_dead_settings.jl` (new) parses the AST rather than grepping, and found three keyword
arguments declared and never read. One was the checkpoint bug above; the other two are
genuinely obsolete and are now gone from every signature, call site, and the settings
banner. Both survive as `SMMRunParams` fields **only** so the 12 stored bundles keep
loading — removing a field shifts field types at the reader and breaks them all. Delete at
the next struct change that migrates bundles anyway.

- **`sa_target_fin`** (`= 0.90`). Intended to shrink the step when the feasible fraction fell
  below it. `_sa_loop` never read it, and the settings banner printed it as though it
  governed the walk. What we tried and why it is not needed: the mechanism it would have
  counteracted — Corana reading an infeasible proposal as a too-large step, since
  `n_prop[j]` increments before the solve — was measured on a perforated surrogate and is
  not there. At realised feasibility 0.88 the walk closes 94.4% of the gap against 97.2% at
  full feasibility, and the adapted step is unchanged (0.171 vs 0.176 of its ceiling).
- **`de_patience`** (`= 20/25`). Superseded by the `de_reheat_flat` + `de_reheat_rate` stall
  test; `_run_de` declared `patience` and never read it.

### Consequences elsewhere

- **`check_forwarding.jl` cannot catch this class.** It asks whether the caller's value
  reaches the callee and passes as soon as the keyword is present at the call site. All three
  findings above were forwarding-clean. `check_dead_settings.jl` is now gate 8.
- **The SA temperature is calibrated against the wrong scale, and is not yet fixed.**
  `T0 = -sa_t0_rel·|Q| / log(sa_t0_accept)` with the shipped `sa_t0_rel = 1e-4` gives
  T0 = 0.0604 at Q = 726.7, cooling to 9.4e-4 by iteration 30,000. The probability of
  accepting a one-plateau uphill move (ΔQ ≈ 3.32) is then 1.3e-24, falling to exactly zero;
  the Metropolis walk that descended 221 units runs at T_eff = 2 and accepts it with
  probability 0.19. Measured on barriered surrogates, shipped-T SA closes 0% as soon as
  barriers exist while T0 = 2.758 closes 57–90%. **The fix is to calibrate against the
  criterion's resolution rather than its level** — `T0 = -plateau / log(t0_accept)` — for the
  same reason `_WIDTH_DQ` is absolute rather than relative: Q's level is dominated by
  moments the model structurally cannot fit, so a fraction of it tracks the misfit floor.
  It is not applied yet because the plateau has not been measured at the current point.
- **T0 and the reheat rule must change together.** Reheat multiplies `T_reheat` cumulatively
  with no ceiling, so raising T0 to 2.758 and leaving reheats on takes T to 88 and closes
  **0% everywhere** — worse than today. Under the current cold T0 reheats are load-bearing
  (31.7% vs 0.0% at B = 0.5), because they are the only thing warming the walk toward useful
  territory, capped at 0.060 × 2⁴ = 0.97.

### Rejected

- **`sa_halflife = 5000` as the cooling schedule.** Measured at T0 = 2.758: halflife = the
  full budget closes 90.0% at B = 0.5 against 70.0% at 5,000, and 56.7% vs 21.7% at B = 1.0.
- **The logarithmic branch, at either exponent.** `cooling_exp = 1.0` is Hajek's form, but
  this parameterisation normalises to `T(1) = T0` and then falls to 15% of T0 by iteration
  100 — it closes 10.0% at B = 0.5 and 0.0% at B = 1.0, and `cooling_exp = 2.0` closes 0.0%
  at both. Kept in the code as a branch; not a candidate for the default.
- **Reheating above T0.** Capping a reheat at T0 (un-cool rather than overheat) closes
  66.7% at B = 1.0 against 56.7% with no reheat, but 66.7% vs 90.0% at B = 0.5; capping at
  2·T0 is worse everywhere. No reheat is the better default until the real barrier height is
  measured.

**Not established:** the barrier height of the actual criterion. The surrogate reproduces the
trap qualitatively but its B = 0.5 gives an SA/walk ratio of 4.5 against the observed 1.44,
so every percentage above is a ranking, not a forecast. Q is also discrete in 3.32 steps, so
differences below ~7% are not resolvable at five seeds.

---

## v19.4.0 — 2026-08-27

**MINOR.** Two MCMC changes, both driven by one measurement: `MCMC_PRIOR` selects the
prior convention, and the proposal's crossover probability moves from 0.25 to 0.95. The
progress line now reports what a tuning decision needs rather than acceptance alone. The
criterion is untouched — `Q(θ)` is identical and no stored estimate moves.

**Comparability.** `Q(θ)` unchanged, so point estimates and objective values remain
comparable across versions. **But the sampled DENSITY changes** under
`MCMC_PRIOR = :flat_t`: a table of chain quantiles produced under one convention is not
comparable with one produced under the other, and a table must say which it used. Measured
on `base_fc`: the `b_S` marginal has mean 9.20e-04 / sd 5.58e-04 under `:flat_theta`
against mean 2.21e-06 / sd 4.50e-05 under `:flat_t`.

### `MCMC_PRIOR` — the prior convention is now a setting

`MCMC_main.jl:logposterior`. `:flat_t` (the new default) reproduces LMR's target,
`logπ = −Q/2`. `:flat_theta` is the previous behaviour, `logπ = −Q/2 + logjac_box(t)`, a
flat prior on θ expressed in the unconstrained coordinate. Neither is privileged by
Chernozhukov-Hong Assumption 4; both are bounded and continuous on Θ. Read as a switch
rather than a deletion so both remain runnable without a source edit, and so old bundles
stay interpretable.

Four sites depend on the convention and all four are convention-aware: the target,
`checkpoint_best` (which inverts `logπ → Q` and would have been off by 46 units), the drift
split, and the `_Q_from_lp` helper that keeps the inversion beside the definition.
`demc.jl` never references the Jacobian and is agnostic.

**Verified:** `logπ(θ̂) = −363.3530` against `−Q/2 = −363.3530` exactly under `:flat_t`,
and `‖dlogπ/dt‖` at the seed goes 2.8821 → 0 by construction, since at an argmin of `Q`
the whole gradient was the Jacobian term.

### `MCMC_CR` 0.25 → 0.95

`MCMC_main.jl:MCMC_CR`. A candidate-by-candidate trace — 23,424 proposals on `base_fc`,
every one logged with its parent pair, realised mask, per-coordinate step, and outcome —
resolved the two effects the old comment described as pulling against each other.

Writing `κ = −Δlogπ / (½‖step‖²)` with the step in the target's own conditional scales,
so `κ = 1` means the step costs exactly what its length predicts:

| coordinates moved | median `κ` | step alignment | acceptance |
|---|---|---|---|
| 1 | 1.307 | 0.125 | 0.260 |
| 4 | 1.171 | 0.513 | 0.062 |
| 8 | 0.938 | 0.601 | 0.047 |
| 11 | 0.781 | 0.686 | 0.024 |

Moving more coordinates is **cheaper per unit length**, because the step then points along
the ridge the population has learned rather than into a coordinate subspace. The gap
survives stratification on step length (ratio 1.19–1.62 within all four quartiles) and is
monotone on ranks (Spearman `ρ(nmask, −κ) = +0.094`, `ρ(alignment, −κ) = +0.175`; Pearson
is near zero only because `κ` is heavy-tailed, sd 7.0 against a median of 1.11). The
jump-floor effect that argued for a low `CR` is real but smaller than the rotation it
trades against.

Not 1.0: that removes the mask entirely and with it the single-coordinate move, whose
acceptance the trace measures at 0.260 against 0.024 at eleven coordinates.

### The progress line reports proposal economics

`demc.jl`. Two windowed statistics added and two low-information fields dropped.

- **`dlp`** — the median feasible proposal's `Δlogπ`. This is the step-scale diagnostic: a
  well-scaled *d*-dimensional proposal sits at −1 to −3, which is what yields acceptance
  ≈ 0.234. The trace measured **−20**, meaning the median proposal is rejected with
  probability 1 − 2e−9 and the entire acceptance rate rides a thin tail — 70% of every
  generation's solves were spent on proposals with no chance. **None of that is visible in
  the acceptance number**, which is why acceptance alone kept misleading the tuning.
- **`esjd`** — expected squared jump distance per proposal, in population-sd units.
  Acceptance is not the efficiency criterion: a shorter step that raises acceptance while
  lowering ESJD is moving less, not mixing better.
- **dropped `fin`** — printed only when it falls below 0.95. At the measured 0.98 it was
  noise on every line, and a real feasibility problem announces itself.
- **dropped cumulative acceptance** — it averages in the collapsed `:at_seed` start
  forever, so it falls monotonically whatever the chain is doing.

The check line now prints only the quantities the stop **gates** on, each beside its
threshold. `R̂` and `ESS` are reported on every generation line and are not gates, so they
are no longer repeated there.

`MCMC_PRINT_EVERY` is env-readable (`env_setting`) and registered; it had been a hardcoded
constant, so a smoke test could not see more than one progress line without a source edit.

### What was tried and rejected

Each of these was proposed as a fix for the acceptance collapse and killed by measurement.
Recorded so they are not re-proposed.

- **Shrinking γ, by any factor.** Rescaling every observed proposal's length by `f`, keeping
  its own measured cost multiplier and direction: ESJD is 1.011 at `f = 1.00`, 0.776 at
  `f = 0.35`, 0.413 at `f = 0.12` — **monotone decreasing in shrinkage**. The shipped scale
  is already ESJD-optimal among pure rescalings. Shrinking buys acceptance 0.055 → 0.24
  while cutting mixing 23%. This is the second time this project has measured that trade
  (the earlier γ × 0.4 arm: acceptance +65%, ESJD −83%).
- **Diagonal preconditioning of the proposal.** A no-op, provably. DE-MC is *exactly*
  equivariant under a diagonal rescaling: with `b_add = 0`, the native and preconditioned
  proposals agree to 2e−16 on the same RNG stream, because the step is built from the
  population, which rescales with it. The only non-equivariant term is `b_add·randn(d)`.
- **Shaping `b_add` per coordinate.** Tested with `b_add` proportional to the measured
  per-coordinate scale, geometric mean matched: acceptance 0.186 against 0.173 isotropic at
  generation 400 — inside seed noise. DE-MC learns a diagonal metric on its own.
- **Raising `MCMC_DELTA`.** δ = 2 gives ESJD 0.93×, δ = 3 gives 0.95× — both inside the 3.3%
  seed noise, and δ = 3 + CR = 0.90 is indistinguishable from CR = 0.90 alone.
- **The ΔQ = 1 width file as a step-scale yardstick.** `output/smm/sa_proposal_widths_base_fc.csv`
  was measured at the pre-R1/R3 optimum (Q ≈ 842) and disagrees with the current criterion's
  geometry by up to 328× across coordinates. On it the population looked 92× anisotropic; on
  the metric measured from the chain's own candidates the span is 8×, and that metric predicts
  rejection far better (acceptance ratio across step quintiles 15.4× against 2.5×). **Do not
  use the stale file for tuning** — regress `Δlogπ` on `step_k²` and read the scale off the
  chain instead.

### Not addressed

**ter Braak snooker updates.** The remaining candidate, and untested on this geometry. A
snooker proposal steps along the line through another chain, a ridge direction by
construction. A Gaussian surrogate gives acceptance 0.181 → 0.334 and ESJD 1.09–1.16× at
10–20% snooker — but that surrogate reproduces only about a third of the real pathology
(median `Δlogπ` −4.3 against the measured −20.0) and exaggerated the masking-direction
penalty by five orders of magnitude relative to the trace. So it is directional evidence
only. Implement only if `CR = 0.95` leaves median `dlp` near −20; if `dlp` lands in the
−3 to −5 range the direction fix was sufficient.

---

## v19.3.0 — 2026-08-27

**MINOR, documented retroactively.** This entry was written a day late, alongside v19.4.0 —
the bump shipped without it, which is a violation of gate 7 of the versioning rules and is
recorded rather than quietly backfilled.

The sequential stop was respecified and the screen-radius settings promoted to live.
`Q(θ)` untouched; comparability unaffected.

- **`stop_rule` replaces the `R̂`/`ESS` gate** (`mcmc_diagnostics.jl`). The old gate fired at
  **0 of 16 checkpoints** on the 18.4.1 `base_fc` chain — false-stop rate and true-stop rate
  both zero, so every run paid its full budget while `R̂` trended *upward* as the budget grew.
  The replacement gates on accepted-move count, on the running maximum of `logπ` having
  stopped climbing, and on `R̂` not having worsened, each at two consecutive checks, plus an
  abort-and-diagnose when windowed acceptance falls below a floor. `MCMC_RHAT_MAX` and
  `MCMC_ESS_MIN` are now **reported rather than gated on**.
- **`MCMC_SCREEN_FRAC`, `_CAP`, `_FLOOR` promoted from `:inert` to `:live`** in the registry.
  They had been marked inert only because the shipped `MCMC_INIT` made them unreachable.
- **`MCMC_CHECK_EVERY` registered.** It had been read via `env_setting` with no registry row
  at all — caught by `check_forwarding.jl` when the stop rule was added, not by the change
  that introduced it. The audit works.
- **Check-window counters separated from print-window counters** (`demc.jl`). The two strides
  are independent, so a print falling between two checks would otherwise zero the wrong
  window and understate the move count.

**Rejected in this version:** switching `MCMC_INIT` to `:screen`. It was made the default and
then reverted within the same day — the argument for it was written before any measurement and
overstated the case, and LMR run this same sampler with every chain started at the seed
(`mpi_mcmc_mod.f90:268`). `:screen` remains implemented and selectable.

---

## v19.2.0 — 2026-08-26

**MINOR.** Two changes to the MCMC path: the chain bundle is now self-contained, and the
convergence gate can terminate. The criterion is untouched — no estimate moves.

**Comparability: unaffected.** `Q(θ)` is unchanged. Chain bundles written before this
version remain readable; they simply lack the `spec` field described below.

### The chain bundle now carries its spec

`MCMC_main.jl:735`. The serialise call wrote `theta_mean`, `se_chain`, `theta_best`,
`params_best`, `Q_best`, `G`, `sigma_hat`, `free`, `labels`, `lb`, `ub` — and **no `spec`**.
Confirmed by deserialising the v18.4.1-era chain bundle under current code: `HAS_SPEC =
false`, no `:spec` and no `:result` key.

**This broke the chain's stated purpose.** `smm_objective(θ, spec)` needs the whole spec
object — fixed parameters, moment targets, sim settings, grids, `W` — and
`free`/`labels`/`lb`/`ub` are not a substitute for any of it. So `theta_best`, the best point
the chain visited, could not be re-evaluated, re-solved for moments, or warm-started from. The
chain reaches points the optimiser cannot, and a point that cannot be re-evaluated is not a
result. Written for every chain bundle, independent of `MCMC_CHECKPOINT` — that flag governs
whether the *estimation's* bundle is overwritten, a separate decision from whether this bundle
is self-contained.

### The convergence gate is per-coordinate, with automatic exemptions

`mcmc_diagnostics.jl` — new `exempt_coordinates`, and `converged_sequential` rewritten.

`converged_sequential` gated on **`minimum(ess)` across all coordinates**, and
`demc.jl:402` is a live early exit (`if done; break`). With one coordinate frozen and `b_S`
railed at its bound, that minimum **could not reach the threshold at any generation count**:
the sequential stop never fired and every run paid its full `MCMC_GENS` budget regardless of
whether the reported numbers had converged. At the measured 0.258 s/solve that is
**18.3 hours** for `MCMC_N = 64 × MCMC_GENS = 4000` — 0.99× the cost of a full SMM run.

Two exemption tests, one per failure mechanism:

- **FROZEN** — a coordinate the proposal never moves has no distribution to mix, and shows
  single-digit `n_distinct` where a live coordinate shows hundreds. This is the
  frozen-coordinate test the diagnosis discipline prescribes and it **was not previously
  computed anywhere in the repo**.
- **AT A BOUND** — measured in **constrained** units, as the fraction of draws within 1% of
  the box width of either edge. It must be constrained: the logit transform puts each bound at
  infinity in `t`, where no threshold detects pile-up.

**Exemption is not a freeze, and the distinction is load-bearing.** An exempt coordinate is
still sampled, still in the chain, still in every output; exemption governs only whether it can
*block termination*, and it is recomputed from the draws at every check, so a coordinate that
starts moving stops being exempt. An automatic freeze would instead convert a loud diagnostic
into a clean table with a spurious zero standard error — which is how a solver defect gets
hidden rather than found, and is the argument on which auto-freezing was rejected earlier.
**All coordinates exempt returns `false`, not converged**: a chain sampling nothing must not
read as success.

`run_demc` gained `lb`/`ub` kwargs (`demc.jl:123`) to carry the bounds to the gate. Both are
optional — omitted, the gate still applies the FROZEN test, so a caller that cannot supply them
degrades to a weaker exemption rather than to a wrong one.

### `MCMC_ESS_MIN` raised 250 → 450, and made a registered setting

**Because what is reported changed, not because 250 was miscalculated.** 250 was sized for a
*standard error*: at ESS 250 an se has relative MC error 4.5%, two stable significant figures,
and that reasoning is still correct for an se. But with **diagonal `W`** the chain's se is not
a valid confidence half-width — Chernozhukov-Hong Thm 3 requires `W = Ω⁻¹`, and
`mcmc_diagnostics.jl:13-15` states outright that diagonal `W` fails it — so the deliverable is
the **quantile pair**, and a quantile is dearer than an sd.

From `MCSE(q_p) = √(p(1−p))/f(F⁻¹(p)) · sd/√ESS`, whose constant is 2.113 at `p` = 0.05:
**ESS 450 is exactly `MCSE(q05) ≤ 0.0996·sd`** — each reported interval endpoint precise to a
tenth of the width it reports. Verified by independent recomputation of both constants (2.1132
and 2.6713).

**Report q05/q95, not q025/q975.** Required ESS scales as the **square** of the constant, so
the tighter tail costs `(2.671/2.113)² − 1 =` **59.8%** more ESS — **714 against 447** at
MCSE ≤ 0.10·sd. *Two revisions of this comment stated 37%, which is neither the squared ratio
nor the unsquared 26.4%; it understated the cost of the very design choice it was cited to
justify.* With 28 moments and 23 free parameters the 2.5% tail is also the least trustworthy
part of the estimate, so the cheaper pair is the better report.

The raise is affordable **only** because the gate is now per-coordinate. Under
`minimum(ess)` it would have made an already-unsatisfiable test more unsatisfiable.

### Measured, not asserted

Four verification gates, all passing:

| gate | result |
|---|---|
| parse, 3 touched files | 0 bad nodes |
| load in driver order | `exempt_coordinates`, `converged_sequential`, `run_demc` resolve; `run_demc` accepts `lb`/`ub` |
| exemption on a synthetic chain (healthy / frozen / railed) | `n_distinct = [1600, 1, 1600]`, `edge_frac = [0.000, 0.000, 1.000]`, `exempt = [false, true, true]` — as designed |
| gate not held hostage; all-exempt refused | 2 of 3 exempt → gates on the live coordinate (`min_ess = 1600`); all exempt → `done = false` |

The railed coordinate is the case that matters: `n_distinct = 1600` (it moves freely in `t`)
but `edge_frac = 1.000` in constrained units. **A `t`-space test would have missed it
entirely** — which is why the exemption is measured after the transform.

### On the SMM stopping rule — an earlier claim of mine, retracted

**The improving fraction does NOT collapse near the optimum.** I reported 0.72% from a single
sweep as strong evidence of local optimality. Across the 17 sweeps of the completed `base_fc`
run: pooled **675/22,080 = 3.06%** [2.84%, 3.29%] over sweeps 2–17 — **4.2× higher** — and
regression on `Q` gives slope +0.00058, **p = 0.57**: statistically **flat** as `Q` descends
739 → 727. The 0.72% sweep also had a median width of 0.00703, **twice** every other sweep's,
so it was not the same instrument. An improving-fraction stop at `f` = 0.02 would **not** have
fired on this run.

**What survives** is the per-reheat decay: slope −0.028/reheat (p = 0.018), last-10 mean gain
0.195 against first-7 mean 0.512, total DE descent 12.79 `Q`. Individual late gains are ~10×
below the ~2 `Q` at which differences clear the grids' disagreement; the aggregate is not.

**On the reheat cap** (user's preference: keep at 20 or fewer, possibly 10). The finished run
hit `de_max_reheats = 20` exactly, so the cap **already is** the binding stopping rule. From
that run's own ledger:

| cap | stops at gen | `Q` reached | gives up | cost |
|---|---|---|---|---|
| 10 | 582 | 732.2358 | **5.53** | 10.0 h |
| 12 | 645 | 731.2060 | 4.50 | 11.1 h |
| 15 | 791 | 729.6065 | 2.90 | 13.6 h |
| 20 | 1011 | **726.7060** | — | 17.4 h |

**Left at 20, and the reason is that 5.53 `Q` is above the ~2 resolution threshold, not
below it** — a cap of 10 would have given up a *resolvable* amount of fit for 7.4 hours. Even
15 gives up 2.90, still above. (An intermediate cell of mine printed "0.4× BELOW" by dividing
the wrong way; corrected in the same turn.) The honest reading is that **this run had not
exhausted its local improvements when it stopped**, so tightening the cap is a real trade
rather than free.

### Known gaps

- **The diagonal-`W` interval error has indeterminate sign**, bounded to roughly
  [0, 3.9]× the curvature se. Determining it needs `Ω`'s off-diagonal block over the 28 active
  moments on the sampling-variance footing, which is not obtainable from `sigma_{w}.csv`
  (rank-deficient, different footing). **Disclose the caveat; do not guess a direction.**
- **`n_distinct_min = 50` and `edge_frac_max = 0.05` are not tuned**, only reasoned:
  on the old chain, `n_distinct < 50` caught `δ_S` (4) and nothing else (median coordinate
  198), and `edge_frac > 0.05` caught `b_S` (0.657). Both figures belong to a chain seeded near
  `Q` = 842 and have not been re-measured at the current point.
- **`MCMC_RHAT_MAX = 1.10`** was not revisited. The `converged_sequential` default is 1.03; the
  caller overrides to 1.10. Not examined this pass.
- **`b_S` reporting.** No interval is valid at a bound. Whether a zero outside option is an
  acceptable economic corner or a specification problem is a modelling question the numerics
  cannot settle.
- **The 0.72% retraction has a loose end:** the improving fraction is flat at ~3%, which is
  *not* the signature of an exhausted local optimum. The local-optimality evidence now rests
  on the reheat-ladder failures and the per-reheat decay, not on the sweep. Whether ~3%
  improving draws at gains below grid resolution constitutes "at the optimum" is **not
  established**.

---

## v19.1.0 — 2026-08-25

**MINOR.** The SA proposal was respecified and the settings layer rebuilt. A run does not
reproduce v19.0.0's optimiser path, but the *criterion* is unchanged — the same θ scores the
same Q — so stored estimates remain comparable as fit statistics.

**Comparability: point estimates are NOT superseded; optimiser traces are.** `Q(θ)` is
identical to v19.0.0 at any θ. What changed is which θ the optimiser reaches and how fast.

### What changed

**The SA proposal is now DE's shape** (`smm/smm.jl:~830`). An independent Bernoulli(`p_move`)
draw per coordinate plus one forced index, so the moved count is Binomial conditioned ≥1 —
random, unbounded above, controlled mean, **no hardcoded count anywhere**. Each moved
coordinate steps by its own adapted scale, seeded from the measured `ΔQ=1` half-widths, with
Corana adaptation. `p_move = SA_SCALE_P_MOVE/d`, dimensionless in `d`, default giving a mean of
about 2 of 23 moving. The realised moved-count **histogram** is printed in every trace line
(`_moved_field`, `smm.jl:649`) rather than the nominal `p` — a nominal setting cannot reveal a
proposal that is not doing what it claims.

**`p_move` is pinned, not adapted, and that is a derived result rather than a preference.** To
second order the Metropolis exponent has mean `−(s²/2T)·Σh_jj` and variance `(s²/T²)·Σg_j²`
over the moved set, both linear in the count — so **acceptance is a function of `k·s²` alone**,
and ESJD is `k·s²` by construction. Verified numerically: at `k·s²` held constant, acceptance
is 0.6827 at `k` = 1, 3, 8 and 23, identical to four digits. Corana already adapts the step
against acceptance; adapting `p` there too is two knobs on one equation. The proposed escape
was feasibility, which is **saturated** here (0.958–1.000 across `k` = 1…23 at two matched
displacement scales, 768 solves) and therefore cannot source `p` either.

**Settings layer rebuilt** — `smm/settings.jl` is new: a single `env_setting` reader with a
`SETTINGS_REGISTRY`, replacing `_env_*` helpers that were **duplicated verbatim** in both entry
points, which is why nothing could audit them. 46 call sites migrated. `print_env_settings()`
reports what each key actually resolved to.

**`print_spec` can no longer lie** (`smm_params.jl:1058`). It took its settings as its own
kwargs, which is the mechanism that let the header assert `subset_k = 3` on runs where
`run_smm` received `0`. It now reads `spec.run` and **cannot be handed a value**.

**Release gate extended** (`scripts/check_forwarding.jl`, 103 → ~370 lines, wired into
`scripts/check_repo.sh`). Four independent checks: forwarding across `run_smm`/`_run_sa`/
`_run_de`/`run_demc` with prefix-aware pairing; registry parity; **name parity**; and legacy
keys. Renamed env keys now **error at startup** instead of being silently ignored.

**A live `MethodError` was fixed** (`smm.jl:997`): `_run_sa` referenced a `scale_cap` argument
deleted when the bisection cap was consolidated into `_WIDTH_CAP`, so `:sa_de` died on entering
stage 1. **Found by running, not by reading** — it survived every parse and load check.

### Measured, not asserted

Head-to-head at 250 matched solves per arm, 3 seeds, at the stored `base_fc` optimum:

| arm | mean ΔQ | acceptance | distance moved | moved/iter |
|---|---|---|---|---|
| scalar shared step (deleted) | 55.44 | 0.016 | 22.10 | 23 exactly |
| fixed `k`=3 (retired) | 81.22 | 0.173 | 21.06 | 3 exactly |
| dense mask, `p·d`=12 | 87.80 | 0.156 | 9.89 | 12.4 [5-19] |
| **mask, `p·d`=1 (shipped)** | **90.27** | 0.201 | 12.38 | 1.95 [1-6] |
| single coordinate | **91.45** | 0.335 | 13.36 | 1 exactly |

The shipped arm beats the retired fixed-`k` by **+9.1** and the deleted scalar path by
**+34.8**, and beats the dense mask on descent **and** on distance moved — so it is not the
acceptance artefact a shorter proposal produces.

**NEGATIVE RESULT, recorded because it constrains future work:** a strictly single-coordinate
move edges the shipped mask by **1.2 ΔQ** (91.45 vs 90.27) at overlapping seed spreads (sd 2.79
and 5.18). **The mask is not established as better than moving one coordinate.** It ships for
the tail capability and because no fixed count may be hardcoded — not on a measured win. Xu,
Wang & Deng (arXiv:2504.17949) reach the same `d = 1` conclusion on smooth benchmarks; that
finding is not refuted here.

**Two claims tested and killed:**
- *A matched-displacement one-shot scan ranked `k`≈11 best with `k`=1 3.2–3.4 se below —
  and chains at equal budget REVERSED it.* Mechanism: a scan measures one-shot gain from the
  incumbent; a chain compounds accepted moves and lets Corana retune. **A large joint move is
  the better single bet and the worse thing to repeat.** The scan is demoted to a diagnostic.
- *The codebase's own justification for joint moves — "coordinates that are individually
  infeasible can be jointly feasible" (`smm.jl:1880`) — did NOT replicate:* **0 of 90 draws**
  had a feasible joint move with all components infeasible, 390 solves. The argument that the
  optimum must exceed `k` = 1 is dead.

### Settings

- **Removed:** `sa_subset_k` — env key, forwarding, **and** the `SMMRunParams` field, with
  bundles migrated in the same pass. It was three switches under one name (see below).
- **Deleted, not switched off:** `_sample_subset!`, the `subset_k==0` isotropic scalar branch,
  its scalar step adaptation (the 0.01–2.0 clamp, ~3× the median measured half-width),
  `_step_field`'s scalar path.
- **Added:** `SA_SCALE_P_MOVE`, `SA_SCALE_PER_K`, `SA_SCALE_SIGMA`, registered and forwarded
  with caller and callee names identical.
- **Renamed:** `sa_cooling_halflife` → `sa_halflife` at the callee, closing the name-matching
  blind spot the previous check documented as invisible to it.

### Consequences elsewhere — what this version may have made obsolete

| candidate | why this version threatens it | traced status |
|---|---|---|
| `sa_max_iter = 30000`, `reheat_patience = 400` | Calibrated for a proposal that could not resolve small improvements. Under the new proposal, 92.8% of SA's descent arrived in the first 100 iterations and the final 1902 bought **exactly 0.0000** — the budget shape may be wrong by an order of magnitude. | **LIVE.** Under review. |
| `sa_rate_tol = 0.05`, `de_avg_tol` | Absolute-`Q` improvement budgets, set against a noise floor of 0.59 **measured on the pre-v19.0.0 objective and never re-measured**. | **LIVE at old values.** Re-measure the floor first. |
| `SA_SCALE_PER_K` (the sparsity scan) | Its own recommendation was reversed by chains, so it no longer informs `p_move`. Default 0. | **KEPT as a diagnostic** — it is what measures the feasibility saturation. Justified, not obsolete. |
| The 0.59 "noise floor" itself | **RETIRED — it never existed.** Measured 2026-08-26 at the live checkpoint: four repeats at identical θ give sd = **1.90e-12**, range 3.98e-12 (~12 ulp, threaded-reduction non-associativity). The objective is **deterministic**; there is no stochastic noise floor, and 0.59 was never a Monte-Carlo measurement. Every claim in this file and in prior sessions that cited "the measured noise floor of 0.59" was inherited and unverified. | **CORRECTED.** Use the grid floor below instead. |
| The 0.061% grid CV | Measured at the *pristine* point. At **P_DE** — the point the optimiser actually reaches (`Q` = 732.2358) — `Q` over `Np_S ∈ {100…200}` gives 730.82, 732.24, 736.11, 736.12, 737.46, 737.42: **CV 0.382%, half-range 3.32 `Q` units** — a factor **6.3 larger**. Whether the descent walked into a worse-resolved region or grid error is simply non-uniform is **NOT separated** by this measurement. | **CORRECTED, cause unresolved.** |

**THE SCALE FACT THAT GOVERNS EVERY THRESHOLD, and it is not the one assumed.** `Q`'s
**level** is resolved only to ±3.32 at P_DE, while paired **differences** clear the grids'
disagreement once `|dQ| ≳ 2` (7 of 7 tested). So a rule reading differences **at the
generator's displacement** (the DE selection rule, the improving-fraction sweep) reads a real
quantity and can carry a tight threshold; a rule reading differences in the **0.08–1.6 band**,
or leaning on `Q`'s level, cannot — **at any threshold**. The old rule is mis-scaled not
because `Q`'s level moved from ~610 to ~842, but because it watches a band the discretisation
does not resolve. The fix is to change *which quantity* the rule reads, not to renumber it.

**And P_DE is not a stationary point.** Single-coordinate `dQ` is **linear in step across four
decades** (exactly ×100 per ×100 of step, both for the narrowest and the median-width
coordinate) — a first-order response. It is a point the optimiser *stopped moving at*, not one
where the gradient vanishes. This does not contradict the local-optimality evidence (the
improving fraction collapses from 179/384 = 0.466 at the pre-SA point to 10/1380 = 0.00725 at
the post-SA point — a factor of **64.3**, Fisher exact **p = 2.0e-121** — and full-width steps
improve *less* often, 127/384 = 0.331, than third-width ones); it sharpens it to: the criterion
has slope there, but no improvement that the machinery can both reach and resolve.

**A tenth finding, from reading a live trace rather than the source — `sa_subset_k` was three
switches wearing one name** (`smm.jl:559-565`, `:616`). Setting it above zero simultaneously
(a) fixed the proposal dimension at exactly `k` coordinates, (b) switched the proposal from one
shared scalar step to the per-coordinate vector, and (c) gated the Corana adaptation, which ran
only on that branch. **A per-coordinate scale was unreachable without also accepting a fixed
count.** That coupling is why forwarding this setting in v19.0.0 re-enabled a restriction
deliberately removed from DE.

*Naming note:* five unrelated `k` concepts exist in `smm/` — `sa_subset_k` (SA proposal
dimension), the generator's loop `k` (sweeps 1…npar building the DE population, `:1082`),
`gen_per_k`, `de_gen_per_k`, and `k_star`. Only the second is about initialisation. Confusing
them has already happened once.

### Known gaps

- **`smm_result_base_fc_diagonalW.jls.pre_v1910` no longer deserializes.** It fails with
  `TypeError: in new, expected Float64, got a value of type Int64` — the *loud* variant of the
  struct-shift failure, because `sa_subset_k` was an `Int` between two `Float64` fields, so
  removing it shifts a `Float64` into the `Int` slot and the reader raises rather than silently
  returning wrong numbers. **That backup cannot serve its purpose**: rolling back requires
  checking out the pre-v19.1.0 struct definition first. Not a data loss (the migrated bundle is
  installed and verified) but the rollback path is not what it appears to be.
- **The DE checkpoint write is gated on `show_gens`** (`smm.jl:1920`), a *verbosity* flag. A
  quiet run writes no DE checkpoints at all. Same defect class as the settings-forwarding bug:
  a flag controlling something outside its stated purpose.
- **The initial DE population is never checkpointed.** 1380 candidates, ~394 s measured — the
  most expensive single unprotected stretch in the pipeline.
- **`base_covid` migrated but its `Q` check could not discriminate.** It loads with `Q = Inf`
  matching a stored `Inf`; `Inf == Inf` is not a reproduction test. It rests on an exact θ
  round-trip (`max|Δθ| = 0.000e+00`) alone. It is also a **25-parameter** point from a
  superseded spec and needs re-estimation regardless.
- **`crisis_fc` and `crisis_covid` were NOT migrated.** Their θ round-trips wrong by 3.34 and
  4.55 because they encode a pre-`P_U`-normalisation **19-parameter** spec. Re-estimation, not
  migration. `.new` files retained for the record.
- **Optimiser stopping rules were not retuned in this version**, only diagnosed. The run they
  produced reached `Q = 732.24` at `DE reheat 10, gen 582` (checkpoint verified to reproduce at
  `|diff| = 4.5e-13`), and the per-effort decay across both stages is the evidence a retune
  must act on.

---

## v19.0.0 — 2026-08-25

**MAJOR.** The criterion itself moved: `Q(θ̂)` on `base_fc` is **842.05** against
**609.75** under v18.4.x at the *same* θ̂ and the same `Np_S = 120`.

**Comparability: BROKEN. Every stored `base_fc` result is superseded.**
`Q = 609.75` must not be reported as a fit statistic — it was an artefact of one grid.
The chain, `smm_estimates_base_fc_diagonalW.csv`, and every stored standard error for that
window go with it. `θ_best` (487.68) was the chain descending *further* into the bias: it
scores ~1975 corrected, and at baseline `Np_S = 480` it scores 2093.12 against θ̂'s 830.3 —
**the ranking reverses.** Re-estimation is required, not optional.
`base_covid` is the exception: its bias was −0.05%, so its point will barely move.

### What changed

**Solver — the free-entry map is now continuous** (`solver/skilled.jl:348-366, 390, 398,
425`). `compute_Jbar_skilled` selected on a hard `1{E1≥E0}` for the on-the-job-search
margin that `solve_stationary_skilled!` already softened, so free entry and the stationary
distribution priced the same margin at different resolutions and the map had a jump — one
5e-08 step in `θ_S` moved `F` by 2119× the tolerance. A map with a jump has no fixed point,
which is why more iterations, a tighter tolerance and Anderson acceleration all failed.
Now softened at all three sites, unconditionally, no switch.

**Solver — the quality densities use exact cell masses** (`build_cell_mass_density`).
The shock density's singularity sits at `p = δ_S`, a *free parameter*, integrated against
fixed Gauss-Legendre nodes — so the quadrature error moved with the parameter and the
optimiser tuned it. `Np_S = 120` is the global argmin of node-distance to `δ_S` over
`N ∈ [60, 1200]`. Nodes and `wp` are unchanged (`wp` doubles as the Lebesgue `dp` measure,
so a Gauss-Jacobi swap is ill-posed: it integrates `∫p dp` to 0.4731 against 0.5).

**Solver — `θ_S` under-relaxation at w = 0.9** (`solver/params.jl`, as a module-level
`const DAMP_THETA_S = Ref(0.9)`). `θ_S` previously had no damping path at all.
Shipped as a `Ref` and **not** as a `SimParams` field on purpose: Julia's serialiser reads
structs positionally, so a new field would make every bundle on disk unreadable.
w = 0.9 is LMR's own value (`params.f90:275`).

**Transition guard** (`transition/transition_solver.jl:346`) — the one genuine pointwise
consumer of the density vector, which the cell-mass change moves by up to 21.5% (offer)
and 71.0% (shock).

**Blocker fixed: `:sa_de` could not complete** (`smm/smm.jl:1378`). `_de_stage` forwarded
`checkpoint_path` into a `_run_de` that did not accept it, so stage 2 `MethodError`ed.
A **mirror defect** was found in the same pass (`smm/smm.jl:2037`): `run_smm`'s `:de`
branch never forwarded `checkpoint_path` at all, so DE reheats wrote nothing while the run
appeared to succeed. Fixing only the signature would have left `:de` silently
checkpoint-less.

**SA settings were computed and never forwarded** (`smm/smm_main.jl:1117`).
`sa_subset_k = 3`, `sa_halflife = 5000`, `sa_rate_tol = 0.05` and `sa_rate_span = 300` were
computed into `run_params` and then not passed to `run_smm`, which silently substituted its
own defaults `(0, 0, 0.0, 0)`. **Live since v18.4.0**; v16.7.1 forwarded them correctly
(`smm_main.jl:987-990` at `62b59b5`) and the v18.4.0 rewrite that moved the loose `SA_*`
constants into `run_params` dropped the kwargs instead of translating them.
Symptom: SA acceptance 0.04 decaying to 0.00, because `subset_k = 0` disables
per-coordinate Corana adaptation entirely and the scalar fallback clamps its step at 0.01 —
roughly 3× the measured `ΔQ=1` half-width, so every proposal overshot. And
`sa_halflife = 0` selected the logarithmic cooling branch whose own comment warns it
"spends its whole descent in the first hundred iterations".

**The log was asserting a configuration the optimiser was not using.** `print_spec` reads
these as its *own* kwargs from the caller, so it reported `subset_k = 3` on runs where
`run_smm` had received `0`. Its SA line is now labelled **"SA (requested)"**, and a new
**`[SA config]`** line is printed from inside `_run_sa` off the values it actually holds
(`smm/smm.jl:492`). A print sourced from the caller cannot detect a forwarding gap; one
sourced from the consumer cannot miss it.

**New release-gate check** (`scripts/check_forwarding.jl`) — parses `run_smm`'s kwargs
against the caller's computed settings and exits non-zero on a gap.

**Prints trimmed** — the `[stage 0/2]` announcement and the "— find the basin" /
"— refine within the basin" phrases removed. The best-candidate retention they described is
unchanged, just silent.

### Measured, not asserted

| quantity | value |
|---|---|
| `Q(θ̂)` shipped / pristine HEAD | 842.051543 / 609.751327 |
| cell masses sum to 1, `Np_S` = 60 / 120 / 480 | 1.000000000000 at all three |
| feasibility: baseline → R1 → R1+R3 → +damping | 58.0% → 94.3% → 95.7% → **98.3%** |
| paired-direction design behind that | n = 192 per radius, 2304 solves, McNemar p = 3.9e-45 |
| grid CV of `Q` over `Np_S ∈ {100…200}` | 20.33% → **0.061%** (a factor 335) |
| grids returning a finite `Q` | 3 of 6 → **6 of 6** |
| damping's own effect at θ̂ | `\|dQ/Q\|` = 1.2e-10 (max 6.5e-07 over 96 controls) |
| `ΔQ=1` half-width ratio, skilled / unskilled block | median 18.62× / 1.14× |

The last row is the inference consequence: **the skilled-block standard errors were
understated by roughly an order of magnitude and the unskilled-block ones were about
right** — the pattern the mechanism predicts, since both defects lived in the skilled block.

### Settings

- **Forwarded, previously inert:** `sa_subset_k`, `sa_cooling_halflife` (spelled
  `sa_halflife` in the caller — a name mismatch, see the gap below), `sa_rate_tol`,
  `sa_rate_span`.
- **Added:** `DAMP_THETA_S` (module-level `Ref`, not a struct field — see above).
- **Inert but harmless:** `nm_rate_tol`, `nm_rate_span`, `nm_simplex_step` are computed and
  not forwarded, and Nelder-Mead is never invoked under `method = :sa_de`. Reported by the
  new check as findings; documented rather than silently allowlisted.

### Known gaps

- **`transition_panel.jl` could not be loaded** during the gate: `Plots`/`GR_jll` fails to
  precompile in the verification sandbox. It is untouched plotting code; the edited
  `transition_solver.jl` loads and runs.
- **The working tree carried inherited uncommitted changes** predating this version
  (`MCMC_JAC_ONLY` default flip, `de_max_reheats` 50 → 20, `param_symbol` display helpers
  which **add a column to `mcmc_results_{window}.csv`**, `_WIDTH_DQ`, VM launcher edits).
  Kept in a separate `inherited_changes.diff`. They are **not** covered by this version's
  release gate — in particular the `MCMC_JAC_ONLY` flip and the CSV column addition change
  behaviour and an output format.
- **`check_forwarding.jl` matches on NAME**, so a setting the caller spells differently is
  invisible to it. `sa_cooling_halflife`/`sa_halflife` is exactly that case — **one of the
  four bugs this version fixed would not have been caught by the check added to prevent
  it.** Keep caller and callee names identical; treat a clean run as necessary, not
  sufficient.
- **1.7% of nearby points still fail** (10 of 576), and 9 of 10 are *unskilled*-block
  non-convergence with the skilled block converging — a different object from the one R1
  fixed. All 25 pre-damping residuals have **exactly one** crossing of
  `F(θ_S) = θ_S` (zero non-existence, zero multiplicity), slope median −1.051, iterate a
  median 3.7e-04 from the crossing against a 1e-7 tolerance. **Solver failures to fix, not
  economics to document.**
- **`b_S` sits at its lower bound before and after the fix.** The corner survives, so
  whether an outside option of essentially zero is defensible is a modelling question the
  numerics cannot settle.
- **`c` is untouched by both fixes** (width ratio 1.13). Its flatness is caused by neither
  defect; its interval will be wide and grid-robust.
- **`crisis_fc` and `crisis_covid` bundles are truncated** (EOFError) and fail identically
  under pristine code. Predates all of this.

### Tried and rejected — do not re-propose without new evidence

- **Gauss-Jacobi quadrature for the quality grid.** Ill-posed: `sg.wp` does double duty as
  the `dΓ` measure *and* the bare Lebesgue `dp` measure, including the closure
  `1 = û + ∫ê dp`. A Jacobi mass rule integrates `∫p dp` to 0.4731 against 0.5, a −5.38%
  error corrupting the stationary normalisation. Its nodes also move with the free shape
  parameters (mid-node span 0.0668) while the Gauss-Legendre node is pinned, and the cutoff
  logic is indexed on that grid. The ability grid *can* use Gauss-Jacobi because its
  weights are only ever mass weights — the two grids have structurally different jobs.
- **A diagonal preconditioner for the DE-MC proposal.** Structurally void: the proposal is
  built from the population's own difference vector, so it is **equivariant** under
  `θ → Dθ` and a diagonal `D` is a bitwise no-op (verified two ways: 0.0 and 2.2e-16).
  Only the additive `ε` has fixed scale, owning ~1.7e-08 of step energy at `b_add = 1e-4`.
  An apparent 58× acceptance gain was measured at a fixed, unadapted population — a state
  the sampler never occupies — and forcing the effect there collapses ESJD by 535×.
  **The anisotropy is real (8523:1); this remedy does not act on it.**
- **A large finite penalty instead of `Inf`** (LMR's design). A bitwise no-op at
  `:at_seed` — identical population hashes over 150 generations. Worse, on the `:screen`
  path it reports 96/96 feasible against 65/96, because a finite value defeats the screen's
  `findall(isfinite)` gate.
- **Softening the `τ` training frontier.** The linearity justification survived
  (exact to 8.3e-17), but it buys no feasibility (5 gained / 5 lost, p = 1.00), costs an
  8.2% `Q` shift, touches 0.5% of the grid, and **silently disables the SMM degeneracy
  gate** at `smm/smm.jl:198`, which stops firing in exactly the case it exists to catch.
- **Softening the `d` policy, and the `u_frac` clamp.** Both cleared as suspects for the
  residual failures: the `d` indicator flips 6035 times under *both* arms without producing
  a map step, and the `u_frac` clamp flips **zero** times.
- **Auto-freezing an immovable coordinate.** Rejected on the strongest available argument:
  it would have *hidden this release's `δ_S` finding*, converting a loud diagnostic into a
  clean table with a spurious zero standard error. Detect and report; never act.

### Consequences elsewhere — what this version may have made obsolete

**This section is the point of the file.** Git records that `skilled.jl` changed. It cannot
record that changing `skilled.jl` made a constant in `smm_main.jl` meaningless, because
nothing in `smm_main.jl` was edited. Every entry below is a candidate *derived from the
change*, with its current status traced in the code — not a list of edits.

| candidate | why this version threatens it | traced status |
|---|---|---|
| **`Np_S = 120`** (`smm/smm_main.jl:719`) | 120 had no principled basis: it is the **global argmin of node-distance to `δ_S` over N ∈ [60,1200]**, i.e. the grid at which the removed artefact was largest. With the artefact gone, 120 is an arbitrary inherited number. | **STILL HARDCODED.** Re-choose on cost-versus-CV grounds now that CV is 0.06% at every grid. |
| **`use_anderson`, `anderson_m`** (`solver/params.jl:232-233`) | Anderson acceleration cannot converge a discontinuous map — it was carried while the map had a jump. R1 removed the jump, so its value is now an open question rather than a necessity, and acceleration on a contractive map can *hurt*. | **LIVE, UNTESTED POST-R1.** Measure on/off. |
| **`damp_pstar_S`** (`solver/params.jl:237`) | Damped the *cutoffs* as a proxy, because `θ_S` had no damping path — `params.jl:27` says so in as many words. v19.0.0 adds `DAMP_THETA_S` at the real margin. Two damping mechanisms now act on one problem. | **BOTH LIVE.** Possibly redundant; test whether `damp_pstar_S` still earns its place. |
| **`conv_streak`** (`solver/params.jl:230`) | The requirement of 4 *consecutive* sub-tolerance steps was calibrated against a chattering map — it is the criterion a period-16 orbit made unsatisfiable. On a continuous map 4-in-a-row is no longer the binding difficulty. | **LIVE at 4.** Likely over-strict now; cheap to re-tune. |
| **`maxit_outer = 300`, `maxit_global = 50`** (`solver/params.jl:362`) | High caps bought a cycling map more chances to stumble into tolerance. A contractive map converges geometrically or not at all, so large caps now only make failures slow. | **LIVE.** Lower once convergence is measured. |
| **Stall detection** (`skilled.jl:598`, `unskilled.jl:394`, `solver.jl:191`) | Added specifically to detect the limit cycle (commit `4a427ff`, "added inner loop stall detection"). The skilled-side cycle is what R1 removed. | **STILL FIRING** — 9 of the 10 residual failures are *unskilled*-block, so the unskilled detector is load-bearing and the skilled one may now be dead code. Trace which fires. |
| **The scalar shared-step SA path** (`smm/smm.jl:624`, `clamp(step, 0.01, 2.0)`) | Reachable only when `sa_subset_k == 0`, which was true only because of the forwarding bug this version fixed. Its 0.01 floor is ~3× the measured `ΔQ=1` half-width, so it cannot propose a move any coordinate wants. | **REACHABLE BUT NO LONGER TAKEN.** Slated for deletion with the SA proposal rebuild. |
| **Absolute-`Q` thresholds** — `sa_rate_tol = 0.05` (`smm_params.jl:271`), `nm_rate_tol = 0.05` (`smm.jl:1883`), `PROMOTE_MIN_DQ = 1e-4` (`MCMC_main.jl:209`), `de_avg_tol` (`smm_params.jl:154`) | All are improvement budgets in `Q` units, set when `Q ≈ 610` and against a noise floor of **0.59 measured on the old objective**. The level moved 38% and the floor has not been re-measured. `smm.jl:1862` reasons explicitly from "Q is a chi-square, so ΔQ = 1 is…" — a chain whose premise moved. | **ALL LIVE AT OLD VALUES.** Re-measure the noise floor first; the thresholds follow from it. |
| **`ee_step_S` held out permanently** | It was excluded because the memoryless redraw mechanically overshoots it. R1 changes how the ladder prices the OJS margin, which is the mechanism generating that step. | **STILL EXCLUDED.** Worth re-checking whether the overshoot survives; if it does not, a moment returns to the battery. |

**A tenth, found by reading the trace of a live run rather than by inspection —
`sa_subset_k` is three switches wearing one name** (`smm/smm.jl:559-565`, `:616`). Setting
it above zero simultaneously (a) fixes the proposal dimension at exactly `k` coordinates
drawn without replacement by `_sample_subset!`, (b) switches the proposal from one shared
scalar `step` to the per-coordinate `step_vec`, and (c) gates the Corana per-coordinate
adaptation, which runs only on that branch. **So a per-coordinate scale is unreachable
without also accepting a fixed count** — the two are welded together, and `k = 0` versus
`k = 3` differ along three axes at once.

That coupling is why forwarding this setting in v19.0.0 *re-enabled a restriction that was
deliberately removed from DE*: the number of coordinates a candidate changes was made
unrestricted-but-controlled there (a Bernoulli mask with a controlled mean, `smm/smm.jl:1516`)
and this is the same restriction under a different name in the other optimiser.
Observed consequence on a live `base_fc` run at the corrected optimum: per-coordinate steps
collapsing from 0.01 to 4.9e-05 over 300 iterations — **69× below the measured baseline
`ΔQ=1` half-width and 467× below the R1+R3 median** — because Corana chases a 0.4-0.6
acceptance target that temperature alone caps below 0.25, so it divides every window. And
because the geometric cooling branch is now live, `T` falls by exactly **one halving** over
the whole 5000-iteration budget, with `curr == best` at every trace line: monotone greedy
descent, not annealing.
**Status: being decoupled** — Bernoulli mask for the count, measured widths for the scale,
Corana independent of both.

*Naming note for whoever reads this next:* five unrelated `k` concepts exist in `smm/` —
`sa_subset_k` (SA proposal dimension), the generator's loop `k` (sweeps 1…npar building the
DE init/reheat population, `:1082`), `gen_per_k` and `de_gen_per_k` (candidates per `k` in
that sweep), and `k_star`. Only the second is about initialisation. Confusing them is easy
and has already happened once.

None of these is asserted as broken. Each is a **derived hypothesis with a traced status**,
which is the form that survives being read six months from now: the reasoning is on the
page, so a future session can check it rather than rediscover it.

### Cold-start settings the corrected objective needs

Configuration, not respecification — but without them a cold run looks like the fix broke
the estimator. `sa_t0_rel = 1e-4` is a **warm-start** temperature: cold it gives
`T0 = 0.0993` against `Q0 = 1195.65` and accepts **0 of 13** proposals, while
`smm_main.jl`'s own header documents **0.05**. And `sa_step = 0.20` is two orders above the
measured `ΔQ=1` half-widths (median 3.4e-03): at 0.20 SA accepted **0 of 150** at 97%
feasibility; at 0.01 it accepted 7 of 150 and descended.

On a **warm** start neither matters the same way — the screen-and-cluster procedure supplies
the dispersion, so SA is not the thing crossing the box and a small `T0` is correct there.

---

## v18.4.0 — 2026-08-16 · `412f80e`

**MAJOR.** SA→DE two-stage optimiser, population generator, `P_U = 1` normalisation.

**Comparability: broken.** The `P_U = 1` normalisation removes an exactly flat direction —
`A → A + s` with `P_U, P_S, b_U, b_S, σ_S` all scaled by `exp(−s)` left all 31 moments
numerically identical (max deviation 0.000e+00 at s = 0.05), so the criterion had a
one-parameter ridge along which every point was indistinguishable. Parameter values are not
comparable to v17.x.

**Also in this version, discovered later and fixed in v19.0.0:** the rewrite that moved the
loose `SA_*` constants into `run_params` dropped four kwargs at the `run_smm` call site,
leaving `sa_subset_k`, `sa_halflife`, `sa_rate_tol` and `sa_rate_span` inert for three
versions. Recorded here because this is where a reader looking for the regression's origin
will look.

---

## v17.0.1 — 2026-08-09 · `000db96`

Fixed the enrolment data target; cleaned up log prints.
**Comparability: broken** — a moment's target changed, so the objective's level and the
fitted parameters move with it.

---

## v16.7.1 — 2026-08-08 · `62b59b5`

The last version whose `run_smm` call site forwarded the SA settings correctly
(`smm_main.jl:987-990`). Recorded for that reason: it is the reference for what the
v18.4.0 rewrite lost.

---

## v15.5 — 2026-08-04 · `f35a803`

`run_all` SMM script added.

---

## v12.0 — 2026-07-16 · `a461fd3`

Described in its own commit message as *"last version that gets Q below 1 at both betas
fixed"*.

---

## Before v12.0

**Not reconstructed here, and the reason is worth stating:** only three commits in the
repository's history ever touched `code/smm/version.jl`, so for everything earlier the
version number exists only inside commit messages and cannot be tied to a tree state with
confidence. `git log --format='%h %ad %s' --date=short` is the record for that period —
notable entries include the Gauss-Jacobi grid switch (`03beebd`), wages moved to logs
(`37398a4`), and the ρ-NILF purge (`1bc4a53`).

**This is exactly the gap this file exists to close.** From v19.0.0 forward, every bump
writes its entry here in the same pass, so no future reader has to reconstruct intent from
a one-line commit subject.

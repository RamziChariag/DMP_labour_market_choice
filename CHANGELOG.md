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

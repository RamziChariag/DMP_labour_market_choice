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

# RoySearch v19.1.0 — settings lifecycle, and the SA proposal respecified

MINOR: the annealing proposal and the settings machinery changed; the objective, the
moment battery, the data targets and the solver did not. A run under v19.0.0 and a run
under v19.1.0 are estimating the same model — the optimiser explores it differently.

Two pieces of work, landed together because the second is what the first was waiting on.

## The annealing proposal

The proposal is no longer a fixed coordinate count at a shared scalar step. Each
iteration draws an independent Bernoulli(`p_move`) mask over coordinates plus one forced
index — DE's crossover construction, so the moved count is random and unbounded above
with a controlled mean — and moves each selected coordinate by its own step, seeded from
that coordinate's measured ΔQ = 1 half-width and adapted by Corana from there. The
scalar shared-step path is deleted, not switched off. `sa_subset_k` is removed from
`SMMRunParams` and the bundles migrated in the same pass.

**What was measured, at the stored `base_fc` optimum, 23 free parameters** — tables in
`output/smm/sa_proposal_*.csv`, method and derivation in `SETTINGS.md`'s graveyard entry:

- Half-widths span 0.0022–0.20 (91x, median 0.021); 10 of 23 coordinates fall below the
  0.01 the retired scalar step clamped at. That step logged 0 accepted of 50 iterations
  at full feasibility — every proposal legal, none absorbable.
- Acceptance and expected squared jump distance are both functions of `k·σ²` to second
  order, so neither identifies the coordinate count separately from the step scale.
  `p_move` is therefore pinned, not adapted against acceptance; Corana owns the step
  alone. Every comparison below reports descent **and** distance moved for the same
  reason.
- Feasibility, the one quantity outside that expansion, was measured as the candidate
  identifying signal and found **saturated**: 0.958–1.000 across k = 1…23. It carries no
  gradient here. `SAScale.feas_k` records it per run so a point where it does spread out
  is visible.
- The retired justification for multi-coordinate moves — that individually infeasible
  coordinates can be jointly feasible — **did not replicate**: 0 of 90 draws.
- Chains at equal solve budget (250 iterations, 3 seeds) rank the proposals: mean ΔQ
  **81.2** at the retired fixed k = 3, **87.8** for a dense mask, **90.3** for the
  shipped mask at `p·d = 1`, **91.5** for a strictly single-coordinate move. The shipped
  mask also moves further than the dense one (path 12.4 against 9.9), so the gain is
  descent rather than an acceptance artefact.
- **A one-shot sparsity scan disagreed with the chains and lost.** At matched
  displacement it ranked k ≈ 11 best and k = 1 3.2–3.4 se below; the chains reverse that.
  A large joint move is the better single bet and the worse thing to repeat. The scan is
  demoted to a diagnostic (`SA_SCALE_PER_K = 0` by default) and sparsity questions get
  measured on chains from here on.

`[SA config]`, printed from inside the walk, reports the proposal the optimiser actually
holds, and the trace reports the **realised** moved-count distribution rather than the
nominal `p_move`.

## Settings lifecycle

The rules are in **`SETTINGS.md`** at the repo root, which is now the reference for adding,
renaming and retiring a run setting. Read its graveyard section before proposing a switch.

## What changed

**A settings layer, declared once.** `code/smm/settings.jl` holds `env_setting`, the
`SETTINGS_REGISTRY` classifying all 50 `ROYSEARCH_*` keys by consumer and reachability,
and `LEGACY_ENV_KEYS`. The `_env_sym/_env_f64/_env_int/_env_bool` helpers were duplicated
verbatim in `smm_main.jl` and `MCMC_main.jl`; both now call the shared reader, and 46 call
sites were migrated. Six other entry points that include `smm.jl` (`transition_main`,
`model_main`, `policy_main`, `plots_and_tables`, `migrate_bundles`, `test_generator`) had
`settings.jl` added to their include list in the same pass.

**Three forwarding gaps closed.**

| gap | consequence |
|---|---|
| `run_smm(sa_subset_k, sa_halflife, sa_rate_tol, sa_rate_span)` — computed into `run_params`, never passed | ran on `run_smm`'s defaults `0/0/0.0/0` for v18.4.0–v19.0.0; the log asserted the intended values |
| `sa_cooling_halflife` vs the caller's `sa_halflife` | a name mismatch the name-matching audit could not see — one of the four original bugs |
| `_run_sa`'s `scale_cap` | referenced a parameter deleted when the cap was consolidated into `_WIDTH_CAP`; `:sa_de` died with a `MethodError` |

**`print_spec` no longer takes settings as keyword arguments.** It reads `spec.run`, which
is what the bundle records. This removes the mechanism by which the header could report a
configuration nothing was using: the caller can no longer hand it a value. Its SA and NM
lines are labelled `(requested)` and name `[SA config]` — printed from inside `_sa_loop` —
as authoritative. A new `[env]` block on both entry points lists the overrides in force,
sourced from the resolution log rather than from any caller's copy.

**Four MCMC keys renamed** out of the bare namespace: `ROYSEARCH_JAC_ONLY` and
`ROYSEARCH_SCREEN_{FRAC,CAP,FLOOR}` → `ROYSEARCH_MCMC_*`. The old spellings are in
`LEGACY_ENV_KEYS`, and `assert_no_legacy_env()` at both entry points now **errors** on a
stale key rather than ignoring it — the failure mode being closed is a launcher that
exports the old name and gets defaults in silence.

**`sa_subset_k` retired.** The env key and the forwarding are deleted; the `SMMRunParams`
field is kept dead and documented, because `Serialization` rebuilds a struct by field
position and removing it invalidates every `.jls` bundle on disk. See the SETTINGS.md
graveyard for what replaced it and why.

## Release gate

| gate | result |
|---|---|
| every file parses | 37 files, no `:error`/`:incomplete` nodes |
| full stack loads in driver order | solver + smm + candidates + mcmc_diagnostics + demc; every edited name resolves; `print_spec` arity 1 |
| `check_forwarding.jl` | **PASS** — 49 registry rows (17 inert), 49 keys read, 0 findings |
| gate fails on a real finding | verified: exit 1 with a finding, 0 clean, and `check_repo.sh` propagates it |
| negative controls | re-omitting `sa_subset_k` → caught (check 1); re-introducing the `sa_cooling_halflife` rename → caught (check 3); an unregistered key → caught (check 2) |
| changed path runs on real data | two full `:sa_de` runs, base_fc, exit 0: Q 842.05 → 787.84 and → 804.51 at reduced budget, scratch output only |
| SA behaviour | `[SA config]` reports `p_move=0.4348` (k\*=10 of 23, measured), 23/23 widths measured, step range 7.3e-4‥6.6e-2, accepted 3/6 — against the 0.04→0.00 acceptance that prompted this work |
| MCMC entry point | loads, `[env]` renders, renamed `ROYSEARCH_MCMC_JAC_ONLY` honoured (takes the JAC_ONLY branch) |
| both version constants agree | 19.0.0 in `smm/version.jl` and `data_processing_main.jl:56` |
| bundles still load | no struct changed shape; `sa_subset_k` deliberately retained for this reason |

**Known gap:** the reduced-budget runs verify the settings path, not the estimate. A
full-budget re-estimation is still owed once the SA respecification is finished, and the
`c`-bucket settings (17 of 49) are inert by path and therefore unexercised by any run —
their status is established by tracing, not by measurement.

# RoySearch v19.0.0

Version 19.0.0. **This is a MAJOR bump: the criterion itself moved.**

Four changes are attributed to this version, in `solver/` and `smm/` plus one guard in
`transition/`. They are listed in §§1–5 below.

## Scope: this is NOT a snapshot of a clean tree

**The working tree also carries uncommitted changes that predate v19.0.0 and are not part
of it.** They were already present (`git status` dirty against HEAD = v18.4.0) before any
v19 edit was made, and they are shipping alongside these four in the same uncommitted
tree. A reader diffing `code/` against v18.4.0 will see them and must not attribute them
here:

| inherited change | file | what it does |
|---|---|---|
| `MCMC_JAC_ONLY` default `true` → `false` | `smm/MCMC_main.jl` | switches the default MCMC path from Jacobian-only to running the full chain |
| `de_max_reheats` default 50 → 20 | `smm/smm_main.jl` | lower DE reheat cap |
| `param_symbol` / `fixed_symbol` / `FIXED_SYMBOLS` / `padr` helpers, and a new `symbol` column on `mcmc_results_{window}.csv` | `smm/smm_params.jl`, `smm/MCMC_main.jl` | display formatting; **changes a downstream CSV's column set** |
| `const _WIDTH_DQ = Ref(...)` + `ROYSEARCH_WIDTH_DQ` | `smm/smm.jl` | configurable ΔQ for the width diagnostic |
| SA reheat checkpointing (`_sa_loop`, `_run_sa`, `run_smm`, and the `smm_main` call site) | `smm/smm.jl`, `smm/smm_main.jl` | the half-finished feature whose missing `_run_de` counterpart is the blocker fixed in §1 |
| VM launcher edits | `scripts/vm_run.sh` | operational |

The last row is why §1 exists at all: the checkpointing feature was landed on the SA side
and left unfinished on the DE side, which is exactly the state that broke `:sa_de`.

`v19.0.0.diff` in the release artifacts is the **v19-only** diff. `inherited_changes.diff`
is the pre-existing set above, kept separate so the two are never conflated. Whoever
commits this tree should decide whether the inherited set belongs in the same commit —
in particular the `MCMC_JAC_ONLY` flip and the `mcmc_results` column addition, which
change behaviour and an output format and are **not** covered by v19's release gate.

## Why MAJOR

Q(θ̂) on `base_fc` is **842.05** under this version against **609.75** under v18.4.x, at
the *same* θ̂ and the same Np_S = 120. The estimate moves, every stored `base_fc` Q and
standard error is superseded, and δ_S is no longer welded to a quadrature node. A reader
of the paper has to be told, so the number is 19.0.0 and not 18.5.

Re-estimation is required. The two SA settings that need changing on a cold start are
documented in §5 below.

## 1. Blocker: `:sa_de` could not complete

`_de_stage` forwarded `checkpoint_path` to `_run_de`, which did not accept it, so every
`:sa_de` run — the shipped default two-stage path — died with a `MethodError` on entering
stage 2. `_run_de` now takes `checkpoint_path` with the same default and docstring as
`_run_sa` (`smm/smm.jl:1378`).

The `:de` branch of `run_smm` had the mirror-image defect: it never forwarded
`checkpoint_path` at all, so DE reheats wrote nothing while appearing to succeed. Both
halves are fixed together (`smm/smm.jl:2037`), and a forced reheat now writes and reloads
a checkpoint bundle.

## 2. R1 — the OJS margin is softened in free entry

`compute_Jbar_skilled` read the on-the-job-search margin as a hard `1{E¹ ≥ E⁰}` while
`solve_stationary_skilled!` smoothed the same margin with `_soft_oj_weight`. Free entry
and the stationary distribution therefore priced one margin at two resolutions, and J̄_S
jumped whenever a single grid cell flipped: the outer θ_S map was discontinuous in the
parameters while the block feeding it was continuous. Both loops now use
`_soft_oj_weight` (`solver/skilled.jl:390`, `:398`, and the fallback at `:425`).

**This is shipped for continuity, not accuracy, and the comment in the file says so.**
Against the hard rule the soft weight is 1.7–1.8× worse in RMS and carries a one-signed
bias about 60× larger, because `_soft_oj_weight` returns the covered fraction of the
interval [p_j, p_{j+1}] while multiplying the Gauss–Legendre *weight* wp_j, which is not
that interval's length. What it buys is that all 8–16 discontinuities per point leave the
free-entry map, and pooled feasibility goes from 58.0% to 94.3% (n = 192 paired
directions per radius, 2304 solves, McNemar p = 3.9e-45). An exact version would weight
by the cell's mass fraction under dΓ rather than by its length fraction.

## 3. R3 — exact-CDF cell masses for the quality densities

`build_skilled_precomp` sampled both quality densities pointwise at the quadrature nodes,
which put δ_S's Jacobian pole wherever the nearest node happened to fall. Both are now
differences of exact Beta CDFs across cell edges, divided by wp_j
(`solver/grids.jl:263`–`278`, consumed in `solver/skilled.jl:43`).

Nodes and weights are untouched, deliberately: `sg.wp` doubles as the Lebesgue dp measure
elsewhere in the solver, and a Gauss–Jacobi swap breaks it (∫ p dp = 0.4731 against the
exact 0.5).

**γ̃ is a cell mass per unit wp, not a pointwise density.** Every dΓ integral in the code
appears as `γ_j * wp_j`, so each carries the exact cell mass and the total is 1 to
machine precision at Np_S = 60, 120 and 480. It must never be read pointwise: the node
error is flat in N (RMS 1.194 at 60, 1.202 at 480). **R3 must never ship without R1** —
alone it makes θ̂ infeasible at Np_S = 120.

## 4. The one pointwise consumer, and the θ_S damping

`transition_solver.jl:353` was the only genuine pointwise read of `pre.γvals`. Rather
than carry a second sampled density under a second meaning, the line is left in cell-mass
form and documented: `path.eS` is itself a per-unit-wp density (every consumer weights it
by `sg.wp`), so both sides of the flow balance are in the same units and ∫(inflow) dp is
exactly f_S·u_S, as the stationary solve has it.

θ_S had no damping path at all — `damp_pstar_S` damps only the cutoffs, and only on the
`use_anderson = false` branch. Free entry is now under-relaxed at w = 0.9, LMR's own value
(`params.f90:275`), inside `solve_skilled_block!` (`solver/skilled.jl:551`). It recovers
15 of the 25 residual failures, loses none of 96 feasible controls, and moves Q by at most
6.5e-07 relative over those controls (1.2e-10 at θ̂ itself).

It ships as `const DAMP_THETA_S = Ref(0.9)` in `solver/params.jl:38`, **not** as a
`SimParams` field: Julia's serialiser reads structs positionally by field count, so a new
field would make every bundle already on disk unreadable.

Cumulative feasibility: 58.0% → 94.3% (R1) → 95.7% (R1+R3) → 98.3% (all four).

## 5. SA settings the corrected objective needs

Both SA knobs were tuned for the warm start that `INIT_MODE = :warmstart` makes the
default, and both are wrong cold. `sa_step`'s default is retuned to **0.01** and exposed
as `ROYSEARCH_SA_STEP`; `sa_t0_rel` keeps its warm default of 1e-4. On a **cold** start
set both from the environment:

    ROYSEARCH_SA_T0_REL=0.05   ROYSEARCH_SA_STEP=0.20

At `sa_t0_rel = 1e-4` a cold start gives T0 = 0.0993 against Q0 = 1195.65 and accepted 0
of 13 proposals — annealing with no uphill acceptance is descent. The 0.20 step is the
mirror image warm: two orders above the measured ΔQ = 1 half-widths (median 3.4e-03), so
it accepted 0 of 150 at 97% feasibility, while 0.01 accepted 7 of 150 and descended.

## Release gate

| gate | result |
|---|---|
| every changed file parses | 10 files, no `:error`/`:incomplete` nodes |
| every entry point loads in driver order | solver + smm + transition; all edited names resolve |
| changed path runs against real data | Q(θ̂) = 842.051543 at Np_S = 120; NM, DE and `:sa_de` all completed; scratch checkpoints only |
| both version constants agree | 19.0.0 in `smm/version.jl` and `data_processing/data_processing_main.jl:56` |
| old bundles still load | 5 of 7 `.jls` load; `crisis_fc` and `crisis_covid` raise `EOFError` **at HEAD too** — pre-existing, not from this version |

No struct changed shape in this version, which is why the damping is a module-level `Ref`.

## Object-to-output crosswalk

| output | produced by |
|---|---|
| `output/smm/smm_result_{window}_diagonalW.jls` | `smm/smm_main.jl` (`ROYSEARCH_WINDOW={window}`) |
| `output/tables/smm_estimates_{window}_diagonalW.csv` | `smm/smm_main.jl`, via `save_results` |
| `output/smm/candidates_{window}_diagonalW.jls` | `smm/smm_main.jl` with `INIT_MODE=:clusters` |
| `output/smm/mcmc_chain_{window}_diagonalW.jls`, `mcmc_results_*.csv` | `smm/MCMC_main.jl` |
| `output/transition/*` | `transition/transition_main.jl` |
| `output/tables/policy_*` | `policy/policy_solver.jl` |
| `data/derived/*` (all estimation inputs) | `data_processing/data_processing_main.jl` — see `data/derived/README.md` |
| `output/settings_audit.csv` | hand-assembled from `smm/settings.jl`'s registry plus a measured `:sa_de` run; regenerate when the settings surface changes |
| settings lifecycle rules | `SETTINGS.md` (repo root) — adding, renaming, retiring a switch; graveyard of what was tried |
| settings release gate | `scripts/check_forwarding.jl`, run by `scripts/check_repo.sh` |

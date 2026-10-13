# Settings lifecycle

How a run setting is added, renamed, retired, and verified in RoySearch. Read this
before adding a switch, and before assigning a version number.

The rules exist because of one measured failure. Between v18.4.0 and v19.0.0,
`smm_main.jl` computed `sa_subset_k = 3`, `sa_halflife = 5000`, `sa_rate_tol = 0.05`
and `sa_rate_span = 300` into `run_params` and then called `run_smm` passing none of
them. Julia substituted `run_smm`'s own defaults — `0, 0, 0.0, 0` — in silence, because
they are keyword arguments with defaults. Nothing crashed. The run produced numbers.
`print_spec` reported the `run_params` values, so **the log asserted a configuration the
optimiser was not using, for four versions.** The visible symptom was an annealing stage
that accepted 4% of proposals and then 0%: `subset_k = 0` disables per-coordinate Corana
adaptation entirely, and the scalar fallback clamps its step at 0.01 — about 3x the
measured ΔQ=1 half-width (median 3.4e-03 at the base_fc optimum), so every proposal
overshot. `sa_halflife = 0` separately selected the logarithmic cooling branch, whose own
comment warns it spends its whole descent in the first hundred iterations.

The lesson is not "be careful". It is that **a keyword argument with a default converts a
wiring mistake into a silent behavioural change**, and that a settings block printed from
the caller cannot detect one.

## The prohibition

> A settings value that is computed and then not forwarded is a **silent defect**.
> Defaults on keyword arguments are what make it silent.

Two consequences, both enforced by `code/gates/check_forwarding.jl`:

- Caller and callee must spell a setting **identically**, allowing only the subsystem
  prefix the codebase already uses (`sa_max_iter` → `_run_sa`'s `max_iter`). The check
  pairs names, so any other rename is a hole in it. `sa_cooling_halflife` spelled
  `sa_halflife` in the caller was exactly that hole, and it was one of the four original
  bugs — the check written to prevent them would not have caught it. It is now renamed to
  match, and check 3 fails on any new mismatch.
- The **authoritative** settings print is sourced from *inside the consumer*, off the
  values it actually holds. `[SA config]` in `_sa_loop` is that line. A print sourced
  from the caller cannot detect a forwarding gap; one sourced from the consumer cannot
  miss it. Where both exist, the caller's copy is labelled `(requested)` and says which
  is authoritative.

## Where a setting is declared

Four places, and a setting is not finished until it exists in all of them.

| # | Place | What it carries |
|---|---|---|
| 1 | the `env_setting(:KEY, default)` call | the **default** and the comment justifying that value |
| 2 | `SETTINGS_REGISTRY` in `code/smm/settings.jl` | the **classification**: which consumer must read it, and whether that path is live |
| 3 | `SMMRunParams` in `code/smm/smm_params.jl` | persistence, *only if* the value must be recoverable from a serialised bundle |
| 4 | the consumer's keyword argument | the point of use |

Place 1 and place 2 are deliberately separate. The call site is where a reader asking
"why this value" looks; the registry carries what the source cannot express — the
intended endpoint and whether it is reachable. `check_forwarding.jl` reads both and fails
when they disagree, which is what makes place 2 a gate rather than a comment.

**Place 3 has a hard constraint.** Julia's serialiser reads structs positionally by field
count, so *adding a field to `SMMRunParams` makes every `.jls` bundle already on disk
unreadable* — verified, not assumed. A new setting that must persist therefore forces a
bundle migration in the same pass (`code/tools/migrate_bundles.jl`); one that need not
persist travels as a keyword argument on `run_smm` instead. `sa_subset_k`, `sa_halflife`,
`sa_rate_tol` and `sa_rate_span` are keyword arguments for exactly this reason, which is
also why they were forwardable-and-not-forwarded in the first place.

## Adding a setting

1. Add the `env_setting(:KEY, default)` call, with the comment that justifies the
   default. Put the justification where the value is, not in a note elsewhere.
2. Add a `SettingDecl` row naming the consumer as `file:function` and the status.
3. Give the consumer a keyword argument **spelled the same** (subsystem prefix aside).
4. Forward it at every call site, explicitly.
5. Make sure it appears in a print sourced from inside the consumer, not from the caller.
6. Run `bash code/gates/check_repo.sh`. A new setting is not done until this is green.

## Where a default should be removed

A keyword default is what makes an omission silent. Remove the default — leaving a
required keyword argument, so omission is a `MethodError` at the call rather than a
degraded run — when this criterion holds:

> **A wrong value degrades results silently rather than crashing.**

Settings that qualify, and why:

| setting | why omission must crash |
|---|---|
| `sa_subset_k` | `0` does not mean "off"; it silently switches the proposal from per-coordinate Corana steps to one shared scalar. This is the setting whose default cost four versions. |
| `sa_halflife` | `0` silently selects a different cooling law, not a disabled one. |
| `sa_rate_tol`, `sa_rate_span` | `0` silently disables the early stop, so a run burns its whole budget and reports normally. |
| `nm_simplex_step` | `0` silently reverts to Optim's `AffineSimplexer`, whose vertices are all infeasible here — thousands of evaluations scoring `Inf`. |

Settings that do **not** qualify, and must keep their defaults: anything whose wrong
value is visible in the output it produces (`trace_stride`, `show_trace_*`), anything
whose default is the documented off-switch and where off is a legitimate configuration
(`de_avg_tol`, `checkpoint_path = ""`), and `rng`, where a default is the reproducibility
mechanism rather than a hazard.

The tension is real and is why this is a criterion rather than a blanket rule: removing a
default breaks every call site that legitimately omits the argument, including the smoke
tests and `test_generator.jl`, which construct these consumers directly. Remove a default
only together with a pass over every call site, and treat it as the MINOR-or-higher change
it is.

## Removing a setting: document → trace → eliminate

In this order. Skipping the first two is how a switch becomes a question someone
re-answers a year later.

1. **Document.** Write the graveyard entry below *first*: what the switch was for, what
   replaced it, and the evidence that retired it. Record the measurement, not the verdict.
2. **Trace.** Find every mention — the `env_setting` call, the registry row, the struct
   field, the keyword argument, every call site, the prints, the launchers in
   `code/ops/*.sh`, and the smoke tests. `grep -rn` for both the ASCII and Unicode
   spellings where the key map applies.
3. **Eliminate.** Delete the switch and its plumbing in one pass. If it was an
   `SMMRunParams` field, the bundle-compatibility constraint above applies in reverse —
   removing a field invalidates existing bundles exactly as adding one does.

A **rename** is a removal plus an addition, with one extra obligation: add the old key to
`LEGACY_ENV_KEYS` in `settings.jl`. Both entry points call `assert_no_legacy_env()` at
startup, so a launcher still exporting the old spelling **fails loudly** instead of
running under silently-substituted defaults. Remove the legacy entry only once no script
or note in circulation could still carry the old name — not at the next release.

## The release gate

Run before assigning a version number. `bash code/gates/check_repo.sh` covers the
settings half; the rest is the versioning gate in full.

1. Every changed file parses (`Meta.parseall`, walking for `:error` / `:incomplete`).
2. Every entry point loads in driver order, and every edited name resolves.
3. **`check_forwarding.jl` exits 0.** Its four checks are: forwarding, registry parity,
   name parity, and legacy keys. A finding is a defect, not a warning — inert settings
   are declared `:inert` in the registry and do not appear.
4. **`check_dead_settings.jl` exits 0.** Forwarding and arrival are not the same thing:
   this parses each function's AST and fails on any keyword argument that is declared and
   never read in its own body. It exists because `sa_target_fin` passed the forwarding
   check for four versions while `_sa_loop` ignored it and the settings banner printed it
   as though it governed the walk — and because the same check found `_run_sa` dropping
   `checkpoint_path`, which silently disabled SA reheat checkpointing. A keyword kept
   deliberately unread (signature compatibility) says so in a comment beside it.
5. **`check_threading.jl` exits 0, run with `-t 2` or more.** A solve inside
   `Threads.@threads for i in 1:2` must return a finite objective and be *bitwise* equal
   to the serial value. It exists because v40.0.0 shipped a solver whose two
   `@threads :static` loops could not nest: `smm_objective` reported the resulting
   exception as `Inf`, so the DE population, the ΔQ=1 width measurement and the
   adaptive-FD Jacobian each read every parameter point as infeasible, and the whole
   regression was invisible because every test at the time ran serially. The bitwise
   half matters as much as the finiteness half — a schedule-dependent reduction would
   trade a loud `Inf` for a silent wrong number.
6. The changed path runs against real data, writing to a scratch directory.
7. Both version constants agree (`code/smm/version.jl` and
   `code/data_and_descriptives/data_processing_main.jl:56`).
8. Old serialised bundles still load, or were migrated in the same pass.
9. `VERSION_NOTES.md` records the settings added, retired or re-defaulted.

The audit `output/settings_audit.csv` (see below) is regenerated when the settings
surface changes, and its "effective value" column must come from an actual run.

## Settings that choose a file

Two settings say which estimate a run reads or writes rather than how it computes. They
exist because from v57.0.0 each estimate has a file of its own (`estimate_path`,
`code/paths.jl`; the file table in `README.md`).

- **`CRISIS_BASE`** (`smm_main.jl`, directly below `WINDOW`; `:smm` or `:mcmc`, default
  `:smm`). A crisis window holds the ability distribution at its base window's SMM
  estimate (`:smm`, writing `estimate_<crisis>W_smm.jls`) or at the base chain's estimate
  (`:mcmc`, writing `estimate_<crisis>W_smm_mcmcbase.jls`, the crisis chain's seed). A
  base window never reads it, so it can stay set while the window changes, and the
  registry declares it `:inert` because the shipped `WINDOW` is a base window. Any other
  value stops the run.
- **`POINT_ESTIMATE`** (`:smm` or `:mcmc`, default `:smm`). Not an environment setting: a
  constant at the top of each script that reads an estimate (`transition_main.jl`,
  `model_main.jl`, `policy_main.jl` and the tools), beside that script's other constants.
  `:smm` reads `estimate_<w>W_smm.jls`, `:mcmc` reads `estimate_<w>W_mcmc.jls`.
  `estimate_path` has no default `point`, so no reader falls back to a file it did not
  name. In `transition_main.jl` it also tags every file the run writes, the transition file
  `transition_<pair>W_<point>.jls` and its tables and plots `…W_<point>`, so the runs at the
  two points coexist.

Two MCMC defaults changed in the same version. **`MCMC_CHECKPOINT`** is `false`, declared
`:inert`: set, it writes the best visited point to `estimate_<w>W_mcmcbest.jls`, and that
file is all it governs. The chain is snapshotted to `output/chains/chain_<w>W.jls` every
`MCMC_LMR_PRINT` generations (500) whether or not it is set, so a run killed before the
sampler returns still leaves its draws, and a chain writes no estimate file but its own.
**`MCMC_BURN`** is `0.8`, which keeps the last 1,000 generations of the 5,000-generation
budget, LMR's retained length.

## Graveyard — what was tried and dropped

For future sessions. Each entry says what the switch did, why it went, and on what
evidence, so it is not re-proposed.

### `SCENARIO_BUNDLES` — the transition's bundle files, per scenario
**Retired in v57.0.0.** A constant in `transition_main.jl` naming the two bundles of each
scenario, each a window (the live estimate) or a file path. Its one non-default entry
pointed the COVID baseline at an ξ_S backup under `output/backups/`, and `check_pair`
rejected that pair: ρ_x = −0.4568189183164799 in the backup against −0.49412406238821294
pinned in the crisis estimate. Replaced by `POINT_ESTIMATE`, which reads both windows of a
pair through `estimate_path` at the same point. The backup stays on disk, untouched.

### `ROYSEARCH_WIDTH_BASE` — the base point of `width_audit.jl`
**Retired in v57.0.0.** An environment key read only by `code/tools/width_audit.jl`, through
`ENV` rather than `env_setting`: `postmean` audited the posterior-mean bundle when one
existed and fell back to the SMM bundle without stopping, `seed` audited the SMM bundle.
Replaced by the tool's `POINT_ESTIMATE` constant (`:mcmc`, `:smm`), which reads the file
it names or stops. The key is in `LEGACY_ENV_KEYS`, so an exported value stops
`width_audit.jl`, `smm_main.jl` and `MCMC_main.jl` with a pointer to the constant.

### `local_k` — fixed-sparsity DE population construction
**Retired before v19.0.0; last remnant (a stale comment in `_run_de`) removed with this
document.** It selected the older population builder, which perturbed a fixed number `k`
of coordinates per member. Replaced by `generate_population`, which draws at *every*
sparsity `k = 1:n_free` and allocates slots by measured yield, so the trade-off is
measured per run and re-measured at each reheat instead of being fixed in advance.
Evidence: useful-draw rate 0.51 at k=3 against 0.08 at k=25, where the loss is not
infeasibility (which barely moves) but the compounding of many small increases in Q. A
fixed `k` cannot express that, and a switch to restore it would only reintroduce a
worse-measured configuration. `de_gen_per_k` and `de_local_sigma` are its replacements.

### `sa_subset_k` / `ROYSEARCH_SA_SUBSET_K` — fixed coordinate count for the SA proposal
**Retired when the SA proposal was respecified. The env key and the forwarding are
deleted; the `SMMRunParams` field survives as a dead field and must not be read.**

It set how many coordinates each annealing iteration perturbs — a fixed `k`, with each
chosen coordinate carrying its own Corana-adapted step. Two things were wrong with it as
a *setting*. The value `0` did not mean "off": it disabled per-coordinate adaptation
altogether and collapsed the whole step vector to one shared scalar, which is what made
the v18.4.0 forwarding gap so damaging rather than merely wrong. And a single fixed `k`
is the same category error as a single fixed step scale — it asserts in advance how many
coordinates a good move touches, on an objective where that is a property of the local
geometry.

Replaced by `sa_proposal_scale`, which measures the proposal at the start point: an
independent Bernoulli(`p_move`) draw per coordinate plus one forced index, so the number
of coordinates moving is Binomial(d, `p_move`) conditioned to be at least one — random
and unbounded above, with `p_move` and the per-coordinate step both read off a sparsity
scan at the measured ΔQ=1 half-widths. This is deliberately the same construction the DE
generator already used for its population, so both stages measure the local geometry the
same way instead of one measuring and the other being told. Configured by `SA_SCALE_PER_K` and
`SA_SCALE_SIGMA`, which mirror the DE generator's `de_gen_per_k` / `de_local_sigma`;
the bisection cap is shared as `_WIDTH_CAP` rather than exposed per proposal.

**Why the field is still there.** `Serialization` rebuilds a struct by field *position*,
so deleting `sa_subset_k :: Int` invalidates every `.jls` bundle on disk exactly as
adding a field would. It is therefore left in place, documented as dead, and removed in
the next pass that migrates bundles for an independent reason. This is the standing
pattern for retiring an `SMMRunParams` field: stop reading it, document it at the
definition, delete it when a migration is already happening.

**The measurement behind `p_move`, and two things it ruled out.** Recorded because both
are natural next ideas and both are dead ends at this point in parameter space. All
figures are at the stored `base_fc` optimum, v19.0.0 solver, 23 free parameters.

*Acceptance cannot identify the coordinate count.* To second order the Metropolis
exponent has mean `−(σ²/2T)·Σ_j h_jj` and variance `(σ²/T²)·Σ_j g_j²` over the moved
set — both linear in the number moved at fixed `σ²`. Acceptance is therefore a function
of the product `k·σ²` alone, and expected squared jump distance is `k·σ²` by
construction, so neither separates the two. Since Corana already adapts the step
against acceptance, adapting `p_move` against acceptance as well would be two knobs on
one equation: the pair drifts along `k·σ² = const` with nothing pinning where it lands.
This is why `p_move` is measured once at the start and held, not adapted during the walk.

*Feasibility was the proposed escape and it is saturated here.* Feasibility sits outside
that expansion — a proposal moving `k` coordinates has `k` independent chances to land
infeasible — so it should identify `k` where acceptance cannot. Measured, it does not:
across `k = 1…23` at two matched displacement scales (0.33 and 1.0 width-normalised,
48 draws per cell, 768 solves), `P(feasible)` ranges only **0.958–1.000**. The feasible
set is not perforated on the scale the proposal operates at, so feasibility carries no
usable gradient in `k`. `SAScale.feas_k` records this per run precisely so a later point
where it *does* spread out is visible rather than assumed away.

*The codebase's own justification for a joint move did not replicate.* The retired
comment claimed "coordinates that are individually infeasible can be jointly feasible,
so a strict sweep cannot reach some improving points." Tested directly — draw a
displacement `u` on a `k`-subset, then evaluate the joint move against each single-
coordinate component of that same `u` — at `k = 2, 3, 5`, 30 draws each, 390 solves:
joint feasibility was 1.000 everywhere, and **0 of 90 draws** had a feasible joint move
whose components were all infeasible. The claim is not supported at this point, and it
is not the reason to move several coordinates.

*And the one-shot scan that replaced it was itself overruled — by chains.* This is the
part worth reading. A scan of one-shot gains from the incumbent at matched displacement
is not flat in `k`: `k = 1` sits **3.2–3.4 combined standard errors below the best `k`**
at both displacement scales, and the gain-weighted mean sparsity is 10.4 (L = 0.33) and
11.7 (L = 1.0), pooling to 11.1 of 23 → `p_move ≈ 0.48`. That was the first
specification. Run as actual annealing chains at equal solve budget — 250 iterations,
3 seeds, from the same point, `output/smm/sa_proposal_arm_comparison_base_fc.csv` — the
ordering **reverses**:

| proposal | mean ΔQ | mean acceptance | mean path (width-normalised) | moved/iter |
|---|---|---|---|---|
| retired fixed `k = 3` | 81.2 | 0.173 | 21.1 | 3 exactly |
| mask at `p·d = 12` (the scan's own answer) | 87.8 | 0.156 | 9.9 | 12.4 [5–19] |
| mask at `p·d = 1` (**shipped**) | 90.3 | 0.201 | 12.4 | 1.95 [1–6] |
| single coordinate, no mask | 91.5 | 0.335 | 13.4 | 1 exactly |

The low-density mask beats the scan's recommendation on descent **and** moves further,
so this is not the acceptance artefact a shorter proposal produces — that check is why
both columns are here. The mechanism: the scan measures the gain a proposal makes *from
the incumbent in one step*, while a chain compounds accepted moves and lets Corana
retune each coordinate's step against its own record. **A large joint move is the better
single bet and the worse thing to repeat.** Any future sparsity question must be measured
on chains, not on one-shot draws.

`SA_SCALE_P_MOVE` ships at 1.0 rather than at the marginally-better single-coordinate
move because the mask's tail is what reaches joint directions at all, and 0.4 in ΔQ over
three seeds does not justify removing that capability. `SA_SCALE_PER_K` defaults to 0:
the scan survives as the diagnostic that measures the feasibility saturation above, and
sets nothing.

Note the matching, which is a trap in its own right: at *unmatched* displacement a draw
at `k = 23` moves √23 ≈ 4.8x as far as one at `k = 1`, so an unmatched sparsity scan
reads move size back to itself. `_sparsity_draws!` takes `match_total_displacement` for
exactly this reason, and its two callers set it differently on purpose — a population
wants spread, a comparison wants matching.

The empirical literature on move schemes reaches a compatible conclusion on smooth
benchmarks (single-particle moves most efficient on Lennard-Jones). That is consistent
with the chain result rather than with the scan, and the mechanism that would have
separated this problem from those benchmarks — a perforated feasible set — is measured
absent here. Re-measure at a materially different point before relying on any of it.

### `de_avg_tol` — convergence stop on population spread
**Kept, permanently defaulted off (`0.0`), declared `:inert`.** It stopped a run when
`(Q_mean − Q_best)/|Q_best|` fell below tolerance. Across a six-configuration sweep at
equal evaluation budget this measure correlated **+0.915 with achieved ΔQ** — it ranks a
prematurely converged run as the most converged one, which is precisely backwards. Not
deleted, because it is the natural thing to reach for and the recorded correlation is the
reason not to; the registry row carries that reason so the next reader finds it before
re-enabling it.

### `sa_cooling_rate`, `sa_cooling_exp` — logarithmic cooling schedule
**Kept but only reachable at `sa_halflife = 0`.** The logarithmic law
`T = T_reheat·(log1p(rate)/log1p(rate·t))^exp` spends its entire descent in the first
~100 iterations, leaving the rest of the budget effectively greedy. Geometric cooling
(`T = T0·2^(−t/H)`, `H = 5000` against a 20,000 budget) keeps T in the band where the
ΔQ ~ 0.01–0.1 moves the run actually makes stay live. The logarithmic branch is retained
as the documented `sa_halflife = 0` fallback rather than deleted, since it is the
schedule most of the annealing literature assumes; **it is not a configuration to
prefer.** Note the trap: `0` here reads like "disabled" and in fact selects this branch.

### `nm_f_tol`, `nm_x_tol` — Nelder-Mead tolerances
**Kept for the gradient methods, `:inert` under Nelder-Mead.** Optim's Nelder-Mead
cannot read either: its `assess_convergence` returns `(false, false, g_converged, false)`
— `x_converged` and `f_converged` are literals (`nelder_mead.jl:317`). Sweeping `f_reltol`
over 1e-12 … 5e-1 leaves the iteration count unchanged at 111 on a smooth problem. They
remain because the same branch serves `:lbfgs` and `:bfgs`, where they are live. Under
the shipped `method = :sa_de` the whole branch is unreached, which is why all eight `NM_*`
keys are declared `:inert` by path rather than deleted.

### `ROYSEARCH_JAC_ONLY`, `ROYSEARCH_SCREEN_*` — unprefixed MCMC keys
**Renamed to `ROYSEARCH_MCMC_*` with this document; old names in `LEGACY_ENV_KEYS`.**
They configured `MCMC_main.jl` only, but sat in the bare `ROYSEARCH_*` namespace where
they read as global run settings — and `SCREEN_FRAC`/`SCREEN_CAP`/`SCREEN_FLOOR` in
particular are meaningless outside `MCMC_INIT = :screen`. Renamed for one namespace per
entry point. The legacy entries are what stop a stale launcher from silently reverting
them to defaults.

### `DAMP_THETA_U` — why it exists and why it is 0.9
**Ships as `const DAMP_THETA_U = Ref(0.9)` in `solver/params.jl`, alongside `DAMP_THETA_S`
and for the same structural reason.** The unskilled free-entry map was the only one of the
two iterated raw, and the skilled block's own comment already recorded that the raw map is
not a contraction. Measured on the `a_ℓ` line at `base_fc`: θ_U ran an exact period-8 orbit
between the clamp ceiling and the floor, nine of nine swept points solve at w = 0.9 against
four shipped, and `Q` at the points that already solved moves by at most 4.1e-07 relative.
The value is not tuned — it is `DAMP_THETA_S`'s, i.e. LMR `params.f90:275`. w = 0.5 was also
tested and recovers eight of nine, so the choice is not perched on a knife edge; the ninth
fails in the GLOBAL loop there, not the unskilled block.

### `OJS_PARTICIPATION` — the on/off versus covered-fraction switch
**Ships as `const OJS_PARTICIPATION = Ref(:fraction)` in `solver/params.jl` (v39.0.0).**
`:indicator` is the pre-v39 node rule and is kept permanently so a stored run can be
reproduced exactly; `:fraction` aggregates the participation decision over the ability cell
and is the default. Not env-settable, and a module-level `Ref` rather than a `SimParams`
field, for the reason the next entry gives. Unlike the dampers this one **does** change the
allocation — `Q` moves 5.6e-05 relative at θ̂ — so it is a specification switch, not a
relaxation knob, and the two must not be confused when reading a bundle's provenance.

### `anderson_m` — Anderson acceleration memory
**Retired in v39.3.0. The `SimParams` field and all six sites are deleted, and the four
stored bundles were migrated in the same pass.** It set how many past iterates the
accelerator mixes, and nothing ever read it: `Anderson1` / `anderson1_update!`
(`solver/grids.jl:416-447`) is hardcoded memory-1, so the value reached the solver and was
ignored. Evidence that it is dead, not merely unused: a run at `anderson_m = 3` is bitwise
identical to shipped — maximum absolute difference `0.0` across every field over all 200
iterations at both test points — and removing it moves `Q` at each window's θ̂ by exactly
zero on all four bundles. `use_anderson` stays: since v45.0.0 it governs the unskilled
block and the global loop only (`unskilled.jl:496,499`, `solver.jl:308`; see the
skilled-block entry below). `anderson_reg` stays as a field but is not read:
`anderson1_update!` (`solver/grids.jl:465`) applies its own fixed guard, `denom < 1e-12`.

The six sites were the field itself (`solver/params.jl:310`) and five constructions of a
`SimParams`: `solver/params.jl:441` inside `initialise_model`, `solver/model_main.jl:164`,
`smm/smm_main.jl:149`, `policy/policy_main.jl:202` and
`transition/transition_simulation.jl:132`. Removal followed the standing pattern above for
a serialised field — it waited for a migration that was happening anyway, here the
addition of `RunProvenance.migrated_from`.

### `DAMP_THETA_S` as a `SimParams` field
**Rejected; ships as `const DAMP_THETA_S = Ref(0.9)` in `solver/params.jl:38`.** Free-entry
under-relaxation on θ_S belongs conceptually in `SimParams`, and that is where it was
first tried. It cannot go there: Julia's serialiser reads structs positionally by field
count, so the added field made every `.jls` bundle on disk unreadable. The module-level
`Ref` is the deliberate alternative. It is **never assigned at runtime** — nothing writes
`DAMP_THETA_S[] = …` — so the value is `0.9` (LMR `params.f90:275`) on every path.

**From v40.0.0 the `Ref` is a CEILING, not the weight itself.** `solve_skilled_block!`
carries a local weight that halves on a sign reversal in the undamped free-entry target
(floor `0.1`) and relaxes back by ×1.2 toward `DAMP_THETA_S[]` otherwise. The state is
local to the call, deliberately: a module-level `Ref` carrying it between solves would make
a result depend on the order in which parameter draws are evaluated. At θ̂ no reversal
occurs and the weight sits at the ceiling for every pass, so the shipped value still
governs where the estimates live.

The correction that matters for the paragraph below: **under-relaxation cannot stabilise
the oscillating roots at all.** The skilled outer map's slope in θ_S is
`(1/η_S)·dlnJ̄_S/dlnθ_S`, which contracts iff `dlnJ̄_S/dlnθ_S < η_S = 0.5`; on the ridge
that quantity measures 0.52–0.90, i.e. slope `b` = 1.04–1.80, and the damped slope is
`1 + w(b−1) > 1` for every `w ∈ (0,1]`. A shorter step there lets the iteration be repelled
from the unstable root and settle on a different, stable one. So the adaptive schedule does
not tame those points — it **selects among their equilibria**, which is the same
branch-dependence the next paragraph documents for a fixed `w`.

It is not env-settable by design, and the reason is not that the damper moves the fixed
point — it does not. Under-relaxation leaves the fixed point of the map algebraically
invariant, since `θ = w·G(θ) + (1−w)·θ` holds exactly when `θ = G(θ)`, as the
`DAMP_THETA_U` entry above already says; measured at `2.7e-10` relative on `Q` at θ̂
between `w = 0.9` and `w = 0.5`. What `w` selects is **which** fixed point the solver
lands on. On a diagonal band of the (`a_ℓ`, `c`) plane at `base_fc` the skilled free-entry
locus `q_S(θ_S)·J̄_S(θ_S)` is non-monotone and has more than one root, and across a 19×19
plane `w = 0.9` against `w = 0.5` recovers 13 points, loses 17 and re-roots 20 — about 50
of 361 — with `Q` moving up to `8.3e-02` relative and `θ_S` up to `0.134`. Three of the
re-rooted points were solved under both values and both states satisfy all three adding-up
identities to machine zero, so they are two genuine equilibria at one parameter vector. A
`Q` recorded from such a point is therefore branch-dependent, and a run that silently used
a different relaxation would not be comparable to a stored estimate.

### Anderson acceleration in the skilled block
**Removed in v45.0.0; the code path is gone, not switched off.** The skilled outer loop
(`solve_skilled_block!`, `solver/skilled.jl:765`) is a plain damped fixed-point iteration:
the cutoffs are installed as `w·proposal + (1−w)·old` with `w = damp_pstar_S` (both p*_S and
p^oj_S), and θ_S takes the adaptive free-entry step under the `DAMP_THETA_S` ceiling. The
unskilled block and the global loop keep Anderson under `use_anderson`, which therefore no
longer covers every loop. There is deliberately no toggle for the skilled block.

What it did and why it went: the joint Anderson(m=1) step on `[θ_S; p*_S; gap]` took its
mixing weight from all components at once, 92% of the denominator from the gap block, so the
θ_S step was set by the cutoffs. In the base_fc multiplicity band it reversed the θ_S step on
45–53% of passes, leaving a drift of at most 9e-5 per pass, and those points never converged.
Without it the 19×19 base_fc (a_ℓ, c) plane goes from 342 to 360 feasible points (none lost),
the estimates keep their equilibrium (|ΔQ|/Q ≤ 1.3e-7), and a restart at the repelling middle
root no longer converges there. The v45.0.0 changelog entry has the measurements.

Removing it in ALL loops (`use_anderson = false`) was measured on v43 and rejected: the plane
collapses to 45/361, because the unskilled block stops converging (304 of 316 failures have
`converged_U = false`).

**`damp_pstar_S` is live from v45.0.0.** Before, it was read only on the non-Anderson skilled
branch, which no shipped entry point selected, so its value did nothing. It now weights every
skilled cutoff install. The stored bundles and `smm_main.jl`/`model_main.jl` carry `0.50`,
which is the value every v45.0.0 measurement used. `policy_main.jl` and `initialise_model`
carry `1.00` (undamped cutoffs); the policy settings converge at all four estimates with
either value. `damp_pstar_U` is still read only when `use_anderson = false`.

### `DAMP_POJ_S` — extra under-relaxation on p^oj_S
**Retired in v45.0.0.** A module-level `Ref(1.0)` that blended p^oj_S toward its previous
value at the end of each skilled pass. It existed because the Anderson branch installed
p^oj_S raw, so it was the only damping that branch could apply to it. With the Anderson branch
gone, `damp_pstar_S` already damps p^oj_S, and a second blend would only compound the first.
Nothing ever assigned it, and at `1.0` its block was skipped, so removing it changes no number.

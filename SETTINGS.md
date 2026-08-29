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

Two consequences, both enforced by `code/scripts/check_forwarding.jl`:

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
bundle migration in the same pass (`code/scripts/migrate_bundles.jl`); one that need not
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
6. Run `bash code/scripts/check_repo.sh`. A new setting is not done until this is green.

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
   `code/scripts/*.sh`, and the smoke tests. `grep -rn` for both the ASCII and Unicode
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

Run before assigning a version number. `bash code/scripts/check_repo.sh` covers the
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
5. The changed path runs against real data, writing to a scratch directory.
6. Both version constants agree (`code/smm/version.jl` and
   `code/data_processing/data_processing_main.jl:56`).
7. Old serialised bundles still load, or were migrated in the same pass.
8. `VERSION_NOTES.md` records the settings added, retired or re-defaulted.

The audit `output/settings_audit.csv` (see below) is regenerated when the settings
surface changes, and its "effective value" column must come from an actual run.

## Graveyard — what was tried and dropped

For future sessions. Each entry says what the switch did, why it went, and on what
evidence, so it is not re-proposed.

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

### `DAMP_THETA_S` as a `SimParams` field
**Rejected; ships as `const DAMP_THETA_S = Ref(0.9)` in `solver/params.jl:38`.** Free-entry
under-relaxation on θ_S belongs conceptually in `SimParams`, and that is where it was
first tried. It cannot go there: Julia's serialiser reads structs positionally by field
count, so the added field made every `.jls` bundle on disk unreadable. The module-level
`Ref` is the deliberate alternative. It is **never assigned at runtime** — nothing writes
`DAMP_THETA_S[] = …` — so the value is `0.9` (LMR `params.f90:275`) on every path, and it
is not env-settable by design: it changes the solver's fixed point, and a run that
silently used a different relaxation would not be comparable to a stored estimate.

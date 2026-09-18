# The LMR-equivalent estimator: one object, one configuration

## The reported object

**Point estimate = mean of the pooled retained chain draws. Standard error = standard deviation
of those same draws.** That is Lise–Meghir–Robin (2016) Appendix C, and it is the only thing
this project reports. In the code it is `post_mean` and `se_chain`.

Not `se_curvature`, not `se_bound`, not a criterion-profile width, not a χ²/dof-inflated
interval. Those exist in the codebase and are printed as companion columns; none of them is the
estimator. Offering a choice among them was a mistake — LMR publish one number per parameter and
this project follows it.

## Why the last 10,000-generation run failed, and what is different now

The v19.8.0 run on `base_fc` collapsed: acceptance 0.183 → 0.006 by generation 1,500, `dlp`
falling linearly in the generation index at −0.052/gen with R² = 0.996, and the population
variance rising monotonically by a factor of 4.6e+16 across the run.

**That run had no prior box.** With `θ = lb + (ub−lb)·σ(t)`, `Q` is asymptotically flat in every
coordinate — measured on this model, `Q` agrees to ten significant figures between `t₀+40` and
`t₀+80` in every coordinate tested, both directions. So the unbounded quasi-posterior has
infinite mass in all 23 directions: there is no density for the chain to converge to, and the
population diffuses without limit until acceptance dies. Every proposal-tuning attempt before
v20.0.0 was therefore addressing a symptom.

LMR bound the unconstrained parameter at ∓20 (`main_mpi.f90:146-147`) and reject out-of-box
candidates by setting the acceptance probability to zero (`mpi_mcmc_mod.f90:347`). v20.0.0 added
exactly that as `MCMC_T_BOX = 20.0`. **This is the one substantive difference between the failed
run and the configuration below.** On a surrogate calibrated to this model's measured anisotropy
the box takes the log-variance drift from +50.7 to −0.033 per 1000 generations; on this model it
is untested, and this run is the test.

## The configuration, and where each value comes from

| setting | value | source |
|---|---|---|
| `MCMC_N` | 95 | LMR `chain_count`, `main_mpi.f90:128` |
| `MCMC_GENS` | 10,000 | LMR `max_iteration`, `main_mpi.f90:129` — here a **cap**, not a target |
| `MCMC_BURN` | 0.9 | LMR Appendix C: last 1,000 of 10,000 pooled |
| `MCMC_CR` | 0.75 | LMR `main_mpi.f90:127` |
| `MCMC_DELTA` | 2 | LMR `main_mpi.f90:131` |
| `MCMC_B_ADD` | 1e-4 | LMR `shock_add_std`, `main_mpi.f90:137` |
| `MCMC_B_MULT` | 1e-2 | LMR `shock_mult_std`, `main_mpi.f90:136` |
| `MCMC_INIT` | `:at_seed` | LMR `one_population_kn = spread(initial_theta,2,N)`, `:268` — all chains at one point; their randomised alternative is commented out at `:41-43` |
| `MCMC_T_BOX` | 20.0 | LMR `prior_lower/upper_bound = ∓20`, `main_mpi.f90:146-147` |
| `MCMC_PRIOR` | `:flat_t` | LMR sample the transformed parameter with a flat prior on it |
| `MCMC_SPACE` | `:t` | as LMR; see the note on `:theta` below |
| `MCMC_OUTLIER_IQR` | 2.0 | LMR replace a chain whose mean log-posterior is below `Q1 − 2·IQR`, `mpi_mcmc_mod.f90:421` |
| `MCMC_OUTLIER_BURN_ONLY` | `false` | LMR apply it every generation, not only during burn-in |
| `MCMC_CHECK_EVERY` | 250 | **departure, deliberate.** LMR have no convergence test — theirs reads `! CEHCKING CONVERGENCE` / `! TO BE DONE!!!` (`mpi_mcmc_mod.f90:473-475`) — so they run a fixed budget. See *The stop* below |
| `MCMC_ACC_FLOOR` | 0.02 | **departure, deliberate.** LMR have no acceptance abort — acceptance is recorded in five categories and printed, never acted on. Zeroing it to match them was reversed: it fires only when the chain has stopped moving, and on the v19.8.0 acceptance trace (0.074 at g=500, 0.025 at 750, 0.012 at 1000) it ends a dead run at generation 1,250 — 1.3 h instead of 10.4. Cheap insurance on the resource that actually binds |
| `MCMC_PRINT_EVERY` | 250 | not LMR — reporting cadence only, cannot affect the chain |
| `MCMC_CHECKPOINT` | `true` | **not LMR.** Writes each new best point to the estimate bundle so a 10-hour run cannot lose its point estimate. It observes the chain and never feeds back into it, so the sampled sequence is identical either way |

Departures from LMR: `MCMC_PRINT_EVERY` and `MCMC_CHECKPOINT` (neither can change a draw), and
the sequential stop, which changes *when the run ends* and not what it samples.

## The stop, and why it is not a weakening

LMR run 10,000 generations because they have no convergence test, not because 10,000 is
required. 950,000 solves is 10.4 hours per window and 42 hours across four windows, which is a
cost this project cannot pay. A sequential stop is not a compromise on their procedure — it is
the same estimator, stopped once the reported numbers have stopped moving.

**The criterion is the right one for what is reported, and it needs no new code.** The estimator
is the mean and SD of the pooled draws, so the defensible stop is that the Monte Carlo error in
the mean is small relative to the reported SD (Flegal, Haran & Jones 2008). With `N` independent
chains, `MCSE_k = sd_c(chain means)/√N`, and since `MCSE/se ≈ √(B/W)/√N` while `R̂² ≈ 1 + B/W`:

| `R̂` | `B/W` | `MCSE/se` at N=95 | |
|---|---|---|---|
| 1.05 | 0.10 | 0.033 | pass |
| **1.10** | 0.21 | **0.047** | pass |
| 1.20 | 0.44 | 0.068 | fail |
| 2.37 | 4.62 | 0.221 | fail |
| **8.50** | 71.3 | **0.866** | fail |

`MCSE/se = 0.05` lands at `R̂ = 1.112`. Read the other way: the failed v19.8.0 run reached
`R̂ = 8.50`, i.e. `MCSE/se = 0.87` — the Monte Carlo error was 87% of the number being reported.
That is why its standard errors meant nothing, stated in the units of the thing reported rather
than as a convention violated.

**An earlier draft of this section claimed the shipped `MCMC_RHAT_MAX = 1.10` gate already *was*
this criterion. That was wrong twice over**, and the correction is the substance of v21.0.0.

First, `stop_rule` never gated on `R̂` as a *threshold*. It gated on `wr <= rhat_prev` — worst `R̂`
merely **not increasing** — because a threshold there had been measured as unsatisfiable at any
budget (worst `R̂` trends upward with budget on this model) and fired at 0 of 16 checkpoints. So
the number in the table was reported, not enforced.

Second, and more seriously, the no-worsening test is not a convergence criterion at all. A chain
can satisfy it together with the accepted-move and flat-drift conditions while its 95 members sit
in different regions — and then the pooled SD measures the between-chain spread rather than the
posterior width, so the run stops with a number that is not a standard error. It detects a chain
that has stopped *deteriorating*, which is a salvage test for a run that never converges: in the
working regime it is redundant, and in the collapse regime it is the only thing that fires.

**So the gate is now the quantity itself.** `mcse_ratio` (`mcmc_diagnostics.jl`) computes
`sd(per-chain means)/√N ÷ sd(pooled)`, maximised over non-exempt coordinates, and `MCMC_MCSE_TARGET
= 0.05` gates on it. No autocorrelation estimator is required because the chains are independent
replicates, which matters specifically here: a series that is 99.4% duplicates defeats a spectral
ESS while leaving this estimator correct.

`MCMC_MOVES_MIN` and `MCMC_DRIFT_FLAT` remain beside it as floors, not as the criterion.
`MOVES_MIN` was lowered 2300 → 230 (`100·d` → `10·d`): `100·d` sizes a stable *full* covariance,
which is not the deliverable, and at N = 95 reaching `mcse = 0.05` near generation 670 yields only
~640 accepted moves — so the old value would have refused a stop the accuracy gate had already
granted. `DRIFT_FLAT` prevents stopping mid-descent, when `Cov(chain)` would measure the
trajectory rather than the curvature.

**VALIDITY CONDITION, and it binds at the shipped initialisation.** Like `R̂`, this diagnostic
requires an over-dispersed start. At `MCMC_INIT = :at_seed` every chain begins at `θ̂`, so early on
the per-chain means agree for want of movement rather than for convergence. Measured at N = 24 it
reads 0.185 at the first check rather than ~0, so it does not fire spuriously there — but the
reading can fall before the chains have separated, and the guard is to require `R̂` to have peaked
and be falling before believing a small `mcse`. An over-dispersed start would remove the caveat;
`MCMC_INIT = :widths` exists for that and is **not** recommended yet (see below).

**The convergence stop keeps the run; the acceptance abort ends a dead one.** These are different
mechanisms and both are on. An abort discards the run's claim to be a posterior sample — the draws
and `theta_best` survive on disk — while a stop certifies it. Zeroing `MCMC_ACC_FLOOR` to match
LMR was reversed once the cost was priced: see the table row above.

## What the stop is worth

| stop at generation | solves | hours (10 threads) | retained draws | accepted moves at acc 0.2 |
|---|---|---|---|---|
| 500 | 47,500 | **0.5** | 4,750 | 950 |
| 1,000 | 95,000 | **1.0** | 9,500 | 1,900 |
| 1,500 | 142,500 | 1.6 | 14,250 | 2,850 |
| 3,000 | 285,000 | 3.1 | 28,500 | 5,700 |
| 10,000 (LMR's budget) | 950,000 | 10.4 | 95,000 | 19,000 |

A stable covariance in d = 23 needs roughly 230–2,300 accepted moves, so **generation 1,000
already clears the band eight times over.** The binding constraint has never been sample size —
it is equilibration, and equilibration is precisely what the stop measures. If the box works,
this run costs one to two hours per window rather than ten.

## Cost and what the retained window buys

950,000 solves at 0.395 s on 10 threads ≈ **10.4 hours**. The retained window is
`0.1 × 10,000 × 95 = 95,000` draws. Since a rejection is stored as a duplicate of the current
point, the information content is the number of *accepted* moves, and a stable covariance in
d = 23 needs roughly 230–2,300:

| acceptance in the retained window | accepted moves | vs the 230 floor |
|---|---|---|
| 0.183 | 17,385 | above |
| 0.020 | 1,900 | above |
| 0.006 | 570 | above |

**At N = 95 the window clears the floor even at the collapsed acceptance rate the failed run
reached.** The earlier run was killed at N = 100 on an extrapolation of `dlp` to generation
9,000; that extrapolation implied essentially zero accepted moves, and it may have been
pessimistic. Either way the box changes the dynamic it extrapolated, so the extrapolation no
longer applies.

## What to read while it runs

Three numbers in the `[demc] gen` line, in order of what they decide.

1. **`dlp`.** In the failed run it fell linearly and without curvature: −4.13 at generation 250,
   −21.03 at 500, −75.53 at 1,500. If the box works, `dlp` flattens instead of marching. This is
   the single decisive reading and it is legible by generation 500.
2. **`acc`.** It starts high (0.758 was measured at N = 40) because the population begins
   collapsed at one point and the step is therefore too short. It should fall toward and settle
   near 0.234, not sail through it.
3. **`box=`.** Absent or small means the box binds only on the flat tails, which is its purpose.
   Persistently large means ±20 is cutting into an identified coordinate and the width needs
   revisiting — that field exists to tell you.

## The `:theta` path, present but off

`MCMC_SPACE = :theta` samples the natural parameter with the economic box enforced by rejection,
giving `dθ/dt = 1` so a coordinate at a bound is reachable. It is **incomplete and must not be
switched on**: `b_add` is a single scalar added to every coordinate, which is harmless in `t`
where the logistic makes everything O(1) and destructive in `θ` where the coordinates span
6.3e-06 to 11.1; and several consumers of the chain vector still treat it as `t`. The motivation
is real — `b_S`'s `se(chain)` is orders of magnitude below the width over which the criterion
responds — but it is not LMR's procedure and it is not needed for the reported object.

## Initialisation: why `:at_seed` is shipped and `:widths` is not

LMR start every chain at one point: `one_population_kn = spread(initial_theta, 2, N)`
(`mpi_mcmc_mod.f90:268`). Line 265, commented out, is the alternative they tried and abandoned —
drawing from the prior box. Their design is internally consistent: a collapsed start, a fixed
budget, and no convergence test. One cannot take their start *and* add a diagnostic without
inheriting the caveat in the previous section, because every convergence diagnostic assumes an
over-dispersed start.

`MCMC_INIT = :widths` was built to remove that caveat, and it is implemented but **not
recommended**. What it does: per-coordinate half-widths from the criterion's own ΔQ = 1 brackets,
kept separate by direction (a coordinate resting on a floor has a bracketable width upward and
none downward), transformed into the sampled space **endpoint-to-endpoint** rather than through a
local derivative — `dθ/dt` for `b_S` is 6.25e-06 at the estimate while its width spans 6.4
`t`-units, so a local-derivative conversion is wrong by three orders of magnitude. The endpoint
transform is exact and imposes no normalisation: each coordinate disperses on the scale its own
criterion gives it, in its own region.

Measured at `MCMC_INIT_DISPERSE = 2.0`, N = 24, 60 generations, against `:at_seed`:

| | `:at_seed` | `:widths` (disperse 2.0) |
|---|---|---|
| `acc` at first check | 0.771 | **0.223** (the RGG optimum is 0.234) |
| `dlp` at first check | −0.05 | **−850.11** |
| `max logπ` | −244.49 | −245.28 (never improved on the seed) |
| `Q(θ̄)` | 488.49 | **1011.22** |

The step scale is exactly right and the population is in a terrible place. A calibration sweep
then showed the joint ΔQ overshooting the quadratic prediction `c²·d` by a **constant** factor of
~350–400 at every multiplier from 0.05 to 2.0. A constant ratio across two decades cannot be a
departure from quadratic (which would grow with `c`) nor the collinear ridges (whose cross terms
vanish in expectation under independent draws) — it is a scale error in the inputs, and it was
mine: `hit_infeasible_up/dn` was read as "this width is unusable", triggering a 1%-of-box
fallback that replaced `a_ℓ`'s measured 8.7e-04 with 0.079. The flag actually marks a bracket that
stopped at the feasibility boundary rather than at ΔQ = 1, which is a *tighter* scale bound, not
an invalid one. The fallback now honours the measured width, falls back to the other direction,
and only then to a fraction of the distance to the near bound.

**The recalibration after that fix was not completed** — it was stopped to free cores for a
production run. Until it is, `MCMC_INIT_DISPERSE` has no measured value and `:widths` should not
be used. `:at_seed` remains the default and is verified to reproduce v20.0.0 bit-for-bit.

## The DE scale: the one place this departs from LMR's algorithm

Everything above concerns *when the run ends*. This concerns *how it proposes*, and it is the
only change to the sampler's mechanics.

LMR hold `gamma = 2.38/sqrt(2*delta*n)` fixed, as in ter Braak (2006). That value is optimal
only when the population already sits at the target's scale, and DE-MC has no way to notice
when it does not: the proposal is `gamma*(X_r1 - X_r2)`, drawn from the population itself, so
a population that has drifted wider proposes proportionally longer steps forever. Nothing in
the algorithm references an absolute scale.

The v20.0.0 `base_fc` run is that failure. Acceptance was 0.193 at the first check — the scale
was right — then 0.027, 0.014, 0.009, and the run aborted at generation 1000. Feasibility is
not the cause: `fin ~ 0.74`, so 98.75% of *feasible* proposals failed the Metropolis test.
Measured against the criterion's own per-coordinate widths the population had reached a median
6.1x, spanning ~500x from 25x too wide on `λ_S` to 0.05x on `b_S` — a shape error, not a scale
error.

A scalar cannot correct a shape error and does not have to. DE-MC reshapes its own population
through **accepted moves**; at 0.9% acceptance there were ~60 per chain in 1000 generations,
far too few to reshape 23 dimensions. Restoring acceptance restarts the covariance adaptation
that fixes the shape.

`MCMC_GAMMA_ADAPT = true` therefore brakes `gamma` toward acceptance 0.234 every
`MCMC_GAMMA_EVERY = 25` generations, by `exp((eta/sqrt(k))*(acc - 0.234))` with
`MCMC_GAMMA_ETA = 2.0`. Three properties are load-bearing:

**One-sided.** The brake shortens and never lengthens. A symmetric controller was tried and
was worse: from the collapsed start acceptance is 0.739 at g=25 — high because the population
has not spread, not because the step is short — so it read "lengthen", raised `gamma` 65%, and
accelerated the expansion (matched arms at g=50: 0.370 braked against 0.196 symmetric). The
asymmetry follows from the measured dynamics: an accepted symmetric proposal adds its squared
step to the population variance and that step is proportional to the variance already there,
with no inward pull, so widening is intrinsic and needs no help. Overshoot is self-correcting
through the same mechanism, which is what makes one-sidedness safe rather than merely simple.

**Diminishing.** The step is `eta/sqrt(k)` in the count of *braking* events, so the adaptation
vanishes asymptotically and ergodicity holds with no freeze point (Roberts & Rosenthal 2007).
That matters here because `burn = burn_frac*gens = 9000` never arrives when the stop fires two
orders of magnitude earlier, so "adapt during burn-in then freeze" has no meaning.

**Calibrated.** `eta` and the cadence come from a closed-loop simulation that reproduces the
observed collapse under fixed `gamma` (acceptance -> 0.0001, step 3.34x optimal). By generation
2000: `eta` 1 / every 50 -> 0.065; 1 / 25 -> 0.138; **2 / 25 -> 0.200, step 1.07x optimal**;
3 / 25 -> 0.212, 1.04x.

`MCMC_GAMMA_ADAPT = false` restores LMR's fixed `gamma` exactly: `gamma_scale` initialises to
1.0 and is only ever multiplied, so with the brake off every proposal is bit-identical to
v20.0.0.

**What this does not fix.** The brake is verified on the aggregate step scale. That the
population's *shape* self-corrects once accepting resumes follows from DE-MC's covariance
adaptation but has not been measured on this model; the per-coordinate spread-to-width table
is the diagnostic to recompute against the first run under this version.

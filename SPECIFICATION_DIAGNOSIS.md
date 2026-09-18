# Why the chain does not work: a specification diagnosis

## The answer in one paragraph

The criterion is not hard to sample because the sampler is badly tuned or the solver is rough.
It is hard to sample because **three moments out of 28 carry 90.5% of it, and those three are
bound to each other by a steady-state accounting identity the model cannot satisfy.** The data
require a training inflow into skill about **1.7× larger** than the model can sustain at the
observed skilled stock, because the model's *outflow* from skill — the cross-market drain, which
is the paper's own headline mechanism — is too weak by that factor. Every parameter that raises
the inflow breaks either the stock or the skilled job-finding rate. The resulting criterion is a
canyon carved by misfit rather than by identification, and no proposal distribution samples a
canyon whose walls are 10 to 14 sampling standard errors high. Lise–Meghir–Robin never met this
problem for two independent reasons: their education margin does not exist (education is a
conditioning variable, estimated as two separate programs), and their criterion is weighted
roughly *n* times more loosely than this one. On the second point they were explicit — see
footnote 21 below.

## 1. What LMR actually did, from their code and their own footnote

**Their weighting is not the optimal one, and they say so.** Footnote 21, page 24, verbatim:

> "To avoid the problem similar to that pointed out by Altonji and Segal (1996) we decided
> against using the optimal weight matrix. **Indeed using it did not give sensible results.** In
> this case the chain converges to a stationary process where the variance is a consistent
> estimator of the inverse of the Hessian J⁻¹ (Theorem 1 of Chernozhukov and Hong, 2003) and the
> sandwich estimator (J⁻¹IJ⁻¹) has to be used to calculate standard errors appropriately. Because
> of the lack of precision in the numerical approximation of the gradient G(θ) = ∇θ m^MS(θ) (with
> I = G(θ)ᵀ W_N⁻¹ G(θ)), we report the variance of the MC chain (J⁻¹). This is still informative."

Three things follow. **They tried the optimal weight matrix and it failed for them too** — this
project has independently reproduced their failed experiment. **They knew the chain SD was the
wrong object** and reported it anyway because their numerical gradient was too imprecise to build
the sandwich. And the main text states the validity condition as a conditional — the chain gives
valid standard errors *"if the optimal weighting matrix is used"* — which the footnote then says
they do not meet.

**What is on their diagonal.** `se_ed1.raw` holds 241 numbers taking **13 distinct values, each
repeated exactly 20 times** — one scale per moment *type*, applied to all 20 elements of that
type's profile. It is a block normaliser, not a per-moment sampling error. `GLOBAL_TRUE_N_MATRIX`
(sample sizes) is allocated and read but never enters the objective, which divides by
`GLOBAL_TRUE_SD_MATRIX` alone. So **the sample size never enters their weights.**

| | LMR (ed1) | this project (base_fc) |
|---|---|---|
| moments | 241 | 28 active of 31 |
| distinct weight values | 13 | 28 |
| median \|scale / moment\| | **100.26%** | **0.77%** |
| does the weight shrink with *n*? | no | yes (it is a sampling variance) |

A sampling standard error is a dispersion measure divided by √n. So for the same underlying
moment the two criteria differ in scale by roughly a factor of *n*: **LMR are running the
Chernozhukov–Hong criterion divided by n, i.e. a tempered quasi-posterior.** That is why their
rough criterion costs them nothing — grid noise of a given physical size costs ~1/n as many
criterion units — and why their reported dispersion looks sensible while needing a sandwich.

**And they have no education margin.** The replication package contains
`estimation_ed1_no_growth` and `estimation_ed2_no_growth`: two separate estimations, one per
education group. There is no training decision, no skill stock, no drain out of skill, hence no
stock–flow constraint of the kind diagnosed below. Their moment module
(`compute_theoritical_moments_mod.f90`) contains **zero** threshold comparisons; the whole model
directory has about ten. This model has at least three moving thresholds on the ability grid —
the training frontier, the unskilled participation margin, and the cross-market drain
(`skilled.jl:248`, `sc.d[i,j] = (sc.U1[i] > U0j) ? 1.0 : 0.0`).

## 2. The misfit is concentrated, not global

At the current `base_fc` estimate, `Q = 480.880318`, `K = 28`, `d = 23`, so `Q/dof = 96.18`.
That number alone reads as a decisively rejected model. The decomposition says otherwise.

| moment | model | target | z (sampling SEs) | share of Q |
|---|---|---|---|---|
| **training_share** | 0.04205 | 0.07168 | **−14.3** | **42.3%** |
| **jfr_S** | 0.26464 | 0.32037 | **−11.5** | **27.6%** |
| **ltu_share_S** | 0.18843 | 0.23264 | **−9.9** | **20.6%** |
| overlap_SltU | 0.21613 | 0.20320 | +3.2 | 2.1% |
| *remaining 24* | | | median 0.72 | 7.4% |

Distribution of \|z\| over all 28: **9 within 0.5 SEs, 16 within 1, 21 within 2, 25 within 5.**
Median \|z\| = 0.718. The three named moments carry **90.5%** of the criterion and the other 25
carry 9.5%. This is not a model that fits nothing; it is a model that fails in one place.

## 3. The three failures are one identity

`skilled_share` is fitted almost exactly (+0.50 SEs) while `training_share` is 14.3 SEs short.
The stock is right and the flow is wrong. In steady state those are not independent: the inflow
into skill must equal the outflow from it. Under either denominator convention for the training
share,

| | implied outflow rate from skill |
|---|---|
| model (`T = 0.04205`, `S = 0.30196`) | 0.0972 (share-of-unskilled) / 0.1393 (share-of-population) |
| data (`T = 0.07168`, `S = 0.30164`) | 0.1660 / 0.2376 |
| **ratio the data demand** | **1.707× either way** |

**The data need the drain out of skill to be about 1.7 times stronger than the model delivers.**
That is why the flow cannot be raised: with the outflow fixed, raising the inflow over-fills the
skilled stock, and the stock is a well-measured moment.

## 4. Four sweeps, measured at the estimate, confirming no parameter closes it

| lever | training_share z | jfr_S z | ltu_share_S z | skilled_share z |
|---|---|---|---|---|
| `b_S` 1e-5 → 0.06 | −14.3 → −14.1 | −11.5 → −16.4 | **−9.9 → −3.3** | — |
| `μ_S` 0.244 → 0.32 | −14.3 → −13.6 | **−11.5 → +3.6** | −9.9 → −25.5 | — |
| `c` 11.14 → 9.0 | **−14.3 → −8.0** | −11.5 → −17.3 | **−9.9 → −1.8** | **+0.5 → +158.1** |
| `P_S` 2.34 → 3.1 | −14.3 → −10.3 | −11.5 → −13.1 | −9.9 → −7.9 | +0.5 → +100.2 |

Read the rows as instruments. `b_S` is the instrument for long-term unemployment and nothing
else: it works through the skilled reservation margin, which is **inactive at the estimate** —
`p*_S = 0` at all 120 abilities, `b_S` sitting at 0.002% of its box — and switching it on makes
some abilities reject *everything* (`p*_S` max jumps straight to 1.0), which costs `jfr_S` more
than it gains. `μ_S` is the instrument for `jfr_S` and fixes it (z → +3.6) at the price of
`ltu_share_S` (→ −25.5) and of `ur_S` and `theta_S`, both of which were fitted. `c` is the only
lever with real purchase on the training flow, and it is the one that exposes the identity:
halving the training-share miss and *fixing* `ltu_share_S` sends `skilled_share` to **+158 SEs**.

Two structural readings follow, both of which matter for the paper rather than the code.

**The skilled block has no dispersion in unemployment duration.** With `p*_S = 0` everywhere,
every skilled unemployed worker accepts every offer, so all of them share the identical hazard.
A single exponential cannot simultaneously deliver a high mean exit rate (`jfr_S`) and a fat
duration tail (`ltu_share_S`), and the data ask for both. This is the standard
unobserved-heterogeneity failure, and the existing device (`b_S`) resolves it in the wrong
direction because it works through *acceptance* rather than through *arrival*.

**The drain out of skill is the under-powered channel, and it is the contribution.** The paper's
pitch is that the boundary between the two markets moves in both directions — into skill through
training, out of it through obsolescence. The measurement says the outward leg is 1.7× too weak.
The mechanism that most needs strengthening is the one the paper is selling.

## 5. Why this makes the chain unsamplable, and why no amount of grid refinement helps

With `W = Diagonal(1/σ̂²_samp)`, one criterion unit is one sampling standard error of a moment.
The quasi-posterior's width is therefore set in units where the criterion's *numerical* noise is
large: `Q(θ̂)` moves 19.4 units between `Np_S` = 120 and 160, and up to 126 units across ability
grids. **The posterior is narrower than the noise floor of the function being sampled.** The
measured consequence, from the v21.0.1 run: between generation 1000 and 2000 the proposal step
fell 1.543× and acceptance rose only 0.185 → 0.202, where a smooth criterion of the same
curvature gives 0.390 — the step bought 8% of its due; and acceptance among feasible candidates
sits flat near 0.26 across a 1.5× step reduction, where a smooth target climbs toward 0.5.

Refinement attacks the noise and buys a factor of 2–4. Closing the gap needs 20–100. The leverage
is not there, which is the correct reading of "no amount of refining will ever fix that."

But note *why* the walls are so high: 90.5% of the criterion is three residuals of 10–14 SEs that
no parameter can reduce. **A specification fix that brings those three inside a few SEs lowers
the canyon walls by an order of magnitude and makes the criterion samplable without any of the
sampler machinery added since v19.8.0.** That is the sense in which the patching has been
treating a symptom.

## 6. Options, ranked, with what each costs and buys

**Option 1 — verify the two data targets are the objects the model computes.** Cheapest by far
and it must come first. The gap is a clean factor of **1.705** between model and target training
share, and no parameter closes it. A factor that stable is what a definitional mismatch looks
like: a flow measured over a different horizon than the model's period, a different denominator
(share of the unskilled vs of the population vs of the age-eligible), or an enrolment concept
that counts spells rather than transitions. If the target is 1.7× off for a definitional reason,
the entire diagnosis dissolves and nothing needs to be added to the model. **Not yet checked;
this is a data-construction question, not a modelling one.**

**Option 2 — power the outward leg of the boundary (recommended if Option 1 clears).** The
identity says the outflow from skill must roughly double. The defensible instrument is an
explicit skill-obsolescence hazard, which the model currently lacks as a free parameter: the only
outward route is the endogenous drain `d[i,j]`, which is a corner-to-corner indicator rather than
a graded rate. Against the design filter: it is monotone and interpretable, it has a moment that
identifies it (the `training_share`/`skilled_share` pair pins it through the identity above), and
it is the paper's own mechanism made explicit rather than a shape device bolted on to hit a
target. It is also citable as skill depreciation. **Caveat to state honestly:** the model's flow
balance has more states than the two-state identity used here, so 1.707 is the direction and
approximate size, not a calibration; the direct evidence is the `c` sweep's `skilled_share`
excursion to +158 SEs, which does not depend on that algebra.

**Option 3 — give the skilled market dispersion in the offer *arrival* rate, not the acceptance
rate.** This is what `jfr_S` and `ltu_share_S` jointly demand and what `b_S` cannot supply. An
ability-dependent skilled contact rate (employer ranking) makes high-ability skilled exit fast
and low-ability skilled wait, so the aggregate job-finding rate and the long-term-unemployment
share move *together* instead of trading off. It also makes the drain out of skill selective by
ability, which is precisely the COVID mechanism the paper needs — the low-`a_S` mass draining out
of skill. Cost: one slope parameter and a functional-form choice, which is a real shape-device
risk; it is justified only because `ltu_share_S` identifies it and ranking is an independently
meaningful economic object. Complementary to Option 2, not a substitute.

**Option 4 — do what LMR did: detune the weighting.** Replace the sampling variances with
scale normalisers, so the criterion becomes ~n times shallower. The chain then moves, the grid
roughness stops mattering, and the reported dispersion is the chain SD with LMR's footnote-21
caveat. This is publishable — it is exactly what appeared in *Review of Economic Dynamics* — and
it requires no model change. What it does **not** do is make the three-moment failure go away: it
makes it invisible in the criterion while leaving `training_share` 14 SEs from its target, which
a referee who computes z-scores will find. Recommended only as a fallback, and if taken, the
z-score table above belongs in the paper rather than being suppressed.

**Rejected.** Dropping `skilled_share` to free the training flow — it is the composition moment
that identifies `ρ_x`, so dropping it un-identifies the copula correlation. Dropping
`training_share` — it is the model's central object. Letting ability decay during non-employment
to strengthen the drain — it contradicts the maintained assumption adopted from Cohen–Johnston–
Lindner (2025) that ability is fixed through non-employment, and should not be reversed silently.

## 7. What this implies for the sampler work

If Option 1 or Option 2 lands, `Q_min/dof` should fall by close to an order of magnitude, the
three 10–14 SE residuals become a few SEs, and the criterion's walls come down to where its
numerical noise is a small fraction of the posterior width. At that point the LMR chain — fixed
`γ`, collapsed start, no convergence test — plus the box prior for propriety is very likely
sufficient, and the brake, the width-based initialisation and the tempering discussion can all be
deleted. The right order of work is specification first, sampler second; it has been the other way
round.

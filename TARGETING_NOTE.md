# What to target, and why a stationary model needs dynamic moments

## 1. The conceptual point

A stationary equilibrium means the cross-sectional densities are time-invariant. It does not
mean individuals are static: in steady state workers are still being hired, climbing, being
shocked and separating. The stationary density is the *fixed point* of that individual-level
process, not a substitute for it.

Put formally, the parameters are the **generator** of a Markov process on worker states, and
the cross-section is its **invariant measure** `π`. The map from generator to invariant measure
is **not injective** — many generators share an invariant measure. The elementary case is
`u = s/(s+f)`: infinitely many separation/finding pairs deliver the same unemployment rate.
Cross-sectional moments identify `π`; only path moments identify the generator.

The most useful way to see the loss: in a stationary ergodic environment the cross-section *is*
the time-average of one worker's career, so **the cross-section is that career with the
time-ordering integrated out.** Duration distributions, tenure profiles and post-transition wage
growth are precisely the discarded ordering, recovered. This is why `μ_S` and `λ_S` sit at
−0.958 in this project: a wide offer distribution with slow climbing and a narrow one with fast
climbing produce the *same* mixture over tenures, hence the same three static quantiles, and
differ only in the mixture's components.

LMR do exactly this. Their paper, page 9: the simulated counterparts are "constructed based on
**S simulated careers** from the model." A stationary model, matched on panel profiles.

## 2. What LMR actually target

`se_ed1.raw` decomposes exactly into **12 blocks of 20 plus one scalar** — 241 moments. Twelve
statistics, each resolved at twenty horizons, not 241 separate statistics. The value ranges
identify their kind: block 1 rises 0.724 → 0.943 (an employment profile), block 5 rises
5.368 → 5.653 (a log-wage profile), block 7 spans −0.007 → 0.072 (wage *growth*, hence the
negatives), block 9 sits at 0.277 → 0.357 (a dispersion profile). Their own text names the
horizons: "the mean wage for those one year out of unemployment," "wage growth `t` periods
following a job move."

The arithmetic that follows is the whole problem. **LMR: 16 parameters, 241 moments — 15 per
parameter. This project: 23 parameters, 28 moments — 1.2 per parameter.** And all 28 are
cross-sectional aggregates or static quantiles; the only duration object in the set is
`ltu_share_S`, a single number.

## 3. What is available here, and what each buys

### A. Unemployment-duration profile, by education — CPS `DURUNEMP`

**Available today with no new extraction.** `ltu_share_S` is already built from `DURUNEMP` and
collapsed to one binary at `≥ 27` weeks; the full duration variable is in
`cps_basic_clean.arrow` (26,774,011 rows, 1,116,270 unemployed person-months with a usable
duration). Changing the aggregation from one binary to a vector of bin shares is the entire
data-side cost.

Occupancy at a ten-bin split (`<2, 2-4, 5-8, 9-13, 14-18, 19-25, 26-38, 39-51, 52-77, 78+`
weeks), per window:

| window | unemployed person-months | smallest skilled cell | largest | skilled relative SE |
|---|---|---|---|---|
| base_fc | 229,404 | 1,248 | 7,703 | 1.00% – 2.73% |
| crisis_fc | 207,819 | 1,436 | 5,564 | 1.24% – 2.65% |
| base_covid | 170,217 | 1,214 | 7,512 | 1.01% – 2.78% |
| crisis_covid | 99,641 | 1,128 | 4,942 | 1.27% – 2.92% |

Twenty moments per window (10 bins × 2 education groups), every cell above 1,100 skilled
observations. Relative sampling errors are **1.0%–2.9% skilled and 0.4%–1.7% unskilled**,
against the current set's median of 0.77% and wage quantiles at 0.05%–0.08%. **These moments
enter loosely weighted** — they add identifying content without adding steep canyon walls,
which is the opposite of what the wage quantiles do.

**Caveats to carry into the construction.** The `<2wk` cell's stock density is *below* the
`2-4wk` cell's, which a survivor function cannot be, so short spells are under-represented by a
monthly survey and that bin should not be used naively. Durations heap at 26 and 52 weeks.
`DURUNEMP == 999` is the IPUMS NIU code and must stay excluded, as the existing pipeline
already does.

**What it identifies.** The exit hazard's *level* separately from its *duration dependence* —
i.e. arrival rates separately from acceptance behaviour. It also resolves the current
`jfr_S`/`ltu_share_S` tension, where two points on the same curve are pulling against each
other because nothing else pins the curve's shape.

**And it is a specification test the current model fails.** Backing the hazard out of the stock
(dropping the unusable first bin) gives a decline from `h ≈ 0.20` around week 5 to
`h ≈ 0.02–0.03` around week 80 — a fall of **5.2× to 17.6×** depending on the window. Your
skilled block has `p*_S = 0` at all 120 abilities, so every skilled unemployed worker shares one
exit rate and the model's survivor is a pure exponential with a **constant** hazard. Adding
these moments will therefore make the missing-heterogeneity problem bind, which is the correct
outcome, but it means an ability-dependent skilled arrival rate becomes necessary rather than
optional.

**It is also a crisis-classification instrument, which is the paper's own claim.** Skilled
shares, base → crisis:

| | `<2wk` | `52-77wk` |
|---|---|---|
| FC | 0.0668 → 0.0429 (0.64×) | 0.0727 → 0.1212 (**1.67×**) |
| COVID | 0.0560 → 0.0509 (0.91×) | 0.0745 → 0.0885 (**1.19×**) |

The financial crisis roughly doubled long-duration mass among the skilled; COVID barely moved
it. Two crises with the same direction of unemployment and different duration signatures — the
thesis of the paper, and currently compressed into one `ltu_share_S` number per window.

### B. Job-tenure profiles — SIPP `EJB<n>_JOBID` / `TJB<n>_MSUM`

The redesign panels (`pu2018`–`pu2023`) are keyed `SSUID, PNUM, MONTHCODE` with `RMESR` and up
to seven job slots per month, each carrying a **persistent job identifier** and monthly
earnings; the classic panels (2001, 2004, 2008) carry the equivalent via `EJBHRS`/`EPPPNUM`, and
`_sipp_merge!` already merges job spells across wave files. `_wchg_record_pair!` already links
consecutive months per person-job. **The person–month–job panel exists and the linking is done**
— tenure profiles are an aggregation change at the last step, not new data construction.

Two profiles follow, both by education:

- **Separation hazard by months of tenure.** This is the moment that separates `α_U` from `ξ_U`
  (currently −0.966 collinear): both produce separations, and only their duration dependence
  distinguishes a damage process from a constant exogenous rate. A single `sep_rate_U` cannot.
- **Mean log wage by months of tenure** — LMR's block 5. Identifies the offer-distribution shape
  (`a_Γ`, `b_Γ`) and the bargaining parameters against the arrival rates.

Sample support is ample for these: `neff_U` = 565,454 and `neff_S` = 75,093 consecutive-month
wage pairs in `base_fc`.

### C. Wage growth by months since an employer-to-employer move — SIPP

LMR's block 7, and the moment that separates `μ_S` from `λ_S` (−0.958 collinear): the ladder's
climb rate against the offer distribution. **You already compute the scalar version and are not
using it** — `ee_step_S` is in the skipped-moment list.

**This one is sample-constrained.** `sipp_ee_rates.csv` records `neff_ee` ≈ 202,990 moves but
`neff_step` ≈ **1,132** wage steps in `base_fc` (937 in `crisis_fc`, 1,927 in `base_covid`,
1,040 in `crisis_covid`). So four to six horizons, not twenty, and the relative SEs will be
several percent — again loosely weighted, which is fine.

### D. What is not available

- **Experience profiles over 20 years**, LMR's horizon. The NLSY follows cohorts for decades;
  SIPP panels run three to four years. Not replicable, and not needed: for identifying *rates*,
  twelve monthly points beat twenty annual ones.
- **A dynamic counterpart to the training margin.** Education transitions within a SIPP panel
  are too rare to build an "employment by months since training" profile. The NSC extract and
  CPS `SCHLCOLL` are the places to look, and this needs design work rather than aggregation.
- **J2J and JOLTS** are quarterly aggregate flows. They discipline levels, not profiles.

## 4. The model side: no career simulator is needed

LMR simulate careers because their model is discrete-time and their moments run over twenty
years. In a continuous-time stationary model every object above is analytic:

- The **duration survivor** is `S(t) = ∫ exp(−h(a)·t) dμ(a)`, with `μ` the ability distribution
  among the newly unemployed and `h(a)` the ability-specific exit hazard — a one-dimensional
  integral over the ability grid you already have. This is the cheapest addition by far and is
  the reason to do (A) first.
- **Tenure-conditional** objects (B, C) need the match-quality distribution *conditional on
  tenure*, which the current solve does not produce — it computes only the stationary
  cross-section. That is genuine new solver machinery (propagating the quality distribution
  forward from a match's start), and it is where the real implementation cost sits.

## 5. Recommended order

1. **Duration profile from `DURUNEMP` (A).** Cheapest on both sides, twenty moments per window,
   loosely weighted, and it converts the paper's crisis claim into a visible object. It will also
   reject the constant-hazard skilled block, which needs to happen.
2. **Reinstate `ee_step_S` as a short profile (C).** The data work is nearly done; the horizon is
   limited by 1,132 observations, so keep it to four to six bins.
3. **Tenure profiles (B).** Highest identification value for the collinear pairs, highest
   implementation cost, because the solver must carry tenure-conditional distributions.
4. **A training-margin profile (D).** Design work, deferred.

The ratio to aim at is not LMR's 15 moments per parameter — that horizon is not available in
these data. But (A) alone takes 28 moments to 48 and moves the ratio from 1.2 to 2.1, with the
new rows carrying duration information that is orthogonal to everything already in the set. That
is the difference that matters, not the count.

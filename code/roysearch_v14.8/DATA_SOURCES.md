# RoySearch — data sources


## Moment consistency (v14.7 audit)

All 31 moments were audited pairwise, data side against model side, and the
model-internal identities were verified numerically:

- `wage_premium` = `mean_wage_S` − `mean_wage_U` (exact, all four windows and the model)
- `p25 < p50 < p75` in both markets, all four windows
- `ur_total` = (1 − `skilled_share`)·`ur_U` + `skilled_share`·`ur_S`
- `sep_rate_j` + `wchg_rate_j` = ξ_j + λ_j (exact; confirms the two moments are a
  clean partition of the same λ_j hazard on identical employed weights, which is
  what identifies the ξ/λ split)

Two definitional inconsistencies were found and corrected:

- **`ltu_share_S` (fixed in v14.6, `solver/equilibrium.jl`).** The survivor hazard
  applied the d = 0 rate κ_S(1 − Γ_o(p*_S)) + ν to the whole skilled-unemployed
  stock, including workers who have crossed to the unskilled market and whose exit
  hazard is f_U + ν. `f_S` a few lines above already split the stock correctly, so
  the two moments were built from the same stock with different hazards. The hazard
  is now split to match `f_S`. At the base_covid estimate the crossing mass is
  exactly zero, so no moment moved; the fix matters in the crisis windows and in the
  transition, where the crossing region opens.
- **`theta_U` (fixed in v14.7, `data_processing/moments.jl`).** The unemployment
  count in θ_U's denominator was taken from the raw CPS frame, while `ur_U` uses the
  train-excluded labour force. `in_training` is an enrolment flag independent of
  `in_lf`, so an enrolled non-BA worker can be counted unemployed and appeared in
  θ_U but not in `ur_U`. Because the model derives the rate and the tightness from
  one θ_j through free entry, the two targets were mutually unsatisfiable. θ_U now
  uses the train-excluded frame; θ_S keeps the unrestricted frame, matching `ur_S`.

Documented asymmetries that are deliberate, not errors:

- `wchg_rate_j`: the data requires a detectable step (ε = $1/week, `SIPP_WCHG_EPS`,
  chosen to remove rounding without discarding real raises); the model counts every
  surviving redraw. The model quantity is therefore an upper bound on what the data
  procedure can detect.
- `ee_rate_S` comes from J2J/SIPP while `jfr_j`/`sep_rate_j` come from CPS
  month-pairs.
- `sep_rate_j` is E→U on both sides; demographic exit ν is separate in the model.

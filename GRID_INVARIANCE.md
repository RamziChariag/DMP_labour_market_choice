# Grid invariance: what is wrong, what fixes it, and what that buys

## The target, in the criterion's own units

`θ̂` should move by less than one standard error when the grid changes. Because
`W = Diagonal(1/σ̂²_samp)` with `σ̂` a *sampling* standard error, one criterion unit **is** one
sampling SE of a moment. For the two moments that carry the grid dependence that means

| moment | `σ̂ / level` | observed spread across `Nx ∈ {120,200,250,400}` | factor over target |
|---|---|---|---|
| `theta_U` | 2.61e-03 | 1.029e-02 | **3.9×** |
| `sep_rate_U` | 6.75e-03 | 1.566e-02 | 2.3× |

So the requirement is a ~4× reduction in `theta_U`'s grid spread. This is a sharper
specification than "refine until it stops moving", and it is the bar every fix below is
measured against.

**A corollary worth stating plainly:** the criterion is not unusually grid-sensitive, it is
unusually tightly weighted. `theta_U` varies by 1.029e-02 across ability grids — economically
nothing — against a bar of 2.61e-03, a ratio of 3.9. Two distinct numbers follow and should not
be confused:

- the **measured** contribution of `theta_U` to the `Q` spread across those grids is
  **10.861 units** (the max−min of `W_k(m_k − t_k)²`, from the moment decomposition);
- the deviation-equivalent figure, `(1.029e-02 / 2.61e-03)² ≈ 15.5` units, is what the spread
  would cost if it were a deviation *from target* rather than a spread *between grids*.

Either way the point holds: a sub-1% discretisation error in `theta_U` becomes of order ten `Q`
units by construction, which is why `ΔQ = 1` has never been a usable ruler on this criterion.

## What was ruled out, and how

Each of these was eliminated by measurement or arithmetic, not by argument.

| candidate | verdict | evidence |
|---|---|---|
| shock density pole at `p = δ_S` | **already fixed** | `skilled.jl:69-70` reads it off the exact Beta CDF. Refining `Np_S` 120→160 cut `Nx` scatter from 126 to 20 units |
| ability quadrature | **exact** | `build_ability_grid` uses Gauss–Jacobi against the true Beta weight `(1−t)^{b−1}(1+t)^{a−1}`; `E_ℓ[a] = dot(x, wa)` is exact for any `N` |
| copula marginals | **exact** | `build_copula` runs Sinkhorn to `max|row sums − wa_U| < 1e-14` |
| `p*_S`, `p^oj_S` cutoffs | **continuous** | `find_cutoff_from_j0` and `find_poj_from_diff_grid` bracket the root then interpolate linearly, so both are continuous in `θ`. Their *consumers* are a separate question: the moment layer read `p^oj` through hard indicators until v27.0.0, and one hard read of `Γ(p^oj)` remains |
| `Np_U` refinement | **falsified** | `theta_U` rel spread is 1.029e-02 at `Np_U` = 120, **240 and 480** — identical to four significant figures |
| solver tolerance | **ruled out** | `tol_outer_U = 1e-6` is 2.2e-06 relative to `theta_U ≈ 0.458`, three orders inside the 4.7e-03 spread |

## What is actually wrong: two hard boundaries in the ability dimension

Every defect this model has produced is the same defect — **a threshold that moves with `θ`,
evaluated on a fixed quadrature grid.** Three instances were fixed or were already continuous.
Two remain, both in `unskilled.jl`, both in the ability dimension, and both feed the three
moments carrying the residual (`theta_U`, `sep_rate_U`, `ur_U` = 69.6% of the `Nx` spread).

### Defect 1 — the unskilled participation margin `a†` (the larger one)

`p*_U(a) = 1.00000000` exactly for all abilities below some `a†`: these workers reject every
unskilled offer. The consumer is a hard switch at `unskilled.jl`, inside
`solve_stationary_unskilled!`:

```julia
f_hire = (pst < 1.0 - 1e-10) ? f : 0.0
usurv  = denom > 0.0 ? (δ + ν) / denom : 0.0      # denom = f_hire + δ + ν
```

`p*_U(a)` is itself continuous, but **`a†` is snapped to the nearest ability node**, so the mass
at the corner is counted rather than integrated.

| `Nx` | 120 | 200 | 250 | 400 |
|---|---|---|---|---|
| abilities at `p*_U = 1` | 20 | 33 | 41 | 66 |
| node-counted corner mass | 0.13657 | 0.13353 | 0.13182 | 0.13323 |
| exact mass `F_ℓ(a†)` | 0.13002 | 0.12960 | 0.12869 | 0.13125 |
| node − exact | **+6.55e-03** | +3.93e-03 | +3.14e-03 | +1.98e-03 |

**13.2–13.7% of population mass sits at this corner**, the node count is biased by
**6.55e-03 at the shipped grid** (5% of the corner mass) with the bias decaying as `O(1/N)` —
the signature of a half-cell over-count — and the mass varies **4.741e-03** across grids.
`theta_U`'s absolute spread over the same grids is **4.7e-03**. Those agree to one significant
figure, which is why this is judged the dominant channel.

**Fix.** `p*_U(a)` is monotone decreasing in ability, so the corner set is the interval
`[0, a†]`. Locate `a†` by the same linear root interpolation the `p`-grid cutoffs already use —
applied in the ability dimension, where it is currently missing — then take the corner mass as
`F_ℓ(a†)` from the exact Beta CDF and split the ability aggregates at `a†` rather than at a node.

**Measured effect of interpolating `a†` alone: 4.741e-03 → 2.561e-03, a 1.9× reduction.** The
residual is that `a†` itself still moves 1.251e-03, because `p*_U` is evaluated only at nodes and
there is a feedback loop (`a†` → aggregates → `θ_U` → `p*_U(a)` → `a†`).

### Defect 2 — the training frontier `τT`

```julia
uc.τT[i, j] = (Utr_j >= uc.Usearch[i]) ? 1.0 : 0.0
```

A hard `0/1` on the `Nx × Nx` ability grid; all composition mass is `Σ τ·W2`, i.e. quadrature on
an indicator — first order, and discontinuous in `θ`.

Measured: **4,971 training cells of 14,400** at `Nx = 120`, with **2.18% of population mass in
cells the frontier cuts through**, decaying `O(1/N)` (0.0262 at `Nx=100` → 0.0109 at `Nx=240`).
Total trained mass varies only 0.220% because the mis-assignments largely cancel — but
cancellation is not exactness and does not hold for functionals that weight the boundary cells
differently, which is where `theta_U` picks it up.

**Verified by falsification probe** (scratch tree, cell-covered-fraction `τ`):

| | hard `τ` | soft `τ` |
|---|---|---|
| `theta_U` rel spread | 1.029e-02 | **7.22e-03** (−30%) |
| `sep_rate_U` rel spread | 1.566e-02 | 1.14e-02 (−27%) |
| `Q` range across `Nx` | 20.38 | 18.01 |
| `training_share` | 3.5% of spread | out of the top 9 |

**Fix.** `Utr(a_S)` is monotone increasing (verified: 119 increments up, 0 down) and `Usearch(a_U)`
monotone increasing, with the frontier interior at 90 of 120 `a_U` nodes. So for each `a_U` there
is a unique critical `a_S*(a_U)`, and for the Gaussian skill copula the conditional mass is closed
form. In rank space `ζ = Φ⁻¹(F_ℓ(a))`, the pair is bivariate normal with correlation `ρ_x`, so
`ζ_S | ζ_U ~ N(ρ_x ζ_U, 1−ρ_x²)` and

```
P(a_S ≥ a_S*(a_U) | a_U) = 1 − Φ( (Φ⁻¹(F_ℓ(a_S*)) − ρ_x ζ_U) / √(1−ρ_x²) )
```

One normal CDF per `a_U` node. **The `a_S` quadrature disappears entirely.** Both branches of
`solve_stationary_unskilled!` are linear in `W2` with rates depending only on `a_U`, so the
substitution is exact.

**THREE call sites must change together, or the fix is silently void.** This was learned the hard
way: the first probe returned bit-identical output because the producer was patched and the
consumer was not.

1. `unskilled.jl` — the producer writes a fraction (or the conditional probability) rather than `0/1`.
2. `solve_stationary_unskilled!` — `if τ[i,j] > 0.5` **re-hardens the fraction**. It must blend:
   `u_out = W2·usurv·(1−w)`, `t_out = ν·W2/(φ+ν)·w`.
3. `smm_objective` — the degeneracy guard `all(iszero, τv) || all(isone, τv)` assumes `0/1` and
   must become tolerance-based.

## What the two fixes will and will not achieve

**Will:** together roughly 2–3× on `theta_U`'s grid spread (1.9× measured for `a†` alone, 30%
for the frontier proxy — and the exact conditional-CDF version should beat that proxy, since the
cell fraction is still `O(1/N)` while the closed form is exact). The bar needs 4×, so **these two
fixes alone are unlikely to close it**; the remainder needs either a finer base grid or a
higher-order treatment of `a†`. The right sequence is: implement both, re-measure, then choose the
base grid empirically against the 2.61e-03 bar.

**Will, and this may matter more:** both fixes make the criterion **smooth in `θ`**, not merely
more accurate. Every pathology recorded against this model — the staircase plateaus, the limit
cycles in the skilled block, the `b_S` profile's 0.0995 step, the ruggedness that has defeated
every optimiser and sampler — is consistent with boundaries snapping between grid nodes as `θ`
moves. A smooth criterion is a precondition for a chain that converges, and it is the reason to do
this before spending another 10 hours of compute on sampling.

**Will not:** repair the outright solve failures. `Nx ∈ {150, 300}` cannot evaluate `θ̂` even with
`Np_S` refined, and `Nx ∈ {160, 180}` fail at `Np_S = 120`. That is a fixed-point robustness
defect with no established mechanism, and nothing about either boundary fix addresses it. It needs
its own diagnosis.

## A note on what these fixes give the paper

Both boundaries are objects the paper is about, and neither can currently be drawn.

`a_S*(a_U)` is the critical skilled ability at which a worker of unskilled ability `a_U` is
indifferent between training and searching — the training frontier, presently implicit in a `0/1`
matrix. `a†` is the ability below which no unskilled job is acceptable — the participation margin.
Together they partition the `(a_U, a_S)` ability space into search, train, and non-participation
regions, and the claim that the boundary between the two markets moves *both ways* in a crisis is
a statement about how those two curves shift between windows. Making them explicit is what turns
a numerical repair into the paper's central figure.

## Caveats on the measurements above

All `Nx` scans hold `Np_U` at its shipped value (justified: `Np_U` was falsified as a channel at
120/240/480) and `Np_S` at 160 unless stated. All are at the single point `θ̂` from
`estimate_base_fc_diagonalW.jls` on `base_fc`; none of the grid dependence has been measured at a
second point in the parameter space, so the *magnitudes* here are specific to `θ̂` even though the
mechanisms are not. The soft-`τ` figures come from a first-order cell-fraction proxy in a scratch
tree, not from the analytic fix, and are therefore a lower bound on what the analytic fix delivers.

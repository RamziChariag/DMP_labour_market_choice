# γ_S — skilled ability gradient (comparative advantage)

Search the tree for `GAMMA_S` to see every edit. **Not run here (no Julia).** Do one quick `model_main` solve before the overnight estimation to confirm it compiles.

## What changed

Skilled production went from `exp(A)·P_S·exp(x)·p` to **`exp(A)·P_S·exp(γ_S·x)·p`**. Unskilled is **unchanged** (`exp(A)·P_U·exp(x)·p`, i.e. slope 1). So the productivity ratio is now

> `π_S/π_U = (P_S/P_U)·exp((γ_S−1)·x)` — increasing in x when `γ_S > 1`.

That makes unskilled comparatively more productive at low x and skilled at high x (single crossing), which is the separation lever. `γ_S = 1` recovers the old exp(x) model exactly (no comparative advantage), so this is a strict generalization.

## Edits (all tagged `GAMMA_S`)

- `solver/params.jl` — new field `γ_S::Float64 = 1.0` in `SkilledParams` (default 1.0 ⇒ backward-compatible).
- `solver/grids.jl` — `PS_of_x(x, P_S, A=1.0, γ_S=1.0) = A·P_S·exp(γ_S·x)` (γ_S is the **4th** arg; A stays 3rd so old 3-arg calls still compile at γ_S=1). `mean_output_S` now uses `E_ℓ[exp(γ_S·x)]`.
- `solver/equilibrium.jl`, `solver/skilled.jl` (both surplus functions), `solver/solver.jl` (p\* seed), `transition/transition_solver.jl` — all skilled production call sites pass the estimated `γ_S`.
- `smm/smm_params.jl` — `ParamSpec(:skl, :γ_S, 1.0, 6.0, 2.0)`; added to both `SkilledParams` build sites, the field map, and `REGIME_SPECIFIC_PARAMS`.
- `solver/model_main.jl` — added `γ_S = 1.0` to the hand-pasted `skl_par` block (see caveat 1).

## Estimation setup

- **γ_S is free** (not in `FIX_PARAMS`). Free-parameter count goes 20 → 21.
- **Bounds `[1.0, 6.0]`, init 2.0.** `lb = 1` pins `γ_S ≥` the unskilled slope, so the single-crossing direction is guaranteed (and `γ_S = 1` is the no-comparative-advantage boundary). `ub = 6` is generous: with `P_U/P_S ≈ 4`, the crossing sits interior only if `γ_S − 1 > ln(P_U/P_S) ≈ 1.4`, so give it room. If `γ_S` pins at 6, widen the ub.
- **Identification:** `γ_S` loads primarily on `wage_premium` (steeper skilled → high-x skilled earn more → premium up), which `α_U`, `Γ`, `β` barely touch — so it shouldn't steal identification. Verify with `local_elasticity(:γ_S)` in the sensitivity notebook: it should move `wage_premium` strongly and the rates little.

## What to watch in the result

If comparative advantage is the missing piece: `wage_premium` rises toward 0.385, `overlap_SltU` improves, `skilled_share`/`training_share` stop overshooting, and `a_ℓ`/`b_ℓ` come off the (8, ~0.15) spike **while still free**. Note `P_U > P_S` at the floor is now the *good* kind (comparative-advantage floor); the premium comes from sorting, not from the levels.

## Caveats

1. **`model_main` hand-pasted block:** the `skl_par` in `model_main.jl` now has a `γ_S` field, currently `1.0` (matches the exp(x) estimate it holds). **After the overnight run, paste the estimated `γ_S` there** or the single-run inspection will use 1.0.
2. **Two cosmetic plot calls** (`plots.jl:507`, `single_run_plots.jl:507`) are still 2-arg `PS_of_x(x, PS)` → they draw the `P_S(x)` diagnostic curve at γ_S=1. They do **not** affect estimation or moments; fix later by threading γ_S into `fig_skilled_employment_by_PS` if you want that figure exact.
3. If `γ_S` unexpectedly rails to 1.0 (no comparative advantage chosen), that's a signal the spike isn't a comparative-advantage problem — revisit with the sensitivity table.

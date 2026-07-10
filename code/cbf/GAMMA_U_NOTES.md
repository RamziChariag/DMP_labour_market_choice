# γ_U — flat unskilled (the Day-1 gate)

Search the tree for `GAMMA_U` to see every edit. **Not run here (no Julia).** Do one `model_main` solve before the estimation to confirm it compiles and prints `γ_U=0.00000`.

## The one-line summary

Unskilled production went from `exp(A)·P_U·exp(x)·p` to **`exp(A)·P_U·exp(γ_U·x)·p`**, and `γ_U` ships **fixed at 0**. `γ_U = 0 ⇒ exp(0)=1 ⇒ unskilled output is flat in x` (just `P_U·p`). Skilled stays `exp(A)·P_S·exp(γ_S·x)·p` with `γ_S` free. So the productivity ratio is now

> `π_S/π_U = (P_S/P_U)·exp(γ_S·x − γ_U·x) = (P_S/P_U)·exp(γ_S·x)`   (at γ_U=0)

— strictly increasing in x. **That is the maximal comparative-advantage configuration**: skilled rises with ability, unskilled does not, so the crossing is guaranteed and low-x workers have nowhere they're "wasted." `γ_U = 1` recovers the old exp(x) unskilled exactly, so this is a strict generalization pinned at one end.

## Why this is the right gate for the a_ℓ rail

The γ_S run relaxed `b_ℓ` but `a_ℓ` still railed to 8 — the estimator was still deleting low-x workers. Reason: with unskilled = `exp(x)`, raising `P_U` to make the low-x floor viable **also** inflates high-x unskilled (`P_U·exp(x)`), which cannibalizes skilled and kills the premium — so the estimator can't use `P_U` freely and instead removes low-x mass by pushing `a_ℓ` up.

Flattening unskilled **decouples** "lift the unskilled floor" from "make high-x unskilled too attractive." With `γ_U=0`, raising `P_U` lifts the whole unskilled schedule (low-x included) **without** making high-x unskilled compete with skilled. If the a_ℓ rail is a comparative-advantage problem, this is the sharpest test of it.

## What passes / fails the gate

**Pass:** `a_ℓ` comes off 8 while free (lands interior), `training_share`/`skilled_share` stop overshooting, `wage_premium` holds near 0.385, and the transition rates stay matched. Two healthy segmented markets.

**Fail:** `a_ℓ` still rails even with unskilled dead-flat ⇒ the low-x problem is **not** comparative advantage — it's the wage *level/floor* (revisit `b_U`, `c`, or the σ_w measurement channel), or it's a variance problem that only the Day-2 moments can fix.

## Edits (all tagged `GAMMA_U`)

- `solver/params.jl` — new field `γ_U::Float64 = 0.0` in `UnskilledParams` (**default 0 = flat**, the shipped gate). Explicit `γ_U = 0.0` in the default `Model()` constructor too.
- `solver/unskilled.jl` — `solve_unskilled_surplus_on_grid!` takes `γ_U` as an 11th positional arg; both surplus terms and the `p*` update use `exp(γ_U·x)`. Both in-file callers pass it.
- `solver/equilibrium.jl` — `γ_U = up.γ_U` in scope; surplus caller passes it; unskilled wage surface uses `exp(γ_U·x)`.
- `solver/grids.jl` — `mean_output_U` now integrates `E_ℓ[exp(γ_U·x)]` (so vacancy-cost scaling tracks the flat map). `prod_map` is now dead code (kept only as the definition; no live caller).
- `solver/solver.jl` — unskilled `p*` seed uses `exp(γ_U·x)`.
- `transition/transition_solver.jl` — unskilled average-wage site uses `exp(γ_U·x)`.
- `solver/model_main.jl` — `γ_U = 0.0` in the hand-pasted `unsk_par`; the param printout now shows **both** `γ_U` and `γ_S`.
- `smm/smm_params.jl` — `γ_U` resolved via `_get(:γ_U, :unsk, 0.0)` in both build passes and the field map. A **commented** `ParamSpec(:unsk, :γ_U, 0.0, 1.0, 0.0)` sits ready to free it.

## Estimation setup (this is the Day-1 config — nothing else to change)

- **γ_U fixed at 0** — it's not in the free set, so `_get` returns 0.0. No `FIX_PARAMS` entry needed.
- **γ_S free** `[1.0, 6.0]`, init 2.0 — unchanged from the γ_S run.
- **Variance moments skipped** — `SKIP_MOMENTS` already lists `emp_var_U, emp_cm3_U, emp_var_S, emp_cm3_S` (plus `ur_total, ee_rate_S`). This is the Day-1 "rates + shares + levels, no dispersion" gate.
- **η, β pinned** (`unsk_eta/skl_eta = 0.5`, `unsk_bet/skl_bet` fixed) — unchanged.
- Free-parameter count is **the same as the γ_S run** (γ_U is fixed, not added to the free set).

## Freeing γ_U later (Day-2, only alongside variance)

Once the gate passes and you re-add the variance moments:

1. Uncomment `ParamSpec(:unsk, :γ_U, 0.0000, 1.0000, 0.0000, …)` in `smm_params.jl`.
2. Uncomment `(:unsk, :γ_U)` in `REGIME_SPECIFIC_PARAMS`.

**Do not free γ_U without the within-unskilled variance moments.** `γ_U` and `α_U` both move unskilled wage dispersion; with no variance target they are not separately identified and one will absorb the other. Bounds `[0, 1]`: 0 = flat, 1 = exp(x); it should never want > 1 (that would re-introduce the cannibalization the gate is removing).

## Caveat

If you later paste estimated params into `model_main.jl` for single-run inspection, remember to set `γ_U` there too (it now prints, so you'll see if it's stale).

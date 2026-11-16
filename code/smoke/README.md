# Smoke tests

Spec-construction tests for a crisis window, and a transition run. The three spec tests
run the driver's own spec construction: `slice_spec.jl` writes a copy of
`smm/smm_main.jl` truncated just before §9 (`smm/main_spec_only.jl`), so everything up to
and including `build_smm_spec` runs verbatim and no optimisation starts.

    bash code/smoke/run_smoke.sh warmstart   # ability parameters pinned; first solve is the warm start
    bash code/smoke/run_smoke.sh clusters    # the baseline optimum reaches SA as a start
    bash code/smoke/run_smoke.sh guard       # the accounting guard fires on a spec with ρ_x unaccounted
    julia --project=. -t 6 code/smoke/smoke_transition.jl

Set `JULIA=/path/to/julia` if `julia` is not on the PATH.

## Requirements

The runner sets `ROYSEARCH_WINDOW=crisis_covid` and `ROYSEARCH_INIT_MODE` to the test's
mode. Needs `data/derived/` and the baseline estimate the crisis window loads
(`output/estimates/estimate_baseline_diagonalW_smm.jls` under the default
`CRISIS_BASE = :smm`). Test 4 starts from the base point only when no crisis estimate is
on disk; otherwise from whichever stored candidate scores lowest.

The runner changes only run settings in the sliced copy: a 40-node solve grid, and for
`clusters` a 32-point Sobol sample (a power of two: Owen scrambling requires it). The
rewrite matches `smm_main.jl`'s exact alignment of those lines; if it no longer matches,
the test runs at the full grid and bank size.

**Run `clusters` in a scratch copy of the repository.** It writes its bank to
`output/estimates/candidates_crisis_covid_diagonalW_smm.jls`, and the cache check does not
compare `n_sample`, so a later estimation would adopt the 32-point bank. `code/paths.jl`
resolves every path from the copy's own root.

## What each test asserts

`warmstart`
1. every `ABILITY_PARAMS` entry is in `spec.fixed`, at the baseline value, under the key
   `unpack_θ` reads back;
2. ρ_x reaching `CommonParams` is the baseline estimate;
3. `assert_all_params_accounted` passes;
4. with no seed bank, `_run_sa`'s first evaluated θ is exactly `pack_θ(spec)`, the warm
   start; its Q matches the objective there to `rtol = 1e-10`.

`clusters`
5. the baseline optimum is in the bank, alone in its cluster;
6. `_sa_starts_from_bank` emits it as a start, bit-identical.

`guard`
7. rebuilding the spec with ρ_x deleted from the pinned block makes
   `assert_all_params_accounted` raise and name `common:ρ_x`.

`smoke_transition.jl` solves both steady states and a capped transition path, and requires
every reported series to be finite (not converged).

## Generated files

The runner leaves `smm/main_spec_only.jl` and `smm/run_smoke_<test>.jl` behind; both are
rewritten on each run and neither is loaded by an entry point.

# Crisis-window spec smoke tests

Regression tests for the two v14.9 fixes (see `../CHANGELOG.md`). They exercise
the driver's real spec-construction path rather than a re-implementation:
`slice_spec.jl` writes a copy of `smm/smm_main.jl` truncated just before §9
(`smm/main_spec_only.jl`), so everything up to and including `build_smm_spec`
runs verbatim and no optimisation starts.

    smoke/run_smoke.sh warmstart   # DEFECT 1 — ability distribution pinned; first solve is the baseline
    smoke/run_smoke.sh clusters    # DEFECT 2 — baseline optimum reaches SA as a start
    smoke/run_smoke.sh guard       # the accounting guard fires on a v14.8-shaped fixed block

Set `JULIA=/path/to/julia` if `julia` is not on the PATH.

## Requirements

`smm_main.jl` with its shipped `WEIGHTING = :diagonal` and `USE_AS_SEED = true`. The
runner sets `ROYSEARCH_WINDOW=crisis_covid` and `ROYSEARCH_INIT_MODE` to the test's
mode. Needs `data/derived/` and the paired baseline bundle
`output/estimates/estimate_base_covid_diagonalW_smm.jls` written by the current major
version (`CRISIS_BASE = :smm`, its default). Test 4 claims the base point only when no
crisis estimate is on disk; otherwise the warm start is whichever candidate scores lowest.

The runner reconfigures only run settings — the coarse 40³ solve grid, and for
`clusters` a 32-point Sobol sample (the size must stay a power of two: Owen
scrambling requires it).

**Run `clusters` in a scratch copy of the repository, not in the working tree.** The
candidate layer writes its bank to `output/estimates/candidates_crisis_covid_diagonalW_smm.jls`,
and the cache check does not compare `n_sample`, so a later estimation would accept the
32-point smoke bank as its own. `code/paths.jl` resolves every path from the copy's own
root, so a copy is fully isolated.

## What each test asserts

`warmstart`
1. every `ABILITY_PARAMS` entry is in `spec.fixed`, at the baseline value, under
   the key `unpack_θ` reads back;
2. ρ_x reaching `CommonParams` is the baseline estimate;
3. `assert_all_params_accounted` passes;
4. with no seed bank, `_run_sa`'s first evaluated θ is exactly `pack_θ(spec)` —
   the warm start, which is the base point when no crisis estimate is on disk. Q is compared at `rtol = 1e-10`, since the solver
   reduces over threads and re-evaluating one θ can differ in the last bits.

`clusters`
5. the baseline optimum is in the bank, alone in its cluster;
6. `_sa_starts_from_bank` emits it as a start, bit-identical.

`guard`
7. rebuilding the spec with ρ_x deleted from the pinned block — the v14.8 state
   — makes `assert_all_params_accounted` raise and name `common:ρ_x`.

## Housekeeping

The runner leaves `smm/main_spec_only.jl` and `smm/run_smoke_<test>.jl` behind
as generated files; both are rewritten on each run and neither is imported by
the estimation entry point.

# Crisis-window spec smoke tests

Regression tests for the two v14.9 fixes (see `../CHANGELOG.md`). They exercise
the driver's real spec-construction path rather than a re-implementation:
`slice_spec.jl` writes a copy of `smm/smm_main.jl` truncated just before §9
(`smm/main_spec_only.jl`), so everything up to and including `build_smm_spec`
runs verbatim and no optimisation starts.

    smoke/run_smoke.sh warmstart   # DEFECT 1 — deep block pinned; first solve is the baseline
    smoke/run_smoke.sh clusters    # DEFECT 2 — baseline optimum reaches SA as a start
    smoke/run_smoke.sh guard       # the accounting guard fires on a v14.8-shaped fixed block

Set `JULIA=/path/to/julia` if `julia` is not on the PATH.

## Requirements

`smm_main.jl` as shipped: `WINDOW = :crisis_covid`, `W_COND_TARGET = 0.0`,
`INIT_MODE = :warmstart`, `USE_AS_SEED = true`. Needs `data/derived/` and the
paired baseline bundle `output/smm/smm_result_base_covid_diagonalW.jls`.

The runner reconfigures only run settings — the coarse 40³ grid, and for
`clusters` a 32-point Sobol sample (the size must stay a power of two: Owen
scrambling requires it).

## What each test asserts

`warmstart`
1. every `DEEP_PARAMS` entry is in `spec.fixed`, at the baseline value, under
   the key `unpack_θ` reads back;
2. ρ_x reaching `CommonParams` is the baseline estimate, not the −0.55 fallback;
3. `assert_all_params_accounted` passes;
4. with no seed bank, `_run_sa`'s first evaluated θ is exactly `pack_θ(spec)` —
   the baseline vector. Q is compared at `rtol = 1e-10`, since the solver
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

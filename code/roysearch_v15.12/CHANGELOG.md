# RoySearch — changelog

## v14.9 — crisis-window deep-block correctness

Two defects in the crisis re-estimation path, both in how a crisis window
inherits from its paired baseline. Estimates from crisis windows produced by
v14.8 or earlier are affected and must be re-run; baseline windows are not.

### ρ_x was silently unpinned in crisis windows

`smm_main.jl` built the crisis window's pinned block as a hand-written
NamedTuple listing five deep parameters — a_ℓ, b_ℓ, bU, bT, bS — while
`DEEP_PARAMS` in `smm_params.jl` has six. ρ_x, the Gaussian-copula ability
correlation, was therefore in neither `spec.free` (the free loop keeps only
`REGIME_SPECIFIC_PARAMS`) nor `spec.fixed`, so `unpack_θ` fell through to its
hard-coded −0.55 fallback. Nothing errored and the run header did not mention
it: both crisis windows were estimated at ρ_x = −0.55 instead of the baseline
estimate.

- The deep block is now assembled by iterating `DEEP_PARAMS_ORDERED`, so a
  parameter added to `DEEP_PARAMS` cannot be missed.
- `fixed_key(block, name)` states the key convention `unpack_θ` reads back —
  bare for a name unique to one block, block-qualified for a shared one — in
  one place instead of at each call site.
- `assert_all_params_accounted(spec)` runs after the spec is built, in BOTH
  branches, and raises naming any (block, name) that is neither estimated nor
  pinned. This is what makes the failure mode loud rather than silent.
- The run header prints the deep block by iterating the same list, so it cannot
  disagree with what was pinned.

### :clusters never evaluated the baseline optimum in a crisis window

Under `INIT_MODE = :clusters` the SA starts came from the Sobol→hclust bank
plus, optionally, `INCLUDE_PREV_OPTIMUM` — which reads the CRISIS window's own
prior optimum, not the baseline's. The Sobol layer samples the box blind, so a
cold cluster search on a crisis window never evaluated the baseline vector,
the one point with a strong prior claim on being near the answer.

The crisis branch now carries the baseline optimum in the crisis spec's own
free-parameter ordering (mapped by (block, name), since the crisis free set is
a subset of the baseline's), and §8b injects it into the seed bank under its
own cluster label so `_sa_starts_from_bank` — which emits the best-Q member of
each cluster — is guaranteed to emit it. The injection is in-memory; the
candidate cache on disk stays a pure product of the Sobol layer.

`USE_AS_SEED` semantics are unchanged: true uses the entered init verbatim,
false perturbs it by `SEED_PERTURB_FRAC`. Under `:warmstart` with
`USE_AS_SEED = true` and no seed bank, `_run_sa` falls back to
`[pack_θ(spec)]`, so the first solve is exactly the baseline vector.

### Compatibility

No struct gained a field, so previously serialised `SMMResult` / `SMMSpec`
bundles still deserialise. `pack_θ` gained a method over a `Vector{ParamSpec}`;
the `SMMSpec` method is unchanged. Weighting, `SKIP_MOMENTS`, and the SA / DE /
NM tuning defaults are untouched.

### Verification

`smoke/run_smoke.sh warmstart|clusters|guard` builds a copy of `smm_main.jl`
truncated just before §9 — so the real spec-construction path runs verbatim —
and checks the deep block against the baseline bundle, the injected cluster,
and that the guard fires on a reconstructed v14.8 fixed block. See
`smoke/README.md`.

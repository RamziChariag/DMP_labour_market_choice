# Smoke test — crisis_covid spec construction under INIT_MODE = :clusters.
#
# Checks that the baseline optimum reaches the optimiser as a start:
#   1. it is present in the seed bank, in a cluster of its own;
#   2. `_sa_starts_from_bank` therefore emits it (one start per cluster);
#   3. the emitted start is the baseline vector to the last bit.
#
# The sliced driver runs with a small cand_n_sample so the Sobol layer is cheap;
# the injection path under test does not depend on the bank's size.

include(joinpath(@__DIR__, "main_spec_only.jl"))

const RULE = "="^70

θ_base = pack_θ([
    ParamSpec(ps.block, ps.name, ps.lb, ps.ub,
              clamp(baseline_optimum_values[(ps.block, ps.name)],
                    ps.lb + 1e-10, ps.ub - 1e-10),
              ps.label)
    for ps in spec.free
])

println("\n$RULE\nSMOKE 5 — baseline optimum injected into the seed bank\n$RULE")
@printf("  bank size          = %d candidates\n", length(seed_bank.candidates))
@printf("  cluster labels     = %s\n", sort(unique(seed_bank.labels)))
@printf("  last label         = %d  (count = %d)\n",
        seed_bank.labels[end], count(==(seed_bank.labels[end]), seed_bank.labels))
@printf("  injected == θ_base = %s\n", seed_bank.candidates[end] == θ_base)
@printf("  its Q              = %s\n",
        isfinite(seed_bank.Q[end]) ? @sprintf("%.6e", seed_bank.Q[end]) : "Inf")
count(==(seed_bank.labels[end]), seed_bank.labels) == 1 ||
    error("SMOKE 5 FAILED: the injected candidate does not have a cluster to itself.")

println("\n$RULE\nSMOKE 6 — the injected point is emitted as an SA start\n$RULE")
starts = _sa_starts_from_bank(seed_bank, prev_optimum)
hits   = findall(==(θ_base), starts)
@printf("  SA starts          = %d\n", length(starts))
@printf("  baseline vector at start index(es) %s\n", hits)
isempty(hits) &&
    error("SMOKE 6 FAILED: the baseline optimum is not among the SA starts.")

println("\n$RULE\nSMOKE PASSED\n$RULE")

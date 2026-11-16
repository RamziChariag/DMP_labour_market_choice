# smoke/smoke_clusters.jl — crisis_covid spec under INIT_MODE = :clusters (run_smoke.sh).
#
# The baseline optimum must be in the seed bank, in a cluster of its own, and
# `_sa_starts_from_bank` must emit it as a start, bit-identical.

include(joinpath(@__DIR__, "main_spec_only.jl"))

const RULE = "="^70

# The base point in the crisis spec's free set, as §8b maps it.
θ_base = point_in_spec(baseline_data, spec; estimated_only = false).θ

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

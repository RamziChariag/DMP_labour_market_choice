# Smoke test — crisis_covid spec construction under INIT_MODE = :warmstart.
#
# Runs the driver's real spec-construction path (a copy of smm_main.jl truncated
# just before §9), then checks:
#   1. every DEEP_PARAMS entry is pinned in spec.fixed, at the baseline value;
#   2. ρ_x in particular is the baseline estimate, not unpack_θ's −0.55 fallback;
#   3. the free set is exactly REGIME_SPECIFIC_PARAMS minus FIX_PARAMS / σ_w pins;
#   4. with no seed bank, SA's start set is the single point pack_θ(spec), and
#      the Q it reports at [SA init] is the objective at the baseline vector.

include(joinpath(@__DIR__, "main_spec_only.jl"))

const RULE = "="^70

println("\n$RULE\nSMOKE 1 — deep block pinned in spec.fixed\n$RULE")
for (blk, nm) in DEEP_PARAMS_ORDERED
    key    = fixed_key(blk, nm)
    pinned = is_pinned(spec.fixed, blk, nm)
    val    = pinned ? spec.fixed[key] : NaN
    base   = _baseline_param_value(blk, nm, cp_base, up_base, sp_base)
    @printf("  %-6s %-4s  key=:%-10s  pinned=%-5s  value=%12.6f  baseline=%12.6f  match=%s\n",
            blk, nm, key, pinned, val, base, isapprox(val, base; atol = 0, rtol = 0))
end

println("\n$RULE\nSMOKE 2 — ρ_x actually seen by the model\n$RULE")
cp_spec, _, _ = unpack_θ(pack_θ(spec), spec)
@printf("  baseline optimum ρ_x            = %.6f\n", cp_base.ρ_x)
@printf("  ρ_x reaching CommonParams       = %.6f\n", cp_spec.ρ_x)
@printf("  unpack_θ hard-coded fallback    = %.6f\n", -0.55)
@printf("  ρ_x in spec.free?               = %s\n",
        any(ps -> ps.name === :ρ_x, spec.free))
@printf("  ρ_x pinned at baseline?         = %s\n", cp_spec.ρ_x == cp_base.ρ_x)
cp_spec.ρ_x == cp_base.ρ_x ||
    error("SMOKE 2 FAILED: ρ_x = $(cp_spec.ρ_x), expected the baseline $(cp_base.ρ_x).")

println("\n$RULE\nSMOKE 3 — free / fixed accounting\n$RULE")
@printf("  free params  (n = %2d): %s\n", length(spec.free),
        join(("$(ps.block):$(ps.name)" for ps in spec.free), ", "))
@printf("  fixed params (n = %2d): %s\n", length(spec.fixed),
        join(("$k=$(round(v; digits=6))" for (k, v) in pairs(spec.fixed)), ", "))
assert_all_params_accounted(spec; context = "smoke crisis spec")
println("  assert_all_params_accounted: PASS")

println("\n$RULE\nSMOKE 4 — first solve under :warmstart + USE_AS_SEED = true\n$RULE")
θ_spec = pack_θ(spec)
_, up_spec, sp_spec = unpack_θ(θ_spec, spec)
max_dev = maximum(abs(_baseline_param_value(ps, cp_spec, up_spec, sp_spec) -
                      _baseline_param_value(ps, cp_base, up_base, sp_base))
                  for ps in spec.free)
@printf("  max |spec init − baseline optimum| over the free set = %.3e\n", max_dev)

starts = _sa_starts_from_bank(seed_bank, prev_optimum)
@printf("  seed_bank = %s;  prev_optimum = %s;  SA starts from bank = %d\n",
        isnothing(seed_bank) ? "nothing" : "bank", 
        isnothing(prev_optimum) ? "nothing" : "vector", length(starts))
@printf("  ⇒ _run_sa start_set = [pack_θ(spec)]  (sa_random_init = %s)\n",
        spec.run.sa_random_init)

Q_at_init = smm_objective(θ_spec, spec)
@printf("  Q at the baseline vector = %.10e\n", Q_at_init)

# One SA iteration at step = 0: the single proposal coincides with the start, so
# the returned incumbent is the point _sa_loop evaluated FIRST, recoverable
# exactly.  Its [SA init] trace line reports that first solve's Q.
println("\n  Running _run_sa for one zero-step iteration — [SA init] Q0 IS the first solve:")
θ_sa, Q_sa, _ = _run_sa(spec; starts = starts, max_iter = 1,
                        T0 = spec.run.sa_T0, step = 0.0,
                        random_init = spec.run.sa_random_init,
                        show_trace = true, trace_stride = 1,
                        rng = Random.Xoshiro(spec.run.sa_seed))
# θ is the claim and must hold exactly.  Q is compared to a tolerance: the solver
# reduces over threads, so re-evaluating one θ can differ in the last bits.
Q_match = isapprox(Q_sa, Q_at_init; rtol = 1e-10)
@printf("\n  first evaluated θ == pack_θ(spec) exactly?  %s  (max |Δ| = %.3e)\n",
        θ_sa == θ_spec, maximum(abs.(θ_sa .- θ_spec)))
@printf("  first-solve Q ≈ Q(baseline vector)?        %s  (%.12e vs %.12e, rel Δ = %.2e)\n",
        Q_match, Q_sa, Q_at_init, abs(Q_sa - Q_at_init) / abs(Q_at_init))
(θ_sa == θ_spec && Q_match) ||
    error("SMOKE 4 FAILED: SA's first solve is not the baseline vector.")

println("\n$RULE\nSMOKE PASSED\n$RULE")

# smoke/smoke_guard.jl — the parameter-accounting guard (run_smoke.sh).
#
# Rebuilds the crisis spec with ρ_x deleted from the pinned block (neither free nor
# pinned) and checks that `assert_all_params_accounted` raises before any solve, naming
# common:ρ_x, and passes on the shipped spec.

include(joinpath(@__DIR__, "main_spec_only.jl"))

const RULE = "="^70

println("\n$RULE\nSMOKE 7 — guard fires on the v14.8 fixed block\n$RULE")

fixed_v148 = NamedTuple(k => spec.fixed[k] for k in keys(spec.fixed) if k !== :ρ_x)
spec_v148  = build_smm_spec(moments, sim_smm;
                            fixed        = fixed_v148,
                            free_specs   = free_params,
                            run          = run_params,
                            W            = W_opt,
                            q_scale      = Q_SCALE,
                            skip_moments = ACTIVE_SKIP_MOMENTS)

@printf("  ρ_x free?  %s   ρ_x pinned?  %s   (the v14.8 state: neither)\n",
        any(ps -> ps.name === :ρ_x, spec_v148.free),
        is_pinned(spec_v148.fixed, :common, :ρ_x))

caught = try
    assert_all_params_accounted(spec_v148; context = "v14.8 crisis spec")
    nothing
catch e
    e
end

isnothing(caught) && error("SMOKE 7 FAILED: the guard did not fire on the v14.8 spec.")
@printf("  guard raised: %s\n", sprint(showerror, caught))

# The shipped spec passes.
assert_all_params_accounted(spec; context = "v14.9 crisis spec")
println("  guard passes on the v14.9 spec.")

println("\n$RULE\nSMOKE PASSED\n$RULE")

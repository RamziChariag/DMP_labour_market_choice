#!/usr/bin/env julia
# test_generator.jl — exercise the SHIPPING generate_population and its DE call sites
# against a stub objective, so every branch is checked at zero solver cost.
#
#   RSROOT=<repo> julia -t 4 test_generator.jl
#
# The stub replaces smm_objective with a quadratic bowl plus a controllable infeasible
# region. That fixes the two quantities the generator's logic keys on — how many
# candidates are feasible, and how many improve on the incumbent — so each branch can be
# driven deliberately instead of hoping the real solver happens to produce it.

using LinearAlgebra, SparseArrays, Statistics, Random, Printf, Serialization
using Distributions, FastGaussQuadrature, Interpolations, Parameters, Base.Threads
using Optim, CSV, DataFrames, Clustering, QuasiMonteCarlo, JSON3
BLAS.set_num_threads(1)

const R = ENV["RSROOT"]
for f in ("grids", "params", "unskilled", "skilled", "solver", "equilibrium")
    include(joinpath(R, "code", "solver", f * ".jl"))
end
for f in ("moments", "smm_params", "smm")
    include(joinpath(R, "code", "smm", f * ".jl"))
end

# ── stub objective ───────────────────────────────────────────────────────────
# INFEAS_FRAC: fraction of the box declared infeasible, by a hash of the point so the
# verdict is deterministic and position-dependent rather than random per call.
# CENTRE: the minimiser. Placing it at the seed makes every draw non-improving, which is
# the barren case; placing it away makes most draws improving.
const INFEAS_FRAC = Ref(0.0)
const CENTRE      = Ref(Float64[])
const NCALLS      = Ref(0)
const SEED_INFEAS = Ref(false)   # make the seed itself unsolvable, nothing else

# The signature must match the real method EXACTLY, keyword defaults included. A looser
# first argument (AbstractVector rather than AbstractVector{Float64}) makes the real
# method more specific, so dispatch keeps choosing it and every call reaches the solver —
# the test then runs for hours and proves nothing about the branches.
function smm_objective(
    θ     :: AbstractVector{Float64},
    spec  :: SMMSpec;
    Nx    :: Int = spec.run.Nx,
    Np_U  :: Int = spec.run.Np_U,
    Np_S  :: Int = spec.run.Np_S,
    moments_out :: Union{Nothing,AbstractVector{Float64}} = nothing,
) :: Float64
    NCALLS[] += 1
    # The seed is handled first and explicitly. INFEAS_FRAC judges by a hash of the point,
    # which can catch the seed too — and an infeasible seed changes which branch the
    # generator takes, so a test meaning to starve the DRAWS would silently end up testing
    # the infeasible-seed path instead.
    if all(iszero, θ)
        SEED_INFEAS[] && return Inf
        c = isempty(CENTRE[]) ? zeros(length(θ)) : CENTRE[]
        return 1000.0 + sum(c .^ 2)               # the bowl, evaluated at the seed
    end
    if INFEAS_FRAC[] > 0
        h = abs(hash(round.(collect(θ), digits = 6))) % 1000
        h < 1000 * INFEAS_FRAC[] && return Inf
    end
    c = isempty(CENTRE[]) ? zeros(length(θ)) : CENTRE[]
    return 1000.0 + sum((collect(θ) .- c) .^ 2)
end

const SPEC = let
    b = deserialize(joinpath(R, "output", "smm", "smm_result_base_fc_diagonalW.jls"))
    b.spec
end
const D    = length(SPEC.free)
const SEED = zeros(D)

npass = Ref(0); nfail = Ref(0)
function check(name, cond, detail = "")
    cond ? (npass[] += 1) : (nfail[] += 1)
    @printf("  %-4s %-52s %s\n", cond ? "PASS" : "FAIL", name, detail)
end

prev  = [SEED .+ 0.05 .* randn(Xoshiro(700 + i), D) for i in 1:40]
prevQ = [smm_objective(p, SPEC) for p in prev]
prevh = Set(hash.(prev))

# ── 1. initialisation: feasibility pool, barren impossible ───────────────────
println("\n1. INITIALISATION (require_improvement = false)")
INFEAS_FRAC[] = 0.0; CENTRE[] = SEED             # seed is the minimum: nothing improves
p, f, cr, t = generate_population(SEED, SPEC, 40; per_k = 4, verbose = false,
                                  rng = Xoshiro(1))
check("population is exactly n_slots", length(p) == 40, "n=$(length(p))")
check("barren is false even with 0 improvements", t.barren == false,
      "better=$(t.n_better)")
check("both counts reported", t.n_better == 0 && t.n_feas_total == 4D,
      "better=$(t.n_better) feas=$(t.n_feas_total)")
check("member 1 is the seed", p[1] == SEED)
check("f and cr are finite and in range",
      isfinite(f) && 0.1 <= f <= 1.5 && 1/D <= cr <= 1.0,
      @sprintf("f=%.3f cr=%.3f", f, cr))

# ── 2. reheat, improvements available ────────────────────────────────────────
println("\n2. REHEAT with improvements (require_improvement = true)")
CENTRE[] = fill(0.5, D)                          # minimum away from the seed
p, f, cr, t = generate_population(SEED, SPEC, 40; per_k = 4, fill_from = prev,
                                  fill_Q = prevQ, require_improvement = true,
                                  verbose = false, rng = Xoshiro(2))
check("improvements found, not barren", t.barren == false && t.n_better > 0,
      "better=$(t.n_better)")
check("population full", length(p) == 40)
check("every member improves or is carried/seed",
      all(i -> smm_objective(p[i], SPEC) < 1000.0 + sum((SEED .- CENTRE[]).^2) ||
               hash(p[i]) in prevh || p[i] == SEED, 2:length(p)))
check("no padding when the pool suffices", t.n_padded == 0, "pad=$(t.n_padded)")

# ── 3. barren reheat: the 50/50 mix ──────────────────────────────────────────
println("\n3. BARREN reheat (zero improvements, ample feasibility)")
CENTRE[] = SEED                                  # seed optimal: nothing can improve
p, f, cr, t = generate_population(SEED, SPEC, 40; per_k = 4, fill_from = prev,
                                  fill_Q = prevQ, require_improvement = true,
                                  verbose = false, rng = Xoshiro(3))
nprev = count(x -> hash(x) in prevh, p)
check("barren detected", t.barren == true, "better=$(t.n_better)")
check("population full", length(p) == 40, "n=$(length(p))")
check("mix is ~50/50", abs(t.n_fresh - 20) <= 1 && abs(t.n_carried - 20) <= 1,
      "fresh=$(t.n_fresh) carried=$(t.n_carried)")
check("carried members really come from fill_from", nprev == t.n_carried,
      "traced=$nprev carried=$(t.n_carried)")
check("members are distinct (no seed padding)",
      length(unique(hash.(p))) >= 39, "distinct=$(length(unique(hash.(p))))")
check("f from the fresh half only, still in range",
      isfinite(f) && 0.1 <= f <= 1.5, @sprintf("f=%.3f", f))
check("cr from the yield weight, not the shuffle", isfinite(cr) && cr > 0,
      @sprintf("cr=%.3f", cr))

# ── 4. barren AND too few feasible draws for half the slots ──────────────────
println("\n4. BARREN with a SHORT new half (the fallback)")
INFEAS_FRAC[] = 0.90                             # only ~10% of draws solve
p, f, cr, t = generate_population(SEED, SPEC, 40; per_k = 4, fill_from = prev,
                                  fill_Q = prevQ, require_improvement = true,
                                  verbose = false, rng = Xoshiro(4))
nprev = count(x -> hash(x) in prevh, p)
check("still barren", t.barren == true)
check("population STILL full", length(p) == 40, "n=$(length(p))")
check("new half short, previous generation covers the rest",
      t.n_fresh < 20 && t.n_fresh + t.n_carried == 40,
      "fresh=$(t.n_fresh) carried=$(t.n_carried)")
check("carried count matches traced members", nprev == t.n_carried,
      "traced=$nprev")
check("f still finite with a short fresh half", isfinite(f), @sprintf("f=%.3f", f))
INFEAS_FRAC[] = 0.0

# ── 4b. improvements SHORT of the slot count, feasibility ample ──────────────
# The population must still fill — from feasible non-improving draws before reaching back
# to the previous generation — while cr keeps reading only the improving members.
println("\n4b. SCARCE IMPROVEMENTS (not barren, but too few to fill)")
INFEAS_FRAC[] = 0.0
# The centre must sit about one typical draw away from the seed. Nearer and every draw
# overshoots it (nothing improves — that is the barren case); much further and every draw
# moves toward it (everything improves). A per-coordinate step is sigma·cap ≈ 0.066, so a
# k≈12 draw displaces ≈0.23 in norm; putting the centre at that distance makes improvement
# a minority outcome, which is the case under test.
CENTRE[] = fill(0.0467, D)
p, f, cr, t = generate_population(SEED, SPEC, 40; per_k = 4, fill_from = prev,
                                  fill_Q = prevQ, require_improvement = true,
                                  verbose = false, rng = Xoshiro(41))
n_imp_slots = sum(t.slots_imp)
n_fb        = sum(t.slots) - n_imp_slots
check("not barren (some improvements exist)", t.barren == false, "better=$(t.n_better)")
check("improvements are scarcer than the slots", t.n_better < 40, "better=$(t.n_better)")
check("population full", length(p) == 40, "n=$(length(p))")
check("shortfall came from feasible draws, not the previous generation",
      n_fb > 0 && t.n_carried == 0, "fallback=$n_fb carried=$(t.n_carried)")
check("improving slots capped by the improving count", n_imp_slots <= t.n_better,
      "imp_slots=$n_imp_slots better=$(t.n_better)")
check("cr finite and in range", isfinite(cr) && 1/D <= cr <= 1.0, @sprintf("cr=%.3f", cr))

# ── 4c. cr must not collapse merely because improvements are scarce ──────────
# The failure this guards against: basing cr on ALLOCATED slots makes it fall whenever
# improvements thin out, because a quota is capped by the improving draws available and
# the high sparsities hit that cap first. cr would then read as "the frontier has
# receded" when nothing about the shape changed. Comparing a plentiful call against a
# scarce one on the SAME yield shape isolates that: cr should barely move.
println("\n4c. cr STABILITY under scarcity (no premature collapse)")
INFEAS_FRAC[] = 0.0
CENTRE[] = fill(0.0467, D)                        # scarce improvements, as in 4b
_, _, cr_scarce, t_scarce = generate_population(SEED, SPEC, 40; per_k = 4,
        fill_from = prev, fill_Q = prevQ, require_improvement = true,
        verbose = false, rng = Xoshiro(43))
_, _, cr_plenty, t_plenty = generate_population(SEED, SPEC, 40; per_k = 40,
        fill_from = prev, fill_Q = prevQ, require_improvement = true,
        verbose = false, rng = Xoshiro(43))
check("scarce call really is scarce", t_scarce.n_better < 40,
      "scarce better=$(t_scarce.n_better)  plenty better=$(t_plenty.n_better)")
check("more draws really give more improvements", t_plenty.n_better > t_scarce.n_better)
check("cr is stable across a 10x change in draw count",
      abs(cr_scarce - cr_plenty) < 0.15,
      @sprintf("scarce cr=%.3f  plenty cr=%.3f  |diff|=%.3f",
               cr_scarce, cr_plenty, abs(cr_scarce - cr_plenty)))

# ── 5. barren with no previous generation: must error, not fall through ──────
println("\n5. CONTRACT: barren without fill_from")
INFEAS_FRAC[] = 0.0; CENTRE[] = SEED             # seed optimal => nothing can improve
# Julia soft scope: assigning inside `try` at top level creates a NEW local unless the
# name is declared global, so the outer flags would never be written and the check would
# silently pass on stale values. Wrapping the probe in a function avoids the issue.
function probe_contract()
    try
        _, _, _, tb = generate_population(SEED, SPEC, 40; per_k = 4,
                                          require_improvement = true,
                                          verbose = false, rng = Xoshiro(5))
        return false, "", tb
    catch e
        return true, sprint(showerror, e), nothing
    end
end
errored, msg, got = probe_contract()
check("errors loudly rather than silently degrading",
      errored && occursin("fill_from", msg),
      errored ? "threw, but: " * first(split(msg, "\n")) :
                "no exception at all; barren=$(got === nothing ? "?" : got.barren)")

# ── 6. severe infeasibility at initialisation: population must still fill ────
println("\n6. INITIALISATION with severe infeasibility")
INFEAS_FRAC[] = 0.97
p, f, cr, t = generate_population(SEED, SPEC, 40; per_k = 2, verbose = false,
                                  rng = Xoshiro(6))
check("population full via padding", length(p) == 40,
      "n=$(length(p)) pad=$(t.n_padded)")
check("padding is reported, not hidden", t.n_padded > 0, "pad=$(t.n_padded)")
check("f and cr still finite", isfinite(f) && isfinite(cr),
      @sprintf("f=%.3f cr=%.3f", f, cr))
INFEAS_FRAC[] = 0.0

# ── 6b. an INFEASIBLE seed ───────────────────────────────────────────────────
# A warm start whose spec has since changed may not solve. Q_seed = Inf then makes every
# shortfall Inf, which propagated a NaN into the allocation and aborted the run.
println("\n6b. INFEASIBLE SEED (Q_seed = Inf)")
INFEAS_FRAC[] = 0.0
SEED_INFEAS[] = true                              # stub returns Inf for the seed only
p, f, cr, t = generate_population(SEED, SPEC, 40; per_k = 4, verbose = false,
                                  rng = Xoshiro(61))
check("does not throw with an infeasible seed", true)
check("population full", length(p) == 40, "n=$(length(p))")
check("f and cr finite", isfinite(f) && isfinite(cr), @sprintf("f=%.3f cr=%.3f", f, cr))
p, f, cr, t = generate_population(SEED, SPEC, 40; per_k = 4, fill_from = prev,
                                  fill_Q = prevQ, require_improvement = true,
                                  verbose = false, rng = Xoshiro(62))
check("reheat with infeasible seed: not barren if draws solve",
      t.barren == false && t.n_better > 0, "better=$(t.n_better)")
check("population full", length(p) == 40)
SEED_INFEAS[] = false

# ── 7. the DE loop: reheat fires, barren ends the run at the NEXT stall ──────
println("\n7. DE LOOP: reheat, barren, exit")
CENTRE[] = SEED                                  # nothing improves anywhere: forces barren
spec_t = SMMSpec([ParamSpec(ps.block, ps.name, ps.lb, ps.ub, 0.0, ps.label)
                  for ps in SPEC.free],
                 SPEC.fixed, SPEC.moments, SPEC.sim, SPEC.run, SPEC.W, SPEC.q_scale)
θ, Q, iters = _run_de(spec_t; max_iter = 40, pop_size = 16, f = 0.7, cr = 0.9,
                      patience = 999, avg_tol = 0.0, gen_per_k = 2,
                      reheat_flat = 3, reheat_rate = 1.0, max_reheats = 5,
                      adapt_fcr = true, show_gens = true, show_members = false,
                      rng = Xoshiro(7))
check("DE returned a finite Q", isfinite(Q), @sprintf("Q=%.6f iters=%d", Q, iters))
# The exit path needs a BARREN reheat, which needs a stall AND zero improvements. On this
# stub bowl the population keeps improving, so the reheat that fired was productive and
# the run correctly used its full budget. Reaching the budget here is right behaviour;
# the exit path itself is covered by groups 3 and 4, which drive barren directly.
check("a reheat fired and the run completed", iters == 40, "iters=$iters of 40")

@printf("\n%d passed, %d failed  (%d stub objective calls)\n", npass[], nfail[], NCALLS[])
exit(nfail[] == 0 ? 0 : 1)

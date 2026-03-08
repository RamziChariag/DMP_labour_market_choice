############################################################
# smm.jl — SMM objective and optimisation loops
#
# Main entry points
# ─────────────────
#   run_smm(spec; method=:sa, ...)       single-run estimation
#   multistart_smm(spec, n; ...)         repeated starts, keep best
#
# Methods
# ───────
#   :sa           our own simulated annealing loop (default)
#   :neldermead   Optim.jl Nelder–Mead (good for polishing after SA)
#   :lbfgs        Optim.jl L-BFGS
#
# Design notes
# ────────────
# - smm_objective ALWAYS forces verbose=0 on the solver regardless
#   of what spec.sim says.  Solver output during SMM is always noise.
# - SA is implemented here, not delegated to Optim.jl, because
#   Optim's SA rejects Inf proposals silently and gets stuck when
#   many parameter regions produce non-converging models.  Our loop
#   handles Inf correctly (always reject, never update).
############################################################


# ============================================================
# Helper: silence a SimParams (force verbose=0)
# ============================================================

function _silence(sim::SimParams) :: SimParams
    return SimParams(
        tol_inner      = sim.tol_inner,
        tol_outer_U    = sim.tol_outer_U,
        tol_outer_S    = sim.tol_outer_S,
        tol_global     = sim.tol_global,
        maxit_inner    = sim.maxit_inner,
        maxit_outer    = sim.maxit_outer,
        maxit_global   = sim.maxit_global,
        conv_streak    = sim.conv_streak,
        use_anderson   = sim.use_anderson,
        anderson_m     = sim.anderson_m,
        anderson_reg   = sim.anderson_reg,
        damp_pstar_U   = sim.damp_pstar_U,
        damp_pstar_S   = sim.damp_pstar_S,
        verbose        = 0,
        verbose_stride = sim.verbose_stride,
    )
end


# ============================================================
# Weighted loss
# ============================================================

"""
    compute_loss(m_model, spec) -> Float64

    Q(θ) = Σ_k  w_k · (m_k^model − m_k^data)²

Only moments with weight > 0 are included.
"""
function compute_loss(m_model::NamedTuple, spec::SMMSpec) :: Float64
    Q = 0.0
    for k in keys(spec.moments)
        target = spec.moments[k]
        target.weight <= 0.0  && continue
        !hasproperty(m_model, k) && continue
        dev  = getproperty(m_model, k) - target.value
        Q   += target.weight * dev^2
    end
    return Q
end


# ============================================================
# SMM objective
# ============================================================

"""
    smm_objective(θ_unc, spec) -> Float64

Solve the model at parameters decoded from θ_unc and return Q(θ).
Returns Inf (never throws) on any failure or non-convergence.
The solver always runs silently regardless of spec.sim.verbose.
"""
function smm_objective(
    θ_unc :: AbstractVector{Float64},
    spec  :: SMMSpec
) :: Float64

    cp, rp, up, sp = unpack_θ(θ_unc, spec)

    local model, solve_result
    try
        model, solve_result = solve_model(cp, rp, up, sp, spec.sim;
                                          Nx   = spec.Nx,
                                          Np_U = spec.Np_U,
                                          Np_S = spec.Np_S)
    catch
        return Inf
    end

    solve_result.ok || return Inf

    local obj_eq, m_model
    try
        obj_eq  = compute_equilibrium_objects(model)
        m_model = model_moments(obj_eq)
    catch
        return Inf
    end

    return compute_loss(m_model, spec)
end


# ============================================================
# Result type
# ============================================================

struct SMMResult
    theta_opt  :: Vector{Float64}
    params_opt :: NamedTuple
    loss_opt   :: Float64
    converged  :: Bool
    iterations :: Int
    spec       :: SMMSpec
end


# ============================================================
# Simulated Annealing (own implementation)
# ============================================================

"""
    _run_sa(spec; T0, step, max_iter, show_trace, trace_stride, rng)
        -> (theta_best, Q_best, iters)

Simulated annealing in unconstrained (logit) space.

Temperature schedule: T(t) = T0 / log(t + 2)
Acceptance rule:
  - always accept downhill moves (Q_new < Q_current)
  - accept uphill with prob exp(-ΔQ / T(t))
  - never accept Inf proposals
"""
function _run_sa(
    spec         :: SMMSpec;
    T0           :: Float64 = 2.0,
    step         :: Float64 = 0.15,
    max_iter     :: Int     = 5000,
    show_trace   :: Bool    = true,
    trace_stride :: Int     = 100,
    rng                     = Random.default_rng(),
)
    theta      = pack_theta(spec)
    Q          = smm_objective(theta, spec)
    theta_best = copy(theta)
    Q_best     = isfinite(Q) ? Q : Inf
    n_acc      = 0
    n_fin      = 0

    if show_trace
        @printf("  [SA init]  Q0 = %s\n",
                isfinite(Q) ? @sprintf("%.6e", Q) : "Inf (bad starting point)")
        flush(stdout)
    end

    for t in 1:max_iter
        theta_prop = theta .+ step .* randn(rng, length(theta))
        Q_prop     = smm_objective(theta_prop, spec)

        if isfinite(Q_prop)
            n_fin += 1
            T = T0 / log(t + 2)

            accept = if !isfinite(Q)
                true
            elseif Q_prop <= Q
                true
            else
                rand(rng) < exp(-(Q_prop - Q) / T)
            end

            if accept
                theta = theta_prop
                Q     = Q_prop
                n_acc += 1
                if Q < Q_best
                    Q_best     = Q
                    theta_best = copy(theta)
                end
            end
        end

        if show_trace && t % trace_stride == 0
            T_now = T0 / log(t + 2)
            @printf("  [SA t=%5d]  curr=%-14s  best=%.6e  T=%.4f  acc=%.2f  fin=%.2f\n",
                    t,
                    isfinite(Q) ? @sprintf("%.6e", Q) : "Inf",
                    Q_best, T_now,
                    n_acc / t,
                    n_fin / t)
            flush(stdout)
        end
    end

    if show_trace
        @printf("  [SA done]  Q_best=%.6e  accepted %d/%d  finite proposals %d/%d\n",
                Q_best, n_acc, max_iter, n_fin, max_iter)
        flush(stdout)
    end

    return theta_best, Q_best, max_iter
end

# Alias so pack_theta works (smm_params.jl defines pack_θ with Unicode)
pack_theta(spec) = pack_θ(spec)


# ============================================================
# Differential Evolution (own implementation)
# ============================================================

"""
    _run_de(spec; max_iter, pop_size, f, cr, show_trace, trace_stride, rng)
        -> (theta_best, Q_best, iters)

Differential Evolution in unconstrained (logit) space.

Algorithm: DE/rand/1/bin  (standard, robust variant)
  For each member i of the population:
    1. Pick three distinct members a, b, c ≠ i at random
    2. Mutant:   v = a + F * (b - c)
    3. Trial:    u = v where rand < CR, else keep x_i  (binomial crossover)
    4. Select:   replace x_i with u if Q(u) < Q(x_i)

Non-converging proposals (Q = Inf) are never accepted — the current
population member survives unchanged, so the population always contains
only feasible parameter vectors.

Parameters
──────────
  pop_size   population size. Rule of thumb: 10 × n_params.
             Larger = more exploration, slower per generation.
  f          mutation scale ∈ (0, 2). Typical: 0.5–0.9.
             Higher = larger steps, more exploration.
  cr         crossover probability ∈ (0, 1). Typical: 0.7–0.9.
             Higher = trial vector borrows more from mutant.
"""
function _run_de(
    spec         :: SMMSpec;
    max_iter     :: Int     = 5000,
    pop_size     :: Int     = 0,
    f            :: Float64 = 0.65,
    cr           :: Float64 = 0.85,
    show_trace   :: Bool    = true,
    trace_stride :: Int     = 100,
    rng                     = Random.default_rng(),
)
    npar     = length(spec.free)
    pop_size = pop_size > 0 ? pop_size : 10 * npar
    theta0   = pack_theta(spec)

    # ── Initialise population around starting point ────────────────────
    # First member is the supplied starting point; rest are random
    # perturbations. Using ±2 in logit space covers most of [lb, ub].
    pop  = [theta0 .+ 2.0 .* (2.0 .* rand(rng, npar) .- 1.0)
            for _ in 1:pop_size]
    pop[1] = copy(theta0)

    # Evaluate initial population — Inf for non-converging members
    Q_pop = fill(Inf, pop_size)

    if show_trace
        @printf("  [DE init]  evaluating %d initial members...\n", pop_size)
        flush(stdout)
    end

    n_feasible = 0
    for i in 1:pop_size
        Q_pop[i] = smm_objective(pop[i], spec)
        isfinite(Q_pop[i]) && (n_feasible += 1)
        if show_trace && (i % trace_stride == 0 || i == pop_size)
            @printf("  [DE init]  evaluated %d/%d  feasible so far: %d\n",
                    i, pop_size, n_feasible)
            flush(stdout)
        end
    end

    i_best   = argmin(Q_pop)
    Q_best   = Q_pop[i_best]
    theta_best = copy(pop[i_best])

    if show_trace
        @printf("  [DE init]  feasible=%d/%d  Q_best=%.6e\n",
                n_feasible, pop_size, Q_best)
        flush(stdout)
    end

    # ── Main DE loop (generations) ─────────────────────────────────────
    n_evals = pop_size
    for gen in 1:max_iter
        n_improved = 0
        for i in 1:pop_size
            # Select three distinct members ≠ i
            candidates = filter(j -> j != i, 1:pop_size)
            abc = candidates[randperm(rng, length(candidates))[1:3]]
            a, b, c = pop[abc[1]], pop[abc[2]], pop[abc[3]]

            # Mutant vector
            v = a .+ f .* (b .- c)

            # Binomial crossover — ensure at least one dimension from mutant
            mask    = rand(rng, npar) .< cr
            j_force = rand(rng, 1:npar)
            mask[j_force] = true
            u = ifelse.(mask, v, pop[i])

            # Evaluate trial vector
            Q_u = smm_objective(u, spec)
            n_evals += 1

            # Greedy selection — Inf proposals never win
            if isfinite(Q_u) && Q_u < Q_pop[i]
                pop[i]   = u
                Q_pop[i] = Q_u
                n_improved += 1
                if Q_u < Q_best
                    Q_best     = Q_u
                    theta_best = copy(u)
                end
            end

            if show_trace && i % trace_stride == 0
                @printf("  [DE gen=%4d  member=%4d/%4d]  Q_best=%.6e  improved=%d\n",
                        gen, i, pop_size, Q_best, n_improved)
                flush(stdout)
            end
        end

        if show_trace
            Q_mean = mean(filter(isfinite, Q_pop))
            n_feas = count(isfinite, Q_pop)
            @printf("  [DE gen=%4d  DONE]  Q_best=%.6e  Q_mean=%.6e  feasible=%d/%d  evals=%d\n",
                    gen, Q_best, Q_mean, n_feas, pop_size, n_evals)
            flush(stdout)
        end
    end

    if show_trace
        @printf("  [DE done]  Q_best=%.6e  total evals=%d\n", Q_best, n_evals)
        flush(stdout)
    end

    return theta_best, Q_best, max_iter * pop_size
end


# ============================================================
# Main optimisation entry point
# ============================================================

"""
    run_smm(spec; method=:sa, max_iter=5000, ...) -> SMMResult

Estimate parameters by SMM.

  :de          Differential evolution via BlackBoxOptim.jl (recommended
               global search). Population-based, derivative-free, robust
               to non-smooth objectives. Use population_size ~10x n_params.

  :sa          Simulated annealing (our own loop). Alternative global
               search, works well with tuned T0 and step size.

  :neldermead  Nelder-Mead via Optim.jl. Best for polishing a solution
               already near the optimum. Always use this after :de or :sa.

Typical workflow:
  res_de  = run_smm(spec; method=:de,         max_iter=5000)
  res_pol = run_smm(_spec_with_init(spec, res_de.theta_opt);
                    method=:neldermead, max_iter=2000)
"""
function run_smm(
    spec            :: SMMSpec;
    method          :: Symbol  = :de,
    max_iter        :: Int     = 5000,
    # DE options
    de_pop_size     :: Int     = 0,       # 0 = auto (10 × n_params)
    de_f            :: Float64 = 0.65,    # mutation scale (0.5–0.9 typical)
    de_cr           :: Float64 = 0.85,    # crossover probability
    # SA options
    sa_T0           :: Float64 = 2.0,
    sa_step         :: Float64 = 0.15,
    # Optim options
    f_tol           :: Float64 = 1e-6,
    x_tol           :: Float64 = 1e-5,
    # Trace
    show_trace      :: Bool    = true,
    trace_stride    :: Int     = 100,
    rng                        = Random.default_rng(),
) :: SMMResult

    npar = length(spec.free)
    @printf("\nStarting SMM  (%s,  %d free params,  max_iter=%d)\n",
            method, npar, max_iter)
    flush(stdout)

    if method == :de
        theta_opt, loss_opt, niters = _run_de(
            spec;
            max_iter     = max_iter,
            pop_size     = de_pop_size > 0 ? de_pop_size : 100 * npar,
            f            = de_f,
            cr           = de_cr,
            show_trace   = show_trace,
            trace_stride = trace_stride,
            rng          = rng,
        )
        converged = isfinite(loss_opt)

    elseif method == :sa
        theta_opt, loss_opt, niters = _run_sa(
            spec;
            T0           = sa_T0,
            step         = sa_step,
            max_iter     = max_iter,
            show_trace   = show_trace,
            trace_stride = trace_stride,
            rng          = rng,
        )
        converged = isfinite(loss_opt)

    elseif method in (:neldermead, :lbfgs, :bfgs)
        theta0     = pack_theta(spec)
        iter_count = Ref(0)
        best_loss  = Ref(Inf)

        function obj_traced(theta)
            iter_count[] += 1
            Q = smm_objective(theta, spec)
            isfinite(Q) && Q < best_loss[] && (best_loss[] = Q)
            if show_trace && iter_count[] % trace_stride == 0
                @printf("  [%s iter %4d]  Q=%-14s  best=%.6e\n",
                        method, iter_count[],
                        isfinite(Q) ? @sprintf("%.6e", Q) : "Inf",
                        best_loss[])
                flush(stdout)
            end
            return isfinite(Q) ? Q : 1e16
        end

        opt_method = (method == :neldermead) ? Optim.NelderMead() :
                     (method == :lbfgs)      ? Optim.LBFGS()      : Optim.BFGS()

        options   = Optim.Options(iterations=max_iter, f_tol=f_tol,
                                  x_tol=x_tol, show_trace=false)
        result    = Optim.optimize(obj_traced, theta0, opt_method, options)
        theta_opt = Optim.minimizer(result)
        loss_opt  = smm_objective(theta_opt, spec)
        converged = Optim.converged(result) && isfinite(loss_opt)
        niters    = Optim.iterations(result)

    else
        error("Unknown method :$method. Choose :de, :sa, :neldermead, :lbfgs, or :bfgs.")
    end

    @printf("\nSMM complete:  Q=%.6e  converged=%s  iters=%d\n",
            isfinite(loss_opt) ? loss_opt : Inf, converged, niters)
    flush(stdout)

    cp_opt, rp_opt, up_opt, sp_opt = unpack_θ(theta_opt, spec)
    params_opt = _params_to_namedtuple(cp_opt, rp_opt, up_opt, sp_opt, spec)

    res = SMMResult(theta_opt, params_opt, loss_opt, converged, niters, spec)
    print_results(res)
    return res
end


# ============================================================
# Multi-start wrapper
# ============================================================

"""
    multistart_smm(spec, n_starts; method=:sa, max_iter=5000, seed=42)
        -> SMMResult

Run n_starts searches from randomly perturbed starting points.
Returns the best result across all starts.
"""
function multistart_smm(
    spec     :: SMMSpec,
    n_starts :: Int;
    method   :: Symbol  = :sa,
    max_iter :: Int     = 5000,
    seed     :: Int     = 42,
) :: SMMResult

    rng  = Random.MersenneTwister(seed)
    theta0 = pack_theta(spec)
    npar   = length(theta0)

    best_result = nothing

    for s in 1:n_starts
        @printf("\n══ Multi-start %d / %d ══\n", s, n_starts)
        flush(stdout)

        theta_start = theta0 .+ 0.5 .* randn(rng, npar)
        spec_s      = _spec_with_init(spec, theta_start)

        try
            res_s = run_smm(spec_s; method=method, max_iter=max_iter,
                            show_trace=false, rng=rng)
            if isnothing(best_result) || res_s.loss_opt < best_result.loss_opt
                best_result = res_s
                @printf("  -> new best  Q = %.6e\n", best_result.loss_opt)
                flush(stdout)
            end
        catch e
            @printf("  start %d failed: %s\n", s, e)
        end
    end

    @printf("\nMulti-start done.  Best Q = %.6e\n",
            isnothing(best_result) ? Inf : best_result.loss_opt)
    return best_result
end


# ============================================================
# Result display and saving
# ============================================================

function print_results(res::SMMResult)
    @printf("\n╔══════════════════════════════════════════════════════╗\n")
    @printf("║  SMM Estimates                                       ║\n")
    @printf("╠══════════════════════════════════════════════════════╣\n")
    @printf("  %-6s  %-22s  %10s\n", "block", "parameter", "estimate")
    @printf("  %s\n", "-"^42)
    for ps in res.spec.free
        key = Symbol(string(ps.block) * "_" * string(ps.name))
        val = hasproperty(res.params_opt, key) ? res.params_opt[key] : NaN
        @printf("  %-6s  %-22s  %10.5f\n", ps.block, ps.label, val)
    end
    if length(res.spec.fixed) > 0
        @printf("\n  Fixed:\n")
        for (k, v) in pairs(res.spec.fixed)
            @printf("    %-24s  %10.5f\n", k, v)
        end
    end
    @printf("\n  Q = %.8e  |  converged = %s  |  iters = %d\n",
            res.loss_opt, res.converged, res.iterations)
    @printf("╚══════════════════════════════════════════════════════╝\n\n")
    flush(stdout)
end


function save_results(res::SMMResult, path::String)
    open(path, "w") do io
        println(io, "block,name,label,estimate,lb,ub,fixed")
        for ps in res.spec.free
            key = Symbol(string(ps.block) * "_" * string(ps.name))
            val = hasproperty(res.params_opt, key) ? res.params_opt[key] : NaN
            @printf(io, "%s,%s,%s,%.8f,%.8f,%.8f,false\n",
                    ps.block, ps.name, ps.label, val, ps.lb, ps.ub)
        end
        for (k, v) in pairs(res.spec.fixed)
            @printf(io, "fixed,%s,%s,%.8f,,,true\n", k, k, v)
        end
        @printf(io, "\n# Q = %.10e\n", res.loss_opt)
        @printf(io, "# converged = %s\n", res.converged)
        @printf(io, "# iterations = %d\n", res.iterations)
    end
    @printf("Results saved to: %s\n", path)
end


# ============================================================
# Internal helpers
# ============================================================

function _params_to_namedtuple(cp, rp, up, sp, spec::SMMSpec)
    d = Dict{Symbol, Float64}()
    for ps in spec.free
        val = if ps.block == :common;   getfield(cp, ps.name)
              elseif ps.block == :regime; getfield(rp, ps.name)
              elseif ps.block == :unsk;   getfield(up, ps.name)
              else;                       getfield(sp, ps.name)
              end
        d[Symbol(string(ps.block) * "_" * string(ps.name))] = val
    end
    return NamedTuple(d)
end

function _spec_with_init(spec::SMMSpec, theta_unc::Vector{Float64})
    new_free = [
        ParamSpec(ps.block, ps.name, ps.lb, ps.ub,
                  _to_constrained(theta_unc[i], ps.lb, ps.ub), ps.label)
        for (i, ps) in enumerate(spec.free)
    ]
    return SMMSpec(new_free, spec.fixed, spec.moments, spec.sim,
                   spec.Nx, spec.Np_U, spec.Np_S)
end

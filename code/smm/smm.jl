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
# - SA is implemented here, not delegated to Optim.jl, because
#   Optim's SA rejects Inf proposals silently and gets stuck when
#   many parameter regions produce non-converging models.  Our loop
#   handles Inf correctly (always reject, never update).
############################################################



# ============================================================
# Weighted loss
# ============================================================


"""
    compute_loss(m_model, spec) -> Float64


    Q(θ) = Σ_k  w_k · [(m_k^model − m_k^data) / |m_k^data|]²


Diagonal-weight loss with scale-normalised deviations.
Each deviation is divided by |m̂_k| (floored at 1e-10) so that
moments of different magnitudes contribute on a comparable scale.
Only moments with weight > 0 are included.
"""
function compute_loss(m_model::NamedTuple, spec::SMMSpec) :: Float64
    Q = 0.0
    for k in keys(spec.moments)
        target = spec.moments[k]
        target.weight <= 0.0  && continue
        !hasproperty(m_model, k) && continue
        scale = max(abs(target.value), 1e-10)
        dev   = (getproperty(m_model, k) - target.value) / scale
        Q    += target.weight * dev^2
    end
    return Q
end



"""
    compute_loss_matrix(m_model, spec, W) -> Float64


    Q(θ) = g̃(θ)' W g̃(θ)     where g̃_k = (m_k^model − m̂_k) / |m̂_k|


Compute the loss using the full K×K optimal weight matrix W = Σ̃⁻¹.


Normalisation convention
───────────────────────
The data pipeline normalises each influence function ψ_k by
|m̂_k| before forming the outer product:
    ψ̃_k = ψ_k / |m̂_k|
    Σ̃ = (1/N) Σ_i ψ̃(z_i) ψ̃(z_i)'
    W = Σ̃⁻¹


Because W lives in normalised space, the deviation vector must
also be normalised:  g̃_k = (m_k − m̂_k) / |m̂_k|.
This ensures that Q(θ) = g̃' Σ̃⁻¹ g̃ is scale-invariant
and consistent with the diagonal-weight loss in compute_loss.


Mathematically, if D = diag(|m̂_k|), then:
    Σ̃ = D⁻¹ Σ_raw D⁻¹   ⟹   W = D Σ_raw⁻¹ D
    g̃' W g̃ = (D⁻¹ g)' (D Σ_raw⁻¹ D) (D⁻¹ g)
            = g' Σ_raw⁻¹ g
So the normalisation cancels and is equivalent to using the
raw optimal W with raw deviations — but numerically the
normalised form is far better conditioned.


Only moments with positive weight in spec.moments are included.
"""
function compute_loss_matrix(
    m_model::NamedTuple,
    spec::SMMSpec,
    W::Matrix{Float64}
) :: Float64


    # Build normalised deviation vector g̃_k = (m_k − m̂_k) / |m̂_k|
    dev_vec = Float64[]
    for k in keys(spec.moments)
        target = spec.moments[k]
        target.weight <= 0.0 && continue
        !hasproperty(m_model, k) && continue
        scale = max(abs(target.value), 1e-10)
        dev   = (getproperty(m_model, k) - target.value) / scale
        push!(dev_vec, dev)
    end


    isempty(dev_vec) && return 0.0


    # Compute Q = g̃' W g̃
    Q = dot(dev_vec, W * dev_vec)
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


If spec.W is not nothing, uses the full weight matrix via
compute_loss_matrix; otherwise uses diagonal weights via compute_loss.
"""
function smm_objective(
    θ_unc :: AbstractVector{Float64},
    spec  :: SMMSpec
) :: Float64


    cp, rp, up, sp = unpack_θ(θ_unc, spec)


    local model, solve_result
    try
        model, solve_result = solve_model(cp, rp, up, sp, spec.sim;
                                          Nx   = spec.run.Nx,
                                          Np_U = spec.run.Np_U,
                                          Np_S = spec.run.Np_S)
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


    # Use full weight matrix if available; otherwise diagonal weights
    if !isnothing(spec.W)
        return compute_loss_matrix(m_model, spec, spec.W)
    else
        return compute_loss(m_model, spec)
    end
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
    spec             :: SMMSpec;
    T0               :: Float64 = 2.0,
    step             :: Float64 = 0.15,
    max_iter         :: Int     = 5000,
    cooling_rate     :: Float64 = 1.0,
    cooling_exp      :: Float64 = 0.5,
    reheat_patience  :: Int     = 200,
    reheat_factor    :: Float64 = 2.0,
    max_reheats      :: Int     = 5,
    adapt_window     :: Int     = 50,
    target_fin       :: Float64 = 0.90,
    show_trace       :: Bool    = true,
    trace_stride     :: Int     = 100,
    rng                         = Random.default_rng(),
)
    theta      = pack_theta(spec)
    Q          = smm_objective(theta, spec)
    theta_best = copy(theta)
    Q_best     = isfinite(Q) ? Q : Inf
    Q_scale    = isfinite(Q) ? Q : Inf
    n_acc      = 0
    n_fin      = 0
    n_reheats  = 0


    steps_since_improvement = 0
    T_current = T0


    # Rolling window for adaptive step
    win_fin = adapt_window > 0 ? zeros(Bool, adapt_window) : Bool[]
    win_idx = 0


    actual_iters = 0


    if show_trace
        @printf("  [SA init]  Q0 = %s  T0=%.4f  step=%.4f\n",
                isfinite(Q) ? @sprintf("%.6e", Q) : "Inf (bad starting point)",
                T0, step)
        flush(stdout)
    end


    # t_local tracks steps since last reheat for per-segment cooling
    t_local    = 0
    T_reheat   = T0   # T at the start of current segment


    for t in 1:max_iter
        actual_iters = t
        # ── Cooling: each segment decays from its own T_reheat ─────────
        # T(t_local) = T_reheat / log(1 + rate * t_local)^exp
        # This means after a reheat the temperature decays smoothly from
        # the reheated value rather than being pinned at it forever.
        t_local  += 1
        T_current = T_reheat / (log(1.0 + cooling_rate * t_local))^cooling_exp
        T_current = max(T_current, 1e-8)


        # ── Proposal ──────────────────────────────────────────────────
        theta_prop = theta .+ step .* randn(rng, length(theta))
        Q_prop     = smm_objective(theta_prop, spec)


        is_fin = isfinite(Q_prop)
        is_fin && (n_fin += 1)


        # ── Adaptive step ─────────────────────────────────────────────
        if adapt_window > 0
            win_idx          = mod1(win_idx + 1, adapt_window)
            win_fin[win_idx] = is_fin
            if t >= adapt_window && t % adapt_window == 0
                fin_rate = mean(win_fin)
                if fin_rate < target_fin * 0.90
                    step *= 0.85
                elseif fin_rate > min(target_fin * 1.05, 0.99)
                    step *= 1.10
                end
                step = clamp(step, 0.01, 2.0)
            end
        end


        # ── Accept / reject ───────────────────────────────────────────
        if is_fin
            if !isfinite(Q_scale)
                Q_scale = Q_prop
            end
            accept = if !isfinite(Q)
                true
            elseif Q_prop <= Q
                true
            else
                rand(rng) < exp(-(Q_prop - Q) / (T_current * Q_scale))
            end


            if accept
                theta = theta_prop
                Q     = Q_prop
                n_acc += 1
                if Q < Q_best
                    Q_best     = Q
                    theta_best = copy(theta)
                    steps_since_improvement = 0
                else
                    steps_since_improvement += 1
                end
            else
                steps_since_improvement += 1
            end
        else
            steps_since_improvement += 1
        end


        # ── Early stop: reheats exhausted and still stagnating ──────────
        if reheat_patience > 0 &&
           max_reheats > 0 && n_reheats >= max_reheats &&
           steps_since_improvement >= reheat_patience
            if show_trace
                @printf("  [SA EARLY STOP  iter=%5d]  reheats exhausted, no improvement for %d steps, Q_best=%.6e\n",
                        t, steps_since_improvement, Q_best)
                flush(stdout)
            end
            break
        end


        # ── Reheating on stagnation ────────────────────────────────────
        if reheat_patience > 0 &&
           steps_since_improvement >= reheat_patience &&
           (max_reheats == 0 || n_reheats < max_reheats)


            n_reheats += 1
            T_before   = T_current
            T_current  = T_current * reheat_factor
            T_reheat   = T_current   # new segment starts here
            t_local    = 0           # reset local clock so decay starts fresh
            theta      = copy(theta_best)
            Q          = Q_best
            steps_since_improvement = 0


            if show_trace
                @printf("  [SA REHEAT #%d  iter=%5d]  T %.4f→%.4f  restarting from Q_best=%.6e\n",
                        n_reheats, t, T_before, T_current, Q_best)
                flush(stdout)
            end
        end


        # ── Progress trace ────────────────────────────────────────────
        if show_trace && t % trace_stride == 0
            @printf("  [SA iter=%5d]  curr=%-14s  best=%.6e  T=%.4f  step=%.4f  acc=%.2f  fin=%.2f  reheats=%d\n",
                    t,
                    isfinite(Q) ? @sprintf("%.6e", Q) : "Inf",
                    Q_best, T_current, step,
                    n_acc / t,
                    n_fin / t,
                    n_reheats)
            flush(stdout)
        end
    end


    if show_trace
        @printf("  [SA done]  Q_best=%.6e  accepted %d/%d  finite %d/%d  reheats=%d\n",
                Q_best, n_acc, actual_iters, n_fin, actual_iters, n_reheats)
        flush(stdout)
    end


    return theta_best, Q_best, actual_iters
end


# Alias so pack_theta works (smm_params.jl defines pack_θ with Unicode)
pack_theta(spec) = pack_θ(spec)



# ============================================================
# Differential Evolution — internal helper
# ============================================================


"""
    _pick3(rng, n, exclude) -> (a, b, c)


Pick three distinct indices from 1:n, all different from `exclude`,
using a rejection sampler.  Expected draws per call ≈ 3 + O(1/n),
much cheaper than shuffling all n−1 candidates.
"""
@inline function _pick3(rng, n::Int, exclude::Int)
    a = exclude
    while a == exclude
        a = rand(rng, 1:n)
    end
    b = exclude
    while b == exclude || b == a
        b = rand(rng, 1:n)
    end
    c = exclude
    while c == exclude || c == a || c == b
        c = rand(rng, 1:n)
    end
    return a, b, c
end



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
    patience     :: Int     = 20,
    avg_tol      :: Float64 = 0.01,   # stop when (Q_mean−Q_best)/|Q_best| < tol; 0 = off
    show_members :: Bool    = false,
    show_gens    :: Bool    = true,
    trace_stride :: Int     = 10,
    rng                     = Random.default_rng(),
)
    npar     = length(spec.free)
    pop_size = pop_size > 0 ? pop_size : 10 * npar
    theta0   = pack_theta(spec)


    # ── Initialise population — uniform in constrained space ──────────────
    # Draw x_k ~ Uniform(lb_k, ub_k) for each parameter independently,
    # then map to logit space.  This gives a flat density over the
    # feasible rectangle, unlike uniform-in-logit which is U-shaped
    # (dense near boundaries, sparse in the interior).
    pop = Vector{Vector{Float64}}(undef, pop_size)
    for j in 1:pop_size
        theta_j = Vector{Float64}(undef, npar)
        for (k, ps) in enumerate(spec.free)
            x_k = ps.lb + (ps.ub - ps.lb) * rand(rng)
            # Clamp away from exact bounds to avoid ±Inf in logit
            x_k = clamp(x_k, ps.lb + 1e-8 * (ps.ub - ps.lb),
                            ps.ub - 1e-8 * (ps.ub - ps.lb))
            theta_j[k] = _to_unconstrained(x_k, ps.lb, ps.ub)
        end
        pop[j] = theta_j
    end
    pop[1] = copy(theta0)   # first member is still the supplied starting point


    # Evaluate initial population — Inf for non-converging members
    Q_pop = fill(Inf, pop_size)


    if show_gens
        @printf("  [DE init]  evaluating %d initial members...\n", pop_size)
        flush(stdout)
    end


    n_feasible = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:pop_size
        Q_pop[i] = smm_objective(pop[i], spec)
        if isfinite(Q_pop[i])
            Threads.atomic_add!(n_feasible, 1)
        end
        if show_members && i % trace_stride == 0
            @printf("  [DE init]  evaluated ~%d/%d  feasible so far: ~%d\n",
                    i, pop_size, n_feasible[])
            flush(stdout)
        end
    end
    # Print final summary after all threads have joined
    if show_members
        @printf("  [DE init]  evaluated %d/%d  feasible: %d\n",
                pop_size, pop_size, n_feasible[])
        flush(stdout)
    end


    i_best     = argmin(Q_pop)
    Q_best     = Q_pop[i_best]
    theta_best = copy(pop[i_best])


    if show_gens
        @printf("  [DE init]  feasible=%d/%d  Q_best=%.6e\n",
                n_feasible[], pop_size, Q_best)
        flush(stdout)
    end


    # ── Main DE loop (generations) ─────────────────────────────────────
    n_evals    = Threads.Atomic{Int}(pop_size)
    stagnation = 0
    actual_gens = 0
    best_lock  = ReentrantLock()

    for gen in 1:max_iter
        actual_gens = gen
        n_improved  = Threads.Atomic{Int}(0)

        Threads.@threads for i in 1:pop_size
            # ── Pick three distinct donors ≠ i ─────────────────────────
            ia, ib, ic = _pick3(Random.default_rng(), pop_size, i)
            a, b, c    = pop[ia], pop[ib], pop[ic]

            # Mutant vector
            v = a .+ f .* (b .- c)

            # Binomial crossover — ensure at least one dimension from mutant
            mask    = rand(Random.default_rng(), npar) .< cr
            j_force = rand(Random.default_rng(), 1:npar)
            mask[j_force] = true
            u = ifelse.(mask, v, pop[i])

            # Evaluate trial vector
            Q_u = smm_objective(u, spec)
            Threads.atomic_add!(n_evals, 1)

            # Greedy selection — Inf proposals never win
            if isfinite(Q_u) && Q_u < Q_pop[i]
                pop[i]   = u
                Q_pop[i] = Q_u
                Threads.atomic_add!(n_improved, 1)

                # Update global best under lock — two fields must stay in sync
                if Q_u < Q_best
                    lock(best_lock) do
                        if Q_u < Q_best          # re-check inside lock
                            Q_best     = Q_u
                            theta_best = copy(u)
                        end
                    end
                end
            end

            # Within-generation progress
            if show_members && i % trace_stride == 0
                Q_i = Q_pop[i]
                @printf("  [DE gen=%4d  member=%4d/%4d]  Q_member=%-14s  improved=%d\n",
                        gen, i, pop_size,
                        isfinite(Q_i) ? @sprintf("%.6e", Q_i) : "Inf",
                        n_improved[])
                flush(stdout)
            end
        end

        # Read atomics once after all threads have joined
        n_imp  = n_improved[]
        n_eval = n_evals[]

        # Stagnation tracking
        if n_imp == 0
            stagnation += 1
        else
            stagnation = 0
        end

        # End-of-generation summary
        if show_gens
            Q_mean = mean(filter(isfinite, Q_pop))
            n_feas = count(isfinite, Q_pop)
            @printf("  [DE gen=%4d DONE]  Q_best=%.6e  Q_mean=%.6e  feasible=%d/%d  improved=%d  stagnation=%d/%d  evals=%d\n",
                    gen, Q_best, Q_mean, n_feas, pop_size, n_imp, stagnation, patience, n_eval)
            flush(stdout)
        end

        # Early stopping — stagnation
        if stagnation >= patience
            show_gens && @printf("  [DE]  early stop: no improvement for %d generations\n", patience)
            flush(stdout)
            break
        end

        # Early stopping — population convergence around best
        if avg_tol > 0.0 && isfinite(Q_best) && Q_best != 0.0
            Q_finite = filter(isfinite, Q_pop)
            if !isempty(Q_finite)
                rel_gap = (mean(Q_finite) - Q_best) / abs(Q_best)
                if rel_gap < avg_tol
                    show_gens && @printf("  [DE]  early stop: Q_mean within %.1e of Q_best (rel gap = %.4e)\n",
                                         avg_tol, rel_gap)
                    flush(stdout)
                    break
                end
            end
        end
    end

    if show_gens
        @printf("  [DE done]  Q_best=%.6e  total evals=%d\n", Q_best, n_evals[])
        flush(stdout)
    end

    return theta_best, Q_best, actual_gens
end



# ============================================================
# Main optimisation entry point
# ============================================================


"""
    run_smm(spec; method=:de, rng=default_rng()) -> SMMResult


Run SMM estimation. All settings (grid sizes, DE parameters,
Nelder-Mead parameters, tracing) are read from `spec.run`.


  :de          Differential evolution — global search (default).
  :sa          Simulated annealing — alternative global search.
  :neldermead  Nelder-Mead — local polish after :de or :sa.


Typical workflow (both stages read their settings from spec.run):
  res_de  = run_smm(spec; method=:de)
  res_pol = run_smm(_spec_with_init(spec, res_de.theta_opt); method=:neldermead)
"""
function run_smm(
    spec   :: SMMSpec;
    method :: Symbol = :de,
    rng             = Random.default_rng(),
) :: SMMResult


    r    = spec.run     # shorthand
    npar = length(spec.free)


    @printf("\nStarting SMM  (%s,  %d free params)\n", method, npar)
    flush(stdout)


    if method == :de
        theta_opt, loss_opt, niters = _run_de(
            spec;
            max_iter     = r.de_max_iter,
            pop_size     = r.de_pop_size > 0 ? r.de_pop_size : 100 * npar,
            f            = r.de_f,
            cr           = r.de_cr,
            patience     = r.de_patience,
            avg_tol      = r.de_avg_tol,
            show_members = r.show_trace_members,
            show_gens    = r.show_trace_generations,
            trace_stride = r.trace_stride,
            rng          = rng,
        )
        converged = isfinite(loss_opt)


    elseif method == :sa
        theta_opt, loss_opt, niters = _run_sa(
            spec;
            max_iter        = r.sa_max_iter,
            T0              = r.sa_T0,
            step            = r.sa_step,
            cooling_rate    = r.sa_cooling_rate,
            cooling_exp     = r.sa_cooling_exp,
            reheat_patience = r.sa_reheat_patience,
            reheat_factor   = r.sa_reheat_factor,
            max_reheats     = r.sa_max_reheats,
            adapt_window    = r.sa_adapt_window,
            target_fin      = r.sa_target_fin,
            show_trace      = r.show_trace_generations,
            trace_stride    = r.trace_stride,
            rng             = rng,
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
            if r.show_trace_generations && iter_count[] % r.trace_stride == 0
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


        options   = Optim.Options(iterations = r.nm_max_iter,
                                  f_tol      = r.nm_f_tol,
                                  x_tol      = r.nm_x_tol,
                                  show_trace = false)
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
    multistart_smm(spec, n_starts; method=:de, seed=42) -> SMMResult


Run `n_starts` searches from randomly perturbed starting points
and return the best result.  All optimiser settings come from
`spec.run` as usual — only the starting point differs across starts.
"""
function multistart_smm(
    spec     :: SMMSpec,
    n_starts :: Int;
    method   :: Symbol = :de,
    seed     :: Int    = 42,
) :: SMMResult


    rng    = Random.MersenneTwister(seed)
    theta0 = pack_theta(spec)
    npar   = length(theta0)


    best_result = nothing


    for s in 1:n_starts
        @printf("\n══ Multi-start %d / %d ══\n", s, n_starts)
        flush(stdout)


        theta_start = theta0 .+ 0.5 .* randn(rng, npar)
        spec_s      = _spec_with_init(spec, theta_start)


        try
            res_s = run_smm(spec_s; method = method, rng = rng)
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
    return SMMSpec(new_free, spec.fixed, spec.moments, spec.sim, spec.run, spec.W)
end
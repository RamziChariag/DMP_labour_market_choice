############################################################
# smm.jl — SMM objective and optimisation loops (v7)
#
# Main entry points
#   run_smm(spec; method, rng)         single-run estimation
#   multistart_smm(spec, n; method)    repeated starts, keep best
#
# Methods
#   :de           differential evolution (default global search)
#   :sa           simulated annealing
#   :neldermead   Nelder–Mead from Optim.jl (local polish)
#   :lbfgs, :bfgs gradient-based polish
#
# SA is implemented here rather than delegated to Optim.jl because
# Optim's SA rejects Inf proposals silently and stalls in regions
# where many parameter draws produce non-converging models.  This
# loop handles Inf correctly (always reject, never update).
#
# v7 note: the SMM objective operates on the stationary-equilibrium
# moment vector (24 moments). The discrete cross-market policy d(x)
# is identically zero in stationary equilibrium and is therefore not
# subject to any feasibility filter here; if the solver returns a
# non-zero d for some parameter draw, the resulting moments still
# enter the objective normally.
############################################################

"""
    _load_smm_bundle(path; delete_on_fail=false, label="file") → Union{Nothing, NamedTuple}

Safely deserialise an SMM .jls bundle of the form
    (result = ::SMMResult, spec = ::SMMSpec, sim = ::SimParams).
Returns the bundle on success, or `nothing` on failure / stale format.
"""
function _load_smm_bundle(path::String; delete_on_fail::Bool=false, label::String="file")
    if !isfile(path)
        return nothing
    end

    data = try
        open(deserialize, path)
    catch e
        @warn "Failed to deserialize $label (stale format — will overwrite): $e"
        if delete_on_fail
            rm(path, force=true)
        end
        return nothing
    end

    ok = false
    if data isa NamedTuple
        ok = haskey(data, :result) && haskey(data, :spec)
    end

    if !ok
        @warn "Invalid $label format at $path (missing :result or :spec) — treating as stale"
        if delete_on_fail
            rm(path, force=true)
        end
        return nothing
    end

    if isnothing(data.result) || isnothing(data.spec)
        @warn "Invalid $label contents at $path (:result or :spec is nothing) — treating as stale"
        if delete_on_fail
            rm(path, force=true)
        end
        return nothing
    end

    return data
end


# ============================================================
# Weighted loss
# ============================================================

"""
    compute_loss(m_model, spec) → Float64

    Q(θ) = Σ_k  w_k · [(m_k^model − m̂_k) / |m̂_k|]²

Diagonal-weight loss with scale-normalised deviations.  Each
deviation is divided by max(|m̂_k|, 1e-10) so moments of different
magnitudes contribute on comparable scales.  Only moments with
positive weight in spec.moments are included.
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
    compute_loss_matrix(m_model, spec, W) → Float64

    Q(θ) = g̃(θ)' W g̃(θ)     where g̃_k = (m_k^model − m̂_k) / |m̂_k|

Full-matrix loss using the optimal weight matrix W = Σ̃⁻¹.  The
deviation vector is normalised by |m̂_k| because the influence
functions in the data pipeline are normalised the same way.
"""
function compute_loss_matrix(
    m_model::NamedTuple,
    spec::SMMSpec,
    W::Matrix{Float64}
) :: Float64

    dev_vec = Float64[]
    for k in keys(spec.moments)
        target = spec.moments[k]
        target.weight <= 0.0 && continue
        !hasproperty(m_model, k) && continue
        push!(dev_vec, getproperty(m_model, k) - target.value)
    end

    isempty(dev_vec) && return 0.0

    if size(W, 1) != length(dev_vec) || size(W, 2) != length(dev_vec)
        error(
            "compute_loss_matrix: W is $(size(W,1))×$(size(W,2)) but deviation vector " *
            "has length $(length(dev_vec)). The W matrix in spec.W is stale — rebuild via " *
            "build_smm_spec with the correct W from load_weight_matrix(..., skip_moments=SKIP_MOMENTS)."
        )
    end

    Q = dot(dev_vec, W * dev_vec)
    return Q
end


# ============================================================
# SMM objective
# ============================================================

"""
    smm_objective(θ_unc, spec) → Float64

Solve the model at parameters decoded from `θ_unc` and return Q(θ).
Returns Inf (never throws) on any failure or non-convergence.  The
solver runs silently regardless of spec.sim.verbose.

Degenerate τ(x) profiles (all train, none train, non-monotone, or
multiple jumps) are rejected as infeasible.
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

    emptol = 1e-12
    if obj_eq.agg_eU < emptol || obj_eq.agg_eS < emptol
        return Inf
    end

    τ  = obj_eq.tauT
    τv = vec(τ)
    if all(iszero, τv) || all(isone, τv)
        return Inf
    end
    if !all(t -> t == 0 || t == 1, τv)
        return Inf
    end
    dτ = diff(τv)
    if any(dτ .< 0) || count(!iszero, dτ) > 1
        return Inf
    end

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
# Corner count (params within `tol` of either bound, in constrained space)
# ============================================================

"""
    _count_corners(theta_unc, spec; tol=0.02) → Int

Count free parameters whose constrained value lies within `tol`
(fraction of bound width) of either `lb` or `ub`.
"""
function _count_corners(
    theta_unc :: AbstractVector{Float64},
    spec      :: SMMSpec;
    tol       :: Float64 = 0.02,
) :: Int
    n = 0
    for (i, ps) in enumerate(spec.free)
        x     = _to_constrained(theta_unc[i], ps.lb, ps.ub)
        width = ps.ub - ps.lb
        width <= 0.0 && continue
        if (x - ps.lb) / width < tol || (ps.ub - x) / width < tol
            n += 1
        end
    end
    return n
end


# ============================================================
# Simulated annealing
# ============================================================

"""
    _run_sa(spec; ...) → (theta_best, Q_best, iters)

Simulated annealing in unconstrained (logit) space.
"""
function _run_sa(
    spec             :: SMMSpec;
    T0               :: Float64 = 0.0,
    step             :: Float64 = 0.15,
    max_iter         :: Int     = 5000,
    cooling_rate     :: Float64 = 1.0,
    cooling_exp      :: Float64 = 0.5,
    reheat_patience  :: Int     = 200,
    reheat_factor    :: Float64 = 2.0,
    max_reheats      :: Int     = 5,
    adapt_window     :: Int     = 50,
    target_fin       :: Float64 = 0.90,
    random_init      :: Bool    = false,
    show_trace       :: Bool    = true,
    trace_stride     :: Int     = 100,
    rng                         = Random.default_rng(),
)
    theta = if random_init
        theta_j = Vector{Float64}(undef, length(spec.free))
        for (k, ps) in enumerate(spec.free)
            x_k = ps.lb + (ps.ub - ps.lb) * rand(rng)
            x_k = clamp(x_k, ps.lb + 1e-8 * (ps.ub - ps.lb),
                             ps.ub - 1e-8 * (ps.ub - ps.lb))
            theta_j[k] = _to_unconstrained(x_k, ps.lb, ps.ub)
        end
        theta_j
    else
        pack_theta(spec)
    end

    Q          = smm_objective(theta, spec)
    theta_best = copy(theta)
    Q_best     = isfinite(Q) ? Q : Inf
    n_acc      = 0
    n_fin      = 0
    n_reheats  = 0

    steps_since_improvement = 0

    if T0 <= 0.0
        dQ_pos = Float64[]
        for _ in 1:50
            θ_probe = theta .+ 0.05 .* randn(rng, length(theta))
            Q_probe = smm_objective(θ_probe, spec)
            isfinite(Q_probe) && isfinite(Q) && Q_probe > Q && push!(dQ_pos, Q_probe - Q)
        end
        T0 = isempty(dQ_pos) ? 100.0 : -median(dQ_pos) / log(0.80)
        if show_trace
            @printf("  [SA T0 auto]  T0 = %.4f  (from %d uphill probes)\n",
                    T0, length(dQ_pos))
            flush(stdout)
        end
    end

    T_current = T0
    T_reheat  = T0
    t_local   = 0

    win_fin = adapt_window > 0 ? zeros(Bool, adapt_window) : Bool[]
    win_acc = adapt_window > 0 ? zeros(Bool, adapt_window) : Bool[]
    win_idx = 0

    actual_iters = 0

    if show_trace
        n_corners_init = _count_corners(theta_best, spec)
        @printf("  [SA init]  Q0 = %s  T0=%.4f  step=%.4f  corners=%d/%d  random_init=%s\n",
                isfinite(Q) ? @sprintf("%.6e", Q) : "Inf (bad starting point)",
                T0, step,
                n_corners_init, length(spec.free),
                random_init)
        flush(stdout)
    end

    for t in 1:max_iter
        actual_iters = t

        t_local  += 1
        T_current = T_reheat / (log(1.0 + cooling_rate * t_local))^cooling_exp
        T_current = max(T_current, 1e-8)

        theta_prop = theta .+ step .* randn(rng, length(theta))
        Q_prop     = smm_objective(theta_prop, spec)

        is_fin = isfinite(Q_prop)
        is_fin && (n_fin += 1)

        accept = false
        if is_fin
            accept = if !isfinite(Q)
                true
            elseif Q_prop <= Q
                true
            else
                rand(rng) < exp(-(Q_prop - Q) / T_current)
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

        if adapt_window > 0
            win_idx          = mod1(win_idx + 1, adapt_window)
            win_fin[win_idx] = is_fin
            win_acc[win_idx] = is_fin && accept

            if t >= adapt_window && t % adapt_window == 0
                fin_rate = mean(win_fin)
                acc_rate = mean(win_acc)

                if fin_rate < target_fin * 0.90
                    step *= 0.85
                elseif acc_rate < 0.15
                    step *= 0.85
                elseif acc_rate > 0.35
                    step *= 1.10
                end
                step = clamp(step, 0.01, 2.0)
            end
        end

        if reheat_patience > 0 &&
           max_reheats > 0 && n_reheats >= max_reheats &&
           steps_since_improvement >= reheat_patience
            if show_trace
                n_corners_es = _count_corners(theta_best, spec)
                @printf("  [SA EARLY STOP  iter=%5d]  reheats exhausted, no improvement for %d steps, Q_best=%.6e  corners=%d/%d\n",
                        t, steps_since_improvement, Q_best,
                        n_corners_es, length(spec.free))
                flush(stdout)
            end
            break
        end

        if reheat_patience > 0 &&
           steps_since_improvement >= reheat_patience &&
           (max_reheats == 0 || n_reheats < max_reheats)

            n_reheats += 1
            T_before   = T_current
            T_current  = T_current * reheat_factor
            T_reheat   = T_current
            t_local    = 0
            theta      = copy(theta_best)
            Q          = Q_best
            steps_since_improvement = 0

            if show_trace
                n_corners_rh = _count_corners(theta_best, spec)
                @printf("  [SA REHEAT #%d  iter=%5d]  T %.4f→%.4f  restarting from Q_best=%.6e  corners=%d/%d\n",
                        n_reheats, t, T_before, T_current, Q_best,
                        n_corners_rh, length(spec.free))
                flush(stdout)
            end
        end

        if show_trace && t % trace_stride == 0
            w_acc = adapt_window > 0 && t >= adapt_window ? mean(win_acc) : n_acc / t
            w_fin = adapt_window > 0 && t >= adapt_window ? mean(win_fin) : n_fin / t
            n_corners = _count_corners(theta_best, spec)
            @printf("  [SA iter=%5d]  curr=%-14s  best=%.6e  T=%.4f  step=%.4f  acc=%.2f  fin=%.2f  corners=%d/%d  reheats=%d\n",
                    t,
                    isfinite(Q) ? @sprintf("%.6e", Q) : "Inf",
                    Q_best, T_current, step,
                    w_acc, w_fin,
                    n_corners, length(spec.free),
                    n_reheats)
            flush(stdout)
        end
    end

    if show_trace
        n_corners_done = _count_corners(theta_best, spec)
        @printf("  [SA done]  Q_best=%.6e  accepted %d/%d  finite %d/%d  corners=%d/%d  reheats=%d\n",
                Q_best, n_acc, actual_iters, n_fin, actual_iters,
                n_corners_done, length(spec.free),
                n_reheats)
        flush(stdout)
    end

    return theta_best, Q_best, actual_iters
end


# Alias for the unicode pack_θ defined in smm_params.jl
pack_theta(spec) = pack_θ(spec)


# ============================================================
# Differential evolution
# ============================================================

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


"""
    _count_basins(pop, Q_pop, spec; min_size) → Int

Count distinct parameter-space basins among the feasible members
using complete-linkage hierarchical clustering on pairwise Euclidean
distances in [0, 1]^d-normalised constrained space.
"""
function _count_basins(
    pop      :: Vector{Vector{Float64}},
    Q_pop    :: Vector{Float64},
    spec     :: SMMSpec;
    min_size :: Int = 5,
) :: Int
    feas_idx = findall(isfinite, Q_pop)
    n = length(feas_idx)
    n < 2 * min_size && return 0

    npar = length(spec.free)

    X = Matrix{Float64}(undef, n, npar)
    for (row, i) in enumerate(feas_idx)
        θ = pop[i]
        for (k, ps) in enumerate(spec.free)
            x_k        = _to_constrained(θ[k], ps.lb, ps.ub)
            X[row, k]  = (x_k - ps.lb) / (ps.ub - ps.lb)
        end
    end

    D = zeros(Float64, n, n)
    for i in 1:n
        for j in i+1:n
            d = 0.0
            for k in 1:npar
                d += (X[i,k] - X[j,k])^2
            end
            D[i,j] = sqrt(d)
            D[j,i] = D[i,j]
        end
    end

    hc = hclust(D; linkage = :complete)

    h       = hc.heights
    gaps    = diff(h)
    gap_idx = argmax(gaps)
    cut_h   = (h[gap_idx] + h[gap_idx + 1]) / 2.0

    labels  = cutree(hc; h = cut_h)
    counts  = zeros(Int, maximum(labels))
    for l in labels
        counts[l] += 1
    end
    return count(c -> c >= min_size, counts)
end


"""
    _run_de(spec; ...) → (theta_best, Q_best, iters)

DE/rand/1/bin in unconstrained (logit) space.
"""
function _run_de(
    spec         :: SMMSpec;
    max_iter     :: Int     = 5000,
    pop_size     :: Int     = 0,
    f            :: Float64 = 0.65,
    cr           :: Float64 = 0.85,
    patience     :: Int     = 20,
    avg_tol      :: Float64 = 0.01,
    show_members :: Bool    = false,
    show_gens    :: Bool    = true,
    trace_stride :: Int     = 10,
    rng                     = Random.default_rng(),
)
    npar     = length(spec.free)
    pop_size = pop_size > 0 ? pop_size : 10 * npar
    theta0   = pack_theta(spec)

    member_rngs = begin
        seeds = rand(rng, UInt64, pop_size)
        [Random.Xoshiro(s) for s in seeds]
    end

    pop = Vector{Vector{Float64}}(undef, pop_size)
    for j in 1:pop_size
        theta_j = Vector{Float64}(undef, npar)
        for (k, ps) in enumerate(spec.free)
            x_k = ps.lb + (ps.ub - ps.lb) * rand(rng)
            x_k = clamp(x_k,
                        ps.lb + 1e-8 * (ps.ub - ps.lb),
                        ps.ub - 1e-8 * (ps.ub - ps.lb))
            theta_j[k] = _to_unconstrained(x_k, ps.lb, ps.ub)
        end
        pop[j] = theta_j
    end
    pop[1] = copy(theta0)

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

    n_evals     = Threads.Atomic{Int}(pop_size)
    stagnation  = 0
    actual_gens = 0

    for gen in 1:max_iter
        actual_gens = gen
        n_improved  = Threads.Atomic{Int}(0)

        pop_old = pop
        Q_old   = Q_pop

        pop_new = Vector{Vector{Float64}}(undef, pop_size)
        Q_new   = Vector{Float64}(undef, pop_size)

        Threads.@threads for i in 1:pop_size
            rng_i = member_rngs[i]

            ia, ib, ic = _pick3(rng_i, pop_size, i)
            a, b, c    = pop_old[ia], pop_old[ib], pop_old[ic]

            v = a .+ f .* (b .- c)

            mask    = rand(rng_i, npar) .< cr
            j_force = rand(rng_i, 1:npar)
            mask[j_force] = true
            u = ifelse.(mask, v, pop_old[i])

            Q_u = smm_objective(u, spec)
            Threads.atomic_add!(n_evals, 1)

            if isfinite(Q_u) && Q_u < Q_old[i]
                pop_new[i] = u
                Q_new[i]   = Q_u
                Threads.atomic_add!(n_improved, 1)
            else
                pop_new[i] = pop_old[i]
                Q_new[i]   = Q_old[i]
            end

            if show_members && i % trace_stride == 0
                Q_i = Q_new[i]
                @printf("  [DE gen=%4d  member=%4d/%4d]  Q_member=%-14s  improved=%d\n",
                        gen, i, pop_size,
                        isfinite(Q_i) ? @sprintf("%.6e", Q_i) : "Inf",
                        n_improved[])
                flush(stdout)
            end
        end

        pop   = pop_new
        Q_pop = Q_new

        i_best     = argmin(Q_pop)
        Q_best     = Q_pop[i_best]
        theta_best = copy(pop[i_best])

        n_imp  = n_improved[]
        n_eval = n_evals[]

        if n_imp == 0
            stagnation += 1
        else
            stagnation = 0
        end

        if show_gens
            Q_finite = filter(isfinite, Q_pop)
            Q_mean   = isempty(Q_finite) ? Inf : mean(Q_finite)
            n_feas   = length(Q_finite)
            n_bas    = n_feas == 0 ? 0 : _count_basins(pop, Q_pop, spec)
            n_corners = _count_corners(theta_best, spec)
            @printf("  [DE gen=%4d DONE]  Q_best=%.6e Q_mean=%-14s  feasible=%d/%d  improved=%d  clusters=%d  corners=%d/%d  evals=%d\n",
                    gen,
                    Q_best,
                    isfinite(Q_mean) ? @sprintf("%.6e", Q_mean) : "Inf",
                    n_feas, pop_size, n_imp, n_bas,
                    n_corners, length(spec.free),
                    n_eval)
            flush(stdout)
        end

        if stagnation >= patience
            show_gens && @printf("  [DE]  early stop: no improvement for %d generations\n", patience)
            flush(stdout)
            break
        end

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
    run_smm(spec; method=:de, rng=default_rng()) → SMMResult

Run SMM estimation.  All settings come from `spec.run`.
"""
function run_smm(
    spec   :: SMMSpec;
    method :: Symbol = :de,
    rng             = Random.default_rng(),
) :: SMMResult

    r    = spec.run
    npar = length(spec.free)

    @printf("\nStarting SMM  (%s,  %d free params)\n", method, npar)
    flush(stdout)

    if method == :de
        theta_opt, loss_opt, niters = _run_de(
            spec;
            max_iter     = r.de_max_iter,
            pop_size     = r.de_pop_size > 0 ? r.de_pop_size : 10 * npar,
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
            random_init     = r.sa_random_init,
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
    multistart_smm(spec, n_starts; method=:sa, seed=42) → SMMResult
"""
function multistart_smm(
    spec     :: SMMSpec,
    n_starts :: Int;
    method   :: Symbol = :sa,
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
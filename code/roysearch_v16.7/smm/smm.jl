############################################################
# smm.jl — SMM objective and optimisation loops
#
# Main entry points
#   run_smm(spec; method, rng,
#           seed_bank, prev_optimum)   single-run estimation; seed_bank seeds
#                                      SA/DE from candidate clusters
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
# The SMM objective operates on the stationary-equilibrium moment
# vector. The discrete cross-market policy d(x) is identically zero
# in stationary equilibrium and is therefore not subject to any
# feasibility filter here; if the solver returns a non-zero d for some
# parameter draw, the resulting moments still enter the objective
# normally.
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
    active_moment_keys(spec) -> Vector{Symbol}

Keys of the moments the objective actually scores, in the order the deviation
vector (and hence `spec.W`) is built.  Single source of truth for that order:
`compute_loss_matrix` and `smm_objective`'s `moments_out` both use it.
"""
active_moment_keys(spec::SMMSpec) =
    Symbol[k for k in keys(spec.moments) if spec.moments[k].weight > 0.0]


"""
    compute_loss_matrix(m_model, spec, W) → Float64

    Q(θ) = g(θ)' W g(θ) / q_scale,   where g_k = m_k^model − m̂_k   (RAW deviations)

Single weighted-loss path for every weighting scheme.  The deviation
vector is in RAW moment units (no |m̂_k| division); all per-moment
scaling and cross-moment weighting live entirely in `W`:

  • Equal weight (relative):  W = Diagonal(weight_k / m̂_k²)
        ⟹ g' W g = Σ_k weight_k · (g_k / m̂_k)²
        i.e. the scale-normalised relative-deviation loss.
  • Diagonal-σ:                W = Diagonal(1 / σ̂²_samp,k)

`W` is built once, outside the hot loop, by `build_smm_spec`
(equal-weight case) or `load_weight_matrix` (diagonal-σ).

`spec.q_scale` is a DISPLAY-ONLY positive constant (default 1.0) that
divides the reported Q.  Because it is constant across all θ, dividing
Q by it does NOT move the argmin — it only rescales the reported number
to a human-readable magnitude.  The optimiser, gradients, and acceptance
ratios all see the same rescaled-by-a-constant surface, so optimisation
is mathematically identical to the un-normalised objective.
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
    return Q / spec.q_scale
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
    spec  :: SMMSpec;
    Nx    :: Int = spec.run.Nx,
    Np_U  :: Int = spec.run.Np_U,
    Np_S  :: Int = spec.run.Np_S,
    moments_out :: Union{Nothing,AbstractVector{Float64}} = nothing,
) :: Float64

    cp, up, sp = unpack_θ(θ_unc, spec)

    local model, solve_result
    try
        model, solve_result = solve_model(cp, up, sp, spec.sim;
                                          Nx   = Nx,
                                          Np_U = Np_U,
                                          Np_S = Np_S)
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

    # Reject a degenerate training margin: with the whole ability grid on one
    # side of the frontier (nobody trains, or everybody trains) training_share
    # carries no identifying variation.  The frontier τ(a_U,a_S) is a 2D
    # indicator, so the interior is checked directly.
    τv = vec(obj_eq.τ_mat)
    if all(iszero, τv) || all(isone, τv)
        return Inf
    end

    # Hand back the scored moments when the caller supplies a buffer (the MCMC
    # driver needs them per draw for the Jacobian regression); order matches
    # spec.W via active_moment_keys.
    if moments_out !== nothing
        ks = active_moment_keys(spec)
        @assert length(moments_out) == length(ks)
        @inbounds for (i, k) in enumerate(ks)
            moments_out[i] = getproperty(m_model, k)
        end
    end

    # spec.W is always present (equal-weight is a diagonal W built in
    # build_smm_spec), so there is a single weighted-loss path.
    return compute_loss_matrix(m_model, spec, spec.W)
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


"""
    _corner_tags(theta_unc, spec; tol=0.02) → String

Inline companion to `_count_corners`: lists which free parameters sit within
`tol` of a bound and whether it is the lower or upper bound, e.g.
` [skl:μ(lower), unsk:k(upper)]`.  Returns "" when nothing is cornered, so it
can be appended straight after a `corners=%d/%d` field with a single `%s`.
"""
function _corner_tags(
    theta_unc :: AbstractVector{Float64},
    spec      :: SMMSpec;
    tol       :: Float64 = 0.02,
) :: String
    parts = String[]
    for (i, ps) in enumerate(spec.free)
        x     = _to_constrained(theta_unc[i], ps.lb, ps.ub)
        width = ps.ub - ps.lb
        width <= 0.0 && continue
        if (x - ps.lb) / width < tol
            push!(parts, "$(ps.block):$(ps.name)(lower)")
        elseif (ps.ub - x) / width < tol
            push!(parts, "$(ps.block):$(ps.name)(upper)")
        end
    end
    return isempty(parts) ? "" : " [" * join(parts, ", ") * "]"
end


# ============================================================
# Simulated annealing
# ============================================================

"""
    _random_theta(spec, rng) → Vector{Float64}

A single random start in unconstrained space: draw each free parameter
uniformly within its bounds, then map to the logit scale.
"""
function _random_theta(spec::SMMSpec, rng)
    theta_j = Vector{Float64}(undef, length(spec.free))
    for (k, ps) in enumerate(spec.free)
        x_k = ps.lb + (ps.ub - ps.lb) * rand(rng)
        x_k = clamp(x_k, ps.lb + 1e-8 * (ps.ub - ps.lb),
                         ps.ub - 1e-8 * (ps.ub - ps.lb))
        theta_j[k] = _to_unconstrained(x_k, ps.lb, ps.ub)
    end
    return theta_j
end


"""
    RateStop(tol, span)

Stop when the incumbent improves at less than `tol` per 100 evaluations, sustained
over `span` PRODUCTIVE evaluations.

The accounting is continuous rather than windowed, and the distinction that makes it
work is between an evaluation that improved the incumbent and one that did not.  A
flat stretch PAUSES the budget — neither advancing nor resetting it — because in this
many dimensions a run can sit against an infeasible wall for thousands of evaluations
and then recover; counting those would stop a descent that has not finished, and
resetting on them would never stop at all.  The position mark advances on every call,
so paused evaluations are discarded rather than deferred: without that the next
productive evaluation absorbs the whole paused gap and the pause is counted after all.
"""
mutable struct RateStop
    tol      :: Float64
    span     :: Int
    mark_Q   :: Float64   # incumbent when the current budget started
    mark_it  :: Int       # evaluation index at the last call
    slow     :: Int       # productive evaluations spent below tol
    last_Q   :: Float64
end
RateStop(tol::Float64, span::Int) = RateStop(tol, span, Inf, 0, 0, Inf)

"""
    rate_stop!(rs, Q_best, iter) -> Bool

Feed the current incumbent and evaluation count; true when the rule fires.
"""
function rate_stop!(rs::RateStop, Q_best::Float64, iter::Int)
    (rs.tol > 0 && isfinite(Q_best)) || return false
    gap        = iter - rs.mark_it
    rs.mark_it = iter
    Q_best < rs.last_Q || return false
    rs.slow  += gap
    rs.last_Q = Q_best
    if !isfinite(rs.mark_Q)
        rs.mark_Q = Q_best
        rs.slow   = 0
        return false
    end
    rate = (rs.mark_Q - Q_best) / max(rs.slow, 1) * 100.0
    if rate >= rs.tol
        rs.mark_Q = Q_best
        rs.slow   = 0
        return false
    end
    return rs.slow >= rs.span
end

@inline function _step_label(step::Float64, step_vec::Vector{Float64}, subset_k::Int)
    subset_k == 0 && return @sprintf("%.4f", step)
    # Under a per-coordinate step the scalar is inert, and the span is the thing worth
    # seeing: it is how far apart the coordinates' scales have been driven so far.
    @sprintf("%.1e‥%.1e", minimum(step_vec), maximum(step_vec))
end

"""
    _sample_subset!(buf, k, d, rng) -> view of k distinct indices

Partial Fisher-Yates on a persistent buffer: draws `k` distinct coordinates out of
`d` without allocating, which matters because this runs once per SA iteration.
"""
@inline function _sample_subset!(buf::Vector{Int}, k::Int, d::Int, rng)
    for i in 1:k
        j = rand(rng, i:d)
        buf[i], buf[j] = buf[j], buf[i]
    end
    return view(buf, 1:k)
end

"""
    _sa_loop(spec, theta_start; ...) → (theta_best, Q_best, iters)

One simulated-annealing chain in unconstrained (logit) space, started
from `theta_start`.  This is the single-chain engine used both for a lone
start and for each parallel start in `_run_sa`.
"""
function _sa_loop(
    spec             :: SMMSpec,
    theta_start      :: AbstractVector{Float64};
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
    subset_k         :: Int     = 0,
    corana_Ns        :: Int     = 20,
    corana_c         :: Float64 = 2.0,
    step_floor_rel   :: Float64 = 1e-4,
    rate_tol         :: Float64 = 0.0,
    rate_span        :: Int     = 0,
    cooling_halflife :: Int     = 0,
    t0_rel           :: Float64 = 0.05,
    t0_accept        :: Float64 = 0.30,
    reheat_reset_tol :: Float64 = 0.25,
    show_trace       :: Bool    = true,
    trace_stride     :: Int     = 100,
    rng                         = Random.default_rng(),
)
    theta      = copy(theta_start)
    Q          = smm_objective(theta, spec)
    theta_best = copy(theta)
    Q_best     = isfinite(Q) ? Q : Inf
    n_acc      = 0
    n_fin      = 0
    n_reheats  = 0

    steps_since_improvement = 0

    if T0 <= 0.0
        # Anchor T0 to the OBJECTIVE LEVEL, not to probed uphill move sizes.
        # Probing ΔQ fails here: the proposal distribution straddles the
        # feasibility boundary, so a probe that lands near it returns a ΔQ two
        # orders of magnitude above a typical local move, and any statistic of
        # that sample (median or quantile) inherits the tail.  A T0 set that way
        # exceeds Q itself by orders of magnitude, the walk then accepts moves
        # that multiply the objective, and it random-walks into a bad region it
        # cannot descend out of once T falls.
        #
        # Instead: T0 solves exp(−t0_rel·Q / T0) = t0_accept, i.e. a move costing
        # t0_rel of the current objective is accepted with probability t0_accept.
        # Scale-free in Q, so it survives a change of weighting matrix, and it
        # costs no extra solves because Q is already in hand.
        Q_anchor = isfinite(Q) ? abs(Q) : 1.0
        T0 = -t0_rel * Q_anchor / log(t0_accept)
        if show_trace
            @printf("  [SA T0 auto]  T0 = %.4f  (%.1f%% of Q0 = %.6e accepted w.p. %.2f)\n",
                    T0, 100 * t0_rel, Q_anchor, t0_accept)
            flush(stdout)
        end
    end

    T_current = T0
    T_reheat  = T0
    t_local   = 0
    # Numerator that normalises the cooling law to T(1) = T_reheat (see below).
    _COOL_NUM = log1p(cooling_rate)

    win_fin = adapt_window > 0 ? zeros(Bool, adapt_window) : Bool[]
    win_acc = adapt_window > 0 ? zeros(Bool, adapt_window) : Bool[]

    # Per-coordinate step vector (Corana, Marchesi, Martini & Ridella 1987).  Each
    # coordinate carries its own step, adapted from its own acceptance rate, so the
    # 1789x span of feasible half-widths is discovered by the run rather than
    # supplied to it.  The floor is relative to the initial step: an absolute floor
    # is what pins the narrow coordinates, since one value cannot be small enough
    # for a_ℓ and large enough for σ_S at the same time.
    d_free     = length(theta)
    step_vec   = fill(step, d_free)
    step_floor = step_floor_rel * step
    subset_idx = collect(1:d_free)
    n_prop     = zeros(Int, d_free)   # proposals per coordinate since the last update
    n_acc_j    = zeros(Int, d_free)   # of which accepted
    corana_win = corana_Ns * max(subset_k, 1)   # iterations to give each coordinate
                                                # corana_Ns outcomes on average
    rate_stop  = RateStop(rate_tol, rate_span)
    win_idx = 0

    actual_iters = 0

    if show_trace
        n_corners_init = _count_corners(theta_best, spec)
        @printf("  [SA init]  Q0 = %s  T0=%.4f  step=%s  corners=%d/%d%s\n",
                isfinite(Q) ? @sprintf("%.6e", Q) : "Inf (rejected start)",
                T0, _step_label(step, step_vec, subset_k),
                n_corners_init, length(spec.free), _corner_tags(theta_best, spec))
        # A rejected start abandons the seed and restarts from the first accepted
        # proposal, so the run is no longer the warm start it reports being.  The
        # vector is printed at full precision, in the same block/name form as the
        # bundle CSV, so it can be diffed against the θ̂ the seed was meant to be.
        if !isfinite(Q)
            println("             rejected θ (compare against the warm-start bundle):")
            for (i, ps) in enumerate(spec.free)
                @printf("             %-6s %-10s %14.8f   [%14.8f, %14.8f]\n",
                        ps.block, ps.name,
                        _to_constrained(theta_best[i], ps.lb, ps.ub), ps.lb, ps.ub)
            end
        end
        flush(stdout)
    end

    for t in 1:max_iter
        actual_iters = t

        t_local  += 1
        # Cooling.  cooling_halflife > 0 selects geometric decay, T = T_reheat·2^(−t/H):
        # the fall is a constant factor per H iterations however long the run, so the
        # profile is set by one number in the units of the budget.  The logarithmic
        # alternative front-loads the whole descent — at rate = 1, exp = 2 it drops by
        # a factor of 12 within ten iterations and is within 2x of its floor by a
        # hundred, which leaves the remaining budget running as a hill-climber.
        #
        # The logarithmic branch is normalised: the bare form T_reheat/log(1+rate·t)^exp
        # divides by a number BELOW ONE while 1+rate·t < e, so it *heats* over the first
        # (e−1)/rate steps.  Dividing by log(1+rate)^exp pins T(1) = T_reheat exactly for
        # every (rate, exp), so those two knobs control only the decay profile.
        T_current = cooling_halflife > 0 ?
            T_reheat * 2.0^(-(t_local - 1) / cooling_halflife) :
            T_reheat * (_COOL_NUM / log1p(cooling_rate * t_local))^cooling_exp
        T_current = max(T_current, 1e-8)

        # Proposal.  With subset_k = 0 this is the isotropic scalar move; with
        # subset_k > 0 it perturbs a random subset of that size, each coordinate by
        # its OWN adapted step.  The subset is what the model needs — coordinates
        # that are individually infeasible can be jointly feasible, so a strictly
        # one-at-a-time sweep cannot reach some improving points — while the
        # per-coordinate scale is what makes any of it acceptable: the feasible
        # half-widths span three orders of magnitude, so a single scalar either
        # overshoots the narrow coordinates or freezes the wide ones.
        theta_prop = copy(theta)
        if subset_k > 0
            moved = _sample_subset!(subset_idx, min(subset_k, d_free), d_free, rng)
            for j in moved
                theta_prop[j] += step_vec[j] * randn(rng)
                n_prop[j] += 1
            end
        else
            moved = 1:0
            theta_prop .+= step .* randn(rng, d_free)
        end
        Q_prop = smm_objective(theta_prop, spec)

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
                for j in moved
                    n_acc_j[j] += 1
                end
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

            # Scalar adaptation drives the isotropic proposal only.  Under
            # subset_k > 0 the step vector is adapted per coordinate below instead,
            # on its own schedule, and this scalar is left untouched.
            if subset_k == 0 && t >= adapt_window && t % adapt_window == 0
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

        # Corana step-vector update.  Each coordinate is steered toward a 0.4-0.6
        # acceptance rate by its own record: too many accepts means the step is
        # small enough to be wasteful, too few means it overshoots that
        # coordinate's feasible width.  Coordinates the subset never drew are left
        # alone rather than adapted on no evidence.
        if subset_k > 0 && t % corana_win == 0
            for j in 1:d_free
                n_prop[j] == 0 && continue
                p = n_acc_j[j] / n_prop[j]
                if p > 0.6
                    step_vec[j] *= 1 + corana_c * (p - 0.6) / 0.4
                elseif p < 0.4
                    step_vec[j] /= 1 + corana_c * (0.4 - p) / 0.4
                end
                step_vec[j] = clamp(step_vec[j], step_floor, 2.0)
            end
            fill!(n_prop, 0)
            fill!(n_acc_j, 0)
        end

        # Rate stop: the descent has flattened below what the moments can resolve.
        # Checked before the reheat logic, since a run this flat gains nothing from
        # being reheated into the same basin.
        if rate_stop!(rate_stop, Q_best, t)
            if show_trace
                n_corners_rs = _count_corners(theta_best, spec)
                @printf("  [SA EARLY STOP  iter=%5d]  ΔQ < %.3g per 100 iters over %d iters  Q_best=%.6e  corners=%d/%d%s\n",
                        t, rate_tol, rate_stop.slow, Q_best,
                        n_corners_rs, length(spec.free), _corner_tags(theta_best, spec))
                flush(stdout)
            end
            break
        end

        if reheat_patience > 0 &&
           max_reheats > 0 && n_reheats >= max_reheats &&
           steps_since_improvement >= reheat_patience
            if show_trace
                n_corners_es = _count_corners(theta_best, spec)
                @printf("  [SA EARLY STOP  iter=%5d]  reheats exhausted, no improvement for %d steps, Q_best=%.6e  corners=%d/%d%s\n",
                        t, steps_since_improvement, Q_best,
                        n_corners_es, length(spec.free), _corner_tags(theta_best, spec))
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
            # Reheat from the best point rather than in place.  Raising T only
            # helps if the walk is sitting somewhere worth escaping from; when it
            # has already drifted far above Q_best — which is what happens once a
            # high T has let it accept a sequence of uphill moves — reheating in
            # place pushes it further away instead.  Snapping back to
            # (theta_best, Q_best) whenever the current point is more than
            # reheat_reset_tol worse keeps the extra energy aimed at the
            # neighbourhood of the incumbent.  Within that tolerance the walk
            # keeps its position, so a genuine basin escape is still possible.
            if isfinite(Q_best) && (!isfinite(Q) || Q > (1.0 + reheat_reset_tol) * abs(Q_best))
                theta .= theta_best
                Q      = Q_best
            end
            steps_since_improvement = 0

            if show_trace
                n_corners_rh = _count_corners(theta, spec)
                @printf("  [SA REHEAT #%d  iter=%5d]  T %.4f→%.4f  in place at Q=%.6e (best=%.6e)  corners=%d/%d%s\n",
                        n_reheats, t, T_before, T_current, isfinite(Q) ? Q : Inf, Q_best,
                        n_corners_rh, length(spec.free), _corner_tags(theta, spec))
                flush(stdout)
            end
        end

        if show_trace && t % trace_stride == 0
            w_acc = adapt_window > 0 && t >= adapt_window ? mean(win_acc) : n_acc / t
            w_fin = adapt_window > 0 && t >= adapt_window ? mean(win_fin) : n_fin / t
            n_corners = _count_corners(theta_best, spec)
            @printf("  [SA iter=%5d]  curr=%-14s  best=%.6e  T=%.4f  step=%s  acc=%.2f  fin=%.2f  corners=%d/%d%s  reheats=%d\n",
                    t,
                    isfinite(Q) ? @sprintf("%.6e", Q) : "Inf",
                    Q_best, T_current, _step_label(step, step_vec, subset_k),
                    w_acc, w_fin,
                    n_corners, length(spec.free), _corner_tags(theta_best, spec),
                    n_reheats)
            flush(stdout)
        end
    end

    if show_trace
        n_corners_done = _count_corners(theta_best, spec)
        @printf("  [SA done]  Q_best=%.6e  accepted %d/%d  finite %d/%d  corners=%d/%d%s  reheats=%d\n",
                Q_best, n_acc, actual_iters, n_fin, actual_iters,
                n_corners_done, length(spec.free), _corner_tags(theta_best, spec),
                n_reheats)
        flush(stdout)
    end

    return theta_best, Q_best, actual_iters
end


"""
    _run_sa(spec; starts, parallel_steps, seed, ...) → (theta_best, Q_best, iters)

Multi-start simulated annealing.  When `starts` holds more than one point
(one per cluster), every start runs for the first `parallel_steps`
iterations; the best chain (lowest Q_best) is then continued to completion.
Because each start sits in a distinct cluster, pruning to the best chain
selects the best basin, not merely the best individual candidate.

The warm-up chains run sequentially (not threaded over) — each model solve
already uses the solver's internal multithreading, and nesting thread pools
oversubscribes the workers and can stall the run.

With zero or one start the routine reduces to a single chain: seeded from
`starts[1]`, or — if `starts` is empty — from a random draw
(`random_init=true`) or `pack_theta(spec)`.

Per-chain RNGs are seeded deterministically as `Xoshiro(seed + j)`, and the
continuation chain uses `Xoshiro(seed)`, so a run is replicable.  Note the
continuation restarts the temperature schedule from the best basin's
incumbent rather than resuming the pruned chain's internal SA state — a
fresh anneal from the selected basin.
"""
function _run_sa(
    spec             :: SMMSpec;
    starts           :: Vector{Vector{Float64}} = Vector{Vector{Float64}}(),
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
    subset_k         :: Int     = 0,
    corana_Ns        :: Int     = 20,
    corana_c         :: Float64 = 2.0,
    step_floor_rel   :: Float64 = 1e-4,
    rate_tol         :: Float64 = 0.0,
    rate_span        :: Int     = 0,
    cooling_halflife :: Int     = 0,
    t0_rel           :: Float64 = 0.05,
    t0_accept        :: Float64 = 0.30,
    reheat_reset_tol :: Float64 = 0.25,
    parallel_steps   :: Int     = 100,
    seed             :: Int     = 20240601,
    random_init      :: Bool    = false,
    show_trace       :: Bool    = true,
    trace_stride     :: Int     = 100,
    rng                         = Random.default_rng(),
)
    # Assemble the start set.
    start_set = if !isempty(starts)
        starts
    else
        [random_init ? _random_theta(spec, rng) : pack_theta(spec)]
    end

    # Single start → original single-chain behaviour.
    if length(start_set) <= 1
        return _sa_loop(spec, start_set[1];
                        T0 = T0, step = step, max_iter = max_iter,
                        cooling_rate = cooling_rate, cooling_exp = cooling_exp,
                        reheat_patience = reheat_patience, reheat_factor = reheat_factor,
                        max_reheats = max_reheats, adapt_window = adapt_window,
                        target_fin = target_fin, subset_k = subset_k,
                        corana_Ns = corana_Ns, corana_c = corana_c,
                        step_floor_rel = step_floor_rel,
                        rate_tol = rate_tol, rate_span = rate_span,
                        cooling_halflife = cooling_halflife,
                        t0_rel = t0_rel, t0_accept = t0_accept,
                        reheat_reset_tol = reheat_reset_tol,
                        show_trace = show_trace,
                        trace_stride = trace_stride, rng = rng)
    end

    # Multi-start warm-up, then prune to the best basin.  Chains run
    # SEQUENTIALLY, not threaded over: each smm_objective already uses the
    # solver's internal multithreading, so wrapping the chains in another
    # Threads.@threads nests thread pools — with few chains that starves the
    # inner solver and can stall the run.  Sequential chains keep every model
    # solve fully parallel internally (where the speed actually comes from).
    nch     = length(start_set)
    p_steps = min(parallel_steps, max_iter)

    chain_theta = Vector{Vector{Float64}}(undef, nch)
    chain_Q     = fill(Inf, nch)

    if show_trace
        @printf("  [SA multistart]  %d chains x %d warm-up steps (sequential; then prune to best basin)\n",
                nch, p_steps)
        flush(stdout)
    end

    for j in 1:nch
        rng_j = Random.Xoshiro(UInt64(seed) + UInt64(j))
        tb, qb, _ = _sa_loop(spec, start_set[j];
                             T0 = T0, step = step, max_iter = p_steps,
                             cooling_rate = cooling_rate, cooling_exp = cooling_exp,
                             reheat_patience = reheat_patience, reheat_factor = reheat_factor,
                             max_reheats = max_reheats, adapt_window = adapt_window,
                             target_fin = target_fin, subset_k = subset_k,
                             corana_Ns = corana_Ns, corana_c = corana_c,
                             step_floor_rel = step_floor_rel,
                             rate_tol = rate_tol, rate_span = rate_span,
                             cooling_halflife = cooling_halflife,
                             t0_rel = t0_rel, t0_accept = t0_accept,
                        reheat_reset_tol = reheat_reset_tol,
                             show_trace = false,
                             trace_stride = trace_stride, rng = rng_j)
        chain_theta[j] = tb
        chain_Q[j]     = qb
        if show_trace
            @printf("  [SA multistart]  chain %d/%d done  Q_best=%s\n",
                    j, nch, isfinite(qb) ? @sprintf("%.6e", qb) : "Inf")
            flush(stdout)
        end
    end

    jbest = argmin(chain_Q)
    if show_trace
        @printf("  [SA multistart]  best basin = chain %d  Q_best=%.6e  (feasible chains: %d/%d)\n",
                jbest, chain_Q[jbest], count(isfinite, chain_Q), nch)
        flush(stdout)
    end

    remaining = max(max_iter - p_steps, 0)
    if remaining == 0
        return chain_theta[jbest], chain_Q[jbest], p_steps
    end

    # Continue the best basin to completion.
    tb, qb, iters = _sa_loop(spec, chain_theta[jbest];
                             T0 = T0, step = step, max_iter = remaining,
                             cooling_rate = cooling_rate, cooling_exp = cooling_exp,
                             reheat_patience = reheat_patience, reheat_factor = reheat_factor,
                             max_reheats = max_reheats, adapt_window = adapt_window,
                             target_fin = target_fin, subset_k = subset_k,
                             corana_Ns = corana_Ns, corana_c = corana_c,
                             step_floor_rel = step_floor_rel,
                             rate_tol = rate_tol, rate_span = rate_span,
                             cooling_halflife = cooling_halflife,
                        t0_rel = t0_rel, t0_accept = t0_accept,
                        reheat_reset_tol = reheat_reset_tol,
                        show_trace = show_trace,
                             trace_stride = trace_stride, rng = Random.Xoshiro(UInt64(seed)))

    if chain_Q[jbest] <= qb
        return chain_theta[jbest], chain_Q[jbest], p_steps + iters
    else
        return tb, qb, p_steps + iters
    end
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
    local_k      :: Int     = 0,
    local_sigma  :: Float64 = 0.33,
    local_sigma_cap :: Float64 = 0.20,
    seed_bank    :: Union{Nothing,SeedBank}        = nothing,
    prev_optimum :: Union{Nothing,Vector{Float64}} = nothing,
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

    # Base population.  A uniform draw over the box is the wrong scale by three
    # orders of magnitude — the box spans ±4 in unconstrained space while the
    # measured feasible half-widths run from 6e-5 to 1e-1 — so essentially every
    # member is infeasible and the difference vectors carry no local geometry.
    # local_k > 0 instead perturbs `local_k` coordinates of theta0 at a time, each
    # by its own measured width: sparse because moving all 25 at once compounds 25
    # small increases in Q (measured useful-draw rate 0.51 at k=3 against 0.08 at
    # k=25), and per-coordinate because no single scale fits that span.
    pop = Vector{Vector{Float64}}(undef, pop_size)
    if local_k > 0
        widths = _feasible_widths(theta0, spec, local_sigma_cap)
        for j in 1:pop_size
            theta_j = copy(theta0)
            for k in randperm(rng, npar)[1:min(local_k, npar)]
                theta_j[k] += local_sigma * abs(widths[k]) * randn(rng)
            end
            pop[j] = theta_j
        end
    else
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
    end

    if local_k > 0
        # The local population is already anchored on theta0; a cluster bank drawn
        # over the whole box would reintroduce exactly the scale error it avoids.
        pop[1] = copy(theta0)
        if show_gens
            @printf("  [DE init]  %d local members: %d coords perturbed at %.2f×width\n",
                    pop_size, local_k, local_sigma)
            flush(stdout)
        end
    elseif seed_bank === nothing
        # No bank: warm the first member with the spec's initial point.
        pop[1] = copy(theta0)
    else
        # Seed from clusters (round-robin, best-Q first); the random draws above
        # remain as the top-up for any slots the bank cannot fill.
        seeded = _seed_pop_from_bank(seed_bank, pop_size)
        for (j, θ) in enumerate(seeded)
            pop[j] = θ
        end
        if show_gens
            @printf("  [DE init]  seeded %d/%d members from candidate clusters\n",
                    length(seeded), pop_size)
            flush(stdout)
        end
    end

    # Guaranteed previous-optimum member (when supplied and valid).
    if prev_optimum !== nothing
        pop[1] = copy(prev_optimum)
    end

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
            @printf("  [DE gen=%4d DONE]  Q_best=%.6e Q_mean=%-14s  feasible=%d/%d  improved=%d  clusters=%d  corners=%d/%d%s  evals=%d\n",
                    gen,
                    Q_best,
                    isfinite(Q_mean) ? @sprintf("%.6e", Q_mean) : "Inf",
                    n_feas, pop_size, n_imp, n_bas,
                    n_corners, length(spec.free), _corner_tags(theta_best, spec),
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
# Seed-bank helpers (candidate-cluster seeding for SA / DE)
# ============================================================

"""
    _bank_clusters(bank) → Vector{Vector{Int}}

Group candidate indices by cluster label, each group sorted by ascending Q.
"""
function _bank_clusters(bank::SeedBank)
    groups = Vector{Vector{Int}}()
    for lab in unique(bank.labels)
        idx = findall(==(lab), bank.labels)
        sort!(idx; by = i -> bank.Q[i])
        push!(groups, idx)
    end
    return groups
end

"""
    _sa_starts_from_bank(bank, prev_optimum) → Vector{Vector{Float64}}

One SA start per cluster (the best-Q member), optionally with `prev_optimum`
appended.  Returns an empty vector when there is no bank and no previous
optimum (the caller then falls back to a single start).
"""
function _sa_starts_from_bank(bank::Union{Nothing,SeedBank},
                              prev_optimum::Union{Nothing,Vector{Float64}})
    starts = Vector{Vector{Float64}}()
    if bank !== nothing
        for grp in _bank_clusters(bank)
            isempty(grp) && continue
            push!(starts, copy(bank.candidates[grp[1]]))   # best-Q member
        end
    end
    prev_optimum !== nothing && push!(starts, copy(prev_optimum))
    return starts
end

"""
    _seed_pop_from_bank(bank, pop_size) → Vector{Vector{Float64}}

Round-robin fill across clusters (best-Q first within each), skipping dry
clusters, up to `pop_size` members.  May return fewer than `pop_size`; the
caller tops up the remainder with random draws.
"""
function _seed_pop_from_bank(bank::SeedBank, pop_size::Int)
    groups  = _bank_clusters(bank)
    pop     = Vector{Vector{Float64}}()
    cursors = ones(Int, length(groups))
    while length(pop) < pop_size
        advanced = false
        for (g, grp) in enumerate(groups)
            cursors[g] <= length(grp) || continue
            push!(pop, copy(bank.candidates[grp[cursors[g]]]))
            cursors[g] += 1
            advanced = true
            length(pop) >= pop_size && break
        end
        advanced || break   # all clusters exhausted
    end
    return pop
end


"""
    FeasibleSimplexer(spec, step) <: Optim.Simplexer

Initial Nelder-Mead simplex whose vertices the objective actually accepts.

Optim's default `AffineSimplexer` places vertex j at `(1 + 0.5)·t_j + 0.025` — a step
proportional to the coordinate's own value, which in transformed space bears no relation
to how far that coordinate can move. Measured at the base_fc optimum, all 25 default
vertices are infeasible on arrival, overshooting the feasible half-width by a median of
296× and up to 20 000×. Nelder-Mead then spends thousands of evaluations shrinking a
simplex whose vertices all score `Inf`, which is the "slow start" before real descent
begins around evaluation 5 000.

This simplexer instead bisects each coordinate outward from θ̂ until it finds the
largest step the objective still scores finite, capped at `step` in transformed units.
Every vertex is therefore feasible by construction and carries information from the
first evaluation.
"""
struct FeasibleSimplexer{S} <: Optim.Simplexer
    spec :: S
    step :: Float64
end

"""
    _feasible_widths(θ, spec, cap; nbisect=12, dq=1.0) -> Vector{Float64}

Largest step per coordinate, in the better of the two directions, that keeps the
objective finite AND within `dq` of Q(θ).  Signed: negative where the downward
direction is the wider one.

The ΔQ criterion is what makes the result useful rather than merely legal.  A
merely-feasible step is a median 762x beyond the scale on which Q varies here, so a
simplex built on feasibility alone puts its vertices where Q is enormous — measured
at the base_fc optimum, 1/25 vertices beat the seed under feasibility against 9/25
under ΔQ<1, with vertex Q running to 2.5e4 in the first case and staying inside
[Q0, Q0+1] in the second.

Embarrassingly parallel: d·nbisect solves once, against the thousands a bad simplex
wastes on uninformative vertices.
"""
function _feasible_widths(θ::AbstractVector, spec::SMMSpec, cap::Float64;
                          nbisect::Int = 12, dq::Float64 = 1.0)
    d  = length(θ)
    Q0 = smm_objective(collect(float.(θ)), spec)
    w  = zeros(d)
    Threads.@threads for j in 1:d
        for dir in (1.0, -1.0)
            lo, hi = 0.0, cap
            for _ in 1:nbisect
                mid = 0.5 * (lo + hi)
                t   = collect(float.(θ)); t[j] += dir * mid
                q   = smm_objective(t, spec)
                (isfinite(q) && q < Q0 + dq) ? (lo = mid) : (hi = mid)
            end
            abs(lo) > abs(w[j]) && (w[j] = dir * lo)
        end
    end
    return w
end

function Optim.simplexer(S::FeasibleSimplexer, initial_x::Tx) where {Tx}
    d       = length(initial_x)
    simplex = Tx[copy(initial_x) for _ in 1:d+1]
    w       = _feasible_widths(initial_x, S.spec, S.step)
    for j in 1:d
        # A coordinate with no usable step at all keeps the seed value: the simplex
        # is then degenerate in that direction, which is the honest representation of a
        # parameter the objective cannot move.
        simplex[j+1][j] += w[j]
    end
    @printf("  [NM simplex]  %d/%d coordinates admit a step within ΔQ<1 (cap %.3g)\n",
            count(!=(0.0), w), d, S.step)
    flush(stdout)
    simplex
end

# ============================================================
# Main optimisation entry point
# ============================================================

"""
    run_smm(spec; method=:de, rng=default_rng()) → SMMResult


Run SMM estimation.  All settings come from `spec.run`.
"""
function run_smm(
    spec         :: SMMSpec;
    method       :: Symbol = :de,
    seed_bank    :: Union{Nothing,SeedBank}        = nothing,
    prev_optimum :: Union{Nothing,Vector{Float64}} = nothing,
    rng                  = Random.default_rng(),
    # Rate-based Nelder-Mead stop. Keyword arguments rather than SMMRunParams fields:
    # Julia's serialiser reads structs positionally by field COUNT, so adding a field
    # to SMMRunParams makes every bundle already on disk unreadable — verified, not
    # assumed. Defaults reproduce the previous behaviour when omitted.
    #
    # Two knobs, both in units the objective is measured in:
    #   nm_rate_tol   improvement budget per window. Q is a chi-square, so ΔQ = 1 is
    #                 one moment moving by one sampling standard error; 0.05 is one
    #                 twentieth of that. Measured solver noise is ΔQ ≈ 1e-6, six
    #                 orders below, so this threshold is statistical, not numerical.
    #   nm_rate_span  how many slow evaluations to tolerate before stopping.
    #
    # The span is checked in fixed 100-evaluation sub-windows rather than as one long
    # window, and that granularity is not cosmetic. A window with NO improvement at
    # all resets the streak, because a flat stretch is the simplex against an
    # infeasible wall — in 25 dimensions it can persist for thousands of evaluations
    # and then recover, so it needs patience rather than a stop. Sub-windows short
    # enough to isolate those exact zeros preserve that reset; one long window
    # averages them together with neighbouring small moves and reports 'small but
    # positive', which is exactly the state the rule stops on.
    #
    # Replaying a full base_fc trace (11 800 evaluations, 1289.45 → 1275.38) shows the
    # cost: 100-evaluation sub-windows stop at ~10 600 leaving 0.35 in Q, while a
    # single 300-evaluation window stops at 4 000 during a plateau the descent later
    # escapes, discarding 10 units. Any span from 200 to 800 lands within 10 500-11 100
    # on that trace, so the rule is insensitive to the span and sensitive to the
    # sub-window — hence one exposed knob and one fixed constant.
    nm_rate_tol    :: Float64 = 0.05,
    nm_rate_span   :: Int     = 300,
    # Initial-simplex bisection cap in transformed units. Optim's default places every
    # vertex far outside the useful region here, so the first thousands of evaluations
    # score Inf and only shrink the simplex. A positive value bisects each coordinate
    # for the largest step keeping Q within 1 of the seed; 0 restores Optim's
    # AffineSimplexer. The ΔQ criterion binds well below this cap on every coordinate
    # (measured half-widths top out at 1.1e-1 at base_fc), so the cap is inert — it
    # matters only under a pure feasibility test, where coordinates stay finite out to
    # ±4 and the cap alone keeps vertices near the seed.
    nm_simplex_step :: Float64 = 0.2,
    # Simulated-annealing proposal. subset_k = 0 keeps the isotropic scalar move;
    # subset_k > 0 perturbs that many random coordinates per iteration, each by its own
    # Corana-adapted step. Both cost one solve per iteration — the subset changes which
    # coordinates move, not how many solves it takes. The subset (rather than one
    # coordinate at a time) is what the model needs: coordinates that are individually
    # infeasible can be jointly feasible, so a strict sweep cannot reach some improving
    # points. corana_Ns is the outcomes-per-coordinate the step update waits for; below
    # ~10 the estimated acceptance rate is too noisy to steer it.
    sa_subset_k     :: Int     = 0,
    sa_corana_Ns    :: Int     = 20,
    sa_corana_c     :: Float64 = 2.0,
    sa_step_floor_rel :: Float64 = 1e-4,
    # Annealing stop and schedule, in the same units as the Nelder-Mead rate rule
    # above and sharing its implementation (RateStop): stop when the incumbent
    # improves at less than sa_rate_tol per 100 iterations, sustained over
    # sa_rate_span PRODUCTIVE iterations, with flat stretches pausing rather than
    # counting. sa_cooling_halflife > 0 replaces the logarithmic schedule with
    # T = T0·2^(−t/H): the logarithmic one spends its whole descent in the first
    # hundred iterations, leaving the rest of the budget effectively greedy.
    sa_rate_tol     :: Float64 = 0.0,
    sa_rate_span    :: Int     = 0,
    sa_cooling_halflife :: Int = 0,
    # DE initial population. de_local_k > 0 builds it by perturbing that many
    # coordinates of the spec's initial point at a time, each scaled by its own
    # ΔQ<1 width; 0 keeps the uniform draw over the box. Sparse because moving all
    # coordinates at once compounds their individual increases in Q (measured
    # useful-draw rate 0.51 at k=3 against 0.08 at k=25).
    de_local_k      :: Int     = 0,
    de_local_sigma  :: Float64 = 0.33,
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
            local_k      = de_local_k,
            local_sigma  = de_local_sigma,
            seed_bank    = seed_bank,
            prev_optimum = prev_optimum,
            show_members = r.show_trace_members,
            show_gens    = r.show_trace_generations,
            trace_stride = r.trace_stride,
            rng          = rng,
        )
        converged = isfinite(loss_opt)
        conv_why  = isfinite(loss_opt) ? "de-stop" : "infeasible"

    elseif method == :sa
        sa_starts = _sa_starts_from_bank(seed_bank, prev_optimum)
        theta_opt, loss_opt, niters = _run_sa(
            spec;
            starts          = sa_starts,
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
            subset_k        = sa_subset_k,
            corana_Ns       = sa_corana_Ns,
            corana_c        = sa_corana_c,
            step_floor_rel  = sa_step_floor_rel,
            rate_tol        = sa_rate_tol,
            rate_span       = sa_rate_span,
            cooling_halflife = sa_cooling_halflife,
            t0_rel          = r.sa_t0_rel,
            t0_accept       = r.sa_t0_accept,
            reheat_reset_tol = r.sa_reheat_reset_tol,
            parallel_steps  = r.sa_parallel_steps,
            seed            = r.sa_seed,
            random_init     = r.sa_random_init,
            show_trace      = r.show_trace_generations,
            trace_stride    = r.trace_stride,
            rng             = rng,
        )
        converged = isfinite(loss_opt)
        conv_why  = isfinite(loss_opt) ? "sa-stop" : "infeasible"

    elseif method in (:neldermead, :lbfgs, :bfgs)
        theta0        = pack_theta(spec)
        iter_count    = Ref(0)
        best_loss     = Ref(Inf)
        best_theta    = Ref(copy(theta0))   # incumbent (best) point, for corner reporting
        last_improve  = Ref(0)       # eval count at which best_loss last improved
        stopped_early = Ref(false)   # set when the no-improvement knob halts NM
        stop_reason   = Ref(:none)   # which early-stop rule fired
        # Rate-rule state, tracked continuously rather than in windows: the best Q
        # and evaluation count at the last point the descent was moving fast enough,
        # and how many PRODUCTIVE evaluations have accumulated since.
        nm_rate       = RateStop(nm_rate_tol, nm_rate_span)

        function obj_traced(theta)
            iter_count[] += 1
            Q = smm_objective(theta, spec)
            if isfinite(Q) && Q < best_loss[]
                best_loss[]    = Q
                best_theta[]   = copy(theta)
                last_improve[] = iter_count[]
            end
            if r.show_trace_generations && iter_count[] % r.trace_stride == 0
                n_c = _count_corners(best_theta[], spec)
                @printf("  [%s iter %4d]  Q=%-14s  best=%.6e  corners=%d/%d%s\n",
                        method, iter_count[],
                        isfinite(Q) ? @sprintf("%.6e", Q) : "Inf",
                        best_loss[], n_c, length(spec.free),
                        _corner_tags(best_theta[], spec))
                flush(stdout)
            end
            return isfinite(Q) ? Q : 1e16
        end

        # Two early stops, both returning true to halt Optim, and both requiring a
        # finite incumbent so a run that has not yet found a feasible point is never
        # cut short.
        #
        #  · no-improve: the best Q has not moved for nm_no_improve evaluations.
        #  · rate: the best Q improved at less than nm_rate_tol per 100 evaluations,
        #    sustained over nm_rate_span PRODUCTIVE evaluations. Evaluations that did
        #    not move the best Q are paused out of the count rather than counted or
        #    reset — that is the simplex against an infeasible wall, not convergence,
        #    and it can persist for thousands of evaluations before recovering.
        function nm_stop_cb(_state)
            isfinite(best_loss[]) || return false

            if r.nm_no_improve > 0 && (iter_count[] - last_improve[]) >= r.nm_no_improve
                stopped_early[] = true; stop_reason[] = :no_improve
                n_c = _count_corners(best_theta[], spec)
                @printf("  [%s EARLY STOP  iter %d]  no improvement for %d evals  best=%.6e  corners=%d/%d%s\n",
                        method, iter_count[], r.nm_no_improve, best_loss[],
                        n_c, length(spec.free), _corner_tags(best_theta[], spec))
                flush(stdout)
                return true
            end

            if rate_stop!(nm_rate, best_loss[], iter_count[])
                stopped_early[] = true; stop_reason[] = :rate
                n_c = _count_corners(best_theta[], spec)
                @printf("  [%s EARLY STOP  iter %d]  ΔQ < %.3g per 100 evals over %d evals  best=%.6e  corners=%d/%d%s\n",
                        method, iter_count[], nm_rate_tol, nm_rate.slow,
                        best_loss[], n_c, length(spec.free),
                        _corner_tags(best_theta[], spec))
                flush(stdout)
                return true
            end
            return false
        end

        # nm_simplex_step ≤ 0 keeps Optim's AffineSimplexer; a positive value builds a
        # simplex whose every vertex the objective accepts, so descent starts at the
        # first evaluation instead of after thousands of Inf-scored contractions.
        opt_method = (method == :neldermead) ?
                       (nm_simplex_step > 0 ?
                          Optim.NelderMead(initial_simplex =
                              FeasibleSimplexer(spec, nm_simplex_step)) :
                          Optim.NelderMead()) :
                     (method == :lbfgs)      ? Optim.LBFGS()      : Optim.BFGS()

        options   = Optim.Options(iterations = r.nm_max_iter,
                                  f_reltol      = r.nm_f_tol,
                                  x_abstol      = r.nm_x_tol,
                                  g_abstol      = r.nm_g_tol,
                                  callback      = nm_stop_cb,
                                  show_trace = false)
        result    = Optim.optimize(obj_traced, theta0, opt_method, options)
        theta_opt = Optim.minimizer(result)
        loss_opt  = smm_objective(theta_opt, spec)
        # Either deliberate early stop counts as a valid finish, mirroring the SA
        # convention: a finite incumbent was found and the rule that halted the run is
        # the one we asked for. `rel_stall` is the fallback for a run that hit the
        # iteration cap having plainly stopped moving — 500 evaluations without any
        # improvement — and is independent of nm_no_improve so that disabling that
        # knob does not make the test fire at zero.
        rel_stall = isfinite(best_loss[]) &&
                    (iter_count[] - last_improve[]) >= 500
        converged = (Optim.converged(result) || stopped_early[] || rel_stall) &&
                    isfinite(loss_opt)
        conv_why  = !isfinite(loss_opt)             ? "infeasible" :
                    stop_reason[] == :rate          ? "rate"       :
                    stop_reason[] == :no_improve    ? "no-improve" :
                    Optim.converged(result)         ? "optim-tol"  :
                    rel_stall                       ? "rel-stall"  : "iter-cap"
        niters    = Optim.iterations(result)

    else
        error("Unknown method :$method. Choose :de, :sa, :neldermead, :lbfgs, or :bfgs.")
    end

    @printf("\nSMM complete:  Q=%.6e  converged=%s (%s)  iters=%d\n",
            isfinite(loss_opt) ? loss_opt : Inf, converged, conv_why, niters)
    flush(stdout)

    cp_opt, up_opt, sp_opt = unpack_θ(theta_opt, spec)
    params_opt = _params_to_namedtuple(cp_opt, up_opt, sp_opt, spec)

    res = SMMResult(theta_opt, params_opt, loss_opt, converged, niters, spec)
    print_results(res; why = conv_why)
    return res
end


# ============================================================
# Result display and saving
# ============================================================

function print_results(res::SMMResult; why::AbstractString = "")
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
    # The reason is passed in rather than stored on SMMResult: adding a field would
    # break every bundle already serialised, and the reason is a property of the run
    # rather than of the estimate.
    @printf("\n  Q = %.8e  |  converged = %s%s  |  iters = %d\n",
            res.loss_opt, res.converged,
            isempty(why) ? "" : @sprintf(" (%s)", why), res.iterations)
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

function _params_to_namedtuple(cp, up, sp, spec::SMMSpec)
    d = Dict{Symbol, Float64}()
    for ps in spec.free
        val = if ps.block == :common; getfield(cp, ps.name)
              elseif ps.block == :unsk; getfield(up, ps.name)
              else;                     getfield(sp, ps.name)
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
    return SMMSpec(new_free, spec.fixed, spec.moments, spec.sim, spec.run, spec.W, spec.q_scale)
end
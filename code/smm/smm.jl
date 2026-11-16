############################################################
# smm/smm.jl — the SMM objective and the optimisers
#
# The objective Q(θ) = g′Wg / q_scale, g the raw deviations of the model moments at the
# equilibrium θ fixes from their targets (`equilibrium_at`, `smm_objective`). The
# optimisers work in the unconstrained (logit) coordinates of the free parameters:
#   :sa            simulated annealing with a measured per-coordinate proposal
#   :de            differential evolution with a measured, yield-allocated population
#   :sa_de         annealing, then DE from its point
#   :neldermead    Optim's Nelder-Mead from a feasible simplex
#   :lbfgs, :bfgs  gradient polish with a measured finite-difference gradient
# Annealing is written here rather than taken from Optim, whose SA mishandles Inf
# proposals; here an Inf proposal is always rejected.
#
# Entry points: run_smm, smm_objective, equilibrium_at, select_equilibrium,
# attracting_check, write_checkpoint, save_results, period_bundle, point_in_spec.
############################################################

"""
    _load_smm_bundle(path; delete_on_fail = false, label = "file") → NamedTuple | nothing

The bundle at `path` if it deserialises to a NamedTuple with non-`nothing` `result` and
`spec`; otherwise `nothing`, with a warning naming `label`, and the file deleted when
`delete_on_fail`.
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

"""
    point_in_spec(b, spec; estimated_only = true) → (θ, unmatched, clamped)

The point of bundle `b` as an unconstrained vector of `spec`. Where `b`'s free set and box
are `spec`'s, `θ` is `b`'s own vector, bit for bit. Otherwise each free parameter of `spec`
takes `b`'s value of the same (block, name), pulled into `spec`'s box (listed in
`clamped`); with `estimated_only`, a parameter `b`'s run did not estimate keeps `spec`'s
start (listed in `unmatched`), and without it every value is read off `b`'s decoded point,
pinned values included. A joint spec is filled period by period.
"""
function point_in_spec(b, spec::SMMSpec; estimated_only::Bool = true)
    same = [(ps.block, ps.name, ps.lb, ps.ub) for ps in b.spec.free] ==
           [(ps.block, ps.name, ps.lb, ps.ub) for ps in spec.free]
    same && return (θ = collect(float.(b.result.theta_opt)), unmatched = String[],
                    clamped = String[])
    if is_joint(spec)
        # Each period from the same period of `b`; a common parameter is written by every
        # period with one value.
        θ = fill(NaN, length(spec.free));  unmatched = String[];  clamped = String[]
        for (w, s, idx) in period_list(spec)
            pt = point_in_spec(period_bundle(b, w), s; estimated_only = estimated_only)
            θ[idx] .= pt.θ
            append!(unmatched, pt.unmatched);  append!(clamped, pt.clamped)
        end
        return (θ = θ, unmatched = unique(unmatched), clamped = unique(clamped))
    end
    cp, up, sp, tight = unpack_θ(b.result.theta_opt, b.spec)
    estimated = Set((ps.block, ps.name) for ps in b.spec.free)
    free = ParamSpec[];  unmatched = String[];  clamped = String[]
    for ps in spec.free
        if estimated_only && !((ps.block, ps.name) in estimated)
            push!(free, ps);  push!(unmatched, "$(ps.block):$(ps.name)")
            continue
        end
        v = _baseline_param_value(ps, cp, up, sp, tight)
        q = with_seed(ps, v)
        q.init == v || push!(clamped, "$(ps.block):$(ps.name)")
        push!(free, q)
    end
    return (θ = pack_θ(free), unmatched = unmatched, clamped = clamped)
end


# ============================================================
# Weighted loss
# ============================================================

"""
    active_moment_keys(spec) → Vector{Symbol}

The moments the objective scores (weight > 0), in the order of the deviation vector and of
`spec.W`.
"""
active_moment_keys(spec::SMMSpec) =
    Symbol[k for k in keys(spec.moments) if spec.moments[k].weight > 0.0]


"""
    compute_loss_matrix(m_model, spec, W) → Float64

Q = g′Wg / q_scale with g_k = m_k − m̂_k over the active moments, in raw moment units; all
scaling lives in `W` (Diagonal(weight_k / m̂_k²) for equal weights, Diagonal(1/σ̂²_k) for
the diagonal weighting). Raises if `W` does not match the number of active moments.
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

const _NO_EQUILIBRIUM = (Q = Inf, moments = nothing, model = nothing, label = :none, eq = nothing)

"""
    equilibrium_at(θ_unc, spec; Nx, Np_U, Np_S, constrained, sim) → (Q, moments, model, k)

The equilibrium the estimation scores at θ. θ includes the tightnesses, so the point is one
equilibrium, solved in one pass (`solve_model(...; tightness)`), with the vacancy costs free
entry implies there, `k = (U, S)`, installed in `model`. Minimising Q over (ω, θ_U, θ_S) is
therefore the best-fitting-equilibrium estimator. Infeasible (`Q = Inf`, `model = nothing`,
`k` NaN) when the solve fails, the residual battery does not certify it, or a scored moment
is not finite. For the joint spec each period is solved at its own coordinates, `Q` is the
sum, moments are keyed `<moment>@<window>` and `model`, `k` hold one entry per period.
"""
function equilibrium_at(
    θ_unc :: AbstractVector{Float64},
    spec  :: SMMSpec;
    Nx    :: Int = spec.run.Nx,
    Np_U  :: Int = spec.run.Np_U,
    Np_S  :: Int = spec.run.Np_S,
    constrained :: Bool = false,
    sim   :: SimParams = spec.sim,
)
    is_joint(spec) && return _joint_equilibrium_at(θ_unc, spec; Nx = Nx, Np_U = Np_U,
                                                   Np_S = Np_S, constrained = constrained,
                                                   sim = sim)
    none = (Q = Inf, moments = nothing, model = nothing, k = (U = NaN, S = NaN))
    cp, up, sp, tight = unpack_θ(θ_unc, spec; constrained = constrained)
    local model, m_model
    try
        model, solve_result = solve_model(cp, up, sp, sim; Nx = Nx, Np_U = Np_U, Np_S = Np_S,
                                          tightness = tight)
        solve_result.ok || return none
        m_model = model_moments(compute_equilibrium_objects(model))
    catch
        return none
    end
    all(k -> isfinite(getproperty(m_model, k)), active_moment_keys(spec)) || return none
    Q = compute_loss_matrix(m_model, spec, spec.W)
    return (Q = Q, moments = m_model, model = model,
            k = (U = model.unsk_par.k, S = model.skl_par.k))
end

"""
    _joint_equilibrium_at(θ_unc, spec; Nx, Np_U, Np_S, constrained, sim) → NamedTuple

`equilibrium_at` for the joint spec. The periods are solved in turn, never in parallel, so
an evaluation is the same alone or inside threaded work.
"""
function _joint_equilibrium_at(θ_unc, spec::SMMSpec; Nx, Np_U, Np_S, constrained, sim)
    pl  = period_list(spec)
    ws  = Tuple(w for (w, _, _) in pl)
    evs = [equilibrium_at(θ_unc[idx], s; Nx = Nx, Np_U = Np_U, Np_S = Np_S,
                          constrained = constrained, sim = sim) for (_, s, idx) in pl]
    per(f) = NamedTuple{ws}(Tuple(f(ev) for ev in evs))
    all(ev -> isfinite(ev.Q), evs) ||
        return (Q = Inf, moments = nothing, model = nothing,
                k = NamedTuple{ws}(Tuple((U = NaN, S = NaN) for _ in ws)))
    moments = NamedTuple(Symbol(k, "@", w) => v
                         for (w, ev) in zip(ws, evs) for (k, v) in pairs(ev.moments))
    return (Q = sum(ev.Q for ev in evs), moments = moments, model = per(ev -> ev.model),
            k = per(ev -> ev.k))
end

"""
    select_equilibrium(model, spec; branch = :best) → (Q, moments, model, label, eq)

At the vacancy costs of `model`, the attracting skilled equilibrium with the lowest Q, for
exercises that hold the costs fixed (counterfactuals, comparative statics). Every
attracting root `skilled_equilibria` finds (one, or L and H) is eligible if the residual
battery certifies it. `branch = :L` or `:H` restricts the choice to that side; a lone root
qualifies when on that side or when the map has no hump (`:U`), so a missing branch gives
`Q = Inf` rather than the other one. `label` is the selected root's (`:none` when
infeasible); `eq` is the detector's report.
"""
function select_equilibrium(model::Model, spec::SMMSpec; branch::Symbol = :best)
    local eq
    try
        eq = skilled_equilibria(model)
    catch
        return _NO_EQUILIBRIUM
    end

    best = (_NO_EQUILIBRIUM..., eq = eq)
    for r in eq.roots
        r.certified || continue
        branch === :best || r.label === branch || r.label === :U || continue
        local m_model
        try
            m_model = model_moments(compute_equilibrium_objects(r.model))
        catch
            continue
        end
        all(k -> isfinite(getproperty(m_model, k)), active_moment_keys(spec)) || continue
        Q = compute_loss_matrix(m_model, spec, spec.W)
        Q < best.Q && (best = (Q = Q, moments = m_model, model = r.model, label = r.label, eq = eq))
    end
    return best
end

"""
    equilibrium_tag(sel) → String

The equilibrium `select_equilibrium` selected, for a report: "unique", e.g. "H of 2", or
"none".
"""
equilibrium_tag(sel) = sel.eq === nothing || sel.model === nothing ? "none" :
                       length(sel.eq.roots) == 1 ? "unique" :
                       @sprintf("%s of %d", sel.label, length(sel.eq.roots))

"""
    smm_objective(θ_unc, spec; Nx, Np_U, Np_S, moments_out, constrained) → Float64

Q at the equilibrium θ fixes (`equilibrium_at`); Inf, never an exception, on any failure.
Corner equilibria (nobody trains, everybody trains, a market holding no one) are scored like
any other. When `moments_out` is given, it is filled with the scored moments in
`active_moment_keys` order.
"""
function smm_objective(
    θ_unc :: AbstractVector{Float64},
    spec  :: SMMSpec;
    Nx    :: Int = spec.run.Nx,
    Np_U  :: Int = spec.run.Np_U,
    Np_S  :: Int = spec.run.Np_S,
    moments_out :: Union{Nothing,AbstractVector{Float64}} = nothing,
    constrained :: Bool = false,
) :: Float64

    ev = equilibrium_at(θ_unc, spec; Nx = Nx, Np_U = Np_U, Np_S = Np_S, constrained = constrained)
    isfinite(ev.Q) || return Inf

    if moments_out !== nothing
        ks = active_moment_keys(spec)
        @assert length(moments_out) == length(ks)
        @inbounds for (i, k) in enumerate(ks)
            moments_out[i] = getproperty(ev.moments, k)
        end
    end
    return ev.Q
end

# Step of the attracting check in ln θ_j: wide against the sawtooth OJS entry puts on the
# skilled free-entry map (≈3e-4 per type), narrow against its curvature.
const ATTRACT_STEP = 0.01

"""
    attracting_check(θ_unc, spec) → (D, market, joint)
    attracting_check(D) → (D, market, joint)

Whether the equilibrium at θ attracts under a tâtonnement in the tightnesses at its own
vacancy costs. `D[i,j] = ∂ ln k_i / ∂ ln θ_j` (i, j ∈ U, S) by central differences of
`ATTRACT_STEP` (4 solves). `market`: D_UU < 0 and D_SS < 0; `joint`: tr D < 0 and
det D > 0. The second method takes a `D` measured elsewhere. For the joint spec, one check
per period, keyed by data window.
"""
function attracting_check(θ_unc::AbstractVector{Float64}, spec::SMMSpec)
    if is_joint(spec)
        pl = period_list(spec)
        return NamedTuple{Tuple(w for (w, _, _) in pl)}(
            Tuple(attracting_check(θ_unc[idx], s) for (_, s, idx) in pl))
    end
    cp, up, sp, tight = unpack_θ(θ_unc, spec)
    grid = (Nx = spec.run.Nx, Np_U = spec.run.Np_U, Np_S = spec.run.Np_S)
    function lnk(tt)
        model, sr = solve_model(cp, up, sp, spec.sim; grid..., tightness = tt)
        sr.ok || return (NaN, NaN)
        return (log(model.unsk_par.k), log(model.skl_par.k))
    end
    h = ATTRACT_STEP
    D = fill(NaN, 2, 2)
    U, S = tight.U, tight.S
    for (j, (tp, tm)) in enumerate((((U = U * exp(h), S = S), (U = U * exp(-h), S = S)),
                                    ((U = U, S = S * exp(h)), (U = U, S = S * exp(-h)))))
        kp = lnk(tp);  km = lnk(tm)
        D[1, j] = (kp[1] - km[1]) / (2h)
        D[2, j] = (kp[2] - km[2]) / (2h)
    end
    return attracting_check(D)
end

attracting_check(D::AbstractMatrix{Float64}) =
    (D = Matrix{Float64}(D), market = D[1, 1] < 0.0 && D[2, 2] < 0.0,
     joint = tr(D) < 0.0 && det(D) > 0.0)

"""
    attracting_line(chk) → String

The line a run prints about an attracting check, one per failing period of the baseline;
empty when every verdict holds.
"""
attracting_line(chk) = !haskey(chk, :D) ?
    join(["$w " * attracting_line(c) for (w, c) in pairs(chk) if !isempty(attracting_line(c))],
         "\n") :
    chk.market && chk.joint ? "" :
    @sprintf("ATTRACTING CHECK FAILED: market %s, joint %s  (D_UU %.3g  D_US %.3g  D_SU %.3g  D_SS %.3g)",
             chk.market ? "ok" : "fails", chk.joint ? "ok" : "fails",
             chk.D[1, 1], chk.D[1, 2], chk.D[2, 1], chk.D[2, 2])


# ============================================================
# Result type
# ============================================================

"""
    SMMResult

An estimate: the unconstrained point `theta_opt`, the free parameters there keyed
`block_name` (`params_opt`), its Q (`loss_opt`), whether the run finished on a deliberate
stop (`converged`), the iteration count and the spec. Serialised positionally in every
bundle: do not change its fields.
"""
struct SMMResult
    theta_opt  :: Vector{Float64}
    params_opt :: NamedTuple
    loss_opt   :: Float64
    converged  :: Bool
    iterations :: Int
    spec       :: SMMSpec
end


# ============================================================
# Corner reporting
# ============================================================

"""
    _count_corners(theta_unc, spec; tol = 0.02) → Int

The number of free parameters within `tol` (a fraction of the box width) of a bound.
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
    _corner_tags(theta_unc, spec; tol = 0.02) → String

The parameters `_count_corners` counts, with the bound each sits at, e.g.
` [skl:μ(lower), unsk:tight(upper)]`; "" when none.
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

A random start: each free parameter uniform in its box, in unconstrained coordinates.
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

State of the rate-based stop shared by SA and Nelder-Mead: stop when the incumbent improves
at less than `tol` per 100 evaluations, sustained over `span` PRODUCTIVE evaluations. A
stretch without improvement pauses the count — neither advancing nor resetting it — since
the search can sit against an infeasible wall and recover; the position mark advances on
every call, so a paused stretch is discarded rather than counted later.
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
    rate_stop!(rs, Q_best, iter) → Bool

Update `rs` with the incumbent `Q_best` at evaluation `iter`; true when the rule fires.
Disabled when `rs.tol` ≤ 0 or `Q_best` is not finite.
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

# ============================================================
# Proposal scale from measured yield
#
# The annealing mask and the DE population share two measurements at a point: how far
# each coordinate can move before Q rises by WIDTH_DQ (`_feasible_widths`), and how many
# coordinates a productive move touches.
# ============================================================

# Bisection cap of `_feasible_widths` in transformed units, shared by SA and DE. Not
# meant to bind (it does only on b_S, δ_S, ξ_S at base_fc): a bisection always returns a
# feasible step.
const _WIDTH_CAP = 0.20

"""
    _sparsity_draws!(cand, ksrc, seed, widths, per_k, sigma, rng;
                     match_total_displacement = false) → (cand, ksrc)

Fill `cand` with `per_k` draws around `seed` at every sparsity `k = 1:d`, each of k random
coordinates displaced by N(0, (σ_k·|width|)²), and record each draw's k in `ksrc`.
σ_k = `sigma` (a population: total displacement grows as √k), or `sigma/√k` with
`match_total_displacement` (comparing sparsities at equal total displacement, since
acceptance depends on k·σ_k²).
"""
function _sparsity_draws!(cand::Vector{Vector{Float64}}, ksrc::Vector{Int},
                          seed::Vector{Float64}, widths::Vector{Float64},
                          per_k::Int, sigma::Float64, rng;
                          match_total_displacement::Bool = false)
    d   = length(seed)
    idx = 0
    for k in 1:d
        σ_k = match_total_displacement ? sigma / sqrt(k) : sigma
        for _ in 1:per_k
            θ = copy(seed)
            for j in randperm(rng, d)[1:k]
                θ[j] += σ_k * abs(widths[j]) * randn(rng)
            end
            idx += 1; cand[idx] = θ; ksrc[idx] = k
        end
    end
    return cand, ksrc
end

"""
    _sparsity_score(Qs, ksrc, d, Q_ref) → (score, n_feas)

Per sparsity k, the mean improvement below `Q_ref` over its feasible draws, and the number
of feasible draws. Improvements are not counted (they vanish near an optimum). When no
sparsity improves, the score is 1/(1 + mean damage) instead.
"""
function _sparsity_score(Qs::Vector{Float64}, ksrc::Vector{Int}, d::Int, Q_ref::Float64)
    score = zeros(d); n_feas = zeros(Int, d); dmg = zeros(d)
    for k in 1:d
        q = [Qs[i] for i in eachindex(Qs) if ksrc[i] == k && isfinite(Qs[i])]
        n_feas[k] = length(q)
        isempty(q) && continue
        score[k] = mean(max.(0.0, Q_ref .- q))
        dmg[k]   = 1.0 / (1.0 + mean(q .- Q_ref))
    end
    all(score .<= 0) && (score = dmg)
    return score, n_feas
end

"""
    _kstar(basis, d) → Int

The sparsity a basis favours: its basis-weighted mean k, rounded into 1:d (the mean, not
argmax, which breaks ties of a near-flat table toward k = 1); `d` when the basis is zero.
"""
_kstar(basis::AbstractVector, d::Int) =
    sum(basis) > 0 ? clamp(round(Int, sum((1:d) .* basis) / sum(basis)), 1, d) : d

"""
    SAScale

The annealing proposal measured at a point (`sa_proposal_scale`); runtime only.

Fields
  step    per-coordinate proposal SD, the seed of the Corana steps
  widths  measured ΔQ half-widths, signed
  p_move  Bernoulli mask probability
  k_star  sparsity favoured by the diagnostic scan (0 when not run)
  n_meas  coordinates whose width could be measured
  n_feas  feasible draws in the scan
  n_imp   scan draws that improved on the seed
  feas_k  feasibility per sparsity, k = 1:d (its spread is the assumption p_move rests on)
"""
struct SAScale
    step   :: Vector{Float64}
    widths :: Vector{Float64}
    p_move :: Float64
    k_star :: Int
    n_meas :: Int
    n_feas :: Int
    n_imp  :: Int
    feas_k :: Vector{Float64}
end

"""
    sa_proposal_scale(seed, spec; p_move_scale, per_k, sigma, cap, step_fallback, rng, verbose)
        → SAScale

Measure the annealing proposal at `seed`: per coordinate, `sigma` times its measured
half-width (`_feasible_widths`, cap `cap`), or `step_fallback` where no width can be
measured; and the mask probability `p_move = p_move_scale / d` (one forced coordinate plus
a Binomial(d−1, p_move) tail, mean 1 + (d−1)·p_move). p_move is fixed, not adapted during
the walk: to second order acceptance and jump distance depend on k·σ² alone, which Corana
already steers through σ, and feasibility across k was measured flat (0.958–1.000).
`per_k > 0` runs a diagnostic sparsity scan at equal total displacement (`per_k·d` solves)
that sets nothing. Costs 24·d solves for the widths.
"""
function sa_proposal_scale(seed::Vector{Float64}, spec::SMMSpec;
                          p_move_scale  :: Float64 = 1.0,
                          per_k         :: Int     = 0,
                          sigma         :: Float64 = 0.33,
                          cap           :: Float64 = _WIDTH_CAP,
                          step_fallback :: Float64 = 0.01,
                          rng                      = Random.default_rng(),
                          verbose       :: Bool    = true)
    d = length(seed)
    if verbose
        @printf("  [SA scale]  measuring the proposal at the start: %d width solves%s\n",
                24 * d, per_k > 0 ?
                    @sprintf(" + %d scan draws (diagnostic)", per_k * d) : "")
        flush(stdout)
    end

    t0     = time()
    widths = _feasible_widths(seed, spec, cap)
    step   = [abs(widths[j]) > 0 ? sigma * abs(widths[j]) : step_fallback for j in 1:d]
    p_move = clamp(p_move_scale / d, 1.0 / d, 1.0)

    k_star = 0
    n_feas = 0
    n_imp  = 0
    feas_k = Float64[]
    if per_k > 0
        cand = Vector{Vector{Float64}}(undef, per_k * d)
        ksrc = Vector{Int}(undef, per_k * d)
        _sparsity_draws!(cand, ksrc, seed, widths, per_k, sigma, rng;
                         match_total_displacement = true)
        Qs = Vector{Float64}(undef, length(cand))
        Threads.@threads for i in eachindex(cand)
            Qs[i] = smm_objective(cand[i], spec)
        end

        # An infeasible seed leaves the worst feasible draw as the only finite reference.
        Q_seed = smm_objective(seed, spec)
        feas   = [i for i in eachindex(Qs) if isfinite(Qs[i])]
        Q_ref  = isfinite(Q_seed) ? Q_seed : (isempty(feas) ? Inf : maximum(Qs[feas]))
        score, n_feas_k = _sparsity_score(Qs, ksrc, d, Q_ref)
        k_star = _kstar(score, d)
        n_feas = length(feas)
        n_imp  = isfinite(Q_ref) ? count(q -> isfinite(q) && q < Q_ref, Qs) : 0
        feas_k = n_feas_k ./ per_k
    end

    sc = SAScale(step, widths, p_move, k_star, count(!=(0.0), widths),
                 n_feas, n_imp, feas_k)
    if verbose
        wm = [abs(w) for w in widths if abs(w) > 0]
        @printf("  [SA scale]  %.0fs:  widths %d/%d measurable, median %.3g, range %.2g‥%.2g;  step median %.3g, range %.2g‥%.2g;  p_move=%.4f (%.1f/d, mean %.1f coords/iter)%s\n",
                time() - t0, sc.n_meas, d,
                isempty(wm) ? NaN : median(wm),
                isempty(wm) ? NaN : minimum(wm), isempty(wm) ? NaN : maximum(wm),
                median(step), minimum(step), maximum(step),
                p_move, p_move_scale, 1 + (d - 1) * p_move,
                per_k > 0 ?
                    @sprintf(";  scan %d feasible %d improving, feasibility across k %.3f‥%.3f, k*=%d (diagnostic only)",
                             n_feas, n_imp, minimum(feas_k), maximum(feas_k), k_star) : "")
        flush(stdout)
    end
    return sc
end

"""
    _step_field(step_vec) → String

The step-range field of the SA trace.
"""
@inline function _step_field(step_vec::Vector{Float64})
    lo, hi = extrema(step_vec)
    lo == hi ? @sprintf("step range=%.6f", lo) :
               @sprintf("step range=%.6f‥%.6f", lo, hi)
end

"""
    _moved_field(n_moved) → String

Mean and range of the number of coordinates each proposal actually moved, from the
histogram `n_moved` (`n_moved[m + 1]` counts proposals that moved m coordinates).
"""
function _moved_field(n_moved::Vector{Int})
    n = sum(n_moved)
    n == 0 && return "—"
    lo = findfirst(>(0), n_moved) - 1
    hi = findlast(>(0), n_moved) - 1
    μ  = sum((m - 1) * n_moved[m] for m in eachindex(n_moved)) / n
    @sprintf("%.1f[%d‥%d]", μ, lo, hi)
end

"""
    _sa_loop(spec, theta_start; T0, step, max_iter, cooling_rate, cooling_exp,
             reheat_patience, reheat_factor, max_reheats, adapt_window, scale, corana_Ns,
             corana_c, step_floor_rel, rate_tol, rate_span, cooling_halflife, t0_rel,
             t0_accept, reheat_reset_tol, show_trace, trace_stride, checkpoint_path, rng)
        → (theta_best, Q_best, iters)

One simulated-annealing chain in unconstrained coordinates from `theta_start`.

- T0 ≤ 0: T0 solves exp(−t0_rel·Q0/T0) = t0_accept, anchored to the objective level.
- Cooling: T = T_reheat·2^(−t/H) for `cooling_halflife` H > 0; otherwise the logarithmic
  T_reheat·(log1p(rate)/log1p(rate·t))^exp, normalised so T(1) = T_reheat. Floor 1e-8.
- Proposal: a Bernoulli(p_move) mask plus one forced coordinate (no no-op proposals), each
  moved coordinate by N(0, step_j²). Inf proposals are rejected.
- Corana update every ≈ `corana_Ns` outcomes per coordinate: steer each coordinate's step
  toward acceptance 0.4–0.6 (factor `corana_c`), clamped to [step_floor_rel × its seeded
  step, its measured width]; coordinates not drawn are left alone.
- Stops on the rate rule (`rate_tol`, `rate_span`), on `max_iter`, or when the reheats are
  exhausted and the walk has not improved for `reheat_patience` steps.
- Reheat after `reheat_patience` steps without improvement: checkpoint the incumbent, then
  T ← T·`reheat_factor` (no cap), snapping back to the incumbent when the walk is more than
  `reheat_reset_tol` above Q_best.

`scale` is the measured proposal (`sa_proposal_scale`), measured here when `nothing`.
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
    scale            :: Union{Nothing,SAScale} = nothing,
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
    checkpoint_path  :: String  = "",
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
        # Anchored to Q, not to probed move sizes: probes straddle the feasibility edge and
        # their ΔQ inherits a tail orders of magnitude above a local move.
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
    _COOL_NUM = log1p(cooling_rate)

    win_fin = adapt_window > 0 ? zeros(Bool, adapt_window) : Bool[]
    win_acc = adapt_window > 0 ? zeros(Bool, adapt_window) : Bool[]

    # Per-coordinate proposal state, seeded from the measured widths (they span two orders
    # of magnitude, so no single step fits); floor and ceiling are each coordinate's own.
    d_free     = length(theta)
    sc         = scale === nothing ?
                 sa_proposal_scale(collect(float.(theta)), spec;
                                   step_fallback = step, rng = rng,
                                   verbose = show_trace) : scale
    step_vec   = copy(sc.step)
    step_seed  = copy(sc.step)
    step_ceil  = [abs(sc.widths[j]) > 0 ? abs(sc.widths[j]) : step for j in 1:d_free]
    p_move     = sc.p_move
    moved_idx  = Int[]                # this iteration's moved coordinates, reused
    n_prop     = zeros(Int, d_free)   # proposals per coordinate since the last update
    n_acc_j    = zeros(Int, d_free)   # of which accepted
    n_moved    = zeros(Int, d_free + 1)  # moved-count histogram, offset by one
    # Iterations for each coordinate to collect corana_Ns outcomes on average: drawn with
    # probability p_move by the mask, plus 1/d as the forced index.
    corana_win = max(1, ceil(Int, corana_Ns / (p_move + (1 - p_move) / d_free)))
    rate_stop  = RateStop(rate_tol, rate_span)
    win_idx = 0

    actual_iters = 0

    if show_trace
        n_corners_init = _count_corners(theta_best, spec)
        # Printed from the values the walk holds: the authoritative record of its settings.
        @printf("  [SA config]  mask p_move=%.4f (mean %.1f of %d coords/iter, Bernoulli + 1 forced);  per-coordinate step from measured widths, %d/%d measurable, Corana Ns=%d every %d iters;  cooling %s;  early stop %s\n",
                p_move, 1 + (d_free - 1) * p_move, d_free, sc.n_meas, d_free,
                corana_Ns, corana_win,
                cooling_halflife > 0 ?
                    @sprintf("geometric, half-life %d", cooling_halflife) :
                    "logarithmic (spends its descent in the first ~100 iters)",
                rate_tol > 0 ? @sprintf("ΔQ < %.3g per 100 iters over %d",
                                        rate_tol, rate_span) : "off")
        @printf("  [SA init]  Q0 = %s  T0=%.4f  %s  corners=%d/%d%s\n",
                isfinite(Q) ? @sprintf("%.6e", Q) : "Inf (rejected start)",
                T0, _step_field(step_vec),
                n_corners_init, length(spec.free), _corner_tags(theta_best, spec))
        # A rejected start is printed in full so it can be compared with the warm-start
        # bundle it was meant to be.
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
        T_current = cooling_halflife > 0 ?
            T_reheat * 2.0^(-(t_local - 1) / cooling_halflife) :
            T_reheat * (_COOL_NUM / log1p(cooling_rate * t_local))^cooling_exp
        T_current = max(T_current, 1e-8)

        theta_prop = copy(theta)
        empty!(moved_idx)
        for j in 1:d_free
            rand(rng) < p_move && push!(moved_idx, j)
        end
        j_force = rand(rng, 1:d_free)
        (j_force in moved_idx) || push!(moved_idx, j_force)
        moved = moved_idx
        for j in moved
            theta_prop[j] += step_vec[j] * randn(rng)
            n_prop[j] += 1
        end
        n_moved[length(moved) + 1] += 1
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

        # Trailing acceptance and feasibility, for the trace only.
        if adapt_window > 0
            win_idx          = mod1(win_idx + 1, adapt_window)
            win_fin[win_idx] = is_fin
            win_acc[win_idx] = is_fin && accept
        end

        # Corana step update.
        if t % corana_win == 0
            for j in 1:d_free
                n_prop[j] == 0 && continue
                p = n_acc_j[j] / n_prop[j]
                if p > 0.6
                    step_vec[j] *= 1 + corana_c * (p - 0.6) / 0.4
                elseif p < 0.4
                    step_vec[j] /= 1 + corana_c * (0.4 - p) / 0.4
                end
                step_vec[j] = clamp(step_vec[j], step_floor_rel * step_seed[j],
                                    step_ceil[j])
            end
            fill!(n_prop, 0)
            fill!(n_acc_j, 0)
        end

        # Rate stop before the reheat logic: a run this flat gains nothing from a reheat.
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
            # Checkpoint before raising T: the incumbent is what a resume wants.
            write_checkpoint(checkpoint_path, theta_best, Q_best, spec;
                             tag = @sprintf("SA reheat %d, iter %d", n_reheats, t))
            T_before   = T_current
            # No cap on the reheated temperature. A cap at T0 helps only together with a T0
            # recalibrated against the plateau; under the shipped T0 the reheats are the
            # walk's only warmth.
            T_current  = T_current * reheat_factor
            T_reheat   = T_current
            t_local    = 0
            # Snap back to the incumbent when the walk has drifted far above it; within the
            # tolerance it reheats in place, so a basin escape stays possible.
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
            @printf("  [SA iter=%5d]  curr=%-14s  best=%.6e  T=%.4f  %s  moved=%s  acc=%.2f  fin=%.2f  corners=%d/%d%s  reheats=%d\n",
                    t,
                    isfinite(Q) ? @sprintf("%.6e", Q) : "Inf",
                    Q_best, T_current, _step_field(step_vec),
                    _moved_field(n_moved),
                    w_acc, w_fin,
                    n_corners, length(spec.free), _corner_tags(theta_best, spec),
                    n_reheats)
            flush(stdout)
        end
    end

    if show_trace
        n_corners_done = _count_corners(theta_best, spec)
        @printf("  [SA done]  Q_best=%.6e  accepted %d/%d  finite %d/%d  moved=%s  corners=%d/%d%s  reheats=%d\n",
                Q_best, n_acc, actual_iters, n_fin, actual_iters,
                _moved_field(n_moved),
                n_corners_done, length(spec.free), _corner_tags(theta_best, spec),
                n_reheats)
        flush(stdout)
    end

    return theta_best, Q_best, actual_iters
end


"""
    _run_sa(spec; starts, parallel_steps, seed, random_init, scale_p_move, scale_per_k,
            scale_sigma, …) → (theta_best, Q_best, iters)

Simulated annealing from `starts` (one per cluster), or from a single start — a uniform draw
with `random_init`, `pack_theta(spec)` otherwise — when `starts` is empty. The proposal is
measured once, at the first start, and shared by every chain so their Q_best are comparable.
With several starts, each runs `parallel_steps` warm-up iterations, sequentially and without
checkpointing, with `Xoshiro(seed + j)`; the best is then annealed afresh for the remaining
budget with `Xoshiro(seed)`, and the better of the two results is returned. The remaining
keywords are `_sa_loop`'s.
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
    scale_p_move     :: Float64 = 1.0,
    scale_per_k      :: Int     = 0,
    scale_sigma      :: Float64 = 0.33,
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
    checkpoint_path  :: String  = "",
    rng                         = Random.default_rng(),
)
    start_set = if !isempty(starts)
        starts
    else
        [random_init ? _random_theta(spec, rng) : pack_theta(spec)]
    end

    scale = sa_proposal_scale(collect(float.(start_set[1])), spec;
                              p_move_scale = scale_p_move,
                              per_k = scale_per_k, sigma = scale_sigma,
                              step_fallback = step,
                              rng = rng, verbose = show_trace)

    if length(start_set) <= 1
        return _sa_loop(spec, start_set[1];
                        T0 = T0, step = step, max_iter = max_iter,
                        cooling_rate = cooling_rate, cooling_exp = cooling_exp,
                        reheat_patience = reheat_patience, reheat_factor = reheat_factor,
                        max_reheats = max_reheats, adapt_window = adapt_window,
                        scale = scale, checkpoint_path = checkpoint_path,
                        corana_Ns = corana_Ns, corana_c = corana_c,
                        step_floor_rel = step_floor_rel,
                        rate_tol = rate_tol, rate_span = rate_span,
                        cooling_halflife = cooling_halflife,
                        t0_rel = t0_rel, t0_accept = t0_accept,
                        reheat_reset_tol = reheat_reset_tol,
                        show_trace = show_trace,
                        trace_stride = trace_stride, rng = rng)
    end

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
                             scale = scale,
                             # A short dispersed chain would overwrite a better incumbent.
                             checkpoint_path = "",
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

    tb, qb, iters = _sa_loop(spec, chain_theta[jbest];
                             T0 = T0, step = step, max_iter = remaining,
                             cooling_rate = cooling_rate, cooling_exp = cooling_exp,
                             reheat_patience = reheat_patience, reheat_factor = reheat_factor,
                             max_reheats = max_reheats, adapt_window = adapt_window,
                             scale = scale, checkpoint_path = checkpoint_path,
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


"""
    pack_theta(spec) → Vector{Float64}

ASCII alias of `pack_θ` (smm_params.jl).
"""
pack_theta(spec) = pack_θ(spec)


# ============================================================
# Differential evolution
# ============================================================

"""
    _pick3(rng, n, exclude) → (a, b, c)

Three distinct indices in 1:n, all different from `exclude`.
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


"""
    _count_basins(pop, Q_pop, spec; min_size = 5) → Int

The number of clusters of at least `min_size` feasible members, by complete-linkage
hierarchical clustering in normalised constrained coordinates cut at the largest gap; 0
with fewer than 2·min_size feasible members. Trace only.
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
    generate_population(seed, spec, n_slots; per_k, sigma, cap, fill_from, fill_Q,
                        prev_widths, require_improvement, verbose, rng) → (pop, f, cr, table)

A DE population of `n_slots` members around `seed`, and the DE parameters the same draws
imply; used for the initial population and for every reheat.

- Widths: re-measured at `seed` (`_feasible_widths`, cap `cap`), so a reheat at a better
  point draws a tighter cloud; a coordinate with no measurable width keeps `prev_widths`.
- Draws: `per_k` at each sparsity k = 1:n_free, each moved coordinate by `sigma` widths.
- Allocation: slots by weight score_k·k/n_free (`_sparsity_score`), filled with the best
  draws of each k; shortfalls spill in descending weight. Admissible: improvements on the
  seed under `require_improvement` (a reheat), feasible draws otherwise or when the seed
  itself is infeasible. Then feasible non-improving draws, the best members of `fill_from`,
  infeasible draws and, last, seed copies. `pop[1]` is the seed.
- Barren (a reheat where nothing improves): half new feasible draws and half `fill_from`,
  both in random order; the caller treats the next stall as convergence.
- cr = `_kstar(score)/n_free` (of the weight on a barren call); f matches the DE step
  f·s_k·√2 to `sigma`·width over coordinates with a spread (fresh members only on a barren
  call), median, clamped to [0.1, 1.5].

`table` reports the allocation (`slots`, `slots_imp`, `n_carried`, `n_padded`, `barren`,
`widths`, …): `n_carried` > 0 means the fresh draws could not fill the population;
`n_padded` > 0 means the draw budget was too small.
"""
function generate_population(seed::Vector{Float64}, spec::SMMSpec, n_slots::Int;
                             per_k       :: Int     = 30,
                             sigma       :: Float64 = 0.33,
                             cap         :: Float64 = 0.20,
                             fill_from   :: Union{Nothing,Vector{Vector{Float64}}} = nothing,
                             fill_Q      :: Union{Nothing,Vector{Float64}}         = nothing,
                             prev_widths :: Union{Nothing,Vector{Float64}}         = nothing,
                             require_improvement :: Bool = false,
                             verbose     :: Bool    = true,
                             rng                   = Random.default_rng())
    npar = length(spec.free)

    if verbose
        @printf("  [DE pop]  starting population generator: %d slots, %d free parameters\n",
                n_slots, npar)
        @printf("  [DE pop]  step 1/2 — ΔQ<1 width per parameter: 12 bisections × 2 directions × %d params = %d solves\n",
                npar, 24 * npar)
        flush(stdout)
    end
    t0     = time()
    widths = _feasible_widths(seed, spec, cap)
    Q_seed = smm_objective(seed, spec)

    # A zero width would freeze the coordinate for the rest of the run.
    n_reverted = 0
    if prev_widths !== nothing
        for j in 1:npar
            if !(isfinite(widths[j]) && abs(widths[j]) > 0) &&
               isfinite(prev_widths[j]) && abs(prev_widths[j]) > 0
                widths[j] = prev_widths[j]
                n_reverted += 1
            end
        end
    end

    if verbose
        wf = filter(w -> isfinite(w) && w > 0, abs.(widths))
        @printf("  [DE pop]  widths done in %.0fs: %d/%d parameters admit a step, median %.3g, range %.2g–%.2g%s\n",
                time() - t0, length(wf), npar,
                isempty(wf) ? NaN : median(wf),
                isempty(wf) ? NaN : minimum(wf), isempty(wf) ? NaN : maximum(wf),
                n_reverted > 0 ? @sprintf("  (%d kept previous scale)", n_reverted) : "")
        @printf("  [DE pop]  step 2/2 — %d candidates at each sparsity k = 1:%d;  Q_seed=%.6e\n",
                per_k, npar, Q_seed)
        @printf("  [DE pop]  evaluating %d candidates...\n", per_k * npar)
        flush(stdout)
    end

    cand = Vector{Vector{Float64}}(undef, per_k * npar)
    ksrc = Vector{Int}(undef, per_k * npar)
    _sparsity_draws!(cand, ksrc, seed, widths, per_k, sigma, rng)
    t1 = time()
    Qs = Vector{Float64}(undef, length(cand))
    Threads.@threads for i in eachindex(cand)
        Qs[i] = smm_objective(cand[i], spec)
    end
    if verbose
        nf = count(isfinite, Qs)
        nb = isfinite(Q_seed) ? count(q -> isfinite(q) && q < Q_seed, Qs) : nf
        @printf("  [DE pop]  %d candidates in %.0fs: %d feasible, %d better than the seed  (selecting on %s)\n",
                length(cand), time() - t1, nf, nb,
                require_improvement ? "improvement" : "feasibility")
        if require_improvement && nb == 0
            @printf("  [DE pop]  BARREN: no candidate improves on the incumbent. Building the final\n")
            @printf("  [DE pop]          population as a 50/50 random mix with the previous generation.\n")
        end
        # Below about twice the slot count the fill takes nearly every admissible draw, and
        # the allocation tracks feasibility rather than the weight.
        n_pool = require_improvement ? nb : nf
        if n_pool > 0 && n_pool < 2 * n_slots
            @printf("  [DE pop]  WARNING: only %d %s for %d slots (%.1fx). The weight cannot\n",
                    n_pool, require_improvement ? "improving" : "feasible",
                    n_slots, n_pool / n_slots)
            @printf("  [DE pop]           select at this headroom — raise gen_per_k to ~%d or lower pop_size.\n",
                    ceil(Int, 2 * n_slots / (max(n_pool / length(cand), 0.05) * npar)))
        end
        flush(stdout)
    end

    # Admissible pool: improvements at a reheat, feasible draws at initialisation or when
    # the seed itself does not solve (Q_ref then keeps the score's shortfalls finite).
    feas_idx = [i for i in eachindex(Qs) if isfinite(Qs[i])]
    seed_ok  = isfinite(Q_seed)
    Q_ref    = seed_ok ? Q_seed :
               (isempty(feas_idx) ? Inf : maximum(Qs[feas_idx]))
    imp_idx  = seed_ok ? [i for i in feas_idx if Qs[i] < Q_seed] : feas_idx
    pool     = Set(require_improvement ? imp_idx : feas_idx)
    barren   = require_improvement && isempty(imp_idx)

    score, n_feas = _sparsity_score(Qs, ksrc, npar, Q_ref)

    # Allocation weight, with fallbacks for a non-finite (NaN ≤ 0 is false) or zero weight.
    w = score .* (collect(1:npar) ./ npar)
    all(isfinite, w) || (w = float.(n_feas))
    all(w .<= 0)     && (w = float.(n_feas))     # nothing scored: rank by feasibility
    (all(isfinite, w) && any(w .> 0)) || (w = collect(1:npar) ./ npar)
    want = round.(Int, n_slots .* w ./ sum(w))

    # Tier 1, admissible draws of each k up to its quota, then shortfalls in descending
    # weight (ascending k would hand them to the most feasible sparsities); tier 2, the
    # feasible draws outside the pool. `taken_imp` counts tier 1 only.
    order = Dict(k => sort([i for i in eachindex(Qs) if ksrc[i] == k && i in pool],
                           by = i -> Qs[i]) for k in 1:npar)
    spare = Dict(k => sort([i for i in feas_idx if ksrc[i] == k && !(i in pool)],
                           by = i -> Qs[i]) for k in 1:npar)
    pop   = Vector{Vector{Float64}}()
    taken = zeros(Int, npar)
    taken_imp = zeros(Int, npar)

    for k in 1:npar, i in order[k][1:min(want[k], length(order[k]))]
        push!(pop, cand[i]); taken[k] += 1; taken_imp[k] += 1
        length(pop) >= n_slots && break
    end
    for k in sortperm(w, rev = true)
        length(pop) >= n_slots && break
        for i in order[k][(taken_imp[k]+1):end]
            push!(pop, cand[i]); taken[k] += 1; taken_imp[k] += 1
            length(pop) >= n_slots && break
        end
    end
    for k in sortperm(w, rev = true)
        length(pop) >= n_slots && break
        for i in spare[k]
            push!(pop, cand[i]); taken[k] += 1
            length(pop) >= n_slots && break
        end
    end
    n_carried = 0
    n_fresh   = 0        # members of a barren mix that came from the new draws

    if barren
        if fill_from === nothing || isempty(fill_from)
            error("generate_population: a barren reheat needs fill_from, the previous generation")
        end
        empty!(pop); fill!(taken, 0); fill!(taken_imp, 0)
        for i in shuffle(rng, feas_idx)               # feasible, since none improve
            length(pop) >= n_slots ÷ 2 && break
            push!(pop, cand[i]); taken[ksrc[i]] += 1
        end
        n_fresh = length(pop)
        for i in shuffle(rng, eachindex(fill_from))
            length(pop) >= n_slots && break
            push!(pop, copy(fill_from[i])); n_carried += 1
        end
    end

    # Short of members: the previous generation's best, which keeps the difference vectors
    # spread where seed copies would collapse them.
    if length(pop) < n_slots && fill_from !== nothing && !isempty(fill_from)
        rank = fill_Q === nothing ? eachindex(fill_from) :
               sort(eachindex(fill_from), by = i -> fill_Q[i])
        for i in rank
            length(pop) >= n_slots && break
            push!(pop, copy(fill_from[i])); n_carried += 1
        end
    end

    # Still short: infeasible draws (they never win a comparison), in ascending k, then seed
    # copies.
    n_padded = 0
    if length(pop) < n_slots
        for k in 1:npar, i in [j for j in eachindex(Qs) if ksrc[j] == k && !isfinite(Qs[j])]
            length(pop) >= n_slots && break
            push!(pop, cand[i]); n_padded += 1
        end
    end
    while length(pop) < n_slots
        push!(pop, copy(seed)); n_padded += 1
    end
    pop[1] = copy(seed)

    # cr from the score, not the slots (quotas are capped by the improving draws available,
    # which biases them low) nor the weight (which counts k twice); the weight only on a
    # barren call, where every score is zero.
    k_star = _kstar(barren ? w : score, npar)
    cr_out = clamp(k_star / npar, 1.0 / npar, 1.0)

    # f: match the DE step to the displacement measured as productive. On a barren call only
    # the fresh members' spread was drawn at the current widths.
    P  = (barren && n_fresh >= 2) ? reduce(hcat, pop[1:n_fresh]) : reduce(hcat, pop)
    fs = Float64[]
    for j in 1:npar
        s_j = std(@view P[j, :])
        (s_j > 0 && isfinite(widths[j])) &&
            push!(fs, sigma * abs(widths[j]) / (s_j * sqrt(2)))
    end
    f_raw   = isempty(fs) ? 0.7 : median(fs)
    f_out   = clamp(f_raw, 0.1, 1.5)
    f_clamp = f_out != f_raw

    if verbose
        top = sortperm(taken, rev = true)[1:min(5, npar)]
        # Improving slots first, total in parentheses when they differ.
        @printf("  [DE pop]  top sparsities by allocated slots: %s\n",
                join([taken_imp[k] == taken[k] ?
                      @sprintf("k=%d→%d", k, taken[k]) :
                      @sprintf("k=%d→%d(%d)", k, taken_imp[k], taken[k])
                      for k in top if taken[k] > 0], "  "))
        n_fb = sum(taken) - sum(taken_imp)
        n_fb > 0 && @printf("  [DE pop]  %d of %d slots filled from feasible non-improving draws\n",
                            n_fb, sum(taken))
        @printf("  [DE pop]  done: f=%.3f%s  cr=%.3f (k*=%d of %d)  best candidate Q=%.6e\n",
                f_out, f_clamp ? @sprintf(" (clamped from %.3f)", f_raw) : "",
                cr_out, k_star, npar,
                isempty(feas_idx) ? NaN : minimum(Qs[feas_idx]))
        flush(stdout)
    end

    table = (k = collect(1:npar), n_feasible = n_feas, score = score,
             weight = w, slots = taken, slots_imp = taken_imp,
             n_carried = n_carried, n_padded = n_padded,
             n_reverted = n_reverted, widths = widths, barren = barren, n_fresh = n_fresh,
             n_feas_total = length(feas_idx), n_better = length(imp_idx))
    return pop, f_out, cr_out, table
end

"""
    _run_de(spec; max_iter, pop_size, f, cr, avg_tol, local_sigma, local_sigma_cap,
            gen_per_k, adapt_fcr, reheat_flat, reheat_rate, max_reheats, seed_bank,
            prev_optimum, show_members, show_gens, trace_stride, checkpoint_path, rng)
        → (theta_best, Q_best, gens)

DE/rand/1/bin in unconstrained coordinates from `pack_theta(spec)`, with a forced crossover
coordinate, greedy parent-child selection and per-member RNGs. The initial population comes
from `generate_population` (`seed_bank` members replace it when supplied; `prev_optimum`
takes slot 1); `adapt_fcr` takes f and cr from the generator. A stall — Q_best flat for
`reheat_flat` generations AND an improved-member rate below `reheat_rate` — rebuilds the
population at the best point and checkpoints it; the run stops at the first stall after a
barren reheat, after `max_reheats` reheats, at `max_iter`, or (when `avg_tol` > 0) when the
population mean is within `avg_tol` of Q_best.
"""
function _run_de(
    spec         :: SMMSpec;
    max_iter     :: Int     = 5000,
    pop_size     :: Int     = 0,
    f            :: Float64 = 0.65,
    cr           :: Float64 = 0.85,
    avg_tol      :: Float64 = 0.01,
    local_sigma  :: Float64 = 0.33,
    local_sigma_cap :: Float64 = 0.20,
    gen_per_k    :: Int     = 30,
    adapt_fcr    :: Bool    = false,
    reheat_flat  :: Int     = 6,
    reheat_rate  :: Float64 = 0.04,
    max_reheats  :: Int     = 50,
    seed_bank    :: Union{Nothing,SeedBank}        = nothing,
    prev_optimum :: Union{Nothing,Vector{Float64}} = nothing,
    show_members :: Bool    = false,
    show_gens    :: Bool    = true,
    trace_stride :: Int     = 10,
    checkpoint_path :: String = "",
    rng                     = Random.default_rng(),
)
    npar     = length(spec.free)
    pop_size = pop_size > 0 ? pop_size : 10 * npar
    theta0   = pack_theta(spec)

    member_rngs = begin
        seeds = rand(rng, UInt64, pop_size)
        [Random.Xoshiro(s) for s in seeds]
    end

    # A uniform draw over the box is the wrong scale by orders of magnitude; the generator
    # draws at the measured local widths.
    pop, f_gen, cr_gen, gen_table = generate_population(theta0, spec, pop_size;
                                                       per_k = gen_per_k, sigma = local_sigma,
                                                       cap = local_sigma_cap,
                                                       verbose = show_gens, rng = rng)
    adapt_fcr && ((f, cr) = (f_gen, cr_gen))
    if show_gens
        @printf("  [DE init]  generator: %d slots from %d draws  f=%.3f cr=%.3f (%s)\n",
                pop_size, gen_per_k * npar, f, cr,
                adapt_fcr ? "from yield" :
                    @sprintf("set; yield would give f=%.3f cr=%.3f", f_gen, cr_gen))
        flush(stdout)
    end

    if seed_bank === nothing
        pop[1] = copy(theta0)
    else
        # Cluster members round-robin, best Q first; generated members fill the rest.
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
    stagnation   = 0
    actual_gens  = 0
    n_reheats    = 0
    Q_prev       = Q_best
    reheats_done = false        # set by a barren reheat: the next stall is convergence
    last_widths  = gen_table.widths

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

        # Stagnation of Q_best, strict: members improve on their parents in every
        # generation, so the improved count alone never reaches zero.
        if Q_best < Q_prev
            stagnation = 0
        else
            stagnation += 1
        end
        Q_prev = Q_best

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

        # Stall: both conditions (Q_best alone is flat for up to 12 generations mid-descent;
        # the improved rate alone dips on ordinary generations). The reheat rebuilds the
        # population at the best point, re-measuring the widths there.
        stalled = stagnation >= reheat_flat && n_imp / pop_size < reheat_rate
        if stalled
            if reheats_done
                show_gens && @printf("  [DE]  stop: converged — stalled after the final reheat, Q=%.6e\n",
                                     Q_best)
                flush(stdout)
                break
            end
            if n_reheats >= max_reheats
                show_gens && @printf("  [DE]  stop: %d reheats exhausted\n", max_reheats)
                flush(stdout)
                break
            end
            Q_before  = Q_best
            n_reheats += 1
            pop, f_gen, cr_gen, gen_table =
                generate_population(theta_best, spec, pop_size;
                                    per_k = gen_per_k, sigma = local_sigma,
                                    cap = local_sigma_cap,
                                    fill_from = pop, fill_Q = Q_pop,
                                    prev_widths = last_widths,
                                    require_improvement = true,
                                    verbose = show_gens, rng = rng)
            adapt_fcr && ((f, cr) = (f_gen, cr_gen))
            last_widths = gen_table.widths
            Q_pop = Vector{Float64}(undef, pop_size)
            Threads.@threads for i in 1:pop_size
                Q_pop[i] = smm_objective(pop[i], spec)
            end
            Threads.atomic_add!(n_evals, pop_size)
            i_best     = argmin(Q_pop)
            Q_best     = Q_pop[i_best]
            theta_best = copy(pop[i_best])
            stagnation = 0

            # A barren reheat is the last: its population is searched, and the next stall
            # ends the run.
            if gen_table.barren
                reheats_done = true
            end

            if show_gens
                write_checkpoint(checkpoint_path, theta_best, Q_best, spec;
                                 tag = @sprintf("DE reheat %d, gen %d", n_reheats, gen))
                @printf("  [DE reheat %d]  gen=%d  Q %.6e -> %.6e  f=%.3f cr=%.3f  k*=%d/%d  better=%d feas=%d of %d%s\n",
                        n_reheats, gen, Q_before, Q_best, f, cr,
                        argmax(gen_table.slots), npar,
                        gen_table.n_better, gen_table.n_feas_total, gen_per_k * npar,
                        gen_table.barren ?
                            @sprintf("  BARREN → FINAL (mix %d new / %d carried)",
                                     gen_table.n_fresh, gen_table.n_carried) :
                            @sprintf("  (carried %d)", gen_table.n_carried))
                flush(stdout)
            end
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
# Seed-bank helpers
# ============================================================

"""
    _bank_clusters(bank) → Vector{Vector{Int}}

Candidate indices grouped by cluster label, each group sorted by ascending Q.
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

One SA start per cluster (its best-Q member), then `prev_optimum` when given; empty when
neither is given.
"""
function _sa_starts_from_bank(bank::Union{Nothing,SeedBank},
                              prev_optimum::Union{Nothing,Vector{Float64}})
    starts = Vector{Vector{Float64}}()
    if bank !== nothing
        for grp in _bank_clusters(bank)
            isempty(grp) && continue
            push!(starts, copy(bank.candidates[grp[1]]))
        end
    end
    prev_optimum !== nothing && push!(starts, copy(prev_optimum))
    return starts
end

"""
    _seed_pop_from_bank(bank, pop_size) → Vector{Vector{Float64}}

Up to `pop_size` bank members taken round-robin across clusters, best Q first within each;
fewer when the bank runs out.
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
        advanced || break
    end
    return pop
end


# ============================================================
# Feasible widths and the Nelder-Mead simplex
# ============================================================

"""
    FeasibleSimplexer(spec, step) <: Optim.Simplexer

Initial Nelder-Mead simplex with vertex j the seed moved along coordinate j by its measured
width (`_feasible_widths`, capped at `step` in transformed units), so every vertex has a
finite Q near the seed's. Optim's AffineSimplexer placed every vertex outside the feasible
region here.
"""
struct FeasibleSimplexer{S} <: Optim.Simplexer
    spec :: S
    step :: Float64
end

# The ΔQ contour the step scales are measured against, absolute in Q units (Q's level is
# dominated by moments the model cannot fit), just above the measured evaluation-noise
# floor of 0.59.
const _WIDTH_DQ = Ref(env_setting(:WIDTH_DQ, 1.0))

"""
    _feasible_widths(θ, spec, cap; nbisect = 12, dq = WIDTH_DQ) → Vector{Float64}

Per coordinate, the largest step in [0, cap] (better of the two directions, signed negative
when downward) keeping Q finite and below Q(θ) + `dq`, by `nbisect` bisections per
direction: 2·nbisect·d solves, threaded over coordinates. A merely feasible step is far
beyond the scale on which Q varies, hence the ΔQ criterion.
"""
function _feasible_widths(θ::AbstractVector, spec::SMMSpec, cap::Float64;
                          nbisect::Int = 12,
                          dq::Float64 = _WIDTH_DQ[])
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

"""
    Optim.simplexer(S::FeasibleSimplexer, initial_x) → Vector

The simplex of `FeasibleSimplexer`; a coordinate with no usable step keeps the seed value
(a degenerate direction).
"""
function Optim.simplexer(S::FeasibleSimplexer, initial_x::Tx) where {Tx}
    d       = length(initial_x)
    simplex = Tx[copy(initial_x) for _ in 1:d+1]
    w       = _feasible_widths(initial_x, S.spec, S.step)
    for j in 1:d
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
    run_smm(spec; method = :de, seed_bank, prev_optimum, checkpoint_path, rng, theta_start,
            fd_step, max_iter, g_tol, nm_rate_tol, nm_rate_span, nm_simplex_step,
            sa_scale_p_move, sa_scale_per_k, sa_scale_sigma, sa_corana_Ns, sa_corana_c,
            sa_step_floor_rel, sa_rate_tol, sa_rate_span, sa_halflife) → SMMResult

Minimise Q for `spec` by `method` (:sa, :de, :sa_de, :neldermead, :lbfgs, :bfgs), print and
return the estimate. Optimiser settings come from `spec.run`, except those passed as
keywords here (SMMRunParams cannot gain fields), each named as the field that feeds it.

- `seed_bank`, `prev_optimum`: SA starts and DE members (:sa, :de, :sa_de).
- `checkpoint_path`: bundle overwritten with the incumbent at every SA and DE reheat.
- `theta_start`: start of the Optim methods (default `pack_theta(spec)`), avoiding a
  θ → t → θ round trip.
- `fd_step`: central-difference step of the :lbfgs/:bfgs gradient, in t units; 1e-6 is
  measured (columns stable over [3e-7, 3e-6]).
- `max_iter`: iteration cap of the Optim methods; 0 keeps `spec.run.nm_max_iter`.
- `g_tol` > 0: stop on ‖∇Q‖∞ ≤ g_tol, with the function and step tolerances and the
  rate/no-improvement callback all off; 0 keeps the Nelder-Mead rules.
- `nm_rate_tol`, `nm_rate_span`: the `RateStop` rule of the Optim methods.
- `nm_simplex_step`: cap of `FeasibleSimplexer`; ≤ 0 keeps Optim's AffineSimplexer.
- `sa_scale_*`: the SA proposal measurement (`sa_proposal_scale`); `sa_corana_*`,
  `sa_step_floor_rel`: its step adaptation; `sa_rate_*`, `sa_halflife`: its stop and cooling.
"""
function run_smm(
    spec         :: SMMSpec;
    method       :: Symbol = :de,
    seed_bank    :: Union{Nothing,SeedBank}        = nothing,
    prev_optimum :: Union{Nothing,Vector{Float64}} = nothing,
    checkpoint_path :: String = "",
    rng                  = Random.default_rng(),
    theta_start  :: Union{Nothing,Vector{Float64}} = nothing,
    fd_step      :: Float64 = 1e-6,
    max_iter     :: Int     = 0,
    g_tol        :: Float64 = 0.0,
    nm_rate_tol    :: Float64 = 0.05,
    nm_rate_span   :: Int     = 300,
    nm_simplex_step :: Float64 = 0.2,
    sa_scale_p_move :: Float64 = 1.0,
    sa_scale_per_k  :: Int     = 0,
    sa_scale_sigma  :: Float64 = 0.33,
    sa_corana_Ns    :: Int     = 20,
    sa_corana_c     :: Float64 = 2.0,
    sa_step_floor_rel :: Float64 = 1e-4,
    sa_rate_tol     :: Float64 = 0.0,
    sa_rate_span    :: Int     = 0,
    sa_halflife     :: Int     = 0,
) :: SMMResult

    r    = spec.run
    npar = length(spec.free)


    @printf("\nStarting SMM  (%s,  %d free params)\n", method, npar)
    flush(stdout)

    # One call site per stage, shared by :sa, :de and :sa_de, so a setting reaches every
    # branch or none.
    _sa_stage(sp, starts) = _run_sa(
        sp;
        starts          = starts,
        max_iter        = r.sa_max_iter,
        T0              = r.sa_T0,
        step            = r.sa_step,
        cooling_rate    = r.sa_cooling_rate,
        cooling_exp     = r.sa_cooling_exp,
        reheat_patience = r.sa_reheat_patience,
        reheat_factor   = r.sa_reheat_factor,
        max_reheats     = r.sa_max_reheats,
        adapt_window    = r.sa_adapt_window,
        scale_p_move    = sa_scale_p_move,
        scale_per_k     = sa_scale_per_k,
        scale_sigma     = sa_scale_sigma,
        corana_Ns       = sa_corana_Ns,
        corana_c        = sa_corana_c,
        step_floor_rel  = sa_step_floor_rel,
        rate_tol        = sa_rate_tol,
        rate_span       = sa_rate_span,
        cooling_halflife = sa_halflife,
        t0_rel          = r.sa_t0_rel,
        t0_accept       = r.sa_t0_accept,
        reheat_reset_tol = r.sa_reheat_reset_tol,
        parallel_steps  = r.sa_parallel_steps,
        seed            = r.sa_seed,
        random_init     = r.sa_random_init,
        show_trace      = r.show_trace_generations,
        trace_stride    = r.trace_stride,
        checkpoint_path = checkpoint_path,
        rng             = rng,
    )

    _de_stage(sp, bank, prev) = _run_de(
        sp;
        max_iter     = r.de_max_iter,
        pop_size     = r.de_pop_size > 0 ? r.de_pop_size : 10 * npar,
        f            = r.de_f,
        cr           = r.de_cr,
        avg_tol      = r.de_avg_tol,
        local_sigma  = r.de_local_sigma,
        gen_per_k    = r.de_gen_per_k,
        adapt_fcr    = r.de_adapt_fcr,
        reheat_flat  = r.de_reheat_flat,
        reheat_rate  = r.de_reheat_rate,
        max_reheats  = r.de_max_reheats,
        seed_bank    = bank,
        prev_optimum = prev,
        show_members = r.show_trace_members,
        show_gens    = r.show_trace_generations,
        trace_stride = r.trace_stride,
        checkpoint_path = checkpoint_path,
        rng          = rng,
    )

    if method == :sa_de
        # Annealing crosses the box; DE then builds the joint moves inside the basin. The
        # start stays in contention, so the reported Q never exceeds the start's.
        θ_start = pack_theta(spec)
        Q_start = smm_objective(θ_start, spec)

        @printf("\n[stage 1/2]  simulated annealing\n"); flush(stdout)
        θ_sa, Q_sa, it_sa = _sa_stage(spec, _sa_starts_from_bank(seed_bank, prev_optimum))

        if isfinite(Q_start) && !(isfinite(Q_sa) && Q_sa <= Q_start)
            # The revert is unconditional; only the message is gated, at 1e-9, since two
            # evaluations of the same θ differ by a few ulp (threaded reductions).
            if !isfinite(Q_sa) || (Q_sa - Q_start) > 1e-9
                @printf("[stage 1/2]  annealing ended at Q=%s, above the start; DE continues from the start\n",
                        isfinite(Q_sa) ? @sprintf("%.6e", Q_sa) : "Inf")
                flush(stdout)
            end
            θ_sa, Q_sa = θ_start, Q_start
        elseif !isfinite(Q_sa)
            println("[stage 1/2]  annealing returned no feasible point; handing the original start to DE")
            flush(stdout)
        end
        # DE starts from SA's point through the spec init; the seed bank is not forwarded
        # (its box-wide spread would undo the generator's local scale).
        spec_de = isfinite(Q_sa) ? _spec_with_init(spec, θ_sa) : spec
        @printf("\n[stage 2/2]  differential evolution  (SA gave Q=%.6e)\n",
                Q_sa); flush(stdout)
        θ_de, Q_de, it_de = _de_stage(spec_de, nothing, nothing)

        cands = [(Q_start, θ_start), (Q_sa, θ_sa), (Q_de, θ_de)]
        filter!(c -> isfinite(c[1]), cands)
        loss_opt, theta_opt = isempty(cands) ? (Inf, θ_start) : cands[argmin(first.(cands))]
        niters    = it_sa + it_de
        converged = isfinite(loss_opt)
        conv_why  = isfinite(loss_opt) ? "sa+de-stop" : "infeasible"
        @printf("\n[stages done]  start Q=%.6e → SA Q=%.6e → DE Q=%.6e   reported Q=%.6e\n",
                Q_start, Q_sa, Q_de, loss_opt); flush(stdout)

    elseif method == :de
        theta_opt, loss_opt, niters = _run_de(
            spec;
            max_iter     = r.de_max_iter,
            pop_size     = r.de_pop_size > 0 ? r.de_pop_size : 10 * npar,
            f            = r.de_f,
            cr           = r.de_cr,
            avg_tol      = r.de_avg_tol,
            local_sigma  = r.de_local_sigma,
            gen_per_k    = r.de_gen_per_k,
            adapt_fcr    = r.de_adapt_fcr,
            reheat_flat  = r.de_reheat_flat,
            reheat_rate  = r.de_reheat_rate,
            max_reheats  = r.de_max_reheats,
            seed_bank    = seed_bank,
            prev_optimum = prev_optimum,
            show_members = r.show_trace_members,
            show_gens    = r.show_trace_generations,
            trace_stride = r.trace_stride,
            checkpoint_path = checkpoint_path,
            rng          = rng,
        )
        converged = isfinite(loss_opt)
        conv_why  = isfinite(loss_opt) ? "de-stop" : "infeasible"

    elseif method == :sa
        theta_opt, loss_opt, niters =
            _sa_stage(spec, _sa_starts_from_bank(seed_bank, prev_optimum))
        converged = isfinite(loss_opt)
        conv_why  = isfinite(loss_opt) ? "sa-stop" : "infeasible"

    elseif method in (:neldermead, :lbfgs, :bfgs)
        theta0        = theta_start === nothing ? pack_theta(spec) : copy(theta_start)
        iter_count    = Ref(0)
        best_loss     = Ref(Inf)
        best_theta    = Ref(copy(theta0))   # incumbent, for corner reporting
        last_improve  = Ref(0)       # evaluation at which best_loss last improved
        stopped_early = Ref(false)
        stop_reason   = Ref(:none)
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

        # Early stops (true halts Optim), only once a finite incumbent exists:
        # no improvement for nm_no_improve evaluations, or the rate rule.
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

        # BackTracking for the gradient methods: HagerZhang evaluates the gradient (2d
        # solves) at every trial step, and with the pinned Optim/LineSearches pair it
        # raises a BoundsError before the first iteration.
        opt_method = (method == :neldermead) ?
                       (nm_simplex_step > 0 ?
                          Optim.NelderMead(initial_simplex =
                              FeasibleSimplexer(spec, nm_simplex_step)) :
                          Optim.NelderMead()) :
                     (method == :lbfgs) ?
                       Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()) :
                       Optim.BFGS(linesearch  = Optim.LineSearches.BackTracking())

        options   = Optim.Options(iterations = (max_iter > 0 ? max_iter : r.nm_max_iter),
                                  f_reltol      = (g_tol > 0 ? 0.0 : r.nm_f_tol),
                                  x_abstol      = (g_tol > 0 ? 0.0 : r.nm_x_tol),
                                  g_abstol      = (g_tol > 0 ? g_tol : r.nm_g_tol),
                                  callback      = (g_tol > 0 ? nothing : nm_stop_cb),
                                  show_trace = false)

        # The gradient calls smm_objective rather than obj_traced, so its 2d solves do not
        # count as evaluations of the rate rule; the line search runs through obj_traced.
        result = if method === :neldermead
            Optim.optimize(obj_traced, theta0, opt_method, options)
        else
            grad_calls = Ref(0)
            # Central differences; one-sided where one side is infeasible, zero where
            # neither solves, both counted in the printed line (a one-sided difference
            # across a jump is huge and corrupts the LBFGS memory).
            function polish_grad!(Gv, t)
                q0 = smm_objective(t, spec)
                n_onesided = 0
                n_dead     = 0
                @inbounds for k in eachindex(t)
                    tp = copy(t);  tp[k] += fd_step
                    tm = copy(t);  tm[k] -= fd_step
                    qp = smm_objective(tp, spec)
                    qm = smm_objective(tm, spec)
                    Gv[k] = if isfinite(qp) && isfinite(qm)
                        (qp - qm) / (2 * fd_step)
                    elseif isfinite(qp) && isfinite(q0)
                        n_onesided += 1
                        (qp - q0) / fd_step
                    elseif isfinite(qm) && isfinite(q0)
                        n_onesided += 1
                        (q0 - qm) / fd_step
                    else
                        n_dead += 1
                        0.0
                    end
                end
                grad_calls[] += 1
                kmax = argmax(abs.(Gv))
                gmax = abs(Gv[kmax])
                @printf("  [%s grad %4d]  ‖g‖∞=%.4e (%s)  ‖g‖₂=%.4e  Q=%s%s%s%s\n",
                        method, grad_calls[], gmax,
                        first(param_symbol(spec.free[kmax]), 6),
                        sqrt(sum(abs2, Gv)),
                        isfinite(q0) ? @sprintf("%.6f", q0) : "Inf",
                        n_onesided > 0 ? @sprintf("  1sided=%d", n_onesided) : "",
                        n_dead     > 0 ? @sprintf("  dead=%d", n_dead)       : "",
                        (g_tol > 0 && gmax <= g_tol) ? "   ← below g_tol" : "")
                flush(stdout)
                return nothing
            end
            Optim.optimize(obj_traced, polish_grad!, theta0, opt_method, options)
        end
        theta_opt = Optim.minimizer(result)
        loss_opt  = smm_objective(theta_opt, spec)
        # A deliberate early stop counts as converged; so does an iteration cap reached
        # 500 evaluations after the last improvement (independent of nm_no_improve).
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

    res = SMMResult(theta_opt, _params_to_namedtuple(theta_opt, spec), loss_opt, converged, niters, spec)
    print_results(res; why = conv_why, k = equilibrium_at(theta_opt, spec).k)
    return res
end


# ============================================================
# Result display and saving
# ============================================================

"""
    print_results(res; why = "", k = nothing)

Print the estimate `res`: free parameters (the vacancy costs `k` in place of the
tightnesses), fixed parameters, Q, convergence and its reason `why`. The reason and the
costs are arguments because `SMMResult` cannot gain fields.
"""
function print_results(res::SMMResult; why::AbstractString = "", k = nothing)
    @printf("\n╔══════════════════════════════════════════════════════╗\n")
    @printf("║  SMM Estimates                                       ║\n")
    @printf("╠══════════════════════════════════════════════════════╣\n")
    # Column widths 2 + 12 + 20 + 20 = 54, the frame's interior width.
    @printf("  %s%s%20s\n", padr("block", 12), padr("param", 20), "estimate")
    @printf("  %s\n", "─"^(12 + 20 + 20))
    for ps in res.spec.free
        is_tightness(ps) && continue
        key = Symbol(string(ps.block) * "_" * string(ps.name))
        val = hasproperty(res.params_opt, key) ? res.params_opt[key] : NaN
        @printf("  %s%s%20.5f\n", padr(ps.block, 12), padr(param_symbol(ps), 20), val)
    end
    if k !== nothing
        for ((blk, _, _, sym), v) in zip(vacancy_cost_rows(res.spec), k_values(k))
            @printf("  %s%s%20.5f\n", padr(blk, 12), padr(sym, 20), v)
        end
    end
    pinned = pinned_table(res.spec)
    if length(pinned) > 0
        @printf("\n  Fixed:\n")
        for (_, sym, v) in pinned
            @printf("    %s%20.5f\n", padr(sym, 30), v)
        end
    end
    @printf("\n  Q = %.8e  |  converged = %s%s  |  iters = %d\n",
            res.loss_opt, res.converged, isempty(why) ? "" : @sprintf(" (%s)", why), res.iterations)
    @printf("╚══════════════════════════════════════════════════════╝\n\n")
    flush(stdout)
end


"""
    write_checkpoint(path, θ, Q, spec; tag = "")

Write the incumbent (θ, Q) to `path` as an `:smm_checkpoint` bundle (no `sim`: a resume
point, not an estimate), with provenance parsed from the path; nothing when `path` is
empty. A failure warns and the run continues.
"""
function write_checkpoint(path::String, θ::AbstractVector, Q::Float64,
                          spec::SMMSpec; tag::String = "")
    isempty(path) && return
    try
        res = SMMResult(collect(float.(θ)), _params_to_namedtuple(collect(float.(θ)), spec),
                        Q, false, 0, spec)
        write_bundle(path; result = res, spec = spec, stage = :smm_checkpoint,
                     provenance = provenance_from_path(path), tag = tag)
        @printf("  [checkpoint]  %s  Q=%.6e -> %s\n", tag, Q, basename(path))
        flush(stdout)
    catch e
        @warn "checkpoint failed (run continues)" tag exception=e
    end
end

"""
    save_results(res, path; k)

Write the estimate `res` to the CSV `path`: one row per free parameter (block, name, label,
estimate, bounds, symbol), per vacancy cost (`k`, no bounds) and per pinned parameter, then
Q, convergence and iterations as comment lines.
"""
function save_results(res::SMMResult, path::String; k)
    open(path, "w") do io
        println(io, "block,name,label,estimate,lb,ub,fixed,symbol")
        for ps in res.spec.free
            key = Symbol(string(ps.block) * "_" * string(ps.name))
            val = hasproperty(res.params_opt, key) ? res.params_opt[key] : NaN
            @printf(io, "%s,%s,%s,%.8f,%.8f,%.8f,false,%s\n",
                    ps.block, ps.name, ps.label, val, ps.lb, ps.ub, param_symbol(ps))
        end
        for ((blk, nm, label, sym), v) in zip(vacancy_cost_rows(res.spec), k_values(k))
            @printf(io, "%s,%s,%s,%.8f,,,false,%s\n", blk, nm, label, v, sym)
        end
        for (key, sym, v) in pinned_table(res.spec)
            @printf(io, "fixed,%s,%s,%.8f,,,true,%s\n", key, key, v, sym)
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

"""
    _params_to_namedtuple(θ_unc, spec) → NamedTuple

The free parameters at the unconstrained point `θ_unc`, constrained, keyed `block_name`.
"""
function _params_to_namedtuple(θ_unc::AbstractVector{Float64}, spec::SMMSpec)
    d = Dict{Symbol, Float64}()
    for (i, ps) in enumerate(spec.free)
        d[Symbol(string(ps.block) * "_" * string(ps.name))] = _to_constrained(θ_unc[i], ps.lb, ps.ub)
    end
    return NamedTuple(d)
end


"""
    _spec_with_init(spec, theta_unc) → SMMSpec

`spec` with its starting values set to the unconstrained point `theta_unc`.
"""
function _spec_with_init(spec::SMMSpec, theta_unc::Vector{Float64})
    new_free = [
        ParamSpec(ps.block, ps.name, ps.lb, ps.ub,
                  _to_constrained(theta_unc[i], ps.lb, ps.ub), ps.label)
        for (i, ps) in enumerate(spec.free)
    ]
    return SMMSpec(new_free, spec.fixed, spec.moments, spec.sim, spec.run, spec.W, spec.q_scale)
end


"""
    period_bundle(b, window) → bundle

Bundle `b` as data window `window` sees it. For the baseline's estimate, period `window` as
a single-period estimate: that period's coordinates and spec, and its share of every
per-period or per-coordinate record (standard errors, vacancy costs, attracting check); its
`result.loss_opt` is NaN, since the stored Q is the joint one. Any other bundle is returned
unchanged.
"""
function period_bundle(b, window::Symbol)
    (b.spec === nothing || !is_joint(b.spec)) && return b
    pl = period_list(b.spec)
    i  = findfirst(p -> p[1] === window, pl)
    i === nothing && error("period_bundle: $window is not a period of this estimate.")
    _, s, idx = pl[i]
    θ  = b.result.theta_opt[idx]
    sp = _spec_with_init(s, θ)
    n  = length(b.spec.free)
    at(x) = x isa NamedTuple ? (haskey(x, window) ? x[window] : map(at, x)) : x
    se = b.se === nothing ? nothing :
         map(v -> v isa AbstractVector && length(v) == n ? v[idx] : v, b.se)
    return merge(b, (result = SMMResult(θ, _params_to_namedtuple(θ, sp), NaN,
                                        b.result.converged, b.result.iterations, sp),
                     spec = sp, se = se, vacancy_costs = at(b.vacancy_costs),
                     attracting = at(b.attracting)))
end

############################################################
# cf_common.jl — what the counterfactual exercises share (RoySearch)
#
# Library file, loaded by counterfactuals_main.jl. The exercises read the transitions
# transition_main.jl saved; a new path is solved by the transition's own outer fixed point,
# warm-started from a stored one.
#
#   load_transition        the saved transition, its path rebuilt
#   self_path              a steady state as a one-step transition of itself
#   with_training_cost     the model with its training cost scaled
#   stationary_at_costs!   the steady state at given vacancy costs, by Newton continuation
#   solve_cf_transition    a new path at the exercise tolerances
#   window_mean            a path series averaged over the crisis window
#   switchers              who starts and who stops training between two steady states
#   steady_state_pools     the reference pools of a steady state
#   path_pools             the unemployed pools at a date of a path
#   set_summary            a set's mass and mean aptitudes
#   cf_log_start, cf_log   the exercise log
############################################################

"""
    load_transition(crisis) -> NamedTuple

The transition `crisis` saved by transition_main.jl at `POINT_ESTIMATE`, with its path
rebuilt as a `TransitionPath`, its `TransitionParams`, and its endpoint models `m0`
(baseline) and `m1` (crisis), both as solved, before the path consumed any copy.
"""
function load_transition(crisis::Symbol)
    tr = read_transition(crisis, W_SUFFIX, POINT_ESTIMATE)
    tr === nothing && error(
        "No transition for :$crisis at point :$(POINT_ESTIMATE): " *
        "$(basename(transition_file(crisis, W_SUFFIX, POINT_ESTIMATE))) is not in " *
        "$(TRANS_OUT_DIR). Run code/transition/transition_main.jl first, with " *
        "POINT_ESTIMATE = :$(POINT_ESTIMATE).")
    hasproperty(tr, :types) ||
        error("$(basename(transition_file(crisis, W_SUFFIX, POINT_ESTIMATE))) predates v61.4.0 " *
              "(no per-type date-0 values); run code/transition/transition_main.jl again.")
    tr.result.converged ||
        @warn "The stored :$crisis transition did not converge (free-entry gap $(tr.result.final_dist))."
    s  = tr.settings
    tp = TransitionParams(T_max = s.T_max, N_steps = s.N_steps, horizon = s.horizon, tol = s.tol,
                          maxit = s.maxit, damp = s.damp, verbose = false)
    path = TransitionPath((getfield(tr.path, f) for f in fieldnames(TransitionPath))...)
    path.tgrid == time_grid(tp) ||
        error("The stored :$crisis path does not lie on the time grid of its own settings.")
    return (tr = tr, path = path, tp = tp, m0 = tr.model_z0, m1 = tr.model_z1,
            months = tr.crisis_window.months)
end

"""
    self_path(model) -> (path, tp)

The one-step transition of the steady state `model` to itself (as `stationary_record`),
with its path: its first date holds the steady state's values and masses.
"""
function self_path(model::Model)
    tp = TransitionParams(T_max = 0.5, N_steps = 1, maxit = 1, verbose = false)
    _, p = solve_transition_path(deepcopy(model), deepcopy(model), tp; scenario = :steady_state)
    return p, tp
end

"""
    with_training_cost(model, s) -> Model

A copy of `model` whose training cost c(a_S) is `s` times its own (the code stores ln c, so
ln c + ln s), keeping `model`'s caches as the warm start of the solve that must follow.
"""
function with_training_cost(model::Model, s::Real)
    s > 0 || error("with_training_cost: the scale must be positive, got $s.")
    m  = deepcopy(model);  cp = m.common
    m.common = CommonParams(; (f => (f === :c ? cp.c + log(s) : getfield(cp, f))
                               for f in fieldnames(CommonParams))...)
    return m
end

# Newton continuation of `stationary_at_costs!`: tolerance on ln k (the solve at given
# tightness meets its battery to about 1e-8), forward-difference step and largest step in
# ln θ, iteration cap.
const STAT_TOL     = 1e-7
const STAT_FD      = 1e-4
const STAT_MAXSTEP = 0.25
const STAT_MAXIT   = 25

"""
    stationary_at_costs!(model, kU, kS, θ0) -> NamedTuple

The stationary equilibrium of `model` at the vacancy costs (kU, kS) (the code's units),
by Newton's method on ln θ from θ0 = (U, S) on the residual ln k(θ) − ln k, k(θ) the costs
`solve_at_tightness!` implies. From a known equilibrium it follows that equilibrium's
branch. On return the model holds the solution and exactly (kU, kS). Returns
`(ok, θU, θS, resid, iters)`.
"""
function stationary_at_costs!(model::Model, kU::Float64, kS::Float64, θ0)
    function F(l)
        r = solve_at_tightness!(model, exp(l[1]), exp(l[2]))
        r.ok || return nothing
        return [log(model.unsk_par.k / kU), log(model.skl_par.k / kS)]
    end
    l = [log(θ0.U), log(θ0.S)];  f = F(l);  it = 0
    f === nothing && return (ok = false, θU = NaN, θS = NaN, resid = Inf, iters = 0)
    while maximum(abs, f) >= STAT_TOL && it < STAT_MAXIT
        it += 1
        J = zeros(2, 2)
        for c in 1:2
            lh = copy(l);  lh[c] += STAT_FD
            fh = F(lh)
            fh === nothing && return (ok = false, θU = exp(l[1]), θS = exp(l[2]),
                                      resid = maximum(abs, f), iters = it)
            J[:, c] = (fh .- f) ./ STAT_FD
        end
        step = -(J \ f)
        step .*= min(1.0, STAT_MAXSTEP / maximum(abs, step))
        l .+= step
        f = F(l)
        f === nothing && return (ok = false, θU = exp(l[1]), θS = exp(l[2]), resid = Inf, iters = it)
    end
    model.unsk_par = _with_k(model.unsk_par, kU)
    model.skl_par  = _with_k(model.skl_par, kS)
    return (ok = maximum(abs, f) < STAT_TOL, θU = exp(l[1]), θS = exp(l[2]),
            resid = maximum(abs, f), iters = it)
end

"""
    solve_cf_transition(m0, m1, tp; guess = nothing, label) -> (result, path, seconds)

The transition from the steady state `m0` to `m1` on the dates of `tp` (the stored
horizon), at `CF_TOL`, `CF_MAXIT`, `CF_DAMP`, from the tightness path
`guess = (tgrid, θU, θS)`, on copies of the models.
"""
function solve_cf_transition(m0::Model, m1::Model, tp::TransitionParams; guess = nothing,
                             label::Symbol = :counterfactual)
    tq = TransitionParams(T_max = tp.T_max, N_steps = tp.N_steps, horizon = tp.horizon,
                          tol = CF_TOL, maxit = CF_MAXIT, damp = CF_DAMP, verbose = true)
    t0 = time()
    res, path = solve_transition_path(deepcopy(m0), deepcopy(m1), tq; scenario = label, guess = guess)
    res.converged ||
        @warn "$label did not converge in $(CF_MAXIT) iterations (free-entry gap $(res.final_dist))."
    return (result = res, path = path, seconds = time() - t0)
end

"""
    window_mean(series, tgrid, months) -> Float64

The unweighted mean of a path series over its dates t < `months` (the crisis window lies
in the uniform fine grid).
"""
window_mean(series::AbstractVector, tgrid::AbstractVector, months::Real) =
    (k = findall(<(months), tgrid); sum(series[k]) / length(k))

# ── Who trains, and the pools they are compared with ───────────────────────────
# The one implementation every exercise uses. A cell is a node (a_U, a_S) of the ability
# grid (rows a_U, columns a_S), with population mass m(a) (`grids.copula.W2`); τ(a) is the
# covered fraction of the cell on the training side of the frontier (`unsk_cache.τT`).
# Between two steady states the covered regions of a cell nest, so its starters are
# m·max(τ¹ − τ⁰, 0) and its stoppers m·max(τ⁰ − τ¹, 0).

# The reference pools of a steady state (`steady_state_pools`), in the order reported.
const REFERENCE_POOLS = (:unskilled_unemployed, :not_training, :skilled_unemployed, :trained)

"""
    switch_fractions(τ0, τ1) -> (starts, stops)

The fractions of a cell that start and that stop training between two steady states with
training fractions τ0 and τ1, each clamped to [0, 1].
"""
function switch_fractions(τ0::Real, τ1::Real)
    τ0 = clamp(τ0, 0.0, 1.0);  τ1 = clamp(τ1, 0.0, 1.0)
    return (starts = max(τ1 - τ0, 0.0), stops = max(τ0 - τ1, 0.0))
end

"""
    switchers(m0, m1) -> (starters, stoppers)

Cell masses of the workers who start and who stop training from the steady state `m0` to
the steady state `m1`: each cell's `switch_fractions` times its population. The two
models must share the ability grid and its distribution.
"""
function switchers(m0::Model, m1::Model)
    W2 = m0.grids.copula.W2
    W2 == m1.grids.copula.W2 || error("switchers: the two steady states have different populations.")
    f = switch_fractions.(m0.unsk_cache.τT, m1.unsk_cache.τT)
    return (starters = W2 .* getfield.(f, :starts), stoppers = W2 .* getfield.(f, :stops))
end

"""
    steady_state_pools(model; o = compute_equilibrium_objects(model)) -> NamedTuple

Cell masses of the reference pools (`REFERENCE_POOLS`) of the steady state `model`:
  unskilled_unemployed  every unemployed worker searching the unskilled market: the untrained
                        unemployed u_U (the never-hired included) and the trained crossers
                        d·m_S (all of the draining share is unemployed, as equilibrium.jl
                        counts it)
  not_training          the side of each cell away from the training frontier, (1 − τ)·m
  skilled_unemployed    the trained unemployed searching the skilled market, u_S − d·m_S
  trained               every trained worker, m_S
and `unskilled_corner`, the part of the unskilled unemployed pool whose reservation quality
is at the corner p* = 0: the untrained unemployed at their row's p*_U(a_U), the crossers at
the reservation of the slice that hires them (the training slice's p*_t for the enclosed
share ω, `enclosed_crossers`; the searching slice's otherwise).
"""
function steady_state_pools(model::Model; o = compute_equilibrium_objects(model))
    uc = model.unsk_cache
    cross  = o.d .* o.mS_mat
    corner = similar(cross)
    @inbounds for j in axes(cross, 2), i in axes(cross, 1)
        ω   = enclosed_crossers(o.d[i, j], clamp(uc.τT[i, j], 0.0, 1.0))
        atU = uc.pstar[i] <= 0.0;  atT = uc.pstar_t[i, j] <= 0.0
        corner[i, j] = atU * o.uU[i, j] + cross[i, j] * ((1.0 - ω) * atU + ω * atT)
    end
    return (unskilled_unemployed = o.uU .+ cross,
            not_training         = (1.0 .- clamp.(uc.τT, 0.0, 1.0)) .* model.grids.copula.W2,
            skilled_unemployed   = max.(o.uS_mat .- cross, 0.0),
            trained              = o.mS_mat,
            unskilled_corner     = corner)
end

"""
    path_pools(path, model, n) -> NamedTuple

Cell masses of the two unemployed pools at date n of `path` under its z₁ `model`:
`unskilled_unemployed` u_U + d·u_S and `skilled_unemployed` (1 − d)·u_S, d the date's
crossing fraction (`_d_matrix`), the seekers the transition's free entry counts.
"""
function path_pools(path::TransitionPath, model::Model, n::Int)
    d = _d_matrix(path, model, n);  uS = view(path.uS, :, :, n)
    return (unskilled_unemployed = path.uU[:, :, n] .+ d .* uS, skilled_unemployed = (1.0 .- d) .* uS)
end

"""
    set_summary(w, x, pop) -> (mass, mean_aU, mean_aS)

A set of workers with cell masses `w` on the ability grid `x`: its mass as a share of the
population `pop`, and its mean a_U and mean a_S weighted by the cell masses (NaN for an
empty set). A non-positive `w` (the negative part of a difference) keeps its sign in `mass`
and is weighted by |w|.
"""
function set_summary(w::AbstractMatrix, x::AbstractVector, pop::Real)
    a = abs.(w);  tot = sum(a);  mass = sum(w) / pop
    tot > 0.0 || return (mass = mass, mean_aU = NaN, mean_aS = NaN)
    return (mass = mass, mean_aU = sum(a .* x) / tot, mean_aS = sum(a .* x') / tot)
end

"""
    cf_log(dir, line)

Print `line` and append it to `dir/log.txt`.
"""
function cf_log(dir::AbstractString, line::AbstractString)
    println(line);  flush(stdout)
    open(io -> println(io, line), joinpath(dir, "log.txt"), "a")
    return nothing
end

"""
    cf_log_start(dir, title) -> dir

Start `dir/log.txt` afresh with `title` and the run's version, point, crises, date and
threads, and print the title banner.
"""
function cf_log_start(dir::AbstractString, title::AbstractString)
    open(joinpath(dir, "log.txt"), "w") do io
        println(io, title)
        println(io, "RoySearch v$(ROYSEARCH_VERSION), point $(POINT_ESTIMATE), crises $(CRISES), ",
                Dates.format(now(), "yyyy-mm-dd HH:MM"), ", $(Threads.nthreads()) threads")
    end
    println("\n", "="^65, "\n  ", title, "\n", "="^65);  flush(stdout)
    return dir
end

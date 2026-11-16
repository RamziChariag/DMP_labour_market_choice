############################################################
# cf_common.jl — what every counterfactual exercise shares (RoySearch)
#
# LIBRARY FILE — loaded by counterfactuals_main.jl.
#
# A transition saved by transition_main.jl holds the whole path, both endpoint models, the
# settings it was solved at and every type's date-0 values and masses.  The exercises read it and never solve it again; the
# exercises that need a new path solve it with the repo's own outer fixed point
# (`solve_transition_path`), warm-started from a stored one.
#
#   load_transition(crisis)               the saved transition, its path rebuilt
#   self_path(model)                      a steady state as a one-step transition of itself
#   with_training_cost(model, s)          the model with its training cost scaled by s
#   stationary_at_costs!(model, kU, kS, θ0)  the stationary equilibrium at given vacancy
#                                         costs, by Newton continuation from θ0
#   unskilled_branch(model, θU, θS, kU)   the zeros of unskilled free entry on a θ_U scan,
#                                         and which of them θU is
#   solve_cf_transition(m0, m1, tp; guess) a new path at the exercise tolerances
#   window_mean(series, tgrid, months)    the mean of a path series over the crisis window
#   cf_log(dir, line) / cf_log_start(dir, title)  console line with flush, copied to log.txt
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

The one-step transition of a steady state to itself, `stationary_record`'s, keeping the
path: its date 1 holds the steady state's values and masses to the stationary tolerance.
"""
function self_path(model::Model)
    tp = TransitionParams(T_max = 0.5, N_steps = 1, maxit = 1, verbose = false)
    _, p = solve_transition_path(deepcopy(model), deepcopy(model), tp; scenario = :steady_state)
    return p, tp
end

"""
    with_training_cost(model, s) -> Model

A copy of `model` whose training cost c(a_S) is `s` times its own.  The code stores ln c, so
the change is ln c + ln s.  The copy keeps the caches of `model`, the warm start of the
stationary solve that must follow.
"""
function with_training_cost(model::Model, s::Real)
    s > 0 || error("with_training_cost: the scale must be positive, got $s.")
    m  = deepcopy(model);  cp = m.common
    m.common = CommonParams(; (f => (f === :c ? cp.c + log(s) : getfield(cp, f))
                               for f in fieldnames(CommonParams))...)
    return m
end

# The Newton continuation of `stationary_at_costs!`: residual tolerance in ln k, forward-
# difference step and largest step in ln θ, iteration cap.  The stationary solve at given
# tightness meets its own battery to about 1e-8, so the residual is not asked for less.
const STAT_TOL     = 1e-7
const STAT_FD      = 1e-4
const STAT_MAXSTEP = 0.25
const STAT_MAXIT   = 25

"""
    stationary_at_costs!(model, kU, kS, θ0) -> NamedTuple

The stationary equilibrium of `model` at the vacancy costs (kU, kS) (months of mean
new-match output, the code's units), found from the tightnesses θ0 = (U, S) by Newton's
method on ln θ: each evaluation is `solve_at_tightness!`, which returns the costs free entry
implies at that tightness, and the residual is ln k(θ) − ln k.  Starting from a known
equilibrium (the estimate's) it follows that equilibrium's branch rather than searching for
one; `unskilled_branch` checks the branch.  On return the model holds the solution and
exactly the costs (kU, kS).
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

# The θ_U scan of `unskilled_branch`, in ln θ_U around the point checked, and its nodes.
const BRANCH_SCAN_LOG = (-3.0, 1.5)
const BRANCH_SCAN_N   = 31

"""
    unskilled_branch(model, θU, θS, kU) -> NamedTuple

Every zero of unskilled free entry of `model` at θ_S = θS and unskilled vacancy cost kU, on a
scan of θ_U (`BRANCH_SCAN_LOG`, `BRANCH_SCAN_N`): the sign changes of ln k_U(θ_U) − ln kU,
with k_U(θ_U) the cost `solve_at_tightness!` implies.  `rank` numbers the zero nearest θU
from the top (1 = the highest zero), so two equilibria are on the same branch when their
scans find as many zeros and give them the same rank.
"""
function unskilled_branch(model::Model, θU::Float64, θS::Float64, kU::Float64)
    m = deepcopy(model)
    grid = exp.(range(log(θU) + BRANCH_SCAN_LOG[1], log(θU) + BRANCH_SCAN_LOG[2];
                      length = BRANCH_SCAN_N))
    g = map(grid) do θ
        r = solve_at_tightness!(m, θ, θS)
        r.ok ? log(m.unsk_par.k / kU) : NaN
    end
    roots = Float64[]
    for n in 1:(length(grid) - 1)
        (isfinite(g[n]) && isfinite(g[n + 1]) && sign(g[n]) != sign(g[n + 1])) || continue
        w = g[n] / (g[n] - g[n + 1])
        push!(roots, exp((1.0 - w) * log(grid[n]) + w * log(grid[n + 1])))
    end
    sort!(roots, rev = true)
    rank = isempty(roots) ? 0 : argmin(abs.(log.(roots) .- log(θU)))
    return (roots = roots, rank = rank, n_failed = count(!isfinite, g))
end

"""
    solve_cf_transition(m0, m1, tp; guess = nothing, label) -> (result, path, seconds)

The perfect-foresight path from the steady state `m0` to the steady state `m1` on the time
grid of `tp`, by the repo's outer fixed point at the exercise tolerances (`CF_TOL`,
`CF_MAXIT`, `CF_DAMP`), from the tightness path `guess = (tgrid, θU, θS)`.  The horizon is
the stored transition's: the horizon test is not repeated.
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

# The mean of a path series over the dates of the crisis window, t < months.
window_mean(series::AbstractVector, tgrid::AbstractVector, months::Real) =
    (k = findall(<(months), tgrid); sum(series[k]) / length(k))

function cf_log(dir::AbstractString, line::AbstractString)
    println(line);  flush(stdout)
    open(io -> println(io, line), joinpath(dir, "log.txt"), "a")
    return nothing
end

function cf_log_start(dir::AbstractString, title::AbstractString)
    open(joinpath(dir, "log.txt"), "w") do io
        println(io, title)
        println(io, "RoySearch v$(ROYSEARCH_VERSION), point $(POINT_ESTIMATE), crises $(CRISES), ",
                Dates.format(now(), "yyyy-mm-dd HH:MM"), ", $(Threads.nthreads()) threads")
    end
    println("\n", "="^65, "\n  ", title, "\n", "="^65);  flush(stdout)
    return dir
end

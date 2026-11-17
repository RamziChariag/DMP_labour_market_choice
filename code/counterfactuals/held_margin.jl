############################################################
# held_margin.jl — each crisis with the education margin held at the baseline (RoySearch)
#
# Library file, loaded by counterfactuals_main.jl; `run_held_margin` is Exercise 1.
#
# The held crisis economy: the crisis parameters and estimated vacancy costs, with who
# trains (τ) and which market trained workers search (d) held at the baseline steady
# state's choices, known to every agent (`HeldChoices`, `Model.held`), and tightness set by
# free entry. Its stationary equilibrium is located on a scan of both free-entry conditions
# (`equilibrium_pairs`) and must be unique. The held path from the baseline steady state
# re-solves free entry at every date of the stored transition, warm-started from the
# estimated (open) path.
#
# Unemployment is per head of population, by market: u_U the unskilled unemployed (the
# never-hired included), u_S every trained unemployed worker (those searching the unskilled
# market included, as the data count them), u = u_U + u_S. With u⁰ the baseline's,
#   Δ_j(s) = u_j^held(s) − u_j^open(s),   share_j(s) = Δ_j(s) / (u^held(s) − u⁰),
# and s = ∞ compares the two end steady states. Also written: the unskilled free-entry
# curve of each crisis economy, open and held.
############################################################

# Equilibrium scan (`equilibrium_pairs`): range in ln θ_U around the estimate, and nodes.
const PAIR_SCAN_LOG = (-3.0, 1.5)
const PAIR_SCAN_N   = 31

# Free-entry curve: θ_U from CURVE_SPAN[1] to CURVE_SPAN[2] times the estimate, CURVE_N nodes.
const CURVE_SPAN = (0.3, 1.6)
const CURVE_N    = 81

# Skilled free entry at a given θ_U (`skilled_free_entry_root`): secant in ln θ_S to this
# tolerance on ln k_S (the solve meets its battery to about 1e-8), at most this many steps;
# a root is accepted within 100 times the tolerance.
const SKL_ROOT_TOL   = 1e-7
const SKL_ROOT_MAXIT = 25

"""
    skilled_free_entry_root(m, θU, l0, kS) -> Float64

ln θ_S at which skilled free entry holds at cost kS with θ_U = θU, by the secant method in
ln θ_S from `l0`; NaN if none is found. `m` is left solved at the root.
"""
function skilled_free_entry_root(m::Model, θU::Float64, l0::Float64, kS::Float64)
    f(l) = (r = solve_at_tightness!(m, θU, exp(l)); r.ok ? log(m.skl_par.k / kS) : NaN)
    a = l0;  fa = f(a)
    isfinite(fa) || return NaN
    b = a + 0.03 * sign(fa);  fb = f(b);  it = 0
    while isfinite(fb) && abs(fb) > SKL_ROOT_TOL && it < SKL_ROOT_MAXIT && fb != fa
        it += 1
        c = clamp(b - fb * (b - a) / (fb - fa), b - 0.5, b + 0.5)
        a, fa = b, fb;  b = c;  fb = f(b)
    end
    return abs(fb) < 100 * SKL_ROOT_TOL ? b : NaN
end

"""
    free_entry_curve(model, kU, kS, θref, grid; objects = false) -> NamedTuple

On the θ_U nodes `grid`: θ_S solving skilled free entry at cost kS, and the unskilled
free-entry residual gU = ln k_U(θ_U, θ_S) − ln kU, k_U the cost free entry implies (an
equilibrium is a zero). Continuation outward from the node nearest θref.U, each direction
starting from θref.S; a node without a skilled root is NaN. With `objects`, also per head
of population the unskilled unemployed, trainees and trained, and the mean a_U of the
unskilled unemployed. Works on a copy of `model`.
"""
function free_entry_curve(model::Model, kU::Float64, kS::Float64, θref, grid::AbstractVector;
                          objects::Bool = false)
    m = deepcopy(model);  x = m.grids.x;  N = length(grid)
    θS = fill(NaN, N);  gU = fill(NaN, N)
    uU = fill(NaN, N);  aU = fill(NaN, N);  tr = fill(NaN, N);  mS = fill(NaN, N)
    n0 = argmin(abs.(log.(grid) .- log(θref.U)))
    for nodes in (n0:N, (n0 - 1):-1:1)
        lS = log(θref.S)
        for n in nodes
            l = skilled_free_entry_root(m, Float64(grid[n]), lS, kS)
            isfinite(l) || continue
            lS = l;  θS[n] = exp(l);  gU[n] = log(m.unsk_par.k / kU)
            objects || continue
            o = compute_equilibrium_objects(m);  uw = vec(sum(o.uU, dims = 2))
            uU[n] = o.agg_uU / o.total_pop;  aU[n] = sum(x .* uw) / sum(uw)
            tr[n] = o.agg_t / o.total_pop;   mS[n] = o.agg_mS / o.total_pop
        end
    end
    return (θU = collect(Float64, grid), θS = θS, gU = gU, uU = uU, mean_aU = aU,
            training = tr, skilled = mS)
end

"""
    equilibrium_pairs(model, kU, kS, θref) -> NamedTuple

Every stationary equilibrium of `model` at the vacancy costs (kU, kS) on the free-entry
curve over `PAIR_SCAN_LOG` around θref.U (`PAIR_SCAN_N` nodes): the sign changes of gU,
located by linear interpolation in ln θ_U and ln θ_S. Returns the scan and the pairs,
highest θ_U first, each with its stability in the unskilled market (`attracting` when gU
falls through zero as θ_U rises).
"""
function equilibrium_pairs(model::Model, kU::Float64, kS::Float64, θref)
    grid = exp.(range(log(θref.U) + PAIR_SCAN_LOG[1], log(θref.U) + PAIR_SCAN_LOG[2];
                      length = PAIR_SCAN_N))
    sc = free_entry_curve(model, kU, kS, θref, grid)
    pairs_ = NamedTuple[]
    for n in 1:(PAIR_SCAN_N - 1)
        g0 = sc.gU[n];  g1 = sc.gU[n + 1]
        (isfinite(g0) && isfinite(g1) && sign(g0) != sign(g1)) || continue
        w = g0 / (g0 - g1)
        push!(pairs_, (θU = exp((1 - w) * log(sc.θU[n]) + w * log(sc.θU[n + 1])),
                       θS = exp((1 - w) * log(sc.θS[n]) + w * log(sc.θS[n + 1])),
                       attracting = g0 > g1))
    end
    sort!(pairs_, by = p -> -p.θU)
    return (scan = sc, pairs = pairs_)
end

"""
    held_end_state(L) -> NamedTuple

The stationary equilibrium of the held crisis economy of the transition `L` (`L.m1` with
`L.m0`'s choices held) at the estimated vacancy costs: the scan's equilibrium, which must be
the only one, refined by `stationary_at_costs!`. Returns the solved model, the Newton
result and the scan.
"""
function held_end_state(L)
    mh = deepcopy(L.m1);  mh.held = HeldChoices(L.m0)
    kU = L.m1.unsk_par.k;  kS = L.m1.skl_par.k
    eq = equilibrium_pairs(mh, kU, kS, (U = L.m1.unsk_cache.θ, S = L.m1.skl_cache.θ))
    length(eq.pairs) == 1 || error(
        "The held crisis economy has $(length(eq.pairs)) equilibria on the scan, (θ_U, θ_S) = " *
        join((@sprintf("(%.5f, %.5f)", p.θU, p.θS) for p in eq.pairs), ", ") *
        "; Exercise 1 is defined for exactly one.")
    p  = eq.pairs[1]
    st = stationary_at_costs!(mh, kU, kS, (U = p.θU, S = p.θS))
    st.ok || error(@sprintf("The held crisis equilibrium near (θ_U, θ_S) = (%.5f, %.5f) was not refined (Newton residual %.2e).",
                            p.θU, p.θS, st.resid))
    return (model = mh, solve = st, eq = eq)
end

"""
    unemployment_per_head(path, n, pop) -> NamedTuple

Unemployment at date n of `path` per head of the population `pop`: `uU`, `uS` and their sum
`u` (see the file header), and the within-market rates as `_build_result` measures them,
ur_U = u_U/(u_U + e_U) and ur_S = u_S/m_S.
"""
function unemployment_per_head(path::TransitionPath, n::Int, pop::Float64)
    uU = sum(view(path.uU, :, :, n));  uS = sum(view(path.uS, :, :, n))
    eU = sum(view(path.eUs, :, :, n)) + sum(view(path.eUt, :, :, n));  mS = sum(view(path.mS, :, :, n))
    return (uU = uU / pop, uS = uS / pop, u = (uU + uS) / pop,
            ur_U = uU + eU > 1e-14 ? uU / (uU + eU) : 0.0, ur_S = mS > 1e-14 ? uS / mS : 0.0)
end

# The markets of the tables: name and field of `unemployment_per_head`.
const HELD_MARKETS = (("U", :uU), ("S", :uS), ("total", :u))

"""
    run_held_margin(crises)

For each crisis, writes
  held_margin_path_<crisis>.csv         by date: unemployment per head and rates, open and held, Δ_j and share_j
  held_margin_<crisis>.csv              by market: Δ_j and share_j at the horizon, over the window and at ∞
  held_margin_end_states_<crisis>.csv   the baseline, open crisis and held crisis steady states
  held_margin_equilibria_<crisis>.csv   the equilibrium of the held crisis economy
  held_margin_equilibria_scan_<crisis>.csv   the scan that locates it
  held_margin_fe_curve_<crisis>.csv     the unskilled free-entry curve, open and held
  held_margin_<crisis>.jls              the above with both TransitionResults
"""
function run_held_margin(crises)
    dir = cf_log_start(out_counterfactuals(:held_margin), "Exercise 1: the education margin held at the baseline")
    for crisis in crises
        t0 = time()
        L  = load_transition(crisis);  m1 = L.m1
        kU = m1.unsk_par.k;  kS = m1.skl_par.k;  θ1 = (U = m1.unsk_cache.θ, S = m1.skl_cache.θ)
        pop = sum(m1.grids.copula.W2)

        H = held_end_state(L)
        cf_log(dir, @sprintf("[%s] held crisis equilibrium: θ_U %.6f, θ_S %.6f (open %.6f, %.6f); unique on the scan",
                             crisis, H.solve.θU, H.solve.θS, θ1.U, θ1.S))
        hp = solve_cf_transition(L.m0, H.model, L.tp; label = :held_margin,
                                 guess = (tgrid = L.path.tgrid, θU = L.path.θU, θS = L.path.θS))
        cf_log(dir, @sprintf("[%s] held path: %s in %d iterations, free-entry gap %.2e, %.0f s", crisis,
                             hp.result.converged ? "converged" : "NOT converged", hp.result.n_iter,
                             hp.result.final_dist, hp.seconds))

        t = L.path.tgrid;  Nt = length(t)
        hp.path.tgrid == t || error("The held path is not on the stored transition's dates.")
        uo = [unemployment_per_head(L.path, n, pop) for n in 1:Nt]
        uh = [unemployment_per_head(hp.path, n, pop) for n in 1:Nt]
        ss = [merge((economy = name, theta_U = m.unsk_cache.θ, theta_S = m.skl_cache.θ),
                    unemployment_per_head(first(self_path(m)), 1, pop))
              for (name, m) in (("baseline", L.m0), ("crisis_open", m1), ("crisis_held", H.model))]
        u0 = ss[1];  u1 = ss[2];  uH = ss[3]

        df = DataFrame(t = t)
        for (tag, u) in (("open", uo), ("held", uh)), f in (:uU, :uS, :u, :ur_U, :ur_S)
            df[!, Symbol(f, "_", tag)] = getfield.(u, f)
        end
        df.theta_U_open = L.path.θU;  df.theta_S_open = L.path.θS
        df.theta_U_held = hp.path.θU;  df.theta_S_held = hp.path.θS
        rise_held = getfield.(uh, :u) .- u0.u
        for (j, f) in HELD_MARKETS
            Δ = getfield.(uh, f) .- getfield.(uo, f)
            df[!, Symbol("delta_", j)] = Δ
            df[!, Symbol("share_", j)] = Δ ./ rise_held
        end
        CSV.write(joinpath(dir, "held_margin_path_$(crisis).csv"), df)

        rows = NamedTuple[]
        for (j, f) in HELD_MARKETS
            Δ = df[!, Symbol("delta_", j)]
            Δinf = getfield(uH, f) - getfield(u1, f)
            push!(rows, (crisis = String(crisis), market = j, horizon_months = t[end],
                         u_baseline = getfield(u0, f),
                         u_open_horizon = getfield(uo[end], f), u_held_horizon = getfield(uh[end], f),
                         delta_horizon = Δ[end], share_horizon = Δ[end] / rise_held[end],
                         delta_window = window_mean(Δ, t, L.months),
                         share_window = window_mean(Δ, t, L.months) / window_mean(rise_held, t, L.months),
                         u_open_inf = getfield(u1, f), u_held_inf = getfield(uH, f),
                         delta_inf = Δinf, share_inf = Δinf / (uH.u - u0.u),
                         delta_horizon_over_inf = Δ[end] / Δinf))
        end
        CSV.write(joinpath(dir, "held_margin_$(crisis).csv"), DataFrame(rows))

        CSV.write(joinpath(dir, "held_margin_end_states_$(crisis).csv"),
                  DataFrame([merge((crisis = String(crisis),), e) for e in ss]))
        CSV.write(joinpath(dir, "held_margin_equilibria_$(crisis).csv"),
                  DataFrame(theta_U = [p.θU for p in H.eq.pairs], theta_S = [p.θS for p in H.eq.pairs],
                            unskilled_attracting = [p.attracting for p in H.eq.pairs],
                            theta_U_refined = H.solve.θU, theta_S_refined = H.solve.θS))
        CSV.write(joinpath(dir, "held_margin_equilibria_scan_$(crisis).csv"),
                  DataFrame(theta_U = H.eq.scan.θU, theta_S = H.eq.scan.θS,
                            log_kU_ratio = H.eq.scan.gU))

        grid = exp.(range(log(CURVE_SPAN[1] * θ1.U), log(CURVE_SPAN[2] * θ1.U); length = CURVE_N))
        curves = [(economy = name, curve = free_entry_curve(m, kU, kS, θ1, grid; objects = true))
                  for (name, m) in (("open", m1), ("held", H.model))]
        CSV.write(joinpath(dir, "held_margin_fe_curve_$(crisis).csv"),
                  vcat((DataFrame(economy = e.economy, theta_U = e.curve.θU, theta_S = e.curve.θS,
                                  log_kU_ratio = e.curve.gU, uU = e.curve.uU,
                                  mean_aU_unemployed = e.curve.mean_aU, training = e.curve.training,
                                  skilled = e.curve.skilled) for e in curves)...))

        open(joinpath(dir, "held_margin_$(crisis).jls"), "w") do io
            serialize(io, (crisis = crisis, version = ROYSEARCH_VERSION, point = POINT_ESTIMATE,
                           months = L.months, estimated = L.tr.result, held = hp.result,
                           path = df, by_market = DataFrame(rows), end_states = ss,
                           equilibrium = H.solve, scan = H.eq, curves = curves))
        end

        for r in rows
            cf_log(dir, @sprintf("[%s] %-5s Δ (held − open, per head): horizon %+.5f, ∞ %+.5f; share horizon %+.3f, ∞ %+.3f; Δ(T)/Δ(∞) %.3f",
                                 crisis, r.market, r.delta_horizon, r.delta_inf, r.share_horizon,
                                 r.share_inf, r.delta_horizon_over_inf))
        end
        cf_log(dir, @sprintf("[%s] done in %.1f min", crisis, (time() - t0) / 60))
    end
    return nothing
end

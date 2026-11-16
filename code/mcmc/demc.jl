############################################################
# mcmc/demc.jl — Differential-Evolution MCMC (DE-MC) sampler
#
# Julia port of the sampler in the Lise–Meghir–Robin (2016) replication package,
# FMPIOpt/src/mpi/mpi_mcmc_mod.f90 (StartOptimizerMaster: population loop and Metropolis
# accept; ComputeParamCandidate: the DE proposal). ter Braak (2006): N chains run
# together and each chain's proposal is built from differences between other chains, so
# its scale and orientation adapt to the target. DREAM-style extras as in LMR: a
# per-coordinate crossover CR, a periodic γ = 1 mode jump, additive and multiplicative
# shocks, and replacement of stuck chains.
#
# Everything runs in the optimiser's unconstrained (logit) space. The caller supplies the
# log quasi-posterior logπ(t), which returns −Inf at an infeasible point.
#
# Entry point: run_demc. Needs min_ess, converged_sequential and stop_rule
# (mcmc_diagnostics.jl).
############################################################

using Random, Statistics, Printf
using Base.Threads: @threads, nthreads

"""
    _de_pairs(N, c, δ, rng) -> (i1, i2)

δ difference pairs of chain indices, all distinct and different from chain `c`
(pickFirstDifferences in the Fortran). Returns two length-δ views.
"""
function _de_pairs(N::Int, c::Int, δ::Int, rng::AbstractRNG)
    need = 2δ
    chosen = Int[]
    while length(chosen) < need
        j = rand(rng, 1:N)
        (j == c || j in chosen) && continue
        push!(chosen, j)
    end
    return @view(chosen[1:δ]), @view(chosen[δ+1:2δ])
end

"""
    _proposal_draws(rng, N, c, δ, d, CR) -> (i1, i2, mask, z_mult, z_add)

The random part of chain `c`'s proposal, in the order the sampler draws it: difference
pairs, crossover mask (at least one coordinate set), multiplicative and additive normal
shocks. Reads nothing but the RNG, so a generation's consumption does not depend on the
target.
"""
function _proposal_draws(rng::AbstractRNG, N::Int, c::Int, δ::Int, d::Int, CR::Float64)
    i1, i2 = _de_pairs(N, c, δ, rng)
    mask = rand(rng, d) .< CR
    any(mask) || (mask[rand(rng, 1:d)] = true)   # always move ≥1 dim
    z_mult = randn(rng, d)
    z_add  = randn(rng, d)
    return i1, i2, mask, z_mult, z_add
end

"""
    _replay_rng!(rng, gens, N, δ, d, CR) -> rng

Advance `rng` through every draw `gens` generations of `run_demc` make (each chain's
`_proposal_draws`, then one Metropolis uniform per chain). From the run's base seed this
rebuilds, without a solve, the stream of a stored `init = :at_seed` run whose file
carries no RNG state (files written before v61.3.0).
"""
function _replay_rng!(rng::AbstractRNG, gens::Int, N::Int, δ::Int, d::Int, CR::Float64)
    for _ in 1:gens
        for c in 1:N
            _proposal_draws(rng, N, c, δ, d, CR)
        end
        for _ in 1:N
            rand(rng)
        end
    end
    return rng
end

"""
    _outcome_from_history(accepted, chain_lp, lp0) -> Matrix{Int8}

LMR outcome codes (N × G) rebuilt for a stored run that did not record them. An accepted
draw is 3, 2 or 1 by the sign of its log-target step from the previous generation (`lp0`
for generation 0); a rejection is 0, since an out-of-box rejection (−1) cannot be told
apart from the history.
"""
function _outcome_from_history(accepted::AbstractMatrix{Bool}, chain_lp::AbstractMatrix,
                               lp0::AbstractVector)
    N, G = size(accepted)
    out  = zeros(Int8, N, G)
    for g in 1:G, c in 1:N
        accepted[c, g] || continue
        dlp = chain_lp[c, g] - (g == 1 ? lp0[c] : chain_lp[c, g - 1])
        out[c, g] = dlp > 0 ? Int8(3) : dlp == 0 ? Int8(2) : Int8(1)
    end
    return out
end

# Sampler state carried between generations besides the population and history arrays.
# Read off run_demc's locals BY NAME (Base.@locals), so this one list serves snapshots,
# the return value and a resume: renaming any of these locals breaks all three.
const _DEMC_STATE = (:rng, :n_replaced, :last_replace, :lp_best, :θ_best,
                     :gamma_scale, :n_adapt, :wacc_ad, :wfin_ad, :acc_ad_last,
                     :wacc, :wfin, :wprop, :weval, :wdlp, :wesjd, :wrep, :wbox,
                     :wacc_chk, :wprop_chk, :lp_best_prev, :rhat_prev, :stop_armed, :diag_armed)

"""
    _demc_state(locals) -> NamedTuple

Deep copy of the `_DEMC_STATE` entries of `locals` (from `Base.@locals()`), so a state
handed out is never mutated by the run it came from.
"""
_demc_state(locals::AbstractDict) =
    deepcopy(NamedTuple{_DEMC_STATE}(map(k -> locals[k], _DEMC_STATE)))

"""
    _eval_population!(out, logπ, M, N, parallel; skip = nothing) -> out

Fill `out[c] = logπ(M[:, c])` for the N columns of `M`, threaded over chains when
`parallel`. Members flagged in `skip` (known to carry zero prior mass) get −Inf without a
solve, which gives the same accept/reject outcome.
"""
function _eval_population!(out::Vector{Float64}, logπ, M::AbstractMatrix,
                          N::Int, parallel::Bool;
                          skip::Union{Nothing,AbstractVector{Bool}} = nothing)
    if parallel
        @threads for c in 1:N
            out[c] = (skip !== nothing && skip[c]) ? -Inf : logπ(view(M, :, c))
        end
    else
        for c in 1:N
            out[c] = (skip !== nothing && skip[c]) ? -Inf : logπ(view(M, :, c))
        end
    end
    return out
end

"""
    run_demc(logπ, θ0; kwargs...) -> NamedTuple

Differential-Evolution MCMC on the log quasi-posterior `logπ` (a function of an
unconstrained parameter vector, thread-safe when `parallel`), seeded at `θ0`
(unconstrained).

Population and proposal:
  N             number of chains (0 ⇒ max(2d, 16))
  gens          generations (the budget cap when a stop rule is on; the TOTAL on resume)
  burn_frac     fraction of generations discarded as burn-in
  CR            per-coordinate crossover probability
  δ             difference pairs per proposal (clamped to (N−1)÷2)
  b_add, b_mult additive and multiplicative shock SDs
  jump_every    γ = 1 every `jump_every` generations (mode jump)
  gamma_scale0  multiplier on ter Braak's γ = 2.38/√(2δ·n_updated)
  t_box         half-width of the flat prior box on t (LMR ±20); Inf is an improper target
  parallel      thread the population evaluation over chains

Initial population (`init`):
  :at_seed  every chain at θ0 (LMR);
  :widths   chain 1 at θ0, others drawn with per-coordinate half-widths `init_widths`
            (up) and `init_widths_dn` (down, defaults to up), times `init_disperse`;
  :screen   `init_screen` candidates (0 ⇒ 20N) around θ0 on per-coordinate scale
            `init_scale`, keeping feasible ones and seeding from the middle of their
            ranking; `init_width` (the unclamped widths) is used only to report
            over-dispersion;
  :resume   continue `resume` = (chain, chain_lp, accepted, outcome or nothing, replaced,
            state) after its last generation; missing state keeps fresh values, the RNG
            is replayed from `rng` with `_replay_rng!`.

Stuck chains: `outlier_iqr` IQRs below Q1 of the trailing mean log-target ⇒ replaced by
the best chain over the retained window (0 disables); `outlier_burn_only` confines this
to burn-in.

Adaptive scale (`gamma_adapt`, every `gamma_every`): one-sided Robbins–Monro brake on
`gamma_scale` toward feasible-candidate acceptance `gamma_target`, step `gamma_eta/√k`.

Sequential stop (`check_every > 0`): every `check_every` generations `stop_rule` is
evaluated with `moves_min`, `drift_flat`, `acc_floor`, `mcse_target` and box bounds `lb`,
`ub`; the run stops when it passes at two consecutive checks and aborts on the
acceptance floor at two consecutive checks. `drift_max > 0` aborts when the running
maximum climbs that far above the seed. `rhat_max` and `ess_min` (≤ 0 ⇒ `min_ess(d)`)
are reported only.

Output: `verbose`, `print_every` (≤ 0 ⇒ final line only), `lmr_print` (LMR-format block
every `lmr_print` generations, 0 disables), `labels` (names for the progress line).
Callbacks: `on_best(θ_best, lp_best, g)` when the incumbent improves;
`on_dump(chain, chain_lp, accepted, outcome, replaced, g, state)` at the `lmr_print`
cadence.

Returns `(; draws, chain, chain_lp, accepted, outcome, replaced, accept, lp, N, gens,
gens_requested, burn, lp_seed, lp_best, theta_best, drift, aborted, abort_why,
gamma_scale, n_adapt, n_replaced, last_replace, state)`: `draws` is the d × (N·kept)
pooled post-burn-in sample, the history arrays are truncated to the `gens` generations
actually run, and `state` is the sampler state a resume continues from.
"""
function run_demc(logπ, θ0::AbstractVector{<:Real};
                  N::Int = 0, gens::Int = 4000, burn_frac::Float64 = 0.5,
                  CR::Float64 = 0.90, δ::Int = 1,
                  b_add::Float64 = 1e-4, b_mult::Float64 = 1e-2,
                  t_box::Float64 = Inf,
                  jump_every::Int = 10, init::Symbol = :at_seed, init_screen::Int = 0,
                  resume::Union{Nothing,NamedTuple} = nothing,
                  init_widths::Union{Nothing,AbstractVector{Float64}} = nothing,
                  init_widths_dn::Union{Nothing,AbstractVector{Float64}} = nothing,
                  init_disperse::Float64 = 2.0,
                  init_scale::Union{Nothing,AbstractVector} = nothing,
                  init_width::Union{Nothing,AbstractVector} = nothing,
                  parallel::Bool = true,
                  rng::AbstractRNG = MersenneTwister(20260624),
                  verbose::Bool = true, print_every::Int = 250,
                  lmr_print::Int = 0,
                  gamma_scale0::Float64 = 1.0,
                  labels::Union{Nothing,Vector{String}} = nothing,
                  on_dump::Union{Nothing,Function} = nothing,
                  outlier_iqr::Float64 = 2.0,
                  outlier_burn_only::Bool = false,
                  check_every::Int = 0, rhat_max::Float64 = 1.03,
                  ess_min::Float64 = 0.0, drift_max::Float64 = 0.0,
                  moves_min::Int      = 2000,
                  drift_flat::Float64 = 0.5,
                  acc_floor::Float64  = 0.02,
                  gamma_adapt::Bool     = true,
                  gamma_target::Float64 = 0.234,   # Roberts-Gelman-Gilks high-dimensional optimum
                  gamma_eta::Float64    = 2.0,
                  gamma_every::Int      = 25,
                  mcse_target::Float64 = 0.05,
                  lb::Union{Nothing,AbstractVector} = nothing,
                  ub::Union{Nothing,AbstractVector} = nothing,
                  on_best = nothing)

    print_every = print_every > 0 ? print_every : typemax(Int)

    d = length(θ0)
    N = N > 0 ? N : max(2d, 16)
    δ = clamp(δ, 1, max(1, (N - 1) ÷ 2))
    # Reference scale for all d coordinates; each proposal uses its realised mask count.
    γ_full = 2.38 / sqrt(2δ * d)
    burn = clamp(floor(Int, burn_frac * gens), 0, gens - 1)
    θ0f = collect(float.(θ0))
    init === :resume && resume === nothing &&
        error("run_demc: init = :resume needs `resume`, the stored run.")
    g0 = init === :resume ? size(resume.chain, 3) : 0     # generations already run

    # The box defines the target, so the header records it.
    verbose && @printf("[demc] d=%d  N=%d chains  gens=%d  δ=%d  CR=%.2f  γ(all d)=%.3f  box=%s  init=%s  threads=%d\n",
                       d, N, gens, δ, CR, γ_full,
                       isfinite(t_box) ? @sprintf("±%.4g", t_box) : "none (improper)",
                       init === :resume ? "resume from gen $g0" : init, nthreads())

    # A seed outside the box has log-target −Inf and nothing could ever be accepted.
    if isfinite(t_box)
        viol = findall(k -> abs(θ0f[k]) > t_box, 1:d)
        isempty(viol) || error("seed lies outside the prior box |t| ≤ $(t_box) in " *
                               "coordinate(s) $(viol) (values $(θ0f[viol])); widen " *
                               "MCMC_T_BOX or re-seed.")
    end

    # ── initial population ──────────────────────────────────────────────────
    # The DE step is the population's own spread, so the start sets the scale: a
    # population wider than the target cannot contract (contraction needs accepted
    # moves), one narrower grows into it. Every mode starts from feasible points.
    X  = Matrix{Float64}(undef, d, N)
    lp = Vector{Float64}(undef, N)
    if init === :at_seed
        # LMR (mpi_mcmc_mod.f90:268). Generation 1 has zero differences; b_add moves it.
        for c in 1:N
            @views X[:, c] .= θ0f
        end
        _eval_population!(lp, logπ, X, N, parallel)
        isfinite(lp[1]) ||
            error("run_demc: the seed itself is infeasible — rerun the estimation, or " *
                  "check that the bundle's spec matches the current one.")
    elseif init === :widths
        # Disperse at the criterion's own per-coordinate widths; chain 1 stays at the seed.
        init_widths === nothing &&
            error("run_demc: init = :widths requires init_widths (per-coordinate scale in " *
                  "the sampled space), read from MCMC_WIDTHS_CSV.")
        length(init_widths) == d ||
            error("run_demc: init_widths has $(length(init_widths)) entries, expected $(d).")
        w = collect(float.(init_widths))
        any(!isfinite, w) &&
            error("run_demc: init_widths contains non-finite entries at coordinate(s) " *
                  "$(findall(!isfinite, w)); the audit could not bracket those and they " *
                  "need a value before this mode can run.")
        wdn = init_widths_dn === nothing ? w : collect(float.(init_widths_dn))
        length(wdn) == d ||
            error("run_demc: init_widths_dn has $(length(wdn)) entries, expected $(d).")
        dev = Vector{Float64}(undef, d)
        @views X[:, 1] .= θ0f
        for c in 2:N
            # Asymmetric draw: up-width upward, down-width downward (a coordinate on a
            # floor has no downward width).
            @inbounds for k in 1:d
                z = randn(rng)
                dev[k] = init_disperse * (z >= 0.0 ? w[k] * z : wdn[k] * z)
            end
            # On infeasibility shrink toward the seed along the ray (≤ 20 halvings), so the
            # direction survives; only a wholly infeasible ray falls back to the seed.
            ok = false; s = 1.0
            for _ in 1:20
                @inbounds for k in 1:d
                    X[k, c] = θ0f[k] + s * dev[k]
                end
                if isfinite(logπ(view(X, :, c))); ok = true; break; end
                s *= 0.5
            end
            ok || (@views X[:, c] .= θ0f)
        end
        _eval_population!(lp, logπ, X, N, parallel)
        isfinite(lp[1]) ||
            error("run_demc: the seed itself is infeasible — rerun the estimation, or " *
                  "check that the bundle's spec matches the current one.")
        nfb = count(c -> all(@views X[:, c] .== θ0f), 2:N)
        verbose && nfb > 0 &&
            @printf("[demc] init=:widths — %d of %d chains fell back to the seed\n",
                    nfb, N - 1)
    elseif init === :screen
        # Draw candidates around the seed, keep the feasible ones, seed the chains from the
        # middle of their log-target ranking (the best alone would leave no spread). The
        # radius multiplier shrinks until enough candidates survive.
        ncand = init_screen > 0 ? init_screen : 20N
        cand  = Matrix{Float64}(undef, d, ncand)
        lpc   = Vector{Float64}(undef, ncand)
        keep  = Int[]
        radius = 0.0
        # Per-coordinate scale; nothing ⇒ isotropic.
        s = init_scale === nothing ? ones(d) : collect(float.(init_scale))
        length(s) == d || error("run_demc: init_scale has length $(length(s)), expected $d")
        all(x -> isfinite(x) && x > 0, s) ||
            error("run_demc: init_scale must be finite and strictly positive")
        for r in (1.0, 0.3, 0.1, 0.03, 0.01)
            for j in 1:ncand, k in 1:d
                cand[k, j] = θ0f[k] + r * s[k] * randn(rng)
            end
            @views cand[:, 1] .= θ0f          # the seed always survives the screen
            _eval_population!(lpc, logπ, cand, ncand, parallel)
            keep   = findall(isfinite, lpc)
            radius = r
            length(keep) >= max(4, N ÷ 4) && break
        end
        length(keep) >= 2 ||
            error("run_demc: init = :screen found $(length(keep))/$ncand feasible " *
                  "candidates. The seed sits in a basin too narrow to sample from; " *
                  "use init = :at_seed, or re-estimate.")
        order = keep[sortperm(lpc[keep], rev = true)]          # best log-target first
        if verbose
            # Over-dispersion is judged against each coordinate's own posterior width
            # (init_width, unclamped), not against s, which is already clamped.
            eff = radius .* s          # the scale actually drawn from, per coordinate
            nover = init_width === nothing ? -1 : count(k -> eff[k] > init_width[k], 1:d)
            @printf("[demc] screen: mult=%.2f  scale in t: min=%.2e med=%.2e max=%.2e  %s\n",
                    radius, minimum(eff), median(eff), maximum(eff),
                    init_width === nothing ? "(no width vector: over-dispersion unchecked)" :
                        @sprintf("over-dispersed: %d/%d", nover, d))
            if nover > 0
                worst = argmax([init_width[k] > 0 ? eff[k] / init_width[k] : 0.0 for k in 1:d])
                @printf("[demc] screen: WARNING %d coordinate(s) start WIDER than their own posterior; \
worst is coord %d at %.2fx. DE-MC cannot contract these, so their reported SD is biased UP.\n",
                        nover, worst, eff[worst] / init_width[worst])
            end
            @printf("[demc] screen: %d/%d feasible  logπ %.4e … %.4e  (median seeded)\n",
                    length(keep), ncand, lpc[order[1]], lpc[order[end]])
        end
        mid = max(1, length(order) ÷ 2)                        # middle of the ranking
        for c in 1:N
            src = order[mod1(mid + (c - 1), length(order))]    # cycle if survivors < N
            @views X[:, c] .= cand[:, src]
            lp[c] = lpc[src]
        end
    elseif init === :resume
        # The stored final population and its log-target; history and state follow below.
        (size(resume.chain, 1), size(resume.chain, 2)) == (d, N) ||
            error("run_demc: the stored run has $(size(resume.chain, 1)) coordinates and " *
                  "$(size(resume.chain, 2)) chains; this run has $d and $N.")
        g0 < gens || error("run_demc: gens = $gens is the total, and the stored run already " *
                           "has $g0 generations.")
        @views X  .= resume.chain[:, :, g0]
        @views lp .= resume.chain_lp[:, g0]
    else
        error("run_demc: init must be :at_seed, :widths, :screen or :resume, got :$init")
    end

    ess_target = ess_min > 0 ? ess_min : min_ess(d)
    g_final    = gens
    lp_seed    = logπ(θ0f)                    # at θ0 itself, whatever the init mode
    θ_best     = copy(θ0f)                    # argmax of logπ seen so far
    lp_best    = lp_seed
    aborted    = false
    abort_why  = :none                        # :drift (better point found) or :acceptance

    chain  = Array{Float64}(undef, d, N, gens)
    cand   = Matrix{Float64}(undef, d, N)
    lpc    = Vector{Float64}(undef, N)
    pop_sd = Vector{Float64}(undef, d)      # per-generation ESJD yardstick
    outbox = falses(N)                      # candidates outside the prior box, solve skipped
    nacc  = 0
    # Counters over the current print window. fin (feasible share of evaluated
    # candidates) separates an uphill rejection from a failed solve.
    wacc  = 0
    wfin  = 0
    wprop = 0
    weval = 0     # proposals sent to the solver: wprop minus box rejections
    # wdlp: Δlogπ of each feasible proposal (median −1 to −3 is a well-scaled step);
    # wesjd: squared jump distance in population-sd units (the mixing measure).
    wdlp  = Float64[]
    wesjd = 0.0
    # Adaptive DE scale and its own acceptance window. The brake's denominator is FEASIBLE
    # candidates (wfin_ad): an infeasible candidate's rejection does not respond to γ.
    gamma_scale = gamma_scale0
    n_adapt     = 0
    wacc_ad     = 0
    wfin_ad     = 0
    acc_ad_last = NaN                       # last conditional acceptance the brake acted on

    # Check-window counters, reset at each check; independent of the print window.
    wacc_chk  = 0
    wprop_chk = 0
    # Two-consecutive-check state; NaN/false means nothing to compare against yet.
    lp_best_prev = NaN
    rhat_prev    = Inf
    stop_armed   = false
    diag_armed   = false
    chain_lp     = Matrix{Float64}(undef, N, gens)   # per-chain log-target history
    # Per-draw records returned with the run:
    #   accepted  the draw is a new state, not a repeat;
    #   outcome   LMR's five-way code (3/2/1/0/−1, see the accept loop);
    #   replaced  overwritten by the stuck-chain rule, hence not a draw from the target.
    accepted     = falses(N, gens)
    outcome      = zeros(Int8, N, gens)
    replaced     = falses(N, gens)
    out_window   = min(250, max(1, burn ÷ 2))   # trailing window of the stuck-chain score
    # Span a replacement overwrites: the retained window (LMR overwrite their whole buffer).
    retain_len   = max(1, gens - burn)
    out_score    = Vector{Float64}(undef, N)
    n_replaced   = 0
    last_replace = 0
    wrep         = 0     # replacements since the last print
    wbox         = 0     # box rejections since the last print

    # A resumed run continues the stored one: its history fills 1..g0 and the loop picks up
    # at g0 + 1 in the stored state (RNG replayed and outcome rebuilt when the file lacks them).
    if init === :resume
        @views chain[:, :, 1:g0] .= resume.chain
        @views chain_lp[:, 1:g0] .= resume.chain_lp
        @views accepted[:, 1:g0] .= resume.accepted
        @views replaced[:, 1:g0] .= resume.replaced
        @views outcome[:, 1:g0]  .= resume.outcome === nothing ?
            _outcome_from_history(resume.accepted, resume.chain_lp, fill(lp_seed, N)) :
            resume.outcome
        nacc = count(resume.accepted)
        haskey(resume.state, :gamma_scale) ||
            !((gamma_adapt && gamma_every > 0) || check_every > 0) ||
            error("run_demc: the stored run records neither an adapted DE scale nor the " *
                  "stopping rule's state (a file written before v61.3.0), so a run with either " *
                  "on cannot continue it as one run.")
        haskey(resume.state, :rng) || _replay_rng!(rng, g0, N, δ, d, CR)
        (; rng, n_replaced, last_replace, lp_best, θ_best, gamma_scale, n_adapt, wacc_ad,
           wfin_ad, acc_ad_last, wacc, wfin, wprop, weval, wdlp, wesjd, wrep, wbox, wacc_chk,
           wprop_chk, lp_best_prev, rhat_prev, stop_armed, diag_armed) =
            deepcopy(merge(_demc_state(Base.@locals()), resume.state))
    end

    for g in (g0 + 1):gens
        # Brake the DE scale toward the acceptance optimum: Robbins–Monro in log γ, step
        # η/√n_adapt. One-sided: it only shortens the step, because the population widens
        # on its own and a symmetric rule amplified the early expansion.
        if gamma_adapt && gamma_every > 0 && g > 1 && (g - 1) % gamma_every == 0 && wfin_ad > 0
            acc_ad = wacc_ad / wfin_ad
            acc_ad_last = acc_ad
            if acc_ad < gamma_target
                n_adapt += 1
                gamma_scale = max(gamma_scale *
                                  exp((gamma_eta / sqrt(n_adapt)) * (acc_ad - gamma_target)),
                                  1e-3)
            end
            wacc_ad = 0; wfin_ad = 0
        end

        Xc   = copy(X)                        # freeze current population
        jump = (g % jump_every == 0)          # periodic mode jump

        for c in 1:N
            i1, i2, mask, z_mult, z_add = _proposal_draws(rng, N, c, δ, d, CR)
            diff = zeros(d)
            @inbounds for j in 1:δ
                @views diff .+= Xc[:, i1[j]] .- Xc[:, i2[j]]
            end
            # gamma_scale multiplies the DE step only; the mode jump keeps γ = 1.
            γ = jump ? 1.0 : gamma_scale * 2.38 / sqrt(2δ * count(mask))
            e = b_mult .* z_mult
            ε = b_add  .* z_add
            @views prop = Xc[:, c] .+ (1.0 .+ e) .* γ .* diff .+ ε
            @inbounds for k in 1:d
                cand[k, c] = mask[k] ? prop[k] : Xc[k, c]
            end
            # Outside the prior box: rejected without a solve. Rejected, not redrawn as in
            # LMR (mpi_mcmc_mod.f90:294-304): redrawing makes the proposal asymmetric and
            # breaks detailed balance.
            outbox[c] = isfinite(t_box) && any(abs(cand[k, c]) > t_box for k in 1:d)
        end

        _eval_population!(lpc, logπ, cand, N, parallel; skip = outbox)
        wbox += count(outbox)

        # Population sd per coordinate, fixed for this generation's ESJD, floored at b_add
        # (the smallest displacement a proposal makes) so a collapsed start does not divide
        # a real step by rounding noise.
        sd_floor = max(b_add, eps())
        @inbounds for k in 1:d
            pop_sd[k] = max(std(@view Xc[k, :]), sd_floor)
        end

        for c in 1:N
            wprop += 1; wprop_chk += 1
            # fin and dlp describe candidates the solver saw; acceptance keeps the full
            # denominator, since an out-of-box proposal is a real rejection.
            if !outbox[c]
                weval += 1
                if isfinite(lpc[c])
                    wfin += 1; wfin_ad += 1
                    push!(wdlp, lpc[c] - lp[c])
                end
            end
            # LMR outcome code (mpi_mcmc_mod.f90:360-378), by the sign of Δ log-target:
            #   3 strong accept (Δ > 0)   2 equal accept (Δ = 0)   1 weak accept (Δ < 0)
            #   0 reject                 −1 out of the box, never evaluated
            dlp = lpc[c] - lp[c]
            if log(rand(rng)) < dlp                      # α = exp(Δ log-posterior)
                jd = 0.0
                @inbounds for k in 1:d
                    jd += ((cand[k, c] - X[k, c]) / pop_sd[k])^2
                end
                wesjd += jd
                @views X[:, c] .= cand[:, c]
                lp[c] = lpc[c]
                nacc += 1; wacc += 1; wacc_chk += 1; wacc_ad += 1
                accepted[c, g] = true
                outcome[c, g] = dlp > 0 ? Int8(3) : dlp == 0 ? Int8(2) : Int8(1)
            else
                outcome[c, g] = outbox[c] ? Int8(-1) : Int8(0)
            end
            chain_lp[c, g] = lp[c]
        end
        @views chain[:, :, g] .= X
        c_best = argmax(lp)
        if lp[c_best] > lp_best
            lp_best = lp[c_best]
            @views θ_best .= X[:, c_best]
            on_best === nothing || on_best(θ_best, lp_best, g)
        end

        # Stuck-chain replacement (LMR mpi_mcmc_mod.f90:419-427). A chain whose mean
        # log-target over the trailing window is more than outlier_iqr IQRs below Q1 takes
        # the best chain's position, and its history over the retained window is
        # overwritten with the best chain's (LMR's scope), marked in `replaced`. Not a
        # reversible transition: outlier_burn_only confines it to burn-in.
        if outlier_iqr > 0 && g > out_window && (!outlier_burn_only || g <= burn)
            for c in 1:N
                @views out_score[c] = mean(chain_lp[c, (g - out_window + 1):g])
            end
            q1, q3 = quantile(out_score, 0.25), quantile(out_score, 0.75)
            cut    = q1 - outlier_iqr * (q3 - q1)
            for c in 1:N
                if out_score[c] < cut
                    @views X[:, c] .= X[:, c_best]
                    lp[c] = lp[c_best]
                    lo = max(1, g - retain_len + 1)
                    @views chain[:, c, lo:g]  .= chain[:, c_best, lo:g]
                    @views chain_lp[c, lo:g]  .= chain_lp[c_best, lo:g]
                    @views replaced[c, lo:g]  .= true
                    n_replaced += 1
                    wrep += 1
                    last_replace = g
                end
            end
        end

        if verbose && (g % print_every == 0 || g == gens)
            # Progress line, all over the print window:
            #   acc    acceptance rate;  moved  chains that moved at least once;
            #   sprd   mean per-coordinate population sd (and the widest coordinate);
            #   dlp    median feasible Δlogπ (step size);  esjd  jump per proposal;
            #   R̂/ESS  over non-exempt coordinates (reported, not gated);
            #   fin    feasible share, shown below 0.95;  rep  replacements, "RETAINED"
            #          past burn-in;  box  box-rejection share, shown when nonzero.
            _, wr, me = converged_sequential(chain, g, burn_frac, d;
                                             rhat_max = rhat_max, ess_min = ess_target,
                                             lb = lb, ub = ub)
            fin_frac = wfin / max(weval, 1)
            plo   = max(1, g - print_every + 1)
            nmove = count(any(view(accepted, i, plo:g)) for i in 1:N)
            psd   = [std(view(chain, k, :, g)) for k in 1:d]
            kw    = argmax(psd)
            @printf("[demc] gen %5d/%d  acc=%.3f  moved=%d/%d  sprd=%.2e \
                     (widest %s %.2e)  dlp=%s  esjd=%.3f  max logπ=%.6e  \
                     R̂=%s  ESS=%s%s%s%s\n",
                    g, gens, wacc / max(wprop, 1), nmove, N, mean(psd),
                    labels === nothing ? string(kw) : last(split(labels[kw])), psd[kw],
                    isempty(wdlp) ? "n/a" : @sprintf("%+.2f", median(wdlp)),
                    wesjd / max(wprop, 1), maximum(lp),
                    isfinite(wr) ? @sprintf("%.3f", wr) : "n/a",
                    isfinite(me) ? @sprintf("%.0f", me) : "n/a",
                    fin_frac < 0.95 ? @sprintf("  fin=%.2f", fin_frac) : "",
                    wrep == 0 ? "" :
                        g > burn ? @sprintf("  rep=%d RETAINED", wrep) :
                                   @sprintf("  rep=%d", wrep),
                    wbox == 0 ? "" :
                        @sprintf("  box=%.2f", wbox / max(wprop, 1))); flush(stdout)
            wacc = 0; wfin = 0; wprop = 0; weval = 0; wrep = 0; wbox = 0
            empty!(wdlp); wesjd = 0.0
        end

        # LMR's periodic block (mpi_mcmc_mod.f90:429-455), shares over their L-generation
        # buffer. Their running-mean and best vectors are omitted (they are in the CSV).
        if lmr_print > 0 && (g % lmr_print == 0 || g == gens)
            lo = max(1, g - retain_len + 1)          # their cyclical buffer, L generations
            n_buf = N * (g - lo + 1)
            share(v) = count(==(Int8(v)), @view outcome[:, lo:g]) / n_buf
            @printf("\n ---------- gen %d, last %d generations ----------\n",
                    g, g - lo + 1)
            # "equal" cannot occur with a continuous criterion; box rejections are on the
            # progress line.
            @printf(" accept  strong %.4f   weak %.4f\n", share(3), share(1))
            @printf(" reject  weak   %.4f\n", share(0))
            @printf(" logπ over the population: min %.6e  median %.6e  max %.6e\n",
                    minimum(lp), median(lp), maximum(lp))
            # Distinct replaced chains, the unit n_replaced uses.
            @printf(" outlier chains replaced: %d in this window, %d in the run\n",
                    count(any(view(replaced, c, lo:g)) for c in 1:N), n_replaced)
            flush(stdout)
        end

        if check_every > 0 && g >= check_every && g % check_every == 0 && g < gens
            drift = lp_best - lp_seed
            if drift_max > 0 && drift > drift_max
                if verbose
                    @printf("[demc] ABORT g=%d  max logπ has climbed %.1f above the seed (>%.1f).\n",
                            g, drift, drift_max)
                    println("       The chain is optimising, not sampling: Cov(chain) would measure")
                    println("       its trajectory, not the curvature. `drift_components` at θ_best")
                    println("       separates a genuinely better Q from a seed parked near a rail;")
                    println("       the local-design Ĵ = Ĝ'WĜ route needs no chain either way.")
                    flush(stdout)
                end
                g_final = g; aborted = true; abort_why = :drift
                break
            end
            # Stop and diagnosis each need two consecutive checks.
            drift_since_last = isnan(lp_best_prev) ? NaN : lp_best - lp_best_prev
            acc_window = wprop_chk > 0 ? wacc_chk / wprop_chk : NaN
            done1, diag1, moves, _, wr, me, mcse_now =
                stop_rule(chain, g, burn_frac, d;
                          moves_min = moves_min, drift_flat = drift_flat,
                          acc_floor = acc_floor, mcse_target = mcse_target,
                          acc_window = acc_window,
                          drift_since_last = drift_since_last,
                          rhat_max = rhat_max, ess_min = ess_target, lb = lb, ub = ub)
            done = done1 && stop_armed
            diagnose = diag1 && diag_armed
            stop_armed = done1; diag_armed = diag1
            lp_best_prev = lp_best; rhat_prev = isfinite(wr) ? wr : rhat_prev
            wacc_chk = 0; wprop_chk = 0

            if verbose
                # The gated quantities beside their thresholds; mcse's ceiling 1/√N and R̂
                # are shown because mcse saturates there. The cumulative drift is defined
                # even on the first check.
                @printf("[demc] check g=%d  mcse=%s/%.3f (ceil %.3f)  R̂=%s  γ×%.3f accf=%s  moves=%d/%d  Δmax logπ=%s/%.2f  (cum %+.3f)  → %s\n",
                        g,
                        isfinite(mcse_now) ? @sprintf("%.4f", mcse_now) : "n/a",
                        mcse_target, 1/sqrt(N),
                        isfinite(wr) ? @sprintf("%.2f", wr) : "n/a",
                        gamma_scale,
                        isfinite(acc_ad_last) ? @sprintf("%.3f", acc_ad_last) : "n/a",
                        moves, moves_min,
                        isfinite(drift_since_last) ? @sprintf("%+.3f", drift_since_last) :
                            "n/a (first check)",
                        drift_flat, lp_best - lp_seed,
                        done ? "STOPPING" : (diagnose ? "ACCEPTANCE FLOOR" : "continuing"))
                flush(stdout)
            end
            if diagnose
                if verbose
                    @printf("[demc] ABORT g=%d  windowed acceptance %.4f < %.3f at two \
                             consecutive checks.\n", g, acc_window, acc_floor)
                    println("       The population is mis-scaled relative to the target: at this")
                    println("       acceptance the chain is not moving, and more generations cannot")
                    println("       fix it. RGG's high-dimensional optimum is 0.234. Check the start")
                    println("       (MCMC_INIT = :screen reports the radius it settled on) before")
                    println("       spending the rest of the budget.")
                    flush(stdout)
                end
                g_final = g; aborted = true; abort_why = :acceptance
                break
            end
            if done
                g_final = g
                break
            end
        end

        # Snapshot at the end of the generation, at the lmr_print cadence, so a run resumes
        # from it exactly. Skipped at g == gens: the caller snapshots when this returns.
        if on_dump !== nothing && lmr_print > 0 && g % lmr_print == 0 && g < gens
            on_dump(view(chain, :, :, 1:g), view(chain_lp, :, 1:g), view(accepted, :, 1:g),
                    view(outcome, :, 1:g), view(replaced, :, 1:g), g,
                    _demc_state(Base.@locals()))
        end
    end

    # Truncate to the generations run; burn is re-derived at g_final so an early stop
    # discards the same fraction as a full run.
    chain    = chain[:, :, 1:g_final]
    chain_lp = chain_lp[:, 1:g_final]
    accepted = accepted[:, 1:g_final]
    outcome  = outcome[:, 1:g_final]
    replaced = replaced[:, 1:g_final]
    burn  = clamp(floor(Int, burn_frac * g_final), 0, g_final - 1)

    draws = reshape(chain[:, :, burn+1:end], d, :)        # pool post-burn-in chains
    return (; draws, chain, chain_lp, accepted, outcome, replaced,
              accept = nacc / (g_final * N), lp, N,
              gens = g_final, gens_requested = gens, burn,
              lp_seed, lp_best, theta_best = θ_best,
              drift = lp_best - lp_seed, aborted, abort_why,
              gamma_scale, n_adapt,
              n_replaced, last_replace,
              state = _demc_state(Base.@locals()))
end

############################################################
# smm/demc.jl — Differential-Evolution MCMC (DE-MC) sampler
#
# Julia port of the algorithm in the Lise–Meghir–Robin (2016)
# replication package:
#   FMPIOpt/src/mpi/mpi_mcmc_mod.f90
#     · StartOptimizerMaster   (population loop, Metropolis accept)
#     · ComputeParamCandidate  (the differential-evolution proposal)
#
# Idea (ter Braak 2006, "A Markov Chain Monte Carlo version of the
# genetic algorithm Differential Evolution"): run N chains in parallel
# (a "population"). The proposal for chain c is built from DIFFERENCES
# between other randomly chosen chains, so the proposal scale and
# orientation self-adapt to the target covariance — no hand-tuned
# proposal matrix. DREAM-style extras carried over from LMR:
#   · per-dimension crossover CR (only a random subset of dims moves),
#   · a periodic γ = 1 "mode jump" (every `jump_every` generations),
#   · small additive (b_add) and multiplicative (b_mult) shocks.
#
# Everything runs in the optimiser's UNCONSTRAINED (logit) space, so any
# draw maps back inside the (lb,ub) box via `_to_constrained`. The caller
# supplies the log quasi-posterior `logπ(θ)` (see MCMC_main.jl). Infeasible
# θ must return logπ = -Inf (handled there) → the proposal is rejected,
# which is exactly LMR's "reject if outside prior".
#
# NOTE on threading: the equilibrium solve is already internally threaded
# over the worker-type grid, and the DE optimiser threads over its
# population (smm.jl). Mirroring that, the population evaluation here can
# thread over chains (`parallel = true`); under the outer @threads the
# solver's inner @threads run serially, so there is no oversubscription.
# Set `parallel = false` to evaluate chains serially (each solve then uses
# the solver's own threads).
############################################################

using Random, Statistics, Printf
using Base.Threads: @threads, nthreads

# Pick δ distinct difference pairs of chain indices, all ≠ the current
# chain `c` and ≠ each other (mirrors pickFirstDifferences in the Fortran).
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

# Evaluate logπ over a population matrix M (d × N), filling `out`.
function _eval_population!(out::Vector{Float64}, logπ, M::AbstractMatrix,
                          N::Int, parallel::Bool)
    if parallel
        @threads for c in 1:N
            out[c] = logπ(view(M, :, c))
        end
    else
        for c in 1:N
            out[c] = logπ(view(M, :, c))
        end
    end
    return out
end

"""
    run_demc(logπ, θ0; kwargs...) → NamedTuple

Differential-Evolution MCMC. `logπ` is the log quasi-posterior (a function
of an unconstrained-space parameter vector); `θ0` is the seed (e.g. the SMM
point estimate, unconstrained).

Keyword arguments (defaults follow LMR where sensible):
  N           number of chains (0 ⇒ max(2·d, 16))
  gens        number of generations
  burn_frac   fraction of generations discarded as burn-in
  CR          per-dimension crossover probability (LMR: 0.75; default 0.90)
  δ           number of difference pairs in the proposal (LMR: 2; default 1)
  b_add       additive shock sd          (LMR shock_add_std = 1e-4)
  b_mult      multiplicative shock sd    (LMR shock_mult_std = 1e-2)
  jump_every  γ is set to 1.0 every `jump_every` generations (mode jumps)
  outlier_iqr IQR multiple below Q1 at which a chain is declared stuck and replaced by
              a copy of the best chain (Lise-Meghir-Robin, mpi_mcmc_mod.f90:419).
              A DE-MC proposal is γ·(Xᵣ₁ − Xᵣ₂), so a stuck chain does not merely waste
              its own draws — its position enters every other chain's difference vector
              and degrades all N. Replacement breaks detailed balance, so it runs ONLY
              during burn-in and the last generation it fired is reported: a replacement
              inside the retained sample would mean those draws are not from the target.
              0 disables. On 32 iid scores this flags 0.08 chains per generation, so on
              a healthy population it is close to silent.
  init        `:at_seed` starts every chain at θ0 exactly; `:screen` draws candidates
              around θ0, keeps only the ones the objective accepts and seeds from the
              middle of their log-target ranking
  init_screen candidate count for `init = :screen` (0 → 20N)
  parallel    thread the population evaluation over chains (see header note)
  rng, verbose, print_every, check_every, rhat_max, ess_min

Set `check_every > 0` to terminate sequentially instead of always running the
full `gens`: every `check_every` generations, R̂ and per-parameter ESS are computed
on the draws so far and sampling stops once R̂ ≤ `rhat_max` for every coordinate
and ESS ≥ `ess_min` for every coordinate (`ess_min ≤ 0` uses the dimension's
Vats–Flegal–Jones minESS floor). `gens` then acts as a budget cap.

Returns `(; draws, chain, accept, lp, N, gens, gens_requested, burn)` where
  draws          : d × (N·kept) matrix of post-burn-in samples (UNCONSTRAINED),
  chain          : d × N × gens history, truncated to the generations actually run,
  gens           : generations actually run (< gens_requested if it stopped early),
  gens_requested : the budget cap that was passed in.
"""
function run_demc(logπ, θ0::AbstractVector{<:Real};
                  N::Int = 0, gens::Int = 4000, burn_frac::Float64 = 0.5,
                  CR::Float64 = 0.90, δ::Int = 1,
                  b_add::Float64 = 1e-4, b_mult::Float64 = 1e-2,
                  jump_every::Int = 10, init::Symbol = :at_seed, init_screen::Int = 0,
                  parallel::Bool = true,
                  rng::AbstractRNG = MersenneTwister(20260624),
                  verbose::Bool = true, print_every::Int = 250,
                  outlier_iqr::Float64 = 2.0,
                  check_every::Int = 0, rhat_max::Float64 = 1.03,
                  ess_min::Float64 = 0.0, drift_max::Float64 = 0.0,
                  on_best = nothing)

    # print_every ≤ 0 means "final line only" rather than a modulo by zero.
    print_every = print_every > 0 ? print_every : typemax(Int)

    d = length(θ0)
    N = N > 0 ? N : max(2d, 16)
    δ = clamp(δ, 1, max(1, (N - 1) ÷ 2))
    # ter Braak's scaling is 2.38/sqrt(2·δ·n) where n is the number of coordinates
    # ACTUALLY updated, not the dimension: the CR mask freezes the rest, so the same
    # step is spread over fewer coordinates and each must move further. LMR compute it
    # per proposal from the realised mask (mpi_mcmc_mod.f90:576). γ_full below is the
    # d-coordinate value, reported in the header as the reference scale.
    γ_full = 2.38 / sqrt(2δ * d)
    burn = clamp(floor(Int, burn_frac * gens), 0, gens - 1)
    θ0f = collect(float.(θ0))

    verbose && @printf("[demc] d=%d  N=%d chains  gens=%d  δ=%d  CR=%.2f  γ(all d)=%.3f  init=%s  threads=%d\n",
                       d, N, gens, δ, CR, γ_full, init, nthreads())

    # ── initial population ──────────────────────────────────────────────────
    # DE-MC's step size IS the population spread: the proposal is γ·(Xᵣ₁ − Xᵣ₂), so
    # the difference vector inherits whatever dispersion the population has, PER
    # COORDINATE. That makes the start decisive and asymmetric — a population wider
    # than the target cannot contract, because contraction requires accepted moves and
    # an over-wide proposal is rejected; a population narrower than the target grows
    # into it within a few hundred generations. Measured on a 25-d target of sd 1e-3:
    # starting from an isotropic 1e-2 cloud leaves the spread at 9.0e-3 with acceptance
    # 0.000 after 600 generations, while starting at the seed reaches 9.5e-4 with
    # acceptance 0.30.
    #
    # So neither mode disperses blindly. Both start from points the objective already
    # accepts and let the difference vectors discover each coordinate's own scale.
    # How fast they do so depends on how far each coordinate has to travel: on a 25-d
    # target whose widths span 300x, the four narrower groups reach 0.92-0.99 of their
    # true scale within 600 generations, but the WIDEST group is still at 0.60 there,
    # reaching 0.94 by 1500 generations and 0.99 by 6000. Growth is the direction DE-MC
    # self-corrects in, so this is burn-in rather than a bias — but it means burn_frac
    # must cover the widest coordinate's growth, not the median's.
    X  = Matrix{Float64}(undef, d, N)
    lp = Vector{Float64}(undef, N)
    if init === :at_seed
        # Every chain starts at exactly the seed, as Lise-Meghir-Robin do
        # (mpi_mcmc_mod.f90:268): generation 1 then has zero difference vectors and the
        # additive shock
        # b_add is the only mover. The population then grows outward to the target's
        # own scale. Their uniform-over-prior alternative sits commented out at lines
        # 265-266 of the same file.
        for c in 1:N
            @views X[:, c] .= θ0f
        end
        _eval_population!(lp, logπ, X, N, parallel)
        isfinite(lp[1]) ||
            error("run_demc: the seed itself is infeasible — rerun the estimation, or " *
                  "check that the bundle's spec matches the current one.")
    elseif init === :screen
        # Screened start: draw candidates around the seed, KEEP ONLY the ones the
        # solver converges on, rank the survivors by log-target and seed the chains
        # from the middle of that ranking. Drawing around an optimum mostly returns
        # non-convergence, so the screen is what makes this usable: it selects a
        # feasible cloud rather than assuming one exists at a chosen radius. Taking
        # the middle rather than the best avoids putting every chain at the mode,
        # which would leave no spread to build a difference vector from.
        # The radius cannot be a fixed constant: drawing at a chosen radius around an
        # optimum mostly returns non-convergence, and how wide is admissible is a
        # property of the basin, not of the sampler. So shrink from a generous radius
        # until enough candidates survive, and report which radius that was — a very
        # small one is itself the finding that the basin is narrow.
        ncand = init_screen > 0 ? init_screen : 20N
        cand  = Matrix{Float64}(undef, d, ncand)
        lpc   = Vector{Float64}(undef, ncand)
        keep  = Int[]
        radius = 0.0
        for r in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
            for j in 1:ncand, k in 1:d
                cand[k, j] = θ0f[k] + r * randn(rng)
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
        verbose && @printf("[demc] screen: radius=%.1e  %d/%d feasible  logπ %.4e … %.4e  (median seeded)\n",
                           radius, length(keep), ncand, lpc[order[1]], lpc[order[end]])
        mid = max(1, length(order) ÷ 2)                        # middle of the ranking
        for c in 1:N
            src = order[mod1(mid + (c - 1), length(order))]    # cycle if survivors < N
            @views X[:, c] .= cand[:, src]
            lp[c] = lpc[src]
        end
    else
        error("run_demc: init must be :at_seed or :screen, got :$init")
    end

    # Sequential termination (Vats–Flegal–Jones 2019; Vats–Knudson 2021): from
    # check_every onwards, stop as soon as R̂ and ESS both clear their thresholds
    # on the draws so far. check_every = 0 runs the full `gens` unconditionally.
    # ess_min ≤ 0 resolves to the dimension's minESS floor.
    ess_target = ess_min > 0 ? ess_min : min_ess(d)
    g_final    = gens
    # A chain whose running maximum keeps climbing above the seed is optimising, not
    # sampling, and no amount of extra budget fixes it: abort so the diagnosis is
    # cheap. drift_max ≤ 0 disables the check.
    lp_seed    = lp[1]                        # chain 1 starts exactly at θ0
    θ_best     = copy(θ0f)                    # argmax of logπ seen so far
    lp_best    = lp_seed
    aborted    = false

    chain = Array{Float64}(undef, d, N, gens)
    cand  = Matrix{Float64}(undef, d, N)
    lpc   = Vector{Float64}(undef, N)
    nacc  = 0
    # Windowed counters, reset at each print. A cumulative acceptance rate hides
    # the current one once the early generations are averaged in, and `nfin`
    # separates a proposal rejected for being uphill from one rejected because
    # the solve failed — the two call for opposite fixes.
    wacc  = 0
    wfin  = 0
    wprop = 0
    # Per-chain log-posterior history, for the stuck-chain score. LMR average over the
    # last half of a 500-draw ring buffer; 250 is that window, capped so a short run
    # still scores over something.
    chain_lp     = Matrix{Float64}(undef, N, gens)
    out_window   = min(250, max(1, burn ÷ 2))
    out_score    = Vector{Float64}(undef, N)
    n_replaced   = 0
    last_replace = 0

    for g in 1:gens
        Xc   = copy(X)                        # freeze current population
        jump = (g % jump_every == 0)          # periodic mode jump

        for c in 1:N
            i1, i2 = _de_pairs(N, c, δ, rng)
            diff = zeros(d)
            @inbounds for j in 1:δ
                @views diff .+= Xc[:, i1[j]] .- Xc[:, i2[j]]
            end
            # Draw the mask BEFORE the proposal: γ depends on how many coordinates it
            # updates, so the realised count has to be known first.
            mask = rand(rng, d) .< CR
            any(mask) || (mask[rand(rng, 1:d)] = true)   # always move ≥1 dim
            γ = jump ? 1.0 : 2.38 / sqrt(2δ * count(mask))
            e = b_mult .* randn(rng, d)
            ε = b_add  .* randn(rng, d)
            @views prop = Xc[:, c] .+ (1.0 .+ e) .* γ .* diff .+ ε
            @inbounds for k in 1:d
                cand[k, c] = mask[k] ? prop[k] : Xc[k, c]
            end
        end

        _eval_population!(lpc, logπ, cand, N, parallel)

        for c in 1:N
            wprop += 1
            isfinite(lpc[c]) && (wfin += 1)
            if log(rand(rng)) < lpc[c] - lp[c]           # α = exp(Δ log-posterior)
                @views X[:, c] .= cand[:, c]
                lp[c] = lpc[c]
                nacc += 1; wacc += 1
            end
            chain_lp[c, g] = lp[c]
        end
        @views chain[:, :, g] .= X
        c_best = argmax(lp)
        if lp[c_best] > lp_best
            lp_best = lp[c_best]
            @views θ_best .= X[:, c_best]
            # Hand the incumbent out as it improves, so a run that never reaches its
            # own end — interrupted, or killed after the abort raises the budget — still
            # leaves the best point on disk rather than only in this frame.
            on_best === nothing || on_best(θ_best, lp_best, g)
        end

        # Stuck-chain replacement (Lise-Meghir-Robin, mpi_mcmc_mod.f90:419). Score each
        # chain by its MEAN log-posterior over the trailing window rather than its
        # current value, so a chain is judged on where it has been living and not on one
        # lucky proposal; replace any chain more than outlier_iqr IQRs below Q1 with the
        # best chain's current position.
        #
        # Confined to burn-in: the replacement is not a reversible transition, so a
        # generation in which it fires cannot contribute to the retained sample. LMR run
        # it throughout, which their 10 000 generations make harmless because it is
        # extinct long before the end; at this budget the gate has to be explicit.
        if outlier_iqr > 0 && g <= burn && g > out_window
            for c in 1:N
                @views out_score[c] = mean(chain_lp[c, (g - out_window + 1):g])
            end
            q1, q3 = quantile(out_score, 0.25), quantile(out_score, 0.75)
            cut    = q1 - outlier_iqr * (q3 - q1)
            for c in 1:N
                if out_score[c] < cut
                    @views X[:, c] .= X[:, c_best]
                    lp[c] = lp[c_best]
                    @views chain[:, c, g] .= X[:, c_best]
                    chain_lp[c, g] = lp[c_best]
                    n_replaced += 1
                    last_replace = g
                end
            end
        end

        if verbose && (g % print_every == 0 || g == gens)
            @printf("[demc] gen %5d/%d  acc=%.3f  fin=%.2f  max logπ=%.6e  (cum acc=%.3f)\n",
                    g, gens, wacc / max(wprop, 1), wfin / max(wprop, 1),
                    maximum(lp), nacc / (g * N)); flush(stdout)
            wacc = 0; wfin = 0; wprop = 0
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
                g_final = g; aborted = true
                break
            end
            done, wr, me = converged_sequential(chain, g, burn_frac, d;
                                                rhat_max = rhat_max,
                                                ess_min  = ess_target)
            if verbose
                @printf("[demc] check g=%d  worst R̂=%.4f (≤%.3f)  min ESS=%.0f (≥%.0f)  %s\n",
                        g, wr, rhat_max, me, ess_target,
                        done ? "→ converged, stopping" : "continuing"); flush(stdout)
            end
            if done
                g_final = g
                break
            end
        end
    end

    # Truncate to what was actually filled, and re-derive burn-in at g_final so a
    # sequential stop discards the same FRACTION as a full-length run.
    chain = chain[:, :, 1:g_final]
    burn  = clamp(floor(Int, burn_frac * g_final), 0, g_final - 1)

    draws = reshape(chain[:, :, burn+1:end], d, :)        # pool post-burn-in chains
    return (; draws, chain, accept = nacc / (g_final * N), lp, N,
              gens = g_final, gens_requested = gens, burn,
              lp_seed, lp_best, theta_best = θ_best,
              drift = lp_best - lp_seed, aborted,
              n_replaced, last_replace)
end

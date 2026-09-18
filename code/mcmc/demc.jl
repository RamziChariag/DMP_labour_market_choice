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
                          N::Int, parallel::Bool;
                          skip::Union{Nothing,AbstractVector{Bool}} = nothing)
    # `skip` marks members already known to carry zero prior mass, so the solve is not
    # worth spending: logπ would return -Inf and the Metropolis test would reject. Writing
    # -Inf directly gives the identical accept/reject outcome for free. Defaults to nothing,
    # which evaluates every member as before.
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
              Timing is set by `outlier_burn_only`; the LMR default is throughout.
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
                  # Half-width of the prior box on the UNCONSTRAINED parameter (LMR's
                  # prior_lower/upper_bound = ∓20, main_mpi.f90:146-147). Inf recovers the
                  # unbounded — and improper — target that every version before 20.0.0 used.
                  t_box::Float64 = Inf,
                  jump_every::Int = 10, init::Symbol = :at_seed, init_screen::Int = 0,
                  # Per-coordinate scale for init = :widths, in the SAMPLED space. Supplying
                  # the criterion's own ΔQ=1 half-widths starts the population at roughly the
                  # scale it would otherwise spend the run growing into: measured on base_fc,
                  # every coordinate began at across-chain sd ~1e-4 against criterion widths
                  # of 1e-4 to 1e-2, so the cloud started one to two orders of magnitude too
                  # small and no coordinate equilibrated within 400 generations.
                  init_widths::Union{Nothing,AbstractVector{Float64}} = nothing,
                  # Down-side half-widths, kept separate because up and down are not
                  # interchangeable: a coordinate resting on a floor has a bracketable
                  # width upward and none downward. Defaults to init_widths (symmetric).
                  init_widths_dn::Union{Nothing,AbstractVector{Float64}} = nothing,
                  # Multiplier on those widths. >1 is REQUIRED for the sequential stop to
                  # have power: at 1.0 the population starts AT the target scale, so
                  # between- and within-chain variance start equal and R-hat starts at 1 —
                  # the same blindness as :at_seed, less severe. Gelman-Rubin needs an
                  # over-dispersed start, and mcse_ratio needs it for the same reason.
                  init_disperse::Float64 = 2.0,
                  init_scale::Union{Nothing,AbstractVector} = nothing,
                  init_width::Union{Nothing,AbstractVector} = nothing,
                  parallel::Bool = true,
                  rng::AbstractRNG = MersenneTwister(20260624),
                  verbose::Bool = true, print_every::Int = 250,
                  outlier_iqr::Float64 = 2.0,
                  # false = LMR: mpi_mcmc_mod.f90:421-427 runs every generation, ungated.
                  # true  = confine to burn-in, keeping the retained sample reversible.
                  outlier_burn_only::Bool = false,
                  check_every::Int = 0, rhat_max::Float64 = 1.03,
                  ess_min::Float64 = 0.0, drift_max::Float64 = 0.0,
                  # Sequential stop (stop_rule, mcmc_diagnostics.jl). rhat_max and ess_min
                  # above are now REPORTED rather than gated on: the stop gates on accepted
                  # moves, drift-flatness and non-worsening R̂. Defaults reproduce the
                  # documented rule; a caller passing moves_min = typemax(Int) disables the
                  # stop without disabling the acceptance-floor diagnosis.
                  moves_min::Int      = 2000,
                  drift_flat::Float64 = 0.5,
                  acc_floor::Float64  = 0.02,
                  # ADAPTIVE DE SCALE. gamma is fixed at 2.38/sqrt(2*delta*n) in ter Braak's DE-MC, which
                  # assumes the population already sits at the target's scale. Measured on the v20.0.0
                  # base_fc run it does not: the population drifted to a median 6.1x the criterion's own
                  # per-coordinate widths, and the error is a SHAPE error rather than a scale one —
                  # it spans ~500x, from 25x too wide on λ_S to 0.05x on b_S. Acceptance fell
                  # 0.193 -> 0.009 between g=250 and g=1000, and the run aborted. No scalar can
                  # correct a shape error, and it does not have to: the population's SHAPE can only
                  # change through accepted moves,
                  # a chain at 0.9% acceptance cannot reshape itself out of the problem — the adaptation
                  # below is what restores the accepted moves that let DE-MC's own covariance adaptation
                  # work. Diminishing step eta_k = gamma_eta/sqrt(k) (Roberts & Rosenthal 2007): the
                  # adaptation vanishes asymptotically, so ergodicity holds with NO freeze point — which
                  # matters here because burn = burn_frac*gens never arrives when the sequential stop
                  # fires two orders of magnitude earlier.
                  gamma_adapt::Bool     = true,
                  gamma_target::Float64 = 0.234,   # Roberts-Gelman-Gilks high-dimensional optimum
                  gamma_eta::Float64    = 2.0,
                  gamma_every::Int      = 25,      # brake more often than the stop checks, so the
                                                   # diminishing step has enough events to converge
                  # The ACCURACY gate: MCSE(mean)/sd(pooled) over non-exempt coordinates.
                  # 0.05 is Flegal-Haran-Jones (2008). This is what "good enough" means for
                  # a run whose deliverable is a mean and a standard deviation, and it
                  # replaces the no-worsening-R-hat gate, which a non-converging chain
                  # satisfies. See mcse_ratio in mcmc_diagnostics.jl.
                  mcse_target::Float64 = 0.05,
                  # Box bounds, for the AT-A-BOUND half of the convergence gate's exemption
                  # test. Pile-up must be measured in CONSTRAINED units because the logit
                  # transform puts each bound at infinity in t, where no threshold detects
                  # it. Optional: with lb/ub omitted the gate still applies its FROZEN test
                  # (distinct visited values), so a caller that cannot supply them degrades
                  # to a weaker exemption rather than to a wrong one.
                  lb::Union{Nothing,AbstractVector} = nothing,
                  ub::Union{Nothing,AbstractVector} = nothing,
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

    # The box defines the target, so it belongs in the header beside δ and CR: a log that
    # does not record it cannot tell a reader which density the chain sampled.
    verbose && @printf("[demc] d=%d  N=%d chains  gens=%d  δ=%d  CR=%.2f  γ(all d)=%.3f  box=%s  init=%s  threads=%d\n",
                       d, N, gens, δ, CR, γ_full,
                       isfinite(t_box) ? @sprintf("±%.4g", t_box) : "none (improper)",
                       init, nthreads())

    # The seed must carry prior mass, or the chain starts at log-target −Inf and nothing can
    # ever be accepted. Caught here rather than as a silent all-reject run.
    if isfinite(t_box)
        viol = findall(k -> abs(θ0f[k]) > t_box, 1:d)
        isempty(viol) || error("seed lies outside the prior box |t| ≤ $(t_box) in " *
                               "coordinate(s) $(viol) (values $(θ0f[viol])); widen " *
                               "MCMC_T_BOX or re-seed.")
    end

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
    elseif init === :widths
        # Disperse the population at a SUPPLIED per-coordinate scale rather than discovering
        # it by diffusion. This is the only init mode that does not have to grow: the widths
        # are the criterion's, so the step γ·(X_r1 − X_r2) is in the right units from
        # generation 1 and acceptance should start near the DE-MC optimum of 0.234 instead of
        # far above it (0.758 was measured on base_fc, which is the signature of a step far
        # too short). Chain 1 is held AT the seed so the best known point is never lost.
        init_widths === nothing &&
            error("run_demc: init = :widths requires init_widths (per-coordinate scale in " *
                  "the sampled space). Run code/tools/width_audit.jl to produce them.")
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
            # Asymmetric draw: the up-width upward, the down-width downward, scaled by
            # init_disperse. A symmetric draw on a coordinate resting against a floor sends
            # half its proposals into infeasibility, and the old fallback then replaced the
            # whole vector with the seed — so the coordinates the criterion resolves least
            # were the ones that ended up with no dispersion at all.
            @inbounds for k in 1:d
                z = randn(rng)
                dev[k] = init_disperse * (z >= 0.0 ? w[k] * z : wdn[k] * z)
            end
            # On infeasibility SHRINK toward the seed along the ray rather than discarding the
            # draw: the direction is kept and most of the dispersion survives. Halving
            # terminates in at most 20 steps, and only a chain whose whole ray is infeasible
            # falls back to the seed.
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
        # PER-COORDINATE scale. One scalar radius cannot serve coordinates whose
        # posterior widths span orders of magnitude: DE-MC's step size is the
        # population's own spread, so it contracts a population that starts too narrow
        # but not one that starts too wide (contraction needs accepted moves, and an
        # over-dispersed proposal is rejected). init_scale = nothing reproduces the
        # v17.1 isotropic draw exactly, so old call sites are unaffected.
        s = init_scale === nothing ? ones(d) : collect(float.(init_scale))
        length(s) == d || error("run_demc: init_scale has length $(length(s)), expected $d")
        all(x -> isfinite(x) && x > 0, s) ||
            error("run_demc: init_scale must be finite and strictly positive")
        # The ladder is now a MULTIPLIER on s: the vector carries the units, so r = 1.0
        # is the intended scale and shrinking happens only if feasibility forces it.
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
        # Report the realised scale, and the one number that says whether the
        # precondition holds: how many coordinates start WIDER than the target scale
        # they were given. That count must be 0 — a non-zero value means the population
        # cannot contract in those directions and their reported SD is biased upward.
        if verbose
            eff = radius .* s          # the scale actually drawn from, per coordinate
            # A coordinate is over-dispersed iff the scale actually drawn from exceeds
            # that coordinate's own POSTERIOR WIDTH. The comparison must be against the
            # width itself (init_width), NOT against s: s is already the clamped scale,
            # so eff .<= s holds by construction and comparing the two would be a
            # tautology that always reports 0. The clamp is exactly where genuine
            # over-dispersion enters — MCMC_SCREEN_CAP raises the scale of any
            # coordinate whose width is below the cap — so the cap is what this must
            # catch. init_width is the unclamped se_t vector in the same t units.
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
    # Which abort fired. Two paths set `aborted`, and they mean opposite things: :drift
    # says the chain found a better point than the seed, :acceptance says the proposal is
    # mis-scaled. Reporting one as the other sends the reader after the wrong fix.
    abort_why  = :none

    chain  = Array{Float64}(undef, d, N, gens)
    cand   = Matrix{Float64}(undef, d, N)
    lpc    = Vector{Float64}(undef, N)
    pop_sd = Vector{Float64}(undef, d)      # per-generation ESJD yardstick
    outbox = falses(N)                      # candidates outside the prior box, solve skipped
    nacc  = 0
    # Windowed counters, reset at each print. A cumulative acceptance rate hides
    # the current one once the early generations are averaged in, and `nfin`
    # separates a proposal rejected for being uphill from one rejected because
    # the solve failed — the two call for opposite fixes.
    wacc  = 0
    wfin  = 0
    wprop = 0
    weval = 0     # proposals actually sent to the solver, i.e. wprop minus box rejections
    # Proposal economics, windowed alongside the counters above. Acceptance says how
    # often a proposal lands; these say WHY it does or does not, which is what a tuning
    # decision needs.
    #
    #   wdlp   the Δlogπ of every feasible proposal. Its MEDIAN is the step-scale
    #          diagnostic: a well-scaled d-dimensional proposal sits at −1 to −3, which
    #          is what yields acceptance ≈ 0.234. A median near −20 means the median
    #          proposal is rejected with probability 1 − 2e−9 and the whole acceptance
    #          rate is riding a thin tail — invisible in the acceptance number itself.
    #   wesjd  expected squared jump distance, in the POPULATION's own metric (each
    #          coordinate standardised by its current population sd, so the measure is
    #          scale-free and computable at runtime without knowing the target). ESJD,
    #          not acceptance, is the mixing criterion: a shorter step that raises
    #          acceptance while lowering ESJD is moving less, not mixing better.
    wdlp  = Float64[]
    wesjd = 0.0
    # Check-window counters, reset at each CHECK rather than at each print. The two
    # strides are independent (print_every 250, check_every 250 today but not by
    # construction), so the stop rule cannot read the print counters: a print between
    # checks would zero them and the acceptance the rule sees would cover the wrong
    # window. Separate counters make the rule's window exactly check-to-check.
    # The adaptive DE scale and its own acceptance window, kept separate from the stop's so
    # the two cadences cannot interfere.
    gamma_scale = 1.0
    n_adapt     = 0
    wacc_ad     = 0
    # Feasible candidates in the adaptation window — the brake's denominator. See the update
    # below for why an infeasible candidate must not enter it.
    wfin_ad     = 0
    # The last conditional acceptance the brake acted on. Reported on the check line because
    # it is the quantity the brake steers by, and without it a log cannot be used to tell a
    # brake that has settled from one still chasing an unreachable target.
    acc_ad_last = NaN
    
    wacc_chk  = 0
    wprop_chk = 0
    # Two-consecutive-check state. NaN/false means "nothing to compare against yet", so
    # neither the stop nor the diagnosis can fire on the first check — one check is a
    # reading, two is evidence.
    lp_best_prev = NaN
    rhat_prev    = Inf
    stop_armed   = false
    diag_armed   = false
    # Per-chain log-posterior history, for the stuck-chain score. LMR average over the
    # last half of a 500-draw ring buffer; 250 is that window, capped so a short run
    # still scores over something.
    chain_lp     = Matrix{Float64}(undef, N, gens)
    # Two per-draw masks, RETURNED so an estimator chosen later can be computed from this
    # run instead of a new one. They cost N*gens bits each (~119 KB at 95x10 000).
    #   accepted  the draw is a NEW state, not a repeat of the previous generation. Lets an
    #             average be taken over unique states rather than over the padded chain.
    #   replaced  the draw was overwritten by the stuck-chain rule and is therefore NOT a
    #             draw from the target. LMR keep exactly this (all_outliner_results_nl,
    #             mpi_mcmc_mod.f90:423) and it is the only way to excise contaminated draws
    #             after the fact — n_replaced and last_replace are a count and a scalar,
    #             which say THAT it happened but not WHERE.
    accepted     = falses(N, gens)
    replaced     = falses(N, gens)
    out_window   = min(250, max(1, burn ÷ 2))
    # The retained window — the span a replacement overwrites for the outlier chain.
    # LMR's chain array is a cyclical buffer and the replacement overwrites all of it
    # (mpi_mcmc_mod.f90:423-424), so the faithful analogue here is the span that will
    # actually be pooled: gens - burn, which is 1000 at the paper's 10,000 with the last
    # 1000 retained (Appendix C p.86). Derived from burn, so it tracks any burn_frac.
    retain_len   = max(1, gens - burn)
    out_score    = Vector{Float64}(undef, N)
    n_replaced   = 0
    last_replace = 0
    # Replacement events since the last print, so the RATE is visible rather than only the
    # run-total. n_replaced and last_replace were returned but never printed, so an
    # 11-hour run gave no live signal on the one mechanism that concentrates the ensemble.
    wrep         = 0
    # Proposals rejected by the prior box since the last print. Reported separately from
    # `fin` because the two say different things: fin counts points where the equilibrium
    # does not exist, wbox counts points the prior excludes. A tuning decision reads them
    # differently — a high wbox means the box is binding, not that the model is fragile.
    wbox         = 0

    for g in 1:gens
        # Brake the DE scale toward the acceptance optimum. Robbins-Monro in log gamma, so the
        # multiplier stays positive.
        #
        # ONE-SIDED, and that is the design rather than a simplification. A symmetric controller
        # was tried and made things worse: from the shipped collapsed start acceptance is 0.739 at
        # g=25 — not because the step is well scaled but because the population has not spread yet
        # — so a symmetric rule reads "step too short", lengthens it 65%, and accelerates the very
        # expansion it exists to brake (acceptance then fell to 0.196 by g=50, against the
        # fixed-gamma control's 0.555 at the same generation). The asymmetry follows from the
        # measured dynamics: a symmetric accepted proposal adds its squared step to the population
        # variance, and that step is proportional to the variance already present, with no inward
        # pull to offset it. Widening is intrinsic and needs no help; only the overshoot needs
        # correcting. n_adapt counts braking events only, so the diminishing step is not spent
        # during the transient.
        # The denominator is FEASIBLE candidates, not all proposals, and that is not a detail.
        # The 0.234 optimum describes the Metropolis accept/reject on the target; an infeasible
        # candidate is a prior-zero rejection, a different mechanism whose rate does not respond
        # to the step. On the v21.0.1 base_fc run `fin` sat at 0.77 while acceptance among
        # feasible candidates saturated near 0.26, so UNCONDITIONAL acceptance was capped at
        # 0.77·0.26 ≈ 0.20 — below the 0.234 target. The brake could therefore never succeed: it
        # shortened the step at every window for 2000 generations (γ× 0.545 → 0.142, still
        # falling) and esjd collapsed 63×, while the population had stopped contracting after
        # g = 750. Under the feasible denominator that run would have braked at g = 500 and 750
        # and then stopped at g = 1000, where conditional acceptance reached 0.243.
        #
        # No dead band is needed: the shortfall (acc − target) shrinks as the gap closes, so the
        # update self-limits rather than oscillating.
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
            i1, i2 = _de_pairs(N, c, δ, rng)
            diff = zeros(d)
            @inbounds for j in 1:δ
                @views diff .+= Xc[:, i1[j]] .- Xc[:, i2[j]]
            end
            # Draw the mask BEFORE the proposal: γ depends on how many coordinates it
            # updates, so the realised count has to be known first.
            mask = rand(rng, d) .< CR
            any(mask) || (mask[rand(rng, 1:d)] = true)   # always move ≥1 dim
            # gamma_scale multiplies the DE step only. The mode jump keeps γ = 1: it is a
            # deliberate full-population hop and rescaling it would defeat its purpose.
            γ = jump ? 1.0 : gamma_scale * 2.38 / sqrt(2δ * count(mask))
            e = b_mult .* randn(rng, d)
            ε = b_add  .* randn(rng, d)
            @views prop = Xc[:, c] .+ (1.0 .+ e) .* γ .* diff .+ ε
            @inbounds for k in 1:d
                cand[k, c] = mask[k] ? prop[k] : Xc[k, c]
            end
            # A candidate outside the prior box has zero prior mass, so it is rejected with
            # certainty and its solve is wasted. Flagging it here skips the solve; the
            # accept/reject outcome is identical.
            #
            # NOTE this is a rejection, not LMR's reject-and-redraw (mpi_mcmc_mod.f90:294-304).
            # Redrawing until the candidate lands in the box makes the effective proposal
            # q(x→y)·1{y∈B}/Z(x) with Z(x) varying over x, so it is no longer symmetric and
            # detailed balance fails — biasing the retained sample, which is the sample the
            # standard errors are computed from. Rejecting is exact and, because the check
            # precedes evaluation, costs nothing.
            outbox[c] = isfinite(t_box) && any(abs(cand[k, c]) > t_box for k in 1:d)
        end

        _eval_population!(lpc, logπ, cand, N, parallel; skip = outbox)
        wbox += count(outbox)

        # Population sd per coordinate, this generation, as the metric for ESJD. Taken
        # BEFORE the accept loop so every jump in this generation is measured against one
        # fixed yardstick rather than one that shifts as chains update.
        #
        # Floored at b_add because that is the smallest displacement the proposal can
        # produce: a coordinate whose population sd is below it has not separated from the
        # seed, and standardising by that sd divides a real step by floating-point noise.
        # Unfloored, an :at_seed start reports esjd ≈ 5e23 in its first generations —
        # cancellation in std() over N nearly identical members, not a large jump.
        sd_floor = max(b_add, eps())
        @inbounds for k in 1:d
            pop_sd[k] = max(std(@view Xc[k, :]), sd_floor)
        end

        for c in 1:N
            wprop += 1; wprop_chk += 1
            # fin and dlp describe candidates the SOLVER saw. An out-of-box candidate was
            # never evaluated, so folding it in would report the prior as model fragility:
            # on a target that is finite everywhere, a binding box would still print
            # fin = 0.35. Acceptance keeps the full denominator — an out-of-box proposal is
            # a real rejection.
            if !outbox[c]
                weval += 1
                if isfinite(lpc[c])
                    wfin += 1; wfin_ad += 1
                    push!(wdlp, lpc[c] - lp[c])
                end
            end
            if log(rand(rng)) < lpc[c] - lp[c]           # α = exp(Δ log-posterior)
                jd = 0.0
                @inbounds for k in 1:d
                    jd += ((cand[k, c] - X[k, c]) / pop_sd[k])^2
                end
                wesjd += jd
                @views X[:, c] .= cand[:, c]
                lp[c] = lpc[c]
                nacc += 1; wacc += 1; wacc_chk += 1; wacc_ad += 1
                accepted[c, g] = true
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
                    # LMR SCOPE (mpi_mcmc_mod.f90:423-424):
                    #   all_population_knl(:,C,:) = all_population_knl(:,best_chain,:)
                    # Their chain array is a cyclical buffer and the `:` overwrites ALL of it,
                    # so an outlier chain's entire retained history becomes a copy of the best
                    # chain's, not just its current position. Since solution_theta averages
                    # that buffer (:485), the estimator can average duplicated chains. Ours
                    # keeps the full gens-length history rather than a ring buffer, so the
                    # faithful analogue is the span that will be pooled: retain_len.
                    # `replaced` marks every generation touched, so the duplication is
                    # auditable after the fact rather than invisible.
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
            # Acceptance alone cannot say whether the run is producing a posterior, nor
            # what to change if it is not. Four numbers answer four distinct questions:
            #
            #   acc   is the chain moving at all
            #   dlp   the MEDIAN feasible proposal's Δlogπ — is the step the right SIZE?
            #         −1 to −3 is well scaled; −20 means the median proposal is hopeless
            #         and acceptance is riding a tail. This is the tuning number.
            #   esjd  expected squared jump per proposal in population-sd units — is the
            #         chain COVERING ground? Read together with dlp: acceptance rising
            #         while esjd falls is a shorter step moving less, not better mixing.
            #   R̂/ESS the deliverable itself, over non-exempt coordinates.
            #
            # `rep` appears only when the stuck-chain rule fired since the last print. On a
            # healthy population it is silent, so its presence is the signal. Past the burn
            # boundary it reads `rep=N RETAINED`, because a replacement there puts duplicated
            # draws in the pooled sample and those draws are not from the target — the
            # requirement CH Theorem 2 integrates against. That is the one reading which
            # changes what the reported SD means, so it is spelled out rather than flagged.
            #
            # `fin` appears only when it drops below 0.95. At the measured 0.98 it is
            # noise on every line; below that it is the difference between a proposal
            # rejected for being uphill and one rejected because the solve failed, which
            # call for opposite fixes. Cumulative acceptance is dropped outright — it
            # averages in the collapsed :at_seed start forever, so it falls monotonically
            # whatever the chain is doing.
            _, wr, me = converged_sequential(chain, g, burn_frac, d;
                                             rhat_max = rhat_max, ess_min = ess_target,
                                             lb = lb, ub = ub)
            fin_frac = wfin / max(weval, 1)
            @printf("[demc] gen %5d/%d  acc=%.3f  dlp=%s  esjd=%.3f  max logπ=%.6e  \
                     R̂=%s  ESS=%s%s%s%s\n",
                    g, gens, wacc / max(wprop, 1),
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
            # The stop gates on accepted moves, drift-flatness and non-worsening R̂ —
            # not on an R̂ threshold. Measured on the 18.4.1 base_fc chain, the old
            # threshold gate fired at 0 of 16 checkpoints (true-stop rate zero as well
            # as false-stop rate zero), because worst R̂ trends UPWARD with budget while
            # accepted moves stay flat. R̂ and ESS are still computed and printed.
            # Both conditions require TWO CONSECUTIVE checks: one check is a reading,
            # two is evidence.
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
                # Only the quantities the stop actually GATES on, each beside its
                # threshold so a reader can see which one is binding. R̂ and ESS are on
                # every generation line already and are reported, not gated, so they are
                # not repeated here.
                # The FIRST check has no previous reading to difference against, so the
                # gated quantity is genuinely undefined there and printed "n/a" — one
                # check is a reading, two is evidence. But "n/a" alone left the first
                # check line carrying no information at all, which reads like a defect.
                # Print the cumulative drift from the seed beside it: that IS defined at
                # the first check, it is the quantity the run is descending on, and its
                # size tells the reader immediately whether the chain is sampling near
                # the seed or still travelling away from it.
                # mcse leads: it is the ACCURACY gate, the one quantity that says whether the
                # reported mean and sd are usable yet. moves and drift are necessary
                # conditions on the sample, not statements about its accuracy.
                # R-hat travels beside mcse because mcse is bounded above by 1/sqrt(N)
                # (= 0.103 at N = 95): a chain at R-hat = 9.7 reads mcse = 0.102, which
                # against a 0.05 target looks like a factor of two rather than the
                # factor of nine it is. gamma is printed so a run that adapts its way
                # out of a mis-scaled population leaves a record of having done so.
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
    end

    # Truncate to what was actually filled, and re-derive burn-in at g_final so a
    # sequential stop discards the same FRACTION as a full-length run.
    chain    = chain[:, :, 1:g_final]
    chain_lp = chain_lp[:, 1:g_final]
    accepted = accepted[:, 1:g_final]
    replaced = replaced[:, 1:g_final]
    burn  = clamp(floor(Int, burn_frac * g_final), 0, g_final - 1)

    draws = reshape(chain[:, :, burn+1:end], d, :)        # pool post-burn-in chains
    # chain_lp, accepted and replaced are returned so the bundle can persist them. Without
    # chain_lp the log-target of a draw is recoverable only by re-solving — 950 000 solves at
    # the shipped budget — which would make any later change of estimator a full re-run.
    return (; draws, chain, chain_lp, accepted, replaced,
              accept = nacc / (g_final * N), lp, N,
              gens = g_final, gens_requested = gens, burn,
              lp_seed, lp_best, theta_best = θ_best,
              drift = lp_best - lp_seed, aborted, abort_why,
              gamma_scale, n_adapt,
              n_replaced, last_replace)
end

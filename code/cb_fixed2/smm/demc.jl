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
  init_spread sd of the Gaussian cloud used to disperse chains around θ0
  parallel    thread the population evaluation over chains (see header note)
  rng, verbose, print_every

Returns `(; draws, chain, accept, lp, N, gens, burn)` where
  draws : d × (N·kept) matrix of post-burn-in samples (UNCONSTRAINED), and
  chain : d × N × gens full history (for trace plots / R̂ / ESS).
"""
function run_demc(logπ, θ0::AbstractVector{<:Real};
                  N::Int = 0, gens::Int = 4000, burn_frac::Float64 = 0.5,
                  CR::Float64 = 0.90, δ::Int = 1,
                  b_add::Float64 = 1e-4, b_mult::Float64 = 1e-2,
                  jump_every::Int = 10, init_spread::Float64 = 1e-2,
                  parallel::Bool = true,
                  rng::AbstractRNG = MersenneTwister(20260624),
                  verbose::Bool = true, print_every::Int = 250)

    d = length(θ0)
    N = N > 0 ? N : max(2d, 16)
    δ = clamp(δ, 1, max(1, (N - 1) ÷ 2))
    γ0 = 2.38 / sqrt(2δ * d)                 # ter Braak / DE-MC scaling
    burn = clamp(floor(Int, burn_frac * gens), 0, gens - 1)
    θ0f = collect(float.(θ0))

    verbose && @printf("[demc] d=%d  N=%d chains  gens=%d  δ=%d  CR=%.2f  γ0=%.3f  threads=%d\n",
                       d, N, gens, δ, CR, γ0, nthreads())

    # ── initial population: disperse around the seed (DE-MC needs spread) ──
    X = repeat(θ0f, 1, N) .+ init_spread .* randn(rng, d, N)
    X[:, 1] .= θ0f                            # keep one chain exactly at the seed
    lp = Vector{Float64}(undef, N)
    _eval_population!(lp, logπ, X, N, parallel)
    for c in 1:N                              # nudge infeasible seeds until finite
        t = 0
        while !isfinite(lp[c]) && t < 200
            @views X[:, c] .= θ0f .+ init_spread .* randn(rng, d)
            lp[c] = logπ(view(X, :, c)); t += 1
        end
        isfinite(lp[c]) ||
            error("run_demc: chain $c has no feasible start near θ0 — check the seed/objective.")
    end

    chain = Array{Float64}(undef, d, N, gens)
    cand  = Matrix{Float64}(undef, d, N)
    lpc   = Vector{Float64}(undef, N)
    nacc  = 0

    for g in 1:gens
        Xc = copy(X)                          # freeze current population (= one_population_kn)
        γ  = (g % jump_every == 0) ? 1.0 : γ0 # periodic mode jump

        for c in 1:N
            i1, i2 = _de_pairs(N, c, δ, rng)
            diff = zeros(d)
            @inbounds for j in 1:δ
                @views diff .+= Xc[:, i1[j]] .- Xc[:, i2[j]]
            end
            e = b_mult .* randn(rng, d)
            ε = b_add  .* randn(rng, d)
            @views prop = Xc[:, c] .+ (1.0 .+ e) .* γ .* diff .+ ε
            mask = rand(rng, d) .< CR
            any(mask) || (mask[rand(rng, 1:d)] = true)   # always move ≥1 dim
            @inbounds for k in 1:d
                cand[k, c] = mask[k] ? prop[k] : Xc[k, c]
            end
        end

        _eval_population!(lpc, logπ, cand, N, parallel)

        for c in 1:N
            if log(rand(rng)) < lpc[c] - lp[c]           # α = exp(Δ log-posterior)
                @views X[:, c] .= cand[:, c]
                lp[c] = lpc[c]
                nacc += 1
            end
        end
        @views chain[:, :, g] .= X

        if verbose && (g % print_every == 0 || g == gens)
            @printf("[demc] gen %5d/%d  accept=%.3f  max logπ=%.6e\n",
                    g, gens, nacc / (g * N), maximum(lp)); flush(stdout)
        end
    end

    draws = reshape(chain[:, :, burn+1:end], d, :)        # pool post-burn-in chains
    return (; draws, chain, accept = nacc / (gens * N), lp, N, gens, burn)
end

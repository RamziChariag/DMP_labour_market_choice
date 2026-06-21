############################################################
# candidates.jl — SMM candidate-generation / initialisation layer
#
# Runs BEFORE SA/DE.  Draws a scrambled-Sobol sample over the free
# parameter box, evaluates each point on a coarse grid, keeps the
# feasible (finite-Q) subset, clusters them with complete-linkage
# hierarchical clustering (`hclust`) in normalised constrained space,
# ranks members within each cluster by Q, and returns a `SeedBank`
# (defined in smm_params.jl).  The bank is cached to disk and reused
# unless stale.
#
# Public entry points
#   generate_candidates(spec; window, ...)          build a fresh bank
#   load_or_generate_candidates(spec, path; ...)     load cache or build
#
# Dependencies (imported in smm_main.jl):
#   QuasiMonteCarlo  — scrambled Sobol sampling
#   Clustering       — hclust / cutree
#   Serialization    — cache I/O
#
# Replicability: the Sobol scramble is seeded from spec.run.cand_seed via
# Random.seed!; the whole layer is therefore a deterministic function of
# (cand_seed, grid, n_sample, bounds, fixed).
############################################################


"""
    _candidate_meta(spec, window, n_sample, seed, min_cluster) → NamedTuple

Staleness / validity key for a candidate cache.  Two caches are
interchangeable iff their structural fields match (see `_meta_matches`).
Free-parameter identity is stored as `(block, name)` pairs so the key
stays correct after the parameter-container refactor.
"""
function _candidate_meta(spec::SMMSpec, window::Symbol,
                         n_sample::Int, seed::Int, min_cluster::Int)
    free_names = [(ps.block, ps.name) for ps in spec.free]
    lb         = [ps.lb for ps in spec.free]
    ub         = [ps.ub for ps in spec.free]
    grid       = (spec.run.cand_Nx, spec.run.cand_Np_U, spec.run.cand_Np_S)
    return (window      = window,
            free_names  = free_names,
            lb          = lb,
            ub          = ub,
            fixed       = spec.fixed,
            grid        = grid,
            n_sample    = n_sample,
            seed        = seed,
            min_cluster = min_cluster)
end


"""
    _meta_matches(a, b) → Bool

Compare two candidate-cache metas for interchangeability.  A cache is
rejected (and regenerated) if the window, free-parameter set, bounds,
fixed-parameter values, or coarse grid differ.  (n_sample / seed /
min_cluster do not invalidate: a cache built with at least the current
content is still usable; change them with CLUSTERS_FORCE_REGEN.)
"""
function _meta_matches(a, b)
    return a.window     == b.window     &&
           a.free_names == b.free_names &&
           a.lb         == b.lb         &&
           a.ub         == b.ub         &&
           a.fixed      == b.fixed      &&
           a.grid       == b.grid
end


"""
    _hclust_labels(X, min_cluster) → Vector{Int}

Cluster labels for the d×m matrix `X` (features in ROWS, observations in
COLUMNS).  Returns one integer label per observation; clusters are 1..K and
noise is 0.

Complete-linkage hierarchical clustering (`hclust`/`cutree`, from Clustering.jl)
on pairwise Euclidean distances over the columns of `X`, cut at the largest gap
in the merge heights.  Clusters with fewer than `min_cluster` members are
relabelled as noise (0); the rest get sequential ids 1..K.
"""
function _hclust_labels(X::AbstractMatrix{Float64}, min_cluster::Int)
    m = size(X, 2)
    m < 2 && return ones(Int, m)

    D = zeros(Float64, m, m)
    @inbounds for i in 1:m
        for j in i+1:m
            s = 0.0
            for k in 1:size(X, 1)
                s += (X[k, i] - X[k, j])^2
            end
            dij = sqrt(s)
            D[i, j] = dij
            D[j, i] = dij
        end
    end

    hc  = hclust(D; linkage = :complete)
    h   = hc.heights
    raw = if length(h) < 2
        ones(Int, m)
    else
        gaps  = diff(h)
        gi    = argmax(gaps)
        cut_h = (h[gi] + h[gi + 1]) / 2
        cutree(hc; h = cut_h)
    end

    counts = Dict{Int,Int}()
    for l in raw
        counts[l] = get(counts, l, 0) + 1
    end

    newid  = Dict{Int,Int}()
    nextid = 0
    labels = zeros(Int, m)
    for (i, l) in enumerate(raw)
        if counts[l] >= min_cluster
            if !haskey(newid, l)
                nextid += 1
                newid[l] = nextid
            end
            labels[i] = newid[l]
        else
            labels[i] = 0
        end
    end
    return labels
end


"""
    generate_candidates(spec; window, n_sample, seed, min_cluster, show_trace)
        → SeedBank

Sample (scrambled Sobol), evaluate on the coarse grid, filter to finite-Q,
cluster with complete-linkage hierarchical clustering (`hclust`), rank within
clusters by Q, and return a `SeedBank`.  If clustering finds no clusters (all
noise), fall back to the best feasible points (one synthetic cluster) so SA/DE
always receive a non-empty bank.
"""
function generate_candidates(spec::SMMSpec;
                             window      :: Symbol = :unknown,
                             n_sample    :: Int    = spec.run.cand_n_sample,
                             seed        :: Int    = spec.run.cand_seed,
                             min_cluster :: Int    = spec.run.cand_min_cluster,
                             show_trace  :: Bool   = true)
    d  = length(spec.free)
    lb = [ps.lb for ps in spec.free]
    ub = [ps.ub for ps in spec.free]
    meta = _candidate_meta(spec, window, n_sample, seed, min_cluster)

    # 1. Scrambled-Sobol sample in the constrained box [lb, ub]  (d × n_sample).
    #    Seed the global RNG so the scramble (and hence the layer) is replicable.
    Random.seed!(seed)
    X = QuasiMonteCarlo.sample(n_sample, lb, ub,
                               SobolSample(R = OwenScramble(base = 2, pad = 32)))

    # Map each column (constrained) → unconstrained θ.
    thetas = Vector{Vector{Float64}}(undef, n_sample)
    for j in 1:n_sample
        θ = Vector{Float64}(undef, d)
        for (k, ps) in enumerate(spec.free)
            x_k  = clamp(X[k, j],
                         ps.lb + 1e-8 * (ps.ub - ps.lb),
                         ps.ub - 1e-8 * (ps.ub - ps.lb))
            θ[k] = _to_unconstrained(x_k, ps.lb, ps.ub)
        end
        thetas[j] = θ
    end

    # 2. Evaluate on the coarse grid (parallel).
    if show_trace
        @printf("  [candidates]  evaluating %d Sobol points on coarse grid (Nx=%d, Np_U=%d, Np_S=%d)...\n",
                n_sample, spec.run.cand_Nx, spec.run.cand_Np_U, spec.run.cand_Np_S)
        flush(stdout)
    end
    Qs = fill(Inf, n_sample)
    Threads.@threads for j in 1:n_sample
        Qs[j] = smm_objective(thetas[j], spec;
                              Nx   = spec.run.cand_Nx,
                              Np_U = spec.run.cand_Np_U,
                              Np_S = spec.run.cand_Np_S)
    end

    # 3. Filter to finite Q.
    feas = findall(isfinite, Qs)
    if show_trace
        @printf("  [candidates]  feasible (finite Q): %d / %d\n", length(feas), n_sample)
        flush(stdout)
    end
    if isempty(feas)
        @warn "candidates: no feasible Sobol points — returning empty bank (SA/DE will use defaults)."
        return SeedBank(Vector{Vector{Float64}}(), Float64[], Int[], meta)
    end

    feas_thetas = thetas[feas]
    feas_Q      = Qs[feas]
    m           = length(feas)

    # 4. Normalise to [0,1]^d in constrained space.  Features in COLUMNS (d × m).
    Xn = Matrix{Float64}(undef, d, m)
    for (col, θ) in enumerate(feas_thetas)
        for (k, ps) in enumerate(spec.free)
            xk = _to_constrained(θ[k], ps.lb, ps.ub)
            Xn[k, col] = (xk - ps.lb) / (ps.ub - ps.lb)
        end
    end

    # 5. Cluster with complete-linkage hierarchical clustering; label 0 = noise.
    labels = _hclust_labels(Xn, min_cluster)
    keep   = findall(>(0), labels)

    if isempty(keep)
        # Fallback: no clusters — seed from the best feasible points (one
        # synthetic cluster) so SA/DE always receive a non-empty bank.
        order = sortperm(feas_Q)
        ntop  = min(length(order), max(2 * min_cluster, 10))
        sel   = order[1:ntop]
        if show_trace
            @printf("  [candidates]  clustering found no clusters — fallback to %d best feasible points.\n", ntop)
            flush(stdout)
        end
        return SeedBank(feas_thetas[sel], feas_Q[sel], fill(1, length(sel)), meta)
    end

    bank_thetas = feas_thetas[keep]
    bank_Q      = feas_Q[keep]
    bank_lab    = labels[keep]

    if show_trace
        @printf("  [candidates]  hclust: %d clusters, %d clustered points (%d noise dropped)\n",
                length(unique(bank_lab)), length(keep), m - length(keep))
        flush(stdout)
    end

    return SeedBank(bank_thetas, bank_Q, bank_lab, meta)
end


"""
    load_or_generate_candidates(spec, path; window, force_regen, show_trace)
        → SeedBank

Load a cached `SeedBank` from `path` if present and not stale; otherwise
generate a fresh one and save it.  With `force_regen=true`, always
regenerate and overwrite.
"""
function load_or_generate_candidates(spec::SMMSpec, path::String;
                                     window      :: Symbol = :unknown,
                                     force_regen :: Bool   = false,
                                     show_trace  :: Bool   = true)
    want_meta = _candidate_meta(spec, window,
                                spec.run.cand_n_sample, spec.run.cand_seed,
                                spec.run.cand_min_cluster)

    if !force_regen && isfile(path)
        bank = try
            open(deserialize, path)
        catch e
            @warn "candidates: failed to read cache ($e) — regenerating."
            nothing
        end
        if bank isa SeedBank && _meta_matches(bank.meta, want_meta)
            show_trace && @printf("  [candidates]  loaded cached bank: %d candidates, %d clusters\n",
                                  length(bank.candidates),
                                  isempty(bank.labels) ? 0 : length(unique(bank.labels)))
            return bank
        elseif bank isa SeedBank
            show_trace && @printf("  [candidates]  cache stale (window/free-set/bounds/fixed/grid changed) — regenerating.\n")
        end
    end

    bank = generate_candidates(spec; window = window, show_trace = show_trace)
    try
        mkpath(dirname(path))
        open(path, "w") do io
            serialize(io, bank)
        end
        show_trace && @printf("  [candidates]  saved bank → %s\n", path)
        flush(stdout)
    catch e
        @warn "candidates: failed to save cache ($e)."
    end
    return bank
end

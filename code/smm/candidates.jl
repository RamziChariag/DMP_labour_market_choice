############################################################
# smm/candidates.jl — candidate start points for INIT_MODE = :clusters
#
# Draws a scrambled Sobol sample over the free-parameter box, evaluates it on the
# coarse candidate grid, keeps the feasible (finite-Q) points, clusters them by
# complete-linkage hierarchical clustering in normalised constrained space and returns
# a `SeedBank` (smm_params.jl). The bank is cached on disk.
#
# The layer is a deterministic function of (cand_seed, grid, n_sample, bounds, fixed).
#
# Entry points: generate_candidates, load_or_generate_candidates.
############################################################


"""
    _candidate_meta(spec, window, n_sample, seed, min_cluster) → NamedTuple

Cache key of a candidate bank: window, free parameters as `(block, name)`, bounds, pinned
values per period, coarse grid, and the sampling settings.
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
            fixed       = pinned_key(spec),
            grid        = grid,
            n_sample    = n_sample,
            seed        = seed,
            min_cluster = min_cluster)
end


"""
    _meta_matches(a, b) → Bool

Whether two cache keys describe interchangeable banks: same window, free set, bounds,
pinned values and coarse grid. `n_sample`, `seed` and `min_cluster` are not compared;
rebuild with CLUSTERS_FORCE_REGEN after changing them.
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

Cluster labels for the columns of the d×m matrix `X`: complete-linkage hierarchical
clustering on Euclidean distances, cut at the largest gap in the merge heights. Clusters
are numbered 1..K; clusters with fewer than `min_cluster` members are labelled 0 (noise).
"""
function _hclust_labels(X::AbstractMatrix{Float64}, min_cluster::Int)
    m = size(X, 2)
    m < 2 && return ones(Int, m)

    D = zeros(Float64, m, m)
    @inbounds for i in 1:m
        for j in i+1:m
            s = 0.0
            for k in axes(X, 1)
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
    generate_candidates(spec; window, n_sample, seed, min_cluster, show_trace) → SeedBank

Build a candidate bank for `spec`: `n_sample` scrambled Sobol points in the box (scramble
seeded by `seed`), scored by `smm_objective` on the coarse candidate grid, the feasible
ones clustered (`_hclust_labels`) with noise dropped. When nothing clusters, the bank is
the best max(2·min_cluster, 10) feasible points as one cluster; when nothing is feasible,
it is empty.
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

    # Sobol sample in the constrained box (d × n_sample); the scramble draws from the
    # global RNG, seeded here.
    Random.seed!(seed)
    X = QuasiMonteCarlo.sample(n_sample, lb, ub,
                               SobolSample(R = OwenScramble(base = 2, pad = 32)))

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

    # Constrained coordinates normalised to [0,1]^d, one column per point.
    Xn = Matrix{Float64}(undef, d, m)
    for (col, θ) in enumerate(feas_thetas)
        for (k, ps) in enumerate(spec.free)
            xk = _to_constrained(θ[k], ps.lb, ps.ub)
            Xn[k, col] = (xk - ps.lb) / (ps.ub - ps.lb)
        end
    end

    labels = _hclust_labels(Xn, min_cluster)
    keep   = findall(>(0), labels)

    if isempty(keep)
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
    load_or_generate_candidates(spec, path; window, force_regen, show_trace) → SeedBank

The candidate bank cached at `path` when it exists and its key matches `spec`
(`_meta_matches`); otherwise, or with `force_regen`, a fresh bank, saved to `path`.
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

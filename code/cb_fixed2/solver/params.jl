############################################################
# params.jl — Structs and default parameter initialisation
#
# Each parameter is stored in the block whose solver code reads it:
#   :unsk  ← PU, bU, bT, α_U
#   :skl   ← gamma_PS, bS, a_Γ, b_Γ
# Storage reflects only which solver block consumes a parameter;
# "regime-specific" denotes the estimation category alone.
############################################################


# ============================================================
# Structs
# ============================================================

##############
# Common
##############

Base.@kwdef struct CommonParams
    r   :: Float64    # discount rate
    ν   :: Float64    # demographic exit rate
    φ   :: Float64    # training completion rate

    a_ℓ :: Float64    # Beta shape 1 for worker-type density ℓ(x)
    b_ℓ :: Float64    # Beta shape 2 for worker-type density ℓ(x)

    c   :: Float64    # SCALE-FREE training cost coefficient c̃: c(x) = (1−x)·exp(c̃−x)
                      # Dollar cost = exp(A)·c(x): callers multiply the FUNCTION by
                      # exp(A) outside (never add A to c̃ inside the exponent), so the
                      # training margin compares T(x) − exp(A)·c(x) with U^search(x)
                      # and is invariant to A like every other policy margin.

    A   :: Float64 = 0.0   # LOG aggregate production scale; the model uses exp(A) as the effective scale (multiplies π_U, π_S, b·, k·, σ_S). A = 0 ⇒ effective scale 1.
end


Base.@kwdef struct CommonGrids
    x   :: Vector{Float64}    # worker-type grid (Gauss–Jacobi nodes, see build_type_grid)
    wx  :: Vector{Float64}    # worker-type POPULATION weights: dot(density, wx) ≈ ∫ density dx
                              # for any density ∝ ℓ, with dot(ℓ, wx) = 1 exactly. ℓ is folded
                              # into wx (Gauss–Jacobi), so wx integrates densities, not bare f(x).
    ℓ   :: Vector{Float64}    # worker-type density Beta(a_ℓ,b_ℓ) evaluated on x-grid (pointwise use)
end


##############
# Unskilled
##############

Base.@kwdef struct UnskilledParams
    μ   :: Float64    # matching efficiency
    η   :: Float64    # matching elasticity
    k   :: Float64    # vacancy posting cost, in months of average unskilled
                      # output: dollar flow = k · π̄_U (mean_output_U, grids.jl)
    β   :: Float64    # worker Nash bargaining weight
    λ   :: Float64    # damage-shock arrival rate

    # Absorbed from the former RegimeParams (read by the unskilled block):
    PU  :: Float64    # unskilled sector productivity shifter
    bU  :: Float64    # flow payoff in unskilled unemployment
    bT  :: Float64    # flow payoff while in training
    α_U :: Float64    # unskilled damage-shock Beta shape (Beta(α_U, 1))

    σ_w :: Float64 = 0.0   # log-wage measurement-error SD (σ_wU); 0 ⇒ no measurement error
end


Base.@kwdef struct UnskilledGrids
    p   :: Vector{Float64}    # unskilled match-quality grid
    wp  :: Vector{Float64}    # quadrature weights on unskilled p-grid
end


Base.@kwdef mutable struct UnskilledCache
    # Values
    Usearch   :: Vector{Float64}    # value of unskilled search
    U         :: Vector{Float64}    # value of untrained unemployment
    T         :: Vector{Float64}    # value of training

    Jfrontier :: Vector{Float64}    # firm value at frontier p = 1
    pstar     :: Vector{Float64}    # unskilled reservation quality cutoff
    τT        :: Vector{Float64}    # training policy indicator

    # Composition
    u         :: Vector{Float64}    # density of untrained unemployed
    t         :: Vector{Float64}    # density of workers in training

    # Cross-market contribution carried from the skilled block:
    # duS_carry(x) = d(x) · u_S(x).  Held constant within one unskilled
    # block solve; refreshed by the global outer loop after each skilled
    # solve, and used in the augmented free-entry condition.
    duS_carry :: Vector{Float64}

    # Cross-market POLICY carried from the skilled block (lagged one global
    # pass, exactly like duS_carry).  d_carry(x) = d(x) ∈ {0,1}.  Gates the
    # training decision: a worker who would search the unskilled market once
    # trained (d = 1) does not initiate training,
    #   τ(x) = 1{−exp(A)·c(x) + T(x) ≥ U^search(x)} · 1{d(x) = 0}.
    # This is the equilibrium refinement that excludes train-then-search-
    # unskilled ("stipend-farming") equilibria under large b_T; equilibria in
    # which the τ and d regions never overlap are numerically unchanged.
    d_carry   :: Vector{Float64}

    # Scalar
    θ         :: Float64            # unskilled market tightness
end


##############
# Skilled
##############

Base.@kwdef struct SkilledParams
    μ   :: Float64    # matching efficiency
    η   :: Float64    # matching elasticity
    k   :: Float64    # vacancy posting cost, in months of average skilled
                      # output: dollar flow = k · π̄_S (mean_output_S, grids.jl)
    β   :: Float64    # worker Nash bargaining weight

    λ   :: Float64    # skilled quality-shock arrival rate
    σ   :: Float64    # flow cost of on-the-job search

    # Absorbed from the former RegimeParams (read by the skilled block):
    gamma_PS :: Float64    # skilled productivity shape: PS(x) = γ·x^{γ−1} (Beta(γ,1) PDF)
    bS       :: Float64    # flow payoff in skilled unemployment
    a_Γ      :: Float64    # skilled match-quality Beta shape 1
    b_Γ      :: Float64    # skilled match-quality Beta shape 2

    ξ        :: Float64 = 0.0   # exogenous skilled separation hazard ξ_S (0 ⇒ recovers the no-ξ model)
    σ_w      :: Float64 = 0.0   # log-wage measurement-error SD (σ_wS); 0 ⇒ no measurement error
end


Base.@kwdef struct SkilledGrids
    p   :: Vector{Float64}    # skilled match-quality grid
    wp  :: Vector{Float64}    # quadrature weights on skilled p-grid
end


Base.@kwdef struct SkilledPrecomp
    Γvals        :: Vector{Float64}    # CDF Γ(p) on skilled p-grid
    γvals        :: Vector{Float64}    # density γ(p) on skilled p-grid
    tail_weights :: Vector{Float64}    # quadrature tail weights for Γ-integrals
end


Base.@kwdef mutable struct SkilledCache
    U     :: Vector{Float64}           # value of skilled unemployment (= max{U^(0), U^(1)})

    E0    :: Matrix{Float64}           # worker value, employed, no OJS
    E1    :: Matrix{Float64}           # worker value, employed, with OJS
    J0    :: Matrix{Float64}           # firm value, filled job, no OJS
    J1    :: Matrix{Float64}           # firm value, filled job, with OJS

    pstar :: Vector{Float64}           # endogenous separation cutoff
    poj   :: Vector{Float64}           # OJS cutoff

    # Cross-market policy on the type grid.
    # d(x) = 1 ⇔ U^(1)_S(x) > U^(0)_S(x), stored as Float64.
    d     :: Vector{Float64}

    u     :: Vector{Float64}           # density of skilled unemployed
    e     :: Matrix{Float64}           # density of skilled employed by (x, p)

    θ     :: Float64                   # skilled market tightness
end


##############
# Simulation controls
##############

Base.@kwdef struct SimParams
    tol_inner      :: Float64    # tolerance for inner fixed point
    tol_outer_U    :: Float64    # tolerance for unskilled outer loop
    tol_outer_S    :: Float64    # tolerance for skilled outer loop
    tol_global     :: Float64    # tolerance for global fixed point

    maxit_inner    :: Int        # max iterations in inner loop
    maxit_outer    :: Int        # max iterations in block outer loops
    maxit_global   :: Int        # max iterations in global loop

    conv_streak    :: Int        # required consecutive convergence hits

    use_anderson   :: Bool       # toggle Anderson acceleration
    anderson_m     :: Int        # Anderson memory
    anderson_reg   :: Float64    # Anderson regularization

    damp_pstar_U   :: Float64    # damping for unskilled pstar
    damp_pstar_S   :: Float64    # damping for skilled pstar updates

    verbose        :: Int        # verbosity level
    verbose_stride :: Int        # print every N iterations

    # Inner value-loop under-relaxation (1.0 = undamped).  Damps the iterated
    # value (skilled U; unskilled U^search) to stabilise oscillatory inner maps
    # in high-fβ regions; consumed in *_inner_loop!.
    damp_inner_U   :: Float64 = 1.0
    damp_inner_S   :: Float64 = 1.0

    # Inner-loop divergence early-abort.  inner_B sweeps are skipped before the
    # no-contraction test fires; inner_B = 0 disables it (and the parameter
    # rejection built on it).  inner_K is BOTH the inner no-contraction window
    # (W ≡ K) AND the number of consecutive divergent outer iterations that
    # rejects the parameter.
    inner_B :: Int = 0
    inner_K :: Int = 5

    # Outer-loop stall detection (same no-contraction test on the block outer
    # loop).  If the outer residual fails to shrink over outer_K iterations after
    # an outer_B burn-in, the block stops and hands control back to the GLOBAL
    # loop — it does NOT reject the parameter (the global warm-start refines it).
    # outer_B = 0 disables the stall handback.
    outer_B :: Int = 0
    outer_K :: Int = 5
end


##############
# Top-level model
##############

Base.@kwdef mutable struct Model
    common      :: CommonParams
    grids       :: CommonGrids

    unsk_par    :: UnskilledParams
    unsk_grids  :: UnskilledGrids
    unsk_cache  :: UnskilledCache

    skl_par     :: SkilledParams
    skl_grids   :: SkilledGrids
    skl_pre     :: SkilledPrecomp
    skl_cache   :: SkilledCache

    sim         :: SimParams
end


# ============================================================
# Default initialisation
# ============================================================

"""
    initialise_model(; Nx, Np_U, Np_S)

Construct a `Model` with default parameters.  `PU, bU, bT, α_U` are
held in the default `UnskilledParams` and `gamma_PS, bS, a_Γ, b_Γ` in
`SkilledParams`.
Returns an unsolved `Model` ready to be passed to `solve_model!`.
"""
function initialise_model(;
    Nx   :: Int = 200,
    Np_U :: Int = 200,
    Np_S :: Int = 200,
)
    # Common parameters (built before the grids so the worker-type grid can
    # be tuned to ℓ = Beta(a_ℓ, b_ℓ)).
    cp = CommonParams(
        r   = 0.05,
        ν   = 0.05,
        φ   = 0.20,
        a_ℓ = 2.0,
        b_ℓ = 5.0,
        c   = 1.70,
        A   = 0.0,   # log scale ⇒ effective scale exp(0) = 1
    )

    # Grids.  Worker-type grid uses Gauss–Jacobi tuned to ℓ = Beta(a_ℓ, b_ℓ)
    # (see build_type_grid); match-quality p-grids stay Gauss–Legendre.
    xgrid,   wx   = build_type_grid(Nx, cp.a_ℓ, cp.b_ℓ)
    pgrid_U, wp_U = build_gl_grid(Np_U)
    pgrid_S, wp_S = build_gl_grid(Np_S)

    ℓvals = pdf.(Beta(cp.a_ℓ, cp.b_ℓ), xgrid)

    grids   = CommonGrids(x = xgrid, wx = wx, ℓ = ℓvals)
    u_grids = UnskilledGrids(p = pgrid_U, wp = wp_U)
    s_grids = SkilledGrids(p = pgrid_S, wp = wp_S)

    # Unskilled parameters (now includes PU, bU, bT, α_U)
    up = UnskilledParams(
        μ   = 0.74,
        η   = 0.60,
        k   = 0.25,
        β   = 0.40,
        λ   = 0.08,
        PU  = 0.70,
        bU  = 0.00,
        bT  = 0.28,
        α_U = 1.00,
    )

    # Skilled parameters (now includes gamma_PS, bS, a_Γ, b_Γ)
    sp = SkilledParams(
        μ        = 0.90,
        η        = 0.50,
        k        = 0.17,
        β        = 0.32,
        λ        = 0.07,
        σ        = 0.01,
        gamma_PS = 1.85,
        bS       = 0.01,
        a_Γ      = 2.0,
        b_Γ      = 5.0,
        ξ        = 0.0,
    )

    # Simulation controls
    sim = SimParams(
        tol_inner      = 1e-8,
        tol_outer_U    = 1e-6,
        tol_outer_S    = 1e-7,
        tol_global     = 1e-3,

        maxit_inner    = 500,
        maxit_outer    = 300,
        maxit_global   = 50,

        conv_streak    = 2,

        use_anderson   = true,
        anderson_m     = 1,
        anderson_reg   = 1e-10,

        damp_pstar_U   = 1.00,
        damp_pstar_S   = 1.00,

        verbose        = 2,
        verbose_stride = 10,
    )

    # Skilled precomputations (reads a_Γ, b_Γ from sp)
    s_pre = build_skilled_precomp(s_grids, sp)

    # Initial conditions
    r = cp.r;  ν = cp.ν;  φ = cp.φ

    US_guess     = sp.bS * exp(cp.A) / (r + ν)
    T_init       = fill((up.bT * exp(cp.A) + φ * US_guess) / (r + φ + ν), Nx)
    Usearch_init = fill(up.bU * exp(cp.A) / (r + ν), Nx)
    U_init       = max.(Usearch_init, T_init)
    t_seed       = [(ν / (2ν + φ)) * ℓvals[ix] for ix in 1:Nx]

    uc = UnskilledCache(
        Usearch   = Usearch_init,
        U         = U_init,
        T         = T_init,
        Jfrontier = zeros(Nx),
        pstar     = fill(0.10, Nx),
        τT        = zeros(Nx),
        u         = 0.4 .* ℓvals,
        t         = t_seed,
        duS_carry = zeros(Nx),
        d_carry   = zeros(Nx),   # first pass unrestricted (d⁰ ≡ 0)
        θ         = 0.5,
    )

    US_init = fill(sp.bS * exp(cp.A) / (r + ν), Nx)

    sc = SkilledCache(
        U     = US_init,
        E0    = zeros(Nx, Np_S),
        E1    = zeros(Nx, Np_S),
        J0    = zeros(Nx, Np_S),
        J1    = zeros(Nx, Np_S),
        pstar = fill(0.10, Nx),
        poj   = fill(0.60, Nx),
        d     = zeros(Nx),
        u     = zeros(Nx),
        e     = zeros(Nx, Np_S),
        θ     = 0.5,
    )

    return Model(
        common     = cp,
        grids      = grids,
        unsk_par   = up,
        unsk_grids = u_grids,
        unsk_cache = uc,
        skl_par    = sp,
        skl_grids  = s_grids,
        skl_pre    = s_pre,
        skl_cache  = sc,
        sim        = sim,
    )
end

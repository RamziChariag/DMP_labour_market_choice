############################################################
# params.jl — Structs and default parameter initialisation
############################################################

# packages loaded by main.jl

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

    c   :: Float64    # training cost coefficient in c(x) = c·exp(−6x)
end


Base.@kwdef struct RegimeParams
    PU  :: Float64    # unskilled sector productivity shifter
    PS  :: Float64    # skilled sector productivity shifter

    bU  :: Float64    # flow payoff in unskilled unemployment
    bT  :: Float64    # flow payoff while in training
    bS  :: Float64    # flow payoff in skilled unemployment

    α_U :: Float64    # unskilled damage-shock Beta shape (Beta(α_U, 1))

    a_Γ :: Float64    # skilled match-quality Beta shape 1
    b_Γ :: Float64    # skilled match-quality Beta shape 2
end


Base.@kwdef struct CommonGrids
    x   :: Vector{Float64}    # worker-type grid
    wx  :: Vector{Float64}    # quadrature weights on x-grid
    ℓ   :: Vector{Float64}    # worker-type density evaluated on x-grid
end


##############
# Unskilled
##############

Base.@kwdef struct UnskilledParams
    μ   :: Float64    # matching efficiency
    η   :: Float64    # matching elasticity
    k   :: Float64    # vacancy posting cost
    β   :: Float64    # worker Nash bargaining weight
    λ   :: Float64    # damage-shock arrival rate
end


Base.@kwdef struct UnskilledGrids
    p   :: Vector{Float64}    # unskilled match-quality grid
    wp  :: Vector{Float64}    # quadrature weights on unskilled p-grid
end


Base.@kwdef mutable struct UnskilledCache
    # values
    Usearch   :: Vector{Float64}    # value of unskilled search
    U         :: Vector{Float64}    # value of untrained unemployment
    T         :: Vector{Float64}    # value of training

    Jfrontier :: Vector{Float64}    # firm value at frontier p = 1
    pstar     :: Vector{Float64}    # unskilled reservation quality cutoff
    τT        :: Vector{Float64}    # training policy indicator

    # composition
    u         :: Vector{Float64}    # density of untrained unemployed
    t         :: Vector{Float64}    # density of workers in training

    # scalar
    θ         :: Float64            # unskilled market tightness
end


##############
# Skilled
##############

Base.@kwdef struct SkilledParams
    μ   :: Float64    # matching efficiency
    η   :: Float64    # matching elasticity
    k   :: Float64    # vacancy posting cost
    β   :: Float64    # worker Nash bargaining weight

    ξ   :: Float64    # exogenous separation rate
    λ   :: Float64    # skilled quality-shock arrival rate
    σ   :: Float64    # flow cost of on-the-job search
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
    U     :: Vector{Float64}           # value of skilled unemployment

    E0    :: Matrix{Float64}           # worker value, employed, no OJS
    E1    :: Matrix{Float64}           # worker value, employed, with OJS
    J0    :: Matrix{Float64}           # firm value, filled job, no OJS
    J1    :: Matrix{Float64}           # firm value, filled job, with OJS

    pstar :: Vector{Float64}           # skilled separation cutoff
    poj   :: Vector{Float64}           # skilled OJS cutoff

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
end


##############
# Top-level model
##############

Base.@kwdef mutable struct Model
    common      :: CommonParams
    regime      :: RegimeParams
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
    initialise_model(; Nx, Np_U, Np_S, regime)

Construct a `Model` with default parameters and a given regime.
Returns an unsolved `Model` ready to be passed to `solve_model!`.
"""
function initialise_model(;
    Nx   :: Int = 200,
    Np_U :: Int = 200,
    Np_S :: Int = 200,
    regime :: RegimeParams = RegimeParams(
        PU  = 0.70,
        PS  = 1.85,
        bU  = 0.00,
        bT  = 0.28,
        bS  = 0.01,
        α_U = 1.00,
        a_Γ = 2.0,
        b_Γ = 5.0,
    )
)
    # ── Grids ──────────────────────────────────────────────────────────────
    xgrid,   wx   = build_gl_grid(Nx)
    pgrid_U, wp_U = build_gl_grid(Np_U)
    pgrid_S, wp_S = build_gl_grid(Np_S)

    # ── Common parameters ─────────────────────────────────────────────────
    cp = CommonParams(
        r   = 0.05,
        ν   = 0.05,
        φ   = 0.20,
        a_ℓ = 2.0,
        b_ℓ = 5.0,
        c   = 1.70,
    )

    ℓvals = pdf.(Beta(cp.a_ℓ, cp.b_ℓ), xgrid)

    grids   = CommonGrids(x = xgrid, wx = wx, ℓ = ℓvals)
    u_grids = UnskilledGrids(p = pgrid_U, wp = wp_U)
    s_grids = SkilledGrids(p = pgrid_S, wp = wp_S)

    # ── Unskilled parameters ──────────────────────────────────────────────
    up = UnskilledParams(
        μ = 0.74,
        η = 0.60,
        k = 0.25,
        β = 0.40,
        λ = 0.08,
    )

    # ── Skilled parameters ────────────────────────────────────────────────
    sp = SkilledParams(
        μ = 0.90,
        η = 0.50,
        k = 0.17,
        β = 0.32,
        ξ = 0.03,
        λ = 0.07,
        σ = 0.01,
    )

    # ── Simulation controls ───────────────────────────────────────────────
    sim = SimParams(
        tol_inner      = 1e-8,
        tol_outer_U    = 1e-6,
        tol_outer_S    = 1e-5,
        tol_global     = 1e-3,

        maxit_inner    = 500,
        maxit_outer    = 300,
        maxit_global   = 50,

        conv_streak    = 2,

        use_anderson   = true,
        anderson_m     = 1,
        anderson_reg   = 1e-10,

        damp_pstar_U   = 1.30,
        damp_pstar_S   = 0.02,

        verbose        = 2,
        verbose_stride = 10,
    )

    # ── Skilled precomputations ───────────────────────────────────────────
    s_pre = build_skilled_precomp(s_grids, regime)

    # ── Initial conditions ────────────────────────────────────────────────
    r = cp.r;  ν = cp.ν;  φ = cp.φ

    US_guess     = regime.bS / (r + ν)
    T_init       = fill((regime.bT + φ * US_guess) / (r + φ + ν), Nx)
    Usearch_init = fill(regime.bU / (r + ν), Nx)
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
        θ         = 1.0,
    )

    US_init = fill(regime.bS / (r + ν), Nx)

    sc = SkilledCache(
        U     = US_init,
        E0    = zeros(Nx, Np_S),
        E1    = zeros(Nx, Np_S),
        J0    = zeros(Nx, Np_S),
        J1    = zeros(Nx, Np_S),
        pstar = fill(0.10, Nx),
        poj   = fill(0.60, Nx),
        u     = zeros(Nx),
        e     = zeros(Nx, Np_S),
        θ     = 1.0,
    )

    return Model(
        common     = cp,
        regime     = regime,
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

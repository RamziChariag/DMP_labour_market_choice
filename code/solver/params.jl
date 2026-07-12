############################################################
# params.jl — Structs and default parameter initialisation (RoySearch)
#
# Worker type is a PAIR of abilities (aU, aS): aptitude for unskilled and
# for skilled work, drawn from a joint distribution with Gaussian-copula
# correlation ρ_x.  Market separation is by comparative advantage (Roy
# 1951; Gola 2016, 2021), not by curvature of production, so the sectoral
# gradients γ_U, γ_S of the single-ability model are gone and a single
# copula correlation ρ_x is added.  The single-ability model is the
# ρ_x → 1 limit.
#
# Each parameter is stored in the block whose solver code reads it:
#   :common ← r, ν, φ, a_ℓ, b_ℓ, ρ_x, c, A
#   :unsk   ← μ_U, η_U, k_U, β_U, λ_U, PU, bU, bT, α_U, σ_wU
#   :skl    ← μ_S, η_S, k_S, β_S, λ_S, σ_S, PS, bS, a_Γ, b_Γ, ξ_S, σ_wS
# Storage reflects only which solver block consumes a parameter;
# the four estimation categories (calibrated / pre-estimated / deep /
# regime-specific) are an orthogonal classification (see the notes).
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

    a_ℓ :: Float64    # Beta shape 1 of each ability marginal ℓ (symmetric baseline)
    b_ℓ :: Float64    # Beta shape 2 of each ability marginal ℓ (symmetric baseline)
    ρ_x :: Float64    # skill-copula correlation on the Gaussian ranks; ρ_x → 1 ⇒ single-ability model

    c   :: Float64    # Training cost coefficient — a plain, UNSCALED linear multiplier
                      # (c = 0 removes the cost).  The cost FUNCTION carries exp(A):
                      # c(aS) = exp(A)·c·(1−aS)·exp(−aS), so the exp(A) is applied in
                      # training_cost, not stored in c.  This puts the cost on the same
                      # scale as T(aS) and U^search(aU) in the training margin, so c is
                      # identified; a fixed-unit cost is dwarfed at the estimated A.

    A   :: Float64 = 0.0   # LOG aggregate production scale; the model uses exp(A) as the effective scale (multiplies π_U, π_S, b·, k·, σ_S). A = 0 ⇒ effective scale 1.
end


Base.@kwdef struct CommonGrids
    x   :: Vector{Float64}    # shared ability grid (Gauss–Jacobi nodes for Beta(a_ℓ,b_ℓ))
    wa_U :: Vector{Float64}   # aU marginal population mass weights: sum = 1, dot(g, wa_U) = E_ℓ[g]
    wa_S :: Vector{Float64}   # aS marginal population mass weights: sum = 1, dot(g, wa_S) = E_ℓ[g]
    copula :: CopulaGrid      # joint aggregation weights W2[i,j] (rebuilt when a_ℓ,b_ℓ,ρ_x change)
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

    PU  :: Float64    # unskilled sector productivity shifter P_U: π_U(aU,p) = A·P_U·aU·p
    bU  :: Float64    # flow payoff in unskilled unemployment
    bT  :: Float64    # flow payoff while in training
    α_U :: Float64    # unskilled damage-shock Beta shape (p' ∼ Beta(α_U, 1), G(p) = p^{α_U})

    σ_w :: Float64 = 0.0   # log-wage measurement-error SD (σ_wU); 0 ⇒ no measurement error
end


Base.@kwdef struct UnskilledGrids
    p   :: Vector{Float64}    # unskilled match-quality grid
    wp  :: Vector{Float64}    # quadrature weights on unskilled p-grid
end


Base.@kwdef mutable struct UnskilledCache
    # Values — one-dimensional in own ability (notes §231, §276)
    Usearch   :: Vector{Float64}    # U^search(aU): value of unskilled search
    T         :: Vector{Float64}    # T(aS): value of training
    Jfrontier :: Vector{Float64}    # J_U(aU, 1): firm value at frontier p = 1
    pstar     :: Vector{Float64}    # p*(aU): unskilled reservation quality cutoff

    # Two-dimensional policy and composition on the (aU, aS) copula grid
    τT        :: Matrix{Float64}    # τ(aU,aS): training policy indicator
    u         :: Matrix{Float64}    # u_U(aU,aS): density of untrained unemployed
    t         :: Matrix{Float64}    # t(aU,aS): density of workers in training

    # Cross-market contribution carried from the skilled block:
    # duS_carry(aU,aS) = d(aU,aS) · u_S(aU,aS).  Held constant within one
    # unskilled block solve; refreshed by the global outer loop after each
    # skilled solve, and used in the augmented free-entry condition.
    duS_carry :: Matrix{Float64}

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

    PS  :: Float64    # skilled productivity LEVEL P_S: π_S(aS,p) = A·P_S·aS·p
    bS  :: Float64    # flow payoff in skilled unemployment
    a_Γ :: Float64    # skilled match-quality Beta shape 1
    b_Γ :: Float64    # skilled match-quality Beta shape 2

    ξ   :: Float64 = 0.0   # exogenous skilled separation hazard ξ_S (0 ⇒ recovers the no-ξ model)
    σ_w :: Float64 = 0.0   # log-wage measurement-error SD (σ_wS); 0 ⇒ no measurement error
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
    # Values — one-dimensional in the SKILLED ability aS (notes §407–444)
    U     :: Vector{Float64}           # U_S(aS) = max{U_S^(0)(aS), U_S^(1)(aU)} folded to the aS-grid via d
    U0    :: Vector{Float64}           # U_S^(0)(aS): stay-skilled candidate
    U1    :: Vector{Float64}           # U_S^(1)(aU): cross-to-unskilled candidate (indexed by aU)

    E0    :: Matrix{Float64}           # E_S^0(aS,p): worker value, employed, no OJS
    E1    :: Matrix{Float64}           # E_S^1(aS,p): worker value, employed, with OJS
    J0    :: Matrix{Float64}           # J_S^0(aS,p): firm value, filled job, no OJS
    J1    :: Matrix{Float64}           # J_S^1(aS,p): firm value, filled job, with OJS

    pstar :: Vector{Float64}           # p*_S(aS): endogenous separation cutoff
    poj   :: Vector{Float64}           # p^oj_S(aS): OJS cutoff

    # Cross-market directed-search policy on the (aU, aS) grid:
    # d(aU,aS) = 1 ⇔ U_S^(1)(aU) > U_S^(0)(aS)  (notes eq:dpolicy).
    d     :: Matrix{Float64}

    # Stationary trained composition.  Per-aS UNIT shapes (total mass 1 for
    # a d = 0 type) — the p-dynamics read only aS, so the aU-dependence is
    # pure scaling by the 2D trained mass m_S (notes §386, §472–483):
    u_frac :: Vector{Float64}          # û_S(aS): unemployed fraction of a d=0 type's mass
    e_frac :: Matrix{Float64}          # ê_S(aS,p): employed density fraction of a d=0 type's mass

    m_S   :: Matrix{Float64}           # m_S(aU,aS): trained-worker mass = φ t / (ν + d f_U)

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
# Grid + cache construction from a parameter set
#
# Grids depend on (a_ℓ, b_ℓ, ρ_x), so they are (re)built for each
# parameter vector rather than once.  build_common_grids assembles the
# shared ability grid, marginal mass weights, and the copula; the block
# caches are seeded to the analytic no-search guesses.
# ============================================================

"""
    build_common_grids(cp, Nx) -> CommonGrids

Shared ability grid, marginal mass weights, and the skill copula for the
common parameters `cp`.  Symmetric-marginals baseline: both abilities use
the same grid and weights.
"""
function build_common_grids(cp::CommonParams, Nx::Int)
    x, wa = build_ability_grid(Nx, cp.a_ℓ, cp.b_ℓ)
    cop   = build_copula(cp.ρ_x, x, wa, wa, cp.a_ℓ, cp.b_ℓ)
    return CommonGrids(x = x, wa_U = wa, wa_S = copy(wa), copula = cop)
end


# ============================================================
# Default initialisation
# ============================================================

"""
    initialise_model(; Nx, Np_U, Np_S)

Construct a `Model` with default parameters on `Nx` ability nodes and
`Np_U`, `Np_S` match-quality nodes.  Returns an unsolved `Model` ready
for `solve_model!`.
"""
function initialise_model(;
    Nx   :: Int = 200,
    Np_U :: Int = 200,
    Np_S :: Int = 200,
)
    cp = CommonParams(
        r   = 0.05,
        ν   = 0.05,
        φ   = 0.20,
        a_ℓ = 2.0,
        b_ℓ = 5.0,
        ρ_x = -0.55,   # negative concordance keeps both markets populated (Gola mechanism)
        c   = 1.70,
        A   = 0.0,     # log scale ⇒ effective scale exp(0) = 1
    )

    grids   = build_common_grids(cp, Nx)
    pU, wpU = build_gl_grid(Np_U);  u_grids = UnskilledGrids(p = pU, wp = wpU)
    pS, wpS = build_gl_grid(Np_S);  s_grids = SkilledGrids(p = pS, wp = wpS)

    up = UnskilledParams(
        μ = 0.74, η = 0.60, k = 0.25, β = 0.40, λ = 0.08,
        PU = 0.70, bU = 0.00, bT = 0.28, α_U = 1.00,
    )

    sp = SkilledParams(
        μ = 0.90, η = 0.50, k = 0.17, β = 0.32, λ = 0.07, σ = 0.01,
        PS = 1.85, bS = 0.01, a_Γ = 2.0, b_Γ = 5.0, ξ = 0.0,
    )

    sim = SimParams(
        tol_inner    = 1e-8, tol_outer_U = 1e-6, tol_outer_S = 1e-7, tol_global = 1e-3,
        maxit_inner  = 500,  maxit_outer = 300,  maxit_global = 50,
        conv_streak  = 2,
        use_anderson = true, anderson_m = 1, anderson_reg = 1e-10,
        damp_pstar_U = 1.00, damp_pstar_S = 1.00,
        verbose      = 2,    verbose_stride = 10,
    )

    return build_model(cp, grids, up, u_grids, sp, s_grids, sim; Nx, Np_S)
end


"""
    build_model(cp, grids, up, u_grids, sp, s_grids, sim; Nx, Np_S) -> Model

Assemble a `Model` from parameter/grid blocks with freshly seeded caches.
Shared by `initialise_model` and the SMM entry point so both construct
identical initial conditions.
"""
function build_model(cp::CommonParams, grids::CommonGrids,
                     up::UnskilledParams, u_grids::UnskilledGrids,
                     sp::SkilledParams, s_grids::SkilledGrids,
                     sim::SimParams; Nx::Int, Np_S::Int)
    s_pre = build_skilled_precomp(s_grids, sp)

    r = cp.r;  ν = cp.ν;  φ = cp.φ
    US_flat  = sp.bS * exp(cp.A) / (r + ν)
    T_init   = fill((up.bT * exp(cp.A) + φ * US_flat) / (r + φ + ν), Nx)
    Us_init  = fill(up.bU * exp(cp.A) / (r + ν), Nx)
    ℓrow     = grids.wa_U                       # marginal mass in aU (for unemployment seed)

    uc = UnskilledCache(
        Usearch   = Us_init,
        T         = T_init,
        Jfrontier = zeros(Nx),
        pstar     = fill(0.10, Nx),
        τT        = zeros(Nx, Nx),
        u         = 0.4 .* grids.copula.W2,
        t         = (ν / (2ν + φ)) .* grids.copula.W2,
        duS_carry = zeros(Nx, Nx),
        θ         = 0.5,
    )

    sc = SkilledCache(
        U      = fill(US_flat, Nx),
        U0     = fill(US_flat, Nx),
        U1     = fill(US_flat, Nx),
        E0     = zeros(Nx, Np_S),
        E1     = zeros(Nx, Np_S),
        J0     = zeros(Nx, Np_S),
        J1     = zeros(Nx, Np_S),
        pstar  = fill(0.10, Nx),
        poj    = fill(0.60, Nx),
        d      = zeros(Nx, Nx),
        u_frac = fill(1.0, Nx),
        e_frac = zeros(Nx, Np_S),
        m_S    = zeros(Nx, Nx),
        θ      = 0.5,
    )

    return Model(
        common = cp, grids = grids,
        unsk_par = up, unsk_grids = u_grids, unsk_cache = uc,
        skl_par = sp, skl_grids = s_grids, skl_pre = s_pre, skl_cache = sc,
        sim = sim,
    )
end

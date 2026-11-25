############################################################
# params.jl — Parameter, grid, cache and model structs
#
# A worker type is a pair of aptitudes (aU, aS) with Beta(a_ℓ, b_ℓ) marginals joined by
# a Gaussian copula with correlation ρ_x; ρ_x → 1 is the single-ability model.
# Parameters are stored in the block whose code reads them:
#   :common ← r, ν, φ, a_ℓ, b_ℓ, ρ_x, c, A
#   :unsk   ← μ_U, η_U, k_U, β_U, λ_U, PU, bU, bT, α_U, ξ_U, σ_wU
#   :skl    ← μ_S, η_S, k_S, β_S, λ_S, σ_S, PS, bS, a_Γ, b_Γ, ξ_S, σ_wS
# Stored units: A = ln A, flows (bU, bS, bT, σ) per unit of A, c = ln c, k_j in months of
# the mean output of a new match (mean_output_U/S, grids.jl).
#
# Entry point: build_model.
############################################################


# ============================================================
# Solver knobs held outside SimParams
#
# Module Refs rather than SimParams fields: structs are serialised by field position, so a
# new field would make every stored bundle unreadable. Documented in SETTINGS.md.
# ============================================================

# Under-relaxation weight w on skilled free entry, θ_S ← w θ_new + (1−w) θ_old, and the
# ceiling of the adaptive weight in solve_skilled_block!. The raw map overshoots.
const DAMP_THETA_S = Ref(0.9)

# The same on unskilled free entry: the raw θ_U map is not a contraction (it orbits between
# the clamps 1e-14 and 100). Under-relaxation leaves the fixed point unchanged.
const DAMP_THETA_U = Ref(0.9)

# Which zero of S¹ − S⁰ above p*_S is p^oj_S: :last (the upper boundary of the searching
# region, which every consumer assumes) or :first (reproduces runs before v38.3.0; it
# collapses p^oj onto p*_S when the difference dips negative at the bottom of the grid).
const POJ_CROSSING = Ref(:last)

# Surpluses the p^oj scan scores: :raw, or :smooth (the smooth_pos forms; same sign, hence
# same crossing cell). The p*_S scan is raw under both.
const POJ_SCORE = Ref(:raw)

# Crossing inside the bracketing cell: linear (false) or the local quadratic through three
# nodes (true), which reduces the kink when the bracket moves from O(h) to O(h²).
const POJ_CROSSING_C1 = Ref(false)

# Type-level OJS participation over an ability cell: :fraction, the share of the cell on
# which M(aS) = max_{p ≥ p*_S} (S¹ − S⁰) > 0, or :indicator, the node rule 1{M(a_k) > 0}
# (reproduces runs up to v38.3.0; a knife-edge type then switches its whole margin each
# iteration and the skilled block cycles).
const OJS_PARTICIPATION = Ref(:fraction)


# ============================================================
# Common
# ============================================================

"""
    CommonParams

Parameters shared by both markets: discount rate `r`, demographic exit rate `ν`, training
completion rate `φ`, Beta shapes `a_ℓ`, `b_ℓ` of each ability marginal, copula correlation
`ρ_x`, log training-cost coefficient `c` (cost (1−aS) exp(c−aS), not scaled by A) and log
production scale `A` (the model uses exp(A)).
"""
Base.@kwdef struct CommonParams
    r   :: Float64
    ν   :: Float64
    φ   :: Float64

    a_ℓ :: Float64
    b_ℓ :: Float64
    ρ_x :: Float64

    c   :: Float64

    A   :: Float64 = 0.0
end


"""
    CommonGrids

The shared ability grid `x` (Gauss–Jacobi nodes for Beta(a_ℓ, b_ℓ)), the marginal mass
weights `wa_U`, `wa_S` (each sums to 1) and the copula weights `copula`.
"""
Base.@kwdef struct CommonGrids
    x   :: Vector{Float64}
    wa_U :: Vector{Float64}
    wa_S :: Vector{Float64}
    copula :: CopulaGrid
end


# ============================================================
# Unskilled
# ============================================================

"""
    UnskilledParams

Unskilled-market parameters: matching efficiency `μ` and elasticity `η`, vacancy cost `k`
(months of mean new-match output, dollar flow k π̄_U), worker bargaining weight `β`,
damage-shock rate `λ`, productivity `PU` (π_U = exp(A) P_U aU p), unemployment flow `bU`
and training flow `bT` (per unit of A), damage-shock shape `α_U` (G(p) = p^α_U),
exogenous separation hazard `ξ` and log-wage measurement-error SD `σ_w`.
"""
Base.@kwdef struct UnskilledParams
    μ   :: Float64
    η   :: Float64
    k   :: Float64
    β   :: Float64
    λ   :: Float64

    PU  :: Float64
    bU  :: Float64
    bT  :: Float64
    α_U :: Float64

    ξ   :: Float64 = 0.0
    σ_w :: Float64 = 0.0
end


"""
    UnskilledGrids

Unskilled match-quality grid `p` and its quadrature weights `wp`.
"""
Base.@kwdef struct UnskilledGrids
    p   :: Vector{Float64}
    wp  :: Vector{Float64}
end


"""
    UnskilledCache

Mutable state of the unskilled block.
"""
Base.@kwdef mutable struct UnskilledCache
    # Values of a searching type, one-dimensional in aU
    Usearch   :: Vector{Float64}    # U^search(aU)
    Jfrontier :: Vector{Float64}    # J_U(aU, 1)
    pstar     :: Vector{Float64}    # p*_U(aU)

    # Training value and the training slice: the unskilled matches of trained hires whose
    # outside option is training (career_choices!)
    T         :: Matrix{Float64}    # T(aU,aS)
    pstar_t   :: Matrix{Float64}    # p*_t(aU,aS)
    S1_t      :: Matrix{Float64}    # S_t(aU,aS,1) before the positive part
    χ_t       :: Matrix{Float64}    # hired share of their meetings

    # Policy and composition on the (aU, aS) grid
    τT        :: Matrix{Float64}    # training fraction of each cell
    u         :: Matrix{Float64}    # untrained unemployed
    uh        :: Matrix{Float64}    # hired branch of u
    n         :: Matrix{Float64}    # in training
    et        :: Matrix{Float64}    # employment of the training slice

    # d·u_S from the skilled block; fixed within an unskilled solve, refreshed by the global loop
    duS_carry :: Matrix{Float64}

    θ         :: Float64            # θ_U
end


# ============================================================
# Skilled
# ============================================================

"""
    SkilledParams

Skilled-market parameters: matching efficiency `μ` and elasticity `η`, vacancy cost `k`
(months of mean new-match output, dollar flow k π̄_S), worker bargaining weight `β`,
quality-shock rate `λ`, on-the-job search flow cost `σ` (per unit of A), productivity `PS`
(π_S = exp(A) P_S aS p), unemployment flow `bS` (per unit of A), offer distribution
Beta(`a_Γ`, `b_Γ`), shock support ratio `δ` (a shocked match redraws δ × an offer draw;
δ = 1: redraw ≡ fresh offer), exogenous separation hazard `ξ` and log-wage
measurement-error SD `σ_w`.
"""
Base.@kwdef struct SkilledParams
    μ   :: Float64
    η   :: Float64
    k   :: Float64
    β   :: Float64

    λ   :: Float64
    σ   :: Float64

    PS  :: Float64
    bS  :: Float64
    a_Γ :: Float64
    b_Γ :: Float64

    δ   :: Float64 = 1.0

    ξ   :: Float64 = 0.0
    σ_w :: Float64 = 0.0
end


"""
    SkilledGrids

Skilled match-quality grid `p` and its quadrature weights `wp`.
"""
Base.@kwdef struct SkilledGrids
    p   :: Vector{Float64}
    wp  :: Vector{Float64}
end


"""
    SkilledPrecomp

Quality-distribution quadrature on the skilled grid, for the offer distribution
Γ_o = Beta(a_Γ, b_Γ) (fresh meetings) and the shock distribution Γ_s(x) = Γ_o(x/δ) (λ_S
redraws): CDF at the nodes (`Γvals`, `Γs_vals`), cell mass per unit `wp` (`γvals`,
`γs_vals`, from `build_cell_mass_density`; never read pointwise) and the suffix sums of
`γ·wp` (`tail_weights`, `tail_weights_s`).
"""
Base.@kwdef struct SkilledPrecomp
    Γvals        :: Vector{Float64}
    γvals        :: Vector{Float64}
    tail_weights :: Vector{Float64}

    Γs_vals        :: Vector{Float64}
    γs_vals        :: Vector{Float64}
    tail_weights_s :: Vector{Float64}
end


"""
    SkilledCache

Mutable state of the skilled block. Values are one-dimensional in aS.
"""
Base.@kwdef mutable struct SkilledCache
    U     :: Vector{Float64}           # U_S^(0)(aS) as the global link variable
    U0    :: Vector{Float64}           # U_S^(0)(aS), inner iterate
    U1    :: Matrix{Float64}           # U_S^(1)(aU,aS), crossing to unskilled (career_choices!)

    E0    :: Matrix{Float64}           # E_S^0(aS,p), employed, no OJS
    E1    :: Matrix{Float64}           # E_S^1(aS,p), employed, OJS
    J0    :: Matrix{Float64}           # J_S^0(aS,p)
    J1    :: Matrix{Float64}           # J_S^1(aS,p)

    pstar :: Vector{Float64}           # p*_S(aS)
    poj   :: Vector{Float64}           # p^oj_S(aS)

    # OJS participation: share of each ability cell on which OJS pays (1 under :indicator)
    s_part :: Vector{Float64}

    d     :: Matrix{Float64}           # drain fraction of each (aU,aS) cell

    # Per-aS unit shapes of a non-draining type's trained mass (scaled by m_S in aggregation)
    u_frac :: Vector{Float64}          # unemployed share
    e_frac :: Matrix{Float64}          # employed density in p

    m_S   :: Matrix{Float64}           # trained mass m_S(aU,aS)

    θ     :: Float64                   # θ_S
end


# ============================================================
# Simulation controls
# ============================================================

"""
    SimParams

Solver controls. Field order is part of the bundle format (positional serialisation).

- `tol_inner`, `tol_outer_U`, `tol_outer_S`: step tolerances of the inner loops and the
  two block outer loops. The global loop has none: it gates on the residual battery.
- `maxit_inner`, `maxit_outer`, `maxit_global`: iteration caps.
- `conv_streak`: consecutive passes inside tolerance required to stop.
- `use_anderson`: Anderson(1) in the unskilled block and the global loop (the skilled block
  is never accelerated). `anderson_reg` is not read.
- `damp_pstar_U`: p*_U damping, read only when `use_anderson = false`; `damp_pstar_S`:
  damping of every skilled cutoff install.
- `verbose`, `verbose_stride`: print level and interval.
- `damp_inner_U`, `damp_inner_S`: under-relaxation of the inner value iteration (1 = none).
- `inner_B`, `inner_K`: after `inner_B` sweeps an inner loop that has not contracted over
  `inner_K` sweeps stops as diverged; `inner_K` diverged inner solves in a row reject the
  parameter. `inner_B = 0` disables both.
- `outer_B`, `outer_K`: the same test on a block outer loop; a stall hands control back to
  the global loop without rejecting. `outer_B = 0` disables it.
- `global_B`, `global_K`: global stall-out when the residual has not beaten its best over
  `global_K` passes after `global_B`, stopping as non-converged. `global_B = 0` disables it.
"""
Base.@kwdef struct SimParams
    tol_inner      :: Float64
    tol_outer_U    :: Float64
    tol_outer_S    :: Float64

    maxit_inner    :: Int
    maxit_outer    :: Int
    maxit_global   :: Int

    conv_streak    :: Int

    use_anderson   :: Bool
    anderson_reg   :: Float64

    damp_pstar_U   :: Float64
    damp_pstar_S   :: Float64

    verbose        :: Int
    verbose_stride :: Int

    damp_inner_U   :: Float64 = 1.0
    damp_inner_S   :: Float64 = 1.0

    inner_B :: Int = 0
    inner_K :: Int = 5

    outer_B :: Int = 0
    outer_K :: Int = 5

    global_B :: Int = 0
    global_K :: Int = 5
end


# ============================================================
# Held education choices
# ============================================================

"""
    HeldChoices

The training choice and the choice of market of a stationary equilibrium, held by a model
that carries them (`Model.held`): wherever the model maximises over training or over the
market it uses these instead.
  x       the ability grid they were solved on
  τ       training fraction of each (aU, aS) cell (training_policy!)
  d       drain fraction of each cell (drain_fraction!)
  trains  node choice in U_U = max{U^search, −c + T}: true where training is chosen
  drains  node choice in U_S = max{U_S^(0), U_S^(1)}: true where U_S^(1) is chosen
Built from a solved model by `HeldChoices(model)` (unskilled.jl).
"""
struct HeldChoices
    x      :: Vector{Float64}
    τ      :: Matrix{Float64}
    d      :: Matrix{Float64}
    trains :: Matrix{Bool}
    drains :: Matrix{Bool}
end


# ============================================================
# Top-level model
# ============================================================

"""
    Model

Parameters, grids and solver state of both markets. `corner_limit` is set by the solver once
a solve has converged with a market holding no one: free entry there is then priced at the
market's limit composition (`marginal_training_cells`). `held` carries the education choices
held fixed (`HeldChoices`), or `nothing` when the model chooses.
"""
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

    corner_limit :: Bool = false

    held :: Union{Nothing,HeldChoices} = nothing
end


# ============================================================
# Construction
# ============================================================

"""
    build_common_grids(cp, Nx) → CommonGrids

Ability grid, marginal mass weights and skill copula for `cp` on `Nx` nodes. Both
aptitudes share the grid and weights (symmetric marginals).
"""
function build_common_grids(cp::CommonParams, Nx::Int)
    x, wa = build_ability_grid(Nx, cp.a_ℓ, cp.b_ℓ)
    cop   = build_copula(cp.ρ_x, x, wa, wa, cp.a_ℓ, cp.b_ℓ)
    return CommonGrids(x = x, wa_U = wa, wa_S = copy(wa), copula = cop)
end


"""
    build_model(cp, grids, up, u_grids, sp, s_grids, sim; Nx, Np_S, held = nothing) → Model

Assemble an unsolved `Model` from its blocks, with the caches seeded at fixed no-search
guesses. Every solve starts from these seeds, so they are part of which equilibrium a solve
finds. `held` (`HeldChoices`) must have been solved on `grids.x`.
"""
function build_model(cp::CommonParams, grids::CommonGrids,
                     up::UnskilledParams, u_grids::UnskilledGrids,
                     sp::SkilledParams, s_grids::SkilledGrids,
                     sim::SimParams; Nx::Int, Np_S::Int,
                     held::Union{Nothing,HeldChoices} = nothing)
    held === nothing || held.x == grids.x ||
        error("build_model: the held choices were solved on another ability grid")
    s_pre = build_skilled_precomp(s_grids, sp)

    r = cp.r;  ν = cp.ν;  φ = cp.φ
    US_flat  = sp.bS * exp(cp.A) / (r + ν)
    T_init   = fill((up.bT * exp(cp.A) + φ * US_flat) / (r + φ + ν), Nx, Nx)
    Us_init  = fill(up.bU * exp(cp.A) / (r + ν), Nx)

    uc = UnskilledCache(
        Usearch   = Us_init,
        Jfrontier = zeros(Nx),
        pstar     = fill(0.10, Nx),
        T         = T_init,
        pstar_t   = ones(Nx, Nx),
        S1_t      = zeros(Nx, Nx),
        χ_t       = zeros(Nx, Nx),
        τT        = zeros(Nx, Nx),
        u         = 0.4 .* grids.copula.W2,
        uh        = zeros(Nx, Nx),
        n         = (ν / (2ν + φ)) .* grids.copula.W2,
        et        = zeros(Nx, Nx),
        duS_carry = zeros(Nx, Nx),
        θ         = 0.5,
    )

    sc = SkilledCache(
        U      = fill(US_flat, Nx),
        U0     = fill(US_flat, Nx),
        U1     = fill(US_flat, Nx, Nx),
        E0     = zeros(Nx, Np_S),
        E1     = zeros(Nx, Np_S),
        J0     = zeros(Nx, Np_S),
        J1     = zeros(Nx, Np_S),
        pstar  = fill(0.10, Nx),
        poj    = fill(0.60, Nx),
        s_part = ones(Nx),
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
        sim = sim, held = held,
    )
end

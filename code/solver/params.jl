############################################################
# params.jl — Structs and default parameter initialisation (RoySearch)
#
# Worker type is a PAIR of abilities (aU, aS): aptitude for unskilled and
# for skilled work, drawn from a joint distribution with Gaussian-copula
# correlation ρ_x.  Both markets are populated through comparative advantage
# (Roy 1951; concordance as in Gola 2021), not through curvature of production, so the sectoral
# gradients γ_U, γ_S of the single-ability model are gone and a single
# copula correlation ρ_x is added.  The single-ability model is the
# ρ_x → 1 limit.
#
# Each parameter is stored in the block whose solver code reads it:
#   :common ← r, ν, φ, a_ℓ, b_ℓ, ρ_x, c, A
#   :unsk   ← μ_U, η_U, k_U, β_U, λ_U, PU, bU, bT, α_U, ξ_U, σ_wU
#   :skl    ← μ_S, η_S, k_S, β_S, λ_S, σ_S, PS, bS, a_Γ, b_Γ, ξ_S, σ_wS
############################################################


# ============================================================
# Solver relaxation knobs held outside SimParams
#
# Free-entry under-relaxation on θ_S: the skilled tightness is updated to
# w·θ_new + (1−w)·θ_old inside solve_skilled_block!.  Damping the cutoffs
# alone (damp_pstar_S) leaves the θ_S update raw, and the raw free-entry map
# overshoots enough to lose otherwise feasible parameter draws: at w = 0.9 it
# recovers 15 of the 25 residual failures, loses none of 96 feasible controls,
# and moves Q by at most 6.5e-07 relative over those controls (1.2e-10 at
# θ̂ itself).  w = 0.9 is LMR's own value (params.f90:275).
#
# This is a module-level Ref rather than a SimParams field on purpose: Julia's
# serialiser reads structs positionally by field count, so adding a field would
# make every SMMResult bundle already on disk unreadable.
# ============================================================

const DAMP_THETA_S = Ref(0.9)

# The same under-relaxation on θ_U, and for the same reason one level down: the
# unskilled free-entry map was iterated RAW, and raw it is not a contraction.
# Measured on the a_ℓ line at base_fc, where it was losing a contiguous band of
# otherwise-feasible draws: θ_U ran an exact period-8 orbit, descending from the
# clamp ceiling of 100 to the floor of 1e-14 over seven iterations and jumping back
# on the eighth, because θ_U at the floor gives κ_U ≈ 0, an unfilled vacancy is then
# worth everything, and free entry answers with the ceiling.  At w = 0.9 all nine
# swept points solve against four shipped, and Q at the points that already solved
# moves by at most 4.1e-07 relative (1.1e-10 at the widest offset) — under-relaxation
# leaves the fixed point of the map exactly invariant and changes only the path to it.
const DAMP_THETA_U = Ref(0.9)

# Which zero of S^1 − S^0 above p*_S is read as p^oj_S.  :last takes the highest
# sign change, :first the lowest.  The OJS margin is a participation margin —
# every consumer of p^oj weights the region BELOW it as the searching one — so
# p^oj is the upper boundary of {p : S^1 > S^0} and :last is the rule that reads
# it.  :first collapses p^oj onto p*_S whenever the difference dips negative at
# the bottom of the p-grid, which shuts the OJS margin for that type and leaves
# the outer loop cycling between the two readings; it is kept only to reproduce
# runs made before v38.3.0.
const POJ_CROSSING = Ref(:last)

# Which form of the surpluses the p^oj scan scores.  :raw is the difference of
# the unsmoothed surpluses; :smooth takes the difference of the smooth_pos forms
# the inner loop prices (skilled.jl:350,355).  smooth_pos is strictly increasing,
# so the two agree on the sign of the difference and therefore on which cell
# holds the crossing — measured no-op at the base_fc estimate, |ΔQ| <= 4.2e-09
# over 25 parameter points.  The p*_S scan stays raw under both: smooth_pos is
# strictly positive, so a smoothed S^max never crosses zero and every
# reservation cutoff would collapse to 0.
const POJ_SCORE = Ref(:raw)

# Crossing rule INSIDE the bracketing cell.  The shipped rule is linear in the
# two bracketing nodes, so the located crossing is continuous in the parameters
# but kinks by O(h) when the bracket index moves; with true the sign change is
# interpolated by the local quadratic through three consecutive nodes, which
# reduces that stencil-switch kink to O(h²).
const POJ_CROSSING_C1 = Ref(false)

# How the type-level OJS participation decision is aggregated over an ability cell.
# On a finite ability grid a "type" is a CELL of positive width: the mass at node k is
# the Gauss-weighted mass of an interval, not a point mass at a_k.  The participation
# score is M(aS) = max_{p >= p*_S} (S^1 − S^0)(aS, p) — on-the-job search pays for aS at
# some acceptable quality iff M(aS) > 0 — so the correct participation weight for cell k
# is the FRACTION of the cell on which M > 0, and the node indicator 1{M(a_k) > 0} is the
# approximation to it rather than the other way round.
#
# :indicator is the node rule and reproduces every run up to v38.3.0.  A type on the
# knife edge M(a_k) ≈ 0 switches its whole OJS margin between consecutive outer
# iterations, p^oj collapses onto p*_S, and the outer map is discontinuous there whatever
# crossing rule is used inside the searching region: measured at the base_fc estimate,
# one ability cell of 160 attains the p^oj sup-norm on 100 of 100 iterations and the
# block runs a closed period-4 orbit for 200 iterations without converging.
# :fraction is the covered-fraction aggregation — the v22.0.0 pattern the training
# frontier, the drain and the tail integrals already carry, applied one level up.  The
# weight moves continuously from 1 to 0 as the frontier crosses the cell, the same state
# converges in 13 iterations, and the 24-point base_fc audit goes 19/24 to 24/24.
const OJS_PARTICIPATION = Ref(:fraction)


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

    c   :: Float64    # Psychological training cost coefficient c̃: c(aS) = (1−aS)·exp(c̃−aS),
                      # in fixed utility units — NOT scaled by A.  The training margin
                      # compares T(aS) − c(aS) with U^search(aU); since T and U^search scale
                      # with A while c does not, the margin shifts with the wage level.
                      # and is invariant to A like every other policy margin.

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
    k   :: Float64    # vacancy posting cost, in months of the mean output of a new
                      # unskilled match: dollar flow = k · π̄_U (mean_output_U, grids.jl)
    β   :: Float64    # worker Nash bargaining weight
    λ   :: Float64    # damage-shock arrival rate

    PU  :: Float64    # unskilled sector productivity shifter P_U: π_U(aU,p) = A·P_U·aU·p
    bU  :: Float64    # flow payoff in unskilled unemployment
    bT  :: Float64    # flow payoff while in training
    α_U :: Float64    # unskilled damage-shock Beta shape (p' ∼ Beta(α_U, 1), G(p) = p^{α_U})

    ξ   :: Float64 = 0.0   # exogenous unskilled separation hazard ξ_U (0 ⇒ recovers the no-ξ model)
    σ_w :: Float64 = 0.0   # log-wage measurement-error SD (σ_wU); 0 ⇒ no measurement error
end


Base.@kwdef struct UnskilledGrids
    p   :: Vector{Float64}    # unskilled match-quality grid
    wp  :: Vector{Float64}    # quadrature weights on unskilled p-grid
end


Base.@kwdef mutable struct UnskilledCache
    # Values of a type that searches — one-dimensional in own ability (notes §231, §276)
    Usearch   :: Vector{Float64}    # U^search(aU): value of unskilled search
    Jfrontier :: Vector{Float64}    # J_U(aU, 1): firm value at frontier p = 1
    pstar     :: Vector{Float64}    # p*(aU): unskilled reservation quality cutoff

    # Training value, and the unskilled matches whose outside option is training: those
    # of the trained workers hired into the unskilled market (career_choices!)
    T         :: Matrix{Float64}    # T(aU,aS): value of training, on U_S(aU,aS)
    pstar_t   :: Matrix{Float64}    # p*_t(aU,aS): their reservation quality
    S1_t      :: Matrix{Float64}    # S_t(aU,aS,1): their entry surplus, before the positive part
    χ_t       :: Matrix{Float64}    # χ_t(aU,aS): hired share of their meetings

    # Two-dimensional policy and composition on the (aU, aS) copula grid
    τT        :: Matrix{Float64}    # τ(aU,aS): training policy indicator
    u         :: Matrix{Float64}    # u_U(aU,aS): density of untrained unemployed
    uh        :: Matrix{Float64}    # hired branch of u_U(aU,aS)
    t         :: Matrix{Float64}    # t(aU,aS): density of workers in training
    et        :: Matrix{Float64}    # unskilled employment of the trained, outside option training

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
    a_Γ :: Float64    # skilled match-quality OFFER Beta shape 1
    b_Γ :: Float64    # skilled match-quality OFFER Beta shape 2

    δ   :: Float64 = 1.0   # shock/offer support ratio: a λ_S-shocked match re-draws quality
                           # δ·(offer draw) on [0,δ], so the shock distribution is a compressed
                           # copy of the offer.  δ = 1 ⇒ redraw ≡ fresh offer (single-distribution
                           # limit, recovering the pre-refinement model exactly).

    ξ   :: Float64 = 0.0   # exogenous skilled separation hazard ξ_S (0 ⇒ recovers the no-ξ model).
                           # Both blocks carry an exogenous hazard, but they are identified
                           # differently: p*_S = 0 at every ability, so skilled separation is
                           # essentially all ξ_S, while p*_U is interior, so ξ_U competes with a
                           # live endogenous margin (Model Notes §sep_asymmetry).
    σ_w :: Float64 = 0.0   # log-wage measurement-error SD (σ_wS); 0 ⇒ no measurement error
end


Base.@kwdef struct SkilledGrids
    p   :: Vector{Float64}    # skilled match-quality grid
    wp  :: Vector{Float64}    # quadrature weights on skilled p-grid
end


Base.@kwdef struct SkilledPrecomp
    # The two γ vectors are EXACT-CDF CELL MASSES PER UNIT wp, not pointwise
    # densities: γ_j ≡ (Γ(e_{j+1}) − Γ(e_j)) / wp_j over node j's cell (built by
    # build_cell_mass_density in grids.jl).  Every dΓ integral in the solver
    # appears as γ_j · wp_j and so carries the exact cell mass.  Reading either
    # vector pointwise is an error whose size does not shrink with the grid.
    #
    # OFFER distribution Γ_o = Beta(a_Γ, b_Γ) on [0,1].  Governs every fresh
    # meeting: hiring out of unemployment, on-the-job-search poaching
    # destinations, and the free-entry expected firm value.
    Γvals        :: Vector{Float64}    # offer CDF Γ_o(p) on skilled p-grid
    γvals        :: Vector{Float64}    # offer cell mass per unit wp
    tail_weights :: Vector{Float64}    # quadrature tail weights for offer Γ-integrals

    # SHOCK distribution Γ_s(x) = Γ_o(x/δ) on [0,δ].  Governs the λ_S redraw of
    # an ongoing match.  δ = 1 makes the shock arrays coincide with the offer
    # arrays exactly; δ < 1 compresses redraws toward lower quality, so a shock
    # can land below the reservation cutoff (reviving endogenous separation).
    Γs_vals        :: Vector{Float64}  # shock CDF Γ_s(p) on skilled p-grid
    γs_vals        :: Vector{Float64}  # shock cell mass per unit wp
    tail_weights_s :: Vector{Float64}  # quadrature tail weights for shock Γ-integrals
end


Base.@kwdef mutable struct SkilledCache
    # Values — one-dimensional in the SKILLED ability aS (notes §407–444)
    U     :: Vector{Float64}           # U_S^(0)(aS): the employed outside option, and the global link
    U0    :: Vector{Float64}           # U_S^(0)(aS): stay-skilled candidate
    U1    :: Matrix{Float64}           # U_S^(1)(aU,aS): cross-to-unskilled candidate (career_choices!)

    E0    :: Matrix{Float64}           # E_S^0(aS,p): worker value, employed, no OJS
    E1    :: Matrix{Float64}           # E_S^1(aS,p): worker value, employed, with OJS
    J0    :: Matrix{Float64}           # J_S^0(aS,p): firm value, filled job, no OJS
    J1    :: Matrix{Float64}           # J_S^1(aS,p): firm value, filled job, with OJS

    pstar :: Vector{Float64}           # p*_S(aS): endogenous separation cutoff
    poj   :: Vector{Float64}           # p^oj_S(aS): OJS cutoff

    # Fraction of ability cell aS on which on-the-job search pays at all, i.e. the share
    # of the cell covered by {aS : max_p (S^1 − S^0) > 0}.  Multiplies the p-coverage
    # weight at every consumer of the OJS margin (_ojs_share, grids.jl), so a cell the
    # frontier cuts contributes that fraction of the searching branch and the complement
    # of the non-searching one.  Identically 1 under OJS_PARTICIPATION[] === :indicator,
    # where p^oj alone carries participation.
    s_part :: Vector{Float64}

    # Cross-market directed-search policy on the (aU, aS) grid:
    # d(aU,aS) = 1 ⇔ U_S^(1)(aU,aS) > U_S^(0)(aS)  (notes eq:dpolicy).
    d     :: Matrix{Float64}

    # Stationary trained composition.  Per-aS UNIT shapes (total mass 1 for
    # a d = 0 type) — the p-dynamics read only aS, so the aU-dependence is
    # pure scaling by the 2D trained mass m_S (notes §386, §472–483):
    u_frac :: Vector{Float64}          # û_S(aS): unemployed fraction of a d=0 type's mass
    e_frac :: Matrix{Float64}          # ê_S(aS,p): employed density fraction of a d=0 type's mass

    m_S   :: Matrix{Float64}           # m_S(aU,aS): trained-worker mass = φ t / (ν + crossing hire rate)

    θ     :: Float64                   # skilled market tightness
end


##############
# Simulation controls
##############

Base.@kwdef struct SimParams
    # Block tolerances.  There is no global counterpart: the global loop converges on the
    # equilibrium residual battery, whose components are ratios to their own internal
    # tolerances and are gated at 1 (solver.jl, global_residuals).
    tol_inner      :: Float64    # tolerance for inner fixed point
    tol_outer_U    :: Float64    # tolerance for unskilled outer loop
    tol_outer_S    :: Float64    # tolerance for skilled outer loop

    maxit_inner    :: Int        # max iterations in inner loop
    maxit_outer    :: Int        # max iterations in block outer loops
    maxit_global   :: Int        # max iterations in global loop

    conv_streak    :: Int        # required consecutive convergence hits

    use_anderson   :: Bool       # Anderson acceleration in the unskilled block and the global
                                 # loop; the skilled block is never accelerated
    anderson_reg   :: Float64    # Anderson regularization

    damp_pstar_U   :: Float64    # damping for unskilled p*, read only when use_anderson = false
    damp_pstar_S   :: Float64    # damping for skilled p*_S and p^oj_S, read on every pass

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

    # Global-loop stall detection.  The coupled U↔S fixed point can limit-cycle
    # instead of converging; if the global residual fails to beat its best-so-far
    # over global_K iterations after a global_B burn-in, the solve stops as
    # non-converged.  Loss-neutral: the objective maps any non-converged solve to
    # Inf, so an early stop returns the identical Inf, only sooner.  global_B = 0
    # disables it.
    global_B :: Int = 0
    global_K :: Int = 5
end


##############
# Held education choices
##############

"""
    HeldChoices

The training choice and the choice of market of a stationary equilibrium, held at every type
by a model that carries them (`Model.held`): wherever the model takes a max over training or
over the market, it uses these instead.  Both cell fractions are held exactly as the solver
weighs them, and so are the node choices the value functions read:
  x       the ability grid they were solved on
  τ       training fraction of each (aU, aS) cell (training_policy!)
  d       drain fraction of each cell (drain_fraction!)
  trains  the node choice in U_U = max{U^search, −c + T}: true where training is chosen
  drains  the node choice in U_S = max{U_S^(0), U_S^(1)}: true where U_S^(1) is chosen
Built from a solved model by `HeldChoices(model)` (unskilled.jl).
"""
struct HeldChoices
    x      :: Vector{Float64}
    τ      :: Matrix{Float64}
    d      :: Matrix{Float64}
    trains :: Matrix{Bool}
    drains :: Matrix{Bool}
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

    # Set by solve_model! once the solve has converged at a corner: free entry at a market
    # that holds no one is then priced at its limit composition (marginal_training_cells).
    corner_limit :: Bool = false

    # The education choices held fixed (a cf1 window), or `nothing` when the model chooses.
    held :: Union{Nothing,HeldChoices} = nothing
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
        tol_inner    = 1e-8, tol_outer_U = 1e-6, tol_outer_S = 1e-7,
        maxit_inner  = 500,  maxit_outer = 300,  maxit_global = 50,
        conv_streak  = 2,
        use_anderson = true, anderson_reg = 1e-10,
        damp_pstar_U = 1.00, damp_pstar_S = 1.00,
        verbose      = 2,    verbose_stride = 10,
    )

    return build_model(cp, grids, up, u_grids, sp, s_grids, sim; Nx, Np_S)
end


"""
    build_model(cp, grids, up, u_grids, sp, s_grids, sim; Nx, Np_S, held = nothing) -> Model

Assemble a `Model` from parameter/grid blocks with freshly seeded caches.
Shared by `initialise_model` and the SMM entry point so both construct
identical initial conditions.  `held` holds the education choices (`HeldChoices`); they
must have been solved on this ability grid.
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
    ℓrow     = grids.wa_U                       # marginal mass in aU (for unemployment seed)

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
        t         = (ν / (2ν + φ)) .* grids.copula.W2,
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

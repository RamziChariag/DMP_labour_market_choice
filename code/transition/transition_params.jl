############################################################
# transition_params.jl — Parameters and path storage for the
#                         RoySearch backward–forward transition.
#
# Contains:
#   TransitionParams   — algorithm control knobs
#   TransitionPath     — full time-indexed working storage (internal)
#   TransitionResult   — serialisable output for downstream use
#
# State layout on the (aU, aS) copula grid
# ────────────────────────────────────────
# Value functions are one-dimensional in ability (unskilled in aU, skilled
# in aS), exactly as in the stationary solver.  Only the segment masses and
# the training policy are two-dimensional; the skilled p-composition is
# per-aS (the p-dynamics read only aS — notes §386).  The path therefore
# stores:
#   · 1D value paths        (Nx × Nt)          indexed by aU or aS
#   · match surplus surfaces (Nx × Np × Nt)    indexed by (own ability, p)
#   · 2D segment masses     (Nx × Nx × Nt)     indexed by (aU, aS)
#   · per-aS employment     (Nx × NpS × Nt)    indexed by (aS, p)
#
# SURPLUSES, NOT WORKER/FIRM VALUES.  The path carries the RAW match surpluses
# — before the soft reservation weight and before the positive part — because
# they are what the backward recursion needs: the raw surplus is the object
# whose time derivative the HJB prices, and its zero crossing IS the
# reservation cutoff.  Nash splitting recovers everything a consumer wants
# (J = (1−β)·ω·smooth_pos(S), E = U + β·ω·smooth_pos(S)), so storing
# E_S^0/E_S^1/J_S^0/J_S^1 as well would be four arrays holding one object's
# transforms — and the ω-weighted forms cannot be inverted where ω = 0.
############################################################

# ──────────────────────────────────────────────────────────
# TransitionParams
# ──────────────────────────────────────────────────────────

"""
    TransitionParams

Controls the backward–forward iteration for transition dynamics.

# Fields
- `T_max`   : horizon length (model time units, e.g. months)
- `N_steps` : number of time steps  ⇒  Nt = N_steps + 1 grid points
- `dt`      : step size = T_max / N_steps  (computed)
- `tol`     : convergence tolerance on the tightness paths (sup-norm)
- `maxit`   : maximum backward–forward iterations
- `damp`    : dampening on the tightness update ∈ (0, 1].  The skilled
              free-entry map is the binding one, and it is well inside its
              stability region: base_fc → crisis_fc converges in 16 iterations at
              0.50, 20 at 0.40, 28 at 0.30 and 54 at 0.15, with base_covid →
              crisis_covid 13/16/22/41 over the same ladder.
- `verbose` : print iteration info
"""
mutable struct TransitionParams
    T_max   :: Float64
    N_steps :: Int
    dt      :: Float64
    tol     :: Float64
    maxit   :: Int
    damp    :: Float64
    verbose :: Bool

    function TransitionParams(;
        T_max   :: Real  = 120.0,   # 10 years in months
        N_steps :: Int   = 240,
        tol     :: Real  = 1e-4,
        maxit   :: Int   = 200,
        damp    :: Real  = 0.3,
        verbose :: Bool  = true,
    )
        dt = Float64(T_max) / N_steps
        new(Float64(T_max), N_steps, dt, Float64(tol), maxit,
            Float64(damp), verbose)
    end
end


# ──────────────────────────────────────────────────────────
# TransitionPath  (internal, mutable working storage)
# ──────────────────────────────────────────────────────────

"""
    TransitionPath

Complete time-indexed path of value functions, policies, tightness, and
distributions over the transition horizon.  `Nt = N_steps + 1` dates; the
ability grid has `Nx` nodes and the skilled match-quality grid `NpS`.
"""
mutable struct TransitionPath
    tgrid :: Vector{Float64}          # (Nt) calendar of model-time nodes

    # Tightness paths
    θU :: Vector{Float64}             # (Nt)
    θS :: Vector{Float64}             # (Nt)

    # Unskilled value paths (indexed by aU, except T_val indexed by aS)
    Usearch   :: Matrix{Float64}      # (Nx,Nt) U^search(aU)
    T_val     :: Matrix{Float64}      # (Nx,Nt) T(aS): value of training
    Jfrontier :: Matrix{Float64}      # (Nx,Nt) J_U(aU,1)
    pstar_U   :: Matrix{Float64}      # (Nx,Nt) p*_U(aU)
    SU        :: Array{Float64,3}     # (Nx,NpU,Nt) raw S_U(aU,p)

    # Skilled value paths (indexed by aS, except US1 indexed by aU)
    US      :: Matrix{Float64}        # (Nx,Nt) U_S^(0)(aS): stay-skilled value
    US1     :: Matrix{Float64}        # (Nx,Nt) U_S^(1)(aU): cross-to-unskilled value
    pstar_S :: Matrix{Float64}        # (Nx,Nt) p*_S(aS)
    poj     :: Matrix{Float64}        # (Nx,Nt) p^oj_S(aS)
    S0      :: Array{Float64,3}       # (Nx,NpS,Nt) raw S_S^0(aS,p), no OJS
    S1      :: Array{Float64,3}       # (Nx,NpS,Nt) raw S_S^1(aS,p), with OJS

    # Two-dimensional segment masses on the (aU,aS) copula grid
    uU :: Array{Float64,3}            # (Nx,Nx,Nt) untrained unemployed
    tU :: Array{Float64,3}            # (Nx,Nx,Nt) in training
    uS :: Array{Float64,3}            # (Nx,Nx,Nt) skilled unemployed
    mS :: Array{Float64,3}            # (Nx,Nx,Nt) trained-segment mass
    τT :: Array{Float64,3}            # (Nx,Nx,Nt) training policy indicator

    # Pre-switch (date-0) state: the initial condition of the forward pass.
    # The training frontier reallocates the unemployed it encloses, so the
    # forward pass needs both the policy one date BEFORE the one it steps and a
    # clean copy of the inherited stocks to restart date 1 from on every outer
    # iteration — reallocating date 1 in place would compound across them.
    τT0 :: Matrix{Float64}            # (Nx,Nx) z₀ training policy
    uU0 :: Matrix{Float64}            # (Nx,Nx) z₀ untrained unemployed
    tU0 :: Matrix{Float64}            # (Nx,Nx) z₀ in training

    # Per-aS skilled employment p-composition (for wages and OJS seekers).
    # UNIT density of a d = 0 type, the convention `SkilledCache.e_frac` uses;
    # aggregates scale it by the non-draining column mass.
    eS :: Array{Float64,3}            # (Nx,NpS,Nt) ê_S(aS,p)
end

"""
    allocate_path(model, tp) -> TransitionPath

Allocate a zero-initialised path sized to `model`'s grids and `tp`'s horizon.
"""
function allocate_path(model::Model, tp::TransitionParams)
    Nt  = tp.N_steps + 1
    Nx  = length(model.grids.x)
    NpU = length(model.unsk_grids.p)
    NpS = length(model.skl_grids.p)
    tgrid = collect(range(0.0, tp.T_max; length = Nt))

    z2(a, b)    = zeros(Float64, a, b)
    z3(a, b, c) = zeros(Float64, a, b, c)

    return TransitionPath(
        tgrid,
        zeros(Float64, Nt), zeros(Float64, Nt),
        z2(Nx, Nt), z2(Nx, Nt), z2(Nx, Nt), z2(Nx, Nt), z3(Nx, NpU, Nt),
        z2(Nx, Nt), z2(Nx, Nt), z2(Nx, Nt), z2(Nx, Nt),
        z3(Nx, NpS, Nt), z3(Nx, NpS, Nt),
        z3(Nx, Nx, Nt), z3(Nx, Nx, Nt), z3(Nx, Nx, Nt), z3(Nx, Nx, Nt), z3(Nx, Nx, Nt),
        z2(Nx, Nx), z2(Nx, Nx), z2(Nx, Nx),
        z3(Nx, NpS, Nt),
    )
end


# ──────────────────────────────────────────────────────────
# TransitionResult  (serialisable output)
# ──────────────────────────────────────────────────────────

"""
    TransitionResult

Downstream-facing summary of a solved transition.  Aggregate series are
length-`Nt` vectors; the distribution fields `uU`, `tU`, `uS`, `mS` are
ability-*marginal densities* (`Nx × Nt`) so the panel/table layer aggregates
them as `dot(wx, field[:, n])` with `wx = wa` the marginal population
weights.  The full 2D masses are not serialised — the marginal profiles
carry everything the plotting layer consumes and keep the bundle small.
"""
struct TransitionResult
    scenario   :: Symbol
    converged  :: Bool
    n_iter     :: Int
    final_dist :: Float64

    tgrid :: Vector{Float64}
    θU    :: Vector{Float64}
    θS    :: Vector{Float64}
    fU    :: Vector{Float64}
    fS    :: Vector{Float64}

    ur_U           :: Vector{Float64}
    ur_S           :: Vector{Float64}
    ur_total       :: Vector{Float64}
    skilled_share  :: Vector{Float64}
    training_share :: Vector{Float64}
    mean_wage_U    :: Vector{Float64}
    mean_wage_S    :: Vector{Float64}

    # Ability-marginal density profiles (Nx × Nt)
    uU :: Matrix{Float64}
    tU :: Matrix{Float64}
    uS :: Matrix{Float64}
    mS :: Matrix{Float64}

    # Grids for downstream integration
    wx :: Vector{Float64}      # ability marginal weights (wa)
    x  :: Vector{Float64}      # ability nodes
    p  :: Vector{Float64}      # skilled match-quality nodes
    wp :: Vector{Float64}      # skilled match-quality weights

    # Transition diagnostics.  Appended after the grids rather than grouped with
    # the aggregate series above: Julia deserialises structs POSITIONALLY, so a
    # field inserted mid-struct silently shifts every later value in a stored
    # bundle, while one appended at the end raises EOFError instead.
    #
    # `frontier_floor` locates the TRAINING frontier F_τ: the lowest point of the
    # line, the aS at which the least unskilled-able worker is indifferent to
    # training.  The other three are the CROSS-MARKET frontier F_d, kept apart
    # because a zero in one says nothing about the others:
    #   d_region  ∬ d ℓ      — is the active side of F_d a non-empty region?
    #   d_mass    ∬ d m_S    — does any trained mass sit on it?
    #   d_flow    f_U ∬ d u_S — does anyone actually cross?
    # The flow is identically zero at both stationary endpoints by self-selection
    # (notes, prop:ss-crossflow) and positive only while the frontier is in
    # motion, so reading it alone cannot tell an inert solver from a channel that
    # is directionally unable to fire on this pair.
    frontier_floor :: Vector{Float64}
    d_region       :: Vector{Float64}
    d_mass         :: Vector{Float64}
    d_flow         :: Vector{Float64}
end

############################################################
# transition_params.jl — Parameters and path storage for the
#                         RoySearch backward–forward transition.
#
# Contains:
#   TransitionParams   — algorithm control knobs
#   TransitionPath     — full time-indexed working storage (internal)
#   TransitionResult   — serialisable output for downstream use
#   transition_file    — the file a run writes, tagged by its weight matrix and point
#
# State layout on the (aU, aS) copula grid
# ────────────────────────────────────────
# Off a steady state a value depends on the whole type (aU, aS): the unskilled
# unemployed price their later training option, and the skilled unemployed their
# later crossing choice.  Values are therefore stored per type (Nx × Nx × Nt).  The
# one-dimensional values the stationary solver uses are kept beside them (Nx × Nt):
# they are the exact values of every type that makes no different choice at any
# later date, which is most of the grid, so the full two-dimensional surfaces are
# solved only on the types that do (see transition_values.jl).
#
# SURFACES ARE RAW BRANCH VALUES.  A stored surface is the date-n value of a match
# before the positive part, continuing on the date-(n+1) envelope; its zero crossing
# is the reservation cutoff.  Nash splitting recovers worker and firm values
# (J = (1−β)·S⁺, E = U + β·S⁺).
############################################################

# ──────────────────────────────────────────────────────────
# TransitionParams
# ──────────────────────────────────────────────────────────

"""
    TransitionParams

Controls the backward–forward iteration for transition dynamics.

# Fields
- `T_max`   : span of the fine part of the time grid (model months)
- `N_steps` : number of fine steps over `[0, T_max]`
- `dt`      : fine step = T_max / N_steps  (computed)
- `horizon` : the date the path reaches the crisis steady state, ≥ `T_max`.  Beyond
              `T_max` the step grows geometrically by `TAIL_GROWTH`, so a long horizon
              adds few dates; the masses are advanced there in sub-steps of at most
              `FORWARD_SUBSTEP`.  `horizon = T_max` is the uniform grid.
- `tol`     : convergence tolerance on the free-entry gap max_n |ln θ^new − ln θ|
- `maxit`   : maximum backward–forward iterations
- `damp`    : weight on the free-entry tightness in the damped update ∈ (0, 1]
- `verbose` : print iteration info
"""
mutable struct TransitionParams
    T_max   :: Float64
    N_steps :: Int
    dt      :: Float64
    horizon :: Float64
    tol     :: Float64
    maxit   :: Int
    damp    :: Float64
    verbose :: Bool

    function TransitionParams(;
        T_max   :: Real  = 120.0,   # 10 years in months
        N_steps :: Int   = 240,
        horizon :: Real  = T_max,
        tol     :: Real  = 1e-4,
        maxit   :: Int   = 200,
        damp    :: Real  = 0.3,
        verbose :: Bool  = true,
    )
        horizon >= T_max || error("TransitionParams: horizon $horizon is shorter than T_max $T_max.")
        dt = Float64(T_max) / N_steps
        new(Float64(T_max), N_steps, dt, Float64(horizon), Float64(tol), maxit,
            Float64(damp), verbose)
    end
end

# Growth of the step beyond the fine grid, and the longest step the explicit laws of
# motion take.  Values are solved implicitly and are stable at any step; the masses are
# explicit, and at the stored estimates no exit rate reaches 0.5 per month, so a
# one-month sub-step keeps every stock positive.
const TAIL_GROWTH     = 1.05
const FORWARD_SUBSTEP = 1.0

"""
    time_grid(tp) -> Vector{Float64}

The dates of the path: `N_steps` equal steps over `[0, T_max]`, then steps growing by
`TAIL_GROWTH` until `horizon`, the last one ending on it.
"""
function time_grid(tp::TransitionParams)
    t = collect(range(0.0, tp.T_max; length = tp.N_steps + 1))
    Δ = tp.dt
    while t[end] < tp.horizon - 1e-9
        Δ *= TAIL_GROWTH
        push!(t, min(t[end] + Δ, tp.horizon))
    end
    return t
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

    # Unskilled values of a type that searches at every later date (the stationary
    # solver's objects, indexed by aU)
    Usearch :: Matrix{Float64}        # (Nx,Nt)       U^search(aU)
    pstar_U :: Matrix{Float64}        # (Nx,Nt)       its reservation p*_U(aU)
    SU1     :: Matrix{Float64}        # (Nx,Nt)       its entry surplus S_U(aU,1) ≥ 0
    SU      :: Array{Float64,3}       # (Nx,NpU,Nt)   its raw surface S_U(aU,p)

    # Unskilled values per type.  A cell holds a searching slice, the fraction 1−τ, and
    # a training slice, τ; each slice's unemployed have their own value and each
    # slice's matches their own outside option.
    UC    :: Array{Float64,3}         # (Nx,Nx,Nt) value of the searching slice's unemployed
    T     :: Array{Float64,3}         # (Nx,Nx,Nt) training value T(a)
    Utr   :: Array{Float64,3}         # (Nx,Nx,Nt) training option −c(aS) + T(a)
    τT    :: Array{Float64,3}         # (Nx,Nx,Nt) training fraction τ(a)
    pUs   :: Array{Float64,3}         # (Nx,Nx,Nt) reservation of the searching slice's matches
    pUt   :: Array{Float64,3}         # (Nx,Nx,Nt) reservation of the training slice's matches
    SUs1  :: Array{Float64,3}         # (Nx,Nx,Nt) searching slice's entry surplus S(a,1) ≥ 0
    SUt1  :: Array{Float64,3}         # (Nx,Nx,Nt) training slice's raw surplus at p = 1 (its positive part continues)
    zUs   :: Array{Float64,3}         # (Nx,Nx,Nt) flow value of unemployment, searching slice
    zUt   :: Array{Float64,3}         # (Nx,Nx,Nt) flow value of unemployment, training slice
    pureU :: Array{Bool,3}            # (Nx,Nx,Nt) the searching slice keeps searching at every later date

    # Skilled values.  Per aS for the types that stay skilled at every later date (the
    # stationary solver's objects); per type for the branch values of the crossing
    # choice, which is re-made at every date.
    US      :: Matrix{Float64}        # (Nx,Nt)       U_S^(0)(aS) of a type that never crosses later
    zS      :: Matrix{Float64}        # (Nx,Nt)       its flow value of unemployment
    pstar_S :: Matrix{Float64}        # (Nx,Nt)       p*_S(aS)
    poj     :: Matrix{Float64}        # (Nx,Nt)       p^oj_S(aS)
    s_part  :: Matrix{Float64}        # (Nx,Nt)       OJS participation weight of cell aS
    S0      :: Array{Float64,3}       # (Nx,NpS,Nt)   raw branch value without OJS
    S1      :: Array{Float64,3}       # (Nx,NpS,Nt)   raw branch value with OJS
    Sd      :: Array{Float64,3}       # (Nx,NpS,Nt)   value of a destination cell ω·[s S¹+(1−s) S⁰]⁺
    U0      :: Array{Float64,3}       # (Nx,Nx,Nt)    U_S^(0)(a): search the skilled market today
    U1      :: Array{Float64,3}       # (Nx,Nx,Nt)    U_S^(1)(a): search the unskilled market today
    pureS   :: Array{Bool,3}          # (Nx,Nx,Nt)    the type stays skilled at every later date

    # Masses per type on the (aU,aS) copula grid, after the date's reallocations
    uU  :: Array{Float64,3}           # (Nx,Nx,Nt) unskilled unemployed (searching slice)
    tU  :: Array{Float64,3}           # (Nx,Nx,Nt) in training
    eUs :: Array{Float64,3}           # (Nx,Nx,Nt) unskilled employment, searching slice
    eUt :: Array{Float64,3}           # (Nx,Nx,Nt) unskilled employment, training slice
    uS  :: Array{Float64,3}           # (Nx,Nx,Nt) skilled unemployed
    mS  :: Array{Float64,3}           # (Nx,Nx,Nt) skilled-segment mass
    eS  :: Array{Float64,3}           # (Nx,NpS,Nt) per-aS unit shape of skilled employment
    eUsp :: Array{Float64,3}          # (Nx,NpU+1,Nt) searching-slice matches by (aU, p), atom at p = 1 last
    eUtp :: Array{Float64,3}          # (Nx,NpU+1,Nt) training-slice matches by (aU, p)

    # Pre-switch state (date 0⁻): the base steady state, per type.
    τT0  :: Matrix{Float64}           # (Nx,Nx)
    uU0  :: Matrix{Float64}           # (Nx,Nx)
    tU0  :: Matrix{Float64}           # (Nx,Nx)
    eU0p :: Array{Float64,3}          # (Nx,Nx,NpU+1) unskilled employment by p, atom at p = 1 last
    pU0  :: Vector{Float64}           # (Nx) base unskilled reservation, the support of eU0p
    pS0  :: Vector{Float64}           # (Nx) base skilled reservation, the support of eS at 0⁻
    χU0  :: Vector{Float64}           # (Nx) base hired fraction χ by aU row, the support of the p = 1 atom

    # Per-date records of the forward pass
    wU_mean    :: Vector{Float64}     # (Nt) mean unskilled log wage
    ended_U    :: Vector{Float64}     # (Nt) unskilled matches ended by a rising cutoff, searching slice
    quit_T     :: Vector{Float64}     # (Nt) unskilled matches ended into training
    ended_S    :: Vector{Float64}     # (Nt) skilled matches ended by a rising cutoff
    swept_T    :: Vector{Float64}     # (Nt) unskilled unemployed moved into training
end

"""
    allocate_path(model, tp) -> TransitionPath

Allocate a zero-initialised path sized to `model`'s grids and `tp`'s horizon.
"""
function allocate_path(model::Model, tp::TransitionParams)
    tgrid = time_grid(tp)
    Nt  = length(tgrid)
    Nx  = length(model.grids.x)
    NpU = length(model.unsk_grids.p)
    NpS = length(model.skl_grids.p)

    z2(a, b)    = zeros(Float64, a, b)
    z3(a, b, c) = zeros(Float64, a, b, c)
    t3() = z3(Nx, Nx, Nt)
    b3() = trues(Nx, Nx, Nt) |> Array{Bool,3}

    return TransitionPath(
        tgrid, zeros(Nt), zeros(Nt),
        z2(Nx, Nt), z2(Nx, Nt), z2(Nx, Nt), z3(Nx, NpU, Nt),
        t3(), t3(), t3(), t3(), t3(), t3(), t3(), t3(), t3(), t3(), b3(),
        z2(Nx, Nt), z2(Nx, Nt), z2(Nx, Nt), z2(Nx, Nt), z2(Nx, Nt),
        z3(Nx, NpS, Nt), z3(Nx, NpS, Nt), z3(Nx, NpS, Nt), t3(), t3(), b3(),
        t3(), t3(), t3(), t3(), t3(), t3(), z3(Nx, NpS, Nt), z3(Nx, NpU + 1, Nt), z3(Nx, NpU + 1, Nt),
        z2(Nx, Nx), z2(Nx, Nx), z2(Nx, Nx), z3(Nx, Nx, NpU + 1), zeros(Nx), zeros(Nx), zeros(Nx),
        zeros(Nt), zeros(Nt), zeros(Nt), zeros(Nt), zeros(Nt),
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
    κU    :: Vector{Float64}
    κS    :: Vector{Float64}

    ur_U           :: Vector{Float64}
    ur_S           :: Vector{Float64}
    ur_total       :: Vector{Float64}
    skilled_share  :: Vector{Float64}
    training_share :: Vector{Float64}
    mean_wage_U    :: Vector{Float64}     # mean log wages, the data moments' units
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
    #   d_mass    ∬ d m_S    — does any skilled mass sit on it?
    #   d_flow    κ_U ∬ d u_S — does anyone actually cross?
    # The flow is identically zero at both stationary endpoints by self-selection
    # (notes, prop:ss-crossflow) and positive only while the frontier is in
    # motion, so reading it alone cannot tell an inert solver from a channel that
    # is directionally unable to fire on this pair.
    frontier_floor :: Vector{Float64}
    d_region       :: Vector{Float64}
    d_mass         :: Vector{Float64}
    d_flow         :: Vector{Float64}
end

# ──────────────────────────────────────────────────────────
# The transition file
# ──────────────────────────────────────────────────────────
# A run is tagged by its weight matrix and the point estimate both windows are read at,
# `<W>_<point>` with point :smm or :mcmc, and every file it writes ends in the tag: the
# transition file below, and the tables and plots made from it.  The runs at the two points
# therefore coexist.  Every writer and reader of a transition file loads this file.

transition_tag(w_suffix::AbstractString, point::Symbol) = "$(w_suffix)_$(point)"

transition_file(scenario::Symbol, w_suffix::AbstractString, point::Symbol) =
    joinpath(TRANS_OUT_DIR, "transition_$(scenario)$(transition_tag(w_suffix, point)).jls")

"""
    read_transition(scenario, w_suffix, point) -> NamedTuple or nothing

The transition of `scenario` solved at `point`, read from `transition_file`; `nothing` when
there is none.  A file that records another point stops the run.
"""
function read_transition(scenario::Symbol, w_suffix::AbstractString, point::Symbol)
    f = transition_file(scenario, w_suffix, point)
    isfile(f) || return nothing
    tr = open(deserialize, f)
    tr.point === point || error("$(basename(f)) records point = :$(tr.point), not :$(point).")
    return tr
end

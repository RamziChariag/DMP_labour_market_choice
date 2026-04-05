############################################################
# transition_params.jl — Parameters and path storage for
#                         backward-forward transition dynamics
#
# Contains:
#   TransitionParams   — algorithm control knobs
#   TransitionPath     — full time-indexed path (internal)
#   TransitionResult   — serialisable output for downstream use
############################################################

# ──────────────────────────────────────────────────────────
# TransitionParams
# ──────────────────────────────────────────────────────────

"""
    TransitionParams

Controls the backward-forward iteration for transition dynamics.

# Fields
- `T_max`   : horizon length (model time units, e.g. months)
- `N_steps` : number of time steps  ⇒  Nt = N_steps + 1 grid points
- `dt`      : step size = T_max / N_steps  (computed)
- `tol`     : convergence tolerance on θ paths (sup-norm)
- `maxit`   : maximum backward-forward iterations
- `damp`    : dampening on θ update ∈ (0, 1]
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

Stores the complete time-indexed path of value functions, policies,
and distributions during the backward-forward iteration.

All matrices are dimensioned (Nx, Nt) or (Nx, Np, Nt) where
Nt = N_steps + 1  is the number of time-grid points (including t=0
and t=T_max).

Convention: column n  ↔  time tgrid[n].
"""
mutable struct TransitionPath
    # ── Time grid ─────────────────────────────────────────
    tgrid :: Vector{Float64}          # length Nt

    # ── Tightness paths ──────────────────────────────────
    θU :: Vector{Float64}             # length Nt
    θS :: Vector{Float64}             # length Nt

    # ── Unskilled value functions  (Nx × Nt) ─────────────
    U        :: Matrix{Float64}       # outside value U_U(x,t)
    Usearch  :: Matrix{Float64}       # search value
    T_val    :: Matrix{Float64}       # training value
    Jfrontier :: Matrix{Float64}      # firm value at p = 1

    # ── Unskilled policies  (Nx × Nt) ────────────────────
    τT      :: Matrix{Float64}        # training indicator
    pstar_U :: Matrix{Float64}        # reservation quality p*_U(x,t)

    # ── Skilled value functions ───────────────────────────
    US :: Matrix{Float64}             # U_S(x,t)  (Nx × Nt)
    E0 :: Array{Float64, 3}          # E⁰_S(x,p,t)  (Nx × NpS × Nt)
    E1 :: Array{Float64, 3}          # E¹_S(x,p,t)
    J0 :: Array{Float64, 3}          # J⁰_S(x,p,t)
    J1 :: Array{Float64, 3}          # J¹_S(x,p,t)

    # ── Skilled policies  (Nx × Nt) ──────────────────────
    pstar_S :: Matrix{Float64}
    poj     :: Matrix{Float64}

    # ── Distributions  (Nx × Nt) ─────────────────────────
    uU :: Matrix{Float64}             # unskilled unemployed density
    tU :: Matrix{Float64}             # training density
    uS :: Matrix{Float64}             # skilled unemployed density
    mS :: Matrix{Float64}             # skilled segment mass  m_S(x,t)

    # ── Skilled employment  (Nx × NpS × Nt) ──────────────
    eS :: Array{Float64, 3}
end

"""
    TransitionPath(model_z0, model_z1, tp)

Pre-allocate all arrays.  Grids are taken from `model_z0` (assumed
identical to `model_z1`).
"""
function TransitionPath(model_z0::Model, ::Model, tp::TransitionParams)
    Nx  = length(model_z0.grids.x)
    NpS = length(model_z0.skl_grids.p)
    Nt  = tp.N_steps + 1

    tgrid = collect(range(0.0, tp.T_max, length = Nt))

    TransitionPath(
        tgrid,
        # tightness
        zeros(Nt), zeros(Nt),
        # unskilled values  (Nx × Nt)
        zeros(Nx, Nt), zeros(Nx, Nt), zeros(Nx, Nt), zeros(Nx, Nt),
        # unskilled policies
        zeros(Nx, Nt), zeros(Nx, Nt),
        # skilled values
        zeros(Nx, Nt),                                     # US
        zeros(Nx, NpS, Nt), zeros(Nx, NpS, Nt),           # E0, E1
        zeros(Nx, NpS, Nt), zeros(Nx, NpS, Nt),           # J0, J1
        # skilled policies
        zeros(Nx, Nt), zeros(Nx, Nt),
        # distributions
        zeros(Nx, Nt), zeros(Nx, Nt), zeros(Nx, Nt), zeros(Nx, Nt),
        # skilled employment
        zeros(Nx, NpS, Nt),
    )
end


# ──────────────────────────────────────────────────────────
# TransitionResult  (serialisable output)
# ──────────────────────────────────────────────────────────

"""
    TransitionResult

Immutable, serialisable container holding the converged transition
path plus aggregate time-series that are convenient for plotting.

Saved to disk via JLD2; everything needed to make any plot or table
is included without re-solving.
"""
struct TransitionResult
    # ── Metadata ──────────────────────────────────────────
    scenario     :: Symbol      # :fc or :covid
    converged    :: Bool
    n_iterations :: Int
    final_dist   :: Float64

    # ── Time grid ─────────────────────────────────────────
    tgrid :: Vector{Float64}    # length Nt

    # ── Tightness ─────────────────────────────────────────
    θU :: Vector{Float64}
    θS :: Vector{Float64}

    # ── Aggregate time-series (length Nt) ─────────────────
    fU             :: Vector{Float64}   # job-finding rate path
    fS             :: Vector{Float64}
    ur_U           :: Vector{Float64}   # unemployment rates
    ur_S           :: Vector{Float64}
    ur_total       :: Vector{Float64}
    skilled_share  :: Vector{Float64}
    training_share :: Vector{Float64}
    mean_wage_U    :: Vector{Float64}
    mean_wage_S    :: Vector{Float64}

    # ── Full distribution paths  (Nx × Nt) ───────────────
    uU :: Matrix{Float64}
    tU :: Matrix{Float64}
    uS :: Matrix{Float64}
    mS :: Matrix{Float64}
    eS :: Array{Float64, 3}            # Nx × NpS × Nt

    # ── Policy paths  (Nx × Nt) ──────────────────────────
    τT      :: Matrix{Float64}
    pstar_U :: Matrix{Float64}
    pstar_S :: Matrix{Float64}
    poj     :: Matrix{Float64}

    # ── Value-function paths  (Nx × Nt; for diagnostics) ─
    U  :: Matrix{Float64}
    US :: Matrix{Float64}

    # ── Grids (for self-contained plotting) ──────────────
    xg  :: Vector{Float64}
    wx  :: Vector{Float64}
    pgS :: Vector{Float64}
    wpS :: Vector{Float64}
end

"""
    transition_params.jl

Parameters and path objects for backward-forward transition dynamics.

The `TransitionParams` struct controls the algorithm convergence and time grid.
The `TransitionPath` struct stores all time-indexed objects: value functions,
policies, and distributions across the transition from regime z₀ to z₁.

See model notes Section 11.9 for the algorithm description.
"""

"""
    TransitionParams

Parameters controlling the backward-forward algorithm for transition dynamics.

# Fields
- `T_max :: Float64`: Maximum horizon (in calendar time)
- `N_steps :: Int`: Number of time steps (so dt = T_max / N_steps)
- `dt :: Float64`: Time step (computed automatically)
- `tol :: Float64`: Convergence tolerance for tightness paths (default 1e-5)
- `maxit :: Int`: Maximum iterations (default 100)
- `verbose :: Bool`: Print iteration info (default true)
- `damp :: Float64`: Dampening factor for tightness updates ∈ (0,1] (default 0.5)
"""
mutable struct TransitionParams
    T_max :: Float64
    N_steps :: Int
    dt :: Float64
    tol :: Float64
    maxit :: Int
    verbose :: Bool
    damp :: Float64

    function TransitionParams(T_max::Real, N_steps::Int; tol=1e-5, maxit=100, verbose=true, damp=0.5)
        dt = T_max / N_steps
        new(Float64(T_max), N_steps, dt, Float64(tol), maxit, verbose, Float64(damp))
    end
end

"""
    TransitionPath

Stores the complete time-indexed path of value functions, policies, and distributions
for the transition from regime z₀ to z₁.

# Time grid
- `tgrid :: Vector{Float64}`: Time points [0, dt, 2*dt, ..., T_max], length N+1

# Tightness paths (length N+1, indexed 1:N+1 for t=0:dt:T_max)
- `theta_U :: Vector{Float64}`: Unskilled market tightness θ_U(t)
- `theta_S :: Vector{Float64}`: Skilled market tightness θ_S(t)

# Policies at each time step (matrices Nx × (N+1), indexed over x and time)
- `tau :: Matrix{Float64}`: Training policy τ(x,t)
- `pstar_U :: Matrix{Float64}`: Unskilled reservation quality p*_U(x,t)
- `pstar_S :: Matrix{Float64}`: Skilled reservation quality p*_S(x,t)
- `poj :: Matrix{Float64}`: OJS cutoff p^oj_S(x,t)

# Value functions at each time step (matrices Nx × (N+1))
- `UU :: Matrix{Float64}`: Unskilled unemployment value U_U(x,t)
- `Usearch :: Matrix{Float64}`: Unskilled search value U_U^search(x,t)
- `T_val :: Matrix{Float64}`: Training value T(x,t)
- `US :: Matrix{Float64}`: Skilled unemployment value U_S(t) [constant in x]
- `Jfrontier :: Matrix{Float64}`: Unskilled firm value J_U(x,1,t) at quality frontier

# Skilled value surfaces (Vector of matrices, one per time step, each Nx × Np_S)
- `E0 :: Vector{Matrix{Float64}}`: Skilled worker value E⁰_S(x,p,t) when q=0
- `E1 :: Vector{Matrix{Float64}}`: Skilled worker value E¹_S(x,p,t) when q=1
- `J0 :: Vector{Matrix{Float64}}`: Skilled firm value J⁰_S(x,p,t) when p is backlog
- `J1 :: Vector{Matrix{Float64}}`: Skilled firm value J¹_S(x,p,t) when p is filled

# Distributions at each time step (matrices Nx × (N+1))
- `u_U :: Matrix{Float64}`: Untrained unemployed density u_U(x,t)
- `t_dens :: Matrix{Float64}`: Training density t(x,t)
- `u_S :: Matrix{Float64}`: Skilled unemployed density u_S(x,t)
- `m_S :: Matrix{Float64}`: Skilled segment mass m_S(x,t) [constant in x]

# Skilled employment distribution
- `e_S :: Vector{Matrix{Float64}}`: Skilled employment e_S(x,p,t), one matrix per time step
"""
mutable struct TransitionPath
    # Time grid
    tgrid :: Vector{Float64}

    # Tightness paths
    theta_U :: Vector{Float64}
    theta_S :: Vector{Float64}

    # Policies (Nx × (N+1))
    tau :: Matrix{Float64}
    pstar_U :: Matrix{Float64}
    pstar_S :: Matrix{Float64}
    poj :: Matrix{Float64}

    # Value functions (Nx × (N+1))
    UU :: Matrix{Float64}
    Usearch :: Matrix{Float64}
    T_val :: Matrix{Float64}
    US :: Vector{Float64}  # Constant in x, but varies in t
    Jfrontier :: Matrix{Float64}

    # Skilled value surfaces (Vector of Nx × Np_S matrices)
    E0 :: Vector{Matrix{Float64}}
    E1 :: Vector{Matrix{Float64}}
    J0 :: Vector{Matrix{Float64}}
    J1 :: Vector{Matrix{Float64}}

    # Distributions (Nx × (N+1))
    u_U :: Matrix{Float64}
    t_dens :: Matrix{Float64}
    u_S :: Matrix{Float64}
    m_S :: Vector{Float64}  # Constant in x, varies in t

    # Skilled employment (Vector of Nx × Np_S matrices)
    e_S :: Vector{Matrix{Float64}}
end

"""
    TransitionPath(model_z0::Model, model_z1::Model, tp::TransitionParams)

Constructor that pre-allocates all arrays to the correct dimensions.

Uses grid sizes from model_z0.grids (assumed same as model_z1.grids).
Returns an uninitialized TransitionPath; call `initialise_transition_path!` to fill it.
"""
function TransitionPath(model_z0::Model, model_z1::Model, tp::TransitionParams)
    Nx = length(model_z0.grids.x)
    Np_S = length(model_z0.skl_grids.p)
    N_time = tp.N_steps + 1  # Number of time points

    tgrid = collect(range(0.0, tp.T_max, length=N_time))

    # Pre-allocate arrays
    theta_U = zeros(N_time)
    theta_S = zeros(N_time)

    tau = zeros(Nx, N_time)
    pstar_U = zeros(Nx, N_time)
    pstar_S = zeros(Nx, N_time)
    poj = zeros(Nx, N_time)

    UU = zeros(Nx, N_time)
    Usearch = zeros(Nx, N_time)
    T_val = zeros(Nx, N_time)
    US = zeros(N_time)
    Jfrontier = zeros(Nx, N_time)

    E0 = [zeros(Nx, Np_S) for _ in 1:N_time]
    E1 = [zeros(Nx, Np_S) for _ in 1:N_time]
    J0 = [zeros(Nx, Np_S) for _ in 1:N_time]
    J1 = [zeros(Nx, Np_S) for _ in 1:N_time]

    u_U = zeros(Nx, N_time)
    t_dens = zeros(Nx, N_time)
    u_S = zeros(Nx, N_time)
    m_S = zeros(N_time)

    e_S = [zeros(Nx, Np_S) for _ in 1:N_time]

    TransitionPath(
        tgrid,
        theta_U, theta_S,
        tau, pstar_U, pstar_S, poj,
        UU, Usearch, T_val, US, Jfrontier,
        E0, E1, J0, J1,
        u_U, t_dens, u_S, m_S,
        e_S
    )
end

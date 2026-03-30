# Transition Dynamics: Backward-Forward Algorithm

This folder implements the backward-forward algorithm for computing transition dynamics after a regime change, as described in the model notes Section 11.9.

## File Structure

```
transition/
  main.jl                 — Entry point: solves z₀ and z₁ models, runs transition
  transition_params.jl    — TransitionParams and TransitionPath struct definitions
  transition_solver.jl    — Core algorithm: backward pass, forward pass, tightness updates
```

## Core Components

### 1. `transition_params.jl`

**TransitionParams** — Algorithm control
- `T_max`: Maximum horizon
- `N_steps`: Number of time steps  
- `tol, maxit, damp`: Convergence parameters

**TransitionPath** — Full time-indexed solution
- Time grid: `tgrid[n]` for n ∈ [1, N+1]
- Tightness paths: `theta_U[n], theta_S[n]`
- Value functions at each date: `UU, Usearch, T_val, US, Jfrontier` (Nx × N+1 matrices)
- Skilled surfaces: `E0, E1, J0, J1` (vectors of Nx × Np_S matrices)
- Policies: `tau, pstar_U, pstar_S, poj`
- Distributions: `u_U, t_dens, u_S` (Nx × N+1) and `m_S, e_S` (time-varying)

### 2. `transition_solver.jl`

**Main entry point**
- `solve_transition(model_z0, model_z1, tp)` → TransitionPath

**Algorithm steps**
1. `initialise_transition_path()` — Set up time grid, boundary conditions, initial guess
2. `backward_pass!()` — Solve value functions and policies from t=T_max to t=0
3. `forward_pass!()` — Evolve distributions forward using policy paths
4. `update_tightness!()` — Recover θ paths from free entry condition
5. `check_convergence()` — Monitor supremum norm of Δθ

**Helper functions**
- `solve_unskilled_block_at_time!()` — Within-period UE block solver at time n
- `solve_skilled_block_at_time!()` — Within-period skilled block solver at time n

### 3. `main.jl`

Complete workflow:
1. Define regime parameters (z₀ baseline, z₁ crisis)
2. Solve stationary models under both regimes
3. Set up transition parameters
4. Call `solve_transition()`
5. Save results to `output/transition_path.jld2`
6. Generate plots of tightness, unemployment, training, and skilled mass paths

## Algorithm Overview

**Input:** Two solved stationary models (z₀ and z₁)

**Step 0. Boundary objects**
- Initial distributions (t=0) from z₀ steady state
- Terminal value functions (t=T_max) from z₁ steady state

**Step 1. Initial tightness guess**
- Linear interpolation between z₀ and z₁ steady-state tightness values

**Step 2. Backward pass**
- At each date tₙ (n=N-1, ..., 0), given θ_U(tₙ) and θ_S(tₙ):
  - Solve unskilled block: compute U_U, U_U^search, T, and match values
  - Solve skilled block: compute U_S and match values
  - Store policies and values in `path` indexed by time

**Step 3. Forward pass**
- Integrate distributions using explicit Euler with dt = T_max / N_steps:
  - ∂_t t(x,t) = τ(x,t) u_U(x,t) - (φ+ν) t(x,t)
  - ∂_t u_U(x,t) = ν ℓ(x) + δ_U(x) e_U(x,t) - (f_U + τ(x,t) + ν) u_U(x,t)
  - ∂_t u_S(x,t) = φ t(x,t) + (ξ_S + δ_S^end(x)) e_S^tot(x,t) - (ν + f_S(1-Γ)) u_S(x,t)
  - ∂_t m_S(t) = φ ∫ t(x,t) dx - ν m_S(t)
  - ∂_t e_S(x,p,t) = match inflows - separations

**Step 4. Update tightness**
- Unskilled: q_U = k_U / ∫ J_U(x,1,tₙ) u_U(x,tₙ) dx  →  θ_U^new = (μ_U/q_U)^(1/η_U)
- Skilled: q_S = k_S / J_S(tₙ)  →  θ_S^new = (μ_S/q_S)^(1/η_S)
- Apply dampening: θ^(k+1) = (1-damp)θ^(k) + damp·θ^new

**Step 5. Convergence**
- Iterate Steps 2-4 until max_n |θ_j^(k+1)(tₙ) - θ_j^(k)(tₙ)| < tol

## Key Design Decisions

**Memory usage:** Unlike the stationary solver (which reuses caches in-place for SMM), the transition solver stores ALL objects at every time step. This enables efficient forward passes without recomputing equilibrium at each date.

**Time discretization:** Uses explicit Euler with fixed time step dt = T_max / N_steps. Could be upgraded to adaptive stepping or implicit schemes if needed.

**Dampening:** Applied to tightness updates (parameter `damp`) to stabilize convergence.

**Boundaries:** Initial conditions at t=0 come from z₀, terminal conditions at t=T_max from z₁.

## Usage

```julia
# Basic usage (from main.jl)
julia transition/main.jl

# Custom usage
include("transition/transition_params.jl")
include("transition/transition_solver.jl")

model_z0 = ...  # Solved model under regime z₀
model_z1 = ...  # Solved model under regime z₁

tp = TransitionParams(T_max=10.0, N_steps=100, tol=1e-4, damp=0.5)
path = solve_transition(model_z0, model_z1, tp)

# Access results
plot(path.tgrid, path.theta_U)  # Tightness path
plot(path.tgrid, path.u_U[i, :])  # Unemployment path for skill x[i]
```

## Integration with Existing Code

The transition solver imports from `../solver/`:
- `params.jl` — Model parameter structs
- `grids.jl` — Grid utilities and matching functions
- `solver_stationary.jl` — Stationary solver (provides `solve_model!`)

The stationary solver is run twice (for z₀ and z₁) before the transition dynamics are computed.

## Notes

- **Placeholders:** Functions `solve_unskilled_block_at_time!()` and `solve_skilled_block_at_time!()` currently copy values from the terminal steady state. In full implementation, these would call the inner loops from the stationary solver and store results indexed by time.

- **Skilled employment distribution:** The forward pass for `e_S` is simplified. Full implementation requires tracking match flows and separations across the quality grid.

- **Visualization:** The `main.jl` script generates four plots:
  - `tightness_paths.png` — θ_U and θ_S over time
  - `unskilled_unemployment.png` — u_U(x,t) for selected productivity levels
  - `skilled_unemployment.png` — u_S(x,t) for selected productivity levels
  - `training_density.png` — t(x,t) for selected productivity levels
  - `skilled_mass.png` — m_S(t) scalar path

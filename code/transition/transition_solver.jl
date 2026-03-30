"""
    transition_solver.jl

Core backward-forward algorithm for solving transition dynamics after a regime change.

The backward-forward algorithm (Section 11.9) computes the transition path from an
initial stationary equilibrium (z₀) to a terminal stationary equilibrium (z₁).

Algorithm overview:
  Step 0: Compute z₀ and z₁ steady states (done externally, passed as model_z0, model_z1)
  Step 1: Initialize tightness paths and distributions
  Step 2: Backward pass — solve value functions and policies at each date
  Step 3: Forward pass — advance distributions using laws of motion
  Step 4: Update tightness paths using free entry condition
  Step 5: Iterate until convergence

Key insight: The transition solver stores all time-indexed objects, whereas the
stationary solver reuses caches in-place for memory efficiency.
"""

"""
    solve_transition(model_z0::Model, model_z1::Model, tp::TransitionParams) -> TransitionPath

Solve the backward-forward transition dynamics from regime z₀ to z₁.

# Arguments
- `model_z0 :: Model`: Solved stationary model under regime z₀ (initial conditions)
- `model_z1 :: Model`: Solved stationary model under regime z₁ (terminal conditions)
- `tp :: TransitionParams`: Algorithm parameters (T_max, N_steps, tol, maxit, damp, etc.)

# Returns
- `TransitionPath`: Complete time-indexed path with value functions, policies, and distributions

# Algorithm
1. Initialize time grid, distributions, and tightness paths
2. Iterate until convergence (max `tp.maxit` times):
   a. Backward pass: solve value functions and policies at each date
   b. Forward pass: evolve distributions using solved policies
   c. Update tightness using free entry condition
   d. Check convergence on tightness paths
3. Return the converged transition path

# Notes
- The stationary models (model_z0, model_z1) must already be solved.
- Initial distributions (t=0) come from z₀ steady state.
- Terminal value functions (t=T_max) come from z₁ steady state.
- Policies and distributions are computed for all intermediate dates.
"""
function solve_transition(model_z0::Model, model_z1::Model, tp::TransitionParams)
    if tp.verbose
        @printf("\n%s\n", "="^60)
        @printf("Transition dynamics: z₀ → z₁\n")
        @printf("T_max = %.2f, N_steps = %d, dt = %.4f\n", tp.T_max, tp.N_steps, tp.dt)
        @printf("%s\n", "="^60)
    end

    # Step 0: Initialize path
    path = initialise_transition_path(model_z0, model_z1, tp)

    # Step 2-4: Iterate backward-forward loop
    for iter in 1:tp.maxit
        # Save old tightness for convergence check
        theta_U_old = copy(path.theta_U)
        theta_S_old = copy(path.theta_S)

        # Step 2: Backward pass (solve value functions and policies)
        backward_pass!(path, model_z0, model_z1, tp)

        # Step 3: Forward pass (evolve distributions)
        forward_pass!(path, model_z0, model_z1, tp)

        # Step 4: Update tightness paths
        update_tightness!(path, model_z1, tp)

        # Step 5: Check convergence
        converged, dist = check_convergence(theta_U_old, theta_S_old, path, tp)

        if tp.verbose
            @printf("Iteration %3d: ||Δθ|| = %.2e, damp = %.2f\n", iter, dist, tp.damp)
            flush(stdout)
        end

        if converged
            if tp.verbose
                @printf("Converged in %d iterations.\n", iter)
                @printf("%s\n", "="^60)
                flush(stdout)
            end
            return path
        end
    end

    if tp.verbose
        _, dist_final = check_convergence(theta_U_old, theta_S_old, path, tp)
        @printf("WARNING: Did not converge after %d iterations. dist = %.2e\n",
                tp.maxit, dist_final)
        flush(stdout)
    end

    return path
end

"""
    initialise_transition_path(model_z0::Model, model_z1::Model, tp::TransitionParams) -> TransitionPath

Initialize the transition path with boundary conditions and initial guess.

# Setup
- Time grid: [0, dt, 2*dt, ..., T_max]
- Initial distributions (t=0): Copy from z₀ steady state
- Terminal value functions (t=T_max): Copy from z₁ steady state
- Initial tightness guess: Linear interpolation between z₀ and z₁ steady-state values

# Returns
- `TransitionPath`: Pre-allocated and partially initialized
"""
function initialise_transition_path(model_z0::Model, model_z1::Model, tp::TransitionParams)
    path = TransitionPath(model_z0, model_z1, tp)
    Nx = length(model_z0.grids.x)
    Np_S = length(model_z0.skl_grids.p)
    N_time = tp.N_steps + 1

    # Initial tightness guess: linear interpolation from z₀ to z₁
    theta_U_0 = model_z0.unsk_cache.θ
    theta_U_1 = model_z1.unsk_cache.θ
    theta_S_0 = model_z0.skl_cache.θ
    theta_S_1 = model_z1.skl_cache.θ

    for n in 1:N_time
        t_frac = (n - 1) / tp.N_steps  # Fraction of time elapsed
        path.theta_U[n] = theta_U_0 + t_frac * (theta_U_1 - theta_U_0)
        path.theta_S[n] = theta_S_0 + t_frac * (theta_S_1 - theta_S_0)
    end

    # Initial distributions: copy from z₀ steady state
    for i in 1:Nx
        path.u_U[i, 1] = model_z0.unsk_cache.u[i]
        path.t_dens[i, 1] = model_z0.unsk_cache.t[i]
        path.u_S[i, 1] = model_z0.skl_cache.u[i]
    end
    path.m_S[1] = sum(model_z0.skl_cache.u .* model_z0.grids.wx) +
                  sum(model_z0.skl_cache.e .* model_z0.skl_grids.wp' .* reshape(model_z0.grids.wx, :, 1))

    for i in 1:Nx, j in 1:Np_S
        path.e_S[1][i, j] = model_z0.skl_cache.e[i, j]
    end

    # Terminal value functions: copy from z₁ steady state
    for i in 1:Nx
        path.UU[i, N_time] = model_z1.unsk_cache.U[i]
        path.Usearch[i, N_time] = model_z1.unsk_cache.Usearch[i]
        path.T_val[i, N_time] = model_z1.unsk_cache.T[i]
        path.Jfrontier[i, N_time] = model_z1.unsk_cache.Jfrontier[i]
        path.tau[i, N_time] = model_z1.unsk_cache.τT[i]
        path.pstar_U[i, N_time] = model_z1.unsk_cache.pstar[i]
        path.pstar_S[i, N_time] = model_z1.skl_cache.pstar[i]
        path.poj[i, N_time] = model_z1.skl_cache.poj[i]
    end
    path.US[N_time] = model_z1.skl_cache.U[1]

    for i in 1:Nx, j in 1:Np_S
        path.E0[N_time][i, j] = model_z1.skl_cache.E0[i, j]
        path.E1[N_time][i, j] = model_z1.skl_cache.E1[i, j]
        path.J0[N_time][i, j] = model_z1.skl_cache.J0[i, j]
        path.J1[N_time][i, j] = model_z1.skl_cache.J1[i, j]
    end

    return path
end

"""
    backward_pass!(path, model_z0, model_z1, tp)

Solve value functions and policies backward from t=T_max to t=0.

At each date tₙ, given tightness θ_U(tₙ) and θ_S(tₙ), solve the
within-period equilibrium under z₁ parameters (the shock has already hit;
values reflect the new regime).

The backward pass computes instantaneous equilibrium values at each date
by solving the Bellman equations with the current (guessed) tightness path.
Terminal conditions at t=T_max come from the z₁ steady state.

# Modifies
- `path.UU, path.Usearch, path.T_val, path.US`
- `path.E0, path.E1, path.J0, path.J1`
- `path.tau, path.pstar_U, path.pstar_S, path.poj, path.Jfrontier`
"""
function backward_pass!(path::TransitionPath, model_z0::Model, model_z1::Model,
                        tp::TransitionParams)
    model = model_z1  # Use post-shock parameters
    Nx = length(model.grids.x)
    Np_S = length(model.skl_grids.p)
    N_time = tp.N_steps + 1

    # Extract parameters
    cp  = model.common
    rp  = model.regime
    up  = model.unsk_par
    sp  = model.skl_par
    pre = model.skl_pre
    gp  = model.grids
    ug  = model.unsk_grids
    sg  = model.skl_grids

    r  = cp.r;  ν = cp.ν;  φ = cp.φ;  c = cp.c
    PU = rp.PU; PS = rp.PS; bU = rp.bU; bT = rp.bT; bS = rp.bS; αU = rp.α_U
    βU = up.β;  λU = up.λ;  μU = up.μ;  ηU = up.η
    βS = sp.β;  ξS = sp.ξ;  λS = sp.λ;  σS = sp.σ;  μS = sp.μ;  ηS = sp.η

    wG  = build_unskilled_G_weights(ug.p, ug.wp, αU)
    wΓ  = pre.γvals .* sg.wp

    Svec = zeros(Float64, length(ug.p))

    # Loop backward from second-to-last to first time step
    for n in (N_time - 1):-1:1
        theta_U_n = path.theta_U[n]
        theta_S_n = path.theta_S[n]

        f_U = theta_U_n * q_from_theta(theta_U_n, μU, ηU)
        f_S = theta_S_n * q_from_theta(theta_S_n, μS, ηS)

        # ── Skilled block at time n ──────────────────────────────────────

        # Use value functions from n+1 as "continuation" to compute
        # the within-period equilibrium at n.  In the stationary solver
        # this is done by fixed-point iteration; for the transition we
        # use the fact that with known tightness and terminal values, the
        # Bellman equations can be solved in a single backward step.

        # Step S1: Compute tail integrals and I(x) from Smax at time n+1
        #          (initialise from previous time step or terminal)
        denom_nb_S = max(1.0 - βS, 1e-14)

        for ix in 1:Nx
            x     = gp.x[ix]
            pstar = clamp01(path.pstar_S[ix, n+1])
            j0    = pcut_index(sg.p, pstar)

            # Tail integral of Smax under dΓ
            tailE = zeros(Float64, Np_S)
            acc = 0.0
            for j in Np_S:-1:1
                Smax_j = max(path.J0[n+1][ix, j], path.J1[n+1][ix, j]) / denom_nb_S
                acc      += Smax_j * wΓ[j]
                tailE[j]  = acc
            end
            I = tailE[j0]

            # Unemployment value: (r+ν) U_S = bS + f_S·βS·I
            US_new = (bS + f_S * βS * I) / (r + ν)
            path.US[n] = US_new  # US is x-invariant in the model

            # Surplus surfaces
            base = r + ν + ξS + λS

            for j in 1:Np_S
                pj = sg.p[j]
                if j < j0
                    path.E0[n][ix, j] = US_new
                    path.E1[n][ix, j] = US_new
                    path.J0[n][ix, j] = 0.0
                    path.J1[n][ix, j] = 0.0
                    continue
                end

                # No-search surplus
                S0 = (PS * x * pj - (r + ν) * US_new + λS * I) / base

                # OJS surplus
                tail_mass_j = pre.tail_weights[j]
                tail_Emax_j = tailE[max(j, j0)]
                S1 = (PS * x * pj - (r + ν) * US_new - σS + λS * I +
                      f_S * βS * tail_Emax_j) / (base + f_S * tail_mass_j)

                path.E0[n][ix, j] = US_new + βS * S0
                path.E1[n][ix, j] = US_new + βS * S1
                path.J0[n][ix, j] = (1.0 - βS) * S0
                path.J1[n][ix, j] = (1.0 - βS) * S1
            end

            # Update cutoffs from surplus
            Smax_vec = zeros(Float64, Np_S)
            diff_vec = zeros(Float64, Np_S)
            for j in 1:Np_S
                S0 = path.J0[n][ix, j] / denom_nb_S
                S1 = path.J1[n][ix, j] / denom_nb_S
                Smax_vec[j] = max(S0, S1)
                diff_vec[j] = S1 - S0
            end

            j0_prev = pcut_index(sg.p, clamp01(path.pstar_S[ix, n+1]))
            path.pstar_S[ix, n] = clamp01(find_cutoff_from_j0(sg.p, Smax_vec, j0_prev))
            raw_poj = clamp01(find_poj_from_diff_grid(sg.p, diff_vec, path.pstar_S[ix, n]))
            path.poj[ix, n] = max(path.pstar_S[ix, n], raw_poj)
        end

        # ── Unskilled block at time n ────────────────────────────────────

        # Get the US value just computed above (or from n+1 terminal condition)
        US_current = path.US[n]

        for ix in 1:Nx
            x = gp.x[ix]

            # Training value: (r + φ + ν) T(x) = bT + φ · U_S
            T_val = (bT + φ * US_current) / (r + φ + ν)
            path.T_val[ix, n] = T_val

            # Unskilled surplus and firm value at p=1
            pstar_x = clamp01(path.pstar_U[ix, n+1])  # Use previous guess

            I_U, _, _ = solve_unskilled_surplus_on_grid!(
                Svec, ug.p, wG, PU, x, r, ν, λU, path.UU[ix, n+1], pstar_x
            )

            S1_U = Svec[end]
            J1_U = (1.0 - βU) * S1_U
            E1_U = path.UU[ix, n+1] + βU * S1_U

            path.Jfrontier[ix, n] = J1_U

            # Search value: (r + ν + f_U) Usearch(x) = bU + f_U · E(x, p=1)
            Usearch = (bU + f_U * E1_U) / (r + ν + f_U)
            path.Usearch[ix, n] = Usearch

            # Training decision
            Utr = -training_cost(x, c) + T_val
            if Utr >= Usearch
                path.UU[ix, n]  = Utr
                path.tau[ix, n] = 1.0
            else
                path.UU[ix, n]  = Usearch
                path.tau[ix, n] = 0.0
            end

            # Update reservation quality pstar_U
            if x > 1e-14 && PU > 1e-14
                path.pstar_U[ix, n] = clamp01(((r + ν) * path.UU[ix, n] - λU * I_U) / (PU * x))
            else
                path.pstar_U[ix, n] = 1.0
            end
        end
    end
end

"""
    forward_pass!(path, model_z0, model_z1, tp)

Evolve distributions forward from t=0 to t=T_max using explicit Euler.

Given the policy paths computed in backward_pass!, integrate forward:
- ∂_t t(x,t) = τ(x,t) u_U(x,t) - (φ+ν) t(x,t)
- ∂_t u_U(x,t) = ν ℓ(x) + δ_U(x) e_U(x,t) - (f_U + τ(x,t) + ν) u_U(x,t)
- ∂_t u_S(x,t) = φ t(x,t) + (ξ_S + δ^end_S(x)) e^tot_S(x,t) - (ν + f_S(1-Γ(p*_S(x,t)))) u_S(x,t)
- ∂_t m_S(x,t) = φ t(x,t) - ν m_S(x,t)
- ∂_t e_S(x,p,t) = inflow - outflow

The distributions at t=0 are set by initialization (copy from z₀ steady state).
The forward pass fills in the intermediate steps up to t=T_max.

# Modifies
- `path.u_U, path.t_dens, path.u_S, path.m_S, path.e_S`
"""
function forward_pass!(path::TransitionPath, model_z0::Model, model_z1::Model,
                       tp::TransitionParams)
    model = model_z1  # Post-shock parameters
    Nx = length(model.grids.x)
    Np_S = length(model.skl_grids.p)
    N_time = tp.N_steps + 1

    # Extract time-invariant parameters
    phi  = model.common.φ
    nu   = model.common.ν
    xi_S = model.skl_par.ξ
    λU   = model.unsk_par.λ
    αU   = model.regime.α_U
    λS   = model.skl_par.λ

    # Loop forward: from t=0 to t=T_max-dt
    for n in 1:(N_time - 1)
        dt = tp.dt

        # Time-dependent job-finding rates from current tightness
        f_U_n = path.theta_U[n] * q_from_theta(path.theta_U[n], model.unsk_par.μ, model.unsk_par.η)
        f_S_n = path.theta_S[n] * q_from_theta(path.theta_S[n], model.skl_par.μ, model.skl_par.η)

        # Get current distributions and policies
        tau_n     = @view path.tau[:, n]
        pstar_U_n = @view path.pstar_U[:, n]
        pstar_S_n = @view path.pstar_S[:, n]
        u_U_n     = @view path.u_U[:, n]
        t_n       = @view path.t_dens[:, n]
        u_S_n     = @view path.u_S[:, n]
        m_S_n     = path.m_S[n]
        e_S_n     = path.e_S[n]

        # Get distributions at next time step (to be filled)
        u_U_next = @view path.u_U[:, n + 1]
        t_next   = @view path.t_dens[:, n + 1]
        u_S_next = @view path.u_S[:, n + 1]
        e_S_next = path.e_S[n + 1]

        # Compute Gamma(pstar_S) for each x (match acceptance probability)
        Gamma_pstar_S = zeros(Nx)
        for i in 1:Nx
            j_ps = pcut_index(model.skl_grids.p, clamp01(pstar_S_n[i]))
            Gamma_pstar_S[i] = model.skl_pre.Γvals[j_ps]
        end

        # ── Update training density: ∂_t t = τ u_U - (φ+ν) t ─────────
        for i in 1:Nx
            dt_train = tau_n[i] * u_U_n[i] - (phi + nu) * t_n[i]
            t_next[i] = max(t_n[i] + dt * dt_train, 0.0)
        end

        # ── Update unskilled unemployed ───────────────────────────────
        for i in 1:Nx
            ell_x   = model.grids.ℓ[i]
            m_U_x   = max(ell_x - (phi / nu) * t_n[i], 0.0)
            e_U_i   = max(m_U_x - u_U_n[i] - t_n[i], 0.0)
            delta_U = λU * clamp01(pstar_U_n[i])^αU
            du_U_dt = nu * ell_x + delta_U * e_U_i - (f_U_n + tau_n[i] + nu) * u_U_n[i]
            u_U_next[i] = max(u_U_n[i] + dt * du_U_dt, 0.0)
        end

        # ── Update skilled unemployed ─────────────────────────────────
        for i in 1:Nx
            e_tot_S_i = sum(e_S_n[i, :] .* model.skl_grids.wp)
            delta_S   = xi_S + λS * Gamma_pstar_S[i]
            du_S_dt   = phi * t_n[i] + delta_S * e_tot_S_i -
                        (nu + f_S_n * (1.0 - Gamma_pstar_S[i])) * u_S_n[i]
            u_S_next[i] = max(u_S_n[i] + dt * du_S_dt, 0.0)
        end

        # ── Update skilled segment mass: ∂_t m_S = φ t - ν m_S ───────
        total_t = sum(t_n .* model.grids.wx)
        dm_S_dt = phi * total_t - nu * m_S_n
        path.m_S[n + 1] = max(m_S_n + dt * dm_S_dt, 0.0)

        # ── Update skilled employment: ∂_t e_S(x,p,t) ────────────────
        # Inflow: f_S · γ(p) · u_S(x)  for p >= p*_S(x)
        # Outflow: (nu + xi_S + lambda_S · Gamma(p)) · e_S(x,p)
        # Quality shock redistribution: lambda_S · gamma(p) · ∫_{p'<p} e_S(x,p') dp'
        for i in 1:Nx
            pstar_i = clamp01(pstar_S_n[i])
            j0 = pcut_index(model.skl_grids.p, pstar_i)

            # Cumulative employment below each quality level
            cum_e = 0.0
            for j in 1:Np_S
                if j >= j0
                    p_j = model.skl_grids.p[j]
                    gamma_j = model.skl_pre.γvals[j]
                    Gamma_j = model.skl_pre.Γvals[j]

                    # Inflow from unemployment (new hires at quality p)
                    inflow_u = f_S_n * gamma_j * u_S_n[i]

                    # Inflow from quality shocks (redistribution from below)
                    inflow_lambda = λS * gamma_j * cum_e

                    # Outflow
                    outflow = (nu + xi_S + λS * Gamma_j) * e_S_n[i, j]

                    e_S_next[i, j] = max(e_S_n[i, j] + dt * (inflow_u + inflow_lambda - outflow), 0.0)
                else
                    e_S_next[i, j] = 0.0
                end

                cum_e += e_S_n[i, j] * model.skl_grids.wp[j]
            end
        end
    end
end

"""
    update_tightness!(path, model_z1, tp)

Update tightness paths using the free entry condition.

At each date tₙ:
- Unskilled: q_U · J_bar_U = k_U  where J_bar_U is the expected firm value
- Skilled: q_S · J_bar_S = k_S  where J_bar_S is the expected firm value

Apply dampening to smooth the update:
  θ^(k+1) = (1 - damp) θ^(k) + damp θ^new

# Modifies
- `path.theta_U, path.theta_S`
"""
function update_tightness!(path::TransitionPath, model_z1::Model, tp::TransitionParams)
    model = model_z1
    Nx = length(model.grids.x)
    Np_S = length(model.skl_grids.p)
    N_time = tp.N_steps + 1

    k_U   = model.unsk_par.k
    k_S   = model.skl_par.k
    eta_U = model.unsk_par.η
    eta_S = model.skl_par.η
    mu_U  = model.unsk_par.μ
    mu_S  = model.skl_par.μ
    βS    = model.skl_par.β

    for n in 1:N_time
        # ── Unskilled free entry ──────────────────────────────────────
        # q_U = k_U / ∫ J_U(x,1) u_U(x) dx / ∫ u_U(x) dx
        J_times_u = 0.0
        u_total = 0.0
        for i in 1:Nx
            J_times_u += path.Jfrontier[i, n] * path.u_U[i, n] * model.grids.wx[i]
            u_total   += path.u_U[i, n] * model.grids.wx[i]
        end

        if J_times_u > 1e-10 && u_total > 1e-10
            q_U_new = k_U * u_total / J_times_u
            theta_U_new = theta_from_q(q_U_new, mu_U, eta_U)
            theta_U_new = clamp(theta_U_new, 1e-14, 50.0)
        else
            theta_U_new = path.theta_U[n]
        end

        # Apply dampening
        path.theta_U[n] = (1.0 - tp.damp) * path.theta_U[n] + tp.damp * theta_U_new

        # ── Skilled free entry ────────────────────────────────────────
        # Expected firm value from posting
        wΓ = model.skl_pre.γvals .* model.skl_grids.wp
        denom_nb = max(1.0 - βS, 1e-14)

        J_S_num = 0.0
        J_S_den = 0.0
        for i in 1:Nx
            wx_i    = model.grids.wx[i]
            pstar_i = clamp01(path.pstar_S[i, n])
            j0      = pcut_index(model.skl_grids.p, pstar_i)

            # Tail integral of J from p*
            tailJ = 0.0
            for j in Np_S:-1:j0
                J_max = max(path.J0[n][i, j], path.J1[n][i, j])
                tailJ += J_max * wΓ[j]
            end

            # Seekers: unemployed + employed doing OJS
            u_i = path.u_S[i, n]
            seeker_e = 0.0
            for j in 1:Np_S
                if path.E1[n][i, j] >= path.E0[n][i, j]
                    seeker_e += path.e_S[n][i, j] * model.skl_grids.wp[j]
                end
            end

            seekers = u_i + seeker_e
            J_S_num += wx_i * seekers * tailJ
            J_S_den += wx_i * seekers
        end

        if J_S_num > 1e-10 && J_S_den > 1e-10
            Jbar_S = J_S_num / J_S_den
            q_S_new = k_S / Jbar_S
            theta_S_new = theta_from_q(q_S_new, mu_S, eta_S)
            theta_S_new = clamp(theta_S_new, 1e-14, 30.0)
        else
            theta_S_new = path.theta_S[n]
        end

        # Apply dampening
        path.theta_S[n] = (1.0 - tp.damp) * path.theta_S[n] + tp.damp * theta_S_new
    end
end

"""
    check_convergence(theta_U_old, theta_S_old, path, tp) -> (converged::Bool, dist::Float64)

Check convergence of tightness paths.

Convergence criterion: max_n |θ_j^(k+1)(tₙ) - θ_j^(k)(tₙ)| < tol for both j ∈ {U, S}

# Returns
- `converged :: Bool`: True if max distance is below tolerance
- `dist :: Float64`: Maximum distance (supremum norm) across both tightness paths
"""
function check_convergence(theta_U_old::Vector, theta_S_old::Vector,
                          path::TransitionPath, tp::TransitionParams)
    dist_U = maximum(abs.(path.theta_U .- theta_U_old))
    dist_S = maximum(abs.(path.theta_S .- theta_S_old))
    dist = max(dist_U, dist_S)

    converged = (dist < tp.tol)
    return converged, dist
end

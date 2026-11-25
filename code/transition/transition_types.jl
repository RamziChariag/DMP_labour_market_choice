############################################################
# transition_types.jl — every type's values and masses at date 0 of a transition
#
# Library file, loaded by transition_main.jl and counterfactuals_main.jl. `date0_types`
# gives, per type (a_U, a_S) and state, the worker's value and the mass in that state at
# date 0 (after the date-0 reallocation) and at the baseline steady state. The per-type
# surfaces exist only inside the backward pass, so they are recomputed here. p-indexed
# arrays are Float32 (most of the size); the unskilled ones carry the atom p = 1 last.
############################################################

"""
    type_values(path, buf, model) -> NamedTuple

Per-type worker values at the first date of `path` in every state, from the path and the
per-type surfaces in slot 1 of `buf` (`_backward_pass!`), the one-dimensional surfaces where
the type is pure. E = outside option + β S⁺:
  UC      unskilled unemployed (searching slice)    T, Utr  in training, −c + T
  EUs_p   unskilled employed, searching slice, by p  EUt_p  training slice (NaN: none)
  U0      trained, searching the skilled market      U1     trained, searching the unskilled market
  US      max{U0, U1}, the trained unemployed        ES_p   skilled employed by p
"""
function type_values(path::TransitionPath, buf, model::Model)
    Nx = length(model.grids.x);  NpU = length(model.unsk_grids.p);  NpS = length(model.skl_grids.p)
    βU = model.unsk_par.β;  βS = model.skl_par.β
    UC = path.UC[:, :, 1];  T = path.T[:, :, 1];  Utr = path.Utr[:, :, 1]
    U0 = path.U0[:, :, 1];  U1 = path.U1[:, :, 1];  US = max.(U0, U1)
    EUs = fill(NaN32, Nx, Nx, NpU + 1);  EUt = fill(NaN32, Nx, Nx, NpU + 1)
    ES  = fill(NaN32, Nx, Nx, NpS)
    @inbounds for j in 1:Nx, i in 1:Nx
        Ss = buf.hasS[i, j, 1] ? view(buf.Ss, i, j, :, 1) : view(path.SU, i, :, 1)
        for q in 1:NpU
            EUs[i, j, q] = UC[i, j] + βU * max(Ss[q], 0.0)
        end
        EUs[i, j, NpU + 1] = UC[i, j] + βU * max(path.SUs1[i, j, 1], 0.0)
        if buf.hasT[i, j, 1]
            for q in 1:NpU
                EUt[i, j, q] = Utr[i, j] + βU * max(buf.St[i, j, q, 1], 0.0)
            end
            EUt[i, j, NpU + 1] = Utr[i, j] + βU * max(path.SUt1[i, j, 1], 0.0)
        end
        S0 = buf.hasK[i, j, 1] ? view(buf.K0, i, j, :, 1) : view(path.S0, j, :, 1)
        S1 = buf.hasK[i, j, 1] ? view(buf.K1, i, j, :, 1) : view(path.S1, j, :, 1)
        for q in 1:NpS
            ES[i, j, q] = US[i, j] + βS * max(0.0, S0[q], S1[q])
        end
    end
    return (UC = UC, T = T, Utr = Utr, U0 = U0, U1 = U1, US = US, EUs_p = EUs, EUt_p = EUt,
            ES_p = ES, tau = path.τT[:, :, 1])
end

"""
    type_density(st, d, model) -> NamedTuple

The masses per type and state of the forward state `st` (`_forward_state`, after a
reallocation) with crossing fractions `d`: the crossers are the share d of the trained
unemployed, and a cell's skilled employment is its non-draining mass (1 − d) m_S spread
over p by the column's unit shape ê_S, as `path_statistics` reads it.
"""
function type_density(st, d::Matrix{Float64}, model::Model)
    sg = model.skl_grids;  Nx = length(model.grids.x);  NpS = length(sg.p)
    eSp = zeros(Float32, Nx, Nx, NpS)
    @inbounds for j in 1:Nx, i in 1:Nx, q in 1:NpS
        eSp[i, j, q] = (1.0 - d[i, j]) * st.mS[i, j] * st.eS[j, q] * sg.wp[q]
    end
    return (uU = copy(st.uU), nU = copy(st.nU), uS_cross = d .* st.uS,
            uS_skilled = (1.0 .- d) .* st.uS, eUs_p = Float32.(st.Es), eUt_p = Float32.(st.Et),
            eS_p = eSp, d = copy(d))
end

"""
    _date1_types(path, model, tp) -> (values, density)

Values and masses per type at the first date of `path` under `model`: one backward pass at
the path's tightness (it reproduces the values of a converged path; it overwrites the
path's values, so pass a copy), then the first reallocation.
"""
function _date1_types(path::TransitionPath, model::Model, tp::TransitionParams)
    buf = _backward_pass!(path, deepcopy(model), tp).buf
    st  = _forward_state(path)
    d   = _d_matrix(path, model, 1)
    _reallocate!(st, path, model, hire_fraction(model), d, 1)
    return (values = type_values(path, buf, model), density = type_density(st, d, model))
end

"""
    date0_types(path, model_z0, model_z1, tp) -> NamedTuple

Every type's value and mass in every state at date 0 of `path` (computed on a copy) and at
the baseline steady state (its self-transition), with the two levels of A for the
normalisation (V_0/A_crisis)/(V_base/A_base).
"""
function date0_types(path::TransitionPath, model_z0::Model, model_z1::Model, tp::TransitionParams)
    at0 = _date1_types(deepcopy(path), model_z1, tp)
    tb  = TransitionParams(T_max = 0.5, N_steps = 1, maxit = 1, verbose = false)
    _, pb = solve_transition_path(deepcopy(model_z0), deepcopy(model_z0), tb; scenario = :steady_state)
    base = _date1_types(pb, model_z0, tb)
    return (values_date0 = at0.values, density_date0 = at0.density,
            values_base = base.values, density_base = base.density,
            A_base = exp(model_z0.common.A), A_crisis = exp(model_z1.common.A))
end

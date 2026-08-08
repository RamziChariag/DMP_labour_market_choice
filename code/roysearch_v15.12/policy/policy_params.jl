############################################################
# policy_params.jl — Policy definitions and result storage
#
# Contains:
#   PolicySpec      — describes one policy experiment
#   PolicyResult    — equilibrium outcomes for one (policy, intensity) pair
#   PolicyTable     — full results table across all experiments
#
# Education subsidies:
#   Policy A:  raise bT  by δ percent   →  bT_new = bT * (1 + δ)
#   Policy B:  lower c(x) by δ percent  →  c_new  = c + log(1 − δ)
#              cost function:  c(x) = exp(c) · (1−x) · e^{−x}
#
# Outside options:
#   Policy bU:  raise bU  by δ percent
#   Policy bS:  raise bS  by δ percent
#   Policy bUS: raise both bU and bS by δ percent
############################################################


# ──────────────────────────────────────────────────────────
# PolicySpec — a single counterfactual experiment
# ──────────────────────────────────────────────────────────

"""
    PolicySpec

Defines one policy experiment.

# Fields
- `label`     : human-readable name (e.g. "Policy A: +25% bT")
- `policy`    : `:A`, `:B`, `:bU`, `:bS`, `:bUS`, or `:baseline`
- `intensity` : subsidy intensity δ ∈ (0, 1), e.g. 0.10 = 10%
"""
struct PolicySpec
    label     :: String
    policy    :: Symbol      # :A or :B
    intensity :: Float64     # δ ∈ (0,1)
end


"""
    build_policy_experiments(; intensities)

Return a vector of `PolicySpec` for both policies at each intensity level,
plus a baseline (no subsidy) entry.
"""
function build_policy_experiments(;
    intensities :: Vector{Float64} = [0.10, 0.25, 0.50]
)
    specs = PolicySpec[]

    # Baseline (no policy)
    push!(specs, PolicySpec("Baseline", :baseline, 0.0))

    # Policy A: raise bT
    for δ in intensities
        pct = round(Int, 100 * δ)
        push!(specs, PolicySpec("bT +$(pct)%", :A, δ))
    end

    # Policy B: lower c(x)
    for δ in intensities
        pct = round(Int, 100 * δ)
        push!(specs, PolicySpec("c(x) −$(pct)%", :B, δ))
    end

    return specs
end


"""
    build_outside_option_experiments(; intensities)

Return a vector of `PolicySpec` for the outside-option exercise:
bU alone, bS alone, and both together, at each intensity level,
plus a baseline entry.
"""
function build_outside_option_experiments(;
    intensities :: Vector{Float64} = [0.01, 0.05, 0.10, 0.20]
)
    specs = PolicySpec[]

    push!(specs, PolicySpec("Baseline", :baseline, 0.0))

    for δ in intensities
        pct = round(Int, 100 * δ)
        push!(specs, PolicySpec("bU +$(pct)%", :bU, δ))
    end

    for δ in intensities
        pct = round(Int, 100 * δ)
        push!(specs, PolicySpec("bS +$(pct)%", :bS, δ))
    end

    for δ in intensities
        pct = round(Int, 100 * δ)
        push!(specs, PolicySpec("bU,bS +$(pct)%", :bUS, δ))
    end

    return specs
end


# ──────────────────────────────────────────────────────────
# Parameter perturbation
# ──────────────────────────────────────────────────────────

"""
    perturb_params(regime, common, spec) → (regime_new, common_new)

Apply the policy perturbation defined by `spec` to the regime and common
parameter structs.  Returns new (immutable) copies.

Education subsidies:
- Policy A:   `bT_new = bT * (1 + δ)`
- Policy B:   `c_new  = c + log(1 − δ)`

Outside options:
- Policy bU:  `bU_new = bU * (1 + δ)`
- Policy bS:  `bS_new = bS * (1 + δ)`
- Policy bUS: both bU and bS raised by δ
"""
function perturb_params(
    regime :: RegimeParams,
    common :: CommonParams,
    spec   :: PolicySpec,
)
    δ = spec.intensity

    if spec.policy == :A
        regime_new = RegimeParams(
            PU = regime.PU, PS = regime.PS,
            bU = regime.bU, bT = regime.bT * (1.0 + δ), bS = regime.bS,
            α_U = regime.α_U, a_Γ = regime.a_Γ, b_Γ = regime.b_Γ,
        )
        return regime_new, common

    elseif spec.policy == :B
        c_new = common.c + log(1.0 - δ)
        common_new = CommonParams(
            r = common.r, ν = common.ν, φ = common.φ,
            a_ℓ = common.a_ℓ, b_ℓ = common.b_ℓ, c = c_new,
        )
        return regime, common_new

    elseif spec.policy == :bU
        regime_new = RegimeParams(
            PU = regime.PU, PS = regime.PS,
            bU = regime.bU * (1.0 + δ), bT = regime.bT, bS = regime.bS,
            α_U = regime.α_U, a_Γ = regime.a_Γ, b_Γ = regime.b_Γ,
        )
        return regime_new, common

    elseif spec.policy == :bS
        regime_new = RegimeParams(
            PU = regime.PU, PS = regime.PS,
            bU = regime.bU, bT = regime.bT, bS = regime.bS * (1.0 + δ),
            α_U = regime.α_U, a_Γ = regime.a_Γ, b_Γ = regime.b_Γ,
        )
        return regime_new, common

    elseif spec.policy == :bUS
        regime_new = RegimeParams(
            PU = regime.PU, PS = regime.PS,
            bU = regime.bU * (1.0 + δ), bT = regime.bT, bS = regime.bS * (1.0 + δ),
            α_U = regime.α_U, a_Γ = regime.a_Γ, b_Γ = regime.b_Γ,
        )
        return regime_new, common

    else
        # Baseline — no perturbation
        return regime, common
    end
end


# ──────────────────────────────────────────────────────────
# PolicyResult — outcomes from one solved equilibrium
# ──────────────────────────────────────────────────────────

"""
    PolicyResult

Stores the key equilibrium outcomes for one (policy, intensity) experiment.
Everything needed for tables and figures in Section 6.
"""
struct PolicyResult
    # ── Identification ────────────────────────────────────
    spec        :: PolicySpec
    converged   :: Bool

    # ── Training cutoff ───────────────────────────────────
    x_bar       :: Float64       # training cutoff (first x where τ(x) = 1)

    # ── Population accounting ─────────────────────────────
    agg_uU      :: Float64       # aggregate unskilled unemployed
    agg_t       :: Float64       # aggregate training stock
    agg_eU      :: Float64       # aggregate unskilled employed
    agg_mU      :: Float64       # unskilled segment mass
    agg_uS      :: Float64       # aggregate skilled unemployed
    agg_eS      :: Float64       # aggregate skilled employed
    agg_mS      :: Float64       # skilled segment mass

    # ── Rates ─────────────────────────────────────────────
    ur_U        :: Float64       # unskilled unemployment rate
    ur_S        :: Float64       # skilled unemployment rate
    ur_total    :: Float64       # overall unemployment rate
    skilled_share :: Float64     # skilled segment share of population
    training_share :: Float64    # training stock share of population

    # ── Tightness ─────────────────────────────────────────
    θU          :: Float64
    θS          :: Float64

    # ── Contact rates ─────────────────────────────────────
    f_U         :: Float64       # unskilled job-finding rate
    f_S         :: Float64       # skilled job-finding rate
    sep_rate_U  :: Float64       # unskilled separation rate
    sep_rate_S  :: Float64       # skilled separation rate
    ee_rate_S   :: Float64       # skilled EE rate

    # ── Wages ─────────────────────────────────────────────
    mean_wage_U :: Float64
    mean_wage_S :: Float64
    wage_premium :: Float64      # log(mean_wage_S / mean_wage_U)

    # ── Welfare ───────────────────────────────────────────
    # Value of unemployment at selected ability quantiles
    UU_x25      :: Float64       # U_U(x) at 25th percentile of x
    UU_x50      :: Float64       # U_U(x) at median x
    UU_x75      :: Float64       # U_U(x) at 75th percentile x
    US_x25      :: Float64       # U_S(x) at 25th percentile
    US_x50      :: Float64       # U_S(x) at median
    US_x75      :: Float64       # U_S(x) at 75th percentile
end


"""
    PolicyTable

Complete results across all experiments for one baseline.
"""
struct PolicyTable
    baseline_label :: String              # e.g. "Pre-FC" or "Pre-COVID"
    results        :: Vector{PolicyResult}
end

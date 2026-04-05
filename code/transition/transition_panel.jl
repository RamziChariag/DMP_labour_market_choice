############################################################
# transition_panel.jl — 2×3 time-series panel plot
#
# Generates one 3×2 panel figure per episode (FC, COVID).
# Each panel overlays:
#   - Data line (black solid): actual monthly time series
#   - SE band (light gray, semi-transparent): ±1.96 × SE
#   - Model line (steelblue solid): flat at baseline SS
#     before the switch, then follows transition path
#   - Vertical line (firebrick dashed): regime switch date
#
# Usage (from project root):
#   julia code/transition/transition_panel.jl
#
# Outputs:
#   output/plots/transition_panel_fc_fullW.png
#   output/plots/transition_panel_covid_fullW.png
############################################################

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

const SHOW_SE_BANDS = false

# Weight-matrix suffix (must match transition output files)
const W_COND_TARGET = 1e8
const W_SUFFIX      = W_COND_TARGET == 0.0 ? "_diagonalW"    :
                      W_COND_TARGET == 1.0 ? "_compressedW"  :
                      W_COND_TARGET == 2.0 ? "_equalW"       : "_fullW"

# ═══════════════════════════════════════════════════════════
# Paths (script lives in code/transition/)
# ═══════════════════════════════════════════════════════════

const TRANSITION_DIR = @__DIR__
const PROJECT_ROOT   = joinpath(TRANSITION_DIR, "..", "..")
const DERIVED_DIR    = joinpath(PROJECT_ROOT, "data", "derived")
const TRANS_OUT_DIR  = joinpath(PROJECT_ROOT, "output", "transition")
const PLOTS_DIR      = joinpath(PROJECT_ROOT, "output", "plots")

# ═══════════════════════════════════════════════════════════
# Packages
# ═══════════════════════════════════════════════════════════

print("Loading packages... "); flush(stdout)

using LinearAlgebra
using Plots
using LaTeXStrings
using JLD2
using CSV
using DataFrames
using Arrow
using Printf
using Statistics
using Dates

println("done."); flush(stdout)

# ═══════════════════════════════════════════════════════════
# TransitionResult struct definition
#
# Copied from transition_params.jl so this script can run
# without loading the full solver (which requires Model etc.).
# Keep in sync with transition_params.jl.
# ═══════════════════════════════════════════════════════════

# Load TransitionParams and TransitionResult from transition_params.jl.
# We need a stub for Model to avoid a load error from the TransitionPath
# constructor signature — it is never called here.
if !isdefined(Main, :TransitionResult)
    # Define a minimal Model stub so that transition_params.jl can be included
    # (the TransitionPath constructor references Model but we never call it).
    if !isdefined(Main, :Model)
        struct Model end
    end
    include(joinpath(TRANSITION_DIR, "transition_params.jl"))
end

# ═══════════════════════════════════════════════════════════
# Shared plot theme (matches existing code)
# ═══════════════════════════════════════════════════════════

function _set_theme!()
    gr()
    theme(:default)
    default(
        fontfamily    = "Computer Modern",
        framestyle    = :box,
        titlefontsize = 11,
        guidefontsize = 10,
        tickfontsize  = 8,
        legendfontsize= 8,
        linewidth     = 1.8,
        grid          = true,
        gridalpha     = 0.05,
        left_margin   = 7Plots.mm,
    )
end

_C1_PANEL = :steelblue   # model line colour
_C2_PANEL = :firebrick   # switch-date vertical line

# ═══════════════════════════════════════════════════════════
# Episode calendar definitions
# ═══════════════════════════════════════════════════════════

"""
    EpisodeCalendar

Holds the calendar metadata for one episode.

Fields:
  start_date   first month of the episode (Date, day=1)
  switch_date  regime-switch month
  end_date     last month of the episode
  base_window  Symbol for baseline window (e.g. :base_fc)
  crisis_window Symbol for crisis window  (e.g. :crisis_fc)
  n_base       number of pre-switch months
  n_crisis     number of post-switch months (= crisis window length)
"""
struct EpisodeCalendar
    start_date    :: Date
    switch_date   :: Date
    end_date      :: Date
    base_window   :: Symbol
    crisis_window :: Symbol
    n_base        :: Int     # months before switch
    n_crisis      :: Int     # months from switch to end (inclusive)
end

"""
    episode_calendar(scenario) → EpisodeCalendar

Return the calendar for :fc or :covid.
"""
function episode_calendar(scenario::Symbol) :: EpisodeCalendar
    if scenario == :fc
        # FC: Jan 2003 – Jun 2009, switch Jan 2008
        start_d  = Date(2003, 1, 1)
        switch_d = Date(2008, 1, 1)
        end_d    = Date(2009, 6, 1)
        return EpisodeCalendar(start_d, switch_d, end_d,
                               :base_fc, :crisis_fc,
                               60, 18)
    elseif scenario == :covid
        # COVID: Jan 2015 – Dec 2021, switch Mar 2020
        start_d  = Date(2015, 1, 1)
        switch_d = Date(2020, 3, 1)
        end_d    = Date(2021, 12, 1)
        return EpisodeCalendar(start_d, switch_d, end_d,
                               :base_covid, :crisis_covid,
                               60, 22)
    else
        error("Unknown scenario: $scenario. Must be :fc or :covid.")
    end
end

"""
    monthly_range(start_date, end_date) → Vector{Date}

Return a vector of first-of-month Dates from start_date to end_date inclusive.
"""
function monthly_range(start_date::Date, end_date::Date) :: Vector{Date}
    dates = Date[]
    d = start_date
    while d <= end_date
        push!(dates, d)
        d = d + Month(1)
    end
    return dates
end

# ═══════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════

"""
    load_sigma_ses(; window, derived_dir) → Dict{Symbol, Float64}

Load sigma_{window}.csv, return Dict mapping MOMENT_NAME → standard error
(= sqrt(diagonal element)).  The CSV is a K×K covariance matrix with
column names matching moment names.
"""
function load_sigma_ses(; window::Symbol, derived_dir::String) :: Dict{Symbol,Float64}
    sigma_file = joinpath(derived_dir, "sigma_$(window).csv")
    if !isfile(sigma_file)
        @warn "sigma file not found: $sigma_file — SE bands will be suppressed for $window"
        return Dict{Symbol,Float64}()
    end
    df  = CSV.read(sigma_file, DataFrame)
    col_names = Symbol.(names(df))
    Σ   = Matrix{Float64}(df)
    se  = Dict{Symbol,Float64}()
    for (j, nm) in enumerate(col_names)
        se[nm] = sqrt(max(Σ[j, j], 0.0))
    end
    return se
end

"""
    load_moments_values(; window, derived_dir) → Dict{Symbol, Float64}

Load moments_{window}.csv, return Dict mapping moment name → value.
"""
function load_moments_values(; window::Symbol, derived_dir::String) :: Dict{Symbol,Float64}
    mom_file = joinpath(derived_dir, "moments_$(window).csv")
    if !isfile(mom_file)
        @warn "moments file not found: $mom_file"
        return Dict{Symbol,Float64}()
    end
    df  = CSV.read(mom_file, DataFrame)
    out = Dict{Symbol,Float64}()
    for row in eachrow(df)
        out[Symbol(row.moment)] = Float64(row.value)
    end
    return out
end


"""
    load_cps_monthly(cal, derived_dir)
        → Dict{Tuple{Int,Int}, NamedTuple}

Read cps_basic_clean.arrow and compute monthly aggregates for every
(YEAR, MONTH) in the episode calendar.  Returns a Dict keyed by (year, month)
with fields:
  ur_total, skilled_share, training_share, u_U, u_S, lf_U, lf_S
All quantities are weighted using WTFINL.
"""
function load_cps_monthly(cal::EpisodeCalendar, derived_dir::String)
    arrow_file = joinpath(derived_dir, "cps_basic_clean.arrow")
    if !isfile(arrow_file)
        @warn "CPS arrow file not found: $arrow_file — data lines will be flat moments only"
        return Dict{Tuple{Int,Int}, NamedTuple}()
    end

    df = DataFrame(Arrow.Table(arrow_file))

    # Determine available columns (in_training may be absent)
    has_training = "in_training" in names(df)
    has_skilled  = "skilled"     in names(df)

    dates    = monthly_range(cal.start_date, cal.end_date)
    out      = Dict{Tuple{Int,Int}, NamedTuple}()

    for d in dates
        yr, mo = year(d), month(d)
        mask = (df.YEAR .== yr) .& (df.MONTH .== mo)
        sub  = df[mask, :]
        isempty(sub) && continue

        w       = Float64.(sub.WTFINL)
        emp     = Bool.(sub.employed)
        unemp   = Bool.(sub.unemployed)
        in_lf   = Bool.(sub.in_lf)

        if has_skilled
            skl = Bool.(sub.skilled)
        else
            skl = fill(false, nrow(sub))
        end

        # Aggregate unemployment rate
        lf_w = sum(w[in_lf])
        if lf_w < 1.0
            continue
        end

        ur_total = sum(w[unemp]) / lf_w

        # Skilled share (share of labour force that is skilled)
        skilled_share = has_skilled ? sum(w[skl .& in_lf]) / lf_w : NaN

        # Training share (share of labour force in training)
        if has_training
            in_tr = Bool.(sub.in_training)
            training_share = sum(w[in_tr .& in_lf]) / lf_w
        else
            training_share = NaN
        end

        # Segment-level unemployment masses (for tightness computation)
        if has_skilled
            mask_U_lf = .!skl .& in_lf
            mask_S_lf =  skl  .& in_lf
            lf_U = sum(w[mask_U_lf])
            lf_S = sum(w[mask_S_lf])
            u_U  = sum(w[unemp .& .!skl])
            u_S  = sum(w[unemp .&  skl])
        else
            lf_U = lf_w; lf_S = 0.0
            u_U  = sum(w[unemp]); u_S = 0.0
        end

        out[(yr, mo)] = (
            ur_total      = ur_total,
            skilled_share = skilled_share,
            training_share= training_share,
            u_U           = u_U,
            u_S           = u_S,
            lf_U          = lf_U,
            lf_S          = lf_S,
        )
    end

    return out
end


"""
    load_jolts_monthly(cal, cps_monthly, derived_dir)
        → (theta_U::Dict{Tuple{Int,Int},Float64},
           theta_S::Dict{Tuple{Int,Int},Float64})

Read jolts_clean.arrow and compute θ_U and θ_S for each (year, month) in
the episode calendar.  Uses CPS-weighted unemployment counts from cps_monthly
as the denominator.
If the JOLTS file is absent, returns empty Dicts.
"""
function load_jolts_monthly(cal::EpisodeCalendar,
                             cps_monthly::Dict{Tuple{Int,Int}, NamedTuple},
                             derived_dir::String)
    jolts_file = joinpath(derived_dir, "jolts_clean.arrow")
    if !isfile(jolts_file)
        @warn "JOLTS arrow file not found: $jolts_file — tightness data will use moments CSV"
        return Dict{Tuple{Int,Int},Float64}(), Dict{Tuple{Int,Int},Float64}()
    end

    df = DataFrame(Arrow.Table(jolts_file))

    # Identify vacancy columns — look for V_U, V_S, or total vacancies
    col_names = Symbol.(names(df))
    has_V_U = :V_U in col_names
    has_V_S = :V_S in col_names
    has_V   = :V   in col_names || :vacancies in col_names || :V_total in col_names

    θU_dict = Dict{Tuple{Int,Int},Float64}()
    θS_dict = Dict{Tuple{Int,Int},Float64}()

    dates = monthly_range(cal.start_date, cal.end_date)

    for d in dates
        yr, mo = year(d), month(d)

        # Match by YEAR + MONTH columns
        if !(:YEAR in col_names && :MONTH in col_names)
            # Try year/month (lowercase) or year_month etc.
            yr_col = findfirst(s -> lowercase(string(s)) in ["year", "yr"], col_names)
            mo_col = findfirst(s -> lowercase(string(s)) in ["month", "mo"], col_names)
            (isnothing(yr_col) || isnothing(mo_col)) && break
            mask = (df[!, col_names[yr_col]] .== yr) .& (df[!, col_names[mo_col]] .== mo)
        else
            mask = (df.YEAR .== yr) .& (df.MONTH .== mo)
        end

        sub = df[mask, :]
        isempty(sub) && continue
        row = sub[1, :]   # take first matching row

        # Get CPS unemployment counts
        cps = get(cps_monthly, (yr, mo), nothing)
        isnothing(cps) && continue

        # θ_U = V_U / U_U
        if has_V_U && cps.u_U > 0
            V_U = Float64(row[:V_U])
            θU_dict[(yr, mo)] = V_U / cps.u_U
        end

        # θ_S = V_S / U_S
        if has_V_S && cps.u_S > 0
            V_S = Float64(row[:V_S])
            θS_dict[(yr, mo)] = V_S / cps.u_S
        end

        # Fallback: if only total vacancies available, split proportionally
        if !has_V_U && has_V
            V_col = :V in col_names ? :V :
                    :vacancies in col_names ? :vacancies : :V_total
            V_tot = Float64(row[V_col])
            u_tot = cps.u_U + cps.u_S
            if u_tot > 0
                θ_tot = V_tot / u_tot
                θU_dict[(yr, mo)] = θ_tot
                θS_dict[(yr, mo)] = θ_tot
            end
        end
    end

    return θU_dict, θS_dict
end


# ═══════════════════════════════════════════════════════════
# Model series construction
# ═══════════════════════════════════════════════════════════

"""
    build_model_series(result, cal)
        → (dates, ur, θU, θS, wage_U, wage_S, train_sh, skilled_sh)

Splice the steady-state flat line (pre-switch) with the transition path
(post-switch) for all six variables.  Returns vectors aligned to the
episode monthly calendar (length = total months in episode).

The model tgrid has dt=0.5 (half-month steps); we interpolate onto
monthly resolution for the crisis window.
"""
function build_model_series(result, cal::EpisodeCalendar)
    dates  = monthly_range(cal.start_date, cal.end_date)
    N      = length(dates)

    # Baseline SS value = t=0 of the transition result
    # (The transition solver starts from the baseline SS)
    ss_ur        = result.ur_total[1]
    ss_θU        = result.θU[1]
    ss_θS        = result.θS[1]
    ss_wage_U    = result.mean_wage_U[1]
    ss_wage_S    = result.mean_wage_S[1]
    ss_train_sh  = result.training_share[1]
    ss_skilled_sh= result.skilled_share[1]

    # Transition path grid: tgrid[1] = 0 corresponds to switch date.
    # Map model "months" 0..n_crisis onto crisis calendar months.
    # tgrid is in model months (dt = 0.5), Nt = 241 for 120 model months.
    tgrid = result.tgrid     # length Nt, 0 to T_max (120 months by default)
    Nt    = length(tgrid)

    # Build interpolation functions for each series over tgrid
    # We use linear interpolation; at month k (0-indexed from switch),
    # evaluate the transition at model time k.
    function interp_at_month(series::Vector{Float64}, month_offset::Float64)
        # month_offset ∈ [0, n_crisis-1]
        # tgrid spans 0..T_max; index in tgrid corresponding to month_offset
        t_query = clamp(month_offset, tgrid[1], tgrid[end])
        # find the two bracketing indices
        idx = searchsortedfirst(tgrid, t_query)
        if idx == 1
            return series[1]
        elseif idx > Nt
            return series[Nt]
        else
            t_lo = tgrid[idx-1]
            t_hi = tgrid[idx]
            α    = (t_hi > t_lo) ? (t_query - t_lo) / (t_hi - t_lo) : 0.0
            return series[idx-1] * (1.0 - α) + series[idx] * α
        end
    end

    # Find the index in dates where the switch occurs
    switch_idx = findfirst(d -> d >= cal.switch_date, dates)
    isnothing(switch_idx) && (switch_idx = N + 1)  # switch after episode

    # Allocate output arrays
    model_ur        = zeros(N)
    model_θU        = zeros(N)
    model_θS        = zeros(N)
    model_wage_U    = zeros(N)
    model_wage_S    = zeros(N)
    model_train_sh  = zeros(N)
    model_skilled_sh= zeros(N)

    for i in 1:N
        if i < switch_idx
            # Pre-switch: flat at baseline SS
            model_ur[i]         = ss_ur
            model_θU[i]         = ss_θU
            model_θS[i]         = ss_θS
            model_wage_U[i]     = ss_wage_U
            model_wage_S[i]     = ss_wage_S
            model_train_sh[i]   = ss_train_sh
            model_skilled_sh[i] = ss_skilled_sh
        else
            # Post-switch: interpolate transition path
            # month_offset = 0 at the switch date
            month_off = Float64(i - switch_idx)

            model_ur[i]         = interp_at_month(result.ur_total,       month_off)
            model_θU[i]         = interp_at_month(result.θU,             month_off)
            model_θS[i]         = interp_at_month(result.θS,             month_off)
            model_wage_U[i]     = interp_at_month(result.mean_wage_U,    month_off)
            model_wage_S[i]     = interp_at_month(result.mean_wage_S,    month_off)
            model_train_sh[i]   = interp_at_month(result.training_share, month_off)
            model_skilled_sh[i] = interp_at_month(result.skilled_share,  month_off)
        end
    end

    return dates, model_ur, model_θU, model_θS,
           model_wage_U, model_wage_S, model_train_sh, model_skilled_sh
end


# ═══════════════════════════════════════════════════════════
# Data series assembly
# ═══════════════════════════════════════════════════════════

"""
    build_data_series(cal, cps_monthly, θU_dict, θS_dict,
                      base_moms, crisis_moms,
                      base_ses, crisis_ses)
        → NamedTuple of (values, ses) per variable, aligned to calendar

Returns a NamedTuple with fields:
  ur_total, theta_U, theta_S, mean_wage_U, mean_wage_S,
  training_share, skilled_share
Each field is itself a NamedTuple with:
  values :: Vector{Float64}   (NaN where not available)
  ses    :: Vector{Float64}   (piecewise constant by window)
"""
function build_data_series(
    cal           :: EpisodeCalendar,
    cps_monthly   :: Dict{Tuple{Int,Int}, NamedTuple},
    θU_dict       :: Dict{Tuple{Int,Int}, Float64},
    θS_dict       :: Dict{Tuple{Int,Int}, Float64},
    base_moms     :: Dict{Symbol,Float64},
    crisis_moms   :: Dict{Symbol,Float64},
    base_ses      :: Dict{Symbol,Float64},
    crisis_ses    :: Dict{Symbol,Float64},
)
    dates  = monthly_range(cal.start_date, cal.end_date)
    N      = length(dates)

    # ── SE piecewise constant: baseline before switch, crisis after ──────
    # For moment name `nm`, SE at month i:
    function get_se(nm::Symbol, i::Int) :: Float64
        d = dates[i]
        w = d < cal.switch_date ? base_ses : crisis_ses
        get(w, nm, NaN)
    end

    # ── Unemployment rate (from CPS) ─────────────────────────────────────
    ur_vals = fill(NaN, N)
    ur_ses  = fill(NaN, N)
    for i in 1:N
        d = dates[i]
        row = get(cps_monthly, (year(d), month(d)), nothing)
        if !isnothing(row)
            ur_vals[i] = row.ur_total
        else
            # Fallback: use window moment
            mom = d < cal.switch_date ? base_moms : crisis_moms
            ur_vals[i] = get(mom, :ur_total, NaN)
        end
        ur_ses[i] = get_se(:ur_total, i)
    end

    # ── Tightness: θ_U and θ_S (from JOLTS + CPS) ────────────────────────
    θU_vals = fill(NaN, N)
    θS_vals = fill(NaN, N)
    θU_ses  = fill(NaN, N)
    θS_ses  = fill(NaN, N)
    for i in 1:N
        d = dates[i]
        k = (year(d), month(d))
        θU_vals[i] = get(θU_dict, k, NaN)
        θS_vals[i] = get(θS_dict, k, NaN)
        # Fallback to moment CSV if JOLTS not available
        if isnan(θU_vals[i])
            mom = d < cal.switch_date ? base_moms : crisis_moms
            θU_vals[i] = get(mom, :theta_U, NaN)
        end
        if isnan(θS_vals[i])
            mom = d < cal.switch_date ? base_moms : crisis_moms
            θS_vals[i] = get(mom, :theta_S, NaN)
        end
        θU_ses[i] = get_se(:theta_U, i)
        θS_ses[i] = get_se(:theta_S, i)
    end

    # ── Wages: annual from ASEC → flat window-average (moment CSV) ────────
    # Show as flat line at window-average; SE band wraps around it.
    wage_U_vals = fill(NaN, N)
    wage_S_vals = fill(NaN, N)
    wage_U_ses  = fill(NaN, N)
    wage_S_ses  = fill(NaN, N)
    for i in 1:N
        d = dates[i]
        mom = d < cal.switch_date ? base_moms : crisis_moms
        ses = d < cal.switch_date ? base_ses  : crisis_ses
        wage_U_vals[i] = get(mom, :mean_wage_U, NaN)
        wage_S_vals[i] = get(mom, :mean_wage_S, NaN)
        wage_U_ses[i]  = get(ses, :mean_wage_U, NaN)
        wage_S_ses[i]  = get(ses, :mean_wage_S, NaN)
    end

    # ── Training share (from CPS) ─────────────────────────────────────────
    train_vals = fill(NaN, N)
    train_ses  = fill(NaN, N)
    for i in 1:N
        d = dates[i]
        row = get(cps_monthly, (year(d), month(d)), nothing)
        if !isnothing(row) && !isnan(row.training_share)
            train_vals[i] = row.training_share
        else
            mom = d < cal.switch_date ? base_moms : crisis_moms
            train_vals[i] = get(mom, :training_share, NaN)
        end
        train_ses[i] = get_se(:training_share, i)
    end

    # ── Skilled share (from CPS) ─────────────────────────────────────────
    sksh_vals = fill(NaN, N)
    sksh_ses  = fill(NaN, N)
    for i in 1:N
        d = dates[i]
        row = get(cps_monthly, (year(d), month(d)), nothing)
        if !isnothing(row) && !isnan(row.skilled_share)
            sksh_vals[i] = row.skilled_share
        else
            mom = d < cal.switch_date ? base_moms : crisis_moms
            sksh_vals[i] = get(mom, :skilled_share, NaN)
        end
        sksh_ses[i] = get_se(:skilled_share, i)
    end

    return (
        ur_total     = (values = ur_vals,    ses = ur_ses),
        theta_U      = (values = θU_vals,    ses = θU_ses),
        theta_S      = (values = θS_vals,    ses = θS_ses),
        mean_wage_U  = (values = wage_U_vals, ses = wage_U_ses),
        mean_wage_S  = (values = wage_S_vals, ses = wage_S_ses),
        training_share=(values = train_vals,  ses = train_ses),
        skilled_share = (values = sksh_vals,  ses = sksh_ses),
    )
end


# ═══════════════════════════════════════════════════════════
# Panel construction helper
# ═══════════════════════════════════════════════════════════

"""
    make_one_panel(dates, data_series, model_series,
                   switch_date, title_str;
                   show_legend, show_xlabel, show_se_bands)
        → Plots.Plot

Build one subplot for the 3×2 panel.
"""
function make_one_panel(
    dates        :: Vector{Date},
    data_vals    :: Vector{Float64},
    data_ses     :: Vector{Float64},
    model_vals   :: Vector{Float64},
    switch_date  :: Date,
    title_str    :: String;
    show_legend  :: Bool = false,
    show_xlabel  :: Bool = false,
    show_se_bands:: Bool = SHOW_SE_BANDS,
)
    N  = length(dates)

    # Use numeric month index as x-axis (1, 2, ..., N)
    xs = 1:N

    # Tick positions and labels: show year boundaries
    tick_pos   = Int[]
    tick_label = String[]
    cur_year   = year(dates[1])
    for i in 1:N
        if year(dates[i]) != cur_year || i == 1
            push!(tick_pos, i)
            push!(tick_label, string(year(dates[i])))
            cur_year = year(dates[i])
        end
    end

    p = plot(;
        title        = title_str,
        xlabel       = show_xlabel ? "Month" : "",
        ylabel       = "",
        legend       = show_legend ? :topright : false,
        bottom_margin= 3Plots.mm,
        top_margin   = 3Plots.mm,
        xticks       = (tick_pos, tick_label),
        xrotation    = 45,
    )

    # ── SE band (light gray, semi-transparent, wraps data line) ──────────
    if show_se_bands
        finite_mask = .!isnan.(data_vals) .& .!isnan.(data_ses)
        if any(finite_mask)
            band_lo = ifelse.(finite_mask, data_vals .- 1.96 .* data_ses, NaN)
            band_hi = ifelse.(finite_mask, data_vals .+ 1.96 .* data_ses, NaN)
            plot!(p, collect(xs), band_lo;
                  fillrange  = band_hi,
                  fillalpha  = 0.20,
                  fillcolor  = :lightgray,
                  linewidth  = 0,
                  linecolor  = :lightgray,
                  label      = show_legend ? "95% CI" : "")
        end
    end

    # ── Data line (black solid) ───────────────────────────────────────────
    finite_data = any(.!isnan.(data_vals))
    if finite_data
        plot!(p, collect(xs), data_vals;
              color     = :black,
              linewidth = 1.8,
              label     = show_legend ? "Data" : "")
    end

    # ── Model line (steelblue solid) ──────────────────────────────────────
    plot!(p, collect(xs), model_vals;
          color     = _C1_PANEL,
          linewidth = 1.8,
          label     = show_legend ? "Model" : "")

    # ── Vertical dashed line at switch date ───────────────────────────────
    switch_idx = findfirst(d -> d >= switch_date, dates)
    if !isnothing(switch_idx)
        vline!(p, [switch_idx];
               color     = _C2_PANEL,
               linestyle = :dash,
               linewidth = 1.4,
               label     = show_legend ? "Switch" : "")
    end

    return p
end


# ─── NaN-aware pairwise average ──────────────────────────────────────────────
function _avg_pair(a::Float64, b::Float64) :: Float64
    isnan(a) && isnan(b) && return NaN
    isnan(a) && return b
    isnan(b) && return a
    return 0.5 * (a + b)
end

# ═══════════════════════════════════════════════════════════
# Main function
# ═══════════════════════════════════════════════════════════

"""
    make_transition_panel(; scenario, suffix)

Generate and save the 3×2 transition dynamics panel for the given scenario.

Arguments:
  scenario  :fc or :covid
  suffix    weight-matrix suffix string (default W_SUFFIX)
"""
function make_transition_panel(;
    scenario :: Symbol = :fc,
    suffix   :: String = W_SUFFIX,
)
    println("\n── Transition panel: $scenario ──"); flush(stdout)
    _set_theme!()

    # ── Calendar ─────────────────────────────────────────────────────────
    cal    = episode_calendar(scenario)
    dates  = monthly_range(cal.start_date, cal.end_date)
    N      = length(dates)
    @printf("  Calendar: %s → %s  (%d months, switch %s)\n",
            cal.start_date, cal.end_date, N, cal.switch_date)

    # ── Load TransitionResult ─────────────────────────────────────────────
    trans_file = joinpath(TRANS_OUT_DIR, "transition_$(scenario)$(suffix).jld2")
    if !isfile(trans_file)
        @warn "Transition result not found: $trans_file\n" *
              "Run transition_main.jl first.  Skipping panel."
        return nothing
    end
    trans = JLD2.load(trans_file, "result")
    @printf("  Loaded transition result (converged=%s, scenario=%s)\n",
            trans.converged, trans.scenario)

    # ── Load moments and SEs ─────────────────────────────────────────────
    base_moms  = load_moments_values(; window=cal.base_window,   derived_dir=DERIVED_DIR)
    crisis_moms= load_moments_values(; window=cal.crisis_window, derived_dir=DERIVED_DIR)
    base_ses   = load_sigma_ses(; window=cal.base_window,   derived_dir=DERIVED_DIR)
    crisis_ses = load_sigma_ses(; window=cal.crisis_window, derived_dir=DERIVED_DIR)

    # ── Load monthly CPS data ─────────────────────────────────────────────
    cps_monthly = load_cps_monthly(cal, DERIVED_DIR)
    @printf("  CPS monthly obs loaded: %d months\n", length(cps_monthly))

    # ── Load monthly JOLTS data ───────────────────────────────────────────
    θU_dict, θS_dict = load_jolts_monthly(cal, cps_monthly, DERIVED_DIR)
    @printf("  JOLTS θ_U obs: %d  θ_S obs: %d\n",
            length(θU_dict), length(θS_dict))

    # ── Build model series ────────────────────────────────────────────────
    dates_m, mod_ur, mod_θU, mod_θS, mod_wU, mod_wS, mod_tr, mod_sk =
        build_model_series(trans, cal)

    # ── Build data series ─────────────────────────────────────────────────
    ds = build_data_series(cal, cps_monthly, θU_dict, θS_dict,
                           base_moms, crisis_moms, base_ses, crisis_ses)

    # ── Composite tightness: mean of θ_U and θ_S ─────────────────────────
    # Data: NaN-aware average of the two tightness measures
    θ_comp_vals = Float64[ _avg_pair(ds.theta_U.values[i], ds.theta_S.values[i]) for i in 1:N ]
    θ_comp_ses  = Float64[ _avg_pair(ds.theta_U.ses[i],    ds.theta_S.ses[i])    for i in 1:N ]
    # Model: average θ_U and θ_S
    mod_θcomp = 0.5 .* (mod_θU .+ mod_θS)

    # ── Build 6 subplots ─────────────────────────────────────────────────
    panel_specs = [
        #  title               data_vals                data_ses               model_vals   row  col
        ("Unemployment",       ds.ur_total.values,      ds.ur_total.ses,       mod_ur,       1,   1),
        ("Tightness",          θ_comp_vals,             θ_comp_ses,            mod_θcomp,    1,   2),
        ("Unskilled wages",    ds.mean_wage_U.values,   ds.mean_wage_U.ses,    mod_wU,       2,   1),
        ("Skilled wages",      ds.mean_wage_S.values,   ds.mean_wage_S.ses,    mod_wS,       2,   2),
        ("Training share",     ds.training_share.values,ds.training_share.ses, mod_tr,       3,   1),
        ("Skilled share",      ds.skilled_share.values, ds.skilled_share.ses,  mod_sk,       3,   2),
    ]

    subplots = Plots.Plot[]
    for (idx, (ttl, dv, ds_ses, mv, row, col)) in enumerate(panel_specs)
        is_bottom  = row == 3
        is_first   = idx == 1
        sp = make_one_panel(
            dates, dv, ds_ses, mv,
            cal.switch_date, ttl;
            show_legend  = is_first,
            show_xlabel  = is_bottom,
            show_se_bands= SHOW_SE_BANDS,
        )
        push!(subplots, sp)
    end

    # ── Combine into 3×2 layout ───────────────────────────────────────────
    fig = plot(subplots...;
               layout = (3, 2),
               size   = (900, 750),
               margin = 4Plots.mm,
               link   = :none)

    # ── Save ──────────────────────────────────────────────────────────────
    mkpath(PLOTS_DIR)
    label   = scenario == :fc ? "fc" : "covid"
    out_file= joinpath(PLOTS_DIR, "transition_panel_$(label)$(suffix).png")
    savefig(fig, out_file)
    @printf("  Saved: %s\n", out_file)
    flush(stdout)

    return out_file
end


# ═══════════════════════════════════════════════════════════
# 3×2 MODEL-ONLY decomposition panel
#
# Layout:
#   Unemployment  |  Vacancies
#   Wages         |  Tightness
#   JFR           |  Sep. rate
#
# Colours: green = skilled, purple = unskilled
# X-axis: 20% pre-shock (flat SS), 80% transition path
# ═══════════════════════════════════════════════════════════

_COL_SKL   = :seagreen
_COL_UNSK  = :mediumpurple

"""
    make_model_decomposition_panel(; scenario, suffix)

Model-only 3×2 panel with skilled (green) and unskilled (purple) lines.
20% of the x-axis is the pre-shock steady state, 80% is the transition.
"""
function make_model_decomposition_panel(;
    scenario :: Symbol = :fc,
    suffix   :: String = W_SUFFIX,
)
    println("\n── Model decomposition panel: $scenario ──"); flush(stdout)
    _set_theme!()

    # ── Load transition result ─────────────────────────────────────────
    trans_file = joinpath(TRANS_OUT_DIR, "transition_$(scenario)$(suffix).jld2")
    if !isfile(trans_file)
        @warn "Transition result not found: $trans_file"
        return nothing
    end
    jld = JLD2.load(trans_file)
    result = jld["result"]
    @printf("  Loaded transition (converged=%s)\n", result.converged)

    tgrid = result.tgrid   # 0 to T_max (120 months), dt=0.5, Nt=241
    Nt    = length(tgrid)
    T_max = tgrid[end]
    wx    = result.wx       # Gauss-Legendre weights for x-grid
    Nx    = length(wx)

    # ── Build x-axis: 20% pre-shock + 80% transition ────────────────
    # Total number of points on the plot
    N_pre  = 30           # flat pre-shock portion
    N_post = 120          # transition portion (one point per model month)
    N_tot  = N_pre + N_post

    # Model months for the transition portion: 0, 1, 2, ..., 119
    # (interpolated from the half-month tgrid)
    function _interp(series::Vector{Float64}, t_query::Float64)
        t_q = clamp(t_query, tgrid[1], tgrid[end])
        idx = searchsortedfirst(tgrid, t_q)
        idx == 1  && return series[1]
        idx > Nt  && return series[Nt]
        t_lo = tgrid[idx-1]; t_hi = tgrid[idx]
        α = (t_hi > t_lo) ? (t_q - t_lo) / (t_hi - t_lo) : 0.0
        return series[idx-1] * (1.0 - α) + series[idx] * α
    end

    # Splice: N_pre points at SS value, then N_post points from transition
    function _splice(series::Vector{Float64})
        ss = series[1]
        out = Vector{Float64}(undef, N_tot)
        for i in 1:N_pre
            out[i] = ss
        end
        for j in 1:N_post
            out[N_pre + j] = _interp(series, Float64(j - 1))
        end
        return out
    end

    # Same for 2-D arrays (Nx × Nt) — aggregate with wx weights
    function _splice_agg(mat::Matrix{Float64})
        # mat is Nx × Nt; aggregate: sum over x with weights wx
        ss_agg = dot(wx, mat[:, 1])
        out = Vector{Float64}(undef, N_tot)
        for i in 1:N_pre
            out[i] = ss_agg
        end
        for j in 1:N_post
            # interpolate each x, then aggregate
            t_q = clamp(Float64(j - 1), tgrid[1], tgrid[end])
            idx = searchsortedfirst(tgrid, t_q)
            if idx == 1
                out[N_pre + j] = ss_agg
            elseif idx > Nt
                out[N_pre + j] = dot(wx, mat[:, Nt])
            else
                t_lo = tgrid[idx-1]; t_hi = tgrid[idx]
                α = (t_hi > t_lo) ? (t_q - t_lo) / (t_hi - t_lo) : 0.0
                out[N_pre + j] = dot(wx, (1.0 - α) .* mat[:, idx-1] .+ α .* mat[:, idx])
            end
        end
        return out
    end

    # ── Splice all series ─────────────────────────────────────────
    m_ur_U  = _splice(result.ur_U)
    m_ur_S  = _splice(result.ur_S)
    m_θU    = _splice(result.θU)
    m_θS    = _splice(result.θS)
    m_wU    = _splice(result.mean_wage_U)
    m_wS    = _splice(result.mean_wage_S)
    m_fU    = _splice(result.fU)
    m_fS    = _splice(result.fS)

    # ── Vacancies: v_j = θ_j × U_j (aggregate unemployment mass) ────
    agg_uU = _splice_agg(result.uU)   # aggregate unskilled unemployed
    agg_uS = _splice_agg(result.uS)   # aggregate skilled unemployed
    m_vU   = m_θU .* agg_uU
    m_vS   = m_θS .* agg_uS

    # ── Separation rates: from flow identity ───────────────────────
    # In the transition, Δu ≈ sep·e - f·u  (plus demographic flows)
    # so sep ≈ (Δu + f·u) / e  where e = pop - u
    # This is approximate; use it for the shape of the dynamics.
    # Compute from the aggregate unemployment paths and JFR.
    function _compute_sep(ur_path, f_path)
        sep = similar(ur_path)
        for i in 1:N_tot
            u = ur_path[i]
            e = max(1.0 - u, 1e-14)
            if i < N_tot
                du = ur_path[i+1] - ur_path[i]
            else
                du = 0.0
            end
            sep[i] = max((du + f_path[i] * u) / e, 0.0)
        end
        return sep
    end
    m_sepU = _compute_sep(m_ur_U, m_fU)
    m_sepS = _compute_sep(m_ur_S, m_fS)

    # ── X-axis and ticks ────────────────────────────────────────
    xs = collect(1:N_tot)
    switch_x = N_pre + 1   # shock line at the 20% mark

    # Ticks: label every 20 months in the transition + "SS" for pre-shock
    tick_pos   = Int[div(N_pre, 2)]   # mid-point of pre-shock
    tick_label = String["SS"]
    for m in 0:20:N_post-1
        push!(tick_pos, N_pre + m + 1)
        push!(tick_label, string(m))
    end

    # ── Subplot helper ──────────────────────────────────────────
    function _panel(title_str; show_xlabel=false, show_legend=false)
        p = plot(;
            title         = title_str,
            xlabel        = show_xlabel ? "Months" : "",
            ylabel        = "",
            legend        = show_legend ? :topright : false,
            xticks        = (tick_pos, tick_label),
            bottom_margin = 3Plots.mm,
            top_margin    = 3Plots.mm,
        )
        vline!(p, [switch_x]; color=:firebrick, ls=:dash, lw=1.2, label="")
        return p
    end

    # ── Panel 1: Unemployment (top left) ───────────────────────────
    p1 = _panel("Unemployment"; show_legend=true)
    plot!(p1, xs, m_ur_S, color=_COL_SKL,  lw=1.6, label="Skilled")
    plot!(p1, xs, m_ur_U, color=_COL_UNSK, lw=1.6, label="Unskilled")

    # ── Panel 2: Vacancies (top right) ─────────────────────────────
    p2 = _panel("Vacancies")
    plot!(p2, xs, m_vS, color=_COL_SKL,  lw=1.6, label="Skilled")
    plot!(p2, xs, m_vU, color=_COL_UNSK, lw=1.6, label="Unskilled")

    # ── Panel 3: Wages (mid left) ──────────────────────────────────
    p3 = _panel("Wages")
    plot!(p3, xs, m_wS, color=_COL_SKL,  lw=1.6, label="Skilled")
    plot!(p3, xs, m_wU, color=_COL_UNSK, lw=1.6, label="Unskilled")

    # ── Panel 4: Tightness (mid right) ─────────────────────────────
    p4 = _panel("Tightness")
    plot!(p4, xs, m_θS, color=_COL_SKL,  lw=1.6, label="Skilled")
    plot!(p4, xs, m_θU, color=_COL_UNSK, lw=1.6, label="Unskilled")

    # ── Panel 5: Job-finding rates (bottom left) ───────────────────
    p5 = _panel("Job-finding rate"; show_xlabel=true)
    plot!(p5, xs, m_fS, color=_COL_SKL,  lw=1.6, label="Skilled")
    plot!(p5, xs, m_fU, color=_COL_UNSK, lw=1.6, label="Unskilled")

    # ── Panel 6: Separation rates (bottom right) ───────────────────
    p6 = _panel("Separation rate"; show_xlabel=true)
    plot!(p6, xs, m_sepS, color=_COL_SKL,  lw=1.6, label="Skilled")
    plot!(p6, xs, m_sepU, color=_COL_UNSK, lw=1.6, label="Unskilled")

    # ── Force y-limits from actual data for each panel individually ──
    function _ylims_from_data(vecs...)
        lo = Inf; hi = -Inf
        for v in vecs
            vals = filter(isfinite, v)
            isempty(vals) && continue
            lo = min(lo, minimum(vals))
            hi = max(hi, maximum(vals))
        end
        pad = 0.05 * max(hi - lo, 1e-14)
        return (lo - pad, hi + pad)
    end

    ylims!(p1, _ylims_from_data(m_ur_U, m_ur_S))
    ylims!(p2, _ylims_from_data(m_vU, m_vS))
    ylims!(p3, _ylims_from_data(m_wU, m_wS))
    ylims!(p4, _ylims_from_data(m_θU, m_θS))
    ylims!(p5, _ylims_from_data(m_fU, m_fS))
    ylims!(p6, _ylims_from_data(m_sepU, m_sepS))

    # ── Combine 3×2 ────────────────────────────────────────────────
    fig = plot(p1, p2, p3, p4, p5, p6;
              layout = (3, 2),
              size   = (900, 750),
              margin = 5Plots.mm,
              link   = :none)

    mkpath(PLOTS_DIR)
    label    = scenario == :fc ? "fc" : "covid"
    out_file = joinpath(PLOTS_DIR, "model_transition_$(label)$(suffix).png")
    savefig(fig, out_file)
    @printf("  Saved: %s\n", out_file)
    flush(stdout)
    return out_file
end


# ═══════════════════════════════════════════════════════════
# Entry point — generate all panels
# ═══════════════════════════════════════════════════════════

println("="^65)
println("  Segmented Search Model — Transition Panel Plots")
println("="^65)
flush(stdout)

for sc in [:fc, :covid]
    make_transition_panel(; scenario=sc, suffix=W_SUFFIX)
    make_model_decomposition_panel(; scenario=sc, suffix=W_SUFFIX)
end

println("\nDone.")
println("="^65)
flush(stdout)

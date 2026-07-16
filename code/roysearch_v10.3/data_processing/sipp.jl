############################################################
# data_processing/sipp.jl
#
# Stage 6b — SIPP within-job wage-change frequency among job stayers,
# wchg_rate_U and wchg_rate_S. These moments separate the exogenous
# separation hazard ξ_j from the endogenous quality-shock margin λ_j·G(p*_j):
# a λ_j redraw that lands at/above the reservation cutoff survives the match
# but re-prices it (a within-job wage change, rate λ_j·[1−G(p*_j)]), while ξ_j
# only ever destroys a match. So wchg_rate loads on λ blind to ξ, and sep_rate
# then pins ξ as the residual (data-and-moments §sipp; identification section).
#
# Reads:  data/raw/sipp/*.csv       (Census public-use core files pu<release>)
# Writes: sipp_wchg_rates.csv        (columns: window, wchg_rate_U, wchg_rate_S,
#                                     neff_U, neff_S — Kish effective N for Σ̂)
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages (incl. Dates) and path consts come from
# data_processing_main.jl.
############################################################

# ── Reference-year → window map (SIPP-local) ───────────────────────────────
# Each redesign release covers the prior calendar year: pu<Y> has reference
# year Y−1. base_covid draws on ref 2016–2020, crisis_covid on ref 2021–2023
# (data-and-moments §sipp, panel-coverage paragraph). This mapping lives HERE
# rather than in assign_window: the global monthly window crisis_covid is
# 2020–2022 by construction, and extending its ym_end to 2023 would pull any
# 2023 rows of CPS Basic / JOLTS / J2J into crisis_covid and shift those
# moments. SIPP ref-2023 is a release-year artifact intended to measure the
# COVID crisis, so it is assigned to crisis_covid locally, leaving every other
# stage's window membership untouched.
function sipp_ref_year_to_window(ref_year::Int)::Symbol
    2016 <= ref_year <= 2020 && return :base_covid
    2021 <= ref_year <= 2023 && return :crisis_covid
    return :none
end

# BLS CPI-U, US city average, all items, annual average (series CUUR0000SA0,
# 1982–84 = 100). The Census public-use SIPP ships no CPI99 (that is IPUMS-
# only), so weekly earnings are deflated with this external CPI-U series
# rebased to 2013, consistent with the ASEC wage construction (setup.jl
# deflate_wage, 2013 base). Only the years the SIPP windows can reference are
# listed; add rows here if a new reference year is introduced.
const SIPP_CPIU = Dict{Int,Float64}(
    2016 => 240.007, 2017 => 245.120, 2018 => 251.107, 2019 => 255.657,
    2020 => 258.811, 2021 => 270.970, 2022 => 292.655, 2023 => 304.702,
)
const SIPP_CPIU_BASE_YEAR = 2013
const SIPP_CPIU_BASE       = 232.957   # CPI-U annual average, 2013

# Parse the reference year from a Census SIPP core filename `pu<release>.csv`
# (release covers the prior calendar year, so ref = release − 1). Returns
# `nothing` when the pattern is absent.
function sipp_ref_year_from_filename(fname::AbstractString)
    m = match(r"pu(\d{4})"i, fname)
    isnothing(m) && return nothing
    return parse(Int, m.captures[1]) - 1
end

# Day-count-neutral real weekly wage from a job's monthly earnings. SIPP
# TJB{n}_MSUM is "monthly earnings varying with the number of days in the
# month", so a flat ×12/52 (or raw MSUM differencing) would flag a spurious
# change every month purely from day-count. Normalise to a 7-day week using
# the calendar month's length BEFORE differencing, then deflate to 2013 $.
function sipp_real_weekly_wage(msum::Float64, ref_year::Int, cal_month::Int,
                               cpi_ref::Float64)::Float64
    nominal_week = msum * 7.0 / Dates.daysinmonth(ref_year, cal_month)
    return deflate_wage(nominal_week, cpi_ref, SIPP_CPIU_BASE)
end

# Long-format per-job records for one person-month: (jobid, msum) for every
# non-missing job slot n = 1..7. Absent/blank slots are skipped.
function _sipp_job_records(row, n_slots::Int=7)
    recs = Tuple{String,Float64}[]
    for n in 1:n_slots
        jid_col = Symbol("EJB$(n)_JOBID")
        msm_col = Symbol("TJB$(n)_MSUM")
        (hasproperty(row, jid_col) && hasproperty(row, msm_col)) || continue
        jid = getproperty(row, jid_col)
        msm = getproperty(row, msm_col)
        (ismissing(jid) || ismissing(msm)) && continue
        jid_s = strip(string(jid))
        (isempty(jid_s) || jid_s == "0") && continue
        msm_f = _sipp_to_float(msm)
        (isnan(msm_f) || msm_f <= 0.0) && continue
        push!(recs, (jid_s, msm_f))
    end
    return recs
end

# Coerce a CSV cell (Int / Float / String / missing) to Float64; NaN on failure.
function _sipp_to_float(v)::Float64
    ismissing(v) && return NaN
    v isa Real && return Float64(v)
    s = strip(string(v))
    isempty(s) && return NaN
    p = tryparse(Float64, s)
    return isnothing(p) ? NaN : p
end

# Coerce a CSV cell to Int; `nothing` when missing/blank/unparseable. SIPP core
# files carry `missing` in the categorical columns (MONTHCODE, RMESR, TAGE,
# EEDUC), so a bare Int(...) is a latent crash — call sites treat `nothing` as
# "cannot qualify this pair" and skip it.
function _sipp_to_int(v)::Union{Int,Nothing}
    f = _sipp_to_float(v)
    (isnan(f) || !isfinite(f)) && return nothing
    return round(Int, f)
end

# SIPP EEDUC (highest level/degree, scale 31–46; dictionary p.888) — skilled
# is Bachelor's degree or higher (43 BA, 44 MA, 45 professional, 46 doctorate),
# matching the project's CPS EDUC>=111 (bachelor's+) definition. EEDUC is a
# distinct variable from CPS EDUC, so the shared is_skilled (educ>=111) never
# fires on this scale and must not be reused here.
is_skilled_sipp(eeduc::Integer)::Bool = eeduc >= 43

# Accumulate WPFINWGT-weighted stayer-pair counts for one SIPP core file into
# `acc[window] = [den_U, chg_U, den_S, chg_S, den2_U, den2_S]`: den_j = Σw
# (all qualifying pairs), chg_j = Σw over pairs with a change, den2_j = Σw²
# (for the Kish effective sample size neff = den²/den2, used by Σ̂). Prints the
# per-file diagnostic (ref-year CPI factor and one nominal/real example pair).
function _sipp_accumulate_file!(acc::Dict{Symbol,Vector{Float64}}, path::String)
    fname    = basename(path)
    ref_year = sipp_ref_year_from_filename(fname)
    if isnothing(ref_year)
        @warn "    $fname: cannot parse reference year from filename — skipped."
        return nothing
    end
    window = sipp_ref_year_to_window(ref_year)
    if window == :none
        @warn "    $fname (ref $ref_year): no estimation window for this reference year — skipped."
        return nothing
    end
    cpi_ref = get(SIPP_CPIU, ref_year, NaN)
    if !isfinite(cpi_ref)
        @warn "    $fname (ref $ref_year): no CPI-U entry for $ref_year in SIPP_CPIU — skipped."
        return nothing
    end
    cpi_factor = SIPP_CPIU_BASE / cpi_ref

    df = CSV.read(path, DataFrame)
    for req in (:SSUID, :PNUM, :MONTHCODE, :RMESR, :EEDUC, :TAGE, :WPFINWGT)
        hasproperty(df, req) || (@warn "    $fname: missing column $req — skipped."; return nothing)
    end

    a = get!(() -> zeros(6), acc, window)   # [den_U,chg_U,den_S,chg_S,den2_U,den2_S]
    example = nothing

    # One stayer observation per continuing job (same EJB{n}_JOBID) across a
    # consecutive MONTHCODE pair, employed all month at both dates (RMESR==1),
    # working-age at both. Weighted by the later month's person-month weight.
    for pg in groupby(df, [:SSUID, :PNUM])
        # Drop rows without a usable calendar month before ordering; a person
        # left with fewer than two valid-month rows then yields no pairs.
        p = sort(filter(row -> !isnothing(_sipp_to_int(row.MONTHCODE)), DataFrame(pg)), :MONTHCODE)
        for i in 1:(nrow(p) - 1)
            r1 = p[i, :]; r2 = p[i + 1, :]
            # Any missing/unparseable qualification field skips the pair.
            mc1 = _sipp_to_int(r1.MONTHCODE); mc2 = _sipp_to_int(r2.MONTHCODE)
            (isnothing(mc1) || isnothing(mc2) || mc2 - mc1 != 1) && continue
            rm1 = _sipp_to_int(r1.RMESR); rm2 = _sipp_to_int(r2.RMESR)
            (isnothing(rm1) || isnothing(rm2) || rm1 != 1 || rm2 != 1) && continue
            ta1 = _sipp_to_int(r1.TAGE); ta2 = _sipp_to_int(r2.TAGE)
            (isnothing(ta1) || isnothing(ta2) || !in_age_range(ta1) || !in_age_range(ta2)) && continue

            jobs1 = Dict(_sipp_job_records(r1))
            jobs2 = Dict(_sipp_job_records(r2))
            ed2 = _sipp_to_int(r2.EEDUC)
            isnothing(ed2) && continue          # cannot classify U/S without education
            skilled = is_skilled_sipp(ed2)
            wgt     = _sipp_to_float(r2.WPFINWGT)
            (isnan(wgt) || wgt <= 0.0) && continue
            m1 = mc1; m2 = mc2

            for (jid, msum1) in jobs1
                haskey(jobs2, jid) || continue
                w1 = sipp_real_weekly_wage(msum1,        ref_year, m1, cpi_ref)
                w2 = sipp_real_weekly_wage(jobs2[jid],   ref_year, m2, cpi_ref)
                (isfinite(w1) && isfinite(w2)) || continue
                changed = abs(w2 - w1) > 1.0          # $1/week floor (locked)
                if skilled
                    a[3] += wgt;  changed && (a[4] += wgt);  a[6] += wgt^2
                else
                    a[1] += wgt;  changed && (a[2] += wgt);  a[5] += wgt^2
                end
                isnothing(example) &&
                    (example = (nominal = msum1 * 7.0 / Dates.daysinmonth(ref_year, m1),
                                real = w1))
            end
        end
    end

    @printf("    %s (ref %d → %s): CPI factor %.4f (CPI-U %.3f → 2013 base %.3f)\n",
            fname, ref_year, window, cpi_factor, cpi_ref, SIPP_CPIU_BASE)
    if !isnothing(example)
        @printf("      example weekly wage: nominal \$%.2f → real(2013) \$%.2f\n",
                example.nominal, example.real)
    end
    return nothing
end

# Probability → continuous-time monthly hazard, p at horizon h=1 (consecutive
# month-pairs). Returns NaN for an empty denominator so the window is held out.
_sipp_hazard(chg::Float64, den::Float64)::Float64 =
    den > 0.0 ? -log(1.0 - min(chg / den, 1.0 - 1e-12)) : NaN

function make_sipp_wchg()
    @info "Stage 6b: SIPP within-job wage-change rates (wchg_rate_U, wchg_rate_S)..."

    sipp_dir = joinpath(RAW_DIR, "sipp")
    if !isdir(sipp_dir)
        @warn "  $sipp_dir not found — no SIPP data yet. Writing no sipp_wchg_rates.csv; " *
              "windows without a row auto-hold-out of the SMM objective on NaN."
        return DataFrame(window = String[], wchg_rate_U = Float64[], wchg_rate_S = Float64[],
                          neff_U = Float64[], neff_S = Float64[])
    end
    files = sort(filter(f -> endswith(lowercase(f), ".csv"), readdir(sipp_dir)))
    if isempty(files)
        @warn "  $sipp_dir is empty — no SIPP data yet. Writing no sipp_wchg_rates.csv; " *
              "windows without a row auto-hold-out of the SMM objective on NaN."
        return DataFrame(window = String[], wchg_rate_U = Float64[], wchg_rate_S = Float64[],
                          neff_U = Float64[], neff_S = Float64[])
    end

    # Accumulate weighted stayer-pair counts across every file first (several
    # reference years map to one window), THEN form p and the hazard per window.
    acc = Dict{Symbol,Vector{Float64}}()
    for f in files
        _sipp_accumulate_file!(acc, joinpath(sipp_dir, f))
    end

    # Kish effective sample size neff = (Σw)² / Σw², carried to Σ̂ so sigma.jl
    # can place a binomial-proportion (delta-method) variance on each hazard
    # without re-reading the SIPP micro-data.
    _neff(den::Float64, den2::Float64)::Float64 = den2 > 0.0 ? den^2 / den2 : 0.0

    # One row per window that has its OWN SIPP data. Each window's hazard is
    # measured only from files whose reference year maps to it (base_covid ←
    # ref 2016–2020, crisis_covid ← ref 2021–2023); the financial-crisis
    # windows require the classic 2004/2008 SIPP panels, whose fixed-width
    # layout this reader does not parse. A window with no SIPP file gets NO row
    # here: moments.jl leaves its wchg_rate_j at NaN, which auto-holds the
    # moment out of the SMM objective for that window. No cross-window borrowing
    # — a window's wage-change hazard is either its own measurement or held out.
    rows = NamedTuple[]
    for wname in WINDOWS_ORDER
        haskey(acc, wname) || continue
        den_U, chg_U, den_S, chg_S, den2_U, den2_S = acc[wname]
        r_U, r_S = _sipp_hazard(chg_U, den_U), _sipp_hazard(chg_S, den_S)
        n_U, n_S = _neff(den_U, den2_U), _neff(den_S, den2_S)
        push!(rows, (window = wname, wchg_rate_U = r_U, wchg_rate_S = r_S,
                     neff_U = n_U, neff_S = n_S))
        @printf("  %-14s wchg_rate_U=%.5f  wchg_rate_S=%.5f  (neff_U=%.1f, neff_S=%.1f)\n",
                string(wname), r_U, r_S, n_U, n_S)
    end

    covered  = [string(r.window) for r in rows]
    heldout  = [string(w) for w in WINDOWS_ORDER if !(string(w) in covered)]
    isempty(heldout) ||
        @warn "  SIPP wchg measured for [$(join(covered, ", "))]; NO SIPP data for " *
              "[$(join(heldout, ", "))] — wchg_rate_U/S held out of the SMM objective " *
              "in those windows. Download the SIPP files for the missing reference years " *
              "(see data-and-moments §sipp) to close the gap."

    df_out = DataFrame(rows)
    if isempty(df_out)
        df_out = DataFrame(window = String[], wchg_rate_U = Float64[], wchg_rate_S = Float64[],
                           neff_U = Float64[], neff_S = Float64[])
    end
    outpath = joinpath(DERIVED_DIR, "sipp_wchg_rates.csv")
    CSV.write(outpath, df_out)
    @info "  Saved: $outpath"
    return df_out
end

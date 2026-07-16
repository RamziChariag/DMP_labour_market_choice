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
# Two file formats feed one per-window measurement:
#   • Redesign (2014-present): comma-delimited pu<Y>.csv annual releases and
#     pu2014w<N>.csv per-wave files, EEDUC / EJB<n>_JOBID / TJB<n>_MSUM (7 job
#     slots), MONTHCODE the reference month within the reference year.
#   • Classic (2001/2004/2008): fixed-width l<YY>puw<N>.dat cores. Census ships
#     the core file for EVERY wave but the core DICTIONARY for wave 1 only
#     (l<YY>puw1d.txt); the other waves' folders carry only Topical-Module
#     dictionaries, which document a different file and lack the core job
#     variables. The core layout is constant across a panel's waves, so the
#     wave-1 core dictionary is resolved ONCE per panel and applied to every
#     wave. Variable NAMES are stable across panels but byte OFFSETS differ per
#     panel, so fields are located by name. EEDUCATE / EENO1-2 / TPMSUM1-2 (2
#     job slots), RHCALYR+RHCALMN the reference month.
#
# Both formats assign each person-month to an estimation window by its OWN
# calendar month (assign_window in setup.jl), accumulate into shared per-window
# weighted counts, and only then convert to the model's monthly hazard. No
# file-level reference year and no cross-window borrowing.
#
# Reads (walked recursively under data/raw/sipp/, so files sitting in per-year
# subfolders are found as well as those directly under sipp/):
#         pu*.csv                        redesign cores (flat annual pu<Y>.csv and
#                                        the 2014 panel's per-wave pu2014w<N>.csv)
#         l<YY>puw<N>.dat                classic cores (in the per-panel subfolders)
#         l<YY>puw1d.txt[.old2/.old.2]   classic wave-1 core dictionary, resolved
#                                        from the same directory as its panel's cores
# Compressed cores (.zip, .dat.gz) with no extracted sibling are skipped with a
# gunzip/unzip hint — the SIPP files ship extracted, so the reader needs no
# zip/gzip dependency.
# Writes: sipp_wchg_rates.csv  (window, wchg_rate_U, wchg_rate_S, neff_U, neff_S)
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages (incl. Dates) and path consts come from
# data_processing_main.jl.
############################################################

# ── CPI-U deflator ─────────────────────────────────────────────────────────
# BLS CPI-U, US city average, all items, annual average (series CUUR0000SA0,
# 1982–84 = 100). The Census public-use SIPP ships no CPI99 (that is IPUMS-
# only), so weekly earnings are deflated with this external CPI-U series
# rebased to 2013, consistent with the ASEC wage construction (setup.jl
# deflate_wage, 2013 base). Covers every calendar year the four estimation
# windows can reference (2003–2022); add rows here if a new window is added.
const SIPP_CPIU = Dict{Int,Float64}(
    2003 => 184.000, 2004 => 188.900, 2005 => 195.300, 2006 => 201.600,
    2007 => 207.342, 2008 => 215.303, 2009 => 214.537, 2010 => 218.056,
    2011 => 224.939, 2012 => 229.594, 2013 => 232.957, 2014 => 236.736,
    2015 => 237.017, 2016 => 240.007, 2017 => 245.120, 2018 => 251.107,
    2019 => 255.657, 2020 => 258.811, 2021 => 270.970, 2022 => 292.655,
)
const SIPP_CPIU_BASE_YEAR = 2013
const SIPP_CPIU_BASE       = 232.957   # CPI-U annual average, 2013

# ── Threshold-robustness grid ──────────────────────────────────────────────
# A within-job wage change is counted when |Δ real weekly wage| exceeds a
# threshold ε ($/week). The production moment uses ε = $1 (SIPP_WCHG_EPS); the
# other thresholds are re-measured in parallel for the diagnostic table printed
# at the end of the stage. Edit this vector to change the diagnostic grid; the
# production definition is fixed at $1.
const SIPP_WCHG_EPS_GRID = [1.0, 5.0, 10.0, 20.0]
const SIPP_WCHG_EPS       = 1.0   # locked production threshold ($/week)

# ── Cell coercion ──────────────────────────────────────────────────────────
# Coerce a CSV / fixed-width cell (Int / Float / String / missing) to Float64;
# NaN on failure.
function _sipp_to_float(v)::Float64
    ismissing(v) && return NaN
    v isa Real && return Float64(v)
    s = strip(string(v))
    isempty(s) && return NaN
    p = tryparse(Float64, s)
    return isnothing(p) ? NaN : p
end

# Coerce a cell to Int; `nothing` when missing/blank/unparseable. SIPP core
# files carry `missing` in the categorical columns (MONTHCODE, RMESR, TAGE,
# EEDUC/EEDUCATE), so a bare Int(...) is a latent crash — call sites treat
# `nothing` as "cannot qualify this record" and skip it.
function _sipp_to_int(v)::Union{Int,Nothing}
    f = _sipp_to_float(v)
    (isnan(f) || !isfinite(f)) && return nothing
    return round(Int, f)
end

# ── Real weekly wage ───────────────────────────────────────────────────────
# Day-count-neutral real weekly wage from a job's monthly earnings. Both SIPP
# formats report earnings as a monthly sum that varies with the number of days
# in the month (redesign TJB{n}_MSUM, classic TPMSUM{n}), so a flat ×12/52 (or
# raw monthly differencing) would flag a spurious change every month purely
# from day-count. Normalise to a 7-day week using the calendar month's length
# BEFORE differencing, then deflate to 2013 $.
function sipp_real_weekly_wage(msum::Float64, cal_year::Int, cal_month::Int,
                               cpi_ref::Float64)::Float64
    nominal_week = msum * 7.0 / Dates.daysinmonth(cal_year, cal_month)
    return deflate_wage(nominal_week, cpi_ref, SIPP_CPIU_BASE)
end

# ── Skilled classification ─────────────────────────────────────────────────
# Redesign EEDUC (highest level/degree, scale 31–46) — skilled is Bachelor's or
# higher (43 BA, 44 MA, 45 professional, 46 doctorate), matching the project's
# CPS EDUC>=111 definition. Distinct variable from CPS EDUC, so the shared
# is_skilled (educ>=111) never fires on this scale and is not reused here.
is_skilled_sipp(eeduc::Integer)::Bool = eeduc >= 43

# Classic EEDUCATE (highest degree/grade) uses a DIFFERENT scale from the
# redesign EEDUC: 44 Bachelor's, 45 Master's, 46 professional, 47 doctorate,
# with 43 an Associate degree (2001/2004/2008 core dictionaries, value labels
# for EEDUCATE). Bachelor's-or-higher is therefore ≥ 44, one code above the
# redesign threshold — the two must never share a classifier.
is_skilled_sipp_classic(eeducate::Integer)::Bool = eeducate >= 44

# ── Per-window, per-threshold weighted counts ──────────────────────────────
# One accumulator per estimation window. For each threshold ε in the grid we
# carry [den_U, chg_U, den_S, chg_S]: den_j = Σw over all qualifying stayer
# pairs, chg_j = Σw over pairs whose |Δw_week| exceeds ε. The denominators are
# ε-invariant (same pairs), stored per ε so each threshold's table cell is
# self-contained. den2_j = Σw² (ε-invariant) gives the Kish effective sample
# size neff = den²/den2 carried to Σ̂. Every reader accumulates into the same
# struct keyed by window, so redesign and classic contributions to a shared
# window simply add.
struct WchgCounts
    eps    :: Vector{Float64}
    den_U  :: Vector{Float64}
    chg_U  :: Vector{Float64}
    den_S  :: Vector{Float64}
    chg_S  :: Vector{Float64}
    den2_U :: Base.RefValue{Float64}
    den2_S :: Base.RefValue{Float64}
end
WchgCounts(eps::Vector{Float64}) =
    WchgCounts(eps, zeros(length(eps)), zeros(length(eps)),
               zeros(length(eps)), zeros(length(eps)), Ref(0.0), Ref(0.0))

# Record one qualifying stayer pair: same job, same person, consecutive months,
# weight `wgt`, real weekly wages `w1`→`w2`. Bins to skilled/unskilled and, at
# each threshold, adds the weight to the denominator and (when the change
# clears ε) to the change count. den2 counts the pair once (ε-invariant).
function _wchg_record_pair!(c::WchgCounts, skilled::Bool, wgt::Float64,
                            w1::Float64, w2::Float64)
    dw = abs(w2 - w1)
    if skilled
        c.den2_S[] += wgt^2
        @inbounds for k in eachindex(c.eps)
            c.den_S[k] += wgt
            dw > c.eps[k] && (c.chg_S[k] += wgt)
        end
    else
        c.den2_U[] += wgt^2
        @inbounds for k in eachindex(c.eps)
            c.den_U[k] += wgt
            dw > c.eps[k] && (c.chg_U[k] += wgt)
        end
    end
    return nothing
end

_wchg_get!(acc::Dict{Symbol,WchgCounts}, w::Symbol, eps::Vector{Float64}) =
    get!(() -> WchgCounts(eps), acc, w)

# ── Redesign reader (comma-delimited pu*.csv) ───────────────────────────────
# Reference year from a redesign core filename. Annual releases pu<Y>.csv cover
# the prior calendar year (ref = Y − 1). The 2014-panel per-wave files
# pu2014w<N>.csv carry annual waves whose reference year is 2012 + N (wave 1 →
# 2013 … wave 4 → 2016), so waves 3–4 land in base_covid's early years. Returns
# `nothing` when the pattern is absent.
function sipp_redesign_ref_year(fname::AbstractString)
    m = match(r"pu2014w(\d+)"i, fname)
    isnothing(m) || return 2012 + parse(Int, m.captures[1])
    m = match(r"pu(\d{4})"i, fname)
    isnothing(m) && return nothing
    return parse(Int, m.captures[1]) - 1
end

# Long-format per-job records for one redesign person-month: (jobid, msum) for
# every non-missing job slot n = 1..7. Absent/blank slots are skipped.
function _sipp_redesign_job_records(row, n_slots::Int=7)
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

# Accumulate one redesign core file into `acc`. Each person's consecutive
# MONTHCODE pairs are qualified (employed all month at both dates, working-age
# at both), matched by EJB{n}_JOBID, and priced with the reference year's CPI.
# The pair's reference month (ref_year, MONTHCODE) assigns it to a window by its
# own calendar month.
function _sipp_accumulate_redesign!(acc::Dict{Symbol,WchgCounts}, path::String)
    fname    = basename(path)
    ref_year = sipp_redesign_ref_year(fname)
    if isnothing(ref_year)
        @warn "    $fname: cannot parse reference year from filename — skipped."
        return nothing
    end
    cpi_ref = get(SIPP_CPIU, ref_year, NaN)
    if !isfinite(cpi_ref)
        @warn "    $fname (ref $ref_year): no CPI-U entry for $ref_year in SIPP_CPIU — skipped."
        return nothing
    end

    df = CSV.read(path, DataFrame)
    for req in (:SSUID, :PNUM, :MONTHCODE, :RMESR, :EEDUC, :TAGE, :WPFINWGT)
        hasproperty(df, req) || (@warn "    $fname: missing column $req — skipped."; return nothing)
    end

    n_pairs = 0
    windows_seen = Set{Symbol}()
    for pg in groupby(df, [:SSUID, :PNUM])
        p = sort(filter(row -> !isnothing(_sipp_to_int(row.MONTHCODE)), DataFrame(pg)), :MONTHCODE)
        for i in 1:(nrow(p) - 1)
            r1 = p[i, :]; r2 = p[i + 1, :]
            mc1 = _sipp_to_int(r1.MONTHCODE); mc2 = _sipp_to_int(r2.MONTHCODE)
            (isnothing(mc1) || isnothing(mc2) || mc2 - mc1 != 1) && continue
            rm1 = _sipp_to_int(r1.RMESR); rm2 = _sipp_to_int(r2.RMESR)
            (isnothing(rm1) || isnothing(rm2) || rm1 != 1 || rm2 != 1) && continue
            ta1 = _sipp_to_int(r1.TAGE); ta2 = _sipp_to_int(r2.TAGE)
            (isnothing(ta1) || isnothing(ta2) || !in_age_range(ta1) || !in_age_range(ta2)) && continue

            # Window from the pair's own reference month (later month).
            window = assign_window(ref_year, mc2)
            window == :none && continue

            jobs1 = Dict(_sipp_redesign_job_records(r1))
            jobs2 = Dict(_sipp_redesign_job_records(r2))
            ed2 = _sipp_to_int(r2.EEDUC)
            isnothing(ed2) && continue          # cannot classify U/S without education
            skilled = is_skilled_sipp(ed2)
            wgt     = _sipp_to_float(r2.WPFINWGT)
            (isnan(wgt) || wgt <= 0.0) && continue

            c = _wchg_get!(acc, window, SIPP_WCHG_EPS_GRID)
            for (jid, msum1) in jobs1
                haskey(jobs2, jid) || continue
                w1 = sipp_real_weekly_wage(msum1,      ref_year, mc1, cpi_ref)
                w2 = sipp_real_weekly_wage(jobs2[jid], ref_year, mc2, cpi_ref)
                (isfinite(w1) && isfinite(w2)) || continue
                _wchg_record_pair!(c, skilled, wgt, w1, w2)
                n_pairs += 1; push!(windows_seen, window)
            end
        end
    end
    @printf("    %s (redesign, ref %d): %d stayer pairs → windows [%s]\n",
            fname, ref_year, n_pairs, join(sort(collect(string.(windows_seen))), ", "))
    return nothing
end

# ── Classic reader (fixed-width l<YY>puw<N>.dat + dictionary) ────────────────
# Parse a classic core dictionary l<YY>puw<N>d.txt into name → (startpos, width).
# Field-definition lines are `D VARNAME width startpos` (1-indexed start), the
# only lines that start with a `D` followed by a name and two integers; value-
# label (`V`), text (`T`) and universe (`U`) lines are ignored. Positions differ
# between panels, which is exactly why fields are located by name here.
function parse_classic_dict(path::String)::Dict{String,Tuple{Int,Int}}
    pos = Dict{String,Tuple{Int,Int}}()
    for line in eachline(path)
        m = match(r"^D\s+([A-Z0-9_]+)\s+(\d+)\s+(\d+)\s*$", line)
        isnothing(m) && continue
        name = m.captures[1]
        haskey(pos, name) && continue          # first D line is the field definition
        pos[name] = (parse(Int, m.captures[3]), parse(Int, m.captures[2]))  # (startpos, width)
    end
    return pos
end

# Extract a fixed-width field by name from a record line, `nothing` when the
# field is out of range or blank. `posmap[name] = (startpos, width)`, 1-indexed.
function _classic_field(line::AbstractString, posmap, name::String)
    haskey(posmap, name) || return nothing
    sp, w = posmap[name]
    stop = sp + w - 1
    stop <= lastindex(line) || return nothing
    s = strip(SubString(line, sp, stop))
    return isempty(s) ? nothing : s
end

# Warn about a compressed core the walk cannot read and skip it. The SIPP release
# ships extracted files and the project depot carries no zip/gzip codec, so a
# .dat.gz or .zip is never decompressed here. When its extracted sibling is
# already present the compressed copy is redundant and ignored silently;
# otherwise the exact gunzip/unzip command to extract it is printed so the user
# can extract it and re-run the data stage. Never adds a dependency, never fails.
function _sipp_warn_compressed(path::String)
    dir, fname = dirname(path), basename(path)
    low = lowercase(fname)
    if endswith(low, ".gz")
        isfile(joinpath(dir, chopsuffix(fname, ".gz"))) && return nothing
        @warn "    $fname is gzipped and not extracted — run `gunzip $path` and re-run the data stage; skipped."
    elseif endswith(low, ".zip")
        isfile(joinpath(dir, chopsuffix(fname, ".zip"))) && return nothing
        @warn "    $fname is a zip archive and not extracted — run `unzip $path -d $dir` and re-run the data stage; skipped."
    end
    return nothing
end

# The classic same-job key per slot: employer entry number EENO<n> paired with
# monthly earnings TPMSUM<n>. Two slots (n = 1, 2), the classic analogue of the
# redesign EJB<n>_JOBID / TJB<n>_MSUM. Returns (eeno_string, msum) for each slot
# with a non-blank employer and positive earnings.
function _sipp_classic_job_records(line::AbstractString, posmap)
    recs = Tuple{String,Float64}[]
    for n in 1:2
        eeno = _classic_field(line, posmap, "EENO$(n)")
        (isnothing(eeno) || eeno == "0") && continue
        msum = _sipp_to_float(_classic_field(line, posmap, "TPMSUM$(n)"))
        (isnan(msum) || msum <= 0.0) && continue
        push!(recs, (String(eeno), msum))
    end
    return recs
end

# Required classic variables (present in the 2001/2004/2008 core dictionaries).
const CLASSIC_REQUIRED = ["SSUID", "EPPPNUM", "RHCALYR", "RHCALMN",
                          "WPFINWGT", "EEDUCATE", "RMESR",
                          "EENO1", "EENO2", "TPMSUM1", "TPMSUM2"]

# Classic panel year from a core filename l<YY>puw<N>.dat[.gz] (YY → 2000+YY).
function sipp_classic_panel_year(fname::AbstractString)
    m = match(r"l(\d{2})puw\d+\.dat"i, fname)
    isnothing(m) && return nothing
    return 2000 + parse(Int, m.captures[1])
end

# Accumulate one classic core wave into `acc`, locating fields by name from the
# panel's wave-1 core dictionary (resolved once by the caller and passed in as
# `posmap`). Streams the fixed-width .dat into per-person records
# (SSUID+EPPPNUM), orders them by calendar month (RHCALYR·12 + RHCALMN), and
# qualifies consecutive same-job pairs exactly as the redesign reader does:
# employed all month at both dates (RMESR==1), working-age at both, matched by
# EENO<n>, priced with the reference year's CPI, and assigned to a window by the
# pair's own calendar month. Within-wave month-to-month measurement mirrors the
# redesign construction; classic waves are 4 months long, so most transitions
# are within a wave (the wave-seam concentration of within-job changes is noted
# in data-and-moments §sipp — this reader does not correct for it).
function _sipp_accumulate_classic!(acc::Dict{Symbol,WchgCounts},
                                   dat_path::String, posmap)
    fname = basename(dat_path)

    # The wave-1 dictionary is applied to every wave of the panel on the premise
    # that the core layout is constant within a panel. Enforce it rather than
    # assume it: the record must be at least as long as the rightmost byte any
    # CLASSIC_REQUIRED field occupies. A shorter record means this wave's layout
    # differs from the wave-1 dictionary and the fields would be misread, so the
    # wave is skipped with a warning instead of read silently.
    reclen_needed = maximum(sp + w - 1 for (sp, w) in (posmap[v] for v in CLASSIC_REQUIRED))

    # Parse every record into a flat row; group and pair afterwards.
    rows = NamedTuple[]
    io = open(dat_path)
    length_checked = false
    try
        for line in eachline(io)
            isempty(strip(line)) && continue
            if !length_checked
                if length(line) < reclen_needed
                    @warn "    $fname: record length $(length(line)) < $reclen_needed implied by " *
                          "the wave-1 dictionary (layout mismatch) — wave skipped."
                    return nothing
                end
                length_checked = true
            end
            yr = _sipp_to_int(_classic_field(line, posmap, "RHCALYR"))
            mn = _sipp_to_int(_classic_field(line, posmap, "RHCALMN"))
            (isnothing(yr) || isnothing(mn) || !(1 <= mn <= 12)) && continue
            ssuid = _classic_field(line, posmap, "SSUID")
            pnum  = _classic_field(line, posmap, "EPPPNUM")
            (isnothing(ssuid) || isnothing(pnum)) && continue
            push!(rows, (
                pid   = string(ssuid, "|", pnum),
                cal_m = yr * 12 + mn,
                yr    = yr,
                mn    = mn,
                rmesr = _sipp_to_int(_classic_field(line, posmap, "RMESR")),
                tage  = _sipp_to_int(_classic_field(line, posmap, "TAGE")),
                educ  = _sipp_to_int(_classic_field(line, posmap, "EEDUCATE")),
                wgt   = _sipp_to_float(_classic_field(line, posmap, "WPFINWGT")),
                jobs  = _sipp_classic_job_records(line, posmap),
            ))
        end
    finally
        close(io)
    end
    isempty(rows) && (@warn "    $fname: no parseable records — skipped."; return nothing)

    df = DataFrame(rows)
    n_pairs = 0
    windows_seen = Set{Symbol}()
    for pg in groupby(df, :pid)
        p = sort(DataFrame(pg), :cal_m)
        for i in 1:(nrow(p) - 1)
            r1 = p[i, :]; r2 = p[i + 1, :]
            r2.cal_m - r1.cal_m == 1 || continue                 # consecutive months
            (r1.rmesr == 1 && r2.rmesr == 1) || continue
            (isnothing(r1.tage) || isnothing(r2.tage) ||
             !in_age_range(r1.tage) || !in_age_range(r2.tage)) && continue

            window = assign_window(r2.yr, r2.mn)
            window == :none && continue
            cpi1 = get(SIPP_CPIU, r1.yr, NaN); cpi2 = get(SIPP_CPIU, r2.yr, NaN)
            (isfinite(cpi1) && isfinite(cpi2)) || continue

            isnothing(r2.educ) && continue
            skilled = is_skilled_sipp_classic(r2.educ)
            (isnan(r2.wgt) || r2.wgt <= 0.0) && continue

            jobs1 = Dict(r1.jobs); jobs2 = Dict(r2.jobs)
            c = _wchg_get!(acc, window, SIPP_WCHG_EPS_GRID)
            for (eeno, msum1) in jobs1
                haskey(jobs2, eeno) || continue
                w1 = sipp_real_weekly_wage(msum1,       r1.yr, r1.mn, cpi1)
                w2 = sipp_real_weekly_wage(jobs2[eeno], r2.yr, r2.mn, cpi2)
                (isfinite(w1) && isfinite(w2)) || continue
                _wchg_record_pair!(c, skilled, r2.wgt, w1, w2)
                n_pairs += 1; push!(windows_seen, window)
            end
        end
    end
    panel = sipp_classic_panel_year(fname)
    @printf("    %s (classic panel %s): %d stayer pairs → windows [%s]\n",
            fname, isnothing(panel) ? "?" : string(panel), n_pairs,
            join(sort(collect(string.(windows_seen))), ", "))
    return nothing
end

# Resolve a classic panel's core dictionary from its wave-1 core dictionary and
# return (filename, posmap). `dict_dir` is the directory the panel's own .dat
# cores live in (its per-year subfolder), so the dictionary is read from beside
# the data it describes rather than from the top sipp/ directory. Census ships
# the core dictionary for wave 1 only, so a single dictionary documents every
# wave of the panel (the core layout is constant within a panel). The plain name
# l<YY>puw1d.txt is preferred; the 2008 panel's plain file no longer exists on
# Census and its real dictionary survives only as a suffixed revision, so `.old2`
# then `.old.2` are tried as fallbacks. A candidate is accepted only if its
# parsed posmap carries every CLASSIC_REQUIRED variable, which rejects the
# truncated `.old`/`.obsolete` stubs — those are never tried by name, and the
# same guard excludes any stub that happens to sit under one of the accepted
# names. Returns `nothing` (with a loud @warn naming the panel) when no usable
# dictionary is found.
function _resolve_classic_dict(dict_dir::String, panel_yy::AbstractString)
    candidates = ["l$(panel_yy)puw1d.txt",
                  "l$(panel_yy)puw1d.txt.old2",
                  "l$(panel_yy)puw1d.txt.old.2"]
    for name in candidates
        path = joinpath(dict_dir, name)
        isfile(path) || continue
        posmap = parse_classic_dict(path)
        missing_vars = filter(v -> !haskey(posmap, v), CLASSIC_REQUIRED)
        isempty(missing_vars) && return (name, posmap)
        @warn "    classic panel 20$panel_yy: dictionary $name missing " *
              "$(join(missing_vars, ", ")) — trying next candidate."
    end
    @warn "    classic panel 20$panel_yy: no usable wave-1 core dictionary " *
          "(looked for $(join(candidates, ", "))) — panel skipped."
    return nothing
end

# ── Directory scan: dispatch each raw SIPP file to the right reader ──────────
# Walk sipp_dir recursively so files are found wherever they sit: the redesign
# annual releases directly under sipp/, and the classic panels plus the 2014
# per-wave files inside per-year subfolders (2001/, 2004/, 2008/, 2014/).
# Redesign cores are pu*.csv (both pu<Y>.csv and pu2014w<N>.csv); classic cores
# are l<YY>puw<N>.dat. Each classic panel's cores share one subfolder and its
# wave-1 core dictionary is resolved once from that same subfolder and applied
# to every wave — Census ships the core dictionary for wave 1 only, so a
# per-wave lookup would drop every wave but the first. Compressed cores with no
# extracted sibling are reported (with the extract command) and skipped; the
# release ships extracted, so no zip/gzip codec is needed.
function _sipp_accumulate_dir!(acc::Dict{Symbol,WchgCounts}, sipp_dir::String)
    redesign = String[]                       # full paths to pu*.csv cores
    classic  = String[]                       # full paths to l<YY>puw<N>.dat cores
    for (root, _, files) in walkdir(sipp_dir)
        for f in files
            if occursin(r"^pu.*\.csv$"i, f)
                push!(redesign, joinpath(root, f))
            elseif occursin(r"^l\d{2}puw\d+\.dat$"i, f)
                push!(classic, joinpath(root, f))
            elseif occursin(r"^(pu.*\.csv|l\d{2}puw\d+\.dat)\.(gz|zip)$"i, f)
                _sipp_warn_compressed(joinpath(root, f))
            end
        end
    end

    for path in sort(redesign)
        _sipp_accumulate_redesign!(acc, path)
    end

    # Group classic cores by panel key (the YY in l<YY>puw<N>.dat). Every wave of
    # a panel must live in one directory, since its wave-1 dictionary is resolved
    # from that directory; a panel split across directories is a layout the reader
    # cannot key a single dictionary to, so it is warned and skipped.
    panels = Dict{String,Vector{String}}()
    for path in classic
        yy = match(r"^l(\d{2})puw"i, basename(path)).captures[1]
        push!(get!(panels, yy, String[]), path)
    end
    for yy in sort(collect(keys(panels)))
        paths = sort(panels[yy])
        dirs  = unique(dirname.(paths))
        if length(dirs) > 1
            @warn "    classic panel 20$yy: waves split across directories " *
                  "($(join(dirs, ", "))) — cannot key one wave-1 dictionary to the panel; skipped."
            continue
        end
        resolved = _resolve_classic_dict(dirs[1], yy)
        isnothing(resolved) && continue
        dict_name, posmap = resolved
        @info "    classic panel 20$yy: dictionary $dict_name applied to $(length(paths)) wave(s)"
        for path in paths
            _sipp_accumulate_classic!(acc, path, posmap)
        end
    end
    return nothing
end

# ── Hazard, effective N, and the production rates ───────────────────────────
# Probability → continuous-time monthly hazard, p at horizon h=1 (consecutive
# month-pairs). NaN for an empty denominator so the window is held out; 0 when
# no pair changed (the hazard is non-negative, so no −log(1) negative zero).
function _sipp_hazard(chg::Float64, den::Float64)::Float64
    den > 0.0 || return NaN
    p = min(chg / den, 1.0 - 1e-12)
    return p > 0.0 ? -log(1.0 - p) : 0.0
end

# Kish effective sample size neff = (Σw)² / Σw², carried to Σ̂ so sigma.jl can
# place a binomial-proportion (delta-method) variance on each hazard without
# re-reading the SIPP micro-data.
_sipp_neff(den::Float64, den2::Float64)::Float64 = den2 > 0.0 ? den^2 / den2 : 0.0

# Index of the production threshold within the ε grid.
_sipp_prod_eps_idx() = findfirst(==(SIPP_WCHG_EPS), SIPP_WCHG_EPS_GRID)

function make_sipp_wchg()
    @info "Stage 6b: SIPP within-job wage-change rates (wchg_rate_U, wchg_rate_S)..."

    sipp_dir = joinpath(RAW_DIR, "sipp")
    empty_out() = DataFrame(window = String[], wchg_rate_U = Float64[], wchg_rate_S = Float64[],
                            neff_U = Float64[], neff_S = Float64[])
    if !isdir(sipp_dir) || isempty(readdir(sipp_dir))
        @warn "  $sipp_dir absent or empty — no SIPP data yet. Writing an empty " *
              "sipp_wchg_rates.csv; every window's wchg_rate_U/S stays NaN and auto-" *
              "holds out of the SMM objective."
        df = empty_out()
        CSV.write(joinpath(DERIVED_DIR, "sipp_wchg_rates.csv"), df)
        return df
    end

    # Accumulate weighted stayer-pair counts from every reader into one shared
    # per-window struct, THEN form p and the hazard per window at the production
    # threshold. A window with no data for a source simply gets no contribution.
    acc = Dict{Symbol,WchgCounts}()
    _sipp_accumulate_dir!(acc, sipp_dir)

    ip = _sipp_prod_eps_idx()
    rows = NamedTuple[]
    for wname in WINDOWS_ORDER
        haskey(acc, wname) || continue
        c = acc[wname]
        r_U = _sipp_hazard(c.chg_U[ip], c.den_U[ip])
        r_S = _sipp_hazard(c.chg_S[ip], c.den_S[ip])
        n_U = _sipp_neff(c.den_U[ip], c.den2_U[])
        n_S = _sipp_neff(c.den_S[ip], c.den2_S[])
        push!(rows, (window = wname, wchg_rate_U = r_U, wchg_rate_S = r_S,
                     neff_U = n_U, neff_S = n_S))
        @printf("  %-14s wchg_rate_U=%.5f  wchg_rate_S=%.5f  (neff_U=%.1f, neff_S=%.1f)\n",
                string(wname), r_U, r_S, n_U, n_S)
    end

    covered = [string(r.window) for r in rows]
    heldout = [string(w) for w in WINDOWS_ORDER if !(string(w) in covered)]
    isempty(heldout) ||
        @warn "  SIPP wchg measured for [$(join(covered, ", "))]; NO SIPP data for " *
              "[$(join(heldout, ", "))] — wchg_rate_U/S held out of the SMM objective " *
              "in those windows. Download the SIPP files covering the missing calendar " *
              "years (see data-and-moments §sipp) to close the gap."

    df_out = isempty(rows) ? empty_out() : DataFrame(rows)
    outpath = joinpath(DERIVED_DIR, "sipp_wchg_rates.csv")
    CSV.write(outpath, df_out)
    @info "  Saved: $outpath"

    print_sipp_wchg_robustness(acc)
    return df_out
end

# ── Threshold-robustness diagnostic ─────────────────────────────────────────
# Print the wage-change hazard for all four windows × {U, S} at every ε in the
# grid. Diagnostic only — it shows how sensitive the estimated hazard is to the
# $1 threshold; the production moment (written to sipp_wchg_rates.csv) stays at
# ε = SIPP_WCHG_EPS. A window with no SIPP contribution shows NaN throughout.
function print_sipp_wchg_robustness(acc::Dict{Symbol,WchgCounts})
    println("\n  ── wchg threshold robustness (monthly hazard −log(1−p); production ε = \$$(SIPP_WCHG_EPS)/week) ──")
    @printf("  %-14s", "window")
    for ε in SIPP_WCHG_EPS_GRID
        @printf(" %8s %8s", @sprintf("U ε%g", ε), @sprintf("S ε%g", ε))
    end
    println()
    for wname in WINDOWS_ORDER
        @printf("  %-14s", string(wname))
        c = get(acc, wname, nothing)
        for k in eachindex(SIPP_WCHG_EPS_GRID)
            if isnothing(c)
                @printf(" %8s %8s", "NaN", "NaN")
            else
                @printf(" %8.5f %8.5f",
                        _sipp_hazard(c.chg_U[k], c.den_U[k]),
                        _sipp_hazard(c.chg_S[k], c.den_S[k]))
            end
        end
        println()
    end
    println()
    return nothing
end

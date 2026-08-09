############################################################
# data_processing/validation.jl
#
# Stage 9 — validation and diagnostics, run at the very end of the
# pipeline. Prints the moment table, published-benchmark range checks,
# cross-window direction checks, and the stationary-identity gap. Reads
# the in-memory results (raw training_share), so it is independent of the
# κ adjustment baked into the saved CSVs.
#
# Reads:  phi_calibration.csv, nu_estimation.csv (+ in-memory moments)
# Writes: — (prints only)
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages and path consts come from data_processing_main.jl.
############################################################

# ee_rate_S source comparison. Reads the two derived EE-rate CSVs and prints,
# per window, the SIPP main-job-change hazard against the shipped Census J2J E4
# hazard, their ratio, and the percentage by which J2J exceeds SIPP. Called from
# the driver right after the SIPP EE stage. Guards a missing CSV or window.
function print_ee_source_comparison()
    j2j_path  = joinpath(DERIVED_DIR, "j2j_ee_rates.csv")
    sipp_path = joinpath(DERIVED_DIR, "sipp_ee_rates.csv")
    j2j  = isfile(j2j_path)  ? CSV.read(j2j_path, DataFrame)  : DataFrame()
    sipp = isfile(sipp_path) ? CSV.read(sipp_path, DataFrame) : DataFrame()
    !isempty(j2j)  && hasproperty(j2j, :window)  && (j2j.window  = Symbol.(j2j.window))
    !isempty(sipp) && hasproperty(sipp, :window) && (sipp.window = Symbol.(sipp.window))

    println("\n── ee_rate_S source comparison (shipped = J2J) ──")
    @printf("  %-13s %18s %22s %10s %10s\n",
            "window", "ee_rate_S (SIPP)", "ee_rate_S (J2J, ship)", "J2J/SIPP", "% higher")
    for wname in WINDOWS_ORDER
        sr = isempty(sipp) ? nothing : filter(r -> r.window == wname, sipp)
        jr = isempty(j2j)  ? nothing : filter(r -> r.window == wname, j2j)
        ee_sipp = (!isnothing(sr) && nrow(sr) > 0) ? Float64(sr.ee_rate_S[1]) : NaN
        ee_j2j  = (!isnothing(jr) && nrow(jr) > 0) ? Float64(jr.ee_rate_S[1]) : NaN
        ratio   = (isfinite(ee_sipp) && isfinite(ee_j2j) && ee_sipp != 0.0) ? ee_j2j / ee_sipp : NaN
        pct     = isfinite(ratio) ? (ratio - 1.0) * 100.0 : NaN
        @printf("  %-13s %18s %22s %10s %10s\n",
                string(wname),
                isfinite(ee_sipp) ? @sprintf("%.5f", ee_sipp) : "NaN",
                isfinite(ee_j2j)  ? @sprintf("%.5f", ee_j2j)  : "NaN",
                isfinite(ratio)   ? @sprintf("%.3f", ratio)   : "NaN",
                isfinite(pct)     ? @sprintf("%+.1f", pct)     : "NaN")
    end
    return nothing
end

function run_validation(all_moments)
    @info "Stage 9: Validation diagnostics..."

    println("\n" * "="^80)
    println("VALIDATION REPORT")
    println("="^80)

    # ── 1. Moment values across windows ───────────────────────────
    println("\n── 1. Moments across windows ──")
    wide = DataFrame(moment = [string(m) for m in MOMENT_NAMES])
    for wname in WINDOWS_ORDER
        haskey(all_moments, wname) || continue
        mdf = all_moments[wname]
        wide[!, string(wname)] = mdf.value
    end
    display(wide)

    # ── 2. Sanity checks with published / data-implied benchmarks ──
    # Wage moments are RAW log real weekly earnings (no within-window
    # normalisation; the model's aggregate scale A absorbs the dollar level).
    # base_covid targets, for reference: mean_wage_U≈6.67, mean_wage_S≈7.06,
    # p25≈6.31/6.74, p50≈6.68/7.06, p75 one quartile up, emp_var≈0.31/0.27. The level bands below
    # bracket plausible US log weekly earnings (~e^6 = $400 to ~e^7.6 = $2000).
    println("\n── 2. Benchmark comparisons ──")
    benchmarks = Dict(
        :skilled_share => (name="Skilled share (BA+)", lo=0.20, hi=0.50),
        # Range set against the NSC target (v17.0), not the old CPS count.
        # The upper bound is the model's own ceiling ν/(φ+ν) ≈ 0.127: above it
        # training_share is unattainable at ANY parameter vector, so a target
        # there is a construction defect rather than a hard fit.
        :training_share => (name="Training share (NSC, attrition-adjusted)", lo=0.02, hi=0.127),
        :theta_U => (name="Unskilled tightness V/U", lo=0.1, hi=5.0),
        :theta_S => (name="Skilled tightness V/U", lo=0.1, hi=10.0),
        :jfr_U => (name="Unskilled JF rate (monthly)", lo=0.10, hi=0.50),
        :jfr_S => (name="Skilled JF rate (monthly)", lo=0.10, hi=0.50),
        :sep_rate_U => (name="Unskilled EU sep rate (monthly)", lo=0.005, hi=0.05),
        :sep_rate_S => (name="Skilled EU sep rate (monthly)", lo=0.002, hi=0.03),
        :ee_rate_S => (name="Skilled EE rate (monthly)", lo=0.005, hi=0.05),
        # Wage levels — RAW log real weekly earnings (not normalised).
        :mean_wage_U => (name="Mean log wage unskilled", lo=6.0, hi=7.2),
        :mean_wage_S => (name="Mean log wage skilled", lo=6.4, hi=7.6),
        :p25_wage_U => (name="p25 log wage unskilled", lo=5.8, hi=7.0),
        :p25_wage_S => (name="p25 log wage skilled", lo=6.2, hi=7.3),
        :p50_wage_U => (name="p50 log wage unskilled", lo=6.0, hi=7.2),
        :p50_wage_S => (name="p50 log wage skilled", lo=6.4, hi=7.6),
        :p75_wage_U => (name="p75 log wage unskilled", lo=6.2, hi=7.4),
        :p75_wage_S => (name="p75 log wage skilled", lo=6.6, hi=7.8),
        # Wage dispersion (includes σ_w measurement component on the data side).
        :emp_var_U => (name="Var log wage unskilled", lo=0.10, hi=0.50),
        :emp_var_S => (name="Var log wage skilled", lo=0.10, hi=0.50),
        :wage_premium => (name="Log skill premium", lo=0.20, hi=0.80),
        # Cross-market / duration moments — data-implied bands (base_covid
        # targets ≈ 0.239 / 0.206 / 0.256). LTU rises in crisis windows, so the
        # upper band is wider than the baseline value.
        :overlap_UgtS => (name="P(w_U > med w_S) overlap", lo=0.05, hi=0.45),
        :overlap_SltU => (name="P(w_S < med w_U) overlap", lo=0.05, hi=0.45),
        :ltu_share_S => (name="Skilled long-term-unemp share (≥27wk)", lo=0.05, hi=0.50),
    )

    n_flags = 0
    for (mname, bm) in benchmarks
        for wname in WINDOWS_ORDER
            haskey(all_moments, wname) || continue
            mdf = all_moments[wname]
            row = filter(r -> r.moment == string(mname), mdf)
            isempty(row) && continue
            val = row.value[1]
            !isfinite(val) && continue
            flag = val < bm.lo || val > bm.hi ? "⚠ OUT OF RANGE" : "✓"
            if flag != "✓"
                @printf("  %s  %-32s %12s = %8.4f  (expected %.4f–%.4f)\n",
                        flag, bm.name, wname, val, bm.lo, bm.hi)
                n_flags += 1
            end
        end
    end
    println("  Flagged: $n_flags values outside expected ranges")

    # ── 3. Crisis signatures ──────────────────────────────────────
    # Unemployment rises in both crises; what separates them is where the
    # economy sits relative to the Beveridge curve. A movement ALONG the curve
    # (u up, θ down) is a symmetric demand contraction; an OUTWARD shift
    # (u up, θ up) is a matching-efficiency or reallocation shock.
    #
    # This block deliberately does not encode a single expected direction for
    # θ. An earlier version asserted "θ_U should fall" in both pairs, which
    # flagged the COVID pattern as an anomaly — it is the finding, not a data
    # defect. What IS asserted is the part common to any crisis: unemployment
    # rises. The θ direction is reported and classified, not judged.
    println("\n── 3. Crisis signatures (Beveridge position) ──")
    for (lab, wb, wc) in (("FC", :base_fc, :crisis_fc),
                          ("COVID", :base_covid, :crisis_covid))
        haskey(all_moments, wb) && haskey(all_moments, wc) || continue
        get_m(w, m) = filter(r -> r.moment == string(m), all_moments[w]).value[1]
        u1, u2 = get_m(wb, :ur_U),    get_m(wc, :ur_U)
        t1, t2 = get_m(wb, :theta_U), get_m(wc, :theta_U)
        s1, s2 = get_m(wb, :theta_S), get_m(wc, :theta_S)
        (!isfinite(u1) || !isfinite(u2) || !isfinite(t1) || !isfinite(t2)) && continue
        # The one assertion: a crisis raises unemployment. If this fails the
        # window definitions are wrong, not the economics.
        flag = u2 > u1 ? "✓" : "⚠ CHECK WINDOWS"
        sig  = t2 < t1 ? "along the curve  (symmetric demand contraction)" :
                         "OUTWARD shift    (matching / reallocation shock)"
        @printf("  %s  %-6s ur_U %+6.1f%%   θ_U %+6.1f%%   θ_S %+6.1f%%   %s\n",
                flag, lab, 100*(u2-u1)/u1, 100*(t2-t1)/t1, 100*(s2-s1)/s1, sig)
    end
    println("  The two pairs must not receive the same classification: telling")
    println("  them apart is what the model is for.")

    # ── 4. Stationary-identity gap by window ──────────────────────
    # Model identity (d ≡ 0 in stationary equilibrium, strict
    # training_share convention):
    #     skilled_share * (1 - training_share)
    #   = (φ/ν) * training_share
    # A small residual is informative; a large residual flags
    # non-stationarity in the window. Computed on the SHIPPED training_share
    # (the NSC target), which is the quantity the SMM is asked to match — an
    # earlier version ran it on the raw CPS count and reported gaps roughly
    # three times larger, for a number no estimation ever sees.
    println("\n── 4. Stationary-identity gap (skilled_share / training_share / φ / ν) ──")

    phi_cal = let p = joinpath(DERIVED_DIR, "phi_calibration.csv")
        isfile(p) ? CSV.read(p, DataFrame).phi[1] : NaN
    end
    nu_tbl = let p = joinpath(DERIVED_DIR, "nu_estimation.csv")
        isfile(p) ? CSV.read(p, DataFrame) : nothing
    end
    nu_lookup = Dict{Symbol, Float64}()
    if nu_tbl !== nothing
        for r in eachrow(nu_tbl)
            nu_lookup[Symbol(r.window)] = r.nu
        end
    end
    # Each crisis pair shares the ν of its baseline
    nu_for = Dict(
        :base_fc      => get(nu_lookup, :base_fc, NaN),
        :crisis_fc    => get(nu_lookup, :base_fc, NaN),
        :base_covid   => get(nu_lookup, :base_covid, NaN),
        :crisis_covid => get(nu_lookup, :base_covid, NaN),
    )

    @printf("    φ = %.6f (pooled)\n", phi_cal)
    for wname in WINDOWS_ORDER
        haskey(all_moments, wname) || continue
        mdf = all_moments[wname]
        ss  = filter(r -> r.moment == "skilled_share", mdf).value[1]
        ts  = filter(r -> r.moment == "training_share", mdf).value[1]
        nu  = nu_for[wname]
        if isfinite(ss) && isfinite(ts) && isfinite(phi_cal) && isfinite(nu) && ts > 0
            lhs = ss * (1 - ts)
            rhs = (phi_cal / nu) * ts
            gap = lhs / rhs - 1
            # Sign is the informative part. LHS > RHS (gap > 0) means the data
            # carry a larger skilled stock than the calibrated flow into it can
            # sustain — the model must find the extra skilled mass somewhere
            # other than the training margin. LHS < RHS means the opposite:
            # more training than the skilled stock reflects, i.e. the trainee
            # state is larger than a stationary read of φ/ν implies.
            side = gap > 0 ? "skilled stock > flow implies" :
                             "training stock > flow implies"
            @printf("    %-14s  ν = %.6f  gap = %+.3f   %s\n",
                    wname, nu, gap, side)
        end
    end

    println("\n" * "="^80)
    println("END VALIDATION REPORT")
    println("="^80)
end
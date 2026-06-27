############################################################
# data_processing/validation.jl
#
# Stage 9 — validation and diagnostics, run at the very end of the
# pipeline. Prints the moment table, published-benchmark range checks,
# cross-window direction checks, the stationary-identity gap, and Σ̂
# condition numbers. Reads the in-memory results (raw training_share),
# so it is independent of the κ adjustment baked into the saved CSVs.
#
# Reads:  phi_calibration.csv, nu_estimation.csv (+ in-memory moments/Σ̂)
# Writes: — (prints only)
#
# Plain include() file: definitions only, no top-level execution.
# `using` packages and path consts come from data_processing_main.jl.
############################################################

function run_validation(all_moments, all_sigma)
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
    # p25≈6.31/6.74, p50≈6.68/7.06, emp_var≈0.31/0.27. The level bands below
    # bracket plausible US log weekly earnings (~e^6 = $400 to ~e^7.6 = $2000).
    println("\n── 2. Benchmark comparisons ──")
    benchmarks = Dict(
        :skilled_share => (name="Skilled share (BA+)", lo=0.20, hi=0.50),
        :training_share => (name="Training share (NILF ∩ train / pop)", lo=0.005, hi=0.05),
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
        # Wage dispersion (includes σ_w measurement component on the data side).
        :emp_var_U => (name="Var log wage unskilled", lo=0.10, hi=0.50),
        :emp_var_S => (name="Var log wage skilled", lo=0.10, hi=0.50),
        :wage_premium => (name="Log skill premium", lo=0.20, hi=0.80),
        # New cross-market / duration moments — data-implied bands (base_covid
        # targets ≈ 0.239 / 0.206 / 0.256). LTU rises in crisis windows, so the
        # upper band is wider than the baseline value. Provisional ranges.
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

    # ── 3. Cross-window direction checks ──────────────────────────
    println("\n── 3. Cross-window direction checks ──")
    expected_directions = [
        (:ur_U,   :base_fc,    :crisis_fc,    +1, "UR_U should rise in FC"),
        (:ur_U,   :base_covid, :crisis_covid, +1, "UR_U should rise in COVID"),
        (:jfr_U,  :base_fc,    :crisis_fc,    -1, "JFR_U should fall in FC"),
        (:theta_U,:base_fc,    :crisis_fc,    -1, "θ_U should fall in FC"),
        (:theta_U,:base_covid, :crisis_covid, -1, "θ_U should fall in COVID"),
    ]
    for (mname, w1, w2, expected_sign, desc) in expected_directions
        haskey(all_moments, w1) && haskey(all_moments, w2) || continue
        v1 = filter(r -> r.moment == string(mname), all_moments[w1]).value[1]
        v2 = filter(r -> r.moment == string(mname), all_moments[w2]).value[1]
        (!isfinite(v1) || !isfinite(v2)) && continue
        actual_sign = sign(v2 - v1)
        flag = actual_sign == expected_sign ? "✓" : "⚠ UNEXPECTED"
        @printf("  %s  %s: %.4f → %.4f (Δ=%.4f)\n", flag, desc, v1, v2, v2-v1)
    end

    # ── 4. Stationary-identity gap by window ──────────────────────
    # Model identity (d ≡ 0 in stationary equilibrium, strict
    # training_share convention):
    #     skilled_share * (1 - training_share)
    #   = (φ/ν) * training_share
    # A small residual is informative; a large residual flags
    # non-stationarity in the window.
    println("\n── 5. Stationary-identity gap (skilled_share / training_share / φ / ν) ──")

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
            @printf("    %-14s  ν = %.6f  gap = %+.3f  (LHS/RHS - 1)\n",
                    wname, nu, gap)
        end
    end

    # ── 5. Σ̂ diagnostics ─────────────────────────────────────────
    println("\n── 5. Σ̂ condition numbers ──")
    for wname in WINDOWS_ORDER
        haskey(all_sigma, wname) || continue
        Sig = all_sigma[wname]
        d = diag(Sig)
        n_zero = count(d .<= 0)
        cn = cond(Sig)
        @printf("  %s: κ = %.2e, zero-diagonal = %d\n", wname, cn, n_zero)
    end

    println("\n" * "="^80)
    println("END VALIDATION REPORT")
    println("="^80)
end
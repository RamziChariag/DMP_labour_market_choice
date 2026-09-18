############################################################
# smm/settings.jl — the environment-settings layer, declared once
#
# Every run setting that can be driven from outside the source passes through
# `env_setting` here, and every such setting is declared in SETTINGS_REGISTRY below.
# The two halves do different jobs and neither duplicates the other:
#
#   the call site  carries the DEFAULT and the comment justifying it, because that is
#                  where a reader asking "why this value" looks;
#   the registry   carries the CLASSIFICATION the source cannot express — which
#                  consumer the value has to reach, and whether that consumer's code
#                  path is live under the shipped configuration.
#
# `check_forwarding.jl` reads both and fails when they disagree: a key read with no
# registry row, or a row no code reads. That is what makes adding a setting a single
# declaration rather than four disconnected edits (env key, struct field, keyword
# argument, header print), which is how four SA settings came to be computed and never
# forwarded between v18.4.0 and v19.0.0.
#
# See SETTINGS.md at the repo root for the lifecycle rules these types enforce.
############################################################

# ============================================================
# The registry
# ============================================================

"""
    SettingDecl

Classification of one `ROYSEARCH_*` environment setting.

Fields
  key      :: Symbol   the key without the `ROYSEARCH_` prefix
  consumer :: String   the function that READS the value, `file:function` form. The
                       chain from the key to this function is what a forwarding audit
                       has to verify; naming it makes the intended endpoint explicit
                       rather than inferred from a grep.
  status   :: Symbol   :live      reached and used under the shipped configuration
                       :inert     forwarded correctly, but its code path is not taken
                                  under the shipped method (documented, not a defect)
  note     :: String   why an :inert setting is inert, or what a :live one selects.
                       Required for :inert — an unexplained inert switch is
                       indistinguishable from a broken one.
"""
struct SettingDecl
    key      :: Symbol
    consumer :: String
    status   :: Symbol
    note     :: String
end

"""
Every `ROYSEARCH_*` key the estimation and standard-error entry points read.

Grouped by consumer. A new setting is not finished until it has a row here and
`check_forwarding.jl` runs clean — that check is what turns this table from
documentation into a gate.
"""
const SETTINGS_REGISTRY = [
    # ── smm_main.jl: window, weighting and seeding ──────────────────────────
    SettingDecl(:WINDOW,        "smm_main.jl:top level",       :live,
                "estimation window; must be a key of data/derived/windows.json"),
    SettingDecl(:INIT_MODE,     "smm_main.jl:top level",       :live,
                ":default, :warmstart or :clusters — selects how the optimiser is seeded"),
    SettingDecl(:W_COND_TARGET, "smm_params.jl:load_weight_matrix", :live,
                "0.0 diagonal-σ, 2.0 equal weights; also selects the bundle suffix"),
    SettingDecl(:LAMBDA_W,      "moments.jl:calibrate_sigma_w", :live,
                "wage-reliability knob; σ_w is calibrated from it, never estimated"),
    SettingDecl(:USE_AS_SEED,           "smm_main.jl:top level", :live,
                "false perturbs the entered init before estimating"),
    SettingDecl(:SEED_PERTURB_FRAC,     "smm_params.jl:perturb_free_init", :inert,
                "read only when USE_AS_SEED = false, which is not the shipped default"),
    SettingDecl(:CLUSTERS_FORCE_REGEN,  "candidates.jl:load_or_generate_candidates", :inert,
                "read only under INIT_MODE = :clusters; the shipped default is :warmstart"),
    SettingDecl(:INCLUDE_PREV_OPTIMUM,  "smm_main.jl:top level", :inert,
                "read only under INIT_MODE = :clusters; the shipped default is :warmstart"),

    # ── Simulated annealing ────────────────────────────────────────────────
    SettingDecl(:SA_MAX_ITER,    "smm.jl:_sa_loop", :live,   "iteration cap per chain"),
    SettingDecl(:SA_STEP,        "smm.jl:sa_proposal_scale", :live,
                "FALLBACK step for a coordinate the width bisection cannot move, not the "
                * "proposal scale: the step is per coordinate and seeded from measured "
                * "ΔQ=1 half-widths. 0 of 23 coordinates needed it at base_fc"),
    SettingDecl(:SA_T0_REL,      "smm.jl:_sa_loop", :live,
                "move size, as a fraction of Q, that T0 keeps live; also start-dependent"),
    SettingDecl(:SA_HALFLIFE,    "smm.jl:_sa_loop", :live,
                "geometric cooling half-life; 0 selects the logarithmic branch instead"),
    SettingDecl(:SMM_POLISH, "smm_main.jl:§9", :live,
                "run an LBFGS polish from the annealed point; SA stops on a criterion rule and leaves the gradient large, which invalidates J⁻¹"),
    SettingDecl(:SMM_POLISH_FD_STEP, "smm.jl:run_smm", :live,
                "central-difference step for the polish gradient, in t units. Measured: every column is stable over h ∈ [3e-7, 3e-6] and several break by 1e-4"),
    SettingDecl(:SMM_POLISH_G_TOL, "smm.jl:run_smm", :live,
                "gradient ∞-norm at which the polish stops; a positive value also disables the function/step tolerances and the rate callback, which stop on a small move rather than a small gradient"),
    SettingDecl(:SMM_POLISH_MAX_ITER, "smm.jl:run_smm", :live,
                "LBFGS iteration cap; each iteration costs 2d gradient solves plus a line search, so this is not an evaluation count"),
    SettingDecl(:SA_SCALE_P_MOVE,  "smm.jl:sa_proposal_scale", :live,
                "mask density in units of 1/d: the proposal moves one forced coordinate "
                * "plus a Binomial(d−1, p) tail, mean ≈ 2 of 23 at 1.0. No fixed count "
                * "is configured anywhere — the realised count is reported in [SA config]"),
    SettingDecl(:SA_SCALE_PER_K,  "smm.jl:sa_proposal_scale", :inert,
                "0 by default, which disables the sparsity scan. The scan is a "
                * "diagnostic for the feasibility spread across sparsities; its k* was "
                * "measured against annealing chains and lost, so it sets nothing"),
    SettingDecl(:SA_SCALE_SIGMA,  "smm.jl:sa_proposal_scale", :live,
                "measurement step as a fraction of each coordinate's own ΔQ<1 width"),
    SettingDecl(:SA_RATE_TOL,    "smm.jl:_sa_loop", :live,  "rate-based stop, ΔQ per 100 iterations"),
    SettingDecl(:SA_RATE_SPAN,   "smm.jl:_sa_loop", :live,  "productive iterations the rate stop spans"),
    SettingDecl(:SA_MAX_REHEATS, "smm.jl:_sa_loop", :live,  "reheat cap per run; 0 unlimited"),

    # ── Differential evolution ─────────────────────────────────────────────
    SettingDecl(:DE_MAX_ITER,     "smm.jl:_run_de", :live, "generation cap"),
    SettingDecl(:DE_POP_SIZE,     "smm.jl:_run_de", :live, "0 ⇒ 10·n_free_params"),
    SettingDecl(:DE_F,            "smm.jl:_run_de", :inert,
                "overridden by the generator's yield table whenever DE_ADAPT_FCR is "
                * "true, which is the shipped default"),
    SettingDecl(:DE_CR,           "smm.jl:_run_de", :inert,
                "overridden by the generator's yield table whenever DE_ADAPT_FCR is "
                * "true, which is the shipped default"),
    SettingDecl(:DE_ADAPT_FCR,    "smm.jl:_run_de", :live,
                "selects whether DE_F and DE_CR are used as set or read off the draws"),
    # DE_PATIENCE retired v19.5.0: it claimed :live and _run_de never read it. The DE stall
    # test is DE_REHEAT_FLAT + DE_REHEAT_RATE. The de_patience field survives in
    # SMMRunParams for bundle compatibility only; see CHANGELOG.md.
    SettingDecl(:DE_AVG_TOL,      "smm.jl:_run_de", :inert,
                "0 disables, which is the shipped default: the measure ranks a "
                * "prematurely converged run as the most converged one (ρ = +0.915 "
                * "with achieved ΔQ across a six-configuration sweep)"),
    SettingDecl(:DE_GEN_PER_K,    "smm.jl:generate_population", :live,
                "candidates drawn at each sparsity k = 1:n_free"),
    SettingDecl(:DE_LOCAL_SIGMA,  "smm.jl:generate_population", :live,
                "perturbation as a fraction of each coordinate's own ΔQ<1 width"),
    SettingDecl(:DE_REHEAT_FLAT,  "smm.jl:_run_de", :live, "flat generations before a reheat"),
    SettingDecl(:DE_REHEAT_RATE,  "smm.jl:_run_de", :live, "improved-member rate below which a reheat fires"),
    SettingDecl(:DE_MAX_REHEATS,  "smm.jl:_run_de", :live, "reheat cap; a barren reheat ends the run first"),

    # ── Nelder-Mead ────────────────────────────────────────────────────────
    # The shipped method is :sa_de, which never enters the Nelder-Mead branch of
    # run_smm. Every NM_* key is therefore inert by PATH, not broken: each is
    # forwarded correctly and would be read the moment method changed to
    # :neldermead / :lbfgs / :bfgs.
    SettingDecl(:NM_MAX_ITER,     "smm.jl:run_smm (:neldermead branch)", :inert,
                "the shipped method is :sa_de, which never enters this branch"),
    SettingDecl(:NM_F_TOL,        "smm.jl:run_smm (:lbfgs / :bfgs branches)", :inert,
                "unread by Optim's Nelder-Mead (assess_convergence returns f_converged "
                * "as a literal false); live only for the gradient methods"),
    SettingDecl(:NM_X_TOL,        "smm.jl:run_smm (:lbfgs / :bfgs branches)", :inert,
                "unread by Optim's Nelder-Mead, as NM_F_TOL"),
    SettingDecl(:NM_G_TOL,        "smm.jl:run_smm (:neldermead branch)", :inert,
                "Nelder-Mead's only built-in test, but unreachable at Q ~ 1e3, and the "
                * "branch itself is not entered under :sa_de"),
    SettingDecl(:NM_NO_IMPROVE,   "smm.jl:run_smm (:neldermead branch)", :inert,
                "the shipped method is :sa_de, which never enters this branch"),
    SettingDecl(:NM_RATE_TOL,     "smm.jl:run_smm (:neldermead branch)", :inert,
                "the shipped method is :sa_de, which never enters this branch"),
    SettingDecl(:NM_RATE_SPAN,    "smm.jl:run_smm (:neldermead branch)", :inert,
                "the shipped method is :sa_de, which never enters this branch"),
    SettingDecl(:NM_SIMPLEX_STEP, "smm.jl:Optim.simplexer(FeasibleSimplexer)", :inert,
                "builds the initial simplex, which only the Nelder-Mead branch builds"),

    # ── Width diagnostic, shared by the DE generator and the NM simplex ────
    SettingDecl(:WIDTH_DQ, "smm.jl:_feasible_widths", :live,
                "the ΔQ contour every measured step scale is taken against; absolute "
                * "in Q units, just above the measured evaluation-noise floor"),

    # ── MCMC_main.jl: DE-MC ────────────────────────────────────────────────
    SettingDecl(:MCMC_N,          "demc.jl:run_demc", :live, "chains; 0 ⇒ 2·d"),
    SettingDecl(:MCMC_GENS,       "demc.jl:run_demc", :live, "generation budget"),
    SettingDecl(:MCMC_CR,         "demc.jl:run_demc", :live, "per-coordinate perturbation probability"),
    SettingDecl(:MCMC_DELTA,      "demc.jl:run_demc", :live, "difference pairs per proposal"),
    SettingDecl(:MCMC_B_MULT,     "demc.jl:run_demc", :live, "multiplicative jitter on the difference vector"),
    SettingDecl(:MCMC_B_ADD,      "demc.jl:run_demc", :live, "additive isotropic jitter"),
    # Was read via env_setting with no registry row — caught by check_forwarding.jl when
    # the stop rule was added, not by the change that introduced it. The audit works.
    SettingDecl(:MCMC_OUTLIER_BURN_ONLY, "demc.jl:run_demc", :live,
                "false = LMR timing (replacement every generation, mpi_mcmc_mod.f90:421-427); " *
                "true = burn-in only, which keeps the retained sample reversible"),
    SettingDecl(:MCMC_BURN,       "demc.jl:run_demc", :live,
                "burn-in fraction; 0.9 matches LMR's 'last 1000 of 10,000' (Appendix C p.86)"),
    SettingDecl(:MCMC_PRIOR,      "MCMC_main.jl:logposterior", :live,
                ":flat_t (LMR's target, −Q/2) or :flat_theta (−Q/2 + logjac_box); changes the sampled density, so a table must say which"),
    SettingDecl(:MCMC_WIDTHS_CSV, "MCMC_main.jl:init widths", :live,
                "path to the criterion-width table for MCMC_INIT = :widths; empty means the default name for this window. Produced by code/tools/width_audit.jl. UNREACHABLE in v20.0.0: MCMC_INIT = :widths requires MCMC_SPACE = :theta, which is disabled"),
    SettingDecl(:MCMC_SPACE,      "MCMC_main.jl:logposterior", :live,
                ":t (the only working value, and the shipped default) samples the logistic preimage and is the v20.x estimand. :theta would sample the natural parameter with the economic box enforced by prior rejection, but is DISABLED by a guard: the proposal has no per-coordinate scale in θ and the constrained-vector flag reaches only logposterior, so it runs and reports nonsense. Selects the ESTIMAND, not just the numerics — a table must say which"),
    SettingDecl(:MCMC_T_BOX,      "MCMC_main.jl:logposterior", :live,
                "half-width of the prior box on t; 20.0 matches LMR (main_mpi.f90:146-147). Q is asymptotically flat in every coordinate, so this is what makes the target proper — Inf gives the improper target used before v20.0.0"),
    SettingDecl(:MCMC_PRINT_EVERY, "demc.jl:run_demc", :live,
                "generations between progress lines; the acc/dlp/esjd averaging window"),
    SettingDecl(:MCMC_CHECK_EVERY, "demc.jl:run_demc", :live,
                "generations between sequential stop checks; 0 disables the stop entirely"),
    SettingDecl(:MCMC_INIT_DISPERSE, "demc.jl:run_demc", :live,
                "multiplier on the criterion widths at init; >1 is required for the convergence gate to have power"),
    SettingDecl(:MCMC_GAMMA_ADAPT,  "demc.jl:run_demc", :live,
                "adapt the DE scale toward the acceptance optimum; false restores ter Braak's fixed gamma"),
    SettingDecl(:MCMC_GAMMA_TARGET, "demc.jl:run_demc", :live,
                "acceptance the adaptation drives toward (Roberts-Gelman-Gilks 0.234)"),
    SettingDecl(:MCMC_GAMMA_ETA,    "demc.jl:run_demc", :live,
                "Robbins-Monro step for the gamma adaptation; the realised step is eta/sqrt(k)"),
    SettingDecl(:MCMC_GAMMA_EVERY,  "demc.jl:run_demc", :live,
                "generations between gamma adaptations; 0 disables it"),
    SettingDecl(:MCMC_MCSE_TARGET, "mcmc_diagnostics.jl:stop_rule", :live,
                "MCSE(mean)/sd(pooled) below which the reported mean and sd are accurate enough to stop; the accuracy gate as of v21.0.0"),
    SettingDecl(:MCMC_MOVES_MIN,  "mcmc_diagnostics.jl:stop_rule", :live,
                "accepted moves in the retained half required to stop; the gated quantity as of v19.3.0"),
    SettingDecl(:MCMC_DRIFT_FLAT, "mcmc_diagnostics.jl:stop_rule", :live,
                "log units the running max may climb between checks and still count as flat"),
    SettingDecl(:MCMC_ACC_FLOOR,  "mcmc_diagnostics.jl:stop_rule", :live,
                "abort-and-diagnose when windowed acceptance falls below this at two consecutive checks"),
    SettingDecl(:MCMC_ESS_MIN,    "mcmc_diagnostics.jl:converged_sequential", :live,
                "per-coordinate ESS the sequential stop requires of NON-EXEMPT coordinates; \
                 450 = MCSE(q05) ≤ 0.10·sd, the precision of a reported interval endpoint. \
                 0.0 ⇒ use min_ess(d), the joint-volume floor, which is a different and much \
                 larger requirement"),
    SettingDecl(:MCMC_CHECKPOINT, "MCMC_main.jl:promote!", :live,
                "promote each new best point to the warm-start bundle during the run"),
    SettingDecl(:MCMC_JAC_ONLY,   "MCMC_main.jl:top level", :live,
                "skip the chain and take Ĵ = Ĝ'WĜ from a local design"),
    SettingDecl(:MCMC_JAC_ALLOW_GAPS, "MCMC_main.jl:top level", :live,
                "report CONDITIONAL standard errors on the measured subspace when :fd cannot \
                 measure every column, instead of raising; off by default because a conditional \
                 column is not comparable to an unconditional one"),
    SettingDecl(:MCMC_JAC_METHOD, "MCMC_main.jl:top level", :live,
                ":fd (shipped) builds Ĝ by per-coordinate central differences with the step set \
                 from the moments' own response and refined by Richardson; :design fits a least \
                 squares plane on a cloud of radius rel_step·(ub−lb), which makes the derivative \
                 depend on the search box and measured a median moment R² of 0.771 at the \
                 shipped radius"),
    # :live as of v19.3.0 — the shipped MCMC_INIT is :screen, so all three are read.
    # They were :inert only because the default was :at_seed.
    SettingDecl(:MCMC_INIT,         "demc.jl:run_demc", :live,
                ":screen (shipped) or :at_seed; :screen draws a cloud and keeps the feasible points"),
    SettingDecl(:MCMC_INIT_SCREEN,  "demc.jl:run_demc", :live,
                "candidates drawn for the :screen start; 0 ⇒ 20·N"),
    SettingDecl(:MCMC_SCREEN_FRAC,  "MCMC_main.jl:_screen_scale", :live,
                "starting radius as a fraction of the curvature width; 48/48 candidates were feasible at 0.3"),
    SettingDecl(:MCMC_SCREEN_CAP,   "MCMC_main.jl:_screen_scale", :live,
                "radius cap in width units; required because b_S's width is enormous in t"),
    SettingDecl(:MCMC_SCREEN_FLOOR, "MCMC_main.jl:_screen_scale", :live,
                "guards a zero or absent curvature width"),
    # Note built with `*` rather than a backslash continuation: check_structure.jl's string
    # tracker resets at each newline, so a multi-line literal inside a call leaves its
    # closing paren uncounted and the whole file then reads as unbalanced.
    SettingDecl(:CORNER_PARAMS,     "MCMC_main.jl:top level", :live,
                "parameters DECLARED to sit at a corner, chosen by hand as SKIP_MOMENTS is; " *
                "a label only — it fills the results CSV's corner_declared column and changes " *
                "no other value, so the shipped empty default leaves the file byte-identical"),
]

"""
Keys that were renamed, mapped to the key that replaced them.

A rename is the one lifecycle step that cannot be made safe by a default: the old key
still parses, still sets nothing, and the run proceeds under a configuration the
operator believes they overrode. `assert_no_legacy_env` turns that silence into an
error, which is the whole reason this table exists rather than a line in a changelog.

Entries are removed once no shell script, note or launcher in circulation could still
carry the old spelling — not on the next release.
"""
const LEGACY_ENV_KEYS = Dict(
    "ROYSEARCH_JAC_ONLY"      => "ROYSEARCH_MCMC_JAC_ONLY",
    "ROYSEARCH_SCREEN_FRAC"   => "ROYSEARCH_MCMC_SCREEN_FRAC",
    "ROYSEARCH_SCREEN_CAP"    => "ROYSEARCH_MCMC_SCREEN_CAP",
    "ROYSEARCH_SCREEN_FLOOR"  => "ROYSEARCH_MCMC_SCREEN_FLOOR",
)

# ============================================================
# Reading a setting
# ============================================================

# What each `env_setting` call actually resolved to, in call order. The authoritative
# settings print is generated from this log rather than from the caller's own
# variables: a print sourced from what the caller MEANT to read cannot detect a key
# that was misspelled or never consulted, and one sourced from the resolution itself
# cannot miss it.
const _ENV_LOG = Vector{NamedTuple{(:key, :value, :from_env),Tuple{String,Any,Bool}}}()

_env_parse(::Type{Symbol},  s::AbstractString) = Symbol(s)
# A set-valued setting, comma-separated: ROYSEARCH_CORNER_PARAMS="b_S,alpha_U". Entries are
# trimmed and empties dropped, so "" and "a, ,b" both behave as a reader would expect.
_env_parse(::Type{Vector{Symbol}}, s::AbstractString) =
    Symbol[Symbol(t) for t in strip.(split(s, ',')) if !isempty(t)]
_env_parse(::Type{Bool},    s::AbstractString) = parse(Bool, s)
_env_parse(::Type{Float64}, s::AbstractString) = parse(Float64, s)
_env_parse(::Type{Int},     s::AbstractString) = parse(Int, s)

"""
    env_setting(key, default) → value

Read `ROYSEARCH_<key>` from the environment, falling back to `default` and taking the
return type from it. `key` is given WITHOUT the `ROYSEARCH_` prefix, so the spelling in
the source matches the registry row and a grep for either finds both.

Every resolution is recorded for `print_env_settings`, so the log reports the value the
run is using rather than the value its caller intended to set.
"""
function env_setting(key::Symbol, default::T) where {T}
    full = "ROYSEARCH_$(key)"
    val, from_env = if haskey(ENV, full)
        raw = ENV[full]
        try
            _env_parse(T, raw), true
        catch
            error("ROYSEARCH_$(key) = \"$raw\" is not a valid $T.")
        end
    else
        default, false
    end
    push!(_ENV_LOG, (key = full, value = val, from_env = from_env))
    return val
end

"""
    assert_no_legacy_env()

Fail on any renamed key still set in the environment. Silently ignoring the old
spelling is exactly the defect class this module exists to remove: the operator would
get a run configured by the defaults while believing they had overridden them.
"""
function assert_no_legacy_env()
    stale = [(old, new) for (old, new) in LEGACY_ENV_KEYS if haskey(ENV, old)]
    isempty(stale) && return nothing
    msg = "Renamed settings are set in the environment and would be ignored:\n"
    for (old, new) in sort(stale)
        msg *= "    $old  →  use $new (currently \"$(ENV[old])\")\n"
    end
    error(msg * "  Update the launcher, then re-run. See SETTINGS.md.")
end

"""
    print_env_settings()

Report the environment overrides in force, from the resolution log rather than from
any caller's copy. Defaults are counted, not listed: the shipped defaults are readable
in the source, whereas an override is the part of a run that cannot be recovered from
the repository afterwards.
"""
function print_env_settings()
    overrides = [e for e in _ENV_LOG if e.from_env]
    @printf("  [env]  %d of %d settings overridden from the environment\n",
            length(overrides), length(_ENV_LOG))
    for e in overrides
        @printf("  [env]    %-34s = %s\n", e.key, e.value)
    end
    flush(stdout)
    return nothing
end

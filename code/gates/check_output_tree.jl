#!/usr/bin/env julia
############################################################
# check_output_tree.jl — enforce the v19.6.0 layout rules rather than remember them.
#
# Nine rules, each a filesystem check or a grep. They exist because the alternative is a
# convention, and a convention that nothing checks drifts: before v19.6.0 output paths were
# declared in seven files, two behind !@isdefined guards, and output/tables/ held nine CSVs
# alongside five .tex files so the directory could not be deleted and regenerated.
#
# A finding is a defect, not a warning. Exit 1 on any.
#
#   julia --project=. code/gates/check_output_tree.jl
############################################################

using Printf

include(joinpath(@__DIR__, "..", "repo_root.jl"))
const ROOT = find_repo_root()

findings = String[]
note(s) = push!(findings, s)

_files(dir) = isdir(dir) ? [f for f in readdir(dir; join = true) if isfile(f)] : String[]
function _walk(dir)
    out = String[]
    isdir(dir) || return out
    for (root, _, fs) in walkdir(dir), f in fs
        push!(out, joinpath(root, f))
    end
    return out
end
_rel(p) = replace(p, ROOT => "")

const IMAGE_EXT = (".png", ".pdf", ".svg", ".jpg", ".jpeg", ".eps")
const OUT       = joinpath(ROOT, "output")
const CODE      = joinpath(ROOT, "code")

println("check_output_tree — v19.6.0 layout\n")

# ── 1. tables/ holds .tex only ─────────────────────────────────────────────
for f in _walk(joinpath(OUT, "tables"))
    endswith(f, ".tex") || occursin(".DS_Store", f) ||
        note("rule 1: non-.tex in tables/ — $(_rel(f))")
end

# ── 2. plots/ holds images only ────────────────────────────────────────────
for f in _walk(joinpath(OUT, "plots"))
    any(endswith(f, e) for e in IMAGE_EXT) || occursin(".DS_Store", f) ||
        note("rule 2: non-image in plots/ — $(_rel(f))")
end

# ── 3. no artifacts under code/ ────────────────────────────────────────────
# The gitignored trees are excluded: they are not part of the shipped codebase.
const CODE_SKIP = ("notebooks", "single_run_plots_all", ".vscode", ".ipynb_checkpoints",
                   "roysearch_v")
for f in _walk(CODE)
    any(occursin(s, f) for s in CODE_SKIP) && continue
    any(endswith(f, e) for e in (IMAGE_EXT..., ".tex", ".csv", ".jls", ".json")) &&
        note("rule 3: artifact under code/ — $(_rel(f))")
end

# ── 4. only plots_and_tables/ writes exhibits ──────────────────────────────
# The one rule that makes "one writer" true rather than aspirational.
for f in _walk(CODE)
    endswith(f, ".jl") || continue
    any(occursin(s, f) for s in CODE_SKIP) && continue
    occursin(joinpath("code", "plots_and_tables"), f) && continue
    src = read(f, String)
    # The WRITE is savefig. out_plots/out_tables are path LOOKUPS, and an entry point
    # resolving a directory to hand to a plotting script is correct — flagging those made
    # the rule fire on model_main.jl doing exactly the right thing.
    occursin(joinpath("code", "gates"), f) && continue    # a gate names the pattern it greps for
    occursin("savefig(", src) &&
        note("rule 4: savefig( outside plots_and_tables/ — $(_rel(f))")
end

# ── 5. exact window / pair subdirectories per category ─────────────────────
const WINDOWS_E = ("base_fc", "crisis_fc", "base_covid", "crisis_covid")
const PAIRS_E   = ("fc", "covid")
for (kind, root) in (("plots", joinpath(OUT, "plots")), ("tables", joinpath(OUT, "tables")))
    isdir(root) || continue
    for (cat, admitted) in (("fit", WINDOWS_E), ("model", WINDOWS_E),
                            ("policy", WINDOWS_E), ("transition", PAIRS_E))
        d = joinpath(root, cat)
        isdir(d) || continue
        for sub in readdir(d)
            isdir(joinpath(d, sub)) || continue
            sub in admitted ||
                note("rule 5: unexpected subdir $(kind)/$(cat)/$(sub) — not in $(admitted)")
        end
    end
    for cat in ("descriptives", "manual")
        d = joinpath(root, cat)
        isdir(d) || continue
        for sub in readdir(d)
            isdir(joinpath(d, sub)) &&
                note("rule 5: $(kind)/$(cat)/ takes no subdirectory, found $(sub)")
        end
    end
end

# ── 6. estimates/ matches only the permitted patterns ──────────────────────
const EST_OK = r"^(estimate|smm_estimates|mcmc_results|candidates|plateau_probe|tau_margin)_.*\.(jls|csv)$|^estimate_.*_postmean\.jls$"
for f in _files(joinpath(OUT, "estimates"))
    b = basename(f)
    b == ".DS_Store" && continue
    occursin(EST_OK, b) || note("rule 6: unexpected file in estimates/ — $(b)")
    for bad in (".pre_migration", ".pre_v", ".new", "_backup_Q")
        occursin(bad, b) && note("rule 6: migration/backup debris in estimates/ — $(b)")
    end
end

# ── 7. no path literal outside paths.jl ────────────────────────────────────
for f in _walk(CODE)
    endswith(f, ".jl") || continue
    any(occursin(s, f) for s in CODE_SKIP) && continue
    occursin(joinpath("code", "paths.jl"), f) && continue
    occursin(joinpath("code", "gates"), f)   && continue     # gates walk the tree by design
    src = read(f, String)
    for pat in ("\"output\", \"smm\"", "joinpath(OUTPUT_DIR, \"smm\")",
                "\"output\", \"tables\"", "\"output\", \"plots\"")
        occursin(pat, src) &&
            note("rule 7: hardcoded output path $(pat) — $(_rel(f))")
    end
end

# ── 8. smm/ never includes from mcmc/ ──────────────────────────────────────
# The MCMC samples the SMM criterion, so it is downstream and the layout should say so.
for f in _files(joinpath(CODE, "smm"))
    endswith(f, ".jl") || continue
    occursin("mcmc", read(f, String)) && occursin("include(", read(f, String)) || continue
    for l in eachline(f)
        occursin("include(", l) && occursin("mcmc", lowercase(l)) &&
            note("rule 8: code/smm includes from mcmc/ — $(_rel(f)): $(strip(l))")
    end
end

# ── 9. every bundle serialize goes through write_bundle ────────────────────
for f in _walk(CODE)
    endswith(f, ".jl") || continue
    any(occursin(s, f) for s in CODE_SKIP) && continue
    endswith(f, "bundle.jl") && continue                     # the writer itself
    endswith(f, "candidates.jl") && continue                 # seed bank, not a bundle
    # Only smm/ and mcmc/ write estimate bundles. transition_simulation serialises a
    # transition result — no spec field, different object, different home — and a gate that
    # flagged it would be asserting a rule that does not apply to it.
    (occursin(joinpath("code", "smm"), f) || occursin(joinpath("code", "mcmc"), f) ||
     occursin(joinpath("code", "tools"), f)) || continue
    for (i, l) in enumerate(eachline(f))
        occursin("serialize(", l) && !occursin("deserialize", l) || continue
        # The chain-draw bundle is a different object with its own shape and its own home.
        occursin("chain", lowercase(l)) && continue
        occursin("res.chain", l) && continue
        # The neutral migration intermediate is raw ON PURPOSE: a struct migration cannot
        # route through write_bundle, because the struct being migrated is the one
        # write_bundle would fail to construct. It stages under /.migration/ and is
        # removed on success.
        occursin("neutral", lowercase(l)) && continue
        note("rule 9: bare serialize of a bundle — $(_rel(f)):$(i)")
    end
end

println(isempty(findings) ? "PASS — all nine layout rules hold." :
        "FINDINGS ($(length(findings))):")
for s in findings
    println("  ", s)
end
exit(isempty(findings) ? 0 : 1)

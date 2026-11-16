#!/usr/bin/env bash
# gates/check_repo.sh — can a clone run the estimation, and what does it contain?
#
# Checks the required inputs and warm-start bundles, runs the settings, threading and
# hiring-margin gates, then (unless --verify-only) audits the size of the tracked set.
# Run before pushing, and on the VM after cloning with --verify-only.
#
#   bash code/gates/check_repo.sh
#   bash code/gates/check_repo.sh --verify-only
set -euo pipefail

# The root is the nearest ancestor holding both markers (as code/repo_root.jl).
find_repo_root() {
    dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    while [ "$dir" != "/" ]; do
        if [ -f "$dir/code/smm/smm_main.jl" ] && [ -f "$dir/data/derived/windows.json" ]; then
            echo "$dir"; return 0
        fi
        dir="$(dirname "$dir")"
    done
    echo "could not find repository root above ${BASH_SOURCE[0]}" >&2
    return 1
}
REPO_ROOT="$(find_repo_root)"
cd "$REPO_ROOT"
VERIFY_ONLY=false
[ "${1:-}" = "--verify-only" ] && VERIFY_ONLY=true

# Files the two estimation entry points read: one moment and one sampling-variance file per
# data window (DATA_WINDOWS, paths.jl), and one estimate per estimation window (WINDOWS).
DATA_WINDOWS=(base_fc base_covid crisis_fc crisis_covid)
WINDOWS=(baseline crisis_fc crisis_covid)
GLOBAL_INPUTS=(data/derived/windows.json data/derived/nu_estimation.csv
               data/derived/phi_calibration.csv
               Project.toml Manifest.toml)

echo "=== required inputs ==="
missing=0
for f in "${GLOBAL_INPUTS[@]}"; do
    if [ -f "$f" ]; then printf "  ok      %s\n" "$f"
    else printf "  MISSING %s\n" "$f"; missing=1; fi
done
for w in "${DATA_WINDOWS[@]}"; do
    for stem in moments sampling_var; do
        f="data/derived/${stem}_${w}.csv"
        [ -f "$f" ] || { printf "  MISSING %s\n" "$f"; missing=1; }
    done
done
[ "$missing" -eq 0 ] && echo "  all per-window moment and sampling-variance files present"

# A :warmstart run scores every stored point that can start a window (smm_main.jl §8a);
# with none it starts from the DEFAULT values.
echo ""
echo "=== warm-start candidates (optional, but a :warmstart run wants one) ==="
for w in "${WINDOWS[@]}"; do
    found=""
    for tag in _smm _smm_mcmcbase; do
        f="output/estimates/estimate_${w}_diagonalW${tag}.jls"
        [ -f "$f" ] && found="$found $(basename "$f")"
    done
    printf "  %-13s %s\n" "$w" "${found:- none — will start from the DEFAULT values}"
done

# Settings lifecycle, also under --verify-only: an undelivered setting still produces
# numbers and reports the intended configuration, so no file listing reveals it.
echo ""
echo "=== settings lifecycle (code/gates/check_forwarding.jl) ==="
JULIA_BIN="${JULIA:-julia}"
if command -v "$JULIA_BIN" >/dev/null 2>&1; then
    if "$JULIA_BIN" --project=. --startup-file=no code/gates/check_forwarding.jl; then
        :
    else
        echo "  check_forwarding reported findings — see SETTINGS.md" >&2
        missing=1
    fi
else
    echo "  SKIPPED: julia not on PATH (set JULIA=/path/to/julia to include this gate)"
fi

# Threaded evaluation, at -t 2: the defect it catches is invisible on one thread.
echo ""
echo "=== threaded evaluation (code/gates/check_threading.jl) ==="
if command -v "$JULIA_BIN" >/dev/null 2>&1; then
    if "$JULIA_BIN" --project=. --startup-file=no -t 2 code/gates/check_threading.jl; then
        :
    else
        echo "  check_threading reported findings — a threaded solve is not reproducing" >&2
        missing=1
    fi
else
    echo "  SKIPPED: julia not on PATH (set JULIA=/path/to/julia to include this gate)"
fi

# Hiring margin: the solved p*_U must select the closed-form hireable set a_U > b̃_U/P_U.
echo ""
echo "=== unskilled hiring margin (code/gates/check_hire_margin.jl) ==="
if command -v "$JULIA_BIN" >/dev/null 2>&1; then
    if "$JULIA_BIN" --project=. --startup-file=no code/gates/check_hire_margin.jl; then
        :
    else
        echo "  check_hire_margin reported findings — p*_U and the closed form disagree" >&2
        missing=1
    fi
else
    echo "  SKIPPED: julia not on PATH (set JULIA=/path/to/julia to include this gate)"
fi

$VERIFY_ONLY && { echo ""; echo "verify-only: skipping the size audit"; exit "$missing"; }

echo ""
echo "=== what a clone would contain ==="
git ls-files -z | xargs -0 -I{} stat -f%z {} 2>/dev/null | \
  awk '{s+=$1; n++} END {printf "  %d tracked files, %.1f MB\n", n, s/1048576}'

echo ""
echo "=== 15 largest tracked files ==="
git ls-files -z | while IFS= read -r -d '' f; do
    [ -f "$f" ] && printf "%s\t%s\n" "$(stat -f%z "$f")" "$f"
done | sort -rn | head -15 | awk -F'\t' '{printf "  %8.2f MB  %s\n", $1/1048576, $2}'

# The pre-commit hook only sees newly staged files over its size limit; this reports
# tracked files that .gitignore now excludes.
echo ""
echo "=== tracked files that .gitignore now excludes (staged bloat the hook missed) ==="
# --no-index is required: without it check-ignore skips tracked paths and reports none.
stale=$(git ls-files | git check-ignore --stdin --no-index 2>/dev/null | head -20 || true)
if [ -z "$stale" ]; then
    echo "  none — tracked set agrees with .gitignore"
else
    echo "$stale" | sed 's/^/  /'
    n=$(git ls-files | git check-ignore --stdin --no-index 2>/dev/null | wc -l | tr -d ' ')
    echo ""
    echo "  $n tracked file(s) match .gitignore. .gitignore does NOT untrack them."
    echo "  Remove from the index, keeping the files on disk:"
    echo "    git ls-files | git check-ignore --stdin --no-index | xargs -I{} git rm --cached -q {}"
    echo "    git commit -m 'stop tracking generated and reference files'"
fi

exit "$missing"

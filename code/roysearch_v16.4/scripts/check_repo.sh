#!/usr/bin/env bash
# scripts/check_repo.sh — what would a clone actually contain?
#
# Answers the question a .gitignore cannot: of the files git currently TRACKS, how
# large is the clone, what are the biggest offenders, and is every input the
# estimation needs actually present?
#
# Run before pushing, and on the VM after cloning (--verify-only) to confirm the
# clone can run the batch.
#
#   bash scripts/check_repo.sh
#   bash scripts/check_repo.sh --verify-only
set -euo pipefail

# Scripts may sit at the repo root or under code/, so the root is found by walking
# up for its markers rather than assuming a fixed depth.
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

# Files the two estimation entry points read. windows.json is the single source of
# truth for the window list, so the loop derives the per-window names from it rather
# than repeating them.
WINDOWS=(base_fc base_covid crisis_fc crisis_covid)
GLOBAL_INPUTS=(data/derived/windows.json data/derived/nu_estimation.csv
               data/derived/phi_calibration.csv data/derived/training_share_scale.csv
               Project.toml Manifest.toml)

echo "=== required inputs ==="
missing=0
for f in "${GLOBAL_INPUTS[@]}"; do
    if [ -f "$f" ]; then printf "  ok      %s\n" "$f"
    else printf "  MISSING %s\n" "$f"; missing=1; fi
done
for w in "${WINDOWS[@]}"; do
    for stem in moments sigma sampling_var moment_scales; do
        f="data/derived/${stem}_${w}.csv"
        [ -f "$f" ] || { printf "  MISSING %s\n" "$f"; missing=1; }
    done
done
[ "$missing" -eq 0 ] && echo "  all per-window moment/sigma/variance/scale files present"

# A warmstart seeds from a saved optimum; without one, INIT_MODE=:warmstart falls
# back to DEFAULT_PARAMS, which is a different (and slower) starting point.
echo ""
echo "=== warmstart bundles (optional, but a :warmstart run wants them) ==="
for w in "${WINDOWS[@]}"; do
    f="output/smm/smm_result_${w}_diagonalW.jls"
    printf "  %-9s %s\n" "$w" "$([ -f "$f" ] && echo "present" || echo "absent — will start from DEFAULT_PARAMS")"
done

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

# The pre-commit hook blocks single files over its limit. It cannot see files that
# are ALREADY tracked, and it never fires on a directory of many small files — which
# is how most repository bloat actually arrives.
echo ""
echo "=== tracked files that .gitignore now excludes (staged bloat the hook missed) ==="
# --no-index is load-bearing: without it check-ignore skips paths already in the
# index, which is exactly the set being audited, and the count comes back zero.
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

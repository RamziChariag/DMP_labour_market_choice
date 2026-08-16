#!/usr/bin/env bash
# Runs ON THE VM inside tmux. Run, push, shut down.
set -u
WINDOW="${ROYSEARCH_WINDOW:-base_covid}"
BRANCH="vm-${WINDOW//_/-}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p output/logs
LOG="output/logs/${WINDOW}_vm.log"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-$(nproc)}"
export OPENBLAS_NUM_THREADS=1

git checkout -q -B "$BRANCH"
julia --project=. code/smm/smm_main.jl 2>&1 | tee "$LOG"

git add -A output/smm output/logs
git add -f "output/tables/smm_estimates_${WINDOW}_diagonalW.csv" 2>/dev/null
git -c user.name="RoySearch VM" -c user.email="vm@roysearch.local" \
    commit -q -m "${WINDOW} estimate"
git push -q -u origin "$BRANCH" && echo "PUSHED to $BRANCH"
sudo shutdown -h now

#!/usr/bin/env bash
# Runs ON THE VM inside tmux. Run, push, shut down — including on a crash.
#
# RUN SETTINGS LIVE HERE. Edit the exports below; nothing is passed by hand.
set -u

WINDOW="${ROYSEARCH_WINDOW:-base_covid}"
BRANCH="vm-${WINDOW//_/-}"

# ── run settings ────────────────────────────────────────────────────────────
export ROYSEARCH_INIT_MODE=default        # base_covid's old bundle is on a stale
                                          # spec (b_T was free); warmstart gives
                                          # start Q=Inf. default seeds from the
                                          # base_fc optimum on the P_U=1 ray.
export ROYSEARCH_DE_ADAPT_FCR=true        # f and cr from the generator's yield
export ROYSEARCH_DE_GEN_PER_K=60          # 60 draws per sparsity: 3x headroom over
                                          # 240 slots, which is what keeps the
                                          # allocation weight-driven not supply-driven
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-$(nproc)}"
export OPENBLAS_NUM_THREADS=1             # or BLAS fights the solver's own threads
# ────────────────────────────────────────────────────────────────────────────

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p output/logs
LOG="output/logs/${WINDOW}_vm.log"

# Fires however the script ends: clean exit, julia crash, or kill.
finish () {
    git add -A output/smm output/logs 2>/dev/null
    git add -f "output/tables/smm_estimates_${WINDOW}_diagonalW.csv" 2>/dev/null
    git -c user.name="RoySearch VM" -c user.email="vm@roysearch.local" \
        commit -q -m "${WINDOW} estimate" 2>/dev/null
    git push -q -u origin "$BRANCH" && echo "PUSHED to $BRANCH"
    sudo shutdown -h now
}
trap finish EXIT

git checkout -q -B "$BRANCH"
julia --project=. code/smm/smm_main.jl 2>&1 | tee "$LOG"

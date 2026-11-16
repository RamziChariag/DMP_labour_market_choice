#!/usr/bin/env bash
# ops/setup_vm.sh — provision a fresh Linux VM to run the estimation.
#
# Installs the pinned Julia, instantiates Project.toml/Manifest.toml (written by
# code/ops/export_env.jl), precompiles, and checks that the estimation packages load.
# Idempotent.
#
#   bash code/ops/setup_vm.sh                 # uses the Julia version below
#   JULIA_VERSION=1.10.10 bash code/ops/setup_vm.sh
set -euo pipefail

# Must match the Manifest's julia_version: a Manifest instantiates on the minor that wrote it.
JULIA_VERSION="${JULIA_VERSION:-1.10.10}"
JULIA_SERIES="${JULIA_VERSION%.*}"
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

case "$(uname -m)" in
  x86_64)          ARCH=x64;   TARBALL_ARCH=x86_64 ;;
  aarch64|arm64)   ARCH=aarch64; TARBALL_ARCH=aarch64 ;;
  *) echo "unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

echo "=== RoySearch VM setup ==="
echo "  repo:   $REPO_ROOT"
echo "  julia:  $JULIA_VERSION ($ARCH)"
echo "  cores:  $(nproc)"
echo "  memory: $(awk '/MemTotal/ {printf "%.1f GB", $2/1048576}' /proc/meminfo)"

if ! command -v julia >/dev/null 2>&1 || \
   [ "$(julia --version | awk '{print $3}')" != "$JULIA_VERSION" ]; then
  echo "--- installing Julia $JULIA_VERSION ---"
  URL="https://julialang-s3.julialang.org/bin/linux/${ARCH}/${JULIA_SERIES}/julia-${JULIA_VERSION}-linux-${TARBALL_ARCH}.tar.gz"
  curl -fsSL "$URL" -o /tmp/julia.tar.gz
  sudo mkdir -p /opt/julia
  sudo tar -xzf /tmp/julia.tar.gz -C /opt/julia --strip-components=1
  sudo ln -sf /opt/julia/bin/julia /usr/local/bin/julia
  rm -f /tmp/julia.tar.gz
fi
julia --version

echo "--- restoring the pinned environment ---"
cd "$REPO_ROOT"
julia --project=. -e '
  using Pkg
  Pkg.instantiate()          # exact versions from Manifest.toml
  Pkg.precompile()           # pay compilation once, not per batch process
  Pkg.status()
'

echo "--- smoke test: can the solver stack load? ---"
julia --project=. --threads=2 -e '
  using LinearAlgebra, SparseArrays, Statistics, Random, Distributions
  using FastGaussQuadrature, Interpolations, Parameters, Printf
  using Optim, CSV, DataFrames, Serialization, Clustering, QuasiMonteCarlo, JSON3
  println("all estimation packages load; threads = ", Threads.nthreads())
'

cat <<EOF

=== ready ===
Run the estimation under tmux so it survives a dropped SSH session. smm_main.jl is
the entry point; every setting is a ROYSEARCH_* environment variable, so the run is
configured before the command rather than by editing source.

  tmux new -s roysearch
  cd $REPO_ROOT
  export ROYSEARCH_WINDOW=base_covid
  export ROYSEARCH_INIT_MODE=warmstart
  export ROYSEARCH_DE_ADAPT_FCR=true
  export ROYSEARCH_DE_GEN_PER_K=60
  export JULIA_NUM_THREADS=\$(nproc)
  export OPENBLAS_NUM_THREADS=1
  julia --project=. code/smm/smm_main.jl 2>&1 | tee output/logs/base_covid.log

Detach with Ctrl-b d (press Ctrl-b, release, then d). Reattach: tmux attach -t roysearch
Detaching does NOT stop the run. Closing the SSH window does NOT stop it either.
EOF

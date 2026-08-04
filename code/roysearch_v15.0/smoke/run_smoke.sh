#!/bin/bash
# Crisis-window spec smoke tests (see README.md).
#
#   ./run_smoke.sh warmstart|clusters|guard
#
# Run from anywhere; paths resolve relative to this script.  Requires
# data/derived/ and the paired baseline bundle in output/smm/.
set -euo pipefail
cd "$(dirname "$0")"

# Test name, and the INIT_MODE the sliced driver runs under.  The guard test
# exercises spec.fixed only, so it reuses the :warmstart configuration.
TEST="${1:-}"
case "$TEST" in
    warmstart|guard) MODE=warmstart ;;
    clusters)        MODE=clusters  ;;
    *) echo "usage: run_smoke.sh warmstart|clusters|guard" >&2; exit 1 ;;
esac

JULIA="${JULIA:-julia}"
SMM_DIR="../smm"
SLICED="$SMM_DIR/main_spec_only.jl"

# Truncate the driver just before §9 so the spec-construction path runs verbatim.
"$JULIA" slice_spec.jl "$SMM_DIR/smm_main.jl" "# 9. Run estimation" "$SLICED"

# Only the RUN CONFIGURATION differs from the shipped driver: the coarse grid
# (these tests build a spec, they do not estimate) and, for :clusters, a small
# Sobol sample.  n_sample must stay a power of two — Owen scrambling requires it.
python3 - "$MODE" "$SLICED" <<'PY'
import re, sys
mode, path = sys.argv[1], sys.argv[2]
s = open(path).read()
s = s.replace("INIT_MODE            = :warmstart", f"INIT_MODE            = :{mode}")
for name in ("Nx      ", "Np_U    ", "Np_S    "):
    s = re.sub(rf"^    {name}= 120,", f"    {name}= 40,", s, flags=re.M)
if mode == "clusters":
    s = re.sub(r"^    cand_n_sample    = 2048,", "    cand_n_sample    = 32,", s, flags=re.M)
    s = re.sub(r"^    cand_min_cluster = 5,",    "    cand_min_cluster = 2,", s, flags=re.M)
open(path, "w").write(s)
PY

# The test runs from smm/ so its @__DIR__ include of the sliced driver resolves.
cp "smoke_${TEST}.jl" "$SMM_DIR/run_smoke_${TEST}.jl"
"$JULIA" --project="$SMM_DIR/../.." --threads auto "$SMM_DIR/run_smoke_${TEST}.jl"

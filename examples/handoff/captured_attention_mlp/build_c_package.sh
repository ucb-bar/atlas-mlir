#!/usr/bin/env bash
# Compile the public two-output graph and export C/H without platform startup.
set -euo pipefail
if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 MERLIN_ROOT PYTHON ATLAS_BIN FRESH_OUT [mandatory|optimized] [COMPILE_INPUTS]" >&2
  exit 2
fi
root=$(cd "$(dirname "$0")/../../.." && pwd)
merlin_root=$(cd "$1" && pwd)
# Keep the virtual-environment executable path; dereferencing its symlink would
# select the base interpreter and lose the installed compiler dependencies.
python=$(command -v "$2")
atlas_bin=$(cd "$3" && pwd)
out=$(realpath -m "$4")
pipeline=${5:-optimized}
if [[ "$pipeline" != mandatory && "$pipeline" != optimized ]]; then
  echo "pipeline must be mandatory or optimized" >&2
  exit 2
fi
if [[ -e "$out" || -L "$out" ]]; then
  echo "output directory must be fresh: $out" >&2
  exit 2
fi
for tool in atlas-opt atlas-emit; do
  test -x "$atlas_bin/$tool"
done
export PYTHONPATH="$merlin_root/src:$root/python${PYTHONPATH:+:$PYTHONPATH}"
export ATLAS_OOT_BIN_DIR="$atlas_bin"
export ATLAS_NATIVE_PIPELINE="$pipeline"
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}
mkdir -p "$out"
source_ir="$root/test/examples/captured_attention_mlp32_two_outputs_linalg.mlir"
policy=${ATLAS_GRAPH_PRECISION_POLICY:-$root/test/examples/attention_mlp_graph_finite_zero_arithmetic.json}
abi="$root/test/examples/attention_mlp_graph_abi.json"
parameter_args=()
if [[ $# -eq 6 ]]; then
  compile_inputs=$(cd "$6" && pwd)
  source_ir="$compile_inputs/program.mlir"
  abi=auto
  parameter_args=(--compile-inputs "$compile_inputs")
fi
abi=${ATLAS_GRAPH_ABI:-$abi}
"$python" -m merlin.targetgen.cli native-build --engine merlin_native --support atlas \
  --cargo-target-dir "${CARGO_TARGET_DIR:-$out/cargo}" \
  --source-revision "$(git -C "$merlin_root" rev-parse HEAD)" --out "$out/snapshot"
"$python" -m merlin.targetgen.cli native-compile --engine merlin_native --support atlas \
  --snapshot "$out/snapshot" --linalg "$source_ir" --linalg-entry forward \
  --diagnostic-graph-policy "$policy" \
  --abi "$abi" \
  --search-limits "$root/test/examples/attention_mlp_graph_search_limits.json" \
  --target-source "$root" --mode diagnostic --out "$out/compiled" --status-file "$out/compile-status.json"
"$python" - "$out/compile-status.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as source:
    status = json.load(source)
if status.get("status") != "emitted" or status.get("artifact_kind") != "program":
    raise SystemExit("this example requires one device program; segmented output is not accepted")
PY
"$python" -m merlin.targetgen.cli native-verify --engine merlin_native --support atlas \
  --snapshot "$out/snapshot" --linalg "$source_ir" --linalg-entry forward \
  --diagnostic-graph-policy "$policy" --mode diagnostic \
  --artifact "$out/compiled" --out "$out/verification.json"
"$python" -m merlin.targetgen.cli native-bridge-graph --linalg "$source_ir" \
  --entry forward --compiled "$out/compiled" --device-symbol merlin_native_graph --out "$out/host-bridge"
"$python" "$root/tools/atlas_graph_model_bridge.py" --linalg "$source_ir" \
  --host-bridge "$out/host-bridge" --compiled "$out/compiled" \
  --c-package-name attention_mlp "${parameter_args[@]}" --out "$out/c-package"

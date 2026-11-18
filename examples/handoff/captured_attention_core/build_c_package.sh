#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 5 ]]; then
  echo "usage: $0 MERLIN_ROOT PYTHON ATLAS_BIN CAPTURE OUT" >&2
  exit 2
fi
merlin_root=$(realpath "$1")
python=$(command -v "$2")
atlas_bin=$(realpath "$3")
capture=$(realpath "$4")
output=$(realpath -m "$5")
application_name=${MERLIN_APPLICATION_NAME:-attention}
atlas_root=$(cd "$(dirname "$0")/../../.." && pwd)
if [[ -e "$output" || -L "$output" ]]; then
  echo "output must be fresh: $output" >&2
  exit 2
fi
mkdir -p "$output"
export PYTHONPATH="$merlin_root/src:$atlas_root/python${PYTHONPATH:+:$PYTHONPATH}"
export ATLAS_OOT_BIN_DIR="$atlas_bin"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
cli="$(dirname "$python")/merlin-targetgen"
cargo_args=()
state_args=()
if [[ -n "${MERLIN_APPLICATION_STATE:-}" ]]; then
  state_args=(--application-state "$MERLIN_APPLICATION_STATE")
fi
if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
  cargo_args=(--cargo-target-dir "$CARGO_TARGET_DIR")
fi
"$python" - "$capture/model.mlir" "$output/precision-policy.json" "$output/search-limits.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from atlas_mlir_native.application_precision import CONTRACTION_POLICY, SCALED_CONTRACTION_POLICY
from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.row_softmax import SOFTMAX_POLICY
from merlin.semantic_compiler.search import SearchLimits

source, policy, limits = map(Path, sys.argv[1:])
selected = {
    "schema": "atlas.diagnostic_application_precision.v1",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "target_identity": TARGET_IDENTITY, "contraction": CONTRACTION_POLICY,
    "row_softmax": SOFTMAX_POLICY, "host_support": ["transpose", "reshape", "copy"],
}
if "ATLAS_OPERAND_SCALE_EXPONENTS" in os.environ:
    exponents = json.loads(os.environ["ATLAS_OPERAND_SCALE_EXPONENTS"])
    if not isinstance(exponents, list) or len(exponents) != 2 or any(
        type(value) is not int or not -32 <= value <= 32 for value in exponents
    ):
        raise ValueError("ATLAS_OPERAND_SCALE_EXPONENTS must be two integer exponents in [-32,32]")
    selected.update(schema="atlas.diagnostic_application_precision.v2", contraction={
        "policy": SCALED_CONTRACTION_POLICY, "operand_scale_exponents": exponents,
    })
if "ATLAS_CONTRACTION_EPILOGUES" in os.environ:
    epilogues = json.loads(os.environ["ATLAS_CONTRACTION_EPILOGUES"])
    if epilogues != ["relu"] or selected["schema"] != "atlas.diagnostic_application_precision.v2":
        raise ValueError("ATLAS_CONTRACTION_EPILOGUES currently requires ['relu'] and explicit operand scales")
    selected.update(schema="atlas.diagnostic_application_precision.v3", epilogues=epilogues)
if "ATLAS_POINTWISE_POLICY" in os.environ:
    from atlas_mlir_native.pointwise import POINTWISE_POLICY

    if os.environ["ATLAS_POINTWISE_POLICY"] != POINTWISE_POLICY or selected["schema"] != "atlas.diagnostic_application_precision.v3":
        raise ValueError("ATLAS_POINTWISE_POLICY requires the named pointwise policy, operand scales and epilogues")
    selected.update(schema="atlas.diagnostic_application_precision.v4", pointwise=POINTWISE_POLICY)
if "ATLAS_HOST_LAYOUT_OPERATIONS" in os.environ:
    from atlas_mlir_native.application import checked_host_layouts

    if selected["schema"] != "atlas.diagnostic_application_precision.v4":
        raise ValueError("ATLAS_HOST_LAYOUT_OPERATIONS requires explicit v4 precision settings")
    operations = checked_host_layouts(json.loads(os.environ["ATLAS_HOST_LAYOUT_OPERATIONS"]))
    selected.update(schema="atlas.diagnostic_application_precision.v5", host_support=list(operations))
if "ATLAS_REDUCTION_POLICY" in os.environ:
    from atlas_mlir_native.reduction import REDUCTION_POLICY

    if os.environ["ATLAS_REDUCTION_POLICY"] != REDUCTION_POLICY or selected["schema"] != "atlas.diagnostic_application_precision.v5":
        raise ValueError("ATLAS_REDUCTION_POLICY requires the named reduction policy and explicit v5 settings")
    selected.update(schema="atlas.diagnostic_application_precision.v6", reduction=REDUCTION_POLICY)
if "ATLAS_POINTWISE_MATH_POLICY" in os.environ:
    from atlas_mlir_native.pointwise_math import MATH_POLICY

    if os.environ["ATLAS_POINTWISE_MATH_POLICY"] != MATH_POLICY or selected["schema"] != "atlas.diagnostic_application_precision.v6":
        raise ValueError("ATLAS_POINTWISE_MATH_POLICY requires the named math policy and explicit v6 settings")
    selected.update(schema="atlas.diagnostic_application_precision.v7", pointwise_math=MATH_POLICY)
if "ATLAS_HOST_EMBEDDING" in os.environ:
    if os.environ["ATLAS_HOST_EMBEDDING"] != "1" or selected["schema"] != "atlas.diagnostic_application_precision.v7":
        raise ValueError("ATLAS_HOST_EMBEDDING requires 1 and explicit v7 settings")
    selected.update(schema="atlas.diagnostic_application_precision.v8", host_support=[*selected["host_support"], "embedding_lookup"])
if "ATLAS_SIGMOID_POLICY" in os.environ:
    from atlas_mlir_native.sigmoid import SIGMOID_POLICY

    if os.environ["ATLAS_SIGMOID_POLICY"] != SIGMOID_POLICY or selected["schema"] != "atlas.diagnostic_application_precision.v8":
        raise ValueError("ATLAS_SIGMOID_POLICY requires the named sigmoid policy and explicit v8 settings")
    selected.update(schema="atlas.diagnostic_application_precision.v9", sigmoid=SIGMOID_POLICY)
if "ATLAS_MASKED_SOFTMAX_POLICY" in os.environ:
    from atlas_mlir_native.row_softmax import MASKED_SOFTMAX_POLICY

    if os.environ["ATLAS_MASKED_SOFTMAX_POLICY"] != MASKED_SOFTMAX_POLICY or selected["schema"] != "atlas.diagnostic_application_precision.v9":
        raise ValueError("ATLAS_MASKED_SOFTMAX_POLICY requires the named policy and explicit v9 settings")
    selected.update(schema="atlas.diagnostic_application_precision.v10", masked_softmax=MASKED_SOFTMAX_POLICY)
if "ATLAS_HOST_INDEXED_UPDATE" in os.environ:
    if os.environ["ATLAS_HOST_INDEXED_UPDATE"] != "1" or selected["schema"] != "atlas.diagnostic_application_precision.v10":
        raise ValueError("ATLAS_HOST_INDEXED_UPDATE requires 1 and explicit v10 settings")
    selected.update(schema="atlas.diagnostic_application_precision.v11", host_support=[*selected["host_support"], "indexed_insert_slice"])
policy.write_text(json.dumps(selected, indent=2) + "\n")
limits.write_text(json.dumps(SearchLimits(candidate_nodes=128).record(), indent=2) + "\n")
PY
"$cli" stage-capture --capture "$capture" --precision-policy "$output/precision-policy.json" \
  --out "$output/compile-inputs" --status-file "$output/staging.json"
"$cli" native-build --engine merlin_native --support atlas \
  --source-revision "$(git -C "$merlin_root" rev-parse HEAD)" "${cargo_args[@]}" --out "$output/snapshot"
"$cli" native-compile --engine merlin_native --support atlas --snapshot "$output/snapshot" \
  --compile-inputs "$output/compile-inputs" --application "${state_args[@]}" --application-name "$application_name" \
  --application-memory-bytes "${MERLIN_APPLICATION_MEMORY_BYTES:-262144}" --abi auto --mode diagnostic --target-source "$atlas_root" \
  --search-limits "$output/search-limits.json" --out "$output/compiled" --status-file "$output/compilation.json"
"$cli" native-verify --engine merlin_native --support atlas --snapshot "$output/snapshot" \
  --compile-inputs "$output/compile-inputs" --application "${state_args[@]}" --mode diagnostic --target-source "$atlas_root" \
  --artifact "$output/compiled" --out "$output/verification.json"

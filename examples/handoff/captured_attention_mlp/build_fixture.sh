#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )) || { (( $# == 3 )) && [[ $3 != --two-output ]]; }; then
  echo "usage: $0 ATLAS_BIN_DIR OUTPUT_DIR [--two-output]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
atlas_bin_dir=$1
output_dir=$2
python_bin=${PYTHON_BIN:-python3}
source_file=captured_attention_mlp32_linalg.mlir
if (( $# == 3 )); then
  source_file=captured_attention_mlp32_two_outputs_linalg.mlir
fi

"$python_bin" "$repo_root/tools/compile_atlas_linalg_tile.py" \
  --linalg "$repo_root/test/examples/$source_file" \
  --argument-manifest "$repo_root/test/examples/captured_attention_mlp32_arguments.json" \
  --weights "$repo_root/test/examples/captured_attention_mlp32_weights.safetensors" \
  --policy "$repo_root/test/examples/atlas_attention_mlp_numeric_policy.json" \
  --atlas-bin-dir "$atlas_bin_dir" \
  --output-dir "$output_dir"

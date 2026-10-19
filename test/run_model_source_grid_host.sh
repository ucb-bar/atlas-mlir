#!/usr/bin/env bash
# Execute one checked model lowering with a fake Atlas device on the host.
# The fake device checks the image and data path, not Atlas arithmetic.
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 <model.ll> <host-bundle> <model-wrapper> <model-link> <fresh-out> [clang]" >&2
  exit 2
fi

source_ll=$1
bundle=$2
wrapper=$3
model_link=$4
out=$5
llvm_clang=${6:-clang}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

for path in "$source_ll" "$bundle/atlas_source_mn_data.h" \
    "$wrapper/atlas_source_mn_model_wrapper.c" \
    "$model_link/atlas_model_entry.c" "$model_link/atlas_model_bindings.c"; do
  if [[ ! -f "$path" ]]; then
    echo "missing source-grid model input: $path" >&2
    exit 2
  fi
done
if [[ -n ${ATLAS_SOURCE_MN_RTL_OUTPUT_DIR:-} ]]; then
  python3 "$repo_root/test/check_model_source_grid_rtl_replay.py" \
    --bundle "$bundle/manifest.json" \
    --rtl-result "$(dirname "$ATLAS_SOURCE_MN_RTL_OUTPUT_DIR")/result.json" \
    --observed "$ATLAS_SOURCE_MN_RTL_OUTPUT_DIR"
fi
if [[ -e "$out" ]]; then
  echo "fresh output directory required: $out" >&2
  exit 2
fi
mkdir -p "$out"
mkdir -p "$out/capture"

"$llvm_clang" -O2 -c "$source_ll" -o "$out/model-x86.o"
cc -std=c11 -O1 -g -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DATLAS_DIAGNOSTIC_SOURCE_MN=1 \
  -DATLAS_DIAGNOSTIC_ALLOW_UNBOUND=1 \
  -DATLAS_DIAGNOSTIC_ASSUME_COHERENT=1 \
  -DATLAS_DIAGNOSTIC_PRECISION_POLICY=1 \
  -I "$repo_root/runtime" -I "$bundle" -I "$wrapper" -I "$model_link" \
  "$repo_root/test/model-source-grid-host-test.c" \
  "$out/model-x86.o" \
  "$wrapper/atlas_source_mn_model_wrapper.c" \
  "$model_link/atlas_model_entry.c" \
  "$model_link/atlas_model_bindings.c" \
  "$repo_root/runtime/atlas_host.c" \
  "$repo_root/runtime/atlas_model_tile.c" \
  "$repo_root/runtime/atlas_ordered_k.c" \
  "$repo_root/runtime/atlas_ordered_mn.c" \
  "$repo_root/runtime/atlas_source_mn.c" \
  "$repo_root/runtime/atlas_source_mn_memref.c" \
  -o "$out/model-host-test"
ATLAS_SOURCE_MN_CAPTURE_DIR="$out/capture" "$out/model-host-test"
echo "model source-grid host check passed: $out/model-host-test"

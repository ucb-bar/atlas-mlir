#!/usr/bin/env bash
set -euo pipefail

if (( $# != 6 && $# != 12 )); then
  echo "usage: $0 MERLIN_ROOT PYTHON LLVM_BIN SOURCE_LINALG COMPILED_NATIVE FRESH_OUT [--platform HOST_MAP SOC_MEMMAP CACHE_SYNC_C LINKER_SCRIPT MAX_POLLS]" >&2
  exit 2
fi

merlin_root=$(realpath "$1")
python_bin=$2
if [[ $python_bin == */* && $python_bin != /* ]]; then python_bin=$PWD/$python_bin; fi
llvm_bin=$(realpath "$3")
source_linalg=$(realpath "$4")
compiled_native=$(realpath "$5")
out=$6
mode=diagnostic
if (( $# == 12 )); then
  if [[ $7 != --platform || ! ${12} =~ ^[1-9][0-9]*$ || ${#12} -gt 9 ]]; then
    echo "platform build requires a positive bounded MAX_POLLS and --platform" >&2
    exit 2
  fi
  mode=platform
  host_map=$(realpath "$8")
  soc_memmap=$(realpath "$9")
  cache_sync_source=$(realpath "${10}")
  linker_script=$(realpath "${11}")
  max_polls=${12}
fi
if [[ -e $out || -L $out ]]; then
  echo "fresh output directory required: $out" >&2
  exit 2
fi
repo_root=$(realpath "$(dirname "$0")/../../..")
mkdir -p "$out/.tmp" "$out/host"
out=$(realpath "$out")

export TMPDIR="$out/.tmp"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$merlin_root/src:$repo_root/python"
export MERLIN_COMPILER_PYTHON="$python_bin"
merlin_mlir_install=$(realpath "$llvm_bin/..")
export MERLIN_MLIR_INSTALL="$merlin_mlir_install"

compile_defs=(-DATLAS_FREESTANDING=1 -DATLAS_DIAGNOSTIC_GRAPH_PRECISION=1)
extra_sources=()
if [[ $mode == platform ]]; then
  "$python_bin" "$repo_root/tools/atlas_graph_platform.py" \
    --compiled "$compiled_native" --host-map "$host_map" \
    --soc-memmap "$soc_memmap" \
    --out-c "$out/host/platform-map.c" \
    --out "$out/host/platform-preflight.json"
  compile_defs+=(-DATLAS_PLATFORM_MAP_BOUND=1 -DATLAS_PLATFORM_CACHE_SYNC=1)
  extra_sources+=("$out/host/platform-map.c" "$cache_sync_source")
else
  linker_script="$repo_root/examples/handoff/captured_attention_mlp/link.ld"
  max_polls=10000000
  compile_defs+=(-DATLAS_DIAGNOSTIC_ALLOW_UNBOUND=1 \
                 -DATLAS_DIAGNOSTIC_ASSUME_COHERENT=1)
fi
compile_defs+=("-DATLAS_MAX_POLLS=$max_polls")

"$python_bin" -m merlin.targetgen.cli native-bridge-graph \
  --linalg "$source_linalg" --entry forward \
  --compiled "$compiled_native" --device-symbol merlin_native_graph \
  --out "$out/host-bridge"
"$python_bin" "$repo_root/tools/atlas_graph_model_bridge.py" \
  --linalg "$source_linalg" --host-bridge "$out/host-bridge" \
  --compiled "$compiled_native" --out "$out/graph-wrapper"
"$python_bin" -m merlin.llvmlower.cli \
  "$out/host-bridge/host-bridge.mlir" --target riscv --out "$out/host-lowered"

"$llvm_bin/clang" --target=riscv64-unknown-elf -march=rv64gcv \
  -mabi=lp64d -mcmodel=medany -std=c11 -O2 \
  -Wall -Wextra -Werror -Wno-gcc-install-dir-libstdcxx \
  -ffreestanding -fno-builtin -fno-stack-protector -fno-pic \
  -nostdlib -fuse-ld=lld \
  "${compile_defs[@]}" \
  -I "$out/graph-wrapper" -I "$repo_root/runtime" \
  -I "$merlin_root/merlin/runtime/c" \
  "$repo_root/examples/handoff/captured_attention_mlp/start.S" \
  "$repo_root/examples/handoff/captured_attention_mlp/host_graph_mailbox.c" \
  "$out/graph-wrapper/atlas_graph_model_wrapper.c" \
  "$out/graph-wrapper/atlas_graph_host_entry.c" \
  "$repo_root/runtime/atlas_graph_memref.c" \
  "$repo_root/runtime/atlas_host.c" \
  ${extra_sources[@]+"${extra_sources[@]}"} \
  "$out/host-lowered/model_rv64gcv.o" \
  -Wl,-T,"$linker_script" \
  -Wl,--no-relax -Wl,-Map,"$out/host/shuttle-graph-host.map" \
  -o "$out/host/shuttle-graph-host.elf"

if [[ $mode == platform ]]; then
  "$python_bin" "$repo_root/tools/atlas_graph_platform.py" \
    --compiled "$compiled_native" --host-map "$host_map" \
    --soc-memmap "$soc_memmap" \
    --host-elf "$out/host/shuttle-graph-host.elf" \
    --out "$out/host/platform-elf-check.json"
fi

"$python_bin" - "$out" "$compiled_native" "$llvm_bin/llvm-nm" "$source_linalg" "$mode" "${cache_sync_source:-}" "$max_polls" <<'PY'
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

out, compiled, nm, source, mode, cache_source = (
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]),
    sys.argv[5], sys.argv[6]
)
max_polls = int(sys.argv[7])
elf = (out / "host/shuttle-graph-host.elf").read_bytes()
if len(elf) < 64 or elf[:6] != b"\x7fELF\x02\x01" or \
        struct.unpack_from("<HH", elf, 16) != (2, 243):
    raise ValueError("host output is not a little-endian RV64 executable")
symbols = subprocess.run([nm, "-P", str(out / "host/shuttle-graph-host.elf")],
                         check=True, capture_output=True, text=True).stdout
wrapper = json.loads((out / "graph-wrapper/manifest.json").read_text())
input_count = wrapper.get("input_count")
output_count = wrapper.get("output_count")
if (wrapper.get("status") != "generated_unlinked" or input_count != 6 or
        type(output_count) is not int or not 1 <= output_count <= 16):
    raise ValueError("graph wrapper has an unsupported mailbox signature")
matches = [row.split() for row in symbols.splitlines()
           if row.startswith("atlas_graph_mailbox ")]
if len(matches) != 1 or len(matches[0]) != 4 or matches[0][1] != "B":
    raise ValueError("host ELF has no unique mailbox symbol")
address = int(matches[0][2], 16)
size = int(matches[0][3], 16)
output_offset = 20 + input_count * 4096
guard_after = output_offset + output_count * 4096
if size != guard_after + 4 or address & 63:
    raise ValueError("host ELF mailbox layout or alignment changed")
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


platform = json.loads((out / "host/platform-elf-check.json").read_text()) if mode == "platform" else None
if platform is not None and platform.get("status") != "platform_elf_checked":
    raise ValueError("final platform ELF check is missing")
record = {
    "schema": "atlas.graph_mailbox_handoff.v2" if platform else
              "atlas.diagnostic_graph_mailbox_handoff.v1",
    "status": "platform_map_checked_execution_unqualified" if platform else
              "diagnostic_unbound",
    "engine": "merlin_native",
    "bitstream_bound": bool(platform),
    "cache_policy": "callback_supplied_unqualified" if platform else
                    "assumed_coherent_for_diagnostic_build",
    "source_sha256": sha(source),
    "host_bridge_sha256": sha(out / "host-bridge/host-bridge.mlir"),
    "native_manifest_sha256": sha(compiled / "manifest.json"),
    "program_sha256": sha(compiled / "program.bin"),
    "elf_sha256": hashlib.sha256(elf).hexdigest(),
    "entry": struct.unpack_from("<Q", elf, 24)[0],
    "mailbox_address": address,
    "mailbox_bytes": size,
    "mailbox_offsets": {
        "request_id": 0, "response_id": 4, "status": 8,
        "guard_before": 16, "inputs": 20, "outputs": output_offset,
        "guard_after": guard_after,
    },
    "input_count": input_count,
    "output_count": output_count,
    "tensor_shape": [32, 32],
    "tensor_dtype": "f32",
    "max_polls": max_polls,
    "invocation": "write all inputs, then request_id; wait for matching response_id",
}
if platform:
    record.update({
        "bitstream_identity": platform["bitstream_identity"],
        "soc_memmap_sha256": platform["soc_memmap_sha256"],
        "host_map_sha256": platform["host_map_sha256"],
        "cache_sync_source_sha256": sha(Path(cache_source)),
        "platform_elf_check_sha256": sha(out / "host/platform-elf-check.json"),
    })
(out / "host/handoff-status.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"status": record["status"],
                  "elf": str(out / "host/shuttle-graph-host.elf"),
                  "mailbox_address": address}, sort_keys=True))
PY

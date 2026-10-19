#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash build.sh VERIFIED_HOST_DATA_DIR FRESH_OUTPUT_DIR" >&2
  exit 2
fi
host_data=$1
output=$2
if [[ -e "$output" ]]; then
  echo "output directory must be fresh: $output" >&2
  exit 2
fi
: "${TMPDIR:?set TMPDIR to a scratch directory}"
source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
startup_root=$source_root/../captured_attention_mlp
stage=$(mktemp -d "$TMPDIR/atlas-source-mn-build.XXXXXX")
trap 'echo "incomplete host build retained at $stage" >&2' ERR
bundle=$stage/host-data
mkdir "$bundle"
python3 - "$host_data" "$bundle" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source, destination = (Path(argument) for argument in sys.argv[1:])
manifest_path = source / "manifest.json"
if source.is_symlink() or not source.is_dir() or manifest_path.is_symlink():
    raise SystemExit("source host bundle is missing or linked")
manifest_bytes = manifest_path.read_bytes()
record = json.loads(manifest_bytes)
required = {
    "atlas_source_mn_data.h", "row_major_k_window.h", "finite_float_boundary.h",
    "atlas_host.h", "atlas_host.c", "atlas_ordered_k.h", "atlas_ordered_k.c",
    "atlas_ordered_mn.h", "atlas_ordered_mn.c", "atlas_source_mn.h",
    "atlas_source_mn.c", "atlas_model_tile.h", "atlas_source_mn_memref.h",
    "atlas_source_mn_memref.c",
}
files = record.get("files")
if (record.get("schema") != "atlas.source_mn_host_data.v1"
        or record.get("engine") != "merlin_native"
        or record.get("status") not in ("diagnostic_unbound", "diagnostic_unqualified")
        or not isinstance(files, dict) or set(files) != required
        or {member.name for member in source.iterdir()} != required | {"manifest.json"}):
    raise SystemExit("source host bundle has unexpected members or identity")
for name, expected in files.items():
    member = source / name
    if member.is_symlink() or not member.is_file():
        raise SystemExit(f"source host member is missing or linked: {name}")
    payload = member.read_bytes()
    if not payload or hashlib.sha256(payload).hexdigest() != expected:
        raise SystemExit(f"source host member changed after export: {name}")
    (destination / name).write_bytes(payload)
(destination / "manifest.json").write_bytes(manifest_bytes)
PY
compiler=${ATLAS_HOST_CC:-clang}
linker=${ATLAS_HOST_LD:-ld.lld}
flags=(--target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d
       -mcmodel=medany -std=c11 -O1 -ffreestanding -fno-builtin
       -fno-stack-protector -fno-vectorize -fno-slp-vectorize
       -Wall -Wextra -Werror -DATLAS_DIAGNOSTIC_SOURCE_MN
       -I"$bundle")
for name in host atlas_host atlas_ordered_k atlas_ordered_mn atlas_source_mn; do
  source=$source_root/host.c
  if [[ "$name" != host ]]; then source=$bundle/$name.c; fi
  "$compiler" "${flags[@]}" -c "$source" -o "$stage/$name.o"
done
"$compiler" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d \
  -c "$startup_root/start.S" -o "$stage/start.o"
"$linker" -m elf64lriscv -T "$startup_root/link.ld" \
  --no-undefined "$stage/start.o" "$stage/host.o" \
  "$stage/atlas_host.o" "$stage/atlas_ordered_k.o" \
  "$stage/atlas_ordered_mn.o" "$stage/atlas_source_mn.o" \
  -o "$stage/atlas-source-mn.elf"
python3 - "$stage" "$source_root/host.c" "$startup_root" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

stage, host_source, startup = (Path(argument) for argument in sys.argv[1:])
def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
record = {
    "schema": "atlas.source_mn_host_build.v1",
    "status": "linked_diagnostic_elf",
    "host_data_manifest_sha256": digest(stage / "host-data/manifest.json"),
    "host_source_sha256": digest(host_source),
    "startup_sha256": digest(startup / "start.S"),
    "linker_script_sha256": digest(startup / "link.ld"),
    "elf_sha256": digest(stage / "atlas-source-mn.elf"),
}
(stage / "build-manifest.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
PY
mkdir -p "$(dirname "$output")"
mv "$stage" "$output"
trap - ERR

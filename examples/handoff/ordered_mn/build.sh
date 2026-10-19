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
source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
startup_root=$source_root
if [[ ! -f "$startup_root/start.S" || ! -f "$startup_root/link.ld" ]]; then
  startup_root=$source_root/../captured_attention_mlp
fi
mkdir -p "$output"
compiler=${ATLAS_HOST_CC:-clang}
linker=${ATLAS_HOST_LD:-ld.lld}
flags=(--target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d
       -mcmodel=medany -std=c11 -O1 -ffreestanding -fno-builtin
       -fno-stack-protector -fno-vectorize -fno-slp-vectorize
       -Wall -Wextra -Werror -DATLAS_DIAGNOSTIC_ORDERED_K
       -I"$host_data")
for name in host atlas_host atlas_ordered_k atlas_ordered_mn; do
  source=$source_root/host.c
  if [[ "$name" != host ]]; then source=$host_data/$name.c; fi
  "$compiler" "${flags[@]}" -c "$source" -o "$output/$name.o"
done
"$compiler" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d \
  -c "$startup_root/start.S" -o "$output/start.o"
"$linker" -m elf64lriscv -T "$startup_root/link.ld" \
  --no-undefined "$output/start.o" "$output/host.o" \
  "$output/atlas_host.o" "$output/atlas_ordered_k.o" \
  "$output/atlas_ordered_mn.o" -o "$output/atlas-ordered-mn.elf"

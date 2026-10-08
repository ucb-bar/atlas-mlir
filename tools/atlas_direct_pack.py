#!/usr/bin/env python3
"""Package a checked Atlas physical stream directly as an IMEM image.

The device image is the bytes emitted by atlas-emit.  The separate Rocket host
loads these bytes; neither LLVM IR nor a RISC-V object is involved in making
the Atlas program.  This first format is one closed reset-entry program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

from atlas_boot_pack import IMEM_BASE, IMEM_SIZE, START_CSR, ECALL, validate_layout


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_words(program: dict, revision: str) -> tuple[int, ...]:
    if (program.get("schema") != "atlas.physical_program.v1" or
            program.get("selected_rtl_revision") != revision or
            program.get("pc_unit") != "instruction_word" or
            program.get("word_endianness") != "little" or
            program.get("entry_word_index") != 0):
        raise ValueError("physical program disagrees with selected Atlas layout")
    rows = program.get("instructions")
    if not isinstance(rows, list) or len(rows) != program.get("word_count") or not rows:
        raise ValueError("physical program has an incomplete instruction list")
    if len(rows) * 4 > IMEM_SIZE:
        raise ValueError("physical program exceeds Atlas IMEM")
    words: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("word_index") != index:
            raise ValueError("physical program has noncontiguous word indexes")
        word = row.get("word_u32")
        if (isinstance(word, bool) or not isinstance(word, int) or
                not 0 <= word <= 0xFFFFFFFF or word & 3 != 3 or
                row.get("word_hex") != f"{word:08x}"):
            raise ValueError("invalid or mismatched 32-bit Atlas word")
        words.append(word)
        control = row.get("control")
        if not isinstance(control, dict):
            raise ValueError("physical instruction lacks control metadata")
        kind = control.get("kind")
        if kind == "jump_register":
            raise ValueError("direct image v1 requires resolved control targets")
        if kind in ("conditional_direct", "jump_direct"):
            target = control.get("target_word_index")
            if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(rows):
                raise ValueError("Atlas branch target escapes the direct image")
            if (index + 1 >= len(rows) or
                    not isinstance(rows[index + 1], dict) or
                    control.get("delay_slot_word_index") != index + 1 or
                    rows[index + 1].get("control", {}).get("kind") in
                    ("conditional_direct", "jump_direct", "jump_register")):
                raise ValueError("Atlas branch has an invalid delay slot")
        elif kind not in ("sequential", "trap"):
            raise ValueError("unknown Atlas control kind")
    if words[-1] != ECALL or rows[-1]["control"]["kind"] != "trap":
        raise ValueError("direct image must end with an ECALL trap")
    if ECALL in words[:-1]:
        raise ValueError("direct image contains an early ECALL")
    return tuple(words)


def pack(source: Path, atlas_emit: Path, layout_path: Path, out_dir: Path) -> dict:
    if out_dir.exists():
        raise ValueError("output directory already exists; fresh package required")
    layout = validate_layout(json.loads(layout_path.read_text()))
    source_bytes = source.read_bytes()
    emitted = subprocess.run([str(atlas_emit), "--program-json", str(source)],
                             capture_output=True, text=True, check=True)
    program = json.loads(emitted.stdout)
    words = checked_words(program, layout["selected_rtl_revision"])
    image = struct.pack(f"<{len(words)}I", *words)
    call = layout.get("call")
    if call is None:
        arguments, returns = [], []
    else:
        arguments = [{"name": "input", "kind": "runtime_dram_pointer",
                      "mailbox_offset_bytes": call["input_pointer_offset_bytes"],
                      "region": call["input_region"], "size_bytes": call["tensor_bytes"]}]
        returns = [{"name": "output", "kind": "runtime_dram_pointer",
                    "mailbox_offset_bytes": call["output_pointer_offset_bytes"],
                    "region": call["output_region"], "size_bytes": call["tensor_bytes"]}]
    manifest = {
        "schema": "atlas.direct-image.v1",
        "selected_rtl_revision": layout["selected_rtl_revision"],
        "source_mlir_sha256": _sha(source_bytes),
        "physical_program_sha256": _sha(emitted.stdout.encode()),
        "program_sha256": _sha(image),
        "program_file": "program.bin",
        "program_words": len(words),
        "entry_pc_word": 0,
        "imem_tl_byte_base": IMEM_BASE,
        "imem_capacity_bytes": IMEM_SIZE,
        "start_csr_tl_byte_address": START_CSR,
        "start_csr_value": 1,
        "completion": {"kind": "ecall_halt", "ecall_pc_word": len(words) - 1},
        "arguments": arguments,
        "returns": returns,
        "dram": layout,
        "register_initialization_proved": False,
        "program_mailbox_binding_proved_by_packer": False,
    }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".atlas-direct-", dir=out_dir.parent))
    try:
        (temporary / "program.bin").write_bytes(image)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                           sort_keys=True) + "\n")
        temporary.rename(out_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--atlas-emit", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = pack(args.source, args.atlas_emit, args.layout, args.out)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError,
            json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

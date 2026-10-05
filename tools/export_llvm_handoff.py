#!/usr/bin/env python3
"""Export the hand-authored Atlas examples and their exact LLVM MLIR stage.

This is a diagnostic handoff, not a model compiler or an execution certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ("handoff_mlp_tile", "handoff_attention_tile")
RTL_REVISION = "0079c0541111197741a231c002e3843fa6f545b2"


def run(command: list[str], *, input_text: str | None = None) -> str:
    return subprocess.run(command, input=input_text, text=True, check=True,
                          capture_output=True).stdout


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-bin-dir", type=Path, required=True)
    parser.add_argument("--llvm-bin-dir", type=Path, required=True)
    parser.add_argument("--linker", type=Path, required=True,
                        help="explicit ld.lld path used only to link the inspectable ELF")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    atlas = args.atlas_bin_dir.resolve(strict=True)
    llvm = args.llvm_bin_dir.resolve(strict=True)
    # Keep the ld.lld basename: some installations symlink it to a multi-tool
    # driver whose linker flavor is selected from argv[0].
    linker = args.linker.absolute()
    if not linker.is_file():
        parser.error("--linker must name an existing ld.lld executable")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be empty to prevent stale handoff artifacts")
    output.mkdir(parents=True, exist_ok=True)
    revision = run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip()
    dirty = bool(run(["git", "-C", str(ROOT), "status", "--porcelain"]))
    manifest: dict[str, object] = {
        "schema": "atlas-llvm-handoff-v3",
        "dialect_revision": revision,
        "working_tree_dirty": dirty,
        "selected_rtl_revision": RTL_REVISION,
        "llvm_version": run([str(llvm / "mlir-translate"), "--version"]).strip(),
        "linker_version": run([str(linker), "--version"]).strip(),
        "scope": "hand-authored fixed 32x32 diagnostic streams; no general MLP or attention claim",
        "examples": {},
    }

    for name in EXAMPLES:
        source = ROOT / "test" / "examples" / f"{name}.mlir"
        raw = source.read_bytes()
        words_text = run([str(atlas / "atlas-emit"), str(source)])
        words = tuple(int(line, 16) for line in words_text.splitlines())
        word_map = json.loads(run([str(atlas / "atlas-emit"), "--map-json",
                                   str(source)]))
        mapped = word_map.get("operations", ())
        if (word_map.get("schema") != "atlas.machine_word_map.v1"
                or word_map.get("word_count") != len(words)
                or len(mapped) != len(words)
                or any(row["word_index"] != index
                       or row["byte_offset"] != 4 * index
                       or row["word_hex"] != f"{words[index]:08x}"
                       for index, row in enumerate(mapped))):
            raise RuntimeError(f"{name}: source operation map differs from emitted words")
        structured_mlir = run([
            str(atlas / "atlas-opt"), "--convert-atlas-to-llvm-calls", str(source),
        ]).rstrip() + "\n"
        if (structured_mlir.count("llvm.call @atlas_emit_") != len(words)
                or "llvm.inline_asm" in structured_mlir):
            raise RuntimeError(f"{name}: structured LLVM call count differs from emitter")
        llvm_mlir = run([
            str(atlas / "atlas-opt"), "--finalize-atlas-llvm-calls", "-",
        ], input_text=structured_mlir).rstrip() + "\n"
        if llvm_mlir.count("llvm.inline_asm has_side_effects") != 1:
            raise RuntimeError(f"{name}: expected one side-effecting LLVM inline asm")
        if llvm_mlir.count(".word 0x") != len(words):
            raise RuntimeError(f"{name}: LLVM word count differs from emitter")
        llvm_ir = run([str(llvm / "mlir-translate"), "--mlir-to-llvmir"],
                      input_text=llvm_mlir)
        asm = run([str(llvm / "llc"), "-mtriple=riscv32-unknown-elf",
                   "-mattr=-c", "-filetype=asm", "-o", "-"],
                  input_text=llvm_ir)
        if asm.count(".word") != len(words):
            raise RuntimeError(f"{name}: LLVM assembly word count differs from emitter")
        obj = output / f"{name}.o"
        subprocess.run([str(llvm / "llc"), "-mtriple=riscv32-unknown-elf",
                        "-mattr=-c", "-filetype=obj", "-o", str(obj)], input=llvm_ir,
                       text=True, check=True, capture_output=True)
        elf = output / f"{name}.elf"
        subprocess.run([str(linker), "-m", "elf32lriscv", "--no-relax",
                        "-Ttext=0", "-e", "atlas_program", "-o", str(elf),
                        str(obj)], check=True, capture_output=True)
        elf_header = run([str(llvm / "llvm-readelf"), "-h", str(elf)])
        if ("Type:                              EXEC" not in elf_header
                or "Machine:                           RISC-V" not in elf_header
                or "Entry point address:               0x0" not in elf_header):
            raise RuntimeError(f"{name}: linked file is not the expected RISC-V ELF")
        text_section = output / f"{name}.text.bin"
        subprocess.run([str(llvm / "llvm-objcopy"), "--dump-section",
                        f".text={text_section}", str(obj)],
                       check=True, capture_output=True)
        expected = b"".join(struct.pack("<I", word) for word in words)
        actual = text_section.read_bytes()
        if not actual.startswith(expected):
            raise RuntimeError(f"{name}: LLVM object words differ from Atlas emitter")
        linked_text = output / f"{name}.elf.text.bin"
        subprocess.run([str(llvm / "llvm-objcopy"), "--dump-section",
                        f".text={linked_text}", str(elf)],
                       check=True, capture_output=True)
        if linked_text.read_bytes() != actual:
            raise RuntimeError(f"{name}: linked ELF changed the Atlas instruction stream")
        raw_disassembly = run([str(llvm / "llvm-objdump"), "-d", str(elf)])
        # llvm-objdump prints its input path above the disassembly. Normalize
        # only that path so snapshots do not depend on the invocation's out root.
        disassembly = raw_disassembly.replace(str(elf), "<linked-elf>")
        if not re.search(r"<atlas_program>:", disassembly):
            raise RuntimeError(f"{name}: linked ELF lost atlas_program symbol")
        shutil.copyfile(source, output / f"{name}.atlas.mlir")
        (output / f"{name}.llvm-structured.mlir").write_text(structured_mlir)
        (output / f"{name}.llvm.mlir").write_text(llvm_mlir)
        (output / f"{name}.ll").write_text(llvm_ir)
        (output / f"{name}.s").write_text(asm)
        (output / f"{name}.disasm.txt").write_text(disassembly)
        (output / f"{name}.words.txt").write_text(words_text)
        map_bytes = (json.dumps(word_map, indent=2, sort_keys=True) + "\n").encode()
        (output / f"{name}.word-map.json").write_bytes(map_bytes)
        manifest["examples"][name] = {
            "word_count": len(words),
            "source_sha256": digest(raw),
            "structured_llvm_mlir_sha256": digest(structured_mlir.encode()),
            "llvm_mlir_sha256": digest(llvm_mlir.encode()),
            "object_sha256": digest(obj.read_bytes()),
            "elf_sha256": digest(elf.read_bytes()),
            "assembly_sha256": digest(asm.encode()),
            "disassembly_sha256": digest(disassembly.encode()),
            "object_prefix_matches_emitter": True,
            "linked_text_matches_object": True,
            "word_map_sha256": digest(map_bytes),
            "word_map_schema": word_map["schema"],
            "text_bytes": len(actual),
        }

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output / "manifest.json")


if __name__ == "__main__":
    main()

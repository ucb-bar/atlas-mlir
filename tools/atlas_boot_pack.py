#!/usr/bin/env python3
"""Pack checked Atlas reset-entry instructions for the selected standalone core.

The direct route uses atlas-emit output without sending Atlas instructions
through LLVM. The legacy ELF-object route remains available for comparison.
Neither route proves that registers are initialized or defines a C-callable ABI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile


IMEM_BASE = 0x20000
IMEM_SIZE = 0x20000
START_CSR = 0x18
DRAM_WINDOW_SIZE = 1 << 20
ECALL = 0x00000073
RET = 0x00008067


def _number(value: int | str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("address or size must be an integer")
    return int(value, 0) if isinstance(value, str) else int(value)


def validate_layout(layout: dict) -> dict:
    schema = layout.get("schema")
    if schema not in ("atlas.reset-entry-layout.v1", "atlas.reset-entry-layout.v2"):
        raise ValueError("unsupported reset-entry layout schema")
    revision = layout.get("selected_rtl_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("selected RTL revision must be a full commit ID")
    base = _number(layout.get("dram_window_base"))
    if base < 0 or base > 0xFFFFFFFF - DRAM_WINDOW_SIZE + 1 or base % DRAM_WINDOW_SIZE:
        raise ValueError("DRAM window base must be aligned and fit 32 bits")
    rows = layout.get("regions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("at least one DRAM region is required")
    normalized = []
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "kind", "address", "size_bytes"}:
            raise ValueError("region requires name, kind, address, and size_bytes")
        name, kind = row["name"], row["kind"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("region name must be a simple identifier")
        if name in names:
            raise ValueError("duplicate DRAM region name")
        names.add(name)
        allowed_kinds = (("input", "output", "guard") if schema.endswith("v1")
                         else ("mailbox", "input_pool", "output_pool", "guard"))
        if kind not in allowed_kinds:
            raise ValueError("region kind is invalid for reset-entry layout schema")
        address, size = _number(row["address"]), _number(row["size_bytes"])
        if (size <= 0 or address % 32 or size % 32 or
                address < base or address + size > base + DRAM_WINDOW_SIZE):
            raise ValueError("region must be a 32-byte-aligned range in DRAM window")
        normalized.append({"name": name, "kind": kind, "address": address,
                           "size_bytes": size})
    ordered = sorted(normalized, key=lambda x: x["address"])
    for left, right in zip(ordered, ordered[1:]):
        if left["address"] + left["size_bytes"] > right["address"]:
            raise ValueError("overlapping DRAM regions")
    if schema.endswith("v1"):
        if not any(row["kind"] == "input" for row in normalized):
            raise ValueError("an input region is required")
        if not any(row["kind"] == "output" for row in normalized):
            raise ValueError("an output region is required")
        return {"schema": schema, "selected_rtl_revision": revision,
                "dram_window_base": base, "regions": normalized}

    call = layout.get("call")
    fields = {"kind", "mailbox_region", "input_region", "output_region",
              "input_pointer_offset_bytes", "output_pointer_offset_bytes",
              "tensor_bytes", "buffer_alignment_bytes"}
    if not isinstance(call, dict) or set(call) != fields or call["kind"] != "mailbox-pointers":
        raise ValueError("v2 requires a complete mailbox-pointers call declaration")
    by_name = {row["name"]: row for row in normalized}
    for field, kind in (("mailbox_region", "mailbox"),
                        ("input_region", "input_pool"),
                        ("output_region", "output_pool")):
        name = call[field]
        if not isinstance(name, str) or name not in by_name or by_name[name]["kind"] != kind:
            raise ValueError(f"{field} must name a {kind} region")
    if (sum(row["kind"] == "mailbox" for row in normalized) != 1 or
            sum(row["kind"] == "input_pool" for row in normalized) != 1 or
            sum(row["kind"] == "output_pool" for row in normalized) != 1):
        raise ValueError("v2 requires one mailbox and one pool per pointer")
    size = _number(call["tensor_bytes"])
    alignment = _number(call["buffer_alignment_bytes"])
    if size <= 0 or size % 32 or alignment < 32 or alignment & (alignment - 1):
        raise ValueError("mailbox tensor size/alignment must be positive 32-byte units")
    for field in ("input_region", "output_region"):
        if by_name[call[field]]["size_bytes"] < size:
            raise ValueError("mailbox buffer pool is smaller than one tensor")
    mailbox = by_name[call["mailbox_region"]]
    offsets = []
    for field in ("input_pointer_offset_bytes", "output_pointer_offset_bytes"):
        offset = _number(call[field])
        if offset < 0 or offset % 4 or offset + 4 > mailbox["size_bytes"]:
            raise ValueError("mailbox pointer offset is outside descriptor")
        offsets.append(offset)
    if offsets[0] == offsets[1]:
        raise ValueError("mailbox pointer fields overlap")
    return {"schema": schema, "selected_rtl_revision": revision,
            "dram_window_base": base, "regions": normalized,
            "call": {**call, "tensor_bytes": size,
                     "buffer_alignment_bytes": alignment,
                     "input_pointer_offset_bytes": offsets[0],
                     "output_pointer_offset_bytes": offsets[1]}}


def validate_captured_layout(imported: dict, layout: dict,
                             program_sha256: str) -> int:
    """Check every imported physical tile fits its packaged DRAM region.

    The compiler and emitter bind the instruction stream to the imported ABI.
    This check only binds that ABI to the package layout; it does not prove
    instruction-level memory effects.
    """
    if layout["schema"] != "atlas.reset-entry-layout.v1":
        raise ValueError("captured import requires a direct reset-entry layout")
    if (imported.get("schema") != "atlas.oot.captured_linalg_import.v1"
            or imported.get("selected_rtl_revision") != layout["selected_rtl_revision"]
            or imported.get("program", {}).get("program_sha256") != program_sha256):
        raise ValueError("captured import identity differs from packaged program or RTL")

    slots = imported.get("input_slots")
    outputs = imported.get("outputs")
    if outputs is None:
        outputs = [imported.get("output")]
    elif imported.get("output") is not None and outputs != [imported["output"]]:
        raise ValueError("captured import has conflicting output descriptions")
    if (not isinstance(slots, list) or not slots or not isinstance(outputs, list)
            or not outputs or any(not isinstance(row, dict) for row in slots + outputs)):
        raise ValueError("captured import has malformed physical slots")

    def inside(kind: str, address: int, size: int) -> bool:
        return any(row["kind"] == kind and row["address"] <= address
                   and address + size <= row["address"] + row["size_bytes"]
                   for row in layout["regions"])

    input_base = slots[0].get("dram_address")
    output_base = outputs[0].get("dram_address")
    if type(input_base) is not int or type(output_base) is not int:
        raise ValueError("captured import has noninteger DRAM bases")
    for index, slot in enumerate(slots):
        size = slot.get("payload_bytes")
        reserved = slot.get("reserved_slot_bytes", size)
        address = slot.get("dram_address")
        if (type(slot.get("slot")) is not int or slot["slot"] != index
                or slot.get("source_kind") not in ("input", "param", "generated")
                or type(size) is not int or size not in (1024, 2048)
                or type(reserved) is not int or reserved < size or reserved > 2048
                or type(address) is not int or address != input_base + index * 2048
                or not inside("input", address, reserved)):
            raise ValueError(f"captured input slot {index} escapes packaged layout")
    for index, output in enumerate(outputs):
        address = output.get("dram_address")
        if (type(output.get("index")) is not int or output["index"] != index
                or output.get("payload_bytes") != 2048
                or output.get("physical_layout") != "bf16_pair_halves"
                or type(address) is not int or address != output_base + index * 2048
                or not inside("output", address, 2048)):
            raise ValueError(f"captured output {index} escapes packaged layout")
    return len(outputs)


def mailbox_descriptor(layout: dict, input_address: int, output_address: int) -> bytes:
    """Validate one runtime invocation and encode its little-endian mailbox.

    This validates the host's declared pointer arguments. It does not inspect
    or alter the compiled instruction stream or infer its pointer dataflow.
    """
    if layout.get("schema") != "atlas.reset-entry-layout.v2":
        raise ValueError("mailbox descriptor requires a v2 layout")
    call = layout["call"]
    regions = {row["name"]: row for row in layout["regions"]}
    size = call["tensor_bytes"]
    alignment = call["buffer_alignment_bytes"]
    for label, address, field in (("input", input_address, "input_region"),
                                  ("output", output_address, "output_region")):
        if isinstance(address, bool) or not isinstance(address, int) or address < 0 or address > 0xFFFFFFFF:
            raise ValueError(f"{label} pointer must be a 32-bit byte address")
        region = regions[call[field]]
        if (address % alignment or address < region["address"] or
                address + size > region["address"] + region["size_bytes"]):
            raise ValueError(f"{label} pointer is outside its aligned declared pool")
    descriptor = bytearray(regions[call["mailbox_region"]]["size_bytes"])
    struct.pack_into("<I", descriptor, call["input_pointer_offset_bytes"], input_address)
    struct.pack_into("<I", descriptor, call["output_pointer_offset_bytes"], output_address)
    return bytes(descriptor)


def validate_text(metadata: dict, text: bytes) -> tuple[int, ...]:
    if metadata["FileSummary"]["Format"] != "elf32-littleriscv":
        raise ValueError("entry requires ELF32 little-endian RISC-V")
    sections = {row["Section"]["Index"]: row["Section"]
                for row in metadata["Sections"]}
    executable = [section for section in sections.values()
                  if section["Name"]["Name"] == ".text"]
    if len(executable) != 1 or executable[0]["Size"] != len(text):
        raise ValueError("one complete .text section required")
    text_section = executable[0]
    flags = {flag["Name"] for flag in text_section["Flags"]["Flags"]}
    if not {"SHF_ALLOC", "SHF_EXECINSTR"} <= flags:
        raise ValueError(".text must be allocated executable code")
    for group in metadata.get("Relocations", []):
        relocation_section = sections[group["SectionIndex"]]
        if relocation_section["Info"] == text_section["Index"] and group["Relocs"]:
            raise ValueError("unapplied .text relocation")
    functions = [row["Symbol"] for row in metadata["Symbols"]
                 if row["Symbol"]["Name"]["Name"] == "atlas_program"]
    if len(functions) != 1:
        raise ValueError("one atlas_program symbol required")
    function = functions[0]
    if (function["Type"]["Name"] != "Function" or
            function["Section"]["Name"] != ".text" or
            function["Value"] != 0 or function["Size"] != len(text)):
        raise ValueError("atlas_program must occupy .text from entry zero")
    if not text or len(text) % 4 or len(text) > IMEM_SIZE:
        raise ValueError("entry exceeds IMEM or is not complete 32-bit words")
    words = struct.unpack(f"<{len(text) // 4}I", text)
    if any(word & 3 != 3 for word in words):
        raise ValueError("entry contains compressed or misaligned instruction")
    if len(words) < 2 or words[-2:] != (ECALL, RET):
        raise ValueError("entry must halt by ECALL before LLVM RET")
    if ECALL in words[:-2]:
        raise ValueError("early ECALL before declared completion")
    return words


def _tool(name: str, llvm_bin: pathlib.Path | None) -> str:
    path = llvm_bin / name if llvm_bin is not None else shutil.which(name)
    if path is None or not pathlib.Path(path).is_file():
        raise ValueError(f"required LLVM tool unavailable: {name}")
    return str(path)


def pack(object_path: pathlib.Path, source_path: pathlib.Path,
         atlas_emit: pathlib.Path, layout_path: pathlib.Path,
         out_dir: pathlib.Path, llvm_bin: pathlib.Path | None) -> dict:
    if out_dir.exists():
        raise ValueError("output directory already exists; fresh package required")
    object_bytes = object_path.read_bytes()
    source_bytes = source_path.read_bytes()
    layout = validate_layout(json.loads(layout_path.read_text()))
    emitted = subprocess.run([str(atlas_emit), str(source_path)],
                             capture_output=True, text=True, check=True)
    lines = emitted.stdout.splitlines()
    if not lines or any(not re.fullmatch(r"[0-9a-fA-F]{8}", line) for line in lines):
        raise ValueError("atlas-emit did not produce complete 32-bit words")
    selected_words = tuple(int(line, 16) for line in lines)
    inspected = subprocess.run(
        [_tool("llvm-readobj", llvm_bin), "--elf-output-style=JSON",
         "--sections", "--symbols", "--relocations", str(object_path)],
        capture_output=True, text=True, check=True)
    parsed = json.loads(inspected.stdout)
    if not isinstance(parsed, list) or len(parsed) != 1:
        raise ValueError("one ELF object required")
    with tempfile.TemporaryDirectory() as temporary:
        text_path = pathlib.Path(temporary) / "text.bin"
        subprocess.run([_tool("llvm-objcopy", llvm_bin), "--dump-section",
                        f".text={text_path}", str(object_path)],
                       capture_output=True, text=True, check=True)
        text = text_path.read_bytes()
    words = validate_text(parsed[0], text)
    if words != selected_words + (RET,):
        raise ValueError("ELF .text differs from checked Atlas source words")
    call = layout.get("call")
    if call is None:
        arguments, returns = [], []
        scope = "selected standalone AtlasCore reset entry; not a callable ABI or ELF loader"
    else:
        arguments = [{"name": "input", "kind": "runtime_dram_pointer",
                      "mailbox_offset_bytes": call["input_pointer_offset_bytes"],
                      "region": call["input_region"], "size_bytes": call["tensor_bytes"]}]
        returns = [{"name": "output", "kind": "runtime_dram_pointer",
                    "mailbox_offset_bytes": call["output_pointer_offset_bytes"],
                    "region": call["output_region"], "size_bytes": call["tensor_bytes"]}]
        scope = ("selected standalone AtlasCore mailbox reset-entry call; "
                 "not a C ABI or general ELF loader")
    manifest = {
        "schema": "atlas.boot-capsule.v1",
        "scope": scope,
        "packer_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "source_object_sha256": hashlib.sha256(object_bytes).hexdigest(),
        "source_mlir_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "program_sha256": hashlib.sha256(text).hexdigest(),
        "program_file": "program.bin",
        "entry_symbol": "atlas_program",
        "entry_pc_word": 0,
        "imem_tl_byte_base": IMEM_BASE,
        "imem_capacity_bytes": IMEM_SIZE,
        "program_words": len(words),
        "start_csr_tl_byte_address": START_CSR,
        "start_csr_value": 1,
        "completion": {"kind": "ecall_halt", "ecall_pc_word": len(words) - 2,
                       "unreachable_llvm_ret_pc_word": len(words) - 1},
        "arguments": arguments, "returns": returns,
        "register_initialization_proved": False,
        "program_mailbox_binding_proved_by_packer": False,
        "checked_atlas_source_word_match": True,
        "dram": layout,
    }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(tempfile.mkdtemp(prefix=".atlas-boot-", dir=out_dir.parent))
    try:
        (temporary / "program.bin").write_bytes(text)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                             sort_keys=True) + "\n")
        temporary.rename(out_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def pack_direct(source_path: pathlib.Path, atlas_emit: pathlib.Path,
                layout_path: pathlib.Path, out_dir: pathlib.Path,
                import_manifest_path: pathlib.Path | None = None) -> dict:
    """Package the checked physical stream emitted from machine Atlas MLIR."""
    if out_dir.exists():
        raise ValueError("output directory already exists; fresh package required")
    source = source_path.read_bytes()
    layout = validate_layout(json.loads(layout_path.read_text()))
    emitted = subprocess.run([str(atlas_emit), str(source_path)],
                             capture_output=True, text=True, check=True)
    lines = emitted.stdout.splitlines()
    if not lines or any(not re.fullmatch(r"[0-9a-fA-F]{8}", line) for line in lines):
        raise ValueError("atlas-emit did not produce complete 32-bit words")
    words = tuple(int(line, 16) for line in lines)
    if (len(words) * 4 > IMEM_SIZE or any(word & 3 != 3 for word in words)
            or words[-1] != ECALL or ECALL in words[:-1]):
        raise ValueError("direct program must fit IMEM and end at one ECALL")
    physical = subprocess.run([str(atlas_emit), "--program-json", str(source_path)],
                              capture_output=True, text=True, check=True)
    program = json.loads(physical.stdout)
    if (program.get("schema") != "atlas.physical_program.v1"
            or program.get("selected_rtl_revision") != layout["selected_rtl_revision"]
            or program.get("word_count") != len(words)
            or [row.get("word_u32") for row in program.get("instructions", [])] != list(words)):
        raise ValueError("physical program, emitted words, and selected RTL disagree")
    binary = struct.pack(f"<{len(words)}I", *words)
    captured_count = None
    if import_manifest_path is not None:
        captured_count = validate_captured_layout(
            json.loads(import_manifest_path.read_text()), layout,
            hashlib.sha256(binary).hexdigest())
    call = layout.get("call")
    if call is None:
        arguments, returns = [], []
        scope = "selected standalone AtlasCore direct reset entry"
    else:
        arguments = [{"name": "input", "kind": "runtime_dram_pointer",
                      "mailbox_offset_bytes": call["input_pointer_offset_bytes"],
                      "region": call["input_region"], "size_bytes": call["tensor_bytes"]}]
        returns = [{"name": "output", "kind": "runtime_dram_pointer",
                    "mailbox_offset_bytes": call["output_pointer_offset_bytes"],
                    "region": call["output_region"], "size_bytes": call["tensor_bytes"]}]
        scope = "selected standalone AtlasCore direct mailbox reset entry"
    manifest = {
        "schema": "atlas.boot-capsule.v1", "scope": scope,
        "packer_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "source_object_sha256": None,
        "source_mlir_sha256": hashlib.sha256(source).hexdigest(),
        "emitter_sha256": hashlib.sha256(atlas_emit.read_bytes()).hexdigest(),
        "program_sha256": hashlib.sha256(binary).hexdigest(),
        "program_file": "program.bin", "entry_symbol": None,
        "entry_pc_word": 0, "imem_tl_byte_base": IMEM_BASE,
        "imem_capacity_bytes": IMEM_SIZE, "program_words": len(words),
        "start_csr_tl_byte_address": START_CSR, "start_csr_value": 1,
        "completion": {"kind": "ecall_halt", "ecall_pc_word": len(words) - 1,
                       "unreachable_llvm_ret_pc_word": None},
        "arguments": arguments, "returns": returns,
        "register_initialization_proved": False,
        "program_mailbox_binding_proved_by_packer": False,
        "checked_atlas_source_word_match": True,
        "physical_program_file": "physical-program.json",
        "dram": layout,
    }
    if import_manifest_path is not None:
        manifest["captured_import_manifest_sha256"] = hashlib.sha256(
            import_manifest_path.read_bytes()).hexdigest()
        manifest["captured_output_count"] = captured_count
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(tempfile.mkdtemp(prefix=".atlas-boot-", dir=out_dir.parent))
    try:
        (temporary / "program.bin").write_bytes(binary)
        (temporary / "physical-program.json").write_text(
            json.dumps(program, indent=2, sort_keys=True) + "\n")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.rename(out_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", type=pathlib.Path,
                        help="legacy ELF object; omit for direct Atlas emission")
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--atlas-emit", type=pathlib.Path, required=True)
    parser.add_argument("--layout", type=pathlib.Path, required=True)
    parser.add_argument("--import-manifest", type=pathlib.Path,
                        help="checked captured-Linalg physical slots for a direct capsule")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--llvm-bin", type=pathlib.Path)
    args = parser.parse_args()
    try:
        if args.object is None:
            result = pack_direct(args.source, args.atlas_emit, args.layout, args.out,
                                 args.import_manifest)
        else:
            if args.import_manifest is not None:
                raise ValueError("captured import binding requires direct Atlas emission")
            result = pack(args.object, args.source, args.atlas_emit, args.layout,
                          args.out, args.llvm_bin)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError,
            subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "out": str(args.out),
                      "program_words": result["program_words"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

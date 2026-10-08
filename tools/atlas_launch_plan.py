#!/usr/bin/env python3
"""Prepare one bounded AtlasCore mailbox launch from a checked boot capsule.

The output is a loader handoff, not execution or a C function call. A driver
must still qualify the selected RTL, program-to-mailbox binding, and memory
visibility before issuing the CSR start write.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
import json
import pathlib
import re
import shutil
import struct
import sys
import tempfile


def _boot_packer():
    path = pathlib.Path(__file__).with_name("atlas-boot-pack")
    if not path.is_file():
        path = pathlib.Path(__file__).with_name("atlas_boot_pack.py")
    if not path.is_file():
        raise ValueError("required atlas-boot-pack is unavailable")
    # CMake installs the sibling packer without a .py suffix.
    spec = spec_from_loader("atlas_boot_pack", SourceFileLoader("atlas_boot_pack", str(path)))
    if spec is None or spec.loader is None:
        raise ValueError("cannot load required atlas-boot-pack")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_launch(capsule: pathlib.Path, input_address: int,
                   output_address: int, expected_rtl_revision: str) -> tuple[dict, bytes]:
    boot = _boot_packer()
    manifest_bytes = (capsule / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "atlas.boot-capsule.v1":
        raise ValueError("unsupported boot capsule schema")
    if manifest.get("checked_atlas_source_word_match") is not True:
        raise ValueError("capsule lacks checked Atlas source/object word match")
    if manifest.get("program_mailbox_binding_proved_by_packer") is not False:
        raise ValueError("unexpected mailbox binding claim in capsule")
    layout = boot.validate_layout(manifest["dram"])
    if layout != manifest["dram"] or layout["schema"] != "atlas.reset-entry-layout.v2":
        raise ValueError("launch requires a normalized v2 mailbox layout")
    if layout["selected_rtl_revision"] != expected_rtl_revision:
        raise ValueError("selected RTL revision differs from requested revision")
    if (manifest.get("program_file") != "program.bin" or
            manifest.get("entry_symbol") != "atlas_program" or
            manifest.get("entry_pc_word") != 0 or
            manifest.get("imem_tl_byte_base") != boot.IMEM_BASE or
            manifest.get("imem_capacity_bytes") != boot.IMEM_SIZE or
            manifest.get("start_csr_tl_byte_address") != boot.START_CSR or
            manifest.get("start_csr_value") != 1):
        raise ValueError("unsupported reset-entry hardware binding")
    code = (capsule / "program.bin").read_bytes()
    count = manifest.get("program_words")
    if (type(count) is not int or count < 2 or len(code) != 4 * count or
            len(code) > boot.IMEM_SIZE):
        raise ValueError("program size differs from capsule manifest")
    if hashlib.sha256(code).hexdigest() != manifest.get("program_sha256"):
        raise ValueError("program hash differs from capsule manifest")
    words = struct.unpack(f"<{count}I", code)
    if (any(word & 3 != 3 for word in words) or
            words[-2:] != (boot.ECALL, boot.RET) or boot.ECALL in words[:-2] or
            manifest.get("completion") != {
                "kind": "ecall_halt", "ecall_pc_word": count - 2,
                "unreachable_llvm_ret_pc_word": count - 1}):
        raise ValueError("program completion differs from checked reset-entry form")
    call = layout["call"]
    expected_args = [{"name": "input", "kind": "runtime_dram_pointer",
                      "mailbox_offset_bytes": call["input_pointer_offset_bytes"],
                      "region": call["input_region"], "size_bytes": call["tensor_bytes"]}]
    expected_returns = [{"name": "output", "kind": "runtime_dram_pointer",
                         "mailbox_offset_bytes": call["output_pointer_offset_bytes"],
                         "region": call["output_region"], "size_bytes": call["tensor_bytes"]}]
    if manifest.get("arguments") != expected_args or manifest.get("returns") != expected_returns:
        raise ValueError("manifest arguments/results differ from mailbox layout")
    descriptor = boot.mailbox_descriptor(layout, input_address, output_address)
    mailbox = next(row for row in layout["regions"] if row["name"] == call["mailbox_region"])
    plan = {
        "schema": "atlas.mailbox-launch.v1",
        "scope": "selected standalone AtlasCore reset-entry mailbox call",
        "selected_rtl_revision": expected_rtl_revision,
        "capsule_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "program_file": "program.bin",
        "program_sha256": manifest["program_sha256"],
        "program_words": count,
        "imem_tl_byte_base": boot.IMEM_BASE,
        "entry_pc_word": 0,
        "mailbox_tl_byte_address": mailbox["address"],
        "mailbox_file": "mailbox.bin",
        "mailbox_sha256": hashlib.sha256(descriptor).hexdigest(),
        "input_address": input_address,
        "output_address": output_address,
        "tensor_bytes": call["tensor_bytes"],
        "start_csr_tl_byte_address": boot.START_CSR,
        "start_csr_value": 1,
        "completion": manifest["completion"],
        "program_mailbox_binding_proved": False,
    }
    return plan, descriptor


def read_launch(capsule: pathlib.Path, launch: pathlib.Path,
                expected_rtl_revision: str) -> tuple[dict, bytes, bytes]:
    """Recheck the handoff immediately before an external driver loads it."""
    plan = json.loads((launch / "launch.json").read_text())
    if not isinstance(plan, dict) or plan.get("schema") != "atlas.mailbox-launch.v1":
        raise ValueError("unsupported mailbox launch schema")
    expected, descriptor = prepare_launch(
        capsule, plan["input_address"], plan["output_address"], expected_rtl_revision)
    if plan != expected:
        raise ValueError("launch plan differs from checked capsule and invocation")
    supplied = (launch / "mailbox.bin").read_bytes()
    if supplied != descriptor:
        raise ValueError("mailbox bytes differ from checked launch plan")
    code = (capsule / "program.bin").read_bytes()
    if hashlib.sha256(code).hexdigest() != plan["program_sha256"]:
        raise ValueError("program changed after launch-plan validation")
    return plan, code, supplied


def _address(value: str) -> int:
    if not re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)", value):
        raise argparse.ArgumentTypeError("address must be an unsigned integer")
    return int(value, 0) if value.lower().startswith("0x") else int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=pathlib.Path, required=True)
    parser.add_argument("--input-address", type=_address)
    parser.add_argument("--output-address", type=_address)
    parser.add_argument("--expected-rtl-revision", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=pathlib.Path)
    mode.add_argument("--verify-launch", type=pathlib.Path)
    args = parser.parse_args()
    try:
        if args.verify_launch is not None:
            if args.input_address is not None or args.output_address is not None:
                raise ValueError("verification reads pointer arguments from launch.json")
            plan, _, _ = read_launch(args.capsule, args.verify_launch,
                                     args.expected_rtl_revision)
            print(json.dumps({"status": "PASS", "launch": str(args.verify_launch),
                              "program_sha256": plan["program_sha256"],
                              "mailbox_sha256": plan["mailbox_sha256"]}, sort_keys=True))
            return 0
        if args.input_address is None or args.output_address is None:
            raise ValueError("preparation requires input and output addresses")
        if args.out.exists():
            raise ValueError("output directory already exists; fresh launch required")
        plan, descriptor = prepare_launch(args.capsule, args.input_address,
                                          args.output_address, args.expected_rtl_revision)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = pathlib.Path(tempfile.mkdtemp(prefix=".atlas-launch-", dir=args.out.parent))
        try:
            (temporary / "mailbox.bin").write_bytes(descriptor)
            (temporary / "launch.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
            temporary.rename(args.out)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

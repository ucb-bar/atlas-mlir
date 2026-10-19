"""Audit emitted words against the selected Atlas decoder inventory.

This checks encoding and stream identity. A BitPat match does not establish
instruction legality, numerical behavior, timing, or hardware execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys
from collections import Counter

if __package__:
    from .check_variant_inventory import INVENTORY, validate_rows
else:
    from check_variant_inventory import INVENTORY, validate_rows


def _mask(pattern: str) -> tuple[int, int]:
    if len(pattern) != 32 or set(pattern) - {"0", "1", "?"}:
        raise ValueError("invalid 32-bit decoder pattern")
    mask = int("".join("0" if bit == "?" else "1" for bit in pattern), 2)
    value = int(pattern.replace("?", "0"), 2)
    return mask, value


def _signed(value: int, width: int) -> int:
    sign = 1 << (width - 1)
    return (value ^ sign) - sign


def _operands(operation: str, word: int, fields: dict) -> dict[str, int]:
    """Extract operand bits independently of AtlasEncoding.cpp's word builder."""
    rd = (word >> 7) & 31
    rs1 = (word >> 15) & 31
    rs2 = (word >> 20) & 31
    imm_i = _signed(word >> 20, 12)
    vd = (word >> 7) & 63
    vs1 = (word >> 13) & 63
    vs2 = (word >> 19) & 63
    if operation in ("atlas.vload", "atlas.vstore"):
        result = {"base": rs1, "offset": imm_i}
        result["dst" if operation == "atlas.vload" else "src"] = vd
        return result
    if operation == "atlas.dma":
        store = fields["direction"] == "store"
        return {"channel": (word >> 12) & 7, "size": (word >> 20) & 31,
                "dram": rd if store else rs1, "reg": rs1 if store else rd}
    if operation == "atlas.dma_config":
        return {"base_reg": rs1, "channel": (word >> 12) & 7}
    if operation == "atlas.dma_wait":
        return {"channel": (word >> 12) & 7}
    if operation == "atlas.mxu_push":
        return {"slot": vd, "src": vs1}
    if operation == "atlas.mxu_matmul":
        return {"acc_slot": vd, "src": vs1, "weight_slot": vs2}
    if operation == "atlas.mxu_pop":
        return {"dst": vd, "scale_reg": vs1, "slot": vs2}
    if operation == "atlas.vpu_binary":
        return {"dst": vd, "lhs": vs1, "rhs": vs2}
    if operation == "atlas.vpu_unary":
        return {"dst": vd, "src": vs1}
    if operation == "atlas.vpu_pack":
        return {"dst": vd, "scale_reg": vs1, "src": vs2}
    if operation == "atlas.vpu_reduce":
        return {"dst": vd, "src": vs1}
    if operation == "atlas.vli":
        return {"dst": vd, "immediate": (word >> 16) & 0xffff}
    if operation == "atlas.xlu_transpose":
        return {"dst": vd, "src": vs1}
    if operation == "atlas.alu_reg":
        return {"dst": rd, "lhs": rs1, "rhs": rs2}
    if operation == "atlas.alu_imm":
        immediate = ((word >> 20) & 31) if fields["kind"] in ("slli", "srli", "srai") else imm_i
        return {"dst": rd, "src": rs1, "immediate": immediate}
    if operation == "atlas.branch":
        imm = (((word >> 31) & 1) << 12) | (((word >> 7) & 1) << 11)
        imm |= (((word >> 25) & 63) << 5) | (((word >> 8) & 15) << 1)
        return {"lhs": rs1, "rhs": rs2, "offset_bytes": _signed(imm, 13)}
    if operation == "atlas.jump":
        if fields["kind"] == "jal":
            imm = (((word >> 31) & 1) << 20) | (((word >> 12) & 255) << 12)
            imm |= (((word >> 20) & 1) << 11) | (((word >> 21) & 1023) << 1)
            return {"dst": rd, "offset": _signed(imm, 21)}
        return {"dst": rd, "base": rs1, "offset": imm_i}
    if operation == "atlas.delay":
        return {"cycles": (word >> 20) & 0xfff}
    if operation == "atlas.upper":
        return {"dst": rd, "immediate": word >> 12}
    if operation == "atlas.csr":
        return {"dst": rd, "source": rs1, "address": word >> 20}
    if operation == "atlas.scalar_load":
        return {"dst": rd, "base": rs1, "offset": imm_i}
    if operation == "atlas.scalar_store":
        imm = ((word >> 25) << 5) | ((word >> 7) & 31)
        return {"src": rs2, "base": rs1, "offset": _signed(imm, 12)}
    if operation in ("atlas.trap", "atlas.fence"):
        return {}
    raise ValueError(f"no independent operand decoder for {operation}")


def decode_binary(inventory: dict, binary: bytes) -> dict:
    """Recover operation modes and operands from emitted words alone.

    The selected BitPats identify the mode; mode attributes and bit fields
    provide the operands. Source annotations and scheduling reasons cannot be
    recovered from an instruction word. This is a decoder check, not execution
    or proof that every reserved-field combination is legal.
    """
    if len(binary) % 4:
        raise ValueError("program binary has a partial instruction word")
    patterns = [(row, *_mask(row["rtl_bitpat"])) for row in inventory["variants"]]
    rows = []
    for index, (word,) in enumerate(struct.iter_unpack("<I", binary)):
        matches = [row for row, mask, value in patterns if word & mask == value]
        if len(matches) != 1:
            raise ValueError(f"word {index}: expected one selected RTL decode mode, found {len(matches)}")
        mode = matches[0]
        fields = dict(mode["mode_attrs"])
        operands = _operands(mode["dialect_op"], word, fields)
        if any(fields[key] != value for key, value in operands.items() if key in fields):
            raise ValueError(f"word {index}: encoded operands disagree with selected mode")
        fields.update(operands)
        rows.append({
            "word_index": index,
            "word_u32": word,
            "word_hex": f"{word:08x}",
            "operation": mode["dialect_op"],
            "mode_id": mode["id"],
            "fields": fields,
        })
    return {
        "schema": "atlas.selected_binary_decode.v1",
        "selected_rtl_revision": inventory["selected_sources"]["rtl_revision"],
        "binary_sha256": hashlib.sha256(binary).hexdigest(),
        "word_count": len(rows),
        "instructions": rows,
    }


def audit_program(inventory: dict, program: dict, binary: bytes) -> dict:
    """Decode each binary word independently of the emitter's word builder."""
    if len(binary) % 4:
        raise ValueError("program binary has a partial instruction word")
    rows = program["instructions"]
    count = len(binary) // 4
    if program["word_count"] != count or len(rows) != count:
        raise ValueError("program word count differs from binary or stream")
    patterns = [(row, *_mask(row["rtl_bitpat"])) for row in inventory["variants"]]
    seen = Counter()
    for index, (word,) in enumerate(struct.iter_unpack("<I", binary)):
        selected = rows[index]
        if selected["word_index"] != index or selected["word_u32"] != word:
            raise ValueError(f"word {index}: binary and structured stream differ")
        if selected["word_hex"].lower() != f"{word:08x}":
            raise ValueError(f"word {index}: hexadecimal text differs from binary")
        matches = [row for row, mask, value in patterns if word & mask == value]
        if len(matches) != 1:
            raise ValueError(f"word {index}: expected one selected RTL decode mode, found {len(matches)}")
        mode = matches[0]
        if selected["operation"] != mode["dialect_op"]:
            raise ValueError(f"word {index}: {selected['operation']} decodes as {mode['id']}")
        fields = selected["fields"]
        if any(fields.get(key) != value for key, value in mode["mode_attrs"].items()):
            raise ValueError(f"word {index}: declared fields disagree with {mode['id']}")
        for key, value in _operands(selected["operation"], word, fields).items():
            if fields.get(key) != value:
                raise ValueError(f"word {index}: {key} field differs from binary operand")
        seen[mode["id"]] += 1
    return {
        "status": "PASS",
        "scope": "selected RTL BitPat, stream identity, and independently decoded operands",
        "inventory_rtl_revision": inventory["selected_sources"]["rtl_revision"],
        "word_count": count,
        "modes": dict(sorted(seen.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--inventory", type=pathlib.Path, default=INVENTORY)
    parser.add_argument("--decoded-out", type=pathlib.Path,
                        help="write a fresh instruction stream decoded only from binary words")
    parser.add_argument("--rtl-instructions", type=pathlib.Path)
    parser.add_argument("--rtl-decode", type=pathlib.Path)
    args = parser.parse_args()
    try:
        if (args.rtl_instructions is None) != (args.rtl_decode is None):
            raise ValueError("supply both selected RTL source files or neither")
        inventory = json.loads(args.inventory.read_text())
        if args.rtl_instructions is not None:
            instructions = args.rtl_instructions.read_bytes()
            decode = args.rtl_decode.read_bytes()
            validate_rows(inventory, instructions.decode(), decode.decode())
        report = audit_program(inventory, json.loads(args.program.read_text()),
                               args.binary.read_bytes())
        if args.decoded_out is not None:
            decoded = decode_binary(inventory, args.binary.read_bytes())
            args.decoded_out.parent.mkdir(parents=True, exist_ok=True)
            with args.decoded_out.open("x") as output:
                output.write(json.dumps(decoded, indent=2, sort_keys=True) + "\n")
            report["decoded_out"] = str(args.decoded_out)
        # Row equality does not establish the source checkout's Git identity.
        report["rtl_rows_matched"] = args.rtl_instructions is not None
        if args.rtl_instructions is not None:
            report["rtl_source_sha256"] = {
                "Instructions.scala": hashlib.sha256(instructions).hexdigest(),
                "IDecode.scala": hashlib.sha256(decode).hexdigest(),
            }
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

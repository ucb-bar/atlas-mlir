#!/usr/bin/env python3
"""Generates tools/rtl_extract/targets/atlas/sequences/core_mxu.yaml: same-MXU command chains on AtlasCore.

Encodings follow lib/AtlasEncoding.cpp (vr: funct7[31:25], vs2[24:19], vs1[18:13], vd[12:7], opcode 0x77).
Chain position p uses source m(4p) (pair 4p:4p+1) and pop destination m(4p+2) (pair 4p+2:4p+3), so every MREG
register of a chain lies in its own physical bank; the op cycle is unrolled to the longest chain.

Usage: gen_core_mxu_chains.py [OUTPUT_YAML [SWEEP_JSON]] [--output PATH]
Without an output path the generated YAML goes to stdout. The historical build/
generator comment is preserved so committed specs reproduce byte for byte.
"""
import argparse
import json
import re
import sys
from pathlib import Path

BASE = {"push_weight": 0, "push_acc_fp8": 2, "push_acc_bf16": 4, "pop_acc_fp8": 6, "pop_acc_bf16": 8,
        "matmul": 10, "matmul_acc": 12}
SHORT = {"pw": "push_weight", "pa8": "push_acc_fp8", "pa16": "push_acc_bf16", "pop8": "pop_acc_fp8",
         "pop16": "pop_acc_bf16", "mm": "matmul", "mma": "matmul_acc"}


def word(kind, unit, p, acc=0, slot=0):
    src, dst = 4 * p, 4 * p + 2
    if kind == "push_weight":
        vd, vs1, vs2 = slot, src, 0
    elif kind.startswith("push_acc"):
        vd, vs1, vs2 = acc, src, 0
    elif kind == "pop_acc_fp8":
        vd, vs1, vs2 = dst, 0, acc
    elif kind == "pop_acc_bf16":
        vd, vs1, vs2 = dst, 0, acc
    else:
        vd, vs1, vs2 = acc, src, slot
    return ((BASE[kind] + unit) << 25) | (vs2 << 19) | (vs1 << 13) | (vd << 7) | 0x77


# name: (description, [(short kind, acc, slot)], lengths, contains a matmul)
CHAINS = {
    "matmul_same_acc_same_slot": ("matmuls on one accumulator and weight slot", [("mm", 0, 0)], "2..4"),
    "matmul_same_acc_other_slot": ("matmuls on one accumulator, alternating weight slots", [("mm", 0, 0), ("mm", 0, 1)], "2..4"),
    "matmul_other_acc_same_slot": ("matmuls alternating accumulators on one weight slot", [("mm", 0, 0), ("mm", 1, 0)], "2..5"),
    "matmul_other_acc_other_slot": ("matmuls alternating accumulators and weight slots", [("mm", 0, 0), ("mm", 1, 1)], "2..5"),
    "matmul_acc_same_acc": ("accumulating matmuls on one accumulator", [("mma", 0, 0)], "2..4"),
    "matmul_acc_other_acc": ("accumulating matmuls alternating accumulators", [("mma", 0, 0), ("mma", 1, 1)], "2..5"),
    "matmul_then_matmul_acc_same_acc": ("matmul, then accumulating matmul into its accumulator", [("mm", 0, 0), ("mma", 0, 1)], "2..4"),
    "push_acc_matmul_acc_same_acc": ("accumulator push, then accumulating matmul on it", [("pa8", 0, 0), ("mma", 0, 0)], "2..4"),
    "push_acc_matmul_acc_other_acc": ("accumulator push and accumulating matmul, alternating accumulators",
                                      [("pa8", 0, 0), ("mma", 0, 0), ("pa8", 1, 0), ("mma", 1, 1)], "2..5"),
    "push_acc_bf16_matmul_acc": ("BF16 accumulator push, then accumulating matmul on it", [("pa16", 0, 0), ("mma", 0, 0)], "2..4"),
    "matmul_pop_same_acc": ("matmul, then BF16 pop of its accumulator", [("mm", 0, 0), ("pop16", 0, 0)], "2..4"),
    "matmul_pop_other_acc": ("matmul, then BF16 pop of the other accumulator", [("mm", 0, 0), ("pop16", 1, 0)], "2..4"),
    "matmul_pop_fp8_same_acc": ("matmul, then FP8 pop of its accumulator", [("mm", 0, 0), ("pop8", 0, 0)], "2..4"),
    "push_weight_matmul_pop_same": ("weight push, matmul, BF16 pop on one weight slot and accumulator",
                                    [("pw", 0, 0), ("mm", 0, 0), ("pop16", 0, 0)], "3..6"),
    "push_weight_matmul_pop_alternating": ("weight push, matmul, BF16 pop, alternating weight slot and accumulator",
                                           [("pw", 0, 0), ("mm", 0, 0), ("pop16", 0, 0), ("pw", 0, 1), ("mm", 1, 1), ("pop16", 1, 0)], "3..6"),
    "push_weight_matmul_other_slot": ("weight push, then matmul on the other weight slot", [("pw", 0, 0), ("mm", 0, 1)], "2..4"),
    "push_weight_same_slot": ("weight pushes on one slot", [("pw", 0, 0)], "2..4"),
    "push_weight_other_slot": ("weight pushes alternating slots", [("pw", 0, 0), ("pw", 0, 1)], "2..4"),
    "push_weight_push_acc": ("weight push, then FP8 accumulator push", [("pw", 0, 0), ("pa8", 1, 0)], "2..4"),
    "pop_fp8_pop_bf16": ("FP8 pop, then BF16 pop of the other accumulator", [("pop8", 0, 0), ("pop16", 1, 0)], "2..4"),
    "matmul_push_weight_pop": ("matmul, weight push into the other slot, pop of the other accumulator",
                               [("mm", 0, 0), ("pw", 0, 1), ("pop16", 1, 0)], "3..6"),
    "push_weight_pop_matmul": ("weight push, pop, matmul on the pushed slot and the popped accumulator",
                               [("pw", 0, 1), ("pop16", 1, 0), ("mm", 1, 1)], "3..6"),
    "matmul_push_acc_matmul_acc": ("matmul, accumulator push into the other accumulator, accumulating matmul on it",
                                   [("mm", 0, 0), ("pa8", 1, 0), ("mma", 1, 1)], "3..6"),
    "push_acc_bf16_pop_matmul_acc": ("BF16 accumulator push, FP8 pop of the other accumulator, accumulating matmul on the pushed one",
                                     [("pa16", 0, 0), ("pop8", 1, 0), ("mma", 0, 0)], "3..6"),
    "pop_push_acc_matmul": ("pop, accumulator push into the popped accumulator, matmul on the other",
                            [("pop16", 0, 0), ("pa8", 0, 0), ("mm", 1, 0)], "3..6"),
}

def generate(sweep=None):
    """Render MXU chains with optional per-length sweep regressions."""
    lines = []
    for unit in (0, 1):
        for name, (description, cycle, lengths) in CHAINS.items():
            has_matmul = any(k in ("mm", "mma") for k, _, _ in cycle)
            top = (100 if unit == 0 else 40) if has_matmul else 40
            lines.append(f"  mxu{unit}_{name}:")
            lines.append(f"    description: {description}")
            lines.append("    ops:")
            # The cycle is unrolled to the longest chain, so every chain position names its own registers.
            longest = int(lengths.split("..")[1])
            for p in range(longest):
                k, acc, slot = cycle[p % len(cycle)]
                kind = SHORT[k]
                lines.append(f"      - {{op: mxu{unit}_matmul, as: mxu{unit}.{kind}, inputs: {{instruction: 0x{word(kind, unit, p, acc, slot):08x}}}}}")
            lines.append(f"    lengths: {lengths}")
            lines.append(f"    gaps: 1..{top}")

    HEADER = """# Chains of commands on one MXU on AtlasCore through the decoder cut (targets/atlas/system/core.yaml) at uniform
# spacing; operation i is issued i * gap ages after the first. Every AtlasCore assertion is an invariant. Each entry
# overrides the instruction word of the carrier operation mxu<unit>_matmul; `as` names the op_timing block it stands
# for. Chain position p reads m(4p) (pair m(4p):m(4p+1)) and pops into m(4p+2) (pair m(4p+2):m(4p+3)), so every MREG
# register of a chain lies in its own physical bank; accumulators and weight slots repeat as the chain names say. The chains
# cover the engine-level MXU chains of sequences/mxu0.yaml and mxu1.yaml and mixes of a matmul with a push and a pop
# whose ports, accumulators and weight slots differ. Generated by build/gen_core_mxu_chains.py from the encodings of
# lib/AtlasEncoding.cpp.
engine: core
invariants: {}
description: Same-MXU command chains through the RTL decoder; all AtlasCore assertions checked.
equivalent:
  vpu_read: [vpu_read0, vpu_read1]
  vpu_write: [vpu_write0, vpu_write1]
sequences:
"""
    text = HEADER + "\n".join(lines) + "\n"
    if sweep is not None:
        records = sweep["op_sequences"]
        expected = {(f"mxu{unit}_{name}", length)
                    for unit in (0, 1)
                    for name, (_, _, lengths) in CHAINS.items()
                    for length in range(int(lengths.split("..")[0]), int(lengths.split("..")[1]) + 1)}
        actual = {(r["sequence"], r["length"]) for r in records}
        if actual != expected or len(actual) != len(records):
            raise ValueError("sweep must contain exactly every MXU chain length")
        expect = {}
        for r in records:
            expect.setdefault(r["sequence"], {})[r["length"]] = r["stable_gap"]
        text += "# Regression values used only by tests: stable spacing per chain length. The extractor never reads them.\nexpect:\n"
        for name in [l.strip()[:-1] for l in lines if l.startswith("  mxu") and l.endswith(":")]:
            text += f"  {name}: {{" + ", ".join(f"{k}: {v}" for k, v in sorted(expect[name].items())) + "}\n"
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_path", type=Path, nargs="?")
    parser.add_argument("sweep", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output_path:
        parser.error("use either the positional output path or --output")
    text = generate(json.loads(args.sweep.read_text()) if args.sweep else None)
    output = args.output or args.output_path
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

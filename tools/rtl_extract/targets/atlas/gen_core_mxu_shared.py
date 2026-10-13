#!/usr/bin/env python3
"""Generates tools/rtl_extract/targets/atlas/pairs/core_mxu_shared.yaml: every ordered pair of commands on one MXU whose
MREG operands share a register or a physical bank, for every way they can overlap.

An overlap pattern lists, for the first command's i-th MREG register and the second's j-th, "i=j" (same register) or
"i~j" (same physical bank, register mod 32, other register), joined by commas; registers are listed in operand order
(a BF16 pair as base, base+1). The compiler computes the same string (lib/AtlasRTLSelection.cpp).

Usage: gen_core_mxu_shared.py FACTS_JSON [OUTPUT_YAML [SWEEP_JSON]] [--output PATH]
Without an output path the generated YAML goes to stdout. The historical build/
generator comment is preserved so committed specs reproduce byte for byte.
"""
import argparse
import json
import re
import sys
from pathlib import Path

KINDS = {"push_weight": (0, False, True), "push_acc_fp8": (2, True, False), "push_acc_bf16": (4, True, False),
         "pop_acc_fp8": (6, True, False), "pop_acc_bf16": (8, True, False), "matmul": (10, True, True),
         "matmul_acc": (12, True, True)}
SHORT = {"push_weight": "pw", "push_acc_fp8": "pa8", "push_acc_bf16": "pa16", "pop_acc_fp8": "pop8",
         "pop_acc_bf16": "pop16", "matmul": "mm", "matmul_acc": "mma"}
PAIRED = {"push_acc_bf16", "pop_acc_bf16"}


def word(kind, unit, reg, acc, slot, scale):
    base = KINDS[kind][0] + unit
    if kind == "push_weight": vd, vs1, vs2 = slot, reg, 0
    elif kind.startswith("push_acc"): vd, vs1, vs2 = acc, reg, 0
    elif kind == "pop_acc_fp8": vd, vs1, vs2 = reg, scale, acc
    elif kind == "pop_acc_bf16": vd, vs1, vs2 = reg, 0, acc
    else: vd, vs1, vs2 = acc, reg, slot
    return (base << 25) | (vs2 << 19) | (vs1 << 13) | (vd << 7) | 0x77


def registers(kind, reg): return [reg, reg + 1] if kind in PAIRED else [reg]


def pattern(first, second):
    parts = [f"{i}={j}" if a == b else f"{i}~{j}" for i, a in enumerate(first) for j, b in enumerate(second) if a % 32 == b % 32]
    return ",".join(parts) or "disjoint"


def placements(a, b):
    """(first register, second register, pattern) for every way the two commands' operands can share a bank."""
    out = {}
    for first in (12, 13):
        if a in PAIRED and first % 2: continue
        for second in range(0, 64):
            if b in PAIRED and second % 2: continue
            p = pattern(registers(a, first), registers(b, second))
            if p != "disjoint" and p not in out:
                out[p] = (first, second)
    return out


def generate(facts, sweep=None):
    """Render ordered MXU pairs from event ages and optional sweep regressions."""
    last = {b["name"]: max(g["last_age"] for g in b["events"].values() if g["count"]) for b in facts["op_timing"]
            if b["engine"] in ("mxu0", "mxu1")}
    expected = {f"mxu{unit}.{kind}" for unit in (0, 1) for kind in KINDS}
    if not expected <= set(last):
        raise ValueError("facts must contain all 14 MXU operations")

    lines, count = [], 0
    for unit in (0, 1):
        for a in KINDS:
            for b in KINDS:
                accs = ["same", "other"] if KINDS[a][1] and KINDS[b][1] else [None]
                slots = ["same", "other"] if KINDS[a][2] and KINDS[b][2] else [None]
                for p, (r1, r2) in sorted(placements(a, b).items()):
                    tag = p.replace(",", "_").replace("=", "r").replace("~", "b")
                    for acc in accs:
                        for slot in slots:
                            name = f"mxu{unit}_{SHORT[a]}_{SHORT[b]}_{tag}" + (f"_acc_{acc}" if acc else "") + (f"_slot_{slot}" if slot else "")
                            relation = {"mreg_banks": p, **({"accumulator": acc} if acc else {}), **({"weight_slot": slot} if slot else {})}
                            w1 = word(a, unit, r1, 0, 0, 0)
                            w2 = word(b, unit, r2, 0 if acc in (None, "same") else 1, 0 if slot in (None, "same") else 1, 1)
                            lines += [f"  {name}:", f"    as: {{first: mxu{unit}.{a}, second: mxu{unit}.{b}}}",
                                      "    relation: {" + ", ".join(f'{k}: "{v}"' for k, v in relation.items()) + "}",
                                      f"    first: mxu{unit}_matmul", f"    first_inputs: {{instruction: 0x{w1:08x}}}",
                                      f"    second: mxu{unit}_matmul", f"    second_inputs: {{instruction: 0x{w2:08x}}}",
                                      f"    gaps: 1..{last[f'mxu{unit}.{a}'] + 6}"]
                            count += 1

    HEADER = f"""# Every ordered pair of commands on one MXU whose MREG operands share a register or a physical bank (register mod 32),
# on AtlasCore through the decoder cut (targets/atlas/system/core.yaml), for every way their operands can overlap and
# every accumulator and weight-slot relation ({count} pairs). `relation.mreg_banks` names the overlap: for the first
# command's i-th MREG register and the second's j-th (operand order; a BF16 pair as base, base+1), "i=j" is the same
# register and "i~j" the same physical bank under another register. The first command reads or writes m12 (m12:13), or
# m13 where only an odd register gives the overlap; the second takes the register that gives it. Every AtlasCore
# assertion is an invariant, so the MREG bank tracker, MregFile's per-bank ports and the MXU sequencers' register
# checks decide beside the issue checks of pairs/core_mxu_same.yaml. Both sides override the instruction word of the
# carrier mxu<unit>_matmul; gaps run to the first command's last event age plus 6. Generated by
# build/gen_core_mxu_shared.py from the encodings of lib/AtlasEncoding.cpp.
engine: core
description: Same-MXU command pairs with shared MREG registers or banks, second issued g ages after the first; all AtlasCore assertions checked.
equivalent:
  vpu_read: [vpu_read0, vpu_read1]
  vpu_write: [vpu_write0, vpu_write1]
shared:
  - {{input: instruction, reason: one instruction per age through the decoder; commands never share an age}}
  - {{input: fire, reason: issue strobe of that instruction}}
  - {{input: address, reason: VLS address operand of that instruction}}
  - {{path: mreg, reason: MREG file; per-bank port conflicts are MregFile invariants}}
  - {{path: vmem, reason: VMEM banks; same-bank conflicts are Vmem invariants}}
invariants: {{}}
pairs:
"""
    text = HEADER + "\n".join(lines) + "\n"
    if sweep is not None:
        records = {r["name"].split(".", 1)[1]: r for r in sweep["op_pairs"]}
        names = [l.strip()[:-1] for l in lines if l.startswith("  ") and not l.startswith("    ")]
        if set(records) != set(names) or len(records) != len(sweep["op_pairs"]):
            raise ValueError("sweep must contain exactly every shared-MXU pair")
        text += ("# Regression values used only by tests (full sweeps); the extractor never reads them. rejected_by matches the result\n"
                 "# at stable_gap - 1 (the first failed part).\nexpect:\n")
        for name in [l.strip()[:-1] for l in lines if l.startswith("  ") and not l.startswith("    ")]:
            r = records[name]
            entry = f"min_gap: {r['min_gap']}, stable_gap: {r['stable_gap']}"
            if r["stable_gap"] > 1:
                below = next(x["result"] for x in r["results"] if x["from"] <= r["stable_gap"] - 1 <= x["to"])
                first = below.split("; ")[0]
                entry += ', rejected_by: "' + re.escape(first).replace("\\ ", " ").replace('"', '\\"').replace("\\", "\\\\") + '"'
            text += f"  {name}: {{{entry}}}\n"
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("facts", type=Path)
    parser.add_argument("output_path", type=Path, nargs="?")
    parser.add_argument("sweep", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output_path:
        parser.error("use either the positional output path or --output")
    text = generate(json.loads(args.facts.read_text()),
                    json.loads(args.sweep.read_text()) if args.sweep else None)
    output = args.output or args.output_path
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate AtlasCore sweep and system specs; expectations are pinned regressions."""
import argparse
import copy
import json
from pathlib import Path
import re
import sys

import yaml

SPECS = Path(__file__).resolve().parent
KINDS = {"push_weight": (0, False, True), "push_acc_fp8": (2, True, False),
    "push_acc_bf16": (4, True, False), "pop_acc_fp8": (6, True, False),
    "pop_acc_bf16": (8, True, False), "matmul": (10, True, True), "matmul_acc": (12, True, True),}
SHORT = {"push_weight": "pw", "push_acc_fp8": "pa8", "push_acc_bf16": "pa16",
         "pop_acc_fp8": "pop8", "pop_acc_bf16": "pop16", "matmul": "mm", "matmul_acc": "mma"}
PAIRED = {"push_acc_bf16", "pop_acc_bf16"}


def word(kind, unit, reg, acc=0, slot=0, scale=0):
    if kind == "push_weight":
        vd, vs1, vs2 = slot, reg, 0
    elif kind.startswith("push_acc"):
        vd, vs1, vs2 = acc, reg, 0
    elif kind == "pop_acc_fp8":
        vd, vs1, vs2 = reg, scale, acc
    elif kind == "pop_acc_bf16":
        vd, vs1, vs2 = reg, 0, acc
    else:
        vd, vs1, vs2 = acc, reg, slot
    return ((KINDS[kind][0] + unit) << 25) | (vs2 << 19) | (vs1 << 13) | (vd << 7) | 0x77


def registers(kind, reg):
    return [reg, reg + 1] if kind in PAIRED else [reg]


def pattern(first, second):
    parts = [f"{i}={j}" if a == b else f"{i}~{j}" for i, a in enumerate(first) for j, b in enumerate(second) if a % 32 == b % 32]
    return ",".join(parts) or "disjoint"


def placements(a, b):
    """(first register, second register, pattern) for every way the two commands' operands can share a bank."""
    out = {}
    for first in (12, 13):
        if a in PAIRED and first % 2:
            continue
        for second in range(64):
            if b in PAIRED and second % 2:
                continue
            p = pattern(registers(a, first), registers(b, second))
            if p != "disjoint" and p not in out:
                out[p] = (first, second)
    return out


def pair_regressions(names, sweep, kind):
    records = {r["name"].split(".", 1)[1]: r for r in sweep["op_pairs"]}
    if set(records) != set(names) or len(records) != len(sweep["op_pairs"]):
        raise ValueError(f"sweep must contain exactly every {kind} pair")
    expected = {}
    for name in names:
        r = records[name]
        expected[name] = {"min_gap": r["min_gap"], "stable_gap": r["stable_gap"]}
        if r["stable_gap"] > 1:
            below = next(x["result"] for x in r["results"] if x["from"] <= r["stable_gap"] - 1 <= x["to"])
            expected[name]["rejected_by"] = re.escape(below.split("; ")[0]).replace("\\ ", " ")
    return expected


def mxu_pairs(facts, sweep=None, shared=False):
    last = {b["name"]: max(g["last_age"] for g in b["events"].values() if g["count"])
            for b in facts["op_timing"] if b["engine"] in ("mxu0", "mxu1")}
    if not {f"mxu{unit}.{kind}" for unit in (0, 1) for kind in KINDS} <= set(last):
        raise ValueError("facts must contain all 14 MXU operations")
    raw = copy.deepcopy(HEADERS["pairs/core_mxu_shared" if shared else "pairs/core_mxu_same"])
    raw["pairs"] = pairs = {}
    for unit in (0, 1):
        for a in KINDS:
            for b in KINDS:
                accs = ["same", "other"] if KINDS[a][1] and KINDS[b][1] else [None]
                slots = ["same", "other"] if KINDS[a][2] and KINDS[b][2] else [None]
                locations = sorted(placements(a, b).items()) if shared else [(None, (12, 16))]
                for p, (r1, r2) in locations:
                    tag = "_" + p.replace(",", "_").replace("=", "r").replace("~", "b") if shared else ""
                    for acc in accs:
                        for slot in slots:
                            name = f"mxu{unit}_{SHORT[a]}_{SHORT[b]}{tag}" + (f"_acc_{acc}" if acc else "") + (f"_slot_{slot}" if slot else "")
                            relation = {**({"mreg_banks": p} if shared else {}),
                                        **({"accumulator": acc} if acc else {}), **({"weight_slot": slot} if slot else {})}
                            pairs[name] = {"as": {"first": f"mxu{unit}.{a}", "second": f"mxu{unit}.{b}"},
                                **({"relation": relation} if relation else {}), "first": f"mxu{unit}_matmul",
                                "first_inputs": {"instruction": word(a, unit, r1 if shared or not a.startswith("pop") else 20)},
                                "second": f"mxu{unit}_matmul", "second_inputs": {"instruction": word(b, unit,
                                    r2 if shared or not b.startswith("pop") else 24, acc=int(acc == "other"), slot=int(slot == "other"), scale=1)},
                                "gaps": f"1..{last[f'mxu{unit}.{a}'] + 6}"}
    if sweep is not None:
        raw["expect"] = pair_regressions(pairs, sweep, "shared-MXU" if shared else "same-MXU")
    return yaml.safe_dump(raw, sort_keys=False)


def vpu_pairs(facts, sweep=None, spec_path=SPECS / "vpu.yaml"):
    words = yaml.safe_load(spec_path.read_text())["decode"]["words"]
    nia = {r["operation"]: r["next_issue_age"] for r in facts["op_timing"] if r["engine"] == "vpu"}
    if set(words) != set(nia) or len(words) != 29:
        raise ValueError("facts must contain exactly the 29 VPU operations in vpu.yaml")
    def instruction(op, vs1, vs2, vd):
        value = words[op] | vd << 7
        return value if op in {"vliAll", "vliRow", "vliCol", "vliOne"} else value | vs1 << 13 | vs2 << 19
    raw = copy.deepcopy(HEADERS["pairs/core_vpu"])
    raw["pairs"] = {f"{a}_{b}": {"as": {"first": f"vpu.{a}", "second": f"vpu.{b}"},
        "first": "add", "first_inputs": {"instruction": instruction(a, 8, 12, 16)},
        "second": "add_hi", "second_inputs": {"instruction": instruction(b, 20, 24, 28)}, "gaps": f"1..{nia[a] + 6}"}
        for a in words for b in words}
    if sweep is not None:
        records = {r["name"].split(".", 1)[1]: r for r in sweep["op_pairs"]}
        if set(records) != set(raw["pairs"]):
            raise ValueError("sweep must contain exactly every ordered VPU pair")
        raw["expect"] = {f"{a}_{b}": {"min_gap": records[f"{a}_{b}"]["min_gap"],
                         "stable_gap": records[f"{a}_{b}"]["stable_gap"], "next_issue_age": nia[a]}
                         for a in words for b in words}
    return yaml.safe_dump(raw, sort_keys=False)


CHAINS = {"matmul_same_acc_same_slot": ("matmuls on one accumulator and weight slot", [("mm", 0, 0)], "2..4"),
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
                            [("pop16", 0, 0), ("pa8", 0, 0), ("mm", 1, 0)], "3..6"),}


def mxu_chains(sweep=None):
    raw = copy.deepcopy(HEADERS["sequences/core_mxu"])
    raw["sequences"] = sequences = {}
    for unit in (0, 1):
        for name, (description, cycle, lengths) in CHAINS.items():
            has_matmul = any(k in ("mm", "mma") for k, _, _ in cycle)
            top = (100 if unit == 0 else 40) if has_matmul else 40
            ops = []
            for p in range(int(lengths.split("..")[1])):
                k, acc, slot = cycle[p % len(cycle)]
                kind = next(kind for kind, short in SHORT.items() if short == k)
                ops.append({"op": f"mxu{unit}_matmul", "as": f"mxu{unit}.{kind}", "inputs": {
                    "instruction": word(kind, unit, 4*p + (2 if kind.startswith("pop") else 0), acc, slot)}})
            sequences[f"mxu{unit}_{name}"] = {"description": description, "ops": ops, "lengths": lengths, "gaps": f"1..{top}"}
    if sweep is not None:
        records = sweep["op_sequences"]
        expected = {(name, length) for name, seq in sequences.items()
                    for length in range(int(seq["lengths"].split("..")[0]), int(seq["lengths"].split("..")[1]) + 1)}
        if {(r["sequence"], r["length"]) for r in records} != expected or len(expected) != len(records):
            raise ValueError("sweep must contain exactly every MXU chain length")
        raw["expect"] = {name: {r["length"]: r["stable_gap"] for r in records if r["sequence"] == name} for name in sequences}
    return yaml.safe_dump(raw, sort_keys=False)


GENERATED = ("pairs/core_vpu.yaml", "pairs/core_mxu_same.yaml", "pairs/core_mxu_shared.yaml", "sequences/core_mxu.yaml",
             "pairs/core_mxu_cross.yaml", "pairs/core_mxu_units.yaml", "pairs/core_mxu.yaml", "pairs/core_dma_mxu.yaml",
             "pairs/core_dma.yaml", "pairs/core.yaml", "sequences/core_vpu.yaml", "system/core.yaml", "system/core_dma.yaml")

EQUIVALENT = {'vpu_read': ['vpu_read0', 'vpu_read1'], 'vpu_write': ['vpu_write0', 'vpu_write1']}
SHARED = [
    {'input': 'instruction', 'reason': 'one instruction per age through the decoder; commands never share an age'},
    {'input': 'fire', 'reason': 'issue strobe of that instruction'},
    {'input': 'address', 'reason': 'VLS address operand of that instruction'},
    {'path': 'mreg', 'reason': 'MREG file; per-bank port conflicts are MregFile invariants'},
    {'path': 'vmem', 'reason': 'VMEM banks; same-bank conflicts are Vmem invariants'},]
HEADERS = {'pairs/core_vpu': {'engine': 'core',
        'description': 'Every ordered VPU operation pair, second issued g ages after the first; all AtlasCore assertions checked.',
        'equivalent': EQUIVALENT, 'shared': SHARED, 'invariants': {}, 'relation': {'mreg_banks': 'disjoint'},},
    'pairs/core_mxu_same': {'engine': 'core',
        'description': 'Every ordered same-MXU command pair, second issued g ages after the first; all AtlasCore assertions checked.',
        'equivalent': EQUIVALENT, 'shared': SHARED, 'invariants': {}, 'relation': {'mreg_banks': 'disjoint'},},
    'pairs/core_mxu_shared': {'engine': 'core',
        'description': 'Same-MXU command pairs with shared MREG registers or banks, second issued g ages after the first; all AtlasCore assertions checked.',
        'equivalent': EQUIVALENT, 'shared': SHARED, 'invariants': {},}, 'sequences/core_mxu': {'engine': 'core',
        'invariants': {},
        'description': 'Same-MXU command chains through the RTL decoder; all AtlasCore assertions checked.',
        'equivalent': EQUIVALENT,}, 'pairs/core_mxu_cross': {'engine': 'core',
        'description': 'MXU command and a VPU, VLS or XLU command g ages apart through the RTL decoder; all AtlasCore assertions checked.',
        'equivalent': EQUIVALENT, 'shared': SHARED,}, 'pairs/core_mxu_units': {'engine': 'core',
        'description': 'Command on one MXU and a command on the other g ages later through the RTL decoder; all AtlasCore assertions checked.',
        'equivalent': EQUIVALENT, 'shared': SHARED,}, 'pairs/core_mxu': {'engine': 'core',
        'description': 'Second instruction issued g ages after the first through the RTL decoder; MXU commands against each other and against VPU, VLS and XLU commands; all AtlasCore assertions checked.',
        'equivalent': EQUIVALENT, 'shared': SHARED,}, 'pairs/core_dma_mxu': {'engine': 'core_dma',
        'description': 'MXU command g ages after a DMA launch, and a DMA launch g ages after an MXU command; all AtlasCore assertions checked.',
        'shared': [{'input': 'instruction', 'reason': 'one instruction per age through the decoder; commands never share an age'}, {'input': 'fire', 'reason': 'issue strobe of that instruction'}, {'input': 'dma_size', 'reason': 'DMA size operand of that instruction, captured at launch'}, {'input': 'dma_vmem_addr', 'reason': 'DMA VMEM operand of that instruction, captured at launch'}, {'input': 'dma_dram_addr', 'reason': 'DMA DRAM operand of that instruction, captured at launch'}, {'path': 'mreg', 'reason': 'MREG file; per-bank port conflicts are MregFile invariants'}, {'path': 'vmem', 'reason': 'VMEM arbiter; DMA grants are the dma_vmem events'}],
    }, 'pairs/core_dma': {'engine': 'core_dma',
        'description': 'Second operation issued g ages after a DMA launch (or a DMA launch after an engine command); all AtlasCore assertions checked.',
        'shared': [{'input': 'instruction', 'reason': 'one instruction per age through the decoder; commands never share an age'}, {'input': 'fire', 'reason': 'issue strobe of that instruction'}, {'input': 'address', 'reason': 'VLS address operand of that instruction'}, {'input': 'dma_size', 'reason': 'DMA size operand of that instruction, captured at launch'}, {'input': 'dma_vmem_addr', 'reason': 'DMA VMEM operand of that instruction, captured at launch'}, {'input': 'dma_dram_addr', 'reason': 'DMA DRAM operand of that instruction, captured at launch'}, {'path': 'mreg', 'reason': 'MREG file; per-bank port conflicts are MregFile invariants'}, {'path': 'vmem', 'reason': 'VMEM arbiter; the LSU is granted unconditionally and DMA grants are the dma_vmem events'}],
    }, 'pairs/core': {'engine': 'core',
        'description': 'Second instruction issued g ages after the first through the RTL decoder; all AtlasCore assertions checked.',
        'equivalent': EQUIVALENT, 'shared': SHARED,}, 'sequences/core_vpu': {'engine': 'vpu', 'invariants': {},
        'description': 'Slot accounting for a third command arriving while two VPU commands overlap.',
        'equivalent': {'read': ['read0', 'read1'], 'write': ['write0', 'write1']},}, 'system/core': {'engine': 'core',
        'module': 'AtlasCore',
        'description': 'One instruction per command age through the RTL decoder into AtlasCore; engine ports observed through probes. Instruction words follow lib/AtlasEncoding.cpp (vd[12:7], vs1[18:13], vs2[24:19]; VLS store flag bit 13).\n',
        'limit': 200,
        'busy': ['lsu_busy', 'vpu_busy', 'xlu_busy', 'mxu0_data_busy', 'mxu0_compute_busy', 'mxu1_data_busy', 'mxu1_compute_busy'],
        'invariants': {},}, 'system/core_dma': {'engine': 'core_dma', 'module': 'AtlasCore',
        'description': 'One instruction per command age through the RTL decoder into AtlasCore with a live DMA engine; the external memory is a declared responder. DMA words follow lib/AtlasEncoding.cpp (load rd=x6 rs1=x1 rs2=x2, store rd=x3 rs1=x8 rs2=x2, channel in [14:12]); VLS and MXU words as in core.yaml.\n',
        'limit': 450,
        'busy': ['lsu_busy', 'vpu_busy', 'xlu_busy', 'mxu0_data_busy', 'mxu0_compute_busy', 'mxu1_data_busy', 'mxu1_compute_busy', 'dma_channel0', 'dma_launched0', 'dma_channel1', 'dma_launched1'],
        'invariants': {},},}

ISSUE_AGES = {**dict.fromkeys(('add cos cube exp exp2 fp8pack log mov mul pairmax pairmin rcp relu sin sqrt square sub '
        'tanh').split(), 65), **dict.fromkeys('cmax cmin csum'.split(), 129), **dict.fromkeys('fp8unpack'.split(), 66),
    **dict.fromkeys('rmax rmin'.split(), 33), **dict.fromkeys('rsum'.split(), 38),
    **dict.fromkeys('vliAll vliRow'.split(), 64), **dict.fromkeys('vliCol vliOne'.split(), 32),}

LAST_EVENTS = {**dict.fromkeys('mxu0.matmul mxu0.matmul_acc'.split(), 94), **dict.fromkeys((
        'mxu0.pop_acc_bf16 mxu0.pop_acc_fp8 mxu0.push_acc_bf16 mxu0.push_acc_fp8 mxu0.push_weight '
        'mxu1.pop_acc_bf16 mxu1.pop_acc_fp8 mxu1.push_acc_bf16 mxu1.push_acc_fp8 mxu1.push_weight').split(), 32),
    **dict.fromkeys('mxu1.matmul mxu1.matmul_acc'.split(), 34),}

def timing():
    return {"op_timing": [*({"name": f"vpu.{op}", "engine": "vpu", "operation": op, "next_issue_age": age}
          for op, age in ISSUE_AGES.items()),
        *({"name": name, "engine": name.split(".")[0], "events": {"last": {"last_age": age, "count": 1}}}
          for name, age in LAST_EVENTS.items())]}


PAIRS_CORE_VPU_EXPECT = [{'min_gap': 65, 'stable_gap': 65, 'next_issue_age': 65},
    {'min_gap': 1, 'stable_gap': 1, 'next_issue_age': 65}, {'min_gap': 38, 'stable_gap': 38, 'next_issue_age': 38},
    {'min_gap': 129, 'stable_gap': 129, 'next_issue_age': 129}, {'min_gap': 1, 'stable_gap': 1, 'next_issue_age': 129},
    {'min_gap': 66, 'stable_gap': 66, 'next_issue_age': 66}, {'min_gap': 1, 'stable_gap': 1, 'next_issue_age': 66},
    {'min_gap': 33, 'stable_gap': 33, 'next_issue_age': 33}, {'min_gap': 32, 'stable_gap': 32, 'next_issue_age': 32},
    {'min_gap': 1, 'stable_gap': 1, 'next_issue_age': 32}, {'min_gap': 64, 'stable_gap': 64, 'next_issue_age': 64},
    {'min_gap': 1, 'stable_gap': 1, 'next_issue_age': 64},]
PAIRS_CORE_VPU_REGRESSIONS = ('0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 '
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 '
    '0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 1 0 0 1 1 1 1 1 0 0 0 1 '
    '0 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 1 0 0 1 1 1 1 1 0 0 0 1 1 0 0 1 1 1 1 1 1 0 1 '
    '1 1 1 0 0 1 1 0 0 1 1 1 1 1 0 0 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 0 0 1 1 0 0 1 1 '
    '1 1 1 0 0 0 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 0 0 1 1 0 0 1 1 1 1 1 0 0 0 1 1 1 1 1 '
    '0 1 1 1 1 0 1 1 1 1 0 0 1 1 0 0 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 1 1 0 1 1 1 1 0 '
    '0 1 1 0 0 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 1 1 0 1 1 1 1 0 0 1 1 0 0 1 1 1 1 1 0 '
    '0 0 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 1 1 0 0 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 '
    '0 0 1 1 1 1 0 0 1 1 0 0 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 '
    '2 2 2 2 2 2 3 3 3 4 4 4 4 4 4 4 4 4 4 3 3 4 4 4 3 3 4 4 3 3 4 4 4 4 4 0 0 0 1 1 '
    '1 1 1 1 1 1 1 1 0 1 0 1 1 0 0 1 1 0 0 1 1 1 1 1 5 5 5 6 6 6 6 6 6 6 6 6 6 5 6 6 '
    '5 6 5 5 6 6 5 5 6 6 6 6 6 0 0 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 0 0 1 1 0 0 1 1 1 '
    '1 1 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 '
    '7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 3 3 3 4 4 4 4 4 4 4 4 4 4 3 4 4 4 4 3 3 '
    '3 4 3 3 4 4 4 4 4 3 3 3 4 4 4 4 4 4 4 4 4 4 3 4 4 4 4 3 3 4 3 3 3 4 4 4 4 4 0 0 '
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 '
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 1 0 0 '
    '0 1 1 1 1 8 8 8 9 9 9 9 9 9 9 9 9 9 8 9 9 9 9 8 8 9 9 8 8 9 8 8 8 8 8 8 8 9 9 9 '
    '9 9 9 9 9 9 9 8 9 9 9 9 8 8 9 9 8 8 9 8 8 8 8 10 10 10 11 11 11 11 11 11 11 11 11 11 10 11 11 11 '
    '11 10 10 11 11 10 10 11 10 10 10 10 10 10 10 11 11 11 11 11 11 11 11 11 11 10 11 11 11 11 10 10 11 11 10 10 11 10 10 10 '
    '10 ')

PAIRS_CORE_MXU_SAME_EXPECT = [{'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'missing mxu0_read1 events'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant SA: PushAccBF16 issued while ReadP0 or ReadP1 is busy \\(/mxu0/seq\\) fired'},
    {'min_gap': 1, 'stable_gap': 1},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant SA: Push issued while both Read ports are busy \\(/mxu0/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant SA: compute issued while ReadP0 is busy and not on its last feed cycle \\(/mxu0/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant SA: PopAccFP8 issued while WriteP0 is busy and not on its last write cycle \\(/mxu0/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant SA: PopAccBF16 issued while WriteP0 or WriteP1 is busy \\(/mxu0/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant AccumulationBuffers: compute read and store read target the same buffer \\(/mxu0/accBuf\\) fired'},
    {'min_gap': 63, 'stable_gap': 63, 'rejected_by': 'invariant SA: PushWeight issued before previous matmul on same wslot has drained \\(/mxu0/seq\\) fired'},
    {'min_gap': 94, 'stable_gap': 94, 'rejected_by': 'invariant AccumulationBuffers: compute write and load target the same buffer \\(/mxu0/accBuf\\) fired'},
    {'min_gap': 64, 'stable_gap': 64, 'rejected_by': "invariant SA: Pop issued before previous compute's row\\-0 writeback \\(/mxu0/seq\\) fired"},
    {'min_gap': 64, 'stable_gap': 64, 'rejected_by': "invariant SA: compute issued before previous compute's row\\-0 writeback on same accSel \\(/mxu0/seq\\) fired"},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: PushWeight targets wslot already being written by another active PushWeight \\(/mxu1/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'missing mxu1_read1 events'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: PushAccBF16 issued while ReadP0 or ReadP1 is busy \\(/mxu1/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: Compute targets wslot currently being written by an active push \\(/mxu1/seq\\) fired'},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'invariant IPT: PushAccFP8 targets accSel already being written by another active push\\-acc \\(/mxu1/seq\\) fired'},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'invariant IPT: Compute targets accSel currently being written by an active push\\-acc \\(/mxu1/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: Push issued while both ReadP0 and ReadP1 are busy \\(/mxu1/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: Compute issued while ReadP0 is busy and not on its last feed cycle \\(/mxu1/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: PopAccFP8 issued while WriteP0 is busy and not on its last write cycle \\(/mxu1/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: PopAccBF16 issued while WriteP0 or WriteP1 is busy \\(/mxu1/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant AccumulationBuffers: compute read and store read target the same buffer \\(/mxu1/accBuf\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: PushWeight targets wslot currently being read by compute feed \\(/mxu1/seq\\) fired'},
    {'min_gap': 34, 'stable_gap': 34, 'rejected_by': 'invariant AccumulationBuffers: compute write and load target the same buffer \\(/mxu1/accBuf\\) fired'},
    {'min_gap': 4, 'stable_gap': 4, 'rejected_by': "invariant IPT: Pop issued before previous compute's row\\-0 writeback on same accSel \\(/mxu1/seq\\) fired"},
]
PAIRS_CORE_MXU_SAME_REGRESSIONS = ('0 0 0 1 2 2 2 2 2 2 0 0 0 1 1 2 2 2 2 2 2 2 2 3 3 3 1 1 2 2 2 2 4 4 4 4 2 2 2 2 '
    '2 5 5 6 6 2 2 7 2 2 2 2 2 2 5 5 6 6 2 2 7 2 8 2 9 2 9 1 10 2 10 2 11 11 4 4 11 11 4 4 '
    '8 2 9 2 9 1 10 2 10 2 11 11 4 4 11 11 4 4 12 13 13 14 2 2 15 2 15 2 13 16 13 14 14 2 2 2 2 17 2 17 '
    '2 18 16 18 14 14 2 2 2 2 17 19 17 19 2 2 2 2 2 20 20 21 21 2 2 22 2 2 2 2 2 2 20 20 21 21 2 2 22 2 '
    '23 2 24 2 24 14 25 2 25 2 19 19 19 19 19 19 19 19 23 2 24 2 24 14 22 2 22 2 19 19 19 19 19 19 19 19 ')

PAIRS_CORE_MXU_SHARED_EXPECT = [
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant MregFile bank conflict: multiple read ports targeting physical bank 12 \\(m12 or m44\\) \\(/mreg\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant SA: PushAccBF16 issued while ReadP0 or ReadP1 is busy \\(/mxu0/seq\\) fired'},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'invariant SA: Pop writes to mreg bank currently being read by an active push/compute \\(/mxu0/seq\\) fired'},
    {'min_gap': 1, 'stable_gap': 1},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant SA: Push issued while both Read ports are busy \\(/mxu0/seq\\) fired'},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'invariant ASSERT FAIL: MREG WAR/WAW hazard: instruction writes bank with pending read/write \\(/scalar\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant SA: compute issued while ReadP0 is busy and not on its last feed cycle \\(/mxu0/seq\\) fired'},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'invariant SA: Push reads from mreg bank currently being written by an active pop \\(/mxu0/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant SA: PopAccFP8 issued while WriteP0 is busy and not on its last write cycle \\(/mxu0/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant SA: PopAccBF16 issued while WriteP0 or WriteP1 is busy \\(/mxu0/seq\\) fired'},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'invariant ASSERT FAIL: MREG RAW hazard: instruction reads bank with pending write \\(/scalar\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant AccumulationBuffers: compute read and store read target the same buffer \\(/mxu0/accBuf\\) fired'},
    {'min_gap': 63, 'stable_gap': 63, 'rejected_by': 'invariant SA: PushWeight issued before previous matmul on same wslot has drained \\(/mxu0/seq\\) fired'},
    {'min_gap': 94, 'stable_gap': 94, 'rejected_by': 'invariant AccumulationBuffers: compute write and load target the same buffer \\(/mxu0/accBuf\\) fired'},
    {'min_gap': 64, 'stable_gap': 64, 'rejected_by': "invariant SA: Pop issued before previous compute's row\\-0 writeback \\(/mxu0/seq\\) fired"},
    {'min_gap': 64, 'stable_gap': 64, 'rejected_by': "invariant SA: compute issued before previous compute's row\\-0 writeback on same accSel \\(/mxu0/seq\\) fired"},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: PushWeight targets wslot already being written by another active PushWeight \\(/mxu1/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: PushAccBF16 issued while ReadP0 or ReadP1 is busy \\(/mxu1/seq\\) fired'},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'invariant IPT: Pop writes to mreg bank currently being read by an active push/compute \\(/mxu1/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: Compute targets wslot currently being written by an active push \\(/mxu1/seq\\) fired'},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'invariant IPT: PushAccFP8 targets accSel already being written by another active push\\-acc \\(/mxu1/seq\\) fired'},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'invariant IPT: Compute targets accSel currently being written by an active push\\-acc \\(/mxu1/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: Push issued while both ReadP0 and ReadP1 are busy \\(/mxu1/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: Compute issued while ReadP0 is busy and not on its last feed cycle \\(/mxu1/seq\\) fired'},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'invariant IPT: Push reads from mreg bank currently being written by an active pop \\(/mxu1/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: PopAccFP8 issued while WriteP0 is busy and not on its last write cycle \\(/mxu1/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: PopAccBF16 issued while WriteP0 or WriteP1 is busy \\(/mxu1/seq\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant AccumulationBuffers: compute read and store read target the same buffer \\(/mxu1/accBuf\\) fired'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'invariant IPT: PushWeight targets wslot currently being read by compute feed \\(/mxu1/seq\\) fired'},
    {'min_gap': 34, 'stable_gap': 34, 'rejected_by': 'invariant AccumulationBuffers: compute write and load target the same buffer \\(/mxu1/accBuf\\) fired'},
    {'min_gap': 4, 'stable_gap': 4, 'rejected_by': "invariant IPT: Pop issued before previous compute's row\\-0 writeback on same accSel \\(/mxu1/seq\\) fired"},
]
PAIRS_CORE_MXU_SHARED_REGRESSIONS = ('0 0 0 0 0 0 1 1 1 1 2 3 2 2 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 '
    '3 3 2 2 2 2 3 3 3 3 0 0 0 0 0 0 0 0 4 4 4 4 4 4 4 4 4 4 4 4 1 1 1 1 2 2 3 3 5 5 '
    '3 3 2 2 3 3 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 7 3 7 7 3 3 7 7 7 7 3 3 3 3 5 5 8 8 '
    '5 5 5 5 9 9 9 9 10 10 3 3 10 10 11 3 7 3 10 3 7 7 3 3 10 10 3 3 7 7 3 3 5 5 8 8 5 5 8 8 '
    '5 5 9 9 10 10 3 3 10 10 3 3 10 10 11 3 10 10 11 3 12 0 12 0 13 0 13 0 13 1 13 1 13 1 13 1 14 2 14 3 '
    '14 2 14 2 14 3 14 3 15 15 6 6 15 15 6 6 15 15 6 6 15 15 6 6 12 0 12 0 13 0 13 0 13 1 13 1 13 1 13 1 '
    '14 2 14 3 14 2 14 2 14 3 14 3 15 15 6 6 15 15 6 6 15 15 6 6 15 15 6 6 16 0 16 0 0 0 17 17 17 17 18 3 '
    '18 18 3 3 19 0 19 0 19 0 19 0 0 0 20 0 20 0 17 17 17 17 17 17 17 17 18 18 3 3 18 18 18 18 3 3 3 3 21 0 '
    '21 0 21 0 21 0 22 22 22 22 20 22 20 22 20 22 20 22 17 17 17 17 18 18 3 3 5 5 3 3 18 18 3 3 21 23 21 23 21 23 '
    '21 23 21 23 21 23 21 23 21 23 24 3 24 24 3 3 24 24 24 24 3 3 3 3 5 5 25 25 5 5 5 5 26 26 26 26 10 10 3 3 '
    '10 10 27 3 24 3 10 3 24 24 3 3 10 10 3 3 24 24 3 3 5 5 25 25 5 5 25 25 5 5 26 26 10 10 3 3 10 10 3 3 '
    '10 10 27 3 10 10 27 3 28 0 28 0 29 0 29 0 29 17 29 17 29 17 29 17 18 18 30 3 18 18 18 18 30 3 30 3 23 23 23 23 '
    '23 23 23 23 23 23 23 23 23 23 23 23 28 0 28 0 29 0 29 0 29 17 29 17 29 17 29 17 18 18 27 3 18 18 18 18 27 3 27 3 '
    '23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 ')

SEQUENCES_CORE_MXU_EXPECT = [{2: 64, 3: 64, 4: 64}, {2: 32, 3: 32, 4: 32, 5: 32}, {2: 1, 3: 94, 4: 94},
    {2: 1, 3: 16, 4: 16, 5: 32}, {2: 32, 3: 94, 4: 94}, {2: 1, 3: 32, 4: 32}, {3: 64, 4: 64, 5: 64, 6: 64},
    {2: 1, 3: 16, 4: 32}, {2: 32, 3: 32, 4: 32}, {3: 1, 4: 22, 5: 22, 6: 22}, {3: 1, 4: 63, 5: 63, 6: 63},
    {3: 16, 4: 32, 5: 47, 6: 47}, {3: 16, 4: 94, 5: 94, 6: 94}, {3: 1, 4: 11, 5: 11, 6: 22}, {2: 33, 3: 34, 4: 34},
    {2: 33, 3: 33, 4: 33, 5: 33}, {2: 4, 3: 16, 4: 16}, {2: 1, 3: 16, 4: 16}, {3: 32, 4: 32, 5: 32, 6: 32},
    {3: 1, 4: 11, 5: 11, 6: 11}, {3: 16, 4: 32, 5: 32, 6: 32}, {3: 33, 4: 33, 5: 33, 6: 33},
    {3: 17, 4: 34, 5: 34, 6: 34},]
SEQUENCES_CORE_MXU_REGRESSIONS = (
    '0 0 1 1 0 1 0 2 3 4 0 5 0 6 6 7 8 8 8 8 9 10 11 12 13 8 8 1 1 8 1 8 14 15 14 16 17 16 18 18 '
    '17 8 8 8 8 19 20 21 22 19 ')

PAIRS_CORE_MXU_CROSS_EXPECT = [{'min_gap': 1, 'stable_gap': 1, 'disjoint': True},
    {'min_gap': 35, 'stable_gap': 35, 'rejected_by': 'MREG RAW hazard: instruction reads bank with pending write', 'disjoint': True},
    {'min_gap': 66, 'stable_gap': 66, 'rejected_by': 'MREG RAW hazard: instruction reads bank with pending write', 'disjoint': True},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'MREG RAW hazard: instruction reads bank with pending write', 'disjoint': True},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'MREG WAR/WAW hazard: instruction writes bank with pending re', 'disjoint': True},
    {'min_gap': 35, 'stable_gap': 35, 'rejected_by': 'MREG WAR/WAW hazard: instruction writes bank with pending re', 'disjoint': True},
    {'min_gap': 64, 'stable_gap': 64, 'rejected_by': 'MREG WAR/WAW hazard: instruction writes bank with pending re', 'disjoint': True},
    {'min_gap': 34, 'stable_gap': 34, 'rejected_by': 'MREG WAR/WAW hazard: instruction writes bank with pending re', 'disjoint': True},
    {'min_gap': 31, 'stable_gap': 31, 'rejected_by': 'multiple read ports targeting physical bank 12 \\(m12 or m44\\)', 'disjoint': True},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'multiple read ports targeting physical bank 12 \\(m12 or m44\\)', 'disjoint': True},
    {'min_gap': 30, 'stable_gap': 30, 'rejected_by': 'multiple write ports targeting physical bank 20 \\(m20 or m52\\)', 'disjoint': True},
    {'min_gap': 34, 'stable_gap': 34, 'rejected_by': 'multiple write ports targeting physical bank 20 \\(m20 or m52\\)', 'disjoint': True},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'multiple read ports targeting physical bank 12 \\(m12 or m44\\)', 'disjoint': True},
]
PAIRS_CORE_MXU_CROSS_REGRESSIONS = ('0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 '
    '0 0 0 0 0 0 0 0 0 0 0 0 1 2 2 3 3 3 0 4 4 5 6 7 4 5 8 9 10 11 12 12 0 0 0 0 0 0 0 0 '
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 '
    '0 0 0 0 0 0 0 0 1 2 2 3 3 3 0 4 4 5 6 7 4 5 8 9 10 11 12 12 0 0 0 0 5 5 ')

PAIRS_CORE_MXU_UNITS_EXPECT = [{'min_gap': 1, 'stable_gap': 1, 'disjoint': True},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'MREG RAW hazard: instruction reads bank with pending write', 'disjoint': True},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'MREG WAR/WAW hazard: instruction writes bank with pending re', 'disjoint': True},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'multiple read ports targeting physical bank 12 \\(m12 or m44\\)', 'disjoint': True},
]
PAIRS_CORE_MXU_UNITS_REGRESSIONS = ('0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 '
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 '
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 2 3 3 ')

PAIRS_CORE_MXU_EXPECT = [
    {'min_gap': 64, 'stable_gap': 64, 'rejected_by': 'SA: compute issued before previous compute.s row\\-0 writeback on same accSel \\(/mxu0/seq\\)'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'SA: compute issued while ReadP0 is busy and not on its last feed cycle \\(/mxu0/seq\\)'},
    {'min_gap': 1, 'stable_gap': 1, 'disjoint': False},
    {'min_gap': 64, 'stable_gap': 64, 'rejected_by': 'SA: Pop issued before previous compute.s row\\-0 writeback \\(/mxu0/seq\\)', 'disjoint': False},
    {'min_gap': 1, 'stable_gap': 1, 'disjoint': True},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'IPT: Compute issued while ReadP0 is busy and not on its last feed cycle \\(/mxu1/seq\\)'},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'IPT: Compute targets wslot currently being written by an active push \\(/mxu1/seq\\)', 'disjoint': False},
    {'min_gap': 4, 'stable_gap': 4, 'rejected_by': 'IPT: Pop issued before previous compute.s row\\-0 writeback on same accSel \\(/mxu1/seq\\)', 'disjoint': False},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'IPT: Compute targets accSel currently being written by an active push\\-acc \\(/mxu1/seq\\)', 'disjoint': False},
]
PAIRS_CORE_MXU_REGRESSIONS = ('0 0 1 1 2 2 3 2 2 2 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 5 5 5 5 6 2 '
    '7 2 8 2 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 ')

PAIRS_CORE_DMA_MXU_EXPECT = [{'min_gap': 1, 'stable_gap': 1, 'disjoint': True},]
PAIRS_CORE_DMA_MXU_REGRESSIONS = ('0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 '
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ')

PAIRS_CORE_DMA_EXPECT = [{'min_gap': None, 'stable_gap': None, 'rejected_by': 'missing dram.request events'},
    {'min_gap': None, 'stable_gap': None, 'rejected_by': 'missing dma_vmem_read events'},
    {'min_gap': 1, 'stable_gap': 1, 'disjoint': True},
    {'min_gap': 1, 'stable_gap': 75, 'rejected_by': 'missing dma_vmem_write events', 'disjoint': True},
    {'min_gap': 1, 'stable_gap': 73, 'rejected_by': 'missing dma_vmem_write events', 'disjoint': True},
    {'min_gap': 32, 'stable_gap': 32, 'rejected_by': 'missing dma_vmem_read events', 'disjoint': True},
    {'min_gap': 30, 'stable_gap': 30, 'rejected_by': 'missing dma_vmem_read events', 'disjoint': True},]
PAIRS_CORE_DMA_REGRESSIONS = ('0 0 1 1 0 2 2 2 2 3 4 5 6 4 2 2 2 2 5 2 2 2 2 ')

PAIRS_CORE_EXPECT = [{'min_gap': 1, 'stable_gap': 1, 'disjoint': True},
    {'min_gap': 30, 'stable_gap': 30, 'rejected_by': 'VLOAD and VSTORE target same VMEM bank \\(/vmem\\)', 'disjoint': True},
    {'min_gap': 34, 'stable_gap': 34, 'rejected_by': 'VLOAD and VSTORE target same VMEM bank \\(/vmem\\)', 'disjoint': True},
    {'min_gap': 35, 'stable_gap': 35, 'rejected_by': 'VLOAD issued while VLOAD path is busy \\(/scalar\\)'},
    {'min_gap': 35, 'stable_gap': 35, 'rejected_by': 'VSTORE issued while VSTORE path is busy \\(/scalar\\)'},
    {'min_gap': 35, 'stable_gap': 35, 'rejected_by': 'MREG RAW hazard', 'disjoint': True},
    {'min_gap': 31, 'stable_gap': 31, 'rejected_by': 'MregFile bank conflict: multiple read ports targeting physical bank 2 ', 'disjoint': True},
    {'min_gap': 65, 'stable_gap': 65, 'rejected_by': 'VPU command issued while its issue-busy bit is set \\(/scalar\\)'},
    {'min_gap': 1, 'stable_gap': 1},
    {'min_gap': 33, 'stable_gap': 33, 'rejected_by': 'VPU command issued while its issue-busy bit is set \\(/scalar\\)'},
    {'min_gap': 38, 'stable_gap': 38, 'rejected_by': 'VPU command issued while its issue-busy bit is set \\(/scalar\\)'},
    {'min_gap': 129, 'stable_gap': 129, 'rejected_by': 'VPU command issued while its issue-busy bit is set \\(/scalar\\)'},
    {'min_gap': 66, 'stable_gap': 66, 'rejected_by': 'VPU command issued while its issue-busy bit is set \\(/scalar\\)'},
    {'min_gap': 64, 'stable_gap': 64, 'rejected_by': 'VPU command issued while its issue-busy bit is set \\(/scalar\\)'},
]
PAIRS_CORE_REGRESSIONS = ('0 0 1 2 3 4 5 0 0 0 0 6 0 0 0 0 7 8 7 7 9 10 11 12 13 ')

SEQUENCES_CORE_VPU_EXPECT = [{3: 33, 4: 33}, {3: 33, 4: 65}, {3: 65, 4: 65}, {3: 32, 4: 33},]
SEQUENCES_CORE_VPU_REGRESSIONS = ('0 1 0 2 2 2 2 3 0 ')

CROSS_OVERLAYS = [('vload_{unit}_matmul_m5', 'vload', '{unit}_matmul_m5', 50),
    ('transpose_{unit}_push_weight_m7', 'transpose', '{unit}_push_weight_m7', 80),
    ('add_{unit}_push_acc_bf16_m10', 'add', '{unit}_push_acc_bf16_m10', 80),
    ('{unit}_pop_acc_bf16_vstore_m20', '{unit}_pop_acc_bf16', 'vstore_m20', None),
    ('{unit}_pop_acc_bf16_exp_m20', '{unit}_pop_acc_bf16', 'exp_m20', None),
    ('{unit}_pop_acc_bf16_transpose_m20', '{unit}_pop_acc_bf16', 'transpose_m20', None),
    ('{unit}_matmul_vload_m12', '{unit}_matmul', 'vload_m12', None),
    ('{unit}_matmul_vli_all_m12', '{unit}_matmul', 'vli_all_m12', None),
    ('{unit}_matmul_transpose_to_m12', '{unit}_matmul', 'transpose_to_m12', None),
    ('vstore_{unit}_pop_acc_bf16_m8', 'vstore', '{unit}_pop_acc_bf16_m8', 50),
    ('add_{unit}_pop_acc_bf16_m2', 'add', '{unit}_pop_acc_bf16_m2', 80),
    ('transpose_{unit}_pop_acc_fp8_m3', 'transpose', '{unit}_pop_acc_fp8_m3', 80),
    ('{unit}_pop_acc_bf16_vload_m20', '{unit}_pop_acc_bf16', 'vload_m20', None),
    ('vload_{unit}_pop_acc_bf16_m4', 'vload', '{unit}_pop_acc_bf16_m4', 50),
    ('{unit}_matmul_vstore_m44', '{unit}_matmul', 'vstore_m44', None),
    ('vstore_m44_{unit}_matmul', 'vstore_m44', '{unit}_matmul', 50),
    ('{unit}_pop_acc_bf16_vload_m52', '{unit}_pop_acc_bf16', 'vload_m52', None),
    ('vload_m52_{unit}_pop_acc_bf16', 'vload_m52', '{unit}_pop_acc_bf16', 50),
    ('{unit}_matmul_exp_m44', '{unit}_matmul', 'exp_m44', None),
    ('exp_m44_{unit}_matmul', 'exp_m44', '{unit}_matmul', 80),
    ('{unit}_matmul_vload_m44', '{unit}_matmul', 'vload_m44', None),
    ('vload_m44_{unit}_matmul', 'vload_m44', '{unit}_matmul', 50),
    ('{unit}_pop_acc_bf16_vstore_m52', '{unit}_pop_acc_bf16', 'vstore_m52', None),
    ('vstore_m52_{unit}_pop_acc_bf16', 'vstore_m52', '{unit}_pop_acc_bf16', 50),]
MXU_SAME_ROWS = [('matmul_matmul_same_acc', 'matmul', 'matmul_m14'),
    ('matmul_matmul_same_acc_other_slot', 'matmul', 'matmul_m14_slot1'),
    ('matmul_matmul_other_acc', 'matmul', 'matmul_m14_acc1'),
    ('matmul_matmul_other_acc_other_slot', 'matmul', 'matmul_m14_acc1_slot1'),
    ('push_weight_matmul_same_slot', 'push_weight', 'matmul_m14'),
    ('push_weight_matmul_other_slot', 'push_weight', 'matmul_m14_slot1'),
    ('matmul_pop_same_acc', 'matmul', 'pop_acc_bf16'), ('matmul_pop_other_acc', 'matmul', 'pop_acc_bf16_acc1'),
    ('push_acc_fp8_matmul_acc_same_acc', 'push_acc_fp8', 'matmul_acc_m14'),
    ('push_acc_fp8_matmul_acc_other_acc', 'push_acc_fp8', 'matmul_acc_m14_acc1'),]
UNIT_OVERLAYS = [('mxu0_pop_acc_bf16_mxu1_matmul_m20', 'mxu0_pop_acc_bf16', 'mxu1_matmul_m20', 110),
    ('mxu1_pop_acc_bf16_mxu0_matmul_m20', 'mxu1_pop_acc_bf16', 'mxu0_matmul_m20', 50),
    ('mxu0_matmul_mxu1_pop_acc_bf16_m12', 'mxu0_matmul', 'mxu1_pop_acc_bf16_m12', 110),
    ('mxu1_matmul_mxu0_pop_acc_bf16_m12', 'mxu1_matmul', 'mxu0_pop_acc_bf16_m12', 50),
    ('mxu0_matmul_mxu1_matmul_m44', 'mxu0_matmul', 'mxu1_matmul_m44', 110),
    ('mxu1_matmul_mxu0_matmul_m44', 'mxu1_matmul', 'mxu0_matmul_m44', 50),]

def opposed(rows, first, second, first_limit, second_limit, first_name=None, second_name=None):
    a, b = first_name or first, second_name or second
    rows.extend([(f"{a}_{b}", first, second, first_limit), (f"{b}_{a}", second, first, second_limit)])


def enumerated_pairs(name):
    rows = []
    if name == "pairs/core_dma_mxu":
        for unit in (0, 1):
            for kind in ("matmul", "matmul_acc", "push_weight", "push_acc_fp8", "push_acc_bf16", "pop_acc_fp8", "pop_acc_bf16"):
                for direction in ("load", "store"):
                    opposed(rows, f"dma_{direction}0", f"mxu{unit}_{kind}", 140, 100, first_name=f"{direction}0")
    elif name == "pairs/core_mxu_units":
        def op(unit, kind):
            suffix = ("_m22" if kind.startswith("pop") else "_m14") if unit == 1 else ""
            if kind == "matmul_acc" and unit == 0:
                suffix = "_m12"
            return f"mxu{unit}_{kind}{suffix}"
        for first in KINDS:
            for second in KINDS:
                rows.extend([(f"mxu0_{first}_mxu1_{second}", op(0, first), op(1, second), 110),
                             (f"mxu1_{first}_mxu0_{second}", op(1, first), op(0, second), 50)])
        rows.extend(UNIT_OVERLAYS)
    elif name == "pairs/core_mxu":
        for unit in (0, 1):
            rows.extend((f"mxu{unit}_{key}", f"mxu{unit}_{a}", f"mxu{unit}_{b}", 70 if unit == 0 else 45)
                        for key, a, b in MXU_SAME_ROWS)
            for kind in ("matmul", "push_weight", "pop_acc_bf16"):
                for other in ("add", "vload", "vstore", "transpose"):
                    opposed(rows, f"mxu{unit}_{kind}", other, 70, 70)
        rows.extend([(f"mxu{u}_matmul_mxu{1-u}_matmul", f"mxu{u}_matmul", f"mxu{1-u}_matmul_m14", 70) for u in (0, 1)])
    elif name == "pairs/core_mxu_cross":
        for unit in (0, 1):
            limit = 110 if unit == 0 else 50
            for kind in ("push_acc_fp8", "push_acc_bf16", "pop_acc_fp8", "matmul_acc_m14"):
                for other in ("add", "vload", "vstore", "transpose"):
                    opposed(rows, f"mxu{unit}_{kind}", other, limit, 50 if other.startswith("v") else 80)
            for kind in ("matmul", "pop_acc_bf16"):
                for other, end in (("exp", 80), ("rmax", 60), ("csum", 140), ("fp8unpack", 80), ("vli_all", 80)):
                    opposed(rows, f"mxu{unit}_{kind}", other, limit, end)
            rows.extend((key.format(unit=f"mxu{unit}"), a.format(unit=f"mxu{unit}"), b.format(unit=f"mxu{unit}"), g or limit)
                        for key, a, b, g in CROSS_OVERLAYS)
        rows.extend([("vstore_transpose_to_m9", "vstore", "transpose_to_m9", 50),
                     ("vstore_vli_all_m8", "vstore", "vli_all_m8", 50)])
    else:
        return globals()[name.replace("/", "_").upper() + "_ROWS"]
    return rows


PAIRS_CORE_DMA_ROWS = [('load0_load1', 'dma_load0', 'dma_load1', 140),
    ('load0_load1_bank3', 'dma_load0', 'dma_load1_bank3', 140), ('store0_store1', 'dma_store0', 'dma_store1', 140),
    ('load0_store1', 'dma_load0', 'dma_store1', 140), ('store0_load1', 'dma_store0', 'dma_load1', 140),
    ('load0_vload', 'dma_load0', 'vload', 140), ('load0_vstore', 'dma_load0', 'vstore', 140),
    ('store0_vload', 'dma_store0', 'vload', 140), ('store0_vstore', 'dma_store0', 'vstore', 140),
    ('load0_vload_bank3', 'dma_load0', 'vload_bank3', 140), ('load0_vstore_bank3', 'dma_load0', 'vstore_bank3', 140),
    ('store0_vload_bank3', 'dma_store0', 'vload_bank3', 140),
    ('store0_vstore_bank3', 'dma_store0', 'vstore_bank3', 140),
    ('load0_vstore_bank3_tile', 'dma_load0', 'vstore_bank3_tile', 140), ('vload_load0', 'vload', 'dma_load0', 40),
    ('vstore_load0', 'vstore', 'dma_load0', 40), ('vload_store0', 'vload', 'dma_store0', 40),
    ('vload_bank3_load0', 'vload_bank3', 'dma_load0', 40), ('vload_bank3_store0', 'vload_bank3', 'dma_store0', 40),
    ('load0_add', 'dma_load0', 'add', 140), ('add_load0', 'add', 'dma_load0', 70),
    ('load0_transpose', 'dma_load0', 'transpose', 140), ('transpose_store0', 'transpose', 'dma_store0', 70),]

PAIRS_CORE_ROWS = [('vload_vstore', 'vload', 'vstore', 40), ('vstore_vload', 'vstore', 'vload', 40),
    ('vload_vstore_same_bank', 'vload', 'vstore_bank1', 40), ('vstore_vload_same_bank', 'vstore', 'vload_bank2', 40),
    ('vload_vload', 'vload', 'vload_bank5', 40), ('vstore_vstore', 'vstore', 'vstore_bank3', 40),
    ('vload_vstore_same_mreg', 'vload', 'vstore_m5', 40), ('add_vload', 'add', 'vload', 70),
    ('vload_add', 'vload', 'add', 40), ('add_vstore', 'add', 'vstore', 70), ('vstore_add', 'vstore', 'add', 40),
    ('add_vstore_same_mreg_bank', 'add', 'vstore_m34', 70), ('transpose_vload', 'transpose', 'vload', 70),
    ('vload_transpose', 'vload', 'transpose', 40), ('transpose_vstore', 'transpose', 'vstore', 70),
    ('vstore_transpose', 'vstore', 'transpose', 40), ('add_add', 'add', 'add_hi', 70),
    ('exp_relu', 'exp', 'relu_hi', 70), ('exp_exp2', 'exp', 'exp2_hi', 70), ('exp_add', 'exp', 'add_hi', 70),
    ('rmax_exp', 'rmax', 'exp_hi', 40), ('rsum_add', 'rsum', 'add_hi', 45), ('csum_add', 'csum', 'add_hi', 135),
    ('fp8unpack_add', 'fp8unpack', 'add_hi', 70), ('vli_all_add', 'vli_all', 'add_hi', 70),]

VPU_CHAINS = [
    ('exp_relu_exp', 'third shares lane logic with the first (exp) but not the second', ('exp', 'relu', 'exp', 'relu')),
    ('exp_relu_sin_cos', 'third (sin) shares nothing; fourth (cos) shares with the third', ('exp', 'relu', 'sin', 'cos')),
    ('sin_relu_cos', 'third shares lane logic with the first (sin) but not the second', ('sin', 'relu', 'cos', 'relu')),
    ('relu_exp_exp2', 'third shares lane logic with the second (exp) but not the first', ('relu', 'exp', 'exp2', 'relu')),
    ('exp_relu_add', 'third needs both read slots', ('exp', 'relu', 'add', 'sin')),
    ('exp_relu_rmax', 'third is a row reduction (double ports)', ('exp', 'relu', 'rmax', 'sin')),
    ('csum_relu_exp', 'a column reduction (129 ages) with two single-input commands behind it', ('csum', 'relu', 'exp', 'sin')),
    ('vliall_relu_exp', 'an immediate fill (64 ages) with two single-input commands behind it', ('vliAll', 'relu', 'exp', 'sin')),
    ('exp_fp8pack_vlicol', 'pack and fill commands overlapping a transcendental', ('exp', 'fp8pack', 'vliCol', 'relu')),
]

CORE_PROBES = {'lsu_busy': {'instance': 'lsu', 'result': 'io_vecBusy'},
    'vpu_busy': {'instance': 'vpu', 'result': 'io_busy'}, 'xlu_busy': {'instance': 'xlu', 'result': 'io_busy'},
    'lsu_vmem_read': {'instance': 'lsu', 'result': 'io_vmemVecRead_valid'},
    'lsu_vmem_read_bank': {'instance': 'lsu', 'result': 'io_vmemVecRead_bits_bankIdx'},
    'lsu_vmem_read_row': {'instance': 'lsu', 'result': 'io_vmemVecRead_bits_bankAddr'},
    'lsu_vmem_write': {'instance': 'lsu', 'result': 'io_vmemVecWrite_valid'},
    'lsu_vmem_write_bank': {'instance': 'lsu', 'result': 'io_vmemVecWrite_bits_bankIdx'},
    'lsu_vmem_write_row': {'instance': 'lsu', 'result': 'io_vmemVecWrite_bits_bankAddr'},
    'lsu_mreg_read': {'instance': 'lsu', 'result': 'io_mregReadReq_valid'},
    'lsu_mreg_read_id': {'instance': 'lsu', 'result': 'io_mregReadReq_bits_mregId'},
    'lsu_mreg_read_row': {'instance': 'lsu', 'result': 'io_mregReadReq_bits_row'},
    'lsu_mreg_write': {'instance': 'lsu', 'result': 'io_mregWriteReq_valid'},
    'lsu_mreg_write_id': {'instance': 'lsu', 'result': 'io_mregWriteReq_bits_mregId'},
    'lsu_mreg_write_row': {'instance': 'lsu', 'result': 'io_mregWriteReq_bits_row'},
    'vpu_read0': {'instance': 'vpu', 'result': 'io_mregReadReq0_valid'},
    'vpu_read0_id': {'instance': 'vpu', 'result': 'io_mregReadReq0_bits_mregId'},
    'vpu_read0_row': {'instance': 'vpu', 'result': 'io_mregReadReq0_bits_row'},
    'vpu_read1': {'instance': 'vpu', 'result': 'io_mregReadReq1_valid'},
    'vpu_read1_id': {'instance': 'vpu', 'result': 'io_mregReadReq1_bits_mregId'},
    'vpu_read1_row': {'instance': 'vpu', 'result': 'io_mregReadReq1_bits_row'},
    'vpu_write0': {'instance': 'vpu', 'result': 'io_mregWriteReq0_valid'},
    'vpu_write0_id': {'instance': 'vpu', 'result': 'io_mregWriteReq0_bits_mregId'},
    'vpu_write0_row': {'instance': 'vpu', 'result': 'io_mregWriteReq0_bits_row'},
    'vpu_write1': {'instance': 'vpu', 'result': 'io_mregWriteReq1_valid'},
    'vpu_write1_id': {'instance': 'vpu', 'result': 'io_mregWriteReq1_bits_mregId'},
    'vpu_write1_row': {'instance': 'vpu', 'result': 'io_mregWriteReq1_bits_row'},
    'xlu_read': {'instance': 'xlu', 'result': 'io_mregReadReq_valid'},
    'xlu_read_id': {'instance': 'xlu', 'result': 'io_mregReadReq_bits_mregId'},
    'xlu_read_row': {'instance': 'xlu', 'result': 'io_mregReadReq_bits_row'},
    'xlu_write': {'instance': 'xlu', 'result': 'io_mregWriteReq_valid'},
    'xlu_write_id': {'instance': 'xlu', 'result': 'io_mregWriteReq_bits_mregId'},
    'xlu_write_row': {'instance': 'xlu', 'result': 'io_mregWriteReq_bits_row'},}
DMA_PROBES = {'dma_channel0': {'instance': 'dma', 'result': 'io_channelBusy_0'},
 'dma_channel1': {'instance': 'dma', 'result': 'io_channelBusy_1'},
 'dma_launched0': {'path': '', 'value': 'dmaLaunched_0'}, 'dma_launched1': {'path': '', 'value': 'dmaLaunched_1'},
 'dma_vmem_write_grant': {'instance': 'vmem', 'result': 'io_dmaWriteGrant'},
 'dma_vmem_write_bank': {'instance': 'dma', 'result': 'io_vmemWrite_bits_bankIdx'},
 'dma_vmem_write_line': {'instance': 'dma', 'result': 'io_vmemWrite_bits_bankAddr'},
 'dma_vmem_read_grant': {'instance': 'vmem', 'result': 'io_dmaReadGrant'},
 'dma_vmem_read_bank': {'instance': 'dma', 'result': 'io_vmemRead_bits_bankIdx'},
 'dma_vmem_read_line': {'instance': 'dma', 'result': 'io_vmemRead_bits_bankAddr'}}
MXU_PROBES = {'{unit}_read0': {'instance': '{unit}', 'result': 'io_mregReadReq0_valid'},
    '{unit}_read0_id': {'instance': '{unit}', 'result': 'io_mregReadReq0_bits_mregId'},
    '{unit}_read0_row': {'instance': '{unit}', 'result': 'io_mregReadReq0_bits_row'},
    '{unit}_read1': {'instance': '{unit}', 'result': 'io_mregReadReq1_valid'},
    '{unit}_read1_id': {'instance': '{unit}', 'result': 'io_mregReadReq1_bits_mregId'},
    '{unit}_read1_row': {'instance': '{unit}', 'result': 'io_mregReadReq1_bits_row'},
    '{unit}_write0': {'instance': '{unit}', 'result': 'io_mregWriteReq0_valid'},
    '{unit}_write0_id': {'instance': '{unit}', 'result': 'io_mregWriteReq0_bits_mregId'},
    '{unit}_write0_row': {'instance': '{unit}', 'result': 'io_mregWriteReq0_bits_row'},
    '{unit}_write1': {'instance': '{unit}', 'result': 'io_mregWriteReq1_valid'},
    '{unit}_write1_id': {'instance': '{unit}', 'result': 'io_mregWriteReq1_bits_mregId'},
    '{unit}_write1_row': {'instance': '{unit}', 'result': 'io_mregWriteReq1_bits_row'},
    '{unit}_compute': {'instance': '{unit}/seq', 'result': 'io_compute_valid'},
    '{unit}_weight_write': {'instance': '{unit}/seq', 'result': 'io_weightWriteReq_valid'},
    '{unit}_acc_write': {'instance': '{unit}/seq', 'result': 'io_accComputeWrite_valid'},
    '{unit}_acc_write_row': {'instance': '{unit}/seq', 'result': 'io_accComputeWrite_bits_rowIdx'},
    '{unit}_acc_read': {'instance': '{unit}/seq', 'result': 'io_accComputeReadEn'},
    '{unit}_acc_read_row': {'instance': '{unit}/seq', 'result': 'io_accComputeReadAddr_rowIdx'},
    '{unit}_acc_load': {'instance': '{unit}/seq', 'result': 'io_accLoadReq_valid'},
    '{unit}_acc_load_row': {'instance': '{unit}/seq', 'result': 'io_accLoadReq_bits_rowIdx'},
    '{unit}_acc_store': {'instance': '{unit}/seq', 'result': 'io_accStoreReadEn'},
    '{unit}_acc_store_row': {'instance': '{unit}/seq', 'result': 'io_accStoreAddr_rowIdx'},
    '{unit}_data_busy': {'instance': '{unit}', 'result': 'io_dataBusy'},
    '{unit}_compute_busy': {'instance': '{unit}', 'result': 'io_computeBusy'},}

CORE_EVENTS = {
    'vmem_read': {'valid': 'lsu_vmem_read', 'fields': {'bankIdx': 'lsu_vmem_read_bank', 'bankAddr': 'lsu_vmem_read_row'}, 'row': 'bankAddr'},
    'vmem_write': {'valid': 'lsu_vmem_write', 'fields': {'bankIdx': 'lsu_vmem_write_bank', 'bankAddr': 'lsu_vmem_write_row'}, 'row': 'bankAddr'},
    'mreg_read': {'valid': 'lsu_mreg_read', 'fields': {'mregId': 'lsu_mreg_read_id', 'row': 'lsu_mreg_read_row'}, 'row': 'row'},
    'mreg_write': {'valid': 'lsu_mreg_write', 'fields': {'mregId': 'lsu_mreg_write_id', 'row': 'lsu_mreg_write_row'}, 'row': 'row'},
    'vpu_read0': {'valid': 'vpu_read0', 'fields': {'mregId': 'vpu_read0_id', 'row': 'vpu_read0_row'}, 'row': 'row', 'split_by': 'mregId'},
    'vpu_read1': {'valid': 'vpu_read1', 'fields': {'mregId': 'vpu_read1_id', 'row': 'vpu_read1_row'}, 'row': 'row', 'split_by': 'mregId'},
    'vpu_write0': {'valid': 'vpu_write0', 'fields': {'mregId': 'vpu_write0_id', 'row': 'vpu_write0_row'}, 'row': 'row', 'split_by': 'mregId'},
    'vpu_write1': {'valid': 'vpu_write1', 'fields': {'mregId': 'vpu_write1_id', 'row': 'vpu_write1_row'}, 'row': 'row', 'split_by': 'mregId'},
    'xlu_read': {'valid': 'xlu_read', 'fields': {'mregId': 'xlu_read_id', 'row': 'xlu_read_row'}, 'row': 'row'},
    'xlu_write': {'valid': 'xlu_write', 'fields': {'mregId': 'xlu_write_id', 'row': 'xlu_write_row'}, 'row': 'row'},}
DMA_EVENTS = {'dma_vmem_write': {'valid': 'dma_vmem_write_grant',
                    'fields': {'bankIdx': 'dma_vmem_write_bank', 'bankAddr': 'dma_vmem_write_line'}},
 'dma_vmem_read': {'valid': 'dma_vmem_read_grant',
                   'fields': {'bankIdx': 'dma_vmem_read_bank', 'bankAddr': 'dma_vmem_read_line'}}}
MXU_EVENTS = {
    '{unit}_read0': {'valid': '{unit}_read0', 'fields': {'mregId': '{unit}_read0_id', 'row': '{unit}_read0_row'}, 'row': 'row'},
    '{unit}_read1': {'valid': '{unit}_read1', 'fields': {'mregId': '{unit}_read1_id', 'row': '{unit}_read1_row'}, 'row': 'row'},
    '{unit}_write0': {'valid': '{unit}_write0', 'fields': {'mregId': '{unit}_write0_id', 'row': '{unit}_write0_row'}, 'row': 'row'},
    '{unit}_write1': {'valid': '{unit}_write1', 'fields': {'mregId': '{unit}_write1_id', 'row': '{unit}_write1_row'}, 'row': 'row'},
    '{unit}_compute': {'valid': '{unit}_compute'}, '{unit}_weight_write': {'valid': '{unit}_weight_write'},
    '{unit}_acc_write': {'valid': '{unit}_acc_write', 'fields': {'row': '{unit}_acc_write_row'}, 'row': 'row'},
    '{unit}_acc_read': {'valid': '{unit}_acc_read', 'fields': {'row': '{unit}_acc_read_row'}, 'row': 'row'},
    '{unit}_acc_load': {'valid': '{unit}_acc_load', 'fields': {'row': '{unit}_acc_load_row'}, 'row': 'row'},
    '{unit}_acc_store': {'valid': '{unit}_acc_store', 'fields': {'row': '{unit}_acc_store_row'}, 'row': 'row'},}

CORE_CUTS = {'instruction': {'path': 'scalar', 'value': 's1_instr'}, 'fire': {'path': 'scalar', 'value': 's1_fire'},
    'address': {'instance': 'scalar/alu', 'result': 'io_out'}, 'host_start': {'path': 'scalar', 'value': 'hostStart'},
    'dma_size': {'instance': 'scalar', 'result': 'io_dmaCmd_bits_size'},
    'dma_vmem_addr': {'instance': 'scalar', 'result': 'io_dmaCmd_bits_vmemAddr'},
    'dma_dram_addr': {'instance': 'scalar', 'result': 'io_dmaCmd_bits_addr'},
    'dma_vmem_read': {'instance': 'dma', 'result': 'io_vmemRead_valid'},
    'dma_vmem_read_bank': {'instance': 'dma', 'result': 'io_vmemRead_bits_bankIdx'},
    'dma_vmem_read_line': {'instance': 'dma', 'result': 'io_vmemRead_bits_bankAddr'},
    'dma_vmem_write': {'instance': 'dma', 'result': 'io_vmemWrite_valid'},
    'dma_vmem_write_bank': {'instance': 'dma', 'result': 'io_vmemWrite_bits_bankIdx'},
    'dma_vmem_write_line': {'instance': 'dma', 'result': 'io_vmemWrite_bits_bankAddr'},}

DRAM = {'dram': {'request': {'valid': 'io_dmaTL_a_valid', 'ready': 'io_dmaTL_a_ready', 'tag': 'io_dmaTL_a_bits_source',
                      'fields': {'address': 'io_dmaTL_a_bits_address', 'opcode': 'io_dmaTL_a_bits_opcode'}},
          'response': {'valid': 'io_dmaTL_d_valid', 'ready': 'io_dmaTL_d_ready', 'tag': 'io_dmaTL_d_bits_source'},
          'latency': 43, 'accept': [1]}}

IDLE_INPUTS = ("instruction", "fire", "address", "host_start", "dma_size", "dma_vmem_addr", "dma_dram_addr",
               "dma_vmem_read", "dma_vmem_read_bank", "dma_vmem_read_line", "dma_vmem_write", "dma_vmem_write_bank", "dma_vmem_write_line")
TL_FIELDS = {"a": "opcode param size source address mask data corrupt", "d": "opcode param size source sink denied data corrupt"}
for interface in ("imem", "csr", "dma", "vmem"):
    request, response = ("d", "a") if interface == "dma" else ("a", "d")
    prefix = f"io_{interface}TL_"
    if interface == "dma":
        IDLE_INPUTS += (prefix + response + "_ready",)
    IDLE_INPUTS += (prefix + request + "_valid",)
    IDLE_INPUTS += tuple(prefix + request + "_bits_" + field for field in TL_FIELDS[request].split())
    if interface != "dma":
        IDLE_INPUTS += (prefix + response + "_ready",)


CORE_COMMANDS = [('vload', 0x00000287, {'address': 65536}), ('vload_bank2', 0x00001507, {'address': 139264}),
    ('vload_bank5', 0x00001507, {'address': 328192}), ('vstore', 0x00002487, {'address': 131072}),
    ('vstore_bank1', 0x00002887, {'address': 73728}), ('vstore_bank3', 0x00002887, {'address': 196864}),
    ('vstore_m5', 0x00002287, {'address': 196864}), ('vstore_m34', 0x00003107, {'address': 131072}),
    ('transpose', 0x000063eb, {}), ('add', 0x00304557, {}), ('exp', 0x84304557, {}), ('rmax', 0x4c304557, {}),
    ('rsum', 0x42304557, {}), ('csum', 0x02304557, {}), ('fp8unpack', 0x8a304557, {}), ('vli_all', 0x0000055f, {}),
    ('add_hi', 0x00b24d57, {}), ('exp_hi', 0x84b24d57, {}), ('exp2_hi', 0x86b24d57, {}), ('relu_hi', 0x90b24d57, {}),
    ('mxu0_matmul', 0x14018077, {}), ('mxu0_matmul_m14', 0x1401c077, {}), ('mxu0_matmul_m14_slot1', 0x1409c077, {}),
    ('mxu0_matmul_m14_acc1', 0x1401c0f7, {}), ('mxu0_matmul_m14_acc1_slot1', 0x1409c0f7, {}),
    ('mxu0_matmul_acc_m14', 0x1801c077, {}), ('mxu0_matmul_acc_m14_acc1', 0x1801c0f7, {}),
    ('mxu0_push_weight', 0x00018077, {}), ('mxu0_push_acc_fp8', 0x04018077, {}), ('mxu0_pop_acc_bf16', 0x10000a77, {}),
    ('mxu0_pop_acc_bf16_acc1', 0x10080a77, {}), ('mxu1_matmul', 0x16018077, {}), ('mxu1_matmul_m14', 0x1601c077, {}),
    ('mxu1_matmul_m14_slot1', 0x1609c077, {}), ('mxu1_matmul_m14_acc1', 0x1601c0f7, {}),
    ('mxu1_matmul_m14_acc1_slot1', 0x1609c0f7, {}), ('mxu1_matmul_acc_m14', 0x1a01c077, {}),
    ('mxu1_matmul_acc_m14_acc1', 0x1a01c0f7, {}), ('mxu1_push_weight', 0x02018077, {}),
    ('mxu1_push_acc_fp8', 0x06018077, {}), ('mxu1_pop_acc_bf16', 0x12000a77, {}),
    ('mxu1_pop_acc_bf16_acc1', 0x12080a77, {}), ('mxu0_push_acc_bf16', 0x08018077, {}),
    ('mxu0_pop_acc_fp8', 0x0c000a77, {}), ('mxu0_matmul_m5', 0x1400a077, {}), ('mxu0_push_weight_m7', 0x0000e077, {}),
    ('mxu0_push_acc_bf16_m10', 0x08014077, {}), ('mxu0_pop_acc_bf16_m8', 0x10000477, {}),
    ('mxu0_pop_acc_bf16_m2', 0x10000177, {}), ('mxu0_pop_acc_fp8_m3', 0x0c0001f7, {}),
    ('mxu0_pop_acc_bf16_m4', 0x10000277, {}), ('mxu0_matmul_m20', 0x14028077, {}),
    ('mxu0_pop_acc_bf16_m12', 0x10000677, {}), ('mxu0_matmul_m44', 0x14058077, {}),
    ('mxu1_push_acc_bf16', 0x0a018077, {}), ('mxu1_pop_acc_fp8', 0x0e000a77, {}), ('mxu1_matmul_m5', 0x1600a077, {}),
    ('mxu1_push_weight_m7', 0x0200e077, {}), ('mxu1_push_acc_bf16_m10', 0x0a014077, {}),
    ('mxu1_pop_acc_bf16_m8', 0x12000477, {}), ('mxu1_pop_acc_bf16_m2', 0x12000177, {}),
    ('mxu1_pop_acc_fp8_m3', 0x0e0001f7, {}), ('mxu1_pop_acc_bf16_m4', 0x12000277, {}),
    ('mxu1_matmul_m20', 0x16028077, {}), ('mxu1_pop_acc_bf16_m12', 0x12000677, {}), ('mxu1_matmul_m44', 0x16058077, {}),
    ('mxu1_push_weight_m14', 0x0201c077, {}), ('mxu1_push_acc_fp8_m14', 0x0601c077, {}),
    ('mxu1_push_acc_bf16_m14', 0x0a01c077, {}), ('mxu1_pop_acc_fp8_m22', 0x0e000b77, {}),
    ('mxu1_pop_acc_bf16_m22', 0x12000b77, {}), ('vstore_m20', 0x00002a07, {'address': 131072}),
    ('vstore_m44', 0x00003607, {'address': 131072}), ('vload_m12', 0x00000607, {'address': 65536}),
    ('vload_m20', 0x00000a07, {'address': 65536}), ('vload_m52', 0x00001a07, {'address': 65536}),
    ('vload_m44', 0x00001607, {'address': 65536}), ('vstore_m52', 0x00003a07, {'address': 131072}),
    ('exp_m20', 0x84328c57, {}), ('exp_m44', 0x84359757, {}), ('vli_all_m12', 0x0000065f, {}),
    ('transpose_m20', 0x00028c6b, {}), ('transpose_to_m12', 0x0000666b, {}), ('mxu0_matmul_acc_m12', 0x18018077, {}),
    ('transpose_to_m9', 0x000064eb, {}), ('vli_all_m8', 0x0000045f, {}),]

DMA_COMMANDS = [('dma_load0', 0x0020837b, {'dma_vmem_addr': 196608, 'dma_dram_addr': 2415919104, 'dma_size': 1024}),
    ('dma_load1', 0x0020937b, {'dma_vmem_addr': 262144, 'dma_dram_addr': 2415923200, 'dma_size': 1024}),
    ('dma_load1_bank3', 0x0020937b, {'dma_vmem_addr': 197120, 'dma_dram_addr': 2415923200, 'dma_size': 1024}),
    ('dma_store0', 0x022401fb, {'dma_vmem_addr': 196608, 'dma_dram_addr': 2415927296, 'dma_size': 1024}),
    ('dma_store1', 0x022411fb, {'dma_vmem_addr': 262144, 'dma_dram_addr': 2415931392, 'dma_size': 1024}),
    ('dma_store1_bank3', 0x022411fb, {'dma_vmem_addr': 197120, 'dma_dram_addr': 2415931392, 'dma_size': 1024}),
    ('vload', 0x00000287, {'address': 65536}), ('vload_bank3', 0x00001507, {'address': 204800}),
    ('vstore', 0x00002487, {'address': 131072}), ('vstore_bank3', 0x00002887, {'address': 196864}),
    ('vstore_bank3_tile', 0x00002887, {'address': 196608}), ('transpose', 0x000063eb, {}), ('add', 0x00304557, {}),
    ('mxu0_matmul', 0x14018077, {}), ('mxu0_matmul_acc', 0x18018077, {}), ('mxu0_push_weight', 0x00018077, {}),
    ('mxu0_push_acc_fp8', 0x04018077, {}), ('mxu0_push_acc_bf16', 0x08018077, {}), ('mxu0_pop_acc_fp8', 0x0c000a77, {}),
    ('mxu0_pop_acc_bf16', 0x10000a77, {}), ('mxu1_matmul', 0x16018077, {}), ('mxu1_matmul_acc', 0x1a018077, {}),
    ('mxu1_push_weight', 0x02018077, {}), ('mxu1_push_acc_fp8', 0x06018077, {}), ('mxu1_push_acc_bf16', 0x0a018077, {}),
    ('mxu1_pop_acc_fp8', 0x0e000a77, {}), ('mxu1_pop_acc_bf16', 0x12000a77, {}),]


SYSTEM_KEYS = {'core': ('engine', 'module', 'description', 'limit', 'unreset', 'inputs', 'cuts', 'probes', 'busy', 'events', 'operations', 'invariants'), 'core_dma': ('engine', 'module', 'description', 'limit', 'unreset', 'inputs', 'cuts', 'responders', 'probes', 'busy', 'events', 'operations', 'invariants')}

def core_system(dma=False):
    spec = copy.deepcopy(HEADERS["system/core_dma" if dma else "system/core"])
    spec["inputs"] = {name: 0 for name in IDLE_INPUTS if not dma or name not in {"dma_vmem_read", "dma_vmem_write", "dma_vmem_read_bank", "dma_vmem_read_line", "dma_vmem_write_bank", "dma_vmem_write_line"}}
    spec["inputs"]["instruction"] = 19
    spec["cuts"] = {k: v for k, v in CORE_CUTS.items() if not dma or k not in {"dma_vmem_read", "dma_vmem_write", "dma_vmem_read_bank", "dma_vmem_read_line", "dma_vmem_write_bank", "dma_vmem_write_line"}}
    spec["probes"] = copy.deepcopy(CORE_PROBES)
    spec["events"] = copy.deepcopy(CORE_EVENTS)
    if dma:
        first = list(CORE_PROBES)[:3]
        spec["probes"] = {**{k: CORE_PROBES[k] for k in first}, **DMA_PROBES,
                          **{k: v for k, v in CORE_PROBES.items() if k not in first}}
        spec["events"].update(DMA_EVENTS)
    spec["unreset"] = [{"path": "lsu", "name": n} for n in ("vloadVmemBase", "vstoreMregId", "vstoreVmemBase")]
    for unit in (0, 1):
        path = f"mxu{unit}/seq"
        names = [f"inflight_{i}_{field}" for i in range(3 if unit == 0 else 2) for field in ("accSel", "rowsWritten")]
        for i in range(2):
            names += [f"p{i}Cmd_{field}" for field in
                      (("mregId", "op") if unit == 0 else ("accSel", "mregId", "op", "weightSlot"))] + [f"p{i}Row"]
        names += ["w0Cmd_op", "w0Row", "w1Row"]
        spec["unreset"] += [{"path": path, "name": n} for n in names]
        for section, template in (("probes", MXU_PROBES), ("events", MXU_EVENTS)):
            spec[section].update(json.loads(json.dumps(template).replace("{unit}", f"mxu{unit}")))
    spec["unreset"].append({"path": "xlu", "name": "activeCmd_srcMregId"})
    if dma:
        spec["unreset"] += [{"path": "dma", "match": "commandQueue_[0-7]_(opType|channelId|vmemLineAddr|transferSize)"},
                            {"path": "dma", "match": "sourceMeta_[0-9]+_(isLoad|slotIdx|lineAddr)"}]
        spec["responders"] = DRAM
    spec["operations"] = {op: {"commands": {0: {"fire": 1, "instruction": instruction, **fields}}}
                          for op, instruction, fields in (DMA_COMMANDS if dma else CORE_COMMANDS)}
    return copy.deepcopy({k: spec[k] for k in SYSTEM_KEYS["core_dma" if dma else "core"]})


def generate(name, facts=None, sweep=None):
    name = str(name).removesuffix(".yaml")
    if name + ".yaml" not in GENERATED:
        raise ValueError(f"Unknown generated spec: {name}")
    if name.startswith("system/"):
        return core_system(name == "system/core_dma")
    if name == "pairs/core_vpu":
        raw = yaml.safe_load(vpu_pairs(timing() if facts is None else facts, sweep))
    elif name in ("pairs/core_mxu_same", "pairs/core_mxu_shared"):
        raw = yaml.safe_load(mxu_pairs(timing() if facts is None else facts, sweep, shared=name.endswith("shared")))
    elif name == "sequences/core_mxu":
        raw = yaml.safe_load(mxu_chains(sweep))
    elif name == "sequences/core_vpu":
        raw = {"sequences": {key: {"description": description,
               "ops": [{"op": op, **({"inputs": {"io_cmd_bits_vs1": 2 + 16*i,
                        "io_cmd_bits_vs2": 6 + 16*i, "io_cmd_bits_vd": 10 + 16*i}} if i else {})}
                       for i, op in enumerate(ops)], "lengths": "3..4", "gaps": "1..135"}
               for key, description, ops in VPU_CHAINS}}
    else:
        rows = enumerated_pairs(name)
        raw = {"pairs": {key: {"first": first, "second": second, "gaps": f"1..{limit}"}
                         for key, first, second, limit in rows}}
    raw = {**copy.deepcopy(HEADERS[name]), **{k: v for k, v in raw.items() if k in ("pairs", "sequences", "expect")}}
    if sweep is None:
        prefix = name.replace("/", "_").upper()
        expected = globals()[prefix + "_EXPECT"]
        indices = globals()[prefix + "_REGRESSIONS"].split()
        keys = raw.get("pairs", raw.get("sequences"))
        if len(keys) != len(indices):
            raise ValueError(f"regression table length differs: {name}")
        raw["expect"] = {key: expected[int(index)].copy() for key, index in zip(keys, indices)}
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=GENERATED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    text = yaml.safe_dump(generate(args.name), sort_keys=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

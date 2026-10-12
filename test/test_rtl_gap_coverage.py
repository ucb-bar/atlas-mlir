"""Compiler issue gaps never undercut the RTL pair sweeps (rtl_extract.op_pairs.v1).

For each pair record, a program issues the pair's two operations with the pair file's operand
overlays (tools/rtl_extract/targets/atlas/pairs), separated by DELAY, and --verify-atlas-timing
finds the smallest accepted issue gap; it must be at least the record's stable_gap. Pairs the
selected provider rejects are reported as unsupported. An informational legacy run lists pairs
whose built-in rules (insert-atlas-delays without a selection) fall below stable_gap.

ATLAS_PAIR_SWEEPS lists sweep files (os.pathsep-separated); the default is every pairs-*.json next to
ATLAS_OP_TIMING. Missing files are skipped. With ATLAS_PAIR_FACTS the pair facts are selected too. System pairs
(engines `core` and `core_dma`) feed instruction words to AtlasCore's decoder; they are decoded into compiler
operations that must encode to the same words, with register fields fed by cuts (VLS base, DMA operands) set up
from the cut values.

A DMA transfer's beats are not compiler facts: its effects are unknown from launch to the matching wait. For a
pair with a DMA launch, a swept gap whose only failure is a moved DMA event (the transfer queued behind another
launch, or its VMEM accesses waiting for the LSU's grant) keeps every other event isolated, so the bound the
compiler must meet is the gap after the last other failure. Two launches never keep isolated timing (the second
is queued), so they bind launch order only. Any firing assertion binds as stable_gap does.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import unittest

from test_delay_insertion import EMIT, OPT, ROOT, addi, nop, program, run, stream
from test_rtl_timing import FACTS, HALT, PAIRS, PROVIDER, VERIFY, delay, selection, vls, word_base
from test_rtl_vmul_timing import pack, reduce, unary, vli, vmul
from test_rtl_xlu_timing import transpose
from test_rtl_dma_timing import IN_FLIGHT, transfer, wait

sys.path.insert(0, str(ROOT / "tools"))

LIMIT = 400  # widest gap probed; every pair here drains well before it
BINARY = {"add": "add", "sub": "sub", "mul": "mul", "pairmax": "max", "pairmin": "min"}
UNARY = {"mov": "mov", "rcp": "recip", "log": "log2",
         **{k: k for k in ("exp", "exp2", "relu", "sin", "cos", "tanh", "sqrt", "square", "cube")}}
REDUCE = {"rsum": "row_sum", "rmax": "row_max", "rmin": "row_min",
          "csum": "col_sum", "cmax": "col_max", "cmin": "col_min"}
VLI = {"vliAll": "all", "vliRow": "row", "vliCol": "col", "vliOne": "one"}
MXU = {0: "push_weight", 2: "push_acc_fp8", 4: "push_acc_bf16", 6: "pop_acc_fp8", 8: "pop_acc_bf16",
       10: "matmul", 12: "matmul_acc"}  # funct7 - unit (lib/AtlasEncoding.cpp)
DMA_FIELDS = (31 << 7) | (31 << 15) | (31 << 20)  # DMA register fields; the operands arrive on the cuts


def sweep_paths():
    listed = os.environ.get("ATLAS_PAIR_SWEEPS")
    if listed:
        return [Path(p) for p in listed.split(os.pathsep) if p]
    if not FACTS:
        return []
    return sorted(Path(FACTS).resolve().parent.glob("pairs-*.json"))


def records():
    found = []
    for path in sweep_paths():
        if path.is_file():
            document = json.loads(path.read_text())
            if document.get("schema") != "rtl_extract.op_pairs.v1":
                raise AssertionError(f"{path}: expected schema rtl_extract.op_pairs.v1")
            found += [(path.name, r) for r in document["op_pairs"]]
    return found


def pair_files():
    from rtl_extract import pairs
    return pairs, pairs.load("atlas")


def operation(engine, name, fields, base):
    """(setup, op) for one RTL command, or None when the operation has no compiler mapping."""
    if engine == "vlsu" and name in ("vload", "vstore"):
        return word_base(base, fields["io_cmd_bits_vmemLineAddr"] * 8), vls(name, fields["io_cmd_bits_mregBank"], base)
    if engine == "vpu":
        vs1, vs2, vd = (fields[f"io_cmd_bits_{k}"] for k in ("vs1", "vs2", "vd"))
        if name in BINARY:
            return [], vmul(vs1, vs2, vd, BINARY[name])
        if name in UNARY:
            return [], unary(UNARY[name], vs1, vd)
        if name in REDUCE:
            return [], reduce(REDUCE[name], vs1, vd)
        if name in VLI:
            return [], vli(VLI[name], vd)
    if engine in ("mxu0", "mxu1"):
        unit, src = engine[-1], fields["io_cmd_bits_mregId"]
        acc, slot = fields.get("io_cmd_bits_accSel", 0), fields.get("io_cmd_bits_weightSlot", 0)
        if name in ("matmul", "matmul_acc"):
            return [], ("mxu_matmul", f"unit = {unit} : i32, src = {src} : i32, weight_slot = {slot} : i32, "
                                      f"acc_slot = {acc} : i32, accumulate = {str(name == 'matmul_acc').lower()}")
        if name in ("push_weight", "push_acc_fp8", "push_acc_bf16"):
            kind, at = ("weight_fp8", slot) if name == "push_weight" else (name[5:], acc)
            return [], ("mxu_push", f'kind = "{kind}", unit = {unit} : i32, src = {src} : i32, slot = {at} : i32')
        if name in ("pop_acc_bf16", "pop_acc_fp8"):
            return [], ("mxu_pop", f'format = "{name[8:]}", unit = {unit} : i32, dst = {src} : i32, '
                                   f'slot = {acc} : i32, scale_reg = 0 : i32')
    return None


def core_operation(fields, base):
    """(setup, op) for an instruction word entering AtlasCore's decoder (system pairs), or None. The VLS
    word address arrives on the ALU cut and DMA operands on the register-file cuts; their registers are
    materialized from the cut values (VMEM, DRAM and size in 2*base..2*base+2 for DMA)."""
    import yaml
    word, opcode = fields["instruction"], fields["instruction"] & 127
    vd, vs1, vs2 = (word >> 7) & 63, (word >> 13) & 63, (word >> 19) & 63
    if opcode == 0x07 and word >> 20 == 0:
        kind = "vstore" if word & (1 << 13) else "vload"
        return word_base(base, fields["address"]), vls(kind, vd, base)
    if opcode == 0x6b:
        return [], transpose(vs1, vd)
    if opcode == 0x77 and (word >> 25) & ~1 in MXU:
        name, unit = MXU[(word >> 25) & ~1], (word >> 25) & 1
        pop = name.startswith("pop")
        return operation(f"mxu{unit}", name, {"io_cmd_bits_mregId": vd if pop else vs1,
                                              "io_cmd_bits_accSel": vs2 if pop else vd,
                                              "io_cmd_bits_weightSlot": vd if name == "push_weight" else vs2}, base)
    if opcode == 0x7b and word >> 26 == 0:
        vmem, dram, size = 2 * base, 2 * base + 1, 2 * base + 2
        setup = [*word_base(vmem, fields["dma_vmem_addr"]), *word_base(dram, fields["dma_dram_addr"]),
                 *word_base(size, fields["dma_size"])]
        return setup, transfer("store" if word >> 25 else "load", (word >> 12) & 7, vmem, dram, size)
    words = yaml.safe_load((ROOT / "tools/rtl_extract/targets/atlas/vpu.yaml").read_text())["decode"]["words"]
    name = next((n for n, w in words.items() if w & 127 == opcode and
                 (w >> 25 == word >> 25 if opcode == 0x57 else (w >> 12) & 7 == (word >> 12) & 7)), None)
    if name in ("fp8unpack", "fp8pack"):
        return [], pack("fp8_to_bf16" if name == "fp8unpack" else "bf16_to_fp8", vs2, vd, vs1)
    if name in VLI:
        return [], vli(VLI[name], vd, word >> 15)
    return operation("vpu", name, {"io_cmd_bits_vs1": vs1, "io_cmd_bits_vs2": vs2, "io_cmd_bits_vd": vd}, base) \
        if name else None


def is_dma(op):
    return op[0] == "dma"


def channel(op):
    return int(op[1].split("channel = ")[1].split(" ")[0])


def waits(*ops):
    """The waits that retire the pair's DMA transfers, so the program ends with none pending."""
    return [wait(channel(op)) for op in ops if is_dma(op)]


def encodes(setup, op, word, ignored):
    """The compiler encodes `op` as the RTL instruction word outside the `ignored` bits."""
    emitted = run(EMIT, program([*setup, op, *waits(op), HALT]))
    return emitted.returncode == 0 and (int(emitted.stdout.split()[-2 - len(waits(op))], 16) ^ word) & ~ignored == 0


def pair_program(record, source=""):
    """(setup, first, second, tail) for a record, from its pair file's commands and overlays; `tail` waits
    for the pair's DMA transfers.

    The pair file is the one named by the sweep file (pairs-<file>.json), else the record's engine.
    """
    pairs, files = pair_files()
    engine, key = record["engine"], record["name"].split(".", 1)[-1]
    stem = source.removeprefix("pairs-").removesuffix(".json")
    raw, spec = files.get(stem if stem in files else engine, ({"pairs": {}}, None))
    pair = raw["pairs"].get(key)
    if not pair or (pair["first"], pair["second"]) != (record["first"], record["second"]):
        return None
    sides = []
    for side, base in (("first", 6), ("second", 8)):
        overlay = record.get(f"{side}_inputs", pair.get(f"{side}_inputs", {}))
        commands = pairs.commands(spec, pair[side], overlay)
        if list(commands) != [0]:
            return None
        fields = {**spec["inputs"], **commands[0]}
        if engine not in ("core", "core_dma"):
            sides.append(operation(engine, pair[side], fields, base))
            continue
        built, word = core_operation(fields, base), fields["instruction"]
        # A VLS names its base register where the cut feeds x0, a DMA its operand registers; single-input VPU
        # ops do not read vs2 (VectorFSM opReadsSecondaryPair).
        ignored = (31 << 15 if word & 127 == 0x07 else DMA_FIELDS if word & 127 == 0x7b else
                   63 << 19 if built and built[1][0] in ("vpu_unary", "vpu_reduce") else 0)
        sides.append(built if built and encodes(*built, word, ignored) else None)
    if None in sides:
        return None
    return sides[0][0] + sides[1][0], sides[0][1], sides[1][1], waits(sides[0][1], sides[1][1])


def options(tail):
    """The facts selection (with ATLAS_PAIR_FACTS, the pair facts too), with the DMA policy when the pair launches a transfer."""
    return selection(dma="wait" if tail else None, pairs=PAIRS)


def bound(record):
    """(gap the compiler must meet, kind) for a record; see the module docstring for DMA pairs."""
    if record["engine"] != "core_dma":
        return record["stable_gap"], "stable"
    moved = lambda r: set(r) == {"gap", "accepted", "events"} and r["events"].split()[1].startswith(("dma_", "dram."))
    binding = [r["gap"] for r in record["rejected"] if not moved(r)]
    if not binding:
        return 1, "order" if record["stable_gap"] is None else "dma"
    if max(binding) >= record["gaps"][1] or record["stable_gap"] is None:
        raise AssertionError(f"{record['name']}: no swept gap clears every non-DMA failure")
    return max(binding) + 1, "dma"


def separated(gap):
    return [] if gap == 1 else [nop()] if gap == 2 else [delay(gap - 2)]


def gapped(setup, first, second, gap, tail=()):
    return program([*setup, first, *separated(gap), second, *tail, delay(LIMIT), nop(), HALT])


def issue_cycles(printed):
    """(op, sorted fields, issue cycle) for each printed stream op."""
    out, cycle = [], 0
    for op, fields, _ in stream(printed):
        out.append((op, sorted(fields.split(", ")), cycle))
        cycle += int(fields.split("cycles = ")[1].split(" ")[0]) + 1 if op == "delay" else 1
    return out


def issue_gap(printed, first, second):
    """Issue cycle of `second` minus that of `first` (negative if reordered)."""
    ops = issue_cycles(printed)
    find = lambda op, skip=-1: next(i for i, (name, fields, _) in enumerate(ops)
                                    if i != skip and (name, fields) == (op[0], sorted(op[1].split(", "))))
    a = find(first)
    return ops[find(second, a)][2] - ops[a][2]


def selected_gap(setup, first, second, tail, limit):
    """Smallest gap accepted by --verify-atlas-timing under the selected provider, None if none up to `limit`, or the rejection."""
    option = options(tail)
    widest = run(OPT, gapped(setup, first, second, limit, tail), option, VERIFY)
    if widest.returncode:
        return None, widest.stderr.strip().splitlines()[0] if widest.stderr else "rejected"
    accepts = lambda g: run(OPT, gapped(setup, first, second, g, tail), option, VERIFY).returncode == 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(1, limit + 1, 16):
            batch = range(start, min(start + 16, limit + 1))
            accepted = list(pool.map(accepts, batch))
            if True in accepted:
                return batch[accepted.index(True)], None
    return limit, None


def consumer_gap(setup, first, second, tail, *selection):
    result = run(OPT, program([*setup, first, second, *tail, HALT]), *selection, "--insert-atlas-delays")
    return issue_gap(result.stdout, first, second) if result.returncode == 0 else None


@unittest.skipUnless(FACTS and OPT.is_file(), "requires explicit RTL timing facts and built Atlas tools")
class RTLGapCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = records()
        if not cls.records:
            raise unittest.SkipTest("no pair sweep files (ATLAS_PAIR_SWEEPS)")

    def test_selected_gaps_cover_stable_gaps(self):
        rows, unsupported, unmapped = [], [], []
        for source, record in self.records:
            built = pair_program(record, source)
            if not built:
                unmapped.append(record["name"])
                continue
            stable, (required, kind) = record["stable_gap"], bound(record)
            gap, why = selected_gap(*built, max(LIMIT // 2, required + 1))
            if gap is None:
                unsupported.append((record["name"], why))
                continue
            rows.append((record["name"], stable, required, kind, gap, consumer_gap(*built, options(built[3]))))
            with self.subTest(pair=record["name"], source=source):
                self.assertGreaterEqual(gap, required, f"compiler accepts gap {gap} < RTL {kind} bound {required}")
        print("\nselected compiler gap vs RTL bound (stable_gap, or for DMA pairs the gap after the last non-DMA "
              "failure; slack = compiler - bound; delays = insert-atlas-delays gap)")
        print(f"  {'pair':40} {'stable':>6} {'bound':>5} {'kind':>6} {'compiler':>8} {'slack':>5} {'delays':>6}")
        for name, stable, required, kind, gap, delays in rows:
            print(f"  {name:40} {str(stable):>6} {required:5} {kind:>6} {gap:8} {gap - required:5} {str(delays):>6}")
        for name, why in unsupported:
            print(f"  {name:40} unsupported: {why}")
        self.assertFalse(unmapped, f"pair records without a compiler operand mapping: {unmapped}")
        self.assertTrue(rows or unsupported)

    def test_legacy_rules_below_stable_gap_are_listed(self):
        print("\nlegacy (unselected) insert-atlas-delays gap vs RTL stable_gap (informational; '<' marks below)")
        below = []
        for source, record in self.records:
            built = pair_program(record, source)
            gap = built and consumer_gap(*built)
            required = bound(record)[0]
            low = gap is not None and gap < required
            below += [record["name"]] if low else []
            print(f"  {record['name']:40} bound {required:4} legacy {str(gap):>4} {'<' if low else ''}")
        print(f"  below stable_gap: {below or 'none'}")

    def test_unsupported_pairs_are_provider_rejections(self):
        # A rejection at the widest gap must come from the selected provider, not from a malformed program.
        for source, record in self.records:
            built = pair_program(record, source)
            if built:
                result = run(OPT, gapped(*built[:3], LIMIT, built[3]), options(built[3]), VERIFY)
                if result.returncode:
                    with self.subTest(pair=record["name"]):
                        # Pending-transfer conflicts come from the provider's DMA rule through the common verifier.
                        self.assertTrue(PROVIDER in result.stderr or IN_FLIGHT in result.stderr, result.stderr)


if __name__ == "__main__":
    unittest.main()

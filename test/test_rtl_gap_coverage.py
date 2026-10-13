"""Compiler issue gaps never undercut the RTL pair sweeps.

For each record, a program issues the pair's two operations with the pair file's operand overlays,
separated by DELAY; the smallest gap --verify-atlas-timing accepts under the selected provider must be
at least the record's bound. System pairs (engines `core` and `core_dma`) are decoded from AtlasCore
instruction words and must re-encode to them. Unsupported pairs must be provider rejections, and an
informational run lists pairs whose legacy rules fall below the bound.

The bound is stable_gap, except for a pair with a DMA launch: a transfer's beats are not compiler facts,
so a swept gap whose only failure is a moved DMA or DRAM event does not bind. Its bound is one past the
last other failing gap (a firing assertion always binds), or 1 when there is none, so a VLS command
beside a pending transfer may be bound below its stable_gap. Two launches bind only their order.

ATLAS_PAIR_SWEEPS lists sweep files (os.pathsep-separated; default every pairs-*.json beside
ATLAS_OP_TIMING); with ATLAS_PAIR_FACTS the pair facts are selected too.
"""
from concurrent.futures import ThreadPoolExecutor
import functools
import json
import os
from pathlib import Path
import unittest

from rtl_test_support import (
    BINARY, EMIT, FACTS, HALT, IN_FLIGHT, LIMIT, MXU_WORDS, OPT, PAIRS, PROVIDER, REDUCE, ROOT,
    UNARY, VERIFY, VLI, gapped, issue_gap, mxu, pack, program, run, selection, transfer,
    transpose, vli, vls, vpu, wait, word_base, pair_spec, decode_mxu,
)


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

@functools.lru_cache(maxsize=None)
def pair_files():
    from rtl_extract import pairs
    return pairs, pairs.load("atlas")

def operation(engine, name, fields, base):
    if engine == "vlsu" and name in ("vload", "vstore"):
        return word_base(base, fields["io_cmd_bits_vmemLineAddr"] * 8), vls(name, fields["io_cmd_bits_mregBank"], base)
    if engine == "vpu":
        vs1, vs2, vd = (fields[f"io_cmd_bits_{k}"] for k in ("vs1", "vs2", "vd"))
        if name in BINARY or name in UNARY or name in REDUCE or name in VLI:
            return [], vpu(name, (vs1, vs2, vd))
    if engine in ("mxu0", "mxu1") and name in MXU_WORDS.values():
        return [], mxu(name, int(engine[-1]), reg=fields["io_cmd_bits_mregId"],
                       acc=fields.get("io_cmd_bits_accSel", 0), slot=fields.get("io_cmd_bits_weightSlot", 0),
                       scale=fields.get("scale", 0))
    return None

def core_operation(fields, base):
    word, opcode = fields["instruction"], fields["instruction"] & 127
    vd, vs1, vs2 = (word >> 7) & 63, (word >> 13) & 63, (word >> 19) & 63
    if opcode == 0x07 and word >> 20 == 0:
        kind = "vstore" if word & (1 << 13) else "vload"
        return word_base(base, fields["address"]), vls(kind, vd, base)
    if opcode == 0x6b:
        return [], transpose(vs1, vd)
    if opcode == 0x77 and (word >> 25) & ~1 in MXU_WORDS:
        return [], decode_mxu(word)
    if opcode == 0x7b and word >> 26 == 0:
        vmem, dram, size = 2 * base, 2 * base + 1, 2 * base + 2
        setup = [*word_base(vmem, fields["dma_vmem_addr"]), *word_base(dram, fields["dma_dram_addr"]),
                 *word_base(size, fields["dma_size"])]
        return setup, transfer("store" if word >> 25 else "load", (word >> 12) & 7, vmem, dram, size)
    words = pair_spec(ROOT / "tools/rtl_extract/targets/atlas/vpu.yaml")["decode"]["words"]
    name = next((n for n, w in words.items() if w & 127 == opcode and
                 (w >> 25 == word >> 25 if opcode == 0x57 else (w >> 12) & 7 == (word >> 12) & 7)), None)
    if name in ("fp8unpack", "fp8pack"):
        return [], pack("fp8_to_bf16" if name == "fp8unpack" else "bf16_to_fp8", vs2, vd, vs1)
    if name in VLI:
        return [], vli(VLI[name], vd, word >> 15)
    return operation("vpu", name, {"io_cmd_bits_vs1": vs1, "io_cmd_bits_vs2": vs2, "io_cmd_bits_vd": vd}, base) \
        if name else None

def waits(*ops):
    return [wait(int(op[1].split("channel = ")[1].split(" ")[0])) for op in ops if op[0] == "dma"]

def encodes(setup, op, word, ignored):
    emitted = run(EMIT, program([*setup, op, *waits(op), HALT]))
    return emitted.returncode == 0 and (int(emitted.stdout.split()[-2 - len(waits(op))], 16) ^ word) & ~ignored == 0

def pair_program(record, source=""):
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
        ignored = (31 << 15 if word & 127 == 0x07 else DMA_FIELDS if word & 127 == 0x7b else
                   63 << 19 if built and built[1][0] in ("vpu_unary", "vpu_reduce") else 0)
        sides.append(built if built and encodes(*built, word, ignored) else None)
    if None in sides:
        return None
    return sides[0][0] + sides[1][0], sides[0][1], sides[1][1], waits(sides[0][1], sides[1][1])

def options(tail):
    return selection(dma="wait" if tail else None, pairs=PAIRS)

def bound(record):
    if record["engine"] != "core_dma":
        return record["stable_gap"], "stable"
    moved = lambda r: set(r) == {"gap", "accepted", "events"} and r["events"].split()[1].startswith(("dma_", "dram."))
    binding = [r["gap"] for r in record["rejected"] if not moved(r)]
    if not binding:
        return 1, "order" if record["stable_gap"] is None else "dma"
    if max(binding) >= record["gaps"][1] or record["stable_gap"] is None:
        raise AssertionError(f"{record['name']}: no swept gap clears every non-DMA failure")
    return max(binding) + 1, "dma"

def selected_gap(setup, first, second, tail, limit):
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
            result = run(OPT, gapped(*built[:3], LIMIT, built[3]), options(built[3]), VERIFY)
            if result.returncode:
                with self.subTest(pair=record["name"]):
                    self.assertTrue(PROVIDER in result.stderr or IN_FLIGHT in result.stderr, result.stderr)
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

if __name__ == "__main__":
    unittest.main()

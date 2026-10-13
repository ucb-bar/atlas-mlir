"""Pair facts (tools/rtl_extract/pairfacts.py): joining sweep records with their pair and sequence files, fail-closed
inputs and the live capacity the runs show.

Synthetic sweep documents stand in for sweeps of the Atlas pair and sequence files. With ATLAS_OP_TIMING set, the
sweeps beside it (pairs-core_vpu.json) and, with ATLAS_VPU_SEQUENCES, the VPU chain sweep build the selected document.
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from rtl_extract import pairfacts

FACTS = os.environ.get("ATLAS_OP_TIMING")
SEQUENCES = os.environ.get("ATLAS_VPU_SEQUENCES")
HW = {"path": "ir.mlir", "sha256": "a" * 64}
CHECKED = {"checked": 7, "excluded": []}


def block(name, next_issue):
    return {"name": name, "engine": name.split(".")[0], "operation": name.split(".")[1],
            "events": {"write": {"first_age": 2, "last_age": next_issue, "count": next_issue - 1}},
            "first_free_age": next_issue, "next_issue_age": next_issue}


OP_TIMING = {"schema": "merlin.op_timing.v1", "hw_ir": HW,
             "op_timing": [block("vpu.exp", 65), block("vpu.relu", 65), block("vpu.sin", 65)]}


def runs(accepted_from, last):
    out = [{"from": 1, "to": accepted_from - 1, "result": "guard set"}] if accepted_from > 1 else []
    return out + [{"from": accepted_from, "to": last, "result": "accepted"}]


def pair(name, first, stable, invariants=CHECKED):
    record = {"name": f"core.{name}", "engine": "core", "module": "AtlasCore", "first": "add", "second": "add_hi",
              "gaps": [1, 71], "min_gap": stable, "stable_gap": stable, "results": runs(stable, 71)}
    return {**record, "invariants": invariants} if invariants else record


def chain(stable, length=3):
    ops = [{"op": "exp"}, {"op": "relu", "inputs": {"io_cmd_bits_vs1": 18, "io_cmd_bits_vs2": 22, "io_cmd_bits_vd": 26}},
           {"op": "exp", "inputs": {"io_cmd_bits_vs1": 34, "io_cmd_bits_vs2": 38, "io_cmd_bits_vd": 42}},
           {"op": "relu", "inputs": {"io_cmd_bits_vs1": 50, "io_cmd_bits_vs2": 54, "io_cmd_bits_vd": 58}}][:length]
    return {"name": f"vpu.exp_relu_exp/{length}", "engine": "vpu", "module": "VectorEngineTop", "sequence": "exp_relu_exp",
            "length": length, "ops": ops, "gaps": [1, 135], "min_gap": stable, "stable_gap": stable,
            "results": runs(stable, 135), "invariants": CHECKED}


class PairFactsTests(unittest.TestCase):
    def build(self, records, chains=(), op_timing=OP_TIMING, hw=HW):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "facts.json").write_text(json.dumps(op_timing))
            (root / "pairs-core_vpu.json").write_text(json.dumps({"schema": "rtl_extract.op_pairs.v1", "hw_ir": hw, "op_pairs": records}))
            sequences = []
            if chains:
                sequences = [root / "sequences-core_vpu.json"]
                sequences[0].write_text(json.dumps({"schema": "rtl_extract.op_sequences.v1", "hw_ir": hw, "op_sequences": list(chains)}))
            return pairfacts.build(root / "facts.json", [root / "pairs-core_vpu.json"], sequences)

    def test_records_name_blocks_and_carry_the_pair_file_relation(self):
        document = self.build([pair("exp_relu", "exp", 1), pair("exp_exp", "exp", 65)])
        self.assertEqual(document["schema"], "merlin.op_pairs.v1")
        self.assertEqual(document["hw_ir"], HW)
        first, second = document["op_pairs"]
        self.assertEqual((first["first"], first["second"], first["engine"]), ("vpu.exp", "vpu.relu", "vpu"))
        self.assertEqual(first["relation"], {"mreg_banks": "disjoint"})
        self.assertEqual((first["stable_gap"], first["gaps"], first["assertions"]), (1, [1, 71], {"checked": 7, "excluded": 0}))
        self.assertEqual(second["source"], "core.exp_exp")
        self.assertNotIn("results", first)

    def test_capacity_is_what_the_runs_show(self):
        self.assertEqual(self.build([pair("exp_exp", "exp", 65)])["capacity"][0]["live"], 1)
        two = self.build([pair("exp_exp", "exp", 65), pair("exp_relu", "exp", 1)])["capacity"]
        self.assertEqual(two, [{"engine": "vpu", "live": 2, "window": "next_issue_age",
                                "witness": {"source": "core.exp_relu", "gap": 1}}])
        # exp, relu, exp at 33: the third issues as the first retires, so two stay live; a run accepted at 1 shows three.
        self.assertEqual(self.build([pair("exp_exp", "exp", 65)], [chain(33)])["capacity"][0]["live"], 2)
        self.assertEqual(self.build([pair("exp_exp", "exp", 65)], [chain(1)])["capacity"][0]["live"], 3)

    def test_inputs_fail_closed(self):
        unchecked = pair("exp_relu", "exp", 1, invariants=None)
        cases = {
            "assertion-checked": [unchecked],
            "does not match": [{**pair("exp_relu", "exp", 1), "first": "sub"}],
            "a second record": [pair("exp_relu", "exp", 1), pair("exp_relu", "exp", 1)],
            "not a resolved op_timing block": [pair("exp_cos", "exp", 1)],
        }
        for message, records in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                self.build(records)
        with self.assertRaisesRegex(ValueError, "different hardware IR"):
            self.build([pair("exp_relu", "exp", 1)], hw={"path": "x", "sha256": "b" * 64})
        unresolved = copy.deepcopy(OP_TIMING)
        unresolved["op_timing"][1]["events"] = None
        with self.assertRaisesRegex(ValueError, "not a resolved op_timing block"):
            self.build([pair("exp_relu", "exp", 1)], op_timing=unresolved)
        wrong = chain(33)
        wrong["ops"][1]["op"] = "sin"
        with self.assertRaisesRegex(ValueError, "does not match sequences/core_vpu.yaml"):
            self.build([pair("exp_relu", "exp", 1)], [wrong])

    def test_written_document_keeps_one_record_per_line(self):
        document = self.build([pair("exp_relu", "exp", 1)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "op-pairs.json"
            pairfacts.write(path, document)
            self.assertEqual(json.loads(path.read_text()), document)
            self.assertIn('    {"first": "vpu.exp", "second": "vpu.relu"', path.read_text())


def mxu_block(name, last):
    """An MXU block: no next_issue_age, live through its last event age."""
    return {"name": name, "engine": name.split(".")[0], "operation": name.split(".")[1],
            "events": {"compute_busy": {"first_age": 1, "last_age": last, "count": last}, "acc_store": {"count": 0}}}


MXU_TIMING = {"schema": "merlin.op_timing.v1", "hw_ir": HW,
              "op_timing": [block("vpu.exp", 65), block("vpu.relu", 65), block("vpu.sin", 65),
                            mxu_block("mxu0.matmul", 94), mxu_block("mxu0.pop_acc_bf16", 32), mxu_block("mxu0.push_weight", 32)]}


def mxu_pair(name, stable, last=100):
    return {"name": f"core.{name}", "engine": "core", "module": "AtlasCore", "first": "mxu0_matmul", "second": "mxu0_matmul",
            "gaps": [1, last], "min_gap": stable, "stable_gap": stable, "results": runs(stable, last), "invariants": CHECKED}


def mxu_chain(stable, length=3):
    return {"name": f"core.mxu0_matmul_other_acc_same_slot/{length}", "engine": "core", "module": "AtlasCore",
            "sequence": "mxu0_matmul_other_acc_same_slot", "length": length,
            "ops": [{"op": "mxu0_matmul", "inputs": {"instruction": w}} for w in (0x14000077, 0x140080f7, 0x14010077)][:length],
            "gaps": [1, 100], "min_gap": stable, "stable_gap": stable, "results": runs(stable, 100), "invariants": CHECKED}


class MxuPairFactsTests(unittest.TestCase):
    """Same-MXU sweeps (pairs/core_mxu_same.yaml, sequences/core_mxu.yaml): relations and the last-event window."""

    def build(self, files, op_timing=MXU_TIMING):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "facts.json").write_text(json.dumps(op_timing))
            paths = {"pairs": [], "sequences": []}
            for n, (stem, records) in enumerate(files):
                kind = "pairs" if stem.startswith("pairs-") else "sequences"
                schema = "rtl_extract.op_pairs.v1" if kind == "pairs" else "rtl_extract.op_sequences.v1"
                path = root / str(n) / f"{stem}.json"
                path.parent.mkdir()
                path.write_text(json.dumps({"schema": schema, "hw_ir": HW, ("op_pairs" if kind == "pairs" else "op_sequences"): records}))
                paths[kind].append(path)
            return pairfacts.build(root / "facts.json", paths["pairs"], paths["sequences"])

    def test_records_carry_accumulator_and_weight_slot_relations(self):
        document = self.build([("pairs-core_mxu_same", [mxu_pair("mxu0_mm_pop16_acc_other", 1), mxu_pair("mxu0_mm_pop16_acc_same", 64),
                                                        mxu_pair("mxu0_mm_pw_slot_other", 1)])])
        relations = {(r["second"], r["source"]): (r["engine"], r["relation"]) for r in document["op_pairs"]}
        self.assertEqual(relations[("mxu0.pop_acc_bf16", "core.mxu0_mm_pop16_acc_other")],
                         ("mxu0", {"mreg_banks": "disjoint", "accumulator": "other"}))
        self.assertEqual(relations[("mxu0.pop_acc_bf16", "core.mxu0_mm_pop16_acc_same")][1]["accumulator"], "same")
        self.assertEqual(relations[("mxu0.push_weight", "core.mxu0_mm_pw_slot_other")][1],
                         {"mreg_banks": "disjoint", "weight_slot": "other"})

    def test_mxu_capacity_counts_commands_live_through_their_last_event(self):
        late = self.build([("pairs-core_mxu_same", [mxu_pair("mxu0_mm_pop16_acc_same", 95)])])["capacity"]
        self.assertEqual(late, [{"engine": "mxu0", "live": 1, "window": "last_event_age",
                                 "witness": {"source": "core.mxu0_mm_pop16_acc_same", "gap": 95}}])
        early = self.build([("pairs-core_mxu_same", [mxu_pair("mxu0_mm_pop16_acc_same", 94)])])["capacity"]
        self.assertEqual((early[0]["live"], early[0]["witness"]["gap"]), (2, 94))
        # Three matmuls alternating accumulators at 32 stay live together; at 48 the third issues after the first.
        pairs = ("pairs-core_mxu_same", [mxu_pair("mxu0_mm_pop16_acc_same", 64)])
        self.assertEqual(self.build([pairs, ("sequences-core_mxu", [mxu_chain(32)])])["capacity"][0]["live"], 3)
        self.assertEqual(self.build([pairs, ("sequences-core_mxu", [mxu_chain(48)])])["capacity"][0]["live"], 2)

    def test_duplicate_records_across_files_fail(self):
        records = [mxu_pair("mxu0_mm_pop16_acc_other", 1)]
        with self.assertRaisesRegex(ValueError, "a second record"):
            self.build([("pairs-core_mxu_same", records), ("pairs-core_mxu_same", records)])

    def test_an_engine_takes_one_window(self):
        mixed = copy.deepcopy(MXU_TIMING)
        mixed["op_timing"][3]["next_issue_age"] = 95
        with self.assertRaisesRegex(ValueError, "take different live windows"):
            self.build([("pairs-core_mxu_same", [mxu_pair("mxu0_mm_pop16_acc_other", 1)])], op_timing=mixed)


@unittest.skipUnless(FACTS and (Path(FACTS).parent / "pairs-core_vpu.json").is_file(),
                     "requires ATLAS_OP_TIMING with pairs-core_vpu.json beside it")
class AtlasPairFactsTests(unittest.TestCase):
    def test_vpu_pairs_and_chains(self):
        sweeps = [Path(FACTS).parent / "pairs-core_vpu.json"]
        document = pairfacts.build(FACTS, sweeps, [SEQUENCES] if SEQUENCES else [])
        records = document["op_pairs"]
        self.assertEqual(len(records), 29 * 29)
        self.assertEqual(sum(r["stable_gap"] == 1 for r in records), 402)
        self.assertTrue(all(r["assertions"]["checked"] == 177 and r["relation"] == {"mreg_banks": "disjoint"} for r in records))
        self.assertEqual([c["live"] for c in document["capacity"] if c["engine"] == "vpu"], [2])


if __name__ == "__main__":
    unittest.main()

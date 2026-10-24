"""Scalar slot proposals compose the lowered instructions with selected facts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from model_evidence import requires_model_evidence
from test_virtual_ssa import run
from test_virtual_scheduling import S, T, const, function, inp, outp, start


SLOT = "!atlas.virtual_spill"


def scalar_slots(word=-1):
    def slot(name, address):
        return (f'%{name} = "atlas.virtual_slot"() {{kind = "i32", words = 1 : i32, '
                f'slot_word = {address} : i32, purpose = "spill"}} : () -> {SLOT}')
    return function("slots", [start(), inp("io0", "s1", "tile", 0),
        slot("src", word), slot("dst", word if word < 0 else word + 1024), const("value", 7),
        f'%s2 = "atlas.virtual_save"(%s1, %src, %value) : ({S}, {SLOT}, i32) -> {S}',
        f'%s3 = "atlas.virtual_slot_copy"(%s2, %src, %dst) : ({S}, {SLOT}, {SLOT}) -> {S}',
        f'%s4, %again = "atlas.virtual_restore"(%s3, %dst) : ({S}, {SLOT}) -> ({S}, i32)',
        outp("s4", "s5", "tile", 0), f"return %s5 : {S}"])


@requires_model_evidence
class ScalarSlotProposalTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.facts = json.loads((Path(os.environ["ATLAS_TEST_MODEL_FACTS"]) / "model-timing.json").read_text())
        # CLI integration consumes genuinely derived facts. Synthetic lifetime
        # mutation/cost propagation is covered by the explicit C++ test API.

    def schedule(self, facts=None, source=None):
        facts = self.facts if facts is None else facts
        path = self.directory / "selected.json"
        path.write_text(json.dumps(facts))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return run("atlas-opt", source or scalar_slots(),
                   f"--select-atlas-model-evidence=test-op-timing={path} test-op-timing-sha256={digest}",
                   f"--schedule-atlas-virtual=provider=model graph-output={self.directory / 'graph.json'}")

    def costs(self, facts=None, word=-1):
        scheduled = self.schedule(facts, scalar_slots(word))
        self.assertEqual(scheduled.returncode, 0, scheduled.stderr)
        graph = json.loads((self.directory / "graph.json").read_text())[0]
        costs = {name.removeprefix("atlas.virtual_"): cost
                 for name, cost in zip(graph["operations"], graph["proposal"]["costs"])
                 if name in ("atlas.virtual_save", "atlas.virtual_restore", "atlas.virtual_slot_copy")}
        for cost in costs.values():
            self.assertEqual(cost["busy"], cost["latency"])
            self.assertEqual(cost["frontend"], cost["latency"])
        return {name: cost["latency"] for name, cost in costs.items()}

    def test_save_restore_copy_follow_operation_facts_and_releases(self):
        self.assertEqual(self.costs(), dict(save=4, restore=6, slot_copy=10))

    def test_scalar_lifetime_changes_require_rederivation(self):
        for operation, free in (("load", 11), ("store", 7)):
            with self.subTest(operation=operation):
                changed = deepcopy(self.facts)
                name = "scalar_lsu." + operation
                block = next(block for block in changed["op_timing"] if block["name"] == name)
                block["first_free_age"] = block["next_issue_age"] = free
                changed["reservation_releases"][name]["unit"].update(release_age=free - 1, available_age=free)
                result = self.schedule(changed)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("derived semantic payload differs from selected timing inputs", result.stderr)

    def test_allocated_addresses_use_lowering_zero_part_elision(self):
        # Both addresses are page aligned: materializeScalar emits LUI only.
        self.assertEqual(self.costs(word=135168), dict(save=3, restore=5, slot_copy=8))
        self.assertEqual(self.costs(word=135169), dict(save=4, restore=6, slot_copy=10))

    def test_missing_scalar_facts_refuse(self):
        for name in ("scalar_lsu.load", "scalar_lsu.store"):
            with self.subTest(operation=name):
                missing = deepcopy(self.facts)
                missing["op_timing"] = [block for block in missing["op_timing"] if block["name"] != name]
                del missing["reservation_releases"][name]
                result = self.schedule(missing)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("derived semantic payload differs from selected timing inputs", result.stderr)

    def test_missing_scalar_reservation_releases_refuse(self):
        for name, resource in (("scalar_lsu.load", "vmem_read"), ("scalar_lsu.store", "vmem_write")):
            with self.subTest(operation=name):
                missing = deepcopy(self.facts)
                del missing["reservation_releases"][name][resource]
                result = self.schedule(missing)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("derived semantic payload differs from selected timing inputs", result.stderr)

    def test_missing_scalar_facts_also_refuse_pressure_spill_penalties(self):
        source = function("pressure", [start(), inp("io0", "s1", "tile", 0),
            *(const(f"c{i}", i) for i in range(40)),
            *(f'%sum{i} = arith.addi %c{i}, %c{i + 1} : i32' for i in range(39)),
            outp("s1", "s2", "tile", 0), f"return %s2 : {S}"])
        missing = deepcopy(self.facts)
        missing["op_timing"] = [block for block in missing["op_timing"] if block["name"] != "scalar_lsu.store"]
        del missing["reservation_releases"]["scalar_lsu.store"]
        result = self.schedule(missing, source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("derived semantic payload differs from selected timing inputs", result.stderr)


if __name__ == "__main__":
    unittest.main()

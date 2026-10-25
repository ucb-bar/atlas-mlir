"""Scalar slot proposals compose the lowered instructions with selected facts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from model_evidence import requires_model_evidence
from test_virtual_ssa import run
from test_dma_ring_waits import fixture_evidence
from test_virtual_scheduling import S, const, function, inp, outp, start


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

    def costs(self, word=-1):
        scheduled = run("atlas-opt", scalar_slots(word), fixture_evidence(self.directory),
                        f"--schedule-atlas-virtual=provider=model graph-output={self.directory / 'graph.json'}")
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

    def test_allocated_addresses_use_lowering_zero_part_elision(self):
        # Both addresses are page aligned: materializeScalar emits LUI only.
        self.assertEqual(self.costs(word=135168), dict(save=3, restore=5, slot_copy=8))
        self.assertEqual(self.costs(word=135169), dict(save=4, restore=6, slot_copy=10))

if __name__ == "__main__":
    unittest.main()

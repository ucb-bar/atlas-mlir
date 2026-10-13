"""Resolved exports describe the checked final stream, including lowered virtual programs."""

import hashlib
import json
import re
import struct
import tempfile
import unittest
from pathlib import Path

from rtl_test_support import (CONSUMERS, EMIT, EXAMPLES, FACTS, OPT, PROVIDER, ROOT, VERIFY, VLS_COPY, VLS_LOOP, block,
    checked, copy, export, facts, facts_variant, program, run, selection, sha256, stream,
    virtual_chain, virtual_pressure, without_delays,)

@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(), "requires explicit RTL timing facts and Atlas tools")
class RTLTimingExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected = {}
        for consumer in CONSUMERS:
            result = run(OPT, VLS_COPY.read_text(), selection(), consumer)
            if result.returncode:
                raise AssertionError(result.stderr)
            cls.selected[consumer] = result.stdout

    def test_export_matches_both_consumers_words(self):
        for consumer, source in self.selected.items():
            with self.subTest(consumer=consumer):
                exported = export(source)
                words = [int(w, 16) for w in checked(self, source, tool=EMIT).split()]
                program = exported["program"]
                self.assertEqual((program["words"], program["word_count"]), (words, len(words)))
                self.assertEqual(program["words_sha256"],
                                 hashlib.sha256(b"".join(struct.pack("<I", w) for w in words)).hexdigest())
                self.assertEqual([i["word_u32"] for i in exported["instructions"]], words)
                vector = [i for i in exported["instructions"] if i["mnemonic"] in ("vload", "vstore")]
                self.assertEqual([i["mnemonic"] for i in vector], ["vload", "vstore"])
                self.assertEqual(exported["instructions"][-1]["event_kind"], "terminal_acceptance")

    def test_header_and_applicability(self):
        value = export(self.selected[CONSUMERS[0]])
        document = facts()
        self.assertEqual(value["schema"], "atlas.resolved_rtl_timing.v0")
        self.assertEqual(value["target_config"], "EE290SimConfig")
        self.assertEqual(value["qualification"], "conditional")
        self.assertIs(value["scheduling_qualified"], False)
        self.assertEqual(value["provider"], {"id": PROVIDER, "version": 1, "dma_policy": "none"})
        self.assertEqual(value["evidence"], {"op_timing_schema": "merlin.op_timing.v1",
                                             "op_timing_sha256": sha256(FACTS),
                                             "hardware_ir_sha256": document["hw_ir"]["sha256"]})
        applicability = value["applicability"]
        latency = block(document, "vlsu.vload")["assumptions"]["scratchpad_read_latency"]
        self.assertIn(f"sram_read_response_{latency}_cycles_after_request", applicability["environment_assumptions"])
        domain = applicability["operand_domain"]
        self.assertEqual((domain["vmem_banks"], domain["mlir_offset_min"], domain["mlir_offset_max"]), (6, -2048, 2047))
        self.assertIn("vmul.bf16", applicability["supported_operations"])
        self.assertNotIn(str(ROOT), json.dumps(value))
        self.assertNotIn(str(Path(FACTS)), json.dumps(value))

    def test_each_block_has_its_own_cycle_origin(self):
        single = export(self.selected[CONSUMERS[0]])
        self.assertEqual([(b["first_word"], b["word_count"], b["exit"], b["successors"]) for b in single["blocks"]],
                         [(0, len(single["instructions"]), "halt", [])])
        self.assertEqual({i["block"] for i in single["instructions"]}, {0})
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                timed = checked(self, program(VLS_LOOP), selection(dma="wait"), consumer, VERIFY)
                exported = export(timed)
                blocks, instructions = exported["blocks"], exported["instructions"]
                self.assertEqual(exported["schema"], "atlas.resolved_rtl_timing.v1")
                self.assertEqual([(b["exit"], b["successors"]) for b in blocks],
                                 [("fall_through", [1]), ("branch_with_delay_slot", [1, 2]), ("halt", [])])
                for number, b in enumerate(blocks):
                    members = instructions[b["first_word"]:b["first_word"] + b["word_count"]]
                    self.assertEqual({i["block"] for i in members}, {number})
                    self.assertEqual((members[0]["minimum_issue_cycle"], members[0]["issue_epoch"], members[0]["epoch_offset"]), (0, 0, 0))
                    self.assertEqual(b["entry_state"], "idle_engines_no_pending_dma")
                self.assertEqual(sum(b["word_count"] for b in blocks), len(instructions))
                branch = next(i for i in instructions if i["mnemonic"] == "blt")
                self.assertEqual(blocks[1]["successor_issue_offset"], branch["minimum_issue_cycle"] + 2)
                self.assertEqual(blocks[0]["successor_issue_offset"], blocks[0]["word_count"])
                self.assertIsNone(blocks[2]["successor_issue_offset"])

    def test_export_fails_closed(self):
        source = self.selected[CONSUMERS[0]]
        changed = re.sub(r'op_timing_sha256 = "[0-9a-f]{64}"', 'op_timing_sha256 = "' + "0" * 64 + '"', source)
        self.assertNotEqual(changed, source)
        for source, diagnostic in ((VLS_COPY.read_text(), "requires selected evidence"),
                                   (without_delays(source), "insufficient issue spacing"), (changed, "mismatch")):
            result = run(EMIT, source, "--rtl-timing-json")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn(diagnostic, result.stderr)

MODEL = "npu-model-rtl-match-v1"
LOWER = ("--verify-atlas-virtual-stream", "--lower-atlas-virtual-to-machine")
SCHEDULED = ("--verify-atlas-virtual-stream", "--schedule-atlas-virtual", "--lower-atlas-virtual-to-machine")
def mxu_beside_transfer(text):
    pending = set()
    for op, fields, _ in stream(text):
        channel = fields.split("channel = ")[1].split(" ")[0] if "channel = " in fields else None
        if op == "dma":
            pending.add(channel)
        elif op == "dma_wait":
            pending.discard(channel)
        elif op.startswith("mxu") and pending:
            return True
    return False

@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(), "requires explicit RTL timing facts and built tools")
class VirtualRTLTimingTest(unittest.TestCase):

    def test_fp8_pack_loop_is_timed_under_each_iteration(self):
        untimed = checked(self, (EXAMPLES / "virtual_fp8_two_layer_mlp.mlir").read_text(), *LOWER)
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                timed = checked(self, untimed, selection(dma="wait"), consumer, VERIFY)
                exported = export(timed)
                loop = next(b for b in exported["blocks"] if "entry_contexts" in b)
                contexts = loop["entry_contexts"]
                self.assertEqual(sorted(e["x13"] for c in contexts for e in c["entries"]), list(range(32)))
                self.assertEqual(exported["instructions"][loop["first_word"]]["mnemonic"], "lw")
                lines = sorted(c["footprints"][0]["accesses"][1]["first"] for c in contexts for _ in c["entries"])
                self.assertEqual(lines, [(0x20000 + 16 * k) // 32 for k in range(32)])

    def test_selection_survives_lowering_and_reordering(self):
        source = (EXAMPLES / "virtual_bf16_vpu_program.mlir").read_text()
        untimed = checked(self, source, selection(dma="wait"), *SCHEDULED)
        self.assertIn("atlas.rtl_evidence", untimed)
        self.assertIn('atlas.timing_state = "untimed"', untimed)
        self.assertNotIn("atlas.timing_provider", untimed)
        checked(self, untimed, tool=EMIT, diagnostic="untimed Atlas stream requires scheduling or delay insertion")
        self.assertTrue(checked(self, untimed, "--allow-untimed", tool=EMIT))
        ordered = checked(self, untimed, "--schedule-atlas-stream=insert-delays=false")
        self.assertIn("atlas.rtl_evidence", ordered)
        self.assertIn('atlas.timing_state = "untimed"', ordered)
        self.assertNotIn('"atlas.delay"', ordered)
        for stage in (untimed, ordered):
            timed = checked(self, stage, "--insert-atlas-delays", "--verify-atlas-timing")
            self.assertIn(f'atlas.timing_provider = "{PROVIDER}"', timed)
            self.assertTrue(checked(self, timed, tool=EMIT))
            structured = checked(self, timed, "--convert-atlas-to-llvm-calls")
            self.assertIn("atlas.rtl_evidence", structured)
            self.assertIn("op_timing_sha256", checked(self, structured, "--finalize-atlas-llvm-calls"))
            self.assertEqual(checked(self, structured, "--finalize-atlas-llvm-calls"),
                             checked(self, timed, "--convert-atlas-to-llvm"))
            checked(self, without_delays(timed), VERIFY,
                    diagnostic="insufficient issue spacing|timing resource conflict")
        checked(self, ordered, "--insert-atlas-delays=provider=" + MODEL, diagnostic="selected RTL evidence excludes")

    def test_queued_transfers_compile_where_the_evidence_rules_hold(self):
        cases = [(f"{fmt} copy", copy(fmt), LOWER, False) for fmt in ("fp8", "bf16")]
        cases += [(name, (EXAMPLES / f"{name}.mlir").read_text(), passes, passes == SCHEDULED)
                  for name in ("virtual_sched_overlap", "virtual_dma_mxu", "virtual_dma_mxu_bf16")
                  for passes in (LOWER, SCHEDULED)]
        for name, source, passes, pending in cases:
            untimed = checked(self, source, *passes)
            for consumer in CONSUMERS:
                with self.subTest(program=name, passes=passes, consumer=consumer):
                    timed = checked(self, untimed, selection(dma="wait"), consumer, VERIFY)
                    self.assertIn(f'atlas.timing_provider = "{PROVIDER}"', timed)
                    self.assertTrue(checked(self, timed, tool=EMIT))
                    if pending and consumer == CONSUMERS[0]:
                        self.assertTrue(mxu_beside_transfer(timed), "no MXU command between a launch and its wait")

    def test_bf16_pairs_sharing_a_physical_bank_are_rejected(self):
        source = virtual_pressure(17).replace('    %io18 = "atlas.virtual_output_bf16"(%io17, %t0)',
            '    %sum = "atlas.virtual_vpu_binary"(%t0, %t16) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16\n'
            '    %io18 = "atlas.virtual_output_bf16"(%io17, %sum)')
        self.assertIn("%sum", source)
        untimed = checked(self, source, *LOWER)
        self.assertIn('lhs = 0 : i32, rhs = 32 : i32', untimed)
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                checked(self, untimed, selection(dma="wait"), consumer,
                         diagnostic=r"VPU register operands must occupy distinct physical banks \(register mod 32\): m0 and m32 share bank 0")

    def test_virtual_scheduler_costs_from_the_selected_provider(self):
        source = (EXAMPLES / "virtual_bf16_vpu_program.mlir").read_text()
        checked(self, source, "--schedule-atlas-virtual=provider=" + PROVIDER, diagnostic="requires selected evidence")
        checked(self, source, selection(dma="wait"), "--schedule-atlas-virtual=provider=" + MODEL, diagnostic="selected RTL evidence excludes")
        by_facts = checked(self, source, selection(dma="wait"), "--schedule-atlas-virtual")
        self.assertIn("atlas.rtl_evidence", by_facts)
        self.assertNotIn("atlas.timing_provider", by_facts)
        timed = checked(self, by_facts, "--lower-atlas-virtual-to-machine", "--insert-atlas-delays", VERIFY)
        self.assertIn(f'atlas.timing_provider = "{PROVIDER}"', timed)
        self.assertTrue(checked(self, timed, tool=EMIT))
        with tempfile.TemporaryDirectory(prefix="atlas-virtual-facts-") as directory:
            missing = facts_variant(directory, lambda d: d["op_timing"].remove(block(d, "vpu.relu")))
            checked(self, virtual_chain(1), selection(missing, dma="wait"), "--schedule-atlas-virtual",
                     diagnostic=f"virtual cost of vrelu.bf16: {PROVIDER}: vpu.relu: facts block is missing")

if __name__ == "__main__":
    unittest.main()

"""Resolved exports describe the checked final stream, including lowered virtual programs."""

import hashlib
import json
import re
import struct
import tempfile
import unittest
from pathlib import Path

from rtl_test_support import (
    CONSUMERS, EMIT, EXAMPLES, FACTS, HALT, OPT, PROVIDER, ROOT, VERIFY, VLS_COPY, VLS_LOOP,
    addi, block, copy, facts, facts_variant, program, run, selected, selection, sha256, stream,
    virtual_chain, virtual_pressure, without_delays,
)


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and Atlas tools")
class RTLTimingExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected = {}
        for consumer in CONSUMERS:
            result = run(OPT, VLS_COPY.read_text(), selection(), consumer)
            if result.returncode:
                raise AssertionError(result.stderr)
            cls.selected[consumer] = result.stdout

    def export(self, source):
        return run(EMIT, source, "--rtl-timing-json")

    def test_export_matches_both_consumers_words_and_facts(self):
        document = facts()
        load, store = block(document, "vlsu.vload"), block(document, "vlsu.vstore")
        for consumer, source in self.selected.items():
            with self.subTest(consumer=consumer):
                result = self.export(source)
                self.assertEqual(result.returncode, 0, result.stderr)
                exported = json.loads(result.stdout)
                emitted = run(EMIT, source)
                self.assertEqual(emitted.returncode, 0, emitted.stderr)
                words = [int(w, 16) for w in emitted.stdout.split()]
                program = exported["program"]
                self.assertEqual((program["words"], program["word_count"]), (words, len(words)))
                self.assertEqual(program["words_sha256"],
                                 hashlib.sha256(b"".join(struct.pack("<I", w) for w in words)).hexdigest())
                self.assertEqual([i["word_u32"] for i in exported["instructions"]], words)
                vector = [i for i in exported["instructions"] if i["mnemonic"] in ("vload", "vstore")]
                self.assertEqual([i["mnemonic"] for i in vector], ["vload", "vstore"])
                self.assertGreaterEqual(vector[1]["logical_issue_cycle"] - vector[0]["logical_issue_cycle"],
                                        load["first_free_age"])
                for entry in vector:
                    is_load = entry["mnemonic"] == "vload"
                    events = (load if is_load else store)["events"]
                    accesses = {a["resource"]: a for a in entry["footprint"]["accesses"]}
                    self.assertEqual(accesses["vmem"]["age"], events["vmem_read" if is_load else "vmem_write"]["first_age"])
                    self.assertEqual(accesses["mreg"]["age"], events["mreg_write" if is_load else "mreg_read"]["first_age"])
                    self.assertEqual((accesses["vmem"]["write"], accesses["mreg"]["write"]), (not is_load, is_load))
                    self.assertEqual(accesses["vmem"]["count"], 32)
                    self.assertEqual(entry["footprint"]["done_age"], load["first_free_age"] - 1)
                self.assertEqual(exported["instructions"][-1]["event_kind"], "terminal_acceptance")

    def test_header_and_applicability(self):
        result = self.export(self.selected[CONSUMERS[0]])
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
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
        self.assertEqual(applicability["admission"]["minimum_vls_issue_gap"],
                         block(document, "vlsu.vload")["first_free_age"])
        self.assertEqual(applicability["admission"]["xlu_first_free_age"],
                         block(document, "xlu.transpose")["first_free_age"])
        domain = applicability["operand_domain"]
        self.assertEqual((domain["vmem_banks"], domain["mlir_offset_min"], domain["mlir_offset_max"]),
                         (6, -2048, 2047))
        self.assertIn("vmul.bf16", applicability["supported_operations"])
        self.assertNotIn(str(ROOT), result.stdout)
        self.assertNotIn(str(Path(FACTS)), result.stdout)

    def test_each_block_has_its_own_cycle_origin(self):
        single = json.loads(self.export(self.selected[CONSUMERS[0]]).stdout)
        self.assertEqual([(b["first_word"], b["word_count"], b["exit"], b["successors"]) for b in single["blocks"]],
                         [(0, len(single["instructions"]), "halt", [])])
        self.assertEqual({i["block"] for i in single["instructions"]}, {0})
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                result = selected(VLS_LOOP, consumer, VERIFY, dma="wait")
                self.assertEqual(result.returncode, 0, result.stderr)
                exported = self.export(result.stdout)
                self.assertEqual(exported.returncode, 0, exported.stderr)
                exported = json.loads(exported.stdout)
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
                                   (without_delays(source), "insufficient issue spacing"),
                                   (changed, "mismatch")):
            result = self.export(source)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn(diagnostic, result.stderr)


MODEL = "npu-model-rtl-match-v1"
LOWER = ("--verify-atlas-virtual-stream", "--lower-atlas-virtual-to-machine")
SCHEDULED = ("--verify-atlas-virtual-stream", "--schedule-atlas-virtual", "--lower-atlas-virtual-to-machine")
SERIAL = ("virtual_bf16_vpu_program", "virtual_bf16_shared_relu", "virtual_bf16_branch_program", "virtual_bf16_loop_program",
          "virtual_bf16_swap_loop_program", "virtual_bf16_dynamic_branch_program", "virtual_fp8_matmul_program",
          "virtual_mxu_seeded_fp8", "virtual_dma_tiles",  # seli sets the FP8 scale register
          # The FP8 pack relayout loop moves its lw/sw addresses every iteration: one entry context each.
          "virtual_fp8_two_layer_mlp", "virtual_fp8_two_layer_mlp_bias")
WAIT = ("dma_wait", "channel = 0 : i32")
# Two transfers in flight on channels 0 and 1, then both waited (test_untimed_pipeline's manual multichannel stream).
TWO_CHANNELS = [addi(6, 0, 0), addi(7, 0, 1024), addi(1, 0, 0), addi(2, 0, 32),
                ("dma", 'direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32'),
                ("dma", 'direction = "load", channel = 1 : i32, reg = 7 : i32, dram = 1 : i32, size = 2 : i32'),
                WAIT, ("dma_wait", "channel = 1 : i32"), HALT]
SERIALIZED = TWO_CHANNELS[:5] + [WAIT] + TWO_CHANNELS[5:6] + TWO_CHANNELS[7:]


def mxu_beside_transfer(text):
    """Some MXU command issues between a DMA launch and the wait on its channel."""
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


def checked(test, source, *options, tool=OPT):
    result = run(tool, source, *options)
    test.assertEqual(result.returncode, 0, result.stderr)
    return result.stdout


def rejected(test, source, *options, diagnostic, tool=OPT):
    result = run(tool, source, *options)
    test.assertNotEqual(result.returncode, 0, result.stdout)
    test.assertRegex(result.stderr, diagnostic)
    return result


def timed_by(test, text, provider):
    test.assertIn('atlas.timing_state = "timed"', text)
    test.assertIn(f'atlas.timing_provider = "{provider}"', text)


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(), "requires explicit RTL timing facts and built tools")
class VirtualRTLTimingTest(unittest.TestCase):
    def test_serial_dma_programs_time_under_the_provider(self):
        for name in SERIAL:
            source = (EXAMPLES / f"{name}.mlir").read_text()
            for passes in (LOWER, SCHEDULED):
                untimed = checked(self, source, *passes)
                self.assertIn('atlas.timing_state = "untimed"', untimed)
                for consumer in CONSUMERS:
                    with self.subTest(program=name, passes=passes[1], consumer=consumer):
                        timed = checked(self, untimed, selection(dma="wait"), consumer, VERIFY)
                        timed_by(self, timed, PROVIDER)
                        self.assertIn("atlas.rtl_evidence", timed)
                        self.assertTrue(checked(self, timed, tool=EMIT))
                        self.assertIn('"provider": {', checked(self, timed, "--rtl-timing-json", tool=EMIT))
                        structured = checked(self, timed, "--convert-atlas-to-llvm-calls")
                        self.assertIn("atlas.rtl_evidence", structured)
                        self.assertEqual(checked(self, structured, "--finalize-atlas-llvm-calls"), checked(self, timed, "--convert-atlas-to-llvm"))
                        # Dropping the delays breaks the timing, or first a contract the delays' CFG ownership established.
                        rejected(self, without_delays(timed), VERIFY,
                                 diagnostic="insufficient issue spacing|timing resource conflict|contract predecessor is not established|cannot prove DMA memory disjointness")

    def test_fp8_pack_loop_is_timed_under_each_iteration(self):
        untimed = checked(self, (EXAMPLES / "virtual_fp8_two_layer_mlp.mlir").read_text(), *LOWER)
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                timed = checked(self, untimed, selection(dma="wait"), consumer, VERIFY)
                exported = json.loads(checked(self, timed, "--rtl-timing-json", tool=EMIT))
                loop = next(b for b in exported["blocks"] if "entry_contexts" in b)
                contexts = loop["entry_contexts"]
                self.assertEqual(sorted(e["x13"] for c in contexts for e in c["entries"]), list(range(32)))
                self.assertEqual(exported["instructions"][loop["first_word"]]["mnemonic"], "lw")
                # The first lw reads x10 = 0x20000 + 16 * iteration.
                lines = sorted(c["footprints"][0]["accesses"][1]["first"] for c in contexts for _ in c["entries"])
                self.assertEqual(lines, [(0x20000 + 16 * k) // 32 for k in range(32)])

    def test_selection_survives_lowering_and_reordering(self):
        source = (EXAMPLES / "virtual_bf16_vpu_program.mlir").read_text()
        # Facts selected on the virtual program are carried by the untimed lowering, which keeps no provider.
        untimed = checked(self, source, selection(dma="wait"), *SCHEDULED)
        self.assertIn("atlas.rtl_evidence", untimed)
        self.assertIn('atlas.timing_state = "untimed"', untimed)
        self.assertNotIn("atlas.timing_provider", untimed)
        rejected(self, untimed, tool=EMIT, diagnostic="untimed Atlas stream requires scheduling or delay insertion")
        self.assertTrue(checked(self, untimed, "--allow-untimed", tool=EMIT))
        # A reorder-only schedule stays untimed and keeps the selection; delay insertion then infers the provider.
        ordered = checked(self, untimed, "--schedule-atlas-stream=insert-delays=false")
        self.assertIn("atlas.rtl_evidence", ordered)
        self.assertIn('atlas.timing_state = "untimed"', ordered)
        self.assertNotIn('"atlas.delay"', ordered)
        for stage in (untimed, ordered):
            timed = checked(self, stage, "--insert-atlas-delays", "--verify-atlas-timing")
            timed_by(self, timed, PROVIDER)
            self.assertTrue(checked(self, timed, tool=EMIT))
        rejected(self, ordered, "--insert-atlas-delays=provider=" + MODEL, diagnostic="selected RTL evidence excludes")

    def test_queued_transfers_compile_where_the_evidence_rules_hold(self):
        source = (EXAMPLES / "virtual_sched_overlap.mlir").read_text()
        for name, untimed in (("two-launch FP8 copy", checked(self, copy("fp8"), *LOWER)), ("two-launch BF16 copy", checked(self, copy("bf16"), *LOWER)),
                              ("source-order overlap program", checked(self, source, *LOWER)),
                              ("manual two channels", program(TWO_CHANNELS)), ("manual serialized channels", program(SERIALIZED))):
            for consumer in CONSUMERS:
                with self.subTest(program=name, consumer=consumer):
                    timed = checked(self, untimed, selection(dma="wait"), consumer, VERIFY)
                    timed_by(self, timed, PROVIDER)
                    self.assertTrue(checked(self, timed, tool=EMIT))
        # Virtual schedules that put MXU work beside a pending transfer (rejected while MXU waited for transfers).
        for name in ("virtual_sched_overlap", "virtual_dma_mxu", "virtual_dma_mxu_bf16"):
            scheduled = checked(self, (EXAMPLES / f"{name}.mlir").read_text(), *SCHEDULED)
            for consumer in CONSUMERS:
                with self.subTest(program=f"scheduled {name}", consumer=consumer):
                    timed = checked(self, scheduled, selection(dma="wait"), consumer, VERIFY)
                    timed_by(self, timed, PROVIDER)
                    self.assertTrue(checked(self, timed, tool=EMIT))
                    if consumer == CONSUMERS[0]:  # delay insertion keeps the virtual order; the stream scheduler may reorder
                        self.assertTrue(mxu_beside_transfer(timed), "no MXU command between a launch and its wait")

    def test_bf16_pairs_sharing_a_physical_bank_are_rejected(self):
        # Seventeen live pairs put colors 0 and 16 in m0/m1 and m32/m33, one VPU bank apart.
        source = virtual_pressure(17).replace(
            '    %io18 = "atlas.virtual_output_bf16"(%io17, %t0)',
            '    %sum = "atlas.virtual_vpu_binary"(%t0, %t16) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16\n'
            '    %io18 = "atlas.virtual_output_bf16"(%io17, %sum)')
        self.assertIn("%sum", source)
        untimed = checked(self, source, *LOWER)
        self.assertIn('lhs = 0 : i32, rhs = 32 : i32', untimed)
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                rejected(self, untimed, selection(dma="wait"), consumer,
                         diagnostic=r"VPU register operands must occupy distinct physical banks \(register mod 32\): m0 and m32 share bank 0")

    def test_virtual_scheduler_costs_from_the_selected_provider(self):
        source = (EXAMPLES / "virtual_bf16_vpu_program.mlir").read_text()
        rejected(self, source, "--schedule-atlas-virtual=provider=" + PROVIDER, diagnostic="requires selected evidence")
        rejected(self, source, selection(dma="wait"), "--schedule-atlas-virtual=provider=" + MODEL, diagnostic="selected RTL evidence excludes")
        by_model = checked(self, source, "--schedule-atlas-virtual")
        by_facts = checked(self, source, selection(dma="wait"), "--schedule-atlas-virtual")
        self.assertIn("atlas.rtl_evidence", by_facts)
        self.assertNotIn("atlas.rtl_evidence", by_model)
        self.assertNotIn("atlas.timing_provider", by_facts)
        timed = checked(self, by_facts, "--lower-atlas-virtual-to-machine", "--insert-atlas-delays", VERIFY)
        timed_by(self, timed, PROVIDER)
        self.assertTrue(checked(self, timed, tool=EMIT))
        # An operation without a resolved facts block fails the cost model rather than borrowing model numbers.
        with tempfile.TemporaryDirectory(prefix="atlas-virtual-facts-") as directory:
            missing = facts_variant(directory, lambda d: d["op_timing"].remove(block(d, "vpu.relu")))
            rejected(self, virtual_chain(1), selection(missing, dma="wait"), "--schedule-atlas-virtual",
                     diagnostic=f"virtual cost of vrelu.bf16: {PROVIDER}: vpu.relu: facts block is missing")


if __name__ == "__main__":
    unittest.main()

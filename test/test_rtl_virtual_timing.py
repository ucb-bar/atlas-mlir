"""Virtual programs lowered untimed and timed by the rtl-op-timing-v1 provider (ATLAS_OOT_BIN_DIR, ATLAS_OP_TIMING).

Transfers queue one per channel into eight command slots with disjoint VMEM windows, waited within their block;
VLS, VPU and XLU run beside them and MXU commands wait for them. A virtual schedule that puts MXU work beside a
pending transfer stays on the model provider. Loops whose addresses move are timed under each concrete entry context.
"""
import json
import tempfile
import unittest

from test_delay_insertion import EMIT, OPT, addi, program, run, without_delays
from test_rtl_timing import CONSUMERS, FACTS, HALT, PROVIDER, VERIFY, block, facts_variant, selection
from test_virtual_dma import copy
from test_virtual_lowering import EXAMPLES, virtual_chain, virtual_pressure

MODEL = "npu-model-rtl-match-v1"
LOWER = ("--verify-atlas-virtual-stream", "--lower-atlas-virtual-to-machine")
SCHEDULED = ("--verify-atlas-virtual-stream", "--schedule-atlas-virtual", "--lower-atlas-virtual-to-machine")
SERIAL = ("virtual_bf16_vpu_program", "virtual_bf16_shared_relu", "virtual_bf16_branch_program", "virtual_bf16_loop_program",
          "virtual_bf16_swap_loop_program", "virtual_bf16_dynamic_branch_program", "virtual_fp8_matmul_program",
          "virtual_mxu_seeded_fp8", "virtual_dma_tiles",  # seli sets the FP8 scale register
          # The FP8 pack relayout loop moves its lw/sw addresses every iteration: one entry context each.
          "virtual_fp8_two_layer_mlp", "virtual_fp8_two_layer_mlp_bias")
MXU_BESIDE_DMA = "'atlas.mxu_push' op order conflict with atlas.dma"
WAIT = ("dma_wait", "channel = 0 : i32")
# Two transfers in flight on channels 0 and 1, then both waited (test_untimed_pipeline's manual multichannel stream).
TWO_CHANNELS = [addi(6, 0, 0), addi(7, 0, 1024), addi(1, 0, 0), addi(2, 0, 32),
                ("dma", 'direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32'),
                ("dma", 'direction = "load", channel = 1 : i32, reg = 7 : i32, dram = 1 : i32, size = 2 : i32'),
                WAIT, ("dma_wait", "channel = 1 : i32"), HALT]
SERIALIZED = TWO_CHANNELS[:5] + [WAIT] + TWO_CHANNELS[5:6] + TWO_CHANNELS[7:]


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
        # The virtual schedule of the overlap program puts a weight push beside a pending transfer; the MXU waits.
        scheduled = checked(self, source, *SCHEDULED)
        for consumer in CONSUMERS:
            with self.subTest(program="scheduled overlap program", consumer=consumer):
                rejected(self, scheduled, selection(dma="wait"), consumer, diagnostic=MXU_BESIDE_DMA)
                timed_by(self, checked(self, scheduled, consumer, VERIFY), MODEL)

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

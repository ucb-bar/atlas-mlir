"""Selected VPU timing (BF16 multiply first) from RTL-computed facts.

Select ATLAS_OOT_BIN_DIR and ATLAS_OP_TIMING explicitly.
"""
import json
import tempfile
import unittest

from test_delay_insertion import OPT, EMIT, addi, nop, program, run, without_delays
from test_rtl_timing import (CONSUMERS, FACTS, HALT, MARKER, RESOLVER, block, delay, export, facts,
                             facts_variant, selected, vls)


def vmul(lhs=0, rhs=2, dst=4, kind="mul"):
    return ("vpu_binary", f'kind = "{kind}", dst = {dst} : i32, '
                          f'lhs = {lhs} : i32, rhs = {rhs} : i32')


def unary(kind, src=0, dst=4):
    return ("vpu_unary", f'kind = "{kind}", dst = {dst} : i32, src = {src} : i32')


def reduce(kind, src=0, dst=4):
    return ("vpu_reduce", f'kind = "{kind}", dst = {dst} : i32, src = {src} : i32')


def pack(direction, src=0, dst=4, scale=3):
    return ("vpu_pack", f'direction = "{direction}", dst = {dst} : i32, src = {src} : i32, scale_reg = {scale} : i32')


def vli(mode, dst=4, immediate=16256):
    return ("vli", f'mode = "{mode}", dst = {dst} : i32, immediate = {immediate} : i32')


# Every VPU operation the facts cover, with its facts block.
VPU_OPERATIONS = {
    **{f"v{kind}.bf16": (vmul(kind=kind), f"vpu.{kind}") for kind in ("add", "sub", "mul")},
    "vmaximum.bf16": (vmul(kind="max"), "vpu.pairmax"), "vminimum.bf16": (vmul(kind="min"), "vpu.pairmin"),
    "vmov": (unary("mov"), "vpu.mov"), "vrecip.bf16": (unary("recip"), "vpu.rcp"),
    **{f"v{kind}.bf16": (unary(kind), f"vpu.{kind}")
       for kind in ("exp", "exp2", "relu", "sin", "cos", "tanh", "sqrt", "square", "cube")},
    "vlog2.bf16": (unary("log2"), "vpu.log"),
    "vredsum.row.bf16": (reduce("row_sum"), "vpu.rsum"), "vredmax.row.bf16": (reduce("row_max"), "vpu.rmax"),
    "vredmin.row.bf16": (reduce("row_min"), "vpu.rmin"), "vredsum.bf16": (reduce("col_sum"), "vpu.csum"),
    "vredmax.bf16": (reduce("col_max"), "vpu.cmax"), "vredmin.bf16": (reduce("col_min"), "vpu.cmin"),
    "vpack.bf16.fp8": (pack("bf16_to_fp8"), "vpu.fp8pack"),
    "vunpack.fp8.bf16": (pack("fp8_to_bf16"), "vpu.fp8unpack"),
    "vli.all": (vli("all"), "vpu.vliAll"), "vli.row": (vli("row"), "vpu.vliRow"),
    "vli.col": (vli("col"), "vpu.vliCol"), "vli.one": (vli("one"), "vpu.vliOne"),
}


def expected_mreg_accesses(record):
    """(age, count, step) of the compiler MReg accesses a facts block implies; a column
    reduction reads its 64 source rows twice, which the compiler keeps as two passes."""
    expected = []
    for name, group in record["events"].items():
        if not group["count"]:
            continue
        if group["count"] == 128:
            expected += [(group["first_age"], 64, group["step"]), (group["first_age"] + 64, 64, group["step"])]
        else:
            expected.append((group["first_age"], group["count"], group["step"]))
    return sorted(expected)


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built tools")
class SelectedVmulTimingTest(unittest.TestCase):
    def selected(self, ops, *passes, **options):
        return selected(ops, *passes, **options)

    def test_both_consumers_and_observed_stream_export(self):
        record = block(facts(), "vpu.mul")
        for lhs, rhs, dst in ((0, 2, 4), (62, 2, 4), (0, 60, 62)):
            for consumer in CONSUMERS:
                with self.subTest(lhs=lhs, rhs=rhs, dst=dst, consumer=consumer):
                    result = self.selected([vmul(lhs, rhs, dst), HALT], consumer, "--verify-atlas-rtl-timing")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(RESOLVER, result.stdout)
                    value = export(result.stdout)
                    self.assertFalse(value["scheduling_qualified"])
                    self.assertEqual(value["qualification"], "conditional")
                    instruction = next(i for i in value["instructions"] if i["mnemonic"] == "vmul.bf16")
                    footprint = instruction["footprint"]
                    accesses = footprint["accesses"]
                    self.assertEqual(len(accesses), 3)
                    self.assertEqual(sorted(a["age"] for a in accesses),
                                     sorted(record["events"][g]["first_age"] for g in ("read0", "read1", "write0")))
                    self.assertTrue(all(a["count"] == 64 and a["step"] == 1 for a in accesses))
                    self.assertEqual(footprint["done_age"], record["first_free_age"] - 1)
                    self.assertEqual(footprint["write_release"], record["next_issue_age"])
                    self.assertEqual(footprint["read_release"], record["events"]["read0"]["last_age"] +
                                     record["assumptions"]["scratchpad_read_latency"])
                    self.assertEqual(sorted(h["unit"] for h in footprint["holds"]),
                                     ["VLOAD path", "VPU", "VSTORE path"])
                    self.assertTrue(all((h["from"], h["to"]) == (0, record["first_free_age"] - 1)
                                        for h in footprint["holds"]))
                    self.assertNotEqual(run(EMIT, without_delays(result.stdout)).returncode, 0)

    def test_pair_domain_is_rejected_in_both_consumers(self):
        pairs = ((1, 2, 4), (0, 3, 4), (0, 2, 5), (0, 2, 64),
                 (-2, 2, 4), (0, 0, 4), (0, 32, 4), (0, 2, 0), (0, 2, 34))
        for pair in pairs:
            for consumer in CONSUMERS:
                with self.subTest(pair=pair, consumer=consumer):
                    result = self.selected([vmul(*pair), HALT], consumer)
                    self.assertNotEqual(result.returncode, 0)
        # Physical bank aliases (register mod 32) pass the dialect verifier and fail in the resolver.
        for ops in ([unary("mov", 0, 32), HALT], [reduce("row_sum", 32, 0), HALT], [reduce("col_sum", 0, 0), HALT],
                    [pack("bf16_to_fp8", 0, 1), HALT], [pack("fp8_to_bf16", 4, 4), HALT], [unary("square", 2, 34), HALT]):
            with self.subTest(ops=ops):
                result = self.selected(ops, CONSUMERS[0])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(RESOLVER, result.stderr)

    def test_every_vpu_operation_takes_its_numbers_from_the_facts(self):
        document = facts()
        for mnemonic, (operation, name) in VPU_OPERATIONS.items():
            record = block(document, name)
            for consumer in CONSUMERS:
                with self.subTest(mnemonic=mnemonic, consumer=consumer):
                    result = self.selected([operation, HALT], consumer, "--verify-atlas-rtl-timing")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                    self.assertNotEqual(run(OPT, without_delays(result.stdout), "--verify-atlas-rtl-timing").returncode, 0)
                    instruction = next(i for i in export(result.stdout)["instructions"] if i["mnemonic"] == mnemonic)
                    footprint = instruction["footprint"]
                    mreg = sorted((a["age"], a["count"], a["step"]) for a in footprint["accesses"]
                                  if a["resource"] == "mreg")
                    self.assertEqual(mreg, expected_mreg_accesses(record))
                    self.assertEqual(footprint["done_age"], record["first_free_age"] - 1)
                    self.assertEqual(footprint["write_release"], record["next_issue_age"])
                    self.assertEqual({h["unit"]: h["to"] for h in footprint["holds"]},
                                     {u: record["first_free_age"] - 1 for u in ("VPU", "VLOAD path", "VSTORE path")})

    def test_serialized_reuse_and_store_boundary(self):
        boundary = block(facts(), "vpu.mul")["first_free_age"]
        for gap in (boundary - 1, boundary, boundary + 1):
            for next_op in (vmul(6, 8, 10), vls("vstore", 4)):
                with self.subTest(gap=gap, next_op=next_op):
                    result = self.selected([addi(6, 0, 0), vmul(), delay(gap - 2),
                                            next_op, delay(64), nop(), HALT], "--verify-atlas-rtl-timing")
                    self.assertEqual(result.returncode == 0, gap >= boundary, result.stderr)

    def test_facts_numbers_drive_the_boundary_and_unresolved_blocks_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="atlas-vpu-facts-") as directory:
            slower = facts_variant(directory, lambda d: block(d, "vpu.mul").update(first_free_age=70, next_issue_age=69))
            for gap, accepted in ((66, False), (69, False), (70, True)):
                with self.subTest(gap=gap):
                    result = self.selected([addi(6, 0, 0), vmul(), delay(gap - 2), vmul(6, 8, 10), delay(68), nop(), HALT],
                                           "--verify-atlas-rtl-timing", path=slower)
                    self.assertEqual(result.returncode == 0, accepted, result.stderr)
            unresolved = facts_variant(directory, lambda d: block(d, "vpu.mul").update(events=None), "unresolved.json")
            rejected = self.selected([vmul(), HALT], CONSUMERS[0], path=unresolved)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("vpu.mul: facts block is unresolved", rejected.stderr)
            accepted = self.selected([vmul(kind="add"), HALT], CONSUMERS[0], "--verify-atlas-rtl-timing", path=unresolved)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            # Structure is the compiler's: a facts write group with the wrong sweep is refused.
            narrow = facts_variant(directory, lambda d: block(d, "vpu.mul")["events"]["write0"].update(count=32), "narrow.json")
            rejected = self.selected([vmul(), HALT], CONSUMERS[0], path=narrow)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("does not sweep 1x64", rejected.stderr)

    def test_producer_consumer_and_completion_are_rechecked(self):
        ops = [addi(6, 0, 0), addi(8, 0, 1024), vls("vload", 0),
               vls("vload", 1, offset=32), vls("vload", 2, offset=64),
               vls("vload", 3, offset=96), vmul(), vls("vstore", 4, 8),
               vls("vstore", 5, 8, 32), addi(1, 0, 1), MARKER, HALT]
        for consumer in CONSUMERS:
            result = self.selected(ops, consumer, "--verify-atlas-rtl-timing")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(run(EMIT, result.stdout).returncode, 0)
            unsafe = without_delays(result.stdout)
            self.assertNotEqual(run(OPT, unsafe, "--verify-atlas-rtl-timing").returncode, 0)
            self.assertNotEqual(run(EMIT, unsafe).returncode, 0)
        for suffix in ([MARKER, delay(64), HALT], [HALT]):
            self.assertNotEqual(self.selected([vmul(), *suffix], "--verify-atlas-rtl-timing").returncode, 0)

    def test_llvm_reconstruction_preserves_evidence_and_rechecks(self):
        result = self.selected([vmul(), HALT], CONSUMERS[0], "--verify-atlas-rtl-timing")
        self.assertEqual(result.returncode, 0, result.stderr)
        unsafe = without_delays(result.stdout)
        for entrypoint in ("--convert-atlas-to-llvm", "--convert-atlas-to-llvm-calls"):
            self.assertNotEqual(run(OPT, unsafe, entrypoint).returncode, 0)
        staged = run(OPT, result.stdout, "--convert-atlas-to-llvm-calls")
        self.assertEqual(staged.returncode, 0, staged.stderr)
        self.assertIn("op_timing_sha256", staged.stdout)
        finalized = run(OPT, staged.stdout, "--finalize-atlas-llvm-calls")
        self.assertEqual(finalized.returncode, 0, finalized.stderr)
        self.assertIn("op_timing_sha256", finalized.stdout)

    def test_dma_wait_and_xlu_share_serialized_compute_policy(self):
        from test_rtl_dma_timing import setup, transfer, wait
        from test_rtl_xlu_timing import transpose
        unsafe = [*setup(), transfer(), vmul(), wait(), HALT]
        self.assertNotEqual(self.selected(unsafe, CONSUMERS[0], dma="wait").returncode, 0)
        for consumer in CONSUMERS:
            result = self.selected([*setup(), transfer(), wait(), transpose(0, 2),
                                    vmul(2, 4, 6), HALT], consumer, "--verify-atlas-rtl-timing", dma="wait")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('dma = "wait"', result.stdout)
            self.assertEqual(run(EMIT, result.stdout).returncode, 0)


if __name__ == "__main__":
    unittest.main()

"""Selected VPU timing from RTL-computed facts (ATLAS_OOT_BIN_DIR, ATLAS_OP_TIMING)."""
import tempfile
import unittest

from test_delay_insertion import OPT, EMIT, addi, nop, run, without_delays
from test_rtl_timing import (CONSUMERS, FACTS, HALT, MARKER, RESOLVER, VERIFY, block, delay, export,
                             facts, facts_variant, selected, vls)


def vmul(lhs=0, rhs=2, dst=4, kind="mul"):
    return ("vpu_binary", f'kind = "{kind}", dst = {dst} : i32, lhs = {lhs} : i32, rhs = {rhs} : i32')


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
    """(age, count, step) MReg accesses a block implies; a column reduction
    reads its 64 source rows twice, kept as two passes."""
    expected = []
    for group in record["events"].values():
        if group["count"] == 128:
            expected += [(group["first_age"], 64, group["step"]), (group["first_age"] + 64, 64, group["step"])]
        elif group["count"]:
            expected.append((group["first_age"], group["count"], group["step"]))
    return sorted(expected)


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built tools")
class SelectedVmulTimingTest(unittest.TestCase):
    def check(self, ops, accepted, *passes, **options):
        result = selected(ops, *passes, **options)
        self.assertEqual(result.returncode == 0, accepted, result.stderr)
        return result

    def test_every_vpu_operation_takes_its_numbers_from_the_facts(self):
        document = facts()
        for mnemonic, (operation, name) in VPU_OPERATIONS.items():
            record = block(document, name)
            busy = record["first_free_age"] - 1
            for consumer in CONSUMERS:
                with self.subTest(mnemonic=mnemonic, consumer=consumer):
                    result = self.check([operation, HALT], True, consumer, VERIFY)
                    self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                    self.assertNotEqual(run(OPT, without_delays(result.stdout), VERIFY).returncode, 0)
                    footprint = next(i for i in export(result.stdout)["instructions"]
                                     if i["mnemonic"] == mnemonic)["footprint"]
                    self.assertEqual(sorted((a["age"], a["count"], a["step"]) for a in footprint["accesses"]
                                            if a["resource"] == "mreg"), expected_mreg_accesses(record))
                    self.assertEqual(footprint["done_age"], busy)
                    self.assertEqual(footprint["write_release"], record["next_issue_age"])
                    self.assertEqual({h["unit"]: (h["from"], h["to"]) for h in footprint["holds"]},
                                     {u: (0, busy) for u in ("VPU", "VLOAD path", "VSTORE path")})
                    if mnemonic == "vmul.bf16":
                        self.assertEqual(footprint["read_release"], record["events"]["read0"]["last_age"] +
                                         record["assumptions"]["scratchpad_read_latency"])

    def test_register_domain(self):
        for pair, accepted in (((62, 2, 4), True), ((0, 60, 62), True), ((1, 2, 4), False), ((0, 3, 4), False),
                               ((0, 2, 5), False), ((0, 2, 64), False), ((-2, 2, 4), False), ((0, 0, 4), False),
                               ((0, 32, 4), False), ((0, 2, 0), False), ((0, 2, 34), False)):
            for consumer in CONSUMERS:
                with self.subTest(pair=pair, consumer=consumer):
                    self.check([vmul(*pair), HALT], accepted, consumer)
        # Physical bank aliases (register mod 32) pass the dialect verifier and fail in the resolver.
        for op in (unary("mov", 0, 32), reduce("row_sum", 32, 0), reduce("col_sum", 0, 0),
                   pack("bf16_to_fp8", 0, 1), pack("fp8_to_bf16", 4, 4), unary("square", 2, 34)):
            with self.subTest(op=op):
                self.assertIn(RESOLVER, self.check([op, HALT], False, CONSUMERS[0]).stderr)

    def test_facts_drive_the_serialized_boundary(self):
        boundary = block(facts(), "vpu.mul")["first_free_age"]
        for gap in (boundary - 1, boundary):
            for next_op in (vmul(6, 8, 10), vls("vstore", 4)):
                with self.subTest(gap=gap, next_op=next_op):
                    self.check([addi(6, 0, 0), vmul(), delay(gap - 2), next_op, delay(64), nop(), HALT],
                               gap >= boundary, VERIFY)
        with tempfile.TemporaryDirectory(prefix="atlas-vpu-facts-") as directory:
            slower = facts_variant(directory, lambda d: block(d, "vpu.mul").update(first_free_age=70, next_issue_age=69))
            for gap in (69, 70):
                with self.subTest(gap=gap):
                    self.check([addi(6, 0, 0), vmul(), delay(gap - 2), vmul(6, 8, 10), delay(68), nop(), HALT],
                               gap >= 70, VERIFY, path=slower)
            unresolved = facts_variant(directory, lambda d: block(d, "vpu.mul").update(events=None))
            self.assertIn("vpu.mul: facts block is unresolved",
                          self.check([vmul(), HALT], False, CONSUMERS[0], path=unresolved).stderr)
            self.check([vmul(kind="add"), HALT], True, CONSUMERS[0], VERIFY, path=unresolved)
            narrow = facts_variant(directory, lambda d: block(d, "vpu.mul")["events"]["write0"].update(count=32))
            self.assertIn("does not sweep 1x64", self.check([vmul(), HALT], False, CONSUMERS[0], path=narrow).stderr)

    def test_producer_consumer_and_completion_are_rechecked(self):
        ops = [addi(6, 0, 0), addi(8, 0, 1024), vls("vload", 0), vls("vload", 1, offset=32),
               vls("vload", 2, offset=64), vls("vload", 3, offset=96), vmul(), vls("vstore", 4, 8),
               vls("vstore", 5, 8, 32), addi(1, 0, 1), MARKER, HALT]
        for consumer in CONSUMERS:
            result = self.check(ops, True, consumer, VERIFY)
            self.assertEqual(run(EMIT, result.stdout).returncode, 0)
            unsafe = without_delays(result.stdout)
            self.assertNotEqual(run(OPT, unsafe, VERIFY).returncode, 0)
            self.assertNotEqual(run(EMIT, unsafe).returncode, 0)
        for suffix in ([MARKER, delay(64), HALT], [HALT]):
            self.check([vmul(), *suffix], False, VERIFY)

    def test_llvm_reconstruction_preserves_evidence_and_rechecks(self):
        result = self.check([vmul(), HALT], True, CONSUMERS[0], VERIFY)
        unsafe = without_delays(result.stdout)
        for entrypoint in ("--convert-atlas-to-llvm", "--convert-atlas-to-llvm-calls"):
            self.assertNotEqual(run(OPT, unsafe, entrypoint).returncode, 0)
        staged = run(OPT, result.stdout, "--convert-atlas-to-llvm-calls")
        self.assertEqual(staged.returncode, 0, staged.stderr)
        finalized = run(OPT, staged.stdout, "--finalize-atlas-llvm-calls")
        self.assertEqual(finalized.returncode, 0, finalized.stderr)
        self.assertIn("op_timing_sha256", finalized.stdout)

    def test_dma_wait_and_xlu_share_serialized_compute_policy(self):
        from test_rtl_dma_timing import setup, transfer, wait
        from test_rtl_xlu_timing import transpose
        self.check([*setup(), transfer(), vmul(), wait(), HALT], False, CONSUMERS[0], dma="wait")
        for consumer in CONSUMERS:
            result = self.check([*setup(), transfer(), wait(), transpose(0, 2), vmul(2, 4, 6), HALT],
                                True, consumer, VERIFY, dma="wait")
            self.assertEqual(run(EMIT, result.stdout).returncode, 0)


if __name__ == "__main__":
    unittest.main()

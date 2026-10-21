"""Guarded loop pointers remain checked through physical DMA lowering."""

import struct
import unittest

from test_virtual_ssa import ROOT, run


SOURCE = (ROOT / "test/examples/virtual_k_loop.mlir").read_text()
PIPELINE = ("--spill-atlas-virtual", "--lower-atlas-virtual-to-machine",
            "--insert-atlas-dma-waits", "--insert-atlas-delays",
            "--verify-atlas-generated-schedule", "--verify-atlas-timing")


def loop_source(count):
    if type(count) is not int or not 1 <= count <= 7:
        raise ValueError("public execution fixture supports one to seven K tiles")
    return SOURCE.replace("%count = arith.constant 3", f"%count = arith.constant {count}")


def loop_inputs(count):
    """Integer-valued sparse contractions, independently packed and evaluated.

    A uses E4M3 encodings for 1, 2 and 3. W has one signed unit per
    output column; every intermediate integer is exactly BF16-representable.
    This checks pointer progress and carried accumulators, not rounding ties.
    """
    codes = (0x38, 0x40, 0x44)
    inputs = []
    result = [[0] * 32 for _ in range(32)]
    for t in range(count):
        a = bytes(codes[(r + k + t) % 3] for r in range(32) for k in range(32))
        w = bytes((0xb8 if c % 2 else 0x38) if k == (c + t) % 32 else 0
                  for c in range(32) for k in range(32))
        inputs.extend(((0x90000000 + t * 1024, a), (0x90010000 + t * 1024, w)))
        for r in range(32):
            for c in range(32):
                result[r][c] += (-1 if c % 2 else 1) * (1 + (r + (c + t) % 32 + t) % 3)
    output = bytearray(2048)
    for r in range(32):
        for c in range(32):
            bits = struct.unpack("<I", struct.pack("<f", result[r][c]))[0]
            assert bits & 0xffff == 0
            struct.pack_into("<H", output, (c // 16) * 1024 + (r * 16 + c % 16) * 2, bits >> 16)
    return inputs, bytes(output)


class BoundedDMAAddressTest(unittest.TestCase):
    def test_k_loop_survives_waits_timing_and_reparse(self):
        result = run("atlas-opt", SOURCE, *PIPELINE)
        self.assertEqual(result.returncode, 0, result.stderr[:2000])
        self.assertIn("dram_span = 3072 : i32", result.stdout)
        checked = run("atlas-opt", result.stdout,
                      "--verify-atlas-generated-schedule", "--verify-atlas-timing")
        self.assertEqual(checked.returncode, 0, checked.stderr[:2000])

    def test_different_trip_counts_reuse_the_same_instruction_body(self):
        sizes = []
        for count in (1, 2, 3, 7):
            machine = run("atlas-opt", loop_source(count), *PIPELINE)
            self.assertEqual(machine.returncode, 0, machine.stderr[:1200])
            emitted = run("atlas-emit", machine.stdout)
            self.assertEqual(emitted.returncode, 0, emitted.stderr[:1200])
            sizes.append(len(emitted.stdout.splitlines()))
        self.assertEqual(len(set(sizes)), 1, sizes)

    def test_changed_source_bound_or_recorded_span_is_rejected(self):
        machine = run("atlas-opt", SOURCE, *PIPELINE)
        self.assertEqual(machine.returncode, 0, machine.stderr[:1200])
        for old, new in (("dram_span = 3072", "dram_span = 4096"),
                         ("dram_value = 5", "dram_value = 6"),
                         ("constant = 3 : i32", "constant = 4 : i32")):
            with self.subTest(field=old):
                self.assertIn(old, machine.stdout)
                mutated = machine.stdout.replace(old, new, 1)
                checked = run("atlas-opt", mutated, "--verify-atlas-generated-schedule")
                self.assertNotEqual(checked.returncode, 0)
                self.assertIn("bounded DMA address", checked.stderr)

    def test_emitted_pointer_and_tile_span_cannot_bypass_source_checks(self):
        machine = run("atlas-opt", SOURCE, *PIPELINE)
        self.assertEqual(machine.returncode, 0, machine.stderr[:1200])
        launch = next(line for line in machine.stdout.splitlines() if '\"atlas.dma\"' in line)
        reg = launch.split("dram = ", 1)[1].split(" :", 1)[0]
        wrong = "1" if reg != "1" else "2"
        mutations = [machine.stdout.replace(launch, launch.replace(f"dram = {reg} : i32", f"dram = {wrong} : i32"), 1),
                     machine.stdout.replace("dram_span_bytes = 3072", "dram_span_bytes = 4096", 1)]
        for mutated in mutations:
            self.assertNotEqual(mutated, machine.stdout)
            checked = run("atlas-opt", mutated, "--verify-atlas-generated-schedule")
            self.assertNotEqual(checked.returncode, 0)
            self.assertEqual(checked.stdout, "")

    def test_control_mailbox_checks_every_iteration_address(self):
        source = SOURCE.replace("@stream()", "@stream(%control: i32)").replace(
            "atlas.input_dram_base =", "atlas.control_dram_base = 2416115712 : i64, atlas.input_dram_base =")
        machine = run("atlas-opt", source, *PIPELINE)
        self.assertEqual(machine.returncode, 0, machine.stderr[:1200])
        # 0x90000800 is the third A tile, beyond the default boundary tile.
        overlap = source.replace("atlas.control_dram_base = 2416115712", "atlas.control_dram_base = 2415921152")
        checked = run("atlas-opt", overlap, *PIPELINE)
        self.assertNotEqual(checked.returncode, 0)
        self.assertIn("explicit DMA span overlaps the control mailbox", checked.stderr)

    def test_unguarded_or_wrapping_inductions_are_rejected(self):
        mutations = [
            ("cmpi ult", "cmpi ule"),
            ("%count = arith.constant 3", "%count = arith.constant 0"),
            ("%one = arith.constant 1", "%one = arith.constant 2"),
            ("%an = arith.addi %ap, %bytes", "%an = arith.addi %ap, %one"),
            ("%a = arith.constant -1879048192", "%a = arith.constant -1024"),
            ("%a = arith.constant -1879048192", "%a = arith.constant -1879048191"),
            ("cf.cond_br %more, ^body(%ls : !atlas.virtual_state), ^exit(%ls, %acc : !atlas.virtual_state, !atlas.virtual_bf16)",
             "cf.cond_br %more, ^exit(%ls, %acc : !atlas.virtual_state, !atlas.virtual_bf16), ^body(%ls : !atlas.virtual_state)"),
        ]
        for old, new in mutations:
            with self.subTest(new=new):
                self.assertIn(old, SOURCE)
                result = run("atlas-opt", SOURCE.replace(old, new), "--verify-atlas-virtual-stream")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()

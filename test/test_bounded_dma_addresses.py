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


def pointer_loop_source(*, start=0x90000000, distance=3072, step=1024):
    """Use the advancing A address itself as the unsigned loop guard."""
    def signed(value):
        return value if value < 2**31 else value-2**32
    source = SOURCE.replace('  %zero = arith.constant 0 : i32\n', '')
    source = source.replace('%s, %zero, %a, %w, %initial : !atlas.virtual_state, i32, i32, i32, !atlas.virtual_bf16',
                            '%s, %a, %w, %initial : !atlas.virtual_state, i32, i32, !atlas.virtual_bf16')
    source = source.replace('%ls: !atlas.virtual_state, %k: i32, %ap: i32',
                            '%ls: !atlas.virtual_state, %ap: i32')
    source = source.replace('%a = arith.constant -1879048192',
                            f'%a = arith.constant {signed(start)}')
    source = source.replace('%count = arith.constant 3',
                            f'%count = arith.constant {signed(start+distance)}')
    source = source.replace('cmpi ult, %k, %count', 'cmpi ult, %ap, %count')
    source = source.replace('  %one = arith.constant 1 : i32\n',
                            f'  %advance = arith.constant {signed(step)} : i32\n')
    source = source.replace('  %kn = arith.addi %k, %one : i32\n', '')
    source = source.replace('%an = arith.addi %ap, %bytes', '%an = arith.addi %ap, %advance')
    return source.replace('%s6, %kn, %an, %wn, %tile : !atlas.virtual_state, i32, i32, i32, !atlas.virtual_bf16',
                          '%s6, %an, %wn, %tile : !atlas.virtual_state, i32, i32, !atlas.virtual_bf16')


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
    def test_pointer_guard_uses_ceil_trip_count_without_a_separate_counter(self):
        for distance, step, span in ((1, 1024, 1024), (2049, 1024, 3072),
                                     (3072, 1024, 3072), (3104, 1536, 4096)):
            with self.subTest(distance=distance, step=step):
                source = pointer_loop_source(distance=distance, step=step)
                self.assertNotIn('%k:', source)
                machine = run('atlas-opt', source, *PIPELINE)
                self.assertEqual(machine.returncode, 0, machine.stderr[:2000])
                self.assertIn(f'dram_span = {span} : i32', machine.stdout)
                checked = run('atlas-opt', machine.stdout,
                              '--verify-atlas-generated-schedule', '--verify-atlas-timing')
                self.assertEqual(checked.returncode, 0, checked.stderr[:2000])

    def test_pointer_guard_replay_rejects_forged_start_limit_and_step(self):
        machine = run('atlas-opt', pointer_loop_source(distance=3104, step=1536), *PIPELINE)
        self.assertEqual(machine.returncode, 0, machine.stderr[:2000])
        for old, new in (('constant = -1879048192 : i32', 'constant = -1879048160 : i32'),
                         ('constant = -1879045088 : i32', 'constant = -1879043552 : i32'),
                         ('constant = 1536 : i32', 'constant = 1568 : i32'),
                         ('dram_span = 4096 : i32', 'dram_span = 5632 : i32')):
            with self.subTest(field=old):
                self.assertIn(old, machine.stdout)
                checked = run('atlas-opt', machine.stdout.replace(old, new, 1),
                              '--verify-atlas-generated-schedule')
                self.assertNotEqual(checked.returncode, 0)
                self.assertIn('bounded DMA address', checked.stderr)

    def test_pointer_guard_refuses_unknown_nonincreasing_and_wrapped_exit(self):
        source = pointer_loop_source()
        invalid = [pointer_loop_source(distance=0), pointer_loop_source(distance=-1024),
                   pointer_loop_source(step=0), pointer_loop_source(step=2**32-1024),
                   # The first DMA tile fits DRAM, but the guard's
                   # final update wraps and would incorrectly enter again.
                   pointer_loop_source(start=0xfffffc00, distance=1023),
                   source.replace('cmpi ult', 'cmpi ule'),
                   source.replace('@stream()', '@stream(%unknown: i32)').replace(
                       '%an = arith.addi %ap, %advance', '%an = arith.addi %ap, %unknown'),
                   source.replace('@stream()', '@stream(%unknown: i32)').replace(
                       'cmpi ult, %ap, %count', 'cmpi ult, %ap, %unknown'),
                   source.replace('@stream()', '@stream(%unknown: i32)').replace(
                       'cf.br ^loop(%s, %a, %w', 'cf.br ^loop(%s, %unknown, %w'),
                   source.replace('%wn = arith.addi %wp, %bytes', '%wn = arith.addi %wp, %ap'),
                   source.replace('%w = arith.constant -1878982656', '%w = arith.constant -1024')]
        for index, changed in enumerate(invalid):
            with self.subTest(case=index):
                checked = run('atlas-opt', changed, '--verify-atlas-virtual-stream')
                self.assertNotEqual(checked.returncode, 0)
                self.assertEqual(checked.stdout, '')

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

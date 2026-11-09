"""Generic byte/half/word slot indexing and independent contract mutations."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
STATE, SLOT, BF16 = (f"!atlas.virtual_{kind}" for kind in ("state", "spill", "bf16"))


def run(source, *flags):
    return subprocess.run([str(BIN / "atlas-opt"), "--mlir-print-op-on-diagnostic=false", *flags, "-"],
                          input=source, text=True, capture_output=True)


def source(width=4, *, mask=None, offset=0, initialize=True, indexed=True):
    load, store = {1: ("lbu", "sb"), 2: ("lhu", "sh"), 4: ("lw", "sw")}[width]
    if mask is None:
        mask = 2048 // width - 1
    index = ", %index" if indexed else ""
    index_type = ", i32" if indexed else ""
    initial = (f'%s1, %input = "atlas.virtual_input_bf16"(%s0) {{index = 0 : i32}} : ({STATE}) -> ({STATE}, {BF16})\n'
               f'%s2 = "atlas.virtual_save"(%s1, %slot, %input) : ({STATE}, {SLOT}, {BF16}) -> {STATE}') if initialize else ''
    return f'''module {{
      func.func @test(%runtime: i32) -> {STATE} attributes {{atlas.control_dram_base = 2415984640 : i64, atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2416050176 : i64}} {{
        %s0 = "atlas.virtual_start"() : () -> {STATE}
        %slot = "atlas.virtual_slot"() {{kind = "bf16", words = 512 : i32, slot_word = -1 : i32, purpose = "scratch"}} : () -> {SLOT}
        %mask = arith.constant {mask} : i32
        %index = arith.andi %runtime, %mask : i32
        %value = arith.constant 42 : i32
        {initial}
        %s3 = "atlas.virtual_slot_store"(%{'s2' if initialize else 's0'}, %slot, %value{index}) {{kind = "{store}", byte_offset = {offset} : i32}} : ({STATE}, {SLOT}, i32{index_type}) -> {STATE}
        %s4, %loaded = "atlas.virtual_slot_load"(%s3, %slot{index}) {{kind = "{load}", byte_offset = {offset} : i32}} : ({STATE}, {SLOT}{index_type}) -> ({STATE}, i32)
        %zero = "atlas.virtual_vpu_splat"() {{bits = 0 : i32}} : () -> {BF16}
        %s5 = "atlas.virtual_output_bf16"(%s4, %zero) {{index = 0 : i32}} : ({STATE}, {BF16}) -> {STATE}
        return %s5 : {STATE}
      }}
    }}'''


def compile_source(text):
    return run(text, "--spill-atlas-virtual", "--lower-atlas-virtual-to-machine", "--insert-atlas-dma-waits",
               "--insert-atlas-delays", "--verify-atlas-generated-schedule", "--verify-atlas-timing")


class IndexedSlotTests(unittest.TestCase):
    def test_widths_and_fixed_offsets(self):
        for width in (1, 2, 4):
            for indexed in (False, True):
                with self.subTest(width=width, indexed=indexed):
                    result = compile_source(source(width, indexed=indexed))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('slot_index_load' if indexed else 'slot_load', result.stdout)

    def test_offset_plus_scaled_range(self):
        result = compile_source(source(2, mask=511, offset=1024))
        self.assertEqual(result.returncode, 0, result.stderr)
        for width, mask, offset in ((2, 1023, 2), (4, 512, 0), (1, 2048, 0)):
            with self.subTest(width=width, mask=mask, offset=offset):
                result = compile_source(source(width, mask=mask, offset=offset))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('proven unsigned range', result.stderr)

    def test_counted_index_uses_guarded_range(self):
        text = source(mask=511)
        text = text.replace('%index = arith.andi %runtime, %mask : i32', '%index = arith.constant 0 : i32')
        start = text.index('        %s3 =')
        end = text.index('        %zero =')
        accesses = text[start:end].replace('%s2, %slot', '%body_state, %slot').replace('%index', '%lane')
        text = text[:start] + f"""
        %one = arith.constant 1 : i32
        %bound = arith.constant 512 : i32
        cf.br ^loop(%s2, %index : {STATE}, i32)
      ^loop(%loop_state: {STATE}, %lane: i32):
        %more = arith.cmpi ult, %lane, %bound : i32
        cf.cond_br %more, ^body(%loop_state : {STATE}), ^done(%loop_state : {STATE})
      ^body(%body_state: {STATE}):
        {accesses}
        %next = arith.addi %lane, %one : i32
        cf.br ^loop(%s4, %next : {STATE}, i32)
      ^done(%done_state: {STATE}):
        %zero = "atlas.virtual_vpu_splat"() {{bits = 0 : i32}} : () -> {BF16}
        %s5 = "atlas.virtual_output_bf16"(%done_state, %zero) {{index = 0 : i32}} : ({STATE}, {BF16}) -> {STATE}
        return %s5 : {STATE}
      }}
    }}"""
        result = compile_source(text)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unbounded_index_rejected(self):
        result = compile_source(source().replace('%index = arith.andi %runtime, %mask : i32',
                                                '%index = arith.addi %runtime, %mask : i32'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('proven unsigned range', result.stderr)

    def test_dynamic_store_cannot_initialize_possible_destinations(self):
        result = compile_source(source(initialize=False))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('uninitialized', result.stderr)

    def test_mutated_index_origin_rejected(self):
        result = compile_source(source())
        self.assertEqual(result.returncode, 0, result.stderr)
        machine, count = re.subn(r'("atlas\.alu_reg"\([^\n]*?kind = "add"[^\n]*?rhs = )\d+',
                                 r'\g<1>0', result.stdout, count=1)
        self.assertEqual(count, 1)
        checked = run(machine, '--verify-atlas-generated-schedule')
        self.assertNotEqual(checked.returncode, 0)

    def test_captured_range_is_replayed(self):
        result = compile_source(source())
        self.assertEqual(result.returncode, 0, result.stderr)
        # Widen only the captured constant feeding the runtime mask; leave the
        # physical machine and tile footprint untouched.
        machine, count = re.subn(r'constant = 511 : i32', 'constant = 1023 : i32', result.stdout, count=1)
        self.assertEqual(count, 1)
        checked = run(machine, '--verify-atlas-generated-schedule')
        self.assertNotEqual(checked.returncode, 0)

    def test_conservative_footprint_covers_the_proven_range(self):
        result = compile_source(source(mask=7))
        self.assertEqual(result.returncode, 0, result.stderr)
        pattern = r'bytes = 32 : i32(?=[^{}]*kind = "slot_index_(?:load|store)")'
        for size, valid in ((64, True), (16, False)):
            machine, count = re.subn(pattern, f'bytes = {size} : i32', result.stdout)
            self.assertEqual(count, 2)
            checked = run(machine, '--verify-atlas-generated-schedule')
            self.assertEqual(checked.returncode == 0, valid, checked.stderr)


if __name__ == '__main__':
    unittest.main()

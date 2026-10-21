"""Temporary DRAM reads require prior writes; allocation does not initialize data."""

import os
import unittest

from test_virtual_ssa import run


def source(*, hole=False, read_first=False):
    state, bf16, fp8 = (
        "!atlas.virtual_state",
        "!atlas.virtual_bf16",
        "!atlas.virtual_fp8",
    )
    load_address = 0x90010800 if hole else 0x90010000
    write = f'    %s2 = "atlas.virtual_dma_store_bf16"(%s1,%a,%temporary,%bytes) : ({state},{bf16},i32,i32) -> {state}'
    read = f'    %s3, %b = "atlas.virtual_dma_load_bf16"(%s2,%read_address,%bytes) : ({state},i32,i32) -> ({state},{bf16})'
    if read_first:
        read = read.replace("%s3, %b", "%s2, %b").replace("(%s2,", "(%s1,")
        write = write.replace("%s2 =", "%s3 =").replace("(%s1,", "(%s2,")
    effects = "\n".join((read, write) if read_first else (write, read))
    return f"""module {{
  func.func @temporary() -> {state} attributes {{atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415984640 : i64, atlas.output_dram_bytes = 6144 : i64}} {{
    %s0 = "atlas.virtual_start"() : () -> {state}
    %input = arith.constant -1879048192 : i32
    %temporary = arith.constant -1878982656 : i32
    %read_address = arith.constant {load_address - (1 << 32)} : i32
    %output = arith.constant -1878978560 : i32
    %bytes = arith.constant 2048 : i32
    %packed_bytes = arith.constant 1024 : i32
    %s1, %a = "atlas.virtual_dma_load_bf16"(%s0,%input,%bytes) : ({state},i32,i32) -> ({state},{bf16})
{effects}
    %packed = "atlas.virtual_pack_fp8"(%b) {{scale_code = 127 : i32}} : ({bf16}) -> {fp8}
    %s4 = "atlas.virtual_dma_store_fp8"(%s3,%packed,%output,%packed_bytes) : ({state},{fp8},i32,i32) -> {state}
    return %s4 : {state}
  }}
}}
"""


@unittest.skipUnless(os.environ.get("ATLAS_OOT_BIN_DIR"), "select Atlas tools")
class MemoryInitializationTest(unittest.TestCase):
    def test_written_temporary_passes_source_and_generated_checks(self):
        compiled = run(
            "atlas-opt",
            source(),
            "--verify-atlas-virtual-stream",
            "--spill-atlas-virtual",
            "--lower-atlas-virtual-to-machine",
            "--insert-atlas-dma-waits",
            "--insert-atlas-delays",
            "--verify-atlas-generated-schedule",
            "--verify-atlas-machine-stream",
            "--verify-atlas-timing",
        )
        self.assertEqual(compiled.returncode, 0, compiled.stderr[:2000])

    def test_unwritten_hole_and_read_before_store_fail(self):
        for options in ({"hole": True}, {"read_first": True}):
            with self.subTest(options=options):
                result = run(
                    "atlas-opt", source(**options), "--verify-atlas-virtual-stream"
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("before they are initialized", result.stderr)

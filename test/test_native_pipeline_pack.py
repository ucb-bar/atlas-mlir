"""Optional native pipeline and exact on-chip PACK relayout regressions."""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MERLIN = os.environ.get("MERLIN_NATIVE_ROOT")
ATLAS_BIN = os.environ.get("ATLAS_OOT_BIN_DIR")
PACK_VIRTUAL = '''module {
  func.func @pack() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415925248 : i64, atlas.output_dram_base = 2415990784 : i64} {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %x = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %y = "atlas.virtual_pack_fp8"(%x) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %out = arith.constant -1878976512 : i32
    %bytes = arith.constant 1024 : i32
    %s2 = "atlas.virtual_dma_store_fp8"(%s1, %y, %out, %bytes) : (!atlas.virtual_state, !atlas.virtual_fp8, i32, i32) -> !atlas.virtual_state
    return %s2 : !atlas.virtual_state
  }
}
'''


@unittest.skipUnless(MERLIN and ATLAS_BIN, "set MERLIN_NATIVE_ROOT and ATLAS_OOT_BIN_DIR")
class NativePipelinePackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        for path in (ROOT / "python", Path(MERLIN) / "src"):
            sys.path.insert(0, str(path))
            cls.addClassCleanup(sys.path.remove, str(path))
        from atlas_mlir_native import binding
        cls.binding = binding
        machine = binding._run(
            [str(Path(ATLAS_BIN) / "atlas-opt"), *binding.NATIVE_PIPELINES["mandatory"], "-"],
            input_text=PACK_VIRTUAL,
        )
        cls.program = json.loads(binding._run(
            [str(Path(ATLAS_BIN) / "atlas-emit"), "--program-json", "-"], input_text=machine,
        ))

    def check_pack(self, program) -> None:
        self.binding.check_pack_stream(
            program, input_address=0x90001800, output_address=0x90011800,
        )

    def test_pipeline_default_and_choices(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.binding._native_pipeline(), "mandatory")
        for selected in ("mandatory", "optimized"):
            with self.subTest(selected=selected), patch.dict(os.environ, ATLAS_NATIVE_PIPELINE=selected):
                self.assertEqual(self.binding._native_pipeline(), selected)
        with patch.dict(os.environ, ATLAS_NATIVE_PIPELINE="typo"):
            with self.assertRaisesRegex(self.binding.NativeCompilationError, "mandatory or optimized"):
                self.binding._native_pipeline()

    def test_optimized_pipeline_accepts_same_raw_virtual(self) -> None:
        machine = self.binding._run(
            [str(Path(ATLAS_BIN) / "atlas-opt"), *self.binding.NATIVE_PIPELINES["optimized"], "-"],
            input_text=PACK_VIRTUAL,
        )
        self.binding._run(
            [str(Path(ATLAS_BIN) / "atlas-opt"), *self.binding.NATIVE_CHECKS, "-"],
            input_text=machine,
        )

    def test_optimized_pipeline_accepts_mxu_pressure(self) -> None:
        from test_mxu_spilling import mxu_pressure

        machine = self.binding._run(
            [str(Path(ATLAS_BIN) / "atlas-opt"), *self.binding.NATIVE_PIPELINES["optimized"], "-"],
            input_text=mxu_pressure(count=5, reset=True),
        )
        self.binding._run(
            [str(Path(ATLAS_BIN) / "atlas-opt"), *self.binding.NATIVE_CHECKS, "-"],
            input_text=machine,
        )

    def test_relu_before_pack_keeps_exact_on_chip_roundtrip(self) -> None:
        source = PACK_VIRTUAL.replace('    %y = "atlas.virtual_pack_fp8"(%x)',
            '    %relu = "atlas.virtual_vpu_unary"(%x) {kind = "relu"} '
            ': (!atlas.virtual_bf16) -> !atlas.virtual_bf16\n' '    %y = "atlas.virtual_pack_fp8"(%relu)')
        machine = self.binding._run([str(Path(ATLAS_BIN) / "atlas-opt"), *self.binding.NATIVE_PIPELINES["mandatory"],
            "-"], input_text=source,)
        program = json.loads(self.binding._run([str(Path(ATLAS_BIN) / "atlas-emit"), "--program-json", "-"],
            input_text=machine,))
        self.binding.check_pack_stream(program, input_address=0x90001800, output_address=0x90011800, source_pair=32,
            relu_pair=(34, 32),)
        self.assertEqual(program["word_count"], 39)

    def test_pack_uses_exact_on_chip_roundtrip(self) -> None:
        self.check_pack(self.program)
        rows = self.program["instructions"]
        conversions = [row["fields"] for row in rows if row["operation"] == "atlas.vpu_pack"]
        self.assertEqual([fields["direction"] for fields in conversions], ["bf16_to_fp8", "fp8_to_bf16"])
        self.assertEqual((conversions[0]["dst"], conversions[1]["src"]), (62, 62))
        self.assertEqual(sum(row["operation"] == "atlas.mxu_push" for row in rows), 1)
        self.assertEqual(sum(row["operation"] == "atlas.mxu_pop" for row in rows), 1)
        self.assertFalse(any(row["operation"] in {"atlas.branch", "atlas.scalar_store"} for row in rows))

    def test_pack_rejects_changed_conversion_fields_scales_and_words(self) -> None:
        rows = self.program["instructions"]
        selected = { "pack": next(i for i, row in enumerate(rows) if row["operation"] == "atlas.vpu_pack"), "unpack":
            next(i for i, row in enumerate(rows) if row["operation"] == "atlas.vpu_pack" and row["fields"]["direction"]
            == "fp8_to_bf16"), "push": next(i for i, row in enumerate(rows) if row["operation"] == "atlas.mxu_push"),
            "pop": next(i for i, row in enumerate(rows) if row["operation"] == "atlas.mxu_pop"), }
        mutations = [(index, "offset", 126) for index, row in enumerate(rows)
                     if row["operation"] == "atlas.scalar_load" and row["fields"].get("kind") == "seli"]
        mutations.extend((selected[name], field, value) for name, field, value in (("pack", "src", 0), ("pack", "dst",
            60), ("pack", "scale_reg", 2), ("unpack", "src", 60), ("unpack", "dst", 36), ("unpack", "direction",
            "bf16_to_fp8"), ("push", "src", 36), ("push", "kind", "acc_fp8"), ("pop", "scale_reg", 2), ("pop", "unit",
            1),))
        for index, field, value in mutations:
            with self.subTest(index=index, field=field):
                program = copy.deepcopy(self.program)
                program["instructions"][index]["fields"][field] = value
                with self.assertRaises(self.binding.NativeCompilationError):
                    self.check_pack(program)
        for index in (*selected.values(), *(i for i, row in enumerate(rows)
                        if row["operation"] == "atlas.scalar_load")):
            with self.subTest(encoded_word=index):
                program = copy.deepcopy(self.program)
                program["instructions"][index]["word_u32"] ^= 1 << 7
                with self.assertRaises(self.binding.NativeCompilationError):
                    self.check_pack(program)

    def test_pack_rejects_shortened_conversion_delays(self) -> None:
        delays = [index for index, row in enumerate(self.program["instructions"]) if row["operation"] == "atlas.delay"]
        for index in delays:
            with self.subTest(delay=index):
                program = copy.deepcopy(self.program)
                program["instructions"][index]["fields"]["cycles"] = 0
                with self.assertRaises(self.binding.NativeCompilationError):
                    self.check_pack(program)


if __name__ == "__main__":
    unittest.main()

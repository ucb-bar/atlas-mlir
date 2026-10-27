"""Optional native pipeline and emitted PACK-copy regressions."""

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

    def test_pack_copy_preserves_all_32_rows(self) -> None:
        self.check_pack(self.program)
        rows = self.program["instructions"]
        branch = next(i for i, row in enumerate(rows) if row["operation"] == "atlas.branch")
        start = branch + rows[branch]["fields"]["offset_bytes"] // 2
        registers = {0: 0, 10: 131072, 11: 131584, 12: 132096, 13: 0, 14: 32}
        memory = {131072 + i: tuple(range(i, i + 4)) for i in range(0, 1024, 4)}
        stores = {}
        for _ in range(32):
            for row in rows[start:branch]:
                op, fields = row["operation"], row["fields"]
                if op == "atlas.scalar_load":
                    registers[fields["dst"]] = memory[registers[fields["base"]] + fields["offset"]]
                elif op == "atlas.scalar_store":
                    address = registers[fields["base"]] + fields["offset"]
                    self.assertNotIn(address, stores)
                    stores[address] = registers[fields["src"]]
                elif op == "atlas.alu_imm":
                    registers[fields["dst"]] = registers[fields["src"]] + fields["immediate"]
                else:
                    self.assertEqual(op, "atlas.delay")
        self.assertEqual(registers[13], registers[14])
        self.assertEqual(len(stores), 256)
        for row in range(32):
            for half in range(2):
                for word in range(4):
                    source = half * 512 + row * 16 + word * 4
                    self.assertEqual(stores[132096 + row * 32 + half * 16 + word * 4],
                                     tuple(range(source, source + 4)))

    def test_pack_rejects_corrupt_address_temporary_and_branch(self) -> None:
        for case in ("address", "last-word-address", "temporary", "branch"):
            with self.subTest(case=case):
                program = copy.deepcopy(self.program)
                if case == "address":
                    row = next(row for row in program["instructions"]
                               if row["operation"] == "atlas.scalar_load" and row["fields"].get("kind") == "lw")
                    row["fields"]["offset"] += 4
                elif case == "last-word-address":
                    row = next(row for row in program["instructions"] if row["operation"] == "atlas.scalar_load"
                        and row["fields"].get("kind") == "lw" and row["fields"]["offset"] == 12)
                    row["fields"]["offset"] += 4
                elif case == "temporary":
                    row = next(row for row in program["instructions"] if row["operation"] == "atlas.scalar_store")
                    row["fields"]["src"] = 17 if row["fields"]["src"] == 16 else 16
                else:
                    row = next(row for row in program["instructions"] if row["operation"] == "atlas.branch")
                    row["fields"]["offset_bytes"] += 2
                with self.assertRaises(self.binding.NativeCompilationError):
                    self.check_pack(program)

    def test_pack_rejects_two_row_advances(self) -> None:
        for register in (13, 10, 11, 12):
            with self.subTest(register=register):
                program = copy.deepcopy(self.program)
                row = next(row for row in program["instructions"] if row["operation"] == "atlas.alu_imm"
                    and row["fields"].get("dst") == register and row["fields"].get("src") == register)
                row["fields"]["immediate"] *= 2
                with self.assertRaises(self.binding.NativeCompilationError):
                    self.check_pack(program)


if __name__ == "__main__":
    unittest.main()

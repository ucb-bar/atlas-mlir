"""Physical program export for a separate Atlas instruction model.

These tests exercise the exported control information. They do not implement
the tensor instruction semantics or claim a cycle-accurate functional model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import unittest

from test_dialect import program as machine_program, variants


ROOT = Path(__file__).resolve().parents[1]
EMIT = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin")) / "atlas-emit"


def emit(source: Path, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(EMIT), *options, str(source)], text=True,
                          capture_output=True, check=False)


def physical_program(name: str) -> dict:
    source = ROOT / "test/examples" / name
    result = emit(source, "--program-json")
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


class FunctionalStreamContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(EMIT.is_file(), "build atlas-emit first")

    def test_branch_fields_are_typed_and_words_are_authoritative(self) -> None:
        source = ROOT / "test/examples/branch_delay.mlir"
        program = physical_program(source.name)
        words = [int(line, 16) for line in emit(source).stdout.splitlines()]
        self.assertEqual(program["schema"], "atlas.physical_program.v1")
        self.assertEqual(program["pc_unit"], "instruction_word")
        self.assertEqual(program["word_count"], len(words))
        self.assertEqual([row["word_u32"] for row in program["instructions"]], words)
        for pc, row in enumerate(program["instructions"]):
            self.assertEqual(row["word_index"], pc)
            self.assertEqual(row["word_hex"], f"{words[pc]:08x}")

        branch = program["instructions"][1]
        self.assertEqual(branch["operation"], "atlas.branch")
        self.assertEqual(branch["fields"],
                         {"kind": "beq", "lhs": 0, "rhs": 0, "offset_bytes": 6})
        self.assertEqual(branch["control"], {
            "kind": "conditional_direct", "delay_slot_word_index": 2,
            "target_word_index": 4, "fallthrough_word_index": 3,
        })
        self.assertEqual(program["instructions"][2]["delay_slot_for_word_index"], 1)
        self.assertEqual(program["instructions"][4]["fields"]["immediate"], -169)
        # A minimal consumer follows the branch, executes the one delay slot,
        # and then arrives at the selected target. Word 3 is not executed.
        trace = [0, 1, branch["control"]["delay_slot_word_index"],
                 branch["control"]["target_word_index"], 5, 6]
        self.assertEqual(trace, [0, 1, 2, 4, 5, 6])

    def test_direct_and_register_jumps_expose_distinct_targets(self) -> None:
        direct = physical_program("jal_direct_target.mlir")["instructions"]
        direct_jump = next(row for row in direct if row["operation"] == "atlas.jump")
        self.assertEqual(direct_jump["control"]["kind"], "jump_direct")
        self.assertIn("target_word_index", direct_jump["control"])
        self.assertEqual(
            direct[direct_jump["control"]["delay_slot_word_index"]]
            ["delay_slot_for_word_index"], direct_jump["word_index"])

        indirect = physical_program("jalr_word_target.mlir")["instructions"]
        register_jump = next(row for row in indirect if row["operation"] == "atlas.jump")
        self.assertEqual(register_jump["control"], {
            "kind": "jump_register", "delay_slot_word_index": 2,
            "base_register": 1, "offset_words": -1,
        })
        self.assertNotIn("target_word_index", register_jump["control"])
        self.assertEqual(indirect[2]["delay_slot_for_word_index"], 1)

    def test_captured_mlp_machine_stream_exports_without_virtual_values(self) -> None:
        source = ROOT / "examples/handoff/captured_mlp/02-atlas-machine.mlir"
        exported = emit(source, "--program-json")
        self.assertEqual(exported.returncode, 0, exported.stderr)
        program = json.loads(exported.stdout)
        words = [int(line, 16) for line in emit(source).stdout.splitlines()]
        self.assertEqual(program["word_count"], 143)
        self.assertEqual([row["word_u32"] for row in program["instructions"]], words)
        self.assertTrue(any(row["operation"] == "atlas.mxu_matmul"
                            for row in program["instructions"]))
        self.assertTrue(all("virtual" not in row["operation"]
                            for row in program["instructions"]))
        self.assertEqual(program["instructions"][-1]["control"]["kind"], "trap")

    def test_all_fifty_selected_custom_variants_keep_typed_fields(self) -> None:
        cases = variants()
        source = machine_program(cases)
        result = subprocess.run([str(EMIT), "--program-json", "-"],
                                input=source, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = json.loads(result.stdout)["instructions"]
        self.assertEqual(len(rows), 50)
        for row, (op, fields, _) in zip(rows, cases, strict=True):
            self.assertEqual(row["operation"], f"atlas.{op}")
            self.assertEqual(row["fields"], fields)
        self.assertIs(rows[next(i for i, case in enumerate(cases)
                                if case[2] == "VMATMUL_ACC_MXU0")]
                      ["fields"]["accumulate"], True)

    def test_virtual_ir_cannot_be_exported_as_a_physical_program(self) -> None:
        source = ROOT / "test/examples/virtual_bf16_cfg.mlir"
        result = emit(source, "--program-json")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()

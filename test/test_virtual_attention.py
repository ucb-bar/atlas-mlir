"""Attention tile operations preserve fields through scheduling and encoding."""

from __future__ import annotations

import json
import re
import unittest

from test_virtual_lowering import EXAMPLES, lower, run, virtual_chain
from verification_support import assert_boundaries, rewrite_line


BF16 = "!atlas.virtual_bf16"
UNARY = {"mov": 0x40, "recip": 0x41, "exp": 0x42, "exp2": 0x43,
         "square": 0x46, "cube": 0x47, "relu": 0x48, "sin": 0x49,
         "cos": 0x4A, "tanh": 0x4B, "log2": 0x4C, "sqrt": 0x4D}
BINARY = {"add": 0, "sub": 2, "mul": 3, "min": 4, "max": 6}
REDUCE = {"row_sum": 0x21, "row_min": 0x24, "row_max": 0x26}


def with_operation(operation: str) -> str:
    original = f'"atlas.virtual_vpu_unary"(%t0) {{kind = "relu"}} : ({BF16}) -> {BF16}'
    return virtual_chain(1).replace(original, operation)


class VirtualAttentionTest(unittest.TestCase):
    def compile(self, source: str) -> list[dict]:
        result = run("atlas-opt", source, "--verify-atlas-virtual-stream",
                     "--schedule-atlas-virtual", "--lower-atlas-virtual-to-machine",
                     "--insert-atlas-delays", "--verify-atlas-generated-schedule")
        self.assertEqual(result.returncode, 0, result.stderr)
        encoded = run("atlas-emit", result.stdout, "--program-json")
        self.assertEqual(encoded.returncode, 0, encoded.stderr)
        return json.loads(encoded.stdout)["instructions"]

    def test_every_admitted_vpu_kind_schedules_lowers_and_encodes(self) -> None:
        for family, kinds in (("unary", UNARY), ("binary", BINARY), ("reduce", REDUCE)):
            operands = "%t0, %t0" if family == "binary" else "%t0"
            types = f"{BF16}, {BF16}" if family == "binary" else BF16
            for kind, funct7 in kinds.items():
                with self.subTest(family=family, kind=kind):
                    instructions = self.compile(with_operation(
                        f'"atlas.virtual_vpu_{family}"({operands}) '
                        f'{{kind = "{kind}"}} : ({types}) -> {BF16}'))
                    matching = [op for op in instructions
                                if op["operation"] == f"atlas.vpu_{family}"]
                    self.assertEqual(len(matching), 1)
                    op = matching[0]
                    fields = op["fields"]
                    self.assertEqual(fields["kind"], kind)
                    src = fields["lhs" if family == "binary" else "src"]
                    dst = fields["dst"]
                    self.assertEqual(dst % 2, 0)
                    self.assertEqual(src % 2, 0)
                    self.assertNotEqual(dst, src)
                    rhs = fields["rhs"] if family == "binary" else 0
                    self.assertEqual(op["word_u32"],
                                     (funct7 << 25) | (rhs << 19) | (src << 13) |
                                     (dst << 7) | 0x57)

    def test_splat_preserves_unsigned_bf16_bits(self) -> None:
        for bits in (0, 0x3E83, 0x8000, 0xFFFF):
            with self.subTest(bits=bits):
                instructions = self.compile(with_operation(
                    f'"atlas.virtual_vpu_splat"() {{bits = {bits} : i32}} : () -> {BF16}'))
                splat, = [op for op in instructions if op["operation"] == "atlas.vli"]
                fields = splat["fields"]
                self.assertEqual(fields["mode"], "all")
                self.assertEqual(fields["immediate"], bits)
                self.assertEqual(fields["dst"] % 2, 0)
                self.assertEqual(splat["word_u32"], (bits << 16) | (fields["dst"] << 7) | 0x5F)

    def test_invalid_reductions_and_splat_bits_are_rejected(self) -> None:
        cases = [
            (f'"atlas.virtual_vpu_reduce"(%t0) {{kind = "{kind}"}} : ({BF16}) -> {BF16}',
             "unknown virtual reduce VPU kind")
            for kind in ("col_sum", "col_max", "col_min", "sum", "unknown")
        ]
        cases += [
            (f'"atlas.virtual_vpu_splat"() {{bits = {bits} : i32}} : () -> {BF16}',
             "bits (raw BF16)")
            for bits in (-1, 65536)
        ]
        cases.append((f'"atlas.virtual_xlu_transpose"(%t0) : ({BF16}) -> {BF16}',
                      "FP8"))
        for operation, diagnostic in cases:
            with self.subTest(operation=operation):
                result = run("atlas-opt", with_operation(operation), "--verify-atlas-virtual-stream")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)

    def test_new_operation_contracts_reject_changed_machine_fields(self) -> None:
        machine = lower((EXAMPLES / "virtual_attention_tile_transposed_key.mlir").read_text())
        mutations = [
            ("atlas.vpu_reduce", lambda line: line.replace('kind = "row_max"', 'kind = "row_min"'),
             "tensor reduce fields or origins differ"),
            ("atlas.vli", lambda line: line.replace("immediate = 16003", "immediate = 16004"),
             "tensor splat fields differ"),
            ("atlas.xlu_transpose", lambda line: re.sub(r"src = \d+ : i32", "src = 31 : i32", line),
             "tensor transpose fields or origins differ"),
        ]
        for name, mutate, diagnostic in mutations:
            with self.subTest(operation=name):
                changed = rewrite_line(machine, lambda line: f'"{name}"' in line, mutate)
                self.assertNotEqual(changed, machine)
                assert_boundaries(self, changed, rejects=diagnostic)


if __name__ == "__main__":
    unittest.main()

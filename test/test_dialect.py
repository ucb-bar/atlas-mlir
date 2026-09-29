"""Executable OOT dialect checks. Set ATLAS_RTL_ROOT for source BitPat crosswalk."""

from __future__ import annotations

import ast
import os
import pathlib
import re
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPT = ROOT / "build/bin/atlas-opt"
EMIT = ROOT / "build/bin/atlas-emit"


def attrs(**fields: object) -> str:
    encoded = []
    for key, value in fields.items():
        if isinstance(value, bool):
            body = "true" if value else "false"
        elif isinstance(value, int):
            body = f"{value} : i32"
        else:
            body = f'"{value}"'
        encoded.append(f"{key} = {body}")
    return ", ".join(encoded)


def program(cases: list[tuple[str, dict[str, object], str]]) -> str:
    lines = ['module {', '  %s0 = "atlas.start"() : () -> !atlas.state']
    for i, (op, fields, _) in enumerate(cases, 1):
        lines.append(
            f'  %s{i} = "atlas.{op}"(%s{i-1}) '
            f'{{{attrs(**fields)}}} : (!atlas.state) -> !atlas.state'
        )
    return "\n".join(lines + ["}", ""])


def run(tool: pathlib.Path, source: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(tool), *args, "-"], input=source, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def variants() -> list[tuple[str, dict[str, object], str]]:
    cases: list[tuple[str, dict[str, object], str]] = [
        ("vload", dict(dst=4, base=3, offset=-2, format="raw"), "VLOAD"),
        ("vstore", dict(src=8, base=5, offset=1, format="raw"), "VSTORE"),
        ("dma", dict(direction="load", channel=7, reg=4, dram=5, size=6), "DMA_LOAD_ANY"),
        ("dma", dict(direction="store", channel=1, reg=4, dram=5, size=6), "DMA_STORE_ANY"),
        ("dma_config", dict(channel=2, base_reg=3), "DMA_CONFIG_ANY"),
        ("dma_wait", dict(channel=2), "DMA_WAIT_ANY"),
    ]
    for unit in (0, 1):
        cases.extend([
            ("mxu_push", dict(kind="weight_fp8", unit=unit, src=4, slot=1), f"VMATPUSH_W_MXU{unit}"),
            ("mxu_push", dict(kind="acc_fp8", unit=unit, src=4, slot=1), f"VMATPUSH_AFP8_MXU{unit}"),
            ("mxu_push", dict(kind="acc_bf16", unit=unit, src=4, slot=1), f"VMATPUSH_ABF16_MXU{unit}"),
            ("mxu_pop", dict(format="fp8", unit=unit, dst=4, slot=1, scale_reg=3), f"VMATPOP_FP8_MXU{unit}"),
            ("mxu_pop", dict(format="bf16", unit=unit, dst=4, slot=1, scale_reg=0), f"VMATPOP_BF16_MXU{unit}"),
            ("mxu_matmul", dict(unit=unit, src=4, weight_slot=1, acc_slot=1, accumulate=False), f"VMATMUL_MXU{unit}"),
            ("mxu_matmul", dict(unit=unit, src=4, weight_slot=1, acc_slot=1, accumulate=True), f"VMATMUL_ACC_MXU{unit}"),
        ])
    for kind, pattern in [
        ("add", "VADD_BF16"), ("sub", "VSUB_BF16"), ("mul", "VMUL_BF16"),
        ("min", "VMIN_BF16"), ("max", "VMAX_BF16"),
    ]:
        cases.append(("vpu_binary", dict(kind=kind, dst=4, lhs=6, rhs=8), pattern))
    for kind, pattern in [
        ("col_sum", "VREDSUM_BF16"), ("col_min", "VREDMIN_BF16"),
        ("col_max", "VREDMAX_BF16"), ("row_sum", "VREDSUM_ROW_BF16"),
        ("row_min", "VREDMIN_ROW_BF16"), ("row_max", "VREDMAX_ROW_BF16"),
    ]:
        cases.append(("vpu_reduce", dict(kind=kind, dst=4, src=6), pattern))
    for kind, pattern in [
        ("mov", "VMOV"), ("recip", "VRECIP_BF16"), ("exp", "VEXP"),
        ("exp2", "VEXP2"), ("square", "VSQUARE_BF16"),
        ("cube", "VCUBE_BF16"), ("relu", "VRELU"), ("sin", "VSIN"),
        ("cos", "VCOS"), ("tanh", "VTANH"), ("log2", "VLOG2"),
        ("sqrt", "VSQRT"),
    ]:
        cases.append(("vpu_unary", dict(kind=kind, dst=4, src=6), pattern))
    cases.extend([
        ("vpu_pack", dict(direction="bf16_to_fp8", dst=4, src=6, scale_reg=3), "VFP8PACK"),
        ("vpu_pack", dict(direction="fp8_to_bf16", dst=4, src=6, scale_reg=3), "VFP8UNPACK"),
    ])
    for mode in ("all", "row", "col", "one"):
        cases.append(("vli", dict(mode=mode, dst=4, immediate=0x3f80), f"VLI_{mode.upper()}"))
    cases.append(("xlu_transpose", dict(dst=4, src=6), "VTRPOSE_XLU"))
    return cases


class AtlasDialectTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(OPT.is_file(), "build atlas-opt first")
        self.assertTrue(EMIT.is_file(), "build atlas-emit first")

    def test_all_fifty_selected_custom_patterns(self) -> None:
        cases = variants()
        self.assertEqual(len(cases), 50)
        source = program(cases)
        parsed = run(OPT, source)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        self.assertEqual(run(OPT, parsed.stdout).returncode, 0)
        emitted = run(EMIT, source)
        self.assertEqual(emitted.returncode, 0, emitted.stderr)
        words = [int(line, 16) for line in emitted.stdout.splitlines()]
        self.assertEqual(len(words), len(cases))
        self.assertEqual(words[0], 0xFFE18207)  # VLS: signed offset, base, destination.
        self.assertEqual(words[6], (1 << 7) | (4 << 13) | 0x77)
        self.assertEqual(words[-1], (6 << 13) | (4 << 7) | 0x6B)
        by_pattern = {pattern: word for word, (_, _, pattern) in zip(words, cases, strict=True)}
        self.assertEqual(by_pattern["VADD_BF16"], (8 << 19) | (6 << 13) | (4 << 7) | 0x57)
        self.assertEqual(by_pattern["VLI_ALL"], (0x3F80 << 16) | (4 << 7) | 0x5F)
        self.assertEqual(by_pattern["DMA_CONFIG_ANY"], (3 << 15) | (2 << 12) | 0x7F)

        rtl_root = os.environ.get("ATLAS_RTL_ROOT")
        if not rtl_root:
            return
        instructions = pathlib.Path(rtl_root) / "src/main/scala/atlas/scalar/Instructions.scala"
        source_patterns = {
            name: bits.replace("_", "") for name, bits in
            re.findall(r'def\s+(\w+)\s*=\s*BitPat\("b([01?_]+)"\)', instructions.read_text())
        }
        self.assertEqual(len(source_patterns), 99)
        for word, (_, _, pattern_name) in zip(words, cases, strict=True):
            bits = source_patterns[pattern_name]
            self.assertEqual(len(bits), 32)
            for position, symbol in enumerate(bits):
                if symbol != "?":
                    self.assertEqual((word >> (31 - position)) & 1, int(symbol), pattern_name)

    def test_negative_verifiers(self) -> None:
        base = variants()
        failures = [
            ("vload", dict(dst=64, base=0, offset=0, format="raw"), "dst"),
            ("vload", dict(dst=0, base=32, offset=0, format="raw"), "base"),
            ("vload", dict(dst=0, base=0, offset=2048, format="raw"), "offset"),
            ("vstore", dict(src=0, base=0, offset=0, format="bf16"), "format"),
            ("dma", dict(direction="load", channel=8, reg=0, dram=0, size=1), "channel"),
            ("dma_config", dict(channel=0, base_reg=32), "base_reg"),
            ("dma_wait", dict(channel=-1), "channel"),
            ("mxu_push", dict(kind="acc_bf16", unit=0, src=63, slot=0), "src"),
            ("mxu_push", dict(kind="acc_bf16", unit=0, src=3, slot=0), "even"),
            ("mxu_matmul", dict(unit=2, src=4, weight_slot=0, acc_slot=0, accumulate=False), "unit"),
            ("mxu_pop", dict(format="bf16", unit=0, dst=63, slot=0, scale_reg=0), "dst"),
            ("vpu_binary", dict(kind="add", dst=5, lhs=6, rhs=8), "even"),
            ("vpu_unary", dict(kind="cube", dst=4, src=63), "src"),
            ("vpu_pack", dict(direction="fp8_to_bf16", dst=5, src=6, scale_reg=0), "even"),
            ("vpu_reduce", dict(kind="row_sum", dst=4, src=7), "even"),
            ("vli", dict(mode="all", dst=4, immediate=65536), "immediate"),
            ("xlu_transpose", dict(dst=64, src=0), "dst"),
        ]
        self.assertEqual(len(failures), 17)
        for op, fields, expected in failures:
            with self.subTest(op=op, fields=fields):
                source = program([(op, fields, "")])
                result = run(OPT, source)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
        self.assertEqual(len(base), 50)

    def test_emitter_rejects_non_linear_state(self) -> None:
        source = '''module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.dma_wait"(%s0) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.dma_wait"(%s0) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
}'''
        result = run(EMIT, source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("linear Atlas state chain", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_pinned_model_encoding_differences_are_visible(self) -> None:
        rtl_root = os.environ.get("ATLAS_RTL_ROOT")
        model_root = os.environ.get("ATLAS_MODEL_ROOT")
        if not (rtl_root and model_root):
            self.skipTest("set ATLAS_RTL_ROOT and ATLAS_MODEL_ROOT for source comparison")
        rtl_text = (
            pathlib.Path(rtl_root) / "src/main/scala/atlas/scalar/Instructions.scala"
        ).read_text()
        model_text = (
            pathlib.Path(model_root) / "npu_model/configs/isa_definition.py"
        ).read_text()
        rtl = {
            name: int(bits.replace("_", "")[:7], 2)
            for name, bits in re.findall(
                r'def\s+(\w+)\s*=\s*BitPat\("b([01?_]+)"\)', rtl_text
            ) if "?" not in bits.replace("_", "")[:7]
        }
        model = {
            node.name: {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in node.keywords if keyword.arg in ("funct3", "funct7")
            }
            for node in ast.parse(model_text).body if isinstance(node, ast.ClassDef)
        }
        self.assertEqual(model["VSQUARE_BF16"]["funct7"], 0x4E)
        self.assertEqual(rtl["VSQUARE_BF16"], 0x46)
        self.assertEqual(model["VCUBE_BF16"]["funct7"], 0x4F)
        self.assertEqual(rtl["VCUBE_BF16"], 0x47)
        self.assertEqual(model["DMA_CONFIG_CH0"]["funct7"], 1)
        self.assertEqual(rtl["DMA_CONFIG_ANY"], 0)
        self.assertEqual(model["CSRRCI"]["funct3"], 4)
        self.assertIn("111?????1110011", rtl_text)
        isa_type_text = (pathlib.Path(model_root) / "npu_model/isa.py").read_text()
        self.assertIn("_mask(vs2, 5)", isa_type_text)
        self.assertIn("_mask(vs1, 7)", isa_type_text)


if __name__ == "__main__":
    unittest.main()

"""Executable OOT dialect checks. Set ATLAS_RTL_ROOT for source BitPat crosswalk."""

from __future__ import annotations

import ast
import os
import pathlib
import random
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
OPT = BIN / "atlas-opt"
EMIT = BIN / "atlas-emit"


def llvm_tool(name: str) -> str | None:
    configured = os.environ.get("ATLAS_LLVM_BIN")
    if configured:
        path = pathlib.Path(configured) / name
        return str(path) if path.is_file() else None
    return shutil.which(name)


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


def scalar_variants() -> list[tuple[str, dict[str, object], str]]:
    cases: list[tuple[str, dict[str, object], str]] = []
    for kind in ("add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and"):
        cases.append(("alu_reg", dict(kind=kind, dst=3, lhs=1, rhs=2), kind.upper()))
    for kind in ("addi", "slti", "sltiu", "xori", "ori", "andi", "slli", "srli", "srai"):
        immediate = 3 if kind in ("slli", "srli", "srai") else -2
        cases.append(("alu_imm", dict(kind=kind, dst=3, src=1, immediate=immediate), kind.upper()))
    scalar_loads = []
    for kind in ("lb", "lh", "lw", "lbu", "lhu", "seld", "seli"):
        offset = 127 if kind == "seli" else -4
        base = 0 if kind == "seli" else 2
        scalar_loads.append(("scalar_load", dict(kind=kind, dst=1, base=base, offset=offset), kind.upper()))
    for index, kind in enumerate(("beq", "bne", "blt", "bge", "bltu", "bgeu")):
        cases.append(("branch", dict(kind=kind, lhs=1, rhs=2, offset_bytes=8), kind.upper()))
        cases.append(scalar_loads[index])  # selected RTL executes one delay slot.
    cases.extend([
        ("jump", dict(kind="jal", dst=1, base=0, offset=8), "JAL"),
        ("upper", dict(kind="lui", dst=2, immediate=0x12345), "LUI"),
        ("jump", dict(kind="jalr", dst=1, base=2, offset=-2), "JALR"),
        ("upper", dict(kind="auipc", dst=2, immediate=0x12345), "AUIPC"),
        ("delay", dict(cycles=3), "DELAY"),
    ])
    for kind, pattern in (
        ("rrw", "CSRRW"), ("rrs", "CSRRS"), ("rrc", "CSRRC"),
        ("rrwi", "CSRRWI"), ("rrsi", "CSRRSI"), ("rrci", "CSRRCI"),
    ):
        cases.append(("csr", dict(kind=kind, dst=1, source=2, address=0xC10), pattern))
    cases.extend([
        ("trap", dict(kind="ecall"), "ECALL"),
        ("trap", dict(kind="ebreak"), "EBREAK"),
        ("fence", {}, "FENCE"),
    ])
    cases.append(scalar_loads[-1])
    for kind in ("sb", "sh", "sw"):
        cases.append(("scalar_store", dict(kind=kind, src=1, base=2, offset=-4), kind.upper()))
    return cases


def randomized_case(
    case: tuple[str, dict[str, object], str], rng: random.Random
) -> tuple[str, dict[str, object], str]:
    op, original, pattern = case
    fields = dict(original)
    for name in ("dst", "src", "lhs", "rhs", "base", "reg", "dram", "size", "base_reg", "source"):
        if name not in fields:
            continue
        if (op == "jump" and fields.get("kind") == "jal" and name == "base") or (
            op == "scalar_load" and fields.get("kind") == "seli" and name == "base"
        ):
            continue
        physical = op in ("vload", "vstore", "mxu_push", "mxu_matmul", "mxu_pop",
                          "vpu_binary", "vpu_unary", "vpu_pack", "vpu_reduce", "vli",
                          "xlu_transpose") and name in ("dst", "src", "lhs", "rhs")
        pair = (op == "vpu_binary" or op == "vpu_unary" or op == "vpu_reduce" or
                op == "mxu_push" and fields.get("kind") == "acc_bf16" and name == "src" or
                op == "mxu_pop" and fields.get("format") == "bf16" and name == "dst" or
                op == "vpu_pack" and
                ((fields["direction"] == "bf16_to_fp8" and name == "src") or
                 (fields["direction"] == "fp8_to_bf16" and name == "dst")) or
                op == "vli" and fields.get("mode") in ("all", "row") and name == "dst")
        if physical and pair:
            fields[name] = 2 * rng.randrange(32)
        elif physical:
            fields[name] = rng.randrange(64)
        else:
            fields[name] = rng.randrange(32)
    if "channel" in fields:
        fields["channel"] = rng.randrange(8)
    for slot in ("slot", "weight_slot", "acc_slot"):
        if slot in fields:
            fields[slot] = rng.randrange(2)
    if "scale_reg" in fields and not (op == "mxu_pop" and fields["format"] == "bf16"):
        fields["scale_reg"] = rng.randrange(32)
    if op in ("vload", "vstore", "scalar_store"):
        fields["offset"] = rng.randrange(-2048, 2048)
    elif op == "scalar_load":
        fields["offset"] = rng.randrange(256) if fields["kind"] == "seli" else rng.randrange(-2048, 2048)
    elif op == "alu_imm":
        fields["immediate"] = rng.randrange(32) if fields["kind"] in ("slli", "srli", "srai") else rng.randrange(-2048, 2048)
    elif op == "branch":
        fields["offset_bytes"] = 2 * rng.randrange(-2048, 2048)
    elif op == "jump":
        fields["offset"] = (2 * rng.randrange(-524288, 524288) if fields["kind"] == "jal"
                            else rng.randrange(-2048, 2048))
    elif op == "delay":
        fields["cycles"] = rng.randrange(4096)
    elif op == "upper":
        fields["immediate"] = rng.randrange(1 << 20)
    elif op == "vli":
        fields["immediate"] = rng.randrange(1 << 16)
    elif op == "csr":
        fields["address"] = rng.choice((0xC00, 0xC01, 0xC10, 0xC11))
    return op, fields, pattern


def check_rtl_bitpats(words: list[int], cases: list[tuple[str, dict[str, object], str]]) -> None:
    rtl_root = os.environ.get("ATLAS_RTL_ROOT")
    if not rtl_root:
        return
    instructions = pathlib.Path(rtl_root) / "src/main/scala/atlas/scalar/Instructions.scala"
    source_patterns = {
        name: bits.replace("_", "") for name, bits in
        re.findall(r'def\s+(\w+)\s*=\s*BitPat\("b([01?_]+)"\)', instructions.read_text())
    }
    assert len(source_patterns) == 99
    for word, (_, _, pattern_name) in zip(words, cases, strict=True):
        bits = source_patterns[pattern_name]
        assert len(bits) == 32
        for position, symbol in enumerate(bits):
            if symbol != "?":
                assert ((word >> (31 - position)) & 1) == int(symbol), pattern_name


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

        check_rtl_bitpats(words, cases)

    def test_all_forty_nine_scalar_patterns(self) -> None:
        cases = scalar_variants()
        self.assertEqual(len(cases), 49)
        source = program(cases)
        parsed = run(OPT, source)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        self.assertEqual(run(OPT, parsed.stdout).returncode, 0)
        emitted = run(EMIT, source)
        self.assertEqual(emitted.returncode, 0, emitted.stderr)
        words = [int(line, 16) for line in emitted.stdout.splitlines()]
        self.assertEqual(len(words), 49)
        by_pattern = {pattern: word for word, (_, _, pattern) in zip(words, cases, strict=True)}
        self.assertEqual(by_pattern["BEQ"], 0x00208463)
        self.assertEqual(by_pattern["JAL"], 0x008000EF)
        self.assertEqual(by_pattern["LUI"], 0x12345137)
        self.assertEqual(by_pattern["CSRRCI"], 0xC10170F3)
        self.assertEqual(by_pattern["ECALL"], 0x00000073)
        self.assertEqual(by_pattern["EBREAK"], 0x00100073)
        check_rtl_bitpats(words, cases)

    def test_seeded_encoding_variants(self) -> None:
        rng = random.Random(0xA71A5)
        cases = [
            randomized_case(case, rng)
            for _ in range(12) for case in variants() + scalar_variants()
        ]
        self.assertEqual(len(cases), 1188)
        emitted = run(EMIT, program(cases))
        self.assertEqual(emitted.returncode, 0, emitted.stderr)
        words = [int(line, 16) for line in emitted.stdout.splitlines()]
        self.assertEqual(len(words), 1188)
        check_rtl_bitpats(words, cases)
        unique = {(pattern, word) for word, (_, _, pattern) in zip(words, cases, strict=True)}
        self.assertGreaterEqual(len(unique), 1000)

    def test_selected_pattern_census_is_exact(self) -> None:
        modeled = [name for _, _, name in variants() + scalar_variants()]
        self.assertEqual(len(modeled), 99)
        self.assertEqual(len(set(modeled)), 99)
        rtl_root = os.environ.get("ATLAS_RTL_ROOT")
        if not rtl_root:
            return
        instructions = pathlib.Path(rtl_root) / "src/main/scala/atlas/scalar/Instructions.scala"
        actual = set(re.findall(r'def\s+(\w+)\s*=\s*BitPat', instructions.read_text()))
        self.assertEqual(set(modeled), actual)

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

    def test_scalar_negative_verifiers(self) -> None:
        failures = [
            ("alu_reg", dict(kind="add", dst=32, lhs=1, rhs=2), "dst"),
            ("alu_imm", dict(kind="addi", dst=1, src=2, immediate=2048), "signed immediate"),
            ("alu_imm", dict(kind="srai", dst=1, src=2, immediate=32), "shift amount"),
            ("branch", dict(kind="beq", lhs=1, rhs=2, offset_bytes=3), "even"),
            ("branch", dict(kind="bne", lhs=1, rhs=2, offset_bytes=4096), "branch byte offset"),
            ("jump", dict(kind="jal", dst=1, base=2, offset=8), "base"),
            ("jump", dict(kind="jalr", dst=1, base=2, offset=2048), "jalr word offset"),
            ("delay", dict(cycles=4096), "cycles"),
            ("upper", dict(kind="lui", dst=1, immediate=1048576), "raw 20-bit"),
            ("csr", dict(kind="rrci", dst=1, source=2, address=4096), "CSR address"),
            ("csr", dict(kind="rrw", dst=1, source=2, address=0xC02), "read-only"),
            ("csr", dict(kind="rrs", dst=1, source=1, address=0xC03), "read-only"),
            ("trap", dict(kind="halt"), "ecall or ebreak"),
            ("scalar_load", dict(kind="seli", dst=1, base=1, offset=127), "base"),
            ("scalar_load", dict(kind="seli", dst=1, base=0, offset=256), "E8M0"),
            ("scalar_store", dict(kind="sw", src=1, base=2, offset=-2049), "signed byte offset"),
        ]
        for op, fields, expected in failures:
            with self.subTest(op=op, fields=fields):
                result = run(OPT, program([(op, fields, "")]))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

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

    def test_branch_delay_slot_must_exist_and_cannot_redirect(self) -> None:
        branch = ("branch", dict(kind="beq", lhs=1, rhs=2, offset_bytes=8), "BEQ")
        jump = ("jump", dict(kind="jal", dst=1, base=0, offset=8), "JAL")
        missing = run(EMIT, program([branch]))
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("lacks its required delay-slot", missing.stderr)
        self.assertEqual(missing.stdout, "")
        nested = run(EMIT, program([branch, jump]))
        self.assertNotEqual(nested.returncode, 0)
        self.assertIn("branch or jump in selected RTL delay slot", nested.stderr)
        self.assertEqual(nested.stdout, "")

    def test_llvm_pass_preserves_one_ordered_encoded_block(self) -> None:
        source = (ROOT / "test/examples/mxu.mlir").read_text()
        emitted = run(EMIT, source)
        lowered = run(OPT, source, "--convert-atlas-to-llvm")
        self.assertEqual(emitted.returncode, 0, emitted.stderr)
        self.assertEqual(lowered.returncode, 0, lowered.stderr)
        self.assertIn("llvm.func @atlas_program", lowered.stdout)
        self.assertIn("llvm.inline_asm has_side_effects", lowered.stdout)
        self.assertIn('"~{memory}"', lowered.stdout)
        self.assertEqual(lowered.stdout.count("llvm.inline_asm"), 1)
        self.assertNotIn("atlas.", lowered.stdout)
        words = emitted.stdout.splitlines()
        self.assertEqual(len(words), 4)
        for word in words:
            self.assertIn(f".word 0x{word}", lowered.stdout)
        self.assertEqual(run(OPT, lowered.stdout).returncode, 0)

    def test_pre_lowering_stream_pass_checks_targets_without_rewriting(self) -> None:
        source = (ROOT / "test/examples/handoff_mlp_tile.mlir").read_text()
        checked = run(OPT, source, "--verify-atlas-machine-stream")
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn('"atlas.mxu_matmul"', checked.stdout)
        self.assertNotIn("llvm.inline_asm", checked.stdout)
        lowered = run(OPT, source, "--verify-atlas-machine-stream",
                      "--convert-atlas-to-llvm")
        self.assertEqual(lowered.returncode, 0, lowered.stderr)
        self.assertEqual(lowered.stdout.count("llvm.inline_asm"), 1)

        branch = ("branch", dict(kind="beq", lhs=1, rhs=2, offset_bytes=8), "")
        slot = ("alu_imm", dict(kind="addi", dst=1, src=1, immediate=1), "")
        escaping = run(OPT, program([branch, slot]),
                       "--verify-atlas-machine-stream")
        self.assertNotEqual(escaping.returncode, 0)
        self.assertIn("escapes the LLVM inline assembly block", escaping.stderr)
        missing_slot = run(OPT, program([branch]),
                           "--verify-atlas-machine-stream")
        self.assertNotEqual(missing_slot.returncode, 0)
        self.assertIn("lacks its required delay-slot", missing_slot.stderr)

    def test_pre_lowering_stream_keeps_external_operation_annotations(self) -> None:
        source = program([
            ("alu_imm", dict(kind="addi", dst=1, src=0, immediate=1), "")
        ])
        annotated = source.replace(
            "immediate = 1 : i32",
            'immediate = 1 : i32, atlas.test_annotation = "source-bound"',
        )
        checked = run(OPT, annotated, "--verify-atlas-machine-stream")
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn('atlas.test_annotation = "source-bound"', checked.stdout)
        self.assertEqual(run(EMIT, annotated).stdout, run(EMIT, source).stdout)

    def test_llvm_pass_to_riscv_object(self) -> None:
        needed = ("mlir-translate", "llc", "llvm-readelf", "llvm-objdump")
        tools = {name: llvm_tool(name) for name in needed}
        if not all(tools.values()):
            self.skipTest("ATLAS_LLVM_BIN lacks MLIR/LLVM object tools")
        source = (ROOT / "test/examples/mxu.mlir").read_text()
        lowered = run(OPT, source, "--convert-atlas-to-llvm")
        self.assertEqual(lowered.returncode, 0, lowered.stderr)
        translated = subprocess.run(
            [tools["mlir-translate"], "--mlir-to-llvmir"],
            input=lowered.stdout, text=True, capture_output=True,
        )
        self.assertEqual(translated.returncode, 0, translated.stderr)
        compiled = subprocess.run(
            [tools["llc"], "-mtriple=riscv32-unknown-elf", "-filetype=obj", "-o", "-"],
            input=translated.stdout.encode(), capture_output=True,
        )
        self.assertEqual(compiled.returncode, 0, compiled.stderr.decode())
        with tempfile.TemporaryDirectory() as tmp:
            obj = pathlib.Path(tmp) / "atlas.o"
            obj.write_bytes(compiled.stdout)
            info = subprocess.run([tools["llvm-readelf"], "-h", str(obj)],
                                  text=True, capture_output=True)
            dump = subprocess.run([tools["llvm-objdump"], "-d", str(obj)],
                                  text=True, capture_output=True)
            self.assertEqual(info.returncode, 0, info.stderr)
            self.assertEqual(dump.returncode, 0, dump.stderr)
            self.assertIn("ELF32", info.stdout)
            self.assertIn("RISC-V", info.stdout)
            for word in run(EMIT, source).stdout.splitlines():
                self.assertIn(f"0x{word}", dump.stdout)

    def test_llvm_pass_rejects_unresolved_control_or_state(self) -> None:
        branch = ("branch", dict(kind="beq", lhs=1, rhs=2, offset_bytes=8), "")
        slot = ("alu_imm", dict(kind="addi", dst=1, src=1, immediate=1), "")
        jalr = ("jump", dict(kind="jalr", dst=1, base=2, offset=0), "")
        for source, message in (
            (program([branch]), "delay-slot"),
            (program([branch, slot]), "escapes"),
            (program([jalr, slot]), "JALR"),
            (program([branch, jalr, slot]), "delay slot"),
        ):
            with self.subTest(message=message):
                result = run(OPT, source, "--convert-atlas-to-llvm")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

        source = '''module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.dma_wait"(%s0) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.dma_wait"(%s0) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
}'''
        result = run(OPT, source, "--convert-atlas-to-llvm")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("linear Atlas state chain", result.stderr)

        nested = '''module {
  %s0 = "atlas.start"() : () -> !atlas.state
  llvm.func @foreign() { llvm.return }
  %s1 = "atlas.dma_wait"(%s0) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
}'''
        result = run(OPT, nested, "--convert-atlas-to-llvm")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("flat, linear Atlas state chain", result.stderr)

    def test_llvm_pass_accepts_in_block_branch_target(self) -> None:
        cases = [
            ("branch", dict(kind="beq", lhs=1, rhs=2, offset_bytes=4), ""),
            ("alu_imm", dict(kind="addi", dst=1, src=1, immediate=1), ""),
            ("dma_wait", dict(channel=0), ""),
        ]
        result = run(OPT, program(cases), "--convert-atlas-to-llvm")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count(".word"), 3)

    def test_llvm_pass_covers_static_selected_pattern_subset(self) -> None:
        # Unconstrained JALR remains excluded; the proved-target fixture covers it.
        cases = variants() + [
            case for case in scalar_variants()
            if not (case[0] == "jump" and case[1]["kind"] == "jalr")
        ]
        self.assertEqual(len(cases), 98)
        result = run(OPT, program(cases), "--convert-atlas-to-llvm")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count(".word"), 98)

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

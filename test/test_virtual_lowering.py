"""Generated virtual SSA/CFG programs through physical Atlas and selected core."""

from __future__ import annotations

import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin")).resolve()
EXAMPLES = ROOT / "test/examples"


def run(tool: str, source: str, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BIN / tool), *options, "-"], input=source,
        text=True, capture_output=True, check=False,
    )


def lower(source: str, *, timed: bool = True) -> str:
    result = run("atlas-opt", source, "--lower-atlas-virtual-to-machine")
    if result.returncode:
        raise AssertionError(result.stderr)
    if timed:
        result = run("atlas-opt", result.stdout, "--insert-atlas-delays")
        if result.returncode:
            raise AssertionError(result.stderr)
    return result.stdout


def shorten_delay(machine: str, *, after: str | None = None, kind: str | None = None) -> str:
    lines = machine.splitlines()
    active = kind is None
    for index, line in enumerate(lines):
        if after and f'"{after}"' in line:
            active = kind is None or f'kind = "{kind}"' in line
        if not active or '"atlas.delay"' not in line or (after and kind is None and f"after {after}" not in line):
            continue
        cycles = re.search(r"cycles = (\d+) : i32", line)
        if cycles and int(cycles[1]) > 1:
            lines[index] = line.replace(cycles[0], "cycles = 1 : i32", 1)
            return "\n".join(lines) + "\n"
    raise AssertionError(f"expected an inserted delay greater than one cycle after {after}")


def emitted(machine: str) -> tuple[int, ...]:
    result = run("atlas-emit", machine)
    if result.returncode:
        raise AssertionError(result.stderr)
    return tuple(int(line, 16) for line in result.stdout.splitlines())


def object_words(machine: str) -> tuple[int, ...]:
    llvm_root = os.environ.get("ATLAS_LLVM_BIN")
    if not llvm_root:
        raise unittest.SkipTest("set ATLAS_LLVM_BIN for LLVM object lowering")
    llvm = Path(llvm_root)
    lowered = run("atlas-opt", machine, "--convert-atlas-to-llvm")
    if lowered.returncode:
        raise AssertionError(lowered.stderr)
    translated = subprocess.run(
        [str(llvm / "mlir-translate"), "--mlir-to-llvmir"],
        input=lowered.stdout, text=True, capture_output=True, check=True,
    )
    with tempfile.TemporaryDirectory() as temporary:
        obj = Path(temporary) / "atlas.o"
        section = Path(temporary) / "text.bin"
        subprocess.run(
            [str(llvm / "llc"), "-mtriple=riscv32-unknown-elf",
             "-filetype=obj", "-o", str(obj)],
            input=translated.stdout.encode(), capture_output=True, check=True,
        )
        subprocess.run(
            [str(llvm / "llvm-objcopy"), "--dump-section",
             f".text={section}", str(obj)], capture_output=True, check=True,
        )
        data = section.read_bytes()
    count = len(emitted(machine))
    if len(data) < count * 4:
        raise AssertionError("LLVM object truncated the Atlas word block")
    return tuple(int.from_bytes(data[i * 4:i * 4 + 4], "little")
                 for i in range(count))


def panel(phase: int) -> bytes:
    codes = (0x0000, 0x3F80, 0xBF80, 0x4000, 0xC000, 0x3E80, 0xBE80)
    values = [codes[(i * 7 + phase + i // 16) % len(codes)] for i in range(1024)]
    return b"".join(struct.pack("<H", code) for code in values)


def relu_bits(source: bytes) -> bytes:
    return b"".join(struct.pack("<H", 0 if code & 0x8000 else code)
                    for code in struct.unpack("<1024H", source))


def virtual_chain(length: int) -> str:
    body = [
        'module {',
        '  func.func @chain() -> !atlas.virtual_state attributes '
        '{atlas.input_dram_base = 2415919104 : i64, '
        'atlas.output_dram_base = 2415923200 : i64} {',
        '    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state',
        '    %io1, %t0 = "atlas.virtual_input_bf16"(%io0) '
        '{index = 0 : i32} : (!atlas.virtual_state) -> '
        '(!atlas.virtual_state, !atlas.virtual_bf16)',
    ]
    for index in range(length):
        body.append(
            f'    %t{index + 1} = "atlas.virtual_vpu_unary"(%t{index}) '
            '{kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'
        )
    body.extend([
        f'    %io2 = "atlas.virtual_output_bf16"(%io1, %t{length}) '
        '{index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) '
        '-> !atlas.virtual_state',
        '    return %io2 : !atlas.virtual_state',
        '  }',
        '}',
    ])
    return '\n'.join(body) + '\n'


def virtual_pressure(count: int) -> str:
    body = [
        'module {',
        '  func.func @pressure() -> !atlas.virtual_state attributes '
        '{atlas.input_dram_base = 2415919104 : i64, '
        'atlas.output_dram_base = 2416050176 : i64} {',
        '    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state',
    ]
    for index in range(count):
        body.append(
            f'    %io{index + 1}, %t{index} = '
            f'"atlas.virtual_input_bf16"(%io{index}) '
            f'{{index = {index} : i32}} : (!atlas.virtual_state) -> '
            '(!atlas.virtual_state, !atlas.virtual_bf16)'
        )
    for index in range(count):
        body.append(
            f'    %io{count + index + 1} = '
            f'"atlas.virtual_output_bf16"(%io{count + index}, %t{index}) '
            f'{{index = {index} : i32}} : '
            '(!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state'
        )
    body.extend([
        f'    return %io{2 * count} : !atlas.virtual_state',
        '  }',
        '}',
    ])
    return '\n'.join(body) + '\n'


class VirtualLoweringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue((BIN / "atlas-opt").is_file(), "build atlas-opt first")
        self.assertTrue((BIN / "atlas-emit").is_file(), "build atlas-emit first")

    def test_loop_branch_and_edge_cycle_lower_to_checked_machine_streams(self) -> None:
        for name in ("loop", "branch", "swap_loop"):
            with self.subTest(name=name):
                source = (EXAMPLES / f"virtual_bf16_{name}_program.mlir").read_text()
                untimed = lower(source, timed=False)
                self.assertIn('atlas.timing_state = "untimed"', untimed)
                self.assertNotIn('"atlas.delay"', untimed)
                machine = lower(source)
                self.assertIn('atlas.timing_state = "timed"', machine)
                for option in ("--verify-atlas-machine-stream",
                               "--verify-atlas-generated-schedule", "--verify-atlas-timing"):
                    checked = run("atlas-opt", machine, option)
                    self.assertEqual(checked.returncode, 0, checked.stderr)
                self.assertIn("atlas.generated_from_virtual", machine)
                self.assertNotIn('"atlas.virtual_', machine)
                self.assertIn('"atlas.dma_wait"', machine)
                self.assertIn('"atlas.branch"', machine)
                self.assertIn('"atlas.jump"', machine)
                self.assertTrue(emitted(machine))
                reparsed = run("atlas-opt", machine)
                self.assertEqual(reparsed.returncode, 0, reparsed.stderr)
                self.assertEqual(reparsed.stdout, machine)

    def test_llvm_object_contains_exact_generated_words(self) -> None:
        machine = lower((EXAMPLES / "virtual_bf16_loop_program.mlir").read_text())
        self.assertEqual(object_words(machine), emitted(machine))
        structured = run("atlas-opt", machine,
                         "--convert-atlas-to-llvm-calls")
        self.assertEqual(structured.returncode, 0, structured.stderr)
        finalized = run("atlas-opt", structured.stdout,
                        "--finalize-atlas-llvm-calls")
        self.assertEqual(finalized.returncode, 0, finalized.stderr)
        self.assertIn("llvm.inline_asm", finalized.stdout)

        lines = structured.stdout.splitlines()
        index = next(i for i, line in enumerate(lines)
                     if 'atlas.source_op = "atlas.delay"' in line
                     and int(re.search(r"cycles = (\d+) : i32", line)[1]) > 1)
        word = int(re.search(r"atlas.word = (-?\d+) : i32", lines[index])[1]) & 0xffffffff
        lines[index] = re.sub(r"cycles = \d+ : i32", "cycles = 1 : i32", lines[index], count=1)
        lines[index] = re.sub(r"atlas.word = -?\d+ : i32", f"atlas.word = {(word & 0xfffff) | (1 << 20)} : i32", lines[index], count=1)
        changed = "\n".join(lines) + "\n"
        self.assertNotEqual(changed, structured.stdout)
        rejected = run("atlas-opt", changed, "--finalize-atlas-llvm-calls")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("timing resource conflict: VLOAD path 0 busy", rejected.stderr)

    def test_scalar_control_argument_has_explicit_register_abi(self) -> None:
        source = (EXAMPLES / "virtual_bf16_dynamic_branch_program.mlir").read_text()
        machine = lower(source)
        self.assertIn("atlas.scalar_arg_regs = array<i32: 10>", machine)
        self.assertIn("atlas.control_dram_base = 2415927296 : i64", machine)
        self.assertIn("lhs = 10 : i32", machine)
        self.assertIn('"atlas.scalar_load"', machine)
        two_controls = source.replace(
            '(%choose: i1)', '(%choose: i1, %other: i1)', 1)
        two_machine = lower(two_controls)
        abi = re.search(r'atlas.scalar_arg_regs = array<i32: (\d+), (\d+)>',
                        two_machine)
        self.assertIsNotNone(abi)
        self.assertNotEqual(abi.group(1), abi.group(2))

    def test_liveness_reuses_pairs_and_rejects_real_pressure(self) -> None:
        machine = lower(virtual_chain(40))
        self.assertEqual(machine.count('"atlas.vpu_unary"'), 40)
        self.assertEqual(len(emitted(machine)) < 2048, True)
        # The source has 41 distinct SSA tiles. A unique-pair allocator would
        # fail; the actual dependency chain needs only two simultaneously.
        self.assertIn('dst = 0 : i32', machine)
        rejected = run('atlas-opt', virtual_pressure(32),
                       '--lower-atlas-virtual-to-machine')
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('interference exceeds 31 physical pairs', rejected.stderr)
        unqualified = virtual_chain(1).replace('kind = "relu"',
                                                'kind = "exp"')
        rejected = run('atlas-opt', unqualified,
                       '--lower-atlas-virtual-to-machine')
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('admission currently requires mov or relu',
                      rejected.stderr)

    def test_bad_abi_and_modified_schedule_fail(self) -> None:
        source = (EXAMPLES / "virtual_bf16_loop_program.mlir").read_text()
        for changed, expected in (
            (source.replace("atlas.input_dram_base = 2415919104 : i64, ", ""),
             "requires explicit"),
            (source.replace("atlas.output_dram_base = 2415923200 : i64",
                            "atlas.output_dram_base = 2415919104 : i64"),
             "overlap"),
            (source.replace("atlas.output_dram_base = 2415923200 : i64",
                            "atlas.output_dram_base = 2415923201 : i64"),
             "aligned"),
        ):
            with self.subTest(expected=expected):
                result = run("atlas-opt", changed,
                             "--lower-atlas-virtual-to-machine")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
        machine = lower(source)
        mutated = shorten_delay(machine)
        self.assertNotEqual(mutated, machine)
        result = run("atlas-emit", mutated)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("timing resource conflict: VLOAD path 0 busy", result.stderr)
        dynamic = (EXAMPLES / "virtual_bf16_dynamic_branch_program.mlir").read_text()
        missing_control = dynamic.replace(
            ", atlas.control_dram_base = 2415927296 : i64", "")
        rejected = run("atlas-opt", missing_control,
                       "--lower-atlas-virtual-to-machine")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("control_dram_base", rejected.stderr)

    def test_selected_core_executes_generated_cfg_and_preserves_guards(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT",
                "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, ModeLIR, and LLVM paths")
        model, state, modelir = (Path(os.environ[key]).resolve(strict=True)
                                 for key in keys[:3])
        old_cwd = Path.cwd()
        sys.path.insert(0, str(modelir))
        try:
            os.chdir(modelir)
            from mlc.backends import cosim_atlas

            original = cosim_atlas.CosimCore

            class SelectedCore(original):
                def peek(self, name: str) -> int:
                    return super().peek("scalar/halt_now" if name == "io_halted"
                                        else name)

                def poke(self, name: str, value: int) -> None:
                    if name in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size"):
                        if name in self._S:
                            raise AssertionError(f"unexpected live ARC input {name}")
                        return
                    super().poke(name, value)

            cosim_atlas.CosimCore = SelectedCore
            try:
                a, b = panel(0), panel(2)
                branch_source = (EXAMPLES / "virtual_bf16_branch_program.mlir").read_text()
                cases = (
                    ("chain40", virtual_chain(40), [(0x90000000, a)],
                     0x90001000, relu_bits(a), 64),
                    ("loop", (EXAMPLES / "virtual_bf16_loop_program.mlir").read_text(),
                     [(0x90000000, a)], 0x90001000, relu_bits(a), 64),
                    ("branch_false", branch_source, [(0x90000000, a)],
                     0x90001000, a, 64),
                    ("branch_true", branch_source.replace(
                        "arith.constant 0 : i1", "arith.constant 1 : i1"),
                     [(0x90000000, a)], 0x90001000, relu_bits(a), 64),
                    ("swap_loop", (EXAMPLES / "virtual_bf16_swap_loop_program.mlir").read_text(),
                     [(0x90000000, a), (0x90000800, b)], 0x90002000, b, 128),
                )
                for name, source, inputs, output_addr, expected, reads in cases:
                    with self.subTest(name=name):
                        words = object_words(lower(source))
                        guard_addr = output_addr + 2048
                        result = cosim_atlas.run_program(
                            model, state, words,
                            preload=[*inputs, (output_addr, b"\xA5" * 2048),
                                     (guard_addr, b"\x5A" * 64)],
                            max_cycles=100000,
                        )
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (reads, 64))
                        self.assertEqual(result.slave.captured(output_addr, 2048),
                                         expected)
                        for address, data in inputs:
                            self.assertEqual(result.slave.captured(address, len(data)),
                                             data)
                        self.assertEqual(result.slave.captured(guard_addr, 64),
                                         b"\x5A" * 64)
                dynamic = (EXAMPLES / "virtual_bf16_dynamic_branch_program.mlir").read_text()
                dynamic_words = object_words(lower(dynamic))
                for choice in (0, 1, 2):
                    with self.subTest(dynamic_choice=choice):
                        mailbox = struct.pack("<I", choice) + bytes(1020)
                        result = cosim_atlas.run_program(
                            model, state, dynamic_words,
                            preload=[(0x90000000, a),
                                     (0x90001000, b"\xA5" * 2048),
                                     (0x90001800, b"\x5A" * 64),
                                     (0x90002000, mailbox)],
                            max_cycles=100000,
                        )
                        expected = relu_bits(a) if choice & 1 else a
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (96, 64))
                        self.assertEqual(result.slave.captured(0x90001000, 2048),
                                         expected)
                        self.assertEqual(result.slave.captured(0x90000000, 2048), a)
                        self.assertEqual(result.slave.captured(0x90001800, 64),
                                         b"\x5A" * 64)
                        self.assertEqual(result.slave.captured(0x90002000, 1024),
                                         mailbox)
                # Two independent ordered outputs exercise different live
                # inputs, VMEM staging windows, and output DRAM tile offsets.
                two_output_words = object_words(lower(virtual_pressure(2)))
                result = cosim_atlas.run_program(
                    model, state, two_output_words,
                    preload=[(0x90000000, a), (0x90000800, b),
                             (0x90020000, b"\xA5" * 4096),
                             (0x90021000, b"\x5A" * 64)],
                    max_cycles=100000,
                )
                self.assertTrue(result.halted)
                self.assertEqual((result.reads, result.writes), (128, 128))
                self.assertEqual(result.slave.captured(0x90020000, 2048), a)
                self.assertEqual(result.slave.captured(0x90020800, 2048), b)
                self.assertEqual(result.slave.captured(0x90000000, 2048), a)
                self.assertEqual(result.slave.captured(0x90000800, 2048), b)
                self.assertEqual(result.slave.captured(0x90021000, 64),
                                 b"\x5A" * 64)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()

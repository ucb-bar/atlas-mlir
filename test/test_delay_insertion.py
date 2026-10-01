"""insert-atlas-delays: minimum delays from the ported npu_model timing model."""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
OPT = BIN / "atlas-opt"
EMIT = BIN / "atlas-emit"
OP = re.compile(r'"atlas\.([a-z_]+)"\(%\w+\)\s*<\{(.*?)\}>(?:\s*\{atlas\.reason = "([^"]*)"\})?')
LINE = re.compile(r'(%\w+) = "atlas\.(\w+)"\((%\w+)\)')


def run(tool: pathlib.Path, source: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(tool), *args, "-"], input=source, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def program(ops: list[tuple[str, str]]) -> str:
    lines = ['module {', '  %s0 = "atlas.start"() : () -> !atlas.state']
    for i, (op, fields) in enumerate(ops, 1):
        lines.append(f'  %s{i} = "atlas.{op}"(%s{i-1}) {{{fields}}} '
                     ': (!atlas.state) -> !atlas.state')
    return "\n".join(lines + ["}", ""])


def without_delays(source: str) -> str:
    lines = source.splitlines()
    ops = [(i, LINE.search(line)) for i, line in enumerate(lines)]
    ops = [(i, m) for i, m in ops if m and m.group(2) != "start"]
    new_index, kept = [], 0
    for _, m in ops:
        new_index.append(kept)
        kept += m.group(2) != "delay"
    new_index.append(kept)
    alias: dict[str, str] = {}
    for old, (i, m) in enumerate(ops):
        result, name, state = m.groups()
        state = alias.get(state, state)
        if name == "delay":
            alias[result] = state
            lines[i] = ""
            continue
        line = lines[i].replace(f"({m.group(3)})", f"({state})", 1)
        offset = re.search(r"offset_bytes = (-?\d+)", line)
        if offset:
            target = new_index[old + int(offset.group(1)) // 2]
            line = line.replace(offset.group(0),
                                f"offset_bytes = {2 * (target - new_index[old])}")
        lines[i] = line
    return "\n".join(line for line in lines if line) + "\n"


def stream(printed: str) -> list[tuple[str, str, str]]:
    return [m.groups(default="") for m in OP.finditer(printed)]


def nop() -> tuple[str, str]:
    return ("alu_imm", 'kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32')


def addi(dst: int, src: int, immediate: int) -> tuple[str, str]:
    return ("alu_imm", f'kind = "addi", dst = {dst} : i32, src = {src} : i32, '
                       f'immediate = {immediate} : i32')


def cycles(fields: str) -> int:
    return int(re.search(r"cycles = (\d+)", fields).group(1))


class DelayInsertionTest(unittest.TestCase):
    def test_handoff_examples(self) -> None:
        for name in ("mlp", "attention"):
            with self.subTest(example=name):
                authored = (ROOT / f"test/examples/handoff_{name}_tile.mlir").read_text()
                rejected = run(OPT, authored, "--insert-atlas-delays")
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("'atlas.delay' op is not allowed in the input", rejected.stderr)

                result = run(OPT, without_delays(authored), "--insert-atlas-delays")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr.count("warning:"), 1, result.stderr)
                self.assertIn("reuses DMA channel 0", result.stderr)

                ops = stream(result.stdout)
                delays = [(cycles(f), reason) for op, f, reason in ops if op == "delay"]
                self.assertTrue(all(reason for _, reason in delays))
                self.assertIn((62, "RAW on mxu0.acc0 after atlas.mxu_matmul"), delays)
                self.assertEqual(sum(1 for d in delays if d == (2, "RAW on x16 after atlas.scalar_load")), 4)
                self.assertIn((31, "RAW on VMEM 0xc00 after atlas.vstore"), delays)

                self.assertNotEqual(run(OPT, result.stdout, "--insert-atlas-delays").returncode, 0)
                checked = run(OPT, result.stdout, "--verify-atlas-machine-stream")
                self.assertEqual(checked.returncode, 0, checked.stderr)
                self.assertEqual(run(EMIT, result.stdout).returncode, 0)

    def test_branches_keep_their_targets(self) -> None:
        loop = [
            addi(10, 0, 0), addi(12, 0, 64), addi(13, 0, 0), addi(14, 0, 4),
            ("scalar_load", 'kind = "lw", dst = 16 : i32, base = 10 : i32, offset = 0 : i32'),
            ("scalar_store", 'kind = "sw", src = 16 : i32, base = 12 : i32, offset = 0 : i32'),
            addi(13, 13, 1),
            ("branch", 'kind = "blt", lhs = 13 : i32, rhs = 14 : i32, offset_bytes = -6 : i32'),
            nop(),
            ("trap", 'kind = "ecall"'),
        ]
        result = run(OPT, program(loop), "--insert-atlas-delays")
        self.assertEqual(result.returncode, 0, result.stderr)
        ops = stream(result.stdout)
        self.assertEqual([op for op, _, _ in ops[4:8]],
                         ["scalar_load", "delay", "scalar_store", "alu_imm"])
        self.assertEqual(cycles(ops[5][1]), 2)
        branch = next(i for i, (op, _, _) in enumerate(ops) if op == "branch")
        offset = int(re.search(r"offset_bytes = (-?\d+)", ops[branch][1]).group(1))
        self.assertEqual(offset, -8)
        self.assertEqual(ops[branch + offset // 2][0], "scalar_load")

    def test_halt_stall_ends_on_a_nop(self) -> None:
        vload = ("vload", 'dst = 0 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"')
        for ops, guard in (
            ([addi(6, 0, 0), vload, ("trap", 'kind = "ecall"')], "a halt does not wait for a delay"),
            ([addi(6, 0, 0), vload, nop(), ("trap", 'kind = "ecall"')], ""),
        ):
            with self.subTest(guard=guard):
                result = run(OPT, program(ops), "--insert-atlas-delays")
                self.assertEqual(result.returncode, 0, result.stderr)
                printed = stream(result.stdout)
                self.assertEqual([op for op, _, _ in printed],
                                 ["alu_imm", "vload", "delay", "alu_imm", "trap"])
                self.assertEqual(cycles(printed[2][1]), 31)
                self.assertEqual(printed[3][2], guard)

    def test_rejects_what_a_delay_cannot_fix(self) -> None:
        dma_then_vload = [
            addi(5, 0, 0), ("dma_config", "channel = 0 : i32, base_reg = 5 : i32"),
            ("dma_wait", "channel = 0 : i32"), addi(6, 0, 0), addi(2, 0, 1024),
            ("upper", 'kind = "lui", dst = 1 : i32, immediate = 589824 : i32'),
            ("dma", 'direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32'),
            ("vload", 'dst = 0 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"'),
        ]
        slow_slot = [
            ("branch", 'kind = "beq", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = 4 : i32'),
            ("vload", 'dst = 0 : i32, base = 0 : i32, offset = 0 : i32, format = "raw"'),
        ]
        for ops, message in (
            ([addi(1, 0, 1), ("delay", "cycles = 3 : i32"), ("trap", 'kind = "ecall"')],
             "is not allowed in the input"),
            (dma_then_vload, "a delay cannot cover a DMA transfer"),
            (slow_slot, "delay-slot instruction must be single-cycle"),
            ([("jump", 'kind = "jalr", dst = 0 : i32, base = 1 : i32, offset = 0 : i32'), nop()],
             "register target"),
            ([("upper", 'kind = "auipc", dst = 1 : i32, immediate = 0 : i32')], "own instruction index"),
        ):
            with self.subTest(message=message):
                result = run(OPT, program(ops), "--insert-atlas-delays")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()

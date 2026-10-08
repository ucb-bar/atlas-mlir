"""Generated DMA schedules may reuse scalar operands captured at launch.

ScalarCore.scala:489-509 forms the command from S1 register values, and
diplomatic/memory/DMA.scala:223-229 queues the complete command. This checks
compiler admission, not physical completion or concurrent VMEM range safety.
"""

from __future__ import annotations

from itertools import product
import unittest

from test_virtual_ssa import BIN, run


STATE = "!atlas.state"
MARKER = "atlas.virtual_dma_transfer"
NOP = ("alu_imm", 'kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32')
DELAY = ("delay", 'cycles = 256 : i32, atlas.delay_reason = "tensor_completion"')


def dma(direction: str = "load", channel: int = 0, identity: int | None = 0) -> tuple[str, str]:
    marker = f", {MARKER} = {identity} : i32" if identity is not None else ""
    return "dma", f'direction = "{direction}", channel = {channel} : i32, reg = 4 : i32, dram = 7 : i32, size = 9 : i32{marker}'


def wait(channel: int = 0, identity: int | None = 0) -> tuple[str, str]:
    marker = f", {MARKER} = {identity} : i32" if identity is not None else ""
    return "dma_wait", f"channel = {channel} : i32{marker}"


def artifact(work: tuple[tuple[str, str], ...] = (), *, direction: str = "load", completion: tuple[str, str] | None = None, identity: int | None = 0) -> str:
    operations = (("upper", 'kind = "lui", dst = 4 : i32, immediate = 32 : i32'),
                  ("upper", 'kind = "lui", dst = 7 : i32, immediate = 589824 : i32'),
                  ("alu_imm", 'kind = "addi", dst = 9 : i32, src = 0 : i32, immediate = 1024 : i32'),
                  dma(direction, identity=identity), *work, completion if completion is not None else wait(identity=identity),
                  ("trap", 'kind = "ecall"'))
    lines = [f'%s0 = "atlas.start"() : () -> {STATE}']
    lines.extend(f'%s{index + 1} = "atlas.{name}"(%s{index}) {{{fields}}} : ({STATE}) -> {STATE}' for index, (name, fields) in enumerate(operations))
    return "module attributes {atlas.generated_from_virtual} {\n" + "\n".join(lines) + "\n}"


class DMACaptureVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue((BIN / "atlas-opt").is_file(), "build atlas-opt first")
        self.assertTrue((BIN / "atlas-emit").is_file(), "build atlas-emit first")

    def accepted(self, source: str) -> None:
        for tool, options in (("atlas-opt", ("--verify-atlas-generated-schedule",)), ("atlas-emit", ()), ("atlas-opt", ("--convert-atlas-to-llvm",))):
            result = run(tool, source, *options)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout)

    def rejected(self, source: str, message: str) -> None:
        for tool, options in (("atlas-opt", ("--verify-atlas-generated-schedule",)), ("atlas-emit", ()), ("atlas-opt", ("--convert-atlas-to-llvm",))):
            result = run(tool, source, *options)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertEqual(result.stdout, "")
            self.assertIn(message, result.stderr)

    def test_all_scalar_write_classes_can_reuse_captured_vmem_dram_and_size_registers(self) -> None:
        for direction, register, kind in product(("load", "store"), (4, 7, 9), ("alu_reg", "alu_imm", "upper")):
            fields = {'alu_reg': f'kind = "add", dst = {register} : i32, lhs = 0 : i32, rhs = 0 : i32',
                      'alu_imm': f'kind = "addi", dst = {register} : i32, src = 0 : i32, immediate = 1 : i32',
                      'upper': f'kind = "lui", dst = {register} : i32, immediate = 33 : i32'}[kind]
            with self.subTest(direction=direction, register=register, kind=kind):
                self.accepted(artifact(((kind, fields),), direction=direction))

    def test_pending_configuration_memory_and_control_restrictions_remain(self) -> None:
        forbidden = (("configuration", (("dma_config", "channel = 0 : i32, base_reg = 5 : i32"),)),
                     ("VMEM load", (("vload", 'dst = 0 : i32, base = 4 : i32, offset = 0 : i32, format = "raw"'), DELAY)),
                     ("VMEM store", (("vstore", 'src = 0 : i32, base = 4 : i32, offset = 0 : i32, format = "raw"'), DELAY)),
                     ("scalar memory", (("scalar_load", 'kind = "lw", dst = 4 : i32, base = 7 : i32, offset = 0 : i32'), ("delay", 'cycles = 8 : i32, atlas.delay_reason = "scalar_load_completion"'))),
                     ("branch", (("branch", 'kind = "beq", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = 2 : i32'), NOP)))
        for name, work in forbidden:
            with self.subTest(name=name):
                message = "pending" if name.startswith("VMEM") else "unexpected instruction while DMA is pending"
                self.rejected(artifact(work), message)

    def test_same_channel_cannot_be_relaunched_before_its_completion(self) -> None:
        self.rejected(artifact((dma(identity=1),)), "another DMA launch while DMA is pending")

    def test_completion_channel_and_identity_must_match_the_pending_transfer(self) -> None:
        for completion in (wait(channel=1), wait(identity=1), wait(identity=None)):
            with self.subTest(completion=completion):
                self.rejected(artifact(completion=completion), "DMA.WAIT must match")

    def test_capture_rule_does_not_relax_unmarked_immediate_wait_or_missing_completion(self) -> None:
        self.accepted(artifact(identity=None))
        self.rejected(artifact((NOP,), identity=None), "generated DMA requires immediate same-channel DMA.WAIT")
        missing = "\n".join(line for line in artifact(completion=NOP).splitlines() if '"atlas.trap"' not in line)
        self.rejected(missing, "generated DMA has no matching DMA.WAIT")


if __name__ == "__main__":
    unittest.main()

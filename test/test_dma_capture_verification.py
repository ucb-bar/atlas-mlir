"""Generated DMA schedules may reuse scalar operands captured at launch.

ScalarCore.scala:489-509 forms the command from S1 register values, and
diplomatic/memory/DMA.scala:223-229 queues the complete command. This checks
compiler admission, not physical completion or concurrent VMEM range safety.
"""

from __future__ import annotations

from itertools import product
import unittest

from generated_fixture import insert_after, remove_line
from test_virtual_dma import copy
from test_virtual_lowering import lower, virtual_chain
from test_virtual_ssa import BIN, run


STATE = "!atlas.state"
MARKER = "atlas.virtual_dma_transfer"
NOP = ("alu_imm", 'kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32')
DELAY = ("delay", 'cycles = 256 : i32, atlas.delay_reason = "tensor_completion"')
BOUNDARIES = (("atlas-opt", ("--verify-atlas-generated-schedule",)), ("atlas-emit", ()), ("atlas-opt", ("--convert-atlas-to-llvm",)))


def dma(direction: str = "load", channel: int = 0, identity: int | None = 0) -> tuple[str, str]:
    marker = f", {MARKER} = {identity} : i32" if identity is not None else ""
    return "dma", f'direction = "{direction}", channel = {channel} : i32, reg = 4 : i32, dram = 7 : i32, size = 9 : i32{marker}'


def wait(channel: int = 0, identity: int | None = 0) -> tuple[str, str]:
    marker = f", {MARKER} = {identity} : i32" if identity is not None else ""
    return "dma_wait", f"channel = {channel} : i32{marker}"


def lines_of(machine: str, name: str, *needles: str) -> list[int]:
    return [i for i, line in enumerate(machine.splitlines()) if f'"atlas.{name}"' in line and all(n in line for n in needles)]


def launch(machine: str, direction: str = "load") -> int:
    """Line of the source transfer's launch in lowered `copy()`: staging x4, DRAM x7, size x9."""
    return lines_of(machine, "dma", f'direction = "{direction}"', f"{MARKER} =")[0]


def replace_line(machine: str, index: int, old: str, new: str) -> str:
    lines = machine.splitlines()
    assert old in lines[index], (old, lines[index])
    lines[index] = lines[index].replace(old, new, 1)
    return "\n".join(lines) + "\n"


class DMACaptureVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue((BIN / "atlas-opt").is_file(), "build atlas-opt first")
        self.assertTrue((BIN / "atlas-emit").is_file(), "build atlas-emit first")
        self.machine = lower(copy())

    def accepted(self, source: str) -> None:
        for tool, options in BOUNDARIES:
            result = run(tool, source, *options)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout)

    def rejected(self, source: str, message: str) -> None:
        for tool, options in BOUNDARIES:
            result = run(tool, source, *options)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertEqual(result.stdout, "")
            self.assertIn(message, result.stderr)

    def test_all_scalar_write_classes_can_reuse_captured_vmem_dram_and_size_registers(self) -> None:
        self.accepted(self.machine)
        for direction, register, kind in product(("load", "store"), (4, 7, 9), ("alu_reg", "alu_imm", "upper")):
            fields = {'alu_reg': f'kind = "add", dst = {register} : i32, lhs = 0 : i32, rhs = 0 : i32',
                      'alu_imm': f'kind = "addi", dst = {register} : i32, src = 0 : i32, immediate = 1 : i32',
                      'upper': f'kind = "lui", dst = {register} : i32, immediate = 33 : i32'}[kind]
            with self.subTest(direction=direction, register=register, kind=kind):
                self.accepted(insert_after(self.machine, launch(self.machine, direction), [(kind, fields)]))

    def test_pending_configuration_memory_and_control_restrictions_remain(self) -> None:
        forbidden = (("configuration", (("dma_config", "channel = 0 : i32, base_reg = 5 : i32"),)),
                     ("VMEM load", (("vload", 'dst = 0 : i32, base = 4 : i32, offset = 0 : i32, format = "raw"'), DELAY)),
                     ("VMEM store", (("vstore", 'src = 0 : i32, base = 4 : i32, offset = 0 : i32, format = "raw"'), DELAY)),
                     ("scalar memory", (("scalar_load", 'kind = "lw", dst = 4 : i32, base = 7 : i32, offset = 0 : i32'), ("delay", 'cycles = 8 : i32, atlas.delay_reason = "scalar_load_completion"'))),
                     ("branch", (("branch", 'kind = "beq", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = 2 : i32'), NOP)))
        for name, work in forbidden:
            with self.subTest(name=name):
                message = "DMA memory conflict" if name.startswith("VMEM") else "unexpected instruction while DMA is pending"
                self.rejected(insert_after(self.machine, launch(self.machine), work), message)

    def test_same_channel_cannot_be_relaunched_before_its_completion(self) -> None:
        self.rejected(insert_after(self.machine, launch(self.machine), [dma(identity=2)]), "another DMA launch while its channel is pending")

    def test_completion_channel_and_identity_must_match_the_pending_transfer(self) -> None:
        index = launch(self.machine) + 1
        self.assertEqual(lines_of(self.machine, "dma_wait", f"{MARKER} = 0 : i32"), [index])
        for old, new, expected in (("channel = 0 : i32", "channel = 1 : i32", "DMA.WAIT has no pending DMA transfer"),
                                   (f"{MARKER} = 0 : i32", f"{MARKER} = 1 : i32", "DMA.WAIT must match"),
                                   (f", {MARKER} = 0 : i32", "", "DMA.WAIT must match")):
            with self.subTest(completion=new or "untagged"):
                self.rejected(replace_line(self.machine, index, old, new), expected)

    def test_untagged_boundary_transfers_use_the_same_pending_policy(self) -> None:
        # Implicit boundary transfers carry no transfer ID; their completion is the next
        # same-channel DMA.WAIT without an ID, not necessarily the next instruction.
        machine = lower(virtual_chain(1))
        launches, waits = lines_of(machine, "dma"), lines_of(machine, "dma_wait")
        self.assertTrue(launches and len(launches) == len(waits))
        self.assertNotIn(MARKER, machine)
        self.assertEqual([w - l for l, w in zip(launches, waits)], [1] * len(launches))
        for index in launches:
            with self.subTest(overlapped=index):
                self.accepted(insert_after(machine, index, [NOP]))
        self.rejected(remove_line(machine, waits[0]), "another DMA launch while its channel is pending")
        truncated = "\n".join(machine.splitlines()[:waits[-1]] + ["}"])
        self.rejected(truncated, "generated DMA has no matching DMA.WAIT")
        self.rejected(replace_line(machine, waits[0], "channel = 0 : i32", "channel = 1 : i32"), "DMA.WAIT has no pending DMA transfer")
        self.rejected(insert_after(machine, launches[0], [wait(channel=1, identity=None)]), "DMA.WAIT has no pending DMA transfer")
        self.rejected(replace_line(machine, waits[0], "atlas.virtual_cfg_block", f"{MARKER} = 0 : i32, atlas.virtual_cfg_block"), "DMA.WAIT must match")


if __name__ == "__main__":
    unittest.main()

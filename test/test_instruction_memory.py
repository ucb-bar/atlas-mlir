"""Final instruction capacity and conditional branch expansion after timing."""

import unittest
import os
import json
import tempfile
from itertools import product
from pathlib import Path

from test_virtual_spilling import run


def stream(fields):
    lines = ["module {", '%s0 = "atlas.start"() : () -> !atlas.state']
    for pc, (op, attrs) in enumerate(fields):
        lines.append(f'%s{pc + 1} = "atlas.{op}"(%s{pc}) {{{attrs}}} : (!atlas.state) -> !atlas.state')
    return "\n".join([*lines, "}"])


NOP = ("alu_imm", 'kind="addi", dst=0:i32, src=0:i32, immediate=0:i32')
HALT = ("trap", 'kind="ecall"')


class InstructionMemoryTest(unittest.TestCase):
    def test_long_branches_with_pending_delay_slots(self):
        # PR #33 leaves slots pending in untimed streams. Filling them can
        # move a target beyond the conditional-branch range; relaxation must
        # see the newly filled slots rather than the old pending marker.
        for backward in (False, True):
            branch = ("branch", f'kind="bne", lhs=2:i32, rhs=0:i32, offset_bytes={-4096 if backward else 4094}:i32')
            load = ("scalar_load", 'kind="lw", dst=1:i32, base=0:i32, offset=0:i32')
            use = ("alu_imm", 'kind="addi", dst=3:i32, src=1:i32, immediate=1:i32')
            fields = (
                [NOP, load, use] + [NOP] * 2045 + [branch, HALT]
                if backward else [branch, load, use] + [NOP] * 2044 + [HALT]
            )
            source = stream(fields).replace("module {", "module attributes {atlas.delay_slots_pending} {", 1)
            for timing in ("--insert-atlas-delays", "--schedule-atlas-stream"):
                with self.subTest(backward=backward, timing=timing):
                    result = run(source, timing, "--verify-atlas-timing", "--verify-atlas-machine-stream")
                    self.assertEqual(result.returncode, 0, result.stderr[:1600])
                    # Moving a filler can keep the scheduled backward branch
                    # in range; do not require an unnecessary trampoline.
                    if not backward or timing == '--insert-atlas-delays':
                        self.assertTrue('atlas.branch_trampoline' in result.stdout, 'missing long-branch expansion')
                    self.assertFalse('atlas.delay_slots_pending' in result.stdout)

    @unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "select the pinned model")
    def test_relaxed_branches_preserve_delay_slot_side_effects_on_model(self):
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from tools.check_program_decode import decode_binary
        from tools.model_instructions import model_instruction
        from test_virtual_ssa import ROOT

        torch.set_num_threads(1)
        inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())

        def add(dst, src, immediate):
            return ("alu_imm", f'kind="addi", dst={dst}:i32, src={src}:i32, immediate={immediate}:i32')

        with tempfile.TemporaryDirectory() as temporary:
            for pending, timing, (backward, count) in product(
                (False, True), ('--insert-atlas-delays', '--schedule-atlas-stream'),
                ((False, 0), (False, 1), (True, 1), (True, 2)),
            ):
                prefix = [add(2, 0, count), add(4, 0, 0), add(3, 0, 0)]
                branch = ("branch", f'kind="bne", lhs=2:i32, rhs=0:i32, offset_bytes={-4096 if backward else 4094}:i32')
                load = ("scalar_load", 'kind="lw", dst=1:i32, base=0:i32, offset=0:i32')
                body = (
                    [load, add(3, 1, 1)] + [NOP] * 2045 + [add(2, 2, -1), branch, add(4, 4, 1), HALT]
                    if backward
                    else [branch, add(4, 4, 1), load, add(3, 1, 1)] + [NOP] * 2043 + [HALT]
                )
                source = stream(prefix + body)
                if pending:
                    source = source.replace('module {', 'module attributes {atlas.delay_slots_pending} {', 1)
                machine = run(source, timing, "--verify-atlas-timing")
                self.assertEqual(machine.returncode, 0, machine.stderr[:1000])
                emitted = run(machine.stdout, tool="atlas-emit")
                self.assertEqual(emitted.returncode, 0, emitted.stderr[:1000])
                binary = b"".join(int(line, 16).to_bytes(4, "little") for line in emitted.stdout.splitlines())
                decoded = decode_binary(inventory, binary)
                program = InstantiableProgram(
                    [model_instruction(row, inventory, isa) for row in decoded["instructions"]]
                )
                simulation = Simulation(
                    DefaultHardwareConfig(),
                    LoggerConfig(filename=str(Path(temporary) / f"{pending}-{timing}-{backward}-{count}.json")),
                    program,
                    verbose=False,
                )
                state = simulation.core.arch_state
                try:
                    state.vmem[:4] = torch.tensor([42, 0, 0, 0], dtype=torch.uint8)
                    simulation.run(max_cycles=100000)
                    self.assertTrue(state.halted)
                    self.assertEqual(simulation.get_stats().runtime_errors, 0)
                    increments = (count if backward else 1) if not pending else (1 if backward or count == 0 else 0)
                    self.assertEqual(state.read_xrf(4), increments)
                    self.assertEqual(state.read_xrf(3), 43 if backward or count == 0 else 0)
                finally:
                    state.close()

    def test_capacity_includes_every_final_instruction(self):
        for count, accepted in ((2100, True), (32768, True), (32769, False)):
            with self.subTest(words=count):
                result = run(stream([NOP] * (count - 1) + [HALT]), "--verify-atlas-machine-stream")
                self.assertEqual(result.returncode == 0, accepted, result.stderr[-1000:])
                if not accepted:
                    self.assertIn("32768-word policy limit", result.stderr)
                    self.assertFalse(result.stdout)

    def test_timing_expansion_relaxes_forward_and_backward_branches(self):
        for backward in (False, True):
            with self.subTest(backward=backward):
                branch = ("branch", f'kind="bne", lhs=2:i32, rhs=0:i32, offset_bytes={-4096 if backward else 4094}:i32')
                load = ("scalar_load", 'kind="lw", dst=1:i32, base=0:i32, offset=0:i32')
                use = ("alu_imm", 'kind="addi", dst=3:i32, src=1:i32, immediate=1:i32')
                fields = (
                    [NOP, load, use] + [NOP] * 2045 + [branch, NOP, HALT]
                    if backward
                    else [branch, NOP, load, use] + [NOP] * 2043 + [HALT]
                )
                result = run(
                    stream(fields), "--insert-atlas-delays", "--verify-atlas-timing", "--verify-atlas-machine-stream"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('atlas.relaxed_branch = "bne"', result.stdout)
                self.assertIn("atlas.branch_trampoline", result.stdout)
                broken = result.stdout.replace('atlas.relaxed_branch = "bne"', 'atlas.relaxed_branch = "beq"')
                rejected = run(broken, "--verify-atlas-machine-stream")
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("malformed relaxed", rejected.stderr)

"""One resident image for explicitly ordered virtual regions, including K loops."""

import dataclasses
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_looped_mnk import source_request

ROOT = Path(__file__).resolve().parents[1]


def composed_source(factor, k=65, *, shape=None, entries=2):
    from atlas_mlir_native.looped_tiling import _virtual, prepare_plan

    plan = prepare_plan(source_request(*(shape or (1, 1, k))))
    wbase = max(0x90004000, 0x90000000 + plan.buffer_bytes[0])
    output_base = max(0x90010000, (wbase + plan.buffer_bytes[1] + 2047) // 2048 * 2048)
    functions = []
    names = ("first", "second", "third")[:entries]
    for index, name in enumerate(names):
        output = output_base + index * plan.buffer_bytes[2]
        source = _virtual(plan, (0x90000000, wbase), output, k_unroll=factor)
        functions.append("\n".join(source.splitlines()[1:-1]).replace("@looped_matmul", "@" + name))
    return f'module attributes {{atlas.program_entries = {json.dumps(names)}}} {{\n' + "\n".join(functions) + "\n}\n"


@unittest.skipUnless(
    os.environ.get("ATLAS_OOT_BIN_DIR") and os.environ.get("MERLIN_NATIVE_ROOT"),
    "select Atlas tools and Merlin",
)
class ProgramCompositionTest(unittest.TestCase):
    def compile(self, source, pipeline="optimized"):
        from atlas_mlir_native.binding import NATIVE_PIPELINES

        tools = Path(os.environ["ATLAS_OOT_BIN_DIR"])
        result = subprocess.run(
            [
                str(tools / "atlas-opt"),
                "--compose-atlas-program",
                *NATIVE_PIPELINES[pipeline],
                "-",
            ],
            input=source,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        emitted = subprocess.run(
            [str(tools / "atlas-emit"), "--program-json", "-"],
            input=result.stdout,
            text=True,
            capture_output=True,
        )
        self.assertEqual(emitted.returncode, 0, emitted.stderr)
        return json.loads(emitted.stdout)

    def test_composed_k_loops_have_one_start_and_one_halt(self):
        for factor in (1, 2, 4, 8):
            with self.subTest(factor=factor):
                program = self.compile(composed_source(factor))
                halts = [row for row in program["instructions"] if row["operation"] == "atlas.trap"]
                self.assertEqual(len(halts), 1)
                self.assertEqual(halts[0]["fields"]["kind"], "ecall")

    def test_composed_operand_caches_share_only_between_initialized_fragments(self):
        tools = Path(os.environ["ATLAS_OOT_BIN_DIR"])
        for shape in ((65, 35, 65), (65, 7, 65), (32, 64, 2592)):
            source = composed_source(4, shape=shape, entries=3)
            joined = subprocess.run([str(tools / "atlas-opt"), "--compose-atlas-program", "-"],
                                    input=source, text=True, capture_output=True)
            self.assertEqual(joined.returncode, 0, joined.stderr)
            slots = joined.stdout.count('"atlas.virtual_slot"')
            self.assertEqual(slots, (shape[2] + 31) // 32)
            for pipeline in ("mandatory", "optimized"):
                self.compile(source, pipeline=pipeline)
        source = composed_source(4, shape=(65, 35, 65))
        # A later fragment cannot read a previous fragment's private scratch
        # unless its own original slot was initialized on every reaching path.
        save = '"atlas.virtual_save"(%cache_0_loaded, %operand_cache_0, %cache_0_value)'
        index = source.rindex(save)
        source = source[:index] + source[index:].replace(save, save.replace("%operand_cache_0", "%operand_cache_1"), 1)
        invalid = subprocess.run([str(tools / "atlas-opt"), "--compose-atlas-program", "-"],
                                 input=source, text=True, capture_output=True)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("not initialized", invalid.stderr)

    def test_straight_line_entries_load_the_control_flow_dialect(self):
        functions = []
        for index, name in enumerate(("first", "second")):
            functions.append(f"""
  func.func @{name}() -> !atlas.virtual_state attributes {{
      atlas.input_dram_base = {0x90000000} : i64,
      atlas.output_dram_base = {0x90010000 + index * 2048} : i64,
      atlas.output_dram_bytes = 2048 : i64}} {{
    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %a = arith.constant {0x90000000 - (1 << 32)} : i32
    %o = arith.constant {0x90010000 + index * 2048 - (1 << 32)} : i32
    %n = arith.constant 2048 : i32
    %next, %v = "atlas.virtual_dma_load_bf16"(%s, %a, %n)
      : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %end = "atlas.virtual_dma_store_bf16"(%next, %v, %o, %n)
      : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    return %end : !atlas.virtual_state
  }}""")
        source = ('module attributes {atlas.program_entries = ["first", "second"]} {'
                  + "".join(functions) + "}")
        self.assertNotIn("cf.", source)
        program = self.compile(source, pipeline="mandatory")
        self.assertEqual(sum(row["operation"] == "atlas.trap" for row in program["instructions"]), 1)

    def test_missing_entries_bounds_and_disjoint_permissions_are_refused(self):
        tools = Path(os.environ["ATLAS_OOT_BIN_DIR"])
        source = composed_source(1)
        for changed in (
            source.replace('["first", "second"]', '["missing"]'),
            source.replace("atlas.output_dram_bytes", "missing_output_bytes"),
            source.replace(str(0x90010800), str(0x90012000)),
        ):
            with self.subTest(source=changed[:100]):
                result = subprocess.run(
                    [str(tools / "atlas-opt"), "--compose-atlas-program", "-"],
                    input=changed,
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)

    def test_composition_preserves_target_attributes_and_smallest_code_budget(self):
        source = composed_source(1)
        source = source.replace("atlas.input_dram_base", "atlas.test_resource = 7 : i64, atlas.input_dram_base")
        # These are ceilings on the composed image, not individual size estimates.
        budgets = iter((8192, 4096))
        source = "\n".join(
            line.replace("attributes {", f"attributes {{atlas.program_size_limit_bytes = {next(budgets)} : i64, ", 1)
            if "func.func" in line
            else line
            for line in source.splitlines()
        )
        tools = Path(os.environ["ATLAS_OOT_BIN_DIR"])
        result = subprocess.run(
            [str(tools / "atlas-opt"), "--compose-atlas-program", "-"], input=source, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("atlas.test_resource = 7 : i64", result.stdout)
        self.assertIn("atlas.program_size_limit_bytes = 4096 : i64", result.stdout)
        changed = source.replace("atlas.test_resource = 7", "atlas.test_resource = 8", 1)
        invalid = subprocess.run(
            [str(tools / "atlas-opt"), "--compose-atlas-program", "-"], input=changed, text=True, capture_output=True
        )
        self.assertNotEqual(invalid.returncode, 0)

    @unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "enable the pinned model")
    def test_whole_program_unroll_variants_execute_in_one_launch(self):
        import torch
        from atlas_mlir_native.looped_tiling import prepare_plan
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from tools.check_program_decode import decode_binary
        from tools.model_instructions import model_instruction

        from merlin.semantic_compiler.ordered_mnk_tiling import (
            materialize_looped_mnk_inputs,
        )

        torch.set_num_threads(1)
        inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
        plan = prepare_plan(source_request(1, 1, 257))
        rows = []
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            for factor in (1, 2, 4, 8):
                program = self.compile(composed_source(factor, 257), pipeline="mandatory")
                binary = b"".join(struct.pack("<I", row["word_u32"]) for row in program["instructions"])
                instructions = [
                    model_instruction(row, inventory, isa) for row in decode_binary(inventory, binary)["instructions"]
                ]
                cycles = []
                for iteration in range(6):
                    variant = iteration % 2
                    a = bytes((0x38 if variant == 0 else 0x40) if i in (0, 256) else 0 for i in range(257))
                    w = bytes((0x38 if variant == 0 else 0xB8) if i in (0, 256) else 0 for i in range(257))
                    packed = materialize_looped_mnk_inputs(plan, {"a": a, "w": w})
                    base = 0x90000000
                    initial = bytearray((i * 13 + 31) % 256 for i in range(0x14000))
                    for offset, payload in zip((0, 0x4000), packed, strict=True):
                        initial[offset : offset + len(payload)] = payload
                    instance = InstantiableProgram(instructions)
                    instance.dram_base = base
                    instance.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
                    config = DefaultHardwareConfig()
                    config.arch_state_config = dataclasses.replace(
                        config.arch_state_config,
                        numerics="rtl",
                        randomize_init=True,
                        init_seed=iteration + 73,
                    )
                    simulation = Simulation(
                        config,
                        LoggerConfig(filename=str(Path(directory) / f"{factor}-{iteration}.log")),
                        instance,
                        verbose=False,
                    )
                    state = simulation.core.arch_state
                    try:
                        simulation.run(max_cycles=500000)
                        self.assertTrue(state.halted)
                        self.assertEqual(simulation.get_stats().runtime_errors, 0)
                        expected = bytearray(2048)
                        struct.pack_into("<H", expected, 0, 0x4000 if variant == 0 else 0xC080)
                        for offset in (0x10000, 0x10800):
                            initial[offset : offset + 2048] = expected
                        self.assertEqual(
                            bytes(state.read_dram(base, len(initial)).tolist()),
                            bytes(initial),
                        )
                        if iteration:
                            cycles.append(simulation.get_stats().cycles)
                    finally:
                        state.close()
                rows.append(
                    {
                        "factor": factor,
                        "bytes": len(binary),
                        "spills": program["spill_stores"],
                        "reloads": program["spill_reloads"],
                        "device_cycles": cycles,
                        "outputs": 2,
                        "launches_per_invocation": 1,
                        "correctness": "passed",
                    }
                )
        if os.environ.get("ATLAS_COMPOSITION_PROFILE_OUT"):
            destination = Path(os.environ["ATLAS_COMPOSITION_PROFILE_OUT"])
            self.assertFalse(destination.exists())
            destination.write_text(
                json.dumps(
                    {
                        "scope": "two independent (1,257,1) contractions in one resident image",
                        "variants": rows,
                    },
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    unittest.main()

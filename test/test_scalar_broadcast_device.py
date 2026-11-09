"""Immutable scalar gather uses one checked device loop with exact raw bits."""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from atlas_mlir_native.binding import DRAM_BASE, TARGET_IDENTITY, TILE_BYTES
from atlas_mlir_native.scalar_broadcast import compile_source, prepare_c_package, prepare_source, virtual_program
from merlin.semantic_compiler.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))


def request(raw=b"\x00\x00\x80\x3f", shape=(3, 257), dtype="f32", encoding="f32-le-raw"):
    scalar = SemanticNode("scalar", "constant", (), TensorType((), dtype, "raw-bit-pattern"), effect="constant")
    output = SemanticNode("broadcast", "gather", (scalar.id,), replace(scalar.type, shape=shape),
                          index_maps=(IndexMap(len(shape), (), ()), pointwise_identity_maps(len(shape), 1)[0]))
    return KernelRequest((scalar, output), (output.id,), ("external",), (), TARGET_IDENTITY, "diagnostic",
                         constants=(ConstantBinding(scalar.id, encoding, raw.hex()),))


class ScalarBroadcastRecognitionTest(unittest.TestCase):
    def test_application_selects_device_broadcast_before_host_layout(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.application_precision import SCALED_CONTRACTION_POLICY
        from atlas_mlir_native.pointwise import POINTWISE_POLICY
        from atlas_mlir_native.row_softmax import SOFTMAX_POLICY

        policy = dict(schema="atlas.diagnostic_application_precision.v4", source_sha256="0" * 64,
                      target_identity=TARGET_IDENTITY,
                      contraction=dict(policy=SCALED_CONTRACTION_POLICY, operand_scale_exponents=[4, 4]),
                      row_softmax=SOFTMAX_POLICY, host_support=["transpose", "reshape", "copy"],
                      epilogues=["relu"], pointwise=POINTWISE_POLICY)
        selected = AtlasApplicationAdapter(policy, original_source_sha256="0" * 64).plan(request()).selected
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].placement, "accelerator")
        self.assertEqual(dict(selected[0].metadata)["kind"], "scalar_broadcast")

    def test_bits_and_private_tail(self):
        for raw, dtype, encoding in ((b"\x34\x12\xc0\x7f", "f32", "f32-le-raw"),
                                     (b"\x00\x80", "bf16", "bf16-le-raw"),
                                     (b"\xff", "i8", "i8"),
                                     (b"\x01\x00\x00\x00\x00\x00\xf0\x7f", "f64", "f64-le-raw")):
            with self.subTest(dtype=dtype):
                plan = prepare_source(request(raw, dtype=dtype, encoding=encoding))
                self.assertEqual(plan.tile, raw * (1024 // len(raw)))
                self.assertGreaterEqual(plan.physical_bytes, plan.output_bytes)
                self.assertLess(plan.physical_bytes - plan.output_bytes, 1024)

    def test_refuses_dynamic_source_and_changed_maps_or_type(self):
        source = request()
        scalar, output = source.nodes
        changed = (replace(source, nodes=(replace(scalar, op="input", effect="input"), output),
                           input_storages=((scalar.id, "external"),), constants=()),
                   replace(source, nodes=(scalar, replace(output, attrs=(("extra", 1),)))),
                   replace(source, nodes=(scalar, replace(output, index_maps=(IndexMap(2, (), ()), IndexMap(2, ((0, 1), (1, 0)), (0, 0)))))),
                   replace(source, nodes=(scalar, replace(output, type=replace(output.type, dtype="bf16")))))
        for region in changed:
            with self.assertRaises(ValueError):
                prepare_source(region)

    def test_selected_window_is_enforced(self):
        with self.assertRaisesRegex(RuntimeError, "I/O window"):
            virtual_program(prepare_source(request(shape=(131072,))))


@unittest.skipUnless((BIN / "atlas-opt").is_file(), "select built Atlas tools")
class ScalarBroadcastCompilationTest(unittest.TestCase):
    def test_checked_loop_c_export_and_publication_replay(self):
        source = request()
        with tempfile.TemporaryDirectory(prefix="atlas-scalar-broadcast-") as directory:
            compiled = Path(directory) / "compiled"
            compile_source(None, source, target_source=ROOT, destination=compiled, limits=None)
            machine = (compiled / "atlas-machine.mlir").read_text()
            self.assertIn("atlas.generated", machine)
            self.assertIn("atlas.virtual_cfg_contract", machine)
            self.assertIn("atlas.virtual_buffer_contract", machine)
            self.assertIn('"atlas.branch"', machine)
            record, files = prepare_c_package(source, compiled, "broadcast", target_source=ROOT)
            self.assertEqual(record["input_node_ids"], [])
            self.assertEqual(record["launches_per_invocation"], 1)
            package = Path(directory) / "c"
            package.mkdir()
            for name, payload in files.items():
                path = package / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            if shutil.which("cc"):
                subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
                                str(package / "broadcast.c"), "-I", str(package)], check=True, capture_output=True)
            # Even a self-consistently rehashed native manifest cannot replace the source loop.
            virtual = compiled / "atlas-virtual.mlir"
            virtual.write_text(virtual.read_text().replace("%count = arith.constant 4", "%count = arith.constant 3"))
            manifest = json.loads((compiled / "manifest.json").read_text())
            import hashlib
            manifest["stage_artifacts"][virtual.name] = hashlib.sha256(virtual.read_bytes()).hexdigest()
            (compiled / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "virtual source differs"):
                prepare_c_package(source, compiled, "broadcast", target_source=ROOT)

    def test_pin_shape_compiles_with_constant_image_size(self):
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options

        sizes = []
        with codegen_options(CodegenOptions(io_memory_bytes=16 * 1024 * 1024)):
            for shape in ((3, 257), (1, 64, 5632)):
                with tempfile.TemporaryDirectory(prefix="atlas-broadcast-shape-") as directory:
                    compiled = Path(directory) / "compiled"
                    compile_source(None, request(shape=shape), target_source=ROOT, destination=compiled, limits=None)
                    sizes.append(len((compiled / "program.bin").read_bytes()) // 4)
        self.assertEqual(sizes, [42, 42])

    @unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "select unchanged npu_model")
    def test_raw_nan_payload_signed_zero_and_tail_on_unchanged_model(self):
        import dataclasses
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from tools.check_program_decode import decode_binary
        from tools.model_instructions import model_instruction

        # Every simulation uses the unchanged selected model and its default cycle bound.
        import npu_model
        model_root = Path(npu_model.__file__).resolve().parents[1]
        self.assertEqual(subprocess.check_output(["git", "-C", str(model_root), "rev-parse", "HEAD"], text=True).strip(),
                         "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5")
        torch.set_num_threads(1)
        inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
        with tempfile.TemporaryDirectory(prefix="atlas-broadcast-model-") as directory:
            for raw, dtype, encoding in ((b"\x34\x12\xc0\x7f", "f32", "f32-le-raw"),
                                         (b"\x00\x80", "bf16", "bf16-le-raw"),
                                         (b"\xfe", "i8", "i8")):
                with self.subTest(dtype=dtype):
                    source = request(raw, shape=(1031,), dtype=dtype, encoding=encoding)
                    plan = prepare_source(source)
                    compiled = Path(directory) / dtype
                    compile_source(None, source, target_source=ROOT, destination=compiled, limits=None)
                    decoded = decode_binary(inventory, (compiled / "program.bin").read_bytes())
                    program = InstantiableProgram([model_instruction(row, inventory, isa) for row in decoded["instructions"]])
                    program.dram_base = DRAM_BASE
                    initial = bytearray((i * 17 + 23) % 256 for i in range(TILE_BYTES + plan.physical_bytes + 64))
                    initial[:1024] = plan.tile
                    program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
                    config = DefaultHardwareConfig()
                    config.arch_state_config = dataclasses.replace(config.arch_state_config, numerics="rtl", randomize_init=True, init_seed=173)
                    simulation = Simulation(config, LoggerConfig(filename=str(Path(directory) / (dtype + ".json"))), program, verbose=False)
                    state = simulation.core.arch_state
                    try:
                        state.vmem.fill_(0xcd)
                        simulation.run()
                        self.assertTrue(state.halted)
                        self.assertEqual(simulation.get_stats().runtime_errors, 0)
                        expected = bytearray(initial)
                        expected[TILE_BYTES:TILE_BYTES + plan.physical_bytes] = plan.tile * plan.tiles
                        actual = bytes(state.read_dram(DRAM_BASE, len(initial)).tolist())
                        self.assertEqual(actual, bytes(expected))
                        self.assertEqual(actual[TILE_BYTES:TILE_BYTES + plan.output_bytes], raw * 1031)
                    finally:
                        state.close()


if __name__ == "__main__":
    unittest.main()
